import asyncio
import logging
import os
from pathlib import Path
from typing import List

from telegram import ChatMember, InlineKeyboardButton, InlineKeyboardMarkup, Update, Message, WebAppInfo
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    MessageHandler,
    filters,
    ContextTypes,
)

from src.config import get_config
from src.downloader import download_audio, extract_video_id, fetch_transcript, fetch_video_title, TranscriptRateLimited
from src.fact_checker import extract_claims, classify_claims, build_context_prompt
from src.memory import MemoryManager
from src.transcriber import transcribe_audio, TranscriptionError
from src.summarizer import summarize_text, SummarizationError
from src.tts import generate_voice_chunked, get_voice_for_language, detect_language_from_text, TTSError

logger = logging.getLogger(__name__)

WEBAPP_URL = os.environ.get("WEBAPP_URL", "https://localhost:8000")

TELEGRAM_MAX_MESSAGE_LENGTH = 4096
PROGRESS_BAR_WIDTH = 15


def _escape_html(text: str) -> str:
    """Escape HTML special characters for Telegram HTML parse mode."""
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _progress_bar(percent: float, label: str) -> str:
    filled = round(PROGRESS_BAR_WIDTH * percent / 100)
    empty = PROGRESS_BAR_WIDTH - filled
    bar = "\u2593" * filled + "\u2591" * empty
    return f"{bar} {int(percent)}%\n{label}"


def split_message(text: str, max_length: int = TELEGRAM_MAX_MESSAGE_LENGTH) -> List[str]:
    """Split a long message into Telegram-safe chunks at paragraph boundaries."""
    if len(text) <= max_length:
        return [text]

    chunks = []
    paragraphs = text.split("\n\n")
    current = ""

    for para in paragraphs:
        if len(current) + len(para) + 2 <= max_length:
            current = current + "\n\n" + para if current else para
        else:
            if current:
                chunks.append(current)
            if len(para) > max_length:
                while len(para) > max_length:
                    cut = para[:max_length].rfind(". ")
                    if cut == -1:
                        cut = max_length
                    else:
                        cut += 1
                    chunks.append(para[:cut])
                    para = para[cut:].strip()
                if para:
                    current = para
                else:
                    current = ""
            else:
                current = para

    if current:
        chunks.append(current)

    return chunks


class ProgressTracker:
    """Edits a single Telegram message to show a detailed progress bar."""

    # Pipeline weight allocation (must sum to 100):
    #   Transcript:  10%
    #   Download:    15%  (only if fallback)
    #   Transcribe:  15%  (only if fallback)
    #   Summarize:   25%
    #   Voice:       45%
    #   Done:       100%

    def __init__(self, status_msg: Message):
        self._msg = status_msg
        self._last_text = ""

    async def update(self, percent: float, label: str):
        text = _progress_bar(min(percent, 100), label)
        if text != self._last_text:
            self._last_text = text
            try:
                await self._msg.edit_text(text)
            except Exception:
                pass  # Telegram rate limit or message unchanged


async def _check_access(update: Update, context: ContextTypes.DEFAULT_TYPE) -> bool:
    config = get_config()
    user_id = update.effective_user.id
    logger.info(f"Request from user {user_id} ({update.effective_user.first_name})")

    # Whitelist check — pass if in list or list is empty
    if config.is_user_allowed(user_id):
        return True

    # Channel membership check
    if config.required_channel:
        try:
            member = await context.bot.get_chat_member(
                chat_id=config.required_channel, user_id=user_id,
            )
            if member.status not in (ChatMember.LEFT, ChatMember.BANNED):
                return True
        except Exception:
            pass

    # Neither condition met
    if config.required_channel:
        await update.message.reply_text(
            f"Please join {config.required_channel} to use this bot."
        )
    else:
        await update.message.reply_text("Sorry, you are not authorized to use this bot.")
    logger.warning(f"Unauthorized access attempt by user {user_id}")
    return False


async def id_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Send the user their Telegram ID so they can add themselves to the whitelist."""
    user_id = update.effective_user.id
    await update.message.reply_text(f"Your Telegram user ID: {user_id}")


async def dashboard_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Send link to the web dashboard."""
    if not await _check_access(update, context):
        return
    await update.message.reply_text(
        "Open your dashboard to view claims, knowledge graph, and export vault:",
        reply_markup=InlineKeyboardMarkup([[
            InlineKeyboardButton("Open Dashboard", web_app=WebAppInfo(url=f"{WEBAPP_URL}/app")),
        ]]),
    )


async def settings_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Show current settings and allow configuration."""
    if not await _check_access(update, context):
        return

    memory_mgr = context.bot_data.get("memory_mgr")
    user_mem_id = f"tg_{update.effective_user.id}"

    lines = ["*Settings*\n"]

    if memory_mgr:
        settings = await memory_mgr.get_user_settings(user_mem_id)
        has_ant = bool(settings.get("anthropic_api_key"))
        has_el = bool(settings.get("elevenlabs_api_key"))
        lines.append(f"Anthropic API key: {'configured' if has_ant else 'not set'}")
        lines.append(f"ElevenLabs API key: {'configured' if has_el else 'not set'}")
        if not has_ant or not has_el:
            lines.append("\n*You must set both API keys to use the bot.*")
    else:
        lines.append("Memory system disabled — settings unavailable.")

    lines.append("\nUse /dashboard to configure API keys.")

    await update.message.reply_text("\n".join(lines), parse_mode="Markdown")


async def setkey_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Set per-user API key. Detects type from prefix."""
    if not await _check_access(update, context):
        return

    memory_mgr = context.bot_data.get("memory_mgr")
    if not memory_mgr:
        await update.message.reply_text("Memory system is disabled. Cannot store settings.")
        return

    args = context.args
    if not args:
        await update.message.reply_text(
            "Usage:\n"
            "`/setkey sk-ant-...` — Anthropic key\n"
            "`/setkey sk_...` — ElevenLabs key",
            parse_mode="Markdown",
        )
        return

    key = args[0]
    user_mem_id = f"tg_{update.effective_user.id}"

    if key.startswith("sk-ant-"):
        await memory_mgr.save_user_settings(user_mem_id, {"anthropic_api_key": key})
        key_type = "Anthropic"
    elif key.startswith("sk_"):
        await memory_mgr.save_user_settings(user_mem_id, {"elevenlabs_api_key": key})
        key_type = "ElevenLabs"
    else:
        await update.message.reply_text(
            "Unrecognized key format.\n"
            "Anthropic keys start with `sk-ant-`, ElevenLabs with `sk_`."
        )
        return

    try:
        await update.message.delete()
    except Exception:
        pass

    await context.bot.send_message(
        chat_id=update.effective_chat.id,
        text=f"{key_type} API key saved. Your message was deleted for security.",
    )


async def removekey_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Remove per-user API keys."""
    if not await _check_access(update, context):
        return

    memory_mgr = context.bot_data.get("memory_mgr")
    if not memory_mgr:
        await update.message.reply_text("Memory system is disabled.")
        return

    user_mem_id = f"tg_{update.effective_user.id}"
    await memory_mgr.save_user_settings(
        user_mem_id, {"anthropic_api_key": "", "elevenlabs_api_key": ""}
    )
    await update.message.reply_text(
        "All API keys have been removed. "
        "You'll need to set new ones to continue using the bot."
    )


async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not await _check_access(update, context):
        return
    await update.message.reply_text(
        "Welcome! Send me a YouTube video link and I'll create a summary for you.\n"
        "You'll receive both a text summary and a voice message.\n\n"
        "Commands:\n"
        "/dashboard — View your knowledge base\n"
        "/settings — View and configure settings\n"
        "/setkey — Set your own Anthropic API key\n"
        "/removekey — Remove your API key"
    )


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not await _check_access(update, context):
        return
    url = update.message.text
    config = get_config()
    audio_path = None
    voice_paths: List[Path] = []

    # Resolve per-user API keys
    memory_mgr = context.bot_data.get("memory_mgr")
    user_mem_id = f"tg_{update.effective_user.id}"
    api_key = None
    elevenlabs_key = None
    if memory_mgr:
        try:
            user_settings = await memory_mgr.get_user_settings(user_mem_id)
            api_key = user_settings.get("anthropic_api_key", "") or None
            elevenlabs_key = user_settings.get("elevenlabs_api_key", "") or None
        except Exception:
            pass

    if not api_key:
        await update.message.reply_text(
            "You need to set your Anthropic API key before using the bot.\n\n"
            "Tap the button below to configure.",
            reply_markup=InlineKeyboardMarkup([[
                InlineKeyboardButton("Configure API Keys", web_app=WebAppInfo(url=f"{WEBAPP_URL}/app")),
            ]]),
        )
        return

    try:
        video_id = extract_video_id(url)

        # Check if this video was already processed for this user
        if memory_mgr:
            existing = await memory_mgr.get_transcript(video_id=video_id, user_id=user_mem_id)
            if existing:
                title = existing.get("title") or video_id
                await update.message.reply_text(
                    f"This video was already processed:\n\n"
                    f"<b>{_escape_html(title)}</b>\n\n"
                    f"Check your knowledge in the dashboard.",
                    parse_mode="HTML",
                    reply_markup=InlineKeyboardMarkup([[
                        InlineKeyboardButton("Open Dashboard", web_app=WebAppInfo(url=f"{WEBAPP_URL}/app")),
                    ]]),
                )
                return

        status_msg = await update.message.reply_text(
            _progress_bar(0, "Starting...")
        )
        progress = ProgressTracker(status_msg)

        # Step 1: Transcript (0% -> 10%)
        await progress.update(2, "Fetching transcript...")
        try:
            transcript_result = await asyncio.to_thread(fetch_transcript, video_id)
        except TranscriptRateLimited:
            await status_msg.edit_text(
                "YouTube is temporarily rate-limiting this server.\n"
                "The video likely has captions — please try again in 2-3 minutes."
            )
            return

        has_captions = False
        if transcript_result:
            transcript_text = transcript_result.text
            language_code = transcript_result.language_code
            has_captions = True
            logger.info(f"Got transcript from YouTube captions ({language_code})")
            await progress.update(10, "Transcript fetched")
        else:
            # Fallback path: download + ElevenLabs (0% -> 10% -> 25%)
            if not elevenlabs_key:
                await update.message.reply_text(
                    "This video has no captions. Transcription requires an ElevenLabs API key.\n\n"
                    "Tap below to add it, then try again.",
                    reply_markup=InlineKeyboardMarkup([[
                        InlineKeyboardButton("Configure API Keys", web_app=WebAppInfo(url=f"{WEBAPP_URL}/app")),
                    ]]),
                )
                return
            await progress.update(5, "No captions found. Downloading audio...")
            logger.info("No captions available, downloading audio for transcription")
            audio_path = await asyncio.to_thread(download_audio, url)
            await progress.update(15, "Audio downloaded. Transcribing...")

            transcription = await asyncio.to_thread(
                transcribe_audio, audio_path, elevenlabs_key
            )
            transcript_text = transcription.text
            language_code = transcription.language_code
            await progress.update(25, "Transcription complete")

        # Step 2: Memory — fact check & retrieve context (10% -> 20%)
        past_context = None
        fact_check_result = None

        # Fetch video title for display
        video_title = fetch_video_title(video_id)

        # Store transcript in separate collection
        if config.memory_enabled and memory_mgr:
            try:
                await memory_mgr.store_transcript(
                    video_id=video_id, user_id=user_mem_id,
                    transcript=transcript_text, language_code=language_code,
                    title=video_title,
                )
            except Exception as e:
                logger.warning(f"Failed to store transcript: {e}")

        if config.memory_enabled and memory_mgr:
            try:
                await progress.update(12, "Analyzing topics...")
                logger.info(f"[MEMORY] Extracting claims for user {user_mem_id}")
                claims = await extract_claims(
                    transcript_text, api_key, config.claude_model,
                    video_id=video_id,
                )
                logger.info(f"[MEMORY] Extracted {len(claims)} claims")
                for c in claims[:5]:
                    logger.info(f"[MEMORY]   claim: {c.text[:80]}")

                await progress.update(15, "Searching your memory...")
                memories = await memory_mgr.search(
                    transcript_text[:500], user_id=user_mem_id
                )
                logger.info(f"[MEMORY] Found {len(memories)} past memories")
                for m in memories[:5]:
                    logger.info(f"[MEMORY]   memory: {m.text[:80]}")

                await progress.update(18, "Checking what's new...")
                fact_check_result = await classify_claims(
                    claims, memories, api_key, config.claude_model
                )
                logger.info(
                    f"[MEMORY] Classification: {len(fact_check_result.new_claims)} new, "
                    f"{len(fact_check_result.supported_claims)} supported, "
                    f"{len(fact_check_result.contradicted_claims)} contradicted"
                )
                past_context = build_context_prompt(fact_check_result)
                if past_context:
                    logger.info(f"[MEMORY] Context prompt ({len(past_context)} chars):\n{past_context[:500]}")
                else:
                    logger.info("[MEMORY] No context to inject (all new or empty)")
                await progress.update(20, "Context ready")
            except Exception as e:
                logger.warning(f"Memory system error, proceeding without context: {e}", exc_info=True)
                past_context = None
                fact_check_result = None

        # Step 3: Summarize (20% -> 50%)
        await progress.update(22, "Generating summary with Claude...")
        summary = await summarize_text(
            transcript_text,
            api_key,
            model=config.claude_model,
            max_tokens=config.max_tokens,
            past_context=past_context,
        )
        await progress.update(50, "Summary generated")

        # Step 3: Generate voice (50% -> 95%)
        await progress.update(52, "Creating voice message...")
        try:
            # Use caption language if available, otherwise detect from summary text
            if has_captions:
                tts_lang = language_code
            else:
                tts_lang = detect_language_from_text(summary)
                logger.info(f"TTS: detected language '{tts_lang}' from summary text (no captions)")
            voice = get_voice_for_language(tts_lang)

            async def on_voice_progress(fraction: float, _total: float):
                pct = 52 + int(43 * fraction)
                await progress.update(pct, f"Generating audio... {int(fraction * 100)}%")

            voice_paths = await generate_voice_chunked(
                summary, "/tmp", video_id, voice=voice,
                on_progress=on_voice_progress,
            )
        except TTSError as e:
            logger.warning(f"TTS failed, skipping voice: {e}")

        # Step 4: Store new facts in memory (95% -> 97%)
        if config.memory_enabled and fact_check_result and memory_mgr:
            try:
                await progress.update(96, "Saving to memory...")

                # Deduplicate claims within this batch to prevent races
                seen_texts = set()
                unique_claims = []
                for cc in fact_check_result.new_claims:
                    if cc.claim.text not in seen_texts:
                        seen_texts.add(cc.claim.text)
                        unique_claims.append(cc)

                sem = asyncio.Semaphore(5)

                async def _store_claim(cc):
                    async with sem:
                        claim = cc.claim
                        meta = {
                            "entity": claim.entity,
                            "relation": claim.relation,
                            "value": claim.value,
                            "confidence": claim.confidence,
                            "video_id": claim.video_id,
                            "timestamp": claim.timestamp,
                        }
                        return await memory_mgr.add_if_new(
                            claim.text, user_id=user_mem_id, metadata=meta
                        )

                results = await asyncio.gather(
                    *[_store_claim(cc) for cc in unique_claims],
                    return_exceptions=True,
                )
                stored = sum(1 for r in results if r is True)
                skipped = sum(1 for r in results if r is False)
                errors = sum(1 for r in results if isinstance(r, Exception))
                logger.info(
                    f"[MEMORY] Stored {stored} new, skipped {skipped} duplicates"
                    f"{f', {errors} errors' if errors else ''} for {user_mem_id}"
                )
            except Exception as e:
                logger.warning(f"Failed to store memories: {e}", exc_info=True)

        # Step 5: Rebuild vault (97% -> 99%)
        if config.memory_enabled and fact_check_result and memory_mgr and stored > 0:
            try:
                await progress.update(97, "Updating knowledge vault...")
                from webapp.build_vault import rebuild_vault
                await asyncio.to_thread(rebuild_vault, user_mem_id)
            except Exception as e:
                logger.warning(f"Vault rebuild failed (non-critical): {e}")

        # Step 6: Done
        await progress.update(100, "Done!")

        # Send voice with caption (title + link)
        video_url = f"https://youtube.com/watch?v={video_id}"
        caption_title = video_title or video_id
        caption = f'<b>{_escape_html(caption_title)}</b>\n{video_url}'
        for vp in voice_paths:
            with open(vp, "rb") as voice_file:
                await update.message.reply_voice(
                    voice=voice_file, caption=caption, parse_mode="HTML",
                )

        # Send summary in expandable blockquote
        summary_parts = split_message(summary)
        for part in summary_parts:
            html = f'<blockquote expandable>{_escape_html(part)}</blockquote>'
            await update.message.reply_text(html, parse_mode="HTML")

    except (ValueError, TranscriptionError, SummarizationError) as e:
        error_messages = {
            ValueError: "Sorry, that doesn't look like a valid YouTube link.",
            TranscriptionError: "Sorry, I couldn't transcribe the audio from this video.",
            SummarizationError: "Sorry, I couldn't generate a summary for this video.",
        }
        msg = error_messages.get(type(e), f"Sorry, an error occurred: {e}")
        await update.message.reply_text(msg)
        logger.error(f"Error processing {url}: {e}", exc_info=True)

    except Exception as e:
        await update.message.reply_text(f"Sorry, an unexpected error occurred: {e}")
        logger.error(f"Unexpected error processing {url}: {e}", exc_info=True)

    finally:
        if audio_path and Path(audio_path).exists():
            try:
                Path(audio_path).unlink()
            except OSError:
                pass
        for vp in voice_paths:
            for ext_path in [vp, vp.with_suffix(".mp3")]:
                if ext_path.exists():
                    try:
                        ext_path.unlink()
                    except OSError:
                        pass


def create_app():
    config = get_config()
    app = ApplicationBuilder().token(config.telegram_bot_token).build()

    if config.memory_enabled:
        try:
            mgr = MemoryManager(config)
            app.bot_data["memory_mgr"] = mgr
            logger.info("Memory system initialized (singleton)")
        except Exception as e:
            logger.warning(f"Failed to init memory system: {e}")

    app.add_handler(CommandHandler("start", start_command))
    app.add_handler(CommandHandler("id", id_command))
    app.add_handler(CommandHandler("dashboard", dashboard_command))
    app.add_handler(CommandHandler("settings", settings_command))
    app.add_handler(CommandHandler("setkey", setkey_command))
    app.add_handler(CommandHandler("removekey", removekey_command))
    app.add_handler(
        MessageHandler(
            filters.TEXT
            & ~filters.COMMAND
            & filters.Regex(r"youtube\.com|youtu\.be"),
            handle_message,
        )
    )
    return app
