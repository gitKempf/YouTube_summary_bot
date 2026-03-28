import asyncio
import logging
from pathlib import Path
from typing import List

from telegram import ChatMember, Update, Message
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    MessageHandler,
    filters,
    ContextTypes,
)

from src.config import get_config
from src.downloader import download_audio, extract_video_id, fetch_transcript
from src.transcriber import transcribe_audio, TranscriptionError
from src.summarizer import summarize_text, SummarizationError
from src.tts import generate_voice_chunked, get_voice_for_language, TTSError

logger = logging.getLogger(__name__)

TELEGRAM_MAX_MESSAGE_LENGTH = 4096
PROGRESS_BAR_WIDTH = 15


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


async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not await _check_access(update, context):
        return
    await update.message.reply_text(
        "Welcome! Send me a YouTube video link and I'll create a summary for you.\n"
        "You'll receive both a text summary and a voice message."
    )


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not await _check_access(update, context):
        return
    url = update.message.text
    config = get_config()
    audio_path = None
    voice_paths: List[Path] = []

    try:
        video_id = extract_video_id(url)

        status_msg = await update.message.reply_text(
            _progress_bar(0, "Starting...")
        )
        progress = ProgressTracker(status_msg)

        # Step 1: Transcript (0% -> 10%)
        await progress.update(2, "Fetching transcript...")
        transcript_result = await asyncio.to_thread(fetch_transcript, video_id)

        if transcript_result:
            transcript_text = transcript_result.text
            language_code = transcript_result.language_code
            logger.info(f"Got transcript from YouTube captions ({language_code})")
            await progress.update(10, "Transcript fetched")
        else:
            # Fallback path: download + ElevenLabs (0% -> 10% -> 25%)
            await progress.update(5, "No captions found. Downloading audio...")
            logger.info("No captions available, downloading audio for transcription")
            audio_path = await asyncio.to_thread(download_audio, url)
            await progress.update(15, "Audio downloaded. Transcribing...")

            transcription = await asyncio.to_thread(
                transcribe_audio, audio_path, config.elevenlabs_api_key
            )
            transcript_text = transcription.text
            language_code = transcription.language_code
            await progress.update(25, "Transcription complete")

        # Step 2: Summarize (-> 50%)
        await progress.update(28, "Generating summary with Claude...")
        summary = await summarize_text(
            transcript_text,
            config.anthropic_api_key,
            model=config.claude_model,
            max_tokens=config.max_tokens,
        )
        await progress.update(50, "Summary generated")

        # Step 3: Generate voice (50% -> 95%)
        await progress.update(52, "Creating voice message...")
        try:
            voice = get_voice_for_language(language_code)

            async def on_voice_progress(fraction: float, _total: float):
                pct = 52 + int(43 * fraction)
                await progress.update(pct, f"Generating audio... {int(fraction * 100)}%")

            voice_paths = await generate_voice_chunked(
                summary, "/tmp", video_id, voice=voice,
                on_progress=on_voice_progress,
            )
        except TTSError as e:
            logger.warning(f"TTS failed, skipping voice: {e}")

        # Step 4: Done
        await progress.update(100, "Done!")

        # Send text
        message_parts = split_message(summary)
        for part in message_parts:
            await update.message.reply_text(part)

        # Send voice
        for vp in voice_paths:
            with open(vp, "rb") as voice_file:
                await update.message.reply_voice(voice=voice_file)

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
    app.add_handler(CommandHandler("start", start_command))
    app.add_handler(CommandHandler("id", id_command))
    app.add_handler(
        MessageHandler(
            filters.TEXT
            & ~filters.COMMAND
            & filters.Regex(r"youtube\.com|youtu\.be"),
            handle_message,
        )
    )
    return app
