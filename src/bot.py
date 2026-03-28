import asyncio
import logging
from pathlib import Path

from telegram import Update
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
from src.tts import generate_voice, convert_to_ogg, get_voice_for_language, TTSError

logger = logging.getLogger(__name__)

PROGRESS_STEPS = [
    "Fetching transcript...",
    "Downloading audio...",
    "Transcribing audio...",
    "Generating summary...",
    "Creating voice message...",
]


def _progress_bar(step: int, total: int, label: str) -> str:
    filled = step
    empty = total - step
    bar = "\u2593" * filled + "\u2591" * empty
    percent = int(step / total * 100)
    return f"{bar} {percent}%\n{label}"


async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Welcome! Send me a YouTube video link and I'll create a summary for you.\n"
        "You'll receive both a text summary and a voice message."
    )


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    url = update.message.text
    config = get_config()
    audio_path = None
    voice_path = None

    try:
        video_id = extract_video_id(url)

        # Send initial progress message that we'll edit throughout
        status_msg = await update.message.reply_text(
            _progress_bar(0, 5, "Starting...")
        )

        # Step 1: Try YouTube captions first (fast)
        await status_msg.edit_text(_progress_bar(1, 5, "Fetching transcript..."))
        transcript_result = await asyncio.to_thread(fetch_transcript, video_id)

        if transcript_result:
            transcript_text = transcript_result.text
            language_code = transcript_result.language_code
            logger.info(f"Got transcript from YouTube captions ({language_code})")
        else:
            # Fallback: download audio + ElevenLabs STT
            await status_msg.edit_text(_progress_bar(1, 5, "Downloading audio..."))
            logger.info("No captions available, downloading audio for transcription")
            audio_path = await asyncio.to_thread(download_audio, url)

            await status_msg.edit_text(_progress_bar(2, 5, "Transcribing audio..."))
            transcription = await asyncio.to_thread(
                transcribe_audio, audio_path, config.elevenlabs_api_key
            )
            transcript_text = transcription.text
            language_code = transcription.language_code

        # Step 2: Summarize
        await status_msg.edit_text(_progress_bar(3, 5, "Generating summary..."))
        summary = await summarize_text(
            transcript_text,
            config.anthropic_api_key,
            model=config.claude_model,
            max_tokens=config.max_tokens,
        )

        # Step 3: Generate voice
        await status_msg.edit_text(_progress_bar(4, 5, "Creating voice message..."))
        voice_sent = False
        try:
            voice = get_voice_for_language(language_code)
            voice_mp3 = Path(f"/tmp/voice_{video_id}.mp3")
            await generate_voice(summary, voice_mp3, voice=voice)
            voice_path = convert_to_ogg(voice_mp3)
            voice_sent = True
        except TTSError as e:
            logger.warning(f"TTS failed, skipping voice: {e}")

        # Step 4: Done — replace progress with summary
        await status_msg.edit_text(_progress_bar(5, 5, "Done!"))
        await update.message.reply_text(summary)

        if voice_sent and voice_path:
            with open(voice_path, "rb") as voice_file:
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
        for path in [audio_path, voice_path]:
            if path and Path(path).exists():
                try:
                    Path(path).unlink()
                except OSError:
                    pass
        # Also clean up the mp3 voice file if ogg was created
        if voice_path:
            mp3_path = voice_path.with_suffix(".mp3")
            if mp3_path.exists():
                try:
                    mp3_path.unlink()
                except OSError:
                    pass


def create_app():
    config = get_config()
    app = ApplicationBuilder().token(config.telegram_bot_token).build()
    app.add_handler(CommandHandler("start", start_command))
    app.add_handler(
        MessageHandler(
            filters.TEXT
            & ~filters.COMMAND
            & filters.Regex(r"youtube\.com|youtu\.be"),
            handle_message,
        )
    )
    return app
