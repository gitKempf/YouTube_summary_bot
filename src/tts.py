import asyncio
import re
import subprocess
from pathlib import Path
from typing import Callable, Awaitable, List, Optional

import edge_tts


class TTSError(Exception):
    pass


# Standard Neural voices — reliable, fast, no timeouts
# (Multilingual voices hang on longer text)
VOICE_MAP = {
    "en": "en-US-RogerNeural",
    "eng": "en-US-RogerNeural",
    "ru": "ru-RU-DmitryNeural",
    "rus": "ru-RU-DmitryNeural",
    "es": "es-ES-AlvaroNeural",
    "spa": "es-ES-AlvaroNeural",
    "fr": "fr-FR-HenriNeural",
    "fra": "fr-FR-HenriNeural",
    "de": "de-DE-ConradNeural",
    "deu": "de-DE-ConradNeural",
    "it": "it-IT-DiegoNeural",
    "ita": "it-IT-DiegoNeural",
    "pt": "pt-BR-AntonioNeural",
    "por": "pt-BR-AntonioNeural",
    "ja": "ja-JP-KeitaNeural",
    "jpn": "ja-JP-KeitaNeural",
    "ko": "ko-KR-InJoonNeural",
    "kor": "ko-KR-InJoonNeural",
    "zh": "zh-CN-YunxiNeural",
    "zho": "zh-CN-YunxiNeural",
    "ar": "ar-SA-HamedNeural",
    "ara": "ar-SA-HamedNeural",
    "hi": "hi-IN-MadhurNeural",
    "hin": "hi-IN-MadhurNeural",
    "uk": "uk-UA-OstapNeural",
    "ukr": "uk-UA-OstapNeural",
    "pl": "pl-PL-MarekNeural",
    "pol": "pl-PL-MarekNeural",
    "tr": "tr-TR-AhmetNeural",
    "tur": "tr-TR-AhmetNeural",
}

TTS_TIMEOUT_SECONDS = 60
TTS_RATE = "+10%"
TTS_OGG_BITRATE = "128k"
TTS_CHUNK_MAX_CHARS = 3000
TTS_MAX_RETRIES = 2

ProgressCallback = Optional[Callable[[int, int], Awaitable[None]]]


def get_voice_for_language(language_code: str) -> str:
    return VOICE_MAP.get(language_code, "en-US-RogerNeural")


def strip_markdown(text: str) -> str:
    """Remove markdown formatting that causes edge-tts to hang or sound wrong."""
    text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)
    text = re.sub(r"\*(.+?)\*", r"\1", text)
    text = re.sub(r"__(.+?)__", r"\1", text)
    text = re.sub(r"_(.+?)_", r"\1", text)
    text = re.sub(r"`(.+?)`", r"\1", text)
    text = re.sub(r"^\s*[-*]\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*\d+\.\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"\[(.+?)\]\(.+?\)", r"\1", text)
    text = re.sub(r"---+", "", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def split_text_chunks(text: str, max_chars: int = TTS_CHUNK_MAX_CHARS) -> List[str]:
    """Split text into chunks at paragraph boundaries, respecting max_chars."""
    if len(text) <= max_chars:
        return [text]

    chunks = []
    paragraphs = text.split("\n\n")
    current_chunk = ""

    for para in paragraphs:
        if len(current_chunk) + len(para) + 2 <= max_chars:
            current_chunk = current_chunk + "\n\n" + para if current_chunk else para
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            if len(para) > max_chars:
                sentences = re.split(r"(?<=[.!?])\s+", para)
                current_chunk = ""
                for sent in sentences:
                    if len(current_chunk) + len(sent) + 1 <= max_chars:
                        current_chunk = current_chunk + " " + sent if current_chunk else sent
                    else:
                        if current_chunk:
                            chunks.append(current_chunk.strip())
                        current_chunk = sent
            else:
                current_chunk = para

    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    return chunks


async def generate_voice(
    text: str,
    output_path: Path,
    voice: str = "en-US-RogerNeural",
) -> Path:
    if not text or not text.strip():
        raise ValueError("Cannot generate voice from empty text")

    clean_text = strip_markdown(text)
    last_error = None

    for attempt in range(TTS_MAX_RETRIES + 1):
        try:
            communicate = edge_tts.Communicate(clean_text, voice, rate=TTS_RATE)
            await asyncio.wait_for(
                communicate.save(str(output_path)),
                timeout=TTS_TIMEOUT_SECONDS,
            )
            return output_path
        except asyncio.TimeoutError:
            last_error = f"TTS timed out after {TTS_TIMEOUT_SECONDS}s (attempt {attempt + 1})"
        except Exception as e:
            last_error = f"Edge-TTS error: {e}"
        # Brief pause before retry
        if attempt < TTS_MAX_RETRIES:
            await asyncio.sleep(1)

    raise TTSError(last_error)


async def generate_voice_chunked(
    text: str,
    output_dir: str,
    video_id: str,
    voice: str = "en-US-RogerNeural",
    on_chunk_done: ProgressCallback = None,
) -> List[Path]:
    """Generate voice for long text by splitting into chunks.

    Args:
        on_chunk_done: async callback(completed, total) called after each chunk finishes.
    """
    clean_text = strip_markdown(text)
    chunks = split_text_chunks(clean_text)
    total = len(chunks)

    mp3_paths = []
    for i, chunk in enumerate(chunks):
        mp3_path = Path(output_dir) / f"voice_{video_id}_part{i}.mp3"
        mp3_paths.append(mp3_path)
        await generate_voice(chunk, mp3_path, voice=voice)
        if on_chunk_done:
            await on_chunk_done(i + 1, total)

    ogg_paths = []
    for mp3_path in mp3_paths:
        ogg_paths.append(convert_to_ogg(mp3_path))

    return ogg_paths


def convert_to_ogg(mp3_path: Path) -> Path:
    ogg_path = mp3_path.with_suffix(".ogg")
    cmd = [
        "ffmpeg", "-y", "-i", str(mp3_path),
        "-c:a", "libopus", "-b:a", TTS_OGG_BITRATE,
        str(ogg_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise TTSError(f"FFmpeg conversion failed: {result.stderr}")
    return ogg_path
