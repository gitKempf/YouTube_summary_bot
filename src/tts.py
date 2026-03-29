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


# Unicode ranges for non-Latin script detection
_SCRIPT_LANG_MAP = [
    (r'[\u0400-\u04FF]', 'ru'),   # Cyrillic → Russian
    (r'[\u0600-\u06FF]', 'ar'),   # Arabic
    (r'[\u3040-\u309F\u30A0-\u30FF]', 'ja'),  # Japanese
    (r'[\uAC00-\uD7AF]', 'ko'),  # Korean
    (r'[\u4E00-\u9FFF]', 'zh'),   # Chinese
    (r'[\u0900-\u097F]', 'hi'),   # Devanagari → Hindi
    (r'[\u0E00-\u0E7F]', 'th'),   # Thai
]

import re as _re

def detect_language_from_text(text: str) -> str:
    """Detect language from text content using script analysis.

    Returns a language code (e.g. 'ru', 'en') based on the dominant script.
    """
    sample = text[:2000]

    for pattern, lang in _SCRIPT_LANG_MAP:
        matches = len(_re.findall(pattern, sample))
        if matches > 20:
            return lang

    return "en"


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


# Approximate bytes of mp3 audio per character of text (empirical)
BYTES_PER_CHAR_ESTIMATE = 300


async def generate_voice(
    text: str,
    output_path: Path,
    voice: str = "en-US-RogerNeural",
    on_progress: ProgressCallback = None,
) -> Path:
    if not text or not text.strip():
        raise ValueError("Cannot generate voice from empty text")

    clean_text = strip_markdown(text)
    last_error = None
    expected_bytes = len(clean_text) * BYTES_PER_CHAR_ESTIMATE

    for attempt in range(TTS_MAX_RETRIES + 1):
        try:
            communicate = edge_tts.Communicate(clean_text, voice, rate=TTS_RATE)

            async def _stream_save():
                audio_bytes = 0
                last_pct = -1
                with open(output_path, "wb") as f:
                    async for chunk in communicate.stream():
                        if chunk["type"] == "audio":
                            f.write(chunk["data"])
                            audio_bytes += len(chunk["data"])
                            if on_progress and expected_bytes > 0:
                                pct = min(audio_bytes / expected_bytes, 0.99)
                                pct_int = int(pct * 100)
                                if pct_int > last_pct:
                                    last_pct = pct_int
                                    await on_progress(pct, 1.0)

            await asyncio.wait_for(
                _stream_save(),
                timeout=TTS_TIMEOUT_SECONDS,
            )
            return output_path
        except asyncio.TimeoutError:
            last_error = f"TTS timed out after {TTS_TIMEOUT_SECONDS}s (attempt {attempt + 1})"
        except Exception as e:
            last_error = f"Edge-TTS error: {e}"
        if attempt < TTS_MAX_RETRIES:
            await asyncio.sleep(1)

    raise TTSError(last_error)


async def generate_voice_chunked(
    text: str,
    output_dir: str,
    video_id: str,
    voice: str = "en-US-RogerNeural",
    on_progress: ProgressCallback = None,
) -> List[Path]:
    """Generate voice with fine-grained progress reporting.

    Args:
        on_progress: async callback(completed_fraction, total=1.0) called frequently
                     during audio generation with values from 0.0 to 1.0.
    """
    clean_text = strip_markdown(text)
    chunks = split_text_chunks(clean_text)
    total_chunks = len(chunks)

    async def _chunk_progress(chunk_idx: int, frac_within_chunk: float, _total: float):
        if on_progress:
            overall = (chunk_idx + frac_within_chunk) / total_chunks
            await on_progress(overall, 1.0)

    mp3_paths = []
    for i, chunk in enumerate(chunks):
        mp3_path = Path(output_dir) / f"voice_{video_id}_part{i}.mp3"
        mp3_paths.append(mp3_path)

        async def _inner_progress(frac, _t, idx=i):
            await _chunk_progress(idx, frac, _t)

        await generate_voice(chunk, mp3_path, voice=voice, on_progress=_inner_progress)
        # Report chunk fully done
        if on_progress:
            await on_progress((i + 1) / total_chunks, 1.0)

    # Concatenate chunks into a single audio file
    if len(mp3_paths) == 1:
        final_mp3 = mp3_paths[0]
    else:
        final_mp3 = Path(output_dir) / f"voice_{video_id}.mp3"
        concat_mp3(mp3_paths, final_mp3)

    return [convert_to_ogg(final_mp3)]


def concat_mp3(parts: List[Path], output: Path) -> Path:
    """Concatenate multiple mp3 files into one using ffmpeg."""
    list_file = output.with_suffix(".txt")
    list_file.write_text("\n".join(f"file '{p}'" for p in parts))
    cmd = [
        "ffmpeg", "-y", "-f", "concat", "-safe", "0",
        "-i", str(list_file), "-c", "copy", str(output),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    list_file.unlink(missing_ok=True)
    if result.returncode != 0:
        raise TTSError(f"FFmpeg concat failed: {result.stderr}")
    return output


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
