import asyncio
import re
import subprocess
from pathlib import Path

import edge_tts


class TTSError(Exception):
    pass


# Upgraded to Multilingual/Copilot-grade voices where available (more natural)
VOICE_MAP = {
    "en": "en-US-AndrewMultilingualNeural",
    "eng": "en-US-AndrewMultilingualNeural",
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

TTS_TIMEOUT_SECONDS = 120
TTS_RATE = "-5%"
TTS_OGG_BITRATE = "128k"


def get_voice_for_language(language_code: str) -> str:
    return VOICE_MAP.get(language_code, "en-US-AndrewMultilingualNeural")


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
    # Collapse multiple newlines
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


async def generate_voice(
    text: str,
    output_path: Path,
    voice: str = "en-US-AndrewMultilingualNeural",
) -> Path:
    if not text or not text.strip():
        raise ValueError("Cannot generate voice from empty text")

    clean_text = strip_markdown(text)

    try:
        communicate = edge_tts.Communicate(clean_text, voice, rate=TTS_RATE)
        await asyncio.wait_for(
            communicate.save(str(output_path)),
            timeout=TTS_TIMEOUT_SECONDS,
        )
    except asyncio.TimeoutError:
        raise TTSError(f"TTS timed out after {TTS_TIMEOUT_SECONDS}s")
    except Exception as e:
        raise TTSError(f"Edge-TTS error: {e}") from e

    return output_path


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
