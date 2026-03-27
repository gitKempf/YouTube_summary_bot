import subprocess
from pathlib import Path

import edge_tts


class TTSError(Exception):
    pass


VOICE_MAP = {
    "en": "en-US-GuyNeural",
    "ru": "ru-RU-DmitryNeural",
    "es": "es-ES-AlvaroNeural",
    "fr": "fr-FR-HenriNeural",
    "de": "de-DE-ConradNeural",
    "it": "it-IT-DiegoNeural",
    "pt": "pt-BR-AntonioNeural",
    "ja": "ja-JP-KeitaNeural",
    "ko": "ko-KR-InJoonNeural",
    "zh": "zh-CN-YunxiNeural",
    "ar": "ar-SA-HamedNeural",
    "hi": "hi-IN-MadhurNeural",
    "uk": "uk-UA-OstapNeural",
    "pl": "pl-PL-MarekNeural",
    "tr": "tr-TR-AhmetNeural",
}


def get_voice_for_language(language_code: str) -> str:
    return VOICE_MAP.get(language_code, "en-US-GuyNeural")


async def generate_voice(
    text: str,
    output_path: Path,
    voice: str = "en-US-GuyNeural",
) -> Path:
    if not text or not text.strip():
        raise ValueError("Cannot generate voice from empty text")

    try:
        communicate = edge_tts.Communicate(text, voice)
        await communicate.save(str(output_path))
    except Exception as e:
        raise TTSError(f"Edge-TTS error: {e}") from e

    return output_path


def convert_to_ogg(mp3_path: Path) -> Path:
    ogg_path = mp3_path.with_suffix(".ogg")
    cmd = [
        "ffmpeg", "-y", "-i", str(mp3_path),
        "-c:a", "libopus", "-b:a", "64k",
        str(ogg_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise TTSError(f"FFmpeg conversion failed: {result.stderr}")
    return ogg_path
