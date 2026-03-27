from dataclasses import dataclass
from pathlib import Path

from elevenlabs.client import ElevenLabs


class TranscriptionError(Exception):
    pass


@dataclass
class TranscriptionResult:
    text: str
    language_code: str


def transcribe_audio(
    audio_path: Path, api_key: str, language_code: str = "en"
) -> TranscriptionResult:
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    client = ElevenLabs(api_key=api_key)

    try:
        with open(audio_path, "rb") as f:
            response = client.speech_to_text.convert(
                file=f,
                model_id="scribe_v2",
                language_code=language_code,
            )
    except TranscriptionError:
        raise
    except Exception as e:
        raise TranscriptionError(f"ElevenLabs API error: {e}") from e

    text = response.text
    if not text or not text.strip():
        raise TranscriptionError("Transcription returned empty text")

    detected_language = getattr(response, "language_code", language_code)

    return TranscriptionResult(text=text, language_code=detected_language)
