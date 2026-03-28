import asyncio
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from pytubefix import YouTube
from youtube_transcript_api import YouTubeTranscriptApi

YOUTUBE_REGEX = re.compile(
    r"(?:youtube\.com/watch\?v=|youtu\.be/)([\w-]{11})"
)


@dataclass
class TranscriptFetchResult:
    text: str
    language_code: str


def extract_video_id(url: str) -> str:
    match = YOUTUBE_REGEX.search(url)
    if not match:
        raise ValueError(f"Could not extract video ID from URL: {url}")
    return match.group(1)


def fetch_transcript(video_id: str) -> Optional[TranscriptFetchResult]:
    """Try to fetch YouTube's built-in captions. Returns None if unavailable."""
    try:
        ytt_api = YouTubeTranscriptApi()
        transcript = ytt_api.fetch(video_id)
        text = " ".join(entry.text for entry in transcript.snippets)
        if text.strip():
            return TranscriptFetchResult(
                text=text, language_code=transcript.language_code
            )
    except Exception:
        pass
    return None


def download_audio(url: str, output_dir: str = "/tmp") -> Path:
    video_id = extract_video_id(url)
    output_path = Path(output_dir) / f"audio_{video_id}.mp4"

    yt = YouTube(url)
    audio_stream = yt.streams.filter(only_audio=True).first()
    if not audio_stream:
        raise RuntimeError(f"No audio stream found for {url}")

    audio_stream.download(output_path=output_dir, filename=f"audio_{video_id}.mp4")
    return output_path
