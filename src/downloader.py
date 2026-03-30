import json
import re
import subprocess
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

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


def fetch_video_title(video_id: str) -> str:
    """Fetch video title via YouTube oEmbed API (no API key needed)."""
    try:
        url = f"https://www.youtube.com/oembed?url=https://youtube.com/watch?v={video_id}&format=json"
        resp = urllib.request.urlopen(url, timeout=5)
        data = json.loads(resp.read())
        return data.get("title", "")
    except Exception:
        return ""


def download_audio(url: str, output_dir: str = "/tmp") -> Path:
    """Download audio from YouTube using yt-dlp (handles bot detection)."""
    video_id = extract_video_id(url)
    output_path = Path(output_dir) / f"audio_{video_id}.mp4"

    output_path = output_path.with_suffix(".m4a")
    cmd = [
        "yt-dlp",
        "--extract-audio",
        "--audio-format", "m4a",
        "--output", str(output_path),
        "--no-playlist",
        "--quiet",
        url,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if result.returncode != 0:
        raise RuntimeError(f"yt-dlp failed: {result.stderr[:300]}")

    # yt-dlp may save with different extension, find the actual file
    if not output_path.exists():
        for ext in [".m4a", ".webm", ".opus", ".mp3"]:
            alt = output_path.with_suffix(ext)
            if alt.exists():
                return alt
        raise RuntimeError(f"Downloaded file not found at {output_path}")

    return output_path
