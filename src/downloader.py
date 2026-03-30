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


def _build_ytdlp_cmd(url: str, output_path: Path, cookies_file: Optional[str] = None,
                      player_client: Optional[str] = None) -> list[str]:
    """Build yt-dlp command with appropriate flags."""
    cmd = [
        "yt-dlp",
        "--extract-audio",
        "--audio-format", "m4a",
        "--output", str(output_path),
        "--no-playlist",
        "--quiet",
        "--force-ipv4",
    ]

    # Build extractor args
    extractor_parts = ["fetch_pot=always"]
    if player_client:
        extractor_parts.append(f"player_client={player_client}")
    cmd.extend(["--extractor-args", f"youtube:{';'.join(extractor_parts)}"])

    # Use cookies file if available
    if cookies_file and Path(cookies_file).exists():
        cmd.extend(["--cookies", cookies_file])

    cmd.append(url)
    return cmd


# Player client combinations to try, in order of likelihood to work
_PLAYER_CLIENTS = [
    None,                    # default (android_vr + web_embedded)
    "web",                   # standard web client
    "tv,mweb",               # TV + mobile web
    "ios,android",           # mobile clients
]


def download_audio(url: str, output_dir: str = "/tmp") -> Path:
    """Download audio from YouTube using yt-dlp with PO token support and client fallback."""
    import os
    video_id = extract_video_id(url)
    output_path = Path(output_dir) / f"audio_{video_id}.m4a"

    cookies_file = os.environ.get("YOUTUBE_COOKIES_FILE", "/app/cookies.txt")
    cookies = cookies_file if Path(cookies_file).exists() else None

    last_error = ""
    is_bot_error = False
    for client in _PLAYER_CLIENTS:
        # Clean up any partial downloads from previous attempts
        for ext in [".m4a", ".webm", ".opus", ".mp3", ".part"]:
            output_path.with_suffix(ext).unlink(missing_ok=True)

        cmd = _build_ytdlp_cmd(url, output_path, cookies, client)
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)

        if result.returncode == 0:
            # Find the output file
            if output_path.exists():
                return output_path
            for ext in [".m4a", ".webm", ".opus", ".mp3"]:
                alt = output_path.with_suffix(ext)
                if alt.exists():
                    return alt

        full_stderr = result.stderr
        last_error = full_stderr[:500]
        # Retry with different clients for bot detection or YouTube API errors
        is_youtube_block = (
            "Sign in to confirm" in full_stderr
            or "LOGIN_REQUIRED" in full_stderr
            or "HTTP Error 400" in full_stderr
        )
        if not is_youtube_block:
            break

    if is_youtube_block:
        raise RuntimeError(
            f"YouTube is blocking audio download for this video from this server. "
            f"The video may not have captions and cannot be processed. "
            f"Try a video with subtitles/captions enabled."
        )
    raise RuntimeError(f"yt-dlp failed: {last_error}")
