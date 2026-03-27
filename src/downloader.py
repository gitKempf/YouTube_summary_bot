import asyncio
import re
from pathlib import Path

import yt_dlp

YOUTUBE_REGEX = re.compile(
    r"(?:youtube\.com/watch\?v=|youtu\.be/)([\w-]{11})"
)


def extract_video_id(url: str) -> str:
    match = YOUTUBE_REGEX.search(url)
    if not match:
        raise ValueError(f"Could not extract video ID from URL: {url}")
    return match.group(1)


def download_audio(url: str, output_dir: str = "/tmp") -> Path:
    video_id = extract_video_id(url)
    outtmpl = f"{output_dir}/audio_{video_id}.%(ext)s"

    ydl_opts = {
        "format": "bestaudio/best",
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "mp3",
                "preferredquality": "192",
            }
        ],
        "outtmpl": outtmpl,
        "quiet": True,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.extract_info(url, download=True)

    return Path(f"{output_dir}/audio_{video_id}.mp3")


async def download_audio_async(url: str, output_dir: str = "/tmp") -> Path:
    return await asyncio.to_thread(download_audio, url, output_dir)
