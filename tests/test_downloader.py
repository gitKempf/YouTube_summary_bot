import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.downloader import extract_video_id, download_audio


class TestExtractVideoId:
    def test_standard_url(self):
        url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
        assert extract_video_id(url) == "dQw4w9WgXcQ"

    def test_short_url(self):
        url = "https://youtu.be/dQw4w9WgXcQ"
        assert extract_video_id(url) == "dQw4w9WgXcQ"

    def test_url_with_extra_params(self):
        url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ&t=120&list=PLtest"
        assert extract_video_id(url) == "dQw4w9WgXcQ"

    def test_invalid_url_raises(self):
        with pytest.raises(ValueError, match="Could not extract video ID"):
            extract_video_id("https://example.com")

    def test_empty_string_raises(self):
        with pytest.raises(ValueError, match="Could not extract video ID"):
            extract_video_id("")


class TestDownloadAudio:
    @patch("src.downloader.yt_dlp.YoutubeDL")
    def test_calls_ytdlp_with_correct_options(self, mock_ydl_class):
        mock_ydl = MagicMock()
        mock_ydl_class.return_value.__enter__ = MagicMock(return_value=mock_ydl)
        mock_ydl_class.return_value.__exit__ = MagicMock(return_value=False)
        mock_ydl.extract_info.return_value = {"id": "dQw4w9WgXcQ"}

        download_audio("https://www.youtube.com/watch?v=dQw4w9WgXcQ", output_dir="/tmp")

        mock_ydl.extract_info.assert_called_once_with(
            "https://www.youtube.com/watch?v=dQw4w9WgXcQ", download=True
        )
        call_opts = mock_ydl_class.call_args[0][0]
        assert call_opts["format"] == "bestaudio/best"
        assert any(p["key"] == "FFmpegExtractAudio" for p in call_opts["postprocessors"])

    @patch("src.downloader.yt_dlp.YoutubeDL")
    def test_returns_file_path(self, mock_ydl_class):
        mock_ydl = MagicMock()
        mock_ydl_class.return_value.__enter__ = MagicMock(return_value=mock_ydl)
        mock_ydl_class.return_value.__exit__ = MagicMock(return_value=False)
        mock_ydl.extract_info.return_value = {"id": "dQw4w9WgXcQ"}

        result = download_audio(
            "https://www.youtube.com/watch?v=dQw4w9WgXcQ", output_dir="/tmp"
        )

        assert isinstance(result, Path)
        assert "dQw4w9WgXcQ" in str(result)
        assert str(result).endswith(".mp3")
