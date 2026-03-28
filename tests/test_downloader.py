import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.downloader import extract_video_id, download_audio, fetch_transcript, TranscriptFetchResult


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


class TestFetchTranscript:
    @patch("src.downloader.YouTubeTranscriptApi")
    def test_returns_transcript_when_available(self, mock_api_class):
        mock_api = MagicMock()
        mock_api_class.return_value = mock_api
        mock_transcript = MagicMock()
        mock_snippet = MagicMock()
        mock_snippet.text = "Hello world"
        mock_transcript.snippets = [mock_snippet]
        mock_transcript.language_code = "en"
        mock_api.fetch.return_value = mock_transcript

        result = fetch_transcript("dQw4w9WgXcQ")

        assert isinstance(result, TranscriptFetchResult)
        assert result.text == "Hello world"
        assert result.language_code == "en"

    @patch("src.downloader.YouTubeTranscriptApi")
    def test_returns_none_on_error(self, mock_api_class):
        mock_api = MagicMock()
        mock_api_class.return_value = mock_api
        mock_api.fetch.side_effect = Exception("No captions")

        result = fetch_transcript("dQw4w9WgXcQ")

        assert result is None

    @patch("src.downloader.YouTubeTranscriptApi")
    def test_returns_none_on_empty_text(self, mock_api_class):
        mock_api = MagicMock()
        mock_api_class.return_value = mock_api
        mock_transcript = MagicMock()
        mock_transcript.snippets = []
        mock_transcript.language_code = "en"
        mock_api.fetch.return_value = mock_transcript

        result = fetch_transcript("dQw4w9WgXcQ")

        assert result is None


class TestDownloadAudio:
    @patch("src.downloader.YouTube")
    def test_downloads_audio_stream(self, mock_yt_class):
        mock_yt = MagicMock()
        mock_yt_class.return_value = mock_yt
        mock_stream = MagicMock()
        mock_yt.streams.filter.return_value.first.return_value = mock_stream

        result = download_audio(
            "https://www.youtube.com/watch?v=dQw4w9WgXcQ", output_dir="/tmp"
        )

        mock_stream.download.assert_called_once_with(
            output_path="/tmp", filename="audio_dQw4w9WgXcQ.mp4"
        )
        assert isinstance(result, Path)
        assert "dQw4w9WgXcQ" in str(result)

    @patch("src.downloader.YouTube")
    def test_raises_when_no_audio_stream(self, mock_yt_class):
        mock_yt = MagicMock()
        mock_yt_class.return_value = mock_yt
        mock_yt.streams.filter.return_value.first.return_value = None

        with pytest.raises(RuntimeError, match="No audio stream found"):
            download_audio("https://www.youtube.com/watch?v=dQw4w9WgXcQ")
