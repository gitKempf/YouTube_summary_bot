import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock, call

from youtube_transcript_api._errors import NoTranscriptFound, TranscriptsDisabled

from src.downloader import (
    extract_video_id, download_audio, fetch_transcript,
    TranscriptFetchResult, _build_ytdlp_cmd, _PLAYER_CLIENTS,
    _TRANSCRIPT_RETRIES,
)


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

    @patch("src.downloader.time.sleep")
    @patch("src.downloader.YouTubeTranscriptApi")
    def test_retries_on_network_error(self, mock_api_class, mock_sleep):
        """Network errors should trigger retries."""
        mock_api = MagicMock()
        mock_api_class.return_value = mock_api
        mock_api.fetch.side_effect = ConnectionError("Network unreachable")

        result = fetch_transcript("dQw4w9WgXcQ")

        assert result is None
        assert mock_api.fetch.call_count == _TRANSCRIPT_RETRIES

    @patch("src.downloader.time.sleep")
    @patch("src.downloader.YouTubeTranscriptApi")
    def test_no_retry_on_transcripts_disabled(self, mock_api_class, mock_sleep):
        """TranscriptsDisabled means no captions — don't retry."""
        mock_api = MagicMock()
        mock_api_class.return_value = mock_api
        mock_api.fetch.side_effect = TranscriptsDisabled("test123")

        result = fetch_transcript("test123")

        assert result is None
        assert mock_api.fetch.call_count == 1

    @patch("src.downloader.time.sleep")
    @patch("src.downloader.YouTubeTranscriptApi")
    def test_no_retry_on_no_transcript_found(self, mock_api_class, mock_sleep):
        """NoTranscriptFound means no matching captions — don't retry."""
        mock_api = MagicMock()
        mock_api_class.return_value = mock_api
        mock_api.fetch.side_effect = NoTranscriptFound("test123", ["en"], [])

        result = fetch_transcript("test123")

        assert result is None
        assert mock_api.fetch.call_count == 1

    @patch("src.downloader.time.sleep")
    @patch("src.downloader.YouTubeTranscriptApi")
    def test_succeeds_after_retry(self, mock_api_class, mock_sleep):
        """Should return transcript if retry succeeds."""
        mock_api = MagicMock()
        mock_api_class.return_value = mock_api
        mock_transcript = MagicMock()
        mock_snippet = MagicMock()
        mock_snippet.text = "Hello"
        mock_transcript.snippets = [mock_snippet]
        mock_transcript.language_code = "en"
        mock_api.fetch.side_effect = [
            ConnectionError("fail"),
            mock_transcript,
        ]

        result = fetch_transcript("dQw4w9WgXcQ")

        assert result is not None
        assert result.text == "Hello"
        assert mock_api.fetch.call_count == 2

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


class TestBuildYtdlpCmd:
    def test_basic_command(self):
        cmd = _build_ytdlp_cmd(
            "https://www.youtube.com/watch?v=test123",
            Path("/tmp/audio_test123.m4a"),
        )
        assert cmd[0] == "yt-dlp"
        assert "--extract-audio" in cmd
        assert "--audio-format" in cmd
        assert cmd[cmd.index("--audio-format") + 1] == "m4a"
        assert "--no-playlist" in cmd
        assert "--force-ipv4" in cmd
        assert "https://www.youtube.com/watch?v=test123" == cmd[-1]

    def test_includes_fetch_pot_always(self):
        cmd = _build_ytdlp_cmd(
            "https://www.youtube.com/watch?v=test123",
            Path("/tmp/audio_test123.m4a"),
        )
        extractor_idx = cmd.index("--extractor-args") + 1
        assert "fetch_pot=always" in cmd[extractor_idx]

    def test_with_player_client(self):
        cmd = _build_ytdlp_cmd(
            "https://www.youtube.com/watch?v=test123",
            Path("/tmp/audio_test123.m4a"),
            player_client="tv,mweb",
        )
        extractor_idx = cmd.index("--extractor-args") + 1
        assert "player_client=tv,mweb" in cmd[extractor_idx]
        assert "fetch_pot=always" in cmd[extractor_idx]

    def test_with_cookies(self):
        with patch.object(Path, "exists", return_value=True):
            cmd = _build_ytdlp_cmd(
                "https://www.youtube.com/watch?v=test123",
                Path("/tmp/audio_test123.m4a"),
                cookies_file="/app/cookies.txt",
            )
        assert "--cookies" in cmd
        assert "/app/cookies.txt" in cmd

    def test_no_cookies_when_file_missing(self):
        with patch.object(Path, "exists", return_value=False):
            cmd = _build_ytdlp_cmd(
                "https://www.youtube.com/watch?v=test123",
                Path("/tmp/audio_test123.m4a"),
                cookies_file="/app/cookies.txt",
            )
        assert "--cookies" not in cmd

    def test_no_cookies_when_none(self):
        cmd = _build_ytdlp_cmd(
            "https://www.youtube.com/watch?v=test123",
            Path("/tmp/audio_test123.m4a"),
            cookies_file=None,
        )
        assert "--cookies" not in cmd


class TestDownloadAudio:
    @patch("src.downloader.subprocess.run")
    @patch("src.downloader.Path.exists", return_value=True)
    def test_downloads_audio_with_ytdlp(self, mock_exists, mock_run):
        mock_run.return_value = MagicMock(returncode=0, stderr="")

        result = download_audio(
            "https://www.youtube.com/watch?v=dQw4w9WgXcQ", output_dir="/tmp"
        )

        mock_run.assert_called_once()
        cmd = mock_run.call_args[0][0]
        assert cmd[0] == "yt-dlp"
        assert "--extract-audio" in cmd
        assert isinstance(result, Path)
        assert "dQw4w9WgXcQ" in str(result)

    @patch("src.downloader.subprocess.run")
    @patch("src.downloader.Path.exists", return_value=True)
    def test_uses_valid_audio_format(self, mock_exists, mock_run):
        """yt-dlp requires m4a not mp4 as audio format."""
        mock_run.return_value = MagicMock(returncode=0, stderr="")
        download_audio("https://www.youtube.com/watch?v=dQw4w9WgXcQ")
        cmd = mock_run.call_args[0][0]
        fmt_idx = cmd.index("--audio-format") + 1
        assert cmd[fmt_idx] == "m4a", f"Expected m4a but got {cmd[fmt_idx]}"

    @patch("src.downloader.subprocess.run")
    @patch("src.downloader.Path.exists", return_value=True)
    def test_output_has_m4a_extension(self, mock_exists, mock_run):
        """Output path must use .m4a extension, not .mp4."""
        mock_run.return_value = MagicMock(returncode=0, stderr="")
        result = download_audio("https://www.youtube.com/watch?v=dQw4w9WgXcQ")
        assert result.suffix == ".m4a"
        cmd = mock_run.call_args[0][0]
        output_arg = cmd[cmd.index("--output") + 1]
        assert output_arg.endswith(".m4a")

    @patch("src.downloader.subprocess.run")
    @patch("src.downloader.Path.exists", return_value=True)
    def test_no_playlist_flag(self, mock_exists, mock_run):
        """yt-dlp must have --no-playlist to avoid downloading entire playlists."""
        mock_run.return_value = MagicMock(returncode=0, stderr="")
        download_audio("https://www.youtube.com/watch?v=dQw4w9WgXcQ")
        cmd = mock_run.call_args[0][0]
        assert "--no-playlist" in cmd

    @patch("src.downloader.subprocess.run")
    @patch("src.downloader.Path.exists", return_value=True)
    def test_force_ipv4_flag(self, mock_exists, mock_run):
        """yt-dlp must use --force-ipv4 to avoid IPv6 issues on cloud servers."""
        mock_run.return_value = MagicMock(returncode=0, stderr="")
        download_audio("https://www.youtube.com/watch?v=dQw4w9WgXcQ")
        cmd = mock_run.call_args[0][0]
        assert "--force-ipv4" in cmd

    @patch("src.downloader.subprocess.run")
    @patch("src.downloader.Path.exists", return_value=True)
    def test_fetch_pot_always(self, mock_exists, mock_run):
        """yt-dlp must use fetch_pot=always extractor arg for PO token support."""
        mock_run.return_value = MagicMock(returncode=0, stderr="")
        download_audio("https://www.youtube.com/watch?v=dQw4w9WgXcQ")
        cmd = mock_run.call_args[0][0]
        extractor_idx = cmd.index("--extractor-args") + 1
        assert "fetch_pot=always" in cmd[extractor_idx]

    @patch("src.downloader.subprocess.run")
    @patch("src.downloader.Path.exists", return_value=True)
    def test_timeout_set(self, mock_exists, mock_run):
        """subprocess.run must have a timeout to prevent hanging."""
        mock_run.return_value = MagicMock(returncode=0, stderr="")
        download_audio("https://www.youtube.com/watch?v=dQw4w9WgXcQ")
        assert mock_run.call_args[1].get("timeout") == 120

    @patch("src.downloader.subprocess.run")
    def test_raises_on_ytdlp_failure(self, mock_run):
        mock_run.return_value = MagicMock(returncode=1, stderr="ERROR: video unavailable")

        with pytest.raises(RuntimeError, match="yt-dlp failed"):
            download_audio("https://www.youtube.com/watch?v=dQw4w9WgXcQ")

    @patch("src.downloader.subprocess.run")
    def test_raises_on_js_runtime_error(self, mock_run):
        """yt-dlp without deno/node should fail with a clear error."""
        mock_run.return_value = MagicMock(
            returncode=1,
            stderr="WARNING: No supported JavaScript runtime could be found"
        )
        with pytest.raises(RuntimeError, match="yt-dlp failed"):
            download_audio("https://www.youtube.com/watch?v=dQw4w9WgXcQ")

    @patch("src.downloader.subprocess.run")
    def test_retries_with_different_clients_on_bot_detection(self, mock_run):
        """Should try multiple player clients when bot detection is hit."""
        mock_run.return_value = MagicMock(
            returncode=1,
            stderr="ERROR: [youtube] xyz: Sign in to confirm you're not a bot"
        )
        with pytest.raises(RuntimeError, match="YouTube is blocking"):
            download_audio("https://www.youtube.com/watch?v=dQw4w9WgXcQ")

        # Should have tried all player client combinations
        assert mock_run.call_count == len(_PLAYER_CLIENTS)

    @patch("src.downloader.subprocess.run")
    def test_retries_on_login_required(self, mock_run):
        """Should retry when LOGIN_REQUIRED appears in stderr."""
        mock_run.return_value = MagicMock(
            returncode=1,
            stderr="WARNING: lots of text... LOGIN_REQUIRED ... more text"
        )
        with pytest.raises(RuntimeError, match="YouTube is blocking"):
            download_audio("https://www.youtube.com/watch?v=dQw4w9WgXcQ")

        assert mock_run.call_count == len(_PLAYER_CLIENTS)

    @patch("src.downloader.subprocess.run")
    def test_retries_on_http_400(self, mock_run):
        """Should retry when YouTube returns HTTP 400 (often bot detection)."""
        mock_run.return_value = MagicMock(
            returncode=1,
            stderr="WARNING: [youtube] HTTP Error 400: Bad Request. Retrying..."
        )
        with pytest.raises(RuntimeError, match="YouTube is blocking"):
            download_audio("https://www.youtube.com/watch?v=dQw4w9WgXcQ")

        assert mock_run.call_count == len(_PLAYER_CLIENTS)

    @patch("src.downloader.subprocess.run")
    def test_bot_detection_error_message_is_user_friendly(self, mock_run):
        """Bot detection error should suggest trying a video with captions."""
        mock_run.return_value = MagicMock(
            returncode=1,
            stderr="ERROR: Sign in to confirm you're not a bot"
        )
        with pytest.raises(RuntimeError, match="Try a video with subtitles"):
            download_audio("https://www.youtube.com/watch?v=dQw4w9WgXcQ")

    @patch("src.downloader.subprocess.run")
    @patch("src.downloader.Path.exists", return_value=True)
    def test_stops_retrying_on_non_bot_error(self, mock_exists, mock_run):
        """Non-bot errors should not trigger retry with different clients."""
        mock_run.return_value = MagicMock(
            returncode=1,
            stderr="ERROR: video unavailable"
        )
        with pytest.raises(RuntimeError, match="yt-dlp failed"):
            download_audio("https://www.youtube.com/watch?v=dQw4w9WgXcQ")

        # Should only try once for non-bot errors
        assert mock_run.call_count == 1

    @patch("src.downloader.subprocess.run")
    @patch("src.downloader.Path.exists", return_value=True)
    def test_succeeds_on_second_client(self, mock_exists, mock_run):
        """Should succeed if a later player client works."""
        mock_run.side_effect = [
            MagicMock(returncode=1, stderr="Sign in to confirm you're not a bot"),
            MagicMock(returncode=0, stderr=""),
        ]
        result = download_audio("https://www.youtube.com/watch?v=dQw4w9WgXcQ")
        assert mock_run.call_count == 2
        assert isinstance(result, Path)

    @patch("src.downloader.subprocess.run")
    def test_raises_when_no_file_found(self, mock_run):
        """Should raise if yt-dlp succeeds but no output file exists."""
        mock_run.return_value = MagicMock(returncode=0, stderr="")
        with patch.object(Path, "exists", return_value=False):
            # After all clients fail to produce a file, the last error is empty
            # so it falls through to "yt-dlp failed" with empty stderr
            with pytest.raises(RuntimeError):
                download_audio("https://www.youtube.com/watch?v=dQw4w9WgXcQ")
