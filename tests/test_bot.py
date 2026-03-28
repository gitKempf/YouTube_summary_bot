import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock, AsyncMock

from src.bot import start_command, handle_message, _progress_bar, split_message, ProgressTracker
from src.downloader import TranscriptFetchResult
from src.transcriber import TranscriptionError, TranscriptionResult
from src.summarizer import SummarizationError
from src.tts import TTSError


@pytest.fixture
def mock_config():
    config = MagicMock()
    config.telegram_bot_token = "fake-token"
    config.elevenlabs_api_key = "fake-el-key"
    config.anthropic_api_key = "fake-ant-key"
    config.tts_voice = "en-US-RogerNeural"
    config.claude_model = "claude-sonnet-4-6"
    config.max_tokens = 4096
    return config


@pytest.fixture
def mock_status_msg():
    msg = MagicMock()
    msg.edit_text = AsyncMock()
    return msg


def _happy_patches(mock_config, transcript=None, summary="Summary"):
    if transcript is None:
        transcript = TranscriptFetchResult(text="Transcript", language_code="en")
    return {
        "config": patch("src.bot.get_config", return_value=mock_config),
        "vid": patch("src.bot.extract_video_id", return_value="dQw4w9WgXcQ"),
        "thread": patch("src.bot.asyncio.to_thread", new_callable=AsyncMock,
                        return_value=transcript),
        "sum": patch("src.bot.summarize_text", new_callable=AsyncMock,
                     return_value=summary),
        "voice": patch("src.bot.get_voice_for_language",
                       return_value="en-US-RogerNeural"),
        "tts": patch("src.bot.generate_voice_chunked", new_callable=AsyncMock,
                     return_value=[Path("/tmp/v.ogg")]),
        "open": patch("builtins.open", MagicMock()),
        "exists": patch("src.bot.Path.exists", return_value=False),
        "unlink": patch("src.bot.Path.unlink"),
    }


class TestProgressBar:
    def test_zero_percent(self):
        result = _progress_bar(0, "Starting...")
        assert "0%" in result
        assert "Starting..." in result

    def test_hundred_percent(self):
        result = _progress_bar(100, "Done!")
        assert "100%" in result
        assert "Done!" in result

    def test_mid_percent(self):
        result = _progress_bar(50, "Summarizing...")
        assert "50%" in result

    def test_bar_width(self):
        result = _progress_bar(50, "Test")
        bar_line = result.split("\n")[0]
        assert len(bar_line.split(" ")[0]) == 15  # PROGRESS_BAR_WIDTH


class TestProgressTracker:
    @pytest.mark.asyncio
    async def test_updates_message(self):
        msg = MagicMock()
        msg.edit_text = AsyncMock()
        tracker = ProgressTracker(msg)
        await tracker.update(50, "Working...")
        msg.edit_text.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_skips_duplicate_update(self):
        msg = MagicMock()
        msg.edit_text = AsyncMock()
        tracker = ProgressTracker(msg)
        await tracker.update(50, "Working...")
        await tracker.update(50, "Working...")
        assert msg.edit_text.await_count == 1

    @pytest.mark.asyncio
    async def test_handles_telegram_error(self):
        msg = MagicMock()
        msg.edit_text = AsyncMock(side_effect=Exception("Rate limited"))
        tracker = ProgressTracker(msg)
        await tracker.update(50, "Working...")  # Should not raise


class TestSplitMessage:
    def test_short_message_unchanged(self):
        assert split_message("Hello") == ["Hello"]

    def test_long_message_splits(self):
        text = "\n\n".join([f"Paragraph {i}." for i in range(100)])
        parts = split_message(text, max_length=200)
        assert len(parts) > 1
        for part in parts:
            assert len(part) <= 200

    def test_preserves_all_content(self):
        text = "\n\n".join([f"Paragraph {i}." for i in range(20)])
        parts = split_message(text, max_length=100)
        combined = "\n\n".join(parts)
        for i in range(20):
            assert f"Paragraph {i}." in combined


class TestStartCommand:
    @pytest.mark.asyncio
    async def test_sends_welcome_message(self, mock_update, mock_context):
        await start_command(mock_update, mock_context)
        mock_update.message.reply_text.assert_awaited_once()
        welcome_text = mock_update.message.reply_text.call_args[0][0]
        assert "YouTube" in welcome_text or "youtube" in welcome_text.lower()


class TestHandleMessageWithCaptions:
    @pytest.mark.asyncio
    async def test_progress_bar_updates(self, mock_update, mock_context, mock_config, mock_status_msg):
        mock_update.message.reply_text = AsyncMock(side_effect=[mock_status_msg, None])
        p = _happy_patches(mock_config)
        with p["config"], p["vid"], p["thread"], p["sum"], p["voice"], \
             p["tts"], p["open"], p["exists"], p["unlink"]:
            await handle_message(mock_update, mock_context)
        # Should have multiple progress edits (start, transcript, summarize, voice, done)
        assert mock_status_msg.edit_text.await_count >= 4

    @pytest.mark.asyncio
    async def test_uses_youtube_captions(self, mock_update, mock_context, mock_config, mock_status_msg):
        mock_update.message.reply_text = AsyncMock(side_effect=[mock_status_msg, None])
        p = _happy_patches(mock_config,
                           TranscriptFetchResult(text="Caption text", language_code="en"))
        with p["config"], p["vid"], p["thread"] as mt, p["sum"] as ms, \
             p["voice"], p["tts"], p["open"], p["exists"], p["unlink"]:
            await handle_message(mock_update, mock_context)
        assert mt.call_count == 1
        assert "Caption text" in str(ms.call_args)

    @pytest.mark.asyncio
    async def test_sends_summary_and_voice(self, mock_update, mock_context, mock_config, mock_status_msg):
        mock_update.message.reply_text = AsyncMock(side_effect=[mock_status_msg, None])
        p = _happy_patches(mock_config, summary="My summary")
        with p["config"], p["vid"], p["thread"], p["sum"], p["voice"], \
             p["tts"], p["open"], p["exists"], p["unlink"]:
            await handle_message(mock_update, mock_context)
        calls = mock_update.message.reply_text.call_args_list
        assert any("My summary" in str(c) for c in calls)
        mock_update.message.reply_voice.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_tts_with_detected_language(self, mock_update, mock_context, mock_config, mock_status_msg):
        mock_update.message.reply_text = AsyncMock(side_effect=[mock_status_msg, None])
        p = _happy_patches(mock_config,
                           TranscriptFetchResult(text="Bonjour", language_code="fr"))
        with p["config"], p["vid"], p["thread"], p["sum"], \
             patch("src.bot.get_voice_for_language", return_value="fr-FR-HenriNeural") as gv, \
             p["tts"], p["open"], p["exists"], p["unlink"]:
            await handle_message(mock_update, mock_context)
        gv.assert_called_once_with("fr")


class TestHandleMessageFallback:
    @pytest.mark.asyncio
    async def test_falls_back_to_audio_download(self, mock_update, mock_context, mock_config, mock_status_msg):
        mock_update.message.reply_text = AsyncMock(side_effect=[mock_status_msg, None])
        with patch("src.bot.get_config", return_value=mock_config), \
             patch("src.bot.extract_video_id", return_value="dQw4w9WgXcQ"), \
             patch("src.bot.asyncio.to_thread", new_callable=AsyncMock,
                   side_effect=[None, Path("/tmp/a.mp4"),
                                TranscriptionResult(text="EL text", language_code="en")]) as mt, \
             patch("src.bot.summarize_text", new_callable=AsyncMock, return_value="S") as ms, \
             patch("src.bot.get_voice_for_language", return_value="en-US-RogerNeural"), \
             patch("src.bot.generate_voice_chunked", new_callable=AsyncMock,
                   return_value=[Path("/tmp/v.ogg")]), \
             patch("builtins.open", MagicMock()), \
             patch("src.bot.Path.exists", return_value=False), \
             patch("src.bot.Path.unlink"):
            await handle_message(mock_update, mock_context)
        assert mt.call_count == 3
        assert "EL text" in str(ms.call_args)


class TestHandleMessageErrors:
    @pytest.mark.asyncio
    async def test_download_error(self, mock_update, mock_context, mock_config, mock_status_msg):
        mock_update.message.reply_text = AsyncMock(side_effect=[mock_status_msg, None])
        with patch("src.bot.get_config", return_value=mock_config), \
             patch("src.bot.extract_video_id", return_value="dQw4w9WgXcQ"), \
             patch("src.bot.asyncio.to_thread", new_callable=AsyncMock,
                   side_effect=[None, RuntimeError("Fail")]), \
             patch("src.bot.Path.exists", return_value=False), \
             patch("src.bot.Path.unlink"):
            await handle_message(mock_update, mock_context)
        assert "Sorry" in mock_update.message.reply_text.call_args_list[-1][0][0]

    @pytest.mark.asyncio
    async def test_tts_failure_still_sends_text(self, mock_update, mock_context, mock_config, mock_status_msg):
        mock_update.message.reply_text = AsyncMock(side_effect=[mock_status_msg, None])
        with patch("src.bot.get_config", return_value=mock_config), \
             patch("src.bot.extract_video_id", return_value="dQw4w9WgXcQ"), \
             patch("src.bot.asyncio.to_thread", new_callable=AsyncMock,
                   return_value=TranscriptFetchResult(text="T", language_code="en")), \
             patch("src.bot.summarize_text", new_callable=AsyncMock, return_value="Summary"), \
             patch("src.bot.get_voice_for_language", return_value="en-US-RogerNeural"), \
             patch("src.bot.generate_voice_chunked", new_callable=AsyncMock,
                   side_effect=TTSError("Fail")), \
             patch("src.bot.Path.exists", return_value=False), \
             patch("src.bot.Path.unlink"):
            await handle_message(mock_update, mock_context)
        calls = mock_update.message.reply_text.call_args_list
        assert any("Summary" in str(c) for c in calls)
        mock_update.message.reply_voice.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_cleanup_on_exception(self, mock_update, mock_context, mock_config, mock_status_msg):
        mock_update.message.reply_text = AsyncMock(side_effect=[mock_status_msg, None])
        with patch("src.bot.get_config", return_value=mock_config), \
             patch("src.bot.extract_video_id", return_value="dQw4w9WgXcQ"), \
             patch("src.bot.asyncio.to_thread", new_callable=AsyncMock,
                   side_effect=[None, Path("/tmp/a.mp4"), TranscriptionError("Fail")]), \
             patch("src.bot.Path.exists", return_value=True), \
             patch("src.bot.Path.unlink") as mu:
            await handle_message(mock_update, mock_context)
        assert mu.called
