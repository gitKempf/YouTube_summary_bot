import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock, AsyncMock

from src.bot import start_command, handle_message, _progress_bar, split_message
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
    config.tts_voice = "en-US-AndrewMultilingualNeural"
    config.claude_model = "claude-sonnet-4-6"
    config.max_tokens = 4096
    return config


@pytest.fixture
def mock_status_msg():
    msg = MagicMock()
    msg.edit_text = AsyncMock()
    return msg


def _happy_path_patches(mock_config, transcript_result=None, summary="Summary"):
    """Common patches for the happy path."""
    if transcript_result is None:
        transcript_result = TranscriptFetchResult(text="Transcript", language_code="en")
    return {
        "config": patch("src.bot.get_config", return_value=mock_config),
        "video_id": patch("src.bot.extract_video_id", return_value="dQw4w9WgXcQ"),
        "to_thread": patch("src.bot.asyncio.to_thread", new_callable=AsyncMock,
                           return_value=transcript_result),
        "summarize": patch("src.bot.summarize_text", new_callable=AsyncMock,
                           return_value=summary),
        "get_voice": patch("src.bot.get_voice_for_language",
                           return_value="en-US-AndrewMultilingualNeural"),
        "tts": patch("src.bot.generate_voice_chunked", new_callable=AsyncMock,
                      return_value=[Path("/tmp/v.ogg")]),
        "open": patch("builtins.open", MagicMock()),
        "exists": patch("src.bot.Path.exists", return_value=False),
        "unlink": patch("src.bot.Path.unlink"),
    }


class TestProgressBar:
    def test_zero_progress(self):
        result = _progress_bar(0, 5, "Starting...")
        assert "0%" in result
        assert "Starting..." in result

    def test_full_progress(self):
        result = _progress_bar(5, 5, "Done!")
        assert "100%" in result
        assert "Done!" in result

    def test_partial_progress(self):
        result = _progress_bar(3, 5, "Working...")
        assert "60%" in result


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
    async def test_edits_progress_messages(self, mock_update, mock_context, mock_config, mock_status_msg):
        mock_update.message.reply_text = AsyncMock(side_effect=[mock_status_msg, None])
        p = _happy_path_patches(mock_config)
        with p["config"], p["video_id"], p["to_thread"], p["summarize"], \
             p["get_voice"], p["tts"], p["open"], p["exists"], p["unlink"]:
            await handle_message(mock_update, mock_context)
        assert mock_status_msg.edit_text.await_count >= 3

    @pytest.mark.asyncio
    async def test_uses_youtube_captions(self, mock_update, mock_context, mock_config, mock_status_msg):
        mock_update.message.reply_text = AsyncMock(side_effect=[mock_status_msg, None])
        p = _happy_path_patches(mock_config,
                                TranscriptFetchResult(text="Caption text", language_code="en"))
        with p["config"], p["video_id"], p["to_thread"] as mock_thread, \
             p["summarize"] as mock_sum, p["get_voice"], p["tts"], p["open"], \
             p["exists"], p["unlink"]:
            await handle_message(mock_update, mock_context)
        assert mock_thread.call_count == 1
        assert "Caption text" in str(mock_sum.call_args)

    @pytest.mark.asyncio
    async def test_sends_summary_text(self, mock_update, mock_context, mock_config, mock_status_msg):
        mock_update.message.reply_text = AsyncMock(side_effect=[mock_status_msg, None])
        p = _happy_path_patches(mock_config, summary="Summary text")
        with p["config"], p["video_id"], p["to_thread"], p["summarize"], \
             p["get_voice"], p["tts"], p["open"], p["exists"], p["unlink"]:
            await handle_message(mock_update, mock_context)
        calls = mock_update.message.reply_text.call_args_list
        assert any("Summary text" in str(c) for c in calls)

    @pytest.mark.asyncio
    async def test_sends_voice_message(self, mock_update, mock_context, mock_config, mock_status_msg):
        mock_update.message.reply_text = AsyncMock(side_effect=[mock_status_msg, None])
        p = _happy_path_patches(mock_config)
        with p["config"], p["video_id"], p["to_thread"], p["summarize"], \
             p["get_voice"], p["tts"], p["open"], p["exists"], p["unlink"]:
            await handle_message(mock_update, mock_context)
        mock_update.message.reply_voice.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_tts_with_detected_language(self, mock_update, mock_context, mock_config, mock_status_msg):
        mock_update.message.reply_text = AsyncMock(side_effect=[mock_status_msg, None])
        p = _happy_path_patches(mock_config,
                                TranscriptFetchResult(text="Bonjour", language_code="fr"))
        with p["config"], p["video_id"], p["to_thread"], p["summarize"], \
             patch("src.bot.get_voice_for_language", return_value="fr-FR-HenriNeural") as mock_gv, \
             p["tts"], p["open"], p["exists"], p["unlink"]:
            await handle_message(mock_update, mock_context)
        mock_gv.assert_called_once_with("fr")


class TestHandleMessageFallbackToElevenLabs:
    @pytest.mark.asyncio
    async def test_falls_back_to_audio_download(self, mock_update, mock_context, mock_config, mock_status_msg):
        mock_update.message.reply_text = AsyncMock(side_effect=[mock_status_msg, None])
        with patch("src.bot.get_config", return_value=mock_config), \
             patch("src.bot.extract_video_id", return_value="dQw4w9WgXcQ"), \
             patch("src.bot.asyncio.to_thread", new_callable=AsyncMock,
                   side_effect=[None, Path("/tmp/audio.mp4"),
                                TranscriptionResult(text="EL transcript", language_code="en")]) as mt, \
             patch("src.bot.summarize_text", new_callable=AsyncMock, return_value="Summary") as ms, \
             patch("src.bot.get_voice_for_language", return_value="en-US-AndrewMultilingualNeural"), \
             patch("src.bot.generate_voice_chunked", new_callable=AsyncMock,
                   return_value=[Path("/tmp/v.ogg")]), \
             patch("builtins.open", MagicMock()), \
             patch("src.bot.Path.exists", return_value=False), \
             patch("src.bot.Path.unlink"):
            await handle_message(mock_update, mock_context)
        assert mt.call_count == 3
        assert "EL transcript" in str(ms.call_args)


class TestHandleMessageErrors:
    @pytest.mark.asyncio
    async def test_download_error(self, mock_update, mock_context, mock_config, mock_status_msg):
        mock_update.message.reply_text = AsyncMock(side_effect=[mock_status_msg, None])
        with patch("src.bot.get_config", return_value=mock_config), \
             patch("src.bot.extract_video_id", return_value="dQw4w9WgXcQ"), \
             patch("src.bot.asyncio.to_thread", new_callable=AsyncMock,
                   side_effect=[None, RuntimeError("Download failed")]), \
             patch("src.bot.Path.exists", return_value=False), \
             patch("src.bot.Path.unlink"):
            await handle_message(mock_update, mock_context)
        error_call = mock_update.message.reply_text.call_args_list[-1]
        assert "Sorry" in error_call[0][0] or "error" in error_call[0][0].lower()

    @pytest.mark.asyncio
    async def test_transcription_error(self, mock_update, mock_context, mock_config, mock_status_msg):
        mock_update.message.reply_text = AsyncMock(side_effect=[mock_status_msg, None])
        with patch("src.bot.get_config", return_value=mock_config), \
             patch("src.bot.extract_video_id", return_value="dQw4w9WgXcQ"), \
             patch("src.bot.asyncio.to_thread", new_callable=AsyncMock,
                   side_effect=[None, Path("/tmp/audio.mp4"), TranscriptionError("Failed")]), \
             patch("src.bot.Path.exists", return_value=False), \
             patch("src.bot.Path.unlink"):
            await handle_message(mock_update, mock_context)
        error_call = mock_update.message.reply_text.call_args_list[-1]
        assert "Sorry" in error_call[0][0]

    @pytest.mark.asyncio
    async def test_summarization_error(self, mock_update, mock_context, mock_config, mock_status_msg):
        mock_update.message.reply_text = AsyncMock(side_effect=[mock_status_msg, None])
        with patch("src.bot.get_config", return_value=mock_config), \
             patch("src.bot.extract_video_id", return_value="dQw4w9WgXcQ"), \
             patch("src.bot.asyncio.to_thread", new_callable=AsyncMock,
                   return_value=TranscriptFetchResult(text="Text", language_code="en")), \
             patch("src.bot.summarize_text", new_callable=AsyncMock,
                   side_effect=SummarizationError("Failed")), \
             patch("src.bot.Path.exists", return_value=False), \
             patch("src.bot.Path.unlink"):
            await handle_message(mock_update, mock_context)
        error_call = mock_update.message.reply_text.call_args_list[-1]
        assert "Sorry" in error_call[0][0]

    @pytest.mark.asyncio
    async def test_tts_failure_still_sends_text(self, mock_update, mock_context, mock_config, mock_status_msg):
        mock_update.message.reply_text = AsyncMock(side_effect=[mock_status_msg, None])
        with patch("src.bot.get_config", return_value=mock_config), \
             patch("src.bot.extract_video_id", return_value="dQw4w9WgXcQ"), \
             patch("src.bot.asyncio.to_thread", new_callable=AsyncMock,
                   return_value=TranscriptFetchResult(text="Text", language_code="en")), \
             patch("src.bot.summarize_text", new_callable=AsyncMock, return_value="Summary text"), \
             patch("src.bot.get_voice_for_language", return_value="en-US-AndrewMultilingualNeural"), \
             patch("src.bot.generate_voice_chunked", new_callable=AsyncMock,
                   side_effect=TTSError("Failed")), \
             patch("src.bot.Path.exists", return_value=False), \
             patch("src.bot.Path.unlink"):
            await handle_message(mock_update, mock_context)
        calls = mock_update.message.reply_text.call_args_list
        assert any("Summary text" in str(c) for c in calls)
        mock_update.message.reply_voice.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_cleanup_on_exception(self, mock_update, mock_context, mock_config, mock_status_msg):
        mock_update.message.reply_text = AsyncMock(side_effect=[mock_status_msg, None])
        with patch("src.bot.get_config", return_value=mock_config), \
             patch("src.bot.extract_video_id", return_value="dQw4w9WgXcQ"), \
             patch("src.bot.asyncio.to_thread", new_callable=AsyncMock,
                   side_effect=[None, Path("/tmp/audio.mp4"), TranscriptionError("Failed")]), \
             patch("src.bot.Path.exists", return_value=True), \
             patch("src.bot.Path.unlink") as mock_unlink:
            await handle_message(mock_update, mock_context)
        assert mock_unlink.called
