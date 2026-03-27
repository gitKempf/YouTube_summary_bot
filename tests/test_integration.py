import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock, AsyncMock

from src.bot import handle_message
from src.transcriber import TranscriptionResult, TranscriptionError
from src.summarizer import SummarizationError
from src.tts import TTSError


@pytest.fixture
def mock_config():
    config = MagicMock()
    config.telegram_bot_token = "fake-token"
    config.elevenlabs_api_key = "fake-el-key"
    config.anthropic_api_key = "fake-ant-key"
    config.tts_voice = "en-US-GuyNeural"
    config.claude_model = "claude-sonnet-4-6"
    config.max_tokens = 4096
    return config


@pytest.fixture
def mock_update():
    update = MagicMock()
    update.message = MagicMock()
    update.message.text = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
    update.message.reply_text = AsyncMock()
    update.message.reply_voice = AsyncMock()
    update.effective_chat = MagicMock()
    update.effective_chat.id = 12345
    return update


@pytest.fixture
def mock_context():
    return MagicMock()


class TestFullPipeline:
    @pytest.mark.asyncio
    async def test_full_pipeline_mocked(self, mock_update, mock_context, mock_config):
        """Verify data flows correctly through the entire pipeline."""
        with patch("src.bot.get_config", return_value=mock_config), \
             patch("src.bot.extract_video_id", return_value="dQw4w9WgXcQ"), \
             patch("src.bot.asyncio.to_thread", new_callable=AsyncMock,
                   side_effect=[
                       Path("/tmp/audio_dQw4w9WgXcQ.mp3"),
                       TranscriptionResult(text="Full transcript here", language_code="en"),
                   ]) as mock_thread, \
             patch("src.bot.summarize_text", new_callable=AsyncMock,
                   return_value="Complete summary") as mock_sum, \
             patch("src.bot.get_voice_for_language", return_value="en-US-GuyNeural"), \
             patch("src.bot.generate_voice", new_callable=AsyncMock,
                   return_value=Path("/tmp/voice.mp3")) as mock_voice, \
             patch("src.bot.convert_to_ogg",
                   return_value=Path("/tmp/voice.ogg")) as mock_ogg, \
             patch("builtins.open", MagicMock()), \
             patch("src.bot.Path.exists", return_value=False), \
             patch("src.bot.Path.unlink"):

            await handle_message(mock_update, mock_context)

        # Verify pipeline order: download -> transcribe -> summarize -> tts -> send
        assert mock_thread.call_count == 2  # download + transcribe
        mock_sum.assert_awaited_once()
        assert "Full transcript here" in str(mock_sum.call_args)
        mock_voice.assert_awaited_once()
        assert "Complete summary" in str(mock_voice.call_args)
        mock_ogg.assert_called_once()

        # Verify both text and voice were sent
        text_calls = mock_update.message.reply_text.call_args_list
        assert len(text_calls) >= 2  # processing + summary
        mock_update.message.reply_voice.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_download_failure_stops_pipeline(self, mock_update, mock_context, mock_config):
        """When download fails, transcription/summarization/TTS should not run."""
        with patch("src.bot.get_config", return_value=mock_config), \
             patch("src.bot.extract_video_id", return_value="dQw4w9WgXcQ"), \
             patch("src.bot.asyncio.to_thread", new_callable=AsyncMock,
                   side_effect=Exception("Download failed")), \
             patch("src.bot.summarize_text", new_callable=AsyncMock) as mock_sum, \
             patch("src.bot.generate_voice", new_callable=AsyncMock) as mock_voice, \
             patch("src.bot.Path.exists", return_value=False), \
             patch("src.bot.Path.unlink"):

            await handle_message(mock_update, mock_context)

        mock_sum.assert_not_awaited()
        mock_voice.assert_not_awaited()
        mock_update.message.reply_voice.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_tts_failure_graceful_degradation(self, mock_update, mock_context, mock_config):
        """When TTS fails, text summary should still be sent."""
        with patch("src.bot.get_config", return_value=mock_config), \
             patch("src.bot.extract_video_id", return_value="dQw4w9WgXcQ"), \
             patch("src.bot.asyncio.to_thread", new_callable=AsyncMock,
                   side_effect=[
                       Path("/tmp/audio.mp3"),
                       TranscriptionResult(text="Text", language_code="en"),
                   ]), \
             patch("src.bot.summarize_text", new_callable=AsyncMock,
                   return_value="Summary"), \
             patch("src.bot.get_voice_for_language", return_value="en-US-GuyNeural"), \
             patch("src.bot.generate_voice", new_callable=AsyncMock,
                   side_effect=TTSError("TTS failed")), \
             patch("src.bot.Path.exists", return_value=False), \
             patch("src.bot.Path.unlink"):

            await handle_message(mock_update, mock_context)

        # Text summary should be sent
        calls = [str(c) for c in mock_update.message.reply_text.call_args_list]
        assert any("Summary" in c for c in calls)
        # Voice should NOT be sent
        mock_update.message.reply_voice.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_cleanup_always_happens(self, mock_update, mock_context, mock_config):
        """Temp files should be cleaned up even when summarization fails."""
        with patch("src.bot.get_config", return_value=mock_config), \
             patch("src.bot.extract_video_id", return_value="dQw4w9WgXcQ"), \
             patch("src.bot.asyncio.to_thread", new_callable=AsyncMock,
                   side_effect=[
                       Path("/tmp/audio_dQw4w9WgXcQ.mp3"),
                       TranscriptionResult(text="Text", language_code="en"),
                   ]), \
             patch("src.bot.summarize_text", new_callable=AsyncMock,
                   side_effect=SummarizationError("Failed")), \
             patch("src.bot.Path.exists", return_value=True), \
             patch("src.bot.Path.unlink") as mock_unlink:

            await handle_message(mock_update, mock_context)

        assert mock_unlink.called
