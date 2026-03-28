import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock, AsyncMock

from src.bot import handle_message
from src.downloader import TranscriptFetchResult
from src.transcriber import TranscriptionResult
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


@pytest.fixture
def mock_update(mock_status_msg):
    update = MagicMock()
    update.message = MagicMock()
    update.message.text = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
    update.message.reply_text = AsyncMock(side_effect=[mock_status_msg, None])
    update.message.reply_voice = AsyncMock()
    update.effective_chat = MagicMock()
    update.effective_chat.id = 12345
    return update


@pytest.fixture
def mock_context():
    return MagicMock()


class TestFullPipeline:
    @pytest.mark.asyncio
    async def test_caption_path_full_pipeline(self, mock_update, mock_context, mock_config, mock_status_msg):
        with patch("src.bot.get_config", return_value=mock_config), \
             patch("src.bot.extract_video_id", return_value="dQw4w9WgXcQ"), \
             patch("src.bot.asyncio.to_thread", new_callable=AsyncMock,
                   return_value=TranscriptFetchResult(text="Full transcript", language_code="en")), \
             patch("src.bot.summarize_text", new_callable=AsyncMock,
                   return_value="Complete summary") as mock_sum, \
             patch("src.bot.get_voice_for_language", return_value="en-US-RogerNeural"), \
             patch("src.bot.generate_voice_chunked", new_callable=AsyncMock,
                   return_value=[Path("/tmp/voice.ogg")]) as mock_voice, \
             patch("builtins.open", MagicMock()), \
             patch("src.bot.Path.exists", return_value=False), \
             patch("src.bot.Path.unlink"):
            await handle_message(mock_update, mock_context)

        mock_sum.assert_awaited_once()
        mock_voice.assert_awaited_once()
        assert any("Complete summary" in str(c) for c in mock_update.message.reply_text.call_args_list)
        mock_update.message.reply_voice.assert_awaited_once()
        assert mock_status_msg.edit_text.await_count >= 3

    @pytest.mark.asyncio
    async def test_fallback_path_full_pipeline(self, mock_update, mock_context, mock_config, mock_status_msg):
        with patch("src.bot.get_config", return_value=mock_config), \
             patch("src.bot.extract_video_id", return_value="dQw4w9WgXcQ"), \
             patch("src.bot.asyncio.to_thread", new_callable=AsyncMock,
                   side_effect=[None, Path("/tmp/audio.mp4"),
                                TranscriptionResult(text="EL text", language_code="fr")]), \
             patch("src.bot.summarize_text", new_callable=AsyncMock,
                   return_value="Summary") as mock_sum, \
             patch("src.bot.get_voice_for_language", return_value="fr-FR-HenriNeural"), \
             patch("src.bot.generate_voice_chunked", new_callable=AsyncMock,
                   return_value=[Path("/tmp/voice.ogg")]), \
             patch("builtins.open", MagicMock()), \
             patch("src.bot.Path.exists", return_value=False), \
             patch("src.bot.Path.unlink"):
            await handle_message(mock_update, mock_context)

        mock_sum.assert_awaited_once()
        assert "EL text" in str(mock_sum.call_args)

    @pytest.mark.asyncio
    async def test_tts_failure_graceful_degradation(self, mock_update, mock_context, mock_config, mock_status_msg):
        with patch("src.bot.get_config", return_value=mock_config), \
             patch("src.bot.extract_video_id", return_value="dQw4w9WgXcQ"), \
             patch("src.bot.asyncio.to_thread", new_callable=AsyncMock,
                   return_value=TranscriptFetchResult(text="Text", language_code="en")), \
             patch("src.bot.summarize_text", new_callable=AsyncMock, return_value="Summary"), \
             patch("src.bot.get_voice_for_language", return_value="en-US-RogerNeural"), \
             patch("src.bot.generate_voice_chunked", new_callable=AsyncMock,
                   side_effect=TTSError("TTS failed")), \
             patch("src.bot.Path.exists", return_value=False), \
             patch("src.bot.Path.unlink"):
            await handle_message(mock_update, mock_context)

        assert any("Summary" in str(c) for c in mock_update.message.reply_text.call_args_list)
        mock_update.message.reply_voice.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_cleanup_always_happens(self, mock_update, mock_context, mock_config, mock_status_msg):
        with patch("src.bot.get_config", return_value=mock_config), \
             patch("src.bot.extract_video_id", return_value="dQw4w9WgXcQ"), \
             patch("src.bot.asyncio.to_thread", new_callable=AsyncMock,
                   side_effect=[None, Path("/tmp/audio.mp4"),
                                TranscriptionResult(text="Text", language_code="en")]), \
             patch("src.bot.summarize_text", new_callable=AsyncMock,
                   side_effect=SummarizationError("Failed")), \
             patch("src.bot.Path.exists", return_value=True), \
             patch("src.bot.Path.unlink") as mock_unlink:
            await handle_message(mock_update, mock_context)

        assert mock_unlink.called
