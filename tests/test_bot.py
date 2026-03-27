import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock, AsyncMock, call

from src.bot import start_command, handle_message
from src.transcriber import TranscriptionError, TranscriptionResult
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


def _patch_pipeline(mock_config):
    """Return a dict of patchers for the full pipeline."""
    return {
        "config": patch("src.bot.get_config", return_value=mock_config),
        "download": patch(
            "src.bot.download_audio",
            return_value=Path("/tmp/audio_dQw4w9WgXcQ.mp3"),
        ),
        "transcribe": patch(
            "src.bot.transcribe_audio",
            return_value=TranscriptionResult(text="Transcript text", language_code="en"),
        ),
        "summarize": patch(
            "src.bot.summarize_text",
            new_callable=AsyncMock,
            return_value="Summary text",
        ),
        "generate_voice": patch(
            "src.bot.generate_voice",
            new_callable=AsyncMock,
            return_value=Path("/tmp/voice_dQw4w9WgXcQ.ogg"),
        ),
        "convert_ogg": patch(
            "src.bot.convert_to_ogg",
            return_value=Path("/tmp/voice_dQw4w9WgXcQ.ogg"),
        ),
        "to_thread": patch(
            "src.bot.asyncio.to_thread",
            new_callable=AsyncMock,
        ),
        "path_exists": patch("src.bot.Path.exists", return_value=False),
        "path_unlink": patch("src.bot.Path.unlink"),
        "open": patch("builtins.open", MagicMock()),
    }


class TestStartCommand:
    @pytest.mark.asyncio
    async def test_sends_welcome_message(self, mock_update, mock_context):
        await start_command(mock_update, mock_context)

        mock_update.message.reply_text.assert_awaited_once()
        welcome_text = mock_update.message.reply_text.call_args[0][0]
        assert "YouTube" in welcome_text or "youtube" in welcome_text.lower()


class TestHandleMessage:
    @pytest.mark.asyncio
    async def test_sends_processing_status(self, mock_update, mock_context, mock_config):
        patches = _patch_pipeline(mock_config)
        with patches["config"], patches["download"], patches["transcribe"], \
             patches["summarize"], patches["generate_voice"], patches["convert_ogg"], \
             patches["path_exists"], patches["path_unlink"], patches["open"] as mock_open_ctx:
            mock_to_thread = AsyncMock(side_effect=[
                Path("/tmp/audio_dQw4w9WgXcQ.mp3"),
                TranscriptionResult(text="Transcript text", language_code="en"),
            ])
            with patch("src.bot.asyncio.to_thread", mock_to_thread):
                await handle_message(mock_update, mock_context)

        first_call = mock_update.message.reply_text.call_args_list[0]
        assert "Processing" in first_call[0][0] or "processing" in first_call[0][0].lower()

    @pytest.mark.asyncio
    async def test_calls_download(self, mock_update, mock_context, mock_config):
        patches = _patch_pipeline(mock_config)
        with patches["config"], patches["download"] as mock_dl, patches["transcribe"], \
             patches["summarize"], patches["generate_voice"], patches["convert_ogg"], \
             patches["path_exists"], patches["path_unlink"], patches["open"]:
            mock_to_thread = AsyncMock(side_effect=[
                Path("/tmp/audio_dQw4w9WgXcQ.mp3"),
                TranscriptionResult(text="Transcript text", language_code="en"),
            ])
            with patch("src.bot.asyncio.to_thread", mock_to_thread):
                await handle_message(mock_update, mock_context)

        # First to_thread call should be download_audio
        first_call = mock_to_thread.call_args_list[0]
        assert first_call[0][0] == mock_dl

    @pytest.mark.asyncio
    async def test_calls_transcribe(self, mock_update, mock_context, mock_config):
        patches = _patch_pipeline(mock_config)
        with patches["config"], patches["download"], patches["transcribe"] as mock_tr, \
             patches["summarize"], patches["generate_voice"], patches["convert_ogg"], \
             patches["path_exists"], patches["path_unlink"], patches["open"]:
            mock_to_thread = AsyncMock(side_effect=[
                Path("/tmp/audio_dQw4w9WgXcQ.mp3"),
                TranscriptionResult(text="Transcript text", language_code="en"),
            ])
            with patch("src.bot.asyncio.to_thread", mock_to_thread):
                await handle_message(mock_update, mock_context)

        # Second to_thread call should be transcribe_audio
        second_call = mock_to_thread.call_args_list[1]
        assert second_call[0][0] == mock_tr

    @pytest.mark.asyncio
    async def test_calls_summarize(self, mock_update, mock_context, mock_config):
        patches = _patch_pipeline(mock_config)
        with patches["config"], patches["download"], patches["transcribe"], \
             patches["summarize"] as mock_sum, patches["generate_voice"], \
             patches["convert_ogg"], patches["path_exists"], patches["path_unlink"], \
             patches["open"]:
            mock_to_thread = AsyncMock(side_effect=[
                Path("/tmp/audio_dQw4w9WgXcQ.mp3"),
                TranscriptionResult(text="Transcript text", language_code="en"),
            ])
            with patch("src.bot.asyncio.to_thread", mock_to_thread):
                await handle_message(mock_update, mock_context)

        mock_sum.assert_awaited_once()
        call_args = mock_sum.call_args
        assert call_args[0][0] == "Transcript text"

    @pytest.mark.asyncio
    async def test_calls_tts_with_detected_language(self, mock_update, mock_context, mock_config):
        patches = _patch_pipeline(mock_config)
        with patches["config"], patches["download"], patches["transcribe"], \
             patches["summarize"], patches["generate_voice"] as mock_voice, \
             patches["convert_ogg"], patches["path_exists"], patches["path_unlink"], \
             patches["open"]:
            mock_to_thread = AsyncMock(side_effect=[
                Path("/tmp/audio_dQw4w9WgXcQ.mp3"),
                TranscriptionResult(text="Transcript text", language_code="fr"),
            ])
            with patch("src.bot.asyncio.to_thread", mock_to_thread), \
                 patch("src.bot.get_voice_for_language", return_value="fr-FR-HenriNeural") as mock_gv:
                await handle_message(mock_update, mock_context)

        mock_gv.assert_called_once_with("fr")
        mock_voice.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_sends_summary_text(self, mock_update, mock_context, mock_config):
        patches = _patch_pipeline(mock_config)
        with patches["config"], patches["download"], patches["transcribe"], \
             patches["summarize"], patches["generate_voice"], patches["convert_ogg"], \
             patches["path_exists"], patches["path_unlink"], patches["open"]:
            mock_to_thread = AsyncMock(side_effect=[
                Path("/tmp/audio_dQw4w9WgXcQ.mp3"),
                TranscriptionResult(text="Transcript text", language_code="en"),
            ])
            with patch("src.bot.asyncio.to_thread", mock_to_thread):
                await handle_message(mock_update, mock_context)

        # Second reply_text should be the summary
        calls = mock_update.message.reply_text.call_args_list
        summary_sent = any("Summary text" in str(c) for c in calls)
        assert summary_sent

    @pytest.mark.asyncio
    async def test_sends_voice_message(self, mock_update, mock_context, mock_config):
        patches = _patch_pipeline(mock_config)
        with patches["config"], patches["download"], patches["transcribe"], \
             patches["summarize"], patches["generate_voice"], patches["convert_ogg"], \
             patches["path_exists"], patches["path_unlink"], patches["open"]:
            mock_to_thread = AsyncMock(side_effect=[
                Path("/tmp/audio_dQw4w9WgXcQ.mp3"),
                TranscriptionResult(text="Transcript text", language_code="en"),
            ])
            with patch("src.bot.asyncio.to_thread", mock_to_thread):
                await handle_message(mock_update, mock_context)

        mock_update.message.reply_voice.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_download_error_sends_message(self, mock_update, mock_context, mock_config):
        with patch("src.bot.get_config", return_value=mock_config), \
             patch("src.bot.extract_video_id", return_value="dQw4w9WgXcQ"), \
             patch("src.bot.asyncio.to_thread", new_callable=AsyncMock,
                   side_effect=Exception("Download failed")), \
             patch("src.bot.Path.exists", return_value=False), \
             patch("src.bot.Path.unlink"):
            await handle_message(mock_update, mock_context)

        error_call = mock_update.message.reply_text.call_args_list[-1]
        assert "Sorry" in error_call[0][0] or "error" in error_call[0][0].lower()

    @pytest.mark.asyncio
    async def test_transcription_error_sends_message(self, mock_update, mock_context, mock_config):
        with patch("src.bot.get_config", return_value=mock_config), \
             patch("src.bot.extract_video_id", return_value="dQw4w9WgXcQ"), \
             patch("src.bot.asyncio.to_thread", new_callable=AsyncMock,
                   side_effect=[Path("/tmp/audio.mp3"), TranscriptionError("Failed")]), \
             patch("src.bot.Path.exists", return_value=False), \
             patch("src.bot.Path.unlink"):
            await handle_message(mock_update, mock_context)

        error_call = mock_update.message.reply_text.call_args_list[-1]
        assert "Sorry" in error_call[0][0] or "error" in error_call[0][0].lower()

    @pytest.mark.asyncio
    async def test_summarization_error_sends_message(self, mock_update, mock_context, mock_config):
        with patch("src.bot.get_config", return_value=mock_config), \
             patch("src.bot.extract_video_id", return_value="dQw4w9WgXcQ"), \
             patch("src.bot.asyncio.to_thread", new_callable=AsyncMock,
                   side_effect=[
                       Path("/tmp/audio.mp3"),
                       TranscriptionResult(text="Text", language_code="en"),
                   ]), \
             patch("src.bot.summarize_text", new_callable=AsyncMock,
                   side_effect=SummarizationError("Failed")), \
             patch("src.bot.Path.exists", return_value=False), \
             patch("src.bot.Path.unlink"):
            await handle_message(mock_update, mock_context)

        error_call = mock_update.message.reply_text.call_args_list[-1]
        assert "Sorry" in error_call[0][0] or "error" in error_call[0][0].lower()

    @pytest.mark.asyncio
    async def test_tts_failure_still_sends_text(self, mock_update, mock_context, mock_config):
        with patch("src.bot.get_config", return_value=mock_config), \
             patch("src.bot.extract_video_id", return_value="dQw4w9WgXcQ"), \
             patch("src.bot.asyncio.to_thread", new_callable=AsyncMock,
                   side_effect=[
                       Path("/tmp/audio.mp3"),
                       TranscriptionResult(text="Text", language_code="en"),
                   ]), \
             patch("src.bot.summarize_text", new_callable=AsyncMock,
                   return_value="Summary text"), \
             patch("src.bot.get_voice_for_language", return_value="en-US-GuyNeural"), \
             patch("src.bot.generate_voice", new_callable=AsyncMock,
                   side_effect=TTSError("TTS failed")), \
             patch("src.bot.Path.exists", return_value=False), \
             patch("src.bot.Path.unlink"):
            await handle_message(mock_update, mock_context)

        # Summary text should still be sent
        calls = mock_update.message.reply_text.call_args_list
        summary_sent = any("Summary text" in str(c) for c in calls)
        assert summary_sent
        # Voice should NOT be sent
        mock_update.message.reply_voice.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_cleanup_on_exception(self, mock_update, mock_context, mock_config):
        with patch("src.bot.get_config", return_value=mock_config), \
             patch("src.bot.extract_video_id", return_value="dQw4w9WgXcQ"), \
             patch("src.bot.asyncio.to_thread", new_callable=AsyncMock,
                   side_effect=[
                       Path("/tmp/audio_dQw4w9WgXcQ.mp3"),
                       TranscriptionError("Failed"),
                   ]), \
             patch("src.bot.Path.exists", return_value=True), \
             patch("src.bot.Path.unlink") as mock_unlink:
            await handle_message(mock_update, mock_context)

        # Should attempt cleanup
        assert mock_unlink.called
