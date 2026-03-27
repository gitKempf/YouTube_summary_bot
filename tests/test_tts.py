import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock, AsyncMock

from src.tts import generate_voice, TTSError, get_voice_for_language, convert_to_ogg


class TestGetVoiceForLanguage:
    def test_english(self):
        assert get_voice_for_language("en") == "en-US-GuyNeural"

    def test_russian(self):
        assert get_voice_for_language("ru") == "ru-RU-DmitryNeural"

    def test_spanish(self):
        assert get_voice_for_language("es") == "es-ES-AlvaroNeural"

    def test_french(self):
        assert get_voice_for_language("fr") == "fr-FR-HenriNeural"

    def test_german(self):
        assert get_voice_for_language("de") == "de-DE-ConradNeural"

    def test_unknown_falls_back_to_english(self):
        assert get_voice_for_language("xx") == "en-US-GuyNeural"


class TestGenerateVoice:
    @pytest.mark.asyncio
    @patch("src.tts.edge_tts.Communicate")
    async def test_calls_edge_tts_with_correct_params(self, mock_comm_class, tmp_path):
        mock_comm = MagicMock()
        mock_comm.save = AsyncMock()
        mock_comm_class.return_value = mock_comm
        output_path = tmp_path / "voice.mp3"

        await generate_voice("Hello world", output_path)

        mock_comm_class.assert_called_once_with("Hello world", "en-US-GuyNeural")
        mock_comm.save.assert_awaited_once_with(str(output_path))

    @pytest.mark.asyncio
    @patch("src.tts.edge_tts.Communicate")
    async def test_returns_output_path(self, mock_comm_class, tmp_path):
        mock_comm = MagicMock()
        mock_comm.save = AsyncMock()
        mock_comm_class.return_value = mock_comm
        output_path = tmp_path / "voice.mp3"

        result = await generate_voice("Hello world", output_path)

        assert result == output_path

    @pytest.mark.asyncio
    @patch("src.tts.edge_tts.Communicate")
    async def test_with_custom_voice(self, mock_comm_class, tmp_path):
        mock_comm = MagicMock()
        mock_comm.save = AsyncMock()
        mock_comm_class.return_value = mock_comm
        output_path = tmp_path / "voice.mp3"

        await generate_voice("Hello", output_path, voice="ru-RU-DmitryNeural")

        mock_comm_class.assert_called_once_with("Hello", "ru-RU-DmitryNeural")

    @pytest.mark.asyncio
    async def test_raises_on_empty_text(self, tmp_path):
        with pytest.raises(ValueError, match="Cannot generate voice from empty text"):
            await generate_voice("", tmp_path / "voice.mp3")

    @pytest.mark.asyncio
    @patch("src.tts.edge_tts.Communicate")
    async def test_raises_on_tts_error(self, mock_comm_class, tmp_path):
        mock_comm = MagicMock()
        mock_comm.save = AsyncMock(side_effect=Exception("TTS failure"))
        mock_comm_class.return_value = mock_comm

        with pytest.raises(TTSError, match="Edge-TTS error"):
            await generate_voice("Hello", tmp_path / "voice.mp3")


class TestConvertToOgg:
    @patch("src.tts.subprocess.run")
    def test_calls_ffmpeg(self, mock_run, tmp_path):
        mp3_file = tmp_path / "voice.mp3"
        mp3_file.write_bytes(b"fake mp3")
        mock_run.return_value = MagicMock(returncode=0)

        result = convert_to_ogg(mp3_file)

        assert result.suffix == ".ogg"
        mock_run.assert_called_once()
        cmd = mock_run.call_args[0][0]
        assert "ffmpeg" in cmd
        assert str(mp3_file) in cmd

    @patch("src.tts.subprocess.run")
    def test_raises_on_ffmpeg_error(self, mock_run, tmp_path):
        mp3_file = tmp_path / "voice.mp3"
        mp3_file.write_bytes(b"fake mp3")
        mock_run.return_value = MagicMock(returncode=1, stderr="error")

        with pytest.raises(TTSError, match="FFmpeg conversion failed"):
            convert_to_ogg(mp3_file)
