import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock, AsyncMock

from src.tts import generate_voice, TTSError, get_voice_for_language, convert_to_ogg, strip_markdown


class TestGetVoiceForLanguage:
    def test_english_2letter(self):
        assert get_voice_for_language("en") == "en-US-GuyNeural"

    def test_english_3letter(self):
        assert get_voice_for_language("eng") == "en-US-GuyNeural"

    def test_russian_2letter(self):
        assert get_voice_for_language("ru") == "ru-RU-DmitryNeural"

    def test_russian_3letter(self):
        assert get_voice_for_language("rus") == "ru-RU-DmitryNeural"

    def test_spanish(self):
        assert get_voice_for_language("es") == "es-ES-AlvaroNeural"

    def test_french(self):
        assert get_voice_for_language("fr") == "fr-FR-HenriNeural"

    def test_german(self):
        assert get_voice_for_language("de") == "de-DE-ConradNeural"

    def test_unknown_falls_back_to_english(self):
        assert get_voice_for_language("xx") == "en-US-GuyNeural"


class TestStripMarkdown:
    def test_strips_headers(self):
        assert strip_markdown("## Hello World") == "Hello World"

    def test_strips_bold(self):
        assert strip_markdown("This is **bold** text") == "This is bold text"

    def test_strips_italic(self):
        assert strip_markdown("This is *italic* text") == "This is italic text"

    def test_strips_bullet_points(self):
        result = strip_markdown("- Item one\n- Item two")
        assert "Item one" in result
        assert "- " not in result

    def test_strips_links(self):
        assert strip_markdown("[click here](http://example.com)") == "click here"

    def test_strips_horizontal_rules(self):
        assert "---" not in strip_markdown("Above\n---\nBelow")

    def test_collapses_newlines(self):
        result = strip_markdown("A\n\n\n\n\nB")
        assert result == "A\n\nB"


class TestGenerateVoice:
    @pytest.mark.asyncio
    @patch("src.tts.edge_tts.Communicate")
    async def test_calls_edge_tts_with_stripped_text(self, mock_comm_class, tmp_path):
        mock_comm = MagicMock()
        mock_comm.save = AsyncMock()
        mock_comm_class.return_value = mock_comm
        output_path = tmp_path / "voice.mp3"

        await generate_voice("Hello world", output_path)

        mock_comm_class.assert_called_once_with("Hello world", "en-US-GuyNeural")

    @pytest.mark.asyncio
    @patch("src.tts.edge_tts.Communicate")
    async def test_strips_markdown_before_tts(self, mock_comm_class, tmp_path):
        mock_comm = MagicMock()
        mock_comm.save = AsyncMock()
        mock_comm_class.return_value = mock_comm
        output_path = tmp_path / "voice.mp3"

        await generate_voice("## Title\n**bold** text", output_path)

        # Should have stripped markdown
        call_text = mock_comm_class.call_args[0][0]
        assert "##" not in call_text
        assert "**" not in call_text
        assert "Title" in call_text
        assert "bold" in call_text

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
