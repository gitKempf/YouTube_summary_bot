import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock, AsyncMock

from src.tts import generate_voice, generate_voice_chunked, TTSError, get_voice_for_language, convert_to_ogg, strip_markdown, split_text_chunks


def _mock_stream(*audio_chunks):
    """Create a mock edge-tts Communicate that yields audio chunks via stream()."""
    mock_comm = MagicMock()

    async def _stream():
        for data in audio_chunks:
            yield {"type": "audio", "data": data}

    mock_comm.stream = _stream
    return mock_comm


class TestGetVoiceForLanguage:
    def test_english_2letter(self):
        assert get_voice_for_language("en") == "en-US-RogerNeural"

    def test_english_3letter(self):
        assert get_voice_for_language("eng") == "en-US-RogerNeural"

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
        assert get_voice_for_language("xx") == "en-US-RogerNeural"


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
    async def test_calls_edge_tts_with_correct_params(self, mock_comm_class, tmp_path):
        mock_comm_class.return_value = _mock_stream(b"audio_data")
        output_path = tmp_path / "voice.mp3"

        await generate_voice("Hello world", output_path)

        mock_comm_class.assert_called_once_with("Hello world", "en-US-RogerNeural", rate="+10%")

    @pytest.mark.asyncio
    @patch("src.tts.edge_tts.Communicate")
    async def test_strips_markdown_before_tts(self, mock_comm_class, tmp_path):
        mock_comm_class.return_value = _mock_stream(b"audio_data")
        output_path = tmp_path / "voice.mp3"

        await generate_voice("## Title\n**bold** text", output_path)

        call_text = mock_comm_class.call_args[0][0]
        assert "##" not in call_text
        assert "**" not in call_text
        assert "Title" in call_text

    @pytest.mark.asyncio
    @patch("src.tts.edge_tts.Communicate")
    async def test_returns_output_path(self, mock_comm_class, tmp_path):
        mock_comm_class.return_value = _mock_stream(b"audio_data")
        output_path = tmp_path / "voice.mp3"

        result = await generate_voice("Hello world", output_path)

        assert result == output_path

    @pytest.mark.asyncio
    @patch("src.tts.edge_tts.Communicate")
    async def test_writes_audio_to_file(self, mock_comm_class, tmp_path):
        mock_comm_class.return_value = _mock_stream(b"chunk1", b"chunk2")
        output_path = tmp_path / "voice.mp3"

        await generate_voice("Hello", output_path)

        assert output_path.read_bytes() == b"chunk1chunk2"

    @pytest.mark.asyncio
    @patch("src.tts.edge_tts.Communicate")
    async def test_calls_progress_callback(self, mock_comm_class, tmp_path):
        mock_comm_class.return_value = _mock_stream(b"a" * 1000, b"b" * 1000)
        output_path = tmp_path / "voice.mp3"
        progress_calls = []

        async def on_prog(frac, total):
            progress_calls.append(frac)

        await generate_voice("Hi", output_path, on_progress=on_prog)

        assert len(progress_calls) > 0

    @pytest.mark.asyncio
    @patch("src.tts.edge_tts.Communicate")
    async def test_with_custom_voice(self, mock_comm_class, tmp_path):
        mock_comm_class.return_value = _mock_stream(b"data")
        output_path = tmp_path / "voice.mp3"

        await generate_voice("Hello", output_path, voice="ru-RU-DmitryNeural")

        mock_comm_class.assert_called_once_with("Hello", "ru-RU-DmitryNeural", rate="+10%")

    @pytest.mark.asyncio
    async def test_raises_on_empty_text(self, tmp_path):
        with pytest.raises(ValueError, match="Cannot generate voice from empty text"):
            await generate_voice("", tmp_path / "voice.mp3")

    @pytest.mark.asyncio
    @patch("src.tts.edge_tts.Communicate")
    async def test_raises_on_tts_error(self, mock_comm_class, tmp_path):
        mock_comm = MagicMock()

        async def _fail_stream():
            raise Exception("TTS failure")
            yield  # make it an async generator

        mock_comm.stream = _fail_stream
        mock_comm_class.return_value = mock_comm

        with pytest.raises(TTSError, match="Edge-TTS error"):
            await generate_voice("Hello", tmp_path / "voice.mp3")


class TestSplitTextChunks:
    def test_short_text_single_chunk(self):
        result = split_text_chunks("Hello world", max_chars=100)
        assert result == ["Hello world"]

    def test_splits_at_paragraph_boundary(self):
        text = "Paragraph one.\n\nParagraph two.\n\nParagraph three."
        result = split_text_chunks(text, max_chars=30)
        assert len(result) >= 2
        assert "Paragraph one." in result[0]

    def test_long_paragraph_splits_by_sentences(self):
        text = "First sentence. Second sentence. Third sentence. Fourth sentence."
        result = split_text_chunks(text, max_chars=40)
        assert len(result) >= 2
        for chunk in result:
            assert len(chunk) <= 40 or "." in chunk

    def test_preserves_all_content(self):
        text = "A\n\nB\n\nC\n\nD\n\nE"
        result = split_text_chunks(text, max_chars=5)
        combined = " ".join(result)
        for letter in "ABCDE":
            assert letter in combined


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
