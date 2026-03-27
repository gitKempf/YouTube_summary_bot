import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock, mock_open

from src.transcriber import transcribe_audio, TranscriptionError, TranscriptionResult


class TestTranscribeAudio:
    @patch("src.transcriber.ElevenLabs")
    def test_calls_elevenlabs_with_correct_params(self, mock_el_class, tmp_path):
        audio_file = tmp_path / "test.mp3"
        audio_file.write_bytes(b"fake audio data")

        mock_client = MagicMock()
        mock_el_class.return_value = mock_client
        mock_response = MagicMock()
        mock_response.text = "Hello world"
        mock_response.language_code = "en"
        mock_client.speech_to_text.convert.return_value = mock_response

        transcribe_audio(audio_file, api_key="fake-key")

        mock_el_class.assert_called_once_with(api_key="fake-key")
        call_kwargs = mock_client.speech_to_text.convert.call_args[1]
        assert call_kwargs["model_id"] == "scribe_v2"

    @patch("src.transcriber.ElevenLabs")
    def test_returns_transcription_result(self, mock_el_class, tmp_path):
        audio_file = tmp_path / "test.mp3"
        audio_file.write_bytes(b"fake audio data")

        mock_client = MagicMock()
        mock_el_class.return_value = mock_client
        mock_response = MagicMock()
        mock_response.text = "This is a test transcription."
        mock_response.language_code = "en"
        mock_client.speech_to_text.convert.return_value = mock_response

        result = transcribe_audio(audio_file, api_key="fake-key")

        assert isinstance(result, TranscriptionResult)
        assert result.text == "This is a test transcription."
        assert result.language_code == "en"

    def test_raises_on_missing_file(self):
        with pytest.raises(FileNotFoundError):
            transcribe_audio(Path("/nonexistent/audio.mp3"), api_key="fake-key")

    @patch("src.transcriber.ElevenLabs")
    def test_raises_on_api_error(self, mock_el_class, tmp_path):
        audio_file = tmp_path / "test.mp3"
        audio_file.write_bytes(b"fake audio data")

        mock_client = MagicMock()
        mock_el_class.return_value = mock_client
        mock_client.speech_to_text.convert.side_effect = Exception("API failure")

        with pytest.raises(TranscriptionError, match="ElevenLabs API error"):
            transcribe_audio(audio_file, api_key="fake-key")

    @patch("src.transcriber.ElevenLabs")
    def test_raises_on_empty_result(self, mock_el_class, tmp_path):
        audio_file = tmp_path / "test.mp3"
        audio_file.write_bytes(b"fake audio data")

        mock_client = MagicMock()
        mock_el_class.return_value = mock_client
        mock_response = MagicMock()
        mock_response.text = ""
        mock_client.speech_to_text.convert.return_value = mock_response

        with pytest.raises(TranscriptionError, match="empty text"):
            transcribe_audio(audio_file, api_key="fake-key")

    @patch("src.transcriber.ElevenLabs")
    def test_returns_detected_language(self, mock_el_class, tmp_path):
        audio_file = tmp_path / "test.mp3"
        audio_file.write_bytes(b"fake audio data")

        mock_client = MagicMock()
        mock_el_class.return_value = mock_client
        mock_response = MagicMock()
        mock_response.text = "Bonjour le monde"
        mock_response.language_code = "fr"
        mock_client.speech_to_text.convert.return_value = mock_response

        result = transcribe_audio(audio_file, api_key="fake-key")

        assert result.language_code == "fr"
