import pytest
from unittest.mock import patch, MagicMock, AsyncMock

from src.summarizer import summarize_text, SummarizationError, SYSTEM_PROMPT


class TestSummarizeText:
    @pytest.mark.asyncio
    @patch("src.summarizer.AsyncAnthropic")
    async def test_calls_anthropic_with_correct_params(self, mock_ant_class):
        mock_client = MagicMock()
        mock_ant_class.return_value = mock_client
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="Summary")]
        mock_client.messages.create = AsyncMock(return_value=mock_response)

        await summarize_text("Some transcript", api_key="fake-key")

        mock_ant_class.assert_called_once_with(api_key="fake-key")
        call_kwargs = mock_client.messages.create.call_args[1]
        assert call_kwargs["model"] == "claude-sonnet-4-6"
        assert call_kwargs["max_tokens"] == 4096
        assert "Some transcript" in call_kwargs["messages"][0]["content"]

    @pytest.mark.asyncio
    @patch("src.summarizer.AsyncAnthropic")
    async def test_returns_summary_string(self, mock_ant_class):
        mock_client = MagicMock()
        mock_ant_class.return_value = mock_client
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="This is the summary.")]
        mock_client.messages.create = AsyncMock(return_value=mock_response)

        result = await summarize_text("Some transcript", api_key="fake-key")

        assert result == "This is the summary."

    @pytest.mark.asyncio
    @patch("src.summarizer.AsyncAnthropic")
    async def test_uses_system_prompt(self, mock_ant_class):
        mock_client = MagicMock()
        mock_ant_class.return_value = mock_client
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="Summary")]
        mock_client.messages.create = AsyncMock(return_value=mock_response)

        await summarize_text("Some transcript", api_key="fake-key")

        call_kwargs = mock_client.messages.create.call_args[1]
        assert call_kwargs["system"] == SYSTEM_PROMPT
        assert "voiceover" in SYSTEM_PROMPT.lower() or "narrator" in SYSTEM_PROMPT.lower()
        assert "no bullet" in SYSTEM_PROMPT.lower() or "no markdown" in SYSTEM_PROMPT.lower()

    @pytest.mark.asyncio
    async def test_raises_on_empty_input(self):
        with pytest.raises(ValueError, match="Cannot summarize empty text"):
            await summarize_text("", api_key="fake-key")

    @pytest.mark.asyncio
    async def test_raises_on_whitespace_input(self):
        with pytest.raises(ValueError, match="Cannot summarize empty text"):
            await summarize_text("   ", api_key="fake-key")

    @pytest.mark.asyncio
    @patch("src.summarizer.AsyncAnthropic")
    async def test_raises_on_api_error(self, mock_ant_class):
        mock_client = MagicMock()
        mock_ant_class.return_value = mock_client
        mock_client.messages.create = AsyncMock(side_effect=Exception("API error"))

        with pytest.raises(SummarizationError, match="Anthropic API error"):
            await summarize_text("Some transcript", api_key="fake-key")

    @pytest.mark.asyncio
    @patch("src.summarizer.AsyncAnthropic")
    async def test_with_custom_model(self, mock_ant_class):
        mock_client = MagicMock()
        mock_ant_class.return_value = mock_client
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="Summary")]
        mock_client.messages.create = AsyncMock(return_value=mock_response)

        await summarize_text(
            "Some transcript", api_key="fake-key",
            model="claude-opus-4-6", max_tokens=2048
        )

        call_kwargs = mock_client.messages.create.call_args[1]
        assert call_kwargs["model"] == "claude-opus-4-6"
        assert call_kwargs["max_tokens"] == 2048

    @pytest.mark.asyncio
    @patch("src.summarizer.AsyncAnthropic")
    async def test_with_past_context_augments_prompt(self, mock_ant_class):
        mock_client = MagicMock()
        mock_ant_class.return_value = mock_client
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="Summary")]
        mock_client.messages.create = AsyncMock(return_value=mock_response)

        await summarize_text(
            "transcript", api_key="fake",
            past_context="User already knows about Python."
        )

        call_kwargs = mock_client.messages.create.call_args[1]
        system = call_kwargs["system"]
        assert "User already knows about Python." in system
        assert "narrator" in system.lower() or "voiceover" in system.lower()

    @pytest.mark.asyncio
    @patch("src.summarizer.AsyncAnthropic")
    async def test_without_past_context_uses_original(self, mock_ant_class):
        mock_client = MagicMock()
        mock_ant_class.return_value = mock_client
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="Summary")]
        mock_client.messages.create = AsyncMock(return_value=mock_response)

        await summarize_text("transcript", api_key="fake")

        call_kwargs = mock_client.messages.create.call_args[1]
        assert call_kwargs["system"] == SYSTEM_PROMPT

    @pytest.mark.asyncio
    @patch("src.summarizer.AsyncAnthropic")
    async def test_with_empty_context_uses_original(self, mock_ant_class):
        mock_client = MagicMock()
        mock_ant_class.return_value = mock_client
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="Summary")]
        mock_client.messages.create = AsyncMock(return_value=mock_response)

        await summarize_text("transcript", api_key="fake", past_context="")

        call_kwargs = mock_client.messages.create.call_args[1]
        assert call_kwargs["system"] == SYSTEM_PROMPT
