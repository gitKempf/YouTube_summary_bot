import json
import pytest
from unittest.mock import patch, MagicMock, AsyncMock

from src.fact_checker import (
    extract_claims,
    classify_claims,
    build_context_prompt,
    Claim,
    ClassifiedClaim,
    ClaimStatus,
    FactCheckResult,
)
from src.memory import MemoryEntry


class TestExtractClaims:
    @pytest.mark.asyncio
    @patch("src.fact_checker.AsyncAnthropic")
    async def test_calls_claude_with_extraction_prompt(self, mock_ant_class):
        mock_client = MagicMock()
        mock_ant_class.return_value = mock_client
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text=json.dumps([
            {"text": "Python 3.12 adds pattern matching", "entity": "Python 3.12",
             "relation": "adds", "value": "pattern matching"}
        ]))]
        mock_client.messages.create = AsyncMock(return_value=mock_response)

        result = await extract_claims("Some transcript", api_key="fake")

        call_kwargs = mock_client.messages.create.call_args[1]
        system = call_kwargs["system"]
        assert "extract" in system.lower()
        assert "claim" in system.lower()
        assert len(result) == 1
        assert result[0].entity == "Python 3.12"
        assert result[0].relation == "adds"
        assert result[0].value == "pattern matching"

    @pytest.mark.asyncio
    async def test_empty_transcript_returns_empty(self):
        result = await extract_claims("", api_key="fake")
        assert result == []

    @pytest.mark.asyncio
    @patch("src.fact_checker.AsyncAnthropic")
    async def test_malformed_response_returns_empty(self, mock_ant_class):
        mock_client = MagicMock()
        mock_ant_class.return_value = mock_client
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="not valid json at all")]
        mock_client.messages.create = AsyncMock(return_value=mock_response)

        result = await extract_claims("transcript", api_key="fake")
        assert result == []


class TestClassifyClaims:
    @pytest.mark.asyncio
    async def test_no_memories_all_new(self):
        claims = [Claim(text="New fact", entity="X", relation="is", value="Y")]
        result = await classify_claims(claims, memories=[], api_key="fake")
        assert len(result.new_claims) == 1
        assert result.new_claims[0].status == ClaimStatus.NEW
        assert len(result.supported_claims) == 0

    @pytest.mark.asyncio
    @patch("src.fact_checker.AsyncAnthropic")
    async def test_supported_claim(self, mock_ant_class):
        mock_client = MagicMock()
        mock_ant_class.return_value = mock_client
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text=json.dumps([
            {"claim_text": "X is Y", "status": "supported", "matching_memory": "X is Y already known"}
        ]))]
        mock_client.messages.create = AsyncMock(return_value=mock_response)

        claims = [Claim(text="X is Y", entity="X", relation="is", value="Y")]
        memories = [MemoryEntry(id="m1", text="X is Y already known", score=0.9)]
        result = await classify_claims(claims, memories, api_key="fake")
        assert len(result.supported_claims) == 1
        assert result.supported_claims[0].status == ClaimStatus.SUPPORTED

    @pytest.mark.asyncio
    @patch("src.fact_checker.AsyncAnthropic")
    async def test_contradicted_claim(self, mock_ant_class):
        mock_client = MagicMock()
        mock_ant_class.return_value = mock_client
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text=json.dumps([
            {"claim_text": "X costs $10", "status": "contradicted",
             "matching_memory": "X costs $5"}
        ]))]
        mock_client.messages.create = AsyncMock(return_value=mock_response)

        claims = [Claim(text="X costs $10", entity="X", relation="costs", value="$10")]
        memories = [MemoryEntry(id="m1", text="X costs $5", score=0.8)]
        result = await classify_claims(claims, memories, api_key="fake")
        assert len(result.contradicted_claims) == 1
        assert result.contradicted_claims[0].status == ClaimStatus.CONTRADICTED


class TestBuildContextPrompt:
    def test_formats_correctly(self):
        result = FactCheckResult(
            new_claims=[
                ClassifiedClaim(Claim("New fact A", "A", "is", "new"), ClaimStatus.NEW),
                ClassifiedClaim(Claim("New fact B", "B", "is", "new"), ClaimStatus.NEW),
            ],
            supported_claims=[
                ClassifiedClaim(Claim("Known fact", "C", "is", "old"), ClaimStatus.SUPPORTED,
                                matching_memory="Already known"),
            ],
            contradicted_claims=[
                ClassifiedClaim(Claim("Changed fact", "D", "was", "X"), ClaimStatus.CONTRADICTED,
                                matching_memory="D was Y"),
            ],
            context_summary="",
        )
        prompt = build_context_prompt(result)
        assert "Known fact" in prompt
        assert "New fact A" in prompt
        assert "Changed fact" in prompt
        assert len(prompt) > 0

    def test_empty_result_returns_empty(self):
        result = FactCheckResult(
            new_claims=[], supported_claims=[],
            contradicted_claims=[], context_summary="",
        )
        prompt = build_context_prompt(result)
        assert prompt == ""
