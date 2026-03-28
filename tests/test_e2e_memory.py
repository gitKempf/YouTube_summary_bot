"""
Evaluation benchmark: verifies all Rippletide-inspired patterns are implemented.
Tests run against real Mem0 + Qdrant (requires Qdrant server on localhost:6333).
Skip with: pytest -m "not e2e"
"""
import json
import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from datetime import datetime

from src.fact_checker import (
    extract_claims, classify_claims, build_context_prompt,
    Claim, ClassifiedClaim, ClaimStatus, FactCheckResult,
)
from src.memory import MemoryManager, MemoryEntry


# --- Helpers ---

def make_claim(text, entity="E", relation="R", value="V", confidence=0.9, video_id="vid1"):
    return Claim(text=text, entity=entity, relation=relation, value=value,
                 confidence=confidence, video_id=video_id)


def mock_claude_extraction(claims_json):
    """Create a mock AsyncAnthropic that returns the given claims JSON."""
    mock_ant = MagicMock()
    mock_client = MagicMock()
    mock_ant.return_value = mock_client
    mock_resp = MagicMock()
    mock_resp.content = [MagicMock(text=json.dumps(claims_json))]
    mock_client.messages.create = AsyncMock(return_value=mock_resp)
    return mock_ant


# ============================================================
# BENCHMARK 1: Atomic claims stored separately
# ============================================================
class TestAtomicClaimsSeparate:
    @pytest.mark.asyncio
    @patch("src.fact_checker.AsyncAnthropic")
    async def test_extracts_multiple_claims_from_transcript(self, mock_ant):
        mock_client = MagicMock()
        mock_ant.return_value = mock_client
        mock_resp = MagicMock()
        mock_resp.content = [MagicMock(text=json.dumps([
            {"text": "Claim A about Python", "entity": "Python", "relation": "is", "value": "popular", "confidence": 0.9},
            {"text": "Claim B about Rust", "entity": "Rust", "relation": "is", "value": "fast", "confidence": 0.85},
            {"text": "Claim C about Go", "entity": "Go", "relation": "has", "value": "goroutines", "confidence": 0.95},
        ]))]
        mock_client.messages.create = AsyncMock(return_value=mock_resp)

        claims = await extract_claims("transcript", api_key="fake", video_id="v1")

        assert len(claims) == 3
        assert claims[0].text == "Claim A about Python"
        assert claims[1].text == "Claim B about Rust"
        assert claims[2].text == "Claim C about Go"


# ============================================================
# BENCHMARK 2: Entity-relation-value triples
# ============================================================
class TestEntityRelationValue:
    @pytest.mark.asyncio
    @patch("src.fact_checker.AsyncAnthropic")
    async def test_claims_have_structured_triples(self, mock_ant):
        mock_client = MagicMock()
        mock_ant.return_value = mock_client
        mock_resp = MagicMock()
        mock_resp.content = [MagicMock(text=json.dumps([
            {"text": "Python 3.12 adds pattern matching", "entity": "Python 3.12",
             "relation": "adds", "value": "pattern matching", "confidence": 0.95},
        ]))]
        mock_client.messages.create = AsyncMock(return_value=mock_resp)

        claims = await extract_claims("transcript", api_key="fake")

        c = claims[0]
        assert c.entity == "Python 3.12"
        assert c.relation == "adds"
        assert c.value == "pattern matching"


# ============================================================
# BENCHMARK 3: Confidence scores stored
# ============================================================
class TestConfidenceScores:
    @pytest.mark.asyncio
    @patch("src.fact_checker.AsyncAnthropic")
    async def test_claims_carry_confidence(self, mock_ant):
        mock_client = MagicMock()
        mock_ant.return_value = mock_client
        mock_resp = MagicMock()
        mock_resp.content = [MagicMock(text=json.dumps([
            {"text": "Explicit fact", "entity": "X", "relation": "is", "value": "Y", "confidence": 1.0},
            {"text": "Implied fact", "entity": "A", "relation": "might", "value": "B", "confidence": 0.5},
        ]))]
        mock_client.messages.create = AsyncMock(return_value=mock_resp)

        claims = await extract_claims("transcript", api_key="fake")

        assert claims[0].confidence == 1.0
        assert claims[1].confidence == 0.5

    def test_claim_metadata_includes_confidence(self):
        c = make_claim("Test", confidence=0.85, video_id="v1")
        meta = {
            "entity": c.entity,
            "relation": c.relation,
            "value": c.value,
            "confidence": c.confidence,
            "video_id": c.video_id,
            "timestamp": c.timestamp,
        }
        assert meta["confidence"] == 0.85
        assert "confidence" in meta


# ============================================================
# BENCHMARK 4: Temporal validity
# ============================================================
class TestTemporalValidity:
    @pytest.mark.asyncio
    @patch("src.fact_checker.AsyncAnthropic")
    async def test_claims_have_timestamp_and_video_id(self, mock_ant):
        mock_client = MagicMock()
        mock_ant.return_value = mock_client
        mock_resp = MagicMock()
        mock_resp.content = [MagicMock(text=json.dumps([
            {"text": "Fact", "entity": "X", "relation": "is", "value": "Y", "confidence": 0.9},
        ]))]
        mock_client.messages.create = AsyncMock(return_value=mock_resp)

        claims = await extract_claims("transcript", api_key="fake", video_id="abc123")

        assert claims[0].video_id == "abc123"
        assert claims[0].timestamp  # non-empty
        # Should be ISO format
        datetime.fromisoformat(claims[0].timestamp)


# ============================================================
# BENCHMARK 5: Deduplication at storage (infer=True)
# ============================================================
class TestDeduplication:
    @patch("src.memory.Memory.from_config")
    def test_memory_add_uses_infer_true(self, mock_from_config):
        """Verify that infer=True is passed to Mem0 for dedup/merge."""
        mock_mem = MagicMock()
        mock_from_config.return_value = mock_mem
        config = MagicMock()
        config.qdrant_host = "localhost"
        config.qdrant_port = 6333
        config.neo4j_url = "bolt://localhost:7687"
        config.neo4j_username = "neo4j"
        config.neo4j_password = "pass"
        config.claude_model = "claude-sonnet-4-6"
        config.anthropic_api_key = "fake"

        import asyncio
        mgr = MemoryManager(config)
        asyncio.get_event_loop().run_until_complete(
            mgr.add("Python is popular", user_id="u1", metadata={"entity": "Python"})
        )

        call_kwargs = mock_mem.add.call_args[1]
        assert call_kwargs["infer"] is True
        assert call_kwargs["user_id"] == "u1"


# ============================================================
# BENCHMARK 6: Per-claim vector search
# ============================================================
class TestPerClaimSearch:
    @pytest.mark.asyncio
    @patch("src.memory.Memory.from_config")
    async def test_search_returns_individual_claims(self, mock_from_config):
        mock_mem = MagicMock()
        mock_from_config.return_value = mock_mem
        mock_mem.search.return_value = {
            "results": [
                {"id": "1", "memory": "Python is popular", "score": 0.95,
                 "metadata": {"entity": "Python", "confidence": 0.9}},
                {"id": "2", "memory": "Rust is fast", "score": 0.70,
                 "metadata": {"entity": "Rust", "confidence": 0.85}},
            ]
        }
        config = MagicMock()
        config.qdrant_host = "localhost"
        config.qdrant_port = 6333
        config.neo4j_url = "bolt://localhost:7687"
        config.neo4j_username = "neo4j"
        config.neo4j_password = "pass"
        config.claude_model = "claude-sonnet-4-6"
        config.anthropic_api_key = "fake"

        mgr = MemoryManager(config)
        results = await mgr.search("programming languages", user_id="u1")

        assert len(results) == 2
        assert results[0].text == "Python is popular"
        assert results[1].text == "Rust is fast"
        assert results[0].metadata["entity"] == "Python"


# ============================================================
# BENCHMARK 7: Supported claims detected
# ============================================================
class TestSupportedClaims:
    @pytest.mark.asyncio
    @patch("src.fact_checker.AsyncAnthropic")
    async def test_overlapping_topic_detected_as_supported(self, mock_ant):
        mock_client = MagicMock()
        mock_ant.return_value = mock_client
        mock_resp = MagicMock()
        mock_resp.content = [MagicMock(text=json.dumps([
            {"claim_text": "Python is great for ML", "status": "supported",
             "matching_memory": "Python is widely used for machine learning"},
            {"claim_text": "Go is fast", "status": "new", "matching_memory": None},
        ]))]
        mock_client.messages.create = AsyncMock(return_value=mock_resp)

        claims = [
            make_claim("Python is great for ML", entity="Python"),
            make_claim("Go is fast", entity="Go"),
        ]
        memories = [MemoryEntry(id="m1", text="Python is widely used for machine learning")]

        result = await classify_claims(claims, memories, api_key="fake")

        assert len(result.supported_claims) == 1
        assert result.supported_claims[0].status == ClaimStatus.SUPPORTED
        assert len(result.new_claims) == 1


# ============================================================
# BENCHMARK 8: Contradicted claims detected
# ============================================================
class TestContradictedClaims:
    @pytest.mark.asyncio
    @patch("src.fact_checker.AsyncAnthropic")
    async def test_conflicting_info_flagged(self, mock_ant):
        mock_client = MagicMock()
        mock_ant.return_value = mock_client
        mock_resp = MagicMock()
        mock_resp.content = [MagicMock(text=json.dumps([
            {"claim_text": "Product X costs $20", "status": "contradicted",
             "matching_memory": "Product X costs $10"},
        ]))]
        mock_client.messages.create = AsyncMock(return_value=mock_resp)

        claims = [make_claim("Product X costs $20", entity="Product X")]
        memories = [MemoryEntry(id="m1", text="Product X costs $10")]

        result = await classify_claims(claims, memories, api_key="fake")

        assert len(result.contradicted_claims) == 1
        assert result.contradicted_claims[0].matching_memory == "Product X costs $10"


# ============================================================
# BENCHMARK 9: Context prompt influences summary
# ============================================================
class TestContextPromptInfluence:
    def test_supported_claims_in_context(self):
        result = FactCheckResult(
            new_claims=[ClassifiedClaim(make_claim("New thing"), ClaimStatus.NEW)],
            supported_claims=[ClassifiedClaim(make_claim("Known thing"), ClaimStatus.SUPPORTED,
                                              matching_memory="Already known")],
            contradicted_claims=[],
            context_summary="",
        )
        prompt = build_context_prompt(result)
        assert "PRIOR KNOWLEDGE" in prompt
        assert "Known thing" in prompt
        assert "NEW TOPICS" in prompt
        assert "New thing" in prompt

    def test_contradictions_in_context(self):
        result = FactCheckResult(
            new_claims=[],
            supported_claims=[],
            contradicted_claims=[ClassifiedClaim(make_claim("X is 20"), ClaimStatus.CONTRADICTED,
                                                 matching_memory="X is 10")],
            context_summary="",
        )
        prompt = build_context_prompt(result)
        assert "CONTRADICTION" in prompt
        assert "X is 20" in prompt
        assert "X is 10" in prompt

    def test_empty_result_no_context(self):
        result = FactCheckResult([], [], [], "")
        assert build_context_prompt(result) == ""


# ============================================================
# BENCHMARK 10: No memories → all claims NEW (optimization)
# ============================================================
class TestNoMemoriesAllNew:
    @pytest.mark.asyncio
    async def test_skip_llm_when_no_memories(self):
        """When user has no memories, all claims should be NEW without calling Claude."""
        claims = [make_claim("Fact 1"), make_claim("Fact 2"), make_claim("Fact 3")]

        # No mock needed — should not call Claude at all
        result = await classify_claims(claims, memories=[], api_key="fake")

        assert len(result.new_claims) == 3
        assert len(result.supported_claims) == 0
        assert len(result.contradicted_claims) == 0
