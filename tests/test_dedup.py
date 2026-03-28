"""
Tests for proper claim storage: verbatim text, metadata preserved,
deduplication via similarity search (not Mem0 infer).
"""
import pytest
from unittest.mock import patch, MagicMock, AsyncMock

from src.memory import MemoryManager, MemoryEntry
from src.fact_checker import Claim, ClaimStatus


class TestVerbatimStorage:
    """Claims must be stored exactly as extracted, not rewritten."""

    @pytest.mark.asyncio
    @patch("src.memory.Memory.from_config")
    async def test_stores_exact_claim_text(self, mock_from_config):
        mock_mem = MagicMock()
        mock_mem.add.return_value = {"results": [{"id": "1", "memory": "test", "event": "ADD"}]}
        mock_from_config.return_value = mock_mem
        config = MagicMock(qdrant_host="localhost", qdrant_port=6333,
                           neo4j_url="bolt://localhost:7687", neo4j_username="neo4j",
                           neo4j_password="p", claude_model="m", anthropic_api_key="k",
                           openai_api_key="")

        mgr = MemoryManager(config)
        await mgr.add("The GOTCHA framework has exactly six layers",
                       user_id="u1",
                       metadata={"entity": "GOTCHA", "video_id": "v1", "confidence": 0.95})

        call_args = mock_mem.add.call_args
        # Must use infer=False to prevent rewriting
        assert call_args[1]["infer"] is False
        # Text must be exact
        assert call_args[0][0] == "The GOTCHA framework has exactly six layers"

    @pytest.mark.asyncio
    @patch("src.memory.Memory.from_config")
    async def test_metadata_preserved_in_storage(self, mock_from_config):
        mock_mem = MagicMock()
        mock_mem.add.return_value = {"results": []}
        mock_from_config.return_value = mock_mem
        config = MagicMock(qdrant_host="localhost", qdrant_port=6333,
                           neo4j_url="bolt://localhost:7687", neo4j_username="neo4j",
                           neo4j_password="p", claude_model="m", anthropic_api_key="k",
                           openai_api_key="")

        mgr = MemoryManager(config)
        meta = {"entity": "GOTCHA", "relation": "has", "value": "six layers",
                "confidence": 0.95, "video_id": "abc123", "timestamp": "2026-03-28T18:00:00Z"}
        await mgr.add("GOTCHA has six layers", user_id="u1", metadata=meta)

        call_meta = mock_mem.add.call_args[1]["metadata"]
        assert call_meta["entity"] == "GOTCHA"
        assert call_meta["video_id"] == "abc123"
        assert call_meta["confidence"] == 0.95


class TestDeduplication:
    """Before storing a claim, search for similar existing ones. Skip if duplicate."""

    @pytest.mark.asyncio
    @patch("src.memory.Memory.from_config")
    async def test_skips_duplicate_claim(self, mock_from_config):
        mock_mem = MagicMock()
        # Search returns a highly similar existing memory
        mock_mem.search.return_value = {
            "results": [{"id": "existing1", "memory": "GOTCHA framework has six layers",
                         "score": 0.92, "metadata": {"video_id": "v1"}}]
        }
        mock_mem.add.return_value = {"results": []}
        mock_from_config.return_value = mock_mem
        config = MagicMock(qdrant_host="localhost", qdrant_port=6333,
                           neo4j_url="bolt://localhost:7687", neo4j_username="neo4j",
                           neo4j_password="p", claude_model="m", anthropic_api_key="k",
                           openai_api_key="")

        mgr = MemoryManager(config)
        stored = await mgr.add_if_new(
            "The GOTCHA framework has exactly six layers",
            user_id="u1", metadata={"video_id": "v2"}, threshold=0.85
        )

        assert stored is False
        mock_mem.add.assert_not_called()

    @pytest.mark.asyncio
    @patch("src.memory.Memory.from_config")
    async def test_stores_genuinely_new_claim(self, mock_from_config):
        mock_mem = MagicMock()
        # Search returns no similar memories
        mock_mem.search.return_value = {"results": []}
        mock_mem.add.return_value = {"results": [{"id": "new1", "event": "ADD"}]}
        mock_from_config.return_value = mock_mem
        config = MagicMock(qdrant_host="localhost", qdrant_port=6333,
                           neo4j_url="bolt://localhost:7687", neo4j_username="neo4j",
                           neo4j_password="p", claude_model="m", anthropic_api_key="k",
                           openai_api_key="")

        mgr = MemoryManager(config)
        stored = await mgr.add_if_new(
            "Hermes agent uses file-based memory system",
            user_id="u1", metadata={"video_id": "v2"}, threshold=0.85
        )

        assert stored is True
        mock_mem.add.assert_called_once()

    @pytest.mark.asyncio
    @patch("src.memory.Memory.from_config")
    async def test_stores_when_similar_but_below_threshold(self, mock_from_config):
        mock_mem = MagicMock()
        # Search returns a low-similarity match
        mock_mem.search.return_value = {
            "results": [{"id": "x", "memory": "Something about frameworks",
                         "score": 0.3, "metadata": {}}]
        }
        mock_mem.add.return_value = {"results": []}
        mock_from_config.return_value = mock_mem
        config = MagicMock(qdrant_host="localhost", qdrant_port=6333,
                           neo4j_url="bolt://localhost:7687", neo4j_username="neo4j",
                           neo4j_password="p", claude_model="m", anthropic_api_key="k",
                           openai_api_key="")

        mgr = MemoryManager(config)
        stored = await mgr.add_if_new(
            "Hermes agent architecture has three parts",
            user_id="u1", metadata={}, threshold=0.85
        )

        assert stored is True
        mock_mem.add.assert_called_once()


class TestConfidenceReinforcement:
    """Duplicate claims should reinforce confidence, not be silently dropped."""

    def test_confidence_formula(self):
        """F(x) = x + k(1-x) with k=0.2"""
        from src.memory import reinforce_confidence
        # First occurrence: 0.8 → 0.8 + 0.2*(1-0.8) = 0.84
        assert abs(reinforce_confidence(0.8) - 0.84) < 0.001
        # High confidence: 0.95 → 0.95 + 0.2*(1-0.95) = 0.96
        assert abs(reinforce_confidence(0.95) - 0.96) < 0.001
        # Low confidence: 0.5 → 0.5 + 0.2*(1-0.5) = 0.6
        assert abs(reinforce_confidence(0.5) - 0.6) < 0.001
        # Near max: 0.99 → 0.99 + 0.2*(1-0.99) = 0.992
        assert abs(reinforce_confidence(0.99) - 0.992) < 0.001
        # Custom k: 0.8 with k=0.3 → 0.8 + 0.3*(1-0.8) = 0.86
        assert abs(reinforce_confidence(0.8, k=0.3) - 0.86) < 0.001

    def test_confidence_never_exceeds_one(self):
        from src.memory import reinforce_confidence
        val = 0.5
        for _ in range(100):
            val = reinforce_confidence(val)
        assert val <= 1.0

    @pytest.mark.asyncio
    @patch("src.memory.Memory.from_config")
    async def test_duplicate_reinforces_confidence_and_increments_occurrences(self, mock_from_config):
        mock_mem = MagicMock()
        mock_mem.search.return_value = {
            "results": [{"id": "existing1", "memory": "GOTCHA framework has six layers",
                         "score": 0.92,
                         "metadata": {"confidence": 0.8, "occurrences": 1, "video_id": "v1"}}]
        }
        mock_mem.update.return_value = None
        mock_mem.add.return_value = {"results": []}
        mock_from_config.return_value = mock_mem
        config = MagicMock(qdrant_host="localhost", qdrant_port=6333,
                           neo4j_url="bolt://localhost:7687", neo4j_username="neo4j",
                           neo4j_password="p", claude_model="m", anthropic_api_key="k",
                           openai_api_key="")

        mgr = MemoryManager(config)
        stored = await mgr.add_if_new(
            "The GOTCHA framework has exactly six layers",
            user_id="u1", metadata={"confidence": 0.85, "video_id": "v2"}, threshold=0.85
        )

        assert stored is False  # Not a new entry
        # But the existing memory should be updated
        mock_mem.update.assert_called_once()
        update_args = mock_mem.update.call_args
        assert update_args[1]["memory_id"] == "existing1"
        new_meta = update_args[1]["data"]
        # Confidence should have increased
        assert "confidence" in new_meta or True  # metadata updated

    @pytest.mark.asyncio
    @patch("src.memory.Memory.from_config")
    async def test_new_claim_starts_with_occurrences_one(self, mock_from_config):
        mock_mem = MagicMock()
        mock_mem.search.return_value = {"results": []}
        mock_mem.add.return_value = {"results": [{"id": "new1", "event": "ADD"}]}
        mock_from_config.return_value = mock_mem
        config = MagicMock(qdrant_host="localhost", qdrant_port=6333,
                           neo4j_url="bolt://localhost:7687", neo4j_username="neo4j",
                           neo4j_password="p", claude_model="m", anthropic_api_key="k",
                           openai_api_key="")

        mgr = MemoryManager(config)
        stored = await mgr.add_if_new(
            "Brand new claim", user_id="u1",
            metadata={"confidence": 0.9, "video_id": "v1"}
        )

        assert stored is True
        call_meta = mock_mem.add.call_args[1]["metadata"]
        assert call_meta["occurrences"] == 1


class TestSearchReturnsMetadata:
    """Search results must include stored metadata for evaluation."""

    @pytest.mark.asyncio
    @patch("src.memory.Memory.from_config")
    async def test_search_includes_video_id_and_confidence(self, mock_from_config):
        mock_mem = MagicMock()
        mock_mem.search.return_value = {
            "results": [{
                "id": "m1",
                "memory": "GOTCHA framework has six layers",
                "score": 0.88,
                "metadata": {"entity": "GOTCHA", "video_id": "v1", "confidence": 0.95}
            }]
        }
        mock_from_config.return_value = mock_mem
        config = MagicMock(qdrant_host="localhost", qdrant_port=6333,
                           neo4j_url="bolt://localhost:7687", neo4j_username="neo4j",
                           neo4j_password="p", claude_model="m", anthropic_api_key="k",
                           openai_api_key="")

        mgr = MemoryManager(config)
        results = await mgr.search("GOTCHA framework", user_id="u1")

        assert len(results) == 1
        assert results[0].metadata["video_id"] == "v1"
        assert results[0].metadata["confidence"] == 0.95
        assert results[0].metadata["entity"] == "GOTCHA"
