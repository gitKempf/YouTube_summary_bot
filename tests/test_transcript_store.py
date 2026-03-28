"""Tests for transcript storage in separate Qdrant collection."""
import pytest
from unittest.mock import patch, MagicMock, AsyncMock

from src.memory import MemoryManager


@pytest.fixture
def mem_mgr():
    with patch("src.memory.Memory.from_config") as mock_mem, \
         patch("src.memory.QdrantClient") as mock_qdrant_cls:
        mock_qdrant = MagicMock()
        mock_qdrant.get_collections.return_value = MagicMock(collections=[])
        mock_qdrant_cls.return_value = mock_qdrant

        config = MagicMock(qdrant_host="localhost", qdrant_port=6333,
                           neo4j_url="bolt://localhost:7687", neo4j_username="neo4j",
                           neo4j_password="p", claude_model="m", anthropic_api_key="k",
                           openai_api_key="")
        mgr = MemoryManager(config)
        yield mgr, mock_qdrant


class TestTranscriptStorage:
    @pytest.mark.asyncio
    async def test_store_transcript_saves_to_qdrant(self, mem_mgr):
        mgr, mock_qdrant = mem_mgr
        mock_qdrant.scroll.return_value = ([], None)

        await mgr.store_transcript(
            video_id="abc123", user_id="tg_42",
            transcript="Full transcript about AI agents...", language_code="en",
        )

        mock_qdrant.upsert.assert_called_once()
        call_kwargs = mock_qdrant.upsert.call_args[1]
        assert call_kwargs["collection_name"] == "youtube_bot_transcripts"
        payload = call_kwargs["points"][0].payload
        assert payload["video_id"] == "abc123"
        assert payload["transcript"] == "Full transcript about AI agents..."

    @pytest.mark.asyncio
    async def test_get_transcript_by_video_id(self, mem_mgr):
        mgr, mock_qdrant = mem_mgr
        mock_qdrant.scroll.return_value = ([
            MagicMock(payload={
                "video_id": "abc123", "user_id": "tg_42",
                "transcript": "Full transcript text...", "language_code": "en",
            })
        ], None)

        result = await mgr.get_transcript(video_id="abc123", user_id="tg_42")

        assert result is not None
        assert result["video_id"] == "abc123"
        assert "Full transcript" in result["transcript"]

    @pytest.mark.asyncio
    async def test_get_transcript_returns_none_if_missing(self, mem_mgr):
        mgr, mock_qdrant = mem_mgr
        mock_qdrant.scroll.return_value = ([], None)

        result = await mgr.get_transcript(video_id="nonexistent", user_id="tg_42")
        assert result is None

    @pytest.mark.asyncio
    async def test_skip_duplicate_transcript(self, mem_mgr):
        mgr, mock_qdrant = mem_mgr
        mock_qdrant.scroll.return_value = ([
            MagicMock(payload={"video_id": "abc123"})
        ], None)

        await mgr.store_transcript(
            video_id="abc123", user_id="tg_42",
            transcript="text", language_code="en",
        )

        mock_qdrant.upsert.assert_not_called()
