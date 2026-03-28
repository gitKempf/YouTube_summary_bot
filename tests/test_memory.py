import pytest
from unittest.mock import patch, MagicMock

from src.memory import MemoryManager, MemoryEntry, MemoryError


@pytest.fixture
def mem_config():
    config = MagicMock()
    config.memory_enabled = True
    config.anthropic_api_key = "fake-ant-key"
    config.claude_model = "claude-sonnet-4-6"
    config.voyage_api_key = "fake-voyage-key"
    config.pg_host = "localhost"
    config.pg_port = 5432
    config.pg_dbname = "mem0"
    config.neo4j_url = "bolt://localhost:7687"
    config.neo4j_username = "neo4j"
    config.neo4j_password = "password"
    return config


class TestMemoryManager:
    @patch("src.memory.Memory.from_config")
    def test_init_creates_mem0_client(self, mock_from_config, mem_config):
        MemoryManager(mem_config)
        mock_from_config.assert_called_once()
        call_kwargs = mock_from_config.call_args[1]
        cfg = call_kwargs["config_dict"]
        assert cfg["vector_store"]["provider"] == "pgvector"
        assert cfg["graph_store"]["provider"] == "neo4j"
        assert cfg["embedder"]["provider"] == "voyageai"

    @pytest.mark.asyncio
    @patch("src.memory.Memory.from_config")
    async def test_search_returns_entries(self, mock_from_config, mem_config):
        mock_mem = MagicMock()
        mock_from_config.return_value = mock_mem
        mock_mem.search.return_value = {
            "results": [
                {"id": "m1", "memory": "Topic A", "score": 0.9},
                {"id": "m2", "memory": "Topic B", "score": 0.7},
            ]
        }
        mgr = MemoryManager(mem_config)
        results = await mgr.search("query", user_id="tg_123")
        assert len(results) == 2
        assert results[0] == MemoryEntry(id="m1", text="Topic A", score=0.9)
        assert results[1] == MemoryEntry(id="m2", text="Topic B", score=0.7)

    @pytest.mark.asyncio
    @patch("src.memory.Memory.from_config")
    async def test_search_empty_results(self, mock_from_config, mem_config):
        mock_mem = MagicMock()
        mock_from_config.return_value = mock_mem
        mock_mem.search.return_value = {"results": []}
        mgr = MemoryManager(mem_config)
        results = await mgr.search("query", user_id="tg_123")
        assert results == []

    @pytest.mark.asyncio
    @patch("src.memory.Memory.from_config")
    async def test_add_stores_text(self, mock_from_config, mem_config):
        mock_mem = MagicMock()
        mock_from_config.return_value = mock_mem
        mgr = MemoryManager(mem_config)
        await mgr.add("New fact about Python", user_id="tg_123")
        mock_mem.add.assert_called_once()
        call_kwargs = mock_mem.add.call_args[1]
        assert call_kwargs["user_id"] == "tg_123"

    @pytest.mark.asyncio
    @patch("src.memory.Memory.from_config")
    async def test_get_all_returns_entries(self, mock_from_config, mem_config):
        mock_mem = MagicMock()
        mock_from_config.return_value = mock_mem
        mock_mem.get_all.return_value = {
            "results": [{"id": "m1", "memory": "Fact 1"}]
        }
        mgr = MemoryManager(mem_config)
        results = await mgr.get_all(user_id="tg_123")
        assert len(results) == 1
        assert results[0].text == "Fact 1"

    @pytest.mark.asyncio
    @patch("src.memory.Memory.from_config")
    async def test_search_handles_error(self, mock_from_config, mem_config):
        mock_mem = MagicMock()
        mock_from_config.return_value = mock_mem
        mock_mem.search.side_effect = Exception("DB connection failed")
        mgr = MemoryManager(mem_config)
        results = await mgr.search("query", user_id="tg_123")
        assert results == []

    @pytest.mark.asyncio
    @patch("src.memory.Memory.from_config")
    async def test_add_handles_error(self, mock_from_config, mem_config):
        mock_mem = MagicMock()
        mock_from_config.return_value = mock_mem
        mock_mem.add.side_effect = Exception("DB write failed")
        mgr = MemoryManager(mem_config)
        await mgr.add("text", user_id="tg_123")  # Should not raise
