import asyncio
import logging
from dataclasses import dataclass
from typing import List

from mem0 import Memory

from src.config import Config

logger = logging.getLogger(__name__)


class MemoryError(Exception):
    pass


@dataclass
class MemoryEntry:
    id: str
    text: str
    score: float = 0.0


class MemoryManager:
    def __init__(self, config: Config):
        mem0_config = {
            "llm": {
                "provider": "anthropic",
                "config": {
                    "model": config.claude_model,
                    "temperature": 0.1,
                    "max_tokens": 2000,
                    "api_key": config.anthropic_api_key,
                },
            },
            "embedder": {
                "provider": "huggingface",
                "config": {
                    "model": "sentence-transformers/all-MiniLM-L6-v2",
                    "embedding_model_dims": 384,
                },
            },
            "vector_store": {
                "provider": "pgvector",
                "config": {
                    "host": config.pg_host,
                    "port": config.pg_port,
                    "dbname": config.pg_dbname,
                    "user": "postgres",
                    "password": "postgres",
                    "embedding_model_dims": 384,
                },
            },
            "graph_store": {
                "provider": "neo4j",
                "config": {
                    "url": config.neo4j_url,
                    "username": config.neo4j_username,
                    "password": config.neo4j_password,
                },
            },
            "version": "v1.1",
        }
        self._memory = Memory.from_config(config_dict=mem0_config)

    async def search(
        self, query: str, user_id: str, limit: int = 10
    ) -> List[MemoryEntry]:
        try:
            result = await asyncio.to_thread(
                self._memory.search, query, user_id=user_id, limit=limit
            )
            return [
                MemoryEntry(
                    id=r.get("id", ""),
                    text=r.get("memory", ""),
                    score=r.get("score", 0.0),
                )
                for r in result.get("results", [])
            ]
        except Exception as e:
            logger.warning(f"Memory search failed: {e}")
            return []

    async def add(self, text: str, user_id: str) -> None:
        try:
            await asyncio.to_thread(self._memory.add, text, user_id=user_id)
        except Exception as e:
            logger.warning(f"Memory add failed: {e}")

    async def get_all(self, user_id: str) -> List[MemoryEntry]:
        try:
            result = await asyncio.to_thread(self._memory.get_all, user_id=user_id)
            return [
                MemoryEntry(
                    id=r.get("id", ""),
                    text=r.get("memory", ""),
                )
                for r in result.get("results", [])
            ]
        except Exception as e:
            logger.warning(f"Memory get_all failed: {e}")
            return []
