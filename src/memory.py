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
                    "embedding_dims": 384,
                },
            },
            "version": "v1.1",
        }

        # Qdrant: use server mode if available, otherwise local disk
        qdrant_config = {
            "collection_name": "youtube_bot_memories",
            "embedding_model_dims": 384,
        }
        try:
            import socket
            s = socket.socket()
            s.settimeout(2)
            s.connect(("localhost", 6333))
            s.close()
            qdrant_config["host"] = "localhost"
            qdrant_config["port"] = 6333
            logger.info("Using Qdrant server mode (concurrent-safe)")
        except Exception:
            qdrant_config["on_disk"] = True
            qdrant_config["path"] = "/tmp/mem0_qdrant"
            logger.info("Using Qdrant local mode (singleton required)")

        mem0_config["vector_store"] = {
            "provider": "qdrant",
            "config": qdrant_config,
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
            await asyncio.to_thread(
                self._memory.add, text, user_id=user_id, infer=False
            )
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
