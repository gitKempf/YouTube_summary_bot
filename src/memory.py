import asyncio
import logging
from dataclasses import dataclass
from typing import List, Optional

from mem0 import Memory

from src.config import Config

logger = logging.getLogger(__name__)


def _patch_mem0_anthropic_tool_choice():
    """Fix Mem0 bug: Anthropic API expects tool_choice as dict, not string."""
    try:
        import mem0.llms.anthropic as mod
        original = mod.AnthropicLLM.generate_response

        def patched(self, messages, response_format=None, tools=None, tool_choice="auto"):
            if isinstance(tool_choice, str):
                tool_choice = {"type": tool_choice}
            return original(self, messages, response_format=response_format,
                            tools=tools, tool_choice=tool_choice)

        mod.AnthropicLLM.generate_response = patched
        logger.info("Patched Mem0 Anthropic tool_choice format")
    except Exception as e:
        logger.warning(f"Could not patch Mem0 Anthropic: {e}")


_patch_mem0_anthropic_tool_choice()


class MemoryError(Exception):
    pass


@dataclass
class MemoryEntry:
    id: str
    text: str
    score: float = 0.0
    metadata: Optional[dict] = None


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
            "vector_store": {
                "provider": "qdrant",
                "config": {
                    "collection_name": "youtube_bot_memories",
                    "embedding_model_dims": 384,
                    "host": config.qdrant_host,
                    "port": config.qdrant_port,
                },
            },
            "version": "v1.1",
        }

        # Neo4j graph store: Mem0's graph module uses OpenAI function-calling
        # format which is incompatible with Anthropic. Graph entity extraction
        # is handled by our fact_checker.py instead. Neo4j integration requires
        # switching Mem0's graph LLM to OpenAI or waiting for Mem0 to fix
        # Anthropic tool format support.
        # TODO: Enable when Mem0 supports Anthropic tools for graph extraction
        logger.info("Using vector-only memory (graph via fact_checker.py)")

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
                    metadata=r.get("metadata"),
                )
                for r in result.get("results", [])
            ]
        except Exception as e:
            logger.warning(f"Memory search failed: {e}")
            return []

    async def add(self, text: str, user_id: str, metadata: dict = None) -> None:
        try:
            kwargs = {"user_id": user_id, "infer": True}
            if metadata:
                kwargs["metadata"] = metadata
            await asyncio.to_thread(self._memory.add, text, **kwargs)
        except Exception as e:
            logger.warning(f"Memory add failed: {e}")

    async def get_all(self, user_id: str) -> List[MemoryEntry]:
        try:
            result = await asyncio.to_thread(self._memory.get_all, user_id=user_id)
            return [
                MemoryEntry(
                    id=r.get("id", ""),
                    text=r.get("memory", ""),
                    metadata=r.get("metadata"),
                )
                for r in result.get("results", [])
            ]
        except Exception as e:
            logger.warning(f"Memory get_all failed: {e}")
            return []
