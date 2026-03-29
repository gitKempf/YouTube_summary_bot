import asyncio
import base64
import hashlib
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Optional

from cryptography.fernet import Fernet, InvalidToken
from mem0 import Memory
from qdrant_client import QdrantClient, models

from src.config import Config

TRANSCRIPT_COLLECTION = "youtube_bot_transcripts"
USER_SETTINGS_COLLECTION = "youtube_bot_user_settings"
_ENCRYPTED_FIELDS = {"anthropic_api_key", "elevenlabs_api_key"}

logger = logging.getLogger(__name__)


def _get_fernet() -> Fernet:
    """Derive a Fernet key from SETTINGS_SECRET env var."""
    secret = os.environ.get("SETTINGS_SECRET", "")
    if not secret:
        raise RuntimeError("SETTINGS_SECRET env var is required for key encryption")
    # Derive a 32-byte key from the secret via SHA-256, then base64-encode for Fernet
    key = base64.urlsafe_b64encode(hashlib.sha256(secret.encode()).digest())
    return Fernet(key)


def _encrypt_settings(settings: Dict) -> Dict:
    """Encrypt sensitive fields in settings dict."""
    try:
        f = _get_fernet()
    except RuntimeError:
        return settings
    out = dict(settings)
    for field in _ENCRYPTED_FIELDS:
        val = out.get(field, "")
        if val and not val.startswith("gAAAAA"):  # already encrypted
            out[field] = f.encrypt(val.encode()).decode()
    return out


def _decrypt_settings(settings: Dict) -> Dict:
    """Decrypt sensitive fields in settings dict."""
    try:
        f = _get_fernet()
    except RuntimeError:
        return settings
    out = dict(settings)
    for field in _ENCRYPTED_FIELDS:
        val = out.get(field, "")
        if val and val.startswith("gAAAAA"):
            try:
                out[field] = f.decrypt(val.encode()).decode()
            except InvalidToken:
                logger.warning(f"Failed to decrypt {field}, clearing it")
                out[field] = ""
    return out


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


def reinforce_confidence(x: float, k: float = 0.2) -> float:
    """F(x) = x + k(1-x) — diminishing returns toward 1.0."""
    return x + k * (1.0 - x)


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

        # Neo4j graph store: uses OpenAI for entity extraction (Mem0's graph
        # module requires OpenAI function-calling format)
        if config.openai_api_key:
            try:
                import socket
                s = socket.socket()
                s.settimeout(2)
                host = config.neo4j_url.split("://")[1].split(":")[0]
                port = int(config.neo4j_url.split(":")[-1])
                s.connect((host, port))
                s.close()
                mem0_config["graph_store"] = {
                    "provider": "neo4j",
                    "config": {
                        "url": config.neo4j_url,
                        "username": config.neo4j_username,
                        "password": config.neo4j_password,
                    },
                    "llm": {
                        "provider": "openai",
                        "config": {
                            "model": "gpt-4o-mini",
                            "api_key": config.openai_api_key,
                        },
                    },
                }
                logger.info("Neo4j graph store connected (OpenAI for entity extraction)")
            except Exception:
                logger.info("Neo4j not available, using vector-only memory")
        else:
            logger.info("No OpenAI key, using vector-only memory (graph disabled)")

        self._memory = Memory.from_config(config_dict=mem0_config)

        # Separate Qdrant client for transcript and settings storage
        self._qdrant = QdrantClient(host=config.qdrant_host, port=config.qdrant_port)
        self._ensure_transcript_collection()
        self._ensure_settings_collection()

    def _ensure_transcript_collection(self):
        try:
            collections = [c.name for c in self._qdrant.get_collections().collections]
            if TRANSCRIPT_COLLECTION not in collections:
                self._qdrant.create_collection(
                    collection_name=TRANSCRIPT_COLLECTION,
                    vectors_config=models.VectorParams(size=1, distance=models.Distance.COSINE),
                )
                logger.info(f"Created collection {TRANSCRIPT_COLLECTION}")
        except Exception as e:
            logger.warning(f"Could not create transcript collection: {e}")

    def _ensure_settings_collection(self):
        try:
            collections = [c.name for c in self._qdrant.get_collections().collections]
            if USER_SETTINGS_COLLECTION not in collections:
                self._qdrant.create_collection(
                    collection_name=USER_SETTINGS_COLLECTION,
                    vectors_config=models.VectorParams(size=1, distance=models.Distance.COSINE),
                )
                logger.info(f"Created collection {USER_SETTINGS_COLLECTION}")
        except Exception as e:
            logger.warning(f"Could not create settings collection: {e}")

    async def get_user_settings(self, user_id: str) -> Dict:
        try:
            point_id = hashlib.md5(f"settings:{user_id}".encode()).hexdigest()
            points = await asyncio.to_thread(
                self._qdrant.retrieve,
                collection_name=USER_SETTINGS_COLLECTION,
                ids=[point_id],
                with_payload=True,
            )
            if points:
                return _decrypt_settings(points[0].payload)
            return {}
        except Exception as e:
            logger.warning(f"Failed to get user settings: {e}")
            return {}

    async def save_user_settings(self, user_id: str, settings: Dict) -> None:
        try:
            point_id = hashlib.md5(f"settings:{user_id}".encode()).hexdigest()
            # Merge with existing (decrypted) settings
            existing = await self.get_user_settings(user_id)
            existing.update(settings)
            existing["user_id"] = user_id
            existing["updated_at"] = datetime.now(timezone.utc).isoformat()
            # Encrypt before storing
            encrypted = _encrypt_settings(existing)
            await asyncio.to_thread(
                self._qdrant.upsert,
                collection_name=USER_SETTINGS_COLLECTION,
                points=[
                    models.PointStruct(
                        id=point_id,
                        vector=[0.0],
                        payload=encrypted,
                    )
                ],
            )
            logger.info(f"[SETTINGS] Saved settings for {user_id}")
        except Exception as e:
            logger.warning(f"Failed to save user settings: {e}")

    async def store_transcript(
        self, video_id: str, user_id: str, transcript: str,
        language_code: str = "en", title: str = "",
    ) -> None:
        try:
            existing = await self.get_transcript(video_id=video_id, user_id=user_id)
            if existing:
                logger.info(f"[TRANSCRIPT] Already stored for video {video_id}")
                return

            point_id = hashlib.md5(f"{user_id}:{video_id}".encode()).hexdigest()
            await asyncio.to_thread(
                self._qdrant.upsert,
                collection_name=TRANSCRIPT_COLLECTION,
                points=[
                    models.PointStruct(
                        id=point_id,
                        vector=[0.0],  # No vector search needed, just storage
                        payload={
                            "video_id": video_id,
                            "user_id": user_id,
                            "transcript": transcript,
                            "language_code": language_code,
                            "title": title,
                            "stored_at": datetime.now(timezone.utc).isoformat(),
                        },
                    )
                ],
            )
            logger.info(f"[TRANSCRIPT] Stored transcript for video {video_id} ({len(transcript)} chars)")
        except Exception as e:
            logger.warning(f"Failed to store transcript: {e}")

    async def get_transcript(self, video_id: str, user_id: str) -> Optional[Dict]:
        try:
            results, _ = await asyncio.to_thread(
                self._qdrant.scroll,
                collection_name=TRANSCRIPT_COLLECTION,
                scroll_filter=models.Filter(
                    must=[
                        models.FieldCondition(key="video_id", match=models.MatchValue(value=video_id)),
                        models.FieldCondition(key="user_id", match=models.MatchValue(value=user_id)),
                    ]
                ),
                limit=1,
            )
            if results:
                return results[0].payload
            return None
        except Exception as e:
            logger.warning(f"Failed to get transcript: {e}")
            return None

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
        """Store claim verbatim with metadata. No LLM rewriting."""
        try:
            kwargs = {"user_id": user_id, "infer": False}
            if metadata:
                kwargs["metadata"] = metadata
            await asyncio.to_thread(self._memory.add, text, **kwargs)
        except Exception as e:
            logger.warning(f"Memory add failed: {e}")

    async def add_if_new(
        self, text: str, user_id: str, metadata: dict = None, threshold: float = 0.85
    ) -> bool:
        """Store claim if new, or reinforce confidence if duplicate.

        Returns True if stored as new, False if duplicate (reinforced).
        """
        if metadata is None:
            metadata = {}

        try:
            existing = await self.search(text, user_id=user_id, limit=3)
            for mem in existing:
                if mem.score >= threshold:
                    # Reinforce existing claim
                    old_conf = (mem.metadata or {}).get("confidence", 0.4)
                    old_occ = (mem.metadata or {}).get("occurrences", 1)
                    new_conf = reinforce_confidence(old_conf)
                    new_occ = old_occ + 1
                    try:
                        await asyncio.to_thread(
                            self._memory.update,
                            memory_id=mem.id,
                            data=mem.text,
                            metadata={
                                **(mem.metadata or {}),
                                "confidence": new_conf,
                                "occurrences": new_occ,
                            },
                        )
                        logger.info(
                            f"[DEDUP] Reinforced (conf {old_conf:.2f}→{new_conf:.2f}, "
                            f"occ {old_occ}→{new_occ}): {mem.text[:60]}..."
                        )
                    except Exception as e:
                        logger.warning(f"[DEDUP] Failed to reinforce: {e}")
                    return False

            # New claim — add with occurrences=1
            metadata["occurrences"] = metadata.get("occurrences", 1)
            await self.add(text, user_id=user_id, metadata=metadata)
            return True
        except Exception as e:
            logger.warning(f"add_if_new failed: {e}")
            return False

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
