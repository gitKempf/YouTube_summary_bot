import os

from fastapi import APIRouter, Query
from pydantic import BaseModel
from qdrant_client import QdrantClient, models

import hashlib
from datetime import datetime, timezone

from src.memory import _encrypt_settings, _decrypt_settings

router = APIRouter()

_qdrant = None
USER_SETTINGS_COLLECTION = "youtube_bot_user_settings"


def get_qdrant():
    global _qdrant
    if _qdrant is None:
        _qdrant = QdrantClient(
            host=os.environ.get("MEM0_QDRANT_HOST", "localhost"),
            port=int(os.environ.get("MEM0_QDRANT_PORT", "6333")),
        )
    return _qdrant


def _ensure_settings_collection(qdrant: QdrantClient):
    try:
        collections = [c.name for c in qdrant.get_collections().collections]
        if USER_SETTINGS_COLLECTION not in collections:
            qdrant.create_collection(
                collection_name=USER_SETTINGS_COLLECTION,
                vectors_config=models.VectorParams(size=1, distance=models.Distance.COSINE),
            )
    except Exception:
        pass


@router.get("/api/settings")
def get_settings(user_id: str = Query(...)):
    qdrant = get_qdrant()
    try:
        claim_count = qdrant.count(
            collection_name="youtube_bot_memories",
            count_filter={"must": [{"key": "user_id", "match": {"value": user_id}}]},
        ).count
    except Exception:
        claim_count = 0

    try:
        points, _ = qdrant.scroll(
            collection_name="youtube_bot_transcripts",
            scroll_filter=models.Filter(
                must=[models.FieldCondition(key="user_id", match=models.MatchValue(value=user_id))]
            ),
            limit=1000,
        )
        video_count = len(points)
    except Exception:
        video_count = 0

    # Check if user has API keys configured (decrypt to verify)
    has_anthropic_key = False
    has_elevenlabs_key = False
    try:
        _ensure_settings_collection(qdrant)
        point_id = hashlib.md5(f"settings:{user_id}".encode()).hexdigest()
        points = qdrant.retrieve(
            collection_name=USER_SETTINGS_COLLECTION,
            ids=[point_id],
            with_payload=True,
        )
        if points:
            decrypted = _decrypt_settings(points[0].payload)
            has_anthropic_key = bool(decrypted.get("anthropic_api_key"))
            has_elevenlabs_key = bool(decrypted.get("elevenlabs_api_key"))
    except Exception:
        pass

    return {
        "user_id": user_id,
        "claim_count": claim_count,
        "video_count": video_count,
        "has_anthropic_key": has_anthropic_key,
        "has_elevenlabs_key": has_elevenlabs_key,
    }


class ApiKeyRequest(BaseModel):
    user_id: str
    api_key: str
    key_type: str = "anthropic"


def _get_existing_settings(qdrant, user_id):
    point_id = hashlib.md5(f"settings:{user_id}".encode()).hexdigest()
    try:
        points = qdrant.retrieve(
            collection_name=USER_SETTINGS_COLLECTION,
            ids=[point_id],
            with_payload=True,
        )
        if points:
            return _decrypt_settings(points[0].payload)
    except Exception:
        pass
    return {}


def _save_settings(qdrant, user_id, settings):
    point_id = hashlib.md5(f"settings:{user_id}".encode()).hexdigest()
    settings["user_id"] = user_id
    settings["updated_at"] = datetime.now(timezone.utc).isoformat()
    encrypted = _encrypt_settings(settings)
    qdrant.upsert(
        collection_name=USER_SETTINGS_COLLECTION,
        points=[
            models.PointStruct(id=point_id, vector=[0.0], payload=encrypted)
        ],
    )


@router.post("/api/settings/apikey")
def save_api_key(req: ApiKeyRequest):
    key_field = f"{req.key_type}_api_key"
    prefix = "sk-ant-" if req.key_type == "anthropic" else "sk_"
    if not req.api_key.startswith(prefix):
        return {"ok": False, "error": f"Invalid format. Keys should start with {prefix}"}

    qdrant = get_qdrant()
    _ensure_settings_collection(qdrant)
    existing = _get_existing_settings(qdrant, req.user_id)
    existing[key_field] = req.api_key
    _save_settings(qdrant, req.user_id, existing)
    return {"ok": True}


class DeleteKeyRequest(BaseModel):
    user_id: str
    key_type: str = "anthropic"


@router.post("/api/settings/apikey/delete")
def delete_api_key(req: DeleteKeyRequest):
    qdrant = get_qdrant()
    _ensure_settings_collection(qdrant)
    existing = _get_existing_settings(qdrant, req.user_id)
    existing[f"{req.key_type}_api_key"] = ""
    _save_settings(qdrant, req.user_id, existing)
    return {"ok": True}
