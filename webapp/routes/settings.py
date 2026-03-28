from fastapi import APIRouter, Query
from qdrant_client import QdrantClient

router = APIRouter()

_qdrant = None

def get_qdrant():
    global _qdrant
    if _qdrant is None:
        _qdrant = QdrantClient(host="localhost", port=6333)
    return _qdrant


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
        from qdrant_client import models
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

    return {
        "user_id": user_id,
        "claim_count": claim_count,
        "video_count": video_count,
    }
