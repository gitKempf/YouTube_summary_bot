from fastapi import APIRouter, Query
from qdrant_client import QdrantClient, models

router = APIRouter()

_qdrant = None

def get_qdrant():
    global _qdrant
    if _qdrant is None:
        _qdrant = QdrantClient(host="localhost", port=6333)
    return _qdrant


@router.get("/api/videos")
def list_videos(user_id: str = Query(...)):
    qdrant = get_qdrant()
    try:
        points, _ = qdrant.scroll(
            collection_name="youtube_bot_transcripts",
            scroll_filter=models.Filter(
                must=[models.FieldCondition(key="user_id", match=models.MatchValue(value=user_id))]
            ),
            limit=100,
            with_payload=True,
        )
        return [
            {
                "video_id": p.payload.get("video_id", ""),
                "language_code": p.payload.get("language_code", ""),
                "stored_at": p.payload.get("stored_at", ""),
            }
            for p in points
        ]
    except Exception:
        return []
