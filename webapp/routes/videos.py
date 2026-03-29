import json
import os
import urllib.request
from collections import Counter

from fastapi import APIRouter, Query
from qdrant_client import QdrantClient, models

router = APIRouter()

_qdrant = None

def get_qdrant():
    global _qdrant
    if _qdrant is None:
        _qdrant = QdrantClient(
            host=os.environ.get("MEM0_QDRANT_HOST", "localhost"),
            port=int(os.environ.get("MEM0_QDRANT_PORT", "6333")),
        )
    return _qdrant


def _fetch_title_from_oembed(video_id: str) -> str:
    try:
        url = f"https://www.youtube.com/oembed?url=https://youtube.com/watch?v={video_id}&format=json"
        resp = urllib.request.urlopen(url, timeout=5)
        data = json.loads(resp.read())
        return data.get("title", "")
    except Exception:
        return ""


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
    except Exception:
        return []

    # Count claims per video from memories collection
    claim_counts = Counter()
    duplicate_counts = Counter()
    try:
        mem_points, _ = qdrant.scroll(
            collection_name="youtube_bot_memories",
            scroll_filter=models.Filter(
                must=[models.FieldCondition(key="user_id", match=models.MatchValue(value=user_id))]
            ),
            limit=10000,
            with_payload=True,
        )
        for mp in mem_points:
            vid = mp.payload.get("video_id") or mp.payload.get("metadata", {}).get("video_id", "")
            if vid:
                occurrences = (
                    mp.payload.get("occurrences")
                    or mp.payload.get("metadata", {}).get("occurrences", 1)
                )
                claim_counts[vid] += 1
                if isinstance(occurrences, (int, float)) and occurrences > 1:
                    duplicate_counts[vid] += 1
    except Exception:
        pass

    results = []
    for p in points:
        video_id = p.payload.get("video_id", "")
        title = p.payload.get("title", "")
        if not title and video_id:
            title = _fetch_title_from_oembed(video_id)
        results.append({
            "video_id": video_id,
            "title": title,
            "language_code": p.payload.get("language_code", ""),
            "stored_at": p.payload.get("stored_at", ""),
            "new_claims": claim_counts.get(video_id, 0),
            "duplicates": duplicate_counts.get(video_id, 0),
        })

    return results
