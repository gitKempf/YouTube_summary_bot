import os
import logging

from fastapi import APIRouter, Query
from qdrant_client import QdrantClient, models

router = APIRouter()
logger = logging.getLogger(__name__)

_qdrant = None


def get_qdrant():
    global _qdrant
    if _qdrant is None:
        _qdrant = QdrantClient(
            host=os.environ.get("MEM0_QDRANT_HOST", "localhost"),
            port=int(os.environ.get("MEM0_QDRANT_PORT", "6333")),
        )
    return _qdrant


@router.get("/api/claims")
def list_claims(user_id: str = Query(...)):
    qdrant = get_qdrant()
    try:
        points, _ = qdrant.scroll(
            collection_name="youtube_bot_memories",
            scroll_filter=models.Filter(
                must=[models.FieldCondition(key="user_id", match=models.MatchValue(value=user_id))]
            ),
            limit=10000,
            with_payload=True,
        )
    except Exception:
        return []

    claims = []
    for p in points:
        payload = p.payload
        meta = payload.get("metadata", {})
        claims.append({
            "id": str(p.id),
            "text": payload.get("data", payload.get("memory", "")),
            "entity": meta.get("entity", payload.get("entity", "")),
            "relation": meta.get("relation", payload.get("relation", "")),
            "value": meta.get("value", payload.get("value", "")),
            "confidence": meta.get("confidence", payload.get("confidence", 0)),
            "occurrences": meta.get("occurrences", payload.get("occurrences", 1)),
            "video_id": meta.get("video_id", payload.get("video_id", "")),
            "timestamp": meta.get("timestamp", payload.get("timestamp", "")),
        })

    return claims


@router.get("/api/entities")
def list_entities(user_id: str = Query(...)):
    """Extract unique entities from claims and optionally from Neo4j."""
    qdrant = get_qdrant()
    entities = {}

    try:
        points, _ = qdrant.scroll(
            collection_name="youtube_bot_memories",
            scroll_filter=models.Filter(
                must=[models.FieldCondition(key="user_id", match=models.MatchValue(value=user_id))]
            ),
            limit=10000,
            with_payload=True,
        )
        for p in points:
            meta = p.payload.get("metadata", {})
            entity = meta.get("entity", p.payload.get("entity", ""))
            if entity and entity not in entities:
                entities[entity] = {"name": entity, "claim_count": 0, "claims": []}
            if entity in entities:
                entities[entity]["claim_count"] += 1
                text = p.payload.get("data", p.payload.get("memory", ""))
                if text:
                    entities[entity]["claims"].append(text[:120])
    except Exception as e:
        logger.warning(f"Failed to fetch entities: {e}")

    # Try to enrich with Neo4j relationships
    try:
        from src.config import get_config
        config = get_config()
        if config.openai_api_key:
            from neo4j import GraphDatabase
            driver = GraphDatabase.driver(
                os.environ.get("MEM0_NEO4J_URL", config.neo4j_url),
                auth=(config.neo4j_username, os.environ.get("MEM0_NEO4J_PASSWORD", config.neo4j_password)),
            )
            with driver.session() as session:
                result = session.run(
                    "MATCH (a)-[r]->(b) WHERE a.user_id IS NOT NULL "
                    "RETURN a.name AS src, type(r) AS rel, b.name AS dst LIMIT 500"
                )
                for r in result:
                    src = r["src"]
                    if src in entities:
                        if "relationships" not in entities[src]:
                            entities[src]["relationships"] = []
                        entities[src]["relationships"].append({
                            "relation": r["rel"],
                            "target": r["dst"],
                        })
            driver.close()
    except Exception:
        pass

    return sorted(entities.values(), key=lambda e: e["claim_count"], reverse=True)
