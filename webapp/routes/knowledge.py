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


@router.get("/api/graph")
def get_graph(user_id: str = Query(...)):
    """Build graph data from claims: entities + videos as nodes, co-occurrence as edges."""
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
        return {"nodes": [], "edges": []}

    # Collect entities per video and entity claim counts
    entity_counts = {}
    video_entities: dict[str, set] = {}  # video_id -> set of entities

    for p in points:
        meta = p.payload.get("metadata", {})
        entity = meta.get("entity", p.payload.get("entity", ""))
        video_id = meta.get("video_id", p.payload.get("video_id", ""))
        if not entity:
            continue
        entity_counts[entity] = entity_counts.get(entity, 0) + 1
        if video_id:
            video_entities.setdefault(video_id, set()).add(entity)

    # Build nodes
    nodes = []
    node_ids = set()
    for entity, count in entity_counts.items():
        nodes.append({"id": entity, "label": entity, "type": "entity", "size": count})
        node_ids.add(entity)

    for vid in video_entities:
        nodes.append({"id": f"v:{vid}", "label": vid[:12], "type": "video", "size": len(video_entities[vid])})
        node_ids.add(f"v:{vid}")

    # Build edges
    edges = []
    edge_set = set()

    # Entity <-> Video edges
    for vid, ents in video_entities.items():
        for ent in ents:
            key = (ent, f"v:{vid}")
            if key not in edge_set:
                edge_set.add(key)
                edges.append({"source": ent, "target": f"v:{vid}", "type": "appears_in"})

    # Entity <-> Entity co-occurrence (same video)
    for vid, ents in video_entities.items():
        ent_list = sorted(ents)
        for i in range(len(ent_list)):
            for j in range(i + 1, len(ent_list)):
                key = (ent_list[i], ent_list[j])
                if key not in edge_set:
                    edge_set.add(key)
                    edges.append({"source": ent_list[i], "target": ent_list[j], "type": "co_occurs"})

    # Neo4j relationships (bonus)
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
                    src, dst = r["src"], r["dst"]
                    key = (src, dst)
                    if key not in edge_set and src in node_ids:
                        if dst not in node_ids:
                            nodes.append({"id": dst, "label": dst, "type": "entity", "size": 1})
                            node_ids.add(dst)
                        edge_set.add(key)
                        edges.append({"source": src, "target": dst, "type": r["rel"]})
            driver.close()
    except Exception:
        pass

    return {"nodes": nodes, "edges": edges}
