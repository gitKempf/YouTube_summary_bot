#!/usr/bin/env python3
"""Export user's memory graph to an Obsidian vault.

Usage:
    python export_obsidian.py --user tg_120292793 --output ./obsidian_vault
"""
import argparse
import logging
from pathlib import Path

from dotenv import load_dotenv
from qdrant_client import QdrantClient, models

from src.config import get_config
from src.obsidian import ObsidianExporter, generate_vault

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def fetch_claims(qdrant: QdrantClient, user_id: str):
    """Fetch all claims for a user from Qdrant."""
    claims = []
    points, offset = qdrant.scroll(
        collection_name="youtube_bot_memories",
        scroll_filter=models.Filter(
            must=[models.FieldCondition(key="user_id", match=models.MatchValue(value=user_id))]
        ),
        limit=1000,
        with_payload=True,
    )
    for p in points:
        claims.append({
            "text": p.payload.get("data", p.payload.get("memory", "")),
            "metadata": {k: v for k, v in p.payload.items() if k not in ("data",)},
        })
    logger.info(f"Fetched {len(claims)} claims")
    return claims


def fetch_transcripts(qdrant: QdrantClient, user_id: str):
    """Fetch all transcripts for a user."""
    transcripts = []
    points, _ = qdrant.scroll(
        collection_name="youtube_bot_transcripts",
        scroll_filter=models.Filter(
            must=[models.FieldCondition(key="user_id", match=models.MatchValue(value=user_id))]
        ),
        limit=100,
        with_payload=True,
    )
    for p in points:
        transcripts.append(p.payload)
    logger.info(f"Fetched {len(transcripts)} transcripts")
    return transcripts


def fetch_entities(config):
    """Fetch entities and relationships from Neo4j."""
    entities = {}
    try:
        from neo4j import GraphDatabase
        driver = GraphDatabase.driver(
            config.neo4j_url,
            auth=(config.neo4j_username, config.neo4j_password),
        )
        with driver.session() as session:
            # Nodes
            result = session.run(
                "MATCH (n) WHERE n.user_id IS NOT NULL "
                "RETURN n.name AS name, labels(n) AS labels, n.user_id AS uid"
            )
            for r in result:
                name = r["name"]
                if name and name not in entities:
                    entities[name] = {
                        "name": name,
                        "type": r["labels"][0] if r["labels"] else "unknown",
                        "relationships": [],
                        "claims": [],
                    }

            # Relationships
            result = session.run(
                "MATCH (a)-[r]->(b) WHERE a.user_id IS NOT NULL "
                "RETURN a.name AS src, type(r) AS rel, b.name AS dst"
            )
            for r in result:
                src = r["src"]
                if src in entities:
                    entities[src]["relationships"].append({
                        "source": r["src"],
                        "rel": r["rel"],
                        "target": r["dst"],
                    })

        driver.close()
        logger.info(f"Fetched {len(entities)} entities from Neo4j")
    except Exception as e:
        logger.warning(f"Could not fetch Neo4j entities: {e}")

    return list(entities.values())


def main():
    parser = argparse.ArgumentParser(description="Export memory to Obsidian vault")
    parser.add_argument("--user", default="tg_120292793", help="User ID")
    parser.add_argument("--output", default="./obsidian_vault", help="Output directory")
    args = parser.parse_args()

    config = get_config()
    qdrant = QdrantClient(host=config.qdrant_host, port=config.qdrant_port)

    claims = fetch_claims(qdrant, args.user)
    transcripts = fetch_transcripts(qdrant, args.user)
    entities = fetch_entities(config)

    output = Path(args.output)
    generate_vault(output, claims=claims, transcripts=transcripts, entities=entities)
    logger.info(f"Obsidian vault exported to {output.absolute()}")
    logger.info(f"  {len(claims)} claims, {len(transcripts)} transcripts, {len(entities)} entities")
    print(f"\nOpen in Obsidian: {output.absolute()}")


if __name__ == "__main__":
    main()
