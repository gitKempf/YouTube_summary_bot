#!/usr/bin/env python3
"""Export user's memory to Quartz content folder and build static site."""
import argparse
import logging
import shutil
import subprocess
from pathlib import Path

from dotenv import load_dotenv
from qdrant_client import QdrantClient

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

QUARTZ_DIR = Path(__file__).parent / "quartz"
CONTENT_DIR = QUARTZ_DIR / "content"
PUBLIC_DIR = QUARTZ_DIR / "public"


def rebuild_vault(user_id: str = "tg_120292793"):
    """Export memories to Quartz content and build static site."""
    from src.config import get_config
    from src.obsidian import generate_vault
    from export_obsidian import fetch_claims, fetch_transcripts, fetch_entities

    config = get_config()
    qdrant = QdrantClient(host=config.qdrant_host, port=config.qdrant_port)

    claims = fetch_claims(qdrant, user_id)
    transcripts = fetch_transcripts(qdrant, user_id)
    entities = fetch_entities(config)

    # Clear old content, keep Quartz config
    if CONTENT_DIR.exists():
        for item in CONTENT_DIR.iterdir():
            if item.name not in ("index.md",):
                if item.is_dir():
                    shutil.rmtree(item)
                else:
                    item.unlink()

    # Generate vault into Quartz content dir
    generate_vault(CONTENT_DIR, claims=claims, transcripts=transcripts, entities=entities)

    # Create index page
    (CONTENT_DIR / "index.md").write_text(
        f"---\ntitle: Knowledge Base\n---\n\n"
        f"# Knowledge Base\n\n"
        f"**{len(claims)}** claims | **{len(entities)}** entities | **{len(transcripts)}** transcripts\n\n"
        f"## Browse\n"
        f"- [[Claims/]] — Extracted facts from videos\n"
        f"- [[Entities/]] — People, tools, concepts\n"
        f"- [[Transcripts/]] — Full video transcripts\n\n"
        f"Use the **Graph View** (top right) to explore connections.\n\n"
        f'<a href="/api/export/obsidian?user_id={user_id}" '
        f'style="display:inline-block;padding:10px 20px;background:#2481cc;color:#fff;'
        f'border-radius:8px;text-decoration:none;font-weight:600">'
        f'Download Obsidian Vault (ZIP)</a>\n'
    )

    logger.info(f"Vault exported: {len(claims)} claims, {len(entities)} entities, {len(transcripts)} transcripts")

    # Build Quartz
    logger.info("Building Quartz static site...")
    result = subprocess.run(
        ["npx", "quartz", "build"],
        cwd=str(QUARTZ_DIR),
        capture_output=True,
        text=True,
        timeout=120,
    )
    if result.returncode == 0:
        logger.info(f"Quartz built → {PUBLIC_DIR}")
    else:
        logger.warning(f"Quartz build failed: {result.stderr[:500]}")

    return PUBLIC_DIR


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--user", default="tg_120292793")
    args = parser.parse_args()
    rebuild_vault(args.user)
