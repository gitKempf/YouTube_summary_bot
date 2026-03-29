import io
import tempfile
import zipfile
from pathlib import Path

from fastapi import APIRouter, Query
from fastapi.responses import Response

router = APIRouter()


def build_vault_zip(user_id: str) -> bytes:
    """Build Obsidian vault and return as ZIP bytes."""
    from qdrant_client import QdrantClient, models

    import os
    qdrant = QdrantClient(
        host=os.environ.get("MEM0_QDRANT_HOST", "localhost"),
        port=int(os.environ.get("MEM0_QDRANT_PORT", "6333")),
    )

    # Use existing export logic
    from export_obsidian import fetch_claims, fetch_transcripts, fetch_entities
    from src.config import get_config
    from src.obsidian import generate_vault

    config = get_config()
    claims = fetch_claims(qdrant, user_id)
    transcripts = fetch_transcripts(qdrant, user_id)
    entities = fetch_entities(config)

    with tempfile.TemporaryDirectory() as tmpdir:
        vault_path = Path(tmpdir) / "vault"
        generate_vault(vault_path, claims=claims, transcripts=transcripts, entities=entities)

        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            for f in vault_path.rglob("*.md"):
                zf.write(f, f.relative_to(vault_path))
        return buf.getvalue()


@router.get("/api/export/obsidian")
def export_obsidian(user_id: str = Query(...)):
    data = build_vault_zip(user_id)
    return Response(
        content=data,
        media_type="application/zip",
        headers={"Content-Disposition": f"attachment; filename=obsidian_vault_{user_id}.zip"},
    )
