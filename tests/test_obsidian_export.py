"""Tests for Obsidian vault export from memory graph."""
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock, AsyncMock

from src.obsidian import ObsidianExporter, generate_vault


class TestObsidianExporter:
    def test_claim_to_markdown(self):
        exporter = ObsidianExporter()
        md = exporter.claim_to_md(
            text="The GOTCHA framework has six layers",
            metadata={
                "entity": "GOTCHA framework",
                "relation": "has",
                "value": "six layers",
                "confidence": 0.72,
                "occurrences": 3,
                "video_id": "abc123",
                "timestamp": "2026-03-28T18:00:00Z",
            },
        )
        assert "GOTCHA framework" in md
        assert "[[GOTCHA framework]]" in md
        assert "confidence: 0.72" in md
        assert "occurrences: 3" in md
        assert "abc123" in md

    def test_transcript_to_markdown(self):
        exporter = ObsidianExporter()
        md = exporter.transcript_to_md(
            video_id="abc123",
            transcript="Full transcript text about AI...",
            language_code="en",
        )
        assert "# Video abc123" in md
        assert "Full transcript text" in md
        assert "language: en" in md

    def test_entity_to_markdown(self):
        exporter = ObsidianExporter()
        md = exporter.entity_to_md(
            name="GOTCHA framework",
            entity_type="framework",
            relationships=[
                {"source": "GOTCHA framework", "rel": "has", "target": "six layers"},
                {"source": "GOTCHA framework", "rel": "bridges", "target": "AI"},
            ],
            claims=["The GOTCHA framework has six layers"],
        )
        assert "# GOTCHA framework" in md
        assert "framework" in md
        assert "[[six layers]]" in md
        assert "[[AI]]" in md

    def test_generate_vault_creates_folder_structure(self, tmp_path):
        claims = [
            {"text": "Python is popular", "metadata": {
                "entity": "Python", "relation": "is", "value": "popular",
                "confidence": 0.6, "occurrences": 2, "video_id": "v1",
                "timestamp": "2026-03-28T18:00:00Z"}},
        ]
        transcripts = [
            {"video_id": "v1", "transcript": "Full text...", "language_code": "en"},
        ]
        entities = [
            {"name": "Python", "type": "language",
             "relationships": [{"source": "Python", "rel": "is", "target": "popular"}],
             "claims": ["Python is popular"]},
        ]

        generate_vault(tmp_path, claims=claims, transcripts=transcripts, entities=entities)

        assert (tmp_path / "Claims").is_dir()
        assert (tmp_path / "Transcripts").is_dir()
        assert (tmp_path / "Entities").is_dir()
        assert len(list((tmp_path / "Claims").glob("*.md"))) == 1
        assert len(list((tmp_path / "Transcripts").glob("*.md"))) == 1
        assert len(list((tmp_path / "Entities").glob("*.md"))) == 1
