"""Export memory graph to Obsidian vault (markdown + wikilinks)."""
import re
from pathlib import Path
from typing import Dict, List, Optional


def _sanitize_filename(name: str) -> str:
    name = re.sub(r'[<>:"/\\|?*]', '_', name)
    name = name.rstrip('. ')  # trailing dots/spaces break URL routing
    return name[:100]


class ObsidianExporter:
    def claim_to_md(self, text: str, metadata: Dict) -> str:
        entity = metadata.get("entity", "unknown")
        relation = metadata.get("relation", "")
        value = metadata.get("value", "")
        confidence = metadata.get("confidence", 0.0)
        occurrences = metadata.get("occurrences", 1)
        video_id = metadata.get("video_id", "")
        timestamp = metadata.get("timestamp", "")

        lines = [
            "---",
            f"entity: \"[[{entity}]]\"",
            f"relation: {relation}",
            f"value: \"{value}\"",
            f"confidence: {confidence}",
            f"occurrences: {occurrences}",
            f"video_id: {video_id}",
            f"timestamp: {timestamp}",
            f"type: claim",
            "---",
            "",
            f"# {text}",
            "",
            f"**Entity:** [[{entity}]]",
            f"**Relation:** {relation}",
            f"**Value:** {value}",
            "",
            f"**Confidence:** {confidence:.2f} | **Occurrences:** {occurrences}",
            f"**Source:** [[Video {video_id}]]",
        ]
        return "\n".join(lines)

    def transcript_to_md(self, video_id: str, transcript: str, language_code: str = "en") -> str:
        lines = [
            "---",
            f"video_id: {video_id}",
            f"language: {language_code}",
            f"type: transcript",
            "---",
            "",
            f"# Video {video_id}",
            "",
            transcript,
        ]
        return "\n".join(lines)

    def entity_to_md(
        self,
        name: str,
        entity_type: str,
        relationships: List[Dict],
        claims: List[str],
    ) -> str:
        lines = [
            "---",
            f"type: entity",
            f"entity_type: {entity_type}",
            "---",
            "",
            f"# {name}",
            f"**Type:** {entity_type}",
            "",
        ]

        if relationships:
            lines.append("## Relationships")
            for r in relationships:
                src = r.get("source", "")
                rel = r.get("rel", "")
                tgt = r.get("target", "")
                lines.append(f"- [[{src}]] **{rel}** [[{tgt}]]")
            lines.append("")

        if claims:
            lines.append("## Claims")
            for c in claims:
                safe = _sanitize_filename(c)
                lines.append(f"- [[{safe}]]")
            lines.append("")

        return "\n".join(lines)


def generate_vault(
    output_dir: Path,
    claims: List[Dict],
    transcripts: List[Dict],
    entities: List[Dict],
) -> Path:
    """Generate an Obsidian vault from memory data."""
    output_dir = Path(output_dir)
    claims_dir = output_dir / "Claims"
    transcripts_dir = output_dir / "Transcripts"
    entities_dir = output_dir / "Entities"

    for d in [claims_dir, transcripts_dir, entities_dir]:
        d.mkdir(parents=True, exist_ok=True)

    exporter = ObsidianExporter()

    for i, claim in enumerate(claims):
        md = exporter.claim_to_md(claim["text"], claim.get("metadata", {}))
        filename = _sanitize_filename(claim["text"][:80])
        (claims_dir / f"{filename}.md").write_text(md)

    for t in transcripts:
        md = exporter.transcript_to_md(
            t["video_id"], t["transcript"], t.get("language_code", "en")
        )
        (transcripts_dir / f"Video {t['video_id']}.md").write_text(md)

    for e in entities:
        md = exporter.entity_to_md(
            e["name"], e.get("type", "unknown"),
            e.get("relationships", []), e.get("claims", []),
        )
        filename = _sanitize_filename(e["name"])
        (entities_dir / f"{filename}.md").write_text(md)

    return output_dir
