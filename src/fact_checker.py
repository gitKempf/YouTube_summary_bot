import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import List, Optional

from anthropic import AsyncAnthropic

from src.memory import MemoryEntry

logger = logging.getLogger(__name__)


class ClaimStatus(Enum):
    NEW = "new"
    SUPPORTED = "supported"
    CONTRADICTED = "contradicted"


@dataclass
class Claim:
    text: str
    entity: str
    relation: str
    value: str
    confidence: float = 0.8
    video_id: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


@dataclass
class ClassifiedClaim:
    claim: Claim
    status: ClaimStatus
    matching_memory: Optional[str] = None


@dataclass
class FactCheckResult:
    new_claims: List[ClassifiedClaim]
    supported_claims: List[ClassifiedClaim]
    contradicted_claims: List[ClassifiedClaim]
    context_summary: str


EXTRACTION_PROMPT = (
    "You are a fact extraction system. Extract atomic factual claims from the "
    "following transcript. Each claim should be a single, verifiable statement.\n\n"
    "Return a JSON array where each element has:\n"
    '- "text": the full claim as a natural sentence\n'
    '- "entity": the main subject/entity\n'
    '- "relation": the relationship or action\n'
    '- "value": the object or value\n'
    '- "confidence": float 0.0-1.0 (how confident is this claim based on the transcript, '
    "1.0 = explicitly stated, 0.5 = implied/inferred)\n\n"
    "Extract only concrete, factual claims — not opinions or filler.\n"
    "Return ONLY valid JSON, no markdown or explanation."
)

CLASSIFICATION_PROMPT = (
    "You are a fact-checking system. Compare each claim against the user's "
    "existing knowledge (memories). Classify each claim as:\n"
    '- "new": not covered by any existing memory\n'
    '- "supported": already known (matches an existing memory)\n'
    '- "contradicted": conflicts with an existing memory\n\n'
    "Return a JSON array where each element has:\n"
    '- "claim_text": the original claim text\n'
    '- "status": one of "new", "supported", "contradicted"\n'
    '- "matching_memory": the relevant memory text (or null if new)\n\n'
    "Return ONLY valid JSON, no markdown or explanation."
)


async def extract_claims(
    transcript: str,
    api_key: str,
    model: str = "claude-sonnet-4-6",
    video_id: str = "",
) -> List[Claim]:
    if not transcript or not transcript.strip():
        return []

    client = AsyncAnthropic(api_key=api_key)
    now = datetime.now(timezone.utc).isoformat()

    try:
        response = await client.messages.create(
            model=model,
            max_tokens=4096,
            system=EXTRACTION_PROMPT,
            messages=[{"role": "user", "content": transcript[:8000]}],
        )
        raw = response.content[0].text
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1].rsplit("```", 1)[0]
        claims_data = json.loads(raw)
        return [
            Claim(
                text=c.get("text", ""),
                entity=c.get("entity", ""),
                relation=c.get("relation", ""),
                value=c.get("value", ""),
                confidence=float(c.get("confidence", 0.8)),
                video_id=video_id,
                timestamp=now,
            )
            for c in claims_data
            if c.get("text")
        ]
    except (json.JSONDecodeError, KeyError, IndexError) as e:
        logger.warning(f"Failed to parse claims from LLM response: {e}")
        return []
    except Exception as e:
        logger.warning(f"Claim extraction failed: {e}")
        return []


async def classify_claims(
    claims: List[Claim],
    memories: List[MemoryEntry],
    api_key: str,
    model: str = "claude-sonnet-4-6",
) -> FactCheckResult:
    if not claims:
        return FactCheckResult([], [], [], "")

    if not memories:
        new = [ClassifiedClaim(claim=c, status=ClaimStatus.NEW) for c in claims]
        return FactCheckResult(new_claims=new, supported_claims=[], contradicted_claims=[], context_summary="")

    client = AsyncAnthropic(api_key=api_key)

    claims_text = "\n".join(f"- {c.text}" for c in claims)
    memories_text = "\n".join(f"- {m.text}" for m in memories)

    try:
        response = await client.messages.create(
            model=model,
            max_tokens=4096,
            system=CLASSIFICATION_PROMPT,
            messages=[
                {
                    "role": "user",
                    "content": (
                        f"CLAIMS FROM NEW VIDEO:\n{claims_text}\n\n"
                        f"USER'S EXISTING KNOWLEDGE:\n{memories_text}"
                    ),
                }
            ],
        )
        raw = response.content[0].text
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1].rsplit("```", 1)[0]
        classifications = json.loads(raw)
    except Exception as e:
        logger.warning(f"Claim classification failed: {e}")
        new = [ClassifiedClaim(claim=c, status=ClaimStatus.NEW) for c in claims]
        return FactCheckResult(new_claims=new, supported_claims=[], contradicted_claims=[], context_summary="")

    claim_map = {c.text: c for c in claims}
    new_claims = []
    supported_claims = []
    contradicted_claims = []

    for item in classifications:
        claim_text = item.get("claim_text", "")
        status_str = item.get("status", "new")
        matching = item.get("matching_memory")

        claim = claim_map.get(claim_text)
        if not claim and claims:
            claim = min(claims, key=lambda c: abs(len(c.text) - len(claim_text)))
        if not claim:
            continue

        status = ClaimStatus(status_str) if status_str in ("new", "supported", "contradicted") else ClaimStatus.NEW
        classified = ClassifiedClaim(claim=claim, status=status, matching_memory=matching)

        if status == ClaimStatus.NEW:
            new_claims.append(classified)
        elif status == ClaimStatus.SUPPORTED:
            supported_claims.append(classified)
        elif status == ClaimStatus.CONTRADICTED:
            contradicted_claims.append(classified)

    return FactCheckResult(
        new_claims=new_claims,
        supported_claims=supported_claims,
        contradicted_claims=contradicted_claims,
        context_summary="",
    )


def build_context_prompt(result: FactCheckResult) -> str:
    if not result.new_claims and not result.supported_claims and not result.contradicted_claims:
        return ""

    parts = []

    if result.supported_claims:
        items = "\n".join(f"- {c.claim.text}" for c in result.supported_claims)
        parts.append(
            f"USER'S PRIOR KNOWLEDGE (briefly reference, don't repeat in detail):\n{items}"
        )

    if result.contradicted_claims:
        items = "\n".join(
            f"- {c.claim.text} (previously known: {c.matching_memory})"
            for c in result.contradicted_claims
        )
        parts.append(f"CONTRADICTIONS TO ADDRESS:\n{items}")

    if result.new_claims:
        items = "\n".join(f"- {c.claim.text}" for c in result.new_claims)
        parts.append(f"FOCUS THE SUMMARY ON THESE NEW TOPICS:\n{items}")

    return "\n\n".join(parts)
