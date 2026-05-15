"""OpenAI-backed Synthesizer.

Emits a SynthesisResult with a structured-output JSON schema. The
prompt pins three rules the eval harness will check:

1. Every grounded claim must list at least one supporting_kos entry.
2. Cross-domain inferences must be emitted in bridged_inferences, never
   mixed into claims.
3. Contradictions in the subgraph must be surfaced as either a softer
   confidence on the affected claim or as an explicit caveat in the
   primary_artifact summary.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any

from anamnesis._hashing import content_hash
from anamnesis.models.intent import NarrativeIntent
from anamnesis.models.subgraph import RetrievedSubgraph
from anamnesis.models.synthesis import BridgedInference, Claim, SynthesisResult

PROMPT_VERSION = "openai-v1"

_SYSTEM_PROMPT = (
    "You synthesize a SynthesisResult from a retrieval subgraph. Rules:\n"
    "- Every claim with kind='grounded' MUST list at least one supporting_kos id.\n"
    "- Cross-domain inferences MUST go in bridged_inferences with kind='inference'.\n"
    "- If the subgraph lists contradictions, lower the affected claim's confidence "
    "or surface a caveat in summary.\n"
    "- Never invent KO ids; only cite ko_ids that appear in the input subgraph.\n"
)

_SYNTH_SCHEMA: dict[str, Any] = {
    "name": "SynthesisPayload",
    "strict": True,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "summary": {"type": "string"},
            "claims": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "text": {"type": "string"},
                        "supporting_kos": {"type": "array", "items": {"type": "string"}},
                        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                    },
                    "required": ["text", "supporting_kos", "confidence"],
                },
            },
            "bridged_inferences": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "text": {"type": "string"},
                        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                    },
                    "required": ["text", "confidence"],
                },
            },
        },
        "required": ["summary", "claims", "bridged_inferences"],
    },
}


def _subgraph_to_prompt(intent: NarrativeIntent, subgraph: RetrievedSubgraph) -> str:
    lines = [
        f"PRIMARY_THEME: {intent.primary_theme}",
        f"OUTPUT_TYPE_HINT: {intent.output_type_hint}",
        "",
        "NODES:",
    ]
    for n in subgraph.nodes:
        lines.append(f"  - {n.ko_id} (score={n.score:.2f})")
    lines.append("")
    lines.append("EDGES:")
    for e in subgraph.edges:
        lines.append(f"  - {e.subject_ko} --{e.predicate}--> {e.object_ko}")
    if subgraph.contradictions:
        lines.append("")
        lines.append("CONTRADICTIONS:")
        for c in subgraph.contradictions:
            lines.append(f"  - {c.locus}: {c.description}")
    if subgraph.gaps:
        lines.append("")
        lines.append("GAPS:")
        for g in subgraph.gaps:
            lines.append(f"  - {g}")
    return "\n".join(lines)


class OpenAISynthesizer:
    name = "openai"
    prompt_version = PROMPT_VERSION

    def __init__(
        self,
        model: str | None = None,
        *,
        api_key: str | None = None,
        timeout: float = 60.0,
        max_retries: int = 3,
    ) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError(
                "OpenAISynthesizer requires the 'openai' package."
            ) from exc
        self._client = OpenAI(api_key=api_key, max_retries=max_retries, timeout=timeout)
        self._model = model or os.getenv("ANAMNESIS_OPENAI_MODEL", "gpt-4o-mini")

    def synthesize(
        self,
        intent: NarrativeIntent,
        subgraph: RetrievedSubgraph,
        lens: str,
    ) -> SynthesisResult:
        valid_ko_ids = {n.ko_id for n in subgraph.nodes}
        resp = self._client.responses.create(
            model=self._model,
            input=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": _subgraph_to_prompt(intent, subgraph)},
            ],
            text={"format": {"type": "json_schema", **_SYNTH_SCHEMA}},
            # Synthesis is the one place non-determinism is welcome.
        )
        text = resp.output_text or resp.output[0].content[0].text
        payload = json.loads(text)

        claims: list[Claim] = []
        provenance: dict[str, list[str]] = {}
        for row in payload["claims"]:
            supporting = [k for k in row["supporting_kos"] if k in valid_ko_ids]
            if not supporting:
                # Demote to inference if the model cited unknown ids.
                claims.append(
                    Claim(
                        text=row["text"],
                        kind="inference",
                        supporting_kos=[],
                        confidence=float(row["confidence"]) * 0.5,
                    )
                )
                continue
            claims.append(
                Claim(
                    text=row["text"],
                    kind="grounded",
                    supporting_kos=supporting,
                    confidence=float(row["confidence"]),
                )
            )
            provenance[str(len(claims) - 1)] = supporting

        bridged = [
            BridgedInference(
                text=row["text"],
                supporting_kos=[],
                confidence=float(row["confidence"]),
            )
            for row in payload["bridged_inferences"]
        ]

        return SynthesisResult(
            synthesis_id=content_hash(
                f"syn::{subgraph.subgraph_id}::{lens}::{self.name}:{self._model}"
            ),
            subgraph_id=subgraph.subgraph_id,
            lens=lens,
            primary_artifact={"lens": lens, "summary": payload["summary"]},
            claims=claims,
            bridged_inferences=bridged,
            provenance_map=provenance,
            created_at=datetime.now(tz=timezone.utc),
            synthesizer_model=f"{self.name}:{self._model}",
        )
