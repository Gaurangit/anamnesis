"""OpenAI-backed AdversarialCritic."""

from __future__ import annotations

import json
import os
from typing import Any

from anamnesis._hashing import content_hash
from anamnesis.models.critique import CritiqueReport, OverreachFlag
from anamnesis.models.subgraph import RetrievedSubgraph, SourceEvidence
from anamnesis.models.synthesis import SynthesisResult

PROMPT_VERSION = "openai-v1"

_SYSTEM_PROMPT = (
    "You are an adversarial critic. Given a SynthesisResult and a complement "
    "subgraph fetched with inverse-polarity relations and a broadened temporal "
    "window, identify claims that are overreaching. For each problematic claim, "
    "specify which complement evidence contradicts it and how much the confidence "
    "should drop. Suggest alternative framings."
)

_CRIT_SCHEMA: dict[str, Any] = {
    "name": "CritiquePayload",
    "strict": True,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "overreach_flags": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "claim_idx": {"type": "integer", "minimum": 0},
                        "reason": {"type": "string"},
                        "contradicting_ko_ids": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                        "confidence_delta": {"type": "number", "minimum": -1, "maximum": 1},
                    },
                    "required": [
                        "claim_idx",
                        "reason",
                        "contradicting_ko_ids",
                        "confidence_delta",
                    ],
                },
            },
            "alternative_framings": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["overreach_flags", "alternative_framings"],
    },
}


def _format_input(synthesis: SynthesisResult, complement: RetrievedSubgraph) -> str:
    lines = ["CLAIMS:"]
    for i, c in enumerate(synthesis.claims):
        lines.append(
            f"  [{i}] (confidence={c.confidence:.2f}, "
            f"supports={c.supporting_kos}) {c.text}"
        )
    lines.append("\nCOMPLEMENT SUBGRAPH NODES:")
    for n in complement.nodes:
        lines.append(f"  - {n.ko_id} (score={n.score:.2f})")
    lines.append("\nCOMPLEMENT SUBGRAPH EDGES:")
    for e in complement.edges:
        lines.append(f"  - {e.subject_ko} --{e.predicate}--> {e.object_ko}")
    if complement.contradictions:
        lines.append("\nCOMPLEMENT CONTRADICTIONS:")
        for c in complement.contradictions:
            lines.append(f"  - {c.locus}: {c.description}")
    return "\n".join(lines)


class OpenAICritic:
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
            raise ImportError("OpenAICritic requires the 'openai' package.") from exc
        self._client = OpenAI(api_key=api_key, max_retries=max_retries, timeout=timeout)
        self._model = model or os.getenv("ANAMNESIS_OPENAI_MODEL", "gpt-4o-mini")

    def critique(
        self,
        synthesis: SynthesisResult,
        complement_subgraph: RetrievedSubgraph,
    ) -> CritiqueReport:
        resp = self._client.responses.create(
            model=self._model,
            input=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": _format_input(synthesis, complement_subgraph)},
            ],
            text={"format": {"type": "json_schema", **_CRIT_SCHEMA}},
            temperature=0,
        )
        text = resp.output_text or resp.output[0].content[0].text
        payload = json.loads(text)

        flags: list[OverreachFlag] = []
        delta: dict[int, float] = {}
        n_claims = len(synthesis.claims)
        for row in payload["overreach_flags"]:
            idx = int(row["claim_idx"])
            if idx >= n_claims:
                continue
            flags.append(
                OverreachFlag(
                    claim_idx=idx,
                    reason=row["reason"],
                    contradicting_evidence=[
                        SourceEvidence(ko_id=k) for k in row["contradicting_ko_ids"]
                    ],
                )
            )
            delta[idx] = float(row["confidence_delta"])

        return CritiqueReport(
            critique_id=content_hash(
                f"crit::{synthesis.synthesis_id}::{complement_subgraph.subgraph_id}::{self.name}"
            ),
            synthesis_id=synthesis.synthesis_id,
            complement_subgraph_id=complement_subgraph.subgraph_id,
            overreach_flags=flags,
            alternative_framings=list(payload.get("alternative_framings", [])),
            confidence_delta=delta,
        )
