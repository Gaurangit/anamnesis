"""Anthropic-backed AdversarialCritic — second-class impl (ADR-0001)."""

from __future__ import annotations

import json
import os
from typing import Any

from anamnesis._hashing import content_hash
from anamnesis.critic.openai_impl import _CRIT_SCHEMA, _SYSTEM_PROMPT, _format_input
from anamnesis.models.critique import CritiqueReport, OverreachFlag
from anamnesis.models.subgraph import RetrievedSubgraph, SourceEvidence
from anamnesis.models.synthesis import SynthesisResult

PROMPT_VERSION = "anthropic-v1"


class AnthropicCritic:
    name = "anthropic"
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
            from anthropic import Anthropic
        except ImportError as exc:
            raise ImportError("AnthropicCritic requires the 'anthropic' package.") from exc
        self._client = Anthropic(api_key=api_key, max_retries=max_retries, timeout=timeout)
        self._model = model or os.getenv(
            "ANAMNESIS_ANTHROPIC_MODEL", "claude-sonnet-4-6"
        )

    def critique(
        self,
        synthesis: SynthesisResult,
        complement_subgraph: RetrievedSubgraph,
    ) -> CritiqueReport:
        tool: dict[str, Any] = {
            "name": "emit_critique",
            "description": "Emit the structured critique payload.",
            "input_schema": _CRIT_SCHEMA["schema"],
        }
        resp = self._client.messages.create(
            model=self._model,
            max_tokens=2048,
            temperature=0,
            system=_SYSTEM_PROMPT,
            tools=[tool],
            tool_choice={"type": "tool", "name": "emit_critique"},
            messages=[
                {
                    "role": "user",
                    "content": _format_input(synthesis, complement_subgraph),
                }
            ],
        )
        payload: dict[str, Any] | None = None
        for block in resp.content:
            if getattr(block, "type", None) == "tool_use":
                raw = block.input
                payload = raw if isinstance(raw, dict) else json.loads(raw)
                break
        if payload is None:
            raise RuntimeError("Anthropic critic received no tool_use response")

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
