"""Anthropic-backed Synthesizer.

Second-class implementation kept for contract-test breadth (ADR-0001).
Mirrors :class:`OpenAISynthesizer` using Claude's tool-use mechanism.
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
from anamnesis.synthesizer.openai_impl import (
    _SYNTH_SCHEMA,
    _SYSTEM_PROMPT,
    _subgraph_to_prompt,
)

PROMPT_VERSION = "anthropic-v1"


class AnthropicSynthesizer:
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
            raise ImportError(
                "AnthropicSynthesizer requires the 'anthropic' package."
            ) from exc
        self._client = Anthropic(api_key=api_key, max_retries=max_retries, timeout=timeout)
        self._model = model or os.getenv(
            "ANAMNESIS_ANTHROPIC_MODEL", "claude-sonnet-4-6"
        )

    def synthesize(
        self,
        intent: NarrativeIntent,
        subgraph: RetrievedSubgraph,
        lens: str,
    ) -> SynthesisResult:
        valid_ko_ids = {n.ko_id for n in subgraph.nodes}
        tool: dict[str, Any] = {
            "name": "emit_synthesis",
            "description": "Emit the structured synthesis payload.",
            "input_schema": _SYNTH_SCHEMA["schema"],
        }
        resp = self._client.messages.create(
            model=self._model,
            max_tokens=4096,
            system=_SYSTEM_PROMPT,
            tools=[tool],
            tool_choice={"type": "tool", "name": "emit_synthesis"},
            messages=[{"role": "user", "content": _subgraph_to_prompt(intent, subgraph)}],
        )
        payload: dict[str, Any] | None = None
        for block in resp.content:
            if getattr(block, "type", None) == "tool_use":
                raw = block.input
                payload = raw if isinstance(raw, dict) else json.loads(raw)
                break
        if payload is None:
            raise RuntimeError("Anthropic synthesizer received no tool_use response")

        claims: list[Claim] = []
        provenance: dict[str, list[str]] = {}
        for row in payload["claims"]:
            supporting = [k for k in row["supporting_kos"] if k in valid_ko_ids]
            if not supporting:
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
