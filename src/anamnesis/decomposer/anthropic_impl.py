"""Anthropic-backed NarrativeDecomposer.

Second-class implementation per ADR-0001: kept for contract-testing
breadth, not equal feature parity. Uses Claude's tool-use mechanism to
get structured output by declaring a single tool whose input schema
matches the intent payload, then forcing the model to invoke it.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any

from anamnesis._hashing import content_hash
from anamnesis.decomposer.openai_impl import _INTENT_SCHEMA, _SYSTEM_PROMPT
from anamnesis.models.intent import (
    EntityMention,
    NarrativeIntent,
    RelationHint,
    TemporalScope,
)

PROMPT_VERSION = "anthropic-v1"


class AnthropicDecomposer:
    """Anthropic-backed decomposer using tool-use for structured output."""

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
                "AnthropicDecomposer requires the 'anthropic' package. "
                "Install with: pip install 'anamnesis[anthropic]'"
            ) from exc
        self._client = Anthropic(api_key=api_key, max_retries=max_retries, timeout=timeout)
        self._model = model or os.getenv(
            "ANAMNESIS_ANTHROPIC_MODEL", "claude-sonnet-4-6"
        )

    def decompose(
        self,
        essay: str,
        *,
        schema_hints: list[str] | None = None,
    ) -> NarrativeIntent:
        user = essay
        if schema_hints:
            user += "\n\nSCHEMA HINTS:\n" + "\n".join(f"- {h}" for h in schema_hints)

        tool: dict[str, Any] = {
            "name": "emit_intent",
            "description": "Emit the structured retrieval intent for this essay.",
            "input_schema": _INTENT_SCHEMA["schema"],
        }

        resp = self._client.messages.create(
            model=self._model,
            max_tokens=4096,
            temperature=0,
            system=_SYSTEM_PROMPT,
            tools=[tool],
            tool_choice={"type": "tool", "name": "emit_intent"},
            messages=[{"role": "user", "content": user}],
        )

        payload: dict[str, Any] | None = None
        for block in resp.content:
            if getattr(block, "type", None) == "tool_use":
                raw = block.input
                payload = raw if isinstance(raw, dict) else json.loads(raw)
                break
        if payload is None:
            raise RuntimeError("Anthropic decomposer received no tool_use response")

        essay_hash = content_hash(essay)
        intent_id = content_hash(
            f"{essay_hash}::{self.name}:{self._model}::{self.prompt_version}"
        )

        entities = [EntityMention(**row) for row in payload["entities"]]
        relations: list[RelationHint] = []
        for row in payload["relations"]:
            fuzzy = row.pop("fuzzy_temporal_label", None)
            relations.append(
                RelationHint(
                    temporal_qualifier=TemporalScope(fuzzy_label=fuzzy) if fuzzy else None,
                    **row,
                )
            )
        temporal_label = payload.get("temporal_fuzzy_label")
        temporal_scope = TemporalScope(fuzzy_label=temporal_label) if temporal_label else None

        return NarrativeIntent(
            intent_id=intent_id,
            essay_hash=essay_hash,
            decomposer_model=f"{self.name}:{self._model}",
            prompt_version=self.prompt_version,
            primary_theme=payload["primary_theme"],
            entities=entities,
            relations=relations,
            temporal_scope=temporal_scope,
            analogy_targets=list(payload.get("analogy_targets", [])),
            output_type_hint=payload["output_type_hint"],
            alternative_interpretations=list(payload.get("alternative_interpretations", [])),
            ambiguity_score=float(payload["ambiguity_score"]),
            created_at=datetime.now(tz=timezone.utc),
        )
