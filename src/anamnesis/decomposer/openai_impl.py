"""OpenAI-backed NarrativeDecomposer.

Mirrors the structured-output JSON-schema pattern from
``knowledge_objects.runtime.quality.cq_eval.OpenAIJudge``. Temperature is
fixed at 0 so identical (essay, model, prompt_version) tuples produce
identical Intents — that is what makes the intent cache useful.

This module is only imported when the caller asks for it; the ``openai``
package is an optional extra so the core anamnesis install stays light.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any

from anamnesis._hashing import content_hash
from anamnesis.models.intent import (
    EntityMention,
    NarrativeIntent,
    RelationHint,
    TemporalScope,
)

PROMPT_VERSION = "openai-v1"

_SYSTEM_PROMPT = (
    "You decompose free-form essays into a structured retrieval intent for a "
    "knowledge-graph query engine. Identify entities, relation hints, temporal "
    "scope, cross-domain analogies, and a primary theme. If the essay is "
    "ambiguous (multiple equally plausible readings), surface them in "
    "alternative_interpretations and set ambiguity_score above 0.5. Never "
    "silently pick one interpretation."
)


_INTENT_SCHEMA: dict[str, Any] = {
    "name": "NarrativeIntentPayload",
    "strict": True,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "primary_theme": {"type": "string"},
            "entities": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "surface": {"type": "string"},
                        "canonical_guess": {"type": ["string", "null"]},
                        "kg_ground_hint": {"type": ["string", "null"]},
                        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                    },
                    "required": [
                        "surface",
                        "canonical_guess",
                        "kg_ground_hint",
                        "confidence",
                    ],
                },
            },
            "relations": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "predicate": {"type": "string"},
                        "subject_ref": {"type": "string"},
                        "object_ref": {"type": "string"},
                        "fuzzy_temporal_label": {"type": ["string", "null"]},
                        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                    },
                    "required": [
                        "predicate",
                        "subject_ref",
                        "object_ref",
                        "fuzzy_temporal_label",
                        "confidence",
                    ],
                },
            },
            "temporal_fuzzy_label": {"type": ["string", "null"]},
            "analogy_targets": {"type": "array", "items": {"type": "string"}},
            "output_type_hint": {
                "type": "string",
                "enum": ["timeline", "network", "article", "map", "auto"],
            },
            "alternative_interpretations": {
                "type": "array",
                "items": {"type": "string"},
            },
            "ambiguity_score": {"type": "number", "minimum": 0, "maximum": 1},
        },
        "required": [
            "primary_theme",
            "entities",
            "relations",
            "temporal_fuzzy_label",
            "analogy_targets",
            "output_type_hint",
            "alternative_interpretations",
            "ambiguity_score",
        ],
    },
}


class OpenAIDecomposer:
    """Live, OpenAI-backed decomposer."""

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
                "OpenAIDecomposer requires the 'openai' package. "
                "Install with: pip install 'anamnesis[openai]'"
            ) from exc
        self._client = OpenAI(api_key=api_key, max_retries=max_retries, timeout=timeout)
        self._model = model or os.getenv("ANAMNESIS_OPENAI_MODEL", "gpt-4o-mini")

    def decompose(
        self,
        essay: str,
        *,
        schema_hints: list[str] | None = None,
    ) -> NarrativeIntent:
        user = essay
        if schema_hints:
            user += "\n\nSCHEMA HINTS:\n" + "\n".join(f"- {h}" for h in schema_hints)

        resp = self._client.responses.create(
            model=self._model,
            input=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user},
            ],
            text={"format": {"type": "json_schema", **_INTENT_SCHEMA}},
            temperature=0,
        )

        text = resp.output_text or resp.output[0].content[0].text
        payload = json.loads(text)

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
