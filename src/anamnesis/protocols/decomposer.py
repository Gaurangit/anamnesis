"""NarrativeDecomposer protocol — essay → NarrativeIntent."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from anamnesis.models.intent import NarrativeIntent


@runtime_checkable
class NarrativeDecomposer(Protocol):
    """Convert a free-form essay into a structured retrieval intent.

    Implementations should set temperature=0 (or equivalent) and bump
    ``prompt_version`` whenever the prompt changes — Anamnesis caches
    Intents by (essay_hash, decomposer_model, prompt_version).
    """

    name: str
    prompt_version: str

    def decompose(
        self,
        essay: str,
        *,
        schema_hints: list[str] | None = None,
    ) -> NarrativeIntent: ...
