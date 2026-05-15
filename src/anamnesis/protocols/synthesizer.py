"""Synthesizer protocol — (intent, subgraph, lens) → SynthesisResult."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from anamnesis.models.intent import NarrativeIntent
from anamnesis.models.subgraph import RetrievedSubgraph
from anamnesis.models.synthesis import SynthesisResult


@runtime_checkable
class Synthesizer(Protocol):
    """Render a RetrievedSubgraph into a lens-specific SynthesisResult.

    Every Claim emitted must either declare ``supporting_kos`` (when
    ``kind='grounded'``) or be marked ``kind='inference'``. The pipeline
    will reject violations downstream.
    """

    name: str
    prompt_version: str

    def synthesize(
        self,
        intent: NarrativeIntent,
        subgraph: RetrievedSubgraph,
        lens: str,
    ) -> SynthesisResult: ...
