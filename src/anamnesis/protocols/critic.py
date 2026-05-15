"""AdversarialCritic protocol — second-pass evidence check."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from anamnesis.models.critique import CritiqueReport
from anamnesis.models.subgraph import RetrievedSubgraph
from anamnesis.models.synthesis import SynthesisResult


@runtime_checkable
class AdversarialCritic(Protocol):
    """Compare a synthesis against a complement subgraph and flag overreach.

    The pipeline calls :meth:`critique` with a complement subgraph fetched
    by inverting relation polarity and broadening the temporal scope. The
    critic identifies claims that the complement evidence contradicts or
    that warrant a softer phrasing.
    """

    name: str
    prompt_version: str

    def critique(
        self,
        synthesis: SynthesisResult,
        complement_subgraph: RetrievedSubgraph,
    ) -> CritiqueReport: ...
