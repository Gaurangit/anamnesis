"""Lens protocol — (synthesis, subgraph) → lens-specific artifact dict."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from anamnesis.models.subgraph import RetrievedSubgraph
from anamnesis.models.synthesis import SynthesisResult


@runtime_checkable
class Lens(Protocol):
    """Render a SynthesisResult into a lens-shaped payload.

    Lenses are pure functions over the (synthesis, subgraph) pair. They
    must never call back into the executor or the KO registry — see the
    "do not couple lenses to retrieval" anti-pattern in the spec.
    """

    name: str

    def render(
        self,
        synthesis: SynthesisResult,
        subgraph: RetrievedSubgraph,
    ) -> dict[str, Any]: ...
