"""Shared scaffolding for lens implementations.

Lenses must be **pure functions** of (synthesis, subgraph). They MUST
NOT call back into the executor or the KO registry — see anti-pattern
§10 in the spec.
"""

from __future__ import annotations

from typing import Any

from anamnesis.models.subgraph import RetrievedSubgraph
from anamnesis.models.synthesis import SynthesisResult


class Lens:
    """Optional base class for lens implementations.

    Concrete lenses don't need to inherit (the Protocol contract is what
    matters) but inheriting gets you ``name`` defaulting and the standard
    artifact envelope from :meth:`_envelope`.
    """

    name: str = "base"

    def render(
        self,
        synthesis: SynthesisResult,
        subgraph: RetrievedSubgraph,
    ) -> dict[str, Any]:
        raise NotImplementedError

    @staticmethod
    def _envelope(
        synthesis: SynthesisResult,
        subgraph: RetrievedSubgraph,
        lens_name: str,
        body: dict[str, Any],
        *,
        svg: str | None = None,
    ) -> dict[str, Any]:
        """Standard outer shape every lens emits.

        Downstream renderers can rely on ``json.body`` being the
        lens-specific payload and ``svg`` being present only when the
        lens chose to render one.
        """
        envelope: dict[str, Any] = {
            "lens": lens_name,
            "synthesis_id": synthesis.synthesis_id,
            "subgraph_id": subgraph.subgraph_id,
            "claim_count": len(synthesis.claims),
            "inference_count": len(synthesis.bridged_inferences),
            "body": body,
        }
        if svg is not None:
            envelope["svg"] = svg
        return envelope
