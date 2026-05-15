"""Deterministic StubSynthesizer.

Produces a fully grounded :class:`SynthesisResult` from any
(intent, subgraph) pair by stitching together one claim per retrieved
edge plus one summary claim per principal entity. Used in tests and as
the offline fallback for the eval harness.
"""

from __future__ import annotations

from datetime import datetime, timezone

from anamnesis._hashing import content_hash
from anamnesis.models.intent import NarrativeIntent
from anamnesis.models.subgraph import RetrievedSubgraph
from anamnesis.models.synthesis import BridgedInference, Claim, SynthesisResult

PROMPT_VERSION = "stub-v1"


class StubSynthesizer:
    """Offline, deterministic synthesizer."""

    name = "stub"
    prompt_version = PROMPT_VERSION

    def synthesize(
        self,
        intent: NarrativeIntent,
        subgraph: RetrievedSubgraph,
        lens: str,
    ) -> SynthesisResult:
        claims: list[Claim] = []
        provenance: dict[str, list[str]] = {}

        ko_ids_in_subgraph = {n.ko_id for n in subgraph.nodes}

        for i, edge in enumerate(subgraph.edges):
            supporting = [edge.subject_ko, edge.object_ko]
            supporting = [k for k in supporting if k in ko_ids_in_subgraph]
            if not supporting:
                continue
            claims.append(
                Claim(
                    text=f"{edge.subject_ko} {edge.predicate} {edge.object_ko}.",
                    kind="grounded",
                    supporting_kos=supporting,
                    confidence=0.7,
                )
            )
            provenance[str(len(claims) - 1)] = list(supporting)

        # One summary claim per high-scoring node, if we have any.
        for node in sorted(subgraph.nodes, key=lambda n: -n.score)[:3]:
            claims.append(
                Claim(
                    text=f"The corpus contains evidence about {node.ko_id}.",
                    kind="grounded",
                    supporting_kos=[node.ko_id],
                    confidence=max(0.3, min(0.9, node.score)),
                )
            )
            provenance[str(len(claims) - 1)] = [node.ko_id]

        bridged: list[BridgedInference] = []
        if intent.analogy_targets:
            bridged.append(
                BridgedInference(
                    text=(
                        f"By analogy to {', '.join(intent.analogy_targets)}, similar "
                        f"patterns may apply to {intent.primary_theme}."
                    ),
                    supporting_kos=[],
                    confidence=0.4,
                )
            )

        return SynthesisResult(
            synthesis_id=content_hash(
                f"syn::{subgraph.subgraph_id}::{lens}::{self.name}"
            ),
            subgraph_id=subgraph.subgraph_id,
            lens=lens,
            primary_artifact={
                "lens": lens,
                "summary": intent.primary_theme,
                "node_count": len(subgraph.nodes),
                "edge_count": len(subgraph.edges),
            },
            claims=claims,
            bridged_inferences=bridged,
            provenance_map=provenance,
            created_at=datetime.now(tz=timezone.utc),
            synthesizer_model=f"{self.name}:0",
        )
