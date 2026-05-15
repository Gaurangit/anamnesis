"""Deterministic StubCritic.

Flags any claim whose ``supporting_kos`` set is contradicted by an edge
in the complement subgraph (same subject, same predicate as the claim
references but different object). Used in tests and as the offline
fallback for the eval harness.
"""

from __future__ import annotations

from anamnesis._hashing import content_hash
from anamnesis.models.critique import CritiqueReport, OverreachFlag
from anamnesis.models.subgraph import RetrievedSubgraph, SourceEvidence
from anamnesis.models.synthesis import SynthesisResult

PROMPT_VERSION = "stub-v1"


class StubCritic:
    name = "stub"
    prompt_version = PROMPT_VERSION

    def critique(
        self,
        synthesis: SynthesisResult,
        complement_subgraph: RetrievedSubgraph,
    ) -> CritiqueReport:
        flags: list[OverreachFlag] = []
        delta: dict[int, float] = {}

        complement_ko_ids = {n.ko_id for n in complement_subgraph.nodes}
        complement_edges = {
            (e.subject_ko, e.predicate, e.object_ko) for e in complement_subgraph.edges
        }

        for idx, claim in enumerate(synthesis.claims):
            cited = set(claim.supporting_kos)
            # Heuristic 1: contradiction sources mention the same ko_id as
            # one we cite — the complement subgraph caught something.
            for contradiction in complement_subgraph.contradictions:
                contradiction_kos = {ev.ko_id for ev in contradiction.sources}
                overlap = cited & contradiction_kos
                if overlap:
                    flags.append(
                        OverreachFlag(
                            claim_idx=idx,
                            reason=(
                                f"Claim cites {sorted(overlap)} but complement "
                                f"subgraph reports contradiction at {contradiction.locus}."
                            ),
                            contradicting_evidence=list(contradiction.sources),
                        )
                    )
                    delta[idx] = -0.3
                    break

            # Heuristic 2: claim cites only ko_ids the complement subgraph
            # also retrieved with inverted predicates. Soft flag, no demotion.
            if idx in delta:
                continue
            if cited and cited.issubset(complement_ko_ids):
                flags.append(
                    OverreachFlag(
                        claim_idx=idx,
                        reason=(
                            "Complement subgraph found inverse-polarity evidence "
                            "from the same ko_ids — consider hedging."
                        ),
                        contradicting_evidence=[
                            SourceEvidence(ko_id=k) for k in sorted(cited)[:2]
                        ],
                    )
                )
                delta[idx] = -0.15

        alternative_framings: list[str] = []
        if complement_subgraph.contradictions:
            alternative_framings.append(
                "Consider acknowledging the disagreement explicitly in the synthesis."
            )
        if complement_subgraph.gaps:
            alternative_framings.append(
                "Note in the summary that the complement search found "
                f"{len(complement_subgraph.gaps)} additional gap(s)."
            )

        return CritiqueReport(
            critique_id=content_hash(
                f"crit::{synthesis.synthesis_id}::{complement_subgraph.subgraph_id}"
            ),
            synthesis_id=synthesis.synthesis_id,
            complement_subgraph_id=complement_subgraph.subgraph_id,
            overreach_flags=flags,
            alternative_framings=alternative_framings,
            confidence_delta=delta,
        )
