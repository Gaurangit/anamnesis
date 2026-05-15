"""Deterministic lens selector — ``pick_lens(hint, subgraph) -> lens_name``."""

from __future__ import annotations

import re

from anamnesis.models.subgraph import RetrievedSubgraph

_YEAR_RE = re.compile(r"\b(1\d{3}|20\d{2})\b")


def _fraction_with_temporal_data(subgraph: RetrievedSubgraph) -> float:
    if not subgraph.nodes:
        return 0.0
    # Use edge source_evidence excerpts as a temporal proxy.
    temporal_hits = 0
    total = max(len(subgraph.nodes), 1)
    for edge in subgraph.edges:
        for ev in edge.source_evidence:
            if ev.excerpt and _YEAR_RE.search(ev.excerpt):
                temporal_hits += 1
                break
    return temporal_hits / total


def pick_lens(hint: str, subgraph: RetrievedSubgraph) -> str:
    """Return the lens name to use given an intent hint and a subgraph.

    Priority order:

    1. Explicit hints (``timeline``, ``network``, ``article``, ``map``)
       are honoured unless the subgraph contradicts them strongly.
    2. With ``auto``: contradictions push us to ``disagreement``,
       temporal density pushes to ``timeline``, edge density to
       ``network``, otherwise fall back to ``sankey``.
    """
    if hint == "timeline":
        return "timeline"
    if hint == "network":
        return "network"
    if hint == "map":
        # No map lens implemented yet — fall back to network.
        return "network"
    if hint == "article":
        # Article-mode renders structured text; ship as network with
        # claims attached so the consumer can flatten to prose.
        return "network"

    # hint == "auto"
    if subgraph.contradictions:
        return "disagreement"
    if _fraction_with_temporal_data(subgraph) >= 0.3:
        return "timeline"
    if len(subgraph.edges) >= max(2, len(subgraph.nodes) // 2):
        return "network"
    return "sankey"
