"""Sankey lens — bucket flows of (predicate → object) and (subject → predicate).

Structured JSON only (no SVG); downstream code (e.g. observable plot or
d3-sankey) takes the JSON and renders.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

from anamnesis.lenses.base import Lens
from anamnesis.models.subgraph import RetrievedSubgraph
from anamnesis.models.synthesis import SynthesisResult


class SankeyLens(Lens):
    name = "sankey"

    def render(
        self,
        synthesis: SynthesisResult,
        subgraph: RetrievedSubgraph,
    ) -> dict[str, Any]:
        # Two-stage Sankey: subject → predicate → object.
        # Each link's value is the count of edges that flow through it.
        subj_to_pred: Counter[tuple[str, str]] = Counter()
        pred_to_obj: Counter[tuple[str, str]] = Counter()
        for edge in subgraph.edges:
            subj_to_pred[(edge.subject_ko, edge.predicate)] += 1
            pred_to_obj[(edge.predicate, edge.object_ko)] += 1

        nodes = sorted(
            {e.subject_ko for e in subgraph.edges}
            | {e.predicate for e in subgraph.edges}
            | {e.object_ko for e in subgraph.edges}
        )
        node_index = {name: i for i, name in enumerate(nodes)}
        links = []
        for (src, dst), value in sorted(subj_to_pred.items()):
            links.append({"source": node_index[src], "target": node_index[dst], "value": value})
        for (src, dst), value in sorted(pred_to_obj.items()):
            links.append({"source": node_index[src], "target": node_index[dst], "value": value})

        body: dict[str, Any] = {
            "nodes": [{"name": name} for name in nodes],
            "links": links,
        }
        return self._envelope(synthesis, subgraph, self.name, body)
