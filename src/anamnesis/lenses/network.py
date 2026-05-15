"""Network lens — node-link diagram over RetrievedSubgraph edges."""

from __future__ import annotations

import math
from typing import Any

from anamnesis.lenses.base import Lens
from anamnesis.models.subgraph import RetrievedSubgraph
from anamnesis.models.synthesis import SynthesisResult


def _layout(node_ids: list[str], width: int = 800, height: int = 600) -> dict[str, tuple[float, float]]:
    """Deterministic radial layout — same input always yields same coords."""
    if not node_ids:
        return {}
    cx, cy = width / 2, height / 2
    radius = min(width, height) / 2 - 40
    positions: dict[str, tuple[float, float]] = {}
    n = len(node_ids)
    for i, ko_id in enumerate(node_ids):
        angle = (2 * math.pi * i) / n
        positions[ko_id] = (cx + radius * math.cos(angle), cy + radius * math.sin(angle))
    return positions


def _render_svg(
    node_ids: list[str],
    edges: list[tuple[str, str, str]],
    positions: dict[str, tuple[float, float]],
) -> str:
    parts = ['<svg xmlns="http://www.w3.org/2000/svg" width="800" height="600" viewBox="0 0 800 600">']
    for subj, _pred, obj in edges:
        if subj not in positions or obj not in positions:
            continue
        sx, sy = positions[subj]
        ex, ey = positions[obj]
        parts.append(
            f'<line x1="{sx:.1f}" y1="{sy:.1f}" x2="{ex:.1f}" y2="{ey:.1f}" '
            'stroke="#888" stroke-width="1"/>'
        )
    for ko_id, (x, y) in positions.items():
        parts.append(
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="6" fill="#0a7"/>'
            f'<text x="{x:.1f}" y="{y - 10:.1f}" font-size="10" text-anchor="middle">{ko_id}</text>'
        )
    parts.append("</svg>")
    return "".join(parts)


class NetworkLens(Lens):
    name = "network"

    def render(
        self,
        synthesis: SynthesisResult,
        subgraph: RetrievedSubgraph,
    ) -> dict[str, Any]:
        node_ids = sorted({n.ko_id for n in subgraph.nodes})
        positions = _layout(node_ids)
        edges = [(e.subject_ko, e.predicate, e.object_ko) for e in subgraph.edges]
        body: dict[str, Any] = {
            "nodes": [
                {
                    "id": ko_id,
                    "x": round(positions[ko_id][0], 2),
                    "y": round(positions[ko_id][1], 2),
                }
                for ko_id in node_ids
            ],
            "edges": [
                {"source": s, "target": o, "predicate": p} for s, p, o in edges
            ],
        }
        return self._envelope(
            synthesis,
            subgraph,
            self.name,
            body,
            svg=_render_svg(node_ids, edges, positions),
        )
