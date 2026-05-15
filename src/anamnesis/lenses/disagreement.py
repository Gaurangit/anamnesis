"""Disagreement lens — explicitly renders contradictions and source-attribution diffs.

Emits both JSON (one record per contradiction with source breakdown)
and an SVG that lists each contradiction as a horizontal bar split by
attribution.
"""

from __future__ import annotations

from typing import Any

from anamnesis.lenses.base import Lens
from anamnesis.models.subgraph import RetrievedSubgraph
from anamnesis.models.synthesis import SynthesisResult


def _render_svg(records: list[dict[str, Any]]) -> str:
    if not records:
        return (
            '<svg xmlns="http://www.w3.org/2000/svg" width="600" height="40">'
            '<text x="20" y="24" font-size="12">No contradictions detected.</text></svg>'
        )
    row_h = 32
    height = row_h * len(records) + 20
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="600" height="{height}" '
        f'viewBox="0 0 600 {height}">',
    ]
    for i, record in enumerate(records):
        y = 20 + i * row_h
        parts.append(
            f'<text x="10" y="{y + 14}" font-size="11" fill="#c33">{record["locus"]}</text>'
            f'<text x="180" y="{y + 14}" font-size="10" fill="#333">'
            f'{record["description"][:80]}</text>'
        )
    parts.append("</svg>")
    return "".join(parts)


class DisagreementLens(Lens):
    name = "disagreement"

    def render(
        self,
        synthesis: SynthesisResult,
        subgraph: RetrievedSubgraph,
    ) -> dict[str, Any]:
        records: list[dict[str, Any]] = []
        for c in subgraph.contradictions:
            records.append(
                {
                    "locus": c.locus,
                    "description": c.description,
                    "sources": [
                        {
                            "ko_id": ev.ko_id,
                            "page_or_offset": ev.page_or_offset,
                            "source_uri": ev.source_uri,
                        }
                        for ev in c.sources
                    ],
                }
            )
        body: dict[str, Any] = {
            "contradictions": records,
            "contradiction_count": len(records),
        }
        return self._envelope(
            synthesis,
            subgraph,
            self.name,
            body,
            svg=_render_svg(records),
        )
