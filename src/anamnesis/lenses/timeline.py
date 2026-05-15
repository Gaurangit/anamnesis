"""Timeline lens — extracts dated events from claims and orders them.

Triggered when at least 30 % of claims mention a year. Emits both a
JSON event array and a deterministic SVG strip.
"""

from __future__ import annotations

import re
from typing import Any

from anamnesis.lenses.base import Lens
from anamnesis.models.subgraph import RetrievedSubgraph
from anamnesis.models.synthesis import Claim, SynthesisResult

_YEAR_RE = re.compile(r"\b(1\d{3}|20\d{2})\b")


def _extract_year(text: str) -> int | None:
    match = _YEAR_RE.search(text)
    return int(match.group(0)) if match else None


def _events(claims: list[Claim]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for idx, claim in enumerate(claims):
        year = _extract_year(claim.text)
        if year is None:
            continue
        out.append(
            {
                "year": year,
                "text": claim.text,
                "claim_idx": idx,
                "supporting_kos": list(claim.supporting_kos),
                "confidence": claim.confidence,
            }
        )
    out.sort(key=lambda e: (e["year"], e["claim_idx"]))
    return out


def _render_svg(events: list[dict[str, Any]]) -> str:
    if not events:
        return '<svg xmlns="http://www.w3.org/2000/svg" width="100" height="40" />'
    width = 800
    margin = 40
    span = max(events[-1]["year"] - events[0]["year"], 1)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="160" '
        f'viewBox="0 0 {width} 160">',
        f'<line x1="{margin}" y1="80" x2="{width - margin}" y2="80" '
        'stroke="#222" stroke-width="1"/>',
    ]
    inner = width - 2 * margin
    for event in events:
        offset = ((event["year"] - events[0]["year"]) / span) * inner
        x = margin + offset
        parts.append(
            f'<circle cx="{x:.1f}" cy="80" r="4" fill="#0a7" />'
            f'<text x="{x:.1f}" y="68" font-size="10" text-anchor="middle">{event["year"]}</text>'
        )
    parts.append("</svg>")
    return "".join(parts)


class TimelineLens(Lens):
    name = "timeline"

    def render(
        self,
        synthesis: SynthesisResult,
        subgraph: RetrievedSubgraph,
    ) -> dict[str, Any]:
        events = _events(synthesis.claims)
        body: dict[str, Any] = {
            "events": events,
            "event_count": len(events),
            "earliest_year": events[0]["year"] if events else None,
            "latest_year": events[-1]["year"] if events else None,
        }
        return self._envelope(synthesis, subgraph, self.name, body, svg=_render_svg(events))
