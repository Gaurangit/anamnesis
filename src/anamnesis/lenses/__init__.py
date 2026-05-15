"""Lens implementations + the deterministic ``pick_lens`` selector.

Per ADR-0003: every lens always emits a structured JSON payload; SVG
is emitted in addition when the lens supports it.
"""

from anamnesis.lenses.base import Lens
from anamnesis.lenses.disagreement import DisagreementLens
from anamnesis.lenses.network import NetworkLens
from anamnesis.lenses.picker import pick_lens
from anamnesis.lenses.sankey import SankeyLens
from anamnesis.lenses.timeline import TimelineLens

__all__ = [
    "DisagreementLens",
    "Lens",
    "NetworkLens",
    "SankeyLens",
    "TimelineLens",
    "pick_lens",
]
