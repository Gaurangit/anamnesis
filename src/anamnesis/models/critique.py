"""CritiqueReport — output of the adversarial critic.

Lists overreach flags (claims contradicted by the complement subgraph)
and suggested confidence adjustments. Applied by
:func:`anamnesis.critic.apply_critique`.
"""

from __future__ import annotations

from pydantic import Field

from anamnesis.models.base import ConfiguredBaseModel
from anamnesis.models.subgraph import SourceEvidence


class OverreachFlag(ConfiguredBaseModel):
    claim_idx: int = Field(..., ge=0)
    reason: str
    contradicting_evidence: list[SourceEvidence] = Field(default_factory=list)


class CritiqueReport(ConfiguredBaseModel):
    critique_id: str
    synthesis_id: str
    complement_subgraph_id: str
    overreach_flags: list[OverreachFlag] = Field(default_factory=list)
    alternative_framings: list[str] = Field(default_factory=list)
    confidence_delta: dict[int, float] = Field(
        default_factory=dict,
        description="claim_idx -> proposed confidence adjustment (signed).",
    )
