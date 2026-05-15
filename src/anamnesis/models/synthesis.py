"""SynthesisResult, Claim, and BridgedInference.

A grounded Claim must declare at least one supporting KO id; the
``apply_critique`` helper hard-fails if it observes a grounded claim
without provenance.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import Field, field_validator

from anamnesis.models.base import ConfiguredBaseModel

ClaimKind = Literal["grounded", "inference"]


class Claim(ConfiguredBaseModel):
    text: str
    kind: ClaimKind = "grounded"
    supporting_kos: list[str] = Field(
        default_factory=list,
        description="Required when kind == 'grounded'.",
    )
    confidence: float = Field(0.0, ge=0.0, le=1.0)

    @field_validator("supporting_kos")
    @classmethod
    def _grounded_requires_support(cls, v: list[str], info: Any) -> list[str]:
        kind = info.data.get("kind", "grounded")
        if kind == "grounded" and not v:
            raise ValueError(
                "Grounded claims must declare at least one supporting_kos entry. "
                "If unsupported, mark kind='inference'."
            )
        return v


class BridgedInference(Claim):
    """A Claim with ``kind='inference'``.

    The pipeline emits these separately from grounded claims so lenses
    can render them with visually distinct treatment.
    """

    kind: ClaimKind = "inference"


class SynthesisResult(ConfiguredBaseModel):
    synthesis_id: str
    subgraph_id: str
    lens: str = Field(..., description="Lens this artifact is shaped for.")
    primary_artifact: dict[str, Any] = Field(
        default_factory=dict,
        description="Lens-specific structured payload (SVG/HTML/JSON).",
    )
    claims: list[Claim] = Field(default_factory=list)
    bridged_inferences: list[BridgedInference] = Field(default_factory=list)
    provenance_map: dict[str, list[str]] = Field(
        default_factory=dict,
        description="Maps claim index (as str) to list of KO ids that justify it.",
    )
    created_at: datetime
    synthesizer_model: str
