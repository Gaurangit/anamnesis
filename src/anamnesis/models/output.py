"""QueryOutput and ClarificationRequest — terminal pipeline shapes."""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from anamnesis.models.base import ConfiguredBaseModel
from anamnesis.models.critique import CritiqueReport
from anamnesis.models.intent import NarrativeIntent
from anamnesis.models.plan import RetrievalPlan
from anamnesis.models.subgraph import RetrievedSubgraph
from anamnesis.models.synthesis import SynthesisResult


class ClarificationRequest(ConfiguredBaseModel):
    """Returned in lieu of synthesis when ``intent.ambiguity_score > 0.5``."""

    intent_id: str
    alternatives: list[str] = Field(
        default_factory=list,
        description="Copy of NarrativeIntent.alternative_interpretations.",
    )
    ambiguity_score: float
    prompt: str = Field(
        ...,
        description="Human-readable question the caller should ask the user.",
    )


class QueryOutput(ConfiguredBaseModel):
    kind: Literal["synthesis", "clarification"] = "synthesis"
    intent: NarrativeIntent
    plan: RetrievalPlan | None = None
    subgraph: RetrievedSubgraph | None = None
    synthesis: SynthesisResult | None = None
    critique: CritiqueReport | None = None
    clarification: ClarificationRequest | None = None

    @classmethod
    def clarification_needed(cls, intent: NarrativeIntent) -> QueryOutput:
        prompt = (
            "The essay can be interpreted in multiple ways. Pick one to continue:\n"
            + "\n".join(f"- {alt}" for alt in intent.alternative_interpretations)
        )
        return cls(
            kind="clarification",
            intent=intent,
            clarification=ClarificationRequest(
                intent_id=intent.intent_id,
                alternatives=list(intent.alternative_interpretations),
                ambiguity_score=intent.ambiguity_score,
                prompt=prompt,
            ),
        )
