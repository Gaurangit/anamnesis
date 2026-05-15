"""Pydantic data structures for the narrative-query pipeline.

Every model inherits :class:`ConfiguredBaseModel`, mirroring the
``meta.ko_model`` pattern used elsewhere in Chariot.
"""

from anamnesis.models.base import ConfiguredBaseModel
from anamnesis.models.critique import CritiqueReport, OverreachFlag
from anamnesis.models.intent import (
    EntityMention,
    NarrativeIntent,
    RelationHint,
    TemporalScope,
)
from anamnesis.models.output import ClarificationRequest, QueryOutput
from anamnesis.models.plan import RetrievalPlan, SubQuery
from anamnesis.models.subgraph import (
    Contradiction,
    RetrievedEdge,
    RetrievedNode,
    RetrievedSubgraph,
    SourceEvidence,
)
from anamnesis.models.synthesis import BridgedInference, Claim, SynthesisResult

__all__ = [
    "BridgedInference",
    "Claim",
    "ClarificationRequest",
    "ConfiguredBaseModel",
    "Contradiction",
    "CritiqueReport",
    "EntityMention",
    "NarrativeIntent",
    "OverreachFlag",
    "QueryOutput",
    "RelationHint",
    "RetrievalPlan",
    "RetrievedEdge",
    "RetrievedNode",
    "RetrievedSubgraph",
    "SourceEvidence",
    "SubQuery",
    "SynthesisResult",
    "TemporalScope",
]
