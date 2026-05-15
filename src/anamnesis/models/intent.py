"""NarrativeIntent and supporting types.

Output of :class:`anamnesis.protocols.decomposer.NarrativeDecomposer`.
Content-addressed by ``intent_id = hash(essay)`` so caches can short-circuit
on identical inputs.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Literal

from pydantic import Field

from anamnesis.models.base import ConfiguredBaseModel

OutputTypeHint = Literal["timeline", "network", "article", "map", "auto"]


class TemporalScope(ConfiguredBaseModel):
    start: date | None = None
    end: date | None = None
    fuzzy_label: str | None = Field(None, description="e.g. 'early 1950s'")


class EntityMention(ConfiguredBaseModel):
    surface: str
    canonical_guess: str | None = None
    kg_ground_hint: str | None = Field(
        None, description="Obvious KG id hint, e.g. 'wd:Q11651'."
    )
    confidence: float = Field(0.0, ge=0.0, le=1.0)


class RelationHint(ConfiguredBaseModel):
    predicate: str = Field(..., description="KO relation vocabulary term.")
    subject_ref: str
    object_ref: str
    temporal_qualifier: TemporalScope | None = None
    confidence: float = Field(0.0, ge=0.0, le=1.0)


class NarrativeIntent(ConfiguredBaseModel):
    intent_id: str = Field(..., description="content_hash(essay)")
    essay_hash: str
    decomposer_model: str = Field(..., description="e.g. 'openai:gpt-5'")
    prompt_version: str = Field(..., description="Bump to invalidate caches.")
    primary_theme: str
    entities: list[EntityMention] = Field(default_factory=list)
    relations: list[RelationHint] = Field(default_factory=list)
    temporal_scope: TemporalScope | None = None
    analogy_targets: list[str] = Field(default_factory=list)
    output_type_hint: OutputTypeHint = "auto"
    alternative_interpretations: list[str] = Field(default_factory=list)
    ambiguity_score: float = Field(0.0, ge=0.0, le=1.0)
    created_at: datetime
