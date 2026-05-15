"""RetrievalPlan and SubQuery.

A plan is content-addressed by ``plan_id = hash(intent)`` and is the
deterministic output of :class:`anamnesis.planner.RetrievalPlanner`.
"""

from __future__ import annotations

from typing import Any, Literal, Union

from pydantic import Field

from anamnesis.models.base import ConfiguredBaseModel
from anamnesis.models.intent import EntityMention, RelationHint, TemporalScope

SubQueryMethod = Literal["vector_hybrid", "sparql", "kg_bridge_lookup", "ko_id_direct"]


class SubQuery(ConfiguredBaseModel):
    subquery_id: str
    target: Union[EntityMention, RelationHint, TemporalScope]
    method: SubQueryMethod
    params: dict[str, Any] = Field(default_factory=dict)
    expansion_depth: int = Field(
        0, ge=0, description="How far to walk refersTo edges from initial hits."
    )
    max_nodes: int = Field(50, gt=0, description="Cost cap per subquery.")


class RetrievalPlan(ConfiguredBaseModel):
    plan_id: str = Field(..., description="content_hash(intent)")
    intent_id: str
    subqueries: list[SubQuery] = Field(default_factory=list)
    total_budget: int = Field(
        200, gt=0, description="Hard cap on KO loads across all subqueries."
    )
    fallback_chain: list[SubQueryMethod] = Field(
        default_factory=lambda: ["vector_hybrid", "sparql", "kg_bridge_lookup"]
    )
    is_complement: bool = Field(
        False,
        description="True when this plan was produced by complement_plan() for the critic.",
    )
