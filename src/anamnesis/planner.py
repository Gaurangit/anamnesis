"""RetrievalPlanner — deterministic NarrativeIntent → RetrievalPlan.

Pure logic, no LLM. Each entity becomes a ``vector_hybrid`` subquery and
each relation hint becomes a ``sparql`` subquery. Entities that look like
they need external grounding (no canonical guess or no KG hint) get an
additional ``kg_bridge_lookup`` subquery scheduled as a fallback.

The planner also exposes :meth:`complement_plan` which generates the
inverse plan used by the adversarial critic (Phase 3): same entities,
broader temporal scope, polarity-inverted relations.
"""

from __future__ import annotations

import re
from copy import deepcopy

from anamnesis._hashing import content_hash
from anamnesis.models.intent import NarrativeIntent, RelationHint
from anamnesis.models.plan import RetrievalPlan, SubQuery

_POLARITY_INVERSIONS: dict[str, str] = {
    "challenged": "supported",
    "supported": "challenged",
    "refuted": "confirmed",
    "confirmed": "refuted",
    "extended": "narrowed",
    "narrowed": "extended",
    "influenced": "was_uninfluenced_by",
    "investigated": "ignored",
    "studied": "ignored",
}


def _invert_predicate(predicate: str) -> str:
    """Return the polarity-inverted predicate for the complement plan.

    Falls back to a ``not_`` prefix if no known inversion exists.
    """
    key = predicate.lower()
    if key in _POLARITY_INVERSIONS:
        return _POLARITY_INVERSIONS[key]
    return f"not_{key}"


class RetrievalPlanner:
    """Builds a deterministic RetrievalPlan from a NarrativeIntent."""

    name = "default"

    def __init__(
        self,
        *,
        max_nodes_per_entity: int = 20,
        max_nodes_per_relation: int = 15,
        expansion_depth: int = 1,
        total_budget: int = 200,
    ) -> None:
        self._max_nodes_per_entity = max_nodes_per_entity
        self._max_nodes_per_relation = max_nodes_per_relation
        self._expansion_depth = expansion_depth
        self._total_budget = total_budget

    def plan(self, intent: NarrativeIntent) -> RetrievalPlan:
        subqueries: list[SubQuery] = []
        idx = 0

        # 1. Entity subqueries — vector_hybrid then optional kg_bridge_lookup.
        for entity in intent.entities:
            idx += 1
            subqueries.append(
                SubQuery(
                    subquery_id=f"sq:{idx}",
                    target=entity,
                    method="vector_hybrid",
                    params={"query": entity.surface, "k": 5},
                    expansion_depth=self._expansion_depth,
                    max_nodes=self._max_nodes_per_entity,
                )
            )
            if not entity.kg_ground_hint and not entity.canonical_guess:
                idx += 1
                subqueries.append(
                    SubQuery(
                        subquery_id=f"sq:{idx}",
                        target=entity,
                        method="kg_bridge_lookup",
                        params={"query": entity.surface, "limit": 5},
                        expansion_depth=0,
                        max_nodes=self._max_nodes_per_entity,
                    )
                )

        # 2. Relation subqueries — always SPARQL against the rdflib graph.
        for relation in intent.relations:
            idx += 1
            subqueries.append(
                SubQuery(
                    subquery_id=f"sq:{idx}",
                    target=relation,
                    method="sparql",
                    params={
                        "subject_surface": relation.subject_ref,
                        "predicate": relation.predicate,
                        "object_surface": relation.object_ref,
                    },
                    expansion_depth=0,
                    max_nodes=self._max_nodes_per_relation,
                )
            )

        # 3. Temporal scope alone never triggers retrieval — it's a filter.
        plan = RetrievalPlan(
            plan_id=content_hash(f"plan::{intent.intent_id}"),
            intent_id=intent.intent_id,
            subqueries=subqueries,
            total_budget=self._total_budget,
        )
        return plan

    def complement_plan(self, intent: NarrativeIntent) -> RetrievalPlan:
        """Build the inverse plan used by :class:`AdversarialCritic`.

        Steps:

        1. Same entity subqueries (we want the same neighbourhood).
        2. Relations get polarity-inverted predicates.
        3. Temporal scope broadens by ±10 years (or stays None if not set).
        """
        complement_intent = deepcopy(intent)
        for relation in complement_intent.relations:
            relation.predicate = _invert_predicate(relation.predicate)

        ts = complement_intent.temporal_scope
        if ts and (ts.start or ts.end):
            if ts.start:
                ts.start = ts.start.replace(year=ts.start.year - 10)
            if ts.end:
                ts.end = ts.end.replace(year=ts.end.year + 10)

        base_plan = self.plan(complement_intent)
        return RetrievalPlan(
            plan_id=content_hash(f"plan-complement::{intent.intent_id}"),
            intent_id=intent.intent_id,
            subqueries=base_plan.subqueries,
            total_budget=base_plan.total_budget,
            is_complement=True,
        )


_SAFE_ID_RE = re.compile(r"[^a-zA-Z0-9_-]+")


def slugify(value: str) -> str:
    """Helper exposed for test fixtures and CLI scripts."""
    return _SAFE_ID_RE.sub("_", value.strip()).lower()
