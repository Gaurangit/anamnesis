"""PlanExecutor — turn a RetrievalPlan into a RetrievedSubgraph.

One method per :class:`SubQuery.method` value:

* ``vector_hybrid`` — :meth:`KOIndex.hybrid_search` over the entity surface.
* ``sparql``        — rdflib SPARQL searching KO labels for both ends
                       of the relation, then synthesising one edge per
                       (subject_ko, predicate, object_ko) match.
* ``kg_bridge_lookup`` — :func:`propose_kg_links` against a synthetic KO
                       dict; emitted KG ids are not nodes themselves
                       but enrich the ``retrieval_path`` of any KO they
                       help ground later in the pipeline.
* ``ko_id_direct`` — load a specified ko_id directly.

Gap detection: any entity in the intent whose subquery returned zero
nodes is recorded in ``subgraph.gaps``. Contradiction detection: two
edges sharing ``(subject_ko, predicate)`` but differing on ``object_ko``
are flagged via :class:`Contradiction`.
"""

from __future__ import annotations

import logging
from typing import Any

from anamnesis._hashing import content_hash
from anamnesis.models.intent import EntityMention, NarrativeIntent, RelationHint
from anamnesis.models.plan import RetrievalPlan, SubQuery
from anamnesis.models.subgraph import (
    Contradiction,
    RetrievedEdge,
    RetrievedNode,
    RetrievedSubgraph,
    SourceEvidence,
)
from anamnesis.runtime import AnamnesisRuntime

logger = logging.getLogger(__name__)


class PlanExecutor:
    """Execute a RetrievalPlan against an AnamnesisRuntime."""

    name = "default"

    def execute(
        self,
        plan: RetrievalPlan,
        runtime: AnamnesisRuntime,
        *,
        intent: NarrativeIntent | None = None,
    ) -> RetrievedSubgraph:
        nodes: dict[str, RetrievedNode] = {}
        edges: list[RetrievedEdge] = []
        loaded_budget = 0

        entity_hit_counts: dict[str, int] = {}
        relation_hit_counts: dict[str, int] = {}

        for sq in plan.subqueries:
            if loaded_budget >= plan.total_budget:
                logger.info("Plan %s hit total_budget=%d; stopping.", plan.plan_id, plan.total_budget)
                break

            method = sq.method
            target = sq.target

            try:
                if method == "vector_hybrid":
                    sq_nodes, sq_edges = self._exec_vector_hybrid(sq, runtime)
                elif method == "sparql":
                    sq_nodes, sq_edges = self._exec_sparql(sq, runtime)
                elif method == "kg_bridge_lookup":
                    sq_nodes, sq_edges = self._exec_kg_bridge(sq, runtime)
                elif method == "ko_id_direct":
                    sq_nodes, sq_edges = self._exec_direct(sq, runtime)
                else:
                    logger.warning("Unknown subquery method: %s", method)
                    continue
            except Exception as exc:  # noqa: BLE001 — surface as a gap, not a crash
                logger.warning("Subquery %s (%s) failed: %s", sq.subquery_id, method, exc)
                sq_nodes, sq_edges = [], []

            if isinstance(target, EntityMention):
                entity_hit_counts[target.surface] = entity_hit_counts.get(
                    target.surface, 0
                ) + len(sq_nodes)
            elif isinstance(target, RelationHint):
                key = f"{target.subject_ref}|{target.predicate}|{target.object_ref}"
                relation_hit_counts[key] = relation_hit_counts.get(key, 0) + len(sq_nodes)

            for n in sq_nodes:
                existing = nodes.get(n.ko_id)
                if existing is None or n.score > existing.score:
                    nodes[n.ko_id] = n
                loaded_budget += 1
                if loaded_budget >= plan.total_budget:
                    break
            edges.extend(sq_edges)

        gaps = self._detect_gaps(intent, entity_hit_counts, relation_hit_counts)
        contradictions = self._detect_contradictions(edges)

        return RetrievedSubgraph(
            subgraph_id=content_hash(
                f"{plan.plan_id}::{runtime.registry_revision}"
            ),
            plan_id=plan.plan_id,
            registry_revision=runtime.registry_revision,
            nodes=list(nodes.values()),
            edges=edges,
            gaps=gaps,
            contradictions=contradictions,
        )

    # ─── Method implementations ───────────────────────────────────────────

    def _exec_vector_hybrid(
        self,
        sq: SubQuery,
        runtime: AnamnesisRuntime,
    ) -> tuple[list[RetrievedNode], list[RetrievedEdge]]:
        query = sq.params.get("query")
        if not isinstance(query, str) or not query:
            return [], []
        k = int(sq.params.get("k", 5))
        hits = runtime.ko_index.hybrid_search(query, k=k)
        nodes: list[RetrievedNode] = []
        for hit in hits[: sq.max_nodes]:
            ko_id = getattr(hit, "id", None)
            score = float(getattr(hit, "score", 0.0))
            if not isinstance(ko_id, str):
                continue
            path = [sq.subquery_id, "vector_hybrid"]
            nodes.append(
                RetrievedNode(
                    ko_id=ko_id,
                    retrieval_path=path,
                    score=score,
                    via_subquery=sq.subquery_id,
                )
            )
        return nodes, []

    def _exec_sparql(
        self,
        sq: SubQuery,
        runtime: AnamnesisRuntime,
    ) -> tuple[list[RetrievedNode], list[RetrievedEdge]]:
        subject_surface = sq.params.get("subject_surface", "")
        object_surface = sq.params.get("object_surface", "")
        predicate = sq.params.get("predicate", "rel")
        if not subject_surface or not object_surface:
            return [], []

        # Find KOs whose rdfs:label contains the subject or object phrase.
        def _ids_matching(phrase: str) -> list[str]:
            query_str = """
                SELECT DISTINCT ?s WHERE {
                    ?s rdfs:label ?label .
                    FILTER(CONTAINS(LCASE(STR(?label)), LCASE(?needle)))
                }
            """
            try:
                rows = runtime.graph.query(
                    query_str,
                    initBindings={"needle": _Literal(phrase)},
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("SPARQL failed for %r: %s", phrase, exc)
                return []
            return [str(row[0]) for row in rows]

        subj_ids = _ids_matching(subject_surface)
        obj_ids = _ids_matching(object_surface)

        nodes: list[RetrievedNode] = []
        for ko_id in (*subj_ids, *obj_ids)[: sq.max_nodes]:
            nodes.append(
                RetrievedNode(
                    ko_id=ko_id,
                    retrieval_path=[sq.subquery_id, "sparql"],
                    score=0.7,
                    via_subquery=sq.subquery_id,
                )
            )

        edges: list[RetrievedEdge] = []
        for s in subj_ids:
            for o in obj_ids:
                if s == o:
                    continue
                edges.append(
                    RetrievedEdge(
                        subject_ko=s,
                        predicate=predicate,
                        object_ko=o,
                        source_evidence=[SourceEvidence(ko_id=s), SourceEvidence(ko_id=o)],
                    )
                )
        return nodes, edges

    def _exec_kg_bridge(
        self,
        sq: SubQuery,
        runtime: AnamnesisRuntime,
    ) -> tuple[list[RetrievedNode], list[RetrievedEdge]]:
        if runtime.kg_search is None:
            return [], []
        query = sq.params.get("query")
        if not isinstance(query, str) or not query:
            return [], []
        try:
            from runtime.kg_bridge.grounding import propose_kg_links
        except ImportError:
            logger.warning("kg_bridge unavailable — skipping subquery %s", sq.subquery_id)
            return [], []

        synthetic = {
            "ko_id": f"anamnesis:transient:{content_hash(query)[:12]}",
            "content": {"definition": query},
        }
        proposals = propose_kg_links(
            synthetic,
            search_client=runtime.kg_search,
            limit=int(sq.params.get("limit", 5)),
        )
        # KG proposals aren't KOs — they're external groundings. We
        # surface the *transient* node so downstream code can still
        # cite the lookup, but mark it clearly with a kg: prefix.
        nodes: list[RetrievedNode] = []
        for p in proposals[: sq.max_nodes]:
            nodes.append(
                RetrievedNode(
                    ko_id=f"kg:{p.kg_source}:{p.kg_id}",
                    retrieval_path=[sq.subquery_id, "kg_bridge_lookup"],
                    score=float(p.kg_confidence),
                    via_subquery=sq.subquery_id,
                )
            )
        return nodes, []

    def _exec_direct(
        self,
        sq: SubQuery,
        runtime: AnamnesisRuntime,
    ) -> tuple[list[RetrievedNode], list[RetrievedEdge]]:
        ko_id = sq.params.get("ko_id")
        if not isinstance(ko_id, str):
            return [], []
        try:
            runtime.ko_runtime.load_raw(ko_id)
        except Exception as exc:  # noqa: BLE001
            logger.warning("ko_id_direct failed for %s: %s", ko_id, exc)
            return [], []
        return (
            [
                RetrievedNode(
                    ko_id=ko_id,
                    retrieval_path=[sq.subquery_id, "ko_id_direct"],
                    score=1.0,
                    via_subquery=sq.subquery_id,
                )
            ],
            [],
        )

    # ─── Gap and contradiction detection ─────────────────────────────────

    def _detect_gaps(
        self,
        intent: NarrativeIntent | None,
        entity_hits: dict[str, int],
        relation_hits: dict[str, int],
    ) -> list[str]:
        if intent is None:
            return []
        gaps: list[str] = []
        for entity in intent.entities:
            if entity_hits.get(entity.surface, 0) == 0:
                gaps.append(f"entity:{entity.surface}")
        for rel in intent.relations:
            key = f"{rel.subject_ref}|{rel.predicate}|{rel.object_ref}"
            if relation_hits.get(key, 0) == 0:
                gaps.append(f"relation:{rel.predicate}({rel.subject_ref}->{rel.object_ref})")
        return gaps

    def _detect_contradictions(
        self,
        edges: list[RetrievedEdge],
    ) -> list[Contradiction]:
        by_sp: dict[tuple[str, str], list[RetrievedEdge]] = {}
        for e in edges:
            by_sp.setdefault((e.subject_ko, e.predicate), []).append(e)
        out: list[Contradiction] = []
        for (subj, pred), bucket in by_sp.items():
            objects = {e.object_ko for e in bucket}
            if len(objects) > 1:
                sources: list[SourceEvidence] = []
                for e in bucket:
                    sources.extend(e.source_evidence)
                out.append(
                    Contradiction(
                        locus=f"{subj}|{pred}",
                        sources=sources,
                        description=(
                            f"Multiple objects retrieved for ({subj}, {pred}): "
                            + ", ".join(sorted(objects))
                        ),
                    )
                )
        return out


def _Literal(value: str) -> Any:
    """Lazy import to keep rdflib out of the cold-import path."""
    from rdflib import Literal as _RDFLiteral

    return _RDFLiteral(value)
