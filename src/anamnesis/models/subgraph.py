"""RetrievedSubgraph and provenance-bearing edges/nodes.

The executor returns a RetrievedSubgraph with every node/edge tagged
back to its source. Per the spec, no claim downstream may lack
provenance — these structures carry that linkage.
"""

from __future__ import annotations

from pydantic import Field

from anamnesis.models.base import ConfiguredBaseModel


class SourceEvidence(ConfiguredBaseModel):
    ko_id: str
    source_uri: str | None = Field(None, description="Original document URI, if known.")
    page_or_offset: str | None = Field(
        None, description="e.g. 'p47', 'para:3', 'char:1024-1090'."
    )
    excerpt: str | None = Field(None, description="Short span if available.")


class RetrievedNode(ConfiguredBaseModel):
    ko_id: str
    retrieval_path: list[str] = Field(
        default_factory=list,
        description="Trace of how this node was reached, e.g. ['sq:3','hybrid_search','refersTo:+1'].",
    )
    score: float = Field(0.0, description="Higher is better.")
    via_subquery: str


class RetrievedEdge(ConfiguredBaseModel):
    subject_ko: str
    predicate: str
    object_ko: str
    source_evidence: list[SourceEvidence] = Field(default_factory=list)


class Contradiction(ConfiguredBaseModel):
    locus: str = Field(..., description="What is contradicted (claim id or entity).")
    sources: list[SourceEvidence] = Field(default_factory=list)
    description: str


class RetrievedSubgraph(ConfiguredBaseModel):
    subgraph_id: str
    plan_id: str
    registry_revision: str = Field(
        ..., description="Identifier for the KO registry state queried."
    )
    nodes: list[RetrievedNode] = Field(default_factory=list)
    edges: list[RetrievedEdge] = Field(default_factory=list)
    gaps: list[str] = Field(
        default_factory=list,
        description="Entities/relations referenced by the intent that returned nothing.",
    )
    contradictions: list[Contradiction] = Field(default_factory=list)
