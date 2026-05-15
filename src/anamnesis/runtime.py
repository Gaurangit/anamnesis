"""AnamnesisRuntime — composite handle bundling the runtime deps the executor needs.

This avoids a name collision with ``knowledge_objects.runtime.ko_runtime.KORuntime``
while keeping a single argument on the pipeline. Callers construct one of
these and hand it to :func:`anamnesis.pipeline.run_query`.

Attributes:
    ko_runtime: The upstream KORuntime, used for ``load_typed`` and the
        :class:`KORegistry` it exposes.
    ko_index: The vector index used by ``vector_hybrid`` subqueries.
    graph: An rdflib :class:`rdflib.Graph` materialised from the registry
        — used by ``sparql`` subqueries. Built lazily if not supplied.
    kg_search: Optional :class:`gekg.provider.KGSearchProvider` used by
        ``kg_bridge_lookup`` subqueries. May be ``None`` if external
        grounding is not available; the executor will then skip those
        subqueries and record them as gaps.
    registry_revision: Identifier stamped on every ``RetrievedSubgraph``
        so caches can invalidate when the underlying registry changes.
        Callers typically pass a content-hash of the registry shard mtimes
        or a VCS revision.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from rdflib import Graph


class _RegistryLike(Protocol):
    """Minimal surface of :class:`KORegistry` the executor relies on."""

    def __iter__(self) -> Any: ...

    def resolve_path(self, ko_id: str, root_dir: Any | None = None) -> Any: ...


class _KORuntimeLike(Protocol):
    """Minimal surface of upstream :class:`KORuntime`."""

    @property
    def registry(self) -> _RegistryLike: ...

    def load_raw(self, ko_id: str) -> dict[str, Any]: ...


class _KOIndexLike(Protocol):
    """Minimal surface of :class:`runtime.vector.index.KOIndex`."""

    def hybrid_search(self, query: str, k: int = 5) -> list[Any]: ...

    def search(self, query: str, k: int = 5, filter: dict[str, Any] | None = None) -> list[Any]: ...


class _KGSearchLike(Protocol):
    provider_name: str

    def search(self, query: str, *, limit: int = 10, **kwargs: Any) -> list[Any]: ...


@dataclass
class AnamnesisRuntime:
    ko_runtime: _KORuntimeLike
    ko_index: _KOIndexLike
    graph: "Graph"
    kg_search: _KGSearchLike | None = None
    registry_revision: str = "unspecified"
