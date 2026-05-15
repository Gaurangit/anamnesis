"""CLI: anamnesis-query --essay-file PATH --lens auto --critic on

This is a thin wrapper around :func:`anamnesis.pipeline.run_query`. It
expects the caller to provide a registry path so the upstream
KORuntime / KOIndex can be wired up; for unit-test-style runs against
fixtures, pass ``--offline`` to use the stub decomposer + synthesizer
with an empty registry.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from anamnesis.cache import AnamnesisCache
from anamnesis.critic.stub_impl import StubCritic
from anamnesis.decomposer.stub_impl import StubDecomposer
from anamnesis.lenses import (
    DisagreementLens,
    NetworkLens,
    SankeyLens,
    TimelineLens,
)
from anamnesis.pipeline import run_query
from anamnesis.runtime import AnamnesisRuntime
from anamnesis.synthesizer.stub_impl import StubSynthesizer


def _build_offline_runtime() -> AnamnesisRuntime:
    """Empty in-memory runtime for offline experimentation."""
    from rdflib import Graph

    class _EmptyRegistry:
        def __iter__(self):
            return iter([])

        def resolve_path(self, ko_id, root_dir=None):  # pragma: no cover
            raise KeyError(ko_id)

    class _EmptyRuntime:
        @property
        def registry(self):
            return _EmptyRegistry()

        def load_raw(self, ko_id):  # pragma: no cover
            raise KeyError(ko_id)

    class _EmptyIndex:
        def hybrid_search(self, query, k=5):
            return []

        def search(self, query, k=5, filter=None):
            return []

    return AnamnesisRuntime(
        ko_runtime=_EmptyRuntime(),
        ko_index=_EmptyIndex(),
        graph=Graph(),
        kg_search=None,
        registry_revision="offline",
    )


def _build_live_runtime(registry_path: Path, ontology_version: str) -> AnamnesisRuntime:
    """Wire up an :class:`AnamnesisRuntime` from a real KO registry."""
    from runtime.ko_runtime import KORuntime
    from runtime.quality.graph_builder import build_graph_from_registry
    from runtime.vector.backends.memory import InMemoryVectorStore
    from runtime.vector.embedder import Embedder
    from runtime.vector.index import KOIndex

    ko_runtime = KORuntime(registry_path, ontology_version)
    graph = build_graph_from_registry(ko_runtime.registry)
    embedder = Embedder()  # default config
    store = InMemoryVectorStore()
    index = KOIndex(store, embedder, ko_runtime.registry)
    index.upsert_all()

    revision = _registry_revision(registry_path)
    return AnamnesisRuntime(
        ko_runtime=ko_runtime,
        ko_index=index,
        graph=graph,
        kg_search=None,
        registry_revision=revision,
    )


def _registry_revision(registry_path: Path) -> str:
    """Cheap content-hash of registry shard mtimes."""
    import hashlib

    h = hashlib.sha256()
    for shard in sorted(registry_path.rglob("*.yaml")):
        h.update(shard.name.encode())
        h.update(str(shard.stat().st_mtime).encode())
    return h.hexdigest()[:12] or "unknown"


_LENS_TABLE: dict[str, Any] = {
    "timeline": TimelineLens(),
    "network": NetworkLens(),
    "sankey": SankeyLens(),
    "disagreement": DisagreementLens(),
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="anamnesis-query")
    parser.add_argument("--essay-file", type=Path, required=True)
    parser.add_argument("--lens", default="auto")
    parser.add_argument("--critic", choices=("on", "off"), default="on")
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Run with an empty in-memory runtime (no KO registry).",
    )
    parser.add_argument(
        "--registry",
        type=Path,
        help="Path to a KO registry (file or directory). Required unless --offline.",
    )
    parser.add_argument(
        "--ontology-version",
        default="1.0.0",
        help="Current ontology version for the upstream KORuntime.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Write JSON output to this path instead of stdout.",
    )
    args = parser.parse_args(argv)

    essay = args.essay_file.read_text()

    if args.offline:
        runtime = _build_offline_runtime()
    else:
        if args.registry is None:
            parser.error("--registry is required unless --offline is given")
        runtime = _build_live_runtime(args.registry, args.ontology_version)

    decomposer = StubDecomposer()
    synthesizer = StubSynthesizer()
    critic = StubCritic() if args.critic == "on" else None
    cache = AnamnesisCache()

    out = run_query(
        essay,
        runtime=runtime,
        decomposer=decomposer,
        synthesizer=synthesizer,
        critic=critic,
        lens=args.lens,
        cache=cache,
    )

    if out.kind == "synthesis" and out.synthesis is not None:
        chosen = out.synthesis.lens
        lens_impl = _LENS_TABLE.get(chosen)
        if lens_impl is not None:
            out.synthesis.primary_artifact = lens_impl.render(
                out.synthesis, out.subgraph or out.synthesis  # type: ignore[arg-type]
            )

    payload = out.model_dump(mode="json")
    text = json.dumps(payload, indent=2)
    if args.output:
        args.output.write_text(text)
    else:
        sys.stdout.write(text + "\n")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
