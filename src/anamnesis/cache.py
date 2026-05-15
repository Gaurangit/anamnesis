"""Content-addressed cache for NarrativeIntent and RetrievalPlan.

Filesystem-backed (one JSON file per cached artifact). Keyed on the
content-addressed ids declared in the spec — ``intent_id`` and
``plan_id`` — so identical (essay, decomposer, prompt_version) and
identical intent inputs short-circuit without touching the LLM.

Subgraphs and SynthesisResults are NOT cached:

* Subgraphs depend on registry revision; staleness risk dominates the
  hit-rate benefit.
* Phrasing freshness is the point of the synthesizer; caching it
  defeats the purpose. (Anti-pattern §10.)
"""

from __future__ import annotations

import json
from pathlib import Path

from anamnesis._hashing import content_hash
from anamnesis.models.intent import NarrativeIntent
from anamnesis.models.plan import RetrievalPlan
from anamnesis.planner import RetrievalPlanner
from anamnesis.protocols.decomposer import NarrativeDecomposer


def _default_cache_dir() -> Path:
    return Path.home() / ".cache" / "anamnesis"


class AnamnesisCache:
    """Two-namespace filesystem cache: ``intents/`` and ``plans/``.

    The cache key for an intent embeds the decomposer's name and
    prompt_version — bumping either invalidates the cached entry without
    requiring a manual wipe.
    """

    def __init__(self, base_dir: Path | None = None) -> None:
        self._base = (base_dir or _default_cache_dir()).expanduser()
        self._intents = self._base / "intents"
        self._plans = self._base / "plans"
        self._intents.mkdir(parents=True, exist_ok=True)
        self._plans.mkdir(parents=True, exist_ok=True)

    # ─── Intent caching ───────────────────────────────────────────────────

    def _intent_key(self, essay: str, decomposer: NarrativeDecomposer) -> str:
        return content_hash(
            f"{content_hash(essay)}::{decomposer.name}::{decomposer.prompt_version}"
        )

    def get_intent(
        self, essay: str, decomposer: NarrativeDecomposer
    ) -> NarrativeIntent | None:
        path = self._intents / f"{self._intent_key(essay, decomposer)}.json"
        if not path.exists():
            return None
        with open(path) as f:
            return NarrativeIntent.model_validate(json.load(f))

    def put_intent(
        self,
        essay: str,
        decomposer: NarrativeDecomposer,
        intent: NarrativeIntent,
    ) -> None:
        path = self._intents / f"{self._intent_key(essay, decomposer)}.json"
        with open(path, "w") as f:
            json.dump(intent.model_dump(mode="json"), f, sort_keys=True)

    def get_or_compute_intent(
        self,
        essay: str,
        decomposer: NarrativeDecomposer,
    ) -> NarrativeIntent:
        existing = self.get_intent(essay, decomposer)
        if existing is not None:
            return existing
        fresh = decomposer.decompose(essay)
        self.put_intent(essay, decomposer, fresh)
        return fresh

    # ─── Plan caching ─────────────────────────────────────────────────────

    def _plan_key(self, intent: NarrativeIntent) -> str:
        return intent.intent_id

    def get_plan(self, intent: NarrativeIntent) -> RetrievalPlan | None:
        path = self._plans / f"{self._plan_key(intent)}.json"
        if not path.exists():
            return None
        with open(path) as f:
            return RetrievalPlan.model_validate(json.load(f))

    def put_plan(self, intent: NarrativeIntent, plan: RetrievalPlan) -> None:
        path = self._plans / f"{self._plan_key(intent)}.json"
        with open(path, "w") as f:
            json.dump(plan.model_dump(mode="json"), f, sort_keys=True)

    def get_or_compute_plan(
        self,
        intent: NarrativeIntent,
        planner: RetrievalPlanner,
    ) -> RetrievalPlan:
        existing = self.get_plan(intent)
        if existing is not None:
            return existing
        fresh = planner.plan(intent)
        self.put_plan(intent, fresh)
        return fresh

    # ─── Maintenance ──────────────────────────────────────────────────────

    def clear(self) -> None:
        for sub in (self._intents, self._plans):
            for f in sub.glob("*.json"):
                f.unlink()
