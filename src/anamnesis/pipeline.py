"""run_query — single entry point for the narrative-query pipeline.

Fixed-stage pipeline (anti-pattern §10: no agent loop). The stages are:

1. Decompose essay → :class:`NarrativeIntent`.
2. If ambiguous, return a :class:`ClarificationRequest`.
3. Plan → :class:`RetrievalPlan` (cacheable).
4. Execute → :class:`RetrievedSubgraph`.
5. Pick lens (or honour the caller's choice).
6. Synthesize → :class:`SynthesisResult`.
7. If a critic is supplied: build complement plan, execute, critique,
   :func:`apply_critique` to fold the report back into the synthesis.

The synthesizer step never reads from the cache (anti-pattern §10:
freshness of phrasing matters).
"""

from __future__ import annotations

from anamnesis.cache import AnamnesisCache
from anamnesis.executor import PlanExecutor
from anamnesis.lenses.picker import pick_lens
from anamnesis.models.output import QueryOutput
from anamnesis.planner import RetrievalPlanner
from anamnesis.protocols.critic import AdversarialCritic
from anamnesis.protocols.decomposer import NarrativeDecomposer
from anamnesis.protocols.synthesizer import Synthesizer
from anamnesis.runtime import AnamnesisRuntime
from anamnesis.synthesizer.apply_critique import apply_critique


def run_query(
    essay: str,
    *,
    runtime: AnamnesisRuntime,
    decomposer: NarrativeDecomposer,
    planner: RetrievalPlanner | None = None,
    executor: PlanExecutor | None = None,
    synthesizer: Synthesizer,
    critic: AdversarialCritic | None = None,
    lens: str = "auto",
    cache: AnamnesisCache | None = None,
) -> QueryOutput:
    """Execute the narrative-query pipeline end to end."""
    planner = planner or RetrievalPlanner()
    executor = executor or PlanExecutor()

    if cache is not None:
        intent = cache.get_or_compute_intent(essay, decomposer)
    else:
        intent = decomposer.decompose(essay)

    if intent.ambiguity_score > 0.5:
        return QueryOutput.clarification_needed(intent)

    if cache is not None:
        plan = cache.get_or_compute_plan(intent, planner)
    else:
        plan = planner.plan(intent)

    subgraph = executor.execute(plan, runtime, intent=intent)
    chosen_lens = pick_lens(intent.output_type_hint, subgraph) if lens == "auto" else lens
    synthesis = synthesizer.synthesize(intent, subgraph, chosen_lens)

    critique = None
    if critic is not None:
        complement_plan = planner.complement_plan(intent)
        complement_subgraph = executor.execute(complement_plan, runtime, intent=intent)
        critique = critic.critique(synthesis, complement_subgraph)
        synthesis = apply_critique(synthesis, critique)

    return QueryOutput(
        kind="synthesis",
        intent=intent,
        plan=plan,
        subgraph=subgraph,
        synthesis=synthesis,
        critique=critique,
    )
