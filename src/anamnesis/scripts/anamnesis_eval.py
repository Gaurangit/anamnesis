"""anamnesis-eval — fixture-driven offline eval harness.

Walks ``tests/fixtures/essays/`` (or a caller-supplied directory), runs
the full offline pipeline with the stub stack, and compares the result
against the expected_intents and synthesis_rubrics fixtures. Emits a
report (JSON to stdout or to ``--output``).

Live LLM judges are out of scope here; this is the reproducible
offline path. Live impls plug in via ``--decomposer/--synthesizer``
once an LLMJudge wrapper for the synthesis rubric arrives.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import yaml

from anamnesis.cache import AnamnesisCache
from anamnesis.critic.stub_impl import StubCritic
from anamnesis.decomposer.stub_impl import StubDecomposer
from anamnesis.pipeline import run_query
from anamnesis.synthesizer.stub_impl import StubSynthesizer

# Defer the runtime import; the offline runtime is what we want here.
from anamnesis.scripts.anamnesis_query import _build_offline_runtime


def _check_intent(intent_payload: dict[str, Any], expected: dict[str, Any]) -> list[str]:
    failures: list[str] = []
    surfaces = {e["surface"] for e in intent_payload.get("entities", [])}
    for req in expected.get("required_entities", []) or []:
        if req not in surfaces:
            failures.append(f"missing required entity: {req}")
    for req in expected.get("required_relations", []) or []:
        match = any(
            r["predicate"] == req["predicate"] and r["subject_ref"] == req["subject_ref"]
            for r in intent_payload.get("relations", [])
        )
        if not match:
            failures.append(f"missing required relation: {req}")
    temporal_spec = expected.get("temporal") or {}
    fuzzy = temporal_spec.get("fuzzy_label_contains")
    if fuzzy:
        scope = intent_payload.get("temporal_scope") or {}
        if fuzzy not in (scope.get("fuzzy_label") or ""):
            failures.append(f"temporal fuzzy_label missing token: {fuzzy}")
    if "ambiguity_score_max" in expected:
        if intent_payload["ambiguity_score"] > expected["ambiguity_score_max"]:
            failures.append(
                f"ambiguity_score {intent_payload['ambiguity_score']} > "
                f"max {expected['ambiguity_score_max']}"
            )
    if "ambiguity_score_min" in expected:
        if intent_payload["ambiguity_score"] < expected["ambiguity_score_min"]:
            failures.append(
                f"ambiguity_score {intent_payload['ambiguity_score']} < "
                f"min {expected['ambiguity_score_min']}"
            )
    if expected.get("output_type_hint"):
        if intent_payload["output_type_hint"] != expected["output_type_hint"]:
            failures.append(
                f"output_type_hint {intent_payload['output_type_hint']} != "
                f"{expected['output_type_hint']}"
            )
    return failures


def _check_synthesis(synthesis: dict[str, Any], rubric: dict[str, Any]) -> list[str]:
    failures: list[str] = []
    claims = synthesis.get("claims") or []
    bridged = synthesis.get("bridged_inferences") or []

    for c in claims:
        if c.get("kind") == "grounded" and not c.get("supporting_kos"):
            failures.append("ungrounded grounded-claim — anti-pattern §10")

    for needle in rubric.get("prohibited_claims") or []:
        for c in claims:
            if needle.lower() in c["text"].lower():
                failures.append(f"prohibited claim text matched: {needle}")

    if rubric.get("requires_at_least_one_grounded_claim"):
        if not any(c.get("kind") == "grounded" for c in claims):
            failures.append("rubric requires at least one grounded claim")

    if rubric.get("requires_at_least_one_bridged_inference"):
        if not bridged:
            failures.append("rubric requires at least one bridged inference")

    return failures


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="anamnesis-eval")
    parser.add_argument("--fixtures-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)

    essays_dir = args.fixtures_dir / "essays"
    intents_dir = args.fixtures_dir / "expected_intents"
    rubrics_dir = args.fixtures_dir / "synthesis_rubrics"

    runtime = _build_offline_runtime()
    decomposer = StubDecomposer()
    synthesizer = StubSynthesizer()
    critic = StubCritic()
    cache = AnamnesisCache()

    reports: list[dict[str, Any]] = []
    overall_pass = True

    for essay_path in sorted(essays_dir.glob("*.txt")):
        name = essay_path.stem
        out = run_query(
            essay_path.read_text(),
            runtime=runtime,
            decomposer=decomposer,
            synthesizer=synthesizer,
            critic=critic,
            cache=cache,
        )
        record: dict[str, Any] = {
            "essay": name,
            "kind": out.kind,
            "intent_failures": [],
            "synthesis_failures": [],
        }

        if out.intent is not None:
            expected_path = intents_dir / f"{name}.yaml"
            if expected_path.exists():
                with open(expected_path) as f:
                    expected = yaml.safe_load(f)
                record["intent_failures"] = _check_intent(
                    out.intent.model_dump(mode="json"), expected
                )

        if out.synthesis is not None:
            rubric_path = rubrics_dir / f"{name}.yaml"
            if rubric_path.exists():
                with open(rubric_path) as f:
                    rubric = yaml.safe_load(f)
                record["synthesis_failures"] = _check_synthesis(
                    out.synthesis.model_dump(mode="json"), rubric
                )

        record["passed"] = (
            not record["intent_failures"] and not record["synthesis_failures"]
        )
        overall_pass = overall_pass and record["passed"]
        reports.append(record)

    payload = {"overall_passed": overall_pass, "reports": reports}
    text = json.dumps(payload, indent=2)
    if args.output:
        args.output.write_text(text)
    else:
        sys.stdout.write(text + "\n")
    return 0 if overall_pass else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
