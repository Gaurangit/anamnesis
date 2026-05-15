"""promote-procedural — HITL gate for draft procedural KOs.

Spec §2 "Draft → stable promotion" forbids bypass paths, so this script
is the only sanctioned route for changing ``ko_status`` from ``draft``
to ``stable`` on KOs under ``knowledge_objects/domains/procedural/``.

Automated checks:

1. Schema validity — load the KO via the generated Pydantic model.
2. Provenance completeness — ``metadata.attributed_to`` non-empty and
   ``metadata.generated_at`` present.
3. Cross-source agreement — at least two distinct entries in
   ``metadata.derived_from`` OR the ``--cross-source-skip`` flag is
   set explicitly (with a recorded reason).

HITL:

* ``--accept`` flips status to ``stable`` after the automated checks
  pass.
* Without ``--accept`` the script reports what would change and exits
  non-zero if checks fail.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import yaml


def _load(path: Path) -> dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f)


def _check_schema(ko: dict[str, Any], model_module) -> list[str]:
    failures: list[str] = []
    ko_type = ko.get("ko_type")
    cls = getattr(model_module, ko_type, None)
    if cls is None:
        failures.append(f"unknown ko_type: {ko_type}")
        return failures
    try:
        cls(**ko)
    except Exception as exc:  # noqa: BLE001 — report, don't crash
        failures.append(f"schema validation failed: {exc}")
    return failures


def _check_provenance(ko: dict[str, Any]) -> list[str]:
    failures: list[str] = []
    metadata = ko.get("metadata") or {}
    attributed = metadata.get("attributed_to") or []
    if not attributed:
        failures.append("metadata.attributed_to is empty")
    if not metadata.get("generated_at"):
        failures.append("metadata.generated_at is missing")
    return failures


def _check_cross_source(ko: dict[str, Any]) -> list[str]:
    metadata = ko.get("metadata") or {}
    derived = metadata.get("derived_from") or []
    distinct = {d for d in derived if isinstance(d, str)}
    if len(distinct) < 2:
        return [
            "cross-source check failed: need ≥2 distinct entries in "
            "metadata.derived_from (or pass --cross-source-skip with --reason)"
        ]
    return []


def _load_procedural_model_module():
    import importlib.util

    here = Path(__file__).resolve()
    model_path = (
        here.parent.parent.parent.parent.parent
        / "knowledge_objects"
        / "domains"
        / "procedural"
        / "procedural_ko_model.py"
    )
    spec = importlib.util.spec_from_file_location("procedural_ko_model", model_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not locate procedural model at {model_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="promote-procedural")
    parser.add_argument("ko_paths", nargs="+", type=Path)
    parser.add_argument("--accept", action="store_true", help="Flip status to stable.")
    parser.add_argument(
        "--cross-source-skip",
        action="store_true",
        help="Skip the cross-source-agreement check.",
    )
    parser.add_argument(
        "--reason",
        help="Required when --cross-source-skip is set.",
    )
    args = parser.parse_args(argv)

    if args.cross_source_skip and not args.reason:
        parser.error("--cross-source-skip requires --reason")

    model_module = _load_procedural_model_module()
    any_failure = False

    for path in args.ko_paths:
        ko = _load(path)
        if (ko.get("metadata") or {}).get("ko_status") != "draft":
            print(f"[skip] {path}: not in draft status", file=sys.stderr)
            continue

        failures: list[str] = []
        failures.extend(_check_schema(ko, model_module))
        failures.extend(_check_provenance(ko))
        if not args.cross_source_skip:
            failures.extend(_check_cross_source(ko))

        if failures:
            any_failure = True
            print(f"[FAIL] {path}", file=sys.stderr)
            for f in failures:
                print(f"  - {f}", file=sys.stderr)
            continue

        print(f"[OK]   {path}: passes all automated checks", file=sys.stderr)
        if args.accept:
            ko.setdefault("metadata", {})["ko_status"] = "stable"
            with open(path, "w") as f:
                yaml.safe_dump(ko, f, sort_keys=False)
            print(f"[promoted] {path} → stable", file=sys.stderr)

    return 1 if any_failure else 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
