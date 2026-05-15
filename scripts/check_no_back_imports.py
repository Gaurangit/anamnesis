#!/usr/bin/env python
"""Enforce the one-directional dependency rule from spec §2.

`anamnesis` is allowed to import `knowledge_objects` and `gekg`. The
reverse is forbidden. Run this script (or wire it up as a pre-commit
hook) inside CI to fail the build if either upstream package starts
importing anamnesis.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]  # chariot/
FORBIDDEN_TREES = [
    REPO_ROOT / "knowledge_objects",
    REPO_ROOT / "gekg",
]
IMPORT_RE = re.compile(r"^\s*(?:from|import)\s+anamnesis\b", re.MULTILINE)


def main() -> int:
    violations: list[tuple[Path, int, str]] = []
    for tree in FORBIDDEN_TREES:
        if not tree.exists():
            continue
        for py_file in tree.rglob("*.py"):
            text = py_file.read_text(errors="ignore")
            for match in IMPORT_RE.finditer(text):
                line_no = text[: match.start()].count("\n") + 1
                line = text.splitlines()[line_no - 1]
                violations.append((py_file, line_no, line.strip()))
    if violations:
        print("Forbidden imports of anamnesis from upstream packages:", file=sys.stderr)
        for path, line_no, line in violations:
            print(f"  {path}:{line_no}: {line}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
