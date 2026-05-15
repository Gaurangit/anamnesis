"""Stable content hashing for content-addressed pipeline artifacts.

Used to compute ``intent_id``, ``plan_id``, and ``subgraph_id``. The
serialisation walks Pydantic models with ``mode='json'`` so dates and
enums normalise to strings, then JSON-dumps with sorted keys to make
the digest insensitive to dict ordering.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

from pydantic import BaseModel


def _canonical(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    return value


def content_hash(value: Any) -> str:
    """Return the SHA-256 hex digest of ``value``.

    Strings hash directly. Pydantic models are dumped via ``model_dump``.
    Other values are JSON-serialised with sorted keys.
    """
    if isinstance(value, str):
        payload = value.encode("utf-8")
    else:
        payload = json.dumps(_canonical(value), sort_keys=True, default=str).encode(
            "utf-8"
        )
    return hashlib.sha256(payload).hexdigest()
