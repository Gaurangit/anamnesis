"""Shared Pydantic base class for anamnesis runtime models.

Mirrors the ``ConfiguredBaseModel`` shape used in
``knowledge_objects.meta.ko_model`` so behaviour stays consistent across
the Chariot stack.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class ConfiguredBaseModel(BaseModel):
    model_config = ConfigDict(
        validate_assignment=True,
        validate_default=True,
        extra="ignore",
        arbitrary_types_allowed=True,
        use_enum_values=True,
        strict=False,
    )
