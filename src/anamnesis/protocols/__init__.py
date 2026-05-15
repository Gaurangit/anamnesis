"""Vendor-neutral Protocol contracts for swappable pipeline components.

Mirrors :mod:`gekg.provider` — concrete implementations live under
``anamnesis/decomposer/``, ``anamnesis/synthesizer/``, ``anamnesis/critic/``
and ``anamnesis/lenses/``.
"""

from anamnesis.protocols.critic import AdversarialCritic
from anamnesis.protocols.decomposer import NarrativeDecomposer
from anamnesis.protocols.lens import Lens
from anamnesis.protocols.synthesizer import Synthesizer

__all__ = [
    "AdversarialCritic",
    "Lens",
    "NarrativeDecomposer",
    "Synthesizer",
]
