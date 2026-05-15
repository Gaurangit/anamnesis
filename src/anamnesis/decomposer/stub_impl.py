"""Deterministic StubDecomposer.

Produces a stable :class:`NarrativeIntent` for any essay via cheap regex
heuristics. No network calls, no LLM. Used as the always-passing target
of contract tests and as the offline fallback for the eval harness.

Heuristics, in order:

* Entities: capitalised multi-word noun phrases, plus standalone
  capitalised tokens not at sentence start.
* Relations: simple ``<entity> <verb> <entity>`` matches over a small
  closed verb vocabulary.
* Temporal scope: explicit four-digit years and a small set of fuzzy
  era labels ('early 1950s', 'late 19th century', ...).
* Ambiguity score: rises when fewer than two entities are detected
  or when the essay is shorter than ~80 chars.
"""

from __future__ import annotations

import re
from datetime import date, datetime, timezone

from anamnesis.models.intent import (
    EntityMention,
    NarrativeIntent,
    RelationHint,
    TemporalScope,
)
from anamnesis._hashing import content_hash

PROMPT_VERSION = "stub-v1"

_VERB_VOCAB = (
    "studied",
    "investigated",
    "proposed",
    "discovered",
    "developed",
    "wrote",
    "led",
    "founded",
    "influenced",
    "challenged",
    "refuted",
    "extended",
)

_ENTITY_RE = re.compile(r"\b(?:[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3})\b")
_YEAR_RE = re.compile(r"\b(1\d{3}|20\d{2})\b")
_FUZZY_ERA_RE = re.compile(
    r"\b(early|mid|late)?\s*(\d{4}s|\d{1,2}(?:st|nd|rd|th)\s+century)\b",
    re.IGNORECASE,
)


_SENTENCE_INITIAL_STOP = {
    "The",
    "A",
    "An",
    "His",
    "Her",
    "Their",
    "It",
    "This",
    "That",
    "These",
    "Those",
    "In",
    "On",
    "At",
    "By",
    "For",
    "What",
    "When",
    "Where",
    "Who",
    "Why",
    "How",
}


def _extract_entities(essay: str) -> list[EntityMention]:
    seen: dict[str, EntityMention] = {}
    multi_word_last_token: set[str] = set()
    sentences = re.split(r"(?<=[.!?])\s+", essay)

    def _accept(surface: str) -> None:
        if len(surface) < 3 or surface.lower() in {"the", "a", "an", "his", "her", "their"}:
            return
        if surface not in seen:
            seen[surface] = EntityMention(
                surface=surface,
                canonical_guess=surface.replace(" ", "_"),
                confidence=0.6 if " " in surface else 0.4,
            )
        if " " in surface:
            multi_word_last_token.add(surface.rsplit(" ", 1)[-1])

    for sentence in sentences:
        leading_consumed = False
        for match in _ENTITY_RE.finditer(sentence):
            surface = match.group(0)
            if match.start() == 0 and not leading_consumed:
                leading_consumed = True
                # Multi-word sentence-initial phrases are almost always
                # proper nouns and should be captured. Single-word ones
                # are mostly stop words / sentence starters; only keep
                # them if we've seen them in a less ambiguous context.
                if " " not in surface:
                    if surface in _SENTENCE_INITIAL_STOP:
                        continue
                    if surface not in seen and surface not in multi_word_last_token:
                        continue
            _accept(surface)
    return list(seen.values())


def _extract_relations(essay: str, entities: list[EntityMention]) -> list[RelationHint]:
    surfaces = sorted({e.surface for e in entities}, key=len, reverse=True)
    if not surfaces:
        return []
    surf_alt = "|".join(re.escape(s) for s in surfaces)
    verb_alt = "|".join(_VERB_VOCAB)
    # Subject must be a known entity; object can be any noun-ish token,
    # because lowercase concept words ("morphogenesis", "rates") may not
    # surface as capitalised entities.
    pattern = re.compile(
        rf"({surf_alt})\s+(?:was\s+)?({verb_alt})\s+(?:the\s+|a\s+|an\s+)?"
        r"([A-Za-z][A-Za-z0-9_-]*(?:\s+[A-Za-z0-9_-]+){0,3})",
        re.IGNORECASE,
    )
    seen: set[tuple[str, str, str]] = set()
    relations: list[RelationHint] = []
    for match in pattern.finditer(essay):
        subj = match.group(1)
        verb = match.group(2).lower()
        obj_raw = match.group(3).strip()
        # Drop trailing prepositions/conjunctions from the object phrase.
        obj_words = obj_raw.split()
        while obj_words and obj_words[-1].lower() in {
            "in", "on", "of", "to", "and", "or", "the", "a", "an", "for", "by",
        }:
            obj_words.pop()
        if not obj_words:
            continue
        obj = " ".join(obj_words)
        key = (subj, verb, obj)
        if key in seen:
            continue
        seen.add(key)
        relations.append(
            RelationHint(
                predicate=verb,
                subject_ref=subj,
                object_ref=obj,
                confidence=0.55,
            )
        )
    return relations


def _extract_temporal(essay: str) -> TemporalScope | None:
    years = sorted({int(y) for y in _YEAR_RE.findall(essay)})
    fuzzy_match = _FUZZY_ERA_RE.search(essay)
    fuzzy_label = fuzzy_match.group(0).strip().lower() if fuzzy_match else None
    if not years and not fuzzy_label:
        return None
    start = date(years[0], 1, 1) if years else None
    end = date(years[-1], 12, 31) if len(years) > 1 else None
    return TemporalScope(start=start, end=end, fuzzy_label=fuzzy_label)


def _compute_ambiguity(essay: str, entities: list[EntityMention]) -> float:
    score = 0.0
    if len(essay) < 80:
        score += 0.4
    if len(entities) < 2:
        score += 0.35
    if "?" in essay:
        score += 0.15
    return min(score, 1.0)


def _pick_output_hint(
    entities: list[EntityMention],
    temporal: TemporalScope | None,
) -> str:
    if temporal is not None:
        return "timeline"
    if len(entities) >= 3:
        return "network"
    return "auto"


class StubDecomposer:
    """Offline, deterministic decomposer used by tests and the offline path."""

    name = "stub"
    prompt_version = PROMPT_VERSION

    def decompose(
        self,
        essay: str,
        *,
        schema_hints: list[str] | None = None,
    ) -> NarrativeIntent:
        essay_hash = content_hash(essay)
        intent_id = content_hash(
            f"{essay_hash}::{self.name}::{self.prompt_version}"
        )
        entities = _extract_entities(essay)
        relations = _extract_relations(essay, entities)
        temporal = _extract_temporal(essay)
        ambiguity = _compute_ambiguity(essay, entities)
        primary_theme = (
            entities[0].surface + " and related figures"
            if entities
            else essay.strip().split(".")[0][:120]
        )
        alternatives: list[str] = []
        if ambiguity > 0.5:
            alternatives = [
                f"Treat '{e.surface}' as the principal subject" for e in entities[:3]
            ] or ["Unable to identify a principal subject — please rephrase."]

        return NarrativeIntent(
            intent_id=intent_id,
            essay_hash=essay_hash,
            decomposer_model=f"{self.name}:0",
            prompt_version=self.prompt_version,
            primary_theme=primary_theme,
            entities=entities,
            relations=relations,
            temporal_scope=temporal,
            analogy_targets=[],
            output_type_hint=_pick_output_hint(entities, temporal),  # type: ignore[arg-type]
            alternative_interpretations=alternatives,
            ambiguity_score=ambiguity,
            created_at=datetime.now(tz=timezone.utc),
        )
