"""Apply a CritiqueReport to a SynthesisResult.

Mechanics:

* For each :class:`OverreachFlag`, lower the targeted claim's confidence
  by the corresponding entry in ``confidence_delta`` (or by 0.2 by
  default).
* If the resulting confidence drops below 0.25 OR the flag carries
  contradicting evidence, demote the claim from grounded → inference
  and shift it from ``claims`` to ``bridged_inferences``.
* Append ``critique.alternative_framings`` to the primary_artifact under
  the key ``"alternative_framings"`` so lenses can render them.

Anti-pattern §10 enforcement: if any claim still in ``claims`` lacks
``supporting_kos`` (kind='grounded'), raise. This is the hard guarantee
the pipeline gives downstream consumers.
"""

from __future__ import annotations

from anamnesis.models.critique import CritiqueReport
from anamnesis.models.synthesis import BridgedInference, Claim, SynthesisResult


def apply_critique(
    synthesis: SynthesisResult,
    critique: CritiqueReport,
) -> SynthesisResult:
    if critique.synthesis_id != synthesis.synthesis_id:
        raise ValueError(
            f"Critique synthesis_id {critique.synthesis_id} does not match "
            f"synthesis {synthesis.synthesis_id}"
        )

    claims = [c.model_copy(deep=True) for c in synthesis.claims]
    bridged = [b.model_copy(deep=True) for b in synthesis.bridged_inferences]
    provenance = {k: list(v) for k, v in synthesis.provenance_map.items()}

    demoted_indices: set[int] = set()

    for flag in critique.overreach_flags:
        if flag.claim_idx >= len(claims) or flag.claim_idx in demoted_indices:
            continue
        claim = claims[flag.claim_idx]
        delta = critique.confidence_delta.get(flag.claim_idx, -0.2)
        new_conf = max(0.0, min(1.0, claim.confidence + delta))
        if new_conf < 0.25 or flag.contradicting_evidence:
            demoted_indices.add(flag.claim_idx)
            bridged.append(
                BridgedInference(
                    text=claim.text + f" [softened: {flag.reason}]",
                    supporting_kos=[],
                    confidence=new_conf,
                )
            )
        else:
            claims[flag.claim_idx] = Claim(
                text=claim.text,
                kind=claim.kind,
                supporting_kos=claim.supporting_kos,
                confidence=new_conf,
            )

    # Drop demoted claims; rebuild provenance with the new indices.
    surviving = [
        (idx, claim) for idx, claim in enumerate(claims) if idx not in demoted_indices
    ]
    new_claims = [claim for _, claim in surviving]
    new_provenance: dict[str, list[str]] = {}
    for new_idx, (old_idx, _) in enumerate(surviving):
        if str(old_idx) in provenance:
            new_provenance[str(new_idx)] = provenance[str(old_idx)]

    # Hard guarantee — anti-pattern §10.
    for c in new_claims:
        if c.kind == "grounded" and not c.supporting_kos:
            raise RuntimeError(
                "Grounded claim lacks supporting_kos after critique application. "
                "This violates the provenance invariant."
            )

    artifact = dict(synthesis.primary_artifact)
    if critique.alternative_framings:
        artifact["alternative_framings"] = list(critique.alternative_framings)

    return synthesis.model_copy(
        update={
            "claims": new_claims,
            "bridged_inferences": bridged,
            "provenance_map": new_provenance,
            "primary_artifact": artifact,
        }
    )
