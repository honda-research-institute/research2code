"""Pydantic model for the insight semantic-review acceptance record.

The externally owned acceptance record the generic-insight contract
reserves (the generic insight artifact contract note (internal, not shipped))
and the reviewer design specifies
(the insight semantic reviewer design note (internal, not shipped)). A
dedicated read-only Think-tier agent (`r2c-insight-semantic-reviewer`)
judges whether each insight record's statement is entailed by the exact
source passages it cites, and writes exactly one of these records to
`.pipeline/generic_insights.semantic_review.json`.

Hard boundaries the shape encodes:
- The record binds to the source and candidate SHA-256 pair it reviewed.
  Regenerating either invalidates it; a stale record is treated as absent
  (enforced by the loader in scripts/insight_semantic_review.py, not
  here — the schema cannot see the live hashes).
- Per-record verdicts are accepted/rejected only; a record the reviewer
  cannot decide goes into `needs_human` with a reason instead of getting
  a guessed verdict.
- The overall verdict is DERIVED, never chosen: any rejection rejects the
  candidate, otherwise any escalation makes it needs_human, otherwise it
  is accepted. An inconsistent overall is a schema error.
- Failure classes are the closed genus observed in the golden corpus.
  Corpus cases never become production heuristics; the genus names the
  failure shape, not any case-specific string.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


SCHEMA_VERSION = "1.0.0"

# The version family this code understands; unknown MAJOR fails closed in
# the loader.
SCHEMA_MAJOR = SCHEMA_VERSION.split(".", 1)[0]


FailureClass = Literal[
    "unsupported_strengthening",
    "safety_scope_conflation",
    "invented_algorithm_or_rule",
    "imperative_from_suggestion",
    "unresolved_ambiguity_resolution",
    "coverage_omission",
]
"""Closed failure genus from the golden corpus's observed corruptions:

- `unsupported_strengthening`: the statement promises more than the cited
  span (e.g. "idempotent" upgraded to "can never produce a different
  result").
- `safety_scope_conflation`: two distinct source concepts merged into one
  claim (e.g. safety conflated with cacheability).
- `invented_algorithm_or_rule`: a mechanism, algorithm, or rule the
  source never states (e.g. a made-up cache-key computation).
- `imperative_from_suggestion`: advisory or optional source language
  rendered as a requirement or command.
- `unresolved_ambiguity_resolution`: the source leaves a question open
  and the statement resolves it anyway.
- `coverage_omission`: the statement's honesty depends on a declared gap
  or omission that misrepresents what the source actually covers.
"""


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class RecordVerdict(_Strict):
    """One decided insight record: entailed by its cited spans, or not."""

    record_id: str = Field(min_length=1)
    verdict: Literal["accepted", "rejected"]
    # Required on rejection, forbidden on acceptance (enforced below).
    failure_class: FailureClass | None = None
    reason: str = Field(
        min_length=1,
        description=(
            "One sentence citing the span content that decides the verdict "
            "— what the source actually says versus what the statement "
            "claims."
        ),
    )

    @model_validator(mode="after")
    def _failure_class_shape(self) -> "RecordVerdict":
        if self.verdict == "rejected" and self.failure_class is None:
            raise ValueError(
                f"{self.record_id}: a rejection must name its failure class")
        if self.verdict == "accepted" and self.failure_class is not None:
            raise ValueError(
                f"{self.record_id}: an acceptance cannot carry a failure class")
        return self


class Escalation(_Strict):
    """One record the reviewer could not decide. The maintainer reads escalations
    only, never every artifact — the reason must say what blocked the
    verdict, not restate the statement."""

    record_id: str = Field(min_length=1)
    reason: str = Field(min_length=1)


class SemanticReviewRecord(_Strict):
    """The acceptance record
    (`.pipeline/generic_insights.semantic_review.json`)."""

    schema_version: str
    reviewer: str = Field(min_length=1)
    source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    candidate_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    verdicts: list[RecordVerdict] = Field(default_factory=list)
    needs_human: list[Escalation] = Field(default_factory=list)
    overall_verdict: Literal["accepted", "rejected", "needs_human"]

    @model_validator(mode="after")
    def _consistent(self) -> "SemanticReviewRecord":
        problems: list[str] = []
        ids = [v.record_id for v in self.verdicts] + \
              [e.record_id for e in self.needs_human]
        if not ids:
            problems.append("record carries no verdicts and no escalations")
        if len(set(ids)) != len(ids):
            problems.append(
                "a record id appears more than once across verdicts and "
                "needs_human")
        expected = derive_overall_verdict(
            [v.verdict for v in self.verdicts], bool(self.needs_human))
        if expected is not None and self.overall_verdict != expected:
            problems.append(
                f"overall_verdict {self.overall_verdict!r} is inconsistent "
                f"with the per-record verdicts (derived: {expected!r})")
        if problems:
            raise ValueError("; ".join(problems))
        return self


def derive_overall_verdict(
    verdicts: list[str], has_escalations: bool
) -> str | None:
    """The only overall verdict consistent with the parts: any rejection
    rejects, otherwise any escalation escalates, otherwise accepted.
    None for the empty case (schema-invalid anyway)."""
    if not verdicts and not has_escalations:
        return None
    if any(v == "rejected" for v in verdicts):
        return "rejected"
    if has_escalations:
        return "needs_human"
    return "accepted"
