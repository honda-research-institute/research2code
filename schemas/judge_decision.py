"""Pydantic model for halt-judge decisions.

The halt-judge (Think-class agent) is invoked when a validator inside a
stage fix-loop fails. Instead of mechanically re-dispatching the producer
that owns the failing artifact, the driver asks the judge to:
  - Classify the failure (producer-fixable, upstream issue, pipeline bug,
    or unclear)
  - Decide an action (dispatch a fixer, or halt)
  - Pick which fixer to dispatch (when applicable)
  - Construct a structured finding for the fix-mode dispatch

The halt-judge writes one decision object to a per-dispatch scratch file under
`.pipeline/judge_decision_parts/`. The driver validates that object and appends
it to `.pipeline/judge_decisions.json`, a canonical JSON array in the run's
pipeline dir. The driver reads the most recent N entries in the same stage to
detect oscillation (judge picks the same target_agent + finding-signature twice
in a row → bypass judge and halt).

Why a separate file from `assumptions.md`: assumptions.md communicates
run-time decisions to the researcher (e.g., "param X was inferred because
the paper didn't state it"). judge_decisions.json is a debug audit log
for measuring judge accuracy and diagnosing failure modes — different
audience, different lifecycle.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


SCHEMA_VERSION = "1.0.0"


Classification = Literal[
    "producer_fixable",
    "upstream_issue",
    "pipeline_bug",
    "unclear",
]
"""Halt taxonomy v1 (see r2c-halt-judge.md for definitions):

- `producer_fixable`: the artifact's producer can plausibly fix this in
  fix-mode with the finding from the validator stderr. Most stage-2.b/2.c
  validator failures fall here.
- `upstream_issue`: the failing artifact is downstream of a bad input
  from an earlier stage; fixing this stage's producer won't help. Judge
  recommends halt; researcher must re-run from the earlier stage.
- `pipeline_bug`: the validator or driver itself has a bug (e.g., the
  validator is asking for something the schema/taxonomy build plan doesn't
  require). Judge recommends halt; the fix is in pipeline source, not
  the run.
- `unclear`: judge can't confidently classify. Halt and surface for human.
"""

VALID_CLASSIFICATIONS = {
    "producer_fixable",
    "upstream_issue",
    "pipeline_bug",
    "unclear",
}


Action = Literal["dispatch_fix", "halt"]
"""What the judge decided to do.

- `dispatch_fix`: re-dispatch `target_agent` in fix-mode with `finding`.
- `halt`: produce a halt artifact; pipeline stops.
"""


TargetAgent = Literal[
    "r2c-architecture-coder",
    "r2c-method-coder",
    "r2c-notebook-generator",
    "r2c-smoke-diagnostician",
    "r2c-method-analyzer",
    "r2c-decomposer",
    "r2c-paper-fidelity-reviewer",
    "r2c-test-generator",
]
"""Fixers the judge can dispatch. Mirrors the `JUDGE_FIX_DISPATCHERS`
table in `run_pipeline.py`. Extend the literal + the table together
when wiring the judge into more stages.

Coverage:
- `r2c-architecture-coder`, `r2c-method-coder`: stage 2.b/2.c/2.d
  producer fix-mode (validator stderr → producer fix).
- `r2c-notebook-generator`: stage 3.a producer fix-mode + stage 3.c
  smoke-fix routing target.
- `r2c-smoke-diagnostician`: stage 3.c REPAIR mode — the diagnostician
  wrote a schema-invalid diagnosis and needs to re-emit with the
  missing/wrong field corrected (NOT a re-diagnosis from scratch).
- `r2c-method-analyzer`, `r2c-decomposer`: stage 1 validator-failure
  fix-mode for method_spec.json and paper_map.json.
- `r2c-paper-fidelity-reviewer`: stage 4 validator-failure fix-mode
  for review_report.json.
- `r2c-test-generator`: stage 2.d element-test ownership — a failing
  generated test the judge attributes to the TEST (its assertion does
  not trace to the paper element's own statement) is regenerated once;
  the code side of the same decision routes to `r2c-method-coder`."""


Confidence = Literal["high", "medium", "low"]


class JudgeFinding(BaseModel):
    """The structured finding the judge constructs for the fix-mode
    dispatch when `action=dispatch_fix`. Shape matches what
    `build_fix_mode_prompt` consumes — `id`, `severity`, `description`,
    `proposed_fix` are surfaced in the fix prompt's findings block."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(
        description="Stable id within this decision (e.g., 'JUDGE001')."
    )
    severity: Literal["critical", "important", "nice-to-have"] = "critical"
    description: str = Field(
        description=(
            "1-3 sentences naming the actual problem from the validator "
            "stderr in plain prose. NOT a copy-paste of the stderr — a "
            "diagnosis the producer can act on."
        )
    )
    proposed_fix: str | None = Field(
        default=None,
        description=(
            "Optional 1-3 sentences naming a concrete change the producer "
            "should make. Treated as a hint, not binding."
        ),
    )


class JudgeDecision(BaseModel):
    """One halt-judge decision object.

    The same shape is used for the scratch output file and for one entry in
    `.pipeline/judge_decisions.json`'s driver-owned top-level array."""

    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="before")
    @classmethod
    def normalize_unknown_classification(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        raw = data.get("classification")
        if not isinstance(raw, str) or raw in VALID_CLASSIFICATIONS:
            return data

        normalized = dict(data)
        normalized["classification"] = "unclear"
        note = (
            f"Judge emitted unrecognized classification {raw!r}; "
            "normalized to 'unclear'."
        )
        rationale = normalized.get("rationale")
        if isinstance(rationale, str) and rationale.strip():
            normalized["rationale"] = f"{rationale.rstrip()} {note}"
        else:
            normalized["rationale"] = note
        return normalized

    schema_version: Literal["1.0.0"] = Field(default=SCHEMA_VERSION)
    stage_id: str = Field(
        description="The stage whose fix-loop invoked the judge (e.g., 'stage_2d')."
    )
    iteration: int = Field(
        ge=0,
        description=(
            "Which retry iteration within the stage's fix-loop this "
            "decision was made on (0 = first failure, before any retry)."
        ),
    )
    validator_label: str = Field(
        description=(
            "The validator script/check that failed (e.g., "
            "'validate_arch_contract.py'). Used by the judge to disambiguate "
            "WHICH check raised the failure."
        )
    )
    classification: Classification
    action: Action
    target_agent: TargetAgent | None = Field(
        default=None,
        description="Required when action='dispatch_fix'; None when action='halt'.",
    )
    finding: JudgeFinding | None = Field(
        default=None,
        description="Required when action='dispatch_fix'; None when action='halt'.",
    )
    rationale: str = Field(
        description=(
            "1-5 sentences explaining how the judge reached this "
            "classification + action. Read at debug time."
        )
    )
    confidence: Confidence
    files_examined: list[str] = Field(
        default_factory=list,
        description=(
            "Relative paths of files the judge read before deciding. "
            "Empty list is suspicious — the judge should always read at "
            "least the validator script and the failing artifact."
        ),
    )
