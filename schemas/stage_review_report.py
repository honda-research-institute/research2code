"""Pydantic model for `<RUN_DIR>/.pipeline/stage_review_<stage_id>.json`.

The stage reviewer (r2c-stage-reviewer) writes one of these after each
producer stage. This model is shaped from the full recorded history of the
artifact (227 files at calibration time, 2026-08-19): every file shares the
same seven top-level keys mandated by dispatch_templates.py, so those are
first-class here. It is deliberately permissive exactly where history is
permissive — `severity` and `issue_type` are plain strings (history carries
"warning" and one-off types outside review_report.py's vocabularies),
`resolution_status` has a default (3 historical findings omit it), and extra
keys are allowed — so validating at the read path can never reject a review
that flows today.

This is the READ-path contract (`_read_review_json_or_err` in
run_pipeline.py). The RECOVERY predicate `_stage_review_valid` ("did the
agent write anything usable") stays independent and looser by design:
hardening it would flip soft retries into halts at five call sites.

The resolution re-ask artifact (`stage_review_<stage_id>_resolution.json`)
shares this exact shape and validates against the same model.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

SCHEMA_VERSION = "1.0.0"


class StageReviewFinding(BaseModel):
    """One stage-review finding. Core keys are present in 100% of recorded
    findings; everything else is optional or defaulted."""

    model_config = ConfigDict(extra="allow")

    id: str
    severity: str  # history: critical | important | nice-to-have | warning
    # The fix loop documents a fallback for findings with no target_agent
    # (route to the stage's own producer), so the contract cannot require it.
    target_agent: str | None = None
    issue_type: str  # permissive: history strays outside the IssueType vocabulary
    file: str
    location: str
    description: str
    check_id: str | None = None
    proposed_fix: str | None = None
    proposed_resolution: dict[str, Any] | None = None
    resolution_status: str = "pending"
    related_elements: list[str] = Field(default_factory=list)


class StageReviewReport(BaseModel):
    """Top-level stage_review_<stage_id>.json contract."""

    model_config = ConfigDict(extra="allow")

    schema_version: str
    stage_id: str | None = None  # one recorded review omits it; the driver's echo check owns strictness
    review_status: Literal["passed", "issues_found"]
    summary: str
    findings: list[StageReviewFinding]
    checks_summary: list[dict[str, Any]] = Field(default_factory=list)
    caveats: list[Any] = Field(default_factory=list)  # history: strings and disclosure objects
