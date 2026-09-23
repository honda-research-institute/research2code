"""Failure-injection helpers for the orchestrator test suite.

Each helper sets up the conditions for a specific failure mode. The
naming convention is `inject_<what_goes_wrong>(...)`, and each helper
documents which failure it's intended to simulate.

These are deliberately small and explicit — tests should read as "given X
failure, the driver does Y". Hiding too much setup behind a helper makes
test intent unclear.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Pre-canned judge decisions (for tests of driver behavior given a decision)
# ---------------------------------------------------------------------------


def queue_judge_decision(
    fake_dispatch, *, stage_id: str, iteration: int, validator_label: str,
    classification: str,
    action: str,
    target_agent: str | None = None,
    finding: dict | None = None,
    rationale: str = "test-rationale",
    confidence: str = "high",
    files_examined: list[str] | None = None,
) -> None:
    """Queue the next dispatch to r2c-halt-judge with a canned decision that
    will appear in the dispatch-scoped `.pipeline/judge_decision_parts/*.json`
    scratch file.

    The driver's `_invoke_judge` dispatches the halt-judge, reads the
    scratch file, then appends the validated decision to driver-owned
    `.pipeline/judge_decisions.json`. The fake dispatcher writes the scratch
    file directly per this canned content.

    Covers the driver-behavior test path (not judge-accuracy)."""
    decision: dict[str, Any] = {
        "schema_version": "1.0.0",
        "stage_id": stage_id,
        "iteration": iteration,
        "validator_label": validator_label,
        "classification": classification,
        "action": action,
        "target_agent": target_agent,
        "finding": finding,
        "rationale": rationale,
        "confidence": confidence,
        "files_examined": files_examined or ["scripts/run_pipeline.py"],
    }
    from run_pipeline import _judge_decision_scratch_relpath  # noqa: PLC0415

    scratch_rel = _judge_decision_scratch_relpath(
        stage_id=stage_id,
        iteration=iteration,
        validator_label=validator_label,
    )
    fake_dispatch.expect(
        agent="r2c-halt-judge",
        writes={scratch_rel: json.dumps(decision, indent=2)},
    )


def queue_judge_decision_object(fake_dispatch, decision: dict) -> None:
    """Queue one judge dispatch from an already-built JudgeDecision dict."""
    from run_pipeline import _judge_decision_scratch_relpath  # noqa: PLC0415

    decision.setdefault("schema_version", "1.0.0")
    scratch_rel = _judge_decision_scratch_relpath(
        stage_id=decision["stage_id"],
        iteration=decision["iteration"],
        validator_label=decision["validator_label"],
    )
    fake_dispatch.expect(
        agent="r2c-halt-judge",
        writes={scratch_rel: json.dumps(decision, indent=2)},
    )


# ---------------------------------------------------------------------------
# Pre-canned pipeline artifacts (use as fake_dispatch writes)
# ---------------------------------------------------------------------------


def minimal_method_spec(
    *, paradigm_id: str = "active_learning",
    pluggable_name: str = "select_batch",
) -> dict:
    """Build the smallest method_spec.json that passes schema validation.

    Tests that need a specific paradigm or pluggable_component name pass
    overrides; defaults match active_learning as the lowest-friction case.
    """
    return {
        "schema_version": "1.0.0",
        "paper": {
            "title": "Test Paper",
            "authors": ["Test Author"],
            "year": 2026,
            "venue": "Test Venue",
        },
        "core_method": {
            "description": "Test core method.",
            "key_elements": ["eq-1"],
        },
        "comparison": {
            "classification": {
                "id": paradigm_id,
            },
            "pluggable_component": {
                "name": pluggable_name,
            },
        },
        "critical_requirements": {
            "model": {"specific_features": []},
            "data": {"specific_features": []},
        },
        "hyperparameters": [],
    }


def minimal_paper_map() -> dict:
    """Smallest paper_map.json that passes schema validation."""
    return {
        "schema_version": "1.0.0",
        "elements": [
            {"id": "eq-1", "type": "equation", "section": "1.0",
             "label": "Eq. 1 — test", "content": "x = y + 1"},
        ],
    }


def write_stage_complete_sentinel(run_dir: Path, stage_id: str) -> None:
    """Backdoor: write a per-stage complete-sentinel directly. Use for tests
    where the prior-stage completion is assumed (skipping prior-stage setup)."""
    p = run_dir / ".pipeline" / f"{stage_id}.complete"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("", encoding="utf-8")


def write_halt(
    run_dir: Path, stage_id: str, reason: str = "test-halt", **context
) -> None:
    """Backdoor: write a per-stage halt artifact directly. Use for tests of
    'resume after halt' behavior."""
    p = run_dir / ".pipeline" / f"{stage_id}.halt"
    p.parent.mkdir(parents=True, exist_ok=True)
    artifact = {
        "status": "halted",
        "stage": stage_id,
        "reason": reason,
        "context": context,
    }
    p.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
