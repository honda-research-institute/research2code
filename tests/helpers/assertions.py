"""Assertion helpers for the orchestrator test suite.

Keep these focused: each helper asserts ONE thing in a way that produces a
clear failure message naming the expected vs actual state. If a test needs
to assert several things, call several helpers — don't combine.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Stage-result assertions
# ---------------------------------------------------------------------------


def assert_stage_completed(result, stage_id: str | None = None) -> None:
    """Assert result.status == 'completed' (and optionally stage_id matches)."""
    assert result.status == "completed", (
        f"expected stage completed, got status={result.status!r} "
        f"(notes={result.notes!r})"
    )
    if stage_id is not None:
        assert result.stage_id == stage_id, (
            f"expected stage_id={stage_id!r}, got {result.stage_id!r}"
        )


def assert_stage_halted(
    result, *, stage_id: str | None = None, reason_contains: str | None = None
) -> None:
    """Assert result.status == 'halted', optionally with a substring match on
    halt_artifact.reason."""
    assert result.status == "halted", (
        f"expected stage halted, got status={result.status!r} "
        f"(notes={result.notes!r})"
    )
    if stage_id is not None:
        assert result.stage_id == stage_id, (
            f"expected stage_id={stage_id!r}, got {result.stage_id!r}"
        )
    if reason_contains is not None:
        reason = (result.halt_artifact or {}).get("reason", "")
        assert reason_contains in reason, (
            f"expected halt reason to contain {reason_contains!r}, "
            f"got reason={reason!r}"
        )


def assert_stage_skipped(result, stage_id: str | None = None) -> None:
    """Assert result.status == 'skipped'."""
    assert result.status == "skipped", (
        f"expected stage skipped, got status={result.status!r} "
        f"(notes={result.notes!r})"
    )
    if stage_id is not None:
        assert result.stage_id == stage_id


def assert_stage_degraded(
    result, *, stage_id: str | None = None, notes_contains: str | None = None,
    run_dir: "Path | None" = None, known_issue_contains: str | None = None,
) -> None:
    """Assert result.status == 'degraded' (the halt→degrade outcome: a usable
    artifact exists, a quality gate couldn't be satisfied, the run continues).
    Optionally check the notes substring and that KNOWN_ISSUES.md (under
    `run_dir`) contains an expected string."""
    assert result.status == "degraded", (
        f"expected stage degraded, got status={result.status!r} "
        f"(notes={result.notes!r})"
    )
    if stage_id is not None:
        assert result.stage_id == stage_id, (
            f"expected stage_id={stage_id!r}, got {result.stage_id!r}"
        )
    if notes_contains is not None:
        assert notes_contains in (result.notes or ""), (
            f"expected notes to contain {notes_contains!r}, got {result.notes!r}"
        )
    if known_issue_contains is not None:
        import run_layout  # noqa: PLC0415 (lazy: needs conftest sys.path setup)
        assert run_dir is not None, "pass run_dir to check KNOWN_ISSUES.md"
        ki = Path(run_dir) / run_layout.KNOWN_ISSUES_MD
        assert ki.is_file(), f"KNOWN_ISSUES.md not written at {ki}"
        text = ki.read_text(encoding="utf-8")
        assert known_issue_contains in text, (
            f"expected KNOWN_ISSUES.md to contain {known_issue_contains!r}"
        )


# ---------------------------------------------------------------------------
# Sentinel assertions
# ---------------------------------------------------------------------------


def assert_sentinel_present(run_dir: Path, stage_id: str) -> None:
    """Assert the per-stage complete-sentinel exists."""
    p = run_dir / ".pipeline" / f"{stage_id}.complete"
    assert p.is_file(), (
        f"expected sentinel {p.relative_to(run_dir)} present; not found"
    )


def assert_sentinel_absent(run_dir: Path, stage_id: str) -> None:
    """Assert the per-stage complete-sentinel does NOT exist."""
    p = run_dir / ".pipeline" / f"{stage_id}.complete"
    assert not p.exists(), (
        f"expected sentinel {p.relative_to(run_dir)} absent; found it"
    )


# ---------------------------------------------------------------------------
# FakeDispatch call-log assertions
# ---------------------------------------------------------------------------


def assert_dispatched(
    fake_dispatch, agent: str, *, times: int | None = None,
    prompt_contains: str | None = None,
) -> None:
    """Assert the fake dispatcher saw ≥1 call to `agent` (or exactly `times`
    calls). Optionally checks a prompt substring on the matching call(s)."""
    matches = [c for c in fake_dispatch.calls if c.agent == agent]
    if times is not None:
        assert len(matches) == times, (
            f"expected {times} dispatch(es) to {agent!r}, got {len(matches)} "
            f"(all calls: {[c.agent for c in fake_dispatch.calls]})"
        )
    else:
        assert matches, (
            f"expected at least one dispatch to {agent!r}, got none "
            f"(all calls: {[c.agent for c in fake_dispatch.calls]})"
        )
    if prompt_contains is not None:
        ok = any(prompt_contains in c.prompt for c in matches)
        assert ok, (
            f"expected at least one dispatch to {agent!r} with prompt "
            f"containing {prompt_contains!r}; no match found"
        )


def assert_not_dispatched(fake_dispatch, agent: str) -> None:
    """Assert the fake dispatcher saw ZERO calls to `agent`."""
    matches = [c for c in fake_dispatch.calls if c.agent == agent]
    assert not matches, (
        f"expected no dispatch to {agent!r}, got {len(matches)} "
        f"(all calls: {[c.agent for c in fake_dispatch.calls]})"
    )


# ---------------------------------------------------------------------------
# Judge-decision assertions
# ---------------------------------------------------------------------------


