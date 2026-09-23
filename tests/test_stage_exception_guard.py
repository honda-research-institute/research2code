"""Last-resort stage exception guard (stage-exception-guard-design.md).

The class-level floor under the ~13 stage functions: an unhandled exception
becomes a normal halt (artifact, catalog story, explanation-only packaging)
instead of a bare traceback and a run dir with no front door — the
Rethinking-Grouping 2026-07-13 crash shape. Decisions baked in (maintainer
decision 2026-07-14): demotes like any other halt; driver_state snapshot in
the halt context, size-capped.
"""

from __future__ import annotations

import json

import pytest

import halt_catalog
from tests.helpers.state import make_state


@pytest.fixture
def quiet_session(monkeypatch):
    """The guard's halt() posts a TUI notice; tests have no server."""
    import run_pipeline

    monkeypatch.setattr(run_pipeline, "_post_halt_to_session",
                        lambda *a, **k: None)


def _boom() -> BaseException:
    try:
        raise ValueError("matrix product with incompatible dimensions")
    except ValueError as exc:
        return exc


def test_guard_converts_exception_into_classified_halt(run_dir, quiet_session):
    from run_pipeline import _last_resort_stage_guard

    state = make_state(run_dir)
    (run_dir / ".pipeline" / "driver_state.json").write_text(
        json.dumps({"slug": "test-paper", "stages": []}), encoding="utf-8")

    result = _last_resort_stage_guard(state, "stage_2x", _boom())
    assert result.status == "halted"
    assert result.stage_id == "stage_2x"

    artifact = json.loads(
        (run_dir / ".pipeline" / "stage_2x.halt").read_text(encoding="utf-8"))
    assert artifact["halt_class"] == "driver_exception"
    context = artifact["context"]
    assert context["exception_type"] == "ValueError"
    assert "incompatible dimensions" in context["exception_message"]
    assert "ValueError" in context["traceback_tail"]
    assert json.loads(context["driver_state_snapshot"])["slug"] == "test-paper"
    assert "overwrote_partial_halt" not in context


def test_guard_snapshot_is_size_capped_and_overwrite_is_loud(run_dir, quiet_session):
    from run_pipeline import _GUARD_STATE_SNAPSHOT_CAP, _last_resort_stage_guard

    state = make_state(run_dir)
    (run_dir / ".pipeline" / "driver_state.json").write_text(
        "x" * (_GUARD_STATE_SNAPSHOT_CAP + 5000), encoding="utf-8")
    (run_dir / ".pipeline" / "stage_2x.halt").write_text(
        '{"status": "halted"}', encoding="utf-8")

    _last_resort_stage_guard(state, "stage_2x", _boom())
    context = json.loads(
        (run_dir / ".pipeline" / "stage_2x.halt").read_text())["context"]
    assert len(context["driver_state_snapshot"]) == _GUARD_STATE_SNAPSHOT_CAP
    assert context["driver_state_snapshot_truncated"] is True
    assert context["overwrote_partial_halt"] is True


def test_guard_floor_survives_packaging_failure(run_dir, quiet_session, monkeypatch):
    """If halt packaging ITSELF raises, the minimal literal fallback still
    lands a halt artifact and a halted StageResult."""
    import run_pipeline
    from run_pipeline import _last_resort_stage_guard

    def _packaging_explodes(*a, **k):
        raise RuntimeError("report renderer died")

    monkeypatch.setattr(run_pipeline, "halt", _packaging_explodes)
    state = make_state(run_dir)
    result = _last_resort_stage_guard(state, "stage_3c", _boom())
    assert result.status == "halted"
    minimal = json.loads(
        (run_dir / ".pipeline" / "stage_3c.halt").read_text(encoding="utf-8"))
    assert minimal["halt_class"] == "driver_exception"
    assert "report renderer died" in minimal["guard_packaging_error"]
    assert "ValueError" in minimal["reason"]


def test_driver_exception_catalog_story_is_researcher_voiced():
    story = halt_catalog.render_halt_story("driver_exception",
                                           stage_id="stage_2x")
    assert story is not None
    assert story.engineering_actionable is True
    assert "pipeline bug, not a problem with your paper" in story.why_stopped
    assert "Resume the run" in story.what_next


def test_guard_produces_explanation_only_manifest(run_dir, quiet_session):
    """The acceptance shape: the crash now packages what exists — the
    explanation-only manifest appears alongside the halt artifact."""
    from run_pipeline import _last_resort_stage_guard

    state = make_state(run_dir)
    _last_resort_stage_guard(state, "stage_2x", _boom())
    manifest_path = run_dir / "details" / "final_manifest.json"
    legacy_path = run_dir / "final_manifest.json"
    manifest_file = manifest_path if manifest_path.is_file() else legacy_path
    assert manifest_file.is_file(), "explanation-only packaging did not run"
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    assert (manifest.get("delivery") or {}).get("label") == "explanation_only"
