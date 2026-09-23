"""Tests for human-facing stage display names.

Internal stage IDs such as `stage_2b` are stable control-plane identifiers.
These tests pin the additive display-name layer so user-facing surfaces can be
clearer without renaming sentinel files, halt artifacts, review contracts, or
`--stop-after` choices.
"""

from __future__ import annotations

import json

from tests.helpers.state import make_state


def test_stage_label_preserves_stable_ids_with_human_display_names():
    # The exact display strings are cross-validated against the SSOT yaml
    # by scripts/refresh_ssot_docs.py (run by tests/test_ssot_docs.py), so
    # this file pins only the layer's SHAPE: a known id gets a distinct
    # human label, the heading carries the stable id, and an unknown id
    # falls back to itself.
    from run_pipeline import stage_heading, stage_label

    assert stage_label("stage_1") != "stage_1"
    assert stage_heading("stage_1") == f"{stage_label('stage_1')} (`stage_1`)"
    assert stage_label("stage_unknown") == "stage_unknown"
    assert stage_heading("stage_unknown") == "`stage_unknown`"


def test_driver_state_records_stage_id_and_display_label(run_dir):
    from run_pipeline import StageResult

    state = make_state(run_dir)
    state.record(
        StageResult(status="completed", stage_id="stage_2b", notes="done")
    )

    driver_state = json.loads(
        (run_dir / ".pipeline" / "driver_state.json").read_text(encoding="utf-8")
    )
    from run_pipeline import stage_label

    stage = driver_state["stages"][0]
    assert stage["stage_id"] == "stage_2b"
    assert stage["stage_label"] == stage_label("stage_2b")
    assert stage["stage_label"] != "stage_2b"
    assert stage["status"] == "completed"


def test_todo_state_uses_display_names_but_keeps_internal_ids_separate():
    from run_pipeline import _init_pipeline_todos

    class _TodoState:
        session_id = "ses_todo_test"
        port = 0
        agent_models: dict = {}
        todos: list[dict] = []

    from run_pipeline import stage_label

    state = _TodoState()
    _init_pipeline_todos(state)  # type: ignore[arg-type]

    assert state.todos[1]["stage_id"] == "stage_1"
    assert state.todos[1]["content"] == stage_label("stage_1")
