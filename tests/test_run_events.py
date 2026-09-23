"""Tests for additive run event logging and replay."""

from __future__ import annotations

import json

import pytest

from tests.helpers.state import make_paths


def test_append_event_sequences_and_replay_stage_status(run_dir):
    from run_events import append_event, load_events, replay_run_dir

    paths = make_paths(run_dir)
    append_event(
        paths.pipeline_dir,
        event_type="stage_started",
        run_id=paths.slug,
        stage_id="stage_1",
        stage_label="Stage 1 - Paper Decomposition & Method Analysis",
        stage_category="setup_analysis",
        status="running",
        timestamp="2026-06-03T00:00:00Z",
    )
    append_event(
        paths.pipeline_dir,
        event_type="validation_passed",
        run_id=paths.slug,
        stage_id="stage_1",
        stage_label="Stage 1 - Paper Decomposition & Method Analysis",
        stage_category="setup_analysis",
        status="passed",
        summary="validate_method_spec.py passed",
        details={"validator": "validate_method_spec.py"},
        timestamp="2026-06-03T00:00:01Z",
    )
    append_event(
        paths.pipeline_dir,
        event_type="stage_completed",
        run_id=paths.slug,
        stage_id="stage_1",
        stage_label="Stage 1 - Paper Decomposition & Method Analysis",
        stage_category="setup_analysis",
        status="completed",
        timestamp="2026-06-03T00:00:02Z",
    )

    events = load_events(paths.pipeline_dir)
    assert [event["sequence"] for event in events] == [1, 2, 3]

    projection = replay_run_dir(run_dir)
    assert projection["last_sequence"] == 3
    assert projection["stages"][0]["stage_id"] == "stage_1"
    assert projection["stages"][0]["status"] == "completed"
    assert projection["stages"][0]["stage_category"] == "setup_analysis"
    assert len(projection["validations"]) == 1


def test_superseded_halt_sidecar_event_replays_as_artifact_decision(run_dir):
    from run_events import append_event, replay_run_dir

    paths = make_paths(run_dir)
    append_event(
        paths.pipeline_dir,
        event_type="artifact_halt_sidecar_superseded",
        run_id=paths.slug,
        stage_id="stage_1",
        stage_label="Stage 1 - Paper Decomposition & Method Analysis",
        stage_category="setup_analysis",
        status="completed",
        summary=(
            "The analyzer completed method_spec.json after its halt signal; "
            "the completed artifact will continue to validation"
        ),
        artifacts=[".pipeline/method_spec.json"],
        details={
            "artifact": ".pipeline/method_spec.json",
            "halt_sidecar": ".pipeline/method_spec.json.halt",
        },
        timestamp="2026-07-15T08:49:42Z",
    )

    projection = replay_run_dir(run_dir)

    assert len(projection["artifacts"]) == 1
    event = projection["artifacts"][0]
    assert event["event_type"] == "artifact_halt_sidecar_superseded"
    assert event["artifacts"] == [".pipeline/method_spec.json"]
    assert "continue to validation" in event["summary"]


def test_load_events_rejects_sequence_drift(run_dir):
    from run_events import EventLogError, event_log_path, load_events

    paths = make_paths(run_dir)
    event_log_path(paths.pipeline_dir).write_text(
        json.dumps({
            "schema_version": "1.0",
            "sequence": 2,
            "event_type": "stage_started",
            "run_id": paths.slug,
            "timestamp": "2026-06-03T00:00:00Z",
        })
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(EventLogError, match="expected 1"):
        load_events(paths.pipeline_dir)


def test_run_directory_lock_blocks_second_owner_and_releases(run_dir):
    from run_events import RunDirectoryLock, RunDirectoryLockError

    paths = make_paths(run_dir)
    lock = RunDirectoryLock(paths.pipeline_dir, run_id=paths.slug).acquire()
    assert (paths.pipeline_dir / "_lock" / "owner.json").is_file()

    with pytest.raises(RunDirectoryLockError):
        RunDirectoryLock(paths.pipeline_dir, run_id=paths.slug).acquire()

    lock.release()
    assert not (paths.pipeline_dir / "_lock").exists()
    RunDirectoryLock(paths.pipeline_dir, run_id=paths.slug).acquire().release()


def test_run_directory_lock_owner_write_failure_cleans_lock(run_dir, monkeypatch):
    from pathlib import Path

    from run_events import RunDirectoryLock

    paths = make_paths(run_dir)
    original_write_text = Path.write_text

    def fake_write_text(self, *args, **kwargs):
        if self.name == "owner.json":
            raise OSError("disk full")
        return original_write_text(self, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", fake_write_text)

    with pytest.raises(OSError, match="disk full"):
        RunDirectoryLock(paths.pipeline_dir, run_id=paths.slug).acquire()

    assert not (paths.pipeline_dir / "_lock").exists()


def test_event_files_are_driver_managed_for_scope_checks():
    from run_pipeline import _is_driver_managed

    assert _is_driver_managed(".pipeline/run_events.jsonl") is True
    assert _is_driver_managed(".pipeline/run_events.jsonl.lock") is True
    assert _is_driver_managed(".pipeline/progress.json") is True
    assert _is_driver_managed(".pipeline/_lock/owner.json") is True


def test_every_event_type_the_driver_emits_is_registered():
    """Registry growth gate. append_event refuses unregistered event
    types, and _append_run_event swallows that refusal by design (events
    are mirror-only) — so an unregistered emit silently drops the event
    from the canonical behavioral record. Found live twice on 2026-07-06:
    the fedavg gap run's provisional_pack_installed and the smoke
    pre-check's smoke_trainability_failed had failed to append since the
    day they were written. This sweep pins the whole class: every string
    literal passed to _append_run_event in the driver must be
    registered."""
    import re
    from pathlib import Path

    from run_events import EVENT_TYPES

    driver = (Path(__file__).resolve().parent.parent
              / "scripts" / "run_pipeline.py").read_text(encoding="utf-8")
    emitted = set(re.findall(
        r'_append_run_event\(\s*[\w.]+,\s*"([a-z0-9_]+)"', driver))
    assert emitted, "sweep found no emit sites — the regex went stale"
    unregistered = sorted(emitted - EVENT_TYPES)
    assert not unregistered, (
        f"event types emitted by run_pipeline.py but missing from "
        f"run_events.EVENT_TYPES (their appends silently fail): "
        f"{unregistered}")


def test_current_invocation_slice_segments_at_last_run_started():
    """Block 8 shared primitive: transient researcher-facing disclosures are
    scoped to the most recent invocation, which starts at the LAST
    `run_started` event in the append-only log."""
    from run_events import current_invocation_slice

    events = [
        {"event_type": "run_started", "sequence": 1},
        {"event_type": "agent_dispatch_recovered", "sequence": 2},
        {"event_type": "run_finished", "sequence": 3},
        {"event_type": "run_started", "sequence": 4},
        {"event_type": "stage_completed", "sequence": 5},
    ]
    current = current_invocation_slice(events)
    assert [e["sequence"] for e in current] == [4, 5]
    # The source list itself is never mutated — append-only evidence stays
    # whole; only the view is scoped.
    assert [e["sequence"] for e in events] == [1, 2, 3, 4, 5]


def test_current_invocation_slice_without_boundary_is_whole_log():
    """Legacy conservatism: a log with no `run_started` boundary (the driver
    never produces one, but a truncated/legacy log could) falls back to the
    whole log rather than an empty slice that could hide real events."""
    from run_events import current_invocation_slice

    events = [
        {"event_type": "stage_completed"},
        {"event_type": "agent_dispatch_recovered"},
    ]
    assert current_invocation_slice(events) == events
    assert current_invocation_slice([]) == []
