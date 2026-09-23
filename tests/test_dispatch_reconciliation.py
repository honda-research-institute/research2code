"""Tests for stage-agnostic dispatch/candidate reconciliation."""

from __future__ import annotations

import json

import pytest

from tests.helpers.state import make_paths


def test_complete_candidate_reconciles_through_normal_promotion(run_dir):
    from artifact_candidates import current_projection_path, write_json_candidate
    from dispatch_reconciliation import reconcile_candidate_dispatches
    from run_events import append_event, load_events, replay_run_dir

    paths = make_paths(run_dir)
    append_event(
        paths.pipeline_dir,
        event_type="agent_dispatch_started",
        run_id=paths.slug,
        stage_id="stage_1",
        details={
            "agent": "r2c-method-analyzer",
            "dispatch_id": "dispatch-1",
        },
        timestamp="2026-06-03T00:00:00Z",
    )
    candidate = write_json_candidate(
        paths.pipeline_dir,
        run_id=paths.slug,
        stage_id="stage_1",
        artifact_id="method_spec",
        attempt=1,
        payload={"schema_version": "1.3.0", "core_method": {"description": "candidate"}},
        owner="r2c-method-analyzer",
    )

    results = reconcile_candidate_dispatches(
        paths.pipeline_dir,
        run_id=paths.slug,
        validators={("stage_1", "method_spec"): lambda path: []},
    )

    assert len(results) == 1
    assert results[0].status == "reconciled"
    current = current_projection_path(
        paths.pipeline_dir,
        stage_id="stage_1",
        artifact_id="method_spec",
    )
    assert current.read_text(encoding="utf-8") == candidate.read_text(encoding="utf-8")
    assert not paths.method_spec.exists(), "reconciliation must not overwrite canonical artifact"
    assert (candidate.parent / "promotion_record.json").is_file()
    record = json.loads(
        (candidate.parent / "reconciliation_record.json").read_text(encoding="utf-8")
    )
    assert record["status"] == "reconciled"

    events = load_events(paths.pipeline_dir)
    assert "dispatch_reconciled" in [event["event_type"] for event in events]
    reconciled = next(event for event in events if event["event_type"] == "dispatch_reconciled")
    assert reconciled["details"]["active_dispatch_ids"] == ["dispatch-1"]
    replay = replay_run_dir(run_dir)
    assert "dispatch_reconciled" in [
        event["event_type"] for event in replay["dispatches"]
    ]


def test_invalid_candidate_fails_cleanly_and_remains_inspectable(run_dir):
    from artifact_candidates import current_projection_path, write_json_candidate
    from dispatch_reconciliation import reconcile_candidate_dispatches
    from run_events import load_events

    paths = make_paths(run_dir)
    candidate = write_json_candidate(
        paths.pipeline_dir,
        run_id=paths.slug,
        stage_id="stage_1",
        artifact_id="method_spec",
        attempt=1,
        payload={"bad": "candidate"},
        owner="r2c-method-analyzer",
    )

    results = reconcile_candidate_dispatches(
        paths.pipeline_dir,
        run_id=paths.slug,
        validators={("stage_1", "method_spec"): lambda path: ["schema drift"]},
    )

    assert len(results) == 1
    assert results[0].status == "failed"
    assert "schema drift" in results[0].reason
    assert candidate.is_file()
    assert (candidate.parent / "candidate_validation.json").is_file()
    record = json.loads(
        (candidate.parent / "reconciliation_record.json").read_text(encoding="utf-8")
    )
    assert record["status"] == "failed"
    assert "schema drift" in record["reason"]
    assert not current_projection_path(
        paths.pipeline_dir,
        stage_id="stage_1",
        artifact_id="method_spec",
    ).exists()
    assert [event["event_type"] for event in load_events(paths.pipeline_dir)] == [
        "artifact_candidate_recorded",
        "artifact_promotion_failed",
        "agent_dispatch_failed",
    ]


def test_partial_candidate_attempt_marks_dispatch_failed(run_dir):
    from artifact_candidates import current_projection_path
    from dispatch_reconciliation import reconcile_candidate_dispatches
    from run_events import append_event, load_events

    paths = make_paths(run_dir)
    append_event(
        paths.pipeline_dir,
        event_type="agent_dispatch_started",
        run_id=paths.slug,
        stage_id="stage_1",
        details={
            "agent": "r2c-method-analyzer",
            "dispatch_id": "dispatch-partial",
        },
    )
    attempt_dir = paths.pipeline_dir / "candidates" / "stage_1" / "method_spec" / "1"
    attempt_dir.mkdir(parents=True)
    (attempt_dir / "notes.txt").write_text("partial output\n", encoding="utf-8")

    results = reconcile_candidate_dispatches(paths.pipeline_dir, run_id=paths.slug)

    assert len(results) == 1
    assert results[0].status == "failed"
    assert "candidate missing" in results[0].reason
    record = json.loads(
        (attempt_dir / "reconciliation_record.json").read_text(encoding="utf-8")
    )
    assert record["status"] == "failed"
    assert not current_projection_path(
        paths.pipeline_dir,
        stage_id="stage_1",
        artifact_id="method_spec",
    ).exists()
    failed = [event for event in load_events(paths.pipeline_dir) if event["event_type"] == "agent_dispatch_failed"]
    assert len(failed) == 1
    assert failed[0]["details"]["active_dispatch_ids"] == ["dispatch-partial"]


def test_terminal_attempts_are_not_reconciled_twice(run_dir):
    from artifact_candidates import write_json_candidate
    from dispatch_reconciliation import reconcile_candidate_dispatches
    from run_events import load_events

    paths = make_paths(run_dir)
    write_json_candidate(
        paths.pipeline_dir,
        run_id=paths.slug,
        stage_id="stage_1",
        artifact_id="method_spec",
        attempt=1,
        payload={"schema_version": "1.3.0", "core_method": {"description": "candidate"}},
        owner="r2c-method-analyzer",
    )
    first = reconcile_candidate_dispatches(
        paths.pipeline_dir,
        run_id=paths.slug,
        validators={("stage_1", "method_spec"): lambda path: []},
    )
    second = reconcile_candidate_dispatches(
        paths.pipeline_dir,
        run_id=paths.slug,
        validators={("stage_1", "method_spec"): lambda path: []},
    )

    assert len(first) == 1
    assert second == []
    assert [event["event_type"] for event in load_events(paths.pipeline_dir)].count(
        "dispatch_reconciled"
    ) == 1


def test_registry_mismatch_is_rejected(run_dir):
    from artifact_candidates import ArtifactRegistryEntry
    from dispatch_reconciliation import DispatchReconciliationError
    from dispatch_reconciliation import reconcile_candidate_dispatches

    paths = make_paths(run_dir)
    registry = {
        ("stage_2x", "params"): ArtifactRegistryEntry(
            stage_id="stage_1",
            artifact_id="method_spec",
            display_name="Bad Registry Entry",
            canonical_rel_path=".pipeline/method_spec.json",
            current_filename="method_spec.json",
        )
    }

    with pytest.raises(DispatchReconciliationError):
        reconcile_candidate_dispatches(
            paths.pipeline_dir,
            run_id=paths.slug,
            registry=registry,
        )

