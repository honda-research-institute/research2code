"""Tests for candidate artifact storage and promotion primitives."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.helpers.state import make_paths


def test_candidate_layout_for_first_method_spec_rollout(run_dir):
    from artifact_candidates import candidate_artifact_path, candidate_attempt_dir

    paths = make_paths(run_dir)

    assert candidate_attempt_dir(
        paths.pipeline_dir,
        stage_id="stage_1",
        artifact_id="method_spec",
        attempt=1,
    ) == paths.pipeline_dir / "candidates" / "stage_1" / "method_spec" / "1"
    assert candidate_artifact_path(
        paths.pipeline_dir,
        stage_id="stage_1",
        artifact_id="method_spec",
        attempt=1,
    ) == (
        paths.pipeline_dir
        / "candidates"
        / "stage_1"
        / "method_spec"
        / "1"
        / "method_spec.json"
    )


def test_validated_candidate_promotes_to_artifacts_and_current(run_dir):
    from artifact_candidates import (
        current_projection_path,
        promote_validated_candidate,
        promoted_artifact_path,
        write_json_candidate,
    )
    from final_manifest import sha256_file
    from run_events import load_events, replay_run_dir

    paths = make_paths(run_dir)
    payload = {"schema_version": "1.3.0", "core_method": {"description": "candidate"}}
    candidate = write_json_candidate(
        paths.pipeline_dir,
        run_id=paths.slug,
        stage_id="stage_1",
        artifact_id="method_spec",
        attempt=1,
        payload=payload,
        owner="r2c-method-analyzer",
    )

    result = promote_validated_candidate(
        paths.pipeline_dir,
        run_id=paths.slug,
        stage_id="stage_1",
        artifact_id="method_spec",
        attempt=1,
        owner="r2c-method-analyzer",
        validator=lambda path: [],
    )

    digest = sha256_file(candidate)
    promoted = promoted_artifact_path(
        paths.pipeline_dir,
        stage_id="stage_1",
        artifact_id="method_spec",
        digest=digest,
    )
    current = current_projection_path(
        paths.pipeline_dir,
        stage_id="stage_1",
        artifact_id="method_spec",
    )
    assert Path(result["promoted"]) == promoted
    assert Path(result["current"]) == current
    assert promoted.read_text(encoding="utf-8") == candidate.read_text(encoding="utf-8")
    assert current.read_text(encoding="utf-8") == candidate.read_text(encoding="utf-8")
    assert not paths.method_spec.exists(), "prep slice must not overwrite canonical artifact"

    events = load_events(paths.pipeline_dir)
    assert [event["event_type"] for event in events] == [
        "artifact_candidate_recorded",
        "artifact_promoted",
    ]
    assert events[-1]["details"]["sha256"] == digest
    replay = replay_run_dir(run_dir)
    assert [event["event_type"] for event in replay["artifacts"]] == [
        "artifact_candidate_recorded",
        "artifact_promoted",
    ]


def test_failed_candidate_remains_inspectable_and_does_not_promote(run_dir):
    from artifact_candidates import (
        CandidateStoreError,
        current_projection_path,
        promote_validated_candidate,
        write_json_candidate,
    )
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

    with pytest.raises(CandidateStoreError, match="schema drift"):
        promote_validated_candidate(
            paths.pipeline_dir,
            run_id=paths.slug,
            stage_id="stage_1",
            artifact_id="method_spec",
            attempt=1,
            owner="r2c-method-analyzer",
            validator=lambda path: ["schema drift"],
        )

    assert candidate.is_file()
    validation = candidate.parent / "candidate_validation.json"
    assert validation.is_file()
    validation_payload = json.loads(validation.read_text(encoding="utf-8"))
    assert validation_payload["status"] == "failed"
    assert validation_payload["diagnostics"] == ["schema drift"]
    assert not current_projection_path(
        paths.pipeline_dir,
        stage_id="stage_1",
        artifact_id="method_spec",
    ).exists()
    assert [event["event_type"] for event in load_events(paths.pipeline_dir)] == [
        "artifact_candidate_recorded",
        "artifact_promotion_failed",
    ]


def test_default_candidate_validator_rejects_malformed_json(run_dir):
    from artifact_candidates import candidate_artifact_path, validate_candidate

    paths = make_paths(run_dir)
    candidate = candidate_artifact_path(
        paths.pipeline_dir,
        stage_id="stage_1",
        artifact_id="method_spec",
        attempt=1,
    )
    candidate.parent.mkdir(parents=True)
    candidate.write_text("{not json", encoding="utf-8")

    diagnostics = validate_candidate(
        paths.pipeline_dir,
        stage_id="stage_1",
        artifact_id="method_spec",
        attempt=1,
    )

    assert len(diagnostics) == 1
    assert "invalid JSON" in diagnostics[0]
