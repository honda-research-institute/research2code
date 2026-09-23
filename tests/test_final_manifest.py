"""Tests for main-owned final manifest generation and verification."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from tests.helpers.state import make_paths, make_state


def _write_text(path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _write_json(path, payload: dict) -> None:
    _write_text(path, json.dumps(payload, indent=2) + "\n")


def _stage_results(status_by_stage: dict[str, str] | None = None):
    from run_pipeline import STAGES, StageResult

    overrides = status_by_stage or {}
    return [
        StageResult(
            status=overrides.get(stage_id, "completed"),
            stage_id=stage_id,
            notes=f"{stage_id} test result",
        )
        for stage_id in STAGES
    ]


def _write_driver_state(run_dir, results) -> None:
    state = make_state(run_dir)
    for result in results:
        state.record(result)


def _write_event_log(paths, results) -> None:
    from final_manifest import STAGE_CATEGORIES, STAGE_LABELS
    from run_events import append_event

    for i, result in enumerate(results):
        stage_id = result.stage_id
        append_event(
            paths.pipeline_dir,
            event_type="stage_started",
            run_id=paths.slug,
            stage_id=stage_id,
            stage_label=STAGE_LABELS[stage_id],
            stage_category=STAGE_CATEGORIES[stage_id],
            status="running",
            timestamp=f"2026-06-03T00:{i:02d}:00Z",
        )
        event_type = {
            "completed": "stage_completed",
            "skipped": "stage_skipped",
            "halted": "stage_halted",
            "degraded": "stage_degraded",
        }[result.status]
        append_event(
            paths.pipeline_dir,
            event_type=event_type,
            run_id=paths.slug,
            stage_id=stage_id,
            stage_label=STAGE_LABELS[stage_id],
            stage_category=STAGE_CATEGORIES[stage_id],
            status=result.status,
            timestamp=f"2026-06-03T00:{i:02d}:30Z",
        )
    append_event(
        paths.pipeline_dir,
        event_type="validation_passed",
        run_id=paths.slug,
        stage_id="stage_3c",
        stage_label=STAGE_LABELS["stage_3c"],
        stage_category=STAGE_CATEGORIES["stage_3c"],
        status="passed",
        summary="smoke_run_notebook.py passed",
        details={"validator": "smoke_run_notebook.py"},
        timestamp="2026-06-03T00:59:00Z",
    )


def _populate_delivery_artifacts(run_dir, *, include_method_py: bool = True):
    paths = make_paths(run_dir)
    _write_text(run_dir / "REPORT.md", "# Run Report\n")
    _write_json(paths.paper_map, {"schema_version": "1.0.0", "elements": []})
    _write_json(paths.method_spec, {"schema_version": "1.3.0", "core_method": {}})
    _write_json(paths.feasibility_gate, {"status": "feasible"})
    _write_text(run_dir / "method" / "README.md", "# Test run\n")
    _write_text(run_dir / "method" / "data.py", "def load_data():\n    return None\n")
    _write_text(run_dir / "method" / "example_data" / "README.md", "# data\n")
    _write_json(paths.pipeline_dir / "arch_contract.json", {"schema_version": "1.0.0"})
    _write_text(run_dir / "method" / "model.py", "class Model:\n    pass\n")
    _write_text(run_dir / "method" / "training.py", "def train_from_scratch():\n    return None\n")
    if include_method_py:
        _write_text(run_dir / "method" / "method.py", "def select_batch():\n    return []\n")
    _write_text(run_dir / "method" / "__init__.py", "from .method import select_batch\n")
    _write_text(run_dir / "requirements.txt", "numpy\n")
    _write_json(paths.pipeline_dir / "params.json", {"schema_version": "1.0.0", "params": {}})
    _write_text(paths.pipeline_dir / "notebook_draft.py", "# %%\nprint('ok')\n")
    _write_json(run_dir / "notebook.ipynb", {"cells": [], "metadata": {}, "nbformat": 4, "nbformat_minor": 5})
    _write_json(paths.pipeline_dir / "review_report.json", {"schema_version": "1.1.0", "findings": []})
    return paths


def _graph_free_evidence_status() -> dict:
    return {
        "execution": {"status": "completed", "reasons": []},
        "evaluation_validity": {"status": "valid", "reasons": []},
        "mechanism": {"status": "not_applicable", "reasons": []},
        "skill": {"status": "demonstrated", "reasons": []},
        "paper_benchmark": {"status": "not_assessed", "reasons": []},
    }


def test_graph_free_demo_evidence_serialization_keeps_its_exact_shape():
    from schemas.final_manifest import DemoEvidenceStatus

    payload = _graph_free_evidence_status()
    before = json.dumps(payload, separators=(",", ":")).encode("utf-8")

    parsed = DemoEvidenceStatus.model_validate(payload)
    after = json.dumps(
        parsed.model_dump(), separators=(",", ":")
    ).encode("utf-8")

    assert after == before
    assert "graph_construction" not in parsed.model_dump()
    assert "graph_alignment" not in parsed.model_dump()
    assert "contribution" not in parsed.model_dump()


@pytest.mark.parametrize(
    "axis",
    ["graph_construction", "graph_alignment", "contribution"],
)
@pytest.mark.parametrize(
    "status",
    ["demonstrated", "not_demonstrated", "undetermined", "not_applicable"],
)
def test_optional_graph_evidence_axes_share_the_mechanism_status_grammar(
    axis, status,
):
    from schemas.final_manifest import DemoEvidenceStatus

    payload = _graph_free_evidence_status()
    payload[axis] = {
        "status": status,
        "reasons": [{"code": "graph-check", "message": "typed evidence"}],
    }

    parsed = DemoEvidenceStatus.model_validate(payload)
    got = getattr(parsed, axis)
    assert got is not None
    assert got.status == status
    assert got.reasons[0].code == "graph-check"


@pytest.mark.parametrize(
    "axis",
    ["graph_construction", "graph_alignment", "contribution"],
)
def test_optional_graph_evidence_axes_reject_statuses_from_other_axes(axis):
    from schemas.final_manifest import DemoEvidenceStatus

    payload = _graph_free_evidence_status()
    payload[axis] = {"status": "valid", "reasons": []}

    with pytest.raises(ValueError, match=axis):
        DemoEvidenceStatus.model_validate(payload)


def test_portable_manifest_schema_exposes_graph_axes_as_optional_mechanism_axes():
    from schemas.final_manifest import DemoEvidenceStatus, FinalManifest

    schema = FinalManifest.model_json_schema()
    evidence = schema["$defs"]["DemoEvidenceStatus"]
    for axis in ("graph_construction", "graph_alignment", "contribution"):
        assert axis in evidence["properties"]
        assert axis not in evidence["required"]
        refs = {
            item["$ref"]
            for item in evidence["properties"][axis]["anyOf"]
            if "$ref" in item
        }
        assert refs == {"#/$defs/MechanismEvidenceAxis"}

    assert DemoEvidenceStatus.model_validate(
        _graph_free_evidence_status()
    ).model_dump() == _graph_free_evidence_status()


@pytest.mark.parametrize(
    ("axis", "status"),
    [
        ("graph_alignment", "undetermined"),
        ("graph_construction", "not_demonstrated"),
        ("contribution", "not_demonstrated"),
        ("mechanism", "not_demonstrated"),
    ],
)
def test_graph_bearing_verified_manifest_requires_the_full_demonstrated_ladder(
    axis, status,
):
    from schemas.final_manifest import DeliveryVerdict

    evidence = _graph_free_evidence_status()
    evidence["mechanism"] = {"status": "demonstrated", "reasons": []}
    for graph_axis in (
        "graph_alignment", "graph_construction", "contribution"
    ):
        evidence[graph_axis] = {"status": "demonstrated", "reasons": []}
    evidence[axis] = {"status": status, "reasons": []}
    payload = {
        "schema_version": "1.6.0",
        "partial": False,
        "label": "verified",
        "reasons": [],
        "disclosures": [],
        "probe_counts": {},
        "stubbed_elements": [],
        "demo_verdict": {
            "schema_version": "2.0.0",
            "verdict": "succeeded",
            "evidence_status": evidence,
        },
    }

    with pytest.raises(ValueError, match="full|requires demonstrated"):
        DeliveryVerdict.model_validate(payload)


def test_graph_bearing_verified_manifest_accepts_all_demonstrated_axes():
    from schemas.final_manifest import DeliveryVerdict

    evidence = _graph_free_evidence_status()
    evidence["mechanism"] = {"status": "demonstrated", "reasons": []}
    for graph_axis in (
        "graph_alignment", "graph_construction", "contribution"
    ):
        evidence[graph_axis] = {"status": "demonstrated", "reasons": []}

    parsed = DeliveryVerdict.model_validate({
        "schema_version": "1.6.0",
        "partial": False,
        "label": "verified",
        "reasons": [],
        "disclosures": [],
        "probe_counts": {},
        "stubbed_elements": [],
        "demo_verdict": {
            "schema_version": "2.0.0",
            "verdict": "succeeded",
            "evidence_status": evidence,
        },
    })

    assert parsed.label == "verified"


def _current_graph_delivery_payload() -> dict:
    evidence = _graph_free_evidence_status()
    evidence["mechanism"] = {"status": "demonstrated", "reasons": []}
    evidence["skill"] = {"status": "undetermined", "reasons": []}
    for graph_axis in (
        "graph_alignment", "graph_construction", "contribution",
    ):
        evidence[graph_axis] = {"status": "demonstrated", "reasons": []}
    return {
        "schema_version": "1.7.0",
        "graph_evidence_applicability": "homogeneous_graph_v1",
        "partial": False,
        "label": "verified",
        "reasons": [],
        "disclosures": [],
        "probe_counts": {"pass": 7},
        "stubbed_elements": [],
        "demo_verdict": {
            "schema_version": "2.0.0",
            "verdict": "succeeded",
            "evidence_status": evidence,
        },
    }


def _current_manifest_payload(delivery: dict | None) -> dict:
    return {
        "schema_version": "1.9.0",
        "run_id": "graph-run",
        "paper_slug": "graph-paper",
        "generated_at": "2026-08-10T00:00:00Z",
        "git_commit": "abc123",
        "config_hash": "sha256:test",
        "run_status": "passed",
        "stage_summary": [],
        "artifacts": [],
        "validation_summary": [],
        "references": {"final_manifest": "final_manifest.json"},
        "diagnostics": [],
        "delivery": delivery,
    }


def test_current_graph_manifest_keeps_graph_skill_and_benchmark_axes_distinct():
    from schemas.final_manifest import DeliveryVerdict

    parsed = DeliveryVerdict.model_validate(_current_graph_delivery_payload())
    evidence = parsed.demo_verdict.evidence_status

    assert evidence.graph_alignment.status == "demonstrated"
    assert evidence.graph_construction.status == "demonstrated"
    assert evidence.mechanism.status == "demonstrated"
    assert evidence.contribution.status == "demonstrated"
    assert evidence.skill.status == "undetermined"
    assert evidence.paper_benchmark.status == "not_assessed"


@pytest.mark.parametrize(
    "removed",
    [
        "graph_evidence_applicability",
        "demo_verdict",
        "graph_alignment",
        "graph_construction",
        "contribution",
    ],
)
def test_current_graph_manifest_rejects_removed_structured_evidence(removed):
    from schemas.final_manifest import DeliveryVerdict

    payload = _current_graph_delivery_payload()
    if removed in {"graph_evidence_applicability", "demo_verdict"}:
        payload.pop(removed)
    else:
        payload["demo_verdict"]["evidence_status"].pop(removed)

    with pytest.raises(
        ValueError, match="applicability|homogeneous-graph|separate"
    ):
        DeliveryVerdict.model_validate(payload)


@pytest.mark.parametrize("malformed", ["1.7", "1.7.x", "banana"])
def test_graph_delivery_rejects_malformed_version_before_evidence_gating(
    malformed,
):
    from schemas.final_manifest import DeliveryVerdict

    payload = _current_graph_delivery_payload()
    payload["schema_version"] = malformed
    payload.pop("graph_evidence_applicability")
    payload.pop("demo_verdict")

    with pytest.raises(ValueError, match="schema_version"):
        DeliveryVerdict.model_validate(payload)


@pytest.mark.parametrize(
    "removed",
    [
        "graph_evidence_applicability",
        "demo_verdict",
        "graph_alignment",
        "graph_construction",
        "contribution",
    ],
)
def test_exported_manifest_schemas_reject_removed_graph_axes(removed):
    from schemas.final_manifest import FinalManifest

    generated = FinalManifest.model_json_schema()
    checked_path = (
        Path(__file__).resolve().parents[1]
        / "schemas" / "final_manifest.schema.json"
    )
    checked = json.loads(checked_path.read_text(encoding="utf-8"))
    assert checked["$defs"]["DeliveryVerdict"]["allOf"] == (
        generated["$defs"]["DeliveryVerdict"]["allOf"]
    )

    delivery = _current_graph_delivery_payload()
    if removed in {"graph_evidence_applicability", "demo_verdict"}:
        delivery.pop(removed)
    else:
        delivery["demo_verdict"]["evidence_status"].pop(removed)
    manifest = _current_manifest_payload(delivery)

    for schema in (generated, checked):
        errors = list(Draft202012Validator(schema).iter_errors(manifest))
        assert errors, f"schema accepted graph delivery without {removed}"


@pytest.mark.parametrize(
    "axis",
    ["graph_alignment", "graph_construction", "mechanism", "contribution"],
)
def test_exported_manifest_schemas_reject_verified_graph_axis_without_proof(axis):
    from schemas.final_manifest import FinalManifest

    generated = FinalManifest.model_json_schema()
    checked_path = (
        Path(__file__).resolve().parents[1]
        / "schemas" / "final_manifest.schema.json"
    )
    checked = json.loads(checked_path.read_text(encoding="utf-8"))
    delivery = _current_graph_delivery_payload()
    delivery["demo_verdict"]["evidence_status"][axis]["status"] = (
        "undetermined"
    )
    manifest = _current_manifest_payload(delivery)

    for schema in (generated, checked):
        errors = list(Draft202012Validator(schema).iter_errors(manifest))
        assert errors, f"schema accepted verified graph with {axis} undetermined"


@pytest.mark.parametrize(
    ("spec_graph_bearing", "declared_applicability"),
    [
        (True, "not_applicable"),
        (False, "homogeneous_graph_v1"),
    ],
)
def test_run_validator_binds_delivery_graph_scope_to_method_spec(
    tmp_path, spec_graph_bearing, declared_applicability
):
    from final_manifest import validate_final_manifest_file

    run = tmp_path / "run"
    pipeline = run / ".pipeline"
    details = run / "details"
    pipeline.mkdir(parents=True)
    details.mkdir()
    spec = {"schema_version": "1.14.0"}
    if spec_graph_bearing:
        spec["methodology_replication_contract"] = {
            "homogeneous_graph_mechanism": {},
        }
    _write_json(pipeline / "method_spec.json", spec)
    delivery = _current_graph_delivery_payload()
    delivery["graph_evidence_applicability"] = declared_applicability
    if declared_applicability == "not_applicable":
        for axis in (
            "graph_alignment", "graph_construction", "contribution",
        ):
            delivery["demo_verdict"]["evidence_status"].pop(axis)
    manifest_path = details / "final_manifest.json"
    _write_json(manifest_path, _current_manifest_payload(delivery))

    diagnostics = validate_final_manifest_file(manifest_path, run_dir=run)

    assert any(
        "graph_evidence_applicability disagrees" in message
        for message in diagnostics
    )


def test_manifest_builder_refuses_graph_scope_disagreement(tmp_path):
    from final_manifest import FinalManifestError, build_final_manifest

    run = tmp_path / "run"
    paths = make_paths(run)
    paths.pipeline_dir.mkdir(parents=True, exist_ok=True)
    _write_json(paths.method_spec, {
        "schema_version": "1.14.0",
        "methodology_replication_contract": {
            "homogeneous_graph_mechanism": {},
        },
    })
    delivery = _current_graph_delivery_payload()
    delivery["graph_evidence_applicability"] = "not_applicable"
    for axis in ("graph_alignment", "graph_construction", "contribution"):
        delivery["demo_verdict"]["evidence_status"].pop(axis)

    with pytest.raises(FinalManifestError, match="method_spec.json"):
        build_final_manifest(paths, delivery=delivery)


def test_graph_method_spec_rejects_missing_delivery_on_build_and_validation(
    tmp_path,
):
    from final_manifest import (
        FinalManifestError,
        build_final_manifest,
        validate_final_manifest_file,
    )

    run = tmp_path / "run"
    paths = make_paths(run)
    _write_json(paths.method_spec, {
        "schema_version": "1.14.0",
        "methodology_replication_contract": {
            "homogeneous_graph_mechanism": {},
        },
    })

    with pytest.raises(FinalManifestError, match="delivery is missing"):
        build_final_manifest(paths, delivery=None)

    manifest_path = run / "details" / "final_manifest.json"
    _write_json(manifest_path, _current_manifest_payload(None))
    diagnostics = validate_final_manifest_file(manifest_path, run_dir=run)
    assert any("delivery is missing" in message for message in diagnostics)


def test_current_graph_scope_requires_readable_method_spec(tmp_path):
    from final_manifest import validate_final_manifest_file

    run = tmp_path / "run"
    manifest_path = run / "details" / "final_manifest.json"
    _write_json(
        manifest_path,
        _current_manifest_payload(_current_graph_delivery_payload()),
    )

    diagnostics = validate_final_manifest_file(manifest_path, run_dir=run)
    assert any(
        "method_spec.json is missing or unreadable" in message
        for message in diagnostics
    )


@pytest.mark.parametrize("malformed", ["1.7", "1.7.x", "banana"])
def test_exported_manifest_schemas_reject_malformed_delivery_version(malformed):
    from schemas.final_manifest import FinalManifest

    generated = FinalManifest.model_json_schema()
    checked_path = (
        Path(__file__).resolve().parents[1]
        / "schemas" / "final_manifest.schema.json"
    )
    checked = json.loads(checked_path.read_text(encoding="utf-8"))
    delivery = _current_graph_delivery_payload()
    delivery["schema_version"] = malformed
    delivery.pop("graph_evidence_applicability")
    delivery.pop("demo_verdict")
    manifest = _current_manifest_payload(delivery)

    for schema in (generated, checked):
        errors = list(Draft202012Validator(schema).iter_errors(manifest))
        assert errors, f"schema accepted malformed delivery version {malformed!r}"


def test_final_manifest_records_hashes_and_passes_for_complete_run(run_dir):
    from final_manifest import validate_final_manifest_file, write_final_manifest

    paths = _populate_delivery_artifacts(run_dir)
    _write_json(paths.pipeline_dir / "probe_report.json", {"verdicts": []})
    _write_json(paths.pipeline_dir / "claims_ledger.json", {"claims": []})
    _write_text(run_dir / "details" / "deferred_findings.md", "# Findings deferred\n")
    _write_text(run_dir / "details" / "KNOWN_ISSUES.md", "# Known issues\n")
    _write_json(paths.pipeline_dir / "progress.json", {"run_status": "running"})
    results = _stage_results()
    _write_driver_state(run_dir, results)
    _write_event_log(paths, results)

    manifest = write_final_manifest(
        paths,
        stage_results=results,
        git_commit="abc123",
        generated_at="2026-06-03T00:00:00Z",
    )

    assert manifest.schema_version == "1.9.0"
    assert manifest.run_status == "passed"
    assert manifest.git_commit == "abc123"
    method_row = next(row for row in manifest.artifacts if row.path == "method/method.py")
    assert method_row.status == "present"
    assert method_row.sha256
    assert method_row.byte_size == (run_dir / "method" / "method.py").stat().st_size
    report_row = next(row for row in manifest.artifacts if row.path == "REPORT.md")
    assert report_row.required_for_delivery is True
    event_row = next(row for row in manifest.artifacts if row.path == ".pipeline/run_events.jsonl")
    assert event_row.sha256 is None
    assert event_row.byte_size is None
    progress_row = next(row for row in manifest.artifacts if row.path == ".pipeline/progress.json")
    assert progress_row.sha256 is None
    assert progress_row.byte_size is None
    # driver_state.json is driver-managed runtime state, same class as the
    # event log and progress file: the driver rewrites it AFTER packaging
    # (recording the terminal stage), so a recorded hash is stale by design.
    state_row = next(row for row in manifest.artifacts
                     if row.path == ".pipeline/driver_state.json")
    assert state_row.sha256 is None
    assert state_row.byte_size is None
    scaling_row = next(
        row for row in manifest.artifacts
        if row.path == ".pipeline/target_scaling_state.json"
    )
    assert scaling_row.status == "missing"
    assert scaling_row.required_for_delivery is False
    history_row = next(
        row for row in manifest.artifacts
        if row.path == ".pipeline/training_history.json"
    )
    assert history_row.status == "missing"
    assert history_row.required_for_delivery is False
    assert manifest.references.report_md == "REPORT.md"
    assert manifest.references.probe_report == ".pipeline/probe_report.json"
    assert manifest.references.claims_ledger == ".pipeline/claims_ledger.json"
    assert manifest.references.deferred_findings_md == "details/deferred_findings.md"
    assert manifest.references.known_issues_md == "details/KNOWN_ISSUES.md"
    assert validate_final_manifest_file(run_dir / "details" / "final_manifest.json") == []

    from run_events import append_event
    append_event(
        paths.pipeline_dir,
        event_type="run_finished",
        run_id=paths.slug,
        status="completed",
        summary="R2C run finished with status=completed",
        timestamp="2026-06-03T01:00:00Z",
    )
    _write_json(paths.pipeline_dir / "progress.json", {"run_status": "completed"})
    assert validate_final_manifest_file(run_dir / "details" / "final_manifest.json") == []


def test_final_manifest_includes_training_history_only_when_present(run_dir):
    from final_manifest import write_final_manifest

    paths = _populate_delivery_artifacts(run_dir)
    results = _stage_results()
    _write_driver_state(run_dir, results)
    _write_event_log(paths, results)

    missing = write_final_manifest(
        paths, stage_results=results, git_commit="abc123"
    )
    missing_row = next(
        row for row in missing.artifacts
        if row.path == ".pipeline/training_history.json"
    )
    assert missing_row.status == "missing"
    assert missing_row.required_for_delivery is False
    assert missing_row.sha256 is None

    _write_json(
        paths.pipeline_dir / "training_history.json",
        {"schema_version": "1.0.0", "status": "recorded"},
    )
    present = write_final_manifest(
        paths, stage_results=results, git_commit="abc123"
    )
    present_row = next(
        row for row in present.artifacts
        if row.path == ".pipeline/training_history.json"
    )
    assert present_row.status == "present"
    assert present_row.required_for_delivery is False
    assert present_row.sha256
    assert present_row.byte_size == (
        paths.pipeline_dir / "training_history.json"
    ).stat().st_size

def test_final_manifest_hash_verification_catches_tamper(run_dir):
    from final_manifest import validate_final_manifest_file, write_final_manifest

    paths = _populate_delivery_artifacts(run_dir)
    results = _stage_results()
    _write_driver_state(run_dir, results)
    _write_event_log(paths, results)
    write_final_manifest(paths, stage_results=results, git_commit="abc123")

    _write_text(run_dir / "method" / "method.py", "def select_batch():\n    return [1]\n")

    diagnostics = validate_final_manifest_file(run_dir / "details" / "final_manifest.json")
    assert any("method/method.py: sha256 mismatch" in item for item in diagnostics)


def test_terminal_driver_state_write_is_not_a_baseline_violation(run_dir):
    # The A8-interlude false alarm: on a halt, packaging writes the manifest
    # and only THEN does main() record the halted stage into
    # driver_state.json, so the delivered-baseline re-verify saw the
    # driver's own bookkeeping as an integrity violation on every
    # halt-packaged run (delivered_baseline_blocked x3, 08-20..08-24).
    from run_pipeline import (StageResult,
                              _delivered_manifest_integrity_violations)
    from final_manifest import write_final_manifest

    paths = _populate_delivery_artifacts(run_dir)
    results = _stage_results()
    _write_driver_state(run_dir, results)
    _write_event_log(paths, results)
    write_final_manifest(paths, stage_results=results, git_commit="abc123")

    # The terminal record() after packaging rewrites driver_state.json.
    _write_driver_state(run_dir, [*results, StageResult(
        status="halted", stage_id="stage_2x", notes="post-manifest record")])

    assert _delivered_manifest_integrity_violations(run_dir) == []


def test_final_manifest_missing_required_artifact_blocks(run_dir):
    from final_manifest import write_final_manifest

    paths = _populate_delivery_artifacts(run_dir, include_method_py=False)
    results = _stage_results()
    _write_driver_state(run_dir, results)
    _write_event_log(paths, results)

    manifest = write_final_manifest(paths, stage_results=results, git_commit="abc123")

    assert manifest.run_status == "blocked"
    method_row = next(row for row in manifest.artifacts if row.path == "method/method.py")
    assert method_row.status == "missing"
    assert method_row.validation_status == "missing"
    assert any("method/method.py: required delivery artifact missing" in item for item in manifest.diagnostics)


def test_final_manifest_halt_reports_not_reached_not_missing(run_dir):
    # L1 (audit 2026-06-25): on a run that halted upstream, downstream delivery
    # artifacts were never produced. Both the manifest and its validator must
    # report them as "not reached", not "missing" — the latter misreads as a
    # delivery failure on a run that simply stopped early.
    from final_manifest import validate_final_manifest_file, write_final_manifest

    from run_pipeline import STAGES, StageResult

    paths = _populate_delivery_artifacts(run_dir)
    (run_dir / "REPORT.md").unlink()  # stage_5 deliverable never produced
    # A real halt records stages only up to the halt; later stages never run and
    # are simply absent from driver_state (the manifest derives them as missing).
    ran = STAGES[: STAGES.index("stage_2x") + 1]
    results = [
        StageResult(status=("halted" if s == "stage_2x" else "completed"),
                    stage_id=s, notes=f"{s} test result")
        for s in ran
    ]
    _write_driver_state(run_dir, results)
    _write_event_log(paths, results)

    manifest = write_final_manifest(paths, stage_results=results, git_commit="abc123")

    assert manifest.run_status == "blocked"
    assert any("REPORT.md" in d and "not produced — run halted at stage_2x" in d
               for d in manifest.diagnostics)
    assert not any("REPORT.md: required delivery artifact missing" in d
                   for d in manifest.diagnostics)
    vdiags = validate_final_manifest_file(run_dir / "details" / "final_manifest.json", run_dir=run_dir)
    assert any("REPORT.md" in d and "not produced — run halted at stage_2x" in d
               for d in vdiags)


def _explanation_only_delivery(stage_id: str = "stage_2x") -> dict:
    return {
        "schema_version": "1.0.0",
        "label": "explanation_only",
        "reasons": [{
            "source": "halt",
            "id": stage_id,
            "message": "review found findings that could not be auto-resolved",
        }],
        "disclosures": [],
        "probe_counts": {},
    }


def test_final_manifest_explanation_only_requires_only_explanation_tier(run_dir):
    # 2026-07-02 batch: four honest explanation-only packages failed manifest
    # validation demanding notebook.ipynb / review_report from stages the run
    # never reached. The required-artifact set keys off the delivery label:
    # the explanation tier (REPORT.md + parsed paper) is the delivery.
    from final_manifest import validate_final_manifest_file, write_final_manifest

    from run_pipeline import STAGES, StageResult

    paths = make_paths(run_dir)
    _write_text(run_dir / "REPORT.md", "# Run Report\n")
    _write_text(run_dir / "METHOD.md", "# Method\n")
    ran = STAGES[: STAGES.index("stage_2x") + 1]
    results = [
        StageResult(status=("halted" if s == "stage_2x" else "completed"),
                    stage_id=s, notes=f"{s} test result")
        for s in ran
    ]
    _write_driver_state(run_dir, results)
    _write_event_log(paths, results)

    manifest = write_final_manifest(
        paths, stage_results=results, git_commit="abc123",
        delivery=_explanation_only_delivery(),
    )

    assert manifest.run_status == "degraded"
    notebook_row = next(r for r in manifest.artifacts if r.path == "notebook.ipynb")
    assert notebook_row.required_for_delivery is False
    review_row = next(r for r in manifest.artifacts
                      if r.path == ".pipeline/review_report.json")
    assert review_row.required_for_delivery is False
    report_row = next(r for r in manifest.artifacts if r.path == "REPORT.md")
    assert report_row.required_for_delivery is True
    paper_row = next(r for r in manifest.artifacts if r.path == ".pipeline/paper.md")
    assert paper_row.required_for_delivery is True
    # The halt and the not-reached stages stay disclosed as diagnostics.
    assert any("stage_2x: stage halted" in d for d in manifest.diagnostics)
    assert any("not reached — run halted at stage_2x" in d
               for d in manifest.diagnostics)
    # The post-hoc validator agrees: nothing blocking on an honest package.
    assert validate_final_manifest_file(
        run_dir / "details" / "final_manifest.json", run_dir=run_dir) == []


def test_final_manifest_explanation_only_still_blocks_without_report(run_dir):
    # Fail-closed stays: the explanation-only front door (REPORT.md) is the
    # one artifact the package cannot be delivered without.
    from final_manifest import write_final_manifest

    from run_pipeline import STAGES, StageResult

    paths = make_paths(run_dir)
    ran = STAGES[: STAGES.index("stage_2x") + 1]
    results = [
        StageResult(status=("halted" if s == "stage_2x" else "completed"),
                    stage_id=s, notes=f"{s} test result")
        for s in ran
    ]
    _write_driver_state(run_dir, results)
    _write_event_log(paths, results)

    manifest = write_final_manifest(
        paths, stage_results=results, git_commit="abc123",
        delivery=_explanation_only_delivery(),
    )

    assert manifest.run_status == "blocked"
    assert any("REPORT.md" in d for d in manifest.diagnostics)


def test_final_manifest_code_deliveries_unaffected_by_label_keying(run_dir):
    # verified/draft deliveries keep the full required set: a missing
    # notebook still blocks.
    from final_manifest import write_final_manifest

    paths = _populate_delivery_artifacts(run_dir)
    (run_dir / "notebook.ipynb").unlink()
    results = _stage_results()
    _write_driver_state(run_dir, results)
    _write_event_log(paths, results)

    manifest = write_final_manifest(
        paths, stage_results=results, git_commit="abc123",
        delivery={
            "schema_version": "1.0.0",
            "label": "draft",
            "reasons": [],
            "disclosures": [],
            "probe_counts": {},
        },
    )

    assert manifest.run_status == "blocked"
    assert any("notebook.ipynb: required delivery artifact missing" in d
               for d in manifest.diagnostics)


def test_final_manifest_skipped_stage_can_pass_when_artifacts_present(run_dir):
    from final_manifest import validate_final_manifest_file, write_final_manifest

    paths = _populate_delivery_artifacts(run_dir)
    results = _stage_results({"stage_2c": "skipped"})
    _write_driver_state(run_dir, results)
    _write_event_log(paths, results)

    manifest = write_final_manifest(paths, stage_results=results, git_commit="abc123")

    assert manifest.run_status == "passed"
    method_stage = next(row for row in manifest.stage_summary if row.stage_id == "stage_2c")
    assert method_stage.status == "skipped"
    method_row = next(row for row in manifest.artifacts if row.path == "method/method.py")
    assert method_row.validation_status == "skipped"
    assert manifest.diagnostics == []
    assert validate_final_manifest_file(run_dir / "details" / "final_manifest.json") == []


def test_final_manifest_degraded_stage_still_degrades(run_dir):
    from final_manifest import write_final_manifest

    paths = _populate_delivery_artifacts(run_dir)
    results = _stage_results({"stage_3c": "degraded"})
    _write_driver_state(run_dir, results)
    _write_event_log(paths, results)

    manifest = write_final_manifest(paths, stage_results=results, git_commit="abc123")

    assert manifest.run_status == "degraded"
    assert any("stage_3c: stage completed with known issues" in item for item in manifest.diagnostics)


def test_delivery_verdict_tolerates_old_two_value_shape():
    # Design note §4 (third label): old manifests written before the
    # uncertified_new_territory label — schema_version 1.0.0, no
    # missing_probe_family, no evidence field on reasons — must stay valid.
    from schemas.final_manifest import DeliveryVerdict

    old = DeliveryVerdict.model_validate({
        "schema_version": "1.0.0",
        "label": "draft",
        "reasons": [{"source": "probe", "id": "UB-7", "verdict": "fail",
                     "message": "dead term"}],
        "disclosures": [],
        "probe_counts": {"pass": 3, "fail": 1},
    })
    assert old.label == "draft"
    assert old.missing_probe_family is None

    new = DeliveryVerdict.model_validate({
        "schema_version": "1.1.0",
        "label": "uncertified_new_territory",
        "reasons": [],
        "disclosures": [],
        "probe_counts": {"pass": 5},
        "missing_probe_family": "trajectory_forecasting",
    })
    assert new.missing_probe_family == "trajectory_forecasting"


def test_disclosure_keeps_observed_verdict_from_binding_gap_demotion():
    """Writer/schema drift guard for DeliveryReason (deep-batch 2026-08-31).

    delivery_label.py's binding-gap path (12a890616) demotes a passing
    contribution check to verdict="binding_gap" while recording what the
    check itself reported as observed_verdict="pass". The strict manifest
    schema rejected that field for three weeks until the first run hit the
    path, failing finalization AFTER every content stage had completed.
    Known-good: the exact shape the writer emits validates. Known-bad:
    extra=forbid still rejects fields the writer does not emit.
    """
    import pydantic
    from schemas.final_manifest import DeliveryVerdict

    verdict = DeliveryVerdict.model_validate({
        "schema_version": "1.1.0",
        "label": "draft",
        "reasons": [],
        "disclosures": [{
            "source": "probe",
            "id": "al_loop.microharness",
            "verdict": "binding_gap",
            "observed_verdict": "pass",
            "message": "unanchored coverage, not certification evidence",
        }],
        "probe_counts": {"pass": 12},
    })
    assert verdict.disclosures[0].observed_verdict == "pass"

    with pytest.raises(pydantic.ValidationError):
        DeliveryVerdict.model_validate({
            "schema_version": "1.1.0",
            "label": "draft",
            "reasons": [],
            "disclosures": [{
                "source": "probe",
                "id": "x",
                "message": "m",
                "not_a_field": True,
            }],
            "probe_counts": {},
        })
