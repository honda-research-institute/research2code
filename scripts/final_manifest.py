#!/usr/bin/env python3
"""Build and validate main-owned final manifests for R2C runs."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from run_events import EventLogError, canonical_json, replay_run_dir, utc_now_iso
import run_layout
from schemas.final_manifest import FinalManifest, SCHEMA_VERSION


REQUIRED_STAGES = [
    "stage_0",
    "stage_1",
    "stage_1x",
    "stage_2a",
    "stage_2b",
    "stage_2c",
    "stage_2d",
    "stage_2x",
    "stage_3a",
    "stage_3b",
    "stage_3c",
    "stage_4",
    "stage_5",
]

STAGE_CATEGORIES = {
    "stage_0": "setup_analysis",
    "stage_1": "setup_analysis",
    "stage_1x": "setup_analysis",
    "stage_2a": "package_generation",
    "stage_2b": "package_generation",
    "stage_2c": "package_generation",
    "stage_2d": "package_generation",
    "stage_2x": "package_generation",
    "stage_3a": "notebook_smoke",
    "stage_3b": "notebook_smoke",
    "stage_3c": "notebook_smoke",
    "stage_4": "review_routing",
    "stage_5": "review_routing",
}

STAGE_LABELS = {
    "stage_0": "Stage 0 - Setup & Paper Ingestion",
    "stage_1": "Stage 1 - Paper Decomposition & Method Analysis",
    "stage_1x": "Stage 1.x - Method Explanation (METHOD.md)",
    "stage_2a": "Stage 2.a - Package Scaffold",
    "stage_2b": "Stage 2.b - Architecture & Training",
    "stage_2c": "Stage 2.c - Method Implementation",
    "stage_2d": "Stage 2.d - Package Finalization",
    "stage_2x": "Stage 2.x - Parameter Derivation",
    "stage_3a": "Stage 3.a - Notebook Authoring",
    "stage_3b": "Stage 3.b - Notebook Rendering",
    "stage_3c": "Stage 3.c - Smoke Execution",
    "stage_4": "Stage 4 - Paper-Fidelity Review",
    "stage_5": "Stage 5 - Findings Routing",
}


class FinalManifestError(RuntimeError):
    """Raised when a final manifest cannot be read or written."""


@dataclass(frozen=True)
class ArtifactSpec:
    rel_path: str
    kind: str
    display_name: str
    producer_stage: str
    required_for_delivery: bool = True


ARTIFACT_SPECS = [
    ArtifactSpec("REPORT.md", "report_md", "Run Report", "stage_5"),
    ArtifactSpec(".pipeline/paper.md", "paper_markdown", "Parsed Paper", "stage_0"),
    ArtifactSpec(".pipeline/paper_map.json", "paper_map", "Paper Map", "stage_1"),
    ArtifactSpec(".pipeline/method_spec.json", "method_spec", "Method Spec", "stage_1"),
    ArtifactSpec(".pipeline/feasibility_gate.json", "feasibility_gate", "Feasibility Gate", "stage_1"),
    # Best-effort by design (stage 1x degrades, never halts) — its absence
    # must not block an otherwise-deliverable run.
    ArtifactSpec("METHOD.md", "method_md", "Method Explanation", "stage_1x", False),
    ArtifactSpec(".pipeline/method_explanations.json", "method_explanations",
                 "Explanation Sidecar", "stage_1x", False),
    ArtifactSpec(run_layout.PACKAGE_README, "researcher_readme", "Package README", "stage_2a"),
    ArtifactSpec("method/data.py", "method_file", "Method Data Loader", "stage_2a"),
    ArtifactSpec("method/example_data/README.md", "method_data_format", "Example Data README", "stage_2a"),
    ArtifactSpec(".pipeline/arch_contract.json", "arch_contract", "Architecture Contract", "stage_2b"),
    ArtifactSpec("method/model.py", "method_file", "Model Code", "stage_2b"),
    ArtifactSpec("method/training.py", "method_file", "Training Code", "stage_2b"),
    ArtifactSpec("method/method.py", "method_file", "Method Implementation", "stage_2c"),
    ArtifactSpec("method/__init__.py", "method_file", "Method Package Init", "stage_2d"),
    ArtifactSpec("requirements.txt", "requirements", "Runtime Requirements", "stage_2d"),
    # Per-element tests (R2C-024): optional by construction — the README
    # is the honest coverage index (present whenever the step ran, even
    # with zero shipped tests); the results record carries the statuses,
    # drops, and mutation labels the README rows summarize.
    ArtifactSpec("method/tests/README.md", "test_coverage_index",
                 "Per-Element Test Coverage", "stage_2d", False),
    ArtifactSpec(".pipeline/element_tests.json", "element_tests",
                 "Per-Element Test Results", "stage_2d", False),
    ArtifactSpec(".pipeline/params.json", "params", "Runtime Parameters", "stage_2x"),
    ArtifactSpec(".pipeline/notebook_draft.py", "notebook_source", "Notebook Source", "stage_3a"),
    ArtifactSpec("notebook.ipynb", "notebook", "Rendered Notebook", "stage_3b"),
    ArtifactSpec(
        ".pipeline/target_scaling_state.json",
        "target_scaling_state",
        "Training-Only Target Scaling State",
        "stage_3c",
        False,
    ),
    ArtifactSpec(
        ".pipeline/training_history.json",
        "training_history",
        "Structured Training and Selection History",
        "stage_3c",
        False,
    ),
    ArtifactSpec(".pipeline/review_report.json", "review_report", "Paper-Fidelity Review", "stage_4", False),
    ArtifactSpec(run_layout.DEFERRED_FINDINGS_MD, "deferred_findings", "Deferred Findings", "stage_5", False),
    ArtifactSpec(run_layout.ASSUMPTIONS_MD, "assumptions", "Assumptions Audit", "stage_5", False),
    ArtifactSpec(run_layout.KNOWN_ISSUES_MD, "known_issues", "Known Issues", "stage_5", False),
    ArtifactSpec(".pipeline/driver_state.json", "driver_state", "Driver State", "stage_5", False),
    ArtifactSpec(".pipeline/run_events.jsonl", "event_log", "Run Event Log", "stage_5", False),
    ArtifactSpec(".pipeline/progress.json", "progress_state", "Progress State", "stage_5", False),
]

# Driver-managed runtime files the driver keeps writing AFTER the manifest
# is packaged (terminal events, progress, and the terminal stage record into
# driver_state.json on the halt path), so a recorded hash is stale by design.
MUTABLE_ARTIFACT_KINDS = {"event_log", "progress_state", "driver_state"}

# The explanation-only delivery (recentering 2.4) promises the understanding
# tier, not code: REPORT.md as the front door plus the parsed paper it
# explains. Code/notebook artifacts a halted run cannot have must not block
# ITS manifest — the required-artifact set keys off the delivery label
# (2026-07-02 batch: four identical validation failures on honest
# explanation-only packages, each demanding notebook.ipynb from a run that
# halted at 2.x). METHOD.md stays best-effort even here (stage 1.x degrades
# by design, never blocks).
EXPLANATION_ONLY_REQUIRED_PATHS = {"REPORT.md", ".pipeline/paper.md"}


def _delivery_label(delivery: dict[str, Any] | None) -> str | None:
    if isinstance(delivery, dict):
        label = delivery.get("label")
        return label if isinstance(label, str) else None
    return None


def _method_spec_graph_applicability(run_dir: Path) -> str | None:
    """Return graph scope from the run's own method contract, when readable."""

    path = Path(run_dir) / ".pipeline" / "method_spec.json"
    try:
        spec = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(spec, dict):
        return None
    methodology = spec.get("methodology_replication_contract")
    graph = (
        methodology.get("homogeneous_graph_mechanism")
        if isinstance(methodology, dict) else None
    )
    return (
        "homogeneous_graph_v1"
        if isinstance(graph, dict)
        else "not_applicable"
    )


def _delivery_graph_scope_problem(
    run_dir: Path,
    delivery: dict[str, Any] | None,
) -> str | None:
    """Bind the manifest discriminator to method-spec graph authority."""

    expected = _method_spec_graph_applicability(run_dir)
    if not isinstance(delivery, dict):
        if expected == "homogeneous_graph_v1":
            return (
                "delivery is missing for a homogeneous graph contract in "
                ".pipeline/method_spec.json"
            )
        return None
    actual = delivery.get("graph_evidence_applicability")
    if expected is None:
        if actual is not None:
            return (
                "delivery.graph_evidence_applicability cannot be bound: "
                ".pipeline/method_spec.json is missing or unreadable"
            )
        return None
    raw_version = delivery.get("schema_version")
    try:
        version = tuple(int(part) for part in str(raw_version).split("."))
    except ValueError:
        # Pydantic owns malformed version syntax. Avoid masking its more exact
        # contract error at the build boundary.
        version = ()
    # A typed homogeneous graph contract did not exist in archived delivery
    # schemas, so it always requires the graph discriminator. Graph-free
    # archives retain their old missing-discriminator compatibility; current
    # schema 1.7+ records must state not_applicable explicitly.
    comparison_required = (
        expected == "homogeneous_graph_v1"
        or actual is not None
        or version >= (1, 7, 0)
    )
    if not comparison_required:
        return None
    if actual == expected:
        return None
    return (
        "delivery.graph_evidence_applicability disagrees with "
        ".pipeline/method_spec.json: "
        f"expected {expected!r}, found {actual!r}"
    )


def _spec_required_for_delivery(
    spec: ArtifactSpec, delivery_label: str | None
) -> bool:
    if delivery_label == "explanation_only":
        return spec.rel_path in EXPLANATION_ONLY_REQUIRED_PATHS
    return spec.required_for_delivery


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _rel_to_run(run_dir: Path, path: Path) -> str:
    return path.relative_to(run_dir).as_posix()


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(
        dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            # Insertion order, NOT sort_keys: the pydantic declaration order
            # is the deliberate reading order — `delivery.partial` must be
            # the block's first rendered field (partial delivery §3.5
            # criterion 1). model_dump() order is deterministic, so diffs
            # stay stable.
            json.dump(payload, handle, indent=2)
            handle.write("\n")
        os.replace(tmp_path, path)
    except Exception:
        try:
            os.unlink(tmp_path)
        except FileNotFoundError:
            pass
        raise


def _git_commit(repo_root: Path) -> str:
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    if proc.returncode != 0:
        return "unknown"
    return proc.stdout.strip() or "unknown"


def _stage_result_rows(stage_results: list[Any] | None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for result in stage_results or []:
        stage_id = str(getattr(result, "stage_id", ""))
        if not stage_id:
            continue
        rows.append({
            "stage_id": stage_id,
            "stage_label": STAGE_LABELS.get(stage_id, stage_id),
            "stage_category": STAGE_CATEGORIES.get(stage_id),
            "status": _normalize_stage_status(getattr(result, "status", "missing")),
            "notes": getattr(result, "notes", ""),
            "started_at": None,
            "completed_at": None,
            "last_event_sequence": None,
        })
    return rows


def _normalize_stage_status(status: Any) -> str:
    text = str(status or "missing")
    if text in {"completed", "skipped", "halted", "degraded"}:
        return text
    return "missing"


def _driver_state_rows(pipeline_dir: Path) -> list[dict[str, Any]]:
    path = pipeline_dir / "driver_state.json"
    if not path.is_file():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return []
    rows: list[dict[str, Any]] = []
    for stage in data.get("stages", []):
        if not isinstance(stage, dict):
            continue
        stage_id = str(stage.get("stage_id") or "")
        if not stage_id:
            continue
        rows.append({
            "stage_id": stage_id,
            "stage_label": stage.get("stage_label") or STAGE_LABELS.get(stage_id, stage_id),
            "stage_category": STAGE_CATEGORIES.get(stage_id),
            "status": _normalize_stage_status(stage.get("status")),
            "notes": stage.get("notes") or "",
            "started_at": None,
            "completed_at": None,
            "last_event_sequence": None,
        })
    return rows


def _stage_rows(
    run_dir: Path,
    *,
    stage_results: list[Any] | None,
    diagnostics: list[str],
) -> list[dict[str, Any]]:
    pipeline_dir = run_dir / ".pipeline"
    rows_by_id: dict[str, dict[str, Any]] = {}
    try:
        replay = replay_run_dir(run_dir)
        for row in replay.get("stages", []):
            if not isinstance(row, dict):
                continue
            stage_id = row.get("stage_id")
            if not isinstance(stage_id, str):
                continue
            rows_by_id[stage_id] = {
                "stage_id": stage_id,
                "stage_label": row.get("stage_label") or STAGE_LABELS.get(stage_id, stage_id),
                "stage_category": row.get("stage_category") or STAGE_CATEGORIES.get(stage_id),
                "status": _normalize_stage_status(row.get("status")),
                "notes": "",
                "started_at": row.get("started_at"),
                "completed_at": row.get("completed_at"),
                "last_event_sequence": row.get("last_event_sequence"),
            }
    except EventLogError as exc:
        diagnostics.append(f"run_events.jsonl could not be replayed: {exc}")

    for row in _driver_state_rows(pipeline_dir):
        rows_by_id[row["stage_id"]] = {**rows_by_id.get(row["stage_id"], {}), **row}
    for row in _stage_result_rows(stage_results):
        rows_by_id[row["stage_id"]] = {**rows_by_id.get(row["stage_id"], {}), **row}

    rows: list[dict[str, Any]] = []
    for stage_id in REQUIRED_STAGES:
        row = rows_by_id.get(stage_id)
        if row is None:
            row = {
                "stage_id": stage_id,
                "stage_label": STAGE_LABELS.get(stage_id, stage_id),
                "stage_category": STAGE_CATEGORIES.get(stage_id),
                "status": "missing",
                "notes": "",
                "started_at": None,
                "completed_at": None,
                "last_event_sequence": None,
            }
        rows.append(row)
    return rows


def _validation_rows(run_dir: Path) -> list[dict[str, Any]]:
    try:
        replay = replay_run_dir(run_dir)
    except EventLogError:
        return []
    rows: list[dict[str, Any]] = []
    for event in replay.get("validations", []):
        if not isinstance(event, dict):
            continue
        event_type = event.get("event_type")
        if event_type == "validation_passed":
            status = "passed"
        elif event_type == "contract_halted":
            status = "blocked"
        elif event_type == "validation_failed":
            status = "failed"
        else:
            status = "unknown"
        stage_id = event.get("stage_id")
        details = event.get("details") if isinstance(event.get("details"), dict) else {}
        rows.append({
            "stage_id": stage_id,
            "stage_label": event.get("stage_label"),
            "validator": str(details.get("validator") or event.get("summary") or event_type),
            "status": status,
            "summary": str(event.get("summary") or ""),
            "artifacts": [str(item) for item in event.get("artifacts", [])],
            "event_sequence": event.get("sequence"),
        })
    return rows


def _artifact_validation_status(
    spec: ArtifactSpec,
    *,
    exists: bool,
    stage_statuses: dict[str, str],
) -> str:
    if not exists:
        return "missing"
    stage_status = stage_statuses.get(spec.producer_stage, "missing")
    if stage_status == "completed":
        return "passed"
    if stage_status == "skipped":
        return "skipped"
    if stage_status == "degraded":
        return "degraded"
    if stage_status == "halted":
        return "blocked"
    return "unknown"


def _artifact_rows(
    run_dir: Path,
    stage_statuses: dict[str, str],
    delivery_label: str | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for spec in ARTIFACT_SPECS:
        path = run_dir / spec.rel_path
        exists = path.is_file()
        row: dict[str, Any] = {
            "path": spec.rel_path,
            "kind": spec.kind,
            "display_name": spec.display_name,
            "producer_stage": spec.producer_stage,
            "producer_stage_label": STAGE_LABELS.get(spec.producer_stage, spec.producer_stage),
            "required_for_delivery": _spec_required_for_delivery(
                spec, delivery_label),
            "status": "present" if exists else "missing",
            "validation_status": _artifact_validation_status(
                spec, exists=exists, stage_statuses=stage_statuses
            ),
            "sha256": None,
            "byte_size": None,
        }
        if exists and spec.kind not in MUTABLE_ARTIFACT_KINDS:
            row["sha256"] = sha256_file(path)
            row["byte_size"] = path.stat().st_size
        rows.append(row)
    return rows


def _config_hash(paths: Any, git_commit: str) -> str:
    pipeline_dir = Path(paths.pipeline_dir)
    inputs: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "paper_slug": paths.slug,
        "input_kind": getattr(paths, "input_kind", "unknown"),
        "git_commit": git_commit,
        "stages": REQUIRED_STAGES,
    }
    for key, rel_path in {
        "paper_map_sha256": "paper_map.json",
        "method_spec_sha256": "method_spec.json",
        "params_sha256": "params.json",
    }.items():
        path = pipeline_dir / rel_path
        inputs[key] = sha256_file(path) if path.is_file() else None
    return sha256_text(canonical_json(inputs))


def _references(run_dir: Path) -> dict[str, str | None]:
    refs = {
        "report_md": run_layout.REPORT_MD,
        "method_spec": ".pipeline/method_spec.json",
        "assumptions_md": run_layout.ASSUMPTIONS_MD,
        "deferred_findings_md": run_layout.DEFERRED_FINDINGS_MD,
        "known_issues_md": run_layout.KNOWN_ISSUES_MD,
        "probe_report": ".pipeline/probe_report.json",
        "claims_ledger": ".pipeline/claims_ledger.json",
        "review_report": ".pipeline/review_report.json",
        "run_events": ".pipeline/run_events.jsonl",
        "driver_state": ".pipeline/driver_state.json",
        "final_manifest": run_layout.FINAL_MANIFEST_JSON,
    }
    return {
        key: rel_path if (run_dir / rel_path).is_file() else None
        for key, rel_path in refs.items()
    } | {"final_manifest": run_layout.FINAL_MANIFEST_JSON}


def _derive_run_status(
    *,
    stage_rows: list[dict[str, Any]],
    artifact_rows: list[dict[str, Any]],
    diagnostics: list[str],
    delivery_label: str | None = None,
) -> str:
    blocked = False
    degraded = False
    # For an explanation-only delivery the halt IS the delivery mode — it is
    # already recorded in delivery.reasons. Stages the run never reached are
    # facts to disclose (diagnostics), not missing delivery evidence, so
    # they degrade the manifest instead of blocking it. Only the
    # explanation-tier required artifacts (see EXPLANATION_ONLY_REQUIRED_PATHS)
    # can block. An explanation-only manifest is never "passed" — the halt
    # keeps it at least degraded.
    explanation_only = delivery_label == "explanation_only"
    # When the run halted, stages and artifacts AFTER the halt were never
    # reached. Reporting those as "missing / no recorded status" reads as a
    # delivery failure when the truth is "the run stopped upstream" (audit
    # 2026-06-25 L1). Resolve the halt point once and phrase not-reached
    # stages/artifacts in terms of it.
    halted_stage = next(
        (row["stage_id"] for row in stage_rows if row["status"] == "halted"),
        None,
    )
    reached = {row["stage_id"] for row in stage_rows if row["status"] != "missing"}
    for row in stage_rows:
        status = row["status"]
        stage_id = row["stage_id"]
        if status == "missing":
            if halted_stage is not None:
                diagnostics.append(
                    f"{stage_id}: not reached — run halted at {halted_stage}")
            else:
                diagnostics.append(f"{stage_id}: required stage has no recorded status")
            if explanation_only and halted_stage is not None:
                degraded = True
            else:
                blocked = True
        elif status == "halted":
            diagnostics.append(f"{stage_id}: stage halted")
            if explanation_only:
                degraded = True
            else:
                blocked = True
        elif status == "degraded":
            diagnostics.append(f"{stage_id}: stage completed with known issues")
            degraded = True
        elif status == "skipped":
            # Skipped stages are normal for resumed runs. The skipped status stays
            # visible in stage_summary, but a resumed run can still be clean when
            # delivery artifacts are present and the end-of-run validators pass.
            continue
    for row in artifact_rows:
        producer = row.get("producer_stage")
        not_reached = (
            halted_stage is not None
            and producer is not None
            and producer not in reached
        )
        if row["required_for_delivery"] and row["status"] != "present":
            if not_reached:
                diagnostics.append(
                    f"{row['path']}: not produced — run halted at {halted_stage} "
                    f"before its stage ({producer}) ran")
            else:
                diagnostics.append(f"{row['path']}: required delivery artifact missing")
            blocked = True
        if row["required_for_delivery"] and row["validation_status"] == "unknown":
            diagnostics.append(
                f"{row['path']}: required delivery artifact validation is {row['validation_status']}"
            )
            degraded = True
        if not row["required_for_delivery"] and row["status"] == "missing" and row["kind"] in {
            "review_report",
            "event_log",
            "driver_state",
        }:
            diagnostics.append(f"{row['path']}: audit artifact missing")
            degraded = True
    if blocked:
        return "blocked"
    if degraded or explanation_only:
        return "degraded"
    return "passed"


def build_final_manifest(
    paths: Any,
    *,
    stage_results: list[Any] | None = None,
    git_commit: str | None = None,
    generated_at: str | None = None,
    delivery: dict[str, Any] | None = None,
) -> FinalManifest:
    run_dir = Path(paths.run_dir)
    graph_scope_problem = _delivery_graph_scope_problem(run_dir, delivery)
    if graph_scope_problem is not None:
        raise FinalManifestError(graph_scope_problem)
    diagnostics: list[str] = []
    commit = git_commit or _git_commit(Path(paths.repo_root))
    stages = _stage_rows(run_dir, stage_results=stage_results, diagnostics=diagnostics)
    stage_statuses = {row["stage_id"]: row["status"] for row in stages}
    delivery_label = _delivery_label(delivery)
    artifacts = _artifact_rows(run_dir, stage_statuses, delivery_label)
    run_status = _derive_run_status(
        stage_rows=stages,
        artifact_rows=artifacts,
        diagnostics=diagnostics,
        delivery_label=delivery_label,
    )
    payload = {
        "schema_version": SCHEMA_VERSION,
        "run_id": paths.slug,
        "paper_slug": paths.slug,
        "generated_at": generated_at or utc_now_iso(),
        "git_commit": commit,
        "config_hash": _config_hash(paths, commit),
        "run_status": run_status,
        "stage_summary": stages,
        "artifacts": artifacts,
        "validation_summary": _validation_rows(run_dir),
        "references": _references(run_dir),
        "diagnostics": diagnostics,
        "delivery": delivery,
    }
    return FinalManifest.model_validate(payload)


def write_final_manifest(
    paths: Any,
    *,
    stage_results: list[Any] | None = None,
    git_commit: str | None = None,
    generated_at: str | None = None,
    delivery: dict[str, Any] | None = None,
) -> FinalManifest:
    manifest = build_final_manifest(
        paths,
        stage_results=stage_results,
        git_commit=git_commit,
        generated_at=generated_at,
        delivery=delivery,
    )
    _atomic_write_json(
        run_layout.run_path(Path(paths.run_dir), run_layout.FINAL_MANIFEST_JSON),
        manifest.model_dump(),
    )
    return manifest


def load_final_manifest(path: Path) -> FinalManifest:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FinalManifestError(f"could not load {path}: {exc}") from exc
    try:
        return FinalManifest.model_validate(data)
    except Exception as exc:
        raise FinalManifestError(f"{path} does not match final_manifest schema: {exc}") from exc


def validate_final_manifest_file(manifest_path: Path, *, run_dir: Path | None = None) -> list[str]:
    manifest = load_final_manifest(manifest_path)
    if run_dir is not None:
        root = run_dir
    elif manifest_path.parent.name == run_layout.DETAILS_DIRNAME:
        # Current layout: the manifest lives in <run_dir>/details/.
        root = manifest_path.parent.parent
    else:
        # Legacy pre-consolidation layout: manifest at the run root.
        root = manifest_path.parent
    diagnostics: list[str] = []
    graph_scope_problem = _delivery_graph_scope_problem(
        root,
        manifest.delivery.model_dump() if manifest.delivery is not None else None,
    )
    if graph_scope_problem is not None:
        diagnostics.append(graph_scope_problem)
    # Halt-aware reporting (L1): artifacts of stages never reached because the
    # run halted upstream are "not produced", not "missing".
    halted_stage = next(
        (s.stage_id for s in manifest.stage_summary if s.status == "halted"),
        None,
    )
    reached = {s.stage_id for s in manifest.stage_summary if s.status != "missing"}
    for artifact in manifest.artifacts:
        path = root / artifact.path
        if not path.is_file():
            if not artifact.required_for_delivery and artifact.status == "missing":
                continue
            if (halted_stage is not None
                    and artifact.producer_stage is not None
                    and artifact.producer_stage not in reached):
                diagnostics.append(
                    f"{artifact.path}: not produced — run halted at {halted_stage} "
                    f"before its stage ({artifact.producer_stage}) ran")
            else:
                diagnostics.append(f"{artifact.path}: referenced artifact missing")
            continue
        if artifact.status != "present":
            diagnostics.append(
                f"{artifact.path}: artifact exists but manifest status is {artifact.status}"
            )
            continue
        if artifact.kind in MUTABLE_ARTIFACT_KINDS:
            continue
        actual_sha = sha256_file(path)
        if artifact.sha256 != actual_sha:
            diagnostics.append(f"{artifact.path}: sha256 mismatch")
        actual_size = path.stat().st_size
        if artifact.byte_size != actual_size:
            diagnostics.append(f"{artifact.path}: byte_size mismatch")
    if (
        manifest.delivery is not None
        and manifest.delivery.label == "explanation_only"
        and manifest.run_status == "passed"
    ):
        diagnostics.append(
            "delivery label is explanation_only but run_status is passed — "
            "an explanation-only package is at most degraded (the halt that "
            "caused it is a known issue by construction)"
        )
    if manifest.run_status == "passed":
        if manifest.diagnostics:
            diagnostics.append("manifest run_status is passed but diagnostics is non-empty")
        stage_statuses = {
            stage.stage_id: stage.status
            for stage in manifest.stage_summary
        }
        for stage in manifest.stage_summary:
            if stage.status not in {"completed", "skipped"}:
                diagnostics.append(
                    f"{stage.stage_id}: manifest run_status is passed but stage status is {stage.status}"
                )
        for artifact in manifest.artifacts:
            allowed_statuses = {"passed"}
            if stage_statuses.get(artifact.producer_stage) == "skipped":
                allowed_statuses.add("skipped")
            if artifact.required_for_delivery and artifact.validation_status not in allowed_statuses:
                diagnostics.append(
                    f"{artifact.path}: manifest run_status is passed but validation_status is "
                    f"{artifact.validation_status}"
                )
    return diagnostics
