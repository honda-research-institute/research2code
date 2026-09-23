#!/usr/bin/env python3
"""Candidate artifact storage and promotion helpers for R2C.

This module is the Step 4 foundation only. It defines candidate storage,
validation-before-promotion, promoted artifact archives, and current
projections. It does not yet route live producer writes through candidates.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from final_manifest import sha256_file
from run_events import append_event, utc_now_iso


class CandidateStoreError(RuntimeError):
    """Raised when a candidate cannot be recorded or promoted."""


Validator = Callable[[Path], list[str]]


@dataclass(frozen=True)
class ArtifactRegistryEntry:
    stage_id: str
    artifact_id: str
    display_name: str
    canonical_rel_path: str
    current_filename: str
    suffix: str = ".json"
    required_for_delivery: bool = True
    validator_label: str | None = None


ARTIFACT_REGISTRY: dict[tuple[str, str], ArtifactRegistryEntry] = {
    (
        "stage_1",
        "method_spec",
    ): ArtifactRegistryEntry(
        stage_id="stage_1",
        artifact_id="method_spec",
        display_name="Stage 1 / Method Spec",
        canonical_rel_path=".pipeline/method_spec.json",
        current_filename="method_spec.json",
        validator_label="validate_method_spec.py --strict",
    ),
}


def artifact_entry(stage_id: str, artifact_id: str) -> ArtifactRegistryEntry:
    try:
        return ARTIFACT_REGISTRY[(stage_id, artifact_id)]
    except KeyError as exc:
        raise CandidateStoreError(
            f"unknown artifact registry entry: {stage_id}/{artifact_id}"
        ) from exc


def candidate_attempt_dir(
    pipeline_dir: Path, *, stage_id: str, artifact_id: str, attempt: int
) -> Path:
    if attempt < 1:
        raise CandidateStoreError("candidate attempt must be >= 1")
    artifact_entry(stage_id, artifact_id)
    return pipeline_dir / "candidates" / stage_id / artifact_id / str(attempt)


def candidate_artifact_path(
    pipeline_dir: Path, *, stage_id: str, artifact_id: str, attempt: int
) -> Path:
    entry = artifact_entry(stage_id, artifact_id)
    return candidate_attempt_dir(
        pipeline_dir,
        stage_id=stage_id,
        artifact_id=artifact_id,
        attempt=attempt,
    ) / entry.current_filename


def promoted_artifact_path(
    pipeline_dir: Path, *, stage_id: str, artifact_id: str, digest: str
) -> Path:
    entry = artifact_entry(stage_id, artifact_id)
    return (
        pipeline_dir
        / "artifacts"
        / stage_id
        / artifact_id
        / digest
        / entry.current_filename
    )


def current_projection_path(pipeline_dir: Path, *, stage_id: str, artifact_id: str) -> Path:
    entry = artifact_entry(stage_id, artifact_id)
    return pipeline_dir / "current" / stage_id / entry.current_filename


def _atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(
        dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(tmp_path, path)
    except Exception:
        try:
            os.unlink(tmp_path)
        except FileNotFoundError:
            pass
        raise


def _atomic_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(
        dir=str(destination.parent), prefix=f".{destination.name}.", suffix=".tmp"
    )
    os.close(fd)
    try:
        shutil.copy2(source, tmp_path)
        os.replace(tmp_path, destination)
    except Exception:
        try:
            os.unlink(tmp_path)
        except FileNotFoundError:
            pass
        raise


def write_json_candidate(
    pipeline_dir: Path,
    *,
    run_id: str,
    stage_id: str,
    artifact_id: str,
    attempt: int,
    payload: dict,
    owner: str,
) -> Path:
    entry = artifact_entry(stage_id, artifact_id)
    candidate = candidate_artifact_path(
        pipeline_dir,
        stage_id=stage_id,
        artifact_id=artifact_id,
        attempt=attempt,
    )
    _atomic_write_json(candidate, payload)
    metadata = {
        "schema_version": "1.0",
        "run_id": run_id,
        "stage_id": stage_id,
        "artifact_id": artifact_id,
        "display_name": entry.display_name,
        "owner": owner,
        "attempt": attempt,
        "candidate": str(candidate),
        "sha256": sha256_file(candidate),
        "created_at": utc_now_iso(),
        "status": "recorded",
    }
    _atomic_write_json(candidate.parent / "candidate_metadata.json", metadata)
    append_event(
        pipeline_dir,
        event_type="artifact_candidate_recorded",
        run_id=run_id,
        stage_id=stage_id,
        status="recorded",
        summary=f"{entry.display_name} candidate recorded",
        artifacts=[str(candidate)],
        details={
            "artifact_id": artifact_id,
            "owner": owner,
            "attempt": attempt,
            "sha256": metadata["sha256"],
        },
    )
    return candidate


def _default_json_validator(path: Path) -> list[str]:
    try:
        json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return [f"{path.name}: invalid JSON: {exc}"]
    return []


def validate_candidate(
    pipeline_dir: Path,
    *,
    stage_id: str,
    artifact_id: str,
    attempt: int,
    validator: Validator | None = None,
) -> list[str]:
    candidate = candidate_artifact_path(
        pipeline_dir,
        stage_id=stage_id,
        artifact_id=artifact_id,
        attempt=attempt,
    )
    if not candidate.is_file():
        return [f"candidate missing: {candidate}"]
    check = validator or _default_json_validator
    return check(candidate)


def promote_validated_candidate(
    pipeline_dir: Path,
    *,
    run_id: str,
    stage_id: str,
    artifact_id: str,
    attempt: int,
    owner: str,
    validator: Validator | None = None,
) -> dict:
    entry = artifact_entry(stage_id, artifact_id)
    candidate = candidate_artifact_path(
        pipeline_dir,
        stage_id=stage_id,
        artifact_id=artifact_id,
        attempt=attempt,
    )
    diagnostics = validate_candidate(
        pipeline_dir,
        stage_id=stage_id,
        artifact_id=artifact_id,
        attempt=attempt,
        validator=validator,
    )
    validation_path = candidate.parent / "candidate_validation.json"
    validation_status = "failed" if diagnostics else "passed"
    _atomic_write_json(
        validation_path,
        {
            "schema_version": "1.0",
            "stage_id": stage_id,
            "artifact_id": artifact_id,
            "attempt": attempt,
            "validator": entry.validator_label or "candidate_validator",
            "status": validation_status,
            "diagnostics": diagnostics,
            "validated_at": utc_now_iso(),
        },
    )
    if diagnostics:
        append_event(
            pipeline_dir,
            event_type="artifact_promotion_failed",
            run_id=run_id,
            stage_id=stage_id,
            status="failed",
            summary=f"{entry.display_name} candidate validation failed",
            artifacts=[str(candidate), str(validation_path)],
            details={
                "artifact_id": artifact_id,
                "owner": owner,
                "attempt": attempt,
                "diagnostics": diagnostics,
            },
        )
        raise CandidateStoreError("; ".join(diagnostics))

    digest = sha256_file(candidate)
    promoted = promoted_artifact_path(
        pipeline_dir,
        stage_id=stage_id,
        artifact_id=artifact_id,
        digest=digest,
    )
    current = current_projection_path(
        pipeline_dir,
        stage_id=stage_id,
        artifact_id=artifact_id,
    )
    _atomic_copy(candidate, promoted)
    _atomic_copy(candidate, current)
    result = {
        "schema_version": "1.0",
        "run_id": run_id,
        "stage_id": stage_id,
        "artifact_id": artifact_id,
        "display_name": entry.display_name,
        "owner": owner,
        "attempt": attempt,
        "candidate": str(candidate),
        "promoted": str(promoted),
        "current": str(current),
        "validation": str(validation_path),
        "sha256": digest,
        "promoted_at": utc_now_iso(),
    }
    _atomic_write_json(candidate.parent / "promotion_record.json", result)
    append_event(
        pipeline_dir,
        event_type="artifact_promoted",
        run_id=run_id,
        stage_id=stage_id,
        status="promoted",
        summary=f"{entry.display_name} candidate promoted",
        artifacts=[str(candidate), str(promoted), str(current)],
        details={
            "artifact_id": artifact_id,
            "owner": owner,
            "attempt": attempt,
            "sha256": digest,
            "current": str(current),
            "promoted": str(promoted),
        },
    )
    return result
