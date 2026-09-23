#!/usr/bin/env python3
"""Stage-agnostic reconciliation for stranded candidate dispatches.

This is the Step 5 foundation. It scans candidate-attempt directories and
repairs the safe cases:

- complete candidate -> validate through the candidate store and promote
- incomplete or invalid candidate -> record an auditable failure

The live pipeline still uses sentinels and canonical artifacts as runtime
authority. Reconciliation only acts on candidate attempts that already exist.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

from artifact_candidates import (
    ARTIFACT_REGISTRY,
    ArtifactRegistryEntry,
    CandidateStoreError,
    Validator,
    candidate_artifact_path,
    promote_validated_candidate,
)
from run_events import append_event, load_events, utc_now_iso


class DispatchReconciliationError(RuntimeError):
    """Raised when dispatch reconciliation cannot inspect candidate state."""


@dataclass(frozen=True)
class ReconciliationResult:
    stage_id: str
    artifact_id: str
    attempt: int
    status: str
    reason: str
    candidate: Path | None = None
    current: Path | None = None
    promoted: Path | None = None


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


def _attempt_numbers(base: Path) -> list[int]:
    if not base.is_dir():
        return []
    attempts: list[int] = []
    for child in base.iterdir():
        if child.is_dir() and child.name.isdigit() and int(child.name) >= 1:
            attempts.append(int(child.name))
    return sorted(attempts)


def _attempt_terminal(attempt_dir: Path) -> bool:
    return (
        (attempt_dir / "promotion_record.json").is_file()
        or (attempt_dir / "reconciliation_record.json").is_file()
    )


def active_dispatches(pipeline_dir: Path) -> dict[str, dict]:
    """Return started dispatches without a terminal event.

    Main now records ``dispatch_id`` in agent dispatch event details. Older
    events without a dispatch id are ignored because they cannot be paired
    safely.
    """

    active: dict[str, dict] = {}
    for event in load_events(pipeline_dir):
        details = event.get("details") if isinstance(event.get("details"), dict) else {}
        dispatch_id = details.get("dispatch_id")
        if not isinstance(dispatch_id, str) or not dispatch_id:
            continue
        event_type = event.get("event_type")
        if event_type == "agent_dispatch_started":
            active[dispatch_id] = event
        elif event_type in {
            "agent_dispatch_completed",
            "agent_dispatch_failed",
            "dispatch_reconciled",
        }:
            active.pop(dispatch_id, None)
    return active


def _record_reconciliation(
    attempt_dir: Path,
    *,
    run_id: str,
    entry: ArtifactRegistryEntry,
    attempt: int,
    status: str,
    reason: str,
    candidate: Path | None,
    promoted: Path | None = None,
    current: Path | None = None,
) -> Path:
    record = {
        "schema_version": "1.0",
        "run_id": run_id,
        "stage_id": entry.stage_id,
        "artifact_id": entry.artifact_id,
        "attempt": attempt,
        "status": status,
        "reason": reason,
        "candidate": str(candidate) if candidate is not None else None,
        "promoted": str(promoted) if promoted is not None else None,
        "current": str(current) if current is not None else None,
        "reconciled_at": utc_now_iso(),
    }
    path = attempt_dir / "reconciliation_record.json"
    _atomic_write_json(path, record)
    return path


def _reconcile_one_attempt(
    pipeline_dir: Path,
    *,
    run_id: str,
    entry: ArtifactRegistryEntry,
    attempt: int,
    owner: str,
    validator: Validator | None,
    active_dispatch_ids: list[str],
) -> ReconciliationResult | None:
    attempt_dir = (
        pipeline_dir
        / "candidates"
        / entry.stage_id
        / entry.artifact_id
        / str(attempt)
    )
    if _attempt_terminal(attempt_dir):
        return None

    candidate = candidate_artifact_path(
        pipeline_dir,
        stage_id=entry.stage_id,
        artifact_id=entry.artifact_id,
        attempt=attempt,
    )
    if not candidate.is_file():
        reason = f"candidate missing: {candidate}"
        record = _record_reconciliation(
            attempt_dir,
            run_id=run_id,
            entry=entry,
            attempt=attempt,
            status="failed",
            reason=reason,
            candidate=candidate,
        )
        append_event(
            pipeline_dir,
            event_type="agent_dispatch_failed",
            run_id=run_id,
            stage_id=entry.stage_id,
            status="failed",
            summary=f"{entry.display_name} candidate reconciliation failed",
            artifacts=[str(attempt_dir), str(record)],
            details={
                "artifact_id": entry.artifact_id,
                "attempt": attempt,
                "reason": reason,
                "reconciled": True,
                "active_dispatch_ids": active_dispatch_ids,
            },
        )
        return ReconciliationResult(
            stage_id=entry.stage_id,
            artifact_id=entry.artifact_id,
            attempt=attempt,
            status="failed",
            reason=reason,
            candidate=candidate,
        )

    try:
        promotion = promote_validated_candidate(
            pipeline_dir,
            run_id=run_id,
            stage_id=entry.stage_id,
            artifact_id=entry.artifact_id,
            attempt=attempt,
            owner=owner,
            validator=validator,
        )
    except CandidateStoreError as exc:
        reason = str(exc)
        record = _record_reconciliation(
            attempt_dir,
            run_id=run_id,
            entry=entry,
            attempt=attempt,
            status="failed",
            reason=reason,
            candidate=candidate,
        )
        append_event(
            pipeline_dir,
            event_type="agent_dispatch_failed",
            run_id=run_id,
            stage_id=entry.stage_id,
            status="failed",
            summary=f"{entry.display_name} candidate reconciliation failed",
            artifacts=[str(candidate), str(record)],
            details={
                "artifact_id": entry.artifact_id,
                "attempt": attempt,
                "reason": reason,
                "reconciled": True,
                "active_dispatch_ids": active_dispatch_ids,
            },
        )
        return ReconciliationResult(
            stage_id=entry.stage_id,
            artifact_id=entry.artifact_id,
            attempt=attempt,
            status="failed",
            reason=reason,
            candidate=candidate,
        )

    promoted = Path(str(promotion["promoted"]))
    current = Path(str(promotion["current"]))
    record = _record_reconciliation(
        attempt_dir,
        run_id=run_id,
        entry=entry,
        attempt=attempt,
        status="reconciled",
        reason="candidate validated and promoted",
        candidate=candidate,
        promoted=promoted,
        current=current,
    )
    append_event(
        pipeline_dir,
        event_type="dispatch_reconciled",
        run_id=run_id,
        stage_id=entry.stage_id,
        status="reconciled",
        summary=f"{entry.display_name} candidate reconciled",
        artifacts=[str(candidate), str(promoted), str(current), str(record)],
        details={
            "artifact_id": entry.artifact_id,
            "attempt": attempt,
            "sha256": promotion["sha256"],
            "promoted": str(promoted),
            "current": str(current),
            "active_dispatch_ids": active_dispatch_ids,
        },
    )
    return ReconciliationResult(
        stage_id=entry.stage_id,
        artifact_id=entry.artifact_id,
        attempt=attempt,
        status="reconciled",
        reason="candidate validated and promoted",
        candidate=candidate,
        promoted=promoted,
        current=current,
    )


def reconcile_candidate_dispatches(
    pipeline_dir: Path,
    *,
    run_id: str,
    validators: dict[tuple[str, str], Validator] | None = None,
    registry: dict[tuple[str, str], ArtifactRegistryEntry] | None = None,
    owner: str = "dispatch-reconciler",
) -> list[ReconciliationResult]:
    """Reconcile every non-terminal candidate attempt in the registry.

    The implementation is stage-agnostic: the registry supplies the stage and
    artifact ids, and callers may pass validators per ``(stage_id, artifact_id)``.
    """

    registry = registry or ARTIFACT_REGISTRY
    validators = validators or {}
    active_dispatch_ids = sorted(active_dispatches(pipeline_dir))
    results: list[ReconciliationResult] = []
    for key, entry in sorted(registry.items()):
        stage_id, artifact_id = key
        if stage_id != entry.stage_id or artifact_id != entry.artifact_id:
            raise DispatchReconciliationError(
                f"registry key {key!r} does not match entry "
                f"{entry.stage_id}/{entry.artifact_id}"
            )
        base = pipeline_dir / "candidates" / stage_id / artifact_id
        for attempt in _attempt_numbers(base):
            result = _reconcile_one_attempt(
                pipeline_dir,
                run_id=run_id,
                entry=entry,
                attempt=attempt,
                owner=owner,
                validator=validators.get(key),
                active_dispatch_ids=active_dispatch_ids,
            )
            if result is not None:
                results.append(result)
    return results
