from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


PROGRESS_FILE_NAME = "progress.json"

# Bumped only when progress.json's shape changes in a way older readers cannot
# understand. It doubles as the run-dir layout-compatibility signal (item 29):
# an existing run dir whose progress.json is absent or carries a different
# version was produced by an older R2C and is not safe to resume against.
PROGRESS_SCHEMA_VERSION = "1.0"


STAGE_TERMINAL_EVENTS = {
    "stage_completed": "completed",
    "stage_skipped": "skipped",
    "stage_halted": "halted",
    "stage_degraded": "degraded",
}


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _parse_timestamp(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def _duration_s(started_at: str | None, ended_at: str | None) -> float | None:
    started = _parse_timestamp(started_at)
    ended = _parse_timestamp(ended_at)
    if started is None or ended is None:
        return None
    return max(0.0, round((ended - started).total_seconds(), 1))


def _elapsed_s(started_at: str | None, updated_at: str | None) -> float | None:
    started = _parse_timestamp(started_at)
    updated = _parse_timestamp(updated_at)
    if started is None or updated is None:
        return None
    return max(0.0, round((updated - started).total_seconds(), 1))


def _load_events(pipeline_dir: Path) -> list[dict[str, Any]]:
    path = pipeline_dir / "run_events.jsonl"
    if not path.is_file():
        return []
    events: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        events.append(json.loads(line))
    return events


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        dir=str(path.parent),
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
            handle.write("\n")
        os.replace(tmp_name, path)
    except Exception:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass
        raise


def build_progress(
    *,
    run_id: str,
    events: Iterable[dict[str, Any]],
    stage_catalog: Iterable[tuple[str, str]],
) -> dict[str, Any]:
    event_rows = list(events)
    catalog = list(stage_catalog)
    now = utc_now_iso()
    first_timestamp = event_rows[0].get("timestamp") if event_rows else now
    updated_at = event_rows[-1].get("timestamp") if event_rows else now

    stages: dict[str, dict[str, Any]] = {
        stage_id: {
            "stage_id": stage_id,
            "label": label,
            "status": "pending",
            "started_at": None,
            "ended_at": None,
            "duration_s": None,
            "notes": "",
        }
        for stage_id, label in catalog
    }
    current_stage_id: str | None = None
    active_dispatch: dict[str, Any] | None = None
    run_status = "waiting"

    for event in event_rows:
        event_type = str(event.get("event_type", ""))
        timestamp = event.get("timestamp")
        stage_id = event.get("stage_id")
        summary = event.get("summary") or ""

        if event_type == "run_started":
            run_status = "running"
        elif event_type == "run_finished":
            run_status = str(event.get("status") or "completed")
            current_stage_id = None
            active_dispatch = None

        if event_type == "stage_started" and isinstance(stage_id, str):
            row = stages.setdefault(
                stage_id,
                {
                    "stage_id": stage_id,
                    "label": event.get("stage_label") or stage_id,
                    "status": "pending",
                    "started_at": None,
                    "ended_at": None,
                    "duration_s": None,
                    "notes": "",
                },
            )
            row["status"] = "running"
            row["started_at"] = timestamp
            row["ended_at"] = None
            row["duration_s"] = None
            row["notes"] = summary
            current_stage_id = stage_id
            active_dispatch = None
        elif event_type in STAGE_TERMINAL_EVENTS and isinstance(stage_id, str):
            row = stages.setdefault(
                stage_id,
                {
                    "stage_id": stage_id,
                    "label": event.get("stage_label") or stage_id,
                    "status": "pending",
                    "started_at": None,
                    "ended_at": None,
                    "duration_s": None,
                    "notes": "",
                },
            )
            row["status"] = STAGE_TERMINAL_EVENTS[event_type]
            row["ended_at"] = timestamp
            row["duration_s"] = _duration_s(row.get("started_at"), row.get("ended_at"))
            row["notes"] = summary
            if current_stage_id == stage_id:
                current_stage_id = None
                active_dispatch = None

        if event_type == "agent_dispatch_started":
            details = event.get("details") if isinstance(event.get("details"), dict) else {}
            active_dispatch = {
                "agent": details.get("agent"),
                "dispatch_id": details.get("dispatch_id"),
                "started_at": timestamp,
                "timeout_s": details.get("timeout_s"),
                "prompt_chars": details.get("prompt_chars"),
            }
        elif event_type in {"agent_dispatch_completed", "agent_dispatch_failed", "agent_dispatch_recovered"}:
            active_dispatch = None

    stage_rows = [stages[stage_id] for stage_id, _ in catalog if stage_id in stages]
    extra_rows = [row for stage_id, row in stages.items() if stage_id not in {sid for sid, _ in catalog}]
    stage_rows.extend(extra_rows)

    current: dict[str, Any] | None = None
    if run_status == "running" and current_stage_id is not None:
        stage = stages[current_stage_id]
        current = {
            "stage_id": current_stage_id,
            "label": stage.get("label") or current_stage_id,
            "started_at": stage.get("started_at"),
            "agent": active_dispatch.get("agent") if active_dispatch else None,
            "agent_started_at": active_dispatch.get("started_at") if active_dispatch else None,
            "agent_timeout_s": active_dispatch.get("timeout_s") if active_dispatch else None,
            "dispatch_id": active_dispatch.get("dispatch_id") if active_dispatch else None,
        }

    done_statuses = {"completed", "skipped", "degraded", "halted"}
    dispatch_started = [
        event for event in event_rows
        if event.get("event_type") == "agent_dispatch_started"
    ]
    dispatch_completed = [
        event for event in event_rows
        if event.get("event_type") == "agent_dispatch_completed"
    ]
    dispatch_failed = [
        event for event in event_rows
        if event.get("event_type") == "agent_dispatch_failed"
    ]
    dispatch_recovered = [
        event for event in event_rows
        if event.get("event_type") == "agent_dispatch_recovered"
    ]
    dispatch_succeeded = [
        event for event in dispatch_completed
        if event.get("status") != "failed"
        and (
            not isinstance(event.get("details"), dict)
            or event["details"].get("completed") is not False
        )
    ]
    dispatch_completed_failed = [
        event for event in dispatch_completed
        if event.get("status") == "failed"
        or (
            isinstance(event.get("details"), dict)
            and event["details"].get("completed") is False
        )
    ]
    dispatch_finished_count = (
        len(dispatch_completed) + len(dispatch_failed) + len(dispatch_recovered)
    )
    totals = {
        "stages_done": sum(1 for row in stage_rows if row.get("status") in done_statuses),
        "stages_total": len(stage_rows),
        "elapsed_s": _elapsed_s(first_timestamp, updated_at),
        "degrades": sum(1 for row in stage_rows if row.get("status") == "degraded"),
        "halts": sum(1 for row in stage_rows if row.get("status") == "halted"),
        "dispatches_started": len(dispatch_started),
        "dispatches_finished": dispatch_finished_count,
        "dispatches_succeeded": len(dispatch_succeeded),
        "dispatches_failed": len(dispatch_failed) + len(dispatch_completed_failed),
        "dispatches_recovered": len(dispatch_recovered),
        # Backward-compatible alias for early watcher builds. Prefer
        # dispatches_succeeded in new display code.
        "dispatches_completed": len(dispatch_succeeded),
    }

    return {
        "schema_version": PROGRESS_SCHEMA_VERSION,
        "slug": run_id,
        "run_status": run_status,
        "started_at": first_timestamp,
        "updated_at": updated_at,
        "current": current,
        "stages": stage_rows,
        "totals": totals,
    }


def write_progress(
    pipeline_dir: Path,
    *,
    run_id: str,
    stage_catalog: Iterable[tuple[str, str]],
) -> Path:
    events = _load_events(pipeline_dir)
    payload = build_progress(run_id=run_id, events=events, stage_catalog=stage_catalog)
    path = pipeline_dir / PROGRESS_FILE_NAME
    _atomic_write_json(path, payload)
    return path


def is_run_dir_layout_compatible(run_dir: Path) -> bool:
    """True iff an existing run dir is a resume the current driver understands.

    The signal is the live progress-state file:
    ``<run_dir>/.pipeline/progress.json`` present, parseable, and stamped with
    the current ``PROGRESS_SCHEMA_VERSION``. ``_append_run_event`` writes
    progress.json on the very first event (lock acquisition at stage 0), so
    every run this driver ever started — running, halted, or delivered —
    carries a current one. A dir missing it, or carrying a stale
    schema_version, or holding corrupt JSON, was produced by an older version
    of R2C; its sentinels and artifacts are not safe to resume against, so the
    caller archives it aside and starts fresh (item 29)."""
    progress_path = run_dir / ".pipeline" / PROGRESS_FILE_NAME
    if not progress_path.is_file():
        return False
    try:
        payload = json.loads(progress_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError, ValueError):
        return False
    return (
        isinstance(payload, dict)
        and payload.get("schema_version") == PROGRESS_SCHEMA_VERSION
    )
