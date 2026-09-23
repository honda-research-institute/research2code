#!/usr/bin/env python3
"""One-off backfill for the run-history ledger (decision 2026-07-14: ship
the script with the block, run it once manually).

Sweeps surviving run dirs for `.pipeline/run_events.jsonl` and mints ledger
rows for what still exists. Backfilled rows are lineage-only (no invocation
split — resumes within a surviving run dir collapse into one row, flagged
`backfilled: true`). Already-wiped history is gone; this recovers only what
the wipes spared.

Idempotent: the invocation id is derived deterministically from the slug and
the first event's timestamp, and rows already present in ledger.jsonl are
skipped on re-run.

Usage:
    python3 scripts/backfill_run_history.py [--roots r2c_runs r2c_runs/archive]
                                            [--history-dir r2c_runs/_history]
                                            [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

import run_history  # noqa: E402

REPO_ROOT = SCRIPT_DIR.parent


def _first_and_last_events(events_path: Path) -> tuple[dict | None, dict | None, int, dict]:
    first, last, count, by_type = None, None, 0, {}
    try:
        with events_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(event, dict):
                    continue
                count += 1
                if first is None:
                    first = event
                last = event
                event_type = event.get("event_type")
                if isinstance(event_type, str):
                    by_type[event_type] = by_type.get(event_type, 0) + 1
    except OSError:
        pass
    return first, last, count, by_type


def _backfill_invocation_id(slug: str, first_event: dict | None) -> str:
    stamp = "unknown"
    if first_event and isinstance(first_event.get("timestamp"), str):
        stamp = first_event["timestamp"].replace("-", "").replace(":", "")
    return f"{slug}-{stamp}-bkfill"


def _run_status_from_events(last_events_by_type: dict, events_path: Path, run_dir: Path) -> str | None:
    manifest = None
    for rel in ("details/final_manifest.json", "final_manifest.json"):
        candidate = run_dir / rel
        if candidate.is_file():
            try:
                manifest = json.loads(candidate.read_text(encoding="utf-8"))
            except Exception:
                manifest = None
            break
    if isinstance(manifest, dict) and isinstance(manifest.get("run_status"), str):
        return manifest["run_status"]
    # Fall back to the last run_finished event's status.
    try:
        status = None
        with events_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(event, dict) and event.get("event_type") == "run_finished":
                    status = event.get("status")
        return status if isinstance(status, str) else None
    except OSError:
        return None


def _existing_invocation_ids(history_dir: Path) -> set[str]:
    ledger = history_dir / run_history.LEDGER_NAME
    ids: set[str] = set()
    if not ledger.is_file():
        return ids
    with ledger.open("r", encoding="utf-8") as handle:
        for line in handle:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict) and isinstance(row.get("invocation_id"), str):
                ids.add(row["invocation_id"])
    return ids


def backfill_run_dir(run_dir: Path, history_dir: Path, existing: set[str], *, dry_run: bool) -> str:
    slug = run_dir.name
    events_path = run_dir / ".pipeline" / "run_events.jsonl"
    first, last, count, by_type = _first_and_last_events(events_path)
    invocation_id = _backfill_invocation_id(slug, first)
    if invocation_id in existing:
        return f"SKIP {slug}: already backfilled ({invocation_id})"
    run_status = _run_status_from_events(by_type, events_path, run_dir) or "unknown"
    if dry_run:
        return f"DRY  {slug}: would mint {invocation_id} (run_status={run_status}, {count} events)"

    invocation_dir = history_dir / invocation_id
    invocation_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(events_path, invocation_dir / run_history.EVENTS_MIRROR_NAME)
    snapshot, truncated = run_history.write_terminal_snapshot(run_dir, invocation_dir)
    progress = run_history._progress_fields(run_dir)
    delivery_label, label_reason_ids = run_history._delivery_fields(run_dir)
    halt_class, halt_stage = run_history._halt_fields(run_dir)
    paradigm, gap_path = run_history._paradigm_and_gap(run_dir, by_type)
    started_at = first.get("timestamp") if first else None
    finished_at = last.get("timestamp") if last else None
    wall_seconds = None
    try:
        from datetime import datetime

        begin = datetime.fromisoformat(str(started_at).replace("Z", "+00:00"))
        end = datetime.fromisoformat(str(finished_at).replace("Z", "+00:00"))
        wall_seconds = int((end - begin).total_seconds())
    except Exception:
        pass
    events_file = invocation_dir / run_history.EVENTS_MIRROR_NAME
    row = {
        "ledger_schema_version": run_history.LEDGER_SCHEMA_VERSION,
        "invocation_id": invocation_id,
        "lineage_id": invocation_id.rsplit("-", 1)[0],
        "invocation_index": None,
        "slug": slug,
        "paper_input": None,
        "paper_sha256": None,
        "input_kind": None,
        "tree_commit": None,
        "tree_dirty": None,
        "think_model": None,
        "code_model": None,
        "per_dispatch_sessions": None,
        "host": None,
        "driver_args": None,
        "batch_manifest": None,
        "run_tag": None,
        "started_at": started_at,
        "finished_at": finished_at,
        "wall_seconds": wall_seconds,
        "stage_durations_s": progress["stage_durations_s"],
        "run_status": run_status,
        "exit_code": None,
        "terminal_stage": progress["terminal_stage"],
        "stages_completed": progress["stages_completed"],
        "delivery_label": delivery_label,
        "label_reason_ids": label_reason_ids,
        "halt_class": halt_class,
        "halt_stage": halt_stage,
        "paradigm": paradigm,
        "gap_path": gap_path,
        "event_count": count,
        "event_type_counts": by_type,
        "dispatch_failed_count": by_type.get("agent_dispatch_failed", 0),
        "cap_burns": by_type.get("cap_burn_turn_detected", 0),
        "cap_burns_recovered": by_type.get("cap_burn_recovered", 0),
        "truncated_turns": by_type.get("truncated_turn_detected", 0),
        "judge_decisions": by_type.get("judge_decision_recorded", 0),
        "assumptions_recorded": run_history._assumptions_count(run_dir),
        "tokens": None,
        "token_dispatches_counted": None,
        "events_file": f"{invocation_id}/{run_history.EVENTS_MIRROR_NAME}",
        "events_bytes": events_file.stat().st_size if events_file.is_file() else 0,
        "terminal_snapshot": snapshot,
        "snapshot_truncated": truncated,
        "backfilled": True,
    }
    run_history.append_ledger_row(history_dir, row)
    existing.add(invocation_id)
    return (
        f"OK   {slug}: {invocation_id} (run_status={run_status}, {count} events, "
        f"{len(snapshot)} terminal files{', truncated: ' + ','.join(truncated) if truncated else ''})"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--roots", nargs="*",
        default=[str(REPO_ROOT / "r2c_runs"), str(REPO_ROOT / "r2c_runs" / "archive")],
        help="directories whose immediate children are candidate run dirs",
    )
    parser.add_argument(
        "--history-dir", default=None,
        help="ledger location (default: <first root>/_history, or R2C_HISTORY_DIR)",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    roots = [Path(r) for r in args.roots if Path(r).is_dir()]
    if not roots:
        print("no candidate roots exist", file=sys.stderr)
        return 1
    history_dir = (
        Path(args.history_dir)
        if args.history_dir
        else run_history.resolve_history_dir(roots[0] / "_placeholder_")
    )
    existing = _existing_invocation_ids(history_dir)
    seen: set[Path] = set()
    lines: list[str] = []
    for root in roots:
        for child in sorted(root.iterdir()):
            child = child.resolve()
            if child in seen or not child.is_dir():
                continue
            seen.add(child)
            if not (child / ".pipeline" / "run_events.jsonl").is_file():
                continue
            lines.append(backfill_run_dir(child, history_dir, existing, dry_run=args.dry_run))
    for line in lines:
        print(line)
    print(f"backfill complete: {len(lines)} run dir(s) considered, history at {history_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
