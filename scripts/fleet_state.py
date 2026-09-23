"""Fleet and timeline projections for the r2c-watch viewer.

Read-only projections over run artifacts for the live-view page
(scoped with maintainer decision 2026-07-06): the fleet view answers "what runs do I
have and what are their labels", the per-run timeline answers "what is
my run doing right now". Two hard properties from that scoping:

- Delivery labels pass through VERBATIM from final_manifest.json. This
  module must never paraphrase or re-derive a trust surface; the fleet
  is an index into reports, not a rival to them.
- Everything here is read-only. No function in this module may write.

Imported by scripts/r2c_watch.py only. The pipeline driver never
executes this file, so it is safe to change while a run is live.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import run_layout
from run_events import run_lock_owner_is_provably_gone

# Curated event classes for the timeline. `summary` on each event is
# already written in plain language by the driver, so the timeline
# passes it through untouched and only adds a category for filtering.
# Unknown event types render as "other" rather than disappearing — a
# new event type must never silently vanish from the behavioral view.
EVENT_CATEGORIES = {
    "run_started": "milestone",
    "run_finished": "milestone",
    "stage_started": "milestone",
    "stage_completed": "milestone",
    "stage_skipped": "milestone",
    "smoke_passed": "milestone",
    "delivery_label_derived": "milestone",
    "stage_halted": "problem",
    "stage_degraded": "problem",
    "agent_dispatch_failed": "problem",
    "truncated_turn_detected": "problem",
    "cap_burn_turn_detected": "problem",
    # Accepted-but-unfixed static validation failures ride the delivery;
    # the researcher should see them as a problem marker in the timeline.
    "notebook_validation_baseline_recorded": "problem",
    "validation_failed": "problem",
    "pipeline_validation_issue": "problem",
    "smoke_failed": "problem",
    # Failure-named events that rendered as "other" until the maintainer's 07-07
    # mock run surfaced the gap live: the trainability gate's silent
    # non-learning finding — the most researcher-relevant event of that
    # smoke stage — sat uncategorized in the timeline. Mapping the
    # remaining registered types (decisions/recoveries) is queued; only
    # the zero-judgment failure class lands here.
    "smoke_trainability_failed": "problem",
    # Demo-success verdict (2026-07-16): a failed headline demo and a
    # committed-family marker gap are researcher-relevant problem marks; a
    # succeeded/undetermined record and a passing/inapplicable data-signal
    # check are routine bookkeeping. The data-signal fail is the
    # impossible-gate burn — a problem the timeline must show.
    "demo_verdict_failed": "problem",
    "demo_kit_coverage_gap": "problem",
    "demo_verdict_recorded": "routine",
    "fix_data_signal_failed": "problem",
    "fix_data_signal_checked": "routine",
    "contract_halted": "problem",
    "artifact_promotion_failed": "problem",
    "judge_decision_recorded": "decision",
    "smoke_diagnosis_driver_fallback": "decision",
    "run_dir_archived": "decision",
    "reviewer_disclosure_routed": "decision",
    # Explanation math-sanity pass (item 3): a mechanism claim refuted by a
    # numeric counterexample is a demoting finding (rejects the entry); a claim
    # dropped by the byte-substring anti-hallucination floor is a harness
    # decision. not_checkable rides in entry metadata, not a per-claim event.
    "math_claim_refuted": "problem",
    "math_claim_dropped": "decision",
    # Fix-loop third option (item 7): the conversion itself is a recorded
    # driver decision; declined gates are a decision too (behavior stays the
    # degrade); a rolled-back conversion is a problem worth a timeline mark.
    "fix_loop_stub_conversion": "decision",
    "fix_loop_stub_gates_unmet": "decision",
    "fix_loop_stub_regate_failed": "problem",
    # Dispatch-recovery ladder (item 8): recoveries are driver decisions
    # (the viewer vocabulary is milestone/problem/decision/routine); a
    # refused re-dispatch over a live session is a problem mark (the T1b
    # zombie hazard fired). agent_dispatch_recovered also serves rung 1.
    "dispatch_retry_after_transport": "decision",
    "dispatch_retry_skipped_live_session": "problem",
    "agent_dispatch_recovered": "decision",
    # Cap-burn corrective resume (fix 2): the attempt and a landed write
    # are driver decisions; an exhausted resume means the burn stands and
    # stage recovery pays for it — a problem mark. The ceiling firing is
    # the runaway-turn problem itself (one SRL dispatch burned ~4 wall
    # hours against a declared 1800s timeout).
    "cap_burn_corrective_resume_attempted": "decision",
    "cap_burn_corrective_resume_succeeded": "decision",
    "cap_burn_corrective_resume_exhausted": "problem",
    "dispatch_extension_ceiling_hit": "problem",
    "agent_dispatch_started": "routine",
    "agent_dispatch_completed": "routine",
    "validation_passed": "routine",
}

# Pure lock bookkeeping; carries no behavioral information a researcher
# would act on.
SKIPPED_EVENT_TYPES = {"run_lock_acquired", "run_lock_released"}

# The only files the viewer will ever serve from a run dir. URL-facing
# flat names mapped to their current-layout location (run_layout owns the
# layout; legacy pre-consolidation runs resolve through its aliases) —
# never anything under .pipeline/.
ARTIFACT_WHITELIST = {
    "REPORT.md": run_layout.REPORT_MD,
    "METHOD.md": run_layout.METHOD_MD,
    "KNOWN_ISSUES.md": run_layout.KNOWN_ISSUES_MD,
    "assumptions.md": run_layout.ASSUMPTIONS_MD,
    "README.md": run_layout.PACKAGE_README,
    "TESTS.md": "method/tests/README.md",
}


def _iso_from_mtime(mtime: float) -> str:
    return (
        datetime.fromtimestamp(mtime, tz=timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def _load_json(path: Path) -> dict | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def is_run_dir(path: Path) -> bool:
    """A run dir carries pipeline state or a delivery manifest. This
    naturally excludes r2c_runs/_batch_logs, manifest .txt files, and
    the archive/ folder-of-folders."""
    return path.is_dir() and (
        (path / ".pipeline").is_dir()
        or run_layout.resolve_existing(path, run_layout.FINAL_MANIFEST_JSON) is not None
    )


def _run_activity_mtime(run_dir: Path) -> float:
    candidates = [
        run_dir / ".pipeline" / "_lock" / "owner.json",
        run_dir / ".pipeline" / "progress.json",
        run_dir / ".pipeline" / "run_events.jsonl",
        run_dir / ".pipeline" / "driver_state.json",
        run_dir,
    ]
    mtimes = [path.stat().st_mtime for path in candidates if path.exists()]
    return max(mtimes) if mtimes else 0.0


def fleet_row(run_dir: Path) -> dict:
    """One fleet-table row. Label verbatim from the manifest; liveness
    from progress.json. progress.json can lag run_events.jsonl when a
    driver dies mid-write — the drill-in run view holds the
    authoritative events projection, the fleet row's last-activity age
    is the cue to open it."""
    progress = _load_json(run_dir / ".pipeline" / "progress.json") or {}
    manifest_path = run_layout.resolve_existing(run_dir, run_layout.FINAL_MANIFEST_JSON)
    manifest = (_load_json(manifest_path) if manifest_path else None) or {}
    delivery = manifest.get("delivery") or {}

    run_status = progress.get("run_status") or manifest.get("run_status")
    # A "running" record whose owning driver is conclusively dead is a
    # ghost: a killed driver never writes terminal state, so its progress
    # advertises running forever (the pdfgnn _10 row sat "running" for 18
    # days). Reuse R2C-076's single staleness rule — the run-lock owner
    # record (pid + host, verified on THIS host only) — rather than invent
    # a second owner surface. Anything short of provably-gone (no owner
    # record, another host, pid alive) keeps the honest "running".
    if run_status == "running" and run_lock_owner_is_provably_gone(
        run_dir / ".pipeline"
    ):
        run_status = "interrupted"
    current = progress.get("current") or {}
    activity = _run_activity_mtime(run_dir)
    return {
        "slug": run_dir.name,
        "run_dir": str(run_dir),
        "label": delivery.get("label"),
        "label_partial": bool(delivery.get("partial")),
        "run_status": run_status,
        "current_stage_label": current.get("label"),
        "updated_at": progress.get("updated_at"),
        "last_activity_at": _iso_from_mtime(activity) if activity else None,
        "last_activity_mtime": activity,
        "live": run_status == "running",
        "report_available": (run_dir / "REPORT.md").is_file(),
    }


def fleet_rows(runs_root: Path) -> list[dict]:
    """All fleet rows, running runs first, then most recent activity."""
    if not runs_root.is_dir():
        return []
    rows = [
        fleet_row(path)
        for path in sorted(runs_root.iterdir())
        if is_run_dir(path)
    ]
    rows.sort(key=lambda r: (not r["live"], -r["last_activity_mtime"]))
    for row in rows:
        row.pop("last_activity_mtime", None)
    return rows


def timeline(run_dir: Path, limit: int = 250) -> dict:
    """Curated behavioral timeline from run_events.jsonl.

    Tolerant of a torn trailing line (the driver appends while we
    read); malformed lines are counted, never fatal. Returns newest
    last. `total` vs `shown` lets the page say "showing the last N of
    M" instead of silently truncating.
    """
    events_path = run_dir / ".pipeline" / "run_events.jsonl"
    entries: list[dict] = []
    malformed = 0
    try:
        raw_lines = events_path.read_text(encoding="utf-8").splitlines()
    except OSError:
        raw_lines = []
    for line in raw_lines:
        line = line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except ValueError:
            malformed += 1
            continue
        if not isinstance(event, dict):
            malformed += 1
            continue
        event_type = str(event.get("event_type") or "")
        if event_type in SKIPPED_EVENT_TYPES:
            continue
        entries.append({
            "sequence": event.get("sequence"),
            "timestamp": event.get("timestamp"),
            "event_type": event_type,
            "category": EVENT_CATEGORIES.get(event_type, "other"),
            "summary": event.get("summary"),
            "stage_id": event.get("stage_id"),
            "stage_label": event.get("stage_label"),
        })
    total = len(entries)
    shown = entries[-limit:] if limit and total > limit else entries
    return {
        "events": shown,
        "total": total,
        "shown": len(shown),
        "skipped_malformed": malformed,
        "events_path": str(events_path),
    }


def resolve_artifact(runs_root: Path, slug: str, name: str) -> Path | None:
    """Whitelisted researcher-facing file under a real run dir, or None.

    The slug must name a directory DIRECTLY under runs_root (checked by
    listing, so path separators and traversal cannot resolve) and the
    filename must be on the fixed whitelist. The whitelist maps the
    URL-facing flat name to its layout location; legacy pre-consolidation
    runs resolve through run_layout's aliases."""
    rel = ARTIFACT_WHITELIST.get(name)
    if rel is None:
        return None
    if not runs_root.is_dir():
        return None
    if slug not in {p.name for p in runs_root.iterdir() if is_run_dir(p)}:
        return None
    candidate = run_layout.resolve_existing(runs_root / slug, rel)
    return candidate if candidate is not None and candidate.is_file() else None
