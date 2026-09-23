"""Fleet and timeline projections for the live-view page (scoped with
maintainer decision 2026-07-06). The two hard properties under test: labels pass
through VERBATIM from final_manifest.json, and the projections tolerate
a live writer (torn trailing event line) without lying about it."""

from __future__ import annotations

import json

import fleet_state


def _mk_run(runs_root, slug, *, manifest=None, progress=None, report=False,
            events_lines=None):
    run_dir = runs_root / slug
    (run_dir / ".pipeline").mkdir(parents=True)
    if manifest is not None:
        (run_dir / "final_manifest.json").write_text(
            json.dumps(manifest), encoding="utf-8")
    if progress is not None:
        (run_dir / ".pipeline" / "progress.json").write_text(
            json.dumps(progress), encoding="utf-8")
    if report:
        (run_dir / "REPORT.md").write_text("# Run Report\n", encoding="utf-8")
    if events_lines is not None:
        (run_dir / ".pipeline" / "run_events.jsonl").write_text(
            "\n".join(events_lines) + "\n", encoding="utf-8")
    return run_dir


# ---------------------------------------------------------------------------
# Fleet rows
# ---------------------------------------------------------------------------


def test_fleet_rows_label_passes_through_verbatim(tmp_path):
    runs = tmp_path / "r2c_runs"
    _mk_run(runs, "done", report=True, manifest={
        "run_status": "passed",
        "delivery": {"label": "uncertified_new_territory", "partial": True},
    })
    rows = fleet_state.fleet_rows(runs)
    assert len(rows) == 1
    row = rows[0]
    # Verbatim: never paraphrased, never re-derived.
    assert row["label"] == "uncertified_new_territory"
    assert row["label_partial"] is True
    assert row["run_status"] == "passed"
    assert row["live"] is False
    assert row["report_available"] is True


def test_fleet_rows_live_run_shows_stage_and_sorts_first(tmp_path):
    runs = tmp_path / "r2c_runs"
    _mk_run(runs, "a-finished", manifest={
        "run_status": "passed", "delivery": {"label": "verified"}})
    _mk_run(runs, "z-live", progress={
        "run_status": "running",
        "updated_at": "2026-07-06T20:00:00Z",
        "current": {"stage_id": "stage_2b",
                    "label": "Stage 2.b - Architecture & Training"},
    })
    rows = fleet_state.fleet_rows(runs)
    assert [r["slug"] for r in rows] == ["z-live", "a-finished"]
    live = rows[0]
    assert live["live"] is True
    assert live["label"] is None
    assert live["current_stage_label"] == "Stage 2.b - Architecture & Training"
    # Internal sort key never leaks into the payload.
    assert "last_activity_mtime" not in live


def test_fleet_rows_exclude_non_run_entries(tmp_path):
    runs = tmp_path / "r2c_runs"
    _mk_run(runs, "real", manifest={"run_status": "passed",
                                    "delivery": {"label": "draft"}})
    (runs / "_batch_logs").mkdir()
    (runs / "_batch_logs" / "batch_summary.txt").write_text("x")
    (runs / "some-manifest.txt").write_text("input_papers/x.md\n")
    # archive/ is a folder of run folders with no pipeline state itself.
    (runs / "archive" / "old-run" / ".pipeline").mkdir(parents=True)
    rows = fleet_state.fleet_rows(runs)
    assert [r["slug"] for r in rows] == ["real"]


def test_fleet_rows_empty_or_missing_root(tmp_path):
    assert fleet_state.fleet_rows(tmp_path / "nope") == []
    (tmp_path / "r2c_runs").mkdir()
    assert fleet_state.fleet_rows(tmp_path / "r2c_runs") == []


# ---------------------------------------------------------------------------
# Timeline
# ---------------------------------------------------------------------------


def _ev(event_type, seq, **extra):
    return json.dumps({
        "event_type": event_type, "sequence": seq,
        "timestamp": f"2026-07-06T20:00:{seq:02d}Z",
        "summary": extra.pop("summary", f"{event_type} happened"),
        **extra,
    })


def test_timeline_curates_categories_and_skips_lock_noise(tmp_path):
    runs = tmp_path / "r2c_runs"
    run_dir = _mk_run(runs, "demo", events_lines=[
        _ev("run_lock_acquired", 1),
        _ev("run_started", 2),
        _ev("agent_dispatch_started", 3, stage_id="stage_2b"),
        _ev("stage_halted", 4, stage_id="stage_2b",
            summary="Stage 2.b halted: dispatch timed out"),
        _ev("judge_decision_recorded", 5),
        _ev("brand_new_event_type", 6),
        _ev("smoke_trainability_failed", 7,
            summary="executed notebook shows silent non-learning"),
        _ev("pipeline_validation_issue", 8,
            summary="architecture validator coverage gap"),
        _ev("run_lock_released", 9),
    ])
    data = fleet_state.timeline(run_dir)
    by_type = {e["event_type"]: e for e in data["events"]}
    # Lock bookkeeping never reaches the researcher view.
    assert "run_lock_acquired" not in by_type
    assert "run_lock_released" not in by_type
    assert by_type["run_started"]["category"] == "milestone"
    assert by_type["agent_dispatch_started"]["category"] == "routine"
    assert by_type["stage_halted"]["category"] == "problem"
    assert by_type["judge_decision_recorded"]["category"] == "decision"
    # Unknown types surface as "other" instead of silently vanishing.
    assert by_type["brand_new_event_type"]["category"] == "other"
    # The trainability gate's finding is a problem, not "other" — the
    # gap the maintainer's 07-07 mock run surfaced live in the timeline.
    assert by_type["smoke_trainability_failed"]["category"] == "problem"
    assert by_type["pipeline_validation_issue"]["category"] == "problem"
    # Summaries are the driver's own plain language, untouched.
    assert by_type["stage_halted"]["summary"] == \
        "Stage 2.b halted: dispatch timed out"
    assert data["total"] == data["shown"] == 7
    assert data["skipped_malformed"] == 0


def test_timeline_tolerates_torn_trailing_line_and_counts_it(tmp_path):
    runs = tmp_path / "r2c_runs"
    run_dir = _mk_run(runs, "demo", events_lines=[_ev("run_started", 1)])
    events = run_dir / ".pipeline" / "run_events.jsonl"
    # A live driver appending mid-read leaves a torn tail.
    events.write_text(events.read_text(encoding="utf-8")
                      + '{"event_type": "stage_star', encoding="utf-8")
    data = fleet_state.timeline(run_dir)
    assert [e["event_type"] for e in data["events"]] == ["run_started"]
    assert data["skipped_malformed"] == 1


def test_timeline_limit_discloses_total_vs_shown(tmp_path):
    runs = tmp_path / "r2c_runs"
    run_dir = _mk_run(runs, "demo", events_lines=[
        _ev("stage_started", i) for i in range(1, 31)])
    data = fleet_state.timeline(run_dir, limit=10)
    assert data["total"] == 30
    assert data["shown"] == 10
    # Newest last, and the cut keeps the tail, not the head.
    assert data["events"][-1]["sequence"] == 30
    assert data["events"][0]["sequence"] == 21


def test_timeline_missing_events_file(tmp_path):
    runs = tmp_path / "r2c_runs"
    run_dir = _mk_run(runs, "demo")
    data = fleet_state.timeline(run_dir)
    assert data["events"] == [] and data["total"] == 0


# ---------------------------------------------------------------------------
# Artifact whitelist (the only files the viewer will ever serve)
# ---------------------------------------------------------------------------


def test_resolve_artifact_whitelisted_file(tmp_path):
    runs = tmp_path / "r2c_runs"
    run_dir = _mk_run(runs, "demo", report=True)
    got = fleet_state.resolve_artifact(runs, "demo", "REPORT.md")
    assert got == run_dir / "REPORT.md"


def test_resolve_artifact_refuses_non_whitelisted_names(tmp_path):
    runs = tmp_path / "r2c_runs"
    run_dir = _mk_run(runs, "demo")
    secret = run_dir / ".pipeline" / "params.json"
    secret.write_text("{}", encoding="utf-8")
    assert fleet_state.resolve_artifact(runs, "demo", "params.json") is None
    assert fleet_state.resolve_artifact(
        runs, "demo", ".pipeline/params.json") is None
    assert fleet_state.resolve_artifact(runs, "demo", "notebook.ipynb") is None


def test_resolve_artifact_refuses_traversal_slugs(tmp_path):
    runs = tmp_path / "r2c_runs"
    _mk_run(runs, "demo", report=True)
    outside = tmp_path / "REPORT.md"
    outside.write_text("outside", encoding="utf-8")
    assert fleet_state.resolve_artifact(runs, "..", "REPORT.md") is None
    assert fleet_state.resolve_artifact(runs, "../..", "REPORT.md") is None
    assert fleet_state.resolve_artifact(runs, "demo/..", "REPORT.md") is None
    assert fleet_state.resolve_artifact(runs, "ghost", "REPORT.md") is None


# ---------------------------------------------------------------------------
# Stale-running detection (2026-08-24): a killed driver never writes
# terminal state, so its progress.json advertises "running" forever. The
# fleet row reuses R2C-076's single staleness rule (the run-lock owner
# record) and renders a provably-dead owner as "interrupted".
# ---------------------------------------------------------------------------


def _write_lock_owner(run_dir, pid, host=None):
    import socket
    lock_dir = run_dir / ".pipeline" / "_lock"
    lock_dir.mkdir(parents=True, exist_ok=True)
    (lock_dir / "owner.json").write_text(json.dumps({
        "schema_version": "1.0",
        "run_id": run_dir.name,
        "pid": pid,
        "host": host or socket.gethostname(),
    }), encoding="utf-8")


def _dead_pid():
    """A pid that is conclusively not running (freshly reaped child)."""
    import os
    pid = os.fork()
    if pid == 0:
        os._exit(0)
    os.waitpid(pid, 0)
    return pid


def test_fleet_row_running_with_dead_owner_renders_interrupted(tmp_path):
    runs = tmp_path / "r2c_runs"
    run_dir = _mk_run(runs, "ghost", progress={
        "run_status": "running",
        "updated_at": "2026-08-06T22:33:26Z",
        "current": {"stage_id": "stage_2b", "label": "Stage 2.b"},
    })
    _write_lock_owner(run_dir, _dead_pid())
    row = fleet_state.fleet_rows(runs)[0]
    assert row["run_status"] == "interrupted"
    assert row["live"] is False


def test_fleet_row_running_with_live_owner_stays_running(tmp_path):
    import os
    runs = tmp_path / "r2c_runs"
    run_dir = _mk_run(runs, "alive", progress={
        "run_status": "running",
        "updated_at": "2026-08-06T22:33:26Z",
        "current": {"stage_id": "stage_2b", "label": "Stage 2.b"},
    })
    _write_lock_owner(run_dir, os.getpid())
    row = fleet_state.fleet_rows(runs)[0]
    assert row["run_status"] == "running"
    assert row["live"] is True


def test_fleet_row_running_without_owner_record_stays_running(tmp_path):
    # No owner record is NOT provably stale (R2C-076 semantics): keep the
    # honest "running" rather than guessing.
    runs = tmp_path / "r2c_runs"
    _mk_run(runs, "no-owner", progress={
        "run_status": "running",
        "updated_at": "2026-08-06T22:33:26Z",
        "current": {"stage_id": "stage_2b", "label": "Stage 2.b"},
    })
    row = fleet_state.fleet_rows(runs)[0]
    assert row["run_status"] == "running"
    assert row["live"] is True


def test_fleet_row_running_with_other_host_owner_stays_running(tmp_path):
    # A pid on another machine says nothing about liveness here.
    runs = tmp_path / "r2c_runs"
    run_dir = _mk_run(runs, "elsewhere", progress={
        "run_status": "running",
        "updated_at": "2026-08-06T22:33:26Z",
        "current": {"stage_id": "stage_2b", "label": "Stage 2.b"},
    })
    _write_lock_owner(run_dir, _dead_pid(), host="some-other-machine.local")
    row = fleet_state.fleet_rows(runs)[0]
    assert row["run_status"] == "running"
    assert row["live"] is True
