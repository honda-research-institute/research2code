from __future__ import annotations

import json

from progress_state import build_progress, write_progress
from run_events import append_event


CATALOG = [
    ("stage_0", "Stage 0 - Setup & Paper Ingestion"),
    ("stage_1", "Stage 1 - Paper Decomposition & Method Analysis"),
]


def test_build_progress_tracks_current_stage_and_agent():
    events = [
        {
            "event_type": "run_started",
            "run_id": "demo",
            "status": "running",
            "timestamp": "2026-06-17T18:00:00Z",
        },
        {
            "event_type": "stage_started",
            "run_id": "demo",
            "stage_id": "stage_0",
            "stage_label": "Stage 0 - Setup & Paper Ingestion",
            "summary": "stage 0 started",
            "timestamp": "2026-06-17T18:00:00Z",
        },
        {
            "event_type": "stage_completed",
            "run_id": "demo",
            "stage_id": "stage_0",
            "stage_label": "Stage 0 - Setup & Paper Ingestion",
            "summary": "paper.md present",
            "timestamp": "2026-06-17T18:05:00Z",
        },
        {
            "event_type": "stage_started",
            "run_id": "demo",
            "stage_id": "stage_1",
            "stage_label": "Stage 1 - Paper Decomposition & Method Analysis",
            "summary": "stage 1 started",
            "timestamp": "2026-06-17T18:05:00Z",
        },
        {
            "event_type": "agent_dispatch_started",
            "run_id": "demo",
            "summary": "decomposer started",
            "details": {
                "agent": "r2c-decomposer",
                "dispatch_id": "r2c-decomposer-abc",
                "timeout_s": 1800,
            },
            "timestamp": "2026-06-17T18:05:01Z",
        },
    ]

    progress = build_progress(run_id="demo", events=events, stage_catalog=CATALOG)

    assert progress["run_status"] == "running"
    assert progress["current"]["stage_id"] == "stage_1"
    assert progress["current"]["agent"] == "r2c-decomposer"
    assert progress["current"]["agent_timeout_s"] == 1800
    assert progress["stages"][0]["status"] == "completed"
    assert progress["stages"][0]["duration_s"] == 300.0
    assert progress["totals"]["stages_done"] == 1
    assert progress["totals"]["dispatches_started"] == 1


def test_write_progress_projects_run_events_jsonl(tmp_path):
    pipeline = tmp_path / ".pipeline"
    append_event(
        pipeline,
        event_type="run_started",
        run_id="demo",
        status="running",
        summary="started",
    )
    append_event(
        pipeline,
        event_type="stage_started",
        run_id="demo",
        stage_id="stage_0",
        stage_label="Stage 0 - Setup & Paper Ingestion",
        status="running",
        summary="stage 0 started",
    )

    path = write_progress(pipeline, run_id="demo", stage_catalog=CATALOG)
    data = json.loads(path.read_text(encoding="utf-8"))

    assert path == pipeline / "progress.json"
    assert data["slug"] == "demo"
    assert data["current"]["stage_id"] == "stage_0"
    assert data["stages"][0]["status"] == "running"


def test_build_progress_counts_failed_dispatches_separately():
    events = [
        {
            "event_type": "run_started",
            "run_id": "demo",
            "status": "running",
            "timestamp": "2026-06-17T18:00:00Z",
        },
        {
            "event_type": "agent_dispatch_started",
            "run_id": "demo",
            "details": {"agent": "r2c-decomposer"},
            "timestamp": "2026-06-17T18:00:01Z",
        },
        {
            "event_type": "agent_dispatch_completed",
            "run_id": "demo",
            "status": "failed",
            "details": {"agent": "r2c-decomposer", "completed": False},
            "timestamp": "2026-06-17T18:00:02Z",
        },
        {
            "event_type": "agent_dispatch_failed",
            "run_id": "demo",
            "status": "failed",
            "details": {"agent": "r2c-analyzer"},
            "timestamp": "2026-06-17T18:00:03Z",
        },
        {
            "event_type": "agent_dispatch_recovered",
            "run_id": "demo",
            "status": "completed",
            "details": {"agent": "r2c-method-coder"},
            "timestamp": "2026-06-17T18:00:04Z",
        },
    ]

    progress = build_progress(run_id="demo", events=events, stage_catalog=CATALOG)

    assert progress["totals"]["dispatches_started"] == 1
    assert progress["totals"]["dispatches_finished"] == 3
    assert progress["totals"]["dispatches_succeeded"] == 0
    assert progress["totals"]["dispatches_failed"] == 2
    assert progress["totals"]["dispatches_recovered"] == 1
    assert progress["totals"]["dispatches_completed"] == 0
