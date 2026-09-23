from __future__ import annotations

import json
import os

import r2c_watch


def test_load_state_reads_progress_json(tmp_path):
    run_dir = tmp_path / "r2c_runs" / "demo"
    progress = run_dir / ".pipeline" / "progress.json"
    progress.parent.mkdir(parents=True)
    progress.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "slug": "demo",
                "run_status": "running",
                "updated_at": "2026-06-17T18:00:00Z",
                "current": None,
                "stages": [],
                "totals": {},
            }
        ),
        encoding="utf-8",
    )

    state = r2c_watch.load_state(run_dir)

    assert state["slug"] == "demo"
    assert state["run_status"] == "running"
    assert state["run_dir"] == str(run_dir)
    assert state["progress_path"] == str(progress)


def test_load_state_handles_missing_progress_json(tmp_path):
    run_dir = tmp_path / "r2c_runs" / "demo"
    (run_dir / ".pipeline").mkdir(parents=True)

    state = r2c_watch.load_state(run_dir)

    assert state["slug"] == "demo"
    assert state["run_status"] == "waiting"
    assert state["current"] is None
    assert state["message"] == "progress.json is not present yet"


def test_load_state_projects_from_run_events_when_progress_missing(tmp_path):
    run_dir = tmp_path / "r2c_runs" / "demo"
    pipeline = run_dir / ".pipeline"
    pipeline.mkdir(parents=True)
    (pipeline / "run_events.jsonl").write_text(
        "\n".join(
            [
                json.dumps({
                    "event_type": "run_started",
                    "run_id": "demo",
                    "status": "running",
                    "timestamp": "2026-06-17T18:00:00Z",
                }),
                json.dumps({
                    "event_type": "stage_started",
                    "run_id": "demo",
                    "stage_id": "stage_0",
                    "stage_label": "Stage 0 - Setup & Paper Ingestion",
                    "status": "running",
                    "timestamp": "2026-06-17T18:00:01Z",
                }),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    state = r2c_watch.load_state(run_dir)

    assert state["run_status"] == "running"
    assert state["current"]["stage_id"] == "stage_0"
    assert state["message"] == "projected from run_events.jsonl; progress.json is not present"


def test_load_state_projects_from_newer_run_events_when_progress_stale(tmp_path):
    run_dir = tmp_path / "r2c_runs" / "demo"
    pipeline = run_dir / ".pipeline"
    pipeline.mkdir(parents=True)
    progress = pipeline / "progress.json"
    progress.write_text(
        json.dumps({
            "schema_version": "1.0",
            "slug": "demo",
            "run_status": "running",
            "updated_at": "2026-06-17T18:00:00Z",
            "current": None,
            "stages": [],
            "totals": {},
        }),
        encoding="utf-8",
    )
    events = pipeline / "run_events.jsonl"
    events.write_text(
        json.dumps({
            "event_type": "run_finished",
            "run_id": "demo",
            "status": "completed",
            "timestamp": "2026-06-17T18:05:00Z",
        })
        + "\n",
        encoding="utf-8",
    )
    os.utime(progress, (1, 1))
    os.utime(events, (2, 2))

    state = r2c_watch.load_state(run_dir)

    assert state["run_status"] == "completed"
    assert state["message"] == "projected from run_events.jsonl; progress.json is stale or not present"


def test_resolve_run_dir_prefers_recent_event_activity(tmp_path, monkeypatch):
    repo = tmp_path
    runs = repo / "r2c_runs"
    older = runs / "older"
    active = runs / "active"
    (older / ".pipeline").mkdir(parents=True)
    (active / ".pipeline").mkdir(parents=True)
    (older / ".pipeline" / "run_events.jsonl").write_text("", encoding="utf-8")
    (active / ".pipeline" / "run_events.jsonl").write_text("", encoding="utf-8")
    os.utime(older, (10, 10))
    os.utime(active, (1, 1))
    os.utime(older / ".pipeline" / "run_events.jsonl", (10, 10))
    os.utime(active / ".pipeline" / "run_events.jsonl", (20, 20))
    monkeypatch.setattr(r2c_watch, "REPO_ROOT", repo)

    assert r2c_watch.resolve_run_dir(None) == active.resolve()


# ---------------------------------------------------------------------------
# Staleness cue + terminal front-door message (failure-path spec §2/§5)
# ---------------------------------------------------------------------------

from datetime import datetime, timedelta, timezone


def _iso(dt) -> str:
    return dt.replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _running_progress(run_dir, *, stage_id, minutes_ago):
    updated = datetime.now(timezone.utc) - timedelta(minutes=minutes_ago)
    (run_dir / ".pipeline").mkdir(parents=True, exist_ok=True)
    (run_dir / ".pipeline" / "progress.json").write_text(json.dumps({
        "schema_version": "1.0",
        "slug": run_dir.name,
        "run_status": "running",
        "started_at": _iso(updated - timedelta(minutes=60)),
        "updated_at": _iso(updated),
        "current": {"stage_id": stage_id, "label": stage_id},
        "stages": [],
        "totals": {},
    }), encoding="utf-8")


def test_staleness_cue_fires_past_the_per_stage_window(tmp_path):
    run_dir = tmp_path / "r2c_runs" / "demo"
    _running_progress(run_dir, stage_id="stage_3b", minutes_ago=15)
    state = r2c_watch.load_state(run_dir)
    cue = state["staleness"]
    # stage_3b's typical quiet window is 5 min; 15 > 2x5.
    assert cue["stale"] is True
    assert "may be stuck" in cue["message"]
    assert "run_events.jsonl" in cue["message"]
    assert cue["typical_quiet_min"] == 5


def test_staleness_threshold_is_per_stage_not_global(tmp_path):
    # The same 15-minute silence is NORMAL inside stage_3c (the legitimate
    # 13-minute smoke-diagnostician dispatch, 2026-07-02) but stale for
    # stage_3b — the spec's per-stage requirement.
    run_dir = tmp_path / "r2c_runs" / "demo"
    _running_progress(run_dir, stage_id="stage_3c", minutes_ago=15)
    state = r2c_watch.load_state(run_dir)
    assert state["staleness"]["stale"] is False


def test_terminal_state_names_the_front_door_first(tmp_path):
    run_dir = tmp_path / "r2c_runs" / "demo"
    (run_dir / ".pipeline").mkdir(parents=True)
    (run_dir / ".pipeline" / "progress.json").write_text(json.dumps({
        "schema_version": "1.0", "slug": "demo", "run_status": "passed",
        "started_at": None, "updated_at": "2026-07-04T01:00:00Z",
        "current": None, "stages": [], "totals": {},
    }), encoding="utf-8")
    state = r2c_watch.load_state(run_dir)
    assert "staleness" not in state
    assert state["terminal_message"].startswith("Done — start at ")
    assert state["terminal_message"].endswith("REPORT.md")
    assert state["front_door"].endswith("REPORT.md")


def test_terminal_failed_state_says_stopped(tmp_path):
    run_dir = tmp_path / "r2c_runs" / "demo"
    (run_dir / ".pipeline").mkdir(parents=True)
    (run_dir / ".pipeline" / "progress.json").write_text(json.dumps({
        "schema_version": "1.0", "slug": "demo", "run_status": "failed",
        "started_at": None, "updated_at": "2026-07-04T01:00:00Z",
        "current": None, "stages": [], "totals": {},
    }), encoding="utf-8")
    state = r2c_watch.load_state(run_dir)
    assert state["terminal_message"].startswith("Stopped — start at ")
    assert "REPORT.md" in state["terminal_message"]


# ---------------------------------------------------------------------------
# Served templates + routing (the live-view app, 2026-07-06)
# ---------------------------------------------------------------------------

import threading
import urllib.error
import urllib.request

from http.server import ThreadingHTTPServer


def test_templates_carry_no_triple_quote_js_bug():
    # The watcher's day-one bug: a Python \" escape inside the template
    # collapsed to a bare `"""` in the served JS — a parse error that
    # killed the whole inline script, so the page never rendered live
    # data. Found 2026-07-06 while building the fleet view. This pins
    # the served text of every template.
    for template in (r2c_watch.HTML, r2c_watch.FLEET_HTML):
        assert '"""' not in template
    # The run page's slug injection point must exist for /run/<slug>.
    assert "__RUN_SLUG_JSON__" in r2c_watch.HTML
    assert "__RUN_SLUG_JSON__" not in r2c_watch.FLEET_HTML


def _serve(runs_root, default_run_dir=None):
    handler = r2c_watch.make_handler(runs_root, default_run_dir)
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, f"http://127.0.0.1:{server.server_address[1]}"


def _get(url):
    with urllib.request.urlopen(url, timeout=5) as response:
        return response.status, response.read().decode("utf-8")


def test_handler_routes_fleet_run_state_events_artifact(tmp_path):
    runs = tmp_path / "r2c_runs"
    pipeline = runs / "demo" / ".pipeline"
    pipeline.mkdir(parents=True)
    (pipeline / "progress.json").write_text(json.dumps({
        "schema_version": "1.0", "slug": "demo", "run_status": "running",
        "updated_at": "2026-07-06T20:00:00Z",
        "current": {"stage_id": "stage_1", "label": "Stage 1"},
        "stages": [], "totals": {},
    }), encoding="utf-8")
    (pipeline / "run_events.jsonl").write_text(json.dumps({
        "event_type": "run_started", "sequence": 1,
        "timestamp": "2026-07-06T20:00:00Z", "summary": "R2C run started",
    }) + "\n", encoding="utf-8")
    (runs / "demo" / "REPORT.md").write_text("# Run Report\n",
                                             encoding="utf-8")
    server, base = _serve(runs)
    try:
        status, body = _get(f"{base}/")
        assert status == 200 and "R2C Runs" in body
        status, body = _get(f"{base}/fleet")
        rows = json.loads(body)
        assert status == 200 and rows[0]["slug"] == "demo"
        status, body = _get(f"{base}/run/demo")
        assert status == 200
        assert 'const RUN_SLUG = "demo";' in body
        assert "__RUN_SLUG_JSON__" not in body
        status, body = _get(f"{base}/state?run=demo")
        assert status == 200 and json.loads(body)["run_status"] == "running"
        status, body = _get(f"{base}/events?run=demo")
        events = json.loads(body)
        assert status == 200
        assert events["events"][0]["event_type"] == "run_started"
        status, body = _get(f"{base}/artifact/demo/REPORT.md")
        assert status == 200 and body == "# Run Report\n"
    finally:
        server.shutdown()


def test_handler_404s_unknown_run_and_off_whitelist_artifacts(tmp_path):
    runs = tmp_path / "r2c_runs"
    pipeline = runs / "demo" / ".pipeline"
    pipeline.mkdir(parents=True)
    (pipeline / "params.json").write_text("{}", encoding="utf-8")
    server, base = _serve(runs)
    try:
        for path in ("/run/ghost", "/artifact/demo/params.json",
                     "/artifact/demo/.pipeline%2Fparams.json",
                     "/artifact/..%2F..%2Fetc/REPORT.md"):
            try:
                status, _ = _get(f"{base}{path}")
            except urllib.error.HTTPError as err:
                status = err.code
            assert status == 404, path
    finally:
        server.shutdown()


def test_state_without_query_falls_back_to_latest_activity(tmp_path):
    runs = tmp_path / "r2c_runs"
    for slug, stamp in (("older", 10), ("active", 20)):
        pipeline = runs / slug / ".pipeline"
        pipeline.mkdir(parents=True)
        (pipeline / "progress.json").write_text(json.dumps({
            "schema_version": "1.0", "slug": slug, "run_status": "halted",
            "updated_at": "2026-07-06T20:00:00Z", "current": None,
            "stages": [], "totals": {},
        }), encoding="utf-8")
        os.utime(pipeline / "progress.json", (stamp, stamp))
        os.utime(runs / slug, (stamp, stamp))
    server, base = _serve(runs)
    try:
        status, body = _get(f"{base}/state")
        assert status == 200 and json.loads(body)["slug"] == "active"
    finally:
        server.shutdown()
