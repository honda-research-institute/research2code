from __future__ import annotations

import argparse
import json
import sys
import webbrowser
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

from fleet_state import fleet_rows, is_run_dir, resolve_artifact, timeline
from progress_state import build_progress


REPO_ROOT = Path(__file__).resolve().parent.parent

FALLBACK_STAGE_CATALOG = [
    ("stage_0", "Stage 0 - Setup & Paper Ingestion"),
    ("stage_1", "Stage 1 - Paper Decomposition & Method Analysis"),
    ("stage_1x", "Stage 1.x - Method Explanation (METHOD.md)"),
    ("stage_2a", "Stage 2.a - Package Scaffold"),
    ("stage_2b", "Stage 2.b - Architecture & Training"),
    ("stage_2c", "Stage 2.c - Method Implementation"),
    ("stage_2d", "Stage 2.d - Package Finalization"),
    ("stage_2x", "Stage 2.x - Parameter Derivation"),
    ("stage_3a", "Stage 3.a - Notebook Authoring"),
    ("stage_3b", "Stage 3.b - Notebook Rendering"),
    ("stage_3c", "Stage 3.c - Smoke Execution"),
    ("stage_4", "Stage 4 - Paper-Fidelity Review"),
    ("stage_5", "Stage 5 - Findings Routing"),
]


FLEET_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>R2C Runs</title>
  <style>
    :root {
      color-scheme: light dark;
      font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: #111827;
      color: #e5e7eb;
    }
    body { margin: 0; min-height: 100vh; background: #111827; }
    main { max-width: 1120px; margin: 0 auto; padding: 28px 20px 40px; }
    h1 { margin: 0 0 4px; font-size: 24px; }
    .meta { color: #9ca3af; font-size: 14px; margin-bottom: 18px; }
    table {
      width: 100%;
      border-collapse: collapse;
      border-radius: 6px;
      border: 1px solid #374151;
      overflow: hidden;
    }
    th, td {
      padding: 10px 12px;
      border-bottom: 1px solid #374151;
      text-align: left;
      font-size: 14px;
      vertical-align: top;
    }
    th {
      color: #9ca3af;
      font-size: 12px;
      text-transform: uppercase;
      background: #1f2937;
    }
    tr:last-child td { border-bottom: 0; }
    tr.run-row { cursor: pointer; }
    tr.run-row:hover td { background: #1f2937; }
    a { color: #93c5fd; text-decoration: none; }
    a:hover { text-decoration: underline; }
    .pill {
      display: inline-flex;
      min-width: 86px;
      justify-content: center;
      padding: 3px 8px;
      border-radius: 999px;
      background: #374151;
      color: #e5e7eb;
      font-size: 12px;
      font-weight: 700;
    }
    .completed, .skipped { background: #14532d; color: #dcfce7; }
    .running { background: #1d4ed8; color: #dbeafe; }
    .halted { background: #7f1d1d; color: #fee2e2; }
    .degraded { background: #854d0e; color: #fef3c7; }
    .pending { background: #374151; color: #d1d5db; }
    .label { font-weight: 700; }
    .nolabel { color: #9ca3af; font-weight: 400; }
    .empty {
      padding: 28px;
      text-align: center;
      color: #9ca3af;
      border: 1px dashed #374151;
      border-radius: 6px;
      margin-top: 16px;
    }
  </style>
</head>
<body>
  <main>
    <h1>R2C Runs</h1>
    <div class="meta">Every run under <code>r2c_runs/</code>. Click a row for the live view.
      Labels come verbatim from each run's <code>final_manifest.json</code>;
      <code>REPORT.md</code> is always the front door for trust.</div>
    <table id="fleet-table">
      <thead>
        <tr>
          <th>Paper</th>
          <th>Delivery label</th>
          <th>Status</th>
          <th>Stage</th>
          <th>Last activity</th>
          <th>Report</th>
        </tr>
      </thead>
      <tbody id="rows"></tbody>
    </table>
    <div class="empty" id="empty" style="display:none">
      No runs yet. Launch one with <code>/r2c-run &lt;paper&gt;</code> in the opencode window
      and this page will pick it up.
    </div>
  </main>
  <script>
    const text = (v) => v === null || v === undefined || v === "" ? "" : String(v);
    const escapeHtml = (v) => text(v).replace(/[&<>"']/g, (ch) => ({
      "&": "&amp;", "<": "&lt;", ">": "&gt;", "\\"": "&quot;", "'": "&#39;",
    }[ch]));
    const STATUS_PILL = {
      passed: "completed", completed: "completed", running: "running",
      halted: "halted", degraded: "degraded", failed: "halted",
    };
    const age = (iso) => {
      if (!iso) return "";
      const t = Date.parse(iso);
      if (Number.isNaN(t)) return "";
      const s = Math.max(0, (Date.now() - t) / 1000);
      if (s < 90) return `${Math.round(s)}s ago`;
      if (s < 5400) return `${Math.round(s / 60)}m ago`;
      if (s < 129600) return `${Math.round(s / 3600)}h ago`;
      return `${Math.round(s / 86400)}d ago`;
    };
    function render(rows) {
      document.getElementById("empty").style.display = rows.length ? "none" : "block";
      document.getElementById("fleet-table").style.display = rows.length ? "" : "none";
      document.getElementById("rows").innerHTML = rows.map((r) => {
        const status = text(r.run_status || "waiting");
        const label = r.label
          ? `<span class="label">${escapeHtml(r.label)}${r.label_partial ? " (partial)" : ""}</span>`
          : `<span class="nolabel">no delivery yet</span>`;
        const report = r.report_available
          ? `<a href="/artifact/${encodeURIComponent(r.slug)}/REPORT.md" target="_blank">REPORT.md</a>`
          : "";
        const stage = r.live ? text(r.current_stage_label) : "";
        return `<tr class="run-row" onclick="window.location='/run/${encodeURIComponent(r.slug)}'">
          <td><a href="/run/${encodeURIComponent(r.slug)}">${escapeHtml(r.slug)}</a></td>
          <td>${label}</td>
          <td><span class="pill ${STATUS_PILL[status] || "pending"}">${escapeHtml(status)}</span></td>
          <td>${escapeHtml(stage)}</td>
          <td>${escapeHtml(age(r.last_activity_at))}</td>
          <td>${report}</td>
        </tr>`;
      }).join("");
    }
    async function refresh() {
      try {
        const response = await fetch("/fleet", { cache: "no-store" });
        render(await response.json());
      } catch (error) { /* keep the last view */ }
    }
    setInterval(refresh, 2500);
    refresh();
  </script>
</body>
</html>
"""


HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>R2C Run Watch</title>
  <style>
    :root {
      color-scheme: light dark;
      font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: #111827;
      color: #e5e7eb;
    }
    body {
      margin: 0;
      min-height: 100vh;
      background: #111827;
    }
    main {
      max-width: 1120px;
      margin: 0 auto;
      padding: 28px 20px 40px;
    }
    header {
      display: flex;
      justify-content: space-between;
      gap: 16px;
      align-items: flex-start;
      padding-bottom: 18px;
      border-bottom: 1px solid #374151;
    }
    h1 {
      margin: 0;
      font-size: 24px;
      line-height: 1.2;
      letter-spacing: 0;
    }
    .meta {
      margin-top: 6px;
      color: #9ca3af;
      font-size: 14px;
    }
    .status {
      min-width: 150px;
      padding: 8px 10px;
      border-radius: 6px;
      background: #1f2937;
      text-align: center;
      font-weight: 700;
      text-transform: uppercase;
      font-size: 13px;
      letter-spacing: 0;
    }
    .summary {
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 12px;
      margin: 18px 0;
    }
    .metric {
      padding: 12px;
      border-radius: 6px;
      background: #1f2937;
      border: 1px solid #374151;
    }
    .metric b {
      display: block;
      font-size: 20px;
      line-height: 1.2;
    }
    .metric span {
      color: #9ca3af;
      font-size: 13px;
    }
    .current {
      padding: 14px;
      margin-bottom: 16px;
      border-radius: 6px;
      border: 1px solid #2563eb;
      background: #172554;
    }
    .current h2 {
      margin: 0 0 6px;
      font-size: 17px;
      letter-spacing: 0;
    }
    .current p {
      margin: 0;
      color: #bfdbfe;
      font-size: 14px;
    }
    table {
      width: 100%;
      border-collapse: collapse;
      overflow: hidden;
      border-radius: 6px;
      border: 1px solid #374151;
    }
    th, td {
      padding: 10px 12px;
      border-bottom: 1px solid #374151;
      text-align: left;
      vertical-align: top;
      font-size: 14px;
    }
    th {
      color: #9ca3af;
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0;
      background: #1f2937;
    }
    tr:last-child td {
      border-bottom: 0;
    }
    .pill {
      display: inline-flex;
      min-width: 86px;
      justify-content: center;
      padding: 3px 8px;
      border-radius: 999px;
      background: #374151;
      color: #e5e7eb;
      font-size: 12px;
      font-weight: 700;
    }
    .completed, .skipped { background: #14532d; color: #dcfce7; }
    .running { background: #1d4ed8; color: #dbeafe; }
    .halted { background: #7f1d1d; color: #fee2e2; }
    .degraded { background: #854d0e; color: #fef3c7; }
    .pending { background: #374151; color: #d1d5db; }
    .note {
      color: #9ca3af;
      max-width: 520px;
    }
    .banner {
      display: none;
      padding: 14px;
      margin: 16px 0;
      border-radius: 6px;
      font-size: 14px;
    }
    .banner.stale {
      border: 1px solid #b45309;
      background: #451a03;
      color: #fde68a;
    }
    .banner.terminal {
      border: 1px solid #15803d;
      background: #052e16;
      color: #bbf7d0;
    }
    .banner.terminal.stopped {
      border-color: #b91c1c;
      background: #450a0a;
      color: #fecaca;
    }
    @media (max-width: 760px) {
      header, .summary {
        display: block;
      }
      .status, .metric {
        margin-top: 12px;
      }
      th:nth-child(3), td:nth-child(3), th:nth-child(4), td:nth-child(4) {
        display: none;
      }
    }
  </style>
</head>
<body>
  <main>
    <header>
      <div>
        <div class="meta"><a href="/" style="color:#93c5fd;text-decoration:none">&#8592; All runs</a></div>
        <h1 id="title">R2C Run</h1>
        <div class="meta" id="meta">Loading...</div>
      </div>
      <div class="status" id="status">waiting</div>
    </header>
    <section class="banner terminal" id="terminal"></section>
    <section class="banner stale" id="staleness"></section>
    <section class="summary" id="summary"></section>
    <section class="current" id="current"></section>
    <table>
      <thead>
        <tr>
          <th>Stage</th>
          <th>Status</th>
          <th>Duration</th>
          <th>Updated</th>
          <th>Notes</th>
        </tr>
      </thead>
      <tbody id="stages"></tbody>
    </table>
    <section style="margin-top: 26px">
      <h2 style="font-size: 17px; margin: 0 0 4px">Behavioral timeline</h2>
      <div class="meta" style="margin-bottom: 8px">
        <span id="timeline-meta">Loading...</span>
        &nbsp;&middot;&nbsp;
        <label style="cursor: pointer">
          <input type="checkbox" id="show-routine">
          show routine dispatch and validation traffic
        </label>
      </div>
      <table>
        <thead>
          <tr>
            <th>Time</th>
            <th>Kind</th>
            <th>Stage</th>
            <th>What happened</th>
          </tr>
        </thead>
        <tbody id="timeline"></tbody>
      </table>
    </section>
  </main>
  <script>
    const RUN_SLUG = __RUN_SLUG_JSON__;
    const RUN_QS = RUN_SLUG ? `?run=${encodeURIComponent(RUN_SLUG)}` : "";
    const fmtDuration = (seconds) => {
      if (seconds === null || seconds === undefined) return "";
      const total = Math.max(0, Math.floor(seconds));
      const h = Math.floor(total / 3600);
      const m = Math.floor((total % 3600) / 60);
      const s = total % 60;
      if (h) return `${h}h ${m}m`;
      if (m) return `${m}m ${s}s`;
      return `${s}s`;
    };
    const elapsedSince = (iso) => {
      if (!iso) return null;
      const start = Date.parse(iso);
      if (Number.isNaN(start)) return null;
      return (Date.now() - start) / 1000;
    };
    const text = (value) => value === null || value === undefined || value === "" ? "" : String(value);
    const escapeHtml = (value) => text(value).replace(/[&<>"']/g, (ch) => ({
      "&": "&amp;",
      "<": "&lt;",
      ">": "&gt;",
      "\\"": "&quot;",
      "'": "&#39;",
    }[ch]));
    let latest = null;
    function render(data) {
      latest = data;
      document.getElementById("title").textContent = data.slug ? `R2C Run: ${data.slug}` : "R2C Run";
      document.getElementById("status").textContent = text(data.run_status || "waiting");
      document.getElementById("meta").textContent = `Updated ${text(data.updated_at)} | ${text(data.run_dir || "")}`;
      const totals = data.totals || {};
      document.getElementById("summary").innerHTML = [
        ["Stages", `${totals.stages_done || 0}/${totals.stages_total || 0}`],
        ["Elapsed", fmtDuration(totals.elapsed_s)],
        ["Dispatches", `${totals.dispatches_succeeded || 0} ok, ${totals.dispatches_failed || 0} failed, ${totals.dispatches_recovered || 0} recovered`],
        ["Issues", `${totals.halts || 0} halted, ${totals.degrades || 0} degraded`],
      ].map(([label, value]) => `<div class="metric"><b>${escapeHtml(value)}</b><span>${escapeHtml(label)}</span></div>`).join("");
      const terminalEl = document.getElementById("terminal");
      if (data.terminal_message) {
        terminalEl.textContent = data.terminal_message;
        terminalEl.className = "banner terminal" +
          (String(data.terminal_message).startsWith("Stopped") ? " stopped" : "");
        terminalEl.style.display = "block";
      } else {
        terminalEl.style.display = "none";
      }
      const staleEl = document.getElementById("staleness");
      if (data.staleness && data.staleness.stale) {
        staleEl.textContent = data.staleness.message ||
          `No activity for ${data.staleness.minutes_since_last_event} min - the run may be stuck.`;
        staleEl.style.display = "block";
      } else {
        staleEl.style.display = "none";
      }
      const current = data.current;
      const currentEl = document.getElementById("current");
      if (current) {
        const stageElapsed = fmtDuration(elapsedSince(current.started_at));
        const agentElapsed = fmtDuration(elapsedSince(current.agent_started_at));
        const agent = current.agent ? `${current.agent} for ${agentElapsed}` : "No active agent dispatch";
        currentEl.innerHTML = `<h2>${escapeHtml(current.label)}</h2><p>Running for ${escapeHtml(stageElapsed)}. ${escapeHtml(agent)}.</p>`;
        currentEl.style.display = "block";
      } else {
        currentEl.style.display = "none";
      }
      document.getElementById("stages").innerHTML = (data.stages || []).map((stage) => {
        const status = text(stage.status || "pending");
        const updated = stage.ended_at || stage.started_at || "";
        return `<tr>
          <td>${escapeHtml(stage.label || stage.stage_id)}</td>
          <td><span class="pill ${escapeHtml(status)}">${escapeHtml(status)}</span></td>
          <td>${escapeHtml(fmtDuration(stage.duration_s))}</td>
          <td>${escapeHtml(updated)}</td>
          <td class="note">${escapeHtml(stage.notes)}</td>
        </tr>`;
      }).join("");
    }
    async function refresh() {
      try {
        const response = await fetch(`/state${RUN_QS}`, { cache: "no-store" });
        render(await response.json());
      } catch (error) {
        render({ run_status: "waiting", updated_at: new Date().toISOString(), totals: {}, stages: [], current: null });
      }
    }
    const CATEGORY_PILL = {
      milestone: "completed", problem: "halted",
      decision: "degraded", routine: "pending", other: "pending",
    };
    let latestTimeline = null;
    function renderTimeline(data) {
      latestTimeline = data;
      const showRoutine = document.getElementById("show-routine").checked;
      const rows = (data.events || []).filter((e) => showRoutine || e.category !== "routine");
      const bits = [`showing ${rows.length} of ${data.total} events`];
      if (data.shown < data.total) bits.push(`last ${data.shown} loaded`);
      if (data.skipped_malformed) bits.push(`${data.skipped_malformed} unreadable line(s) skipped`);
      document.getElementById("timeline-meta").textContent = bits.join(", ");
      document.getElementById("timeline").innerHTML = rows.slice().reverse().map((e) => `<tr>
        <td>${escapeHtml(e.timestamp || "")}</td>
        <td><span class="pill ${CATEGORY_PILL[e.category] || "pending"}">${escapeHtml(e.category || "")}</span></td>
        <td class="note">${escapeHtml(e.stage_label || e.stage_id || "")}</td>
        <td class="note">${escapeHtml(e.summary || e.event_type || "")}</td>
      </tr>`).join("");
    }
    async function refreshTimeline() {
      try {
        const response = await fetch(`/events${RUN_QS}`, { cache: "no-store" });
        renderTimeline(await response.json());
      } catch (error) { /* keep the last view */ }
    }
    document.getElementById("show-routine").addEventListener("change", () => {
      if (latestTimeline) renderTimeline(latestTimeline);
    });
    setInterval(() => { if (latest) render(latest); }, 1000);
    setInterval(refresh, 1500);
    setInterval(refreshTimeline, 3000);
    refresh();
    refreshTimeline();
  </script>
</body>
</html>
"""


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


# Minutes of event silence that are NORMAL per stage (failure-path spec §2:
# the cue threshold is per-stage, not global — the 2026-07-02 run's longest
# legitimate silent gap was a 13-minute smoke-diagnostician dispatch inside
# stage_3c). The cue fires beyond 2x the typical window.
STAGE_TYPICAL_QUIET_MIN = {
    "stage_0": 5,
    "stage_1": 15,
    "stage_1x": 10,
    "stage_2a": 10,
    "stage_2b": 15,
    "stage_2c": 15,
    "stage_2d": 10,
    "stage_2x": 15,
    "stage_3a": 15,
    "stage_3b": 5,
    "stage_3c": 20,
    "stage_4": 15,
    "stage_5": 10,
}
DEFAULT_TYPICAL_QUIET_MIN = 15


def _staleness(data: dict) -> dict | None:
    """Staleness cue for a RUNNING run (failure-path spec §2): time since
    the last event against the current stage's typical quiet window, so a
    stuck run is impossible to confuse with a working one. None for
    terminal/waiting states."""
    if data.get("run_status") != "running":
        return None
    try:
        last = datetime.fromisoformat(
            str(data.get("updated_at")).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    age_min = (datetime.now(timezone.utc) - last).total_seconds() / 60.0
    stage_id = str((data.get("current") or {}).get("stage_id") or "")
    typical = STAGE_TYPICAL_QUIET_MIN.get(stage_id, DEFAULT_TYPICAL_QUIET_MIN)
    cue = {
        "stale": age_min > 2 * typical,
        "minutes_since_last_event": int(round(age_min)),
        "typical_quiet_min": typical,
    }
    if cue["stale"]:
        events_path = Path(str(data.get("run_dir") or "")) / ".pipeline" / "run_events.jsonl"
        cue["message"] = (
            f"No activity for {cue['minutes_since_last_event']} min — this "
            f"stage typically shows activity within {typical}. The run may "
            f"be stuck: check the tail of {events_path}."
        )
    return cue


def _terminal_message(data: dict, run_dir: Path) -> str | None:
    """One-path terminal message (failure-path spec §5): every terminal
    state names the front door, r2c_runs/<slug>/REPORT.md, first."""
    status = data.get("run_status")
    if status in (None, "running", "waiting"):
        return None
    front_door = run_dir / "REPORT.md"
    if status == "passed":
        return f"Done — start at {front_door}"
    return (f"Stopped — start at {front_door} "
            f"(the stopped-run summary is at the top)")


def resolve_run_dir(slug: str | None) -> Path:
    runs_root = REPO_ROOT / "r2c_runs"
    if slug:
        direct = Path(slug)
        if direct.is_dir():
            return direct.resolve()
        candidate = runs_root / slug
        if candidate.is_dir():
            return candidate.resolve()
        raise SystemExit(f"run not found: {slug}")
    candidates = [path for path in runs_root.iterdir() if path.is_dir()] if runs_root.is_dir() else []
    if not candidates:
        raise SystemExit("no runs found under r2c_runs/")
    return max(candidates, key=_run_activity_mtime).resolve()


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


def _load_progress_from_events(run_dir: Path, *, message: str) -> dict:
    events_path = run_dir / ".pipeline" / "run_events.jsonl"
    events = [
        json.loads(line)
        for line in events_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    data = build_progress(
        run_id=run_dir.name,
        events=events,
        stage_catalog=FALLBACK_STAGE_CATALOG,
    )
    data["message"] = message
    return data


def load_state(run_dir: Path) -> dict:
    progress_path = run_dir / ".pipeline" / "progress.json"
    events_path = run_dir / ".pipeline" / "run_events.jsonl"
    if (
        events_path.is_file()
        and (
            not progress_path.is_file()
            or events_path.stat().st_mtime > progress_path.stat().st_mtime
        )
    ):
        message = (
            "projected from run_events.jsonl; progress.json is stale or not present"
            if progress_path.is_file()
            else "projected from run_events.jsonl; progress.json is not present"
        )
        data = _load_progress_from_events(run_dir, message=message)
    elif progress_path.is_file():
        data = json.loads(progress_path.read_text(encoding="utf-8"))
    else:
        data = {
            "schema_version": "1.0",
            "slug": run_dir.name,
            "run_status": "waiting",
            "started_at": None,
            "updated_at": _now_iso(),
            "current": None,
            "stages": [],
            "totals": {
                "stages_done": 0,
                "stages_total": 0,
                "elapsed_s": None,
                "degrades": 0,
                "halts": 0,
                "dispatches_started": 0,
                "dispatches_finished": 0,
                "dispatches_succeeded": 0,
                "dispatches_failed": 0,
                "dispatches_recovered": 0,
                "dispatches_completed": 0,
            },
            "message": "progress.json is not present yet",
        }
    data["run_dir"] = str(run_dir)
    data["progress_path"] = str(progress_path)
    data["watcher_updated_at"] = _now_iso()
    staleness = _staleness(data)
    if staleness is not None:
        data["staleness"] = staleness
    terminal = _terminal_message(data, run_dir)
    if terminal is not None:
        data["terminal_message"] = terminal
        data["front_door"] = str(run_dir / "REPORT.md")
    return data


def _request_run_dir(runs_root: Path, slug: str) -> Path | None:
    """A run dir DIRECTLY under runs_root, validated by listing so a
    slug can never traverse (no separators or dot-dots resolve)."""
    if not runs_root.is_dir():
        return None
    for path in runs_root.iterdir():
        if path.name == slug and is_run_dir(path):
            return path.resolve()
    return None


def _waiting_state() -> dict:
    return {
        "schema_version": "1.0",
        "slug": None,
        "run_status": "waiting",
        "started_at": None,
        "updated_at": _now_iso(),
        "current": None,
        "stages": [],
        "totals": {},
        "message": "no runs found under r2c_runs/",
    }


def make_handler(runs_root: Path, default_run_dir: Path | None):
    class Handler(BaseHTTPRequestHandler):
        def _send(self, body: bytes, content_type: str) -> None:
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", content_type)
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _send_json(self, payload: dict | list) -> None:
            self._send(json.dumps(payload).encode("utf-8"),
                       "application/json; charset=utf-8")

        def _resolve_query_run(self, query: str) -> Path | None:
            """Run dir for /state and /events: explicit ?run=<slug>,
            else the server's pinned default, else latest activity."""
            slug = (parse_qs(query).get("run") or [None])[0]
            if slug:
                return _request_run_dir(runs_root, unquote(slug))
            if default_run_dir is not None:
                return default_run_dir
            candidates = [p for p in runs_root.iterdir() if is_run_dir(p)] \
                if runs_root.is_dir() else []
            if not candidates:
                return None
            return max(candidates, key=_run_activity_mtime).resolve()

        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            path = parsed.path
            if path in {"/", "/index.html"}:
                self._send(FLEET_HTML.encode("utf-8"),
                           "text/html; charset=utf-8")
                return
            if path == "/fleet":
                self._send_json(fleet_rows(runs_root))
                return
            if path.startswith("/run/"):
                run_dir = _request_run_dir(runs_root, unquote(path[len("/run/"):]))
                if run_dir is None:
                    self.send_error(HTTPStatus.NOT_FOUND, "run not found")
                    return
                body = HTML.replace("__RUN_SLUG_JSON__", json.dumps(run_dir.name))
                self._send(body.encode("utf-8"), "text/html; charset=utf-8")
                return
            if path == "/state":
                run_dir = self._resolve_query_run(parsed.query)
                self._send_json(load_state(run_dir) if run_dir else _waiting_state())
                return
            if path == "/events":
                run_dir = self._resolve_query_run(parsed.query)
                if run_dir is None:
                    self._send_json({"events": [], "total": 0, "shown": 0,
                                     "skipped_malformed": 0})
                    return
                self._send_json(timeline(run_dir))
                return
            if path.startswith("/artifact/"):
                parts = [unquote(p) for p in path.split("/") if p]
                # parts = ["artifact", slug, filename]
                artifact = resolve_artifact(runs_root, parts[1], parts[2]) \
                    if len(parts) == 3 else None
                if artifact is None:
                    self.send_error(HTTPStatus.NOT_FOUND, "not found")
                    return
                self._send(artifact.read_bytes(),
                           "text/plain; charset=utf-8")
                return
            self.send_error(HTTPStatus.NOT_FOUND, "not found")

        def log_message(self, fmt: str, *args) -> None:
            return

    return Handler


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Serve a read-only R2C viewer: fleet of all runs at /, "
                    "live per-run view with behavioral timeline at /run/<slug>.")
    parser.add_argument("run", nargs="?", help="run slug under r2c_runs/ or a run directory")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--open", action="store_true", help="open the page in the default browser")
    parser.add_argument("--once", action="store_true", help="print the current state JSON and exit")
    args = parser.parse_args(argv)

    if args.once:
        json.dump(load_state(resolve_run_dir(args.run)), sys.stdout, indent=2)
        sys.stdout.write("\n")
        return 0

    runs_root = REPO_ROOT / "r2c_runs"
    default_run_dir = resolve_run_dir(args.run) if args.run else None
    server = ThreadingHTTPServer(
        (args.host, args.port), make_handler(runs_root, default_run_dir))
    base = f"http://{args.host}:{args.port}"
    url = f"{base}/run/{default_run_dir.name}" if default_run_dir else f"{base}/"
    print(f"serving R2C viewer for {runs_root} at {base}/ "
          f"(landing: {url})", flush=True)
    if args.open:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
