"""Fleet index for the run companion (queue item 28).

One call, one line per run, never raises on malformed run dirs — the
index exists so the companion's greeting is a single tool call instead
of the per-run read burst that killed the first live greeting turn.
"""

from __future__ import annotations

import json
from pathlib import Path

from scripts.r2c_fleet_index import build_index


def _make_run(root: Path, name: str, *, status: str, label: str | None,
              halted_stage: str | None = None) -> None:
    run = root / "r2c_runs" / name
    (run / ".pipeline").mkdir(parents=True)
    stages = []
    if halted_stage:
        stages.append({"stage_id": halted_stage, "status": "halted"})
    (run / ".pipeline" / "progress.json").write_text(json.dumps({
        "run_status": status, "stages": stages,
    }))
    if label is not None:
        (run / "REPORT.md").write_text(
            f"# Run Report\n\nDelivery label: **{label}**.\n"
        )


def test_index_rows_cover_status_label_and_halt_point(tmp_path):
    _make_run(tmp_path, "alpha", status="degraded", label="draft, needs attention")
    _make_run(tmp_path, "bravo", status="halted", label="explanation only",
              halted_stage="stage_2b")
    (tmp_path / "r2c_runs" / "_batch_logs").mkdir()  # skipped
    (tmp_path / "r2c_runs" / "broken").mkdir()       # no pipeline at all
    rows = build_index(tmp_path)
    assert "alpha | degraded | draft, needs attention" in rows
    assert "bravo | halted @ stage_2b | explanation only" in rows
    assert "broken | ? | ?" in rows
    assert not any(r.startswith("_batch_logs") for r in rows)
