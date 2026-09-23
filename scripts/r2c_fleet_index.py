#!/usr/bin/env python3
"""One-call fleet index for the run companion (queue item 28).

Prints one line per run under r2c_runs/.
if it halted. Exists so the companion's greeting is ONE tool call — the
first live validation (2026-07-08) showed the per-run read burst (23
parallel progress.json reads) provoking a malformed tool call from the
Code tier, killing the greeting turn.

Never raises on a malformed run dir; unreadable fields print as "?".

Usage:
    python scripts/r2c_fleet_index.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

_LABEL_RE = re.compile(r"Delivery label:\s*\*\*(.+?)\*\*")


def _run_row(run_dir: Path, *, tag: str = "") -> str:
    status = "?"
    stopped = ""
    try:
        progress = json.loads(
            (run_dir / ".pipeline" / "progress.json").read_text(encoding="utf-8")
        )
        status = str(progress.get("run_status") or "?")
        if status in ("halted", "running"):
            for stage in progress.get("stages") or []:
                if stage.get("status") in ("halted", "running"):
                    stopped = f" @ {stage.get('stage_id')}"
                    break
    except Exception:
        pass
    label = "?"
    try:
        head = (run_dir / "REPORT.md").read_text(encoding="utf-8")[:2000]
        m = _LABEL_RE.search(head)
        if m:
            label = m.group(1)
    except Exception:
        pass
    return f"{run_dir.name}{tag} | {status}{stopped} | {label}"


def build_index(root: Path = ROOT) -> list[str]:
    rows: list[str] = []
    runs = root / "r2c_runs"
    if runs.is_dir():
        for entry in sorted(runs.iterdir()):
            if not entry.is_dir() or entry.name.startswith(("_", ".")):
                continue
            rows.append(_run_row(entry))
    return rows


def main() -> int:
    rows = build_index()
    if not rows:
        print("no runs found under r2c_runs/")
        return 0
    print("run | status | delivery label")
    for row in rows:
        print(row)
    return 0


if __name__ == "__main__":
    sys.exit(main())
