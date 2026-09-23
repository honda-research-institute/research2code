"""B-04 rider: nothing lints scripts/ itself, which is how two unused
imports lived long enough to be written up as an architecture finding.

This guard runs pyflakes over every script and fails on the exact class
that motivated it — unused imports — while honoring the repo's deliberate
`# noqa`-marked availability probes (`import torch` inside try/except) and
re-export blocks (validate_notebook_output's vendoring seam), which
pyflakes itself cannot suppress. Style-class findings (f-string without
placeholders, unused locals) are out of scope on purpose: they carry no
drift risk and cleaning them wholesale would churn live-run files.

Runs in CI through the existing pytest step; pyflakes is an optional
dependency (lint_generated_code.py degrades without it), so the guard
skips at function level when it is absent — module-level gates are pinned
by test_dependency_gate_pins.py and this must not join that map.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_no_unused_imports_in_scripts():
    pytest.importorskip("pyflakes")
    files = sorted(str(p) for p in (REPO_ROOT / "scripts").rglob("*.py"))
    assert files
    proc = subprocess.run(
        [sys.executable, "-m", "pyflakes", *files],
        capture_output=True, text=True, cwd=REPO_ROOT,
    )
    offenders = []
    for line in proc.stdout.splitlines():
        if "imported but unused" not in line:
            continue
        try:
            path, lineno, _rest = line.split(":", 2)
            src_line = Path(path).read_text(encoding="utf-8").splitlines()[int(lineno) - 1]
        except (ValueError, OSError, IndexError):
            offenders.append(line)
            continue
        if "noqa" in src_line:
            continue  # deliberate availability probe or re-export seam
        offenders.append(line)
    assert not offenders, (
        "unused imports in scripts/ (delete them, or mark a deliberate "
        "availability-probe/re-export with # noqa: F401):\n"
        + "\n".join(offenders)
    )
