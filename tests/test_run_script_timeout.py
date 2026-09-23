"""run_script converts a subprocess timeout into a failed CompletedProcess.

Regression for the Rethinking-Grouping 2026-07-13 crash: a stage-2d validator
dry-run stalled on a network fetch, subprocess.TimeoutExpired unwound past
every packaging path, and the run finished with no manifest of any kind. The
driver's contract is that validator failures are ordinary nonzero-returncode
results that route into fix-loop / halt machinery, so a timeout must come
back the same way instead of raising.
"""

from __future__ import annotations

import subprocess

import run_pipeline


def test_timeout_returns_failed_proc_instead_of_raising():
    proc = run_pipeline.run_script(
        "stage_test", ["-c", "import time; time.sleep(30)"], timeout=1
    )
    assert isinstance(proc, subprocess.CompletedProcess)
    assert proc.returncode == 124
    assert "timed out after 1s" in proc.stderr
    # The stderr tail is what fix-loop findings and halt reasons quote, so
    # the message has to name the timeout in plain language.
    assert "error:" in proc.stderr


def test_normal_completion_is_unchanged():
    proc = run_pipeline.run_script(
        "stage_test", ["-c", "print('ok')"], timeout=30
    )
    assert proc.returncode == 0
    assert "ok" in proc.stdout


def test_failing_script_still_reports_its_own_stderr():
    proc = run_pipeline.run_script(
        "stage_test",
        ["-c", "import sys; sys.stderr.write('real error\\n'); sys.exit(2)"],
        timeout=30,
    )
    assert proc.returncode == 2
    assert "real error" in proc.stderr
