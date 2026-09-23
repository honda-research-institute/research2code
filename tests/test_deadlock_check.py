"""Regression tests for `_detect_session_deadlock_risk` in run_pipeline.

The detection's purpose: refuse to launch the driver synchronously inside
an opencode bash tool, where the driver's own producer-agent POSTs would
queue behind the bash tool and deadlock forever.

Each test mocks the underlying OS probes (env vars, isatty, ps subprocess,
fstat) to drive the three signals (TTY presence, opencode ancestor,
stdout type) independently. See README.md ("How it works") for the production scenario
and the documented fix (`/r2c-run`).
"""

from __future__ import annotations

import os
import stat
import subprocess
import sys


def _import_target():
    """Return the detection function under test."""
    from run_pipeline import _detect_session_deadlock_risk  # noqa: PLC0415
    return _detect_session_deadlock_risk


# ---------------------------------------------------------------------------
# Bypasses
# ---------------------------------------------------------------------------


def test_env_var_bypass_short_circuits(monkeypatch):
    """R2C_SKIP_DEADLOCK_CHECK=1 → no risk regardless of other signals."""
    fn = _import_target()
    monkeypatch.setenv("R2C_SKIP_DEADLOCK_CHECK", "1")
    # Even if the other probes would say "yes deadlock", env var wins
    monkeypatch.setattr(os, "isatty", lambda fd: False)
    is_risk, diag = fn()
    assert is_risk is False
    assert "R2C_SKIP_DEADLOCK_CHECK" in diag


def test_tty_on_stdout_bypasses_check(monkeypatch):
    """If stdout (or any standard stream) is a TTY → interactive run; no risk."""
    fn = _import_target()
    monkeypatch.delenv("R2C_SKIP_DEADLOCK_CHECK", raising=False)
    # Pretend stdout is a TTY (fd=1)
    monkeypatch.setattr(os, "isatty", lambda fd: fd == 1)
    is_risk, diag = fn()
    assert is_risk is False
    assert "tty" in diag.lower()


# ---------------------------------------------------------------------------
# Ancestor-chain check
# ---------------------------------------------------------------------------


def test_no_opencode_ancestor_returns_false(monkeypatch):
    """If the ancestor chain has no opencode process, no risk (even with no TTY)."""
    fn = _import_target()
    monkeypatch.delenv("R2C_SKIP_DEADLOCK_CHECK", raising=False)
    monkeypatch.setattr(os, "isatty", lambda fd: False)
    monkeypatch.setattr(os, "getppid", lambda: 1000)

    # Mock ps so every parent is a generic shell
    def fake_run(*a, **kw):
        return subprocess.CompletedProcess(
            args=a[0], returncode=0, stdout="bash 1\n", stderr="",
        )
    monkeypatch.setattr(subprocess, "run", fake_run)

    is_risk, diag = fn()
    assert is_risk is False
    assert "no opencode" in diag


def test_opencode_ancestor_plus_no_tty_plus_pipe_stdout_is_risk(monkeypatch):
    """All three signals match → deadlock detected."""
    fn = _import_target()
    monkeypatch.delenv("R2C_SKIP_DEADLOCK_CHECK", raising=False)
    monkeypatch.setattr(os, "isatty", lambda fd: False)
    monkeypatch.setattr(os, "getppid", lambda: 1000)

    # Simulate ps: parent (pid=1000) is bash, grandparent (pid=500) is opencode
    def fake_run(*a, **kw):
        cmd = a[0]
        # cmd is like ["ps", "-p", "<pid>", "-o", "comm=,ppid="]
        target_pid = cmd[2]
        if target_pid == "1000":
            return subprocess.CompletedProcess(
                args=cmd, returncode=0, stdout="bash 500\n", stderr="",
            )
        elif target_pid == "500":
            return subprocess.CompletedProcess(
                args=cmd, returncode=0, stdout="opencode 1\n", stderr="",
            )
        return subprocess.CompletedProcess(args=cmd, returncode=1, stdout="", stderr="")
    monkeypatch.setattr(subprocess, "run", fake_run)

    # Simulate stdout = pipe (S_ISFIFO true)
    class _FakeStat:
        st_mode = stat.S_IFIFO | 0o600
        st_rdev = 0
    monkeypatch.setattr(os, "fstat", lambda fd: _FakeStat())

    is_risk, diag = fn()
    assert is_risk is True
    assert "opencode ancestor" in diag


def test_opencode_ancestor_but_stdout_is_regular_file_returns_false(monkeypatch):
    """nohup-style launch: opencode is in ancestor chain (the bash that ran
    `nohup ... &`), but stdout is a regular file (the log redirect). Safe."""
    fn = _import_target()
    monkeypatch.delenv("R2C_SKIP_DEADLOCK_CHECK", raising=False)
    monkeypatch.setattr(os, "isatty", lambda fd: False)
    monkeypatch.setattr(os, "getppid", lambda: 1000)

    def fake_run(*a, **kw):
        cmd = a[0]
        target_pid = cmd[2]
        if target_pid == "1000":
            return subprocess.CompletedProcess(
                args=cmd, returncode=0, stdout="opencode 1\n", stderr="",
            )
        return subprocess.CompletedProcess(args=cmd, returncode=1, stdout="", stderr="")
    monkeypatch.setattr(subprocess, "run", fake_run)

    # Simulate stdout = regular file
    class _FakeStat:
        st_mode = stat.S_IFREG | 0o644
        st_rdev = 0
    monkeypatch.setattr(os, "fstat", lambda fd: _FakeStat())

    is_risk, diag = fn()
    assert is_risk is False
    assert "regular file" in diag


def test_opencode_ancestor_but_stdout_is_devnull_returns_false(monkeypatch, tmp_path):
    """If stdout is /dev/null (alternate nohup pattern), still safe."""
    fn = _import_target()
    monkeypatch.delenv("R2C_SKIP_DEADLOCK_CHECK", raising=False)
    monkeypatch.setattr(os, "isatty", lambda fd: False)
    monkeypatch.setattr(os, "getppid", lambda: 1000)

    def fake_run(*a, **kw):
        cmd = a[0]
        target_pid = cmd[2]
        if target_pid == "1000":
            return subprocess.CompletedProcess(
                args=cmd, returncode=0, stdout="opencode 1\n", stderr="",
            )
        return subprocess.CompletedProcess(args=cmd, returncode=1, stdout="", stderr="")
    monkeypatch.setattr(subprocess, "run", fake_run)

    # Get the actual /dev/null st_rdev so we match correctly
    devnull_st = os.stat(os.devnull)

    class _FakeStat:
        st_mode = stat.S_IFCHR | 0o666
        st_rdev = devnull_st.st_rdev
    monkeypatch.setattr(os, "fstat", lambda fd: _FakeStat())

    is_risk, diag = fn()
    assert is_risk is False
    assert "/dev/null" in diag


# ---------------------------------------------------------------------------
# Probe failures fall through safely (open-fail rather than block legit runs)
# ---------------------------------------------------------------------------


def test_ps_probe_failure_returns_false(monkeypatch):
    """If `ps` is missing or fails, the check falls through to False — we'd
    rather miss a deadlock case than block a legitimate run."""
    fn = _import_target()
    monkeypatch.delenv("R2C_SKIP_DEADLOCK_CHECK", raising=False)
    monkeypatch.setattr(os, "isatty", lambda fd: False)
    monkeypatch.setattr(os, "getppid", lambda: 1000)

    def fake_run(*a, **kw):
        raise FileNotFoundError("ps not found")
    monkeypatch.setattr(subprocess, "run", fake_run)

    is_risk, diag = fn()
    assert is_risk is False
    assert "ps probe failed" in diag


def test_retired_self_test_flag_exits_loudly_without_deadlock_check():
    """B-07 step 3 repoint: `--self-test` is retired (the checks live in
    tests/test_driver_selftest.py). The flag must exit NONZERO with the
    moved message — never print nothing and exit 0 (the vacuous-green
    trap) — and must still not invoke the deadlock check, which only the
    `--paper` path runs."""
    repo = __import__("pathlib").Path(__file__).resolve().parent.parent
    proc = subprocess.run(
        [sys.executable, "scripts/run_pipeline.py", "--self-test"],
        cwd=str(repo), capture_output=True, text=True, timeout=30,
    )
    assert proc.returncode != 0, "retired flag must exit nonzero"
    combined = proc.stdout + proc.stderr
    assert "tests/test_driver_selftest.py" in combined, (
        f"moved message must name the new home.\nstdout: {proc.stdout[-500:]}\n"
        f"stderr: {proc.stderr[-500:]}")
    assert "deadlock" not in combined.lower(), (
        "the retired flag must exit before the deadlock check runs")
