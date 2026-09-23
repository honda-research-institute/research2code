"""Delivered-baseline git repo (run-companion chat, queue item 28).

The baseline commit is the provenance anchor for post-delivery chat edits:
it must capture the delivered artifacts, exclude runtime side traffic, be
idempotent across re-finalization, and never raise (a git failure must
never block a delivery).
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import scripts.finalize_run_git as finalize_run_git
from scripts.finalize_run_git import ensure_delivered_baseline
from run_events import RunDirectoryLock


def _seed_run_dir(tmp_path: Path) -> Path:
    run_dir = tmp_path / "run"
    (run_dir / "method" / "__pycache__").mkdir(parents=True)
    (run_dir / "method" / "example_data" / "_cache").mkdir(parents=True)
    (run_dir / ".pipeline").mkdir()
    (run_dir / "REPORT.md").write_text("# Run Report\n")
    (run_dir / "method" / "method.py").write_text("def f():\n    return 1\n")
    (run_dir / "method" / "__pycache__" / "method.cpython-314.pyc").write_bytes(b"\x00")
    (run_dir / "method" / "example_data" / "_cache" / "data.pt").write_bytes(b"\x00")
    (run_dir / ".pipeline" / "paper_map.json").write_text("{}")
    return run_dir


def _tracked_files(run_dir: Path) -> set[str]:
    out = subprocess.run(
        ["git", "-C", str(run_dir), "ls-files"],
        capture_output=True, text=True, check=True,
    )
    return set(out.stdout.split())


def _status(run_dir: Path) -> str:
    return subprocess.run(
        ["git", "-C", str(run_dir), "status", "--porcelain"],
        capture_output=True, text=True, check=True,
    ).stdout


def test_module_supports_package_import_and_direct_cli():
    repo_root = Path(__file__).resolve().parent.parent
    package_import = subprocess.run(
        [sys.executable, "-c", "import scripts.finalize_run_git"],
        cwd=repo_root, capture_output=True, text=True,
    )
    direct_cli = subprocess.run(
        [sys.executable, "scripts/finalize_run_git.py", "--help"],
        cwd=repo_root, capture_output=True, text=True,
    )

    assert package_import.returncode == 0, package_import.stderr
    assert direct_cli.returncode == 0, direct_cli.stderr


def test_baseline_commits_artifacts_and_ignores_runtime_side_traffic(tmp_path):
    run_dir = _seed_run_dir(tmp_path)
    ok, msg = ensure_delivered_baseline(run_dir)
    assert ok, msg
    tracked = _tracked_files(run_dir)
    assert "REPORT.md" in tracked
    assert "method/method.py" in tracked
    assert ".pipeline/paper_map.json" in tracked
    assert not any("__pycache__" in t for t in tracked), tracked
    assert not any("_cache" in t for t in tracked), tracked


def test_no_change_repeat_keeps_same_clean_baseline(tmp_path):
    run_dir = _seed_run_dir(tmp_path)
    ok, msg = ensure_delivered_baseline(run_dir)
    assert ok, msg
    head = subprocess.run(
        ["git", "-C", str(run_dir), "rev-parse", "HEAD"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()

    ok, msg = ensure_delivered_baseline(run_dir)

    assert ok, msg
    assert "left untouched" in msg
    assert _status(run_dir) == ""
    assert subprocess.run(
        ["git", "-C", str(run_dir), "rev-parse", "HEAD"],
        capture_output=True, text=True, check=True,
    ).stdout.strip() == head


def test_baseline_is_idempotent_and_never_moves(tmp_path):
    run_dir = _seed_run_dir(tmp_path)
    ok, _ = ensure_delivered_baseline(run_dir)
    assert ok
    head = subprocess.run(
        ["git", "-C", str(run_dir), "rev-parse", "HEAD"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    # A later edit plus re-finalization must NOT fold into the baseline —
    # the baseline is delivery-time truth.
    (run_dir / "method" / "method.py").write_text("def f():\n    return 2\n")
    ok, msg = ensure_delivered_baseline(run_dir)
    assert ok
    assert "left untouched" in msg
    head_after = subprocess.run(
        ["git", "-C", str(run_dir), "rev-parse", "HEAD"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    assert head == head_after


def test_baseline_refuses_active_run_lock(tmp_path):
    run_dir = _seed_run_dir(tmp_path)
    lock_dir = run_dir / ".pipeline" / "_lock"
    lock_dir.mkdir()
    (lock_dir / "owner.json").write_text("{}\n", encoding="utf-8")

    ok, msg = ensure_delivered_baseline(run_dir)

    assert not ok
    assert "run lock is still active" in msg
    assert not (run_dir / ".git").exists()


def test_baseline_refuses_parseable_running_progress(tmp_path):
    run_dir = _seed_run_dir(tmp_path)
    (run_dir / ".pipeline" / "progress.json").write_text(
        '{"run_status": "running"}\n', encoding="utf-8",
    )

    ok, msg = ensure_delivered_baseline(run_dir)

    assert not ok
    assert "run progress is still running" in msg
    assert not (run_dir / ".git").exists()


def test_racing_new_invocation_cannot_enter_committed_snapshot(
        tmp_path, monkeypatch):
    run_dir = _seed_run_dir(tmp_path)
    progress_path = run_dir / ".pipeline" / "progress.json"
    progress_path.write_text(
        '{"run_status": "completed"}\n', encoding="utf-8",
    )
    contender = None

    class RacingBaselineLock(RunDirectoryLock):
        def release(self):
            nonlocal contender
            super().release()
            if contender is None:
                contender = RunDirectoryLock(
                    run_dir / ".pipeline", run_id="next-invocation",
                ).acquire()
                progress_path.write_text(
                    '{"run_status": "running"}\n', encoding="utf-8",
                )

    monkeypatch.setattr(
        finalize_run_git, "RunDirectoryLock", RacingBaselineLock,
    )

    ok, msg = ensure_delivered_baseline(run_dir)

    assert ok, msg
    committed_progress = subprocess.run(
        ["git", "-C", str(run_dir), "show", "HEAD:.pipeline/progress.json"],
        capture_output=True, text=True, check=True,
    ).stdout
    assert json.loads(committed_progress)["run_status"] == "completed"
    assert not any(path.startswith(".pipeline/_lock/")
                   for path in _tracked_files(run_dir))
    assert contender is not None
    contender.release()


def test_baseline_never_raises_on_missing_dir(tmp_path):
    ok, msg = ensure_delivered_baseline(tmp_path / "does-not-exist")
    assert not ok
    assert "no run dir" in msg
