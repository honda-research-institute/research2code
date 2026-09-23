#!/usr/bin/env python3
"""Delivered-baseline git repo for a run directory (run-companion chat).

After the driver records terminal state and releases the run lock, the run dir
becomes its own git repository with an "as delivered" commit. Every
post-delivery edit the researcher accepts through the run companion then lands
as one commit — diffable, revertable, and the provenance record for the
modified-after-delivery banner. See
the run companion chat design note (internal, not shipped).

Nested-repo safe: r2c_runs/ is ignored by the outer repository. Runtime
side traffic (bytecode, data caches) is excluded by a run-local .gitignore
so re-running the notebook never dirties the baseline.

Best-effort by contract: `ensure_delivered_baseline` never raises — a git
failure must never block or degrade a delivery. Callable lazily too (the
companion initializes older runs on first edit).

Usage:
    python scripts/finalize_run_git.py --run-dir r2c_runs/<slug>
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

try:
    from trusted import trusted_path
except ImportError:  # imported as scripts.finalize_run_git from the repo root
    from scripts.trusted import trusted_path

try:  # Package import (`import scripts.finalize_run_git`).
    from .run_events import RunDirectoryLock, RunDirectoryLockError
except ImportError:  # Direct CLI (`python scripts/finalize_run_git.py`).
    from run_events import RunDirectoryLock, RunDirectoryLockError

RUN_GITIGNORE = """\
__pycache__/
*.pyc
_cache/
.ipynb_checkpoints/
.pipeline/_lock/
.pipeline/run_events.jsonl.lock
"""

_TRANSIENT_EXCLUDES = (
    ".pipeline/_lock/",
    ".pipeline/run_events.jsonl.lock",
)

_GIT_IDENTITY = [
    "-c", "user.name=r2c-delivery",
    "-c", "user.email=r2c-delivery@local",
]


def _git(run_dir: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", "-C", str(trusted_path(run_dir)), *_GIT_IDENTITY, *args],
        capture_output=True, text=True, timeout=120,
    )


def _has_commit(run_dir: Path) -> bool:
    return _git(run_dir, "rev-parse", "--verify", "HEAD").returncode == 0


def _running_progress_reason(run_dir: Path) -> str | None:
    """Return why a first delivery snapshot would capture running state.

    The helper owns the run-directory lock while it checks and stages, so a new
    invocation cannot change progress between this read and ``git add``. A
    progress-write failure that leaves the last parseable state as running is
    refused instead of becoming delivery provenance.
    """
    progress_path = run_dir / ".pipeline" / "progress.json"
    try:
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return None
    if isinstance(progress, dict) and progress.get("run_status") == "running":
        return f"run progress is still running at {progress_path}"
    return None


def _ensure_transient_excludes(run_dir: Path) -> None:
    """Exclude authoritative/event locks even with a custom gitignore."""
    exclude_path = run_dir / ".git" / "info" / "exclude"
    existing = (
        exclude_path.read_text(encoding="utf-8")
        if exclude_path.is_file() else ""
    )
    existing_lines = set(existing.splitlines())
    missing = [
        pattern for pattern in _TRANSIENT_EXCLUDES
        if pattern not in existing_lines
    ]
    if not missing:
        return
    exclude_path.parent.mkdir(parents=True, exist_ok=True)
    separator = "" if not existing or existing.endswith("\n") else "\n"
    with exclude_path.open("a", encoding="utf-8") as handle:
        handle.write(separator + "\n".join(missing) + "\n")


def ensure_delivered_baseline(
    run_dir: Path, *, message: str = "as delivered",
) -> tuple[bool, str]:
    """Idempotently ensure `run_dir` is a git repo with a baseline commit.

    Returns (ok, message). Never raises. A repo that already has a HEAD
    commit is left untouched (the baseline is delivery-time truth; later
    commits belong to the companion, not to re-finalization)."""
    try:
        run_dir = Path(run_dir)
        if not run_dir.is_dir():
            return False, f"no run dir at {run_dir}"
        baseline_lock = RunDirectoryLock(
            run_dir / ".pipeline",
            run_id=f"{run_dir.name}:delivery-baseline",
        )
        try:
            baseline_lock.acquire()
        except RunDirectoryLockError:
            return False, (
                "delivery baseline refused: run lock is still active at "
                f"{baseline_lock.path}"
            )

        # Freeze the delivery index while the authoritative lock prevents a
        # new invocation from mutating terminal state. The commit itself comes
        # only after this temporary lock is physically removed, satisfying the
        # no-active-lock invariant without a check/add race.
        try:
            running_reason = _running_progress_reason(run_dir)
            if running_reason:
                return False, f"delivery baseline refused: {running_reason}"
            if not (run_dir / ".git").exists():
                init = _git(run_dir, "init", "--quiet")
                if init.returncode != 0:
                    return False, f"git init failed: {init.stderr.strip()[:200]}"
            _ensure_transient_excludes(run_dir)
            gitignore = run_dir / ".gitignore"
            if not gitignore.is_file():
                gitignore.write_text(RUN_GITIGNORE, encoding="utf-8")
            if _has_commit(run_dir):
                return True, "delivered baseline already exists; left untouched"
            add = _git(run_dir, "add", "-A")
            if add.returncode != 0:
                return False, f"git add failed: {add.stderr.strip()[:200]}"
        finally:
            baseline_lock.release()

        commit = _git(run_dir, "commit", "--quiet", "-m", message)
        if commit.returncode != 0:
            return False, f"git commit failed: {commit.stderr.strip()[:200]}"
        return True, f"delivered baseline committed ({message!r})"
    except Exception as e:  # noqa: BLE001 — best-effort by contract
        return False, f"{type(e).__name__}: {e}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    args = parser.parse_args()
    ok, msg = ensure_delivered_baseline(args.run_dir)
    print(msg)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
