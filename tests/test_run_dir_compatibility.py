"""Item 29 — stale run-dir collision auto-archive.

A researcher's first solo run (07-08) failed because a `pdwa/` dir from a previous
version of R2C sat under r2c_runs/; he had to delete it by hand. The fix keeps
same-dir resume (load-bearing: the documented halt-recovery path) but only
reuses a dir the current driver understands, and moves an older-version dir
aside — never deletes it — so the run proceeds fresh.

These exercise the decision logic directly:
  - progress_state.is_run_dir_layout_compatible  (resume vs incompatible)
  - run_pipeline._archive_incompatible_run_dir    (move-aside vs leave-alone)
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
from pathlib import Path

import pytest

import fleet_state
import progress_state
import run_pipeline
from progress_state import PROGRESS_SCHEMA_VERSION, is_run_dir_layout_compatible
from run_events import RunDirectoryLockError, run_lock_path


def _make_run_dir(root: Path, slug: str, *, schema_version: str | None) -> Path:
    """A run dir with a .pipeline/progress.json at the given schema version.
    schema_version=None writes no progress.json (an old-version / foreign dir)."""
    run_dir = root / slug
    pipeline = run_dir / ".pipeline"
    pipeline.mkdir(parents=True)
    if schema_version is not None:
        (pipeline / "progress.json").write_text(
            json.dumps({"schema_version": schema_version, "slug": slug, "stages": []}),
            encoding="utf-8",
        )
    return run_dir


def _make_old_layout_dir(root: Path, slug: str) -> Path:
    """A pre-consolidation delivery dir with no .pipeline/progress.json — the
    shape of the gbald/RACIL dirs and of the researcher's stale pdwa."""
    run_dir = root / slug
    (run_dir / "method").mkdir(parents=True)
    (run_dir / "REPORT.md").write_text("# old delivery\n", encoding="utf-8")
    (run_dir / "method" / "method.py").write_text("# old code\n", encoding="utf-8")
    return run_dir


# --- is_run_dir_layout_compatible -----------------------------------------


def test_current_progress_is_compatible(tmp_path):
    run_dir = _make_run_dir(tmp_path, "paper", schema_version=PROGRESS_SCHEMA_VERSION)
    assert is_run_dir_layout_compatible(run_dir) is True


def test_missing_progress_is_incompatible(tmp_path):
    run_dir = _make_run_dir(tmp_path, "paper", schema_version=None)
    assert is_run_dir_layout_compatible(run_dir) is False


def test_old_layout_no_pipeline_is_incompatible(tmp_path):
    run_dir = _make_old_layout_dir(tmp_path, "paper")
    assert is_run_dir_layout_compatible(run_dir) is False


def test_stale_schema_is_incompatible(tmp_path):
    run_dir = _make_run_dir(tmp_path, "paper", schema_version="0.9")
    assert is_run_dir_layout_compatible(run_dir) is False


def test_corrupt_progress_is_incompatible(tmp_path):
    run_dir = _make_run_dir(tmp_path, "paper", schema_version=PROGRESS_SCHEMA_VERSION)
    (run_dir / ".pipeline" / "progress.json").write_text("not json {", encoding="utf-8")
    assert is_run_dir_layout_compatible(run_dir) is False


def test_absent_dir_is_incompatible(tmp_path):
    assert is_run_dir_layout_compatible(tmp_path / "never-created") is False


# --- _archive_incompatible_run_dir -----------------------------------------


def test_archive_noop_when_dir_absent(tmp_path):
    result = run_pipeline._archive_incompatible_run_dir(tmp_path / "paper")
    assert result is None


def test_archive_leaves_compatible_dir_untouched(tmp_path):
    """A current-layout dir (halted or delivered) still resumes in place —
    the move-aside must never touch it."""
    run_dir = _make_run_dir(tmp_path, "paper", schema_version=PROGRESS_SCHEMA_VERSION)
    result = run_pipeline._archive_incompatible_run_dir(run_dir, today="2026-07-09")
    assert result is None
    assert run_dir.exists()
    assert (run_dir / ".pipeline" / "progress.json").is_file()


def test_archive_moves_old_layout_dir_and_preserves_content(tmp_path):
    run_dir = _make_old_layout_dir(tmp_path, "pdwa")
    result = run_pipeline._archive_incompatible_run_dir(run_dir, today="2026-07-09")

    assert result == tmp_path / "pdwa_pre-upgrade-2026-07-09"
    assert result.is_dir()
    # The run dir path is now free for a fresh run.
    assert not run_dir.exists()
    # Nothing deleted: every file moved into the archive verbatim.
    assert (result / "REPORT.md").read_text(encoding="utf-8") == "# old delivery\n"
    assert (result / "method" / "method.py").read_text(encoding="utf-8") == "# old code\n"


def test_archive_collision_uses_numeric_suffix_never_deletes(tmp_path):
    """Two archives on the same day (or a stray pre-existing archive) must
    coexist — the second gets -2, the first is untouched."""
    existing = tmp_path / "pdwa_pre-upgrade-2026-07-09"
    existing.mkdir()
    (existing / "sentinel.txt").write_text("prior archive\n", encoding="utf-8")

    run_dir = _make_old_layout_dir(tmp_path, "pdwa")
    result = run_pipeline._archive_incompatible_run_dir(run_dir, today="2026-07-09")

    assert result == tmp_path / "pdwa_pre-upgrade-2026-07-09-2"
    assert result.is_dir()
    # The prior archive survives intact — nothing is ever deleted.
    assert (existing / "sentinel.txt").read_text(encoding="utf-8") == "prior archive\n"
    assert not run_dir.exists()


def test_archive_default_stamp_is_today_iso(tmp_path):
    """Without an injected date the archive name carries today's UTC date."""
    run_dir = _make_old_layout_dir(tmp_path, "paper")
    result = run_pipeline._archive_incompatible_run_dir(run_dir)
    assert result is not None
    stamp = result.name.rsplit("_pre-upgrade-", 1)[1]
    assert stamp == progress_state.utc_now_iso()[:10]  # YYYY-MM-DD


# --- _archive_for_fresh_roll (phase 2: the deliberate /r2c-run --fresh) -----


def test_fresh_archive_noop_when_dir_absent(tmp_path):
    assert run_pipeline._archive_for_fresh_roll(tmp_path / "paper") is None


def test_fresh_archives_current_layout_delivery(tmp_path):
    """Unlike the incompatible-archive, --fresh moves aside a CURRENT-layout dir
    too — it is the researcher's deliberate 'start over, keep the old one'."""
    run_dir = _make_run_dir(tmp_path, "paper", schema_version=PROGRESS_SCHEMA_VERSION)
    (run_dir / "REPORT.md").write_text("# prior delivery\n", encoding="utf-8")
    result = run_pipeline._archive_for_fresh_roll(run_dir)
    assert result == tmp_path / "paper_1"
    assert not run_dir.exists()
    assert (result / "REPORT.md").read_text(encoding="utf-8") == "# prior delivery\n"


def test_fresh_archives_old_layout_too(tmp_path):
    """A fresh roll archives regardless of layout — no compatibility check."""
    run_dir = _make_old_layout_dir(tmp_path, "paper")
    result = run_pipeline._archive_for_fresh_roll(run_dir)
    assert result == tmp_path / "paper_1"
    assert (result / "method" / "method.py").is_file()


def test_fresh_suffix_increments_never_deletes(tmp_path):
    """Repeated fresh rolls stack: _1, _2, ... and every prior roll survives —
    the numeric suffix exists only because a researcher chose a fresh roll."""
    (tmp_path / "paper_1").mkdir()
    (tmp_path / "paper_1" / "keep.txt").write_text("first roll\n", encoding="utf-8")
    run_dir = _make_run_dir(tmp_path, "paper", schema_version=PROGRESS_SCHEMA_VERSION)
    result = run_pipeline._archive_for_fresh_roll(run_dir)
    assert result == tmp_path / "paper_2"
    assert (tmp_path / "paper_1" / "keep.txt").read_text(encoding="utf-8") == "first roll\n"


# --- run_stage_0 --fresh wiring ---------------------------------------------


def _md_setup(repo: Path, run_dir: Path) -> dict:
    return {
        "repo_root": str(repo),
        "input_path": str(repo / "input_papers" / "sample.md"),
        "input_kind": "markdown",
        "slug": "sample",
        "run_dir": str(run_dir),
        "pipeline_dir": str(run_dir / ".pipeline"),
        "paper_md_path": str(run_dir / ".pipeline" / "paper.md"),
        "paper_md_present": True,
    }


def _completed(args, stdout="", stderr="", code=0):
    return subprocess.CompletedProcess(args, code, stdout=stdout, stderr=stderr)


def test_run_stage0_fresh_archives_current_delivery(tmp_path, monkeypatch):
    """End to end: --fresh moves a current-layout delivery aside to <slug>_1 and
    rolls clean, where the default (resume) reuses it in place."""
    repo = tmp_path
    (repo / "input_papers").mkdir()
    (repo / "input_papers" / "sample.md").write_text("# paper\n", encoding="utf-8")
    runs = repo / "r2c_runs"
    run_dir = _make_run_dir(runs, "sample", schema_version=PROGRESS_SCHEMA_VERSION)
    (run_dir / "REPORT.md").write_text("# prior delivery\n", encoding="utf-8")
    setup = _md_setup(repo, run_dir)

    def fake_run_script(stage_id, args, *, timeout=300):
        if args[0] == "scripts/setup_pipeline_dirs.py":
            # Mimic materialization: recreate the fresh dir + paper.md.
            (run_dir / ".pipeline").mkdir(parents=True, exist_ok=True)
            (run_dir / ".pipeline" / "paper.md").write_text(
                " ".join(["method"] * 20), encoding="utf-8")
            return _completed(args, stdout=json.dumps(setup))
        raise AssertionError(args)

    monkeypatch.setattr(run_pipeline, "run_script", fake_run_script)

    paths, result = run_pipeline.run_stage_0(
        "sample", workspace=str(repo), fresh=True,
        before_parse=lambda p: p.pipeline_dir.mkdir(parents=True, exist_ok=True),
    )

    assert result.status == "completed"
    # The prior delivery is archived to sample_1, content intact, nothing deleted.
    assert (runs / "sample_1" / "REPORT.md").read_text(encoding="utf-8") == "# prior delivery\n"
    # A run_dir_archived event fired with the fresh-roll reason.
    events = (run_dir / ".pipeline" / "run_events.jsonl").read_text(encoding="utf-8")
    assert '"reason":"fresh_flag"' in events


def test_run_stage0_fresh_refused_when_run_is_live(tmp_path, monkeypatch):
    """--fresh must never yank a dir a live run holds: with a lock present it
    skips the archive and lets the lock acquisition refuse the launch."""
    repo = tmp_path
    (repo / "input_papers").mkdir()
    (repo / "input_papers" / "sample.md").write_text("# paper\n", encoding="utf-8")
    runs = repo / "r2c_runs"
    run_dir = _make_run_dir(runs, "sample", schema_version=PROGRESS_SCHEMA_VERSION)
    # A live run holds the lock.
    run_lock_path(run_dir / ".pipeline").mkdir(parents=True)
    setup = _md_setup(repo, run_dir)

    def fake_run_script(stage_id, args, *, timeout=300):
        if args[0] == "scripts/setup_pipeline_dirs.py":
            return _completed(args, stdout=json.dumps(setup))
        raise AssertionError(args)

    monkeypatch.setattr(run_pipeline, "run_script", fake_run_script)

    def locked(_paths):
        raise RunDirectoryLockError("run directory is already locked")

    with pytest.raises(RunDirectoryLockError):
        run_pipeline.run_stage_0(
            "sample", workspace=str(repo), fresh=True, before_parse=locked,
        )

    # The live dir was NOT archived — no sample_1 exists and it is intact.
    assert not (runs / "sample_1").exists()
    assert (run_dir / ".pipeline" / "progress.json").is_file()


def _dead_pid() -> int:
    """A pid that is not running. Forked, reaped, and confirmed gone."""
    pid = os.fork()
    if pid == 0:  # pragma: no cover - the child never returns
        os._exit(0)
    os.waitpid(pid, 0)
    return pid


def _plant_owned_lock(pipeline: Path, *, pid: int, host: str | None = None) -> None:
    lock = run_lock_path(pipeline)
    lock.mkdir(parents=True, exist_ok=True)
    (lock / "owner.json").write_text(json.dumps({
        "schema_version": "1.0",
        "run_id": "sample",
        "pid": pid,
        "host": host if host is not None else socket.gethostname(),
        "created_at": "2026-08-06T21:22:05Z",
    }), encoding="utf-8")


def _fresh_roll_against_lock(tmp_path, monkeypatch, *, pid, host=None):
    """Drive run_stage_0 --fresh over a run dir whose lock is already held."""
    repo = tmp_path
    (repo / "input_papers").mkdir()
    (repo / "input_papers" / "sample.md").write_text("# paper\n", encoding="utf-8")
    runs = repo / "r2c_runs"
    run_dir = _make_run_dir(runs, "sample", schema_version=PROGRESS_SCHEMA_VERSION)
    (run_dir / "REPORT.md").write_text("# prior delivery\n", encoding="utf-8")
    _plant_owned_lock(run_dir / ".pipeline", pid=pid, host=host)
    setup = _md_setup(repo, run_dir)

    def fake_run_script(stage_id, args, *, timeout=300):
        if args[0] == "scripts/setup_pipeline_dirs.py":
            (run_dir / ".pipeline").mkdir(parents=True, exist_ok=True)
            (run_dir / ".pipeline" / "paper.md").write_text(
                " ".join(["method"] * 20), encoding="utf-8")
            return _completed(args, stdout=json.dumps(setup))
        raise AssertionError(args)

    monkeypatch.setattr(run_pipeline, "run_script", fake_run_script)
    return repo, runs, run_dir, setup


def test_run_stage0_fresh_archives_when_the_lock_owner_is_dead(tmp_path, monkeypatch):
    """A --fresh roll over a DEAD run's lock must still archive (R2C-076 follow-up).

    The known-bad this guards is worse than a refusal. --fresh asked only
    whether a lock directory existed, so a lock left behind by a killed roll
    made it skip the archive on the theory that the acquisition below would
    refuse the launch. Once the acquisition learned to reclaim a provably-dead
    lock, that theory stopped holding: the roll proceeded, unarchived, writing
    into the previous roll's directory on top of its stale stage-complete
    sentinels. A researcher who asked for a clean roll got a dirty resume, and
    nothing said so.
    """
    repo, runs, run_dir, _ = _fresh_roll_against_lock(
        tmp_path, monkeypatch, pid=_dead_pid())

    paths, result = run_pipeline.run_stage_0(
        "sample", workspace=str(repo), fresh=True,
        before_parse=lambda p: p.pipeline_dir.mkdir(parents=True, exist_ok=True),
    )

    assert result.status == "completed"
    # The dead roll's delivery is archived intact, and the fresh dir is clean.
    assert (runs / "sample_1" / "REPORT.md").read_text(encoding="utf-8") == "# prior delivery\n"
    assert not (run_dir / "REPORT.md").exists()
    events = (run_dir / ".pipeline" / "run_events.jsonl").read_text(encoding="utf-8")
    assert '"reason":"fresh_flag"' in events


def test_run_stage0_fresh_refused_when_the_lock_owner_is_alive(tmp_path, monkeypatch):
    """The counterpart: a lock whose owner is running still blocks the archive."""
    repo, runs, run_dir, _ = _fresh_roll_against_lock(
        tmp_path, monkeypatch, pid=os.getpid())

    def locked(_paths):
        raise RunDirectoryLockError("run directory is already locked")

    with pytest.raises(RunDirectoryLockError):
        run_pipeline.run_stage_0(
            "sample", workspace=str(repo), fresh=True, before_parse=locked,
        )

    assert not (runs / "sample_1").exists()
    assert (run_dir / "REPORT.md").is_file()


def test_run_stage0_fresh_refused_when_the_lock_owner_is_on_another_host(
        tmp_path, monkeypatch):
    """A pid on another machine says nothing about liveness here, so --fresh
    must keep its hands off the directory."""
    repo, runs, run_dir, _ = _fresh_roll_against_lock(
        tmp_path, monkeypatch, pid=_dead_pid(), host="some-other-machine")

    def locked(_paths):
        raise RunDirectoryLockError("run directory is already locked")

    with pytest.raises(RunDirectoryLockError):
        run_pipeline.run_stage_0(
            "sample", workspace=str(repo), fresh=True, before_parse=locked,
        )

    assert not (runs / "sample_1").exists()
    assert (run_dir / "REPORT.md").is_file()


# --- researcher-surface wiring ---------------------------------------------


def test_run_dir_archived_event_is_categorized(tmp_path):
    """The move-aside event must land in the behavioral timeline as a driver
    decision, not fall through to the uncategorized 'other' bucket."""
    assert fleet_state.EVENT_CATEGORIES["run_dir_archived"] == "decision"
