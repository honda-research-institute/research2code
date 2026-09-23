"""Repo-scope dispatch write boundary (repo_scope_boundary.py + the
_dispatch_with_scope_check wiring).

The run-dir ownership scan only walks the active run dir, so two recorded
escapes were silent: an agent wrote a hallucinated sibling of the runs root
(`<runs-sibling>/<slug>/.pipeline/method_spec.json`), and a halt judge wrote
an audit-style report to a fresh repo-root directory. The boundary diffs
git-porcelain state plus the top-level listings of the repo root and the
runs root around every dispatch, quarantines NEW strays into
`<run_dir>/.pipeline/quarantine/<dispatch-label>/`, and raises the existing
OutOfScopeWritesError so halt-class resolution engages unchanged.

All network collaborators are patched; nothing here dispatches anywhere.
Tests that need the porcelain layer build a real (temp) git repo and skip
when git is not installed.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

import repo_scope_boundary
import run_pipeline
from opencode_client import DispatchResult
from run_pipeline import OutOfScopeWritesError, _dispatch_with_scope_check
from tests.helpers.state import make_state

needs_git = pytest.mark.skipif(
    shutil.which("git") is None, reason="git not installed")

OUT = [".pipeline/out.json"]


def _result() -> DispatchResult:
    return DispatchResult(
        session_id="fake-session", user_message_id="u1",
        assistant_message_id="a1", completed=True, elapsed_s=0.1,
    )


def _events(state) -> list[dict]:
    path = state.paths.pipeline_dir / "run_events.jsonl"
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text().splitlines()]


def _repo_scope_events(state) -> list[dict]:
    return [e for e in _events(state)
            if e["event_type"] == "repo_scope_violation_detected"]


def _git(repo: Path, *args: str) -> None:
    subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True, capture_output=True, text=True)


def _make_repo(tmp_path: Path, *, git: bool = True):
    """A generic repo layout mirroring production: a runs root with one
    active run dir, a tracked source tree, production-shaped .gitignore
    (runs root + scratch valve are invisible to git)."""
    repo = tmp_path / "repo"
    run_dir = repo / "r2c_runs" / "active-paper"
    run_dir.mkdir(parents=True)
    (repo / "scripts").mkdir()
    (repo / "scripts" / "tracked.py").write_text("x = 1\n", encoding="utf-8")
    (repo / ".gitignore").write_text(
        "r2c_runs/\n.opencode/scratch/\n__pycache__/\n*.pyc\n",
        encoding="utf-8")
    (repo / ".opencode" / "scratch").mkdir(parents=True)
    (repo / ".opencode" / "agents").mkdir()
    (repo / ".opencode" / "agents" / "guard.md").write_text(
        "tracked\n", encoding="utf-8")
    if git:
        _git(repo, "init", "-q")
        _git(repo, "config", "user.email", "t@example.com")
        _git(repo, "config", "user.name", "T")
        _git(repo, "add", "-A")
        _git(repo, "commit", "-q", "-m", "init")
    state = make_state(run_dir, slug="active-paper")
    state.paths.repo_root = repo  # make_state guesses run_dir.parent
    return state, repo


def _dispatch(state, fake, monkeypatch, agent="r2c-test-producer"):
    monkeypatch.setattr(run_pipeline, "dispatch_agent", fake)
    return _dispatch_with_scope_check(
        state=state, agent=agent, prompt="p", timeout_s=5,
        writeable_paths_override=OUT)


def _write_out(state) -> None:
    (state.paths.run_dir / ".pipeline" / "out.json").write_text(
        "{}", encoding="utf-8")


# ---------------------------------------------------------------------------
# Recorded case 1: a hallucinated sibling of the runs root
# ---------------------------------------------------------------------------


@needs_git
def test_stray_sibling_of_runs_root_is_quarantined_and_raises(
        tmp_path, monkeypatch):
    state, repo = _make_repo(tmp_path)

    def fake(*, state, agent, prompt, timeout_s):
        _write_out(state)
        stray = repo / "runs_shadow" / "active-paper" / ".pipeline"
        stray.mkdir(parents=True)
        (stray / "method_spec.json").write_text("{}", encoding="utf-8")
        return _result()

    with pytest.raises(OutOfScopeWritesError) as exc:
        _dispatch(state, fake, monkeypatch, agent="r2c-method-analyzer")

    msg = str(exc.value)
    assert "runs_shadow" in msg and "quarantined" in msg
    # Repo left clean; the whole tree moved into the run dir as evidence.
    assert not (repo / "runs_shadow").exists()
    qroot = state.paths.pipeline_dir / "quarantine"
    moved = list(qroot.rglob("method_spec.json"))
    assert len(moved) == 1
    assert "runs_shadow/active-paper/.pipeline" in moved[0].as_posix()
    # The existing halt-class routing engages unchanged.
    assert run_pipeline._dispatch_error_halt_class(exc.value) == \
        "out_of_scope_write"
    events = _repo_scope_events(state)
    assert len(events) == 1
    details = events[0]["details"]
    assert details["agent"] == "r2c-method-analyzer"
    assert "runs_shadow" in details["new_repo_top_entries"]
    assert details["quarantined"]["runs_shadow"].startswith(
        ".pipeline/quarantine/r2c-method-analyzer-")


# ---------------------------------------------------------------------------
# Recorded case 2: an audit-style report at the repo root
# ---------------------------------------------------------------------------


@needs_git
def test_repo_root_untracked_file_is_quarantined_and_raises(
        tmp_path, monkeypatch):
    state, repo = _make_repo(tmp_path)

    def fake(*, state, agent, prompt, timeout_s):
        _write_out(state)
        reports = repo / "review_reports"
        reports.mkdir()
        (reports / "halt_report.md").write_text("# report\n", encoding="utf-8")
        return _result()

    with pytest.raises(OutOfScopeWritesError) as exc:
        _dispatch(state, fake, monkeypatch, agent="r2c-halt-judge")

    assert "review_reports" in str(exc.value)
    assert not (repo / "review_reports").exists()
    moved = list((state.paths.pipeline_dir / "quarantine").rglob(
        "halt_report.md"))
    assert len(moved) == 1
    assert _repo_scope_events(state)


# ---------------------------------------------------------------------------
# Tracked files: listed, never restored (halt-and-investigate)
# ---------------------------------------------------------------------------


@needs_git
def test_modified_tracked_file_raises_without_restore(tmp_path, monkeypatch):
    state, repo = _make_repo(tmp_path)
    tracked = repo / "scripts" / "tracked.py"

    def fake(*, state, agent, prompt, timeout_s):
        _write_out(state)
        tracked.write_text("x = 2  # clobbered\n", encoding="utf-8")
        return _result()

    with pytest.raises(OutOfScopeWritesError) as exc:
        _dispatch(state, fake, monkeypatch)

    assert "scripts/tracked.py" in str(exc.value)
    assert "left in place" in str(exc.value)
    # Halt-and-investigate: the file keeps the agent's bytes, nothing moved.
    assert tracked.read_text(encoding="utf-8") == "x = 2  # clobbered\n"
    details = _repo_scope_events(state)[0]["details"]
    assert details["modified_tracked"] == ["scripts/tracked.py"]
    assert "scripts/tracked.py" not in details["quarantined"]


@needs_git
def test_deleted_tracked_file_raises_without_restore(tmp_path, monkeypatch):
    state, repo = _make_repo(tmp_path)
    tracked = repo / "scripts" / "tracked.py"

    def fake(*, state, agent, prompt, timeout_s):
        _write_out(state)
        tracked.unlink()
        return _result()

    with pytest.raises(OutOfScopeWritesError) as exc:
        _dispatch(state, fake, monkeypatch)

    assert "not restored" in str(exc.value)
    assert not tracked.exists()
    details = _repo_scope_events(state)[0]["details"]
    assert details["deleted_tracked"] == ["scripts/tracked.py"]


# ---------------------------------------------------------------------------
# Listdir layer: gitignore-invisible strays under the runs root
# ---------------------------------------------------------------------------


@needs_git
def test_foreign_run_slug_dir_caught_despite_gitignore(tmp_path, monkeypatch):
    state, repo = _make_repo(tmp_path)

    def fake(*, state, agent, prompt, timeout_s):
        _write_out(state)
        foreign = repo / "r2c_runs" / "other-paper"
        foreign.mkdir()
        (foreign / "notes.txt").write_text("stray\n", encoding="utf-8")
        return _result()

    with pytest.raises(OutOfScopeWritesError) as exc:
        _dispatch(state, fake, monkeypatch)

    # r2c_runs/ is gitignored, so only the listing layer can see this.
    assert "r2c_runs/other-paper" in str(exc.value)
    assert not (repo / "r2c_runs" / "other-paper").exists()
    moved = list((state.paths.pipeline_dir / "quarantine").rglob("notes.txt"))
    assert len(moved) == 1
    details = _repo_scope_events(state)[0]["details"]
    assert details["new_runs_top_entries"] == ["r2c_runs/other-paper"]


# ---------------------------------------------------------------------------
# Clean and exempt writes raise nothing
# ---------------------------------------------------------------------------


@needs_git
def test_run_dir_confined_writes_raise_nothing(tmp_path, monkeypatch):
    state, repo = _make_repo(tmp_path)

    def fake(*, state, agent, prompt, timeout_s):
        _write_out(state)
        return _result()

    result = _dispatch(state, fake, monkeypatch)
    assert result.completed is True
    assert not (state.paths.pipeline_dir / "quarantine").exists()
    assert not _repo_scope_events(state)


@needs_git
def test_scratch_and_driver_ledger_writes_are_exempt(tmp_path, monkeypatch):
    state, repo = _make_repo(tmp_path)

    def fake(*, state, agent, prompt, timeout_s):
        _write_out(state)
        (repo / ".opencode" / "scratch" / "draft.md").write_text(
            "draft\n", encoding="utf-8")
        history = repo / "r2c_runs" / "_history"
        history.mkdir()
        (history / "ledger.jsonl").write_text("{}\n", encoding="utf-8")
        return _result()

    result = _dispatch(state, fake, monkeypatch)
    assert result.completed is True
    assert not _repo_scope_events(state)
    assert (repo / ".opencode" / "scratch" / "draft.md").is_file()
    assert (repo / "r2c_runs" / "_history" / "ledger.jsonl").is_file()


def test_quarantine_dir_is_driver_managed():
    # The run-dir scan must never blame an agent for the driver's own
    # quarantine writes on a later snapshot diff.
    assert run_pipeline._is_driver_managed(
        ".pipeline/quarantine/some-agent-20260722T000000Z/stray.md")


# ---------------------------------------------------------------------------
# Rollback lever
# ---------------------------------------------------------------------------


@needs_git
def test_env_lever_disables_the_check(tmp_path, monkeypatch):
    monkeypatch.setenv("R2C_REPO_SCOPE_CHECK", "0")
    state, repo = _make_repo(tmp_path)

    def fake(*, state, agent, prompt, timeout_s):
        _write_out(state)
        (repo / "stray_report.md").write_text("stray\n", encoding="utf-8")
        return _result()

    result = _dispatch(state, fake, monkeypatch)
    assert result.completed is True
    # Lever off: the stray stays where the agent put it, nothing quarantined.
    assert (repo / "stray_report.md").is_file()
    assert not (state.paths.pipeline_dir / "quarantine").exists()
    assert not _repo_scope_events(state)


# ---------------------------------------------------------------------------
# Degradation: no git means the listing layer still guards the roots
# ---------------------------------------------------------------------------


def test_without_git_listing_layer_still_catches_sibling(
        tmp_path, monkeypatch):
    state, repo = _make_repo(tmp_path, git=False)

    def fake(*, state, agent, prompt, timeout_s):
        _write_out(state)
        (repo / "runs_shadow").mkdir()
        (repo / "runs_shadow" / "spec.json").write_text(
            "{}", encoding="utf-8")
        return _result()

    with pytest.raises(OutOfScopeWritesError) as exc:
        _dispatch(state, fake, monkeypatch)

    assert "runs_shadow" in str(exc.value)
    assert not (repo / "runs_shadow").exists()
    details = _repo_scope_events(state)[0]["details"]
    assert details["git_degraded"] is True


def test_without_git_deep_tracked_edit_is_not_caught(tmp_path, monkeypatch):
    # The documented degraded blind spot: with no porcelain layer, an edit
    # deep inside a pre-existing tree is invisible (no new top entries).
    state, repo = _make_repo(tmp_path, git=False)

    def fake(*, state, agent, prompt, timeout_s):
        _write_out(state)
        (repo / "scripts" / "tracked.py").write_text(
            "x = 2\n", encoding="utf-8")
        return _result()

    result = _dispatch(state, fake, monkeypatch)
    assert result.completed is True
    assert not _repo_scope_events(state)


# ---------------------------------------------------------------------------
# Unit surface: detect_violations exemptions + stray collapsing
# ---------------------------------------------------------------------------


def _snap(porcelain, repo_top=(), runs_top=()):
    return repo_scope_boundary.RepoScopeSnapshot(
        porcelain=(frozenset(porcelain) if porcelain is not None else None),
        repo_top=frozenset(repo_top),
        runs_top=frozenset(runs_top),
    )


def _detect(before, after, tmp_path, path_exempt=None):
    repo = tmp_path / "repo"
    return repo_scope_boundary.detect_violations(
        before, after,
        repo_root=repo,
        runs_root=repo / "r2c_runs",
        run_dir=repo / "r2c_runs" / "active-paper",
        path_exempt=path_exempt,
    )


def test_detect_exempts_run_dir_scratch_and_predicate_paths(tmp_path):
    before = _snap([])
    after = _snap([
        ("??", "r2c_runs/active-paper/method/model.py"),  # active run dir
        ("??", ".opencode/scratch/draft.md"),             # sanctioned valve
        ("??", "scripts/__pycache__/x.pyc"),              # predicate exempt
        ("??", "stray.md"),                               # genuine violation
    ])
    v = _detect(before, after, tmp_path,
                path_exempt=run_pipeline._repo_scope_exempt_rel_path)
    assert v.new_untracked == ["stray.md"]
    assert v.any_violation()


def test_detect_ignores_preexisting_dirt(tmp_path):
    # Background dirt (parallel session work) present in BOTH snapshots is
    # never attributed to the dispatch.
    dirt = [(" M", "scripts/parallel_work.py"), ("??", "docs/wip/")]
    v = _detect(_snap(dirt), _snap(dirt), tmp_path)
    assert not v.any_violation()


def test_detect_flags_new_tracked_changes_by_kind(tmp_path):
    before = _snap([])
    after = _snap([(" M", "scripts/a.py"), (" D", "scripts/b.py"),
                   ("??", "new_dir/")])
    v = _detect(before, after, tmp_path)
    assert v.modified_tracked == ["scripts/a.py"]
    assert v.deleted_tracked == ["scripts/b.py"]
    assert v.new_untracked == ["new_dir"]


def test_detect_git_degraded_flag_and_listing_layer(tmp_path):
    before = _snap(None, repo_top={"scripts", "r2c_runs"},
                   runs_top={"active-paper"})
    after = _snap(None, repo_top={"scripts", "r2c_runs", "runs_shadow"},
                  runs_top={"active-paper", "other-paper", "_history"})
    v = _detect(before, after, tmp_path)
    assert v.git_degraded is True
    assert v.new_repo_top_entries == ["runs_shadow"]
    # _history is the driver's ledger; the foreign slug is the violation.
    assert v.new_runs_top_entries == ["r2c_runs/other-paper"]


def test_new_stray_paths_dedupes_and_collapses_ancestors():
    v = repo_scope_boundary.RepoScopeViolations(
        new_untracked=["a", "a/b/c", "a-x"],
        new_repo_top_entries=["a"],
        new_runs_top_entries=["r2c_runs/foreign"],
    )
    assert v.new_stray_paths() == ["a", "a-x", "r2c_runs/foreign"]


def test_capture_degrades_outside_a_git_work_tree(tmp_path):
    plain = tmp_path / "plain"
    (plain / "r2c_runs").mkdir(parents=True)
    snap = repo_scope_boundary.capture(plain, plain / "r2c_runs")
    assert snap.porcelain is None
    assert snap.git_warning
    assert "r2c_runs" in snap.repo_top
