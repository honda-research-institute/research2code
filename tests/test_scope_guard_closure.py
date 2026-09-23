"""B-13 scope-guard bypass closure (Track A Block A3).

Change 1: the pack-author dispatch factory was the LAST bare dispatch_agent
caller in the driver, so a pack author writing outside the run dir was
invisible to the repo-scope write boundary — and when the boundary CAN fire
there, the halt must not be classified as the benign "gap_pack_rejected".

Change 2: the delivered-baseline commit is the durable "as delivered"
record; a late overwrite of a manifest-hashed artifact from ANY source must
block that commit and surface, never get baked in.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

import run_pipeline
from author_field_guide_proposal import AuthoringError
from opencode_client import OpencodeClientError
from run_pipeline import OutOfScopeWritesError
from tests.helpers.state import make_state

needs_git = pytest.mark.skipif(
    shutil.which("git") is None, reason="requires git")


def _git(repo: Path, *args: str) -> None:
    subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True, capture_output=True, text=True)


def _make_repo(tmp_path: Path):
    """Same production-mirroring layout as test_repo_scope_boundary."""
    repo = tmp_path / "repo"
    run_dir = repo / "r2c_runs" / "active-paper"
    run_dir.mkdir(parents=True)
    (repo / "scripts").mkdir()
    (repo / "scripts" / "tracked.py").write_text("x = 1\n", encoding="utf-8")
    (repo / ".gitignore").write_text(
        "r2c_runs/\n.opencode/scratch/\n__pycache__/\n*.pyc\n",
        encoding="utf-8")
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "t@example.com")
    _git(repo, "config", "user.name", "T")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "init")
    state = make_state(run_dir, slug="active-paper")
    state.paths.repo_root = repo
    return state, repo


def _result():
    from opencode_client import DispatchResult

    return DispatchResult(
        session_id="s", user_message_id="u", assistant_message_id="a",
        completed=True, error=None, elapsed_s=0.01)


# ---------------------------------------------------------------------------
# Change 1: the pack-author dispatch fn goes through the scope check
# ---------------------------------------------------------------------------


@needs_git
def test_pack_author_stray_repo_write_is_quarantined_and_raises(
        tmp_path, monkeypatch):
    """Known-bad: with the old bare dispatch_agent transport this stray was
    invisible; through _dispatch_with_scope_check the repo bracket
    quarantines it and raises."""
    state, repo = _make_repo(tmp_path)

    def fake(*, state, agent, prompt, timeout_s):
        (repo / "STRAY-AUDIT.md").write_text("oops\n", encoding="utf-8")
        return _result()

    monkeypatch.setattr(run_pipeline, "dispatch_agent", fake)
    dispatch_fn = run_pipeline._make_pack_author_dispatch_fn(state)

    with pytest.raises(OutOfScopeWritesError) as exc:
        dispatch_fn("author the pack")

    assert "quarantined" in str(exc.value)
    assert not (repo / "STRAY-AUDIT.md").exists()
    qroot = state.paths.pipeline_dir / "quarantine"
    assert list(qroot.rglob("STRAY-AUDIT.md"))


def test_pack_author_clean_dispatch_passes_through(tmp_path, monkeypatch):
    """Known-good: a clean pack-author dispatch is unchanged by the swap."""
    state = make_state(tmp_path / "run")
    calls = {}

    def fake(*, state, agent, prompt, timeout_s):
        calls["agent"] = agent
        return _result()

    monkeypatch.setattr(run_pipeline, "dispatch_agent", fake)
    result = run_pipeline._make_pack_author_dispatch_fn(state)("p")
    assert result.completed
    assert calls["agent"] == run_pipeline.PACK_AUTHOR_AGENT


def test_pack_authoring_halt_class_resolves_per_exception():
    """A write-integrity or transport failure in the authoring loop must not
    reach the researcher as the benign 'gap_pack_rejected' coverage gap."""
    assert run_pipeline._pack_authoring_halt_class(
        AuthoringError("pack invalid")) == "gap_pack_rejected"
    assert run_pipeline._pack_authoring_halt_class(
        OutOfScopeWritesError("agent", ["stray"], [])) == "out_of_scope_write"
    assert run_pipeline._pack_authoring_halt_class(
        OpencodeClientError("boom")) == "transport_failure"


# ---------------------------------------------------------------------------
# Change 2: delivered-baseline integrity gate
# ---------------------------------------------------------------------------


def _delivered_run(tmp_path: Path):
    state = make_state(tmp_path / "run")
    run_dir = state.paths.run_dir
    report = run_dir / "REPORT.md"
    report.write_text("delivered content\n", encoding="utf-8")
    from final_manifest import sha256_file

    manifest = {
        "artifacts": [
            {"path": "REPORT.md", "sha256": sha256_file(report),
             "status": "present"},
            {"path": ".pipeline/run_events.jsonl", "sha256": None,
             "status": "present"},  # mutable kind: skipped by the check
        ],
    }
    import run_layout

    manifest_path = run_dir / run_layout.FINAL_MANIFEST_JSON
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    state.delivery_baseline_pending = True
    return state, report


def test_late_overwrite_blocks_the_delivered_baseline(tmp_path, monkeypatch):
    """Known-bad: mutate a manifest-hashed file after the manifest wrote its
    hash; the baseline commit must be blocked, the event recorded, and the
    violation land in KNOWN_ISSUES.md."""
    state, report = _delivered_run(tmp_path)
    report.write_text("silently overwritten after manifest\n",
                      encoding="utf-8")

    import scripts.finalize_run_git as frg

    called = {}
    monkeypatch.setattr(
        frg, "ensure_delivered_baseline",
        lambda run_dir: called.setdefault("yes", True) or (True, "ok"))

    run_pipeline._finalize_delivered_baseline(state)

    assert "yes" not in called, "baseline commit must be blocked"
    events = [
        json.loads(line)
        for line in (state.paths.pipeline_dir / "run_events.jsonl")
        .read_text(encoding="utf-8").splitlines()
    ]
    blocked = [e for e in events
               if e["event_type"] == "delivered_baseline_blocked"]
    assert len(blocked) == 1
    assert blocked[0]["details"]["violations"]
    assert "REPORT.md" in blocked[0]["details"]["violations"][0]

    import run_layout

    known = (state.paths.run_dir / run_layout.KNOWN_ISSUES_MD).read_text(
        encoding="utf-8")
    assert "Delivered-baseline integrity violation" in known
    assert "REPORT.md" in known


def test_clean_delivery_commits_the_baseline_unchanged(tmp_path, monkeypatch):
    """Known-good: hashes match, the baseline commit proceeds."""
    state, _report = _delivered_run(tmp_path)

    import scripts.finalize_run_git as frg

    called = {}
    monkeypatch.setattr(
        frg, "ensure_delivered_baseline",
        lambda run_dir: called.setdefault("yes", True) or (True, "ok"))

    run_pipeline._finalize_delivered_baseline(state)
    assert called.get("yes"), "clean flow must still commit the baseline"


def test_missing_manifest_means_no_violations(tmp_path):
    assert run_pipeline._delivered_manifest_integrity_violations(
        tmp_path) == []


def test_vanished_artifact_is_a_violation(tmp_path):
    state, report = _delivered_run(tmp_path)
    report.unlink()
    violations = run_pipeline._delivered_manifest_integrity_violations(
        state.paths.run_dir)
    assert len(violations) == 1 and "missing from disk" in violations[0]
