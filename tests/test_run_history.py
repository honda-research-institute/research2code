"""Tests for the run-history ledger (write-through mirror + terminal rows).

Design: the run history ledger design note (internal, not shipped). The whole
module is mirror-only and best-effort — the containment tests matter as much
as the happy paths.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest


def _init(run_dir: Path, **kwargs):
    from run_history import init_invocation

    defaults = dict(
        slug="run",
        paper_input="input_papers/run.md",
        input_kind="markdown",
        driver_args={"fresh": False, "stop_after": None},
    )
    defaults.update(kwargs)
    return init_invocation(run_dir, **defaults)


@pytest.fixture
def history_env(run_dir, tmp_path, monkeypatch):
    """Point the history at a tmp dir and keep the kill switch off."""
    history_dir = tmp_path / "history"
    monkeypatch.setenv("R2C_HISTORY_DIR", str(history_dir))
    monkeypatch.delenv("R2C_HISTORY_DISABLE", raising=False)
    return history_dir


def test_tee_mirrors_appended_event_byte_identically(run_dir, history_env):
    from run_events import append_event, event_log_path

    pointer = _init(run_dir)
    assert pointer is not None
    event = append_event(
        run_dir / ".pipeline",
        event_type="run_started",
        run_id="run",
        status="running",
        timestamp="2026-07-14T00:00:00Z",
    )
    assert event is not None
    local = event_log_path(run_dir / ".pipeline").read_text(encoding="utf-8")
    mirror = (
        history_env / pointer["invocation_id"] / "events.jsonl"
    ).read_text(encoding="utf-8")
    assert mirror == local
    assert json.loads(mirror)["event_type"] == "run_started"


def test_tee_without_pointer_is_silent_noop(run_dir, history_env):
    from run_events import append_event

    append_event(
        run_dir / ".pipeline",
        event_type="run_started",
        run_id="run",
    )
    assert not history_env.exists()


def test_unwritable_history_never_blocks_appends(run_dir, history_env, capsys):
    from run_events import append_event, load_events

    _init(run_dir)
    # Make the history dir unwritable by replacing the invocation dir with a file.
    pointer = json.loads((run_dir / ".pipeline" / "history_pointer.json").read_text())
    invocation_dir = history_env / pointer["invocation_id"]
    import shutil

    shutil.rmtree(invocation_dir)
    invocation_dir.write_text("not a directory", encoding="utf-8")

    for index in range(3):
        append_event(
            run_dir / ".pipeline",
            event_type="stage_started",
            run_id="run",
            stage_id="stage_1",
            timestamp=f"2026-07-14T00:00:0{index}Z",
        )
    assert len(load_events(run_dir / ".pipeline")) == 3
    # Failure logged once, not per event.
    err = capsys.readouterr().err
    assert err.count("[history][mirror_failed]") <= 1


def test_disable_env_kills_mirror_and_init(run_dir, history_env, monkeypatch):
    monkeypatch.setenv("R2C_HISTORY_DISABLE", "1")
    assert _init(run_dir) is None
    assert not (run_dir / ".pipeline" / "history_pointer.json").exists()
    assert not history_env.exists()


def test_lineage_survives_resume_and_index_increments(run_dir, history_env):
    first = _init(run_dir)
    second = _init(run_dir)
    assert first["invocation_index"] == 1
    assert second["invocation_index"] == 2
    assert second["lineage_id"] == first["lineage_id"]
    assert second["invocation_id"] != first["invocation_id"]


def test_init_provenance_handles_missing_git(run_dir, history_env, monkeypatch):
    import run_history

    monkeypatch.setattr(
        run_history.subprocess, "run",
        lambda *a, **k: (_ for _ in ()).throw(OSError("no git")),
    )
    pointer = _init(run_dir)
    assert pointer is not None
    assert pointer["tree_commit"] is None
    assert pointer["tree_dirty"] is None
    assert pointer["host"]


def test_git_provenance_ignores_untracked_but_keeps_tracked_changes(tmp_path):
    from run_history import collect_git_provenance

    repo = tmp_path / "repo"
    repo.mkdir()

    def git(*args):
        return subprocess.run(
            ["git", *args],
            cwd=repo,
            check=True,
            capture_output=True,
            text=True,
        )

    git("init", "-q")
    tracked = repo / "tracked.txt"
    tracked.write_text("baseline\n", encoding="utf-8")
    git("add", "tracked.txt")
    git(
        "-c", "user.name=R2C Test",
        "-c", "user.email=r2c@example.invalid",
        "-c", "commit.gpgsign=false",
        "commit", "-qm", "baseline",
    )

    commit, dirty = collect_git_provenance(repo)
    assert commit == git("rev-parse", "HEAD").stdout.strip()
    assert dirty is False

    scratch = repo / "scratch" / "untracked.txt"
    scratch.parent.mkdir()
    scratch.write_text("runtime output\n", encoding="utf-8")
    assert collect_git_provenance(repo) == (commit, False)

    tracked.write_text("unstaged change\n", encoding="utf-8")
    assert collect_git_provenance(repo) == (commit, True)

    git("add", "tracked.txt")
    assert collect_git_provenance(repo) == (commit, True)


def test_finalize_writes_ledger_row_with_terminal_fields(run_dir, history_env):
    from run_events import append_event
    from run_history import finalize_invocation

    pointer = _init(run_dir)
    for event_type, stage in [
        ("run_started", None),
        ("stage_started", "stage_1"),
        ("cap_burn_recovered", "stage_1"),
        ("stage_completed", "stage_1"),
        ("run_finished", None),
    ]:
        append_event(
            run_dir / ".pipeline",
            event_type=event_type,
            run_id="run",
            stage_id=stage,
        )
    (run_dir / ".pipeline" / "progress.json").write_text(json.dumps({
        "slug": "run", "run_status": "completed", "started_at": "x",
        "current": None,
        "stages": [
            {"stage_id": "stage_1", "status": "completed", "duration_s": 12.5},
            {"stage_id": "stage_2a", "status": "pending", "duration_s": None},
        ],
    }), encoding="utf-8")
    details = run_dir / "details"
    details.mkdir()
    (details / "final_manifest.json").write_text(json.dumps({
        "run_status": "passed",
        "delivery": {"label": "draft", "reasons": [{"id": "UB-6", "message": "m"}]},
    }), encoding="utf-8")
    (run_dir / "REPORT.md").write_text("# report\n", encoding="utf-8")
    (run_dir / ".pipeline" / "token_usage.json").write_text(json.dumps({
        "totals": {"input": 10, "output": 20, "reasoning": 0,
                   "cache_read": 0, "cache_write": 0},
        "dispatches_counted": 2,
    }), encoding="utf-8")

    row = finalize_invocation(
        run_dir, run_status="completed",
        think_model="r2c-think/Qwen3.6-27B", code_model="r2c-code/x",
        per_dispatch_sessions=True,
    )
    assert row is not None
    ledger_lines = (history_env / "ledger.jsonl").read_text().splitlines()
    assert len(ledger_lines) == 1
    persisted = json.loads(ledger_lines[0])
    assert persisted["invocation_id"] == pointer["invocation_id"]
    assert persisted["per_dispatch_sessions"] is True
    assert persisted["run_status"] == "completed"
    assert persisted["exit_code"] == 0
    assert persisted["stage_durations_s"] == {"stage_1": 12.5}
    assert persisted["stages_completed"] == ["stage_1"]
    assert persisted["terminal_stage"] == "stage_1"
    assert persisted["delivery_label"] == "draft"
    assert persisted["label_reason_ids"] == ["UB-6"]
    assert persisted["cap_burns_recovered"] == 1
    assert persisted["event_count"] == 5
    assert persisted["tokens"]["output"] == 20
    assert persisted["think_model"] == "r2c-think/Qwen3.6-27B"
    assert persisted["backfilled"] is False
    # Terminal snapshot captured what existed.
    snapshot = persisted["terminal_snapshot"]
    assert "progress.json" in snapshot
    assert "final_manifest.json" in snapshot
    assert "REPORT.md" in snapshot
    terminal_dir = history_env / pointer["invocation_id"] / "terminal"
    assert (terminal_dir / "REPORT.md").read_text() == "# report\n"
    # README sentinel present.
    assert (history_env / "README.md").exists()


def test_finalize_on_crash_path_without_progress(run_dir, history_env):
    """The crashed-run shape: pointer exists, almost nothing else does."""
    from run_history import finalize_invocation

    _init(run_dir)
    row = finalize_invocation(run_dir, run_status="failed")
    assert row is not None
    assert row["run_status"] == "failed"
    assert row["exit_code"] == 1
    assert row["stage_durations_s"] == {}
    assert row["delivery_label"] is None


def test_snapshot_cap_truncates_and_names_loudly(run_dir, history_env):
    from run_history import write_terminal_snapshot

    pointer = _init(run_dir)
    (run_dir / "REPORT.md").write_text("x" * 1000, encoding="utf-8")
    (run_dir / "METHOD.md").write_text("small", encoding="utf-8")
    invocation_dir = history_env / pointer["invocation_id"]
    snapshot, truncated = write_terminal_snapshot(
        run_dir, invocation_dir, cap_bytes=100)
    assert truncated == ["REPORT.md"]
    assert snapshot["REPORT.md"] == 100
    assert (invocation_dir / "terminal" / "REPORT.md").stat().st_size == 100
    assert (invocation_dir / "terminal" / "METHOD.md").read_text() == "small"


def test_halt_fields_read_newest_halt_artifact(run_dir, history_env):
    from run_history import finalize_invocation

    _init(run_dir)
    (run_dir / ".pipeline" / "stage_2a.halt").write_text(json.dumps({
        "status": "halted", "stage": "stage_2a",
        "halt_class": "gap_pack_rejected",
    }), encoding="utf-8")
    row = finalize_invocation(run_dir, run_status="halted")
    assert row["halt_class"] == "gap_pack_rejected"
    assert row["halt_stage"] == "stage_2a"
    assert row["exit_code"] == 1
    assert "stage_2a.halt" in row["terminal_snapshot"]


def test_token_tally_and_dispatch_accumulation(run_dir, history_env):
    from run_history import record_dispatch_tokens, tally_message_tokens

    messages = [
        {"info": {"role": "user"}, "parts": []},
        {"info": {"role": "assistant"}, "parts": [
            {"type": "step-start"},
            {"type": "step-finish",
             "tokens": {"input": 100, "output": 50, "reasoning": 25,
                        "cache": {"read": 10, "write": 5}}},
            {"type": "step-start"},
            {"type": "step-finish", "tokens": {"input": 1, "output": 2}},
        ]},
    ]
    assert tally_message_tokens(messages) == {
        "input": 101, "output": 52, "reasoning": 25,
        "cache_read": 10, "cache_write": 5,
        # B-14 ledger-semantics guard: opencode's `input` is NET of cache
        # reads, so the comparable prompt volume is input + cache_read.
        "prompt": 111,
    }
    record_dispatch_tokens(run_dir, agent="r2c-method-analyzer", messages=messages)
    record_dispatch_tokens(run_dir, agent="r2c-method-analyzer", messages=messages)
    usage = json.loads((run_dir / ".pipeline" / "token_usage.json").read_text())
    assert usage["totals"]["input"] == 202
    assert usage["totals"]["prompt"] == 222
    assert usage["per_agent"]["r2c-method-analyzer"]["output"] == 104
    assert usage["dispatches_counted"] == 2


def test_prompt_volume_survives_a_cache_reporting_changeover(run_dir, history_env):
    """B-14: the day the endpoint starts populating cached_tokens, step
    `input` drops by the cached share while `prompt` stays comparable —
    the exact cross-window breakage the guard exists to prevent."""
    from run_history import tally_message_tokens

    def _msg(inp, cached):
        return [{"info": {"role": "assistant"}, "parts": [
            {"type": "step-finish",
             "tokens": {"input": inp, "output": 1,
                        "cache": {"read": cached, "write": 0}}},
        ]}]

    before = tally_message_tokens(_msg(74_000, 0))       # today: unreported
    after = tally_message_tokens(_msg(7_400, 66_600))    # post-changeover
    assert before["input"] == 74_000 and after["input"] == 7_400
    assert before["prompt"] == after["prompt"] == 74_000


def test_backfill_mints_flagged_row_and_is_idempotent(run_dir, history_env, tmp_path):
    from run_events import append_event
    from backfill_run_history import backfill_run_dir, _existing_invocation_ids

    # A surviving run dir with events but no pointer (pre-ledger run).
    for event_type in ("run_started", "run_finished"):
        append_event(
            run_dir / ".pipeline",
            event_type=event_type,
            run_id="run",
            timestamp="2026-07-01T00:00:00Z",
        )
    existing = _existing_invocation_ids(history_env)
    first = backfill_run_dir(run_dir, history_env, existing, dry_run=False)
    assert first.startswith("OK")
    second = backfill_run_dir(run_dir, history_env, existing, dry_run=False)
    assert second.startswith("SKIP")
    rows = [json.loads(line) for line in
            (history_env / "ledger.jsonl").read_text().splitlines()]
    assert len(rows) == 1
    assert rows[0]["backfilled"] is True
    assert rows[0]["invocation_index"] is None
    assert rows[0]["event_count"] == 2
    mirrored = history_env / rows[0]["events_file"]
    assert mirrored.read_text() == (
        run_dir / ".pipeline" / "run_events.jsonl").read_text()


def test_history_and_hygiene_files_are_driver_managed():
    """The 0715 matrix live catch (40 minutes in): token_usage.json is
    written between the dispatch scope snapshot and the violation check,
    so without the driver-managed exemption the drift detector attributed
    the driver's own write to the agent and deleted it every dispatch —
    and a dispatch with invalid output would have burned its corrective
    redispatch or halted with false attribution."""
    from run_pipeline import _is_driver_managed

    for rel in (
        ".pipeline/token_usage.json",
        ".pipeline/history_pointer.json",
        ".pipeline/data_flow_slice.json",
        ".pipeline/notebook_for_review.ipynb",
    ):
        assert _is_driver_managed(rel) is True, rel
    # Agent artifacts stay in scope.
    assert _is_driver_managed("method/model.py") is False
