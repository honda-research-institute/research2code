"""main()-loop tier: the stage-sequencing layer that no other test executes.

Every other test file calls stage functions directly and hand-simulates the
main loop's side effects (test_cross_stage.py admits this explicitly). This
tier drives `run_pipeline.main()` itself with all eleven stage functions
replaced by scripted fakes, so the unit under test is the LOOP:

- sequential execution order and halt short-circuit
- sentinel writes on completion / clears on halt+degrade
- the 2d/3c upstream-digest hooks firing only on completion
- degrade-continues vs halt-stops
- --stop-after early exit with finalizers
- run_status derivation (completed / degraded / halted / failed) in the
  run_finished event
- run-lock acquire/release and the unreachable-server / lock-contention exits

Stage-INTERNAL behavior (skip-if-done, fix loops, judge routing) is covered by
the per-stage files; resume semantics at loop level are "sentinels preserved",
asserted here.

Nothing in these tests touches the network: health_check, discover_session,
_prime_agent_models, todo posting, and all stage fns are monkeypatched.
(Important locally: live opencode servers may be up on 4096-4098; the fake
port 9999 plus patched client functions guarantee no test traffic reaches
them.)
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests.helpers.state import make_paths
from trusted import trusted_path, trusted_text


STAGE_IDS = [
    "stage_1", "stage_1x", "stage_2a", "stage_2b", "stage_2c", "stage_2d",
    "stage_2x", "stage_3a", "stage_3b", "stage_3c", "stage_4", "stage_5",
]


class LoopHarness:
    """Patches main()'s collaborators; scripts per-stage outcomes.

    `script` maps stage_id -> status ("completed" by default). Records the
    order stages ran, todo advances, digest-hook and finalizer calls.
    """

    def __init__(self, monkeypatch, tmp_path: Path, script: dict[str, str] | None = None):
        import run_pipeline as rp

        self.rp = rp
        self.script = script or {}
        self.ran: list[str] = []
        self.todo_advances: list[tuple[str, str]] = []
        self.digests: list[str] = []
        self.finalizers: list[str] = []
        self.halt_notices: list[tuple[str, str]] = []
        self.notice_lock_states: list[bool] = []
        self.manifest_result: tuple[bool, str] = (True, "passed")

        run_dir = tmp_path / "run"
        (run_dir / ".pipeline").mkdir(parents=True)
        self.paths = make_paths(run_dir, slug="loop-test")

        def fake_stage_0(paper, workspace, fresh=False, before_parse=None):
            if before_parse is not None:
                before_parse(self.paths)
            s0 = rp.StageResult(status="completed", stage_id="stage_0")
            return self.paths, s0

        def make_stage_fn(stage_id: str):
            def fake_stage(state):
                self.ran.append(stage_id)
                if self.script.get(stage_id) == "halt_via_helper":
                    return rp.halt(
                        state.paths,
                        stage_id,
                        reason=f"{stage_id} blocked",
                        user_message=f"{stage_id} blocked",
                        state=state,
                    )
                return rp.StageResult(
                    status=self.script.get(stage_id, "completed"),
                    stage_id=stage_id,
                )
            return fake_stage

        monkeypatch.setattr(rp, "_check_for_session_deadlock_or_exit", lambda: None)
        monkeypatch.setattr(rp, "run_stage_0", fake_stage_0)
        for sid in STAGE_IDS:
            monkeypatch.setattr(rp, f"run_{sid}", make_stage_fn(sid))
        monkeypatch.setattr(rp, "health_check", lambda *, port: True)
        monkeypatch.setattr(
            rp, "discover_session", lambda *, directory, port: "fake-session"
        )
        monkeypatch.setattr(
            rp, "_prime_agent_models", lambda port, names: {n: None for n in names}
        )
        monkeypatch.setattr(rp, "_reconcile_candidate_dispatches",
                            lambda paths, context: None)
        monkeypatch.setattr(
            rp,
            "_post_halt_to_session",
            lambda state, stage_id, reason, halt_path, user_message,
            **kwargs: self.halt_notices.append((stage_id, reason)),
        )
        monkeypatch.setattr(rp, "_init_pipeline_todos", lambda state: None)
        monkeypatch.setattr(
            rp, "_advance_todos",
            lambda state, stage_id, status: self.todo_advances.append((stage_id, status)),
        )
        monkeypatch.setattr(
            rp, "write_stage_3c_upstream_digest",
            lambda paths: self.digests.append("stage_3c"),
        )
        monkeypatch.setattr(
            rp, "write_stage_2d_upstream_digest",
            lambda paths: self.digests.append("stage_2d"),
        )
        monkeypatch.setattr(
            rp, "finalize_assumptions_md",
            lambda state: self.finalizers.append("assumptions"),
        )
        monkeypatch.setattr(
            rp, "finalize_run_report",
            lambda state: self.finalizers.append("report"),
        )
        monkeypatch.setattr(
            rp, "run_delivery_gating",
            lambda state: (self.finalizers.append("gating"),
                           {"label": "verified", "reasons": [],
                            "disclosures": [], "probe_counts": {}})[1],
        )
        monkeypatch.setattr(
            rp, "finalize_delivery_banner",
            lambda state, delivery: self.finalizers.append("banner"),
        )
        monkeypatch.setattr(
            rp, "finalize_claims_report",
            lambda state, delivery=None, stage_results=None:
            self.finalizers.append("front_door"),
        )
        def fake_write_final_manifest(
            paths, *, stage_results=None, git_commit=None, generated_at=None,
            delivery=None,
        ):
            self.finalizers.append("manifest")
            manifest_path = paths.run_dir / "details" / "final_manifest.json"
            manifest_path.parent.mkdir(parents=True, exist_ok=True)
            manifest_path.write_text("{}\n", encoding="utf-8")
            return SimpleNamespace(
                run_status=self.manifest_result[1],
                diagnostics=[],
                artifacts=[],
            )

        monkeypatch.setattr(rp, "write_final_manifest", fake_write_final_manifest)
        monkeypatch.setattr(
            rp, "_post_end_of_run_notice",
            lambda state, *, run_status, manifest_status, delivery=None:
            (
                self.finalizers.append("notice"),
                self.notice_lock_states.append(self.lock_dir_exists()),
            ),
        )

    def main(self, *extra_args: str) -> int:
        return self.rp.main(
            ["--paper", "loop-test", "--port", "9999", *extra_args]
        )

    # -- assertion helpers ---------------------------------------------------

    def events(self) -> list[dict]:
        log = self.paths.pipeline_dir / "run_events.jsonl"
        if not log.exists():
            return []
        return [json.loads(line) for line in log.read_text().splitlines() if line.strip()]

    def run_finished_status(self) -> str | None:
        finished = [e for e in self.events() if e.get("event_type") == "run_finished"]
        assert len(finished) <= 1
        return finished[0]["status"] if finished else None

    def progress(self) -> dict:
        return json.loads(
            (self.paths.pipeline_dir / "progress.json").read_text(encoding="utf-8")
        )

    def sentinel_exists(self, stage_id: str) -> bool:
        return self.rp._stage_complete_sentinel_path(self.paths, stage_id).exists()

    def lock_dir_exists(self) -> bool:
        return (self.paths.pipeline_dir / "_lock").exists()

    def git_status(self) -> str:
        return subprocess.run(
            ["git", "-C", str(trusted_path(self.paths.run_dir)), "status", "--porcelain"],
            capture_output=True, text=True, check=True,
        ).stdout

    def tracked_files(self) -> set[str]:
        output = subprocess.run(
            ["git", "-C", str(trusted_path(self.paths.run_dir)), "ls-files"],
            capture_output=True, text=True, check=True,
        ).stdout
        return set(output.splitlines())

    def committed_text(self, rel_path: str) -> str:
        return subprocess.run(
            ["git", "-C", str(trusted_path(self.paths.run_dir)), "show",
             f"HEAD:{trusted_text(rel_path)}"],
            capture_output=True, text=True, check=True,
        ).stdout


@pytest.fixture
def harness_factory(monkeypatch, tmp_path):
    def factory(script: dict[str, str] | None = None) -> LoopHarness:
        return LoopHarness(monkeypatch, tmp_path, script)
    return factory


def test_happy_path_runs_all_stages_in_order(harness_factory):
    h = harness_factory()
    rc = h.main()
    assert rc == 0
    assert h.ran == STAGE_IDS
    # Completed stages persist sentinels; the 2d/3c digest hooks fired.
    for sid in STAGE_IDS:
        assert h.sentinel_exists(sid), sid
    assert h.digests == ["stage_2d", "stage_3c"]
    # End-of-run finalizers each ran once; gating decides the label before
    # the banner, REPORT.md is written before the manifest, and the final
    # session notice runs after the manifest status is known.
    assert h.finalizers == ["assumptions", "report", "gating", "banner", "front_door",
                            "manifest", "notice"]
    assert h.run_finished_status() == "completed"
    assert h.progress()["run_status"] == "completed"
    assert h.progress()["current"] is None
    # One stage_started event per loop stage + the retrospective stage_0 pair.
    started = [e for e in h.events() if e.get("event_type") == "stage_started"]
    assert [e["stage_id"] for e in started] == ["stage_0", *STAGE_IDS]
    # Todos advanced for stage_0 seed + every loop stage.
    assert h.todo_advances == [("stage_0", "completed")] + [
        (sid, "completed") for sid in STAGE_IDS
    ]
    # Lock released in the finally block.
    assert not h.lock_dir_exists()
    assert h.notice_lock_states == [False]
    assert h.git_status() == ""
    assert not any(path.startswith(".pipeline/_lock/")
                   for path in h.tracked_files())
    committed_state = json.loads(h.committed_text(".pipeline/driver_state.json"))
    assert committed_state["stages"][-1]["stage_id"] == "stage_5"
    assert committed_state["stages"][-1]["status"] == "completed"
    committed_progress = json.loads(h.committed_text(".pipeline/progress.json"))
    assert committed_progress["run_status"] == "completed"
    assert committed_progress["current"] is None
    committed_events = [
        json.loads(line)
        for line in h.committed_text(".pipeline/run_events.jsonl").splitlines()
        if line.strip()
    ]
    terminal_events = [
        event for event in committed_events
        if event["event_type"] in {
            "stage_completed", "run_finished", "run_lock_released",
        }
    ]
    assert [event["event_type"] for event in terminal_events[-3:]] == [
        "stage_completed", "run_finished", "run_lock_released",
    ]
    assert terminal_events[-3]["stage_id"] == "stage_5"


def test_legacy_shared_session_cli_flag_is_rejected(harness_factory):
    h = harness_factory()
    with pytest.raises(SystemExit) as exc:
        h.main("--no-per-dispatch-sessions")
    assert exc.value.code == 2
    assert h.ran == []


def test_legacy_shared_session_env_cannot_change_new_run_provenance(
        harness_factory, monkeypatch):
    h = harness_factory()
    history_dir = h.paths.run_dir.parent / "history"
    monkeypatch.delenv("R2C_HISTORY_DISABLE", raising=False)
    monkeypatch.setenv("R2C_HISTORY_DIR", str(history_dir))
    monkeypatch.setenv("R2C_SHARED_SESSION", "1")

    assert h.main() == 0

    pointer = json.loads(
        (h.paths.pipeline_dir / "history_pointer.json").read_text())
    rows = [json.loads(line) for line in
            (history_dir / "ledger.jsonl").read_text().splitlines()]
    row = next(r for r in rows
               if r["invocation_id"] == pointer["invocation_id"])
    assert row["per_dispatch_sessions"] is True
    assert "per_dispatch_sessions" not in row["driver_args"]


def test_stage0_exception_after_lock_releases(monkeypatch, tmp_path):
    import run_pipeline as rp

    run_dir = tmp_path / "run"
    (run_dir / ".pipeline").mkdir(parents=True)
    paths = make_paths(run_dir, slug="loop-test")

    def fake_stage_0(paper, workspace, fresh=False, before_parse=None):
        assert before_parse is not None
        before_parse(paths)
        raise RuntimeError("parse exploded")

    monkeypatch.setattr(rp, "_check_for_session_deadlock_or_exit", lambda: None)
    monkeypatch.setattr(rp, "run_stage_0", fake_stage_0)

    with pytest.raises(RuntimeError, match="parse exploded"):
        rp.main(["--paper", "loop-test", "--port", "9999"])

    assert not (paths.pipeline_dir / "_lock").exists()
    events = [
        json.loads(line)
        for line in (paths.pipeline_dir / "run_events.jsonl").read_text().splitlines()
        if line.strip()
    ]
    assert events[-2]["event_type"] == "run_finished"
    assert events[-2]["status"] == "failed"
    assert events[-1]["event_type"] == "run_lock_released"


def test_halt_short_circuits_and_clears_sentinel(harness_factory):
    h = harness_factory(script={"stage_2c": "halted"})
    rc = h.main()
    assert rc == 1
    assert h.ran == ["stage_1", "stage_1x", "stage_2a", "stage_2b",
                     "stage_2c"]
    # Halted stage's sentinel cleared; prior completions preserved (this IS
    # the loop-level resume contract: a re-run skips 1/2a/2b, re-attempts 2c).
    assert h.sentinel_exists("stage_2b")
    assert not h.sentinel_exists("stage_2c")
    # A synthetic halted StageResult only exercises loop short-circuiting. Real
    # stage halts call halt(..., state=state), covered below.
    assert h.finalizers == []
    assert h.run_finished_status() == "halted"
    assert h.todo_advances[-1] == ("stage_2c", "halted")
    assert not h.lock_dir_exists()


def test_halt_helper_packages_before_loop_exit(harness_factory):
    h = harness_factory(script={"stage_2c": "halt_via_helper"})
    rc = h.main()
    assert rc == 1
    assert h.ran == ["stage_1", "stage_1x", "stage_2a", "stage_2b",
                     "stage_2c"]
    assert h.finalizers == ["assumptions", "report", "banner", "front_door",
                            "manifest"]
    assert h.halt_notices == [("stage_2c", "stage_2c blocked")]
    assert h.run_finished_status() == "halted"
    assert h.todo_advances[-1] == ("stage_2c", "halted")
    assert not h.lock_dir_exists()
    assert h.git_status() == ""
    tracked = h.tracked_files()
    assert ".pipeline/stage_2c.halt" in tracked
    assert not any(path.startswith(".pipeline/_lock/") for path in tracked)
    committed_state = json.loads(h.committed_text(".pipeline/driver_state.json"))
    assert committed_state["stages"][-1]["stage_id"] == "stage_2c"
    assert committed_state["stages"][-1]["status"] == "halted"
    committed_progress = json.loads(h.committed_text(".pipeline/progress.json"))
    assert committed_progress["run_status"] == "halted"
    assert committed_progress["current"] is None
    committed_events = [
        json.loads(line)
        for line in h.committed_text(".pipeline/run_events.jsonl").splitlines()
        if line.strip()
    ]
    assert [event["event_type"] for event in committed_events[-3:]] == [
        "stage_halted", "run_finished", "run_lock_released",
    ]
    assert committed_events[-3]["stage_id"] == "stage_2c"


def test_degraded_stage_continues_to_completion(harness_factory):
    h = harness_factory(script={"stage_3c": "degraded"})
    rc = h.main()
    assert rc == 0
    assert h.ran == STAGE_IDS
    # Degraded is not clean: sentinel cleared so a re-run re-attempts it,
    # and the 3c digest hook must NOT fire (completion-only).
    assert not h.sentinel_exists("stage_3c")
    assert h.sentinel_exists("stage_3b")
    assert h.digests == ["stage_2d"]
    assert h.finalizers == ["assumptions", "report", "gating", "banner", "front_door",
                            "manifest", "notice"]
    assert h.run_finished_status() == "degraded"


def test_manifest_degraded_downgrades_clean_run(harness_factory):
    h = harness_factory()
    h.manifest_result = (True, "degraded")
    rc = h.main()
    assert rc == 0
    assert h.run_finished_status() == "degraded"


def test_manifest_blocked_fails_run(harness_factory):
    h = harness_factory()
    h.manifest_result = (False, "blocked")
    rc = h.main()
    assert rc == 2
    assert h.run_finished_status() == "failed"


def test_manifest_generation_failure_does_not_create_baseline(
        harness_factory, monkeypatch):
    h = harness_factory()

    def fail_manifest(*args, **kwargs):
        raise OSError("manifest write failed")

    monkeypatch.setattr(h.rp, "write_final_manifest", fail_manifest)

    assert h.main() == 2
    assert h.run_finished_status() == "failed"
    assert not (h.paths.run_dir / ".git").exists()


@pytest.mark.parametrize(
    ("script", "expected_rc"),
    [({}, 0), ({"stage_2c": "halt_via_helper"}, 1)],
)
def test_baseline_failure_never_changes_pipeline_result(
        harness_factory, monkeypatch, script, expected_rc):
    h = harness_factory(script=script)
    import scripts.finalize_run_git as finalize_run_git

    monkeypatch.setattr(
        finalize_run_git,
        "ensure_delivered_baseline",
        lambda run_dir: (False, "git unavailable"),
    )

    assert h.main() == expected_rc
    assert not (h.paths.run_dir / ".git").exists()


def test_final_notice_failure_does_not_mask_exit(harness_factory, monkeypatch):
    h = harness_factory()

    def boom(*args, **kwargs):
        raise ValueError("notice failed")

    monkeypatch.setattr(h.rp, "_post_end_of_run_notice", boom)

    rc = h.main()

    assert rc == 0
    assert h.run_finished_status() == "completed"
    assert not h.lock_dir_exists()


def test_stop_after_exits_early_with_finalizers(harness_factory):
    h = harness_factory()
    rc = h.main("--stop-after", "stage_2b")
    assert rc == 0
    assert h.ran == ["stage_1", "stage_1x", "stage_2a", "stage_2b"]
    # stop-after finalizes the researcher-facing reports but only stage_5
    # triggers the manifest.
    assert h.finalizers == ["assumptions", "report"]
    assert h.run_finished_status() == "completed"
    assert not (h.paths.run_dir / ".git").exists()


def test_unreachable_server_fails_before_any_stage(harness_factory, monkeypatch):
    h = harness_factory()
    monkeypatch.setattr(h.rp, "health_check", lambda *, port: False)
    rc = h.main()
    assert rc == 2
    assert h.ran == []
    assert h.run_finished_status() == "failed"
    assert not h.lock_dir_exists()


def test_lock_contention_exits_2_without_events(harness_factory):
    h = harness_factory()
    # Simulate a concurrent (or crashed) run holding the lock.
    (h.paths.pipeline_dir / "_lock").mkdir()
    rc = h.main()
    assert rc == 2
    assert h.ran == []
    # Lock acquisition failed before the event log's run_started — no
    # run_finished either (the finally block is inside the acquired branch).
    assert h.run_finished_status() is None
    # The foreign lock is NOT stolen.
    assert h.lock_dir_exists()
