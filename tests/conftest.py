"""Pytest configuration + shared fixtures for the R2C orchestrator test suite.

The suite tests driver-level behavior (run_pipeline.py, judge framework,
sentinels, fix loops) WITHOUT spinning up an opencode server or making real
LLM calls. The FakeDispatch class below substitutes for `dispatch_agent`;
tests queue expected calls + canned outputs, then assert post-dispatch state.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

# `scripts/` is importable via pyproject's `pythonpath = ["scripts"]`
# (B-09); the repo root rides pytest's prepend mode through tests/__init__.py.
REPO_ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# FakeDispatch — substitute for run_pipeline.dispatch_agent
# ---------------------------------------------------------------------------


@dataclass
class _ExpectedCall:
    """One queued expectation for FakeDispatch."""

    agent: str
    writes: dict[str, str | bytes] = field(default_factory=dict)
    completed: bool = True
    error: dict | None = None


@dataclass
class _RecordedCall:
    """One actual call FakeDispatch received. Tests inspect via .calls."""

    agent: str
    prompt: str
    timeout_s: int
    write_paths: list[str]


class FakeDispatch:
    """Substitute for `run_pipeline.dispatch_agent`. Queue-based.

    Usage:
        fake_dispatch.expect(
            agent="r2c-architecture-coder",
            writes={".pipeline/arch_contract.json": "...json...",
                    "method/model.py": "..."},
        )
        # ... drive the pipeline ...
        assert fake_dispatch.calls[0].agent == "r2c-architecture-coder"

    `assert_all_consumed()` is called automatically at fixture teardown to
    catch tests that queue more than they fire.
    """

    def __init__(self):
        self._queue: list[_ExpectedCall] = []
        self.calls: list[_RecordedCall] = []

    def expect(
        self,
        agent: str,
        *,
        writes: dict[str, str | bytes] | None = None,
        completed: bool = True,
        error: dict | None = None,
    ) -> None:
        """Queue an expected dispatch + its on-disk side effects.

        `writes` maps run-dir-relative paths to file content (str or bytes).
        Parents are created as needed.
        """
        self._queue.append(_ExpectedCall(
            agent=agent,
            writes=writes or {},
            completed=completed,
            error=error,
        ))

    def __call__(
        self, *, state, agent: str, prompt: str, timeout_s: int
    ):
        """The fake dispatch_agent. Pops the next expectation, writes its
        canned outputs, returns a fake DispatchResult."""
        # Local import — the schemas module is what run_pipeline imports from.
        from opencode_client import DispatchResult  # noqa: PLC0415

        if not self._queue:
            raise AssertionError(
                f"FakeDispatch: unexpected dispatch to {agent!r}; "
                f"no more expectations queued"
            )
        expected = self._queue.pop(0)
        if expected.agent != agent:
            raise AssertionError(
                f"FakeDispatch: expected dispatch to {expected.agent!r}, "
                f"got {agent!r}"
            )

        run_dir = state.paths.run_dir
        written: list[str] = []
        for rel_path, content in expected.writes.items():
            target = run_dir / rel_path
            target.parent.mkdir(parents=True, exist_ok=True)
            if isinstance(content, bytes):
                target.write_bytes(content)
            else:
                target.write_text(content, encoding="utf-8")
            written.append(rel_path)

        self.calls.append(_RecordedCall(
            agent=agent,
            prompt=prompt,
            timeout_s=timeout_s,
            write_paths=written,
        ))

        return DispatchResult(
            session_id="fake-session",
            user_message_id=f"fake-user-{len(self.calls)}",
            assistant_message_id=f"fake-asst-{len(self.calls)}",
            completed=expected.completed,
            error=expected.error,
            elapsed_s=0.01,
        )

    def assert_all_consumed(self) -> None:
        """At teardown: verify no expectations were left unfired."""
        if self._queue:
            remaining = [e.agent for e in self._queue]
            raise AssertionError(
                f"FakeDispatch has {len(self._queue)} unfired expectation(s): "
                f"{remaining}"
            )


# ---------------------------------------------------------------------------
# Pytest fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _probe_runtime_tier_needs_torch(request):
    """Tests marked probe_runtime execute generated packages and fixtures
    that import torch. Without torch (a fresh `pip install -r
    requirements.txt`, or CI) they would fail on the missing import, not
    on the behavior under test, so the whole tier skips instead."""
    if request.node.get_closest_marker("probe_runtime") is not None:
        pytest.importorskip("torch")


@pytest.fixture(autouse=True)
def _base_timeouts_by_default(monkeypatch):
    """Dispatch budgets are multiplied by R2C_TIMEOUT_SCALE (default 3.0)
    for slower hosted models. Tests pin the base values; the scale has its
    own test in test_dispatch_hardening.py."""
    monkeypatch.setenv("R2C_TIMEOUT_SCALE", "1")


@pytest.fixture(autouse=True)
def _skip_pip_preflight_by_default(monkeypatch):
    """Item 30: the Stage 2.d package-source preflight makes a real network
    probe, but only after a local metadata check finds missing requirements.
    Skip it by default so orchestration tests never touch the network. Tests
    that exercise the source preflight delete this variable and stub
    run_pipeline._pip_index_reachable (see tests/test_pip_preflight.py)."""
    monkeypatch.setenv("R2C_SKIP_PIP_PREFLIGHT", "1")


@pytest.fixture(autouse=True)
def _stub_work_session_transport_by_default(monkeypatch):
    """Agent work always creates a fresh opencode child session.

    Direct `dispatch_agent` unit tests patch the message POST but historically
    relied on the retired shared-session default to avoid the session-create
    request. Stub both read-only transport seams suite-wide so orchestration
    tests remain network-free; focused transport tests override these stubs.
    Return a distinct id each time to preserve the production isolation
    invariant even in tests that dispatch more than once.
    """
    import run_pipeline  # noqa: PLC0415

    created = 0

    def _create_session(**_kwargs):
        nonlocal created
        created += 1
        return f"fake-work-session-{created}"

    monkeypatch.setattr(run_pipeline, "create_session", _create_session)
    monkeypatch.setattr(
        run_pipeline.opencode_client,
        "fetch_session_messages",
        lambda _session_id, *, port: [],
    )


@pytest.fixture(autouse=True)
def _zero_halt_post_retry_delay(monkeypatch):
    """The halt-notice post ladder pauses HALT_POST_RETRY_DELAY_S (=15s in
    production) between attempts 1 and 2. 43 halt-path tests hit that ladder,
    so the literal sleep used to cost the suite ~10 minutes of wall time.
    Zero the module constant suite-wide (production value unchanged). Tests
    that need the real pacing restore the constant themselves — none do
    today; the srl-runaway reproducer paces its own HTTP handler and never
    reaches this ladder."""
    import run_pipeline  # noqa: PLC0415
    monkeypatch.setattr(run_pipeline, "HALT_POST_RETRY_DELAY_S", 0)


@pytest.fixture(autouse=True)
def _stub_halt_post_transport_by_default(monkeypatch):
    """make_state hard-codes port 9999, and the halt/end-of-run notice
    ladders call the real dispatch_and_wait, so halt-path tests used to make
    real TCP connects to 127.0.0.1:9999 (fail-fast today, but each test
    would block 2x180s if anything ever listened there). Stub the transport
    suite-wide with the same observable behavior a refused connect produces
    (OpencodeClientError raised). Focused dispatch tests monkeypatch
    run_pipeline.dispatch_and_wait themselves, which overrides this stub;
    tests driving a real socket use opencode_client directly."""
    import run_pipeline  # noqa: PLC0415
    from opencode_client import OpencodeClientError  # noqa: PLC0415

    def _refused(**_kwargs):
        raise OpencodeClientError(
            "stubbed transport: no opencode server in the test suite "
            "(tests/conftest.py _stub_halt_post_transport_by_default)"
        )

    monkeypatch.setattr(run_pipeline, "dispatch_and_wait", _refused)


@pytest.fixture(autouse=True)
def _skip_cap_burn_resume_by_default(monkeypatch):
    """Cap-burn fix 2: a fatal cap burn triggers ONE corrective resume of
    the burned work session — a real message POST. Disable it suite-wide
    (same pattern as the rung-3 skip below) so every pre-existing
    no-write/cap-burn test keeps its detection-only, one-dispatch shape
    and stays network-free. The focused resume tests in
    test_dispatch_hardening.py restore the real function and stub
    `run_pipeline.dispatch_and_wait` themselves."""
    import run_pipeline  # noqa: PLC0415
    monkeypatch.setattr(run_pipeline, "_cap_burn_corrective_resume",
                        lambda *a, **k: None)


@pytest.fixture(autouse=True)
def _skip_rung3_redispatch_by_default(monkeypatch):
    """Item 8: rung 3 waits out the 120s probation cooldown and probes the
    opencode server for session liveness — a real sleep and a real network
    touch. Disable it suite-wide (same pattern as the pip preflight) so
    every pre-existing dispatch-failure test keeps its one-attempt shape.
    The rung-3 tests in test_wiring_reliability.py restore the real
    function and stub the sleep + liveness probe themselves."""
    import run_pipeline  # noqa: PLC0415
    monkeypatch.setattr(run_pipeline, "_rung3_transport_redispatch",
                        lambda *a, **k: None)


@pytest.fixture
def fake_dispatch(monkeypatch):
    """Provide a FakeDispatch and monkey-patch run_pipeline.dispatch_agent.

    `_dispatch_with_scope_check` calls `dispatch_agent` via the module's own
    namespace, so monkey-patching `run_pipeline.dispatch_agent` catches every
    call site that flows through the scope check.

    At teardown, asserts no expectations were left unfired — catches tests
    that queue more than the driver actually invokes.
    """
    fd = FakeDispatch()
    import run_pipeline  # noqa: PLC0415
    monkeypatch.setattr(run_pipeline, "dispatch_agent", fd)
    yield fd
    fd.assert_all_consumed()


@pytest.fixture
def run_dir(tmp_path: Path) -> Path:
    """Provide an empty run dir under tmp_path with .pipeline/ pre-created."""
    rd = tmp_path / "run"
    (rd / ".pipeline").mkdir(parents=True)
    return rd


# ---------------------------------------------------------------------------
# FakeSubprocess — substitute for run_pipeline.run_script + _run_stage2_script
# ---------------------------------------------------------------------------


@dataclass
class _FakeScriptResult:
    """Mimics subprocess.CompletedProcess for run_script callers."""
    returncode: int
    stdout: str
    stderr: str


@dataclass
class _ExpectedScript:
    """One queued expectation for FakeSubprocess.run_script."""
    result: _FakeScriptResult
    writes: dict[str, str | bytes] = field(default_factory=dict)


class FakeSubprocess:
    """Substitute for `run_pipeline.run_script` AND `_run_stage2_script`.

    `run_script` returns a CompletedProcess-like object; `_run_stage2_script`
    returns the driver's backward-compatible Stage2ScriptOutcome. Both are
    queue-driven, separate queues.

    Tests configure expected outcomes via `expect_script(...)` and
    `expect_stage2_script(...)`. Each call pops from the matching queue.

    `expect_script(writes={...})` simulates files the real script would have
    written as side effects (e.g., `check_feasibility.py` writes a halt
    sidecar at `paths.feasibility_halt` to block the pipeline). Without
    this, side-effect-driven checks (existence of the sidecar) would never
    see the file the real script would have produced.
    """

    def __init__(self):
        self._script_queue: list[_ExpectedScript] = []
        self._stage2_queue: list[_FakeScriptResult] = []
        self.script_calls: list[dict] = []
        self.stage2_calls: list[dict] = []
        self._run_dir: Path | None = None

    def expect_script(
        self, *, returncode: int = 0, stdout: str = "", stderr: str = "",
        writes: dict[str, str | bytes] | None = None,
    ) -> None:
        """Queue an expected `run_script(...)` outcome. `writes` (optional)
        is a map of run-dir-relative paths to content; those files are
        materialized when the call fires, simulating side effects the real
        script would have produced."""
        self._script_queue.append(_ExpectedScript(
            result=_FakeScriptResult(returncode, stdout, stderr),
            writes=writes or {},
        ))

    def set_run_dir(self, run_dir: Path) -> None:
        """Tests that use `expect_script(writes=...)` must declare the run_dir
        so the fake knows where to root the relative paths."""
        self._run_dir = run_dir

    def expect_stage2_script(
        self,
        *,
        ok: bool = True,
        err: str = "",
        returncode: int | None = None,
        stdout: str = "",
    ) -> None:
        """Queue an expected `_run_stage2_script(...)` outcome."""
        resolved_returncode = returncode
        if resolved_returncode is None:
            resolved_returncode = 0 if ok else 1
        self._stage2_queue.append(
            _FakeScriptResult(resolved_returncode, stdout, err)
        )

    def run_script(self, stage_id: str, args: list[str], timeout: int = 120):
        """Substitute for `run_pipeline.run_script`."""
        self.script_calls.append({"stage_id": stage_id, "args": list(args), "timeout": timeout})
        if not self._script_queue:
            raise AssertionError(
                f"FakeSubprocess: unexpected run_script call for stage {stage_id!r} "
                f"with args {args!r}; no more expectations queued"
            )
        expected = self._script_queue.pop(0)
        # Materialize any side-effect writes.
        if expected.writes:
            if self._run_dir is None:
                raise AssertionError(
                    "FakeSubprocess.expect_script(writes=...) requires the test "
                    "to call fake_subprocess.set_run_dir(run_dir) first"
                )
            for rel_path, content in expected.writes.items():
                target = self._run_dir / rel_path
                target.parent.mkdir(parents=True, exist_ok=True)
                if isinstance(content, bytes):
                    target.write_bytes(content)
                else:
                    target.write_text(content, encoding="utf-8")
        return expected.result

    def run_stage2_script(self, state, *, stage_id: str, args: list[str], timeout: int = 120):
        """Substitute for `run_pipeline._run_stage2_script`."""
        self.stage2_calls.append({"stage_id": stage_id, "args": list(args), "timeout": timeout})
        if not self._stage2_queue:
            raise AssertionError(
                f"FakeSubprocess: unexpected _run_stage2_script call for stage "
                f"{stage_id!r} with args {args!r}; no more expectations queued"
            )
        expected = self._stage2_queue.pop(0)
        import run_pipeline
        validator = Path(args[0]).name if args else "stage_script"
        return run_pipeline._stage2_script_outcome(
            validator=validator,
            returncode=expected.returncode,
            stdout=expected.stdout,
            stderr=expected.stderr,
        )

    def assert_all_consumed(self) -> None:
        """At teardown: no unfired script or stage2-script expectations."""
        errs: list[str] = []
        if self._script_queue:
            errs.append(f"{len(self._script_queue)} unfired run_script expectations")
        if self._stage2_queue:
            errs.append(f"{len(self._stage2_queue)} unfired _run_stage2_script expectations")
        if errs:
            raise AssertionError("FakeSubprocess: " + "; ".join(errs))


@pytest.fixture
def fake_subprocess(monkeypatch):
    """Provide a FakeSubprocess and monkey-patch run_pipeline's script
    helpers. Used alongside `fake_dispatch` for stage tests that exercise
    both LLM agents AND subprocess validators."""
    fs = FakeSubprocess()
    import run_pipeline
    monkeypatch.setattr(run_pipeline, "run_script", fs.run_script)
    monkeypatch.setattr(run_pipeline, "_run_stage2_script", fs.run_stage2_script)
    yield fs
    fs.assert_all_consumed()
