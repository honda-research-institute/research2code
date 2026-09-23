"""Canary test for the orchestrator test suite infrastructure.

If this passes, the FakeDispatch monkey-patch + helpers
all work end-to-end. Failures here indicate a problem in the test scaffolding
itself, not in the driver being tested.
"""

from __future__ import annotations

from pathlib import Path

from tests.helpers.state import make_state


def test_canary_fake_dispatch_writes_canned_file(fake_dispatch, run_dir):
    """The FakeDispatch fixture should: (1) be installed as
    run_pipeline.dispatch_agent, (2) write canned outputs to disk under
    state.paths.run_dir when called, (3) record the call for later assertions.
    """
    state = make_state(run_dir)
    fake_dispatch.expect(
        agent="r2c-test-agent",
        writes={".pipeline/canary.json": '{"hello": "world"}'},
    )

    # Import and call through the patched namespace.
    import run_pipeline
    result = run_pipeline.dispatch_agent(
        state=state, agent="r2c-test-agent", prompt="ignored", timeout_s=1,
    )

    # File was written to run_dir.
    canary = run_dir / ".pipeline" / "canary.json"
    assert canary.is_file()
    assert canary.read_text() == '{"hello": "world"}'

    # DispatchResult shape is preserved.
    assert result.completed is True
    assert result.session_id == "fake-session"

    # Call was recorded.
    assert len(fake_dispatch.calls) == 1
    assert fake_dispatch.calls[0].agent == "r2c-test-agent"
    assert fake_dispatch.calls[0].write_paths == [".pipeline/canary.json"]


def test_canary_unexpected_dispatch_fails_clearly(fake_dispatch, run_dir):
    """Calling the fake without queuing an expectation should raise an
    AssertionError naming the agent — protects tests from silently
    over-dispatching."""
    state = make_state(run_dir)
    import run_pipeline

    try:
        run_pipeline.dispatch_agent(
            state=state, agent="r2c-unexpected", prompt="", timeout_s=1,
        )
    except AssertionError as e:
        assert "r2c-unexpected" in str(e)
        return
    raise AssertionError("expected AssertionError for unexpected dispatch")


def test_canary_agent_mismatch_fails_clearly(fake_dispatch, run_dir):
    """If a queued expectation names agent X but the dispatch goes to Y,
    fail with a clear message naming both."""
    state = make_state(run_dir)
    fake_dispatch.expect(agent="r2c-expected-x")

    import run_pipeline
    try:
        run_pipeline.dispatch_agent(
            state=state, agent="r2c-actual-y", prompt="", timeout_s=1,
        )
    except AssertionError as e:
        msg = str(e)
        assert "r2c-expected-x" in msg
        assert "r2c-actual-y" in msg
        # Drain the queue so the teardown assertion passes.
        fake_dispatch._queue.clear()
        return
    raise AssertionError("expected AssertionError for agent mismatch")


def test_canary_unconsumed_expectation_fails_at_teardown(monkeypatch, tmp_path):
    """If a test queues an expectation but never fires it, the fixture
    teardown should raise. We exercise the FakeDispatch directly here rather
    than via the fixture (since the fixture is what we're testing)."""
    from tests.conftest import FakeDispatch
    fd = FakeDispatch()
    fd.expect(agent="r2c-never-called")
    try:
        fd.assert_all_consumed()
    except AssertionError as e:
        assert "r2c-never-called" in str(e)
        return
    raise AssertionError("expected AssertionError for unfired expectation")
