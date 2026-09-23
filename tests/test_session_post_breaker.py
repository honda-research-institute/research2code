"""R2C-061 — progress posts stop retrying a dead session.

Across the 2026-08-04 night rolls and the 2026-08-05 day roll every
progress-todo post targeted the same stale parent session, every one returned
a server 500, each retried once with a 30s pause, and the driver never
learned: about eleven failure pairs in one log. Discovery could not
self-correct because the broken session's freshness is renewed by the very
child dispatches that work (`zombie_notification_target`).

Zero wall-clock cost (fire-and-forget daemon threads) and one hundred percent
log noise, which materially distorted the perceived reliability of the run.
"""

from __future__ import annotations

import pytest

from tests.helpers.state import make_state


def test_the_breaker_state_machine():
    from run_pipeline import SessionPostBreaker

    breaker = SessionPostBreaker()
    assert not breaker.disabled and not breaker.wants_rediscovery()

    breaker.record_failure()
    assert not breaker.wants_rediscovery(), "one failure is just a retry"

    breaker.record_failure()
    assert breaker.wants_rediscovery() and not breaker.disabled

    breaker.note_rediscovery()
    assert not breaker.wants_rediscovery(), "exactly one rediscovery"

    breaker.record_failure()
    assert breaker.disabled, "a failure after rediscovery ends the mechanism"


def test_one_success_clears_everything():
    from run_pipeline import SessionPostBreaker

    breaker = SessionPostBreaker()
    breaker.record_failure()
    breaker.record_failure()
    breaker.note_rediscovery()
    breaker.record_success()

    assert not breaker.wants_rediscovery()
    breaker.record_failure()
    breaker.record_failure()
    assert not breaker.disabled, "a live session earns a fresh rediscovery"


def _flaky(monkeypatch, *, fail_times: int) -> dict:
    import run_pipeline
    from opencode_client import OpencodeClientError

    calls = {"n": 0, "sessions": []}

    def _dispatch(**kwargs):
        calls["n"] += 1
        calls["sessions"].append(kwargs.get("session_id"))
        if calls["n"] <= fail_times:
            raise OpencodeClientError("session busy")
        return {"parts": [{"type": "text", "text": "todos updated."}]}

    monkeypatch.setattr(run_pipeline, "dispatch_and_wait", _dispatch)
    monkeypatch.setattr(run_pipeline.time, "sleep", lambda s: None)
    return calls


def test_a_dead_target_is_rediscovered_once_then_the_posts_stop(
    tmp_path, monkeypatch,
):
    """The recorded shape end to end: the stale session refuses everything,
    the driver switches targets once, and gives up for the run."""
    import run_pipeline
    from run_pipeline import _post_todos, _post_todos_worker

    state = make_state(tmp_path / "run")
    state.session_id = "ses_stale"
    state.workspace = "/workspace"
    state.todos = [{"stage_id": "stage_1", "content": "x", "activeForm": "x",
                    "status": "in_progress"}]
    calls = _flaky(monkeypatch, fail_times=99)
    monkeypatch.setattr(run_pipeline, "discover_session",
                        lambda **kw: "ses_fresh")

    _post_todos_worker(state, "update")

    assert calls["sessions"] == ["ses_stale", "ses_stale", "ses_fresh"]
    assert state.post_breaker.disabled
    assert state.session_id == "ses_fresh"

    # Every later stage transition now costs nothing at all.
    before = calls["n"]
    _post_todos(state)
    assert calls["n"] == before


def test_a_headless_run_never_posts_a_todo(tmp_path, monkeypatch):
    """Launched with no interactive consumer, the mechanism has no reader."""
    import run_pipeline
    from run_pipeline import _post_todos

    state = make_state(tmp_path / "run")
    state.todos = [{"stage_id": "stage_1", "content": "x", "activeForm": "x",
                    "status": "in_progress"}]
    monkeypatch.setenv(run_pipeline.NO_PROGRESS_TODOS_ENV, "1")

    def _explode(*args, **kwargs):
        raise AssertionError("a headless run must not spawn a post thread")

    monkeypatch.setattr(run_pipeline.threading, "Thread", _explode)
    _post_todos(state)  # must not raise


@pytest.mark.parametrize("poster,args", [
    ("_post_halt_to_session", ("stage_2b", "reason")),
    ("_post_end_of_run_notice", None),
])
def test_must_deliver_notices_rediscover_and_are_never_disabled(
    tmp_path, monkeypatch, poster, args,
):
    """A halt notice that silently dies leaves someone watching a session
    that never learns the run ended, so these paths keep the rediscovery arm
    and drop the disable arm."""
    import run_pipeline

    state = make_state(tmp_path / "run")
    state.session_id = "ses_stale"
    state.workspace = "/workspace"
    state.post_breaker.disabled = True  # cosmetic posts are off; these are not
    calls = _flaky(monkeypatch, fail_times=2)
    monkeypatch.setattr(run_pipeline, "discover_session",
                        lambda **kw: "ses_fresh")

    fn = getattr(run_pipeline, poster)
    if args is None:
        fn(state, run_status="completed", manifest_status="ok",
           delivery={"label": "verified"})
    else:
        halt_path = tmp_path / "run" / ".pipeline" / "stage_2b.halt"
        halt_path.parent.mkdir(parents=True, exist_ok=True)
        halt_path.write_text("{}", encoding="utf-8")
        fn(state, args[0], args[1], halt_path)

    assert calls["sessions"] == ["ses_stale", "ses_stale", "ses_fresh"]
    assert calls["n"] == 3, "the third attempt goes to the fresh session"
