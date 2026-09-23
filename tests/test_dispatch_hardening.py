"""Wiring slice 3: truncated-turn detection + notice-delivery retries.

The M-LOAD-1 matrix's reliability classes, hardened at the dispatch layer:

- A producer dispatch that "completes" without writing a single file (the
  no-write turn class, B-002) is DETECTED from the existing scope-check
  snapshot and logged as a first-class run event. Detection only, no
  dispatch-layer retry: every stage already owns its recovery for the
  class (missing-output retries, the judge no-write retry, the coder fix
  loops), and stacking a retry under those doubles dispatch counts on
  persistent failures.
- Todo and halt notices get one retry after a pause: the dominant live
  failure was a session busy with a long agent turn timing the post out
  at 180s, and halt notices are the researcher's only stop signal.

All network collaborators are patched; nothing here dispatches anywhere.
"""

from __future__ import annotations

import json

import pytest

import run_layout
import run_pipeline
from opencode_client import (DispatchResult, OpencodeClientError,
                             PER_STEP_OUTPUT_CAP, cap_burn_steps)
from run_pipeline import (_dispatch_with_scope_check, _post_end_of_run_notice,
                          _post_halt_to_session, _post_todos_worker,
                          _render_end_of_run_notice)
from tests.helpers.state import make_state


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


# ---------------------------------------------------------------------------
# Truncated-turn detection (_dispatch_with_scope_check)
# ---------------------------------------------------------------------------


WRITEABLE = ["method/*", ".pipeline/out.json"]


def test_no_write_dispatch_emits_detection_event(tmp_path, monkeypatch):
    state = make_state(tmp_path / "run")
    calls = {"n": 0}

    def fake_dispatch(*, state, agent, prompt, timeout_s):
        calls["n"] += 1
        return _result()

    monkeypatch.setattr(run_pipeline, "dispatch_agent", fake_dispatch)
    result = _dispatch_with_scope_check(
        state=state, agent="r2c-test-producer", prompt="p", timeout_s=5,
        writeable_paths_override=WRITEABLE)
    assert result.completed is True
    # Detection only — exactly ONE dispatch; stage-level recovery owns
    # the retry (missing-output retries, judge retry, fix loops).
    assert calls["n"] == 1
    truncated = [e for e in _events(state)
                 if e["event_type"] == "truncated_turn_detected"]
    assert len(truncated) == 1
    assert truncated[0]["details"]["agent"] == "r2c-test-producer"


def test_writing_dispatch_emits_no_detection_event(tmp_path, monkeypatch):
    state = make_state(tmp_path / "run")

    def fake_dispatch(*, state, agent, prompt, timeout_s):
        (state.paths.run_dir / ".pipeline" / "out.json").write_text("{}")
        return _result()

    monkeypatch.setattr(run_pipeline, "dispatch_agent", fake_dispatch)
    _dispatch_with_scope_check(
        state=state, agent="r2c-test-producer", prompt="p", timeout_s=5,
        writeable_paths_override=WRITEABLE)
    assert not [e for e in _events(state)
                if e["event_type"] == "truncated_turn_detected"]


def test_out_of_scope_enforcement_unchanged(tmp_path, monkeypatch):
    # The detection hook must not weaken ownership enforcement.
    state = make_state(tmp_path / "run")

    def fake_dispatch(*, state, agent, prompt, timeout_s):
        (state.paths.run_dir / "rogue.txt").write_text("out of scope")
        return _result()

    monkeypatch.setattr(run_pipeline, "dispatch_agent", fake_dispatch)
    with pytest.raises(run_pipeline.OutOfScopeWritesError) as exc:
        _dispatch_with_scope_check(
            state=state, agent="r2c-test-producer", prompt="p", timeout_s=5,
            writeable_paths_override=WRITEABLE)
    assert "rogue.txt" in str(exc.value)


# ---------------------------------------------------------------------------
# Cap-burn signature (item 23's last piece, the item 8 taxonomy datum):
# the truncated-turn class splits into cap_burn (final step at the per-step
# output cap, no tool call — deterministic, resumable) vs short_empty vs
# unmeasured. Signature measured live on the ACC 2026-07-08 run: three
# casualty turns, each with a reasoning-only final step at exactly 20000
# output tokens.
# ---------------------------------------------------------------------------


def _assistant(parts: list[dict]) -> dict:
    return {"info": {"role": "assistant"}, "parts": parts}


def _finish(output: int) -> dict:
    return {"type": "step-finish",
            "tokens": {"total": output, "input": 0, "output": output,
                       "reasoning": 0, "cache": {"read": 0, "write": 0}}}


def test_cap_burn_detected_on_reasoning_only_step():
    messages = [
        {"info": {"role": "user"}, "parts": [{"type": "text"}]},
        _assistant([{"type": "step-start"}, {"type": "reasoning"},
                    _finish(PER_STEP_OUTPUT_CAP)]),
    ]
    burns = cap_burn_steps(messages)
    assert burns == [{"output_tokens": PER_STEP_OUTPUT_CAP,
                      "cap": PER_STEP_OUTPUT_CAP,
                      "tool_call_truncated": False}]


def test_no_burn_when_the_step_issued_a_valid_tool_call():
    messages = [_assistant([
        {"type": "step-start"}, {"type": "reasoning"},
        {"type": "tool", "tool": "write"}, _finish(PER_STEP_OUTPUT_CAP)])]
    assert cap_burn_steps(messages) == []


def test_truncated_tool_call_at_the_cap_is_a_burn():
    # The 2026-07-13 shape the final-step-only detector missed: the write
    # call itself outgrew the cap and its JSON arrived cut off — the
    # server records the tool name as the literal "invalid" (seen in
    # ICRA, twice in iDb-RRT, once self-recovered in ms3d).
    messages = [_assistant([
        {"type": "step-start"}, {"type": "reasoning"},
        {"type": "tool", "tool": "invalid"}, _finish(PER_STEP_OUTPUT_CAP)])]
    burns = cap_burn_steps(messages)
    assert len(burns) == 1
    assert burns[0]["tool_call_truncated"] is True


def test_no_burn_below_the_cap_or_without_step_finish():
    below = [_assistant([{"type": "step-start"}, {"type": "reasoning"},
                         _finish(PER_STEP_OUTPUT_CAP - 1)])]
    assert cap_burn_steps(below) == []
    # Some dying turns never get a step-finish part: unmeasurable, not
    # guessed.
    unfinished = [_assistant([{"type": "step-start"},
                              {"type": "reasoning"}])]
    assert cap_burn_steps(unfinished) == []
    assert cap_burn_steps([]) == []
    assert cap_burn_steps(
        [{"info": {"role": "user"}, "parts": []}]) == []


def test_burns_are_found_in_any_step_of_any_assistant_message():
    # ICRA 2026-07-13: the burn sat one assistant message BEFORE a short
    # stub step and the final-step-only detector labeled the turn
    # short_empty. Whether a burn was fatal or self-recovered is the
    # caller's context (did the dispatch write anything), so a burn
    # followed by a healthy tool step is still reported.
    healed = [_assistant([
        {"type": "step-start"}, {"type": "reasoning"},
        _finish(PER_STEP_OUTPUT_CAP),
        {"type": "step-start"}, {"type": "tool", "tool": "write"},
        _finish(150)])]
    assert len(cap_burn_steps(healed)) == 1
    earlier_message = [
        _assistant([{"type": "step-start"}, {"type": "reasoning"},
                    _finish(PER_STEP_OUTPUT_CAP)]),
        _assistant([{"type": "step-start"}, {"type": "reasoning"},
                    _finish(166)]),
    ]
    assert len(cap_burn_steps(earlier_message)) == 1
    # The inverse: a short first step, then the cap death.
    died = [_assistant([
        {"type": "step-start"}, {"type": "tool", "tool": "read"},
        _finish(150),
        {"type": "step-start"}, {"type": "reasoning"},
        _finish(PER_STEP_OUTPUT_CAP)])]
    assert len(cap_burn_steps(died)) == 1


def test_no_write_cap_burn_dispatch_emits_the_dedicated_event(
        tmp_path, monkeypatch):
    state = make_state(tmp_path / "run")
    monkeypatch.setattr(run_pipeline, "dispatch_agent",
                        lambda **kw: _result())
    fetched = []

    def _messages(session_id, *, port):
        fetched.append(session_id)
        return [_assistant(
            [{"type": "step-start"}, {"type": "reasoning"},
             _finish(PER_STEP_OUTPUT_CAP)])]

    monkeypatch.setattr(
        run_pipeline.opencode_client, "fetch_session_messages",
        _messages)
    _dispatch_with_scope_check(
        state=state, agent="r2c-test-producer", prompt="p", timeout_s=5,
        writeable_paths_override=WRITEABLE)
    truncated = [e for e in _events(state)
                 if e["event_type"] == "truncated_turn_detected"]
    assert truncated[0]["details"]["turn_shape"] == "cap_burn"
    assert truncated[0]["details"]["final_step_output_tokens"] == \
        PER_STEP_OUTPUT_CAP
    cap_events = [e for e in _events(state)
                  if e["event_type"] == "cap_burn_turn_detected"]
    assert len(cap_events) == 1
    assert cap_events[0]["details"]["session_id"] == "fake-session"
    assert fetched == ["fake-session"]
    usage = json.loads(
        (state.paths.pipeline_dir / "token_usage.json").read_text())
    assert usage["dispatches_counted"] == 1


def test_written_dispatch_records_tokens_once_from_work_session(
        tmp_path, monkeypatch):
    state = make_state(tmp_path / "run")

    def _dispatch(**_kwargs):
        output = state.paths.run_dir / ".pipeline" / "out.json"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text("{}", encoding="utf-8")
        return _result()

    fetched = []

    def _messages(session_id, *, port):
        fetched.append(session_id)
        return [_assistant([
            {"type": "step-start"},
            {"type": "tool", "tool": "write"},
            _finish(150),
        ])]

    monkeypatch.setattr(run_pipeline, "dispatch_agent", _dispatch)
    monkeypatch.setattr(
        run_pipeline.opencode_client, "fetch_session_messages", _messages)

    _dispatch_with_scope_check(
        state=state, agent="r2c-test-producer", prompt="p", timeout_s=5,
        writeable_paths_override=WRITEABLE)

    assert fetched == ["fake-session"]
    usage = json.loads(
        (state.paths.pipeline_dir / "token_usage.json").read_text())
    assert usage["dispatches_counted"] == 1


def test_no_write_short_turn_is_short_empty_without_the_cap_event(
        tmp_path, monkeypatch):
    state = make_state(tmp_path / "run")
    monkeypatch.setattr(run_pipeline, "dispatch_agent",
                        lambda **kw: _result())
    monkeypatch.setattr(
        run_pipeline.opencode_client, "fetch_session_messages",
        lambda session_id, *, port: [_assistant(
            [{"type": "step-start"}, {"type": "text"}, _finish(84)])])
    _dispatch_with_scope_check(
        state=state, agent="r2c-test-producer", prompt="p", timeout_s=5,
        writeable_paths_override=WRITEABLE)
    truncated = [e for e in _events(state)
                 if e["event_type"] == "truncated_turn_detected"]
    assert truncated[0]["details"]["turn_shape"] == "short_empty"
    assert not [e for e in _events(state)
                if e["event_type"] == "cap_burn_turn_detected"]


def test_unreadable_session_reports_unmeasured_and_never_breaks_dispatch(
        tmp_path, monkeypatch):
    state = make_state(tmp_path / "run")
    monkeypatch.setattr(run_pipeline, "dispatch_agent",
                        lambda **kw: _result())

    def _boom(session_id, *, port):
        raise OpencodeClientError("session gone")

    monkeypatch.setattr(
        run_pipeline.opencode_client, "fetch_session_messages", _boom)
    result = _dispatch_with_scope_check(
        state=state, agent="r2c-test-producer", prompt="p", timeout_s=5,
        writeable_paths_override=WRITEABLE)
    assert result.completed is True
    truncated = [e for e in _events(state)
                 if e["event_type"] == "truncated_turn_detected"]
    assert truncated[0]["details"]["turn_shape"] == "unmeasured"
    assert truncated[0]["details"]["final_step_output_tokens"] is None
    assert not [e for e in _events(state)
                if e["event_type"] == "cap_burn_turn_detected"]


# ---------------------------------------------------------------------------
# Graceful out-of-scope-write recovery — branch A (corrective re-dispatch).
# Branch B (benign extra write → revert + continue silently) pre-existed; these
# pin the new wander path and the two fixes folded in with it (fix 1: a
# completeness-check's side-effect file is not deleted; fix 2: a tuple-returning
# check's failure is not read as a pass). See
# the out of scope recovery design note (internal, not shipped).
# ---------------------------------------------------------------------------


SCOPED_OUT = [".pipeline/out.json"]


def _out_check(state) -> bool:
    return (state.paths.run_dir / ".pipeline" / "out.json").is_file()


def test_branch_a_wander_recovers_on_corrective_redispatch(tmp_path, monkeypatch):
    """Opted-in agent wanders (writes a stray, no valid output); reverting the
    stray doesn't help; one corrective re-dispatch produces valid in-scope
    output → continue. Loud: attempted+succeeded events + assumptions.md."""
    state = make_state(tmp_path / "run")
    calls = {"n": 0}

    def fake_dispatch(*, state, agent, prompt, timeout_s):
        calls["n"] += 1
        if "Scope correction" in prompt:
            (state.paths.run_dir / ".pipeline" / "out.json").write_text("{}")
        else:
            (state.paths.run_dir / "rogue.txt").write_text("stray")
        return _result()

    monkeypatch.setattr(run_pipeline, "dispatch_agent", fake_dispatch)
    result = _dispatch_with_scope_check(
        state=state, agent="r2c-stage-reviewer", prompt="do the review", timeout_s=5,
        recovery_check_fn=_out_check, writeable_paths_override=SCOPED_OUT,
        allow_corrective_redispatch=True)

    assert result.completed is True
    assert calls["n"] == 2  # one wander + exactly one corrective re-dispatch
    assert not (state.paths.run_dir / "rogue.txt").exists()  # stray removed
    assert (state.paths.run_dir / ".pipeline" / "out.json").is_file()  # output kept
    types = [e["event_type"] for e in _events(state)]
    assert "out_of_scope_corrective_redispatch_attempted" in types
    assert "out_of_scope_corrective_redispatch_succeeded" in types
    md = state.paths.run_dir / run_layout.ASSUMPTIONS_MD
    assert md.is_file() and "Out-of-scope write recovered" in md.read_text()


def test_branch_a_halts_when_corrective_also_invalid(tmp_path, monkeypatch):
    """Agent wanders again on the corrective re-dispatch → revert both strays
    and halt (re-dispatch is once, not a loop)."""
    state = make_state(tmp_path / "run")
    calls = {"n": 0}

    def fake_dispatch(*, state, agent, prompt, timeout_s):
        calls["n"] += 1
        (state.paths.run_dir / f"rogue{calls['n']}.txt").write_text("stray")
        return _result()

    monkeypatch.setattr(run_pipeline, "dispatch_agent", fake_dispatch)
    with pytest.raises(run_pipeline.OutOfScopeWritesError):
        _dispatch_with_scope_check(
            state=state, agent="r2c-method-coder", prompt="p", timeout_s=5,
            recovery_check_fn=_out_check, writeable_paths_override=SCOPED_OUT,
            allow_corrective_redispatch=True)

    assert calls["n"] == 2  # first + one corrective, then halt
    assert not list(state.paths.run_dir.glob("rogue*.txt"))  # both strays reverted
    types = [e["event_type"] for e in _events(state)]
    assert "out_of_scope_corrective_redispatch_attempted" in types
    exhausted = [e for e in _events(state)
                 if e["event_type"] == "out_of_scope_corrective_redispatch_exhausted"]
    assert exhausted and exhausted[-1]["details"]["reason"] == "still_invalid"


def test_branch_a_whole_run_cap_halts_recurring_wanderer(tmp_path, monkeypatch):
    """Two prior corrective attempts already logged → a third wander halts
    immediately, with NO further re-dispatch (whole-run cap, resume-safe via
    the event log)."""
    state = make_state(tmp_path / "run")
    for _ in range(2):
        run_pipeline._append_run_event(
            state.paths, "out_of_scope_corrective_redispatch_attempted",
            details={"agent": "r2c-method-coder"})
    calls = {"n": 0}

    def fake_dispatch(*, state, agent, prompt, timeout_s):
        calls["n"] += 1
        (state.paths.run_dir / "rogue.txt").write_text("stray")
        return _result()

    monkeypatch.setattr(run_pipeline, "dispatch_agent", fake_dispatch)
    with pytest.raises(run_pipeline.OutOfScopeWritesError):
        _dispatch_with_scope_check(
            state=state, agent="r2c-method-coder", prompt="p", timeout_s=5,
            recovery_check_fn=_out_check, writeable_paths_override=SCOPED_OUT,
            allow_corrective_redispatch=True)

    assert calls["n"] == 1  # cap blocked the corrective re-dispatch
    assert not (state.paths.run_dir / "rogue.txt").exists()  # stray still reverted
    exhausted = [e for e in _events(state)
                 if e["event_type"] == "out_of_scope_corrective_redispatch_exhausted"]
    assert exhausted and exhausted[-1]["details"]["reason"] == "whole_run_cap"


def test_wander_hard_halts_when_not_opted_in(tmp_path, monkeypatch):
    """Default (allow_corrective_redispatch=False): a wander hard-halts with NO
    re-dispatch, even with a recovery_check_fn present. This is also how Stage-1
    canonical dispatches keep handing their chunk-mode fallback to the stage
    handler instead of being corrected here."""
    state = make_state(tmp_path / "run")
    calls = {"n": 0}

    def fake_dispatch(*, state, agent, prompt, timeout_s):
        calls["n"] += 1
        (state.paths.run_dir / "rogue.txt").write_text("stray")
        return _result()

    monkeypatch.setattr(run_pipeline, "dispatch_agent", fake_dispatch)
    with pytest.raises(run_pipeline.OutOfScopeWritesError):
        _dispatch_with_scope_check(
            state=state, agent="r2c-decomposer", prompt="p", timeout_s=5,
            recovery_check_fn=_out_check, writeable_paths_override=SCOPED_OUT)

    assert calls["n"] == 1  # no corrective re-dispatch
    assert not [e for e in _events(state) if "corrective" in e["event_type"]]


def test_branch_a_restores_modified_preexisting_file(tmp_path, monkeypatch):
    """A wander that CLOBBERS a pre-existing out-of-scope file is restored to
    its original bytes (not deleted) along the corrective path."""
    state = make_state(tmp_path / "run")
    keep = state.paths.run_dir / "keep.txt"
    keep.write_text("original")
    calls = {"n": 0}

    def fake_dispatch(*, state, agent, prompt, timeout_s):
        calls["n"] += 1
        if "Scope correction" in prompt:
            (state.paths.run_dir / ".pipeline" / "out.json").write_text("{}")
        else:
            keep.write_text("CLOBBERED")
        return _result()

    monkeypatch.setattr(run_pipeline, "dispatch_agent", fake_dispatch)
    _dispatch_with_scope_check(
        state=state, agent="r2c-method-coder", prompt="p", timeout_s=5,
        recovery_check_fn=_out_check, writeable_paths_override=SCOPED_OUT,
        allow_corrective_redispatch=True)

    assert keep.is_file() and keep.read_text() == "original"  # restored, not deleted


def test_branch_a_skipped_for_unrestorable_violation(tmp_path, monkeypatch):
    """An out-of-scope MODIFICATION of a pre-existing file too large to snapshot
    is unrestorable → hard halt, no re-dispatch, file left as-is (we cannot
    restore its bytes, so we do not pretend to recover)."""
    monkeypatch.setattr(run_pipeline, "_CONTENT_SNAPSHOT_MAX_BYTES", 4)
    state = make_state(tmp_path / "run")
    big = state.paths.run_dir / "big.bin"
    big.write_text("0123456789")  # 10 bytes > 4-byte cap → excluded from content snapshot
    calls = {"n": 0}

    def fake_dispatch(*, state, agent, prompt, timeout_s):
        calls["n"] += 1
        big.write_text("MODIFIED!!")
        return _result()

    monkeypatch.setattr(run_pipeline, "dispatch_agent", fake_dispatch)
    with pytest.raises(run_pipeline.OutOfScopeWritesError):
        _dispatch_with_scope_check(
            state=state, agent="r2c-method-coder", prompt="p", timeout_s=5,
            recovery_check_fn=_out_check, writeable_paths_override=SCOPED_OUT,
            allow_corrective_redispatch=True)

    assert calls["n"] == 1  # no re-dispatch
    assert big.read_text() == "MODIFIED!!"  # not reverted; we halted instead
    assert not [e for e in _events(state) if "corrective" in e["event_type"]]


def test_branch_b_benign_extra_write_continues_silently(tmp_path, monkeypatch):
    """Branch B unchanged: valid output PLUS a benign stray → drop the stray and
    continue, with NO corrective re-dispatch."""
    state = make_state(tmp_path / "run")
    calls = {"n": 0}

    def fake_dispatch(*, state, agent, prompt, timeout_s):
        calls["n"] += 1
        (state.paths.run_dir / ".pipeline" / "out.json").write_text("{}")
        (state.paths.run_dir / "extra.txt").write_text("benign")
        return _result()

    monkeypatch.setattr(run_pipeline, "dispatch_agent", fake_dispatch)
    result = _dispatch_with_scope_check(
        state=state, agent="r2c-method-coder", prompt="p", timeout_s=5,
        recovery_check_fn=_out_check, writeable_paths_override=SCOPED_OUT,
        allow_corrective_redispatch=True)

    assert result.completed is True
    assert calls["n"] == 1  # no re-dispatch
    assert not (state.paths.run_dir / "extra.txt").exists()  # benign stray dropped
    assert (state.paths.run_dir / ".pipeline" / "out.json").is_file()
    assert not [e for e in _events(state) if "corrective" in e["event_type"]]


def test_branch_a_does_not_delete_check_side_effect_file(tmp_path, monkeypatch):
    """Critical fix 1: the completeness check is not side-effect-free. A file the
    check itself renders (here `rendered.out`, standing in for `notebook.ipynb`)
    must NOT be mistaken for a fresh stray and deleted on the corrective path —
    detection runs BEFORE the check."""
    state = make_state(tmp_path / "run")
    rendered = state.paths.run_dir / "rendered.out"  # NOT in the allowlist

    def check(s):
        draft = s.paths.run_dir / ".pipeline" / "draft.txt"
        if draft.is_file():
            rendered.write_text("rendered from draft")  # side effect of the check
            return True
        return False

    calls = {"n": 0}

    def fake_dispatch(*, state, agent, prompt, timeout_s):
        calls["n"] += 1
        if "Scope correction" in prompt:
            (state.paths.run_dir / ".pipeline" / "draft.txt").write_text("draft")
        else:
            (state.paths.run_dir / "rogue.txt").write_text("stray")
        return _result()

    monkeypatch.setattr(run_pipeline, "dispatch_agent", fake_dispatch)
    result = _dispatch_with_scope_check(
        state=state, agent="r2c-notebook-generator", prompt="p", timeout_s=5,
        recovery_check_fn=check, writeable_paths_override=[".pipeline/draft.txt"],
        allow_corrective_redispatch=True)

    assert result.completed is True
    assert calls["n"] == 2
    assert rendered.is_file() and rendered.read_text() == "rendered from draft"
    assert not (state.paths.run_dir / "rogue.txt").exists()


def test_tuple_returning_failed_check_does_not_falsely_recover(tmp_path, monkeypatch):
    """Critical fix 2: a (False, "...") tuple is truthy; before the fix it read
    as a PASS and falsely recovered. A correctly-failing tuple check on a
    not-opted-in dispatch must HALT."""
    state = make_state(tmp_path / "run")

    def fake_dispatch(*, state, agent, prompt, timeout_s):
        (state.paths.run_dir / "rogue.txt").write_text("stray")
        return _result()

    monkeypatch.setattr(run_pipeline, "dispatch_agent", fake_dispatch)
    with pytest.raises(run_pipeline.OutOfScopeWritesError):
        _dispatch_with_scope_check(
            state=state, agent="r2c-method-coder", prompt="p", timeout_s=5,
            recovery_check_fn=lambda s: (False, "still broken"),
            writeable_paths_override=SCOPED_OUT)


def test_tuple_returning_passed_check_recovers(tmp_path, monkeypatch):
    """The other half of fix 2: a (True, "") tuple reads as a PASS, so branch B
    still recovers a benign extra write when the real verdict is good."""
    state = make_state(tmp_path / "run")

    def fake_dispatch(*, state, agent, prompt, timeout_s):
        (state.paths.run_dir / ".pipeline" / "out.json").write_text("{}")
        (state.paths.run_dir / "extra.txt").write_text("benign")
        return _result()

    monkeypatch.setattr(run_pipeline, "dispatch_agent", fake_dispatch)
    result = _dispatch_with_scope_check(
        state=state, agent="r2c-method-coder", prompt="p", timeout_s=5,
        recovery_check_fn=lambda s: (True, "ok") if _out_check(s) else (False, "no"),
        writeable_paths_override=SCOPED_OUT)
    assert result.completed is True
    assert not (state.paths.run_dir / "extra.txt").exists()


def test_branch_a_halts_when_stray_matches_another_agent_owned_path(tmp_path, monkeypatch):
    """A wander whose stray lands on ANOTHER agent's owned output path (here a
    reviewer creating `method/method.py`, which method-coder owns) is too serious
    to auto-recover: halt for investigation, strays preserved, NO re-dispatch."""
    state = make_state(tmp_path / "run")
    calls = {"n": 0}

    def fake_dispatch(*, state, agent, prompt, timeout_s):
        calls["n"] += 1
        method = state.paths.run_dir / "method" / "method.py"
        method.parent.mkdir(parents=True, exist_ok=True)
        method.write_text("# wandered onto method-coder's file")
        return _result()

    monkeypatch.setattr(run_pipeline, "dispatch_agent", fake_dispatch)
    with pytest.raises(run_pipeline.OutOfScopeWritesError):
        _dispatch_with_scope_check(
            state=state, agent="r2c-stage-reviewer", prompt="p", timeout_s=5,
            recovery_check_fn=_out_check, writeable_paths_override=SCOPED_OUT,
            allow_corrective_redispatch=True)

    assert calls["n"] == 1  # no corrective re-dispatch
    assert (state.paths.run_dir / "method" / "method.py").is_file()  # stray preserved for investigation
    assert not [e for e in _events(state) if "corrective" in e["event_type"]]


def test_branch_a_opencode_error_reverts_and_propagates(tmp_path, monkeypatch):
    """A plain OpencodeClientError (not a transport subclass) on the corrective
    re-dispatch must be caught: revert any new strays, log `_exhausted`, and
    re-raise the original error so the stage handler halts — not leave strays on
    disk with the error escaping uncaught."""
    state = make_state(tmp_path / "run")
    calls = {"n": 0}

    def fake_dispatch(*, state, agent, prompt, timeout_s):
        calls["n"] += 1
        if "Scope correction" in prompt:
            (state.paths.run_dir / "corrective_stray.txt").write_text("stray")
            raise OpencodeClientError("HTTP 500 from opencode")
        (state.paths.run_dir / "rogue.txt").write_text("stray")
        return _result()

    monkeypatch.setattr(run_pipeline, "dispatch_agent", fake_dispatch)
    with pytest.raises(OpencodeClientError):  # original error, not OutOfScopeWritesError
        _dispatch_with_scope_check(
            state=state, agent="r2c-method-coder", prompt="p", timeout_s=5,
            recovery_check_fn=_out_check, writeable_paths_override=SCOPED_OUT,
            allow_corrective_redispatch=True)

    assert calls["n"] == 2  # first wander + the failing corrective
    assert not (state.paths.run_dir / "corrective_stray.txt").exists()  # reverted
    assert not (state.paths.run_dir / "rogue.txt").exists()  # first wander reverted earlier
    exhausted = [e for e in _events(state)
                 if e["event_type"] == "out_of_scope_corrective_redispatch_exhausted"]
    assert exhausted and exhausted[-1]["details"]["reason"] == "dispatch_error"


def test_branch_a_opt_in_limited_to_audited_dispatches():
    """Source pin: branch A is opt-in (default off in the signature) and only the
    audited dispatches opt in. The halt-judge opts in per the
    _dispatch_with_scope_check docstring; its owned-path strays are reverted
    before halting (audit 2026-06-25 J1) rather than corrective-re-dispatched.
    The paper-fidelity reviewer's STAGE-4 dispatch opted in 2026-07-03 (the maintainer
    signed off after the detr-distill sidecar-overwrite wander; reverses the
    2026-06-25 exclusion for that one dispatch — its fix/incremental dispatches
    keep the hard halt). The architecture-coder's 2 dispatches opted in
    2026-07-10 (queue item 19, the sketch the maintainer reviewed: mirror the
    method-coder's drift recovery after the detr-distill run-root _cache
    halt). Stage-1 canonical, notebook-generator and diagnostician
    dispatches keep the hard halt."""
    from pathlib import Path
    src = Path(run_pipeline.__file__).read_text(encoding="utf-8")
    assert "allow_corrective_redispatch: bool = False" in src, (
        "branch A must be opt-in (default off)")
    assert src.count("allow_corrective_redispatch=True") == 12, (
        "exactly the 4 stage-reviewer (generic + retry + stage_1 + resolution "
        "re-ask) + 2 method-coder + 2 arch-coder + 1 shared halt-judge "
        "builder (B-06 trio merge: the validator, reviewer-findings, and "
        "element-test ownership dispatches all route through "
        "_dispatch_judge_with_task, so the three audited judge opt-ins are "
        "one source site) + 1 paper-fidelity stage-4 + 2 test-generator "
        "(R2C-024: generation + fix mode, mirroring the method-coder's "
        "drift recovery with a parseable-test-module check) dispatches may "
        "opt in")
    # The three named judge entry points must still exist (two are
    # string-patched by tests) and must all delegate to the one audited
    # shared builder — a fourth judge path that dispatches directly would
    # silently widen the opt-in surface this pin audits.
    for name in ("_dispatch_halt_judge", "_dispatch_halt_judge_for_findings",
                 "_dispatch_halt_judge_for_element_test"):
        assert hasattr(run_pipeline, name), name
        import inspect
        body = inspect.getsource(getattr(run_pipeline, name))
        assert "_dispatch_judge_with_task(" in body, (
            f"{name} must route through the shared audited builder")
        assert "allow_corrective_redispatch" not in body, (
            f"{name} must not carry its own opt-in site")


# ---------------------------------------------------------------------------
# Notice-delivery retries
# ---------------------------------------------------------------------------


def _patch_flaky_dispatch(monkeypatch, fail_times: int) -> dict:
    calls = {"n": 0}

    def fake_dispatch_and_wait(**kwargs):
        calls["n"] += 1
        if calls["n"] <= fail_times:
            raise OpencodeClientError("session busy")
        return _result()

    monkeypatch.setattr(run_pipeline, "dispatch_and_wait",
                        fake_dispatch_and_wait)
    monkeypatch.setattr(run_pipeline.time, "sleep", lambda s: None)
    return calls


def test_todo_post_retries_once_then_succeeds(tmp_path, monkeypatch):
    state = make_state(tmp_path / "run")
    calls = _patch_flaky_dispatch(monkeypatch, fail_times=1)
    _post_todos_worker(state, "update")
    assert calls["n"] == 2


def test_todo_post_gives_up_after_one_retry(tmp_path, monkeypatch):
    # Two failures: logged and dropped, never raised (daemon-thread path).
    # With no workspace there is nothing to rediscover, so the ladder ends
    # at two attempts exactly as before the breaker existed.
    state = make_state(tmp_path / "run")
    calls = _patch_flaky_dispatch(monkeypatch, fail_times=10)
    _post_todos_worker(state, "update")
    assert calls["n"] == 2


def test_halt_notice_retries_once_then_succeeds(tmp_path, monkeypatch):
    state = make_state(tmp_path / "run")
    calls = _patch_flaky_dispatch(monkeypatch, fail_times=1)
    _post_halt_to_session(state, "stage_2b", "reason text",
                          state.paths.pipeline_dir / "stage_2b.halt")
    assert calls["n"] == 2


def test_halt_notice_gives_up_after_one_retry(tmp_path, monkeypatch):
    state = make_state(tmp_path / "run")
    calls = _patch_flaky_dispatch(monkeypatch, fail_times=10)
    _post_halt_to_session(state, "stage_2b", "reason text",
                          state.paths.pipeline_dir / "stage_2b.halt")
    assert calls["n"] == 2


def test_end_of_run_notice_renders_manifest_status_without_false_degraded(tmp_path):
    state = make_state(tmp_path / "run")
    notice = _render_end_of_run_notice(
        state.paths,
        run_status="completed",
        manifest_status="passed",
        delivery={"label": "verified", "reasons": [], "disclosures": []},
    )
    assert "Manifest status: `passed`." in notice
    assert "Delivery label: verified." in notice
    assert "No delivery-blocking or delivery-demoting findings" in notice
    assert "degraded" not in notice


def test_end_of_run_notice_scrubs_probe_and_finding_codes(tmp_path):
    state = make_state(tmp_path / "run")
    notice = _render_end_of_run_notice(
        state.paths,
        run_status="degraded",
        manifest_status="degraded",
        delivery={
            "label": "draft",
            "reasons": [{
                "message": "US-8 failed after F001 remained stale",
            }],
            "disclosures": [],
        },
    )
    assert "notebook prose numbers match the params table" in notice
    assert "the linked finding" in notice
    assert "US-8" not in notice
    assert "F001" not in notice


def test_end_of_run_notice_prioritizes_blocked_manifest_diagnostics(tmp_path):
    state = make_state(tmp_path / "run")
    run_layout.run_path(
        state.paths.run_dir, run_layout.FINAL_MANIFEST_JSON,
    ).write_text(json.dumps({
        "diagnostics": ["REPORT.md: required delivery artifact missing"],
    }) + "\n", encoding="utf-8")

    notice = _render_end_of_run_notice(
        state.paths,
        run_status="failed",
        manifest_status="blocked",
        delivery={
            "label": "draft",
            "reasons": [{"message": "US-8 remained stale"}],
            "disclosures": [],
        },
    )

    manifest_pos = notice.index("Manifest note: REPORT.md")
    finding_pos = notice.index(
        'Finding: the "notebook prose numbers match the params table" check')
    assert manifest_pos < finding_pos


def test_end_of_run_notice_retries_once_then_succeeds(tmp_path, monkeypatch):
    state = make_state(tmp_path / "run")
    calls = _patch_flaky_dispatch(monkeypatch, fail_times=1)
    _post_end_of_run_notice(
        state,
        run_status="completed",
        manifest_status="passed",
        delivery={"label": "verified", "reasons": [], "disclosures": []},
    )
    assert calls["n"] == 2


def test_end_of_run_notice_gives_up_on_unexpected_exception(tmp_path, monkeypatch):
    state = make_state(tmp_path / "run")
    calls = {"n": 0}

    def fake_dispatch_and_wait(**kwargs):
        calls["n"] += 1
        raise ValueError("malformed response")

    monkeypatch.setattr(run_pipeline, "dispatch_and_wait", fake_dispatch_and_wait)
    monkeypatch.setattr(run_pipeline.time, "sleep", lambda s: None)

    _post_end_of_run_notice(
        state,
        run_status="completed",
        manifest_status="passed",
        delivery={"label": "verified", "reasons": [], "disclosures": []},
    )

    assert calls["n"] == 2


# ---------------------------------------------------------------------------
# Retired shared-session recovery (dispatch_agent)
# ---------------------------------------------------------------------------


def test_summary_collision_does_not_retry_in_fresh_session(tmp_path, monkeypatch):
    # Session summarization was a long shared-transcript failure mode. A fresh
    # work session must not carry its same-session retry assumption forward:
    # an unexpected assistant error is loud and gets the normal failure path.
    state = make_state(tmp_path / "run")
    calls = {"n": 0}

    def fake_dispatch_and_wait(**kwargs):
        calls["n"] += 1
        raise OpencodeClientError(
            'dispatch produced an assistant error: {"name": '
            '"UnknownError", "data": {"message": "Tool call not '
            'allowed while generating summary: read"}}')

    monkeypatch.setattr(run_pipeline, "dispatch_and_wait",
                        fake_dispatch_and_wait)
    with pytest.raises(OpencodeClientError, match="generating summary"):
        run_pipeline.dispatch_agent(
            state=state, agent="r2c-method-coder", prompt="p", timeout_s=5)
    assert calls["n"] == 1


def test_other_assistant_errors_do_not_retry(tmp_path, monkeypatch):
    state = make_state(tmp_path / "run")
    calls = {"n": 0}

    def fake_dispatch_and_wait(**kwargs):
        calls["n"] += 1
        raise OpencodeClientError(
            'dispatch produced an assistant error: {"name": "APIError", '
            '"data": {"message": "The model `X` does not exist."}}')

    monkeypatch.setattr(run_pipeline, "dispatch_and_wait",
                        fake_dispatch_and_wait)
    with pytest.raises(OpencodeClientError):
        run_pipeline.dispatch_agent(
            state=state, agent="r2c-method-coder", prompt="p", timeout_s=5)
    assert calls["n"] == 1


# ---------------------------------------------------------------------------
# Per-dispatch sessions (dispatch_agent, Section 4.1). Every agent runs in a
# FRESH session nested under the TUI session so a prior stage's transcript
# cannot bleed into its turn (the GBALD 2026-06-29 stage_3a drift).
# ---------------------------------------------------------------------------


def test_dispatch_always_creates_fresh_nested_session(tmp_path, monkeypatch):
    state = make_state(tmp_path / "run")
    state.workspace = "/ws"
    created = {"n": 0, "kwargs": None}
    dispatched = {"session_id": None}

    def fake_create_session(*, title, parent_id, directory, port):
        created["n"] += 1
        created["kwargs"] = {"title": title, "parent_id": parent_id,
                             "directory": directory, "port": port}
        return "ses_fresh_1"

    def fake_dispatch_and_wait(*, session_id, agent, prompt, model, port, timeout_s):
        dispatched["session_id"] = session_id
        return _result()

    monkeypatch.setattr(run_pipeline, "create_session", fake_create_session)
    monkeypatch.setattr(run_pipeline, "dispatch_and_wait", fake_dispatch_and_wait)
    run_pipeline.dispatch_agent(
        state=state, agent="r2c-notebook-generator", prompt="p", timeout_s=5)

    assert created["n"] == 1                                  # one fresh session
    assert dispatched["session_id"] == "ses_fresh_1"          # dispatched into it
    assert created["kwargs"]["parent_id"] == "fake-session"   # nested under the TUI session
    assert created["kwargs"]["directory"] == "/ws"            # workspace pinned
    assert "r2c-notebook-generator" in created["kwargs"]["title"]
    started = [e for e in _events(state)
               if e["event_type"] == "agent_dispatch_started"]
    assert started[0]["details"]["per_dispatch_session"] is True
    # The completed event records the work session id for traceability.
    completed = [e for e in _events(state)
                 if e["event_type"] == "agent_dispatch_completed"]
    assert completed and completed[0]["details"]["session_id"] == "ses_fresh_1"

def test_work_session_create_failure_fails_loud(tmp_path, monkeypatch):
    # A POST /session failure propagates as a dispatch error (logged as
    # agent_dispatch_failed) rather than silently falling back to the parent
    # session, which would reintroduce cross-stage context bleed.
    state = make_state(tmp_path / "run")

    def fake_create_session(**kwargs):
        raise OpencodeClientError("POST /session returned 500: boom")

    def fake_dispatch_and_wait(**kwargs):
        raise AssertionError("dispatch must not run when session-create fails")

    monkeypatch.setattr(run_pipeline, "create_session", fake_create_session)
    monkeypatch.setattr(run_pipeline, "dispatch_and_wait", fake_dispatch_and_wait)
    with pytest.raises(OpencodeClientError):
        run_pipeline.dispatch_agent(
            state=state, agent="r2c-method-coder", prompt="p", timeout_s=5)
    failed = [e for e in _events(state)
              if e["event_type"] == "agent_dispatch_failed"]
    assert failed, "session-create failure must be logged as a dispatch failure"


# ---------------------------------------------------------------------------
# Dispatch-prompt assembly: the producer anti-review clause must not reach a
# review-only agent (deep-batch-active-learning, 2026-06-15 stage_2x halt).
# ---------------------------------------------------------------------------

from dispatch_templates import WRITEABLE_PATHS, format_writeable_paths_block  # noqa: E402


def test_producer_block_keeps_anti_review_clause():
    block = format_writeable_paths_block(WRITEABLE_PATHS["r2c-method-coder"])
    # Producers must be told not to inline the reviewer's job (B-004).
    assert "your entire job" in block
    assert "stage_review_focus" in block


@pytest.mark.parametrize("agent", ["r2c-stage-reviewer", "r2c-paper-fidelity-reviewer"])
def test_review_only_agent_block_omits_producer_clause(agent):
    """A review-only agent's listed output IS a review file; the producer-framed
    clause ("producing your listed file(s) is your entire job; reviewing them is
    a separate agent's dispatch") inverts its task — it drove the stage_2x
    reviewer to author a notebook.py. The block keeps the generic scope
    constraint and names the agent's own output, but drops the producer clause."""
    block = format_writeable_paths_block(WRITEABLE_PATHS[agent])
    assert "You MUST only create or modify files matching the patterns" in block
    assert WRITEABLE_PATHS[agent][0] in block
    assert "your entire job" not in block
    assert "reviewing them is" not in block


def test_diagnosis_and_decision_producers_keep_clause():
    """Smoke-diagnostician and halt-judge PRODUCE a diagnosis/decision (not a
    review), so producing their listed file is genuinely their whole job — they
    keep the clause. Guards the predicate against over-broadening to any
    JSON-only agent."""
    for agent in ("r2c-smoke-diagnostician", "r2c-halt-judge"):
        assert "your entire job" in format_writeable_paths_block(WRITEABLE_PATHS[agent])


def test_fidelity_reviewer_sidecar_overwrite_recovers_via_branch_a(
    tmp_path, monkeypatch
):
    """The detr-distill 2026-07-03 wander, end to end through the fidelity
    reviewer's own wiring: the reviewer overwrites the completed
    method_explanations.json sidecar (a driver-managed merged artifact, in
    NO agent's allowlist) instead of writing review_report.json. Branch A
    restores the sidecar byte-for-byte from the content snapshot and the
    one corrective re-dispatch produces the review."""
    from run_pipeline import _dispatch_paper_fidelity_reviewer

    state = make_state(tmp_path / "run")
    sidecar = state.paths.pipeline_dir / "method_explanations.json"
    original = json.dumps({"explanations": [{"id": f"e{i}"} for i in range(12)]})
    sidecar.write_text(original, encoding="utf-8")
    calls = {"n": 0}

    def fake_dispatch(*, state, agent, prompt, timeout_s):
        calls["n"] += 1
        if "Scope correction" in prompt:
            (state.paths.pipeline_dir / "review_report.json").write_text(
                json.dumps({"findings": []}), encoding="utf-8")
        else:
            sidecar.write_text(
                json.dumps({"equation_analysis": "reviewer drift"}),
                encoding="utf-8")
        return _result()

    monkeypatch.setattr(run_pipeline, "dispatch_agent", fake_dispatch)
    result = _dispatch_paper_fidelity_reviewer(state)

    assert result.completed is True
    assert calls["n"] == 2
    assert sidecar.read_text(encoding="utf-8") == original
    assert (state.paths.pipeline_dir / "review_report.json").is_file()


def test_end_of_run_notice_names_the_front_door_first(tmp_path):
    state = make_state(tmp_path / "run")
    notice = _render_end_of_run_notice(
        state.paths,
        run_status="completed",
        manifest_status="passed",
        delivery={"label": "verified", "reasons": [], "disclosures": []},
    )
    lines = notice.splitlines()
    # Failure-path spec §5: the first content line after the heading names
    # the one front-door path, repo-relative.
    assert lines[2] == ("Start at `run/REPORT.md` — every other artifact is "
                        "linked from there.")
    assert notice.index("REPORT.md") < notice.index("Manifest status")


def test_halt_notice_names_the_front_door(tmp_path):
    from run_pipeline import _render_halt_notice
    halt_path = tmp_path / "run" / ".pipeline" / "stage_2b.halt"
    notice = _render_halt_notice(
        "stage_2b", "arch validator failed", halt_path, None)
    front_door = tmp_path / "run" / "REPORT.md"
    assert f"Start at `{front_door}`" in notice
    assert "stopped-run summary is at the top" in notice
    # The front-door line leads the researcher story, before the details.
    assert notice.index("REPORT.md") < notice.index("<details>")


# ---------------------------------------------------------------------------
# Write-first missing-output retries (stage 4 + judge share the shape)
# ---------------------------------------------------------------------------


def test_stage_4_missing_output_retry_carries_write_first_preamble(
    tmp_path, monkeypatch
):
    """When the fidelity reviewer's first dispatch completes with no
    review_report.json, the retry must NOT be an identical prompt — it
    carries the shared write-first preamble (detr 2026-07-04: identical
    re-dispatches die identically when the first turn burned its output
    budget on pre-write reasoning). First dispatch stays preamble-free."""
    from run_pipeline import run_stage_4

    state = make_state(tmp_path / "run")
    prompts: list[str] = []

    def fake_dispatch(*, state, agent, prompt, timeout_s):
        prompts.append(prompt)
        if len(prompts) == 2:
            (state.paths.pipeline_dir / "review_report.json").write_text(
                "{}", encoding="utf-8")
        return _result()

    monkeypatch.setattr(run_pipeline, "dispatch_agent", fake_dispatch)
    monkeypatch.setattr(
        run_pipeline, "_run_review_report_validator", lambda s: (True, ""))

    result = run_stage_4(state)
    assert result.status == "completed"
    assert len(prompts) == 2
    assert "WRITE FIRST" not in prompts[0], (
        "initial stage-4 dispatch must not carry the retry preamble")
    assert "WRITE FIRST" in prompts[1], (
        "missing-output retry must carry the write-first preamble")
    assert "FIRST tool call must be the Write" in prompts[1]
    assert "imperfect but honest review report" in prompts[1]
    # The original task survives unchanged after the preamble.
    assert "Your original task (unchanged) follows." in prompts[1]
    assert "Stage 4 — review the generated package" in prompts[1]



def test_runtime_data_cache_writes_are_not_scope_violations(tmp_path, monkeypatch):
    """BADGE matrix row 2026-07-05: the arch-coder's designed
    self-verification ran data.py, whose first call downloaded MNIST into
    method/example_data/_cache/ on the wiped run dir, and the scope check
    hard-halted a healthy dispatch over nine runtime cache artifacts.
    Runtime cache is side traffic like bytecode: exempt. Authored files
    outside the allowlist still halt."""
    from run_pipeline import OutOfScopeWritesError

    state = make_state(tmp_path / "run")

    def cache_writing_dispatch(*, state, agent, prompt, timeout_s):
        run_dir = state.paths.run_dir
        (run_dir / ".pipeline" / "out.json").write_text("{}")
        cache = run_dir / "method" / "example_data" / "_cache" / "_torchvision"
        cache.mkdir(parents=True)
        (cache / "train-images-idx3-ubyte").write_bytes(b"\x00")
        return _result()

    monkeypatch.setattr(run_pipeline, "dispatch_agent", cache_writing_dispatch)
    result = _dispatch_with_scope_check(
        state=state, agent="r2c-test-producer", prompt="p", timeout_s=5,
        writeable_paths_override=WRITEABLE)
    assert result.completed is True  # no OutOfScopeWritesError

    def authored_stray_dispatch(*, state, agent, prompt, timeout_s):
        run_dir = state.paths.run_dir
        (run_dir / ".pipeline" / "out.json").write_text("{}")
        (run_dir / "rogue.py").write_text("x = 1")
        return _result()

    state2 = make_state(tmp_path / "run2")
    monkeypatch.setattr(run_pipeline, "dispatch_agent", authored_stray_dispatch)
    try:
        _dispatch_with_scope_check(
            state=state2, agent="r2c-test-producer", prompt="p", timeout_s=5,
            writeable_paths_override=WRITEABLE)
        raise AssertionError("expected OutOfScopeWritesError for authored stray")
    except OutOfScopeWritesError as e:
        assert "rogue.py" in str(e)


def test_run_root_cache_stray_is_exempt_authored_stray_still_halts(
    tmp_path, monkeypatch,
):
    """detr-distill 2026-07-08 (queue item 19): the generated loader caches
    under `<caller path>/_cache` resolved against the working directory, so
    a contract-encouraged self-test wrote `_cache/synthetic_detection.pt`
    at the RUN ROOT and hard-halted a stage-2b dispatch whose declared
    outputs were all written. Any `_cache/` directory segment is runtime
    side traffic; authored files outside the allowlist still halt."""
    from run_pipeline import OutOfScopeWritesError, _is_runtime_data_cache

    # Unit surface: the exemption predicate.
    assert _is_runtime_data_cache("_cache/synthetic_detection.pt")
    assert _is_runtime_data_cache("method/example_data/_cache/x.pt")
    assert _is_runtime_data_cache("method/_cache/foo/bar.bin")
    assert not _is_runtime_data_cache("method/model.py")
    assert not _is_runtime_data_cache("_cache")  # a FILE named _cache
    assert not _is_runtime_data_cache("notes_cache/x.pt")  # substring only

    # Dispatch surface: root-level stray exempt, authored stray halts.
    state = make_state(tmp_path / "run")

    def root_cache_dispatch(*, state, agent, prompt, timeout_s):
        run_dir = state.paths.run_dir
        (run_dir / ".pipeline" / "out.json").write_text("{}")
        cache = run_dir / "_cache"
        cache.mkdir(parents=True)
        (cache / "synthetic_detection.pt").write_bytes(b"\x00")
        return _result()

    monkeypatch.setattr(run_pipeline, "dispatch_agent", root_cache_dispatch)
    result = _dispatch_with_scope_check(
        state=state, agent="r2c-test-producer", prompt="p", timeout_s=5,
        writeable_paths_override=WRITEABLE)
    assert result.completed is True

    def cache_plus_authored_dispatch(*, state, agent, prompt, timeout_s):
        run_dir = state.paths.run_dir
        (run_dir / ".pipeline" / "out.json").write_text("{}")
        (run_dir / "_cache").mkdir(parents=True)
        (run_dir / "_cache" / "x.pt").write_bytes(b"\x00")
        (run_dir / "helper.py").write_text("x = 1")
        return _result()

    state2 = make_state(tmp_path / "run2")
    monkeypatch.setattr(
        run_pipeline, "dispatch_agent", cache_plus_authored_dispatch)
    try:
        _dispatch_with_scope_check(
            state=state2, agent="r2c-test-producer", prompt="p", timeout_s=5,
            writeable_paths_override=WRITEABLE)
        raise AssertionError("expected OutOfScopeWritesError for authored stray")
    except OutOfScopeWritesError as e:
        assert "helper.py" in str(e)
        assert "_cache" not in str(e.violations)


def test_arch_coder_drift_recovers_when_outputs_valid(tmp_path, monkeypatch):
    """Item 19 structural half: the architecture-coder is opted into the
    same branch-B drift recovery as the method-coder. A dispatch that
    writes all three declared outputs plus an out-of-scope authored file
    recovers (stray reverted, outputs kept) instead of hard-halting."""
    from run_pipeline import _dispatch_arch_coder

    state = make_state(tmp_path / "run")
    (state.paths.run_dir / "method").mkdir(parents=True, exist_ok=True)

    def drifting_arch_dispatch(*, state, agent, prompt, timeout_s):
        run_dir = state.paths.run_dir
        (run_dir / "method" / "model.py").write_text(
            "class Net:\n    pass\n")
        (run_dir / "method" / "training.py").write_text(
            "def build_model():\n    return None\n")
        (run_dir / ".pipeline" / "arch_contract.json").write_text("{}")
        # The out-of-scope authored side effect (not a _cache stray, so it
        # exercises recovery rather than the exemption).
        (run_dir / "scratch_notes.py").write_text("x = 1")
        return _result()

    monkeypatch.setattr(run_pipeline, "dispatch_agent", drifting_arch_dispatch)
    result = _dispatch_arch_coder(state)
    assert result.completed is True
    # Stray reverted, canonical outputs kept.
    assert not (state.paths.run_dir / "scratch_notes.py").exists()
    assert (state.paths.run_dir / "method" / "model.py").is_file()
    assert (state.paths.run_dir / "method" / "training.py").is_file()
    assert (state.paths.pipeline_dir / "arch_contract.json").is_file()


def test_arch_coder_drift_with_invalid_outputs_still_fails(
    tmp_path, monkeypatch,
):
    """The recovery is gated on the canonical outputs being sound: an
    out-of-scope write alongside a missing/unparseable output must not be
    silently recovered into the fix loop's happy path."""
    from run_pipeline import OutOfScopeWritesError, _dispatch_arch_coder

    state = make_state(tmp_path / "run")
    (state.paths.run_dir / "method").mkdir(parents=True, exist_ok=True)

    def broken_arch_dispatch(*, state, agent, prompt, timeout_s):
        run_dir = state.paths.run_dir
        (run_dir / "method" / "model.py").write_text("def broken(:\n")
        (run_dir / "scratch_notes.py").write_text("x = 1")
        return _result()

    monkeypatch.setattr(run_pipeline, "dispatch_agent", broken_arch_dispatch)
    try:
        _dispatch_arch_coder(state)
        raise AssertionError(
            "expected OutOfScopeWritesError when outputs are invalid")
    except OutOfScopeWritesError as e:
        assert "scratch_notes.py" in str(e)


# ---------------------------------------------------------------------------
# Dispatch-discipline tranche (cap-burn measurement note, approved
# 2026-07-17). Three pieces:
#   fix 1 — tool-call-first writing discipline in the file-producing
#           agents' dispatch prompts (all 19 July 13-15 fatal burns were
#           the no-tool-call ramble; post-hygiene mass on the coders);
#   fix 2 — ONE corrective same-session resume when a dispatch dies with
#           turn shape cap_burn (SRL 07-15: three consecutive identical
#           method-coder burns under the stage-owned same-prompt re-roll);
#   fix 3 — a hard wall-clock ceiling on the proof-of-life extension a
#           still-generating backend can earn (SRL matrix row: declared
#           1800s, completed 14019s —
#           unbounded_timeout_extension_on_generating_backend).
# ---------------------------------------------------------------------------

# Captured at import time, before the conftest autouse fixture swaps the
# attribute out for the suite-wide no-op (same pattern as _REAL_RUNG3 in
# test_wiring_reliability.py).
_REAL_CAP_RESUME = run_pipeline._cap_burn_corrective_resume


# --- fix 1: template rendering ---------------------------------------------


def test_writing_discipline_rides_the_three_coder_dispatch_prompts(
        tmp_path, fake_dispatch):
    """The tool-call-first rule reaches architecture-coder, method-coder,
    and notebook-generator dispatches — the post-input-hygiene fatal
    cap-burn sites — through the shared template surface."""
    from run_pipeline import (_dispatch_arch_coder, _dispatch_method_coder,
                              _dispatch_notebook_generator)

    state = make_state(tmp_path / "run")
    fake_dispatch.expect(
        "r2c-architecture-coder",
        writes={"method/model.py": "x = 1\n", "method/training.py": "y = 1\n",
                ".pipeline/arch_contract.json": "{}"})
    fake_dispatch.expect(
        "r2c-method-coder", writes={"method/method.py": "z = 1\n"})
    fake_dispatch.expect(
        "r2c-notebook-generator",
        writes={".pipeline/notebook_draft.py": "# %%\n"})
    _dispatch_arch_coder(state)
    _dispatch_method_coder(state)
    _dispatch_notebook_generator(state)
    for call in fake_dispatch.calls:
        assert "## File-writing discipline" in call.prompt, call.agent
        assert "bounded chunks" in call.prompt, call.agent
        # Early position: ahead of the run-paths block, so the rule is in
        # working memory before the agent starts composing.
        assert (call.prompt.index("File-writing discipline")
                < call.prompt.index("**Run paths:**")), call.agent


def test_writing_discipline_stays_out_of_unrelated_dispatches(
        tmp_path, fake_dispatch, monkeypatch):
    """Snapshot-adjacent guard: reviewer/judge-class dispatches write one
    small artifact each and carry their own write-first instructions —
    their prompts must not pick up the coder discipline block."""
    from dispatch_templates import build_fix_mode_prompt
    from run_pipeline import _dispatch_stage_reviewer

    state = make_state(tmp_path / "run")
    monkeypatch.setattr(
        run_pipeline, "_stage_review_valid", lambda stage_id: (lambda s: True))
    fake_dispatch.expect(
        "r2c-stage-reviewer",
        writes={".pipeline/stage_review_stage_2b_architecture.json": "{}"})
    _dispatch_stage_reviewer(state, "stage_2b_architecture")
    assert "File-writing discipline" not in fake_dispatch.calls[0].prompt
    # Fix-mode prompts (a different template surface with its own act-first
    # retry preamble) are untouched too.
    fix_prompt = build_fix_mode_prompt(
        target_agent="r2c-method-coder",
        findings=[{"id": "F001", "severity": "critical",
                   "description": "d", "proposed_fix": "f"}],
        paths=run_pipeline.build_paths_block(state.paths),
        writeable_paths=run_pipeline.WRITEABLE_PATHS["r2c-method-coder"],
    )
    assert "File-writing discipline" not in fix_prompt


# --- fix 2: cap-burn corrective resume --------------------------------------


def _work_result() -> DispatchResult:
    """A completed dispatch in a DISTINCT fresh work session. The resume
    guard refuses a session that aliases the parent (make_state's
    "fake-session"), mirroring the production per-dispatch isolation."""
    return DispatchResult(
        session_id="work-ses-1", user_message_id="u1",
        assistant_message_id="a1", completed=True, elapsed_s=0.1,
    )


def _burn_shape(n_burns: int = 1) -> list[dict]:
    parts = []
    for _ in range(n_burns):
        parts.extend([{"type": "step-start"}, {"type": "reasoning"},
                      _finish(PER_STEP_OUTPUT_CAP)])
    return [_assistant(parts)]


def test_fatal_cap_burn_triggers_exactly_one_corrective_resume(
        tmp_path, monkeypatch):
    """Success path: the resume lands the allowlisted write in the SAME
    session, the events are registered and replayable, and the result
    flows on through the normal allowlist flow."""
    from run_events import load_events

    state = make_state(tmp_path / "run")
    monkeypatch.setattr(
        run_pipeline, "_cap_burn_corrective_resume", _REAL_CAP_RESUME)
    monkeypatch.setattr(run_pipeline, "dispatch_agent",
                        lambda **kw: _work_result())
    monkeypatch.setattr(
        run_pipeline.opencode_client, "fetch_session_messages",
        lambda session_id, *, port: _burn_shape(1))
    resume_calls = []

    def _resume(**kw):
        resume_calls.append(kw)
        out = state.paths.run_dir / ".pipeline" / "out.json"
        out.write_text("{}", encoding="utf-8")
        return DispatchResult(
            session_id=kw["session_id"], user_message_id="u2",
            assistant_message_id="a2-resume", completed=True, elapsed_s=2.0)

    monkeypatch.setattr(run_pipeline, "dispatch_and_wait", _resume)
    result = _dispatch_with_scope_check(
        state=state, agent="r2c-test-producer", prompt="p", timeout_s=7,
        writeable_paths_override=WRITEABLE)
    assert result.assistant_message_id == "a2-resume"
    assert len(resume_calls) == 1
    call = resume_calls[0]
    # SAME work session (a resume, never a re-roll), same declared timeout,
    # and the short corrective nudge — not the original prompt.
    assert call["session_id"] == "work-ses-1"
    assert call["timeout_s"] == 7
    assert "per-step output cap" in call["prompt"]
    assert call["prompt"] != "p"
    events = load_events(state.paths.pipeline_dir)  # replay-validates
    types = [e["event_type"] for e in events]
    # The burn stays on the books; the resume is attempted then succeeds.
    assert "cap_burn_turn_detected" in types
    assert types.count("cap_burn_corrective_resume_attempted") == 1
    assert types.count("cap_burn_corrective_resume_succeeded") == 1
    assert "cap_burn_corrective_resume_exhausted" not in types
    succeeded = [e for e in events
                 if e["event_type"] == "cap_burn_corrective_resume_succeeded"]
    assert succeeded[0]["details"]["writes"] == [".pipeline/out.json"]


def test_resume_that_burns_again_falls_through_to_stage_recovery(
        tmp_path, monkeypatch):
    """Second burn: exactly one resume attempt, the exhausted event names
    the second burn, and the original completed-no-write result falls
    through unchanged so the stage-owned recovery proceeds as today."""
    state = make_state(tmp_path / "run")
    monkeypatch.setattr(
        run_pipeline, "_cap_burn_corrective_resume", _REAL_CAP_RESUME)
    monkeypatch.setattr(run_pipeline, "dispatch_agent",
                        lambda **kw: _work_result())
    fetches = {"n": 0}

    def _messages(session_id, *, port):
        fetches["n"] += 1
        # First fetch (classification): one burn. Post-resume fetch: two.
        return _burn_shape(1 if fetches["n"] == 1 else 2)

    monkeypatch.setattr(
        run_pipeline.opencode_client, "fetch_session_messages", _messages)
    resume_calls = {"n": 0}

    def _resume(**kw):
        resume_calls["n"] += 1
        return DispatchResult(
            session_id=kw["session_id"], user_message_id="u2",
            assistant_message_id="a2-resume", completed=True, elapsed_s=1.0)

    monkeypatch.setattr(run_pipeline, "dispatch_and_wait", _resume)
    result = _dispatch_with_scope_check(
        state=state, agent="r2c-test-producer", prompt="p", timeout_s=7,
        writeable_paths_override=WRITEABLE)
    assert resume_calls["n"] == 1
    assert result.assistant_message_id == "a1"  # the original result
    exhausted = [e for e in _events(state)
                 if e["event_type"] == "cap_burn_corrective_resume_exhausted"]
    assert len(exhausted) == 1
    assert exhausted[0]["details"]["reason"] == "second_cap_burn"
    assert not [e for e in _events(state)
                if e["event_type"] == "cap_burn_corrective_resume_succeeded"]


def test_resume_dispatch_failure_is_contained_and_falls_through(
        tmp_path, monkeypatch):
    """Any resume failure is contained: no raise, no breaker changes, the
    original result falls through, budgets never stack."""
    from opencode_client import ServerUnreachable

    state = make_state(tmp_path / "run")
    monkeypatch.setattr(
        run_pipeline, "_cap_burn_corrective_resume", _REAL_CAP_RESUME)
    monkeypatch.setattr(run_pipeline, "dispatch_agent",
                        lambda **kw: _work_result())
    monkeypatch.setattr(
        run_pipeline.opencode_client, "fetch_session_messages",
        lambda session_id, *, port: _burn_shape(1))

    def _resume(**kw):
        raise ServerUnreachable("opencode server unreachable")

    monkeypatch.setattr(run_pipeline, "dispatch_and_wait", _resume)
    result = _dispatch_with_scope_check(
        state=state, agent="r2c-test-producer", prompt="p", timeout_s=7,
        writeable_paths_override=WRITEABLE)
    assert result.assistant_message_id == "a1"
    assert state.backend_unreachable is False
    exhausted = [e for e in _events(state)
                 if e["event_type"] == "cap_burn_corrective_resume_exhausted"]
    assert len(exhausted) == 1
    assert exhausted[0]["details"]["reason"] == "resume_dispatch_failed"


def test_resume_never_fires_on_short_empty_or_self_recovered_shapes(
        tmp_path, monkeypatch):
    """The resume is cap_burn-only: a short-empty no-write turn and a
    self-recovered burn (the dispatch landed its writes) never resume."""
    state = make_state(tmp_path / "run")
    monkeypatch.setattr(
        run_pipeline, "_cap_burn_corrective_resume", _REAL_CAP_RESUME)
    resume_calls = {"n": 0}

    def _resume(**kw):
        resume_calls["n"] += 1
        raise AssertionError("resume must not fire")

    monkeypatch.setattr(run_pipeline, "dispatch_and_wait", _resume)
    # Short-empty: completed, wrote nothing, small output.
    monkeypatch.setattr(run_pipeline, "dispatch_agent",
                        lambda **kw: _work_result())
    monkeypatch.setattr(
        run_pipeline.opencode_client, "fetch_session_messages",
        lambda session_id, *, port: [_assistant(
            [{"type": "step-start"}, {"type": "text"}, _finish(84)])])
    _dispatch_with_scope_check(
        state=state, agent="r2c-test-producer", prompt="p", timeout_s=5,
        writeable_paths_override=WRITEABLE)
    # Self-recovered: the dispatch wrote its output despite a burned step.
    def _writing_dispatch(**kw):
        (state.paths.run_dir / ".pipeline" / "out.json").write_text(
            "{}", encoding="utf-8")
        return _result()

    monkeypatch.setattr(run_pipeline, "dispatch_agent", _writing_dispatch)
    monkeypatch.setattr(
        run_pipeline.opencode_client, "fetch_session_messages",
        lambda session_id, *, port: _burn_shape(1))
    _dispatch_with_scope_check(
        state=state, agent="r2c-test-producer", prompt="p", timeout_s=5,
        writeable_paths_override=WRITEABLE)
    assert resume_calls["n"] == 0
    types = [e["event_type"] for e in _events(state)]
    assert "cap_burn_recovered" in types  # the self-recovered sweep still ran
    assert "cap_burn_corrective_resume_attempted" not in types


def test_resume_output_still_goes_through_the_allowlist_flow(
        tmp_path, monkeypatch):
    """A resume that lands its write but ALSO strays out of scope is caught
    by the same ownership enforcement as a first-try write."""
    state = make_state(tmp_path / "run")
    monkeypatch.setattr(
        run_pipeline, "_cap_burn_corrective_resume", _REAL_CAP_RESUME)
    monkeypatch.setattr(run_pipeline, "dispatch_agent",
                        lambda **kw: _work_result())
    monkeypatch.setattr(
        run_pipeline.opencode_client, "fetch_session_messages",
        lambda session_id, *, port: _burn_shape(1))

    def _resume(**kw):
        (state.paths.run_dir / ".pipeline" / "out.json").write_text(
            "{}", encoding="utf-8")
        (state.paths.run_dir / "rogue.txt").write_text(
            "stray", encoding="utf-8")
        return DispatchResult(
            session_id=kw["session_id"], user_message_id="u2",
            assistant_message_id="a2-resume", completed=True, elapsed_s=1.0)

    monkeypatch.setattr(run_pipeline, "dispatch_and_wait", _resume)
    with pytest.raises(run_pipeline.OutOfScopeWritesError) as exc:
        _dispatch_with_scope_check(
            state=state, agent="r2c-test-producer", prompt="p", timeout_s=5,
            writeable_paths_override=WRITEABLE)
    assert "rogue.txt" in str(exc.value)


def test_cap_burn_resume_events_are_registered():
    from run_events import EVENT_TYPES

    assert "cap_burn_corrective_resume_attempted" in EVENT_TYPES
    assert "cap_burn_corrective_resume_succeeded" in EVENT_TYPES
    assert "cap_burn_corrective_resume_exhausted" in EVENT_TYPES
    assert "dispatch_extension_ceiling_hit" in EVENT_TYPES
    from fleet_state import EVENT_CATEGORIES

    assert EVENT_CATEGORIES["cap_burn_corrective_resume_exhausted"] == "problem"
    assert EVENT_CATEGORIES["dispatch_extension_ceiling_hit"] == "problem"


# --- fix 3: proof-of-life extension ceiling ---------------------------------


def test_extension_ceiling_defaults_to_twice_the_declared_timeout(
        monkeypatch):
    from opencode_client import (DISPATCH_EXTENSION_CEILING_ENV,
                                 dispatch_extension_ceiling_s)

    monkeypatch.delenv(DISPATCH_EXTENSION_CEILING_ENV, raising=False)
    assert dispatch_extension_ceiling_s(1800) == 3600
    # Env override is a unitless multiplier of the declared timeout.
    monkeypatch.setenv(DISPATCH_EXTENSION_CEILING_ENV, "4")
    assert dispatch_extension_ceiling_s(1800) == 7200
    # Below 1 clamps to 1: the ceiling can never undercut the declared
    # timeout (that would break ordinary slow-but-fine dispatches).
    monkeypatch.setenv(DISPATCH_EXTENSION_CEILING_ENV, "0.25")
    assert dispatch_extension_ceiling_s(1800) == 1800
    # Garbage falls back to the default rather than crashing a dispatch.
    monkeypatch.setenv(DISPATCH_EXTENSION_CEILING_ENV, "unbounded")
    assert dispatch_extension_ceiling_s(600) == 1200


class _FakeTrickleResponse:
    """A response whose read1 yields one small chunk per call — the
    still-generating backend shape — optionally ending after n chunks."""

    def __init__(self, clock, chunks=None, step_s=0.1):
        self._clock = clock
        self._chunks = chunks  # None = trickle forever
        self._step_s = step_s

    def read1(self, _n):
        self._clock["t"] += self._step_s
        if self._chunks is not None:
            if not self._chunks:
                return b""
            return self._chunks.pop(0)
        return b"x"


def test_wall_ceiling_read_abandons_a_forever_trickling_backend(monkeypatch):
    """Unit shape of the SRL runaway: every received chunk is implicit
    proof of life, and without the ceiling the read loop never exits."""
    import opencode_client
    from opencode_client import (DispatchExtensionCeiling,
                                 _read_with_wall_ceiling)

    clock = {"t": 1000.0}
    monkeypatch.setattr(opencode_client.time, "monotonic", lambda: clock["t"])
    with pytest.raises(DispatchExtensionCeiling) as exc:
        _read_with_wall_ceiling(
            _FakeTrickleResponse(clock), started=1000.0, ceiling_s=3.0,
            timeout=1.5, base="http://127.0.0.1:0")
    assert "extension ceiling" in str(exc.value)
    assert "unbounded_timeout_extension_on_generating_backend" in str(exc.value)


def test_wall_ceiling_read_preserves_grace_below_the_ceiling(monkeypatch):
    """A slow-but-alive backend that finishes below the ceiling keeps its
    extension grace: the body is returned whole, exactly as today."""
    import opencode_client
    from opencode_client import _read_with_wall_ceiling

    clock = {"t": 500.0}
    monkeypatch.setattr(opencode_client.time, "monotonic", lambda: clock["t"])
    body = _read_with_wall_ceiling(
        _FakeTrickleResponse(clock, chunks=[b'{"ok"', b": true}"]),
        started=500.0, ceiling_s=10.0, timeout=1.0,
        base="http://127.0.0.1:0")
    assert body == '{"ok": true}'


def test_wall_ceiling_read_raises_incomplete_read_on_truncated_body(
        monkeypatch):
    """read1 does not enforce Content-Length, so a server dying mid-body
    would quietly yield a partial buffer (which could even parse as JSON
    and be accepted as a real dispatch response). When the response length
    is known, an early EOF must raise the same loud incomplete-read the
    base read path produced."""
    import http.client

    import opencode_client
    from opencode_client import _read_with_wall_ceiling

    clock = {"t": 100.0}
    monkeypatch.setattr(opencode_client.time, "monotonic", lambda: clock["t"])

    class _Dying(_FakeTrickleResponse):
        length = 7  # bytes the Content-Length still owes at EOF

    with pytest.raises(http.client.IncompleteRead) as exc:
        _read_with_wall_ceiling(
            _Dying(clock, chunks=[b'{"info"', b": {}}"]),
            started=100.0, ceiling_s=60.0, timeout=1.0,
            base="http://127.0.0.1:0")
    assert exc.value.partial == b'{"info": {}}'
    assert exc.value.expected == 7


def test_wall_ceiling_overshoot_is_bounded_by_one_socket_window(monkeypatch):
    """B-15: the trickle-then-silence shape, the one shape production hit
    POST-fix (the 2026-08-04 method-analyzer dispatch failed at 4,555s
    against a 1800s declared / 3600s nominal ceiling). The wall check runs
    BETWEEN chunk reads, so a backend that trickles below the ceiling and
    then goes silent times out inside one final in-flight socket window:
    the crossing emits no ceiling event and surfaces as a plain dispatch
    timeout, but the total is bounded by ceiling + one socket window."""
    import socket

    import opencode_client
    from opencode_client import _read_with_wall_ceiling

    clock = {"t": 0.0}
    monkeypatch.setattr(opencode_client.time, "monotonic", lambda: clock["t"])
    ceiling_s, window_s = 3.0, 1.5

    class _TrickleThenSilence:
        def read1(self, _n):
            if clock["t"] + 0.4 < ceiling_s:
                clock["t"] += 0.4  # still trickling, below the ceiling
                return b"x"
            # The final in-flight read: the backend goes silent, the socket
            # blocks one full window past the ceiling, then times out. The
            # wall check never gets to see the crossing.
            clock["t"] += window_s
            raise socket.timeout("read timed out")

    # Surfaces as the socket timeout (wrapped into DispatchResponseTimeout
    # at the message-POST layer), NOT as DispatchExtensionCeiling.
    with pytest.raises((TimeoutError, socket.timeout)):
        _read_with_wall_ceiling(
            _TrickleThenSilence(), started=0.0, ceiling_s=ceiling_s,
            timeout=window_s, base="http://127.0.0.1:0")
    assert clock["t"] > ceiling_s, "the ceiling WAS crossed (silently)"
    assert clock["t"] <= ceiling_s + window_s, (
        f"overshoot must be bounded by one socket window: total "
        f"{clock['t']:.1f}s vs bound {ceiling_s + window_s:.1f}s")


def test_srl_runaway_reproducer_post_message_hits_the_ceiling(monkeypatch):
    """Named reproducer, end to end over a real socket: the SRL matrix
    row's paper-fidelity-reviewer dispatch declared timeout=1800s and
    completed at 14019s because the backend trickled data forever. Scaled
    down: declared 0.3s, ceiling 0.6s, a byte every 0.05s — the POST must
    abandon the wait at the ceiling instead of riding the trickle."""
    import http.server
    import threading
    import time as _time

    from opencode_client import (DISPATCH_EXTENSION_CEILING_ENV,
                                 DispatchExtensionCeiling, post_message)

    # A developer's exported multiplier would move the 0.6s ceiling and
    # flake the elapsed-window assertion below.
    monkeypatch.delenv(DISPATCH_EXTENSION_CEILING_ENV, raising=False)

    class _Runaway(http.server.BaseHTTPRequestHandler):
        def do_POST(self):
            length = int(self.headers.get("Content-Length", 0) or 0)
            if length:
                self.rfile.read(length)
            self.send_response(200)
            self.end_headers()
            try:
                for _ in range(600):  # bounded so a regression can't hang
                    self.wfile.write(b" ")
                    self.wfile.flush()
                    _time.sleep(0.05)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def log_message(self, *a):
            pass

    server = http.server.HTTPServer(("127.0.0.1", 0), _Runaway)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        started = _time.time()
        with pytest.raises(DispatchExtensionCeiling):
            post_message(
                session_id="runaway", agent="r2c-paper-fidelity-reviewer",
                prompt="p", port=server.server_address[1], timeout_s=0.3)
        elapsed = _time.time() - started
        # Abandoned near the 0.6s ceiling — not at the socket timeout
        # (data kept flowing) and not after the server's full 30s trickle.
        assert 0.5 < elapsed < 5.0, elapsed
    finally:
        server.shutdown()


def test_ceiling_error_takes_the_backend_alive_timeout_path(
        tmp_path, monkeypatch):
    """Driver handling: the ceiling error is logged loudly (registered
    event, failure class named) and then treated exactly like the existing
    backend-alive timeout — breaker closed, error propagated into
    stage-owned recovery, the generating session abandoned."""
    from opencode_client import DispatchExtensionCeiling

    state = make_state(tmp_path / "run")
    monkeypatch.setattr(
        run_pipeline, "create_session", lambda **_k: "fresh-work-ses")
    monkeypatch.setattr(
        run_pipeline, "dispatch_and_wait",
        lambda **_k: (_ for _ in ()).throw(DispatchExtensionCeiling(
            "dispatch POST crossed the extension ceiling")))
    monkeypatch.setattr(
        run_pipeline.opencode_client, "fetch_session_messages",
        lambda session_id, *, port: [
            {"info": {"role": "assistant"},
             "parts": [{"type": "reasoning"}]}])
    aborted_sessions = []
    monkeypatch.setattr(
        run_pipeline, "abort_session",
        lambda session_id, *, port: aborted_sessions.append(session_id) or True)
    with pytest.raises(DispatchExtensionCeiling):
        run_pipeline.dispatch_agent(
            state=state, agent="r2c-paper-fidelity-reviewer", prompt="x",
            timeout_s=1800)
    assert state.backend_unreachable is False  # proof of life kept it closed
    types = [e["event_type"] for e in _events(state)]
    assert "dispatch_extension_ceiling_hit" in types
    assert "dispatch_timeout_backend_alive" in types
    hit = [e for e in _events(state)
           if e["event_type"] == "dispatch_extension_ceiling_hit"][0]
    assert hit["details"]["failure_class"] == \
        "unbounded_timeout_extension_on_generating_backend"
    assert hit["details"]["timeout_s"] == 1800
    # 2026-08-24 wander: the abandoned still-generating turn must be
    # aborted server-side, and the abort recorded as a registered event.
    assert aborted_sessions == ["fresh-work-ses"]
    assert "abandoned_turn_aborted" in types


def test_resume_hitting_the_ceiling_is_labeled_and_loud(
        tmp_path, monkeypatch):
    """The runaway class striking DURING a corrective resume stays visible:
    the exhausted reason is resume_extension_ceiling and the loud ceiling
    event fires with the resume context, then the original result falls
    through to stage-owned recovery as with any resume failure."""
    from opencode_client import DispatchExtensionCeiling

    state = make_state(tmp_path / "run")
    monkeypatch.setattr(
        run_pipeline, "_cap_burn_corrective_resume", _REAL_CAP_RESUME)
    monkeypatch.setattr(run_pipeline, "dispatch_agent",
                        lambda **kw: _work_result())
    monkeypatch.setattr(
        run_pipeline.opencode_client, "fetch_session_messages",
        lambda session_id, *, port: _burn_shape(1))
    monkeypatch.setattr(
        run_pipeline, "dispatch_and_wait",
        lambda **_k: (_ for _ in ()).throw(DispatchExtensionCeiling(
            "dispatch POST crossed the extension ceiling")))
    result = _dispatch_with_scope_check(
        state=state, agent="r2c-test-producer", prompt="p", timeout_s=7,
        writeable_paths_override=WRITEABLE)
    assert result.assistant_message_id == "a1"  # original falls through
    events = _events(state)
    exhausted = [e for e in events
                 if e["event_type"] == "cap_burn_corrective_resume_exhausted"]
    assert exhausted[0]["details"]["reason"] == "resume_extension_ceiling"
    hits = [e for e in events
            if e["event_type"] == "dispatch_extension_ceiling_hit"]
    assert len(hits) == 1
    assert hits[0]["details"]["context"] == "cap_burn_corrective_resume"
    assert hits[0]["details"]["failure_class"] == \
        "unbounded_timeout_extension_on_generating_backend"


def test_dispatch_budgets_scale_by_env(monkeypatch, tmp_path):
    """Declared budgets were calibrated on Qwen 27B; R2C_TIMEOUT_SCALE
    (default 3.0) stretches them for slower hosted models, 0 means
    unlimited, garbage falls back to the default, and the driver applies
    it once so the logged budget is the enforced one."""
    import opencode_client as oc
    monkeypatch.delenv("R2C_TIMEOUT_SCALE")
    assert oc.scaled_timeout_s(1800) == 5400
    monkeypatch.setenv("R2C_TIMEOUT_SCALE", "1.5")
    assert oc.scaled_timeout_s(1800) == 2700
    monkeypatch.setenv("R2C_TIMEOUT_SCALE", "0")
    assert oc.scaled_timeout_s(1800) >= 365 * 24 * 3600
    monkeypatch.setenv("R2C_TIMEOUT_SCALE", "junk")
    assert oc.scaled_timeout_s(1800) == 5400

    monkeypatch.setenv("R2C_TIMEOUT_SCALE", "2")
    state = make_state(tmp_path / "run")
    seen = {}
    monkeypatch.setattr(
        run_pipeline, "create_session", lambda **_k: "fresh-work-ses")

    def fake_dispatch(**kw):
        seen["timeout_s"] = kw["timeout_s"]
        return _result()
    monkeypatch.setattr(run_pipeline, "dispatch_and_wait", fake_dispatch)
    monkeypatch.setattr(
        run_pipeline.opencode_client, "fetch_session_messages",
        lambda session_id, *, port: [])
    run_pipeline.dispatch_agent(
        state=state, agent="r2c-paper-fidelity-reviewer", prompt="x",
        timeout_s=1800)
    assert seen["timeout_s"] == 3600
    started = [e for e in _events(state)
               if e["event_type"] == "agent_dispatch_started"][0]
    assert started["details"]["timeout_s"] == 3600


def test_output_cap_resolves_from_the_served_default_model(monkeypatch):
    """The cap-burn detector keys on >= the model's output limit, so the
    driver reads that limit from the server instead of pinning it: wrong in
    either direction manufactures or misses burn classifications."""
    import opencode_client as oc
    responses = {
        "/config": (200, json.dumps({"model": "google/gemini-flash-latest"})),
        "/config/providers": (200, json.dumps({"providers": [
            {"id": "openai", "models": {"gpt-5": {"limit": {"output": 128000}}}},
            {"id": "google", "models": {
                "gemini-flash-latest": {"limit": {"output": 65536}}}},
        ]})),
    }
    monkeypatch.setattr(oc, "_http", lambda path, **kw: responses[path])
    monkeypatch.delenv("R2C_PER_STEP_OUTPUT_CAP", raising=False)
    monkeypatch.setattr(oc, "PER_STEP_OUTPUT_CAP", 32768)
    assert oc.configure_output_cap(port=1) == 65536
    assert oc.PER_STEP_OUTPUT_CAP == 65536
    # the detector picks the new value up at call time
    msgs = [{"info": {"role": "assistant"},
             "parts": [{"type": "step-start"}, _finish(65536)]}]
    assert oc.cap_burn_steps(msgs)
    assert not oc.cap_burn_steps(
        [{"info": {"role": "assistant"},
          "parts": [{"type": "step-start"}, _finish(32768)]}])
    # an explicit pin wins over the server
    monkeypatch.setenv("R2C_PER_STEP_OUTPUT_CAP", "1")
    monkeypatch.setattr(oc, "PER_STEP_OUTPUT_CAP", 32768)
    assert oc.configure_output_cap(port=1) == 32768
    # an unreachable server keeps the current value
    monkeypatch.delenv("R2C_PER_STEP_OUTPUT_CAP")
    monkeypatch.setattr(oc, "_http", lambda path, **kw: (_ for _ in ()).throw(oc.ServerUnreachable("down")))
    assert oc.configure_output_cap(port=1) == 32768


def test_plain_http_calls_get_a_default_wall_ceiling(monkeypatch):
    """2026-08-23 A8 wedge: the driver sat ~19h past every declared
    deadline on one established socket. Whatever the exact site, no _http
    call may read unbounded: a call that declares NO explicit ceiling gets
    max(2x timeout, timeout+120) and abandons a forever-trickling response
    loudly."""
    import opencode_client
    from opencode_client import DispatchExtensionCeiling, _http

    clock = {"t": 1000.0}
    monkeypatch.setattr(opencode_client.time, "monotonic", lambda: clock["t"])

    class _FakeOpenerResponse(_FakeTrickleResponse):
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    # step_s chosen so the default ceiling (150s for timeout=30) is crossed
    # after a bounded number of chunks.
    monkeypatch.setattr(
        opencode_client._OPENER, "open",
        lambda req, timeout=None: _FakeOpenerResponse(clock, step_s=40.0))
    with pytest.raises(DispatchExtensionCeiling):
        _http("/session", port=1, timeout=30.0)


def test_explicit_wall_ceiling_still_overrides_the_default(monkeypatch):
    """post_message's dispatch ceiling (multiplier x declared timeout)
    must keep winning over the plain-call default."""
    import opencode_client
    from opencode_client import DispatchExtensionCeiling, _http

    clock = {"t": 1000.0}
    monkeypatch.setattr(opencode_client.time, "monotonic", lambda: clock["t"])

    class _FakeOpenerResponse(_FakeTrickleResponse):
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    monkeypatch.setattr(
        opencode_client._OPENER, "open",
        lambda req, timeout=None: _FakeOpenerResponse(clock, step_s=40.0))
    with pytest.raises(DispatchExtensionCeiling) as exc:
        _http("/session", port=1, timeout=30.0, wall_ceiling_s=3600.0)
    assert "3600" in str(exc.value)


def test_backend_alive_timeout_aborts_the_abandoned_turn(
        tmp_path, monkeypatch):
    """A plain DispatchTimeout whose session shows in-window generation is
    abandoned — and the abandonment must ABORT the turn server-side: the
    2026-08-24 test-generator wander kept writing repo files minutes after
    the scope-guard halt because nothing stopped the turn. A failed abort
    is recorded (status failed) and never masks the original timeout."""
    from opencode_client import DispatchTimeout

    state = make_state(tmp_path / "run")
    monkeypatch.setattr(
        run_pipeline, "create_session", lambda **_k: "wandering-ses")
    monkeypatch.setattr(
        run_pipeline, "dispatch_and_wait",
        lambda **_k: (_ for _ in ()).throw(DispatchTimeout(
            "dispatch POST timed out after 900s")))
    monkeypatch.setattr(
        run_pipeline.opencode_client, "fetch_session_messages",
        lambda session_id, *, port: [
            {"info": {"role": "assistant"},
             "parts": [{"type": "reasoning"}, {"type": "tool"}]}])
    monkeypatch.setattr(
        run_pipeline, "abort_session", lambda session_id, *, port: False)
    with pytest.raises(DispatchTimeout):
        run_pipeline.dispatch_agent(
            state=state, agent="r2c-test-generator", prompt="x",
            timeout_s=900)
    events = _events(state)
    aborts = [e for e in events if e["event_type"] == "abandoned_turn_aborted"]
    assert len(aborts) == 1
    assert aborts[0]["status"] == "failed"
    assert aborts[0]["details"]["aborted"] is False
