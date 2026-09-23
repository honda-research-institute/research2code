"""R2C-060 — an empty turn with completed investigation gets a resume.

The 2026-08-05 pdfgnn day roll's smoke diagnostician died twice on iteration
2. Attempt one ingested 20,026 input tokens and emitted exactly one token in
3.7s. Attempt two did four real investigation steps (seven file reads,
context growing 20k to 74k tokens) and then emitted 15,606 tokens of
reasoning in its final step without ever issuing the write. Both sat UNDER
the 20k per-step cap, so the cap-burn corrective resume never fired, and the
driver fallback authored a generic diagnosis that then SHIPPED as the
delivered known-issues entry (`abandoned_investigation`).

The control case is the same run's method coder, which cap-burned an 886s
turn and was rescued by the existing resume.
"""

from __future__ import annotations

import pytest

import run_pipeline
from scripts.opencode_client import (
    PER_STEP_OUTPUT_CAP, cap_burn_steps, completed_tool_calls,
    final_step_output_tokens,
)
from scripts.run_pipeline import _short_empty_resume_qualifies

# Captured at import, before conftest's suite-wide stub replaces it.
_REAL_RESUME = run_pipeline._cap_burn_corrective_resume


def _assistant(steps: list[list[dict]]) -> dict:
    parts: list[dict] = []
    for step in steps:
        parts.extend(step)
    return {"info": {"role": "assistant"}, "parts": parts}


def _step(*, output: int, tools: list[str] | None = None) -> list[dict]:
    parts: list[dict] = [{"type": "step-start"}]
    for tool in tools or []:
        parts.append({"type": "tool", "tool": tool})
    parts.append({"type": "step-finish", "tokens": {"output": output}})
    return parts


# The two recorded 2026-08-05 diagnostician shapes.
_ONE_TOKEN_TURN = [_assistant([_step(output=1)])]
_INVESTIGATED_THEN_DIED = [_assistant([
    _step(output=180, tools=["read", "read"]),
    _step(output=240, tools=["read", "grep"]),
    _step(output=310, tools=["read", "read", "read"]),
    _step(output=15606),
])]
# The known-good control: a cap burn, which the existing path already owns.
# Keyed to the live constant so a deliberate cap change cannot silently
# demote this control into a short-empty shape (2026-08-20 raise).
_CAP_BURN_TURN = [_assistant([_step(output=200, tools=["read"]),
                              _step(output=PER_STEP_OUTPUT_CAP)])]


def test_the_fifteen_thousand_token_no_write_turn_qualifies():
    """Four investigation steps and seven reads sit in that session."""
    tool_calls = completed_tool_calls(_INVESTIGATED_THEN_DIED)
    assert tool_calls == 7
    assert cap_burn_steps(_INVESTIGATED_THEN_DIED) == [], "under the cap"
    assert final_step_output_tokens(_INVESTIGATED_THEN_DIED) == 15606
    assert _short_empty_resume_qualifies(
        tool_calls=tool_calls, final_step_output=15606)


def test_the_one_token_turn_qualifies_with_no_work_at_all():
    """An inference-side immediate stop is transient; a resume costs one
    prompt and there is nothing to lose."""
    assert completed_tool_calls(_ONE_TOKEN_TURN) == 0
    assert final_step_output_tokens(_ONE_TOKEN_TURN) == 1
    assert _short_empty_resume_qualifies(tool_calls=0, final_step_output=1)


def test_a_session_with_no_work_and_a_real_output_keeps_the_fresh_roll():
    """Nothing to preserve, so the existing re-dispatch path is correct."""
    assert not _short_empty_resume_qualifies(
        tool_calls=0, final_step_output=900)
    assert not _short_empty_resume_qualifies(
        tool_calls=0, final_step_output=None)


def test_a_truncated_tool_call_is_not_completed_work():
    """The server records a cut-off write as the literal `invalid` tool, so
    it must not count as investigation to preserve."""
    messages = [_assistant([_step(output=20000, tools=["invalid"])])]
    assert completed_tool_calls(messages) == 0


def test_the_cap_burn_control_is_untouched_by_the_new_condition():
    assert len(cap_burn_steps(_CAP_BURN_TURN)) == 1
    # Classification puts a cap-burned turn on the cap_burn branch before the
    # short-empty condition is ever consulted, whatever the tool count.
    assert completed_tool_calls(_CAP_BURN_TURN) == 1


def test_user_messages_never_contribute_tool_calls():
    messages = [{"info": {"role": "user"},
                 "parts": [{"type": "tool", "tool": "read"}]}]
    assert completed_tool_calls(messages) == 0
    assert final_step_output_tokens(messages) is None


@pytest.mark.parametrize("shape,nudge_marker", [
    ("cap_burn", "per-step output cap"),
    ("short_empty", "ended without writing any of the files"),
])
def test_each_shape_gets_wording_that_matches_what_happened(
    shape, nudge_marker,
):
    """A short-empty turn told it hit the cap would be told a falsehood, and
    the nudge's whole job is to be believed."""
    from scripts.dispatch_templates import (
        CAP_BURN_CORRECTIVE_RESUME_NUDGE,
        SHORT_EMPTY_CORRECTIVE_RESUME_NUDGE,
    )

    nudge = (CAP_BURN_CORRECTIVE_RESUME_NUDGE if shape == "cap_burn"
             else SHORT_EMPTY_CORRECTIVE_RESUME_NUDGE)
    assert nudge_marker in nudge
    assert "do NOT read more files" in nudge or "Do NOT" in nudge
    if shape == "short_empty":
        assert "per-step output cap" not in nudge


def test_the_resume_records_which_shape_fired(tmp_path, monkeypatch):
    """History has to stay comparable across the extension, so the shared
    event names carry the shape in their details."""
    import json

    import run_pipeline
    from run_pipeline import DispatchResult
    from tests.helpers.state import make_state

    # conftest stubs the resume suite-wide (it is a real message POST); the
    # focused tests restore it and stub the transport instead.
    _cap_burn_corrective_resume = _REAL_RESUME
    state = make_state(tmp_path / "run")
    state.session_id = "ses_parent"
    target = state.paths.pipeline_dir / "smoke_diagnosis.json"

    def _result(session_id: str) -> DispatchResult:
        return DispatchResult(
            session_id=session_id, user_message_id="m1",
            assistant_message_id="m2", completed=True, elapsed_s=1.0)

    def _dispatch(**kwargs):
        target.write_text("{}", encoding="utf-8")
        return _result("ses_work")

    monkeypatch.setattr(run_pipeline, "dispatch_and_wait", _dispatch)
    monkeypatch.setattr(run_pipeline, "_changed_paths_since_snapshot",
                        lambda *a, **k: [".pipeline/smoke_diagnosis.json"])

    resumed = _cap_burn_corrective_resume(
        state, agent="r2c-smoke-diagnostician",
        first_result=_result("ses_work"),
        timeout_s=60, writeable=[".pipeline/*"], snapshot={},
        first_burns=[], shape="short_empty",
        nudge=run_pipeline.SHORT_EMPTY_CORRECTIVE_RESUME_NUDGE)

    assert resumed is not None
    events = [json.loads(line) for line in
              (state.paths.pipeline_dir / "run_events.jsonl")
              .read_text(encoding="utf-8").splitlines() if line.strip()]
    shapes = [e["details"]["shape"] for e in events
              if "shape" in (e.get("details") or {})]
    assert shapes == ["short_empty", "short_empty"]
