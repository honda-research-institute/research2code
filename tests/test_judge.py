"""Phase 3 — Halt-judge framework unit tests.

Tests `_invoke_judge`, `JudgeOutcome`, the `JUDGE_FIX_DISPATCHERS` table,
and `local_dispatchers` overrides. Each test fakes the judge's dispatch
(writing a canned scratch decision object) and asserts the driver-side
plumbing validates it, appends canonical `judge_decisions.json`, and turns
the result into the right `JudgeOutcome`.

Driver-behavior tests, not judge-accuracy tests — the judge's actual
classification quality is a separate problem with hand-labeled gold
standards, not in v1.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.helpers.state import make_state


# ---------------------------------------------------------------------------
# Helpers specific to judge tests
# ---------------------------------------------------------------------------


def _build_decision(
    *,
    stage_id: str = "stage_2d",
    iteration: int = 0,
    validator_label: str = "validate_arch_contract.py",
    classification: str = "producer_fixable",
    action: str = "dispatch_fix",
    target_agent: str | None = "r2c-architecture-coder",
    finding: dict | None = None,
    rationale: str = "test rationale",
    confidence: str = "high",
    files_examined: list[str] | None = None,
) -> dict:
    """Construct a single JudgeDecision dict. Defaults are a happy
    dispatch_fix decision to architecture-coder."""
    if action == "dispatch_fix" and finding is None:
        finding = {
            "id": "JUDGE001",
            "severity": "critical",
            "description": "test finding description",
            "proposed_fix": "test proposed fix",
        }
    if action == "halt":
        target_agent = None
        finding = None
    return {
        "schema_version": "1.0.0",
        "stage_id": stage_id,
        "iteration": iteration,
        "validator_label": validator_label,
        "classification": classification,
        "action": action,
        "target_agent": target_agent,
        "finding": finding,
        "rationale": rationale,
        "confidence": confidence,
        "files_examined": files_examined or ["scripts/run_pipeline.py"],
    }


def _queue_judge_writing(
    fake_dispatch, decision: dict | list[dict], *,
    output_stage_id: str | None = None,
    output_iteration: int | None = None,
    output_validator_label: str | None = None,
    extra_writes: dict[str, str] | None = None,
) -> None:
    """Queue a single judge dispatch whose canned output is one scratch
    decision object. A list is accepted for old test readability; the last
    entry is the current scratch decision."""
    if isinstance(decision, list):
        if not decision:
            raise ValueError("decision list must not be empty")
        decision = decision[-1]
    from run_pipeline import _judge_decision_scratch_relpath

    scratch_rel = _judge_decision_scratch_relpath(
        stage_id=output_stage_id or decision["stage_id"],
        iteration=(
            output_iteration
            if output_iteration is not None
            else decision["iteration"]
        ),
        validator_label=output_validator_label or decision["validator_label"],
    )
    writes = {scratch_rel: json.dumps(decision, indent=2)}
    if extra_writes:
        writes.update(extra_writes)
    fake_dispatch.expect(
        agent="r2c-halt-judge",
        writes=writes,
    )


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


def test_invoke_judge_dispatch_fix_producer_fixable_returns_dispatcher(
    fake_dispatch, run_dir
):
    """Judge decides producer_fixable → dispatch architecture-coder. The
    returned JudgeOutcome carries the resolved dispatcher callable."""
    from run_pipeline import (
        _dispatch_arch_coder_fix,
        _invoke_judge,
        JudgeOutcome,
    )

    state = make_state(run_dir)
    decision = _build_decision(
        classification="producer_fixable",
        action="dispatch_fix",
        target_agent="r2c-architecture-coder",
    )
    _queue_judge_writing(fake_dispatch, [decision])

    outcome = _invoke_judge(
        state, stage_id="stage_2d",
        validator_label="validate_arch_contract.py",
        stderr_tail="schema error", iteration=0,
    )

    assert isinstance(outcome, JudgeOutcome)
    assert outcome.action == "dispatch_fix"
    assert outcome.classification == "producer_fixable"
    assert outcome.target_agent == "r2c-architecture-coder"
    assert outcome.dispatcher is _dispatch_arch_coder_fix
    assert outcome.finding is not None
    assert outcome.finding["id"] == "JUDGE001"
    assert outcome.confidence == "high"


def test_invoke_judge_writes_decisions_file_via_dispatch(fake_dispatch, run_dir):
    """The judge's dispatch writes a scratch object and the driver appends
    canonical `judge_decisions.json`. End-to-end sanity for the file-based
    contract."""
    from run_pipeline import _invoke_judge

    state = make_state(run_dir)
    decision = _build_decision()
    _queue_judge_writing(fake_dispatch, [decision])

    _invoke_judge(
        state, stage_id="stage_2d",
        validator_label="validate_arch_contract.py",
        stderr_tail="x", iteration=0,
    )

    decisions_path = run_dir / ".pipeline" / "judge_decisions.json"
    assert decisions_path.is_file()
    on_disk = json.loads(decisions_path.read_text())
    assert isinstance(on_disk, list)
    assert on_disk[0]["iteration"] == 0


# ---------------------------------------------------------------------------
# Halt action
# ---------------------------------------------------------------------------


def test_invoke_judge_halt_action_returns_outcome_with_no_dispatcher(
    fake_dispatch, run_dir
):
    """When the judge decides action=halt, the outcome carries the rationale
    and no dispatcher/finding."""
    from run_pipeline import _invoke_judge

    state = make_state(run_dir)
    decision = _build_decision(
        classification="upstream_issue",
        action="halt",
        rationale="upstream paradigm mismatch",
    )
    _queue_judge_writing(fake_dispatch, [decision])

    outcome = _invoke_judge(
        state, stage_id="stage_2d",
        validator_label="validate_arch_contract.py",
        stderr_tail="x", iteration=0,
    )

    assert outcome.action == "halt"
    assert outcome.classification == "upstream_issue"
    assert outcome.target_agent is None
    assert outcome.finding is None
    assert outcome.dispatcher is None
    assert "upstream paradigm mismatch" in outcome.rationale


def test_invoke_judge_unknown_classification_token_normalizes_to_unclear(
    fake_dispatch, run_dir
):
    """A malformed diagnostic label must not discard a valid halt action.

    The 2026-06-29 GBALD Stage 2.b run reproduced this with
    classification="production_bug": action=halt was correct, but the label
    was outside the literal taxonomy. Because classification is diagnostic,
    normalize it to unclear and keep the raw token in the rationale.
    """
    from run_pipeline import _invoke_judge

    state = make_state(run_dir)
    decision = _build_decision(
        stage_id="stage_2b",
        validator_label="validate_architecture_coder_output.py",
        classification="production_bug",
        action="halt",
        rationale="oscillation: producer kept missing method/training.py",
    )
    _queue_judge_writing(fake_dispatch, decision)

    outcome = _invoke_judge(
        state,
        stage_id="stage_2b",
        validator_label="validate_architecture_coder_output.py",
        stderr_tail="method/training.py not present",
        iteration=0,
    )

    assert outcome.action == "halt"
    assert outcome.classification == "unclear"
    assert "production_bug" in outcome.rationale
    assert "missing method/training.py" in outcome.rationale

    history = json.loads(
        (run_dir / ".pipeline" / "judge_decisions.json").read_text()
    )
    assert history[0]["classification"] == "unclear"
    assert "production_bug" in history[0]["rationale"]


# ---------------------------------------------------------------------------
# Safety downgrades
# ---------------------------------------------------------------------------


def test_invoke_judge_unknown_target_agent_downgrades_to_halt(
    fake_dispatch, run_dir
):
    """If the judge picks a target_agent not in JUDGE_FIX_DISPATCHERS (and
    not in local_dispatchers), the driver downgrades the outcome to halt
    rather than blindly calling a missing dispatcher."""
    from run_pipeline import _invoke_judge

    state = make_state(run_dir)
    # Use a target_agent that's in the TargetAgent literal but NOT in the
    # global table (r2c-smoke-diagnostician is local-only).
    decision = _build_decision(
        stage_id="stage_3c",
        validator_label="schema_validation:smoke_diagnosis",
        action="dispatch_fix",
        target_agent="r2c-smoke-diagnostician",
    )
    _queue_judge_writing(fake_dispatch, [decision])

    outcome = _invoke_judge(
        state, stage_id="stage_3c",
        validator_label="schema_validation:smoke_diagnosis",
        stderr_tail="x", iteration=0,
        # NO local_dispatchers — diagnostician should NOT be reachable here
    )

    assert outcome.action == "halt"
    assert "unknown target_agent" in outcome.rationale
    assert "r2c-smoke-diagnostician" in outcome.rationale
    assert outcome.dispatcher is None


def test_invoke_judge_dispatch_fix_without_finding_downgrades_to_halt(
    fake_dispatch, run_dir
):
    """A dispatch_fix decision with no finding is malformed — driver
    downgrades to halt to prevent dispatching the producer with nothing
    to fix."""
    from run_pipeline import _invoke_judge

    state = make_state(run_dir)
    decision = _build_decision(
        action="dispatch_fix",
        target_agent="r2c-architecture-coder",
    )
    # Wipe the finding the helper set for us.
    decision["finding"] = None
    _queue_judge_writing(fake_dispatch, [decision])

    outcome = _invoke_judge(
        state, stage_id="stage_2d",
        validator_label="validate_arch_contract.py",
        stderr_tail="x", iteration=0,
    )

    assert outcome.action == "halt"
    assert "no finding" in outcome.rationale
    assert outcome.dispatcher is None


# ---------------------------------------------------------------------------
# local_dispatchers override
# ---------------------------------------------------------------------------


def test_invoke_judge_local_dispatchers_make_smoke_diagnostician_dispatchable(
    fake_dispatch, run_dir
):
    """At stage 3.c, the smoke-diagnostician needs runtime smoke context
    to do a repair-mode dispatch. `local_dispatchers` provides a context-
    bound closure; with it set, the judge can route to the diagnostician."""
    from run_pipeline import _invoke_judge

    state = make_state(run_dir)
    captured: list[tuple] = []

    def fake_diagnostician_repair(s, findings):
        captured.append((s, findings))
        return None  # mimics dispatch_with_scope_check return

    decision = _build_decision(
        stage_id="stage_3c",
        validator_label="schema_validation:smoke_diagnosis",
        action="dispatch_fix",
        target_agent="r2c-smoke-diagnostician",
        finding={"id": "JUDGE001", "severity": "critical",
                 "description": "missing value_origin_trace"},
    )
    _queue_judge_writing(fake_dispatch, [decision])

    outcome = _invoke_judge(
        state, stage_id="stage_3c",
        validator_label="schema_validation:smoke_diagnosis",
        stderr_tail="x", iteration=0,
        local_dispatchers={"r2c-smoke-diagnostician": fake_diagnostician_repair},
    )

    assert outcome.action == "dispatch_fix"
    assert outcome.target_agent == "r2c-smoke-diagnostician"
    assert outcome.dispatcher is fake_diagnostician_repair


def test_invoke_judge_local_dispatcher_overrides_global_for_same_agent(
    fake_dispatch, run_dir
):
    """When local_dispatchers has an entry for an agent that's ALSO in the
    global table, local wins."""
    from run_pipeline import _dispatch_arch_coder_fix, _invoke_judge

    state = make_state(run_dir)
    def local_arch_coder(s, findings):
        return None

    decision = _build_decision(
        action="dispatch_fix",
        target_agent="r2c-architecture-coder",
    )
    _queue_judge_writing(fake_dispatch, [decision])

    outcome = _invoke_judge(
        state, stage_id="stage_2d",
        validator_label="validate_arch_contract.py",
        stderr_tail="x", iteration=0,
        local_dispatchers={"r2c-architecture-coder": local_arch_coder},
    )

    assert outcome.dispatcher is local_arch_coder
    assert outcome.dispatcher is not _dispatch_arch_coder_fix


# ---------------------------------------------------------------------------
# Decision accumulation across iterations
# ---------------------------------------------------------------------------


def test_invoke_judge_reads_correct_iteration_from_accumulated_file(
    fake_dispatch, run_dir
):
    """When multiple iterations accumulate in judge_decisions.json, the
    driver matches by (stage_id, iteration) and returns the right one."""
    from run_pipeline import _invoke_judge

    state = make_state(run_dir)
    decision_iter0 = _build_decision(iteration=0, action="dispatch_fix",
                                      rationale="first")
    decision_iter1 = _build_decision(iteration=1, action="dispatch_fix",
                                      rationale="second")

    # Pre-populate the file with iter 0 (as if a prior _invoke_judge call ran)
    (run_dir / ".pipeline" / "judge_decisions.json").write_text(
        json.dumps([decision_iter0], indent=2)
    )

    # The next dispatch writes only iter 1; the driver appends it.
    _queue_judge_writing(fake_dispatch, decision_iter1)

    outcome = _invoke_judge(
        state, stage_id="stage_2d",
        validator_label="validate_arch_contract.py",
        stderr_tail="x", iteration=1,
    )

    assert outcome.action == "dispatch_fix"
    assert "second" in outcome.rationale


def test_invoke_judge_only_reads_decisions_for_current_stage(
    fake_dispatch, run_dir
):
    """Decisions from OTHER stages in the same file are filtered out."""
    from run_pipeline import _invoke_judge

    state = make_state(run_dir)
    other_stage_decision = _build_decision(stage_id="stage_2b", iteration=0,
                                            rationale="other-stage")
    current_stage_decision = _build_decision(stage_id="stage_2d", iteration=0,
                                              rationale="current-stage")

    _queue_judge_writing(
        fake_dispatch, [other_stage_decision, current_stage_decision]
    )

    outcome = _invoke_judge(
        state, stage_id="stage_2d",
        validator_label="validate_arch_contract.py",
        stderr_tail="x", iteration=0,
    )

    assert "current-stage" in outcome.rationale


def test_invoke_judge_preserves_prior_canonical_history(fake_dispatch, run_dir):
    """The judge writes only the current scratch decision; the driver appends
    it to existing canonical history without dropping prior stages."""
    from run_pipeline import _invoke_judge

    state = make_state(run_dir)
    prior = _build_decision(stage_id="stage_2b", iteration=0,
                            rationale="prior-stage")
    current = _build_decision(stage_id="stage_2d", iteration=0,
                              rationale="current-stage")
    (run_dir / ".pipeline" / "judge_decisions.json").write_text(
        json.dumps([prior], indent=2),
        encoding="utf-8",
    )
    _queue_judge_writing(fake_dispatch, current)

    _invoke_judge(
        state, stage_id="stage_2d",
        validator_label="validate_arch_contract.py",
        stderr_tail="x", iteration=0,
    )

    history = json.loads(
        (run_dir / ".pipeline" / "judge_decisions.json").read_text()
    )
    assert [d["rationale"] for d in history] == ["prior-stage", "current-stage"]


def test_invoke_judge_normalizes_legacy_single_decision_history(
    fake_dispatch, run_dir
):
    """A halted run may have the old bug shape: canonical history is one
    valid decision object. The next append normalizes it to an array."""
    from run_pipeline import _invoke_judge

    state = make_state(run_dir)
    legacy = _build_decision(stage_id="stage_2b", iteration=0,
                             rationale="legacy-single")
    current = _build_decision(stage_id="stage_2d", iteration=0,
                              rationale="current")
    (run_dir / ".pipeline" / "judge_decisions.json").write_text(
        json.dumps(legacy, indent=2),
        encoding="utf-8",
    )
    _queue_judge_writing(fake_dispatch, current)

    _invoke_judge(
        state, stage_id="stage_2d",
        validator_label="validate_arch_contract.py",
        stderr_tail="x", iteration=0,
    )

    history = json.loads(
        (run_dir / ".pipeline" / "judge_decisions.json").read_text()
    )
    assert isinstance(history, list)
    assert [d["rationale"] for d in history] == ["legacy-single", "current"]


def test_invoke_judge_recovers_when_judge_also_corrupts_canonical(
    fake_dispatch, run_dir
):
    """Regression for the observed halt shape: the judge writes a valid
    scratch decision, then also writes an invalid canonical object. Scope
    recovery drops the out-of-scope canonical write and the driver records the
    scratch decision itself."""
    from run_pipeline import _invoke_judge

    state = make_state(run_dir)
    decision = _build_decision(rationale="scratch-wins")
    _queue_judge_writing(
        fake_dispatch,
        decision,
        extra_writes={
            ".pipeline/judge_decisions.json": '{"not": "driver-owned"}',
        },
    )

    _invoke_judge(
        state, stage_id="stage_2d",
        validator_label="validate_arch_contract.py",
        stderr_tail="x", iteration=0,
    )

    history = json.loads(
        (run_dir / ".pipeline" / "judge_decisions.json").read_text()
    )
    assert isinstance(history, list)
    assert len(history) == 1
    assert history[0]["rationale"] == "scratch-wins"


def test_invoke_judge_for_findings_uses_scoped_scratch_and_appends_history(
    fake_dispatch, run_dir
):
    """Reviewer cap-exhaustion uses the same scratch-object protocol and a
    deterministic validator_label derived from the reviewer stage."""
    from run_pipeline import _invoke_judge_for_findings

    state = make_state(run_dir)
    decision = _build_decision(
        stage_id="stage_2b",
        iteration=5,
        validator_label="reviewer_findings:stage_2b_architecture",
        target_agent="r2c-architecture-coder",
        rationale="one more architecture fix",
    )
    _queue_judge_writing(fake_dispatch, decision)

    outcome = _invoke_judge_for_findings(
        state,
        stage_id="stage_2b",
        reviewer_stage_id="stage_2b_architecture",
        findings=[{
            "id": "F001",
            "severity": "critical",
            "target_agent": "architecture-coder",
            "description": "still broken",
        }],
        iteration=5,
    )

    assert outcome.action == "dispatch_fix"
    assert outcome.target_agent == "r2c-architecture-coder"
    history = json.loads(
        (run_dir / ".pipeline" / "judge_decisions.json").read_text()
    )
    assert history[-1]["validator_label"] == "reviewer_findings:stage_2b_architecture"


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


def test_invoke_judge_no_matching_iteration_raises_with_on_disk_keys(
    fake_dispatch, run_dir
):
    """If the judge dispatch returns but writes the wrong iteration in the
    expected scratch file, _invoke_judge raises a contract error."""
    from run_pipeline import _invoke_judge

    state = make_state(run_dir)
    # Judge writes a decision for iteration=99 but we ask about iteration=0
    decision = _build_decision(iteration=99)
    for _ in range(2):  # driver retries once on an unusable decision
        _queue_judge_writing(
            fake_dispatch,
            decision,
            output_stage_id="stage_2d",
            output_iteration=0,
            output_validator_label="validate_arch_contract.py",
        )

    with pytest.raises(ValueError, match="contract"):
        _invoke_judge(
            state, stage_id="stage_2d",
            validator_label="validate_arch_contract.py",
            stderr_tail="x", iteration=0,
        )


def test_invoke_judge_missing_decisions_file_raises_distinct_error(
    fake_dispatch, run_dir
):
    """If the judge dispatch completes but writes NO scratch file, surface a
    distinct error pointing at 'did not write'."""
    from run_pipeline import _invoke_judge

    state = make_state(run_dir)
    # Queue TWO no-write dispatches — the driver retries once before the
    # distinct error surfaces.
    fake_dispatch.expect(agent="r2c-halt-judge", writes={})
    fake_dispatch.expect(agent="r2c-halt-judge", writes={})

    with pytest.raises(ValueError, match="did not write"):
        _invoke_judge(
            state, stage_id="stage_2d",
            validator_label="validate_arch_contract.py",
            stderr_tail="x", iteration=0,
        )


def test_invoke_judge_malformed_json_raises_distinct_error(fake_dispatch, run_dir):
    """If the judge writes invalid JSON, surface a distinct error pointing
    at 'not valid JSON' with the parse error."""
    from run_pipeline import _invoke_judge

    state = make_state(run_dir)
    from run_pipeline import _judge_decision_scratch_relpath

    scratch_rel = _judge_decision_scratch_relpath(
        stage_id="stage_2d",
        iteration=0,
        validator_label="validate_arch_contract.py",
    )
    for _ in range(2):  # driver retries once on an unusable decision
        fake_dispatch.expect(
            agent="r2c-halt-judge",
            writes={scratch_rel: "{ not json at all"},
        )

    with pytest.raises(ValueError, match="not valid JSON"):
        _invoke_judge(
            state, stage_id="stage_2d",
            validator_label="validate_arch_contract.py",
            stderr_tail="x", iteration=0,
        )


def test_invoke_judge_non_object_top_level_raises_distinct_error(
    fake_dispatch, run_dir
):
    """If the judge writes valid JSON but the scratch top-level value isn't
    an object, surface a distinct error."""
    from run_pipeline import _invoke_judge
    from run_pipeline import _judge_decision_scratch_relpath

    state = make_state(run_dir)
    scratch_rel = _judge_decision_scratch_relpath(
        stage_id="stage_2d",
        iteration=0,
        validator_label="validate_arch_contract.py",
    )
    # Valid JSON but an array instead of a dict at top level; queued
    # twice (the driver retries once on an unusable decision).
    for _ in range(2):
        fake_dispatch.expect(
            agent="r2c-halt-judge",
            writes={scratch_rel: '[{"not": "an object"}]'},
        )

    with pytest.raises(ValueError, match="not a top-level object"):
        _invoke_judge(
            state, stage_id="stage_2d",
            validator_label="validate_arch_contract.py",
            stderr_tail="x", iteration=0,
        )


def test_invoke_judge_schema_invalid_decision_raises(fake_dispatch, run_dir):
    """If the judge writes a decision that fails pydantic validation, the
    driver raises ValueError naming the validation error."""
    from run_pipeline import _invoke_judge

    state = make_state(run_dir)
    # Decision missing the required `rationale` field
    bad_decision = {
        "schema_version": "1.0.0",
        "stage_id": "stage_2d",
        "iteration": 0,
        "validator_label": "validate_arch_contract.py",
        "classification": "producer_fixable",
        "action": "dispatch_fix",
        "target_agent": "r2c-architecture-coder",
        "finding": {"id": "X", "severity": "critical", "description": "x"},
        # rationale: missing
        "confidence": "high",
    }
    for _ in range(2):  # driver retries once on an unusable decision
        _queue_judge_writing(fake_dispatch, [bad_decision])

    with pytest.raises(ValueError, match="invalid decision"):
        _invoke_judge(
            state, stage_id="stage_2d",
            validator_label="validate_arch_contract.py",
            stderr_tail="x", iteration=0,
        )


# ---------------------------------------------------------------------------
# Dispatcher table coherence
# ---------------------------------------------------------------------------


def test_judge_fix_dispatchers_subset_of_target_agent_literal():
    """Every key in JUDGE_FIX_DISPATCHERS must appear in the TargetAgent
    literal in schemas/judge_decision.py. Drift here is a bug."""
    import typing
    from run_pipeline import JUDGE_FIX_DISPATCHERS
    from schemas.judge_decision import TargetAgent

    literal_agents = set(typing.get_args(TargetAgent))
    table_agents = set(JUDGE_FIX_DISPATCHERS.keys())
    extras = table_agents - literal_agents
    assert not extras, (
        f"JUDGE_FIX_DISPATCHERS has agents not in the TargetAgent literal: "
        f"{extras}. Extend schemas/judge_decision.py's TargetAgent."
    )


def test_judge_fix_dispatchers_includes_all_global_agents():
    """The global JUDGE_FIX_DISPATCHERS table must include every agent that
    doesn't require runtime context. Today: arch-coder, method-coder,
    method-analyzer, decomposer, notebook-generator, paper-fidelity-reviewer.
    smoke-diagnostician is intentionally local-only (needs smoke context)."""
    from run_pipeline import JUDGE_FIX_DISPATCHERS

    expected_global = {
        "r2c-architecture-coder",
        "r2c-method-coder",
        "r2c-method-analyzer",
        "r2c-decomposer",
        "r2c-notebook-generator",
        "r2c-paper-fidelity-reviewer",
        "r2c-test-generator",
    }
    missing = expected_global - set(JUDGE_FIX_DISPATCHERS.keys())
    assert not missing, (
        f"JUDGE_FIX_DISPATCHERS is missing expected global entries: {missing}. "
        f"Likely a late-registration hook (after agent definition) didn't fire."
    )


# ---------------------------------------------------------------------------
# Class B reliability: Think-class contract-field-verbatim verification
# ---------------------------------------------------------------------------


def test_invoke_judge_stage_id_contract_violation_halts_cleanly(
    fake_dispatch, run_dir
):
    """R2C 2026-05-20 regression: the halt-judge wrote `stage_id='stage_1a'`
    instead of `stage_id='stage_1'` because the Think-class agent didn't
    copy the contract field verbatim. The driver must catch this with a
    clear 'contract violation' halt — not a cryptic 'no decision matches'
    error that obscures the actual failure mode."""
    from run_pipeline import _invoke_judge

    state = make_state(run_dir)
    # Build a decision with the WRONG stage_id (driver asked for stage_1,
    # agent wrote stage_1a). The (stage_id, iteration) lookup will miss.
    decision = _build_decision(stage_id="stage_1a", iteration=0)
    for _ in range(2):  # driver retries once on an unusable decision
        _queue_judge_writing(
            fake_dispatch,
            decision,
            output_stage_id="stage_1",
            output_iteration=0,
            output_validator_label="validate_paper_map.py",
        )

    try:
        _invoke_judge(
            state, stage_id="stage_1",
            validator_label="validate_paper_map.py",
            stderr_tail="schema validation failed", iteration=0,
        )
        raise AssertionError("expected ValueError for contract violation")
    except ValueError as e:
        msg = str(e)
        # The error must specifically name the contract violation —
        # not the generic "no decision matches" pre-fix message.
        assert "contract" in msg.lower(), f"halt msg missing 'contract': {msg}"
        assert "stage_1" in msg, f"halt msg should name expected stage_id: {msg}"
        assert "stage_1a" in msg, f"halt msg should name actual stage_id: {msg}"


def test_invoke_judge_retry_after_no_decision_carries_write_first_preamble(
    fake_dispatch, run_dir
):
    """When the first judge dispatch produces no usable decision, the retry
    dispatch must NOT be an identical prompt — it carries the write-first
    preamble (detr 2026-07-04: identical-prompt retries die identically when
    the first turn burned its output budget on pre-write reasoning). The
    first dispatch stays preamble-free."""
    from run_pipeline import _invoke_judge

    state = make_state(run_dir)
    # Attempt 1: judge writes nothing. Attempt 2: valid decision.
    fake_dispatch.expect(agent="r2c-halt-judge", writes={})
    decision = _build_decision(
        stage_id="stage_1", validator_label="validate_paper_map.py",
        action="halt", classification="upstream_issue",
    )
    _queue_judge_writing(fake_dispatch, decision)

    outcome = _invoke_judge(
        state, stage_id="stage_1",
        validator_label="validate_paper_map.py",
        stderr_tail="schema validation failed", iteration=0,
    )
    assert outcome.action == "halt"
    assert len(fake_dispatch.calls) == 2
    first, second = fake_dispatch.calls
    assert "WRITE FIRST" not in first.prompt, (
        "initial judge dispatch must not carry the retry preamble")
    assert "WRITE FIRST" in second.prompt, (
        "no-decision retry must carry the write-first preamble")
    assert "FIRST tool call must be the Write" in second.prompt
    assert "imperfect but honest decision" in second.prompt
    # The preamble prepends; the original task must survive unchanged after it.
    assert "Your original task (unchanged) follows." in second.prompt
    assert "Halt-recovery judgment" in second.prompt


def test_invoke_judge_for_findings_retry_carries_write_first_preamble(
    fake_dispatch, run_dir
):
    """Same write-first retry contract on the reviewer-findings judge path
    (tier-3 cap exhaustion) — both judge dispatchers share the recovery."""
    from run_pipeline import _invoke_judge_for_findings

    state = make_state(run_dir)
    fake_dispatch.expect(agent="r2c-halt-judge", writes={})
    validator_label = "reviewer_findings:stage_2c_method"
    decision = _build_decision(
        stage_id="stage_2c", validator_label=validator_label,
        action="halt", classification="upstream_issue",
    )
    _queue_judge_writing(fake_dispatch, decision)

    outcome = _invoke_judge_for_findings(
        state, stage_id="stage_2c", reviewer_stage_id="stage_2c_method",
        findings=[{"id": "F001", "severity": "critical",
                   "description": "persisting", "file": "method/method.py"}],
        iteration=0,
    )
    assert outcome.action == "halt"
    assert len(fake_dispatch.calls) == 2
    assert "WRITE FIRST" not in fake_dispatch.calls[0].prompt
    assert "WRITE FIRST" in fake_dispatch.calls[1].prompt


def test_invoke_judge_no_decisions_distinct_from_contract_violation(
    fake_dispatch, run_dir
):
    """When no scratch decision exists at all, we should NOT report a
    contract violation — that's a distinct missing-output failure mode."""
    from run_pipeline import _invoke_judge

    state = make_state(run_dir)
    fake_dispatch.expect(agent="r2c-halt-judge", writes={})
    fake_dispatch.expect(agent="r2c-halt-judge", writes={})

    try:
        _invoke_judge(
            state, stage_id="stage_1",
            validator_label="validate_paper_map.py",
            stderr_tail="x", iteration=0,
        )
        raise AssertionError("expected ValueError")
    except ValueError as e:
        msg = str(e)
        # Missing output should stay distinct from contract-field mismatch.
        assert "contract" not in msg.lower(), (
            f"missing scratch output should NOT report as contract violation: {msg}"
        )
        assert "did not write" in msg


def test_verify_agent_contract_fields_match_returns_none():
    """Happy path: every expected field present with matching value → None."""
    from run_pipeline import _verify_agent_contract_fields

    err = _verify_agent_contract_fields(
        written={"stage_id": "stage_1", "iteration": 0, "extra": "ignored"},
        expected={"stage_id": "stage_1", "iteration": 0},
        agent="r2c-test", artifact="test.json",
    )
    assert err is None


def test_verify_agent_contract_fields_mismatch_returns_specific_error():
    """Value mismatch: error names the agent, artifact, field, both values."""
    from run_pipeline import _verify_agent_contract_fields

    err = _verify_agent_contract_fields(
        written={"stage_id": "stage_1a"},
        expected={"stage_id": "stage_1"},
        agent="r2c-halt-judge", artifact="judge_decisions.json",
    )
    assert err is not None
    assert "r2c-halt-judge" in err
    assert "judge_decisions.json" in err
    assert "stage_1a" in err  # what agent wrote
    assert "stage_1'" in err or "'stage_1'" in err  # what we sent (delimited)
    assert "verbatim" in err.lower()


def test_verify_agent_contract_fields_missing_field_returns_specific_error():
    """Missing field: error names it specifically — distinct from value-mismatch."""
    from run_pipeline import _verify_agent_contract_fields

    err = _verify_agent_contract_fields(
        written={"iteration": 0},  # no stage_id
        expected={"stage_id": "stage_1"},
        agent="r2c-halt-judge", artifact="judge_decisions.json",
    )
    assert err is not None
    assert "missing required field" in err
    assert "stage_id" in err


# ---------------------------------------------------------------------------
# A7 decision execution (parked-to-Track-A escalation): the judge's routing
# VOCABULARY and its routing TABLE must agree, everywhere the vocabulary is
# written down. A template or schema offering an agent the table cannot
# resolve makes the judge's pick silently downgrade to halt.
# ---------------------------------------------------------------------------


def test_judge_routing_vocabulary_matches_the_dispatcher_table():
    import re
    import typing

    import run_pipeline
    from schemas.judge_decision import TargetAgent

    vocabulary = set(typing.get_args(TargetAgent))
    table = set(run_pipeline.JUDGE_FIX_DISPATCHERS.keys())

    # The one documented exception: the smoke-diagnostician's dispatcher is
    # context-bound (stage 3.c passes it via local_dispatchers) and is
    # DELIBERATELY absent from the global table — outside stage 3.c the
    # judge's pick must downgrade to halt.
    assert "r2c-smoke-diagnostician" in vocabulary
    assert "r2c-smoke-diagnostician" not in table

    assert vocabulary - {"r2c-smoke-diagnostician"} == table, (
        f"schema vocabulary and JUDGE_FIX_DISPATCHERS disagree — "
        f"only in schema: {sorted(vocabulary - {'r2c-smoke-diagnostician'} - table)}; "
        f"only in table: {sorted(table - vocabulary)}. "
        f"Extend the TargetAgent literal and the table together.")

    # Every agent name a judge-facing template offers must be in the schema
    # vocabulary (the element-test template names the code/test pair).
    import dispatch_templates as dt

    for template_name in ("JUDGE_TASK_TEMPLATE", "JUDGE_REVIEWER_TASK_TEMPLATE",
                          "JUDGE_ELEMENT_TEST_TEMPLATE"):
        offered = set(re.findall(r"r2c-[a-z-]+", getattr(dt, template_name)))
        assert offered <= vocabulary, (
            f"{template_name} offers {sorted(offered - vocabulary)} "
            f"outside the TargetAgent vocabulary")

    # The halt-judge agent doc's target_agent enum line must offer exactly
    # names the schema knows (the doc may omit some; it must invent none).
    doc = (Path(__file__).resolve().parent.parent
           / ".opencode" / "agents" / "r2c-halt-judge.md").read_text(encoding="utf-8")
    enum_lines = [l for l in doc.splitlines() if '"target_agent":' in l and "|" in l]
    assert enum_lines, "the judge doc no longer carries its target_agent enum line"
    for line in enum_lines:
        offered = set(re.findall(r"r2c-[a-z-]+", line))
        assert offered <= vocabulary, (
            f"r2c-halt-judge.md enum offers {sorted(offered - vocabulary)} "
            f"outside the TargetAgent vocabulary")
