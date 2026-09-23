"""Phase 4a — Stage 2.b (architecture-coder: model.py + training.py + arch_contract).

Tests `run_stage_2b` which uses the standard `run_fix_loop(use_judge=True)`
pattern: producer dispatch → validator → reviewer. Most failure paths route
through the halt-judge.

Same fix-loop pattern as 2.c (different producer); these tests amortize.
"""

from __future__ import annotations

import json
from pathlib import Path

from tests.helpers.assertions import (
    assert_dispatched,
    assert_not_dispatched,
    assert_stage_completed,
    assert_stage_degraded,
    assert_stage_halted,
    assert_stage_skipped,
)
from tests.helpers.inject import (
    minimal_method_spec,
    queue_judge_decision_object,
    write_stage_complete_sentinel,
)
from tests.helpers.state import make_state

import run_layout


TRAINING_WITH_BUILD_MODEL = "def build_model(*args, **kwargs):\n    return None\n"


def _seed_spec(state, paradigm_id: str = "supervised_ml/active_learning"):
    spec = minimal_method_spec(paradigm_id=paradigm_id)
    state.paths.method_spec.write_text(json.dumps(spec, indent=2), encoding="utf-8")


def _seed_2b_outputs(state):
    """Seed model.py + training.py + arch_contract.json as if a prior
    stage-2.b run produced them. Used by skip-check tests."""
    rd = state.paths.run_dir
    (rd / "method").mkdir(parents=True, exist_ok=True)
    (rd / "method" / "model.py").write_text("# fake model\n")
    (rd / "method" / "training.py").write_text(TRAINING_WITH_BUILD_MODEL)
    (rd / ".pipeline" / "arch_contract.json").write_text("{}\n")
    (rd / ".pipeline" / "stage_review_stage_2b_architecture.json").write_text(
        "{}\n"
    )


def _clean_stage_review() -> str:
    return json.dumps({
        "schema_version": "1.0.0",
        "stage_id": "stage_2b_architecture",
        "review_status": "passed",
        "summary": "All checks pass.",
        "findings": [],
        "checks_summary": [],
        "caveats": [],
    }, indent=2)


def _stage_review_with_critical(finding_id: str = "F001") -> str:
    return json.dumps({
        "schema_version": "1.0.0",
        "stage_id": "stage_2b_architecture",
        "review_status": "issues_found",
        "summary": "Critical finding.",
        "findings": [{
            "id": finding_id,
            "check_id": "test_check",
            "severity": "critical",
            "target_agent": "architecture-coder",
            "issue_type": "other",
            "file": "method/model.py",
            "location": "class TestModel",
            "description": "test finding",
            "proposed_fix": "fix it",
            "proposed_resolution": None,
            "resolution_status": "pending",
        }],
        "checks_summary": [],
        "caveats": [],
    }, indent=2)


def _judge_decision_dispatch_fix(
    iteration: int = 0,
    validator_label: str = "validate_architecture_coder_output.py",
) -> dict:
    return {
        "schema_version": "1.0.0",
        "stage_id": "stage_2b",
        "iteration": iteration,
        "validator_label": validator_label,
        "classification": "producer_fixable",
        "action": "dispatch_fix",
        "target_agent": "r2c-architecture-coder",
        "finding": {
            "id": "JUDGE001", "severity": "critical",
            "description": "test", "proposed_fix": "fix",
        },
        "rationale": "test rationale",
        "confidence": "high",
        "files_examined": ["scripts/run_pipeline.py"],
    }


def _judge_decision_halt(
    iteration: int,
    validator_label: str = "reviewer_findings:stage_2b_architecture",
) -> dict:
    return {
        "schema_version": "1.0.0",
        "stage_id": "stage_2b",
        "iteration": iteration,
        "validator_label": validator_label,
        "classification": "pipeline_bug",
        "action": "halt",
        "target_agent": None,
        "finding": None,
        "rationale": "test halt rationale",
        "confidence": "high",
        "files_examined": ["scripts/run_pipeline.py"],
    }


# ---------------------------------------------------------------------------
# Skip behavior
# ---------------------------------------------------------------------------


def test_stage_2b_skipped_when_sentinel_and_outputs_present(run_dir):
    """Sentinel + outputs + no halt → skip, no LLM dispatched."""
    from run_pipeline import run_stage_2b
    state = make_state(run_dir)
    _seed_2b_outputs(state)
    write_stage_complete_sentinel(run_dir, "stage_2b")

    result = run_stage_2b(state)
    assert_stage_skipped(result, "stage_2b")


def test_stage_2b_outputs_present_no_sentinel_runs(
    fake_dispatch, fake_subprocess, run_dir
):
    """bev-distill 2026-05-19 regression at stage 2.b: outputs alone don't
    skip without the complete-sentinel."""
    from run_pipeline import run_stage_2b
    state = make_state(run_dir)
    _seed_spec(state)
    _seed_2b_outputs(state)  # outputs but NO sentinel

    # Producer dispatch (initial) + validator + reviewer (all happy)
    fake_dispatch.expect(
        agent="r2c-architecture-coder",
        writes={"method/model.py": "# fresh model\n",
                "method/training.py": TRAINING_WITH_BUILD_MODEL,
                ".pipeline/arch_contract.json": "{}\n"},
    )
    fake_subprocess.expect_stage2_script(ok=True)  # validate_architecture_coder_output
    fake_dispatch.expect(
        agent="r2c-stage-reviewer",
        writes={".pipeline/stage_review_stage_2b_architecture.json":
                _clean_stage_review()},
    )

    result = run_stage_2b(state)
    assert_stage_completed(result, "stage_2b")


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


def test_stage_2b_happy_path_completes(fake_dispatch, fake_subprocess, run_dir):
    """First dispatch produces clean outputs; validator + reviewer pass."""
    from run_pipeline import run_stage_2b
    state = make_state(run_dir)
    _seed_spec(state)

    fake_dispatch.expect(
        agent="r2c-architecture-coder",
        writes={"method/model.py": "# model\n",
                "method/training.py": TRAINING_WITH_BUILD_MODEL,
                ".pipeline/arch_contract.json": "{}\n"},
    )
    fake_subprocess.expect_stage2_script(ok=True)
    fake_dispatch.expect(
        agent="r2c-stage-reviewer",
        writes={".pipeline/stage_review_stage_2b_architecture.json":
                _clean_stage_review()},
    )

    result = run_stage_2b(state)
    assert_stage_completed(result, "stage_2b")
    assert_dispatched(fake_dispatch, "r2c-architecture-coder", times=1)
    assert_dispatched(fake_dispatch, "r2c-stage-reviewer", times=1)


# ---------------------------------------------------------------------------
# Validator failure → judge → fix
# ---------------------------------------------------------------------------


def test_stage_2b_validator_fails_judge_recovers(
    fake_dispatch, fake_subprocess, run_dir
):
    """validate_architecture_coder_output.py fails iter 0 → judge → arch-
    coder fix-mode → iter 1 validator passes → reviewer passes → completed."""
    from run_pipeline import run_stage_2b
    state = make_state(run_dir)
    _seed_spec(state)

    fake_dispatch.expect(
        agent="r2c-architecture-coder",
        writes={"method/model.py": "# bad model\n",
                "method/training.py": TRAINING_WITH_BUILD_MODEL,
                ".pipeline/arch_contract.json": "{}\n"},
    )
    # iter 0: validator fails
    fake_subprocess.expect_stage2_script(ok=False, err="VAL001: bad model")
    # judge invoked
    queue_judge_decision_object(
        fake_dispatch, _judge_decision_dispatch_fix(iteration=0),
    )
    # arch-coder fix-mode
    fake_dispatch.expect(
        agent="r2c-architecture-coder",
        writes={"method/model.py": "# fixed model\n"},
    )
    # iter 1: validator passes
    fake_subprocess.expect_stage2_script(ok=True)
    # reviewer dispatches + clean
    fake_dispatch.expect(
        agent="r2c-stage-reviewer",
        writes={".pipeline/stage_review_stage_2b_architecture.json":
                _clean_stage_review()},
    )

    result = run_stage_2b(state)
    assert_stage_completed(result, "stage_2b")
    assert_dispatched(fake_dispatch, "r2c-architecture-coder", times=2)
    assert_dispatched(fake_dispatch, "r2c-halt-judge", times=1)


def test_stage_2b_validator_fix_no_write_gets_one_immediate_retry(
    fake_dispatch, fake_subprocess, run_dir
):
    """A fix-mode producer turn that writes no allowed artifact is stochastic.
    Retry it once immediately with the stronger no-write finding before
    burning another validator/judge cycle.
    """
    from run_pipeline import run_stage_2b
    state = make_state(run_dir)
    _seed_spec(state)

    fake_dispatch.expect(
        agent="r2c-architecture-coder",
        writes={"method/model.py": "# bad model\n",
                "method/training.py": TRAINING_WITH_BUILD_MODEL,
                ".pipeline/arch_contract.json": "{}\n"},
    )
    fake_subprocess.expect_stage2_script(ok=False, err="VAL001: bad model")
    queue_judge_decision_object(
        fake_dispatch, _judge_decision_dispatch_fix(iteration=0),
    )
    # First fix-mode dispatch completes but writes nothing.
    fake_dispatch.expect(agent="r2c-architecture-coder", writes={})
    # Immediate no-write retry produces the actual fix.
    fake_dispatch.expect(
        agent="r2c-architecture-coder",
        writes={"method/model.py": "# fixed model\n"},
    )
    fake_subprocess.expect_stage2_script(ok=True)
    fake_dispatch.expect(
        agent="r2c-stage-reviewer",
        writes={".pipeline/stage_review_stage_2b_architecture.json":
                _clean_stage_review()},
    )

    result = run_stage_2b(state)
    assert_stage_completed(result, "stage_2b")
    assert_dispatched(fake_dispatch, "r2c-architecture-coder", times=3)
    assert_dispatched(fake_dispatch, "r2c-halt-judge", times=1)
    assert "completed without writing any allowed output" in (
        fake_dispatch.calls[3].prompt
    )


def test_stage_2b_judge_halt_with_missing_required_artifact_is_actionable(
    fake_dispatch, fake_subprocess, run_dir
):
    """When the judge halts and a required producer artifact is still absent,
    the halt classifies as a judge halt so the researcher surfaces render
    the catalog story, with the judge's rationale preserved in the technical
    reason and classification context — never as the headline (item 12
    rule 3, which retired the earlier quoted-rationale user_message here).
    """
    from run_pipeline import run_stage_2b
    state = make_state(run_dir)
    state.backend_unreachable = True  # keep the test off the TUI-post path
    _seed_spec(state)

    fake_dispatch.expect(
        agent="r2c-architecture-coder",
        writes={"method/model.py": "# model only\n",
                ".pipeline/arch_contract.json": "{}\n"},
    )
    fake_subprocess.expect_stage2_script(
        ok=False,
        err="FAIL: method/training.py not present",
    )
    decision = _judge_decision_halt(
        iteration=0,
        validator_label="validate_architecture_coder_output.py",
    )
    decision["classification"] = "unclear"
    decision["rationale"] = "oscillation: method/training.py stayed missing"
    queue_judge_decision_object(fake_dispatch, decision)

    result = run_stage_2b(state)
    assert_stage_halted(
        result, stage_id="stage_2b", reason_contains="method/training.py"
    )
    assert result.halt_artifact["halt_class"] == "judge_halt"
    assert "user_message" not in result.halt_artifact
    # The rationale is preserved for engineering in the technical reason.
    assert "method/training.py stayed missing" in result.halt_artifact["reason"]
    assert result.halt_artifact["context"]["judge_classification"] == "unclear"


def test_stage_2b_validator_cap_exhausted_degrades_when_required_outputs_exist(
    fake_dispatch, fake_subprocess, run_dir
):
    """Known-good for `required_producer_contract_missing`: validator failures
    may still degrade when model.py, training.py, and arch_contract.json all
    exist, so downstream stages have a complete producer contract to consume."""
    from run_pipeline import STAGE_2_RETRY_CAP, run_stage_2b
    state = make_state(run_dir)
    _seed_spec(state)

    # Initial producer dispatch
    fake_dispatch.expect(
        agent="r2c-architecture-coder",
        writes={"method/model.py": "# bad\n",
                "method/training.py": TRAINING_WITH_BUILD_MODEL,
                ".pipeline/arch_contract.json": "{}\n"},
    )
    # STAGE_2_RETRY_CAP+1 validator failures
    for i in range(STAGE_2_RETRY_CAP + 1):
        fake_subprocess.expect_stage2_script(ok=False, err=f"iter {i} bad")
        if i < STAGE_2_RETRY_CAP:
            decision = _judge_decision_dispatch_fix(iteration=i)
            queue_judge_decision_object(fake_dispatch, decision)
            fake_dispatch.expect(agent="r2c-architecture-coder",
                                  writes={"method/model.py": f"# attempt {i}\n"})

    result = run_stage_2b(state)
    assert_stage_degraded(result, stage_id="stage_2b", run_dir=run_dir,
                          known_issue_contains=f"cap={STAGE_2_RETRY_CAP}")
    assert_dispatched(fake_dispatch, "r2c-halt-judge", times=STAGE_2_RETRY_CAP)


def test_stage_2b_validator_cap_exhausted_halts_when_arch_contract_missing(
    fake_dispatch, fake_subprocess, run_dir
):
    """Known-bad for `required_producer_contract_missing`: architecture-coder
    may write code files but omit its contract artifact. That is not a usable
    degraded architecture stage because method/notebook stages depend on
    `.pipeline/arch_contract.json`; halt for a clean re-run instead."""
    from run_pipeline import STAGE_2_RETRY_CAP, run_stage_2b
    state = make_state(run_dir)
    _seed_spec(state)

    fake_dispatch.expect(
        agent="r2c-architecture-coder",
        writes={"method/model.py": "# bad\n",
                "method/training.py": TRAINING_WITH_BUILD_MODEL},
    )
    for i in range(STAGE_2_RETRY_CAP + 1):
        fake_subprocess.expect_stage2_script(
            ok=False,
            err=f"iter {i}: .pipeline/arch_contract.json missing",
        )
        if i < STAGE_2_RETRY_CAP:
            queue_judge_decision_object(
                fake_dispatch, _judge_decision_dispatch_fix(iteration=i),
            )
            fake_dispatch.expect(
                agent="r2c-architecture-coder",
                writes={"method/model.py": f"# attempt {i}\n"},
            )

    result = run_stage_2b(state)
    assert_stage_halted(
        result, stage_id="stage_2b",
        reason_contains=".pipeline/arch_contract.json",
    )
    assert_dispatched(fake_dispatch, "r2c-halt-judge", times=STAGE_2_RETRY_CAP)


def test_stage_2b_validator_cap_exhausted_halts_when_build_model_missing(
    fake_dispatch, fake_subprocess, run_dir
):
    """Known-bad for `required_producer_contract_missing`: a training.py file
    without the required build_model entrypoint is not a consumable Stage 2.b
    contract, even when all required files exist.

    Seeds the resolvable legacy AL id: the required symbols are now DERIVED
    from the paradigm's build-plan manifest (item 25, overnight 2026-07-08 —
    the prior hardcoded table fabricated `build_model` halt reasons on
    non-AL papers), so this test's spec must resolve to the AL plan for
    build_model to be genuinely required."""
    from run_pipeline import STAGE_2_RETRY_CAP, run_stage_2b
    state = make_state(run_dir)
    _seed_spec(state, paradigm_id="active_learning")

    fake_dispatch.expect(
        agent="r2c-architecture-coder",
        writes={"method/model.py": "# bad\n",
                "method/training.py": "# no build_model here\n",
                ".pipeline/arch_contract.json": "{}\n"},
    )
    for i in range(STAGE_2_RETRY_CAP + 1):
        fake_subprocess.expect_stage2_script(
            ok=False,
            err=f"iter {i}: method/training.py missing build_model",
        )
        if i < STAGE_2_RETRY_CAP:
            queue_judge_decision_object(
                fake_dispatch, _judge_decision_dispatch_fix(iteration=i),
            )
            fake_dispatch.expect(
                agent="r2c-architecture-coder",
                writes={"method/model.py": f"# attempt {i}\n"},
            )

    result = run_stage_2b(state)
    assert_stage_halted(
        result, stage_id="stage_2b",
        reason_contains="method/training.py::build_model",
    )
    assert_dispatched(fake_dispatch, "r2c-halt-judge", times=STAGE_2_RETRY_CAP)


# ---------------------------------------------------------------------------
# Reviewer critical findings
# ---------------------------------------------------------------------------


def test_stage_2b_reviewer_critical_findings_dispatch_arch_coder_fix_recovers(
    fake_dispatch, fake_subprocess, run_dir
):
    """Reviewer reports critical findings → architecture-coder fix-mode →
    iter 1 validator + reviewer both clean → completed."""
    from run_pipeline import run_stage_2b
    state = make_state(run_dir)
    _seed_spec(state)

    fake_dispatch.expect(
        agent="r2c-architecture-coder",
        writes={"method/model.py": "# model\n",
                "method/training.py": TRAINING_WITH_BUILD_MODEL,
                ".pipeline/arch_contract.json": "{}\n"},
    )
    # iter 0: validator passes
    fake_subprocess.expect_stage2_script(ok=True)
    # reviewer reports critical
    fake_dispatch.expect(
        agent="r2c-stage-reviewer",
        writes={".pipeline/stage_review_stage_2b_architecture.json":
                _stage_review_with_critical()},
    )
    # arch-coder fix-mode (reviewer-finding path is DIRECT — no judge involved
    # for reviewer findings in v1 per `run_fix_loop`'s tier-3 path)
    fake_dispatch.expect(
        agent="r2c-architecture-coder",
        writes={"method/model.py": "# fixed\n"},
    )
    # iter 1: validator + reviewer both clean
    fake_subprocess.expect_stage2_script(ok=True)
    fake_dispatch.expect(
        agent="r2c-stage-reviewer",
        writes={".pipeline/stage_review_stage_2b_architecture.json":
                _clean_stage_review()},
    )

    result = run_stage_2b(state)
    assert_stage_completed(result, "stage_2b")
    # Judge NOT invoked for reviewer-finding fix (that's the design — reviewer
    # findings already carry target_agent; only validator failures route
    # through the judge).
    assert_not_dispatched(fake_dispatch, "r2c-halt-judge")


def test_stage_2b_reviewer_critical_persists_after_cap_degrades(
    fake_dispatch, fake_subprocess, run_dir
):
    """Reviewer keeps reporting critical findings through STAGE_2_RETRY_CAP
    iterations → at cap, halt-judge invoked on persisting findings → judge
    decides halt → stage DEGRADES (halt→degrade): the architecture artifact
    exists, so the rationale is logged to KNOWN_ISSUES.md and the run continues."""
    from run_pipeline import STAGE_2_RETRY_CAP, run_stage_2b
    state = make_state(run_dir)
    _seed_spec(state)

    # Initial producer dispatch
    fake_dispatch.expect(
        agent="r2c-architecture-coder",
        writes={"method/model.py": "# model\n",
                "method/training.py": TRAINING_WITH_BUILD_MODEL,
                ".pipeline/arch_contract.json": "{}\n"},
    )
    # iters 0..cap-1: validator pass, reviewer critical, arch-coder fix
    for i in range(STAGE_2_RETRY_CAP):
        fake_subprocess.expect_stage2_script(ok=True)
        fake_dispatch.expect(
            agent="r2c-stage-reviewer",
            writes={".pipeline/stage_review_stage_2b_architecture.json":
                    _stage_review_with_critical(finding_id=f"F{i:03d}")},
        )
        fake_dispatch.expect(agent="r2c-architecture-coder",
                              writes={"method/model.py": f"# attempt {i}\n"})
    # iter cap: validator + reviewer (still critical) + halt-judge on
    # persisting findings → judge halts.
    fake_subprocess.expect_stage2_script(ok=True)
    fake_dispatch.expect(
        agent="r2c-stage-reviewer",
        writes={".pipeline/stage_review_stage_2b_architecture.json":
                _stage_review_with_critical(finding_id="F_final")},
    )
    queue_judge_decision_object(
        fake_dispatch, _judge_decision_halt(iteration=STAGE_2_RETRY_CAP),
    )

    result = run_stage_2b(state)
    assert_stage_degraded(result, stage_id="stage_2b", run_dir=run_dir,
                          known_issue_contains="halt-judge decided to halt")
    assert_dispatched(fake_dispatch, "r2c-halt-judge")


# ---------------------------------------------------------------------------
# Tier-3 target_agent routing (the bev-distill 2026-05-20 regression)
# ---------------------------------------------------------------------------


def _stage_review_with_critical_for_agent(
    target_agent: str | None, finding_id: str = "F001",
    file: str = "method/model.py",
) -> str:
    """Variant of `_stage_review_with_critical` parameterized by
    target_agent. Used to test the tier-3 routing primitive."""
    finding = {
        "id": finding_id,
        "check_id": "test_check",
        "severity": "critical",
        "issue_type": "other",
        "file": file,
        "location": "class TestModel",
        "description": "test finding",
        "proposed_fix": "fix it",
        "proposed_resolution": None,
        "resolution_status": "pending",
    }
    if target_agent is not None:
        finding["target_agent"] = target_agent
    return json.dumps({
        "schema_version": "1.0.0",
        "stage_id": "stage_2b_architecture",
        "review_status": "issues_found",
        "summary": "Critical finding.",
        "findings": [finding],
        "checks_summary": [],
        "caveats": [],
    }, indent=2)


def test_stage_2b_reviewer_short_form_target_agent_normalized_routes(
    fake_dispatch, fake_subprocess, run_dir
):
    """Reviewer finding with short-form target_agent='architecture-coder'
    (per schemas/review_report.py) is normalized to 'r2c-architecture-coder'
    and routed through JUDGE_FIX_DISPATCHERS to the right dispatcher."""
    from run_pipeline import run_stage_2b
    state = make_state(run_dir)
    _seed_spec(state)

    fake_dispatch.expect(
        agent="r2c-architecture-coder",
        writes={"method/model.py": "# model\n",
                "method/training.py": TRAINING_WITH_BUILD_MODEL,
                ".pipeline/arch_contract.json": "{}\n"},
    )
    fake_subprocess.expect_stage2_script(ok=True)
    # Reviewer flags target_agent="architecture-coder" (short form)
    fake_dispatch.expect(
        agent="r2c-stage-reviewer",
        writes={".pipeline/stage_review_stage_2b_architecture.json":
                _stage_review_with_critical_for_agent("architecture-coder")},
    )
    # Normalization + JUDGE_FIX_DISPATCHERS lookup must route to arch-coder
    fake_dispatch.expect(
        agent="r2c-architecture-coder",
        writes={"method/model.py": "# fixed\n"},
    )
    # iter 1 clean
    fake_subprocess.expect_stage2_script(ok=True)
    fake_dispatch.expect(
        agent="r2c-stage-reviewer",
        writes={".pipeline/stage_review_stage_2b_architecture.json":
                _clean_stage_review()},
    )
    result = run_stage_2b(state)
    assert_stage_completed(result, "stage_2b")


def test_stage_2b_reviewer_cross_agent_target_routes_to_other_producer(
    fake_dispatch, fake_subprocess, run_dir
):
    """Reviewer flags a finding with target_agent != stage's own producer
    (the bev-distill 2026-05-20 case: stage 3.a notebook reviewer flagging
    training.py issues for r2c-architecture-coder). The routing primitive
    must dispatch the named agent, NOT fall back to fix_dispatch_fn."""
    from run_pipeline import run_stage_2b
    state = make_state(run_dir)
    _seed_spec(state)

    # Initial arch-coder dispatch
    fake_dispatch.expect(
        agent="r2c-architecture-coder",
        writes={"method/model.py": "# model\n",
                "method/training.py": TRAINING_WITH_BUILD_MODEL,
                ".pipeline/arch_contract.json": "{}\n"},
    )
    fake_subprocess.expect_stage2_script(ok=True)
    # Reviewer flags finding owned by method-coder (cross-agent)
    fake_dispatch.expect(
        agent="r2c-stage-reviewer",
        writes={".pipeline/stage_review_stage_2b_architecture.json":
                _stage_review_with_critical_for_agent(
                    "r2c-method-coder", file="method/method.py")},
    )
    # The routing primitive must dispatch method-coder, not arch-coder
    fake_dispatch.expect(
        agent="r2c-method-coder",
        writes={"method/method.py": "# fixed by method-coder\n"},
    )
    fake_subprocess.expect_stage2_script(ok=True)
    fake_dispatch.expect(
        agent="r2c-stage-reviewer",
        writes={".pipeline/stage_review_stage_2b_architecture.json":
                _clean_stage_review()},
    )
    result = run_stage_2b(state)
    assert_stage_completed(result, "stage_2b")
    assert_dispatched(fake_dispatch, "r2c-method-coder")


def test_stage_2b_reviewer_unknown_target_agent_halts_with_clear_reason(
    fake_dispatch, fake_subprocess, run_dir
):
    """Reviewer flags a finding with target_agent that has no registered
    dispatcher. The tier-3 routing must halt cleanly with a specific
    reason — NOT silently fall back to the stage producer (the old bug
    that caused F001/F002 to spin against notebook-generator)."""
    from run_pipeline import run_stage_2b
    state = make_state(run_dir)
    _seed_spec(state)

    fake_dispatch.expect(
        agent="r2c-architecture-coder",
        writes={"method/model.py": "# model\n",
                "method/training.py": TRAINING_WITH_BUILD_MODEL,
                ".pipeline/arch_contract.json": "{}\n"},
    )
    fake_subprocess.expect_stage2_script(ok=True)
    fake_dispatch.expect(
        agent="r2c-stage-reviewer",
        writes={".pipeline/stage_review_stage_2b_architecture.json":
                _stage_review_with_critical_for_agent("r2c-mystery-agent")},
    )
    # No further dispatches expected — halt immediately on unknown target_agent
    result = run_stage_2b(state)
    assert_stage_halted(result, stage_id="stage_2b",
                        reason_contains="r2c-mystery-agent")
    assert_not_dispatched(fake_dispatch, "r2c-mystery-agent")


def test_stage_2b_cap_exhausted_judge_dispatch_fix_recovers(
    fake_dispatch, fake_subprocess, run_dir
):
    """Cap-exhaustion + judge picks dispatch_fix → one post-cap producer
    dispatch + final validate+review → reviewer clean → stage completes.
    Exercises the post-cap recovery path in `_tier3_cap_exhausted_judge`."""
    from run_pipeline import STAGE_2_RETRY_CAP, run_stage_2b
    state = make_state(run_dir)
    _seed_spec(state)

    fake_dispatch.expect(
        agent="r2c-architecture-coder",
        writes={"method/model.py": "# model\n",
                "method/training.py": TRAINING_WITH_BUILD_MODEL,
                ".pipeline/arch_contract.json": "{}\n"},
    )
    # iters 0..cap-1: pass validator, critical reviewer, arch-coder fix
    for i in range(STAGE_2_RETRY_CAP):
        fake_subprocess.expect_stage2_script(ok=True)
        fake_dispatch.expect(
            agent="r2c-stage-reviewer",
            writes={".pipeline/stage_review_stage_2b_architecture.json":
                    _stage_review_with_critical(finding_id=f"F{i:03d}")},
        )
        fake_dispatch.expect(agent="r2c-architecture-coder",
                              writes={"method/model.py": f"# attempt {i}\n"})
    # iter cap: validator + reviewer (critical) + halt-judge → dispatch_fix
    fake_subprocess.expect_stage2_script(ok=True)
    fake_dispatch.expect(
        agent="r2c-stage-reviewer",
        writes={".pipeline/stage_review_stage_2b_architecture.json":
                _stage_review_with_critical(finding_id="F_final")},
    )
    queue_judge_decision_object(
        fake_dispatch,
        _judge_decision_dispatch_fix(
            iteration=STAGE_2_RETRY_CAP,
            validator_label="reviewer_findings:stage_2b_architecture",
        ),
    )
    # Post-cap producer dispatch + final validate + final reviewer (clean)
    fake_dispatch.expect(
        agent="r2c-architecture-coder",
        writes={"method/model.py": "# post-cap fix\n"},
    )
    fake_subprocess.expect_stage2_script(ok=True)
    fake_dispatch.expect(
        agent="r2c-stage-reviewer",
        writes={".pipeline/stage_review_stage_2b_architecture.json":
                _clean_stage_review()},
    )
    result = run_stage_2b(state)
    assert_stage_completed(result, "stage_2b")
    assert_dispatched(fake_dispatch, "r2c-halt-judge")


def test_stage_2b_cap_exhausted_judge_dispatch_fix_still_failing_degrades(
    fake_dispatch, fake_subprocess, run_dir
):
    """Cap-exhaustion + judge picks dispatch_fix → post-cap dispatch lands
    but the final review STILL has critical findings → DEGRADE (halt→degrade):
    the architecture artifact exists, so log to KNOWN_ISSUES.md and continue."""
    from run_pipeline import STAGE_2_RETRY_CAP, run_stage_2b
    state = make_state(run_dir)
    _seed_spec(state)

    fake_dispatch.expect(
        agent="r2c-architecture-coder",
        writes={"method/model.py": "# model\n",
                "method/training.py": TRAINING_WITH_BUILD_MODEL,
                ".pipeline/arch_contract.json": "{}\n"},
    )
    for i in range(STAGE_2_RETRY_CAP):
        fake_subprocess.expect_stage2_script(ok=True)
        fake_dispatch.expect(
            agent="r2c-stage-reviewer",
            writes={".pipeline/stage_review_stage_2b_architecture.json":
                    _stage_review_with_critical(finding_id=f"F{i:03d}")},
        )
        fake_dispatch.expect(agent="r2c-architecture-coder",
                              writes={"method/model.py": f"# attempt {i}\n"})
    fake_subprocess.expect_stage2_script(ok=True)
    fake_dispatch.expect(
        agent="r2c-stage-reviewer",
        writes={".pipeline/stage_review_stage_2b_architecture.json":
                _stage_review_with_critical(finding_id="F_final")},
    )
    queue_judge_decision_object(
        fake_dispatch,
        _judge_decision_dispatch_fix(
            iteration=STAGE_2_RETRY_CAP,
            validator_label="reviewer_findings:stage_2b_architecture",
        ),
    )
    fake_dispatch.expect(
        agent="r2c-architecture-coder",
        writes={"method/model.py": "# post-cap fix didn't help\n"},
    )
    fake_subprocess.expect_stage2_script(ok=True)
    # Final reviewer still has critical findings
    fake_dispatch.expect(
        agent="r2c-stage-reviewer",
        writes={".pipeline/stage_review_stage_2b_architecture.json":
                _stage_review_with_critical(finding_id="F_post_cap")},
    )
    result = run_stage_2b(state)
    assert_stage_degraded(
        result, stage_id="stage_2b", run_dir=run_dir,
        known_issue_contains="STILL persist after post-cap judge dispatch",
    )


# ---------------------------------------------------------------------------
# Class B: stage-reviewer contract-field-verbatim verification
# ---------------------------------------------------------------------------


def _stage_review_with_wrong_stage_id(actual_stage_id: str) -> str:
    """Build a clean (no critical findings) stage-review output whose
    `stage_id` is the WRONG value — to simulate a Think-class agent that
    didn't copy the contract field verbatim from the prompt."""
    return json.dumps({
        "schema_version": "1.0.0",
        "stage_id": actual_stage_id,  # WRONG — should be stage_2b_architecture
        "review_status": "passed",
        "summary": "test",
        "findings": [],
        "checks_summary": [],
        "caveats": [],
    }, indent=2)


def test_stage_2b_reviewer_writes_wrong_stage_id_degrades_with_contract_violation_logged(
    fake_dispatch, fake_subprocess, run_dir
):
    """Stage-reviewer writes its output with stage_id='stage_2b' instead of the
    prompt-passed 'stage_2b_architecture'. The architecture artifact passed its
    validator, so under R2 (audit 2026-06-25) the driver first RETRIES the
    reviewer once (an unusable envelope is reviewer-output fragility, not a
    producer fault); when the retry ALSO writes the wrong stage_id it DEGRADES,
    logging the contract-violation detail — naming the field + expected + actual
    values — to KNOWN_ISSUES.md rather than halting the run."""
    from run_pipeline import run_stage_2b
    state = make_state(run_dir)
    _seed_spec(state)

    fake_dispatch.expect(
        agent="r2c-architecture-coder",
        writes={"method/model.py": "# model\n",
                "method/training.py": TRAINING_WITH_BUILD_MODEL,
                ".pipeline/arch_contract.json": "{}\n"},
    )
    fake_subprocess.expect_stage2_script(ok=True)
    # Reviewer's output has the WRONG stage_id — both the initial dispatch and
    # the R2 unusable-output retry, so the degrade path is reached.
    fake_dispatch.expect(
        agent="r2c-stage-reviewer",
        writes={".pipeline/stage_review_stage_2b_architecture.json":
                _stage_review_with_wrong_stage_id("stage_2b")},
    )
    fake_dispatch.expect(
        agent="r2c-stage-reviewer",
        writes={".pipeline/stage_review_stage_2b_architecture.json":
                _stage_review_with_wrong_stage_id("stage_2b")},
    )

    result = run_stage_2b(state)
    assert_stage_degraded(result, stage_id="stage_2b", run_dir=run_dir,
                          known_issue_contains="violated its output contract")
    # KNOWN_ISSUES.md should name both the expected and actual values.
    ki = (run_dir / run_layout.KNOWN_ISSUES_MD).read_text(encoding="utf-8")
    assert "stage_2b_architecture" in ki
    assert "stage_2b'" in ki or "'stage_2b'" in ki


def test_no_write_retry_finding_demands_skeleton_only_when_artifact_missing():
    """The no-write retry re-ask is skeleton-first BY FIAT (maintainer-approved
    2026-07-04): the previous softer wording was ignored three times on the
    detr roll — the model composed the whole file in its reasoning channel
    into the 20k per-step cap. The retry now demands the write as the FIRST
    tool call and scopes a missing-artifact turn to the skeleton only."""
    from run_pipeline import _no_write_fix_retry_finding

    finding = {
        "id": "VAL001",
        "severity": "critical",
        "description": "method/method.py missing",
        "proposed_fix": "Produce method/method.py implementing Eq. 10.",
    }
    retry = _no_write_fix_retry_finding(
        finding,
        validator_label="validate_method_coder_output.py",
        stderr_tail="FAIL: method/method.py missing",
    )
    fix = retry["proposed_fix"]
    assert "FIRST tool call" in fix
    assert "ONE write call" in fix
    assert "Do NOT implement the algorithm this turn" in fix
    # The original ask survives as context but is explicitly subordinated.
    assert "Eq. 10" in fix
    assert "overrides its scope" in fix
    # The description still carries the validator evidence.
    assert "method/method.py missing" in retry["description"]
