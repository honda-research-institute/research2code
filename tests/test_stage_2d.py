"""Phase 2 — Stage 2.d (init-finalizer + package-imports + arch-contract).

Tests the fix-loop at the stage's contract-validation phase: validators
either pass and the stage completes, or fail and the halt-judge is
invoked to classify + route. Cap=5 iterations.

Validators here are subprocess scripts; tests mock them via `fake_subprocess`
to control outcome sequences. Real validator integration is covered by the
validators' own `--self-test` blocks, not this suite. Agents (halt-judge,
architecture-coder fix) are faked via `fake_dispatch`.
"""

from __future__ import annotations

import json
from pathlib import Path

from scripts.arch_contract_semantics import SemanticIssue, format_semantic_issue
from tests.helpers.assertions import (
    assert_dispatched,
    assert_not_dispatched,
    assert_sentinel_present,
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


def _seed_spec(state, paradigm_id: str = "active_learning"):
    """Write a minimal method_spec.json so field_guide lookup works."""
    spec = minimal_method_spec(paradigm_id=paradigm_id)
    state.paths.method_spec.write_text(json.dumps(spec, indent=2), encoding="utf-8")


def _seed_2d_inputs(state):
    """Write the producer-side outputs Stage 2.d expects to find at entry
    (post-2.b, post-2.c, post-init-finalize). Just file presence; content is
    minimal because validators are faked."""
    rd = state.paths.run_dir
    (rd / "method" / "__init__.py").parent.mkdir(parents=True, exist_ok=True)
    (rd / "method" / "__init__.py").write_text("# fake __init__\n")
    (rd / "requirements.txt").write_text(
        "r2c-stage2d-test-dependency-never-installed\n"
    )


def _semantic_stderr(*issues: SemanticIssue, label: str = "arch_contract") -> str:
    lines = [f"FAIL: {len(issues)} {label} validation error(s):"]
    lines.extend(f"  - {format_semantic_issue(issue)}" for issue in issues)
    return "\n".join(lines)


def _seed_phase_2(fake_subprocess, state) -> None:
    _seed_spec(state)
    _seed_2d_inputs(state)
    fake_subprocess.expect_script(returncode=0)
    fake_subprocess.expect_script(returncode=0)
    fake_subprocess.expect_stage2_script(ok=True)


def _build_judge_decision_dispatch_fix(iteration: int = 0) -> dict:
    return {
        "schema_version": "1.0.0",
        "stage_id": "stage_2d",
        "iteration": iteration,
        "validator_label": "validate_arch_contract.py",
        "classification": "producer_fixable",
        "action": "dispatch_fix",
        "target_agent": "r2c-architecture-coder",
        "finding": {
            "id": "JUDGE001", "severity": "critical",
            "description": "test arch_contract fix",
            "proposed_fix": "test",
        },
        "rationale": "test rationale",
        "confidence": "high",
        "files_examined": ["scripts/run_pipeline.py"],
    }


def _build_judge_decision_halt(iteration: int = 0,
                                rationale: str = "upstream issue") -> dict:
    return {
        "schema_version": "1.0.0",
        "stage_id": "stage_2d",
        "iteration": iteration,
        "validator_label": "validate_arch_contract.py",
        "classification": "upstream_issue",
        "action": "halt",
        "target_agent": None,
        "finding": None,
        "rationale": rationale,
        "confidence": "high",
        "files_examined": ["scripts/run_pipeline.py"],
    }


# ---------------------------------------------------------------------------
# Skip behavior
# ---------------------------------------------------------------------------


def test_stage_2d_skipped_when_sentinel_and_outputs_present(run_dir):
    """Sentinel + outputs + digest + no halt → stage skipped without
    subprocess or dispatch calls. (Post-R2C-2026-05-22 the smart-skip also
    requires the upstream content digest; this test models a clean prior
    completion that wrote both.)"""
    from run_pipeline import run_stage_2d, write_stage_2d_upstream_digest
    state = make_state(run_dir)
    # Seed upstream files (digest needs SOMETHING to hash)
    (run_dir / "method").mkdir(parents=True, exist_ok=True)
    (run_dir / "method" / "method.py").write_text("# method\n")
    (run_dir / ".pipeline").mkdir(parents=True, exist_ok=True)
    (run_dir / ".pipeline" / "arch_contract.json").write_text("{}\n")
    (run_dir / ".pipeline" / "method_spec.json").write_text("{}\n")
    _seed_2d_inputs(state)  # __init__.py + requirements.txt
    write_stage_complete_sentinel(run_dir, "stage_2d")
    write_stage_2d_upstream_digest(state.paths)

    result = run_stage_2d(state)
    assert_stage_skipped(result, "stage_2d")


def test_stage_2d_outputs_present_no_sentinel_runs(
    fake_dispatch, fake_subprocess, run_dir
):
    """**bev-distill 2026-05-19 regression**: outputs without sentinel must
    trigger a re-run (validators get exercised). This is the cascade-root
    test at the stage level."""
    from run_pipeline import run_stage_2d
    state = make_state(run_dir)
    _seed_spec(state)
    _seed_2d_inputs(state)  # outputs on disk
    # NO sentinel

    # finalize_package_init.py + validate_package_imports.py + both arch_contract validators
    fake_subprocess.expect_script(returncode=0)
    fake_subprocess.expect_script(returncode=0)  # pip install package requirements
    fake_subprocess.expect_stage2_script(ok=True)  # validate_package_imports
    fake_subprocess.expect_stage2_script(ok=True)  # validate_arch_contract.py
    fake_subprocess.expect_stage2_script(ok=True)  # validate_arch_contract_runtime.py

    result = run_stage_2d(state)
    assert_stage_completed(result, "stage_2d")
    # If skip had fired incorrectly, no subprocess calls would have happened.
    assert len(fake_subprocess.script_calls) == 2  # finalize + pip install
    assert len(fake_subprocess.stage2_calls) == 3


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


def test_stage_2d_happy_path_completes(fake_dispatch, fake_subprocess, run_dir):
    """All steps pass on first iteration → stage returns completed, no
    LLM agents dispatched."""
    from run_pipeline import run_stage_2d
    state = make_state(run_dir)
    _seed_spec(state)
    _seed_2d_inputs(state)

    fake_subprocess.expect_script(returncode=0)  # finalize_package_init.py
    fake_subprocess.expect_script(returncode=0)  # pip install package requirements
    fake_subprocess.expect_stage2_script(ok=True)  # validate_package_imports.py
    fake_subprocess.expect_stage2_script(ok=True)  # validate_arch_contract.py
    fake_subprocess.expect_stage2_script(ok=True)  # validate_arch_contract_runtime.py

    result = run_stage_2d(state)
    assert_stage_completed(result, "stage_2d")
    assert len(fake_dispatch.calls) == 0


# ---------------------------------------------------------------------------
# Deterministic-script failures (Phase 1: finalize + package imports)
# ---------------------------------------------------------------------------


def test_stage_2d_finalize_failure_judge_halt_decision_halts(
    fake_dispatch, fake_subprocess, run_dir
):
    """finalize_package_init.py exit != 0 → judge invoked; judge classifies
    it as not producer-fixable (e.g. a Stage 1 mis-declaration) and decides
    halt → stage halts with the judge-classified reason. The pre-2026-07-20
    behavior was a first-strike hard halt with no judge at all."""
    from run_pipeline import run_stage_2d
    state = make_state(run_dir)
    _seed_spec(state)
    _seed_2d_inputs(state)

    fake_subprocess.expect_script(returncode=1, stderr="finalize bug")
    decision = _build_judge_decision_halt(
        iteration=0, rationale="the spec's declared symbol itself is wrong")
    decision["validator_label"] = "finalize_package_init.py"
    queue_judge_decision_object(fake_dispatch, decision)

    result = run_stage_2d(state)
    assert_stage_halted(result, stage_id="stage_2d",
                        reason_contains="halt-judge decided to halt")
    assert_dispatched(fake_dispatch, "r2c-halt-judge", times=1)


def test_stage_2d_finalize_failure_judge_routes_fix_and_recovers(
    fake_dispatch, fake_subprocess, run_dir
):
    """The detr 2026-07-20 shape: the finalizer rejects a promised bridge
    symbol the method-coder delivered under another name → judge routes a
    fix to the method-coder → the finalizer passes on the retry and the
    stage completes. Under the old first-strike halt this exact case became
    an explanation-only delivery one alias away from a full one."""
    from run_pipeline import run_stage_2d
    state = make_state(run_dir)
    _seed_spec(state)
    _seed_2d_inputs(state)

    # iter 0: finalizer rejects the unkept symbol promise
    fake_subprocess.expect_script(
        returncode=2,
        stderr=("error: the spec's try_it_out.system_provides entry promises "
                "the importable symbol `train_from_scratch`, but no method/ "
                "module defines it (method_coder violated the spec surface)"))
    decision = _build_judge_decision_dispatch_fix(iteration=0)
    decision["validator_label"] = "finalize_package_init.py"
    decision["target_agent"] = "r2c-method-coder"
    decision["finding"] = {
        "id": "JUDGE001", "severity": "critical",
        "description": "rename train_with_distillation or add a public alias",
        "proposed_fix": "add `train_from_scratch = train_with_distillation`",
    }
    queue_judge_decision_object(fake_dispatch, decision)
    fake_dispatch.expect(agent="r2c-method-coder", writes={})

    # iter 1: finalizer passes; rest of the stage passes
    fake_subprocess.expect_script(returncode=0)  # finalize_package_init.py
    fake_subprocess.expect_script(returncode=0)  # pip install package requirements
    fake_subprocess.expect_stage2_script(ok=True)  # validate_package_imports.py
    fake_subprocess.expect_stage2_script(ok=True)  # validate_arch_contract.py
    fake_subprocess.expect_stage2_script(ok=True)  # validate_arch_contract_runtime.py

    result = run_stage_2d(state)
    assert_stage_completed(result, "stage_2d")
    assert_dispatched(fake_dispatch, "r2c-halt-judge", times=1)
    assert_dispatched(fake_dispatch, "r2c-method-coder", times=1)


def test_stage_2d_finalize_cap_exhausted_halts(
    fake_dispatch, fake_subprocess, run_dir
):
    """The finalizer keeps failing through every judge-routed fix → halt at
    the cap with retry_count recorded, never an infinite loop."""
    from run_pipeline import STAGE_2_RETRY_CAP, run_stage_2d
    state = make_state(run_dir)
    _seed_spec(state)
    _seed_2d_inputs(state)

    for i in range(STAGE_2_RETRY_CAP):
        fake_subprocess.expect_script(returncode=2, stderr="still violated")
        decision = _build_judge_decision_dispatch_fix(iteration=i)
        decision["validator_label"] = "finalize_package_init.py"
        decision["target_agent"] = "r2c-method-coder"
        queue_judge_decision_object(fake_dispatch, decision)
        fake_dispatch.expect(agent="r2c-method-coder", writes={})
    fake_subprocess.expect_script(returncode=2, stderr="still violated")

    result = run_stage_2d(state)
    assert_stage_halted(
        result, stage_id="stage_2d",
        reason_contains=f"after cap={STAGE_2_RETRY_CAP} judge-routed fixes")
    assert_dispatched(fake_dispatch, "r2c-halt-judge", times=STAGE_2_RETRY_CAP)


def test_stage_2d_dependency_install_failure_halts_with_researcher_message(
    fake_dispatch, fake_subprocess, run_dir
):
    """pip install of the package's own requirements fails → halt with an
    actionable researcher message (network hint + resume re-attempts), and
    the import validator never runs. Pins the 2026-07-07 fresh-clone dry-run
    find: the first `from method import *` happens at stage 2.d, three
    stages before the notebook's install cell, so a fresh environment
    without torch halted on the README's own example paper.

    The stderr is faithful to real offline pip output: the retry line
    with its connection error precedes the final no-distribution line.
    The wire marker is what keeps this classified transport_failure —
    a BARE no-distribution line now reads as a resolution failure (the
    pipeline shipped an uninstallable requirements.txt), pinned
    separately in test_halt_catalog.py."""
    from run_pipeline import run_stage_2d
    state = make_state(run_dir)
    _seed_spec(state)
    _seed_2d_inputs(state)

    fake_subprocess.expect_script(returncode=0)  # finalize_package_init.py
    fake_subprocess.expect_script(
        returncode=1,
        stderr=(
            "WARNING: Retrying (Retry(total=0)) after connection broken "
            "by 'NewConnectionError(...)': /simple/torch/\n"
            "ERROR: No matching distribution found for torch"))

    result = run_stage_2d(state)
    assert_stage_halted(result, stage_id="stage_2d",
                        reason_contains="package dependency install failed")
    halt = json.loads(
        (run_dir / ".pipeline" / "stage_2d.halt").read_text(encoding="utf-8"))
    assert "could not be installed" in (halt.get("user_message") or "")
    assert "resume" in (halt.get("user_message") or "").lower()
    assert len(fake_dispatch.calls) == 0


def test_stage_2d_validate_package_imports_failure_halts(
    fake_dispatch, fake_subprocess, run_dir
):
    """validate_package_imports.py failure → halt. No LLM dispatched —
    fix-loop only covers arch_contract validators (Phase 2 of the stage)."""
    from run_pipeline import run_stage_2d
    state = make_state(run_dir)
    _seed_spec(state)
    _seed_2d_inputs(state)

    fake_subprocess.expect_script(returncode=0)
    fake_subprocess.expect_script(returncode=0)  # pip install package requirements
    fake_subprocess.expect_stage2_script(ok=False, err="ImportError: foo")

    result = run_stage_2d(state)
    assert_stage_halted(result, stage_id="stage_2d",
                        reason_contains="validate_package_imports.py failed")
    assert len(fake_dispatch.calls) == 0


# ---------------------------------------------------------------------------
# Arch-contract fix loop with judge routing
# ---------------------------------------------------------------------------


def test_stage_2d_pipeline_owned_static_issue_degrades_without_dispatch(
    fake_dispatch, fake_subprocess, run_dir
):
    """Trusted static coverage gaps belong to the pipeline, not its producer."""
    from run_pipeline import run_stage_2d

    state = make_state(run_dir)
    _seed_phase_2(fake_subprocess, state)
    issue = SemanticIssue(
        code="unsupported_validator_feature",
        message="relational form is outside the static validator boundary",
        roots=[
            "methodology_replication_contract.elements[0].relational_structure",
            "scripts/validate_arch_contract.py",
        ],
        values={"unsupported_kind": "hypergraph"},
    )
    fake_subprocess.expect_stage2_script(
        ok=False, returncode=3, err=_semantic_stderr(issue)
    )

    result = run_stage_2d(state)

    assert_stage_degraded(
        result,
        stage_id="stage_2d",
        run_dir=run_dir,
        known_issue_contains="relational form is outside",
    )
    assert_not_dispatched(fake_dispatch, "r2c-halt-judge")
    assert_not_dispatched(fake_dispatch, "r2c-architecture-coder")
    events = [
        json.loads(line)
        for line in (state.paths.pipeline_dir / "run_events.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    routed = [
        event for event in events
        if event["event_type"] == "pipeline_validation_issue"
    ]
    assert len(routed) == 1
    assert routed[0]["details"]["issues"][0]["owner"] == "pipeline"


def test_stage_2d_pipeline_owned_runtime_issue_degrades_without_dispatch(
    fake_dispatch, fake_subprocess, run_dir
):
    """The runtime validator uses the same owner-preserving route."""
    from run_pipeline import run_stage_2d

    state = make_state(run_dir)
    _seed_phase_2(fake_subprocess, state)
    fake_subprocess.expect_stage2_script(ok=True)
    issue = SemanticIssue(
        code="unsupported_validator_feature",
        message="typed fixture exceeds the bounded allocation capability",
        roots=["architecture.model.forward.input.x"],
        values={"max_elements": 1000000},
    )
    fake_subprocess.expect_stage2_script(
        ok=False,
        returncode=3,
        err=_semantic_stderr(issue, label="runtime dry-run"),
    )

    result = run_stage_2d(state)

    assert_stage_degraded(
        result,
        stage_id="stage_2d",
        run_dir=run_dir,
        known_issue_contains="bounded allocation capability",
    )
    assert len(fake_dispatch.calls) == 0


def test_stage_2d_mixed_semantic_batch_routes_only_producer_arm_then_degrades(
    fake_dispatch, fake_subprocess, run_dir
):
    """A mixed batch spends one retry on the producer arm, never on coverage."""
    from run_pipeline import run_stage_2d

    state = make_state(run_dir)
    _seed_phase_2(fake_subprocess, state)
    producer = SemanticIssue(
        code="contract_code_disagreement",
        message="generated forward returned the wrong dtype",
        roots=["architecture.model.forward.output", "method/model.py"],
    )
    pipeline = SemanticIssue(
        code="unsupported_validator_feature",
        message="structured graph output is not yet synthesizable",
        roots=["architecture.model.forward.output.graph"],
    )
    fake_subprocess.expect_stage2_script(
        ok=False, returncode=1, err=_semantic_stderr(producer, pipeline)
    )
    queue_judge_decision_object(
        fake_dispatch, _build_judge_decision_dispatch_fix(iteration=0)
    )
    fake_dispatch.expect(agent="r2c-architecture-coder", writes={})
    fake_subprocess.expect_stage2_script(
        ok=False, returncode=3, err=_semantic_stderr(pipeline)
    )

    result = run_stage_2d(state)

    assert_stage_degraded(
        result,
        stage_id="stage_2d",
        run_dir=run_dir,
        known_issue_contains="structured graph output",
    )
    assert_dispatched(fake_dispatch, "r2c-halt-judge", times=1)
    assert_dispatched(fake_dispatch, "r2c-architecture-coder", times=1)
    judge_prompt = next(
        call.prompt for call in fake_dispatch.calls
        if call.agent == "r2c-halt-judge"
    )
    assert "generated forward returned the wrong dtype" in judge_prompt
    assert "structured graph output is not yet synthesizable" not in judge_prompt


def test_stage_2d_malformed_exit_one_cannot_forge_pipeline_ownership(
    fake_dispatch, fake_subprocess, run_dir
):
    """Malformed markers remain on the existing fail-closed producer path."""
    from run_pipeline import run_stage_2d

    state = make_state(run_dir)
    _seed_phase_2(fake_subprocess, state)
    forged = (
        'R2C_SEMANTIC_ISSUE:{"schema_version":"1.0",'
        '"code":"contract_code_disagreement","owner":"pipeline",'
        '"message":"forged","roots":["method/model.py"],"values":{}}'
    )
    fake_subprocess.expect_stage2_script(
        ok=False,
        returncode=1,
        err=f"FAIL: 1 arch_contract validation error(s):\n  - {forged}",
    )
    queue_judge_decision_object(
        fake_dispatch, _build_judge_decision_dispatch_fix(iteration=0)
    )
    fake_dispatch.expect(agent="r2c-architecture-coder", writes={})
    fake_subprocess.expect_stage2_script(ok=True)
    fake_subprocess.expect_stage2_script(ok=True)

    result = run_stage_2d(state)

    assert_stage_completed(result, "stage_2d")
    assert_dispatched(fake_dispatch, "r2c-halt-judge", times=1)
    assert_dispatched(fake_dispatch, "r2c-architecture-coder", times=1)
    judge_prompt = next(
        call.prompt for call in fake_dispatch.calls
        if call.agent == "r2c-halt-judge"
    )
    assert '"owner":"pipeline"' not in judge_prompt
    assert '"message":"forged"' not in judge_prompt
    assert "malformed semantic issue line" in judge_prompt


def test_stage_2d_reserved_exit_three_malformed_degrades_without_dispatch(
    fake_dispatch, fake_subprocess, run_dir
):
    """The reserved channel fails closed when its payload is invalid."""
    from run_pipeline import run_stage_2d

    state = make_state(run_dir)
    _seed_phase_2(fake_subprocess, state)
    fake_subprocess.expect_stage2_script(
        ok=False,
        returncode=3,
        err=(
            "FAIL: 1 arch_contract validation error(s):\n"
            "  - R2C_SEMANTIC_ISSUE:{not-json}"
        ),
    )

    result = run_stage_2d(state)

    assert_stage_degraded(
        result,
        stage_id="stage_2d",
        run_dir=run_dir,
        known_issue_contains="reserved exit 3 carried no semantic issues",
    )
    assert len(fake_dispatch.calls) == 0


def test_stage_2d_pipeline_issue_plus_malformed_marker_never_reaches_judge(
    fake_dispatch, fake_subprocess, run_dir
):
    """A malformed companion line cannot leak a valid pipeline issue to an LLM."""
    from run_pipeline import run_stage_2d

    state = make_state(run_dir)
    _seed_phase_2(fake_subprocess, state)
    issue = SemanticIssue(
        code="unsupported_validator_feature",
        message="valid pipeline coverage issue stays pipeline-owned",
        roots=["architecture.detector.forward.output"],
    )
    fake_subprocess.expect_stage2_script(
        ok=False,
        returncode=1,
        err=(
            "FAIL: 2 arch_contract validation error(s):\n"
            f"  - {format_semantic_issue(issue)}\n"
            "  - R2C_SEMANTIC_ISSUE:{not-json}"
        ),
    )

    result = run_stage_2d(state)

    assert_stage_degraded(
        result,
        stage_id="stage_2d",
        run_dir=run_dir,
        known_issue_contains="valid pipeline coverage issue",
    )
    assert len(fake_dispatch.calls) == 0


def test_stage_2d_setup_exit_two_degrades_without_producer_retry(
    fake_dispatch, fake_subprocess, run_dir
):
    """A validator setup failure is pipeline-owned even without a wire issue."""
    from run_pipeline import run_stage_2d

    state = make_state(run_dir)
    _seed_phase_2(fake_subprocess, state)
    fake_subprocess.expect_stage2_script(
        ok=False,
        returncode=2,
        err="error: spec-derived build plan is unavailable",
    )

    result = run_stage_2d(state)

    assert_stage_degraded(
        result,
        stage_id="stage_2d",
        run_dir=run_dir,
        known_issue_contains="spec-derived build plan is unavailable",
    )
    assert len(fake_dispatch.calls) == 0


def test_stage_2d_parses_pipeline_issue_before_excerpting_full_stderr(
    fake_dispatch, fake_subprocess, run_dir
):
    """Ownership survives when the issue line lies outside the display tail."""
    from run_pipeline import run_stage_2d

    state = make_state(run_dir)
    _seed_phase_2(fake_subprocess, state)
    issue = SemanticIssue(
        code="unsupported_validator_feature",
        message="early coverage marker remains authoritative",
        roots=["optimizer_state.slot"],
    )
    err = _semantic_stderr(issue) + "\n" + ("late diagnostic filler\n" * 300)
    fake_subprocess.expect_stage2_script(
        ok=False, returncode=3, err=err
    )

    result = run_stage_2d(state)

    assert_stage_degraded(
        result,
        stage_id="stage_2d",
        run_dir=run_dir,
        known_issue_contains="early coverage marker remains authoritative",
    )
    assert len(fake_dispatch.calls) == 0


def test_stage_2d_exit_zero_with_semantic_issue_degrades_as_protocol_failure(
    fake_dispatch, fake_subprocess, run_dir
):
    """A validator cannot launder an issue by returning success."""
    from run_pipeline import run_stage_2d

    state = make_state(run_dir)
    _seed_phase_2(fake_subprocess, state)
    issue = SemanticIssue(
        code="contract_code_disagreement",
        message="success exit contradicted a reported mismatch",
        roots=["architecture.model.forward.output"],
    )
    fake_subprocess.expect_stage2_script(
        ok=True, returncode=0, err=_semantic_stderr(issue)
    )

    result = run_stage_2d(state)

    assert_stage_degraded(
        result,
        stage_id="stage_2d",
        run_dir=run_dir,
        known_issue_contains="exit 0 carried semantic failure issues",
    )
    assert len(fake_dispatch.calls) == 0


def test_stage_2d_arch_contract_fail_judge_dispatch_fix_recovers(
    fake_dispatch, fake_subprocess, run_dir
):
    """validate_arch_contract.py fails iter 0 → judge invoked → judge
    decides dispatch_fix → architecture-coder dispatched → iter 1 validators
    pass → stage completes."""
    from run_pipeline import run_stage_2d
    state = make_state(run_dir)
    _seed_spec(state)
    _seed_2d_inputs(state)

    # iter 0: finalize + imports pass; structural validator fails
    fake_subprocess.expect_script(returncode=0)
    fake_subprocess.expect_script(returncode=0)  # pip install package requirements
    fake_subprocess.expect_stage2_script(ok=True)  # validate_package_imports
    fake_subprocess.expect_stage2_script(ok=False, err="schema reject")  # arch_contract structural

    # judge dispatched, decides dispatch_fix
    decision = _build_judge_decision_dispatch_fix(iteration=0)
    queue_judge_decision_object(fake_dispatch, decision)
    # architecture-coder fix-mode dispatched
    fake_dispatch.expect(agent="r2c-architecture-coder", writes={})

    # iter 1: both arch_contract validators pass
    fake_subprocess.expect_stage2_script(ok=True)  # arch_contract structural
    fake_subprocess.expect_stage2_script(ok=True)  # arch_contract runtime dry-run

    result = run_stage_2d(state)
    assert_stage_completed(result, "stage_2d")
    assert_dispatched(fake_dispatch, "r2c-halt-judge", times=1)
    assert_dispatched(fake_dispatch, "r2c-architecture-coder", times=1)


def test_stage_2d_arch_contract_runtime_fail_routes_through_judge(
    fake_dispatch, fake_subprocess, run_dir
):
    """validate_arch_contract_runtime.py (the dry-run, T2) failing also
    routes through the judge — same code path as T1."""
    from run_pipeline import run_stage_2d
    state = make_state(run_dir)
    _seed_spec(state)
    _seed_2d_inputs(state)

    # iter 0: finalize + imports + T1 pass; T2 dry-run fails
    fake_subprocess.expect_script(returncode=0)
    fake_subprocess.expect_script(returncode=0)  # pip install package requirements
    fake_subprocess.expect_stage2_script(ok=True)  # validate_package_imports
    fake_subprocess.expect_stage2_script(ok=True)  # arch_contract structural
    fake_subprocess.expect_stage2_script(ok=False, err="RuntimeError: BN bug")  # dry-run

    decision = _build_judge_decision_dispatch_fix(iteration=0)
    decision["validator_label"] = "validate_arch_contract_runtime.py"
    queue_judge_decision_object(fake_dispatch, decision)
    fake_dispatch.expect(agent="r2c-architecture-coder", writes={})

    # iter 1: both pass
    fake_subprocess.expect_stage2_script(ok=True)
    fake_subprocess.expect_stage2_script(ok=True)

    result = run_stage_2d(state)
    assert_stage_completed(result, "stage_2d")


def test_stage_2d_arch_contract_fail_judge_halt_decision_degrades_stage(
    fake_dispatch, fake_subprocess, run_dir
):
    """When the judge classifies as pipeline_bug / upstream_issue and decides
    halt, stage 2.d DEGRADES (not halts): the package imports and notebook gen
    doesn't read arch_contract, so the run continues with the contract gap
    logged to KNOWN_ISSUES.md (halt→degrade redesign, 2026-05-27)."""
    from run_pipeline import run_stage_2d
    state = make_state(run_dir)
    _seed_spec(state)
    _seed_2d_inputs(state)

    fake_subprocess.expect_script(returncode=0)
    fake_subprocess.expect_script(returncode=0)  # pip install package requirements
    fake_subprocess.expect_stage2_script(ok=True)
    fake_subprocess.expect_stage2_script(ok=False, err="schema reject")

    decision = _build_judge_decision_halt(
        iteration=0, rationale="field guide is broken"
    )
    queue_judge_decision_object(fake_dispatch, decision)

    result = run_stage_2d(state)
    assert_stage_degraded(result, stage_id="stage_2d",
                          run_dir=run_dir,
                          known_issue_contains="field guide is broken")
    # The judge's rationale is surfaced researcher-facing in KNOWN_ISSUES.md.
    assert_stage_degraded(result, run_dir=run_dir,
                          known_issue_contains="arch_contract")
    # Judge ran exactly once; architecture-coder NOT dispatched (judge said halt)
    assert_dispatched(fake_dispatch, "r2c-halt-judge", times=1)
    assert_not_dispatched(fake_dispatch, "r2c-architecture-coder")


# ---------------------------------------------------------------------------
# Cap exhaustion
# ---------------------------------------------------------------------------


def test_stage_2d_cap_exhausted_degrades_with_cumulative_stderr(
    fake_dispatch, fake_subprocess, run_dir
):
    """When STAGE_2_RETRY_CAP iterations all fail the arch_contract validators,
    the stage DEGRADES (halt→degrade redesign) — logs the cap-exhausted reason +
    last stderr to KNOWN_ISSUES.md and continues — rather than halting."""
    from run_pipeline import STAGE_2_RETRY_CAP, run_stage_2d
    state = make_state(run_dir)
    _seed_spec(state)
    _seed_2d_inputs(state)

    fake_subprocess.expect_script(returncode=0)
    fake_subprocess.expect_script(returncode=0)  # pip install package requirements
    fake_subprocess.expect_stage2_script(ok=True)  # validate_package_imports

    # STAGE_2_RETRY_CAP + 1 iterations of validators; each iteration:
    #   - validate_arch_contract.py fails (so dry-run isn't reached)
    # Each iteration (except the last) is followed by judge + arch-coder dispatch.
    for i in range(STAGE_2_RETRY_CAP + 1):
        # arch_contract structural fails each iter
        fake_subprocess.expect_stage2_script(ok=False, err=f"iter {i} schema reject")
        if i < STAGE_2_RETRY_CAP:
            # judge invoked + arch-coder dispatched
            decision = _build_judge_decision_dispatch_fix(iteration=i)
            queue_judge_decision_object(fake_dispatch, decision)
            fake_dispatch.expect(agent="r2c-architecture-coder", writes={})

    result = run_stage_2d(state)
    assert_stage_degraded(result, stage_id="stage_2d", run_dir=run_dir,
                          known_issue_contains=f"cap={STAGE_2_RETRY_CAP}")
    # Exactly STAGE_2_RETRY_CAP judge invocations + arch-coder dispatches
    assert_dispatched(fake_dispatch, "r2c-halt-judge", times=STAGE_2_RETRY_CAP)
    assert_dispatched(fake_dispatch, "r2c-architecture-coder",
                      times=STAGE_2_RETRY_CAP)


# ---------------------------------------------------------------------------
# Smart skip: content-digest check catches stale __init__.py when method/*.py
# changed (R2C 2026-05-22 bev-distill cascade regression)
# ---------------------------------------------------------------------------


def _seed_2d_upstream(run_dir: Path) -> None:
    """Seed the method/*.py files + arch_contract + method_spec that stage 2.d
    digests."""
    (run_dir / "method").mkdir(parents=True, exist_ok=True)
    (run_dir / "method" / "model.py").write_text("# model\n")
    (run_dir / "method" / "training.py").write_text("# training\n")
    (run_dir / "method" / "method.py").write_text("# method\n")
    (run_dir / "method" / "data.py").write_text("# data\n")
    (run_dir / ".pipeline").mkdir(parents=True, exist_ok=True)
    (run_dir / ".pipeline" / "arch_contract.json").write_text("{}\n")
    (run_dir / ".pipeline" / "method_spec.json").write_text("{}\n")
    # Also seed the outputs of stage 2.d itself (so the smart-skip's
    # output-presence check passes)
    (run_dir / "method" / "__init__.py").write_text("# init (stale or fresh)\n")
    (run_dir / "requirements.txt").write_text(
        "r2c-stage2d-test-dependency-never-installed\n"
    )


def _write_2d_sentinel_and_digest(run_dir: Path) -> None:
    """Write `stage_2d.complete` + the content digest. Mirrors what the
    driver's main loop does on stage 2.d completion."""
    from run_pipeline import write_stage_2d_upstream_digest
    state = make_state(run_dir)
    (run_dir / ".pipeline" / "stage_2d.complete").write_text("")
    write_stage_2d_upstream_digest(state.paths)


def test_stage_2d_skips_when_sentinel_and_content_digest_matches(
    fake_dispatch, fake_subprocess, run_dir
):
    """**Counterpoint to the bev-distill 2026-05-22 cascade**: when nothing
    upstream has changed (sentinel + digest both match current state), stage
    2.d skips without re-running the finalize script."""
    from run_pipeline import run_stage_2d
    state = make_state(run_dir)
    _seed_spec(state)
    _seed_2d_upstream(run_dir)
    _write_2d_sentinel_and_digest(run_dir)

    result = run_stage_2d(state)

    assert result.status == "skipped", f"expected skipped, got {result.status}"
    assert "content digest matches" in (result.notes or "")
    # No subprocess calls, no agent dispatches — finalize never ran
    assert len(fake_subprocess.script_calls) == 0
    assert len(fake_dispatch.calls) == 0


def test_stage_2d_re_runs_when_method_py_content_changed(
    fake_dispatch, fake_subprocess, run_dir
):
    """**R2C 2026-05-22 bev-distill cascade regression**: stage 2.c regenerated
    method.py with a public-API rename. Stage 2.d's standard sentinel-check
    would have skipped (sentinel + outputs both present) — leaving stale
    __init__.py. The content-digest check correctly detects the upstream
    change and re-runs the finalize script."""
    from run_pipeline import run_stage_2d
    state = make_state(run_dir)
    _seed_spec(state)
    _seed_2d_upstream(run_dir)
    _write_2d_sentinel_and_digest(run_dir)

    # Stage 2.c equivalent: change method.py content (rename, add, remove —
    # any public-API alteration). The mtime is also newer but the digest
    # check doesn't care about mtime — it cares about content.
    (run_dir / "method" / "method.py").write_text("# method (renamed API)\n")

    # Stage 2.d re-runs. Set up finalize_package_init + validators all pass.
    fake_subprocess.expect_script(returncode=0)  # finalize_package_init
    fake_subprocess.expect_script(returncode=0)  # pip install package requirements
    fake_subprocess.expect_stage2_script(ok=True)  # validate_package_imports
    fake_subprocess.expect_stage2_script(ok=True)  # validate_arch_contract
    fake_subprocess.expect_stage2_script(ok=True)  # validate_arch_contract_runtime

    result = run_stage_2d(state)
    assert_stage_completed(result, "stage_2d")
    # Stage ran (4 subprocess calls), not skipped
    assert len(fake_subprocess.script_calls) == 2  # finalize + pip install
    assert len(fake_subprocess.stage2_calls) == 3


def test_stage_2d_re_runs_when_sentinel_present_but_digest_missing(
    fake_dispatch, fake_subprocess, run_dir
):
    """**R2C 2026-05-22 user-facing fix path**: sentinel was written by a prior
    driver version without digest support. We can't verify the sentinel was
    written when current upstream existed (specifically, the bev-distill case
    where 2.c regen'd method.py BETWEEN the sentinel write and the new run).
    Safe default: treat as cache-miss and re-run. The completion path writes
    the digest; subsequent runs use it normally."""
    from run_pipeline import _stage_2d_upstream_digest_path, run_stage_2d
    state = make_state(run_dir)
    _seed_spec(state)
    _seed_2d_upstream(run_dir)
    # Sentinel exists but digest file does NOT — bootstrap case
    (run_dir / ".pipeline" / "stage_2d.complete").write_text("")

    digest_path = _stage_2d_upstream_digest_path(state.paths)
    assert not digest_path.exists()

    # Stage 2.d re-runs the finalize script + validators
    fake_subprocess.expect_script(returncode=0)  # finalize_package_init
    fake_subprocess.expect_script(returncode=0)  # pip install package requirements
    fake_subprocess.expect_stage2_script(ok=True)  # validate_package_imports
    fake_subprocess.expect_stage2_script(ok=True)  # validate_arch_contract
    fake_subprocess.expect_stage2_script(ok=True)  # validate_arch_contract_runtime

    result = run_stage_2d(state)
    assert_stage_completed(result, "stage_2d")
    # Stage RAN, not skipped — finalize + validators all fired
    assert len(fake_subprocess.script_calls) == 2  # finalize + pip install
    assert len(fake_subprocess.stage2_calls) == 3


def test_stage_2d_re_runs_when_arch_contract_content_changed(
    fake_dispatch, fake_subprocess, run_dir
):
    """arch_contract.json content changing also invalidates stage 2.d's
    cached result (the runtime dry-run consumes it directly)."""
    from run_pipeline import run_stage_2d
    state = make_state(run_dir)
    _seed_spec(state)
    _seed_2d_upstream(run_dir)
    _write_2d_sentinel_and_digest(run_dir)

    # Stage 2.b equivalent: change arch_contract.json content
    (run_dir / ".pipeline" / "arch_contract.json").write_text(
        '{"updated": true}\n'
    )

    fake_subprocess.expect_script(returncode=0)
    fake_subprocess.expect_script(returncode=0)  # pip install package requirements
    fake_subprocess.expect_stage2_script(ok=True)
    fake_subprocess.expect_stage2_script(ok=True)
    fake_subprocess.expect_stage2_script(ok=True)

    result = run_stage_2d(state)
    assert_stage_completed(result, "stage_2d")
    assert len(fake_subprocess.script_calls) == 2  # finalize + pip install


def test_stage_2d_skips_when_mtime_changed_but_content_identical(
    fake_dispatch, fake_subprocess, run_dir
):
    """The content-digest approach must NOT spuriously miss on idempotent
    re-writes (where mtime changes but content stays the same). Distinct
    from the mtime-based approach we initially shipped for stage 3.c, which
    had this same false-positive bug."""
    import os
    from run_pipeline import run_stage_2d
    state = make_state(run_dir)
    _seed_spec(state)
    _seed_2d_upstream(run_dir)
    _write_2d_sentinel_and_digest(run_dir)

    # Idempotent rewrite: same content, newer mtime
    later_mtime = (
        run_dir / ".pipeline" / "stage_2d.complete"
    ).stat().st_mtime + 100
    method_py = run_dir / "method" / "method.py"
    method_py.write_text(method_py.read_text())  # same content
    os.utime(method_py, (later_mtime, later_mtime))

    result = run_stage_2d(state)
    assert result.status == "skipped", f"expected skipped, got {result.status}"
    assert "content digest matches" in (result.notes or "")
    assert len(fake_subprocess.script_calls) == 0


# ---------------------------------------------------------------------------
# R2C-050 — producer-selected uninstallable dependency routing
# ---------------------------------------------------------------------------

_DGL_SHAPE_STDERR = (
    "ERROR: Could not find a version that satisfies the requirement "
    "fakegraphlib (from versions: none)\n"
    "ERROR: No matching distribution found for fakegraphlib\n"
)


def _seed_producer_import(state, requirement: str = "fakegraphlib"):
    """model.py imports the doomed library (the pdfgnn dgl shape: the
    architecture coder's choice), and requirements.txt carries it."""
    rd = state.paths.run_dir
    (rd / "method").mkdir(parents=True, exist_ok=True)
    (rd / "method" / "model.py").write_text(
        f"import {requirement}\n\nclass Net:\n    pass\n"
    )
    (rd / "requirements.txt").write_text(f"{requirement}\n")


def test_stage_2d_uninstallable_dependency_routes_to_producer_and_recovers(
    fake_dispatch, fake_subprocess, run_dir
):
    """The pdfgnn 2026-08-03 shape, converted: pip's resolver rejects a
    requirement that traces to the architecture coder's own import → the
    coder gets a fix finding (no judge needed, the trace is deterministic),
    the finalizer regenerates requirements.txt from the fixed imports, and
    the retry installs clean."""
    from run_pipeline import run_stage_2d
    state = make_state(run_dir)
    _seed_spec(state, paradigm_id="active_learning")
    _seed_2d_inputs(state)
    _seed_producer_import(state)
    fake_subprocess.set_run_dir(run_dir)

    fake_subprocess.expect_script(returncode=0)  # finalize (initial)
    fake_subprocess.expect_script(               # pip install — resolver says no
        returncode=1, stderr=_DGL_SHAPE_STDERR)
    fake_dispatch.expect(                        # the routed producer fix
        agent="r2c-architecture-coder",
        writes={"method/model.py": "class Net:\n    pass\n"},
    )
    fake_subprocess.expect_script(               # re-finalize regenerates reqs
        returncode=0,
        writes={"requirements.txt": "numpy>=1.24\n"},
    )
    fake_subprocess.expect_script(returncode=0)  # pip install — clean now
    fake_subprocess.expect_stage2_script(ok=True)  # validate_package_imports
    fake_subprocess.expect_stage2_script(ok=True)  # validate_arch_contract
    fake_subprocess.expect_stage2_script(ok=True)  # validate_arch_contract_runtime

    result = run_stage_2d(state)
    assert_stage_completed(result, "stage_2d")
    assert_dispatched(fake_dispatch, "r2c-architecture-coder", times=1)
    assert_not_dispatched(fake_dispatch, "r2c-halt-judge")


def test_stage_2d_uninstallable_dependency_exhaustion_halts_as_producer_owned(
    fake_dispatch, fake_subprocess, run_dir
):
    """The fix loop is bounded. When the producer keeps re-importing the
    doomed library, the terminal halt names the PRODUCER's choice
    (producer_output_invalid), never the old report-to-engineering story."""
    from run_pipeline import STAGE_2_RETRY_CAP, run_stage_2d
    state = make_state(run_dir)
    _seed_spec(state, paradigm_id="active_learning")
    _seed_2d_inputs(state)
    _seed_producer_import(state)

    fake_subprocess.expect_script(returncode=0)  # finalize (initial)
    fake_subprocess.expect_script(returncode=1, stderr=_DGL_SHAPE_STDERR)
    for _ in range(STAGE_2_RETRY_CAP):
        fake_dispatch.expect(agent="r2c-architecture-coder", writes={})
        fake_subprocess.expect_script(returncode=0)  # re-finalize
        fake_subprocess.expect_script(                # pip fails again
            returncode=1, stderr=_DGL_SHAPE_STDERR)

    result = run_stage_2d(state)
    assert_stage_halted(result, stage_id="stage_2d",
                        reason_contains="producer-selected")
    halt_record = json.loads(
        (run_dir / ".pipeline" / "stage_2d.halt").read_text())
    assert halt_record["halt_class"] == "producer_output_invalid"
    traced = halt_record["context"]["traced_requirements"]
    assert traced[0]["requirement"] == "fakegraphlib"
    assert traced[0]["files"] == ["method/model.py"]
    assert traced[0]["producers"] == ["architecture_coder"]
    assert_dispatched(fake_dispatch, "r2c-architecture-coder",
                      times=STAGE_2_RETRY_CAP)


def test_stage_2d_untraceable_resolution_failure_keeps_internal_halt(
    fake_dispatch, fake_subprocess, run_dir
):
    """The DomIndOnto 2026-07-21 shape stays correctly classified: a
    resolution failure on a name NO producer file imports (a finalizer
    artifact) is pipeline-owned, so no producer is dispatched and the
    internal-bug halt renders unchanged."""
    from run_pipeline import run_stage_2d
    state = make_state(run_dir)
    _seed_spec(state)
    _seed_2d_inputs(state)
    # model.py exists but does NOT import the failing name.
    (run_dir / "method" / "model.py").write_text("import torch\n")

    fake_subprocess.expect_script(returncode=0)  # finalize
    fake_subprocess.expect_script(
        returncode=1,
        stderr=("ERROR: Could not find a version that satisfies the "
                "requirement method (from versions: none)\n"))

    result = run_stage_2d(state)
    assert_stage_halted(result, stage_id="stage_2d",
                        reason_contains="dependency resolution failed")
    halt_record = json.loads(
        (run_dir / ".pipeline" / "stage_2d.halt").read_text())
    assert halt_record["halt_class"] == "internal_contract_violation"
    assert len(fake_dispatch.calls) == 0


def test_stage_2d_transport_failure_is_untouched_by_the_routing(
    fake_dispatch, fake_subprocess, run_dir
):
    """A wire failure keeps the reconnect story: no trace, no dispatch,
    transport_failure with the user-facing message."""
    from run_pipeline import run_stage_2d
    state = make_state(run_dir)
    _seed_spec(state, paradigm_id="active_learning")
    _seed_2d_inputs(state)
    _seed_producer_import(state)  # even WITH a traceable import present

    fake_subprocess.expect_script(returncode=0)  # finalize
    fake_subprocess.expect_script(
        returncode=1,
        stderr=("WARNING: connection broken by ProxyError\n"
                "ERROR: No matching distribution found for fakegraphlib\n"))

    result = run_stage_2d(state)
    assert_stage_halted(result, stage_id="stage_2d",
                        reason_contains="install failed")
    halt_record = json.loads(
        (run_dir / ".pipeline" / "stage_2d.halt").read_text())
    assert halt_record["halt_class"] == "transport_failure"
    assert len(fake_dispatch.calls) == 0
