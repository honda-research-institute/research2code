"""Phase 5 — Cross-stage interaction tests.

Captures failure scenarios that span multiple stages, focused on the
bev-distill 2026-05-19 cascade as the canonical regression test:

  1. Stage 2.b's architecture-coder writes model.py with a latent bug
     (e.g., BatchNorm placement that breaks on real-shape input).
  2. Stage 2.b's validator + stage-reviewer pass (the bug is paradigm-
     specific and runtime-dependent — not caught by static checks).
  3. Stage 2.d's runtime dry-run catches the bug → halts.
  4. WITHOUT sentinels (legacy): user clears halt + resumes → stage 2.d's
     outputs are partially present → skip-check skips validators →
     BN bug survives to stage 3.c → smoke loop exhausts cap.
  5. WITH sentinels (current behavior): stage_2d.complete sentinel was
     never written (stage halted), so on re-invocation stage 2.d re-runs
     validators → dry-run again catches the bug → judge → architecture-
     coder fix-mode → bug fixed → stage 2.d completes + sentinel written.

These tests don't run the full pipeline end-to-end; they simulate the
sentinel state across stage invocations and verify the recovery
invariants. Adding a full-pipeline integration test would be valuable
later but is too brittle to mock at this stage.
"""

from __future__ import annotations

import json
from pathlib import Path

from tests.helpers.assertions import (
    assert_sentinel_absent,
    assert_sentinel_present,
    assert_stage_completed,
    assert_stage_halted,
    assert_stage_skipped,
)
from tests.helpers.inject import (
    minimal_method_spec,
    queue_judge_decision_object,
    write_halt,
    write_stage_complete_sentinel,
)
from tests.helpers.state import make_state


# ---------------------------------------------------------------------------
# The bev-distill 2026-05-19 cascade regression
# ---------------------------------------------------------------------------


def test_bev_distill_cascade_prevented_by_sentinel_plus_judge(
    fake_dispatch, fake_subprocess, run_dir
):
    """End-to-end regression for the bev-distill 2026-05-19 cascade.

    Scenario:
      - Prior run halted at stage 2.d on a different issue (arch_contract
        schema). User cleared the halt + committed the schema fix.
      - On resume: stage 2.d outputs (__init__.py, requirements.txt) are
        still on disk, but stage_2d.complete sentinel is missing.
      - Stage 2.b had also produced a model.py with a latent BN bug that
        stage 2.d's dry-run can catch.

    Expected behavior (the fix): stage 2.d re-runs because no sentinel,
    validators run, dry-run catches the BN bug, judge dispatches arch-
    coder fix-mode, the fix lands, stage 2.d completes, sentinel written.
    The cascade into stage 3.c is prevented.
    """
    from run_pipeline import (
        _write_stage_complete_sentinel,
        run_stage_2d,
    )
    state = make_state(run_dir)

    # Seed the spec so build_paths_block + judge can resolve the paradigm.
    spec = minimal_method_spec()
    state.paths.method_spec.write_text(json.dumps(spec, indent=2), encoding="utf-8")

    # Pre-state: stage 2.b/2.c outputs + sentinels are present (they completed
    # cleanly in the prior run). Stage 2.d outputs are partially present
    # (init_finalizer ran before the validator halted). Sentinel for 2.d is
    # MISSING — this is the cascade trigger.
    rd = run_dir
    (rd / "method").mkdir(parents=True, exist_ok=True)
    (rd / "method" / "model.py").write_text("# model with BN bug\n")
    (rd / "method" / "training.py").write_text("# training\n")
    (rd / "method" / "method.py").write_text("# method\n")
    (rd / "method" / "__init__.py").write_text("# fake __init__\n")
    (rd / "requirements.txt").write_text(
        "r2c-cross-stage-test-dependency-never-installed\n"
    )
    (rd / ".pipeline" / "arch_contract.json").write_text("{}\n")
    _write_stage_complete_sentinel(state.paths, "stage_2b")
    _write_stage_complete_sentinel(state.paths, "stage_2c")
    # NOTE: no stage_2d.complete sentinel

    # Stage 2.d kicks off: finalize_package_init.py + package_imports OK
    fake_subprocess.expect_script(returncode=0)  # finalize_package_init
    fake_subprocess.expect_script(returncode=0)  # pip install package requirements
    fake_subprocess.expect_stage2_script(ok=True)  # validate_package_imports
    # arch_contract structural validator passes (Phase 1 of T2)
    fake_subprocess.expect_stage2_script(ok=True)
    # Runtime dry-run catches the BN bug → halts on iter 0
    fake_subprocess.expect_stage2_script(
        ok=False, err="RuntimeError: running_mean should contain 16 elements not 64",
    )
    # Judge invoked → producer_fixable → arch-coder fix-mode
    queue_judge_decision_object(fake_dispatch, {
        "schema_version": "1.0.0",
        "stage_id": "stage_2d",
        "iteration": 0,
        "validator_label": "validate_arch_contract_runtime.py",
        "classification": "producer_fixable",
        "action": "dispatch_fix",
        "target_agent": "r2c-architecture-coder",
        "finding": {
            "id": "JUDGE001", "severity": "critical",
            "description": "teacher's BN placement is wrong for 3D input",
            "proposed_fix": "wrap point-MLP to flatten before BN",
        },
        "rationale": "BN bug in model.py teacher's point_mlp",
        "confidence": "high",
        "files_examined": ["scripts/run_pipeline.py", "method/model.py"],
    })
    fake_dispatch.expect(
        agent="r2c-architecture-coder",
        writes={"method/model.py": "# fixed BN placement\n"},
    )
    # iter 1: both arch_contract validators pass
    fake_subprocess.expect_stage2_script(ok=True)
    fake_subprocess.expect_stage2_script(ok=True)

    result = run_stage_2d(state)
    assert_stage_completed(result, "stage_2d")

    # The crucial post-condition: the BN bug was caught and fixed in stage 2.d
    # rather than propagating to stage 3.c.


# ---------------------------------------------------------------------------
# Sentinel + halt-artifact interaction across stage invocations
# ---------------------------------------------------------------------------


def test_completed_stage_then_skipped_on_re_invocation(
    fake_dispatch, fake_subprocess, run_dir
):
    """A stage that completes writes its sentinel; a subsequent re-invocation
    sees the sentinel + outputs + no halt and skips. This is the main
    loop's sentinel-write hook combined with `_skip_if_done`'s check."""
    from run_pipeline import (
        _write_stage_complete_sentinel,
        run_stage_2d,
    )
    state = make_state(run_dir)
    spec = minimal_method_spec()
    state.paths.method_spec.write_text(json.dumps(spec, indent=2), encoding="utf-8")

    # First invocation: stage 2.d runs to completion. Stage 2.d's skip-check
    # looks at method/__init__.py + requirements.txt — pre-create them so
    # the second invocation's skip-check sees outputs + sentinel.
    rd = run_dir
    (rd / "method").mkdir(parents=True, exist_ok=True)
    (rd / "method" / "model.py").write_text("# model\n")
    (rd / "method" / "__init__.py").write_text("# init\n")
    (rd / "requirements.txt").write_text(
        "r2c-cross-stage-test-dependency-never-installed\n"
    )
    fake_subprocess.expect_script(returncode=0)  # finalize_package_init
    fake_subprocess.expect_script(returncode=0)  # pip install package requirements
    fake_subprocess.expect_stage2_script(ok=True)  # validate_package_imports
    fake_subprocess.expect_stage2_script(ok=True)  # arch_contract structural
    fake_subprocess.expect_stage2_script(ok=True)  # arch_contract dry-run
    result1 = run_stage_2d(state)
    assert_stage_completed(result1, "stage_2d")

    # Simulate the main loop's completion writes: sentinel + upstream digest
    # (the test harness doesn't invoke main(); it calls stage functions
    # directly). Post-R2C-2026-05-22 the smart-skip also requires the digest.
    from run_pipeline import write_stage_2d_upstream_digest
    _write_stage_complete_sentinel(state.paths, "stage_2d")
    write_stage_2d_upstream_digest(state.paths)
    assert_sentinel_present(rd, "stage_2d")

    # Second invocation: same state, sentinel + outputs + digest present,
    # no halt → skip.
    result2 = run_stage_2d(state)
    assert_stage_skipped(result2, "stage_2d")
    # No additional subprocess or dispatch calls happened (skip short-circuits)


def test_halted_stage_re_runs_on_re_invocation(
    fake_dispatch, fake_subprocess, run_dir
):
    """A stage that halted has no sentinel; a subsequent re-invocation
    re-runs validators (the sentinel-system invariant). Halt artifact is
    cleared by `_skip_if_done` before the stage runs."""
    from run_pipeline import (
        _clear_stage_complete_sentinel,
        run_stage_2d,
    )
    state = make_state(run_dir)
    spec = minimal_method_spec()
    state.paths.method_spec.write_text(json.dumps(spec, indent=2), encoding="utf-8")

    rd = run_dir
    (rd / "method").mkdir(parents=True, exist_ok=True)
    (rd / "method" / "model.py").write_text("# model\n")
    # Simulate a halted prior run: outputs partially present, halt artifact
    # present, sentinel ABSENT.
    write_halt(rd, "stage_2d", reason="prior validate_arch_contract.py failed")
    _clear_stage_complete_sentinel(state.paths, "stage_2d")  # idempotent
    assert_sentinel_absent(rd, "stage_2d")

    # Re-invoke stage 2.d: validators run (the halt is cleared by skip-check,
    # not by us). The finalizer's requirements.txt output is mandatory.
    fake_subprocess.set_run_dir(run_dir)
    fake_subprocess.expect_script(
        returncode=0,
        writes={
            "requirements.txt": "r2c-cross-stage-test-dependency-never-installed\n"
        },
    )  # finalize_package_init
    fake_subprocess.expect_script(returncode=0)  # pip install package requirements
    fake_subprocess.expect_stage2_script(ok=True)
    fake_subprocess.expect_stage2_script(ok=True)
    fake_subprocess.expect_stage2_script(ok=True)

    result = run_stage_2d(state)
    assert_stage_completed(result, "stage_2d")
    # Halt artifact was cleared by skip-check on stage entry.
    halt_path = rd / ".pipeline" / "stage_2d.halt"
    assert not halt_path.exists()


def test_b004_writeable_block_forbids_producers_writing_reviews():
    """B-004 (pdwa 2026-05-27): the shared writeable-paths block tells every
    producer dispatch that the field guide's `stage_review_focus` is the
    stage-reviewer's contract, not theirs — and self-scopes so reviewers (who
    own `.pipeline/stage_review_*.json`) are still permitted by the same clause.

    Root cause this guards: the method-coder reads the field guide for its own
    contract (pluggable_component + package_manifest) but the same file carries
    the reviewer-only `stage_review_focus` block; the pdwa method-coder inlined
    it and wrote `stage_review_stage_2c_method.json` out-of-scope, halting 2.c.
    """
    from dispatch_templates import WRITEABLE_PATHS, format_writeable_paths_block

    producer = format_writeable_paths_block(WRITEABLE_PATHS["r2c-method-coder"])
    assert "stage_review_focus" in producer, "clause must name the leak surface"
    assert "stage_review_*.json" in producer, "clause must name the artifact producers must not write"
    # The method-coder's allowlist is method/method.py only — review file NOT listed.
    assert "method/method.py" in producer
    assert ".pipeline/stage_review_*.json" not in producer.split("Producing your listed")[0]

    # Same clause, opposite effect: the reviewer owns the review-file glob, so
    # the "unless it matches a pattern above" carve-out permits it.
    reviewer = format_writeable_paths_block(WRITEABLE_PATHS["r2c-stage-reviewer"])
    assert ".pipeline/stage_review_*.json" in reviewer
