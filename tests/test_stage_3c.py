"""Phase 4a — Stage 3.c (smoke gate + diagnostician + producer fix routing).

The most complex single stage by failure-mode count (16 leaves). Tests
cover the smoke-fix loop, the diagnostician-output schema-validation path
(added 2026-05-19 to handle bev-distill's `value_origin_trace` regression),
and the cap-exhaustion judge invocation (also new in 2026-05-20).

Notebook execution is faked via run_script returning canned smoke stderr.
Each producer-fix iteration includes a re-render (run_script) + re-validate
(_run_stage2_script) step — tests queue all of these.

Per-iteration flow when smoke fails:
    smoke_run_notebook.py (script, exit 1)
    → diagnostician dispatch
    → producer fix dispatch
    → render_notebook.py (script)
    → validate_notebook_output.py (stage2 script)
    → loop back to smoke_run_notebook.py
"""

from __future__ import annotations

import json
from pathlib import Path

from tests.helpers.assertions import (
    assert_dispatched,
    assert_stage_completed,
    assert_stage_degraded,
    assert_stage_halted,
)
from tests.helpers.inject import queue_judge_decision_object
from tests.helpers.state import make_state


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _minimal_notebook_json(num_cells: int = 10) -> dict:
    """A `notebook.ipynb` with N code cells. Tests that simulate the cap-
    exhaustion loop need ≥ STAGE_2_RETRY_CAP+1 cells so each iteration can
    fail at a different cell (avoiding the forbidden-target retry path
    that fires when the same cell fails twice in a row)."""
    return {
        "cells": [
            {
                "cell_type": "code",
                "source": f"print('cell {i}')\n",
                "execution_count": i + 1,
                "metadata": {},
                "outputs": [],
            }
            for i in range(num_cells)
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3", "language": "python", "name": "python3",
            },
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def _seed_notebook(run_dir: Path) -> None:
    (run_dir / "notebook.ipynb").write_text(
        json.dumps(_minimal_notebook_json()), encoding="utf-8"
    )


def _smoke_stderr_cell_fails(cell: int = 0,
                              tb: str = "RuntimeError: BN bug") -> str:
    return (
        f"  [code  1/1  cell  {cell}/1]  done:    0.1s\n"
        f"notebook execution failed at cell {cell}\n\n"
        f"--- failing cell source (code) ---\n"
        f"print('hello')\n\n"
        f"--- error traceback ---\n"
        f"{tb}\n"
    )


def _valid_smoke_diagnosis(
    target_agent: str = "r2c-method-coder",
    target_file: str = "method/method.py",
    root_cause: str = "Test root cause.",
) -> str:
    return json.dumps({
        "schema_version": "1.0.0",
        "target_agent": target_agent,
        "target_file": target_file,
        "bug_shape": "uncatalogued",
        "root_cause": root_cause,
        "proposed_fix": "Test proposed fix.",
        "reasoning": "Test reasoning with value-origin trace.",
        "paper_fidelity_check": "Paper-faithful.",
        "value_origin_trace": [
            "step1: failure site",
            "step2: upstream",
            "step3: origin",
        ],
    }, indent=2)


def _schema_invalid_smoke_diagnosis_missing_trace() -> str:
    return json.dumps({
        "schema_version": "1.0.0",
        "target_agent": "r2c-method-coder",
        "target_file": "method/method.py",
        "bug_shape": "uncatalogued",
        "root_cause": "Test root cause.",
        "proposed_fix": "Test proposed fix.",
        "reasoning": "Test reasoning.",
        "paper_fidelity_check": "Paper-faithful.",
        # value_origin_trace: MISSING
    }, indent=2)


def _queue_post_fix_render_and_validate(fake_subprocess):
    """After each producer fix dispatch, the smoke loop re-renders the
    notebook + re-runs validate_notebook_output before re-attempting smoke."""
    fake_subprocess.expect_script(returncode=0)  # render_notebook.py
    fake_subprocess.expect_stage2_script(ok=True)  # validate_notebook_output.py


# ---------------------------------------------------------------------------
# Happy path + environmental failures
# ---------------------------------------------------------------------------


def test_stage_3c_happy_path_completes(fake_dispatch, fake_subprocess, run_dir):
    """Smoke gate exit_code=0 → stage completes immediately."""
    from run_pipeline import run_stage_3c
    state = make_state(run_dir)
    _seed_notebook(run_dir)

    fake_subprocess.expect_script(returncode=0, stdout="ok", stderr="")

    result = run_stage_3c(state)
    assert_stage_completed(result, "stage_3c")
    assert len(fake_dispatch.calls) == 0


def test_stage_3c_environmental_failure_degrades_without_retry(
    fake_dispatch, fake_subprocess, run_dir
):
    """smoke_run_notebook.py exit_code=3 (nbclient/kernel unavailable) is an
    ENVIRONMENTAL failure: the rendered notebook exists, so the stage degrades
    and the run continues to review + delivery — no LLM retry, no hard halt.
    (ENV-1: consistent with stage_2d's environmental degrade.)"""
    from run_pipeline import run_stage_3c
    state = make_state(run_dir)
    _seed_notebook(run_dir)

    fake_subprocess.expect_script(returncode=3, stderr="No module named 'nbclient'")

    result = run_stage_3c(state)
    assert_stage_degraded(result, stage_id="stage_3c")
    assert len(fake_dispatch.calls) == 0


def test_stage_3c_pipeline_fault_setup_error_halts_without_retry(
    fake_dispatch, fake_subprocess, run_dir
):
    """smoke_run_notebook.py exit_code=2 (notebook missing/unparseable) is a
    pipeline fault — nothing to ship — so it still halts immediately."""
    from run_pipeline import run_stage_3c
    state = make_state(run_dir)
    _seed_notebook(run_dir)

    fake_subprocess.expect_script(returncode=2, stderr="cannot parse notebook.ipynb")

    result = run_stage_3c(state)
    assert_stage_halted(result, stage_id="stage_3c",
                        reason_contains="smoke gate setup error")
    assert len(fake_dispatch.calls) == 0


def test_stage_3c_unparseable_stderr_halts(fake_dispatch, fake_subprocess, run_dir):
    """smoke fails (exit 1) but stderr has no recognizable cell index → halt."""
    from run_pipeline import run_stage_3c
    state = make_state(run_dir)
    _seed_notebook(run_dir)

    fake_subprocess.expect_script(returncode=1, stderr="some random garbage")

    result = run_stage_3c(state)
    assert_stage_halted(result, stage_id="stage_3c",
                        reason_contains="could not parse failing cell")


# ---------------------------------------------------------------------------
# Cell raises → diagnostician → producer fix → recovery
# ---------------------------------------------------------------------------


def test_stage_3c_cell_raises_diagnostician_method_coder_recovers(
    fake_dispatch, fake_subprocess, run_dir
):
    """Full smoke-fix happy cycle: cell raises → diagnostician writes valid
    diagnosis routing to method-coder → method-coder dispatched → re-render
    + re-validate pass → next smoke iteration passes → stage completes."""
    from run_pipeline import run_stage_3c
    state = make_state(run_dir)
    _seed_notebook(run_dir)

    # iter 0: smoke fails
    fake_subprocess.expect_script(returncode=1,
                                   stderr=_smoke_stderr_cell_fails(0))
    fake_dispatch.expect(
        agent="r2c-smoke-diagnostician",
        writes={".pipeline/smoke_diagnosis.json": _valid_smoke_diagnosis()},
    )
    fake_dispatch.expect(
        agent="r2c-method-coder",
        writes={"method/method.py": "# fixed method.py\n"},
    )
    _queue_post_fix_render_and_validate(fake_subprocess)

    # iter 1: smoke passes
    fake_subprocess.expect_script(returncode=0)

    result = run_stage_3c(state)
    assert_stage_completed(result, "stage_3c")
    assert_dispatched(fake_dispatch, "r2c-smoke-diagnostician", times=1)
    assert_dispatched(fake_dispatch, "r2c-method-coder", times=1)


# ---------------------------------------------------------------------------
# Diagnostician output schema-invalid → judge → repair-mode
# ---------------------------------------------------------------------------


def test_stage_3c_schema_invalid_diagnosis_judge_repair_recovers(
    fake_dispatch, fake_subprocess, run_dir
):
    """**bev-distill 2026-05-19 regression**: diagnostician writes a
    diagnosis missing `value_origin_trace` → judge → diagnostician repair
    → smoke-fix proceeds → stage completes."""
    from run_pipeline import run_stage_3c
    state = make_state(run_dir)
    _seed_notebook(run_dir)

    # iter 0: smoke fails
    fake_subprocess.expect_script(returncode=1,
                                   stderr=_smoke_stderr_cell_fails(0))
    # diagnostician writes INVALID diagnosis (missing value_origin_trace)
    fake_dispatch.expect(
        agent="r2c-smoke-diagnostician",
        writes={".pipeline/smoke_diagnosis.json":
                _schema_invalid_smoke_diagnosis_missing_trace()},
    )
    # judge invoked
    judge_decision = {
        "schema_version": "1.0.0",
        "stage_id": "stage_3c",
        "iteration": 0,
        "validator_label": "schema_validation:smoke_diagnosis",
        "classification": "producer_fixable",
        "action": "dispatch_fix",
        "target_agent": "r2c-smoke-diagnostician",
        "finding": {
            "id": "JUDGE001", "severity": "critical",
            "description": "missing value_origin_trace",
            "proposed_fix": "add 3-entry value_origin_trace",
        },
        "rationale": "diagnosis missing required field",
        "confidence": "high",
        "files_examined": ["scripts/run_pipeline.py"],
    }
    queue_judge_decision_object(fake_dispatch, judge_decision)
    # diagnostician repair-dispatched, writes valid diagnosis
    fake_dispatch.expect(
        agent="r2c-smoke-diagnostician",
        writes={".pipeline/smoke_diagnosis.json": _valid_smoke_diagnosis()},
    )
    # smoke-fix continues
    fake_dispatch.expect(
        agent="r2c-method-coder",
        writes={"method/method.py": "# fixed\n"},
    )
    _queue_post_fix_render_and_validate(fake_subprocess)

    # iter 1: smoke passes
    fake_subprocess.expect_script(returncode=0)

    result = run_stage_3c(state)
    assert_stage_completed(result, "stage_3c")
    assert_dispatched(fake_dispatch, "r2c-halt-judge", times=1)
    assert_dispatched(fake_dispatch, "r2c-smoke-diagnostician", times=2)


def test_stage_3c_missing_diagnosis_file_direct_retry_recovers(
    fake_dispatch, fake_subprocess, run_dir
):
    """Diagnostician writes NO file → ONE direct missing-output retry
    (the Think stop-early pattern, NO judge involved) → ok. The judge is
    reserved for the schema-invalid case and for a retry that also writes
    nothing (detr 2026-07-04: a zero-write turn went straight to the judge,
    which halted a run with a routable fix behind it)."""
    from run_pipeline import run_stage_3c
    state = make_state(run_dir)
    _seed_notebook(run_dir)

    fake_subprocess.expect_script(returncode=1,
                                   stderr=_smoke_stderr_cell_fails(0))
    fake_dispatch.expect(agent="r2c-smoke-diagnostician", writes={})
    fake_dispatch.expect(
        agent="r2c-smoke-diagnostician",
        writes={".pipeline/smoke_diagnosis.json": _valid_smoke_diagnosis()},
    )
    fake_dispatch.expect(
        agent="r2c-method-coder",
        writes={"method/method.py": "# fixed\n"},
    )
    _queue_post_fix_render_and_validate(fake_subprocess)
    fake_subprocess.expect_script(returncode=0)

    result = run_stage_3c(state)
    assert_stage_completed(result, "stage_3c")
    assert_dispatched(fake_dispatch, "r2c-smoke-diagnostician", times=2)
    assert_dispatched(fake_dispatch, "r2c-halt-judge", times=0)
    # The retry dispatch carries retry_mode's missing-file framing, and the
    # write-first enforcement (detr 2026-07-04: both diagnostician turns
    # reasoned past the per-step output cap and wrote nothing — the retry
    # must demand the best-current-hypothesis Write as the FIRST tool call).
    retry_call = [c for c in fake_dispatch.calls
                  if c.agent == "r2c-smoke-diagnostician"][1]
    assert "smoke_diagnosis.json" in retry_call.prompt
    assert "FIRST tool call" in retry_call.prompt
    assert "BEST CURRENT HYPOTHESIS" in retry_call.prompt


def test_stage_3c_double_zero_write_gets_driver_fallback_diagnosis(
    fake_dispatch, fake_subprocess, run_dir
):
    """No file after the direct retry either → the DRIVER authors the
    fallback diagnosis and the fix loop proceeds without the judge
    (maintainer-approved 2026-07-05, option 2 of the detr Think-tier brief: the
    write must not depend on a model turn surviving its own thinking
    budget). The fallback is loud (run event + assumptions.md), labeled
    driver-authored in every judgment field, and carries the
    driver-recorded dispatch evidence."""
    from run_pipeline import run_stage_3c
    state = make_state(run_dir)
    _seed_notebook(run_dir)

    fake_subprocess.expect_script(returncode=1,
                                   stderr=_smoke_stderr_cell_fails(0))
    fake_dispatch.expect(agent="r2c-smoke-diagnostician", writes={})
    fake_dispatch.expect(agent="r2c-smoke-diagnostician", writes={})
    # Mechanical routing for a no-method-frame traceback at cell 0 is the
    # notebook-generator; the fallback routes the fix there directly.
    fake_dispatch.expect(
        agent="r2c-notebook-generator",
        writes={".pipeline/notebook_draft.py": "# fixed\n"},
    )
    _queue_post_fix_render_and_validate(fake_subprocess)
    fake_subprocess.expect_script(returncode=0)

    result = run_stage_3c(state)
    assert_stage_completed(result, "stage_3c")
    assert_dispatched(fake_dispatch, "r2c-halt-judge", times=0)

    # The fallback diagnosis is on disk, schema-valid, and unmistakably
    # driver-authored.
    diag = json.loads(
        (run_dir / ".pipeline" / "smoke_diagnosis.json").read_text())
    assert diag["target_agent"] == "r2c-notebook-generator"
    assert diag["target_file"] == ".pipeline/notebook_draft.py"
    assert "DRIVER-AUTHORED FALLBACK" in diag["root_cause"]
    assert "Not assessed (driver-authored fallback)" in diag["paper_fidelity_check"]
    assert "did NOT time out" in diag["reasoning"]

    # The producer's fix prompt carries the fallback framing (SMOKE001
    # injects root_cause verbatim), so the producer knows the routing is
    # coarse.
    fix_call = next(
        c for c in fake_dispatch.calls if c.agent == "r2c-notebook-generator")
    assert "DRIVER-AUTHORED FALLBACK" in fix_call.prompt

    # Loud: assumptions.md entry + run event.
    import run_layout
    assumptions = (run_dir / run_layout.ASSUMPTIONS_MD).read_text()
    assert "Driver-authored fallback smoke diagnosis" in assumptions
    events = [json.loads(l) for l in
              (run_dir / ".pipeline" / "run_events.jsonl").read_text().splitlines()]
    assert any(e["event_type"] == "smoke_diagnosis_driver_fallback" for e in events)


def test_driver_fallback_diagnosis_refuses_burned_targets(run_dir):
    """No honest fallback exists when the mechanical pick's file is already
    in failed_targets — the helper returns None (and the caller keeps the
    judge path, which the schema-invalid tests pin)."""
    from run_pipeline import _write_driver_fallback_smoke_diagnosis
    state = make_state(run_dir)

    fallback = _write_driver_fallback_smoke_diagnosis(
        state,
        mechanical_producer="notebook-generator",
        stderr_tail=_smoke_stderr_cell_fails(0),
        fail_cell=0, section=0, exception_class="RuntimeError",
        failed_targets=[".pipeline/notebook_draft.py"],
        dispatch_evidence="evidence",
    )
    assert fallback is None
    assert not (run_dir / ".pipeline" / "smoke_diagnosis.json").exists()

    # A traceback with a method/ frame anchors target_file to that frame.
    stderr = (
        "notebook execution failed at cell 3\n"
        "--- error traceback ---\n"
        'File "method/method.py", line 120, in _hungarian_matching\n'
        "RuntimeError: shape mismatch\n"
    )
    fallback = _write_driver_fallback_smoke_diagnosis(
        state,
        mechanical_producer="method-coder",
        stderr_tail=stderr,
        fail_cell=3, section=5, exception_class="RuntimeError",
        failed_targets=[],
        dispatch_evidence="evidence",
    )
    assert fallback is not None
    assert fallback["target_file"] == "method/method.py"
    assert "method/method.py:120 in _hungarian_matching" in fallback["root_cause"]


def test_diagnostician_dispatch_evidence_bounds_mechanism_claims():
    """Unit: the evidence block states completed-vs-not from DispatchResult
    fields and only names the output-budget class when a completed no-write
    attempt exists. Empty attempts render nothing (no block in the prompt)."""
    from opencode_client import DispatchResult

    from run_pipeline import _diagnostician_dispatch_evidence

    def _result(completed, elapsed, error=None):
        return DispatchResult(
            session_id="s", user_message_id="u", assistant_message_id="a",
            completed=completed, error=error, elapsed_s=elapsed,
        )

    assert _diagnostician_dispatch_evidence([]) == ""

    completed = _diagnostician_dispatch_evidence(
        [("attempt 1", _result(True, 755.8)),
         ("attempt 2 (write-first retry)", _result(True, 299.5))])
    assert "COMPLETED in 755.8s" in completed
    assert "COMPLETED in 299.5s" in completed
    assert "did NOT time out" in completed
    assert "output-budget class" in completed
    assert "Do NOT attribute it to a dispatch timeout" in completed

    failed = _diagnostician_dispatch_evidence(
        [("attempt 1", _result(False, 12.0, error={"name": "AbortedError"}))])
    assert "did not complete" in failed
    assert "AbortedError" in failed
    # No completed no-write attempt: the output-budget reading is NOT
    # asserted (mechanism claims stay bounded by the evidence).
    assert "output-budget class" not in failed


# ---------------------------------------------------------------------------
# Cap exhaustion — judge as last-chance routing
# ---------------------------------------------------------------------------


def _queue_failed_iter(fake_dispatch, fake_subprocess, *, iter_index: int):
    """Queue one full failure iteration: smoke fail (at a different cell
    per iter, to avoid triggering the forbidden-target retry which fires
    when the same cell fails twice in a row) + diag + producer fix +
    re-render + re-validate."""
    fake_subprocess.expect_script(
        returncode=1, stderr=_smoke_stderr_cell_fails(iter_index),
    )
    fake_dispatch.expect(
        agent="r2c-smoke-diagnostician",
        writes={".pipeline/smoke_diagnosis.json": _valid_smoke_diagnosis()},
    )
    fake_dispatch.expect(
        agent="r2c-method-coder",
        writes={"method/method.py": f"# iter {iter_index}\n"},
    )
    _queue_post_fix_render_and_validate(fake_subprocess)


def test_stage_3c_cap_exhausted_judge_degrades_with_rationale(
    fake_dispatch, fake_subprocess, run_dir
):
    """At cap exhaustion with the judge deciding NOT to recover, stage 3.c
    DEGRADES (halt→degrade redesign): the notebook was generated + rendered and
    runs up to the failing cell, so it ships with the failing-cell pointer +
    judge rationale logged to KNOWN_ISSUES.md rather than halting the run."""
    from run_pipeline import STAGE_2_RETRY_CAP, run_stage_3c
    state = make_state(run_dir)
    _seed_notebook(run_dir)

    # STAGE_2_RETRY_CAP iterations of fix-producer failures
    for i in range(STAGE_2_RETRY_CAP):
        _queue_failed_iter(fake_dispatch, fake_subprocess, iter_index=i)
    # cap iteration: smoke fails one more time
    fake_subprocess.expect_script(returncode=1,
                                   stderr=_smoke_stderr_cell_fails(0))
    # judge invoked at cap with halt action
    judge_decision = {
        "schema_version": "1.0.0",
        "stage_id": "stage_3c",
        "iteration": STAGE_2_RETRY_CAP,
        "validator_label": f"smoke_loop_cap_exhausted:cap={STAGE_2_RETRY_CAP}",
        "classification": "upstream_issue",
        "action": "halt",
        "target_agent": None,
        "finding": None,
        "rationale": "upstream BN bug survived stage 2.d's dry-run",
        "confidence": "high",
        "files_examined": ["scripts/run_pipeline.py", "method/model.py"],
    }
    queue_judge_decision_object(fake_dispatch, judge_decision)

    result = run_stage_3c(state)
    assert_stage_degraded(result, stage_id="stage_3c", run_dir=run_dir,
                          known_issue_contains="upstream BN bug")
    # The failing cell pointer is surfaced to the researcher.
    assert_stage_degraded(result, run_dir=run_dir,
                          known_issue_contains="notebook.ipynb")
    assert_dispatched(fake_dispatch, "r2c-halt-judge", times=1)


def test_stage_3c_cap_exhausted_judge_recovery_dispatch_recovers(
    fake_dispatch, fake_subprocess, run_dir
):
    """At cap exhaustion, judge HIGH confidence dispatches ONE more
    producer fix. Final smoke passes → stage completes via judge
    cap-recovery."""
    from run_pipeline import STAGE_2_RETRY_CAP, run_stage_3c
    state = make_state(run_dir)
    _seed_notebook(run_dir)

    for i in range(STAGE_2_RETRY_CAP):
        _queue_failed_iter(fake_dispatch, fake_subprocess, iter_index=i)
    # cap iteration: smoke fails one more time, at a different cell
    fake_subprocess.expect_script(
        returncode=1, stderr=_smoke_stderr_cell_fails(STAGE_2_RETRY_CAP),
    )
    # judge with high-confidence dispatch_fix
    judge_decision = {
        "schema_version": "1.0.0",
        "stage_id": "stage_3c",
        "iteration": STAGE_2_RETRY_CAP,
        "validator_label": f"smoke_loop_cap_exhausted:cap={STAGE_2_RETRY_CAP}",
        "classification": "producer_fixable",
        "action": "dispatch_fix",
        "target_agent": "r2c-architecture-coder",
        "finding": {
            "id": "JUDGE001", "severity": "critical",
            "description": "actual bug is the teacher's BN placement",
            "proposed_fix": "wrap point-MLP to flatten before BN",
        },
        "rationale": "iterations targeted method.py but bug is in model.py",
        "confidence": "high",
        "files_examined": ["scripts/run_pipeline.py", "method/model.py"],
    }
    queue_judge_decision_object(fake_dispatch, judge_decision)
    fake_dispatch.expect(agent="r2c-architecture-coder",
                         writes={"method/model.py": "# fixed BN\n"})
    # Re-render after the blessed fix, THEN the final smoke. Until
    # 2026-07-14 the cap-recovery path skipped the render and re-smoked
    # the STALE notebook: the Rethinking run's blessed fix landed in
    # notebook_draft.py, smoke "passed" a notebook that did not contain
    # it, and the delivery shipped without its own final fix.
    fake_subprocess.expect_script(returncode=0)  # render_notebook.py
    fake_subprocess.expect_script(returncode=0)  # smoke passes

    result = run_stage_3c(state)
    assert_stage_completed(result, "stage_3c")
    assert_dispatched(fake_dispatch, "r2c-architecture-coder", times=1)


# ---------------------------------------------------------------------------
# Anti-fixation guard: same cell, different exception class = cascading progress
# (R2C 2026-05-21 bev-distill regression)
# ---------------------------------------------------------------------------


def test_extract_exception_class_finds_last_match():
    """Helper extracts the LAST exception line (the actual raised one,
    not chained-from frames)."""
    from run_pipeline import _extract_exception_class

    assert _extract_exception_class("RuntimeError: shape mismatch\n") == "RuntimeError"
    assert _extract_exception_class(
        "Traceback...\nRuntimeError: outer\n\nDuring handling...\n"
        "TypeError: inner\n"
    ) == "TypeError"
    assert _extract_exception_class("ValueError: x\n") == "ValueError"
    assert _extract_exception_class("IndexError: y\n") == "IndexError"
    # ANSI-stripped
    assert _extract_exception_class(
        "\x1b[31mRuntimeError\x1b[39m                   Traceback\n"
        "\x1b[31mRuntimeError\x1b[39m: failed\n"
    ) == "RuntimeError"
    # No exception → None
    assert _extract_exception_class("no error here\nmoved on\n") is None
    assert _extract_exception_class("") is None


def test_stage_3c_same_cell_different_exception_recognized_as_progress(
    fake_dispatch, fake_subprocess, run_dir
):
    """**R2C 2026-05-21 bev-distill regression**: smoke fails twice at the
    same cell, but with DIFFERENT exception classes (RuntimeError → TypeError).
    The anti-fixation guard must recognize this as cascading progress (prior
    fix landed and exposed the next layer), NOT as fixation. The diagnostician
    must remain free to pick the same target_file on iter 1 without tripping
    the forbidden-target retry."""
    from run_pipeline import run_stage_3c
    state = make_state(run_dir)
    _seed_notebook(run_dir)

    # iter 0: cell 0 fails with RuntimeError
    fake_subprocess.expect_script(
        returncode=1,
        stderr=_smoke_stderr_cell_fails(0, tb="RuntimeError: dtype mismatch"),
    )
    fake_dispatch.expect(
        agent="r2c-smoke-diagnostician",
        writes={".pipeline/smoke_diagnosis.json": _valid_smoke_diagnosis(
            target_agent="r2c-notebook-generator",
            target_file=".pipeline/notebook_draft.py",
        )},
    )
    fake_dispatch.expect(
        agent="r2c-notebook-generator",
        writes={".pipeline/notebook_draft.py": "# iter 0 fix\n"},
    )
    _queue_post_fix_render_and_validate(fake_subprocess)

    # iter 1: same cell 0 fails with a DIFFERENT exception (TypeError).
    # Anti-fixation must NOT add notebook_draft.py to failed_targets —
    # the diagnostician picking it again is legitimate cascading progress.
    fake_subprocess.expect_script(
        returncode=1,
        stderr=_smoke_stderr_cell_fails(0, tb="TypeError: bad arg"),
    )
    fake_dispatch.expect(
        agent="r2c-smoke-diagnostician",
        writes={".pipeline/smoke_diagnosis.json": _valid_smoke_diagnosis(
            target_agent="r2c-notebook-generator",
            target_file=".pipeline/notebook_draft.py",
        )},
    )
    # NO forbidden-target retry dispatch — guard recognized progress.
    # Direct producer fix dispatch proceeds.
    fake_dispatch.expect(
        agent="r2c-notebook-generator",
        writes={".pipeline/notebook_draft.py": "# iter 1 fix\n"},
    )
    _queue_post_fix_render_and_validate(fake_subprocess)

    # iter 2: smoke passes
    fake_subprocess.expect_script(returncode=0)

    result = run_stage_3c(state)
    assert_stage_completed(result, "stage_3c")
    assert_dispatched(fake_dispatch, "r2c-smoke-diagnostician", times=2)
    assert_dispatched(fake_dispatch, "r2c-notebook-generator", times=2)


def test_stage_3c_same_cell_same_exception_triggers_fixation_guard(
    fake_dispatch, fake_subprocess, run_dir
):
    """Counterpoint to the cascading-progress test: when the exception class
    is unchanged across iterations at the same cell, the prior fix really
    didn't land. Anti-fixation guard fires: diagnostician's same-file pick
    triggers forbidden-target retry, then halts if the retry still picks the
    same target. This preserves the original 2026-05-14 bev-distill cascade
    protection."""
    from run_pipeline import run_stage_3c
    state = make_state(run_dir)
    _seed_notebook(run_dir)

    # iter 0: cell 0 fails with RuntimeError
    fake_subprocess.expect_script(
        returncode=1,
        stderr=_smoke_stderr_cell_fails(0, tb="RuntimeError: same bug"),
    )
    fake_dispatch.expect(
        agent="r2c-smoke-diagnostician",
        writes={".pipeline/smoke_diagnosis.json": _valid_smoke_diagnosis(
            target_agent="r2c-method-coder", target_file="method/method.py",
        )},
    )
    fake_dispatch.expect(
        agent="r2c-method-coder",
        writes={"method/method.py": "# iter 0 fix (didn't land)\n"},
    )
    _queue_post_fix_render_and_validate(fake_subprocess)

    # iter 1: same cell 0, SAME exception (RuntimeError). Fix didn't unstick.
    # Anti-fixation guard adds method/method.py to failed_targets. Then
    # diagnostician picks it again → forbidden-target retry → still picks
    # the same target → halt.
    fake_subprocess.expect_script(
        returncode=1,
        stderr=_smoke_stderr_cell_fails(0, tb="RuntimeError: same bug"),
    )
    fake_dispatch.expect(
        agent="r2c-smoke-diagnostician",
        writes={".pipeline/smoke_diagnosis.json": _valid_smoke_diagnosis(
            target_agent="r2c-method-coder", target_file="method/method.py",
        )},
    )
    # Forbidden-target retry fires
    fake_dispatch.expect(
        agent="r2c-smoke-diagnostician",
        writes={".pipeline/smoke_diagnosis.json": _valid_smoke_diagnosis(
            target_agent="r2c-method-coder", target_file="method/method.py",
        )},
    )
    # Still picks forbidden target → halt
    result = run_stage_3c(state)
    assert_stage_halted(result, stage_id="stage_3c",
                        reason_contains="failed_targets")


# ---------------------------------------------------------------------------
# Smart skip: stage 3.c re-runs only when upstream files have changed
# (R2C 2026-05-22 — resume-from-stage-4-halt should NOT re-execute smoke)
# ---------------------------------------------------------------------------


def _seed_3c_upstream(run_dir: Path) -> None:
    """Seed the upstream files stage 3.c's smart-skip checks against."""
    (run_dir / "method").mkdir(parents=True, exist_ok=True)
    (run_dir / "method" / "model.py").write_text("# model\n")
    (run_dir / "method" / "training.py").write_text("# training\n")
    (run_dir / "method" / "method.py").write_text("# method\n")
    (run_dir / "method" / "data.py").write_text("# data\n")
    (run_dir / "method" / "__init__.py").write_text("# init\n")
    (run_dir / ".pipeline" / "notebook_draft.py").write_text("# nb\n")
    (run_dir / ".pipeline" / "params.json").write_text("{}\n")
    (run_dir / "requirements.txt").write_text("# reqs\n")
    _seed_notebook(run_dir)


def _write_sentinel_and_digest(run_dir: Path) -> None:
    """Write `stage_3c.complete` + a content digest of every upstream
    file. Mirrors what the driver's main loop does on stage 3.c completion."""
    from run_pipeline import write_stage_3c_upstream_digest
    from tests.helpers.state import make_state
    state = make_state(run_dir)
    (run_dir / ".pipeline" / "stage_3c.complete").write_text("")
    write_stage_3c_upstream_digest(state.paths)


def test_stage_3c_skips_when_sentinel_and_content_digest_matches(
    fake_dispatch, fake_subprocess, run_dir
):
    """**R2C 2026-05-22 regression**: resume-from-stage-4-halt should skip
    stage 3.c when no upstream file content has changed. Sentinel + digest
    both present, contents unchanged → cache hit, skip."""
    from run_pipeline import run_stage_3c
    state = make_state(run_dir)
    _seed_3c_upstream(run_dir)
    _write_sentinel_and_digest(run_dir)

    result = run_stage_3c(state)

    assert result.status == "skipped", f"expected skipped, got {result.status}"
    assert "content digest matches" in (result.notes or "")
    # No subprocess calls, no agent dispatches — smoke gate never ran
    assert len(fake_subprocess.script_calls) == 0
    assert len(fake_dispatch.calls) == 0


def test_stage_3c_skips_when_mtime_changed_but_content_identical(
    fake_dispatch, fake_subprocess, run_dir
):
    """**R2C 2026-05-22 root cause**: stage 2.d's idempotent
    `finalize_package_init.py` re-writes `method/__init__.py` and
    `requirements.txt` on every resume, bumping their mtimes but keeping
    content identical. An mtime-based check spuriously misses; a
    content-digest check correctly sees no real change and skips."""
    import os
    from run_pipeline import run_stage_3c
    state = make_state(run_dir)
    _seed_3c_upstream(run_dir)
    _write_sentinel_and_digest(run_dir)

    # Simulate idempotent re-run: rewrite files with IDENTICAL content but
    # newer mtime than the sentinel.
    later_mtime = (run_dir / ".pipeline" / "stage_3c.complete").stat().st_mtime + 100
    init_path = run_dir / "method" / "__init__.py"
    reqs_path = run_dir / "requirements.txt"
    init_path.write_text(init_path.read_text())  # same content
    reqs_path.write_text(reqs_path.read_text())  # same content
    os.utime(init_path, (later_mtime, later_mtime))
    os.utime(reqs_path, (later_mtime, later_mtime))

    result = run_stage_3c(state)

    assert result.status == "skipped", f"expected skipped, got {result.status}"
    assert "content digest matches" in (result.notes or "")
    assert len(fake_subprocess.script_calls) == 0


def test_stage_3c_re_runs_when_method_py_content_changed(
    fake_dispatch, fake_subprocess, run_dir
):
    """If `method/method.py`'s actual CONTENT changed (e.g., method-coder
    fix-dispatched after the prior 3.c completion), the digest mismatches
    and the gate re-executes."""
    from run_pipeline import run_stage_3c
    state = make_state(run_dir)
    _seed_3c_upstream(run_dir)
    _write_sentinel_and_digest(run_dir)

    # Actually change content of method/method.py
    (run_dir / "method" / "method.py").write_text("# method (updated)\n")

    fake_subprocess.expect_script(returncode=0)

    result = run_stage_3c(state)
    assert_stage_completed(result, "stage_3c")
    assert len(fake_subprocess.script_calls) == 1


def test_stage_3c_re_runs_when_no_sentinel(
    fake_dispatch, fake_subprocess, run_dir
):
    """First-time runs (no sentinel) must re-run the gate. Smart-skip
    helper returns None when the sentinel is absent — no digest comparison
    even attempted."""
    from run_pipeline import run_stage_3c
    state = make_state(run_dir)
    _seed_3c_upstream(run_dir)
    # NO sentinel written

    fake_subprocess.expect_script(returncode=0)

    result = run_stage_3c(state)
    assert_stage_completed(result, "stage_3c")
    assert len(fake_subprocess.script_calls) == 1


def test_stage_3c_re_runs_when_notebook_draft_content_changed(
    fake_dispatch, fake_subprocess, run_dir
):
    """Notebook-side content changes (notebook_draft.py rewritten by a
    stage 3.a re-dispatch) trigger digest mismatch and re-run. Tests the
    `.pipeline/` branch of upstream-file enumeration."""
    from run_pipeline import run_stage_3c
    state = make_state(run_dir)
    _seed_3c_upstream(run_dir)
    _write_sentinel_and_digest(run_dir)

    # notebook_draft.py content actually changed
    (run_dir / ".pipeline" / "notebook_draft.py").write_text("# nb (updated)\n")

    fake_subprocess.expect_script(returncode=0)  # render_notebook.py
    fake_subprocess.expect_stage2_script(ok=True)  # validate_notebook_output.py
    fake_subprocess.expect_script(returncode=0)  # smoke_run_notebook.py

    result = run_stage_3c(state)
    assert_stage_completed(result, "stage_3c")
    assert fake_subprocess.script_calls[0]["args"][0] == "scripts/render_notebook.py"
    assert fake_subprocess.stage2_calls[0]["args"] == [
        "scripts/validate_notebook_output.py",
    ]
    assert fake_subprocess.script_calls[1]["args"][0] == "scripts/smoke_run_notebook.py"


def test_stage_3c_rerenders_when_params_content_changed(
    fake_dispatch, fake_subprocess, run_dir
):
    """params.json is rendered into the notebook params cell and table. A
    Stage 3.c digest miss caused by params changes must regenerate the
    notebook before smoke execution."""
    from run_pipeline import run_stage_3c
    state = make_state(run_dir)
    _seed_3c_upstream(run_dir)
    _write_sentinel_and_digest(run_dir)

    (run_dir / ".pipeline" / "params.json").write_text(
        '{"params": {"R_0": {"value": 7.843}}}\n',
        encoding="utf-8",
    )

    fake_subprocess.expect_script(returncode=0)  # render_notebook.py
    fake_subprocess.expect_stage2_script(ok=True)  # validate_notebook_output.py
    fake_subprocess.expect_script(returncode=0)  # smoke_run_notebook.py

    result = run_stage_3c(state)

    assert_stage_completed(result, "stage_3c")
    assert fake_subprocess.script_calls[0]["args"][0] == "scripts/render_notebook.py"
    assert fake_subprocess.script_calls[1]["args"][0] == "scripts/smoke_run_notebook.py"


def test_stage_3c_bootstraps_digest_when_sentinel_present_but_digest_missing(
    fake_dispatch, fake_subprocess, run_dir
):
    """**R2C 2026-05-22 user-facing fix path**: sentinel was written by a
    prior driver version without digest support. On first run with the
    new code, trust the sentinel as authoritative, snapshot the current
    upstream state as the new ground truth, and skip the gate. Subsequent
    runs use the snapshot."""
    from run_pipeline import _stage_3c_upstream_digest_path, run_stage_3c
    state = make_state(run_dir)
    _seed_3c_upstream(run_dir)
    # Sentinel exists but digest file does NOT — bootstrap case
    (run_dir / ".pipeline" / "stage_3c.complete").write_text("")

    digest_path = _stage_3c_upstream_digest_path(state.paths)
    assert not digest_path.exists()

    result = run_stage_3c(state)

    assert result.status == "skipped", f"expected skipped, got {result.status}"
    assert "bootstrapped" in (result.notes or "")
    # Digest file written from current state
    assert digest_path.is_file()
    # Smoke gate never ran
    assert len(fake_subprocess.script_calls) == 0


# ---------------------------------------------------------------------------
# Tail-truncated diagnosis salvage (detr-distill 2026-07-03: two
# diagnostician outputs in a row ended at EOF one `}` short; each cost a
# halt the content didn't warrant).
# ---------------------------------------------------------------------------


def test_read_smoke_diagnosis_salvages_tail_truncation(run_dir):
    """A valid diagnosis missing only its closing brace parses via the
    salvage path and passes full schema validation."""
    from run_pipeline import _read_smoke_diagnosis
    state = make_state(run_dir)
    full = _valid_smoke_diagnosis()
    truncated = full.rstrip().removesuffix("}").rstrip().removesuffix(",")
    (state.paths.pipeline_dir / "smoke_diagnosis.json").write_text(
        truncated, encoding="utf-8")

    diag, err = _read_smoke_diagnosis(state)

    assert err is None
    assert diag["target_agent"] == "r2c-method-coder"
    assert len(diag["value_origin_trace"]) == 3


def test_read_smoke_diagnosis_truncation_inside_string_fails_closed(run_dir):
    """Truncation that eats a REQUIRED field still halts: the salvage
    parse yields an object, but schema validation rejects it — nothing
    partial ships."""
    from run_pipeline import _read_smoke_diagnosis
    state = make_state(run_dir)
    full = _valid_smoke_diagnosis()
    # Cut mid-way through the document: value_origin_trace (required,
    # min 3 entries) is lost entirely.
    cut = full.index('"value_origin_trace"')
    (state.paths.pipeline_dir / "smoke_diagnosis.json").write_text(
        full[:cut], encoding="utf-8")

    diag, err = _read_smoke_diagnosis(state)

    assert diag is None
    assert err is not None


def test_salvage_refuses_mid_file_corruption():
    """A defect that is NOT at the tail (real corruption) is never
    'repaired' — the original parse error surfaces."""
    from run_pipeline import _salvage_truncated_json
    corrupt = '{"a": 1, "b": ]]], "c": 2, "d": 3, "e": 4}'
    assert _salvage_truncated_json(corrupt) is None


def test_salvage_recovers_the_live_detr_shape():
    """The exact 2026-07-03 shape: a complete object cut right before the
    final closing brace, ending on a string value + newline."""
    from run_pipeline import _salvage_truncated_json
    doc = ('{\n  "schema_version": "1.0.0",\n  "target_agent": "r2c-method-coder",\n'
           '  "root_cause": "shape mismatch in _focal_loss return"\n')
    out = _salvage_truncated_json(doc)
    assert out is not None
    assert out["root_cause"].startswith("shape mismatch")


# ---------------------------------------------------------------------------
# Smoke-time trainability pre-check (UB-6's beats-chance rule pulled forward;
# GBALD 2026-07-03 round 2: reshape bug → clean execution, below-chance curve,
# caught only as a stage-5 demoter after the fix loop had closed).
# ---------------------------------------------------------------------------


def _nb_with_outputs(acc_values, n_classes=10) -> str:
    lines = [f"Classes: {n_classes}\n"] + [
        f"round {i}: test_acc={v}\n" for i, v in enumerate(acc_values)]
    return json.dumps({
        "cells": [
            {"cell_type": "markdown", "source": "## 5. Evaluation\n",
             "metadata": {}},
            {"cell_type": "code", "source": "run_eval()\n",
             "execution_count": 1, "metadata": {},
             "outputs": [{"output_type": "stream", "name": "stdout",
                          "text": lines}]},
        ],
        "metadata": {}, "nbformat": 4, "nbformat_minor": 5,
    })


def test_trainability_check_flags_below_chance_series(run_dir):
    from run_pipeline import _trainability_smoke_check
    state = make_state(run_dir)
    (run_dir / "notebook.ipynb").write_text(
        _nb_with_outputs([0.05, 0.087, 0.06]), encoding="utf-8")

    result = _trainability_smoke_check(state)

    assert result is not None
    assert result.cell == 1  # the metric-printing code cell
    assert "SILENT NON-LEARNING" in result.stderr
    assert "0.087" in result.stderr
    assert "chance" in result.stderr.lower()
    # The arm names itself, so the driver's log line and run event cannot
    # describe a non-finite metric as non-learning (R2C-071).
    assert result.kind == "below_chance"


def test_trainability_check_passes_learning_curve(run_dir):
    from run_pipeline import _trainability_smoke_check
    state = make_state(run_dir)
    (run_dir / "notebook.ipynb").write_text(
        _nb_with_outputs([0.15, 0.42, 0.88]), encoding="utf-8")
    assert _trainability_smoke_check(state) is None


def test_trainability_check_skips_when_not_checkable(run_dir):
    """No series / no class count / unreadable notebook → None, never a
    failure — the stage-5 battery discloses those honestly."""
    from run_pipeline import _trainability_smoke_check
    state = make_state(run_dir)
    # No accuracy series at all.
    _seed_notebook(run_dir)
    assert _trainability_smoke_check(state) is None
    # Series present but no detectable class count.
    nb = json.loads(_nb_with_outputs([0.05, 0.06]))
    nb["cells"][1]["outputs"][0]["text"] = [
        "round 0: test_acc=0.05\n", "round 1: test_acc=0.06\n"]
    (run_dir / "notebook.ipynb").write_text(json.dumps(nb), encoding="utf-8")
    assert _trainability_smoke_check(state) is None
    # Single-point series is not a curve.
    (run_dir / "notebook.ipynb").write_text(
        _nb_with_outputs([0.05]), encoding="utf-8")
    assert _trainability_smoke_check(state) is None


def test_stage_3c_silent_non_learning_enters_fix_loop_and_recovers(
    fake_dispatch, fake_subprocess, run_dir
):
    """Smoke exits 0 but the executed outputs never beat chance → the
    synthetic failure routes through diagnostician → producer fix →
    re-render (writes a learning notebook) → smoke re-run → completed."""
    from run_pipeline import run_stage_3c
    state = make_state(run_dir)
    (run_dir / "notebook.ipynb").write_text(
        _nb_with_outputs([0.05, 0.087, 0.06]), encoding="utf-8")
    fake_subprocess.set_run_dir(run_dir)

    # iter 0: smoke passes mechanically; trainability pre-check fails.
    fake_subprocess.expect_script(returncode=0)
    # diagnostician gets the synthetic failure, targets the method-coder
    fake_dispatch.expect(
        agent="r2c-smoke-diagnostician",
        writes={".pipeline/smoke_diagnosis.json": _valid_smoke_diagnosis()},
    )
    fake_dispatch.expect(
        agent="r2c-method-coder",
        writes={"method/method.py": "# reshape fixed\n"},
    )
    # re-render writes a notebook whose executed outputs now learn
    fake_subprocess.expect_script(
        returncode=0,
        writes={"notebook.ipynb": _nb_with_outputs([0.15, 0.42, 0.88])},
    )
    fake_subprocess.expect_stage2_script(ok=True)  # validate_notebook_output
    # iter 1: smoke passes AND the curve beats chance.
    fake_subprocess.expect_script(returncode=0)

    result = run_stage_3c(state)

    assert_stage_completed(result, "stage_3c")
    assert_dispatched(fake_dispatch, "r2c-smoke-diagnostician", times=1)
    assert_dispatched(fake_dispatch, "r2c-method-coder", times=1)
    # The diagnostician prompt carried the synthetic failure, not a traceback.
    diag_call = next(c for c in fake_dispatch.calls
                     if c.agent == "r2c-smoke-diagnostician")
    assert "SILENT NON-LEARNING" in diag_call.prompt


# ---------------------------------------------------------------------------
# Post-smoke requirements reconcile (O2 gap, repo-map verdict 2026-07-04)
# ---------------------------------------------------------------------------


def _record_reconcile_calls(monkeypatch) -> list[str]:
    """Patch the driver-level reconcile helper to record its stage_id args."""
    import run_pipeline

    calls: list[str] = []

    def _recorder(state, stage_id="stage_3a"):
        calls.append(stage_id)

    monkeypatch.setattr(
        run_pipeline, "_reconcile_requirements_with_notebook", _recorder,
    )
    return calls


def test_stage_3c_completion_reconciles_requirements(
    monkeypatch, fake_dispatch, fake_subprocess, run_dir,
):
    """A clean smoke pass must re-run the requirements reconcile: the 3.c fix
    loop can rewrite the notebook (and add imports) after stage 3.a's
    reconcile already ran."""
    from run_pipeline import run_stage_3c
    state = make_state(run_dir)
    _seed_notebook(run_dir)
    calls = _record_reconcile_calls(monkeypatch)

    fake_subprocess.expect_script(returncode=0, stdout="ok", stderr="")

    result = run_stage_3c(state)
    assert_stage_completed(result, "stage_3c")
    assert calls == ["stage_3c"]


def test_stage_3c_environmental_degrade_reconciles_requirements(
    monkeypatch, fake_dispatch, fake_subprocess, run_dir,
):
    """An environmental degrade (exit 3) continues to delivery with the
    rendered notebook — its imports must be covered too."""
    from run_pipeline import run_stage_3c
    state = make_state(run_dir)
    _seed_notebook(run_dir)
    calls = _record_reconcile_calls(monkeypatch)

    fake_subprocess.expect_script(returncode=3, stderr="No module named 'nbclient'")

    result = run_stage_3c(state)
    assert_stage_degraded(result, stage_id="stage_3c")
    assert calls == ["stage_3c"]


def test_stage_3c_halt_does_not_reconcile(
    monkeypatch, fake_dispatch, fake_subprocess, run_dir,
):
    """A pipeline-fault halt (exit 2) stops the run — no reconcile."""
    from run_pipeline import run_stage_3c
    state = make_state(run_dir)
    _seed_notebook(run_dir)
    calls = _record_reconcile_calls(monkeypatch)

    fake_subprocess.expect_script(returncode=2, stderr="cannot parse notebook.ipynb")

    result = run_stage_3c(state)
    assert_stage_halted(result, stage_id="stage_3c",
                        reason_contains="smoke gate setup error")
    assert calls == []


def test_stage_3c_skip_does_not_reconcile(
    monkeypatch, fake_dispatch, fake_subprocess, run_dir,
):
    """A smart-skip means no upstream file changed since the sentinel — the
    notebook is byte-identical to the one stage 3.a already reconciled."""
    from run_pipeline import run_stage_3c
    state = make_state(run_dir)
    _seed_notebook(run_dir)
    calls = _record_reconcile_calls(monkeypatch)
    # Bootstrap skip path: sentinel present, no digest file yet.
    (run_dir / ".pipeline" / "stage_3c.complete").write_text("", encoding="utf-8")

    result = run_stage_3c(state)
    assert result.status == "skipped"
    assert calls == []


# ---------------------------------------------------------------------------
# Fix-loop third option (queue item 7): the adversarial fixture set. The
# deterministic G1-G5 gates may convert THE single failing component to a
# stub at a terminal give-up point. Fixture A (fixable-bug shape, one target,
# no owned frames) is pinned by the existing cap-exhaustion tests above —
# gates unmet leaves the degrade byte-identical. Below: fixture B (must
# fire), fixture C (G4 plausibility mismatch must HALT), and the re-gate
# rollback.
# ---------------------------------------------------------------------------


def _smoke_stderr_isolated(cell: int = 0,
                           frame_file: str = "method/training.py") -> str:
    """A smoke failure whose traceback isolates to one owned file (G3)."""
    return (
        f"notebook execution failed at cell {cell}\n\n"
        f"--- failing cell source (code) ---\nprint('hello')\n\n"
        f"--- error traceback ---\n"
        f"File {frame_file}:12, in run_component\n"
        f"RuntimeError: boom\n"
    )


def _seed_stub_conversion_tree(run_dir):
    """Draft with the failing cell (cell 0 of the seeded notebook) plus the
    supporting component file the failures isolate to."""
    (run_dir / "notebook_draft.py").write_text(
        "# %% [markdown]\n# # T\n\n"
        "# %%\nprint('cell 0')\n\n"
        "# %%\nprint('cell 1')\n",
        encoding="utf-8")
    (run_dir / "method").mkdir(exist_ok=True)
    (run_dir / "method" / "training.py").write_text(
        "def run_component(x):\n    return x\n", encoding="utf-8")


def _queue_isolated_failed_iter(fake_dispatch, fake_subprocess, *,
                                iter_index: int, target_agent: str,
                                target_file: str,
                                root_cause: str = "Uncatalogued bug."):
    """One failure iteration whose traceback isolates to method/training.py
    while the diagnosis targets `target_file` (G2 breadth comes from varying
    the target across iterations)."""
    fake_subprocess.expect_script(
        returncode=1, stderr=_smoke_stderr_isolated(cell=iter_index))
    fake_dispatch.expect(
        agent="r2c-smoke-diagnostician",
        writes={".pipeline/smoke_diagnosis.json": _valid_smoke_diagnosis(
            target_agent, target_file, root_cause=root_cause)})
    fake_dispatch.expect(agent=target_agent,
                         writes={target_file: f"# iter {iter_index}\n"})
    _queue_post_fix_render_and_validate(fake_subprocess)


def _queue_exhaustion_to_judge_decline(fake_dispatch, fake_subprocess, *,
                                       last_root_cause="Uncatalogued bug."):
    """Drive the loop to give-up point 1: cap iterations with two distinct
    targets and isolated frames, cap smoke failure, judge declines."""
    from run_pipeline import STAGE_2_RETRY_CAP
    targets = [("r2c-architecture-coder", "method/training.py"),
               ("r2c-method-coder", "method/method.py"),
               ("r2c-architecture-coder", "method/training.py")]
    for i in range(STAGE_2_RETRY_CAP):
        agent, tf = targets[i % len(targets)]
        _queue_isolated_failed_iter(
            fake_dispatch, fake_subprocess, iter_index=i,
            target_agent=agent, target_file=tf,
            root_cause=last_root_cause if i == STAGE_2_RETRY_CAP - 1
            else "Uncatalogued bug.")
    # Cap iteration: smoke fails once more at cell 0, frames still isolated.
    fake_subprocess.expect_script(
        returncode=1, stderr=_smoke_stderr_isolated(cell=0))
    queue_judge_decision_object(fake_dispatch, {
        "schema_version": "1.0.0",
        "stage_id": "stage_3c",
        "iteration": STAGE_2_RETRY_CAP,
        "validator_label": f"smoke_loop_cap_exhausted:cap={STAGE_2_RETRY_CAP}",
        "classification": "upstream_issue",
        "action": "halt",
        "target_agent": None,
        "finding": None,
        "rationale": "no producer fix left to try",
        "confidence": "high",
        "files_examined": ["method/training.py"],
    })


def test_fix_loop_stub_conversion_fires_and_delivers_partial(
    fake_dispatch, fake_subprocess, run_dir
):
    """Fixture B: exhaustion-shaped, MUST fire. Two distinct targets, every
    frame isolated to method/training.py, uncataloged diagnosis, no prior
    stub. The component ships as a raising stub, the draft carries the
    criterion-4 marker, the smoke re-gate passes, and the stage completes
    toward a PARTIAL delivery."""
    from run_pipeline import run_stage_3c
    state = make_state(run_dir)
    _seed_notebook(run_dir)
    _seed_stub_conversion_tree(run_dir)
    _queue_exhaustion_to_judge_decline(fake_dispatch, fake_subprocess)
    # Conversion: re-render + re-validate + smoke re-gate, all clean.
    fake_subprocess.expect_script(returncode=0)   # render_notebook.py
    fake_subprocess.expect_stage2_script(ok=True)  # validate_notebook_output
    fake_subprocess.expect_script(returncode=0)   # smoke re-gate

    result = run_stage_3c(state)

    assert_stage_completed(result, "stage_3c")
    assert "PARTIAL" in result.notes and "method-training" in result.notes
    # The stub record, work order, stub module, and draft marker all landed.
    records = json.loads(
        (run_dir / ".pipeline" / "stubbed_elements.json").read_text())
    [rec] = records["stubbed_elements"]
    assert rec["element_id"] == "method-training"
    assert rec["role"] == "supporting"
    assert rec["stub_path"] == "method/training.py"
    wo = (run_dir / "work_orders" / "method-training.md").read_text()
    assert "PARTIAL" in wo
    stub_src = (run_dir / "method" / "training.py").read_text()
    assert "NotImplementedError" in stub_src and "run_component" not in stub_src
    draft = (run_dir / "notebook_draft.py").read_text()
    assert "# %% PLACEHOLDER: component_stub:method-training" in draft
    assert "print('cell 0')" not in draft
    # Loud and honest: the run event and the assumptions entry both landed.
    events = (run_dir / ".pipeline" / "run_events.jsonl").read_text()
    assert "fix_loop_stub_conversion" in events
    assumptions = (run_dir / "details" / "assumptions.md").read_text()
    assert "method-training" in assumptions and "PARTIAL" in assumptions


def test_fix_loop_stub_g4_mismatch_halts_with_screen_finding(
    fake_dispatch, fake_subprocess, run_dir
):
    """Fixture C: fixture B's shape except the recorded diagnosis says
    pathological complexity while the failing cell is a bare print/plot
    cell. G4 HALTS: the diagnosis chain is broken, stubbing would hide a
    machinery bug. No stub record is written."""
    from run_pipeline import run_stage_3c
    state = make_state(run_dir)
    _seed_notebook(run_dir)
    _seed_stub_conversion_tree(run_dir)
    _queue_exhaustion_to_judge_decline(
        fake_dispatch, fake_subprocess,
        last_root_cause="Pathological complexity: an infinite loop burns "
                        "the cell budget.")

    result = run_stage_3c(state)

    assert_stage_halted(result, stage_id="stage_3c",
                        reason_contains="plausibility screen")
    assert not (run_dir / ".pipeline" / "stubbed_elements.json").exists()
    # The screen's verdict is in the halt context for the machinery
    # investigation.
    assert any("G4 MISMATCH" in r
               for r in result.halt_artifact["context"]["stub_gate_verdict"])


def test_fix_loop_stub_regate_failure_rolls_back_to_todays_degrade(
    fake_dispatch, fake_subprocess, run_dir
):
    """The mandatory re-gate failing rolls the conversion back completely:
    component restored byte-identical, no record, no work order, and the
    stage degrades exactly as today."""
    from run_pipeline import run_stage_3c
    state = make_state(run_dir)
    _seed_notebook(run_dir)
    _seed_stub_conversion_tree(run_dir)
    _queue_exhaustion_to_judge_decline(fake_dispatch, fake_subprocess)
    fake_subprocess.expect_script(returncode=0)   # render_notebook.py
    fake_subprocess.expect_stage2_script(ok=True)  # validate_notebook_output
    fake_subprocess.expect_script(
        returncode=1, stderr=_smoke_stderr_isolated(cell=1))  # re-gate FAILS
    fake_subprocess.expect_script(returncode=0)   # post-rollback re-render

    result = run_stage_3c(state)

    assert_stage_degraded(result, stage_id="stage_3c")
    # Restored to the pre-conversion state: the LAST fix dispatch's content
    # (iterations legitimately rewrote the component), never the stub.
    restored = (run_dir / "method" / "training.py").read_text()
    assert restored == "# iter 2\n"
    assert "NOT IMPLEMENTED" not in restored
    assert not (run_dir / ".pipeline" / "stubbed_elements.json").exists()
    assert not (run_dir / "work_orders" / "method-training.md").exists()
    draft = (run_dir / "notebook_draft.py").read_text()
    assert "print('cell 0')" in draft and "PLACEHOLDER" not in draft
    events = (run_dir / ".pipeline" / "run_events.jsonl").read_text()
    assert "fix_loop_stub_regate_failed" in events


def test_fix_loop_stub_never_fires_on_single_target_exhaustion(
    fake_dispatch, fake_subprocess, run_dir
):
    """Fixture A sharpened: isolated frames but ONE fix target retried —
    G2 refuses (fixation, not unfixability) and the degrade ships exactly
    as before, with the gate verdict on the event log."""
    from run_pipeline import STAGE_2_RETRY_CAP, run_stage_3c
    state = make_state(run_dir)
    _seed_notebook(run_dir)
    _seed_stub_conversion_tree(run_dir)
    for i in range(STAGE_2_RETRY_CAP):
        _queue_isolated_failed_iter(
            fake_dispatch, fake_subprocess, iter_index=i,
            target_agent="r2c-architecture-coder",
            target_file="method/training.py")
    fake_subprocess.expect_script(
        returncode=1, stderr=_smoke_stderr_isolated(cell=0))
    queue_judge_decision_object(fake_dispatch, {
        "schema_version": "1.0.0",
        "stage_id": "stage_3c",
        "iteration": STAGE_2_RETRY_CAP,
        "validator_label": f"smoke_loop_cap_exhausted:cap={STAGE_2_RETRY_CAP}",
        "classification": "upstream_issue",
        "action": "halt", "target_agent": None, "finding": None,
        "rationale": "same target keeps failing", "confidence": "high",
        "files_examined": ["method/training.py"],
    })

    result = run_stage_3c(state)

    assert_stage_degraded(result, stage_id="stage_3c")
    assert not (run_dir / ".pipeline" / "stubbed_elements.json").exists()
    # The fix dispatches legitimately rewrote the file; the stub machinery
    # never touched it.
    stub_src = (run_dir / "method" / "training.py").read_text()
    assert "NOT IMPLEMENTED" not in stub_src
    events = (run_dir / ".pipeline" / "run_events.jsonl").read_text()
    assert "fix_loop_stub_gates_unmet" in events


# ---------------------------------------------------------------------------
# Item 20 — smoke-cell timeout is a first-class failure kind
# (bev-distill overnight 07-08: a timeout masqueraded as an error on every
# surface — KNOWN_ISSUES promised a nonexistent traceback, and the fix loop
# asked for a bug fix on a cell whose only defect was cost)
# ---------------------------------------------------------------------------


def _smoke_stderr_cell_timeout(cell: int = 0, budget: int = 180,
                               wall: float = 186.2) -> str:
    """The smoke runner's genuine-timeout report, verbatim shape."""
    return (
        f"notebook execution TIMED OUT at cell {cell} "
        f"after {budget}s (wall {wall}s).\n"
        f"This cell ran longer than the per-cell budget — either:\n"
        f"  (a) the cell legitimately needs more time on this hardware "
        f"(re-run with `--timeout <larger>`), or\n"
        f"  (b) the cell has an infinite loop / pathological complexity "
        f"(inspect the source and route to the appropriate producer for fix).\n\n"
        f"--- timed-out cell source ---\n"
        f"for epoch in range(3):\n    train_one_epoch()\n"
    )


def test_smoke_timeout_facts_parses_runner_report():
    from run_pipeline import _smoke_timeout_facts

    facts = _smoke_timeout_facts(_smoke_stderr_cell_timeout(44, 180, 186.2))
    assert facts == {"cell": 44, "budget_s": 180, "wall_s": 186.2}
    # An error failure is not a timeout.
    assert _smoke_timeout_facts(_smoke_stderr_cell_fails(44)) is None
    assert _smoke_timeout_facts("") is None


def test_smoke_degrade_fields_timeout_names_budget_never_traceback():
    from run_pipeline import _smoke_degrade_fields

    what_failed, where, what_to_do = _smoke_degrade_fields(
        _smoke_stderr_cell_timeout(44, 180, 186.2), 44, 5)
    assert what_failed == \
        "the notebook does not finish within the smoke time budget"
    assert "180s per-cell budget" in where and "186s" in where
    blob = " ".join([what_failed, where, what_to_do]).lower()
    # No traceback PROMISE and no error language; the honest "there is no
    # traceback" disclosure is the point.
    assert "traceback is in" not in blob
    assert "there is no traceback" in blob
    assert "raises" not in blob
    assert "fewer epochs" in what_to_do

    # The error kind keeps the existing wording, traceback promise included.
    what_failed, where, what_to_do = _smoke_degrade_fields(
        _smoke_stderr_cell_fails(3), 3, 2)
    assert what_failed == "the notebook does not run end-to-end"
    assert "traceback" in what_to_do.lower()


def test_stage_3c_timeout_fix_dispatch_names_the_class(
    fake_dispatch, fake_subprocess, run_dir
):
    """The producer fix dispatch for a timed-out cell must say it is a
    timeout and constrain the ask to work reduction — not send the producer
    bug hunting."""
    from run_pipeline import run_stage_3c
    state = make_state(run_dir)
    _seed_notebook(run_dir)

    fake_subprocess.expect_script(
        returncode=1, stderr=_smoke_stderr_cell_timeout(0, 180, 186.2))
    fake_dispatch.expect(
        agent="r2c-smoke-diagnostician",
        writes={".pipeline/smoke_diagnosis.json": _valid_smoke_diagnosis(
            target_agent="r2c-notebook-generator",
            target_file=".pipeline/notebook_draft.py",
        )},
    )
    fake_dispatch.expect(
        agent="r2c-notebook-generator",
        writes={".pipeline/notebook_draft.py": "# fewer epochs\n"},
    )
    _queue_post_fix_render_and_validate(fake_subprocess)
    fake_subprocess.expect_script(returncode=0)  # smoke passes

    result = run_stage_3c(state)
    assert_stage_completed(result, "stage_3c")

    fix_prompts = [c.prompt for c in fake_dispatch.calls
                   if c.agent == "r2c-notebook-generator"]
    assert len(fix_prompts) == 1
    assert "THIS IS A TIMEOUT, NOT AN ERROR" in fix_prompts[0]
    assert "fewer epochs" in fix_prompts[0]
    assert "did NOT raise" in fix_prompts[0]


def test_stage_3c_repeated_timeout_is_fixation_not_exception_none(
    fake_dispatch, fake_subprocess, run_dir, capsys
):
    """A second timeout at the same cell means the fix did not reduce the
    cell's work: the prior target is burned (and the log names the timeout
    class instead of 'exception unchanged: None'). The diagnostician
    re-picking the burned target then trips the forbidden-target retry."""
    from run_pipeline import run_stage_3c
    state = make_state(run_dir)
    _seed_notebook(run_dir)

    # iter 0: cell 0 times out; fix dispatched to the notebook draft.
    fake_subprocess.expect_script(
        returncode=1, stderr=_smoke_stderr_cell_timeout(0))
    fake_dispatch.expect(
        agent="r2c-smoke-diagnostician",
        writes={".pipeline/smoke_diagnosis.json": _valid_smoke_diagnosis(
            target_agent="r2c-notebook-generator",
            target_file=".pipeline/notebook_draft.py",
        )},
    )
    fake_dispatch.expect(
        agent="r2c-notebook-generator",
        writes={".pipeline/notebook_draft.py": "# logic-only 'fix'\n"},
    )
    _queue_post_fix_render_and_validate(fake_subprocess)

    # iter 1: SAME cell times out again → fixation; the diagnostician
    # picks the burned target and gets the forbidden-target retry, then
    # picks a different target which is dispatched.
    fake_subprocess.expect_script(
        returncode=1, stderr=_smoke_stderr_cell_timeout(0))
    fake_dispatch.expect(
        agent="r2c-smoke-diagnostician",
        writes={".pipeline/smoke_diagnosis.json": _valid_smoke_diagnosis(
            target_agent="r2c-notebook-generator",
            target_file=".pipeline/notebook_draft.py",
        )},
    )
    fake_dispatch.expect(
        agent="r2c-smoke-diagnostician",
        writes={".pipeline/smoke_diagnosis.json": _valid_smoke_diagnosis(
            target_agent="r2c-method-coder",
            target_file="method/method.py",
        )},
    )
    fake_dispatch.expect(
        agent="r2c-method-coder",
        writes={"method/method.py": "# cheaper work\n"},
    )
    _queue_post_fix_render_and_validate(fake_subprocess)

    # iter 2: smoke passes.
    fake_subprocess.expect_script(returncode=0)

    result = run_stage_3c(state)
    assert_stage_completed(result, "stage_3c")

    out = capsys.readouterr().out
    assert "did not reduce the cell's work" in out
    assert "exception unchanged: None" not in out


# ---------------------------------------------------------------------------
# Accepted-validation baseline (the ICRA 2026-07-13 re-litigated-degrade
# halt): stage 3a degrades past static validation failures by design, so
# later revalidations judge a fix on whether it made the validation state
# WORSE — never on findings the run already accepted. Live shape: a correct
# smoke fix (torch.rad2deg) landed, then the post-fix revalidation halted
# the run on the pre-existing private-import failure 3a had degraded past.
# ---------------------------------------------------------------------------


ACCEPTED_BULLET = ("notebook imports `from method.model import "
                   "['_AvoidanceClassifier']` but those names are not "
                   "defined at top-level in method/model.py")
NEW_BULLET = "notebook section §3 is missing its required markdown header"


def _validator_fail_stderr(*bullets: str) -> str:
    return (f"FAIL: {len(bullets)} validation error(s):\n"
            + "".join(f"  - {b}\n" for b in bullets))


def _seed_accepted_baseline(run_dir: Path, bullets: list[str]) -> None:
    (run_dir / ".pipeline" / "notebook_validation_accepted.json").write_text(
        json.dumps({
            "schema_version": "1.0",
            "validator": "validate_notebook_output.py",
            "recorded_at_stage": "stage_3a",
            "accepted_failures": [" ".join(b.split()) for b in bullets],
        }), encoding="utf-8")


def test_failure_bullets_parse_and_normalize():
    from run_pipeline import _notebook_validation_failure_bullets
    stderr = ("noise line\nFAIL: 2 validation error(s):\n"
              "  - first   failure\twith  odd spacing\n"
              "  - second failure\n")
    assert _notebook_validation_failure_bullets(stderr) == [
        "first failure with odd spacing", "second failure"]
    assert _notebook_validation_failure_bullets("") == []


def test_new_failures_without_baseline_are_all_new(run_dir):
    from run_pipeline import _notebook_validation_new_failures
    state = make_state(run_dir)
    stderr = _validator_fail_stderr(ACCEPTED_BULLET)
    assert _notebook_validation_new_failures(state, stderr) == \
        [" ".join(ACCEPTED_BULLET.split())]


def test_new_failures_subtract_the_accepted_baseline(run_dir):
    from run_pipeline import _notebook_validation_new_failures
    state = make_state(run_dir)
    _seed_accepted_baseline(run_dir, [ACCEPTED_BULLET])
    only_accepted = _validator_fail_stderr(ACCEPTED_BULLET)
    assert _notebook_validation_new_failures(state, only_accepted) == []
    mixed = _validator_fail_stderr(ACCEPTED_BULLET, NEW_BULLET)
    assert _notebook_validation_new_failures(state, mixed) == [NEW_BULLET]


def test_unparseable_validator_failure_always_halts(run_dir):
    from run_pipeline import _notebook_validation_new_failures
    state = make_state(run_dir)
    _seed_accepted_baseline(run_dir, [ACCEPTED_BULLET])
    # No bullets to compare: the failure must stay halt-worthy.
    assert _notebook_validation_new_failures(
        state, "error: spec not found: /nope/spec.json") != []
    # A corrupt baseline file degrades to all-new, never to all-accepted.
    (run_dir / ".pipeline" / "notebook_validation_accepted.json").write_text(
        "{corrupt", encoding="utf-8")
    assert _notebook_validation_new_failures(
        state, _validator_fail_stderr(ACCEPTED_BULLET)) != []


def test_record_accepted_baseline_writes_failing_bullets(
        run_dir, monkeypatch):
    import run_pipeline as rp
    state = make_state(run_dir)

    class _Proc:
        returncode = 1
        stdout = ""
        stderr = _validator_fail_stderr(ACCEPTED_BULLET)

    monkeypatch.setattr(rp, "run_script", lambda *a, **k: _Proc())
    rp._record_accepted_notebook_validation(state)
    recorded = json.loads(
        (run_dir / ".pipeline" / "notebook_validation_accepted.json")
        .read_text(encoding="utf-8"))
    assert recorded["accepted_failures"] == [" ".join(ACCEPTED_BULLET.split())]
    events = [json.loads(line) for line in
              (run_dir / ".pipeline" / "run_events.jsonl")
              .read_text().splitlines()]
    assert any(e["event_type"] == "notebook_validation_baseline_recorded"
               for e in events)

    # A passing validator records an empty baseline: the degrade was for
    # reviewer-side reasons and every later failure stays halt-worthy.
    class _OkProc:
        returncode = 0
        stdout = "ok"
        stderr = ""

    monkeypatch.setattr(rp, "run_script", lambda *a, **k: _OkProc())
    rp._record_accepted_notebook_validation(state)
    recorded = json.loads(
        (run_dir / ".pipeline" / "notebook_validation_accepted.json")
        .read_text(encoding="utf-8"))
    assert recorded["accepted_failures"] == []


def test_post_fix_revalidation_continues_on_accepted_failures_only(
        fake_dispatch, fake_subprocess, run_dir):
    """The ICRA replay: smoke fix lands, the post-fix static validator
    still fails on the 3a-accepted failure ONLY, and the run continues to
    the next smoke attempt instead of halting."""
    from run_pipeline import run_stage_3c
    state = make_state(run_dir)
    _seed_notebook(run_dir)
    _seed_accepted_baseline(run_dir, [ACCEPTED_BULLET])

    # iter 0: smoke fails → diagnosis → fix dispatch
    fake_subprocess.expect_script(returncode=1,
                                  stderr=_smoke_stderr_cell_fails(0))
    fake_dispatch.expect(
        agent="r2c-smoke-diagnostician",
        writes={".pipeline/smoke_diagnosis.json": _valid_smoke_diagnosis()},
    )
    fake_dispatch.expect(
        agent="r2c-method-coder",
        writes={"method/method.py": "# fixed method.py\n"},
    )
    # post-fix: render ok, static validator fails on the ACCEPTED bullet
    fake_subprocess.expect_script(returncode=0)  # render_notebook.py
    fake_subprocess.expect_stage2_script(
        ok=False, err=_validator_fail_stderr(ACCEPTED_BULLET))

    # iter 1: smoke passes → stage completes
    fake_subprocess.expect_script(returncode=0)

    result = run_stage_3c(state)
    assert_stage_completed(result, "stage_3c")
    events = [json.loads(line) for line in
              (run_dir / ".pipeline" / "run_events.jsonl")
              .read_text().splitlines()]
    assert any(e["event_type"] == "accepted_validation_failures_only"
               for e in events)


def test_post_fix_revalidation_still_halts_on_a_new_failure(
        fake_dispatch, fake_subprocess, run_dir):
    """A fix that makes the static validation state WORSE still halts, and
    the halt names only the new failure."""
    from run_pipeline import run_stage_3c
    state = make_state(run_dir)
    _seed_notebook(run_dir)
    _seed_accepted_baseline(run_dir, [ACCEPTED_BULLET])

    fake_subprocess.expect_script(returncode=1,
                                  stderr=_smoke_stderr_cell_fails(0))
    fake_dispatch.expect(
        agent="r2c-smoke-diagnostician",
        writes={".pipeline/smoke_diagnosis.json": _valid_smoke_diagnosis()},
    )
    fake_dispatch.expect(
        agent="r2c-method-coder",
        writes={"method/method.py": "# fixed method.py\n"},
    )
    fake_subprocess.expect_script(returncode=0)  # render_notebook.py
    fake_subprocess.expect_stage2_script(
        ok=False, err=_validator_fail_stderr(ACCEPTED_BULLET, NEW_BULLET))

    result = run_stage_3c(state)
    assert result.status == "halted"
    assert "not in the stage-3a accepted baseline" in \
        result.halt_artifact["reason"]
    assert result.halt_artifact["context"]["new_failures"] == [NEW_BULLET]
