"""B-07 step 2: dispatch_templates' self-test body, moved verbatim.

The module had ZERO self-patching sites, so the 183 asserts run unchanged
here. The module's __main__ arm is repointed at this file (never deleted:
a silently-empty `python3 scripts/dispatch_templates.py` exiting 0 is the
vacuous green the W1 verdict warned about).
"""

from __future__ import annotations

import json  # noqa: F401 - the verbatim body uses it

from dispatch_templates import (
    CAP_BURN_CORRECTIVE_RESUME_NUDGE,
    DispatchPaths,
    FILE_WRITING_DISCIPLINE,
    FIX_MODE_BODY,
    INCREMENTAL_REREVIEW_TEMPLATE,
    JUDGE_TASK_TEMPLATE,
    PRODUCER_ANTI_REVIEW_CLAUSE,
    SCOPE_CONTRACT,
    STAGE_1_OUTPUT_MODE_WRITEABLE_PATHS,
    STAGE_REVIEW_TASK_TEMPLATE,
    STAGE_TASK_SUMMARIES,
    WRITEABLE_PATHS,
    _writes_only_review_artifacts,
    build_corrective_redispatch_preamble,
    build_dispatch_prompt,
    build_fix_mode_prompt,
    build_halt_artifact,
    build_insight_review_prompt,
    build_smoke_diagnosis_prompt,
    build_write_first_retry_preamble,
    format_symbol_ownership_block,
    format_writeable_paths_block,
)


def test_dispatch_templates_self_test():
    sample_paths = DispatchPaths(
        spec="/tmp/run/.pipeline/method_spec.json",
        paper="/tmp/run/.pipeline/paper.md",
        paper_map="/tmp/run/.pipeline/paper_map.json",
        run_dir="/tmp/run",
        taxonomy_source="taxonomy:knowledge_distillation",
    )

    # SCOPE_CONTRACT: header + generic phrases only (B-18 item B dropped the
    # concrete AL baseline list from the parenthetical, 2026-08-20)
    assert SCOPE_CONTRACT.startswith("## SCOPE CONTRACT (v2 — read first)"), "scope contract header drift"
    assert "comparison.standard_baselines" in SCOPE_CONTRACT
    assert "Notebook §5 runs ONE method" in SCOPE_CONTRACT
    assert "implement baselines" in SCOPE_CONTRACT
    assert "extend `__all__` to include baseline functions" in SCOPE_CONTRACT

    # FIX_MODE_BODY: contains (a)–(e) skeleton, formattable
    body = FIX_MODE_BODY.format(target_agent="method-coder")
    assert "method-coder agent in fix mode" in body
    for letter in "(a) (b) (c) (d) (e)".split():
        assert letter in body, f"missing fix-mode step {letter}"
    assert "{target_agent}" not in body, "format placeholder leaked"
    # No-source-reads rule (agent dispatch optimization 2026-09-01): fix
    # dispatches must not reverse-engineer validators from pipeline source.
    assert "Pipeline source is OFF-LIMITS" in body
    assert "complete acceptance rule" in body

    # Per-stage summaries present
    for stage_id in ["stage_1_decomposer", "stage_1_analyzer",
                     "stage_2b_architecture", "stage_2c_method",
                     "stage_3a_notebook", "stage_4_review"]:
        assert stage_id in STAGE_TASK_SUMMARIES, f"missing summary for {stage_id}"
        assert len(STAGE_TASK_SUMMARIES[stage_id]) > 50, f"{stage_id} summary suspiciously short"

    # Stage 1.a/1.b reflect the architectural split: analyzer no longer
    # produces paper_map.json (that's the decomposer's job); analyzer's
    # key_elements must reference paper_map IDs. The Stage 1.b prompt
    # foregrounds the write action because the Think-class analyzer LLM was
    # observed to finish its thinking turn without invoking the Write tool
    # when the task framing leaned analytical — see the GBALD run log
    # (stage_1.halt at 11:43, transcript_gbald.md line 4024).
    analyzer_summary = STAGE_TASK_SUMMARIES["stage_1_analyzer"]
    assert "method_spec.json" in analyzer_summary
    assert "produce paper_map" not in analyzer_summary, "analyzer no longer produces paper_map"
    assert "USE YOUR WRITE TOOL" in analyzer_summary, "Stage 1.b prompt must foreground the write action"
    assert "paper_map element IDs" in analyzer_summary
    decomposer_summary = STAGE_TASK_SUMMARIES["stage_1_decomposer"]
    assert "paper_map.json" in decomposer_summary
    assert "Stage 1.a" in decomposer_summary
    # Same Think-model "stop early after thinking" pattern as the analyzer:
    # the prompt must foreground the write action. Observed in bev-distill's
    # first decomposer dispatch — 15.7s with no file produced.
    assert "USE YOUR WRITE TOOL" in decomposer_summary, (
        "Stage 1.a prompt must foreground the write action (Think-model "
        "stop-early failure mode)"
    )

    # Stage-review template — must foreground the write action because
    # r2c-stage-reviewer runs on a Think-class model (the Think endpoint); the
    # bev-distill Phase 7 run halted at Stage 2.c because the reviewer
    # produced a thinking trace and stopped without writing the JSON.
    # Same Think-model stop-early pattern as the analyzer (Phase 3) and
    # decomposer (Phase 5); same fix.
    s = STAGE_REVIEW_TASK_TEMPLATE.format(stage_id="stage_2c_method")
    assert "stage_2c_method" in s
    assert "{stage_id}" not in s
    assert "USE YOUR WRITE TOOL EXACTLY ONCE" in s, "Stage-review template must enforce single-write"
    assert "After that file is written, return" in s, "Stage-review template must enforce return-after-write"
    # Procedural anchoring against drift into producer behavior (gbald 2026-05-14
    # case: reviewer wrote stage_review_stage_2b JSON correctly, then continued
    # to implement method.py + __init__.py + a stage_2c review of its own work).
    assert "implement, edit, or create source code" in s, "Stage-review template must forbid source-file writes"
    assert "not the producer" in s, "Stage-review template must explicitly disclaim producer identity"
    assert "review JSONs for stages other than" in s, "Stage-review template must forbid cross-stage review writes"
    # Equation-verification baseline (R2C 2026-05-22): for stage_2c_method, the
    # reviewer MUST run the symbol-by-symbol equation-verification baseline AFTER
    # the taxonomy semantic_checks. Bev-distill C1 (quality-score no-op via
    # scalar-vs-vector aggregation order) is the canonical failure mode.
    assert "Equation-verification reminder (stage_2c_method ONLY)" in s, (
        "Stage-review template must include the equation-verification baseline "
        "reminder for stage_2c_method"
    )
    assert "paper-element: eq-" in s, (
        "Stage-review template must reference the paper-element comment anchor"
    )
    assert "paper_map.json" in s, (
        "Stage-review template must reference paper_map.json as the equation lookup source"
    )
    assert "bev-distill 2026-05-22" in s, (
        "Stage-review template must name the canonical bev-distill failure mode"
    )
    # Symmetry check: when stage_id is NOT stage_2c_method, the reminder text
    # still appears in the template (since the template is shared), but it's
    # explicitly scoped via 'If {stage_id} == `stage_2c_method`'.
    s_other = STAGE_REVIEW_TASK_TEMPLATE.format(stage_id="stage_2b_architecture")
    assert "If stage_2b_architecture == `stage_2c_method`" in s_other, (
        "The equation-verification reminder must be explicitly scoped; the "
        "template substitution should make it a no-op for other stages"
    )
    assert "For any other stage_id, this baseline does NOT apply" in s_other, (
        "The reviewer must be told the baseline is stage_2c-specific"
    )
    assert "paper-truth structured fields" in s
    assert "not the demo/default value" in s

    assert "MethodSpec methodology approximations are keyed by element status" in body
    assert "replication_feasibility.{verdict,approved_approximations}" in body

    judge_prompt = JUDGE_TASK_TEMPLATE.format(
        stage_id="stage_1",
        iteration=0,
        validator_label="validate_method_spec.py --strict",
        decision_output_path="/tmp/judge.json",
        stderr_tail="replication_feasibility.approved_approximations mismatch",
        prior_decisions_block="[]",
    )
    assert "MethodSpec methodology-fidelity fields" in judge_prompt
    assert "Any dispatch_fix finding must name every dependent site" in judge_prompt

    # Incremental re-review template
    s = INCREMENTAL_REREVIEW_TEMPLATE.format(iteration=2, prior_finding_ids="F001, F002")
    assert "iteration 2" in s
    assert "F001, F002" in s

    # build_dispatch_prompt assembly
    prompt = build_dispatch_prompt(
        task_summary=STAGE_TASK_SUMMARIES["stage_2c_method"],
        paths=sample_paths,
    )
    assert prompt.startswith("## SCOPE CONTRACT")
    assert "Stage 2.c — produce method.py" in prompt
    assert "method_spec.json" in prompt
    assert prompt.endswith("Do NOT run the render script or any validator.")
    # Without `writeable_paths`, the prompt must NOT include the block.
    assert "Writeable paths" not in prompt, "writeable_paths block leaked without opt-in"

    # build_dispatch_prompt with writeable_paths
    prompt_with_scope = build_dispatch_prompt(
        task_summary=STAGE_TASK_SUMMARIES["stage_2c_method"],
        paths=sample_paths,
        writeable_paths=WRITEABLE_PATHS["r2c-method-coder"],
    )
    assert "**Writeable paths**" in prompt_with_scope
    assert "method/method.py" in prompt_with_scope
    assert "contract violation" in prompt_with_scope, "must warn agent the driver halts on out-of-scope writes"
    assert prompt_with_scope.endswith("Do NOT run the render script or any validator.")

    # B-004 (pdwa 2026-05-27): the producer writeable block warns producers that
    # the taxonomy stage_review_focus is the reviewer's contract, not
    # theirs, and that producing their listed file is their whole job. That
    # clause is PRODUCER-framed and must NOT reach a review-only agent — sent to
    # the stage_2x reviewer it inverted the task and drove it to author a
    # notebook.py (deep-batch run, 2026-06-15 halt). Producers keep it.
    producer_block = format_writeable_paths_block(WRITEABLE_PATHS["r2c-method-coder"])
    assert "stage_review_focus" in producer_block
    assert "stage_review_*.json" in producer_block, "must name the review artifact it must not write"
    assert "your entire job" in producer_block, "producer must keep the anti-review clause"
    # Review-only agents get the scope constraint WITHOUT the producer-framed
    # anti-review clause (which would tell them not to review — their job).
    for review_agent in ("r2c-stage-reviewer", "r2c-paper-fidelity-reviewer"):
        rb = format_writeable_paths_block(WRITEABLE_PATHS[review_agent])
        assert "You MUST only create or modify files matching the patterns" in rb, (
            f"{review_agent} lost the generic scope constraint")
        assert WRITEABLE_PATHS[review_agent][0] in rb, f"{review_agent} must name its own output"
        assert "your entire job" not in rb, f"{review_agent} got the producer anti-review clause"
        assert "reviewing them is" not in rb, f"{review_agent} got producer-framed review language"

    # Without think_anchor_output_path, the Think-class anchor block must NOT leak.
    assert "Think-class procedural anchor" not in prompt_with_scope, (
        "THINK_CLASS_ANCHOR must be opt-in via think_anchor_output_path"
    )

    # build_dispatch_prompt with think_anchor_output_path — exercises the
    # shared procedural-anchor block (P1). Catches the after-write drift
    # pattern that surfaced in three Think-class agents.
    prompt_with_anchor = build_dispatch_prompt(
        task_summary=STAGE_TASK_SUMMARIES["stage_1_decomposer"],
        paths=sample_paths,
        writeable_paths=WRITEABLE_PATHS["r2c-decomposer"],
        think_anchor_output_path="/tmp/run/.pipeline/paper_map.json",
    )
    assert "Think-class procedural anchor" in prompt_with_anchor
    assert "/tmp/run/.pipeline/paper_map.json" in prompt_with_anchor
    assert "TodoWrite" in prompt_with_anchor
    assert "OUTPUT MODE: CANONICAL ONLY" in prompt_with_anchor
    # Block lists the three observed drift patterns by name so the agent
    # has concrete anti-examples.
    assert "diagnostician" in prompt_with_anchor
    assert "stage-reviewer" in prompt_with_anchor
    assert "build agent" in prompt_with_anchor
    # Ordering: SCOPE_CONTRACT → task → THINK_CLASS_ANCHOR → paths → writeable → closing
    assert prompt_with_anchor.index("Think-class procedural anchor") > prompt_with_anchor.index("Stage 1.a")
    assert prompt_with_anchor.index("Think-class procedural anchor") < prompt_with_anchor.index("**Run paths:**")

    # Tool-call-first writing discipline (cap-burn fix 1): opt-in only.
    # Without the flag the assembled prompt is unchanged — reviewer, judge,
    # and Think-class dispatches must not pick up the coder discipline.
    assert "File-writing discipline" not in prompt
    assert "File-writing discipline" not in prompt_with_scope
    assert "File-writing discipline" not in prompt_with_anchor
    prompt_disciplined = build_dispatch_prompt(
        task_summary=STAGE_TASK_SUMMARIES["stage_2c_method"],
        paths=sample_paths,
        writeable_paths=WRITEABLE_PATHS["r2c-method-coder"],
        writing_discipline=True,
    )
    assert "File-writing discipline" in prompt_disciplined
    assert "bounded chunks" in prompt_disciplined
    assert "BEFORE any extended reasoning" in prompt_disciplined
    # Early position: right after the task summary, ahead of the paths block.
    assert (prompt_disciplined.index("File-writing discipline")
            > prompt_disciplined.index("Stage 2.c — produce method.py"))
    assert (prompt_disciplined.index("File-writing discipline")
            < prompt_disciplined.index("**Run paths:**"))
    # The flag adds exactly one section and nothing else drifts.
    assert prompt_disciplined.replace(
        "\n\n" + FILE_WRITING_DISCIPLINE, "", 1) == prompt_with_scope

    # Cap-burn corrective-resume nudge (fix 2): short, self-contained,
    # agent-neutral — it rides an in-session resume, so it must not carry
    # role framing or task-specific paths.
    assert "per-step output cap" in CAP_BURN_CORRECTIVE_RESUME_NUDGE
    assert "bounded chunks" in CAP_BURN_CORRECTIVE_RESUME_NUDGE
    assert "write tool call NOW" in CAP_BURN_CORRECTIVE_RESUME_NUDGE
    for role_word in ("producer", "reviewer"):
        assert role_word not in CAP_BURN_CORRECTIVE_RESUME_NUDGE.lower(), (
            "resume nudge must stay role-neutral (the 2026-06-15 "
            "task-inversion hazard)")

    assert STAGE_1_OUTPUT_MODE_WRITEABLE_PATHS["r2c-decomposer"]["canonical"] == [
        ".pipeline/paper_map.json",
    ]
    assert ".pipeline/paper_map_parts/*.json" in (
        STAGE_1_OUTPUT_MODE_WRITEABLE_PATHS["r2c-decomposer"]["chunk"]
    )
    assert ".pipeline/method_spec_parts/*.json" in (
        STAGE_1_OUTPUT_MODE_WRITEABLE_PATHS["r2c-method-analyzer"]["chunk"]
    )

    # build_fix_mode_prompt assembly
    sample_findings = [
        {"id": "F001", "severity": "critical",
         "description": "Function signature drifts from spec.",
         "proposed_fix": "Match spec.comparison.pluggable_component.signature byte-for-byte."},
        {"id": "F002", "severity": "important",
         "description": "Essential feature missing implementation."},
    ]
    fix_prompt = build_fix_mode_prompt(
        target_agent="method-coder",
        findings=sample_findings,
        paths=sample_paths,
    )
    assert "method-coder agent in fix mode" in fix_prompt
    assert "F001 (critical)" in fix_prompt
    assert "F002 (important)" in fix_prompt
    assert "Match spec.comparison.pluggable_component.signature" in fix_prompt
    assert "{target_agent}" not in fix_prompt
    assert "Writeable paths" not in fix_prompt, "fix-mode writeable_paths block leaked without opt-in"
    # sample_paths points at a run dir with no derivable bridge data, so the
    # symbol-ownership block must be absent (byte-identity for such runs).
    assert "Symbol-ownership map" not in fix_prompt, (
        "ownership block leaked into a run without bridge data")

    # format_symbol_ownership_block — pure renderer for the naming-bridge
    # ownership map (fedavg 2026-07-21: a fix producer re-defined a symbol
    # another module owned; the prompt now carries the owner list).
    ownership_block = format_symbol_ownership_block(
        {"federated_train": "method/method.py"})
    assert "**Symbol-ownership map**" in ownership_block
    assert "`federated_train` — defined in `method/method.py`" in ownership_block
    assert "IMPORT the symbol from its owning module" in ownership_block
    assert "second definition of an owned symbol is itself a defect" in ownership_block

    # build_fix_mode_prompt with writeable_paths
    fix_prompt_scoped = build_fix_mode_prompt(
        target_agent="r2c-method-coder",
        findings=sample_findings,
        paths=sample_paths,
        writeable_paths=WRITEABLE_PATHS["r2c-method-coder"],
    )
    assert "**Writeable paths**" in fix_prompt_scoped
    assert "method/method.py" in fix_prompt_scoped
    # Without smoke_context, no SMOKE_FIX_GUIDANCE block leaks in.
    assert "Smoke-gate failure" not in fix_prompt_scoped
    assert "Trace the failing value to its source" not in fix_prompt_scoped

    # build_fix_mode_prompt with smoke_context — exercises the L1 smoke
    # guidance path. SMOKE_FIX_GUIDANCE inserts between FIX_MODE_BODY and
    # the findings block; traceback_hint anchors the agent on the right
    # frame. The (i)–(iv) checklist must be in the rendered prompt.
    fix_prompt_smoke = build_fix_mode_prompt(
        target_agent="r2c-method-coder",
        findings=sample_findings,
        paths=sample_paths,
        writeable_paths=WRITEABLE_PATHS["r2c-method-coder"],
        smoke_context={"traceback_hint": "method/method.py:175 in compute_3d_box_iou"},
    )
    assert "Smoke-gate failure" in fix_prompt_smoke, "SMOKE_FIX_GUIDANCE must appear when smoke_context is provided"
    assert "Trace the failing value to its source" in fix_prompt_smoke, "checklist item (i) missing"
    assert "Surface ≠ bug" in fix_prompt_smoke, "checklist item (ii) missing"
    assert "architecture is the contract" in fix_prompt_smoke, "checklist item (iii) missing"
    assert "Scope escape hatch" in fix_prompt_smoke, "checklist item (iv) missing"
    assert "method/method.py:175 in compute_3d_box_iou" in fix_prompt_smoke, "traceback_hint must be rendered"
    assert "Deepest traceback frame" in fix_prompt_smoke
    # Ordering: SCOPE → FIX_MODE_BODY → SMOKE_FIX_GUIDANCE → findings → paths → writeable_paths
    assert fix_prompt_smoke.index("Smoke-gate failure") < fix_prompt_smoke.index("Fix findings")
    assert fix_prompt_smoke.index("Smoke-gate failure") > fix_prompt_smoke.index("Fix mode")
    assert fix_prompt_smoke.index("Fix findings") < fix_prompt_smoke.index("**Writeable paths**")

    # Empty smoke_context (dict but no hint) — guidance renders, no hint anchor.
    fix_prompt_smoke_no_hint = build_fix_mode_prompt(
        target_agent="r2c-method-coder",
        findings=sample_findings,
        paths=sample_paths,
        smoke_context={},
    )
    assert "Smoke-gate failure" in fix_prompt_smoke_no_hint
    assert "Deepest traceback frame" not in fix_prompt_smoke_no_hint, "no hint anchor when traceback_hint absent"

    # build_smoke_diagnosis_prompt — L2 diagnostician dispatch.
    diag_prompt = build_smoke_diagnosis_prompt(
        paths=sample_paths,
        stderr_tail="IndexError: index 8 is out of bounds for dimension of size 8",
        failing_cell_source="x = boxes[:, 8]\ny = boxes[:, 9]",
        failing_cell_index=22,
        section=4,
        diagnosis_output_path="/tmp/run/.pipeline/smoke_diagnosis.json",
    )
    assert "smoke-gate diagnosis" in diag_prompt, "task summary must name the diagnostician's job"
    assert "USE YOUR WRITE TOOL" in diag_prompt, "Think-class prompt must foreground the write action"
    assert "`smoke_diagnosis.json` only" in diag_prompt, "must scope Writes to the diagnosis file (anti-contamination)"
    # Long-turn survivability (detr 2026-07-04: the INITIAL dispatch also died
    # on the per-step output cap — 677.8s, 75k chars of reasoning, no Write).
    # The initial prompt must carry the budget-your-reasoning checkpoint rule,
    # not just the retry.
    assert "Budget your pre-write reasoning" in diag_prompt, "initial dispatch must carry the output-budget rule"
    assert "best current hypothesis" in diag_prompt, "checkpoint-write rule missing"
    assert "at most ONCE" in diag_prompt, "refining re-Write must stay capped at one"
    assert "Do not Edit any source file" in diag_prompt, "must explicitly forbid source-file edits"
    assert "fix-mode status table" in diag_prompt, "must explicitly disclaim the producer's procedure"
    assert "Steps 1–7" in diag_prompt, "must reference the 7-step procedure"
    assert "index 8 is out of bounds" in diag_prompt, "stderr tail must be embedded"
    assert "boxes[:, 8]" in diag_prompt, "failing cell source must be embedded"
    assert "index 22" in diag_prompt, "failing cell index must appear"
    assert "§4" in diag_prompt, "section number must appear"
    # File-ownership map present for all three smoke-routable agents.
    assert "`r2c-method-coder` owns" in diag_prompt
    assert "`r2c-architecture-coder` owns" in diag_prompt
    assert "`r2c-notebook-generator` owns" in diag_prompt
    assert "method/method.py" in diag_prompt
    assert "method/model.py" in diag_prompt
    # Writeable paths for the diagnostician itself.
    assert "smoke_diagnosis.json" in diag_prompt
    # Scope contract is included (defensive — diagnostician shouldn't propose baseline fixes).
    assert "SCOPE CONTRACT" in diag_prompt
    # Without prior_iterations, no history block leaks in.
    assert "Prior iterations in this smoke fix loop" not in diag_prompt, (
        "prior-iterations block must be opt-in via the kwarg"
    )

    # build_smoke_diagnosis_prompt with prior_iterations — exercises the
    # cross-iteration memory path (Phase 1 change #3). The diagnostician at
    # iter N sees the prior iterations' diagnoses + producer dispatches so
    # it can distinguish regression / new-layer / misroute.
    prior = [
        {
            "iteration": 0,
            "failing_cell": 29,
            "section": 5,
            "mechanical_routing": "method-coder",
            "diagnosis": {
                "target_agent": "r2c-architecture-coder",
                "target_file": "method/model.py",
                "root_cause": "Teacher backbone is nn.Linear; can't accept 3D point cloud input.",
                "proposed_fix": "Use per-point Linear so point structure is preserved.",
            },
            "producer_dispatched": "r2c-architecture-coder",
        },
        {
            "iteration": 1,
            "failing_cell": 29,
            "section": 5,
            "mechanical_routing": "method-coder",
            "diagnosis": {
                "target_agent": "r2c-method-coder",
                "target_file": "method/method.py",
                "root_cause": "compute_fg_guided_mask indexes gt_boxes[b] but receives 2D input.",
                "proposed_fix": "Handle the 2D vs 3D gt_boxes input shape correctly.",
            },
            "producer_dispatched": "r2c-method-coder",
        },
    ]
    diag_prompt_w_history = build_smoke_diagnosis_prompt(
        paths=sample_paths,
        stderr_tail="some new error",
        failing_cell_source="x = ...",
        failing_cell_index=29,
        section=5,
        diagnosis_output_path="/tmp/run/.pipeline/smoke_diagnosis.json",
        prior_iterations=prior,
    )
    assert "Prior iterations in this smoke fix loop" in diag_prompt_w_history
    assert "Iteration 0:" in diag_prompt_w_history
    assert "Iteration 1:" in diag_prompt_w_history
    assert "Teacher backbone is nn.Linear" in diag_prompt_w_history
    assert "compute_fg_guided_mask" in diag_prompt_w_history
    assert "regression" in diag_prompt_w_history.lower(), "must teach the agent how to interpret history"
    assert "new layer" in diag_prompt_w_history.lower()
    assert "re-route" in diag_prompt_w_history.lower() or "wrong agent" in diag_prompt_w_history.lower()
    # Prior block appears between SMOKE_DIAGNOSIS_TASK and the failing-cell block.
    idx_task = diag_prompt_w_history.index("smoke-gate diagnosis")
    idx_history = diag_prompt_w_history.index("Prior iterations")
    idx_cell = diag_prompt_w_history.index("Failing cell:")
    assert idx_task < idx_history < idx_cell, "prior-iterations block must sit between task and cell"

    # Empty prior_iterations list — same as None (no block).
    diag_prompt_empty_history = build_smoke_diagnosis_prompt(
        paths=sample_paths,
        stderr_tail="x",
        failing_cell_source="y",
        failing_cell_index=1,
        section=1,
        diagnosis_output_path="/tmp/foo.json",
        prior_iterations=[],
    )
    assert "Prior iterations" not in diag_prompt_empty_history
    # Without retry_mode, no retry preamble leaks.
    assert "RETRY" not in diag_prompt_empty_history

    # retry_mode=True prepends the missing-output retry preamble. Used by
    # the driver after the first diagnostician dispatch fails to produce
    # smoke_diagnosis.json (Think-class stop-early pattern).
    diag_prompt_retry = build_smoke_diagnosis_prompt(
        paths=sample_paths,
        stderr_tail="x",
        failing_cell_source="y",
        failing_cell_index=1,
        section=1,
        diagnosis_output_path="/tmp/foo.json",
        retry_mode=True,
    )
    assert "Stage 3.c diagnostician RETRY" in diag_prompt_retry
    assert "did NOT produce" in diag_prompt_retry
    # Write-first enforcement (detr 2026-07-04: two long turns reasoned past
    # the per-step output cap and wrote nothing).
    assert "FIRST tool call must be the Write" in diag_prompt_retry
    assert "BEST CURRENT HYPOTHESIS" in diag_prompt_retry
    # Retry preamble sits before the standard task summary.
    assert diag_prompt_retry.index("RETRY") < diag_prompt_retry.index("smoke-gate diagnosis")

    # r2c-smoke-diagnostician is in WRITEABLE_PATHS.
    assert "r2c-smoke-diagnostician" in WRITEABLE_PATHS
    assert WRITEABLE_PATHS["r2c-smoke-diagnostician"] == [".pipeline/smoke_diagnosis.json"]

    try:
        build_fix_mode_prompt(target_agent="x", findings=[], paths=sample_paths)
        raise AssertionError("build_fix_mode_prompt should reject empty findings")
    except ValueError:
        pass

    # OUT_OF_SCOPE_CORRECTIVE_PREAMBLE — branch-A scope-correction note. Must be
    # agent-neutral (NOT producer-vs-reviewer framed; that cross-wire was the
    # 2026-06-15 morning halt's root cause), past-tense + self-contained for a
    # fresh corrective work session, and NOT missing-output framed. Names the
    # removed strays, forbids recreating them, and restates only the positive
    # allowlist.
    corrective = build_corrective_redispatch_preamble(
        strays=["notebook.py", ".pipeline/rogue.json"],
        writeable_paths=[".pipeline/stage_review_stage_2x_params.json"],
    )
    assert "An earlier dispatch of this task created" in corrective, (
        "corrective preamble must be past-tense + self-contained")
    assert "notebook.py" in corrective and ".pipeline/rogue.json" in corrective, (
        "corrective preamble must name the removed strays")
    assert ".pipeline/stage_review_stage_2x_params.json" in corrective, (
        "corrective preamble must restate the positive allowlist")
    assert "do NOT recreate" in corrective, "corrective preamble must forbid recreating strays"
    assert "{stray_lines}" not in corrective and "{allowed_lines}" not in corrective, (
        "corrective preamble format placeholders leaked")
    _low = corrective.lower()
    for _banned in ("producer", "reviewer", "your entire job", "reviewing them is"):
        assert _banned not in _low, (
            f"corrective preamble must be agent-neutral (no role framing); found {_banned!r}")
    assert "did not produce" not in _low and "never called write" not in _low, (
        "corrective preamble must not be missing-output framed (the agent DID write)")
    # Empty stray list still renders a sensible bullet, no placeholder leak.
    corrective_empty = build_corrective_redispatch_preamble(
        strays=[], writeable_paths=["method/method.py"])
    assert "the removed file(s)" in corrective_empty
    assert "{stray_lines}" not in corrective_empty

    # WRITE_FIRST_RETRY_PREAMBLE — the generalized missing-output retry shape
    # (detr 2026-07-04: identical-prompt retries die identically under
    # output-cap deaths; write-first is the proven recovery). Must be
    # agent-neutral and role-neutral like the corrective preamble, demand the
    # Write as the FIRST tool call, carry the imperfect-but-honest framing,
    # and allow exactly one refining re-Write.
    wf = build_write_first_retry_preamble(
        artifact_name="judge_decision.json", artifact_kind="decision",
    )
    assert "WRITE FIRST" in wf
    assert "judge_decision.json" in wf
    assert "FIRST tool call must be the Write" in wf
    assert "imperfect but honest decision" in wf, "artifact_kind must thread into the framing"
    assert "re-Write the file once" in wf, "must allow exactly one refining re-Write"
    assert "do NOT read additional files first" in wf
    assert "Your original task (unchanged) follows." in wf, "must hand back to the original prompt"
    assert "{artifact_name}" not in wf and "{artifact_kind}" not in wf, "format placeholders leaked"
    _wf_low = wf.lower()
    for _banned in ("producer", "reviewer", "your entire job"):
        assert _banned not in _wf_low, (
            f"write-first preamble must be role-neutral; found {_banned!r}")
    wf_review = build_write_first_retry_preamble(
        artifact_name="review_report.json", artifact_kind="review report",
    )
    assert "imperfect but honest review report" in wf_review

    # WRITEABLE_PATHS — every producer/reviewer in the pipeline has an entry,
    # and every path is relative (we resolve against run_dir at check time).
    expected_agents = {
        "r2c-decomposer", "r2c-method-analyzer",
        "r2c-architecture-coder", "r2c-method-coder",
        "r2c-test-generator",
        "r2c-notebook-generator", "r2c-method-explainer",
        "r2c-stage-reviewer", "r2c-paper-fidelity-reviewer",
        "r2c-smoke-diagnostician",
        "r2c-halt-judge",
        "r2c-insight-semantic-reviewer",
        "r2c-insight-producer",
    }
    assert set(WRITEABLE_PATHS) == expected_agents, (
        f"WRITEABLE_PATHS agent set drifted: "
        f"missing={expected_agents - set(WRITEABLE_PATHS)}, "
        f"extra={set(WRITEABLE_PATHS) - expected_agents}"
    )
    for agent, patterns in WRITEABLE_PATHS.items():
        assert patterns, f"WRITEABLE_PATHS[{agent}] is empty"
        for p in patterns:
            assert not p.startswith("/"), (
                f"WRITEABLE_PATHS[{agent}] entries must be relative to "
                f"run_dir: {p!r}"
            )
            assert ".." not in p.split("/"), (
                f"WRITEABLE_PATHS[{agent}] entries must not escape run_dir: {p!r}"
            )

    # build_halt_artifact
    halt = build_halt_artifact(stage="2.x", reason="finding F001", retry_count=0,
                               target_script="scripts/derive_params.py")
    assert halt["status"] == "halted"
    assert halt["stage"] == "2.x"
    assert halt["target_script"] == "scripts/derive_params.py"
    assert "context" not in halt
    json.dumps(halt)  # must be JSON-serializable

    halt_with_ctx = build_halt_artifact(stage="fix_loop", reason="cap reached",
                                        retry_count=3, context={"open_findings": 2})
    assert halt_with_ctx["context"] == {"open_findings": 2}

    # Insight semantic-review prompt: task + anchor + bundle + writeable
    # paths; no SCOPE_CONTRACT (nothing to implement); the acceptance
    # record path reads as a review artifact so the producer-framed
    # anti-review clause (which inverted a reviewer's task on 2026-06-15)
    # never reaches this agent.
    review_prompt = build_insight_review_prompt(
        bundle_json='{"records": []}',
        record_path="/tmp/out/generic_insights.semantic_review.json",
        source_sha256="a" * 64,
        candidate_sha256="b" * 64,
    )
    assert "USE YOUR WRITE TOOL EXACTLY ONCE" in review_prompt
    assert "semantic entailment review" in review_prompt
    assert "unsupported_strengthening" in review_prompt
    assert ("a" * 64) in review_prompt and ("b" * 64) in review_prompt
    assert "{record_path}" not in review_prompt, "format placeholder leaked"
    assert "{{" not in review_prompt, "unrendered brace escape leaked"
    assert "SCOPE CONTRACT" not in review_prompt
    assert PRODUCER_ANTI_REVIEW_CLAUSE not in review_prompt, (
        "producer-framed anti-review clause must not reach a review-only "
        "agent"
    )
    assert _writes_only_review_artifacts(
        [".pipeline/generic_insights.semantic_review.json"])

