"""Deferred US-3 over-claims must not fall through the auto-resolve seam.

The pdwa 2026-07-07 halt (queue item 12b): the deterministic gate deferred a
`c_safe` over-claim to the stage-reviewer relabel path, the reviewer flagged
a DIFFERENT param (r_obs, correctly auto-applied), nobody ever proposed a
relabel for the deferred param, and the run halted with a reason naming the
wrong class ("internal inconsistency between applier and validator" — both
behaved correctly). The fix: synthesize reviewer-shaped findings from the
validator's own bullets, give the reviewer ONE targeted re-ask before
halting, and pick the halt reason by whether the still-failing param was
among the applied locations.
"""

from __future__ import annotations

import json

import run_layout
import run_pipeline
from tests.helpers.state import make_state

# The live pdwa stderr, verbatim shape.
US3_ERR = (
    "FAIL: 1 validation error(s):\n"
    "  - provenance probe US-3: c_safe: c_safe=0.25 is source=paper but no "
    "rendering of the value (0.25, 25%, 25 %, 2.5e-1, 2.5e-01, 2.5×10^-1, "
    "2.5x10^-1, 2.5×10⁻¹) appears in the paper text"
)


# -- unit: the bullet parser -------------------------------------------------


def test_overclaim_bullets_parse_the_live_stderr():
    bullets = run_pipeline._params_overclaim_bullets(US3_ERR)
    assert [name for name, _ in bullets] == ["c_safe"]
    assert bullets[0][1].startswith("provenance probe US-3: c_safe:")


def test_overclaim_bullets_ignore_non_us3_and_dedupe():
    err = ("FAIL: 3 validation error(s):\n"
           "  - provenance probe US-3: alpha: alpha=1 is source=paper but ...\n"
           "  - provenance probe US-3: alpha: alpha=1 is source=paper but ...\n"
           "  - provenance probe US-1: beta out of range\n")
    assert [n for n, _ in run_pipeline._params_overclaim_bullets(err)] == ["alpha"]


# -- unit: the truthful halt reason -----------------------------------------


def test_halt_reason_names_the_unaddressed_deferred_param():
    # The pdwa case: r_obs was applied, c_safe still fails — the old code
    # called this an applier/validator inconsistency.
    reason = run_pipeline._post_resolve_halt_reason(
        US3_ERR, {"params.r_obs"})
    assert "c_safe" in reason
    assert "no resolution ever addressed" in reason
    assert "internal inconsistency" not in reason


def test_halt_reason_keeps_inconsistency_when_applied_param_still_fails():
    reason = run_pipeline._post_resolve_halt_reason(
        US3_ERR, {"params.c_safe"})
    assert "internal inconsistency" in reason


def test_halt_reason_keeps_inconsistency_for_non_us3_failures():
    reason = run_pipeline._post_resolve_halt_reason(
        "FAIL: 1 validation error(s):\n  - params.json schema: value must "
        "be a number", {"params.r_obs"})
    assert "internal inconsistency" in reason


# -- unit: the synthesized findings ------------------------------------------


def test_synthesized_findings_carry_the_validator_bullet_verbatim():
    found = run_pipeline._synthesize_deferred_overclaim_findings(
        US3_ERR, {"params.r_obs"})
    assert len(found) == 1
    f = found[0]
    assert f["id"] == "DEF-c_safe"
    assert f["location"] == "params.c_safe"
    assert f["severity"] == "important"
    assert f["check_id"] == "paper_source_values_match_paper"
    assert f["description"].startswith("provenance probe US-3: c_safe:")
    assert "proposed_resolution" not in f  # must be a re-ask candidate


def test_synthesized_findings_exclude_applied_params():
    # An applied param that still fails is a genuine inconsistency: halt,
    # never re-ask.
    assert run_pipeline._synthesize_deferred_overclaim_findings(
        US3_ERR, {"params.c_safe"}) == []


# -- integration: the pdwa shape through run_stage_2x ------------------------


PARAMS_JSON = json.dumps({
    "params": {
        "c_safe": {
            "value": 0.25,
            "source": "paper",
            "paper_section": "Section 3.2, Equation (5)",
        },
        "r_obs": {
            "value": 0.15,
            "source": "spec_default",
            "reasoning": "Value from the pluggable_component signature.",
        },
    },
}, indent=2)


def _review(findings: list[dict]) -> str:
    # Real reask artifacts carry the full seven-key stage-review shape with
    # complete findings (see example_runs/pdwa's resolution file); the read
    # path validates against schemas/stage_review_report.py (B-01). Pad the
    # test's delta-shaped findings with the core keys the reviewer always
    # writes, keeping each test's id/resolution content authoritative.
    core = {
        "severity": "important", "target_agent": "parameter-deriver",
        "issue_type": "provenance_reasoning_inaccurate",
        "file": ".pipeline/params.json", "location": "params.c_safe",
        "description": "fabricated test finding",
    }
    return json.dumps({
        "schema_version": "1.0.0",
        "stage_id": "stage_2x_params",
        "review_status": "issues_found" if findings else "passed",
        "summary": "fabricated test review",
        "findings": [{**core, **f} for f in findings],
        "checks_summary": [], "caveats": [],
    }, indent=2)


R_OBS_FINDING = {
    "id": "F001",
    "check_id": "paper_source_values_match_paper",
    "severity": "important",
    "location": "params.r_obs",
    "description": "r_obs is labelled spec_default but Table 1 states 0.15 m.",
    "proposed_resolution": {
        "kind": "relabel_param_source",
        "param": "r_obs",
        "new_source": "paper",
        "add_fields": {"paper_section": "Section 4, Table 1"},
    },
}


def _drive_stage_2x(monkeypatch, run_dir, fake_dispatch, fake_subprocess,
                    *, reask_findings: list[dict],
                    revalidations: list[tuple[bool, str]]):
    """Drive run_stage_2x through the pdwa shape: derive → validator defers
    the c_safe over-claim → reviewer flags only r_obs (auto-applied) →
    re-validation still fails on c_safe → synthesized re-ask answers with
    `reask_findings` → the queued `revalidations` decide the ending."""
    state = make_state(run_dir)
    # Orthogonal 2x machinery (binding disclosure, scale calibration, budget
    # sufficiency) is not under test here.
    for name in ("_log_per_dataset_binding", "_apply_scale_calibration"):
        monkeypatch.setattr(run_pipeline, name, lambda *a, **k: None)
    monkeypatch.setattr(run_pipeline, "_check_training_budget_sufficiency",
                        lambda *a, **k: None)

    fake_subprocess.set_run_dir(run_dir)
    fake_subprocess.expect_script(  # derive_params.py
        returncode=0, writes={".pipeline/params.json": PARAMS_JSON})
    fake_subprocess.expect_stage2_script(ok=False, err=US3_ERR)  # gate 1
    fake_dispatch.expect(  # stage reviewer: flags ONLY r_obs
        agent="r2c-stage-reviewer",
        writes={".pipeline/stage_review_stage_2x_params.json":
                _review([R_OBS_FINDING])})
    fake_subprocess.expect_stage2_script(ok=False, err=US3_ERR)  # post-resolve
    fake_dispatch.expect(  # the targeted re-ask for DEF-c_safe
        agent="r2c-stage-reviewer",
        writes={".pipeline/stage_review_stage_2x_params_resolution.json":
                _review(reask_findings)})
    for ok, err in revalidations:
        fake_subprocess.expect_stage2_script(ok=ok, err=err)

    return run_pipeline.run_stage_2x(state)


def test_deferred_overclaim_resolves_through_the_targeted_reask(
        monkeypatch, run_dir, fake_dispatch, fake_subprocess):
    result = _drive_stage_2x(
        monkeypatch, run_dir, fake_dispatch, fake_subprocess,
        reask_findings=[{
            "id": "DEF-c_safe",
            "proposed_resolution": {
                # The live 2026-07-07 resume resolution shape: substantively
                # perfect AND carrying a helpful non-schema key
                # (component_sources) that used to revert the whole edit.
                "kind": "relabel_param_source",
                "param": "c_safe",
                "new_source": "system_inferred",
                "remove_fields": ["paper_section", "note"],
                "add_fields": {
                    "reasoning": "Derived: r_robot + r_obs = 0.10 + 0.15.",
                    "component_sources": "r_robot=0.1 Table 1; r_obs=0.15 "
                                         "Table 1; formula Eq (5)",
                },
            },
        }],
        revalidations=[(True, "")],
    )
    assert result.status == "completed"
    assert "auto-resolved 2 finding(s)" in result.notes
    params = json.loads(
        (run_dir / ".pipeline" / "params.json").read_text(encoding="utf-8"))
    assert params["params"]["c_safe"]["source"] == "system_inferred"
    assert "component_sources" not in params["params"]["c_safe"]
    assert params["params"]["r_obs"]["source"] == "paper"
    # The dropped key's content is preserved on the researcher surface.
    assumptions = (run_dir / run_layout.ASSUMPTIONS_MD).read_text(
        encoding="utf-8")
    assert "component_sources" in assumptions
    # Both dispatches were the stage reviewer: the review, then the re-ask.
    assert [c.agent for c in fake_dispatch.calls] == ["r2c-stage-reviewer"] * 2


def test_reask_needs_user_halts_as_a_deliberate_user_escalation(
        monkeypatch, run_dir, fake_dispatch, fake_subprocess):
    """An explicit needs_user answer on the re-ask is a reviewer decision,
    not a producer defect: the halt carries the needs_user_input catalog
    class (reserved until now) and the reason names the judgment being
    asked instead of implying a defect nobody addressed."""
    result = _drive_stage_2x(
        monkeypatch, run_dir, fake_dispatch, fake_subprocess,
        reask_findings=[{
            "id": "DEF-c_safe",
            "resolution_status": "needs_user",
        }],
        revalidations=[],  # nothing applied, no further validator run
    )
    assert result.status == "halted"
    assert "needs your judgment" in result.notes
    assert "c_safe" in result.notes
    assert "internal inconsistency" not in result.notes
    assert "no resolution ever addressed" not in result.notes
    halt_artifact = json.loads(
        (run_dir / ".pipeline" / "stage_2x.halt").read_text(encoding="utf-8"))
    assert halt_artifact["halt_class"] == "needs_user_input"


def test_reask_silence_on_the_finding_stays_producer_output_invalid(
        monkeypatch, run_dir, fake_dispatch, fake_subprocess):
    """The needs_user reclassification requires an EXPLICIT answer. A re-ask
    that returns no decision for the deferred param keeps the defect class
    and the no-resolution-ever-addressed reason."""
    result = _drive_stage_2x(
        monkeypatch, run_dir, fake_dispatch, fake_subprocess,
        reask_findings=[{
            "id": "DEF-unrelated",
            "resolution_status": "needs_user",
        }],
        revalidations=[],
    )
    assert result.status == "halted"
    assert "no resolution ever addressed" in result.notes
    halt_artifact = json.loads(
        (run_dir / ".pipeline" / "stage_2x.halt").read_text(encoding="utf-8"))
    assert halt_artifact["halt_class"] == "producer_output_invalid"


def test_reask_apply_failure_halts_with_the_apply_failed_reason(
        monkeypatch, run_dir, fake_dispatch, fake_subprocess):
    """A resolution that is genuinely schema-invalid (system_inferred with
    no reasoning anywhere) must still revert and halt — and the reason must
    say the resolution failed to APPLY, not that none was ever proposed."""
    result = _drive_stage_2x(
        monkeypatch, run_dir, fake_dispatch, fake_subprocess,
        reask_findings=[{
            "id": "DEF-c_safe",
            "proposed_resolution": {
                "kind": "relabel_param_source",
                "param": "c_safe",
                "new_source": "system_inferred",
                "remove_fields": ["paper_section", "note"],
                # No reasoning: the entry cannot validate as system_inferred.
            },
        }],
        revalidations=[],  # apply reverted; no further validator run
    )
    assert result.status == "halted"
    assert "failed to apply" in result.notes
    assert "c_safe" in result.notes
    assert "no resolution ever addressed" not in result.notes
    assert "internal inconsistency" not in result.notes
    # The revert left the original over-claim label in place, untouched.
    params = json.loads(
        (run_dir / ".pipeline" / "params.json").read_text(encoding="utf-8"))
    assert params["params"]["c_safe"]["source"] == "paper"


def test_halt_reason_prefers_apply_failed_over_unaddressed():
    reason = run_pipeline._post_resolve_halt_reason(
        US3_ERR, set(), {"c_safe"})
    assert "failed to apply" in reason
    assert "internal inconsistency" not in reason


def test_applier_drops_unknown_add_fields_and_records_them(run_dir):
    """The live 2026-07-07 resolution, verbatim: perfect relabel plus a
    helpful non-schema key. The old behavior reverted the whole edit."""
    state = make_state(run_dir)
    params_path = state.paths.pipeline_dir / "params.json"
    params_path.write_text(PARAMS_JSON, encoding="utf-8")
    finding = {
        "id": "DEF-c_safe",
        "description": "provenance probe US-3: c_safe: ...",
        "proposed_resolution": {
            "kind": "relabel_param_source",
            "param": "c_safe",
            "new_source": "system_inferred",
            "remove_fields": ["paper_section", "note"],
            "add_fields": {
                "reasoning": "Sum of two paper-sourced radii.",
                "component_sources": "r_robot=0.1; r_obs=0.15 (Table 1)",
            },
            "expert_reasoning": "The paper never writes 0.25.",
        },
    }
    ok, aid = run_pipeline._apply_relabel_param_source_resolution(
        state, finding)
    assert ok is True
    entry = json.loads(params_path.read_text(encoding="utf-8"))["params"]["c_safe"]
    assert entry["source"] == "system_inferred"
    assert entry["reasoning"] == "Sum of two paper-sourced radii."
    assert "component_sources" not in entry
    assumptions = (run_dir / run_layout.ASSUMPTIONS_MD).read_text(
        encoding="utf-8")
    assert "component_sources" in assumptions
    assert "r_obs=0.15 (Table 1)" in assumptions


def test_applier_reverts_a_relabel_that_leaves_an_inline_paper_claim(run_dir):
    # bayesian-active-learning 2026-07-27: the applier relabeled the source
    # and returned success, but the reasoning still claimed the paper gives
    # the value. US-3b fires on every source, so the next validator run
    # halted the stage with an "internal inconsistency between applier and
    # validator". The applier now refuses the resolution instead.
    state = make_state(run_dir)
    params_path = state.paths.pipeline_dir / "params.json"
    params_path.write_text(PARAMS_JSON, encoding="utf-8")
    state.paths.paper_md.write_text(
        "The safety margin is discussed qualitatively only.\n",
        encoding="utf-8")
    finding = {
        "id": "DEF-c_safe",
        "description": "provenance probe US-3: c_safe: ...",
        "proposed_resolution": {
            "kind": "relabel_param_source",
            "param": "c_safe",
            "new_source": "system_inferred",
            "remove_fields": ["paper_section"],
            "add_fields": {
                "reasoning": "The paper states c_safe=0.25 in Section 3.2.",
            },
            "expert_reasoning": "The paper never writes 0.25.",
        },
    }
    ok, msg = run_pipeline._apply_relabel_param_source_resolution(
        state, finding)
    assert ok is False
    assert "cannot clear the finding" in msg
    assert "c_safe=0.25" in msg
    entry = json.loads(params_path.read_text(encoding="utf-8"))["params"]["c_safe"]
    assert entry["source"] == "paper"  # reverted


def test_applier_accepts_the_same_relabel_with_honest_reasoning(run_dir):
    # The other half of the fix: reworded without the name=value claim, the
    # identical relabel goes through.
    state = make_state(run_dir)
    params_path = state.paths.pipeline_dir / "params.json"
    params_path.write_text(PARAMS_JSON, encoding="utf-8")
    state.paths.paper_md.write_text(
        "The safety margin is discussed qualitatively only.\n",
        encoding="utf-8")
    finding = {
        "id": "DEF-c_safe",
        "description": "provenance probe US-3: c_safe: ...",
        "proposed_resolution": {
            "kind": "relabel_param_source",
            "param": "c_safe",
            "new_source": "system_inferred",
            "remove_fields": ["paper_section"],
            "add_fields": {
                "reasoning": ("The paper discusses the safety margin "
                              "qualitatively without prescribing a value."),
            },
            "expert_reasoning": "The paper never writes 0.25.",
        },
    }
    ok, aid = run_pipeline._apply_relabel_param_source_resolution(
        state, finding)
    assert ok is True, aid
    entry = json.loads(params_path.read_text(encoding="utf-8"))["params"]["c_safe"]
    assert entry["source"] == "system_inferred"


def test_applier_safety_net_still_reverts_truly_invalid_resolutions(run_dir):
    state = make_state(run_dir)
    params_path = state.paths.pipeline_dir / "params.json"
    params_path.write_text(PARAMS_JSON, encoding="utf-8")
    finding = {
        "id": "DEF-c_safe",
        "description": "provenance probe US-3: c_safe: ...",
        "proposed_resolution": {
            "kind": "relabel_param_source",
            "param": "c_safe",
            "new_source": "system_inferred",
            "remove_fields": ["paper_section", "note"],
            # No reasoning anywhere: schema-invalid, must revert.
        },
    }
    ok, msg = run_pipeline._apply_relabel_param_source_resolution(
        state, finding)
    assert ok is False
    assert "schema-invalid" in msg
    entry = json.loads(params_path.read_text(encoding="utf-8"))["params"]["c_safe"]
    assert entry["source"] == "paper"  # reverted
