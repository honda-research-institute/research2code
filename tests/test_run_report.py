"""Full REPORT.md front-door assembler tests."""

from __future__ import annotations

import json
import re

import pytest

from delivery_label import derive_delivery_label
from generate_method_md import generate_method_md
from render_run_report import render_run_report, write_run_report
from schemas.method_spec import EvaluationProtocol
from schemas.params import Params


def _write_json(path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _write_text(path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _report_protocol_spec() -> dict:
    return {
        "comparison": {
            "evaluation_protocol": {
                "scheme": {
                    "kind": None,
                    "paper_value_status": "paper_unspecified",
                    "description": "The paper does not specify origin traversal.",
                    "evidence_quote": (
                        "The evaluation section lists validation and test durations."
                    ),
                    "paper_section": "Evaluation protocol",
                    "paper_element_ids": ["protocol-scheme"],
                },
                "quantities": [
                    {
                        "role": "context_length",
                        "parameter_name": "context_length",
                        "paper_names": ["context length"],
                        "paper_symbols": [],
                        "value": 10,
                        "unit": "week",
                        "granularity": 1,
                        "axis_evidence_quote": (
                            "Weekly demand observations define the target time series."
                        ),
                        "axis_paper_section": "Data cadence",
                        "axis_paper_element_ids": ["protocol-axis"],
                        "paper_value_status": "paper_stated",
                        "evidence_quote": "The context length is 10 weeks.",
                        "paper_section": "Method inputs",
                        "paper_element_ids": ["protocol-context"],
                    },
                    {
                        "role": "forecast_call_horizon",
                        "parameter_name": "forecast_horizon",
                        "paper_names": ["forecast horizon"],
                        "paper_symbols": ["K"],
                        "value": None,
                        "unit": "week",
                        "granularity": 1,
                        "axis_evidence_quote": (
                            "Weekly demand observations define the target time series."
                        ),
                        "axis_paper_section": "Data cadence",
                        "axis_paper_element_ids": ["protocol-axis"],
                        "paper_value_status": "paper_unspecified",
                        "evidence_quote": (
                            "The forecast horizon K denotes the future forecast steps."
                        ),
                        "paper_section": "Forecast definition",
                        "paper_element_ids": ["protocol-horizon"],
                    },
                    {
                        "role": "validation_span",
                        "parameter_name": None,
                        "paper_names": ["validation span"],
                        "paper_symbols": [],
                        "value": 13,
                        "unit": "week",
                        "granularity": 1,
                        "axis_evidence_quote": (
                            "Weekly demand observations define the target time series."
                        ),
                        "axis_paper_section": "Data cadence",
                        "axis_paper_element_ids": ["protocol-axis"],
                        "paper_value_status": "paper_stated",
                        "evidence_quote": "The validation span is 13 weeks.",
                        "paper_section": "Evaluation splits",
                        "paper_element_ids": ["protocol-validation"],
                    },
                    {
                        "role": "test_span",
                        "parameter_name": None,
                        "paper_names": ["test span"],
                        "paper_symbols": [],
                        "value": 26,
                        "unit": "week",
                        "granularity": 1,
                        "axis_evidence_quote": (
                            "Weekly demand observations define the target time series."
                        ),
                        "axis_paper_section": "Data cadence",
                        "axis_paper_element_ids": ["protocol-axis"],
                        "paper_value_status": "paper_stated",
                        "evidence_quote": "The test span is 26 weeks.",
                        "paper_section": "Evaluation splits",
                        "paper_element_ids": ["protocol-test"],
                    },
                ],
            },
        },
    }


def test_front_door_orders_claims_then_probe_summary(tmp_path):
    run = tmp_path / "paper-run"
    pipeline = run / ".pipeline"
    pipeline.mkdir(parents=True)
    _write_json(pipeline / "claims_ledger.json", {
        "claims": [{
            "kind": "lifted",
            "status": "untested_at_this_scale",
            "name": "headline accuracy",
            "reasoning": "We did not verify this claim at smoke scale.",
            "what_would_verify": "a full-scale run",
        }],
    })
    _write_json(pipeline / "probe_report.json", {
        "verdicts": [
            {"probe_id": "US-3", "verdict": "fail",
             "message": "learning_rate quote not found"},
            {"probe_id": "UB-6", "verdict": "unprobeable",
             "message": "notebook has no executed outputs"},
            {"probe_id": "AL-3", "verdict": "warn",
             "message": "fresh retrain noted by AL-4"},
            {"probe_id": "CT-1", "verdict": "pass",
             "message": "selection differs"},
        ],
    })
    _write_json(pipeline / "driver_state.json", {
        "stages": [{
            "stage_id": "stage_3c",
            "stage_label": "Stage 3.c - Smoke Execution",
            "status": "degraded",
            "notes": "smoke execution hit F001 after US-3 failed",
        }],
    })
    _write_text(run / "assumptions.md",
                "# Pipeline assumptions\n\n## A001 - NEEDS USER ATTENTION - finding F001\n\nbody\n")
    _write_text(run / "deferred_findings.md",
                "# Findings deferred\n\n## Important findings (1)\n\n### F001\n\nbody\n")
    _write_text(run / "KNOWN_ISSUES.md",
                "# Known issues\n\n## stage_3c - notebook does not run\n\nbody\n")

    md = render_run_report(run, delivery={"label": "draft"})

    assert md.index("# Run Report") < md.index("## Verification")
    assert md.index("## Verification") < md.index("## Automated checks")
    assert "paper-sourced values are findable in the paper" in md
    assert "executed notebook outputs are sane" in md
    assert "fresh model weights are verified behaviorally" in md
    assert "notebook execution" in md
    # Quoted-noun gloss form: grammatical in attributive positions (the
    # pdwa 2026-07-05 gibberish class), codes still never bare.
    assert ('smoke execution hit the linked finding after the '
            '"paper-sourced values are findable in the paper" check '
            'failed') in md
    assert "1 assumption entry; 1 needs researcher attention" in md
    assert "1 deferred finding" in md
    assert "1 known issue" in md
    assert re.search(r"\b(?:US|UB|AL|MP|KD|CT)-\d+[a-z]?\b", md) is None
    assert re.search(r"\bF\d{3}\b", md) is None


def test_not_applicable_checks_are_counted_and_rendered_separately(tmp_path):
    run = tmp_path / "conditional-probe-run"
    pipeline = run / ".pipeline"
    pipeline.mkdir(parents=True)
    _write_json(pipeline / "probe_report.json", {
        "verdicts": [
            {"probe_id": "US-1", "verdict": "pass",
             "message": "parameter values are plausible"},
            {"probe_id": "AL-3", "verdict": "not_applicable",
             "message": "fresh retraining is not required by this contract"},
            {"probe_id": "UB-6", "verdict": "unprobeable",
             "message": "executed outputs are unavailable"},
        ],
    })

    md = render_run_report(run, delivery={"label": "verified"})

    assert "Results: 1 passed, 1 not applicable, 1 not checked." in md
    assert "### Checks that could not run" in md
    assert "### Checks that did not apply" in md
    assert "fresh retraining is not required by this contract" in md
    assert "outside the declared method shape by design" in md


def test_report_renders_paper_protocol_and_demo_values_separately(tmp_path):
    """Synthetic renderer control; not evidence for a second TSF paper."""
    run = tmp_path / "forecast-run"
    pipeline = run / ".pipeline"
    pipeline.mkdir(parents=True)
    spec = _report_protocol_spec()
    EvaluationProtocol.model_validate(
        spec["comparison"]["evaluation_protocol"]
    )
    params = {
        "params": {
            "context_length": {
                "value": 10,
                "source": "paper",
                "paper_section": "Method inputs",
                "paper_says": "The context length is 10 weeks.",
                "protocol_role": "context_length",
                "protocol_value": 10,
                "protocol_unit": "week",
                "protocol_granularity": 1,
                "protocol_axis_says": (
                    "Weekly demand observations define the target time series."
                ),
                "protocol_axis_section": "Data cadence",
                "protocol_axis_element_ids": ["protocol-axis"],
                "paper_value_status": "paper_stated",
                "paper_element_ids": ["protocol-context"],
            },
            "forecast_horizon": {
                "value": 4,
                "source": "system_inferred",
                "reasoning": "System-owned demo choice.",
                "paper_section": "Forecast definition",
                "paper_says": (
                    "The forecast horizon K denotes the future forecast steps."
                ),
                "protocol_role": "forecast_call_horizon",
                "protocol_value": None,
                "protocol_unit": "week",
                "protocol_granularity": 1,
                "protocol_axis_says": (
                    "Weekly demand observations define the target time series."
                ),
                "protocol_axis_section": "Data cadence",
                "protocol_axis_element_ids": ["protocol-axis"],
                "paper_value_status": "paper_unspecified",
                "paper_element_ids": ["protocol-horizon"],
            },
        },
    }
    Params.model_validate(params)
    _write_json(pipeline / "method_spec.json", spec)
    _write_json(pipeline / "params.json", params)

    report = render_run_report(run)

    assert "## Evaluation protocol" in report
    assert "**Paper evaluation scheme:** **paper-unspecified**" in report
    assert "paper IDs: `protocol-scheme`" in report
    assert "| role/value evidence | axis evidence |" in report
    assert "One-call forecast horizon** | **paper-unspecified**" in report
    assert "`forecast_horizon` = 4 (system inferred)" in report
    assert (
        "Forecast definition; paper IDs: `protocol-horizon` | "
        "Data cadence; paper IDs: `protocol-axis` |"
    ) in report
    assert "Context length** | 10 (paper-stated)" in report
    assert "Validation span** | 13 (paper-stated)" in report
    assert "Test span** | 26 (paper-stated)" in report
    assert report.count("spec-only fact; no runtime parameter") == 2


def test_report_renders_demo_evidence_as_five_separate_axes(tmp_path):
    run = tmp_path / "forecast-run"
    (run / ".pipeline").mkdir(parents=True)
    delivery = {
        "label": "draft",
        "reasons": [{
            "source": "demo_verdict",
            "id": "demo_evaluation_invalid",
            "message": "the scored evaluation is invalid",
        }],
        "disclosures": [],
        "demo_verdict": {
            "schema_version": "2.0.0",
            "verdict": "undetermined",
            "evidence_status": {
                "execution": {
                    "status": "completed",
                    "reasons": [{"code": "smoke", "message": "the notebook ran"}],
                },
                "evaluation_validity": {
                    "status": "invalid",
                    "reasons": [{
                        "code": "overlap",
                        "message": "fitting and reported evaluation overlap",
                    }],
                },
                "mechanism": {
                    "status": "undetermined",
                    "reasons": [{
                        "code": "no_probe",
                        "message": "no bound behavioral probe ran",
                    }],
                },
                "skill": {
                    "status": "undetermined",
                    "reasons": [{
                        "code": "invalid_eval",
                        "message": "invalid evaluation cannot demonstrate skill",
                    }],
                },
                "paper_benchmark": {
                    "status": "not_assessed",
                    "reasons": [{
                        "code": "smoke_scale",
                        "message": "demo scale is not the paper benchmark",
                    }],
                },
            },
        },
    }

    md = render_run_report(run, delivery=delivery)

    assert "## Demo evidence" in md
    assert "| Execution | **completed** | the notebook ran |" in md
    assert "| Evaluation validity | **invalid** |" in md
    assert "| Mechanism evidence | **undetermined** |" in md
    assert "| Demo task skill | **undetermined** |" in md
    assert "| Paper benchmark reproduction | **not assessed** |" in md
    assert "**Demo evaluation validity**" in md
    assert md.index("## Demo evidence") < md.index("## Why this label")


def test_report_renders_graph_evidence_ladder_as_separate_axes(tmp_path):
    run = tmp_path / "graph-forecast-run"
    (run / ".pipeline").mkdir(parents=True)
    delivery = {
        "label": "uncertified_new_territory",
        "reasons": [],
        "disclosures": [],
        "demo_verdict": {
            "schema_version": "2.0.0",
            "verdict": "undetermined",
            "evidence_status": {
                "execution": {"status": "completed", "reasons": []},
                "evaluation_validity": {"status": "valid", "reasons": []},
                "graph_construction": {
                    "status": "demonstrated",
                    "reasons": [{"message": "parameters and topology agree"}],
                },
                "graph_alignment": {
                    "status": "demonstrated",
                    "reasons": [{"message": "entity identities stay aligned"}],
                },
                "mechanism": {
                    "status": "demonstrated",
                    "reasons": [{"message": "topology and neighbors affect output"}],
                },
                "contribution": {
                    "status": "undetermined",
                    "reasons": [{"message": "the declared null could not run"}],
                },
                "skill": {"status": "demonstrated", "reasons": []},
                "paper_benchmark": {"status": "not_assessed", "reasons": []},
            },
        },
    }

    md = render_run_report(run, delivery=delivery)

    assert "| Graph construction | **demonstrated** |" in md
    assert "| Graph/entity alignment | **demonstrated** |" in md
    assert "| Mechanism liveness | **demonstrated** |" in md
    assert "| Contribution ablation | **undetermined** |" in md
    assert md.index("Graph construction") < md.index("Graph/entity alignment")
    assert md.index("Graph/entity alignment") < md.index("Mechanism liveness")
    assert md.index("Mechanism liveness") < md.index("Contribution ablation")
    assert md.index("Contribution ablation") < md.index("Demo task skill")


def test_report_discloses_missing_graph_evidence_as_separate_axes(tmp_path):
    run = tmp_path / "graph-forecast-run"
    (run / ".pipeline").mkdir(parents=True)
    delivery = {
        "schema_version": "1.7.0",
        "graph_evidence_applicability": "homogeneous_graph_v1",
        "label": "draft",
        "reasons": [{
            "source": "demo_verdict",
            "id": "structured_demo_evidence_missing",
            "message": "the structured graph evidence record is missing",
        }],
        "disclosures": [],
    }

    md = render_run_report(run, delivery=delivery)

    assert "## Demo evidence" in md
    assert (
        "homogeneous graph contract is present, but its complete structured "
        "graph evidence record is missing" in md
    )
    assert "| Graph construction | **undetermined** |" in md
    assert "| Graph/entity alignment | **undetermined** |" in md
    assert "| Mechanism liveness | **undetermined** |" in md
    assert "| Contribution ablation | **undetermined** |" in md
    assert "| Demo task skill | **undetermined** |" in md
    assert "| Paper benchmark reproduction | **undetermined** |" in md
    assert md.index("Graph construction") < md.index("Graph/entity alignment")
    assert md.index("Graph/entity alignment") < md.index("Mechanism liveness")
    assert md.index("Mechanism liveness") < md.index("Contribution ablation")
    assert md.index("Contribution ablation") < md.index("Demo task skill")


def test_report_renders_bounded_structured_training_history(tmp_path):
    from tests.test_time_series_training_history import _record

    run = tmp_path / "forecast-run"
    pipeline = run / ".pipeline"
    pipeline.mkdir(parents=True)
    losses = [
        {
            "index": index,
            "value": 10.0 / (index + 1),
            "sample_weight": 8,
            "checkpoint_id": f"checkpoint-{index}",
        }
        for index in range(30)
    ]
    record = _record(
        checkpoint_id="checkpoint-29",
        loss_observations=losses,
        selection_range=None,
        selection={
            "status": "not_performed",
            "reason": "the fitting entry returns its final checkpoint",
        },
    )
    _write_json(pipeline / "training_history.json", record)

    md = render_run_report(run)

    assert "## Training and model selection" in md
    assert "| Model | `forecast-model` |" in md
    assert "| Evaluated checkpoint | `checkpoint-29` |" in md
    assert "`batch.targets` [0, 3)" in md
    assert "30 epoch(s); start `10`, best `0.333333`, end `0.333333`" in md
    assert "not performed; the fitting entry returns its final checkpoint" in md
    assert ".pipeline/training_history.json" in md
    assert record["record_digest"] in md
    # The front door summarizes the series instead of dumping every event.
    assert "checkpoint-15" not in md


def test_report_renders_explicit_nontraining_history_without_inventing_ranges(
    tmp_path,
):
    from scripts.time_series_training_history import (
        not_applicable_training_history,
    )

    run = tmp_path / "nontraining-run"
    pipeline = run / ".pipeline"
    pipeline.mkdir(parents=True)
    record = not_applicable_training_history(
        reason="the declared method has no training phase"
    )
    _write_json(pipeline / "training_history.json", record)

    md = render_run_report(run)

    assert "## Training and model selection" in md
    assert "**Not applicable.** the declared method has no training phase" in md
    assert "Fitting targets" not in md


def test_report_omits_training_section_when_no_structured_artifact(tmp_path):
    run = tmp_path / "legacy-run"
    (run / ".pipeline").mkdir(parents=True)

    assert "## Training and model selection" not in render_run_report(run)


def test_report_adds_no_protocol_section_for_legacy_spec(tmp_path):
    run = tmp_path / "legacy-run"
    pipeline = run / ".pipeline"
    pipeline.mkdir(parents=True)
    _write_json(pipeline / "method_spec.json", {"comparison": {}})
    _write_json(pipeline / "params.json", {"params": {}})

    assert "## Evaluation protocol" not in render_run_report(run)


def _write_events(run_dir, events):
    (run_dir / ".pipeline").mkdir(parents=True, exist_ok=True)
    (run_dir / ".pipeline" / "run_events.jsonl").write_text(
        "\n".join(json.dumps(e) for e in events) + "\n", encoding="utf-8")


def test_resolved_catches_render_instead_of_blanket_sentence(tmp_path):
    """R2C-040 (RCA 2026-08-03 finding 8): a bumpy-but-recovered run must
    disclose what the gates caught and fixed, never the 'no failures were
    recorded' sentence."""
    from render_run_report import _stage_failures

    run = tmp_path / "run"
    _write_events(run, [
        {"event_type": "run_started", "summary": "leg"},
        {"event_type": "validation_failed", "stage_id": "stage_2x",
         "stage_label": "Stage 2.x - Parameter Derivation",
         "details": {"validator": "validate_params_output.py",
                     "stderr_tail": "FAIL: 1 validation error(s):\n"
                                    "  - provenance probe US-3: phi "
                                    "difference thresholds"}},
        {"event_type": "validation_passed", "stage_id": "stage_2x",
         "details": {"validator": "validate_params_output.py"}},
        {"event_type": "stage_completed", "stage_id": "stage_2x"},
        {"event_type": "smoke_failed", "stage_id": "stage_3c",
         "stage_label": "Stage 3.c - Smoke Execution",
         "details": {"exit_code": 1}},
        {"event_type": "smoke_passed", "stage_id": "stage_3c"},
        {"event_type": "stage_completed", "stage_id": "stage_3c"},
    ])

    section = _stage_failures(run, [])

    assert "quality gates recorded 2 problems" in section
    assert "Stage 2.x - Parameter Derivation" in section
    assert "notebook execution attempt failed" in section
    assert "passed the same `validate_params_output.py` check" in section
    assert "later notebook execution passed the same check" in section
    assert "No unresolved failures remained at delivery." in section
    assert "No method understanding" not in section
    # Probe codes never bare (US-3 must arrive glossed by the scrubber).
    assert re.search(r"\bUS-\d+\b", section) is None


def test_unrelated_validator_pass_does_not_claim_correction(tmp_path):
    from render_run_report import _stage_failures

    run = tmp_path / "run"
    _write_events(run, [
        {"event_type": "run_started", "summary": "leg"},
        {"event_type": "validation_failed", "stage_id": "stage_1",
         "stage_label": "Stage 1 - Paper Decomposition & Method Analysis",
         "details": {"validator": "validate_method_spec.py --strict",
                     "stderr_tail": "schema validation failed"}},
        {"event_type": "validation_passed", "stage_id": "stage_1",
         "details": {"validator": "check_feasibility.py"}},
        {"event_type": "stage_completed", "stage_id": "stage_1"},
    ])

    section = _stage_failures(run, [])

    assert "validate_method_spec.py --strict" in section
    assert "no passing re-check" in section
    assert "corrected attempt" not in section
    assert "caught and fixed" not in section
    assert "re-verified" not in section
    assert "No method understanding" not in section
    assert "No stage remained halted or degraded" in section


def test_completed_stage_without_recheck_does_not_claim_correction(tmp_path):
    from render_run_report import _stage_failures

    run = tmp_path / "run"
    _write_events(run, [
        {"event_type": "run_started", "summary": "leg"},
        {"event_type": "validation_failed", "stage_id": "stage_2x",
         "details": {"validator": "validate_params_output.py",
                     "stderr_tail": "parameter contract failed"}},
        {"event_type": "stage_completed", "stage_id": "stage_2x"},
        # Even an exact pass after completion cannot retroactively prove the
        # delivering stage re-ran its failed check.
        {"event_type": "validation_passed", "stage_id": "stage_2x",
         "details": {"validator": "validate_params_output.py"}},
    ])

    section = _stage_failures(run, [])

    assert "validate_params_output.py" in section
    assert "no passing re-check" in section
    assert "corrected attempt" not in section
    assert "caught and fixed" not in section
    assert "re-verified" not in section


def test_clean_event_log_keeps_the_clean_sentence(tmp_path):
    from render_run_report import _stage_failures

    run = tmp_path / "run"
    _write_events(run, [
        {"event_type": "run_started", "summary": "leg"},
        {"event_type": "validation_passed", "stage_id": "stage_1"},
        {"event_type": "stage_completed", "stage_id": "stage_1"},
    ])

    section = _stage_failures(run, [])

    assert "No method understanding" in section
    assert "caught and fixed" not in section


def test_catches_scoped_to_the_delivering_invocation(tmp_path):
    # A previous leg's caught failure is history, not part of this delivery.
    from render_run_report import _stage_failures

    run = tmp_path / "run"
    _write_events(run, [
        {"event_type": "run_started", "summary": "leg 1"},
        {"event_type": "validation_failed", "stage_id": "stage_1",
         "details": {"validator": "validate_paper_map.py",
                     "stderr_tail": "schema validation failed"}},
        {"event_type": "stage_completed", "stage_id": "stage_1"},
        {"event_type": "run_started", "summary": "leg 2 (delivery)"},
        {"event_type": "stage_completed", "stage_id": "stage_1"},
    ])

    section = _stage_failures(run, [])

    assert "No method understanding" in section
    assert "caught and fixed" not in section


def test_baseline_accepted_failures_are_not_catches(tmp_path):
    # The degraded row owns baseline-accepted failures; the accepting event
    # lands right after the validator's own failure event.
    from render_run_report import _stage_failures

    run = tmp_path / "run"
    _write_events(run, [
        {"event_type": "run_started", "summary": "leg"},
        {"event_type": "validation_failed", "stage_id": "stage_5",
         "details": {"validator": "validate_notebook_output.py",
                     "stderr_tail": "FAIL: 2 validation error(s):"}},
        {"event_type": "accepted_validation_failures_only",
         "stage_id": "stage_5"},
        {"event_type": "stage_completed", "stage_id": "stage_5"},
    ])

    section = _stage_failures(run, [])

    assert "caught and fixed" not in section


def test_unresolved_failure_stage_is_not_a_catch(tmp_path):
    # A failure whose stage never completed belongs to the terminal table,
    # not the catch block.
    from render_run_report import _resolved_catches

    run = tmp_path / "run"
    _write_events(run, [
        {"event_type": "run_started", "summary": "leg"},
        {"event_type": "validation_failed", "stage_id": "stage_2c",
         "details": {"validator": "validate_method_coder_output.py",
                     "stderr_tail": "FAIL: 1 validation error(s):"}},
    ])

    assert _resolved_catches(run) == []


def test_dispatch_failures_are_not_catches(tmp_path):
    # Dispatch recoveries have their own disclosure row.
    from render_run_report import _resolved_catches

    run = tmp_path / "run"
    _write_events(run, [
        {"event_type": "run_started", "summary": "leg"},
        {"event_type": "agent_dispatch_failed", "stage_id": "stage_2b"},
        {"event_type": "agent_dispatch_recovered", "stage_id": "stage_2b"},
        {"event_type": "stage_completed", "stage_id": "stage_2b"},
    ])

    assert _resolved_catches(run) == []


def test_write_run_report_handles_missing_inputs(tmp_path):
    run = tmp_path / "empty-run"
    (run / ".pipeline").mkdir(parents=True)

    assert write_run_report(run) is True
    md = (run / "REPORT.md").read_text(encoding="utf-8")

    assert "Delivery label: **not recorded yet**" in md
    assert "claims ledger was not available" in md
    assert "No automated-check report was available" in md
    assert "No method understanding" in md
    assert "not produced" in md


def test_report_uses_legacy_deferred_findings_fallback(tmp_path):
    run = tmp_path / "legacy-run"
    pipeline = run / ".pipeline"
    pipeline.mkdir(parents=True)
    _write_json(pipeline / "claims_ledger.json", {"claims": []})
    _write_text(pipeline / "deferred_findings.md",
                "# Findings deferred\n\n## Important findings (1)\n\n### F001\n\nbody\n")

    md = render_run_report(run)

    assert "`.pipeline/deferred_findings.md`" in md
    assert "1 deferred finding" in md
    assert "F001" not in md


def test_delivery_fallback_scrubs_finding_codes(tmp_path):
    run = tmp_path / "fallback-run"
    (run / ".pipeline").mkdir(parents=True)

    md = render_run_report(
        run,
        delivery={
            "label": "draft",
            "reasons": [{
                "source": "battery",
                "message": "battery failed after F001 and US-3",
            }],
        },
    )

    assert "the linked finding" in md
    assert "paper-sourced values are findable in the paper" in md
    assert "F001" not in md
    assert "US-3" not in md


def test_stage_scoped_probe_codes_are_scrubbed(tmp_path):
    """AL-S1-style ids (a stage segment between family and number) leaked
    bare into researcher-facing reports because the scrub pattern only knew
    FAMILY-NUMBER shapes (found in the 2026-07-22 design review). They must
    gloss like any other code."""
    run = tmp_path / "fallback-run"
    (run / ".pipeline").mkdir(parents=True)

    md = render_run_report(
        run,
        delivery={
            "label": "draft",
            "reasons": [{
                "source": "battery",
                "message": "constant scores adjudicated by AL-S1-1; AL-S1-2 concurs",
            }],
        },
    )

    assert "AL-S1-1" not in md
    assert "AL-S1-2" not in md
    assert "the selector returns valid, unique picks" in md
    assert "selections are not stuck in the first cluster" in md


def test_code_scrub_pattern_is_shared_and_structural():
    """The AL-S1 leak happened because the scrub pattern lived as three
    drifting copies (run report, claims report, driver notices) and only
    one was widened. One definition now owns the grammar
    (render_claims_report, the stdlib-only leaf all three import), and it
    must cover the id shapes the probe naming convention can produce —
    FAMILY[-S<stage>]-<number>[letter] — not just today's literal ids."""
    import render_claims_report
    import render_run_report as rrr
    import run_pipeline

    assert rrr._CODE_RE.pattern == render_claims_report.CODE_RE.pattern
    assert (run_pipeline._NOTICE_CODE_RE.pattern
            == render_claims_report.CODE_RE.pattern)
    assert (run_pipeline._NOTICE_FINDING_RE.pattern
            == render_claims_report.FINDING_RE.pattern)

    # Every glossed id is covered by the pattern.
    for pid in rrr.PROBE_GLOSS:
        assert rrr._CODE_RE.fullmatch(pid), f"gloss key {pid} not scrubbable"

    # Structural shapes the emitting convention can produce beyond today's
    # ids: stage segments follow pipeline stage naming (digits plus an
    # optional letter, e.g. S1, S2a), numbers may carry a variant letter.
    for pid in (
        "AL-S1-1",
        "MP-S2a-3",
        "US-S3c-2b",
        "KD-4b",
        "TSF-S2c-5",
    ):
        assert rrr._CODE_RE.fullmatch(pid), f"structural shape {pid} leaks"

    # Prose that merely resembles a code must not be scrubbed.
    for not_a_code in ("AL-S1", "S1-1", "USA-1", "AL-"):
        assert rrr._CODE_RE.fullmatch(not_a_code) is None, not_a_code


def test_notice_scrub_glosses_stage_scoped_codes():
    """The driver's researcher-notice scrubber is the third consumer of the
    scrub grammar; AL-S1-style ids leaked bare there too. Glossed ids gloss,
    unglossed ids still never render bare."""
    from run_pipeline import _scrub_notice_codes

    out = _scrub_notice_codes("blocked by AL-S1-3 and US-99")
    assert "AL-S1-3" not in out
    assert 'the "the paper\'s selection parameter has a live effect" check' in out
    assert "US-99" not in out
    assert "this verification check" in out


def test_why_this_label_renders_demote_rationale(tmp_path):
    # bev-distill 2026-07-02: the report said "draft, needs attention" while
    # the demoting findings lived only in final_manifest.json. The rationale
    # must render in the report, right under the label, with probe ids
    # glossed to plain language.
    run = tmp_path / "paper-run"
    (run / ".pipeline").mkdir(parents=True)
    delivery = {
        "label": "draft",
        "reasons": [
            {"id": "UB-6", "source": "probe", "verdict": "flag_for_researcher",
             "severity": None,
             "message": "executed metric series 'loss' is degenerate"},
            {"id": "F005", "source": "fidelity_review", "severity": "important",
             "verdict": None,
             "message": "data.py boxes format mismatches method.py mask"},
        ],
        "disclosures": [
            {"id": "KD-1", "source": "probe", "verdict": "unprobeable",
             "severity": None,
             "message": "KD-1 could not exercise the loss signature"},
        ],
    }
    text = render_run_report(run, delivery=delivery)
    assert "## Why this label" in text
    assert "hold this delivery below `verified`" in text
    assert "degenerate" in text
    assert "boxes format" in text
    assert "Paper-fidelity review, important finding" in text
    assert "disclosed for awareness" in text
    # The section leads: right after the header, before the claims section.
    assert text.index("## Why this label") < text.index("## Verification")
    # No bare probe code survives in the disclosure line.
    assert "KD-1 could not" not in text


def test_why_this_label_absent_on_verified(tmp_path):
    run = tmp_path / "paper-run"
    (run / ".pipeline").mkdir(parents=True)
    text = render_run_report(run, delivery={
        "label": "verified", "reasons": [], "disclosures": []})
    assert "## Why this label" not in text


def test_proximity_line_leads_when_one_finding(tmp_path):
    # Failure-path spec §4: state the SIZE of the gap before the findings.
    run = tmp_path / "paper-run"
    (run / ".pipeline").mkdir(parents=True)
    text = render_run_report(run, delivery={
        "label": "draft",
        "reasons": [{"id": "UB-6", "source": "probe", "verdict": "fail",
                     "message": "curve below chance"}],
        "disclosures": [],
        "probe_counts": {"pass": 12, "fail": 1},
    })
    assert "One specific finding holds this delivery below `verified`" in text
    assert "Everything else we checked passed (12 checks)." in text


def test_proximity_line_groups_demoters_sharing_one_evidence_trace(tmp_path):
    # The 2026-07-02 GBALD lesson: two findings, one root function — saying
    # so is the single highest-value sentence in the report.
    run = tmp_path / "paper-run"
    (run / ".pipeline").mkdir(parents=True)
    text = render_run_report(run, delivery={
        "label": "draft",
        "reasons": [
            {"id": "UB-7", "source": "probe", "verdict": "fail",
             "message": "term inert", "evidence": "method/ranking.py"},
            {"id": "AL-5", "source": "probe", "verdict": "fail",
             "message": "batch unresponsive", "evidence": "method/ranking.py"},
        ],
        "disclosures": [],
        "probe_counts": {"pass": 10, "fail": 2},
    })
    assert "trace to one place (`method/ranking.py`)" in text
    assert "likely a single underlying defect" in text


def test_uncertified_new_territory_report_block_kit_blocked(tmp_path):
    # The third label's front door when the family kit EXISTS but could not
    # bind (a contribution probe attempted and came back unprobeable): the
    # report must say the checks exist and could not run — never "checks
    # don't exist yet" above a probe table full of that family's rows (the
    # detr-distill 2026-07-05 incoherence). Full phrase in the header,
    # never bare "uncertified".
    run = tmp_path / "paper-run"
    pipeline = run / ".pipeline"
    pipeline.mkdir(parents=True)
    (pipeline / "probe_report.json").write_text(json.dumps({
        "target": "x",
        "verdicts": [
            {"probe_id": "UB-5", "verdict": "pass", "message": "learns"},
            {"probe_id": "US-1", "verdict": "pass", "message": "plausible"},
            {"probe_id": "KD-1", "verdict": "unprobeable",
             "message": "loss signature not found"},
        ],
    }))
    text = render_run_report(run, delivery={
        "label": "uncertified_new_territory",
        "reasons": [],
        "disclosures": [{"id": "KD-1", "source": "probe",
                         "verdict": "unprobeable",
                         "message": "loss signature not found"}],
        "probe_counts": {"pass": 2, "unprobeable": 1},
        "missing_probe_family": "knowledge_distillation.bev",
    })
    assert "uncertified — new territory" in text
    assert "## Why this label" in text
    # The truthful contribution clause, in the header line and the block.
    assert "exist, but none could run against this package" in text
    assert "don't exist yet" not in text
    assert "buildable" not in text
    assert "a fix on our side" in text
    assert "What WAS checked and passed:" in text
    # The floor renders by glossed name, not probe code.
    assert "the live training config learns a toy dataset" in text
    assert "parameter values are physically plausible" in text
    assert "disclosed for awareness" in text


def test_uncertified_new_territory_report_block_no_kit(tmp_path):
    # The genuinely-new-family case: no contribution probe ever attempted,
    # so "checks don't exist yet" is the truth and the buildable pointer
    # renders.
    run = tmp_path / "paper-run"
    pipeline = run / ".pipeline"
    pipeline.mkdir(parents=True)
    (pipeline / "probe_report.json").write_text(json.dumps({
        "target": "x",
        "verdicts": [
            {"probe_id": "UB-5", "verdict": "pass", "message": "learns"},
            {"probe_id": "US-1", "verdict": "pass", "message": "plausible"},
        ],
    }))
    text = render_run_report(run, delivery={
        "label": "uncertified_new_territory",
        "reasons": [],
        "disclosures": [],
        "probe_counts": {"pass": 2},
        "missing_probe_family": "stochastic_optimization",
    })
    assert "uncertified — new territory" in text
    assert "checks for its core contribution don't exist yet" in text
    assert "buildable" in text
    assert "What WAS checked and passed:" in text


@pytest.mark.parametrize(
    ("probe_ref", "element_ids", "expected_gap"),
    [
        (
            "claims.wrong_ref",
            ["paper-contribution"],
            "declares no such verification ref",
        ),
        (
            "claims.contribution_floor",
            ["not-in-the-contract"],
            "names element(s) the methodology contract does not carry",
        ),
    ],
)
def test_passing_but_unbound_contribution_is_visible_in_report(
    tmp_path, probe_ref, element_ids, expected_gap,
):
    run = tmp_path / "paper-run"
    pipeline = run / ".pipeline"
    pipeline.mkdir(parents=True)
    report = {"target": "x", "verdicts": [{
        "probe_id": "CT-1",
        "probe_ref": probe_ref,
        "verdict": "pass",
        "message": "the contribution behavior passed",
        "element_ids": element_ids,
    }]}
    spec = {"methodology_replication_contract": {"elements": [{
        "element_id": "contribution",
        "role": "core_methodology",
        "replication_status": "must_replicate",
        "technical_concept": "contribution",
        "required_behavior": "the contribution changes the result",
        "paper_section": "Section 3",
        "acceptable_approximations": [],
        "forbidden_substitutions": [],
        "paper_element_ids": ["paper-contribution"],
        "verification_probe_refs": ["claims.contribution_floor"],
    }]}}
    _write_json(pipeline / "probe_report.json", report)
    delivery = derive_delivery_label(
        report, None, paradigm="active_learning", spec=spec,
    )

    text = render_run_report(run, delivery=delivery)

    assert delivery["label"] == "uncertified_new_territory"
    assert "exist and ran" in text
    assert "don't exist yet" not in text
    assert expected_gap in text


def test_not_applicable_contribution_reads_as_a_completed_family_check(
    tmp_path,
):
    run = tmp_path / "paper-run"
    pipeline = run / ".pipeline"
    pipeline.mkdir(parents=True)
    report = {"target": "x", "verdicts": [{
        "probe_id": "AL-1",
        "verdict": "not_applicable",
        "message": "this selector has no stochastic acquisition path",
    }]}
    _write_json(pipeline / "probe_report.json", report)
    delivery = derive_delivery_label(
        report, None, paradigm="active_learning",
    )

    text = render_run_report(run, delivery=delivery)

    assert delivery["label"] == "uncertified_new_territory"
    assert "exist and completed" in text
    assert "completed conditional result" in text
    assert "don't exist yet" not in text
    assert "this selector has no stochastic acquisition path" in text


# ---------------------------------------------------------------------------
# Stopped-run halt block (failure-path spec §3, landed 2026-07-04)
# ---------------------------------------------------------------------------

class _Stage:
    def __init__(self, stage_id, status, notes=""):
        self.stage_id = stage_id
        self.status = status
        self.notes = notes


def test_halt_block_leads_with_four_plain_language_parts(tmp_path):
    run = tmp_path / "halted-run"
    pipeline = run / ".pipeline"
    pipeline.mkdir(parents=True)
    _write_json(pipeline / "stage_2x.halt", {
        "status": "halted", "stage": "stage_2x",
        "reason": "validate_params_output failed after cap",
        "user_message": "The paper's stated learning rate is outside any "
                        "plausible range. Confirm the value in Section 4.2, "
                        "correct params.json, and re-run.",
    })
    _write_text(run / "METHOD.md", "# Method\n")
    _write_json(pipeline / "paper_map.json", {"sections": []})

    report = render_run_report(
        run, stage_results=[
            _Stage("stage_2b", "completed"),
            _Stage("stage_2x", "halted", "validator failed at cap"),
        ])

    assert "## This run stopped — it needs a decision" in report
    # All four mandatory parts, in plain language.
    assert "**What happened.**" in report
    assert "deriving the paper's parameter values" in report
    assert "**Why we stopped instead of guessing.**" in report
    assert "outside any plausible range" in report
    assert "(Failure area: code generation.)" in report
    assert "**What you still have.**" in report
    assert "METHOD.md" in report and "paper_map.json" in report
    assert "**What to do next.**" in report
    assert "`/r2c-run halted-run`" in report
    # No stage id in the headline story; the technical detail keeps it.
    story = report.split("<details>")[0]
    assert "stage_2x" not in story.split("## This run stopped")[1]
    # The block leads: before the probe/issue sections.
    assert report.index("## This run stopped") < report.index("## Issues To Review")


def test_halt_block_engineering_variant_without_user_message(tmp_path):
    run = tmp_path / "halted-run"
    pipeline = run / ".pipeline"
    pipeline.mkdir(parents=True)
    _write_json(pipeline / "stage_3c.halt", {
        "status": "halted", "stage": "stage_3c",
        "reason": "smoke gate setup error (exit 2)",
    })
    report = render_run_report(
        run, stage_results=[_Stage("stage_3c", "halted", "setup error")])
    assert "## This run stopped — it needs a decision" in report
    assert "internal error" in report
    assert "Resume the run first" in report
    assert "`/r2c-run halted-run`" in report
    assert "report it with the technical detail below attached" in report
    assert "running the demo notebook end to end" in report


def test_halt_block_absent_on_clean_run_with_stale_artifact(tmp_path):
    run = tmp_path / "clean-run"
    pipeline = run / ".pipeline"
    pipeline.mkdir(parents=True)
    # A leftover halt artifact from a pre-resume attempt must not resurrect
    # the block once the run delivers cleanly.
    _write_json(pipeline / "stage_2x.halt", {
        "status": "halted", "stage": "stage_2x", "reason": "old halt",
    })
    report = render_run_report(
        run, stage_results=[
            _Stage("stage_2x", "completed"),
            _Stage("stage_5", "completed"),
        ])
    assert "## This run stopped" not in report


def test_terminal_scope_gap_surfaces_recorded_action_without_resume(
    tmp_path,
):
    """RFC-shaped terminal gaps are coverage decisions, not retry advice."""
    run = tmp_path / "protocol-spec"
    pipeline = run / ".pipeline"
    pipeline.mkdir(parents=True)
    _write_json(pipeline / "paper_map.json", {
        "title": "Protocol Specification",
        "elements": [
            {
                "id": "concept-request",
                "type": "concept",
                "name": "Request semantics",
                "section": "Section 2",
                "code_role": "implement",
                "description": "Defines a safe request method.",
            },
            {
                "id": "prop-safe",
                "type": "property",
                "name": "Safety property",
                "section": "Section 2",
                "code_role": "demonstrate",
                "description": "The request does not mutate server state.",
            },
        ],
    })
    _write_text(run / "METHOD.md", generate_method_md(run))
    recorded_action = "Reject this input from the current method pipeline."
    _write_json(pipeline / "stage_1.halt", {
        "status": "halted",
        "stage": "stage_1",
        "reason": "analyzer halted after understanding the protocol",
        # Legacy artifacts can predate halt_class while carrying the complete
        # structured gap context.
        # Pre-change artifacts can retain this stale user message. Structured
        # context must win when the report is regenerated.
        "user_message": (
            "R2C supports active learning, knowledge distillation, domain "
            "adaptation, and motion planning. Resume after choosing one."
        ),
        "context": {"paradigm_gap_report": {
            "decision": "unsupported_or_unclear",
            "paper_paradigm_summary": "An HTTP protocol specification.",
            "registered_paradigms": [
                "vision_transformer", "stochastic_optimization",
            ],
            "recommended_next_action": recorded_action,
        }},
    })

    report = render_run_report(
        run,
        delivery={"label": "explanation_only"},
        stage_results=[_Stage("stage_1", "halted", "paradigm mismatch")],
    )
    stopped = report.split("## This run stopped", 1)[1].split(
        "<details>", 1
    )[0]

    assert "The document was understood" in stopped
    assert "coverage or routing" in stopped
    assert recorded_action in stopped
    assert "do not resume the unchanged run" in stopped
    assert "/r2c-run" not in stopped
    assert "active learning, knowledge distillation" not in stopped
    assert "a decomposition source index" in stopped
    assert (
        "substantive plain-language method explanation was not produced"
        in stopped
    )
    assert "decomposition source index produced" in report
    assert "produced; no pending explanation markers found" not in report


def test_legacy_placeholder_only_method_is_recognized_as_index(tmp_path):
    """Re-rendering an existing pre-marker RFC artifact fixes its front door."""
    run = tmp_path / "legacy-protocol"
    (run / ".pipeline").mkdir(parents=True)
    _write_text(run / "METHOD.md", """# Protocol — the method, explained

*(method summary unavailable — no spec and no algorithm element)*

*(the paper map records no implement/demonstrate equations — see the source map below)*

*(no algorithm element in the paper map — see the equation sections above)*

## Source map

| element | type |
|---|---|
| request semantics | concept |
""")

    report = render_run_report(run, delivery={"label": "draft"})

    assert "decomposition source index; substantive method explanation unavailable" in report
    assert "decomposition source index produced" in report
    assert "the method explained" not in report
    assert "produced; no pending explanation markers found" not in report


def test_empty_method_file_is_not_advertised_as_an_explanation(tmp_path):
    run = tmp_path / "empty-method"
    (run / ".pipeline").mkdir(parents=True)
    _write_text(run / "METHOD.md", "  \n")

    report = render_run_report(run, delivery={"label": "draft"})

    assert "file is empty; substantive method explanation unavailable" in report
    assert "the method explained" not in report
    assert "produced; no pending explanation markers found" not in report


def test_substantive_method_without_pending_markers_keeps_complete_status(
    tmp_path,
):
    run = tmp_path / "supported-method"
    (run / ".pipeline").mkdir(parents=True)
    _write_text(
        run / "METHOD.md",
        "# Supported method — the method, explained\n\n"
        "The algorithm repeatedly updates the selected batch.\n",
    )

    report = render_run_report(run, delivery={"label": "draft"})

    assert "the method explained" in report
    assert "produced; no pending explanation markers found" in report


def test_new_territory_block_qualifies_floor_on_universal_gap(tmp_path):
    """Report-side twin of the banner test: a universal-tier check that
    could not run qualifies the floor sentence instead of overclaiming."""
    run = tmp_path / "paper-run"
    (run / ".pipeline").mkdir(parents=True)
    _write_json(run / ".pipeline" / "probe_report.json", {
        "target": "x",
        "verdicts": [
            {"probe_id": "US-1", "verdict": "pass", "message": "plausible"},
            {"probe_id": "UB-6", "verdict": "unprobeable",
             "message": "notebook has no executed outputs"},
        ],
    })
    text = render_run_report(run, delivery={
        "label": "uncertified_new_territory",
        "reasons": [],
        "disclosures": [{"id": "UB-6", "source": "probe",
                         "verdict": "unprobeable",
                         "message": "notebook has no executed outputs"}],
        "probe_counts": {"pass": 1, "unprobeable": 1},
        "missing_probe_family": "knowledge_distillation.bev",
    })
    assert "Every universal check passed." not in text
    assert "could run passed" in text
    assert "1 universal check could not run" in text


def test_recovered_dispatches_disclosed_in_issue_summary(tmp_path):
    """Item 8 (recovery ladder): recovered transport failures surface as a
    disclosure row so an unattended run's operator sees the flap without
    reading the event log. Zero recoveries adds no row."""
    from render_run_report import _count_recovered_dispatches, _issue_summary

    run_dir = tmp_path / "run"
    (run_dir / ".pipeline").mkdir(parents=True)
    assert _count_recovered_dispatches(run_dir) == 0
    assert "Model-call recoveries" not in _issue_summary(run_dir)

    events = [
        {"event_type": "agent_dispatch_failed", "summary": "x"},
        {"event_type": "agent_dispatch_recovered", "summary": "rung 3"},
        {"event_type": "stage_completed", "summary": "y"},
    ]
    (run_dir / ".pipeline" / "run_events.jsonl").write_text(
        "\n".join(json.dumps(e) for e in events) + "\n")
    assert _count_recovered_dispatches(run_dir) == 1
    summary = _issue_summary(run_dir)
    assert "Model-call recoveries" in summary
    assert "1 model call failed in transport" in summary
    assert "disclosure only" in summary
    # Doubling as the Block 8 legacy pin: this log has no `run_started`
    # boundary, so the count conservatively covers the whole file.


def test_recovery_count_scoped_to_delivering_invocation(tmp_path):
    """Block 8 (SRL 2026-07-15): REPORT counted a July 13 invocation's
    transport recovery in the July 15 delivery because the counter scanned
    the whole append-only event log. The count is now scoped to the last
    `run_started` boundary; earlier legs' recoveries remain in the event
    log (history preservation) without being presented as current."""
    from render_run_report import _count_recovered_dispatches, _issue_summary

    run_dir = tmp_path / "run"
    (run_dir / ".pipeline").mkdir(parents=True)
    events_path = run_dir / ".pipeline" / "run_events.jsonl"

    # Leg 1 halts after a recovered transport call; leg 2 delivers clean.
    legs = [
        {"event_type": "run_started", "summary": "leg 1"},
        {"event_type": "agent_dispatch_recovered", "summary": "rung 3"},
        {"event_type": "run_finished", "summary": "halted"},
        {"event_type": "run_started", "summary": "leg 2 (delivery)"},
        {"event_type": "stage_completed", "summary": "smoke"},
    ]
    original = "\n".join(json.dumps(e) for e in legs) + "\n"
    events_path.write_text(original)
    assert _count_recovered_dispatches(run_dir) == 0
    assert "Model-call recoveries" not in _issue_summary(run_dir)
    # Rendering never mutates the append-only evidence.
    assert events_path.read_text() == original

    # A recovery inside the delivering leg still surfaces: the disclosure
    # keeps its job; only cross-invocation attribution is gone.
    with events_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"event_type": "agent_dispatch_recovered",
                            "summary": "rung 1"}) + "\n")
    assert _count_recovered_dispatches(run_dir) == 1
    summary = _issue_summary(run_dir)
    assert "1 model call failed in transport" in summary


def test_math_sanity_disclosure_names_claim_free_coverage(tmp_path):
    """ADAM 2026-07-14 blind spot: entries with no checkable claims now
    surface on the disclosure line, with the coverage ratio."""
    from render_run_report import _not_checkable_disclosure

    run_dir = tmp_path / "run"
    (run_dir / ".pipeline").mkdir(parents=True)
    sidecar = run_dir / ".pipeline" / "method_explanations.json"

    import json as _json
    sidecar.write_text(_json.dumps({"math_sanity": {
        "refuted": 0, "consistent": 4, "not_checkable": 2, "dropped": 0,
        "entries_total": 10, "entries_examined": 10, "entries_claim_free": 6,
    }}), encoding="utf-8")
    text = _not_checkable_disclosure(run_dir)
    assert "2 mechanism claims were not machine-checkable" in text
    assert "nothing to verify in 6 of 10 explanation sections" in text
    assert "disclosure only, not a defect" in text

    # Full coverage, everything checkable: no disclosure at all.
    sidecar.write_text(_json.dumps({"math_sanity": {
        "refuted": 0, "consistent": 4, "not_checkable": 0, "dropped": 0,
        "entries_total": 4, "entries_examined": 4, "entries_claim_free": 0,
    }}), encoding="utf-8")
    assert _not_checkable_disclosure(run_dir) == ""

    # Pre-coverage totals (old runs): the old wording still renders.
    sidecar.write_text(_json.dumps({"math_sanity": {
        "refuted": 0, "consistent": 1, "not_checkable": 1, "dropped": 0,
    }}), encoding="utf-8")
    assert "1 mechanism claim was not machine-checkable" in _not_checkable_disclosure(run_dir)


def test_label_scale_line_shows_rank(tmp_path):
    """Researcher feedback 2026-07: a bare label gave no sense of where the run sits
    on the overall scale (he guessed "3 out of 5"). Every known label
    renders the full ladder with this run's position; unknown or missing
    labels render no scale line."""
    run = tmp_path / "run"
    (run / ".pipeline").mkdir(parents=True)

    md = render_run_report(run, delivery={"label": "draft"})
    assert "Label scale, least to most reliable:" in md
    assert "This run sits at 3 of 4." in md
    assert "**draft**" in md

    md = render_run_report(run, delivery={"label": "verified",
                                          "reasons": [], "disclosures": []})
    assert "This run sits at 4 of 4." in md

    md = render_run_report(run, delivery={"label": "explanation_only"})
    assert "This run sits at 1 of 4." in md

    md = render_run_report(run)
    assert "Label scale" not in md


def test_adoption_row_names_each_quote_surface(tmp_path):
    """The quote_adoptions.json sidecar is shared by two surfaces: equation
    quotes (no surface key) and method-spec quotes (surface="method_spec").
    The disclosure row's wording follows the rows' actual surfaces instead
    of calling every adoption an equation quote (the maintainer's 2026-08-03 call;
    the SRL 2026-07-29 delivery shipped a method-spec adoption under the
    equation wording)."""
    from render_run_report import _issue_summary

    run_dir = tmp_path / "run"
    (run_dir / ".pipeline").mkdir(parents=True)
    sidecar = run_dir / ".pipeline" / "quote_adoptions.json"

    # Equation-only adoptions keep the equation wording, with no
    # method-spec mention.
    sidecar.write_text(json.dumps([
        {"element_id": "eq-a", "similarity": 0.99},
    ]))
    summary = _issue_summary(run_dir)
    assert "1 equation quote replaced" in summary
    assert "method-spec" not in summary

    # A mixed sidecar names both surfaces with their own counts and still
    # lists every adoption.
    sidecar.write_text(json.dumps([
        {"element_id": "eq-a", "similarity": 0.99},
        {"element_id": "param-r:meaning_quote", "similarity": 0.9655,
         "surface": "method_spec"},
        {"element_id": "param-v:meaning_quote", "similarity": 0.97,
         "surface": "method_spec"},
    ]))
    summary = _issue_summary(run_dir)
    assert "1 equation quote and 2 method-spec quotes replaced" in summary
    assert "`param-r:meaning_quote` at similarity 0.9655" in summary

    # Spec-only adoptions never claim an equation surface.
    sidecar.write_text(json.dumps([
        {"element_id": "param-r:meaning_quote", "similarity": 0.9655,
         "surface": "method_spec"},
    ]))
    summary = _issue_summary(run_dir)
    assert "1 method-spec quote replaced" in summary
    assert "equation" not in summary
