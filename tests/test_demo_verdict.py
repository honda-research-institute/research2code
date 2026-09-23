"""Demo-success verdict + fix-loop data-signal check (design approved 2026-07-16).

The reusable failure class under test: **demo_failure_invisible_to_smoke** —
a notebook executes cleanly while its headline demonstration visibly fails in
its own printed output, and (before this tranche) no pipeline surface named
that. Acceptance is fixture-first: the reconstructed 2026-07-14 shapes under
tests/fixtures/zoo/ must yield `failed` plus the first-class demoter, the
adjacent-good shapes must yield `succeeded` with ZERO added demotion, and the
no-markers family yields `undetermined` (plus the kit-coverage finding we own
on a committed family). Probes are unchanged — where the verdict and UB-6
disagree, the delivery records a probe-bug disclosure.

The data-signal side pins the Rethinking 2026-07-14 burn: label-free/random
fix data fails the check, class-conditional separated synthetic data passes,
and the driver injects the failure into the NEXT fix dispatch's prompt.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import taxonomy
from demo_verdict import (DEMO_EVALUATION_PREFIX, DEMO_VERDICT_JSON,
                          derive_demo_verdict, headline_section_cells,
                          record_demo_verdict)
from delivery_label import derive_delivery_label
from scripts.build_plan import TSF_TARGET_SCALING
from scripts.target_scaling_post_smoke import STATE_ARTIFACT, STATE_MARKER
from scripts.time_series_target_scaling import (
    fit_target_scaling_state,
    resolve_fitting_target_range,
)
from tests.helpers.assertions import assert_dispatched, assert_stage_completed
from tests.helpers.state import make_state

ZOO = Path(__file__).resolve().parent / "fixtures" / "zoo"


def _zoo_notebook(scenario: str) -> dict:
    return json.loads(
        (ZOO / scenario / "notebook.ipynb").read_text(encoding="utf-8"))


def _mp_context() -> tuple[dict, dict]:
    return (taxonomy.load_demo_success("motion_planning"),
            taxonomy.load_notebook_layout("motion_planning"))


def _al_context() -> tuple[dict, dict]:
    return (taxonomy.load_demo_success("active_learning"),
            taxonomy.load_notebook_layout("active_learning"))


# ---------------------------------------------------------------------------
# Zoo acceptance — the recorded 2026-07-14 shapes + adjacent-good guards
# ---------------------------------------------------------------------------


def test_idbrrt_timeout_shape_yields_failed_with_quoted_evidence():
    ds, layout = _mp_context()
    v = derive_demo_verdict(
        _zoo_notebook("idbrrt-demo-timeout-reconstructed"), ds, layout,
        paradigm="motion_planning", family_kit_committed=True)
    assert v["verdict"] == "failed"
    assert v["decided_by"] == "failure_marker"
    assert "Planning status: timeout" in v["evidence_line"]
    assert isinstance(v["evidence_cell"], int)


def test_icra_goal_miss_shape_yields_failed():
    ds, layout = _mp_context()
    v = derive_demo_verdict(
        _zoo_notebook("icra-demo-goal-miss-reconstructed"), ds, layout,
        paradigm="motion_planning", family_kit_committed=True)
    assert v["verdict"] == "failed"
    assert "Status: timeout" in v["evidence_line"]


def test_icra_collision_shape_yields_failed():
    # Third recorded MP evidence shape (ICRA acceptance run 2026-07-20):
    # the rollout collides and says so, but no PlanResult status line is
    # printed anywhere, so the pre-fix vocabulary left this undetermined.
    ds, layout = _mp_context()
    v = derive_demo_verdict(
        _zoo_notebook("icra-demo-collision-reconstructed"), ds, layout,
        paradigm="motion_planning", family_kit_committed=True)
    assert v["verdict"] == "failed"
    assert v["decided_by"] == "failure_marker"
    assert "Collision at step 29" in v["evidence_line"]


def test_mp_success_shape_yields_succeeded_adjacent_good():
    ds, layout = _mp_context()
    v = derive_demo_verdict(
        _zoo_notebook("mp-demo-success-reconstructed"), ds, layout,
        paradigm="motion_planning", family_kit_committed=True)
    assert v["verdict"] == "succeeded"
    assert v["decided_by"] == "success_marker"
    assert "success" in v["evidence_line"].lower()
    assert v["kit_coverage_finding"] is None


def test_chance_flat_shape_yields_failed_via_beats_chance():
    ds, layout = _al_context()
    v = derive_demo_verdict(
        _zoo_notebook("demo-chance-flat-reconstructed"), ds, layout,
        paradigm="active_learning", family_kit_committed=True)
    assert v["verdict"] == "failed"
    assert v["decided_by"] == "beats_chance"
    assert "chance" in v["evidence_line"]


def test_beats_chance_shape_yields_succeeded_adjacent_good():
    ds, layout = _al_context()
    v = derive_demo_verdict(
        _zoo_notebook("demo-beats-chance-reconstructed"), ds, layout,
        paradigm="active_learning", family_kit_committed=True)
    assert v["verdict"] == "succeeded"
    assert v["decided_by"] == "beats_chance"


def test_family_without_markers_yields_undetermined_and_committed_kit_finding():
    nb = _zoo_notebook("idbrrt-demo-timeout-reconstructed")
    _, layout = _mp_context()
    committed = derive_demo_verdict(
        nb, {}, layout, paradigm="knowledge_distillation",
        family_kit_committed=True)
    assert committed["verdict"] == "undetermined"
    assert committed["markers_declared"] is False
    finding = committed["kit_coverage_finding"]
    assert finding and finding["id"] == "demo_markers_missing"
    assert "coverage gap we own" in finding["message"]

    provisional = derive_demo_verdict(
        nb, {}, layout, paradigm="new_gap_family",
        family_kit_committed=False)
    assert provisional["verdict"] == "undetermined"
    assert provisional["kit_coverage_finding"] is None


# ---------------------------------------------------------------------------
# Verdict engine unit behavior
# ---------------------------------------------------------------------------


def _tiny_nb(headline_stdout: str | None, *, extra_cells: list | None = None):
    cells = [
        {"cell_type": "markdown", "source": "## 4. The method"},
        {"cell_type": "code", "source": "x=1",
         "outputs": [{"output_type": "stream",
                      "text": "Planning status: timeout (component demo)\n"}]},
        {"cell_type": "markdown",
         "source": "## 5. Running the planner end-to-end"},
    ]
    if headline_stdout is not None:
        cells.append({
            "cell_type": "code", "source": "plan()",
            "outputs": ([{"output_type": "stream", "text": headline_stdout}]
                        if headline_stdout else []),
        })
    cells.extend(extra_cells or [])
    return {"cells": cells}


_TINY_LAYOUT = {"sections": [
    {"id": "title", "title": "Paper title"},
    {"id": "running", "title": "## 5. Running the planner end-to-end"},
]}

_TINY_MARKERS = {
    "failure_markers": [
        {"pattern": r"(?im)^.*\bstatus\s*:\s*timeout.*$",
         "gloss": "the planner reports timeout"}],
    "success_markers": [
        {"pattern": r"(?im)^.*\bstatus\s*:\s*success.*$",
         "gloss": "the planner reports success"}],
}


def _tsf_context() -> tuple[dict, dict]:
    return (taxonomy.load_demo_success("time_series_forecasting"),
            taxonomy.load_notebook_layout("time_series_forecasting"))


def test_tsf_nan_metrics_shape_yields_failed():
    """The 2026-08-06 pdfgnn loop2 shape (R2C-070's shaping known-bad): the
    executed evaluation printed RMSE/MAE/WMAPE = nan over a held-out window
    with no ground truth, and the roll's verdict was vacuously undetermined
    because the run-authored pack declared no markers. The committed TE-TSF
    node's metric-NaN failure marker must decide it — with no dependence on
    the notebook printing a Demo verdict line."""
    ds, layout = _tsf_context()
    v = derive_demo_verdict(
        _zoo_notebook("tsf-demo-nan-metrics-reconstructed"), ds, layout,
        paradigm="time_series_forecasting", family_kit_committed=True)
    assert v["verdict"] == "failed"
    assert v["decided_by"] == "failure_marker"
    assert "nan" in v["evidence_line"].lower()


def test_tsf_pass_shape_yields_succeeded_adjacent_good():
    """The adjacent-good twin: finite held-out metrics, baselines printed
    beside the model, and the deterministic `Demo verdict: PASS` line the
    TE-TSF notebook layout's evaluation subsection specifies."""
    ds, layout = _tsf_context()
    v = derive_demo_verdict(
        _zoo_notebook("tsf-demo-pass-reconstructed"), ds, layout,
        paradigm="time_series_forecasting", family_kit_committed=True)
    assert v["verdict"] == "succeeded"
    assert v["decided_by"] == "success_marker"
    assert "Demo verdict: PASS" in v["evidence_line"]
    assert v["kit_coverage_finding"] is None


def test_tsf_nan_metric_outranks_a_pass_line():
    """Failure markers evaluate first, so a notebook that prints a PASS
    verdict line while a headline metric is NaN still fails."""
    ds, layout = _tsf_context()
    nb = {"cells": [
        {"cell_type": "markdown",
         "source": "## 5. Training and forecasting end-to-end"},
        {"cell_type": "code", "source": "evaluate()", "outputs": [
            {"output_type": "stream", "name": "stdout",
             "text": "RMSE: nan\nDemo verdict: PASS — oops\n"}]},
        {"cell_type": "markdown", "source": "## 6. Use your own data"},
    ]}
    v = derive_demo_verdict(
        nb, ds, layout,
        paradigm="time_series_forecasting", family_kit_committed=True)
    assert v["verdict"] == "failed"
    assert v["decided_by"] == "failure_marker"


def _structured_tsf_record(*, model: float, repeat: float) -> tuple[dict, dict]:
    from demo_skill_evidence import (compute_actual_values_id,
                                     compute_finite_mask_id)

    rows = ["series-a@8"]
    actuals = [10.0]
    mask = [True]
    positions = [8]
    units = "target_units"
    actual_id = compute_actual_values_id(rows, actuals, units)
    mask_id = compute_finite_mask_id(rows, mask)
    common = {
        "row_ids": rows,
        "evaluation_positions": positions,
        "units": units,
        "actual_values_id": actual_id,
        "finite_mask_id": mask_id,
    }
    record = {
        "schema_version": "1.0.0",
        "target_root": "series",
        "model_id": "model",
        "metric_id": "rmse",
        "units": units,
        "aggregation": "root_mean_squared_error",
        "row_ids": rows,
        "evaluation_positions": positions,
        "actuals": actuals,
        "finite_mask": mask,
        "model_predictions": [model],
        "model_metric": abs(10.0 - model),
        "actual_values_id": actual_id,
        "finite_mask_id": mask_id,
        "comparators": {
            "predict_zero": {
                **common,
                "implementation": "predict_zero",
                "predictions": [0.0],
                "metric": 10.0,
                "input_positions": [],
            },
            "repeat_last": {
                **common,
                "implementation": "repeat_last_pre_window",
                "predictions": [repeat],
                "metric": abs(10.0 - repeat),
                "input_values": [repeat],
                "input_positions": [7],
            },
        },
    }
    receipt = {
        "status": "valid",
        **common,
        "target_root": "series",
        "model_id": "model",
    }
    return record, receipt


def _scaling_split_row() -> dict:
    return {
        "root": "batch.targets",
        "model_id": "model",
        "protocol_range": {"start": 0, "stop": 3},
        "subset": None,
        "certainty": "exact",
    }


def _attach_target_scaling_proof(record: dict) -> tuple[str, dict]:
    split_receipt = {
        "schema_version": "1.0.0",
        "status": "valid",
        "validator": "eval_split_lineage",
        "reasons": [],
        "evidence": {"fitting_target_ranges": [_scaling_split_row()]},
    }
    state = fit_target_scaling_state(
        TSF_TARGET_SCALING,
        entity_ids=["series-a"],
        targets=np.asarray([[1.0, 2.0, 3.0, 1.0e9]]),
        fitting_range=resolve_fitting_target_range(
            split_receipt,
            target_root="batch.targets",
            model_id="model",
        ),
    )
    original_actual = float(record["actuals"][0])
    original_prediction = float(record["model_predictions"][0])
    record["target_scaling_proof"] = {
        "schema_version": "1.0.0",
        "state_digest": state["state_digest"],
        "entity_ids": ["series-a"],
        "entity_id_root": "evaluation.entity_ids",
        "scaled_actuals": [[original_actual / 2.0]],
        "original_actuals": [[original_actual]],
        "actual_output_role": "location",
        "actual_entity_axis": 0,
        "scaled_predictions": [[original_prediction / 2.0]],
        "original_predictions": [[original_prediction]],
        "prediction_output_role": "location",
        "prediction_entity_axis": 0,
    }
    marker = STATE_MARKER + json.dumps(
        state,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
    return marker, state


def _trusted_tsf_split_receipt() -> dict:
    return {
        "schema_version": "1.0.0",
        "status": "valid",
        "validator": "eval_split_lineage",
        "reasons": ["fitting_to_reported_evaluation_disjointness_proved"],
        "evidence": {
            "fitting_target_ranges": [_scaling_split_row()],
            "fitting_to_reported_evaluation_relations": [{
                "model_id": "model",
                "root": "series",
                "first": {
                    "role": "fitting",
                    "read_kind": "target",
                    "protocol_range": {"start": 0, "stop": 8},
                },
                "second": {
                    "role": "reported_evaluation",
                    "read_kind": "target",
                    "protocol_range": {"start": 8, "stop": 9},
                },
                "status": "disjoint",
            }],
        },
    }


def _write_tsf_record_notebook(
    run_dir: Path,
    record: dict,
    *extra_output_lines: str,
) -> None:
    pipeline = run_dir / ".pipeline"
    pipeline.mkdir(parents=True, exist_ok=True)
    notebook = {"cells": [
        {
            "cell_type": "markdown",
            "source": "## 5. Training and forecasting end-to-end",
        },
        {
            "cell_type": "code",
            "source": "evaluate()",
            "outputs": [{
                "output_type": "stream",
                "name": "stdout",
                "text": [
                    DEMO_EVALUATION_PREFIX + json.dumps(record) + "\n",
                    *(line + "\n" for line in extra_output_lines),
                    "Demo verdict: PASS — derived presentation\n",
                ],
            }],
        },
    ]}
    (run_dir / "notebook.ipynb").write_text(json.dumps(notebook))
    (pipeline / "method_spec.json").write_text(json.dumps({
        "comparison": {
            "classification": {"id": "time_series_forecasting"},
        },
    }))


def test_tsf_structured_loss_outranks_fake_printed_pass():
    import taxonomy

    ds, layout = _tsf_context()
    record, receipt = _structured_tsf_record(model=9.0, repeat=10.0)
    nb = {"cells": [
        {"cell_type": "markdown",
         "source": "## 5. Training and forecasting end-to-end"},
        {"cell_type": "code", "source": "evaluate()", "outputs": [{
            "output_type": "stream", "name": "stdout",
            "text": "Demo verdict: PASS — fake prose\n",
        }]},
    ]}

    verdict = derive_demo_verdict(
        nb, ds, layout,
        paradigm="time_series_forecasting",
        family_kit_committed=True,
        demo_skill=taxonomy.load_demo_skill("time_series_forecasting"),
        executed_evaluation=record,
        evaluation_validity=receipt,
        evaluation_record_cell=1,
    )

    assert verdict["verdict"] == "failed"
    assert verdict["decided_by"] == "structured_demo_skill"
    assert verdict["evidence_status"]["execution"]["status"] == "completed"
    assert verdict["evidence_status"]["evaluation_validity"]["status"] == "valid"
    assert verdict["evidence_status"]["mechanism"]["status"] == "undetermined"
    assert verdict["evidence_status"]["skill"]["status"] == "not_demonstrated"
    assert verdict["evidence_status"]["paper_benchmark"]["status"] == "not_assessed"
    assert verdict["presentation_marker"]["line"].startswith("Demo verdict: PASS")


def test_tsf_invalid_evaluation_keeps_execution_but_cannot_yield_skill_pass():
    import taxonomy

    ds, layout = _tsf_context()
    record, _ = _structured_tsf_record(model=10.0, repeat=5.0)
    nb = {"cells": [
        {"cell_type": "markdown",
         "source": "## 5. Training and forecasting end-to-end"},
        {"cell_type": "code", "source": "evaluate()", "outputs": [{
            "output_type": "stream", "name": "stdout",
            "text": "Demo verdict: PASS — fake prose\n",
        }]},
    ]}

    verdict = derive_demo_verdict(
        nb, ds, layout,
        paradigm="time_series_forecasting",
        family_kit_committed=True,
        demo_skill=taxonomy.load_demo_skill("time_series_forecasting"),
        executed_evaluation=record,
        evaluation_validity={
            "status": "invalid",
            "reasons": ["exact_temporal_target_overlap"],
        },
    )

    assert verdict["verdict"] == "undetermined"
    assert verdict["evidence_status"]["execution"]["status"] == "completed"
    assert verdict["evidence_status"]["evaluation_validity"]["status"] == "invalid"
    assert verdict["evidence_status"]["skill"]["status"] == "undetermined"


def test_verdict_reads_headline_section_only():
    """The §4 cell prints a failure line; the headline §5 prints success —
    per the maintainer's 2026-07-16 decision the verdict reads the headline only."""
    v = derive_demo_verdict(
        _tiny_nb("Planning status: success\n"), _TINY_MARKERS, _TINY_LAYOUT,
        paradigm="motion_planning", family_kit_committed=True)
    assert v["verdict"] == "succeeded"


def test_failure_marker_wins_over_success_marker():
    v = derive_demo_verdict(
        _tiny_nb("Planning status: timeout\nRetry status: success\n"),
        _TINY_MARKERS, _TINY_LAYOUT,
        paradigm="motion_planning", family_kit_committed=True)
    assert v["verdict"] == "failed"


def test_markers_declared_but_unmatched_is_undetermined_without_kit_finding():
    v = derive_demo_verdict(
        _tiny_nb("nothing recognizable here\n"), _TINY_MARKERS, _TINY_LAYOUT,
        paradigm="motion_planning", family_kit_committed=True)
    assert v["verdict"] == "undetermined"
    assert "none matched" in v["reason"]
    assert v["kit_coverage_finding"] is None


def test_layout_without_sections_is_kit_gap_on_committed_family():
    """vision_transformer today: beats_chance declared, no notebook_layout —
    a kit-side coverage gap we own, not a silent disclosure."""
    v = derive_demo_verdict(
        _tiny_nb("Planning status: success\n"), _TINY_MARKERS, {},
        paradigm="vision_transformer", family_kit_committed=True)
    assert v["verdict"] == "undetermined"
    assert v["kit_coverage_finding"] is not None
    assert "notebook_layout" in v["kit_coverage_finding"]["message"]


def test_notebook_missing_headline_section_is_disclosure_not_kit_finding():
    nb = {"cells": [{"cell_type": "markdown", "source": "## 1. Setup"}]}
    v = derive_demo_verdict(
        nb, _TINY_MARKERS, _TINY_LAYOUT,
        paradigm="motion_planning", family_kit_committed=True)
    assert v["verdict"] == "undetermined"
    assert "no code cells" in v["reason"]
    assert v["kit_coverage_finding"] is None


def test_headline_section_without_outputs_is_undetermined():
    v = derive_demo_verdict(
        _tiny_nb(""), _TINY_MARKERS, _TINY_LAYOUT,
        paradigm="motion_planning", family_kit_committed=True)
    assert v["verdict"] == "undetermined"
    assert "no executed outputs" in v["reason"]


def test_single_metric_point_is_undetermined_in_both_directions():
    """Symmetric two-point floor: one low print never demotes AND one high
    print never certifies — the shared accuracy regex also matches config
    echoes ("target accuracy: 0.95"), so a single matched number is too weak
    evidence either way."""
    ds = {"checks": [{"kind": "beats_chance"}]}
    layout = {"sections": [{"id": "running", "title": "## 5. Running it"}]}

    def nb(acc_line):
        return {"cells": [
            {"cell_type": "code", "source": "load",
             "outputs": [{"output_type": "stream", "text": "Classes: 10\n"}]},
            {"cell_type": "markdown", "source": "## 5. Running it"},
            {"cell_type": "code", "source": "run",
             "outputs": [{"output_type": "stream", "text": acc_line}]},
        ]}

    low = derive_demo_verdict(nb("final test_acc=0.09\n"), ds, layout,
                              paradigm="active_learning",
                              family_kit_committed=True)
    assert low["verdict"] == "undetermined"
    high = derive_demo_verdict(nb("target accuracy: 0.95\n"), ds, layout,
                               paradigm="active_learning",
                               family_kit_committed=True)
    assert high["verdict"] == "undetermined"


def test_heading_section_number_parses_multi_digit_sections():
    from demo_verdict import heading_section_number

    assert heading_section_number("## 10. Appendix experiments") == 10
    assert heading_section_number("### 12.1 Sub") == 12
    assert heading_section_number("## §11 alt form") == 11
    assert heading_section_number("10. a plain list line") is None


def test_headline_section_cells_ignore_numbered_list_items():
    """A markdown LIST item starting `1.` must not move the section pointer
    (the route_findings parser quirk, fixed here by requiring a heading)."""
    nb = {"cells": [
        {"cell_type": "markdown", "source": "## 5. Running it"},
        {"cell_type": "markdown", "source": "1. first do this\n2. then this"},
        {"cell_type": "code", "source": "run()", "outputs": []},
    ]}
    cells = headline_section_cells(nb, 5)
    assert [idx for idx, _ in cells] == [2]


# ---------------------------------------------------------------------------
# Delivery derivation — first-class demotion, disclosures, disagreement
# ---------------------------------------------------------------------------


def _report(*verdicts):
    return {"verdicts": [
        {"probe_id": pid, "verdict": v, "message": f"{pid} message"}
        for pid, v in verdicts]}


_FAILED_VERDICT = {
    "verdict": "failed",
    "evidence_line": "Planning status: timeout",
    "evidence_cell": 12,
    "decided_by": "failure_marker",
}


def _structured_verdict(
    *, evaluation: str = "valid", skill: str = "demonstrated",
) -> dict:
    projection = {
        "demonstrated": "succeeded",
        "not_demonstrated": "failed",
    }.get(skill, "undetermined")
    return {
        "schema_version": "2.0.0",
        "verdict": projection,
        "decided_by": "structured_demo_skill",
        "evidence_line": "Demo skill: structured result",
        "evidence_status": {
            "execution": {"status": "completed", "reasons": []},
            "evaluation_validity": {"status": evaluation, "reasons": []},
            "mechanism": {"status": "undetermined", "reasons": []},
            "skill": {"status": skill, "reasons": []},
            "paper_benchmark": {"status": "not_assessed", "reasons": []},
        },
    }


def _reference_spec(*refs: str) -> dict:
    return {"methodology_replication_contract": {"elements": [{
        "element_id": "contribution",
        "role": "core_methodology",
        "replication_status": "must_replicate",
        "technical_concept": "contribution",
        "required_behavior": "the contribution changes the result",
        "paper_section": "Section 3",
        "acceptable_approximations": [],
        "forbidden_substitutions": [],
        "paper_element_ids": ["paper-contribution"],
        "verification_probe_refs": list(refs),
    }]}}


_GRAPH_REFS = {
    "HG-1": "graph_mechanism.alignment_prerequisite",
    "HG-2": "graph_mechanism.parameter_agreement",
    "HG-3": "graph_mechanism.construction_semantics",
    "HG-4": "graph_mechanism.topology_sensitivity",
    "HG-5": "graph_mechanism.neighbor_sensitivity",
    "HG-6": "graph_mechanism.permutation_equivalence",
    "HG-7": "graph_mechanism.contribution_ablation",
}


def _graph_reference_spec(
    *,
    permutation_applicability="equivariant",
    contribution_control=True,
    discriminator="graph_mechanism.neighbor_sensitivity",
) -> dict:
    def element(element_id: str, *refs: str) -> dict:
        return {
            "element_id": element_id,
            "role": "core_methodology",
            "replication_status": "must_replicate",
            "technical_concept": element_id,
            "required_behavior": f"exercise {element_id}",
            "paper_section": "Section 3",
            "acceptable_approximations": [],
            "forbidden_substitutions": [],
            "paper_element_ids": [f"paper-{element_id}"],
            "verification_probe_refs": list(refs),
        }

    elements = [
        element("graph-alignment", _GRAPH_REFS["HG-1"]),
        element(
            "graph-construction",
            _GRAPH_REFS["HG-2"],
            _GRAPH_REFS["HG-3"],
        ),
        element(
            "message-passing",
            _GRAPH_REFS["HG-4"],
            _GRAPH_REFS["HG-5"],
            *(
                [_GRAPH_REFS["HG-6"]]
                if permutation_applicability == "equivariant" else []
            ),
        ),
    ]
    if contribution_control:
        elements.append(element("graph-ablation", _GRAPH_REFS["HG-7"]))
    mechanism = {
        "schema_version": "1.0",
        "alignment_element_id": "graph-alignment",
        "construction": {
            "element_id": "graph-construction",
            "callable": {
                "module": "method.model",
                "qualname": "construct_graph_exact",
            },
            "feature_input_root": "batch.features",
            "feature_parameter": "features_exact",
            "output_selector": {"kind": "return_value"},
            "threshold": {
                "metric": "cosine_similarity",
                "comparison": "greater_than_or_equal",
                "parameter": {
                    "params_name": "similarity_threshold",
                    "callable_parameter": "threshold_exact",
                },
            },
            "cap": {"kind": "none"},
            "self_loop_policy": "forbidden",
            "direction_policy": "undirected_bidirectional",
        },
        "message_passing": {
            "element_id": "message-passing",
            "callable": {
                "module": "method.model",
                "qualname": "execute_graph_exact",
            },
            "graph_parameter": "edge_index",
            "neighbor_signal_root": "batch.demand_history",
            "neighbor_signal_parameter": "demand_history",
            "output_root": "outputs.graph_embeddings",
        },
        "permutation_applicability": permutation_applicability,
        "contribution_ablation": (
            {
                "kind": "non_graph_decoder",
                "element_id": "graph-ablation",
                "discriminating_probe_ref": discriminator,
                "callable": {
                    "module": "method.model",
                    "qualname": "run_non_graph_decoder",
                },
                "graph_parameter": "edge_index",
                "neighbor_signal_parameter": "demand_history",
                "output_root": "outputs.graph_embeddings",
            }
            if contribution_control else None
        ),
        "probe_refs": {
            "parameter_agreement": _GRAPH_REFS["HG-2"],
            "construction": _GRAPH_REFS["HG-3"],
            "topology": _GRAPH_REFS["HG-4"],
            "neighbor_signal": _GRAPH_REFS["HG-5"],
            "permutation": (
                _GRAPH_REFS["HG-6"]
                if permutation_applicability == "equivariant" else None
            ),
            "contribution_ablation": (
                _GRAPH_REFS["HG-7"] if contribution_control else None
            ),
        },
    }
    return {
        "schema_version": "1.14.0",
        "methodology_replication_contract": {
            "elements": elements,
            "homogeneous_graph_mechanism": mechanism,
        },
    }


def _graph_plan(spec: dict | None = None) -> dict:
    spec = spec or _graph_reference_spec()
    mechanism = spec["methodology_replication_contract"][
        "homogeneous_graph_mechanism"
    ]
    construction = mechanism["construction"]
    message = mechanism["message_passing"]
    ablation = mechanism.get("contribution_ablation")
    owner_callables = {
        "HG-1": (mechanism["alignment_element_id"],
                 ["method.training:_prepare_graph_batch"]),
        "HG-2": (construction["element_id"],
                 ["method.model:construct_graph_exact"]),
        "HG-3": (construction["element_id"],
                 ["method.model:construct_graph_exact"]),
        "HG-4": (message["element_id"],
                 ["method.model:execute_graph_exact"]),
        "HG-5": (message["element_id"],
                 ["method.model:execute_graph_exact"]),
    }
    if mechanism["permutation_applicability"] == "equivariant":
        owner_callables["HG-6"] = (
            message["element_id"], ["method.model:execute_graph_exact"]
        )
    if isinstance(ablation, dict):
        owner_callables["HG-7"] = (
            ablation["element_id"],
            [
                "method.model:execute_graph_exact",
                "method.model:run_non_graph_decoder",
            ],
        )
    return {
        "schema_version": "1.0",
        "status": "ready",
        "representation": "sparse_edge_index",
        "alignment": {
            "status": "pass",
            "preparation_callable": {
                "module": "method.training",
                "name": "_prepare_graph_batch",
            },
        },
        "alignment_runtime_receipt": {
            "status": "pass",
            "verified": True,
            "reason": "stage_2d_runtime_alignment_verified",
            "authority_digests": {
                ".pipeline/arch_contract.json": f"sha256:{'c' * 64}",
            },
        },
        "callable_liveness_receipt": {
            "status": "pass",
            "verified": True,
            "reason": "graph_callable_liveness_verified",
        },
        "construction": {
            "callable": construction["callable"],
            "feature_input_root": construction["feature_input_root"],
            "threshold": {
                **construction["threshold"],
                "value": 0.5,
            },
            "self_loop_policy": construction["self_loop_policy"],
            "direction_policy": construction["direction_policy"],
            "cap": construction["cap"],
        },
        "parameter_authority": {
            "status": "pass",
            "bindings": {
                "similarity_threshold": {
                    "carrier_value": 0.5,
                    "params_value": 0.5,
                    "paper_value": None,
                    "source": "paper",
                    "suppressed": False,
                    "duplicates": [],
                }
            },
        },
        "execution": {
            "callable": message["callable"],
            "neighbor_signal_root": message["neighbor_signal_root"],
            "output_root": message["output_root"],
            "seed": 1729,
            "tolerance": 1e-8,
        },
        "ablation": ablation,
        "groundings": {
            _GRAPH_REFS[probe_id]: {
                "element_ids": [owner],
                "bound_callables": callables,
            }
            for probe_id, (owner, callables) in owner_callables.items()
        },
        "fixture": {
            "scope": "demo-verdict-graph-fixture",
            "entity_ids": ["a", "b", "c"],
            "expected_graph": [[0, 1], [1, 0]],
            "comparison_witness": {
                "metric": "cosine_similarity",
                "comparison": "greater_than_or_equal",
                "threshold_value": 1.0,
                "entity_axis": 0,
                "feature_input": [
                    [1.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                ],
                "expected_graph": [[0, 1], [1, 0]],
            },
            "parameter_interventions": {
                "similarity_threshold": {
                    "nominal_value": 0.5,
                    "alternate_value": 0.75,
                    "value": 0.75,
                    "callable_parameter": "threshold_exact",
                    "expected_graph": [[], []],
                }
            },
            "topology_intervention": {
                "remove_edges": [[0, 1]],
                "targets": [1],
            },
            "neighbor_intervention": {
                "source": 0,
                "target": 1,
                "delta": 1.0,
                "control_targets": [2],
            },
            "permutation": [1, 0, 2],
        },
    }


def _graph_pass_evidence(probe_id: str, trace: dict) -> dict:
    evidence = {"trace": trace}
    if probe_id == "HG-1":
        evidence.update({
            "validator": "validate_arch_contract_runtime.py",
            "authority_paths": [".pipeline/arch_contract.json"],
        })
    elif probe_id == "HG-2":
        evidence["parameters"] = [{
            "role": "threshold",
            "params_name": "similarity_threshold",
            "carrier_value": 0.5,
            "params_value": 0.5,
            "paper_value": None,
            "planned_value": 0.5,
            "runtime_value": 0.5,
            "source": "paper",
            "suppressed": False,
            "duplicates": [],
        }]
        evidence["parameter_interventions"] = [{
            "role": "threshold",
            "params_name": "similarity_threshold",
            "callable_parameter": "threshold_exact",
            "nominal_value": 0.5,
            "intervention_value": 0.75,
            "runtime_value": 0.75,
            "expected_nominal_edge_symmetric_difference": 2,
            "observed_nominal_edge_symmetric_difference": 2,
            "expected_edge_count": 0,
            "constructed_edge_count": 0,
        }]
    elif probe_id == "HG-3":
        evidence.update({
            "representation": "sparse_edge_index",
            "node_count": 3,
            "expected_edge_count": 2,
            "constructed_edge_count": 2,
            "observed_edge_symmetric_difference": 0,
            "self_loop_policy": "forbidden",
            "direction_policy": "undirected_bidirectional",
            "cap_kind": "none",
            "comparison_witness": {
                "metric": "cosine_similarity",
                "comparison": "greater_than_or_equal",
                "threshold_value": 1.0,
                "identical_entity_positions": [0, 1],
                "runtime_threshold_value": 1.0,
                "expected_edge_count": 2,
                "constructed_edge_count": 2,
                "observed_edge_symmetric_difference": 0,
            },
        })
    elif probe_id == "HG-4":
        evidence.update({
            "effect_threshold": 1e-8,
            "baseline_max_abs": 1.0,
            "changed_max_abs": 1.0,
            "effect_scale": 1.0,
            "seed": 1729,
            "tolerance": 1e-8,
            "removed_edges": [[0, 1], [1, 0]],
            "target_entities": [1],
            "target_max_abs_deltas": [1.0],
            "maximum_target_delta": 1.0,
        })
    elif probe_id == "HG-5":
        evidence.update({
            "effect_threshold": 1e-8,
            "baseline_max_abs": 1.0,
            "changed_max_abs": 1.0,
            "effect_scale": 1.0,
            "seed": 1729,
            "tolerance": 1e-8,
            "source_entity": 0,
            "target_entity": 1,
            "signal_delta": 1.0,
            "target_max_abs_delta": 1.0,
            "control_max_abs_deltas": {"2": 0.0},
        })
    elif probe_id == "HG-6":
        evidence.update({
            "effect_threshold": 1e-8,
            "baseline_max_abs": 1.0,
            "changed_max_abs": 1.0,
            "effect_scale": 1.0,
            "seed": 1729,
            "tolerance": 1e-8,
            "permutation_new_to_old": [1, 0, 2],
            "restored_max_abs_delta": 0.0,
        })
    elif probe_id == "HG-7":
        real = _graph_pass_evidence("HG-5", trace)
        real.pop("trace")
        null = dict(real)
        null["target_max_abs_delta"] = 0.0
        evidence.update({
            "seed": 1729,
            "tolerance": 1e-8,
            "ablation_kind": "non_graph_decoder",
            "discriminating_probe_ref": _GRAPH_REFS["HG-5"],
            "real_status": "pass",
            "real_evidence": real,
            "null_status": "fail",
            "null_reason_code": "neighbor_response_absent",
            "null_evidence": null,
        })
    return evidence


def _graph_row(
    probe_id: str,
    verdict: str,
    plan: dict | None = None,
) -> dict:
    from probes.graph_mechanism import graph_plan_trace

    plan = plan or _graph_plan()
    owners = {
        "HG-1": "graph-alignment",
        "HG-2": "graph-construction",
        "HG-3": "graph-construction",
        "HG-4": "message-passing",
        "HG-5": "message-passing",
        "HG-6": "message-passing",
        "HG-7": "graph-ablation",
    }
    callables = {
        "HG-1": ["method.training:_prepare_graph_batch"],
        "HG-2": ["method.model:construct_graph_exact"],
        "HG-3": ["method.model:construct_graph_exact"],
        "HG-4": ["method.model:execute_graph_exact"],
        "HG-5": ["method.model:execute_graph_exact"],
        "HG-6": ["method.model:execute_graph_exact"],
        "HG-7": [
            "method.model:execute_graph_exact",
            "method.model:run_non_graph_decoder",
        ],
    }
    return {
        "probe_id": probe_id,
        "probe_ref": _GRAPH_REFS[probe_id],
        "verdict": verdict,
        "message": f"{probe_id} {verdict}",
        "tier": "behavioral",
        "element_ids": [owners[probe_id]],
        "bound_callables": callables[probe_id],
        "evidence": json.dumps(
            _graph_pass_evidence(probe_id, graph_plan_trace(plan))
            if verdict == "pass" else {"trace": graph_plan_trace(plan)}
        ),
    }


def test_failed_demo_verdict_demotes_with_its_own_reason():
    out = derive_delivery_label(
        _report(("US-1", "pass"), ("CT-1", "pass")), None,
        demo_verdict=_FAILED_VERDICT)
    assert out["label"] == "draft"
    (reason,) = out["reasons"]
    assert reason["source"] == "demo_verdict"
    assert reason["id"] == "demo_failed"
    assert reason["message"].startswith(
        "the notebook runs end to end, but its demonstration does not succeed")
    assert "Planning status: timeout" in reason["message"]
    assert out["demo_verdict"]["verdict"] == "failed"


def test_succeeded_demo_verdict_changes_nothing_but_records():
    out = derive_delivery_label(
        _report(("US-1", "pass"), ("CT-1", "pass")), None,
        demo_verdict={"verdict": "succeeded",
                      "evidence_line": "Planning status: success",
                      "decided_by": "success_marker"})
    assert out["label"] == "verified"
    assert out["reasons"] == []
    assert out["demo_verdict"]["verdict"] == "succeeded"


def test_structured_skill_failure_demotes_without_collapsing_other_axes():
    report = _report(("TSF-1", "pass"), ("CT-1", "pass"))
    report["verdicts"][0].update({
        "tier": "behavioral",
        "element_ids": ["forecast-distribution"],
    })
    out = derive_delivery_label(
        report, None,
        demo_verdict=_structured_verdict(skill="not_demonstrated"),
    )

    assert out["label"] == "draft"
    assert {reason["id"] for reason in out["reasons"]} == {
        "demo_skill_not_demonstrated"
    }
    axes = out["demo_verdict"]["evidence_status"]
    assert axes["execution"]["status"] == "completed"
    assert axes["evaluation_validity"]["status"] == "valid"
    assert axes["mechanism"]["status"] == "demonstrated"
    assert axes["skill"]["status"] == "not_demonstrated"
    assert axes["paper_benchmark"]["status"] == "not_assessed"


def test_migrated_contract_rejects_candidate_only_mechanism_evidence():
    report = _report(("TSF-1", "pass"), ("CT-1", "pass"))
    report["verdicts"][0].update({
        "tier": "behavioral",
        "element_ids": ["paper-contribution"],
        "probe_ref": "probes.time_series.sample_genuineness",
    })
    spec = _reference_spec("probes.time_series.own_history_sensitivity")

    out = derive_delivery_label(
        report,
        None,
        demo_verdict=_structured_verdict(),
        spec=spec,
    )

    mechanism = out["demo_verdict"]["evidence_status"]["mechanism"]
    assert mechanism["status"] == "undetermined"
    assert mechanism["reasons"][0]["code"] == "no_bound_behavioral_probe"

    report["verdicts"][0]["probe_ref"] = (
        "probes.time_series.own_history_sensitivity"
    )
    qualified = derive_delivery_label(
        report,
        None,
        demo_verdict=_structured_verdict(),
        spec=spec,
    )
    assert (
        qualified["demo_verdict"]["evidence_status"]["mechanism"]["status"]
        == "demonstrated"
    )


def test_graph_construction_passes_cannot_stand_in_for_mechanism_or_contribution():
    spec = _graph_reference_spec()
    plan = _graph_plan(spec)
    report = {"verdicts": [
        _graph_row("HG-1", "pass", plan),
        _graph_row("HG-2", "pass", plan),
        _graph_row("HG-3", "pass", plan),
        _graph_row("HG-4", "unprobeable", plan),
        _graph_row("HG-5", "unprobeable", plan),
        _graph_row("HG-6", "not_applicable", plan),
        _graph_row("HG-7", "unprobeable", plan),
    ]}

    out = derive_delivery_label(
        report,
        None,
        demo_verdict=_structured_verdict(),
        spec=spec,
        graph_execution_plan=plan,
    )

    axes = out["demo_verdict"]["evidence_status"]
    assert axes["graph_construction"]["status"] == "demonstrated"
    assert axes["graph_alignment"]["status"] == "demonstrated"
    assert axes["mechanism"]["status"] == "undetermined"
    assert axes["contribution"]["status"] == "undetermined"


def test_graph_liveness_and_discriminating_contribution_are_separate_axes():
    spec = _graph_reference_spec(permutation_applicability="not_applicable")
    plan = _graph_plan(spec)
    report = {"verdicts": [
        _graph_row("HG-1", "pass", plan),
        _graph_row("HG-2", "pass", plan),
        _graph_row("HG-3", "pass", plan),
        _graph_row("HG-4", "pass", plan),
        _graph_row("HG-5", "pass", plan),
        _graph_row("HG-6", "not_applicable", plan),
        _graph_row("HG-7", "unprobeable", plan),
    ]}
    # A real not-applicable row has no invented element/ref ownership.  The
    # typed nullable contract—not a fake binding—closes this conditional rung.
    report["verdicts"][5]["element_ids"] = []
    report["verdicts"][5]["bound_callables"] = []

    live_only = derive_delivery_label(
        report, None, demo_verdict=_structured_verdict(), spec=spec,
        graph_execution_plan=plan,
    )
    live_axes = live_only["demo_verdict"]["evidence_status"]
    assert live_axes["mechanism"]["status"] == "demonstrated"
    assert live_axes["contribution"]["status"] == "undetermined"
    assert live_only["label"] == "uncertified_new_territory"

    report["verdicts"][-1] = _graph_row("HG-7", "pass", plan)
    discriminating = derive_delivery_label(
        report, None, demo_verdict=_structured_verdict(), spec=spec,
        graph_execution_plan=plan,
    )
    axes = discriminating["demo_verdict"]["evidence_status"]
    assert axes["mechanism"]["status"] == "demonstrated"
    assert axes["contribution"]["status"] == "demonstrated"
    assert axes["skill"]["status"] == "demonstrated"
    assert axes["paper_benchmark"]["status"] == "not_assessed"
    assert discriminating["label"] == "verified"


@pytest.mark.parametrize("missing_probe_id", ["HG-1", "HG-2", "HG-3"])
def test_graph_lower_prerequisite_is_required_by_every_higher_rung(
    missing_probe_id,
):
    spec = _graph_reference_spec()
    plan = _graph_plan(spec)
    report = {"verdicts": [
        _graph_row(probe_id, "pass", plan)
        for probe_id in _GRAPH_REFS
        if probe_id != missing_probe_id
    ]}

    out = derive_delivery_label(
        report,
        None,
        demo_verdict=_structured_verdict(),
        spec=spec,
        graph_execution_plan=plan,
    )

    axes = out["demo_verdict"]["evidence_status"]
    if missing_probe_id == "HG-1":
        assert axes["graph_alignment"]["status"] == "undetermined"
    assert axes["graph_construction"]["status"] == "undetermined"
    assert axes["mechanism"]["status"] == "undetermined"
    assert axes["contribution"]["status"] == "undetermined"
    assert out["label"] == "uncertified_new_territory"


def test_equivariant_graph_requires_an_actual_permutation_pass():
    spec = _graph_reference_spec(permutation_applicability="equivariant")
    plan = _graph_plan(spec)
    report = {"verdicts": [
        _graph_row(
            probe_id,
            "not_applicable" if probe_id == "HG-6" else "pass",
            plan,
        )
        for probe_id in _GRAPH_REFS
    ]}

    out = derive_delivery_label(
        report,
        None,
        demo_verdict=_structured_verdict(),
        spec=spec,
        graph_execution_plan=plan,
    )

    axes = out["demo_verdict"]["evidence_status"]
    assert axes["mechanism"]["status"] == "undetermined"
    assert axes["contribution"]["status"] == "undetermined"
    assert out["label"] == "uncertified_new_territory"


def test_not_applicable_graph_rejects_bound_permutation_evidence():
    spec = _graph_reference_spec(permutation_applicability="not_applicable")
    plan = _graph_plan(spec)
    report = {"verdicts": [
        _graph_row(probe_id, "pass", plan) for probe_id in _GRAPH_REFS
    ]}

    out = derive_delivery_label(
        report,
        None,
        demo_verdict=_structured_verdict(),
        spec=spec,
        graph_execution_plan=plan,
    )

    axes = out["demo_verdict"]["evidence_status"]
    assert axes["mechanism"]["status"] == "undetermined"
    assert axes["contribution"]["status"] == "undetermined"
    assert out["label"] == "uncertified_new_territory"


def test_graph_contribution_requires_a_typed_paper_grounded_null():
    spec = _graph_reference_spec(contribution_control=False)
    plan = _graph_plan(spec)
    report = {"verdicts": [
        _graph_row(probe_id, "pass", plan) for probe_id in _GRAPH_REFS
    ]}

    out = derive_delivery_label(
        report, None, demo_verdict=_structured_verdict(), spec=spec,
        graph_execution_plan=plan,
    )

    axes = out["demo_verdict"]["evidence_status"]
    assert axes["mechanism"]["status"] == "demonstrated"
    assert axes["contribution"]["status"] == "undetermined"
    assert out["label"] == "uncertified_new_territory"


@pytest.mark.parametrize(
    "tamper", ["wrong_ref", "missing_callable", "duplicate", "bad_control"]
)
def test_tampered_graph_evidence_cannot_demonstrate_or_verify(tamper):
    spec = _graph_reference_spec()
    plan = _graph_plan(spec)
    report = {"verdicts": [
        _graph_row(probe_id, "pass", plan) for probe_id in _GRAPH_REFS
    ]}
    if tamper == "wrong_ref":
        report["verdicts"][4]["probe_ref"] = _GRAPH_REFS["HG-4"]
    elif tamper == "missing_callable":
        report["verdicts"][4]["bound_callables"] = []
    elif tamper == "duplicate":
        report["verdicts"].append(_graph_row("HG-5", "pass", plan))
    else:
        spec["methodology_replication_contract"][
            "homogeneous_graph_mechanism"
        ]["contribution_ablation"]["discriminating_probe_ref"] = (
            "graph_mechanism.contribution_ablation"
        )

    out = derive_delivery_label(
        report, None, demo_verdict=_structured_verdict(), spec=spec,
        graph_execution_plan=plan,
    )

    axes = out["demo_verdict"]["evidence_status"]
    assert axes["contribution"]["status"] == "undetermined"
    assert out["label"] == "uncertified_new_territory"


def test_graph_free_structured_evidence_keeps_the_original_five_axes():
    report = _report(("TSF-1", "pass"), ("CT-1", "pass"))
    report["verdicts"][0].update({
        "tier": "behavioral",
        "element_ids": ["forecast-distribution"],
    })

    out = derive_delivery_label(
        report, None, demo_verdict=_structured_verdict(), spec=None,
    )

    assert list(out["demo_verdict"]["evidence_status"]) == [
        "execution",
        "evaluation_validity",
        "mechanism",
        "skill",
        "paper_benchmark",
    ]


def test_structured_invalid_evaluation_demotes_but_skill_stays_undetermined():
    out = derive_delivery_label(
        _report(("US-1", "pass"), ("CT-1", "pass")), None,
        demo_verdict=_structured_verdict(
            evaluation="invalid", skill="undetermined",
        ),
    )

    assert out["label"] == "draft"
    assert {reason["id"] for reason in out["reasons"]} == {
        "demo_evaluation_invalid"
    }
    axes = out["demo_verdict"]["evidence_status"]
    assert axes["execution"]["status"] == "completed"
    assert axes["evaluation_validity"]["status"] == "invalid"
    assert axes["skill"]["status"] == "undetermined"


def test_structured_unresolved_evaluation_discloses_without_skill_claim():
    out = derive_delivery_label(
        _report(("US-1", "pass"), ("CT-1", "pass")), None,
        demo_verdict=_structured_verdict(
            evaluation="unresolved", skill="undetermined",
        ),
    )

    assert out["label"] == "verified"
    assert "demo_evaluation_unresolved" in {
        disclosure["id"] for disclosure in out["disclosures"]
    }


def test_required_structured_demo_artifact_absence_demotes_but_schema1_survives():
    report = _report(("US-1", "pass"), ("CT-1", "pass"))

    missing = derive_delivery_label(
        report, None, structured_demo_required=True,
    )
    assert missing["label"] == "draft"
    assert {reason["id"] for reason in missing["reasons"]} == {
        "structured_demo_evidence_missing"
    }

    legacy = derive_delivery_label(
        report,
        None,
        demo_verdict={
            "schema_version": "1.0.0",
            "verdict": "succeeded",
            "evidence_line": "Demo verdict: PASS",
        },
        structured_demo_required=True,
    )
    assert legacy["label"] == "verified"
    assert legacy["reasons"] == []


def test_required_structured_demo_malformed_schema2_demotes_manifest_cleanly(
    tmp_path,
):
    from final_manifest import build_final_manifest

    out = derive_delivery_label(
        _report(("US-1", "pass"), ("CT-1", "pass")),
        None,
        demo_verdict={
            "schema_version": "2.0.0",
            "verdict": "succeeded",
            "evidence_status": {"skill": {"status": "demonstrated"}},
        },
        structured_demo_required=True,
    )

    assert out["label"] == "draft"
    assert {reason["id"] for reason in out["reasons"]} == {
        "structured_demo_evidence_malformed"
    }
    assert "demo_verdict" not in out

    state = make_state(tmp_path / "run")
    state.paths.method_spec.write_text(json.dumps({
        "schema_version": "1.14.0",
        "methodology_replication_contract": {"elements": []},
    }))
    manifest = build_final_manifest(
        state.paths, stage_results=[], delivery=out,
    )
    assert manifest.delivery.demo_verdict is None


def test_undetermined_demo_verdict_discloses_without_demoting():
    out = derive_delivery_label(
        _report(("US-1", "pass"), ("CT-1", "pass")), None,
        demo_verdict={"verdict": "undetermined",
                      "reason": "the family declares no demo markers yet",
                      "kit_coverage_finding": None})
    assert out["label"] == "verified"
    ids = {d["id"] for d in out["disclosures"]}
    assert "demo_not_checked" in ids
    assert "demo_markers_missing" not in ids


def test_committed_family_kit_finding_rides_as_disclosure():
    out = derive_delivery_label(
        _report(("US-1", "pass"), ("CT-1", "pass")), None,
        demo_verdict={
            "verdict": "undetermined",
            "reason": "the knowledge_distillation family declares no markers",
            "kit_coverage_finding": {
                "id": "demo_markers_missing",
                "family": "knowledge_distillation",
                "message": "coverage gap we own: add demo_success markers"},
        })
    assert out["label"] == "verified"
    ids = {d["id"] for d in out["disclosures"]}
    assert {"demo_not_checked", "demo_markers_missing"} <= ids


def test_missing_demo_verdict_changes_nothing():
    """Pre-feature runs and resumed old runs: absent artifact → the same-code
    derivation is identical with and without the new argument (legacy
    outputs differ only in the bumped schema version string)."""
    report = _report(("US-1", "pass"), ("UB-6", "unprobeable"))
    assert (derive_delivery_label(report, None)
            == derive_delivery_label(report, None, demo_verdict=None))
    assert "demo_verdict" not in derive_delivery_label(report, None)


def test_verdict_probe_disagreement_is_disclosed_both_directions():
    # failed verdict vs passing UB-6
    out = derive_delivery_label(
        _report(("UB-6", "pass"), ("CT-1", "pass")), None,
        demo_verdict=_FAILED_VERDICT)
    assert out["label"] == "draft"  # the demotion stands
    assert any(d["id"] == "demo_probe_disagreement"
               for d in out["disclosures"])
    # succeeded verdict vs failing UB-6 (the probe's demotion stands)
    out = derive_delivery_label(
        _report(("UB-6", "fail"), ("CT-1", "pass")), None,
        demo_verdict={"verdict": "succeeded",
                      "evidence_line": "test_acc=0.9"})
    assert out["label"] == "draft"
    assert any(d["id"] == "demo_probe_disagreement"
               for d in out["disclosures"])
    # agreement adds nothing
    out = derive_delivery_label(
        _report(("UB-6", "fail"), ("CT-1", "pass")), None,
        demo_verdict=_FAILED_VERDICT)
    assert not any(d["id"] == "demo_probe_disagreement"
                   for d in out["disclosures"])


def test_manifest_schema_accepts_and_re_checks_demo_verdict():
    from schemas.final_manifest import DeliveryVerdict

    ok = DeliveryVerdict.model_validate({
        "label": "draft",
        "reasons": [{"source": "demo_verdict", "id": "demo_failed",
                     "message": "the demo does not succeed"}],
        "demo_verdict": {"verdict": "failed",
                         "evidence_line": "Planning status: timeout",
                         "evidence_cell": 12,
                         "decided_by": "failure_marker"},
    })
    assert ok.demo_verdict.verdict == "failed"
    with pytest.raises(ValueError, match="failed demo verdict"):
        DeliveryVerdict.model_validate({
            "label": "verified",
            "demo_verdict": {"verdict": "failed"},
        })

    structured = _structured_verdict(
        evaluation="invalid", skill="undetermined",
    )
    parsed = DeliveryVerdict.model_validate({
        "label": "draft",
        "reasons": [{
            "source": "demo_verdict",
            "id": "demo_evaluation_invalid",
            "message": "the scored evaluation is invalid",
        }],
        "demo_verdict": structured,
    })
    assert parsed.demo_verdict is not None
    assert parsed.demo_verdict.evidence_status is not None
    assert (parsed.demo_verdict.evidence_status.evaluation_validity.status
            == "invalid")
    with pytest.raises(ValueError, match="invalid demo evaluation"):
        DeliveryVerdict.model_validate({
            "label": "verified",
            "demo_verdict": structured,
        })


# ---------------------------------------------------------------------------
# Driver wiring — stage 3.c records the verdict, stage 5 consumes it
# ---------------------------------------------------------------------------


def _events(state) -> list[dict]:
    path = state.paths.pipeline_dir / "run_events.jsonl"
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text().splitlines()]


def _seed_run(run_dir: Path, scenario: str, paradigm: str) -> None:
    (run_dir / "notebook.ipynb").write_text(
        (ZOO / scenario / "notebook.ipynb").read_text(encoding="utf-8"),
        encoding="utf-8")
    (run_dir / ".pipeline" / "method_spec.json").write_text(json.dumps({
        "comparison": {"classification": {"id": paradigm}}}))


def test_stage_3c_completion_records_failed_demo_verdict(
    fake_dispatch, fake_subprocess, run_dir
):
    from run_pipeline import run_stage_3c
    state = make_state(run_dir)
    _seed_run(run_dir, "idbrrt-demo-timeout-reconstructed", "motion_planning")

    fake_subprocess.expect_script(returncode=0)  # smoke clean

    result = run_stage_3c(state)
    assert_stage_completed(result, "stage_3c")
    payload = json.loads(
        (run_dir / ".pipeline" / DEMO_VERDICT_JSON).read_text())
    assert payload["verdict"] == "failed"
    assert "Planning status: timeout" in payload["evidence_line"]
    events = [e["event_type"] for e in _events(state)]
    assert "demo_verdict_failed" in events


def test_stage_3c_completion_records_succeeded_demo_verdict(
    fake_dispatch, fake_subprocess, run_dir
):
    from run_pipeline import run_stage_3c
    state = make_state(run_dir)
    _seed_run(run_dir, "mp-demo-success-reconstructed", "motion_planning")

    fake_subprocess.expect_script(returncode=0)

    result = run_stage_3c(state)
    assert_stage_completed(result, "stage_3c")
    payload = json.loads(
        (run_dir / ".pipeline" / DEMO_VERDICT_JSON).read_text())
    assert payload["verdict"] == "succeeded"
    events = [e["event_type"] for e in _events(state)]
    assert "demo_verdict_recorded" in events
    assert "demo_verdict_failed" not in events


def test_stage_3c_completion_records_kit_coverage_gap_event(
    fake_dispatch, fake_subprocess, run_dir
):
    """knowledge_distillation is committed but declares no markers — the
    coverage finding we own must land on the event stream."""
    from run_pipeline import run_stage_3c
    state = make_state(run_dir)
    _seed_run(run_dir, "idbrrt-demo-timeout-reconstructed",
              "knowledge_distillation")

    fake_subprocess.expect_script(returncode=0)

    result = run_stage_3c(state)
    assert_stage_completed(result, "stage_3c")
    events = [e["event_type"] for e in _events(state)]
    assert "demo_kit_coverage_gap" in events
    payload = json.loads(
        (run_dir / ".pipeline" / DEMO_VERDICT_JSON).read_text())
    assert payload["verdict"] == "undetermined"


def test_stage_3c_rerun_clears_stale_demo_verdict_on_degraded_exit(
    fake_dispatch, fake_subprocess, run_dir
):
    """The dead-attempt-leak direction (review finding 1): a prior leg's
    verdict must never ride into delivery when THIS leg re-ran smoke and
    exited degraded — the artifact may only describe the notebook the
    current leg executed."""
    from run_pipeline import run_stage_3c
    state = make_state(run_dir)
    _seed_run(run_dir, "mp-demo-success-reconstructed", "motion_planning")
    stale = run_dir / ".pipeline" / DEMO_VERDICT_JSON
    stale.write_text(json.dumps({"verdict": "succeeded",
                                 "evidence_line": "Planning status: success"}))
    stale_scaling = run_dir / ".pipeline" / "target_scaling_state.json"
    stale_scaling.write_text(json.dumps({"state_digest": "stale"}))
    stale_history = run_dir / ".pipeline" / "training_history.json"
    stale_history.write_text(json.dumps({"record_digest": "stale"}))

    # Cache miss (no sentinel) → smoke re-runs and exits 3 (environmental
    # degrade): the stage degrades, and NO verdict may survive.
    fake_subprocess.expect_script(returncode=3,
                                  stderr="No module named 'nbclient'")

    result = run_stage_3c(state)
    assert result.status == "degraded"
    assert not stale.exists()
    assert not stale_scaling.exists()
    assert not stale_history.exists()


def test_stage_3c_skip_keeps_prior_demo_verdict(
    fake_dispatch, fake_subprocess, run_dir
):
    """The other direction: a resume that never re-runs smoke still delivers
    the prior leg's executed outputs, so that leg's verdict stands."""
    from run_pipeline import run_stage_3c, write_stage_3c_upstream_digest
    state = make_state(run_dir)
    _seed_run(run_dir, "mp-demo-success-reconstructed", "motion_planning")
    prior = run_dir / ".pipeline" / DEMO_VERDICT_JSON
    prior_payload = json.dumps({"verdict": "succeeded",
                                "evidence_line": "Planning status: success"})
    prior.write_text(prior_payload)
    prior_scaling = run_dir / ".pipeline" / "target_scaling_state.json"
    prior_scaling_payload = json.dumps({"state_digest": "prior"})
    prior_scaling.write_text(prior_scaling_payload)
    prior_history = run_dir / ".pipeline" / "training_history.json"
    prior_history_payload = json.dumps({"record_digest": "prior"})
    prior_history.write_text(prior_history_payload)
    # Cache hit: sentinel + matching content digest → stage skips.
    (run_dir / ".pipeline" / "stage_3c.complete").write_text("")
    write_stage_3c_upstream_digest(state.paths)

    result = run_stage_3c(state)
    assert result.status == "skipped"
    assert prior.read_text() == prior_payload
    assert prior_scaling.read_text() == prior_scaling_payload
    assert prior_history.read_text() == prior_history_payload
    assert len(fake_subprocess.script_calls) == 0


def test_record_demo_verdict_without_notebook_writes_nothing(tmp_path):
    run_dir = tmp_path / "run"
    (run_dir / ".pipeline").mkdir(parents=True)
    assert record_demo_verdict(run_dir) is None
    assert not (run_dir / ".pipeline" / DEMO_VERDICT_JSON).exists()


def test_record_demo_verdict_extracts_structured_tsf_record_and_binds_split(
    tmp_path, monkeypatch,
):
    import demo_verdict

    run_dir = tmp_path / "run"
    pipeline = run_dir / ".pipeline"
    pipeline.mkdir(parents=True)
    record, _ = _structured_tsf_record(model=9.0, repeat=5.0)
    scaling_marker, scaling_state = _attach_target_scaling_proof(record)
    notebook = {"cells": [
        {"cell_type": "markdown",
         "source": "## 5. Training and forecasting end-to-end"},
        {"cell_type": "code", "source": "evaluate()", "outputs": [{
            "output_type": "stream", "name": "stdout",
            "text": [
                demo_verdict.DEMO_EVALUATION_PREFIX
                + json.dumps(record) + "\n",
                scaling_marker + "\n",
                "Demo verdict: PASS — derived presentation\n",
            ],
        }]},
    ]}
    (run_dir / "notebook.ipynb").write_text(json.dumps(notebook))
    (pipeline / "method_spec.json").write_text(json.dumps({
        "comparison": {"classification": {"id": "time_series_forecasting"}},
    }))
    split_receipt = {
        "schema_version": "1.0.0",
        "status": "valid",
        "validator": "eval_split_lineage",
        "reasons": ["fitting_to_reported_evaluation_disjointness_proved"],
        "evidence": {
            "fitting_target_ranges": [_scaling_split_row()],
            "fitting_to_reported_evaluation_relations": [{
                "model_id": "model",
                "root": "series",
                "first": {
                    "role": "fitting", "read_kind": "target",
                    "protocol_range": {"start": 0, "stop": 8},
                },
                "second": {
                    "role": "reported_evaluation", "read_kind": "target",
                    "protocol_range": {"start": 8, "stop": 9},
                },
                "status": "disjoint",
            }],
        },
    }
    monkeypatch.setattr(
        demo_verdict, "_trusted_split_receipt",
        lambda run, nb: split_receipt,
    )

    payload = record_demo_verdict(run_dir)

    assert payload is not None
    assert payload["decided_by"] == "structured_demo_skill"
    assert payload["evidence_status"]["evaluation_validity"]["status"] == "valid"
    assert payload["evidence_status"]["skill"]["status"] == "demonstrated"
    assert payload["executed_evaluation"] == record
    assert payload["evaluation_validity_receipt"]["status"] == "valid"
    assert payload["presentation_marker"]["line"].startswith("Demo verdict: PASS")
    scaling = payload["target_scaling_evidence"]
    assert scaling["status"] == "valid"
    assert scaling["state_digest"] == scaling_state["state_digest"]
    assert scaling["original_unit_comparison"]["status"] == "valid"
    assert json.loads((run_dir / STATE_ARTIFACT).read_text()) == scaling_state


def test_target_scaling_missing_marker_is_producer_invalid_and_demotes(
    tmp_path,
    monkeypatch,
):
    import demo_verdict

    run_dir = tmp_path / "missing-marker"
    record, _ = _structured_tsf_record(model=9.0, repeat=5.0)
    _attach_target_scaling_proof(record)
    _write_tsf_record_notebook(run_dir, record)
    monkeypatch.setattr(
        demo_verdict,
        "_trusted_split_receipt",
        lambda run, nb: _trusted_tsf_split_receipt(),
    )

    payload = record_demo_verdict(run_dir)

    evidence = payload["target_scaling_evidence"]
    assert evidence["status"] == "invalid"
    assert evidence["responsibility"] == "producer"
    assert evidence["consume_producer_retry"] is True
    assert evidence["reason"]["code"] == "target_scaling_marker_count"
    assert payload["evidence_status"]["evaluation_validity"]["status"] == "invalid"
    assert payload["evidence_status"]["skill"]["status"] == "undetermined"
    assert payload["verdict"] == "undetermined"
    assert not (run_dir / STATE_ARTIFACT).exists()


@pytest.mark.parametrize(
    "proof_case, expected_code",
    [
        ("missing", "target_scaling_original_unit_proof_missing"),
        ("malformed", "target_scaling_original_unit_proof_shape"),
        ("wrong_schema", "target_scaling_original_unit_proof_schema"),
        (
            "wrong_state",
            "target_scaling_original_unit_state_disagreement",
        ),
    ],
)
def test_target_scaling_missing_or_malformed_supported_proof_is_producer_invalid(
    tmp_path,
    monkeypatch,
    proof_case,
    expected_code,
):
    import demo_verdict

    run_dir = tmp_path / proof_case
    record, _ = _structured_tsf_record(model=9.0, repeat=5.0)
    scaling_marker, _ = _attach_target_scaling_proof(record)
    if proof_case == "missing":
        record.pop("target_scaling_proof")
    elif proof_case == "malformed":
        record["target_scaling_proof"].pop("prediction_entity_axis")
    elif proof_case == "wrong_schema":
        record["target_scaling_proof"]["schema_version"] = "2.0.0"
    else:
        record["target_scaling_proof"]["state_digest"] = "sha256:wrong"
    _write_tsf_record_notebook(run_dir, record, scaling_marker)
    monkeypatch.setattr(
        demo_verdict,
        "_trusted_split_receipt",
        lambda run, nb: _trusted_tsf_split_receipt(),
    )

    payload = record_demo_verdict(run_dir)

    evidence = payload["target_scaling_evidence"]
    assert evidence["status"] == "invalid"
    assert evidence["reason"]["code"] == expected_code
    assert evidence["responsibility"] == "producer"
    assert evidence["consume_producer_retry"] is True
    assert payload["evidence_status"]["evaluation_validity"]["status"] == "invalid"
    assert not (run_dir / STATE_ARTIFACT).exists()


def test_target_scaling_original_unit_disagreement_is_producer_invalid(
    tmp_path,
    monkeypatch,
):
    import demo_verdict

    run_dir = tmp_path / "wrong-original-units"
    record, _ = _structured_tsf_record(model=9.0, repeat=5.0)
    scaling_marker, _ = _attach_target_scaling_proof(record)
    record["target_scaling_proof"]["original_predictions"] = [[900.0]]
    _write_tsf_record_notebook(run_dir, record, scaling_marker)
    monkeypatch.setattr(
        demo_verdict,
        "_trusted_split_receipt",
        lambda run, nb: _trusted_tsf_split_receipt(),
    )

    payload = record_demo_verdict(run_dir)

    evidence = payload["target_scaling_evidence"]
    assert evidence["status"] == "invalid"
    assert evidence["reason"]["code"] == (
        "target_output_original_unit_disagreement"
    )
    assert evidence["consume_producer_retry"] is True
    assert not (run_dir / STATE_ARTIFACT).exists()


def test_target_scaling_unknown_output_role_is_pipeline_coverage_without_retry(
    tmp_path,
    monkeypatch,
):
    import demo_verdict

    run_dir = tmp_path / "unknown-role"
    record, _ = _structured_tsf_record(model=9.0, repeat=5.0)
    scaling_marker, _ = _attach_target_scaling_proof(record)
    record["target_scaling_proof"]["prediction_output_role"] = "quantile"
    _write_tsf_record_notebook(run_dir, record, scaling_marker)
    monkeypatch.setattr(
        demo_verdict,
        "_trusted_split_receipt",
        lambda run, nb: _trusted_tsf_split_receipt(),
    )

    payload = record_demo_verdict(run_dir)

    evidence = payload["target_scaling_evidence"]
    assert evidence["status"] == "unsupported"
    assert evidence["responsibility"] == "pipeline"
    assert evidence["consume_producer_retry"] is False
    assert evidence["reason"]["code"] == "target_output_role_unsupported"
    assert payload["evidence_status"]["evaluation_validity"]["status"] == (
        "unresolved"
    )
    assert not (run_dir / STATE_ARTIFACT).exists()


def test_record_demo_verdict_schema2_role_comes_from_typed_method_spec(
    tmp_path, monkeypatch,
):
    import demo_verdict
    from tests.test_evaluation_protocol import _pdfgnn_spec

    run_dir = tmp_path / "run"
    pipeline = run_dir / ".pipeline"
    pipeline.mkdir(parents=True)
    record, _ = _structured_tsf_record(model=9.0, repeat=5.0)
    record.update({
        "schema_version": "2.0.0",
        "evaluation_protocol_role": "test_span",
    })
    scaling_marker, _ = _attach_target_scaling_proof(record)
    notebook = {"cells": [
        {"cell_type": "markdown",
         "source": "## 5. Training and forecasting end-to-end"},
        {"cell_type": "code", "source": "evaluate()", "outputs": [{
            "output_type": "stream", "name": "stdout",
            "text": [
                demo_verdict.DEMO_EVALUATION_PREFIX
                + json.dumps(record) + "\n",
                scaling_marker + "\n",
                "Demo verdict: PASS — derived presentation\n",
            ],
        }]},
    ]}
    (run_dir / "notebook.ipynb").write_text(json.dumps(notebook))
    (pipeline / "method_spec.json").write_text(json.dumps(_pdfgnn_spec()))
    split_receipt = {
        "schema_version": "1.0.0",
        "status": "valid",
        "validator": "eval_split_lineage",
        "reasons": ["fitting_to_reported_evaluation_disjointness_proved"],
        "evidence": {
            "fitting_target_ranges": [_scaling_split_row()],
            "fitting_to_reported_evaluation_relations": [{
                "model_id": "model",
                "root": "series",
                "first": {
                    "role": "fitting", "read_kind": "target",
                    "protocol_range": {"start": 0, "stop": 8},
                },
                "second": {
                    "role": "reported_evaluation", "read_kind": "target",
                    "protocol_range": {"start": 8, "stop": 9},
                },
                "status": "disjoint",
            }],
        },
    }
    monkeypatch.setattr(
        demo_verdict, "_trusted_split_receipt",
        lambda run, nb: split_receipt,
    )

    payload = record_demo_verdict(run_dir)

    assert payload is not None
    carrier = payload["evaluation_protocol_role"]
    assert carrier["role"] == "test_span"
    assert carrier["scheme_kind"] == "single_holdout"
    assert carrier["scheme_paper_element_ids"] == ["evaluation-split"]
    assert carrier["role_paper_element_ids"] == ["evaluation-split"]
    assert payload["evaluation_validity_receipt"][
        "evaluation_protocol_role"
    ] == carrier
    assert payload["evidence_status"]["skill"]["status"] == "demonstrated"


def test_record_demo_verdict_schema2_missing_role_is_named_unresolved(
    tmp_path, monkeypatch,
):
    import demo_verdict
    from tests.test_evaluation_protocol import _pdfgnn_spec

    run_dir = tmp_path / "run"
    pipeline = run_dir / ".pipeline"
    pipeline.mkdir(parents=True)
    record, _ = _structured_tsf_record(model=9.0, repeat=5.0)
    record["schema_version"] = "2.0.0"
    scaling_marker, _ = _attach_target_scaling_proof(record)
    notebook = {"cells": [
        {"cell_type": "markdown",
         "source": "## 5. Training and forecasting end-to-end"},
        {"cell_type": "code", "source": "evaluate()", "outputs": [{
            "output_type": "stream", "name": "stdout",
            "text": [
                demo_verdict.DEMO_EVALUATION_PREFIX
                + json.dumps(record) + "\n",
                scaling_marker + "\n",
            ],
        }]},
    ]}
    (run_dir / "notebook.ipynb").write_text(json.dumps(notebook))
    (pipeline / "method_spec.json").write_text(json.dumps(_pdfgnn_spec()))
    monkeypatch.setattr(
        demo_verdict, "_trusted_split_receipt",
        lambda run, nb: {
            "schema_version": "1.0.0",
            "status": "valid",
            "validator": "eval_split_lineage",
            "reasons": [],
            "evidence": {
                "fitting_target_ranges": [_scaling_split_row()],
            },
        },
    )

    payload = record_demo_verdict(run_dir)

    assert payload is not None
    receipt = payload["evaluation_validity_receipt"]
    assert receipt["status"] == "unresolved"
    assert receipt["reasons"] == [
        "executed_evaluation_protocol_role_missing_or_unknown"
    ]
    assert payload["evaluation_protocol_role"] is None
    assert payload["evidence_status"]["evaluation_validity"]["status"] == (
        "unresolved"
    )


def test_structured_verdict_pass_error_records_unresolved_schema2_fallback(
    tmp_path, monkeypatch,
):
    import demo_verdict
    from run_pipeline import _record_demo_verdict

    state = make_state(tmp_path / "run")
    state.paths.method_spec.write_text(json.dumps({
        "comparison": {
            "classification": {"id": "time_series_forecasting"},
        },
    }))

    def crash(_run_dir):
        raise RuntimeError("synthetic evidence-pass failure")

    monkeypatch.setattr(demo_verdict, "record_demo_verdict", crash)

    _record_demo_verdict(state)

    payload = json.loads(
        (state.paths.pipeline_dir / DEMO_VERDICT_JSON).read_text()
    )
    assert payload["schema_version"] == "2.0.0"
    assert payload["decided_by"] == "structured_demo_skill_error"
    assert payload["evidence_status"]["execution"]["status"] == "completed"
    assert (
        payload["evidence_status"]["evaluation_validity"]["status"]
        == "unresolved"
    )
    assert payload["evidence_status"]["skill"]["status"] == "undetermined"

    _patch_battery(monkeypatch, state, _report(("CT-1", "pass")))
    from run_pipeline import run_delivery_gating

    delivery = run_delivery_gating(state)
    assert delivery["label"] == "draft"
    assert {reason["id"] for reason in delivery["reasons"]} == {
        "structured_demo_evidence_error"
    }
    assert delivery["demo_verdict"]["evidence_status"][
        "evaluation_validity"
    ]["status"] == "unresolved"


def test_structured_requirement_taxonomy_resolution_error_fails_closed(
    tmp_path, monkeypatch,
):
    import demo_verdict

    run_dir = tmp_path / "run"
    pipeline = run_dir / ".pipeline"
    pipeline.mkdir(parents=True)
    (pipeline / "method_spec.json").write_text(json.dumps({
        "comparison": {"classification": {"id": "synthetic_family"}},
    }))

    def crash(**_kwargs):
        raise RuntimeError("taxonomy unavailable")

    monkeypatch.setattr(taxonomy, "load_taxonomy", crash)

    paradigm, required = demo_verdict.structured_demo_requirement(run_dir)

    assert paradigm == "synthetic_family"
    assert required is True


def _patch_battery(monkeypatch, state, report: dict):
    import subprocess

    import run_pipeline

    if not state.paths.method_spec.is_file():
        state.paths.method_spec.write_text(json.dumps({
            "schema_version": "1.11.0",
            "comparison": {"classification": {"id": "active_learning"}},
            "methodology_replication_contract": {"elements": []},
        }))

    def fake_run(cmd, **kwargs):
        (state.paths.pipeline_dir / "probe_report.json").write_text(
            json.dumps(report))
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
    monkeypatch.setattr(run_pipeline.subprocess, "run", fake_run)


def test_delivery_gating_consumes_demo_verdict_artifact(tmp_path, monkeypatch):
    from run_pipeline import run_delivery_gating
    state = make_state(tmp_path / "run")
    _patch_battery(monkeypatch, state, _report(("CT-1", "pass")))
    (state.paths.pipeline_dir / "demo_verdict.json").write_text(
        json.dumps(_FAILED_VERDICT))

    delivery = run_delivery_gating(state)
    assert delivery["label"] == "draft"
    assert [r["id"] for r in delivery["reasons"]] == ["demo_failed"]
    assert delivery["demo_verdict"]["verdict"] == "failed"


def test_delivery_gating_without_artifact_matches_pre_feature_shape(
    tmp_path, monkeypatch
):
    from run_pipeline import run_delivery_gating
    state = make_state(tmp_path / "run")
    _patch_battery(monkeypatch, state, _report(("CT-1", "pass")))

    delivery = run_delivery_gating(state)
    assert delivery["label"] == "verified"
    assert "demo_verdict" not in delivery


def test_delivery_gating_demotes_missing_required_structured_artifact(
    tmp_path, monkeypatch,
):
    from run_pipeline import run_delivery_gating

    state = make_state(tmp_path / "run")
    _patch_battery(monkeypatch, state, _report(("CT-1", "pass")))
    state.paths.method_spec.write_text(json.dumps({
        "comparison": {
            "classification": {"id": "time_series_forecasting"},
        },
    }))

    delivery = run_delivery_gating(state)

    assert delivery["label"] == "draft"
    assert {reason["id"] for reason in delivery["reasons"]} == {
        "structured_demo_evidence_missing"
    }


def test_manifest_carries_demo_verdict_through_delivery(tmp_path):
    from final_manifest import build_final_manifest
    state = make_state(tmp_path / "run")
    state.paths.method_spec.write_text(json.dumps({
        "schema_version": "1.14.0",
        "methodology_replication_contract": {"elements": []},
    }))
    delivery = derive_delivery_label(
        _report(("CT-1", "pass")), None, demo_verdict=_FAILED_VERDICT)
    manifest = build_final_manifest(
        state.paths, stage_results=[], delivery=delivery)
    dumped = manifest.model_dump()
    assert dumped["delivery"]["demo_verdict"]["verdict"] == "failed"
    assert dumped["delivery"]["label"] == "draft"


def test_report_renders_demo_reason_in_plain_language(tmp_path):
    from render_run_report import _reason_lead, _why_this_label
    delivery = derive_delivery_label(
        _report(("CT-1", "pass")), None, demo_verdict=_FAILED_VERDICT)
    assert _reason_lead(delivery["reasons"][0]) == "Headline demo outcome"
    block = _why_this_label(delivery, None)
    assert "its demonstration does not succeed" in block


# ---------------------------------------------------------------------------
# Data-signal check — the Rethinking impossible-gate burn
# ---------------------------------------------------------------------------


def test_class_signal_passes_on_separated_synthetic_data():
    from fix_data_signal import class_signal_verdict
    from probes.fixtures import make_classification_fixture

    fixture = make_classification_fixture(n_classes=3, n_features=12,
                                          n_per_class_pool=40, seed=0)
    out = class_signal_verdict(fixture.x_pool, fixture.y_pool)
    assert out["verdict"] == "pass"
    assert out["stats"]["split_half_accuracy"] > 0.9


def test_class_signal_fails_on_random_labels():
    from fix_data_signal import class_signal_verdict

    rng = np.random.default_rng(0)
    x = rng.normal(size=(240, 16))
    y = rng.integers(0, 10, size=240)
    out = class_signal_verdict(x, y)
    assert out["verdict"] == "fail"
    assert "can never pass" in out["message"]


def test_class_signal_fails_on_constant_labels():
    from fix_data_signal import class_signal_verdict

    rng = np.random.default_rng(0)
    out = class_signal_verdict(rng.normal(size=(40, 8)), np.zeros(40, int))
    assert out["verdict"] == "fail"
    assert "constant" in out["message"]


def test_label_free_data_yields_no_pair():
    from fix_data_signal import classify_candidates, find_labeled_pair

    rng = np.random.default_rng(0)
    # Features only — the Rethinking fix-data shape (no label array at all).
    candidates = [rng.normal(size=(200, 32)).astype(np.float32),
                  rng.normal(size=(50, 32)).astype(np.float32)]
    assert find_labeled_pair(candidates) is None
    tag, _ = classify_candidates(candidates)
    assert tag == "label_free"


def test_float_encoded_labels_coerce_and_pass():
    """Review finding 2: BCE-style float 0.0/1.0 targets are labels, not
    label-free data."""
    from fix_data_signal import class_signal_verdict, find_labeled_pair
    from probes.fixtures import make_classification_fixture

    fixture = make_classification_fixture(n_classes=2, n_features=12,
                                          n_per_class_pool=60, seed=0)
    y_float = fixture.y_pool.astype(np.float32)
    pair = find_labeled_pair([fixture.x_pool, y_float])
    assert pair is not None
    x, y = pair
    assert np.issubdtype(y.dtype, np.integer)
    assert class_signal_verdict(x, y)["verdict"] == "pass"


def test_one_hot_labels_coerce_and_pass():
    from fix_data_signal import class_signal_verdict, find_labeled_pair
    from probes.fixtures import make_classification_fixture

    fixture = make_classification_fixture(n_classes=3, n_features=12,
                                          n_per_class_pool=40, seed=0)
    one_hot = np.eye(3, dtype=np.float32)[fixture.y_pool]
    pair = find_labeled_pair([fixture.x_pool, one_hot])
    assert pair is not None
    x, y = pair
    assert (y == fixture.y_pool).all()
    assert class_signal_verdict(x, y)["verdict"] == "pass"


def test_uncoercible_label_shapes_read_unevaluable_not_fail():
    """A 1-D continuous float array might be a regression target or a loss
    curve — declaring the gate unwinnable over it is the misdirection the
    check exists to prevent, pointed the other way."""
    from fix_data_signal import classify_candidates

    rng = np.random.default_rng(0)
    x = rng.normal(size=(100, 8))
    continuous = rng.normal(size=100)  # not integral → not confidently labels
    tag, n = classify_candidates([x, continuous])
    assert tag == "uncoercible"
    assert n == 1


def test_find_labeled_pair_rejects_index_arrays():
    from fix_data_signal import find_labeled_pair

    rng = np.random.default_rng(0)
    x = rng.normal(size=(100, 8))
    index_like = np.arange(100)          # enumerates every sample — not labels
    labels = rng.integers(0, 4, size=100)
    pair = find_labeled_pair([x, index_like, labels])
    assert pair is not None
    assert pair[1] is labels


def _data_signal_run(tmp_path: Path, data_src: str,
                     paradigm: str = "active_learning") -> Path:
    """A minimal run dir whose rendered notebook carries `data_src` as its §3
    setup cell. The §5 cell RAISES, pinning that the check never executes the
    demo sections."""
    rd = tmp_path / "dsrun"
    (rd / ".pipeline").mkdir(parents=True, exist_ok=True)
    (rd / ".pipeline" / "method_spec.json").write_text(json.dumps({
        "comparison": {"classification": {"id": paradigm}}}))
    nb = {"cells": [
        {"cell_type": "markdown", "source": "## 0. Install"},
        {"cell_type": "code", "source": "%pip install -r requirements.txt",
         "outputs": []},
        {"cell_type": "markdown", "source": "## 1. Setup"},
        {"cell_type": "code",
         "source": "%matplotlib inline\nimport numpy as np\nnp.random.seed(0)",
         "outputs": []},
        {"cell_type": "markdown", "source": "## 2. Parameters"},
        {"cell_type": "code", "source": "cfg = {'n': 240}", "outputs": []},
        {"cell_type": "markdown", "source": "## 3. The setup pieces"},
        {"cell_type": "code", "source": data_src, "outputs": []},
        {"cell_type": "markdown", "source": "## 5. Running it"},
        {"cell_type": "code",
         "source": "raise RuntimeError('demo cells must never run here')",
         "outputs": []},
    ]}
    (rd / "notebook.ipynb").write_text(json.dumps(nb))
    return rd


def test_evaluate_run_fails_label_free_notebook_swap(tmp_path):
    """The Rethinking channel: the fix swapped label-free random data in via
    the notebook source; the check reads the notebook's own data cells."""
    from fix_data_signal import evaluate_run

    rd = _data_signal_run(
        tmp_path,
        "x_pool = np.random.rand(cfg['n'], 32).astype('float32')\n"
        "x_test = np.random.rand(60, 32).astype('float32')")
    out = evaluate_run(rd)
    assert out["verdict"] == "fail"
    assert out["channel"] == "notebook"
    assert "no label array" in out["message"]


def test_evaluate_run_passes_class_conditional_notebook_data(tmp_path):
    from fix_data_signal import evaluate_run

    rd = _data_signal_run(
        tmp_path,
        "protos = np.linspace(-1, 1, 3)\n"
        "y_pool = np.repeat(np.arange(3), 80)\n"
        "x_pool = protos[y_pool][:, None] + 0.1*np.random.randn(240, 16)\n")
    out = evaluate_run(rd)
    assert out["verdict"] == "pass"
    assert out["channel"] == "notebook"


def test_evaluate_run_not_applicable_without_beats_chance(tmp_path):
    from fix_data_signal import evaluate_run

    rd = _data_signal_run(tmp_path, "x = 1", paradigm="motion_planning")
    assert evaluate_run(rd)["verdict"] == "not_applicable"


def test_setup_section_blob_ignores_demo_cell_changes(tmp_path):
    """Review finding 4: the data-swap digest hashes only the setup-section
    slice, so a cosmetic fix in a demo cell never triggers the data-signal
    subprocess, while a §3 data-cell change does."""
    from demo_verdict import setup_section_source_blob

    rd = tmp_path / "run"
    (rd / ".pipeline").mkdir(parents=True)
    (rd / ".pipeline" / "method_spec.json").write_text(json.dumps({
        "comparison": {"classification": {"id": "active_learning"}}}))

    def nb(setup_src, demo_src):
        return json.dumps({"cells": [
            {"cell_type": "markdown", "source": "## 3. The setup pieces"},
            {"cell_type": "code", "source": setup_src, "outputs": []},
            {"cell_type": "markdown",
             "source": "## 5. Running active learning end-to-end"},
            {"cell_type": "code", "source": demo_src, "outputs": []},
        ]})

    (rd / "notebook.ipynb").write_text(nb("x = load_data()", "run_loop()"))
    baseline = setup_section_source_blob(rd)
    (rd / "notebook.ipynb").write_text(
        nb("x = load_data()", "run_loop()  # cosmetic plot tweak"))
    assert setup_section_source_blob(rd) == baseline
    (rd / "notebook.ipynb").write_text(
        nb("x = np.random.rand(200, 32)", "run_loop()"))
    assert setup_section_source_blob(rd) != baseline


def test_declares_beats_chance_is_taxonomy_driven(tmp_path):
    from fix_data_signal import declares_beats_chance

    def run_with(paradigm):
        rd = tmp_path / paradigm.replace("/", "_")
        (rd / ".pipeline").mkdir(parents=True)
        (rd / ".pipeline" / "method_spec.json").write_text(json.dumps({
            "comparison": {"classification": {"id": paradigm}}}))
        return rd

    assert declares_beats_chance(run_with("active_learning")) is True
    assert declares_beats_chance(run_with("vision_transformer")) is True
    assert declares_beats_chance(run_with("motion_planning")) is False
    empty = tmp_path / "empty"
    (empty / ".pipeline").mkdir(parents=True)
    assert declares_beats_chance(empty) is False


# ---------------------------------------------------------------------------
# Fix-loop wiring — a data swap triggers the check; a failure reaches the
# next dispatch prompts
# ---------------------------------------------------------------------------


def _minimal_notebook_json(num_cells: int = 10) -> dict:
    return {
        "cells": [
            {"cell_type": "code", "source": f"print('cell {i}')\n",
             "execution_count": i + 1, "metadata": {}, "outputs": []}
            for i in range(num_cells)
        ],
        "metadata": {"kernelspec": {"display_name": "Python 3",
                                    "language": "python", "name": "python3"},
                     "language_info": {"name": "python"}},
        "nbformat": 4,
        "nbformat_minor": 5,
    }


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


def _valid_smoke_diagnosis(target_agent: str = "r2c-method-coder",
                           target_file: str = "method/method.py") -> str:
    return json.dumps({
        "schema_version": "1.0.0",
        "target_agent": target_agent,
        "target_file": target_file,
        "bug_shape": "uncatalogued",
        "root_cause": "Test root cause.",
        "proposed_fix": "Test proposed fix.",
        "reasoning": "Test reasoning with value-origin trace.",
        "paper_fidelity_check": "Paper-faithful.",
        "value_origin_trace": ["step1: failure site", "step2: upstream",
                               "step3: origin"],
    }, indent=2)


def test_fix_loop_data_swap_failing_signal_reaches_next_dispatches(
    fake_dispatch, fake_subprocess, run_dir
):
    """The Rethinking 2026-07-14 shape, closed: iteration 0's notebook fix
    swaps in demo data (the draft is the ONLY surface a fix-mode producer can
    write), the deterministic check says the data can never pass the
    beats-chance gate, and BOTH iteration-1 dispatches (diagnostician and
    producer) carry the plain statement so the next fix targets the data."""
    from run_pipeline import run_stage_3c
    state = make_state(run_dir)
    fake_subprocess.set_run_dir(run_dir)
    (run_dir / "notebook.ipynb").write_text(
        json.dumps(_minimal_notebook_json()), encoding="utf-8")
    (run_dir / ".pipeline" / "method_spec.json").write_text(json.dumps({
        "comparison": {"classification": {"id": "active_learning"}}}))
    (run_dir / ".pipeline" / "notebook_draft.py").write_text(
        "# %%\nprint('original draft')\n")
    # The re-rendered notebook the fake render materializes: its §3 setup
    # cell now synthesizes data inline — the slice the data-swap digest
    # hashes (whole-draft cosmetic changes must NOT trigger the check).
    swapped_nb = json.dumps({
        "cells": [
            {"cell_type": "markdown", "metadata": {},
             "source": "## 3. The setup pieces"},
            {"cell_type": "code", "metadata": {}, "execution_count": 1,
             "source": "import numpy as np\nx = np.random.rand(200, 32)\n",
             "outputs": []},
            *_minimal_notebook_json()["cells"],
        ],
        "metadata": _minimal_notebook_json()["metadata"],
        "nbformat": 4, "nbformat_minor": 5,
    })

    # iter 0: smoke fails; the notebook-generator fix REWRITES the draft
    # (the demo-data swap the digest comparison must catch once the
    # re-render lands it in the notebook's setup sections).
    fake_subprocess.expect_script(returncode=1,
                                  stderr=_smoke_stderr_cell_fails(0))
    fake_dispatch.expect(
        agent="r2c-smoke-diagnostician",
        writes={".pipeline/smoke_diagnosis.json": _valid_smoke_diagnosis(
            target_agent="r2c-notebook-generator",
            target_file=".pipeline/notebook_draft.py")},
    )
    fake_dispatch.expect(
        agent="r2c-notebook-generator",
        writes={".pipeline/notebook_draft.py":
                "# %%\nimport numpy as np\nx = np.random.rand(200, 32)\n"},
    )
    fake_subprocess.expect_script(                 # render_notebook.py
        returncode=0, writes={"notebook.ipynb": swapped_nb})
    fake_subprocess.expect_stage2_script(ok=True)  # validate_notebook_output
    # data-signal check subprocess (fix_data_signal.py) → fail verdict
    fake_subprocess.expect_script(returncode=0, stdout=json.dumps({
        "verdict": "fail",
        "message": "labels do not correlate with features: split-half "
                   "nearest-class-mean accuracy 0.101 is at chance",
    }))

    # iter 1: smoke fails at a DIFFERENT cell — the next fix cycle runs with
    # the standing data-signal note.
    fake_subprocess.expect_script(
        returncode=1,
        stderr=_smoke_stderr_cell_fails(1, tb="TypeError: other bug"))
    fake_dispatch.expect(
        agent="r2c-smoke-diagnostician",
        writes={".pipeline/smoke_diagnosis.json": _valid_smoke_diagnosis()},
    )
    fake_dispatch.expect(
        agent="r2c-method-coder",
        writes={"method/method.py": "# fixed again\n"},
    )
    # Demo-data surface unchanged this time → no data-signal re-check queued.
    fake_subprocess.expect_script(returncode=0)    # render
    fake_subprocess.expect_stage2_script(ok=True)  # validate

    # iter 2: smoke passes
    fake_subprocess.expect_script(returncode=0)

    result = run_stage_3c(state)
    assert_stage_completed(result, "stage_3c")
    assert_dispatched(fake_dispatch, "r2c-notebook-generator", times=1)
    assert_dispatched(fake_dispatch, "r2c-method-coder", times=1)

    # The iteration-0 dispatch predates the check; the iteration-1
    # dispatches carry the statement plainly.
    producer_prompts = [c.prompt for c in fake_dispatch.calls
                        if c.agent in ("r2c-notebook-generator",
                                       "r2c-method-coder")]
    assert "DATA-SIGNAL CHECK FAILED" not in producer_prompts[0]
    assert "DATA-SIGNAL CHECK FAILED" in producer_prompts[1]
    assert "fix the data, not the model" in producer_prompts[1]
    diag_prompts = [c.prompt for c in fake_dispatch.calls
                    if c.agent == "r2c-smoke-diagnostician"]
    assert "DATA-SIGNAL CHECK FAILED" in diag_prompts[1]

    events = [e["event_type"] for e in _events(state)]
    assert "fix_data_signal_failed" in events


def test_fix_loop_without_data_change_never_runs_the_check(
    fake_dispatch, fake_subprocess, run_dir
):
    """Adjacent-good: a fix that leaves the demo-data surface untouched must
    not invoke the data-signal subprocess (no extra queued script) even on a
    beats-chance family."""
    from run_pipeline import run_stage_3c
    state = make_state(run_dir)
    (run_dir / "notebook.ipynb").write_text(
        json.dumps(_minimal_notebook_json()), encoding="utf-8")
    (run_dir / ".pipeline" / "method_spec.json").write_text(json.dumps({
        "comparison": {"classification": {"id": "active_learning"}}}))

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
    fake_subprocess.expect_script(returncode=0)   # render
    fake_subprocess.expect_stage2_script(ok=True)  # validate
    fake_subprocess.expect_script(returncode=0)   # iter 1 smoke passes

    result = run_stage_3c(state)
    assert_stage_completed(result, "stage_3c")
    events = [e["event_type"] for e in _events(state)]
    assert "fix_data_signal_failed" not in events
    assert "fix_data_signal_checked" not in events
    # teardown asserts no unconsumed script expectations — the check never ran
