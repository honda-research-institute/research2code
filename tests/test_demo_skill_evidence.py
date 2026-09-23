"""Structured task-skill evidence and comparison policy (R2C-086)."""

from __future__ import annotations

import hashlib
import json
import math
from copy import deepcopy
from pathlib import Path

from scripts.demo_skill_evidence import (
    bind_split_validity_receipt,
    compute_actual_values_id,
    compute_finite_mask_id,
    derive_demo_skill_evidence,
)
from scripts.taxonomy import load_demo_skill


FIXTURES = Path(__file__).parent / "fixtures" / "evidence" / "demo-skill"


def test_actual_identity_uses_canonical_json_null_for_masked_missing_values():
    payload = {
        "actuals": [3.0, None],
        "row_ids": ["a", "b"],
        "units": "target_units",
    }
    canonical = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    expected = "sha256:" + hashlib.sha256(canonical).hexdigest()

    assert compute_actual_values_id(
        payload["row_ids"], [3.0, float("nan")], payload["units"]
    ) == expected


def _record(*, model=None, repeat=None) -> dict:
    rows = ["series-a@8", "series-b@8"]
    actuals = [10.0, 20.0]
    mask = [True, True]
    positions = [8, 8]
    units = "target_units"
    actual_id = compute_actual_values_id(rows, actuals, units)
    mask_id = compute_finite_mask_id(rows, mask)

    def common(predictions):
        return {
            "row_ids": rows,
            "evaluation_positions": positions,
            "units": units,
            "actual_values_id": actual_id,
            "finite_mask_id": mask_id,
            "predictions": predictions,
        }

    repeat = [10.0, 20.0] if repeat is None else repeat
    model = [9.0, 19.0] if model is None else model

    def rmse(predictions):
        return math.sqrt(sum(
            (actual - prediction) ** 2
            for actual, prediction in zip(actuals, predictions)
        ) / len(actuals))

    return {
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
        "model_predictions": model,
        "model_metric": rmse(model),
        "actual_values_id": actual_id,
        "finite_mask_id": mask_id,
        "comparators": {
            "predict_zero": {
                "implementation": "predict_zero",
                "input_positions": [],
                "metric": rmse([0.0, 0.0]),
                **common([0.0, 0.0]),
            },
            "repeat_last": {
                "implementation": "repeat_last_pre_window",
                "input_values": repeat,
                "input_positions": [7, 7],
                "metric": rmse(repeat),
                **common(repeat),
            },
        },
    }


def _split_receipt() -> dict:
    return {
        "schema_version": "1.0.0",
        "status": "valid",
        "validator": "eval_split_lineage",
        "reasons": ["fitting_to_reported_evaluation_disjointness_proved"],
        "evidence": {
            "findings": [],
            "relevant_unresolved": [],
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
                "overlap": None,
                "certainty": "exact",
            }],
        },
    }


def _evaluation_protocol(*, role: str = "test_span") -> dict:
    return {
        "scheme": {
            "kind": "single_holdout",
            "paper_element_ids": ["evaluation-split"],
        },
        "quantities": [{
            "role": role,
            "paper_element_ids": [f"{role}-evidence"],
        }],
    }


def _derive(record: dict, contract: dict | None = None) -> dict:
    receipt = bind_split_validity_receipt(_split_receipt(), record)
    return derive_demo_skill_evidence(
        contract or load_demo_skill("time_series_forecasting"),
        record,
        receipt,
    )


def test_current_pdfgnn_printed_pass_is_invalid_and_skill_undetermined():
    fixture = json.loads((FIXTURES / "current_pdfgnn.json").read_text())

    result = derive_demo_skill_evidence(
        load_demo_skill("time_series_forecasting"),
        fixture["executed_evaluation_record"],
        fixture["evaluation_validity_receipt"],
    )

    assert fixture["printed_verdict"] == "PASS"
    assert fixture["observed_metrics"] == {
        "model_rmse": 6.1784,
        "predict_zero_rmse": 8.6023,
        "repeat_last_rmse": 4.3951,
    }
    assert result["execution"]["status"] == "completed"
    assert result["evaluation_validity"]["status"] == "invalid"
    assert result["skill"]["status"] == "undetermined"


def test_valid_model_beats_zero_but_loses_to_persistence():
    result = _derive(_record())

    assert result["evaluation_validity"]["status"] == "valid"
    assert result["skill"]["status"] == "not_demonstrated"
    outcomes = {row["comparator_id"]: row["status"]
                for row in result["comparisons"]}
    assert outcomes == {"predict_zero": "passed", "repeat_last": "failed"}


def test_valid_model_beats_both_required_comparators():
    result = _derive(_record(repeat=[5.0, 15.0]))

    assert result["skill"]["status"] == "demonstrated"
    assert {row["status"] for row in result["comparisons"]} == {"passed"}


def test_schema2_record_binds_exact_typed_role_without_window_inference():
    record = _record(repeat=[5.0, 15.0])
    record.update({
        "schema_version": "2.0.0",
        "evaluation_protocol_role": "test_span",
    })

    receipt = bind_split_validity_receipt(
        _split_receipt(), record,
        evaluation_protocol=_evaluation_protocol(),
    )
    result = derive_demo_skill_evidence(
        load_demo_skill("time_series_forecasting"), record, receipt,
    )

    carrier = {
        "role": "test_span",
        "scheme_kind": "single_holdout",
        "scheme_paper_element_ids": ["evaluation-split"],
        "role_paper_element_ids": ["test_span-evidence"],
    }
    assert receipt["evaluation_protocol_role"] == carrier
    assert result["evaluation_protocol_role"] == carrier
    assert result["evaluation_validity"]["status"] == "valid"
    assert result["skill"]["status"] == "demonstrated"


def test_schema2_role_is_never_inferred_from_evaluation_window_length():
    record = _record(repeat=[5.0, 15.0])
    record["schema_version"] = "2.0.0"

    receipt = bind_split_validity_receipt(
        _split_receipt(), record,
        evaluation_protocol=_evaluation_protocol(),
    )
    result = derive_demo_skill_evidence(
        load_demo_skill("time_series_forecasting"), record, receipt,
    )

    assert receipt["status"] == "unresolved"
    assert "executed_evaluation_protocol_role_missing_or_unknown" in (
        receipt["reasons"]
    )
    assert "evaluation_protocol_role" not in receipt
    assert result["evaluation_protocol_role"] is None
    assert result["evaluation_validity"]["status"] == "unresolved"
    assert result["skill"]["status"] == "undetermined"


def test_schema2_role_must_resolve_once_against_exact_protocol_identity():
    record = _record(repeat=[5.0, 15.0])
    record.update({
        "schema_version": "2.0.0",
        "evaluation_protocol_role": "test_span",
    })

    receipt = bind_split_validity_receipt(
        _split_receipt(), record,
        evaluation_protocol=_evaluation_protocol(role="validation_span"),
    )

    assert receipt["status"] == "unresolved"
    assert receipt["reasons"][-1] == (
        "executed_evaluation_protocol_role_not_unique"
    )
    assert "evaluation_protocol_role" not in receipt


def test_schema2_rejects_a_receipt_role_without_exact_source_groundings():
    record = _record(repeat=[5.0, 15.0])
    record.update({
        "schema_version": "2.0.0",
        "evaluation_protocol_role": "test_span",
    })
    receipt = bind_split_validity_receipt(
        _split_receipt(), record,
        evaluation_protocol=_evaluation_protocol(),
    )
    del receipt["evaluation_protocol_role"]["role_paper_element_ids"]

    result = derive_demo_skill_evidence(
        load_demo_skill("time_series_forecasting"), record, receipt,
    )

    assert result["evaluation_protocol_role"] is None
    assert result["evaluation_validity"]["status"] == "unresolved"
    assert result["evaluation_validity"]["reasons"][0]["code"] == (
        "executed_evaluation_protocol_role_unresolved"
    )


def test_fake_printed_pass_cannot_override_structured_loss():
    record = _record()
    record["printed_verdict"] = "PASS — trust me"

    assert _derive(record)["skill"]["status"] == "not_demonstrated"


def test_equality_and_tolerance_are_inconclusive():
    equal = _derive(_record(model=[10.0, 20.0]))
    assert equal["skill"]["status"] == "undetermined"
    assert any(row["status"] == "inconclusive" for row in equal["comparisons"])

    contract = deepcopy(load_demo_skill("time_series_forecasting"))
    for item in contract["comparators"]:
        if item["id"] == "repeat_last":
            item["tolerance"] = 0.1
    inside = _derive(
        _record(model=[9.05, 19.05], repeat=[9.0, 19.0]), contract
    )
    assert inside["skill"]["status"] == "undetermined"


def test_missing_or_nonfinite_comparison_evidence_is_undetermined():
    missing = _record(repeat=[5.0, 15.0])
    del missing["comparators"]["repeat_last"]
    assert _derive(missing)["skill"]["status"] == "undetermined"

    nonfinite_model = _record(model=[float("nan"), 19.0])
    result = _derive(nonfinite_model)
    assert result["evaluation_validity"]["status"] == "valid"
    assert result["skill"]["status"] == "undetermined"

    nonfinite_baseline = _record(repeat=[float("inf"), 15.0])
    nonfinite_baseline["comparators"]["repeat_last"]["input_values"] = [
        float("inf"), 15.0,
    ]
    assert _derive(nonfinite_baseline)["skill"]["status"] == "undetermined"


def test_finite_mask_cannot_cherry_pick_finite_rows():
    record = _record(model=[9.0, 0.0], repeat=[5.0, 20.0])
    record["finite_mask"] = [True, False]
    record["model_metric"] = 1.0
    record["finite_mask_id"] = compute_finite_mask_id(
        record["row_ids"], record["finite_mask"]
    )
    for comparator_id, comparator in record["comparators"].items():
        comparator["finite_mask_id"] = record["finite_mask_id"]
        comparator["metric"] = 10.0 if comparator_id == "predict_zero" else 5.0

    result = _derive(record)

    assert result["evaluation_validity"]["status"] == "invalid"
    assert result["evaluation_validity"]["reasons"][0]["code"] == (
        "finite_mask_disagrees_with_actuals"
    )
    assert result["skill"]["status"] == "undetermined"


def test_masked_missing_actual_uses_json_null_and_remains_eligible_control():
    record = _record(repeat=[5.0, 15.0])
    record["actuals"][1] = None
    record["finite_mask"] = [True, False]
    record["actual_values_id"] = compute_actual_values_id(
        record["row_ids"], record["actuals"], record["units"]
    )
    record["finite_mask_id"] = compute_finite_mask_id(
        record["row_ids"], record["finite_mask"]
    )
    record["model_metric"] = 1.0
    for comparator_id, comparator in record["comparators"].items():
        comparator["actual_values_id"] = record["actual_values_id"]
        comparator["finite_mask_id"] = record["finite_mask_id"]
        comparator["metric"] = 10.0 if comparator_id == "predict_zero" else 5.0

    result = _derive(record)

    assert result["evaluation_validity"]["status"] == "valid"
    assert result["skill"]["status"] == "demonstrated"


def test_selected_nonfinite_actual_invalidates_evaluation():
    record = _record(repeat=[5.0, 15.0])
    record["actuals"][0] = float("nan")
    record["actual_values_id"] = compute_actual_values_id(
        record["row_ids"], record["actuals"], record["units"]
    )
    for comparator in record["comparators"].values():
        comparator["actual_values_id"] = record["actual_values_id"]

    result = _derive(record)
    assert result["evaluation_validity"]["status"] == "invalid"
    assert result["skill"]["status"] == "undetermined"


def test_repeat_last_cannot_read_the_first_held_out_actual():
    record = _record(repeat=[5.0, 15.0])
    record["comparators"]["repeat_last"]["input_positions"] = [8, 7]

    result = _derive(record)
    assert result["evaluation_validity"]["status"] == "invalid"
    assert result["skill"]["status"] == "undetermined"


def test_repeat_last_cannot_read_an_earlier_row_of_a_multistep_window():
    record = _record(repeat=[5.0, 15.0])
    record["evaluation_positions"] = [8, 9]
    for comparator in record["comparators"].values():
        comparator["evaluation_positions"] = [8, 9]
    record["comparators"]["repeat_last"]["input_positions"] = [7, 8]
    split = _split_receipt()
    split["evidence"]["fitting_to_reported_evaluation_relations"][0][
        "second"
    ]["protocol_range"]["stop"] = 10

    receipt = bind_split_validity_receipt(split, record)
    result = derive_demo_skill_evidence(
        load_demo_skill("time_series_forecasting"), record, receipt,
    )

    assert receipt["status"] == "valid"
    assert result["evaluation_validity"]["status"] == "invalid"
    assert result["evaluation_validity"]["reasons"][0]["code"] == (
        "repeat_last_leaks_evaluation"
    )


def test_repeat_last_cannot_choose_an_arbitrarily_older_weak_value():
    record = _record(repeat=[5.0, 15.0])
    record["comparators"]["repeat_last"]["input_positions"] = [6, 6]

    result = _derive(record)

    assert result["evaluation_validity"]["status"] == "invalid"
    assert result["evaluation_validity"]["reasons"][0]["code"] == (
        "repeat_last_not_window_boundary"
    )


def test_higher_is_better_contract_uses_family_declared_constant_comparator():
    contract = {
        "schema_version": "1.0.0",
        "primary_metric": {
            "id": "accuracy",
            "direction": "higher_is_better",
            "aggregation": "accuracy",
            "units": "class_id",
        },
        "eligibility": {"minimum_finite_rows": 1,
                        "require_nonzero_actuals": False},
        "decision_policy": "all_required",
        "comparators": [{
            "id": "constant_zero",
            "role": "degeneracy_floor",
            "implementation": "constant_prediction",
            "constant_value": 0.0,
            "required": True,
            "absolute_margin": 0.0,
            "tolerance": 0.0,
        }],
    }
    record = _record(repeat=[5.0, 15.0])
    record.update({
        "metric_id": "accuracy",
        "aggregation": "accuracy",
        "units": "class_id",
        "actuals": [1, 0],
        "model_predictions": [1, 0],
        "model_metric": 1.0,
    })
    record["actual_values_id"] = compute_actual_values_id(
        record["row_ids"], record["actuals"], record["units"]
    )
    record["comparators"] = {"constant_zero": {
        "implementation": "constant_prediction",
        "metric": 0.5,
        "predictions": [0.0, 0.0],
        "row_ids": record["row_ids"],
        "evaluation_positions": record["evaluation_positions"],
        "units": record["units"],
        "actual_values_id": record["actual_values_id"],
        "finite_mask_id": record["finite_mask_id"],
    }}

    result = _derive(record, contract)
    assert result["skill"]["status"] == "demonstrated"

    record["comparators"]["constant_zero"]["predictions"] = [-1.0, -1.0]
    record["comparators"]["constant_zero"]["metric"] = 0.0
    rejected = _derive(record, contract)
    assert rejected["evaluation_validity"]["status"] == "invalid"
    assert rejected["skill"]["status"] == "undetermined"


def test_family_without_comparison_contract_is_not_applicable():
    result = derive_demo_skill_evidence({}, _record(), {})
    assert result["evaluation_validity"]["status"] == "not_applicable"
    assert result["skill"]["status"] == "not_applicable"


def test_valid_receipt_must_bind_exact_model_root_and_positions():
    record = _record(repeat=[5.0, 15.0])
    record["evaluation_positions"] = [9, 9]
    for comparator in record["comparators"].values():
        comparator["evaluation_positions"] = [9, 9]

    bound = bind_split_validity_receipt(_split_receipt(), record)
    assert bound["status"] == "unresolved"
    assert bound["reasons"] == ["executed_window_not_bound_to_split_proof"]


def test_comparator_identity_is_json_type_exact():
    record = _record(repeat=[5.0, 15.0])
    record["row_ids"] = [1, 2]
    record["actual_values_id"] = compute_actual_values_id(
        record["row_ids"], record["actuals"], record["units"]
    )
    record["finite_mask_id"] = compute_finite_mask_id(
        record["row_ids"], record["finite_mask"]
    )
    for comparator in record["comparators"].values():
        comparator["actual_values_id"] = record["actual_values_id"]
        comparator["finite_mask_id"] = record["finite_mask_id"]
        comparator["row_ids"] = [1.0, 2.0]

    result = _derive(record)

    assert result["evaluation_validity"]["status"] == "invalid"
    assert result["evaluation_validity"]["reasons"][0]["code"] == (
        "comparator_identity_mismatch"
    )
