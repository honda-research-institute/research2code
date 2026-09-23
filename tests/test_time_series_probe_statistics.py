"""Zoo-calibrated pure decisions for the five forecasting probes."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from build_plan import TSF_RUNTIME_EXECUTION
from demo_skill_evidence import (
    bind_split_validity_receipt,
    compute_actual_values_id,
    compute_finite_mask_id,
)
from probes.time_series_statistics import (
    MAGNITUDE_MIN_RATIO,
    MIN_SAMPLE_COUNT,
    OWN_HISTORY_MIN_ABSOLUTE_RESPONSE,
    OWN_HISTORY_MIN_RELATIVE_RESPONSE,
    OWN_HISTORY_PERTURBATION_FACTOR,
    PATH_DEPENDENCE_FAIL_MAX,
    PATH_DEPENDENCE_PASS_MIN,
    SAMPLE_SPREAD_RATIO_MIN,
    SUPPORTED_STUDENT_T_OUTPUT_GRAMMARS,
    assess_autoregressive_path_dependence,
    assess_held_out_skill,
    assess_magnitude_collapse,
    assess_own_history_sensitivity,
    assess_sample_genuineness,
    student_t_output_grammar,
)
from taxonomy import load_demo_skill


FIXTURE = json.loads((
    Path(__file__).parent
    / "fixtures" / "probes" / "time_series_forecasting"
    / "statistical_cases.json"
).read_text(encoding="utf-8"))


def _sampling_case(case: dict) -> tuple[np.ndarray, dict]:
    mu = np.asarray(case["mu"], dtype=np.float64)
    scale = np.asarray(case["scale"], dtype=np.float64)
    df = np.asarray(case["df"], dtype=np.float64)
    if case["kind"] == "broadcast_mean":
        samples = np.broadcast_to(
            mu, (case["sample_count"], *mu.shape)
        ).copy()
    else:
        rng = np.random.default_rng(case["seed"])
        standardized = rng.standard_t(
            df, size=(case["sample_count"], *df.shape)
        )
        samples = mu + scale * standardized
    params = {
        "mu": mu,
        case["scale_key"]: scale,
        "df": df,
    }
    if extra := case.get("extra_parameter"):
        params[extra] = scale ** 2
    return samples, params


def _path_case(case: dict, *, seed: int | None = None) -> np.ndarray:
    rng = np.random.default_rng(case["seed"] if seed is None else seed)
    shape = (
        case["sample_count"], case["series_count"], case["horizon"]
    )
    if case["kind"] == "independent_normal":
        return rng.normal(size=shape)
    samples = np.empty(shape, dtype=np.float64)
    samples[:, :, 0] = rng.normal(size=shape[:2])
    for step in range(1, shape[2]):
        samples[:, :, step] = (
            case["feedback"] * samples[:, :, step - 1]
            + case["innovation_scale"] * rng.normal(size=shape[:2])
        )
    return samples


def _history_assessment(case: dict):
    return assess_own_history_sensitivity(
        case["baseline_history"],
        case["perturbed_history"],
        case["baseline_forecast"],
        case["perturbed_forecast"],
        series_index=case["series_index"],
    )


def _skill_inputs(case: dict) -> tuple[dict, dict]:
    rows = ["series-a@8", "series-b@8"]
    positions = [8, 8]
    actuals = case["actuals"]
    mask = [True, True]
    units = "target_units"
    actual_id = compute_actual_values_id(rows, actuals, units)
    mask_id = compute_finite_mask_id(rows, mask)

    def rmse(predictions):
        return math.sqrt(sum(
            (actual - prediction) ** 2
            for actual, prediction in zip(actuals, predictions)
        ) / len(actuals))

    def common(predictions):
        return {
            "row_ids": rows,
            "evaluation_positions": positions,
            "units": units,
            "actual_values_id": actual_id,
            "finite_mask_id": mask_id,
            "predictions": predictions,
        }

    model = case["model_predictions"]
    repeat = case["repeat_last"]
    role = case.get("evaluation_protocol_role", "test_span")
    record = {
        "schema_version": "2.0.0",
        "evaluation_protocol_role": role,
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
    split_receipt = {
        "schema_version": "1.0.0",
        "status": case["receipt_status"],
        "validator": "eval_split_lineage",
        "reasons": ["fixture_split_status"],
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
    evaluation_protocol = {
        "scheme": {
            "kind": "single_holdout",
            "paper_element_ids": ["evaluation-split"],
        },
        "quantities": [{
            "role": role,
            "paper_element_ids": [f"{role}-evidence"],
        }],
    }
    return record, bind_split_validity_receipt(
        split_receipt,
        record,
        evaluation_protocol=evaluation_protocol,
    )


def test_student_t_output_grammar_is_closed_explicit_and_json_normalizable():
    # Tuple containers are JSON arrays; no enum or framework type leaks into
    # the freeze authority.
    frozen = json.loads(json.dumps(
        SUPPORTED_STUDENT_T_OUTPUT_GRAMMARS, allow_nan=False,
    ))
    assert [row["parameter_keys"] for row in frozen] == [
        ["mu", "scale", "df"], ["mu", "sigma", "df"],
    ]
    assert {row["scale_semantics"] for row in frozen} == {"student_t_scale"}

    assert student_t_output_grammar({"mu": 1, "scale": 1, "df": 6})
    assert student_t_output_grammar({"mu": 1, "sigma": 1, "df": 6})
    assert student_t_output_grammar({"mu": 1, "std": 1, "df": 6}) is None
    assert student_t_output_grammar(
        {"mu": 1, "scale": 1, "sigma": 1, "df": 6}
    ) is None
    assert student_t_output_grammar(
        {"mu": 1, "scale": 1, "df": 6, "variance": 1}
    ) is None

    build_plan_grammar = TSF_RUNTIME_EXECUTION["output_grammar"][
        "distribution_params"
    ]
    assert build_plan_grammar["kind"] == "student_t"
    assert build_plan_grammar["closed"] is True
    assert build_plan_grammar["allowed_key_sets"] == [
        {
            "location": row["location_key"],
            "scale": row["scale_key"],
            "degrees_of_freedom": row["degrees_of_freedom_key"],
        }
        for row in SUPPORTED_STUDENT_T_OUTPUT_GRAMMARS
    ]


def test_sample_genuineness_separates_broadcast_and_both_scale_grammars():
    cases = FIXTURE["sample_genuineness"]
    bad = assess_sample_genuineness(*_sampling_case(cases["bad"]))
    scale = assess_sample_genuineness(*_sampling_case(cases["good_scale"]))
    sigma = assess_sample_genuineness(*_sampling_case(cases["good_sigma"]))

    assert bad.status == "fail"
    assert bad.statistics["median_empirical_to_expected_spread_ratio"] == 0.0
    assert scale.status == sigma.status == "pass"
    assert scale.statistics["scale_key"] == "scale"
    assert sigma.statistics["scale_key"] == "sigma"
    assert scale.statistics["median_empirical_to_expected_spread_ratio"] \
        > 2 * SAMPLE_SPREAD_RATIO_MIN
    assert sigma.statistics["median_empirical_to_expected_spread_ratio"] \
        > 2 * SAMPLE_SPREAD_RATIO_MIN


def test_sample_genuineness_is_honest_about_unsupported_or_undefined_shapes():
    case = FIXTURE["sample_genuineness"]["unbindable"]
    unsupported = assess_sample_genuineness(*_sampling_case(case))
    assert unsupported.status == "unprobeable"
    assert unsupported.reason_code == "student_t_output_grammar_unsupported"
    assert unsupported.responsibility == "pipeline_coverage"
    assert "exact supported" in unsupported.reason

    samples, params = _sampling_case(FIXTURE["sample_genuineness"]["good_scale"])
    params["df"] = np.full_like(params["df"], 2.0)
    undefined = assess_sample_genuineness(samples, params)
    assert undefined.status == "unprobeable"
    assert undefined.reason_code == "student_t_variance_undefined"
    assert "undefined" in undefined.reason


def test_sample_genuineness_routes_supported_malformed_outputs_to_producer():
    mean = np.ones((2, 3), dtype=np.float64)
    params = {
        "mu": mean,
        "scale": np.ones_like(mean),
        "df": np.full_like(mean, 6.0),
    }
    wrong_rank = assess_sample_genuineness(mean.copy(), params)
    complex_draws = assess_sample_genuineness(
        np.ones((MIN_SAMPLE_COUNT, 2, 3), dtype=np.complex128), params,
    )

    assert wrong_rank.status == "fail"
    assert wrong_rank.reason_code == "samples_rank_mismatch"
    assert wrong_rank.responsibility == "producer"
    assert complex_draws.status == "fail"
    assert complex_draws.reason_code == "samples_non_real_dtype"
    assert complex_draws.responsibility == "producer"


def test_sample_genuineness_rejects_direct_mean_alias_before_shape_coercion():
    mean = np.ones((2, 3), dtype=np.float64)
    result = assess_sample_genuineness(
        mean,
        {
            "mu": mean,
            "scale": np.ones_like(mean),
            "df": np.full_like(mean, 6.0),
        },
    )
    assert result.status == "fail"
    assert "alias" in result.reason


def test_path_dependence_separates_independent_marginals_from_feedback():
    cases = FIXTURE["path_dependence"]
    bad = assess_autoregressive_path_dependence(
        _path_case(cases["bad"]),
        applicability_status="required",
        sample_genuineness_status="pass",
    )
    good = assess_autoregressive_path_dependence(
        _path_case(cases["good"]),
        applicability_status="required",
        sample_genuineness_status="pass",
    )
    bad_score = bad.statistics[
        "median_absolute_consecutive_residual_correlation"
    ]
    good_score = good.statistics[
        "median_absolute_consecutive_residual_correlation"
    ]

    assert bad.status == "fail"
    assert good.status == "pass"
    assert bad_score < PATH_DEPENDENCE_FAIL_MAX
    assert good_score > PATH_DEPENDENCE_PASS_MIN
    assert good_score - PATH_DEPENDENCE_PASS_MIN \
        > PATH_DEPENDENCE_FAIL_MAX - bad_score


def test_path_dependence_unbindable_and_non_autoregressive_stay_distinct():
    cases = FIXTURE["path_dependence"]
    unbindable = assess_autoregressive_path_dependence(
        _path_case(cases["unbindable"]),
        applicability_status="required",
        sample_genuineness_status="pass",
    )
    not_applicable = assess_autoregressive_path_dependence(
        _path_case(cases["not_applicable"]),
        applicability_status="not_applicable",
        sample_genuineness_status="pass",
    )
    blocked = assess_autoregressive_path_dependence(
        _path_case(cases["good"]),
        applicability_status="required",
        sample_genuineness_status="fail",
    )

    unresolved = assess_autoregressive_path_dependence(
        _path_case(cases["good"]),
        applicability_status="unresolved",
        sample_genuineness_status="pass",
    )
    unknown = assess_autoregressive_path_dependence(
        _path_case(cases["good"]),
        applicability_status="",
        sample_genuineness_status="pass",
    )

    assert unbindable.status == "unprobeable"
    assert not_applicable.status == "not_applicable"
    assert blocked.status == "unprobeable"
    assert unresolved.status == "unprobeable"
    assert unresolved.reason_code == "autoregressive_applicability_unresolved"
    assert unknown.status == "unprobeable"
    assert unknown.reason_code == "autoregressive_applicability_status_unknown"


def test_path_dependence_floor_has_low_false_positive_seed_sweep():
    case = FIXTURE["path_dependence"]["low_false_positive_sweep"]
    assert case["sample_count"] == MIN_SAMPLE_COUNT == 100

    assessments = [
        assess_autoregressive_path_dependence(
            _path_case(case, seed=seed),
            applicability_status="required",
            sample_genuineness_status="pass",
        )
        for seed in case["seeds"]
    ]

    assert all(result.status != "pass" for result in assessments)
    assert max(
        result.statistics[
            "median_absolute_consecutive_residual_correlation"
        ]
        for result in assessments
    ) < PATH_DEPENDENCE_PASS_MIN


def test_own_history_sensitivity_separates_insensitive_and_graph_free_live_paths():
    cases = FIXTURE["own_history_sensitivity"]
    bad = _history_assessment(cases["bad"])
    graph_free_good = _history_assessment(cases["good_graph_free"])

    assert cases["good_graph_free"]["graph"] is None
    assert bad.status == "fail"
    assert bad.statistics["relative_forecast_response"] == 0.0
    assert graph_free_good.status == "pass"
    assert graph_free_good.statistics["perturbation_factor"] == (
        OWN_HISTORY_PERTURBATION_FACTOR
    )
    assert graph_free_good.statistics["relative_forecast_response"] \
        > 100 * OWN_HISTORY_MIN_RELATIVE_RESPONSE


def test_own_history_sensitivity_rejects_noise_and_threshold_equality():
    noise = _history_assessment(
        FIXTURE["own_history_sensitivity"]["numerical_noise"]
    )
    assert noise.status == "fail"
    assert noise.reason_code == "own_history_response_below_tolerance"
    assert noise.statistics["forecast_response_rms"] < (
        OWN_HISTORY_MIN_ABSOLUTE_RESPONSE
    )

    case = FIXTURE["own_history_sensitivity"]["numerical_noise"]
    exact_boundary = dict(case)
    exact_boundary["perturbed_forecast"] = [
        case["perturbed_forecast"][0],
        [OWN_HISTORY_MIN_ABSOLUTE_RESPONSE] * 2,
        case["perturbed_forecast"][2],
    ]
    boundary = _history_assessment(exact_boundary)
    assert boundary.status == "fail"
    assert boundary.statistics["forecast_response_rms"] == (
        OWN_HISTORY_MIN_ABSOLUTE_RESPONSE
    )


def test_own_history_sensitivity_refuses_missing_or_uncontrolled_inputs():
    cases = FIXTURE["own_history_sensitivity"]
    assert _history_assessment(cases["unbindable"]).status == "unprobeable"

    uncontrolled = dict(cases["good_graph_free"])
    uncontrolled["perturbed_history"] = [
        [200.0, 300.0, 400.0, 500.0],
        *uncontrolled["perturbed_history"][1:],
    ]
    result = _history_assessment(uncontrolled)
    assert result.status == "unprobeable"
    assert "outside the selected" in result.reason

    wrong_factor = _history_assessment(cases["wrong_intervention_factor"])
    assert wrong_factor.status == "unprobeable"
    assert wrong_factor.reason_code == "history_intervention_factor_mismatch"


def test_held_out_skill_maps_shared_demo_evidence_without_reimplementation():
    cases = FIXTURE["held_out_skill"]
    assessments = {}
    for name, case in cases.items():
        record, receipt = _skill_inputs(case)
        assessments[name] = assess_held_out_skill(
            load_demo_skill("time_series_forecasting"), record, receipt
        )

    assert assessments["bad"].status == "fail"
    assert assessments["bad"].statistics["skill_status"] == "not_demonstrated"
    assert assessments["good"].status == "pass"
    assert assessments["good"].statistics["skill_status"] == "demonstrated"
    assert assessments["good"].statistics["evaluation_protocol_role"] == {
        "role": "test_span",
        "scheme_kind": "single_holdout",
        "scheme_paper_element_ids": ["evaluation-split"],
        "role_paper_element_ids": ["test_span-evidence"],
    }
    assert assessments["unbindable"].status == "unprobeable"
    assert assessments["unbindable"].statistics[
        "evaluation_validity_status"
    ] == "invalid"
    assert "duplicate split diagnosis" in assessments["unbindable"].reason


def test_held_out_skill_absent_family_contract_is_not_applicable():
    record, receipt = _skill_inputs(FIXTURE["held_out_skill"]["good"])
    result = assess_held_out_skill({}, record, receipt)
    assert result.status == "not_applicable"


def test_magnitude_collapse_separates_unscaled_and_original_unit_forecasts():
    cases = FIXTURE["magnitude_collapse"]
    bad = assess_magnitude_collapse(
        cases["bad"]["forecast"], cases["bad"]["history_tail"]
    )
    good = assess_magnitude_collapse(
        cases["good"]["forecast"], cases["good"]["history_tail"]
    )

    assert bad.status == "fail"
    assert bad.statistics["forecast_to_history_magnitude_ratio"] \
        < MAGNITUDE_MIN_RATIO / 2
    assert bad.statistics["collapsed_entity_indices"] == [0, 1]
    assert good.status == "pass"
    assert good.statistics["forecast_to_history_magnitude_ratio"] \
        > 50 * MAGNITUDE_MIN_RATIO
    assert good.statistics["collapsed_entity_indices"] == []


def test_magnitude_collapse_is_per_entity_and_threshold_is_strict():
    cases = FIXTURE["magnitude_collapse"]
    mixed = assess_magnitude_collapse(
        cases["mixed_entity_collapse"]["forecast"],
        cases["mixed_entity_collapse"]["history_tail"],
    )
    boundary = assess_magnitude_collapse(
        cases["exact_threshold"]["forecast"],
        cases["exact_threshold"]["history_tail"],
    )

    assert mixed.statistics["forecast_to_history_magnitude_ratio"] \
        > MAGNITUDE_MIN_RATIO
    assert mixed.status == "fail"
    assert mixed.statistics["collapsed_entity_indices"] == [0]
    assert boundary.status == "fail"
    assert boundary.statistics[
        "minimum_eligible_entity_ratio"
    ] == MAGNITUDE_MIN_RATIO


def test_magnitude_collapse_missing_and_zero_tail_are_both_unprobeable():
    cases = FIXTURE["magnitude_collapse"]
    unbindable = assess_magnitude_collapse(
        cases["unbindable"]["forecast"],
        cases["unbindable"]["history_tail"],
    )
    zero_history = assess_magnitude_collapse(
        cases["zero_history_unbindable"]["forecast"],
        cases["zero_history_unbindable"]["history_tail"],
    )
    assert unbindable.status == "unprobeable"
    assert zero_history.status == "unprobeable"
    assert "no nonzero scale" in zero_history.reason


def _fixture_assessment_payloads() -> dict[str, dict]:
    payloads: dict[str, dict] = {}
    for name, case in FIXTURE["sample_genuineness"].items():
        payloads[f"sample/{name}"] = assess_sample_genuineness(
            *_sampling_case(case)
        ).to_dict()

    for name, case in FIXTURE["path_dependence"].items():
        if name == "not_applicable":
            applicability = "not_applicable"
        else:
            applicability = "required"
        payloads[f"path/{name}"] = assess_autoregressive_path_dependence(
            _path_case(case),
            applicability_status=applicability,
            sample_genuineness_status="pass",
        ).to_dict()

    for name, case in FIXTURE["own_history_sensitivity"].items():
        payloads[f"history/{name}"] = _history_assessment(case).to_dict()

    contract = load_demo_skill("time_series_forecasting")
    for name, case in FIXTURE["held_out_skill"].items():
        record, receipt = _skill_inputs(case)
        payloads[f"skill/{name}"] = assess_held_out_skill(
            contract, record, receipt,
        ).to_dict()

    for name, case in FIXTURE["magnitude_collapse"].items():
        payloads[f"magnitude/{name}"] = assess_magnitude_collapse(
            case["forecast"], case["history_tail"],
        ).to_dict()
    return payloads


def test_every_fixture_assessment_is_deterministic_and_strict_json():
    first = _fixture_assessment_payloads()
    second = _fixture_assessment_payloads()
    assert first == second

    encoded = json.dumps(first, allow_nan=False, sort_keys=True)
    assert json.loads(encoded) == first


def test_extreme_finite_inputs_never_emit_nonfinite_statistics_or_crash():
    large = 1e308
    samples = np.empty((MIN_SAMPLE_COUNT, 1, 1), dtype=np.float64)
    samples[::2] = large
    samples[1::2] = -large
    sample = assess_sample_genuineness(
        samples,
        {
            "mu": np.zeros((1, 1)),
            "scale": np.full((1, 1), large),
            "df": np.full((1, 1), 6.0),
        },
    )
    magnitude = assess_magnitude_collapse(
        [[large, large]], [[large, large]],
    )
    nonrepresentable_intervention = assess_own_history_sensitivity(
        [[large]], [[large]], [[0.0]], [[0.0]], series_index=0,
    )
    huge_integer = assess_magnitude_collapse([[10 ** 10000]], [[1.0]])

    assert sample.status in {"pass", "fail", "unprobeable"}
    assert magnitude.status == "pass"
    assert nonrepresentable_intervention.status == "unprobeable"
    assert huge_integer.status == "fail"
    for assessment in (
        sample, magnitude, nonrepresentable_intervention, huge_integer,
    ):
        json.dumps(assessment.to_dict(), allow_nan=False)
