"""Generated-package coverage for the schema-2 forecasting probe adapter."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from arch_contract_runtime_plan import normalize_schema2_forecasting_plan
from demo_skill_evidence import (
    bind_split_validity_receipt,
    compute_actual_values_id,
    compute_finite_mask_id,
)
from probes.time_series_forecasting import (
    HOLDOUT_REF,
    MAGNITUDE_REF,
    METHOD_PROBE_REFS,
    PROBE_REFS,
    run_time_series_forecasting_probes,
)
from taxonomy import load_demo_skill
from tests.test_demo_skill_evidence import (
    _evaluation_protocol,
    _record,
    _split_receipt,
)
from tests.test_tsf_runtime_plan import (
    _build_plan,
    _graph_free_contract,
    _method_spec,
    _relational_contract,
)


ALL_METHOD_REFS = frozenset(METHOD_PROBE_REFS)
ALL_PROBE_REFS = frozenset(PROBE_REFS)


def _execution_plan(
    *,
    autoregressive: bool = False,
    representation: str | None = None,
    degree_semantics: str = "source_graph",
) -> dict:
    contract = (
        _graph_free_contract()
        if representation is None
        else _relational_contract(
            representation=representation,
            degree_semantics=degree_semantics,
        )
    )
    # The normalized fixture remains exact, but a statistical sampling probe
    # needs more than the constructor helper's small call-smoke value of 11.
    contract["pluggable_component"]["input"]["num_samples"]["source"][
        "literal"
    ] = 512
    return normalize_schema2_forecasting_plan(
        contract,
        _build_plan(),
        method_spec=_method_spec(autoregressive=autoregressive),
    )


def _write_generated_package(
    run_dir: Path,
    *,
    mode: str = "healthy",
    builder_result: str = "declared",
    output_type: str = "declared",
) -> Path:
    method_dir = run_dir / "method"
    method_dir.mkdir(parents=True)
    (method_dir / "__init__.py").write_text("", encoding="utf-8")
    (method_dir / "model.py").write_text(
        """import random


class ForecastModel:
    def __init__(self, hidden_width):
        self.hidden_width = hidden_width
        self.constructor_draw = random.random()


class WrongModel:
    pass
""",
        encoding="utf-8",
    )
    returned_model = (
        "ForecastModel(hidden_width)"
        if builder_result == "declared"
        else "WrongModel()"
    )
    (method_dir / "training.py").write_text(
        "from .model import ForecastModel, WrongModel\n\n"
        "def build_model(hidden_width):\n"
        f"    return {returned_model}\n",
        encoding="utf-8",
    )
    returned_type = "ForecastResult" if output_type == "declared" else "OtherResult"
    (method_dir / "method.py").write_text(
        f'''from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np


@dataclass
class ForecastResult:
    mean: object
    variance: object
    samples: object
    distribution_params: object


@dataclass
class OtherResult:
    mean: object
    variance: object
    samples: object
    distribution_params: object


MODE = {mode!r}


def _record_call(model, graph, seed, num_samples):
    if graph is None:
        graph_summary = None
    else:
        array = np.asarray(graph)
        graph_summary = {{
            "shape": list(array.shape),
            "minimum": float(array.min()),
            "maximum": float(array.max()),
            "rows_differ": bool(
                array.ndim == 2
                and array.shape[0] >= 2
                and not np.array_equal(array[0], array[1])
            ),
            "asymmetric": bool(
                array.ndim == 2
                and array.shape[0] == array.shape[1]
                and not np.array_equal(array, array.T)
            ),
        }}
    with Path(__file__).with_name("probe_calls.jsonl").open(
        "a", encoding="utf-8"
    ) as handle:
        handle.write(json.dumps({{
            "graph": graph_summary,
            "seed": int(seed),
            "num_samples": int(num_samples),
            "constructor_draw": float(model.constructor_draw),
        }}, sort_keys=True) + "\\n")


# Positional-only model/data inputs and keyword-only controls make any call
# convention drift a TypeError at the generated-code boundary.
def forecast(
    model,
    history,
    static_features,
    time_varying_features,
    graph,
    entity_ids,
    target_scaling_state,
    /,
    *,
    seed,
    num_samples,
):
    _record_call(model, graph, seed, num_samples)
    history = np.asarray(history, dtype=np.float64)
    entity_count = history.shape[0]
    horizon = 2
    if MODE == "known_bad":
        mean = np.full((entity_count, horizon), 1e-12, dtype=np.float64)
    elif MODE == "severed_relational":
        other_series = np.roll(history[:, -1], 1)
        mean = np.repeat(other_series[:, None], horizon, axis=1)
    else:
        mean = np.repeat(history[:, -1, None], horizon, axis=1)
    if MODE == "wrong_shape":
        mean = np.vstack([mean, mean[:1]])
    elif MODE == "permuted_output":
        mean = mean[::-1].copy()

    mu = mean.copy()
    scale = np.ones_like(mean)
    df = np.full_like(mean, 5.0)
    sample_count = int(num_samples)
    if MODE == "known_bad":
        samples = np.broadcast_to(mu, (sample_count, *mu.shape)).copy()
    else:
        rng = np.random.default_rng(int(seed))
        samples = rng.standard_t(5.0, size=(sample_count, *mu.shape))
        if MODE == "ancestral":
            innovation = rng.standard_t(
                5.0, size=(sample_count, entity_count)
            )
            feedback = 0.85
            samples[:, :, 1] = (
                feedback * samples[:, :, 0]
                + np.sqrt(1.0 - feedback ** 2) * innovation
            )
        samples = mu[None, :, :] + scale[None, :, :] * samples
    variance = scale ** 2 * df / (df - 2.0)
    return {returned_type}(
        mean=mean,
        variance=variance,
        samples=samples,
        distribution_params={{"mu": mu, "scale": scale, "df": df}},
    )
''',
        encoding="utf-8",
    )
    return run_dir


def _by_ref(verdicts) -> dict:
    return {verdict.probe_ref: verdict for verdict in verdicts}


def _call_rows(run_dir: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in (run_dir / "method" / "probe_calls.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]


def _ready_heldout_evidence(
    *,
    model: list[float] | None = None,
    repeat: list[float] | None = None,
    validity_status: str = "valid",
) -> dict:
    record = _record(
        model=model,
        repeat=[5.0, 15.0] if repeat is None else repeat,
    )
    record.update({
        "schema_version": "2.0.0",
        "evaluation_protocol_role": "test_span",
    })
    receipt = bind_split_validity_receipt(
        _split_receipt(),
        record,
        evaluation_protocol=_evaluation_protocol(),
    )
    receipt["status"] = validity_status
    if validity_status != "valid":
        receipt["reasons"] = ["synthetic_invalid_split_control"]
    return {
        "status": "ready",
        "family_contract": load_demo_skill("time_series_forecasting"),
        "executed_record": record,
        "validity_receipt": receipt,
        "evaluation_protocol_role": receipt["evaluation_protocol_role"],
    }


def test_graph_free_healthy_package_uses_exact_call_and_repeats_deterministically(
    tmp_path,
):
    run_dir = _write_generated_package(tmp_path / "healthy")
    plan = _execution_plan()

    first = run_time_series_forecasting_probes(
        run_dir,
        enabled_refs=ALL_METHOD_REFS,
        execution_plan=plan,
    )
    second = run_time_series_forecasting_probes(
        run_dir,
        enabled_refs=ALL_METHOD_REFS,
        execution_plan=plan,
    )

    assert [row.to_dict() for row in first] == [
        row.to_dict() for row in second
    ]
    assert {ref: row.verdict for ref, row in _by_ref(first).items()} == {
        "time_series_forecasting.sample_genuineness": "pass",
        "time_series_forecasting.path_dependence": "not_applicable",
        "time_series_forecasting.own_history_sensitivity": "pass",
    }
    assert all(row.bound_callables == ["forecast"] for row in first)
    calls = _call_rows(run_dir)
    assert len(calls) == 8
    assert all(row["graph"] is None for row in calls)
    assert all(row["num_samples"] == 512 and row["seed"] == 7 for row in calls)
    assert len({row["constructor_draw"] for row in calls}) == 1


def test_known_bad_severs_history_broadcasts_samples_and_collapses_magnitude(
    tmp_path,
):
    run_dir = _write_generated_package(tmp_path / "known_bad", mode="known_bad")
    verdicts = _by_ref(run_time_series_forecasting_probes(
        run_dir,
        enabled_refs=ALL_METHOD_REFS,
        execution_plan=_execution_plan(),
    ))

    assert verdicts[
        "time_series_forecasting.sample_genuineness"
    ].verdict == "fail"
    assert verdicts[
        "time_series_forecasting.sample_genuineness"
    ].reason == "sample_spread_too_small"
    assert verdicts[
        "time_series_forecasting.own_history_sensitivity"
    ].verdict == "fail"
    assert verdicts[
        "time_series_forecasting.own_history_sensitivity"
    ].reason == "own_history_response_below_tolerance"


@pytest.mark.parametrize(
    ("mode", "expected"),
    [("healthy", "fail"), ("ancestral", "pass")],
)
def test_autoregressive_independent_paths_fail_and_ancestral_paths_pass(
    tmp_path,
    mode,
    expected,
):
    run_dir = _write_generated_package(tmp_path / mode, mode=mode)
    verdicts = _by_ref(run_time_series_forecasting_probes(
        run_dir,
        enabled_refs={
            "time_series_forecasting.sample_genuineness",
            "time_series_forecasting.path_dependence",
        },
        execution_plan=_execution_plan(autoregressive=True),
    ))

    assert verdicts[
        "time_series_forecasting.sample_genuineness"
    ].verdict == "pass"
    assert verdicts[
        "time_series_forecasting.path_dependence"
    ].verdict == expected


@pytest.mark.parametrize(
    ("representation", "degree_semantics"),
    [
        ("sparse_edge_index", "source_graph"),
        ("dense_adjacency", "induced_graph"),
    ],
)
def test_relational_plans_preserve_asymmetric_source_identity_and_local_bounds(
    tmp_path,
    representation,
    degree_semantics,
):
    run_dir = _write_generated_package(tmp_path / representation)
    plan = _execution_plan(
        representation=representation,
        degree_semantics=degree_semantics,
    )
    verdicts = run_time_series_forecasting_probes(
        run_dir,
        enabled_refs=ALL_METHOD_REFS,
        execution_plan=plan,
    )

    relational = plan["relational"]
    assert relational["source_fixture"]["stable_entity_ids"] == [
        101, 205, 309, 412, 518,
    ]
    assert relational["fitting_identity"]["local_to_source"] == [0, 2, 4]
    assert relational["fitting_identity"]["local_endpoint_bounds"] == [0, 3]
    assert relational["inference_identity"]["local_endpoint_bounds"] == [0, 5]
    assert all(row.verdict != "unprobeable" for row in verdicts)
    history = _by_ref(verdicts)[
        "time_series_forecasting.own_history_sensitivity"
    ]
    assert json.loads(history.evidence)["series_index"] == 3

    call = _call_rows(run_dir)[0]
    assert call["graph"]["rows_differ"] is True
    if representation == "sparse_edge_index":
        assert call["graph"]["shape"] == [2, 7]
        assert call["graph"]["minimum"] >= 0
        assert call["graph"]["maximum"] < 5
    else:
        assert call["graph"]["shape"] == [5, 5]
        assert call["graph"]["asymmetric"] is True


def test_relational_own_history_severance_fails_while_restored_self_path_passes(
    tmp_path,
):
    plan = _execution_plan(representation="sparse_edge_index")
    severed = _write_generated_package(
        tmp_path / "severed-relational",
        mode="severed_relational",
    )
    restored = _write_generated_package(tmp_path / "restored-relational")

    severed_verdict = run_time_series_forecasting_probes(
        severed,
        enabled_refs={"time_series_forecasting.own_history_sensitivity"},
        execution_plan=plan,
    )[0]
    restored_verdict = run_time_series_forecasting_probes(
        restored,
        enabled_refs={"time_series_forecasting.own_history_sensitivity"},
        execution_plan=plan,
    )[0]

    assert severed_verdict.verdict == "fail"
    assert severed_verdict.reason == "own_history_response_below_tolerance"
    assert restored_verdict.verdict == "pass"
    assert restored_verdict.reason == "own_history_response_present"


@pytest.mark.parametrize(
    ("kwargs", "message_fragment"),
    [
        ({"builder_result": "wrong"}, "builder returned"),
        ({"output_type": "wrong"}, "forecast returned"),
    ],
)
def test_supported_builder_and_output_disagreements_are_producer_failures(
    tmp_path,
    kwargs,
    message_fragment,
):
    run_dir = _write_generated_package(tmp_path / "wrong", **kwargs)
    verdicts = run_time_series_forecasting_probes(
        run_dir,
        enabled_refs=ALL_METHOD_REFS,
        execution_plan=_execution_plan(),
    )

    assert len(verdicts) == len(ALL_METHOD_REFS)
    assert all(row.verdict == "fail" for row in verdicts)
    assert all(row.reason == "contract_code_disagreement" for row in verdicts)
    assert all(message_fragment in row.message for row in verdicts)
    expected_callables = (
        ["build_model"]
        if kwargs.get("builder_result") == "wrong"
        else ["build_model", "forecast"]
    )
    assert all(row.bound_callables == expected_callables for row in verdicts)


def test_absent_forecast_reports_no_callable_as_exercised(tmp_path):
    run_dir = _write_generated_package(tmp_path / "missing-forecast")
    method_path = run_dir / "method" / "method.py"
    method_path.write_text(
        method_path.read_text(encoding="utf-8").replace(
            "def forecast(\n", "def forecast_absent(\n", 1
        ),
        encoding="utf-8",
    )

    verdicts = run_time_series_forecasting_probes(
        run_dir,
        enabled_refs=ALL_METHOD_REFS,
        execution_plan=_execution_plan(),
    )

    assert all(row.verdict == "fail" for row in verdicts)
    assert all(row.reason == "contract_code_disagreement" for row in verdicts)
    assert all(row.bound_callables == [] for row in verdicts)


def test_wrong_mean_shape_fails_its_consumers_without_erasing_other_evidence(
    tmp_path,
):
    run_dir = _write_generated_package(tmp_path / "wrong-shape", mode="wrong_shape")
    verdicts = _by_ref(run_time_series_forecasting_probes(
        run_dir,
        enabled_refs=ALL_METHOD_REFS,
        execution_plan=_execution_plan(),
    ))

    assert verdicts[
        "time_series_forecasting.sample_genuineness"
    ].verdict == "pass"
    assert verdicts[
        "time_series_forecasting.path_dependence"
    ].verdict == "not_applicable"
    history = verdicts["time_series_forecasting.own_history_sensitivity"]
    assert history.verdict == "fail"
    assert history.reason == "contract_code_disagreement"
    assert "forecast output shape" in history.message


def test_coherent_entity_permutation_detects_wrong_output_order(tmp_path):
    run_dir = _write_generated_package(
        tmp_path / "permuted-output", mode="permuted_output"
    )
    verdicts = _by_ref(run_time_series_forecasting_probes(
        run_dir,
        enabled_refs=ALL_METHOD_REFS,
        execution_plan=_execution_plan(representation="sparse_edge_index"),
    ))

    assert verdicts[
        "time_series_forecasting.sample_genuineness"
    ].verdict == "pass"
    assert verdicts[
        "time_series_forecasting.path_dependence"
    ].verdict == "not_applicable"
    history = verdicts["time_series_forecasting.own_history_sensitivity"]
    assert history.verdict == "fail"
    assert history.reason == "contract_code_disagreement"
    assert "stable entity order" in history.message


@pytest.mark.parametrize(
    "mutate",
    [
        lambda plan: plan.update({"source_contract_schema": "1.0.0"}),
        lambda plan: plan["construction"]["route"].update(
            {"kind": "legacy_constructor"}
        ),
    ],
)
def test_schema1_and_unsupported_execution_plans_are_pipeline_coverage(
    tmp_path,
    mutate,
):
    run_dir = _write_generated_package(tmp_path / "unsupported")
    plan = copy.deepcopy(_execution_plan())
    mutate(plan)
    verdicts = run_time_series_forecasting_probes(
        run_dir,
        enabled_refs=ALL_METHOD_REFS,
        execution_plan=plan,
    )

    assert len(verdicts) == len(ALL_METHOD_REFS)
    assert all(row.verdict == "unprobeable" for row in verdicts)
    assert all(
        row.reason == "unsupported_forecasting_runtime_grammar"
        for row in verdicts
    )
    assert all(row.bound_callables == [] for row in verdicts)


def test_heldout_typed_role_is_independent_of_runtime_coverage(tmp_path):
    run_dir = _write_generated_package(tmp_path / "heldout")
    unsupported = copy.deepcopy(_execution_plan())
    unsupported["source_contract_schema"] = "1.0.0"
    verdicts = _by_ref(run_time_series_forecasting_probes(
        run_dir,
        enabled_refs={
            HOLDOUT_REF,
            "time_series_forecasting.sample_genuineness",
        },
        execution_plan=unsupported,
        demo_skill_evidence=_ready_heldout_evidence(),
    ))

    heldout = verdicts[HOLDOUT_REF]
    assert heldout.verdict == "pass"
    assert heldout.bound_callables == []
    assert heldout.element_ids == ["evaluation-split", "test_span-evidence"]
    assert verdicts[
        "time_series_forecasting.sample_genuineness"
    ].verdict == "unprobeable"


@pytest.mark.parametrize(
    ("model", "expected", "reason"),
    [
        ([9.0, 19.0], "pass", "forecast_magnitude_preserved"),
        ([1.0e-12, -1.0e-12], "fail", "forecast_magnitude_collapsed"),
    ],
)
def test_magnitude_uses_coherent_executed_rows_without_runtime_call(
    tmp_path,
    model,
    expected,
    reason,
):
    run_dir = tmp_path / expected
    run_dir.mkdir()
    unsupported_plan = copy.deepcopy(_execution_plan())
    unsupported_plan["source_contract_schema"] = "1.0.0"
    groundings = _execution_plan()["probe_groundings"]

    verdict = run_time_series_forecasting_probes(
        run_dir,
        enabled_refs={MAGNITUDE_REF},
        execution_plan=unsupported_plan,
        demo_skill_evidence=_ready_heldout_evidence(
            model=model, repeat=[10.0, 20.0]
        ),
        probe_groundings=groundings,
    )[0]

    assert verdict.verdict == expected
    assert verdict.reason == reason
    assert verdict.bound_callables == []
    assert verdict.element_ids == [
        "target-scaling", "paper-target-scaling",
    ]
    assert not (run_dir / "method" / "probe_calls.jsonl").exists()


def test_magnitude_reports_exact_collapsed_row_identity(tmp_path):
    verdict = run_time_series_forecasting_probes(
        tmp_path,
        enabled_refs={MAGNITUDE_REF},
        demo_skill_evidence=_ready_heldout_evidence(
            model=[1.0e-12, 19.0], repeat=[10.0, 20.0]
        ),
        probe_groundings=_execution_plan()["probe_groundings"],
    )[0]

    assert verdict.verdict == "fail"
    statistics = json.loads(verdict.evidence)
    assert statistics["collapsed_row_ids"] == ["series-a@8"]
    assert "collapsed_entity_indices" not in statistics


def test_magnitude_is_independent_of_split_validity(tmp_path):
    groundings = _execution_plan()["probe_groundings"]
    valid = _by_ref(run_time_series_forecasting_probes(
        tmp_path,
        enabled_refs={HOLDOUT_REF, MAGNITUDE_REF},
        demo_skill_evidence=_ready_heldout_evidence(
            model=[9.0, 19.0], repeat=[5.0, 15.0]
        ),
        probe_groundings=groundings,
    ))
    invalid = _by_ref(run_time_series_forecasting_probes(
        tmp_path,
        enabled_refs={HOLDOUT_REF, MAGNITUDE_REF},
        demo_skill_evidence=_ready_heldout_evidence(
            model=[9.0, 19.0],
            repeat=[5.0, 15.0],
            validity_status="invalid",
        ),
        probe_groundings=groundings,
    ))

    assert valid[MAGNITUDE_REF].to_dict() == invalid[MAGNITUDE_REF].to_dict()
    assert valid[HOLDOUT_REF].verdict == "pass"
    assert invalid[HOLDOUT_REF].verdict == "unprobeable"


def test_magnitude_ignores_unrelated_top_level_evidence_status(tmp_path):
    evidence = _ready_heldout_evidence(
        model=[9.0, 19.0], repeat=[10.0, 20.0]
    )
    evidence["status"] = "unresolved"
    evidence["reasons"] = ["evaluation_protocol_role_unresolved"]

    verdict = run_time_series_forecasting_probes(
        tmp_path,
        enabled_refs={MAGNITUDE_REF},
        demo_skill_evidence=evidence,
        probe_groundings=_execution_plan()["probe_groundings"],
    )[0]

    assert verdict.verdict == "pass"
    assert verdict.reason == "forecast_magnitude_preserved"


def _mutate_schema1(record: dict) -> None:
    record["schema_version"] = "1.0.0"


def _mutate_missing_repeat(record: dict) -> None:
    del record["comparators"]["repeat_last"]


def _mutate_wrong_repeat_implementation(record: dict) -> None:
    record["comparators"]["repeat_last"]["implementation"] = "predict_zero"


def _mutate_leaking_repeat_position(record: dict) -> None:
    record["comparators"]["repeat_last"]["input_positions"] = [8, 7]


def _mutate_repeat_row_order(record: dict) -> None:
    record["comparators"]["repeat_last"]["row_ids"] = list(
        reversed(record["row_ids"])
    )


def _mutate_nonfinite_repeat(record: dict) -> None:
    record["comparators"]["repeat_last"]["input_values"][0] = float("inf")
    record["comparators"]["repeat_last"]["predictions"][0] = float("inf")


def _mutate_ragged_model(record: dict) -> None:
    record["model_predictions"].pop()


@pytest.mark.parametrize(
    ("mutate", "expected", "reason"),
    [
        (_mutate_schema1, "unprobeable", "executed_record_schema_unsupported"),
        (_mutate_missing_repeat, "unprobeable", "repeat_last_evidence_missing"),
        (
            _mutate_wrong_repeat_implementation,
            "fail",
            "repeat_last_implementation_disagrees",
        ),
        (
            _mutate_leaking_repeat_position,
            "fail",
            "repeat_last_not_window_boundary",
        ),
        (_mutate_repeat_row_order, "fail", "repeat_last_identity_mismatch"),
        (_mutate_nonfinite_repeat, "fail", "repeat_last_values_nonfinite"),
        (_mutate_ragged_model, "fail", "magnitude_row_lengths_disagree"),
    ],
)
def test_magnitude_rejects_unresolved_or_incoherent_evidence(
    tmp_path,
    mutate,
    expected,
    reason,
):
    evidence = _ready_heldout_evidence(repeat=[10.0, 20.0])
    mutate(evidence["executed_record"])

    verdict = run_time_series_forecasting_probes(
        tmp_path,
        enabled_refs={MAGNITUDE_REF},
        demo_skill_evidence=evidence,
        probe_groundings=_execution_plan()["probe_groundings"],
    )[0]

    assert verdict.verdict == expected
    assert verdict.reason == reason
    assert verdict.bound_callables == []


def test_magnitude_zero_history_is_unprobeable_and_missing_ref_is_unbound(
    tmp_path,
):
    zero = run_time_series_forecasting_probes(
        tmp_path,
        enabled_refs={MAGNITUDE_REF},
        demo_skill_evidence=_ready_heldout_evidence(
            model=[0.0, 0.0], repeat=[0.0, 0.0]
        ),
        probe_groundings=_execution_plan()["probe_groundings"],
    )[0]
    unbound = run_time_series_forecasting_probes(
        tmp_path,
        enabled_refs={MAGNITUDE_REF},
        demo_skill_evidence=_ready_heldout_evidence(
            model=[9.0, 19.0], repeat=[10.0, 20.0]
        ),
        probe_groundings={MAGNITUDE_REF: {
            "element_ids": [], "paper_element_ids": [],
        }},
    )[0]

    assert zero.verdict == "unprobeable"
    assert zero.reason == "eligible_history_has_no_nonzero_scale"
    assert unbound.verdict == "pass"
    assert unbound.element_ids == []
    assert unbound.bound_callables == []


def test_magnitude_uses_only_finite_mask_selected_rows(tmp_path):
    evidence = _ready_heldout_evidence(
        model=[9.0, 1.0e-12], repeat=[10.0, 1000.0]
    )
    record = evidence["executed_record"]
    record["actuals"][1] = None
    record["finite_mask"][1] = False
    record["model_predictions"][1] = None
    record["actual_values_id"] = compute_actual_values_id(
        record["row_ids"], record["actuals"], record["units"]
    )
    record["finite_mask_id"] = compute_finite_mask_id(
        record["row_ids"], record["finite_mask"]
    )
    for comparator in record["comparators"].values():
        comparator["actual_values_id"] = record["actual_values_id"]
        comparator["finite_mask_id"] = record["finite_mask_id"]

    verdict = run_time_series_forecasting_probes(
        tmp_path,
        enabled_refs={MAGNITUDE_REF},
        demo_skill_evidence=evidence,
        probe_groundings=_execution_plan()["probe_groundings"],
    )[0]

    assert verdict.verdict == "pass"
    assert json.loads(verdict.evidence)["eligible_row_ids"] == ["series-a@8"]
