"""Frozen schema-2 execution plans for the forecasting probe kit."""

from __future__ import annotations

import copy
import json

import numpy as np
import pytest

from scripts.arch_contract_runtime_plan import (
    RuntimePlanError,
    execution_plan_digest,
    normalize_schema2_forecasting_plan,
)
from scripts.build_plan import (
    STATIC_PLAN_BY_PARADIGM,
    TSF_RUNTIME_EXECUTION,
    TSF_TARGET_SCALING,
    TSF_TARGET_SCALING_EXECUTION,
    TSF_TRAINING_HISTORY_EXECUTION,
    load_build_plan,
)
from scripts.typed_fixture import (
    FixtureMaterializationError,
    materialize_frozen_forecast_call,
    materialize_frozen_forecast_inputs,
    materialize_frozen_relational_source,
    materialize_frozen_target_scaling_inputs,
    materialize_frozen_training_history_inputs,
    materialize_typed_fixture,
)


def _dimension(value: int) -> dict:
    return {"expression": {"kind": "literal", "value": value}}


def _array(
    kind: str,
    dtype: str,
    *dimensions: str,
    constraint: dict | None = None,
) -> dict:
    descriptor = {
        "kind": kind,
        "dtype": dtype,
        "dimensions": [{"dimension": dimension} for dimension in dimensions],
    }
    if kind == "tensor":
        descriptor["device"] = "cpu"
    if constraint is not None:
        descriptor["constraint"] = constraint
    return descriptor


def _scalar(dtype: str, value) -> dict:
    return {"kind": "scalar", "dtype": dtype, "source": {"literal": value}}


def _opaque(label: str) -> dict:
    return {
        "kind": "opaque",
        "type_description": label,
        "reason": "The family execution binding supplies this exact object.",
    }


def _build_plan() -> dict:
    return {
        "pluggable_component": {"name": "forecast"},
        "runtime_execution": copy.deepcopy(TSF_RUNTIME_EXECUTION),
        "target_scaling": copy.deepcopy(TSF_TARGET_SCALING),
        "target_scaling_execution": copy.deepcopy(
            TSF_TARGET_SCALING_EXECUTION
        ),
        "training_history_execution": copy.deepcopy(
            TSF_TRAINING_HISTORY_EXECUTION
        ),
    }


def _forecast_inputs(graph: dict) -> dict:
    return {
        "model": _opaque("constructed ForecastModel"),
        "history": _array(
            "ndarray", "float64", "source_entity_count", "context_steps"
        ),
        "static_features": _array(
            "ndarray", "float32", "source_entity_count", "feature_width"
        ),
        "time_varying_features": _array(
            "ndarray",
            "float32",
            "source_entity_count",
            "feature_width",
            "context_steps",
        ),
        "graph": copy.deepcopy(graph),
        "entity_ids": _array(
            "ndarray", "int64", "source_entity_count"
        ),
        "target_scaling_state": _opaque(
            "validated target scaling state mapping"
        ),
        "seed": _scalar("int64", 7),
        "num_samples": _scalar("int32", 11),
    }


def _graph_free_contract() -> dict:
    history = _array(
        "ndarray", "float64", "source_entity_count", "context_steps"
    )
    forecast = _array(
        "ndarray", "float64", "source_entity_count", "forecast_steps"
    )
    return {
        "schema_version": "2.0.0",
        "paradigm_id": "time_series_forecasting",
        "dimensions": {
            "source_entity_count": _dimension(5),
            "context_steps": _dimension(3),
            "forecast_steps": _dimension(2),
            "feature_width": _dimension(2),
        },
        "data_loader": {"load_data_returns": {}},
        "architecture": {
            "model": {
                "class_name": "ForecastModel",
                "constructor_args": {
                    "hidden_width": {"literal": 13},
                },
                "forward": {"input": {"history": history}, "output": forecast},
                "additional_methods": {},
            }
        },
        "pluggable_component": {
            "name": "forecast",
            "input": _forecast_inputs(_opaque("None for a graph-free method")),
            "output": _opaque("ForecastResult attribute record"),
        },
        "training_loop": {
            "function_name": "train_model",
            "input": {
                "model": _opaque("constructed ForecastModel"),
                "entity_ids": _array(
                    "ndarray", "int64", "source_entity_count"
                ),
                "targets": copy.deepcopy(history),
                "fitting_range": _opaque(
                    "pipeline-observed fitting target range"
                ),
                "selection_range": _opaque(
                    "pipeline-observed selection target range"
                ),
                "seed": _scalar("int64", 7),
                "config_id": _opaque("deterministic configuration identity"),
            },
            "output": _opaque(
                "mapping with model, target_scaling_state, and training_history"
            ),
        },
        "optimizer_state": None,
        "family_components": {},
        "relational_indexing": None,
    }


def _relational_contract(
    *,
    representation: str = "sparse_edge_index",
    degree_semantics: str = "source_graph",
    output_order: str = "canonical_source_order",
) -> dict:
    source_ids = _array("ndarray", "int64", "source_entity_count")
    batch_ids = _array("ndarray", "int64", "fitting_entity_count")
    features = _array(
        "ndarray", "float32", "source_entity_count", "feature_width"
    )
    degrees = _array("ndarray", "float32", "source_entity_count")
    if representation == "sparse_edge_index":
        graph = _array(
            "ndarray",
            "int64",
            "endpoint_count",
            "edge_count",
            constraint={
                "kind": "index",
                "indexed_dimension": "source_entity_count",
            },
        )
        graph_root = "batch.edge_index"
    else:
        graph = _array(
            "ndarray",
            "float32",
            "source_entity_count",
            "source_entity_count",
        )
        graph_root = "batch.adjacency"

    relational = {
        "schema_version": "1.0",
        "methodology_element_ids": ["graph-message-passing"],
        "entity_axis": "series",
        "stable_entity_id_root": "batch.entity_ids",
        "graph_root": graph_root,
        "representation": representation,
        "source_endpoint_index_space": "source_entity_axis_positions",
        "prepared_endpoint_index_space": "local_batch_positions",
        "edge_orientation": "source_to_destination",
        "coindexed_roots": {
            "batch.entity_ids": 0,
            "batch.demand": 0,
            "batch.targets": 0,
            "batch.static_features": 0,
            "batch.time_varying_features": 0,
            "batch.in_degree": 0,
            "outputs.forecasts": 0,
        },
        "phase_batch_modes": {
            "fitting": "induced_subgraph",
            "model_selection": "canonical_full_graph",
            "inference": "canonical_full_graph",
            "reported_evaluation": "canonical_full_graph",
        },
        "degree_root": "batch.in_degree",
        "degree_kind": "in_degree",
        "degree_semantics": degree_semantics,
        "output_order": output_order,
        "preparation_callable": {
            "module": "method.training",
            "name": "_prepare_graph_batch",
            "tensor_backend": "numpy",
        },
        "execution": {
            "entity_dimension": "source_entity_count",
            "fitting": {
                "model_input_root": "training_loop.input.model",
                "source_entity_ids_input_root": (
                    "training_loop.input.entity_ids"
                ),
                "batch_entity_ids_input_root": (
                    "training_loop.input.batch_entity_ids"
                ),
                "graph_input_root": "training_loop.input.graph",
                "degree_input_root": "training_loop.input.degrees",
                "coindexed_input_roots": {
                    "batch.demand": "training_loop.input.features",
                    "batch.targets": "training_loop.input.targets",
                },
                "fitting_entry_is_one_epoch": True,
            },
            "inference": {
                "architecture_block": "model",
                "graph_input_root": "architecture.model.forward.input.graph",
                "degree_input_root": (
                    "architecture.model.forward.input.degrees"
                ),
                "coindexed_input_roots": {
                    "batch.demand": (
                        "architecture.model.forward.input.features"
                    ),
                },
                "output_coindexed_root": "outputs.forecasts",
            },
        },
    }
    contract = {
        "schema_version": "2.0.0",
        "paradigm_id": "time_series_forecasting",
        "dimensions": {
            "source_entity_count": _dimension(5),
            "fitting_entity_count": _dimension(3),
            "endpoint_count": _dimension(2),
            "edge_count": _dimension(7),
            "context_steps": _dimension(3),
            "forecast_steps": _dimension(2),
            "feature_width": _dimension(2),
        },
        "data_loader": {"load_data_returns": {}},
        "architecture": {
            "model": {
                "class_name": "ForecastModel",
                "constructor_args": {"hidden_width": {"literal": 13}},
                "forward": {
                    "input": {
                        "features": copy.deepcopy(features),
                        "graph": copy.deepcopy(graph),
                        "degrees": copy.deepcopy(degrees),
                    },
                    "output": copy.deepcopy(features),
                },
                "additional_methods": {},
            }
        },
        "pluggable_component": {
            "name": "forecast",
            "input": _forecast_inputs(graph),
            "output": _opaque("ForecastResult attribute record"),
        },
        "training_loop": {
            "function_name": "train_model",
            "input": {
                "model": _opaque("constructed ForecastModel"),
                "entity_ids": source_ids,
                "batch_entity_ids": batch_ids,
                "features": copy.deepcopy(features),
                "targets": _array(
                    "ndarray",
                    "float64",
                    "source_entity_count",
                    "context_steps",
                ),
                "fitting_range": _opaque(
                    "pipeline-observed fitting target range"
                ),
                "selection_range": _opaque(
                    "pipeline-observed selection target range"
                ),
                "seed": _scalar("int64", 7),
                "config_id": _opaque("deterministic configuration identity"),
                "graph": copy.deepcopy(graph),
                "degrees": copy.deepcopy(degrees),
            },
            "output": _opaque(
                "mapping with model, target_scaling_state, and training_history"
            ),
        },
        "optimizer_state": None,
        "family_components": {},
        "relational_indexing": relational,
    }
    return contract


def _method_spec(*, autoregressive: bool) -> dict:
    path_ref = "time_series_forecasting.path_dependence"
    magnitude_ref = "time_series_forecasting.magnitude_collapse"
    return {
        "schema_version": "1.12.0",
        "methodology_replication_contract": {
            "elements": [
                {
                    "element_id": "decoder-feedback",
                    "paper_element_ids": ["paper-decoder-feedback"],
                    "verification_probe_refs": (
                        [path_ref] if autoregressive else []
                    ),
                },
                {
                    "element_id": "target-scaling",
                    "paper_element_ids": ["paper-target-scaling"],
                    "verification_probe_refs": [magnitude_ref],
                },
            ]
        },
    }


def test_committed_tsf_plan_carries_exact_runtime_authority():
    carrier = STATIC_PLAN_BY_PARADIGM["time_series_forecasting"][
        "runtime_execution"
    ]
    assert carrier == TSF_RUNTIME_EXECUTION
    assert carrier["construction"] == {
        "architecture_root": "architecture.model",
        "route": {
            "kind": "builder",
            "module": "method.training",
            "callable": "build_model",
        },
    }
    assert carrier["pluggable_call"]["callable"] == "forecast"
    assert carrier["pluggable_call"]["positional_parameters"] == [
        "model",
        "history",
        "static_features",
        "time_varying_features",
        "graph",
        "entity_ids",
        "target_scaling_state",
    ]
    assert carrier["pluggable_call"]["keyword_parameters"] == ["seed"]
    assert carrier["pluggable_call"]["input_bindings"][
        "pluggable_component.input.model"
    ]["architecture_root"] == "architecture.model"
    scaling = STATIC_PLAN_BY_PARADIGM["time_series_forecasting"][
        "target_scaling"
    ]
    assert scaling == TSF_TARGET_SCALING == {
        "schema_version": "1.0.0",
        "mode": "per_series_affine",
        "supported_modes": ["none", "per_series_affine"],
        "fitting_role": "fitting",
        "entity_id_root": "batch.entity_ids",
        "target_root": "batch.targets",
        "target_entity_axis": 0,
        "target_protocol_axis": 1,
        "statistic": "mean_absolute",
        "centering": "none",
        "epsilon": 1.0e-8,
        "nonfinite_policy": "reject",
        "zero_series_policy": (
            "unit_scale_when_all_absolute_values_lte_epsilon"
        ),
        "state_artifact": ".pipeline/target_scaling_state.json",
        "output_inversion": {
            "location": "multiply_scale",
            "samples": "multiply_scale",
            "distribution_scale": "multiply_absolute_scale",
            "variance": "multiply_squared_scale",
            "unitless_shape": "unchanged",
        },
    }
    scaling_execution = STATIC_PLAN_BY_PARADIGM[
        "time_series_forecasting"
    ]["target_scaling_execution"]
    assert scaling_execution == TSF_TARGET_SCALING_EXECUTION
    assert scaling_execution["training_call"] == {
        "module": "method.training",
        "callable": "train_model",
        "model_input_root": "training_loop.input.model",
        "entity_ids_input_root": "training_loop.input.entity_ids",
        "targets_input_root": "training_loop.input.targets",
        "fitting_range_input_root": "training_loop.input.fitting_range",
        "state_result": {
            "kind": "mapping_key",
            "key": "target_scaling_state",
        },
    }
    history_execution = STATIC_PLAN_BY_PARADIGM[
        "time_series_forecasting"
    ]["training_history_execution"]
    assert history_execution == TSF_TRAINING_HISTORY_EXECUTION
    assert history_execution["training_call"] == {
        "module": "method.training",
        "callable": "train_model",
        "fitting_range_input_root": "training_loop.input.fitting_range",
        "selection_range_input_root": "training_loop.input.selection_range",
        "seed_input_root": "training_loop.input.seed",
        "config_id_input_root": "training_loop.input.config_id",
        "history_result": {
            "kind": "mapping_key",
            "key": "training_history",
        },
    }

    resolved = load_build_plan(
        {
            "comparison": {
                "classification": {"id": "time_series_forecasting"},
                "pluggable_component": {
                    "name": "forecast",
                    "signature": (
                        "forecast(model, history, static_features, "
                        "time_varying_features, graph, entity_ids, "
                        "target_scaling_state, seed) -> ForecastResult"
                    ),
                    "seed_param": "seed",
                },
            }
        }
    )
    assert resolved is not None
    assert resolved["runtime_execution"] == carrier
    assert resolved["target_scaling"] == scaling
    assert resolved["target_scaling_execution"] == scaling_execution
    assert resolved["training_history_execution"] == history_execution
    assert resolved["pluggable_component"]["contract"][
        "fixed_positional_args"
    ] == carrier["pluggable_call"]["positional_parameters"]
    assert resolved["pluggable_component"]["seed_param"] in carrier[
        "pluggable_call"
    ]["keyword_parameters"]


def test_graph_free_plan_is_json_frozen_and_materializes_nonzero_inputs():
    plan = normalize_schema2_forecasting_plan(
        _graph_free_contract(),
        _build_plan(),
        method_spec=_method_spec(autoregressive=False),
    )
    assert json.loads(json.dumps(plan)) == plan
    assert len(execution_plan_digest(plan)) == 64
    assert plan["construction"]["constructor_kwargs"] == {"hidden_width": 13}
    assert plan["construction"]["declared_class"] == {
        "module": "method.model",
        "name": "ForecastModel",
    }
    assert plan["forecast_call"]["input_bindings"]["model"] == {
        "kind": "constructed_model",
        "architecture_block": "model",
    }
    assert plan["forecast_call"]["input_bindings"]["graph"] == {
        "kind": "literal",
        "value": None,
        "semantic_role": "graph",
    }
    assert plan["relational"] is None
    assert plan["target_scaling"]["training_call"] == {
        "module": "method.training",
        "callable": "train_model",
        "roots": {
            "model": "training_loop.input.model",
            "entity_ids": "training_loop.input.entity_ids",
            "targets": "training_loop.input.targets",
            "fitting_range": "training_loop.input.fitting_range",
        },
        "parameters": {
            "model": "model",
            "entity_ids": "entity_ids",
            "targets": "targets",
            "fitting_range": "fitting_range",
        },
        "state_result": {
            "kind": "mapping_key",
            "key": "target_scaling_state",
        },
    }
    assert plan["target_scaling"]["entity_dimension"] == (
        "source_entity_count"
    )
    assert plan["target_scaling"]["protocol_dimension"] == "context_steps"
    scaling_inputs = materialize_frozen_target_scaling_inputs(plan)
    assert scaling_inputs["entity_ids"].tolist() == [101, 205, 309, 412, 518]
    assert scaling_inputs["state"]["entity_count"] == 5
    assert scaling_inputs["fitting_range"] == {
        "validator": "eval_split_lineage",
        "target_root": "batch.targets",
        "model_id": "r2c-schema2-runtime-fixture",
        "start": 0,
        "stop": 3,
        "certainty": "exact",
    }
    assert plan["training_history"]["training_call"] == {
        "module": "method.training",
        "callable": "train_model",
        "roots": {
            "fitting_range": "training_loop.input.fitting_range",
            "selection_range": "training_loop.input.selection_range",
            "seed": "training_loop.input.seed",
            "config_id": "training_loop.input.config_id",
        },
        "parameters": {
            "fitting_range": "fitting_range",
            "selection_range": "selection_range",
            "seed": "seed",
            "config_id": "config_id",
        },
        "history_result": {
            "kind": "mapping_key",
            "key": "training_history",
        },
    }
    history_inputs = materialize_frozen_training_history_inputs(plan)
    assert int(history_inputs["seed"]) == 7
    assert history_inputs["config_id"] == "r2c-schema2-runtime-config-v1"
    assert history_inputs["fitting_range"] == scaling_inputs["fitting_range"]
    assert history_inputs["selection_range"] == {
        "validator": "eval_split_lineage",
        "target_root": "batch.targets",
        "model_id": "r2c-schema2-runtime-fixture",
        "start": 3,
        "stop": 5,
        "certainty": "exact",
    }
    assert all(
        row["sample_weight"] == 5
        for row in history_inputs["loss_observations"]
    )
    assert all(
        set(row) == {"index", "value", "checkpoint_id"}
        for row in history_inputs["selection"]["observations"]
    )
    assert set(history_inputs["record"]) == {
        "schema_version",
        "status",
        "model_id",
        "checkpoint_id",
        "seed",
        "config_id",
        "target_scaling_state_id",
        "fitting_range",
        "loss_index_kind",
        "loss_observations",
        "selection_range",
        "selection",
        "record_digest",
    }
    assert history_inputs["record"]["target_scaling_state_id"] == (
        scaling_inputs["state"]["state_digest"]
    )
    malformed_history_plan = copy.deepcopy(plan)
    del malformed_history_plan["training_history"]["validation_fixture"][
        "inputs"
    ]["config_id"]
    with pytest.raises(
        FixtureMaterializationError,
        match="closed API/record shape",
    ):
        materialize_frozen_training_history_inputs(malformed_history_plan)
    assert materialize_frozen_relational_source(plan) is None
    assert plan["output_grammar"]["distribution_params"] == {
        "kind": "student_t",
        "closed": True,
        "allowed_key_sets": [
            {
                "location": "mu",
                "scale": "scale",
                "degrees_of_freedom": "df",
            },
            {
                "location": "mu",
                "scale": "sigma",
                "degrees_of_freedom": "df",
            },
        ],
    }
    assert plan["output_grammar"]["module"] == "method.method"
    assert plan["forecast_call"]["positional_parameters"] == [
        "model",
        "history",
        "static_features",
        "time_varying_features",
        "graph",
        "entity_ids",
        "target_scaling_state",
    ]
    assert plan["forecast_call"]["keyword_parameters"] == [
        "seed",
        "num_samples",
    ]
    assert plan["applicability"]["autoregressive_path_dependence"][
        "status"
    ] == "not_applicable"

    model = object()
    kwargs = materialize_frozen_forecast_inputs(
        plan, constructed_models={"model": model}
    )
    replay = materialize_frozen_forecast_inputs(
        plan, constructed_models={"model": model}
    )
    assert kwargs["model"] is model
    assert kwargs["graph"] is None
    assert kwargs["entity_ids"].tolist() == [101, 205, 309, 412, 518]
    assert kwargs["target_scaling_state"]["entity_count"] == 5
    assert kwargs["history"].shape == (5, 3)
    assert kwargs["history"].dtype == np.float64
    assert np.all(kwargs["history"] > 0)
    assert float(np.ptp(kwargs["history"])) > 0
    assert np.array_equal(kwargs["history"], replay["history"])
    assert kwargs["static_features"].dtype == np.float32
    assert kwargs["seed"].dtype == np.int64
    assert int(kwargs["seed"]) == 7
    assert kwargs["num_samples"].dtype == np.int32
    args, keyword_args = materialize_frozen_forecast_call(
        plan, constructed_models={"model": model}
    )
    assert args[0] is model
    assert len(args) == 7
    assert list(keyword_args) == ["seed", "num_samples"]
    with pytest.raises(FixtureMaterializationError, match="model.*absent"):
        materialize_frozen_forecast_inputs(plan, constructed_models={})


@pytest.mark.parametrize(
    ("representation", "degree_semantics"),
    [
        ("sparse_edge_index", "source_graph"),
        ("dense_adjacency", "induced_graph"),
    ],
)
def test_relational_plan_preserves_asymmetry_identity_and_typed_source(
    representation,
    degree_semantics,
):
    plan = normalize_schema2_forecasting_plan(
        _relational_contract(
            representation=representation,
            degree_semantics=degree_semantics,
        ),
        _build_plan(),
        method_spec=_method_spec(autoregressive=True),
    )
    relational = plan["relational"]
    assert relational["representation"] == representation
    assert relational["source_endpoint_index_space"] == (
        "source_entity_axis_positions"
    )
    assert relational["prepared_endpoint_index_space"] == "local_batch_positions"
    assert relational["edge_orientation"] == "source_to_destination"
    assert relational["degree_semantics"] == degree_semantics
    assert relational["output_order"] == "canonical_source_order"
    assert plan["target_scaling"]["training_call"]["roots"][
        "entity_ids"
    ] == relational["execution"]["fitting_roots"]["source_entity_ids"]
    assert plan["target_scaling"]["training_call"]["roots"][
        "targets"
    ] == relational["execution"]["fitting_roots"]["coindexed"][
        "batch.targets"
    ]
    assert plan["training_history"]["training_call"]["roots"][
        "fitting_range"
    ] == plan["target_scaling"]["training_call"]["roots"][
        "fitting_range"
    ]
    assert relational["source_fixture"]["stable_entity_ids"] == [
        101,
        205,
        309,
        412,
        518,
    ]
    assert relational["fitting_identity"] == {
        "requested_source_positions": [4, 2, 0],
        "requested_entity_ids": [518, 309, 101],
        "local_to_source": [0, 2, 4],
        "source_to_local": [0, -1, 1, -1, 2],
        "output_entity_ids": [101, 309, 518],
        "local_endpoint_bounds": [0, 3],
    }
    assert relational["inference_identity"]["local_endpoint_bounds"] == [0, 5]
    assert plan["applicability"]["autoregressive_path_dependence"] == {
        "probe_ref": "time_series_forecasting.path_dependence",
        "authority": "methodology_verification_probe_refs",
        "status": "required",
        "element_ids": ["decoder-feedback"],
    }
    assert plan["probe_groundings"][
        "time_series_forecasting.magnitude_collapse"
    ] == {
        "element_ids": ["target-scaling"],
        "paper_element_ids": ["paper-target-scaling"],
    }

    source = materialize_frozen_relational_source(plan)
    assert source is not None
    assert source["stable_entity_ids"].dtype == np.int64
    assert source["graph"].shape == (
        (2, 7) if representation == "sparse_edge_index" else (5, 5)
    )
    if representation == "sparse_edge_index":
        assert source["graph"].dtype == np.int64
        assert int(source["graph"].min()) >= 0
        assert int(source["graph"].max()) < 5
    else:
        assert source["graph"].dtype == np.float32
        assert source["graph"][0, 1] != source["graph"][1, 0]
    assert source["degrees"].dtype == np.float32

    kwargs = materialize_frozen_forecast_inputs(
        plan, constructed_models={"model": object()}
    )
    assert kwargs["graph"] is source["graph"] or np.array_equal(
        kwargs["graph"], source["graph"]
    )


def test_normalizer_rejects_missing_carrier_legacy_and_unbound_opaque():
    with pytest.raises(RuntimePlanError, match="runtime_execution"):
        normalize_schema2_forecasting_plan(_graph_free_contract(), {})

    legacy = {
        "schema_version": "1.0.0",
        "paradigm_id": "time_series_forecasting",
        "data_loader": {"load_data_returns": {}},
        "architecture": {
            "model": {
                "class_name": "ForecastModel",
                "forward": {"input": {}, "output_type": "array"},
                "additional_methods": {},
            }
        },
        "pluggable_component": {
            "name": "forecast",
            "input_shapes": {},
            "output_type": "ForecastResult",
        },
    }
    with pytest.raises(RuntimePlanError, match="schema 2.0.0"):
        normalize_schema2_forecasting_plan(legacy, _build_plan())

    opaque = _graph_free_contract()
    opaque["pluggable_component"]["input"]["history"] = _opaque(
        "unbound history batch"
    )
    with pytest.raises(RuntimePlanError, match="history.*opaque"):
        normalize_schema2_forecasting_plan(opaque, _build_plan())


def test_output_grammar_and_graph_free_none_are_exact_not_inferred():
    wrong_callable = _build_plan()
    wrong_callable["runtime_execution"]["pluggable_call"]["callable"] = (
        "predict"
    )
    with pytest.raises(RuntimePlanError, match="disagrees"):
        normalize_schema2_forecasting_plan(
            _graph_free_contract(), wrong_callable
        )

    wrong_output = _build_plan()
    wrong_output["runtime_execution"]["output_grammar"]["fields"][
        "samples"
    ] = "draws"
    with pytest.raises(RuntimePlanError, match="four exact"):
        normalize_schema2_forecasting_plan(_graph_free_contract(), wrong_output)

    typed_graph = _graph_free_contract()
    typed_graph["pluggable_component"]["input"]["graph"] = _array(
        "ndarray", "float32", "source_entity_count", "source_entity_count"
    )
    with pytest.raises(RuntimePlanError, match="explicitly opaque"):
        normalize_schema2_forecasting_plan(typed_graph, _build_plan())


def test_target_scaling_roots_are_exact_for_graph_free_and_relational_contracts():
    missing_execution = _build_plan()
    del missing_execution["target_scaling_execution"]
    with pytest.raises(RuntimePlanError, match="target_scaling_execution"):
        normalize_schema2_forecasting_plan(
            _graph_free_contract(), missing_execution
        )

    wrong_entity_dimension = _graph_free_contract()
    wrong_entity_dimension["training_loop"]["input"]["entity_ids"] = (
        _array("ndarray", "int64", "forecast_steps")
    )
    with pytest.raises(RuntimePlanError, match="share one exact.*entity"):
        normalize_schema2_forecasting_plan(
            wrong_entity_dimension, _build_plan()
        )

    invented_range_tensor = _graph_free_contract()
    invented_range_tensor["training_loop"]["input"]["fitting_range"] = (
        _array("ndarray", "int64", "forecast_steps")
    )
    with pytest.raises(RuntimePlanError, match="opaque pipeline lineage"):
        normalize_schema2_forecasting_plan(
            invented_range_tensor, _build_plan()
        )

    relational_target_drift = _build_plan()
    relational_target_drift["target_scaling_execution"]["training_call"][
        "targets_input_root"
    ] = "training_loop.input.features"
    with pytest.raises(RuntimePlanError, match="targets root disagrees"):
        normalize_schema2_forecasting_plan(
            _relational_contract(), relational_target_drift
        )


def test_training_history_roots_and_result_are_exact():
    missing_execution = _build_plan()
    del missing_execution["training_history_execution"]
    with pytest.raises(RuntimePlanError, match="training_history_execution"):
        normalize_schema2_forecasting_plan(
            _graph_free_contract(), missing_execution
        )

    typed_selection_range = _graph_free_contract()
    typed_selection_range["training_loop"]["input"]["selection_range"] = (
        _array("ndarray", "int64", "forecast_steps")
    )
    with pytest.raises(RuntimePlanError, match="selection ranges.*opaque"):
        normalize_schema2_forecasting_plan(
            typed_selection_range, _build_plan()
        )

    floating_seed = _graph_free_contract()
    floating_seed["training_loop"]["input"]["seed"] = _scalar(
        "float64", 7.0
    )
    with pytest.raises(RuntimePlanError, match="typed integer scalar"):
        normalize_schema2_forecasting_plan(floating_seed, _build_plan())

    typed_config = _graph_free_contract()
    typed_config["training_loop"]["input"]["config_id"] = _scalar(
        "int64", 3
    )
    with pytest.raises(RuntimePlanError, match="config_id.*opaque"):
        normalize_schema2_forecasting_plan(typed_config, _build_plan())

    fitting_range_drift = _build_plan()
    fitting_range_drift["training_history_execution"]["training_call"][
        "fitting_range_input_root"
    ] = "training_loop.input.selection_range"
    with pytest.raises(RuntimePlanError, match="reuse the exact target-scaling"):
        normalize_schema2_forecasting_plan(
            _graph_free_contract(), fitting_range_drift
        )

    wrong_result = _build_plan()
    wrong_result["training_history_execution"]["training_call"][
        "history_result"
    ]["key"] = "history"
    with pytest.raises(RuntimePlanError, match="exact training_history key"):
        normalize_schema2_forecasting_plan(_graph_free_contract(), wrong_result)


def test_materializer_preserves_tensor_dtype_device_shape_and_bounds():
    torch = pytest.importorskip("torch")
    spec = {
        "kind": "tensor",
        "dtype": "int64",
        "device": "cpu",
        "shape": [2, 7],
        "synthesizable": True,
        "constraint": {"kind": "index", "indexed_dimension": "nodes"},
        "constraint_upper_bound": 5,
    }
    value = materialize_typed_fixture(spec, salt="graph")
    assert isinstance(value, torch.Tensor)
    assert value.dtype == torch.int64
    assert str(value.device) == "cpu"
    assert tuple(value.shape) == (2, 7)
    assert int(value.min().item()) >= 0
    assert int(value.max().item()) < 5
    assert bool(torch.any(value != 0))

    with pytest.raises(FixtureMaterializationError, match="opaque"):
        materialize_typed_fixture(
            {"kind": "opaque", "synthesizable": False}, salt="nope"
        )
