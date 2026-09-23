"""Stage-2.d execution coverage for the fixed forecasting scaling seam."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import scripts.validate_arch_contract_runtime as runtime_validator
from scripts.arch_contract_semantics import parse_semantic_issue
from scripts.build_plan import STATIC_PLAN_BY_PARADIGM
from scripts.validate_arch_contract_runtime import validate
from tests.test_tsf_runtime_plan import (
    _graph_free_contract,
    _method_spec,
    _relational_contract,
)
import pytest

pytestmark = pytest.mark.probe_runtime


ROOT = Path(__file__).resolve().parent.parent


def _spec() -> dict:
    spec = _method_spec(autoregressive=False)
    spec["comparison"] = {
        "classification": {"id": "time_series_forecasting"},
        "pluggable_component": {
            "name": "forecast",
            "signature": (
                "forecast(model, history, static_features, "
                "time_varying_features, graph, entity_ids, "
                "target_scaling_state, seed, "
                "*paradigm_extras_by_name) -> ForecastResult"
            ),
            "seed_param": "seed",
        },
    }
    return spec


def _write_graph_free_package(
    tmp_path: Path,
    *,
    permute_fit_ids: bool = False,
    dead_transform: bool = False,
    dead_inverse: bool = False,
    discard_transform: bool = False,
    discard_inverse: bool = False,
    partial_inverse: bool = False,
    module_alias_import: bool = False,
    mapping_output: bool = False,
    discard_history: bool = False,
    history_result_key: str = "training_history",
    drift_history_config: bool = False,
    omit_model_result: bool = False,
    alternate_history_outcomes: bool = False,
    no_history_selection: bool = True,
    drift_history_fitting_range: bool = False,
    history_via_helper: bool = False,
) -> tuple[dict, Path]:
    run_dir = tmp_path / "run"
    pipeline = run_dir / ".pipeline"
    method = run_dir / "method"
    pipeline.mkdir(parents=True)
    method.mkdir()
    contract = _graph_free_contract()
    (pipeline / "arch_contract.json").write_text(
        json.dumps(contract), encoding="utf-8"
    )
    spec = _spec()
    (pipeline / "method_spec.json").write_text(
        json.dumps(spec), encoding="utf-8"
    )
    (method / "target_scaling.py").write_text(
        (
            ROOT
            / "paradigms/time_series_forecasting/templates/method/"
            "target_scaling.py.template"
        ).read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    (method / "training_history.py").write_text(
        (
            ROOT
            / "paradigms/time_series_forecasting/templates/method/"
            "training_history.py.template"
        ).read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    (method / "model.py").write_text(
        """import numpy as np


class ForecastModel:
    def __init__(self, hidden_width):
        self.hidden_width = hidden_width

    def forward(self, history):
        return np.asarray(history)[:, :2]

    def __call__(self, history):
        return self.forward(history)
""",
        encoding="utf-8",
    )
    ids_expression = "entity_ids[::-1]" if permute_fit_ids else "entity_ids"
    transform_statement = (
        "    if False:\n"
        "        transform_targets(targets, entity_ids, state)\n"
        if dead_transform
        else (
            "    scaled = scaling.transform_targets(\n"
            "        targets, entity_ids, state\n"
            "    )\n"
            if module_alias_import
            else "    scaled = transform_targets(targets, entity_ids, state)\n"
        )
    )
    model_value = "targets" if dead_transform or discard_transform else "scaled"
    training_import = (
        "from . import target_scaling as scaling\n"
        "from . import training_history as history_recorder\n"
        if module_alias_import
        else "from .target_scaling import (\n"
        "    fit_target_scaling_state, transform_targets,\n"
        ")\n"
        "from .training_history import record_training_history\n"
    )
    fit_call = (
        f"scaling.fit_target_scaling_state({ids_expression}, targets, "
        if module_alias_import
        else f"fit_target_scaling_state({ids_expression}, targets, "
    )
    mapping_import = "from collections import UserDict\n" if mapping_output else ""
    imported_history_call = (
        "history_recorder.record_training_history"
        if module_alias_import else "record_training_history"
    )
    history_call = "_record_history" if history_via_helper \
        else imported_history_call
    history_helper_definition = (
        "def _record_history(**kwargs):\n"
        f"    return {imported_history_call}(**kwargs)\n\n"
        if history_via_helper else ""
    )
    config_expression = (
        'config_id + "-drift"' if drift_history_config else "config_id"
    )
    checkpoint_id = "checkpoint-001"
    loss_index_kind = "epoch"
    loss_observations = [
        {
            "index": 0,
            "value": 3.0,
            "sample_weight": 5,
            "checkpoint_id": "checkpoint-000",
        },
        {
            "index": 1,
            "value": 1.0,
            "sample_weight": 5,
            "checkpoint_id": "checkpoint-001",
        },
        {
            "index": 2,
            "value": 1.0,
            "sample_weight": 5,
            "checkpoint_id": "checkpoint-002",
        },
    ]
    selection = {
        "status": "performed",
        "metric_id": "validation_loss",
        "direction": "minimize",
        "observations": [
            {"index": 0, "value": 2.0, "checkpoint_id": "checkpoint-000"},
            {"index": 1, "value": 1.0, "checkpoint_id": "checkpoint-001"},
            {"index": 2, "value": 1.0, "checkpoint_id": "checkpoint-002"},
        ],
        "selected_checkpoint_id": checkpoint_id,
        "tie_break": "best_then_earliest",
    }
    if alternate_history_outcomes:
        checkpoint_id = "actual-checkpoint-b"
        loss_index_kind = "step"
        loss_observations = [
            {
                "index": 4,
                "value": 8.5,
                "sample_weight": 5,
                "checkpoint_id": "actual-checkpoint-a",
            },
            {
                "index": 9,
                "value": 4.25,
                "sample_weight": 5,
                "checkpoint_id": "actual-checkpoint-b",
            },
            {
                "index": 15,
                "value": 2.75,
                "sample_weight": 5,
                "checkpoint_id": "actual-checkpoint-c",
            },
        ]
        selection = {
            "status": "performed",
            "metric_id": "actual_holdout_mae",
            "direction": "minimize",
            "observations": [
                {
                    "index": 4,
                    "value": 0.9,
                    "checkpoint_id": "actual-checkpoint-a",
                },
                {
                    "index": 9,
                    "value": 0.2,
                    "checkpoint_id": "actual-checkpoint-b",
                },
                {
                    "index": 15,
                    "value": 0.4,
                    "checkpoint_id": "actual-checkpoint-c",
                },
            ],
            "selected_checkpoint_id": checkpoint_id,
            "tie_break": "best_then_earliest",
        }
    if no_history_selection:
        checkpoint_id = loss_observations[-1]["checkpoint_id"]
        selection = {
            "status": "not_performed",
            "reason": "single final checkpoint; no model selection performed",
        }
    selection_range_expression = (
        "None" if no_history_selection else "selection_range"
    )
    fitting_range_expression = (
        "dict(fitting_range, stop=fitting_range['stop'] - 1)"
        if drift_history_fitting_range else "fitting_range"
    )
    history_statement = (
        f"    {history_call}(\n" if discard_history
        else f"    history_record = {history_call}(\n"
    ) + (
        "        model_id=fitting_range[\"model_id\"],\n"
        f"        checkpoint_id={checkpoint_id!r},\n"
        "        seed=int(seed),\n"
        f"        config_id={config_expression},\n"
        "        target_scaling_state=state,\n"
        f"        fitting_range={fitting_range_expression},\n"
        f"        loss_index_kind={loss_index_kind!r},\n"
        f"        loss_observations={loss_observations!r},\n"
        f"        selection_range={selection_range_expression},\n"
        f"        selection={selection!r},\n"
        "    )\n"
    )
    history_value = "None" if discard_history else "history_record"
    model_entry = "" if omit_model_result else '"model": model, '
    result_literal = (
        "{" + model_entry + "\"target_scaling_state\": state, "
        f"{history_result_key!r}: {history_value}}}"
    )
    training_result = (
        f"UserDict({result_literal})" if mapping_output else result_literal
    )
    (method / "training.py").write_text(
        f"{mapping_import}"
        "from .model import ForecastModel\n"
        f"{training_import}\n"
        f"{history_helper_definition}"
        "def build_model(hidden_width):\n"
        "    return ForecastModel(hidden_width)\n\n"
        "def train_model(\n"
        "    model, entity_ids, targets, fitting_range, selection_range, "
        "seed, config_id,\n"
        "):\n"
        f"    state = {fit_call}"
        "fitting_range)\n"
        f"{transform_statement}"
        f"    model({model_value})\n"
        f"{history_statement}"
        f"    return {training_result}\n",
        encoding="utf-8",
    )
    inverse_call = (
        "scaling.inverse_forecast_output"
        if module_alias_import else "inverse_forecast_output"
    )
    if dead_inverse:
        inverse_statement = (
            "    mean = scaled_mean\n"
            "    if False:\n"
            f"        {inverse_call}(mean, entity_ids, "
            "target_scaling_state, output_role=\"location\")\n"
            "    variance = np.ones_like(scaled_mean)\n"
            "    samples = np.repeat(\n"
            "        scaled_mean[None, :, :], int(num_samples), axis=0\n"
            "    )\n"
            "    distribution_scale = np.ones_like(scaled_mean)\n"
        )
    elif discard_inverse:
        discarded_calls = "".join(
            f"    {inverse_call}(\n"
            "        "
            + (
                "np.repeat(scaled_mean[None, :, :], int(num_samples), axis=0)"
                if role == "samples" else "np.ones_like(scaled_mean)"
                if role in {"variance", "distribution_scale"}
                else "scaled_mean"
            )
            + ", entity_ids, target_scaling_state, "
            f"output_role={role!r}"
            + (", entity_axis=1" if role == "samples" else "")
            + "\n    )\n"
            for role in (
                "location", "variance", "samples", "distribution_scale"
            )
        )
        inverse_statement = (
            discarded_calls
            + "    mean = scaled_mean\n"
            "    variance = np.ones_like(scaled_mean)\n"
            "    samples = np.repeat(\n"
            "        scaled_mean[None, :, :], int(num_samples), axis=0\n"
            "    )\n"
            "    distribution_scale = np.ones_like(scaled_mean)\n"
        )
    elif partial_inverse:
        inverse_statement = (
            f"    mean = {inverse_call}(\n"
            "        scaled_mean, entity_ids, target_scaling_state, "
            "output_role=\"location\"\n"
            "    )\n"
            "    variance = np.ones_like(scaled_mean)\n"
            "    samples = np.repeat(\n"
            "        scaled_mean[None, :, :], int(num_samples), axis=0\n"
            "    )\n"
            "    distribution_scale = np.ones_like(scaled_mean)\n"
        )
    else:
        inverse_statement = (
            f"    mean = {inverse_call}(\n"
            "        scaled_mean, entity_ids, target_scaling_state, "
            "output_role=\"location\"\n"
            "    )\n"
            f"    variance = {inverse_call}(\n"
            "        np.ones_like(scaled_mean), entity_ids, "
            "target_scaling_state, output_role=\"variance\"\n"
            "    )\n"
            f"    samples = {inverse_call}(\n"
            "        np.repeat(\n"
            "            scaled_mean[None, :, :], int(num_samples), axis=0\n"
            "        ), entity_ids, target_scaling_state, "
            "output_role=\"samples\", entity_axis=1\n"
            "    )\n"
            f"    distribution_scale = {inverse_call}(\n"
            "        np.ones_like(scaled_mean), entity_ids, "
            "target_scaling_state, output_role=\"distribution_scale\"\n"
            "    )\n"
        )
    method_import = (
        "from . import target_scaling as scaling\n\n"
        if module_alias_import
        else "from .target_scaling import inverse_forecast_output\n\n"
    )
    (method / "method.py").write_text(
        "from dataclasses import dataclass\n"
        "import numpy as np\n"
        f"{method_import}"
        "@dataclass\n"
        "class ForecastResult:\n"
        "    mean: object\n"
        "    variance: object\n"
        "    samples: object\n"
        "    distribution_params: object\n\n"
        "def forecast(model, history, static_features, "
        "time_varying_features, graph, entity_ids, target_scaling_state, "
        "seed, num_samples=11):\n"
        "    scaled_mean = model(history)\n"
        f"{inverse_statement}"
        "    return ForecastResult(mean, variance, samples, "
        "{\"mu\": mean, \"scale\": distribution_scale, "
        "\"df\": np.full_like(mean, 5.0)})\n",
        encoding="utf-8",
    )
    (method / "data.py").write_text("", encoding="utf-8")
    (method / "__init__.py").write_text(
        "from .model import ForecastModel\n"
        "from .method import ForecastResult, forecast\n"
        "from .training import build_model, train_model\n",
        encoding="utf-8",
    )
    return spec, run_dir


def test_graph_free_real_fitting_call_uses_exact_fixed_scaling_helpers(tmp_path):
    spec, run_dir = _write_graph_free_package(tmp_path)

    assert validate(spec, run_dir) == []


def test_real_fitting_rejects_stable_identity_drift(tmp_path):
    spec, run_dir = _write_graph_free_package(tmp_path, permute_fit_ids=True)

    errors = validate(spec, run_dir)

    assert any("exact typed stable ids" in error for error in errors)
    assert all("contract_code_disagreement" in error for error in errors)


def test_static_gate_rejects_dead_training_transform(tmp_path):
    spec, run_dir = _write_graph_free_package(tmp_path, dead_transform=True)

    errors = validate(spec, run_dir)

    assert any(
        "does not reach fixed method.target_scaling.transform_targets" in error
        for error in errors
    )


def test_static_gate_rejects_dead_forecast_inversion(tmp_path):
    spec, run_dir = _write_graph_free_package(tmp_path, dead_inverse=True)

    errors = validate(spec, run_dir)

    assert any(
        "does not reach fixed method.target_scaling.inverse_forecast_output"
        in error
        for error in errors
    )


def test_static_gate_rejects_live_but_discarded_training_transform(tmp_path):
    spec, run_dir = _write_graph_free_package(
        tmp_path, discard_transform=True
    )

    errors = validate(spec, run_dir)

    assert any(
        "transform_targets return" in error
        and "call-and-discard" in error
        for error in errors
    )


def test_static_gate_rejects_live_but_discarded_forecast_inversion(tmp_path):
    spec, run_dir = _write_graph_free_package(
        tmp_path, discard_inverse=True
    )

    errors = validate(spec, run_dir)

    assert any(
        "inverse_forecast_output does not close" in error
        and "call-and-discard" in error
        for error in errors
    )


def test_static_gate_rejects_partial_forecast_inversion(tmp_path):
    spec, run_dir = _write_graph_free_package(
        tmp_path, partial_inverse=True
    )

    errors = validate(spec, run_dir)

    assert any(
        "missing required inverse output role(s)" in error
        and "distribution_scale" in error
        and "samples" in error
        and "variance" in error
        for error in errors
    )


def test_module_alias_helpers_and_mapping_training_result_are_supported(tmp_path):
    spec, run_dir = _write_graph_free_package(
        tmp_path,
        module_alias_import=True,
        mapping_output=True,
    )

    assert validate(spec, run_dir) == []


def test_family_owned_normalization_gap_remains_pipeline_owned(
    tmp_path, monkeypatch
):
    spec, run_dir = _write_graph_free_package(tmp_path)
    build_plan = copy.deepcopy(
        STATIC_PLAN_BY_PARADIGM["time_series_forecasting"]
    )
    del build_plan["target_scaling_execution"]
    monkeypatch.setattr(
        runtime_validator,
        "load_build_plan",
        lambda *args, **kwargs: build_plan,
    )

    errors = runtime_validator.validate(spec, run_dir)

    issues = [parse_semantic_issue(error) for error in errors]
    assert len(issues) == 1
    assert issues[0] is not None
    assert issues[0].code == "unsupported_validator_feature"
    assert issues[0].owner == "pipeline"


def test_relational_fitting_shares_graph_identity_and_scaling_domain(tmp_path):
    run_dir = tmp_path / "relational"
    pipeline = run_dir / ".pipeline"
    method = run_dir / "method"
    pipeline.mkdir(parents=True)
    method.mkdir()
    contract = _relational_contract(
        representation="sparse_edge_index",
        degree_semantics="source_graph",
    )
    spec = _spec()
    (pipeline / "arch_contract.json").write_text(
        json.dumps(contract), encoding="utf-8"
    )
    (method / "target_scaling.py").write_text(
        (
            ROOT
            / "paradigms/time_series_forecasting/templates/method/"
            "target_scaling.py.template"
        ).read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    (method / "training_history.py").write_text(
        (
            ROOT
            / "paradigms/time_series_forecasting/templates/method/"
            "training_history.py.template"
        ).read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    (method / "model.py").write_text(
        """class ForecastModel:
    def __init__(self, hidden_width):
        self.hidden_width = hidden_width

    def forward(self, features, graph, degrees):
        if graph.size and (graph.min() < 0 or graph.max() >= len(features)):
            raise ValueError("local graph endpoint escaped the batch domain")
        return features

    def __call__(self, features, graph, degrees):
        return self.forward(features, graph, degrees)
""",
        encoding="utf-8",
    )
    (method / "training.py").write_text(
        """import numpy as np
from .model import ForecastModel
from .target_scaling import fit_target_scaling_state, transform_targets
from .training_history import record_training_history


class _FitLoss:
    def backward(self):
        return None


def _fit_loss(prediction, targets):
    np.asarray(prediction)
    np.asarray(targets)
    return _FitLoss()


def build_model(hidden_width):
    return ForecastModel(hidden_width)


def _prepare_graph_batch(
    *, source_entity_ids, batch_entity_ids, coindexed, graph, degrees
):
    positions = np.asarray([
        int(np.flatnonzero(source_entity_ids == entity_id)[0])
        for entity_id in batch_entity_ids
    ], dtype=np.int64)
    positions = np.sort(positions)
    source_to_local = np.full(len(source_entity_ids), -1, dtype=np.int64)
    source_to_local[positions] = np.arange(len(positions), dtype=np.int64)
    prepared = {}
    for root, value in coindexed.items():
        axes = [
            axis for axis, size in enumerate(value.shape)
            if size == len(source_entity_ids)
        ]
        if len(axes) != 1:
            raise ValueError("co-indexed root lacks one source entity axis")
        prepared[root] = np.take(value, positions, axis=axes[0])
    source_local = source_to_local[graph[0]]
    destination_local = source_to_local[graph[1]]
    keep = (source_local >= 0) & (destination_local >= 0)
    prepared_graph = np.stack((source_local[keep], destination_local[keep]))
    prepared_degrees = degrees[positions]
    prepared["batch.in_degree"] = prepared_degrees
    return {
        "local_to_source": positions,
        "source_to_local": source_to_local,
        "coindexed": prepared,
        "graph": prepared_graph,
        "degrees": prepared_degrees,
        "output_entity_ids": source_entity_ids[positions],
    }


def train_model(
    model,
    entity_ids,
    batch_entity_ids,
    features,
    targets,
    fitting_range,
    selection_range,
    seed,
    config_id,
    graph,
    degrees,
):
    state = fit_target_scaling_state(entity_ids, targets, fitting_range)
    prepared = _prepare_graph_batch(
        source_entity_ids=entity_ids,
        batch_entity_ids=batch_entity_ids,
        coindexed={
            "batch.entity_ids": entity_ids,
            "batch.demand": features,
            "batch.targets": targets,
            "batch.in_degree": degrees,
        },
        graph=graph,
        degrees=degrees,
    )
    scaled_targets = transform_targets(
        prepared["coindexed"]["batch.targets"],
        prepared["output_entity_ids"],
        state,
    )
    prediction = model(
        features=prepared["coindexed"]["batch.demand"],
        graph=prepared["graph"],
        degrees=prepared["degrees"],
    )
    loss = _fit_loss(prediction, scaled_targets)
    loss.backward()
    training_history = record_training_history(
        model_id=fitting_range["model_id"],
        checkpoint_id="checkpoint-002",
        seed=int(seed),
        config_id=config_id,
        target_scaling_state=state,
        fitting_range=fitting_range,
        loss_index_kind="epoch",
        loss_observations=[
            {"index": 0, "value": 3.0, "sample_weight": 5,
             "checkpoint_id": "checkpoint-000"},
            {"index": 1, "value": 1.0, "sample_weight": 5,
             "checkpoint_id": "checkpoint-001"},
            {"index": 2, "value": 1.0, "sample_weight": 5,
             "checkpoint_id": "checkpoint-002"},
        ],
        selection_range=None,
        selection={
            "status": "not_performed",
            "reason": "schema-2 fixture has no selection-target carrier",
        },
    )
    return {
        "model": model,
        "target_scaling_state": state,
        "training_history": training_history,
    }
""",
        encoding="utf-8",
    )
    (method / "method.py").write_text(
        """from dataclasses import dataclass
import numpy as np
from .target_scaling import inverse_forecast_output


@dataclass
class ForecastResult:
    mean: object
    variance: object
    samples: object
    distribution_params: object


def forecast(
    model, history, static_features, time_varying_features, graph,
    entity_ids, target_scaling_state, seed, num_samples=11,
):
    scaled_mean = np.asarray(history)[:, :2]
    mean = inverse_forecast_output(
        scaled_mean, entity_ids, target_scaling_state, output_role="location"
    )
    variance = inverse_forecast_output(
        np.ones_like(scaled_mean), entity_ids, target_scaling_state,
        output_role="variance",
    )
    samples = inverse_forecast_output(
        np.repeat(scaled_mean[None, :, :], int(num_samples), axis=0),
        entity_ids, target_scaling_state, output_role="samples", entity_axis=1,
    )
    distribution_scale = inverse_forecast_output(
        np.ones_like(scaled_mean), entity_ids, target_scaling_state,
        output_role="distribution_scale",
    )
    return ForecastResult(
        mean, variance, samples,
        {"mu": mean, "scale": distribution_scale, "df": np.full_like(mean, 5.0)},
    )
""",
        encoding="utf-8",
    )
    (method / "data.py").write_text("", encoding="utf-8")
    (method / "__init__.py").write_text(
        "from .model import ForecastModel\n"
        "from .method import ForecastResult, forecast\n"
        "from .training import build_model, train_model\n",
        encoding="utf-8",
    )

    assert validate(spec, run_dir) == []

    training_path = method / "training.py"
    training_source = training_path.read_text(encoding="utf-8")
    old_transform = (
        '        prepared["coindexed"]["batch.targets"],\n'
        '        prepared["output_entity_ids"],\n'
    )
    wrong_subset = (
        '        prepared["coindexed"]["batch.targets"][:1],\n'
        '        prepared["output_entity_ids"][:1],\n'
    )
    assert old_transform in training_source
    training_path.write_text(
        training_source.replace(old_transform, wrong_subset, 1),
        encoding="utf-8",
    )

    errors = validate(spec, run_dir)

    assert any(
        "exact validated fitting targets and stable ids" in error
        for error in errors
    )
