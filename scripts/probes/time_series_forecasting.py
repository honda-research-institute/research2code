"""Exact schema-2 runtime adapter for the forecasting probe kit.

The numerical decisions live in :mod:`probes.time_series_statistics`.  This
module owns the evidence boundary around them: load one normalized execution
plan, construct the declared architecture through its family-owned builder,
materialize the exact typed inputs (including the R2C-084 asymmetric graph),
call ``forecast`` without a spelling or signature ladder, and stamp every
outcome with its exact taxonomy reference and callable binding.

Unsupported contract grammar is a visible pipeline coverage result.  Once a
plan has normalized successfully, a missing callable, rejected legal input,
or malformed supported output is a producer disagreement and therefore a
failing verdict.  The distinction is structural; no exception-message parsing
decides ownership.
"""

from __future__ import annotations

import importlib
import json
import random
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from demo_skill_evidence import validate_forecast_magnitude_evidence
from probes import ProbeVerdict
from probes.package_loader import ProbeLoadError, imported_method_package
from probes.time_series_statistics import (
    SUPPORTED_STUDENT_T_OUTPUT_GRAMMARS,
    StatisticalAssessment,
    assess_autoregressive_path_dependence,
    assess_held_out_skill,
    assess_magnitude_collapse,
    assess_own_history_sensitivity,
    assess_sample_genuineness,
)
from typed_fixture import (
    FixtureMaterializationError,
    materialize_frozen_forecast_call,
)


PROBE_REFS: dict[str, str] = {
    "time_series_forecasting.sample_genuineness": "TSF-1",
    "time_series_forecasting.path_dependence": "TSF-2",
    "time_series_forecasting.own_history_sensitivity": "TSF-3",
    "time_series_forecasting.holdout_skill": "TSF-4",
    "time_series_forecasting.magnitude_collapse": "TSF-5",
}
METHOD_PROBE_REFS = frozenset({
    "time_series_forecasting.sample_genuineness",
    "time_series_forecasting.path_dependence",
    "time_series_forecasting.own_history_sensitivity",
})
HOLDOUT_REF = "time_series_forecasting.holdout_skill"
MAGNITUDE_REF = "time_series_forecasting.magnitude_collapse"


class ForecastPlanCoverageError(ValueError):
    """The pipeline cannot execute this declared grammar without guessing."""


class ForecastContractDisagreement(RuntimeError):
    """Generated code disagrees with a successfully normalized plan."""

    def __init__(
        self,
        message: str,
        *,
        bound_callables: list[str] | tuple[str, ...] = (),
    ) -> None:
        super().__init__(message)
        self.bound_callables = list(bound_callables)


@dataclass(frozen=True)
class ForecastOutput:
    mean: Any
    variance: Any
    samples: Any
    distribution_params: Mapping[str, Any]


def _strict_json(value: object) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def _assessment_message(assessment: StatisticalAssessment) -> tuple[str, str]:
    statistics = dict(assessment.statistics)
    evidence = _strict_json(statistics) if statistics else ""
    return assessment.reason, evidence


def _method_verdict(
    ref: str,
    assessment: StatisticalAssessment,
    *,
    force_status: str | None = None,
) -> ProbeVerdict:
    message, evidence = _assessment_message(assessment)
    return ProbeVerdict(
        PROBE_REFS[ref],
        force_status or assessment.status,
        message,
        evidence=evidence,
        bound_callables=["forecast"],
        reason=str(getattr(assessment, "reason_code", "") or ""),
        probe_ref=ref,
    )


def _coverage_verdict(ref: str, message: str, *, reason: str) -> ProbeVerdict:
    return ProbeVerdict(
        PROBE_REFS[ref],
        "unprobeable",
        message,
        bound_callables=[],
        reason=reason,
        probe_ref=ref,
    )


def _producer_verdict(
    ref: str,
    message: str,
    *,
    reason: str,
    bound_callables: list[str] | tuple[str, ...] = (),
) -> ProbeVerdict:
    return ProbeVerdict(
        PROBE_REFS[ref],
        "fail",
        message,
        bound_callables=list(bound_callables),
        reason=reason,
        probe_ref=ref,
    )


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ForecastPlanCoverageError(f"{label} is unreadable: {exc}") from exc
    if not isinstance(value, dict):
        raise ForecastPlanCoverageError(f"{label} must contain a JSON object")
    return value


def load_live_forecasting_execution_plan(run_dir: Path) -> dict[str, Any]:
    """Resolve the live run through the same schema-2 runtime authority.

    The portable harness never calls this function: it supplies the frozen
    JSON plan recorded at delivery time.  Keeping live resolution here avoids
    vendoring Pydantic, taxonomy loaders, or build-plan logic.
    """

    spec = _load_json(
        run_dir / ".pipeline" / "method_spec.json", label="method_spec.json"
    )
    raw_contract = _load_json(
        run_dir / ".pipeline" / "arch_contract.json",
        label="arch_contract.json",
    )
    try:
        from scripts.arch_contract_runtime_plan import (  # noqa: PLC0415
            RuntimePlanError,
            normalize_schema2_forecasting_plan,
        )
        from scripts.build_plan import load_build_plan  # noqa: PLC0415
        from scripts.taxonomy import run_overlay_dir  # noqa: PLC0415
        from scripts.validate_arch_contract_runtime import (  # noqa: PLC0415
            _load_typed_bundle_dimension_facts,
        )

        repo_root = Path(__file__).resolve().parents[2]
        build_plan = load_build_plan(
            spec,
            repo_root,
            provisional_packs_dir=run_overlay_dir(run_dir),
        )
        if build_plan is None:
            raise ForecastPlanCoverageError(
                "no exact taxonomy build plan serves this forecasting run"
            )
        return normalize_schema2_forecasting_plan(
            raw_contract,
            build_plan,
            bundle_facts=_load_typed_bundle_dimension_facts(run_dir),
            method_spec=spec,
        )
    except ForecastPlanCoverageError:
        raise
    except (RuntimePlanError, KeyError, TypeError, ValueError) as exc:
        raise ForecastPlanCoverageError(str(exc)) from exc


def load_live_forecasting_probe_groundings(
    run_dir: Path,
) -> dict[str, dict[str, list[str]]]:
    """Resolve exact methodology refs without constructing a runtime plan."""

    spec = _load_json(
        run_dir / ".pipeline" / "method_spec.json", label="method_spec.json"
    )
    try:
        from scripts.arch_contract_runtime_plan import (  # noqa: PLC0415
            normalize_forecasting_probe_groundings,
        )

        return normalize_forecasting_probe_groundings(spec)
    except (ImportError, TypeError, ValueError) as exc:
        raise ForecastPlanCoverageError(
            f"forecasting probe grounding is unsupported: {exc}"
        ) from exc


def _output_key_sets(grammar: Mapping[str, Any]) -> set[frozenset[str]]:
    distribution = grammar.get("distribution_params")
    if not isinstance(distribution, Mapping):
        raise ForecastPlanCoverageError(
            "frozen output grammar lacks distribution_params"
        )
    raw_sets = distribution.get("allowed_key_sets")
    if not isinstance(raw_sets, list):
        raise ForecastPlanCoverageError(
            "frozen output grammar lacks closed Student-t key sets"
        )
    roles = {"location", "scale", "degrees_of_freedom"}
    result: set[frozenset[str]] = set()
    for entry in raw_sets:
        if not isinstance(entry, Mapping) or set(entry) != roles:
            raise ForecastPlanCoverageError(
                "frozen Student-t key-set grammar is malformed"
            )
        values = list(entry.values())
        if any(not isinstance(value, str) or not value for value in values):
            raise ForecastPlanCoverageError(
                "frozen Student-t parameter name is malformed"
            )
        result.add(frozenset(values))
    return result


def _validate_output_grammar(plan: Mapping[str, Any]) -> Mapping[str, Any]:
    grammar = plan.get("output_grammar")
    if not isinstance(grammar, Mapping):
        raise ForecastPlanCoverageError("frozen plan lacks output_grammar")
    expected_fields = {
        "mean": "mean",
        "variance": "variance",
        "samples": "samples",
        "distribution_params": "distribution_params",
    }
    if (
        grammar.get("kind") != "attribute_record"
        or grammar.get("module") != "method.method"
        or grammar.get("type_name") != "ForecastResult"
        or grammar.get("fields") != expected_fields
    ):
        raise ForecastPlanCoverageError(
            "frozen ForecastResult grammar disagrees with the committed family"
        )
    frozen_sets = _output_key_sets(grammar)
    statistical_sets = {
        frozenset(row["parameter_keys"])
        for row in SUPPORTED_STUDENT_T_OUTPUT_GRAMMARS
    }
    if frozen_sets != statistical_sets:
        raise ForecastPlanCoverageError(
            "frozen and statistical Student-t grammars disagree"
        )
    return grammar


def _extract_output(
    value: object,
    grammar: Mapping[str, Any],
    output_type: type,
) -> ForecastOutput:
    if not isinstance(value, output_type):
        raise ForecastContractDisagreement(
            f"forecast returned {type(value).__module__}.{type(value).__name__}, "
            f"expected {output_type.__module__}.{output_type.__name__}"
        )
    fields = grammar["fields"]
    extracted: dict[str, Any] = {}
    for role, attribute in fields.items():
        if not hasattr(value, attribute):
            raise ForecastContractDisagreement(
                f"ForecastResult omits declared attribute {attribute!r}"
            )
        extracted[role] = getattr(value, attribute)
    params = extracted["distribution_params"]
    if not isinstance(params, Mapping):
        raise ForecastContractDisagreement(
            "ForecastResult.distribution_params is not a mapping"
        )
    return ForecastOutput(
        mean=extracted["mean"],
        variance=extracted["variance"],
        samples=extracted["samples"],
        distribution_params=params,
    )


def _numeric_array(value: object, *, label: str, rank: int) -> np.ndarray:
    try:
        candidate = value.detach() if hasattr(value, "detach") else value
        candidate = candidate.cpu() if hasattr(candidate, "cpu") else candidate
        candidate = candidate.numpy() if hasattr(candidate, "numpy") else candidate
        raw = np.asarray(candidate)
    except (TypeError, ValueError, RuntimeError, OverflowError) as exc:
        raise ForecastContractDisagreement(
            f"{label} is not a readable numeric array: {exc}"
        ) from exc
    if raw.dtype.kind not in "iuf" or raw.ndim != rank or 0 in raw.shape:
        raise ForecastContractDisagreement(
            f"{label} must be a nonempty real rank-{rank} array; "
            f"got dtype={raw.dtype}, shape={tuple(raw.shape)}"
        )
    array = raw.astype(np.float64, copy=False)
    if not np.isfinite(array).all():
        raise ForecastContractDisagreement(
            f"{label} contains a missing or non-finite value"
        )
    return array


def _copy_value(value: object) -> object:
    if hasattr(value, "clone"):
        return value.clone()
    if hasattr(value, "copy"):
        return value.copy()
    raise ForecastPlanCoverageError(
        "history fixture has no deterministic copy operation"
    )


def _construction_seed(plan: Mapping[str, Any]) -> int:
    call = plan.get("forecast_call")
    bindings = call.get("input_bindings") if isinstance(call, Mapping) else None
    seed_binding = bindings.get("seed") if isinstance(bindings, Mapping) else None
    root = seed_binding.get("root") if isinstance(seed_binding, Mapping) else None
    fixtures = plan.get("fixtures")
    fixture = fixtures.get(root) if isinstance(fixtures, Mapping) else None
    value = fixture.get("scalar_value") if isinstance(fixture, Mapping) else None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ForecastPlanCoverageError(
            "frozen plan lacks one exact integer construction seed"
        )
    return value


def _call_builder_deterministically(
    builder: object,
    constructor_kwargs: Mapping[str, Any],
    *,
    seed: int,
) -> object:
    """Construct without leaking or inheriting process-global RNG state."""
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_module = None
    torch_state = None
    try:
        try:
            import torch as torch_module  # type: ignore[no-redef]  # noqa: PLC0415
            torch_state = torch_module.random.get_rng_state()
        except ImportError:
            torch_module = None
        random.seed(seed)
        np.random.seed(seed % (2 ** 32))
        if torch_module is not None:
            torch_module.manual_seed(seed)
        return builder(**dict(constructor_kwargs))  # type: ignore[operator]
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        if torch_module is not None and torch_state is not None:
            torch_module.random.set_rng_state(torch_state)


def _preferred_series_index(
    plan: Mapping[str, Any],
    history: object,
) -> int:
    """Choose one stable entity, preferring zero relevant graph degree."""
    history_array = _numeric_array(history, label="history", rank=2)
    entity_count = history_array.shape[0]
    relational = plan.get("relational")
    source = relational.get("source_fixture") \
        if isinstance(relational, Mapping) else None
    raw_graph = source.get("graph") if isinstance(source, Mapping) else None
    representation = relational.get("representation") \
        if isinstance(relational, Mapping) else None
    degree_kind = relational.get("degree_kind") \
        if isinstance(relational, Mapping) else None
    if raw_graph is not None and representation == "sparse_edge_index":
        graph = _numeric_array(raw_graph, label="relational graph", rank=2)
        if graph.shape[0] == 2:
            endpoint_row = 1 if degree_kind == "in_degree" else 0
            counts = np.bincount(
                graph[endpoint_row].astype(np.int64), minlength=entity_count
            )
            isolated = np.flatnonzero(counts == 0)
            if isolated.size:
                return int(isolated[0])
    elif raw_graph is not None and representation == "dense_adjacency":
        graph = _numeric_array(raw_graph, label="relational graph", rank=2)
        if graph.shape == (entity_count, entity_count):
            axis = 0 if degree_kind == "in_degree" else 1
            counts = np.count_nonzero(graph, axis=axis)
            isolated = np.flatnonzero(counts == 0)
            if isolated.size:
                return int(isolated[0])
    return 1 if entity_count > 1 else 0


def _parameter_value(
    args: list[object],
    kwargs: dict[str, object],
    call: Mapping[str, Any],
    name: str,
) -> object:
    positional = list(call.get("positional_parameters") or [])
    if name in positional:
        return args[positional.index(name)]
    if name in kwargs:
        return kwargs[name]
    raise ForecastPlanCoverageError(
        f"frozen forecast call does not carry parameter {name!r}"
    )


def _set_parameter_value(
    args: list[object],
    kwargs: dict[str, object],
    call: Mapping[str, Any],
    name: str,
    value: object,
) -> None:
    positional = list(call.get("positional_parameters") or [])
    if name in positional:
        args[positional.index(name)] = value
    elif name in kwargs:
        kwargs[name] = value
    else:
        raise ForecastPlanCoverageError(
            f"frozen forecast call does not carry parameter {name!r}"
        )


def _value_like(original: object, array: np.ndarray) -> object:
    if hasattr(original, "new_tensor"):
        return original.new_tensor(array)  # type: ignore[attr-defined]
    raw = np.asarray(original)
    return np.asarray(array, dtype=raw.dtype)


def _coherently_permuted_call(
    plan: Mapping[str, Any],
    args: list[object],
    kwargs: dict[str, object],
) -> tuple[list[object], dict[str, object], np.ndarray]:
    """Permute every entity-bearing forecast input in one stable-id domain."""
    call = plan.get("forecast_call")
    bindings = call.get("input_bindings") if isinstance(call, Mapping) else None
    if not isinstance(call, Mapping) or not isinstance(bindings, Mapping):
        raise ForecastPlanCoverageError(
            "frozen plan lacks forecast input bindings for permutation control"
        )
    history = _parameter_value(args, kwargs, call, _history_parameter(plan))
    entity_count = _numeric_array(history, label="history", rank=2).shape[0]
    if entity_count < 2:
        return args, kwargs, np.arange(entity_count, dtype=np.int64)
    order = np.roll(np.arange(entity_count, dtype=np.int64), -1)
    permuted_args = list(args)
    permuted_kwargs = dict(kwargs)
    for name, binding in bindings.items():
        if not isinstance(binding, Mapping):
            continue
        role = binding.get("semantic_role")
        if role not in {
            "history",
            "static_features",
            "time_varying_features",
            "stable_entity_ids",
        }:
            continue
        value = _parameter_value(permuted_args, permuted_kwargs, call, str(name))
        array = np.asarray(value.detach().cpu().numpy()) \
            if hasattr(value, "detach") else np.asarray(value)
        if array.shape[0] != entity_count:
            raise ForecastPlanCoverageError(
                f"forecast input {name!r} does not share the entity axis"
            )
        _set_parameter_value(
            permuted_args,
            permuted_kwargs,
            call,
            str(name),
            _value_like(value, np.take(array, order, axis=0)),
        )

    graph_names = [
        str(name) for name, binding in bindings.items()
        if isinstance(binding, Mapping) and binding.get("semantic_role") == "graph"
    ]
    if len(graph_names) != 1:
        raise ForecastPlanCoverageError(
            "frozen forecast call must identify exactly one graph input"
        )
    graph_name = graph_names[0]
    graph_value = _parameter_value(
        permuted_args, permuted_kwargs, call, graph_name
    )
    if graph_value is not None:
        graph = np.asarray(
            graph_value.detach().cpu().numpy()
            if hasattr(graph_value, "detach") else graph_value
        )
        inverse = np.empty(entity_count, dtype=np.int64)
        inverse[order] = np.arange(entity_count, dtype=np.int64)
        relational = plan.get("relational")
        representation = relational.get("representation") \
            if isinstance(relational, Mapping) else None
        if representation == "sparse_edge_index" and graph.shape[0] == 2:
            permuted_graph = inverse[graph.astype(np.int64)]
        elif representation == "dense_adjacency" \
                and graph.shape == (entity_count, entity_count):
            permuted_graph = graph[np.ix_(order, order)]
        else:
            raise ForecastPlanCoverageError(
                "frozen relational graph cannot be coherently permuted"
            )
        _set_parameter_value(
            permuted_args,
            permuted_kwargs,
            call,
            graph_name,
            _value_like(graph_value, permuted_graph),
        )
    return permuted_args, permuted_kwargs, order


def _output_order_error(
    baseline: ForecastOutput,
    permuted: ForecastOutput,
    order: np.ndarray,
) -> str | None:
    baseline_mean = _numeric_array(
        baseline.mean, label="ForecastResult.mean", rank=2
    )
    permuted_mean = _numeric_array(
        permuted.mean, label="permuted ForecastResult.mean", rank=2
    )
    if baseline_mean.shape != permuted_mean.shape \
            or baseline_mean.shape[0] != len(order):
        return "coherent entity permutation changed the forecast output shape"
    restored = permuted_mean[np.argsort(order)]
    if not np.allclose(
        baseline_mean, restored, rtol=1.0e-9, atol=1.0e-12, equal_nan=False
    ):
        return (
            "forecast rows do not preserve the declared stable entity order "
            "under a coherent input/graph permutation"
        )
    return None


def _history_parameter(plan: Mapping[str, Any]) -> str:
    forecast = plan.get("forecast_call")
    bindings = forecast.get("input_bindings") \
        if isinstance(forecast, Mapping) else None
    if not isinstance(bindings, Mapping):
        raise ForecastPlanCoverageError("frozen plan lacks forecast input bindings")
    matches = [
        str(name)
        for name, binding in bindings.items()
        if isinstance(binding, Mapping) and binding.get("semantic_role") == "history"
    ]
    if len(matches) != 1:
        raise ForecastPlanCoverageError(
            "frozen plan must identify exactly one history input"
        )
    return matches[0]


def _entity_view(
    plan: Mapping[str, Any],
    history: object,
    forecast: object,
    *,
    series_index: int,
) -> tuple[np.ndarray, np.ndarray]:
    history_array = _numeric_array(history, label="history", rank=2)
    forecast_array = _numeric_array(forecast, label="ForecastResult.mean", rank=2)
    if history_array.shape[0] != forecast_array.shape[0]:
        raise ForecastContractDisagreement(
            "history and ForecastResult.mean disagree on the stable entity axis"
        )
    if not 0 <= series_index < history_array.shape[0]:
        raise ForecastPlanCoverageError(
            "frozen probe entity is outside the stable source domain"
        )

    relational = plan.get("relational")
    if relational is not None:
        source = relational.get("source_fixture") \
            if isinstance(relational, Mapping) else None
        if not isinstance(source, Mapping):
            raise ForecastPlanCoverageError(
                "frozen relational plan lacks stable source identity"
            )
        stable_ids = source.get("stable_entity_ids")
        if not isinstance(stable_ids, list) \
                or len(stable_ids) != history_array.shape[0] \
                or len(set(stable_ids)) != len(stable_ids):
            raise ForecastPlanCoverageError(
                "frozen relational stable-entity mapping is malformed"
            )
        # Forecast's family contract is source-entity ordered.  The same
        # asymmetric source payload and identity table feed fitting and
        # inference; R2C-084 has already validated every local batch mapping
        # and endpoint bound carried beside it.
        for phase in ("fitting_identity", "inference_identity"):
            identity = relational.get(phase)
            bounds = identity.get("local_endpoint_bounds") \
                if isinstance(identity, Mapping) else None
            output_ids = identity.get("output_entity_ids") \
                if isinstance(identity, Mapping) else None
            if (
                not isinstance(bounds, list)
                or len(bounds) != 2
                or bounds[0] != 0
                or not isinstance(bounds[1], int)
                or bounds[1] < 0
                or not isinstance(output_ids, list)
                or len(output_ids) != bounds[1]
                or any(entity_id not in stable_ids for entity_id in output_ids)
            ):
                raise ForecastPlanCoverageError(
                    f"frozen relational {phase} mapping or local endpoint "
                    "bounds are malformed"
                )
    return history_array, forecast_array


def _construct_and_call(
    run_dir: Path,
    plan: Mapping[str, Any],
) -> tuple[dict[str, Any], ForecastOutput, ForecastOutput, ForecastOutput]:
    if plan.get("schema_version") != "1.0" \
            or plan.get("source_contract_schema") != "2.0.0":
        raise ForecastPlanCoverageError(
            "forecast probes require one normalized schema-2 execution plan"
        )
    grammar = _validate_output_grammar(plan)
    construction = plan.get("construction")
    forecast_call = plan.get("forecast_call")
    if not isinstance(construction, Mapping) or not isinstance(
        forecast_call, Mapping
    ):
        raise ForecastPlanCoverageError(
            "frozen plan lacks construction or forecast_call"
        )
    route = construction.get("route")
    if not isinstance(route, Mapping) or route.get("kind") != "builder":
        raise ForecastPlanCoverageError(
            "frozen construction is not an exact builder route"
        )
    constructor_kwargs = construction.get("constructor_kwargs")
    if not isinstance(constructor_kwargs, Mapping):
        raise ForecastPlanCoverageError(
            "frozen construction lacks exact constructor kwargs"
        )
    architecture_block = construction.get("architecture_block")
    if not isinstance(architecture_block, str):
        raise ForecastPlanCoverageError(
            "frozen construction lacks architecture-block identity"
        )

    builder_module_name = route.get("module")
    builder_name = route.get("callable")
    forecast_module_name = forecast_call.get("module")
    forecast_name = forecast_call.get("callable")
    if any(
        not isinstance(name, str) or not name
        for name in (
            builder_module_name,
            builder_name,
            forecast_module_name,
            forecast_name,
        )
    ):
        raise ForecastPlanCoverageError(
            "frozen callable route contains a missing identity"
        )

    exercised_callables: list[str] = []
    try:
        with imported_method_package(run_dir):
            builder_module = importlib.import_module(builder_module_name)
            forecast_module = importlib.import_module(forecast_module_name)
            builder = getattr(builder_module, builder_name, None)
            forecast_fn = getattr(forecast_module, forecast_name, None)
            if not callable(builder):
                raise ForecastContractDisagreement(
                    f"declared builder {builder_module_name}.{builder_name} is absent"
                )
            if not callable(forecast_fn):
                raise ForecastContractDisagreement(
                    f"declared forecasting entry {forecast_module_name}."
                    f"{forecast_name} "
                    "is absent"
                )
            declared_class = construction.get("declared_class")
            if not isinstance(declared_class, Mapping):
                raise ForecastPlanCoverageError(
                    "frozen construction lacks exact declared-class identity"
                )
            class_module_name = declared_class.get("module")
            class_name = declared_class.get("name")
            output_module_name = grammar.get("module")
            output_type_name = grammar.get("type_name")
            if any(
                not isinstance(name, str) or not name
                for name in (
                    class_module_name,
                    class_name,
                    output_module_name,
                    output_type_name,
                )
            ):
                raise ForecastPlanCoverageError(
                    "frozen construction/output type identity is malformed"
                )
            class_module = importlib.import_module(class_module_name)
            output_module = importlib.import_module(output_module_name)
            model_type = getattr(class_module, class_name, None)
            output_type = getattr(output_module, output_type_name, None)
            if not isinstance(model_type, type) or not isinstance(output_type, type):
                raise ForecastContractDisagreement(
                    "declared model or ForecastResult class is absent"
                )
            try:
                exercised_callables.append(str(builder_name))
                model = _call_builder_deterministically(
                    builder,
                    constructor_kwargs,
                    seed=_construction_seed(plan),
                )
            except Exception as exc:  # noqa: BLE001 - subject code disagreement
                raise ForecastContractDisagreement(
                    f"declared builder rejected exact constructor kwargs: "
                    f"{type(exc).__name__}: {exc}"
                ) from exc
            if not isinstance(model, model_type):
                raise ForecastContractDisagreement(
                    f"declared builder returned {type(model).__module__}."
                    f"{type(model).__name__}, expected "
                    f"{model_type.__module__}.{model_type.__name__}"
                )
            try:
                baseline_args, baseline_kwargs = materialize_frozen_forecast_call(
                    plan,
                    constructed_models={architecture_block: model},
                )
                repeated_args, repeated_kwargs = materialize_frozen_forecast_call(
                    plan,
                    constructed_models={architecture_block: model},
                )
                perturbed_args, perturbed_kwargs = materialize_frozen_forecast_call(
                    plan,
                    constructed_models={architecture_block: model},
                )
                permutation_args, permutation_kwargs = (
                    materialize_frozen_forecast_call(
                        plan,
                        constructed_models={architecture_block: model},
                    )
                )
            except FixtureMaterializationError as exc:
                raise ForecastPlanCoverageError(str(exc)) from exc
            try:
                exercised_callables.append(str(forecast_name))
                baseline_raw = forecast_fn(*baseline_args, **baseline_kwargs)
                baseline = _extract_output(baseline_raw, grammar, output_type)
                repeated_raw = forecast_fn(*repeated_args, **repeated_kwargs)
                repeated = _extract_output(repeated_raw, grammar, output_type)
            except Exception as exc:  # noqa: BLE001 - subject code disagreement
                if isinstance(exc, ForecastContractDisagreement):
                    raise
                raise ForecastContractDisagreement(
                    "declared forecast rejected the normalized typed fixture: "
                    f"{type(exc).__name__}: {exc}"
                ) from exc

            history_name = _history_parameter(plan)
            positional_parameters = list(
                forecast_call.get("positional_parameters") or []
            )
            if history_name in positional_parameters:
                history_position = positional_parameters.index(history_name)
                baseline_history = baseline_args[history_position]
                perturbed_history = _copy_value(
                    perturbed_args[history_position]
                )
            else:
                baseline_history = baseline_kwargs[history_name]
                perturbed_history = _copy_value(
                    perturbed_kwargs[history_name]
                )
            series_index = _preferred_series_index(plan, perturbed_history)
            perturbed_history[series_index] = (
                perturbed_history[series_index] * 100
            )
            if history_name in positional_parameters:
                perturbed_args[history_position] = perturbed_history
            else:
                perturbed_kwargs[history_name] = perturbed_history
            try:
                perturbed_raw = forecast_fn(
                    *perturbed_args, **perturbed_kwargs
                )
                perturbed = _extract_output(
                    perturbed_raw, grammar, output_type
                )
            except Exception as exc:  # noqa: BLE001 - subject code disagreement
                if isinstance(exc, ForecastContractDisagreement):
                    raise
                raise ForecastContractDisagreement(
                    "declared forecast rejected the controlled 100x history "
                    f"intervention: {type(exc).__name__}: {exc}"
                ) from exc

            permutation_error = None
            try:
                (
                    permutation_args,
                    permutation_kwargs,
                    permutation_order,
                ) = _coherently_permuted_call(
                    plan, permutation_args, permutation_kwargs
                )
                permutation_raw = forecast_fn(
                    *permutation_args, **permutation_kwargs
                )
                permutation_output = _extract_output(
                    permutation_raw, grammar, output_type
                )
                permutation_error = _output_order_error(
                    baseline, permutation_output, permutation_order
                )
            except (ForecastContractDisagreement, ForecastPlanCoverageError) as exc:
                permutation_error = str(exc)
            except Exception as exc:  # noqa: BLE001 - subject code disagreement
                permutation_error = (
                    "declared forecast rejected the coherent entity/graph "
                    f"permutation: {type(exc).__name__}: {exc}"
                )

            return (
                {
                    "baseline_history": baseline_history,
                    "perturbed_history": perturbed_history,
                    "series_index": series_index,
                    "output_order_error": permutation_error,
                },
                baseline,
                repeated,
                perturbed,
            )
    except ForecastContractDisagreement as exc:
        if not exc.bound_callables:
            exc.bound_callables = list(dict.fromkeys(exercised_callables))
        raise
    except ProbeLoadError:
        raise
    except ImportError as exc:
        raise ProbeLoadError(
            f"forecast package requires a missing dependency: {exc}",
            missing_dependency=getattr(exc, "name", None),
        ) from exc


def _repeatable(left: ForecastOutput, right: ForecastOutput) -> bool:
    try:
        for first, second, rank in (
            (left.mean, right.mean, 2),
            (left.samples, right.samples, 3),
        ):
            a = _numeric_array(first, label="repeatability baseline", rank=rank)
            b = _numeric_array(second, label="repeatability replay", rank=rank)
            if a.shape != b.shape or not np.array_equal(a, b):
                return False
        return True
    except ForecastContractDisagreement:
        return False


def _method_assessments(
    plan: Mapping[str, Any],
    call_inputs: Mapping[str, Any],
    baseline: ForecastOutput,
    repeated: ForecastOutput,
    perturbed: ForecastOutput,
) -> dict[str, StatisticalAssessment]:
    sample = assess_sample_genuineness(
        baseline.samples, baseline.distribution_params
    )
    applicability = plan.get("applicability")
    path = applicability.get("autoregressive_path_dependence") \
        if isinstance(applicability, Mapping) else None
    status = path.get("status") if isinstance(path, Mapping) else "unresolved"
    path_assessment = assess_autoregressive_path_dependence(
        baseline.samples,
        applicability_status=status,
        sample_genuineness_status=sample.status,
    )

    try:
        series_index = call_inputs.get("series_index")
        if isinstance(series_index, bool) or not isinstance(series_index, int):
            raise ForecastPlanCoverageError(
                "frozen runtime did not preserve the probed stable entity"
            )
        output_order_problem = call_inputs.get("output_order_error")
        if output_order_problem:
            raise ForecastContractDisagreement(str(output_order_problem))
        base_history, base_mean = _entity_view(
            plan,
            call_inputs["baseline_history"],
            baseline.mean,
            series_index=series_index,
        )
        perturbed_history, perturbed_mean = _entity_view(
            plan,
            call_inputs["perturbed_history"],
            perturbed.mean,
            series_index=series_index,
        )
        if not _repeatable(baseline, repeated):
            history_assessment = StatisticalAssessment(
                status="unprobeable",
                reason=(
                    "same-seed forecast replay is not deterministic enough "
                    "to judge own-history sensitivity"
                ),
                reason_code="same_seed_forecast_not_repeatable",
                responsibility="evidence",
                statistics={"repeatable": False},
            )
        else:
            history_assessment = assess_own_history_sensitivity(
                base_history,
                perturbed_history,
                base_mean,
                perturbed_mean,
                series_index=series_index,
            )
    except ForecastContractDisagreement as exc:
        history_assessment = StatisticalAssessment(
            status="fail",
            reason=str(exc),
            reason_code="contract_code_disagreement",
            responsibility="producer",
        )
    except ForecastPlanCoverageError as exc:
        history_assessment = StatisticalAssessment(
            status="unprobeable",
            reason=str(exc),
            reason_code="unsupported_forecasting_runtime_grammar",
            responsibility="pipeline_coverage",
        )

    return {
        "time_series_forecasting.sample_genuineness": sample,
        "time_series_forecasting.path_dependence": path_assessment,
        "time_series_forecasting.own_history_sensitivity": history_assessment,
    }


def _heldout_verdict(evidence: object) -> ProbeVerdict:
    ref = HOLDOUT_REF
    if not isinstance(evidence, Mapping):
        return _coverage_verdict(
            ref,
            "no frozen schema-2 demo-skill evidence with typed evaluation role",
            reason="demo_skill_evidence_missing",
        )
    if evidence.get("status") != "ready":
        reasons = evidence.get("reasons")
        return _coverage_verdict(
            ref,
            "typed held-out evidence is unresolved"
            + (f": {reasons!r}" if reasons else ""),
            reason="demo_skill_evidence_unresolved",
        )
    role = evidence.get("evaluation_protocol_role")
    if not isinstance(role, Mapping):
        return _coverage_verdict(
            ref,
            "schema-2 held-out evidence has no exact evaluation-protocol role",
            reason="evaluation_protocol_role_missing",
        )
    scheme_ids = role.get("scheme_paper_element_ids")
    role_ids = role.get("role_paper_element_ids")
    if (
        not isinstance(scheme_ids, list)
        or not isinstance(role_ids, list)
        or not scheme_ids
        or not role_ids
        or any(not isinstance(item, str) or not item for item in scheme_ids + role_ids)
    ):
        return _coverage_verdict(
            ref,
            "schema-2 evaluation role lacks exact scheme/role paper grounding",
            reason="evaluation_protocol_grounding_missing",
        )
    assessment = assess_held_out_skill(
        evidence.get("family_contract"),
        evidence.get("executed_record"),
        evidence.get("validity_receipt"),
    )
    if assessment.statistics.get("evaluation_protocol_role") != dict(role):
        return _coverage_verdict(
            ref,
            "shared demo-skill evidence did not preserve the exact frozen "
            "evaluation-protocol role",
            reason="evaluation_protocol_role_not_preserved",
        )
    message, evidence_text = _assessment_message(assessment)
    ids = list(dict.fromkeys([*scheme_ids, *role_ids]))
    return ProbeVerdict(
        PROBE_REFS[ref],
        assessment.status,
        message,
        evidence=evidence_text,
        element_ids=ids,
        bound_callables=[],
        reason=str(getattr(assessment, "reason_code", "") or ""),
        probe_ref=ref,
    )


def _probe_element_groundings(
    groundings: Mapping[str, Any] | None,
    ref: str,
) -> list[str]:
    entry = groundings.get(ref) if isinstance(groundings, Mapping) else None
    if not isinstance(entry, Mapping):
        return []
    ids: list[str] = []
    for key in ("element_ids", "paper_element_ids"):
        values = entry.get(key)
        if not isinstance(values, list) or any(
            not isinstance(item, str) or not item for item in values
        ) or len(set(values)) != len(values):
            return []
        for item in values:
            if item not in ids:
                ids.append(item)
    return ids


def _magnitude_verdict(
    evidence: object,
    probe_groundings: Mapping[str, Any] | None,
) -> ProbeVerdict:
    """Assess delivered forecast scale from the executed evaluation record."""
    ref = MAGNITUDE_REF
    if not isinstance(evidence, Mapping):
        return _coverage_verdict(
            ref,
            "no frozen schema-2 executed forecast/history evidence is available",
            reason="demo_magnitude_evidence_missing",
        )
    validated = validate_forecast_magnitude_evidence(
        evidence.get("family_contract"), evidence.get("executed_record")
    )
    evidence_status = validated.get("status")
    reason = str(validated.get("reason") or "magnitude_evidence_unresolved")
    message = str(validated.get("message") or reason)
    if evidence_status == "unsupported":
        return _coverage_verdict(
            ref,
            message,
            reason=reason,
        )
    if evidence_status != "ready":
        return _producer_verdict(
            ref,
            message,
            reason=reason,
            bound_callables=[],
        )
    predictions = validated["model_predictions"]
    history = validated["history_values"]
    row_ids = validated["row_ids"]
    assessment = assess_magnitude_collapse(
        np.asarray(predictions)[:, None],
        np.asarray(history)[:, None],
    )
    statistics = dict(assessment.statistics)
    for old, new in (
        ("eligible_entity_indices", "eligible_row_ids"),
        (
            "per_entity_forecast_to_history_magnitude_ratio",
            "per_row_forecast_to_history_magnitude_ratio",
        ),
        ("minimum_eligible_entity_ratio", "minimum_eligible_row_ratio"),
        ("median_eligible_entity_ratio", "median_eligible_row_ratio"),
        ("collapsed_entity_indices", "collapsed_row_ids"),
        (
            "ratio_above_float_range_entity_indices",
            "ratio_above_float_range_row_ids",
        ),
    ):
        value = statistics.pop(old, None)
        if value is None:
            continue
        if old.endswith("_indices"):
            value = [row_ids[index] for index in value]
        statistics[new] = value
    evidence_text = _strict_json(statistics) if statistics else ""
    return ProbeVerdict(
        PROBE_REFS[ref],
        assessment.status,
        assessment.reason,
        evidence=evidence_text,
        element_ids=_probe_element_groundings(probe_groundings, ref),
        bound_callables=[],
        reason=str(getattr(assessment, "reason_code", "") or ""),
        probe_ref=ref,
    )


def run_time_series_forecasting_probes(
    run_dir: Path,
    *,
    enabled_refs: set[str] | frozenset[str],
    execution_plan: Mapping[str, Any] | None = None,
    demo_skill_evidence: Mapping[str, Any] | None = None,
    probe_groundings: Mapping[str, Any] | None = None,
) -> list[ProbeVerdict]:
    """Run exactly the declared forecasting refs, always emitting one row.

    TSF-4 and TSF-5 consume the exact frozen executed-evaluation record rather
    than pretending a fresh synthetic model is delivery evidence.  TSF-1/2/3
    share one deterministically constructed model and four calls (baseline,
    repeatability, exact 100x own-history, and coherent-permutation control).
    """

    requested = [ref for ref in PROBE_REFS if ref in enabled_refs]
    unknown = set(enabled_refs) - set(PROBE_REFS)
    if unknown:
        raise ValueError(f"unknown forecasting probe refs: {sorted(unknown)!r}")
    out: dict[str, ProbeVerdict] = {}
    method_refs = [ref for ref in requested if ref in METHOD_PROBE_REFS]
    plan = dict(execution_plan) if execution_plan is not None else None
    groundings = dict(probe_groundings) \
        if isinstance(probe_groundings, Mapping) else None
    if groundings is None and isinstance(plan, Mapping):
        raw_groundings = plan.get("probe_groundings")
        if isinstance(raw_groundings, Mapping):
            groundings = dict(raw_groundings)
    plan_error: ForecastPlanCoverageError | None = None
    if plan is None and method_refs:
        try:
            plan = load_live_forecasting_execution_plan(Path(run_dir))
        except ForecastPlanCoverageError as exc:
            plan_error = exc
    if groundings is None and MAGNITUDE_REF in requested:
        try:
            groundings = load_live_forecasting_probe_groundings(Path(run_dir))
        except ForecastPlanCoverageError:
            groundings = None
    if HOLDOUT_REF in requested:
        out[HOLDOUT_REF] = _heldout_verdict(demo_skill_evidence)
    if MAGNITUDE_REF in requested:
        out[MAGNITUDE_REF] = _magnitude_verdict(
            demo_skill_evidence,
            groundings,
        )

    if method_refs:
        if plan_error is not None:
            for ref in method_refs:
                out[ref] = _coverage_verdict(
                    ref,
                    f"forecasting runtime coverage is unsupported: {plan_error}",
                    reason="unsupported_forecasting_runtime_grammar",
                )
        else:
            assert plan is not None
            try:
                call_inputs, baseline, repeated, perturbed = _construct_and_call(
                    Path(run_dir), plan
                )
                assessments = _method_assessments(
                    plan, call_inputs, baseline, repeated, perturbed
                )
                for ref in method_refs:
                    out[ref] = _method_verdict(ref, assessments[ref])
            except ForecastPlanCoverageError as exc:
                for ref in method_refs:
                    out[ref] = _coverage_verdict(
                        ref,
                        f"forecasting runtime coverage is unsupported: {exc}",
                        reason="unsupported_forecasting_runtime_grammar",
                    )
            except ProbeLoadError as exc:
                for ref in method_refs:
                    out[ref] = _coverage_verdict(
                        ref,
                        str(exc),
                        reason="forecast_package_unloadable",
                    )
            except ForecastContractDisagreement as exc:
                for ref in method_refs:
                    out[ref] = _producer_verdict(
                        ref,
                        f"generated forecast disagrees with its normalized "
                        f"schema-2 contract: {exc}",
                        reason="contract_code_disagreement",
                        bound_callables=exc.bound_callables,
                    )

    return [out[ref] for ref in requested]
