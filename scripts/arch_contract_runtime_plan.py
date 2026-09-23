"""Freeze one exact schema-2 forecasting execution plan.

This is the reusable boundary between architecture-contract resolution and
behavioral probes.  It consumes the same semantic fixture resolver as Stage
2.d and the same relational execution resolver as R2C-084, then emits only
JSON-normalizable data.  Frozen-plan consumers can materialize values through
``scripts.typed_fixture`` without importing Pydantic or re-reading a contract.

The time-series build plan owns the otherwise-missing bridge from the opaque
``forecast(model, ...)`` parameter to ``architecture.model`` and its exact
builder.  This module refuses to infer that bridge from a parameter name, a
sole architecture block, a class description, or a discovered ``build_*``.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any

from schemas.arch_contract_v2 import ArchContractV2
from scripts.arch_contract_semantics import (
    BundleFact,
    SemanticIssue,
    load_arch_contract,
    resolve_contract,
)
from scripts.time_series_target_scaling import (
    FittingTargetRange,
    TargetScalingError,
    fit_target_scaling_state,
    validate_target_scaling_contract,
)
from scripts.typed_fixture import (
    FixtureMaterializationError,
    cast_typed_fixture,
    materialize_typed_fixture,
)


class RuntimePlanError(ValueError):
    """The contract/build-plan pair cannot produce an exact frozen plan."""


def _require_mapping(value: Any, *, root: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise RuntimePlanError(f"{root} must be a mapping")
    return value


def _require_exact_text(value: Any, *, root: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise RuntimePlanError(f"{root} must be non-blank and already stripped")
    return value


def _semantic_error(issues: list[SemanticIssue]) -> RuntimePlanError:
    payload = [
        issue.model_dump(mode="json", exclude_none=True) for issue in issues
    ]
    return RuntimePlanError(
        "schema-2 runtime plan resolution failed: "
        + json.dumps(payload, sort_keys=True, separators=(",", ":"))
    )


def _architecture_block_from_root(
    contract: ArchContractV2,
    root: str,
) -> str:
    prefix = "architecture."
    if not root.startswith(prefix) or root == prefix:
        raise RuntimePlanError(
            "runtime_execution construction architecture_root must be an "
            f"exact root below {prefix!r}; got {root!r}"
        )
    block_name = root[len(prefix):]
    if "." in block_name or block_name not in contract.architecture:
        raise RuntimePlanError(
            f"runtime_execution names unknown architecture root {root!r}"
        )
    return block_name


def _resolve_family_binding(
    contract: ArchContractV2,
    build_plan: Mapping[str, Any],
) -> dict[str, Any]:
    runtime = _require_mapping(
        build_plan.get("runtime_execution"),
        root="build_plan.runtime_execution",
    )
    if runtime.get("schema_version") != "1.0":
        raise RuntimePlanError(
            "build_plan.runtime_execution.schema_version must equal '1.0'"
        )

    construction = _require_mapping(
        runtime.get("construction"),
        root="build_plan.runtime_execution.construction",
    )
    architecture_root = _require_exact_text(
        construction.get("architecture_root"),
        root=(
            "build_plan.runtime_execution.construction.architecture_root"
        ),
    )
    architecture_block = _architecture_block_from_root(
        contract, architecture_root
    )
    route = _require_mapping(
        construction.get("route"),
        root="build_plan.runtime_execution.construction.route",
    )
    if route.get("kind") != "builder":
        raise RuntimePlanError(
            "forecasting runtime construction requires the exact family-owned "
            "builder route; class or inferred routes are unsupported"
        )
    builder_module = _require_exact_text(
        route.get("module"),
        root="build_plan.runtime_execution.construction.route.module",
    )
    builder_callable = _require_exact_text(
        route.get("callable"),
        root="build_plan.runtime_execution.construction.route.callable",
    )

    pluggable_call = _require_mapping(
        runtime.get("pluggable_call"),
        root="build_plan.runtime_execution.pluggable_call",
    )
    forecast_module = _require_exact_text(
        pluggable_call.get("module"),
        root="build_plan.runtime_execution.pluggable_call.module",
    )
    forecast_callable = _require_exact_text(
        pluggable_call.get("callable"),
        root="build_plan.runtime_execution.pluggable_call.callable",
    )
    if forecast_callable != contract.pluggable_component.name:
        raise RuntimePlanError(
            "build-plan forecast callable disagrees with "
            "arch_contract.pluggable_component.name: "
            f"{forecast_callable!r} != {contract.pluggable_component.name!r}"
        )
    plan_pluggable = build_plan.get("pluggable_component")
    if isinstance(plan_pluggable, Mapping) and (
        plan_pluggable.get("name") != forecast_callable
    ):
        raise RuntimePlanError(
            "resolved build-plan pluggable name disagrees with its exact "
            "runtime_execution callable"
        )

    positional_parameters = pluggable_call.get("positional_parameters")
    keyword_parameters = pluggable_call.get("keyword_parameters")
    if (
        not isinstance(positional_parameters, list)
        or not positional_parameters
        or any(
            not isinstance(name, str) or not name or name != name.strip()
            for name in positional_parameters
        )
        or len(set(positional_parameters)) != len(positional_parameters)
    ):
        raise RuntimePlanError(
            "runtime_execution positional_parameters must be a non-empty "
            "ordered list of exact names"
        )
    if (
        not isinstance(keyword_parameters, list)
        or any(
            not isinstance(name, str) or not name or name != name.strip()
            for name in keyword_parameters
        )
        or len(set(keyword_parameters)) != len(keyword_parameters)
        or set(positional_parameters) & set(keyword_parameters)
    ):
        raise RuntimePlanError(
            "runtime_execution keyword_parameters must be exact, unique, "
            "and disjoint from positional_parameters"
        )
    if isinstance(plan_pluggable, Mapping):
        plan_contract = plan_pluggable.get("contract")
        if isinstance(plan_contract, Mapping) and (
            list(plan_contract.get("fixed_positional_args") or [])
            != positional_parameters
        ):
            raise RuntimePlanError(
                "runtime_execution positional order disagrees with the "
                "resolved family pluggable contract"
            )
        seed_param = plan_pluggable.get("seed_param")
        if isinstance(seed_param, str) and seed_param not in keyword_parameters:
            raise RuntimePlanError(
                "runtime_execution keyword parameters omit the resolved "
                f"family seed parameter {seed_param!r}"
            )

    raw_bindings = _require_mapping(
        pluggable_call.get("input_bindings"),
        root="build_plan.runtime_execution.pluggable_call.input_bindings",
    )
    if not raw_bindings:
        raise RuntimePlanError(
            "forecasting runtime execution must bind its fixed pluggable inputs"
        )
    input_bindings: dict[str, dict[str, Any]] = {}
    input_prefix = "pluggable_component.input."
    for raw_input_root, raw_binding in raw_bindings.items():
        input_root = _require_exact_text(
            raw_input_root,
            root="runtime_execution input root",
        )
        if not input_root.startswith(input_prefix) or input_root == input_prefix:
            raise RuntimePlanError(
                f"constructed input root must be below {input_prefix!r}; "
                f"got {input_root!r}"
            )
        parameter = input_root[len(input_prefix):]
        if "." in parameter or parameter not in contract.pluggable_component.input:
            raise RuntimePlanError(
                f"constructed input binding names unknown root {input_root!r}"
            )
        binding = _require_mapping(
            raw_binding,
            root=f"runtime_execution input binding {input_root}",
        )
        kind = binding.get("kind")
        if kind == "constructed_model":
            target_root = _require_exact_text(
                binding.get("architecture_root"),
                root=f"runtime_execution constructed target for {input_root}",
            )
            target_block = _architecture_block_from_root(contract, target_root)
            if target_block != architecture_block:
                raise RuntimePlanError(
                    f"constructed input {input_root!r} targets {target_root!r}, "
                    f"but construction owns {architecture_root!r}"
                )
            input_bindings[parameter] = {
                "kind": kind,
                "architecture_block": target_block,
            }
        elif kind in {
            "typed_fixture",
            "relational_graph_or_graph_free_none",
            "target_scaling_entity_ids",
            "target_scaling_state",
        }:
            semantic_role = _require_exact_text(
                binding.get("semantic_role"),
                root=f"runtime_execution semantic role for {input_root}",
            )
            input_bindings[parameter] = {
                "kind": kind,
                "semantic_role": semantic_role,
            }
            if "relational_root" in binding:
                input_bindings[parameter]["relational_root"] = (
                    _require_exact_text(
                        binding.get("relational_root"),
                        root=(
                            "runtime_execution relational root for "
                            f"{input_root}"
                        ),
                    )
                )
        else:
            raise RuntimePlanError(
                f"runtime_execution input {input_root!r} has unsupported "
                f"binding kind {kind!r}"
            )

    if pluggable_call.get("additional_inputs") != (
        "typed_paradigm_extras_by_exact_name"
    ):
        raise RuntimePlanError(
            "forecasting runtime_execution must declare exact-name typed "
            "paradigm extras"
        )
    core_parameters = [*positional_parameters, *keyword_parameters]
    if set(core_parameters) != set(input_bindings):
        raise RuntimePlanError(
            "forecasting call-style parameters and semantic input bindings "
            "must name the same fixed family inputs"
        )

    output_grammar = _require_mapping(
        runtime.get("output_grammar"),
        root="build_plan.runtime_execution.output_grammar",
    )
    if (
        output_grammar.get("contract_root") != "pluggable_component.output"
        or output_grammar.get("kind") != "attribute_record"
        or output_grammar.get("module") != "method.method"
        or output_grammar.get("type_name") != "ForecastResult"
    ):
        raise RuntimePlanError(
            "forecasting output grammar must exactly bind the ForecastResult "
            "attribute record at pluggable_component.output"
        )
    fields = _require_mapping(
        output_grammar.get("fields"),
        root="build_plan.runtime_execution.output_grammar.fields",
    )
    expected_fields = {
        "mean": "mean",
        "variance": "variance",
        "samples": "samples",
        "distribution_params": "distribution_params",
    }
    if dict(fields) != expected_fields:
        raise RuntimePlanError(
            "ForecastResult output grammar must use the four exact committed "
            "field names"
        )
    distribution = _require_mapping(
        output_grammar.get("distribution_params"),
        root=(
            "build_plan.runtime_execution.output_grammar."
            "distribution_params"
        ),
    )
    allowed_key_sets = distribution.get("allowed_key_sets")
    required_roles = {"location", "scale", "degrees_of_freedom"}
    if (
        distribution.get("kind") != "student_t"
        or distribution.get("closed") is not True
        or not isinstance(allowed_key_sets, list)
        or len(allowed_key_sets) != 2
        or any(
            not isinstance(entry, Mapping)
            or set(entry) != required_roles
            or any(
                not isinstance(value, str)
                or not value
                or value != value.strip()
                for value in entry.values()
            )
            for entry in allowed_key_sets
        )
        or len(
            {
                tuple(sorted(dict(entry).items()))
                for entry in allowed_key_sets
            }
        )
        != 2
    ):
        raise RuntimePlanError(
            "Student-t output grammar must declare two distinct closed key sets"
        )

    conditional_probes = _require_mapping(
        runtime.get("conditional_probes"),
        root="build_plan.runtime_execution.conditional_probes",
    )

    return {
        "architecture_root": architecture_root,
        "architecture_block": architecture_block,
        "builder_module": builder_module,
        "builder_callable": builder_callable,
        "forecast_module": forecast_module,
        "forecast_callable": forecast_callable,
        "positional_parameters": list(positional_parameters),
        "keyword_parameters": list(keyword_parameters),
        "additional_inputs": "typed_paradigm_extras_by_exact_name",
        "input_bindings": input_bindings,
        "output_grammar": json.loads(json.dumps(output_grammar)),
        "conditional_probes": json.loads(json.dumps(conditional_probes)),
    }


def _stable_entity_ids(count: int) -> list[int]:
    base = [101, 205, 309, 412, 518]
    return (base + [1000 + index for index in range(max(0, count - 5))])[
        :count
    ]


def _asymmetric_graph(
    *,
    count: int,
    representation: str,
    graph_shape: list[int] | tuple[int, ...],
) -> list[Any]:
    candidates = [
        (0, 1),
        (0, 2),
        (0, 2),
        (1, 2),
        (2, min(4, count - 1)),
        (min(4, count - 1), 0),
        (min(3, count - 1), 1),
    ]
    candidates = [
        edge for edge in candidates if edge[0] < count and edge[1] < count
    ]
    if representation == "sparse_edge_index":
        if len(graph_shape) != 2 or graph_shape[0] != 2:
            raise RuntimePlanError(
                "resolved sparse relational graph must have shape (2, E)"
            )
        edge_count = int(graph_shape[1])
        if edge_count and not candidates:
            raise RuntimePlanError(
                "cannot construct the asymmetric sparse relational fixture"
            )
        edges = [candidates[index % len(candidates)] for index in range(edge_count)]
        return [
            [source for source, _ in edges],
            [destination for _, destination in edges],
        ]

    if list(graph_shape) != [count, count]:
        raise RuntimePlanError(
            "resolved dense relational graph must be square on the source "
            "entity domain"
        )
    graph: list[list[float]] = [
        [0.0 for _ in range(count)] for _ in range(count)
    ]
    for edge_number, (source, destination) in enumerate(candidates, start=1):
        graph[source][destination] = float(edge_number)
    return graph


def _degrees(
    graph: list[Any],
    *,
    count: int,
    representation: str,
    kind: str,
    semantics: str,
) -> list[float]:
    if semantics == "source_graph":
        # Deliberately not derived from edges: source-graph metadata must be
        # mapped, never silently recomputed after batching.
        return [float(11 + 2 * index) for index in range(count)]
    degrees = [0.0 for _ in range(count)]
    if representation == "sparse_edge_index":
        endpoint_row = 1 if kind == "in_degree" else 0
        for endpoint in graph[endpoint_row]:
            degrees[int(endpoint)] += 1.0
        return degrees
    if kind == "in_degree":
        for destination in range(count):
            degrees[destination] = float(
                sum(graph[source][destination] != 0 for source in range(count))
            )
    else:
        for source in range(count):
            degrees[source] = float(
                sum(value != 0 for value in graph[source])
            )
    return degrees


def _phase_identity(
    positions: list[int],
    *,
    source_ids: list[int],
    output_order: str,
) -> dict[str, Any]:
    expected = sorted(positions) if output_order == "canonical_source_order" else list(positions)
    source_to_local = [-1 for _ in source_ids]
    for local, source in enumerate(expected):
        source_to_local[source] = local
    return {
        "requested_source_positions": list(positions),
        "requested_entity_ids": [source_ids[index] for index in positions],
        "local_to_source": expected,
        "source_to_local": source_to_local,
        "output_entity_ids": [source_ids[index] for index in expected],
        "local_endpoint_bounds": [0, len(expected)],
    }


def _normalize_relational(
    contract: ArchContractV2,
    resolved_dimensions: Mapping[str, Any],
    runtime_fixtures: Mapping[str, Mapping[str, Any]],
    *,
    permitted_opaque_fitting_roots: frozenset[str] = frozenset(),
) -> dict[str, Any] | None:
    relational = contract.relational_indexing
    if relational is None:
        return None

    # Import lazily to keep the frozen consumer path independent of the
    # Stage-2.d driver module while using its exact R2C-084 authority here.
    from scripts.validate_arch_contract_runtime import (  # noqa: PLC0415
        _resolve_relational_execution_plan,
    )

    execution, issues = _resolve_relational_execution_plan(
        contract,
        resolved_dimensions=dict(resolved_dimensions),
        typed_fixtures={root: dict(spec) for root, spec in runtime_fixtures.items()},
        permitted_opaque_fitting_roots=permitted_opaque_fitting_roots,
    )
    if issues:
        raise _semantic_error(issues)
    if execution is None:
        raise RuntimePlanError(
            "relational contract did not yield an execution plan"
        )

    count = int(execution["entity_count"])
    fitting_roots = execution["fitting_roots"]
    graph_root = fitting_roots["graph"]
    graph_spec = runtime_fixtures[graph_root]
    graph_shape = list(graph_spec.get("shape") or [])
    graph = _asymmetric_graph(
        count=count,
        representation=relational.representation,
        graph_shape=graph_shape,
    )
    source_ids = _stable_entity_ids(count)
    degree_values = (
        _degrees(
            graph,
            count=count,
            representation=relational.representation,
            kind=str(relational.degree_kind),
            semantics=str(relational.degree_semantics),
        )
        if relational.degree_root is not None
        else None
    )

    return {
        "schema_version": "1.0",
        "methodology_element_ids": list(relational.methodology_element_ids),
        "entity_axis": relational.entity_axis,
        "entity_dimension": relational.execution.entity_dimension,
        "representation": relational.representation,
        "source_endpoint_index_space": relational.source_endpoint_index_space,
        "prepared_endpoint_index_space": relational.prepared_endpoint_index_space,
        "edge_orientation": relational.edge_orientation,
        "stable_entity_id_root": relational.stable_entity_id_root,
        "graph_root": relational.graph_root,
        "coindexed_roots": dict(relational.coindexed_roots),
        "degree_root": relational.degree_root,
        "degree_kind": relational.degree_kind,
        "degree_semantics": relational.degree_semantics,
        "output_order": relational.output_order,
        "phase_batch_modes": relational.phase_batch_modes.model_dump(mode="json"),
        "preparation_callable": relational.preparation_callable.model_dump(
            mode="json"
        ),
        "execution": execution,
        "source_fixture": {
            "stable_entity_ids": source_ids,
            "graph": graph,
            "degrees": degree_values,
            "typed_roots": {
                "stable_entity_ids": fitting_roots["source_entity_ids"],
                "graph": graph_root,
                "degrees": fitting_roots.get("degrees"),
            },
        },
        "fitting_identity": _phase_identity(
            list(execution["fitting_positions"]),
            source_ids=source_ids,
            output_order=relational.output_order,
        ),
        "inference_identity": _phase_identity(
            list(execution["inference_positions"]),
            source_ids=source_ids,
            output_order=relational.output_order,
        ),
    }


def _normalize_target_scaling(
    contract: ArchContractV2,
    build_plan: Mapping[str, Any],
    frozen_fixtures: Mapping[str, Mapping[str, Any]],
    relational: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Bind the closed scaling policy to exact schema-2 fitting roots."""
    try:
        policy = validate_target_scaling_contract(
            _require_mapping(
                build_plan.get("target_scaling"),
                root="build_plan.target_scaling",
            )
        )
    except TargetScalingError as exc:
        raise RuntimePlanError(str(exc)) from exc

    execution = _require_mapping(
        build_plan.get("target_scaling_execution"),
        root="build_plan.target_scaling_execution",
    )
    expected_execution_keys = {
        "schema_version", "helper", "training_call",
    }
    if set(execution) != expected_execution_keys \
            or execution.get("schema_version") != "1.0.0":
        raise RuntimePlanError(
            "target_scaling_execution must use the closed schema_version "
            "'1.0.0' shape"
        )
    helper = _require_mapping(
        execution.get("helper"), root="target_scaling_execution.helper"
    )
    expected_helper = {
        "module": "method.target_scaling",
        "fit_callable": "fit_target_scaling_state",
        "transform_callable": "transform_targets",
        "inverse_callable": "inverse_forecast_output",
    }
    if dict(helper) != expected_helper:
        raise RuntimePlanError(
            "target_scaling_execution.helper must equal the fixed "
            "scaffolder-owned helper surface"
        )

    training = _require_mapping(
        execution.get("training_call"),
        root="target_scaling_execution.training_call",
    )
    expected_training_keys = {
        "module",
        "callable",
        "model_input_root",
        "entity_ids_input_root",
        "targets_input_root",
        "fitting_range_input_root",
        "state_result",
    }
    if set(training) != expected_training_keys \
            or training.get("module") != "method.training":
        raise RuntimePlanError(
            "target_scaling_execution.training_call has unsupported shape"
        )
    if contract.training_loop is None:
        raise RuntimePlanError(
            "forecasting target scaling requires a declared schema-2 "
            "training_loop"
        )
    training_callable = _require_exact_text(
        training.get("callable"), root="target_scaling training callable"
    )
    if training_callable != contract.training_loop.function_name:
        raise RuntimePlanError(
            "target scaling training callable disagrees with "
            "arch_contract.training_loop.function_name"
        )
    if contract.training_loop.output is None \
            or contract.training_loop.output.kind != "opaque":
        raise RuntimePlanError(
            "target scaling training_loop.output must be an explicit opaque "
            "mapping carrier; schema 2 does not claim its interior"
        )

    root_fields = {
        "model": "model_input_root",
        "entity_ids": "entity_ids_input_root",
        "targets": "targets_input_root",
        "fitting_range": "fitting_range_input_root",
    }
    roots: dict[str, str] = {}
    parameters: dict[str, str] = {}
    for role, field in root_fields.items():
        root = _require_exact_text(
            training.get(field), root=f"target_scaling training {field}"
        )
        prefix = "training_loop.input."
        if not root.startswith(prefix) or root == prefix:
            raise RuntimePlanError(
                f"target scaling {field} must be a full root below {prefix!r}"
            )
        parameter = root[len(prefix):]
        if "." in parameter or parameter not in contract.training_loop.input:
            raise RuntimePlanError(
                f"target scaling {field} names unknown typed root {root!r}"
            )
        roots[role] = root
        parameters[role] = parameter

    model_fixture = frozen_fixtures[roots["model"]]
    ids_fixture = frozen_fixtures[roots["entity_ids"]]
    targets_fixture = frozen_fixtures[roots["targets"]]
    range_fixture = frozen_fixtures[roots["fitting_range"]]
    if model_fixture.get("kind") != "opaque":
        raise RuntimePlanError(
            "target scaling model input must be the exact constructed-model "
            "opaque boundary"
        )
    if ids_fixture.get("kind") not in {"tensor", "ndarray"} \
            or ids_fixture.get("dtype") not in {"int32", "int64"} \
            or len(ids_fixture.get("shape") or []) != 1:
        raise RuntimePlanError(
            "target scaling entity_ids input must be one typed integer vector"
        )
    if targets_fixture.get("kind") not in {"tensor", "ndarray"} \
            or targets_fixture.get("dtype") not in {"float32", "float64"} \
            or len(targets_fixture.get("shape") or []) != 2:
        raise RuntimePlanError(
            "target scaling targets input must be one typed floating "
            "[entity, protocol] array"
        )
    if range_fixture.get("kind") != "opaque":
        raise RuntimePlanError(
            "target scaling fitting_range must remain an opaque pipeline "
            "lineage carrier rather than invented tensor or partition truth"
        )
    entity_count = int(ids_fixture["shape"][0])
    target_shape = [int(axis) for axis in targets_fixture["shape"]]
    if target_shape[0] != entity_count:
        raise RuntimePlanError(
            "target scaling entity_ids and targets do not share one exact "
            "entity-axis extent"
        )
    ids_descriptor = contract.training_loop.input[parameters["entity_ids"]]
    targets_descriptor = contract.training_loop.input[parameters["targets"]]
    id_dimensions = [
        use.dimension
        for use in (getattr(ids_descriptor, "dimensions", None) or [])
    ]
    target_dimensions = [
        use.dimension
        for use in (getattr(targets_descriptor, "dimensions", None) or [])
    ]
    entity_axis = int(policy["target_entity_axis"])
    protocol_axis = int(policy["target_protocol_axis"])
    if entity_axis != 0 or protocol_axis != 1 \
            or id_dimensions != [target_dimensions[entity_axis]]:
        raise RuntimePlanError(
            "target scaling entity_ids and targets do not share one exact "
            "schema-2 entity dimension on the declared axes"
        )

    if relational is not None:
        if relational.get("stable_entity_id_root") != policy["entity_id_root"]:
            raise RuntimePlanError(
                "target scaling entity-id logical root disagrees with "
                "relational_indexing.stable_entity_id_root"
            )
        coindexed = relational.get("coindexed_roots") or {}
        if coindexed.get(policy["target_root"]) != entity_axis:
            raise RuntimePlanError(
                "target scaling target logical root must be a declared "
                "relational co-indexed root on the entity axis"
            )
        rel_execution = relational.get("execution") or {}
        fitting_roots = rel_execution.get("fitting_roots") or {}
        if fitting_roots.get("source_entity_ids") != roots["entity_ids"]:
            raise RuntimePlanError(
                "target scaling entity_ids root disagrees with the R2C-084 "
                "source stable-id fitting root"
            )
        if (fitting_roots.get("coindexed") or {}).get(
            policy["target_root"]
        ) != roots["targets"]:
            raise RuntimePlanError(
                "target scaling targets root disagrees with the R2C-084 "
                "co-indexed fitting root"
            )
        if target_dimensions[entity_axis] != relational.get("entity_dimension"):
            raise RuntimePlanError(
                "target scaling targets do not preserve the relational source "
                "entity dimension"
            )

    state_result = _require_mapping(
        training.get("state_result"),
        root="target_scaling_execution.training_call.state_result",
    )
    if dict(state_result) != {
        "kind": "mapping_key", "key": "target_scaling_state",
    }:
        raise RuntimePlanError(
            "target scaling training result must expose the exact "
            "target_scaling_state mapping key"
        )
    if relational is None:
        validation_entity_ids = _stable_entity_ids(entity_count)
    else:
        raw_validation_entity_ids = (
            relational.get("source_fixture") or {}
        ).get("stable_entity_ids")
        if not isinstance(raw_validation_entity_ids, list) \
                or len(raw_validation_entity_ids) != entity_count:
            raise RuntimePlanError(
                "target scaling relational validation fixture does not retain "
                "the exact R2C-084 source stable-entity domain"
            )
        validation_entity_ids = list(raw_validation_entity_ids)

    fitting_range_payload = {
        "validator": "eval_split_lineage",
        "target_root": str(policy["target_root"]),
        "model_id": "r2c-schema2-runtime-fixture",
        "start": 0,
        "stop": target_shape[protocol_axis],
        "certainty": "exact",
    }
    try:
        validation_ids = cast_typed_fixture(
            validation_entity_ids, ids_fixture
        )
        validation_targets = materialize_typed_fixture(
            targets_fixture, salt=roots["targets"]
        )
        validation_state = fit_target_scaling_state(
            policy,
            entity_ids=validation_ids,
            targets=validation_targets,
            fitting_range=FittingTargetRange(**fitting_range_payload),
        )
    except (FixtureMaterializationError, TargetScalingError, TypeError) as exc:
        raise RuntimePlanError(
            "target scaling validator-only fixture cannot be frozen without "
            f"guessing: {exc}"
        ) from exc
    if hasattr(validation_targets, "detach"):
        validation_targets_payload = validation_targets.detach().cpu().tolist()
    else:
        validation_targets_payload = validation_targets.tolist()

    return {
        "schema_version": "1.0.0",
        "policy": policy,
        "helper": dict(helper),
        "training_call": {
            "module": "method.training",
            "callable": training_callable,
            "roots": roots,
            "parameters": parameters,
            "state_result": dict(state_result),
        },
        "entity_dimension": target_dimensions[entity_axis],
        "protocol_dimension": target_dimensions[protocol_axis],
        # This is a bounded validator input, never a paper partition claim.
        # Live acceptance rebinds producer state to the independently minted
        # R2C-077 split-lineage receipt instead of trusting this fixture.
        "validation_fixture": {
            "scope": "schema2_runtime_validator_only",
            "entity_ids": validation_entity_ids,
            "targets": validation_targets_payload,
            "fitting_range": fitting_range_payload,
            "state": validation_state,
        },
    }


def _normalize_training_history(
    contract: ArchContractV2,
    build_plan: Mapping[str, Any],
    frozen_fixtures: Mapping[str, Mapping[str, Any]],
    target_scaling: Mapping[str, Any],
) -> dict[str, Any]:
    """Bind the fixed history recorder to exact schema-2 training roots."""

    from scripts.time_series_training_history import (  # noqa: PLC0415
        TrainingHistoryError,
        record_training_history,
    )

    execution = _require_mapping(
        build_plan.get("training_history_execution"),
        root="build_plan.training_history_execution",
    )
    if set(execution) != {"schema_version", "helper", "training_call"} \
            or execution.get("schema_version") != "1.0.0":
        raise RuntimePlanError(
            "training_history_execution must use the closed schema_version "
            "'1.0.0' shape"
        )
    helper = _require_mapping(
        execution.get("helper"), root="training_history_execution.helper"
    )
    expected_helper = {
        "module": "method.training_history",
        "record_callable": "record_training_history",
    }
    if dict(helper) != expected_helper:
        raise RuntimePlanError(
            "training_history_execution.helper must equal the fixed "
            "scaffolder-owned recorder"
        )
    training = _require_mapping(
        execution.get("training_call"),
        root="training_history_execution.training_call",
    )
    expected_training_keys = {
        "module",
        "callable",
        "fitting_range_input_root",
        "selection_range_input_root",
        "seed_input_root",
        "config_id_input_root",
        "history_result",
    }
    if set(training) != expected_training_keys \
            or training.get("module") != "method.training":
        raise RuntimePlanError(
            "training_history_execution.training_call has unsupported shape"
        )
    if contract.training_loop is None:
        raise RuntimePlanError(
            "forecasting training history requires a declared schema-2 "
            "training_loop"
        )
    callable_name = _require_exact_text(
        training.get("callable"), root="training-history training callable"
    )
    if callable_name != contract.training_loop.function_name:
        raise RuntimePlanError(
            "training-history callable disagrees with "
            "arch_contract.training_loop.function_name"
        )
    root_fields = {
        "fitting_range": "fitting_range_input_root",
        "selection_range": "selection_range_input_root",
        "seed": "seed_input_root",
        "config_id": "config_id_input_root",
    }
    roots: dict[str, str] = {}
    parameters: dict[str, str] = {}
    for role, field in root_fields.items():
        root = _require_exact_text(
            training.get(field), root=f"training-history {field}"
        )
        prefix = "training_loop.input."
        parameter = root[len(prefix):] if root.startswith(prefix) else ""
        if not parameter or "." in parameter \
                or parameter not in contract.training_loop.input:
            raise RuntimePlanError(
                f"training-history {field} names unknown typed root {root!r}"
            )
        roots[role] = root
        parameters[role] = parameter
    if roots["fitting_range"] != (
        (target_scaling.get("training_call") or {}).get("roots") or {}
    ).get("fitting_range"):
        raise RuntimePlanError(
            "training-history fitting range must reuse the exact target-scaling "
            "fitting carrier"
        )
    fitting_fixture = frozen_fixtures[roots["fitting_range"]]
    selection_fixture = frozen_fixtures[roots["selection_range"]]
    seed_fixture = frozen_fixtures[roots["seed"]]
    config_fixture = frozen_fixtures[roots["config_id"]]
    if fitting_fixture.get("kind") != "opaque" \
            or selection_fixture.get("kind") != "opaque":
        raise RuntimePlanError(
            "training-history fitting and selection ranges must remain opaque "
            "pipeline lineage carriers"
        )
    if config_fixture.get("kind") != "opaque":
        raise RuntimePlanError(
            "training-history config_id must remain an opaque string boundary"
        )
    if seed_fixture.get("kind") != "scalar" \
            or seed_fixture.get("dtype") not in {"int32", "int64"}:
        raise RuntimePlanError(
            "training-history seed must be one typed integer scalar"
        )
    history_result = _require_mapping(
        training.get("history_result"),
        root="training_history_execution.training_call.history_result",
    )
    if dict(history_result) != {
        "kind": "mapping_key", "key": "training_history",
    }:
        raise RuntimePlanError(
            "training-history result must expose the exact training_history key"
        )

    scaling_validation = target_scaling.get("validation_fixture") or {}
    fitting_range = scaling_validation.get("fitting_range")
    scaling_state = scaling_validation.get("state")
    if not isinstance(fitting_range, Mapping) \
            or not isinstance(scaling_state, Mapping):
        raise RuntimePlanError(
            "training-history fixture cannot reuse the frozen scaling lineage"
        )
    seed_value = materialize_typed_fixture(
        seed_fixture, salt=roots["seed"]
    )
    if hasattr(seed_value, "item"):
        seed_value = seed_value.item()
    seed_value = int(seed_value)
    model_id = str(fitting_range.get("model_id") or "")
    selected_checkpoint = "checkpoint-001"
    # Pure fixed-helper oracle only: schema 2 has no held-out selection value
    # carrier. Stage 2.d never passes this synthetic performed arm to the real
    # fitting entry; it exercises selection.status=not_performed instead.
    selection_range = {
        "validator": "eval_split_lineage",
        "target_root": str(fitting_range.get("target_root") or ""),
        "model_id": model_id,
        "start": int(fitting_range.get("stop") or 0),
        "stop": int(fitting_range.get("stop") or 0) + 2,
        "certainty": "exact",
    }
    inputs: dict[str, Any] = {
        "model_id": model_id,
        "checkpoint_id": selected_checkpoint,
        "seed": seed_value,
        "config_id": "r2c-schema2-runtime-config-v1",
        "target_scaling_state": dict(scaling_state),
        "fitting_range": dict(fitting_range),
        "loss_index_kind": "epoch",
        "loss_observations": [
            {
                "index": 0,
                "value": 3.0,
                "sample_weight": int(scaling_state["entity_count"]),
                "checkpoint_id": "checkpoint-000",
            },
            {
                "index": 1,
                "value": 1.0,
                "sample_weight": int(scaling_state["entity_count"]),
                "checkpoint_id": "checkpoint-001",
            },
            {
                "index": 2,
                "value": 1.0,
                "sample_weight": int(scaling_state["entity_count"]),
                "checkpoint_id": "checkpoint-002",
            },
        ],
        "selection_range": selection_range,
        "selection": {
            "status": "performed",
            "metric_id": "validation_loss",
            "direction": "minimize",
            "observations": [
                {"index": 0, "checkpoint_id": "checkpoint-000", "value": 2.0},
                {"index": 1, "checkpoint_id": "checkpoint-001", "value": 1.0},
                {"index": 2, "checkpoint_id": "checkpoint-002", "value": 1.0},
            ],
            "selected_checkpoint_id": selected_checkpoint,
            "tie_break": "best_then_earliest",
        },
    }
    try:
        record = record_training_history(**inputs)
    except (TrainingHistoryError, TypeError, ValueError) as exc:
        raise RuntimePlanError(
            "training-history validator-only fixture cannot be frozen without "
            f"guessing: {exc}"
        ) from exc
    return {
        "schema_version": "1.0.0",
        "helper": dict(helper),
        "training_call": {
            "module": "method.training",
            "callable": callable_name,
            "roots": roots,
            "parameters": parameters,
            "history_result": dict(history_result),
        },
        "validation_fixture": {
            "scope": "schema2_fixed_helper_oracle_only",
            "inputs": inputs,
            "record": record,
        },
    }


def _normalize_applicability(
    family: Mapping[str, Any],
    method_spec: Mapping[str, Any] | None,
) -> dict[str, Any]:
    conditional = _require_mapping(
        family.get("conditional_probes"),
        root="runtime_execution.conditional_probes",
    )
    raw = _require_mapping(
        conditional.get("autoregressive_path_dependence"),
        root=(
            "runtime_execution.conditional_probes."
            "autoregressive_path_dependence"
        ),
    )
    probe_ref = _require_exact_text(
        raw.get("probe_ref"),
        root="autoregressive_path_dependence.probe_ref",
    )
    if raw.get("authority") != "methodology_verification_probe_refs":
        raise RuntimePlanError(
            "autoregressive applicability must use exact methodology probe refs"
        )

    result: dict[str, Any] = {
        "probe_ref": probe_ref,
        "authority": "methodology_verification_probe_refs",
        "status": "unresolved",
        "element_ids": [],
    }
    if not isinstance(method_spec, Mapping):
        return result
    from scripts.probe_spec_join import (  # noqa: PLC0415
        reference_contract_enabled,
    )

    raw_spec = dict(method_spec)
    if not reference_contract_enabled(raw_spec):
        return result
    contract = raw_spec.get("methodology_replication_contract")
    elements = contract.get("elements") if isinstance(contract, Mapping) else None
    if not isinstance(elements, list):
        return result
    matched: list[str] = []
    for element in elements:
        if not isinstance(element, Mapping):
            continue
        refs = element.get("verification_probe_refs")
        if not isinstance(refs, list) or probe_ref not in refs:
            continue
        element_id = element.get("element_id")
        if isinstance(element_id, str) and element_id and element_id not in matched:
            matched.append(element_id)
    result["element_ids"] = matched
    result["status"] = "required" if matched else "not_applicable"
    return result


def normalize_forecasting_probe_groundings(
    method_spec: Mapping[str, Any] | None,
) -> dict[str, dict[str, list[str]]]:
    """Resolve exact-ref TSF obligation identities without a runtime plan.

    Output-evidence probes do not construct the model merely to discover
    methodology grounding.  This lightweight carrier depends only on the
    reference-aware method spec and is therefore safe to freeze even when
    architecture or relational runtime grammar is unsupported.
    """
    refs = (
        "time_series_forecasting.sample_genuineness",
        "time_series_forecasting.path_dependence",
        "time_series_forecasting.own_history_sensitivity",
        "time_series_forecasting.holdout_skill",
        "time_series_forecasting.magnitude_collapse",
    )
    out = {
        ref: {"element_ids": [], "paper_element_ids": []}
        for ref in refs
    }
    if not isinstance(method_spec, Mapping):
        return out
    from scripts.probe_spec_join import (  # noqa: PLC0415
        reference_contract_enabled,
    )

    raw_spec = dict(method_spec)
    if not reference_contract_enabled(raw_spec):
        return out
    contract = raw_spec.get("methodology_replication_contract")
    elements = contract.get("elements") if isinstance(contract, Mapping) else None
    if not isinstance(elements, list):
        return out
    for element in elements:
        if not isinstance(element, Mapping):
            continue
        declared = element.get("verification_probe_refs")
        if not isinstance(declared, list):
            continue
        element_id = element.get("element_id")
        paper_ids = element.get("paper_element_ids")
        for ref in refs:
            if ref not in declared:
                continue
            if isinstance(element_id, str) and element_id \
                    and element_id not in out[ref]["element_ids"]:
                out[ref]["element_ids"].append(element_id)
            if isinstance(paper_ids, list):
                for paper_id in paper_ids:
                    if isinstance(paper_id, str) and paper_id \
                            and paper_id not in out[ref]["paper_element_ids"]:
                        out[ref]["paper_element_ids"].append(paper_id)
    return out


def normalize_schema2_forecasting_plan(
    contract: ArchContractV2 | Mapping[str, Any],
    build_plan: Mapping[str, Any],
    *,
    bundle_facts: Mapping[str, BundleFact] | None = None,
    method_spec: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return one exact, JSON-normalizable forecasting execution plan.

    Schema 1 is rejected rather than adapted. Ordinary forecast inputs must
    have supported typed descriptors; the sole allowed opaque input is one
    explicitly bound by the committed family plan to an architecture block.
    Unsupported relational grammar is surfaced by the existing R2C-084
    resolver before a frozen plan is emitted.
    """

    if not isinstance(contract, ArchContractV2):
        raw_version = (
            contract.get("schema_version", "1.0.0")
            if isinstance(contract, Mapping)
            else None
        )
        if raw_version != "2.0.0":
            raise RuntimePlanError(
                "forecast execution plans require schema 2.0.0; legacy "
                "shape or constructor adapters are not permitted"
            )
        try:
            parsed = load_arch_contract(dict(contract))
        except (TypeError, ValueError) as exc:
            raise RuntimePlanError(
                f"architecture contract cannot be parsed: {exc}"
            ) from exc
        if not isinstance(parsed, ArchContractV2):
            raise RuntimePlanError(
                "forecast execution plans require schema 2.0.0; legacy "
                "shape or constructor adapters are not permitted"
            )
        contract = parsed

    family = _resolve_family_binding(contract, build_plan)
    resolution = resolve_contract(contract, bundle_facts)
    if resolution.issues:
        raise _semantic_error(resolution.issues)

    runtime_fixtures = {
        root: fixture.model_dump(mode="json", exclude_none=True)
        for root, fixture in resolution.fixtures.items()
    }
    frozen_fixtures = {
        root: fixture.model_dump(mode="json", exclude_none=False)
        for root, fixture in resolution.fixtures.items()
    }

    raw_scaling_execution = build_plan.get("target_scaling_execution")
    raw_scaling_training = (
        raw_scaling_execution.get("training_call")
        if isinstance(raw_scaling_execution, Mapping) else None
    )
    raw_fitting_range_root = (
        raw_scaling_training.get("fitting_range_input_root")
        if isinstance(raw_scaling_training, Mapping) else None
    )
    raw_history_execution = build_plan.get("training_history_execution")
    raw_history_training = (
        raw_history_execution.get("training_call")
        if isinstance(raw_history_execution, Mapping) else None
    )
    permitted_opaque_fitting_roots = frozenset(
        root
        for root in (
            raw_fitting_range_root,
            raw_history_training.get("selection_range_input_root")
            if isinstance(raw_history_training, Mapping) else None,
            raw_history_training.get("config_id_input_root")
            if isinstance(raw_history_training, Mapping) else None,
        )
        if isinstance(root, str)
    )
    relational = _normalize_relational(
        contract,
        resolution.dimensions,
        runtime_fixtures,
        permitted_opaque_fitting_roots=permitted_opaque_fitting_roots,
    )
    if relational is not None and (
        relational["execution"]["architecture_block"]
        != family["architecture_block"]
    ):
        raise RuntimePlanError(
            "relational inference architecture block disagrees with the "
            "family-owned forecast construction route"
        )
    target_scaling = _normalize_target_scaling(
        contract,
        build_plan,
        frozen_fixtures,
        relational,
    )
    training_history = _normalize_training_history(
        contract,
        build_plan,
        frozen_fixtures,
        target_scaling,
    )

    required_family_inputs = set(family["input_bindings"])
    contract_inputs = set(contract.pluggable_component.input)
    missing_family_inputs = sorted(required_family_inputs - contract_inputs)
    if missing_family_inputs:
        raise RuntimePlanError(
            "forecast contract omits committed family input binding(s): "
            f"{missing_family_inputs!r}"
        )
    input_bindings: dict[str, dict[str, Any]] = {}
    for parameter in contract.pluggable_component.input:
        root = f"pluggable_component.input.{parameter}"
        fixture = resolution.fixtures[root]
        declared_binding = family["input_bindings"].get(parameter)
        if declared_binding is None:
            if fixture.kind == "opaque" or not fixture.synthesizable:
                raise RuntimePlanError(
                    f"forecast extra input {root!r} is not a supported exact "
                    "typed fixture"
                )
            input_bindings[parameter] = {
                "kind": "fixture",
                "root": root,
                "semantic_role": parameter,
            }
            continue

        binding_kind = declared_binding["kind"]
        semantic_role = declared_binding.get("semantic_role")
        if binding_kind == "constructed_model":
            if fixture.kind != "opaque":
                raise RuntimePlanError(
                    f"constructed input {root!r} must be declared opaque, not "
                    f"{fixture.kind!r}"
                )
            input_bindings[parameter] = {
                "kind": "constructed_model",
                "architecture_block": declared_binding["architecture_block"],
            }
            continue
        if binding_kind == "relational_graph_or_graph_free_none":
            if relational is None:
                if fixture.kind != "opaque":
                    raise RuntimePlanError(
                        "graph-free forecasting requires the graph input to be "
                        "explicitly opaque; runtime_execution supplies literal "
                        "None only under relational_indexing=None"
                    )
                input_bindings[parameter] = {
                    "kind": "literal",
                    "value": None,
                    "semantic_role": semantic_role,
                }
                continue
            if fixture.kind == "opaque" or not fixture.synthesizable:
                raise RuntimePlanError(
                    "a relational forecast graph must have an exact typed "
                    "fixture rather than opaque grammar"
                )
            source_graph_root = relational["source_fixture"]["typed_roots"][
                "graph"
            ]
            source_graph_fixture = frozen_fixtures[source_graph_root]
            comparable_fields = (
                "kind",
                "dtype",
                "device",
                "shape",
                "constraint",
                "constraint_upper_bound",
            )
            disagreements = {
                field: (frozen_fixtures[root].get(field), source_graph_fixture.get(field))
                for field in comparable_fields
                if frozen_fixtures[root].get(field) != source_graph_fixture.get(field)
            }
            if disagreements:
                raise RuntimePlanError(
                    "forecast graph typed convention disagrees with the "
                    f"R2C-084 source graph: {disagreements!r}"
                )
            if contract.training_loop is None or not source_graph_root.startswith(
                "training_loop.input."
            ):
                raise RuntimePlanError(
                    "relational source graph has no exact training descriptor"
                )
            source_graph_parameter = source_graph_root[
                len("training_loop.input."):
            ]
            source_graph_descriptor = contract.training_loop.input[
                source_graph_parameter
            ].model_dump(mode="json", exclude_none=True)
            forecast_graph_descriptor = contract.pluggable_component.input[
                parameter
            ].model_dump(mode="json", exclude_none=True)
            if forecast_graph_descriptor != source_graph_descriptor:
                raise RuntimePlanError(
                    "forecast graph descriptor does not preserve the exact "
                    "R2C-084 source-graph semantic dimensions and constraints"
                )
            input_bindings[parameter] = {
                "kind": "relational_graph",
                "root": root,
                "semantic_role": semantic_role,
            }
            continue
        if binding_kind == "target_scaling_entity_ids":
            if semantic_role != "stable_entity_ids" \
                    or declared_binding.get("relational_root") != (
                        target_scaling["policy"]["entity_id_root"]
                    ):
                raise RuntimePlanError(
                    "forecast target-scaling entity ids must bind the exact "
                    "stable identity root from the closed scaling policy"
                )
            if fixture.kind not in {"tensor", "ndarray"} \
                    or fixture.dtype not in {"int32", "int64"} \
                    or len(fixture.shape or ()) != 1:
                raise RuntimePlanError(
                    f"forecast input {root!r} must be one typed integer "
                    "stable-id vector"
                )
            training_parameter = target_scaling["training_call"][
                "parameters"
            ]["entity_ids"]
            training_descriptor = contract.training_loop.input[
                training_parameter
            ].model_dump(mode="json", exclude_none=True)
            forecast_descriptor = contract.pluggable_component.input[
                parameter
            ].model_dump(mode="json", exclude_none=True)
            if forecast_descriptor != training_descriptor:
                raise RuntimePlanError(
                    "forecast stable entity ids do not preserve the exact "
                    "schema-2 fitting identity descriptor"
                )
            input_bindings[parameter] = {
                "kind": "target_scaling_entity_ids",
                "root": root,
                "source_root": target_scaling["training_call"]["roots"][
                    "entity_ids"
                ],
                "semantic_role": semantic_role,
                "relational_root": target_scaling["policy"]["entity_id_root"],
                "relational_axis": 0,
            }
            continue
        if binding_kind == "target_scaling_state":
            if semantic_role != "target_scaling_state" \
                    or fixture.kind != "opaque":
                raise RuntimePlanError(
                    "forecast target_scaling_state must be the exact opaque "
                    "state mapping supplied by the fixed scaling helper"
                )
            input_bindings[parameter] = {
                "kind": "target_scaling_state",
                "semantic_role": semantic_role,
            }
            continue
        if binding_kind != "typed_fixture":
            raise RuntimePlanError(
                f"unsupported normalized family input binding {binding_kind!r}"
            )
        if fixture.kind == "opaque" or not fixture.synthesizable:
            raise RuntimePlanError(
                f"forecast input {root!r} is opaque or unsupported; the "
                "runtime plan will not invent its structure"
            )
        relational_root = declared_binding.get("relational_root")
        relational_axis = None
        if relational is not None and relational_root is not None:
            relational_axis = relational["coindexed_roots"].get(relational_root)
            if relational_axis is None:
                raise RuntimePlanError(
                    f"forecast input {root!r} binds undeclared relational root "
                    f"{relational_root!r}"
                )
            descriptor = contract.pluggable_component.input[parameter]
            dimensions = [
                use.dimension
                for use in (getattr(descriptor, "dimensions", None) or [])
            ]
            if (
                len(dimensions) <= relational_axis
                or dimensions[relational_axis]
                != relational["entity_dimension"]
            ):
                raise RuntimePlanError(
                    f"forecast input {root!r} does not preserve exact entity "
                    f"dimension {relational['entity_dimension']!r} on "
                    f"relational axis {relational_axis}; got {dimensions!r}"
                )
        input_bindings[parameter] = {
            "kind": "fixture",
            "root": root,
            "semantic_role": semantic_role,
        }
        if relational is not None and relational_root is not None:
            input_bindings[parameter]["relational_root"] = relational_root
            input_bindings[parameter]["relational_axis"] = relational_axis

    extra_parameters = [
        parameter
        for parameter in contract.pluggable_component.input
        if parameter not in family["input_bindings"]
    ]

    output_root = "pluggable_component.output"
    output_fixture = resolution.fixtures[output_root]
    if output_fixture.kind != "opaque":
        raise RuntimePlanError(
            "ForecastResult must use the explicitly matched opaque schema-2 "
            "boundary plus the committed closed output grammar; a flat typed "
            "descriptor cannot recover its fields"
        )

    return {
        "schema_version": "1.0",
        "source_contract_schema": contract.schema_version,
        "paradigm_id": contract.paradigm_id,
        "construction": {
            "architecture_block": family["architecture_block"],
            "architecture_root": family["architecture_root"],
            "declared_class": {
                "module": "method.model",
                "name": contract.architecture[
                    family["architecture_block"]
                ].class_name,
            },
            "route": {
                "kind": "builder",
                "module": family["builder_module"],
                "callable": family["builder_callable"],
            },
            "constructor_kwargs": resolution.constructor_args[
                family["architecture_block"]
            ],
        },
        "forecast_call": {
            "module": family["forecast_module"],
            "callable": family["forecast_callable"],
            "positional_parameters": family["positional_parameters"],
            "keyword_parameters": [
                *family["keyword_parameters"],
                *extra_parameters,
            ],
            "additional_inputs": family["additional_inputs"],
            "input_bindings": input_bindings,
            "output_root": output_root,
        },
        "output_grammar": family["output_grammar"],
        "fixtures": frozen_fixtures,
        "relational": relational,
        "target_scaling": target_scaling,
        "training_history": training_history,
        "applicability": {
            "autoregressive_path_dependence": _normalize_applicability(
                family, method_spec
            )
        },
        "probe_groundings": normalize_forecasting_probe_groundings(
            method_spec
        ),
    }


def execution_plan_digest(plan: Mapping[str, Any]) -> str:
    """Stable digest for a frozen JSON execution plan."""

    payload = json.dumps(
        dict(plan),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()
