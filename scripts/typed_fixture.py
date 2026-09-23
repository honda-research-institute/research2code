"""Dependency-light materialization for frozen schema-2 fixture specs.

The architecture semantic resolver emits JSON-shaped ``ResolvedFixtureSpec``
records.  Runtime consumers that replay a frozen execution plan should not
need to import Pydantic or re-resolve the architecture contract, so this
module accepts only those normalized mappings and imports NumPy/Torch lazily.

Unlike the Stage-2.d breadth smoke, which deliberately uses zeros, behavioral
probes need signal-bearing inputs.  Array fixtures therefore contain a stable
non-zero pattern while retaining the declared container, dtype, device,
shape, and any index/class-id bounds.
"""

from __future__ import annotations

import copy
import hashlib
from collections.abc import Mapping
from typing import Any


class FixtureMaterializationError(ValueError):
    """A normalized fixture cannot be materialized without guessing."""


_DTYPES = frozenset({"float32", "float64", "int32", "int64", "bool"})
_ARRAY_KINDS = frozenset({"ndarray", "tensor"})


def _normalized_spec(spec: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(spec, Mapping):
        raise FixtureMaterializationError("fixture spec must be a mapping")
    normalized = dict(spec)
    kind = normalized.get("kind")
    if kind == "opaque":
        raise FixtureMaterializationError(
            "opaque fixture has no deterministic materialization"
        )
    if kind not in {*_ARRAY_KINDS, "scalar"}:
        raise FixtureMaterializationError(
            f"unsupported normalized fixture kind {kind!r}"
        )
    if normalized.get("synthesizable") is not True:
        raise FixtureMaterializationError(
            "normalized fixture is not marked synthesizable"
        )
    dtype = normalized.get("dtype")
    if dtype not in _DTYPES:
        raise FixtureMaterializationError(
            f"unsupported normalized fixture dtype {dtype!r}"
        )
    if kind in _ARRAY_KINDS:
        shape = normalized.get("shape")
        if not isinstance(shape, (list, tuple)) or any(
            isinstance(axis, bool) or not isinstance(axis, int) or axis <= 0
            for axis in shape
        ):
            raise FixtureMaterializationError(
                f"array fixture requires a positive integer shape; got {shape!r}"
            )
        normalized["shape"] = tuple(shape)
    elif tuple(normalized.get("shape") or ()) != ():
        raise FixtureMaterializationError(
            "scalar fixture must have an empty resolved shape"
        )
    if kind == "tensor" and normalized.get("device", "cpu") != "cpu":
        raise FixtureMaterializationError(
            "frozen fixture materialization supports only declared cpu tensors"
        )
    return normalized


def _stable_offset(salt: str) -> int:
    digest = hashlib.sha256(salt.encode("utf-8")).digest()
    return 1 + digest[0] % 29


def _numpy_dtype(np: Any, dtype: str) -> Any:
    return {
        "float32": np.float32,
        "float64": np.float64,
        "int32": np.int32,
        "int64": np.int64,
        "bool": np.bool_,
    }[dtype]


def _torch_dtype(torch: Any, dtype: str) -> Any:
    return {
        "float32": torch.float32,
        "float64": torch.float64,
        "int32": torch.int32,
        "int64": torch.int64,
        "bool": torch.bool,
    }[dtype]


def _bounded_values(np: Any, spec: Mapping[str, Any], *, salt: str) -> Any:
    shape = tuple(spec["shape"])
    count = int(np.prod(shape))
    offset = _stable_offset(salt)
    dtype = str(spec["dtype"])
    constraint = spec.get("constraint") or {}
    constraint_kind = constraint.get("kind")

    if dtype == "bool":
        values = (np.arange(count, dtype=np.int64) + offset) % 2 == 1
        if count and not bool(values.any()):
            values[0] = True
    elif constraint_kind in {"index", "class_id"}:
        upper_bound = spec.get("constraint_upper_bound")
        if (
            isinstance(upper_bound, bool)
            or not isinstance(upper_bound, int)
            or upper_bound <= 0
        ):
            raise FixtureMaterializationError(
                f"{constraint_kind} fixture requires a positive exclusive bound"
            )
        start = (
            0
            if upper_bound == 1
            else 1 + (offset - 1) % (upper_bound - 1)
        )
        values = (np.arange(count, dtype=np.int64) + start) % upper_bound
    elif dtype in {"int32", "int64"}:
        values = np.arange(count, dtype=np.int64) + offset
    else:
        # Positive, non-constant values keep magnitude/history perturbations
        # observable and avoid a zero-input false pass.
        values = np.arange(count, dtype=np.float64) + float(offset)
        values = values / float(max(count, 1))

    return np.asarray(values, dtype=_numpy_dtype(np, dtype)).reshape(shape)


def _validate_bounds(value: Any, spec: Mapping[str, Any]) -> None:
    constraint = spec.get("constraint") or {}
    if constraint.get("kind") not in {"index", "class_id"} or value.size == 0:
        return
    upper_bound = int(spec["constraint_upper_bound"])
    if int(value.min()) < 0 or int(value.max()) >= upper_bound:
        raise FixtureMaterializationError(
            "fixture values escape their declared exclusive bound "
            f"[0, {upper_bound})"
        )


def cast_typed_fixture(
    values: Any,
    spec: Mapping[str, Any],
) -> Any:
    """Cast explicit values through one normalized fixture declaration.

    This is used for the asymmetric relational payload, whose edge and stable
    identity values are semantic and therefore cannot be replaced by the
    generic signal pattern.
    """

    normalized = _normalized_spec(spec)
    kind = normalized["kind"]
    dtype = normalized["dtype"]
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - repository runtime has numpy
        raise FixtureMaterializationError(
            "typed fixture materialization requires numpy"
        ) from exc

    if kind == "scalar":
        source = normalized.get("scalar_value") if values is None else values
        try:
            return _numpy_dtype(np, dtype)(source)
        except (TypeError, ValueError, OverflowError) as exc:
            raise FixtureMaterializationError(
                f"scalar value cannot be represented as {dtype}"
            ) from exc

    array = np.asarray(values, dtype=_numpy_dtype(np, dtype))
    if tuple(array.shape) != tuple(normalized["shape"]):
        raise FixtureMaterializationError(
            f"fixture payload shape {tuple(array.shape)!r} disagrees with "
            f"declared shape {tuple(normalized['shape'])!r}"
        )
    _validate_bounds(array, normalized)
    if kind == "ndarray":
        return array

    try:
        import torch
    except ImportError as exc:  # pragma: no cover - exercised only without torch
        raise FixtureMaterializationError(
            "declared tensor fixture requires torch"
        ) from exc
    return torch.as_tensor(
        array,
        dtype=_torch_dtype(torch, dtype),
        device=normalized.get("device") or "cpu",
    )


def materialize_typed_fixture(
    spec: Mapping[str, Any],
    *,
    salt: str = "fixture",
) -> Any:
    """Materialize one frozen fixture spec with deterministic signal.

    Scalars retain their exact declared literal/dimension value. Arrays use a
    stable, non-all-zero pattern; constrained ids remain inside their exact
    exclusive bound. The return container and dtype follow the frozen spec.
    """

    normalized = _normalized_spec(spec)
    if normalized["kind"] == "scalar":
        return cast_typed_fixture(None, normalized)
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - repository runtime has numpy
        raise FixtureMaterializationError(
            "typed fixture materialization requires numpy"
        ) from exc
    values = _bounded_values(np, normalized, salt=salt)
    return cast_typed_fixture(values, normalized)


def materialize_frozen_target_scaling_inputs(
    plan: Mapping[str, Any],
) -> dict[str, Any]:
    """Materialize the validator-only scaling carrier from exact plan roots.

    The frozen fixture is deliberately not paper protocol truth.  It exists
    only to exercise the declared schema-2 fitting and inference seams on one
    bounded typed payload; live acceptance separately rebinds emitted state to
    the pipeline's trusted split-lineage receipt.
    """

    scaling = plan.get("target_scaling")
    fixtures = plan.get("fixtures")
    if not isinstance(scaling, Mapping) or not isinstance(fixtures, Mapping):
        raise FixtureMaterializationError(
            "frozen execution plan lacks target_scaling or fixtures"
        )
    training = scaling.get("training_call")
    validation = scaling.get("validation_fixture")
    policy = scaling.get("policy")
    if not all(isinstance(item, Mapping) for item in (
        training, validation, policy
    )):
        raise FixtureMaterializationError(
            "frozen target-scaling plan is malformed"
        )
    if validation.get("scope") != "schema2_runtime_validator_only":
        raise FixtureMaterializationError(
            "target-scaling fixture must remain validator-only"
        )
    roots = training.get("roots")
    if not isinstance(roots, Mapping):
        raise FixtureMaterializationError(
            "frozen target-scaling training call lacks exact roots"
        )
    ids_root = roots.get("entity_ids")
    targets_root = roots.get("targets")
    if (
        not isinstance(ids_root, str)
        or ids_root not in fixtures
        or not isinstance(targets_root, str)
        or targets_root not in fixtures
    ):
        raise FixtureMaterializationError(
            "target-scaling roots do not name frozen typed fixtures"
        )
    raw_entity_ids = validation.get("entity_ids")
    raw_targets = validation.get("targets")
    fitting_range = validation.get("fitting_range")
    frozen_state = validation.get("state")
    if not isinstance(raw_entity_ids, list) or not isinstance(raw_targets, list) \
            or not isinstance(
        fitting_range, Mapping
    ) or not isinstance(frozen_state, Mapping):
        raise FixtureMaterializationError(
            "target-scaling validation fixture omits identity, range, or state"
        )
    entity_ids = cast_typed_fixture(raw_entity_ids, fixtures[ids_root])
    targets = cast_typed_fixture(raw_targets, fixtures[targets_root])
    return {
        "entity_ids": entity_ids,
        "targets": targets,
        "fitting_range": dict(fitting_range),
        "state": copy.deepcopy(dict(frozen_state)),
    }


def materialize_frozen_training_history_inputs(
    plan: Mapping[str, Any],
) -> dict[str, Any]:
    """Copy the bounded history fixture from one normalized TSF plan."""

    history = plan.get("training_history")
    fixtures = plan.get("fixtures")
    if not isinstance(history, Mapping) or not isinstance(fixtures, Mapping):
        raise FixtureMaterializationError(
            "frozen execution plan lacks training_history or fixtures"
        )
    training = history.get("training_call")
    validation = history.get("validation_fixture")
    if not isinstance(training, Mapping) or not isinstance(validation, Mapping):
        raise FixtureMaterializationError(
            "frozen training-history plan is malformed"
        )
    if validation.get("scope") != "schema2_fixed_helper_oracle_only":
        raise FixtureMaterializationError(
            "training-history fixture must remain a fixed-helper oracle"
        )
    roots = training.get("roots")
    inputs = validation.get("inputs")
    record = validation.get("record")
    if not isinstance(roots, Mapping) or not isinstance(inputs, Mapping) \
            or not isinstance(record, Mapping):
        raise FixtureMaterializationError(
            "training-history fixture omits exact roots, inputs, or record"
        )
    expected_roots = {"fitting_range", "selection_range", "seed", "config_id"}
    expected_inputs = {
        "model_id",
        "checkpoint_id",
        "seed",
        "config_id",
        "target_scaling_state",
        "fitting_range",
        "loss_index_kind",
        "loss_observations",
        "selection_range",
        "selection",
    }
    expected_record = {
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
    if set(roots) != expected_roots or set(inputs) != expected_inputs \
            or set(record) != expected_record:
        raise FixtureMaterializationError(
            "training-history fixture does not preserve the closed API/record "
            "shape"
        )
    if any(
        not isinstance(roots.get(role), str) or roots[role] not in fixtures
        for role in expected_roots
    ):
        raise FixtureMaterializationError(
            "training-history root crosswalk is not frozen"
        )
    if fixtures[roots["fitting_range"]].get("kind") != "opaque" \
            or fixtures[roots["selection_range"]].get("kind") != "opaque" \
            or fixtures[roots["config_id"]].get("kind") != "opaque":
        raise FixtureMaterializationError(
            "training-history opaque inputs changed typed convention"
        )
    seed_root = roots.get("seed")
    if fixtures[seed_root].get("kind") != "scalar" \
            or fixtures[seed_root].get("dtype") not in {"int32", "int64"}:
        raise FixtureMaterializationError(
            "training-history seed root is not one frozen integer scalar"
        )
    seed = cast_typed_fixture(inputs.get("seed"), fixtures[seed_root])
    materialized = copy.deepcopy(dict(inputs))
    materialized["seed"] = seed
    materialized["record"] = copy.deepcopy(dict(record))
    return materialized


def materialize_frozen_forecast_inputs(
    plan: Mapping[str, Any],
    *,
    constructed_models: Mapping[str, Any],
) -> dict[str, Any]:
    """Materialize kwargs from a normalized forecasting execution plan.

    ``constructed_models`` is keyed by the plan's exact architecture-block
    identity. The function never searches by parameter name or Python type.
    """

    if plan.get("schema_version") != "1.0":
        raise FixtureMaterializationError(
            "unsupported frozen execution-plan schema"
        )
    forecast = plan.get("forecast_call")
    fixtures = plan.get("fixtures")
    if not isinstance(forecast, Mapping) or not isinstance(fixtures, Mapping):
        raise FixtureMaterializationError(
            "frozen execution plan lacks forecast_call or fixtures"
        )
    bindings = forecast.get("input_bindings")
    if not isinstance(bindings, Mapping):
        raise FixtureMaterializationError(
            "frozen forecast call lacks exact input bindings"
        )

    kwargs: dict[str, Any] = {}
    relational_source: dict[str, Any] | None = None
    target_scaling_inputs: dict[str, Any] | None = None
    for parameter, raw_binding in bindings.items():
        if not isinstance(parameter, str) or not isinstance(raw_binding, Mapping):
            raise FixtureMaterializationError(
                "frozen forecast input binding is malformed"
            )
        kind = raw_binding.get("kind")
        if kind == "constructed_model":
            block = raw_binding.get("architecture_block")
            if not isinstance(block, str) or block not in constructed_models:
                raise FixtureMaterializationError(
                    f"constructed model for architecture block {block!r} is absent"
                )
            kwargs[parameter] = constructed_models[block]
        elif kind == "fixture":
            root = raw_binding.get("root")
            if not isinstance(root, str) or root not in fixtures:
                raise FixtureMaterializationError(
                    f"forecast input {parameter!r} names missing fixture {root!r}"
                )
            kwargs[parameter] = materialize_typed_fixture(
                fixtures[root], salt=root
            )
        elif kind == "literal":
            if "value" not in raw_binding:
                raise FixtureMaterializationError(
                    f"forecast input {parameter!r} omits its literal value"
                )
            kwargs[parameter] = raw_binding["value"]
        elif kind == "relational_graph":
            if relational_source is None:
                relational_source = materialize_frozen_relational_source(plan)
            if relational_source is None:
                raise FixtureMaterializationError(
                    "relational graph binding appears on a graph-free plan"
                )
            kwargs[parameter] = relational_source["graph"]
        elif kind in {
            "target_scaling_entity_ids", "target_scaling_state",
        }:
            if target_scaling_inputs is None:
                target_scaling_inputs = (
                    materialize_frozen_target_scaling_inputs(plan)
                )
            source_key = (
                "entity_ids"
                if kind == "target_scaling_entity_ids"
                else "state"
            )
            kwargs[parameter] = target_scaling_inputs[source_key]
        else:
            raise FixtureMaterializationError(
                f"forecast input {parameter!r} has unsupported binding {kind!r}"
            )
    return kwargs


def materialize_frozen_forecast_call(
    plan: Mapping[str, Any],
    *,
    constructed_models: Mapping[str, Any],
) -> tuple[list[Any], dict[str, Any]]:
    """Return exact positional args and keyword args for the frozen call."""

    values = materialize_frozen_forecast_inputs(
        plan, constructed_models=constructed_models
    )
    forecast = plan.get("forecast_call")
    if not isinstance(forecast, Mapping):
        raise FixtureMaterializationError(
            "frozen execution plan lacks forecast_call"
        )
    positional = forecast.get("positional_parameters")
    keywords = forecast.get("keyword_parameters")
    if not isinstance(positional, list) or not isinstance(keywords, list):
        raise FixtureMaterializationError(
            "frozen forecast call lacks exact call-style parameters"
        )
    ordered = [*positional, *keywords]
    if (
        any(not isinstance(name, str) or name not in values for name in ordered)
        or len(set(ordered)) != len(ordered)
        or set(ordered) != set(values)
    ):
        raise FixtureMaterializationError(
            "frozen call-style parameters do not close the materialized inputs"
        )
    return (
        [values[name] for name in positional],
        {name: values[name] for name in keywords},
    )


def materialize_frozen_relational_source(
    plan: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Replay the frozen asymmetric relational source payload exactly.

    Graph-free plans return ``None`` explicitly. Relational containers and
    dtypes come from their exact typed roots; stable ids, asymmetric edges,
    and source-degree semantics come from the normalized R2C-084 payload.
    """

    relational = plan.get("relational")
    if relational is None:
        return None
    fixtures = plan.get("fixtures")
    if not isinstance(relational, Mapping) or not isinstance(fixtures, Mapping):
        raise FixtureMaterializationError(
            "frozen relational execution plan is malformed"
        )
    source = relational.get("source_fixture")
    if not isinstance(source, Mapping):
        raise FixtureMaterializationError(
            "frozen relational plan lacks source_fixture"
        )
    roots = source.get("typed_roots")
    if not isinstance(roots, Mapping):
        raise FixtureMaterializationError(
            "frozen relational source lacks typed roots"
        )

    def typed(role: str, payload_key: str) -> Any:
        root = roots.get(role)
        if not isinstance(root, str) or root not in fixtures:
            raise FixtureMaterializationError(
                f"relational role {role!r} names missing fixture {root!r}"
            )
        return cast_typed_fixture(source.get(payload_key), fixtures[root])

    result = {
        "stable_entity_ids": typed("stable_entity_ids", "stable_entity_ids"),
        "graph": typed("graph", "graph"),
        "degrees": None,
        "fitting_identity": relational.get("fitting_identity"),
        "inference_identity": relational.get("inference_identity"),
    }
    degree_root = roots.get("degrees")
    if degree_root is not None:
        result["degrees"] = typed("degrees", "degrees")
    return result
