"""Training-only, stable-identity target scaling for forecasting.

The family build plan owns the closed version-one policy.  This module owns
only deterministic mechanics and validation:

* resolve one exact fitting-target interval from the pipeline's R2C-077 split
  receipt;
* fit per-series mean-absolute state by typed stable identity;
* transform targets and invert typed forecast outputs without row-position
  joins; and
* bind persisted state back to the pipeline-minted fitting carrier.

No evaluation span, forecast horizon, array length, variable spelling, or
paper identity is used to infer a fitting boundary.  Unsupported lineage
grammar is a pipeline coverage issue; disagreements inside the supported
contract are producer-owned.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import PurePosixPath
import numpy as np


STATE_SCHEMA_VERSION = "1.0.0"
CONTRACT_SCHEMA_VERSION = "1.0.0"
SUPPORTED_MODES = ("none", "per_series_affine")
OUTPUT_INVERSION_RULES = {
    "location": "multiply_scale",
    "samples": "multiply_scale",
    "distribution_scale": "multiply_absolute_scale",
    "variance": "multiply_squared_scale",
    "unitless_shape": "unchanged",
}


class TargetScalingError(ValueError):
    """Typed failure carrying retry ownership for stage routing."""

    owner = "pipeline"
    consume_producer_retry = False

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


class TargetScalingCoverageError(TargetScalingError):
    """The fixed v1 grammar cannot honestly resolve the requested case."""


class TargetScalingProducerError(TargetScalingError):
    """Generated contract, code, or persisted state disagrees with authority."""

    owner = "producer"
    consume_producer_retry = True


class TargetScalingUpstreamError(TargetScalingError):
    """An existing upstream split failure remains the owning failure."""

    owner = "upstream"


@dataclass(frozen=True)
class FittingTargetRange:
    """One exact pipeline-observed fitting-target half-open interval."""

    validator: str
    target_root: str
    model_id: str
    start: int
    stop: int
    certainty: str = "exact"

    def as_dict(self) -> dict[str, object]:
        return {
            "validator": self.validator,
            "target_root": self.target_root,
            "model_id": self.model_id,
            "start": self.start,
            "stop": self.stop,
            "certainty": self.certainty,
        }


_CONTRACT_KEYS = {
    "schema_version",
    "mode",
    "supported_modes",
    "fitting_role",
    "entity_id_root",
    "target_root",
    "target_entity_axis",
    "target_protocol_axis",
    "statistic",
    "centering",
    "epsilon",
    "nonfinite_policy",
    "zero_series_policy",
    "state_artifact",
    "output_inversion",
}


def _exact_text(value: object, *, root: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise TargetScalingCoverageError(
            "target_scaling_contract_unsupported",
            f"{root} must be non-blank and already stripped",
        )
    return value


def _exact_int(value: object, *, root: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TargetScalingCoverageError(
            "target_scaling_contract_unsupported",
            f"{root} must be an integer",
        )
    return value


def _canonical_json(value: object) -> str:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise TargetScalingProducerError(
            "target_scaling_state_not_json",
            f"target scaling state is not canonical JSON: {exc}",
        ) from exc


def _digest(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def validate_target_scaling_contract(
    contract: Mapping[str, object],
) -> dict[str, object]:
    """Return one normalized closed v1 family contract."""
    if not isinstance(contract, Mapping):
        raise TargetScalingCoverageError(
            "target_scaling_contract_unsupported",
            "build_plan.target_scaling must be a mapping",
        )
    keys = set(contract)
    if keys != _CONTRACT_KEYS:
        raise TargetScalingCoverageError(
            "target_scaling_contract_unsupported",
            "build_plan.target_scaling has unsupported shape; missing="
            f"{sorted(_CONTRACT_KEYS - keys)}, unknown={sorted(keys - _CONTRACT_KEYS)}",
        )
    if contract.get("schema_version") != CONTRACT_SCHEMA_VERSION:
        raise TargetScalingCoverageError(
            "target_scaling_contract_unsupported",
            "build_plan.target_scaling.schema_version must equal "
            f"{CONTRACT_SCHEMA_VERSION!r}",
        )

    raw_supported = contract.get("supported_modes")
    if not isinstance(raw_supported, list) or tuple(raw_supported) != SUPPORTED_MODES:
        raise TargetScalingCoverageError(
            "target_scaling_contract_unsupported",
            "build_plan.target_scaling.supported_modes must be exactly "
            f"{list(SUPPORTED_MODES)!r}",
        )
    mode = _exact_text(contract.get("mode"), root="target_scaling.mode")
    if mode not in SUPPORTED_MODES:
        raise TargetScalingCoverageError(
            "unsupported_target_scaling_mode",
            f"target scaling mode {mode!r} is outside the v1 grammar "
            f"{list(SUPPORTED_MODES)!r}",
        )
    exact_values = {
        "fitting_role": "fitting",
        "statistic": "mean_absolute",
        "centering": "none",
        "nonfinite_policy": "reject",
        "zero_series_policy": (
            "unit_scale_when_all_absolute_values_lte_epsilon"
        ),
    }
    for field, expected in exact_values.items():
        actual = _exact_text(
            contract.get(field), root=f"target_scaling.{field}"
        )
        if actual != expected:
            raise TargetScalingCoverageError(
                "target_scaling_contract_unsupported",
                f"target_scaling.{field} must equal {expected!r}; got {actual!r}",
            )

    entity_id_root = _exact_text(
        contract.get("entity_id_root"), root="target_scaling.entity_id_root"
    )
    target_root = _exact_text(
        contract.get("target_root"), root="target_scaling.target_root"
    )
    entity_axis = _exact_int(
        contract.get("target_entity_axis"),
        root="target_scaling.target_entity_axis",
    )
    protocol_axis = _exact_int(
        contract.get("target_protocol_axis"),
        root="target_scaling.target_protocol_axis",
    )
    if entity_axis < 0 or protocol_axis < 0 or entity_axis == protocol_axis:
        raise TargetScalingCoverageError(
            "target_scaling_contract_unsupported",
            "target scaling entity/protocol axes must be distinct non-negative integers",
        )
    epsilon = contract.get("epsilon")
    if (
        isinstance(epsilon, bool)
        or not isinstance(epsilon, (int, float))
        or not math.isfinite(float(epsilon))
        or float(epsilon) <= 0.0
    ):
        raise TargetScalingCoverageError(
            "target_scaling_contract_unsupported",
            "target_scaling.epsilon must be a positive finite number",
        )

    state_artifact = _exact_text(
        contract.get("state_artifact"), root="target_scaling.state_artifact"
    )
    artifact_path = PurePosixPath(state_artifact)
    if (
        artifact_path.is_absolute()
        or ".." in artifact_path.parts
        or state_artifact != artifact_path.as_posix()
    ):
        raise TargetScalingCoverageError(
            "target_scaling_contract_unsupported",
            "target_scaling.state_artifact must be a normalized relative path",
        )

    inversion = contract.get("output_inversion")
    if not isinstance(inversion, Mapping) or dict(inversion) != OUTPUT_INVERSION_RULES:
        raise TargetScalingCoverageError(
            "target_scaling_contract_unsupported",
            "target_scaling.output_inversion must equal the closed v1 role grammar",
        )

    normalized = dict(contract)
    normalized.update({
        "mode": mode,
        "entity_id_root": entity_id_root,
        "target_root": target_root,
        "target_entity_axis": entity_axis,
        "target_protocol_axis": protocol_axis,
        "epsilon": float(epsilon),
        "state_artifact": state_artifact,
        "output_inversion": dict(inversion),
        "supported_modes": list(raw_supported),
    })
    return normalized


def resolve_fitting_target_range(
    split_receipt: Mapping[str, object] | None,
    *,
    target_root: str,
    model_id: str,
) -> FittingTargetRange:
    """Resolve one exact fitting carrier from a trusted split receipt.

    Identical repeated observations deduplicate.  Distinct ranges, subset
    supports, unresolved lineage, and malformed receipt grammar never widen to
    a guessed interval.
    """
    target_root = _exact_text(target_root, root="target_root")
    model_id = _exact_text(model_id, root="model_id")
    if not isinstance(split_receipt, Mapping):
        raise TargetScalingCoverageError(
            "fitting_range_receipt_missing",
            "eval_split_lineage receipt is required to certify target scaling",
        )
    status = split_receipt.get("status")
    if status == "invalid":
        raise TargetScalingUpstreamError(
            "split_lineage_invalid",
            "eval_split_lineage already reports an invalid target split; "
            "preserve that upstream finding",
        )
    if status != "valid":
        raise TargetScalingCoverageError(
            "fitting_range_lineage_unresolved",
            "target scaling requires a valid eval_split_lineage receipt; "
            f"got status {status!r}",
        )
    if split_receipt.get("validator") != "eval_split_lineage":
        raise TargetScalingCoverageError(
            "fitting_range_receipt_untrusted",
            "target scaling fitting authority must come from eval_split_lineage",
        )
    evidence = split_receipt.get("evidence")
    rows = evidence.get("fitting_target_ranges") \
        if isinstance(evidence, Mapping) else None
    if not isinstance(rows, list):
        raise TargetScalingCoverageError(
            "fitting_range_carrier_missing",
            "eval_split_lineage receipt omits evidence.fitting_target_ranges",
        )

    matching: list[tuple[int, int]] = []
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise TargetScalingCoverageError(
                "fitting_range_carrier_malformed",
                f"fitting_target_ranges[{index}] must be a mapping",
            )
        row_root = row.get("root")
        row_model = row.get("model_id")
        if not isinstance(row_root, str) or not isinstance(row_model, str):
            raise TargetScalingCoverageError(
                "fitting_range_carrier_malformed",
                f"fitting_target_ranges[{index}] omits exact root/model identity",
            )
        if row_root != target_root or row_model != model_id:
            continue
        if row.get("subset") is not None or row.get("certainty") != "exact":
            raise TargetScalingCoverageError(
                "fitting_range_grammar_unsupported",
                "matching fitting target lineage is a subset/support envelope; "
                "v1 requires one exact interval",
            )
        span = row.get("protocol_range")
        if not isinstance(span, Mapping):
            raise TargetScalingCoverageError(
                "fitting_range_carrier_malformed",
                "matching fitting target lineage omits protocol_range",
            )
        start = span.get("start")
        stop = span.get("stop")
        if (
            isinstance(start, bool)
            or isinstance(stop, bool)
            or not isinstance(start, int)
            or not isinstance(stop, int)
            or start < 0
            or stop <= start
        ):
            raise TargetScalingCoverageError(
                "fitting_range_grammar_unsupported",
                "matching fitting target lineage must be one nonempty absolute "
                "half-open interval",
            )
        matching.append((start, stop))

    distinct = sorted(set(matching))
    if not distinct:
        raise TargetScalingProducerError(
            "fitting_range_binding_disagreement",
            "generated target scaling binding names root/model "
            f"({target_root!r}, {model_id!r}) absent from the valid pipeline receipt",
        )
    if len(distinct) != 1:
        raise TargetScalingCoverageError(
            "fitting_range_grammar_unsupported",
            "multiple distinct fitting intervals exist for root/model "
            f"({target_root!r}, {model_id!r}); v1 does not union or choose among "
            f"{distinct!r}",
        )
    start, stop = distinct[0]
    return FittingTargetRange(
        validator="eval_split_lineage",
        target_root=target_root,
        model_id=model_id,
        start=start,
        stop=stop,
    )


def _identity_scalar(value: object, *, root: str) -> tuple[dict[str, object], tuple[str, object]]:
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, bool):
        raise TargetScalingProducerError(
            "stable_entity_id_unsupported",
            f"{root} contains boolean identity {value!r}; expected integer or string",
        )
    if isinstance(value, int):
        return {"type": "integer", "value": value}, ("integer", value)
    if isinstance(value, str):
        if not value:
            raise TargetScalingProducerError(
                "stable_entity_id_unsupported",
                f"{root} contains an empty string identity",
            )
        return {"type": "string", "value": value}, ("string", value)
    raise TargetScalingProducerError(
        "stable_entity_id_unsupported",
        f"{root} contains unsupported identity {value!r}; expected integer or string",
    )


def _identity_values(values: object, *, root: str) -> list[tuple[dict[str, object], tuple[str, object]]]:
    if hasattr(values, "detach") and hasattr(values, "cpu"):
        values = values.detach().cpu().tolist()
    elif isinstance(values, np.ndarray):
        values = values.tolist()
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise TargetScalingProducerError(
            "stable_entity_ids_not_vector",
            f"{root} must be a one-dimensional sequence of stable ids",
        )
    raw = list(values)
    if not raw or any(isinstance(item, (list, tuple, dict)) for item in raw):
        raise TargetScalingProducerError(
            "stable_entity_ids_not_vector",
            f"{root} must be a nonempty one-dimensional stable-id vector",
        )
    typed = [_identity_scalar(item, root=root) for item in raw]
    seen: set[tuple[str, object]] = set()
    duplicates: list[dict[str, object]] = []
    for serialized, key in typed:
        if key in seen:
            duplicates.append(serialized)
        seen.add(key)
    if duplicates:
        raise TargetScalingProducerError(
            "duplicate_stable_entity_ids",
            f"{root} contains duplicate typed stable ids {duplicates!r}",
        )
    return typed


def _numpy_fitting_array(targets: object, *, target_root: str) -> np.ndarray:
    if hasattr(targets, "detach") and hasattr(targets, "cpu"):
        array = targets.detach().cpu().numpy()
    elif isinstance(targets, np.ndarray):
        array = targets
    else:
        raise TargetScalingProducerError(
            "target_container_unsupported",
            f"{target_root} must be a NumPy array or torch tensor",
        )
    if array.dtype.kind not in {"f"}:
        raise TargetScalingProducerError(
            "target_dtype_unsupported",
            f"{target_root} must use a real floating dtype; got {array.dtype}",
        )
    return np.asarray(array)


def _entry_sort_key(entry: Mapping[str, object]) -> str:
    return _canonical_json(entry.get("entity_id"))


def fit_target_scaling_state(
    contract: Mapping[str, object],
    *,
    entity_ids: object,
    targets: object,
    fitting_range: FittingTargetRange,
) -> dict[str, object]:
    """Fit one canonical state over exactly ``fitting_range[start:stop]``."""
    normalized = validate_target_scaling_contract(contract)
    if not isinstance(fitting_range, FittingTargetRange):
        raise TargetScalingProducerError(
            "fitting_range_binding_missing",
            "target scaling requires an explicit FittingTargetRange carrier",
        )
    if (
        fitting_range.validator != "eval_split_lineage"
        or fitting_range.certainty != "exact"
        or not fitting_range.target_root
        or not fitting_range.model_id
        or isinstance(fitting_range.start, bool)
        or isinstance(fitting_range.stop, bool)
        or not isinstance(fitting_range.start, int)
        or not isinstance(fitting_range.stop, int)
    ):
        raise TargetScalingProducerError(
            "fitting_range_binding_invalid",
            "target scaling fitting carrier must retain exact "
            "eval_split_lineage root/model/range authority",
        )
    ids = _identity_values(
        entity_ids, root=str(normalized["entity_id_root"])
    )
    array = _numpy_fitting_array(
        targets, target_root=str(normalized["target_root"])
    )
    entity_axis = int(normalized["target_entity_axis"])
    protocol_axis = int(normalized["target_protocol_axis"])
    if max(entity_axis, protocol_axis) >= array.ndim:
        raise TargetScalingProducerError(
            "target_axes_disagree",
            f"{normalized['target_root']} shape {array.shape!r} does not carry "
            f"declared entity/protocol axes {(entity_axis, protocol_axis)!r}",
        )
    if array.shape[entity_axis] != len(ids):
        raise TargetScalingProducerError(
            "target_identity_count_disagrees",
            f"{normalized['target_root']} entity-axis size {array.shape[entity_axis]} "
            f"does not match {normalized['entity_id_root']} count {len(ids)}",
        )
    if fitting_range.start < 0 or fitting_range.stop <= fitting_range.start \
            or fitting_range.stop > array.shape[protocol_axis]:
        raise TargetScalingProducerError(
            "fitting_range_out_of_bounds",
            f"pipeline fitting range [{fitting_range.start}, {fitting_range.stop}) "
            f"is outside {normalized['target_root']} protocol-axis size "
            f"{array.shape[protocol_axis]}",
        )

    aligned = np.moveaxis(array, (entity_axis, protocol_axis), (0, 1))
    fitting = aligned[:, fitting_range.start:fitting_range.stop, ...]
    mode = str(normalized["mode"])
    epsilon = float(normalized["epsilon"])
    entries: list[dict[str, object]] = []
    for index, (serialized_id, _) in enumerate(ids):
        values = np.asarray(fitting[index], dtype=np.float64).reshape(-1)
        if mode == "none":
            statistic_value: float | None = None
            scale = 1.0
            resolution = "mode_none"
            observation_count = 0
        else:
            if not np.isfinite(values).all():
                raise TargetScalingProducerError(
                    "nonfinite_fitting_target",
                    f"{normalized['target_root']} has non-finite fitting values for "
                    f"stable id {serialized_id!r}; policy is reject",
                )
            absolute = np.abs(values)
            statistic_value = float(np.mean(absolute))
            observation_count = int(values.size)
            if bool(np.all(absolute <= epsilon)):
                scale = 1.0
                resolution = "unit_scale"
            else:
                scale = statistic_value
                resolution = "not_applied"
            if not math.isfinite(scale) or scale <= 0.0:
                raise TargetScalingProducerError(
                    "invalid_fitted_scale",
                    f"{normalized['target_root']} produced invalid scale {scale!r} "
                    f"for stable id {serialized_id!r}",
                )
        entries.append({
            "entity_id": serialized_id,
            "statistic_value": statistic_value,
            "scale": scale,
            "fitting_observation_count": observation_count,
            "zero_series_resolution": resolution,
            "certainty": "exact",
        })
    entries.sort(key=_entry_sort_key)

    state: dict[str, object] = {
        "schema_version": STATE_SCHEMA_VERSION,
        "contract_digest": _digest(normalized),
        "mode": mode,
        "entity_id_root": normalized["entity_id_root"],
        "target_root": normalized["target_root"],
        "target_entity_axis": entity_axis,
        "target_protocol_axis": protocol_axis,
        "fitting_lineage": fitting_range.as_dict(),
        "statistic": normalized["statistic"],
        "centering": normalized["centering"],
        "epsilon": epsilon,
        "nonfinite_policy": normalized["nonfinite_policy"],
        "zero_series_policy": normalized["zero_series_policy"],
        "state_artifact": normalized["state_artifact"],
        "output_inversion": dict(normalized["output_inversion"]),
        "entity_count": len(entries),
        "entries": entries,
    }
    state["state_digest"] = _digest(state)
    return state


_STATE_KEYS = {
    "schema_version",
    "contract_digest",
    "mode",
    "entity_id_root",
    "target_root",
    "target_entity_axis",
    "target_protocol_axis",
    "fitting_lineage",
    "statistic",
    "centering",
    "epsilon",
    "nonfinite_policy",
    "zero_series_policy",
    "state_artifact",
    "output_inversion",
    "entity_count",
    "entries",
    "state_digest",
}


def validate_target_scaling_state(
    state: Mapping[str, object],
    *,
    contract: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Validate integrity, typed identity, and optional contract agreement."""
    if not isinstance(state, Mapping) or set(state) != _STATE_KEYS:
        keys = set(state) if isinstance(state, Mapping) else set()
        raise TargetScalingProducerError(
            "target_scaling_state_shape",
            "target scaling state has wrong shape; missing="
            f"{sorted(_STATE_KEYS - keys)}, unknown={sorted(keys - _STATE_KEYS)}",
        )
    normalized = json.loads(_canonical_json(dict(state)))
    digest = normalized.pop("state_digest", None)
    if not isinstance(digest, str) or digest != _digest(normalized):
        raise TargetScalingProducerError(
            "target_scaling_state_digest",
            f"state artifact {state.get('state_artifact')!r} failed its canonical digest",
        )
    normalized["state_digest"] = digest
    if normalized.get("schema_version") != STATE_SCHEMA_VERSION:
        raise TargetScalingProducerError(
            "target_scaling_state_schema",
            f"state artifact {normalized.get('state_artifact')!r} has unsupported schema",
        )
    if contract is not None:
        family = validate_target_scaling_contract(contract)
        if normalized.get("contract_digest") != _digest(family):
            raise TargetScalingProducerError(
                "target_scaling_contract_disagreement",
                f"state artifact {normalized.get('state_artifact')!r} does not match "
                "build_plan.target_scaling",
            )
    entries = normalized.get("entries")
    entity_count = normalized.get("entity_count")
    if (
        isinstance(entity_count, bool)
        or not isinstance(entity_count, int)
        or not isinstance(entries, list)
        or entity_count != len(entries)
        or entity_count <= 0
    ):
        raise TargetScalingProducerError(
            "target_scaling_state_entity_count",
            f"state artifact {normalized.get('state_artifact')!r} has incoherent entity_count",
        )
    seen: set[tuple[str, object]] = set()
    for index, entry in enumerate(entries):
        if not isinstance(entry, Mapping):
            raise TargetScalingProducerError(
                "target_scaling_state_entry",
                f"state artifact entry {index} must be a mapping",
            )
        raw_id = entry.get("entity_id")
        if not isinstance(raw_id, Mapping) or set(raw_id) != {"type", "value"}:
            raise TargetScalingProducerError(
                "target_scaling_state_identity",
                f"state artifact entry {index} has malformed typed identity",
            )
        identity_type = raw_id.get("type")
        identity_value = raw_id.get("value")
        if identity_type == "integer" and isinstance(identity_value, int) \
                and not isinstance(identity_value, bool):
            key = ("integer", identity_value)
        elif identity_type == "string" and isinstance(identity_value, str) \
                and bool(identity_value):
            key = ("string", identity_value)
        else:
            raise TargetScalingProducerError(
                "target_scaling_state_identity",
                f"state artifact entry {index} has unsupported typed identity {raw_id!r}",
            )
        if key in seen:
            raise TargetScalingProducerError(
                "target_scaling_state_duplicate_identity",
                f"state artifact {normalized.get('state_artifact')!r} repeats "
                f"typed identity {raw_id!r}",
            )
        seen.add(key)
        scale = entry.get("scale")
        if (
            isinstance(scale, bool)
            or not isinstance(scale, (int, float))
            or not math.isfinite(float(scale))
            or float(scale) <= 0.0
        ):
            raise TargetScalingProducerError(
                "target_scaling_state_scale",
                f"state artifact entry {index} has invalid scale {scale!r}",
            )
    return normalized


def validate_training_only_state(
    state: Mapping[str, object],
    split_receipt: Mapping[str, object] | None,
    *,
    contract: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Bind persisted range/root/model back to pipeline-minted authority."""
    normalized = validate_target_scaling_state(state, contract=contract)
    lineage = normalized.get("fitting_lineage")
    if not isinstance(lineage, Mapping):
        raise TargetScalingProducerError(
            "target_scaling_state_lineage",
            "target scaling state omits fitting_lineage",
        )
    target_root = lineage.get("target_root")
    model_id = lineage.get("model_id")
    if not isinstance(target_root, str) or not isinstance(model_id, str):
        raise TargetScalingProducerError(
            "target_scaling_state_lineage",
            "target scaling state fitting_lineage omits root/model identity",
        )
    authority = resolve_fitting_target_range(
        split_receipt, target_root=target_root, model_id=model_id
    )
    if lineage != authority.as_dict():
        raise TargetScalingProducerError(
            "target_scaling_state_range_disagreement",
            "state artifact fitting lineage disagrees with the exact pipeline "
            f"carrier: state={dict(lineage)!r}, pipeline={authority.as_dict()!r}",
        )
    return normalized


def _state_scale_lookup(
    state: Mapping[str, object],
) -> tuple[dict[tuple[str, object], float], dict[str, object]]:
    normalized = validate_target_scaling_state(state)
    lookup: dict[tuple[str, object], float] = {}
    for entry in normalized["entries"]:
        raw_id = entry["entity_id"]
        lookup[(raw_id["type"], raw_id["value"])] = float(entry["scale"])
    return lookup, normalized


def _scale_vector(
    state: Mapping[str, object],
    entity_ids: object,
    *,
    batch_entity_id_root: str,
) -> tuple[list[float], dict[str, object]]:
    batch_entity_id_root = _exact_text(
        batch_entity_id_root, root="batch_entity_id_root"
    )
    lookup, normalized = _state_scale_lookup(state)
    typed = _identity_values(entity_ids, root=batch_entity_id_root)
    unknown = [serialized for serialized, key in typed if key not in lookup]
    if unknown:
        raise TargetScalingProducerError(
            "unknown_target_scaling_identity",
            f"state root {normalized['state_artifact']!r} keyed by "
            f"{normalized['entity_id_root']!r} has no scale for typed ids "
            f"{unknown!r} from batch root {batch_entity_id_root!r}",
        )
    return [lookup[key] for _, key in typed], normalized


def _reshape_scales(values: object, scales: list[float], *, entity_axis: int) -> object:
    shape = getattr(values, "shape", None)
    ndim = getattr(values, "ndim", None)
    if not isinstance(ndim, int) or not isinstance(shape, tuple):
        # torch.Size is tuple-like but not a tuple on every supported release.
        try:
            shape = tuple(shape)
            ndim = len(shape)
        except (TypeError, ValueError):
            raise TargetScalingProducerError(
                "target_output_container_unsupported",
                "scaled values must be a NumPy array or torch tensor",
            ) from None
    if entity_axis < 0:
        entity_axis += ndim
    if entity_axis < 0 or entity_axis >= ndim:
        raise TargetScalingProducerError(
            "target_output_entity_axis",
            f"entity axis {entity_axis} is outside output shape {shape!r}",
        )
    if shape[entity_axis] != len(scales):
        raise TargetScalingProducerError(
            "target_output_identity_count",
            f"output entity-axis size {shape[entity_axis]} does not match stable-id "
            f"count {len(scales)}",
        )
    reshape = [1] * ndim
    reshape[entity_axis] = len(scales)
    if isinstance(values, np.ndarray):
        if values.dtype.kind != "f":
            raise TargetScalingProducerError(
                "target_output_dtype_unsupported",
                f"scaled NumPy output must be floating; got {values.dtype}",
            )
        return np.asarray(scales, dtype=values.dtype).reshape(reshape)
    if hasattr(values, "new_tensor") and hasattr(values, "reshape"):
        if not getattr(values.dtype, "is_floating_point", False):
            raise TargetScalingProducerError(
                "target_output_dtype_unsupported",
                f"scaled torch output must be floating; got {values.dtype}",
            )
        return values.new_tensor(scales).reshape(reshape)
    raise TargetScalingProducerError(
        "target_output_container_unsupported",
        "scaled values must be a NumPy array or torch tensor",
    )


def transform_targets(
    values: object,
    entity_ids: object,
    state: Mapping[str, object],
    *,
    batch_entity_id_root: str,
    entity_axis: int = 0,
) -> object:
    """Divide target values by the stable-id keyed scale vector."""
    scales, _ = _scale_vector(
        state, entity_ids, batch_entity_id_root=batch_entity_id_root
    )
    scale_array = _reshape_scales(values, scales, entity_axis=entity_axis)
    return values / scale_array


def inverse_forecast_output(
    values: object,
    entity_ids: object,
    state: Mapping[str, object],
    *,
    output_role: str,
    batch_entity_id_root: str,
    entity_axis: int,
) -> object:
    """Invert one output using the closed unit semantics for its typed role."""
    if output_role not in OUTPUT_INVERSION_RULES:
        raise TargetScalingCoverageError(
            "target_output_role_unsupported",
            f"forecast output role {output_role!r} is outside the v1 grammar "
            f"{sorted(OUTPUT_INVERSION_RULES)!r}",
        )
    scales, normalized = _scale_vector(
        state, entity_ids, batch_entity_id_root=batch_entity_id_root
    )
    declared = normalized.get("output_inversion")
    if not isinstance(declared, Mapping) \
            or declared.get(output_role) != OUTPUT_INVERSION_RULES[output_role]:
        raise TargetScalingProducerError(
            "target_output_inversion_disagreement",
            f"state artifact {normalized.get('state_artifact')!r} disagrees with "
            f"the {output_role!r} inversion rule",
        )
    scale_array = _reshape_scales(values, scales, entity_axis=entity_axis)
    if output_role == "variance":
        return values * scale_array * scale_array
    if output_role == "unitless_shape":
        # Still validate identity and entity-axis alignment above; only the
        # arithmetic is an identity.
        return values * (scale_array / scale_array)
    # Scales fitted here are positive. abs is still the declared distribution
    # rule and keeps the implementation correct if an affine family later
    # carries a signed multiplicative coefficient under a new schema.
    if output_role == "distribution_scale":
        if isinstance(scale_array, np.ndarray):
            scale_array = np.abs(scale_array)
        else:
            scale_array = scale_array.abs()
    return values * scale_array


__all__ = [
    "FittingTargetRange",
    "OUTPUT_INVERSION_RULES",
    "SUPPORTED_MODES",
    "TargetScalingCoverageError",
    "TargetScalingError",
    "TargetScalingProducerError",
    "TargetScalingUpstreamError",
    "fit_target_scaling_state",
    "inverse_forecast_output",
    "resolve_fitting_target_range",
    "transform_targets",
    "validate_target_scaling_contract",
    "validate_target_scaling_state",
    "validate_training_only_state",
]
