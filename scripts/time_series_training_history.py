"""Closed single-run/single-holdout training evidence for forecasting.

Generated training code returns one pure JSON record.  The pipeline rebinds
its fitting and optional model-selection ranges to the independently minted
R2C-077 receipt and its scaling identity to the R2C-089 state.  Version one
does not merge ranges, infer roles, or represent folds, rolling origins,
row-set partitions, final refits, or multiple training runs.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from scripts.time_series_target_scaling import validate_target_scaling_state


SCHEMA_VERSION = "1.0.0"
LOSS_INDEX_KINDS = ("epoch", "step")
SELECTION_DIRECTIONS = ("minimize", "maximize")
TIE_BREAK = "best_then_earliest"

TRAINING_HISTORY_POLICY: dict[str, object] = {
    "schema_version": SCHEMA_VERSION,
    "loss_index_kinds": list(LOSS_INDEX_KINDS),
    "selection_statuses": ["performed", "not_performed"],
    "selection_directions": list(SELECTION_DIRECTIONS),
    "tie_break": TIE_BREAK,
    "nonfinite_policy": "reject",
    "history_artifact": ".pipeline/training_history.json",
    "target_scaling_state_artifact": ".pipeline/target_scaling_state.json",
}


class TrainingHistoryError(ValueError):
    """Typed failure carrying retry ownership through pipeline adapters."""

    owner = "pipeline"
    consume_producer_retry = False

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


class TrainingHistoryCoverageError(TrainingHistoryError):
    """The closed v1 grammar cannot honestly decide this shape."""


class TrainingHistoryProducerError(TrainingHistoryError):
    """Generated code or evidence disagrees with supported authority."""

    owner = "producer"
    consume_producer_retry = True


class TrainingHistoryUpstreamError(TrainingHistoryError):
    """An existing upstream split failure remains the owning failure."""

    owner = "upstream"


@dataclass(frozen=True)
class TrainingTargetRange:
    """One exact pipeline-observed target interval."""

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


_RANGE_KEYS = frozenset({
    "validator", "target_root", "model_id", "start", "stop", "certainty",
})
_LOSS_OBSERVATION_KEYS = frozenset({
    "index", "value", "sample_weight", "checkpoint_id",
})
_SELECTION_OBSERVATION_KEYS = frozenset({
    "index", "value", "checkpoint_id",
})
_RECORDED_KEYS = frozenset({
    "schema_version", "status", "model_id", "checkpoint_id", "seed",
    "config_id", "target_scaling_state_id", "fitting_range",
    "loss_index_kind", "loss_observations", "selection_range", "selection",
    "record_digest",
})
_NOT_APPLICABLE_KEYS = frozenset({
    "schema_version", "status", "reason", "record_digest",
})
_PERFORMED_KEYS = frozenset({
    "status", "metric_id", "direction", "observations",
    "selected_checkpoint_id", "tie_break",
})
_NOT_PERFORMED_KEYS = frozenset({"status", "reason"})


def _canonical_json(value: object) -> str:
    try:
        return json.dumps(
            value, sort_keys=True, separators=(",", ":"),
            ensure_ascii=False, allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise TrainingHistoryProducerError(
            "training_history_not_json",
            f"training history is not canonical JSON: {exc}",
        ) from exc


def _digest(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _text(value: object, *, root: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise TrainingHistoryProducerError(
            "training_history_identity_malformed",
            f"{root} must be non-blank and already stripped",
        )
    return value


def _integer(value: object, *, root: str, nonnegative: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int) \
            or (nonnegative and value < 0):
        qualifier = "nonnegative " if nonnegative else ""
        raise TrainingHistoryProducerError(
            "training_history_integer_malformed",
            f"{root} must be a {qualifier}integer",
        )
    return value


def _range(value: object, *, model_id: str, label: str) -> dict[str, object]:
    if not isinstance(value, Mapping) or set(value) != _RANGE_KEYS:
        keys = set(value) if isinstance(value, Mapping) else set()
        raise TrainingHistoryProducerError(
            "training_history_range_shape",
            f"{label} has unsupported shape; missing="
            f"{sorted(_RANGE_KEYS - keys)}, unknown={sorted(keys - _RANGE_KEYS)}",
        )
    normalized = json.loads(_canonical_json(dict(value)))
    if normalized.get("validator") != "eval_split_lineage" \
            or normalized.get("certainty") != "exact":
        raise TrainingHistoryProducerError(
            "training_history_range_identity",
            f"{label} must retain exact eval_split_lineage authority",
        )
    normalized["target_root"] = _text(
        normalized.get("target_root"), root=f"{label}.target_root"
    )
    range_model = _text(
        normalized.get("model_id"), root=f"{label}.model_id"
    )
    if range_model != model_id:
        raise TrainingHistoryProducerError(
            "training_history_model_disagreement",
            f"{label} model {range_model!r} disagrees with record model "
            f"{model_id!r}",
        )
    start = _integer(
        normalized.get("start"), root=f"{label}.start", nonnegative=True
    )
    stop = _integer(
        normalized.get("stop"), root=f"{label}.stop", nonnegative=True
    )
    if stop <= start:
        raise TrainingHistoryProducerError(
            "training_history_range_empty",
            f"{label} [{start}, {stop}) must be nonempty",
        )
    return normalized


def _observations(
    value: object, *, label: str, selection: bool = False,
) -> list[dict[str, object]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) \
            or not value:
        raise TrainingHistoryProducerError(
            "training_history_observations_missing",
            f"{label} requires at least one observation",
        )
    expected = (
        _SELECTION_OBSERVATION_KEYS
        if selection else _LOSS_OBSERVATION_KEYS
    )
    rows: list[dict[str, object]] = []
    prior = -1
    checkpoints: set[str] = set()
    for offset, raw in enumerate(value):
        if not isinstance(raw, Mapping) or set(raw) != expected:
            keys = set(raw) if isinstance(raw, Mapping) else set()
            raise TrainingHistoryProducerError(
                "training_history_observation_shape",
                f"{label}[{offset}] has unsupported shape; missing="
                f"{sorted(expected - keys)}, "
                f"unknown={sorted(keys - expected)}",
            )
        index = _integer(
            raw.get("index"), root=f"{label}[{offset}].index", nonnegative=True
        )
        if index <= prior:
            raise TrainingHistoryProducerError(
                "training_history_event_order",
                f"{label} indices must be strictly increasing",
            )
        prior = index
        observed = raw.get("value")
        if isinstance(observed, bool) or not isinstance(observed, (int, float)) \
                or not math.isfinite(float(observed)):
            raise TrainingHistoryProducerError(
                "training_history_nonfinite",
                f"{label}[{offset}].value must be finite",
            )
        checkpoint = _text(
            raw.get("checkpoint_id"),
            root=f"{label}[{offset}].checkpoint_id",
        )
        if checkpoint in checkpoints:
            raise TrainingHistoryProducerError(
                "training_history_checkpoint_duplicate",
                f"{label} repeats checkpoint {checkpoint!r}",
            )
        checkpoints.add(checkpoint)
        row: dict[str, object] = {
            "index": index,
            "value": float(observed),
            "checkpoint_id": checkpoint,
        }
        if not selection:
            weight = _integer(
                raw.get("sample_weight"),
                root=f"{label}[{offset}].sample_weight",
            )
            if weight <= 0:
                raise TrainingHistoryProducerError(
                    "training_history_sample_weight",
                    f"{label}[{offset}].sample_weight must be positive",
                )
            row["sample_weight"] = weight
        rows.append(row)
    return rows


def _best_checkpoint(
    observations: Sequence[Mapping[str, object]], direction: str,
) -> str:
    choose = min if direction == "minimize" else max
    best = choose(float(row["value"]) for row in observations)
    # Observations are in strict event order, so first equality implements the
    # closed best-then-earliest tie rule.
    return next(
        str(row["checkpoint_id"])
        for row in observations
        if float(row["value"]) == best
    )


def _normalize_recorded(
    *,
    model_id: object,
    checkpoint_id: object,
    seed: object,
    config_id: object,
    target_scaling_state_id: object,
    fitting_range: object,
    loss_index_kind: object,
    loss_observations: object,
    selection_range: object,
    selection: object,
) -> dict[str, object]:
    model = _text(model_id, root="model_id")
    checkpoint = _text(checkpoint_id, root="checkpoint_id")
    seed_value = _integer(seed, root="seed", nonnegative=True)
    config = _text(config_id, root="config_id")
    state_id = _text(
        target_scaling_state_id, root="target_scaling_state_id"
    )
    fitting = _range(fitting_range, model_id=model, label="fitting_range")
    if loss_index_kind not in LOSS_INDEX_KINDS:
        raise TrainingHistoryProducerError(
            "training_history_index_kind",
            f"loss_index_kind must be one of {list(LOSS_INDEX_KINDS)!r}",
        )
    losses = _observations(loss_observations, label="loss_observations")
    loss_checkpoints = {
        (int(row["index"]), str(row["checkpoint_id"])) for row in losses
    }
    if checkpoint not in {row["checkpoint_id"] for row in losses}:
        raise TrainingHistoryProducerError(
            "training_history_checkpoint_unbound",
            "record checkpoint_id must occur in loss_observations",
        )
    if not isinstance(selection, Mapping):
        raise TrainingHistoryProducerError(
            "training_history_selection_shape", "selection must be a mapping"
        )
    selection_status = selection.get("status")
    if selection_status == "performed":
        if set(selection) != _PERFORMED_KEYS:
            raise TrainingHistoryProducerError(
                "training_history_selection_shape",
                "performed selection has unsupported shape",
            )
        selected_range = _range(
            selection_range, model_id=model, label="selection_range"
        )
        if selected_range["target_root"] != fitting["target_root"]:
            raise TrainingHistoryProducerError(
                "training_history_root_disagreement",
                "fitting and selection ranges must name the same target root",
            )
        if max(int(fitting["start"]), int(selected_range["start"])) \
                < min(int(fitting["stop"]), int(selected_range["stop"])):
            raise TrainingHistoryProducerError(
                "training_history_range_overlap",
                "fitting and selection ranges must be disjoint",
            )
        metric_id = _text(selection.get("metric_id"), root="selection.metric_id")
        direction = selection.get("direction")
        if direction not in SELECTION_DIRECTIONS:
            raise TrainingHistoryProducerError(
                "training_history_selection_direction",
                "selection.direction must be 'minimize' or 'maximize'",
            )
        selection_rows = _observations(
            selection.get("observations"), label="selection.observations",
            selection=True,
        )
        if any(
            (int(row["index"]), str(row["checkpoint_id"])) not in loss_checkpoints
            for row in selection_rows
        ):
            raise TrainingHistoryProducerError(
                "training_history_checkpoint_unbound",
                "every selection observation must bind to the same checkpoint "
                "and index in loss_observations",
            )
        if selection.get("tie_break") != TIE_BREAK:
            raise TrainingHistoryProducerError(
                "training_history_tie_break",
                f"selection.tie_break must equal {TIE_BREAK!r}",
            )
        chosen = _best_checkpoint(selection_rows, str(direction))
        declared = _text(
            selection.get("selected_checkpoint_id"),
            root="selection.selected_checkpoint_id",
        )
        if declared != chosen or checkpoint != chosen:
            raise TrainingHistoryProducerError(
                "training_history_checkpoint_disagreement",
                f"selected and recorded checkpoint must equal {TIE_BREAK} "
                f"result {chosen!r}",
            )
        normalized_selection: dict[str, object] = {
            "status": "performed",
            "metric_id": metric_id,
            "direction": direction,
            "observations": selection_rows,
            "selected_checkpoint_id": chosen,
            "tie_break": TIE_BREAK,
        }
    elif selection_status == "not_performed":
        if set(selection) != _NOT_PERFORMED_KEYS or selection_range is not None:
            raise TrainingHistoryProducerError(
                "training_history_selection_shape",
                "not_performed selection requires selection_range=None and "
                "only status/reason",
            )
        final_checkpoint = str(losses[-1]["checkpoint_id"])
        if checkpoint != final_checkpoint:
            raise TrainingHistoryProducerError(
                "training_history_no_selection_checkpoint_disagreement",
                "without model selection, the recorded checkpoint must equal "
                f"the final ordered loss checkpoint {final_checkpoint!r}",
            )
        normalized_selection = {
            "status": "not_performed",
            "reason": _text(selection.get("reason"), root="selection.reason"),
        }
        selected_range = None
    else:
        raise TrainingHistoryProducerError(
            "training_history_selection_status",
            "selection.status must be 'performed' or 'not_performed'",
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "recorded",
        "model_id": model,
        "checkpoint_id": checkpoint,
        "seed": seed_value,
        "config_id": config,
        "target_scaling_state_id": state_id,
        "fitting_range": fitting,
        "loss_index_kind": loss_index_kind,
        "loss_observations": losses,
        "selection_range": selected_range,
        "selection": normalized_selection,
    }


def record_training_history(
    *,
    model_id: str,
    checkpoint_id: str,
    seed: int,
    config_id: str,
    target_scaling_state: Mapping[str, object],
    fitting_range: Mapping[str, object],
    loss_index_kind: str,
    loss_observations: Sequence[Mapping[str, object]],
    selection_range: Mapping[str, object] | None,
    selection: Mapping[str, object],
) -> dict[str, object]:
    """Return one canonical record beside the trained model; never write."""
    try:
        scaling = validate_target_scaling_state(target_scaling_state)
    except Exception as exc:
        raise TrainingHistoryProducerError(
            "training_history_scaling_state_invalid",
            f"target_scaling_state is invalid: {exc}",
        ) from exc
    record = _normalize_recorded(
        model_id=model_id,
        checkpoint_id=checkpoint_id,
        seed=seed,
        config_id=config_id,
        target_scaling_state_id=scaling["state_digest"],
        fitting_range=fitting_range,
        loss_index_kind=loss_index_kind,
        loss_observations=loss_observations,
        selection_range=selection_range,
        selection=selection,
    )
    record["record_digest"] = _digest(record)
    return validate_training_history(record)


def not_applicable_training_history(*, reason: str) -> dict[str, object]:
    """Return the explicit arm for a method with no training phase."""
    record: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "status": "not_applicable",
        "reason": _text(reason, root="reason"),
    }
    record["record_digest"] = _digest(record)
    return validate_training_history(record)


def validate_training_history(
    record: Mapping[str, object],
) -> dict[str, object]:
    """Validate the closed record without consulting notebook prose."""
    if not isinstance(record, Mapping):
        raise TrainingHistoryProducerError(
            "training_history_shape", "training history must be a mapping"
        )
    status = record.get("status")
    expected = _NOT_APPLICABLE_KEYS if status == "not_applicable" else _RECORDED_KEYS
    if set(record) != expected:
        raise TrainingHistoryProducerError(
            "training_history_shape",
            "training history has unsupported shape; missing="
            f"{sorted(expected - set(record))}, unknown={sorted(set(record) - expected)}",
        )
    normalized = json.loads(_canonical_json(dict(record)))
    digest = normalized.pop("record_digest", None)
    if not isinstance(digest, str) or digest != _digest(normalized):
        raise TrainingHistoryProducerError(
            "training_history_digest", "training history failed its canonical digest"
        )
    if normalized.get("schema_version") != SCHEMA_VERSION:
        raise TrainingHistoryProducerError(
            "training_history_schema", "training history schema is unsupported"
        )
    if status == "not_applicable":
        _text(normalized.get("reason"), root="reason")
        normalized["record_digest"] = digest
        return normalized
    if status != "recorded":
        raise TrainingHistoryProducerError(
            "training_history_status", "status must be recorded or not_applicable"
        )
    rebuilt = _normalize_recorded(
        model_id=normalized.get("model_id"),
        checkpoint_id=normalized.get("checkpoint_id"),
        seed=normalized.get("seed"),
        config_id=normalized.get("config_id"),
        target_scaling_state_id=normalized.get("target_scaling_state_id"),
        fitting_range=normalized.get("fitting_range"),
        loss_index_kind=normalized.get("loss_index_kind"),
        loss_observations=normalized.get("loss_observations"),
        selection_range=normalized.get("selection_range"),
        selection=normalized.get("selection"),
    )
    if rebuilt != normalized:
        raise TrainingHistoryProducerError(
            "training_history_canonical_disagreement",
            "training history is not the canonical closed version-one record",
        )
    normalized["record_digest"] = digest
    return normalized


def _receipt_rows(
    split_receipt: Mapping[str, object] | None, *, role: str,
) -> list[Mapping[str, object]]:
    if not isinstance(split_receipt, Mapping):
        raise TrainingHistoryCoverageError(
            "training_history_split_receipt_missing",
            "eval_split_lineage receipt is required",
        )
    status = split_receipt.get("status")
    if status == "invalid":
        raise TrainingHistoryUpstreamError(
            "split_lineage_invalid",
            "eval_split_lineage already reports an invalid target split",
        )
    if status != "valid":
        raise TrainingHistoryCoverageError(
            "training_history_split_unresolved",
            f"training history requires a valid split receipt; got {status!r}",
        )
    if split_receipt.get("validator") != "eval_split_lineage":
        raise TrainingHistoryCoverageError(
            "training_history_split_receipt_untrusted",
            "range authority must come from eval_split_lineage",
        )
    evidence = split_receipt.get("evidence")
    key = "fitting_target_ranges" if role == "fitting" \
        else "selection_target_ranges"
    rows = evidence.get(key) if isinstance(evidence, Mapping) else None
    if not isinstance(rows, list) or any(
        not isinstance(row, Mapping) for row in rows
    ):
        raise TrainingHistoryCoverageError(
            "training_history_range_carrier_missing",
            f"eval_split_lineage receipt omits valid evidence.{key}",
        )
    return list(rows)


def resolve_training_target_range(
    split_receipt: Mapping[str, object] | None,
    *,
    role: str,
    target_root: str,
    model_id: str,
) -> TrainingTargetRange:
    """Resolve one exact fitting or selection interval without widening."""
    if role not in {"fitting", "model_selection"}:
        raise TrainingHistoryCoverageError(
            "training_history_role_unsupported",
            "version one resolves only fitting and model_selection",
        )
    target_root = _text(target_root, root="target_root")
    model_id = _text(model_id, root="model_id")
    distinct: set[tuple[int, int]] = set()
    for row in _receipt_rows(split_receipt, role=role):
        if row.get("root") != target_root or row.get("model_id") != model_id:
            continue
        if row.get("subset") is not None or row.get("certainty") != "exact":
            raise TrainingHistoryCoverageError(
                "training_history_range_grammar_unsupported",
                f"matching {role} lineage is a subset/support envelope",
            )
        span = row.get("protocol_range")
        if not isinstance(span, Mapping):
            raise TrainingHistoryCoverageError(
                "training_history_range_carrier_malformed",
                f"matching {role} lineage omits protocol_range",
            )
        start, stop = span.get("start"), span.get("stop")
        if isinstance(start, bool) or isinstance(stop, bool) \
                or not isinstance(start, int) or not isinstance(stop, int) \
                or start < 0 or stop <= start:
            raise TrainingHistoryCoverageError(
                "training_history_range_grammar_unsupported",
                f"matching {role} lineage must be one exact nonempty interval",
            )
        distinct.add((start, stop))
    if not distinct:
        raise TrainingHistoryProducerError(
            "training_history_range_binding_disagreement",
            f"{role} root/model ({target_root!r}, {model_id!r}) is absent "
            "from the valid receipt",
        )
    if len(distinct) != 1:
        raise TrainingHistoryCoverageError(
            "training_history_range_grammar_unsupported",
            f"multiple {role} intervals exist; v1 does not merge or choose",
        )
    start, stop = next(iter(distinct))
    return TrainingTargetRange(
        validator="eval_split_lineage", target_root=target_root,
        model_id=model_id, start=start, stop=stop,
    )


def validate_training_history_lineage(
    record: Mapping[str, object],
    split_receipt: Mapping[str, object] | None,
    *,
    target_scaling_state: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Bind one record to trusted ranges and, when supplied, scaling state."""
    normalized = validate_training_history(record)
    if normalized["status"] == "not_applicable":
        return normalized
    if target_scaling_state is not None:
        try:
            scaling = validate_target_scaling_state(target_scaling_state)
        except Exception as exc:
            raise TrainingHistoryUpstreamError(
                "target_scaling_state_invalid", str(exc)
            ) from exc
        if normalized["target_scaling_state_id"] != scaling["state_digest"]:
            raise TrainingHistoryProducerError(
                "training_history_scaling_identity_disagreement",
                "training history names a different target-scaling state",
            )
    model_id = str(normalized["model_id"])
    fitting = normalized["fitting_range"]
    authority = resolve_training_target_range(
        split_receipt, role="fitting",
        target_root=str(fitting["target_root"]), model_id=model_id,
    )
    if fitting != authority.as_dict():
        raise TrainingHistoryProducerError(
            "training_history_range_disagreement",
            "fitting range disagrees with pipeline authority",
        )
    selection = normalized["selection"]
    if selection["status"] == "performed":
        selected_range = normalized["selection_range"]
        authority = resolve_training_target_range(
            split_receipt, role="model_selection",
            target_root=str(selected_range["target_root"]), model_id=model_id,
        )
        if selected_range != authority.as_dict():
            raise TrainingHistoryProducerError(
                "training_history_range_disagreement",
                "selection range disagrees with pipeline authority",
            )
    else:
        matching = [
            row for row in _receipt_rows(split_receipt, role="model_selection")
            if row.get("root") == fitting["target_root"]
            and row.get("model_id") == model_id
        ]
        if matching:
            if any(row.get("subset") is not None
                   or row.get("certainty") != "exact" for row in matching):
                raise TrainingHistoryCoverageError(
                    "training_history_range_grammar_unsupported",
                    "selection exists only as unsupported subset/support evidence",
                )
            raise TrainingHistoryProducerError(
                "training_history_no_selection_disagreement",
                "record claims no selection although the trusted receipt records it",
            )
    return normalized


__all__ = [
    "LOSS_INDEX_KINDS",
    "SCHEMA_VERSION",
    "SELECTION_DIRECTIONS",
    "TIE_BREAK",
    "TRAINING_HISTORY_POLICY",
    "TrainingHistoryCoverageError",
    "TrainingHistoryError",
    "TrainingHistoryProducerError",
    "TrainingHistoryUpstreamError",
    "TrainingTargetRange",
    "not_applicable_training_history",
    "record_training_history",
    "resolve_training_target_range",
    "validate_training_history",
    "validate_training_history_lineage",
]
