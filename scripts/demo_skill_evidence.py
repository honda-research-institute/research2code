"""Pure structured demo-skill evaluation (R2C-086).

Notebook prose is presentation.  This module accepts only a family-owned
comparison contract, one executed-evaluation record, and a pipeline-owned
validity receipt.  It recomputes every metric over the same rows and mask and
keeps execution, evaluation validity, and demonstrated skill separate.

The module is stdlib-only so both the post-smoke adapter and the portable
probe harness can vendor the same decision primitive.
"""

from __future__ import annotations

import hashlib
import json
import math
from copy import deepcopy
from typing import Any, Mapping, Sequence


SCHEMA_VERSION = "1.0.0"
LEGACY_EXECUTED_RECORD_SCHEMA_VERSION = "1.0.0"
EXECUTED_RECORD_SCHEMA_VERSION = "2.0.0"
EVALUATION_PROTOCOL_ROLES = frozenset({
    "context_length",
    "forecast_call_horizon",
    "validation_span",
    "test_span",
})
RECORD_PREFIX = "R2C_DEMO_EVALUATION_JSON: "


def _json_identity(payload: object) -> str:
    """Stable identity using the producer's canonical JSON-null recipe."""

    def normalize(value: object) -> object:
        if isinstance(value, float) and not math.isfinite(value):
            return None
        if isinstance(value, Mapping):
            return {
                str(key): normalize(item)
                for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
            }
        if isinstance(value, (list, tuple)):
            return [normalize(item) for item in value]
        return value

    raw = json.dumps(
        normalize(payload), ensure_ascii=False, sort_keys=True,
        separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def compute_actual_values_id(
    row_ids: Sequence[object], actuals: Sequence[object], units: str,
) -> str:
    """Identity of the exact scored values, row order, and units."""
    return _json_identity({
        "row_ids": list(row_ids),
        "actuals": list(actuals),
        "units": units,
    })


def compute_finite_mask_id(
    row_ids: Sequence[object], finite_mask: Sequence[object],
) -> str:
    """Identity of the exact scored mask in row order."""
    return _json_identity({
        "row_ids": list(row_ids),
        "finite_mask": list(finite_mask),
    })


def _reason(code: str, message: str) -> dict[str, str]:
    return {"code": code, "message": message}


def _status(status: str, reasons: Sequence[dict[str, str]] = ()) -> dict[str, Any]:
    return {"status": status, "reasons": list(reasons)}


def _base_result() -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "execution": _status("completed"),
        "evaluation_validity": _status("unresolved"),
        "evaluation_protocol_role": None,
        "skill": _status("undetermined"),
        "comparisons": [],
        "presentation_headline": (
            "Demo skill: UNDETERMINED — structured evidence is incomplete."
        ),
    }


def _nonempty_string(value: object) -> str | None:
    return value if isinstance(value, str) and value.strip() else None


def _number(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    converted = float(value)
    return converted if math.isfinite(converted) else None


def _canonical_equal(left: object, right: object) -> bool:
    """JSON-typed equality (unlike Python, ``1`` and ``1.0`` differ)."""
    return _json_identity(left) == _json_identity(right)


def _exact_id_list(value: object) -> list[str] | None:
    """Return one exact, nonempty identifier list without normalization."""
    if not isinstance(value, list) or not value \
            or any(not isinstance(item, str) or not item.strip() for item in value) \
            or len(set(value)) != len(value):
        return None
    return list(value)


def _exact_role_carrier(value: object) -> dict[str, Any] | None:
    """Validate the portable carrier without accepting aliases or prose."""
    if not isinstance(value, Mapping):
        return None
    role = value.get("role")
    scheme_kind = value.get("scheme_kind")
    scheme_ids = _exact_id_list(value.get("scheme_paper_element_ids"))
    role_ids = _exact_id_list(value.get("role_paper_element_ids"))
    if role not in EVALUATION_PROTOCOL_ROLES \
            or not isinstance(scheme_kind, str) or not scheme_kind.strip() \
            or scheme_ids is None or role_ids is None:
        return None
    return {
        "role": role,
        "scheme_kind": scheme_kind,
        "scheme_paper_element_ids": scheme_ids,
        "role_paper_element_ids": role_ids,
    }


def resolve_executed_evaluation_protocol_role(
    executed_record: Mapping[str, object] | None,
    evaluation_protocol: Mapping[str, object] | None,
) -> tuple[dict[str, Any] | None, str | None]:
    """Resolve a schema-2 record's declared role against R2C-082 truth.

    The executed record names the role; it never derives one from the number
    of evaluation positions.  The returned carrier is a small portable slice
    of the already-validated method-spec protocol: exact scheme identity plus
    the scheme and role-specific paper groundings an evaluation-only probe may
    honestly bind.  Archived schema-1 records deliberately return no carrier.
    """
    if not isinstance(executed_record, Mapping):
        return None, "executed_evaluation_record_missing"
    version = executed_record.get("schema_version")
    if version == LEGACY_EXECUTED_RECORD_SCHEMA_VERSION:
        return None, None
    if version != EXECUTED_RECORD_SCHEMA_VERSION:
        return None, "executed_record_schema_unsupported"

    role = executed_record.get("evaluation_protocol_role")
    if not isinstance(role, str) or role not in EVALUATION_PROTOCOL_ROLES:
        return None, "executed_evaluation_protocol_role_missing_or_unknown"
    if not isinstance(evaluation_protocol, Mapping):
        return None, "typed_evaluation_protocol_missing"

    scheme = evaluation_protocol.get("scheme")
    quantities = evaluation_protocol.get("quantities")
    if not isinstance(scheme, Mapping) or not isinstance(quantities, list):
        return None, "typed_evaluation_protocol_malformed"
    matching = [
        quantity for quantity in quantities
        if isinstance(quantity, Mapping) and quantity.get("role") == role
    ]
    if len(matching) != 1:
        return None, "executed_evaluation_protocol_role_not_unique"

    scheme_kind = scheme.get("kind")
    scheme_ids = _exact_id_list(scheme.get("paper_element_ids"))
    role_ids = _exact_id_list(matching[0].get("paper_element_ids"))
    if not isinstance(scheme_kind, str) or not scheme_kind.strip() \
            or scheme_ids is None or role_ids is None:
        return None, "typed_evaluation_protocol_grounding_missing"

    carrier = {
        "role": role,
        "scheme_kind": scheme_kind,
        "scheme_paper_element_ids": scheme_ids,
        "role_paper_element_ids": role_ids,
    }
    return _exact_role_carrier(carrier), None


def _metric(
    aggregation: str,
    actuals: Sequence[float],
    predictions: Sequence[float],
) -> float | None:
    if not actuals or len(actuals) != len(predictions):
        return None
    if aggregation == "root_mean_squared_error":
        return math.sqrt(sum(
            (actual - prediction) ** 2
            for actual, prediction in zip(actuals, predictions)
        ) / len(actuals))
    if aggregation == "mean_absolute_error":
        return sum(
            abs(actual - prediction)
            for actual, prediction in zip(actuals, predictions)
        ) / len(actuals)
    if aggregation == "accuracy":
        return sum(
            1 for actual, prediction in zip(actuals, predictions)
            if actual == prediction
        ) / len(actuals)
    return None


def _contract_parts(
    contract: Mapping[str, object],
) -> tuple[Mapping[str, object] | None, list[Mapping[str, object]], list[dict[str, str]]]:
    errors: list[dict[str, str]] = []
    metric = contract.get("primary_metric")
    if not isinstance(metric, Mapping):
        errors.append(_reason("contract_metric_missing", "primary_metric is missing"))
        metric = None
    comparators = contract.get("comparators")
    if not isinstance(comparators, list):
        errors.append(_reason("contract_comparators_missing", "comparators are missing"))
        comparator_rows: list[Mapping[str, object]] = []
    else:
        comparator_rows = [item for item in comparators if isinstance(item, Mapping)]
        if len(comparator_rows) != len(comparators):
            errors.append(_reason(
                "contract_comparator_malformed",
                "every comparator declaration must be a mapping",
            ))
    if contract.get("decision_policy") != "all_required":
        errors.append(_reason(
            "contract_policy_unsupported",
            "only the all_required decision policy is supported",
        ))
    return metric, comparator_rows, errors


def validate_forecast_magnitude_evidence(
    family_contract: Mapping[str, object] | None,
    executed_record: Mapping[str, object] | None,
) -> dict[str, Any]:
    """Validate the exact schema-2 record consumed by TSF-5.

    Magnitude collapse is intentionally independent of split validity: it
    compares each executed model prediction with the same row's declared
    pre-window repeat-last input.  The comparator is usable only when the
    family contract names that implementation and the record preserves exact
    row, position, unit, actual-value, and mask identity.

    ``unsupported`` means the required evidence grammar is absent, while
    ``invalid`` means a schema-2 producer record disagrees with the supported
    grammar.  This small stdlib-only result is shared by live and portable
    forecasting probes.
    """

    def result(
        status: str,
        code: str,
        message: str,
        **values: object,
    ) -> dict[str, Any]:
        return {
            "status": status,
            "reason": code,
            "message": message,
            **values,
        }

    if not isinstance(family_contract, Mapping):
        return result(
            "unsupported",
            "magnitude_family_contract_missing",
            "the family comparison contract is missing",
        )
    declarations = family_contract.get("comparators")
    if not isinstance(declarations, list):
        return result(
            "unsupported",
            "repeat_last_contract_missing",
            "the family declares no repeat-last comparator grammar",
        )
    repeat_declarations = [
        item for item in declarations
        if isinstance(item, Mapping)
        and item.get("implementation") == "repeat_last_pre_window"
    ]
    if len(repeat_declarations) != 1:
        return result(
            "unsupported",
            "repeat_last_contract_not_unique",
            "the family must declare exactly one repeat_last_pre_window comparator",
        )
    comparator_id = repeat_declarations[0].get("id")
    if not isinstance(comparator_id, str) or not comparator_id.strip():
        return result(
            "unsupported",
            "repeat_last_contract_id_missing",
            "the repeat-last comparator has no exact identifier",
        )

    if not isinstance(executed_record, Mapping):
        return result(
            "unsupported",
            "executed_evaluation_record_missing",
            "the executed evaluation record is missing",
        )
    if executed_record.get("schema_version") != EXECUTED_RECORD_SCHEMA_VERSION:
        return result(
            "unsupported",
            "executed_record_schema_unsupported",
            "magnitude evidence requires an exact schema-2 executed record",
        )

    row_ids = executed_record.get("row_ids")
    actuals = executed_record.get("actuals")
    finite_mask = executed_record.get("finite_mask")
    model_predictions = executed_record.get("model_predictions")
    positions = executed_record.get("evaluation_positions")
    units = executed_record.get("units")
    vectors = (row_ids, actuals, finite_mask, model_predictions, positions)
    if not all(isinstance(value, list) for value in vectors) or not row_ids:
        return result(
            "invalid",
            "magnitude_rows_malformed",
            "schema-2 magnitude rows must carry five nonempty vectors",
        )
    if not (
        len(row_ids) == len(actuals) == len(finite_mask)
        == len(model_predictions) == len(positions)
    ):
        return result(
            "invalid",
            "magnitude_row_lengths_disagree",
            "model, actual, mask, row, and position vectors must share one length",
        )
    if len({_json_identity(item) for item in row_ids}) != len(row_ids):
        return result(
            "invalid",
            "magnitude_row_ids_duplicate",
            "schema-2 magnitude row ids must be unique",
        )
    if any(
        not isinstance(position, int) or isinstance(position, bool)
        for position in positions
    ):
        return result(
            "invalid",
            "magnitude_evaluation_positions_malformed",
            "schema-2 magnitude positions must be integer target offsets",
        )
    if not _nonempty_string(units):
        return result(
            "invalid",
            "magnitude_units_missing",
            "schema-2 magnitude rows must name their target units",
        )
    if not all(isinstance(item, bool) for item in finite_mask):
        return result(
            "invalid",
            "magnitude_finite_mask_malformed",
            "schema-2 magnitude finite_mask must contain booleans",
        )
    expected_mask = [_number(value) is not None for value in actuals]
    if finite_mask != expected_mask:
        return result(
            "invalid",
            "magnitude_finite_mask_disagrees",
            "schema-2 magnitude finite_mask disagrees with actual values",
        )
    actual_id = compute_actual_values_id(row_ids, actuals, str(units))
    mask_id = compute_finite_mask_id(row_ids, finite_mask)
    if executed_record.get("actual_values_id") != actual_id \
            or executed_record.get("finite_mask_id") != mask_id:
        return result(
            "invalid",
            "magnitude_record_identity_mismatch",
            "schema-2 magnitude record has stale actual-value or mask identity",
        )

    comparators = executed_record.get("comparators")
    repeat = comparators.get(comparator_id) \
        if isinstance(comparators, Mapping) else None
    if not isinstance(repeat, Mapping):
        return result(
            "unsupported",
            "repeat_last_evidence_missing",
            f"executed evidence for comparator {comparator_id!r} is missing",
        )
    if repeat.get("implementation") != "repeat_last_pre_window":
        return result(
            "invalid",
            "repeat_last_implementation_disagrees",
            "executed repeat-last evidence disagrees with the family implementation",
        )
    for key, expected in (
        ("row_ids", row_ids),
        ("evaluation_positions", positions),
        ("units", units),
        ("actual_values_id", actual_id),
        ("finite_mask_id", mask_id),
    ):
        if not _canonical_equal(repeat.get(key), expected):
            return result(
                "invalid",
                "repeat_last_identity_mismatch",
                "repeat-last evidence does not preserve exact row, position, unit, actual, and mask identity",
            )

    input_values = repeat.get("input_values")
    input_positions = repeat.get("input_positions")
    repeat_predictions = repeat.get("predictions")
    if not all(isinstance(value, list) for value in (
        input_values, input_positions, repeat_predictions,
    )):
        return result(
            "unsupported",
            "repeat_last_inputs_missing",
            "repeat-last evidence lacks input values, positions, or predictions",
        )
    if not (
        len(input_values) == len(input_positions)
        == len(repeat_predictions) == len(row_ids)
    ):
        return result(
            "invalid",
            "repeat_last_row_lengths_disagree",
            "repeat-last vectors must share the executed row count",
        )
    window_boundary = min(positions) - 1
    if any(
        not isinstance(position, int) or isinstance(position, bool)
        or position != window_boundary
        for position in input_positions
    ):
        return result(
            "invalid",
            "repeat_last_not_window_boundary",
            "repeat-last inputs must use the final position before the complete evaluation window",
        )

    model_values = [_number(value) for value in model_predictions]
    history_values = [_number(value) for value in input_values]
    repeated_values = [_number(value) for value in repeat_predictions]
    eligible_indices = [
        index for index, eligible in enumerate(finite_mask) if eligible
    ]
    if not eligible_indices:
        return result(
            "unsupported",
            "magnitude_rows_ineligible",
            "schema-2 magnitude evidence has no finite eligible rows",
        )
    if any(model_values[index] is None for index in eligible_indices):
        return result(
            "invalid",
            "magnitude_model_predictions_nonfinite",
            "every eligible schema-2 model prediction used by magnitude evidence must be finite",
        )
    if any(value is None for value in history_values) \
            or any(value is None for value in repeated_values):
        return result(
            "invalid",
            "repeat_last_values_nonfinite",
            "every repeat-last input and prediction used by magnitude evidence must be finite",
        )
    if repeated_values != history_values:
        return result(
            "invalid",
            "repeat_last_not_derived_from_history",
            "repeat-last predictions must equal the exact pre-window input values",
        )
    return result(
        "ready",
        "magnitude_evidence_ready",
        "schema-2 model predictions and repeat-last history are exactly coidentified",
        row_ids=[row_ids[index] for index in eligible_indices],
        model_predictions=[
            float(model_values[index])  # type: ignore[arg-type]
            for index in eligible_indices
        ],
        history_values=[
            float(history_values[index])  # type: ignore[arg-type]
            for index in eligible_indices
        ],
    )


def bind_split_validity_receipt(
    split_receipt: Mapping[str, object] | None,
    executed_record: Mapping[str, object] | None,
    *,
    evaluation_protocol: Mapping[str, object] | None = None,
) -> dict[str, Any]:
    """Bind a lineage receipt to the exact executed rows and split range.

    The notebook cannot self-assert validity.  For a ``valid`` split receipt,
    this adapter requires an exact model/root relation and proves that every
    executed evaluation position lies inside that relation's reported-target
    half-open range before attaching the record identities.
    """
    if not isinstance(split_receipt, Mapping):
        bound: dict[str, Any] = {
            "schema_version": "1.0.0",
            "status": "unresolved",
            "validator": "eval_split_lineage",
            "reasons": ["split_validity_receipt_missing"],
            "evidence": {},
        }
    else:
        bound = deepcopy(dict(split_receipt))

    role_carrier, role_error = resolve_executed_evaluation_protocol_role(
        executed_record, evaluation_protocol,
    )
    if role_carrier is not None:
        bound["evaluation_protocol_role"] = role_carrier
    elif role_error not in (None, "executed_record_schema_unsupported"):
        reasons = list(bound.get("reasons") or [])
        if role_error not in reasons:
            reasons.append(role_error)
        bound["reasons"] = reasons
        if bound.get("status") == "valid":
            bound["status"] = "unresolved"

    if bound.get("status") != "valid":
        return bound
    if not isinstance(executed_record, Mapping):
        bound["status"] = "unresolved"
        bound["reasons"] = ["executed_evaluation_record_missing"]
        return bound

    row_ids = executed_record.get("row_ids")
    actuals = executed_record.get("actuals")
    finite_mask = executed_record.get("finite_mask")
    positions = executed_record.get("evaluation_positions")
    units = executed_record.get("units")
    root = executed_record.get("target_root")
    model_id = executed_record.get("model_id")
    if not all(isinstance(value, list) for value in (
        row_ids, actuals, finite_mask, positions,
    )) or not _nonempty_string(units) or not _nonempty_string(root) \
            or not _nonempty_string(model_id):
        bound["status"] = "unresolved"
        bound["reasons"] = ["executed_evaluation_binding_fields_missing"]
        return bound
    if not (len(row_ids) == len(actuals) == len(finite_mask) == len(positions)):
        bound["status"] = "invalid"
        bound["reasons"] = ["executed_evaluation_row_lengths_disagree"]
        return bound

    actual_id = compute_actual_values_id(row_ids, actuals, str(units))
    mask_id = compute_finite_mask_id(row_ids, finite_mask)
    if executed_record.get("actual_values_id") != actual_id \
            or executed_record.get("finite_mask_id") != mask_id:
        bound["status"] = "invalid"
        bound["reasons"] = ["executed_evaluation_identity_mismatch"]
        return bound

    evidence = bound.get("evidence")
    relations = evidence.get("fitting_to_reported_evaluation_relations") \
        if isinstance(evidence, Mapping) else None
    matching_ranges: list[tuple[int, int]] = []
    for relation in relations if isinstance(relations, list) else []:
        if not isinstance(relation, Mapping) or relation.get("status") != "disjoint" \
                or relation.get("root") != root or relation.get("model_id") != model_id:
            continue
        for side in (relation.get("first"), relation.get("second")):
            if not isinstance(side, Mapping) \
                    or side.get("role") != "reported_evaluation" \
                    or side.get("read_kind") != "target":
                continue
            span = side.get("protocol_range")
            if isinstance(span, Mapping) \
                    and isinstance(span.get("start"), int) \
                    and isinstance(span.get("stop"), int):
                matching_ranges.append((span["start"], span["stop"]))
    if not matching_ranges or not all(
        isinstance(position, int) and not isinstance(position, bool)
        and any(start <= position < stop for start, stop in matching_ranges)
        for position in positions
    ):
        bound["status"] = "unresolved"
        bound["reasons"] = ["executed_window_not_bound_to_split_proof"]
        return bound

    bound.update({
        "actual_values_id": actual_id,
        "finite_mask_id": mask_id,
        "row_ids": list(row_ids),
        "evaluation_positions": list(positions),
        "units": units,
        "target_root": root,
        "model_id": model_id,
    })
    return bound


def derive_demo_skill_evidence(
    family_contract: Mapping[str, object] | None,
    executed_record: Mapping[str, object] | None,
    validity_receipt: Mapping[str, object] | None,
) -> dict[str, Any]:
    """Recompute task skill without consulting printed notebook prose."""
    result = _base_result()
    if not isinstance(family_contract, Mapping) or not family_contract:
        result["evaluation_validity"] = _status("not_applicable")
        result["skill"] = _status("not_applicable")
        result["presentation_headline"] = (
            "Demo skill: NOT APPLICABLE — this family declares no skill contract."
        )
        return result

    raw_receipt_role = validity_receipt.get("evaluation_protocol_role") \
        if isinstance(validity_receipt, Mapping) else None
    receipt_role = _exact_role_carrier(raw_receipt_role)
    if receipt_role is not None:
        result["evaluation_protocol_role"] = deepcopy(receipt_role)

    receipt_status = validity_receipt.get("status") \
        if isinstance(validity_receipt, Mapping) else None
    receipt_reasons = (validity_receipt.get("reasons") or []) \
        if isinstance(validity_receipt, Mapping) else []
    receipt_reason_rows = [
        _reason("evaluation_" + str(item), str(item).replace("_", " "))
        for item in receipt_reasons if isinstance(item, str)
    ]
    if receipt_status == "invalid":
        result["evaluation_validity"] = _status("invalid", receipt_reason_rows)
        result["skill"] = _status("undetermined", [_reason(
            "evaluation_invalid", "invalid evaluation cannot demonstrate skill",
        )])
        result["presentation_headline"] = (
            "Demo skill: UNDETERMINED — the evaluation is invalid."
        )
        return result
    if receipt_status != "valid":
        reasons = receipt_reason_rows or [_reason(
            "evaluation_validity_unresolved",
            "a trusted valid evaluation receipt is unavailable",
        )]
        result["evaluation_validity"] = _status("unresolved", reasons)
        result["skill"] = _status("undetermined", [_reason(
            "evaluation_unresolved",
            "unresolved evaluation validity cannot demonstrate skill",
        )])
        return result

    if not isinstance(executed_record, Mapping):
        result["evaluation_validity"] = _status("unresolved", [_reason(
            "executed_record_missing", "the structured executed-evaluation record is missing",
        )])
        result["skill"] = _status("undetermined", [_reason(
            "comparison_evidence_missing", "no structured comparison evidence is available",
        )])
        return result
    executed_schema = executed_record.get("schema_version")
    if executed_schema not in {
        LEGACY_EXECUTED_RECORD_SCHEMA_VERSION,
        EXECUTED_RECORD_SCHEMA_VERSION,
    }:
        reason = [_reason(
            "executed_record_schema_unsupported",
            "executed evaluation schema must be legacy 1.0.0 or current 2.0.0",
        )]
        result["evaluation_validity"] = _status("unresolved", reason)
        result["skill"] = _status("undetermined", reason)
        return result
    if executed_schema == EXECUTED_RECORD_SCHEMA_VERSION:
        role = executed_record.get("evaluation_protocol_role")
        if not isinstance(role, str) or role not in EVALUATION_PROTOCOL_ROLES:
            reason = [_reason(
                "executed_evaluation_protocol_role_missing_or_unknown",
                "schema-2 executed evaluation must name one exact R2C-082 protocol role",
            )]
            result["evaluation_validity"] = _status("unresolved", reason)
            result["skill"] = _status("undetermined", reason)
            return result
        if receipt_role is None:
            reason = [_reason(
                "executed_evaluation_protocol_role_unresolved",
                "schema-2 executed evaluation role is not bound to the typed method-spec protocol",
            )]
            result["evaluation_validity"] = _status("unresolved", reason)
            result["skill"] = _status("undetermined", reason)
            return result
        if receipt_role.get("role") != role:
            invalid = [_reason(
                "executed_evaluation_protocol_role_mismatch",
                "the executed record and trusted receipt name different evaluation-protocol roles",
            )]
            result["evaluation_validity"] = _status("invalid", invalid)
            result["skill"] = _status("undetermined", invalid)
            return result

    metric, comparator_contracts, contract_errors = _contract_parts(family_contract)
    if contract_errors or metric is None:
        result["evaluation_validity"] = _status("unresolved", contract_errors)
        result["skill"] = _status("undetermined", contract_errors)
        return result

    row_ids = executed_record.get("row_ids")
    actuals = executed_record.get("actuals")
    finite_mask = executed_record.get("finite_mask")
    predictions = executed_record.get("model_predictions")
    positions = executed_record.get("evaluation_positions")
    units = executed_record.get("units")
    record_lists = (row_ids, actuals, finite_mask, predictions, positions)
    if not all(isinstance(value, list) for value in record_lists) or not row_ids \
            or not (len(row_ids) == len(actuals) == len(finite_mask)
                    == len(predictions) == len(positions)):
        invalid = [_reason(
            "executed_rows_malformed",
            "row ids, actuals, mask, model predictions, and positions must have one shared nonzero length",
        )]
        result["evaluation_validity"] = _status("invalid", invalid)
        result["skill"] = _status("undetermined", invalid)
        return result
    if any(
        not isinstance(position, int) or isinstance(position, bool)
        for position in positions
    ):
        invalid = [_reason(
            "evaluation_positions_malformed",
            "evaluation positions must be integer offsets in the target root",
        )]
        result["evaluation_validity"] = _status("invalid", invalid)
        result["skill"] = _status("undetermined", invalid)
        return result
    if len({_json_identity(item) for item in row_ids}) != len(row_ids):
        invalid = [_reason("duplicate_row_ids", "evaluation row ids must be unique")]
        result["evaluation_validity"] = _status("invalid", invalid)
        result["skill"] = _status("undetermined", invalid)
        return result
    if not all(isinstance(item, bool) for item in finite_mask):
        invalid = [_reason("finite_mask_malformed", "finite_mask must contain booleans")]
        result["evaluation_validity"] = _status("invalid", invalid)
        result["skill"] = _status("undetermined", invalid)
        return result
    expected_mask = [_number(value) is not None for value in actuals]
    if finite_mask != expected_mask:
        invalid = [_reason(
            "finite_mask_disagrees_with_actuals",
            "finite_mask must select every and only finite executed actual value",
        )]
        result["evaluation_validity"] = _status("invalid", invalid)
        result["skill"] = _status("undetermined", invalid)
        return result

    metric_id = metric.get("id")
    aggregation = metric.get("aggregation")
    direction = metric.get("direction")
    contract_units = metric.get("units")
    if executed_record.get("metric_id") != metric_id \
            or executed_record.get("aggregation") != aggregation \
            or units != contract_units or direction not in {
                "lower_is_better", "higher_is_better",
            }:
        invalid = [_reason(
            "metric_contract_mismatch",
            "executed metric id, aggregation, units, or direction disagrees with the family contract",
        )]
        result["evaluation_validity"] = _status("invalid", invalid)
        result["skill"] = _status("undetermined", invalid)
        return result

    actual_id = compute_actual_values_id(row_ids, actuals, str(units))
    mask_id = compute_finite_mask_id(row_ids, finite_mask)
    identity_fields = {
        "actual_values_id": actual_id,
        "finite_mask_id": mask_id,
        "row_ids": row_ids,
        "evaluation_positions": positions,
        "units": units,
        "target_root": executed_record.get("target_root"),
        "model_id": executed_record.get("model_id"),
    }
    if any(executed_record.get(key) != value for key, value in (
        ("actual_values_id", actual_id), ("finite_mask_id", mask_id),
    )) or any(validity_receipt.get(key) != value for key, value in identity_fields.items()):
        invalid = [_reason(
            "evaluation_identity_mismatch",
            "the executed record and trusted receipt do not name the same values, mask, rows, positions, units, root, and model",
        )]
        result["evaluation_validity"] = _status("invalid", invalid)
        result["skill"] = _status("undetermined", invalid)
        return result

    selected_actuals: list[float] = []
    selected_model: list[float] = []
    selected_indices: list[int] = []
    for index, keep in enumerate(finite_mask):
        if not keep:
            continue
        actual = _number(actuals[index])
        if actual is None:
            invalid = [_reason(
                "selected_actual_nonfinite",
                "every row selected by finite_mask must carry a finite actual value",
            )]
            result["evaluation_validity"] = _status("invalid", invalid)
            result["skill"] = _status("undetermined", invalid)
            return result
        selected_actuals.append(actual)
        selected_indices.append(index)
        prediction = _number(predictions[index])
        if prediction is None:
            result["evaluation_validity"] = _status("valid", receipt_reason_rows)
            result["skill"] = _status("undetermined", [_reason(
                "model_metric_nonfinite",
                "the model prediction or primary metric is missing or non-finite",
            )])
            return result
        selected_model.append(prediction)

    eligibility = family_contract.get("eligibility")
    eligibility = eligibility if isinstance(eligibility, Mapping) else {}
    minimum = eligibility.get("minimum_finite_rows", 1)
    if isinstance(minimum, bool) or not isinstance(minimum, int) \
            or len(selected_actuals) < minimum:
        result["evaluation_validity"] = _status("valid", receipt_reason_rows)
        result["skill"] = _status("undetermined", [_reason(
            "insufficient_finite_rows",
            "the evaluation has too few eligible finite rows",
        )])
        return result
    if eligibility.get("require_nonzero_actuals") is True \
            and not any(value != 0.0 for value in selected_actuals):
        result["evaluation_validity"] = _status("valid", receipt_reason_rows)
        result["skill"] = _status("undetermined", [_reason(
            "actuals_have_no_signal", "eligible actual values are identically zero",
        )])
        return result

    model_metric = _metric(str(aggregation), selected_actuals, selected_model)
    if model_metric is None or not math.isfinite(model_metric):
        result["evaluation_validity"] = _status("valid", receipt_reason_rows)
        result["skill"] = _status("undetermined", [_reason(
            "model_metric_unavailable", "the primary model metric cannot be recomputed",
        )])
        return result
    recorded_model_metric = _number(executed_record.get("model_metric"))
    if recorded_model_metric is None or not math.isclose(
        recorded_model_metric, model_metric, rel_tol=1e-12, abs_tol=1e-12
    ):
        invalid = [_reason(
            "recorded_model_metric_mismatch",
            "the recorded model metric does not match pipeline recomputation",
        )]
        result["evaluation_validity"] = _status("invalid", invalid)
        result["skill"] = _status("undetermined", invalid)
        return result

    record_comparators = executed_record.get("comparators")
    record_comparators = record_comparators if isinstance(record_comparators, Mapping) else {}
    evidence_errors: list[dict[str, str]] = []
    invalid_evaluation: list[dict[str, str]] = []
    comparisons: list[dict[str, Any]] = []
    for declaration in comparator_contracts:
        comparator_id = declaration.get("id")
        entry = record_comparators.get(comparator_id)
        if not isinstance(comparator_id, str) or not isinstance(entry, Mapping):
            if declaration.get("required") is True:
                evidence_errors.append(_reason(
                    "required_comparator_missing",
                    f"structured evidence for comparator {comparator_id!r} is missing",
                ))
            continue
        if entry.get("implementation") != declaration.get("implementation") \
                or any(not _canonical_equal(entry.get(key), value) for key, value in (
                    ("row_ids", row_ids),
                    ("evaluation_positions", positions),
                    ("units", units),
                    ("actual_values_id", actual_id),
                    ("finite_mask_id", mask_id),
                )):
            invalid_evaluation.append(_reason(
                "comparator_identity_mismatch",
                f"comparator {comparator_id!r} does not use the exact evaluation rows, mask, positions, units, and actual values",
            ))
            continue
        comparator_predictions = entry.get("predictions")
        if not isinstance(comparator_predictions, list) \
                or len(comparator_predictions) != len(row_ids):
            evidence_errors.append(_reason(
                "comparator_predictions_missing",
                f"comparator {comparator_id!r} has no complete prediction vector",
            ))
            continue
        implementation = declaration.get("implementation")
        if implementation == "predict_zero":
            if any(_number(value) != 0.0 for value in comparator_predictions):
                invalid_evaluation.append(_reason(
                    "predict_zero_disagrees",
                    "the predict-zero comparator contains a nonzero prediction",
                ))
                continue
        elif implementation == "repeat_last_pre_window":
            input_values = entry.get("input_values")
            input_positions = entry.get("input_positions")
            if not isinstance(input_values, list) or not isinstance(input_positions, list) \
                    or len(input_values) != len(row_ids) \
                    or len(input_positions) != len(row_ids):
                evidence_errors.append(_reason(
                    "repeat_last_inputs_missing",
                    "repeat-last requires one input value and position per evaluation row",
                ))
                continue
            evaluation_window_start = min(positions)
            if any(
                not isinstance(source, int) or isinstance(source, bool)
                or source >= evaluation_window_start
                for source in input_positions
            ):
                invalid_evaluation.append(_reason(
                    "repeat_last_leaks_evaluation",
                    "every repeat-last input must precede the complete evaluation window",
                ))
                continue
            if any(
                source != evaluation_window_start - 1
                for source in input_positions
            ):
                invalid_evaluation.append(_reason(
                    "repeat_last_not_window_boundary",
                    "repeat-last must use the final position immediately before the evaluation window",
                ))
                continue
            if any(
                _number(prediction) is None or _number(source) is None
                or _number(prediction) != _number(source)
                for prediction, source in zip(comparator_predictions, input_values)
            ):
                invalid_evaluation.append(_reason(
                    "repeat_last_not_derived_from_history",
                    "repeat-last predictions must equal the recorded pre-window inputs",
                ))
                continue
        elif implementation == "constant_prediction":
            constant = _number(declaration.get("constant_value"))
            if constant is None or any(
                _number(value) != constant for value in comparator_predictions
            ):
                invalid_evaluation.append(_reason(
                    "constant_comparator_disagrees",
                    "the constant comparator must use the family-declared value",
                ))
                continue
        else:
            evidence_errors.append(_reason(
                "comparator_implementation_unsupported",
                f"comparator implementation {implementation!r} is unsupported",
            ))
            continue

        selected_comparator: list[float] = []
        for index in selected_indices:
            value = _number(comparator_predictions[index])
            if value is None:
                evidence_errors.append(_reason(
                    "comparator_metric_nonfinite",
                    f"comparator {comparator_id!r} has a missing or non-finite eligible prediction",
                ))
                break
            selected_comparator.append(value)
        else:
            comparator_metric = _metric(
                str(aggregation), selected_actuals, selected_comparator
            )
            margin = _number(declaration.get("absolute_margin"))
            tolerance = _number(declaration.get("tolerance"))
            if comparator_metric is None or margin is None or tolerance is None \
                    or margin < 0 or tolerance < 0:
                evidence_errors.append(_reason(
                    "comparison_threshold_invalid",
                    f"comparator {comparator_id!r} cannot be evaluated with its declared thresholds",
                ))
                continue
            recorded_comparator_metric = _number(entry.get("metric"))
            if recorded_comparator_metric is None or not math.isclose(
                recorded_comparator_metric, comparator_metric,
                rel_tol=1e-12, abs_tol=1e-12,
            ):
                invalid_evaluation.append(_reason(
                    "recorded_comparator_metric_mismatch",
                    f"recorded metric for comparator {comparator_id!r} does not match pipeline recomputation",
                ))
                continue
            improvement = (
                comparator_metric - model_metric
                if direction == "lower_is_better"
                else model_metric - comparator_metric
            )
            if improvement > margin + tolerance:
                comparison_status = "passed"
            elif improvement < margin - tolerance:
                comparison_status = "failed"
            else:
                comparison_status = "inconclusive"
            comparisons.append({
                "comparator_id": comparator_id,
                "role": declaration.get("role"),
                "required": declaration.get("required") is True,
                "status": comparison_status,
                "model_metric": model_metric,
                "comparator_metric": comparator_metric,
                "absolute_improvement": improvement,
                "required_margin": margin,
                "tolerance": tolerance,
            })

    result["comparisons"] = comparisons
    if invalid_evaluation:
        result["evaluation_validity"] = _status("invalid", invalid_evaluation)
        result["skill"] = _status("undetermined", [_reason(
            "evaluation_invalid", "incoherent or leaking comparator evidence invalidates evaluation",
        )])
        result["presentation_headline"] = (
            "Demo skill: UNDETERMINED — comparator evaluation is invalid."
        )
        return result
    result["evaluation_validity"] = _status("valid", receipt_reason_rows)
    if evidence_errors:
        result["skill"] = _status("undetermined", evidence_errors)
        return result

    required = [row for row in comparisons if row["required"]]
    if not required:
        result["skill"] = _status("not_applicable", [_reason(
            "required_comparator_absent",
            "the family contract declares no required comparator",
        )])
    elif any(row["status"] == "failed" for row in required):
        failed = [str(row["comparator_id"]) for row in required
                  if row["status"] == "failed"]
        result["skill"] = _status("not_demonstrated", [_reason(
            "required_comparator_not_beaten",
            "model did not beat required comparator(s): " + ", ".join(failed),
        )])
        result["presentation_headline"] = (
            "Demo skill: NOT DEMONSTRATED — required comparator not beaten."
        )
    elif any(row["status"] == "inconclusive" for row in required):
        result["skill"] = _status("undetermined", [_reason(
            "comparison_within_tolerance",
            "at least one required comparison is equal or inside tolerance",
        )])
        result["presentation_headline"] = (
            "Demo skill: UNDETERMINED — a required comparison is inconclusive."
        )
    elif all(row["status"] == "passed" for row in required):
        result["skill"] = _status("demonstrated")
        result["presentation_headline"] = (
            "Demo skill: DEMONSTRATED — every required comparator was beaten."
        )
    return result
