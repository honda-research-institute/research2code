"""Pipeline-owned post-smoke evidence for forecasting target scaling.

The generated notebook may *report* the state fitted by the fixed target
scaling helper.  It never owns the durable artifact.  This module is the
small pipeline boundary that:

* accepts exactly one canonical JSON marker from executed output;
* rebinds that candidate to the closed family contract and the independently
  derived ``eval_split_lineage`` receipt;
* writes only ``.pipeline/target_scaling_state.json``, by atomic replacement;
  and
* proves that typed scaled actuals and predictions invert to the values the
  notebook exposes in original units.

The functions do not discover runs, mutate generated source/notebooks, or
infer entity order, fitting spans, output roles, or axes.  They preserve the
typed ownership carried by :mod:`scripts.time_series_target_scaling`:
unsupported grammar is pipeline-owned and retry-free, while disagreement
inside the supported grammar is producer-owned.
"""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from scripts.time_series_target_scaling import (
    TargetScalingCoverageError,
    TargetScalingProducerError,
    inverse_forecast_output,
    validate_target_scaling_contract,
    validate_training_only_state,
)


STATE_MARKER = "R2C_TARGET_SCALING_STATE_JSON: "
STATE_ARTIFACT = Path(".pipeline/target_scaling_state.json")
ORIGINAL_UNIT_PROOF_SCHEMA_VERSION = "1.0.0"
ORIGINAL_UNIT_RTOL = 1.0e-6
ORIGINAL_UNIT_ATOL = 1.0e-8


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
            "target_scaling_marker_not_json",
            f"target scaling marker is not canonical JSON: {exc}",
        ) from exc


def extract_target_scaling_state(output_text: str) -> dict[str, object]:
    """Extract exactly one line-anchored canonical target-scaling marker."""
    if not isinstance(output_text, str):
        raise TargetScalingProducerError(
            "target_scaling_marker_output_malformed",
            "executed notebook output for target scaling must be text",
        )
    marker_lines = [
        line for line in output_text.splitlines()
        if line.startswith(STATE_MARKER)
    ]
    if len(marker_lines) != 1:
        raise TargetScalingProducerError(
            "target_scaling_marker_count",
            "executed notebook output must contain exactly one line beginning "
            f"{STATE_MARKER!r}; found {len(marker_lines)}",
        )

    payload_text = marker_lines[0][len(STATE_MARKER):]
    try:
        payload = json.loads(payload_text)
    except (json.JSONDecodeError, ValueError) as exc:
        raise TargetScalingProducerError(
            "target_scaling_marker_not_json",
            f"target scaling marker payload is not JSON: {exc}",
        ) from exc
    if not isinstance(payload, dict):
        raise TargetScalingProducerError(
            "target_scaling_marker_shape",
            "target scaling marker payload must be a JSON object",
        )
    if payload_text != _canonical_json(payload):
        raise TargetScalingProducerError(
            "target_scaling_marker_not_canonical",
            "target scaling marker payload must use canonical JSON with sorted "
            "keys and no insignificant whitespace",
        )
    return payload


def _closed_contract(build_plan: Mapping[str, object]) -> dict[str, object]:
    if not isinstance(build_plan, Mapping):
        raise TargetScalingCoverageError(
            "target_scaling_build_plan_unsupported",
            "target scaling post-smoke validation requires a build-plan mapping",
        )
    contract = build_plan.get("target_scaling")
    if not isinstance(contract, Mapping):
        raise TargetScalingCoverageError(
            "target_scaling_contract_unsupported",
            "build_plan.target_scaling must be a mapping",
        )
    normalized = validate_target_scaling_contract(contract)
    if normalized["state_artifact"] != STATE_ARTIFACT.as_posix():
        raise TargetScalingCoverageError(
            "target_scaling_artifact_path_unsupported",
            "target scaling post-smoke v1 writes only "
            f"{STATE_ARTIFACT.as_posix()!r}; build plan declares "
            f"{normalized['state_artifact']!r}",
        )
    return normalized


def _validate_state_contract_projection(
    state: Mapping[str, object],
    contract: Mapping[str, object],
) -> None:
    """Close the state-to-contract projection beyond the digest assertion.

    ``validate_training_only_state`` authenticates the canonical contract
    digest and fitting carrier.  The explicit projection also prevents a
    candidate from retaining that digest while self-minting changed policy
    fields inside the state object.
    """
    projected_fields = (
        "mode",
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
    )
    disagreements = {
        field: {"state": state.get(field), "contract": contract.get(field)}
        for field in projected_fields
        if state.get(field) != contract.get(field)
    }
    if disagreements:
        raise TargetScalingProducerError(
            "target_scaling_contract_disagreement",
            "target scaling state policy fields disagree with the closed "
            f"build-plan contract: {disagreements!r}",
        )


def validate_post_smoke_target_scaling_state(
    output_text: str,
    *,
    build_plan: Mapping[str, object],
    split_receipt: Mapping[str, object] | None,
) -> dict[str, object]:
    """Validate marker evidence without mutating the run directory."""
    contract = _closed_contract(build_plan)
    candidate = extract_target_scaling_state(output_text)
    normalized = validate_training_only_state(
        candidate,
        split_receipt,
        contract=contract,
    )
    _validate_state_contract_projection(normalized, contract)
    return normalized


def _atomic_write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        dir=str(path.parent),
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def persist_post_smoke_target_scaling_state(
    run_dir: Path,
    output_text: str,
    *,
    build_plan: Mapping[str, object],
    split_receipt: Mapping[str, object] | None,
) -> dict[str, object]:
    """Validate evidence, then atomically persist the pipeline-owned state."""
    normalized = validate_post_smoke_target_scaling_state(
        output_text,
        build_plan=build_plan,
        split_receipt=split_receipt,
    )
    _atomic_write_json(Path(run_dir) / STATE_ARTIFACT, normalized)
    return normalized


def _comparison_array(value: object, *, root: str) -> np.ndarray:
    try:
        if hasattr(value, "detach") and hasattr(value, "cpu"):
            value = value.detach().cpu().numpy()
        elif isinstance(value, np.ndarray):
            pass
        elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            value = np.asarray(value)
        else:
            raise TypeError
        array = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise TargetScalingProducerError(
            "target_output_container_unsupported",
            f"{root} must be a numeric array, tensor, or JSON array",
        ) from exc
    if not array.size or array.dtype.kind not in {"i", "u", "f"}:
        raise TargetScalingProducerError(
            "target_output_dtype_unsupported",
            f"{root} must be a nonempty real numeric array; got dtype {array.dtype}",
        )
    normalized = np.asarray(array, dtype=np.float64)
    if not np.isfinite(normalized).all():
        raise TargetScalingProducerError(
            "target_output_nonfinite",
            f"{root} contains non-finite values and cannot prove original units",
        )
    return normalized


def _exact_axis(value: object, *, root: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TargetScalingProducerError(
            "target_output_entity_axis",
            f"{root} must be an explicitly declared integer entity axis",
        )
    return value


def _compare_one_output(
    *,
    label: str,
    state: Mapping[str, object],
    entity_ids: object,
    entity_id_root: str,
    scaled: object,
    original: object,
    output_role: str,
    entity_axis: object,
) -> dict[str, object]:
    axis = _exact_axis(entity_axis, root=f"{label}.entity_axis")
    if (
        not isinstance(output_role, str)
        or not output_role
        or output_role != output_role.strip()
    ):
        raise TargetScalingProducerError(
            "target_output_role_malformed",
            f"{label}.output_role must be an explicitly declared non-blank string",
        )
    scaled_array = _comparison_array(scaled, root=f"scaled_{label}")
    original_array = _comparison_array(original, root=f"original_{label}")
    inverted = inverse_forecast_output(
        scaled_array,
        entity_ids,
        state,
        output_role=output_role,
        batch_entity_id_root=entity_id_root,
        entity_axis=axis,
    )
    inverted_array = np.asarray(inverted, dtype=np.float64)
    if inverted_array.shape != original_array.shape:
        raise TargetScalingProducerError(
            "target_output_original_unit_shape",
            f"inverted {label} shape {inverted_array.shape!r} disagrees with "
            f"original-facing shape {original_array.shape!r}",
        )
    absolute_error = np.abs(inverted_array - original_array)
    max_absolute_error = float(np.max(absolute_error))
    if not np.allclose(
        inverted_array,
        original_array,
        rtol=ORIGINAL_UNIT_RTOL,
        atol=ORIGINAL_UNIT_ATOL,
        equal_nan=False,
    ):
        raise TargetScalingProducerError(
            "target_output_original_unit_disagreement",
            f"inverted {label} disagrees with original-facing values; "
            f"max_absolute_error={max_absolute_error!r}",
        )
    return {
        "output_role": output_role,
        "entity_axis": axis,
        "shape": list(inverted_array.shape),
        "max_absolute_error": max_absolute_error,
    }


def validate_original_unit_comparison(
    state: Mapping[str, object],
    *,
    entity_ids: object,
    entity_id_root: str,
    scaled_actuals: object,
    original_actuals: object,
    actual_output_role: str,
    actual_entity_axis: int,
    scaled_predictions: object,
    original_predictions: object,
    prediction_output_role: str,
    prediction_entity_axis: int,
) -> dict[str, Any]:
    """Prove scaled actuals and predictions invert to original-facing units.

    Entity ids, output roles, and entity axes are mandatory inputs.  No row
    order or output semantics are inferred.  The existing scaling oracle
    performs the typed identity join and role-specific inversion.
    """
    if (
        not isinstance(entity_id_root, str)
        or not entity_id_root
        or entity_id_root != entity_id_root.strip()
    ):
        raise TargetScalingProducerError(
            "target_output_entity_id_root_malformed",
            "entity_id_root must be an explicitly declared non-blank string",
        )
    actuals = _compare_one_output(
        label="actuals",
        state=state,
        entity_ids=entity_ids,
        entity_id_root=entity_id_root,
        scaled=scaled_actuals,
        original=original_actuals,
        output_role=actual_output_role,
        entity_axis=actual_entity_axis,
    )
    predictions = _compare_one_output(
        label="predictions",
        state=state,
        entity_ids=entity_ids,
        entity_id_root=entity_id_root,
        scaled=scaled_predictions,
        original=original_predictions,
        output_role=prediction_output_role,
        entity_axis=prediction_entity_axis,
    )
    state_digest = state.get("state_digest")
    if not isinstance(state_digest, str):
        # The inversion oracle already validates state shape and digest.  This
        # branch keeps the returned proof typed for static callers.
        raise TargetScalingProducerError(
            "target_scaling_state_digest",
            "target scaling state omits its canonical digest",
        )
    return {
        "schema_version": ORIGINAL_UNIT_PROOF_SCHEMA_VERSION,
        "status": "valid",
        "state_digest": state_digest,
        "entity_id_root": entity_id_root,
        "tolerances": {
            "relative": ORIGINAL_UNIT_RTOL,
            "absolute": ORIGINAL_UNIT_ATOL,
        },
        "actuals": actuals,
        "predictions": predictions,
    }


__all__ = [
    "ORIGINAL_UNIT_ATOL",
    "ORIGINAL_UNIT_PROOF_SCHEMA_VERSION",
    "ORIGINAL_UNIT_RTOL",
    "STATE_ARTIFACT",
    "STATE_MARKER",
    "extract_target_scaling_state",
    "persist_post_smoke_target_scaling_state",
    "validate_original_unit_comparison",
    "validate_post_smoke_target_scaling_state",
]
