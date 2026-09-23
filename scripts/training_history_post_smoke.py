"""Pipeline-owned post-smoke persistence for structured training history.

The executed notebook may emit one candidate history produced by the fixed
family helper.  It does not own the durable artifact.  This boundary accepts
exactly one canonical, line-anchored JSON marker and writes only
``.pipeline/training_history.json`` after the candidate has been validated and
rebound to pipeline-owned split, scaling-state, and evaluated-model evidence.

The schema-specific rebinding lives below the extraction and atomic-write
primitives so malformed output can never become a partially written artifact.
"""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path

from scripts.time_series_training_history import (
    TrainingHistoryCoverageError,
    TrainingHistoryError,
    TrainingHistoryProducerError,
    TrainingHistoryUpstreamError,
    validate_training_history,
    validate_training_history_lineage,
)


TRAINING_HISTORY_MARKER = "R2C_TRAINING_HISTORY_JSON: "
TRAINING_HISTORY_ARTIFACT = Path(".pipeline/training_history.json")
TrainingHistoryPostSmokeError = TrainingHistoryError


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
        raise TrainingHistoryProducerError(
            "training_history_marker_not_json",
            f"training-history marker is not canonical JSON: {exc}",
        ) from exc


def extract_training_history(output_text: str) -> dict[str, object]:
    """Extract exactly one line-anchored canonical history marker."""
    if not isinstance(output_text, str):
        raise TrainingHistoryProducerError(
            "training_history_marker_output_malformed",
            "executed notebook output for training history must be text",
        )
    marker_lines = [
        line
        for line in output_text.splitlines()
        if line.startswith(TRAINING_HISTORY_MARKER)
    ]
    if len(marker_lines) != 1:
        raise TrainingHistoryProducerError(
            "training_history_marker_count",
            "executed notebook output must contain exactly one line beginning "
            f"{TRAINING_HISTORY_MARKER!r}; found {len(marker_lines)}",
        )

    payload_text = marker_lines[0][len(TRAINING_HISTORY_MARKER):]
    try:
        payload = json.loads(payload_text)
    except (json.JSONDecodeError, ValueError) as exc:
        raise TrainingHistoryProducerError(
            "training_history_marker_not_json",
            f"training-history marker payload is not JSON: {exc}",
        ) from exc
    if not isinstance(payload, dict):
        raise TrainingHistoryProducerError(
            "training_history_marker_shape",
            "training-history marker payload must be a JSON object",
        )
    if payload_text != _canonical_json(payload):
        raise TrainingHistoryProducerError(
            "training_history_marker_not_canonical",
            "training-history marker payload must use canonical JSON with "
            "sorted keys and no insignificant whitespace",
        )
    return payload


def _atomic_write_json(path: Path, payload: Mapping[str, object]) -> None:
    """Write one validated record without exposing a partial destination."""
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


def _evaluated_identity(
    executed_evaluation: Mapping[str, object] | None,
) -> tuple[str, str, str]:
    """Return the exact evaluated model, checkpoint, and configuration ids."""
    if not isinstance(executed_evaluation, Mapping):
        raise TrainingHistoryProducerError(
            "training_history_evaluation_missing",
            "recorded training history requires one structured executed "
            "evaluation record",
        )
    values: list[str] = []
    for field in ("model_id", "checkpoint_id", "config_id"):
        value = executed_evaluation.get(field)
        if not isinstance(value, str) or not value or value != value.strip():
            raise TrainingHistoryProducerError(
                "training_history_evaluation_identity_missing",
                "executed evaluation must carry exact non-blank model_id, "
                "checkpoint_id, and config_id",
            )
        values.append(value)
    return values[0], values[1], values[2]


def validate_post_smoke_training_history(
    output_text: str,
    *,
    split_receipt: Mapping[str, object] | None,
    target_scaling_state: Mapping[str, object] | None,
    executed_evaluation: Mapping[str, object] | None,
    require_recorded: bool = False,
) -> dict[str, object]:
    """Validate and rebind one marker without mutating the run directory."""
    candidate = extract_training_history(output_text)
    normalized = validate_training_history(candidate)
    if normalized["status"] == "not_applicable":
        if require_recorded:
            raise TrainingHistoryProducerError(
                "training_history_not_applicable_disagreement",
                "the declared execution plan has a training phase, so its "
                "history cannot be not_applicable",
            )
        return normalized
    if not isinstance(target_scaling_state, Mapping):
        raise TrainingHistoryUpstreamError(
            "target_scaling_state_missing",
            "recorded training history requires the pipeline-validated "
            "R2C-089 target-scaling state",
        )

    # The core owns all range and scaling-state grammar.  This adapter supplies
    # only independently produced authorities and never normalizes a range or
    # reconstructs scaling identity from notebook prose.
    normalized = validate_training_history_lineage(
        normalized,
        split_receipt,
        target_scaling_state=target_scaling_state,
    )
    evaluated_model, evaluated_checkpoint, evaluated_config = (
        _evaluated_identity(executed_evaluation)
    )
    disagreements = {
        field: {"history": normalized[field], "evaluated": evaluated}
        for field, evaluated in (
            ("model_id", evaluated_model),
            ("checkpoint_id", evaluated_checkpoint),
            ("config_id", evaluated_config),
        )
        if normalized[field] != evaluated
    }
    if disagreements:
        raise TrainingHistoryProducerError(
            "training_history_evaluation_identity_disagreement",
            "training history does not identify the evaluated model, "
            f"checkpoint, and configuration: {disagreements!r}",
        )
    return normalized


def persist_post_smoke_training_history(
    run_dir: Path,
    output_text: str,
    *,
    split_receipt: Mapping[str, object] | None,
    target_scaling_state: Mapping[str, object] | None,
    executed_evaluation: Mapping[str, object] | None,
    require_recorded: bool = False,
) -> dict[str, object]:
    """Validate all authorities, then atomically persist the pipeline record."""
    normalized = validate_post_smoke_training_history(
        output_text,
        split_receipt=split_receipt,
        target_scaling_state=target_scaling_state,
        executed_evaluation=executed_evaluation,
        require_recorded=require_recorded,
    )
    _atomic_write_json(Path(run_dir) / TRAINING_HISTORY_ARTIFACT, normalized)
    return normalized


__all__ = [
    "TRAINING_HISTORY_ARTIFACT",
    "TRAINING_HISTORY_MARKER",
    "TrainingHistoryCoverageError",
    "TrainingHistoryPostSmokeError",
    "TrainingHistoryProducerError",
    "TrainingHistoryUpstreamError",
    "extract_training_history",
    "persist_post_smoke_training_history",
    "validate_post_smoke_training_history",
]
