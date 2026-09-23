"""Pure calibration-context normalization and observer dispatch.

R2C-091 closes the type-erasure bug where one free
``assumes_data_scale`` string selected every scale observer.  New specs carry
one typed ``calibration_context``; legacy specs receive one closed exact-label
mapping for compatibility.  Unknown text is never guessed from substrings and
is preserved verbatim as ``other`` evidence.

This module is deliberately stdlib-only because the same dispatcher is used
by live Stage 2x, Stage 5 probes, and the vendored portable harness.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import asdict, dataclass


FEATURE_MAGNITUDE_OBSERVER = "feature_magnitude"
TARGET_BOX_GRID_OBSERVER = "target_box_grid"

_FEATURE_EXPECTED = {
    "raw_pixel_unnormalized": "raw",
    "pixel_zero_one": "zero_one",
    "pixel_centered": "centered",
    "standardized": "centered",
}
_FEATURE_SCALES = frozenset({*_FEATURE_EXPECTED, "unit_norm"})


@dataclass(frozen=True)
class CalibrationDispatch:
    """Normalized context plus the only observer allowed to measure it."""

    kind: str
    source: str
    observer: str | None
    expected_observation: str | None
    scale: str | None = None
    convention: str | None = None
    label: str | None = None
    reason: str | None = None

    def as_dict(self) -> dict[str, str | None]:
        return asdict(self)


def _feature(scale: str, *, source: str) -> CalibrationDispatch:
    observer = (
        FEATURE_MAGNITUDE_OBSERVER if scale != "unit_norm" else None
    )
    reason = (
        None if observer else
        "unit_norm has no supported magnitude-range observer in calibration v1"
    )
    return CalibrationDispatch(
        kind="feature_magnitude",
        source=source,
        observer=observer,
        expected_observation=_FEATURE_EXPECTED.get(scale),
        scale=scale,
        reason=reason,
    )


def _target_box_grid(*, source: str) -> CalibrationDispatch:
    return CalibrationDispatch(
        kind="representation_convention",
        source=source,
        observer=TARGET_BOX_GRID_OBSERVER,
        expected_observation="grid_coordinates",
        convention="target_box_grid",
    )


def _other(*, source: str, label: str, reason: str) -> CalibrationDispatch:
    return CalibrationDispatch(
        kind="other",
        source=source,
        observer=None,
        expected_observation=None,
        label=label,
        reason=reason,
    )


def _legacy_token(label: str) -> str:
    """Normalize separators only; semantic substring matching is forbidden."""
    token = re.sub(r"[^a-z0-9]+", "_", label.strip().casefold()).strip("_")
    return re.sub(r"_+", "_", token)


_LEGACY_FEATURE_SCALES = {
    "raw_0_255": "raw_pixel_unnormalized",
    "raw_pixel": "raw_pixel_unnormalized",
    "raw_pixels": "raw_pixel_unnormalized",
    "raw_pixel_unnormalized": "raw_pixel_unnormalized",
    "unnormalized": "raw_pixel_unnormalized",
    "0_1_normalized": "pixel_zero_one",
    "min_max": "pixel_zero_one",
    "pixel_zero_one": "pixel_zero_one",
    "zero_one": "pixel_zero_one",
    "zero_to_one": "pixel_zero_one",
    "pixel_centered": "pixel_centered",
    "centered": "pixel_centered",
    "standardized": "standardized",
    "z_score": "standardized",
    "unit_norm": "unit_norm",
}
_LEGACY_TARGET_BOX_GRID = frozenset({
    "target_box_grid",
    "grid_coordinates",
    "bev_grid_coordinates",
    "bev_grid_128x128",
    "bev_grid_cells",
})


def _dispatch_typed(context: object) -> CalibrationDispatch:
    if not isinstance(context, Mapping):
        return _other(
            source="typed",
            label=type(context).__name__,
            reason="calibration_context is not an object",
        )
    kind = context.get("kind")
    if kind == "feature_magnitude":
        scale = context.get("scale")
        if isinstance(scale, str) and scale in _FEATURE_SCALES:
            return _feature(scale, source="typed")
        return _other(
            source="typed",
            label=str(scale),
            reason="feature_magnitude scale is outside the closed v1 vocabulary",
        )
    if kind == "representation_convention":
        convention = context.get("convention")
        if convention == "target_box_grid":
            return _target_box_grid(source="typed")
        return _other(
            source="typed",
            label=str(convention),
            reason="representation convention is outside the closed v1 vocabulary",
        )
    if kind == "other":
        label = context.get("label")
        reason = context.get("reason")
        if isinstance(label, str) and label.strip() \
                and isinstance(reason, str) and reason.strip():
            return _other(
                source="typed",
                label=label,
                reason=reason,
            )
        return _other(
            source="typed",
            label=label if isinstance(label, str) else str(label),
            reason=(
                reason if isinstance(reason, str) and reason
                else "typed other context requires a nonblank label and reason"
            ),
        )
    return _other(
        source="typed",
        label=str(kind),
        reason=f"unsupported typed calibration_context kind {kind!r}",
    )


def _dispatch_legacy(value: object) -> CalibrationDispatch:
    if not isinstance(value, str) or not value.strip():
        return _other(
            source="legacy",
            label=value if isinstance(value, str) else str(value),
            reason="legacy assumes_data_scale is missing or non-text",
        )
    exact = value.strip().casefold()
    if exact == "[-1,1]":
        return _feature("pixel_centered", source="legacy")
    token = _legacy_token(value)
    scale = _LEGACY_FEATURE_SCALES.get(token)
    if scale is not None:
        return _feature(scale, source="legacy")
    if token in _LEGACY_TARGET_BOX_GRID:
        return _target_box_grid(source="legacy")
    return _other(
        source="legacy",
        label=value,
        reason="legacy assumes_data_scale has no supported calibration observer",
    )


def dispatch_calibration_context(
    entry: Mapping[str, object],
) -> CalibrationDispatch:
    """Normalize one typed-or-legacy scale-dependent parameter entry.

    Mixed typed and legacy authority is invalid upstream and remains visibly
    unprobeable here.  No graph-statistic, temporal, physical-unit, or
    scale-free label is inferred into a supported observer.
    """
    # Pydantic dumps may retain the inactive optional carrier as explicit
    # null.  Authority is present only when the carrier has a value.
    has_typed = entry.get("calibration_context") is not None
    has_legacy = entry.get("assumes_data_scale") is not None
    if has_typed and has_legacy:
        return _other(
            source="conflict",
            label="calibration_context + assumes_data_scale",
            reason="typed and legacy calibration contexts cannot both be set",
        )
    if has_typed:
        return _dispatch_typed(entry.get("calibration_context"))
    if has_legacy:
        return _dispatch_legacy(entry.get("assumes_data_scale"))
    return _other(
        source="missing",
        label="",
        reason="no calibration_context or legacy assumes_data_scale declared",
    )


__all__ = [
    "CalibrationDispatch",
    "FEATURE_MAGNITUDE_OBSERVER",
    "TARGET_BOX_GRID_OBSERVER",
    "dispatch_calibration_context",
]
