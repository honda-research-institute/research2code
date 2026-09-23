"""Pure statistical decisions for the time-series forecasting probe kit.

This module deliberately knows nothing about method-package loading, taxonomy
dispatch, methodology-element binding, or :class:`~probes.ProbeVerdict`.  The
runtime adapter owns those concerns.  Keeping the numerical decisions here
makes the zoo thresholds directly testable and lets the delivery-time portable
harness vendor the exact same calculations.

All assessments are deterministic over their inputs and return strict-JSON
evidence. Unsupported grammar or unavailable pipeline-owned fixtures are
``unprobeable``; malformed outputs under the supported contract are producer
failures. A conditional check that the declared method shape does not require
is ``not_applicable``.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from typing import Any, Mapping

import numpy as np


# Closed output grammar for the probabilistic sampling arm.  Both names carry
# Student-t *scale* semantics; neither is a standard deviation.  The constant
# is intentionally JSON-normalizable so taxonomy/freeze wiring can reuse the
# authority instead of maintaining a second name-guessing table.
SUPPORTED_STUDENT_T_OUTPUT_GRAMMARS = (
    {
        "distribution": "student_t",
        "parameter_keys": ("mu", "scale", "df"),
        "location_key": "mu",
        "scale_key": "scale",
        "scale_semantics": "student_t_scale",
        "degrees_of_freedom_key": "df",
    },
    {
        "distribution": "student_t",
        "parameter_keys": ("mu", "sigma", "df"),
        "location_key": "mu",
        "scale_key": "sigma",
        "scale_semantics": "student_t_scale",
        "degrees_of_freedom_key": "df",
    },
)

MIN_SAMPLE_COUNT = 100
SAMPLE_SPREAD_RATIO_MIN = 0.25
SAMPLE_SPREAD_RATIO_MAX = 4.0
PATH_DEPENDENCE_FAIL_MAX = 0.15
PATH_DEPENDENCE_PASS_MIN = 0.35
OWN_HISTORY_MIN_RELATIVE_RESPONSE = 1e-4
OWN_HISTORY_MIN_ABSOLUTE_RESPONSE = 1e-6
OWN_HISTORY_PERTURBATION_FACTOR = 100.0
MAGNITUDE_MIN_RATIO = 0.01
_NUMERICAL_FLOOR = 1e-12
_INTERVENTION_RTOL = 1e-6
_INTERVENTION_ATOL = 1e-10

ASSESSMENT_STATUSES = (
    "pass",
    "fail",
    "flag_for_researcher",
    "unprobeable",
    "not_applicable",
)

ASSESSMENT_RESPONSIBILITIES = (
    "none",
    "producer",
    "pipeline_coverage",
    "evidence",
)


@dataclass(frozen=True)
class StatisticalAssessment:
    """One pure decision, ready for a runtime wrapper to map to a verdict."""

    status: str
    reason: str
    reason_code: str
    responsibility: str = "none"
    statistics: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.status not in ASSESSMENT_STATUSES:
            raise ValueError(f"unknown statistical assessment {self.status!r}")
        if not isinstance(self.reason_code, str) or not self.reason_code:
            raise ValueError("statistical assessment reason_code must be nonempty")
        if self.responsibility not in ASSESSMENT_RESPONSIBILITIES:
            raise ValueError(
                f"unknown assessment responsibility {self.responsibility!r}"
            )
        # Python's default JSON encoder accepts NaN and Infinity even though
        # neither is JSON.  Probe reports and frozen harness inputs require the
        # strict grammar, so reject them at the shared result boundary.
        json.dumps(self.statistics, allow_nan=False)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _assessment(
    status: str,
    reason_code: str,
    reason: str,
    *,
    responsibility: str = "none",
    **statistics: Any,
) -> StatisticalAssessment:
    return StatisticalAssessment(
        status=status,
        reason=reason,
        reason_code=reason_code,
        responsibility=responsibility,
        statistics=statistics,
    )


def _numeric_array(
    value: object, *, name: str, code_name: str, ndim: int,
) -> tuple[np.ndarray | None, tuple[str, str] | None]:
    """Convert an array-like without importing a framework or guessing rank."""

    candidate = value
    # CPU torch tensors are a supported array carrier, but torch is never a
    # dependency of this pure module.  Detach explicitly so a diagnostic does
    # not become an autograd operation.
    try:
        if hasattr(candidate, "detach"):
            candidate = candidate.detach()
        if hasattr(candidate, "cpu"):
            candidate = candidate.cpu()
        if hasattr(candidate, "numpy"):
            candidate = candidate.numpy()
        raw = np.asarray(candidate)
    except (AttributeError, OverflowError, TypeError, ValueError, RuntimeError) as exc:
        return None, (
            f"{code_name}_conversion_failed",
            f"{name} is not a numeric array: {exc}",
        )
    if raw.dtype.kind not in {"i", "u", "f"}:
        return None, (
            f"{code_name}_non_real_dtype",
            f"{name} must contain real numeric values, got dtype {raw.dtype}",
        )
    try:
        array = raw.astype(np.float64, copy=False)
    except (OverflowError, TypeError, ValueError) as exc:
        return None, (
            f"{code_name}_conversion_failed",
            f"{name} is not a numeric array: {exc}",
        )
    if array.ndim != ndim:
        return None, (
            f"{code_name}_rank_mismatch",
            f"{name} must have rank {ndim}, got shape {tuple(array.shape)}",
        )
    if any(size == 0 for size in array.shape):
        return None, (
            f"{code_name}_empty",
            f"{name} must have no empty dimensions",
        )
    if not np.isfinite(array).all():
        return None, (
            f"{code_name}_nonfinite",
            f"{name} contains a missing or non-finite value",
        )
    return array, None


def _root_mean_square(array: np.ndarray) -> float:
    """Overflow-safe RMS for an already finite real array."""

    maximum = float(np.max(np.abs(array)))
    if maximum == 0.0:
        return 0.0
    scaled = array / maximum
    result = maximum * math.sqrt(float(np.mean(scaled * scaled)))
    return float(result)


def _difference_root_mean_square(left: np.ndarray, right: np.ndarray) -> float:
    """Overflow-safe RMS of ``right - left`` for finite arrays."""

    maximum = max(
        float(np.max(np.abs(left))),
        float(np.max(np.abs(right))),
    )
    if maximum == 0.0:
        return 0.0
    difference = (right / maximum) - (left / maximum)
    scaled_rms = _root_mean_square(difference)
    if scaled_rms > 1.0 \
            and maximum > np.finfo(np.float64).max / scaled_rms:
        return math.inf
    return float(maximum * scaled_rms)


def _mean_absolute(array: np.ndarray, *, axis: int | None = None) -> Any:
    """Overflow-safe mean absolute value, scalar or one value per axis row."""

    absolute = np.abs(array)
    if axis is None:
        maximum = float(np.max(absolute))
        if maximum == 0.0:
            return 0.0
        return float(maximum * float(np.mean(absolute / maximum)))

    maximum = np.max(absolute, axis=axis)
    safe_maximum = np.where(maximum == 0.0, 1.0, maximum)
    expanded = np.expand_dims(safe_maximum, axis=axis)
    means = safe_maximum * np.mean(absolute / expanded, axis=axis)
    return np.where(maximum == 0.0, 0.0, means)


def _stable_sample_std(draws: np.ndarray) -> np.ndarray:
    """Per-cell sample std without overflowing on large finite values."""

    maximum = np.max(np.abs(draws), axis=0)
    safe_maximum = np.where(maximum == 0.0, 1.0, maximum)
    normalized = draws / safe_maximum[None, ...]
    centered = normalized - np.mean(normalized, axis=0, keepdims=True)
    variance = np.sum(centered * centered, axis=0) / (draws.shape[0] - 1)
    return safe_maximum * np.sqrt(variance)


def _stable_correlation(left: np.ndarray, right: np.ndarray) -> float | None:
    """Pearson correlation after scale normalization, or None if undefined."""

    left_scale = float(np.max(np.abs(left)))
    right_scale = float(np.max(np.abs(right)))
    if left_scale == 0.0 or right_scale == 0.0:
        return None
    left_centered = left / left_scale
    right_centered = right / right_scale
    left_centered -= np.mean(left_centered)
    right_centered -= np.mean(right_centered)
    left_norm = _root_mean_square(left_centered)
    right_norm = _root_mean_square(right_centered)
    if left_norm <= _NUMERICAL_FLOOR or right_norm <= _NUMERICAL_FLOOR:
        return None
    correlation = float(np.mean(
        (left_centered / left_norm) * (right_centered / right_norm)
    ))
    if not math.isfinite(correlation):
        return None
    # Roundoff can exceed the mathematical bounds by a few ulps.
    return max(-1.0, min(1.0, correlation))


def _finite_ratio(numerator: float, denominator: float) -> float | None:
    """Return a finite nonnegative ratio; None means it exceeds float range."""

    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        ratio = float(numerator / denominator)
    return ratio if math.isfinite(ratio) else None


def student_t_output_grammar(
    distribution_params: Mapping[str, object] | None,
) -> Mapping[str, object] | None:
    """Return the exact supported grammar, with no fuzzy key fallback."""

    if not isinstance(distribution_params, Mapping):
        return None
    actual = set(distribution_params)
    for grammar in SUPPORTED_STUDENT_T_OUTPUT_GRAMMARS:
        if actual == set(grammar["parameter_keys"]):
            return grammar
    return None


def assess_sample_genuineness(
    samples: object,
    distribution_params: Mapping[str, object] | None,
) -> StatisticalAssessment:
    """Compare sample-axis spread with the declared Student-t variance.

    For ``df > 2``, a Student-t with scale ``s`` has standard deviation
    ``s * sqrt(df / (df - 2))``.  Treating either supported scale key as a
    Gaussian standard deviation would therefore be a semantic error.
    """

    grammar = student_t_output_grammar(distribution_params)
    if grammar is None:
        return _assessment(
            "unprobeable",
            "student_t_output_grammar_unsupported",
            "distribution_params do not match either exact supported "
            "Student-t grammar",
            responsibility="pipeline_coverage",
            observed_parameter_keys=(
                sorted(str(key) for key in distribution_params)
                if isinstance(distribution_params, Mapping) else []
            ),
        )
    assert distribution_params is not None  # narrowed by the grammar match
    location_raw = distribution_params[str(grammar["location_key"])]
    if samples is location_raw:
        return _assessment(
            "fail",
            "samples_alias_conditional_mean",
            "samples alias the conditional-mean object",
            responsibility="producer",
            empirical_to_expected_spread_ratio=0.0,
            scale_key=grammar["scale_key"],
        )

    draws, error = _numeric_array(
        samples, name="samples", code_name="samples", ndim=3,
    )
    if error:
        return _assessment(
            "fail", error[0], error[1], responsibility="producer",
        )
    assert draws is not None
    if draws.shape[0] < MIN_SAMPLE_COUNT:
        return _assessment(
            "unprobeable",
            "sample_count_below_statistical_minimum",
            f"sample-axis size {draws.shape[0]} is below the statistical "
            f"minimum {MIN_SAMPLE_COUNT}",
            responsibility="pipeline_coverage",
            sample_count=int(draws.shape[0]),
        )

    expected_shape = draws.shape[1:]
    arrays: dict[str, np.ndarray] = {}
    for semantic, key in (
        ("location", str(grammar["location_key"])),
        ("scale", str(grammar["scale_key"])),
        ("degrees of freedom", str(grammar["degrees_of_freedom_key"])),
    ):
        array, error = _numeric_array(
            distribution_params[key],
            name=f"Student-t {semantic}",
            code_name="student_t_" + semantic.replace(" ", "_"),
            ndim=2,
        )
        if error:
            return _assessment(
                "fail", error[0], error[1], responsibility="producer",
            )
        assert array is not None
        if array.shape != expected_shape:
            return _assessment(
                "fail",
                "student_t_parameter_shape_mismatch",
                f"Student-t {semantic} shape {tuple(array.shape)} does not "
                f"match sample output shape {tuple(expected_shape)}",
                responsibility="producer",
            )
        arrays[semantic] = array

    scale = arrays["scale"]
    df = arrays["degrees of freedom"]
    if np.any(scale <= 0.0):
        return _assessment(
            "fail",
            "student_t_scale_nonpositive",
            "Student-t scale must be positive",
            responsibility="producer",
        )
    if np.any(df <= 2.0):
        return _assessment(
            "unprobeable",
            "student_t_variance_undefined",
            "Student-t variance is undefined when any declared df is <= 2",
            responsibility="pipeline_coverage",
            minimum_df=float(np.min(df)),
        )

    empirical_std = _stable_sample_std(draws)
    variance_factor = np.sqrt(df / (df - 2.0))
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        ratios = (empirical_std / scale) / variance_factor
    if not np.isfinite(ratios).all():
        return _assessment(
            "fail",
            "sample_spread_ratio_nonfinite",
            "forecast sample spread cannot be represented relative to the "
            "declared positive Student-t scale",
            responsibility="producer",
            sample_count=int(draws.shape[0]),
            nonfinite_ratio_count=int(np.count_nonzero(~np.isfinite(ratios))),
            scale_key=grammar["scale_key"],
            scale_semantics=grammar["scale_semantics"],
        )
    median_ratio = float(np.median(ratios))
    statistics = {
        "sample_count": int(draws.shape[0]),
        "median_empirical_to_expected_spread_ratio": median_ratio,
        "minimum_empirical_to_expected_spread_ratio": float(np.min(ratios)),
        "maximum_empirical_to_expected_spread_ratio": float(np.max(ratios)),
        "scale_key": grammar["scale_key"],
        "scale_semantics": grammar["scale_semantics"],
    }
    if median_ratio < SAMPLE_SPREAD_RATIO_MIN:
        return _assessment(
            "fail",
            "sample_spread_too_small",
            "forecast samples have too little spread for the declared Student-t scale",
            responsibility="producer",
            **statistics,
        )
    if median_ratio > SAMPLE_SPREAD_RATIO_MAX:
        return _assessment(
            "fail",
            "sample_spread_too_large",
            "forecast samples have too much spread for the declared Student-t scale",
            responsibility="producer",
            **statistics,
        )
    return _assessment(
        "pass",
        "sample_spread_consistent",
        "sample spread is consistent with the declared Student-t scale semantics",
        **statistics,
    )


def assess_autoregressive_path_dependence(
    samples: object,
    *,
    applicability_status: str,
    sample_genuineness_status: str,
) -> StatisticalAssessment:
    """Measure consecutive-step residual dependence across sample paths."""

    if applicability_status == "not_applicable":
        return _assessment(
            "not_applicable",
            "autoregressive_path_not_declared",
            "the declared forecasting contract is not autoregressive",
        )
    if applicability_status == "unresolved":
        return _assessment(
            "unprobeable",
            "autoregressive_applicability_unresolved",
            "the exact methodology contract does not resolve whether "
            "autoregressive path dependence is required",
            responsibility="pipeline_coverage",
        )
    if applicability_status != "required":
        return _assessment(
            "unprobeable",
            "autoregressive_applicability_status_unknown",
            "autoregressive applicability must be one of required, "
            "not_applicable, or unresolved",
            responsibility="pipeline_coverage",
            observed_applicability_status=str(applicability_status),
        )
    if sample_genuineness_status != "pass":
        return _assessment(
            "unprobeable",
            "sample_genuineness_prerequisite_not_passed",
            "path dependence requires a passing sample-genuineness assessment",
            responsibility="evidence",
            sample_genuineness_status=sample_genuineness_status,
        )
    draws, error = _numeric_array(
        samples, name="samples", code_name="samples", ndim=3,
    )
    if error:
        return _assessment(
            "fail", error[0], error[1], responsibility="producer",
        )
    assert draws is not None
    if draws.shape[0] < MIN_SAMPLE_COUNT or draws.shape[2] < 2:
        return _assessment(
            "unprobeable",
            "path_dependence_evidence_insufficient",
            f"path dependence requires at least {MIN_SAMPLE_COUNT} samples "
            "and two horizon steps",
            responsibility="pipeline_coverage",
            sample_count=int(draws.shape[0]),
            horizon=int(draws.shape[2]),
        )

    correlations: list[float] = []
    for series in range(draws.shape[1]):
        for step in range(draws.shape[2] - 1):
            correlation = _stable_correlation(
                draws[:, series, step], draws[:, series, step + 1]
            )
            if correlation is not None:
                correlations.append(abs(correlation))
    if not correlations:
        return _assessment(
            "unprobeable",
            "path_correlation_undefined",
            "consecutive-step residual correlation is undefined",
            responsibility="pipeline_coverage",
        )

    median_abs = float(np.median(correlations))
    statistics = {
        "median_absolute_consecutive_residual_correlation": median_abs,
        "correlation_count": len(correlations),
        "fail_max": PATH_DEPENDENCE_FAIL_MAX,
        "pass_min": PATH_DEPENDENCE_PASS_MIN,
    }
    if median_abs <= PATH_DEPENDENCE_FAIL_MAX:
        return _assessment(
            "fail",
            "autoregressive_path_dependence_absent",
            "sampled horizon steps behave like independent marginals",
            responsibility="producer",
            **statistics,
        )
    if median_abs < PATH_DEPENDENCE_PASS_MIN:
        return _assessment(
            "flag_for_researcher",
            "autoregressive_path_dependence_borderline",
            "path-dependence signature falls in the zoo-calibrated border band",
            responsibility="evidence",
            **statistics,
        )
    return _assessment(
        "pass",
        "autoregressive_path_dependence_present",
        "sampled trajectories carry consecutive-step path dependence",
        **statistics,
    )


def assess_own_history_sensitivity(
    baseline_history: object,
    perturbed_history: object,
    baseline_forecast: object,
    perturbed_forecast: object,
    *,
    series_index: int,
) -> StatisticalAssessment:
    """Compare one controlled own-history perturbation at a fixed seed."""

    arrays: dict[str, np.ndarray] = {}
    for name, value, producer_owned in (
        ("baseline_history", baseline_history, False),
        ("perturbed_history", perturbed_history, False),
        ("baseline_forecast", baseline_forecast, True),
        ("perturbed_forecast", perturbed_forecast, True),
    ):
        array, error = _numeric_array(
            value, name=name, code_name=name, ndim=2,
        )
        if error:
            return _assessment(
                "fail" if producer_owned else "unprobeable",
                error[0],
                error[1],
                responsibility="producer" if producer_owned else "pipeline_coverage",
            )
        assert array is not None
        arrays[name] = array
    if arrays["baseline_history"].shape != arrays["perturbed_history"].shape:
        return _assessment(
            "unprobeable",
            "history_comparison_shape_mismatch",
            "history comparison shapes disagree",
            responsibility="pipeline_coverage",
        )
    if arrays["baseline_forecast"].shape != arrays["perturbed_forecast"].shape:
        return _assessment(
            "fail",
            "forecast_comparison_shape_mismatch",
            "forecast comparison shapes disagree",
            responsibility="producer",
        )
    if arrays["baseline_history"].shape[0] != arrays["baseline_forecast"].shape[0]:
        return _assessment(
            "fail",
            "forecast_history_entity_axis_mismatch",
            "history and forecast stable-entity axes disagree",
            responsibility="producer",
        )
    if isinstance(series_index, bool) or not isinstance(series_index, int) \
            or not 0 <= series_index < arrays["baseline_history"].shape[0]:
        return _assessment(
            "unprobeable",
            "series_index_out_of_bounds",
            "series_index is outside the entity axis",
            responsibility="pipeline_coverage",
        )

    baseline_selected = arrays["baseline_history"][series_index]
    with np.errstate(over="ignore", invalid="ignore"):
        expected_perturbation = (
            baseline_selected * OWN_HISTORY_PERTURBATION_FACTOR
        )
    if not np.isfinite(expected_perturbation).all():
        return _assessment(
            "unprobeable",
            "history_intervention_not_representable",
            "the exact 100x own-history intervention is not finite in the "
            "declared numeric carrier",
            responsibility="pipeline_coverage",
        )
    if not np.allclose(
        arrays["perturbed_history"][series_index],
        expected_perturbation,
        rtol=_INTERVENTION_RTOL,
        atol=_INTERVENTION_ATOL,
    ):
        return _assessment(
            "unprobeable",
            "history_intervention_factor_mismatch",
            "the selected series history is not the declared exact 100x "
            "intervention",
            responsibility="pipeline_coverage",
            expected_perturbation_factor=OWN_HISTORY_PERTURBATION_FACTOR,
        )
    control_indices = [
        index for index in range(arrays["baseline_history"].shape[0])
        if index != series_index
    ]
    if control_indices and not np.array_equal(
        arrays["baseline_history"][control_indices],
        arrays["perturbed_history"][control_indices],
    ):
        return _assessment(
            "unprobeable",
            "history_intervention_not_isolated",
            "the comparison changes histories outside the selected series",
            responsibility="pipeline_coverage",
        )

    selected_history_rms = _difference_root_mean_square(
        baseline_selected, arrays["perturbed_history"][series_index]
    )
    if not math.isfinite(selected_history_rms):
        return _assessment(
            "unprobeable",
            "history_intervention_statistic_nonfinite",
            "the selected history intervention magnitude cannot be represented",
            responsibility="pipeline_coverage",
        )
    if selected_history_rms <= _NUMERICAL_FLOOR:
        return _assessment(
            "unprobeable",
            "history_intervention_zero",
            "the selected series history was not perturbed",
            responsibility="pipeline_coverage",
        )

    baseline = arrays["baseline_forecast"][series_index]
    perturbed = arrays["perturbed_forecast"][series_index]
    response_rms = _difference_root_mean_square(baseline, perturbed)
    reference_rms = max(
        _root_mean_square(baseline),
        _root_mean_square(perturbed),
    )
    if not math.isfinite(response_rms) or not math.isfinite(reference_rms):
        return _assessment(
            "unprobeable",
            "forecast_response_statistic_nonfinite",
            "the own-history forecast response cannot be represented safely",
            responsibility="pipeline_coverage",
        )
    relative_response = (
        response_rms / reference_rms if reference_rms > 0.0 else 0.0
    )
    required_response = max(
        OWN_HISTORY_MIN_ABSOLUTE_RESPONSE,
        OWN_HISTORY_MIN_RELATIVE_RESPONSE * reference_rms,
    )
    statistics = {
        "series_index": series_index,
        "perturbation_factor": OWN_HISTORY_PERTURBATION_FACTOR,
        "selected_history_rms_change": selected_history_rms,
        "forecast_response_rms": response_rms,
        "relative_forecast_response": relative_response,
        "minimum_absolute_response": OWN_HISTORY_MIN_ABSOLUTE_RESPONSE,
        "minimum_relative_response": OWN_HISTORY_MIN_RELATIVE_RESPONSE,
        "required_forecast_response": required_response,
    }
    if response_rms <= required_response:
        return _assessment(
            "fail",
            "own_history_response_below_tolerance",
            "the selected series forecast is insensitive to its own changed history",
            responsibility="producer",
            **statistics,
        )
    return _assessment(
        "pass",
        "own_history_response_present",
        "the selected series forecast responds to its own changed history",
        **statistics,
    )


def assess_held_out_skill(
    family_contract: Mapping[str, object] | None,
    executed_record: Mapping[str, object] | None,
    validity_receipt: Mapping[str, object] | None,
) -> StatisticalAssessment:
    """Map the shared R2C-086 evidence result without redoing its policy."""

    # Imported lazily so importing the numerical core never loads taxonomy or
    # the pipeline.  This helper is stdlib-only and is vendored beside the
    # forecasting module by the eventual portable-harness wiring.
    from demo_skill_evidence import derive_demo_skill_evidence  # noqa: PLC0415

    result = derive_demo_skill_evidence(
        family_contract, executed_record, validity_receipt
    )
    evaluation_status = str(result.get("evaluation_validity", {}).get("status"))
    skill_status = str(result.get("skill", {}).get("status"))
    raw_role = result.get("evaluation_protocol_role")
    evaluation_protocol_role = (
        json.loads(json.dumps(raw_role, allow_nan=False))
        if isinstance(raw_role, Mapping) else None
    )
    statistics = {
        "evaluation_validity_status": evaluation_status,
        "skill_status": skill_status,
        "comparisons": result.get("comparisons", []),
        "evaluation_protocol_role": evaluation_protocol_role,
    }
    if evaluation_status == "not_applicable" or skill_status == "not_applicable":
        return _assessment(
            "not_applicable",
            "held_out_skill_not_applicable",
            "the family declares no applicable held-out skill comparison",
            **statistics,
        )
    if evaluation_status == "invalid":
        return _assessment(
            "unprobeable",
            "held_out_evaluation_invalid",
            "evaluation-invalid evidence cannot be reused as a duplicate split diagnosis",
            responsibility="evidence",
            **statistics,
        )
    if evaluation_status != "valid":
        return _assessment(
            "unprobeable",
            "held_out_evaluation_unresolved",
            "held-out evaluation validity is unresolved",
            responsibility="evidence",
            **statistics,
        )
    if skill_status == "demonstrated":
        return _assessment(
            "pass",
            "held_out_skill_demonstrated",
            "the model beats every required held-out comparator",
            **statistics,
        )
    if skill_status == "not_demonstrated":
        return _assessment(
            "fail",
            "held_out_skill_not_demonstrated",
            "the model does not beat every required held-out comparator",
            responsibility="evidence",
            **statistics,
        )
    return _assessment(
        "unprobeable",
        "held_out_skill_undetermined",
        "held-out skill is undetermined under the shared comparison policy",
        responsibility="evidence",
        **statistics,
    )


def assess_magnitude_collapse(
    forecast: object,
    eligible_history_tail: object,
) -> StatisticalAssessment:
    """Compare forecast magnitude with a nonzero eligible history tail."""

    predicted, error = _numeric_array(
        forecast, name="forecast", code_name="forecast", ndim=2,
    )
    if error:
        return _assessment(
            "fail", error[0], error[1], responsibility="producer",
        )
    history, error = _numeric_array(
        eligible_history_tail,
        name="eligible_history_tail",
        code_name="eligible_history_tail",
        ndim=2,
    )
    if error:
        return _assessment(
            "unprobeable", error[0], error[1],
            responsibility="pipeline_coverage",
        )
    assert predicted is not None and history is not None
    if predicted.shape[0] != history.shape[0]:
        return _assessment(
            "fail",
            "forecast_history_entity_axis_mismatch",
            "forecast and history stable-entity axes disagree",
            responsibility="producer",
        )

    history_by_entity = np.asarray(_mean_absolute(history, axis=1))
    forecast_by_entity = np.asarray(_mean_absolute(predicted, axis=1))
    eligible_indices = [
        int(index) for index, magnitude in enumerate(history_by_entity)
        if float(magnitude) > _NUMERICAL_FLOOR
    ]
    if not eligible_indices:
        return _assessment(
            "unprobeable",
            "eligible_history_has_no_nonzero_scale",
            "the eligible history tail has no nonzero scale to compare",
            responsibility="pipeline_coverage",
            mean_absolute_history=0.0,
        )

    per_entity_ratios: list[float | None] = []
    collapsed_indices: list[int] = []
    overflow_ratio_indices: list[int] = []
    for index in range(history.shape[0]):
        history_magnitude = float(history_by_entity[index])
        forecast_magnitude = float(forecast_by_entity[index])
        if index not in eligible_indices:
            per_entity_ratios.append(None)
            continue
        ratio = _finite_ratio(forecast_magnitude, history_magnitude)
        per_entity_ratios.append(ratio)
        if ratio is None:
            overflow_ratio_indices.append(index)
        if forecast_magnitude <= MAGNITUDE_MIN_RATIO * history_magnitude:
            collapsed_indices.append(index)

    history_magnitude = float(_mean_absolute(history))
    forecast_magnitude = float(_mean_absolute(predicted))
    global_ratio = _finite_ratio(forecast_magnitude, history_magnitude)
    finite_eligible_ratios = [
        ratio for index, ratio in enumerate(per_entity_ratios)
        if index in eligible_indices and ratio is not None
    ]
    statistics = {
        "mean_absolute_forecast": forecast_magnitude,
        "mean_absolute_history": history_magnitude,
        "forecast_to_history_magnitude_ratio": global_ratio,
        "eligible_entity_indices": eligible_indices,
        "per_entity_forecast_to_history_magnitude_ratio": per_entity_ratios,
        "minimum_eligible_entity_ratio": (
            min(finite_eligible_ratios) if finite_eligible_ratios else None
        ),
        "median_eligible_entity_ratio": (
            float(np.median(finite_eligible_ratios))
            if finite_eligible_ratios else None
        ),
        "collapsed_entity_indices": collapsed_indices,
        "ratio_above_float_range_entity_indices": overflow_ratio_indices,
        "minimum_ratio": MAGNITUDE_MIN_RATIO,
    }
    if collapsed_indices:
        return _assessment(
            "fail",
            "forecast_magnitude_collapsed",
            "forecast magnitude has collapsed relative to eligible history",
            responsibility="producer",
            **statistics,
        )
    return _assessment(
        "pass",
        "forecast_magnitude_preserved",
        "forecast magnitude occupies the history's data scale",
        **statistics,
    )
