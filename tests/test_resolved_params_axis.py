"""R2C-081: typed protocol values reconcile against compatible ranges."""

from __future__ import annotations

import copy
import json

import pytest

from tests.helpers.state import make_state


def _quantity(
    role: str,
    parameter_name: str | None,
    value: int | float | None,
    *,
    status: str = "paper_stated",
    unit: str = "week",
    granularity: int | float = 1,
) -> dict:
    label = role.replace("_", " ")
    return {
        "role": role,
        "parameter_name": parameter_name,
        "paper_names": [label],
        "paper_symbols": ["K"] if role == "forecast_call_horizon" else [],
        "value": value,
        "unit": unit,
        "granularity": granularity,
        "paper_value_status": status,
        "evidence_quote": f"Synthetic control states {label}={value!r}.",
        "paper_section": "Synthetic protocol",
        "paper_element_ids": [f"synthetic-{role}"],
        "axis_evidence_quote": (
            f"Synthetic target observations have cadence {granularity} {unit}."
        ),
        "axis_paper_section": "Synthetic axis",
        "axis_paper_element_ids": ["synthetic-axis"],
    }


def _spec(
    *,
    horizon: int | float | None = 12,
    validation: int | float = 13,
) -> dict:
    status = "paper_stated" if horizon is not None else "paper_unspecified"
    return {
        "comparison": {
            "classification": {"id": "time_series_forecasting"},
            "pluggable_component": {"signature": ""},
            "evaluation_protocol": {
                "scheme": {"kind": "single_holdout"},
                "quantities": [
                    _quantity("context_length", "context_length", 10),
                    _quantity(
                        "forecast_call_horizon",
                        "forecast_horizon",
                        horizon,
                        status=status,
                    ),
                    _quantity("validation_span", None, validation),
                    _quantity("test_span", None, 26),
                ],
            },
        },
        # Known-bad legacy values remain deliberately contradictory. They are
        # negative controls: neither surface participates in reconciliation.
        "critical_requirements": {
            "param_glossary": [
                {
                    "name": "K",
                    "aliases": ["forecast_horizon"],
                    "meaning_quote": "K future steps define the forecast.",
                    "paper_section": "Legacy fixture",
                    "paper_value": 26,
                },
            ],
            "scale_dependent_hyperparameters": [{
                "name": "K (forecast horizon)",
                "paper_value": 26,
                "assumes_data_scale": "148 weekly steps",
                "paper_section": "Legacy fixture",
            }],
        },
    }


def _nonunit_spec(*, divisible: bool = True) -> dict:
    spec = _spec()
    quantities = spec["comparison"]["evaluation_protocol"]["quantities"]
    physical_values = {
        "context_length": 5,
        "forecast_call_horizon": 6 if divisible else 6.25,
        "validation_span": 6.5,
        "test_span": 13,
    }
    for quantity in quantities:
        quantity.update({
            "value": physical_values[quantity["role"]],
            "unit": "hour",
            "granularity": 0.5,
            "axis_evidence_quote": (
                "Synthetic target observations have cadence 0.5 hour."
            ),
        })
    return spec


def _manifest(
    usable_steps: int,
    *,
    unit: str = "week",
    granularity: int | float = 1,
    include_cadence: bool = True,
) -> dict:
    axis = {
        "column": "date",
        "steps_kept": usable_steps,
        "live_steps": usable_steps,
    }
    if include_cadence:
        axis["source_cadence"] = {
            "unit": unit,
            "granularity": granularity,
        }
    return {"files": [{"file": "sales.csv", "time_axis": axis}]}


def _seed_bundle(run_dir, usable_steps: int, **manifest_kwargs) -> None:
    example_data = run_dir / "method" / "example_data"
    example_data.mkdir(parents=True, exist_ok=True)
    (example_data / "PROVENANCE.json").write_text(
        json.dumps(_manifest(usable_steps, **manifest_kwargs)),
        encoding="utf-8",
    )


def test_real_pdfgnn_symbolic_k_remains_system_owned_despite_stale_k26(
    tmp_path,
):
    from derive_params import derive

    run_dir = tmp_path / "run"
    _seed_bundle(run_dir, 1033)

    horizon = derive(_spec(horizon=None), run_dir=run_dir)["params"][
        "forecast_horizon"
    ]

    assert horizon["value"] == 4
    assert horizon["source"] == "system_inferred"
    assert horizon["protocol_value"] is None
    assert horizon["paper_value_status"] == "paper_unspecified"
    assert "paper_value" not in horizon
    assert "26" not in horizon["reasoning"]


def test_explicit_k12_adopts_on_compatible_sufficient_axis(tmp_path):
    from derive_params import derive

    run_dir = tmp_path / "run"
    _seed_bundle(run_dir, 49)

    horizon = derive(_spec(), run_dir=run_dir)["params"]["forecast_horizon"]

    assert horizon["value"] == 12
    assert horizon["source"] == "paper"
    assert "paper_value" not in horizon
    assert horizon["protocol_value"] == 12
    assert horizon["protocol_unit"] == "week"
    assert horizon["protocol_granularity"] == 1
    assert "K=12 <= validation_span=13 is True" in horizon["note"]
    assert "K=12 <= test_span=26 is True" in horizon["note"]
    assert "context_length=10 + K=12 = 22 <= compatible realized axis=49" in (
        horizon["note"]
    )
    assert "26" not in horizon["note"].split("test_span=26", 1)[1]


def test_explicit_k12_keeps_runtime_when_validation_span_is_insufficient(
    tmp_path,
):
    from derive_params import derive

    run_dir = tmp_path / "run"
    # Static R2C-065 coverage still passes: C10 + V10 + T26 = 46 and the
    # compatible axis has 46 steps. The independent per-call K<=V boundary
    # is what fails, so this shape is reachable past Stage 2a.
    _seed_bundle(run_dir, 46)

    horizon = derive(
        _spec(validation=10), run_dir=run_dir,
    )["params"]["forecast_horizon"]

    assert horizon["value"] == 4
    assert horizon["source"] == "system_default"
    assert horizon["paper_value"] == 12
    assert horizon["protocol_value"] == 12
    assert "K=12 <= validation_span=10 is False" in horizon["reasoning"]
    assert "K=12 <= test_span=26 is True" in horizon["reasoning"]
    assert "context_length=10 + K=12 = 22 <= compatible realized axis=46" in (
        horizon["reasoning"]
    )
    assert "is False" in horizon["reasoning"]


@pytest.mark.parametrize(
    "manifest_kwargs",
    [
        {"include_cadence": False},
        {"unit": "day", "granularity": 1},
    ],
)
def test_missing_or_mismatched_cadence_has_no_axis_authority(
    tmp_path, manifest_kwargs,
):
    from derive_params import derive

    run_dir = tmp_path / "run"
    _seed_bundle(run_dir, 4, **manifest_kwargs)

    horizon = derive(_spec(), run_dir=run_dir)["params"]["forecast_horizon"]

    assert horizon["value"] == 4
    assert horizon["source"] == "system_default"
    assert horizon["paper_value"] == 12
    assert "compatible realized axis" not in horizon["reasoning"]
    assert "insufficient" not in horizon["reasoning"]


def test_no_bundle_has_no_axis_authority(tmp_path):
    from derive_params import derive

    horizon = derive(_spec(), run_dir=tmp_path / "run")["params"][
        "forecast_horizon"
    ]

    assert horizon["value"] == 4
    assert horizon["source"] == "system_default"
    assert horizon["paper_value"] == 12
    assert "compatible realized axis" not in horizon["reasoning"]


def test_validator_independently_rejects_stale_runtime_on_feasible_axis(
    tmp_path,
):
    from derive_params import derive
    from validate_params_output import validate

    run_dir = tmp_path / "run"
    output = derive(_spec(), run_dir=run_dir)
    _seed_bundle(run_dir, 49)
    pipeline = run_dir / ".pipeline"
    pipeline.mkdir()
    (pipeline / "params.json").write_text(json.dumps(output), encoding="utf-8")

    errors = validate(_spec(), run_dir)

    finding = next(
        error for error in errors
        if "typed_authority_consumer_bypass" in error
    )
    assert "params.forecast_horizon.value=4" in finding
    assert "quantities[1] role='forecast_call_horizon'" in finding
    assert "12 week / 1 week/step = 12 runtime steps" in finding
    assert "K=12 <= validation_span=13 is True" in finding
    assert "K=12 <= test_span=26 is True" in finding
    assert "context_length=10 + K=12 = 22 <= compatible realized axis=49" in (
        finding
    )
    assert "PROVENANCE.json" in finding


def test_assumption_copies_exact_insufficient_range_arithmetic(tmp_path):
    import run_pipeline
    from derive_params import derive
    from scripts import run_layout

    state = make_state(tmp_path / "run")
    _seed_bundle(state.paths.run_dir, 46)
    params = derive(_spec(validation=10), run_dir=state.paths.run_dir)
    state.paths.pipeline_dir.mkdir(parents=True, exist_ok=True)
    state.paths.method_spec.write_text(
        json.dumps(_spec(validation=10)), encoding="utf-8"
    )
    (state.paths.pipeline_dir / "params.json").write_text(
        json.dumps(params), encoding="utf-8"
    )
    finding = {
        "id": "F001",
        "check_id": "system_default_reasoning_explains_deviation",
        "severity": "important",
        "location": "forecast_horizon",
        "description": "The demo uses a smaller forecast horizon.",
        "proposed_fix": "Use the paper horizon when the data supports it.",
    }

    keep, deferred = run_pipeline._defer_disclosed_demo_scale_findings(
        state, "stage_2x", [finding]
    )

    assert keep == [] and deferred == ["F001"]
    assumptions = (
        state.paths.run_dir / run_layout.ASSUMPTIONS_MD
    ).read_text(encoding="utf-8")
    # assumptions.md is HTML-escaped on write (Veracode CWE-80), so `<=`
    # lands as `&lt;=`; Markdown viewers render it back as `<=`.
    assert "K=12 &lt;= validation_span=10 is False" in assumptions
    assert "context_length=10 + K=12 = 22 &lt;= compatible realized axis=46" in (
        assumptions
    )


def test_nonunit_integral_quantity_keeps_raw_value_and_derived_steps_separate(
    tmp_path,
):
    from derive_params import derive
    from schemas.params import Params
    from validate_params_output import _evaluation_protocol_errors

    run_dir = tmp_path / "run"
    _seed_bundle(run_dir, 98, unit="hour", granularity=0.5)

    output = derive(_nonunit_spec(), run_dir=run_dir)
    horizon = output["params"]["forecast_horizon"]

    assert horizon["protocol_value"] == 6
    assert horizon["protocol_unit"] == "hour"
    assert horizon["protocol_granularity"] == 0.5
    assert horizon["value"] == 12
    assert horizon["source"] == "paper"
    assert "paper_value" not in horizon
    assert "6 hour / 0.5 hour/step = 12 runtime steps" in horizon["note"]
    Params.model_validate(output)
    assert _evaluation_protocol_errors(_nonunit_spec(), output["params"]) == []


def test_nondivisible_physical_quantity_fails_closed_without_rounding(tmp_path):
    from derive_params import derive
    from schemas.method_spec import MethodSpec
    from scripts.validate_method_spec import ROOT, cross_check_evaluation_protocol
    from tests.test_evaluation_protocol import (
        _pdfgnn_spec,
        _protocol_quantity,
    )

    validated_spec = _pdfgnn_spec()
    horizon = _protocol_quantity(validated_spec, "forecast_call_horizon")
    horizon.update({
        "value": 6.25,
        "paper_value_status": "paper_stated",
        "paper_names": ["forecast horizon"],
        "paper_symbols": ["K"],
        "unit": "hour",
        "granularity": 0.5,
        "evidence_quote": "The forecast horizon K is 6.25 hours.",
        "paper_section": "Synthetic protocol",
        "paper_element_ids": ["synthetic-nondivisible-k"],
        "axis_evidence_quote": "Target demand is recorded every 0.5 hours.",
        "axis_paper_section": "Synthetic axis",
        "axis_paper_element_ids": ["synthetic-half-hour-axis"],
    })
    validated = MethodSpec.model_validate(validated_spec)
    errors = cross_check_evaluation_protocol(validated, ROOT)
    assert len(errors) == 1
    assert "not a positive integral physical-to-step conversion" in errors[0]
    assert "value=6.25 hour" in errors[0]
    assert "gives 12.5 steps" in errors[0]

    with pytest.raises(ValueError, match="not an integral number"):
        derive(_nonunit_spec(divisible=False), run_dir=tmp_path / "run")


def test_stale_lane_k26_cannot_change_explicit_typed_k12(tmp_path):
    from derive_params import derive

    spec = copy.deepcopy(_spec())
    run_dir = tmp_path / "run"
    _seed_bundle(run_dir, 49)

    horizon = derive(spec, run_dir=run_dir)["params"]["forecast_horizon"]

    assert horizon["value"] == 12
    assert horizon["protocol_value"] == 12
    assert "26" not in horizon["note"].split("test_span=26", 1)[1]
