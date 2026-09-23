"""Canonical public-call contract for time-series forecasting."""

from __future__ import annotations

from scripts.taxonomy import load_pluggable_component_contract


def test_forecast_contract_carries_stable_identity_and_scaling_state():
    contract = load_pluggable_component_contract("time_series_forecasting")

    assert contract is not None
    assert contract["signature_template"] == (
        "forecast(model, history, static_features, time_varying_features, "
        "graph, entity_ids, target_scaling_state, seed, "
        "*paradigm_extras_by_name) -> ForecastResult"
    )
    assert contract["contract"]["fixed_positional_args"] == [
        "model",
        "history",
        "static_features",
        "time_varying_features",
        "graph",
        "entity_ids",
        "target_scaling_state",
    ]


def test_forecast_result_contract_requires_original_target_units():
    contract = load_pluggable_component_contract("time_series_forecasting")

    assert contract is not None
    definition = contract["return_type_definition"]
    rationale = contract["rationale"]
    assert "mean" in definition and "original target units" in definition
    assert "variance" in definition and "squared original target units" in definition
    assert "samples" in definition and "original target units" in definition
    assert "typed stable entity ids" in rationale
    assert "rather than row position" in rationale
    assert "variance multiplies by squared scale" in rationale
