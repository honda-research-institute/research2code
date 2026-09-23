"""Family-owned typed offline-demo contract for forecasting (R2C-052)."""

from __future__ import annotations

import copy
from pathlib import Path

import pytest
import yaml

from scripts import build_plan, taxonomy
from scripts.validate_taxonomy import lint


ROOT = Path(__file__).resolve().parents[1]


def _spec(paradigm_id: str = "time_series_forecasting") -> dict:
    return {
        "comparison": {
            "classification": {"id": paradigm_id},
            "pluggable_component": {
                "name": "forecast",
                "signature": (
                    "forecast(model, history, static_features, "
                    "time_varying_features, graph, entity_ids, "
                    "target_scaling_state, seed) -> ForecastResult"
                ),
                "seed_param": "seed",
            },
        }
    }


def _raw_taxonomy() -> dict:
    return yaml.safe_load(
        (ROOT / "docs/ssot/taxonomies.yaml").read_text(encoding="utf-8")
    )


def _tsf_fields(raw: dict) -> dict:
    return raw["method_roots"]["TE"]["families"]["TE-TSF"]


def test_tsf_declares_one_closed_offline_panel_schema():
    tax = taxonomy.load_taxonomy()
    node = taxonomy.serves("time_series_forecasting", tax)
    assert node is not None
    declared = taxonomy.node_implementation(node, tax)["build_plan"][
        "offline_demo_data"
    ]

    assert list(declared) == [
        "schema_id",
        "synthetic_feasible",
        "generator_id",
        "generator_version",
        "entity_count",
        "family_fitting_prefix_steps",
        "supported_cadences",
        "tables",
    ]
    assert declared["schema_id"] == "time_series_panel_v1"
    assert declared["synthetic_feasible"] is True
    assert declared["generator_id"] == "time_series_offline_fallback"
    assert declared["generator_version"] == "1.0.0"
    assert declared["entity_count"] == 8
    assert declared["family_fitting_prefix_steps"] == 24
    assert declared["supported_cadences"] == [
        "week", "day", "hour", "minute", "second"
    ]
    assert list(declared["tables"]) == ["series.csv", "observations.csv"]
    assert declared["tables"]["series.csv"] == {
        "columns": [
            {
                "name": "series_id",
                "dtype": "string",
                "semantic_role": "entity_id",
            },
            {
                "name": "static_level",
                "dtype": "float64",
                "semantic_role": "static_numeric_level",
            },
            {
                "name": "static_amplitude",
                "dtype": "float64",
                "semantic_role": "static_numeric_amplitude",
            },
        ],
        "primary_key": ["series_id"],
    }
    assert declared["tables"]["observations.csv"] == {
        "columns": [
            {
                "name": "series_id",
                "dtype": "string",
                "semantic_role": "entity_id",
            },
            {
                "name": "timestamp",
                "dtype": "datetime_iso8601_utc",
                "semantic_role": "time_index",
            },
            {
                "name": "target",
                "dtype": "float64",
                "semantic_role": "target",
            },
            {
                "name": "season_sin",
                "dtype": "float64",
                "semantic_role": "known_time_varying_covariate",
            },
            {
                "name": "season_cos",
                "dtype": "float64",
                "semantic_role": "known_time_varying_covariate",
            },
            {
                "name": "time_fraction",
                "dtype": "float64",
                "semantic_role": "known_time_varying_covariate",
            },
        ],
        "primary_key": ["series_id", "timestamp"],
    }


def test_resolved_plan_copies_taxonomy_schema_and_adds_single_sourced_k():
    tax = taxonomy.load_taxonomy()
    node = taxonomy.serves("time_series_forecasting", tax)
    assert node is not None
    declared = taxonomy.node_implementation(node, tax)["build_plan"][
        "offline_demo_data"
    ]

    resolved = build_plan.load_build_plan(_spec())
    assert resolved is not None
    expected = copy.deepcopy(declared)
    expected["forecast_horizon_demo_value"] = 4
    assert resolved["offline_demo_data"] == expected
    assert "offline_demo_data" not in build_plan.STATIC_PLAN_BY_PARADIGM[
        "time_series_forecasting"
    ]


def test_resolved_k_is_read_from_the_protocol_role_declaration(monkeypatch):
    monkeypatch.setattr(
        build_plan.taxonomy,
        "load_params_derivation",
        lambda *_args, **_kwargs: {
            "forecast_horizon": {"kind": "config_path", "demo_value": 7}
        },
    )
    resolved = build_plan.load_build_plan(_spec())
    assert resolved is not None
    assert resolved["offline_demo_data"]["forecast_horizon_demo_value"] == 7


@pytest.mark.parametrize("bad_value", [None, True, 0, -1, 4.0, "4"])
def test_resolved_plan_refuses_an_invalid_system_owned_k(
    monkeypatch, bad_value
):
    monkeypatch.setattr(
        build_plan.taxonomy,
        "load_params_derivation",
        lambda *_args, **_kwargs: {
            "forecast_horizon": {
                "kind": "config_path",
                "demo_value": bad_value,
            }
        },
    )
    with pytest.raises(ValueError, match="positive integral"):
        build_plan.load_build_plan(_spec())


def test_adjacent_families_do_not_inherit_a_forecasting_fallback():
    for paradigm_id in ("active_learning", "knowledge_distillation"):
        resolved = build_plan.load_build_plan(_spec(paradigm_id))
        assert resolved is not None
        assert "offline_demo_data" not in resolved


@pytest.mark.parametrize(
    ("source", "inherits_offline_contract"),
    [("neutral", False), ("inherit_parent", True)],
)
def test_provisional_tsf_child_honors_its_build_plan_source(
    tmp_path, source, inherits_offline_contract,
):
    from tests.test_provisional_pack_overlay import (
        GAP_SHAPED_PACK,
        _install_run_pack,
    )

    pack = copy.deepcopy(GAP_SHAPED_PACK)
    pack.update({
        "legacy_paradigm": "graph_time_series_forecasting",
        "extends": "time_series_forecasting",
        "taxonomy_id": "TE-TSF/time_series_forecasting/graph_variant",
        "build_plan": {
            "source": source,
            "reasoning": "paired family-contract inheritance control",
        },
    })
    packs = _install_run_pack(tmp_path / source, pack)
    resolved = build_plan.load_build_plan(
        _spec("graph_time_series_forecasting"),
        provisional_packs_dir=packs,
    )

    assert resolved is not None
    assert ("offline_demo_data" in resolved) is inherits_offline_contract
    if inherits_offline_contract:
        assert resolved["offline_demo_data"]["schema_id"] == (
            "time_series_panel_v1"
        )
        assert resolved["offline_demo_data"][
            "forecast_horizon_demo_value"
        ] == 4


def test_taxonomy_lint_accepts_the_canonical_offline_contract():
    offline_codes = {
        diagnostic.code
        for diagnostic in lint(taxonomy.load_taxonomy())
        if diagnostic.code.startswith("offline_demo_data")
    }
    assert offline_codes == set()


def test_taxonomy_lint_rejects_column_substitution_and_extra_keys():
    raw = _raw_taxonomy()
    offline = _tsf_fields(raw)["build_plan"]["offline_demo_data"]
    offline["tables"]["series.csv"]["columns"][1]["name"] = (
        "static_category"
    )
    offline["caller_overrides"] = True
    tax = taxonomy.load_taxonomy_uncached(ROOT, raw)
    codes = {diagnostic.code for diagnostic in lint(tax)}
    assert "offline_demo_data_keys" in codes
    assert "offline_demo_data_tables" in codes


def test_taxonomy_lint_rejects_a_non_integral_horizon_default():
    raw = _raw_taxonomy()
    _tsf_fields(raw)["params_derivation"]["forecast_horizon"][
        "demo_value"
    ] = 4.0
    tax = taxonomy.load_taxonomy_uncached(ROOT, raw)
    assert "offline_demo_data_forecast_horizon" in {
        diagnostic.code for diagnostic in lint(tax)
    }
