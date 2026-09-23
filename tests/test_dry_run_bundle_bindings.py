"""R2C-063 — the stage 2.d dry-run binds its dimensions from real sources.

The dry-run built its test tensors only from the contract's own declarations,
so a model consistent with its own contract passed even when both disagreed
with the dataset the delivery ships. The 2026-08-05 pdfgnn roll spent its
whole smoke budget plus the judge's extra dispatch on that disagreement.

The binding ladder is weakest-source-first: generic defaults, the paper's
stated values, the run's derived params, then facts measured from the bundle.
"""

from __future__ import annotations

import json
from pathlib import Path


def _bundle(run_dir: Path, *, steps: int | None, table: str = "sales.csv"):
    example = run_dir / "method" / "example_data"
    example.mkdir(parents=True, exist_ok=True)
    entry: dict = {"file": table, "rows_kept": 100, "truncated": True}
    if steps is not None:
        entry["time_axis"] = {"column": "date", "steps_kept": steps,
                              "steps_in_source": steps,
                              "first_step": "2017-01-02",
                              "last_step": "2019-10-31",
                              "rows_per_step_cap": 48}
    (example / "PROVENANCE.json").write_text(
        json.dumps({"source": {}, "files": [entry]}), encoding="utf-8")


def _spec(**glossary: int) -> dict:
    return {"critical_requirements": {"param_glossary": [
        {"name": name, "paper_value": value}
        for name, value in glossary.items()
    ]}}


def test_time_axis_symbols_bind_to_the_bundled_axis(tmp_path):
    from validate_arch_contract_runtime import (
        _load_bindings_from_bundle,
        _load_bundle_dimension_facts,
    )

    _bundle(tmp_path, steps=1036)
    # Current constructor contracts consume only the semantic fact. The
    # spelling expansion below is retained solely for legacy shape strings.
    assert _load_bundle_dimension_facts(tmp_path) == {"time_axis_steps": 1036}
    bindings = _load_bindings_from_bundle(tmp_path)
    assert bindings["T_total"] == 1036
    assert bindings["T"] == 1036
    assert bindings["seq_len"] == 1036
    assert bindings["L"] == 1036


def test_the_degenerate_axis_reaches_the_dry_run_as_itself(tmp_path):
    """The known-bad: 4 steps must become the dry-run's 4, not the
    contract-flattering default of 8."""
    from validate_arch_contract_runtime import (
        DRY_RUN_BINDINGS, _load_bindings_from_bundle,
    )

    _bundle(tmp_path, steps=4)
    assert _load_bindings_from_bundle(tmp_path)["T_total"] == 4
    assert "T_total" not in DRY_RUN_BINDINGS


def test_no_bundle_leaves_every_binding_untouched(tmp_path):
    """Scope guard: the non-bundle paradigms keep byte-identical behavior."""
    from validate_arch_contract_runtime import (
        _load_bindings_from_bundle,
        _load_bundle_dimension_facts,
    )

    assert _load_bindings_from_bundle(tmp_path) == {}
    assert _load_bundle_dimension_facts(tmp_path) == {}
    _bundle(tmp_path, steps=None)
    assert _load_bindings_from_bundle(tmp_path) == {}
    assert _load_bundle_dimension_facts(tmp_path) == {}


def test_the_longest_bundled_axis_wins(tmp_path):
    from validate_arch_contract_runtime import _load_bindings_from_bundle

    example = tmp_path / "method" / "example_data"
    example.mkdir(parents=True)
    (example / "PROVENANCE.json").write_text(json.dumps({"files": [
        {"file": "lookup.csv", "time_axis": {"column": "valid_from",
                                             "steps_kept": 3}},
        {"file": "sales.csv", "time_axis": {"column": "date",
                                           "steps_kept": 900}},
    ]}), encoding="utf-8")
    assert _load_bindings_from_bundle(tmp_path)["T"] == 900


def test_paper_stated_values_bind_under_every_alias():
    from validate_arch_contract_runtime import _load_bindings_from_spec

    spec = {"critical_requirements": {"param_glossary": [
        {"name": "P", "aliases": ["lags_p", "context_length"],
         "paper_value": 10},
        {"name": "K", "aliases": ["prediction_horizon"], "paper_value": 26},
        {"name": "similarity threshold", "aliases": ["similarity_cutoff"],
         "paper_value": 0.95},
    ]}}
    bindings = _load_bindings_from_spec(spec)
    assert bindings["P"] == 10 and bindings["lags_p"] == 10
    assert bindings["context_length"] == 10 and bindings["K"] == 26
    # A non-integer value is not a dimension, and a name with a space is not
    # a symbol, so neither reaches the table.
    assert "similarity_cutoff" not in bindings
    assert "similarity threshold" not in bindings


def test_paper_unspecified_protocol_value_suppresses_legacy_dimensions():
    """K=1/K=26 cannot return through either superseded carrier."""
    from validate_arch_contract_runtime import _load_bindings_from_spec

    spec = {
        "comparison": {"evaluation_protocol": {"quantities": [{
            "role": "forecast_call_horizon",
            "parameter_name": "forecast_horizon",
            "paper_names": ["forecast horizon"],
            "paper_value_status": "paper_unspecified",
            "value": None,
            "unit": "week",
            "granularity": 1,
        }]}},
        "critical_requirements": {
            "param_glossary": [
                {"name": "K", "aliases": ["forecast_horizon"],
                 "paper_value": 26},
                {"name": "D", "aliases": ["hidden_dim"],
                 "paper_value": 8},
            ],
            "scale_dependent_hyperparameters": [
                {"name": "K", "paper_value": 1},
                {"name": "max_neighbors", "paper_value": 10},
            ],
        },
    }

    bindings = _load_bindings_from_spec(spec)
    assert "K" not in bindings
    assert "forecast_horizon" not in bindings
    assert bindings["D"] == 8 and bindings["hidden_dim"] == 8
    assert bindings["max_neighbors"] == 10


def test_paper_stated_protocol_value_overrides_legacy_numeric_claims():
    from validate_arch_contract_runtime import _load_bindings_from_spec

    spec = {
        "comparison": {"evaluation_protocol": {"quantities": [{
            "role": "forecast_call_horizon",
            "parameter_name": "forecast_horizon",
            "paper_names": ["forecast horizon"],
            "paper_value_status": "paper_stated",
            "value": 4,
            "unit": "week",
            "granularity": 1,
        }]}},
        "critical_requirements": {
            "param_glossary": [{
                "name": "K", "aliases": ["forecast_horizon"],
                "paper_value": 26,
            }],
            "scale_dependent_hyperparameters": [{
                "name": "K", "paper_value": 1,
            }],
        },
    }

    assert _load_bindings_from_spec(spec) == {
        "K": 4,
        "forecast_horizon": 4,
    }


def test_declared_protocol_symbol_binds_without_a_legacy_glossary():
    """Architecture dimensions consume the typed identity bridge directly."""
    from validate_arch_contract_runtime import _load_bindings_from_spec

    spec = {
        "comparison": {"evaluation_protocol": {"quantities": [{
            "role": "forecast_call_horizon",
            "parameter_name": "forecast_horizon",
            "paper_names": ["forecast horizon"],
            "paper_symbols": ["K"],
            "paper_value_status": "paper_stated",
            "value": 12,
            "unit": "week",
            "granularity": 1,
        }]}},
        "critical_requirements": {
            "param_glossary": [],
            "scale_dependent_hyperparameters": [],
        },
    }

    assert _load_bindings_from_spec(spec) == {
        "K": 12,
        "forecast_horizon": 12,
    }


def test_nonunit_protocol_dimension_binds_the_integral_step_equivalent():
    from validate_arch_contract_runtime import _load_bindings_from_spec

    spec = {
        "comparison": {"evaluation_protocol": {"quantities": [{
            "role": "forecast_call_horizon",
            "parameter_name": "forecast_horizon",
            "paper_names": ["forecast horizon"],
            "paper_symbols": ["K"],
            "paper_value_status": "paper_stated",
            "value": 6,
            "unit": "hour",
            "granularity": 0.5,
        }]}},
        "critical_requirements": {
            "param_glossary": [],
            "scale_dependent_hyperparameters": [],
        },
    }

    assert _load_bindings_from_spec(spec) == {
        "K": 12,
        "forecast_horizon": 12,
    }


def test_protocol_symbols_preserve_case_in_architecture_bindings():
    from validate_arch_contract_runtime import _load_bindings_from_spec

    spec = {
        "comparison": {"evaluation_protocol": {"quantities": [{
            "role": "forecast_call_horizon",
            "parameter_name": "forecast_horizon",
            "paper_names": ["forecast horizon"],
            "paper_symbols": ["K"],
            "paper_value_status": "paper_stated",
            "value": 12,
            "unit": "week",
            "granularity": 1,
        }]}},
        "critical_requirements": {"param_glossary": [{
            "name": "k", "aliases": [], "paper_value": 3,
        }]},
    }

    bindings = _load_bindings_from_spec(spec)
    assert bindings["K"] == 12
    assert bindings["k"] == 3


def test_protocol_binding_does_not_capture_a_different_temporal_role():
    """A test span and one-call horizon may share a unit, not authority."""
    from validate_arch_contract_runtime import _load_bindings_from_spec

    spec = {
        "comparison": {"evaluation_protocol": {"quantities": [{
            "role": "test_span",
            "parameter_name": "test_span",
            "paper_names": ["test span"],
            "paper_value_status": "paper_stated",
            "value": 26,
            "unit": "week",
            "granularity": 1,
        }]}},
        "critical_requirements": {"param_glossary": [{
            "name": "K", "aliases": ["forecast_horizon"],
            "paper_value": 4,
        }]},
    }

    assert _load_bindings_from_spec(spec) == {
        "K": 4,
        "forecast_horizon": 4,
        "test_span": 26,
    }


def test_the_ladder_orders_data_last(tmp_path):
    """Defaults lose to the paper, the paper loses to derived params, and
    everything loses to a measured fact about the shipped data."""
    from validate_arch_contract_runtime import (
        DRY_RUN_BINDINGS, _load_bindings_from_bundle,
        _load_bindings_from_params, _load_bindings_from_spec,
    )

    pipeline = tmp_path / ".pipeline"
    pipeline.mkdir()
    (pipeline / "params.json").write_text(
        json.dumps({"T_total": {"value": 52}, "P": {"value": 12}}),
        encoding="utf-8")
    _bundle(tmp_path, steps=1036)

    resolved = {
        **DRY_RUN_BINDINGS,
        **_load_bindings_from_spec(_spec(P=10, T_total=999)),
        **_load_bindings_from_params(tmp_path),
        **_load_bindings_from_bundle(tmp_path),
    }
    assert resolved["P"] == 12       # params beat the paper's 10
    assert resolved["T_total"] == 1036  # the bundle beats both
    assert resolved["N"] == DRY_RUN_BINDINGS["N"]  # untouched default


def test_a_malformed_provenance_file_is_not_a_crash(tmp_path):
    from validate_arch_contract_runtime import _load_bindings_from_bundle

    example = tmp_path / "method" / "example_data"
    example.mkdir(parents=True)
    (example / "PROVENANCE.json").write_text("{not json", encoding="utf-8")
    assert _load_bindings_from_bundle(tmp_path) == {}
