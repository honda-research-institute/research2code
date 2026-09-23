"""R2C-065 — a bundled dataset must cover the axis the method consumes.

The 2026-08-05 pdfgnn roll is the known-bad throughout: the materializer
kept "the header plus the first 50,000 data rows" of a date-sorted daily
sales table, which is 4 unique dates, against a protocol needing at least
36 steps. Volume survived and the axis the method consumes did not.

Three seams are covered here: the subsample rule itself (thin the entity
dimension, keep the span), the protocol floor that refuses a bundle the
method cannot run on, and the coder-side arm that forbids silently clamping
a value the paper states.
"""

from __future__ import annotations

import io
import json
import zipfile
from datetime import date, timedelta
from pathlib import Path

import pytest


def _panel_csv(
    *, dates: int, rows_per_date: int, start_day: int = 2, step_days: int = 1,
) -> bytes:
    """A long-format panel sorted by date, then product — the real shape."""
    lines = ["product_id,store_id,date,sales"]
    first = date(2017, 1, start_day)
    for d in range(dates):
        day = (first + timedelta(days=d * step_days)).isoformat()
        for p in range(rows_per_date):
            lines.append(f"P{p:04d},S0001,{day},{p}.0")
    return ("\n".join(lines) + "\n").encode()


def _acquisition(tmp_path: Path, payload: bytes, filename: str):
    from dataset_acquisition import Acquisition, classify_source

    path = tmp_path / filename
    path.write_bytes(payload)
    source = classify_source(
        "https://www.kaggle.com/datasets/berkayalan/retail-sales-data")
    return Acquisition(source=source, path=path, sha256="cd" * 32,
                       bytes_written=len(payload), truncated=False)


def _zip(members: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for name, content in members.items():
            z.writestr(name, content)
    return buf.getvalue()


# --------------------------------------------------------------------------
# The subsample rule
# --------------------------------------------------------------------------


def test_row_cap_alone_would_have_destroyed_the_axis(tmp_path):
    """The known-bad arithmetic, stated as a test so the regression is
    visible: 60 dates x 500 rows capped at 1,000 leading rows is 2 dates."""
    from dataset_acquisition import materialize_demo_bundle

    payload = _zip({"sales.csv/sales.csv": _panel_csv(dates=60, rows_per_date=500)})
    manifest = materialize_demo_bundle(
        _acquisition(tmp_path, payload, "retail.zip"), tmp_path / "bundle",
        max_rows_per_table=1000, max_rows_per_axis_table=1000)

    axis = manifest["files"][0]["time_axis"]
    assert axis["column"] == "date"
    assert axis["steps_kept"] == 60, "the whole span must survive the cap"
    assert axis["steps_in_source"] == 60
    assert axis["rows_per_step_cap"] == 16  # 1000 // 60
    assert manifest["files"][0]["rows_kept"] <= 1000


def test_axis_subsample_keeps_every_date_and_thins_entities(tmp_path):
    import csv

    from dataset_acquisition import materialize_demo_bundle

    dest = tmp_path / "bundle"
    payload = _zip({"sales.csv": _panel_csv(dates=40, rows_per_date=200)})
    materialize_demo_bundle(_acquisition(tmp_path, payload, "retail.zip"), dest,
                            max_rows_per_table=400,
                            max_rows_per_axis_table=400)

    with (dest / "sales.csv").open() as fh:
        rows = list(csv.reader(fh))
    header, data = rows[0], rows[1:]
    assert header == ["product_id", "store_id", "date", "sales"]
    dates = [r[2] for r in data]
    assert len(set(dates)) == 40
    # 400 // 40 = 10 rows per step, and the kept rows are the leading ones
    # within each step, so the entity set stays stable across the span.
    assert all(dates.count(d) == 10 for d in set(dates))
    assert {r[0] for r in data} == {f"P{i:04d}" for i in range(10)}


def test_a_table_under_the_budget_is_never_thinned(tmp_path):
    """Axis awareness must not cost rows a plain cap would have kept."""
    from dataset_acquisition import materialize_demo_bundle

    payload = _zip({"sales.csv": _panel_csv(dates=30, rows_per_date=5)})
    manifest = materialize_demo_bundle(
        _acquisition(tmp_path, payload, "retail.zip"), tmp_path / "bundle",
        max_rows_per_table=1000)

    entry = manifest["files"][0]
    assert entry["rows_kept"] == 150
    assert entry["truncated"] is False
    assert entry["time_axis"]["rows_per_step_cap"] is None


def test_more_steps_than_the_budget_keeps_one_row_each(tmp_path):
    from dataset_acquisition import materialize_demo_bundle

    # One row per step, 50 steps, budget of 20: the first 20 steps survive.
    rows = ["sensor,timestamp,value"]
    rows += [f"s1,2017-01-01T00:{i:02d}:00,{i}" for i in range(50)]
    payload = _zip({"readings.csv": ("\n".join(rows) + "\n").encode()})
    manifest = materialize_demo_bundle(
        _acquisition(tmp_path, payload, "readings.zip"), tmp_path / "bundle",
        max_rows_per_table=20, max_rows_per_axis_table=20)

    axis = manifest["files"][0]["time_axis"]
    assert axis["steps_kept"] == 20
    assert axis["steps_in_source"] == 50
    assert axis["rows_per_step_cap"] == 1
    assert manifest["files"][0]["truncated"] is True


def test_an_axis_table_gets_a_bigger_budget_than_a_flat_one(tmp_path):
    """An axis table funds two dimensions out of one budget, so it gets the
    multiplied one by default. At the motivating table's density the base
    budget buys the full span at four articles, which runs the protocol and
    makes the paper's article graph trivial."""
    from dataset_acquisition import (
        AXIS_ROW_BUDGET_MULTIPLIER, materialize_demo_bundle,
    )

    payload = _zip({"sales.csv": _panel_csv(dates=30, rows_per_date=100)})
    manifest = materialize_demo_bundle(
        _acquisition(tmp_path, payload, "retail.zip"), tmp_path / "bundle",
        max_rows_per_table=300)

    entry = manifest["files"][0]
    assert AXIS_ROW_BUDGET_MULTIPLIER == 10
    # 3,000 source rows against a 3,000-row axis budget: nothing is cut,
    # where the base budget of 300 would have kept 10 rows per date.
    assert entry["rows_kept"] == 3000
    assert entry["truncated"] is False
    assert entry["time_axis"]["entities_kept"] == 100


def test_the_entity_breadth_the_demo_actually_has_is_recorded(tmp_path):
    """Preserving the span costs entity breadth, so how much was thinned has
    to be a stated fact rather than something a reader infers."""
    from dataset_acquisition import materialize_demo_bundle

    payload = _zip({"sales.csv": _panel_csv(dates=20, rows_per_date=100)})
    manifest = materialize_demo_bundle(
        _acquisition(tmp_path, payload, "retail.zip"), tmp_path / "bundle",
        max_rows_per_table=200, max_rows_per_axis_table=200)

    axis = manifest["files"][0]["time_axis"]
    assert axis["entity_column"] == "product_id"
    assert axis["entities_kept"] == 10, "10 rows per step, leading products"


def test_regular_source_timestamps_record_exact_cadence(tmp_path):
    from dataset_acquisition import materialize_demo_bundle

    daily = materialize_demo_bundle(
        _acquisition(
            tmp_path, _zip({"daily.csv": _panel_csv(
                dates=8, rows_per_date=2,
            )}), "daily.zip",
        ),
        tmp_path / "daily-bundle",
    )
    weekly = materialize_demo_bundle(
        _acquisition(
            tmp_path, _zip({"weekly.csv": _panel_csv(
                dates=8, rows_per_date=2, step_days=7,
            )}), "weekly.zip",
        ),
        tmp_path / "weekly-bundle",
    )

    assert daily["files"][0]["time_axis"]["source_cadence"] == {
        "unit": "day", "granularity": 1,
    }
    assert weekly["files"][0]["time_axis"]["source_cadence"] == {
        "unit": "week", "granularity": 1,
    }


def test_regular_but_ambiguous_timestamps_have_no_cadence(tmp_path):
    from dataset_acquisition import materialize_demo_bundle

    rows = (
        "series,date,value\n"
        "s1,01/02/2017,1\n"
        "s1,01/09/2017,2\n"
        "s1,01/16/2017,3\n"
    ).encode()
    manifest = materialize_demo_bundle(
        _acquisition(tmp_path, _zip({"series.csv": rows}), "series.zip"),
        tmp_path / "bundle",
    )

    assert "source_cadence" not in manifest["files"][0]["time_axis"]


def test_parseable_but_irregular_timestamps_have_no_cadence(tmp_path):
    from dataset_acquisition import materialize_demo_bundle

    rows = (
        "series,date,value\n"
        "s1,2017-01-02,1\n"
        "s1,2017-01-09,2\n"
        "s1,2017-01-23,3\n"
    ).encode()
    manifest = materialize_demo_bundle(
        _acquisition(tmp_path, _zip({"series.csv": rows}), "series.zip"),
        tmp_path / "bundle",
    )

    assert "source_cadence" not in manifest["files"][0]["time_axis"]


def test_a_single_time_step_table_is_row_exchangeable(tmp_path):
    """One timestamp is not an axis: such a table has nothing to preserve and
    must keep the plain row cap rather than the multiplied axis budget."""
    from dataset_acquisition import materialize_demo_bundle

    payload = _zip({"snapshot.csv": _panel_csv(dates=1, rows_per_date=200)})
    manifest = materialize_demo_bundle(
        _acquisition(tmp_path, payload, "retail.zip"), tmp_path / "bundle",
        max_rows_per_table=20)

    entry = manifest["files"][0]
    assert "time_axis" not in entry
    assert entry["rows_kept"] == 20


def test_a_table_with_no_time_axis_falls_back_to_the_row_cap(tmp_path):
    """The scope guard: every non-time-series paradigm keeps its old shape."""
    from dataset_acquisition import materialize_demo_bundle

    rows = ["image_id,label,width"] + [f"i{i},{i % 10},32" for i in range(50)]
    payload = _zip({"labels.csv": ("\n".join(rows) + "\n").encode()})
    manifest = materialize_demo_bundle(
        _acquisition(tmp_path, payload, "images.zip"), tmp_path / "bundle",
        max_rows_per_table=20)

    entry = manifest["files"][0]
    assert "time_axis" not in entry
    assert entry["rows_kept"] == 20
    assert entry["truncated"] is True


def test_a_time_named_column_holding_prose_is_not_an_axis(tmp_path):
    """Name alone is not enough — the first value must parse as a time
    point, or a free-text `update_time` column would define the axis."""
    from dataset_acquisition import materialize_demo_bundle

    rows = ["record_id,update_time,value"]
    rows += [f"r{i},last tuesday,{i}" for i in range(30)]
    payload = _zip({"records.csv": ("\n".join(rows) + "\n").encode()})
    manifest = materialize_demo_bundle(
        _acquisition(tmp_path, payload, "records.zip"), tmp_path / "bundle",
        max_rows_per_table=10)

    assert "time_axis" not in manifest["files"][0]


def test_the_real_date_column_outranks_its_component_columns():
    from dataset_acquisition import detect_time_axis_column

    header = ["year", "month", "order_date", "promo_discount_type_2", "qty"]
    first = ["2017", "1", "2017-01-02", "PR03", "4"]
    assert detect_time_axis_column(header, first) == 2


def test_subsample_rule_states_the_axis_it_preserved(tmp_path):
    from dataset_acquisition import materialize_demo_bundle

    payload = _zip({"sales.csv": _panel_csv(dates=20, rows_per_date=100)})
    manifest = materialize_demo_bundle(
        _acquisition(tmp_path, payload, "retail.zip"), tmp_path / "bundle",
        max_rows_per_table=200, max_rows_per_axis_table=200)

    rule = manifest["subsample_rule"]
    assert "ENTITY" in rule
    assert "20 steps on `date`" in rule
    assert "at most 10 rows per step" in rule
    assert "10 distinct `product_id` value(s)" in rule


def test_a_bare_csv_fetch_is_axis_aware_too(tmp_path):
    from dataset_acquisition import materialize_demo_bundle

    dest = tmp_path / "bundle"
    acq = _acquisition(tmp_path, _panel_csv(dates=25, rows_per_date=40),
                       "sales.csv")
    manifest = materialize_demo_bundle(acq, dest, max_rows_per_table=100,
                                       max_rows_per_axis_table=100)

    assert manifest["files"][0]["time_axis"]["steps_kept"] == 25
    assert not acq.path.exists(), "the fetched original is always cleaned up"


def test_an_in_place_bundle_does_not_truncate_its_own_source(tmp_path):
    """The two-pass read makes source-equals-destination a real hazard."""
    from dataset_acquisition import materialize_demo_bundle

    dest = tmp_path / "bundle"
    dest.mkdir()
    acq = _acquisition(dest, _panel_csv(dates=12, rows_per_date=10),
                       "sales.csv")
    manifest = materialize_demo_bundle(acq, dest, max_rows_per_table=60,
                                       max_rows_per_axis_table=60)

    assert manifest["files"][0]["time_axis"]["steps_kept"] == 12
    assert manifest["files"][0]["rows_kept"] == 60


# --------------------------------------------------------------------------
# The protocol floor
# --------------------------------------------------------------------------


def _typed_spec(
    *,
    context: int | float = 10,
    horizon: int | float | None = None,
    validation: int | float = 13,
    test: int | float = 26,
    unit: str = "week",
    granularity: int | float = 1,
    scheme: str | None = "single_holdout",
) -> dict:
    def quantity(role: str, value: int | float | None) -> dict:
        return {
            "role": role,
            "parameter_name": (
                "context_length" if role == "context_length"
                else "forecast_horizon"
                if role == "forecast_call_horizon"
                else None
            ),
            "value": value,
            "unit": unit,
            "granularity": granularity,
            "paper_value_status": (
                "paper_stated" if value is not None else "paper_unspecified"
            ),
        }

    return {"comparison": {"evaluation_protocol": {
        "scheme": {
            "kind": scheme,
            "paper_value_status": (
                "paper_stated" if scheme is not None else "paper_unspecified"
            ),
        },
        "quantities": [
            quantity("context_length", context),
            quantity("forecast_call_horizon", horizon),
            quantity("validation_span", validation),
            quantity("test_span", test),
        ],
    }}}


def _manifest(
    steps: int, *, unit: str | None = "week",
    granularity: int | float = 1,
) -> dict:
    axis = {
        "column": "date", "steps_kept": steps, "steps_in_source": steps,
        "first_step": "2017-01-02", "last_step": "2017-01-05",
        "rows_per_step_cap": 12,
    }
    if unit is not None:
        axis["source_cadence"] = {
            "unit": unit, "granularity": granularity,
        }
    return {"files": [{"file": "sales.csv", "time_axis": axis}]}


def test_pdfgnn_floor_is_context_plus_declared_evaluation_spans_not_k():
    from bundle_axis_floor import check_bundle_axis, protocol_axis_floor

    requirement = protocol_axis_floor(_typed_spec(horizon=None))
    assert requirement is not None
    assert requirement.floor == 49  # 10 context + 13 validation + 26 test

    verdict = check_bundle_axis(_manifest(4), requirement)
    assert verdict is not None and verdict.feasible is False
    message = verdict.message()
    assert "4 steps" in message
    assert "10 context steps" in message
    assert "13 validation steps" in message
    assert "26 test steps" in message
    assert "49 minimum" in message
    assert "forecast-call horizon is a separate per-call quantity" in message


def test_native_weekly_identity_axis_passes_the_typed_floor():
    from bundle_axis_floor import check_bundle_axis, protocol_axis_floor

    requirement = protocol_axis_floor(_typed_spec(horizon=999))
    verdict = check_bundle_axis(_manifest(1036), requirement)
    assert verdict is not None and verdict.feasible is True
    assert "protocol-feasible" in verdict.message()
    assert requirement.floor == 49, "even stated K is excluded from coverage"


def test_non_tsf_or_legacy_spec_without_typed_protocol_gets_no_floor():
    from bundle_axis_floor import protocol_axis_floor

    assert protocol_axis_floor({}) is None
    assert protocol_axis_floor(
        {"critical_requirements": {"param_glossary": [
            {"name": "similarity threshold", "aliases": ["similarity_cutoff"],
             "paper_value": 0.95}]}}) is None


def test_nonunit_protocol_values_convert_to_positive_integral_steps():
    from bundle_axis_floor import check_bundle_axis, protocol_axis_floor

    requirement = protocol_axis_floor(_typed_spec(
        context=24, validation=12, test=12, unit="hour", granularity=0.5,
    ))
    assert requirement is not None
    assert (requirement.context_steps, requirement.validation_steps,
            requirement.test_steps, requirement.floor) == (48, 24, 24, 96)
    verdict = check_bundle_axis(
        _manifest(96, unit="hour", granularity=0.5), requirement,
    )
    assert verdict is not None and verdict.feasible is True


def test_nondivisible_protocol_value_is_explicitly_unresolved():
    import pytest

    from bundle_axis_floor import ProtocolAxisUnresolved, protocol_axis_floor

    with pytest.raises(ProtocolAxisUnresolved, match="not an integral number"):
        protocol_axis_floor(_typed_spec(
            context=10, validation=13, test=26,
            unit="hour", granularity=3,
        ))


def test_protocol_roles_must_share_one_exact_unit_and_granularity():
    import pytest

    from bundle_axis_floor import ProtocolAxisUnresolved, protocol_axis_floor

    spec = _typed_spec()
    validation = next(
        item
        for item in spec["comparison"]["evaluation_protocol"]["quantities"]
        if item["role"] == "validation_span"
    )
    validation["unit"] = "day"

    with pytest.raises(ProtocolAxisUnresolved, match="one exact unit"):
        protocol_axis_floor(spec)


def test_a_bundle_with_no_time_column_yields_no_verdict():
    """An image bundle under a temporal spec is a detection gap, never a
    refusal: refusing on absence would delete a bundle for having no dates."""
    from bundle_axis_floor import check_bundle_axis, protocol_axis_floor

    requirement = protocol_axis_floor(_typed_spec())
    assert check_bundle_axis({"files": [{"file": "train.csv"}]},
                             requirement) is None


def test_the_longest_axis_across_tables_is_the_one_judged():
    from bundle_axis_floor import check_bundle_axis, protocol_axis_floor

    requirement = protocol_axis_floor(_typed_spec(
        context=1, validation=2, test=2,
    ))
    manifest = {"files": [
        {"file": "lookup.csv", "time_axis": {"column": "valid_from",
                                             "steps_kept": 2,
                                             "source_cadence": {
                                                 "unit": "week",
                                                 "granularity": 1,
                                             }}},
        {"file": "sales.csv", "time_axis": {"column": "date",
                                           "steps_kept": 90,
                                           "source_cadence": {
                                               "unit": "week",
                                               "granularity": 1,
                                           }}},
    ]}
    verdict = check_bundle_axis(manifest, requirement)
    assert verdict is not None and verdict.feasible is True
    assert verdict.table == "sales.csv"


def test_daily_source_under_weekly_protocol_is_unresolved_not_converted():
    from bundle_axis_floor import check_bundle_axis, protocol_axis_floor

    requirement = protocol_axis_floor(_typed_spec())
    verdict = check_bundle_axis(_manifest(1033, unit="day"), requirement)

    assert verdict is not None and verdict.feasible is None
    assert "source cadence is 1 day/step" in verdict.message()
    assert "no realized aggregation/resampling carrier" in verdict.message()


def test_missing_source_cadence_is_unresolved_not_infeasible():
    from bundle_axis_floor import check_bundle_axis, protocol_axis_floor

    requirement = protocol_axis_floor(_typed_spec())
    verdict = check_bundle_axis(_manifest(4, unit=None), requirement)

    assert verdict is not None and verdict.feasible is None
    assert "no exact regular parseable source cadence" in verdict.message()


# --------------------------------------------------------------------------
# The coder-side arm: no silent clamp of a paper-stated value
# --------------------------------------------------------------------------


def _clamp_hits(source: str, spec: dict) -> list[tuple]:
    import ast

    from scripts.validate_method_coder_output import (
        _paper_stated_param_tokens, _silently_clamped_paper_values,
    )

    return _silently_clamped_paper_values(
        ast.parse(source), _paper_stated_param_tokens(spec))


_EPOCH_SPEC = {"critical_requirements": {"training": {"num_epochs": 50,
                                                      "learning_rate": 5e-3}}}


def test_the_delivered_epoch_clamp_is_caught():
    """The verbatim 2026-08-05 shape: the table says 50, the demo does 10."""
    hits = _clamp_hits(
        "def train_and_forecast(series):\n"
        "    T_total = series.shape[1]\n"
        "    max_epochs = min(50, max(10, T_total // 2))\n"
        "    return max_epochs\n",
        _EPOCH_SPEC)
    assert len(hits) == 1
    line, name, spec_name, _ = hits[0]
    assert (line, name, spec_name) == (3, "max_epochs", "num_epochs")


def test_using_the_declared_value_is_clean():
    hits = _clamp_hits(
        "def train_and_forecast(series, max_epochs=50):\n"
        "    if series.shape[1] < 36:\n"
        "        raise ValueError('need >= 36 time steps, got %d'\n"
        "                         % series.shape[1])\n"
        "    return max_epochs\n",
        _EPOCH_SPEC)
    assert hits == []


def test_clamping_an_internal_the_paper_never_states_is_allowed():
    """R4 still stands: fitting a count to its operand is not this defect."""
    hits = _clamp_hits(
        "def select_batch(x_unlabeled, core_set_size):\n"
        "    n_clusters = min(core_set_size, len(x_unlabeled))\n"
        "    return n_clusters\n",
        _EPOCH_SPEC)
    assert hits == []


def test_a_glossary_stated_window_is_covered_by_the_same_arm():
    hits = _clamp_hits(
        "def forecast(series, lags_p=10):\n"
        "    lags_p = min(lags_p, series.shape[1])\n"
        "    return lags_p\n",
        {"critical_requirements": {"param_glossary": [
            {"name": "P", "aliases": ["lags_p"], "paper_value": 10},
            {"name": "K", "aliases": ["prediction_horizon"],
             "paper_value": 26},
        ]}})
    assert len(hits) == 1
    assert hits[0][1] == "lags_p"


def test_paper_unspecified_protocol_value_is_not_a_clamp_authority():
    """Stale glossary and scale-lane numbers cannot turn symbolic K stated."""
    spec = {
        "comparison": {"evaluation_protocol": {"quantities": [{
            "role": "forecast_call_horizon",
            "parameter_name": "forecast_horizon",
            "paper_names": ["forecast horizon"],
            "paper_value_status": "paper_unspecified",
            "value": None,
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

    hits = _clamp_hits(
        "def forecast(series, forecast_horizon=4):\n"
        "    forecast_horizon = min(forecast_horizon, series.shape[1])\n"
        "    return forecast_horizon\n",
        spec,
    )
    assert hits == []


def test_paper_stated_protocol_fact_drives_the_clamp_guard():
    """The typed fact stays authoritative even when legacy numbers disagree."""
    spec = {
        "comparison": {"evaluation_protocol": {"quantities": [{
            "role": "forecast_call_horizon",
            "parameter_name": "forecast_horizon",
            "paper_names": ["forecast horizon"],
            "paper_value_status": "paper_stated",
            "value": 4,
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

    hits = _clamp_hits(
        "def forecast(series, forecast_horizon=4):\n"
        "    forecast_horizon = min(forecast_horizon, series.shape[1])\n"
        "    return forecast_horizon\n",
        spec,
    )
    assert len(hits) == 1
    assert hits[0][2] == "forecast_horizon"


def test_typed_protocol_symbol_drives_clamp_guard_without_glossary():
    spec = {
        "comparison": {"evaluation_protocol": {"quantities": [{
            "role": "forecast_call_horizon",
            "parameter_name": "forecast_horizon",
            "paper_names": ["forecast horizon"],
            "paper_symbols": ["K"],
            "paper_value_status": "paper_stated",
            "value": 12,
        }]}},
        "critical_requirements": {
            "param_glossary": [],
            "scale_dependent_hyperparameters": [],
        },
    }

    hits = _clamp_hits(
        "def forecast(series, K=12):\n"
        "    K = min(K, series.shape[1])\n"
        "    return K\n",
        spec,
    )
    assert len(hits) == 1
    assert hits[0][1:3] == ("K", "K")


def test_protocol_symbol_clamp_guard_preserves_exact_case():
    spec = {
        "comparison": {"evaluation_protocol": {"quantities": [{
            "role": "forecast_call_horizon",
            "parameter_name": "forecast_horizon",
            "paper_names": ["forecast horizon"],
            "paper_symbols": ["K"],
            "paper_value_status": "paper_stated",
            "value": 12,
        }]}},
        "critical_requirements": {
            "param_glossary": [],
            "scale_dependent_hyperparameters": [],
        },
    }

    assert _clamp_hits(
        "def forecast(series, k=12):\n"
        "    k = min(k, series.shape[1])\n"
        "    return k\n",
        spec,
    ) == []


def test_typed_test_span_does_not_suppress_legacy_forecast_horizon():
    """Same-axis roles remain separate unless parameter_name joins them."""
    spec = {
        "comparison": {"evaluation_protocol": {"quantities": [{
            "role": "test_span",
            "parameter_name": "test_span",
            "paper_names": ["test span"],
            "paper_value_status": "paper_stated",
            "value": 26,
        }]}},
        "critical_requirements": {"param_glossary": [{
            "name": "K", "aliases": ["forecast_horizon"],
            "paper_value": 4,
        }]},
    }

    hits = _clamp_hits(
        "def forecast(series, forecast_horizon=4):\n"
        "    forecast_horizon = min(forecast_horizon, series.shape[1])\n"
        "    return forecast_horizon\n",
        spec,
    )
    assert len(hits) == 1
    assert hits[0][2] == "forecast_horizon"


def test_stage_2a_refuses_a_bundle_the_protocol_cannot_run_on(
    fake_subprocess, run_dir, monkeypatch,
):
    """An axis-degenerate public bundle is replaced by the typed third tier."""
    import dataset_acquisition
    from run_pipeline import run_stage_2a
    from tests.helpers.assertions import assert_stage_completed
    from tests.helpers.inject import minimal_method_spec
    from tests.helpers.state import make_state

    state = make_state(run_dir)
    spec = minimal_method_spec(paradigm_id="time_series_forecasting")
    spec["comparison"]["evaluation_protocol"] = (
        _typed_spec()["comparison"]["evaluation_protocol"]
    )
    protocol = spec["comparison"]["evaluation_protocol"]
    protocol["scheme"].update({
        "paper_value_status": "paper_stated",
        "paper_element_ids": ["chronological-split"],
    })
    for quantity in protocol["quantities"]:
        quantity["paper_element_ids"] = [f"value-{quantity['role']}"]
        quantity["axis_paper_element_ids"] = ["weekly-axis"]
    spec["data_requirements"] = {"synthetic_feasible": True}
    state.paths.method_spec.write_text(json.dumps(spec), encoding="utf-8")
    state.paths.paper_md.write_text(
        "We evaluate on public data: "
        "https://www.kaggle.com/datasets/berkayalan/retail-sales-data\n",
        encoding="utf-8")
    (run_dir / ".pipeline" / "provisional_packs" / "x").mkdir(
        parents=True, exist_ok=True)

    # Four native weekly steps against the typed 49-week coverage floor.
    payload = _zip({"sales.csv": _panel_csv(
        dates=4, rows_per_date=20, step_days=7,
    )})

    class _Response:
        status = 200
        headers = {"Content-Type": "application/zip",
                   "Content-Length": str(len(payload))}

        def __init__(self):
            self._stream = io.BytesIO(payload)

        def read(self, n=-1):
            return self._stream.read(n)

        def close(self):
            pass

    monkeypatch.setattr(dataset_acquisition, "_default_opener",
                        lambda url: _Response())
    monkeypatch.delenv("R2C_OFFLINE", raising=False)
    fake_subprocess.expect_script(returncode=0)
    fake_subprocess.expect_stage2_script(ok=True)

    result = run_stage_2a(state)

    assert_stage_completed(result, "stage_2a")
    example_data = run_dir / "method" / "example_data"
    assert not (example_data / "sales.csv").exists()
    assert (example_data / "series.csv").is_file()
    assert (example_data / "observations.csv").is_file()
    provenance = json.loads((example_data / "PROVENANCE.json").read_text())
    assert provenance["tier"] == "family_owned_synthetic"
    record = json.loads(
        (run_dir / ".pipeline" / "dataset_acquisition.json").read_text())
    refusal = record["bundle"]["refused"]
    assert refusal["realized_steps"] == 4
    assert refusal["protocol_floor_steps"] == 49
    assert record["status"] == "fallback_generated"
    assert record["offline_fallback"]["status"] == "generated"
    events = (run_dir / ".pipeline" / "run_events.jsonl").read_text()
    assert "demo_dataset_refused" in events
    assert "demo_dataset_fallback_generated" in events
    assert "demo_dataset_acquired" not in events


def test_stage_2a_axis_refusal_plus_fallback_refusal_has_no_acquired_state(
    fake_subprocess, run_dir, monkeypatch,
):
    import dataset_acquisition
    from run_pipeline import run_stage_2a
    from tests.helpers.assertions import assert_stage_completed
    from tests.helpers.inject import minimal_method_spec
    from tests.helpers.state import make_state

    state = make_state(run_dir)
    spec = minimal_method_spec(paradigm_id="time_series_forecasting")
    spec["comparison"]["evaluation_protocol"] = (
        _typed_spec()["comparison"]["evaluation_protocol"]
    )
    protocol = spec["comparison"]["evaluation_protocol"]
    protocol["scheme"].update({
        "paper_value_status": "paper_stated",
        "paper_element_ids": ["chronological-split"],
    })
    for quantity in protocol["quantities"]:
        quantity["paper_element_ids"] = [f"value-{quantity['role']}"]
        quantity["axis_paper_element_ids"] = ["weekly-axis"]
    spec["data_requirements"] = {"synthetic_feasible": False}
    state.paths.method_spec.write_text(json.dumps(spec), encoding="utf-8")
    state.paths.paper_md.write_text(
        "We evaluate on public data: "
        "https://www.kaggle.com/datasets/berkayalan/retail-sales-data\n",
        encoding="utf-8",
    )
    (run_dir / ".pipeline" / "provisional_packs" / "x").mkdir(
        parents=True, exist_ok=True,
    )
    payload = _zip({"sales.csv": _panel_csv(
        dates=4, rows_per_date=20, step_days=7,
    )})

    class _Response:
        status = 200
        headers = {
            "Content-Type": "application/zip",
            "Content-Length": str(len(payload)),
        }

        def __init__(self):
            self._stream = io.BytesIO(payload)

        def read(self, n=-1):
            return self._stream.read(n)

        def close(self):
            pass

    monkeypatch.setattr(
        dataset_acquisition, "_default_opener", lambda url: _Response(),
    )
    monkeypatch.delenv("R2C_OFFLINE", raising=False)
    fake_subprocess.expect_script(returncode=0)
    fake_subprocess.expect_stage2_script(ok=True)

    result = run_stage_2a(state)

    assert_stage_completed(result, "stage_2a")
    example_data = run_dir / "method" / "example_data"
    assert not (example_data / "sales.csv").exists()
    assert not (example_data / "series.csv").exists()
    assert not (example_data / "observations.csv").exists()
    assert not (example_data / "PROVENANCE.json").exists()
    record = json.loads(
        (run_dir / ".pipeline" / "dataset_acquisition.json").read_text()
    )
    assert record["status"] == "fallback_refused"
    assert record["bundle"]["refused"]["status"] == "infeasible"
    assert record["offline_fallback"]["status"] == "refused"
    assert record["offline_fallback"]["code"] == "offline_fallback_not_feasible"


def test_stage_2a_records_unresolved_cadence_without_deleting_bundle(
    fake_subprocess, run_dir, monkeypatch,
):
    import dataset_acquisition
    from run_pipeline import run_stage_2a
    from tests.helpers.assertions import assert_stage_completed
    from tests.helpers.inject import minimal_method_spec
    from tests.helpers.state import make_state

    state = make_state(run_dir)
    spec = minimal_method_spec(paradigm_id="time_series_forecasting")
    spec["comparison"]["evaluation_protocol"] = (
        _typed_spec()["comparison"]["evaluation_protocol"]
    )
    state.paths.method_spec.write_text(
        json.dumps(spec), encoding="utf-8",
    )
    state.paths.paper_md.write_text(
        "We evaluate on public data: "
        "https://www.kaggle.com/datasets/berkayalan/retail-sales-data\n",
        encoding="utf-8",
    )
    (run_dir / ".pipeline" / "provisional_packs" / "x").mkdir(
        parents=True, exist_ok=True,
    )
    payload = _zip({"sales.csv": _panel_csv(
        dates=60, rows_per_date=2, step_days=1,
    )})

    class _Response:
        status = 200
        headers = {
            "Content-Type": "application/zip",
            "Content-Length": str(len(payload)),
        }

        def __init__(self):
            self._stream = io.BytesIO(payload)

        def read(self, n=-1):
            return self._stream.read(n)

        def close(self):
            pass

    monkeypatch.setattr(
        dataset_acquisition, "_default_opener", lambda url: _Response(),
    )
    monkeypatch.delenv("R2C_OFFLINE", raising=False)
    fake_subprocess.expect_script(returncode=0)
    fake_subprocess.expect_stage2_script(ok=True)

    result = run_stage_2a(state)

    assert_stage_completed(result, "stage_2a")
    example_data = run_dir / "method" / "example_data"
    assert (example_data / "sales.csv").is_file()
    assert (example_data / "PROVENANCE.json").is_file()
    record = json.loads(
        (run_dir / ".pipeline" / "dataset_acquisition.json").read_text()
    )
    assessment = record["bundle"]["axis_feasibility"]
    assert assessment["status"] == "unresolved"
    assert assessment["protocol_floor_steps"] == 49
    assert assessment["source_unit"] == "day"
    assert "refused" not in record["bundle"]
    assert "offline_fallback" not in record
    events = (state.paths.pipeline_dir / "run_events.jsonl").read_text(
        encoding="utf-8"
    )
    assert "demo_dataset_axis_unresolved" in events
    assert "demo_dataset_refused" not in events
    assert "demo_dataset_acquired" in events


def test_the_briefing_carries_the_axis_and_the_column_warts(tmp_path):
    """R2C-062 plus the axis facts: the two smoke failures the 2026-08-05
    roll spent fix cycles on were both predictable from information already
    on disk at section-build time."""
    from run_pipeline import _bundled_demo_data_section

    example_data = tmp_path / "method" / "example_data"
    example_data.mkdir(parents=True)
    (example_data / "sales.csv").write_text(
        "product_id,date,sales,size\n"
        "P1,2017-01-02,4.0,\n"
        "P2,2017-01-03,5.0,M\n", encoding="utf-8")
    (example_data / "PROVENANCE.json").write_text(json.dumps({
        "source": {"cited_url": "https://example.org/data"},
        "files": [{"file": "sales.csv", "rows_kept": 2, "truncated": True,
                   "time_axis": {"column": "date", "steps_kept": 2,
                                 "steps_in_source": 900,
                                 "first_step": "2017-01-02",
                                 "last_step": "2017-01-03",
                                 "rows_per_step_cap": 1}}],
    }), encoding="utf-8")

    class _Paths:
        run_dir = tmp_path

    section = _bundled_demo_data_section(_Paths())
    assert section is not None
    # The mechanical contract, which the pandas prior got wrong.
    assert "column name -> np.ndarray" in section
    assert ".groupby" in section
    # The axis, which nothing had ever stated.
    assert "column `date` carries 2 distinct steps" in section
    assert "out of 900 in the source" in section
    # The warts, which a bare float() cast died on.
    assert "column `size`: text (e.g. 'M'), 1 of the first 2 rows empty" in section


def test_the_notebook_generator_gets_the_same_facts_as_the_coder(tmp_path):
    """R2C-062 reopened (night3): `_notebook_bundle_section` borrowed the
    coder section's facts by string-filtering its RENDERED lines for the
    `"  - "` prefix, which dropped every indented continuation line — the
    dtypes, the axis, the return contract. Three of the roll's four smoke
    failures belonged to that consumer, and the first (`float('cluster_6')`)
    is the dropped dtype line verbatim. The two sections must share the
    BUILT body, so every measured fact asserts here on the notebook side."""
    from run_pipeline import _bundled_demo_data_section, _notebook_bundle_section

    example_data = tmp_path / "method" / "example_data"
    example_data.mkdir(parents=True)
    (example_data / "sales.csv").write_text(
        "cluster_id,date,sales,size\n"
        "cluster_5,2017-01-02,4.0,\n"
        "cluster_6,2017-01-03,5.0,M\n", encoding="utf-8")
    (example_data / "PROVENANCE.json").write_text(json.dumps({
        "source": {"cited_url": "https://example.org/data"},
        "files": [{"file": "sales.csv", "rows_kept": 2, "truncated": True,
                   "time_axis": {"column": "date", "steps_kept": 2,
                                 "steps_in_source": 900,
                                 "first_step": "2017-01-02",
                                 "last_step": "2017-01-03",
                                 "rows_per_step_cap": 1}}],
    }), encoding="utf-8")

    class _Paths:
        run_dir = tmp_path

    section = _notebook_bundle_section(_Paths())
    assert section is not None
    # The return contract with its usage example.
    assert "column name -> np.ndarray" in section
    assert "tables = load_data()" in section
    # The dtype line whose absence cost the first smoke iteration.
    assert "column `cluster_id`: text (e.g. 'cluster_5'), 0 of the first 2 rows empty" in section
    # The axis facts.
    assert "column `date` carries 2 distinct steps" in section
    # The framing stays the notebook generator's own.
    assert "the demo MUST run on it" in section
    # The coder-only paragraph does not leak into the notebook section.
    assert "arch_contract.json" not in section
    # And the two consumers carry the identical built facts block.
    coder = _bundled_demo_data_section(_Paths())
    fact_lines = [ln for ln in coder.splitlines()
                  if ln.startswith(("  - ", "      "))]
    assert fact_lines and all(ln in section for ln in fact_lines)


@pytest.mark.manual_only
def test_the_whole_delivered_fleet_flags_no_new_clamp_class():
    """Noise floor, measured rather than assumed: every delivered method.py
    in the tree, against its own spec, flags nothing outside the one known
    bad (the 2026-08-05 day roll's `max_epochs` clamp). Run dirs get
    archived under `_N` suffixes and deleted before re-rolls, so the pin is
    on the defect class, never on a dir name or an exact count — the
    positive detection is pinned by the synthetic fixtures above. Corpus:
    the live fleet plus the committed example_runs/ floor; empty corpus
    skips instead of passing vacuously (W3 verdicts §5)."""
    import ast
    import re

    from scripts.validate_method_coder_output import (
        _paper_stated_param_tokens, _silently_clamped_paper_values,
    )

    scanned = 0
    flagged: list[str] = []
    for root in (Path("r2c_runs"), Path("example_runs")):
        for method_py in sorted(root.glob("*/method/method.py")):
            run_dir = method_py.parent.parent
            spec_path = run_dir / ".pipeline" / "method_spec.json"
            if not spec_path.is_file():
                continue
            # DELIVERED fleet only: an explanation-only package declares
            # its code artifacts unreliable in its own REPORT, so its
            # method.py is not a delivered surface. Without this filter a
            # mid-pipeline halted run parked on disk (the 2026-08-24
            # bayesian roll, halted at stage 2x) reads as fleet drift.
            manifest_path = run_dir / "details" / "final_manifest.json"
            if manifest_path.is_file():
                try:
                    manifest = json.loads(
                        manifest_path.read_text(encoding="utf-8"))
                except (OSError, ValueError):
                    manifest = {}
                label = ((manifest.get("delivery") or {}).get("label")
                         or "")
                if label == "explanation_only":
                    continue
            try:
                spec = json.loads(spec_path.read_text(encoding="utf-8"))
                tree = ast.parse(method_py.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            scanned += 1
            for _, name, _, _ in _silently_clamped_paper_values(
                    tree, _paper_stated_param_tokens(spec)):
                slug = re.sub(r"_\d+$", "", method_py.parent.parent.name)
                flagged.append(f"{slug}:{name}")

    if scanned == 0:
        pytest.skip("live fleet absent")
    known_bad = {
        "probabilistic-demand-forecasting-with-graph-neural-networks"
        ":max_epochs"
    }
    assert set(flagged) <= known_bad, flagged
