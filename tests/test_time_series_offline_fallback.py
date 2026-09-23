"""Focused deterministic acceptance for R2C-052's closed TSF fallback."""

from __future__ import annotations

import copy
import csv
import hashlib
import importlib.util
import json
from datetime import datetime
from importlib.machinery import SourceFileLoader
from pathlib import Path

import numpy as np
import pytest

import scripts.time_series_offline_fallback as fallback
from scripts import build_plan
from scripts.bundle_axis_floor import check_bundle_axis, protocol_axis_floor
from scripts.time_series_offline_fallback import (
    OBSERVATIONS_FILENAME,
    PROVENANCE_FILENAME,
    SERIES_FILENAME,
    TRANSACTION_FILENAME,
    generate_time_series_offline_fallback,
)


def _column(name: str, dtype: str, role: str) -> dict:
    return {"name": name, "dtype": dtype, "semantic_role": role}


def _build_plan(*, horizon: int = 4) -> dict:
    return {"offline_demo_data": {
        "schema_id": "time_series_panel_v1",
        "synthetic_feasible": True,
        "generator_id": "time_series_offline_fallback",
        "generator_version": "1.0.0",
        "entity_count": 8,
        "family_fitting_prefix_steps": 24,
        "supported_cadences": ["week", "day", "hour", "minute", "second"],
        "forecast_horizon_demo_value": horizon,
        "tables": {
            "series.csv": {
                "columns": [
                    _column("series_id", "string", "entity_id"),
                    _column(
                        "static_level", "float64", "static_numeric_level",
                    ),
                    _column(
                        "static_amplitude", "float64",
                        "static_numeric_amplitude",
                    ),
                ],
                "primary_key": ["series_id"],
            },
            "observations.csv": {
                "columns": [
                    _column("series_id", "string", "entity_id"),
                    _column(
                        "timestamp", "datetime_iso8601_utc", "time_index",
                    ),
                    _column("target", "float64", "target"),
                    _column(
                        "season_sin", "float64",
                        "known_time_varying_covariate",
                    ),
                    _column(
                        "season_cos", "float64",
                        "known_time_varying_covariate",
                    ),
                    _column(
                        "time_fraction", "float64",
                        "known_time_varying_covariate",
                    ),
                ],
                "primary_key": ["series_id", "timestamp"],
            },
        },
    }}


def _quantity(
    role: str,
    value: int | float | None,
    *,
    status: str = "paper_stated",
    unit: str = "week",
    granularity: int | float = 1,
) -> dict:
    return {
        "role": role,
        "parameter_name": (
            "context_length" if role == "context_length"
            else "forecast_horizon" if role == "forecast_call_horizon"
            else None
        ),
        "paper_names": [role.replace("_", " ")],
        "paper_symbols": ["K"] if role == "forecast_call_horizon" else [],
        "value": value,
        "unit": unit,
        "granularity": granularity,
        "axis_evidence_quote": f"Observations occur every {granularity} {unit}.",
        "axis_paper_section": "Data",
        "axis_paper_element_ids": [f"axis-{role}"],
        "paper_value_status": status,
        "evidence_quote": f"The paper states the {role.replace('_', ' ')}.",
        "paper_section": "Evaluation",
        "paper_element_ids": [f"value-{role}"],
    }


def _spec(
    *,
    context: int | float = 10,
    horizon: int | float | None = None,
    validation: int | float = 13,
    test: int | float = 26,
    unit: str = "week",
    granularity: int | float = 1,
) -> dict:
    return {
        "comparison": {
            "classification": {"id": "time_series_forecasting"},
            "pluggable_component": {
                "name": "forecast",
                "signature": (
                    "forecast(model, history, static_features, "
                    "time_varying_features, graph, entity_ids, "
                    "target_scaling_state, seed) -> ForecastResult"
                ),
                "seed_param": "seed",
            },
            "evaluation_protocol": {
                "scheme": {
                    "kind": "single_holdout",
                    "paper_value_status": "paper_stated",
                    "paper_element_ids": ["evaluation-split"],
                },
                "quantities": [
                    _quantity(
                        "context_length", context, unit=unit,
                        granularity=granularity,
                    ),
                    _quantity(
                        "forecast_call_horizon", horizon,
                        status=(
                            "paper_stated"
                            if horizon is not None else "paper_unspecified"
                        ),
                        unit=unit,
                        granularity=granularity,
                    ),
                    _quantity(
                        "validation_span", validation, unit=unit,
                        granularity=granularity,
                    ),
                    _quantity(
                        "test_span", test, unit=unit,
                        granularity=granularity,
                    ),
                ],
            },
        },
        "data_requirements": {"synthetic_feasible": True},
    }


def _attempts() -> list[dict]:
    return [{
        "host": None,
        "cited_url": None,
        "status": "unavailable",
        "reason": "public acquisition raised before a source was resolved",
        "http_code": None,
    }]


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _assert_no_bundle(destination: Path) -> None:
    assert all(
        not (destination / name).exists()
        for name in (SERIES_FILENAME, OBSERVATIONS_FILENAME, PROVENANCE_FILENAME)
    )


def test_generates_canonical_full_panel_with_exhaustive_provenance(tmp_path):
    destination = tmp_path / "example_data"

    outcome = generate_time_series_offline_fallback(
        _spec(), _build_plan(), destination, seed=17,
        acquisition_attempts=_attempts(),
    )

    assert outcome.status == "generated", outcome.as_dict()
    assert outcome.written_files == (
        "series.csv", "observations.csv", "PROVENANCE.json",
    )
    series = _read_csv(destination / SERIES_FILENAME)
    observations = _read_csv(destination / OBSERVATIONS_FILENAME)
    assert list(series[0]) == [
        "series_id", "static_level", "static_amplitude",
    ]
    assert list(observations[0]) == [
        "series_id", "timestamp", "target", "season_sin", "season_cos",
        "time_fraction",
    ]
    ids = [f"series_{index:03d}" for index in range(8)]
    assert [row["series_id"] for row in series] == ids
    assert len(observations) == 8 * 73
    assert [
        (row["series_id"], row["timestamp"]) for row in observations
    ] == sorted(
        (row["series_id"], row["timestamp"]) for row in observations
    )
    for series_id in ids:
        rows = [row for row in observations if row["series_id"] == series_id]
        assert len(rows) == 73
        parsed = [
            datetime.fromisoformat(row["timestamp"].replace("Z", "+00:00"))
            for row in rows
        ]
        assert len({right - left for left, right in zip(parsed, parsed[1:])}) == 1

    manifest = json.loads(
        (destination / PROVENANCE_FILENAME).read_text(encoding="utf-8")
    )
    assert set(manifest) == {
        "schema_version", "schema_id", "tier", "generator", "files",
        "bundle_digest", "entity_ids", "entity_mapping", "cadence",
        "protocol_capacity", "generation_mechanics",
        "acquisition_attempts", "honesty_note",
    }
    assert manifest["schema_version"] == "1.0.0"
    assert manifest["schema_id"] == "time_series_panel_v1"
    assert manifest["tier"] == "family_owned_synthetic"
    assert manifest["entity_ids"] == ids
    assert manifest["entity_mapping"] == [
        {"position": index, "series_id": series_id}
        for index, series_id in enumerate(ids)
    ]
    capacity = manifest["protocol_capacity"]
    assert capacity["context_steps"] == 10
    assert capacity["forecast_horizon_steps"] == 4
    assert capacity["validation_steps"] == 13
    assert capacity["test_steps"] == 26
    assert capacity["static_axis_floor_steps"] == 49
    assert capacity["family_fitting_prefix_steps"] == 24
    assert capacity["usable_steps"] == 73
    assert capacity["static_axis_formula"] == "C + V + T"
    assert capacity["usable_formula"] == (
        "family_fitting_prefix_steps + C + V + T"
    )
    assert capacity["forecast_call_checks"] == {
        "K_lte_validation_span": True,
        "K_lte_test_span": True,
        "C_plus_K_lte_usable_axis": True,
    }
    assert "fitting_capacity_steps" not in capacity
    assert "fitting_formula" not in capacity
    assert capacity["capacity_only"] is True
    assert "does not certify" in capacity["capacity_limit"]
    horizon = next(
        item for item in capacity["quantities"]
        if item["role"] == "forecast_call_horizon"
    )
    assert horizon["paper_value_status"] == "paper_unspecified"
    assert horizon["raw_value"] is None
    assert horizon["value_source"] == (
        "build_plan.offline_demo_data.forecast_horizon_demo_value"
    )
    assert manifest["acquisition_attempts"] == _attempts()
    assert "not the paper's dataset" in manifest["honesty_note"]
    assert "not paper-comparable or benchmark evidence" in manifest["honesty_note"]

    files = {entry["file"]: entry for entry in manifest["files"]}
    assert set(files) == {"series.csv", "observations.csv"}
    assert files["series.csv"]["rows_kept"] == 8
    assert files["observations.csv"]["rows_kept"] == 8 * 73
    assert all(entry["truncated"] is False for entry in files.values())
    axis = files["observations.csv"]["time_axis"]
    assert axis["steps_kept"] == axis["live_steps"] == 73
    assert axis["source_cadence"] == {"unit": "week", "granularity": 1}
    assert axis["first_step"] == manifest["cadence"]["first_timestamp"]
    assert axis["last_live_step"] == manifest["cadence"]["last_timestamp"]
    for filename, entry in files.items():
        payload = (destination / filename).read_bytes()
        assert entry["sha256"] == hashlib.sha256(payload).hexdigest()
        assert entry["bytes"] == len(payload)
    digest = manifest.pop("bundle_digest")
    canonical = json.dumps(
        manifest, sort_keys=True, separators=(",", ":"),
        ensure_ascii=False, allow_nan=False,
    )
    assert digest == hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def test_same_seed_is_byte_identical_and_different_seed_changes_only_jitter(
    tmp_path,
):
    destinations = [tmp_path / name for name in ("a", "b", "c")]
    first = generate_time_series_offline_fallback(
        _spec(), _build_plan(), destinations[0], seed=91,
        acquisition_attempts=[],
    )
    second = generate_time_series_offline_fallback(
        _spec(), _build_plan(), destinations[1], seed=91,
        acquisition_attempts=[],
    )
    third = generate_time_series_offline_fallback(
        _spec(), _build_plan(), destinations[2], seed=92,
        acquisition_attempts=[],
    )
    assert first.generated and second.generated and third.generated
    for name in (SERIES_FILENAME, OBSERVATIONS_FILENAME, PROVENANCE_FILENAME):
        assert (destinations[0] / name).read_bytes() == (
            destinations[1] / name
        ).read_bytes()
    assert (destinations[0] / SERIES_FILENAME).read_bytes() == (
        destinations[2] / SERIES_FILENAME
    ).read_bytes()
    assert (destinations[0] / OBSERVATIONS_FILENAME).read_bytes() != (
        destinations[2] / OBSERVATIONS_FILENAME
    ).read_bytes()
    assert first.provenance["bundle_digest"] != third.provenance["bundle_digest"]


def test_committed_tsf_loader_and_axis_consumers_accept_the_generated_bundle(
    tmp_path,
):
    spec = _spec()
    resolved = build_plan.load_build_plan(spec)
    assert resolved is not None
    destination = tmp_path / "example_data"
    outcome = generate_time_series_offline_fallback(
        spec, resolved, destination, seed=5, acquisition_attempts=[],
    )
    assert outcome.generated, outcome.as_dict()

    template = (
        Path(__file__).resolve().parents[1]
        / "paradigms/time_series_forecasting/templates/method/data.py.template"
    )
    loader = SourceFileLoader("r2c_tsf_data_template", str(template))
    module_spec = importlib.util.spec_from_loader(loader.name, loader)
    assert module_spec is not None
    module = importlib.util.module_from_spec(module_spec)
    loader.exec_module(module)
    loaded = module.load_data(path=destination)

    assert set(loaded) == {"series", "observations"}
    assert set(loaded["series"]) == {
        "series_id", "static_level", "static_amplitude",
    }
    assert set(loaded["observations"]) == {
        "series_id", "timestamp", "target", "season_sin", "season_cos",
        "time_fraction",
    }
    assert loaded["series"]["series_id"].dtype.kind in {"U", "S"}
    assert loaded["observations"]["series_id"].dtype.kind in {"U", "S"}
    assert loaded["observations"]["timestamp"].dtype.kind in {"U", "S"}
    for name in ("static_level", "static_amplitude"):
        assert loaded["series"][name].dtype.kind == "f"
        assert np.isfinite(loaded["series"][name]).all()
    for name in ("target", "season_sin", "season_cos", "time_fraction"):
        assert loaded["observations"][name].dtype.kind == "f"
        assert np.isfinite(loaded["observations"][name]).all()
    assert np.ptp(loaded["observations"]["target"]) > 1.0

    manifest = json.loads(
        (destination / PROVENANCE_FILENAME).read_text(encoding="utf-8")
    )
    requirement = protocol_axis_floor(spec)
    assert requirement is not None
    verdict = check_bundle_axis(manifest, requirement)
    assert verdict is not None
    assert verdict.feasible is True
    assert verdict.realized_steps == 73
    assert verdict.table == "observations.csv"


def test_resolved_k_is_dynamic_but_never_changes_the_static_axis_floor(tmp_path):
    system = generate_time_series_offline_fallback(
        _spec(), _build_plan(horizon=3), tmp_path / "system", seed=0,
        acquisition_attempts=[],
    )
    paper = generate_time_series_offline_fallback(
        _spec(horizon=2), _build_plan(horizon=3), tmp_path / "paper", seed=0,
        acquisition_attempts=[],
    )

    assert system.generated and paper.generated
    assert system.provenance["protocol_capacity"][
        "forecast_horizon_steps"
    ] == 3
    assert system.provenance["protocol_capacity"][
        "static_axis_floor_steps"
    ] == 49
    assert system.provenance["protocol_capacity"]["usable_steps"] == 73
    assert paper.provenance["protocol_capacity"][
        "forecast_horizon_steps"
    ] == 2
    assert paper.provenance["protocol_capacity"][
        "static_axis_floor_steps"
    ] == 49
    assert paper.provenance["protocol_capacity"]["usable_steps"] == 73
    paper_horizon = next(
        item for item in paper.provenance["protocol_capacity"]["quantities"]
        if item["role"] == "forecast_call_horizon"
    )
    assert paper_horizon["value_source"] == (
        "method_spec.comparison.evaluation_protocol"
    )


def test_numerically_equal_granularities_share_one_typed_cadence(tmp_path):
    spec = _spec(
        context=5, horizon=2, validation=4, test=6,
        unit="hour", granularity=0.5,
    )
    granularities = [1 / 2, 0.50, 0.5, 0.500]
    for quantity, granularity in zip(
        spec["comparison"]["evaluation_protocol"]["quantities"],
        granularities,
    ):
        quantity["granularity"] = granularity

    outcome = generate_time_series_offline_fallback(
        spec, _build_plan(), tmp_path / "half-hour", seed=4,
        acquisition_attempts=[],
    )

    assert outcome.generated, outcome.as_dict()
    capacity = outcome.provenance["protocol_capacity"]
    assert capacity["context_steps"] == 10
    assert capacity["forecast_horizon_steps"] == 4
    assert capacity["validation_steps"] == 8
    assert capacity["test_steps"] == 12
    assert outcome.provenance["cadence"]["step"] == 0.5


@pytest.mark.parametrize(
    ("mutator", "code"),
    [
        (
            lambda spec, plan: plan["offline_demo_data"].update(
                schema_id="generic_arrays_v1"
            ),
            "offline_fallback_schema_unsupported",
        ),
        (
            lambda spec, plan: spec["data_requirements"].update(
                synthetic_feasible=False
            ),
            "offline_fallback_not_feasible",
        ),
        (
            lambda spec, plan: spec["comparison"]["classification"].update(
                id="active_learning"
            ),
            "offline_fallback_family_unsupported",
        ),
        (
            lambda spec, plan: spec["comparison"]["evaluation_protocol"][
                "scheme"
            ].update(kind="rolling_origin"),
            "offline_fallback_protocol_unsupported",
        ),
        (
            lambda spec, plan: spec["comparison"]["evaluation_protocol"][
                "quantities"
            ].pop(),
            "offline_fallback_protocol_roles",
        ),
        (
            lambda spec, plan: [
                quantity.update(unit="month")
                for quantity in spec["comparison"]["evaluation_protocol"][
                    "quantities"
                ]
            ],
            "offline_fallback_cadence_unsupported",
        ),
        (
            lambda spec, plan: plan["offline_demo_data"]["tables"][
                "observations.csv"
            ]["columns"][2].update(name="value"),
            "offline_fallback_table_schema",
        ),
        (
            lambda spec, plan: plan["offline_demo_data"].update(entity_count=9),
            "offline_fallback_entity_contract",
        ),
    ],
)
def test_unsupported_family_contracts_refuse_without_writes(
    tmp_path, mutator, code,
):
    spec = _spec()
    plan = _build_plan()
    mutator(spec, plan)
    destination = tmp_path / code

    outcome = generate_time_series_offline_fallback(
        spec, plan, destination, seed=0, acquisition_attempts=[],
    )

    assert outcome.status == "refused"
    assert outcome.code == code
    _assert_no_bundle(destination)


@pytest.mark.parametrize(
    ("spec", "code"),
    [
        (
            _spec(horizon=5, validation=4, test=8),
            "offline_fallback_horizon_exceeds_validation",
        ),
        (
            _spec(horizon=5, validation=8, test=4),
            "offline_fallback_horizon_exceeds_test",
        ),
        (
            _spec(context=10.5),
            "offline_fallback_protocol_nonintegral",
        ),
    ],
)
def test_protocol_arithmetic_refuses_without_writes(tmp_path, spec, code):
    destination = tmp_path / code

    outcome = generate_time_series_offline_fallback(
        spec, _build_plan(), destination, seed=0, acquisition_attempts=[],
    )

    assert outcome.code == code
    _assert_no_bundle(destination)


def test_timestamp_overflow_is_a_typed_refusal_before_generation(tmp_path):
    enormous = 10**20
    spec = _spec(
        context=enormous,
        horizon=enormous,
        validation=enormous,
        test=enormous,
        unit="second",
        granularity=enormous,
    )
    destination = tmp_path / "overflow"

    outcome = generate_time_series_offline_fallback(
        spec, _build_plan(), destination, seed=0, acquisition_attempts=[],
    )

    assert outcome.code == "offline_fallback_timestamp_overflow"
    _assert_no_bundle(destination)


def test_malformed_attempt_trail_refuses_without_inventing_evidence(tmp_path):
    destination = tmp_path / "bad-attempt"
    attempt = _attempts()[0]
    attempt.pop("reason")

    outcome = generate_time_series_offline_fallback(
        _spec(), _build_plan(), destination, seed=0,
        acquisition_attempts=[attempt],
    )

    assert outcome.code == "offline_fallback_contract_shape"
    _assert_no_bundle(destination)


def test_destination_conflict_is_preserved_and_never_overwritten(tmp_path):
    destination = tmp_path / "existing"
    destination.mkdir()
    sentinel = destination / SERIES_FILENAME
    sentinel.write_text("researcher data\n", encoding="utf-8")

    outcome = generate_time_series_offline_fallback(
        _spec(), _build_plan(), destination, seed=0,
        acquisition_attempts=[],
    )

    assert outcome.code == "offline_fallback_destination_conflict"
    assert sentinel.read_text(encoding="utf-8") == "researcher data\n"
    assert not (destination / OBSERVATIONS_FILENAME).exists()
    assert not (destination / PROVENANCE_FILENAME).exists()


def test_publish_failure_rolls_back_all_generator_owned_files(
    tmp_path, monkeypatch,
):
    destination = tmp_path / "existing-docs"
    destination.mkdir()
    readme = destination / "README.md"
    readme.write_text("keep me\n", encoding="utf-8")
    real_replace = fallback.os.replace
    calls = 0

    def fail_second(source, target):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("injected publish failure")
        return real_replace(source, target)

    monkeypatch.setattr(fallback.os, "replace", fail_second)

    outcome = generate_time_series_offline_fallback(
        _spec(), _build_plan(), destination, seed=0,
        acquisition_attempts=[],
    )

    assert outcome.code == "offline_fallback_write_failed"
    _assert_no_bundle(destination)
    assert readme.read_text(encoding="utf-8") == "keep me\n"


def test_post_rename_publish_error_rolls_back_the_moved_synthetic_file(
    tmp_path, monkeypatch,
):
    destination = tmp_path / "method" / "example_data"
    real_replace = fallback.os.replace
    calls = 0

    def fail_after_first_bundle_rename(source, target):
        nonlocal calls
        calls += 1
        real_replace(source, target)
        if calls == 2:  # marker, then series.csv
            raise OSError("injected post-rename failure")

    monkeypatch.setattr(
        fallback.os, "replace", fail_after_first_bundle_rename,
    )

    outcome = generate_time_series_offline_fallback(
        _spec(), _build_plan(), destination, seed=0,
        acquisition_attempts=[],
        staging_parent=tmp_path / ".pipeline" / "_dataset_staging",
    )

    assert outcome.code == "offline_fallback_write_failed"
    _assert_no_bundle(destination)
    assert not (destination / TRANSACTION_FILENAME).exists()
    assert not any((tmp_path / "method").glob(".r2c-tsf-offline-*"))


def test_process_interruption_marker_recovers_only_owned_partial_files(
    tmp_path, monkeypatch,
):
    destination = tmp_path / "interrupted"
    real_replace = fallback.os.replace
    calls = 0

    def interrupt_after_first_bundle_file(source, target):
        nonlocal calls
        calls += 1
        if calls == 3:  # marker, series.csv, then interruption
            raise KeyboardInterrupt("injected process interruption")
        return real_replace(source, target)

    monkeypatch.setattr(
        fallback.os, "replace", interrupt_after_first_bundle_file,
    )
    with pytest.raises(KeyboardInterrupt, match="process interruption"):
        generate_time_series_offline_fallback(
            _spec(), _build_plan(), destination, seed=0,
            acquisition_attempts=[],
        )

    assert (destination / TRANSACTION_FILENAME).is_file()
    assert (destination / SERIES_FILENAME).is_file()
    assert not (destination / OBSERVATIONS_FILENAME).exists()
    monkeypatch.setattr(fallback.os, "replace", real_replace)

    recovered = generate_time_series_offline_fallback(
        _spec(), _build_plan(), destination, seed=0,
        acquisition_attempts=[],
    )

    assert recovered.generated, recovered.as_dict()
    assert not (destination / TRANSACTION_FILENAME).exists()
    assert all((destination / name).is_file() for name in (
        SERIES_FILENAME, OBSERVATIONS_FILENAME, PROVENANCE_FILENAME,
    ))


def test_interruption_recovery_preserves_a_changed_partial_file(
    tmp_path, monkeypatch,
):
    destination = tmp_path / "changed-interrupted"
    real_replace = fallback.os.replace
    calls = 0

    def interrupt_after_first_bundle_file(source, target):
        nonlocal calls
        calls += 1
        if calls == 3:
            raise KeyboardInterrupt("injected process interruption")
        return real_replace(source, target)

    monkeypatch.setattr(
        fallback.os, "replace", interrupt_after_first_bundle_file,
    )
    with pytest.raises(KeyboardInterrupt):
        generate_time_series_offline_fallback(
            _spec(), _build_plan(), destination, seed=0,
            acquisition_attempts=[],
        )
    changed = destination / SERIES_FILENAME
    changed.write_text("researcher changed this file\n", encoding="utf-8")
    monkeypatch.setattr(fallback.os, "replace", real_replace)

    refused = generate_time_series_offline_fallback(
        _spec(), _build_plan(), destination, seed=0,
        acquisition_attempts=[],
    )

    assert refused.code == "offline_fallback_transaction_conflict"
    assert changed.read_text(encoding="utf-8") == "researcher changed this file\n"
    assert (destination / TRANSACTION_FILENAME).is_file()
    assert not (destination / OBSERVATIONS_FILENAME).exists()
