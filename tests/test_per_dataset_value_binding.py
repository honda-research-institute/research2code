"""Per-dataset paper-value binding (per-dataset-value-binding-design.md).

The live case (GBALD, 7/2 night): the paper states initial labeled sets of
"20, 1000, 1000 random samples" for MNIST/SVHN/CIFAR-10; the analyzer bound
1000 (SVHN) while the demo runs MNIST, and the misbound value read as a
confident paper claim for the wrong dataset. These tests pin the design's
four acceptance criteria: bind-by-demo-dataset, old-spec tolerance, the
honest scalar fallback, and the unused-by-method disclosure.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from schemas.method_spec import DataSetup
from schemas.params import Params
from scripts.derive_params import (
    _detect_demo_dataset,
    _mark_unused_by_method,
    _resolve_per_dataset_bindings,
    derive,
)
from scripts.validate_params_provenance import paper_claim_findable

GBALD_MAP = {"initial_labeled": {"MNIST": 20, "SVHN": 1000, "CIFAR-10": 1000}}

GBALD_PAPER_TEXT = (
    "We evaluate on MNIST, SVHN, and CIFAR-10 with initial labeled sets of "
    "20, 1000, 1000 random samples respectively (Section 7.4). Each round "
    "acquires 100 labels."
)


def _al_spec(*, per_dataset_values: dict | None, initial_labeled: int = 1000) -> dict:
    return {
        "comparison": {
            "classification": {"id": "active_learning/bayesian"},
            "pluggable_component": {
                "signature": "select_batch(model, x_unlabeled, batch_size, seed)",
                "seed_param": "seed",
            },
        },
        "critical_requirements": {
            "data_setup": {
                "initial_labeled": initial_labeled,
                "batch_size": 100,
                "total_budget": 10000,
                "num_rounds": 90,
                "benchmark_name": "SVHN main benchmark",
                "paper_section": "Section 7.4",
                **(
                    {"per_dataset_values": per_dataset_values}
                    if per_dataset_values is not None
                    else {}
                ),
            },
        },
    }


def _run_dir_with_method(tmp_path: Path, data_py: str) -> Path:
    run_dir = tmp_path / "run"
    method = run_dir / "method"
    method.mkdir(parents=True)
    (method / "data.py").write_text(data_py, encoding="utf-8")
    (method / "method.py").write_text(
        "def select_batch(model, x_unlabeled, batch_size, seed):\n"
        "    return list(range(batch_size))\n",
        encoding="utf-8",
    )
    return run_dir


# --- criterion 1: the GBALD-shaped fixture binds the demo dataset's value ---

def test_gbald_shaped_fixture_binds_demo_dataset_value(tmp_path):
    run_dir = _run_dir_with_method(
        tmp_path,
        "# Falls through to an MNIST download when no user data is present.\n"
        "def load_data(pool_size=800):\n"
        "    return _download_mnist(pool_size)\n",
    )
    output = derive(_al_spec(per_dataset_values=GBALD_MAP), run_dir=run_dir)

    entry = output["params"]["initial_labeled"]
    assert entry["value"] == 20
    assert entry["bound_dataset"] == "MNIST"
    record = output["per_dataset_binding"]
    assert record["demo_dataset"] == "MNIST"
    assert record["bound"][0]["param"] == "initial_labeled"
    assert record["fallbacks"] == []
    # The bound value is the findability check's subject and it IS in the
    # paper (today's misbound 1000 also was — the point is the value now
    # names the demo's dataset).
    assert paper_claim_findable("initial_labeled", entry["value"], GBALD_PAPER_TEXT)
    # The full output stays schema-valid.
    Params.model_validate(output)


# --- criterion 2: old specs without the field validate unchanged ------------

def test_old_spec_without_map_validates_and_derives_unchanged(tmp_path):
    ds = DataSetup(
        initial_labeled=1000, batch_size=100, total_budget=10000,
        num_rounds=90, paper_section="Section 7.4",
    )
    assert ds.per_dataset_values is None

    run_dir = _run_dir_with_method(tmp_path, "def load_data():\n    return None\n")
    output = derive(_al_spec(per_dataset_values=None), run_dir=run_dir)
    assert output["params"]["initial_labeled"]["value"] == 1000
    assert "bound_dataset" not in output["params"]["initial_labeled"]
    assert "per_dataset_binding" not in output


def test_scalar_contradicting_the_map_is_a_validation_error():
    with pytest.raises(ValidationError, match="contradicts"):
        DataSetup(
            initial_labeled=555, batch_size=100, total_budget=10000,
            num_rounds=90, paper_section="Section 7.4",
            per_dataset_values=GBALD_MAP,
        )


def test_unknown_param_key_in_map_is_a_validation_error():
    with pytest.raises(ValidationError, match="unknown parameter"):
        DataSetup(
            initial_labeled=1000, batch_size=100, total_budget=10000,
            num_rounds=90, paper_section="Section 7.4",
            per_dataset_values={"learning_rate": {"MNIST": 1}},
        )


# --- criterion 3: unbindable demo dataset falls back honestly, never halts --

def test_demo_dataset_absent_from_map_falls_back_with_record(tmp_path):
    run_dir = _run_dir_with_method(
        tmp_path,
        "def load_data():\n    return _synthetic_blobs()\n",  # names no dataset
    )
    output = derive(_al_spec(per_dataset_values=GBALD_MAP), run_dir=run_dir)

    entry = output["params"]["initial_labeled"]
    assert entry["value"] == 1000  # the spec's primary-dataset scalar
    assert "bound_dataset" not in entry
    record = output["per_dataset_binding"]
    assert record["demo_dataset"] is None
    assert record["bound"] == []
    [fallback] = record["fallbacks"]
    assert fallback["param"] == "initial_labeled"
    assert fallback["scalar_value"] == 1000
    Params.model_validate(output)


def test_ambiguous_dataset_mentions_fall_back_not_guess(tmp_path):
    run_dir = _run_dir_with_method(
        tmp_path,
        "# Supports MNIST and SVHN download fallbacks.\n"
        "def load_data():\n    return None\n",
    )
    _, record = _resolve_per_dataset_bindings(
        {"initial_labeled": 1000, "per_dataset_values": GBALD_MAP},
        run_dir,
    )
    assert record["demo_dataset"] is None
    assert record["fallbacks"][0]["param"] == "initial_labeled"


def test_detect_demo_dataset_without_run_dir_is_honest():
    dataset, reason = _detect_demo_dataset(["MNIST", "SVHN"], None)
    assert dataset is None
    assert "run-dir" in reason


# --- driver wiring: fallbacks log assumptions, clean binds only log events --

def test_driver_logs_one_assumption_per_fallback(tmp_path, monkeypatch):
    import json

    import run_pipeline
    from tests.helpers.state import make_state

    state = make_state(tmp_path / "run")
    captured: list[dict] = []
    monkeypatch.setattr(run_pipeline, "_next_assumption_id", lambda s: "A900")
    monkeypatch.setattr(
        run_pipeline, "_append_assumption",
        lambda state, **kw: captured.append(kw),
    )

    record = {
        "demo_dataset": None,
        "bound": [],
        "fallbacks": [{
            "param": "initial_labeled",
            "dataset_map": GBALD_MAP["initial_labeled"],
            "scalar_value": 1000,
            "reason": "no per-dataset map key is named by the generated package sources",
        }],
    }
    (state.paths.pipeline_dir / "params.json").write_text(
        json.dumps({"schema_version": "1.1.0", "params": {},
                    "per_dataset_binding": record}),
        encoding="utf-8",
    )
    run_pipeline._log_per_dataset_binding(state, "stage_2x")
    assert len(captured) == 1
    assert captured[0]["aid"] == "A900"
    assert "initial_labeled" in captured[0]["title"]

    # A clean bind logs no assumption.
    captured.clear()
    record["fallbacks"] = []
    record["bound"] = [{"param": "initial_labeled", "dataset": "MNIST",
                        "value": 20, "evidence": "method/data.py names MNIST"}]
    (state.paths.pipeline_dir / "params.json").write_text(
        json.dumps({"schema_version": "1.1.0", "params": {},
                    "per_dataset_binding": record}),
        encoding="utf-8",
    )
    run_pipeline._log_per_dataset_binding(state, "stage_2x")
    assert captured == []


# --- criterion 4: unused-by-method disclosure from derivation ---------------

def test_unused_by_method_marks_only_consumerless_params(tmp_path):
    run_dir = _run_dir_with_method(
        tmp_path,
        # pool_size has a consumer in the data loader; initial_labeled and
        # num_rounds have none anywhere in the package (the GBALD
        # core-set-bootstrap shape).
        "def load_data(pool_size=800):\n    return None\n",
    )
    spec = _al_spec(per_dataset_values=None)
    params = {
        "initial_labeled": {"value": 1000, "source": "paper",
                            "paper_section": "Section 7.4"},
        "batch_size": {"value": 100, "source": "paper",
                       "paper_section": "Section 7.4"},
        "pool_size": {"value": 800, "source": "system_default",
                      "paper_value": "full training set",
                      "reasoning": "smoke subsample"},
    }
    _mark_unused_by_method(params, spec, run_dir)

    # No consumer anywhere: disclosed.
    assert params["initial_labeled"].get("unused_by_method") is True
    # In the pluggable signature: a consumer.
    assert "unused_by_method" not in params["batch_size"]
    # In the generated data loader: a consumer.
    assert "unused_by_method" not in params["pool_size"]
    Params.model_validate({"schema_version": "1.1.0", "params": params})
