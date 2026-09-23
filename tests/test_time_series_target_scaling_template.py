"""Portable generated-package helper parity for R2C-089."""

from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

from scripts.build_plan import TSF_TARGET_SCALING
from scripts.scaffold_package import scaffold
from scripts.time_series_target_scaling import (
    FittingTargetRange,
    fit_target_scaling_state as oracle_fit,
    inverse_forecast_output as oracle_inverse,
    transform_targets as oracle_transform,
    validate_training_only_state,
)


ROOT = Path(__file__).resolve().parent.parent
FITTING_RANGE = {
    "validator": "eval_split_lineage",
    "target_root": "batch.targets",
    "model_id": "forecast-model",
    "start": 0,
    "stop": 3,
    "certainty": "exact",
}


def _spec() -> dict[str, object]:
    return {
        "schema_version": "1.2.0",
        "paper": {
            "title": "Target scaling template test",
            "authors": "R2C",
            "repo_url": None,
        },
        "core_method": {
            "name": "Scaling parity",
            "summary": "Exercise the portable scaling seam.",
            "type": "algorithm",
            "paper_sections": [],
            "key_elements": [],
        },
        "comparison": {
            "classification": {
                "id": "time_series_forecasting",
                "detection_reasoning": "template unit test",
            },
        },
    }


def _scaffolded_helper(tmp_path: Path) -> tuple[ModuleType, Path]:
    run_dir = tmp_path / "run"
    pipeline_dir = run_dir / ".pipeline"
    pipeline_dir.mkdir(parents=True)
    spec_path = pipeline_dir / "method_spec.json"
    spec_path.write_text(json.dumps(_spec()), encoding="utf-8")
    assert scaffold(spec_path, run_dir, ROOT) == 0

    helper_path = run_dir / "method" / "target_scaling.py"
    assert helper_path.is_file()
    module_spec = importlib.util.spec_from_file_location(
        f"generated_target_scaling_{tmp_path.name}", helper_path
    )
    assert module_spec is not None and module_spec.loader is not None
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    return module, run_dir


def _oracle_carrier() -> FittingTargetRange:
    return FittingTargetRange(**FITTING_RANGE)


def _receipt() -> dict[str, object]:
    return {
        "schema_version": "1.0.0",
        "status": "valid",
        "validator": "eval_split_lineage",
        "reasons": [],
        "evidence": {
            "fitting_target_ranges": [{
                "root": FITTING_RANGE["target_root"],
                "model_id": FITTING_RANGE["model_id"],
                "protocol_range": {
                    "start": FITTING_RANGE["start"],
                    "stop": FITTING_RANGE["stop"],
                },
                "subset": None,
                "certainty": "exact",
            }],
        },
    }


def _targets() -> np.ndarray:
    # The final two positions are held out and deliberately extreme.
    return np.asarray([
        [-1.0, 2.0, -3.0, 1.0e8, -1.0e8],
        [-100.0, 200.0, -300.0, -1.0e9, 1.0e9],
        [0.0, 0.0, 0.0, 7.0e8, 7.0e8],
        [0.0, 0.0, 6.0, 9.0e8, 9.0e8],
    ], dtype=np.float64)


def test_scaffolded_fit_matches_pipeline_oracle_without_writing_state(tmp_path: Path):
    helper, run_dir = _scaffolded_helper(tmp_path)
    ids = [10, "hundred", "zero", "intermittent"]
    targets = _targets()
    before = sorted(path.relative_to(run_dir) for path in run_dir.rglob("*"))

    state = helper.fit_target_scaling_state(ids, targets, FITTING_RANGE)
    oracle = oracle_fit(
        TSF_TARGET_SCALING,
        entity_ids=ids,
        targets=targets,
        fitting_range=_oracle_carrier(),
    )
    assert state == oracle
    assert validate_training_only_state(
        state,
        _receipt(),
        contract=TSF_TARGET_SCALING,
    )["state_digest"] == state["state_digest"]
    assert not (run_dir / ".pipeline" / "target_scaling_state.json").exists()
    assert sorted(path.relative_to(run_dir) for path in run_dir.rglob("*")) == before

    heldout_changed = targets.copy()
    heldout_changed[:, 3:] *= -12345.0
    assert helper.fit_target_scaling_state(
        ids, heldout_changed, FITTING_RANGE
    ) == state


def test_scaffolded_fit_and_transform_are_permutation_and_subset_safe(tmp_path: Path):
    helper, _ = _scaffolded_helper(tmp_path)
    ids = [10, "hundred", "zero", "intermittent"]
    targets = _targets()
    state = helper.fit_target_scaling_state(ids, targets, FITTING_RANGE)

    permutation = [2, 0, 3, 1]
    permuted = helper.fit_target_scaling_state(
        [ids[index] for index in permutation],
        targets[permutation],
        FITTING_RANGE,
    )
    assert permuted == state

    subset_values = np.asarray([[400.0, -200.0], [2.0, -4.0]])
    subset_ids = ["hundred", 10]
    generated = helper.transform_targets(subset_values, subset_ids, state)
    expected = oracle_transform(
        subset_values,
        subset_ids,
        state,
        batch_entity_id_root="batch.entity_ids",
        entity_axis=0,
    )
    np.testing.assert_array_equal(generated, expected)


@pytest.mark.parametrize(
    ("output_role", "entity_axis"),
    [
        ("location", 0),
        ("samples", 1),
        ("distribution_scale", 0),
        ("variance", 0),
        ("unitless_shape", 0),
    ],
)
def test_scaffolded_numpy_output_roles_match_pipeline_oracle(
    tmp_path: Path,
    output_role: str,
    entity_axis: int,
):
    helper, _ = _scaffolded_helper(tmp_path)
    ids = [10, "hundred", "zero", "intermittent"]
    state = helper.fit_target_scaling_state(ids, _targets(), FITTING_RANGE)
    batch_ids = ["hundred", 10]
    shape = (3, 2, 4) if entity_axis == 1 else (2, 3)
    values = np.arange(1, np.prod(shape) + 1, dtype=np.float32).reshape(shape)

    generated = helper.inverse_forecast_output(
        values,
        batch_ids,
        state,
        output_role=output_role,
        entity_axis=entity_axis,
    )
    expected = oracle_inverse(
        values,
        batch_ids,
        state,
        output_role=output_role,
        batch_entity_id_root="batch.entity_ids",
        entity_axis=entity_axis,
    )
    assert generated.dtype == values.dtype
    np.testing.assert_array_equal(generated, expected)


def test_scaffolded_torch_fit_transform_and_samples_match_oracle(tmp_path: Path):
    torch = pytest.importorskip("torch")
    helper, _ = _scaffolded_helper(tmp_path)
    ids = torch.tensor([11, 22], dtype=torch.int64)
    targets = torch.tensor(
        [[1.0, 2.0, 3.0, 1.0e8], [100.0, 200.0, 300.0, 1.0e9]],
        dtype=torch.float32,
    )
    state = helper.fit_target_scaling_state(ids, targets, FITTING_RANGE)
    oracle = oracle_fit(
        TSF_TARGET_SCALING,
        entity_ids=ids,
        targets=targets,
        fitting_range=_oracle_carrier(),
    )
    assert state == oracle

    values = torch.tensor([[200.0, 400.0], [2.0, 4.0]], dtype=torch.float32)
    batch_ids = torch.tensor([22, 11], dtype=torch.int64)
    generated_scaled = helper.transform_targets(values, batch_ids, state)
    oracle_scaled = oracle_transform(
        values,
        batch_ids,
        state,
        batch_entity_id_root="batch.entity_ids",
    )
    torch.testing.assert_close(generated_scaled, oracle_scaled)

    samples = torch.ones((5, 2, 3), dtype=torch.float32)
    generated_samples = helper.inverse_forecast_output(
        samples,
        batch_ids,
        state,
        output_role="samples",
        entity_axis=1,
    )
    oracle_samples = oracle_inverse(
        samples,
        batch_ids,
        state,
        output_role="samples",
        batch_entity_id_root="batch.entity_ids",
        entity_axis=1,
    )
    assert generated_samples.dtype == samples.dtype
    assert generated_samples.device == samples.device
    torch.testing.assert_close(generated_samples, oracle_samples)


def test_scaffolded_none_mode_is_canonical_identity(tmp_path: Path):
    helper, _ = _scaffolded_helper(tmp_path)
    ids = [1, "1"]
    targets = np.asarray([[np.nan, 2.0, 3.0], [np.inf, 5.0, 6.0]])
    state = helper.fit_target_scaling_state(
        ids,
        targets,
        FITTING_RANGE,
        mode="none",
    )
    none_contract = copy.deepcopy(TSF_TARGET_SCALING)
    none_contract["mode"] = "none"
    assert state == oracle_fit(
        none_contract,
        entity_ids=ids,
        targets=targets,
        fitting_range=_oracle_carrier(),
    )
    values = np.asarray([[7.0], [9.0]], dtype=np.float64)
    np.testing.assert_array_equal(
        helper.transform_targets(values, ids, state),
        values,
    )


def test_scaffolded_helper_rejects_duplicate_unknown_and_self_minted_shapes(tmp_path: Path):
    helper, _ = _scaffolded_helper(tmp_path)
    with pytest.raises(helper.TargetScalingError, match="duplicate typed stable ids"):
        helper.fit_target_scaling_state(
            [1, 1],
            np.ones((2, 3), dtype=np.float64),
            FITTING_RANGE,
        )

    state = helper.fit_target_scaling_state(
        [1, "1"],
        np.asarray([[1.0, 2.0, 3.0], [10.0, 20.0, 30.0]]),
        FITTING_RANGE,
    )
    with pytest.raises(helper.TargetScalingError, match="no scale"):
        helper.transform_targets(np.ones((1, 2)), [2], state)
    with pytest.raises(helper.TargetScalingError, match="duplicate typed stable ids"):
        helper.transform_targets(np.ones((2, 2)), [1, 1], state)

    malformed = dict(FITTING_RANGE)
    malformed["trusted"] = True
    with pytest.raises(helper.TargetScalingError, match="caller-supplied lineage mapping"):
        helper.fit_target_scaling_state(
            [1],
            np.ones((1, 3), dtype=np.float64),
            malformed,
        )
