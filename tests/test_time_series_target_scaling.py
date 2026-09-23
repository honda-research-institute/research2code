"""R2C-089 — training-only per-series scaling by stable identity."""

from __future__ import annotations

import copy
import hashlib
import json

import numpy as np
import pytest

from scripts.build_plan import TSF_TARGET_SCALING
from scripts.time_series_target_scaling import (
    FittingTargetRange,
    TargetScalingCoverageError,
    TargetScalingProducerError,
    TargetScalingUpstreamError,
    fit_target_scaling_state,
    inverse_forecast_output,
    resolve_fitting_target_range,
    transform_targets,
    validate_target_scaling_state,
    validate_training_only_state,
)


def _receipt(
    *rows: dict[str, object],
    status: str = "valid",
) -> dict[str, object]:
    return {
        "schema_version": "1.0.0",
        "status": status,
        "validator": "eval_split_lineage",
        "reasons": [],
        "evidence": {"fitting_target_ranges": list(rows)},
    }


def _range_row(
    start: int = 0,
    stop: int = 3,
    *,
    root: str = "series",
    model_id: str = "model",
    subset: str | None = None,
) -> dict[str, object]:
    return {
        "root": root,
        "model_id": model_id,
        "protocol_range": {"start": start, "stop": stop},
        "subset": subset,
        "certainty": "exact" if subset is None else "support",
    }


def _carrier(start: int = 0, stop: int = 3) -> FittingTargetRange:
    return resolve_fitting_target_range(
        _receipt(_range_row(start, stop)),
        target_root="series",
        model_id="model",
    )


def _targets(dtype: np.dtype = np.dtype("float64")) -> np.ndarray:
    # Fitting values are the first three columns. Held-out columns contain the
    # leakage trap and must never affect state.
    return np.asarray([
        [-1.0, 2.0, -3.0, 1.0e8, -1.0e8],
        [-100.0, 200.0, -300.0, -1.0e9, 1.0e9],
        [0.0, 0.0, 0.0, 7.0e8, 7.0e8],
        [0.0, 0.0, 6.0, 9.0e8, 9.0e8],
    ], dtype=dtype)


def _entry_by_id(state: dict[str, object]) -> dict[tuple[str, object], dict]:
    return {
        (entry["entity_id"]["type"], entry["entity_id"]["value"]): entry
        for entry in state["entries"]
    }


def _resign(state: dict[str, object]) -> None:
    payload = copy.deepcopy(state)
    payload.pop("state_digest", None)
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    state["state_digest"] = hashlib.sha256(encoded).hexdigest()


def test_exact_fitting_carrier_deduplicates_only_identical_rows():
    row = _range_row()
    carrier = resolve_fitting_target_range(
        _receipt(row, copy.deepcopy(row)),
        target_root="series",
        model_id="model",
    )
    assert carrier.as_dict() == {
        "validator": "eval_split_lineage",
        "target_root": "series",
        "model_id": "model",
        "start": 0,
        "stop": 3,
        "certainty": "exact",
    }

    with pytest.raises(
        TargetScalingCoverageError,
        match="does not union or choose",
    ) as ambiguous:
        resolve_fitting_target_range(
            _receipt(row, _range_row(0, 4)),
            target_root="series",
            model_id="model",
        )
    assert ambiguous.value.owner == "pipeline"
    assert ambiguous.value.consume_producer_retry is False


def test_fitting_carrier_routing_preserves_coverage_upstream_and_producer_ownership():
    with pytest.raises(TargetScalingCoverageError, match="subset/support") as subset:
        resolve_fitting_target_range(
            _receipt(_range_row(subset="boolean_mask")),
            target_root="series",
            model_id="model",
        )
    assert subset.value.owner == "pipeline"
    assert subset.value.consume_producer_retry is False

    with pytest.raises(TargetScalingCoverageError, match="status 'unresolved'"):
        resolve_fitting_target_range(
            _receipt(status="unresolved"),
            target_root="series",
            model_id="model",
        )

    with pytest.raises(TargetScalingUpstreamError, match="upstream finding") as invalid:
        resolve_fitting_target_range(
            _receipt(status="invalid"),
            target_root="series",
            model_id="model",
        )
    assert invalid.value.consume_producer_retry is False

    with pytest.raises(
        TargetScalingProducerError,
        match="absent from the valid pipeline receipt",
    ) as mismatch:
        resolve_fitting_target_range(
            _receipt(_range_row(root="other")),
            target_root="series",
            model_id="model",
        )
    assert mismatch.value.owner == "producer"
    assert mismatch.value.consume_producer_retry is True


def test_fitting_uses_only_half_open_range_and_records_1x_100x_zero_and_intermittent():
    ids = ["one", "hundred", "zero", "intermittent"]
    state = fit_target_scaling_state(
        TSF_TARGET_SCALING,
        entity_ids=ids,
        targets=_targets(),
        fitting_range=_carrier(),
    )
    changed_heldout = _targets()
    changed_heldout[:, 3:] *= -12345.0
    control = fit_target_scaling_state(
        TSF_TARGET_SCALING,
        entity_ids=ids,
        targets=changed_heldout,
        fitting_range=_carrier(),
    )

    assert state == control
    entries = _entry_by_id(state)
    assert entries[("string", "one")]["scale"] == pytest.approx(2.0)
    assert entries[("string", "hundred")]["scale"] == pytest.approx(200.0)
    assert entries[("string", "intermittent")]["scale"] == pytest.approx(2.0)
    assert entries[("string", "zero")]["scale"] == 1.0
    assert entries[("string", "zero")]["zero_series_resolution"] == "unit_scale"
    assert all(entry["fitting_observation_count"] == 3 for entry in entries.values())
    assert state["fitting_lineage"] == _carrier().as_dict()


def test_epsilon_is_a_zero_detection_rule_not_a_floor_for_nonzero_mean():
    values = np.asarray([[0.0, 0.0, 2.0e-8]], dtype=np.float64)
    state = fit_target_scaling_state(
        TSF_TARGET_SCALING,
        entity_ids=["small"],
        targets=values,
        fitting_range=_carrier(),
    )
    entry = state["entries"][0]
    assert entry["statistic_value"] == pytest.approx(2.0e-8 / 3.0)
    assert entry["scale"] == pytest.approx(2.0e-8 / 3.0)
    assert entry["zero_series_resolution"] == "not_applied"


def test_coherent_permutation_has_canonical_state_and_subset_lookup_is_by_typed_id():
    ids = [10, 20, "zero", "intermittent"]
    targets = _targets()
    state = fit_target_scaling_state(
        TSF_TARGET_SCALING,
        entity_ids=ids,
        targets=targets,
        fitting_range=_carrier(),
    )
    permutation = [2, 0, 3, 1]
    permuted = fit_target_scaling_state(
        TSF_TARGET_SCALING,
        entity_ids=[ids[index] for index in permutation],
        targets=targets[permutation],
        fitting_range=_carrier(),
    )
    assert permuted == state

    # An induced/subset batch can reorder entities without receiving a
    # transient-position scale.
    normalized = transform_targets(
        np.asarray([[200.0, -400.0], [2.0, -4.0]], dtype=np.float64),
        [20, 10],
        state,
        batch_entity_id_root="prepared.entity_ids",
        entity_axis=0,
    )
    np.testing.assert_allclose(normalized, [[1.0, -2.0], [1.0, -2.0]])


def test_graph_free_control_uses_explicit_ids_without_graph_or_relational_inference():
    state = fit_target_scaling_state(
        TSF_TARGET_SCALING,
        entity_ids=["north", "south"],
        targets=np.asarray([[1.0, -3.0, 2.0], [10.0, -30.0, 20.0]]),
        fitting_range=_carrier(),
    )
    scaled = transform_targets(
        np.asarray([[20.0], [2.0]], dtype=np.float64),
        ["south", "north"],
        state,
        batch_entity_id_root="graph_free.series_ids",
    )
    np.testing.assert_allclose(scaled, [[1.0], [1.0]])


def test_duplicate_unknown_and_typed_identity_fail_with_state_and_batch_roots():
    with pytest.raises(TargetScalingProducerError, match="batch.entity_ids.*duplicate"):
        fit_target_scaling_state(
            TSF_TARGET_SCALING,
            entity_ids=[1, 1],
            targets=np.ones((2, 3), dtype=np.float64),
            fitting_range=_carrier(),
        )
    with pytest.raises(TargetScalingProducerError, match="boolean identity"):
        fit_target_scaling_state(
            TSF_TARGET_SCALING,
            entity_ids=[True],
            targets=np.ones((1, 3), dtype=np.float64),
            fitting_range=_carrier(),
        )
    with pytest.raises(TargetScalingProducerError, match="unsupported identity 1.0"):
        fit_target_scaling_state(
            TSF_TARGET_SCALING,
            entity_ids=[1.0],
            targets=np.ones((1, 3), dtype=np.float64),
            fitting_range=_carrier(),
        )

    state = fit_target_scaling_state(
        TSF_TARGET_SCALING,
        entity_ids=[1, "1"],
        targets=np.asarray([[1.0, 1.0, 1.0], [10.0, 10.0, 10.0]]),
        fitting_range=_carrier(),
    )
    with pytest.raises(TargetScalingProducerError) as unknown:
        transform_targets(
            np.ones((1, 2), dtype=np.float64),
            [2],
            state,
            batch_entity_id_root="current.entity_ids",
        )
    message = str(unknown.value)
    assert ".pipeline/target_scaling_state.json" in message
    assert "batch.entity_ids" in message
    assert "current.entity_ids" in message


def test_corrupt_missing_and_duplicate_state_entries_fail_closed():
    state = fit_target_scaling_state(
        TSF_TARGET_SCALING,
        entity_ids=["a", "b"],
        targets=np.asarray([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]]),
        fitting_range=_carrier(),
    )
    missing = copy.deepcopy(state)
    missing["entries"].pop()
    _resign(missing)
    with pytest.raises(
        TargetScalingProducerError,
        match="target_scaling_state.json.*entity_count",
    ):
        validate_target_scaling_state(missing)

    duplicate = copy.deepcopy(state)
    duplicate["entries"].append(copy.deepcopy(duplicate["entries"][0]))
    duplicate["entity_count"] = 3
    _resign(duplicate)
    with pytest.raises(
        TargetScalingProducerError,
        match="target_scaling_state.json.*repeats typed identity",
    ):
        validate_target_scaling_state(duplicate)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_numpy_round_trip_and_output_unit_rules(dtype):
    state = fit_target_scaling_state(
        TSF_TARGET_SCALING,
        entity_ids=["a", "b"],
        targets=np.asarray([[1.0, 2.0, 3.0], [100.0, 200.0, 300.0]], dtype=dtype),
        fitting_range=_carrier(),
    )
    original = np.asarray([[4.0, 6.0], [400.0, 600.0]], dtype=dtype)
    normalized = transform_targets(
        original,
        ["a", "b"],
        state,
        batch_entity_id_root="batch.entity_ids",
        entity_axis=0,
    )
    restored = inverse_forecast_output(
        normalized,
        ["a", "b"],
        state,
        output_role="location",
        batch_entity_id_root="batch.entity_ids",
        entity_axis=0,
    )
    assert restored.dtype == dtype
    np.testing.assert_allclose(restored, original, rtol=1.0e-6)

    samples = np.ones((3, 2, 2), dtype=dtype)
    inverted_samples = inverse_forecast_output(
        samples,
        ["a", "b"],
        state,
        output_role="samples",
        batch_entity_id_root="outputs.entity_ids",
        entity_axis=1,
    )
    np.testing.assert_allclose(inverted_samples[:, 0], 2.0)
    np.testing.assert_allclose(inverted_samples[:, 1], 200.0)

    scale = inverse_forecast_output(
        np.ones((2, 2), dtype=dtype),
        ["a", "b"],
        state,
        output_role="distribution_scale",
        batch_entity_id_root="outputs.entity_ids",
        entity_axis=0,
    )
    variance = inverse_forecast_output(
        np.ones((2, 2), dtype=dtype),
        ["a", "b"],
        state,
        output_role="variance",
        batch_entity_id_root="outputs.entity_ids",
        entity_axis=0,
    )
    unitless = np.asarray([[3.0], [5.0]], dtype=dtype)
    unchanged = inverse_forecast_output(
        unitless,
        ["a", "b"],
        state,
        output_role="unitless_shape",
        batch_entity_id_root="outputs.entity_ids",
        entity_axis=0,
    )
    np.testing.assert_allclose(scale[:, 0], [2.0, 200.0])
    np.testing.assert_allclose(variance[:, 0], [4.0, 40000.0])
    np.testing.assert_array_equal(unchanged, unitless)


def test_torch_transform_and_inversion_preserve_dtype_device_and_sample_order():
    torch = pytest.importorskip("torch")
    targets = torch.tensor(
        [[1.0, 2.0, 3.0], [100.0, 200.0, 300.0]],
        dtype=torch.float32,
    )
    state = fit_target_scaling_state(
        TSF_TARGET_SCALING,
        entity_ids=torch.tensor([11, 22], dtype=torch.int64),
        targets=targets,
        fitting_range=_carrier(),
    )
    samples = torch.ones((4, 2, 3), dtype=torch.float32)
    inverted = inverse_forecast_output(
        samples,
        torch.tensor([22, 11], dtype=torch.int64),
        state,
        output_role="samples",
        batch_entity_id_root="prepared.entity_ids",
        entity_axis=1,
    )
    assert inverted.dtype == torch.float32
    assert inverted.device == samples.device
    torch.testing.assert_close(inverted[:, 0], torch.full((4, 3), 200.0))
    torch.testing.assert_close(inverted[:, 1], torch.full((4, 3), 2.0))


def test_explicit_none_arm_is_identity_and_does_not_fit_nonfinite_values():
    contract = copy.deepcopy(TSF_TARGET_SCALING)
    contract["mode"] = "none"
    targets = np.asarray([[np.nan, 2.0, 3.0], [np.inf, 5.0, 6.0]])
    state = fit_target_scaling_state(
        contract,
        entity_ids=["a", "b"],
        targets=targets,
        fitting_range=_carrier(),
    )
    assert [entry["scale"] for entry in state["entries"]] == [1.0, 1.0]
    assert all(
        entry["zero_series_resolution"] == "mode_none"
        and entry["statistic_value"] is None
        and entry["fitting_observation_count"] == 0
        for entry in state["entries"]
    )
    values = np.asarray([[7.0], [9.0]])
    np.testing.assert_array_equal(
        transform_targets(
            values,
            ["a", "b"],
            state,
            batch_entity_id_root="batch.entity_ids",
        ),
        values,
    )


def test_unsupported_mode_and_output_role_are_pipeline_coverage_not_retry():
    contract = copy.deepcopy(TSF_TARGET_SCALING)
    contract["mode"] = "global"
    with pytest.raises(TargetScalingCoverageError, match="outside the v1 grammar") as mode:
        fit_target_scaling_state(
            contract,
            entity_ids=["a"],
            targets=np.ones((1, 3), dtype=np.float64),
            fitting_range=_carrier(),
        )
    assert mode.value.consume_producer_retry is False

    state = fit_target_scaling_state(
        TSF_TARGET_SCALING,
        entity_ids=["a"],
        targets=np.ones((1, 3), dtype=np.float64),
        fitting_range=_carrier(),
    )
    with pytest.raises(TargetScalingCoverageError, match="output role") as role:
        inverse_forecast_output(
            np.ones((1, 1), dtype=np.float64),
            ["a"],
            state,
            output_role="quantile",
            batch_entity_id_root="outputs.entity_ids",
            entity_axis=0,
        )
    assert role.value.consume_producer_retry is False


def test_nonfinite_fitting_values_refuse_but_nonfinite_heldout_values_do_not_enter_state():
    fitting_bad = np.asarray([[1.0, np.nan, 3.0, 4.0]])
    with pytest.raises(TargetScalingProducerError, match="non-finite fitting values"):
        fit_target_scaling_state(
            TSF_TARGET_SCALING,
            entity_ids=["a"],
            targets=fitting_bad,
            fitting_range=_carrier(),
        )

    heldout_nonfinite = np.asarray([[1.0, 2.0, 3.0, np.inf]])
    state = fit_target_scaling_state(
        TSF_TARGET_SCALING,
        entity_ids=["a"],
        targets=heldout_nonfinite,
        fitting_range=_carrier(),
    )
    assert state["entries"][0]["scale"] == pytest.approx(2.0)


def test_state_must_rebind_to_pipeline_minted_range_before_training_only_certification():
    receipt = _receipt(_range_row())
    state = fit_target_scaling_state(
        TSF_TARGET_SCALING,
        entity_ids=["a"],
        targets=np.ones((1, 4), dtype=np.float64),
        fitting_range=resolve_fitting_target_range(
            receipt, target_root="series", model_id="model"
        ),
    )
    assert validate_training_only_state(
        state, receipt, contract=TSF_TARGET_SCALING
    )["state_digest"] == state["state_digest"]

    self_minted = copy.deepcopy(state)
    self_minted["fitting_lineage"]["stop"] = 4
    _resign(self_minted)
    with pytest.raises(
        TargetScalingProducerError,
        match="disagrees with the exact pipeline carrier",
    ) as disagreement:
        validate_training_only_state(
            self_minted, receipt, contract=TSF_TARGET_SCALING
        )
    assert disagreement.value.consume_producer_retry is True
