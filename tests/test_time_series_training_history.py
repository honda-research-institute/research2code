"""Closed v1 training-history mechanics and trusted lineage binding."""

from __future__ import annotations

import copy

import numpy as np
import pytest

from scripts.build_plan import TSF_TARGET_SCALING
from scripts.time_series_target_scaling import (
    FittingTargetRange,
    fit_target_scaling_state,
)
from scripts.time_series_training_history import (
    TIE_BREAK,
    TrainingHistoryCoverageError,
    TrainingHistoryProducerError,
    TrainingHistoryUpstreamError,
    not_applicable_training_history,
    record_training_history,
    resolve_training_target_range,
    validate_training_history,
    validate_training_history_lineage,
)


MODEL_ID = "forecast-model"
FITTING_RANGE = {
    "validator": "eval_split_lineage",
    "target_root": "batch.targets",
    "model_id": MODEL_ID,
    "start": 0,
    "stop": 3,
    "certainty": "exact",
}
SELECTION_RANGE = {
    "validator": "eval_split_lineage",
    "target_root": "batch.targets",
    "model_id": MODEL_ID,
    "start": 3,
    "stop": 4,
    "certainty": "exact",
}
LOSSES = [
    {"index": 0, "value": 3.0, "sample_weight": 8, "checkpoint_id": "c0"},
    {"index": 1, "value": 2.0, "sample_weight": 8, "checkpoint_id": "c1"},
    {"index": 2, "value": 1.0, "sample_weight": 8, "checkpoint_id": "c2"},
]
PERFORMED = {
    "status": "performed",
    "metric_id": "validation_nll",
    "direction": "minimize",
    "observations": [
        {"index": 0, "value": 2.0, "checkpoint_id": "c0"},
        {"index": 1, "value": 1.0, "checkpoint_id": "c1"},
        {"index": 2, "value": 1.0, "checkpoint_id": "c2"},
    ],
    "selected_checkpoint_id": "c1",
    "tie_break": TIE_BREAK,
}


def _state() -> dict:
    return fit_target_scaling_state(
        TSF_TARGET_SCALING,
        entity_ids=np.asarray([10, 20], dtype=np.int64),
        targets=np.asarray(
            [[1.0, 2.0, 3.0, 4.0], [10.0, 20.0, 30.0, 40.0]],
            dtype=np.float64,
        ),
        fitting_range=FittingTargetRange(
            validator="eval_split_lineage",
            target_root="batch.targets",
            model_id=MODEL_ID,
            start=0,
            stop=3,
        ),
    )


def _receipt(*, selection_rows: list[dict] | None = None) -> dict:
    return {
        "schema_version": "1.0.0",
        "status": "valid",
        "validator": "eval_split_lineage",
        "reasons": ["fitting_to_reported_evaluation_disjointness_proved"],
        "evidence": {
            "fitting_target_ranges": [{
                "root": "batch.targets",
                "model_id": MODEL_ID,
                "protocol_range": {"start": 0, "stop": 3},
                "subset": None,
                "certainty": "exact",
            }],
            "selection_target_ranges": (
                selection_rows if selection_rows is not None else [{
                    "root": "batch.targets",
                    "model_id": MODEL_ID,
                    "protocol_range": {"start": 3, "stop": 4},
                    "subset": None,
                    "certainty": "exact",
                }]
            ),
        },
    }


def _record(**changes) -> dict:
    kwargs = {
        "model_id": MODEL_ID,
        "checkpoint_id": "c1",
        "seed": 7,
        "config_id": "sha256:training-config",
        "target_scaling_state": _state(),
        "fitting_range": FITTING_RANGE,
        "loss_index_kind": "epoch",
        "loss_observations": LOSSES,
        "selection_range": SELECTION_RANGE,
        "selection": PERFORMED,
    }
    kwargs.update(changes)
    return record_training_history(**kwargs)


def test_performed_history_is_canonical_and_selects_best_then_earliest():
    record = _record()

    assert set(record) == {
        "schema_version", "status", "model_id", "checkpoint_id", "seed",
        "config_id", "target_scaling_state_id", "fitting_range",
        "loss_index_kind", "loss_observations", "selection_range",
        "selection", "record_digest",
    }
    assert record["status"] == "recorded"
    assert record["selection"]["selected_checkpoint_id"] == "c1"
    assert record["selection"]["tie_break"] == "best_then_earliest"
    assert record["target_scaling_state_id"] == _state()["state_digest"]
    assert validate_training_history(record) == record


def test_maximize_direction_uses_same_earliest_tie_rule():
    selection = copy.deepcopy(PERFORMED)
    selection.update(direction="maximize", selected_checkpoint_id="c1")
    selection["observations"] = [
        {"index": 0, "value": 0.1, "checkpoint_id": "c0"},
        {"index": 1, "value": 0.9, "checkpoint_id": "c1"},
        {"index": 2, "value": 0.9, "checkpoint_id": "c2"},
    ]

    assert _record(selection=selection)["checkpoint_id"] == "c1"


@pytest.mark.parametrize("seed", [-1, True])
def test_seed_must_be_a_nonnegative_integer(seed):
    with pytest.raises(
        TrainingHistoryProducerError,
        match="seed must be a nonnegative integer",
    ):
        _record(seed=seed)


@pytest.mark.parametrize("field", ["model_id", "checkpoint_id", "config_id"])
def test_run_identities_must_be_exact_nonblank_text(field):
    with pytest.raises(TrainingHistoryProducerError):
        _record(**{field: " "})


def test_target_scaling_state_is_validated_and_only_its_identity_is_stored():
    state = _state()
    state["entries"][0]["scale"] = 999.0

    with pytest.raises(
        TrainingHistoryProducerError,
        match="target_scaling_state is invalid",
    ):
        _record(target_scaling_state=state)


@pytest.mark.parametrize(
    ("mutator", "code"),
    [
        (lambda rows: rows.__setitem__(1, {**rows[1], "index": 0}),
         "training_history_event_order"),
        (lambda rows: rows.__setitem__(1, {**rows[1], "value": float("nan")}),
         "training_history_nonfinite"),
        (lambda rows: rows.__setitem__(1, {**rows[1], "sample_weight": 0}),
         "training_history_sample_weight"),
    ],
)
def test_loss_series_requires_finite_strict_order_and_positive_weight(
    mutator, code,
):
    rows = copy.deepcopy(LOSSES)
    mutator(rows)

    with pytest.raises(TrainingHistoryProducerError) as failure:
        _record(loss_observations=rows)

    assert failure.value.code == code


def test_selection_observations_have_the_closed_smaller_shape():
    selection = copy.deepcopy(PERFORMED)
    selection["observations"][0]["sample_weight"] = 2

    with pytest.raises(TrainingHistoryProducerError) as failure:
        _record(selection=selection)

    assert failure.value.code == "training_history_observation_shape"


def test_selection_observation_must_bind_same_loss_index_and_checkpoint():
    selection = copy.deepcopy(PERFORMED)
    selection["observations"][0]["checkpoint_id"] = "different"

    with pytest.raises(TrainingHistoryProducerError) as failure:
        _record(selection=selection)

    assert failure.value.code == "training_history_checkpoint_unbound"


def test_declared_checkpoint_must_equal_computed_best_checkpoint():
    with pytest.raises(TrainingHistoryProducerError) as failure:
        _record(checkpoint_id="c2")

    assert failure.value.code == "training_history_checkpoint_disagreement"


def test_fitting_and_selection_ranges_must_be_exact_disjoint_same_domain():
    overlap = {**SELECTION_RANGE, "start": 2}
    with pytest.raises(TrainingHistoryProducerError) as failure:
        _record(selection_range=overlap)
    assert failure.value.code == "training_history_range_overlap"

    other_root = {**SELECTION_RANGE, "target_root": "other.targets"}
    with pytest.raises(TrainingHistoryProducerError) as failure:
        _record(selection_range=other_root)
    assert failure.value.code == "training_history_root_disagreement"


def test_no_selection_arm_is_explicit_and_has_no_selection_range():
    record = _record(
        checkpoint_id="c2",
        selection_range=None,
        selection={"status": "not_performed", "reason": "final_epoch_used"},
    )

    assert record["selection"] == {
        "status": "not_performed", "reason": "final_epoch_used",
    }
    assert validate_training_history_lineage(
        record, _receipt(selection_rows=[]), target_scaling_state=_state()
    ) == record


def test_no_selection_requires_the_final_ordered_checkpoint():
    with pytest.raises(TrainingHistoryProducerError) as failure:
        _record(
            checkpoint_id="c0",
            selection_range=None,
            selection={
                "status": "not_performed",
                "reason": "the caller claims no checkpoint selection",
            },
        )

    assert failure.value.code == (
        "training_history_no_selection_checkpoint_disagreement"
    )


def test_no_selection_arm_rejects_a_range_or_trusted_selection_evidence():
    selection = {"status": "not_performed", "reason": "final_epoch_used"}
    with pytest.raises(TrainingHistoryProducerError):
        _record(selection=selection)

    record = _record(
        checkpoint_id="c2", selection_range=None, selection=selection,
    )
    with pytest.raises(TrainingHistoryProducerError) as failure:
        validate_training_history_lineage(record, _receipt())
    assert failure.value.code == "training_history_no_selection_disagreement"


def test_not_applicable_arm_is_closed_and_does_not_claim_ranges():
    record = not_applicable_training_history(reason="method_has_no_training")

    assert set(record) == {
        "schema_version", "status", "reason", "record_digest",
    }
    assert record["status"] == "not_applicable"
    assert validate_training_history_lineage(record, None) == record


def test_record_digest_detects_mutation():
    record = _record()
    record["seed"] = 8

    with pytest.raises(TrainingHistoryProducerError) as failure:
        validate_training_history(record)

    assert failure.value.code == "training_history_digest"


def test_lineage_binding_accepts_exact_ranges_and_scaling_identity():
    record = _record()

    assert validate_training_history_lineage(
        record, _receipt(), target_scaling_state=_state()
    ) == record
    assert resolve_training_target_range(
        _receipt(), role="model_selection", target_root="batch.targets",
        model_id=MODEL_ID,
    ).as_dict() == SELECTION_RANGE


def test_supported_range_disagreement_is_producer_owned():
    record = _record()
    receipt = _receipt()
    receipt["evidence"]["selection_target_ranges"][0]["protocol_range"] = {
        "start": 4, "stop": 5,
    }

    with pytest.raises(TrainingHistoryProducerError) as failure:
        validate_training_history_lineage(record, receipt)

    assert failure.value.owner == "producer"
    assert failure.value.consume_producer_retry is True


def test_support_or_multiple_ranges_are_pipeline_coverage_without_retry():
    support = _receipt()
    support["evidence"]["selection_target_ranges"][0].update(
        subset="boolean_mask", certainty="support",
    )
    with pytest.raises(TrainingHistoryCoverageError) as failure:
        validate_training_history_lineage(_record(), support)
    assert failure.value.owner == "pipeline"
    assert failure.value.consume_producer_retry is False

    multiple = _receipt()
    multiple["evidence"]["selection_target_ranges"].append({
        "root": "batch.targets",
        "model_id": MODEL_ID,
        "protocol_range": {"start": 4, "stop": 5},
        "subset": None,
        "certainty": "exact",
    })
    with pytest.raises(TrainingHistoryCoverageError):
        validate_training_history_lineage(_record(), multiple)


def test_invalid_or_unresolved_split_preserves_ownership():
    invalid = _receipt()
    invalid["status"] = "invalid"
    with pytest.raises(TrainingHistoryUpstreamError) as upstream:
        validate_training_history_lineage(_record(), invalid)
    assert upstream.value.owner == "upstream"
    assert upstream.value.consume_producer_retry is False

    unresolved = _receipt()
    unresolved["status"] = "unresolved"
    with pytest.raises(TrainingHistoryCoverageError) as pipeline:
        validate_training_history_lineage(_record(), unresolved)
    assert pipeline.value.owner == "pipeline"
    assert pipeline.value.consume_producer_retry is False
