"""R2C-090 pipeline-owned training-history marker and persistence boundary."""

from __future__ import annotations

import json

import numpy as np
import pytest

import scripts.training_history_post_smoke as post_smoke
from scripts.build_plan import TSF_TARGET_SCALING
from scripts.time_series_target_scaling import (
    FittingTargetRange,
    fit_target_scaling_state,
)
from scripts.time_series_training_history import (
    TrainingHistoryCoverageError,
    TrainingHistoryUpstreamError,
    not_applicable_training_history,
    record_training_history,
)
from scripts.training_history_post_smoke import (
    TRAINING_HISTORY_ARTIFACT,
    TRAINING_HISTORY_MARKER,
    TrainingHistoryProducerError,
    extract_training_history,
    persist_post_smoke_training_history,
    validate_post_smoke_training_history,
)


def _canonical(value: object) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _range(start: int, stop: int) -> dict[str, object]:
    return {
        "validator": "eval_split_lineage",
        "target_root": "series",
        "model_id": "model",
        "start": start,
        "stop": stop,
        "certainty": "exact",
    }


def _receipt(
    *,
    fitting: tuple[int, int] = (0, 3),
    selection: tuple[int, int] | None = (3, 5),
    status: str = "valid",
) -> dict[str, object]:
    def row(span: tuple[int, int]) -> dict[str, object]:
        return {
            "root": "series",
            "model_id": "model",
            "protocol_range": {"start": span[0], "stop": span[1]},
            "subset": None,
            "certainty": "exact",
        }

    return {
        "schema_version": "1.0.0",
        "status": status,
        "validator": "eval_split_lineage",
        "reasons": [],
        "evidence": {
            "fitting_target_ranges": [row(fitting)],
            "selection_target_ranges": [] if selection is None else [row(selection)],
        },
    }


def _state(*, multiplier: float = 1.0) -> dict[str, object]:
    targets = np.asarray([
        [1.0, 2.0, 3.0, 8.0, 9.0, 10.0],
        [10.0, 20.0, 30.0, 80.0, 90.0, 100.0],
    ]) * multiplier
    return fit_target_scaling_state(
        TSF_TARGET_SCALING,
        entity_ids=["a", "b"],
        targets=targets,
        fitting_range=FittingTargetRange(
            validator="eval_split_lineage",
            target_root="series",
            model_id="model",
            start=0,
            stop=3,
        ),
    )


def _record(
    *,
    state: dict[str, object] | None = None,
    selection: bool = True,
) -> dict[str, object]:
    scaling = state or _state()
    losses = [
        {"index": 0, "value": 3.0, "sample_weight": 6,
         "checkpoint_id": "checkpoint-0"},
        {"index": 1, "value": 2.0, "sample_weight": 6,
         "checkpoint_id": "checkpoint-1"},
        {"index": 2, "value": 1.0, "sample_weight": 6,
         "checkpoint_id": "checkpoint-2"},
    ]
    if selection:
        selection_range = _range(3, 5)
        selection_record = {
            "status": "performed",
            "metric_id": "validation_loss",
            "direction": "minimize",
            "observations": [
                {"index": 0, "value": 2.5,
                 "checkpoint_id": "checkpoint-0"},
                {"index": 1, "value": 1.5,
                 "checkpoint_id": "checkpoint-1"},
                {"index": 2, "value": 0.5,
                 "checkpoint_id": "checkpoint-2"},
            ],
            "selected_checkpoint_id": "checkpoint-2",
            "tie_break": "best_then_earliest",
        }
    else:
        selection_range = None
        selection_record = {
            "status": "not_performed",
            "reason": "the training call returns its final checkpoint",
        }
    return record_training_history(
        model_id="model",
        checkpoint_id="checkpoint-2",
        seed=7,
        config_id="config-abc",
        target_scaling_state=scaling,
        fitting_range=_range(0, 3),
        loss_index_kind="epoch",
        loss_observations=losses,
        selection_range=selection_range,
        selection=selection_record,
    )


def _evaluation(**overrides: object) -> dict[str, object]:
    return {
        "model_id": "model",
        "checkpoint_id": "checkpoint-2",
        "config_id": "config-abc",
        **overrides,
    }


def _output(record: dict[str, object]) -> str:
    return f"smoke output\n{TRAINING_HISTORY_MARKER}{_canonical(record)}\ndone\n"


def test_extract_training_history_requires_one_line_anchored_canonical_marker():
    record = {"schema_version": "1.0.0", "status": "not_applicable"}

    assert extract_training_history(
        f"smoke output\n{TRAINING_HISTORY_MARKER}{_canonical(record)}\ndone\n"
    ) == record


@pytest.mark.parametrize(
    ("output", "code"),
    [
        ("no marker\n", "training_history_marker_count"),
        (
            f"{TRAINING_HISTORY_MARKER}{{}}\n"
            f"{TRAINING_HISTORY_MARKER}{{}}\n",
            "training_history_marker_count",
        ),
        (
            f"prefix {TRAINING_HISTORY_MARKER}{{}}\n",
            "training_history_marker_count",
        ),
        (
            f"{TRAINING_HISTORY_MARKER}not-json\n",
            "training_history_marker_not_json",
        ),
        (
            f"{TRAINING_HISTORY_MARKER}[]\n",
            "training_history_marker_shape",
        ),
        (
            f'{TRAINING_HISTORY_MARKER}{{"status": "not_applicable"}}\n',
            "training_history_marker_not_canonical",
        ),
    ],
)
def test_extract_training_history_refuses_ambiguous_or_malformed_output(
    output: str,
    code: str,
):
    with pytest.raises(TrainingHistoryProducerError) as failure:
        extract_training_history(output)

    assert failure.value.code == code
    assert failure.value.owner == "producer"
    assert failure.value.consume_producer_retry is True


def test_atomic_write_does_not_replace_existing_artifact_on_failure(
    tmp_path,
    monkeypatch,
):
    destination = tmp_path / ".pipeline" / "training_history.json"
    destination.parent.mkdir(parents=True)
    destination.write_text('{"old":true}\n', encoding="utf-8")

    def fail_replace(_source, _destination):
        raise OSError("replace denied")

    monkeypatch.setattr(post_smoke.os, "replace", fail_replace)
    with pytest.raises(OSError, match="replace denied"):
        post_smoke._atomic_write_json(destination, {"new": True})

    assert destination.read_text(encoding="utf-8") == '{"old":true}\n'
    assert not list(destination.parent.glob(".training_history.json.*.tmp"))


def test_recorded_history_rebinds_all_authorities_then_persists_atomically(
    tmp_path,
):
    state = _state()
    record = _record(state=state)

    normalized = persist_post_smoke_training_history(
        tmp_path,
        _output(record),
        split_receipt=_receipt(),
        target_scaling_state=state,
        executed_evaluation=_evaluation(),
    )

    assert normalized == record
    artifact = tmp_path / TRAINING_HISTORY_ARTIFACT
    assert json.loads(artifact.read_text(encoding="utf-8")) == record
    assert artifact.read_text(encoding="utf-8").endswith("\n")


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("model_id", "other-model"),
        ("checkpoint_id", "checkpoint-1"),
        ("config_id", "other-config"),
    ],
)
def test_evaluated_identity_must_equal_history_identity(
    field: str,
    value: str,
):
    state = _state()
    with pytest.raises(TrainingHistoryProducerError) as failure:
        validate_post_smoke_training_history(
            _output(_record(state=state)),
            split_receipt=_receipt(),
            target_scaling_state=state,
            executed_evaluation=_evaluation(**{field: value}),
        )

    assert failure.value.code == (
        "training_history_evaluation_identity_disagreement"
    )
    assert failure.value.consume_producer_retry is True


def test_fitting_and_selection_ranges_rebind_without_widening():
    state = _state()
    output = _output(_record(state=state))

    with pytest.raises(TrainingHistoryProducerError) as fitting_failure:
        validate_post_smoke_training_history(
            output,
            split_receipt=_receipt(fitting=(0, 2)),
            target_scaling_state=state,
            executed_evaluation=_evaluation(),
        )
    assert fitting_failure.value.code == "training_history_range_disagreement"

    with pytest.raises(TrainingHistoryProducerError) as selection_failure:
        validate_post_smoke_training_history(
            output,
            split_receipt=_receipt(selection=(4, 6)),
            target_scaling_state=state,
            executed_evaluation=_evaluation(),
        )
    assert selection_failure.value.code == "training_history_range_disagreement"


def test_scaling_state_digest_and_split_ownership_are_preserved():
    state = _state()
    output = _output(_record(state=state))

    with pytest.raises(TrainingHistoryProducerError) as state_failure:
        validate_post_smoke_training_history(
            output,
            split_receipt=_receipt(),
            target_scaling_state=_state(multiplier=2.0),
            executed_evaluation=_evaluation(),
        )
    assert state_failure.value.code == (
        "training_history_scaling_identity_disagreement"
    )
    assert state_failure.value.owner == "producer"
    assert state_failure.value.consume_producer_retry is True

    with pytest.raises(TrainingHistoryUpstreamError) as missing_state:
        validate_post_smoke_training_history(
            output,
            split_receipt=_receipt(),
            target_scaling_state=None,
            executed_evaluation=_evaluation(),
        )
    assert missing_state.value.code == "target_scaling_state_missing"
    assert missing_state.value.owner == "upstream"
    assert missing_state.value.consume_producer_retry is False

    with pytest.raises(TrainingHistoryCoverageError) as unresolved:
        validate_post_smoke_training_history(
            output,
            split_receipt=_receipt(status="unresolved"),
            target_scaling_state=state,
            executed_evaluation=_evaluation(),
        )
    assert unresolved.value.owner == "pipeline"
    assert unresolved.value.consume_producer_retry is False

    with pytest.raises(TrainingHistoryUpstreamError) as invalid:
        validate_post_smoke_training_history(
            output,
            split_receipt=_receipt(status="invalid"),
            target_scaling_state=state,
            executed_evaluation=_evaluation(),
        )
    assert invalid.value.owner == "upstream"
    assert invalid.value.consume_producer_retry is False


def test_no_selection_must_agree_with_absent_selection_lineage():
    state = _state()
    output = _output(_record(state=state, selection=False))

    assert validate_post_smoke_training_history(
        output,
        split_receipt=_receipt(selection=None),
        target_scaling_state=state,
        executed_evaluation=_evaluation(),
    )["selection"] == {
        "status": "not_performed",
        "reason": "the training call returns its final checkpoint",
    }

    with pytest.raises(TrainingHistoryProducerError) as disagreement:
        validate_post_smoke_training_history(
            output,
            split_receipt=_receipt(),
            target_scaling_state=state,
            executed_evaluation=_evaluation(),
        )
    assert disagreement.value.code == "training_history_no_selection_disagreement"


def test_not_applicable_history_is_explicit_and_needs_no_invented_authority(
    tmp_path,
):
    record = not_applicable_training_history(
        reason="the declared family has no training phase"
    )

    persisted = persist_post_smoke_training_history(
        tmp_path,
        _output(record),
        split_receipt=None,
        target_scaling_state=None,
        executed_evaluation=None,
    )

    assert persisted == record
    assert json.loads(
        (tmp_path / TRAINING_HISTORY_ARTIFACT).read_text(encoding="utf-8")
    ) == record


def test_declared_training_phase_cannot_emit_not_applicable_history(tmp_path):
    record = not_applicable_training_history(
        reason="the generated method claims it has no training phase"
    )

    with pytest.raises(TrainingHistoryProducerError) as disagreement:
        persist_post_smoke_training_history(
            tmp_path,
            _output(record),
            split_receipt=None,
            target_scaling_state=None,
            executed_evaluation=None,
            require_recorded=True,
        )

    assert disagreement.value.code == (
        "training_history_not_applicable_disagreement"
    )
    assert disagreement.value.owner == "producer"
    assert disagreement.value.consume_producer_retry is True
    assert not (tmp_path / TRAINING_HISTORY_ARTIFACT).exists()
