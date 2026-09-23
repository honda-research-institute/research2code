"""Generated training-history helper stays portable and oracle-equivalent."""

from __future__ import annotations

import copy
import importlib.util
from pathlib import Path

import pytest

from scripts.time_series_training_history import (
    TIE_BREAK,
    not_applicable_training_history as oracle_not_applicable,
    record_training_history as oracle_record,
)
from tests.test_time_series_training_history import (
    FITTING_RANGE,
    LOSSES,
    MODEL_ID,
    PERFORMED,
    SELECTION_RANGE,
    _state,
)


ROOT = Path(__file__).resolve().parent.parent
TEMPLATE = (
    ROOT
    / "paradigms/time_series_forecasting/templates/method/"
    "training_history.py.template"
)


def _load(tmp_path: Path):
    helper_path = tmp_path / "training_history.py"
    helper_path.write_text(TEMPLATE.read_text(encoding="utf-8"), encoding="utf-8")
    spec = importlib.util.spec_from_file_location(
        f"generated_training_history_{tmp_path.name}", helper_path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _kwargs() -> dict:
    return {
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


def test_template_performed_record_matches_pipeline_oracle(tmp_path):
    helper = _load(tmp_path)

    generated = helper.record_training_history(**_kwargs())

    assert generated == oracle_record(**_kwargs())
    assert generated["selection"]["tie_break"] == TIE_BREAK
    assert not (tmp_path / ".pipeline" / "training_history.json").exists()


def test_template_no_selection_record_matches_pipeline_oracle(tmp_path):
    helper = _load(tmp_path)
    kwargs = _kwargs()
    kwargs.update({
        "checkpoint_id": "c2",
        "selection_range": None,
        "selection": {
            "status": "not_performed", "reason": "final_epoch_used",
        },
    })

    assert helper.record_training_history(**kwargs) == oracle_record(**kwargs)


def test_template_no_selection_rejects_an_earlier_checkpoint(tmp_path):
    helper = _load(tmp_path)
    kwargs = _kwargs()
    kwargs.update({
        "checkpoint_id": "c0",
        "selection_range": None,
        "selection": {
            "status": "not_performed",
            "reason": "the caller claims no checkpoint selection",
        },
    })

    with pytest.raises(helper.TrainingHistoryError, match="final ordered"):
        helper.record_training_history(**kwargs)


def test_template_not_applicable_record_matches_pipeline_oracle(tmp_path):
    helper = _load(tmp_path)

    generated = helper.not_applicable_training_history(
        reason="method_has_no_training"
    )

    assert generated == oracle_not_applicable(reason="method_has_no_training")


@pytest.mark.parametrize("seed", [-1, True])
def test_template_rejects_negative_and_boolean_seeds(tmp_path, seed):
    helper = _load(tmp_path)
    kwargs = _kwargs()
    kwargs["seed"] = seed

    with pytest.raises(helper.TrainingHistoryError, match="nonnegative integer"):
        helper.record_training_history(**kwargs)


def test_template_rejects_nonfinite_or_unordered_losses(tmp_path):
    helper = _load(tmp_path)
    kwargs = _kwargs()
    rows = copy.deepcopy(LOSSES)
    rows[1]["value"] = float("inf")
    kwargs["loss_observations"] = rows
    with pytest.raises(helper.TrainingHistoryError, match="must be finite"):
        helper.record_training_history(**kwargs)

    rows = copy.deepcopy(LOSSES)
    rows[1]["index"] = 0
    kwargs["loss_observations"] = rows
    with pytest.raises(helper.TrainingHistoryError, match="strictly increasing"):
        helper.record_training_history(**kwargs)


def test_template_rejects_selection_sample_weight_and_wrong_checkpoint(tmp_path):
    helper = _load(tmp_path)
    kwargs = _kwargs()
    selection = copy.deepcopy(PERFORMED)
    selection["observations"][0]["sample_weight"] = 1
    kwargs["selection"] = selection
    with pytest.raises(helper.TrainingHistoryError, match="unsupported shape"):
        helper.record_training_history(**kwargs)

    kwargs = _kwargs()
    kwargs["checkpoint_id"] = "c2"
    with pytest.raises(helper.TrainingHistoryError, match="must equal"):
        helper.record_training_history(**kwargs)


def test_template_record_digest_detects_mutation(tmp_path):
    helper = _load(tmp_path)
    record = helper.record_training_history(**_kwargs())
    record["config_id"] = "changed"

    with pytest.raises(helper.TrainingHistoryError, match="record_digest"):
        helper.validate_training_history(record)
