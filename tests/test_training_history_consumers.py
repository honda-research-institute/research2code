"""Post-smoke activation and retry routing for R2C-090 consumers."""

from __future__ import annotations

import copy
import json

import pytest

import scripts.demo_verdict as demo_verdict
from scripts.time_series_training_history import (
    not_applicable_training_history,
)
from scripts.training_history_post_smoke import TRAINING_HISTORY_ARTIFACT
from scripts.training_history_notebook_flow import TRAINING_HISTORY_MARKER
from tests.test_training_history_post_smoke import (
    _evaluation,
    _output,
    _receipt,
    _record,
    _state,
)


PLAN = {
    "training_history_execution": {
        "schema_version": "1.0.0",
        "training_call": {
            "module": "method.training",
            "callable": "train_model",
            "history_result": {
                "kind": "mapping_key",
                "key": "training_history",
            },
        },
    },
}


def _write_contract(run_dir, *, schema="2.0.0", training_loop=True):
    pipeline = run_dir / ".pipeline"
    pipeline.mkdir(parents=True, exist_ok=True)
    contract = {"schema_version": schema}
    if training_loop:
        contract["training_loop"] = {"function_name": "train_model"}
    (pipeline / "arch_contract.json").write_text(
        json.dumps(contract), encoding="utf-8"
    )


def _notebook_output(record, *, emit_output=True, source=None):
    return {
        "cells": [{
            "cell_type": "code",
            "source": source or (
                "import json\n"
                "from method.training import train_model\n"
                "training_result = train_model(model, targets)\n"
                "training_history = training_result['training_history']\n"
                "print(\n"
                f"    {TRAINING_HISTORY_MARKER!r}\n"
                "    + json.dumps(\n"
                "        training_history, sort_keys=True,\n"
                "        separators=(',', ':'), ensure_ascii=False,\n"
                "        allow_nan=False,\n"
                "    )\n"
                ")\n"
            ),
            "outputs": ([{
                "output_type": "stream",
                "name": "stdout",
                "text": _output(record),
            }] if emit_output else []),
        }],
    }


def test_schema2_training_loop_persists_valid_history_and_exact_identity(
    tmp_path,
):
    run = tmp_path / "schema2"
    _write_contract(run)
    state = _state()
    (run / ".pipeline" / "target_scaling_state.json").write_text(
        json.dumps(state), encoding="utf-8"
    )
    record = _record(state=state)

    evidence = demo_verdict._training_history_evidence(
        run,
        _notebook_output(record),
        build_plan=PLAN,
        split_receipt=_receipt(),
        executed_evaluation=_evaluation(),
    )

    assert evidence == {
        "schema_version": "1.0.0",
        "status": "valid",
        "responsibility": "pipeline_validation",
        "consume_producer_retry": False,
        "history_artifact": TRAINING_HISTORY_ARTIFACT.as_posix(),
        "record_digest": record["record_digest"],
        "model_id": "model",
        "checkpoint_id": "checkpoint-2",
        "config_id": "config-abc",
        "target_scaling_state_id": state["state_digest"],
        "notebook_value_flow": {
            "training_module": "method.training",
            "training_callable": "train_model",
            "history_key": "training_history",
            "marker": TRAINING_HISTORY_MARKER,
        },
    }
    assert json.loads(
        (run / TRAINING_HISTORY_ARTIFACT).read_text(encoding="utf-8")
    ) == record


def test_schema2_training_loop_rejects_not_applicable_as_producer_disagreement(
    tmp_path,
):
    run = tmp_path / "schema2-not-applicable"
    _write_contract(run)
    record = not_applicable_training_history(
        reason="the generated TSF method claims it did not train"
    )

    evidence = demo_verdict._training_history_evidence(
        run,
        _notebook_output(record),
        build_plan=PLAN,
        split_receipt=None,
        executed_evaluation=None,
    )

    assert evidence["status"] == "invalid"
    assert evidence["responsibility"] == "producer"
    assert evidence["consume_producer_retry"] is True
    assert evidence["reason"]["code"] == (
        "training_history_not_applicable_disagreement"
    )
    assert not (run / TRAINING_HISTORY_ARTIFACT).exists()


def test_schema2_training_loop_missing_marker_is_producer_owned(tmp_path):
    run = tmp_path / "schema2-missing-marker"
    _write_contract(run)

    evidence = demo_verdict._training_history_evidence(
        run,
        _notebook_output(_record(state=_state()), emit_output=False),
        build_plan=PLAN,
        split_receipt=_receipt(),
        executed_evaluation=_evaluation(),
    )

    assert evidence["status"] == "invalid"
    assert evidence["responsibility"] == "producer"
    assert evidence["consume_producer_retry"] is True
    assert evidence["reason"]["code"] == "training_history_marker_count"
    assert not (run / TRAINING_HISTORY_ARTIFACT).exists()


def test_history_activation_leaves_only_legacy_and_nontraining_unchanged(
    tmp_path,
):
    legacy = tmp_path / "legacy"
    _write_contract(legacy, schema="1.0.0")
    assert demo_verdict._training_history_evidence(
        legacy,
        {"cells": []},
        build_plan=PLAN,
        split_receipt=None,
        executed_evaluation=None,
    ) is None

    schema2_without_loop = tmp_path / "schema2-without-loop"
    _write_contract(schema2_without_loop, training_loop=False)
    evidence = demo_verdict._training_history_evidence(
        schema2_without_loop,
        {"cells": []},
        build_plan=PLAN,
        split_receipt=None,
        executed_evaluation=None,
    )
    assert evidence["status"] == "invalid"
    assert evidence["responsibility"] == "producer"
    assert evidence["consume_producer_retry"] is True
    assert evidence["reason"]["code"] == (
        "training_history_required_training_loop_missing"
    )

    nontraining = tmp_path / "nontraining"
    _write_contract(nontraining)
    assert demo_verdict._training_history_evidence(
        nontraining,
        {"cells": []},
        build_plan={},
        split_receipt=None,
        executed_evaluation=None,
    ) is None


@pytest.mark.parametrize(
    ("source", "code"),
    [
        (
            "import json\n"
            "from method.training import train_model\n"
            "training_result = train_model(model, targets)\n"
            "print('R2C_TRAINING_HISTORY_JSON: ' + json.dumps(\n"
            "    {'status': 'recorded'}, sort_keys=True, separators=(',', ':'),\n"
            "    ensure_ascii=False, allow_nan=False))\n",
            "training_history_notebook_marker_literal",
        ),
        (
            "import json\n"
            "from method.training import train_model\n"
            "training_result = train_model(model, targets)\n"
            "training_history = training_result['history']\n"
            "print('R2C_TRAINING_HISTORY_JSON: ' + json.dumps(\n"
            "    training_history, sort_keys=True, separators=(',', ':'),\n"
            "    ensure_ascii=False, allow_nan=False))\n",
            "training_history_notebook_result_key_disagreement",
        ),
        (
            "import json\n"
            "from method.training import train_model\n"
            "train_model(model, targets)\n"
            "print('R2C_TRAINING_HISTORY_JSON: ' + json.dumps(\n"
            "    {'status': 'recorded'}, sort_keys=True, separators=(',', ':'),\n"
            "    ensure_ascii=False, allow_nan=False))\n",
            "training_history_notebook_training_result_discarded",
        ),
    ],
)
def test_supported_source_disagreements_are_producer_owned(
    tmp_path, source, code,
):
    run = tmp_path / code
    _write_contract(run)

    evidence = demo_verdict._training_history_evidence(
        run,
        _notebook_output(_record(state=_state()), source=source),
        build_plan=PLAN,
        split_receipt=_receipt(),
        executed_evaluation=_evaluation(),
    )

    assert evidence["status"] == "invalid"
    assert evidence["responsibility"] == "producer"
    assert evidence["consume_producer_retry"] is True
    assert evidence["reason"]["code"] == code
    assert not (run / TRAINING_HISTORY_ARTIFACT).exists()


def test_interprocedural_source_grammar_is_pipeline_owned_without_retry(
    tmp_path,
):
    run = tmp_path / "interprocedural"
    _write_contract(run)
    source = (
        "import json\n"
        "from method.training import train_model\n"
        "def fit():\n"
        "    return train_model(model, targets)\n"
        "training_result = fit()\n"
        "training_history = training_result['training_history']\n"
        "print('R2C_TRAINING_HISTORY_JSON: ' + json.dumps(\n"
        "    training_history, sort_keys=True, separators=(',', ':'),\n"
        "    ensure_ascii=False, allow_nan=False))\n"
    )

    evidence = demo_verdict._training_history_evidence(
        run,
        _notebook_output(_record(state=_state()), source=source),
        build_plan=PLAN,
        split_receipt=_receipt(),
        executed_evaluation=_evaluation(),
    )

    assert evidence["status"] == "unsupported"
    assert evidence["responsibility"] == "pipeline"
    assert evidence["consume_producer_retry"] is False
    assert evidence["reason"]["code"] == (
        "training_history_notebook_training_grammar_unsupported"
    )
    assert not (run / TRAINING_HISTORY_ARTIFACT).exists()


def test_missing_scaling_authority_stays_upstream_owned_at_consumer_boundary(
    tmp_path,
):
    run = tmp_path / "missing-state"
    _write_contract(run)
    state = _state()

    evidence = demo_verdict._training_history_evidence(
        run,
        _notebook_output(_record(state=state)),
        build_plan=PLAN,
        split_receipt=_receipt(),
        executed_evaluation=_evaluation(),
    )

    assert evidence["status"] == "upstream_invalid"
    assert evidence["responsibility"] == "upstream"
    assert evidence["consume_producer_retry"] is False
    assert evidence["reason"]["code"] == "target_scaling_state_missing"
    assert not (run / TRAINING_HISTORY_ARTIFACT).exists()


def test_stale_scaling_identity_is_producer_owned_at_consumer_boundary(
    tmp_path,
):
    run = tmp_path / "stale-state"
    _write_contract(run)
    recorded_state = _state()
    current_state = _state(multiplier=2.0)
    (run / ".pipeline" / "target_scaling_state.json").write_text(
        json.dumps(current_state), encoding="utf-8"
    )

    evidence = demo_verdict._training_history_evidence(
        run,
        _notebook_output(_record(state=recorded_state)),
        build_plan=PLAN,
        split_receipt=_receipt(),
        executed_evaluation=_evaluation(),
    )

    assert evidence["status"] == "invalid"
    assert evidence["responsibility"] == "producer"
    assert evidence["consume_producer_retry"] is True
    assert evidence["reason"]["code"] == (
        "training_history_scaling_identity_disagreement"
    )
    assert not (run / TRAINING_HISTORY_ARTIFACT).exists()


@pytest.mark.parametrize(
    ("identity", "value"),
    [
        ("checkpoint_id", "different-checkpoint"),
        ("config_id", "different-config"),
    ],
)
def test_evaluated_checkpoint_and_config_mismatch_are_producer_owned(
    tmp_path,
    identity,
    value,
):
    run = tmp_path / identity
    _write_contract(run)
    state = _state()
    (run / ".pipeline" / "target_scaling_state.json").write_text(
        json.dumps(state), encoding="utf-8"
    )

    evidence = demo_verdict._training_history_evidence(
        run,
        _notebook_output(_record(state=state)),
        build_plan=PLAN,
        split_receipt=_receipt(),
        executed_evaluation=_evaluation(**{identity: value}),
    )

    assert evidence["status"] == "invalid"
    assert evidence["responsibility"] == "producer"
    assert evidence["consume_producer_retry"] is True
    assert evidence["reason"]["code"] == (
        "training_history_evaluation_identity_disagreement"
    )
    assert not (run / TRAINING_HISTORY_ARTIFACT).exists()


def test_unsupported_split_grammar_is_pipeline_owned_without_retry(tmp_path):
    run = tmp_path / "unsupported-split"
    _write_contract(run)
    state = _state()
    (run / ".pipeline" / "target_scaling_state.json").write_text(
        json.dumps(state), encoding="utf-8"
    )
    receipt = copy.deepcopy(_receipt())
    receipt["evidence"]["fitting_target_ranges"].append({
        **receipt["evidence"]["fitting_target_ranges"][0],
        "protocol_range": {"start": 4, "stop": 6},
    })

    evidence = demo_verdict._training_history_evidence(
        run,
        _notebook_output(_record(state=state)),
        build_plan=PLAN,
        split_receipt=receipt,
        executed_evaluation=_evaluation(),
    )

    assert evidence["status"] == "unsupported"
    assert evidence["responsibility"] == "pipeline"
    assert evidence["consume_producer_retry"] is False
    assert evidence["reason"]["code"] == (
        "training_history_range_grammar_unsupported"
    )
    assert not (run / TRAINING_HISTORY_ARTIFACT).exists()


def test_unsupported_selection_grammar_is_pipeline_owned_without_retry(
    tmp_path,
):
    run = tmp_path / "unsupported-selection"
    _write_contract(run)
    state = _state()
    (run / ".pipeline" / "target_scaling_state.json").write_text(
        json.dumps(state), encoding="utf-8"
    )
    receipt = copy.deepcopy(_receipt())
    receipt["evidence"]["selection_target_ranges"][0]["subset"] = {
        "kind": "boolean_mask"
    }

    evidence = demo_verdict._training_history_evidence(
        run,
        _notebook_output(_record(state=state)),
        build_plan=PLAN,
        split_receipt=receipt,
        executed_evaluation=_evaluation(),
    )

    assert evidence["status"] == "unsupported"
    assert evidence["responsibility"] == "pipeline"
    assert evidence["consume_producer_retry"] is False
    assert evidence["reason"]["code"] == (
        "training_history_range_grammar_unsupported"
    )
    assert not (run / TRAINING_HISTORY_ARTIFACT).exists()
