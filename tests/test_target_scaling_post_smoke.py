"""R2C-089 post-smoke state ownership and original-unit evidence."""

from __future__ import annotations

import copy
import hashlib
import json

import numpy as np
import pytest

import scripts.target_scaling_post_smoke as post_smoke
from scripts.build_plan import TSF_TARGET_SCALING
from scripts.target_scaling_post_smoke import (
    STATE_ARTIFACT,
    STATE_MARKER,
    extract_target_scaling_state,
    persist_post_smoke_target_scaling_state,
    validate_original_unit_comparison,
    validate_post_smoke_target_scaling_state,
)
from scripts.time_series_target_scaling import (
    TargetScalingCoverageError,
    TargetScalingProducerError,
    TargetScalingUpstreamError,
    fit_target_scaling_state,
    resolve_fitting_target_range,
)


def _build_plan() -> dict[str, object]:
    return {"target_scaling": copy.deepcopy(TSF_TARGET_SCALING)}


def _range_row(
    *,
    start: int = 0,
    stop: int = 3,
    root: str = "batch.targets",
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


def _state() -> dict[str, object]:
    receipt = _receipt(_range_row())
    fitting_range = resolve_fitting_target_range(
        receipt,
        target_root="batch.targets",
        model_id="model",
    )
    return fit_target_scaling_state(
        TSF_TARGET_SCALING,
        entity_ids=["low", "high"],
        targets=np.asarray([
            [1.0, 2.0, 3.0, 1.0e9],
            [100.0, 200.0, 300.0, -1.0e9],
        ]),
        fitting_range=fitting_range,
    )


def _canonical(value: object) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _output(state: dict[str, object] | None = None) -> str:
    candidate = _state() if state is None else state
    return f"smoke log\n{STATE_MARKER}{_canonical(candidate)}\ndone\n"


def _resign(state: dict[str, object]) -> None:
    payload = copy.deepcopy(state)
    payload.pop("state_digest", None)
    state["state_digest"] = hashlib.sha256(
        _canonical(payload).encode("utf-8")
    ).hexdigest()


def test_exact_marker_is_validated_then_persisted_only_in_pipeline_dir(tmp_path):
    run_dir = tmp_path / "graph-free"
    generated = run_dir / "method" / "training.py"
    generated.parent.mkdir(parents=True)
    generated.write_text("producer-owned\n", encoding="utf-8")
    receipt = _receipt(_range_row())

    normalized = persist_post_smoke_target_scaling_state(
        run_dir,
        _output(),
        build_plan=_build_plan(),
        split_receipt=receipt,
    )

    artifact = run_dir / STATE_ARTIFACT
    assert artifact.is_file()
    assert json.loads(artifact.read_text(encoding="utf-8")) == normalized
    assert artifact.read_text(encoding="utf-8").endswith("\n")
    assert generated.read_text(encoding="utf-8") == "producer-owned\n"
    assert {
        path.relative_to(run_dir).as_posix()
        for path in run_dir.rglob("*")
        if path.is_file()
    } == {
        ".pipeline/target_scaling_state.json",
        "method/training.py",
    }


@pytest.mark.parametrize(
    "output_text, code",
    [
        ("smoke completed\n", "target_scaling_marker_count"),
        (
            f"{STATE_MARKER}{{}}\n{STATE_MARKER}{{}}\n",
            "target_scaling_marker_count",
        ),
        (f"{STATE_MARKER}not-json\n", "target_scaling_marker_not_json"),
        (f"{STATE_MARKER}[]\n", "target_scaling_marker_shape"),
        (
            f"{STATE_MARKER}{json.dumps(_state(), sort_keys=True)}\n",
            "target_scaling_marker_not_canonical",
        ),
    ],
)
def test_missing_duplicate_malformed_and_noncanonical_markers_are_producer_invalid(
    tmp_path,
    output_text,
    code,
):
    with pytest.raises(TargetScalingProducerError) as failure:
        persist_post_smoke_target_scaling_state(
            tmp_path,
            output_text,
            build_plan=_build_plan(),
            split_receipt=_receipt(_range_row()),
        )
    assert failure.value.code == code
    assert failure.value.owner == "producer"
    assert failure.value.consume_producer_retry is True
    assert not (tmp_path / STATE_ARTIFACT).exists()


def test_supported_state_disagreements_are_producer_owned_and_do_not_replace_artifact(
    tmp_path,
):
    artifact = tmp_path / STATE_ARTIFACT
    artifact.parent.mkdir(parents=True)
    artifact.write_text('{"preserved":true}\n', encoding="utf-8")

    state = _state()
    state["fitting_lineage"]["stop"] = 4
    _resign(state)
    with pytest.raises(TargetScalingProducerError) as failure:
        persist_post_smoke_target_scaling_state(
            tmp_path,
            _output(state),
            build_plan=_build_plan(),
            split_receipt=_receipt(_range_row()),
        )
    assert failure.value.code == "target_scaling_state_range_disagreement"
    assert failure.value.owner == "producer"
    assert failure.value.consume_producer_retry is True
    assert artifact.read_text(encoding="utf-8") == '{"preserved":true}\n'


def test_state_policy_projection_cannot_self_mint_around_contract_digest():
    state = _state()
    state["mode"] = "none"
    _resign(state)

    with pytest.raises(TargetScalingProducerError) as failure:
        validate_post_smoke_target_scaling_state(
            _output(state),
            build_plan=_build_plan(),
            split_receipt=_receipt(_range_row()),
        )
    assert failure.value.code == "target_scaling_contract_disagreement"
    assert failure.value.consume_producer_retry is True


def test_unsupported_contract_and_lineage_grammar_stay_pipeline_owned(tmp_path):
    unsupported_plan = _build_plan()
    unsupported_plan["target_scaling"]["mode"] = "global"
    with pytest.raises(TargetScalingCoverageError) as contract_failure:
        persist_post_smoke_target_scaling_state(
            tmp_path,
            _output(),
            build_plan=unsupported_plan,
            split_receipt=_receipt(_range_row()),
        )
    assert contract_failure.value.owner == "pipeline"
    assert contract_failure.value.consume_producer_retry is False

    unsupported_path = _build_plan()
    unsupported_path["target_scaling"]["state_artifact"] = "state.json"
    with pytest.raises(TargetScalingCoverageError) as path_failure:
        persist_post_smoke_target_scaling_state(
            tmp_path,
            _output(),
            build_plan=unsupported_path,
            split_receipt=_receipt(_range_row()),
        )
    assert path_failure.value.code == "target_scaling_artifact_path_unsupported"
    assert path_failure.value.consume_producer_retry is False

    with pytest.raises(TargetScalingCoverageError) as lineage_failure:
        persist_post_smoke_target_scaling_state(
            tmp_path,
            _output(),
            build_plan=_build_plan(),
            split_receipt=_receipt(
                _range_row(stop=3),
                _range_row(stop=4),
            ),
        )
    assert lineage_failure.value.code == "fitting_range_grammar_unsupported"
    assert lineage_failure.value.owner == "pipeline"
    assert lineage_failure.value.consume_producer_retry is False
    assert not (tmp_path / STATE_ARTIFACT).exists()


def test_existing_upstream_split_failure_is_preserved_without_retry(tmp_path):
    with pytest.raises(TargetScalingUpstreamError) as failure:
        persist_post_smoke_target_scaling_state(
            tmp_path,
            _output(),
            build_plan=_build_plan(),
            split_receipt=_receipt(status="invalid"),
        )
    assert failure.value.owner == "upstream"
    assert failure.value.consume_producer_retry is False
    assert not (tmp_path / STATE_ARTIFACT).exists()


def test_atomic_replace_failure_preserves_previous_artifact_and_cleans_temp(
    tmp_path,
    monkeypatch,
):
    artifact = tmp_path / STATE_ARTIFACT
    artifact.parent.mkdir(parents=True)
    artifact.write_text('{"old":true}\n', encoding="utf-8")

    def fail_replace(source, destination):
        raise OSError(f"cannot replace {source} with {destination}")

    monkeypatch.setattr(post_smoke.os, "replace", fail_replace)
    with pytest.raises(OSError, match="cannot replace"):
        persist_post_smoke_target_scaling_state(
            tmp_path,
            _output(),
            build_plan=_build_plan(),
            split_receipt=_receipt(_range_row()),
        )
    assert artifact.read_text(encoding="utf-8") == '{"old":true}\n'
    assert list(artifact.parent.glob(f".{artifact.name}.*.tmp")) == []


def test_graph_free_original_unit_proof_joins_permuted_stable_ids():
    state = _state()
    proof = validate_original_unit_comparison(
        state,
        entity_ids=["high", "low"],
        entity_id_root="evaluation.entity_ids",
        scaled_actuals=[[2.0, 3.0], [2.0, 3.0]],
        original_actuals=[[400.0, 600.0], [4.0, 6.0]],
        actual_output_role="location",
        actual_entity_axis=0,
        scaled_predictions=[[1.5, 2.5], [1.5, 2.5]],
        original_predictions=[[300.0, 500.0], [3.0, 5.0]],
        prediction_output_role="location",
        prediction_entity_axis=0,
    )

    assert proof == {
        "schema_version": "1.0.0",
        "status": "valid",
        "state_digest": state["state_digest"],
        "entity_id_root": "evaluation.entity_ids",
        "tolerances": {"relative": 1.0e-6, "absolute": 1.0e-8},
        "actuals": {
            "output_role": "location",
            "entity_axis": 0,
            "shape": [2, 2],
            "max_absolute_error": 0.0,
        },
        "predictions": {
            "output_role": "location",
            "entity_axis": 0,
            "shape": [2, 2],
            "max_absolute_error": 0.0,
        },
    }


def test_original_unit_proof_honors_declared_sample_entity_axis():
    state = _state()
    proof = validate_original_unit_comparison(
        state,
        entity_ids=["low", "high"],
        entity_id_root="evaluation.entity_ids",
        scaled_actuals=[[2.0, 3.0], [2.0, 3.0]],
        original_actuals=[[4.0, 6.0], [400.0, 600.0]],
        actual_output_role="location",
        actual_entity_axis=0,
        scaled_predictions=np.ones((3, 2, 2)),
        original_predictions=np.asarray([
            [[2.0, 2.0], [200.0, 200.0]],
            [[2.0, 2.0], [200.0, 200.0]],
            [[2.0, 2.0], [200.0, 200.0]],
        ]),
        prediction_output_role="samples",
        prediction_entity_axis=1,
    )
    assert proof["predictions"]["entity_axis"] == 1
    assert proof["predictions"]["shape"] == [3, 2, 2]


def test_original_unit_disagreement_is_producer_owned_and_unknown_role_is_coverage():
    common = {
        "state": _state(),
        "entity_ids": ["low", "high"],
        "entity_id_root": "evaluation.entity_ids",
        "scaled_actuals": [[2.0], [2.0]],
        "original_actuals": [[4.0], [400.0]],
        "actual_output_role": "location",
        "actual_entity_axis": 0,
        "scaled_predictions": [[1.0], [1.0]],
        "prediction_output_role": "location",
        "prediction_entity_axis": 0,
    }
    with pytest.raises(TargetScalingProducerError) as disagreement:
        validate_original_unit_comparison(
            **common,
            original_predictions=[[2.0], [201.0]],
        )
    assert disagreement.value.code == "target_output_original_unit_disagreement"
    assert disagreement.value.consume_producer_retry is True

    unsupported = dict(common)
    unsupported["prediction_output_role"] = "quantile"
    with pytest.raises(TargetScalingCoverageError) as role_failure:
        validate_original_unit_comparison(
            **unsupported,
            original_predictions=[[2.0], [200.0]],
        )
    assert role_failure.value.code == "target_output_role_unsupported"
    assert role_failure.value.owner == "pipeline"
    assert role_failure.value.consume_producer_retry is False


def test_original_unit_proof_rejects_duplicate_ids_bad_axis_shape_and_nonfinite():
    base = {
        "state": _state(),
        "entity_ids": ["low", "high"],
        "entity_id_root": "evaluation.entity_ids",
        "scaled_actuals": [[2.0], [2.0]],
        "original_actuals": [[4.0], [400.0]],
        "actual_output_role": "location",
        "actual_entity_axis": 0,
        "scaled_predictions": [[1.0], [1.0]],
        "original_predictions": [[2.0], [200.0]],
        "prediction_output_role": "location",
        "prediction_entity_axis": 0,
    }
    duplicate = dict(base)
    duplicate["entity_ids"] = ["low", "low"]
    with pytest.raises(TargetScalingProducerError, match="duplicate typed stable ids"):
        validate_original_unit_comparison(**duplicate)

    bad_axis = dict(base)
    bad_axis["prediction_entity_axis"] = "rows"
    with pytest.raises(TargetScalingProducerError) as axis_failure:
        validate_original_unit_comparison(**bad_axis)
    assert axis_failure.value.code == "target_output_entity_axis"

    bad_role = dict(base)
    bad_role["prediction_output_role"] = ["location"]
    with pytest.raises(TargetScalingProducerError) as role_failure:
        validate_original_unit_comparison(**bad_role)
    assert role_failure.value.code == "target_output_role_malformed"

    bad_shape = dict(base)
    bad_shape["original_predictions"] = [[2.0, 3.0], [200.0, 300.0]]
    with pytest.raises(TargetScalingProducerError) as shape_failure:
        validate_original_unit_comparison(**bad_shape)
    assert shape_failure.value.code == "target_output_original_unit_shape"

    nonfinite = dict(base)
    nonfinite["original_predictions"] = [[2.0], [float("nan")]]
    with pytest.raises(TargetScalingProducerError) as finite_failure:
        validate_original_unit_comparison(**nonfinite)
    assert finite_failure.value.code == "target_output_nonfinite"


def test_marker_extractor_requires_line_anchor():
    with pytest.raises(TargetScalingProducerError) as failure:
        extract_target_scaling_state(
            f"log prefix {STATE_MARKER}{_canonical(_state())}\n"
        )
    assert failure.value.code == "target_scaling_marker_count"
