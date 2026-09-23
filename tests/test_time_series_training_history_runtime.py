"""Stage-2.d execution coverage for the fixed forecasting history seam."""

from __future__ import annotations

import copy

import scripts.validate_arch_contract_runtime as runtime_validator
from scripts.arch_contract_semantics import parse_semantic_issue
from scripts.build_plan import STATIC_PLAN_BY_PARADIGM
from scripts.validate_arch_contract_runtime import validate
from tests.test_time_series_target_scaling_runtime import (
    _write_graph_free_package,
)
import pytest

pytestmark = pytest.mark.probe_runtime


def test_real_fitting_accepts_different_valid_training_outcomes(tmp_path):
    spec, run_dir = _write_graph_free_package(
        tmp_path,
        alternate_history_outcomes=True,
    )

    assert validate(spec, run_dir) == []


def test_static_gate_rejects_live_but_discarded_history_record(tmp_path):
    spec, run_dir = _write_graph_free_package(
        tmp_path,
        discard_history=True,
    )

    errors = validate(spec, run_dir)

    assert any(
        "record_training_history does not flow" in error
        and "call-and-discard" in error
        for error in errors
    )
    assert all("contract_code_disagreement" in error for error in errors)


def test_static_gate_rejects_different_training_result_key(tmp_path):
    spec, run_dir = _write_graph_free_package(
        tmp_path,
        history_result_key="history",
    )

    errors = validate(spec, run_dir)

    assert any(
        "exact key 'training_history'" in error
        and "different result key" in error
        for error in errors
    )
    assert all("contract_code_disagreement" in error for error in errors)


def test_interprocedural_history_flow_gap_is_pipeline_owned(tmp_path):
    spec, run_dir = _write_graph_free_package(
        tmp_path,
        history_via_helper=True,
    )

    errors = validate(spec, run_dir)

    issues = [parse_semantic_issue(error) for error in errors]
    assert len(issues) == 1
    assert issues[0] is not None
    assert issues[0].code == "unsupported_validator_feature"
    assert issues[0].owner == "pipeline"
    assert "separate helper function" in issues[0].message


def test_real_fitting_rejects_exact_history_input_disagreement(tmp_path):
    spec, run_dir = _write_graph_free_package(
        tmp_path,
        drift_history_config=True,
    )

    errors = validate(spec, run_dir)

    assert any(
        "must receive the exact pipeline-owned model identity, seed, config"
        in error
        for error in errors
    )
    assert all("contract_code_disagreement" in error for error in errors)


def test_real_fitting_rejects_fitting_range_drift(tmp_path):
    spec, run_dir = _write_graph_free_package(
        tmp_path,
        drift_history_fitting_range=True,
    )

    errors = validate(spec, run_dir)

    assert any(
        "must receive the exact pipeline-owned model identity, seed, config"
        in error
        for error in errors
    )
    assert all("contract_code_disagreement" in error for error in errors)


def test_real_fitting_requires_model_beside_state_and_history(tmp_path):
    spec, run_dir = _write_graph_free_package(
        tmp_path,
        omit_model_result=True,
    )

    errors = validate(spec, run_dir)

    assert any(
        "retain the trained model beside target_scaling_state and "
        "training_history" in error
        for error in errors
    )
    assert all("contract_code_disagreement" in error for error in errors)


def test_family_owned_history_normalization_gap_is_pipeline_owned(
    tmp_path,
    monkeypatch,
):
    spec, run_dir = _write_graph_free_package(tmp_path)
    build_plan = copy.deepcopy(
        STATIC_PLAN_BY_PARADIGM["time_series_forecasting"]
    )
    del build_plan["training_history_execution"]
    monkeypatch.setattr(
        runtime_validator,
        "load_build_plan",
        lambda *args, **kwargs: build_plan,
    )

    errors = runtime_validator.validate(spec, run_dir)

    issues = [parse_semantic_issue(error) for error in errors]
    assert len(issues) == 1
    assert issues[0] is not None
    assert issues[0].code == "unsupported_validator_feature"
    assert issues[0].owner == "pipeline"
