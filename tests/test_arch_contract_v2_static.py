"""Static-validator controls for the version-2 architecture contract."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import scripts.validate_arch_contract as validator
from scripts.arch_contract_semantics import (
    parse_semantic_issue,
    semantic_issue_exit_code,
)


def _opaque(label: str) -> dict:
    return {
        "kind": "opaque",
        "type_description": label,
        "reason": "Static completeness fixture; runtime synthesis is out of scope.",
    }


def _callable(
    class_name: str, *, input_names: tuple[str, ...] = ("x",)
) -> dict:
    return {
        "class_name": class_name,
        "constructor_args": {},
        "forward": {
            "input": {name: _opaque(f"{name} input") for name in input_names},
            "output": _opaque(f"{class_name} output"),
        },
    }


def _write_contract(run_dir: Path, contract: dict) -> None:
    pipeline_dir = run_dir / ".pipeline"
    pipeline_dir.mkdir(parents=True, exist_ok=True)
    (pipeline_dir / "arch_contract.json").write_text(
        json.dumps(contract), encoding="utf-8"
    )


def _spec(paradigm_id: str, pluggable_name: str) -> dict:
    return {
        "comparison": {
            "classification": {"id": paradigm_id},
            "pluggable_component": {"name": pluggable_name},
        }
    }


def _v2_active_learning_contract() -> dict:
    return {
        "schema_version": "2.0.0",
        "paradigm_id": "active_learning",
        "dimensions": {},
        "data_loader": {
            "load_data_returns": {
                name: _opaque(name)
                for name in ("x_pool", "y_pool", "x_test", "y_test")
            }
        },
        "architecture": {"model": _callable("Net")},
        "pluggable_component": {
            "name": "select_batch",
            "input": {"x_unlabeled": _opaque("unlabeled pool")},
            "output": _opaque("selected indices"),
        },
        "training_loop": {
            "function_name": "train_from_scratch",
            "input": {
                "x_train": _opaque("training features"),
                "y_train": _opaque("training labels"),
            },
        },
    }


@pytest.mark.parametrize(
    ("contract_factory", "spec"),
    [
        (
            _v2_active_learning_contract,
            _spec("active_learning", "select_batch"),
        ),
    ],
)
def test_v2_typed_requirements_view_accepts_input_output_and_batch_dict_paths(
    tmp_path, contract_factory, spec
):
    """Known-good generic and KD paths resolve against their typed fields."""
    _write_contract(tmp_path, contract_factory())
    assert validator.validate(spec, tmp_path) == []


@pytest.mark.parametrize(
    "interior_keys",
    [
        ["inputs"],
        ["student_inputs", "teacher_inputs", "targets"],
    ],
)
def test_v2_structured_batch_requirement_is_pipeline_owned(interior_keys):
    """Generic and cross-modal KD dict interiors need a later typed grammar."""
    entries, errors = validator._v2_requirements_view({
        "pluggable_component.batch_dict_shape": {
            "type": "dict[str, shape_string]",
            "min_keys": interior_keys,
        }
    })

    assert entries == []
    assert len(errors) == 1
    issue = parse_semantic_issue(errors[0])
    assert issue is not None
    assert issue.owner == "pipeline"
    assert issue.roots[:2] == [
        (
            "arch_contract_requirements.required_blocks."
            "pluggable_component.batch_dict_shape"
        ),
        "arch_contract.pluggable_component.input",
    ]
    assert "opaque batch parameter cannot certify interior keys" in issue.message
    assert semantic_issue_exit_code(errors) == 3


def test_v2_typed_requirements_view_reports_missing_required_input_key(tmp_path):
    """Known-bad completeness still names the typed consumer and missing key."""
    contract = _v2_active_learning_contract()
    contract["pluggable_component"]["input"] = {}
    _write_contract(tmp_path, contract)

    errors = validator.validate(
        _spec("active_learning", "select_batch"), tmp_path
    )

    assert any(
        "arch_contract.pluggable_component.input" in error
        and "x_unlabeled" in error
        for error in errors
    ), errors


def test_missing_taxonomy_build_plan_is_pipeline_owned(tmp_path, monkeypatch):
    """A missing validator authority cannot be repaired by contract retries."""
    contract = _v2_active_learning_contract()
    contract["paradigm_id"] = "unregistered_family"
    _write_contract(tmp_path, contract)
    monkeypatch.setattr(validator, "load_build_plan", lambda *_args, **_kwargs: None)

    errors = validator.validate(
        _spec("unregistered_family", "select_batch"), tmp_path
    )

    assert len(errors) == 1
    issue = parse_semantic_issue(errors[0])
    assert issue is not None
    assert issue.owner == "pipeline"
    assert issue.roots[:2] == [
        "spec.comparison.classification.id",
        "docs/ssot/taxonomies.yaml",
    ]
    assert semantic_issue_exit_code(errors) == 3


def test_legacy_requirements_paths_remain_unchanged(tmp_path):
    """Legacy control: v1 continues to use shape-string field names."""
    contract = {
        "schema_version": "1.0.0",
        "paradigm_id": "active_learning",
        "data_loader": {
            "load_data_returns": {
                name: "(fixture,)"
                for name in ("x_pool", "y_pool", "x_test", "y_test")
            }
        },
        "architecture": {
            "model": {
                "class_name": "Net",
                "forward": {
                    "input": {"x": "(B, n_features)"},
                    "output_type": "tensor",
                    "output_shape": "(B, n_classes)",
                },
            }
        },
        "pluggable_component": {
            "name": "select_batch",
            "input_shapes": {"x_unlabeled": "(N_pool, n_features)"},
            "output_shape": "list[int]",
        },
        "training_loop": {
            "function_name": "train_from_scratch",
            "input_shapes": {
                "x_train": "(N, n_features)",
                "y_train": "(N,)",
            },
        },
    }
    _write_contract(tmp_path, contract)

    assert validator.validate(
        _spec("active_learning", "select_batch"), tmp_path
    ) == []


def test_v2_family_component_single_and_multi_forms_are_checked_safely(
    tmp_path, monkeypatch
):
    """Single values prove presence; only multi values expose named entries."""
    contract = _v2_active_learning_contract()
    contract["paradigm_id"] = "typed_family"
    contract["family_components"] = {
        "reward_function": {"value": _opaque("reward callable")},
        "named_state": {"entries": {"active": _opaque("active flag")}},
    }
    _write_contract(tmp_path, contract)

    declarations = {
        "reward_function": {"required": True, "required_entries": []},
        "named_state": {"required": True, "required_entries": ["active"]},
    }

    def _plan(*_args, **_kwargs):
        return {
            "pluggable_component": {"name": "select_batch"},
            "arch_contract_requirements": {
                "required_blocks": {},
                "family_components": declarations,
            },
        }

    monkeypatch.setattr(validator, "load_build_plan", _plan)
    spec = _spec("typed_family", "select_batch")
    assert validator.validate(spec, tmp_path) == []

    declarations["reward_function"] = {
        "required": True,
        "required_entries": ["named_output"],
    }
    errors = validator.validate(spec, tmp_path)
    assert any(
        "family_components.reward_function.entries.named_output" in error
        and "missing" in error
        for error in errors
    ), errors


def test_v2_opaque_output_cannot_satisfy_structured_output_keys(tmp_path):
    """Unsupported control: typed opaque output does not launder key shape."""
    contract = _v2_active_learning_contract()
    contract["paradigm_id"] = "domain_adaptation"
    contract["architecture"] = {"detector": _callable("Detector")}
    contract["data_loader"] = {"load_data_returns": {}}
    contract["pluggable_component"] = {
        "name": "generate_pseudo_labels",
        "input": {},
        "output": _opaque("pseudo-label records"),
    }
    contract.pop("training_loop")
    _write_contract(tmp_path, contract)

    errors = validator.validate(
        _spec("domain_adaptation", "generate_pseudo_labels"), tmp_path
    )

    structured = [parse_semantic_issue(error) for error in errors]
    assert len(structured) == 3, errors
    assert all(issue is not None for issue in structured)
    assert all(issue.owner == "pipeline" for issue in structured if issue)
    assert semantic_issue_exit_code(errors) == 3
    assert all(
        "opaque descriptor cannot certify" in issue.message
        for issue in structured
        if issue
    )
    assert all(
        issue.roots[0].startswith(
            "arch_contract_requirements.required_blocks."
            "architecture.detector.forward.output_keys."
        )
        and issue.roots[1] == "arch_contract.architecture.detector.forward.output"
        for issue in structured
        if issue
    )
