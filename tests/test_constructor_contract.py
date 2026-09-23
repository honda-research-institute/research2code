"""R2C-075 — declared architecture construction at the Stage-2d seam.

Schema 1.1 contracts name every constructor keyword and its value source.
Schema 1.0 contracts retain the historical candidate/direct-construction
adapter so checked-in deliveries keep their established behavior.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from schemas.arch_contract import ArchContract, ConstructorArg
from scripts.validate_arch_contract_runtime import validate

pytestmark = pytest.mark.probe_runtime


_FORWARD = {
    "input": {"x": "(B, n_features)"},
    "output_type": "tensor",
    "output_shape": "(B, n_features)",
}


def _block(
    class_name: str = "Net",
    constructor_args: dict | None = None,
) -> dict:
    block = {"class_name": class_name, "forward": _FORWARD}
    if constructor_args is not None:
        block["constructor_args"] = constructor_args
    return block


def _contract(
    architecture: dict[str, dict], *, version: str = "1.1.0"
) -> dict:
    return {
        "schema_version": version,
        "paradigm_id": "test_family",
        "data_loader": {"load_data_returns": {}},
        "architecture": architecture,
        "pluggable_component": {
            "name": "contribution",
            "input_shapes": {},
            "output_shape": "scalar",
        },
    }


def _write_package(
    tmp_path: Path,
    *,
    contract: dict,
    model_src: str,
    training_src: str,
    init_src: str | None = None,
) -> Path:
    run_dir = tmp_path / "run"
    method_dir = run_dir / "method"
    pipeline_dir = run_dir / ".pipeline"
    method_dir.mkdir(parents=True)
    pipeline_dir.mkdir()
    (method_dir / "model.py").write_text(model_src, encoding="utf-8")
    (method_dir / "training.py").write_text(training_src, encoding="utf-8")
    (method_dir / "data.py").write_text("", encoding="utf-8")
    if init_src is None:
        init_src = "from .model import *\nfrom .training import *\n"
    (method_dir / "__init__.py").write_text(init_src, encoding="utf-8")
    (pipeline_dir / "arch_contract.json").write_text(
        json.dumps(contract), encoding="utf-8"
    )
    return run_dir


_NET = """
class Net:
    def __call__(self, x):
        return x
"""


def _paper_dimensions(**values: int) -> dict:
    return {
        "critical_requirements": {
            "param_glossary": [
                {"name": name, "paper_value": value}
                for name, value in values.items()
            ]
        }
    }


def test_constructor_arg_exactly_one_source_preserves_json_literals():
    assert ConstructorArg.model_validate({"literal": None}).model_fields_set == {
        "literal"
    }
    assert ConstructorArg.model_validate({"literal": [16, 8]}).literal == [16, 8]
    assert ConstructorArg.model_validate({"dimension": " P "}).dimension == "P"
    with pytest.raises(ValidationError, match="exactly one"):
        ConstructorArg.model_validate({})
    with pytest.raises(ValidationError, match="exactly one"):
        ConstructorArg.model_validate({"literal": 8, "dimension": "P"})


def test_schema_versions_separate_current_declarations_from_legacy_reads():
    current_missing = _contract({"model": _block()})
    with pytest.raises(ValidationError, match="requires constructor_args"):
        ArchContract.model_validate(current_missing)

    current_null = _contract({"model": _block()})
    current_null["architecture"]["model"]["constructor_args"] = None
    with pytest.raises(ValidationError, match="requires constructor_args"):
        ArchContract.model_validate(current_null)

    legacy = _contract({"model": _block()}, version="1.0.0")
    assert ArchContract.model_validate(legacy).schema_version == "1.0.0"

    omitted_version = _contract({"model": _block()}, version="1.0.0")
    omitted_version.pop("schema_version")
    assert (
        ArchContract.model_validate(omitted_version).schema_version == "1.0.0"
    )

    legacy_with_current_field = _contract(
        {"model": _block(constructor_args={})}, version="1.0.0"
    )
    with pytest.raises(ValidationError, match="requires schema_version 1.1.0"):
        ArchContract.model_validate(legacy_with_current_field)

    legacy_with_null_field = _contract({"model": _block()}, version="1.0.0")
    legacy_with_null_field["architecture"]["model"]["constructor_args"] = None
    with pytest.raises(ValidationError, match="requires schema_version 1.1.0"):
        ArchContract.model_validate(legacy_with_null_field)


def test_pdfgnn_required_builder_arguments_resolve_from_exact_declarations(
    tmp_path: Path,
):
    contract = _contract(
        {
            "model": _block(
                constructor_args={
                    "num_demand_lags": {"dimension": "P"},
                    "num_static_features": {"literal": 9},
                    "num_time_varying_features": {"literal": 14},
                    "gnn_hidden_sizes": {"literal": [16, 8]},
                }
            )
        }
    )
    training = """
from .model import Net
def build_model(num_demand_lags, num_static_features,
                num_time_varying_features, gnn_hidden_sizes=None,
                dropout=0.2):
    got = (num_demand_lags, num_static_features,
           num_time_varying_features, gnn_hidden_sizes, dropout)
    expected = (10, 9, 14, [16, 8], 0.2)
    if got != expected:
        raise AssertionError("constructor mismatch: %r" % (got,))
    return Net()
"""
    run_dir = _write_package(
        tmp_path, contract=contract, model_src=_NET, training_src=training
    )
    assert validate(_paper_dimensions(P=10), run_dir) == []


def test_empty_constructor_mapping_is_intentional_and_leaks_no_candidates(
    tmp_path: Path,
):
    contract = _contract({"model": _block(constructor_args={})})
    training = """
from .model import Net
def build_model(hidden_dim=99, optional_width=7):
    if (hidden_dim, optional_width) != (99, 7):
        raise AssertionError("undeclared candidate leaked into builder")
    return Net()
"""
    run_dir = _write_package(
        tmp_path, contract=contract, model_src=_NET, training_src=training
    )
    assert validate({}, run_dir) == []


@pytest.mark.parametrize(
    ("constructor_args", "signature", "message"),
    [
        ({}, "required", "missing required keyword argument"),
        (
            {"invented": {"literal": 3}},
            "required=3",
            "unexpected constructor_args keyword",
        ),
        (
            {"required": {"literal": 3}},
            "required, /",
            "positional-only",
        ),
    ],
)
def test_current_constructor_declaration_must_match_exact_signature(
    tmp_path: Path,
    constructor_args: dict,
    signature: str,
    message: str,
):
    contract = _contract(
        {"model": _block(constructor_args=constructor_args)}
    )
    training = (
        "from .model import Net\n"
        f"def build_model({signature}):\n"
        "    return Net()\n"
    )
    run_dir = _write_package(
        tmp_path, contract=contract, model_src=_NET, training_src=training
    )
    errors = validate({}, run_dir)
    assert any(message in error for error in errors), errors


def test_variadic_keyword_builder_accepts_only_the_declared_extra_values(
    tmp_path: Path,
):
    contract = _contract(
        {
            "model": _block(
                constructor_args={
                    "required": {"literal": 3},
                    "future_option": {"literal": 5},
                }
            )
        }
    )
    training = """
from .model import Net
def build_model(required, **kwargs):
    if required != 3 or kwargs != {"future_option": 5}:
        raise AssertionError("wrong exact kwargs: %r %r" % (required, kwargs))
    return Net()
"""
    run_dir = _write_package(
        tmp_path, contract=contract, model_src=_NET, training_src=training
    )
    assert validate({}, run_dir) == []


def test_declared_builder_returning_none_is_not_a_construction_success(
    tmp_path: Path,
):
    contract = _contract({"model": _block(constructor_args={})})
    run_dir = _write_package(
        tmp_path,
        contract=contract,
        model_src=_NET,
        training_src="def build_model(): return None\n",
    )
    errors = validate({}, run_dir)
    assert any("returned None" in error for error in errors), errors


def test_unresolved_dimension_fails_before_generated_package_import(tmp_path: Path):
    contract = _contract(
        {
            "model": _block(
                constructor_args={"width": {"dimension": "unknown_width"}}
            )
        }
    )
    init_src = """
from pathlib import Path
Path(__file__).with_name("IMPORTED").write_text("imported", encoding="utf-8")
from .model import *
from .training import *
"""
    training = "from .model import Net\ndef build_model(width): return Net()\n"
    run_dir = _write_package(
        tmp_path,
        contract=contract,
        model_src=_NET,
        training_src=training,
        init_src=init_src,
    )
    (run_dir / ".pipeline" / "params.json").write_text(
        json.dumps({"unknown_width": {"value": 23}}), encoding="utf-8"
    )

    errors = validate({}, run_dir)
    assert any("unknown_width" in error and "DEFAULT_BINDING=8" in error for error in errors)
    assert not (run_dir / "method" / "IMPORTED").exists()


def test_bundle_overrides_only_the_explicit_time_axis_semantic_dimension(
    tmp_path: Path,
):
    contract = _contract(
        {
            "model": _block(
                constructor_args={
                    "feature_width": {"dimension": "L"},
                    "time_steps": {"dimension": "time_axis_steps"},
                }
            )
        }
    )
    training = """
from .model import Net
def build_model(feature_width, time_steps):
    if (feature_width, time_steps) != (14, 31):
        raise AssertionError("semantic dimensions crossed: %r" % ((feature_width, time_steps),))
    return Net()
"""
    run_dir = _write_package(
        tmp_path, contract=contract, model_src=_NET, training_src=training
    )
    provenance = run_dir / "method" / "example_data" / "PROVENANCE.json"
    provenance.parent.mkdir()
    provenance.write_text(
        json.dumps(
            {
                "files": [
                    {"file": "series.csv", "time_axis": {"steps_kept": 31}}
                ]
            }
        ),
        encoding="utf-8",
    )
    # Neither a late params value nor the bundle's legacy L alias may affect
    # current constructor resolution.
    (run_dir / ".pipeline" / "params.json").write_text(
        json.dumps(
            {
                "L": {"value": 999},
                "time_axis_steps": {"value": 888},
            }
        ),
        encoding="utf-8",
    )
    assert validate(_paper_dimensions(L=14, time_axis_steps=12), run_dir) == []


def test_constructor_mappings_are_block_local_for_kd_builders(tmp_path: Path):
    contract = _contract(
        {
            "student": _block(
                "Student", {"width": {"literal": 3}}
            ),
            "teacher": _block(
                "Teacher", {"width": {"literal": 5}}
            ),
        }
    )
    model_src = """
class _Base:
    def __call__(self, x):
        return x
class Student(_Base):
    pass
class Teacher(_Base):
    pass
"""
    training = """
from .model import Student, Teacher
def build_student(width):
    if width != 3: raise AssertionError("student received %r" % width)
    return Student()
def build_teacher(width):
    if width != 5: raise AssertionError("teacher received %r" % width)
    return Teacher()
"""
    run_dir = _write_package(
        tmp_path, contract=contract, model_src=model_src, training_src=training
    )
    assert validate({}, run_dir) == []


def test_multiblock_contract_can_mix_an_exact_builder_and_direct_class(
    tmp_path: Path,
):
    contract = _contract(
        {
            "student": _block(
                "Student", {"student_width": {"literal": 3}}
            ),
            "teacher": _block(
                "Teacher", {"teacher_width": {"literal": 5}}
            ),
        }
    )
    model_src = """
class Student:
    def __call__(self, x):
        return x
class Teacher:
    def __init__(self, teacher_width):
        if teacher_width != 5: raise AssertionError("wrong teacher width")
    def __call__(self, x):
        return x
"""
    training = """
from .model import Student
def build_student(student_width):
    if student_width != 3: raise AssertionError("wrong student width")
    return Student()
"""
    run_dir = _write_package(
        tmp_path, contract=contract, model_src=model_src, training_src=training
    )
    assert validate({}, run_dir) == []


def test_sole_builder_is_not_routed_to_an_unmatched_multiblock_role(
    tmp_path: Path,
):
    contract = _contract(
        {
            "student": _block("Student", constructor_args={}),
            "teacher": _block("Teacher", constructor_args={}),
        }
    )
    model_src = """
class Student:
    def __call__(self, x):
        return x
class Teacher:
    def __init__(self, required_teacher_state):
        self.required_teacher_state = required_teacher_state
    def __call__(self, x):
        return x
"""
    training = """
from .model import Student
def build_student():
    return Student()
"""
    run_dir = _write_package(
        tmp_path, contract=contract, model_src=model_src, training_src=training
    )

    errors = validate({}, run_dir)

    assert any(
        "architecture.teacher.constructor_args via class Teacher" in error
        and "missing required keyword argument" in error
        for error in errors
    ), errors
    assert not any(
        "architecture.teacher.constructor_args via builder build_student"
        in error
        for error in errors
    ), errors


def test_current_direct_class_constructor_uses_the_declared_mapping(tmp_path: Path):
    contract = _contract(
        {"model": _block(constructor_args={"width": {"literal": 5}})}
    )
    model_src = """
class Net:
    def __init__(self, width):
        if width != 5: raise AssertionError("wrong width")
    def __call__(self, x):
        return x
"""
    run_dir = _write_package(
        tmp_path, contract=contract, model_src=model_src, training_src=""
    )
    assert validate({}, run_dir) == []


def test_legacy_builder_keeps_candidate_construction_behavior(tmp_path: Path):
    contract = _contract({"model": _block()}, version="1.0.0")
    training = """
from .model import Net
def build_model(input_dim, n_classes):
    if (input_dim, n_classes) != (8, 3):
        raise AssertionError("legacy candidates changed")
    return Net()
"""
    run_dir = _write_package(
        tmp_path, contract=contract, model_src=_NET, training_src=training
    )
    assert validate({}, run_dir) == []


def test_legacy_direct_class_required_argument_remains_a_skip(tmp_path: Path):
    """Motion-planning-style 1.0 classes had no constructor declaration."""
    contract = _contract({"model": _block()}, version="1.0.0")
    model_src = """
class Net:
    def __init__(self, required_environment):
        self.required_environment = required_environment
    def __call__(self, x):
        return x
"""
    run_dir = _write_package(
        tmp_path, contract=contract, model_src=model_src, training_src=""
    )
    assert validate({}, run_dir) == []
