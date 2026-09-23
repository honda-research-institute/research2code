"""Regression test: `scripts/validate_architecture_coder_output.py` must
drive its class-count + function-name checks from the paradigm's
`package_manifest`, NOT from a hardcoded single-class universal template.

The 2026-05-19 bev-distill halt traced to this exact bug: the validator
hardcoded "exactly one nn.Module subclass + build_model + train_from_scratch"
(matching active_learning's shape) and rejected correct knowledge_distillation
output (student + teacher + build_student + build_teacher +
train_with_distillation). The halt-judge classified it as `pipeline_bug` and
halted with a precise diagnosis.

These tests exercise both shapes (single-class AL + multi-class KD) against
the taxonomy build plans, plus an explicit failure case where a single-class
paradigm gets multi-class output (extra public class → reject).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent


def _make_spec(paradigm_id: str) -> dict:
    """Minimal method_spec.json sufficient for the validator's needs.
    The validator reads classification id and a few other fields."""
    return {
        "schema_version": "1.0.0",
        "paper": {"title": "T", "authors": ["A"], "year": 2026, "venue": "V"},
        "core_method": {"description": "x", "key_elements": []},
        "comparison": {
            "classification": {
                "id": paradigm_id,
            },
            "pluggable_component": {"name": "compute_distillation_loss"},
        },
        "critical_requirements": {
            "model": {"specific_features": []},
            "data": {"specific_features": []},
        },
        "hyperparameters": [],
    }


def _write_minimal_paper_map(pipeline_dir: Path) -> None:
    """Validator reads paper_map.json for paper-element ID validation."""
    pipeline_dir.mkdir(parents=True, exist_ok=True)
    (pipeline_dir / "paper_map.json").write_text(
        json.dumps({"schema_version": "1.0.0", "elements": []}, indent=2)
    )


def _write_arch_contract(
    pipeline_dir: Path,
    *,
    paradigm_id: str,
    class_name: str = "TestModel",
    pluggable_name: str = "compute_distillation_loss",
    training_function: str = "train_from_scratch",
) -> None:
    """Write a minimal newly-authored schema-2 architecture contract.

    Stage 2.b validates the versioned outer contract and requires v2 for new
    producer output. Deeper paradigm-specific completeness remains covered by
    Stage 2.d tests, so intentionally non-executable values are represented as
    honest opaque descriptors rather than invented family-specific tensors.
    """
    pipeline_dir.mkdir(parents=True, exist_ok=True)
    opaque = {
        "kind": "opaque",
        "type_description": "fixture value",
        "reason": "Stage-2b manifest test does not execute contract values.",
    }
    contract = {
        "schema_version": "2.0.0",
        "paradigm_id": paradigm_id,
        "dimensions": {},
        "data_loader": {"load_data_returns": {}},
        "architecture": {
            "model": {
                "class_name": class_name,
                "constructor_args": {},
                "forward": {
                    "input": {},
                    "output": opaque,
                },
            },
        },
        "pluggable_component": {
            "name": pluggable_name,
            "input": {},
            "output": opaque,
        },
        "training_loop": {
            "function_name": training_function,
            "input": {},
            "output": opaque,
        },
    }
    (pipeline_dir / "arch_contract.json").write_text(json.dumps(contract, indent=2))


def _write_legacy_arch_contract(
    pipeline_dir: Path,
    *,
    schema_version: str,
    paradigm_id: str = "active_learning",
) -> None:
    """Write one schema-valid legacy contract for the activation controls."""
    pipeline_dir.mkdir(parents=True, exist_ok=True)
    block = {
        "class_name": "MLPClassifier",
        "forward": {
            "input": {"x": "(B, n_features)"},
            "output_type": "tensor",
            "output_shape": "(B, n_classes)",
        },
    }
    if schema_version == "1.1.0":
        block["constructor_args"] = {}
    contract = {
        "schema_version": schema_version,
        "paradigm_id": paradigm_id,
        "data_loader": {"load_data_returns": {}},
        "architecture": {"model": block},
        "pluggable_component": {
            "name": "acquisition_function",
            "input_shapes": {},
            "output_shape": "list[int] of length batch_size",
        },
        "training_loop": {
            "function_name": "train_from_scratch",
            "input_shapes": {},
        },
    }
    (pipeline_dir / "arch_contract.json").write_text(json.dumps(contract, indent=2))


def _write_valid_active_learning_outputs(run_dir: Path) -> Path:
    """Write valid Stage-2b Python outputs and return their pipeline dir."""
    method_dir = run_dir / "method"
    pipeline_dir = run_dir / ".pipeline"
    method_dir.mkdir(parents=True)
    (method_dir / "model.py").write_text("""
import torch.nn as nn

class MLPClassifier(nn.Module):
    def __init__(self, input_dim, n_classes):
        super().__init__()
        self.fc = nn.Linear(input_dim, n_classes)
    def forward(self, x):
        return self.fc(x)
""")
    (method_dir / "training.py").write_text("""
import torch
from .model import MLPClassifier

def build_model(input_dim, n_classes):
    return MLPClassifier(input_dim, n_classes)

def train_from_scratch(x_train, y_train, *, learning_rate=1e-3, num_epochs=10, batch_size=32, seed=0):
    torch.manual_seed(seed)
    model = MLPClassifier(x_train.shape[1], int(y_train.max()) + 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    return model
""")
    _write_minimal_paper_map(pipeline_dir)
    return pipeline_dir


# ---------------------------------------------------------------------------
# Single-class paradigm (active_learning) — preserves legacy behavior
# ---------------------------------------------------------------------------


def test_validator_accepts_v2_contract_for_new_stage_2b_output(tmp_path):
    """Active learning declares 1 class + build_model + train_from_scratch.
    Valid Python output plus a newly-authored typed contract passes."""
    from scripts.validate_architecture_coder_output import validate

    run_dir = tmp_path / "run"
    pipeline_dir = _write_valid_active_learning_outputs(run_dir)
    _write_arch_contract(pipeline_dir, paradigm_id="active_learning")

    spec = _make_spec(paradigm_id="active_learning")
    errors = validate(spec, run_dir, REPO_ROOT)
    assert errors == [], f"expected no errors, got: {errors}"


@pytest.mark.parametrize("schema_version", ["1.0.0", "1.1.0"])
def test_validator_reads_legacy_contract_but_requires_v2_for_new_output(
    tmp_path, schema_version,
):
    """A schema-valid v1 contract gets one producer activation diagnostic.

    Continuing after the version finding is important: the shared loader keeps
    the historical contract readable and this validator can still report any
    independent structural defects during resume diagnosis.
    """
    from scripts.validate_architecture_coder_output import validate

    run_dir = tmp_path / "run"
    pipeline_dir = _write_valid_active_learning_outputs(run_dir)
    _write_legacy_arch_contract(
        pipeline_dir,
        schema_version=schema_version,
    )

    errors = validate(_make_spec("active_learning"), run_dir, REPO_ROOT)

    assert len(errors) == 1, errors
    assert f"legacy schema_version {schema_version!r}" in errors[0]
    assert "new Stage 2.b architecture-coder output must use schema_version '2.0.0'" in errors[0]
    assert "archived or resumed consumers" in errors[0]


def test_validator_keeps_unknown_contract_version_as_schema_failure(tmp_path):
    """An unknown version is malformed input, not a legacy activation case."""
    from scripts.validate_architecture_coder_output import validate

    run_dir = tmp_path / "run"
    pipeline_dir = _write_valid_active_learning_outputs(run_dir)
    _write_arch_contract(pipeline_dir, paradigm_id="active_learning")
    contract_path = pipeline_dir / "arch_contract.json"
    raw = json.loads(contract_path.read_text())
    raw["schema_version"] = "9.0.0"
    contract_path.write_text(json.dumps(raw, indent=2))

    errors = validate(_make_spec("active_learning"), run_dir, REPO_ROOT)

    schema_errors = [
        error for error in errors
        if ".pipeline/arch_contract.json failed schema validation" in error
    ]
    assert len(schema_errors) == 1, errors
    assert "unsupported arch_contract schema_version '9.0.0'" in schema_errors[0]
    assert not any("new Stage 2.b architecture-coder output" in error for error in errors)


def test_validator_rejects_present_but_schema_invalid_arch_contract(tmp_path):
    """A present but narrative/free-form arch_contract must stay in the
    architecture-coder fix loop.

    Reproduces the 2026-06-29 GBALD shape: the file existed, but it used
    keys such as `architecture.name` and omitted the universal
    `paradigm_id`, `data_loader`, and `pluggable_component` skeleton.
    """
    from scripts.validate_architecture_coder_output import validate

    run_dir = tmp_path / "run"
    method_dir = run_dir / "method"
    pipeline_dir = run_dir / ".pipeline"
    method_dir.mkdir(parents=True)
    pipeline_dir.mkdir(parents=True)

    (method_dir / "model.py").write_text("""
import torch.nn as nn

class MLPClassifier(nn.Module):
    def __init__(self, input_dim, n_classes):
        super().__init__()
        self.fc = nn.Linear(input_dim, n_classes)
    def forward(self, x):
        return self.fc(x)
""")
    (method_dir / "training.py").write_text("""
import torch
from .model import MLPClassifier

def build_model(input_dim, n_classes):
    return MLPClassifier(input_dim, n_classes)

def train_from_scratch(x_train, y_train, *, seed=0):
    torch.manual_seed(seed)
    model = MLPClassifier(x_train.shape[1], int(y_train.max()) + 1)
    optimizer = torch.optim.Adam(model.parameters())
    return model
""")
    (pipeline_dir / "arch_contract.json").write_text(json.dumps({
        "schema_version": "1.0.0",
        "architecture": {
            "name": "GbaldNet",
            "kind": "cnn",
        },
        "build": {
            "module": "method.model",
            "function": "build_arch",
        },
    }, indent=2))
    _write_minimal_paper_map(pipeline_dir)

    spec = _make_spec(paradigm_id="active_learning")
    errors = validate(spec, run_dir, REPO_ROOT)
    assert any(
        ".pipeline/arch_contract.json failed schema validation" in e
        for e in errors
    ), f"expected arch_contract schema error, got: {errors}"
    assert any("paradigm_id" in e for e in errors), (
        f"expected missing universal skeleton fields in error, got: {errors}"
    )


# ---------------------------------------------------------------------------
# Multi-class paradigm (knowledge_distillation) — the regression case
# ---------------------------------------------------------------------------


def test_validator_passes_for_multi_class_knowledge_distillation(tmp_path):
    """**bev-distill 2026-05-19 regression**: KD declares 2 classes (student +
    teacher) + 3 functions (build_student, build_teacher,
    train_with_distillation). The validator MUST accept this shape."""
    from scripts.validate_architecture_coder_output import validate

    run_dir = tmp_path / "run"
    method_dir = run_dir / "method"
    pipeline_dir = run_dir / ".pipeline"
    method_dir.mkdir(parents=True)

    (method_dir / "model.py").write_text("""
import torch.nn as nn

class StudentNet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.fc = nn.Linear(10, num_classes)
    def forward(self, x):
        return self.fc(x)

class TeacherNet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.fc = nn.Linear(10, num_classes)
    def forward(self, x):
        return self.fc(x)
""")
    (method_dir / "training.py").write_text("""
import torch
from .model import StudentNet, TeacherNet

def build_student(input_dim, n_classes, **kw):
    return StudentNet(n_classes)

def build_teacher(input_dim, n_classes, **kw):
    return TeacherNet(n_classes)

def train_with_distillation(student, teacher, x_train, y_train, *,
                             distillation_loss_fn, learning_rate=1e-3,
                             num_epochs=10, batch_size=32, seed=0, **kw):
    torch.manual_seed(seed)
    optimizer = torch.optim.Adam(student.parameters(), lr=learning_rate)
    return student
""")
    _write_arch_contract(pipeline_dir, paradigm_id="knowledge_distillation")
    _write_minimal_paper_map(pipeline_dir)

    spec = _make_spec(paradigm_id="knowledge_distillation")
    errors = validate(spec, run_dir, REPO_ROOT)
    assert errors == [], (
        f"expected no errors for multi-class KD output, got: {errors}. "
        f"This is the bev-distill 2026-05-19 cascade root cause: validator "
        f"was hardcoded to the single-class active_learning template."
    )


def test_validator_resolves_methods_and_inherits_through_local_base(tmp_path):
    """detr-distill 2026-07-08 regression: both public KD classes inherit
    `forward` from a shared private `_Base(nn.Module)` class. The body-only
    method lookup and the direct-bases-only inherits check both rejected
    this idiomatic package. Methods and nn.Module inheritance must resolve
    through same-module base classes."""
    from scripts.validate_architecture_coder_output import validate

    run_dir = tmp_path / "run"
    method_dir = run_dir / "method"
    pipeline_dir = run_dir / ".pipeline"
    method_dir.mkdir(parents=True)

    (method_dir / "model.py").write_text("""
import torch.nn as nn

class _Base(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.fc = nn.Linear(10, num_classes)
    def forward(self, x):
        return self.fc(x)

class StudentNet(_Base):
    pass

class TeacherNet(_Base):
    pass
""")
    (method_dir / "training.py").write_text("""
import torch
from .model import StudentNet, TeacherNet

def build_student(input_dim, n_classes, **kw):
    return StudentNet(n_classes)

def build_teacher(input_dim, n_classes, **kw):
    return TeacherNet(n_classes)

def train_with_distillation(student, teacher, x_train, y_train, *,
                             distillation_loss_fn, learning_rate=1e-3,
                             num_epochs=10, batch_size=32, seed=0, **kw):
    torch.manual_seed(seed)
    return student
""")
    _write_arch_contract(pipeline_dir, paradigm_id="knowledge_distillation")
    _write_minimal_paper_map(pipeline_dir)

    spec = _make_spec(paradigm_id="knowledge_distillation")
    errors = validate(spec, run_dir, REPO_ROOT)
    assert errors == [], f"expected no errors for base-class KD output, got: {errors}"


# ---------------------------------------------------------------------------
# Non-ML paradigm (motion_planning) — classes do NOT inherit from nn.Module
# ---------------------------------------------------------------------------


def test_validator_passes_for_non_module_classes_motion_planning(tmp_path):
    """**pdwa 2026-05-26 regression**: motion_planning declares plain-Python
    classes (SystemDynamics + CollisionModel) with paradigm-specific
    `required_methods` (`step`/`step_jacobian`, `is_in_collision`/
    `distance_to_obstacle`) — NOT nn.Module subclasses with `forward()`.

    The prior validator hardcoded `_find_module_subclasses` + a `forward()`
    check, returning zero matches for motion_planning's pure-Python classes
    and halting at stage 2.b. The fix drives the class-count + method
    checks off the manifest's `inherits:` (optional) and `required_methods:`
    declarations, supporting both supervised-ML and non-ML paradigms.
    """
    from scripts.validate_architecture_coder_output import validate

    run_dir = tmp_path / "run"
    method_dir = run_dir / "method"
    pipeline_dir = run_dir / ".pipeline"
    method_dir.mkdir(parents=True)

    (method_dir / "model.py").write_text("""
from __future__ import annotations
import torch
from torch import Tensor

class UnicycleDynamics:
    def __init__(self, v_max: float = 1.5, omega_max: float = 1.0):
        self.v_max = v_max
        self.omega_max = omega_max

    def step(self, state: Tensor, control: Tensor, dt: float) -> Tensor:
        return state

    def step_jacobian(self, state: Tensor, control: Tensor, dt: float) -> tuple[Tensor, Tensor]:
        return torch.eye(3), torch.zeros(3, 2)


class CircularCollisionModel:
    def __init__(self, robot_radius: float = 0.1):
        self.robot_radius = robot_radius

    def is_in_collision(self, state: Tensor, obstacles: list) -> bool:
        return False

    def distance_to_obstacle(self, state: Tensor, obstacles: list) -> float:
        return 1.0
""")
    (method_dir / "training.py").write_text("""
from __future__ import annotations
from typing import Any, List
from .model import UnicycleDynamics

def precompute_motion_primitives(
    dynamics: UnicycleDynamics,
    *,
    n_primitives: int = 100,
    seed: int = 0,
) -> List[Any]:
    return []
""")
    _write_arch_contract(
        pipeline_dir,
        paradigm_id="motion_planning",
        training_function="precompute_motion_primitives",
    )
    _write_minimal_paper_map(pipeline_dir)

    spec = _make_spec(paradigm_id="motion_planning")
    errors = validate(spec, run_dir, REPO_ROOT)
    assert errors == [], f"expected no errors, got: {errors}"


def test_validator_rejects_motion_planning_class_missing_required_method(tmp_path):
    """Companion to the pass case: if a motion_planning class is missing
    one of its manifest-declared required_methods (e.g., CollisionModel
    without `distance_to_obstacle`), the validator names the missing
    method and the unmatched expected class.

    Pins down that the new required-methods check actually enforces the
    constraint — not just that it's permissive."""
    from scripts.validate_architecture_coder_output import validate

    run_dir = tmp_path / "run"
    method_dir = run_dir / "method"
    pipeline_dir = run_dir / ".pipeline"
    method_dir.mkdir(parents=True)

    (method_dir / "model.py").write_text("""
from __future__ import annotations
import torch
from torch import Tensor

class UnicycleDynamics:
    def step(self, state: Tensor, control: Tensor, dt: float) -> Tensor:
        return state
    def step_jacobian(self, state: Tensor, control: Tensor, dt: float) -> tuple[Tensor, Tensor]:
        return torch.eye(3), torch.zeros(3, 2)


class BrokenCollisionModel:
    # Missing `distance_to_obstacle` — should trigger a no-match error
    # for the CollisionModel expected entry.
    def is_in_collision(self, state: Tensor, obstacles: list) -> bool:
        return False
""")
    (method_dir / "training.py").write_text("""
from __future__ import annotations
from typing import Any, List
from .model import UnicycleDynamics

def precompute_motion_primitives(
    dynamics: UnicycleDynamics,
    *,
    n_primitives: int = 100,
    seed: int = 0,
) -> List[Any]:
    return []
""")
    _write_arch_contract(
        pipeline_dir,
        paradigm_id="motion_planning",
        training_function="precompute_motion_primitives",
    )
    _write_minimal_paper_map(pipeline_dir)

    spec = _make_spec(paradigm_id="motion_planning")
    errors = validate(spec, run_dir, REPO_ROOT)
    assert any("distance_to_obstacle" in e for e in errors), (
        f"expected unmet-required-method error naming distance_to_obstacle; got: {errors}"
    )


# ---------------------------------------------------------------------------
# Overnight 2026-07-08 (iDb-RRT run 6 / detr-distill run 2): the build-plan
# merge broadcast the spec's required_model_methods onto EVERY class, so a
# correct two-class package failed 2.b on a physically meaningless
# requirement (step_jacobian demanded of the collision model). Spec methods
# now land in the manifest's `any_class_required_methods` bucket for
# multi-class paradigms; the gate requires each on at least one class.
# ---------------------------------------------------------------------------


_MP_TWO_CLASS_MODEL = """
from __future__ import annotations
import torch
from torch import Tensor

class UnicycleDynamics:
    def step(self, state: Tensor, control: Tensor, dt: float) -> Tensor:
        return state
    def step_jacobian(self, state: Tensor, control: Tensor, dt: float) -> tuple[Tensor, Tensor]:
        return torch.eye(3), torch.zeros(3, 2)


class CircularCollisionModel:
    def is_in_collision(self, state: Tensor, obstacles: list) -> bool:
        return False
    def distance_to_obstacle(self, state: Tensor, obstacles: list) -> float:
        return 1.0
"""

_MP_TRAINING = """
from __future__ import annotations
from typing import Any, List
from .model import UnicycleDynamics

def precompute_motion_primitives(
    dynamics: UnicycleDynamics,
    *,
    n_primitives: int = 100,
    seed: int = 0,
) -> List[Any]:
    return []
"""


def _make_mp_spec_with_required_methods(methods: list[str]) -> dict:
    spec = _make_spec(paradigm_id="motion_planning")
    spec["critical_requirements"]["required_model_methods"] = [
        {"name": s.split("(", 1)[0], "signature": s} for s in methods
    ]
    return spec


def test_validator_accepts_spec_method_owned_by_one_of_several_classes(tmp_path):
    """The iDb-RRT run 6 regression: the spec declares step_jacobian as a
    paper-derived required model method, the dynamics class implements it,
    and the collision class (correctly) does not. Must PASS."""
    from scripts.validate_architecture_coder_output import validate

    run_dir = tmp_path / "run"
    method_dir = run_dir / "method"
    pipeline_dir = run_dir / ".pipeline"
    method_dir.mkdir(parents=True)
    (method_dir / "model.py").write_text(_MP_TWO_CLASS_MODEL)
    (method_dir / "training.py").write_text(_MP_TRAINING)
    _write_arch_contract(
        pipeline_dir,
        paradigm_id="motion_planning",
        training_function="precompute_motion_primitives",
    )
    _write_minimal_paper_map(pipeline_dir)

    spec = _make_mp_spec_with_required_methods([
        "step_jacobian(self, state: Tensor, control: Tensor, dt: float) -> tuple[Tensor, Tensor]",
    ])
    errors = validate(spec, run_dir, REPO_ROOT)
    assert errors == [], f"expected no errors, got: {errors}"


def test_validator_rejects_spec_method_missing_from_all_classes(tmp_path):
    """Companion enforcement case (the detr-distill run 2 shape): a
    spec-derived method absent from every public class must fail with the
    at-least-one message, so the bucket is a real gate rather than a
    permissive no-op."""
    from scripts.validate_architecture_coder_output import validate

    run_dir = tmp_path / "run"
    method_dir = run_dir / "method"
    pipeline_dir = run_dir / ".pipeline"
    method_dir.mkdir(parents=True)
    (method_dir / "model.py").write_text(_MP_TWO_CLASS_MODEL)
    (method_dir / "training.py").write_text(_MP_TRAINING)
    _write_arch_contract(
        pipeline_dir,
        paradigm_id="motion_planning",
        training_function="precompute_motion_primitives",
    )
    _write_minimal_paper_map(pipeline_dir)

    spec = _make_mp_spec_with_required_methods([
        "forward_with_intermediates(self, images: Tensor) -> dict",
    ])
    errors = validate(spec, run_dir, REPO_ROOT)
    assert any(
        "forward_with_intermediates" in e and "at least one public" in e
        for e in errors
    ), f"expected at-least-one error naming forward_with_intermediates; got: {errors}"


# ---------------------------------------------------------------------------
# Failure cases — wrong class count, missing function
# ---------------------------------------------------------------------------


def test_validator_rejects_extra_public_class_in_single_class_paradigm(tmp_path):
    """For active_learning (declares 1 class), a model.py with TWO public
    nn.Module subclasses is rejected with a clear message naming both."""
    from scripts.validate_architecture_coder_output import validate

    run_dir = tmp_path / "run"
    method_dir = run_dir / "method"
    pipeline_dir = run_dir / ".pipeline"
    method_dir.mkdir(parents=True)

    (method_dir / "model.py").write_text("""
import torch.nn as nn

class ModelA(nn.Module):
    def forward(self, x):
        return x

class ModelB(nn.Module):
    def forward(self, x):
        return x
""")
    (method_dir / "training.py").write_text("""
import torch
from .model import ModelA

def build_model(input_dim, n_classes):
    return ModelA()

def train_from_scratch(x_train, y_train, *, seed=0):
    torch.manual_seed(seed)
    optimizer = torch.optim.Adam([])
    return ModelA()
""")
    _write_arch_contract(pipeline_dir, paradigm_id="active_learning")
    _write_minimal_paper_map(pipeline_dir)

    spec = _make_spec(paradigm_id="active_learning")
    errors = validate(spec, run_dir, REPO_ROOT)
    # 2026-05-26: error wording changed from "nn.Module subclass(es)" to
    # paradigm-agnostic "class(es)" after the validator was generalized to
    # support non-ML paradigms (motion_planning halt fix). The behavior
    # under test (reject when class count > manifest's declared count) is
    # preserved — only the message text shifted.
    assert any(
        "found 2 top-level public class(es)" in e for e in errors
    ), f"expected class-count error, got: {errors}"
    assert any(
        "paradigm manifest declares 1" in e for e in errors
    ), f"expected manifest-declared count in error message, got: {errors}"


def test_validator_rejects_extra_required_training_param(tmp_path):
    """Regression for the GBALD training-binding drift (queue 11b): the
    architecture-coder added a REQUIRED `batch_train_size` to
    train_from_scratch that the manifest signature never declared. A required
    extra is unbindable by the probe harness / notebook driver, so it must be
    rejected at the arch-coder gate (the training-function analog of the
    pluggable extra-required check, commit 382264d46). The existing
    single-class pass test covers the complement: an OPTIONAL batch arg (with
    a default) is allowed."""
    from scripts.validate_architecture_coder_output import validate

    run_dir = tmp_path / "run"
    method_dir = run_dir / "method"
    pipeline_dir = run_dir / ".pipeline"
    method_dir.mkdir(parents=True)

    (method_dir / "model.py").write_text("""
import torch.nn as nn

class MLPClassifier(nn.Module):
    def __init__(self, input_dim, n_classes):
        super().__init__()
        self.fc = nn.Linear(input_dim, n_classes)
    def forward(self, x):
        return self.fc(x)
""")
    (method_dir / "training.py").write_text("""
import torch
from .model import MLPClassifier

def build_model(input_dim, n_classes):
    return MLPClassifier(input_dim, n_classes)

def train_from_scratch(model, x_train, y_train, max_epochs, learning_rate,
                       batch_train_size, train_until_accuracy=None, seed=0):
    torch.manual_seed(seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    return model
""")
    _write_arch_contract(pipeline_dir, paradigm_id="active_learning")
    _write_minimal_paper_map(pipeline_dir)

    spec = _make_spec(paradigm_id="active_learning")
    errors = validate(spec, run_dir, REPO_ROOT)
    assert any(
        "extra required parameter" in e and "batch_train_size" in e
        for e in errors
    ), f"expected extra-required-training-param error, got: {errors}"


def test_validator_rejects_missing_required_function_in_kd(tmp_path):
    """KD declares train_with_distillation. If training.py only has the
    builders + a wrongly-named training function, validator rejects."""
    from scripts.validate_architecture_coder_output import validate

    run_dir = tmp_path / "run"
    method_dir = run_dir / "method"
    pipeline_dir = run_dir / ".pipeline"
    method_dir.mkdir(parents=True)

    (method_dir / "model.py").write_text("""
import torch.nn as nn

class StudentNet(nn.Module):
    def forward(self, x):
        return x

class TeacherNet(nn.Module):
    def forward(self, x):
        return x
""")
    # training.py has build_student + build_teacher but the training
    # function is misnamed `train_from_scratch` (legacy AL name) instead of
    # the KD manifest's `train_with_distillation`.
    (method_dir / "training.py").write_text("""
import torch
from .model import StudentNet, TeacherNet

def build_student(**kw):
    return StudentNet()

def build_teacher(**kw):
    return TeacherNet()

def train_from_scratch(student, teacher, x, y, *, seed=0):
    torch.manual_seed(seed)
    optimizer = torch.optim.Adam([])
    return student
""")
    _write_arch_contract(pipeline_dir, paradigm_id="knowledge_distillation")
    _write_minimal_paper_map(pipeline_dir)

    spec = _make_spec(paradigm_id="knowledge_distillation")
    errors = validate(spec, run_dir, REPO_ROOT)
    assert any(
        "missing top-level `train_with_distillation` function" in e for e in errors
    ), f"expected missing-function error, got: {errors}"


def test_validator_against_real_bev_distill_outputs():
    """Two live-run pins on the delivered bev-distill artifacts.

    (1) The 2026-05-19 manifest-MATCHING bug must not resurface: with the
    spec's required_model_methods stripped, the validator reports only the
    expected schema-v2 activation finding for this preserved legacy contract
    (the classes still satisfy the paradigm conventions and the greedy class
    matching still binds them).

    (2) The 2026-07-02 enforcement gap must stay closed: with the real
    spec (which declares get_bev_features / get_instance_features), the
    validator REJECTS these outputs — this exact package shipped with the
    spec interface unimplemented and was only caught by the stage-4
    fidelity review (F001/F002), a demote that should have been a 2.b
    retry."""
    from scripts.validate_architecture_coder_output import validate

    run_dir = REPO_ROOT / "r2c_runs" / "bev-distill"
    if not run_dir.is_dir():
        # Skip if the run has been wiped (it's not part of the test fixtures)
        import pytest
        pytest.skip("r2c_runs/bev-distill/ not present; skipping live regression")

    spec_path = run_dir / ".pipeline" / "method_spec.json"
    if not spec_path.is_file():
        import pytest
        pytest.skip("bev-distill method_spec.json missing")
    if not (run_dir / "method" / "model.py").is_file():
        # Live-run pin precondition: both assertions validate DELIVERED
        # architecture outputs. A fresh roll that halted upstream of 2.b
        # (2026-07-03: feasibility-gate halt) leaves no model.py to pin.
        import pytest
        pytest.skip("bev-distill run dir has no delivered model.py "
                    "(halted upstream); live pin needs delivered artifacts")

    spec = json.loads(spec_path.read_text(encoding="utf-8"))

    stripped = json.loads(json.dumps(spec))
    stripped.get("critical_requirements", {}).pop("required_model_methods", None)
    errors = validate(stripped, run_dir, REPO_ROOT)
    legacy_activation_errors = [
        error for error in errors
        if "uses legacy schema_version" in error
        and "new Stage 2.b architecture-coder output" in error
    ]
    substantive_errors = [
        error for error in errors if error not in legacy_activation_errors
    ]
    assert len(legacy_activation_errors) == 1, errors
    assert substantive_errors == [], (
        f"bev-distill regression: with spec methods stripped, the validator "
        f"must find no defect beyond the expected version activation in the "
        f"stage 2.b output that triggered the 2026-05-19 halt (manifest "
        f"matching regressed). Errors: {errors}"
    )

    errors = validate(spec, run_dir, REPO_ROOT)
    substantive_errors = [
        error for error in errors
        if not (
            "uses legacy schema_version" in error
            and "new Stage 2.b architecture-coder output" in error
        )
    ]
    if not substantive_errors:
        # The 2026-07-02 weekend batch re-rolled bev-distill with the 2.b
        # interface gate live: a fresh roll whose model.py implements the
        # spec's required methods replaced the historical enforcement-gap
        # artifacts (the old dir was wiped by the batch). The enforcement
        # direction stays covered by the synthetic tests above; the live pin
        # only applies while the historical package is on disk.
        required = [
            m.get("name", "")
            for m in spec.get("critical_requirements", {}).get(
                "required_model_methods", [])
            if isinstance(m, dict)
        ]
        if not required:
            # Third live shape (2026-07-04 re-roll): the fresh spec declares
            # NO required model methods, so the gate has nothing to enforce
            # and zero errors is the correct verdict — the historical
            # enforcement-gap artifact is gone either way.
            import pytest
            pytest.skip(
                "bev-distill was re-rolled with a spec declaring no "
                "required_model_methods; the enforcement pin has nothing "
                "to bite on")
        model_src = (run_dir / "method" / "model.py").read_text(
            encoding="utf-8")
        if all(name in model_src for name in required):
            import pytest
            pytest.skip(
                "bev-distill was re-rolled with the required interface "
                "implemented; the historical enforcement-gap artifact is gone")
    assert substantive_errors, (
        "bev-distill regression: the delivered model.py lacks the spec's "
        "required_model_methods and must now fail the 2.b gate"
    )
    assert any("get_bev_features" in e for e in substantive_errors), errors


def test_parse_or_error_rejects_compile_only_syntax_errors(tmp_path):
    """ast.parse accepts a misplaced `from __future__` import; only
    compile()/real import rejects it. The SRL 2026-07-05 matrix row died at
    stage 2d (no fix loop) because the 2b validator only parsed: the
    skeleton-first fill-in pass had re-added the `__future__` header after
    the docstring. The parse helper must catch the class at 2b, where the
    fix loop can route it."""
    from scripts.validate_architecture_coder_output import _parse_or_error

    bad = tmp_path / "training.py"
    bad.write_text(
        "from __future__ import annotations\n"
        '"""Docstring."""\n'
        "from __future__ import annotations\n"
        "import math\n"
    )
    tree, err = _parse_or_error(bad)
    assert tree is None
    assert err is not None and "fails to compile" in err
    assert "__future__" in err

    good = tmp_path / "ok.py"
    good.write_text(
        '"""Docstring."""\n'
        "from __future__ import annotations\n"
        "import math\n"
    )
    tree, err = _parse_or_error(good)
    assert err is None and tree is not None
