"""Paradigm-neutrality regression tests for the deterministic pipeline gates.

These pin the Batch-1/2 fixes from the 2026-05-26 paradigm-pipeline-readiness
audit (docs/paradigm-pipeline-readiness-audit.md). The shared failure shape:
gates that hardcode supervised-ML / active-learning assumptions (nn.Module,
forward(), build_*, AL param fields) and halt on non-ML paradigms
(motion_planning). Each test exercises both a supervised-ML shape (regression
guard) and the motion_planning shape (the fix).
"""

from __future__ import annotations

import ast
import json
import shutil
import tempfile
from pathlib import Path

import pytest

pytestmark = pytest.mark.probe_runtime

REPO_ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# PA-D2 — finalize_package_init: paradigm-agnostic class discovery
# ---------------------------------------------------------------------------


def test_finalizer_discovers_non_nn_module_classes():
    """The finalizer's class discovery must find plain-Python classes (no
    nn.Module base), not just nn.Module subclasses. motion_planning's
    UnicycleDynamics / CircularCollisionModel inherit from nothing."""
    from scripts.finalize_package_init import _find_public_top_level_class_names

    src = (
        "class UnicycleDynamics:\n"
        "    def step(self, s, c, dt): return s\n"
        "class CircularCollisionModel:\n"
        "    def is_in_collision(self, s): return False\n"
        "class _Helper:\n"  # private — excluded
        "    pass\n"
    )
    names = _find_public_top_level_class_names(ast.parse(src))
    assert names == ["UnicycleDynamics", "CircularCollisionModel"]


def test_finalizer_still_finds_nn_module_classes():
    """Regression: supervised-ML nn.Module subclasses are still discovered."""
    from scripts.finalize_package_init import _find_public_top_level_class_names

    src = (
        "import torch.nn as nn\n"
        "class MLPClassifier(nn.Module):\n"
        "    def forward(self, x): return x\n"
    )
    assert _find_public_top_level_class_names(ast.parse(src)) == ["MLPClassifier"]


# ---------------------------------------------------------------------------
# PA-D10 — ArchContract schema allows scalar (bool/float/int) output_type
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("output_type", ["bool", "float", "int", "scalar", "tensor", "dict", "tuple"])
def test_arch_contract_output_type_accepts_scalars(output_type):
    """A method that returns a plain scalar (collision check -> bool, distance
    -> float) must be expressible in the contract."""
    from schemas.arch_contract import ForwardSignature

    kwargs = {"input": {"x": "(state_dim,)"}, "output_type": output_type}
    if output_type == "tensor":
        kwargs["output_shape"] = "(state_dim,)"
    elif output_type == "dict":
        kwargs["output_keys"] = {"k": "(1,)"}
    elif output_type == "tuple":
        kwargs["output_shapes"] = ["(1,)"]
    ForwardSignature.model_validate(kwargs)  # must not raise


# ---------------------------------------------------------------------------
# PA-D1 — validate_arch_contract_runtime: paradigm-agnostic dry-run
# ---------------------------------------------------------------------------


def _write_runtime_pkg(tmp: Path, *, model_src, init_src, training_src, data_src, contract):
    run = tmp / "run"
    method = run / "method"
    pipe = run / ".pipeline"
    method.mkdir(parents=True)
    pipe.mkdir(parents=True)
    (method / "model.py").write_text(model_src)
    (method / "training.py").write_text(training_src)
    (method / "data.py").write_text(data_src)
    (method / "__init__.py").write_text(init_src)
    (pipe / "arch_contract.json").write_text(json.dumps(contract))
    (pipe / "method_spec.json").write_text("{}")
    return run


_MP_CONTRACT = {
    "schema_version": "1.0.0",
    "paradigm_id": "motion_planning",
    "data_loader": {"load_data_returns": {"start": "(state_dim,)", "environment": "(opaque)"}},
    "architecture": {
        "dynamics": {
            "class_name": "UnicycleDynamics",
            "forward": {
                "input": {"x": "(state_dim,)", "control": "(control_dim,)", "dt": "scalar"},
                "output_type": "tensor",
                "output_shape": "(state_dim,)",
            },
            "additional_methods": {
                "step_jacobian": {
                    "input": {"x": "(state_dim,)", "control": "(control_dim,)", "dt": "scalar"},
                    "output_type": "tuple",
                    "output_shapes": ["(state_dim, state_dim)", "(state_dim, control_dim)"],
                }
            },
        },
        "collision_model": {
            "class_name": "CircularCollisionModel",
            "forward": {"input": {"x": "(state_dim,)"}, "output_type": "bool"},
            "additional_methods": {
                "distance_to_obstacle": {
                    "input": {"x": "(state_dim,)", "obstacles": "list[(3,)]"},
                    "output_type": "float",
                }
            },
        },
    },
    "pluggable_component": {"name": "plan", "input_shapes": {"seed": "int"}, "output_shape": "PlanResult"},
    "training_loop": {"function_name": "precompute_motion_primitives", "input_shapes": {"seed": "int"}},
}

_MP_TRAINING = (
    "from typing import Any, List\n"
    "from .model import UnicycleDynamics\n"
    "def precompute_motion_primitives(dynamics, *, n_primitives=100, seed=0):\n"
    "    return []\n"
)
_MP_DATA = (
    "def load_environment(name='x', *, seed=0):\n    return None\n"
    "def load_problem(name='x', *, seed=0):\n    return None\n"
)
_MP_INIT = (
    "from .model import UnicycleDynamics, CircularCollisionModel\n"
    "from .training import precompute_motion_primitives\n"
    "from .data import load_environment, load_problem\n"
    "def plan(*a, **k):\n    return None\n"
)
_MP_MODEL_OK = '''
import torch
from torch import Tensor
class UnicycleDynamics:
    def step(self, state, control, dt):
        return state
    def step_jacobian(self, state, control, dt):
        return torch.eye(8), torch.zeros(8, 8)
class CircularCollisionModel:
    def is_in_collision(self, state):
        return False
    def distance_to_obstacle(self, state, obstacles):
        return 1.0
'''


def test_arch_contract_requirements_accept_named_forward_inputs(tmp_path):
    """A-006 regression: the arch_contract *completeness* schema must accept a
    motion_planning forward.input with NAMED keys (state/control/dt), not require
    a literal `x` key. Before the fix, the (since-retired) arch_contract_schema
    required `forward.input.x` (the supervised-ML nn.Module.forward(x) assumption)
    and rejected correct motion_planning contracts at stage 2.d (pdwa halt)."""
    from scripts.validate_arch_contract import validate
    pipe = tmp_path / ".pipeline"
    pipe.mkdir()
    contract = {
        "schema_version": "1.0.0",
        "paradigm_id": "motion_planning",
        "data_loader": {"load_data_returns": {"start": "(state_dim,)", "goal": "(state_dim,)",
                                              "environment": "(opaque)", "dynamics": "(opaque)"}},
        "architecture": {
            "dynamics": {"class_name": "UnicycleDynamics",
                         "forward": {"input": {"state": "(state_dim,)", "control": "(control_dim,)", "dt": "float"},
                                     "output_type": "tensor", "output_shape": "(state_dim,)"}},
            "collision_model": {"class_name": "CircularCollisionModel",
                                "forward": {"input": {"state": "(state_dim,)", "obstacles": "list[(2,)]"},
                                            "output_type": "bool"}},
        },
        "pluggable_component": {"name": "plan", "input_shapes": {"start": "(state_dim,)", "seed": "int"},
                                "output_shape": "PlanResult"},
    }
    (pipe / "arch_contract.json").write_text(json.dumps(contract))
    spec = {"comparison": {"classification": {
        "id": "motion_planning"}}}
    errs = validate(spec, tmp_path)
    assert not any("forward.input" in e for e in errs), \
        f"named forward inputs must satisfy the schema (A-006 regression): {errs}"
    assert errs == [], f"expected clean validation, got: {errs}"


def test_runtime_validator_passes_motion_planning(tmp_path):
    """A correct motion_planning package (plain-Python classes, no builders,
    no forward()) passes the runtime dry-run."""
    from scripts.validate_arch_contract_runtime import validate

    run = _write_runtime_pkg(
        tmp_path, model_src=_MP_MODEL_OK, init_src=_MP_INIT,
        training_src=_MP_TRAINING, data_src=_MP_DATA, contract=_MP_CONTRACT,
    )
    assert validate({}, run) == []


def test_runtime_validator_catches_broken_motion_planning_method(tmp_path):
    """The dry-run still CATCHES a real error: an additional_method that
    raises is surfaced (not silently skipped)."""
    from scripts.validate_arch_contract_runtime import validate

    broken = _MP_MODEL_OK.replace(
        "    def step_jacobian(self, state, control, dt):\n        return torch.eye(8), torch.zeros(8, 8)",
        "    def step_jacobian(self, state, control, dt):\n        raise ValueError('deliberate bug')",
    )
    run = _write_runtime_pkg(
        tmp_path, model_src=broken, init_src=_MP_INIT,
        training_src=_MP_TRAINING, data_src=_MP_DATA, contract=_MP_CONTRACT,
    )
    errs = validate({}, run)
    assert any("step_jacobian" in e and "deliberate bug" in e for e in errs), errs


def test_runtime_validator_catches_broken_supervised_ml_forward(tmp_path):
    """Regression: the builder + forward(x) path still catches a forward that
    raises for supervised-ML paradigms."""
    from scripts.validate_arch_contract_runtime import validate

    al_contract = {
        "schema_version": "1.0.0",
        "paradigm_id": "active_learning",
        "data_loader": {"load_data_returns": {"x_pool": "(N_pool, n_features)"}},
        "architecture": {
            "model": {
                "class_name": "Net",
                "forward": {"input": {"x": "(B, n_features)"}, "output_type": "tensor", "output_shape": "(B, n_classes)"},
            }
        },
        "pluggable_component": {"name": "select_batch", "input_shapes": {"seed": "int"}, "output_shape": "list[int]"},
        "training_loop": {"function_name": "train_from_scratch", "input_shapes": {"seed": "int"}},
    }
    model = (
        "import torch, torch.nn as nn\n"
        "class Net(nn.Module):\n"
        "    def __init__(self, input_dim=8, n_classes=3):\n"
        "        super().__init__()\n"
        "        self.fc = nn.Linear(input_dim, n_classes)\n"
        "    def forward(self, x):\n"
        "        raise RuntimeError('deliberate forward bug')\n"
    )
    training = (
        "from .model import Net\n"
        "def build_model(input_dim, n_classes):\n    return Net(input_dim, n_classes)\n"
        "def train_from_scratch(*a, seed=0, **k):\n    return None\n"
    )
    data = "def load_data(*a, **k):\n    return ()\n"
    init = (
        "from .model import Net\n"
        "from .training import build_model, train_from_scratch\n"
        "def select_batch(*a, **k):\n    return []\n"
        "from .data import load_data\n"
    )
    run = _write_runtime_pkg(
        tmp_path, model_src=model, init_src=init,
        training_src=training, data_src=data, contract=al_contract,
    )
    errs = validate({}, run)
    assert any("forward" in e and "deliberate forward bug" in e for e in errs), errs


def _al_runtime_contract(*, data_x_shape: str, model_x_shape: str) -> dict:
    return {
        "schema_version": "1.0.0",
        "paradigm_id": "active_learning",
        "data_loader": {
            "load_data_returns": {
                "x_train": data_x_shape.replace("N_pool", "N"),
                "y_train": "(N,)",
                "x_pool": data_x_shape,
                "y_pool": "(N_pool,)",
                "x_test": data_x_shape.replace("N_pool", "N_test"),
                "y_test": "(N_test,)",
            },
        },
        "architecture": {
            "model": {
                "class_name": "Net",
                "forward": {
                    "input": {"x": model_x_shape},
                    "output_type": "tensor",
                    "output_shape": "(B, n_classes)",
                },
            },
        },
        "pluggable_component": {
            "name": "select_batch",
            "input_shapes": {"x_unlabeled": data_x_shape},
            "output_shape": "list[int] of length batch_size",
        },
        "training_loop": {
            "function_name": "train_from_scratch",
            "input_shapes": {
                "x_train": data_x_shape.replace("N_pool", "N"),
                "y_train": "(N,)",
            },
        },
    }


_AL_RUNTIME_MODEL = (
    "import torch, torch.nn as nn\n"
    "class Net(nn.Module):\n"
    "    def __init__(self, input_dim=8, n_classes=3, **kwargs):\n"
    "        super().__init__()\n"
    "        self.n_classes = n_classes\n"
    "    def forward(self, x):\n"
    "        return torch.zeros(x.shape[0], self.n_classes)\n"
)

_AL_RUNTIME_TRAINING = (
    "from .model import Net\n"
    "def build_model(input_dim=8, n_classes=3, **kwargs):\n"
    "    return Net(input_dim=input_dim, n_classes=n_classes)\n"
    "def train_from_scratch(model, x_train=None, y_train=None, **kwargs):\n"
    "    return model\n"
)

_AL_RUNTIME_INIT = (
    "from .model import Net\n"
    "from .training import build_model, train_from_scratch\n"
    "from .data import load_data\n"
    "def select_batch(model, x_unlabeled, batch_size, seed=0, **kwargs):\n"
    "    return list(range(min(batch_size, len(x_unlabeled))))\n"
)

_AL_RUNTIME_FLAT_DATA = (
    "import torch\n"
    "def load_data(pool_size=4, train_size=4, n_test=4, seed=0):\n"
    "    return (\n"
    "        torch.zeros(train_size, 8), torch.zeros(train_size, dtype=torch.long),\n"
    "        torch.zeros(pool_size, 8), torch.zeros(pool_size, dtype=torch.long),\n"
    "        torch.zeros(n_test, 8), torch.zeros(n_test, dtype=torch.long),\n"
    "    )\n"
)


def test_runtime_validator_accepts_matching_active_learning_data_model_shape(tmp_path):
    """Known-good: flat AL data and a flat model input remain valid."""
    from scripts.validate_arch_contract_runtime import validate

    run = _write_runtime_pkg(
        tmp_path,
        model_src=_AL_RUNTIME_MODEL,
        init_src=_AL_RUNTIME_INIT,
        training_src=_AL_RUNTIME_TRAINING,
        data_src=_AL_RUNTIME_FLAT_DATA,
        contract=_al_runtime_contract(
            data_x_shape="(N_pool, n_features)",
            model_x_shape="(B, input_dim)",
        ),
    )
    assert validate({}, run) == []


def test_runtime_validator_catches_data_model_sample_shape_mismatch(tmp_path):
    """Known-bad: each half is internally valid, but flat loader outputs cannot
    satisfy an image-shaped model input contract."""
    from scripts.validate_arch_contract_runtime import validate

    run = _write_runtime_pkg(
        tmp_path,
        model_src=_AL_RUNTIME_MODEL,
        init_src=_AL_RUNTIME_INIT,
        training_src=_AL_RUNTIME_TRAINING,
        data_src=_AL_RUNTIME_FLAT_DATA,
        contract=_al_runtime_contract(
            data_x_shape="(N_pool, n_features)",
            model_x_shape="(B, 3, H, W)",
        ),
    )
    errs = validate({}, run)
    assert any(
        "architecture.model.forward.input.x" in e
        and "data_loader.load_data_returns.x_train" in e
        and "sample ranks differ" in e
        for e in errs
    ), errs


# ---------------------------------------------------------------------------
# PA-D3 / PA-D4 — derive_params: paradigm-aware param builders
# ---------------------------------------------------------------------------


def _mp_spec() -> dict:
    return {
        "comparison": {
            "classification": {"id": "motion_planning"},
            "pluggable_component": {
                "signature": "plan(start, goal, environment, dynamics, seed, mc_samples: int = 100, risk_weights: dict = None) -> PlanResult",
            },
        },
        "critical_requirements": {
            "model": {"architecture": "unicycle"},
            "scale_dependent_hyperparameters": [
                {"name": "d_safe", "paper_value": 0.375, "formula": "V_max / 4",
                 "assumes_data_scale": "meters", "description": "Safety distance.",
                 "paper_section": "Section 3.3"},
                {"name": "clearance_obs", "paper_value": None, "formula": "||P_obs - P_robot||",
                 "assumes_data_scale": "meters", "description": "Runtime distance.",
                 "paper_section": "Section 3.3"},
            ],
        },
    }


def test_derive_params_motion_planning_no_al_params():
    """motion_planning must NOT get fabricated AL params (num_rounds,
    pool_size, initial_labeled, train_until_accuracy, max_epochs) or the
    nn.Module-specific hidden_dim."""
    from scripts.derive_params import derive

    out = derive(_mp_spec())
    keys = set(out["params"].keys())
    forbidden = {"num_rounds", "pool_size", "initial_labeled",
                 "train_until_accuracy", "max_epochs", "hidden_dim", "learning_rate"}
    assert not (keys & forbidden), f"AL/model params leaked into motion_planning: {keys & forbidden}"


def test_derive_params_motion_planning_surfaces_scale_dep_hyperparam():
    """d_safe (paper_value 0.375) becomes a param with paper provenance;
    clearance_obs (null paper_value) does NOT (it's runtime-computed)."""
    from scripts.derive_params import derive

    params = derive(_mp_spec())["params"]
    assert "d_safe" in params
    assert params["d_safe"]["value"] == 0.375
    assert params["d_safe"]["source"] == "paper"
    assert "clearance_obs" not in params


def test_derive_params_motion_planning_keeps_signature_extras():
    """The plan() signature's keyword extras (mc_samples, risk_weights) are
    still added via _add_paradigm_extras."""
    from scripts.derive_params import derive

    params = derive(_mp_spec())["params"]
    assert "mc_samples" in params
    assert "risk_weights" in params


def test_derive_params_active_learning_unchanged():
    """Regression: an AL spec still gets its data/training/model params."""
    from scripts.derive_params import derive

    al_spec = {
        "comparison": {
            "classification": {"id": "active_learning/batch_acquisition"},
            "pluggable_component": {"signature": "select_batch(model, x_unlabeled, batch_size, seed) -> List[int]"},
        },
        "critical_requirements": {
            "model": {"architecture": "MLP 256"},
            "data_setup": {"initial_labeled": 100, "batch_size": 100, "total_budget": 1000,
                           "num_rounds": 10, "paper_section": "S4"},
            "training": {"optimizer": "Adam", "learning_rate": "0.001", "protocol": "scratch",
                         "mc_samples": 0, "paper_section": "S4"},
        },
    }
    params = derive(al_spec)["params"]
    # AL params present
    assert "num_rounds" in params
    assert "hidden_dim" in params


def test_derive_params_uses_contract_paper_truth_for_mc_samples_conflict():
    """Stage 2.x can recover an older completed spec whose training.mc_samples
    contains a demo default while the methodology contract records paper truth."""
    from scripts.derive_params import derive

    al_spec = {
        "comparison": {
            "classification": {"id": "active_learning/bayesian"},
            "pluggable_component": {
                "signature": (
                    "select_batch(model, x_unlabeled, x_labeled, batch_size, "
                    "seed, mc_samples: int = 100) -> List[int]"
                ),
            },
        },
        "critical_requirements": {
            "model": {"architecture": "MLP 256"},
            "data_setup": {
                "initial_labeled": 20,
                "batch_size": 100,
                "total_budget": 600,
                "num_rounds": 6,
                "paper_section": "S7",
            },
            "training": {
                "optimizer": "Adam",
                "learning_rate": "0.001",
                "protocol": "scratch",
                "mc_samples": 100,
                "paper_section": "S7",
            },
        },
        "methodology_replication_contract": {
            "schema_version": "1.0",
            "elements": [
                {
                    "element_id": "mc_sample_count",
                    "role": "supporting_mechanism",
                    "replication_status": "faithful_approximation_allowed",
                    "paper_section": "S7",
                    "paper_evidence": "Paper uses 2000 MC dropout samples.",
                    "technical_concept": "MC dropout sample count",
                    "required_behavior": "Use MC dropout samples for BALD scoring.",
                    "demo_scale_implementation": "Use fewer samples for demo scale.",
                    "acceptable_approximations": [
                        "Reducing mc_samples from 2000 to 100 for demo runs."
                    ],
                    "forbidden_substitutions": [],
                    "required_controls": [],
                    "fairness_checks": [],
                    "feasibility_rationale": "Count reduction changes noise, not algorithm.",
                    "verification_expectations": [],
                    "blockers": [],
                },
            ],
        },
    }

    params = derive(al_spec)["params"]
    assert params["mc_samples"]["value"] == 20
    assert params["mc_samples"]["source"] == "system_default"
    assert params["mc_samples"]["paper_value"] == 2000
    assert "Paper uses 2000 MC dropout samples" in params["mc_samples"]["reasoning"]


def test_derive_params_uses_scale_dependent_hyperparam_for_signature_extra():
    """AL method-specific scale params should not fall back to spec_default
    when Stage 1 already extracted a paper-stated scale-dependent value."""
    from scripts.derive_params import derive

    al_spec = {
        "comparison": {
            "classification": {"id": "active_learning/batch_acquisition"},
            "pluggable_component": {
                "signature": (
                    "select_batch(model, x_unlabeled, batch_size, seed, "
                    "R_0: float = 2000.0) -> List[int]"
                ),
            },
        },
        "critical_requirements": {
            "model": {"architecture": "MLP 256"},
            "data_setup": {
                "initial_labeled": 20,
                "batch_size": 100,
                "total_budget": 600,
                "num_rounds": 6,
                "paper_section": "S7",
            },
            "training": {
                "optimizer": "Adam",
                "learning_rate": "0.001",
                "protocol": "scratch",
                "mc_samples": None,
                "paper_section": "S7",
            },
            "scale_dependent_hyperparameters": [
                {
                    "name": "R_0",
                    "paper_value": 2000.0,
                    "formula": None,
                    "assumes_data_scale": "raw_pixel_unnormalized",
                    "description": "Distance threshold for the geometric probability model.",
                    "paper_section": "Equation 10",
                }
            ],
        },
    }

    params = derive(al_spec)["params"]
    assert params["R_0"]["source"] == "paper"
    assert params["R_0"]["value"] == 2000.0
    assert params["R_0"]["paper_section"] == "Equation 10"


def test_derive_params_uses_contract_value_for_unstructured_signature_extra():
    """Generic signature extras can still recover paper truth from the
    methodology contract when no dedicated structured field exists yet."""
    from scripts.derive_params import derive

    al_spec = {
        "comparison": {
            "classification": {"id": "active_learning/batch_acquisition"},
            "pluggable_component": {
                "signature": (
                    "select_batch(model, x_unlabeled, batch_size, seed, "
                    "eta: float = 0.9) -> List[int]"
                ),
            },
        },
        "critical_requirements": {
            "model": {"architecture": "MLP 256"},
            "data_setup": {
                "initial_labeled": 20,
                "batch_size": 100,
                "total_budget": 600,
                "num_rounds": 6,
                "paper_section": "S7",
            },
            "training": {
                "optimizer": "Adam",
                "learning_rate": "0.001",
                "protocol": "scratch",
                "mc_samples": None,
                "paper_section": "S7",
            },
        },
        "methodology_replication_contract": {
            "elements": [
                {
                    "paper_evidence": "The paper sets eta=0.9 for the geometric probability update.",
                    "acceptable_approximations": [
                        "Use eta=0.9 at demo scale; this parameter is not rescaled."
                    ],
                }
            ]
        },
    }

    params = derive(al_spec)["params"]
    assert params["eta"]["source"] == "paper"
    assert params["eta"]["value"] == 0.9
    assert "methodology_replication_contract" in params["eta"]["note"]


def test_derive_params_uses_paper_map_value_for_unstructured_signature_extra():
    """Generic signature extras can recover explicit paper truth from paper_map
    when Stage 1 did not promote the value into a structured spec field."""
    from scripts.derive_params import derive

    al_spec = {
        "comparison": {
            "classification": {"id": "active_learning/batch_acquisition"},
            "pluggable_component": {
                "signature": (
                    "select_batch(model, x_unlabeled, batch_size, seed, "
                    "eta: float = 0.9) -> List[int]"
                ),
            },
        },
        "critical_requirements": {
            "model": {"architecture": "MLP 256"},
            "data_setup": {
                "initial_labeled": 20,
                "batch_size": 100,
                "total_budget": 600,
                "num_rounds": 6,
                "paper_section": "S7",
            },
            "training": {
                "optimizer": "Adam",
                "learning_rate": "0.001",
                "protocol": "scratch",
                "mc_samples": None,
                "paper_section": "S7",
            },
        },
    }
    paper_map = {
        "elements": [
            {
                "id": "hyp-defaults",
                "source_text": (
                    "The parameter settings are R_0 = 2.0e + 3 and eta = 0.9."
                ),
            }
        ]
    }

    params = derive(al_spec, paper_map=paper_map)["params"]
    assert params["eta"]["source"] == "paper"
    assert params["eta"]["value"] == 0.9
    assert "paper_map" in params["eta"]["note"]


def test_derive_params_marks_signature_default_as_system_default_when_paper_value_differs():
    """If paper text states a different value from the signature default, keep
    the runtime default but preserve paper_value instead of claiming paper provenance."""
    from scripts.derive_params import derive

    al_spec = {
        "comparison": {
            "classification": {"id": "active_learning/batch_acquisition"},
            "pluggable_component": {
                "signature": (
                    "select_batch(model, x_unlabeled, batch_size, seed, "
                    "eta: float = 0.5) -> List[int]"
                ),
            },
        },
        "critical_requirements": {
            "model": {"architecture": "MLP 256"},
            "data_setup": {
                "initial_labeled": 20,
                "batch_size": 100,
                "total_budget": 600,
                "num_rounds": 6,
                "paper_section": "S7",
            },
            "training": {
                "optimizer": "Adam",
                "learning_rate": "0.001",
                "protocol": "scratch",
                "mc_samples": None,
                "paper_section": "S7",
            },
        },
    }
    paper_map = {
        "elements": [
            {"source_text": "Ellipsoid geodesic is adjusted by eta = 0.9."}
        ]
    }

    params = derive(al_spec, paper_map=paper_map)["params"]
    assert params["eta"]["source"] == "system_default"
    assert params["eta"]["value"] == 0.5
    assert params["eta"]["paper_value"] == 0.9
    assert "signature default" in params["eta"]["reasoning"]


# ---------------------------------------------------------------------------
# PA-FG1 — sub-paradigm stage_review_focus checks are merged (not dropped)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("paradigm_id,expected_check", [
    ("motion_planning/sampling_based", "MP-sampling-goal-bias"),
    ("motion_planning/optimization_based", "MP-opt-cost-function"),
    ("motion_planning/rl_collision_avoidance", "MP-RLCA-method-training-loop-real"),
])
def test_motion_planning_subparadigm_review_checks_merge(paradigm_id, expected_check):
    """The sub-paradigm's stage_2c_method semantic checks must appear in the
    taxonomy-rendered review focus."""
    from scripts import taxonomy

    merged = taxonomy.merged_stage_review_focus(None, paradigm_id, "stage_2c_method")
    check_ids = [c.get("id") for c in merged.get("semantic_checks", [])]
    assert expected_check in check_ids, (
        f"{expected_check} missing from merged stage_2c_method checks; got {check_ids}"
    )
    # Parent checks must still be present (merge, not replace).
    assert len(check_ids) > 1


# ---------------------------------------------------------------------------
# PA-FG3 — must_equal: pluggable_component.name resolves against the SPEC
# (variant-aware), not the field-guide default.
# ---------------------------------------------------------------------------


def _mp_arch_contract(pluggable_name: str) -> dict:
    """A motion_planning arch contract that satisfies the paradigm's
    arch-contract requirements, with a settable pluggable name."""
    return {
        "schema_version": "1.0.0",
        "paradigm_id": "motion_planning",
        "data_loader": {"load_data_returns": {"start": "(state_dim,)", "environment": "(opaque)"}},
        "architecture": {
            "dynamics": {
                "class_name": "UnicycleDynamics",
                "forward": {"input": {"x": "(state_dim,)", "control": "(control_dim,)", "dt": "scalar"},
                            "output_type": "tensor", "output_shape": "(state_dim,)"},
            },
            "collision_model": {
                "class_name": "CircularCollisionModel",
                "forward": {"input": {"x": "(state_dim,)"}, "output_type": "bool"},
            },
        },
        "pluggable_component": {
            "name": pluggable_name,
            "input_shapes": {"start": "(state_dim,)", "seed": "int"},
            "output_shape": "PlanResult",
        },
        "training_loop": {"function_name": "precompute_motion_primitives", "input_shapes": {"seed": "int"}},
    }


def _mp_spec_for_contract(pluggable_name: str) -> dict:
    return {
        "comparison": {
            "classification": {
                "id": "motion_planning",
            },
            "pluggable_component": {"name": pluggable_name},
        },
    }


def _run_arch_contract_validate(tmp: Path, contract: dict, spec: dict):
    from scripts.validate_arch_contract import validate
    pipe = tmp / ".pipeline"
    pipe.mkdir(parents=True)
    (pipe / "arch_contract.json").write_text(json.dumps(contract))
    return validate(spec, tmp)


def _al_spec_for_contract() -> dict:
    return {
        "comparison": {
            "classification": {"id": "active_learning"},
            "pluggable_component": {"name": "select_batch"},
        },
    }


def _al_arch_contract(load_data_returns: dict) -> dict:
    return {
        "schema_version": "1.0.0",
        "paradigm_id": "active_learning",
        "data_loader": {"load_data_returns": load_data_returns},
        "architecture": {
            "model": {
                "class_name": "Net",
                "forward": {
                    "input": {"x": "(B, n_features)"},
                    "output_type": "tensor",
                    "output_shape": "(B, n_classes)",
                },
            },
        },
        "pluggable_component": {
            "name": "select_batch",
            "input_shapes": {"x_unlabeled": "(N_pool, n_features)"},
            "output_shape": "list[int] of length batch_size",
        },
        "training_loop": {
            "function_name": "train_from_scratch",
            "input_shapes": {"x_train": "(N, n_features)", "y_train": "(N,)"},
        },
    }


def test_active_learning_load_data_contract_accepts_scaffold_shape(tmp_path):
    """loader_contract_mismatch known-good: AL load_data returns the four
    tensors the scaffold and notebook consume; class count is derived from
    labels rather than returned as a fifth value."""
    errs = _run_arch_contract_validate(
        tmp_path,
        _al_arch_contract({
            "x_pool": "(N_pool, n_features)",
            "y_pool": "(N_pool,)",
            "x_test": "(N_test, n_features)",
            "y_test": "(N_test,)",
        }),
        _al_spec_for_contract(),
    )
    assert not [e for e in errs if "load_data_returns" in e]


def test_active_learning_load_data_contract_rejects_extra_n_classes(tmp_path):
    """loader_contract_mismatch known-bad: a generated arch contract must not
    claim `load_data()` returns `n_classes` when the shared AL loader returns
    exactly pool/test tensors."""
    errs = _run_arch_contract_validate(
        tmp_path,
        _al_arch_contract({
            "x_pool": "(N_pool, n_features)",
            "y_pool": "(N_pool,)",
            "x_test": "(N_test, n_features)",
            "y_test": "(N_test,)",
            "n_classes": "int",
        }),
        _al_spec_for_contract(),
    )
    assert any("load_data_returns" in e and "n_classes" in e for e in errs), errs


def test_arch_contract_pluggable_name_matches_spec_variant(tmp_path):
    """An MPC optimization/motion_planning paper picks `solve_step`, not the
    paradigm default `plan`. The must_equal check resolves against the spec's
    pluggable name, so a `solve_step` contract + `solve_step` spec passes
    (PA-FG3)."""
    errs = _run_arch_contract_validate(
        tmp_path, _mp_arch_contract("solve_step"), _mp_spec_for_contract("solve_step")
    )
    name_errs = [e for e in errs if "pluggable_component.name" in e]
    assert name_errs == [], f"unexpected pluggable-name error: {name_errs}"


def test_arch_contract_pluggable_name_default_still_works(tmp_path):
    """Regression: the common case (contract + spec both `plan`, matching the
    field-guide default) still passes."""
    errs = _run_arch_contract_validate(
        tmp_path, _mp_arch_contract("plan"), _mp_spec_for_contract("plan")
    )
    name_errs = [e for e in errs if "pluggable_component.name" in e]
    assert name_errs == [], f"unexpected pluggable-name error: {name_errs}"


def test_arch_contract_pluggable_name_disagreement_caught(tmp_path):
    """If the contract's pluggable name disagrees with the spec's, that IS an
    error (the contract must name the paper's actual pluggable)."""
    errs = _run_arch_contract_validate(
        tmp_path, _mp_arch_contract("solve_step"), _mp_spec_for_contract("plan")
    )
    assert any("pluggable_component.name" in e for e in errs), (
        f"expected a pluggable-name mismatch error; got {errs}"
    )


# ---------------------------------------------------------------------------
# PA-D9 — requirements floor is paradigm-neutral (no unconditional torchvision)
# ---------------------------------------------------------------------------


def test_requirements_floor_has_no_unconditional_torchvision():
    """torchvision is image-paradigm-specific; it must not be in the universal
    floor (motion_planning would carry a dead dep)."""
    from scripts.finalize_package_init import DEFAULT_BASE_REQUIREMENTS

    floor_names = {name for name, _ in DEFAULT_BASE_REQUIREMENTS}
    assert "torchvision" not in floor_names
    # The genuinely-universal deps remain.
    assert {"numpy", "torch", "matplotlib"} <= floor_names


def test_requirements_adds_torchvision_when_imported():
    """Regression: image paradigms (AL/KD) whose data.py imports torchvision
    still get it via the import-detection sweep."""
    from scripts.finalize_package_init import _build_requirements_txt

    txt = _build_requirements_txt({"torchvision"}, "active_learning")
    assert "torchvision" in txt


# ---------------------------------------------------------------------------
# Pure-numpy paradigm support (the researcher's ROMAN25 run, 2026-07-14): the contract
# can say output_type="ndarray", and the runtime dry-run synthesizes numpy
# inputs for numpy-shaped signatures instead of torch tensors.
# ---------------------------------------------------------------------------


def test_arch_contract_output_type_accepts_ndarray():
    from schemas.arch_contract import ForwardSignature

    ForwardSignature.model_validate({
        "input": {"x": "(state_dim,)"},
        "output_type": "ndarray",
        "output_shape": "(state_dim,)",
    })  # must not raise


_NP_CONTRACT = {
    "schema_version": "1.0.0",
    "paradigm_id": "motion_planning",
    "data_loader": {"load_data_returns": {}},
    "architecture": {
        "dynamics": {
            "class_name": "PointDynamics",
            "forward": {
                "input": {"x": "(state_dim,)", "u": "(control_dim,)"},
                "output_type": "ndarray",
                "output_shape": "(state_dim,)",
            },
            # A bool-returning method in the same pure-numpy contract:
            # inherits the contract default kind (numpy), because np
            # code rejects torch tensors at runtime.
            "additional_methods": {
                "in_bounds": {
                    "input": {"x": "(state_dim,)"},
                    "output_type": "bool",
                },
            },
        },
    },
    "pluggable_component": {"name": "plan", "input_shapes": {"seed": "int"},
                            "output_shape": "PlanResult"},
    "training_loop": {"function_name": "precompute",
                      "input_shapes": {"seed": "int"}},
}

_NP_MODEL = '''
import numpy as np
class PointDynamics:
    def forward(self, x, u):
        if not isinstance(x, np.ndarray) or not isinstance(u, np.ndarray):
            raise TypeError("pure-numpy method got a non-numpy input: %s"
                            % type(x).__name__)
        return np.asarray(x)
    def in_bounds(self, x):
        # np.ndarray.all() exists; a torch tensor here would still pass
        # this line, so assert the family explicitly like real np code
        # that calls np-only APIs.
        if not isinstance(x, np.ndarray):
            raise TypeError("expected ndarray, got %s" % type(x).__name__)
        return bool((np.abs(x) < 100).all())
'''

_NP_TRAINING = "def precompute(*a, **k):\n    return []\n"
_NP_DATA = "def load_problem(name='x', *, seed=0):\n    return None\n"
_NP_INIT = (
    "from .model import PointDynamics\n"
    "def plan(*a, **k):\n    return None\n"
)


def test_runtime_validator_synthesizes_numpy_for_ndarray_contract(tmp_path):
    """A pure-numpy package whose contract declares ndarray returns gets
    numpy test inputs (torch tensors would raise inside the method and
    reject a correct package, the ROMAN25 stage-2d failure)."""
    from scripts.validate_arch_contract_runtime import validate

    run = _write_runtime_pkg(
        tmp_path, model_src=_NP_MODEL, init_src=_NP_INIT,
        training_src=_NP_TRAINING, data_src=_NP_DATA, contract=_NP_CONTRACT,
    )
    assert validate({}, run) == []


def test_runtime_validator_flags_ndarray_contract_returning_tensor(tmp_path):
    """The ndarray output check has teeth: a method that claims ndarray but
    returns something else fails the dry-run."""
    from scripts.validate_arch_contract_runtime import validate

    bad_model = _NP_MODEL.replace("return np.asarray(x)", "return list(x)")
    run = _write_runtime_pkg(
        tmp_path, model_src=bad_model, init_src=_NP_INIT,
        training_src=_NP_TRAINING, data_src=_NP_DATA, contract=_NP_CONTRACT,
    )
    errs = validate({}, run)
    assert any("output_type=ndarray but got" in e for e in errs), errs


# ---------------------------------------------------------------------------
# Network-free 2d dry-run (the Rethinking-Grouping 2026-07-13 stall class):
# the dry-run subprocess always runs with R2C_OFFLINE=1, template-derived
# downloaders refuse loudly under it, and the runner treats the refusal like
# a missing optional dependency (synthetic-fixture retry, else clean skip).
# ---------------------------------------------------------------------------

_OFF_MODEL = (
    "import torch\n"
    "class Net(torch.nn.Module):\n"
    "    def forward(self, x):\n"
    "        return x\n"
)
_OFF_INIT = (
    "from .model import Net\n"
    "def select_batch(*a, **k):\n    return None\n"
)
_OFF_TRAINING = "def train_from_scratch(*a, **k):\n    return None\n"
_OFF_CONTRACT = {
    "schema_version": "1.0.0",
    "paradigm_id": "active_learning",
    "data_loader": {"load_data_returns": {"x_pool": "(N_pool, n_features)",
                                          "y_pool": "(N_pool,)"}},
    "architecture": {
        "model": {
            "class_name": "Net",
            "forward": {
                "input": {"x": "(B, n_features)"},
                "output_type": "tensor",
                "output_shape": "(B, n_features)",
            },
        },
    },
    "pluggable_component": {"name": "select_batch",
                            "input_shapes": {"seed": "int"},
                            "output_shape": "SelectionResult"},
    "training_loop": {"function_name": "train_from_scratch",
                      "input_shapes": {"seed": "int"}},
}

# Mirrors the template downloaders' behavior: honors R2C_OFFLINE with a loud
# refusal; if the env var were MISSING in the dry-run subprocess this loader
# raises an AssertionError instead, which fails the validation — so the
# passing test also proves the subprocess env injection.
_OFF_DATA_WITH_PATH = (
    "import json, os\n"
    "from pathlib import Path\n"
    "import torch\n"
    "def load_data(path=None, *, pool_size=4, n_test=4, seed=0):\n"
    "    if path is None:\n"
    "        if os.environ.get('R2C_OFFLINE'):\n"
    "            raise RuntimeError('R2C_OFFLINE is set: refusing to download')\n"
    "        raise AssertionError('network download attempted: R2C_OFFLINE "
    "not set in dry-run subprocess')\n"
    "    d = json.loads((Path(path) / 'data.json').read_text())\n"
    "    return torch.tensor(d['x_pool']), torch.tensor(d['y_pool'])\n"
)

_OFF_DATA_NO_PATH = (
    "import os\n"
    "def load_data(*, pool_size=4, n_test=4, seed=0):\n"
    "    if os.environ.get('R2C_OFFLINE'):\n"
    "        raise RuntimeError('R2C_OFFLINE is set: refusing to download')\n"
    "    raise AssertionError('network download attempted')\n"
)


def test_dry_run_offline_refusal_retries_with_synthetic_fixture(tmp_path):
    """A loader that refuses to download under R2C_OFFLINE gets the same
    synthetic-fixture retry as a missing optional dependency, and the
    return-shape dry-run still runs (no error, no network)."""
    from scripts.validate_arch_contract_runtime import validate

    run = _write_runtime_pkg(
        tmp_path, model_src=_OFF_MODEL, init_src=_OFF_INIT,
        training_src=_OFF_TRAINING, data_src=_OFF_DATA_WITH_PATH, contract=_OFF_CONTRACT,
    )
    assert validate({}, run) == []


def test_dry_run_offline_refusal_without_path_skips_cleanly(tmp_path):
    """No `path` param means no synthetic fixture is possible; the offline
    refusal must be a clean skip (environment gap), never an error."""
    from scripts.validate_arch_contract_runtime import validate

    run = _write_runtime_pkg(
        tmp_path, model_src=_OFF_MODEL, init_src=_OFF_INIT,
        training_src=_OFF_TRAINING, data_src=_OFF_DATA_NO_PATH, contract=_OFF_CONTRACT,
    )
    assert validate({}, run) == []


def test_dry_run_non_offline_exception_is_still_an_error(tmp_path):
    """The refusal carve-out must not swallow genuine loader crashes."""
    from scripts.validate_arch_contract_runtime import validate

    bad_data = (
        "def load_data(path=None, *, pool_size=4, n_test=4, seed=0):\n"
        "    raise ValueError('genuine loader bug')\n"
    )
    run = _write_runtime_pkg(
        tmp_path, model_src=_OFF_MODEL, init_src=_OFF_INIT,
        training_src=_OFF_TRAINING, data_src=bad_data, contract=_OFF_CONTRACT,
    )
    errs = validate({}, run)
    assert any("genuine loader bug" in e for e in errs), errs


def test_dry_run_subprocess_timeout_is_bounded_and_named(tmp_path, monkeypatch):
    """A hang inside package code is cut off by the validator's own timeout
    (below the driver's 180s wall) with a precise message, instead of eating
    the whole run_script window (the 2026-07-13 crash shape)."""
    from scripts.validate_arch_contract_runtime import validate

    monkeypatch.setenv("R2C_DRY_RUN_TIMEOUT_S", "3")
    hang_data = (
        "import time\n"
        "time.sleep(60)\n"
        "def load_data(path=None):\n"
        "    return ()\n"
    )
    run = _write_runtime_pkg(
        tmp_path, model_src=_OFF_MODEL, init_src=_OFF_INIT,
        training_src=_OFF_TRAINING, data_src=hang_data, contract=_OFF_CONTRACT,
    )
    errs = validate({}, run)
    assert len(errs) == 1 and "timed out after 3s" in errs[0], errs
    assert "launched with R2C_OFFLINE=1" in errs[0]
    assert "may be a dataset download" in errs[0]
    assert "not a dataset download" not in errs[0]
    assert "genuine hang" not in errs[0]
