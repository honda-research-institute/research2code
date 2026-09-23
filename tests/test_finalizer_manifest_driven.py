"""Regression test: `scripts/finalize_package_init.py` must drive its
class-count + import lists off the paradigm's `package_manifest`, NOT off
a hardcoded single-class universal template.

The 2026-05-20 bev-distill halt at stage 2.d was the same bug class as
the stage 2.b validator bug fixed earlier: the finalizer hardcoded
"exactly one nn.Module subclass + build_model + train_from_scratch" and
rejected correct knowledge_distillation output with two classes + three
training functions. The fix mirrors the validator: read public_symbols
from the manifest and drive the imports/class-count check from there.

These tests cover both paradigm shapes (single-class AL + multi-class
KD) against the taxonomy build plans, plus an explicit failure case.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def _make_spec(
    paradigm_id: str,
    pluggable_name: str = "compute_distillation_loss",
) -> dict:
    return {
        "schema_version": "1.0.0",
        "paper": {"title": "T", "authors": ["A"], "year": 2026, "venue": "V"},
        "core_method": {"description": "x", "key_elements": []},
        "comparison": {
            "classification": {
                "id": paradigm_id,
            },
            "pluggable_component": {"name": pluggable_name},
        },
        "critical_requirements": {
            "model": {"specific_features": []},
            "data": {"specific_features": []},
        },
        "hyperparameters": [],
    }


def _seed_run_dir(tmp_path: Path, model_py: str, training_py: str,
                  method_py: str, data_py: str, spec: dict) -> Path:
    """Set up a minimal run dir with the four method/ files + spec."""
    run_dir = tmp_path / "run"
    method_dir = run_dir / "method"
    pipeline_dir = run_dir / ".pipeline"
    method_dir.mkdir(parents=True)
    pipeline_dir.mkdir(parents=True)
    (method_dir / "model.py").write_text(model_py)
    (method_dir / "training.py").write_text(training_py)
    (method_dir / "method.py").write_text(method_py)
    (method_dir / "data.py").write_text(data_py)
    spec_path = pipeline_dir / "method_spec.json"
    spec_path.write_text(json.dumps(spec, indent=2))
    return run_dir


def _run_finalizer(run_dir: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "scripts/finalize_package_init.py",
         "--spec", str(run_dir / ".pipeline" / "method_spec.json"),
         "--run-dir", str(run_dir)],
        cwd=str(REPO_ROOT), capture_output=True, text=True, timeout=30,
    )


# ---------------------------------------------------------------------------
# Single-class paradigm (active_learning)
# ---------------------------------------------------------------------------


def test_finalizer_handles_single_class_active_learning(tmp_path):
    """AL declares 1 class + build_model + train_from_scratch + load_data.
    Finalizer must produce a valid __init__.py for that shape."""
    spec = _make_spec(
        paradigm_id="active_learning",
        pluggable_name="select_batch",
    )
    run_dir = _seed_run_dir(
        tmp_path,
        model_py=(
            "import torch.nn as nn\n"
            "class MLP(nn.Module):\n"
            "    def forward(self, x):\n"
            "        return x\n"
        ),
        training_py=(
            "import torch\n"
            "from .model import MLP\n"
            "def build_model(input_dim, n_classes, **kw):\n"
            "    return MLP()\n"
            "def train_from_scratch(x_train, y_train, *, seed=0):\n"
            "    torch.manual_seed(seed)\n"
            "    return MLP()\n"
        ),
        method_py=(
            "import torch\n"
            "def select_batch(x_unlabeled, *, batch_size, seed):\n"
            "    return list(range(batch_size))\n"
        ),
        data_py=(
            "import torch\n"
            "def load_data(*, train_size=10, n_test=5, seed=0):\n"
            "    return torch.zeros(train_size, 3), torch.zeros(train_size),\\\n"
            "           torch.zeros(n_test, 3), torch.zeros(n_test)\n"
        ),
        spec=spec,
    )
    result = _run_finalizer(run_dir)
    assert result.returncode == 0, (
        f"finalizer failed for single-class AL.\n"
        f"stdout: {result.stdout}\nstderr: {result.stderr}"
    )
    init = (run_dir / "method" / "__init__.py").read_text()
    assert "from .model import MLP" in init
    assert "from .training import build_model, train_from_scratch" in init
    assert "from .data import load_data" in init
    assert "select_batch" in init


# ---------------------------------------------------------------------------
# Multi-class paradigm (knowledge_distillation) — the regression case
# ---------------------------------------------------------------------------


def test_finalizer_handles_multi_class_knowledge_distillation(tmp_path):
    """**bev-distill 2026-05-20 regression**: KD declares 2 classes
    (student + teacher) + 3 functions (build_student, build_teacher,
    train_with_distillation) + load_data. Finalizer must produce a valid
    __init__.py importing each."""
    spec = _make_spec(
        paradigm_id="knowledge_distillation",
        pluggable_name="compute_distillation_loss",
    )
    run_dir = _seed_run_dir(
        tmp_path,
        model_py=(
            "import torch.nn as nn\n"
            "class StudentNet(nn.Module):\n"
            "    def forward(self, x):\n"
            "        return x\n"
            "class TeacherNet(nn.Module):\n"
            "    def forward(self, x):\n"
            "        return x\n"
        ),
        training_py=(
            "import torch\n"
            "from .model import StudentNet, TeacherNet\n"
            "def build_student(**kw):\n"
            "    return StudentNet()\n"
            "def build_teacher(**kw):\n"
            "    return TeacherNet()\n"
            "def train_with_distillation(student, teacher, x, y, *,\n"
            "                             distillation_loss_fn, seed=0, **kw):\n"
            "    torch.manual_seed(seed)\n"
            "    return student\n"
        ),
        method_py=(
            "import torch\n"
            "def compute_distillation_loss(student, teacher, batch, seed):\n"
            "    return torch.zeros(1)\n"
        ),
        data_py=(
            "import torch\n"
            "def load_data(*, train_size=10, n_test=5, seed=0):\n"
            "    return torch.zeros(train_size), torch.zeros(train_size),\\\n"
            "           torch.zeros(n_test), torch.zeros(n_test)\n"
        ),
        spec=spec,
    )
    result = _run_finalizer(run_dir)
    assert result.returncode == 0, (
        f"finalizer failed for multi-class KD (the bev-distill 2026-05-20 regression).\n"
        f"stdout: {result.stdout}\nstderr: {result.stderr}"
    )
    init = (run_dir / "method" / "__init__.py").read_text()
    # Both classes imported from .model
    assert "from .model import StudentNet, TeacherNet" in init or (
        "StudentNet" in init and "TeacherNet" in init
    )
    # All three training functions
    assert "build_student" in init
    assert "build_teacher" in init
    assert "train_with_distillation" in init
    # Pluggable function
    assert "compute_distillation_loss" in init


# ---------------------------------------------------------------------------
# Failure case: wrong class count
# ---------------------------------------------------------------------------


def test_finalizer_rejects_extra_public_class_in_single_class_paradigm(tmp_path):
    """For active_learning (manifest declares 1 class), model.py with 2
    public nn.Module subclasses must be rejected with a clear error
    naming both the actual count and the manifest's expected count."""
    spec = _make_spec(
        paradigm_id="active_learning",
        pluggable_name="select_batch",
    )
    run_dir = _seed_run_dir(
        tmp_path,
        model_py=(
            "import torch.nn as nn\n"
            "class ModelA(nn.Module):\n"
            "    def forward(self, x):\n"
            "        return x\n"
            "class ModelB(nn.Module):\n"
            "    def forward(self, x):\n"
            "        return x\n"
        ),
        training_py=(
            "import torch\n"
            "from .model import ModelA\n"
            "def build_model(**kw):\n"
            "    return ModelA()\n"
            "def train_from_scratch(x, y, *, seed=0):\n"
            "    torch.manual_seed(seed)\n"
            "    return ModelA()\n"
        ),
        method_py=(
            "def select_batch(x_unlabeled, *, batch_size, seed):\n"
            "    return list(range(batch_size))\n"
        ),
        data_py=(
            "def load_data(*, train_size=10, n_test=5, seed=0):\n"
            "    return None, None, None, None\n"
        ),
        spec=spec,
    )
    result = _run_finalizer(run_dir)
    assert result.returncode != 0, (
        f"finalizer should have rejected 2-class AL, but returned 0.\n"
        f"stdout: {result.stdout}"
    )
    assert "2 public top-level nn.Module subclass" in result.stderr or (
        "ModelA" in result.stderr and "ModelB" in result.stderr
    ), f"expected count-mismatch error, got stderr: {result.stderr}"
    assert "manifest declares 1" in result.stderr, (
        f"expected manifest count in error, got stderr: {result.stderr}"
    )


# ---------------------------------------------------------------------------
# Live regression: re-run against the bev-distill outputs that triggered the halt
# ---------------------------------------------------------------------------


def test_finalizer_passes_against_real_bev_distill_outputs():
    """Re-run the finalizer against the bev-distill stage 2.b/2.c outputs
    that triggered the 2026-05-20 halt. With the manifest-driven fix, it
    must succeed."""
    run_dir = REPO_ROOT / "r2c_runs" / "bev-distill"
    if not run_dir.is_dir():
        import pytest
        pytest.skip("r2c_runs/bev-distill/ not present")
    spec_path = run_dir / ".pipeline" / "method_spec.json"
    if not spec_path.is_file():
        import pytest
        pytest.skip("bev-distill method_spec.json missing")
    if not (run_dir / "method").is_dir():
        # Live-run pin precondition: the finalizer needs delivered 2.b/2.c
        # outputs. A fresh roll that halted upstream (2026-07-03:
        # feasibility-gate halt) leaves no method/ package to finalize.
        import pytest
        pytest.skip("bev-distill run dir has no delivered method/ package "
                    "(halted upstream); live pin needs delivered artifacts")
    result = subprocess.run(
        [sys.executable, "scripts/finalize_package_init.py",
         "--spec", str(spec_path), "--run-dir", str(run_dir)],
        cwd=str(REPO_ROOT), capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, (
        f"bev-distill regression: finalizer still rejects the actual stage 2.b/2.c "
        f"outputs that triggered the 2026-05-20 halt.\n"
        f"stdout: {result.stdout}\nstderr: {result.stderr}"
    )
    # Sanity: both expected classes ended up in __init__.py
    init_text = (run_dir / "method" / "__init__.py").read_text()
    assert "from .model import" in init_text
    # Multi-class KD: two class names should appear in the .model import line
    model_line = [
        line for line in init_text.splitlines() if line.startswith("from .model import")
    ]
    assert model_line, f"no model import line found in: {init_text}"
    # The line should have BOTH student + teacher (comma-separated)
    assert "," in model_line[0], (
        f"expected multi-class model import, got: {model_line[0]!r}"
    )
