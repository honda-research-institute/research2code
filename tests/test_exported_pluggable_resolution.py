"""R2C-054 — the exported pluggable must resolve to the method-coder's module.

The fedavg 2026-08-04 defect (`exported_pluggable_resolves_to_wrong_module`):
on the generic provisional plan, training.py's placeholder symbols re-export
everything training.py defines. When training.py also defined a public
function with the pluggable's name, `_build_init_py`'s subtraction rule saw
the name as "claimed by training" and dropped the `.method` import entirely,
so the package-level name bound the training module's function (different
defaults, different return type) and `__all__` listed the name twice. The
stage 2d import validator passed it, because presence in `__all__` and
importability were its only assertions.

Two surfaces, tested together here:
  - generator: `finalize_package_init.py` must bind the pluggable to the
    method module no matter what other modules define, and dedupe `__all__`.
  - backstop: `validate_package_imports.py` must fail when the bound object's
    `__module__` is not `method.method`, and on duplicate `__all__` entries.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

from tests.test_provisional_templates import _gap_spec_dict, _install_run_pack


def _seed_method_files(run_dir: Path, *, training_defines_pluggable: bool) -> Path:
    """The fedavg shape on the generic plan: method.py owns the pluggable,
    training.py optionally defines a public function with the same name."""
    method = run_dir / "method"
    method.mkdir(parents=True, exist_ok=True)
    (method / "model.py").write_text(
        "class ValueNetwork:\n    pass\n\n"
        "class TargetNetwork:\n    pass\n",
        encoding="utf-8",
    )
    training_lines = [
        "def run_training_episodes(env, value_network, seed):",
        "    return value_network",
    ]
    if training_defines_pluggable:
        training_lines += [
            "def train_policy(env=None, value_network=None, seed=0):",
            "    return {'history': []}",
        ]
    (method / "training.py").write_text(
        "\n".join(training_lines) + "\n", encoding="utf-8")
    (method / "method.py").write_text(
        "def train_policy(env, value_network, seed, *, num_episodes=3000):\n"
        "    return None\n",
        encoding="utf-8",
    )
    (method / "data.py").write_text(
        "def load_data(path=None, *, seed=0):\n    return {}\n",
        encoding="utf-8",
    )
    return method


def _resolved_module(run_dir: Path, name: str) -> str:
    proc = subprocess.run(
        [sys.executable, "-c",
         f"import method; print(getattr(method, {name!r}).__module__)"],
        cwd=str(run_dir), capture_output=True, text=True, timeout=30,
    )
    assert proc.returncode == 0, proc.stderr
    return proc.stdout.strip()


# ---------------------------------------------------------------------------
# Generator repair
# ---------------------------------------------------------------------------


def test_finalizer_binds_pluggable_to_method_module_on_collision(tmp_path):
    """The fedavg reproducer: training.py also defines the pluggable's name.
    The finalizer must import the pluggable from .method, keep training's
    other functions, drop training's colliding version from the public
    surface, and write each __all__ entry once."""
    from scripts.finalize_package_init import finalize

    run_dir = tmp_path / "run"
    _install_run_pack(run_dir)
    spec_path = run_dir / ".pipeline" / "method_spec.json"
    spec_path.write_text(json.dumps(_gap_spec_dict()), encoding="utf-8")
    _seed_method_files(run_dir, training_defines_pluggable=True)

    rc = finalize(spec_path, run_dir)
    assert rc == 0, f"finalize exited {rc}"
    init_text = (run_dir / "method" / "__init__.py").read_text(encoding="utf-8")

    # training.py's non-colliding function is still re-exported; its
    # colliding train_policy is not on the .training import line.
    assert "run_training_episodes" in init_text
    for line in init_text.splitlines():
        if line.startswith("from .training import"):
            assert "train_policy" not in line, line

    # __all__ lists the pluggable exactly once.
    assert init_text.count('"train_policy"') == 1

    # The binding that actually matters: the public name resolves to the
    # method-coder's definition.
    assert _resolved_module(run_dir, "train_policy") == "method.method"


def test_finalizer_collision_free_output_unchanged(tmp_path):
    """Known-good guard: without a collision, the repair must not disturb
    the existing generic-plan output shape."""
    from scripts.finalize_package_init import finalize

    run_dir = tmp_path / "run"
    _install_run_pack(run_dir)
    spec_path = run_dir / ".pipeline" / "method_spec.json"
    spec_path.write_text(json.dumps(_gap_spec_dict()), encoding="utf-8")
    _seed_method_files(run_dir, training_defines_pluggable=False)

    rc = finalize(spec_path, run_dir)
    assert rc == 0, f"finalize exited {rc}"
    init_text = (run_dir / "method" / "__init__.py").read_text(encoding="utf-8")
    assert "from .training import run_training_episodes" in init_text
    assert init_text.count('"train_policy"') == 1
    assert _resolved_module(run_dir, "train_policy") == "method.method"


# ---------------------------------------------------------------------------
# Validator backstop
# ---------------------------------------------------------------------------


def _seed_for_validator(tmp_path: Path, init_text: str) -> tuple[dict, Path]:
    run_dir = tmp_path / "run"
    _seed_method_files(run_dir, training_defines_pluggable=True)
    (run_dir / "method" / "__init__.py").write_text(init_text, encoding="utf-8")
    (run_dir / "requirements.txt").write_text("", encoding="utf-8")
    spec = _gap_spec_dict()
    return spec, run_dir


def test_validator_flags_wrong_module_export(tmp_path):
    """The delivered fedavg shape: pluggable imported from .training, name
    listed twice. Both new assertions must fire, and the wrong-module error
    must name the module the export actually resolved to."""
    from scripts.validate_package_imports import validate

    spec, run_dir = _seed_for_validator(
        tmp_path,
        "from .data import load_data\n"
        "from .model import ValueNetwork, TargetNetwork\n"
        "from .training import run_training_episodes, train_policy\n"
        "\n"
        "__all__ = [\n"
        '    "train_policy",\n'
        '    "ValueNetwork",\n'
        '    "TargetNetwork",\n'
        '    "run_training_episodes",\n'
        '    "train_policy",\n'
        '    "load_data",\n'
        "]\n",
    )
    errors = validate(spec, run_dir)
    dup_errors = [e for e in errors if "duplicate" in e]
    wrong_module = [e for e in errors
                    if "exported_pluggable_resolves_to_wrong_module" in e]
    assert dup_errors, errors
    assert wrong_module, errors
    assert "method.training" in wrong_module[0]


def test_validator_passes_correct_resolution(tmp_path):
    """A correctly finalized package produces no findings."""
    from scripts.validate_package_imports import validate

    spec, run_dir = _seed_for_validator(
        tmp_path,
        "from .data import load_data\n"
        "from .method import train_policy\n"
        "from .model import ValueNetwork, TargetNetwork\n"
        "from .training import run_training_episodes\n"
        "\n"
        "__all__ = [\n"
        '    "train_policy",\n'
        '    "ValueNetwork",\n'
        '    "TargetNetwork",\n'
        '    "run_training_episodes",\n'
        '    "load_data",\n'
        "]\n",
    )
    assert validate(spec, run_dir) == []


def test_validator_accepts_naming_bridge_alias(tmp_path):
    """The naming bridge's private-definition remedy stays legal:
    `from .method import _x as x` binds an object whose __module__ is still
    method.method, so the identity check passes."""
    from scripts.validate_package_imports import validate

    spec, run_dir = _seed_for_validator(
        tmp_path,
        "from .data import load_data\n"
        "from .method import _train_policy as train_policy\n"
        "from .model import ValueNetwork, TargetNetwork\n"
        "from .training import run_training_episodes\n"
        "\n"
        "__all__ = [\n"
        '    "train_policy",\n'
        '    "ValueNetwork",\n'
        '    "TargetNetwork",\n'
        '    "run_training_episodes",\n'
        '    "load_data",\n'
        "]\n",
    )
    (run_dir / "method" / "method.py").write_text(
        "def _train_policy(env, value_network, seed, *, num_episodes=3000):\n"
        "    return None\n",
        encoding="utf-8",
    )
    assert validate(spec, run_dir) == []
