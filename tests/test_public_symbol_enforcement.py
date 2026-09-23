"""Item 31b — public-symbol enforcement at stage 2b.

Design: the public symbol enforcement design note (internal, not shipped).
Two deterministic checks, both fail-mode (maintainer decision 2026-07-14):

- self-resolution (sub-shape b, the ROMAN25 `__Trajectory`/`_Trajectory`
  class): every name the arch coder's own files reference must be bound
  somewhere in that file or imported;
- contract public surface (the deterministic slice of sub-shape a, the
  ICRA private-classifier genus): arch-contract architecture blocks must
  name public classes.

The zoo-style no-false-positive gate sweeps the committed example_runs/
packages (real known-good pipeline output).
"""

from __future__ import annotations

import pytest

import ast
import json
from pathlib import Path

from scripts.validate_architecture_coder_output import _undefined_name_references

ROOT = Path(__file__).resolve().parents[1]


def _scan(src: str):
    return _undefined_name_references(ast.parse(src))


def test_live_run_shape_dunder_class_referenced_by_single_underscore():
    src = (
        "class __Trajectory:\n"
        "    pass\n"
        "def build():\n"
        "    return _Trajectory()\n"
        "def extend(t):\n"
        "    assert isinstance(t, _Trajectory)\n"
        "    return _Trajectory(), _Trajectory()\n"
    )
    findings = _scan(src)
    assert len(findings) == 1
    name, count, first_line = findings[0]
    assert name == "_Trajectory"
    assert count == 4
    assert first_line == 4


def test_known_good_module_has_no_undefined_names():
    src = (
        "from __future__ import annotations\n"
        "import math\n"
        "from typing import TYPE_CHECKING\n"
        "if TYPE_CHECKING:\n"
        "    from collections import OrderedDict\n"
        "class Model:\n"
        "    def forward(self, x):\n"
        "        y = math.sqrt(x)\n"
        "        return [v * y for v in range(int(x))]\n"
        "def annotated(d: 'OrderedDict', **kwargs) -> Model:\n"
        "    try:\n"
        "        return Model()\n"
        "    except ValueError as exc:\n"
        "        raise RuntimeError(str(exc)) from exc\n"
    )
    assert _scan(src) == []


def test_star_import_files_are_skipped():
    src = (
        "from os.path import *\n"
        "def f():\n"
        "    return join('a', basename('b'))\n"
    )
    assert _scan(src) == []


def test_scoped_bindings_count_everywhere():
    """The binding set is deliberately scope-blind: a name bound in any
    scope suppresses the undefined finding (over-approximation keeps false
    positives out; dynamic corners pass silently)."""
    src = (
        "def f():\n"
        "    helper = 1\n"
        "    return helper\n"
        "def g():\n"
        "    return helper\n"  # cross-scope: unresolved at runtime, but suppressed by design
    )
    assert _scan(src) == []


def test_2b_validator_fails_on_undefined_reference_and_private_contract(tmp_path):
    """Integration through validate(): both 31b checks fire with
    architecture-coder attribution, using the generic provisional plan
    fixture (flexible class count keeps the count check out of the way)."""
    from scripts.validate_architecture_coder_output import validate
    from tests.test_provisional_templates import (
        _gap_spec_dict,
        _install_run_pack,
        _typed_gap_arch_contract,
    )

    run_dir = tmp_path / "run"
    _install_run_pack(run_dir)
    method = run_dir / "method"
    method.mkdir(parents=True)
    (method / "model.py").write_text(
        "class __Trajectory:\n"
        "    pass\n"
        "class ValueNetwork:\n"
        "    def evaluate(self, observation):\n"
        "        return _Trajectory()\n",
        encoding="utf-8",
    )
    (method / "training.py").write_text(
        "def run_training_episodes(env, value_network, seed):\n"
        "    return value_network\n",
        encoding="utf-8",
    )
    contract = _typed_gap_arch_contract(class_name="_ValueNetwork")
    (run_dir / ".pipeline").mkdir(parents=True, exist_ok=True)
    (run_dir / ".pipeline" / "arch_contract.json").write_text(
        json.dumps(contract), encoding="utf-8")

    errors = validate(_gap_spec_dict(), run_dir, ROOT)
    assert any("`_Trajectory`" in e and "NameError" in e for e in errors), errors
    assert any("architecture.value_network" in e and "`_ValueNetwork`" in e
               for e in errors), errors


def test_example_runs_sweep_no_false_positives():
    """Zoo-style gate: the committed example_runs/ packages are real
    known-good pipeline output — the self-resolution scan must stay silent
    on every one of them."""
    if not (ROOT / "example_runs").is_dir():
        pytest.skip("example_runs/ (finished packages) not present")
    swept = 0
    failures = []
    for py in sorted((ROOT / "example_runs").rglob("method/*.py")):
        try:
            tree = ast.parse(py.read_text(encoding="utf-8"))
        except SyntaxError:
            continue
        findings = _undefined_name_references(tree)
        swept += 1
        if findings:
            failures.append((str(py.relative_to(ROOT)), findings))
    assert swept > 0, "sweep found no example_runs packages — gate is broken"
    assert not failures, f"false positives on known-good code: {failures}"
