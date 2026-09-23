"""Pin the dependency-gated test surface (decision 3, corrected shape).

A module-level `pytest.importorskip(...)` hides EVERY test in its file
behind ONE reported skip, so a pinned skip count moves by zero when a
hundred gated tests are added (proven empirically: the eight gated files
report 9 skips, not 153). This pin counts the thing that matters — which
files are gated and how many tests each hides — and runs identically with
and without torch, locally and in CI. Adding or removing a gate shows up
in the same diff that reviews it: update the literal dict here.
"""

from __future__ import annotations

import ast
from pathlib import Path

TESTS_DIR = Path(__file__).resolve().parent

# {file name: number of test functions hidden behind its module-level gate}
GATED_TESTS = {
    "test_al_loop_probes.py": 21,
    "test_al_stage1_probes.py": 10,
    "test_budget_sufficiency.py": 6,
    "test_claims_probes.py": 61,
    "test_kd_probes.py": 18,
    "test_selector_crash_arm.py": 16,
    "test_term_ablation_probes.py": 16,
    "test_trainability_probe.py": 5,
}


def _module_level_importorskip(tree: ast.Module) -> bool:
    for node in tree.body:
        if not isinstance(node, (ast.Expr, ast.Assign, ast.AnnAssign)):
            continue
        for sub in ast.walk(node):
            if (isinstance(sub, ast.Call)
                    and isinstance(sub.func, ast.Attribute)
                    and sub.func.attr == "importorskip"):
                return True
    return False


def _test_count(tree: ast.Module) -> int:
    n = sum(1 for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name.startswith("test"))
    n += sum(1 for node in tree.body if isinstance(node, ast.ClassDef)
             for sub in node.body
             if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef))
             and sub.name.startswith("test"))
    return n


def test_module_level_dependency_gates_match_the_pinned_map():
    found: dict[str, int] = {}
    for path in sorted(TESTS_DIR.glob("test_*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        if _module_level_importorskip(tree):
            found[path.name] = _test_count(tree)
    assert found == GATED_TESTS, (
        "module-level importorskip surface changed; review the gate and "
        f"update GATED_TESTS. Found: {found}"
    )
