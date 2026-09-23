"""Per-element generated tests — the deterministic core (R2C-024).

The approved design (per-element-tests-design-review.md, Section 3 note 3)
ships one generated test module per implemented paper element under
`method/tests/`, behind deterministic guards. This module is everything
that never needs a model call:

- **Eligibility plan** (`eligible_elements`): the anchor join table bounds
  the testable surface honestly — an element is eligible when its
  `code_role` is `implement` and its primary anchor site maps to a
  function. Concepts and background with no code mapping never appear.
- **Vacuity floor** (`validate_test_module` / `validate_tests_dir`): a
  generated test must name exactly one eligible element id, call the
  mapped function, carry at least one value-level assertion (shape-only
  never counts as verifying a claim), and — when the element's pseudocode
  states usable numbers — pin at least one constant traceable to it.
- **Runner** (`run_test_module`): executes one test module against the
  run's own package in a subprocess, without writing bytecode or pytest
  caches into the run tree.
- **Advisory mutation check** (`mutation_check`, approved decision 1
  option C): one bounded mutant per tested function — perturb the first
  numeric constant, else flip the first comparison operator — applied to
  a COPY of the package, never in place. A test that catches its mutant
  is `tested`; a survivor still ships but its coverage row says plainly
  that it could not demonstrate it distinguishes correct from broken
  code. Nothing gates on this in the first cut.
- **Coverage README** (`render_coverage_readme`): every eligible element
  gets a row — tested, weak, no mutant applicable, or "generated but
  could not be verified, not shipped" — so survivorship is visible
  instead of silent. The framing states the researcher's two levels: per-component
  tests verify each piece against the paper's own statement, the notebook
  demo is the integration test.

Standalone use over an existing `method/tests/` directory:

    python3 scripts/element_tests.py --run-dir r2c_runs/<slug>

Exit 0 all shipped tests pass their floor and run green, 1 otherwise,
2 setup error. The pipeline's generation step (the agent side) builds on
these functions; nothing here dispatches a model.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from trusted import trusted_path
from build_anchor_join_table import (
    ARTIFACT_REL_PATH,
    JoinTableError,
    JoinTableSetupError,
    build_join_table,
)

TESTS_REL_DIR = Path("method") / "tests"
README_NAME = "README.md"
TEST_TIMEOUT_S = 120

_ANCHOR_RE = re.compile(r"#\s*paper-element:\s*([A-Za-z][A-Za-z0-9_+\-]*)")
# Numbers a pseudocode line can pin a test constant against.
_NUMBER_RE = re.compile(r"(?<![\w.])-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")
# Approximate-equality callables that make an assert value-level even
# without a bare comparison.
_APPROX_CALLS = frozenset({
    "isclose", "allclose", "approx", "assert_allclose",
    "assert_almost_equal", "assert_array_almost_equal",
})
# Operands that make a comparison structural rather than value-level.
_SHAPE_ATTRS = frozenset({"shape", "ndim", "dtype"})
_SHAPE_CALLS = frozenset({"len", "type"})

_FLIPPED_OPS = {
    ast.Lt: ast.GtE, ast.GtE: ast.Lt,
    ast.Gt: ast.LtE, ast.LtE: ast.Gt,
    ast.Eq: ast.NotEq, ast.NotEq: ast.Eq,
}


# ---------------------------------------------------------------------------
# Eligibility plan
# ---------------------------------------------------------------------------

def _load_join_elements(run_dir: Path) -> dict:
    """Artifact-first join-table elements (same precedence the METHOD.md
    generator uses), else an in-process build."""
    artifact = run_dir / ARTIFACT_REL_PATH
    if artifact.is_file():
        try:
            table = json.loads(artifact.read_text(encoding="utf-8"))
            return table.get("elements", {})
        except (json.JSONDecodeError, OSError):
            pass
    return build_join_table(run_dir).get("elements", {})


def eligible_elements(run_dir: Path) -> dict[str, dict]:
    """The honestly-bounded testable surface: paper-map elements whose
    `code_role` is `implement` and whose primary anchor site maps to a
    function. Returns `{element_id: plan_entry}`; raises the join table's
    own errors when the run cannot be scanned."""
    paper_map_path = run_dir / ".pipeline" / "paper_map.json"
    if not paper_map_path.is_file():
        raise JoinTableSetupError(f"no paper_map.json under {run_dir}")
    paper_map = json.loads(paper_map_path.read_text(encoding="utf-8"))
    by_id = {e["id"]: e for e in paper_map.get("elements", [])}

    plan: dict[str, dict] = {}
    for element_id, joined in sorted(_load_join_elements(run_dir).items()):
        element = by_id.get(element_id)
        if element is None or element.get("code_role") != "implement":
            continue
        primary = joined.get("primary") or {}
        if not primary.get("qualname") or not primary.get("function_line"):
            continue  # module-level anchor: no single function to exercise
        plan[element_id] = {
            "id": element_id,
            "name": element.get("name", ""),
            "type": element.get("type", ""),
            "section": element.get("section", ""),
            "source_text": element.get("source_text", ""),
            "pseudocode": element.get("pseudocode", ""),
            "dependencies": element.get("dependencies", []),
            "file": primary["file"],
            "qualname": primary["qualname"],
            "function_line": primary["function_line"],
        }
    return plan


# ---------------------------------------------------------------------------
# Vacuity floor
# ---------------------------------------------------------------------------

def _pseudocode_numbers(entry: dict) -> set[float]:
    """Numbers the element's own computable content states. Pseudocode
    only — prose and section references would pin tests to citation
    numerals. A set of only 0/1 is too weak to demand traceability."""
    numbers = {float(m) for m in _NUMBER_RE.findall(entry.get("pseudocode", ""))}
    return numbers if numbers - {0.0, 1.0, -1.0} else set()


def _test_numeric_literals(tree: ast.Module) -> set[float]:
    out: set[float] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) \
                and not isinstance(node.value, bool):
            out.add(float(node.value))
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub) \
                and isinstance(node.operand, ast.Constant) \
                and isinstance(node.operand.value, (int, float)):
            out.add(float(-node.operand.value))
    return out


def _is_shape_operand(node: ast.expr) -> bool:
    if isinstance(node, ast.Attribute) and node.attr in _SHAPE_ATTRS:
        return True
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) \
            and node.func.id in _SHAPE_CALLS:
        return True
    return False


def _assert_is_value_level(node: ast.Assert) -> bool:
    """A value-level assertion compares data values or uses an
    approximate-equality call. A comparison whose operands include
    `.shape`/`.ndim`/`.dtype`/`len()`/`type()` is structural and does
    not count as verifying a claim."""
    for sub in ast.walk(node.test):
        if isinstance(sub, ast.Call):
            name = sub.func.attr if isinstance(sub.func, ast.Attribute) \
                else sub.func.id if isinstance(sub.func, ast.Name) else ""
            if name in _APPROX_CALLS:
                return True
        if isinstance(sub, ast.Compare):
            operands = [sub.left, *sub.comparators]
            if not any(_is_shape_operand(op) for op in operands):
                return True
    return False


def _called_names(tree: ast.Module) -> set[str]:
    out: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                out.add(node.func.id)
            elif isinstance(node.func, ast.Attribute):
                out.add(node.func.attr)
    return out


def validate_test_module(path: Path, plan: dict[str, dict]) -> tuple[str | None, list[str]]:
    """The vacuity floor for one generated test module. Returns
    `(element_id or None, problems)` — an empty problem list means the
    module clears the floor."""
    problems: list[str] = []
    try:
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(path))
    except (OSError, SyntaxError) as e:
        return None, [f"unreadable or unparseable test module: {e}"]

    anchors = _ANCHOR_RE.findall(source)
    if len(set(anchors)) != 1:
        problems.append(
            "a test module must carry exactly one `# paper-element: <id>` "
            f"anchor naming the element it verifies (found {sorted(set(anchors))})")
        return None, problems
    element_id = anchors[0]
    entry = plan.get(element_id)
    if entry is None:
        return element_id, [
            f"anchor {element_id!r} is not an eligible implemented element"]

    function_name = entry["qualname"].split(".")[-1]
    if function_name not in _called_names(tree):
        problems.append(
            f"the test never calls {function_name!r}, the function that "
            f"implements {element_id}")

    asserts = [n for n in ast.walk(tree) if isinstance(n, ast.Assert)]
    if not asserts:
        problems.append("no assertions at all")
    elif not any(_assert_is_value_level(a) for a in asserts):
        problems.append(
            "no value-level assertion — shape/len/dtype checks alone do "
            "not verify a claim")

    stated = _pseudocode_numbers(entry)
    if stated and not (stated & _test_numeric_literals(tree)):
        problems.append(
            "the element's pseudocode states concrete numbers "
            f"({sorted(stated)[:6]}) but no test constant traces to any "
            "of them")
    return element_id, problems


def validate_tests_dir(run_dir: Path, plan: dict[str, dict]) -> dict[str, dict]:
    """Floor results for every `test_*.py` under `method/tests/`, keyed by
    module name: `{module: {"element_id": ..., "problems": [...]}}`."""
    tests_dir = run_dir / TESTS_REL_DIR
    out: dict[str, dict] = {}
    if not tests_dir.is_dir():
        return out
    for module in sorted(tests_dir.glob("test_*.py")):
        element_id, problems = validate_test_module(module, plan)
        out[module.name] = {"element_id": element_id, "problems": problems}
    return out


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_test_module(run_dir: Path, module: Path,
                    timeout_s: int = TEST_TIMEOUT_S) -> dict:
    """Run one test module against the run's own package. Nothing is
    written into the run tree (no bytecode, no pytest cache)."""
    import os  # noqa: PLC0415
    env = dict(os.environ)
    safe_run_dir = trusted_path(run_dir.resolve())
    env["PYTHONPATH"] = str(safe_run_dir)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    try:
        proc = subprocess.run(
            [sys.executable, "-m", "pytest", str(trusted_path(module)), "-q",
             "--no-header", "-p", "no:cacheprovider"],
            cwd=safe_run_dir, env=env, capture_output=True, text=True,
            timeout=timeout_s)
    except subprocess.TimeoutExpired:
        return {"status": "timeout",
                "detail": f"exceeded {timeout_s}s at fixture scale"}
    status = "passed" if proc.returncode == 0 else "failed"
    tail = (proc.stdout + proc.stderr)[-2000:]
    return {"status": status, "detail": tail}


# ---------------------------------------------------------------------------
# Advisory mutation check
# ---------------------------------------------------------------------------

class _FirstMutant(ast.NodeTransformer):
    """Apply exactly one bounded mutation inside one function body:
    perturb the first numeric constant, else flip the first comparison
    operator. Docstrings are never touched."""

    def __init__(self) -> None:
        self.description: str | None = None
        self._docstrings: set[int] = set()

    def mark_docstrings(self, fn: ast.AST) -> None:
        for node in ast.walk(fn):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef,
                                 ast.ClassDef)):
                if (node.body and isinstance(node.body[0], ast.Expr)
                        and isinstance(node.body[0].value, ast.Constant)
                        and isinstance(node.body[0].value.value, str)):
                    self._docstrings.add(id(node.body[0].value))

    def visit_Constant(self, node: ast.Constant):  # noqa: N802
        if (self.description is None and id(node) not in self._docstrings
                and isinstance(node.value, (int, float))
                and not isinstance(node.value, bool)):
            mutated = node.value + 1
            self.description = (
                f"numeric constant {node.value!r} -> {mutated!r} "
                f"(line {node.lineno})")
            return ast.copy_location(ast.Constant(value=mutated), node)
        return node

    def visit_Compare(self, node: ast.Compare):  # noqa: N802
        self.generic_visit(node)
        if self.description is None and node.ops:
            flipped = _FLIPPED_OPS.get(type(node.ops[0]))
            if flipped is not None:
                self.description = (
                    f"comparison {type(node.ops[0]).__name__} -> "
                    f"{flipped.__name__} (line {node.lineno})")
                node.ops = [flipped(), *node.ops[1:]]
        return node


def _locate_function(tree: ast.Module, qualname: str) -> ast.AST | None:
    node: ast.AST = tree
    for part in qualname.split("."):
        found = None
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef,
                                  ast.ClassDef)) and child.name == part:
                found = child
                break
        if found is None:
            return None
        node = found
    return node


def mutate_function_source(source: str, qualname: str) -> tuple[str, str] | None:
    """One bounded mutant of the named function, or None when neither
    mutation class applies. Returns `(mutated_module_source, description)`.
    Comments (including anchors) are lost in the unparse — the mutated
    copy exists only to be executed and discarded."""
    tree = ast.parse(source)
    fn = _locate_function(tree, qualname)
    if fn is None or isinstance(fn, ast.ClassDef):
        return None
    mutator = _FirstMutant()
    mutator.mark_docstrings(fn)
    # Constants first (rarely-equivalent class); a second pass flips a
    # comparison only when no constant existed.
    mutator.visit(fn)
    if mutator.description is None:
        return None
    ast.fix_missing_locations(tree)
    return ast.unparse(tree), mutator.description


def mutation_check(run_dir: Path, entry: dict, module: Path,
                   timeout_s: int = TEST_TIMEOUT_S) -> dict:
    """Run the module's test against a mechanically broken COPY of its
    function. `caught` = the test failed on the mutant (it distinguishes
    correct from broken code); `survived` = it passed anyway; `not_applicable`
    = the function offers neither mutation class. The run tree is never
    modified."""
    target = run_dir / entry["file"]
    try:
        source = target.read_text(encoding="utf-8")
    except OSError as e:
        return {"status": "not_applicable", "detail": f"unreadable target: {e}"}
    mutated = mutate_function_source(source, entry["qualname"])
    if mutated is None:
        return {"status": "not_applicable",
                "detail": "no numeric constant or comparison to perturb"}
    mutated_source, description = mutated

    with tempfile.TemporaryDirectory(prefix="r2c_mutant_") as tmp:
        root = Path(tmp)
        shutil.copytree(
            run_dir / "method", root / "method",
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "tests"))
        (root / entry["file"]).write_text(mutated_source, encoding="utf-8")
        module_copy = root / module.name
        shutil.copy2(module, module_copy)
        result = run_test_module(root, module_copy, timeout_s=timeout_s)

    if result["status"] == "failed":
        return {"status": "caught", "detail": description}
    if result["status"] == "passed":
        return {"status": "survived", "detail": description}
    return {"status": result["status"], "detail": description}


# ---------------------------------------------------------------------------
# Coverage README
# ---------------------------------------------------------------------------

_STATUS_SENTENCES = {
    "tested": "Tested — passes, and fails on a mechanically broken copy "
              "of its function, so it distinguishes correct from broken code.",
    "weak": "Weak test — passes, but ALSO passed against a mechanically "
            "broken copy of its function, so it could not demonstrate it "
            "distinguishes correct from broken code.",
    "no_mutant": "Tested (no mutant applicable) — passes; the function "
                 "offers no bounded mutation to check the test against.",
    "not_shipped": "Generated but could not be verified — not shipped.",
}


def render_coverage_readme(plan: dict[str, dict],
                           statuses: dict[str, dict]) -> str:
    """The honest coverage index: one row per eligible element.
    `statuses` maps element id to `{"status": <key>, "module": <name>,
    "detail": ...}`; an eligible element with no entry renders as not
    shipped so omissions are visible by construction."""
    lines = [
        "# Per-element tests",
        "",
        "Two levels of verification ship with this delivery. The tests in",
        "this directory verify individual implemented paper elements",
        "against the paper's own statements, one small deterministic test",
        "per element, at fixture scale. The notebook demo is the",
        "integration test that runs everything together.",
        "",
        "Run them from the run directory:",
        "",
        "    python -m pytest method/tests -q",
        "",
        "Every eligible element (implemented in code and mapped to a",
        "function) is listed below, including the ones no shipped test",
        "covers — coverage gaps are disclosed, never silent.",
        "",
    ]
    for element_id in sorted(plan):
        entry = plan[element_id]
        status = statuses.get(element_id) or {"status": "not_shipped"}
        sentence = _STATUS_SENTENCES[status["status"]]
        title = entry["name"] or element_id
        lines.append(f"- **{title}** (`{element_id}`, {entry['section']})")
        lines.append(f"  - {sentence}")
        if status.get("module"):
            lines.append(f"  - Module: `{status['module']}`, exercises "
                         f"`{entry['qualname']}`.")
    if not plan:
        lines.append("- No implemented element maps to a single function "
                     "in this delivery, so no per-element tests were "
                     "generated.")
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Standalone entrypoint: floor + run + mutation over existing tests
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--write-readme", action="store_true",
                        help="rewrite method/tests/README.md from the "
                             "results (the pipeline's generation step "
                             "does this; standalone default is dry)")
    args = parser.parse_args(argv)
    run_dir = Path(args.run_dir)
    if not run_dir.is_dir():
        print(f"FAIL: run dir not found: {run_dir}", file=sys.stderr)
        return 2
    try:
        plan = eligible_elements(run_dir)
    except (JoinTableError, JoinTableSetupError) as e:
        print(f"FAIL: {e}", file=sys.stderr)
        return 2

    floor = validate_tests_dir(run_dir, plan)
    statuses: dict[str, dict] = {}
    any_bad = False
    for module_name, checked in floor.items():
        element_id = checked["element_id"]
        module = run_dir / TESTS_REL_DIR / module_name
        if checked["problems"]:
            any_bad = True
            print(f"FLOOR {module_name}: " + "; ".join(checked["problems"]))
            continue
        result = run_test_module(run_dir, module)
        if result["status"] != "passed":
            any_bad = True
            print(f"{result['status'].upper()} {module_name}")
            continue
        mutant = mutation_check(run_dir, plan[element_id], module)
        key = {"caught": "tested", "survived": "weak"}.get(
            mutant["status"], "no_mutant")
        statuses[element_id] = {"status": key, "module": module_name,
                                "detail": mutant["detail"]}
        print(f"OK {module_name}: {key} ({mutant['detail']})")

    covered = sum(1 for s in statuses.values() if s["status"] == "tested")
    print(f"{covered} tested / {len(statuses)} shipped / "
          f"{len(plan)} eligible elements")
    if args.write_readme and (run_dir / TESTS_REL_DIR).is_dir():
        readme = run_dir / TESTS_REL_DIR / README_NAME
        readme.write_text(render_coverage_readme(plan, statuses),
                          encoding="utf-8")
        print(f"wrote {readme}")
    return 1 if any_bad else 0


if __name__ == "__main__":
    sys.exit(main())
