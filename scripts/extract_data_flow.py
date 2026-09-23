"""Stage 2.d: extract a structured data-flow context for the generated package.

The smoke-gate diagnostician (and any future agent that needs to trace
how a value flows through the package) consumes
`<RUN_DIR>/.pipeline/data_flow.json` instead of reading every .py file
to infer the chain. The extractor AST-walks every Python source file in
the run (method/*.py + the notebook draft) and records, per named
symbol, all assignment sites and read sites.

Diagnostician usage: when a smoke failure says "y_train has labels in
[0, 7] but n_classes=3 requires [0, 2]", the diagnostician looks up
`n_classes` in data_flow.json, sees the assignment site in
`notebook_draft.py`, reads that line's expression
(`len(y_pool.unique())`), and immediately knows the bug is in the
notebook's derivation — not in the validator that surfaced the error.
This compresses the "trace the value to its origin" step from "read N
files until you find the assignment" into a structured lookup.

The extractor is deliberately simple: textual occurrences of names at
assignment + call sites. It does NOT attempt full scope/use-def
analysis (which would require type inference + cross-file symbol
resolution). For the diagnostician's needs, a high-recall list of
"places where this name is set" and "places where this name is read"
is sufficient.

Run from the orchestrator after Stage 2.d's init_finalizer step. Output
goes to `<RUN_DIR>/.pipeline/data_flow.json`. Failure to extract is
NOT fatal — the diagnostician falls back to reading files manually.

Usage:

    python scripts/extract_data_flow.py --run-dir <output_dir>

Exit codes:
  0  extraction complete (data_flow.json written)
  1  could not write data_flow.json (will be skipped by diagnostician)
  2  setup error
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


SCHEMA_VERSION = "1.0.0"


# Files we scan: method/*.py plus the rendered notebook source (notebook_draft.py).
# data.py is paradigm-fixed (scaffolder-written) but is still in scope —
# tracing a value's origin often lands in load_data().
def _files_to_scan(run_dir: Path) -> list[Path]:
    candidates: list[Path] = []
    method_dir = run_dir / "method"
    if method_dir.is_dir():
        candidates.extend(sorted(method_dir.rglob("*.py")))
    nb_draft = run_dir / ".pipeline" / "notebook_draft.py"
    if nb_draft.is_file():
        candidates.append(nb_draft)
    # Skip auto-generated / cache files.
    return [
        p for p in candidates
        if "__pycache__" not in p.parts and not p.name.startswith("_test_")
    ]


def _rel(path: Path, run_dir: Path) -> str:
    """Return a stable POSIX-form relative path."""
    return path.relative_to(run_dir).as_posix()


def _strip_jupytext_magics(source: str) -> str:
    """Strip IPython magic lines so the source is parseable as Python.

    jupytext-percent files (notebook_draft.py) contain `%matplotlib inline`,
    `%pip install ...`, and similar IPython directives. These are valid in
    a notebook cell but invalid Python. Replace them with `pass` (preserves
    line numbers so AST line attributes still map back to the original
    file). Cell-marker comments (`# %%`) and shebangs are already valid
    Python comments and don't need replacing."""
    out_lines: list[str] = []
    for line in source.splitlines():
        stripped = line.lstrip()
        if (
            stripped.startswith(("%", "!", "?"))
            and not stripped.startswith(("# ", "#!"))
        ):
            # Preserve indent so the replacement matches surrounding context.
            indent = line[: len(line) - len(stripped)]
            out_lines.append(f"{indent}pass  # IPython magic stripped")
        else:
            out_lines.append(line)
    return "\n".join(out_lines)


def _expr_summary(node: ast.AST, max_len: int = 120) -> str:
    """One-line summary of an expression. Uses ast.unparse for fidelity;
    truncates long expressions to keep data_flow.json bounded."""
    try:
        s = ast.unparse(node)
    except Exception:
        s = type(node).__name__
    s = " ".join(s.split())  # collapse whitespace
    if len(s) > max_len:
        s = s[: max_len - 3] + "..."
    return s


def _enclosing_context(node: ast.AST, ancestors: list[ast.AST]) -> str:
    """Best-effort name of the enclosing scope: function, class, or
    `<module>` for module-level."""
    for a in reversed(ancestors):
        if isinstance(a, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return a.name
        if isinstance(a, ast.ClassDef):
            return a.name
    return "<module>"


def _extract_from_tree(
    tree: ast.Module, file_rel: str
) -> dict[str, dict[str, list[dict[str, Any]]]]:
    """Walk a single file's AST and return:

        {
          name: {
            "assignments": [{"file", "line", "scope", "expr"}],
            "reads": [{"file", "line", "scope", "context"}],
          }
        }

    `name` is keyed by whatever simple Python name appears on the LHS
    of an assignment or in a Call's arg/positional. We ignore subscripts
    + attribute accesses (e.g., `model.fc_out.out_features` is recorded
    only as a context string, not as a separate symbol)."""
    out: dict[str, dict[str, list[dict[str, Any]]]] = {}

    def _ensure(name: str) -> dict[str, list[dict[str, Any]]]:
        if name not in out:
            out[name] = {"assignments": [], "reads": []}
        return out[name]

    # Walk with parent-stack so we know the enclosing scope for each node.
    stack: list[ast.AST] = []

    def _visit(node: ast.AST) -> None:
        # Assignments — LHS names become assignment sites with the RHS expr.
        if isinstance(node, ast.Assign):
            for target in node.targets:
                names = _names_in_target(target)
                for name in names:
                    _ensure(name)["assignments"].append({
                        "file": file_rel,
                        "line": node.lineno,
                        "scope": _enclosing_context(node, stack),
                        "expr": _expr_summary(node.value),
                    })
            # RHS reads are walked below via recursion.
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            if isinstance(node.target, ast.Name):
                _ensure(node.target.id)["assignments"].append({
                    "file": file_rel,
                    "line": node.lineno,
                    "scope": _enclosing_context(node, stack),
                    "expr": _expr_summary(node.value),
                })
        elif isinstance(node, (ast.AugAssign,)) and isinstance(node.target, ast.Name):
            _ensure(node.target.id)["assignments"].append({
                "file": file_rel,
                "line": node.lineno,
                "scope": _enclosing_context(node, stack),
                "expr": f"{node.target.id} {type(node.op).__name__} {_expr_summary(node.value, max_len=60)}",
            })
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            # Function parameters are "assignments" of arguments to local names.
            for arg in (
                node.args.args
                + node.args.kwonlyargs
                + (node.args.posonlyargs if hasattr(node.args, "posonlyargs") else [])
            ):
                _ensure(arg.arg)["assignments"].append({
                    "file": file_rel,
                    "line": node.lineno,
                    "scope": node.name,
                    "expr": f"<parameter of {node.name}>",
                })
            if node.args.vararg is not None:
                _ensure(node.args.vararg.arg)["assignments"].append({
                    "file": file_rel,
                    "line": node.lineno,
                    "scope": node.name,
                    "expr": f"<*{node.args.vararg.arg} parameter of {node.name}>",
                })
            if node.args.kwarg is not None:
                _ensure(node.args.kwarg.arg)["assignments"].append({
                    "file": file_rel,
                    "line": node.lineno,
                    "scope": node.name,
                    "expr": f"<**{node.args.kwarg.arg} parameter of {node.name}>",
                })
        elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            # Read site — but skip the common case where the Name is the
            # callee of a Call (the callee is the function being invoked,
            # not a value being read in the data-flow sense the diagnostician
            # cares about — except that it IS a read of that function name).
            # We keep all Name-load occurrences; filtering happens at consumer
            # side if needed.
            _ensure(node.id)["reads"].append({
                "file": file_rel,
                "line": node.lineno,
                "scope": _enclosing_context(node, stack),
                "context": _surrounding_context(node, stack),
            })

        # Recurse with stack management.
        stack.append(node)
        for child in ast.iter_child_nodes(node):
            _visit(child)
        stack.pop()

    _visit(tree)
    return out


def _names_in_target(target: ast.AST) -> list[str]:
    """LHS of an assignment can be a Name, Tuple, List, or nested. Return
    every simple Name id we find."""
    out: list[str] = []
    if isinstance(target, ast.Name):
        out.append(target.id)
    elif isinstance(target, (ast.Tuple, ast.List)):
        for elt in target.elts:
            out.extend(_names_in_target(elt))
    elif isinstance(target, ast.Starred):
        out.extend(_names_in_target(target.value))
    # Attribute / Subscript targets are skipped — they're modifications of
    # an existing object, not new bindings for the diagnostician's purposes.
    return out


def _surrounding_context(node: ast.AST, stack: list[ast.AST], max_len: int = 100) -> str:
    """Return a short snippet of the nearest enclosing statement so the
    consumer (diagnostician) can see HOW the name is being read."""
    for a in reversed(stack):
        if isinstance(a, (
            ast.Assign, ast.AnnAssign, ast.AugAssign, ast.Return,
            ast.Expr, ast.If, ast.While, ast.For, ast.Raise,
        )):
            try:
                s = ast.unparse(a)
            except Exception:
                s = type(a).__name__
            s = " ".join(s.split())
            if len(s) > max_len:
                s = s[: max_len - 3] + "..."
            return s
    return "<expr>"


def _merge(target: dict, source: dict) -> None:
    """Merge per-file extraction results into the cross-file map."""
    for name, sections in source.items():
        if name not in target:
            target[name] = {"assignments": [], "reads": []}
        target[name]["assignments"].extend(sections["assignments"])
        target[name]["reads"].extend(sections["reads"])


def extract(run_dir: Path) -> dict[str, Any]:
    """Walk every Python file in scope and build the data-flow map."""
    symbols: dict[str, dict[str, list[dict[str, Any]]]] = {}
    files_scanned: list[str] = []
    for path in _files_to_scan(run_dir):
        rel = _rel(path, run_dir)
        try:
            source = path.read_text(encoding="utf-8")
            # Preprocess for jupytext-percent files (notebook_draft.py contains
            # IPython magics that break stock Python parsing).
            if path.name == "notebook_draft.py":
                source = _strip_jupytext_magics(source)
            tree = ast.parse(source, filename=str(path))
        except (SyntaxError, OSError) as e:
            print(
                f"warning: could not parse {rel}: {e}; skipping",
                file=sys.stderr,
            )
            continue
        files_scanned.append(rel)
        _merge(symbols, _extract_from_tree(tree, rel))

    return {
        "schema_version": SCHEMA_VERSION,
        "files_scanned": sorted(files_scanned),
        "symbols": symbols,
    }

