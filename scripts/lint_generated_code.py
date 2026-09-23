"""US-10: static lint gate over generated code, BEFORE any smoke execution.

Evidence (the probe catalog note (internal, not shipped)): in the 2026-06-09 GBALD run, 2 of
4 smoke-loop iterations were one-line statically-detectable bugs (a missing
`torch.nn` import and a NameError), each costing a 5-11 minute producer
dispatch plus a diagnostician round before the interesting bug was even
reachable. This gate catches that class in milliseconds, pre-execution.

Engine: pyflakes when importable (full undefined-name/unused-import analysis;
add `pyflakes` to the environment — it ships in requirements.txt), with a
syntax-only `compile()` fallback that reports its degraded capability rather
than silently passing. (v3 had a same-named gate in its orphaned verification
layer; this is the revival, simplified.)

Severity policy: syntax errors and undefined names are ERRORS (they are
guaranteed smoke failures); unused imports/variables and redefinitions are
WARNINGS (noise, not breakage). The reversed-argsort idiom is also an ERROR:
it executes cleanly but silently scrambles an intended descending ranking
(bayesian-active-learning 2026-07-29 selector, RCA 2026-08-03 finding 6).
Exit 0 = no errors, 1 = errors, 2 = bad invocation.

Usage:
    python3 scripts/lint_generated_code.py --run-dir <run_dir> [--report out.json]
    python3 scripts/lint_generated_code.py --file a.py --file b.py
"""

from __future__ import annotations

import argparse
import ast
import builtins
import json
import re
import sys
from pathlib import Path

# pyflakes message classes that are guaranteed runtime breakage.
_ERROR_CLASSES = {"UndefinedName", "UndefinedLocal", "UndefinedExport", "SyntaxError"}


def _lintable_source(path: Path) -> str:
    """Read source with IPython magics/shell-escapes neutralized.

    Notebook drafts (jupytext percent format) legitimately contain lines like
    `%pip install -r requirements.txt`, which are valid in a notebook but a
    SyntaxError as plain Python (first observed live on the 2026-06-10 pdwa
    run). Replace them with `pass` so line numbers are preserved — the same
    approach extract_data_flow.py uses.
    """
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    out = []
    for line in lines:
        stripped = line.lstrip()
        indent = line[: len(line) - len(stripped)]
        newline = "\n" if line.endswith("\n") else ""
        placeholder = re.match(r"#\s*%%\s*PLACEHOLDER:\s*(\w+)", stripped)
        if placeholder:
            # The renderer's actual placeholder form (`# %% PLACEHOLDER: x`,
            # confirmed on the real pdwa draft): inject the symbol the
            # rendered cell will define.
            sym = "cfg = {}" if placeholder.group(1) == "params_dict" else "pass"
            out.append(f"{indent}{sym}{newline}")
        elif stripped.startswith(("%", "!")):
            out.append(f"{indent}pass{newline}")
        elif stripped.startswith("{{") and "}}" in stripped:
            # Render-time placeholders: `{{params_dict}}` becomes the `cfg`
            # assignment only in the RENDERED notebook, so the draft would
            # otherwise lint with cfg undefined everywhere (7 false positives
            # on the first real-run battery, 2026-06-10). Substitute the
            # symbol the renderer will define.
            if "params_dict" in stripped:
                out.append(f"{indent}cfg = {{}}{newline}")
            else:
                out.append(f"{indent}pass{newline}")
        else:
            out.append(line)
    return "".join(out)


def _lint_with_pyflakes(path: Path) -> list[dict]:
    from pyflakes import api as pyflakes_api  # noqa: PLC0415
    from pyflakes import reporter as pyflakes_reporter  # noqa: PLC0415

    findings: list[dict] = []

    class _Collector(pyflakes_reporter.Reporter):
        def __init__(self):
            pass

        def unexpectedError(self, filename, msg):  # noqa: N802
            findings.append({
                "probe": "US-10", "file": str(path), "line": None,
                "severity": "error", "message": f"lint error: {msg}",
            })

        def syntaxError(self, filename, msg, lineno, offset, text):  # noqa: N802
            findings.append({
                "probe": "US-10", "file": str(path), "line": lineno,
                "severity": "error", "message": f"SyntaxError: {msg}",
            })

        def flake(self, message):
            cls = type(message).__name__
            findings.append({
                "probe": "US-10",
                "file": str(path),
                "line": message.lineno,
                "severity": "error" if cls in _ERROR_CLASSES else "warn",
                "message": f"{cls}: {message.message % message.message_args}",
            })

    pyflakes_api.check(_lintable_source(path), str(path), _Collector())
    return findings


_BUILTIN_NAMES = set(dir(builtins)) | {
    "__annotations__",
    "__builtins__",
    "__doc__",
    "__file__",
    "__name__",
    "__package__",
}


def _target_names(node: ast.AST) -> set[str]:
    names: set[str] = set()
    if isinstance(node, ast.Name):
        names.add(node.id)
    elif isinstance(node, (ast.Tuple, ast.List)):
        for elt in node.elts:
            names.update(_target_names(elt))
    elif isinstance(node, ast.Starred):
        names.update(_target_names(node.value))
    return names


def _pattern_names(node: ast.AST) -> set[str]:
    names: set[str] = set()
    if isinstance(node, ast.MatchAs):
        if node.name:
            names.add(node.name)
        if node.pattern:
            names.update(_pattern_names(node.pattern))
    elif isinstance(node, ast.MatchStar):
        if node.name:
            names.add(node.name)
    else:
        for child in ast.iter_child_nodes(node):
            names.update(_pattern_names(child))
    return names


def _arg_names(args: ast.arguments) -> set[str]:
    names = {arg.arg for arg in args.posonlyargs + args.args + args.kwonlyargs}
    if args.vararg:
        names.add(args.vararg.arg)
    if args.kwarg:
        names.add(args.kwarg.arg)
    return names


def _collect_scope_bindings(nodes: list[ast.stmt]) -> set[str]:
    bindings: set[str] = set()

    class _BindingCollector(ast.NodeVisitor):
        def visit_Import(self, node: ast.Import) -> None:  # noqa: N802
            for alias in node.names:
                bindings.add(alias.asname or alias.name.split(".")[0])

        def visit_ImportFrom(self, node: ast.ImportFrom) -> None:  # noqa: N802
            for alias in node.names:
                if alias.name != "*":
                    bindings.add(alias.asname or alias.name)

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:  # noqa: N802
            bindings.add(node.name)

        def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:  # noqa: N802
            bindings.add(node.name)

        def visit_ClassDef(self, node: ast.ClassDef) -> None:  # noqa: N802
            bindings.add(node.name)

        def visit_Lambda(self, node: ast.Lambda) -> None:  # noqa: N802
            return None

        def visit_ListComp(self, node: ast.ListComp) -> None:  # noqa: N802
            return None

        def visit_SetComp(self, node: ast.SetComp) -> None:  # noqa: N802
            return None

        def visit_DictComp(self, node: ast.DictComp) -> None:  # noqa: N802
            return None

        def visit_GeneratorExp(self, node: ast.GeneratorExp) -> None:  # noqa: N802
            return None

        def visit_NamedExpr(self, node: ast.NamedExpr) -> None:  # noqa: N802
            bindings.update(_target_names(node.target))
            self.visit(node.value)

        def visit_For(self, node: ast.For) -> None:  # noqa: N802
            bindings.update(_target_names(node.target))
            self.generic_visit(node)

        def visit_AsyncFor(self, node: ast.AsyncFor) -> None:  # noqa: N802
            bindings.update(_target_names(node.target))
            self.generic_visit(node)

        def visit_With(self, node: ast.With) -> None:  # noqa: N802
            for item in node.items:
                if item.optional_vars:
                    bindings.update(_target_names(item.optional_vars))
            self.generic_visit(node)

        def visit_AsyncWith(self, node: ast.AsyncWith) -> None:  # noqa: N802
            for item in node.items:
                if item.optional_vars:
                    bindings.update(_target_names(item.optional_vars))
            self.generic_visit(node)

        def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:  # noqa: N802
            if node.name:
                bindings.add(node.name)
            self.generic_visit(node)

        def visit_Match(self, node: ast.Match) -> None:  # noqa: N802
            for case in node.cases:
                bindings.update(_pattern_names(case.pattern))
                for stmt in case.body:
                    self.visit(stmt)

        def visit_Name(self, node: ast.Name) -> None:  # noqa: N802
            if isinstance(node.ctx, (ast.Store, ast.Del)):
                bindings.add(node.id)

    collector = _BindingCollector()
    for node in nodes:
        collector.visit(node)
    return bindings


def _lint_with_ast_names(path: Path) -> list[dict]:
    """Conservative undefined-name fallback for environments without pyflakes."""
    try:
        tree = ast.parse(_lintable_source(path), filename=str(path))
    except SyntaxError as e:
        return [{
            "probe": "US-10", "file": str(path), "line": e.lineno,
            "severity": "error", "message": f"SyntaxError: {e.msg}",
        }]

    findings: list[dict] = []

    class _Scope:
        def __init__(self, parent: "_Scope | None", bindings: set[str]):
            self.parent = parent
            self.bindings = set(bindings)

        def resolves(self, name: str) -> bool:
            scope: _Scope | None = self
            while scope is not None:
                if name in scope.bindings:
                    return True
                scope = scope.parent
            return name in _BUILTIN_NAMES

    def _error(path: Path, node: ast.AST, name: str) -> None:
        findings.append({
            "probe": "US-10",
            "file": str(path),
            "line": getattr(node, "lineno", None),
            "severity": "error",
            "message": f"UndefinedName: undefined name '{name}'",
        })

    def _analyze_expr(node: ast.AST, scope: _Scope) -> None:
        if isinstance(node, ast.Name):
            if isinstance(node.ctx, ast.Load) and not scope.resolves(node.id):
                _error(path, node, node.id)
            return
        if isinstance(node, ast.Lambda):
            lambda_scope = _Scope(scope, _arg_names(node.args))
            for default in node.args.defaults + node.args.kw_defaults:
                if default is not None:
                    _analyze_expr(default, scope)
            _analyze_expr(node.body, lambda_scope)
            return
        if isinstance(node, (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)):
            comp_scope = _Scope(scope, set())
            for generator in node.generators:
                _analyze_expr(generator.iter, comp_scope)
                comp_scope.bindings.update(_target_names(generator.target))
                for cond in generator.ifs:
                    _analyze_expr(cond, comp_scope)
            if isinstance(node, ast.DictComp):
                _analyze_expr(node.key, comp_scope)
                _analyze_expr(node.value, comp_scope)
            else:
                _analyze_expr(node.elt, comp_scope)
            return
        for child in ast.iter_child_nodes(node):
            _analyze_expr(child, scope)

    def _analyze_body(nodes: list[ast.stmt], parent: _Scope | None = None, initial: set[str] | None = None) -> _Scope:
        scope = _Scope(parent, (initial or set()) | _collect_scope_bindings(nodes))
        for node in nodes:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                for dec in node.decorator_list:
                    _analyze_expr(dec, scope)
                for default in node.args.defaults + node.args.kw_defaults:
                    if default is not None:
                        _analyze_expr(default, scope)
                if node.returns:
                    _analyze_expr(node.returns, scope)
                _analyze_body(node.body, scope, _arg_names(node.args))
            elif isinstance(node, ast.ClassDef):
                for dec in node.decorator_list:
                    _analyze_expr(dec, scope)
                for base in node.bases:
                    _analyze_expr(base, scope)
                for keyword in node.keywords:
                    _analyze_expr(keyword.value, scope)
                _analyze_body(node.body, scope)
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                continue
            else:
                _analyze_expr(node, scope)
        return scope

    _analyze_body(tree.body)
    return findings


def _is_full_reversal(node: ast.AST) -> bool:
    """True for `x[::-1]` (any expression, bare full-reversal slice)."""
    return (
        isinstance(node, ast.Subscript)
        and isinstance(node.slice, ast.Slice)
        and node.slice.lower is None
        and node.slice.upper is None
        and isinstance(node.slice.step, ast.UnaryOp)
        and isinstance(node.slice.step.op, ast.USub)
        and isinstance(node.slice.step.operand, ast.Constant)
        and node.slice.step.operand.value == 1
    )


def _lint_reversed_argsort(path: Path) -> list[dict]:
    """Flag `argsort(x[::-1])` and `x[::-1].argsort()`.

    An argsort over a reversed array ranks the reversed copy, so its indices
    are meaningless against the original — the idiom appears when a coder
    intends a descending ranking. The correct descending forms
    (`argsort(x)[::-1]`, `argsort(-x)`) do not match. Unlike the
    crash direction (a negative-stride RESULT fed to torch indexing,
    gbald-negstride-selector), this direction executes cleanly, so it must
    die at lint rather than at smoke."""
    try:
        tree = ast.parse(_lintable_source(path), filename=str(path))
    except SyntaxError:
        return []  # the engine lints report the syntax error

    findings: list[dict] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Attribute) and func.attr == "argsort":
            subject = node.args[0] if node.args else func.value
        elif isinstance(func, ast.Name) and func.id == "argsort" and node.args:
            subject = node.args[0]
        else:
            continue
        if _is_full_reversal(subject):
            findings.append({
                "probe": "US-10",
                "file": str(path),
                "line": node.lineno,
                "severity": "error",
                "message": (
                    "ReversedArgsort: argsort over a reversed array "
                    "(`argsort(x[::-1])`) ranks the reversed copy — its "
                    "indices are meaningless against the original array. "
                    "For a descending ranking use `argsort(x)[::-1]` or "
                    "`argsort(-x)`."
                ),
            })
    return findings


def lint_paths(paths: list[Path]) -> tuple[list[dict], str]:
    """Lint files; returns (findings, engine) where engine notes capability."""
    try:
        import pyflakes  # noqa: F401, PLC0415
        engine = "pyflakes"
        lint = _lint_with_pyflakes
    except ImportError:
        engine = "ast-fallback (pyflakes not installed — limited undefined-name detection)"
        lint = _lint_with_ast_names

    findings: list[dict] = []
    for path in paths:
        if path.is_file():
            findings.extend(lint(path))
            findings.extend(_lint_reversed_argsort(path))
    return findings, engine


def default_targets(run_dir: Path) -> list[Path]:
    targets = sorted((run_dir / "method").glob("*.py"))
    draft = run_dir / ".pipeline" / "notebook_draft.py"
    if draft.is_file():
        targets.append(draft)
    return targets


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--run-dir", help="lint method/*.py + notebook_draft.py")
    parser.add_argument("--file", action="append", default=[],
                        help="explicit file target (repeatable)")
    parser.add_argument("--report", help="write probe verdicts JSON here")
    args = parser.parse_args(argv)

    targets: list[Path] = [Path(f) for f in args.file]
    if args.run_dir:
        run_dir = Path(args.run_dir)
        if not run_dir.is_dir():
            print(f"FAIL: run dir not found: {run_dir}", file=sys.stderr)
            return 2
        targets.extend(default_targets(run_dir))
    if not targets:
        print("FAIL: nothing to lint (pass --run-dir or --file)", file=sys.stderr)
        return 2

    findings, engine = lint_paths(targets)
    if args.report:
        Path(args.report).write_text(
            json.dumps({"validator": "lint_generated_code", "engine": engine,
                        "findings": findings}, indent=2) + "\n",
            encoding="utf-8",
        )

    errors = [f for f in findings if f["severity"] == "error"]
    warns = [f for f in findings if f["severity"] == "warn"]
    for f in warns:
        print(f"WARN {f['file']}:{f['line']}: {f['message']}")
    if errors:
        print(f"FAIL: {len(errors)} lint error(s) (engine: {engine}):")
        for f in errors:
            print(f"  - {f['file']}:{f['line']}: {f['message']}")
        return 1
    print(f"ok: lint gate passed over {len(targets)} file(s) "
          f"({len(warns)} warning(s); engine: {engine})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
