"""Shared AST helpers for signature/contract validation.

Extracted 2026-07-04 from validate_method_package.py (extract-then-retire,
repo-map verdict §3): the package-level validator was superseded by the
per-producer gates, but its AST helpers are load-bearing for
validate_method_spec.py and validate_method_coder_output.py. Underscore
names are kept so the import sites stay byte-compatible.
"""

from __future__ import annotations

import ast
from pathlib import Path


def _find_function_def(tree: ast.Module, name: str) -> ast.FunctionDef | ast.AsyncFunctionDef | None:
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return node
    return None


def _signature_string(func: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    """Reconstruct a callable signature from the AST (modulo whitespace).

    B-04 merged the two drifted reconstructors onto the scaffolder
    validator's version, the strictly more faithful one: it emits the `/`
    positional-only marker (the other never did). If anyone flips that
    back, manifest signatures declaring `/` start mismatching at the
    scaffolder gate — there is a positional-only fixture pinning this."""
    args = func.args
    parts: list[str] = []

    posonly = list(args.posonlyargs)
    posn = posonly + list(args.args)
    defaults = list(args.defaults)
    n_def = len(defaults)
    for i, arg in enumerate(posn):
        s = arg.arg
        if arg.annotation is not None:
            s += ": " + ast.unparse(arg.annotation)
        if i >= len(posn) - n_def:
            di = i - (len(posn) - n_def)
            s += " = " + ast.unparse(defaults[di]) if arg.annotation else "=" + ast.unparse(defaults[di])
        parts.append(s)
        if i == len(posonly) - 1 and posonly:
            parts.append("/")

    if args.vararg:
        s = "*" + args.vararg.arg
        if args.vararg.annotation is not None:
            s += ": " + ast.unparse(args.vararg.annotation)
        parts.append(s)
    elif args.kwonlyargs:
        parts.append("*")
    for i, arg in enumerate(args.kwonlyargs):
        s = arg.arg
        if arg.annotation is not None:
            s += ": " + ast.unparse(arg.annotation)
        if args.kw_defaults[i] is not None:
            s += " = " + ast.unparse(args.kw_defaults[i]) if arg.annotation else "=" + ast.unparse(args.kw_defaults[i])
        parts.append(s)
    if args.kwarg:
        s = "**" + args.kwarg.arg
        if args.kwarg.annotation is not None:
            s += ": " + ast.unparse(args.kwarg.annotation)
        parts.append(s)

    sig = f"{func.name}({', '.join(parts)})"
    if func.returns is not None:
        sig += " -> " + ast.unparse(func.returns)
    return sig


def _all_param_names(func: ast.FunctionDef | ast.AsyncFunctionDef) -> list[str]:
    args = func.args
    names = [a.arg for a in args.posonlyargs + args.args + args.kwonlyargs]
    if args.vararg:
        names.append(args.vararg.arg)
    if args.kwarg:
        names.append(args.kwarg.arg)
    return names


def _parse_signature_string_param_names(sig_str: str) -> list[str] | None:
    """Parse a spec signature string like
    `select_batch(model, x_unlabeled, batch_size, seed, mc_samples: int = 100) -> List[int]`
    and return the parameter names. Returns None if parsing fails (e.g., placeholder text
    instead of a real signature)."""
    try:
        stub = f"def {sig_str}:\n    pass\n"
        tree = ast.parse(stub)
    except (SyntaxError, ValueError):
        return None
    if not tree.body or not isinstance(tree.body[0], (ast.FunctionDef, ast.AsyncFunctionDef)):
        return None
    return _all_param_names(tree.body[0])


def _name_referenced_in_body(func: ast.FunctionDef | ast.AsyncFunctionDef, name: str) -> bool:
    for node in ast.walk(func):
        if isinstance(node, ast.Name) and node.id == name:
            return True
    return False


def _is_stochastic_call(call: ast.Call) -> bool:
    """Return True if the call looks like a stochastic operation that should be seeded."""
    func = call.func
    parts: list[str] = []
    while isinstance(func, ast.Attribute):
        parts.insert(0, func.attr)
        func = func.value
    if isinstance(func, ast.Name):
        parts.insert(0, func.id)

    if len(parts) < 2:
        return False
    if parts[0] in ("np", "numpy") and len(parts) >= 3 and parts[1] == "random":
        return True
    if parts[0] == "torch" and len(parts) >= 2 and parts[1] in (
        "rand", "randn", "randint", "randperm", "rand_like", "randn_like",
        "multinomial", "bernoulli", "poisson", "normal",
    ):
        return True
    if parts[0] == "random" and len(parts) >= 2:
        return True
    return False


def _find_stochastic_calls(func: ast.FunctionDef | ast.AsyncFunctionDef) -> list[ast.Call]:
    return [n for n in ast.walk(func) if isinstance(n, ast.Call) and _is_stochastic_call(n)]


def _format_call_location(call: ast.Call, file_path: Path) -> str:
    return f"{file_path.name}:{call.lineno} {ast.unparse(call.func)}(...)"


def _normalize_sig(s: str) -> str:
    return "".join(s.split())
