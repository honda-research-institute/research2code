"""Notebook AST analysis for AL acquisition loops — dependency-free.

Extracted from validate_notebook_output.py (R2C-019 block 2): the probe
battery's AL-loop probes consume these helpers, and the portable harness
vendors the probes package without the validator's nbformat and taxonomy
imports. This module is deliberately stdlib-only (ast + re) so the
vendored battery carries it verbatim. validate_notebook_output re-imports
everything here, so there is exactly one implementation.

The analysis walks notebook code cells as AST: which functions each loop
body calls (expanded through notebook-defined helpers), whether a loop
iterates the configured round count, and whether statements mutate the
labeled/unlabeled index sets or append learning-curve points. The two
top-level detectors pin real audit findings: the M-003 warm-start bug and
the post-merge-count-for-pre-merge-eval curve mislabeling.
"""

from __future__ import annotations

import ast
import re


def _called_names_within(node: ast.AST) -> set[str]:
    """Every function name called anywhere within an AST subtree.

    Both `build_model(...)` (ast.Name) and `obj.build_model(...)` (ast.Attribute)
    contribute `build_model`. Used to inspect what a loop body invokes."""
    names: set[str] = set()
    for n in ast.walk(node):
        if isinstance(n, ast.Call):
            fn = n.func
            if isinstance(fn, ast.Name):
                names.add(fn.id)
            elif isinstance(fn, ast.Attribute):
                names.add(fn.attr)
    return names


def _notebook_call_graph(code_cells: list[str]) -> dict[str, set[str]]:
    """{function name: names it calls} for every def in the notebook."""
    graph: dict[str, set[str]] = {}
    for src in code_cells:
        try:
            tree = ast.parse(src)
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                graph[node.name] = _called_names_within(node)
    return graph


def _expand_through_helpers(direct: set[str],
                            graph: dict[str, set[str]]) -> set[str]:
    """Close a call set over notebook-defined helpers (visited-bounded)."""
    seen = set(direct)
    stack = [n for n in direct if n in graph]
    while stack:
        for callee in graph.get(stack.pop(), ()):
            if callee not in seen:
                seen.add(callee)
                if callee in graph:
                    stack.append(callee)
    return seen


def _al_loop_warm_starts(code_cells: list[str], *, train_fn: str, build_fn: str) -> bool | None:
    """Detect the M-003 warm-start bug in an active-learning acquisition loop.

    Walk every `for`/`while` loop across the code cells. If a loop body calls
    `train_fn` (per-round training) it must ALSO call `build_fn` (construct a
    fresh model that round) — otherwise the loop reuses one model object across
    rounds, i.e. warm-starts, which retrain-from-scratch AL forbids.

    Calls are expanded through functions DEFINED in the notebook: a loop that
    retrains via a local helper (GBALD's `model = train_eval_current(seed)`,
    whose body calls build_model + train_from_scratch) is in-loop training,
    not invisible — the 2026-06-12 GBALD audit found this exact blind spot
    reporting "no in-loop training" on a loop that retrains every round.

    Returns True if a warm-starting loop is found, False if an in-loop-training
    loop rebuilds correctly, None if no in-loop training is present (check N/A —
    don't false-positive on notebooks that train only once, e.g. a bootstrap)."""
    graph = _notebook_call_graph(code_cells)
    saw_inloop_training = False
    for src in code_cells:
        try:
            tree = ast.parse(src)
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, (ast.For, ast.While)):
                calls = _expand_through_helpers(
                    _called_names_within(node), graph)
                if train_fn in calls:
                    saw_inloop_training = True
                    if build_fn not in calls:
                        return True
    return False if saw_inloop_training else None


def _cfg_key(node: ast.AST) -> str | None:
    if not isinstance(node, ast.Subscript):
        return None
    if not isinstance(node.value, ast.Name) or node.value.id != "cfg":
        return None
    slc = node.slice
    if isinstance(slc, ast.Constant) and isinstance(slc.value, str):
        return slc.value
    return None


def _integer_literal(node: ast.AST) -> int | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, int):
        return node.value
    return None


def _is_num_rounds_expr(node: ast.AST) -> bool:
    return (
        (isinstance(node, ast.Name) and node.id == "num_rounds")
        or _cfg_key(node) == "num_rounds"
    )


def _iterates_num_rounds(node: ast.For) -> bool:
    if not isinstance(node.iter, ast.Call):
        return False
    if not isinstance(node.iter.func, ast.Name) or node.iter.func.id != "range":
        return False
    return bool(node.iter.args and _is_num_rounds_expr(node.iter.args[0]))


def _stmt_calls_name(stmt: ast.AST, name: str) -> bool:
    return name in _called_names_within(stmt)


def _stmt_appends_learning_curve(stmt: ast.AST) -> bool:
    for node in ast.walk(stmt):
        if not isinstance(node, ast.Call):
            continue
        if not isinstance(node.func, ast.Attribute) or node.func.attr != "append":
            continue
        target = node.func.value
        if isinstance(target, ast.Name) and target.id == "learning_curve":
            return True
    return False


_LABEL_INDEX_RE = re.compile(r"(?:^|_)(?:un)?labeled_(?:idx|indices)$")


def _is_label_index_target(node: ast.AST) -> bool:
    if isinstance(node, ast.Name):
        return bool(_LABEL_INDEX_RE.search(node.id))
    if isinstance(node, (ast.Tuple, ast.List)):
        return any(_is_label_index_target(elt) for elt in node.elts)
    return False


def _stmt_mutates_label_indices(stmt: ast.AST) -> bool:
    """True when a statement updates the notebook's labeled/unlabeled sets."""
    for node in ast.walk(stmt):
        if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if any(_is_label_index_target(target) for target in targets):
                return True
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            target = node.func.value
            if (
                isinstance(target, ast.Name)
                and _LABEL_INDEX_RE.search(target.id)
                and node.func.attr in {"append", "extend", "add", "update"}
            ):
                return True
    return False


def _stmt_calls_expanded(
    stmt: ast.AST, graph: dict[str, set[str]], name: str
) -> bool:
    return name in _expand_through_helpers(_called_names_within(stmt), graph)


def _al_loop_has_postmerge_count_for_premerge_eval(
    code_cells: list[str],
    *,
    pluggable_name: str,
    train_fn: str = "train_from_scratch",
) -> bool:
    """Detect learning-curve points that label pre-merge model results with
    post-merge label counts.

    Failure shape:
      for r in range(cfg["num_rounds"]):
          model = train_from_scratch(... current labeled set ...)
          acc = evaluate(model, ...)
          pos = select_batch(...)
          labeled_idx = np.union1d(labeled_idx, chosen)
          learning_curve.append((len(labeled_idx), acc))

    The append is after the acquisition state changed, but no retraining
    happened after that change; the point is therefore plotted at the new
    label budget while measuring the old model. Correct loops either append
    before acquisition (initial eval only) or retrain/evaluate after the
    merge before appending the post-acquisition point.
    """
    graph = _notebook_call_graph(code_cells)
    for src in code_cells:
        try:
            tree = ast.parse(src)
        except SyntaxError:
            continue
        for loop in ast.walk(tree):
            if not isinstance(loop, ast.For) or not _iterates_num_rounds(loop):
                continue
            if pluggable_name not in _called_names_within(loop):
                continue
            saw_acquisition = False
            saw_label_update_after_acquisition = False
            retrained_after_update = False
            for stmt in loop.body:
                if _stmt_calls_name(stmt, pluggable_name):
                    saw_acquisition = True
                if saw_acquisition and _stmt_mutates_label_indices(stmt):
                    saw_label_update_after_acquisition = True
                    retrained_after_update = False
                elif saw_label_update_after_acquisition and _stmt_calls_expanded(
                    stmt, graph, train_fn
                ):
                    retrained_after_update = True
                if (
                    saw_label_update_after_acquisition
                    and _stmt_appends_learning_curve(stmt)
                    and not retrained_after_update
                ):
                    return True
    return False
