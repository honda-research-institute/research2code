"""Stage 2.c — validator for method_coder output.

Runs after the method_coder agent. Cross-references the run directory against
the matched paradigm's `package_manifest.files` block (entry with
`produced_by: method_coder`) and the `pluggable_component.contract`. Confirms:

  1. `<run_dir>/method/method.py` exists and parses.
  2. The pluggable function (named per spec.comparison.pluggable_component.name)
     is defined at top level with the spec's signature (delegates to the
     AST checks now shared via scripts/signature_ast.py; the package-level validator was retired 2026-07-04).
  3. No parameter (in any top-level function in method.py) has a name in
     `pluggable_component.contract.forbidden_param_names` (today: no `config`).
  3b. Package-import convention (R2C-036): no bare sibling imports
     (`from model import ...` / `import training`) — the package is consumed
     as `method.<module>`, so the bare form dies in every real consumer.
     Caught here so the coder's fix loop repairs it instead of the run
     halting at 2.d finalization (DomIndOnto 0728c resume).
  4. Stateless rules:
     - No `hasattr(model, ...)` calls in any function body.
     - No module-level assignment that gets re-assigned inside a function
       (heuristic for "module-level mutable state populated by a call").
  5. Seed plumbing:
     - The pluggable function's `seed_param` is referenced in its body.
     - No unseeded stochastic calls (delegates to signature_ast's
       _find_stochastic_calls — same C-22 logic as the existing validator).
  6. Library-boundary anti-patterns (heuristic AST checks):
     - No `.unsqueeze()`, `.detach()`, `.numpy()`, `.cpu()` calls on names that
       look numpy-shaped (best-effort; the receiver type isn't always derivable
       from AST, so we only flag obvious cases and let the reviewer catch the rest).
     - No `<python_list_literal>.tolist()` patterns.
  7. Forbidden default RNG usage:
     - No `np.random.default_rng()` with no argument.
     - No bare `np.random.choice` / `np.random.shuffle` / etc. without an `rng=`.
  8. **Paper-element ID validity**: every `# paper-element: <id>` annotation in
     method.py must reference an id that exists in `<run_dir>/.pipeline/paper_map.json`.
     A broken trace misleads any reader (or reviewer) trying to follow the code →
     paper link.
  9. **Essential-annotation coverage** (cross-file): for every feature with
     `severity: essential` in spec.critical_requirements.model.specific_features,
     a `# essential: <feature-name>` annotation must appear somewhere in the
     `method/` package (model.py, training.py, or method.py — model.py is the
     most common spot for architecture-essential features; method.py for
     algorithm-essential ones). This check runs at the method-coder gate
     because both architecture-coder and method-coder have produced their files
     by then.

Deterministic gate. On failure, exit 1 with a structured error list on stderr;
the orchestrator re-dispatches the method_coder with the errors verbatim.

Usage:

    python scripts/validate_method_coder_output.py --spec <method_spec.json> --run-dir <output_dir>

Exit codes:
  0  validation passed (zero errors)
  1  validation failures
  2  setup error (missing spec / taxonomy build plan / manifest)
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from schemas.method_spec import (  # noqa: E402
    evaluation_protocol_identity_labels,
    evaluation_protocol_quantity_for_param,
    normalized_seed_param,
)
from scripts.build_plan import load_build_plan  # noqa: E402
from scripts.taxonomy import run_overlay_dir  # noqa: E402
from scripts.validate_architecture_coder_output import (  # noqa: E402
    _load_paper_map_ids,
)
from scripts.paper_element_anchors import (  # noqa: E402
    extract_paper_element_ids as _extract_paper_element_ids,
)
from scripts.signature_ast import (  # noqa: E402
    _all_param_names,
    _find_function_def,
    _find_stochastic_calls,
    _name_referenced_in_body,
    _normalize_sig,
    _signature_string,
)


METHOD_CODER_ROLE = "method_coder"


# ---------------------------------------------------------------------------
# AST helpers
# ---------------------------------------------------------------------------


def _attribute_chain(node: ast.AST) -> list[str]:
    """Walk an Attribute / Name expression, return ['module', 'attr1', ...]."""
    parts: list[str] = []
    cur = node
    while isinstance(cur, ast.Attribute):
        parts.insert(0, cur.attr)
        cur = cur.value
    if isinstance(cur, ast.Name):
        parts.insert(0, cur.id)
    return parts


def _has_hasattr_model_call(tree: ast.Module) -> list[tuple[int, str]]:
    """Find any hasattr(model, ...) call. Returns [(line, snippet), ...]."""
    hits: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "hasattr":
            if node.args and isinstance(node.args[0], ast.Name) and node.args[0].id == "model":
                hits.append((node.lineno, ast.unparse(node)))
    return hits


def _module_level_names_assigned_in_functions(tree: ast.Module) -> list[tuple[int, str]]:
    """Heuristic: find module-level names that are reassigned inside a function body.

    Catches `_CACHE = {}` at module scope being mutated/replaced inside a function.
    Doesn't catch every form (subscript, method calls), but flags obvious cases.
    """
    module_level: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for tgt in node.targets:
                if isinstance(tgt, ast.Name):
                    module_level.add(tgt.id)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            module_level.add(node.target.id)

    hits: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for inner in ast.walk(node):
                if isinstance(inner, (ast.Assign, ast.AugAssign)):
                    targets = inner.targets if isinstance(inner, ast.Assign) else [inner.target]
                    for tgt in targets:
                        if isinstance(tgt, ast.Name) and tgt.id in module_level:
                            hits.append((inner.lineno, f"function `{node.name}` reassigns module-level name `{tgt.id}`"))
    return hits


def _bare_default_rng_calls(tree: ast.Module) -> list[int]:
    """Find np.random.default_rng() calls with no arguments."""
    hits: list[int] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and not node.args and not node.keywords:
            chain = _attribute_chain(node.func)
            if chain and chain[-1] == "default_rng" and "random" in chain:
                hits.append(node.lineno)
    return hits


def _global_seed_mutation_calls(tree: ast.Module) -> list[tuple[int, str]]:
    """Calls that mutate process-global RNG state (torch.manual_seed,
    torch.cuda.manual_seed[_all], np.random.seed, random.seed) anywhere in
    method.py. The method module is the LIBRARY surface: its functions run
    per-call inside notebooks and probes, so reseeding global state there
    silently derails the caller's randomness and makes repeated calls with
    the same seed return identical draws (the pdfgnn 2026-08-04 fidelity
    demoter: torch.manual_seed inside sample_from_t_distribution). Seeding
    the globals belongs to the training entry point in training.py, where
    the architecture validator REQUIRES it before the optimizer. The
    library-side remedies: a local generator threaded from the seed param
    (`g = torch.Generator(); g.manual_seed(seed)` / `np.random.default_rng(seed)`),
    or — for ops with no `generator=` argument, MC dropout being the canonical
    case — seeding inside a `torch.random.fork_rng()` context, which restores
    the caller's state on exit. Calls lexically inside such a `with` block are
    therefore exempt."""

    def _is_fork_rng_with(node: ast.AST) -> bool:
        if not isinstance(node, ast.With):
            return False
        for item in node.items:
            expr = item.context_expr
            if isinstance(expr, ast.Call):
                chain = ".".join(_attribute_chain(expr.func))
                if chain in ("torch.random.fork_rng", "torch.fork_rng"):
                    return True
        return False

    fork_spans: list[tuple[int, int]] = []
    for node in ast.walk(tree):
        if _is_fork_rng_with(node):
            fork_spans.append((node.lineno, node.end_lineno or node.lineno))

    hits: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        chain = ".".join(_attribute_chain(node.func))
        if chain in (
            "torch.manual_seed",
            "torch.cuda.manual_seed",
            "torch.cuda.manual_seed_all",
            "np.random.seed",
            "numpy.random.seed",
            "random.seed",
        ):
            if any(lo <= node.lineno <= hi for lo, hi in fork_spans):
                continue
            hits.append((node.lineno, chain))
    return hits


_CLAMP_CALLS = frozenset({"min", "max", "clip", "clamp", "minimum", "maximum"})
# Prefixes a coder adds to a paper's parameter name without changing which
# parameter it is (`num_epochs` in the spec, `max_epochs` in the signature).
_PARAM_NAME_FILLER = frozenset({"num", "max", "min", "n", "total", "the"})


def _paper_stated_param_tokens(spec: dict) -> dict[frozenset[str], str]:
    """Token-sets of every parameter the paper states a value for, mapped to
    the spec's own name for it.

    Three carriers, all of which mean "the paper pinned this": the glossary's
    `paper_value`, the scale-dependent lane's `paper_value`, and the training
    block's non-null fields. When ``comparison.evaluation_protocol`` binds a
    temporal parameter, its role-typed status replaces the two legacy carrier
    signals: paper-unspecified suppresses them, while paper-stated contributes
    only the typed fact. Filler words are dropped so the coder's `max_epochs`
    still resolves to the spec's `num_epochs`."""
    crit = spec.get("critical_requirements")
    if not isinstance(crit, dict):
        return {}

    comparison = spec.get("comparison")
    protocol = comparison.get("evaluation_protocol") \
        if isinstance(comparison, dict) else None
    quantities = protocol.get("quantities") \
        if isinstance(protocol, dict) else None
    temporal_roles = {
        "context_length", "forecast_call_horizon",
        "validation_span", "test_span",
    }
    protocol_bindings: dict[str, dict] = {}
    if isinstance(quantities, list):
        for raw in quantities:
            if not isinstance(raw, dict):
                continue
            name = raw.get("parameter_name")
            role = raw.get("role")
            if not isinstance(name, str) or not name or role not in temporal_roles:
                continue
            quantity = evaluation_protocol_quantity_for_param(spec, name)
            if isinstance(quantity, dict):
                protocol_bindings[name] = quantity

    names: list[str] = []
    exact_symbols: dict[str, str] = {}
    if not protocol_bindings:
        for key in ("param_glossary", "scale_dependent_hyperparameters"):
            for entry in crit.get(key) or []:
                if not isinstance(entry, dict) or entry.get("paper_value") is None:
                    continue
                names.append(str(entry.get("name") or ""))
                names.extend(str(a) for a in entry.get("aliases") or [])
        training = crit.get("training")
        if isinstance(training, dict):
            names.extend(k for k, v in training.items()
                         if v is not None and k != "paper_section")
        return _param_token_map(names)

    def _is_typed_paper_value(quantity: dict) -> bool:
        if quantity.get("paper_value_status") != "paper_stated":
            return False
        value = quantity.get("value")
        return (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and value > 0
        )

    # The exact parameter_name is the authority-bearing carrier. A glossary
    # alias may identify the same code parameter, but contributes no value or
    # role of its own.
    for quantity in protocol_bindings.values():
        if _is_typed_paper_value(quantity):
            symbols = {
                str(item) for item in (quantity.get("paper_symbols") or [])
                if isinstance(item, str)
            }
            exact_symbols.update({symbol: symbol for symbol in symbols})
            parameter_name = str(quantity.get("parameter_name") or "")
            if parameter_name:
                # The runtime carrier is the canonical diagnostic display
                # name when a prose identity normalizes to the same token set
                # (``forecast horizon`` vs ``forecast_horizon``).  Sets do
                # not get to make error messages nondeterministic.
                names.append(parameter_name)
            names.extend(sorted(
                label
                for label in evaluation_protocol_identity_labels(quantity)
                if label not in symbols and label != parameter_name
            ))
    protocol_labels: dict[str, set[str]] = {
        name: evaluation_protocol_identity_labels(quantity)
        for name, quantity in protocol_bindings.items()
    }
    for entry in crit.get("param_glossary") or []:
        if not isinstance(entry, dict):
            continue
        labels = [entry.get("name"), *list(entry.get("aliases") or [])]
        labels = [label for label in labels if isinstance(label, str)]
        matched = [name for name in protocol_bindings if name in labels]
        if matched:
            for name in matched:
                protocol_labels[name].update(labels)
            if (
                len(matched) == 1
                and _is_typed_paper_value(protocol_bindings[matched[0]])
            ):
                names.extend(labels)
            continue
        if entry.get("paper_value") is not None:
            names.extend(labels)

    def _lane_name(value: str) -> str:
        prefix = value.split("(", 1)[0]
        return re.sub(r"[^a-z0-9]+", "_", prefix.lower()).strip("_")

    normalized_protocol_labels = {
        name: {_lane_name(label) for label in labels}
        for name, labels in protocol_labels.items()
    }
    for entry in crit.get("scale_dependent_hyperparameters") or []:
        if not isinstance(entry, dict):
            continue
        name = entry.get("name")
        lane_name = _lane_name(name) if isinstance(name, str) else ""
        if lane_name and any(
            lane_name in labels for labels in normalized_protocol_labels.values()
        ):
            continue
        if entry.get("paper_value") is not None:
            names.append(str(name or ""))
            names.extend(str(a) for a in entry.get("aliases") or [])

    # Training fields are outside the legacy glossary/scale-lane authority
    # handoff and retain their existing behavior.
    training = crit.get("training")
    if isinstance(training, dict):
        names.extend(k for k, v in training.items()
                     if v is not None and k != "paper_section")
    token_map = _param_token_map(names)
    token_map.update({
        frozenset({f"@exact-symbol:{symbol}"}): display
        for symbol, display in exact_symbols.items()
    })
    return token_map


def _param_token_map(names: list[str]) -> dict[frozenset[str], str]:
    """Normalize spec parameter spellings for the clamp detector."""
    out: dict[frozenset[str], str] = {}
    for name in names:
        tokens = {t for t in re.split(r"[^a-z0-9]+", name.lower()) if t}
        core = frozenset(tokens - _PARAM_NAME_FILLER) or frozenset(tokens)
        if core:
            out.setdefault(core, name)
    return out


def _silently_clamped_paper_values(
    tree: ast.Module, stated: dict[frozenset[str], str],
) -> list[tuple[int, str, str, str]]:
    """Assignments that quietly compute a different value than the paper's.

    The failure class (`param_runtime_drift`): the paper states 50 epochs,
    params.json says 50, the delivered table says 50, and the runtime computes
    `max_epochs = min(50, max(10, T_total // 2))` = 10. Every provenance
    surface stays truthful about the DECLARED value while the demo honors
    another one, so a researcher reading the table cannot know what ran. The
    2026-08-05 pdfgnn review found two of these echoed into the result stats
    as if honored.

    Scoped to names the paper actually pins, because clamping an internal the
    paper never mentions is the coder's own business. Returns
    (line, assigned name, spec name, source snippet)."""
    if not stated:
        return []
    hits: list[tuple[int, str, str, str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name):
            continue
        tokens = {t for t in re.split(r"[^a-z0-9]+", target.id.lower()) if t}
        core = frozenset(tokens - _PARAM_NAME_FILLER) or frozenset(tokens)
        spec_name = stated.get(
            frozenset({f"@exact-symbol:{target.id}"})
        ) or stated.get(core)
        if spec_name is None:
            continue
        for inner in ast.walk(node.value):
            if not isinstance(inner, ast.Call):
                continue
            chain = _attribute_chain(inner.func)
            if chain and chain[-1] in _CLAMP_CALLS:
                hits.append((node.lineno, target.id, spec_name,
                             ast.unparse(node)[:160]))
                break
    return hits


def _bare_np_random_calls(tree: ast.Module) -> list[tuple[int, str]]:
    """Find calls to np.random.choice/shuffle/randint/etc. without rng= keyword.

    These are unseeded numpy globals — should use a seeded `rng = np.random.default_rng(seed)`
    instead.
    """
    suspect = {"choice", "shuffle", "randint", "rand", "randn", "permutation", "uniform", "normal", "integers"}
    hits: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            chain = _attribute_chain(node.func)
            if len(chain) >= 3 and chain[0] in ("np", "numpy") and chain[1] == "random" and chain[-1] in suspect:
                # Bare global numpy random call — flag.
                hits.append((node.lineno, ast.unparse(node)))
    return hits


def _suspect_torch_methods_on_numpy_names(tree: ast.Module) -> list[tuple[int, str]]:
    """Heuristic: flag calls to .unsqueeze(), .detach(), .numpy(), .cpu() on names
    that look numpy-shaped (e.g., variable name contains 'np' or '_np' suffix).

    This is genuinely best-effort — the receiver's actual type isn't always
    derivable. We flag obvious patterns; the reviewer agent catches the rest.
    """
    torch_only = {"unsqueeze", "detach", "numpy", "cpu"}
    hits: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Attribute) and func.attr in torch_only:
                # Receiver is the value (Attribute.value)
                recv = func.value
                if isinstance(recv, ast.Name):
                    name = recv.id
                    # Heuristic: name looks numpy-ish
                    if name.endswith("_np") or name.endswith("_numpy") or "ndarray" in name:
                        hits.append((node.lineno, f"`{name}.{func.attr}(...)` — `{name}` looks numpy but `{func.attr}` is torch-only"))
    return hits


def _constant_like(node: ast.AST) -> bool:
    """A numeric literal, signed literal, float('inf'), or np/torch `.inf`
    attribute — the saturation-branch value shapes."""
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) \
            and not isinstance(node.value, bool):
        return True
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        return _constant_like(node.operand)
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) \
            and node.func.id == "float" and node.args \
            and isinstance(node.args[0], ast.Constant) \
            and str(node.args[0].value).lower().lstrip("+-") == "inf":
        return True
    chain = _attribute_chain(node)
    return bool(chain) and chain[-1] == "inf"


def _names_in(node: ast.AST) -> set[str]:
    return {n.id for n in ast.walk(node) if isinstance(n, ast.Name)}


def _saturation_capped_rankings(tree: ast.Module) -> list[tuple[int, str]]:
    """R11 deterministic gate: a ranking must order candidates by the RAW
    distance ratio. Capping within-radius scores at a constant (the Eq-5
    probability clamp) ties every saturated candidate, and a tie-heavy
    ranking collapses to input order — the method silently degenerates to
    its upstream stage (finding class M-004). Both concrete cases were
    `np.where(dists <= R_0, <const>, ratio)` inside a rank-named function
    (the 2026-06-30 audited GBALD cap, selected-first; the 2026-07-02
    fresh-roll recurrence, deprioritized-last). Prompt rule R11 held on
    direction but decayed on the cap in a fresh roll, hence this gate.

    Flags, inside any top-level function whose name contains "rank":
      (a) a `where(...)` call whose condition is a comparison involving a
          function parameter (directly or through a mask variable) and
          whose branches include a constant-like value;
      (b) a mask-subscript assignment of a constant-like value where the
          mask derives from a comparison involving a function parameter.
    """
    hits: list[tuple[int, str]] = []
    for fn in tree.body:
        if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if "rank" not in fn.name.lower():
            continue
        arg_nodes = fn.args.posonlyargs + fn.args.args + fn.args.kwonlyargs
        params = {a.arg for a in arg_nodes}
        # Names bound to comparisons involving a parameter
        # (`within = dists <= R_0`) — the mask idiom.
        mask_names: set[str] = set()
        for node in ast.walk(fn):
            if isinstance(node, ast.Assign) \
                    and isinstance(node.value, ast.Compare) \
                    and _names_in(node.value) & params:
                mask_names.update(t.id for t in node.targets
                                  if isinstance(t, ast.Name))

        def _param_comparison(node: ast.AST) -> bool:
            if isinstance(node, ast.Compare) and _names_in(node) & params:
                return True
            return isinstance(node, ast.Name) and node.id in mask_names

        for node in ast.walk(fn):
            if isinstance(node, ast.Call):
                chain = _attribute_chain(node.func)
                if chain and chain[-1] == "where" and len(node.args) >= 2 \
                        and _param_comparison(node.args[0]) \
                        and any(_constant_like(a) for a in node.args[1:]):
                    hits.append((node.lineno,
                                 f"`{fn.name}`: {ast.unparse(node)[:100]}"))
            elif isinstance(node, ast.Assign) and _constant_like(node.value):
                for tgt in node.targets:
                    if isinstance(tgt, ast.Subscript) \
                            and _param_comparison(tgt.slice):
                        hits.append((node.lineno,
                                     f"`{fn.name}`: {ast.unparse(node)[:100]}"))
    return hits


def _list_literal_tolist_calls(tree: ast.Module) -> list[int]:
    """Find `[...].tolist()` patterns — Python lists don't have .tolist()."""
    hits: list[int] = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "tolist"
            and isinstance(node.func.value, ast.List)
        ):
            hits.append(node.lineno)
    return hits


def _range_name(node: ast.AST) -> str | None:
    """Return the first argument name from `range(name)`, if present."""
    if not isinstance(node, ast.Call):
        return None
    if not isinstance(node.func, ast.Name) or node.func.id != "range":
        return None
    if not node.args or not isinstance(node.args[0], ast.Name):
        return None
    return node.args[0].id


def _direct_append_targets(loop: ast.For) -> set[str]:
    """Collection names directly appended to by a loop body."""
    targets: set[str] = set()
    for stmt in loop.body:
        if not isinstance(stmt, ast.Expr) or not isinstance(stmt.value, ast.Call):
            continue
        call = stmt.value
        if not isinstance(call.func, ast.Attribute) or call.func.attr != "append":
            continue
        if isinstance(call.func.value, ast.Name):
            targets.add(call.func.value.id)
    return targets


def _remaining_quota_assignment(node: ast.Assign | ast.AnnAssign) -> tuple[str, str, str] | None:
    """Return `(remaining_name, quota_name, collection_name)` for `r = q - len(xs)`."""
    target: ast.AST | None = None
    value: ast.AST | None = None
    if isinstance(node, ast.Assign):
        if len(node.targets) != 1:
            return None
        target = node.targets[0]
        value = node.value
    elif isinstance(node, ast.AnnAssign):
        target = node.target
        value = node.value
    if not isinstance(target, ast.Name):
        return None
    if not (
        isinstance(value, ast.BinOp)
        and isinstance(value.op, ast.Sub)
        and isinstance(value.left, ast.Name)
        and isinstance(value.right, ast.Call)
        and isinstance(value.right.func, ast.Name)
        and value.right.func.id == "len"
        and len(value.right.args) == 1
        and isinstance(value.right.args[0], ast.Name)
    ):
        return None
    return target.id, value.left.id, value.right.args[0].id


def _dead_remaining_quota_loops(tree: ast.Module) -> list[tuple[int, str]]:
    """Find quota loops that are likely dead by construction.

    Pattern:
      for _ in range(quota):
          selected.append(...)
      remaining = quota - len(selected)
      for _ in range(remaining):
          ...

    If the first loop appends once per quota iteration, `remaining` is zero and
    the second phase never runs. This is algorithm-neutral and catches silent
    "implemented but unreachable" failures in multi-phase methods.
    """
    hits: list[tuple[int, str]] = []
    for func in [n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]:
        fills: list[tuple[str, str, int]] = []
        remaining_assignments: list[tuple[str, str, str, int]] = []
        remaining_loops: list[tuple[str, int]] = []

        for node in ast.walk(func):
            if isinstance(node, ast.For):
                quota_name = _range_name(node.iter)
                if quota_name:
                    for collection_name in _direct_append_targets(node):
                        fills.append((quota_name, collection_name, node.lineno))
                    remaining_loops.append((quota_name, node.lineno))
                continue
            if isinstance(node, (ast.Assign, ast.AnnAssign)):
                assignment = _remaining_quota_assignment(node)
                if assignment is not None:
                    remaining_name, quota_name, collection_name = assignment
                    remaining_assignments.append(
                        (remaining_name, quota_name, collection_name, node.lineno)
                    )

        for remaining_name, quota_name, collection_name, assign_line in remaining_assignments:
            matching_fills = [
                fill_line
                for fill_quota, fill_collection, fill_line in fills
                if fill_quota == quota_name
                and fill_collection == collection_name
                and fill_line < assign_line
            ]
            if not matching_fills:
                continue
            for loop_remaining_name, loop_line in remaining_loops:
                if loop_remaining_name == remaining_name and loop_line > assign_line:
                    hits.append((
                        loop_line,
                        f"function `{func.name}` computes `{remaining_name} = {quota_name} - "
                        f"len({collection_name})` after filling `{collection_name}` in "
                        f"`range({quota_name})`, then loops over `range({remaining_name})`. "
                        "That second phase is likely dead by construction when the first "
                        "phase appends once per quota iteration."
                    ))
    return hits


def _required_param_names(func: ast.FunctionDef | ast.AsyncFunctionDef) -> list[str]:
    args = func.args
    positional = [*args.posonlyargs, *args.args]
    required_positional_count = len(positional) - len(args.defaults)
    required = [arg.arg for arg in positional[:required_positional_count]]
    required.extend(
        arg.arg for arg, default in zip(args.kwonlyargs, args.kw_defaults)
        if default is None
    )
    return required


def _parse_signature_function(sig_str: str) -> ast.FunctionDef | ast.AsyncFunctionDef | None:
    try:
        tree = ast.parse(f"def {sig_str}:\n    pass\n")
    except (SyntaxError, ValueError):
        return None
    if not tree.body or not isinstance(tree.body[0], (ast.FunctionDef, ast.AsyncFunctionDef)):
        return None
    return tree.body[0]


def _pluggable_signature_contract_errors(
    *,
    pluggable_name: str,
    expected_sig: str,
    actual_fn: ast.FunctionDef | ast.AsyncFunctionDef,
) -> list[str]:
    """Return hard contract errors for call-shape drift that breaks automation.

    Formatting drift, annotations, and optional extras remain warnings via the
    existing normalized-signature check. Required parameter-name drift is a hard
    error because probes and notebooks call the pluggable by the spec names.
    """
    expected_fn = _parse_signature_function(expected_sig)
    if expected_fn is None:
        return []
    expected_params = _all_param_names(expected_fn)
    actual_params = _all_param_names(actual_fn)
    actual_required = _required_param_names(actual_fn)

    errors: list[str] = []
    missing = [name for name in expected_params if name not in actual_params]
    if missing:
        errors.append(
            f"method/method.py: `{pluggable_name}` is missing spec parameter(s) "
            f"{missing}; expected signature `{expected_sig}`. Parameter names are "
            "case-sensitive because probes call the pluggable by the spec contract."
        )

    extra_required = [name for name in actual_required if name not in expected_params]
    if extra_required:
        errors.append(
            f"method/method.py: `{pluggable_name}` declares extra required parameter(s) "
            f"{extra_required} not present in spec signature `{expected_sig}`. Extra "
            "required parameters make the pluggable unprobeable; make them optional "
            "with defaults or update the method spec."
        )
    return errors


def _slice_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    return None


def _unlabeled_subscript_index_names(node: ast.AST, *, unlabeled_names: set[str]) -> set[str]:
    """Names used as indices in expressions like `x_unlabeled[idx_name]`."""
    names: set[str] = set()
    for inner in ast.walk(node):
        if not isinstance(inner, ast.Subscript):
            continue
        if not isinstance(inner.value, ast.Name) or inner.value.id not in unlabeled_names:
            continue
        idx_name = _slice_name(inner.slice)
        if idx_name:
            names.add(idx_name)
    return names


def _assigned_name(stmt: ast.stmt) -> str | None:
    target: ast.AST | None = None
    if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1:
        target = stmt.targets[0]
    elif isinstance(stmt, ast.AnnAssign):
        target = stmt.target
    if isinstance(target, ast.Name):
        return target.id
    return None


def _return_references_name_or_dependent(func: ast.FunctionDef | ast.AsyncFunctionDef, name: str) -> bool:
    """One-pass dependency check: does any returned expression include `name`?

    Also accepts `out = name + other; return out` style forwarding. The goal is
    not complete data-flow; it is enough to distinguish hidden acquisition state
    from indices that flow back to the notebook contract.
    """
    dependent_names = {name}
    for stmt in ast.walk(func):
        if isinstance(stmt, (ast.Assign, ast.AnnAssign)):
            target = _assigned_name(stmt)
            value = stmt.value if isinstance(stmt, (ast.Assign, ast.AnnAssign)) else None
            if not target or value is None:
                continue
            if any(_expr_explicitly_forwards_name(value, dep) for dep in dependent_names):
                dependent_names.add(target)
        elif isinstance(stmt, ast.Return) and stmt.value is not None:
            if any(_expr_explicitly_forwards_name(stmt.value, dep) for dep in dependent_names):
                return True
    return False


def _expr_explicitly_forwards_name(node: ast.AST, name: str) -> bool:
    """True when `node` is visibly constructing/returning index values from name."""
    if isinstance(node, ast.Name):
        return node.id == name
    if isinstance(node, ast.Starred):
        return _expr_explicitly_forwards_name(node.value, name)
    if isinstance(node, ast.Subscript):
        return _expr_explicitly_forwards_name(node.value, name)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        return (
            _expr_explicitly_forwards_name(node.left, name)
            or _expr_explicitly_forwards_name(node.right, name)
        )
    if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
        return any(_expr_explicitly_forwards_name(elt, name) for elt in node.elts)
    if isinstance(node, ast.ListComp):
        return any(_expr_explicitly_forwards_name(gen.iter, name) for gen in node.generators)
    if isinstance(node, ast.Call):
        return any(_expr_explicitly_forwards_name(arg, name) for arg in node.args)
    return False


def _hidden_unreturned_pluggable_acquisitions(
    func: ast.FunctionDef | ast.AsyncFunctionDef,
) -> list[tuple[int, str]]:
    """Detect AL pluggable functions that hide extra unlabeled acquisitions.

    The active-learning notebook owns `labeled_idx`/`unlabeled_idx`. A
    pluggable acquisition function may score candidates, but if it augments
    `x_labeled` from `x_unlabeled[some_indices]`, those positions must be part
    of the return value. Otherwise the function has silently used newly
    "labeled" examples that the notebook never adds to the labeled set.
    """
    arg_names = {arg.arg for arg in func.args.args + func.args.kwonlyargs}
    unlabeled_names = {name for name in arg_names if "unlabeled" in name.lower()}
    labeled_names = {
        name
        for name in arg_names
        if "labeled" in name.lower() and "unlabeled" not in name.lower()
    }
    if not unlabeled_names or not labeled_names:
        return []

    hits: list[tuple[int, str]] = []
    seen: set[tuple[int, str]] = set()
    for stmt in ast.walk(func):
        if not isinstance(stmt, (ast.Assign, ast.AnnAssign)):
            continue
        target_name = _assigned_name(stmt)
        if target_name not in labeled_names:
            continue
        value = stmt.value
        if value is None:
            continue
        for idx_name in sorted(_unlabeled_subscript_index_names(value, unlabeled_names=unlabeled_names)):
            if _return_references_name_or_dependent(func, idx_name):
                continue
            key = (getattr(stmt, "lineno", func.lineno), idx_name)
            if key in seen:
                continue
            seen.add(key)
            hits.append((
                getattr(stmt, "lineno", func.lineno),
                f"`{func.name}` augments `{target_name}` with unlabeled examples indexed by "
                f"`{idx_name}`, but `{idx_name}` does not flow into the returned indices"
            ))
    return hits


def _active_learning_selector_static_errors(
    func: ast.FunctionDef | ast.AsyncFunctionDef,
) -> list[str]:
    """Hard active-learning selector contract checks visible from AST alone."""
    params = set(_all_param_names(func))
    errors: list[str] = []

    output_count_aliases = sorted(params & {"batch_output", "batch_outputs"})
    if output_count_aliases:
        errors.append(
            f"method/method.py:{func.lineno}: `{func.name}` declares "
            f"legacy output-count parameter(s) {output_count_aliases}. "
            "Active-learning selectors have one output-count contract: the "
            "`batch_size` argument. Use `batch_returns` only for a larger candidate "
            "prefilter, and return exactly `batch_size` unique in-range positions."
        )

    if "batch_size" in params and not _name_referenced_in_body(func, "batch_size"):
        errors.append(
            f"method/method.py:{func.lineno}: `{func.name}` declares `batch_size` "
            "but never references it. Active-learning `select_batch` must return "
            "exactly `batch_size` unique in-range positions; hard-coded counts or "
            "separate output-size parameters are not allowed."
        )
    return errors


def _active_learning_selector_behavioral_errors(
    *,
    run_dir: Path,
    pluggable_name: str,
    spec: dict | None = None,
) -> tuple[list[str], list[str]]:
    """Run the selector-level AL-5 contract before notebook generation when possible.

    Stage 5 still runs the full battery. This early arm exists so interface
    contract bugs get routed back through method-coder fix-mode at Stage 2.c.
    Harness limitations stay warnings here; demonstrated contract failures are
    hard errors.

    The tier upgrade (R2C-047, maintainer decision 2026-08-05): a `flag_for_researcher` on a
    mechanism the run's OWN methodology contract marks must-replicate with no
    approved approximation is not researcher homework. The contract and the
    observation cannot both be right, so the flag becomes a hard error and
    enters the fix loop that is already running at this stage, where repair is
    cheap. A flag the contract does not contradict keeps its warning, which is
    the recorded neutrality decision holding exactly where it belongs.
    """
    method_path = run_dir / "method" / "method.py"
    errors: list[str] = []
    warnings: list[str] = []
    try:
        from probes.al_loop import probe_acquisition_contract  # noqa: PLC0415
        from probes.package_loader import ProbeLoadError, load_module_from_path  # noqa: PLC0415

        module = load_module_from_path(method_path)
        verdict = probe_acquisition_contract(module, pluggable_name)
    except ProbeLoadError as e:
        warnings.append(
            f"early AL-5 acquisition-contract probe could not load method.py: {e}. "
            "Stage 2.d import validation and the Stage 5 probe battery remain the "
            "runtime authorities."
        )
        return errors, warnings
    except Exception as e:  # noqa: BLE001 — generated code/probe deps can fail arbitrarily
        warnings.append(
            f"early AL-5 acquisition-contract probe could not run: "
            f"{type(e).__name__}: {e}. Stage 5 will retry in the full battery."
        )
        return errors, warnings

    _bind_verdict_elements(run_dir, verdict)
    join = _spec_join(verdict, spec)
    contract_says = _spec_disclosure(join)

    if verdict.verdict == "fail":
        errors.append(
            f"method/method.py: early AL-5 acquisition contract failed for "
            f"`{pluggable_name}`: {verdict.message}. Active-learning selectors "
            "must return exactly the requested `batch_size` positions before the "
            "notebook route is generated." + contract_says
        )
    elif join is not None and join.producer_fixable:
        errors.append(
            f"method/method.py: the early AL-5 acquisition-contract probe "
            f"flagged `{pluggable_name}`: {verdict.message}." + contract_says
            + " Fix the mechanism here rather than deferring it to the "
            "researcher."
        )
    elif verdict.verdict in {"flag_for_researcher", "unprobeable"}:
        warnings.append(
            f"early AL-5 acquisition-contract probe returned {verdict.verdict}: "
            f"{verdict.message}" + contract_says
        )
    return errors, warnings


def _forecasting_behavioral_errors(
    *,
    run_dir: Path,
    spec: dict,
) -> tuple[list[str], list[str]]:
    """Run the cheap TSF sample/path arms at the producing seam.

    Both rows come from the same family wrapper the delivery battery uses, so
    their exact ``probe_ref``, callable anchors, and R2C-047 join cannot drift
    across enforcement points. A demonstrated supported disagreement belongs
    in the method-coder retry already in progress. Unsupported schema/output/
    relational grammar is pipeline coverage and stays a warning without
    consuming that retry.
    """

    errors: list[str] = []
    warnings: list[str] = []
    refs = {
        "time_series_forecasting.sample_genuineness",
        "time_series_forecasting.path_dependence",
    }
    try:
        from probes.time_series_forecasting import (  # noqa: PLC0415
            run_time_series_forecasting_probes,
        )

        verdicts = run_time_series_forecasting_probes(
            run_dir,
            enabled_refs=refs,
        )
    except Exception as exc:  # noqa: BLE001 - probe coverage cannot kill gate
        warnings.append(
            "early forecasting probes could not run because the pipeline "
            f"adapter raised {type(exc).__name__}: {exc}. Stage 5 will retry "
            "the same exact refs."
        )
        return errors, warnings

    for verdict in verdicts:
        _bind_verdict_elements(run_dir, verdict)
        join = _spec_join(verdict, spec)
        contract_says = _spec_disclosure(join)
        label = f"early {verdict.probe_id} ({verdict.probe_ref})"
        if verdict.verdict == "fail":
            errors.append(
                f"method package: {label} failed: {verdict.message}."
                + contract_says
                + " Repair the supported forecast contract/code "
                "disagreement at the method producer seam."
            )
        elif join is not None and join.producer_fixable:
            errors.append(
                f"method package: {label} returned {verdict.verdict}: "
                f"{verdict.message}." + contract_says
                + " The run's exact methodology obligation requires this "
                "mechanism, so repair it before notebook generation."
            )
        elif verdict.verdict in {"flag_for_researcher", "unprobeable"}:
            warnings.append(
                f"{label} returned {verdict.verdict}: {verdict.message}."
                + contract_says
            )
    return errors, warnings


def _bind_verdict_elements(run_dir: Path, verdict: object) -> None:
    """Resolve the probe's declared callables to paper-map element ids
    (R2C-072), the same way the stage 5 battery does.

    Both enforcement points must reach the same binding, or a finding routed
    here would be adjudicated differently from the identical finding at
    delivery. Fails open: an unresolvable binding leaves the verdict unbound,
    which is the neutral branch.
    """
    declared = list(getattr(verdict, "bound_callables", None) or [])
    if not declared:
        return
    try:
        from paper_element_anchors import (  # noqa: PLC0415
            element_ids_for_callables,
        )

        resolved = element_ids_for_callables(run_dir / "method", declared)
    except Exception:  # noqa: BLE001 — binding must never break validation
        return
    existing = list(getattr(verdict, "element_ids", None) or [])
    verdict.element_ids = existing + [i for i in resolved if i not in existing]


def _spec_join(verdict: object, spec: dict | None):
    """Join one probe verdict to the run's methodology contract, or None.

    Fails open on any import or shape problem: this validator's existing checks
    must keep running when the join cannot be computed, and a missing join only
    ever means today's behavior (the flag stays a warning)."""
    if spec is None:
        return None
    try:
        from probe_spec_join import join_verdict  # noqa: PLC0415

        return join_verdict(verdict, spec)
    except Exception:  # noqa: BLE001 — the join must never break validation
        return None


def _spec_disclosure(join) -> str:
    """The contract's own sentence about this finding, or nothing when it is
    silent. Leading space so callers can concatenate it onto a message."""
    if join is None:
        return ""
    try:
        from probe_spec_join import disclosure, unbound_gap_note  # noqa: PLC0415

        text = disclosure(join) or unbound_gap_note(join) or ""
    except Exception:  # noqa: BLE001
        return ""
    return f" {text}" if text else ""


_INVERSE_DISTANCE_RE = re.compile(
    r"(?is)(?:\b1\s*/\s*(?:min_)?(?:dist|distance)|\bR\s*[_-]?\s*0\s*/[^.\n]{0,120}(?:dist|distance)|/[^.\n]{0,120}(?:dist|distance))"
)
_DESCENDING_RANK_RE = re.compile(
    r"(?is)(?:"
    r"argsort\([^)]*\)[^\n]{0,120}\[::-1\]"
    r"|argsort\([^)]*descending\s*=\s*True"
    r"|sort(?:ed)?\([^)]*reverse\s*=\s*True"
    r"|\.sort\([^)]*reverse\s*=\s*True"
    r"|topk\([^)]*largest\s*=\s*True"
    r")"
)
_FARTHER_BETTER_RE = re.compile(
    r"(?is)(?:"
    r"\bfarther\b[^.\n]{0,120}\b(?:higher|larger|greater|more\s+representative|higher\s+score|selected|preferred|favored|better)\b"
    r"|"
    r"\b(?:higher|larger|greater)\s+distances?\b"
    r"|"
    r"\b(?:higher|larger|greater)\b[^.\n]{0,120}\b(?:farther|more\s+distant)\b"
    r"|"
    r"\bmore\s+distant\b[^.\n]{0,120}\b(?:higher|larger|greater|representative|selected|preferred|favored|better)\b"
    r"|"
    r"\bfar\s+from\b[^.\n]{0,160}\b(?:high|higher|larger|greater|more\s+representative|higher\s+score|selected|preferred|favored|better)\b"
    r"|"
    r"\b(?:high|higher|larger|greater)\b[^.\n]{0,160}\bfar\s+from\b"
    r")"
)


def _inverse_distance_descending_prose_contradictions(
    tree: ast.Module, source: str
) -> list[tuple[int, str]]:
    """Find functions whose inverse-distance descending rank contradicts prose.

    For a score like `R_0 / distance`, sorting scores descending gives higher
    rank to smaller distances. If the same function's comments/docstring say
    farther or more distant samples get the higher score/preference, the
    researcher-facing explanation is mechanically inconsistent with the code.
    """
    lines = source.splitlines()
    hits: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        end_lineno = getattr(node, "end_lineno", node.lineno)
        segment = "\n".join(lines[node.lineno - 1:end_lineno])
        if (
            _INVERSE_DISTANCE_RE.search(segment)
            and _DESCENDING_RANK_RE.search(segment)
            and _FARTHER_BETTER_RE.search(segment)
        ):
            hits.append((
                node.lineno,
                f"function `{node.name}` ranks an inverse-distance score in descending order "
                "but its prose claims farther/more distant samples receive the higher score "
                "or preference"
            ))
    return hits


def _slice_contains_newaxis(node: ast.AST) -> bool:
    if isinstance(node, ast.Constant) and node.value is None:
        return True
    if isinstance(node, ast.Name) and node.id == "newaxis":
        return True
    if isinstance(node, ast.Attribute) and node.attr == "newaxis":
        return True
    if isinstance(node, ast.Tuple):
        return any(_slice_contains_newaxis(elt) for elt in node.elts)
    return False


def _expr_has_newaxis_subscript(node: ast.AST) -> bool:
    return any(
        isinstance(inner, ast.Subscript) and _slice_contains_newaxis(inner.slice)
        for inner in ast.walk(node)
    )


def _mentions_kmeanspp(func: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    haystack = f"{func.name}\n{ast.get_docstring(func) or ''}".lower()
    return (
        "kmeans" in haystack
        or "k_means" in haystack
        or "k-means" in haystack
    )


def _quadratic_kmeanspp_distance_broadcasts(
    tree: ast.Module,
) -> list[tuple[int, str]]:
    """Find k-means++ helpers that materialize all center distances at once.

    Full `(N, t, D)` broadcasts are easy to generate and pass small demos, but
    they blow up at active-learning smoke scale when `t=batch_size` and
    `D=n_classes*hidden_dim`. K-means++ only needs a running `(N,)` nearest
    squared-distance vector updated one center at a time.
    """
    hits: list[tuple[int, str]] = []
    for func in ast.walk(tree):
        if not isinstance(func, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if not _mentions_kmeanspp(func):
            continue
        for node in ast.walk(func):
            if not (
                isinstance(node, ast.BinOp)
                and isinstance(node.op, ast.Sub)
                and (
                    _expr_has_newaxis_subscript(node.left)
                    or _expr_has_newaxis_subscript(node.right)
                )
            ):
                continue
            snippet = ast.unparse(node)
            hits.append((
                getattr(node, "lineno", func.lineno),
                f"function `{func.name}` uses a newaxis/None broadcast in "
                f"k-means++ distance computation (`{snippet}`)"
            ))
    return hits


def _sibling_module_names(package_dir: Path) -> set[str]:
    """Importable names defined inside the generated package: module stems
    (model, training, data) and subpackage directory names (tests)."""
    names: set[str] = set()
    if not package_dir.is_dir():
        return names
    for entry in package_dir.iterdir():
        if entry.is_file() and entry.suffix == ".py" and entry.stem != "__init__":
            names.add(entry.stem)
        elif entry.is_dir() and (entry / "__init__.py").is_file():
            names.add(entry.name)
    return names


def _bare_sibling_imports(
    tree: ast.AST, sibling_names: set[str], package_name: str = "method"
) -> list[tuple[int, str]]:
    """Absolute imports that name a sibling module of the generated package.

    The package is consumed as `method.<module>` by both the notebook process
    and the 2.d import smoke test; a bare `from model import ...` resolves
    only with method/ itself on sys.path, which neither consumer has
    (DomIndOnto 0728c resume: the defect was written at this seam and only
    surfaced at finalization, where the sole outcome is a halt). Relative
    imports and `method.`-qualified absolute imports are the sanctioned forms
    and pass untouched.

    `package_name` is load-bearing, not decoration (R2C-079). The generated
    package always contains `method/method.py`, because that is where the
    spec's pluggable lives, so the stem `method` lands in `sibling_names`
    beside `model` and `training`. Without this guard the sanctioned
    `from method.model import X` was reported as a bare sibling import, since
    its root is a real module stem — the check rejected the exact form its own
    docstring blesses. Observed live on the 2026-08-07 forecasting roll, where
    it cost a fix-loop iteration."""
    if not sibling_names:
        return []
    hits: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if node.level:
                continue
            root = (node.module or "").split(".")[0]
            if root == package_name:
                continue  # the sanctioned package-qualified absolute form
            if root in sibling_names:
                names = ", ".join(alias.name for alias in node.names)
                hits.append((
                    node.lineno,
                    f"bare sibling import `from {node.module} import {names}` — "
                    f"`{root}` is a module of the generated `{package_name}` "
                    f"package, and the bare form only resolves with "
                    f"{package_name}/ itself on sys.path (neither the notebook "
                    f"nor the finalization import test runs that way). Use the "
                    f"package-relative form: "
                    f"`from .{node.module} import {names}`.",
                ))
        elif isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                if root == package_name:
                    continue  # `import method.model` resolves for both consumers
                if root in sibling_names:
                    hits.append((
                        node.lineno,
                        f"bare sibling import `import {alias.name}` — `{root}` is "
                        f"a module of the generated `{package_name}` package, and "
                        f"the bare form only resolves with {package_name}/ itself "
                        f"on sys.path. Use the package-relative form "
                        f"(`from . import {root}` or "
                        f"`from .{root} import <names>`).",
                    ))
    return sorted(hits)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def _assigned_name_targets(target: ast.AST):
    """Yield the names an assignment TARGET binds — Store-ctx `ast.Name`
    nodes only, recursing through tuple/list/starred unpacking. Attribute
    and subscript targets mutate an existing object and define nothing."""
    if isinstance(target, ast.Name) and isinstance(target.ctx, ast.Store):
        yield target.id
    elif isinstance(target, (ast.Tuple, ast.List)):
        for elt in target.elts:
            yield from _assigned_name_targets(elt)
    elif isinstance(target, ast.Starred):
        yield from _assigned_name_targets(target.value)


def duplicate_public_definitions(
    module_trees: dict[str, ast.Module],
) -> dict[str, list[str]]:
    """Public CLASSES and FUNCTIONS defined in more than one `method/` module.

    R2C-053; moved here from the architecture-coder validator (B-04) — this
    validator is its sole production caller. The ownership-collision check
    fails loud on a duplicate definition, but only for symbols the spec
    declares as bridge symbols. A public type neither file's manifest
    declares falls straight through it, and the finalizer then picks one
    definition for `__init__.py` by import order.

    That is how the pdfgnn delivery shipped `ProbabilisticForecast` twice:
    model.py's version is exported in `__all__`, method.py's version is what
    `forecast()` actually returns, and a researcher checking
    `isinstance(result, ProbabilisticForecast)` against the exported name gets
    False. The pack's own open_questions had predicted exactly this.

    Restricted to classes and functions on purpose. Module-level ASSIGNMENTS
    are excluded: two modules independently binding a constant such as
    `DEFAULT_DATA_DIR` is ordinary and carries no identity semantics — and a
    later top-level assignment REBINDING a def/class name masks it (later
    bindings win, matching Python runtime semantics)."""
    sites: dict[str, list[str]] = {}
    for rel, tree in module_trees.items():
        kinds: dict[str, str] = {}
        for node in tree.body:
            if isinstance(node, ast.ClassDef):
                kinds[node.name] = "class"
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                kinds[node.name] = "function"
            elif isinstance(node, ast.Assign):
                for target in node.targets:
                    for name in _assigned_name_targets(target):
                        kinds[name] = "assignment"
            elif isinstance(node, ast.AnnAssign):
                for name in _assigned_name_targets(node.target):
                    kinds[name] = "assignment"
        for name, kind in kinds.items():
            if kind in ("class", "function") and not name.startswith("_"):
                sites.setdefault(name, []).append(rel)
    return {
        name: sorted(rels)
        for name, rels in sorted(sites.items())
        if len(rels) > 1
    }


def validate(spec: dict, run_dir: Path, repo_root: Path) -> tuple[list[str], list[str]]:
    """Return (errors, warnings). Errors block the pipeline; warnings inform the reviewer."""
    errors: list[str] = []
    warnings: list[str] = []

    build_plan = load_build_plan(
        spec, repo_root, provisional_packs_dir=run_overlay_dir(run_dir))
    if build_plan is None:
        paradigm_id = (spec.get("comparison") or {}).get("classification", {}).get("id")
        # (errors, warnings) tuple — a bare list here crashed main()'s unpack.
        return [f"no taxonomy build plan for comparison.classification.id={paradigm_id!r}"], []
    contract = ((build_plan.get("pluggable_component") or {}).get("contract") or {})
    forbidden_params = set(contract.get("forbidden_param_names") or [])

    method_path = run_dir / "method/method.py"

    # 1. File presence + parse
    if not method_path.is_file():
        errors.append(f"method/method.py missing at {method_path}")
        return errors, warnings

    try:
        method_src = method_path.read_text(encoding="utf-8")
        tree = ast.parse(method_src, filename=str(method_path))
    except SyntaxError as e:
        errors.append(f"method/method.py fails to parse as Python: {e}")
        return errors, warnings
    # compile() catches placement rules ast.parse accepts (misplaced
    # `from __future__`, the SRL 2026-07-05 duplicate-header class) so the
    # defect surfaces here, where the fix loop can route it, not at 2d.
    try:
        compile(method_src, str(method_path), "exec")
    except SyntaxError as e:
        errors.append(f"method/method.py fails to compile as Python: {e}")
        return errors, warnings

    # 2. Pluggable function contract
    pc = (spec.get("comparison") or {}).get("pluggable_component") or {}
    pluggable_name = pc.get("name")
    expected_sig = pc.get("signature")
    seed_param = normalized_seed_param(pc.get("seed_param"))

    if not pluggable_name:
        errors.append("spec.comparison.pluggable_component.name missing")
        return errors, warnings

    pluggable_fn = _find_function_def(tree, pluggable_name)
    if pluggable_fn is None:
        errors.append(f"method/method.py: pluggable function `{pluggable_name}` not defined at top level")
        return errors, warnings

    # Stylistic signature drift is a warning, but required parameter-name drift
    # is an error: probes call the pluggable by the spec contract.
    if expected_sig:
        actual_sig = _signature_string(pluggable_fn)
        errors.extend(
            _pluggable_signature_contract_errors(
                pluggable_name=pluggable_name,
                expected_sig=expected_sig,
                actual_fn=pluggable_fn,
            )
        )
        if _normalize_sig(actual_sig) != _normalize_sig(expected_sig):
            warnings.append(
                f"`{pluggable_name}` signature drifts from spec — expected `{expected_sig}`; "
                f"got `{actual_sig}`. May be stylistic (type annotations, kw-only markers) "
                f"or genuine (extra required params). Reviewer to judge."
            )

    # Seed param present + referenced
    if seed_param:
        param_names = _all_param_names(pluggable_fn)
        if seed_param not in param_names:
            errors.append(
                f"method/method.py: `{pluggable_name}` is missing required `{seed_param}` parameter"
            )
        elif not _name_referenced_in_body(pluggable_fn, seed_param):
            errors.append(
                f"method/method.py: `{pluggable_name}` declares `{seed_param}` but does not "
                f"reference it in its body — every stochastic op MUST seed from this parameter (C-22)."
            )

    # 3. Forbidden params anywhere in the file's top-level functions
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            params = set(_all_param_names(node))
            hits = sorted(params & forbidden_params)
            if hits:
                errors.append(
                    f"method/method.py: function `{node.name}` declares forbidden parameter(s) {hits}: "
                    f"forbidden_param_names is {sorted(forbidden_params)}. Use named keyword params with defaults."
                )

    # 3b. Package-import convention (R2C-036). Catch the bare sibling import
    # at the producing seam, where the coder's existing fix loop repairs it;
    # the 2.d subprocess import test stays the backstop.
    for lineno, msg in _bare_sibling_imports(
        tree, _sibling_module_names(method_path.parent),
        method_path.parent.name,
    ):
        errors.append(f"method/method.py:{lineno}: {msg}")

    # 3c. Duplicate public definitions across the package (R2C-053). Checked
    # HERE rather than at 2.b because the collision that matters spans
    # producers: at 2.b only model.py and training.py exist, and the pdfgnn
    # case was model.py versus method.py.
    package_trees: dict[str, ast.Module] = {}
    for py_path in sorted((run_dir / "method").glob("*.py")):
        if py_path.name == "__init__.py":
            continue
        try:
            package_trees[f"method/{py_path.name}"] = ast.parse(
                py_path.read_text(encoding="utf-8"), filename=str(py_path)
            )
        except (SyntaxError, OSError):
            # A sibling that does not parse is its own producer's failure,
            # reported at that producer's gate. Skip it rather than masking
            # this file's findings behind someone else's syntax error.
            continue
    for name, sites in duplicate_public_definitions(package_trees).items():
        errors.append(
            f"method/ package: the public symbol `{name}` is defined in "
            f"{len(sites)} files ({', '.join(sites)}) — ambiguous ownership. "
            f"__init__.py re-exports exactly one of them, so a caller "
            f"comparing against the exported name can silently disagree with "
            f"what the other module returns (an isinstance check across the "
            f"boundary is False). Define it in ONE module and import it in "
            f"the others."
        )

    # 4. Stateless rules
    hasattr_hits = _has_hasattr_model_call(tree)
    for line, snippet in hasattr_hits:
        errors.append(
            f"method/method.py:{line}: `hasattr(model, ...)` call found ({snippet!r}). "
            f"The pluggable function is re-invoked fresh by the notebook driver each call; "
            f"`hasattr`-gated branches tend to encode hidden per-call state or a one-time-init "
            f"assumption that doesn't hold across calls. If the algorithm has a one-time init "
            f"step, expose it as a separate top-level function the notebook calls once at bootstrap."
        )

    state_hits = _module_level_names_assigned_in_functions(tree)
    for line, msg in state_hits:
        errors.append(
            f"method/method.py:{line}: {msg}. Module-level mutable state breaks the Step 6c "
            f"run-consistency check (which spawns a fresh process)."
        )

    paradigm_id = ((spec.get("comparison") or {}).get("classification") or {}).get("id", "")
    if paradigm_id.startswith("active_learning"):
        errors.extend(_active_learning_selector_static_errors(pluggable_fn))
        hidden_acquisition_hits = _hidden_unreturned_pluggable_acquisitions(pluggable_fn)
        for line, msg in hidden_acquisition_hits:
            errors.append(
                f"method/method.py:{line}: {msg}. Active-learning `select_batch` must be "
                "a pure acquisition scorer from the notebook's point of view: it returns "
                "positions into the x_unlabeled tensor it was passed. If a method has a "
                "bootstrap/core-set initialization step, expose it as a separate helper "
                "that the notebook calls once and records in labeled_idx. Do not add "
                "unlabeled examples to x_labeled inside select_batch unless those exact "
                "positions are included in the returned list."
            )
        al5_errors, al5_warnings = _active_learning_selector_behavioral_errors(
            run_dir=run_dir,
            pluggable_name=pluggable_name,
            spec=spec,
        )
        errors.extend(al5_errors)
        warnings.extend(al5_warnings)
    if build_plan.get("plan_key") == "time_series_forecasting":
        tsf_errors, tsf_warnings = _forecasting_behavioral_errors(
            run_dir=run_dir,
            spec=spec,
        )
        errors.extend(tsf_errors)
        warnings.extend(tsf_warnings)

    # 5. Seed plumbing — unseeded stochastic calls in the pluggable function
    if seed_param and pluggable_fn is not None and _name_referenced_in_body(pluggable_fn, seed_param):
        # Only flag stochastic calls if seed IS referenced (otherwise the missing-seed-ref
        # error already covers it). We trust developers to thread seed correctly when it's
        # referenced; rely on the shared signature_ast check for the AST-level
        # unseeded-call detection.
        pass
    elif seed_param and pluggable_fn is not None:
        stochastic = _find_stochastic_calls(pluggable_fn)
        if stochastic:
            errors.append(
                f"method/method.py: `{pluggable_name}` declares `{seed_param}` but does not "
                f"reference it in its body, AND has {len(stochastic)} stochastic call(s) — "
                f"these are unseeded and break reproducibility."
            )

    # 6. Bare `np.random.default_rng()` (no seed)
    bare_rng_lines = _bare_default_rng_calls(tree)
    for line in bare_rng_lines:
        errors.append(
            f"method/method.py:{line}: `np.random.default_rng()` called with no arguments. "
            f"Pass a seed: `np.random.default_rng(seed)`."
        )

    # 7. Bare numpy random globals
    bare_np_hits = _bare_np_random_calls(tree)
    for line, snippet in bare_np_hits:
        errors.append(
            f"method/method.py:{line}: bare numpy global random call `{snippet}` — uses unseeded "
            f"global state. Use `rng = np.random.default_rng(seed)` and call `rng.<...>` instead."
        )

    # 7b. Global RNG mutation in the library module (rng_threading). Seeding
    # the process globals belongs to training.py's entry point (the arch
    # validator requires it there); a method.py function that reseeds them
    # per call derails the caller's randomness and pins repeated same-seed
    # calls to identical draws.
    for line, chain in _global_seed_mutation_calls(tree):
        errors.append(
            f"method/method.py:{line}: `{chain}(...)` mutates process-global "
            f"RNG state inside the library module. Thread the seed through a "
            f"LOCAL generator instead (`g = torch.Generator(); "
            f"g.manual_seed(seed)` for torch sampling ops, or "
            f"`rng = np.random.default_rng(seed)` for numpy). For ops with "
            f"no generator argument (e.g. MC dropout), seed inside "
            f"`with torch.random.fork_rng():` so the caller's state is "
            f"restored on exit. Bare global seeding belongs only to the "
            f"training entry point in training.py (rng_threading)."
        )

    # 7c. Silent clamping of a paper-stated value (param_runtime_drift,
    # R2C-065 piece 4). The declared value must be what runs, or the code
    # must refuse loudly with the arithmetic — never quietly honor another
    # number while every provenance surface reports the paper's.
    for line, name, spec_name, snippet in _silently_clamped_paper_values(
            tree, _paper_stated_param_tokens(spec)):
        errors.append(
            f"method/method.py:{line}: `{snippet}` silently computes a "
            f"different runtime value for `{name}`, which the paper states "
            f"as `{spec_name}`. Every provenance surface (params.json, the "
            f"delivered parameter table, the report) will claim the paper's "
            f"value while the demo honors this one. Either use the declared "
            f"value, or validate the input up front and RAISE with the "
            f"arithmetic in the message (e.g. "
            f"\"needs >= {{needed}} time steps for {name}={{declared}}, got "
            f"{{actual}}\"). A silent clamp is the one shape that is not "
            f"allowed (param_runtime_drift)."
        )

    # 7d. Evaluation-split integrity (eval_split_aliasing, R2C-066): the
    # reported numbers must not be computed on data the model trained on.
    from scripts.eval_split_integrity import find_split_aliasing  # noqa: PLC0415

    for finding in find_split_aliasing(tree):
        errors.append(finding.message("method/method.py"))

    # 7d.2. Reachable temporal target ranges (R2C-077): enforce concrete
    # method-coder-owned fitting, selection, and reported-evaluation ranges at
    # this producer seam. Caller-bound relationships remain for Stage 3a.
    from scripts.eval_split_validation import (  # noqa: PLC0415
        method_split_lineage_errors,
    )

    errors.extend(method_split_lineage_errors(
        spec, build_plan, run_dir
    ))

    # 7e. Aliased return tuple (aliased_return_tuple): distinct promised
    # outputs must be distinct objects. The night3 decode returned
    # `(mu, sigma, nu, mu)`, so `predicted_means` WAS `mu`, the sampled
    # path was discarded, and the element test asserting their equality
    # became tautological.
    from scripts.api_surface_checks import find_aliased_return_tuples  # noqa: PLC0415

    for line, name in find_aliased_return_tuples(tree):
        errors.append(
            f"method/method.py:{line}: the return tuple carries `{name}` "
            f"twice — two promised outputs are the SAME object, so one of "
            f"them is a mislabeled alias and whatever it was supposed to "
            f"carry is silently dropped. Return the distinct quantity in "
            f"each slot, or collapse the API to one name if they are "
            f"genuinely identical (aliased_return_tuple)."
        )

    # 8. Library-boundary heuristics
    suspect_torch = _suspect_torch_methods_on_numpy_names(tree)
    for line, msg in suspect_torch:
        errors.append(f"method/method.py:{line}: {msg}")

    list_tolist = _list_literal_tolist_calls(tree)
    for line in list_tolist:
        errors.append(
            f"method/method.py:{line}: list-literal `.tolist()` — Python lists don't have .tolist(). "
            f"The list is already a list."
        )

    for line, snippet in _saturation_capped_rankings(tree):
        errors.append(
            f"method/method.py:{line}: ranking saturation cap — {snippet}. "
            f"A ranking must order candidates by the RAW distance ratio: "
            f"capping within-radius scores at a constant ties the saturated "
            f"candidates, and a tie-heavy ranking collapses to input order, "
            f"so the method silently degenerates to its upstream stage "
            f"(finding class M-004). Apply the Eq-5-style cap ONLY where an "
            f"actual probability is required; rank by the uncapped ratio or "
            f"raw distances (rule R11 — the ranking is then scale-invariant "
            f"and does not depend on the radius parameter at all)."
        )

    # 8b. Multi-phase quota liveness. This catches a general class of silent
    # failures where code fills a budget in phase 1, computes zero remaining
    # budget, and therefore never executes the core phase 2 loop.
    dead_quota_hits = _dead_remaining_quota_loops(tree)
    for line, msg in dead_quota_hits:
        errors.append(
            f"method/method.py:{line}: {msg} Use separate quotas for initialization "
            "and acquisition/ranking phases, or compute remaining budget before "
            "the quota-filling loop if the second phase must execute."
        )

    inverse_distance_prose_hits = _inverse_distance_descending_prose_contradictions(
        tree, method_src
    )
    for line, msg in inverse_distance_prose_hits:
        errors.append(
            f"method/method.py:{line}: {msg}. For `R_0 / distance` or `1 / distance` "
            "scores sorted descending, closer/lower-distance samples receive the "
            "higher score. Fix the comments/docstring/prose or change the ranking "
            "formula/sort direction."
        )

    kmeanspp_broadcast_hits = _quadratic_kmeanspp_distance_broadcasts(tree)
    for line, msg in kmeanspp_broadcast_hits:
        errors.append(
            f"method/method.py:{line}: {msg}. This materializes an O(N*k*D) "
            "distance tensor and can time out or swap at active-learning smoke "
            "scale. Maintain a running `(N,)` nearest-distance array and update "
            "it one newly selected center at a time; the k-means++ semantics are "
            "identical and peak memory is O(N*D)."
        )

    # 9. Paper-element ID validity in method.py.
    paper_map_ids = _load_paper_map_ids(run_dir)
    if paper_map_ids is not None:
        method_path = run_dir / "method/method.py"
        for lineno, pid in _extract_paper_element_ids(method_path):
            if pid not in paper_map_ids:
                errors.append(
                    f"method/method.py:{lineno}: `# paper-element: {pid}` references an ID "
                    f"that doesn't exist in paper_map.json. The trace from code to paper is "
                    f"broken — pick a valid id (one of paper_map's existing element ids) or "
                    f"remove the annotation."
                )

    # 10. Essential-annotation coverage (cross-file).
    # For every spec-declared severity:essential feature, check that some file
    # in method/ has a `# essential: <feature-name>` annotation. Method-coder
    # gate is the right place: by now both architecture-coder and method-coder
    # have produced their files.
    cr_features = (
        ((spec.get("critical_requirements") or {}).get("model") or {}).get("specific_features") or []
    )
    essential_features = [
        f.get("feature", "") for f in cr_features if f.get("severity") == "essential"
    ]
    if essential_features:
        # Collect every `# essential: <name>` annotation across method/*.py.
        present_annotations: set[str] = set()
        method_dir = run_dir / "method"
        for py_path in sorted(method_dir.glob("*.py")):
            for line in py_path.read_text(encoding="utf-8").splitlines():
                m = re.search(r"#\s*essential:\s*(.+?)\s*$", line)
                if m:
                    present_annotations.add(m.group(1).strip())

        for feature in essential_features:
            if feature not in present_annotations:
                errors.append(
                    f"spec marks feature `{feature}` as severity:essential, but no "
                    f"`# essential: {feature}` annotation appears in any file under "
                    f"`<run_dir>/method/`. The implementation may be present, but the "
                    f"trace from spec to code is incomplete — add the annotation to the "
                    f"function or class that implements the feature."
                )

    return errors, warnings


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    args = parser.parse_args()

    if not args.spec.is_file():
        print(f"error: spec not found: {args.spec}", file=sys.stderr)
        return 2

    try:
        spec = json.loads(args.spec.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        print(f"error: spec is not valid JSON ({e})", file=sys.stderr)
        return 2

    errors, warnings = validate(spec, args.run_dir, args.repo_root)
    if errors:
        print(f"FAIL: {len(errors)} validation error(s):", file=sys.stderr)
        for err in errors:
            print(f"  - {err}", file=sys.stderr)
        for w in warnings:
            print(f"  warning: {w}", file=sys.stderr)
        return 1

    print(f"ok: method_coder output at {args.run_dir} validates against the manifest.")
    if warnings:
        print(f"  ({len(warnings)} warning(s))")
        for w in warnings:
            print(f"  warning: {w}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
