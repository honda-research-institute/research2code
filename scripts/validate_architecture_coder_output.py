"""Stage 2.b — validator for architecture_coder output.

Runs after the architecture_coder agent. Cross-references the run directory
against the matched taxonomy build plan's `package_manifest.files` block (entries with
`produced_by: architecture_coder`) and confirms:

  1. `<run_dir>/method/model.py` and `<run_dir>/method/training.py` exist and parse.
  2. model.py defines the number of top-level public classes inheriting
     from `nn.Module` that the paradigm's `package_manifest.files[method/model.py]
     .public_symbols` declares (kind=class entries). 1 for active_learning;
     2 for knowledge_distillation (student + teacher). Private classes
     prefixed with `_` don't count.
  3. Each architecture class has a `forward(self, x) -> ...` method.
  4. If the spec's critical_requirements.model.specific_features lists features
     implying paradigm-specific architecture hooks (e.g., "penultimate layer
     representations" → `forward_with_embedding`; "MC dropout" → at least one
     `nn.Dropout` layer in `__init__`), at least one of the architecture
     classes satisfies them.
  5. training.py defines every function the paradigm manifest declares
     (active_learning: `build_model` + `train_from_scratch`;
     knowledge_distillation: `build_student` + `build_teacher` +
     `train_with_distillation`).
  6. No declared function in training.py has a parameter whose name appears
     in `pluggable_component.contract.forbidden_param_names` (today: no
     `config` parameter).
  7. training.py's `from .model import <ArchitectureClass>` matches the class
     name in model.py.
  8. `train_from_scratch` calls `torch.manual_seed(...)` before any optimizer
     instantiation (heuristic AST check; the C-22 seed-plumbing rule).
  9. `.pipeline/arch_contract.json` validates through the versioned contract
     loader and uses schema 2.0.0 for newly authored Stage-2b output. Legacy
     1.0.0/1.1.0 contracts remain readable for diagnostics and resume
     consumers, but cannot pass this producer gate. Paradigm-specific
     completeness and runtime dry-run remain Stage 2.d checks.
  10. **Every `# paper-element: <id>` annotation in model.py / training.py
     references an ID that exists in `<run_dir>/.pipeline/paper_map.json`.**
     A broken trace (annotation cites an ID the paper map doesn't know about)
     misleads a reviewer who tries to follow the code → paper link. Catching
     this at the architecture-coder gate prevents the issue from cascading
     into downstream agents (notebook-generator) and into the end-of-pipeline
     reviewer.
  11. **Naming bridge (item 31b, the general fix for sub-shape a).** Every
     `try_it_out.system_provides` entry carrying a `symbol` must resolve to a
     PUBLIC top-level name in the generated package: defined public in a
     method/ module, or re-exported by method/__init__.py. Two files defining
     the same declared symbol is an ownership collision — fail loud, no
     silent dedup (maintainer decision 2026-07-16). Symbols the manifest assigns to a
     producer that runs AFTER the architecture coder (e.g. the pluggable
     function in method/method.py) are deferred, not failed, at this gate —
     the Stage 2.d finalizer re-checks every declared symbol once the whole
     package exists (deferral is never terminal).
     The class-count check (item 2) admits spec-declared extras: an extra
     public class in model.py is legal exactly when a declared bridge symbol
     resolves to it. Specs without symbols get no bridge enforcement.

Deterministic gate. On failure, exit 1 with a structured error list on stderr.

Usage:

    python scripts/validate_architecture_coder_output.py --spec <method_spec.json> --run-dir <output_dir>

Exit codes:
  0  validation passed (zero errors)
  1  validation failures
  2  setup error (missing spec / taxonomy build plan / manifest)
"""

from __future__ import annotations

import argparse
import ast
import json
import keyword
import os
import py_compile
import re
import sys
import tempfile
from pathlib import Path
from typing import Any

try:  # Package import (`import scripts.validate_architecture_coder_output`).
    from .paper_element_anchors import (
        PAPER_ELEMENT_PATTERN,
        extract_paper_element_ids,
    )
except ImportError:  # Direct CLI / scripts-on-path import.
    from paper_element_anchors import (
        PAPER_ELEMENT_PATTERN,
        extract_paper_element_ids,
    )


# The `# paper-element: <id>` scan moved to scripts/paper_element_anchors.py so
# the probe battery can reach it from the portable harness, which cannot import
# this validator (R2C-072). Re-exported under the historical private names,
# which several modules and tests import.
_PAPER_ELEMENT_PATTERN = PAPER_ELEMENT_PATTERN
_extract_paper_element_ids = extract_paper_element_ids


def _load_paper_map_ids(run_dir: Path) -> set[str] | None:
    """Load the set of paper-element IDs from `<run_dir>/.pipeline/paper_map.json`.

    Returns None if the paper_map is missing or unparseable — callers treat
    the check as "skipped" in that case (it's an upstream issue).
    """
    path = run_dir / ".pipeline" / "paper_map.json"
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    elements = data.get("elements") or []
    return {e.get("id") for e in elements if isinstance(e, dict) and e.get("id")}

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.build_plan import load_build_plan  # noqa: E402
from scripts.arch_contract_semantics import load_arch_contract  # noqa: E402
from scripts.taxonomy import run_overlay_dir  # noqa: E402
from scripts.validate_arch_contract import _check_family_components  # noqa: E402
from schemas.arch_contract_v2 import SCHEMA_VERSION as ARCH_CONTRACT_SCHEMA_VERSION  # noqa: E402
from scripts.signature_ast import _all_param_names  # noqa: E402


ARCHITECTURE_CODER_ROLE = "architecture_coder"


def _legacy_arch_contract_activation_error(schema_version: str) -> str:
    """Producer-owned diagnostic for a valid, but no-longer-authorable v1 contract.

    The versioned loader must still read legacy contracts so this gate can
    report their other useful structural diagnostics.  New Stage-2b output,
    however, must not pass on the ambiguous shape-string contract.
    """
    return (
        ".pipeline/arch_contract.json uses legacy schema_version "
        f"{schema_version!r}; new Stage 2.b architecture-coder output must use "
        f"schema_version {ARCH_CONTRACT_SCHEMA_VERSION!r} with typed dimensions "
        "and value descriptors. Version-1 contracts remain readable only for "
        "archived or resumed consumers."
    )


# ---------------------------------------------------------------------------
# AST helpers
# ---------------------------------------------------------------------------


def _parse_or_error(path: Path) -> tuple[ast.Module | None, str | None]:
    if not path.is_file():
        return None, f"{path} not present"
    text = path.read_text(encoding="utf-8")
    try:
        tree = ast.parse(text, filename=str(path))
    except SyntaxError as e:
        return None, f"{path.name} fails to parse as Python: {e}"
    # ast.parse accepts what real import rejects: a misplaced
    # `from __future__` import is the canonical case (SRL matrix row 1,
    # 2026-07-05 -- the skeleton-first fill-in pass duplicated the header,
    # ast.parse passed it at 2b where the fix loop lives, and the run died
    # at 2d's import validator, which has none). compile() enforces the
    # placement rules, so the class surfaces HERE and routes to the fixer.
    # py_compile applies the same checks as compile() from the file on disk;
    # the bytecode lands in a scratch dir so the run tree stays untouched.
    try:
        with tempfile.TemporaryDirectory(prefix="r2c-pycompile-") as tmp:
            py_compile.compile(
                str(path), cfile=os.path.join(tmp, "check.pyc"), doraise=True)
    except py_compile.PyCompileError as e:
        return None, f"{path.name} fails to compile as Python: {e.exc_value}"
    return tree, None


def _is_private_name(name: str) -> bool:
    return name.startswith("_")


def _bases_unparse(node: ast.ClassDef) -> list[str]:
    return [ast.unparse(b) for b in node.bases]


def _inherits_named(node: ast.ClassDef, base_decl: str) -> bool:
    """Check if class inherits from a base named `base_decl`.

    Matches by the LAST component of the dotted name. So
    `base_decl="torch.nn.Module"` matches base expressions in any acceptable
    Python form: bare `Module` (after `from torch.nn import Module`),
    `nn.Module` (after `import torch.nn as nn` or `from torch import nn`),
    or fully-qualified `torch.nn.Module`.

    Used to enforce a paradigm-declared `inherits:` constraint on a class.
    For paradigms whose manifest does NOT declare `inherits` (e.g.,
    motion_planning's plain-Python SystemDynamics + CollisionModel), the
    caller skips this check entirely.
    """
    target_leaf = base_decl.split(".")[-1]
    for b in node.bases:
        chain: list[str] = []
        cur: ast.AST = b
        while isinstance(cur, ast.Attribute):
            chain.insert(0, cur.attr)
            cur = cur.value
        if isinstance(cur, ast.Name):
            chain.insert(0, cur.id)
        if chain and chain[-1] == target_leaf:
            return True
    return False


def _find_public_top_level_classes(tree: ast.Module) -> list[ast.ClassDef]:
    """Return all public top-level class definitions (no `_`-prefixed names).

    Replaces the prior `_find_module_subclasses` which filtered to nn.Module
    subclasses only — a supervised-ML assumption that broke motion_planning
    (pdwa 2026-05-26 halt). The class-count + per-class checks downstream
    now drive off the manifest's `public_symbols[*]` declarations
    (specifically `inherits:` and `required_methods:`) rather than hardcoded
    inheritance + method names.
    """
    out: list[ast.ClassDef] = []
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and not _is_private_name(node.name):
            out.append(node)
    return out


def _method_name_from_signature(sig: str) -> str:
    """Extract the leading method name from a signature like
    `"step(self, state: Tensor, ...) -> Tensor"` → `"step"`. Returns an
    empty string if the signature is malformed."""
    open_paren = sig.find("(")
    if open_paren <= 0:
        return ""
    return sig[:open_paren].strip()


def _class_names_in_function_signatures(tree: ast.Module) -> set[str]:
    """Collect every identifier appearing in any top-level function's
    argument annotations or return annotation. Used to determine which
    classes a training.py module actually references (so we can require
    just those — not every class declared in model.py — to be imported)."""
    names: set[str] = set()

    def _collect(node: ast.AST) -> None:
        if isinstance(node, ast.Name):
            names.add(node.id)
        elif isinstance(node, ast.Attribute):
            cur: ast.AST = node
            while isinstance(cur, ast.Attribute):
                cur = cur.value
            if isinstance(cur, ast.Name):
                names.add(cur.id)
        elif isinstance(node, ast.Subscript):
            _collect(node.value)
            _collect(node.slice)
        elif isinstance(node, ast.Tuple):
            for elt in node.elts:
                _collect(elt)

    for n in tree.body:
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
            args = n.args
            for a in args.posonlyargs + args.args + args.kwonlyargs:
                if a.annotation is not None:
                    _collect(a.annotation)
            if n.returns is not None:
                _collect(n.returns)
    return names


def _find_method_in_class(cls: ast.ClassDef, method_name: str) -> ast.FunctionDef | ast.AsyncFunctionDef | None:
    for node in cls.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == method_name:
            return node
    return None


def _local_class_map(tree: ast.Module) -> dict[str, ast.ClassDef]:
    """All top-level classes in the module by name, private included."""
    return {n.name: n for n in tree.body if isinstance(n, ast.ClassDef)}


def _class_and_local_ancestors(
    cls: ast.ClassDef, class_map: dict[str, ast.ClassDef],
) -> list[ast.ClassDef]:
    """`cls` plus every base class defined in the same module, transitively
    (cycle-guarded). Method and inheritance checks walk this chain so an
    idiomatic shared private base is not mis-read as "method missing"
    (detr-distill 2026-07-08: SimpleDETRStudent / SimpleDETRTeacher inherit
    `forward` and `forward_with_intermediates` from a local `_DETR(nn.Module)`
    base, and the body-only lookup reported all of it absent)."""
    out: list[ast.ClassDef] = []
    seen: set[str] = set()
    stack = [cls]
    while stack:
        cur = stack.pop()
        if cur.name in seen:
            continue
        seen.add(cur.name)
        out.append(cur)
        for b in cur.bases:
            if isinstance(b, ast.Name) and b.id in class_map:
                stack.append(class_map[b.id])
    return out


def _find_method_in_class_or_bases(
    cls: ast.ClassDef, method_name: str, class_map: dict[str, ast.ClassDef],
) -> ast.FunctionDef | ast.AsyncFunctionDef | None:
    for node_cls in _class_and_local_ancestors(cls, class_map):
        found = _find_method_in_class(node_cls, method_name)
        if found is not None:
            return found
    return None


def _inherits_named_in_chain(
    cls: ast.ClassDef, base_decl: str, class_map: dict[str, ast.ClassDef],
) -> bool:
    """`_inherits_named` through locally-defined intermediate bases, so
    `class Student(_DETR)` with `class _DETR(nn.Module)` satisfies an
    `inherits: torch.nn.Module` declaration."""
    return any(
        _inherits_named(node_cls, base_decl)
        for node_cls in _class_and_local_ancestors(cls, class_map)
    )


def _class_has_dropout_layer(cls: ast.ClassDef) -> bool:
    """Heuristic: any constructor call to nn.Dropout(...) anywhere in the class body."""
    for node in ast.walk(cls):
        if isinstance(node, ast.Call):
            func = node.func
            chain: list[str] = []
            cur: ast.AST = func
            while isinstance(cur, ast.Attribute):
                chain.insert(0, cur.attr)
                cur = cur.value
            if isinstance(cur, ast.Name):
                chain.insert(0, cur.id)
            name = chain[-1] if chain else ""
            # Match every torch dropout layer: Dropout, Dropout1d/2d/3d (the
            # dimensional variants — `endswith("Dropout")` is False for these, the
            # bug that spuriously rejected correct MC-dropout CNNs), AlphaDropout,
            # FeatureAlphaDropout. The detector is a coarse heuristic ("any nn.Dropout*
            # constructor in the class body"), so substring-match fits its contract.
            if "Dropout" in name:
                return True
    return False


def _find_top_level_def(tree: ast.Module, name: str) -> ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef | None:
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node.name == name:
            return node
    return None


def _required_param_names(func: ast.FunctionDef | ast.AsyncFunctionDef) -> list[str]:
    """Params with no default (positional + kw-only). These are the names a
    caller MUST supply; an unexpected one here is what makes a function
    unbindable by the probe harness / notebook driver."""
    args = func.args
    positional = [*args.posonlyargs, *args.args]
    n_required_positional = len(positional) - len(args.defaults)
    required = [a.arg for a in positional[:n_required_positional]]
    required.extend(
        a.arg for a, default in zip(args.kwonlyargs, args.kw_defaults)
        if default is None
    )
    return required


# Module dunders CPython binds implicitly; dir(builtins) does not cover all.
_MODULE_IMPLICIT_NAMES = {
    "__file__", "__name__", "__doc__", "__package__", "__spec__",
    "__loader__", "__builtins__", "__debug__", "__annotations__",
    "__dict__", "__all__", "__path__", "__cached__", "__class__",
}


def _module_binding_names(tree: ast.Module) -> tuple[set[str], bool]:
    """(every name bound anywhere in the module, has_star_import).

    Deliberately scope-blind: a name bound in ANY scope counts as defined
    everywhere, so the undefined-reference check over-approximates
    definitions and only genuinely unbound names surface."""
    names: set[str] = set()
    star_import = False
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
            names.add(node.id)
        elif isinstance(node, ast.arg):
            names.add(node.arg)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                names.add((alias.asname or alias.name).split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                if alias.name == "*":
                    star_import = True
                else:
                    names.add(alias.asname or alias.name)
        elif isinstance(node, (ast.Global, ast.Nonlocal)):
            names.update(node.names)
        elif isinstance(node, ast.ExceptHandler) and node.name:
            names.add(node.name)
        elif isinstance(node, (ast.MatchAs, ast.MatchStar)) and node.name:
            names.add(node.name)
        elif isinstance(node, ast.MatchMapping) and node.rest:
            names.add(node.rest)
    return names, star_import


def _undefined_name_references(tree: ast.Module) -> list[tuple[str, int, int]]:
    """[(name, reference_count, first_lineno)] for names loaded but never
    bound or imported anywhere in the module (item 31b sub-shape b). Files
    with star imports return [] — their namespace is unknowable statically."""
    bound, star_import = _module_binding_names(tree)
    if star_import:
        return []
    import builtins as _builtins

    known = bound | set(dir(_builtins)) | _MODULE_IMPLICIT_NAMES
    undefined: dict[str, tuple[int, int]] = {}
    for node in ast.walk(tree):
        if (isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
                and node.id not in known):
            count, first = undefined.get(node.id, (0, node.lineno))
            undefined[node.id] = (count + 1, min(first, node.lineno))
    return sorted(
        (name, count, first) for name, (count, first) in undefined.items()
    )


# Manifest training signatures carry `<Placeholder>` annotations (e.g.
# `model: <ArchitectureClass>`) that are not valid Python, so a naive
# ast.parse of the signature string fails. Swap any `<Ident>` placeholder for a
# valid dummy type before parsing so we can recover the parameter names.
_SIGNATURE_PLACEHOLDER_RE = re.compile(r"<[A-Za-z_][A-Za-z0-9_]*>")


def _manifest_signature_shape(sig_str: str) -> tuple[set[str], bool] | None:
    """Return (declared_param_names, accepts_extra_kwargs) for a manifest
    signature string, or None when it cannot be parsed.

    `accepts_extra_kwargs` is True when the contract signature ends in
    `**kwargs` (e.g. build_model's `**arch_kwargs`) — such functions are
    explicitly allowed to take extra params, so the extra-required check is
    skipped for them."""
    sanitized = _SIGNATURE_PLACEHOLDER_RE.sub("object", sig_str)
    try:
        tree = ast.parse(f"def {sanitized}:\n    pass\n")
    except (SyntaxError, ValueError):
        return None
    if not tree.body or not isinstance(tree.body[0], (ast.FunctionDef, ast.AsyncFunctionDef)):
        return None
    func = tree.body[0]
    return set(_all_param_names(func)), func.args.kwarg is not None


def _import_alias_for_class(tree: ast.Module, class_name: str) -> bool:
    """Return True if there's a `from .model import <class_name>` (or aliased).

    Walks the WHOLE tree, not just module top level: a perfectly valid
    `if TYPE_CHECKING: from .model import X` was invisible to the old
    tree.body loop and degraded ACC's stage 2b on 2026-07-14 (validator
    false-negative, judge-classified pipeline_bug)."""
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if module == "model" or module.endswith(".model") or node.level >= 1 and node.module == "model":
                for alias in node.names:
                    if alias.name == class_name:
                        return True
                    if alias.asname == class_name:
                        return True
    return False


def _seed_called_before_optimizer(func: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    """Heuristic: the first statement that mentions 'optimizer' or 'Adam' is preceded by a manual_seed call.

    Walks the function body in order; tracks whether any node has called manual_seed yet, and
    whether we've encountered an optimizer instantiation. Returns False if optimizer is created
    before any manual_seed call.
    """
    seen_manual_seed = False
    for stmt in func.body:
        # Check this statement for manual_seed or optimizer patterns.
        for node in ast.walk(stmt):
            if isinstance(node, ast.Call):
                func_node = node.func
                chain: list[str] = []
                cur: ast.AST = func_node
                while isinstance(cur, ast.Attribute):
                    chain.insert(0, cur.attr)
                    cur = cur.value
                if isinstance(cur, ast.Name):
                    chain.insert(0, cur.id)
                if chain and chain[-1] == "manual_seed":
                    seen_manual_seed = True
                if chain and chain[-1] == "Adam":
                    if not seen_manual_seed:
                        return False
                if "optim" in ".".join(chain) and chain and chain[-1] != "manual_seed":
                    # heuristic: any torch.optim.* construction
                    if not seen_manual_seed:
                        return False
    return True


# ---------------------------------------------------------------------------
# Spec helpers
# ---------------------------------------------------------------------------


def _required_architecture_hooks(spec: dict) -> dict[str, Any]:
    """Translate spec.critical_requirements.model.specific_features into structured hook flags.

    Returns a dict with keys:
      - needs_forward_with_embedding: bool
      - needs_dropout_layers: bool
      - essential_features: list[str] — names of severity:essential features
    """
    cr = spec.get("critical_requirements") or {}
    model = cr.get("model") or {}
    features = model.get("specific_features") or []

    needs_fwe = False
    needs_dropout = False
    essential: list[str] = []

    for feat in features:
        name = (feat.get("feature") or "").lower()
        sev = feat.get("severity") or ""
        if sev == "essential":
            essential.append(feat.get("feature") or "")
        if "penultimate" in name or "embedding" in name:
            needs_fwe = True
        if "mc dropout" in name or "dropout" in name and "active at inference" in name:
            needs_dropout = True
        if "monte" in name and "carlo" in name and "dropout" in name:
            needs_dropout = True
        # GBALD spec-style phrasing
        if "dropout layers active at inference" in name:
            needs_dropout = True

    # `forward_with_embedding(x) -> (logits, z)` is the ACTIVE-LEARNING
    # gradient-embedding interface convention (BADGE/GBALD). The keyword
    # heuristic above fires on any feature naming "embedding" — which is
    # paradigm-agnostic vocabulary (detr-distill 2026-07-03: a KD spec's
    # "teacher query embeddings" feature, already served by the spec's own
    # required_model_methods, halted 2.b on this irrelevant AL hook; the
    # halt-judge classified it pipeline_bug/high). Scope the heuristic to
    # active_learning; an EXPLICIT required_model_methods declaration of
    # forward_with_embedding wins for any paradigm. Non-AL interface needs
    # are enforced by the required_model_methods gate, not this heuristic.
    paradigm_id = (
        ((spec.get("comparison") or {}).get("classification") or {}).get("id")
        or ""
    )
    declared_methods = {
        (m.get("name") or "") if isinstance(m, dict) else str(m)
        for m in (model.get("required_model_methods") or [])
    }
    if "forward_with_embedding" in declared_methods:
        needs_fwe = True
    elif not paradigm_id.startswith("active_learning"):
        needs_fwe = False

    return {
        "needs_forward_with_embedding": needs_fwe,
        "needs_dropout_layers": needs_dropout,
        "essential_features": essential,
    }


def _declared_method_names(exp: dict) -> list[str]:
    """Method names a manifest class declaration requires."""
    names = [
        _method_name_from_signature(s)
        for s in (exp.get("required_methods") or [])
    ]
    return [m for m in names if m]


def _class_satisfies_declaration(
    exp: dict, found: ast.ClassDef, class_map: dict[str, ast.ClassDef],
) -> bool:
    """Does `found` satisfy a manifest class declaration's `inherits:`
    (optional) and `required_methods:` constraints (resolved through
    same-module base classes)?"""
    inherits_decl = exp.get("inherits")
    if inherits_decl and not _inherits_named_in_chain(found, inherits_decl, class_map):
        return False
    return all(
        _find_method_in_class_or_bases(found, m, class_map) is not None
        for m in _declared_method_names(exp)
    )


# ---------------------------------------------------------------------------
# Naming bridge (item 31b, the general fix for sub-shape a — the ICRA
# 2026-07-13 private-classifier genus, spec_promise_not_importable). Stage 1
# declares an optional importable `symbol` per try_it_out.system_provides
# entry; this gate resolves each declared symbol to a public top-level name
# in the generated package. Design:
# the naming bridge design note (internal, not shipped).
# ---------------------------------------------------------------------------


# Producers whose files exist by the time this validator runs. Symbols the
# manifest assigns to a LATER producer (method_coder's method/method.py) are
# deferred at this gate rather than failed — failing them here would
# mis-attribute a not-yet-generated file to the architecture coder. The
# init_finalizer is deliberately NOT a later owner: its __init__.py is
# derived (it re-exports names already public in the modules), so it can
# never rescue a symbol no module defines.
_STAGE_2B_PRESENT_PRODUCERS = {"package_scaffolder", ARCHITECTURE_CODER_ROLE}


def _spec_bridge_entries(spec: dict) -> list[dict]:
    """Declared naming-bridge symbols from `try_it_out.system_provides`.

    Returns [{"symbol", "name", "type"}, ...]. Entries without a symbol get
    no bridge enforcement (conservative by construction — omission is legal,
    never guessed). A syntactically invalid symbol is skipped too: the Stage 1
    schema enforces identifier syntax, so a bad value here is a legacy or
    hand-edited spec, and failing the ARCHITECTURE coder for it would
    mis-attribute a Stage 1 defect.
    """
    entries = (spec.get("try_it_out") or {}).get("system_provides") or []
    out: list[dict] = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        symbol = entry.get("symbol")
        if (
            not isinstance(symbol, str)
            or not symbol.isidentifier()
            or keyword.iskeyword(symbol)  # mirror the schema: keywords are not importable
            or symbol.startswith("_")
        ):
            continue
        # symbol_kind (schema v1.8.0, plan item 10): the promised surface's
        # kind. An unknown value is treated as undeclared (name-only
        # enforcement) for the same legacy-spec reason invalid symbols are
        # skipped above — failing THIS producer for a Stage 1 defect would
        # mis-attribute it.
        symbol_kind = entry.get("symbol_kind")
        if symbol_kind not in ("class", "function"):
            symbol_kind = None
        out.append({
            "symbol": symbol,
            "name": entry.get("name") or symbol,
            "type": entry.get("type") or "",
            "symbol_kind": symbol_kind,
        })
    return out


def _assigned_name_targets(target: ast.AST):
    """Yield the names an assignment TARGET binds — Store-ctx `ast.Name`
    nodes only, recursing through tuple/list/starred unpacking. Attribute
    and subscript targets (`X.THRESHOLD = 0.5`, `registry["x"] = ...`)
    mutate an existing object and define nothing, so they never count as a
    definition site (aligned with `_module_binding_names`'s Store-ctx rule;
    a blind walk here produced spurious ownership collisions)."""
    if isinstance(target, ast.Name) and isinstance(target.ctx, ast.Store):
        yield target.id
    elif isinstance(target, (ast.Tuple, ast.List)):
        for elt in target.elts:
            yield from _assigned_name_targets(elt)
    elif isinstance(target, ast.Starred):
        yield from _assigned_name_targets(target.value)


def _top_level_defined_names(tree: ast.Module) -> set[str]:
    """Every name a module DEFINES at top level (classes, functions, and
    Store-ctx assignments), private included. Import bindings are
    deliberately excluded — `from .model import X` in training.py makes X
    importable but does not make training.py a second definition site for
    the ownership-collision check."""
    names: set[str] = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                names.update(_assigned_name_targets(target))
        elif isinstance(node, ast.AnnAssign):
            names.update(_assigned_name_targets(node.target))
    return names


def _top_level_public_definitions(tree: ast.Module) -> set[str]:
    """Public subset of `_top_level_defined_names` — the definition sites the
    bridge resolution and ownership-collision checks bind against."""
    return {n for n in _top_level_defined_names(tree) if not _is_private_name(n)}


def dead_module_members(
    tree: ast.Module,
    external_trees: tuple[ast.Module, ...] = (),
) -> list[tuple[int, str, str]]:
    """dead_code_masking — layers constructed in `__init__` and never read
    anywhere in the package.

    The pdfgnn 2026-08-04 delivery built a `decoder_projection` layer whose
    docstring said it maps the full covariate concatenation down to the
    LSTM's input size, then never called it: `forward` fed the LSTM raw, so
    passing the paper-mandated covariates crashed, and the dead layer MASKED
    the missing mechanism (the contract dry-run's expected-9-got-25 failure,
    and the fidelity review's HIGH finding 1). A constructed-but-never-read
    submodule is exactly that shape, and it is decidable at the producing
    stage.

    The use-set is PACKAGE-WIDE attribute loads, not class-internal ones:
    knowledge distillation legitimately constructs projection heads on the
    model classes and applies them from the training module
    (`student.proj_head_2d(...)` in bev-distill's distillation loop), so
    `external_trees` carries the sibling method/ modules. Conservative on
    purpose: only `self.X = <call on an nn./torch.nn. chain>` assignments
    count as constructions, any attribute LOAD of the name anywhere counts
    as a use, and a class that touches `getattr`/`setattr` on self is
    skipped wholesale (dynamic access)."""
    loaded: set[str] = set()
    for t in (tree, *external_trees):
        for sub in ast.walk(t):
            if isinstance(sub, ast.Attribute) and isinstance(sub.ctx, ast.Load):
                loaded.add(sub.attr)

    findings: list[tuple[int, str, str]] = []
    for node in tree.body:
        if not isinstance(node, ast.ClassDef):
            continue
        dynamic = False
        for sub in ast.walk(node):
            if (isinstance(sub, ast.Call) and isinstance(sub.func, ast.Name)
                    and sub.func.id in ("getattr", "setattr")
                    and sub.args
                    and isinstance(sub.args[0], ast.Name)
                    and sub.args[0].id == "self"):
                dynamic = True
                break
        if dynamic:
            continue
        constructed: dict[str, int] = {}
        for sub in ast.walk(node):
            if not isinstance(sub, ast.Assign):
                continue
            if not isinstance(sub.value, ast.Call):
                continue
            chain = []
            f = sub.value.func
            while isinstance(f, ast.Attribute):
                chain.append(f.attr)
                f = f.value
            if isinstance(f, ast.Name):
                chain.append(f.id)
            chain_str = ".".join(reversed(chain))
            if not (chain_str.startswith("nn.") or chain_str.startswith("torch.nn.")):
                continue
            for tgt in sub.targets:
                if (isinstance(tgt, ast.Attribute)
                        and isinstance(tgt.value, ast.Name)
                        and tgt.value.id == "self"
                        and not tgt.attr.startswith("__")):
                    constructed[tgt.attr] = sub.lineno
        for attr, lineno in sorted(constructed.items(), key=lambda kv: kv[1]):
            if attr not in loaded:
                findings.append((lineno, node.name, attr))
    return findings


def _top_level_definition_kinds(tree: ast.Module) -> dict[str, str]:
    """Map every top-level defined name to its surface kind: "class",
    "function", or "assignment". The kind check only fails on a DEFINITE
    mismatch (class vs function); an assignment binding can alias anything,
    so it satisfies any declared kind (conservative by construction, same
    posture as the rest of the bridge). Later bindings win, matching Python
    runtime semantics for a rebound name."""
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
    return kinds


def _top_level_names_all(tree: ast.Module) -> set[str]:
    """Alias kept for the underscore-variant hint scan — every top-level
    defined name, private included."""
    return _top_level_defined_names(tree)


def _init_public_bindings(
    init_path: Path, module_bindings: dict[str, set[str]],
) -> set[str]:
    """Public names `method/__init__.py` binds — the re-export surface.

    Imports count here (unlike module definition sites): a re-export
    `from .model import _AvoidanceClassifier as AvoidanceClassifier` is
    exactly the second accepted remedy, and an imported-then-re-exported
    third-party name resolves the same way.

    A package-internal re-export only counts when the SOURCE module actually
    binds the imported name (`module_bindings` maps module stem to its
    scope-blind `_module_binding_names` set). Without that check, a phantom
    `from .model import AvoidanceClassifier` — dead at import time, or worse
    wrapped in try/except — would satisfy the bridge while the delivered
    package still cannot serve the promise."""
    if not init_path.is_file():
        return set()
    try:
        tree = ast.parse(init_path.read_text(encoding="utf-8"))
    except SyntaxError:
        return set()
    names = set(_top_level_public_definitions(tree))
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            module = node.module or ""
            package_internal = (
                node.level > 0
                or module == "method"
                or module.startswith("method.")
            )
            source_stem: str | None = None
            if package_internal:
                parts = [p for p in module.split(".") if p]
                if parts and parts[0] == "method":
                    parts = parts[1:]
                source_stem = parts[-1] if parts else None
            for alias in node.names:
                if alias.name == "*":
                    # `from .model import *` cannot be verified statically;
                    # conservative: it never satisfies the bridge (the
                    # finalizer-generated __init__ always names its exports).
                    continue
                bound = alias.asname or alias.name
                if _is_private_name(bound):
                    continue
                if not package_internal:
                    # Third-party re-export — accepted as-is (the design's
                    # imported-then-re-exported allowance).
                    names.add(bound)
                elif source_stem is None:
                    # `from . import <submodule>` binds a module object, not
                    # a promised symbol.
                    continue
                elif alias.name in module_bindings.get(source_stem, set()):
                    names.add(bound)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                bound = alias.asname or alias.name.split(".")[0]
                if not _is_private_name(bound):
                    names.add(bound)
    return names


def _package_module_trees(
    run_dir: Path,
    model_tree: ast.Module | None,
    training_tree: ast.Module | None,
) -> dict[str, ast.Module]:
    """`{relative_path: tree}` for every parseable method/*.py module
    (``__init__.py`` excluded — it is the re-export surface, not a module).
    Unparseable files owned by other producers are skipped; their own gates
    report them."""
    trees: dict[str, ast.Module] = {}
    method_dir = run_dir / "method"
    if not method_dir.is_dir():
        return trees
    # Deliberately non-recursive: the manifest declares method/ as a flat
    # package of top-level modules, and the top-level modules ARE the import
    # surface the promise resolves against. A subpackage would be a manifest
    # violation caught by its own gates, not silently scanned here.
    for py in sorted(method_dir.glob("*.py")):
        if py.name == "__init__.py":
            continue
        rel = f"method/{py.name}"
        if py.name == "model.py" and model_tree is not None:
            trees[rel] = model_tree
            continue
        if py.name == "training.py" and training_tree is not None:
            trees[rel] = training_tree
            continue
        try:
            trees[rel] = ast.parse(py.read_text(encoding="utf-8"), filename=str(py))
        except SyntaxError:
            continue
    return trees


def _later_producer_may_own(
    symbol: str,
    entry_type: str,
    files_block: list[dict],
    symbol_kind: str | None = None,
) -> bool:
    """True when the manifest assigns a file to a producer that runs after
    the architecture coder and that file may own `symbol`: a concrete
    public_symbols name match (the pluggable function), or an open-ended
    function bucket (`all_top_level_functions_are_public`, the method
    helpers).

    `type: model` promises NEVER defer — a promised model is a class on the
    architecture surface (enforced right here, the ICRA genus), regardless
    of any name coincidence with a later file's declared symbols. The
    declared `symbol_kind` extends that axis (plan item 10, the ADAM genus):
    a `class` promise never defers to the open-ended FUNCTION bucket (a
    function-only producer cannot deliver a class), and a concrete name
    match only defers when the manifest entry's own `kind` does not
    contradict the promise. Deferral is not terminal for the rest either:
    the Stage 2.d finalizer re-checks every declared symbol once the whole
    package exists and fails with method-coder attribution if one is still
    unresolved."""
    if entry_type == "model":
        return False
    for f in files_block:
        produced_by = f.get("produced_by")
        if produced_by in _STAGE_2B_PRESENT_PRODUCERS or produced_by == "init_finalizer":
            continue
        for sym in (f.get("public_symbols") or []):
            if sym.get("name") == symbol:
                manifest_kind = sym.get("kind")
                if (
                    symbol_kind
                    and manifest_kind in ("class", "function")
                    and manifest_kind != symbol_kind
                ):
                    continue  # the later file's declared kind contradicts the promise
                return True
            if (
                sym.get("all_top_level_functions_are_public")
                and symbol_kind != "class"
            ):
                return True
    return False


def class_surplus_is_bridge_covered(
    model_tree: ast.Module,
    expected_classes: list[dict],
    bridge_symbol_names: set[str],
) -> bool:
    """The leak-free class-count allowance, shared by the 2.b gate and the
    2.d finalizer: True iff a one-to-one constraint-satisfying assignment of
    the manifest's class declarations to model.py's public classes exists in
    which EVERY unassigned (extra) class is named by a declared bridge
    symbol.

    Assignment-based on purpose. Plain arithmetic ("classes minus
    bridge-named classes <= expected") leaks: a symbol naming one of the
    manifest's own role classes would free an allowance slot for an
    undeclared stray. Here a role class consumed by the assignment frees
    nothing, and one symbol can never cover two classes (coverage is by
    exact name)."""
    arch_classes = _find_public_top_level_classes(model_tree)
    class_map = _local_class_map(model_tree)
    candidates = [
        [c for c in arch_classes if _class_satisfies_declaration(exp, c, class_map)]
        for exp in expected_classes
    ]

    def _assign(i: int, used: set[int]) -> bool:
        if i == len(expected_classes):
            return all(
                c.name in bridge_symbol_names
                for c in arch_classes
                if id(c) not in used
            )
        for c in candidates[i]:
            if id(c) in used:
                continue
            used.add(id(c))
            if _assign(i + 1, used):
                return True
            used.discard(id(c))
        return False

    return _assign(0, set())


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate(spec: dict, run_dir: Path, repo_root: Path) -> list[str]:
    errors: list[str] = []

    build_plan = load_build_plan(
        spec, repo_root, provisional_packs_dir=run_overlay_dir(run_dir))
    if build_plan is None:
        paradigm_id = (spec.get("comparison") or {}).get("classification", {}).get("id")
        return [f"no taxonomy build plan for comparison.classification.id={paradigm_id!r}"]
    manifest = build_plan.get("package_manifest") or {}
    pluggable_contract = (
        (build_plan.get("pluggable_component") or {}).get("contract") or {}
    )

    files_block = manifest.get("files") or []
    arch_files = [f for f in files_block if f.get("produced_by") == ARCHITECTURE_CODER_ROLE]
    if not arch_files:
        return [f"manifest declares no files with produced_by: {ARCHITECTURE_CODER_ROLE}"]

    # Extract per-paradigm expected shape from the manifest. The validator
    # was hardcoded to "exactly one nn.Module subclass + build_model +
    # train_from_scratch" (the active_learning shape) and rejected correct
    # multi-class outputs from paradigms like knowledge_distillation that
    # declare student + teacher + multiple builders. The bev-distill
    # 2026-05-19 cascade halted at exactly this gate; the halt-judge
    # classified it as a pipeline bug. Fix: drive the class-count and
    # function-name checks off the paradigm's declared public_symbols.
    expected_classes: list[dict] = []  # symbols with kind="class" in model.py entry
    expected_funcs: list[dict] = []    # symbols with kind="function" in training.py entry
    any_class_methods: list[str] = []  # spec-derived methods, no class attribution
    # The generic provisional plan (item 4) cannot know a brand-new family's
    # component count, so its model.py row declares `class_count: flexible`:
    # at least one public class (the zero-class check below stays), no exact
    # count. Committed plans never set the flag — their counts stay pinned.
    flexible_class_count = False
    for f in arch_files:
        rel = f.get("path") or ""
        if rel == "method/model.py":
            any_class_methods.extend(f.get("any_class_required_methods") or [])
            if f.get("class_count") == "flexible":
                flexible_class_count = True
        for sym in (f.get("public_symbols") or []):
            kind = sym.get("kind")
            if rel == "method/model.py" and kind == "class":
                expected_classes.append(sym)
            elif rel == "method/training.py" and kind == "function":
                expected_funcs.append(sym)
    if not expected_classes:
        # Defensive — every paradigm manifest we ship declares at least one
        # class. If we get here it's a paradigm-authoring bug; surface it.
        return [
            f"manifest's architecture_coder entry for method/model.py has no "
            f"public_symbols with kind=class. Taxonomy build-plan bug."
        ]
    if not expected_funcs:
        return [
            f"manifest's architecture_coder entry for method/training.py has no "
            f"public_symbols with kind=function. Taxonomy build-plan bug."
        ]
    expected_class_count = len(expected_classes)
    expected_func_names = [
        s.get("name") for s in expected_funcs if s.get("name")
    ]

    forbidden_params = set(pluggable_contract.get("forbidden_param_names") or [])

    # 1. File presence + parse
    model_path = run_dir / "method/model.py"
    training_path = run_dir / "method/training.py"
    arch_contract_path = run_dir / ".pipeline/arch_contract.json"

    model_tree, model_err = _parse_or_error(model_path)
    if model_err:
        errors.append(model_err)
    training_tree, training_err = _parse_or_error(training_path)
    if training_err:
        errors.append(training_err)
    # arch_contract.json is the third architecture-coder output. Check the
    # versioned universal schema here so a narrative/free-form contract stays
    # in the architecture-coder fix loop instead of cascading to method-coder
    # and Stage 2.d. The shared loader deliberately keeps v1.0/v1.1 readable
    # for archived/resumed consumers; this authoring gate adds one explicit
    # producer-owned activation error for an otherwise valid legacy contract
    # and continues, preserving useful structural diagnostics. Paradigm
    # completeness + runtime consistency remain Stage 2.d.
    if not arch_contract_path.is_file():
        errors.append(
            f"{arch_contract_path} missing — architecture-coder must produce "
            f"`.pipeline/arch_contract.json` alongside model.py and training.py. "
            f"See schemas/arch_contract.py for the structure and the matched "
            f"paradigm's `arch_contract_requirements` block for the "
            f"required interior."
        )
    else:
        try:
            arch_contract_raw = json.loads(arch_contract_path.read_text(encoding="utf-8"))
            arch_contract_model = load_arch_contract(arch_contract_raw)
        except Exception as e:
            errors.append(f".pipeline/arch_contract.json failed schema validation: {e}")
        else:
            if arch_contract_model.schema_version != ARCH_CONTRACT_SCHEMA_VERSION:
                errors.append(_legacy_arch_contract_activation_error(
                    arch_contract_model.schema_version
                ))
            # Family-component names fail at THIS gate, not only at Stage
            # 2.d: before the family_components container existed, a stray
            # top-level contract key died right here via extra='forbid', and
            # the typo class must keep failing at the producer's own gate
            # instead of two stages later (post-container review finding,
            # 2026-07-17). Stage 2.d repeats the check as the authoritative
            # completeness gate.
            _declared = ((build_plan.get("arch_contract_requirements") or {})
                         .get("family_components") or {})
            _overlay = run_overlay_dir(run_dir)
            _pack_hint = (
                f"the run's provisional pack under {_overlay}"
                if _overlay is not None
                else "the family's taxonomy node (docs/ssot/taxonomies.yaml)"
            )
            errors.extend(_check_family_components(
                arch_contract_model, _declared, _pack_hint))
    if model_tree is None or training_tree is None:
        return errors

    # Naming-bridge declarations (item 31b general fix). Computed before the
    # class-count check because spec-declared symbols that resolve to public
    # classes in model.py are legal EXTRAS for that count (the ICRA trap:
    # renaming the promised classifiers public must not convert the
    # private-name failure into a count failure). Empty for legacy specs and
    # for specs that declare no symbols — every check below then behaves
    # exactly as before.
    bridge_entries = _spec_bridge_entries(spec)
    # The count allowance only admits extra classes for symbols that can BE
    # classes: kind-undeclared (legacy, conservative) or declared `class`. A
    # declared `function` symbol frees no class slot (plan item 10 — without
    # the filter, a function promise's name colliding with a stray class
    # would launder the stray past the count check).
    bridge_symbol_names = {
        e["symbol"] for e in bridge_entries
        if e.get("symbol_kind") in (None, "class")
    }

    # 1c. Dead constructed submodules (dead_code_masking). A layer built in
    # __init__ and never read ANYWHERE in the package is either a missing
    # mechanism wearing a mask (the pdfgnn decoder_projection) or dead
    # weight; both are the coder's to resolve: wire it into the data path or
    # delete it. Sibling method/ modules join the use-set because KD-style
    # packages legitimately apply model-owned heads from the training loop
    # (bev-distill's proj_head_2d/proj_head_3d).
    sibling_trees: list[ast.Module] = []
    for sibling in sorted((run_dir / "method").glob("*.py")):
        if sibling.name in ("model.py", "__init__.py"):
            continue
        try:
            sibling_trees.append(
                ast.parse(sibling.read_text(encoding="utf-8"), filename=str(sibling)))
        except (OSError, SyntaxError):
            continue
    for lineno, class_name, attr in dead_module_members(
            model_tree, tuple(sibling_trees)):
        errors.append(
            f"method/model.py:{lineno}: `{class_name}.__init__` constructs "
            f"`self.{attr}` (an nn.* submodule) but no method of the class "
            f"ever reads `self.{attr}`. A constructed-but-unused layer either "
            f"masks a mechanism the contract expects on the data path or is "
            f"dead weight — wire it in or remove it (dead_code_masking)."
        )

    # 2. Architecture class count matches the manifest's declaration.
    # Single-class paradigms (active_learning) declare 1 class; multi-class
    # paradigms (knowledge_distillation: student + teacher) declare more;
    # non-ML paradigms (motion_planning) declare plain-Python classes
    # (SystemDynamics + CollisionModel) that do NOT inherit from nn.Module.
    # The validator counts public top-level classes regardless of inheritance
    # and matches them against the manifest's `inherits:` (optional) and
    # `required_methods:` declarations below.
    arch_classes = _find_public_top_level_classes(model_tree)
    if len(arch_classes) == 0:
        errors.append(
            f"method/model.py: no public top-level class found "
            f"(paradigm manifest declares {expected_class_count} class(es))"
        )
        return errors
    # Spec-declared bridge symbols admit extra public classes: an extra class
    # is legal exactly when a declared symbol resolves to it (naming-bridge
    # count allowance, assignment-based via `class_surplus_is_bridge_covered`
    # so a symbol naming a manifest ROLE class frees no slot for an
    # undeclared stray). Typo safety is unchanged — an extra class nobody
    # declared still fails, and fewer classes than the manifest declares
    # still fails. With no declared symbols this reduces to the exact-count
    # check, byte-identical to the pre-bridge behavior.
    if len(arch_classes) != expected_class_count and not flexible_class_count:
        surplus_covered = (
            bool(bridge_symbol_names)
            and len(arch_classes) > expected_class_count
            and class_surplus_is_bridge_covered(
                model_tree, expected_classes, bridge_symbol_names)
        )
        if not surplus_covered:
            names = [c.name for c in arch_classes]
            expected_names = [s.get("name") for s in expected_classes]
            bridge_note = ""
            if bridge_symbol_names:
                undeclared = [n for n in names if n not in bridge_symbol_names]
                bridge_note = (
                    f" Spec-declared system_provides symbols admit extra "
                    f"public classes beyond the manifest's roles; the "
                    f"class(es) not declared by any spec symbol: "
                    f"{undeclared}."
                )
            errors.append(
                f"method/model.py: found {len(arch_classes)} top-level public "
                f"class(es) ({names}); paradigm manifest declares {expected_class_count} "
                f"({expected_names}). Make extra helpers private (leading underscore) or "
                f"add the missing class(es).{bridge_note}"
            )
            # Don't early-return — continue with the classes we DID find so any
            # method/hook errors surface in the same dispatch.

    # 3. Match each expected class to a found class using the manifest's
    # `inherits:` (optional) and `required_methods:` declarations. Greedy
    # by declaration order: for each expected class, find an unmatched
    # found class that satisfies both constraints; if none, error.
    #
    # This replaces the prior hardcoded `forward()` check that assumed every
    # architecture class inherits nn.Module + defines forward — true for
    # supervised-ML paradigms (AL, KD) but false for motion_planning and
    # other non-ML paradigms whose classes have paradigm-specific methods
    # (step / step_jacobian for SystemDynamics; is_in_collision /
    # distance_to_obstacle for CollisionModel).
    # Method and inheritance constraints resolve through locally-defined
    # base classes (detr-distill 2026-07-08: a shared private `_DETR` base
    # carried forward + the probe methods, and the body-only lookup rejected
    # a correct package). The prior greedy declaration-order match is
    # replaced with a backtracking assignment so match order cannot consume
    # the only class satisfying a later, stricter declaration.
    class_map = _local_class_map(model_tree)

    candidates_by_exp = [
        [f for f in arch_classes if _class_satisfies_declaration(exp, f, class_map)]
        for exp in expected_classes
    ]

    def _perfect_assignment(i: int, used: set[int]) -> bool:
        if i == len(expected_classes):
            return True
        for f in candidates_by_exp[i]:
            if id(f) in used:
                continue
            used.add(id(f))
            if _perfect_assignment(i + 1, used):
                return True
            used.discard(id(f))
        return False

    if not _perfect_assignment(0, set()):
        for exp, candidates in zip(expected_classes, candidates_by_exp):
            if candidates:
                continue
            exp_name = exp.get("name") or "<unnamed>"
            method_names = _declared_method_names(exp)
            inherits_decl = exp.get("inherits")
            methods_str = ", ".join(method_names) if method_names else "(none declared)"
            inheritance_str = (
                f" inheriting from `{inherits_decl}`" if inherits_decl else ""
            )
            errors.append(
                f"method/model.py: paradigm manifest declares class `{exp_name}`"
                f"{inheritance_str} with required methods [{methods_str}], but "
                f"no public top-level class in model.py satisfies these "
                f"requirements (methods inherited from same-module base "
                f"classes count). Found classes: {[c.name for c in arch_classes]}"
            )
        if all(candidates_by_exp):
            # Every declaration is individually satisfiable but no one-to-one
            # assignment exists: two declarations compete for the same class.
            errors.append(
                f"method/model.py: the paradigm's declared classes "
                f"{[e.get('name') for e in expected_classes]} cannot be "
                f"matched one-to-one to the public classes "
                f"{[c.name for c in arch_classes]} — at least two "
                f"declarations are only satisfied by the same class. Give "
                f"each declared role its own class."
            )

    # 3b. Spec-derived methods without class attribution. Multi-class
    # manifests route the spec's required_model_methods into the entry-level
    # `any_class_required_methods` bucket (see build_plan.py) because the
    # spec does not say which class owns a paper-derived method. Each must
    # exist on AT LEAST ONE public class; which class *should* own it is
    # paradigm knowledge (kit territory) and the stage-4 fidelity review is
    # the backstop for wrong-class placement.
    for signature in any_class_methods:
        m = _method_name_from_signature(signature)
        if not m:
            continue
        if not any(
            _find_method_in_class_or_bases(c, m, class_map) is not None
            for c in arch_classes
        ):
            errors.append(
                f"method/model.py: the spec's paper-derived interface "
                f"requires method `{m}` on at least one public architecture "
                f"class, but none of {[c.name for c in arch_classes]} "
                f"defines or inherits it. Implement it on the class that "
                f"owns this behavior per the paper. Spec signature: "
                f"`{signature}`."
            )

    # 4. Paradigm-required hooks from spec. For multi-class paradigms, hooks
    # apply to whichever class is the "trainable" one — typically the first
    # one declared (student in KD). For single-class paradigms there's only
    # one candidate. We check: at least ONE class must satisfy each required
    # hook. If none do, error names the first class (the primary).
    hooks = _required_architecture_hooks(spec)
    primary_class = arch_classes[0]
    primary_name = primary_class.name
    if hooks["needs_forward_with_embedding"]:
        any_has_fwe = any(
            _find_method_in_class_or_bases(c, "forward_with_embedding", class_map)
            is not None
            for c in arch_classes
        )
        if not any_has_fwe:
            errors.append(
                f"method/model.py: spec lists penultimate-layer / embedding feature as essential, "
                f"but no architecture class has `forward_with_embedding(self, x) -> (logits, z)`. "
                f"Add the method to class `{primary_name}`. Without it, downstream gradient-embedding "
                f"computation breaks."
            )
    if hooks["needs_dropout_layers"]:
        # Scan ALL top-level classes, private included: dropout-at-inference
        # is a property of whatever torch module the package uses, not of the
        # manifest's public interface classes. ICRA21_HICA 2026-07-06: the
        # motion_planning manifest declares two plain-Python public classes
        # (dynamics + collision), so the producer correctly put its
        # dropout-carrying backbone in a private class — and the public-only
        # scan trapped it between the class-count check (cannot add a third
        # public class) and this check (cannot see the private one). The
        # halt-judge diagnosed the trap precisely (pipeline_bug/high).
        # (forward_with_embedding above deliberately stays public-only: it is
        # a called interface, not a module property — no live case yet.)
        all_classes = [
            node for node in model_tree.body if isinstance(node, ast.ClassDef)
        ]
        any_has_dropout = any(_class_has_dropout_layer(c) for c in all_classes)
        if not any_has_dropout:
            errors.append(
                "method/model.py: spec lists MC dropout / dropout-active-at-"
                "inference as essential, but no class in model.py (public or "
                "private) carries an nn.Dropout layer. Add an nn.Dropout "
                "layer to the torch module that produces the stochastic "
                "forward passes the spec's uncertainty estimate requires — "
                "without it, repeated forward passes are deterministic and "
                "the Monte-Carlo uncertainty collapses to a constant."
            )

    # 5. training.py: every function declared in the manifest must exist
    # as a top-level def. For active_learning that's build_model + train_from_scratch;
    # for knowledge_distillation it's build_student + build_teacher + train_with_distillation.
    found_funcs: dict[str, ast.FunctionDef | ast.AsyncFunctionDef | None] = {}
    for fname in expected_func_names:
        if fname.startswith("<"):
            # Producer-defined placeholder (the generic provisional plan's
            # `<training_functions>` row): the family is brand new, so the
            # manifest cannot pin function names. Existence/signature checks
            # for these land with the spec-driven validators downstream.
            continue
        fn = _find_top_level_def(training_tree, fname)
        if fn is None or not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
            errors.append(f"method/training.py: missing top-level `{fname}` function")
            found_funcs[fname] = None
        else:
            found_funcs[fname] = fn

    # 6. forbidden params in any of the declared training.py functions
    for fname, fn in found_funcs.items():
        if isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
            params = set(_all_param_names(fn))
            hits = sorted(params & forbidden_params)
            if hits:
                errors.append(
                    f"method/training.py: `{fname}` declares forbidden parameter(s) {hits}: "
                    f"the matched paradigm's `pluggable_component.contract.forbidden_param_names` "
                    f"is {sorted(forbidden_params)}. Use named keyword params with defaults instead."
                )

    # 6b. No EXTRA REQUIRED parameter beyond the manifest's declared training
    # signature. This is the training-function analog of the pluggable
    # extra-required check (validate_method_coder_output.py, commit 382264d46):
    # a required parameter the contract never declared is unbindable — the
    # probe harness's _fill_kwargs and the notebook driver supply arguments by
    # the contract names, so an unexpected required arg silently breaks them.
    # The GBALD run added a required `batch_train_size` to train_from_scratch
    # (the canonical signature has no batch arg); the notebook quietly aliased
    # it from batch_size, but the UB-5 trainability probe could not bind it and
    # went unprobeable — so the one check that would have surfaced the training
    # starvation never ran. Optional extras (a param WITH a default) are fine:
    # the probe/notebook fall back to the default. Only required extras break
    # binding. Functions whose contract signature ends in **kwargs (build_model)
    # are exempt — they are explicitly allowed to take extra params.
    expected_func_sigs = {
        s.get("name"): s.get("signature")
        for s in expected_funcs
        if s.get("name") and s.get("signature")
    }
    for fname, fn in found_funcs.items():
        if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        sig_str = expected_func_sigs.get(fname)
        if not sig_str:
            continue
        shape = _manifest_signature_shape(sig_str)
        if shape is None:
            continue  # unparseable contract signature — taxonomy-authoring issue, not arch-coder's
        declared_params, accepts_extra_kwargs = shape
        if accepts_extra_kwargs:
            continue
        extra_required = [
            name for name in _required_param_names(fn) if name not in declared_params
        ]
        if extra_required:
            errors.append(
                f"method/training.py: `{fname}` declares extra required parameter(s) "
                f"{sorted(extra_required)} not present in the manifest signature "
                f"`{sig_str}`. Extra required parameters make the function unbindable "
                f"by the probe harness and the notebook driver (they supply arguments "
                f"by the contract names). Give them defaults, derive them from the data "
                f"inside the function, or — if the value is a real hyperparameter — add "
                f"it to the method spec / params so it has a resolvable source."
            )

    # 7. training.py imports every architecture class it actually references
    # in a function signature. The prior version required EVERY model.py
    # class to be importable in training.py, which was correct for
    # supervised-ML paradigms (build_student needs StudentClass, etc.) but
    # wrong for motion_planning where training.py's `precompute_motion_primitives`
    # references SystemDynamics but not CollisionModel.
    referenced_in_training = _class_names_in_function_signatures(training_tree)
    arch_class_names = {c.name for c in arch_classes}
    for c_name in sorted(referenced_in_training & arch_class_names):
        if not _import_alias_for_class(training_tree, c_name):
            errors.append(
                f"method/training.py: references class `{c_name}` in a function "
                f"signature but does not import it. Add "
                f"`from .model import {c_name}` (or equivalent)."
            )

    # 8. Seed plumbing: the training-loop function (heuristic: a declared
    # function whose name does NOT start with `build_`) must call
    # `torch.manual_seed(...)` before any optimizer instantiation.
    # For active_learning: `train_from_scratch`; for knowledge_distillation:
    # `train_with_distillation`. Builder functions are exempt — they don't
    # instantiate optimizers.
    training_loop_fns = {
        n: fn for n, fn in found_funcs.items()
        if not n.startswith("build_") and isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    for fname, fn in training_loop_fns.items():
        if not _seed_called_before_optimizer(fn):
            errors.append(
                f"method/training.py: `{fname}` instantiates the optimizer before "
                f"calling `torch.manual_seed(...)`. Per C-22: seed BEFORE any optimizer/state init."
            )

    # 9. Paper-element ID validity — every `# paper-element: <id>` annotation in
    # model.py / training.py must reference an id that exists in paper_map.json.
    # If paper_map.json is missing or unparseable, skip the check with a warning;
    # that's an upstream issue.
    paper_map_ids = _load_paper_map_ids(run_dir)
    if paper_map_ids is None:
        # Don't fail; log only. Validators are permissive on upstream missing artifacts.
        pass
    else:
        for filename in ("model.py", "training.py"):
            file_path = run_dir / "method" / filename
            for lineno, pid in _extract_paper_element_ids(file_path):
                if pid not in paper_map_ids:
                    errors.append(
                        f"method/{filename}:{lineno}: `# paper-element: {pid}` references an ID "
                        f"that doesn't exist in paper_map.json. The trace from code to paper is "
                        f"broken — pick a valid id (e.g., one of the paper_map's existing element "
                        f"ids) or remove the annotation."
                    )

    # 10. Self-resolution (item 31b, sub-shape b — the ROMAN25 2026-07-14
    # `__Trajectory` vs `_Trajectory` class): every name the producer's own
    # files reference must be bound somewhere in that file or imported. An
    # unresolvable name is never shippable, and no downstream stage can fix
    # it in scope, so the producer fails while its context is still hot
    # (fail mode per the maintainer's 2026-07-14 decision). Names only, no type
    # analysis: the binding set is a whole-module over-approximation
    # (cross-scope), so dynamic patterns pass silently and only genuinely
    # UNDEFINED names fail. Files with star imports are skipped.
    for tree, filename in ((model_tree, "model.py"), (training_tree, "training.py")):
        if tree is None:
            continue
        for name, count, first_line in _undefined_name_references(tree):
            errors.append(
                f"method/{filename}:{first_line}: references `{name}` "
                f"({count} site(s)) but nothing in the file defines or "
                f"imports that name — it will raise NameError at runtime. "
                f"Fix the definition or the references so they use one "
                f"name (watch leading-underscore mismatches: a class "
                f"defined `__X` is not referenceable as `_X`)."
            )

    # 10b. Evaluation-split integrity (eval_split_aliasing, R2C-066): the
    # training loop owns supervision AND the early-stopping reference, so a
    # "validation" loss reading the training targets is this file's defect to
    # fix. Same primitive as the method-coder seam's arm.
    from scripts.eval_split_integrity import find_split_aliasing  # noqa: PLC0415

    for tree, filename in ((model_tree, "model.py"), (training_tree, "training.py")):
        if tree is None:
            continue
        for finding in find_split_aliasing(tree):
            errors.append(finding.message(f"method/{filename}"))

    # 10b.2. Reachable temporal target ranges (R2C-077): where the facts
    # available at this producer seam resolve trainer-local fitting and model
    # selection bounds, compare those consumed ranges rather than local names.
    # Caller-owned and later params-dependent relationships remain for the
    # notebook seam instead of being guessed here.
    from scripts.eval_split_validation import (  # noqa: PLC0415
        architecture_split_lineage_errors,
    )

    errors.extend(architecture_split_lineage_errors(
        spec, build_plan, run_dir
    ))

    # 10c. Aliased return tuple (aliased_return_tuple): distinct promised
    # outputs must be distinct objects — same primitive as the method-coder
    # seam's arm, on this producer's files.
    from scripts.api_surface_checks import find_aliased_return_tuples  # noqa: PLC0415

    for tree, filename in ((model_tree, "model.py"), (training_tree, "training.py")):
        if tree is None:
            continue
        for line, name in find_aliased_return_tuples(tree):
            errors.append(
                f"method/{filename}:{line}: the return tuple carries "
                f"`{name}` twice — two promised outputs are the SAME "
                f"object, so one of them is a mislabeled alias and whatever "
                f"it was supposed to carry is silently dropped. Return the "
                f"distinct quantity in each slot, or collapse the API to "
                f"one name if they are genuinely identical "
                f"(aliased_return_tuple)."
            )

    # 11. Contract public surface (item 31b, sub-shape a, the deterministic
    # slice — the ICRA 2026-07-13 private-classifier genus): the arch
    # contract's architecture blocks are consumed by the method coder, the
    # notebook generator, and the 2d dry-run; a block whose class_name is
    # underscore-private promises a component nothing downstream may import.
    arch_contract_path = run_dir / ".pipeline" / "arch_contract.json"
    try:
        contract_blocks = (
            json.loads(arch_contract_path.read_text(encoding="utf-8"))
            .get("architecture") or {}
        )
    except Exception:
        contract_blocks = {}
    for block_key, block in contract_blocks.items():
        class_name = (block or {}).get("class_name") if isinstance(block, dict) else None
        if isinstance(class_name, str) and class_name.startswith("_"):
            errors.append(
                f".pipeline/arch_contract.json: architecture.{block_key} names "
                f"the private class `{class_name}` — contract components are "
                f"user-facing by construction and must be public. Rename the "
                f"class public (drop the leading underscore) in model.py and "
                f"the contract."
            )

    # 12. Naming bridge (item 31b, the general fix for sub-shape a —
    # spec_promise_not_importable, the ICRA 2026-07-13 private-classifier
    # genus). Every spec-declared symbol must resolve to a PUBLIC top-level
    # name in the generated package: defined public in a method/ module, or
    # re-exported by method/__init__.py. `try_it_out.system_provides` is the
    # researcher-facing promise surface; a promise you cannot import is a
    # defect, and it is first broken at this producer.
    if bridge_entries:
        module_trees = _package_module_trees(run_dir, model_tree, training_tree)
        module_definitions = {
            rel: _top_level_public_definitions(tree)
            for rel, tree in module_trees.items()
        }
        module_definition_kinds = {
            rel: _top_level_definition_kinds(tree)
            for rel, tree in module_trees.items()
        }
        module_all_names = {
            rel: _top_level_names_all(tree)
            for rel, tree in module_trees.items()
        }
        # Scope-blind Store-ctx binding sets per module, so a re-export in
        # __init__.py only resolves when its source module actually binds the
        # imported name (no phantom re-exports).
        module_bindings = {
            Path(rel).stem: _module_binding_names(tree)[0]
            for rel, tree in module_trees.items()
        }
        init_bindings = _init_public_bindings(
            run_dir / "method" / "__init__.py", module_bindings)

        for entry in bridge_entries:
            symbol = entry["symbol"]
            declared_kind = entry.get("symbol_kind")
            definition_sites = sorted(
                rel for rel, defined in module_definitions.items()
                if symbol in defined
            )
            if len(definition_sites) > 1:
                # Ownership collision — fail loud, never silently dedup
                # (maintainer decision 2026-07-16): downstream consumers and the package
                # re-export cannot tell which definition the spec promises.
                errors.append(
                    f"method/ package: the spec-declared symbol `{symbol}` "
                    f"(system_provides entry `{entry['name']}`) is defined in "
                    f"{len(definition_sites)} files "
                    f"({', '.join(definition_sites)}) — ambiguous ownership. "
                    f"Keep exactly one defining module and import the symbol "
                    f"everywhere else."
                )
                continue
            if definition_sites:
                # Resolved to a public definition. When the spec declares the
                # promised surface's KIND (plan item 10, the ADAM genus),
                # check the definition's AST kind: a class promised where a
                # function is defined (or vice versa) is a promise the
                # delivered package cannot keep by rename or re-export.
                # Assignment bindings can alias anything, so only a definite
                # class-vs-function mismatch fails.
                site = definition_sites[0]
                found_kind = module_definition_kinds.get(site, {}).get(symbol)
                if (
                    declared_kind
                    and found_kind in ("class", "function")
                    and found_kind != declared_kind
                ):
                    errors.append(
                        f"method/ package: the spec's try_it_out."
                        f"system_provides entry `{entry['name']}` promises "
                        f"`{symbol}` as a {declared_kind}, but {site} defines "
                        f"it as a {found_kind} — the promised KIND of surface "
                        f"does not match the delivered one, and no rename or "
                        f"re-export can fix a kind mismatch. If the delivered "
                        f"shape is the method's honest structure, the spec's "
                        f"declared symbol_kind itself is wrong — the halt "
                        f"judge should classify that case as an upstream "
                        f"(Stage 1) issue rather than re-dispatching the "
                        f"producer. Only if the promise is right should the "
                        f"owning producer restructure the definition to the "
                        f"declared kind."
                    )
                continue
            if symbol in init_bindings:
                continue  # resolved via re-export (an alias may bind anything)
            if _later_producer_may_own(
                symbol, entry["type"], files_block, declared_kind
            ):
                # A producer that runs after this gate (method_coder's
                # method/method.py) may own the symbol — defer rather than
                # mis-attribute a not-yet-generated file to this producer.
                # Kind-aware: a `class` promise never defers to the
                # open-ended function bucket, so the ADAM shape (class-shaped
                # promises over a pure-function delivery) fails HERE, with
                # 2.b judge routing, instead of hard-halting at 2.d after
                # full generation.
                continue
            # Unresolved. Point at the underscore-private variant when one
            # exists (the ICRA shape: `_AvoidanceClassifier` defined,
            # `AvoidanceClassifier` promised).
            variant_hint = ""
            for rel in sorted(module_all_names):
                variants = sorted(
                    n for n in module_all_names[rel]
                    if n != symbol and n.lstrip("_") == symbol
                )
                if variants:
                    variant_hint = (
                        f" {rel} defines `{variants[0]}` — a private name "
                        f"researchers cannot import."
                    )
                    break
            kind_note = ""
            if declared_kind:
                kind_note = (
                    f" The spec declares this promise as a {declared_kind} "
                    f"surface; a package that delivers the capability under "
                    f"a different KIND of surface (for example a "
                    f"pure-function API where a class was promised) cannot "
                    f"satisfy it by rename — that is the upstream (Stage 1) "
                    f"case."
                )
            errors.append(
                f"method/ package: the spec's try_it_out.system_provides "
                f"entry `{entry['name']}` promises the importable symbol "
                f"`{symbol}`, but no method/ module defines it as a public "
                f"top-level name and method/__init__.py does not re-export "
                f"it.{variant_hint} The spec's promise is the "
                f"researcher-facing import surface. Two accepted remedies: "
                f"rename the component public (drop the leading underscore), "
                f"or add a re-export to method/__init__.py (e.g. "
                f"`from .model import _{symbol} as {symbol}`). If neither "
                f"remedy fits because the package genuinely has no such "
                f"artifact, the spec's declared symbol itself may be wrong — "
                f"the halt judge should classify that case as an upstream "
                f"(Stage 1) issue rather than re-dispatching the "
                f"architecture coder.{kind_note}"
            )

    return errors


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

    errors = validate(spec, args.run_dir, args.repo_root)
    if errors:
        print(f"FAIL: {len(errors)} validation error(s):", file=sys.stderr)
        for err in errors:
            print(f"  - {err}", file=sys.stderr)
        return 1

    print(f"ok: architecture_coder output at {args.run_dir} validates against the manifest.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
