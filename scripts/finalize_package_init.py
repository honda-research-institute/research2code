"""Stage 2.d — init_finalizer.

Runs after all three Stage 2 producer agents (package_scaffolder, architecture_coder,
method_coder) have produced their files. Walks the run directory's `method/` subtree
with AST, discovers public symbols across `data.py`, `model.py`, `training.py`, and
`method.py`, and assembles two derived files:

  - `<run_dir>/method/__init__.py` — re-exports every public symbol with `__all__`
    in the order declared by the manifest's `content_rule`:
      [pluggable_component.name, *method_helpers, ArchitectureClass,
       build_model, train_from_scratch, load_data]

  - `<run_dir>/requirements.txt` — paradigm-base packages plus any third-party
    imports the method's files actually use (e.g., sklearn if the method-coder
    pulled it in for clustering).

Public-symbol detection:
  - `data.py` → top-level functions whose name doesn't start with `_`
    (typically just `load_data`).
  - `model.py` → all public top-level classes inheriting from `nn.Module`
    (ignoring private/underscore-prefixed helpers). The expected count is
    declared in the taxonomy build plan's
    `package_manifest.files[method/model.py].public_symbols` — single-class
    paradigms (active_learning) declare 1, multi-class paradigms
    (knowledge_distillation: student + teacher) declare 2.
  - `training.py` → the function names declared in the paradigm manifest
    (active_learning: `build_model` + `train_from_scratch`;
    knowledge_distillation: `build_student` + `build_teacher` +
    `train_with_distillation`).
  - `method.py` → the pluggable function (named per
    `spec.comparison.pluggable_component.name`) plus any top-level
    non-private functions (the method helpers).

Naming bridge (the naming bridge design note (internal, not shipped)): the
spec's `try_it_out.system_provides` entries may declare an importable
`symbol`. This script is the finalize-time half of that contract — it
re-checks every declared symbol against the complete package (2.b defers
symbols owned by producers that had not run yet), re-exports promised
symbols the standard collection would skip, emits deterministic alias
re-exports for promises whose definition is underscore-private, and applies
the same spec-declared class-count allowance as the 2.b gate. Because the
aliases are re-derived from the spec on every run, regeneration (including
the Stage 5 re-finalization pass) never erases the remedy.

This script is fully deterministic — no LLM calls. After it runs, the
`validate_package_imports.py` validator confirms `from method import *` works
in a fresh subprocess.

Usage:

    python scripts/finalize_package_init.py --spec <method_spec.json> --run-dir <output_dir>

Exit codes:
  0  files written successfully
  1  setup error (missing spec / run dir / required Python files)
  2  no architecture class found, or other contract violation that init_finalizer
     cannot resolve mechanically
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

# Manifest-loading helper (taxonomy build plan → package_manifest block).
# The same helper validate_architecture_coder_output.py uses; keeps the two
# scripts' view of the manifest in sync.
from scripts.build_plan import load_build_plan  # noqa: E402
from scripts.taxonomy import run_overlay_dir  # noqa: E402
# Naming-bridge helpers shared with the 2.b gate, so both gates read the
# spec's declared symbols and the class-count allowance identically
# (the naming bridge design note (internal, not shipped)). The finalizer is the
# re-check point for symbols 2.b deferred to later producers, and it is
# where both accepted remedies must SURVIVE: this script regenerates
# method/__init__.py, so it re-derives the private-definition alias
# re-exports deterministically instead of erasing them.
from scripts.validate_architecture_coder_output import (  # noqa: E402
    _spec_bridge_entries,
    _top_level_defined_names,
    _top_level_definition_kinds,
    class_surplus_is_bridge_covered,
)


# Import-name → pip-package-name mapping for cases where they differ.
# Most third-party packages match their import name, so we only list exceptions.
PIP_NAME_OVERRIDES = {
    "sklearn": "scikit-learn",
    "cv2": "opencv-python",
    "PIL": "pillow",
    "yaml": "pyyaml",
    "skimage": "scikit-image",
}

# Default base requirements applied to every paradigm today. The notebook
# imports matplotlib + jupyter even though method/ files don't, so they
# must be in requirements.txt for the user's first install to succeed.
# Paradigm-specific deps (e.g., scipy for KD, PIL for image-based paradigms)
# are added on top via the `detected_imports` scan in _build_requirements_txt.
#
# The floor is restricted to deps EVERY generated package needs regardless of
# paradigm: numeric (numpy), tensors (torch), plotting (matplotlib), notebook
# execution (jupyter). Paradigm-specific deps are added on top via the
# `detected_imports` scan in _build_requirements_txt — including `torchvision`,
# which the image paradigms (active_learning, knowledge_distillation) import in
# their data.py (MNIST/CIFAR loaders) and which the scan therefore re-adds for
# them. It was previously in the universal floor, which forced a dead
# torchvision dep onto non-image paradigms like motion_planning (PA-D9).
#
# TODO: when a paradigm needs a substantially different base, lift this into
# the paradigm's taxonomy `package_manifest` (e.g., a `base_requirements:`
# block) keyed off the already-passed `paradigm_id`.
DEFAULT_BASE_REQUIREMENTS = [
    ("numpy", ">=1.24"),
    ("torch", ">=2.0"),
    ("matplotlib", ">=3.7"),
    ("jupyter", ">=1.0"),
]
# Backward-compat alias — older callers may import the original name.
AL_BASE_REQUIREMENTS = DEFAULT_BASE_REQUIREMENTS

# stdlib top-level module names — we skip these when computing requirements.
# Falls back to a hardcoded list if sys.stdlib_module_names is unavailable
# (Python < 3.10). The hardcoded set covers everything an AL method might use.
try:
    _STDLIB = set(sys.stdlib_module_names)  # type: ignore[attr-defined]
except AttributeError:
    _STDLIB = {
        "abc", "argparse", "ast", "collections", "concurrent", "contextlib", "copy",
        "csv", "dataclasses", "datetime", "enum", "functools", "io", "itertools",
        "json", "logging", "math", "multiprocessing", "operator", "os", "pathlib",
        "pickle", "random", "re", "shutil", "string", "subprocess", "sys", "tempfile",
        "textwrap", "threading", "time", "traceback", "types", "typing", "uuid", "warnings",
    }


# ---------------------------------------------------------------------------
# AST helpers — public symbol discovery
# ---------------------------------------------------------------------------


def _parse_or_die(path: Path) -> ast.Module:
    if not path.is_file():
        raise FileNotFoundError(f"required file not found: {path}")
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _public_top_level_functions(tree: ast.Module) -> list[str]:
    """Top-level function names that don't start with underscore."""
    return [
        node.name
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and not node.name.startswith("_")
    ]


def _find_public_top_level_class_names(tree: ast.Module) -> list[str]:
    """Find all top-level public class names in source order — regardless of
    inheritance. Empty if there are none.

    Replaces a prior helper that filtered to
    nn.Module subclasses (a supervised-ML assumption that counted
    motion_planning's plain-Python SystemDynamics + CollisionModel as zero
    classes and halted the finalizer). The caller cross-references the count
    against the paradigm manifest's declared `kind: class` symbols and uses
    the names for __init__.py's re-exports; the manifest is the source of
    truth for how many classes to expect, not the inheritance heuristic.

    For single-class paradigms (active_learning) this returns a 1-element
    list; for multi-class paradigms (knowledge_distillation: student +
    teacher; motion_planning: dynamics + collision model) it returns the
    declared classes in declaration order."""
    return [
        node.name
        for node in tree.body
        if isinstance(node, ast.ClassDef)
        and not node.name.startswith("_")
    ]


# ---------------------------------------------------------------------------
# AST helpers — third-party import discovery
# ---------------------------------------------------------------------------


def _top_level_imports(tree: ast.Module) -> set[str]:
    """Return the set of top-level imported module names (e.g., 'torch', 'numpy', 'sklearn')."""
    out: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                out.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            # Skip relative imports (from .model import ...)
            if node.level > 0:
                continue
            if node.module:
                out.add(node.module.split(".")[0])
    return out


# The generated package's own name — imported by the notebook (`from method
# import ...`) and sometimes absolutely by the package's own modules (`from
# method.model import ...` — legal, producer import-style variance) but never
# a pip dependency, so BOTH third-party scans must exclude it (the DomIndOnto
# 2026-07-21 pip failure: `method` landed in requirements.txt).
_LOCAL_PACKAGE_NAME = "method"


def _detect_third_party_imports(method_dir: Path) -> set[str]:
    """Walk method/*.py and collect third-party import names (skipping
    stdlib, relative, and the local package itself)."""
    detected: set[str] = set()
    # Sibling modules of the package itself: a producer writing a bare
    # `import model` for method/model.py must not turn the sibling into a
    # pip requirement — pip aborts the whole stage on the nonexistent
    # distribution (DomIndOnto 2026-07-28: `model` reached pip and halted
    # stage 2.d with an internal contract violation).
    local_modules = {p.stem for p in method_dir.glob("*.py")}
    for py_path in sorted(method_dir.glob("*.py")):
        if py_path.name == "__init__.py":
            continue
        try:
            tree = ast.parse(py_path.read_text(encoding="utf-8"), filename=str(py_path))
        except SyntaxError:
            continue
        for name in _top_level_imports(tree):
            if name in _STDLIB:
                continue
            if name.startswith("_"):
                continue
            if name == _LOCAL_PACKAGE_NAME:
                continue
            if name in local_modules:
                continue
            detected.add(name)
    return detected


def _detect_notebook_third_party_imports(notebook_path: Path) -> set[str]:
    """Third-party top-level imports across a rendered notebook's code cells,
    using the same stdlib/relative/private filtering as the method-package
    scan. IPython magics and shell-escape lines (`%pip`, `!pip`, `%matplotlib`)
    are stripped before parsing — they are not valid Python. The local `method`
    package is excluded: it is the generated package, not a pip dependency."""
    detected: set[str] = set()
    try:
        nb = json.loads(notebook_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return detected
    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        src = cell.get("source", "")
        if isinstance(src, list):
            src = "".join(src)
        src = "\n".join(
            line for line in src.splitlines()
            if not line.lstrip().startswith(("%", "!"))
        )
        try:
            tree = ast.parse(src)
        except SyntaxError:
            continue
        for name in _top_level_imports(tree):
            if name in _STDLIB or name.startswith("_") or name == _LOCAL_PACKAGE_NAME:
                continue
            detected.add(name)
    return detected


def reconcile_requirements_for_notebook(
    run_dir: Path, spec_path: Path,
) -> tuple[bool, list[str]]:
    """Re-derive `<run_dir>/requirements.txt` so it ALSO covers the rendered
    notebook's third-party imports.

    `requirements.txt` is built at Stage 2.d from the method package alone, but
    the notebook is authored one stage later (3.a) and can import a library the
    package never used — e.g. scikit-learn for a plot. Left unreconciled, a
    clean `pip install -r requirements.txt` then run hits an ImportError (the
    F001 draft-demotion class, 2026-06-15). Idempotent: re-derives from the
    union of method + notebook imports and only rewrites on a real change.

    Best-effort: returns `(False, [])` if the spec, method dir, or rendered
    notebook is missing/unreadable. Returns `(changed, added_requirement_lines)`."""
    notebook = run_dir / "notebook.ipynb"
    method_dir = run_dir / "method"
    req_path = run_dir / "requirements.txt"
    if not notebook.is_file() or not method_dir.is_dir() or not spec_path.is_file():
        return False, []
    try:
        spec = json.loads(spec_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False, []
    paradigm_id = (spec.get("comparison") or {}).get("classification", {}).get("id", "")
    detected = (
        _detect_third_party_imports(method_dir)
        | _detect_notebook_third_party_imports(notebook)
    )
    new_text = _build_requirements_txt(detected, paradigm_id)
    old_text = req_path.read_text(encoding="utf-8") if req_path.is_file() else ""
    if new_text == old_text:
        return False, []
    old_lines = {ln.strip() for ln in old_text.splitlines() if ln.strip()}
    added = [ln.strip() for ln in new_text.splitlines()
             if ln.strip() and ln.strip() not in old_lines]
    req_path.write_text(new_text, encoding="utf-8")
    return True, added


# ---------------------------------------------------------------------------
# File writers
# ---------------------------------------------------------------------------


def _build_init_py(
    public_api: list[str],
    pluggable_name: str,
    arch_class_names: list[str],
    training_func_names: list[str],
    data_func_names: list[str],
    bridge_aliases: list[tuple[str, str, str]] = (),
) -> str:
    """Render method/__init__.py text. `public_api` is the already-ordered
    list of every symbol re-exported; the other args drive which symbols
    are imported from which source module.

    Multi-class paradigms (KD: student + teacher) pass multiple
    `arch_class_names`; single-class paradigms (AL) pass one. Same shape
    for training functions: AL passes `["build_model", "train_from_scratch"]`,
    KD passes `["build_student", "build_teacher", "train_with_distillation"]`.

    `bridge_aliases` is the naming bridge's private-definition remedy:
    `(module_kind, private_name, public_name)` triples rendered as
    `from .<module_kind> import <private_name> as <public_name>` so a
    spec-promised symbol whose definition is underscore-private stays
    importable from the package front door. Re-derived from the spec on
    every finalize, so the remedy survives regeneration (including the
    Stage 5 re-finalization pass)."""
    alias_names = [public_name for _, _, public_name in bridge_aliases]
    # The spec-promised pluggable is owned by method/method.py: no other
    # module's import list may carry its name, or the later `from .training
    # import ...` line would silently rebind the public name to a different
    # function (the fedavg 2026-08-04 defect,
    # exported_pluggable_resolves_to_wrong_module). A naming-bridge alias is
    # the one sanctioned other source (the private-definition remedy).
    arch_class_imports = [n for n in arch_class_names if n != pluggable_name]
    training_imports = [n for n in training_func_names if n != pluggable_name]
    data_imports = [n for n in data_func_names if n != pluggable_name]
    for src, names in (("model", arch_class_names), ("training", training_func_names),
                       ("data", data_func_names)):
        if pluggable_name in names:
            print(
                f"note: {src}.py also defines public `{pluggable_name}`; the "
                f"package export binds the method-module pluggable and the "
                f"{src} version is not re-exported "
                f"(exported_pluggable_resolves_to_wrong_module guard)."
            )
    # __all__ must list each public name once; the ordering rule puts the
    # pluggable first, so dedup-by-first-occurrence keeps the promised order.
    public_api = list(dict.fromkeys(public_api))
    # Method imports are everything in public_api that isn't from the other modules.
    other = set(arch_class_imports + training_imports + data_imports + alias_names)
    method_imports = [name for name in public_api if name not in other]

    lines = [
        '"""Method package — re-exports the public API.',
        "",
        "Generated by `scripts/finalize_package_init.py` after the Stage 2 producer agents",
        "complete. The `__all__` list below is ordered per the taxonomy build plan's",
        "`package_manifest.files[__init__.py].content_rule`:",
        "  [pluggable_component.name, *method_helpers, *architecture_classes,",
        "   *training_functions, load_data].",
        '"""',
        "",
    ]

    if data_imports:
        lines.append(f"from .data import {', '.join(sorted(data_imports))}")
    if method_imports:
        if len(method_imports) == 1:
            lines.append(f"from .method import {method_imports[0]}")
        else:
            lines.append("from .method import (")
            for name in sorted(method_imports):
                lines.append(f"    {name},")
            lines.append(")")
    if arch_class_imports:
        if len(arch_class_imports) == 1:
            lines.append(f"from .model import {arch_class_imports[0]}")
        else:
            lines.append(f"from .model import {', '.join(arch_class_imports)}")
    if training_imports:
        lines.append(f"from .training import {', '.join(sorted(training_imports))}")
    if bridge_aliases:
        lines.append(
            "# Spec-promised symbols re-exported from private definitions "
            "(naming bridge)."
        )
        for module_kind, private_name, public_name in bridge_aliases:
            lines.append(
                f"from .{module_kind} import {private_name} as {public_name}"
            )

    lines.append("")
    lines.append("__all__ = [")
    for name in public_api:
        lines.append(f'    "{name}",')
    lines.append("]")
    lines.append("")
    return "\n".join(lines)


_PIP_UNRESOLVABLE_PATTERNS = (
    re.compile(
        r"could not find a version that satisfies the requirement\s+(\S+)",
        re.IGNORECASE,
    ),
    re.compile(r"no matching distribution found for\s+(\S+)", re.IGNORECASE),
)


def unresolvable_requirement_names(pip_output: str) -> list[str]:
    """Distribution names pip's RESOLVER rejected, parsed from its output.

    R2C-050. Version specifiers, extras, and environment markers are
    stripped so `dgl>=1.0` and `dgl` identify the same distribution.
    De-duplicated case-insensitively, in first-seen order, because pip
    prints both resolver phrases for the same name."""
    names: list[str] = []
    seen: set[str] = set()
    for pattern in _PIP_UNRESOLVABLE_PATTERNS:
        for match in pattern.finditer(pip_output or ""):
            token = match.group(1).strip().strip("'\"()")
            name = re.split(r"[\s;<>=!~\[]", token, maxsplit=1)[0].rstrip(",")
            key = name.lower().replace("_", "-")
            if not name or key in seen:
                continue
            seen.add(key)
            names.append(name)
    return names


def _import_names_for_requirement(requirement: str) -> set[str]:
    """Import names that would have put `requirement` into requirements.txt.

    The inverse of the PIP_NAME_OVERRIDES mapping the builder applies
    (import `cv2` becomes requirement `opencv-python`), plus the identity
    and the dash-to-underscore convention. Comparison is case-insensitive
    because pip distribution names are."""
    wanted = requirement.lower().replace("_", "-")
    names = {
        imp
        for imp, pip_name in PIP_NAME_OVERRIDES.items()
        if pip_name.lower().replace("_", "-") == wanted
    }
    names.add(requirement)
    names.add(requirement.replace("-", "_"))
    return names


def trace_requirement_to_producers(
    requirement: str,
    method_dir: Path,
    manifest_files: list[dict] | None,
) -> dict | None:
    """Which producer imports put `requirement` into requirements.txt.

    R2C-050. requirements.txt is reconciled FROM the producers' own imports,
    so an unresolvable requirement traces back to the file and producer that
    chose the library — and that producer, not engineering, is who can act
    (the 2026-08-03 pdfgnn halt: the architecture coder imported dgl, no
    distribution exists for this platform, and the halt shipped the
    report-to-engineering story).

    Returns None when no producer file imports it, which is exactly the
    finalizer-artifact case (the DomIndOnto 2026-07-21 leak of the local
    package name) where the internal-bug classification is CORRECT and must
    be kept. Ownership comes from the build-plan manifest's `produced_by`;
    a file the manifest does not list yields no owner rather than a guess."""
    import_names = {n.lower() for n in _import_names_for_requirement(requirement)}
    owners_by_path = {
        str(f.get("path") or ""): str(f.get("produced_by") or "")
        for f in (manifest_files or [])
    }

    files: list[str] = []
    producers: set[str] = set()
    if not method_dir.is_dir():
        return None
    for py_path in sorted(method_dir.glob("*.py")):
        if py_path.name == "__init__.py":
            continue
        try:
            tree = ast.parse(py_path.read_text(encoding="utf-8"), filename=str(py_path))
        except (SyntaxError, OSError):
            continue
        if not any(name.lower() in import_names for name in _top_level_imports(tree)):
            continue
        rel = f"method/{py_path.name}"
        files.append(rel)
        owner = owners_by_path.get(rel)
        if owner:
            producers.add(owner)

    if not files:
        return None
    return {
        "requirement": requirement,
        "files": files,
        "producers": sorted(producers),
    }


def _build_requirements_txt(detected_imports: set[str], paradigm_id: str) -> str:
    """Render requirements.txt text. Universal base + detected third-party
    extras. See `DEFAULT_BASE_REQUIREMENTS`'s docstring for the per-paradigm
    extension plan when bases need to diverge."""
    base = DEFAULT_BASE_REQUIREMENTS

    base_names = {name for name, _ in base}

    # Compute extras: detected imports that aren't already in base
    extras: list[tuple[str, str]] = []
    for imp in sorted(detected_imports):
        pip_name = PIP_NAME_OVERRIDES.get(imp, imp)
        if pip_name in base_names:
            continue
        extras.append((pip_name, ""))

    out: list[str] = []
    for name, version_spec in base + extras:
        if version_spec:
            out.append(f"{name}{version_spec}")
        else:
            out.append(name)
    out.append("")  # trailing newline
    return "\n".join(out)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def finalize(spec_path: Path, run_dir: Path) -> int:
    if not spec_path.is_file():
        print(f"error: spec not found: {spec_path}", file=sys.stderr)
        return 1
    spec = json.loads(spec_path.read_text(encoding="utf-8"))

    method_dir = run_dir / "method"
    if not method_dir.is_dir():
        print(f"error: method/ directory not found at {method_dir}", file=sys.stderr)
        return 1

    paths = {
        "data": method_dir / "data.py",
        "model": method_dir / "model.py",
        "training": method_dir / "training.py",
        "method": method_dir / "method.py",
    }
    for kind, p in paths.items():
        if not p.is_file():
            print(f"error: required file missing: method/{p.name} (from {kind} stage)", file=sys.stderr)
            return 1

    try:
        trees = {kind: _parse_or_die(p) for kind, p in paths.items()}
    except SyntaxError as e:
        print(f"error: failed to parse a method/ file ({e})", file=sys.stderr)
        return 1

    # Discover symbols
    pluggable_name = (spec.get("comparison") or {}).get("pluggable_component", {}).get("name")
    if not pluggable_name:
        print("error: spec.comparison.pluggable_component.name missing", file=sys.stderr)
        return 1

    # Load the package plan to discover expected class count, function names,
    # and ownership.
    build_plan = load_build_plan(
        spec, ROOT, provisional_packs_dir=run_overlay_dir(run_dir))
    if build_plan is None:
        paradigm_id = (spec.get("comparison") or {}).get("classification", {}).get("id")
        print(f"error: no taxonomy build plan for comparison.classification.id={paradigm_id!r}", file=sys.stderr)
        return 1
    manifest = build_plan.get("package_manifest") or {}
    expected_arch_class_count = 0
    expected_class_symbols: list[dict] = []
    expected_training_funcs: list[str] = []
    expected_data_funcs: list[str] = []
    flexible_class_count = False
    training_placeholder = False
    training_declared_empty = False
    for f in (manifest.get("files") or []):
        rel = f.get("path") or ""
        if rel == "method/model.py" and f.get("class_count") == "flexible":
            flexible_class_count = True
        if rel == "method/training.py" and f.get("public_symbols") == []:
            # A present-and-EMPTY list is a declaration, not an omission:
            # the pack says this paradigm exports nothing from training.py
            # (DomIndOnto KBP 2026-07-29 — a docstring-only training.py is
            # the correct package shape, not an R2 violation).
            training_declared_empty = True
        for sym in (f.get("public_symbols") or []):
            kind = sym.get("kind")
            name = sym.get("name") or ""
            # `<placeholder>` names are producer-defined rows from the
            # generic provisional plan (item 4) — a brand-new family's
            # function names are unknowable at plan time. The 2b validator
            # skips them; taking them literally here halted SRL's 2026-07-15
            # gap run at 2d ("missing required function <training_functions>").
            if rel == "method/model.py" and kind == "class":
                expected_arch_class_count += 1
                expected_class_symbols.append(sym)
            elif rel == "method/training.py" and kind == "function" and name:
                if name.startswith("<"):
                    training_placeholder = True
                else:
                    expected_training_funcs.append(name)
            elif rel == "method/data.py" and kind == "function" and name:
                if not name.startswith("<"):
                    expected_data_funcs.append(name)
    if expected_arch_class_count < 1:
        print(
            f"error: paradigm manifest declares no architecture classes for "
            f"method/model.py (taxonomy build-plan bug)",
            file=sys.stderr,
        )
        return 1
    if (not expected_training_funcs and not training_placeholder
            and not training_declared_empty):
        print(
            f"error: paradigm manifest declares no functions for "
            f"method/training.py (taxonomy build-plan bug)",
            file=sys.stderr,
        )
        return 1
    # data.py functions: fall back to ["load_data"] if the manifest doesn't
    # declare any (the scaffolder template owns data.py; older paradigm
    # manifests may not list its public_symbols explicitly).
    if not expected_data_funcs:
        expected_data_funcs = ["load_data"]

    if training_placeholder and not expected_training_funcs:
        # Producer-defined training surface: re-export what the arch coder
        # actually wrote (all public top-level functions of training.py).
        expected_training_funcs = sorted(
            _public_top_level_functions(trees["training"]))
        if not expected_training_funcs:
            print(
                "error: training.py defines no public top-level functions "
                "(architecture_coder violated R2).",
                file=sys.stderr,
            )
            return 2

    # Naming-bridge declarations (the naming bridge design note, internal,
    # not shipped): the spec's promised importable symbols.
    # Consulted twice below — the class-count allowance mirrors the 2.b
    # gate's, and after the public API is assembled every declared symbol is
    # re-checked against the COMPLETE package (2.b defers symbols owned by
    # producers that had not run yet; deferral is never terminal).
    bridge_entries = _spec_bridge_entries(spec)
    # Class-count allowance set, mirroring the 2.b gate's kind filter (plan
    # item 10): only kind-undeclared or declared-`class` symbols can admit
    # an extra public class; a declared `function` symbol frees no slot.
    bridge_symbol_names = {
        e["symbol"] for e in bridge_entries
        if e.get("symbol_kind") in (None, "class")
    }

    arch_class_names = _find_public_top_level_class_names(trees["model"])
    if flexible_class_count:
        # Generic provisional plan: at least one public class, no exact count.
        if not arch_class_names:
            print(
                "error: model.py defines no public top-level class "
                "(architecture_coder violated R1).",
                file=sys.stderr,
            )
            return 2
    elif len(arch_class_names) != expected_arch_class_count:
        # Same allowance as the 2.b gate: an extra public class is legal
        # exactly when a spec-declared bridge symbol resolves to it
        # (assignment-based, so a symbol naming a manifest role class frees
        # no slot for an undeclared stray). Without this, the rename-public
        # remedy would pass 2.b and then die right here.
        surplus_covered = (
            bool(bridge_symbol_names)
            and len(arch_class_names) > expected_arch_class_count
            and class_surplus_is_bridge_covered(
                trees["model"], expected_class_symbols, bridge_symbol_names)
        )
        if not surplus_covered:
            bridge_note = ""
            if bridge_symbol_names:
                undeclared = [
                    n for n in arch_class_names if n not in bridge_symbol_names
                ]
                bridge_note = (
                    f" Spec-declared system_provides symbols admit extra "
                    f"public classes beyond the manifest's roles; the "
                    f"class(es) not declared by any spec symbol: {undeclared}."
                )
            print(
                f"error: model.py defines {len(arch_class_names)} public top-level "
                f"class(es) ({arch_class_names}); paradigm manifest "
                f"declares {expected_arch_class_count} (architecture_coder violated R1)."
                f"{bridge_note}",
                file=sys.stderr,
            )
            return 2

    method_top_level = _public_top_level_functions(trees["method"])
    if pluggable_name not in method_top_level:
        print(
            f"error: pluggable function `{pluggable_name}` not found at method/method.py top "
            f"level (method_coder violated R1).",
            file=sys.stderr,
        )
        return 2

    method_helpers = sorted(name for name in method_top_level if name != pluggable_name)

    # Build ordered public API per manifest content_rule:
    #   [pluggable_component.name, *method_helpers, *architecture_classes,
    #    *training_functions, *data_functions]
    public_api = (
        [pluggable_name]
        + method_helpers
        + list(arch_class_names)
        + list(expected_training_funcs)
        + list(expected_data_funcs)
    )

    # Sanity-check that every manifest-declared training + data function exists
    training_top_level = _public_top_level_functions(trees["training"])
    for required in expected_training_funcs:
        if required not in training_top_level:
            print(
                f"error: training.py is missing required function `{required}` "
                f"(architecture_coder violated R2).",
                file=sys.stderr,
            )
            return 2
    data_top_level = _public_top_level_functions(trees["data"])
    for required in expected_data_funcs:
        if required not in data_top_level:
            print(
                f"error: data.py is missing required function `{required}` "
                f"(package_scaffolder template drift?).",
                file=sys.stderr,
            )
            return 2

    # Naming bridge, the finalize-time pass. The whole package exists here,
    # so this is where 2.b's deferrals are re-checked and where both accepted
    # remedies must survive regeneration:
    #   - a declared symbol already re-exported by the standard collection
    #     (public class in model.py, public function in method.py, a
    #     manifest-declared training/data function) needs nothing;
    #   - a declared symbol defined PUBLIC in a module the standard
    #     collection skips (an extra training/data function, a class in
    #     method.py) gets added to that module's re-export list;
    #   - a declared symbol whose only definition is underscore-PRIVATE gets
    #     a deterministic alias re-export (`from .model import _X as X`) —
    #     the alias is re-derived from the spec on every finalize, so a 2.b
    #     fix or a Stage 5 re-finalization never erases it;
    #   - a declared symbol with NO definition anywhere fails loud: the
    #     promise cannot be kept in the delivered package.
    module_kinds = ("data", "model", "training", "method")
    all_defs = {kind: _top_level_defined_names(trees[kind]) for kind in module_kinds}
    public_defs = {
        kind: {n for n in names if not n.startswith("_")}
        for kind, names in all_defs.items()
    }
    def_kinds = {
        kind: _top_level_definition_kinds(trees[kind]) for kind in module_kinds
    }

    def _kind_mismatch_error(
        entry: dict, defined_name: str, module_kind: str, found_kind: str
    ) -> None:
        """The declared symbol_kind contradicts the definition's AST kind
        (plan item 10, the ADAM genus) — no rename or re-export can fix a
        kind mismatch, so this fails loud with the same upstream-pointing
        guidance as the 2.b gate."""
        print(
            f"error: the spec's try_it_out.system_provides entry "
            f"`{entry['name']}` promises `{entry['symbol']}` as a "
            f"{entry['symbol_kind']}, but method/{module_kind}.py defines "
            f"`{defined_name}` as a {found_kind} — the promised KIND of "
            f"surface does not match the delivered one, and no rename or "
            f"re-export can fix a kind mismatch. If the delivered shape is "
            f"the method's honest structure, the spec's declared symbol_kind "
            f"itself is wrong — the halt judge should classify that case as "
            f"an upstream (Stage 1) issue rather than re-dispatching the "
            f"producer.",
            file=sys.stderr,
        )

    bridge_aliases: list[tuple[str, str, str]] = []
    extra_exports: dict[str, list[str]] = {kind: [] for kind in module_kinds}
    bridge_extra_api: list[str] = []
    for entry in bridge_entries:
        symbol = entry["symbol"]
        declared_kind = entry.get("symbol_kind")
        public_owners = [k for k in module_kinds if symbol in public_defs[k]]
        # Kind re-check on a uniquely-owned public definition (plan item 10)
        # BEFORE any resolution shortcut: the whole package exists here, so
        # this is where a kind mismatch 2.b could not see (a later producer's
        # file) surfaces. Assignment bindings satisfy any declared kind.
        if declared_kind and len(public_owners) == 1:
            found_kind = def_kinds[public_owners[0]].get(symbol)
            if found_kind in ("class", "function") and found_kind != declared_kind:
                _kind_mismatch_error(entry, symbol, public_owners[0], found_kind)
                return 2
        if symbol in public_api:
            continue  # already re-exported by the standard collection
        if len(public_owners) > 1:
            # Ownership collision — fail loud, never silently dedup (the maintainer
            # 2026-07-16). 2.b catches this for files it sees; repeating it
            # here covers post-2.b drift (Stage 5 re-finalization).
            sites = ", ".join(f"method/{k}.py" for k in public_owners)
            print(
                f"error: the spec-declared symbol `{symbol}` (system_provides "
                f"entry `{entry['name']}`) is defined in {len(public_owners)} "
                f"files ({sites}) — ambiguous ownership. Keep exactly one "
                f"defining module and import the symbol everywhere else.",
                file=sys.stderr,
            )
            return 2
        if public_owners:
            extra_exports[public_owners[0]].append(symbol)
            bridge_extra_api.append(symbol)
            continue
        variant_sites = [
            (kind, name)
            for kind in module_kinds
            for name in sorted(all_defs[kind])
            if name != symbol and name.startswith("_") and name.lstrip("_") == symbol
        ]
        if len(variant_sites) > 1:
            sites = ", ".join(f"`{n}` in method/{k}.py" for k, n in variant_sites)
            print(
                f"error: the spec-declared symbol `{symbol}` has multiple "
                f"private definition candidates ({sites}) — ambiguous "
                f"ownership, cannot emit a single re-export. Keep exactly "
                f"one definition.",
                file=sys.stderr,
            )
            return 2
        if variant_sites:
            kind, private_name = variant_sites[0]
            if declared_kind:
                # The alias remedy re-exports the private definition under
                # the promised name — a wrong-kind definition would keep the
                # name and break the promise's shape, so the kind check
                # applies before the alias is emitted (plan item 10).
                found_kind = def_kinds[kind].get(private_name)
                if (
                    found_kind in ("class", "function")
                    and found_kind != declared_kind
                ):
                    _kind_mismatch_error(entry, private_name, kind, found_kind)
                    return 2
            bridge_aliases.append((kind, private_name, symbol))
            bridge_extra_api.append(symbol)
            continue
        # A promised class is an architecture surface, same axis as the
        # existing `type: model` rule (plan item 10 extends it to declared
        # `symbol_kind: class` promises).
        attributed = (
            "architecture_coder"
            if entry["type"] == "model" or declared_kind == "class"
            else "method_coder"
        )
        kind_note = ""
        if declared_kind:
            kind_note = (
                f" The spec declares this promise as a {declared_kind} "
                f"surface; a package that delivers the capability under a "
                f"different KIND of surface (for example a pure-function API "
                f"where a class was promised) cannot satisfy it by rename — "
                f"that is the upstream (Stage 1) case."
            )
        print(
            f"error: the spec's try_it_out.system_provides entry "
            f"`{entry['name']}` promises the importable symbol `{symbol}`, "
            f"but no method/ module defines it, public or private — the "
            f"naming-bridge promise cannot be kept in the delivered package "
            f"({attributed} violated the spec surface). Define the symbol in "
            f"the owning module. If the package genuinely has no such "
            f"artifact, the spec's declared symbol itself may be wrong — the "
            f"halt judge should classify that case as an upstream (Stage 1) "
            f"issue rather than re-dispatching the producer.{kind_note}",
            file=sys.stderr,
        )
        return 2

    # Bridge extras and aliases land at the end of the public API, after the
    # content_rule-ordered standard symbols.
    public_api = public_api + bridge_extra_api
    training_export_names = list(expected_training_funcs) + extra_exports["training"]
    data_export_names = list(expected_data_funcs) + extra_exports["data"]
    # model.py extras (a public non-class definition a symbol promises) join
    # the .model import line; method.py extras flow through the public_api
    # remainder into the .method import block.
    model_export_names = list(arch_class_names) + extra_exports["model"]

    # Detect third-party imports
    detected = _detect_third_party_imports(method_dir)
    paradigm_id = (spec.get("comparison") or {}).get("classification", {}).get("id", "")

    # Render and write
    init_text = _build_init_py(
        public_api, pluggable_name,
        arch_class_names=model_export_names,
        training_func_names=training_export_names,
        data_func_names=data_export_names,
        bridge_aliases=bridge_aliases,
    )
    requirements_text = _build_requirements_txt(detected, paradigm_id)

    init_path = method_dir / "__init__.py"
    requirements_path = run_dir / "requirements.txt"

    init_path.write_text(init_text, encoding="utf-8")
    requirements_path.write_text(requirements_text, encoding="utf-8")

    print(f"wrote {init_path.relative_to(run_dir.parent)} ({len(public_api)} public symbols)")
    print(f"wrote {requirements_path.relative_to(run_dir.parent)} (detected: {sorted(detected)})")
    print()
    print("Public API order:")
    for name in public_api:
        print(f"  {name}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    return finalize(args.spec, args.run_dir)


if __name__ == "__main__":
    sys.exit(main())
