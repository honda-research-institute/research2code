"""Stage 2.a — validator for package_scaffolder output.

Runs after `scaffold_package.py`. Cross-references the run directory against
the matched taxonomy build plan's `package_manifest.files` block
and confirms:

  1. Every file declared with `produced_by: package_scaffolder` exists.
  2. Each Python file parses without SyntaxError.
  3. For each paradigm_fixed file with declared `public_symbols`, the symbol
     is present at top level with the right `kind` (function | class | etc.)
     and matches the declared `signature` modulo whitespace and type-hint
     formatting.
  4. The scaffolder's slot rendering ran clean: no `{{ name }}` placeholders
     remain in any rendered file (those would cause downstream syntax errors).

This is a deterministic gate. On failure, exit 1 with a structured error list
on stderr; the orchestrator re-dispatches the scaffolder with the errors
verbatim.

Usage:

    python scripts/validate_scaffolder_output.py --spec <method_spec.json> --run-dir <output_dir>

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

from scripts.build_plan import load_build_plan  # noqa: E402
from scripts.taxonomy import run_overlay_dir  # noqa: E402
from scripts.signature_ast import _normalize_sig, _signature_string  # noqa: E402


SLOT_PATTERN = re.compile(r"\{\{\s*[a-zA-Z_][a-zA-Z0-9_]*\s*\}\}")
SCAFFOLDER_ROLE = "package_scaffolder"


# ---------------------------------------------------------------------------
# AST helpers
# ---------------------------------------------------------------------------


def _find_top_level_def(tree: ast.Module, name: str) -> ast.AST | None:
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node.name == name:
            return node
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == name:
                    return node
    return None


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate(spec: dict, run_dir: Path, repo_root: Path) -> list[str]:
    """Return a list of error strings; empty means validation passed."""
    errors: list[str] = []

    build_plan = load_build_plan(
        spec, repo_root, provisional_packs_dir=run_overlay_dir(run_dir))
    if build_plan is None:
        paradigm_id = (spec.get("comparison") or {}).get("classification", {}).get("id")
        return [f"no taxonomy build plan for comparison.classification.id={paradigm_id!r}"]
    manifest = build_plan.get("package_manifest") or {}

    files_block = manifest.get("files")
    if not isinstance(files_block, list):
        return ["`package_manifest.files` is missing or not a list"]

    # Collect declarations the scaffolder is responsible for.
    scaffolder_files = [f for f in files_block if f.get("produced_by") == SCAFFOLDER_ROLE]

    if not scaffolder_files:
        return [f"manifest declares no files with `produced_by: {SCAFFOLDER_ROLE}`"]

    for entry in scaffolder_files:
        rel = entry.get("path")
        if not rel:
            errors.append(f"manifest entry missing `path`: {entry!r}")
            continue
        target = run_dir / rel

        # 1. File presence
        if not target.is_file():
            errors.append(f"{SCAFFOLDER_ROLE} did not produce {rel} at {target}")
            continue

        text = target.read_text(encoding="utf-8")

        # 2. No unrendered slots
        leftover_slots = SLOT_PATTERN.findall(text)
        if leftover_slots:
            errors.append(
                f"{rel} contains unrendered slot(s) {sorted(set(leftover_slots))} — "
                f"the scaffolder failed to substitute them. Re-run with a spec that "
                f"populates these fields, or fix the template."
            )

        # 3. If it's Python, must parse
        if target.suffix == ".py":
            try:
                tree = ast.parse(text, filename=str(target))
            except SyntaxError as e:
                errors.append(f"{rel} fails to parse as Python: {e}")
                continue
            # compile() catches placement rules ast.parse accepts
            # (misplaced `from __future__`, the SRL 2026-07-05 class).
            try:
                compile(text, str(target), "exec")
            except SyntaxError as e:
                errors.append(f"{rel} fails to compile as Python: {e}")
                continue

            # 4. Symbol checks for paradigm_fixed Python files
            public_symbols = entry.get("public_symbols") or []
            for sym in public_symbols:
                name = sym.get("name")
                if not name or name.startswith("<"):
                    # Method-shaped placeholder; not the scaffolder's responsibility
                    continue
                node = _find_top_level_def(tree, name)
                if node is None:
                    errors.append(f"{rel} missing top-level symbol `{name}`")
                    continue

                expected_sig = sym.get("signature")
                kind = sym.get("kind")
                if kind == "function" and expected_sig:
                    if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        errors.append(f"{rel}: `{name}` declared as function but is {type(node).__name__}")
                    else:
                        actual_sig = _signature_string(node)
                        if _normalize_sig(actual_sig) != _normalize_sig(expected_sig):
                            errors.append(
                                f"{rel}: `{name}` signature drifts from manifest — "
                                f"expected `{expected_sig}`; got `{actual_sig}`"
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

    print(f"ok: package_scaffolder output at {args.run_dir} validates against the manifest.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
