"""Stage 2.d — final package-import validator.

Runs after `finalize_package_init.py`. Confirms the assembled package can
actually be imported in a fresh subprocess:

  1. `<run_dir>/method/__init__.py` and `<run_dir>/requirements.txt` exist.
  2. `python -c "from method import *"` returns 0 from a fresh subprocess
     started with `<run_dir>` on the path.
  3. Every symbol in `__all__` is actually importable (no NameError, no missing
     re-export).
  4. The pluggable function (named per spec.comparison.pluggable_component.name)
     is in `__all__`. Architecture classes are validated separately by
     `finalize_package_init.py` and `validate_architecture_coder_output.py`
     against the paradigm's `package_manifest` (count varies by paradigm —
     1 for active_learning, 2 for knowledge_distillation).
  5. `__all__` carries no duplicate entries.
  6. The exported pluggable RESOLVES to the method-coder's module: importing
     the package and reading `__module__` off the bound object must name
     `method.method`. A name can be present in `__all__` and importable while
     bound to a different module's function with a different signature and
     return type (the fedavg 2026-08-04 defect,
     `exported_pluggable_resolves_to_wrong_module`) — identity, not presence,
     is the promise the spec makes.

A subprocess-level smoke test is the right gate here because it catches issues
the AST validators can't — e.g., circular imports inside method/, runtime
errors in module-level code, missing transitive dependencies. The fresh
subprocess also matches the conditions under which the notebook will run.

Deterministic gate. On failure, exit 1 with a structured error list on stderr.

Usage:

    python scripts/validate_package_imports.py --spec <method_spec.json> --run-dir <output_dir>

Exit codes:
  0  validation passed
  1  validation failures
  2  setup error (missing spec / run dir / required files)
"""

from __future__ import annotations

import argparse
import ast
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def _read_all_list(init_path: Path) -> list[str] | None:
    """Parse __init__.py and return the __all__ list contents (or None if not found)."""
    try:
        tree = ast.parse(init_path.read_text(encoding="utf-8"), filename=str(init_path))
    except SyntaxError:
        return None
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for tgt in node.targets:
                if isinstance(tgt, ast.Name) and tgt.id == "__all__":
                    if isinstance(node.value, ast.List):
                        out: list[str] = []
                        for elt in node.value.elts:
                            if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                                out.append(elt.value)
                        return out
    return None


def validate(spec: dict, run_dir: Path) -> list[str]:
    errors: list[str] = []

    init_path = run_dir / "method" / "__init__.py"
    requirements_path = run_dir / "requirements.txt"

    # 1. File presence
    if not init_path.is_file():
        errors.append(f"method/__init__.py missing at {init_path}")
    if not requirements_path.is_file():
        errors.append(f"requirements.txt missing at {requirements_path}")
    if errors:
        return errors

    # 2. __all__ list parses
    all_list = _read_all_list(init_path)
    if all_list is None:
        errors.append("method/__init__.py: could not parse __all__ as a list of string literals")
        return errors
    if not all_list:
        errors.append("method/__init__.py: __all__ is empty")

    # 3. Pluggable function + architecture class are in __all__
    pluggable_name = (spec.get("comparison") or {}).get("pluggable_component", {}).get("name")
    if pluggable_name and pluggable_name not in all_list:
        errors.append(
            f"method/__init__.py: pluggable function `{pluggable_name}` is not in __all__ — "
            f"the notebook §1 imports the public API and will fail to find it."
        )

    # 3b. __all__ carries no duplicates (a duplicate is the reliable smell of
    # two producers publishing the same name — the fedavg shape).
    seen: set[str] = set()
    dupes = sorted({name for name in all_list if name in seen or seen.add(name)})
    if dupes:
        errors.append(
            f"method/__init__.py: __all__ lists duplicate entries: {dupes} — "
            f"each public name must be exported exactly once."
        )

    # 4. Subprocess import smoke test — `from method import *` from run_dir
    test_script = "from method import *; import sys; sys.exit(0)"
    proc = subprocess.run(
        [sys.executable, "-c", test_script],
        cwd=str(run_dir),
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        errors.append(
            f"`from method import *` failed with exit {proc.returncode} when run from "
            f"{run_dir}:\n  stdout: {proc.stdout.strip()}\n  stderr: {proc.stderr.strip()}"
        )
        return errors

    # 5. Verify each __all__ symbol is actually importable individually
    quoted = ", ".join(repr(name) for name in all_list)
    test_script = (
        "import method, sys\n"
        "missing = []\n"
        f"for name in [{quoted}]:\n"
        "    if not hasattr(method, name):\n"
        "        missing.append(name)\n"
        "if missing:\n"
        "    print(','.join(missing))\n"
        "    sys.exit(1)\n"
        "sys.exit(0)\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", test_script],
        cwd=str(run_dir),
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        missing = (proc.stdout or "").strip().split(",")
        errors.append(
            f"method/__init__.py: __all__ lists symbols that aren't actually importable: "
            f"{missing}"
        )

    # 6. The exported pluggable resolves to the method-coder's module. `__module__`
    # names where the object was DEFINED, so it survives import aliasing
    # (including the naming bridge's `from .method import _x as x` remedy) and
    # catches a binding stolen by a later import line from another module.
    if pluggable_name and pluggable_name in all_list:
        test_script = (
            "import method, sys\n"
            f"obj = getattr(method, {pluggable_name!r}, None)\n"
            "if obj is None:\n"
            "    print('<unbound>'); sys.exit(1)\n"
            "print(getattr(obj, '__module__', '<no __module__>'))\n"
            "sys.exit(0)\n"
        )
        proc = subprocess.run(
            [sys.executable, "-c", test_script],
            cwd=str(run_dir),
            capture_output=True,
            text=True,
        )
        resolved = (proc.stdout or "").strip()
        if proc.returncode != 0 or resolved != "method.method":
            errors.append(
                f"method/__init__.py: the exported pluggable `{pluggable_name}` "
                f"resolves to `{resolved or '<unknown>'}`, not to the "
                f"method-coder's module `method.method` — the public name is "
                f"bound to a different function than the spec promised "
                f"(exported_pluggable_resolves_to_wrong_module)."
            )

    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()

    if not args.spec.is_file():
        print(f"error: spec not found: {args.spec}", file=sys.stderr)
        return 2

    try:
        spec = json.loads(args.spec.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        print(f"error: spec is not valid JSON ({e})", file=sys.stderr)
        return 2

    errors = validate(spec, args.run_dir)
    if errors:
        print(f"FAIL: {len(errors)} validation error(s):", file=sys.stderr)
        for err in errors:
            print(f"  - {err}", file=sys.stderr)
        return 1

    print(f"ok: package at {args.run_dir} imports cleanly; __all__ resolves.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
