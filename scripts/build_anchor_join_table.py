"""Anchor join table builder — the shared foundation for notes 3, 4, and 5.

The equation-to-code map already exists in the delivered code as
`# paper-element: <id>` comment anchors, validated against the paper map's
closed id set by the coder gates (the shared extractor lives in
scripts/validate_architecture_coder_output.py and is REUSED here, never
copied). This script lifts those anchors into a run-level JSON artifact so
downstream surfaces (METHOD.md "implemented by" lines, notebook
source-listing blocks, per-element test generation) can answer "which
function implements element X" without re-parsing source comments.

Design (the researcher-feedback section 3 designs note (internal, not shipped),
"Shared foundation for notes 3, 4, and 5"):

- **The mapping is many-to-many and the table says so.** Orchestrator
  functions carry anchors at the call sites of the dedicated
  implementations, so one id can be anchored in several functions. The
  table stores ALL anchor sites per id (file, enclosing function or method
  qualname, line) plus ONE deterministically computed primary site.
- **Primary-site heuristic** (validated against every ms3d case in the
  design review): prefer an anchor in a function's leading block (the
  region between the `def` line and the first non-docstring statement —
  docstring-embedded anchors included, since real deliveries carry them
  there) over mid-body call-site anchors; tie-break by the enclosing
  function carrying the fewest distinct ids; final tie-break by
  (file, line) so the result is deterministic under any input.
- **Enclosing-scope attribution is AST-based** and handles methods and
  nested functions via dotted qualnames (`MultiFrameDetector.predict`).
  An anchor outside any function is attributed to its class body
  (placement "class_body", qualname = the class) or the module
  (placement "module", qualname null) and ranks below any in-function
  site for primary selection.
- **Scope + honesty.** The builder scans every `method/*.py` file (flat,
  matching the manifest's flat-package rule). The coder gates already
  validate anchors in model.py / training.py / method.py; anchors found
  in files no gate validates (data.py etc.) get their validation HERE:
  an anchor id not in the paper map's id set fails table-build with an
  honest error naming every offending site. A package with no anchors
  produces an empty table, not an error.

Artifact home: `<run_dir>/.pipeline/anchor_join_table.json` — a run-level
derived artifact alongside data_flow.json / claims_ledger.json, consumed
at render time (notes 4/5) and later at generation time (note 3).

Table schema (schema_version 1.0.0):

    {
      "schema_version": "1.0.0",
      "files_scanned": ["method/data.py", "method/method.py", ...],
      "elements": {
        "<element_id>": {
          "sites": [
            {"file": "method/method.py", "line": 99,
             "qualname": "kde_mode_1d", "function_line": 84,
             "placement": "leading"},
            ...
          ],
          "primary": { ...same shape as one site... }
        }
      }
    }

`placement` is one of: "leading", "body", "class_body", "module".
`function_line` is the `def` line of the enclosing function (or the
`class` line for class_body sites; null for module sites). Sites are
sorted by (file, line); element keys sort lexicographically; same inputs
produce the same bytes.

Usage:

    python scripts/build_anchor_join_table.py --run-dir <run_dir> \
        [--output <path>]

Exit codes:
  0  table written (an anchor-free package writes an empty table)
  1  table-build failure (unknown anchor id, unparseable method file)
  2  setup error (missing run dir / method dir / paper map)
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.paper_element_anchors import (  # noqa: E402
    PLACEMENT_RANK as _PLACEMENT_RANK,
)
from scripts.paper_element_anchors import (  # noqa: E402
    attribute_site as _attribute_site,
)
from scripts.paper_element_anchors import (  # noqa: E402
    collect_scopes as _collect_scopes,
)
from scripts.paper_element_anchors import (  # noqa: E402
    extract_paper_element_ids as _extract_paper_element_ids,
)
from scripts.validate_architecture_coder_output import (  # noqa: E402
    _load_paper_map_ids,
)

SCHEMA_VERSION = "1.0.0"
ARTIFACT_REL_PATH = ".pipeline/anchor_join_table.json"


class JoinTableError(Exception):
    """Honest table-build failure — the message names every offending site."""


class JoinTableSetupError(Exception):
    """The run dir is not a finished run (missing method/ or paper map)."""


# ---------------------------------------------------------------------------
# Table build
# ---------------------------------------------------------------------------


def _select_primary(sites: list[dict], ids_per_scope: dict) -> dict:
    """The deterministic primary site. Preference: leading over body over
    class_body/module; then the enclosing scope carrying the fewest
    distinct ids; then (file, line)."""

    def key(site: dict):
        rank = _PLACEMENT_RANK[site["placement"]]
        n_ids = ids_per_scope[(site["file"], site["qualname"])]
        return (rank, n_ids, site["file"], site["line"])

    return min(sites, key=key)


def build_join_table(run_dir: Path) -> dict:
    """Build the join table for a finished run dir. Raises
    JoinTableSetupError / JoinTableError; never writes anything."""
    run_dir = Path(run_dir)
    if not run_dir.is_dir():
        raise JoinTableSetupError(f"run dir not found: {run_dir}")
    method_dir = run_dir / "method"
    if not method_dir.is_dir():
        raise JoinTableSetupError(
            f"{method_dir} not found — the join table is built from a "
            f"finished run's delivered method/ package")
    paper_map_ids = _load_paper_map_ids(run_dir)
    if paper_map_ids is None:
        raise JoinTableSetupError(
            f"{run_dir / '.pipeline/paper_map.json'} missing or unparseable "
            f"— anchor ids are validated against the paper map's id set, so "
            f"the table cannot be built honestly without it")

    # Flat scan, matching the manifest's flat-package rule (see
    # validate_architecture_coder_output._package_module_trees).
    files_scanned: list[str] = []
    errors: list[str] = []
    # element id -> list of site dicts
    sites_by_id: dict[str, list[dict]] = {}
    # (file, qualname) -> set of distinct ids anchored in that scope
    ids_per_scope: dict[tuple, set] = {}

    for py in sorted(method_dir.glob("*.py")):
        rel = f"method/{py.name}"
        files_scanned.append(rel)
        anchors = _extract_paper_element_ids(py)
        if not anchors:
            continue
        try:
            tree = ast.parse(py.read_text(encoding="utf-8"), filename=str(py))
        except SyntaxError as e:
            errors.append(
                f"{rel} carries anchors but fails to parse as Python "
                f"({e}) — enclosing-function attribution needs a parseable "
                f"file")
            continue
        scopes = _collect_scopes(tree)
        for lineno, element_id in anchors:
            if element_id not in paper_map_ids:
                errors.append(
                    f"{rel}:{lineno}: `# paper-element: {element_id}` "
                    f"references an id that doesn't exist in "
                    f".pipeline/paper_map.json — the code-to-paper trace is "
                    f"broken, so the table build refuses to ship it")
                continue
            site = {"file": rel, "line": lineno,
                    **_attribute_site(lineno, scopes)}
            sites_by_id.setdefault(element_id, []).append(site)
            ids_per_scope.setdefault(
                (rel, site["qualname"]), set()).add(element_id)

    if errors:
        raise JoinTableError(
            "anchor join table build failed:\n  - " + "\n  - ".join(errors))

    id_counts = {scope: len(ids) for scope, ids in ids_per_scope.items()}
    elements: dict[str, dict] = {}
    for element_id in sorted(sites_by_id):
        sites = sorted(sites_by_id[element_id],
                       key=lambda s: (s["file"], s["line"]))
        elements[element_id] = {
            "sites": sites,
            "primary": _select_primary(sites, id_counts),
        }

    return {
        "schema_version": SCHEMA_VERSION,
        "files_scanned": files_scanned,
        "elements": elements,
    }


def write_join_table(run_dir: Path, output: Path | None = None) -> Path:
    """Build and write the artifact. Returns the written path."""
    table = build_join_table(run_dir)
    out = output or (Path(run_dir) / ARTIFACT_REL_PATH)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(table, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None,
                        help=f"default: <run-dir>/{ARTIFACT_REL_PATH}")
    args = parser.parse_args(argv)

    try:
        out = write_join_table(args.run_dir, args.output)
    except JoinTableSetupError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    except JoinTableError as e:
        print(f"FAIL: {e}", file=sys.stderr)
        return 1
    table = json.loads(out.read_text(encoding="utf-8"))
    n_ids = len(table["elements"])
    n_sites = sum(len(v["sites"]) for v in table["elements"].values())
    print(f"wrote {out} ({n_ids} element id(s), {n_sites} anchor site(s))")
    return 0


if __name__ == "__main__":
    sys.exit(main())
