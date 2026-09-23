"""Shared refresh engine for derived blocks in delivered documents.

The freshness problem, solved once (the researcher-feedback section 3
designs note, internal, not shipped, "Shared foundation for notes 3, 4,
and 5"): the run companion edits delivered code after delivery and never
re-renders documents, so any document surface derived from the code
(source listings, line-number links) can go stale through a real,
documented flow. The fix is this single deterministic script: derived
blocks injected into delivered documents (METHOD.md, notebook markdown
cells) carry machine-readable begin/end markers; the refresh re-derives
each marked block's content from the CURRENT delivered code and rewrites
ONLY those blocks, leaving everything else byte-identical.

Marker convention — consistent with what the renderers already do.
METHOD.md's machine-readable generation record is an HTML comment
carrying a JSON payload (`<!-- method-md {record} -->`,
scripts/generate_method_md.py), and HTML comments render invisibly in
both markdown files and notebook markdown cells. A derived block is:

    <!-- derived-block: begin {"kind": "source_listing", "spec": {...}} -->
    ...derived content, owned entirely by the engine...
    <!-- derived-block: end -->

Each marker sits on its own line. The begin payload is JSON with two
keys: `kind` (a name in the block-kind registry) and `spec` (the kind's
own deterministic derivation inputs). Everything between the markers is
engine-owned and replaced wholesale on refresh; producers should mint
blocks through `make_block()` so injected content is born identical to
what a refresh would derive. (The draft-side `# %% PLACEHOLDER: <name>`
convention in scripts/render_notebook.py is deliberately NOT reused: a
placeholder is consumed once at render time and does not survive into
the delivered document, while these markers must live in the delivered
bytes so post-delivery refreshes can find their blocks.)

Block-kind registry. Kinds are registered by name with a deriver
`(spec: dict, run_dir: Path) -> str`; notes 4 and 5 register their kinds
when they land. One trivial built-in kind ships now so the engine is
fully testable:

    source_listing — spec {"file": "method/method.py", "qualname": "kbf_fuse"}
        renders the named function's (or method's, via dotted qualname)
        current source from the delivered code as a fenced python block,
        headed by its file:line location.

An unknown kind, an unmatched marker pair, or a spec the deriver cannot
serve fails honestly — no block is ever silently skipped or emptied.
Derivation is fully deterministic (no model calls), so running the
refresh twice changes nothing.

Usage:

    python scripts/refresh_derived_blocks.py --run-dir <run_dir> \
        [--target <rel_path> ...]

Default targets: METHOD.md and notebook.ipynb (each only if present).
Files are rewritten only when a block's content actually changed.

Exit codes:
  0  refresh completed (0 or more blocks rewritten)
  1  refresh failure (unknown kind, unmatched markers, underivable spec)
  2  setup error (missing run dir / unreadable target)
"""

from __future__ import annotations

import argparse
import ast
import importlib
import json
import re
import sys
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

BEGIN_MARKER_RE = re.compile(
    r"^\s*<!--\s*derived-block:\s*begin\s+(\{.*\})\s*-->\s*$")
END_MARKER_RE = re.compile(r"^\s*<!--\s*derived-block:\s*end\s*-->\s*$")

# Default refreshable documents, run-dir relative (run_layout names).
DEFAULT_TARGETS = ("METHOD.md", "notebook.ipynb")


class DerivedBlockError(Exception):
    """Honest refresh failure — never a silent skip."""


class RefreshSetupError(Exception):
    """The run dir or a requested target is missing/unreadable."""


# ---------------------------------------------------------------------------
# Block-kind registry
# ---------------------------------------------------------------------------

_BLOCK_KINDS: dict[str, Callable[[dict, Path], str]] = {}


def register_block_kind(kind: str,
                        deriver: Callable[[dict, Path], str]) -> None:
    """Register a block kind. Later block producers (notes 4/5) call this
    at import time; re-registering an existing kind is a programming error
    and fails loudly rather than silently swapping derivations."""
    if kind in _BLOCK_KINDS:
        raise ValueError(f"block kind {kind!r} is already registered")
    _BLOCK_KINDS[kind] = deriver


def registered_block_kinds() -> list[str]:
    return sorted(_BLOCK_KINDS)


# Block-kind producers: the renderers register their kinds at import time
# (notes 4 and 5). The engine imports them lazily, and only on a kind miss —
# a top-level import would be circular (producers import this engine to mint
# their blocks), and source_listing-only flows never pay the import.
_PRODUCER_MODULES = ("render_notebook", "generate_method_md")


def _import_producer_kinds() -> None:
    scripts_dir = str(Path(__file__).resolve().parent)
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    for name in _PRODUCER_MODULES:
        if name not in sys.modules:
            importlib.import_module(name)


def derive_block_content(kind: str, spec: dict, run_dir: Path) -> str:
    """Re-derive one block's content from the current delivered code."""
    deriver = _BLOCK_KINDS.get(kind)
    if deriver is None:
        _import_producer_kinds()
        deriver = _BLOCK_KINDS.get(kind)
    if deriver is None:
        raise DerivedBlockError(
            f"unknown derived-block kind {kind!r} — registered kinds: "
            f"{registered_block_kinds() or 'none'}")
    return deriver(spec, Path(run_dir))


def format_begin_marker(kind: str, spec: dict) -> str:
    payload = json.dumps({"kind": kind, "spec": spec}, sort_keys=True)
    return f"<!-- derived-block: begin {payload} -->"


END_MARKER = "<!-- derived-block: end -->"


def make_block(kind: str, spec: dict, run_dir: Path) -> str:
    """A complete marked block for producers to inject. Minting through
    the engine keeps injected content identical to a refresh's output, so
    a block is idempotent from birth."""
    content = derive_block_content(kind, spec, run_dir)
    return "\n".join([format_begin_marker(kind, spec), content, END_MARKER])


# ---------------------------------------------------------------------------
# Built-in kind: source_listing
# ---------------------------------------------------------------------------


def _find_def_by_qualname(tree: ast.Module, qualname: str):
    """The FunctionDef/AsyncFunctionDef matching a dotted qualname
    (`Class.method`, `outer.inner`), or None."""
    parts = qualname.split(".")

    def descend(node: ast.AST, remaining: list[str]):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef,
                                  ast.ClassDef)) and child.name == remaining[0]:
                if len(remaining) == 1:
                    return child
                found = descend(child, remaining[1:])
                if found is not None:
                    return found
            elif not isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef,
                                        ast.ClassDef)):
                found = descend(child, remaining)
                if found is not None:
                    return found
        return None

    node = descend(tree, parts)
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return node
    return None


def _derive_source_listing(spec: dict, run_dir: Path) -> str:
    rel = spec.get("file")
    qualname = spec.get("qualname")
    if not rel or not qualname:
        raise DerivedBlockError(
            f"source_listing spec needs 'file' and 'qualname'; got {spec!r}")
    path = run_dir / rel
    if not path.is_file():
        raise DerivedBlockError(
            f"source_listing: {rel} not found under {run_dir}")
    try:
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(path))
    except (SyntaxError, UnicodeDecodeError) as e:
        raise DerivedBlockError(
            f"source_listing: {rel} fails to parse as Python: {e}")
    node = _find_def_by_qualname(tree, qualname)
    if node is None:
        raise DerivedBlockError(
            f"source_listing: no function or method `{qualname}` in {rel} — "
            f"if the code was renamed after delivery, update the block's "
            f"spec rather than leaving a stale listing")
    lines = source.split("\n")
    start = node.lineno
    if node.decorator_list:
        start = min(d.lineno for d in node.decorator_list)
    end = node.end_lineno or node.lineno
    listing = "\n".join(lines[start - 1:end])
    # A fence must be longer than any backtick run inside the listing.
    fence = "```"
    while fence in listing:
        fence += "`"
    return "\n".join([
        f"**Source: `{qualname}` — {rel}:{node.lineno}**",
        "",
        f"{fence}python",
        listing,
        fence,
    ])


register_block_kind("source_listing", _derive_source_listing)


# ---------------------------------------------------------------------------
# Text refresh
# ---------------------------------------------------------------------------


def refresh_text(text: str, run_dir: Path, *, where: str = "document",
                 ) -> tuple[str, int, int]:
    """Rewrite every marked block in `text` from the current delivered
    code. Returns (new_text, blocks_seen, blocks_changed). Unmarked
    content round-trips byte-identically (`split("\\n")` / join — no line
    is ever re-wrapped or re-terminated)."""
    lines = text.split("\n")
    out: list[str] = []
    blocks_seen = 0
    blocks_changed = 0
    i = 0
    while i < len(lines):
        line = lines[i]
        if END_MARKER_RE.match(line):
            raise DerivedBlockError(
                f"{where}: derived-block end marker without a begin "
                f"(line {i + 1})")
        begin = BEGIN_MARKER_RE.match(line)
        if not begin:
            out.append(line)
            i += 1
            continue
        try:
            header = json.loads(begin.group(1))
        except json.JSONDecodeError as e:
            raise DerivedBlockError(
                f"{where}: derived-block begin marker carries invalid JSON "
                f"(line {i + 1}): {e}")
        kind = header.get("kind")
        spec = header.get("spec")
        if not isinstance(kind, str) or not isinstance(spec, dict):
            raise DerivedBlockError(
                f"{where}: derived-block begin marker needs string 'kind' "
                f"and object 'spec' (line {i + 1}): {header!r}")
        # Collect the old interior up to the matching end marker.
        j = i + 1
        old_interior: list[str] = []
        while j < len(lines) and not END_MARKER_RE.match(lines[j]):
            if BEGIN_MARKER_RE.match(lines[j]):
                raise DerivedBlockError(
                    f"{where}: nested derived-block begin at line {j + 1} — "
                    f"blocks do not nest")
            old_interior.append(lines[j])
            j += 1
        if j >= len(lines):
            raise DerivedBlockError(
                f"{where}: derived-block begun at line {i + 1} has no end "
                f"marker")
        blocks_seen += 1
        content = derive_block_content(kind, spec, run_dir)
        new_interior = content.split("\n")
        if new_interior != old_interior:
            blocks_changed += 1
        out.append(line)            # begin marker, byte-identical
        out.extend(new_interior)
        out.append(lines[j])        # end marker, byte-identical
        i = j + 1
    return "\n".join(out), blocks_seen, blocks_changed


# ---------------------------------------------------------------------------
# File refreshers
# ---------------------------------------------------------------------------


def refresh_markdown_file(path: Path, run_dir: Path) -> tuple[int, int]:
    """Refresh a markdown document in place. Returns (seen, changed);
    writes only when a block actually changed."""
    text = path.read_text(encoding="utf-8")
    new_text, seen, changed = refresh_text(
        text, run_dir, where=str(path.name))
    if changed:
        path.write_text(new_text, encoding="utf-8")
    return seen, changed


def refresh_notebook(path: Path, run_dir: Path) -> tuple[int, int]:
    """Refresh marked blocks in a notebook's MARKDOWN cells in place.
    Returns (seen, changed). The file is rewritten (via nbformat, the
    same writer that produced it) only when a block actually changed, so
    an all-fresh notebook is untouched byte-for-byte."""
    import nbformat  # local: markdown-only refreshes never need it

    nb = nbformat.read(str(path), as_version=4)
    seen = 0
    changed = 0
    for idx, cell in enumerate(nb.cells):
        if cell.cell_type != "markdown":
            continue
        new_source, cell_seen, cell_changed = refresh_text(
            cell.source, run_dir, where=f"{path.name} cell {idx}")
        seen += cell_seen
        changed += cell_changed
        if cell_changed:
            cell.source = new_source
    if changed:
        nbformat.write(nb, str(path))
    return seen, changed


def refresh_run_documents(run_dir: Path,
                          targets: list[str] | None = None,
                          ) -> dict[str, tuple[int, int]]:
    """Refresh the run's delivered documents. `targets` are run-dir
    relative paths; default is METHOD.md + notebook.ipynb, each only if
    present. An explicitly requested target that is missing is a setup
    error. Returns {rel_path: (blocks_seen, blocks_changed)}."""
    run_dir = Path(run_dir)
    if not run_dir.is_dir():
        raise RefreshSetupError(f"run dir not found: {run_dir}")
    explicit = targets is not None
    rels = list(targets) if explicit else list(DEFAULT_TARGETS)
    results: dict[str, tuple[int, int]] = {}
    for rel in rels:
        path = run_dir / rel
        if not path.is_file():
            if explicit:
                raise RefreshSetupError(f"target not found: {path}")
            continue
        if path.suffix == ".ipynb":
            results[rel] = refresh_notebook(path, run_dir)
        else:
            results[rel] = refresh_markdown_file(path, run_dir)
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--target", action="append", default=None,
                        metavar="REL_PATH",
                        help="run-dir-relative document to refresh "
                             "(repeatable); default: METHOD.md and "
                             "notebook.ipynb, each only if present")
    args = parser.parse_args(argv)

    try:
        results = refresh_run_documents(args.run_dir, args.target)
    except RefreshSetupError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    except DerivedBlockError as e:
        print(f"FAIL: {e}", file=sys.stderr)
        return 1
    for rel, (seen, changed) in results.items():
        print(f"{rel}: {seen} derived block(s), {changed} rewritten")
    if not results:
        print("no refreshable documents found (nothing to do)")
    return 0


if __name__ == "__main__":
    # Run through the canonical module instance, not the `__main__` one:
    # producers register their block kinds into the module imported as
    # `refresh_derived_blocks`, and the CLI must consult THAT registry —
    # otherwise the lazy producer import would register into a parallel
    # module instance and every producer kind would look unknown here.
    from refresh_derived_blocks import main as _canonical_main
    sys.exit(_canonical_main())
