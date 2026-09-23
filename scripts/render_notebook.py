"""Stage 2.x — render notebook draft (jupytext-percent format) to .ipynb.

Input: `<run_dir>/.pipeline/notebook_draft.py` produced by the
notebook-generator agent. The draft is a Python file with jupytext-percent
cell markers (`# %%` for code, `# %% [markdown]` for markdown).

The agent inserts **placeholder cells** for content that should be rendered
deterministically from `<run_dir>/.pipeline/params.json` instead of being
hand-written by the LLM. Placeholders this script recognizes:

  - `# %% PLACEHOLDER: params_dict`         — replaced with the `params = {...}`
                                              code cell + `unpack()` helper +
                                              `cfg = unpack(params)` + the
                                              optional `# cfg.update({...})`
                                              paper-faithful override comment.
  - `# %% [markdown] PLACEHOLDER: params_table` — replaced with the rendered
                                              provenance table (markdown),
                                              followed by a role-separated
                                              evaluation-protocol cell when
                                              method_spec.json declares one.
  - `# %% PLACEHOLDER: component_stub:<element_id>` — partial delivery
                                              (§3.5 criterion 4): expands to a
                                              markdown notice PLUS a raising
                                              code cell for a component recorded
                                              in `.pipeline/stubbed_elements.json`.
                                              When any stub is recorded, a PARTIAL
                                              banner cell is also injected under
                                              the title regardless of markers.

This split exists because:
  1. LLMs occasionally mis-transcribe parameter values from JSON; a script
     reading params.json gets it byte-for-byte right.
  2. The provenance table is structural; deterministic rendering keeps it
     consistent across papers.
  3. Paper protocol declarations and runtime demo choices have different
     owners; rendering both from their structured artifacts prevents a test
     span from being restated as a one-call forecast horizon.

**Implementation notes** (Japan-feedback note 4) are a second deterministic
addition, derived rather than placeholder-driven: each demo code cell whose
calls resolve to the delivered package's anchored implementing functions
gains a "how this is implemented" markdown cell — function name, method.py
location, METHOD.md equation links, and the function's source collapsed
under <details> (signature + docstring only above ~60 lines). The
association is derived at render time (AST call scan × public imports ×
the anchor join table); no notebook-generator prompt changes, no markers
required from the draft. Injected blocks carry derived-block markers so
scripts/refresh_derived_blocks.py re-derives them after companion edits.
A run without anchors renders exactly as before.

Output: `<run_dir>/notebook.ipynb` — a valid Jupyter notebook ready for the user.

Usage:

    python scripts/render_notebook.py --run-dir <output_dir>

Exit codes:
  0  notebook.ipynb written successfully
  1  setup error (missing draft / params.json / invalid jupytext format)
  2  unsubstituted placeholders or other render-time failure
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

import nbformat  # noqa: E402
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook  # noqa: E402

from generate_method_md import load_join_table_elements  # noqa: E402
from evaluation_protocol_rendering import (  # noqa: E402
    render_evaluation_protocol_block,
)
from refresh_derived_blocks import (  # noqa: E402
    DerivedBlockError,
    _find_def_by_qualname,
    derive_block_content,
    make_block,
    register_block_kind,
    registered_block_kinds,
)


PLACEHOLDER_PATTERN = re.compile(r"^# %%(\s*\[markdown\])?\s*PLACEHOLDER:\s*(\S+)\s*$")
CELL_DIVIDER = re.compile(r"^# %%(\s*\[markdown\])?\s*$")


# ---------------------------------------------------------------------------
# Jupytext parsing
# ---------------------------------------------------------------------------


def _parse_jupytext(text: str) -> list[tuple[str, str, str | None]]:
    """Parse jupytext-percent text into cells.

    Returns a list of (kind, source, placeholder_name) tuples, where:
      - kind: "code" or "markdown"
      - source: cell content (with leading "# " stripped from markdown lines)
      - placeholder_name: the name from `PLACEHOLDER: <name>` if present, else None
    """
    lines = text.splitlines()
    # Skip any leading lines until first cell divider.
    i = 0
    while i < len(lines) and not CELL_DIVIDER.match(lines[i]) and not PLACEHOLDER_PATTERN.match(lines[i]):
        i += 1

    cells: list[tuple[str, str, str | None]] = []
    current_kind: str | None = None
    current_placeholder: str | None = None
    current_lines: list[str] = []

    def flush() -> None:
        if current_kind is None:
            return
        if current_placeholder is not None:
            # Placeholder cells have no content beyond the marker line.
            cells.append((current_kind, "", current_placeholder))
            return
        text_block = "\n".join(current_lines).rstrip("\n")
        if current_kind == "markdown":
            md_lines: list[str] = []
            for ln in text_block.splitlines():
                if ln.startswith("# "):
                    md_lines.append(ln[2:])
                elif ln == "#":
                    md_lines.append("")
                else:
                    md_lines.append(ln)
            cells.append(("markdown", "\n".join(md_lines), None))
        else:
            cells.append(("code", text_block, None))

    for ln in lines[i:]:
        ph = PLACEHOLDER_PATTERN.match(ln)
        cd = CELL_DIVIDER.match(ln)
        if ph:
            flush()
            current_kind = "markdown" if ph.group(1) else "code"
            current_placeholder = ph.group(2)
            current_lines = []
        elif cd:
            flush()
            current_kind = "markdown" if cd.group(1) else "code"
            current_placeholder = None
            current_lines = []
        else:
            current_lines.append(ln)
    flush()
    return cells


# ---------------------------------------------------------------------------
# Deterministic placeholder rendering
# ---------------------------------------------------------------------------


def _render_params_dict_cell(params_data: dict) -> str:
    """Render a `params = {...}` code cell from params.json contents.

    Includes the `unpack(params)` helper and the `cfg = unpack(params)` call.
    The notebook's downstream cells reference `cfg`.
    """
    params = params_data["params"]
    lines: list[str] = ["params = {"]
    for name, entry in params.items():
        lines.append(f'    "{name}": {{')
        for key, val in entry.items():
            if isinstance(val, str):
                # Use triple-quoted string for long reasoning text to keep formatting nice.
                if "\n" in val or len(val) > 80:
                    escaped = val.replace('"""', '\\"\\"\\"')
                    lines.append(f'        "{key}": (')
                    # Wrap long text with indentation.
                    wrapped = _wrap_string_for_dict(escaped)
                    for w in wrapped:
                        lines.append(f"            {w}")
                    lines.append("        ),")
                else:
                    lines.append(f'        "{key}": {val!r},')
            elif isinstance(val, bool):
                lines.append(f'        "{key}": {val},')
            else:
                lines.append(f'        "{key}": {val!r},')
        lines.append("    },")
    lines.append("}")
    lines.append("")
    lines.append("")
    lines.append("def unpack(p: dict) -> dict:")
    lines.append('    """Strip provenance and return a flat name -> value dict."""')
    lines.append('    return {k: v["value"] for k, v in p.items() if v.get("used_in_notebook", True)}')
    lines.append("")
    lines.append("")
    lines.append("cfg = unpack(params)")
    return "\n".join(lines)


def _wrap_string_for_dict(s: str, width: int = 70) -> list[str]:
    """Wrap a long string into multiple `"…"` lines for nicer dict formatting."""
    import textwrap
    chunks = textwrap.wrap(s, width=width, break_long_words=False, break_on_hyphens=False)
    if not chunks:
        return ['""']
    return [f'"{chunk} "' if i < len(chunks) - 1 else f'"{chunk}"' for i, chunk in enumerate(chunks)]


def _render_params_table_cell(params_data: dict) -> str:
    """Render the provenance markdown table from params.json.

    Columns: Parameter | Variable from paper | Value from paper | Paper value |
    System value | Where in paper | Used? | Notes. The "Where in paper" column
    (researcher feedback 2026-07, Japan-feedback plan §2): each parameter shows the paper
    location its provenance traces to, when the deriver recorded one.
    """
    rows = [
        "| Parameter | Variable from paper | Value from paper | Paper value | System value | Where in paper | Used? | Notes |",
        "|---|:-:|:-:|---|---|---|:-:|---|",
    ]
    for name, entry in params_data["params"].items():
        s = entry["source"]
        typed_protocol = entry.get("protocol_role") is not None
        paper_defines_variable = (
            s in ("paper", "system_default")
            or (typed_protocol and bool(entry.get("paper_says")))
        )
        paper_states_value = (
            s in ("paper", "system_default")
            or entry.get("paper_value_status") == "paper_stated"
        )
        var_from_paper = "✅" if paper_defines_variable else "❌"
        val_from_paper = "✅" if paper_states_value else "❌"
        if s == "paper":
            paper_val, sys_val = repr(entry["value"]), "—"
        elif s == "system_default":
            paper_val = repr(entry["paper_value"]) if "paper_value" in entry else "—"
            sys_val = repr(entry["value"])
        else:  # system_inferred
            paper_val = "—"
            sys_val = repr(entry["value"])
        where = str(entry.get("paper_section") or "—").replace("|", "\\|")
        used = "✅" if entry.get("used_in_notebook", True) else "❌"

        # Pick the right note/reasoning text to show. The paper's own
        # explanation of the parameter (paper_says) leads when present —
        # it is the reader-facing meaning; unused_reason next (most
        # informative for an unused param), then derivation notes.
        note = (
            entry.get("paper_says")
            or entry.get("unused_reason")
            or entry.get("note")
            or entry.get("reasoning")
            or ""
        )
        if len(note) > 80:
            cut = note[:77].rsplit(" ", 1)[0]
            note = cut + "..."
        note = note.replace("|", "\\|")

        rows.append(
            f"| `{name}` | {var_from_paper} | {val_from_paper} | {paper_val} | {sys_val} | {where} | {used} | {note} |"
        )
    return "\n".join(rows)


PLACEHOLDER_RENDERERS = {
    "params_dict": _render_params_dict_cell,
    "params_table": _render_params_table_cell,
}


# ---------------------------------------------------------------------------
# Partial delivery (§3.5 criterion 4): a notebook can never run silently
# past a stubbed component. The generator marks the spot with
# `# %% PLACEHOLDER: component_stub:<element_id>`; the render expands it
# deterministically into a markdown notice PLUS a raising code cell, and a
# PARTIAL banner cell is always injected after the title when any stub is
# recorded — even if the generator forgot every marker.
# ---------------------------------------------------------------------------

COMPONENT_STUB_PREFIX = "component_stub:"


def _render_stub_markdown(rec: dict) -> str:
    what = ("the paper's core mechanism" if rec.get("role") == "core"
            else "a supporting component")
    return (f"> ⚠️ **PARTIAL delivery — the next cell raises.** "
            f"`{rec['element_id']}` ({what}) is NOT implemented in this "
            f"package; it ships as a self-identifying stub. Work order: "
            f"[`{rec['work_order']}`]({rec['work_order']}).")


def _render_stub_code(rec: dict) -> str:
    message = (f"PARTIAL delivery: `{rec['element_id']}` is NOT implemented. "
               f"See the work order: {rec['work_order']}")
    return f"raise NotImplementedError(\n    {message!r})"


def _render_partial_banner(run_dir: Path, stubs: list[dict]) -> str:
    from partial_delivery import (completeness_counts, completeness_statement,
                                  core_gap_clause, stub_display_lines)
    counts = completeness_counts(run_dir / ".pipeline", stubs)
    lines = [f"> ⚠️ **PARTIAL delivery**: {completeness_statement(counts)}.",
             ">"]
    core = core_gap_clause(stubs)
    if core:
        lines.extend([f"> **{core}**", ">"])
    lines.append("> Stubbed (NOT implemented) components:")
    lines.append(">")
    lines.extend(f"> - {line}" for line in stub_display_lines(stubs))
    lines.append(">")
    lines.append("> Cells that exercise a stubbed component do not run: "
                 "they raise, pointing at the component's work order.")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Table of contents
# ---------------------------------------------------------------------------
#
# We inject an explicit `<a id="…"></a>` HTML anchor before each ## / ### heading
# instead of relying on Jupyter's auto-anchor (which varies across classic /
# Lab / nbconvert / VSCode notebook renderers, especially with emojis and
# punctuation). The TOC links to those explicit anchors so navigation works
# in every host.

_HEADING_RE = re.compile(r"^(#{2,6})\s+(.*)$")


def _heading_anchor(text: str) -> str:
    """Slugify a heading into a stable anchor ID."""
    s = text.lower().strip()
    # Drop everything that isn't a-z0-9, whitespace, or hyphen — strips emojis,
    # dots, parens, colons. The anchor is for navigation, not for display.
    s = re.sub(r"[^a-z0-9\s-]", "", s)
    s = re.sub(r"[-\s]+", "-", s).strip("-")
    return f"sec-{s}" if s else "sec-untitled"


def _extract_and_anchor_headings(cells: list) -> list[tuple[int, str, str]]:
    """For each markdown cell whose first non-blank line is a ## (or deeper) heading,
    inject an HTML anchor before the heading and record an entry for the TOC.

    Mutates `cells` in place. Returns a list of `(level, heading_text, anchor_id)`
    tuples in document order. h1 (`#`) headings are skipped — that's the title.
    """
    toc: list[tuple[int, str, str]] = []
    for cell in cells:
        if cell.cell_type != "markdown":
            continue
        src = cell.source
        lines = src.split("\n")
        # Find the first non-blank line; only act if it's a ## or deeper heading.
        for j, line in enumerate(lines):
            stripped = line.strip()
            if not stripped:
                continue
            m = _HEADING_RE.match(stripped)
            if not m:
                break  # cell doesn't start with a heading; skip
            level = len(m.group(1))
            heading_text = m.group(2).strip()
            anchor = _heading_anchor(heading_text)
            # Inject anchor on its own line above the heading, separated by a blank line.
            new_lines = lines[:j] + [f'<a id="{anchor}"></a>', ""] + lines[j:]
            cell.source = "\n".join(new_lines)
            toc.append((level, heading_text, anchor))
            break
    return toc


def _render_toc_cell(entries: list[tuple[int, str, str]]) -> str:
    """Render a markdown TOC cell from extracted heading entries.

    h2 entries are top-level bullets; h3+ entries are nested with two spaces
    of indent per level beyond h2.
    """
    lines = ["## Contents", ""]
    for level, text, anchor in entries:
        indent = "  " * (level - 2)
        lines.append(f"{indent}- [{text}](#{anchor})")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Implementation notes (Japan-feedback note 4)
#
# Wrapper calls hide the math: the reader sees the LaTeX and the call, never
# the implementation. Each associated demo code cell therefore gains a
# deterministic "how this is implemented" markdown cell UNDER the existing
# LaTeX markdown (i.e. directly above the code cell). The demo cell still
# calls the imported function — the listing sits beside it for
# understanding, never inlined into executed cells.
#
# The cell-to-element association is DERIVED, not assumed (the draft's
# markdown carries prose, no machine-readable ids): AST-parse each code
# cell's calls, resolve them through the notebook's own package imports
# (the public-API intersection), and join function -> element ids through
# the anchor join table. Only ids whose PRIMARY anchor site sits in a
# called function associate — the cap-to-primary rule — so a cell calling
# an orchestrator that touches nine ids at call sites links only the ids
# that LIVE in the orchestrator, and a cell calling only wrappers skips.
# Each implementing function is annotated once, at its first associated
# cell, and a cell resolving to many implementing functions is an overview
# cell and skips entirely: orchestrator cells must not drown in links.
# ---------------------------------------------------------------------------

IMPLEMENTATION_NOTE_KIND = "implementation_note"

# Size policy (note 4): inline the function's source up to roughly 60
# lines; above that, signature and docstring with the file link only (the
# ms3d demo functions total ~665 lines — uncapped inlining would roughly
# double the notebook).
_INLINE_SOURCE_MAX_LINES = 60

# A cell resolving to more than this many implementing functions is an
# overview cell, not a per-component demo — it gets no notes at all.
_MAX_IMPL_FUNCTIONS_PER_CELL = 3

_IPYTHON_LINE_RE = re.compile(r"^\s*[%!]")


def _paper_map_names(run_dir: Path) -> dict[str, str]:
    """Element id -> reader-facing name, for the METHOD.md links."""
    path = Path(run_dir) / ".pipeline" / "paper_map.json"
    try:
        paper_map = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        raise DerivedBlockError(
            f"implementation_note: {path} missing or unparseable — the "
            f"METHOD.md links need the paper map's element names")
    return {str(e.get("id")): str(e.get("name") or e.get("id"))
            for e in paper_map.get("elements", [])}


def _derive_implementation_note(spec: dict, run_dir: Path) -> str:
    """Deriver for one implementation-note block. The spec pins the
    render-time association (file, qualname, element ids); everything
    line-number- or content-bearing is re-derived from the CURRENT
    delivered code, so the shared refresh script keeps listings honest
    after companion edits."""
    rel = spec.get("file")
    qualname = spec.get("qualname")
    element_ids = spec.get("element_ids")
    if not rel or not qualname or not isinstance(element_ids, list) \
            or not element_ids:
        raise DerivedBlockError(
            f"implementation_note spec needs 'file', 'qualname' and a "
            f"non-empty 'element_ids' list; got {spec!r}")
    run_dir = Path(run_dir)
    names = _paper_map_names(run_dir)
    missing = [eid for eid in element_ids if eid not in names]
    if missing:
        raise DerivedBlockError(
            f"implementation_note: element id(s) {missing} are not in "
            f".pipeline/paper_map.json — the code-to-paper trace is broken")
    path = run_dir / rel
    if not path.is_file():
        raise DerivedBlockError(
            f"implementation_note: {rel} not found under {run_dir}")
    try:
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(path))
    except (SyntaxError, UnicodeDecodeError) as e:
        raise DerivedBlockError(
            f"implementation_note: {rel} fails to parse as Python: {e}")
    node = _find_def_by_qualname(tree, qualname)
    if node is None:
        raise DerivedBlockError(
            f"implementation_note: no function or method `{qualname}` in "
            f"{rel} — if the code was renamed after delivery, update the "
            f"block's spec rather than leaving a stale note")
    start = node.lineno
    if node.decorator_list:
        start = min(d.lineno for d in node.decorator_list)
    end = node.end_lineno or node.lineno
    n_lines = end - start + 1
    links = ", ".join(f"[{names[eid]}](METHOD.md#{eid})"
                      for eid in element_ids)
    header = (f"**How this is implemented:** [`{qualname}`]({rel}) "
              f"(`{rel}:{node.lineno}`) implements {links}.")
    if n_lines <= _INLINE_SOURCE_MAX_LINES:
        # The full source, collapsed so the tutorial flow survives. The
        # listing itself is the shared source_listing derivation — one
        # source of truth for extraction and fence escaping.
        listing = derive_block_content(
            "source_listing", {"file": rel, "qualname": qualname}, run_dir)
        return "\n".join([
            header,
            "",
            "<details>",
            f"<summary>Source of <code>{qualname}</code> "
            f"({n_lines} lines)</summary>",
            "",
            listing,
            "",
            "</details>",
        ])
    # Size cap: signature and docstring with the file link only.
    body = node.body
    if body and isinstance(body[0], ast.Expr) \
            and isinstance(body[0].value, ast.Constant) \
            and isinstance(body[0].value.value, str):
        excerpt_end = body[0].end_lineno or body[0].lineno
    elif body:
        excerpt_end = body[0].lineno - 1
    else:
        excerpt_end = end
    excerpt_end = max(excerpt_end, node.lineno)
    excerpt = "\n".join(source.split("\n")[start - 1:excerpt_end])
    fence = "```"
    while fence in excerpt:
        fence += "`"
    return "\n".join([
        header,
        "",
        f"The full implementation is {n_lines} lines (too long to inline "
        f"here) — read it at [`{rel}:{node.lineno}`]({rel}). Signature and "
        f"docstring:",
        "",
        f"{fence}python",
        excerpt,
        fence,
    ])


def _cell_ast(source: str) -> ast.Module | None:
    """AST for a code cell, with IPython magic/shell lines blanked (the
    setup cell legitimately opens with `%matplotlib inline` and `%pip`).
    None when the cell still fails to parse — an unparseable cell simply
    never associates."""
    cleaned = "\n".join(
        "" if _IPYTHON_LINE_RE.match(line) else line
        for line in source.split("\n"))
    try:
        return ast.parse(cleaned)
    except SyntaxError:
        return None


def _package_import_aliases(
    trees: list[ast.Module], package: str,
) -> tuple[dict[str, str], set[str], bool]:
    """How the notebook's code refers to the delivered package.

    Returns (from_aliases, module_aliases, star): from_aliases maps a local
    name to the package symbol it aliases (`from method import kbf_fuse as
    kf`); module_aliases are local names bound to the package or one of its
    modules (`import method`, `import method.method as mm`); star marks a
    `from method import *`."""
    from_aliases: dict[str, str] = {}
    module_aliases: set[str] = set()
    star = False
    for tree in trees:
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                if node.level != 0 or not node.module:
                    continue
                if node.module != package and \
                        not node.module.startswith(package + "."):
                    continue
                for alias in node.names:
                    if alias.name == "*":
                        star = True
                    else:
                        from_aliases[alias.asname or alias.name] = alias.name
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name == package or \
                            alias.name.startswith(package + "."):
                        module_aliases.add(
                            alias.asname or alias.name.split(".")[0])
    return from_aliases, module_aliases, star


def _called_package_symbols(
    tree: ast.Module,
    from_aliases: dict[str, str],
    module_aliases: set[str],
    star: bool,
    known_symbols: set[str],
) -> list[str]:
    """The package symbols this cell CALLS, ordered by first call site. A
    bare-name call resolves through the from-import aliases (or a star
    import); an attribute call resolves through a package-module alias
    (`mm.kbf_fuse(...)`). Everything else — numpy calls, cell-local
    helpers — is ignored: this is the public-API intersection."""
    hits: list[tuple[int, int, str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        symbol = None
        if isinstance(func, ast.Name):
            symbol = from_aliases.get(func.id)
            if symbol is None and star and func.id in known_symbols:
                symbol = func.id
        elif isinstance(func, ast.Attribute) \
                and isinstance(func.value, ast.Name):
            if func.value.id in module_aliases:
                symbol = func.attr
        if symbol is not None and symbol in known_symbols:
            hits.append((func.lineno, func.col_offset, symbol))
    ordered: list[str] = []
    for _, _, symbol in sorted(hits):
        if symbol not in ordered:
            ordered.append(symbol)
    return ordered


def _primary_functions_by_symbol(
    elements: dict,
) -> dict[str, list[tuple[str, str, list[str]]]]:
    """Top-level symbol -> [(file, qualname, [element ids])]. Only ids
    whose PRIMARY anchor site sits inside a function are associable (the
    cap-to-primary rule): an id that is merely mentioned at an
    orchestrator's call site keeps its note on the dedicated
    implementation. A method's ids resolve through its class name, which
    is what a demo cell actually calls."""
    grouped: dict[str, dict[tuple[str, str], list[str]]] = {}
    for element_id in sorted(elements):
        primary = elements[element_id].get("primary") or {}
        qualname = primary.get("qualname")
        if not qualname or primary.get("placement") not in ("leading", "body"):
            continue  # module-level / class-body primaries: nothing to list
        symbol = qualname.split(".")[0]
        grouped.setdefault(symbol, {}).setdefault(
            (primary.get("file", ""), qualname), []).append(element_id)
    return {
        symbol: [(file, qualname, ids)
                 for (file, qualname), ids in sorted(entries.items())]
        for symbol, entries in grouped.items()
    }


def _inject_implementation_notes(cells: list, run_dir: Path) -> list:
    """The note-4 association + injection pass, run after placeholder
    substitution and before heading anchoring. Returns the cells list
    UNCHANGED (the same object) whenever nothing associates — a run
    without anchors, without a join table, or without package imports
    renders exactly as before this feature existed."""
    elements = load_join_table_elements(run_dir)
    if not elements:
        return cells
    by_symbol = _primary_functions_by_symbol(elements)
    if not by_symbol:
        return cells
    # The package name comes from the join table's own file paths (the
    # run-layout package dir), never from anything paper-specific.
    files = sorted(e["primary"]["file"] for e in elements.values()
                   if (e.get("primary") or {}).get("file"))
    if not files or "/" not in files[0]:
        return cells
    package = files[0].split("/", 1)[0]

    code_trees: dict[int, ast.Module] = {}
    for idx, cell in enumerate(cells):
        if cell.cell_type == "code":
            tree = _cell_ast(cell.source)
            if tree is not None:
                code_trees[idx] = tree
    from_aliases, module_aliases, star = _package_import_aliases(
        list(code_trees.values()), package)
    if not from_aliases and not module_aliases and not star:
        return cells

    known_symbols = set(by_symbol)
    out: list = []
    emitted: set[tuple[str, str]] = set()
    annotated = 0
    for idx, cell in enumerate(cells):
        tree = code_trees.get(idx)
        if tree is None:
            out.append(cell)
            continue
        symbols = _called_package_symbols(
            tree, from_aliases, module_aliases, star, known_symbols)
        functions: list[tuple[str, str, list[str]]] = []
        for symbol in symbols:
            for file, qualname, ids in by_symbol.get(symbol, []):
                if all((file, qualname) != (f, q) for f, q, _ in functions):
                    functions.append((file, qualname, ids))
        if not functions or len(functions) > _MAX_IMPL_FUNCTIONS_PER_CELL:
            # Nothing resolved: a plain non-demo cell. Too many resolved:
            # an overview cell — skip entirely (the cap-or-skip rule).
            out.append(cell)
            continue
        blocks: list[str] = []
        for file, qualname, ids in functions:
            if (file, qualname) in emitted:
                continue  # one note per implementing function, at first use
            emitted.add((file, qualname))
            try:
                blocks.append(make_block(
                    IMPLEMENTATION_NOTE_KIND,
                    {"file": file, "qualname": qualname,
                     "element_ids": ids},
                    run_dir))
            except DerivedBlockError as e:
                # The note is an enrichment; the notebook render must
                # survive a single underivable note.
                print(f"warning: implementation note skipped for "
                      f"`{qualname}`: {e}", file=sys.stderr)
        if blocks:
            out.append(new_markdown_cell("\n\n".join(blocks)))
            annotated += 1
        out.append(cell)
    if not annotated:
        return cells
    print(f"implementation notes: {annotated} demo cell(s) annotated from "
          f"the anchor join table")
    return out


# Registered at import time so the refresh engine's lazy producer import
# finds the kind. Guarded: this module can legitimately be imported under
# two names (`render_notebook` and `scripts.render_notebook`), and the
# second instance must not crash on re-registration.
if IMPLEMENTATION_NOTE_KIND not in registered_block_kinds():
    register_block_kind(IMPLEMENTATION_NOTE_KIND, _derive_implementation_note)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def render(run_dir: Path) -> int:
    draft_path = run_dir / ".pipeline" / "notebook_draft.py"
    params_path = run_dir / ".pipeline" / "params.json"
    spec_path = run_dir / ".pipeline" / "method_spec.json"

    if not draft_path.is_file():
        print(f"error: notebook draft not found at {draft_path}", file=sys.stderr)
        return 1
    if not params_path.is_file():
        print(f"error: params.json not found at {params_path}", file=sys.stderr)
        return 1

    try:
        params_data = json.loads(params_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        print(f"error: params.json invalid JSON: {e}", file=sys.stderr)
        return 1

    spec_data = None
    if spec_path.is_file():
        try:
            loaded_spec = json.loads(spec_path.read_text(encoding="utf-8"))
            if isinstance(loaded_spec, dict):
                spec_data = loaded_spec
        except json.JSONDecodeError:
            # method_spec validation owns malformed-spec failures. Rendering
            # keeps its historical partial-delivery behavior here: params and
            # the draft can still produce a notebook, without a protocol block.
            spec_data = None

    # Partial delivery: an unreadable stub record could hide a partial
    # state, so it fails the render loudly (never a silent complete-looking
    # notebook).
    from partial_delivery import load_stubbed_elements
    stubs, stub_error = load_stubbed_elements(run_dir / ".pipeline")
    if stub_error:
        print(f"error: {stub_error}", file=sys.stderr)
        return 1
    stubs_by_id = {rec["element_id"]: rec for rec in stubs}

    draft_text = draft_path.read_text(encoding="utf-8")
    cells_raw = _parse_jupytext(draft_text)

    # Substitute placeholders.
    cells: list = []
    unhandled_placeholders: list[str] = []
    for kind, source, placeholder_name in cells_raw:
        if placeholder_name:
            if placeholder_name.startswith(COMPONENT_STUB_PREFIX):
                eid = placeholder_name[len(COMPONENT_STUB_PREFIX):]
                rec = stubs_by_id.get(eid)
                if rec is None:
                    unhandled_placeholders.append(placeholder_name)
                    continue
                # Criterion 4: the notice AND the raise, always as a pair.
                # The raising cell is tagged `raises-exception` so the smoke
                # gate (nbclient, allow_errors=False) tolerates EXACTLY the
                # recorded stub cells and nothing else — "PARTIAL cells
                # excepted". Only this renderer path can mint the tag, and
                # only paired with a validated stub record, so a generator
                # cannot use it to sneak a broken cell past the gate. A
                # researcher running the notebook still hits the raise.
                cells.append(new_markdown_cell(_render_stub_markdown(rec)))
                cells.append(new_code_cell(
                    _render_stub_code(rec),
                    metadata={"tags": ["raises-exception"]}))
                continue
            renderer = PLACEHOLDER_RENDERERS.get(placeholder_name)
            if renderer is None:
                unhandled_placeholders.append(placeholder_name)
                continue
            content = renderer(params_data)
            if placeholder_name == "params_dict":
                cells.append(new_code_cell(content))
            else:  # markdown placeholder
                cells.append(new_markdown_cell(content))
                if placeholder_name == "params_table":
                    protocol_block = render_evaluation_protocol_block(
                        spec_data,
                        params_data,
                        heading="### Evaluation protocol",
                    )
                    if protocol_block:
                        cells.append(new_markdown_cell(protocol_block))
        else:
            if kind == "markdown":
                cells.append(new_markdown_cell(source))
            else:
                cells.append(new_code_cell(source))

    if unhandled_placeholders:
        print(
            f"error: unrecognized placeholder name(s): {unhandled_placeholders}. "
            f"Known: {sorted(PLACEHOLDER_RENDERERS.keys())} plus "
            f"`{COMPONENT_STUB_PREFIX}<element_id>` for element ids recorded "
            f"in .pipeline/stubbed_elements.json "
            f"(recorded: {sorted(stubs_by_id) or 'none'})",
            file=sys.stderr,
        )
        return 2

    # Implementation notes (note 4): derived cell-to-element association,
    # injected before heading anchoring so the delivered notebook carries
    # section anchors above every note. No-ops (same cells object) when the
    # run has no anchors.
    cells = _inject_implementation_notes(cells, run_dir)

    # Inject anchors into ## / ### headings and build the TOC. Insert the TOC
    # cell right after the title (cell 0) so the reader sees it immediately.
    toc_entries = _extract_and_anchor_headings(cells)
    if toc_entries:
        toc_cell = new_markdown_cell(_render_toc_cell(toc_entries))
        cells.insert(1, toc_cell)

    # The PARTIAL banner sits directly under the title, ahead of the TOC —
    # injected deterministically whenever stubs are recorded, so a
    # generator that forgot every stub marker still cannot ship a
    # complete-looking notebook.
    if stubs:
        cells.insert(1, new_markdown_cell(_render_partial_banner(run_dir, stubs)))

    # Build and write notebook.
    nb = new_notebook(
        cells=cells,
        metadata={
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
    )

    output_path = run_dir / "notebook.ipynb"
    nbformat.write(nb, output_path)

    n_md = sum(1 for c in cells if c.cell_type == "markdown")
    n_code = sum(1 for c in cells if c.cell_type == "code")
    print(f"wrote {output_path} ({len(cells)} cells: {n_md} markdown, {n_code} code)")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    return render(args.run_dir)


if __name__ == "__main__":
    sys.exit(main())
