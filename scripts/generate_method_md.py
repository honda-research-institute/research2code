"""METHOD.md generator — the standalone explanation deliverable (slice 2.1).

The founding ask (Kiura-san, top priority since 2026-05-21): researchers
need to understand WHAT the math is doing, WHY it's novel, and WHAT the
intuition is — not just see it transcribed. The product rules from
hri-requirements.md:

- **Survives any downstream failure.** Generated from stage-0/1 artifacts
  alone: paper_map.json is the only REQUIRED input. method_spec.json and
  params.json enrich it when present; their absence degrades the affected
  section honestly (a visible "not derived" note, never a silent gap).
  When the decomposition contains no method-shaped material, the output is
  labeled as a source-map index rather than a completed explanation.
- **Two-layer authorship, merged deterministically.** This script owns the
  structure: per-equation sections with the paper's own statement quoted,
  the algorithm walkthrough, the parameter provenance table, the source
  map. The plain-language prose (what / why-novel / intuition per element)
  comes from the explainer agent as a STRUCTURED JSON sidecar
  (--explanations), so the LLM writes content into typed slots and the
  US-3-style quote checks can verify any paper claims it makes. Without
  the sidecar, every explanation slot renders an explicit pending marker —
  the document is useful (structure, quotes, provenance) and honest about
  what's missing.
- **Chat-readable from day one** (the design-ahead rule): stable per-element
  anchors keyed by paper_map element ids, a machine-readable generation
  record in an HTML comment, deterministic output (same inputs, same bytes).
- **Equation-level code correspondence** (Japan-feedback note 5): when the
  run's delivered code carries validated `# paper-element:` anchors, each
  anchored equation/algorithm section gains an "implemented by" line (the
  primary implementing function, extra anchor sites as "also used in"), and
  the source map gains an implementation column — the single complete
  correspondence table. Line-number-bearing content is emitted as marked
  derived blocks so scripts/refresh_derived_blocks.py keeps it fresh after
  companion edits. Elements with no implementing function render exactly
  as before; a run without anchors renders byte-identical to the
  pre-correspondence output.

Explanations sidecar schema:
    {"schema_version": "1.0.0",
     "explanations": {"<element_id>": {
         "what": str, "why_novel": str, "intuition": str}}}

Usage:
    python3 scripts/generate_method_md.py --run-dir r2c_runs/<slug> \
        [--explanations <path>] [--output <path>]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from refresh_derived_blocks import (
    BEGIN_MARKER_RE,
    DerivedBlockError,
    make_block,
    register_block_kind,
    registered_block_kinds,
)
from evaluation_protocol_rendering import render_evaluation_protocol_block

SCHEMA_VERSION = "1.0.0"
# True on both paths that leave a field empty: the explainer pass was
# skipped entirely, or this specific part failed parsing after retry.
PENDING = ("*(explanation pending — the automatic explainer could not "
           "complete this part; the structure, quotes and provenance "
           "below are complete)*")
UNAVAILABLE = (
    "**Method explanation unavailable.** The decomposition contains no method "
    "specification, algorithm element, or implement/demonstrate equation. The "
    "sections below are availability notes and a source-map index, not a "
    "complete plain-language explanation."
)

_LEGACY_UNAVAILABLE_MARKERS = (
    "method summary unavailable — no spec and no algorithm element",
    "paper map records no implement/demonstrate equations",
    "no algorithm element in the paper map",
)

# Equation/concept elements with these code_role values appear in the key
# section, in this order; the rest land in the source map only.
_EXPLAINED_ROLES = ("implement", "demonstrate")


def method_md_has_substantive_explanation(text: str) -> bool:
    """Whether a generated METHOD.md contains method-shaped explanation.

    New documents carry ``UNAVAILABLE`` explicitly.  The marker conjunction
    keeps reports honest when re-rendering pre-change concept/property-only
    outputs such as RFC specifications, without treating an algorithm-only or
    equation-led document as unavailable merely because one section is empty.
    Unknown hand-authored/legacy documents remain present-by-default.
    """
    if not text.strip():
        return False
    if UNAVAILABLE in text:
        return False
    lowered = text.lower()
    return not all(marker in lowered for marker in _LEGACY_UNAVAILABLE_MARKERS)


def _load_json(path: Path) -> dict | None:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def _anchor(element_id: str) -> str:
    return f'<a id="{element_id}"></a>'


# ---------------------------------------------------------------------------
# Equation-level code correspondence (Japan-feedback note 5)
#
# The anchor join table (scripts/build_anchor_join_table.py) answers "which
# function implements element X" from the delivered code's validated
# `# paper-element:` anchors. This section renders that answer into the
# document: an "implemented by" line per anchored equation/algorithm section
# and an implementation column on the source map. Both are line-number
# bearing, so both are emitted as marked derived blocks
# (scripts/refresh_derived_blocks.py) and re-derived from the CURRENT code
# on refresh after companion edits.
# ---------------------------------------------------------------------------

IMPLEMENTED_BY_KIND = "implemented_by"
SOURCE_MAP_KIND = "source_map"
# The notebook renderer's block kind (registered by scripts/
# render_notebook.py). The literal is repeated here so this module never
# imports the nbformat-heavy notebook renderer; equality is pinned by
# tests/test_method_md_correspondence.py.
NOTEBOOK_NOTE_KIND = "implementation_note"

_SEC_ANCHOR_RE = re.compile(r'<a id="(sec-[^"]*)"></a>')


def load_join_table_elements(run_dir: Path) -> dict:
    """The anchor join table's elements map for a run: the artifact at
    .pipeline/anchor_join_table.json when the run carries one, else built
    in-process from the delivered code (renderer-only consumption — the
    driver stays unwired, per the note-4/5 design). Returns {} when the run
    has no correspondence to offer (no method/ package, no paper map, no
    anchors), which is what keeps an anchor-free run's output byte-identical
    to the pre-correspondence renderers. Broken anchors (ids missing from
    the paper map) degrade with a warning rather than failing the whole
    document — the coder gates own that failure, not the renderers."""
    from build_anchor_join_table import (ARTIFACT_REL_PATH, JoinTableError,
                                         JoinTableSetupError, build_join_table)
    run_dir = Path(run_dir)
    artifact = _load_json(run_dir / ARTIFACT_REL_PATH)
    if artifact and isinstance(artifact.get("elements"), dict):
        return artifact["elements"]
    try:
        return build_join_table(run_dir).get("elements", {})
    except JoinTableSetupError:
        return {}
    except JoinTableError as e:
        print(f"warning: anchor join table unavailable, correspondence "
              f"surfaces skipped — {e}", file=sys.stderr)
        return {}


def _fresh_join_elements(run_dir: Path) -> dict:
    """Refresh-side join table: ALWAYS rebuilt from the current delivered
    code, never read from the (possibly stale) artifact — freshness is the
    whole point of a derived block."""
    from build_anchor_join_table import (JoinTableError, JoinTableSetupError,
                                         build_join_table)
    try:
        return build_join_table(Path(run_dir)).get("elements", {})
    except (JoinTableError, JoinTableSetupError) as e:
        raise DerivedBlockError(
            f"anchor join table cannot be rebuilt from the delivered code: "
            f"{e}")


def _dedupe_sites(sites: list[dict]) -> list[dict]:
    """One entry per enclosing scope, ordered by (file, line) — an id
    anchored twice in the same function is one 'used in' mention."""
    seen: set = set()
    out: list[dict] = []
    for site in sorted(sites, key=lambda s: (s["file"], s["line"])):
        key = (site["file"], site.get("qualname"))
        if key in seen:
            continue
        seen.add(key)
        out.append(site)
    return out


def _site_ref(site: dict, *, link: bool) -> str:
    """One anchor site as reader-facing markdown. The line number is the
    enclosing def/class line (where a reader navigates), not the anchor's
    own line."""
    rel = site["file"]
    qualname = site.get("qualname")
    if qualname is None:
        text = f"`{rel}:{site['line']}` (module level)"
        return f"module-level code in [`{rel}:{site['line']}`]({rel})" \
            if link else text
    line = site.get("function_line") or site["line"]
    name = f"[`{qualname}`]({rel})" if link else f"`{qualname}`"
    return f"{name} (`{rel}:{line}`)"


def _notebook_demo_link(element_id: str, run_dir: Path) -> str | None:
    """Best-effort 'demonstrated in the notebook' cross-link, riding note
    4's derived cell association: emitted ONLY when the delivered notebook
    carries an implementation-note block naming this element — nothing is
    promised where the association did not resolve. The fragment is the
    nearest preceding section anchor the notebook renderer injected (an
    anchor-based fragment; resolution is viewer-dependent by design)."""
    nb_path = Path(run_dir) / "notebook.ipynb"
    if not nb_path.is_file():
        return None
    try:
        nb = json.loads(nb_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None
    sources: list[tuple[str, str]] = []
    for cell in nb.get("cells") or []:
        src = cell.get("source", "")
        if isinstance(src, list):
            src = "".join(src)
        sources.append((cell.get("cell_type", ""), src))
    for idx, (cell_type, src) in enumerate(sources):
        if cell_type != "markdown":
            continue
        for line in src.split("\n"):
            marker = BEGIN_MARKER_RE.match(line)
            if not marker:
                continue
            try:
                header = json.loads(marker.group(1))
            except json.JSONDecodeError:
                continue
            if header.get("kind") != NOTEBOOK_NOTE_KIND:
                continue
            spec = header.get("spec") or {}
            if element_id not in (spec.get("element_ids") or []):
                continue
            fragment = None
            for j in range(idx, -1, -1):
                if sources[j][0] != "markdown":
                    continue
                anchors = _SEC_ANCHOR_RE.findall(sources[j][1])
                if anchors:
                    fragment = anchors[-1]
                    break
            target = f"notebook.ipynb#{fragment}" if fragment \
                else "notebook.ipynb"
            return f"demonstrated in [the notebook]({target})"
    return None


def _implemented_by_line(element_id: str, entry: dict, run_dir: Path) -> str:
    """The 'implemented by' line: primary implementing function with file
    location, additional anchor sites as 'also used in' (never a bare
    multi-function list), plus the best-effort notebook cross-link."""
    primary = entry["primary"]
    parts = [f"**Implemented by:** {_site_ref(primary, link=True)}"]
    primary_key = (primary["file"], primary.get("qualname"))
    extras = [s for s in _dedupe_sites(entry["sites"])
              if (s["file"], s.get("qualname")) != primary_key]
    if extras:
        parts.append("also used in "
                     + ", ".join(_site_ref(s, link=False) for s in extras))
    demo = _notebook_demo_link(element_id, run_dir)
    if demo:
        parts.append(demo)
    return " · ".join(parts)


def _derive_implemented_by(spec: dict, run_dir: Path) -> str:
    element_id = spec.get("element_id")
    if not element_id or not isinstance(element_id, str):
        raise DerivedBlockError(
            f"implemented_by spec needs a string 'element_id'; got {spec!r}")
    entry = _fresh_join_elements(run_dir).get(element_id)
    if entry is None:
        raise DerivedBlockError(
            f"implemented_by: `{element_id}` has no anchor in the delivered "
            f"code — if the anchors were removed after delivery, remove this "
            f"block rather than leaving a stale implementation claim")
    return _implemented_by_line(element_id, entry, Path(run_dir))


def _implementation_cell(entry: dict | None) -> str:
    """The source map's implementation column for one element: the primary
    implementing function, or an honest em-dash when no anchor claims the
    element (the role column already says 'explained only' in role terms)."""
    if not entry:
        return "—"
    primary = entry["primary"]
    qualname = primary.get("qualname")
    if qualname is None:
        return f"`{primary['file']}:{primary['line']}` (module level)"
    line = primary.get("function_line") or primary["line"]
    return f"`{qualname}` — `{primary['file']}:{line}`"


def _derive_source_map(spec: dict, run_dir: Path) -> str:
    paper_map = _load_json(Path(run_dir) / ".pipeline" / "paper_map.json")
    if paper_map is None:
        raise DerivedBlockError(
            "source_map: .pipeline/paper_map.json missing or unparseable — "
            "the correspondence table cannot be rebuilt without it")
    return _source_map(paper_map, _fresh_join_elements(run_dir))


def _blockquote(text: str) -> str:
    lines = (text or "").strip().splitlines() or [""]
    return "\n".join(f"> {line}".rstrip() for line in lines)


def _stated_quote(element: dict) -> str:
    """The "The paper states:" body for an element.

    Verbatim math quotes (the 2026-07-07 equation-rendering pick) carry
    the paper's own LaTeX, `$`/`$$` delimiters included. A `> ` prefix
    breaks display-math rendering in VS Code and GitHub, so math quotes
    stand alone; quotes without math delimiters — all pre-change
    artifacts included — keep the blockquote unchanged.

    A missing verbatim quote falls back to the decomposition's own
    description, VISIBLY marked as a paraphrase (researcher feedback 2026-07:
    readers used the block to check explanations against the paper's
    words and could not tell quote from paraphrase)."""
    quote = str(element.get("source_text") or "")
    if "$" in quote:
        return quote.strip()
    if quote:
        return _blockquote(quote)
    return (
        "*(paraphrase — the verbatim quote was not captured for this "
        "element, so the text below is the decomposition's description "
        "of it, not the paper's own words)*\n\n"
        + _blockquote(element.get("description", "")))


def _explained_equations(paper_map: dict) -> list[dict]:
    eqs = [e for e in paper_map.get("elements", [])
           if e.get("type") == "equation"
           and e.get("code_role") in _EXPLAINED_ROLES]
    role_rank = {r: i for i, r in enumerate(_EXPLAINED_ROLES)}
    return sorted(eqs, key=lambda e: (role_rank.get(e.get("code_role"), 9),
                                      str(e.get("section", "")),
                                      str(e.get("id", ""))))


def _equation_section(element: dict, explanation: dict | None,
                      used_by: list[dict] | None = None,
                      impl_block: str | None = None) -> str:
    parts = [
        f"### {_anchor(element.get('id', ''))}{element.get('name', element.get('id', '?'))}",
        "",
        f"*{element.get('section', 'section not recorded')}"
        f" · role: {element.get('code_role', '?')}*",
    ]
    if impl_block:
        parts += ["", impl_block]
    parts += [
        "",
        "**The paper states:**",
        "",
        _stated_quote(element),
    ]
    pseudo = element.get("pseudocode")
    if pseudo:
        parts += ["", "```text", str(pseudo).strip(), "```"]
    exp = explanation or {}
    for label, key in (("What it does", "what"),
                       ("Why it's novel", "why_novel"),
                       ("The intuition", "intuition")):
        body = str(exp.get(key, "")).strip() or PENDING
        parts += ["", f"**{label}:** {body}"]
    deps = element.get("dependencies") or []
    if deps:
        links = ", ".join(f"[{d}](#{d})" for d in deps)
        parts += ["", f"*Builds on: {links}*"]
    if used_by:
        links = ", ".join(
            f"[{a.get('name', a.get('id', '?'))}](#{a.get('id', '')})"
            for a in used_by)
        parts += ["", "*Used in the "
                      "[algorithm walkthrough](#algorithm-walkthrough): "
                      f"{links}*"]
    return "\n".join(parts)


def _algorithm_walkthrough(paper_map: dict, spec: dict | None,
                           explained_names: dict[str, str] | None = None,
                           impl_blocks: dict[str, str] | None = None
                           ) -> str:
    explained_names = explained_names or {}
    impl_blocks = impl_blocks or {}
    parts: list[str] = []
    for element in paper_map.get("elements", []):
        if element.get("type") != "algorithm":
            continue
        parts += [
            f"### {_anchor(element.get('id', ''))}{element.get('name', '?')}",
            "",
            f"*{element.get('section', 'section not recorded')}*",
        ]
        impl_block = impl_blocks.get(str(element.get("id", "")))
        if impl_block:
            parts += ["", impl_block]
        parts += [
            "",
            str(element.get("description", "")).strip(),
        ]
        pseudo = element.get("pseudocode")
        if pseudo:
            parts += ["", "```text", str(pseudo).strip(), "```"]
        eq_deps = [d for d in (element.get("dependencies") or [])
                   if d in explained_names]
        if eq_deps:
            links = ", ".join(f"[{explained_names[d]}](#{d})"
                              for d in eq_deps)
            parts += ["", f"*Equations in this algorithm, explained "
                          f"above: {links}*"]
        parts += [""]
    if spec:
        pc = (spec.get("comparison") or {}).get("pluggable_component") or {}
        if pc.get("name"):
            parts += [
                "### Where the method plugs in",
                "",
                f"The paper's contribution is packaged as "
                f"`{pc.get('signature', pc['name'])}` — "
                f"{str(pc.get('description', '')).strip()}",
            ]
    if not parts:
        parts = ["*(no algorithm element in the paper map — see the "
                 "equation sections above)*"]
    return "\n".join(parts).rstrip()


def _provenance_table(params: dict | None) -> str:
    if not params:
        return ("*(not derived — the run stopped before parameter "
                "derivation; equations and walkthrough above are "
                "unaffected)*")
    rows = ["| parameter | value | source | paper says |",
            "|---|---|---|---|"]
    entries = params.get("params", params)
    for name in sorted(entries):
        entry = entries[name]
        if not isinstance(entry, dict):
            rows.append(f"| {name} | {entry} | ? | |")
            continue
        source = str(entry.get("source", "?"))
        paper_bits = []
        if entry.get("paper_section"):
            paper_bits.append(str(entry["paper_section"]))
        if entry.get("paper_value") not in (None, ""):
            paper_bits.append(f"paper value: {entry['paper_value']}")
        if entry.get("paper_says"):
            paper_bits.append(f'"{entry["paper_says"]}"')
        rows.append(f"| {name} | {entry.get('value', '?')} | {source} | "
                    f"{'; '.join(paper_bits)} |")
    return "\n".join(rows)


def _provenance_summary(params: dict | None) -> str:
    """A one-line confidence indicator above the provenance table: how many
    parameters come straight from the paper versus how many were supplemented
    where the paper is silent. The supplemented count is the rough "how far to
    trust the code against the paper" signal a reviewer asked for. Any source
    that is neither the paper's own value nor a demo-scaled paper value counts
    as supplemented, so the three buckets always sum to the total."""
    if not params:
        return ""
    entries = params.get("params", params)
    if not isinstance(entries, dict):
        return ""
    total = sum(1 for e in entries.values() if isinstance(e, dict))
    if not total:
        return ""

    def _count(src: str) -> int:
        return sum(1 for e in entries.values()
                   if isinstance(e, dict) and str(e.get("source")) == src)

    from_paper = _count("paper")
    scaled = _count("system_default")
    supplemented = total - from_paper - scaled
    lines = [f"**{from_paper} of {total} parameters** are taken directly "
             f"from the paper."]
    if scaled:
        lines.append(
            f"{scaled} use a runtime value that differs from the paper's "
            "stated value, with the paper value preserved for comparison."
        )
    if supplemented:
        lines.append(f"{supplemented} were supplemented from field conventions "
                     f"where the paper does not specify them, and are the "
                     f"values to scrutinise most when judging how closely the "
                     f"code follows the paper.")
    return "_" + " ".join(lines) + "_"


def _source_map(paper_map: dict,
                impl_elements: dict | None = None) -> str:
    """The source-map table. With `impl_elements` (the join table's elements
    map), an implementation column completes the correspondence: element,
    paper location, role, implementing code. Without it — every anchor-free
    run — the table renders byte-identically to the pre-correspondence
    format."""
    if impl_elements is None:
        rows = ["| element | type | paper location | role |",
                "|---|---|---|---|"]
    else:
        rows = ["| element | type | paper location | role | implementation |",
                "|---|---|---|---|---|"]
    for element in paper_map.get("elements", []):
        row = (
            f"| {_anchor(element.get('id', ''))}{element.get('name', '?')} "
            f"| {element.get('type', '?')} "
            f"| {element.get('section', '?')} "
            f"| {element.get('code_role', '?')} |")
        if impl_elements is not None:
            impl = _implementation_cell(impl_elements.get(
                str(element.get("id", ""))))
            row += f" {impl} |"
        rows.append(row)
    return "\n".join(rows)


def generate_method_md(run_dir: Path,
                       explanations_path: Path | None = None) -> str:
    pipeline = run_dir / ".pipeline"
    paper_map = _load_json(pipeline / "paper_map.json")
    if paper_map is None:
        raise FileNotFoundError(
            f"paper_map.json missing or unparseable under {pipeline} — "
            f"METHOD.md needs at least the stage-1 decomposition")
    spec = _load_json(pipeline / "method_spec.json")
    params = _load_json(pipeline / "params.json")
    explanations: dict = {}
    explained_with = "none"
    if explanations_path is not None:
        sidecar = _load_json(explanations_path)
        if sidecar:
            explanations = sidecar.get("explanations", {})
            explained_with = explanations_path.name

    title = paper_map.get("title", run_dir.name)
    equations = _explained_equations(paper_map)
    summary = ""
    if spec:
        cm = spec.get("core_method") or {}
        summary = str(cm.get("summary", "")).strip()
    algs = [e for e in paper_map.get("elements", [])
            if e.get("type") == "algorithm"]
    if not summary:
        summary = str(algs[0].get("description", "")).strip() if algs else \
            "*(method summary unavailable — no spec and no algorithm element)*"

    # Equation-level code correspondence (note 5): which elements the
    # delivered code claims via validated anchors. {} for every run without
    # anchors, and every surface below renders exactly as before.
    correspondence = load_join_table_elements(run_dir)

    def _impl_block(element_id: str) -> str | None:
        """A minted implemented-by derived block, or None where the join
        table claims nothing (explained-only elements render as today). A
        derivation failure degrades that one line with a warning — the
        document itself must survive."""
        if element_id not in correspondence:
            return None
        try:
            return make_block(IMPLEMENTED_BY_KIND,
                              {"element_id": element_id}, run_dir)
        except DerivedBlockError as e:
            print(f"warning: implemented-by line skipped for "
                  f"`{element_id}`: {e}", file=sys.stderr)
            return None

    explained_names = {
        str(e.get("id", "")): str(e.get("name", e.get("id", "?")))
        for e in equations}
    walkthrough = _algorithm_walkthrough(
        paper_map, spec, explained_names,
        impl_blocks={str(a.get("id", "")): block for a in algs
                     if (block := _impl_block(str(a.get("id", ""))))})
    # Reverse index for the equation-section cross-links: which algorithm
    # elements list each explained equation as a dependency.
    used_by: dict[str, list[dict]] = {}
    for alg in algs:
        for dep in alg.get("dependencies") or []:
            if dep in explained_names:
                used_by.setdefault(dep, []).append(alg)
    substantive = bool(
        equations
        or algs
        or not walkthrough.startswith("*(no algorithm element in the paper map")
        or not summary.startswith("*(method summary unavailable")
    )

    record = json.dumps({
        "schema_version": SCHEMA_VERSION,
        "inputs": {
            "paper_map": True,
            "method_spec": spec is not None,
            "params": params is not None,
            "explanations": explained_with,
        },
        "explained_equations": [e.get("id") for e in equations],
    }, sort_keys=True)

    title_suffix = "the method, explained" if substantive else \
        "method explanation unavailable"
    if substantive:
        introduction = (
            "This document explains the paper's method on its own: what the "
            "key equations do, why they're novel, and the intuition behind "
            "them, with every claim anchored to the paper. It is generated "
            "from the pipeline's paper decomposition and survives any "
            "downstream failure — code or no code, this explanation stands "
            "alone."
        )
    else:
        introduction = UNAVAILABLE

    sections = [
        f"<!-- method-md {record} -->",
        f"# {title} — {title_suffix}",
        "",
        introduction,
        "",
    ]
    if substantive:
        sections += [
            "**How to read this document.** The equation sections follow "
            "the paper's own order. If you prefer to start from the "
            "algorithm and follow the math from there, jump to the "
            "[algorithm walkthrough](#algorithm-walkthrough) — its steps "
            "link back to each equation's explanation.",
            "",
        ]
    sections += [
        "## What this method does",
        "",
        summary,
        "",
        "## The key equations",
        "",
        "Each section below covers one equation the pipeline implements "
        "or demonstrates in code. The **The paper states** block quotes "
        "the paper's own verbatim statement of the equation, so you can "
        "check the explanation against the paper's words (a marked "
        "paraphrase appears when the verbatim quote was not captured). "
        "Equations with any other role are not explained here; they are "
        "indexed in the [source map](#source-map) at the end.",
        "",
    ]
    if equations:
        sections.append("\n\n".join(
            _equation_section(e, explanations.get(e.get("id")),
                              used_by.get(str(e.get("id", ""))),
                              impl_block=_impl_block(str(e.get("id", ""))))
            for e in equations))
    else:
        sections.append("*(the paper map records no implement/demonstrate "
                        "equations — see the source map below for "
                        "everything it does record)*")
    prov_summary = _provenance_summary(params)
    prov_block = _provenance_table(params)
    if not substantive and not params:
        prov_block = (
            "*(not derived — the run stopped before parameter derivation; "
            "the decomposition did not expose a substantive method "
            "explanation)*"
        )
    if prov_summary:
        prov_block = f"{prov_summary}\n\n{prov_block}"
    evaluation_protocol_block = render_evaluation_protocol_block(spec, params)
    # The source map: with correspondence, a marked derived block carrying
    # the implementation column; without, the pre-correspondence table,
    # byte-identical to before.
    source_map_block = _source_map(paper_map)
    if correspondence:
        try:
            source_map_block = make_block(SOURCE_MAP_KIND, {}, run_dir)
        except DerivedBlockError as e:
            print(f"warning: source-map implementation column skipped: {e}",
                  file=sys.stderr)
    sections += [
        "",
        '## <a id="algorithm-walkthrough"></a>Algorithm walkthrough',
        "",
        walkthrough,
    ]
    if evaluation_protocol_block:
        sections += [
            "",
            evaluation_protocol_block,
        ]
    sections += [
        "",
        "## Parameters and provenance",
        "",
        prov_block,
        "",
        '## <a id="source-map"></a>Source map',
        "",
        "This table is the completeness index and citation trail for the "
        "whole document: one row for every element the decomposition "
        "extracted from the paper — including the ones that did not get "
        "a full section above — with the paper location it came from and "
        "the role the pipeline assigned it. Use it to confirm nothing "
        "the decomposition found was silently dropped, and to jump from "
        "any element to where it lives in the paper.",
        "",
        source_map_block,
        "",
    ]
    return "\n".join(sections)


# Note-5 block kinds, registered at import time so the refresh engine's
# lazy producer import (refresh_derived_blocks._import_producer_kinds) finds
# them. Guarded: this module can legitimately be imported under two names
# (`generate_method_md` and `scripts.generate_method_md`), and the second
# instance must not crash on re-registration.
if IMPLEMENTED_BY_KIND not in registered_block_kinds():
    register_block_kind(IMPLEMENTED_BY_KIND, _derive_implemented_by)
if SOURCE_MAP_KIND not in registered_block_kinds():
    register_block_kind(SOURCE_MAP_KIND, _derive_source_map)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--explanations", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None,
                        help="default: <run-dir>/METHOD.md")
    args = parser.parse_args(argv)

    try:
        content = generate_method_md(args.run_dir, args.explanations)
    except FileNotFoundError as e:
        print(f"FAIL: {e}", file=sys.stderr)
        return 1
    out = args.output or (args.run_dir / "METHOD.md")
    out.write_text(content, encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
