"""Routing helpers for review/validator findings.

Pure data-transformation functions: group findings by target agent, filter by
severity, map smoke-gate failure to producer. No I/O.

Self-tests live in tests/test_route_findings.py (B-07).
"""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Iterable


SEVERITY_ORDER = ["nice-to-have", "important", "critical"]


# Notebook §N → producer mapping. Source: the archived v2 orchestrator spec (internal, not shipped) Stage 3.c
# section. Where multiple producers could plausibly own a section (e.g., §1
# imports or §5 call-site-vs-function), this picks the more common case and
# documents the alternative in the comment so callers can override.
SECTION_TO_PRODUCER: dict[int, str] = {
    0: "notebook-generator",   # %pip install — notebook owns this
    1: "notebook-generator",   # imports / setup — call-site, almost always notebook
    2: "notebook-generator",   # params block — wired by notebook
    3: "architecture-coder",   # model build / training / bootstrap
    4: "method-coder",         # component demos — function-internal more often than call-site
    5: "method-coder",         # acquisition loop — function-internal more often than call-site
    6: "notebook-generator",   # own-data appendix — notebook-owned
}


def severity_rank(severity: str) -> int:
    """Lower = less severe. Unknown severities rank at -1."""
    try:
        return SEVERITY_ORDER.index(severity)
    except ValueError:
        return -1


def group_by_target_agent(findings: Iterable[dict]) -> dict[str, list[dict]]:
    """Group findings by their `target_agent` field. Missing field → 'unknown'."""
    out: dict[str, list[dict]] = defaultdict(list)
    for f in findings:
        out[f.get("target_agent") or "unknown"].append(f)
    return dict(out)


def filter_by_severity(
    findings: Iterable[dict], *, min_severity: str
) -> list[dict]:
    """Return findings with severity >= min_severity (per SEVERITY_ORDER)."""
    floor = severity_rank(min_severity)
    if floor < 0:
        raise ValueError(f"unknown min_severity: {min_severity}")
    return [f for f in findings if severity_rank(f.get("severity", "")) >= floor]


def smoke_cell_to_producer(section: int) -> str:
    """Map a notebook section number (0–6) to the producer who likely owns the bug.

    Per the archived v2 orchestrator spec (internal, not shipped) Stage 3.c routing. Returns
    'notebook-generator' as a safe default for unknown sections — it's the
    producer with the broadest scope across the notebook.
    """
    return SECTION_TO_PRODUCER.get(section, "notebook-generator")


# method/<file>.py → producer responsible. data.py / __init__.py issues
# surface at call sites the notebook owns, so notebook-generator is the
# right fix target for those.
_METHOD_FILE_TO_PRODUCER: dict[str, str] = {
    "method": "method-coder",
    "model": "architecture-coder",
    "training": "architecture-coder",
    "data": "notebook-generator",
    "__init__": "notebook-generator",
}

# Strips ANSI color escape sequences that nbclient/IPython emit in tracebacks.
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def smoke_traceback_to_producer(stderr: str) -> str | None:
    """Identify the producer responsible for a smoke-gate failure by walking
    the traceback for frames in the run's `method/` package.

    The section-based routing in `smoke_cell_to_producer` is correct when a
    cell fails AT a call site into producer-owned code, but misroutes when
    the failure is in code the notebook-generator wrote inline (e.g., a
    misused numpy API in a §3 bootstrap cell). The traceback is a more
    reliable signal: whichever `method/*.py` appears deepest in the stack
    owns the bug. No `method/` frame in the traceback → the bug is in the
    notebook code itself → notebook-generator.

    Returns one of 'method-coder' / 'architecture-coder' /
    'notebook-generator', or None when the traceback gives no usable signal
    (caller should fall back to `smoke_cell_to_producer`)."""
    text = _ANSI_RE.sub("", stderr)
    # All method/<file>.py frames, in the order they appear. The traceback's
    # deepest frame is the LAST match because tracebacks print
    # outermost-to-innermost.
    matches = re.findall(r"\bmethod/(\w+)\.py\b", text)
    if matches:
        deepest = matches[-1]
        if deepest in _METHOD_FILE_TO_PRODUCER:
            return _METHOD_FILE_TO_PRODUCER[deepest]
        # Unknown file under method/ — should not happen given the producers'
        # contracts, but route conservatively to notebook-generator.
        return "notebook-generator"

    # No method/ frame: the bug is in code the notebook-generator wrote
    # (either notebook source or stdlib/3rd-party calls made from the notebook).
    if "Cell In[" in text or "<ipython-input-" in text:
        return "notebook-generator"

    return None  # truly ambiguous; fall back to section-based routing


def smoke_traceback_deepest_owned_frame(
    stderr: str, owned_patterns: list[str]
) -> str | None:
    """Return a human-readable anchor for the deepest traceback frame whose
    relative path matches any pattern in `owned_patterns` (fnmatch globs).

    Used by the smoke-gate fix path to anchor the producer agent on the
    specific frame in its WRITEABLE_PATHS where the error fires. The output
    is a single line like:

        method/method.py:175 in compute_3d_box_iou

    suitable for direct inclusion in the dispatch prompt. Returns None when
    no traceback frame matches any owned path (the producer's bug may be at
    a call site outside its scope; the fix-mode (i)–(iv) checklist then
    handles routing via the scope escape hatch)."""
    import fnmatch
    text = _ANSI_RE.sub("", stderr)
    # Two traceback formats:
    # 1. Standard Python:   File "<path>", line <N>, in <func>
    # 2. IPython rich:      File <path>:<N>, in <func>
    # nbclient emits format 2 by default; subprocess-captured Python tracebacks
    # use format 1. Both must match — the bev-distill 2026-05-14 smoke run
    # exposed the gap when only format 1 was supported.
    patterns = [
        re.compile(r'(method/[\w./]+\.py)", line (\d+), in (\w+)'),
        re.compile(r'(method/[\w./]+\.py):(\d+),?\s+in\s+(\w+)'),
    ]
    seen_positions: set[int] = set()
    matches: list[tuple[int, str, str, str]] = []
    for pat in patterns:
        for m in pat.finditer(text):
            if m.start() in seen_positions:
                continue
            seen_positions.add(m.start())
            matches.append((m.start(), m.group(1), m.group(2), m.group(3)))
    if not matches:
        return None
    # Tracebacks print outermost-to-innermost — later in text = deeper.
    matches.sort(key=lambda t: t[0])
    for _, rel_path, line_no, func in reversed(matches):
        if any(fnmatch.fnmatchcase(rel_path, p) for p in owned_patterns):
            return f"{rel_path}:{line_no} in {func}"
    return None


def cell_index_to_section(cell_index: int, notebook_json: dict) -> int:
    """Walk a notebook's cells and find which §N section the given code cell is in.

    Sections are denoted by markdown headers matching `## N.` or `## §N`. The
    driver passes the notebook JSON it already has from rendering; this function
    is purely a search.
    """
    section = 0
    for i, cell in enumerate(notebook_json.get("cells", [])):
        if cell.get("cell_type") == "markdown":
            src = "".join(cell.get("source", []))
            for line in src.splitlines():
                stripped = line.lstrip("# ").lstrip()
                # Match "0." "1." ... "6." at start of a heading
                if len(stripped) >= 2 and stripped[0].isdigit() and stripped[1] == ".":
                    try:
                        section = int(stripped[0])
                    except ValueError:
                        pass
                # Match "§0", "§1", etc.
                elif stripped.startswith("§") and len(stripped) >= 2 and stripped[1].isdigit():
                    section = int(stripped[1])
        if i == cell_index:
            return section
    return section


def critical_findings(findings: Iterable[dict]) -> list[dict]:
    return filter_by_severity(findings, min_severity="critical")


def important_findings(findings: Iterable[dict]) -> list[dict]:
    out = filter_by_severity(findings, min_severity="important")
    return [f for f in out if severity_rank(f.get("severity", "")) < severity_rank("critical")]


def nice_to_have_findings(findings: Iterable[dict]) -> list[dict]:
    return [f for f in findings if f.get("severity") == "nice-to-have"]


def should_halt_stage(findings: Iterable[dict]) -> bool:
    """A stage should halt (or fix-loop) if there's any critical/important finding."""
    return len(filter_by_severity(findings, min_severity="important")) > 0


if __name__ == "__main__":
    raise SystemExit(
        "self-tests moved to tests/test_route_findings.py — run: "
        "python3 -m pytest tests/test_route_findings.py"
    )
