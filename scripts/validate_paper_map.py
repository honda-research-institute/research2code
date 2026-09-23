"""Validate a paper_map.json against the canonical schema.

Stage 1 Step 2 gate. Runs after the decomposer subagent emits paper_map.json,
before Step 3 consumes it. Catches typos, missing fields, wrong types, unknown
fields, duplicate element IDs, unresolved cross-references, and same-type
elements mixing confusable id prefixes (`eq-` vs `equation-`, the id-mutation
invitation from queue item 22) — issues the v1 pipeline silently passed
downstream. Since 2026-07-07 it also enforces the
equation verbatim-quote floor: equation source_text must be a whitespace-
normalized substring of paper.md (found next to the map, or via --paper-md),
so METHOD.md can display the paper's own LaTeX and a backslash that lost its
JSON double-escape fails loudly instead of shipping garbled math.

Usage:
    python scripts/validate_paper_map.py path/to/paper_map.json

Exit codes:
  0  valid
  1  schema validation failed
  3  file not found / unreadable / unparseable JSON
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pydantic import ValidationError  # noqa: E402

from schemas.paper_map import PaperMap  # noqa: E402


def _normalize_ws(text: str) -> str:
    return " ".join((text or "").split())


def check_equation_quotes(paper_map: dict, paper_text: str) -> list[str]:
    """Verbatim-quote floor for equation elements (2026-07-07 pick).

    An equation element's `source_text` promises the paper's own words —
    since 2026-07-07 that means the equation EXACTLY as the paper writes
    it, LaTeX and delimiters included. The check is a whitespace-
    normalized substring test against paper.md, the same anti-
    hallucination genus as the explanation quote floor. It is also the
    JSON-corruption detector for backslash-heavy quotes: a `\\theta`
    that lost its double-escape parses as tab + "heta" and can never be
    a substring of the paper, so the corruption fails loudly here
    instead of shipping garbled math."""
    haystack = _normalize_ws(paper_text)
    errors: list[str] = []
    for element in paper_map.get("elements", []):
        if element.get("type") != "equation":
            continue
        quote = str(element.get("source_text") or "")
        if not quote.strip():
            continue
        if _normalize_ws(quote) not in haystack:
            head = _normalize_ws(quote)[:120]
            # Over-escape detector: when collapsing doubled backslashes in
            # the PARSED quote produces a real paper passage, the file
            # carried one escaping level too many (\\\\dot where the paper
            # has \dot). The repr rendering below makes that nearly
            # invisible to judges and fixers — both iDb-RRT stage-1
            # terminal failures and ACC's quote rejection on 2026-07-13
            # were this exact shape, twice misdiagnosed as a delimiter
            # problem — so the error names it outright.
            collapsed = quote.replace("\\\\", "\\")
            if collapsed != quote and _normalize_ws(collapsed) in haystack:
                errors.append(
                    f"  - {element.get('id', '?')}: source_text has its "
                    f"backslashes doubled ONE LEVEL TOO MANY — collapsing "
                    f"each \\\\ to \\ yields a verbatim paper passage. In "
                    f"the JSON file, write exactly TWO backslash characters "
                    f"per LaTeX backslash (\\\\theta in the file bytes), "
                    f"not four. Fix: halve the backslashes in this quote. "
                    f"Got: {head!r}"
                )
                continue
            errors.append(
                f"  - {element.get('id', '?')}: source_text is not a verbatim "
                f"passage of the paper (whitespace-normalized substring check "
                f"failed). Quote equations EXACTLY as the paper writes them, "
                f"LaTeX and delimiters included; remember every backslash in "
                f"the JSON file must be doubled (\\\\theta, not \\theta). "
                f"Got: {head!r}"
            )
    return errors


def check_id_prefix_consistency(paper_map: dict) -> list[str]:
    """Same-type elements must not mix confusable id prefixes (item 22).

    The iDb-RRT map wrote `equation-continuous-dynamics` next to six
    `eq-*` equations, and downstream agents regenerated the odd one out
    from language (`eq-dynamics-continuous`), which strict consumers then
    rejected. The confusable shape is specific: two prefixes in use for
    ONE element type where one prefix is a leading substring of the other
    (`eq` vs `equation`, `prop` vs `property`). Genuinely different
    prefixes coexisting is not flagged."""
    by_type: dict[str, dict[str, list[str]]] = {}
    for element in paper_map.get("elements", []):
        el_id = str(element.get("id") or "")
        el_type = str(element.get("type") or "")
        if not el_id or not el_type:
            continue
        prefix = el_id.split("-")[0]
        by_type.setdefault(el_type, {}).setdefault(prefix, []).append(el_id)

    errors: list[str] = []
    for el_type, groups in sorted(by_type.items()):
        prefixes = sorted(groups)
        for i, shorter in enumerate(prefixes):
            for longer in prefixes[i + 1:]:
                if not longer.startswith(shorter):
                    continue
                minority, majority = (
                    (longer, shorter)
                    if len(groups[longer]) <= len(groups[shorter])
                    else (shorter, longer)
                )
                errors.append(
                    f"  - {el_type} elements mix the id prefixes "
                    f"`{shorter}-` and `{longer}-`, which invites agents "
                    f"downstream to regenerate one as the other. Rename "
                    f"the `{minority}-` element(s) to the `{majority}-` "
                    f"prefix ({', '.join(sorted(groups[minority]))}) and "
                    f"update every dependencies/related_equations entry "
                    f"that references them."
                )
    return errors


def _format_pydantic_errors(exc: ValidationError) -> str:
    lines = [f"schema validation failed ({exc.error_count()} error(s)):"]
    for err in exc.errors():
        loc = ".".join(str(x) for x in err["loc"])
        msg = err["msg"]
        lines.append(f"  - {loc}: {msg} [{err['type']}]")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("paper_map_path", type=Path)
    parser.add_argument(
        "--paper-md", type=Path, default=None,
        help="paper.md for the equation verbatim-quote check "
             "(default: paper.md next to the paper map; the check is "
             "skipped when absent)")
    args = parser.parse_args()

    if not args.paper_map_path.exists():
        print(f"error: file not found: {args.paper_map_path}", file=sys.stderr)
        return 3

    try:
        text = args.paper_map_path.read_text(encoding="utf-8")
    except OSError as e:
        print(f"error: cannot read {args.paper_map_path}: {e}", file=sys.stderr)
        return 3

    try:
        json.loads(text)
    except json.JSONDecodeError as e:
        print(f"error: {args.paper_map_path} is not valid JSON: {e}", file=sys.stderr)
        return 3

    try:
        paper_map = PaperMap.model_validate_json(text)
    except ValidationError as exc:
        print(_format_pydantic_errors(exc), file=sys.stderr)
        return 1

    prefix_errors = check_id_prefix_consistency(json.loads(text))
    if prefix_errors:
        print(f"id prefix consistency check failed "
              f"({len(prefix_errors)} error(s)):", file=sys.stderr)
        for line in prefix_errors:
            print(line, file=sys.stderr)
        return 1

    paper_md = args.paper_md or args.paper_map_path.parent / "paper.md"
    if paper_md.exists():
        quote_errors = check_equation_quotes(
            json.loads(text), paper_md.read_text(encoding="utf-8"))
        if quote_errors:
            print(f"equation verbatim-quote check failed "
                  f"({len(quote_errors)} element(s)):", file=sys.stderr)
            for line in quote_errors:
                print(line, file=sys.stderr)
            return 1
    else:
        print(f"note: {paper_md} not found; equation verbatim-quote "
              f"check skipped (standalone validation)")

    type_counts: dict[str, int] = {}
    for e in paper_map.elements:
        type_counts[e.type.value] = type_counts.get(e.type.value, 0) + 1
    counts_str = ", ".join(f"{t}={n}" for t, n in sorted(type_counts.items()))
    print(
        f"ok: {args.paper_map_path} validates against schema "
        f"v{paper_map.schema_version} ({len(paper_map.elements)} elements: {counts_str})"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
