"""Deterministic gate for the method-explainer's sidecar (slice 2.1).

The explainer agent writes structured explanations
(.pipeline/method_explanations.json) that generate_method_md.py merges
into METHOD.md. This validator makes the explanation layer hold to the
same honesty bar as the params layer:

- every explained id must exist in paper_map.json (no fabricated elements)
- every implement/demonstrate equation must be covered (the founding ask
  is per-KEY-equation explanation; partial coverage is an error that
  lists exactly what's missing)
- each entry carries non-trivial what / why_novel / intuition prose
- any verbatim quotation (double-quoted span of 5+ words) must actually
  appear in the paper — the fabricated-quote class (US-3) applied to
  explanations, reusing the same windowed token matcher so Marker
  artifacts and emphasis runs don't false-positive

Exit 1 on errors; --report writes a chat-readable verdict JSON.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from generate_method_md import _explained_equations  # noqa: E402
from passage_locator import locate_passages  # noqa: E402
from schemas.paper_map import nearest_id  # noqa: E402
from validate_params_provenance import _norm, _quote_in_paper, _tokens  # noqa: E402

MIN_FIELD_CHARS = 20
_QUOTED_RE = re.compile(r'"([^"]{20,400})"')
REQUIRED_FIELDS = ("what", "why_novel", "intuition")


def check_explanations(sidecar: dict, paper_map: dict,
                       paper_text: str | None) -> list[dict]:
    findings: list[dict] = []
    explanations = sidecar.get("explanations")
    if not isinstance(explanations, dict):
        return [{"severity": "error", "check": "schema",
                 "message": "sidecar lacks an 'explanations' object"}]

    known_ids = {e.get("id") for e in paper_map.get("elements", []) if e.get("id")}
    for eid in explanations:
        if eid not in known_ids:
            # The retry-convergence hint (item 22): the ADAM run keyed an
            # entry `eq-temporal-averaging` for the map's
            # `eq-temporal-average`, and the retry repeated the near-miss
            # because the message named only what failed, not the real id.
            near = nearest_id(eid, known_ids)
            hint = (f" Nearest existing id: {near!r} — if that is the "
                    f"element you meant, re-key your entry to it EXACTLY."
                    if near else "")
            findings.append({
                "severity": "error", "check": "fabricated_element",
                "element_id": eid,
                "message": f"explained element {eid!r} does not exist in "
                           f"the paper map — explanations must anchor to "
                           f"real extracted elements.{hint}"})

    required = [e.get("id") for e in _explained_equations(paper_map)]
    missing = [eid for eid in required if eid not in explanations]
    if missing:
        findings.append({
            "severity": "error", "check": "coverage",
            "element_ids": list(missing),
            "message": f"key equations without explanations: "
                       f"{', '.join(missing)} — every implement/demonstrate "
                       f"equation needs its what/why/intuition"})

    paper_tokens = _tokens(_norm(paper_text)) if paper_text else None
    for eid, entry in explanations.items():
        if not isinstance(entry, dict):
            findings.append({
                "severity": "error", "check": "schema", "element_id": eid,
                "message": f"{eid}: explanation entry is not an object"})
            continue
        for field in REQUIRED_FIELDS:
            text = str(entry.get(field, "")).strip()
            if len(text) < MIN_FIELD_CHARS:
                findings.append({
                    "severity": "error", "check": "empty_field",
                    "element_id": eid,
                    "message": f"{eid}.{field}: missing or too short "
                               f"(<{MIN_FIELD_CHARS} chars) — a slot-filler "
                               f"is not an explanation"})
        if paper_tokens:
            prose = " ".join(str(entry.get(f, "")) for f in REQUIRED_FIELDS)
            for quote in _QUOTED_RE.findall(prose):
                if len(quote.split()) >= 5 and \
                        not _quote_in_paper(quote, paper_tokens):
                    finding = {
                        "severity": "error", "check": "fabricated_quote",
                        "element_id": eid,
                        "message": f"{eid}: quoted text not found in the "
                                   f"paper: \"{quote[:80]}…\" — paraphrase "
                                   f"without quote marks, or quote "
                                   f"verbatim"}
                    # Item 23 part 3: carry the paper's best-matching
                    # passage so the retry can quote verbatim without
                    # re-reading the paper. Advisory only — the retry
                    # note's wording makes adoption conditional.
                    candidates = locate_passages(
                        quote, paper_text or "", max_passage_chars=600)
                    if candidates:
                        finding["candidate_passage"] = \
                            candidates[0]["passage"]
                    findings.append(finding)
    return findings


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--explanations", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, default=None)
    args = parser.parse_args(argv)

    pipeline = args.run_dir / ".pipeline"
    try:
        sidecar = json.loads(args.explanations.read_text(encoding="utf-8"))
        paper_map = json.loads(
            (pipeline / "paper_map.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        print(f"FAIL: {e}", file=sys.stderr)
        return 1
    paper_path = pipeline / "paper.md"
    paper_text = paper_path.read_text(encoding="utf-8") \
        if paper_path.is_file() else None

    findings = check_explanations(sidecar, paper_map, paper_text)
    errors = [f for f in findings if f["severity"] == "error"]
    if args.report:
        args.report.write_text(json.dumps({
            "valid": not errors, "findings": findings}, indent=2) + "\n",
            encoding="utf-8")
    for f in findings:
        print(f"[{f['severity']}][{f['check']}] {f['message']}")
    print(f"{'FAIL' if errors else 'OK'}: {len(errors)} error(s), "
          f"{len(findings) - len(errors)} other finding(s)")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
