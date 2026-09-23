"""Render the claims ledger (`<RUN_DIR>/.pipeline/claims_ledger.json`) as
REPORT.md's verdict section (claims-ledger design §5).

The ledger leads the report's verdict section, and this renderer is its
trust-critical, researcher-facing content:

- an honest tally, with the contradicted count shown even when it is zero so
  its absence is a stated result;
- a table sorted most-actionable first (fix-our-side, then the reserved paper
  alarm, then the untested gap, then verified);
- per-status wording that is never accusatory toward the paper, with no bare
  probe codes and plain conversational sentences.

Pure function from the ledger dict to markdown. The full REPORT.md assembler
imports `render_claims_report`; the `write_claims_report` helper and CLI remain
available for CT-3-only inspection of a committed ledger.

Stdlib-only (the `delivery_label.py` convention) so it imports cleanly anywhere;
the status strings mirror `probes.claims` and `tests/test_claims_report.py`
guards against drift.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

VERIFIED_AT_SCALE = "verified_at_scale"
UNTESTED_AT_THIS_SCALE = "untested_at_this_scale"
SUSPECT_OUR_IMPLEMENTATION = "suspect_our_implementation"
CONTRADICTED = "contradicted"
DIRECTIONALLY_CHECKED = "directionally_checked"

# Most-actionable first: the researcher fixes our-side issues, weighs a full run
# for the untested headline claims, then notes what is verified.
_STATUS_ORDER = {
    SUSPECT_OUR_IMPLEMENTATION: 0,
    CONTRADICTED: 1,
    UNTESTED_AT_THIS_SCALE: 2,
    DIRECTIONALLY_CHECKED: 3,
    VERIFIED_AT_SCALE: 4,
}

# Plain-language gloss for every status — no bare codes in the report.
_VERDICT_GLOSS = {
    VERIFIED_AT_SCALE: "Verified at smoke scale",
    SUSPECT_OUR_IMPLEMENTATION: "Needs a fix on our side",
    UNTESTED_AT_THIS_SCALE: "Untested at this scale",
    CONTRADICTED: "Contradicted (needs careful human review)",
    DIRECTIONALLY_CHECKED: "Directionally checked (advisory)",
}

# The one suspect cause whose own note REFUSES to attribute the problem: the
# contribution did not change behavior, and we cannot tell our wiring from the
# method being inert as published (claims-ledger locked decision 10.6). Its
# status is still suspect_our_implementation, because the row must keep
# demoting, but rendering it as "Needs a fix on our side" put a label on the
# table that the note beside it contradicted in the same row (R2C-047). Mirrors
# `probes.claims.UNATTRIBUTED_CAUSE`; this module stays stdlib-only and
# import-free, and a test pins the two together.
UNATTRIBUTED_CAUSE = "method_appears_inert_as_published"
_UNATTRIBUTED_GLOSS = "Unresolved (needs your judgment)"

# Probe-id scrub grammar — the single definition; render_run_report and the
# driver's notice scrubber import it rather than keeping copies. It lives
# here because this module is the stdlib-only dependency leaf all three
# researcher-facing scrub paths already reach. The pattern is structural
# over the shapes the probe naming convention can produce —
# FAMILY[-S<stage>]-<number>[letter], with stage tokens following pipeline
# stage naming (S1, S2a) — because enumerating today's literal shapes is
# how the stage-scoped AL-S1 ids leaked bare (three drifting copies, only
# one widened; found in the 2026-07-22 design review).
CODE_RE = re.compile(
    r"\b(?:US|UB|AL|MP|KD|TSF|HG|CT|SC)(?:-S\d+[a-z]?)?-\d+[a-z]?\b"
)
FINDING_RE = re.compile(r"\bF\d{3}\b")

_INTRO = (
    "This is the claims ledger. The honest default is untested at smoke scale. "
    "A claim is marked verified only when a scale-free behavioral check passes "
    "and also fails for a baseline with the contribution removed, so the check "
    "is meaningful. Where our run fell short, the ledger says so and attributes "
    "it to our own implementation, never to the paper."
)


def _tally_line(tally: dict, unattributed: int = 0) -> str:
    """The headline tally. Contradicted is always shown, even at zero.

    `unattributed` is the slice of the suspect count whose rows will not
    attribute their cause. It is carved OUT of the fix-on-our-side number and
    stated on its own, so the headline cannot claim more repairable defects
    than the table shows (R2C-047)."""
    v = int(tally.get(VERIFIED_AT_SCALE, 0) or 0)
    c = int(tally.get(CONTRADICTED, 0) or 0)
    s = int(tally.get(SUSPECT_OUR_IMPLEMENTATION, 0) or 0) - int(unattributed or 0)
    u = int(tally.get(UNTESTED_AT_THIS_SCALE, 0) or 0)
    d = int(tally.get(DIRECTIONALLY_CHECKED, 0) or 0)
    verified = f"{v} verified at smoke scale"
    if v:
        verified += " (behavioral propert" + ("y" if v == 1 else "ies") + ")"
    parts = [
        verified,
        f"{c} contradicted",
        f"{s} need{'s' if s == 1 else ''} a fix on our side",
        f"{u} untested at this scale",
    ]
    if unattributed:
        parts.append(f"{unattributed} unresolved (needs your judgment)")
    if d:
        parts.append(f"{d} directionally checked")
    return "**Claims checked:** " + ", ".join(parts) + "."


def _scrub_codes(text: object) -> str:
    text = CODE_RE.sub("the linked check", str(text or ""))
    return FINDING_RE.sub("the linked finding", text)


def _escape_cell(text: object) -> str:
    return _scrub_codes(text).replace("|", "\\|").replace("\n", " ").strip()


def _claim_label(row: dict) -> str:
    name = str(row.get("name") or "").strip()
    if name:
        return name
    text = str(row.get("claim_text") or "").strip()
    if len(text) > 80:
        return text[:77].rstrip() + "..."
    return text or "(unnamed claim)"


def _assessment(row: dict) -> str:
    reasoning = str(row.get("reasoning") or row.get("status_note") or "").strip()
    wwv = str(row.get("what_would_verify") or "").strip()
    if wwv and row.get("status") == UNTESTED_AT_THIS_SCALE:
        reasoning = f"{reasoning} To verify it: {wwv}."
    return reasoning


def _is_unattributed(row: dict) -> bool:
    """A suspect row that will not say whose defect it is."""
    return (row.get("status") == SUSPECT_OUR_IMPLEMENTATION
            and row.get("suspected_cause") == UNATTRIBUTED_CAUSE)


def _verdict_gloss(row: dict) -> str:
    """The Verdict cell for one row. Status alone is not enough: the
    unattributed suspect cause renders as unresolved, so the label and the note
    beside it tell the researcher the same thing."""
    if _is_unattributed(row):
        return _UNATTRIBUTED_GLOSS
    status = row.get("status")
    gloss = _VERDICT_GLOSS.get(status)
    if gloss:
        return gloss
    return status if isinstance(status, str) and status else "?"


def _sort_key(row: dict):
    # Behavioral rows before lifted within a status (the actionable detail
    # first), stable on the original order otherwise.
    return (_STATUS_ORDER.get(row.get("status"), 9),
            0 if row.get("kind") == "behavioral" else 1)


def _count_statuses(claims: list) -> dict:
    """Recompute the tally from the rows we actually render, so the headline
    can never overstate what the table shows (the design's honest-tally rule),
    even if a corrupted ledger carried a divergent stored tally."""
    out = {VERIFIED_AT_SCALE: 0, CONTRADICTED: 0,
           SUSPECT_OUR_IMPLEMENTATION: 0, UNTESTED_AT_THIS_SCALE: 0,
           DIRECTIONALLY_CHECKED: 0}
    for row in claims:
        if isinstance(row, dict):
            out[row.get("status")] = out.get(row.get("status"), 0) + 1
    return out


def render_claims_report(ledger: dict) -> str:
    """The verdict section as markdown. Pure; no I/O. Defensive on shape: a
    non-dict ledger or non-list claims renders as the empty case rather than
    crashing, and the tally is recomputed from the rows, never trusted from a
    (possibly corrupted) stored field."""
    raw = ledger.get("claims") if isinstance(ledger, dict) else None
    claims = [r for r in raw if isinstance(r, dict)] if isinstance(raw, list) else []
    tally = _count_statuses(claims)
    unattributed = sum(1 for row in claims if _is_unattributed(row))
    out = [
        "## Verification — what we could and could not confirm",
        "",
        _INTRO,
        "",
        _tally_line(tally, unattributed),
        "",
    ]
    if not claims:
        out.append("_No experiment claims were found in the paper map, so "
                   "there is nothing to verify here._")
        return "\n".join(out) + "\n"
    out.append("| Claim | Verdict | What this means |")
    out.append("|---|---|---|")
    for row in sorted(claims, key=_sort_key):
        out.append(
            f"| {_escape_cell(_claim_label(row))} | {_verdict_gloss(row)} | "
            f"{_escape_cell(_assessment(row))} |")
    out.append("")
    return "\n".join(out) + "\n"


def write_claims_report(run_dir) -> bool:
    """Render the run's ledger to `<RUN_DIR>/REPORT.md`. Returns False (no-op)
    when there is no ledger to render, so a missing/late ledger never breaks
    finalization."""
    run_dir = Path(run_dir)
    ledger_path = run_dir / ".pipeline" / "claims_ledger.json"
    if not ledger_path.is_file():
        return False
    try:
        ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if not isinstance(ledger, dict):
        # valid JSON but not a ledger object (a partial write, a foreign file):
        # no-op rather than crash, so finalization is never broken by it.
        return False
    (run_dir / "REPORT.md").write_text(
        render_claims_report(ledger), encoding="utf-8")
    return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args(argv)
    if not write_claims_report(args.run_dir):
        print("no .pipeline/claims_ledger.json to render", file=sys.stderr)
        return 1
    print(f"wrote {Path(args.run_dir) / 'REPORT.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
