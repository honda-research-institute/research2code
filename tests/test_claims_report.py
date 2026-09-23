"""REPORT.md verdict-section renderer (CT-3 §5): honest tally (contradicted
shown even at zero), most-actionable-first ordering, never-accusatory wording,
no bare probe codes, graceful empty/null cases."""

from __future__ import annotations

import json

import pytest

from probes.claims import build_ledger  # noqa: E402
import render_claims_report as rcr  # noqa: E402
from render_claims_report import (  # noqa: E402
    render_claims_report,
    write_claims_report,
)

# Probe-id codes that must never appear bare in the researcher-facing report.
_BARE_CODES = ("CT-1", "CT-5", "US-4", "UB-6", "UB-7", "UB-9", "AL-1")


def _al(verdicts, paper_map=None, findings=None):
    return build_ledger(paper_map or {"elements": []}, verdicts,
                        "active_learning/bayesian", findings=findings)


# ---------------------------------------------------------------------------
# Drift guard + tally honesty
# ---------------------------------------------------------------------------


def test_renderer_statuses_are_ledger_tally_keys():
    # The renderer keys off status strings the ledger writes; pin them together
    # so a rename in probes.claims can't silently break the report.
    tally = build_ledger({"elements": []}, [], "")["tally"]
    for status in (rcr.VERIFIED_AT_SCALE, rcr.UNTESTED_AT_THIS_SCALE,
                   rcr.SUSPECT_OUR_IMPLEMENTATION, rcr.CONTRADICTED,
                   rcr.DIRECTIONALLY_CHECKED):
        assert status in tally, status


def test_tally_line_shows_contradicted_even_at_zero():
    line = rcr._tally_line({"verified_at_scale": 2,
                            "suspect_our_implementation": 1,
                            "untested_at_this_scale": 14})
    assert "0 contradicted" in line                       # absence is stated
    assert "2 verified at smoke scale (behavioral properties)" in line
    assert "1 needs a fix on our side" in line
    assert "14 untested at this scale" in line


def test_tally_singular_grammar():
    line = rcr._tally_line({"verified_at_scale": 1,
                            "suspect_our_implementation": 1,
                            "untested_at_this_scale": 1})
    assert "1 verified at smoke scale (behavioral property)" in line
    assert "1 needs a fix on our side" in line


# ---------------------------------------------------------------------------
# Verified / untested rendering (BADGE-shape)
# ---------------------------------------------------------------------------


def test_verified_and_untested_render():
    pm = {"elements": [{"id": "exp-1", "type": "experiment",
                        "name": "main comparison",
                        "source_text": "Method reaches 0.99 accuracy on MNIST."}]}
    ledger = _al([{"probe_id": "CT-1", "verdict": "pass", "message": "differs"}],
                 paper_map=pm)
    md = render_claims_report(ledger)
    assert "## Verification" in md
    assert "Verified at smoke scale" in md
    assert "Untested at this scale" in md
    assert "To verify it:" in md                           # the recipe shows
    assert "0 contradicted" in md
    # The verified line carries the design's caveat, not a headline-number claim.
    assert "does not confirm the paper" in md


def test_no_bare_probe_codes_in_report():
    pm = {"elements": [{"id": "e", "type": "experiment",
                        "source_text": "outperforms BALD, 0.99 on MNIST"}]}
    ledger = _al([
        {"probe_id": "CT-1", "verdict": "pass", "message": "differs"},
        {"probe_id": "US-4", "verdict": "fail", "message": "R_0 unrescaled"},
    ], paper_map=pm)
    md = render_claims_report(ledger)
    for code in _BARE_CODES:
        assert code not in md, f"bare code {code} leaked into REPORT.md"


def test_no_bare_finding_codes_in_report():
    md = render_claims_report({"claims": [{
        "kind": "behavioral",
        "status": "suspect_our_implementation",
        "name": "linked finding F001",
        "reasoning": "F001 and US-4 explain this shortfall.",
    }]})
    assert "F001" not in md
    assert "US-4" not in md
    assert "the linked finding" in md
    assert "the linked check" in md


def test_stage_scoped_probe_codes_are_scrubbed():
    """AL-S1-style ids (a stage segment between family and number) slipped
    past this renderer's scrub pattern after only the run-report copy was
    widened (found in the 2026-07-22 design review). Every id shape the
    probe modules emit must scrub here too, never render bare."""
    md = render_claims_report({"claims": [{
        "kind": "behavioral",
        "status": "suspect_our_implementation",
        "name": "selector picks",
        "reasoning": "Constant scores adjudicated by AL-S1-1; AL-S1-2 concurs.",
    }]})
    assert "AL-S1-1" not in md
    assert "AL-S1-2" not in md
    assert "the linked check" in md


# ---------------------------------------------------------------------------
# Suspect rendering (GBALD-shape) — never accusatory toward the paper
# ---------------------------------------------------------------------------


def test_suspect_rows_render_as_our_implementation_never_accusing_paper():
    ledger = _al([
        {"probe_id": "CT-1", "verdict": "pass", "message": "differs"},
        {"probe_id": "US-4", "verdict": "fail", "message": "R_0 unrescaled"},
    ])
    md = render_claims_report(ledger)
    assert "Needs a fix on our side" in md
    assert "most likely our implementation" in md
    # The only reassurance phrasing mentioning the paper is the do-not-blame one.
    assert "do not read it as the paper being wrong" in md
    assert "Contradicted" not in md                        # not reachable yet


def test_contradicted_row_renders_with_human_review_wording_and_tally():
    # A contradicted row (forced — structurally unreachable in the live pipeline)
    # renders with its careful-human-review gloss, sorts near the top (actionable),
    # and is counted in the tally. The §5 wording flows through from `reasoning`.
    rows = [
        {"kind": "behavioral", "status": "contradicted",
         "name": "an invariant the paper relies on",
         "reasoning": "At the paper's own scale ... this warrants a careful human "
                      "look before drawing conclusions about the paper.",
         "what_would_verify": ""},
        {"kind": "verified", "status": "verified_at_scale",
         "name": "a verified property", "reasoning": "ok"},
    ]
    md = render_claims_report({"tally": {}, "claims": rows})
    assert "1 contradicted" in md
    assert "Contradicted (needs careful human review)" in md
    assert "careful human look" in md
    # Actionable-first: the paper alarm renders above the (done) verified row.
    assert md.index("an invariant") < md.index("a verified property")


def test_most_actionable_sorted_first():
    # suspect (fix-our-side) must render above verified (done), even when the
    # verified row appears first in the claims list.
    rows = [
        {"kind": "behavioral", "status": "verified_at_scale",
         "name": "verified thing", "reasoning": "ok", "what_would_verify": ""},
        {"kind": "behavioral", "status": "suspect_our_implementation",
         "name": "suspect thing", "reasoning": "our bug", "what_would_verify": ""},
    ]
    md = render_claims_report({"tally": {}, "claims": rows})
    assert md.index("suspect thing") < md.index("verified thing")


# ---------------------------------------------------------------------------
# Robustness: pipe escaping, empty ledger, write/no-op
# ---------------------------------------------------------------------------


def test_pipe_in_claim_does_not_break_table():
    rows = [{"kind": "lifted", "status": "untested_at_this_scale",
             "name": "acc | f1 tradeoff", "claim_text": "a | b",
             "reasoning": "x | y", "what_would_verify": "z"}]
    md = render_claims_report({"tally": {}, "claims": rows})
    assert "acc \\| f1 tradeoff" in md
    # Header + separator + exactly one data row (the pipe did not split it).
    data_rows = [ln for ln in md.splitlines()
                 if ln.startswith("|") and "---" not in ln
                 and "Claim" not in ln]
    assert len(data_rows) == 1


def test_empty_ledger_is_graceful():
    md = render_claims_report({"tally": {}, "claims": []})
    assert "nothing to verify" in md
    assert "0 contradicted" in md


def test_write_claims_report_roundtrip(tmp_path):
    run = tmp_path / "run"
    (run / ".pipeline").mkdir(parents=True)
    pm = {"elements": [{"id": "exp-1", "type": "experiment",
                        "source_text": "improves accuracy by 4%."}]}
    ledger = _al([{"probe_id": "CT-1", "verdict": "pass", "message": "differs"}],
                 paper_map=pm)
    (run / ".pipeline" / "claims_ledger.json").write_text(json.dumps(ledger))
    assert write_claims_report(run) is True
    report = (run / "REPORT.md").read_text()
    assert "## Verification" in report
    assert "Verified at smoke scale" in report


def test_write_claims_report_noop_without_ledger(tmp_path):
    run = tmp_path / "run"
    (run / ".pipeline").mkdir(parents=True)
    assert write_claims_report(run) is False
    assert not (run / "REPORT.md").exists()


# ---------------------------------------------------------------------------
# Hardening from the step-4 red-team: the honest tally is recomputed from the
# rows (never trusted from a corrupted stored field), and malformed ledgers
# render/no-op gracefully instead of crashing finalization.
# ---------------------------------------------------------------------------


def test_tally_recomputed_from_rows_not_trusted_from_input():
    # A corrupted stored tally claiming 5 verified must NOT overstate: the
    # headline is recomputed from the rows actually rendered (here, 0 verified).
    ledger = {"tally": {"verified_at_scale": 5, "untested_at_this_scale": 1},
              "claims": [{"kind": "lifted", "status": "untested_at_this_scale",
                          "name": "X", "reasoning": "u"}]}
    md = render_claims_report(ledger)
    assert "0 verified at smoke scale" in md
    assert "5 verified" not in md
    assert "1 untested at this scale" in md


def test_non_dict_ledger_renders_empty_not_crash():
    for bad in ([1, 2, 3], None, "nope", 42):
        md = render_claims_report(bad if isinstance(bad, dict) else {"claims": bad})
        assert "nothing to verify" in md          # graceful empty case
        assert "0 contradicted" in md


def test_malformed_status_falls_back_to_question_mark():
    md = render_claims_report({"claims": [{"name": "a"}, {"status": None,
                                                          "name": "b"}]})
    # No literal "None" verdict, no crash; unknown/None glosses to "?".
    assert "| None |" not in md


def test_write_claims_report_noop_on_non_dict_json(tmp_path):
    run = tmp_path / "run"
    (run / ".pipeline").mkdir(parents=True)
    (run / ".pipeline" / "claims_ledger.json").write_text("[1, 2, 3]")
    assert write_claims_report(run) is False         # no crash, no REPORT.md
    assert not (run / "REPORT.md").exists()
