"""Item 2 — reviewer-caveat disclosure routing.

The 07-06 hostile audit found the verified GBALD package resolved the paper's
central formula-versus-prose contradiction (Eq. 13) with the disclosure buried
in deferred_findings.md, a file no researcher reads. The stage reviewer had
explicitly asked "Document in assumptions.md" and that request went nowhere.

These build against the design note's eight tests: a caveat that requests
disclosure (structured field or prose bridge) is promoted to assumptions.md at
stage-5 routing, disclosure only, idempotent, never touching the label.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import run_pipeline
from delivery_label import derive_delivery_label
from run_layout import ASSUMPTIONS_MD, DEFERRED_FINDINGS_MD
from tests.helpers.state import make_state

_FIXTURE = (
    Path(__file__).parent
    / "fixtures/evidence/gbald-verified-first/stage_review_stage_2c_method.json"
)


def _events(state) -> list[dict]:
    path = state.paths.pipeline_dir / "run_events.jsonl"
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _assumptions(state) -> str:
    # Consolidated layout: assumptions.md lives under details/.
    path = state.paths.run_dir / ASSUMPTIONS_MD
    return path.read_text(encoding="utf-8") if path.exists() else ""


def _deferred(state) -> str:
    path = state.paths.run_dir / DEFERRED_FINDINGS_MD
    return path.read_text(encoding="utf-8") if path.exists() else ""


def _entry_count(body: str) -> int:
    return len(re.findall(r"(?m)^## A\d{3} ", body))


def _install_stage2c_review(state) -> tuple[list[dict], list[dict]]:
    """Drop the archived verified-roll stage_2c review into the run dir and run
    the real extraction, so tests exercise the whole chain. Returns the two
    tiers this file is about (the important tier is R2C-068's territory, see
    tests/test_important_finding_routing.py)."""
    dest = state.paths.pipeline_dir / "stage_review_stage_2c_method.json"
    dest.write_text(_FIXTURE.read_text(encoding="utf-8"), encoding="utf-8")
    _important, nice, caveats = run_pipeline._stage_review_deferred_findings(state)
    return nice, caveats


# --- Test 1: the GBALD caveat, verbatim -------------------------------------


def test_gbald_eq13_caveat_promoted_to_assumptions(tmp_path):
    state = make_state(tmp_path / "run")
    nice, caveats = _install_stage2c_review(state)
    # caveat 1 is the Eq. 13 controversy with "Document in assumptions.md".
    eq13 = next(c for c in caveats if "Eq. 13" in c["description"])

    run_pipeline._promote_reviewer_disclosures(state, nice + caveats)

    body = _assumptions(state)
    assert _entry_count(body) == 1  # exactly one entry promoted
    assert "Eq. 13" in body
    assert "farthest-first" in body
    assert "**To override.**" in body

    events = [e for e in _events(state) if e["event_type"] == "reviewer_disclosure_routed"]
    assert len(events) == 1
    assert events[0]["details"]["caveat_id"] == eq13["id"]
    assert events[0]["details"]["via"] == "text_bridge"


# --- Test 2: negative control (sibling caveat stays in deferred only) --------


def test_sibling_caveat_without_request_is_not_promoted(tmp_path):
    state = make_state(tmp_path / "run")
    _, caveats = _install_stage2c_review(state)
    tryfinally = next(c for c in caveats if "try/finally" in c["description"])

    run_pipeline._promote_reviewer_disclosures(state, [tryfinally])

    assert _entry_count(_assumptions(state)) == 0
    assert [e for e in _events(state) if e["event_type"] == "reviewer_disclosure_routed"] == []


# --- Test 3: structured path (disclosure_request field, no magic words) ------


def test_structured_disclosure_request_routes_without_bridge(tmp_path):
    state = make_state(tmp_path / "run")
    item = {
        "id": "stage_2b_architecture:caveat:1",
        "severity": "nice-to-have",
        "target_agent": "human",
        "file": "stage_review_stage_2b_architecture.json",
        # No documentation verb + surface anywhere in the prose.
        "description": "The forward pass fuses BEV features bilinearly, matching the paper.",
        "disclosure_request": {
            "surface": "assumptions",
            "note": "bilinear BEV fusion follows the paper's stated method",
        },
    }
    # Prove the bridge would NOT have fired — routing is via the structured field.
    assert run_pipeline._caveat_requests_disclosure(item["description"]) is False

    run_pipeline._promote_reviewer_disclosures(state, [item])

    body = _assumptions(state)
    assert _entry_count(body) == 1
    assert "bilinear BEV fusion follows the paper" in body  # the note became the reasoning
    events = [e for e in _events(state) if e["event_type"] == "reviewer_disclosure_routed"]
    assert events[0]["details"]["via"] == "structured"


# --- Test 4: resume idempotence ---------------------------------------------


def test_promotion_is_idempotent_across_reruns(tmp_path):
    state = make_state(tmp_path / "run")
    _, caveats = _install_stage2c_review(state)
    eq13 = next(c for c in caveats if "Eq. 13" in c["description"])

    run_pipeline._promote_reviewer_disclosures(state, [dict(eq13)])
    run_pipeline._promote_reviewer_disclosures(state, [dict(eq13)])  # re-run

    assert _entry_count(_assumptions(state)) == 1  # not duplicated
    marker = json.loads(
        (state.paths.pipeline_dir / "reviewer_disclosures.json").read_text()
    )
    assert list(marker) == [eq13["id"]]
    routed = [e for e in _events(state) if e["event_type"] == "reviewer_disclosure_routed"]
    assert len(routed) == 1  # event fired once


# --- Test 5: label invariance -----------------------------------------------


def test_promotion_never_moves_the_delivery_label(tmp_path):
    state = make_state(tmp_path / "run")
    _, caveats = _install_stage2c_review(state)
    probe = {"verdicts": [{"tier": "contribution", "status": "pass", "element_ids": ["e1"]}]}
    review = {"findings": []}

    before = derive_delivery_label(probe, review)
    run_pipeline._promote_reviewer_disclosures(state, caveats)
    after = derive_delivery_label(probe, review)

    assert json.dumps(before, sort_keys=True) == json.dumps(after, sort_keys=True)


# --- Test 6: cross-reference ------------------------------------------------


def test_deferred_entry_cross_references_the_assumption(tmp_path):
    state = make_state(tmp_path / "run")
    _, caveats = _install_stage2c_review(state)
    eq13 = next(c for c in caveats if "Eq. 13" in c["description"])

    run_pipeline._promote_reviewer_disclosures(state, caveats)
    # Stage 5 logs the caveats AFTER promotion annotates them in place.
    run_pipeline._ask_user_log_only(state, caveats, "Completed stage-review caveats")

    deferred = _deferred(state)
    aid = eq13["disclosure_promoted_to"]
    assert aid is not None
    assert f"Promoted to assumptions.md as {aid}." in deferred
    # The A-id it names exists in assumptions.md.
    assert f"## {aid} — " in _assumptions(state)
    # Exactly one cross-reference line.
    assert deferred.count("Promoted to assumptions.md as") == 1


# --- Test 7: cold reader (real renderers) -----------------------------------


def test_cold_reader_five_minute_path(tmp_path):
    from render_run_report import _count_assumptions, render_run_report

    state = make_state(tmp_path / "run")
    _, caveats = _install_stage2c_review(state)
    run_pipeline._promote_reviewer_disclosures(state, caveats)
    run_pipeline.finalize_assumptions_md(state)

    body = _assumptions(state)
    # A confused researcher greps the equation identifier and finds it.
    assert "Eq. 13" in body
    # The Issues-table counter (the real function REPORT.md uses) sees it.
    total, _needs = _count_assumptions(body)
    assert total == 1
    # And it surfaces in the rendered REPORT.md through the real renderer.
    report = render_run_report(state.paths.run_dir)
    assert "assumption entr" in report


# --- Unattended-safe: a write failure never kills the run -------------------


def test_promotion_write_failure_logs_and_continues(tmp_path, monkeypatch):
    state = make_state(tmp_path / "run")
    _, caveats = _install_stage2c_review(state)
    eq13 = next(dict(c) for c in caveats if "Eq. 13" in c["description"])

    def boom(*a, **k):
        raise OSError("disk full")

    monkeypatch.setattr(run_pipeline, "_append_assumption", boom)
    # Must not raise — nice-to-have promotion never kills a run.
    run_pipeline._promote_reviewer_disclosures(state, [eq13])

    # Left unmarked so a resume retries it, and nothing was recorded.
    assert eq13.get("disclosure_promoted_to") is None
    assert not (state.paths.pipeline_dir / "reviewer_disclosures.json").exists()


# --- Test 8: bridge precision -----------------------------------------------


def test_bridge_requires_request_verb_plus_surface(tmp_path):
    fires = run_pipeline._caveat_requests_disclosure
    # Fires: request verb + surface.
    assert fires("Document in assumptions.md.") is True
    assert fires("This should be documented in assumptions.md before release.") is True
    assert fires("needs an assumptions.md entry") is True
    assert fires("This should be flagged in assumptions.md") is True
    assert fires("the divergence should be recorded in assumptions.md") is True
    assert fires("the right fix is to document this in assumptions.md") is True
    assert fires("This must be clearly documented in assumptions.md") is True
    # Does not fire: a surface mentioned without a documentation request.
    assert fires("this is already visible in REPORT.md") is False
    assert fires("see REPORT.md for the full table") is False
    assert fires("The try/finally pattern would be preferable for robustness.") is False
    # Does not fire: a STATEMENT that something is documented, or the verb
    # used as a noun — the request shape is required, not verb presence.
    assert fires("This divergence is already documented in METHOD.md") is False
    assert fires("the dropout flag in REPORT.md controls rendering") is False
    assert fires("for the record, REPORT.md shows the full table") is False
    assert fires("the assumptions.md file already carries an override line") is False
