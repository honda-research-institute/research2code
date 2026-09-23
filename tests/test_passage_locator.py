"""Fuzzy passage locator + carriers (item 23 part 3, design approved
2026-07-10: k=1, explainer half in the same block, 0.5 score floor).

When a verbatim-quote floor rejects a quote, the retry finding carries
the paper's own best-matching passage so the fixer copies instead of
re-reading the paper (the output-cap burn shape from the ACC 2026-07-08
run). The fixture is that run's harvested paper map + paper.md verbatim,
still failing the equation floor on 11 elements — real paper size, real
Marker artifacts, by design.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import run_pipeline
from passage_locator import locate_passages
from validate_method_explanations import check_explanations
from validate_paper_map import _normalize_ws, check_equation_quotes

from tests.helpers.state import make_state

FIXTURE = (Path(__file__).parent / "fixtures" / "evidence"
           / "overnight-0707" / "item-23-passage-locator-acc")


def _acc():
    paper_map = json.loads(
        (FIXTURE / "paper_map.json").read_text(encoding="utf-8"))
    paper_text = (FIXTURE / "paper.md").read_text(encoding="utf-8")
    return paper_map, paper_text


def _failing_elements(paper_map, paper_text):
    errors = check_equation_quotes(paper_map, paper_text)
    ids = {m.group(1) for e in errors
           if (m := re.match(r"\s*-\s*(\S+):", e))}
    return [e for e in paper_map["elements"] if e.get("id") in ids]


# --- the locator -------------------------------------------------------------


def test_every_acc_rejected_quote_locates_a_floor_passing_passage():
    paper_map, paper_text = _acc()
    failing = _failing_elements(paper_map, paper_text)
    assert len(failing) == 11  # the harvested halt shape, pinned
    haystack = _normalize_ws(paper_text)
    for element in failing:
        candidates = locate_passages(
            element["source_text"], paper_text, element_type="equation")
        assert candidates, f"no candidate for {element['id']}"
        top = candidates[0]
        # Raw paper bytes: the candidate passes the floor by construction.
        assert _normalize_ws(top["passage"]) in haystack, element["id"]
        # And carries the display math the fixer needs to copy.
        assert "$$" in top["passage"], element["id"]
        assert top["score"] >= 0.5


def test_stitched_quote_gets_display_math_via_the_adjacency_guard():
    # eq-barrier-quartic: the quote stitches the equation to prose that
    # sits behind a figure caption in the Marker output, so the matched
    # block alone has no $$ — the guard pulls in the adjacent math block.
    paper_map, paper_text = _acc()
    element = next(e for e in paper_map["elements"]
                   if e["id"] == "eq-barrier-quartic")
    without_guard = locate_passages(element["source_text"], paper_text)
    with_guard = locate_passages(
        element["source_text"], paper_text, element_type="equation")
    assert "$$" not in without_guard[0]["passage"]
    assert "$$" in with_guard[0]["passage"]
    assert "h_t^i" in with_guard[0]["passage"]  # the right equation


def test_json_corrupted_quote_still_locates_its_passage():
    # The eaten-escape shape: a backslash lost its JSON double-escape, so
    # part of a LaTeX command became a control character + fragment. The
    # token normalization must still find the passage.
    paper_map, paper_text = _acc()
    element = next(e for e in paper_map["elements"]
                   if e["id"] == "eq-racing-cost")
    corrupted = element["source_text"].replace("\\beta", "\beta")
    candidates = locate_passages(corrupted, paper_text,
                                 element_type="equation")
    assert candidates
    assert _normalize_ws(candidates[0]["passage"]) in _normalize_ws(paper_text)


def test_garbage_and_empty_quotes_return_no_candidate():
    _, paper_text = _acc()
    assert locate_passages("zzzqqq xxxwww yyyvvv uuuttt", paper_text) == []
    assert locate_passages("", paper_text) == []
    assert locate_passages("real quote", "") == []


# --- the stage-1 finding enricher -------------------------------------------


def _acc_state(run_dir):
    state = make_state(run_dir)
    paper_map, paper_text = _acc()
    state.paths.paper_map.write_text(json.dumps(paper_map),
                                     encoding="utf-8")
    state.paths.paper_md.write_text(paper_text, encoding="utf-8")
    return state


def test_enricher_appends_candidates_block_to_the_finding(run_dir):
    state = _acc_state(run_dir)
    finding = {"id": "VAL001", "severity": "critical",
               "description": "validator output here"}
    enriched = run_pipeline._paper_map_quote_enricher(state)(finding)
    desc = enriched["description"]
    assert desc.startswith("validator output here")
    assert "Candidate paper passages" in desc
    assert "eq-barrier-quartic" in desc
    assert "```" in desc  # passages ride fenced
    assert "ONLY if it is the one you meant" in desc  # advisory wording
    assert "backslash must be doubled" in desc
    # The original finding dict is not mutated.
    assert finding["description"] == "validator output here"


def test_enricher_block_cap_names_omitted_elements(run_dir, monkeypatch):
    state = _acc_state(run_dir)
    monkeypatch.setattr(run_pipeline, "_QUOTE_CANDIDATES_BLOCK_CAP", 2500)
    desc = run_pipeline._paper_map_quote_enricher(state)(
        {"description": "x"})["description"]
    assert "block size cap" in desc
    assert "locate those in the paper yourself" in desc


def test_enricher_leaves_finding_alone_when_nothing_fails(run_dir):
    state = make_state(run_dir)  # placeholder paper, no failing quotes
    state.paths.paper_map.write_text(
        json.dumps({"title": "T", "elements": []}), encoding="utf-8")
    finding = {"description": "unchanged"}
    assert run_pipeline._paper_map_quote_enricher(state)(finding) is finding


def test_validator_retry_dispatches_the_enriched_finding(
        fake_dispatch, run_dir):
    """The enricher hook fires on the mechanical routing path, and an
    enricher failure never kills the loop."""
    state = make_state(run_dir)
    outcomes = [(False, "boom (1 element(s)):\n  - eq-x: bad"), (True, "")]

    captured: list[str] = []

    def fix_dispatch(st, findings):
        captured.append(findings[0]["description"])

    result = run_pipeline._validator_retry(
        state, stage_id="stage_1",
        validator_fn=lambda _s: outcomes.pop(0),
        fix_dispatch_fn=fix_dispatch,
        validator_label="validate_paper_map.py",
        cap=2, use_judge=False,
        finding_enricher=lambda f: {**f, "description":
                                    f["description"] + "\nENRICHED-MARKER"},
    )
    assert result is None
    assert captured and captured[0].endswith("ENRICHED-MARKER")

    outcomes2 = [(False, "boom (1 element(s)):\n  - eq-x: bad"), (True, "")]
    captured.clear()

    def _raises(_f):
        raise RuntimeError("enricher bug")

    result = run_pipeline._validator_retry(
        state, stage_id="stage_1",
        validator_fn=lambda _s: outcomes2.pop(0),
        fix_dispatch_fn=fix_dispatch,
        validator_label="validate_paper_map.py",
        cap=2, use_judge=False,
        finding_enricher=_raises,
    )
    assert result is None  # loop survived, unenriched finding dispatched
    assert captured and "ENRICHED-MARKER" not in captured[0]


# --- the explainer half ------------------------------------------------------


def test_fabricated_quote_finding_carries_the_candidate_passage():
    paper_map, paper_text = _acc()
    entry = {
        "what": "x" * 30, "why_novel": "y" * 30,
        # A near-quote of the paper's CBF geometry with one wrong numeric
        # token (2l_9): the provenance matcher requires EVERY numeric
        # token present, so the quote is fabricated, while the locator's
        # token score stays far above its 0.5 floor.
        "intuition": 'The paper says "racing cars hold the shape of '
                     'rectangle with a length as 2l_9 and width as 2l_2" '
                     'in the setup.',
    }
    sidecar = {"schema_version": "1.0.0",
               "explanations": {"eq-barrier-quartic": entry}}
    findings = check_explanations(sidecar, paper_map, paper_text)
    fabricated = [f for f in findings if f["check"] == "fabricated_quote"]
    assert len(fabricated) == 1
    passage = fabricated[0].get("candidate_passage")
    assert passage
    assert _normalize_ws(passage) in _normalize_ws(paper_text)
    assert len(passage) <= 600


def test_fabricated_quote_without_confident_match_has_no_passage():
    paper_map, paper_text = _acc()
    entry = {
        "what": "x" * 30, "why_novel": "y" * 30,
        "intuition": 'They claim "gerbils frolic beneath the moonlit '
                     'trampoline while accountants weep openly" in it.',
    }
    sidecar = {"schema_version": "1.0.0",
               "explanations": {"eq-barrier-quartic": entry}}
    findings = check_explanations(sidecar, paper_map, paper_text)
    fabricated = [f for f in findings if f["check"] == "fabricated_quote"]
    assert len(fabricated) == 1
    assert "candidate_passage" not in fabricated[0]
