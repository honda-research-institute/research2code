"""Render-equivalent quote re-anchoring (queue item 11, 2026-07-21).

Every positive case here is a REAL delta shape from the three overnight
failures (SRL decomposer, ACC + Rethinking insight rows): producers emit
render-equivalent but byte-different quotes and cannot transcribe the
difference even when handed the exact bytes. The re-anchor treats the
quote as a locator and takes the source's own bytes, uniquely or not at
all.
"""

from __future__ import annotations

import json

from tests.helpers.state import make_state

from quote_reanchor import find_unique_span, reanchor_quote


# --- Class A: combining vs spacing diacritics (9 of SRL's 11) ----------

def test_combining_vs_spacing_tilde_reanchors():
    # paper.md holds U+02DC SMALL TILDE; the decomposer emitted U+0303
    # COMBINING TILDE — identical when rendered, different bytes.
    source = "Prose before.\n\nargmin\nπ(s, ˜so)\nE [tg|s0, ˜so\n0, π]\n(1)\n\nProse after."
    quote = "argmin\nπ(s, ̃so)\nE [tg|s0, ̃so\n0, π]\n(1)"
    anchored = reanchor_quote(quote, source)
    assert anchored is not None
    assert "˜so" in anchored
    assert "̃" not in anchored
    assert anchored in source


def test_combining_vs_spacing_macron_reanchors():
    source = "Intro.\n\n¯te = 1\nn\nPn\ni=1[t(i)\ne ]\n\nMore."
    quote = "̄te = 1\nn\nPn\ni=1[t(i)\ne ]"
    anchored = reanchor_quote(quote, source)
    assert anchored is not None
    assert anchored.startswith("¯te")


# --- Class B: math-delimiter substitution (15 of ACC's 20) -------------

def test_mathjax_delimiters_locate_dollar_math():
    source = ("We note that if  $\\gamma$  becomes relatively small,  "
              "$\\mathcal{S}_{\\mathrm{cbf},k}$  will be smaller.")
    quote = ("if \\(\\gamma\\)  becomes relatively small, "
             "\\(\\mathcal{S}_{\\mathrm{cbf},k}\\)  will be smaller.")
    anchored = reanchor_quote(quote, source)
    assert anchored is not None
    assert "$\\gamma$" in anchored
    assert "\\(" not in anchored


def test_display_math_flattened_to_inline_reanchors():
    source = ("the CBF constraints in (10f) becomes,\n\n"
              "$$h(\\mathbf{x}_{t+k+1|t}) \\ge 0.$$\n\nIf that holds.")
    quote = ("the CBF constraints in (10f) becomes, "
             "\\(h(\\mathbf{x}_{t+k+1|t}) \\ge 0\\).")
    anchored = reanchor_quote(quote, source)
    assert anchored is not None
    assert "$$" in anchored  # the paper's own display form survives


# --- Classes C/D: markup loss, dashes, mid-token whitespace ------------

def test_stripped_markdown_emphasis_reanchors_with_markup():
    source = ("limited by a global receptive **field**, and **fixed "
              "grouping pattern**. We demonstrate this.")
    quote = "global receptive field, and fixed grouping pattern."
    anchored = reanchor_quote(quote, source)
    assert anchored is not None
    assert "**field**" in anchored


def test_ascii_double_hyphen_matches_em_dash():
    source = ("fully random grouping—where the fixed grouping pattern "
              "is disrupted—the performance drops.")
    quote = ("fully random grouping--where the fixed grouping pattern "
             "is disrupted--the performance")
    anchored = reanchor_quote(quote, source)
    assert anchored is not None
    assert "—" in anchored


def test_typographic_apostrophe_matches_ascii_apostrophe():
    # SRL eq-state-vectors, 2026-07-27: the paper writes "agent’s" with
    # U+2019 and the producer wrote an ASCII apostrophe. The two render
    # identically, so three fix retries could not close it and stage 1
    # halted on this single character.
    source = ("so = [px, py, vx, vy, r] ∈R5; let the unobservable states\n"
              "be the agent’s intended goal position, preferred speed, and\n"
              "orientation, sh = [pgx, pgy, vpref, ψ] ∈R4.")
    quote = ("so = [px, py, vx, vy, r] ∈R5; let the unobservable states "
             "be the agent's intended goal position, preferred speed, and "
             "orientation, sh = [pgx, pgy, vpref, ψ] ∈R4.")
    anchored = reanchor_quote(quote, source)
    assert anchored is not None
    assert "agent’s" in anchored, "must return the paper's own bytes"


def test_typographic_double_quotes_match_ascii_double_quotes():
    source = ('the reward term is the “social norm” penalty applied at '
              'every step of the trajectory.')
    quote = ('the reward term is the "social norm" penalty applied at '
             'every step')
    anchored = reanchor_quote(quote, source)
    assert anchored is not None
    assert "“social norm”" in anchored


def test_quote_fold_does_not_make_a_paraphrase_match():
    # The fold is leniency on characters that render identically, never on
    # meaning: different words still fail.
    source = "the agent’s preferred speed is held fixed across episodes."
    quote = "the agent's chosen velocity is held fixed across episodes."
    assert reanchor_quote(quote, source) is None


def test_midtoken_whitespace_split_reanchors():
    # SRL eq-collision-constraint carried '≥ r' where the paper has '≥r'
    # (a space between non-space chars survives the floor's
    # whitespace normalization and fails).
    source = ("Collision.\n\n||pt −˜pt||2 ≥r + ˜r\n∀t\n(2)\n\nGoal next.")
    quote = "||pt −̃pt||2 ≥ r + ̃r\n∀t\n(2)"
    anchored = reanchor_quote(quote, source)
    assert anchored is not None
    assert "≥r" in anchored


def test_below_length_floor_stays_on_the_fix_loop():
    # SRL eq-goal-constraint folds to 9 chars — too little context to
    # trust a unique match; it keeps the honest fix-loop path (which now
    # carries fair-share candidate passages).
    source = "Goal reaching.\n\nptg = pg\n(3)\n\nDiscussion follows here."
    assert reanchor_quote("pt\ng = pg\n(3)", source) is None


# --- Honest failure paths ----------------------------------------------

def test_ambiguous_match_returns_none():
    block = "the update rule x = x + 1 applies"
    source = block + "\n\nlater again: " + block
    assert find_unique_span("x = x + 1 applies", source) is None


def test_paraphrase_returns_none():
    source = "The tradeoff between safety and feasibility is captured."
    assert reanchor_quote(
        "safety and feasibility in terms of the choice of gamma",
        source) is None


def test_short_quotes_are_refused():
    assert find_unique_span("x = 1", "prose x = 1 prose") is None


def test_region_bound_search():
    block = "the shared phrase appears here verbatim today"
    source = block + " ... and " + block
    # Ambiguous globally, unique within the region.
    assert find_unique_span(block, source) is None
    span = find_unique_span(block, source, region=(0, len(block)))
    assert span is not None
    assert source[span[0]:span[1]] == block


# --- Driver pass: repairs land, ambiguity stays on the fix loop --------

def test_driver_reanchor_pass_repairs_unique_and_leaves_ambiguous(tmp_path):
    import run_pipeline
    from validate_paper_map import check_equation_quotes

    state = make_state(tmp_path / "run")
    paper = ("Setup.\n\nargmin\nπ(s, ˜so)\nE [tg|s0]\n(1)\n\n"
             "dup dup dup equation body\n\nmiddle\n\ndup dup dup equation body\n")
    (state.paths.pipeline_dir / "paper.md").write_text(paper, encoding="utf-8")
    elements = [
        {"id": "eq-fixable", "type": "equation", "name": "Objective",
         "section": "S1", "code_role": "implement",
         "source_text": "argmin\nπ(s, ̃so)\nE [tg|s0]\n(1)"},
        {"id": "eq-ambiguous", "type": "equation", "name": "Dup",
         "section": "S2", "code_role": "implement",
         "source_text": "dup  dup  dup equation body extra"},
    ]
    state.paths.paper_map.write_text(
        json.dumps({"title": "T", "elements": elements}), encoding="utf-8")

    repaired = run_pipeline._reanchor_failing_equation_quotes(state)

    assert repaired == 1
    fixed = json.loads(state.paths.paper_map.read_text(encoding="utf-8"))
    by_id = {e["id"]: e for e in fixed["elements"]}
    assert "˜so" in by_id["eq-fixable"]["source_text"]
    # The repaired quote now passes the floor; the ambiguous one still fails.
    remaining = check_equation_quotes(fixed, paper)
    assert any("eq-ambiguous" in line for line in remaining)
    assert not any("eq-fixable" in line for line in remaining)
    events = (state.paths.pipeline_dir / "run_events.jsonl").read_text(
        encoding="utf-8")
    assert "equation_quotes_reanchored" in events
