"""Accept-the-candidate adoption for verbatim-quote failures (R2C-030).

Every positive case here is the REAL recorded shape: a producer holds
the correct passage, differs by one character it cannot perceive, and
returns the identical bytes on every retry (SRL apostrophe 2026-07-27,
bayesian-active-learning closing bracket 2026-07-28). Adoption takes the
paper's own bytes under the approved uniqueness rule; ambiguity of any
kind stays on the honest halt path.
"""

from __future__ import annotations

import json

from tests.helpers.state import make_state

import quote_adopt as qa
from quote_adopt import (ADOPTION_FLOOR, ADOPTION_MARGIN, find_adoption,
                         folded_similarity, resolve_section_region,
                         scan_candidates)

# The recorded bayesian-active-learning shape, condensed: a multi-block
# display equation where the producer wrote a closing \big) for the
# paper's closing \big].
_PAPER_EQ = (
    "$$\\mathbb{E}_{y^{*}}\\big[\\log p(\\theta\\mid\\mathcal{D}_0,"
    "\\,x^{*},\\,y^{*})\\big]$$\n\n"
    "$$= \\log p(\\theta\\mid\\mathcal{D}_0) + \\sum_{i=1}^{N}\\Big("
    "\\mathbb{E}_{y_{i}}\\big[\\log p(y_{i}\\mid x_{i},\\theta) + "
    "H\\big[y_{i}\\mid x_{i},\\mathcal{D}_0\\big]\\big]\\Big).$$"
)
_SLIPPED_QUOTE = _PAPER_EQ.replace("\\big]\\Big)", "\\big)\\Big)")
assert _SLIPPED_QUOTE != _PAPER_EQ

PAPER = (
    "# 3 Method\n\nIntro prose about the posterior model.\n\n"
    "## 3.2 Complete-data posterior\n\nWe write the expectation as\n\n"
    + _PAPER_EQ +
    "\n\nwhich concludes the derivation.\n\n"
    "## 3.3 Acquisition\n\nUnrelated prose about acquisition scoring "
    "and batch selection, long enough to be its own region.\n"
)


# --- The adoption rule itself ------------------------------------------

def test_recorded_bracket_slip_adopts_paper_bytes():
    region = resolve_section_region("Section 3.2, Eq. (2)", PAPER)
    assert region is not None
    adoption = find_adoption(_SLIPPED_QUOTE, PAPER, region=region)
    assert adoption is not None
    assert adoption["passage"] == _PAPER_EQ
    assert adoption["score"] >= ADOPTION_FLOOR
    assert adoption["region_resolved"] is True
    # The adopted bytes are a verbatim slice, so the floor passes by
    # construction.
    assert adoption["passage"] in PAPER


def test_ambiguous_twin_equations_still_halt():
    eq_a = "$$y = \\alpha x + \\beta_{1} z + \\gamma w + \\delta v$$"
    eq_b = "$$y = \\alpha x + \\beta_{2} z + \\gamma w + \\delta v$$"
    doc = ("# 1 Intro\n\nprose\n\n" + eq_a + "\n\nmore prose\n\n" + eq_b
           + "\n\ntail prose\n")
    near_quote = "$$y = \\alpha x + \\beta_{9} z + \\gamma w + \\delta v$$"
    assert find_adoption(near_quote, doc) is None
    scores = [c["score"] for c in scan_candidates(near_quote, doc)]
    # Both twins are near-matches — uniqueness is what refuses them.
    assert len([s for s in scores if s >= ADOPTION_FLOOR]) >= 2


def test_below_floor_paraphrase_still_halts():
    paraphrase = ("The expected complete-data log posterior decomposes "
                  "into the prior term plus the expected log likelihood "
                  "and the predictive entropy over the pool points.")
    assert find_adoption(paraphrase, PAPER) is None
    # Clearly below, not marginally: the floor minus margin still clears
    # it with room.
    assert folded_similarity(paraphrase, _PAPER_EQ) < (
        ADOPTION_FLOOR - 2 * ADOPTION_MARGIN)


def test_unique_candidate_outside_named_region_is_refused():
    region = resolve_section_region("Section 3.3", PAPER)
    assert region is not None
    # The only near-match lives in 3.2; an element anchored to 3.3 must
    # not adopt it — that would be a silent retarget.
    assert find_adoption(_SLIPPED_QUOTE, PAPER, region=region) is None


def test_unresolved_region_falls_back_to_global_uniqueness():
    adoption = find_adoption(_SLIPPED_QUOTE, PAPER, region=None)
    assert adoption is not None
    assert adoption["passage"] == _PAPER_EQ
    assert adoption["region_resolved"] is False


def test_floor_calibration_recorded_shapes():
    # SRL 2026-07-27: typographic vs ASCII apostrophe folds identical.
    assert folded_similarity("the agent's policy improves",
                             "the agent’s policy improves") == 1.0
    # bayesian-active-learning 2026-07-28: one bracket in a long
    # equation sits above the floor.
    assert folded_similarity(_SLIPPED_QUOTE, _PAPER_EQ) >= ADOPTION_FLOOR


def test_short_quotes_are_refused():
    assert find_adoption("$$x=1$$", "prose $$x=2$$ prose") is None


# --- Section-region resolution -----------------------------------------

def test_region_resolution_by_section_number():
    region = resolve_section_region("Section 3.2, Eq. (2)", PAPER)
    assert region is not None
    body = PAPER[region[0]:region[1]]
    assert body.startswith("## 3.2")
    assert "## 3.3" not in body


def test_region_resolution_fails_honestly():
    assert resolve_section_region("Appendix references", PAPER) is None
    assert resolve_section_region("", PAPER) is None


def test_reference_number_in_label_does_not_resolve_a_wrong_region():
    # The SRL holdout (2026-07-28 overnight batch): "Section II-A
    # (Equation 5)" must not parse the equation number as a section
    # number. Roman-only labels resolve to None, and the strictly
    # stronger global-uniqueness fallback applies.
    assert resolve_section_region("Section II-A (Equation 5)", PAPER) is None
    assert resolve_section_region("Section II-A (Eq. 5)", PAPER) is None


def test_reference_stripping_keeps_arabic_section_numbers():
    region = resolve_section_region("Section 3.2, Eq. (2)", PAPER)
    assert region is not None
    assert PAPER[region[0]:region[1]].startswith("## 3.2")


def test_multi_section_label_resolves_every_named_region():
    # The bayesian mc-dropout holdout (2026-07-29 re-roll): the label
    # names the section that introduces the equation AND the section
    # holding the quoted passage ("Section 4.3, Section 7"). Every
    # section-tokened number resolves, and containment may pass in any,
    # so an element whose passage lives in the SECOND named section
    # still adopts.
    from quote_adopt import resolve_section_regions

    regions = resolve_section_regions("Section 3.3, Section 3.2", PAPER)
    assert len(regions) == 2
    bodies = [PAPER[a:b] for a, b in regions]
    assert any(body.startswith("## 3.3") for body in bodies)
    assert any(body.startswith("## 3.2") for body in bodies)

    # The only near-match lives in 3.2, listed second — adoption passes.
    adoption = find_adoption(_SLIPPED_QUOTE, PAPER, region=regions)
    assert adoption is not None
    assert adoption["passage"] == _PAPER_EQ
    assert adoption["region_resolved"] is True


def test_bare_trailing_number_never_adds_a_region():
    # "Section 3.3, Theorem 3" must resolve ONLY 3.3: a theorem (or
    # lemma, algorithm, year) number widening the region set would
    # weaken the containment refusal, which is the retarget guard.
    from quote_adopt import resolve_section_regions

    regions = resolve_section_regions("Section 3.3, Theorem 3", PAPER)
    assert len(regions) == 1
    assert PAPER[regions[0][0]:regions[0][1]].startswith("## 3.3")
    # The near-match lives in 3.2 — still refused.
    assert find_adoption(_SLIPPED_QUOTE, PAPER, region=regions) is None


def test_trailing_similar_tokens_do_not_sink_the_score():
    # The bayesian eq-entropy-integral holdout (2026-07-29 re-roll):
    # prose right after the equation reuses its math tokens, the
    # outermost matching blocks overran the passage by 135 characters,
    # and a 0.978 bracket-slip near-match scored 0.811 — under the
    # floor. The refine step must pick the trim that maximizes the
    # score, not the outermost matches.
    doc = (
        "# 3 Method\n\n## 3.2 Complete-data posterior\n\n"
        + _PAPER_EQ +
        "\n\nwhich can be approximated by  "
        "$\\log p(\\theta\\mid\\mathcal{D}_0) + \\sum_{i=1}^{N}"
        "\\mathbb{E}_{y_{i}}\\big[\\log p(y_{i}\\mid x_{i},\\theta)\\big]$  "
        "following the derivation, where the posterior "
        "$p(\\theta\\mid\\mathcal{D}_0)$ and the entropy "
        "$H\\big[y_{i}\\mid x_{i},\\mathcal{D}_0\\big]$ reappear.\n"
    )
    top = scan_candidates(_SLIPPED_QUOTE, doc)[0]
    assert top["score"] >= ADOPTION_FLOOR
    adoption = find_adoption(_SLIPPED_QUOTE, doc)
    assert adoption is not None
    assert adoption["passage"].startswith("$$\\mathbb{E}")


# --- Driver pass: adoptions land loud, ambiguity stays on the loop -----

def test_driver_adoption_pass_records_assumption_and_sidecar(tmp_path):
    import run_pipeline
    from validate_paper_map import check_equation_quotes

    state = make_state(tmp_path / "run")
    (state.paths.pipeline_dir / "paper.md").write_text(
        PAPER, encoding="utf-8")
    elements = [
        {"id": "eq-posterior", "type": "equation", "name": "Posterior",
         "section": "Section 3.2, Eq. (2)", "code_role": "implement",
         "source_text": _SLIPPED_QUOTE},
        {"id": "eq-paraphrased", "type": "equation", "name": "Paraphrase",
         "section": "Section 3.3", "code_role": "implement",
         "source_text": "a paraphrased equation description with no "
                        "verbatim counterpart anywhere in this paper"},
    ]
    state.paths.paper_map.write_text(
        json.dumps({"title": "T", "elements": elements}), encoding="utf-8")

    adopted = run_pipeline._adopt_failing_equation_quotes(state)

    assert adopted == 1
    fixed = json.loads(state.paths.paper_map.read_text(encoding="utf-8"))
    by_id = {e["id"]: e for e in fixed["elements"]}
    assert by_id["eq-posterior"]["source_text"] == _PAPER_EQ
    remaining = check_equation_quotes(fixed, PAPER)
    assert not any("eq-posterior" in line for line in remaining)
    assert any("eq-paraphrased" in line for line in remaining)

    # The adoption is loud on every surface: run event, assumption entry
    # with the original quote preserved, and the report sidecar.
    events = (state.paths.pipeline_dir / "run_events.jsonl").read_text(
        encoding="utf-8")
    assert "equation_quotes_adopted" in events
    import run_layout
    assumptions = run_layout.run_path(
        state.paths.run_dir, run_layout.ASSUMPTIONS_MD).read_text(
        encoding="utf-8")
    assert "eq-posterior" in assumptions
    sidecar = json.loads(
        (state.paths.pipeline_dir / "quote_adoptions.json").read_text(
            encoding="utf-8"))
    assert len(sidecar) == 1
    assert sidecar[0]["element_id"] == "eq-posterior"
    assert sidecar[0]["original_quote"] == _SLIPPED_QUOTE
    assert sidecar[0]["adopted_passage"] == _PAPER_EQ
    assert sidecar[0]["similarity"] >= ADOPTION_FLOOR
    assert sidecar[0]["assumption_id"] in assumptions


def test_driver_adoption_pass_no_failing_quotes_is_a_noop(tmp_path):
    import run_pipeline

    state = make_state(tmp_path / "run")
    (state.paths.pipeline_dir / "paper.md").write_text(
        PAPER, encoding="utf-8")
    state.paths.paper_map.write_text(
        json.dumps({"title": "T", "elements": [
            {"id": "eq-ok", "type": "equation", "name": "OK",
             "section": "Section 3.2", "code_role": "implement",
             "source_text": _PAPER_EQ}]}), encoding="utf-8")
    assert run_pipeline._adopt_failing_equation_quotes(state) == 0
    assert not (state.paths.pipeline_dir / "quote_adoptions.json").exists()


# ---------------------------------------------------------------------------
# Math transliteration (2026-08-06 forecasting stage-1 halt)
#
# A producer copying a paper sentence containing inline math rewrites the math
# into plain text and leaves the prose alone. The fold cannot see through that,
# because it deliberately keeps LaTeX command spelling so twin equations stay
# apart, so adoption scores the same span a second way.
# ---------------------------------------------------------------------------

_LAG_SENTENCE = (
    "Since article attributes are static and can be directly inputted in the "
    "temporal component, we limit node features to P demand lags:  "
    r"$\mathcal{N}_i^t = (y_i^{t-P+1}, y_i^{t-P+2}, \dots, y_i^t)$ . "
    "The number of lags serves as a hyperparameter. In addition, we calculate "
    "the node in-degree, which measures the number of neighbors."
)
_LAG_TRANSLITERATED = (
    "we limit node features to P demand lags: "
    "N_i^t = (y_i^{t-P+1}, y_i^{t-P+2}, ..., y_i^t). "
    "The number of lags serves as a hyperparameter."
)


def test_transliterated_math_adopts_the_papers_own_bytes():
    got = qa.find_adoption(_LAG_TRANSLITERATED, _LAG_SENTENCE)
    assert got is not None
    assert got["score"] == 1.0
    assert r"\mathcal{N}_i^t" in got["passage"]


def test_the_plain_fold_alone_would_have_refused_it():
    # Pins WHY the second pass exists: scored the old way, the trimmed passage
    # is a ~0.92 near-match, under the floor, and the run halts on a quote that
    # is the paper's own sentence with its math retyped.
    best = qa.scan_candidates(_LAG_TRANSLITERATED, _LAG_SENTENCE)[0]
    assert best["folded_score"] < qa.ADOPTION_FLOOR
    passage = _LAG_SENTENCE[best["start"]:best["end"]]
    assert qa.folded_similarity(_LAG_TRANSLITERATED, passage) < qa.ADOPTION_FLOOR
    assert qa.transliterated_similarity(_LAG_TRANSLITERATED, passage) == 1.0


def test_an_unknown_command_is_left_standing_so_twins_stay_apart():
    # The protection the conservative rule buys. Stripping every `\command`
    # would collapse these two into one another and let adoption retarget a
    # quote onto the wrong equation.
    alpha = r"$\alpha_i^t = \sum_j w_{ij} h_j$ over the neighbourhood"
    beta = r"$\beta_i^t = \sum_j w_{ij} h_j$ over the neighbourhood"
    assert qa.transliterated_similarity(alpha, beta) < qa.ADOPTION_FLOOR
    assert r"\alpha" in qa.transliterate_math(alpha)


def test_a_paraphrase_is_not_rescued_by_transliteration():
    paraphrase = ("The model uses several previous demand values as inputs "
                  "and the count of those values is tunable.")
    assert qa.find_adoption(paraphrase, _LAG_SENTENCE) is None


def test_a_reconstructed_table_row_is_still_refused():
    # The other half of the same halt. An appendix grid has no definition
    # sentence, the quote is assembled from a header and a cell, and the row
    # repeats — so uniqueness refuses it however it is scored.
    table = (
        "| Component | Hyperparameter | retail | e-commerce |\n"
        "| GNN | Hidden size       | _                  | [16, 8]            |\n"
        "| GNN | Learning rate     | 0.001              | 0.001              |\n"
        "| GNN2 | Hidden size       | _                  | [16, 8]            |\n"
    )
    assert qa.find_adoption("Hidden size [16, 8]", table) is None


def test_known_transliterations_normalize_both_directions():
    assert qa.transliterate_math(r"$G_i \in \mathbb{R}^{D \times P}$") == \
        qa.transliterate_math("G_i in R^{D x P}")
    assert qa.transliterate_math(r"a \dots b") == qa.transliterate_math("a ... b")


def test_nested_wrappers_unwind():
    assert qa.transliterate_math(r"\mathbf{\mathrm{X}}") == "X"


def test_candidates_carry_both_scores_for_evidence():
    cands = qa.scan_candidates(_LAG_TRANSLITERATED, _LAG_SENTENCE)
    assert cands
    best = cands[0]
    assert best["transliterated_score"] == 1.0
    assert best["folded_score"] < 1.0
    assert best["score"] == 1.0


# --- R2C-080: named operators and fractions (the 2026-08-07 halt) -------------
#
# The second transliteration shape, from a live roll. The paper writes
#   $$(A_i, A_j) := \cos(X_i, X_j) = \frac{X_i \cdot X_j}{||X_i||||X_j||}$$
# and the producer retypes it as
#   (A_i, A_j) := cos(X_i, X_j) = (X_i * X_j) / (||X_i|| ||X_j||)
# Neither `\cos` nor `\frac` was expressible in the original symbol-swap table.

_COS_PAPER = (
    "similarity \n"
    r"$$(A_i, A_j) := \cos(X_i, X_j) = \frac{X_i \cdot X_j}{||X_i||||X_j||}$$"
)
_COS_RETYPED = (
    "similarity (A_i, A_j) := cos(X_i, X_j) = (X_i * X_j) / (||X_i|| ||X_j||)"
)


def test_named_operators_drop_their_backslash():
    """The spelling IS the command name, so this preserves the symbol rather
    than guessing at it."""
    assert qa.transliterate_math(r"\cos(x)") == "cos(x)"
    assert qa.transliterate_math(r"\log(x)") == "log(x)"
    assert qa.transliterate_math(r"\max(a, b)") == "max(a, b)"


def test_longer_operator_names_win_over_their_prefixes():
    """`\\cosh` must not become `cosh` via `\\cos` + a stray h, and `\\arccos`
    must not become `arccos` by accident of ordering."""
    assert qa.transliterate_math(r"\cosh(x)") == "cosh(x)"
    assert qa.transliterate_math(r"\arccos(x)") == "arccos(x)"
    assert qa.transliterate_math(r"\limsup_n a_n") == "limsup_n a_n"


def test_an_unlisted_command_is_still_left_standing():
    """The conservatism R2C-073 argued for is unchanged: an operator we do not
    know keeps its backslash rather than being guessed at."""
    assert qa.transliterate_math(r"\erf(x)") == r"\erf(x)"
    assert qa.transliterate_math(r"\alpha") == r"\alpha"


def test_fractions_become_parenthesised_division():
    assert qa.transliterate_math(r"\frac{a}{b}") == "(a) / (b)"
    assert qa.transliterate_math(r"\dfrac{a}{b}") == "(a) / (b)"


def test_nested_fractions_unwind_innermost_first():
    assert qa.transliterate_math(r"\frac{\frac{a}{b}}{c}") == "((a) / (b)) / (c)"


def test_fractions_preserve_their_operands_verbatim():
    """The structural rule must stay symbol-preserving, so two fractions that
    differ only in a subscript cannot fold into each other."""
    one = qa.transliterate_math(r"\frac{X_i}{Y_i}")
    two = qa.transliterate_math(r"\frac{X_j}{Y_j}")
    assert one != two


def test_alpha_beta_twins_stay_apart_under_the_extension():
    """R2C-073's measured guardrail. Aggressive backslash-stripping scores this
    pair at 1.0 and would let adoption retarget onto the wrong equation."""
    a = r"$L = \alpha \cdot x + \beta$"
    b = r"$L = \beta \cdot x + \alpha$"
    assert qa.transliterated_similarity(a, b) < 0.95


def test_the_complete_cosine_equation_now_transliterates_to_adoptable():
    """The extension's own arm: scored over the COMPLETE paper equation, the
    retyped quote clears the floor (0.82 folded -> 0.99 transliterated)."""
    assert qa.folded_similarity(_COS_RETYPED, _COS_PAPER) < 0.95
    assert qa.transliterated_similarity(_COS_RETYPED, _COS_PAPER) >= 0.95


def test_the_candidate_window_still_caps_this_shape_below_the_floor():
    """KNOWN REMAINING GAP, measured rather than assumed (R2C-080 second half).

    The table extension is necessary but not sufficient. `scan_candidates`
    sizes its window against the QUOTE, and the paper's LaTeX is longer than
    its plain-text rendering, so the window truncates before the closing
    `}$$`. An incomplete `\\frac{...}{...}` cannot match, so the rule that
    would fire never sees a complete fraction.

    This test pins the CURRENT behaviour so the window fix has a measurable
    before. When that lands, this assertion flips and the docstring goes.
    """
    cands = qa.scan_candidates(_COS_RETYPED, _COS_PAPER)
    assert cands
    best = cands[0]
    span = _COS_PAPER[best["start"]:best["end"]]
    # The window ends mid-fraction, which is the whole mechanism.
    assert "}$$" not in span
    assert best["score"] < 0.95
    # ...while the same text scored complete is comfortably adoptable.
    assert qa.transliterated_similarity(_COS_RETYPED, _COS_PAPER) >= 0.95
