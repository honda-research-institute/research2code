"""Explanation math-sanity pass — checker + quote-anchoring floor (item 3).

Two live defects of one genus motivate this pass: the archived GBALD "Maximized
Weighted Likelihood" invented-mechanism claim (audit 1/3 finding 1) and the
fresh roll's inverted core-set equivalence that survived the 07-06 prompt fixes.
The `quote` fields below are byte-exact copies from the real deliveries
(bayesian-active-learning-20260703-first-verified and bayesian-active-learning_4)
so the anti-hallucination floor is exercised against real text.

These cover the design's checker-level tests (1-6) and the safe-evaluator. The
stage-1x wiring tests (label invariance #7, run-event registry #8) live with the
wiring change.
"""

from __future__ import annotations

import explanation_math_sanity as ms


# --- verbatim fixtures (copied character-for-character from the deliveries) --

# Item 1: archived confabulation. The "what"/"intuition" assert the product form
# has an emergent "diminishing return" mechanism; algebraically logp*(1-logp) is
# just logp - logp**2, so it introduces no new behavior.
CONFAB_ENTRY = {
    "what": (
        "The term log p(yᵢ|xᵢ,θ) · (1 − log p(yᵢ|xᵢ,θ)) "
        "jointly rewards high likelihood (the log p factor) while penalizing "
        "overconfidence (the 1 − log p factor acts as a diminishing return: "
        "when log p approaches 1, the whole product shrinks)."
    ),
    "why_novel": "The paper frames this as its weighted-likelihood contribution.",
    "intuition": (
        "**The intuition:** This objective has a built-in diminishing return "
        "for already-represented regions."
    ),
}
CONFAB_CLAIM = {
    "claim_type": "novelty",
    "quote": "This objective has a built-in diminishing return for already-represented regions.",
    "formulas": [],
    "formal": {
        "lhs": "logp * (1 - logp)",
        "rhs": "logp - logp*logp",
        "relation": "introduces_new_behavior",
        "variables": [{"name": "logp", "domain": [-5.0, 0.0]}],
    },
}

# Item 2: live false equivalence. log-prob is monotone DECREASING in distance,
# so maximizing min log-prob picks the CLOSEST point, not the farthest — the two
# argmax forms are not equivalent.
CORESET_ENTRY = {
    "what": "The simplified selector uses a max-min geometric core-set rule.",
    "why_novel": (
        "The likelihood whose logarithm is a monotone decreasing function of "
        "distance (log(R_0 / dist) = log R_0 - log dist). This means maximizing "
        "the minimum log-probability is mathematically equivalent to maximizing "
        "the minimum distance (the additive constant log R_0 cancels in the "
        "argmax)."
    ),
    "intuition": "Pick the point whose nearest labeled neighbor is most informative.",
}
CORESET_CLAIM = {
    "claim_type": "equivalence",
    "quote": (
        "maximizing the minimum log-probability is mathematically equivalent to "
        "maximizing the minimum distance"
    ),
    "formulas": [],
    "formal": {
        "lhs": "log(R0) - log(d)",
        "rhs": "d",
        "relation": "argmax_equivalent",
        "variables": [
            {"name": "d", "domain": [0.05, 1.0]},
            {"name": "R0", "domain": [1.0, 10.0]},
        ],
    },
}

# Item 3a: honest factorization (correct dual of item 1) — an identity claim.
HONEST_ENTRY = {
    "what": "The algebraic rewriting frames each candidate's contribution as a trade-off.",
    "why_novel": (
        "However, this factorization is a direct algebraic consequence of the "
        "entropy substitution and does not introduce new algorithmic behavior."
    ),
    "intuition": "It is the same objective, written to expose the trade-off.",
}
HONEST_CLAIM = {
    "claim_type": "novelty",
    "quote": (
        "this factorization is a direct algebraic consequence of the entropy "
        "substitution and does not introduce new algorithmic behavior"
    ),
    "formulas": [],
    "formal": {
        "lhs": "logp * (1 - logp)",
        "rhs": "logp - logp*logp",
        "relation": "algebraic_identity",
        "variables": [{"name": "logp", "domain": [-5.0, 0.0]}],
    },
}

# Item 3b: healthy domain/sign claim — log p is non-positive on (0,1].
DOMAIN_ENTRY = {
    "what": (
        "Note: log p is non-positive for all p in (0,1], so the log-probability "
        "term is always a penalty or zero — the maximization trades off "
        "large model divergence against proximity to labeled data."
    ),
    "why_novel": "A standard property of the log on the unit interval.",
    "intuition": "Probabilities never exceed one, so their logs never exceed zero.",
}
DOMAIN_CLAIM = {
    "claim_type": "domain",
    "quote": (
        "log p is non-positive for all p in (0,1], so the log-probability term "
        "is always a penalty or zero"
    ),
    "formulas": [],
    "formal": {
        "lhs": "log(p)",
        "relation": "nonpositive",
        "variables": [{"name": "p", "domain": [0.001, 1.0]}],
    },
}


# --- design test 1: archived confabulation, refuted ------------------------


def test_archived_confabulation_novelty_refuted():
    result = ms.evaluate_entry("eq-8", CONFAB_ENTRY, [CONFAB_CLAIM])
    assert result.dropped == []          # the quote IS in the entry
    assert len(result.refuted) == 1
    v = result.refuted[0]
    assert v.verdict == ms.REFUTED
    assert "no new behavior" in v.detail
    findings = ms.findings_from_result(result)
    assert findings and findings[0]["check"] == "math_sanity_refuted"
    assert findings[0]["element_id"] == "eq-8"


# --- design test 2: live false equivalence, refuted with a counterexample --


def test_live_false_equivalence_refuted_with_counterexample():
    result = ms.evaluate_entry("eq-coreset", CORESET_ENTRY, [CORESET_CLAIM])
    assert len(result.refuted) == 1
    v = result.refuted[0]
    assert v.verdict == ms.REFUTED
    # A concrete, researcher-readable counterexample with the flipping points.
    assert v.counterexample is not None
    assert {"point_a", "point_b"} <= set(v.counterexample)
    assert "select different points" in v.detail


def test_false_equivalence_counterexample_is_reproducible():
    a = ms.check_claim(CORESET_CLAIM)
    b = ms.check_claim(CORESET_CLAIM)
    assert a.verdict == b.verdict == ms.REFUTED
    assert a.counterexample == b.counterexample   # seed-fixed


# --- design test 3: faithful sections pass untouched (false-positive floor) -


def test_healthy_factorization_consistent():
    result = ms.evaluate_entry("eq-8h", HONEST_ENTRY, [HONEST_CLAIM])
    assert result.refuted == []
    assert result.consistent_count == 1
    assert ms.findings_from_result(result) == []


def test_healthy_domain_claim_consistent():
    result = ms.evaluate_entry("eq-dom", DOMAIN_ENTRY, [DOMAIN_CLAIM])
    assert result.refuted == []
    assert result.consistent_count == 1
    assert ms.findings_from_result(result) == []


def test_healthy_fixture_set_zero_rejections():
    """Both healthy entries together produce zero rejections — the false-positive
    floor the whole pass is judged against."""
    findings = []
    findings += ms.findings_from_result(
        ms.evaluate_entry("eq-8h", HONEST_ENTRY, [HONEST_CLAIM]))
    findings += ms.findings_from_result(
        ms.evaluate_entry("eq-dom", DOMAIN_ENTRY, [DOMAIN_CLAIM]))
    assert findings == []


# --- design test 4: quote anchoring is a hard floor ------------------------


def test_quote_not_in_entry_is_dropped():
    bad = {**CORESET_CLAIM, "quote": "a sentence the explainer never wrote"}
    result = ms.evaluate_entry("eq-coreset", CORESET_ENTRY, [bad])
    assert result.verdicts == []          # never reached the checker
    assert len(result.dropped) == 1
    assert result.dropped[0]["reason"] == "quote_not_in_entry"


def test_formula_label_absent_is_dropped():
    bad = {**DOMAIN_CLAIM, "formulas": ["eq-999-not-here"]}
    kept, dropped = ms.anchor_claims([bad], "\n".join(DOMAIN_ENTRY.values()))
    assert kept == []
    assert dropped[0]["reason"] == "formula_label_absent"


def test_formula_label_present_in_spec_is_kept():
    claim = {**DOMAIN_CLAIM, "formulas": ["eq-12"]}
    kept, dropped = ms.anchor_claims(
        [claim], "\n".join(DOMAIN_ENTRY.values()), spec_labels={"eq-12"})
    assert len(kept) == 1 and dropped == []


# --- design test 5: not-checkable never blocks -----------------------------


def test_prose_only_claim_is_not_checkable_and_never_blocks():
    prose_only = {
        "claim_type": "domain",
        "quote": DOMAIN_CLAIM["quote"],
        "formulas": [],
        "formal": {"lhs": "log(p)", "relation": "nonpositive",
                   "variables": []},   # no domain for p ⇒ cannot sample
    }
    result = ms.evaluate_entry("eq-dom", DOMAIN_ENTRY, [prose_only])
    assert result.not_checkable_count == 1
    assert result.refuted == []
    assert ms.findings_from_result(result) == []     # never a rejection


# --- design test 6: budget honored -----------------------------------------


def test_pathological_sample_space_is_not_checkable():
    """More free variables than the search budget ⇒ not_checkable, no hang."""
    huge = {
        "claim_type": "equivalence",
        "quote": CORESET_CLAIM["quote"],
        "formulas": [],
        "formal": {
            "lhs": "a + b + c + d + e",
            "rhs": "e + d + c + b + a",
            "relation": "algebraic_identity",
            "variables": [{"name": n, "domain": [0.0, 1.0]}
                          for n in ("a", "b", "c", "d", "e")],
        },
    }
    v = ms.check_claim(huge)
    assert v.verdict == ms.NOT_CHECKABLE
    assert "budget" in v.detail


def test_zero_budget_yields_not_checkable_not_false_consistent():
    v = ms.check_claim(HONEST_CLAIM, budget_s=0.0)
    assert v.verdict == ms.NOT_CHECKABLE


# --- the safe evaluator is not an eval() ------------------------------------


def test_attribute_access_is_not_checkable_not_executed():
    evil = {
        "claim_type": "domain", "quote": DOMAIN_CLAIM["quote"], "formulas": [],
        "formal": {"lhs": "p.__class__", "relation": "nonpositive",
                   "variables": [{"name": "p", "domain": [0.1, 1.0]}]},
    }
    v = ms.check_claim(evil)
    assert v.verdict == ms.NOT_CHECKABLE


def test_unknown_call_is_not_checkable():
    evil = {
        "claim_type": "domain", "quote": DOMAIN_CLAIM["quote"], "formulas": [],
        "formal": {"lhs": "__import__('os')", "relation": "nonpositive",
                   "variables": []},
    }
    v = ms.check_claim(evil)
    assert v.verdict == ms.NOT_CHECKABLE


def test_safe_eval_computes_whitelisted_math():
    tree = ms._compile("log(exp(2)) + sqrt(9)")
    assert abs(ms._safe_eval(tree, {}) - 5.0) < 1e-9


# --- per-strategy engine checks (synthetic, direction coverage) -------------


def test_monotone_increasing_refuted_when_decreasing():
    claim = {"claim_type": "monotonicity", "quote": "q", "formulas": [],
             "formal": {"lhs": "-x", "relation": "monotone_increasing",
                        "variables": [{"name": "x", "domain": [0.0, 1.0]}]}}
    assert ms.check_claim(claim).verdict == ms.REFUTED


def test_monotone_decreasing_consistent():
    claim = {"claim_type": "monotonicity", "quote": "q", "formulas": [],
             "formal": {"lhs": "-x", "relation": "monotone_decreasing",
                        "variables": [{"name": "x", "domain": [0.0, 1.0]}]}}
    assert ms.check_claim(claim).verdict == ms.CONSISTENT


def test_identity_refuted_when_not_identical():
    claim = {"claim_type": "equivalence", "quote": "q", "formulas": [],
             "formal": {"lhs": "x*x", "rhs": "x", "relation": "algebraic_identity",
                        "variables": [{"name": "x", "domain": [2.0, 3.0]}]}}
    assert ms.check_claim(claim).verdict == ms.REFUTED


def test_nonnegative_refuted_by_negative_sample():
    claim = {"claim_type": "domain", "quote": "q", "formulas": [],
             "formal": {"lhs": "x", "relation": "nonnegative",
                        "variables": [{"name": "x", "domain": [-1.0, -0.1]}]}}
    assert ms.check_claim(claim).verdict == ms.REFUTED


# --- role=parameter: shared constants never manufacture a counterexample ----


def _param(name, lo, hi):
    return {"name": name, "domain": [lo, hi], "role": "parameter"}


def test_shared_constant_does_not_refute_true_scaling_equivalence():
    """"Minimizing s is equivalent to minimizing s/n" is TRUE when n is the
    batch size shared by every candidate. Sampling n per point used to refute
    it; with role=parameter, n is held fixed within each comparison."""
    claim = {"claim_type": "equivalence", "quote": "q", "formulas": [],
             "formal": {"lhs": "s", "rhs": "s / n",
                        "relation": "argmin_equivalent",
                        "variables": [{"name": "s", "domain": [0.1, 10.0]},
                                      _param("n", 1.0, 100.0)]}}
    assert ms.check_claim(claim).verdict == ms.CONSISTENT


def test_genuinely_inequivalent_forms_still_refuted_with_a_parameter():
    """The parameter machinery must not blunt refutation: forms that rank
    candidates oppositely are refuted even when a shared constant rides
    along."""
    claim = {"claim_type": "equivalence", "quote": "q", "formulas": [],
             "formal": {"lhs": "s", "rhs": "-s + n",
                        "relation": "argmin_equivalent",
                        "variables": [{"name": "s", "domain": [0.1, 10.0]},
                                      _param("n", 1.0, 100.0)]}}
    v = ms.check_claim(claim)
    assert v.verdict == ms.REFUTED
    # The counterexample names the constant it was found at.
    assert "n" in v.counterexample["point_a"]


def test_monotone_checkable_with_a_shared_constant():
    """One decision variable plus a parameter is checkable (was not_checkable
    under the old exactly-one-variable rule), in both directions."""
    good = {"claim_type": "monotonicity", "quote": "q", "formulas": [],
            "formal": {"lhs": "x / n", "relation": "monotone_increasing",
                       "variables": [{"name": "x", "domain": [0.1, 1.0]},
                                     _param("n", 1.0, 100.0)]}}
    assert ms.check_claim(good).verdict == ms.CONSISTENT
    bad = {"claim_type": "monotonicity", "quote": "q", "formulas": [],
           "formal": {"lhs": "-x / n", "relation": "monotone_increasing",
                      "variables": [{"name": "x", "domain": [0.1, 1.0]},
                                    _param("n", 1.0, 100.0)]}}
    assert ms.check_claim(bad).verdict == ms.REFUTED


def test_unknown_role_is_not_checkable_never_a_guess():
    claim = {"claim_type": "monotonicity", "quote": "q", "formulas": [],
             "formal": {"lhs": "x", "relation": "monotone_increasing",
                        "variables": [{"name": "x", "domain": [0.0, 1.0],
                                       "role": "hyperparameter"}]}}
    v = ms.check_claim(claim)
    assert v.verdict == ms.NOT_CHECKABLE
    assert "role" in v.detail


def test_extractor_prompt_documents_the_parameter_role():
    prompt = ms.build_extractor_prompt(
        "eq-coreset", CORESET_ENTRY, "Eq. 7: log(R_0/dist)",
        "/tmp/x/part.json")
    assert '"role": "parameter"' in prompt
    assert "shared constant" in prompt


# --- extractor prompt builder -----------------------------------------------


def test_extractor_prompt_inlines_entry_and_holds_schema():
    prompt = ms.build_extractor_prompt(
        "eq-coreset", CORESET_ENTRY, "Eq. 7: log(R_0/dist)",
        "/tmp/x/part.json")
    # The entry prose is inlined (the sole quotable source) ...
    assert "mathematically equivalent" in prompt
    # ... the formulas are inlined ...
    assert "log(R_0/dist)" in prompt
    # ... the write target + schema are specified ...
    assert "/tmp/x/part.json" in prompt
    assert '"relation"' in prompt
    # ... and the never-judge + verbatim-quote contract is stated.
    assert "do NOT judge" in prompt or "not judge" in prompt.lower()
    assert "verbatim" in prompt.lower()


# --- extractor response parsing (robust to how the agent wraps JSON) --------


def _one_claim_obj():
    return {"claims": [CORESET_CLAIM]}


def test_parse_accepts_decoded_dict():
    claims = ms.parse_extractor_response(_one_claim_obj())
    assert len(claims) == 1 and claims[0]["claim_type"] == "equivalence"


def test_parse_accepts_json_string():
    import json
    claims = ms.parse_extractor_response(json.dumps(_one_claim_obj()))
    assert len(claims) == 1


def test_parse_accepts_fenced_and_prose_wrapped_json():
    import json
    raw = ("Here are the claims I extracted:\n```json\n"
           + json.dumps(_one_claim_obj()) + "\n```\nDone.")
    claims = ms.parse_extractor_response(raw)
    assert len(claims) == 1


def test_parse_drops_malformed_claims_never_raises():
    obj = {"claims": [
        {"claim_type": "bogus", "quote": "q", "formal": {"relation": "x"}},  # bad type
        {"claim_type": "domain", "quote": "", "formal": {"relation": "nonpositive"}},  # empty quote
        {"claim_type": "domain", "quote": "q"},                               # no formal
        CORESET_CLAIM,                                                        # the one good one
    ]}
    claims = ms.parse_extractor_response(obj)
    assert len(claims) == 1
    assert claims[0]["claim_type"] == "equivalence"


def test_parse_garbage_yields_empty():
    assert ms.parse_extractor_response("not json at all") == []
    assert ms.parse_extractor_response(12345) == []
    assert ms.parse_extractor_response({"no_claims_key": True}) == []


# --- cross-entry orchestration ----------------------------------------------


def test_math_sanity_findings_aggregates_refuted_and_metadata():
    sidecar = {"schema_version": "1.0.0", "explanations": {
        "eq-coreset": CORESET_ENTRY,   # refuted
        "eq-dom": DOMAIN_ENTRY,        # consistent
    }}
    extractions = {
        "eq-coreset": {"claims": [CORESET_CLAIM]},
        "eq-dom": {"claims": [DOMAIN_CLAIM]},
    }
    findings, meta = ms.math_sanity_findings(sidecar, extractions)
    assert len(findings) == 1
    assert findings[0]["element_id"] == "eq-coreset"
    assert meta["totals"]["refuted"] == 1
    assert meta["totals"]["consistent"] == 1
    assert meta["entries"]["eq-dom"]["consistent"] == 1


def test_math_sanity_findings_skips_entries_without_extractions():
    sidecar = {"schema_version": "1.0.0",
               "explanations": {"eq-dom": DOMAIN_ENTRY}}
    findings, meta = ms.math_sanity_findings(sidecar, {})   # no extraction for it
    assert findings == []
    assert meta["entries"] == {}


def test_math_sanity_findings_counts_dropped_unanchored():
    sidecar = {"schema_version": "1.0.0",
               "explanations": {"eq-dom": DOMAIN_ENTRY}}
    bad = {**DOMAIN_CLAIM, "quote": "text the explainer never wrote"}
    findings, meta = ms.math_sanity_findings(sidecar, {"eq-dom": {"claims": [bad]}})
    assert findings == []
    assert meta["entries"]["eq-dom"]["dropped"] == 1


# --- design test 8 (partial): run events registered (growth gate) -----------


def test_math_sanity_events_are_categorized():
    import fleet_state
    assert fleet_state.EVENT_CATEGORIES["math_claim_refuted"] == "problem"
    assert fleet_state.EVENT_CATEGORIES["math_claim_dropped"] == "decision"


# --- coverage counters (the ADAM 2026-07-14 claim-free blind spot) ----------


def test_math_sanity_totals_carry_coverage_counters():
    sidecar = {"schema_version": "1.0.0", "explanations": {
        "eq-dom": DOMAIN_ENTRY,        # one consistent claim
        "eq-empty": DOMAIN_ENTRY,      # extractor found nothing to check
        "eq-unexamined": DOMAIN_ENTRY,  # never extracted (dispatch failed)
    }}
    extractions = {
        "eq-dom": {"claims": [DOMAIN_CLAIM]},
        "eq-empty": {"claims": []},
    }
    _, meta = ms.math_sanity_findings(sidecar, extractions)
    totals = meta["totals"]
    assert totals["entries_total"] == 3
    assert totals["entries_examined"] == 2
    assert totals["entries_claim_free"] == 1


def test_math_sanity_dropped_only_entry_counts_claim_free():
    """An entry whose every claim was dropped at the anti-hallucination
    floor was never actually checked — it counts as claim-free coverage."""
    sidecar = {"schema_version": "1.0.0",
               "explanations": {"eq-dom": DOMAIN_ENTRY}}
    bad = {**DOMAIN_CLAIM, "quote": "text the explainer never wrote"}
    _, meta = ms.math_sanity_findings(sidecar, {"eq-dom": {"claims": [bad]}})
    assert meta["totals"]["entries_claim_free"] == 1
