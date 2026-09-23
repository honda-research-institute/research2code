"""Explanation-sidecar gate (slice 2.1): coverage, fabrication, substance.

The explanation layer holds to the params layer's honesty bar: real
element ids only, every key equation covered, no fabricated quotes, no
slot-filler prose.
"""

from __future__ import annotations

from validate_method_explanations import check_explanations

PAPER_MAP = {
    "elements": [
        {"id": "eq-core", "type": "equation", "code_role": "implement",
         "name": "Core"},
        {"id": "eq-demo", "type": "equation", "code_role": "demonstrate",
         "name": "Demo"},
        {"id": "eq-theory", "type": "equation", "code_role": "theoretical",
         "name": "Theory"},
    ],
}

PAPER_TEXT = ("We define the gradient embedding as the derivative of the "
              "loss with respect to the last layer weights, and select a "
              "diverse batch via k-means seeding over those embeddings.")

GOOD_ENTRY = {
    "what": "Computes the gradient of the loss for each candidate point.",
    "why_novel": "Standard machinery; included because the embedding is "
                 "built from it.",
    "intuition": "Points the model is unsure about produce large "
                 "gradients, so gradient size is a stand-in for value.",
}


def _sidecar(ids):
    return {"schema_version": "1.0.0",
            "explanations": {i: dict(GOOD_ENTRY) for i in ids}}


def test_full_coverage_passes():
    findings = check_explanations(
        _sidecar(["eq-core", "eq-demo"]), PAPER_MAP, PAPER_TEXT)
    assert findings == []


def test_missing_key_equation_is_an_error():
    findings = check_explanations(_sidecar(["eq-core"]), PAPER_MAP, PAPER_TEXT)
    assert any(f["check"] == "coverage" and "eq-demo" in f["message"]
               for f in findings)


def test_fabricated_element_id_is_an_error():
    findings = check_explanations(
        _sidecar(["eq-core", "eq-demo", "eq-invented"]), PAPER_MAP, PAPER_TEXT)
    assert any(f["check"] == "fabricated_element" and "eq-invented"
               in f["message"] for f in findings)


def test_slot_filler_prose_is_an_error():
    sidecar = _sidecar(["eq-core", "eq-demo"])
    sidecar["explanations"]["eq-core"]["intuition"] = "It works."
    findings = check_explanations(sidecar, PAPER_MAP, PAPER_TEXT)
    assert any(f["check"] == "empty_field" and "eq-core.intuition"
               in f["message"] for f in findings)


def test_fabricated_quote_is_an_error_real_quote_passes():
    sidecar = _sidecar(["eq-core", "eq-demo"])
    # Verbatim from the paper: passes the windowed matcher.
    sidecar["explanations"]["eq-core"]["what"] = (
        'The paper states: "select a diverse batch via k-means seeding '
        'over those embeddings" and this computes the embedding step.')
    # Invented attribution: fails.
    sidecar["explanations"]["eq-demo"]["what"] = (
        'The authors claim "this converges in exactly seven iterations '
        'on every dataset" which motivates the demo.')
    findings = check_explanations(sidecar, PAPER_MAP, PAPER_TEXT)
    fabricated = [f for f in findings if f["check"] == "fabricated_quote"]
    assert len(fabricated) == 1
    assert "eq-demo" in fabricated[0]["message"]


def test_no_paper_text_skips_quote_check_only():
    sidecar = _sidecar(["eq-core", "eq-demo"])
    sidecar["explanations"]["eq-demo"]["what"] = (
        'The authors claim "this converges in exactly seven iterations '
        'on every dataset" which would fail with paper text present.')
    findings = check_explanations(sidecar, PAPER_MAP, None)
    assert not any(f["check"] == "fabricated_quote" for f in findings)
