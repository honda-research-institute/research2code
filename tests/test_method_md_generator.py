"""METHOD.md generator (slice 2.1): structure, degradation, determinism.

The contract from hri-requirements.md: paper_map.json alone must yield a
useful document (the downstream-failure fallback); spec/params enrich;
explanation slots are explicitly pending until the explainer sidecar
fills them — never silently empty.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from generate_method_md import (
    PENDING,
    UNAVAILABLE,
    generate_method_md,
    method_md_has_substantive_explanation,
)
from schemas.method_spec import EvaluationProtocol
from schemas.params import Params

# The BADGE snapshot is a committed fixture (example_runs/ became the
# researcher-facing curated set on 2026-07-06 and is no longer a test
# surface). pdwa points at the curated example, which is the real
# 2026-07-05 draft delivery and carries a full .pipeline/.
BADGE_RUN = Path("tests/fixtures/delivery/badge-run-20260610")
PDWA_RUN = Path("example_runs/pdwa")


def _mini_paper_map() -> dict:
    return {
        "title": "Test Paper",
        "elements": [
            {"id": "eq-core", "type": "equation", "name": "Core equation",
             "section": "Section 3", "code_role": "implement",
             "source_text": "The paper defines f(x) = x + 1.",
             "pseudocode": "f(x) = x + 1",
             "dependencies": ["concept-base"]},
            {"id": "eq-side", "type": "equation", "name": "Side note",
             "section": "Appendix", "code_role": "theoretical",
             "source_text": "A bound."},
            {"id": "alg-main", "type": "algorithm", "name": "Main loop",
             "section": "Algorithm 1",
             "description": "Iterate until done."},
            {"id": "concept-base", "type": "concept", "name": "Base idea",
             "section": "Section 1", "code_role": "implement",
             "source_text": "The base idea."},
        ],
    }


def _concept_property_paper_map() -> dict:
    return {
        "title": "Protocol Specification",
        "elements": [
            {
                "id": "concept-request",
                "type": "concept",
                "name": "Request semantics",
                "section": "Section 2",
                "code_role": "implement",
                "description": "Defines the request semantics in detail.",
                "source_text": "A request is safe and idempotent.",
            },
            {
                "id": "prop-safe",
                "type": "property",
                "name": "Safety property",
                "section": "Section 2.1",
                "code_role": "demonstrate",
                "description": "The request does not change server state.",
                "source_text": "The request is safe.",
            },
        ],
    }


def _make_run(tmp_path: Path, paper_map: dict | None) -> Path:
    run = tmp_path / "run"
    (run / ".pipeline").mkdir(parents=True)
    if paper_map is not None:
        (run / ".pipeline" / "paper_map.json").write_text(
            json.dumps(paper_map), encoding="utf-8")
    return run


def _protocol_spec(*, horizon: int | None) -> dict:
    horizon_status = "paper_stated" if horizon is not None else "paper_unspecified"
    return {
        "core_method": {"summary": "Forecast related series."},
        "comparison": {
            "evaluation_protocol": {
                "scheme": {
                    "kind": "single_holdout",
                    "paper_value_status": "paper_stated",
                    "description": "One chronological holdout split.",
                    "evidence_quote": "We use one chronological holdout split.",
                    "paper_section": "Evaluation protocol",
                    "paper_element_ids": ["alg-main"],
                },
                "quantities": [
                    {
                        "role": "context_length",
                        "parameter_name": "context_length",
                        "paper_names": ["context length"],
                        "paper_symbols": [],
                        "value": 10,
                        "unit": "week",
                        "granularity": 1,
                        "axis_evidence_quote": (
                            "Weekly demand observations define the target time series."
                        ),
                        "axis_paper_section": "Data cadence",
                        "axis_paper_element_ids": ["axis-weekly"],
                        "paper_value_status": "paper_stated",
                        "evidence_quote": "The context length is 10 weeks.",
                        "paper_section": "Method inputs",
                        "paper_element_ids": ["alg-main"],
                    },
                    {
                        "role": "forecast_call_horizon",
                        "parameter_name": "forecast_horizon",
                        "paper_names": ["forecast horizon"],
                        "paper_symbols": ["K"],
                        "value": horizon,
                        "unit": "week",
                        "granularity": 1,
                        "axis_evidence_quote": (
                            "Weekly demand observations define the target time series."
                        ),
                        "axis_paper_section": "Data cadence",
                        "axis_paper_element_ids": ["axis-weekly"],
                        "paper_value_status": horizon_status,
                        "evidence_quote": (
                            "The forecast horizon K denotes the future forecast steps."
                            if horizon is None
                            else f"The forecast horizon K = {horizon} weeks."
                        ),
                        "paper_section": "Forecast definition",
                        "paper_element_ids": ["alg-main"],
                    },
                    {
                        "role": "validation_span",
                        "parameter_name": None,
                        "paper_names": ["validation"],
                        "paper_symbols": [],
                        "value": 13,
                        "unit": "week",
                        "granularity": 1,
                        "axis_evidence_quote": (
                            "Weekly demand observations define the target time series."
                        ),
                        "axis_paper_section": "Data cadence",
                        "axis_paper_element_ids": ["axis-weekly"],
                        "paper_value_status": "paper_stated",
                        "evidence_quote": "Validation covers 13 weeks.",
                        "paper_section": "Evaluation splits",
                        "paper_element_ids": ["alg-main"],
                    },
                    {
                        "role": "test_span",
                        "parameter_name": None,
                        "paper_names": ["test"],
                        "paper_symbols": [],
                        "value": 26,
                        "unit": "week",
                        "granularity": 1,
                        "axis_evidence_quote": (
                            "Weekly demand observations define the target time series."
                        ),
                        "axis_paper_section": "Data cadence",
                        "axis_paper_element_ids": ["axis-weekly"],
                        "paper_value_status": "paper_stated",
                        "evidence_quote": "Test covers 26 weeks.",
                        "paper_section": "Evaluation splits",
                        "paper_element_ids": ["alg-main"],
                    },
                ],
            },
        },
    }


def _write_protocol_inputs(run: Path, *, horizon: int | None) -> None:
    spec = _protocol_spec(horizon=horizon)
    EvaluationProtocol.model_validate(
        spec["comparison"]["evaluation_protocol"]
    )
    (run / ".pipeline" / "method_spec.json").write_text(
        json.dumps(spec), encoding="utf-8"
    )
    horizon_entry = {
        "value": 4,
        "source": "system_inferred" if horizon is None else "system_default",
        "reasoning": "System-owned demo horizon.",
        "paper_says": (
            "The forecast horizon K denotes the future forecast steps."
            if horizon is None
            else f"The forecast horizon K = {horizon} weeks."
        ),
        "paper_section": "Forecast definition",
        "protocol_role": "forecast_call_horizon",
        "protocol_value": horizon,
        "protocol_unit": "week",
        "protocol_granularity": 1,
        "protocol_axis_says": (
            "Weekly demand observations define the target time series."
        ),
        "protocol_axis_section": "Data cadence",
        "protocol_axis_element_ids": ["axis-weekly"],
        "paper_value_status": (
            "paper_unspecified" if horizon is None else "paper_stated"
        ),
        "paper_element_ids": ["alg-main"],
    }
    if horizon is not None:
        horizon_entry["paper_value"] = horizon
    params = {
        "params": {
            "context_length": {
                "value": 10,
                "source": "paper",
                "paper_says": "The context length is 10 weeks.",
                "paper_section": "Method inputs",
                "protocol_role": "context_length",
                "protocol_value": 10,
                "protocol_unit": "week",
                "protocol_granularity": 1,
                "protocol_axis_says": (
                    "Weekly demand observations define the target time series."
                ),
                "protocol_axis_section": "Data cadence",
                "protocol_axis_element_ids": ["axis-weekly"],
                "paper_value_status": "paper_stated",
                "paper_element_ids": ["alg-main"],
            },
            "forecast_horizon": horizon_entry,
        },
    }
    Params.model_validate(params)
    (run / ".pipeline" / "params.json").write_text(
        json.dumps(params),
        encoding="utf-8",
    )


def test_paper_map_alone_yields_complete_skeleton(tmp_path):
    run = _make_run(tmp_path, _mini_paper_map())
    md = generate_method_md(run)
    # Implement-role equation gets a full section with the paper's words.
    assert '<a id="eq-core"></a>' in md
    assert "The paper defines f(x) = x + 1." in md
    assert "f(x) = x + 1" in md
    # Theoretical-role equation stays out of the key section but is in
    # the source map (everything extracted is cited somewhere).
    assert "### <a id=\"eq-side\"></a>" not in md
    assert "eq-side" in md
    # Dependencies cross-link by element id.
    assert "[concept-base](#concept-base)" in md
    # Missing spec/params degrade with explicit notes, never silently.
    assert "not derived" in md
    assert md.count(PENDING) == 3  # what / why-novel / intuition for eq-core
    # Summary falls back to the algorithm description without a spec.
    assert "Iterate until done." in md
    assert method_md_has_substantive_explanation(md) is True


def test_concept_property_map_is_labeled_as_source_index_not_explanation(
    tmp_path,
):
    """RFC-shaped maps are rich decompositions, but not method explanations."""
    run = _make_run(tmp_path, _concept_property_paper_map())

    md = generate_method_md(run)

    assert UNAVAILABLE in md
    assert "# Protocol Specification — method explanation unavailable" in md
    assert "not a complete plain-language explanation" in md
    assert "Request semantics" in md
    assert "Safety property" in md
    assert PENDING not in md
    assert "this explanation stands alone" not in md
    assert method_md_has_substantive_explanation(md) is False


def test_algorithm_only_map_remains_a_substantive_explanation(tmp_path):
    paper_map = {
        "title": "Algorithm Paper",
        "elements": [{
            "id": "alg-main",
            "type": "algorithm",
            "name": "Main algorithm",
            "section": "Algorithm 1",
            "code_role": "implement",
            "description": "Repeat the update until the score converges.",
        }],
    }
    run = _make_run(tmp_path, paper_map)

    md = generate_method_md(run)

    assert "# Algorithm Paper — the method, explained" in md
    assert "Repeat the update until the score converges." in md
    assert UNAVAILABLE not in md
    assert method_md_has_substantive_explanation(md) is True


def test_equation_only_map_remains_a_substantive_explanation(tmp_path):
    paper_map = {
        "title": "Equation Paper",
        "elements": [{
            "id": "eq-update",
            "type": "equation",
            "name": "Update rule",
            "section": "Equation 1",
            "code_role": "implement",
            "source_text": "x_next = x - eta * grad(x)",
        }],
    }
    run = _make_run(tmp_path, paper_map)

    md = generate_method_md(run)

    assert "# Equation Paper — the method, explained" in md
    assert "x_next = x - eta * grad(x)" in md
    assert PENDING in md
    assert UNAVAILABLE not in md
    assert method_md_has_substantive_explanation(md) is True


def test_explanations_sidecar_fills_slots(tmp_path):
    run = _make_run(tmp_path, _mini_paper_map())
    sidecar = tmp_path / "explanations.json"
    sidecar.write_text(json.dumps({
        "schema_version": "1.0.0",
        "explanations": {"eq-core": {
            "what": "Adds one to the input.",
            "why_novel": "First paper to add one.",
            "intuition": "Counting up.",
        }},
    }), encoding="utf-8")
    md = generate_method_md(run, sidecar)
    assert "Adds one to the input." in md
    assert "First paper to add one." in md
    assert "Counting up." in md
    assert PENDING not in md
    # The generation record names its explanation source.
    assert "explanations.json" in md.splitlines()[0]


def test_missing_paper_map_is_a_hard_failure(tmp_path):
    run = _make_run(tmp_path, None)
    with pytest.raises(FileNotFoundError):
        generate_method_md(run)


def test_determinism(tmp_path):
    run = _make_run(tmp_path, _mini_paper_map())
    assert generate_method_md(run) == generate_method_md(run)


def test_protocol_block_keeps_symbolic_horizon_separate_from_spans_and_runtime(
    tmp_path,
):
    """The pdfgnn known-bad shape: neither 1 nor test span 26 becomes K."""
    run = _make_run(tmp_path, _mini_paper_map())
    _write_protocol_inputs(run, horizon=None)

    md = generate_method_md(run)

    assert "## Evaluation protocol" in md
    assert "**Paper evaluation scheme:** **single holdout**" in md
    assert "paper IDs: `alg-main`" in md
    assert "| role/value evidence | axis evidence |" in md
    assert "Context length** | 10 (paper-stated)" in md
    assert (
        "| **One-call forecast horizon** | **paper-unspecified** | "
        "week; 1 week per protocol step | `forecast_horizon` = 4 "
        "(system inferred) | Forecast definition; paper IDs: `alg-main` | "
        "Data cadence; paper IDs: `axis-weekly` |"
    ) in md
    assert "| **Validation span** | 13 (paper-stated) |" in md
    assert "| **Test span** | 26 (paper-stated) |" in md
    assert md.count("spec-only fact; no runtime parameter") == 2
    assert "One-call forecast horizon** | 1" not in md
    assert "One-call forecast horizon** | 26" not in md


def test_protocol_block_keeps_explicit_paper_horizon_and_demo_value_separate(
    tmp_path,
):
    """Synthetic explicit-K renderer control; not second-paper evidence."""
    run = _make_run(tmp_path, _mini_paper_map())
    _write_protocol_inputs(run, horizon=8)

    md = generate_method_md(run)

    assert (
        "| **One-call forecast horizon** | 8 (paper-stated) | "
        "week; 1 week per protocol step | `forecast_horizon` = 4 "
        "(system default) | Forecast definition; paper IDs: `alg-main` | "
        "Data cadence; paper IDs: `axis-weekly` |"
    ) in md


def test_legacy_method_spec_adds_no_evaluation_protocol_block(tmp_path):
    run = _make_run(tmp_path, _mini_paper_map())
    (run / ".pipeline" / "method_spec.json").write_text(
        json.dumps({"core_method": {"summary": "Legacy summary."}}),
        encoding="utf-8",
    )
    (run / ".pipeline" / "params.json").write_text(
        json.dumps({"params": {}}), encoding="utf-8"
    )

    assert "## Evaluation protocol" not in generate_method_md(run)


@pytest.mark.skipif(not BADGE_RUN.is_dir(), reason="BADGE run dir absent")
def test_real_badge_run_generates_fully():
    md = generate_method_md(BADGE_RUN)
    record = json.loads(
        md.splitlines()[0].removeprefix("<!-- method-md ").removesuffix(" -->"))
    assert record["inputs"]["method_spec"] is True
    assert record["inputs"]["params"] is True
    assert len(record["explained_equations"]) >= 5
    # Every explained equation has its anchor and three pending slots.
    for eid in record["explained_equations"]:
        assert f'<a id="{eid}"></a>' in md
    assert md.count(PENDING) == 3 * len(record["explained_equations"])
    # Provenance table carries real derived entries.
    assert "| initial_labeled |" in md
    assert "select_batch" in md  # pluggable contract in the walkthrough


@pytest.mark.skipif(not PDWA_RUN.is_dir(), reason="pdwa run dir absent")
def test_real_pdwa_run_generates_across_paradigms():
    md = generate_method_md(PDWA_RUN)
    assert "# " in md and "## The key equations" in md
    assert "## Parameters and provenance" in md


def test_latex_equation_quote_renders_as_display_math(tmp_path):
    """The 2026-07-07 equation-rendering pick: a verbatim LaTeX quote
    stands alone (display math renders in VS Code/GitHub), never inside
    a blockquote whose '> ' prefix would break it."""
    pm = _mini_paper_map()
    latex = (r"$$x^* = \arg \max_{x \in \mathcal{D}_u} "
             r"H[\theta|\mathcal{D}_0], \quad (1)$$")
    pm["elements"][0]["source_text"] = latex
    run = _make_run(tmp_path, pm)
    md = generate_method_md(run)
    assert latex in md
    assert f"> {latex[:20]}" not in md  # not blockquoted
    # The ASCII pseudocode fence is untouched.
    assert "f(x) = x + 1" in md


def test_ascii_quote_keeps_the_blockquote(tmp_path):
    """Pre-change artifacts (every run before 2026-07-07 evening) carry
    ASCII quotes; their rendering must not change."""
    run = _make_run(tmp_path, _mini_paper_map())
    md = generate_method_md(run)
    assert "> The paper defines f(x) = x + 1." in md


def test_walkthrough_navigation_and_cross_links(tmp_path):
    """Researcher feedback 2026-07: readers who start from the algorithm need a route —
    a jump link at the top, per-algorithm links to the equations it uses,
    and a used-in backlink on each equation section."""
    pm = _mini_paper_map()
    pm["elements"][2]["dependencies"] = ["eq-core", "eq-side"]
    run = _make_run(tmp_path, pm)
    md = generate_method_md(run)

    assert "[algorithm walkthrough](#algorithm-walkthrough)" in md
    assert '## <a id="algorithm-walkthrough"></a>Algorithm walkthrough' in md
    assert '## <a id="source-map"></a>Source map' in md
    # The walkthrough links only EXPLAINED equations (eq-side is
    # theoretical, so it stays out of the links and in the source map).
    assert ("*Equations in this algorithm, explained above: "
            "[Core equation](#eq-core)*") in md
    assert "[Side note](#eq-side)" not in md.split("Algorithm walkthrough")[1]
    # The equation section links back to the algorithm that uses it.
    assert "[Main loop](#alg-main)" in md


def test_unavailable_document_has_no_navigation_line(tmp_path):
    """The how-to-read pointer only renders when there is a walkthrough
    worth jumping to; a source-map index keeps its honest banner alone."""
    run = _make_run(tmp_path, _concept_property_paper_map())
    md = generate_method_md(run)
    assert "**How to read this document.**" not in md


def test_missing_verbatim_quote_is_marked_as_paraphrase(tmp_path):
    """Researcher feedback 2026-07: the paper-states block is a check-against-the-paper
    surface, so a description fallback must be visibly labeled — a reader
    must never mistake the decomposition's paraphrase for the paper's
    words."""
    pm = _mini_paper_map()
    pm["elements"][0]["source_text"] = ""
    pm["elements"][0]["description"] = "Adds one to the input value."
    run = _make_run(tmp_path, pm)
    md = generate_method_md(run)
    assert "*(paraphrase — the verbatim quote was not captured" in md
    assert "> Adds one to the input value." in md


def test_verbatim_quote_carries_no_paraphrase_marker(tmp_path):
    run = _make_run(tmp_path, _mini_paper_map())
    md = generate_method_md(run)
    assert "paraphrase — the verbatim quote was not captured" not in md
    assert "> The paper defines f(x) = x + 1." in md
