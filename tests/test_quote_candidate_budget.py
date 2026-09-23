"""Stage-1 quote-candidate budgeting (queue item 11, 2026-07-21).

The SRL fix_loop_exhausted root cause: 15 equations failed the verbatim
quote floor, but the flat 1500-char per-passage cap let only ~7
candidate passages fit the 12000-char block cap — the other 8 were
dropped as "block size cap", never saw the paper's bytes, and failed
every retry (trajectory 15 -> 14 -> 11, then exhaustion). Two fixes
pinned here:

- fair-share budgeting: the block budget is split across ALL
  currently-failing elements, so every element gets a candidate;
- equation-mode math slicing: when an element's share is tighter than
  its enclosing block, the locator prefers the $$-delimited math (the
  bytes the fixer must copy) over prose context before truncating.
"""

from __future__ import annotations

import json

from tests.helpers.state import make_state

from passage_locator import locate_passages


_EQ = ("$$L(w) = \\sum_{k=1}^{K} \\frac{n_k}{n} F_k(w), \\quad "
       "F_k(w) = \\frac{1}{n_k} \\sum_{i \\in P_k} f_i(w) \\tag{1}$$")


def _paper_with_equations(n: int) -> str:
    blocks = ["# A Paper\n\nIntroductory prose about the method."]
    for i in range(n):
        blocks.append(
            f"Section {i} explains objective variant {i} with a long "
            f"contextual paragraph about equation variant{i} and its "
            f"role in the optimization pipeline, padding the enclosing "
            f"block well past a starved per-element share. "
            + f"filler{i} " * 40)
        blocks.append(_EQ.replace("L(w)", f"L_{{{i}}}(w) + variant{i}"))
    return "\n\n".join(blocks)


def test_equation_mode_slices_to_math_when_budget_is_tight():
    paper = _paper_with_equations(1)
    # The quote is rendered-ish soup sharing tokens with block 0's prose
    # and math — the SRL emission shape.
    quote = "L_0(w) + variant0 = sum n_k / n F_k(w) equation variant0"
    generous = locate_passages(quote, paper, element_type="equation",
                               max_passage_chars=1500)
    tight = locate_passages(quote, paper, element_type="equation",
                            max_passage_chars=300)
    assert generous and tight
    # Generous budget keeps context; tight budget must still deliver the
    # copyable math bytes rather than a mid-prose truncation.
    assert "$$" in tight[0]["passage"]
    assert "\\sum_{k=1}^{K}" in tight[0]["passage"]
    assert len(tight[0]["passage"]) <= 300


def test_fair_share_budget_attaches_a_candidate_for_every_failing_equation(
        tmp_path):
    """The SRL reproduction: many failing equations, one block cap. Every
    failing element must get a candidate — zero 'block size cap' drops."""
    import run_pipeline

    n = 15
    state = make_state(tmp_path / "run")
    paper = _paper_with_equations(n)
    (state.paths.pipeline_dir / "paper.md").write_text(paper,
                                                       encoding="utf-8")
    elements = [
        {"id": f"eq-{i}", "type": "equation", "name": f"Objective {i}",
         "section": f"Section {i}", "code_role": "implement",
         # Rendered-soup source_text that FAILS the verbatim floor but
         # shares tokens with its block (the real failure shape).
         "source_text": f"L_{i}(w) + variant{i} = sum n_k/n F_k(w) "
                        f"equation variant{i}"}
        for i in range(n)
    ]
    state.paths.paper_map.write_text(
        json.dumps({"title": "T", "elements": elements}), encoding="utf-8")

    block = run_pipeline._equation_quote_candidates_block(state)

    assert "block size cap" not in block
    for i in range(n):
        assert f"`eq-{i}`" in block, f"eq-{i} got no candidate"
    # The attached passages carry the copyable math bytes.
    assert block.count("$$") >= 2 * n
