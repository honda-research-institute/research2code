"""Equation verbatim-quote floor (the 2026-07-07 equation-rendering pick).

Equation `source_text` promises the paper's own words — the equation
EXACTLY as the paper writes it, LaTeX and delimiters included — so
METHOD.md can display real math instead of an ASCII transliteration
(the live gap: the maintainer's mock-run read, eq (1) of bayesian-active-learning).
The floor is a whitespace-normalized substring check against paper.md,
and doubles as the JSON-corruption detector: a backslash that lost its
double-escape (`\theta` parsing as tab + "heta") can never be a
substring of the paper.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from validate_paper_map import check_equation_quotes

# The live shape: eq (1) of bayesian-active-learning as Marker wrote it.
PAPER_LATEX = (
    r"$$x^* = \arg \max_{x \in \mathcal{D}_n} H[\theta|\mathcal{D}_0] - "
    r"\mathbb{E}_{y \sim p(y|x,\mathcal{D}_0)} "
    r"[H[\theta|x,y,\mathcal{D}_0]], \quad (1)$$"
)
PAPER_TEXT = (
    "## 3.1 BALD\n\nThe acquisition criterion is defined as\n\n"
    + PAPER_LATEX
    + "\n\nwhere the model parameters are updated each round.\n"
)


def _eq(source_text: str, el_id: str = "eq-bald") -> dict:
    return {"id": el_id, "type": "equation", "name": "BALD criterion",
            "section": "Section 3.1, Eq (1)", "description": "Acquisition.",
            "source_text": source_text, "code_role": "implement"}


def test_verbatim_latex_passes_across_line_wraps():
    # The decomposer may wrap the quote differently than Marker did;
    # whitespace normalization makes that immaterial.
    wrapped = PAPER_LATEX.replace(" - ", "\n- ").replace(", \\quad", ",\n\\quad")
    errors = check_equation_quotes(
        {"elements": [_eq(wrapped)]}, PAPER_TEXT)
    assert errors == []


def test_ascii_transliteration_fails_naming_the_element():
    # The pre-change behavior, verbatim from the live mock-run artifact.
    ascii_form = ("x* = arg max_{x in D_u} H[theta|D_0] - "
                  "E_{y ~ p(y|x,D_0)} [H[theta|x,y,D_0]]")
    errors = check_equation_quotes(
        {"elements": [_eq(ascii_form)]}, PAPER_TEXT)
    assert len(errors) == 1
    assert "eq-bald" in errors[0]
    assert "verbatim" in errors[0]


def test_eaten_escape_corruption_fails():
    # `\theta` written with a single backslash in the JSON file parses
    # as a tab + "heta" — silent corruption the floor must catch.
    corrupted = PAPER_LATEX.replace("\\theta", "\theta")
    errors = check_equation_quotes(
        {"elements": [_eq(corrupted)]}, PAPER_TEXT)
    assert len(errors) == 1
    assert "backslash" in errors[0]


def test_over_escaped_backslashes_get_the_halved_backslash_hint():
    # The 2026-07-13 genus (both iDb-RRT terminal failures and ACC's one
    # rejection): the file carries FOUR backslash characters per LaTeX
    # backslash, so the parsed quote has `\\theta` where the paper has
    # `\theta`. Judges twice misdiagnosed it as a delimiter problem, so
    # the error must name the over-escape and the one-edit fix.
    over_escaped = PAPER_LATEX.replace("\\", "\\\\")
    errors = check_equation_quotes(
        {"elements": [_eq(over_escaped)]}, PAPER_TEXT)
    assert len(errors) == 1
    assert "ONE LEVEL TOO MANY" in errors[0]
    assert "halve" in errors[0]


def test_over_escape_hint_does_not_fire_on_genuine_mismatch():
    # A quote that is wrong even after collapsing doubled backslashes gets
    # the generic verbatim message, not the over-escape hint.
    wrong = "$$\\\\alpha + \\\\beta = \\\\gamma$$"
    errors = check_equation_quotes(
        {"elements": [_eq(wrong)]}, PAPER_TEXT)
    assert len(errors) == 1
    assert "ONE LEVEL TOO MANY" not in errors[0]
    assert "verbatim" in errors[0]


def test_prose_elements_and_empty_quotes_are_exempt():
    prose = {"id": "concept-al", "type": "concept", "name": "Active learning",
             "section": "Section 1", "description": "Setting.",
             "source_text": "A free paraphrase that is nowhere in the paper.",
             "code_role": "theoretical"}
    empty = _eq("", el_id="eq-unquoted")
    errors = check_equation_quotes(
        {"elements": [prose, empty]}, PAPER_TEXT)
    assert errors == []


def _write_run(tmp_path: Path, source_text: str, with_paper: bool = True):
    tmp_path.mkdir(parents=True, exist_ok=True)
    pm = {"title": "T", "elements": [_eq(source_text)]}
    map_path = tmp_path / "paper_map.json"
    map_path.write_text(json.dumps(pm), encoding="utf-8")
    if with_paper:
        (tmp_path / "paper.md").write_text(PAPER_TEXT, encoding="utf-8")
    return map_path


def _run_cli(map_path: Path, *extra: str):
    return subprocess.run(
        [sys.executable, "scripts/validate_paper_map.py",
         str(map_path), *extra],
        capture_output=True, text=True)


def test_cli_passes_verbatim_and_fails_transliteration(tmp_path):
    good = _run_cli(_write_run(tmp_path / "good", PAPER_LATEX))
    assert good.returncode == 0, good.stderr

    bad = _run_cli(_write_run(tmp_path / "bad", "x* = arg max over D_u"))
    assert bad.returncode == 1
    assert "eq-bald" in bad.stderr
    assert "verbatim" in bad.stderr


def test_cli_skips_floor_without_paper_md(tmp_path):
    # Standalone validation (fixtures, zoo) keeps working; the skip is
    # announced, never silent.
    res = _run_cli(_write_run(tmp_path, PAPER_LATEX, with_paper=False))
    assert res.returncode == 0, res.stderr
    assert "skipped" in res.stdout


def test_cli_explicit_paper_md_flag(tmp_path):
    map_path = _write_run(tmp_path / "run", PAPER_LATEX, with_paper=False)
    paper = tmp_path / "elsewhere.md"
    paper.write_text(PAPER_TEXT, encoding="utf-8")
    res = _run_cli(map_path, "--paper-md", str(paper))
    assert res.returncode == 0, res.stderr
    assert "skipped" not in res.stdout
