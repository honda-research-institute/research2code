"""US-10 lint-gate tests: the statically-detectable smoke-killer class.

Fail-side: synthetic reconstructions of the June-9 GBALD smoke-loop bugs
(missing import → NameError class, plus a syntax error). The real buggy
intermediates weren't preserved (only the post-fix package was), so these are
labeled reconstructions per the zoo convention. Pass-side: the real committed
method packages (BADGE/GBALD/pdwa) must produce zero errors — they all
executed, so any error here is a false positive.
"""

from __future__ import annotations

from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
ZOO = REPO / "tests" / "fixtures" / "zoo"

from lint_generated_code import lint_paths  # noqa: E402


def _errors(findings):
    return [f for f in findings if f["severity"] == "error"]


def test_syntax_error_is_an_error_under_any_engine(tmp_path):
    bad = tmp_path / "method.py"
    bad.write_text("def select_batch(model, x_unlabeled:\n    return []\n")
    findings, _engine = lint_paths([bad])
    assert _errors(findings), findings


def test_missing_import_is_an_error():
    pytest.importorskip("pyflakes")
    # The June-9 GBALD smoke iteration 1 class: torch.nn used, never imported.
    src = (
        "import torch\n"
        "def build_model(input_dim):\n"
        "    return nn.Linear(input_dim, 10)\n"
    )
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        f = Path(td) / "model.py"
        f.write_text(src)
        findings, engine = lint_paths([f])
    assert engine == "pyflakes"
    errs = _errors(findings)
    assert errs and "UndefinedName" in errs[0]["message"], findings


def test_unused_import_is_warn_not_error(tmp_path):
    pytest.importorskip("pyflakes")
    f = tmp_path / "data.py"
    f.write_text("import os\nimport json\n\nVALUE = 1\n")
    findings, _ = lint_paths([f])
    assert not _errors(findings)
    assert any(f_["severity"] == "warn" for f_ in findings), findings


def test_clean_file_passes(tmp_path):
    f = tmp_path / "training.py"
    f.write_text("def train(x):\n    return x * 2\n")
    findings, _ = lint_paths([f])
    assert findings == []


@pytest.mark.parametrize("method_file", [
    ZOO / "badge-goodmethod-badnotebook" / "method.py",
    ZOO / "gbald-lr74-never-learns" / "method.py",
    ZOO / "pdwa-omega-degenerate" / "method.py",
])
def test_real_executed_packages_produce_zero_errors(method_file):
    # False-positive guard: these files all ran end-to-end; flagging them
    # would make the gate untrustworthy. (Warnings are acceptable.)
    findings, _ = lint_paths([method_file])
    assert _errors(findings) == [], _errors(findings)


def test_notebook_magics_are_not_syntax_errors(tmp_path):
    # The 2026-06-10 pdwa draft: `%pip install -r requirements.txt` is valid
    # notebook syntax and must not read as a SyntaxError.
    f = tmp_path / "notebook_draft.py"
    f.write_text(
        "# %%\n"
        "%pip install -r requirements.txt\n"
        "# %%\n"
        "import numpy as np\n"
        "x = np.ones(3)\n"
    )
    findings, _ = lint_paths([f])
    assert _errors(findings) == [], findings


def test_magic_stripping_preserves_real_errors_and_line_numbers(tmp_path):
    pytest.importorskip("pyflakes")
    f = tmp_path / "notebook_draft.py"
    f.write_text("%pip install foo\n\ny = undefined_name\n")
    findings, _ = lint_paths([f])
    errs = _errors(findings)
    assert len(errs) == 1 and errs[0]["line"] == 3, errs


def test_render_placeholders_define_their_symbols(tmp_path):
    pytest.importorskip("pyflakes")
    # The draft's `{{params_dict}}` line becomes the cfg assignment only at
    # render time; linting the draft must not flag cfg as undefined.
    f = tmp_path / "notebook_draft.py"
    f.write_text(
        "# %%\n"
        "{{params_dict}}\n"
        "# %%\n"
        "print(cfg[\"learning_rate\"])\n"
    )
    findings, _ = lint_paths([f])
    assert _errors(findings) == [], findings


def test_renderer_comment_placeholder_form(tmp_path):
    pytest.importorskip("pyflakes")
    # The actual renderer form, confirmed on the real pdwa draft:
    # `# %% PLACEHOLDER: params_dict`.
    f = tmp_path / "notebook_draft.py"
    f.write_text(
        "# %% PLACEHOLDER: params_dict\n"
        "# %%\n"
        "print(cfg[\"dt\"])\n"
    )
    findings, _ = lint_paths([f])
    assert _errors(findings) == [], findings


def _reversed_argsort_errors(findings):
    return [f for f in _errors(findings) if "ReversedArgsort" in f["message"]]


def test_reversed_argsort_zoo_fixture_is_an_error():
    # The 2026-07-29 bayesian delivery's silent direction (RCA 2026-08-03,
    # finding 6): argsort OVER a reversed score array, executes cleanly,
    # scrambles the intended descending ranking. Must fire under any engine.
    method = ZOO / "bayesian-reversed-argsort-reconstructed" / "method.py"
    findings, _ = lint_paths([method])
    errs = _reversed_argsort_errors(findings)
    assert len(errs) == 1, findings
    src_lines = method.read_text(encoding="utf-8").splitlines()
    assert "np.argsort(bald_scores.cpu().numpy()[::-1])" in src_lines[errs[0]["line"] - 1]


def test_reversed_argsort_method_call_form_is_an_error(tmp_path):
    f = tmp_path / "method.py"
    f.write_text(
        "def rank(scores):\n"
        "    return scores[::-1].argsort()\n"
    )
    findings, _ = lint_paths([f])
    assert len(_reversed_argsort_errors(findings)) == 1, findings


def test_correct_descending_idioms_do_not_fire(tmp_path):
    # False-positive contract: the correct descending forms — including the
    # crash-direction fixture's `np.argsort(x)[::-1]`, whose failure is a
    # torch stride interaction, not the argsort operand — must never fire.
    f = tmp_path / "method.py"
    f.write_text(
        "import numpy as np\n"
        "def rank(scores):\n"
        "    a = np.argsort(scores)[::-1]\n"
        "    b = np.argsort(-scores)\n"
        "    c = scores[::-1]\n"
        "    return a, b, c\n"
    )
    findings, _ = lint_paths([f])
    assert _reversed_argsort_errors(findings) == [], findings

    negstride = ZOO / "gbald-negstride-selector" / "method.py"
    findings, _ = lint_paths([negstride])
    assert _reversed_argsort_errors(findings) == [], findings
