"""The adversarial golden test (partial-delivery design §3.5 criterion 7).

A detr-shaped partial run — one CORE component (the Hungarian matcher)
stubbed at fix-loop exhaustion, everything else built, contribution
evidence present — is pushed through the REAL surface renderers: the
delivery gate, the README banner, REPORT.md, the notebook render, the
end-of-run notice, the manifest. Then every §3.5 criterion is asserted
adversarially, cold-reader style: the FIRST sentence of every entry
surface must say PARTIAL or name the missing component, without
cross-referencing anything else.

This file gates the build: if a future change lets any surface open
without the partial state in its first line, this is the test that goes
red.
"""

from __future__ import annotations

import json
import re
import subprocess

import nbformat
import pytest

from partial_delivery import render_work_order, write_stub
from render_notebook import render as render_notebook
from render_run_report import write_run_report
from run_pipeline import (_render_end_of_run_notice, finalize_delivery_banner,
                          finalize_final_manifest, run_delivery_gating)
from tests.helpers.state import make_state

CORE_ID = "hungarian-matching"

_DRAFT = """\
# %% [markdown]
# # DETR-style Detection Demo

# %% [markdown]
# ## Setup

# %%
print("setup")

# %% PLACEHOLDER: component_stub:hungarian-matching

# %%
print("training loop")
"""


@pytest.fixture(scope="module")
def partial_run(tmp_path_factory, monkeypatch_module=None):
    """Build the whole partial package once, through the real code path."""
    tmp = tmp_path_factory.mktemp("golden")
    state = make_state(tmp / "run")
    run_dir = state.paths.run_dir
    pipeline = state.paths.pipeline_dir

    # The pipeline's own per-element knowledge: a four-element contract
    # with the matcher as core methodology.
    (pipeline / "method_spec.json").write_text(json.dumps({
        "comparison": {"classification": {"id": "detection.detr"}},
        "methodology_replication_contract": {"elements": [
            {"element_id": CORE_ID, "role": "core_methodology"},
            {"element_id": "set-prediction-loss", "role": "core_methodology"},
            {"element_id": "transformer-decoder", "role": "supporting_mechanism"},
            {"element_id": "training-schedule", "role": "paper_scale_detail"},
        ]}}), encoding="utf-8")
    # Contribution evidence exists — a complete package would have read
    # verified, which is exactly what makes this adversarial.
    (pipeline / "probe_report.json").write_text(json.dumps({
        "verdicts": [
            {"probe_id": "CT-1", "verdict": "pass", "message": "ok"},
            {"probe_id": "US-1", "verdict": "pass", "message": "ok"},
        ]}), encoding="utf-8")
    (pipeline / "params.json").write_text(
        json.dumps({"params": {}}), encoding="utf-8")
    (pipeline / "notebook_draft.py").write_text(_DRAFT, encoding="utf-8")
    (run_dir / "method").mkdir(parents=True, exist_ok=True)
    (run_dir / "method" / "README.md").write_text(
        "# DETR-style Detection Package\n\nUsage notes.\n", encoding="utf-8")

    # The stub, exactly as a producer would write it (§6 acceptance case 1:
    # paper quotes + the final fix-loop diagnosis in the work order).
    write_stub(
        run_dir,
        element_id=CORE_ID,
        role="core_methodology",
        stub_rel_path="method/matcher.py",
        work_order_markdown=render_work_order(
            element_id=CORE_ID,
            role="core_methodology",
            interface={
                "signature": 'def match(cost_matrix: "Tensor") -> "Tensor"',
                "arguments": ["cost_matrix: [B, N, M] float32"],
                "returns": "assignment indices [B, N] int64",
                "call_sites": ["method/method.py: loss assembly"],
            },
            paper_anchor={
                "section": "Section 3.1, Eq. 2",
                "quotes": ["we search for a permutation of N elements "
                           "with the lowest cost"],
            },
            why_not_built=("fix loop exhausted after 3 iterations against "
                           "this single component; final diagnosis attached"),
            verified_neighborhood=["set-prediction-loss: KD-1 pass",
                                   "transformer-decoder: import + contract "
                                   "dry-run pass"],
            fix_history="iteration 3: same failing cell, same target file",
        ),
        signatures=['def match(cost_matrix)'],
    )

    # The real delivery gate (battery subprocess faked; the report above is
    # what it reads).
    class _P:
        returncode = 0
        stdout = ""
        stderr = ""

    real_run = subprocess.run
    subprocess.run = lambda *a, **k: _P()
    try:
        delivery = run_delivery_gating(state)
    finally:
        subprocess.run = real_run

    finalize_delivery_banner(state, delivery)
    write_run_report(run_dir, delivery=delivery, stage_results=[])
    assert render_notebook(run_dir) == 0
    notice = _render_end_of_run_notice(
        state.paths, run_status="completed", manifest_status="degraded",
        delivery=delivery)
    finalize_final_manifest(state, delivery, stage_results=[])

    return {
        "run_dir": run_dir,
        "delivery": delivery,
        "readme": (run_dir / "method" / "README.md").read_text(encoding="utf-8"),
        "report": (run_dir / "REPORT.md").read_text(encoding="utf-8"),
        "notice": notice,
        "notebook": nbformat.read(run_dir / "notebook.ipynb", as_version=4),
        "manifest": json.loads(
            (run_dir / "details" / "final_manifest.json").read_text(encoding="utf-8")),
    }


def _first_sentence(text: str) -> str:
    # Skip lines with no words (a JSON block opens with a bare brace).
    first_line = next(l for l in text.splitlines()
                      if re.search(r"[A-Za-z]", l))
    return re.split(r"(?<=[.!?])\s", first_line.strip(), maxsplit=1)[0]


# ---------------------------------------------------------------------------
# Criterion 7 — the cold reader. The first sentence of every entry surface,
# read in isolation, must say partial or name the missing component.
# ---------------------------------------------------------------------------


def test_cold_reader_first_sentence_on_every_entry_surface(partial_run):
    delivery_block = json.dumps(partial_run["manifest"]["delivery"], indent=2)
    notebook_entry = partial_run["notebook"].cells[1].source
    stub_file = (partial_run["run_dir"] / "method" / "matcher.py").read_text(
        encoding="utf-8")
    work_order = (partial_run["run_dir"] / "work_orders" / f"{CORE_ID}.md"
                  ).read_text(encoding="utf-8")
    surfaces = {
        "README": partial_run["readme"],
        "REPORT": partial_run["report"],
        "end-of-run notice": partial_run["notice"],
        "manifest delivery block": delivery_block,
        "notebook banner": notebook_entry,
        "stub module": stub_file,
        "work order": work_order,
    }
    for name, text in surfaces.items():
        sentence = _first_sentence(text)
        assert ("partial" in sentence.lower() or CORE_ID in sentence), (
            f"cold reader on {name}: first sentence {sentence!r} does not "
            f"say partial or name the missing component")


# ---------------------------------------------------------------------------
# Criterion 1 — the word PARTIAL in the FIRST LINE of every entry surface.
# ---------------------------------------------------------------------------


def test_partial_in_first_line_of_every_entry_surface(partial_run):
    assert "PARTIAL" in partial_run["readme"].splitlines()[0]
    assert "PARTIAL" in partial_run["report"].splitlines()[0]
    assert "PARTIAL" in partial_run["notice"].splitlines()[0]
    # The manifest's delivery block: `partial` is the first rendered field.
    block_lines = json.dumps(partial_run["manifest"]["delivery"],
                             indent=2).splitlines()
    assert '"partial": true' in block_lines[1]
    assert "PARTIAL" in partial_run["notebook"].cells[1].source.splitlines()[0]


# ---------------------------------------------------------------------------
# Criterion 2 — numbers, not adjectives, then the stubs by name.
# ---------------------------------------------------------------------------


def test_completeness_numbers_lead_the_report(partial_run):
    report = partial_run["report"]
    m = re.search(r"3 of 4 components implemented, 1 stubbed", report)
    assert m, "REPORT.md lacks the N-of-M completeness statement"
    named = report.find(f"`{CORE_ID}`")
    assert 0 < named < report.find("Run: `"), (
        "the stub must be named before anything else about the package")


# ---------------------------------------------------------------------------
# Criterion 3 — the stub file self-identifies, docstring AND body.
# ---------------------------------------------------------------------------


def test_stub_file_self_identifies_when_opened_directly(partial_run):
    text = (partial_run["run_dir"] / "method" / "matcher.py").read_text(
        encoding="utf-8")
    docstring = text.split('"""')[1]
    assert "NOT IMPLEMENTED" in docstring
    assert "NotImplementedError" in text
    assert f"work_orders/{CORE_ID}.md" in text


# ---------------------------------------------------------------------------
# Criterion 4 — the notebook cannot run silently past the stub.
# ---------------------------------------------------------------------------


def test_notebook_states_and_raises_at_the_stub(partial_run):
    cells = partial_run["notebook"].cells
    md_idx = next(i for i, c in enumerate(cells)
                  if c.cell_type == "markdown" and "PARTIAL" in c.source
                  and CORE_ID in c.source and "raises" in c.source)
    raising = cells[md_idx + 1]
    assert raising.cell_type == "code"
    assert "raise NotImplementedError" in raising.source
    assert f"work_orders/{CORE_ID}.md" in raising.source
    # And validation would refuse a notebook that lost the pair.
    from validate_notebook_output import _partial_stub_errors
    ok = _partial_stub_errors(
        partial_run["run_dir"],
        [c.source for c in cells if c.cell_type == "markdown"],
        [c.source for c in cells if c.cell_type == "code"])
    assert ok == []
    broken = _partial_stub_errors(partial_run["run_dir"], ["# clean"],
                                  ["print('x')"])
    assert len(broken) == 2


# ---------------------------------------------------------------------------
# Criterion 5 — structured, not prose: the digest can count without parsing.
# ---------------------------------------------------------------------------


def test_manifest_field_is_structured(partial_run):
    d = partial_run["manifest"]["delivery"]
    assert d["partial"] is True
    assert d["stubbed_elements"] == [{
        "element_id": CORE_ID,
        "role": "core",
        "work_order": f"work_orders/{CORE_ID}.md",
        "stub_path": "method/matcher.py",
    }]
    assert any(r["source"] == "partial_delivery" for r in d["reasons"])


# ---------------------------------------------------------------------------
# Criterion 6 — nothing partial reads verified; a stubbed core forces the
# core-mechanism headline (§3.2: this package is a scaffold).
# ---------------------------------------------------------------------------


def test_nothing_partial_reads_verified_and_core_headline_fires(partial_run):
    # This run's probe sweep would certify a complete package (CT-1 pass,
    # zero demoters) — and the label still cannot be verified.
    assert partial_run["delivery"]["label"] == "draft"
    core_headline = f"core mechanism (`{CORE_ID}`) is NOT implemented"
    assert core_headline in partial_run["report"]
    assert core_headline in partial_run["readme"].splitlines()[0]

    from schemas.final_manifest import DeliveryVerdict
    with pytest.raises(ValueError):
        DeliveryVerdict.model_validate(
            {**partial_run["manifest"]["delivery"], "label": "verified"})


# ---------------------------------------------------------------------------
# The negative control: a complete run's surfaces stay quiet — the alarm
# only means something if it cannot false-fire.
# ---------------------------------------------------------------------------


def test_complete_run_surfaces_never_say_partial(tmp_path):
    state = make_state(tmp_path / "run")
    (state.paths.run_dir / "method").mkdir(parents=True, exist_ok=True)
    (state.paths.run_dir / "method" / "README.md").write_text("# Package\n",
                                                   encoding="utf-8")
    delivery = {"schema_version": "1.2.0", "partial": False,
                "label": "verified", "reasons": [], "disclosures": [],
                "probe_counts": {"pass": 5}, "stubbed_elements": []}
    finalize_delivery_banner(state, delivery)
    write_run_report(state.paths.run_dir, delivery=delivery,
                     stage_results=[])
    notice = _render_end_of_run_notice(
        state.paths, run_status="completed", manifest_status="passed",
        delivery=delivery)
    readme = (state.paths.run_dir / "method" / "README.md").read_text(encoding="utf-8")
    report = (state.paths.run_dir / "REPORT.md").read_text(encoding="utf-8")
    for text in (readme, report, notice):
        assert "PARTIAL" not in text
