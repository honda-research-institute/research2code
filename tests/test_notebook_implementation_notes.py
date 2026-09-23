"""Tests for the notebook implementation notes (Japan-feedback note 4,
scripts/render_notebook.py).

Covers: the derived cell-to-element association on real call patterns
(magic-bearing setup cell included), the cap-to-primary rule on an
orchestrator-heavy cell, the overview-cell skip, one-note-per-function
dedupe, the ~60-line size cap, link resolution, the refresh cycle over
injected blocks, and the no-regression property that a run without anchors
renders exactly as before the feature existed. An opt-in integration test
exercises the association against the live ms3d run dir, read-only.
"""

from __future__ import annotations

import json
import os
import re
import shutil
from pathlib import Path

import pytest

import render_notebook
from generate_method_md import generate_method_md
from evaluation_protocol_rendering import render_evaluation_protocol_block
from schemas.method_spec import EvaluationProtocol
from schemas.params import Params
from refresh_derived_blocks import BEGIN_MARKER_RE, refresh_run_documents
from render_notebook import (
    IMPLEMENTATION_NOTE_KIND,
    _inject_implementation_notes,
    render,
)
from validate_notebook_output import _evaluation_protocol_render_errors
from tests.helpers.correspondence_run import (
    make_correspondence_run,
)


ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _read_nb(run: Path) -> dict:
    return json.loads((run / "notebook.ipynb").read_text(encoding="utf-8"))


def _sources(nb: dict) -> list[tuple[str, str]]:
    out = []
    for cell in nb["cells"]:
        src = cell["source"]
        if isinstance(src, list):
            src = "".join(src)
        out.append((cell["cell_type"], src))
    return out


def _markers(source: str) -> list[dict]:
    """Parsed begin-marker headers in one markdown source."""
    headers = []
    for line in source.split("\n"):
        m = BEGIN_MARKER_RE.match(line)
        if m:
            headers.append(json.loads(m.group(1)))
    return headers


def _note_before(nb: dict, code_needle: str) -> str:
    """The markdown source immediately above the code cell containing
    `code_needle` — where note 4 injects the implementation note."""
    cells = _sources(nb)
    idx = next(i for i, (kind, src) in enumerate(cells)
               if kind == "code" and code_needle in src)
    kind, src = cells[idx - 1]
    assert kind == "markdown", f"no markdown above the {code_needle!r} cell"
    return src


def _rendered(tmp_path: Path, **kwargs) -> Path:
    run = make_correspondence_run(tmp_path, **kwargs)
    assert render(run) == 0
    return run


def _normalized(nb: dict) -> dict:
    """Notebook JSON with cell ids replaced by their index — nbformat mints
    RANDOM cell ids, so byte-level comparison across two renders was never a
    property of this surface, before or after this feature."""
    nb = json.loads(json.dumps(nb))
    for i, cell in enumerate(nb.get("cells", [])):
        cell["id"] = str(i)
    return nb


def _make_protocol_notebook_run(tmp_path: Path, *, typed: bool) -> Path:
    run = tmp_path / "run"
    pipeline = run / ".pipeline"
    pipeline.mkdir(parents=True)
    (pipeline / "notebook_draft.py").write_text(
        """\
# %% [markdown]
# # Protocol notebook

# %% [markdown] PLACEHOLDER: params_table

# %% PLACEHOLDER: params_dict
""",
        encoding="utf-8",
    )
    params_payload = {
        "params": {
            "forecast_horizon": {
                "value": 4,
                "source": "system_inferred",
                "reasoning": "System-owned demo choice.",
            },
        },
    }
    comparison = {}
    if typed:
        comparison["evaluation_protocol"] = {
            "scheme": {
                "kind": "rolling_origin",
                "paper_value_status": "paper_stated",
                "description": "Forecast origins advance through the test interval.",
                "evidence_quote": "Forecast origins advance through the test interval.",
                "paper_section": "Evaluation protocol",
                "paper_element_ids": ["protocol-scheme"],
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
                    "axis_paper_element_ids": ["protocol-axis"],
                    "paper_value_status": "paper_stated",
                    "evidence_quote": "The context length is 10 weeks.",
                    "paper_section": "Method inputs",
                    "paper_element_ids": ["protocol-context"],
                },
                {
                    "role": "forecast_call_horizon",
                    "parameter_name": "forecast_horizon",
                    "paper_names": ["forecast horizon"],
                    "paper_symbols": ["K"],
                    "value": None,
                    "unit": "week",
                    "granularity": 1,
                    "axis_evidence_quote": (
                        "Weekly demand observations define the target time series."
                    ),
                    "axis_paper_section": "Data cadence",
                    "axis_paper_element_ids": ["protocol-axis"],
                    "paper_value_status": "paper_unspecified",
                    "evidence_quote": (
                        "The forecast horizon K denotes the future forecast steps."
                    ),
                    "paper_section": "Forecast implementation",
                    "paper_element_ids": ["protocol-horizon"],
                },
                {
                    "role": "validation_span",
                    "parameter_name": None,
                    "paper_names": ["validation span"],
                    "paper_symbols": [],
                    "value": 13,
                    "unit": "week",
                    "granularity": 1,
                    "axis_evidence_quote": (
                        "Weekly demand observations define the target time series."
                    ),
                    "axis_paper_section": "Data cadence",
                    "axis_paper_element_ids": ["protocol-axis"],
                    "paper_value_status": "paper_stated",
                    "evidence_quote": "The validation span is 13 weeks.",
                    "paper_section": "Evaluation splits",
                    "paper_element_ids": ["protocol-validation"],
                },
                {
                    "role": "test_span",
                    "parameter_name": None,
                    "paper_names": ["test span"],
                    "paper_symbols": [],
                    "value": 26,
                    "unit": "week",
                    "granularity": 1,
                    "axis_evidence_quote": (
                        "Weekly demand observations define the target time series."
                    ),
                    "axis_paper_section": "Data cadence",
                    "axis_paper_element_ids": ["protocol-axis"],
                    "paper_value_status": "paper_stated",
                    "evidence_quote": "The test span is 26 weeks.",
                    "paper_section": "Evaluation splits",
                    "paper_element_ids": ["protocol-test"],
                },
            ],
        }
        params_payload["params"] = {
            "context_length": {
                "value": 10,
                "source": "paper",
                "paper_section": "Method inputs",
                "paper_says": "The context length is 10 weeks.",
                "protocol_role": "context_length",
                "protocol_value": 10,
                "protocol_unit": "week",
                "protocol_granularity": 1,
                "protocol_axis_says": (
                    "Weekly demand observations define the target time series."
                ),
                "protocol_axis_section": "Data cadence",
                "protocol_axis_element_ids": ["protocol-axis"],
                "paper_value_status": "paper_stated",
                "paper_element_ids": ["protocol-context"],
            },
            "forecast_horizon": {
                "value": 4,
                "source": "system_inferred",
                "reasoning": "System-owned demo choice.",
                "paper_section": "Forecast implementation",
                "paper_says": (
                    "The forecast horizon K denotes the future forecast steps."
                ),
                "protocol_role": "forecast_call_horizon",
                "protocol_value": None,
                "protocol_unit": "week",
                "protocol_granularity": 1,
                "protocol_axis_says": (
                    "Weekly demand observations define the target time series."
                ),
                "protocol_axis_section": "Data cadence",
                "protocol_axis_element_ids": ["protocol-axis"],
                "paper_value_status": "paper_unspecified",
                "paper_element_ids": ["protocol-horizon"],
            },
        }
        EvaluationProtocol.model_validate(comparison["evaluation_protocol"])
        Params.model_validate(params_payload)
    (pipeline / "params.json").write_text(
        json.dumps(params_payload),
        encoding="utf-8",
    )
    (pipeline / "method_spec.json").write_text(
        json.dumps({"comparison": comparison}), encoding="utf-8"
    )
    return run


# ---------------------------------------------------------------------------
# Association on the fixture draft's real call patterns
# ---------------------------------------------------------------------------


def test_kde_cell_gains_note_with_collapsed_source(tmp_path):
    run = _rendered(tmp_path)
    note = _note_before(_read_nb(run), "kde_mode_1d([1.0, 2.0]")
    headers = _markers(note)
    assert len(headers) == 1
    assert headers[0]["kind"] == IMPLEMENTATION_NOTE_KIND
    assert headers[0]["spec"] == {
        "element_ids": ["eq-kde"],
        "file": "method/method.py",
        "qualname": "kde_mode_1d",
    }
    # Function name, method.py location, METHOD.md link, collapsed source.
    assert "**How this is implemented:** [`kde_mode_1d`](method/method.py)" \
        in note
    assert "[Weighted KDE mode](METHOD.md#eq-kde)" in note
    assert "<details>" in note and "</details>" in note
    assert "def kde_mode_1d(values, weights):" in note
    # The demo cell still calls the imported function — the listing sits
    # beside it, nothing was inlined into the code cell.
    code = next(src for kind, src in _sources(_read_nb(run))
                if kind == "code" and "kde_mode_1d([1.0, 2.0]" in src)
    assert "def kde_mode_1d" not in code


def test_association_uses_imports_from_the_magic_bearing_setup_cell(tmp_path):
    # The setup cell opens with `%matplotlib inline`; if magics broke the
    # AST pass, no imports would resolve and nothing would associate.
    run = _rendered(tmp_path)
    nb_text = json.dumps(_read_nb(run))
    assert IMPLEMENTATION_NOTE_KIND in nb_text


def test_fusion_cell_caps_to_primary_and_dedupes(tmp_path):
    run = _rendered(tmp_path)
    nb = _read_nb(run)
    note = _note_before(nb, "kbf_fuse([[1.0], [2.0]])")
    headers = _markers(note)
    # The cell calls kbf_fuse AND kde_mode_1d, but the kde note already
    # exists at its own demo cell — one note per implementing function.
    assert [h["spec"]["qualname"] for h in headers] == ["kbf_fuse"]
    # eq-kde is anchored in kbf_fuse too, but its PRIMARY site is
    # kde_mode_1d, so the kbf note claims only concept-kbf.
    assert headers[0]["spec"]["element_ids"] == ["concept-kbf"]
    assert "METHOD.md#eq-kde" not in note
    # Dedupe across the whole notebook: exactly one kde_mode_1d note.
    all_specs = [h["spec"]["qualname"]
                 for _, src in _sources(nb) for h in _markers(src)]
    assert all_specs.count("kde_mode_1d") == 1


def test_unanchored_helper_cell_gets_no_note(tmp_path):
    run = _rendered(tmp_path)
    nb = _read_nb(run)
    cells = _sources(nb)
    idx = next(i for i, (kind, src) in enumerate(cells)
               if kind == "code" and "plot_helper(fused)" in src)
    # The cell above it is the fusion CODE cell — nothing was injected.
    assert cells[idx - 1][0] == "code"
    assert not any("plot_helper" in h["spec"]["qualname"]
                   for _, src in cells for h in _markers(src))


def test_overview_cell_resolving_many_functions_skips_entirely(tmp_path):
    # The overview cell calls four implementing functions — more than the
    # per-cell cap — so it is an overview, not a component demo: no notes,
    # and the refinement functions are annotated nowhere in the notebook.
    run = _rendered(tmp_path)
    cells = _sources(_read_nb(run))
    qualnames = [h["spec"]["qualname"]
                 for _, src in cells for h in _markers(src)]
    assert "refine_a" not in qualnames
    assert "refine_b" not in qualnames
    notes = "\n".join(src for _, src in cells if _markers(src))
    assert "concept-refine_a" not in notes
    assert "concept-refine_b" not in notes


def test_orchestrator_cell_caps_to_primary_ids_and_size_cap(tmp_path):
    run = _rendered(tmp_path)
    note = _note_before(_read_nb(run), "generate_pipeline([[1.0]])")
    headers = _markers(note)
    assert len(headers) == 1
    spec = headers[0]["spec"]
    assert spec["qualname"] == "generate_pipeline"
    # The orchestrator touches eq-kde and concept-kbf at call sites, but
    # only the ids that LIVE in it (its algorithm id and the call-site-only
    # tracking id) are claimed — the cap-to-primary rule.
    assert spec["element_ids"] == ["alg-pipeline", "concept-tracking"]
    assert "METHOD.md#alg-pipeline" in note
    assert "METHOD.md#concept-tracking" in note
    assert "METHOD.md#concept-kbf" not in note
    assert "METHOD.md#eq-kde" not in note
    # >60 lines: signature and docstring only, with the file link.
    assert "too long to inline" in note
    assert "Signature and docstring:" in note
    assert "def generate_pipeline(data):" in note
    assert "step_10 = 10" not in note
    assert "<details>" not in note


def test_links_resolve(tmp_path):
    run = _rendered(tmp_path)
    notes = [src for _, src in _sources(_read_nb(run)) if _markers(src)]
    assert notes
    method_md = generate_method_md(run)
    for note in notes:
        for target in re.findall(r"\]\(([^)]+)\)", note):
            if target.startswith("METHOD.md#"):
                element_id = target.split("#", 1)[1]
                assert f'<a id="{element_id}"></a>' in method_md
            else:
                assert (run / target).is_file(), f"dead link: {target}"


# ---------------------------------------------------------------------------
# Refresh cycle over the injected blocks
# ---------------------------------------------------------------------------


def test_fresh_render_is_idempotent_under_refresh(tmp_path):
    run = _rendered(tmp_path)
    results = refresh_run_documents(run, ["notebook.ipynb"])
    seen, changed = results["notebook.ipynb"]
    assert seen == 3  # kde, kbf, orchestrator
    assert changed == 0


def test_companion_edit_then_refresh_updates_only_the_listing(tmp_path):
    run = _rendered(tmp_path)
    before = _read_nb(run)

    method_py = run / "method" / "method.py"
    method_py.write_text(
        method_py.read_text(encoding="utf-8").replace(
            "return values[0]", "return values[-1]"),
        encoding="utf-8")

    results = refresh_run_documents(run, ["notebook.ipynb"])
    assert results["notebook.ipynb"] == (3, 1)

    after = _read_nb(run)
    assert len(after["cells"]) == len(before["cells"])
    changed = [i for i, (b, a) in enumerate(zip(before["cells"],
                                                after["cells"])) if b != a]
    assert len(changed) == 1
    src = after["cells"][changed[0]]["source"]
    src = "".join(src) if isinstance(src, list) else src
    assert "return values[-1]" in src
    assert "return values[0]" not in src

    # And the refresh is idempotent from there.
    assert refresh_run_documents(run, ["notebook.ipynb"]) == {
        "notebook.ipynb": (3, 0)}


# ---------------------------------------------------------------------------
# The no-regression property: no anchors, no change
# ---------------------------------------------------------------------------


def test_anchor_free_run_renders_identically(tmp_path, monkeypatch):
    # Same fixture minus every anchor comment: render with the feature
    # active, then with the injection pass stubbed to the pre-change
    # identity, and compare the notebooks (cell ids normalized — nbformat
    # mints random ids, so byte identity across two renders was never a
    # property of this surface).
    run_a = _rendered(tmp_path / "a", anchors=False)
    with_feature = _read_nb(run_a)
    assert IMPLEMENTATION_NOTE_KIND not in json.dumps(with_feature)

    monkeypatch.setattr(render_notebook, "_inject_implementation_notes",
                        lambda cells, run_dir: cells)
    run_b = _rendered(tmp_path / "b", anchors=False)
    without_feature = _read_nb(run_b)

    assert _normalized(with_feature) == _normalized(without_feature)


def test_injection_pass_is_the_identity_without_correspondence(tmp_path):
    # The strongest form of the property: the pass hands back the very same
    # cells object when the run offers no correspondence.
    run = make_correspondence_run(tmp_path, anchors=False)
    cells: list = []
    assert _inject_implementation_notes(cells, run) is cells

    shutil.rmtree(run / "method")
    assert _inject_implementation_notes(cells, run) is cells


def test_paper_map_only_run_renders_without_notes(tmp_path):
    # The METHOD.md-survives-anything shape: draft + params, no method/.
    run = make_correspondence_run(tmp_path)
    shutil.rmtree(run / "method")
    assert render(run) == 0
    assert IMPLEMENTATION_NOTE_KIND not in json.dumps(_read_nb(run))


def test_notebook_renderer_injects_role_separated_protocol_after_params(tmp_path):
    """Synthetic rolling-origin renderer control; not second-paper evidence."""
    run = _make_protocol_notebook_run(tmp_path, typed=True)

    assert render(run) == 0

    sources = _sources(_read_nb(run))
    protocol_index = next(
        i for i, (kind, source) in enumerate(sources)
        if kind == "markdown" and "### Evaluation protocol" in source
    )
    params_index = next(
        i for i, (kind, source) in enumerate(sources)
        if kind == "markdown" and "| Parameter | Variable from paper |" in source
    )
    assert protocol_index == params_index + 1
    protocol = sources[protocol_index][1]
    assert "**Paper evaluation scheme:** **rolling origin**" in protocol
    assert "paper IDs: `protocol-scheme`" in protocol
    assert "| role/value evidence | axis evidence |" in protocol
    assert "Context length** | 10 (paper-stated)" in protocol
    assert "One-call forecast horizon** | **paper-unspecified**" in protocol
    assert "`forecast_horizon` = 4 (system inferred)" in protocol
    assert (
        "Forecast implementation; paper IDs: `protocol-horizon` | "
        "Data cadence; paper IDs: `protocol-axis` |"
    ) in protocol
    assert "Validation span** | 13 (paper-stated)" in protocol
    assert "Test span** | 26 (paper-stated)" in protocol
    assert protocol.count("spec-only fact; no runtime parameter") == 2


def test_notebook_renderer_is_protocol_noop_for_legacy_spec(tmp_path):
    run = _make_protocol_notebook_run(tmp_path, typed=False)

    assert render(run) == 0

    assert "Evaluation protocol" not in json.dumps(_read_nb(run))


def test_notebook_validator_requires_exact_typed_protocol_block(tmp_path):
    run = _make_protocol_notebook_run(tmp_path, typed=True)
    assert render(run) == 0
    spec = json.loads(
        (run / ".pipeline" / "method_spec.json").read_text(encoding="utf-8")
    )
    params = json.loads(
        (run / ".pipeline" / "params.json").read_text(encoding="utf-8")
    )
    markdown = [
        source for kind, source in _sources(_read_nb(run)) if kind == "markdown"
    ]

    assert _evaluation_protocol_render_errors(spec, params, markdown) == []
    altered = [
        source.replace("paper-unspecified", "paper-stated")
        if "### Evaluation protocol" in source
        else source
        for source in markdown
    ]
    errors = _evaluation_protocol_render_errors(spec, params, altered)
    assert len(errors) == 1
    assert "role-separated" in errors[0]

    competing = [
        *markdown,
        "### Evaluation protocol\n\nOne-call forecast horizon K = 26 weeks.",
    ]
    errors = _evaluation_protocol_render_errors(spec, params, competing)
    assert len(errors) == 2
    assert any("competing hand-written evaluation-protocol" in item for item in errors)
    assert any("competing numeric claim" in item for item in errors)
    assert any("cell(s)" in item for item in errors)


@pytest.mark.parametrize(
    "claim",
    [
        "The paper reports forecast horizon K = 4 weeks.",
        "Demo forecast horizon K = 12 weeks.",
        "One-call forecast horizon K =\n26 weeks.",
        (
            "According to the paper, forecast horizon K is configured after "
            "all preprocessing and batching decisions have been applied to "
            "the weekly target series at 26 weeks."
        ),
    ],
)
def test_notebook_protocol_narratives_reject_wrong_attributed_values(
    tmp_path,
    claim,
):
    run = _make_protocol_notebook_run(tmp_path, typed=True)
    spec = json.loads(
        (run / ".pipeline" / "method_spec.json").read_text(encoding="utf-8")
    )
    params = json.loads(
        (run / ".pipeline" / "params.json").read_text(encoding="utf-8")
    )
    canonical = render_evaluation_protocol_block(
        spec, params, heading="### Evaluation protocol"
    )

    errors = _evaluation_protocol_render_errors(
        spec, params, [canonical, claim]
    )
    assert len(errors) == 1
    assert "competing numeric claim" in errors[0]


def test_notebook_protocol_narratives_allow_correct_role_attribution(tmp_path):
    run = _make_protocol_notebook_run(tmp_path, typed=True)
    spec = json.loads(
        (run / ".pipeline" / "method_spec.json").read_text(encoding="utf-8")
    )
    params = json.loads(
        (run / ".pipeline" / "params.json").read_text(encoding="utf-8")
    )
    canonical = render_evaluation_protocol_block(
        spec, params, heading="### Evaluation protocol"
    )
    assert _evaluation_protocol_render_errors(
        spec,
        params,
        [canonical, "This notebook uses demo forecast horizon K = 4 weeks."],
    ) == []

    horizon = next(
        quantity
        for quantity in spec["comparison"]["evaluation_protocol"]["quantities"]
        if quantity["role"] == "forecast_call_horizon"
    )
    horizon.update({
        "value": 12,
        "paper_value_status": "paper_stated",
        "evidence_quote": "The forecast horizon K = 12 future weeks.",
    })
    params["params"]["forecast_horizon"].update({
        "source": "system_default",
        "paper_value": 12,
        "paper_value_status": "paper_stated",
        "paper_says": "The forecast horizon K = 12 future weeks.",
    })
    canonical = render_evaluation_protocol_block(
        spec, params, heading="### Evaluation protocol"
    )
    assert _evaluation_protocol_render_errors(
        spec,
        params,
        [canonical, "The paper reports forecast horizon K = 12 weeks."],
    ) == []


def test_notebook_agent_keeps_protocol_roles_and_provenance_separate():
    guidance = (
        ROOT / ".opencode" / "agents" / "r2c-notebook-generator.md"
    ).read_text(encoding="utf-8")

    assert "comparison.evaluation_protocol" in guidance
    assert "T+1` does not mean `K=1" in guidance
    assert "26-week test" in guidance
    assert "does not mean `K=26" in guidance
    assert "Do not hand-write a second" in guidance


# ---------------------------------------------------------------------------
# Opt-in integration against the live ms3d run (read-only on the run dir:
# inputs are copied into tmp, the render writes only there).
# ---------------------------------------------------------------------------

_REAL_MS3D = Path(__file__).resolve().parent.parent / "r2c_runs" / "ms3d"


@pytest.mark.manual_only
@pytest.mark.skipif(
    os.environ.get("R2C_LIVE_RUN_TESTS") != "1" or not _REAL_MS3D.is_dir(),
    reason="integration against a real green run dir — opt in with "
           "R2C_LIVE_RUN_TESTS=1 (needs r2c_runs/ms3d present)",
)
def test_integration_real_ms3d_render(tmp_path):
    run = tmp_path / "ms3d"
    (run / ".pipeline").mkdir(parents=True)
    shutil.copytree(_REAL_MS3D / "method", run / "method",
                    ignore=shutil.ignore_patterns("__pycache__"))
    for name in ("notebook_draft.py", "params.json", "paper_map.json",
                 "stubbed_elements.json"):
        src = _REAL_MS3D / ".pipeline" / name
        if src.is_file():
            shutil.copy(src, run / ".pipeline" / name)

    assert render(run) == 0
    nb = _read_nb(run)
    specs = [h["spec"] for _, src in _sources(nb) for h in _markers(src)]
    by_qualname = {s["qualname"]: s for s in specs}

    # The per-component demo cells associate with their dedicated
    # implementations.
    assert "kde_mode_1d" in by_qualname
    assert by_qualname["kde_mode_1d"]["element_ids"] == ["eq-kde"]
    assert "kbf_fuse" in by_qualname
    assert by_qualname["kbf_fuse"]["element_ids"] == ["concept-kbf"]

    # The end-to-end cell reaches the orchestrator (which touches many ids
    # at call sites) and caps to the ids that live in it.
    assert "generate_pseudo_labels" in by_qualname
    assert "concept-kbf" not in \
        by_qualname["generate_pseudo_labels"]["element_ids"]
    assert "eq-kde" not in \
        by_qualname["generate_pseudo_labels"]["element_ids"]

    # A healthy spread of annotated components, and every injected block
    # is refresh-idempotent from birth.
    assert len(specs) >= 5
    seen, changed = refresh_run_documents(run, ["notebook.ipynb"])[
        "notebook.ipynb"]
    assert seen == len(specs)
    assert changed == 0
