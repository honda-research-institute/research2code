"""Tests for METHOD.md's equation-level code correspondence (Japan-feedback
note 5, scripts/generate_method_md.py).

Covers: the exact "implemented by" line format (primary function with file
location, extra anchor sites as "also used in", never a bare multi-function
list), the walkthrough's algorithm lines, the source map's implementation
column, the best-effort notebook cross-link riding note 4's association,
derived-block marking of everything line-number-bearing, the refresh cycle
(companion edit -> line numbers update, everything unmarked byte-identical),
and the no-regression property that a run without anchors generates
byte-identical output to the pre-correspondence renderer. An opt-in
integration test reads the live ms3d run, read-only.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

import generate_method_md as gmm
from generate_method_md import (
    IMPLEMENTED_BY_KIND,
    NOTEBOOK_NOTE_KIND,
    SOURCE_MAP_KIND,
    generate_method_md,
)
from refresh_derived_blocks import (
    BEGIN_MARKER_RE,
    DerivedBlockError,
    END_MARKER,
    refresh_run_documents,
)
from render_notebook import IMPLEMENTATION_NOTE_KIND, render
from tests.helpers.correspondence_run import (
    def_line,
    make_correspondence_run,
)

_KDE_LINE = def_line("def kde_mode_1d")
_KBF_LINE = def_line("def kbf_fuse")
_PIPELINE_LINE = def_line("def generate_pipeline")

_EXPECTED_EQ_KDE_LINE = (
    "**Implemented by:** [`kde_mode_1d`](method/method.py) "
    f"(`method/method.py:{_KDE_LINE}`) · also used in "
    f"`kbf_fuse` (`method/method.py:{_KBF_LINE}`), "
    f"`generate_pipeline` (`method/method.py:{_PIPELINE_LINE}`)"
)


def _strip_block_interiors(text: str) -> list[str]:
    """Every line OUTSIDE derived blocks (marker lines kept) — the refresh
    contract says only block interiors may change."""
    out: list[str] = []
    inside = False
    for line in text.split("\n"):
        if BEGIN_MARKER_RE.match(line):
            inside = True
            out.append(line)
        elif line == END_MARKER:
            inside = False
            out.append(line)
        elif not inside:
            out.append(line)
    return out


# ---------------------------------------------------------------------------
# The "implemented by" line
# ---------------------------------------------------------------------------


def test_equation_section_implemented_by_line_exact_format(tmp_path):
    run = make_correspondence_run(tmp_path, draft=False)
    md = generate_method_md(run)
    # No notebook delivered -> no cross-link promised; the line is exactly
    # the primary + also-used-in form.
    assert _EXPECTED_EQ_KDE_LINE in md
    assert "demonstrated in" not in md
    # It sits inside the eq-kde equation section, under the meta line.
    section = md.split('<a id="eq-kde"></a>')[1].split("###")[0]
    assert _EXPECTED_EQ_KDE_LINE in section
    assert section.index("role: implement") < section.index("Implemented by")
    assert section.index("Implemented by") < section.index("The paper states")


def test_implemented_by_is_a_marked_derived_block(tmp_path):
    run = make_correspondence_run(tmp_path, draft=False)
    lines = generate_method_md(run).split("\n")
    idx = lines.index(_EXPECTED_EQ_KDE_LINE)
    begin = BEGIN_MARKER_RE.match(lines[idx - 1])
    assert begin, "implemented-by line is not derived-block marked"
    header = json.loads(begin.group(1))
    assert header == {"kind": IMPLEMENTED_BY_KIND,
                      "spec": {"element_id": "eq-kde"}}
    assert lines[idx + 1] == END_MARKER


def test_walkthrough_algorithm_section_gains_implemented_by(tmp_path):
    run = make_correspondence_run(tmp_path, draft=False)
    md = generate_method_md(run)
    heading = '## <a id="algorithm-walkthrough"></a>Algorithm walkthrough'
    walkthrough = md.split(heading)[1].split("## Parameters")[0]
    assert ("**Implemented by:** [`generate_pipeline`](method/method.py) "
            f"(`method/method.py:{_PIPELINE_LINE}`)") in walkthrough


def test_explained_only_elements_render_as_today(tmp_path):
    run = make_correspondence_run(tmp_path, draft=False)
    md = generate_method_md(run)
    # Exactly two implemented-by lines: the anchored equation section and
    # the anchored algorithm section. Nothing claims correspondence for
    # concepts or for the unanchored element.
    assert md.count("**Implemented by:**") == 2
    unanchored_row = next(ln for ln in md.split("\n")
                          if '<a id="concept-unanchored"></a>' in ln)
    assert unanchored_row.endswith("| — |")


# ---------------------------------------------------------------------------
# The source map's implementation column
# ---------------------------------------------------------------------------


def test_source_map_gains_implementation_column(tmp_path):
    run = make_correspondence_run(tmp_path, draft=False)
    md = generate_method_md(run)
    assert "| element | type | paper location | role | implementation |" in md
    eq_row = next(ln for ln in md.split("\n")
                  if '<a id="eq-kde"></a>' in ln and ln.startswith("|"))
    assert f"| `kde_mode_1d` — `method/method.py:{_KDE_LINE}` |" in eq_row
    tracking_row = next(ln for ln in md.split("\n")
                        if '<a id="concept-tracking"></a>' in ln)
    assert f"`generate_pipeline` — `method/method.py:{_PIPELINE_LINE}`" \
        in tracking_row


def test_source_map_table_is_a_marked_derived_block(tmp_path):
    run = make_correspondence_run(tmp_path, draft=False)
    lines = generate_method_md(run).split("\n")
    header_idx = lines.index(
        "| element | type | paper location | role | implementation |")
    begin = BEGIN_MARKER_RE.match(lines[header_idx - 1])
    assert begin
    assert json.loads(begin.group(1)) == {"kind": SOURCE_MAP_KIND, "spec": {}}
    end_idx = next(i for i in range(header_idx, len(lines))
                   if lines[i] == END_MARKER)
    assert all(lines[i].startswith("|")
               for i in range(header_idx, end_idx))


# ---------------------------------------------------------------------------
# The notebook cross-link (best-effort, riding note 4)
# ---------------------------------------------------------------------------


def test_notebook_cross_link_appears_only_where_association_resolved(
        tmp_path):
    run = make_correspondence_run(tmp_path)
    assert render(run) == 0
    md = generate_method_md(run)
    # eq-kde's demo cell associated -> link with the section fragment.
    assert (_EXPECTED_EQ_KDE_LINE + " · demonstrated in "
            "[the notebook](notebook.ipynb#sec-42-the-kde-building-block)"
            ) in md
    # The orchestrator's algorithm id links its end-to-end section.
    assert "[the notebook](notebook.ipynb#sec-5-end-to-end)" in md
    # concept-refine_a never associated (overview cell skipped) -> its
    # source-map presence promises nothing about the notebook.
    assert md.count("demonstrated in [the notebook]") == 2


def test_refresh_adds_the_notebook_link_after_delivery(tmp_path):
    # METHOD.md is generated at stage 1.x, before any notebook exists; the
    # cross-link arrives via the shared refresh once note 4's association
    # is in the delivered notebook.
    run = make_correspondence_run(tmp_path)
    before = generate_method_md(run)
    assert "demonstrated in" not in before
    (run / "METHOD.md").write_text(before, encoding="utf-8")

    assert render(run) == 0
    seen, changed = refresh_run_documents(run, ["METHOD.md"])["METHOD.md"]
    assert seen == 3  # eq-kde + alg-pipeline lines, source map table
    assert changed == 2  # both implemented-by lines gain their link

    after = (run / "METHOD.md").read_text(encoding="utf-8")
    assert "demonstrated in [the notebook](notebook.ipynb#sec-42" in after
    # Only block interiors changed.
    assert _strip_block_interiors(before) == _strip_block_interiors(after)


# ---------------------------------------------------------------------------
# Refresh cycle: companion edits, honest failures, idempotence
# ---------------------------------------------------------------------------


def test_companion_edit_refreshes_line_numbers_only(tmp_path):
    run = make_correspondence_run(tmp_path, draft=False)
    before = generate_method_md(run)
    (run / "METHOD.md").write_text(before, encoding="utf-8")
    assert refresh_run_documents(run, ["METHOD.md"]) == {"METHOD.md": (3, 0)}

    # The companion prepends a line — every def shifts by one.
    method_py = run / "method" / "method.py"
    method_py.write_text("# shifted by a companion edit\n"
                         + method_py.read_text(encoding="utf-8"),
                         encoding="utf-8")
    assert refresh_run_documents(run, ["METHOD.md"]) == {"METHOD.md": (3, 3)}

    after = (run / "METHOD.md").read_text(encoding="utf-8")
    assert f"(`method/method.py:{_KDE_LINE + 1}`)" in after
    assert f"| `kde_mode_1d` — `method/method.py:{_KDE_LINE + 1}` |" in after
    assert _strip_block_interiors(before) == _strip_block_interiors(after)
    # Idempotent from there.
    assert refresh_run_documents(run, ["METHOD.md"]) == {"METHOD.md": (3, 0)}


def test_removed_anchors_fail_the_refresh_honestly(tmp_path):
    run = make_correspondence_run(tmp_path, draft=False)
    (run / "METHOD.md").write_text(generate_method_md(run), encoding="utf-8")
    stripped = "\n".join(
        line for line in
        (run / "method" / "method.py").read_text(encoding="utf-8").split("\n")
        if "# paper-element:" not in line)
    (run / "method" / "method.py").write_text(stripped, encoding="utf-8")
    with pytest.raises(DerivedBlockError, match="eq-kde"):
        refresh_run_documents(run, ["METHOD.md"])


def test_refresh_cli_serves_the_renderer_registered_kinds(tmp_path):
    # A fresh interpreter (no producer pre-imported) must still refresh
    # implemented_by/source_map/implementation_note blocks — the engine's
    # lazy producer import, exercised end to end.
    run = make_correspondence_run(tmp_path)
    (run / "METHOD.md").write_text(generate_method_md(run), encoding="utf-8")
    assert render(run) == 0
    proc = subprocess.run(
        [sys.executable, "scripts/refresh_derived_blocks.py",
         "--run-dir", str(run)],
        capture_output=True, text=True,
        cwd=Path(__file__).resolve().parent.parent)
    assert proc.returncode == 0, proc.stderr
    # The pre-notebook METHOD.md gains its cross-links; the notebook's own
    # blocks are already fresh from birth.
    assert "METHOD.md: 3 derived block(s), 2 rewritten" in proc.stdout
    assert "notebook.ipynb: 3 derived block(s), 0 rewritten" in proc.stdout


# ---------------------------------------------------------------------------
# The no-regression property: no anchors, byte-identical output
# ---------------------------------------------------------------------------


def test_anchor_free_run_generates_byte_identical_output(tmp_path,
                                                         monkeypatch):
    run = make_correspondence_run(tmp_path, draft=False, anchors=False)
    md = generate_method_md(run)

    # Nothing correspondence-shaped leaked in...
    assert "Implemented by" not in md
    assert "| implementation |" not in md
    assert "derived-block" not in md
    assert "| element | type | paper location | role |" in md

    # ...and the output is byte-identical to a generation where the
    # correspondence pass does not exist at all.
    monkeypatch.setattr(gmm, "load_join_table_elements", lambda run_dir: {})
    assert generate_method_md(run) == md


def test_no_method_dir_run_is_also_unchanged(tmp_path):
    # The survives-any-downstream-failure shape: paper map only.
    import shutil
    run = make_correspondence_run(tmp_path, draft=False)
    shutil.rmtree(run / "method")
    md = generate_method_md(run)
    assert "Implemented by" not in md
    assert "derived-block" not in md


def test_determinism_with_correspondence(tmp_path):
    run = make_correspondence_run(tmp_path, draft=False)
    assert generate_method_md(run) == generate_method_md(run)


def test_broken_anchor_ids_degrade_with_warning_not_failure(tmp_path,
                                                            capsys):
    # An anchor id the paper map does not know: the coder gates own that
    # failure; the document renderer degrades to the pre-correspondence
    # output with a warning.
    run = make_correspondence_run(tmp_path, draft=False)
    method_py = run / "method" / "method.py"
    method_py.write_text(
        method_py.read_text(encoding="utf-8").replace(
            "# paper-element: eq-kde", "# paper-element: eq-nonexistent"),
        encoding="utf-8")
    md = generate_method_md(run)
    assert "Implemented by" not in md
    assert "eq-nonexistent" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# Cross-module consistency
# ---------------------------------------------------------------------------


def test_notebook_note_kind_literal_matches_the_renderer():
    # generate_method_md repeats the kind name so it never imports the
    # nbformat-heavy notebook renderer; this pin keeps the literal honest.
    assert NOTEBOOK_NOTE_KIND == IMPLEMENTATION_NOTE_KIND


def test_artifact_is_preferred_over_in_process_build(tmp_path):
    # When the driver (or a test) materialized the artifact, the loader
    # reads it instead of re-parsing the package.
    from build_anchor_join_table import write_join_table
    run = make_correspondence_run(tmp_path, draft=False)
    write_join_table(run)
    # Poison the package: an artifact-first loader must not notice.
    (run / "method" / "method.py").write_text("def broken(:\n",
                                              encoding="utf-8")
    elements = gmm.load_join_table_elements(run)
    assert "eq-kde" in elements


# ---------------------------------------------------------------------------
# Opt-in integration against the live ms3d run (pure read: the generator
# returns a string; nothing is written into the run dir).
# ---------------------------------------------------------------------------

_REAL_MS3D = Path(__file__).resolve().parent.parent / "r2c_runs" / "ms3d"


@pytest.mark.manual_only
@pytest.mark.skipif(
    os.environ.get("R2C_LIVE_RUN_TESTS") != "1" or not _REAL_MS3D.is_dir(),
    reason="integration against a real green run dir — opt in with "
           "R2C_LIVE_RUN_TESTS=1 (needs r2c_runs/ms3d present)",
)
def test_integration_real_ms3d_method_md():
    md = generate_method_md(_REAL_MS3D)
    # The design review's validated case: eq-kde is primarily implemented
    # by the dedicated KDE function, with the fusion + orchestrator sites
    # as also-used-in.
    eq_kde = md.split('<a id="eq-kde"></a>')[1].split("###")[0]
    assert "**Implemented by:** [`kde_mode_1d`](method/method.py)" in eq_kde
    assert "also used in" in eq_kde
    assert "`kbf_fuse`" in eq_kde
    # The correspondence table is complete: implementation column present,
    # and the un-anchored VMFI concept stays honestly empty.
    assert "| element | type | paper location | role | implementation |" \
        in md
    vmfi_row = next(ln for ln in md.split("\n")
                    if '<a id="concept-vmfi"></a>' in ln)
    assert vmfi_row.endswith("| — |")
    # Every implemented-by line is derived-block marked.
    lines = md.split("\n")
    for i, line in enumerate(lines):
        if line.startswith("**Implemented by:**"):
            assert BEGIN_MARKER_RE.match(lines[i - 1])
