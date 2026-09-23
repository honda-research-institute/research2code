"""R2C-067 — a degraded delivery tells one story on every surface.

The 2026-08-05 delivery disclosed its crash in REPORT.md's trouble table
while the notebook promised "a runnable tutorial that trains GraphDeepAR"
and the package README said to click Run All to verify. The surfaces a
researcher opens first were the optimistic ones, because they are generated
hours before the smoke gate has an opinion (`surface_story_drift`).

All three fixes read state the run already recorded, so the known-good case
is a clean run whose surfaces must come out byte-identical.
"""

from __future__ import annotations

import json
from pathlib import Path

from tests.helpers.state import make_state


_README = """# Method package

## Running the notebook

Click **Run All**. First run loads the smoke environment from
`method/example_data/`; the full notebook takes about a minute.

## Layout
"""

_NOTEBOOK = {
    "cells": [
        {"cell_type": "markdown", "metadata": {},
         "source": ["# GraphDeepAR: a runnable tutorial\n"]},
        {"cell_type": "code", "metadata": {}, "source": ["import method\n"],
         "outputs": [], "execution_count": None},
    ],
    "metadata": {}, "nbformat": 4, "nbformat_minor": 5,
}

_KNOWN_ISSUES = """# Known issues

## stage_3c — the notebook does not run end-to-end

**Where.** notebook.ipynb — cell 29 (section 6)

**What to do.** Hand the notebook and this file to an AI assistant.

**Technical detail.** smoke gate failed after cap=3.

---
"""

_FALLBACK_DIAGNOSIS = {
    "schema_version": "1.0.0",
    "target_agent": "r2c-method-coder",
    "target_file": "method/method.py",
    "root_cause": ("DRIVER-AUTHORED FALLBACK, not an analyzed diagnosis: "
                   "notebook cell 29 raised RuntimeError; the actual root "
                   "cause is unanalyzed."),
}


def _seed(run_dir: Path, *, smoke_status: str | None,
          diagnosis: dict | None = None):
    """A delivered run dir with the three surfaces in place."""
    import run_layout
    from run_pipeline import StageResult

    state = make_state(run_dir)
    (run_dir / "method").mkdir(parents=True, exist_ok=True)
    (run_dir / "details").mkdir(parents=True, exist_ok=True)
    run_layout.run_path(run_dir, run_layout.PACKAGE_README).write_text(
        _README, encoding="utf-8")
    run_layout.run_path(run_dir, run_layout.NOTEBOOK_IPYNB).write_text(
        json.dumps(_NOTEBOOK, indent=1), encoding="utf-8")
    if smoke_status is not None:
        run_layout.run_path(run_dir, run_layout.KNOWN_ISSUES_MD).write_text(
            _KNOWN_ISSUES, encoding="utf-8")
        state.stage_results.append(
            StageResult(status=smoke_status, stage_id="stage_3c",
                        notes="degraded: the notebook does not run end-to-end"))
    else:
        state.stage_results.append(
            StageResult(status="completed", stage_id="stage_3c",
                        notes="smoke clean"))
    if diagnosis is not None:
        (state.paths.pipeline_dir / "smoke_diagnosis.json").write_text(
            json.dumps(diagnosis), encoding="utf-8")
    return state


def _surfaces(run_dir: Path) -> tuple[str, dict, str]:
    import run_layout

    readme = run_layout.run_path(
        run_dir, run_layout.PACKAGE_README).read_text(encoding="utf-8")
    notebook = json.loads(run_layout.run_path(
        run_dir, run_layout.NOTEBOOK_IPYNB).read_text(encoding="utf-8"))
    issues_path = run_layout.run_path(run_dir, run_layout.KNOWN_ISSUES_MD)
    issues = issues_path.read_text(encoding="utf-8") if issues_path.is_file() \
        else ""
    return readme, notebook, issues


def test_a_degraded_run_stamps_the_notebook_and_disarms_run_all(run_dir):
    from run_pipeline import finalize_degraded_surfaces

    state = _seed(run_dir, smoke_status="degraded")
    finalize_degraded_surfaces(state)
    readme, notebook, _ = _surfaces(run_dir)

    banner = "".join(notebook["cells"][0]["source"])
    assert "does NOT run end to end" in banner
    assert "cell 29" in banner
    assert "KNOWN_ISSUES.md" in banner
    # The original header survives, one cell down.
    assert "runnable tutorial" in "".join(notebook["cells"][1]["source"])

    assert "Click **Run All**" not in readme
    assert "Do not click Run All yet" in readme
    assert "## Layout" in readme, "the rest of the README is untouched"


def test_a_clean_run_leaves_every_surface_byte_identical(run_dir):
    from run_pipeline import finalize_degraded_surfaces

    state = _seed(run_dir, smoke_status=None)
    before = _surfaces(run_dir)
    finalize_degraded_surfaces(state)
    assert _surfaces(run_dir) == before


def test_a_fallback_diagnosis_is_labeled_as_routing_not_analysis(run_dir):
    from run_pipeline import finalize_degraded_surfaces

    state = _seed(run_dir, smoke_status="degraded",
                  diagnosis=_FALLBACK_DIAGNOSIS)
    finalize_degraded_surfaces(state)
    _, _, issues = _surfaces(run_dir)

    assert "Diagnosis provenance" in issues
    assert "root cause is UNANALYZED" in issues
    assert "`method/method.py`" in issues
    # The label lands ABOVE the fix pointer a researcher would act on.
    assert issues.index("Diagnosis provenance") < issues.index("What to do")


def test_an_analyzed_diagnosis_gets_no_provenance_warning(run_dir):
    from run_pipeline import finalize_degraded_surfaces

    state = _seed(run_dir, smoke_status="degraded", diagnosis={
        "schema_version": "1.0.0", "target_agent": "r2c-method-coder",
        "target_file": "method/method.py",
        "root_cause": "The lag slice returns width 4 because T_total is 4.",
    })
    finalize_degraded_surfaces(state)
    _, _, issues = _surfaces(run_dir)
    assert "Diagnosis provenance" not in issues


def test_the_pass_is_idempotent_across_a_resume(run_dir):
    """A resumed run finalizes again; banners must never stack."""
    from run_pipeline import finalize_degraded_surfaces

    state = _seed(run_dir, smoke_status="degraded",
                  diagnosis=_FALLBACK_DIAGNOSIS)
    finalize_degraded_surfaces(state)
    once = _surfaces(run_dir)
    finalize_degraded_surfaces(state)
    twice = _surfaces(run_dir)

    assert once == twice
    assert len(twice[1]["cells"]) == 3
    assert twice[2].count("Diagnosis provenance") == 1


def test_a_failed_smoke_stage_is_treated_like_a_degraded_one(run_dir):
    from run_pipeline import finalize_degraded_surfaces

    state = _seed(run_dir, smoke_status="failed")
    finalize_degraded_surfaces(state)
    readme, notebook, _ = _surfaces(run_dir)
    assert "does NOT run end to end" in "".join(notebook["cells"][0]["source"])
    assert "Click **Run All**" not in readme


def test_a_missing_notebook_or_readme_is_not_a_crash(run_dir):
    """Finalization runs on every terminal path, including thin packages."""
    import run_layout
    from run_pipeline import finalize_degraded_surfaces

    state = _seed(run_dir, smoke_status="degraded")
    run_layout.run_path(run_dir, run_layout.NOTEBOOK_IPYNB).unlink()
    run_layout.run_path(run_dir, run_layout.PACKAGE_README).unlink()
    finalize_degraded_surfaces(state)  # must not raise
