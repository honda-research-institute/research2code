"""R2C-071 — a NaN metric re-enters its own fix loop on any paradigm.

`paradigm_scoped_universal_rule`: a rule that is universal by design enforced
only inside one paradigm's branch, so every other family silently loses the
enforcement point. The smoke-time sanity pre-check's docstring promised "one
rule, two enforcement points", and its only rule was accuracy-versus-chance,
gated on a detected class count. On the 2026-08-06 pdfgnn loop-2 roll it
returned None on its third line because forecasting is not classification: the
notebook executed cleanly with `RMSE: nan`, `MAE: nan`, `WMAPE: nan`, and the
NaN surfaced for the first time as a stage-5 delivery demoter with four
iterations of smoke budget unspent.

`symptom_asserted_as_diagnosis`: the stage-5 wording asserted "the run
diverged". Nothing diverged. The independent reviewer reproduced the notebook
end to end, found 0 of 207,563 weights NaN and every forecast finite, and
traced the NaN to a held-out window whose ground truth was missing.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import run_pipeline
from probes.universal import (
    first_nan_inf_with_cell, nonfinite_metric_message, withheld_target_note,
)
from tests.helpers.state import make_state

_ZOO = Path(__file__).parent / "fixtures/zoo"
_NAN_FIXTURE = _ZOO / "tsf-demo-nan-metrics-reconstructed/notebook.ipynb"
_PASS_FIXTURE = _ZOO / "tsf-demo-pass-reconstructed/notebook.ipynb"


def _state_with(tmp_path: Path, notebook: Path):
    state = make_state(tmp_path / "run")
    (state.paths.run_dir / "notebook.ipynb").write_text(
        notebook.read_text(encoding="utf-8"), encoding="utf-8")
    return state


def _classification_notebook(tmp_path: Path, lines: list[str]) -> Path:
    nb = {"cells": [{
        "cell_type": "code", "source": [""], "metadata": {},
        "execution_count": 1,
        "outputs": [{"output_type": "stream", "name": "stdout",
                     "text": [line + "\n" for line in lines]}],
    }], "metadata": {}, "nbformat": 4, "nbformat_minor": 5}
    path = tmp_path / "nb.ipynb"
    path.write_text(json.dumps(nb), encoding="utf-8")
    return path


# --- The hoist: the rule now runs on a non-classification paradigm ----------


def test_a_forecasting_notebook_with_a_nan_metric_fails_smoke(tmp_path):
    state = _state_with(tmp_path, _NAN_FIXTURE)

    result = run_pipeline._trainability_smoke_check(state)

    assert result is not None, "the pre-check bailed on a non-classification paper"
    cell, stderr = result.cell, result.stderr
    # Routed at the cell that PRINTED the value, like the chance arm.
    assert cell == 13
    # The arm names itself, so the driver's log line and run event describe
    # what actually happened instead of reporting every failure as
    # non-learning.
    assert result.kind == "nonfinite_metric"
    assert result.summary == "executed metric 'RMSE' printed 'nan'"
    assert "NON-FINITE METRIC" in stderr
    assert "printed 'nan'" in stderr
    assert "RMSE" in stderr
    # Says why it is a smoke failure rather than a note.
    assert "demonstrates \nnothing" in stderr or "demonstrates nothing" in stderr
    # Names the mechanism it belongs to, so the next reader can find it.
    assert "stage-5 delivery demoter" in stderr


def test_the_finite_twin_stays_silent(tmp_path):
    """The adjacent-good fixture: same layout, same metrics, real numbers. A
    pre-check that fires here would block every healthy forecasting demo."""
    state = _state_with(tmp_path, _PASS_FIXTURE)

    assert run_pipeline._trainability_smoke_check(state) is None


def test_the_classification_chance_arm_is_untouched(tmp_path):
    """The existing rule keeps its behavior in both directions, and it is
    reached because the divergence arm is silent on finite outputs."""
    state = make_state(tmp_path / "run")
    below = _classification_notebook(tmp_path, [
        "Classes: 10",
        "round 1: accuracy=0.087",
        "round 2: accuracy=0.085",
    ])
    (state.paths.run_dir / "notebook.ipynb").write_text(
        below.read_text(encoding="utf-8"), encoding="utf-8")

    result = run_pipeline._trainability_smoke_check(state)
    assert result is not None
    assert result.kind == "below_chance"
    assert "SILENT NON-LEARNING" in result.stderr
    assert "chance level" in result.stderr
    assert "peaks at 0.087" in result.summary

    above = _classification_notebook(tmp_path, [
        "Classes: 10",
        "round 1: accuracy=0.412",
        "round 2: accuracy=0.688",
    ])
    (state.paths.run_dir / "notebook.ipynb").write_text(
        above.read_text(encoding="utf-8"), encoding="utf-8")

    assert run_pipeline._trainability_smoke_check(state) is None


def test_a_nan_outranks_the_chance_arm(tmp_path):
    """Priority matches the stage-5 battery: divergence is the primary verdict,
    so a classification run with a NaN reports the NaN rather than a chance
    number computed from broken values."""
    state = make_state(tmp_path / "run")
    nb = _classification_notebook(tmp_path, [
        "Classes: 10",
        "round 1: accuracy=nan",
        "round 2: accuracy=nan",
    ])
    (state.paths.run_dir / "notebook.ipynb").write_text(
        nb.read_text(encoding="utf-8"), encoding="utf-8")

    result = run_pipeline._trainability_smoke_check(state)

    assert result is not None
    assert result.kind == "nonfinite_metric"
    assert "NON-FINITE METRIC" in result.stderr
    assert "SILENT NON-LEARNING" not in result.stderr


def test_an_unreadable_or_output_free_notebook_is_not_a_failure(tmp_path):
    """A smoke gate must not fail on a missing signal — the stage-5 battery
    discloses those honestly as unprobeable."""
    state = make_state(tmp_path / "run")
    (state.paths.run_dir / "notebook.ipynb").write_text("{not json",
                                                        encoding="utf-8")
    assert run_pipeline._trainability_smoke_check(state) is None

    (state.paths.run_dir / "notebook.ipynb").write_text(json.dumps({
        "cells": [{"cell_type": "code", "source": [""], "outputs": [],
                   "metadata": {}, "execution_count": None}],
        "metadata": {}, "nbformat": 4, "nbformat_minor": 5,
    }), encoding="utf-8")
    assert run_pipeline._trainability_smoke_check(state) is None


def test_prose_containing_info_does_not_trip_the_arm(tmp_path):
    """The scan is value-position only. `inference`/`info` in prose must not
    read as an infinite metric, or every notebook that logs a status line
    fails smoke."""
    state = make_state(tmp_path / "run")
    nb = _classification_notebook(tmp_path, [
        "Running inference on the held-out window",
        "info: 30 articles",
        "RMSE: 12.31",
    ])
    (state.paths.run_dir / "notebook.ipynb").write_text(
        nb.read_text(encoding="utf-8"), encoding="utf-8")

    assert run_pipeline._trainability_smoke_check(state) is None


def test_the_locator_agrees_with_the_stage5_scan():
    """One rule, two enforcement points: the cell-aware locator must find the
    same token the stage-5 helper finds."""
    from probes.universal import _cell_output_texts, first_nan_inf

    nb = json.loads(_NAN_FIXTURE.read_text(encoding="utf-8"))
    key, token, cell = first_nan_inf_with_cell(nb)
    assert (key, token) == first_nan_inf(_cell_output_texts(nb))
    assert nb["cells"][cell]["cell_type"] == "code"


# --- The misdiagnosis rider -------------------------------------------------


def test_the_message_does_not_assert_divergence():
    message = nonfinite_metric_message("RMSE", "nan")

    assert "a non-finite value reached the metric computation" in message
    assert "the run diverged" not in message
    # Every candidate source named, with divergence as one of them.
    assert "missing or all-zero ground truth" in message
    assert "training divergence" in message
    assert "division by zero" in message
    assert "before treating this as divergence" in message


def test_the_bundles_own_provenance_becomes_the_leading_candidate(tmp_path):
    """When the run's own artifacts already record a withheld target, the
    message names it instead of listing generic possibilities. This is the
    2026-08-06 case: the answer was on disk in the bundle the run itself
    built."""
    example_data = tmp_path / "method" / "example_data"
    example_data.mkdir(parents=True)
    (example_data / "PROVENANCE.json").write_text(json.dumps({
        "files": [{"file": "sales.csv", "time_axis": {
            "column": "date", "steps_kept": 1092, "live_steps": 1033,
            "last_live_step": "2019-10-31",
            "dead_tail_columns": [{"column": "sales",
                                   "last_live_step": "2019-10-31",
                                   "live_steps": 1033,
                                   "dead_tail_steps": 59}],
        }}],
    }), encoding="utf-8")

    note = withheld_target_note(tmp_path)
    assert note is not None
    assert "`sales` after 2019-10-31" in note
    assert "59 step(s)" in note

    message = nonfinite_metric_message("RMSE", "nan", tmp_path)
    assert "bundle provenance offers one" in message
    assert "2019-10-31" in message
    assert "training divergence" in message  # still named, still not asserted


def test_no_bundle_means_no_invented_explanation(tmp_path):
    assert withheld_target_note(tmp_path) is None
    message = nonfinite_metric_message("RMSE", "nan", tmp_path)
    assert "bundle provenance" not in message
    assert "missing or all-zero ground truth" in message


def test_a_complete_bundle_offers_no_withheld_target_note(tmp_path):
    example_data = tmp_path / "method" / "example_data"
    example_data.mkdir(parents=True)
    (example_data / "PROVENANCE.json").write_text(json.dumps({
        "files": [{"file": "sales.csv", "time_axis": {
            "column": "date", "steps_kept": 200, "live_steps": 200,
            "last_live_step": "2017-07-19", "dead_tail_columns": [],
        }}],
    }), encoding="utf-8")

    assert withheld_target_note(tmp_path) is None


def test_the_smoke_failure_carries_the_provenance_diagnosis(tmp_path):
    """End to end at the seam: the synthetic stderr the fix loop receives
    names the real cause, so the producer is not sent to add gradient
    clipping."""
    state = _state_with(tmp_path, _NAN_FIXTURE)
    example_data = state.paths.run_dir / "method" / "example_data"
    example_data.mkdir(parents=True)
    (example_data / "PROVENANCE.json").write_text(json.dumps({
        "files": [{"file": "sales.csv", "time_axis": {
            "column": "date", "steps_kept": 1092, "live_steps": 1033,
            "last_live_step": "2019-10-31",
            "dead_tail_columns": [{"column": "sales",
                                   "last_live_step": "2019-10-31",
                                   "live_steps": 1033,
                                   "dead_tail_steps": 59}],
        }}],
    }), encoding="utf-8")

    stderr = run_pipeline._trainability_smoke_check(state).stderr

    assert "bundle provenance offers one" in stderr
    assert "2019-10-31" in stderr
    assert "evaluation \nwindowing" in stderr or "evaluation windowing" in stderr


# --- Noise floor ------------------------------------------------------------


def test_the_delivered_fleet_trips_only_the_known_bad():
    """Measured, not assumed: across every delivered notebook in the tree the
    scan must fire on the loop-2 delivery and nothing else. A false positive
    here costs real fix-loop iterations, because this arm now FAILS smoke."""
    tripped: list[str] = []
    total = 0
    for root in (Path("r2c_runs"), Path("example_runs")):
        if not root.is_dir():
            continue
        for nb_path in sorted(root.glob("*/notebook.ipynb")):
            try:
                nb = json.loads(nb_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            total += 1
            if first_nan_inf_with_cell(nb) is not None:
                tripped.append(nb_path.parent.name)
    if total == 0:
        pytest.skip("no delivered notebooks in the tree")
    # Every tripped notebook must be a genuine nan-metric delivery. The
    # loop-2 roll is the only one in the tree as of 2026-08-06, and the pin is
    # on the property rather than the name because run dirs get deleted.
    for name in tripped:
        nb = json.loads((Path("r2c_runs") / name / "notebook.ipynb").read_text()
                        if (Path("r2c_runs") / name).is_dir()
                        else (Path("example_runs") / name / "notebook.ipynb").read_text())
        key, token, _cell = first_nan_inf_with_cell(nb)
        assert token.lower().lstrip("+-") in ("nan", "inf"), (name, key, token)
    assert len(tripped) <= 1, f"more than one fleet notebook trips: {tripped}"
