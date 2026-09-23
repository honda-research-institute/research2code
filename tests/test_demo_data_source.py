"""Whether a family can get its own demo data is declared, not inferred (R2C-074).

Demo dataset acquisition used to fire on gap-path runs only, on the reasoning
that a committed family owns its data story. That was true while every
committed family's template downloaded its own data. Promoting a family OUT of
the gap path with the gap path's file-only loader still in its templates moved
it across the gate and switched its data off with nothing saying so: the
forecasting family was committed on 2026-08-05 and its next run fetched
nothing, then delivered a demo with no data to train on.
"""

from __future__ import annotations

import json

import pytest

from scripts.taxonomy import (
    DEMO_DATA_NEEDS_ACQUISITION,
    DEMO_DATA_SOURCES,
    DEMO_DATA_TEMPLATE_DOWNLOADS,
    demo_data_source,
    load_taxonomy,
)
from scripts.validate_taxonomy import lint


# ---------------------------------------------------------------------------
# The declaration itself
# ---------------------------------------------------------------------------


def test_the_promoted_family_declares_that_it_needs_acquisition():
    # The case the record was opened on. Its templates read local files only,
    # so a run of it must try to bundle the paper's own cited dataset.
    assert demo_data_source("time_series_forecasting") == DEMO_DATA_NEEDS_ACQUISITION


@pytest.mark.parametrize("slug", ["active_learning", "knowledge_distillation",
                                  "vision_transformer"])
def test_the_downloading_families_say_so(slug):
    # These three were the silent assumption behind the old gate. Saying it out
    # loud is what makes the assumption checkable.
    assert demo_data_source(slug) == DEMO_DATA_TEMPLATE_DOWNLOADS


def test_an_undeclared_family_reads_as_undeclared_not_as_a_default():
    # The five families nobody has examined keep today's behavior. Inventing a
    # default for them would change five families on one family's evidence.
    assert demo_data_source("domain_adaptation") is None
    assert demo_data_source("stochastic_optimization") is None


def test_an_unknown_or_missing_paradigm_is_undeclared():
    assert demo_data_source(None) is None
    assert demo_data_source("no_such_family") is None


def test_an_unrecognised_value_does_not_resolve():
    # A typo must read as undeclared rather than as a third mode.
    from scripts import taxonomy as tx

    assert "typo_value" not in DEMO_DATA_SOURCES
    assert set(DEMO_DATA_SOURCES) == {
        tx.DEMO_DATA_TEMPLATE_DOWNLOADS, tx.DEMO_DATA_NEEDS_ACQUISITION}


# ---------------------------------------------------------------------------
# The lint arm
# ---------------------------------------------------------------------------


def test_lint_reports_undeclared_templated_nodes_as_a_warning_not_an_error():
    diags = lint(load_taxonomy())
    undeclared = [d for d in diags if d.code == "demo_data_source_undeclared"]
    assert undeclared, "the undeclared families must stay visible"
    assert all(d.severity == "warning" for d in undeclared)
    assert not [d for d in diags if d.severity == "error"]


def test_lint_does_not_warn_about_a_family_that_declared_it():
    diags = lint(load_taxonomy())
    warned = {d.node for d in diags if d.code == "demo_data_source_undeclared"}
    assert not any(n and "TSF" in n for n in warned)


# ---------------------------------------------------------------------------
# The gate
# ---------------------------------------------------------------------------


def _run_dir(tmp_path, paradigm_id: str, *, gap: bool):
    run = tmp_path / "run"
    (run / ".pipeline").mkdir(parents=True)
    (run / "method" / "example_data").mkdir(parents=True)
    (run / ".pipeline" / "method_spec.json").write_text(json.dumps(
        {"comparison": {"classification": {"id": paradigm_id}}}), encoding="utf-8")
    if gap:
        (run / ".pipeline" / "provisional_packs").mkdir()
    return run


@pytest.fixture()
def gate(monkeypatch):
    """Call the acquisition gate and report whether it reached the fetch."""
    import run_pipeline as rp

    def _run(run, *, paradigm_id):
        reached = []
        monkeypatch.setattr(
            "dataset_acquisition.acquire_for_paper",
            lambda *a, **k: reached.append(True) or (_ for _ in ()).throw(
                RuntimeError("stop after the gate")))

        class _Paths:
            run_dir = run
            pipeline_dir = run / ".pipeline"
            paper_md = run / ".pipeline" / "paper.md"
            method_spec = run / ".pipeline" / "method_spec.json"

        _Paths.paper_md.write_text("a paper about forecasting", encoding="utf-8")

        class _State:
            paths = _Paths()

        rp._acquire_demo_dataset_if_needed(_State(), stage_id="stage_2a")
        return bool(reached)

    return _run


def test_a_committed_family_that_needs_acquisition_now_gets_it(gate, tmp_path):
    # The regression, directly. Before the fix this returned at the gap check
    # and fetched nothing.
    run = _run_dir(tmp_path, "time_series_forecasting", gap=False)
    assert gate(run, paradigm_id="time_series_forecasting") is True


def test_a_gap_run_still_acquires(gate, tmp_path):
    run = _run_dir(tmp_path, "no_such_family", gap=True)
    assert gate(run, paradigm_id="no_such_family") is True


def test_a_downloading_family_is_left_alone(gate, tmp_path):
    run = _run_dir(tmp_path, "active_learning", gap=False)
    assert gate(run, paradigm_id="active_learning") is False


def test_an_undeclared_family_keeps_todays_behaviour(gate, tmp_path):
    run = _run_dir(tmp_path, "domain_adaptation", gap=False)
    assert gate(run, paradigm_id="domain_adaptation") is False


def test_bundled_data_short_circuits_before_any_declaration_matters(gate, tmp_path):
    run = _run_dir(tmp_path, "time_series_forecasting", gap=False)
    (run / "method" / "example_data" / "demo.csv").write_text("a,b\n1,2\n",
                                                              encoding="utf-8")
    assert gate(run, paradigm_id="time_series_forecasting") is False


def test_an_unreadable_spec_does_not_break_the_gate(gate, tmp_path):
    run = _run_dir(tmp_path, "time_series_forecasting", gap=False)
    (run / ".pipeline" / "method_spec.json").write_text("{not json",
                                                        encoding="utf-8")
    assert gate(run, paradigm_id="time_series_forecasting") is False
