"""R2C-069 — a demo evaluates on a window that actually holds data.

Two failure classes, one root. `axis_live_extent_unmeasured`: a bundle's
nominal span is treated as its usable span. `empty_eval_window`: metrics
computed against a target slice with no finite values.

The shaping case is the 2026-08-06 pdfgnn loop-2 bundle. Of 1,092 daily steps
spanning 2017-01-02 to 2019-12-29, every row dated 2019-11-01 or later has a
blank `sales` cell, in the SOURCE, because that period is the competition's own
prediction window. The demo held out its last four weeks per the R2C-066
evaluation-window check, that window sat entirely inside the withheld region,
and RMSE, MAE and WMAPE all printed nan. Three seams were blind: the
materializer measured only the nominal axis, the R2C-065 feasibility floor
compared the protocol against that nominal count, and the briefing's per-column
line reported `sales: float, 0/5000 empty` because it samples the head of a
499,044-row file.

The tail shape is reconstructed at fixture scale rather than vendored: the
bundled table is 27 MB.
"""

from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.bundle_axis_floor import (
    AxisRequirement, check_bundle_axis, protocol_axis_floor,
)
from scripts.dataset_acquisition import _axis_aware_copy, _file_opener


def _write_panel(
    path: Path, *, steps: int, entities: int = 2, dead_tail: int = 0,
    dead_columns: tuple[str, ...] = ("sales",),
) -> None:
    """A long-format daily panel: one row per (entity, day), the last
    `dead_tail` days carrying blanks in `dead_columns`. This is the layout of
    every public retail-forecasting export, including the paper's own."""
    header = ["product_id", "date", "sales", "revenue", "price"]
    rows = [",".join(header)]
    for step in range(steps):
        day = (date(2017, 1, 1) + timedelta(days=step)).isoformat()
        dead = step >= steps - dead_tail
        for entity in range(entities):
            cells = [f"P{entity}", day, "4.0", "8.0", "2.0"]
            if dead:
                for column in dead_columns:
                    cells[header.index(column)] = ""
            rows.append(",".join(cells))
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def _bundle(tmp_path: Path, **panel) -> dict:
    """Run the real materializer copy over a generated panel and return the
    axis facts it recorded."""
    src = tmp_path / "src.csv"
    _write_panel(src, **panel)
    _rows, _truncated, axis = _axis_aware_copy(
        _file_opener(src), tmp_path / "out.csv", suffix=".csv", max_rows=100_000)
    assert axis is not None
    return axis.as_dict()


# --- The measurement ---------------------------------------------------------


def test_the_withheld_tail_is_measured(tmp_path):
    facts = _bundle(tmp_path, steps=100, dead_tail=12,
                    dead_columns=("sales", "revenue"))

    # The nominal axis is unchanged: nothing was dropped or trimmed.
    assert facts["steps_kept"] == 100
    assert facts["last_step"] == "2017-04-10"
    # The usable axis stops where the target stops.
    assert facts["live_steps"] == 88
    assert facts["last_live_step"] == "2017-03-29"
    assert [(d["column"], d["dead_tail_steps"]) for d in facts["dead_tail_columns"]] \
        == [("sales", 12), ("revenue", 12)]
    # A column that runs to the end is not listed.
    assert "price" not in [d["column"] for d in facts["dead_tail_columns"]]


def test_a_complete_table_reports_its_whole_axis_as_usable(tmp_path):
    facts = _bundle(tmp_path, steps=40)

    assert facts["dead_tail_columns"] == []
    assert facts["live_steps"] == facts["steps_kept"] == 40
    assert facts["last_live_step"] == facts["last_step"]


def test_a_column_empty_everywhere_is_not_a_dead_tail(tmp_path):
    """An all-blank column has no data at all, which the briefing's
    missing-count line already states. Treating it as a dead tail would drag
    the table's usable extent to zero and refuse every real bundle."""
    src = tmp_path / "src.csv"
    src.write_text(
        "date,sales,unused\n"
        "2017-01-01,4.0,\n"
        "2017-01-02,5.0,\n"
        "2017-01-03,6.0,\n", encoding="utf-8")
    _rows, _truncated, axis = _axis_aware_copy(
        _file_opener(src), tmp_path / "out.csv", suffix=".csv", max_rows=100)

    facts = axis.as_dict()
    assert facts["dead_tail_columns"] == []
    assert facts["live_steps"] == 3


def test_the_boundary_is_keyed_by_the_rows_own_step(tmp_path):
    """A panel is exported entity-major as often as date-major: all of P0's
    days, then all of P1's. Each step's ordinal is fixed when it first appears,
    so a later row for an earlier step cannot move a boundary forward."""
    src = tmp_path / "entity_major.csv"
    src.write_text(
        "product_id,date,sales\n"
        "P0,2017-01-01,1.0\nP0,2017-01-02,2.0\nP0,2017-01-03,\n"
        "P1,2017-01-01,3.0\nP1,2017-01-02,4.0\nP1,2017-01-03,\n",
        encoding="utf-8")

    facts = _axis_aware_copy(_file_opener(src), tmp_path / "out.csv",
                             suffix=".csv", max_rows=100)[2].as_dict()

    assert facts["steps_kept"] == 3
    assert facts["live_steps"] == 2
    assert facts["last_live_step"] == "2017-01-02"


def test_a_non_chronological_export_reports_no_dead_tail(tmp_path):
    """Position means lateness only in a chronological file. A shuffled export
    gets no dead tail rather than a wrong one: missing a withheld tail costs
    the briefing a warning, while inventing one would send a producer to trim
    live data."""
    src = tmp_path / "shuffled.csv"
    src.write_text(
        "date,sales\n"
        "2017-01-01,1.0\n2017-01-03,\n2017-01-02,2.0\n", encoding="utf-8")

    facts = _axis_aware_copy(_file_opener(src), tmp_path / "out.csv",
                             suffix=".csv", max_rows=100)[2].as_dict()

    assert facts["dead_tail_columns"] == []
    assert facts["live_steps"] == facts["steps_kept"] == 3


def test_the_loop2_tail_shape_reproduces(tmp_path):
    """The shaping case at fixture scale: three columns withheld over the same
    final period, the rest of the table complete. The real bundle measured
    1,033 usable of 1,092 steps ending 2019-10-31, on sales, revenue and
    stock."""
    facts = _bundle(tmp_path, steps=200, entities=3, dead_tail=59,
                    dead_columns=("sales", "revenue"))

    assert facts["steps_kept"] == 200
    assert facts["live_steps"] == 141
    assert {d["column"] for d in facts["dead_tail_columns"]} == {"sales", "revenue"}
    assert {d["dead_tail_steps"] for d in facts["dead_tail_columns"]} == {59}


# --- The feasibility floor ---------------------------------------------------


_REQUIREMENT = AxisRequirement(
    context_steps=28, validation_steps=7, test_steps=7,
    unit="day", granularity=1,
)


def _manifest(axis: dict) -> dict:
    return {"files": [{"file": "sales.csv", "time_axis": axis}]}


def test_the_floor_measures_the_usable_axis(tmp_path):
    """A bundle long enough nominally but too short once the withheld tail is
    removed must fail. Floor here is 28 + 7 + 7 = 42."""
    facts = _bundle(tmp_path, steps=60, dead_tail=25)
    assert facts["steps_kept"] == 60      # clears 42 nominally
    assert facts["live_steps"] == 35      # does not clear it usably

    verdict = check_bundle_axis(_manifest(facts), _REQUIREMENT)

    assert verdict is not None
    assert verdict.feasible is False
    assert verdict.realized_steps == 35
    assert verdict.nominal_steps == 60
    assert "longest usable bundled axis is 35 steps" in verdict.message()
    assert "the table spans 60 steps" in verdict.message()
    assert "`sales` stop carrying values" in verdict.message()
    assert "the last 25 hold no target" in verdict.message()


def test_a_feasible_bundle_still_states_its_dead_tail(tmp_path):
    """The loop-2 case: the usable axis clears the floor 20 times over, so the
    bundle is right to ship. The verdict still has to say the tail is empty,
    because that is what the demo went on to score against."""
    facts = _bundle(tmp_path, steps=200, dead_tail=59)

    verdict = check_bundle_axis(_manifest(facts), _REQUIREMENT)

    assert verdict.feasible is True
    assert verdict.realized_steps == 141
    assert "141 usable steps" in verdict.message()
    assert "the last 59 hold no target" in verdict.message()


def test_a_complete_bundle_message_carries_no_dead_tail_note(tmp_path):
    facts = _bundle(tmp_path, steps=60)

    verdict = check_bundle_axis(_manifest(facts), _REQUIREMENT)

    assert verdict.feasible is True
    assert verdict.realized_steps == 60
    assert "USABLE" not in verdict.message()
    assert "hold no target" not in verdict.message()


def test_a_bundle_predating_cadence_measurement_is_unresolved_not_zero_length():
    """Legacy provenance keeps its nominal extent but cannot prove units."""
    legacy = {"column": "date", "steps_kept": 90, "steps_in_source": 90,
              "first_step": "2017-01-01", "last_step": "2017-03-31",
              "rows_per_step_cap": None}

    verdict = check_bundle_axis(_manifest(legacy), _REQUIREMENT)

    assert verdict.feasible is None
    assert verdict.realized_steps == 90
    assert verdict.nominal_steps == 90
    assert "no exact regular parseable source cadence" in verdict.message()
    assert "hold no target" not in verdict.message()


def test_the_longest_axis_is_chosen_by_usable_length(tmp_path):
    """Two tables, and the one with the longer NOMINAL axis is the one whose
    target is withheld. The floor has to reason about the usable one."""
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    long_dead = _bundle(tmp_path / "a", steps=200, dead_tail=180)
    short_live = _bundle(tmp_path / "b", steps=100)
    manifest = {"files": [
        {"file": "withheld.csv", "time_axis": long_dead},
        {"file": "complete.csv", "time_axis": short_live},
    ]}

    verdict = check_bundle_axis(manifest, _REQUIREMENT)

    assert verdict.table == "complete.csv"
    assert verdict.realized_steps == 100


def test_no_time_axis_still_yields_no_verdict():
    assert check_bundle_axis({"files": [{"file": "x.csv"}]}, _REQUIREMENT) is None


# --- The briefing ------------------------------------------------------------


def _run_dir_with(tmp_path: Path, axis: dict) -> SimpleNamespace:
    example_data = tmp_path / "method" / "example_data"
    example_data.mkdir(parents=True, exist_ok=True)
    _write_panel(example_data / "sales.csv", steps=3)
    (example_data / "PROVENANCE.json").write_text(json.dumps({
        "source": {"cited_url": "https://example.org/data"},
        "files": [{"file": "sales.csv", "rows_kept": 6, "truncated": True,
                   "time_axis": axis}],
    }), encoding="utf-8")
    return SimpleNamespace(run_dir=tmp_path)


def test_the_briefing_states_where_usable_data_ends(tmp_path):
    from run_pipeline import _bundled_demo_data_section, _notebook_bundle_section

    (tmp_path / "measure").mkdir()
    facts = _bundle(tmp_path / "measure", steps=200, dead_tail=59)
    paths = _run_dir_with(tmp_path, facts)

    for section in (_bundled_demo_data_section(paths),
                    _notebook_bundle_section(paths)):
        assert section is not None
        assert "USABLE DATA ENDS EARLY" in section
        assert "`sales` ends at" in section
        assert "59 step(s) with no value after it" in section
        # The rule the notebook generator has to apply, and the consequence
        # the loop-2 roll paid for not knowing it.
        assert "must end at or before the last live step" in section
        assert "prints nan" in section


def test_the_briefing_says_nothing_extra_for_a_complete_table(tmp_path):
    from run_pipeline import _bundled_demo_data_section

    (tmp_path / "measure").mkdir()
    facts = _bundle(tmp_path / "measure", steps=40)
    paths = _run_dir_with(tmp_path, facts)

    section = _bundled_demo_data_section(paths)
    assert "USABLE DATA ENDS EARLY" not in section


def test_the_column_line_no_longer_claims_completeness(tmp_path):
    """The line that misled the generator. It reported `sales: float, 0/5000
    empty` on a table whose last 59 of 1,092 days were blank, because the
    sample is the head of a 499,044-row file. It must scope its own claim."""
    from run_pipeline import _bundled_demo_data_section

    (tmp_path / "measure").mkdir()
    facts = _bundle(tmp_path / "measure", steps=40)
    paths = _run_dir_with(tmp_path, facts)

    section = _bundled_demo_data_section(paths)
    assert "of the first 6 rows empty" in section
    assert "/6 empty" not in section


# --- The researcher surface --------------------------------------------------


def test_provenance_carries_a_prose_usable_extent_note(tmp_path):
    from scripts.dataset_acquisition import _usable_extent_note

    (tmp_path / "measure").mkdir()
    facts = _bundle(tmp_path / "measure", steps=200, dead_tail=59)
    note = _usable_extent_note([{"file": "sales.csv", "time_axis": facts}])

    assert "sales.csv spans 200 steps" in note
    assert "carries no values for `sales`" in note
    assert "yields nan" in note
    # And nothing at all for a complete bundle.
    (tmp_path / "clean").mkdir()
    clean = _bundle(tmp_path / "clean", steps=40)
    assert _usable_extent_note([{"file": "sales.csv", "time_axis": clean}]) == ""


# --- Noise floor -------------------------------------------------------------


def test_the_delivered_fleet_reports_no_spurious_dead_tails(tmp_path):
    """Every bundled table in the tree, re-measured through the real code
    path: a dead tail must be reported only where the data genuinely has one,
    and the usable extent must never come out as zero. Run dirs are deleted
    before re-rolls, so this asserts on the property rather than on a count or
    a directory name. Writes only into tmp_path — never into a run dir."""
    checked = 0
    for root in (Path("r2c_runs"), Path("example_runs")):
        if not root.is_dir():
            continue
        for provenance in sorted(root.glob("*/method/example_data/PROVENANCE.json")):
            try:
                manifest = json.loads(provenance.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            for entry in manifest.get("files") or []:
                table = provenance.parent / str(entry.get("file") or "")
                if not table.is_file() or table.suffix.lower() != ".csv":
                    continue
                if table.stat().st_size > 5_000_000:
                    continue  # the 27 MB loop-2 table has its own reproduction
                _r, _t, remeasured = _axis_aware_copy(
                    _file_opener(table), tmp_path / f"{checked}.csv",
                    suffix=".csv", max_rows=1_000_000)
                if remeasured is None:
                    continue  # no time axis: not this check's business
                facts = remeasured.as_dict()
                checked += 1
                assert 0 < facts["live_steps"] <= facts["steps_kept"]
                for dead in facts["dead_tail_columns"]:
                    # A reported dead tail is a real one: the column carries
                    # values somewhere and stops before the axis does.
                    assert 0 < dead["live_steps"] < facts["steps_kept"], (
                        f"{table}: {dead}")
                    assert dead["dead_tail_steps"] > 0
    if checked == 0:
        pytest.skip("no small bundled csv with a time axis in the tree")
