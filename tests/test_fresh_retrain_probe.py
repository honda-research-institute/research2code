"""AL-3 fresh-retrain gate, registered under its catalog id.

Registration, never duplication: the probe calls the stage validator's own
AST gate (validate_notebook_output._al_loop_warm_starts). These tests pin
the catalog-id wrapper against the zoo matrix — the badloop mutant must
fail (M-003), the corrected badge loop and the gbald double-retrain loop
must pass, warm-intending papers are N/A, and bootstrap-only notebooks are
honestly unprobeable. Static AST only, no torch needed.
"""

from __future__ import annotations

import json
from pathlib import Path

from probes.al_loop import probe_fresh_retrain_static

ZOO = Path(__file__).parent / "fixtures" / "zoo"


def test_al3_fails_badloop_mutant():
    v = probe_fresh_retrain_static(
        ZOO / "badge-badloop-reconstructed" / "notebook.ipynb")
    assert v.verdict == "fail"
    assert v.finding_class == "M-003"
    assert "warm-starting" in v.message


def test_al3_passes_corrected_badge_loop():
    v = probe_fresh_retrain_static(
        ZOO / "badge-goodmethod-badnotebook" / "notebook.ipynb")
    assert v.verdict == "pass", v.message


def test_al3_passes_gbald_double_retrain_loop():
    # The legitimate train-twice-per-round shape must not false-positive.
    v = probe_fresh_retrain_static(
        ZOO / "gbald-double-retrain-loop" / "notebook.ipynb")
    assert v.verdict == "pass", v.message


def test_al3_not_applicable_for_warm_start_protocol():
    v = probe_fresh_retrain_static(
        ZOO / "badge-badloop-reconstructed" / "notebook.ipynb",
        protocol="Warm-start the model between rounds per Section 5")
    assert v is None


def test_al3_unprobeable_without_inloop_training(tmp_path):
    nb = {"cells": [{"cell_type": "code", "id": "c1", "source": [
        "model = build_model(input_dim=4, n_classes=2)\n",
        "model = train_from_scratch(model, x, y)\n"]}]}
    path = tmp_path / "notebook.ipynb"
    path.write_text(json.dumps(nb))
    v = probe_fresh_retrain_static(path)
    assert v.verdict == "unprobeable"
    assert "no in-loop training" in v.message


def test_al3_unprobeable_when_notebook_missing(tmp_path):
    v = probe_fresh_retrain_static(tmp_path / "absent.ipynb")
    assert v.verdict == "unprobeable"


# ---------------------------------------------------------------------------
# Helper-call expansion (the 2026-06-12 GBALD audit blind spot): a loop that
# retrains through a locally defined helper is in-loop training.
# ---------------------------------------------------------------------------


def _nb(tmp_path, *cell_sources: str):
    nb = {"cells": [{"cell_type": "code", "id": f"c{i}", "source": [s]}
                    for i, s in enumerate(cell_sources)]}
    path = tmp_path / "notebook.ipynb"
    path.write_text(json.dumps(nb))
    return path


GBALD_SHAPE = """
def train_eval_current(round_seed):
    model = build_model(input_dim=784, n_classes=10)
    model = train_from_scratch(model, x, y, seed=round_seed)
    return model

model = train_eval_current(0)
for r in range(5):
    positions = select_batch(model, x_pool, x_lab, 3, r)
    model = train_eval_current(r + 1)
"""


def test_al3_passes_helper_routed_fresh_retrain(tmp_path):
    v = probe_fresh_retrain_static(_nb(tmp_path, GBALD_SHAPE))
    assert v.verdict == "pass", v.message


def test_al3_fails_helper_that_trains_without_rebuilding(tmp_path):
    src = """
def retrain(m):
    return train_from_scratch(m, x, y)

model = build_model(input_dim=4, n_classes=2)
for r in range(5):
    model = retrain(model)
"""
    v = probe_fresh_retrain_static(_nb(tmp_path, src))
    assert v.verdict == "fail"
    assert v.finding_class == "M-003"


def test_al3_follows_helpers_across_cells_and_nesting(tmp_path):
    cell_a = """
def fresh_round(seed):
    return _inner(seed)

def _inner(seed):
    m = build_model(input_dim=4, n_classes=2)
    return train_from_scratch(m, x, y, seed=seed)
"""
    cell_b = """
for r in range(3):
    model = fresh_round(r)
"""
    v = probe_fresh_retrain_static(_nb(tmp_path, cell_a, cell_b))
    assert v.verdict == "pass", v.message
