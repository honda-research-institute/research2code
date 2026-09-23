"""Reviewer/diagnostician input hygiene (0714 ladder finding 7).

Oversized inputs induce Think-tier turn deaths: ICRA's stage 4 read the
executed notebook with a 144KB embedded image (~109k input tokens, two
dead-stops); detr's diagnostician read the whole 477KB data_flow.json and
burned its caps. The fixes: the fidelity reviewer reads an
outputs-stripped notebook copy, and the diagnostician gets a pre-sliced
data-flow file scoped to the failing cell + traceback.
"""

from __future__ import annotations

import json

from tests.helpers.state import make_state


def _notebook_with_heavy_outputs() -> dict:
    return {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {},
        "cells": [
            {"cell_type": "markdown", "source": ["# Title"]},
            {
                "cell_type": "code",
                "source": ["plot()\n"],
                "outputs": [
                    {"output_type": "display_data",
                     "data": {"image/png": "A" * 144_000,
                              "text/plain": ["<Figure>"]}},
                    {"output_type": "stream", "name": "stdout",
                     "text": ["line-head\n"] + ["x" * 80 + "\n"] * 200
                             + ["line-tail\n"]},
                ],
                "execution_count": 1,
                "metadata": {},
            },
        ],
    }


def test_strip_notebook_outputs_replaces_images_and_clips_text():
    from run_pipeline import _strip_notebook_outputs_for_review

    stripped = _strip_notebook_outputs_for_review(_notebook_with_heavy_outputs())
    outputs = stripped["cells"][1]["outputs"]
    assert "stripped for review" in outputs[0]["data"]["image/png"]
    assert len(outputs[0]["data"]["image/png"]) < 200
    text = outputs[1]["text"]
    assert text.startswith("line-head")          # head kept
    assert text.rstrip().endswith("line-tail")   # tail kept (the traceback lesson)
    assert "chars stripped" in text
    # Code cells untouched.
    assert stripped["cells"][1]["source"] == ["plot()\n"]


def test_write_review_notebook_shrinks_and_is_best_effort(run_dir):
    from run_pipeline import _write_review_notebook

    state = make_state(run_dir)
    # No notebook yet: best-effort None.
    assert _write_review_notebook(state) is None

    nb_path = run_dir / "notebook.ipynb"
    nb_path.write_text(json.dumps(_notebook_with_heavy_outputs()),
                       encoding="utf-8")
    out = _write_review_notebook(state)
    assert out is not None and out.name == "notebook_for_review.ipynb"
    assert out.stat().st_size < nb_path.stat().st_size / 5
    json.loads(out.read_text(encoding="utf-8"))  # stays valid JSON


def test_slice_data_flow_scopes_to_failing_symbols(run_dir):
    from run_pipeline import _slice_data_flow_for_diagnosis

    state = make_state(run_dir)
    data_flow = {
        "schema_version": "1.0",
        "files_scanned": ["notebook_draft.py"],
        "symbols": {
            "n_classes": {"assignments": [
                {"file": "notebook_draft.py", "line": 3, "scope": "<module>",
                 "expr": "len(y_pool.unique())"}], "reads": []},
            "y_pool": {"assignments": [
                {"file": "notebook_draft.py", "line": 2, "scope": "<module>",
                 "expr": "load_data()"}], "reads": []},
            "unrelated_symbol": {"assignments": [], "reads": []},
        },
    }
    full_path = run_dir / ".pipeline" / "data_flow.json"
    full_path.write_text(json.dumps(data_flow), encoding="utf-8")

    out = _slice_data_flow_for_diagnosis(
        state, full_path,
        failing_cell_source="model.fit(x, n_classes)",
        stderr_tail="ValueError: labels out of [0, n_classes)")
    assert out is not None
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["seed_symbols"] == ["n_classes"]
    # One upstream hop: y_pool appears in n_classes' assignment expr.
    assert set(payload["symbols"]) == {"n_classes", "y_pool"}
    assert "unrelated_symbol" not in payload["symbols"]

    # No matching symbol: no slice, caller falls back to the full map.
    assert _slice_data_flow_for_diagnosis(
        state, full_path,
        failing_cell_source="totally.different()",
        stderr_tail="KeyError: 'zap'") is None


def test_diagnosis_prompt_prefers_the_slice():
    from dispatch_templates import build_smoke_diagnosis_prompt
    from run_pipeline import build_paths_block
    from tests.helpers.state import make_paths

    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmp:
        paths = make_paths(Path(tmp) / "run")
        prompt = build_smoke_diagnosis_prompt(
            paths=build_paths_block(paths),
            stderr_tail="boom",
            failing_cell_source="x = 1",
            failing_cell_index=3,
            section=5,
            diagnosis_output_path="/x/smoke_diagnosis.json",
            data_flow_path="/x/data_flow.json",
            data_flow_slice_path="/x/data_flow_slice.json",
        )
        assert "data_flow_slice.json" in prompt
        assert "Read the SLICE, not the full map" in prompt

        # Without a slice, the original full-map wording stands.
        prompt_full = build_smoke_diagnosis_prompt(
            paths=build_paths_block(paths),
            stderr_tail="boom",
            failing_cell_source="x = 1",
            failing_cell_index=3,
            section=5,
            diagnosis_output_path="/x/smoke_diagnosis.json",
            data_flow_path="/x/data_flow.json",
        )
        assert "pre-extracted a structured map" in prompt_full
