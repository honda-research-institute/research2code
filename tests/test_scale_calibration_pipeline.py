"""Stage 2x application policy for typed calibration outcomes."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import run_pipeline
from tests.helpers.state import make_state


@pytest.mark.parametrize("context_kind", ["other", "unit_norm"])
def test_unprobeable_context_is_log_only_without_researcher_assumption(
    tmp_path, monkeypatch, context_kind,
):
    state = make_state(tmp_path / "run")
    assessment = {
        "applicable": True,
        "entries": [{
            "name": "threshold",
            "paper_value": 2.0,
            "status": "unprobeable",
            "reason": f"{context_kind} has no supported observer",
        }],
    }

    def fake_run_script(stage_id, command, timeout):
        del stage_id, timeout
        out = Path(command[command.index("--out") + 1])
        out.write_text(json.dumps(assessment), encoding="utf-8")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    logs: list[tuple[str, str, str]] = []
    assumptions: list[dict] = []
    monkeypatch.setattr(run_pipeline, "run_script", fake_run_script)
    monkeypatch.setattr(
        run_pipeline, "log",
        lambda stage, status, message: logs.append((stage, status, message)),
    )
    monkeypatch.setattr(
        run_pipeline, "_append_assumption",
        lambda *args, **kwargs: assumptions.append(kwargs),
    )
    monkeypatch.setattr(
        run_pipeline, "_next_assumption_id",
        lambda *_: pytest.fail("unprobeable context requested an assumption id"),
    )

    run_pipeline._apply_scale_calibration(state, "stage_2x_params")

    assert assumptions == []
    assert [status for _, status, _ in logs] == [
        "scale_calibration_unprobeable"
    ]
    assert context_kind in logs[0][2]


def test_supported_unresolved_mismatch_still_creates_attention_assumption(
    tmp_path, monkeypatch,
):
    state = make_state(tmp_path / "run")
    assessment = {
        "applicable": True,
        "entries": [{
            "name": "sigma",
            "paper_value": 2.0,
            "assumes_declared": {
                "kind": "representation_convention",
                "convention": "target_box_grid",
            },
            "status": "needs_attention",
            "reason": "target boxes are normalized and no conversion is defined",
        }],
    }

    def fake_run_script(stage_id, command, timeout):
        del stage_id, timeout
        out = Path(command[command.index("--out") + 1])
        out.write_text(json.dumps(assessment), encoding="utf-8")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    assumptions: list[dict] = []
    monkeypatch.setattr(run_pipeline, "run_script", fake_run_script)
    monkeypatch.setattr(run_pipeline, "log", lambda *args: None)
    monkeypatch.setattr(run_pipeline, "_next_assumption_id", lambda *_: "A001")
    monkeypatch.setattr(
        run_pipeline, "_append_assumption",
        lambda *args, **kwargs: assumptions.append(kwargs),
    )

    run_pipeline._apply_scale_calibration(state, "stage_2x_params")

    assert len(assumptions) == 1
    assert assumptions[0]["aid"] == "A001"
    assert "sigma" in assumptions[0]["title"]
