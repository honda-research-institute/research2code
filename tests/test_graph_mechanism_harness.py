"""Delivery-time and portable receipts for R2C-088 graph probes."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from build_probe_harness import (
    _frozen_graph_mechanism_execution_plan,
    _stage_2d_alignment_receipt,
)
from scripts.run_pipeline import _compute_stage_2d_upstream_digest


GRAPH_REF = "graph_mechanism.alignment_prerequisite"


def _seed_stage_2d_receipt(run_dir: Path) -> None:
    pipeline = run_dir / ".pipeline"
    method = run_dir / "method"
    pipeline.mkdir(parents=True)
    method.mkdir()
    (method / "model.py").write_text("# validated model\n", encoding="utf-8")
    (pipeline / "method_spec.json").write_text("{}\n", encoding="utf-8")
    (pipeline / "arch_contract.json").write_text("{}\n", encoding="utf-8")
    (pipeline / "params.json").write_text("{}\n", encoding="utf-8")
    paths = SimpleNamespace(run_dir=run_dir, pipeline_dir=pipeline)
    digest = _compute_stage_2d_upstream_digest(paths)
    (pipeline / "stage_2d.upstream_digest.json").write_text(
        json.dumps(digest, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (pipeline / "stage_2d.complete").write_text("", encoding="utf-8")


def _ready_plan() -> dict:
    return {
        "schema_version": "1.0",
        "status": "ready",
        "reason": None,
        "representation": "sparse_edge_index",
        "alignment": {
            "status": "ready",
            "reason": None,
            "element_id": "align-1",
        },
        "fixture": {
            "scope": "r2c084_asymmetric_relational_fixture",
            "entity_ids": [101, 203, 307, 409, 503],
        },
    }


def test_stage_2d_receipt_passes_only_for_exact_validated_authority(tmp_path):
    run = tmp_path / "run"
    _seed_stage_2d_receipt(run)

    receipt = _stage_2d_alignment_receipt(run, _ready_plan())
    assert receipt["status"] == "pass"
    assert receipt["verified"] is True
    assert receipt["probe_ref"] == GRAPH_REF
    assert receipt["element_id"] == "align-1"
    assert receipt["entity_ids"] == [101, 203, 307, 409, 503]
    assert receipt["authority_digests"]

    (run / "method" / "model.py").write_text(
        "# changed after validation\n", encoding="utf-8"
    )
    stale = _stage_2d_alignment_receipt(run, _ready_plan())
    assert stale["status"] == "unprobeable"
    assert stale["verified"] is False
    assert stale["reason"] == "stage_2d_runtime_validation_authority_drift"


def test_graph_freezer_promotes_only_a_verified_runtime_receipt(
    tmp_path,
    monkeypatch,
):
    run = tmp_path / "run"
    _seed_stage_2d_receipt(run)

    from scripts import graph_mechanism_runtime_plan as graph_plan

    monkeypatch.setattr(
        graph_plan,
        "normalize_graph_mechanism_runtime_plan",
        lambda method_spec, params, arch_contract: _ready_plan(),
    )
    frozen = _frozen_graph_mechanism_execution_plan(
        run, {GRAPH_REF: {"id": "HG1"}}
    )
    assert frozen is not None
    assert frozen["alignment"]["status"] == "pass"
    assert frozen["alignment_runtime_receipt"]["verified"] is True

    (run / ".pipeline" / "stage_2d.complete").unlink()
    unresolved = _frozen_graph_mechanism_execution_plan(
        run, {GRAPH_REF: {"id": "HG1"}}
    )
    assert unresolved is not None
    assert unresolved["alignment"]["status"] == "ready"
    assert unresolved["alignment_runtime_receipt"]["status"] == "unprobeable"


def test_graph_freezer_is_absent_without_an_exact_graph_ref(tmp_path):
    assert _frozen_graph_mechanism_execution_plan(
        tmp_path, {"time_series_forecasting.samples_contract": {}}
    ) is None
