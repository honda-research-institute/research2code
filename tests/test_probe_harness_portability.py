"""The portable check harness (R2C-019): built from a run, executed from
OUTSIDE the repository, reproducing the delivered check set exactly.

The adversarial review of the harness design proved three failure modes,
each pinned here: a copied battery dies on repo-internal imports (the
harness must be self-contained), a vendored battery without frozen gating
silently runs a different probe set than the delivered report (the
run-everything fallback), and a battery run used to rewrite the run's
internal artifacts (the harness always uses the isolated output mode).
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent

from build_probe_harness import build_harness  # noqa: E402
from run_probes import main as run_probes_main  # noqa: E402

pytestmark = pytest.mark.probe_runtime


def _make_al_run(tmp_path: Path) -> Path:
    """A minimal run whose paradigm has a real node-declared check set
    (active_learning), producing report + ledger on a battery pass."""
    run = tmp_path / "delivered-run"
    (run / ".pipeline").mkdir(parents=True)
    (run / "method").mkdir()
    (run / "method" / "method.py").write_text(
        "def select_batch(model, x_unlabeled, batch_size, seed):\n"
        "    return list(range(batch_size))\n")
    (run / ".pipeline" / "params.json").write_text(json.dumps({
        "schema_version": "1.0.0",
        "params": {"batch_size": {"value": 8, "source": "paper",
                                  "note": "stated."}}}))
    (run / ".pipeline" / "paper.md").write_text("some paper text")
    (run / ".pipeline" / "method_spec.json").write_text(json.dumps({
        "comparison": {"classification": {"id": "active_learning"},
                       "pluggable_component": {"name": "select_batch"}}}))
    (run / ".pipeline" / "paper_map.json").write_text(json.dumps({
        "elements": [{"id": "exp-1", "type": "experiment",
                      "source_text": "Method improves accuracy by 4%."}]}))
    return run


def _tree_hashes(root: Path) -> dict[str, str]:
    return {
        p.relative_to(root).as_posix():
            hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(root.rglob("*")) if p.is_file()
    }


def _run_harness_outside_repo(harness: Path, run: Path, out: Path,
                              cwd: Path) -> subprocess.CompletedProcess:
    """Execute the built entrypoint as a researcher would: from a working
    directory outside the repository, with no repo path importable."""
    env = {k: v for k, v in os.environ.items()
           if k not in ("PYTHONPATH", "PYTHONSTARTUP")}
    return subprocess.run(
        [sys.executable, str(harness / "run_checks.py"),
         "--run-dir", str(run), "--output-dir", str(out)],
        cwd=cwd, env=env, capture_output=True, text=True, timeout=600)


def test_built_harness_runs_outside_repo_and_freezes_the_check_set(tmp_path):
    run = _make_al_run(tmp_path)
    harness = run / "validation" / "harness"
    manifest = build_harness(run, harness)
    assert manifest["paradigm"] == "active_learning"
    assert manifest["battery_version"] not in ("", None)
    frozen = json.loads((harness / "frozen_gating.json").read_text())
    assert frozen["training_history_required"] is False

    # The delivered check set, produced by the IN-REPO battery on the same
    # run (isolated output so the fixture tree stays pristine).
    in_repo_out = tmp_path / "in-repo-results"
    run_probes_main(["--run-dir", str(run),
                     "--output-dir", str(in_repo_out)])
    delivered = json.loads((in_repo_out / "probe_report.json").read_text())

    before = _tree_hashes(run)
    out = tmp_path / "portable-results"
    proc = _run_harness_outside_repo(harness, run, out, cwd=tmp_path)
    assert proc.returncode in (0, 1), (
        f"harness crashed outside the repo:\n{proc.stderr[-2000:]}")

    # Nothing under the run tree changed — including .pipeline/.
    assert _tree_hashes(run) == before, "portable run mutated the run tree"

    portable = json.loads((out / "probe_report.json").read_text())
    assert (out / "claims_ledger.json").is_file()

    # The frozen gating reproduces the delivered check set EXACTLY: same
    # probe ids in the same order, not a fallback superset.
    delivered_ids = [v["probe_id"] for v in delivered["verdicts"]]
    portable_ids = [v["probe_id"] for v in portable["verdicts"]]
    assert portable_ids == delivered_ids

    # Both reports carry the version stamp; the portable one is frozen
    # from the same checkout here, so they match.
    assert portable["battery_version"] == manifest["battery_version"]
    assert delivered["battery_version"]


def test_typed_calibration_dispatch_is_vendored_with_stage5_parity(tmp_path):
    """The portable US-4 path uses the same pure typed-context dispatcher."""
    run = _make_al_run(tmp_path)
    spec_path = run / ".pipeline" / "method_spec.json"
    spec = json.loads(spec_path.read_text())
    spec["critical_requirements"] = {
        "scale_dependent_hyperparameters": [{
            "name": "R_0",
            "paper_value": 2000.0,
            "description": "Distance radius calibrated to raw pixels.",
            "paper_section": "Section 4",
            "calibration_context": {
                "kind": "feature_magnitude",
                "scale": "raw_pixel_unnormalized",
            },
        }],
    }
    spec_path.write_text(json.dumps(spec), encoding="utf-8")
    params_path = run / ".pipeline" / "params.json"
    params = json.loads(params_path.read_text())
    params["params"]["R_0"] = {
        "value": 2000.0,
        "source": "paper",
        "paper_section": "Section 4",
    }
    params_path.write_text(json.dumps(params), encoding="utf-8")
    example = run / "method" / "example_data"
    example.mkdir()
    (example / "pool.json").write_text(json.dumps({
        "x_pool": [[0.0, 0.25], [0.5, 1.0]],
    }), encoding="utf-8")

    delivered_out = tmp_path / "delivered-results"
    run_probes_main([
        "--run-dir", str(run), "--output-dir", str(delivered_out),
    ])
    delivered = json.loads(
        (delivered_out / "probe_report.json").read_text(encoding="utf-8")
    )

    harness = run / "validation" / "harness"
    build_harness(run, harness)
    assert (harness / "probes" / "calibration_context.py").is_file()
    portable_out = tmp_path / "portable-results"
    proc = _run_harness_outside_repo(
        harness, run, portable_out, cwd=tmp_path,
    )
    assert proc.returncode in (0, 1), proc.stderr[-2000:]
    portable = json.loads(
        (portable_out / "probe_report.json").read_text(encoding="utf-8")
    )

    def us4_rows(report):
        return [v for v in report["verdicts"] if v["probe_id"] == "US-4"]

    assert us4_rows(delivered) == us4_rows(portable)
    assert us4_rows(portable)[0]["verdict"] == "fail"


def test_frozen_null_context_round_trips_as_full_battery(tmp_path):
    """A paradigm no node served at delivery froze declared_context as
    null; the vendored battery must reproduce the full-battery fallback,
    not treat null as an empty check set."""
    from run_probes import run_battery

    run = _make_al_run(tmp_path)
    frozen = {"schema_version": "1.0", "battery_version": "x",
              "paradigm": "some_reserved_family", "declared_context": None}
    report_frozen = run_battery(
        run, frozen_gating=frozen,
        ledger_path=tmp_path / "l1" / "claims_ledger.json")
    report_fallback = run_battery(
        run, frozen_gating=None,
        ledger_path=tmp_path / "l2" / "claims_ledger.json")
    # Same run, so the fallback comparison needs the same paradigm route:
    # the frozen-null path must at minimum include the universal tiers and
    # produce a non-empty verdict list identical in shape to running with
    # no gating declaration at all for this fixture.
    assert [v.probe_id for v in report_frozen.verdicts]
    assert len(report_frozen.verdicts) >= len(report_fallback.verdicts)


def test_harness_readme_and_manifest_enumerate_the_frozen_set(tmp_path):
    run = _make_al_run(tmp_path)
    harness = tmp_path / "harness"
    manifest = build_harness(run, harness)

    readme = (harness / "README.md").read_text(encoding="utf-8")
    frozen = json.loads((harness / "frozen_gating.json").read_text())
    assert frozen["declared_context"], "AL node should declare checks"
    for ref in frozen["declared_context"]:
        assert f"`{ref}`" in readme, f"README omits frozen check {ref}"
    assert manifest["battery_version"] in readme

    # Manifest digests match the shipped bytes (tamper evidence).
    for rel, digest in manifest["files"].items():
        actual = hashlib.sha256((harness / rel).read_bytes()).hexdigest()
        assert actual == digest, f"manifest digest mismatch for {rel}"


def test_harness_freezes_exact_demo_skill_inputs_and_role_carrier(tmp_path):
    run = _make_al_run(tmp_path)
    carrier = {
        "role": "test_span",
        "scheme_kind": "single_holdout",
        "scheme_paper_element_ids": ["evaluation-split"],
        "role_paper_element_ids": ["test-window"],
    }
    verdict = {
        "schema_version": "2.0.0",
        "decided_by": "structured_demo_skill",
        "demo_skill_contract": {"schema_version": "1.0.0"},
        "executed_evaluation": {
            "schema_version": "2.0.0",
            "evaluation_protocol_role": "test_span",
        },
        "evaluation_validity_receipt": {
            "status": "valid",
            "reasons": [],
            "evaluation_protocol_role": carrier,
        },
        "evaluation_protocol_role": carrier,
    }
    (run / ".pipeline" / "demo_verdict.json").write_text(
        json.dumps(verdict), encoding="utf-8"
    )

    harness = tmp_path / "harness"
    build_harness(run, harness)
    frozen = json.loads((harness / "frozen_gating.json").read_text())
    evidence = frozen["demo_skill_evidence"]

    assert evidence["status"] == "ready"
    assert evidence["family_contract"] == verdict["demo_skill_contract"]
    assert evidence["executed_record"] == verdict["executed_evaluation"]
    assert evidence["validity_receipt"] == verdict[
        "evaluation_validity_receipt"
    ]
    assert evidence["evaluation_protocol_role"] == carrier
    assert (harness / "demo_skill_evidence.py").is_file()


def test_harness_freezes_schema2_missing_role_as_named_unresolved(tmp_path):
    run = _make_al_run(tmp_path)
    verdict = {
        "schema_version": "2.0.0",
        "decided_by": "structured_demo_skill",
        "demo_skill_contract": {"schema_version": "1.0.0"},
        "executed_evaluation": {"schema_version": "2.0.0"},
        "evaluation_validity_receipt": {
            "status": "unresolved",
            "reasons": [
                "executed_evaluation_protocol_role_missing_or_unknown"
            ],
        },
        "evaluation_protocol_role": None,
    }
    (run / ".pipeline" / "demo_verdict.json").write_text(
        json.dumps(verdict), encoding="utf-8"
    )

    harness = tmp_path / "harness"
    build_harness(run, harness)
    evidence = json.loads(
        (harness / "frozen_gating.json").read_text()
    )["demo_skill_evidence"]

    assert evidence["status"] == "unresolved"
    assert evidence["evaluation_protocol_role"] is None
    assert (
        "executed_evaluation_protocol_role_missing_or_unknown"
        in evidence["reasons"]
    )
    assert (
        "executed_evaluation_protocol_role_unresolved" in evidence["reasons"]
    )


def test_harness_rejects_top_level_role_carrier_drift(tmp_path):
    run = _make_al_run(tmp_path)
    receipt_carrier = {
        "role": "test_span",
        "scheme_kind": "single_holdout",
        "scheme_paper_element_ids": ["evaluation-split"],
        "role_paper_element_ids": ["test-window"],
    }
    verdict = {
        "schema_version": "2.0.0",
        "decided_by": "structured_demo_skill",
        "demo_skill_contract": {"schema_version": "1.0.0"},
        "executed_evaluation": {
            "schema_version": "2.0.0",
            "evaluation_protocol_role": "test_span",
        },
        "evaluation_validity_receipt": {
            "status": "valid",
            "reasons": [],
            "evaluation_protocol_role": receipt_carrier,
        },
        "evaluation_protocol_role": {
            **receipt_carrier,
            "role_paper_element_ids": ["different-element"],
        },
    }
    (run / ".pipeline" / "demo_verdict.json").write_text(
        json.dumps(verdict), encoding="utf-8"
    )

    harness = tmp_path / "harness"
    build_harness(run, harness)
    evidence = json.loads(
        (harness / "frozen_gating.json").read_text()
    )["demo_skill_evidence"]

    assert evidence["status"] == "unresolved"
    assert evidence["evaluation_protocol_role"] is None
    assert "evaluation_protocol_role_carrier_mismatch" in evidence["reasons"]


def test_harness_keeps_legacy_demo_record_readable_without_role_proof(tmp_path):
    run = _make_al_run(tmp_path)
    verdict = {
        "schema_version": "2.0.0",
        "decided_by": "structured_demo_skill",
        "demo_skill_contract": {"schema_version": "1.0.0"},
        "executed_evaluation": {"schema_version": "1.0.0"},
        "evaluation_validity_receipt": {"status": "valid", "reasons": []},
        "evaluation_protocol_role": None,
    }
    (run / ".pipeline" / "demo_verdict.json").write_text(
        json.dumps(verdict), encoding="utf-8"
    )

    harness = tmp_path / "harness"
    build_harness(run, harness)
    evidence = json.loads(
        (harness / "frozen_gating.json").read_text()
    )["demo_skill_evidence"]

    assert evidence["status"] == "ready"
    assert evidence["executed_record"] == {"schema_version": "1.0.0"}
    assert evidence["evaluation_protocol_role"] is None


def test_harness_replays_frozen_schema2_forecasting_plan_outside_repo(tmp_path):
    """The portable arm executes normalized JSON, never a live shape guess."""
    from tests.test_time_series_forecasting_probes import (
        _ready_heldout_evidence,
        _write_generated_package,
    )
    from tests.test_tsf_runtime_plan import (
        _graph_free_contract,
        _method_spec,
    )
    from tests.test_time_series_training_history import _record

    run = _write_generated_package(tmp_path / "forecasting-run")
    (run / ".pipeline").mkdir()
    contract = _graph_free_contract()
    contract["pluggable_component"]["input"]["num_samples"]["source"][
        "literal"
    ] = 512
    spec = _method_spec(autoregressive=False)
    spec["comparison"] = {
        "classification": {"id": "time_series_forecasting"},
        "pluggable_component": {
            "name": "forecast",
            "signature": (
                "forecast(model, history, static_features, "
                "time_varying_features, graph, entity_ids, "
                "target_scaling_state, seed) -> ForecastResult"
            ),
            "seed_param": "seed",
        },
    }
    (run / ".pipeline" / "arch_contract.json").write_text(
        json.dumps(contract), encoding="utf-8"
    )
    (run / ".pipeline" / "method_spec.json").write_text(
        json.dumps(spec), encoding="utf-8"
    )
    demo = _ready_heldout_evidence()
    (run / ".pipeline" / "demo_verdict.json").write_text(
        json.dumps({
            "schema_version": "2.0.0",
            "decided_by": "structured_demo_skill",
            "demo_skill_contract": demo["family_contract"],
            "executed_evaluation": demo["executed_record"],
            "evaluation_validity_receipt": demo["validity_receipt"],
            "evaluation_protocol_role": demo[
                "evaluation_protocol_role"
            ],
        }),
        encoding="utf-8",
    )
    (run / ".pipeline" / "training_history.json").write_text(
        json.dumps(_record()), encoding="utf-8"
    )
    from tests.test_training_history_post_smoke import _state

    (run / ".pipeline" / "params.json").write_text(
        json.dumps({"schema_version": "1.0.0", "params": {}}),
        encoding="utf-8",
    )
    (run / ".pipeline" / "target_scaling_state.json").write_text(
        json.dumps(_state()), encoding="utf-8"
    )
    (run / "requirements.txt").write_text("numpy\n", encoding="utf-8")
    (run / "notebook.ipynb").write_text(
        json.dumps({"cells": []}), encoding="utf-8"
    )

    harness = tmp_path / "forecasting-harness"
    build_harness(run, harness)
    frozen = json.loads((harness / "frozen_gating.json").read_text())
    plan = frozen["forecasting_execution_plan"]

    assert plan["source_contract_schema"] == "2.0.0"
    assert plan["construction"]["route"] == {
        "kind": "builder",
        "module": "method.training",
        "callable": "build_model",
    }
    assert plan["relational"] is None
    assert frozen["forecasting_probe_groundings"][
        "time_series_forecasting.magnitude_collapse"
    ] == {
        "element_ids": ["target-scaling"],
        "paper_element_ids": ["paper-target-scaling"],
    }
    assert frozen["training_history_required"] is True
    authority = frozen["training_history_authority_digests"]
    assert {
        "method/model.py",
        "method/training.py",
        "notebook.ipynb",
        ".pipeline/method_spec.json",
        ".pipeline/arch_contract.json",
        ".pipeline/training_history.json",
    } <= set(authority)
    assert authority["method/training.py"]
    assert authority["notebook.ipynb"]
    assert (harness / "typed_fixture.py").is_file()
    assert (harness / "scripts" / "time_series_training_history.py").is_file()
    assert (harness / "scripts" / "time_series_target_scaling.py").is_file()

    outside = tmp_path / "outside"
    outside.mkdir()
    out = tmp_path / "portable-forecasting-results"
    proc = _run_harness_outside_repo(harness, run, out, cwd=outside)
    assert proc.returncode in (0, 1), proc.stderr[-2000:]
    report = json.loads((out / "probe_report.json").read_text())
    rows = {
        row["probe_id"]: row
        for row in report["verdicts"]
        if row["probe_id"].startswith("TSF-")
    }
    assert {key: value["verdict"] for key, value in rows.items()} == {
        "TSF-1": "pass",
        "TSF-2": "not_applicable",
        "TSF-3": "pass",
        "TSF-4": "pass",
        "TSF-5": "pass",
    }
    assert all(rows[probe_id]["probe_ref"] for probe_id in rows)
    portable_history = next(
        row for row in report["verdicts"] if row["probe_id"] == "UB-9"
    )
    assert portable_history["verdict"] == "pass"
    assert "record_digest=" in portable_history["evidence"]

    from run_probes import run_battery

    live = run_battery(
        run,
        ledger_path=tmp_path / "live-ledger" / "claims_ledger.json",
    )
    live_history = next(
        row for row in live.verdicts if row.probe_id == "UB-9"
    )
    assert live_history.to_dict() == portable_history


@pytest.mark.parametrize("drift_target", ["params", "method_source"])
def test_harness_replays_graph_rows_and_blocks_stale_authority_outside_repo(
    tmp_path, monkeypatch, drift_target,
):
    """HG-1..7 share exact frozen evidence in repo and in the copied kit."""
    import build_probe_harness as harness_builder
    from tests.test_graph_mechanism_consumers import (
        GRAPH_REFS,
        _declared_context,
        _frozen,
        _ready_plan,
        _write_run,
    )

    run = _write_run(tmp_path / "graph-source")
    plan = _ready_plan(run)
    context = _declared_context(*GRAPH_REFS)
    monkeypatch.setattr(
        harness_builder, "_node_declared_probe_context", lambda _: context
    )
    monkeypatch.setattr(
        harness_builder,
        "_frozen_graph_mechanism_execution_plan",
        lambda *_: json.loads(json.dumps(plan)),
    )

    # Write the same frozen plan used by the copied harness rather than
    # invoking a different live normalizer.
    live_frozen = _frozen(plan)
    live_frozen_path = tmp_path / "graph-live-frozen-gating.json"
    live_frozen_path.write_text(
        json.dumps(live_frozen), encoding="utf-8"
    )
    live_out = tmp_path / "graph-live-results"
    assert run_probes_main([
        "--run-dir", str(run),
        "--output-dir", str(live_out),
        "--frozen-gating", str(live_frozen_path),
    ]) in (0, 1)
    live_report = json.loads(
        (live_out / "probe_report.json").read_text(encoding="utf-8")
    )

    harness = tmp_path / "graph-harness"
    build_harness(run, harness)
    assert (harness / "graph_callable_liveness.py").is_file()
    assert (harness / "probes" / "graph_mechanism.py").is_file()
    frozen = json.loads(
        (harness / "frozen_gating.json").read_text(encoding="utf-8")
    )
    assert frozen["graph_mechanism_execution_plan"] == plan

    before = _tree_hashes(run)
    outside = tmp_path / "outside-graph"
    outside.mkdir()
    portable_out = tmp_path / "graph-portable-results"
    proc = _run_harness_outside_repo(
        harness, run, portable_out, cwd=outside
    )
    assert proc.returncode in (0, 1), proc.stderr[-2000:]
    assert _tree_hashes(run) == before
    portable_report = json.loads(
        (portable_out / "probe_report.json").read_text(encoding="utf-8")
    )

    def graph_rows(report):
        return [
            row for row in report["verdicts"]
            if row["probe_id"].startswith("HG-")
        ]

    assert graph_rows(portable_report) == graph_rows(live_report)
    assert [row["verdict"] for row in graph_rows(portable_report)] == [
        "pass"
    ] * 7

    if drift_target == "params":
        params_path = run / ".pipeline" / "params.json"
        params = json.loads(params_path.read_text(encoding="utf-8"))
        params["post_delivery_drift"] = True
        params_path.write_text(json.dumps(params), encoding="utf-8")
    else:
        with (run / "method" / "model.py").open(
            "a", encoding="utf-8"
        ) as stream:
            stream.write("\n# post-delivery drift\n")
    drifted = _tree_hashes(run)
    stale_out = tmp_path / "graph-stale-results"
    stale = _run_harness_outside_repo(
        harness, run, stale_out, cwd=outside
    )
    assert stale.returncode in (0, 1), stale.stderr[-2000:]
    assert _tree_hashes(run) == drifted
    stale_report = json.loads(
        (stale_out / "probe_report.json").read_text(encoding="utf-8")
    )
    stale_rows = graph_rows(stale_report)
    assert stale_rows[0]["verdict"] == (
        "pass" if drift_target == "params" else "unprobeable"
    )
    assert [row["verdict"] for row in stale_rows[1:]] == [
        "unprobeable"
    ] * 6
    if drift_target == "params":
        assert all(
            row["reason"] == "graph_callable_liveness_receipt_stale"
            for row in stale_rows[1:]
        )
    else:
        assert stale_rows[0]["reason"] == "alignment_runtime_receipt_stale"
        assert all(
            "HG-1 alignment prerequisite did not pass" in row["reason"]
            for row in stale_rows[1:]
        )


def _make_training_history_authority_run(tmp_path: Path) -> Path:
    from tests.test_time_series_training_history import _record
    from tests.test_training_history_post_smoke import _state

    run = _make_al_run(tmp_path)
    (run / "method" / "training.py").write_text(
        "def train_model(*args, **kwargs):\n    return {}\n",
        encoding="utf-8",
    )
    (run / "method" / "model.py").write_text(
        "class Model:\n    pass\n", encoding="utf-8"
    )
    (run / "method" / "example_data").mkdir()
    (run / "method" / "example_data" / "values.csv").write_text(
        "entity,value\na,1\n", encoding="utf-8"
    )
    (run / "notebook.ipynb").write_text(
        json.dumps({"cells": []}), encoding="utf-8"
    )
    (run / "requirements.txt").write_text("numpy\n", encoding="utf-8")
    (run / ".pipeline" / "arch_contract.json").write_text(
        json.dumps({
            "schema_version": "2.0.0",
            "training_loop": {"function_name": "train_model"},
        }),
        encoding="utf-8",
    )
    (run / ".pipeline" / "target_scaling_state.json").write_text(
        json.dumps(_state()), encoding="utf-8"
    )
    (run / ".pipeline" / "demo_verdict.json").write_text(
        json.dumps({}), encoding="utf-8"
    )
    (run / ".pipeline" / "training_history.json").write_text(
        json.dumps(_record()), encoding="utf-8"
    )
    return run


@pytest.mark.parametrize(
    ("relative", "action", "replacement"),
    [
        (
            "method/model.py",
            "write",
            "# edited model source\n",
        ),
        (
            "method/example_data/values.csv",
            "write",
            "entity,value\na,2\n",
        ),
        (
            ".pipeline/params.json",
            "reformat_json",
            None,
        ),
        (
            ".pipeline/training_history.json",
            "reformat_json",
            None,
        ),
        (
            "notebook.ipynb",
            "write",
            json.dumps({
                "cells": [{"cell_type": "markdown", "source": "edited"}],
            }),
        ),
        (
            "method/new_training_helper.py",
            "write",
            "VALUE = 1\n",
        ),
        (
            "method/training.py",
            "delete",
            None,
        ),
    ],
)
def test_portable_ub9_rejects_stale_delivery_history_after_authority_drift(
    tmp_path, monkeypatch, relative, action, replacement,
):
    """Outside-repo UB-9 rejects source/data/config/artifact set drift."""
    import build_probe_harness as harness_builder
    import run_probes
    run = _make_training_history_authority_run(tmp_path)
    monkeypatch.setattr(
        harness_builder, "_live_training_history_required", lambda _: True
    )
    harness = tmp_path / "history-harness"
    build_harness(run, harness)

    target = run / relative
    if action == "write":
        target.write_text(replacement, encoding="utf-8")
    elif action == "reformat_json":
        target.write_text(
            json.dumps(json.loads(target.read_text()), indent=2) + "\n",
            encoding="utf-8",
        )
    else:
        assert action == "delete"
        target.unlink()
    outside = tmp_path / "outside-history"
    outside.mkdir()
    out = tmp_path / "portable-history-results"
    proc = _run_harness_outside_repo(harness, run, out, cwd=outside)
    assert proc.returncode in (0, 1), proc.stderr[-2000:]
    portable = json.loads((out / "probe_report.json").read_text())
    history = next(
        row for row in portable["verdicts"] if row["probe_id"] == "UB-9"
    )
    assert history["verdict"] == "unprobeable"
    assert history["reason"] == "training_history_authority_stale"
    assert relative in history["evidence"]

    # Live execution has no frozen-delivery contract and retains the existing
    # behavior: it evaluates the current pipeline-owned artifact directly.
    monkeypatch.setattr(
        run_probes, "_live_training_history_required", lambda _: True
    )
    live = run_probes.run_battery(
        run,
        ledger_path=tmp_path / "live-history" / "claims_ledger.json",
    )
    live_history = next(
        row for row in live.verdicts if row.probe_id == "UB-9"
    )
    assert live_history.verdict == "pass"


def test_portable_ub9_requires_complete_authority_at_harness_build(
    tmp_path, monkeypatch,
):
    import build_probe_harness as harness_builder

    run = _make_training_history_authority_run(tmp_path)
    (run / "requirements.txt").unlink()
    monkeypatch.setattr(
        harness_builder, "_live_training_history_required", lambda _: True
    )
    harness = tmp_path / "incomplete-authority-harness"
    build_harness(run, harness)
    frozen = json.loads((harness / "frozen_gating.json").read_text())
    assert frozen["training_history_authority_digests"][
        "requirements.txt"
    ] is None

    outside = tmp_path / "outside-incomplete"
    outside.mkdir()
    out = tmp_path / "incomplete-results"
    proc = _run_harness_outside_repo(harness, run, out, cwd=outside)
    assert proc.returncode in (0, 1), proc.stderr[-2000:]
    report = json.loads((out / "probe_report.json").read_text())
    history = next(
        row for row in report["verdicts"] if row["probe_id"] == "UB-9"
    )
    assert history["verdict"] == "unprobeable"
    assert history["reason"] == (
        "training_history_authority_digest_unavailable"
    )


# ---------------------------------------------------------------------------
# Hardware block + capability preflight (R2C-019 block 3): quotes and
# measurements, never inventions. The paper-scale tier was cut down to a
# preflight because the deliverable has no paper-scale code to run; the
# refusal names what it would take, and a paper that states no hardware
# renders an honest absence instead of a forced number.
# ---------------------------------------------------------------------------

SRL_HW_SENTENCE = (
    "In particular, on a computer with an i7-5820K CPU, a Python "
    "implementation of a four-agent policy takes on average 8.7ms for "
    "each query of the value network.")


def test_hardware_statement_extraction_quote_or_absence():
    from build_probe_harness import paper_hardware_statement

    paper = ("# Title\n\nIntro prose.\n\n## IV. EXPERIMENTS\n\n"
             + SRL_HW_SENTENCE + " More prose follows.\n")
    stmt = paper_hardware_statement(paper)
    assert stmt is not None
    assert "i7-5820K" in stmt["text"]
    assert stmt["text"] in " ".join(paper.split()), "not verbatim"
    assert "EXPERIMENTS" in (stmt["section"] or "")
    # Honest absence: no hardware statement at all.
    assert paper_hardware_statement("# T\n\nA methods paper.") is None
    assert paper_hardware_statement(None) is None
    # A related-work aside with no experiment context never binds.
    assert paper_hardware_statement(
        "# T\n\nGPU acceleration is a popular direction [12].") is None
    # A generic mention WITH experiment context does bind.
    stmt = paper_hardware_statement(
        "# T\n\n## Setup\n\nAll models were trained on a single GPU.")
    assert stmt is not None and "single GPU" in stmt["text"]


def test_preflight_refuses_gpu_class_without_cuda_and_notes_otherwise():
    from harness_preflight import render_preflight

    block = {"compute_class": "Single GPU required",
             "estimated_time": "9 hours on one V100",
             "paper_statement": {"text": "Trained on a single V100 GPU.",
                                 "section": "Section 5"}}
    no_gpu = {"cpu_count": 8, "ram_gb": 16.0, "torch": "2.3.0",
              "cuda_available": False, "python": "3.12", "platform": "x"}
    text = render_preflight(block, no_gpu)
    assert "REFUSED" in text
    assert "Trained on a single V100 GPU." in text  # names what it takes
    assert "demo scale" in text
    # CPU-sufficient paper on the same machine: no refusal, honest note.
    text = render_preflight({"compute_class": "CPU sufficient",
                             "estimated_time": "minutes",
                             "paper_statement": None}, no_gpu)
    assert "REFUSED" not in text
    assert "does not state its hardware" in text
    # GPU-class paper WITH a GPU: no refusal either.
    with_gpu = dict(no_gpu, cuda_available=True)
    text = render_preflight(block, with_gpu)
    assert "REFUSED" not in text


def test_built_harness_ships_hardware_block_and_preflight_output(tmp_path):
    run = _make_al_run(tmp_path)
    (run / ".pipeline" / "method_spec.json").write_text(json.dumps({
        "comparison": {"classification": {"id": "active_learning"},
                       "pluggable_component": {"name": "select_batch"}},
        "dependencies": {"compute": "CPU sufficient",
                         "estimated_time": "under an hour"}}))
    (run / ".pipeline" / "paper.md").write_text(
        "# Paper\n\n## Experiments\n\n" + SRL_HW_SENTENCE + "\n")
    harness = run / "validation" / "harness"
    build_harness(run, harness)

    block = json.loads((harness / "hardware_block.json").read_text())
    assert block["compute_class"] == "CPU sufficient"
    assert "i7-5820K" in block["paper_statement"]["text"]

    out = tmp_path / "results"
    proc = _run_harness_outside_repo(harness, run, out, cwd=tmp_path)
    assert proc.returncode in (0, 1), proc.stderr[-1500:]
    assert "Capability preflight" in proc.stdout
    assert "i7-5820K" in proc.stdout
    preflight = json.loads((out / "preflight.json").read_text())
    assert preflight["hardware_block"]["compute_class"] == "CPU sufficient"
    assert preflight["machine"]["cpu_count"]
