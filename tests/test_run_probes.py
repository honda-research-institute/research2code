"""Tests for the one-command probe battery (scripts/run_probes.py).

Synthetic run dirs assembled from zoo pieces: the battery must aggregate the
tiers, gate on failures, write the chat-readable report, and stay graceful
(unprobeable, not crash) on missing artifacts.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
ZOO = REPO / "tests" / "fixtures" / "zoo"

from run_probes import main as run_probes_main  # noqa: E402
from run_probes import run_battery  # noqa: E402
import pytest  # noqa: E402

pytestmark = pytest.mark.probe_runtime


def _make_run(tmp_path: Path, *, params=None, paper="", notebook_src=None,
              paradigm="", pluggable="") -> Path:
    run = tmp_path / "run"
    (run / ".pipeline").mkdir(parents=True)
    (run / "method").mkdir()
    (run / "method" / "method.py").write_text(
        "def select_batch(model, x_unlabeled, batch_size, seed):\n"
        "    return list(range(batch_size))\n")
    if params is not None:
        (run / ".pipeline" / "params.json").write_text(
            json.dumps({"schema_version": "1.0.0", "params": params}))
    if paper:
        (run / ".pipeline" / "paper.md").write_text(paper)
    if notebook_src is not None:
        shutil.copy(notebook_src, run / "notebook.ipynb")
    if paradigm:
        (run / ".pipeline" / "method_spec.json").write_text(json.dumps({
            "comparison": {"classification": {"id": paradigm},
                           "pluggable_component": {"name": pluggable}}}))
    return run


def test_al_probe_dispatch_is_driven_by_the_taxonomy_node():
    """Phase 1.2: the populated active_learning node declares which behavioural
    probes apply (via semantic_checks[].probe). The harness reads that set
    rather than a hardcoded list, and a still-`reserved` paradigm returns None
    (caller falls back to the full battery)."""
    from run_probes import _node_declared_probe_refs, _probe_enabled

    refs = _node_declared_probe_refs("active_learning")
    assert refs == {
        "al_loop.microharness",
        "al_loop.fresh_retrain",
        "al_loop.eval_label_alignment",
        "al_loop.acquisition_contract",
        "al_loop.demo_config_reachability",
        "al_stage1.coreset_construction",
        "claims.contribution_floor",
    }
    # Sub-variants inherit the parent's probe declarations.
    assert refs <= _node_declared_probe_refs("active_learning/bayesian")
    # Motion-planning is now served and declares its MP probe refs through
    # the same taxonomy seam — including the scenario-dimension detector
    # bindings (R2C-025 slice B), which ride the same declared context.
    assert _node_declared_probe_refs("motion_planning") == {
        "motion_planning.scenario_dynamics",
        "motion_planning.steering_responsiveness",
        "motion_planning.goal_progress",
        "motion_planning.scenario_geometry",
        "motion_planning.scenario_population",
        "motion_planning.scenario_dynamics_setup",
    }
    # A still-reserved paradigm is not served -> None -> full-battery fallback.
    assert _node_declared_probe_refs("adaptive_moment_optimizer") is None
    assert _probe_enabled("al_loop.microharness", refs) is True
    assert _probe_enabled("al_loop.microharness", None) is True          # fallback
    assert _probe_enabled("al_loop.not_declared", refs) is False


def test_battery_gates_on_known_bad_artifacts(tmp_path):
    # gbald lr=7.4 params + the chance-level executed notebook.
    run = _make_run(
        tmp_path,
        params=json.loads(
            (ZOO / "gbald-lr74-never-learns" / "params.json").read_text()
        )["params"],
        paper="some paper text",
        notebook_src=ZOO / "gbald-lr74-never-learns" / "notebook.ipynb",
    )
    report = run_battery(run)
    failed_probes = {v.probe_id for v in report.gating_failures}
    assert "US-1" in failed_probes   # lr=7.4
    assert "UB-6" in failed_probes   # chance-level curve
    rc = run_probes_main(["--run-dir", str(run)])
    assert rc == 1
    written = json.loads((run / ".pipeline" / "probe_report.json").read_text())
    assert written["counts"]["fail"] >= 2


def test_battery_clean_run_exits_zero(tmp_path):
    run = _make_run(
        tmp_path,
        params={"learning_rate": {"value": 0.001, "source": "system_inferred",
                                  "reasoning": "standard default"}},
        paper="a paper",
    )
    report = run_battery(run)
    assert report.gating_failures == []
    # No notebook -> UB-6 unprobeable, never a silent pass.
    assert any(v.probe_id == "UB-6" and v.verdict == "unprobeable"
               for v in report.verdicts)
    assert run_probes_main(["--run-dir", str(run)]) == 0


def test_battery_malformed_notebook_is_unprobeable_not_crash(tmp_path):
    run = _make_run(tmp_path)
    (run / "notebook.ipynb").write_text("not json", encoding="utf-8")

    report = run_battery(run)

    ub6 = [v for v in report.verdicts if v.probe_id == "UB-6"]
    assert ub6 and ub6[0].verdict == "unprobeable"
    assert "could not be read" in ub6[0].message


def test_battery_emits_us8_for_stale_notebook_prose(tmp_path):
    run = _make_run(
        tmp_path,
        params={"R_0": {"value": 7.843}},
    )
    (run / "notebook.ipynb").write_text(json.dumps({
        "cells": [{
            "cell_type": "markdown",
            "source": "This notebook still says `R_0=2000.0`.",
        }],
        "metadata": {},
        "nbformat": 4,
        "nbformat_minor": 5,
    }) + "\n")

    report = run_battery(run)

    us8 = [v for v in report.verdicts if v.probe_id == "US-8"]
    assert len(us8) == 1
    assert us8[0].verdict == "flag_for_researcher"


def test_battery_runs_paradigm_tier_from_spec(tmp_path):
    # motion_planning spec routes the MP-1 probe, with the pluggable name
    # taken from the spec (select_velocity), not from defaults.
    run = _make_run(
        tmp_path,
        params={},
        notebook_src=ZOO / "pdwa-frozen-obstacles-reconstructed" / "notebook.ipynb",
        paradigm="motion_planning",
        pluggable="select_velocity",
    )
    report = run_battery(run)
    mp1 = [v for v in report.verdicts if v.probe_id == "MP-1"]
    assert mp1 and mp1[0].verdict == "fail"
    assert mp1[0].finding_class == "M-004"


def test_battery_missing_run_dir_is_an_invocation_error(tmp_path):
    assert run_probes_main(["--run-dir", str(tmp_path / "nope")]) == 2


def test_battery_surfaces_scale_dependent_tier(tmp_path):
    """The battery runs the scale-dependent tier: US-4 (static scale-vs-
    calibration) and US-4b (behavioral ranking scale-invariance). On the
    R_0=2000-on-normalized-data fixture US-4 fails; US-4b also reports (the
    fixture has no importable selector, so it is unprobeable there — the
    fail/pass behavior is covered in test_scale_mismatch_probe)."""
    run = tmp_path / "run"
    shutil.copytree(ZOO / "gbald-r0-scale-mismatch", run)
    report = run_battery(run)
    by_id = {v.probe_id for v in report.verdicts}
    assert "US-4" in by_id, [v.probe_id for v in report.verdicts]
    assert "US-4b" in by_id, [v.probe_id for v in report.verdicts]
    us4 = [v for v in report.verdicts if v.probe_id == "US-4"][0]
    assert us4.verdict == "fail" and us4.finding_class == "M-004"


def test_battery_runs_objective_probes_for_mp_method_package(tmp_path):
    import pytest
    pytest.importorskip("torch")
    run = _make_run(tmp_path, params={}, paradigm="motion_planning",
                    pluggable="plan")
    (run / "method" / "__init__.py").write_text(
        (ZOO / "pdwa-jterms-fresh" / "method.py").read_text())
    report = run_battery(run)
    verdicts = {v.probe_id: v for v in report.verdicts}
    # Added penalties in a maximized objective: strict direction inversion.
    assert verdicts["UB-4"].verdict == "fail"
    # Four of six j-terms consume no decision variable: ranking-inert.
    assert verdicts["UB-7"].verdict == "flag_for_researcher"
    assert verdicts["UB-7"].finding_class == "M-005"


def test_battery_runs_al_selector_perturbation_from_spec(tmp_path):
    import pytest
    pytest.importorskip("torch")
    run = _make_run(tmp_path, params={}, paradigm="active_learning/bayesian",
                    pluggable="select_batch")
    (run / "method" / "__init__.py").write_text(
        "import numpy as np\n"
        "\n"
        "\n"
        "def dead_prior(x):\n"
        "    return np.ones(len(x))\n"
        "\n"
        "\n"
        "def utility(x):\n"
        "    return np.asarray(x.sum(axis=1))\n"
        "\n"
        "\n"
        "def select_batch(x_unlabeled, batch_size, seed):\n"
        "    s = utility(x_unlabeled.numpy()) * dead_prior(x_unlabeled)\n"
        "    return [int(i) for i in np.argsort(s)[-batch_size:]]\n")
    report = run_battery(run)
    ub7 = [v for v in report.verdicts if v.probe_id == "UB-7"]
    assert ub7 and ub7[0].verdict == "flag_for_researcher"
    assert "dead_prior" in ub7[0].evidence.split("alive=")[0]
    # UB-5 has no build_model/train_from_scratch here: honest unprobeable.
    ub5 = [v for v in report.verdicts if v.probe_id == "UB-5"]
    assert ub5 and ub5[0].verdict == "unprobeable"
    # CT-1 rides the same arm: data-driven selection beats the null floor.
    ct1 = [v for v in report.verdicts if v.probe_id == "CT-1"]
    assert ct1 and ct1[0].verdict == "pass"


def test_battery_runs_kd_tier_from_spec(tmp_path):
    import pytest
    pytest.importorskip("torch")
    run = _make_run(tmp_path, params={}, paradigm="knowledge_distillation",
                    pluggable="compute_distillation_loss")
    (run / "method" / "method.py").write_text(
        "def compute_distillation_loss(student_logits, teacher_logits,"
        " temperature=2.0):\n"
        "    del teacher_logits, temperature\n"
        "    return (student_logits ** 2).mean()\n")
    report = run_battery(run)
    kd = {v.probe_id: v.verdict for v in report.verdicts
          if v.probe_id.startswith("KD")}
    # Teacher-ignoring loss with an inert temperature: both probes fail.
    assert kd.get("KD-1") == "fail", report.to_dict()
    assert kd.get("KD-2") == "fail", report.to_dict()


# A miniature model-and-batch KD package (the bev-distill interface shape);
# mirrors tests/test_kd_probes.py's kit fixtures.
_KD_KIT_LOSS_SRC = '''
import torch
from torch import nn


class _Det(nn.Module):
    def __init__(self, num_classes=3, hidden_dim=8, num_queries=5):
        super().__init__()
        self.lin = nn.Linear(3, hidden_dim)
        self.cls = nn.Linear(hidden_dim, num_classes)
        self.box = nn.Linear(hidden_dim, 4)
        self.num_queries = num_queries

    def forward(self, x):
        if x.dim() == 4:
            x = x.flatten(2).transpose(1, 2)
        h = torch.tanh(self.lin(x[:, : self.num_queries, :]))
        return {"pred_logits": self.cls(h),
                "pred_boxes": self.box(h).abs() + 0.1,
                "pred_features": h}


def build_student(num_classes, hidden_dim=8, num_queries=5):
    return _Det(num_classes, hidden_dim, num_queries)


def build_teacher(num_classes, hidden_dim=8, num_queries=5):
    return _Det(num_classes, hidden_dim, num_queries)


def compute_distillation_loss(student, teacher, batch, seed, tau=0.07):
    s = student(batch["student_inputs"])
    with torch.no_grad():
        t = teacher(batch["teacher_inputs"])
    feat = ((s["pred_features"] - t["pred_features"]) ** 2).mean()
    a = s["pred_features"].flatten(0, 1)
    b = t["pred_features"].flatten(0, 1)
    sim = (a @ b.t()) / tau
    nce = torch.logsumexp(sim, dim=-1).mean() - sim.diag().mean()
    return feat + nce
'''

_KD_KIT_ARCH_CONTRACT = {
    "pluggable_component": {
        "name": "compute_distillation_loss",
        "batch_dict_shape": {
            "student_inputs": "(B, 3, H_img, W_img)",
            "teacher_inputs": "(B, N_points, 3)",
            "targets": "list[dict] of length B, keys: boxes (N_obj, 4), "
                       "labels (N_obj,)",
        },
    },
}


def test_battery_runs_kd_kit_for_model_batch_losses(tmp_path):
    """A model-and-batch loss (the bev-distill shape) routes through the
    contract-driven kit: the battery imports the whole package, builds the
    models via its own builders, synthesizes the batch from
    arch_contract.json, and returns real verdicts (not unprobeable)."""
    import pytest
    pytest.importorskip("torch")
    run = _make_run(tmp_path, params={}, paradigm="knowledge_distillation",
                    pluggable="compute_distillation_loss")
    (run / "method" / "method.py").write_text(_KD_KIT_LOSS_SRC)
    (run / "method" / "__init__.py").write_text(
        "from .method import (build_student, build_teacher,\n"
        "                     compute_distillation_loss)\n")
    (run / ".pipeline" / "arch_contract.json").write_text(
        json.dumps(_KD_KIT_ARCH_CONTRACT))
    report = run_battery(run)
    kd = {v.probe_id: v.verdict for v in report.verdicts
          if v.probe_id.startswith("KD")}
    assert kd.get("KD-1") == "pass", report.to_dict()
    assert kd.get("KD-2") == "pass", report.to_dict()


def test_battery_kd_kit_unprobeable_reason_reaches_both_probes(tmp_path):
    """Kit construction failure (here: no arch contract to synthesize the
    batch from) discloses the same named reason on KD-1 and KD-2."""
    import pytest
    pytest.importorskip("torch")
    run = _make_run(tmp_path, params={}, paradigm="knowledge_distillation",
                    pluggable="compute_distillation_loss")
    (run / "method" / "method.py").write_text(_KD_KIT_LOSS_SRC)
    (run / "method" / "__init__.py").write_text(
        "from .method import (build_student, build_teacher,\n"
        "                     compute_distillation_loss)\n")
    report = run_battery(run)
    kd = {v.probe_id: v for v in report.verdicts
          if v.probe_id.startswith("KD")}
    assert kd["KD-1"].verdict == "unprobeable"
    assert "batch_dict_shape" in kd["KD-1"].message
    assert kd["KD-2"].verdict == "unprobeable"


def test_battery_reports_al6_without_method_dir(tmp_path):
    # AL-6 needs only params.json — a run halted between params derivation
    # and codegen must still get its reachability verdict (2026-06-11
    # adversarial-review catch: the method-dir guard silently skipped it).
    run = _make_run(
        tmp_path,
        params={"initial_labeled": {"value": 1000},
                "pool_size": {"value": 800}},
        paradigm="active_learning/bayesian",
        pluggable="select_batch",
    )
    shutil.rmtree(run / "method")
    report = run_battery(run)
    al6 = [v for v in report.verdicts if v.probe_id == "AL-6"]
    assert len(al6) == 1
    assert al6[0].verdict == "fail"
    assert "unreachable" in al6[0].message


# ---------------------------------------------------------------------------
# Isolated output mode (R2C-019): a post-delivery battery run must never
# rewrite delivery-time truth. The adversarial review proved the battery as
# built WROTE into the run's internal artifacts (it unconditionally rewrote
# the claims ledger when run against a run copy), so re-validation destroyed
# the baseline it needed for comparison.
# ---------------------------------------------------------------------------


def _make_ct3_run(tmp_path: Path) -> Path:
    """A run whose battery pass produces BOTH write-side artifacts (verdict
    report and CT-3 claims ledger)."""
    run = _make_run(tmp_path, params={
        "batch_size": {"value": 8, "source": "paper", "note": "stated."},
    }, paper="some paper text")
    (run / ".pipeline" / "paper_map.json").write_text(json.dumps({
        "elements": [{"id": "exp-1", "type": "experiment",
                      "source_text": "Method improves accuracy by 4%."}]}))
    return run


def _tree_hashes(root: Path) -> dict[str, str]:
    import hashlib
    return {
        p.relative_to(root).as_posix():
            hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(root.rglob("*")) if p.is_file()
    }


def test_output_dir_isolates_both_writes_and_run_stays_byte_identical(
        tmp_path):
    run = _make_ct3_run(tmp_path)
    out = tmp_path / "revalidation"
    before = _tree_hashes(run)

    rc = run_probes_main(["--run-dir", str(run), "--output-dir", str(out)])

    assert _tree_hashes(run) == before, (
        "isolated mode wrote into the run dir")
    report = json.loads((out / "probe_report.json").read_text())
    assert report["verdicts"], "verdict report missing from output dir"
    ledger = json.loads((out / "claims_ledger.json").read_text())
    assert ledger["claims"][0]["claim_id"] == "exp-1"
    assert rc in (0, 1)  # gating semantics unchanged by isolation


def test_output_dir_report_override_still_wins(tmp_path):
    run = _make_ct3_run(tmp_path)
    out = tmp_path / "revalidation"
    explicit = tmp_path / "elsewhere" / "report.json"
    run_probes_main(["--run-dir", str(run), "--output-dir", str(out),
                     "--report", str(explicit)])
    assert explicit.is_file()
    assert not (out / "probe_report.json").exists()
    # The ledger still follows the output dir — never the run dir.
    assert (out / "claims_ledger.json").is_file()
    assert not (run / ".pipeline" / "claims_ledger.json").exists()


def test_default_mode_write_paths_unchanged(tmp_path):
    run = _make_ct3_run(tmp_path)
    run_probes_main(["--run-dir", str(run)])
    assert (run / ".pipeline" / "probe_report.json").is_file()
    assert (run / ".pipeline" / "claims_ledger.json").is_file()
