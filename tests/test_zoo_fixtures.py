"""Pin the bug-zoo fixtures to their documented defect/health properties.

tests/fixtures/zoo/ is the acceptance contract for the verification work
(the recentering plan (internal, not shipped) Section 1): known-bads must keep their bugs, the
known-goods must keep their health. These tests pin the FIXTURES themselves —
not any probe — so an accidental regeneration, "helpful fix", or schema sweep
that silently alters the evidence fails CI immediately. The expected-flags
source of truth is tests/fixtures/zoo/README.md; each assertion cites the
finding ID it preserves.

(Standing lesson behind this file: the original M-002/M-003/M-004 artifacts
were lost within days because nothing pinned them.)
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

ZOO = Path(__file__).resolve().parent / "fixtures" / "zoo"
EVIDENCE = Path(__file__).resolve().parent / "fixtures" / "evidence"


def _cell_source(nb_path: Path, cell_id: str) -> str:
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    for cell in nb["cells"]:
        if cell.get("id") == cell_id:
            return "".join(cell["source"])
    raise AssertionError(f"cell {cell_id} not found in {nb_path}")


def _all_code_source(nb_path: Path) -> str:
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    return "\n".join(
        "".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"
    )


# ---------------------------------------------------------------------------
# gbald-lr74-never-learns (frozen evidence)
# ---------------------------------------------------------------------------


def test_gbald_params_keep_lr74_and_r0_2000():
    params = json.loads(
        (ZOO / "gbald-lr74-never-learns" / "params.json").read_text()
    )["params"]
    # The flagship silent failure: the paper's section number as learning rate,
    # labeled as an Adam convention.
    assert params["learning_rate"]["value"] == 7.4
    assert "adam" in params["learning_rate"]["reasoning"].lower()
    # R_0 at the unrescaled paper value. Here it is source=spec_default with no
    # spec declaration, so the *static* scale-mismatch probe (US-4) cannot key
    # on this dir — its degeneracy is caught behaviorally (UB-2/UB-7). The
    # static US-4 fail-side is gbald-r0-scale-mismatch/ (pinned below).
    assert params["R_0"]["value"] == 2000.0
    # A-001 class: paper-sourced label whose own locator admits a convention.
    dropout = params["dropout_rate"]
    assert dropout["source"] == "paper"
    assert "convention" in dropout.get("paper_section", "").lower()


def test_gbald_notebook_keeps_chance_level_outputs():
    nb = json.loads(
        (ZOO / "gbald-lr74-never-learns" / "notebook.ipynb").read_text()
    )
    text = json.dumps(nb)
    # Executed outputs ARE the evidence: chance-level accuracies and the
    # constant geometric-prior demo must stay embedded.
    assert "1.0000" in text  # degenerate prior printed for every sample
    assert nb["cells"], "executed outputs stripped — fixture regenerated?"


def test_gbald_r0_scale_mismatch_fixture_keeps_evidence():
    """gbald-r0-scale-mismatch: the self-contained US-4 fail-side. All three
    inputs must stay aligned — the spec declaration, the unrescaled shipped
    value, and the [0,1] data scale — or the scale-mismatch acceptance is
    no longer exercised."""
    scenario = ZOO / "gbald-r0-scale-mismatch"
    params = json.loads((scenario / ".pipeline" / "params.json").read_text())["params"]
    # Shipped unrescaled, labeled paper-sourced — the defect under test.
    assert params["R_0"]["value"] == 2000.0
    assert params["R_0"]["source"] == "paper"
    # The spec declares R_0 scale-dependent at raw-pixel calibration.
    spec = json.loads((scenario / ".pipeline" / "method_spec.json").read_text())
    sdh = spec["critical_requirements"]["scale_dependent_hyperparameters"][0]
    assert sdh["name"] == "R_0"
    assert sdh["paper_value"] == 2000.0
    assert sdh["assumes_data_scale"] == "raw_pixel_unnormalized"
    # The delivered data is genuinely [0,1]-normalized (the mismatch).
    data = json.loads(
        (scenario / "method" / "example_data" / "mnist_subset.json").read_text()
    )
    flat = [v for row in data["x_pool"] for v in row]
    assert min(flat) >= 0.0 and max(flat) <= 1.0


# ---------------------------------------------------------------------------
# pdwa-omega-degenerate (frozen evidence)
# ---------------------------------------------------------------------------


def test_pdwa_method_keeps_omega_degeneracy_and_plus_penalties():
    src = (ZOO / "pdwa-omega-degenerate" / "method.py").read_text()
    # M-005a: the objective accepts omega...
    assert "def evaluate_objective_function" in src
    obj = src.split("def evaluate_objective_function", 1)[1]
    header = obj.split(")", 1)[0]
    assert "omega" in header
    # ...and the seeded random tie-break that omega-degeneracy produces.
    assert "tie" in src.lower() or "rng" in src or "random" in src.lower()


def test_pdwa_notebook_keeps_literal_collision_claim_and_moving_obstacles():
    src = _all_code_source(ZOO / "pdwa-omega-degenerate" / "notebook.ipynb")
    # US-7 fail-side: result-claiming literal print with no collision check.
    assert "Collision-free: Yes" in src
    # This frozen snapshot is the M-004 PASS-side: obstacles DO move here
    # (the fail-side is the reconstruction below).
    assert "obs['x'] +=" in src


# ---------------------------------------------------------------------------
# badge-goodmethod-badnotebook (frozen evidence — both files known-good)
# ---------------------------------------------------------------------------


def test_badge_notebook_keeps_correct_loop():
    src = _cell_source(
        ZOO / "badge-goodmethod-badnotebook" / "notebook.ipynb", "c53ec107"
    )
    # The corrected loop is the PASS-side guard for AL probes: global-index
    # mapping, set-difference bookkeeping, fresh model per round.
    assert "unlabeled_idx[np.asarray(batch_positions)]" in src
    assert "np.setdiff1d" in src
    loop = src.split("# Acquisition loop", 1)[1]
    assert "build_model" in loop and "train_from_scratch" in loop


# ---------------------------------------------------------------------------
# Reconstruction mutants (synthetic, built 2026-06-10)
# ---------------------------------------------------------------------------


def test_badge_badloop_mutant_keeps_m002_m003_shapes():
    src = _cell_source(
        ZOO / "badge-badloop-reconstructed" / "notebook.ipynb", "c53ec107"
    )
    # M-002: positional-offset arithmetic + contiguous-prefix assumption +
    # no set-difference.
    assert "i - len(initial_indices)" in src
    assert "x_pool[len(initial_indices):]" in src
    assert "setdiff1d" not in src
    # M-003: one persisted model — no build_model inside the round loop,
    # while the narrative still claims retraining.
    loop = src.split("# Acquisition loop", 1)[1]
    assert "build_model" not in loop
    assert "train_from_scratch" in loop
    assert "Retrain from scratch" in src


def test_pdwa_frozen_obstacles_mutant_keeps_m004_shape():
    src = _cell_source(
        ZOO / "pdwa-frozen-obstacles-reconstructed" / "notebook.ipynb",
        "e78a615b",
    )
    # M-004: obstacles set once, never updated — with the dynamic narrative
    # intact (the claims-vs-behavior gap is the property under test).
    assert "obs['x'] +=" not in src
    assert "x_prev'], obs['y_prev']" not in src.replace('"', "'")
    assert "dynamic obstacles" in src.lower()
    assert "select_velocity" in src


def test_kd_mutant_docstring_claim_stays_wrong():
    src = (ZOO / "kd-wrong-op-reconstructed" / "method.py").read_text()
    # M-001: confident shape claim + the wrong op, both load-bearing.
    assert "(B, N_queries, 1)" in src
    assert "torch.diag_embed" in src


def test_kd_mutant_shape_mismatch_is_real_and_silent():
    torch = pytest.importorskip("torch")
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "kd_mutant", ZOO / "kd-wrong-op-reconstructed" / "method.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    s, t = torch.randn(2, 5, 8), torch.randn(2, 5, 8)
    w = mod.compute_instance_weights(s, t)
    # Claimed (B, N, 1); the actual rank-4 result is the M-001 evidence.
    assert tuple(w.shape) == (2, 5, 5, 5)
    # And the failure is silent: downstream loss is a finite scalar.
    loss = mod.weighted_feature_loss(s, t)
    assert loss.numel() == 1 and torch.isfinite(loss)


# ---------------------------------------------------------------------------
# june9 evidence run (fabricated-provenance fail-side)
# ---------------------------------------------------------------------------


def test_june9_run_keeps_fabricated_provenance_evidence():
    run = EVIDENCE / "june9-gbald-run"
    params = json.loads((run / "pipeline" / "params.json").read_text())["params"]
    paper = (run / "pipeline" / "paper.md").read_text(encoding="utf-8")
    tua = params["train_until_accuracy"]
    # source=paper with a quote the paper does not contain — the provenance
    # quote-match probe's real (non-synthetic) fail-side.
    assert tua["source"] == "paper"
    assert tua["value"] == 0.99
    assert "99%" not in paper


def test_negstride_selector_keeps_crash_and_noop_config():
    """gbald-negstride-selector: the 2026-06-10 validation-rerun evidence.

    Pins (1) the negative-stride indexing path at select_batch Step 2 —
    `np.argsort(...)[::-1]` feeding torch tensor indexing, which crashes on
    every non-empty pool; (2) the demo config that hid it (initial_labeled
    >= pool_size: the unlabeled pool was empty from round 0, so smoke
    "passed" without ever running the selector); (3) the before-state probe
    report whose three unprobeables motivated the AL-5 crash arm.
    """
    scenario = ZOO / "gbald-negstride-selector"
    method = (scenario / "method.py").read_text(encoding="utf-8")
    assert "np.argsort(bald_scores)[::-1][:batch_returns]" in method
    assert "x_candidates = x_unlabeled[top_b_indices]" in method
    # No contiguity rescue between the two lines — that's the bug.
    step2 = method.split("np.argsort(bald_scores)[::-1]")[1][:200]
    assert ".copy()" not in step2

    params = json.loads((scenario / "params.json").read_text())["params"]
    assert params["initial_labeled"]["value"] == 1000
    assert params["pool_size"]["value"] == 800

    report = json.loads((scenario / "probe_report.json").read_text())
    stride_hits = [v for v in report["verdicts"]
                   if "negative" in v.get("message", "")]
    assert {v["probe_id"] for v in stride_hits} == {"UB-7", "CT-1", "AL-5"}
    assert all(v["verdict"] == "unprobeable" for v in stride_hits)


def test_idbrrt_no_path_mutant_keeps_its_shape():
    torch = pytest.importorskip("torch")
    src = (ZOO / "idbrrt-no-path-reconstructed" / "method.py").read_text()
    # The defining evidence: an honest timeout result with NO trajectory and
    # zero optimization attempts, behind the goal-STATE signature that raises
    # the live size-mismatch error on a 2-dim goal.
    assert '"timeout"' in src and "optimization_attempts" in src
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "idbrrt_mutant", ZOO / "idbrrt-no-path-reconstructed" / "method.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    env = mod.Environment(
        name="t", state_bounds_lower=-10 * torch.ones(3),
        state_bounds_upper=10 * torch.ones(3), obstacles=[])
    dyn = mod.UnicycleDynamics()
    start = torch.zeros(3, dtype=torch.float64)
    with pytest.raises(RuntimeError):
        mod.plan(start, torch.tensor([4.0, 0.0]), env, dyn, 0)
    result = mod.plan(start, torch.tensor([4.0, 0.0, 0.0]), env, dyn, 0)
    assert result.trajectory is None
    assert result.status == "timeout"
    assert result.stats["optimization_attempts"] == 0
