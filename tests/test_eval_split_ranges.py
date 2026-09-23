"""The range arm of R2C-066 (eval_split_range_overlap).

The alias arm catches the same EXPRESSION under two roles; this arm
catches different expressions whose index WINDOWS provably overlap —
the night3 notebook's shape: train on the full history, condition the
forecast on the full history, score against a slice of it.
"""

import ast

import pytest

from scripts.eval_split_ranges import find_notebook_split_leakage

PARAMS = {"forecast_horizon": 26, "context_length": 10}


def _findings(source: str, params=None):
    return find_notebook_split_leakage(
        ast.parse(source), PARAMS if params is None else params)


def test_the_night3_shape_train_and_conditioning_leak_into_the_metrics():
    """Known-bad, the 2026-08-05 night3 notebook reduced: full-history
    training and conditioning, metrics on an in-history window whose
    bounds fold from params (K known) and an opaque T_total symbol."""
    findings = _findings(
        "K = cfg['forecast_horizon']\n"
        "demand_tensor = torch.tensor(demand_history, dtype=torch.float32)\n"
        "model = train_model(model, demand_history=demand_tensor, seed=SEED)\n"
        "result = forecast(model=model, historical_demand=demand_history)\n"
        "test_start = T_total - K\n"
        "test_actual = demand_history[:, test_start:T_total]\n"
        "test_predicted = result.predicted_means.numpy()\n"
        "rmse = np.sqrt(np.mean((test_actual - test_predicted) ** 2))\n"
        "wmape = np.sum(np.abs(test_actual - test_predicted)) / np.sum(np.abs(test_actual))\n"
    )
    roles = {f.fit_role for f in findings}
    assert {f.root for f in findings} == {"demand_history"}
    assert "training" in roles
    assert "the inference call's conditioning context" in roles
    assert all("provably overlap" in f.message() for f in findings)


def test_a_proper_holdout_protocol_is_silent():
    """Known-good: training and conditioning end where the evaluation
    window begins."""
    findings = _findings(
        "K = cfg['forecast_horizon']\n"
        "split = T_total - K\n"
        "train_model(model, demand_history=demand_history[:, :split])\n"
        "result = forecast(model, historical_demand=demand_history[:, :split])\n"
        "test_actual = demand_history[:, split:T_total]\n"
        "mae = np.mean(np.abs(test_actual - result.predicted_means.numpy()))\n"
    )
    assert findings == []


def test_a_callee_internal_split_is_silent_by_construction():
    """Negative pin, the boundary: both reads are whole-tensor, so the
    split may live inside the callee and this seam cannot see it."""
    findings = _findings(
        "train_model(model, demand_history)\n"
        "score = evaluate_model(model, demand_history)\n"
    )
    assert findings == []


def test_symbolic_loop_windows_prove_nothing_and_stay_silent():
    """Negative pin: a training window over a loop variable cannot be
    compared with the evaluation window."""
    findings = _findings(
        "K = cfg['forecast_horizon']\n"
        "for t in range(T_total - K):\n"
        "    train_model(model, demand_history[:, t:t + K])\n"
        "test_actual = demand_history[:, T_total - K:T_total]\n"
        "rmse = np.mean((test_actual - preds) ** 2)\n"
    )
    assert findings == []


def test_the_model_argument_is_plumbing_not_data():
    findings = _findings(
        "train_model(model, x_train, y_train)\n"
        "acc = evaluate(model, x_test[:100], y_test[:100])\n"
    )
    assert findings == []


def test_distinct_roots_never_join():
    findings = _findings(
        "train_model(model, y_train[:800])\n"
        "rmse = np.mean((y_test[:200] - preds) ** 2)\n"
    )
    assert findings == []


def test_wraps_propagate_the_root_through_tensor_conversion():
    """torch.tensor / .astype / .float chains keep the root identity."""
    findings = _findings(
        "K = cfg['forecast_horizon']\n"
        "data = torch.tensor(series.astype('float32')).float()\n"
        "train_model(model, data)\n"
        "rmse = np.mean((series[T - K:T] - preds) ** 2)\n"
    )
    assert [f.root for f in findings] == ["series"]


def test_eval_verb_calls_count_as_evaluation_consumers():
    findings = _findings(
        "K = cfg['forecast_horizon']\n"
        "train_model(model, series)\n"
        "evaluate_model(model, series[T - K:T])\n"
    )
    assert [f.root for f in findings] == ["series"]
    assert findings[0].fit_role == "training"


def test_known_bounds_prove_disjointness_without_symbols():
    """Fully numeric bounds: [0, 800) vs [800, 1000) is disjoint."""
    findings = _findings(
        "train_model(model, series[:, 0:800])\n"
        "rmse = np.mean((series[:, 800:1000] - preds) ** 2)\n",
        params={},
    )
    assert findings == []


def test_known_bounds_prove_overlap_without_symbols():
    findings = _findings(
        "train_model(model, series[:, 0:900])\n"
        "rmse = np.mean((series[:, 800:1000] - preds) ** 2)\n",
        params={},
    )
    assert [f.root for f in findings] == ["series"]


def test_split_helper_calls_are_not_training_consumers():
    """train_test_split carries both verbs and produces splits."""
    findings = _findings(
        "x_train, x_test = train_test_split(series[:, 0:1000])\n"
        "rmse = np.mean((series[:, 800:1000] - preds) ** 2)\n",
        params={},
    )
    assert findings == []


def test_a_rebound_name_stays_its_own_root_and_still_joins():
    """The real night3 notebook rebinds demand_history through a
    list-indexed subscript (`demand_demo[:, [ ... ]]`); a binding with
    wholly-unknown ranges must not poison later reads of the name."""
    findings = _findings(
        "K = cfg['forecast_horizon']\n"
        "demand_history = demand_demo[:, [all_weeks.index(w) for w in ws]]\n"
        "train_model(model, torch.tensor(demand_history))\n"
        "test_actual = demand_history[:, T_total - K:T_total]\n"
        "rmse = np.mean((test_actual - preds) ** 2)\n"
    )
    assert [f.root for f in findings] == ["demand_history"]


@pytest.mark.manual_only
def test_the_night3_notebook_itself_flags_and_the_fleet_is_otherwise_quiet():
    """The real artifact as known-bad, the delivered fleet as the noise
    floor: nothing outside the night3 delivery may flag. Run dirs get
    archived under _N suffixes, so the pin is on the (deduped) slug.
    Corpus: the live fleet plus the committed example_runs/ floor; empty
    corpus skips instead of passing vacuously (W3 verdicts §5)."""
    import json
    import re
    from pathlib import Path

    scanned = 0
    flagged: set[str] = set()
    flagged_dirs: set[str] = set()
    fleet_notebooks = [
        nb_path
        for root in (Path("r2c_runs"), Path("example_runs"))
        for nb_path in sorted(root.glob("*/notebook.ipynb"))
    ]
    for nb_path in fleet_notebooks:
        run_dir = nb_path.parent
        params: dict = {}
        params_path = run_dir / "method" / "params.json"
        if not params_path.is_file():
            params_path = run_dir / ".pipeline" / "params.json"
        if params_path.is_file():
            try:
                loaded = json.loads(params_path.read_text(encoding="utf-8"))
                if isinstance(loaded, dict):
                    values = loaded.get("params", loaded)
                    if isinstance(values, dict):
                        params = {
                            k: (v.get("value") if isinstance(v, dict) else v)
                            for k, v in values.items()}
            except (OSError, ValueError):
                params = {}
        try:
            nb = json.loads(nb_path.read_text(encoding="utf-8"))
            source = "\n".join(
                "\n".join(line for line in "".join(c.get("source", []))
                          .splitlines()
                          if not line.lstrip().startswith(("%", "!")))
                for c in nb.get("cells", [])
                if c.get("cell_type") == "code")
            tree = ast.parse(source)
        except (OSError, ValueError, SyntaxError):
            continue
        scanned += 1
        if find_notebook_split_leakage(tree, params):
            flagged.add(re.sub(r"_\d+$", "", run_dir.name))
            flagged_dirs.add(run_dir.name)

    if scanned == 0:
        pytest.skip("live fleet absent")
    known_bad = {"probabilistic-demand-forecasting-with-graph-neural-networks"}
    assert flagged <= known_bad, flagged
    # The night3 delivery must flag ITSELF, not merely share a slug with a
    # roll that does. The 2026-08-06 loop2 roll archived night3 to the _7
    # suffix and delivered a properly held-out split in the unsuffixed dir
    # (which this check correctly does NOT flag), so the known-bad pointer
    # follows the archived copy while it stays on disk.
    night3 = Path("r2c_runs/probabilistic-demand-forecasting-with-graph"
                  "-neural-networks_7/notebook.ipynb")
    if night3.is_file():
        assert night3.parent.name in flagged_dirs, flagged_dirs


def test_train_labeled_metric_on_trained_data_is_a_fit_diagnostic_not_a_leak():
    """GBALD 2026-09-01 stage-3c false positive: the notebook trains on the
    core-set indices and then reports `warmup_train_acc` on those same
    indices — a train-split diagnostic mandated by the paper's
    train-to-99%-accuracy protocol — with the real test metric on held-out
    x_test one line below. A metric name that declares the TRAIN split must
    not register its reads as evaluation; a name claiming the eval split
    (`test_acc`) over the same trained window still flags."""
    trained_then_train_acc = (
        "idx = select_core_set(x_pool, budget)\n"
        "model = train_from_scratch(model, x_pool[0:100], y_pool[0:100])\n"
        "warmup_train_acc = (model(x_pool[0:100]).argmax(1)"
        " == y_pool[0:100]).float().mean().item()\n"
    )
    assert _findings(trained_then_train_acc, {}) == []

    trained_then_test_acc = (
        "model = train_from_scratch(model, x_pool[0:100], y_pool[0:100])\n"
        "test_acc = (model(x_pool[0:100]).argmax(1)"
        " == y_pool[0:100]).float().mean().item()\n"
    )
    assert _findings(trained_then_test_acc, {}) != []
