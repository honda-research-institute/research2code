"""R2C-066 — a delivered demo must not report metrics on its training data.

Both known-bads come from delivered pdfgnn runs. The 2026-08-05 shape is the
one this static arm owns: one target tensor handed to the trainer and to the
evaluator, so the reported RMSE, MAE and WMAPE are training fit. The
2026-08-04 shape (validation WINDOWS overlapping training windows while the
expressions genuinely differ) is index-range overlap rather than expression
aliasing, and is covered here only by the negative test that pins what this
arm does not claim to catch.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

from scripts.eval_split_integrity import find_split_aliasing


def _findings(source: str):
    return find_split_aliasing(ast.parse(source))


def test_the_delivered_shape_is_caught_and_names_both_roles():
    findings = _findings(
        "def train_and_forecast(series, horizon):\n"
        "    target_t = series[:, 10:10 + horizon]\n"
        "    model = train_model(net, features, target_t, max_epochs=50)\n"
        "    return evaluate_model(model, features, target_t)\n"
    )
    assert len(findings) == 1
    finding = findings[0]
    assert finding.expression == "target_t"
    assert {finding.first_role, finding.second_role} == {
        "training supervision", "the reported evaluation"}
    message = finding.message("method/method.py")
    assert "train_model" in message and "evaluate_model" in message
    assert "training supervision data range" in message
    assert "reported evaluation data range" in message
    assert "Required protocol-axis overlap: empty" in message
    assert "split arithmetic or slice bounds" in message
    assert "target_t" not in message
    assert "DIFFERENT ground-truth tensor" not in message
    assert "eval_split_aliasing" in message


def test_a_validation_loss_reading_the_training_targets_is_caught():
    """Early stopping watching the training objective is the same class."""
    findings = _findings(
        "def train_model(model, features, target, patience=5):\n"
        "    for _ in range(10):\n"
        "        loss = student_t_nll(model(features), target)\n"
        "        loss.backward()\n"
        "        val_loss = student_t_nll(model(features), target)\n"
        "    return val_loss\n"
    )
    assert len(findings) == 1
    finding = findings[0]
    assert finding.expression == "target"
    assert {finding.first_role, finding.second_role} == {
        "training supervision", "model selection"}
    message = finding.message("method/training.py")
    assert "training supervision data range" in message
    assert "model selection data range" in message
    assert "`target`" not in message


def test_disjoint_slices_are_clean():
    findings = _findings(
        "def train_and_forecast(series, horizon, val_start, val_end):\n"
        "    train_y = series[:, :val_start]\n"
        "    val_y = series[:, val_start:val_end]\n"
        "    test_y = series[:, val_end:val_end + horizon]\n"
        "    model = train_model(net, features, train_y, val_targets=val_y)\n"
        "    return evaluate_model(model, features, test_y)\n"
    )
    assert findings == []


def test_sharing_features_and_the_model_across_roles_is_fine():
    """Only ground truth must be disjoint; inputs are shared by design."""
    findings = _findings(
        "def run(features, adjacency, train_y, test_y):\n"
        "    model = train_model(features, adjacency, train_y)\n"
        "    return evaluate_model(model, features, adjacency, test_y)\n"
    )
    assert findings == []


def test_a_loss_function_named_score_is_not_an_evaluation_call():
    """The ms3d delivery's shape, verified as a false positive: a loss whose
    name carries an eval verb is still supervision."""
    findings = _findings(
        "def train(model, preds, target_scores):\n"
        "    loss_score = score_loss_fn(preds, target_scores)\n"
        "    loss_score.backward()\n"
        "    return loss_score\n"
    )
    assert findings == []


def test_a_slice_bound_cannot_decide_what_a_tensor_is():
    """`pred_scores[:, :n_gt]` is a PREDICTION whose slice bound happens to
    carry a ground-truth token. Keying on every identifier in the expression
    read it as ground truth (found on the ms3d delivery); keying on the
    expression's root name does not."""
    from scripts.eval_split_integrity import _is_target_expression, _root_name

    node = ast.parse("f(pred_scores[:, :n_gt])").body[0].value.args[0]
    assert _root_name(node) == "pred_scores"
    assert not _is_target_expression(node)


def test_a_strong_marker_wins_and_a_weak_one_yields_to_predictions():
    """`forecast_targets` is ground truth despite `forecast`; `y_pred` is not
    ground truth despite `y`."""
    from scripts.eval_split_integrity import _is_target_expression

    def _arg(src: str):
        return ast.parse(f"f({src})").body[0].value.args[0]

    assert _is_target_expression(_arg("forecast_targets"))
    assert _is_target_expression(_arg("labels.detach()"))
    assert _is_target_expression(_arg("y[a:b]"))
    assert not _is_target_expression(_arg("y_pred"))
    assert not _is_target_expression(_arg("mu_np"))


def test_a_train_val_split_helper_is_not_a_role():
    findings = _findings(
        "def run(y):\n"
        "    train_y, val_y = train_val_split(y)\n"
        "    model = fit(train_y)\n"
        "    return evaluate(model, val_y)\n"
    )
    assert findings == []


def test_the_2026_08_04_window_overlap_shape_is_out_of_this_arm_s_scope():
    """Pinned so the boundary is explicit rather than assumed: distinct
    expressions over overlapping index ranges need the runtime arm."""
    findings = _findings(
        "def train_model(demand, P, K, T, val_start):\n"
        "    for start in range(T - P - K):\n"
        "        batch_targets = demand[:, start + P:start + P + K]\n"
        "        loss = nll(model(demand), batch_targets)\n"
        "    for start_v in range(T - val_start - K):\n"
        "        batch_targets_v = demand[:, start_v + P:start_v + P + K]\n"
        "        val_loss = nll(model(demand), batch_targets_v)\n"
        "    return val_loss\n"
    )
    assert findings == []


@pytest.mark.manual_only
def test_the_delivered_fleet_flags_no_new_aliasing_class():
    """Noise floor measured against every delivered producer file: nothing
    outside the one known bad (the 2026-08-05 day roll's `target_t` handed
    to both the trainer and the evaluator). Run dirs get archived under
    `_N` suffixes and deleted before re-rolls, so the pin is on the defect
    class, never on a dir name or an exact count — the positive detection
    is pinned by the synthetic fixtures above. Corpus: the live fleet plus
    the committed example_runs/ floor; empty corpus skips instead of
    passing vacuously (W3 verdicts §5)."""
    import re

    scanned = 0
    flagged: list[str] = []
    for root in (Path("r2c_runs"), Path("example_runs")):
        for path in sorted(root.glob("*/method/*.py")):
            if path.name not in ("method.py", "training.py", "model.py"):
                continue
            try:
                tree = ast.parse(path.read_text(encoding="utf-8"))
            except (OSError, SyntaxError, ValueError):
                continue
            scanned += 1
            for finding in find_split_aliasing(tree):
                slug = re.sub(r"_\d+$", "", path.parent.parent.name)
                flagged.append(f"{slug}/{path.name}:{finding.expression}")
    if scanned == 0:
        pytest.skip("live fleet absent")

    known_bad = {
        "probabilistic-demand-forecasting-with-graph-neural-networks/"
        "method.py:target_t"
    }
    assert set(flagged) <= known_bad, flagged


def test_the_check_is_wired_into_the_method_coder_seam(tmp_path):
    """The finding must reach the fix loop, which means the validator's own
    error list, not just the primitive."""
    from tests.helpers.inject import minimal_method_spec

    run_dir = tmp_path / "run"
    (run_dir / "method").mkdir(parents=True)
    (run_dir / ".pipeline").mkdir()
    spec = minimal_method_spec(paradigm_id="active_learning")
    (run_dir / "method" / "method.py").write_text(
        "def select_batch(model, x_unlabeled, batch_size, *, seed):\n"
        "    labels = x_unlabeled[:batch_size]\n"
        "    trained = train_model(model, labels)\n"
        "    evaluate_model(trained, labels)\n"
        "    return list(range(batch_size))\n", encoding="utf-8")

    from scripts.validate_method_coder_output import validate

    errors, _ = validate(spec, run_dir, Path.cwd())
    assert any("eval_split_aliasing" in e for e in errors), errors
    assert json.dumps(spec)  # spec stays JSON-safe for the fix-loop payload
