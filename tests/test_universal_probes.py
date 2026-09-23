"""Zoo acceptance tests for the universal probes (UB-1/2/3/6/8).

Each test is a row of the zoo acceptance matrix executed for real: known-bad
artifacts must FAIL (or flag), known-good must PASS, unexecuted/unloadable
material must come back `unprobeable` rather than silently passing.

Pass-side note for UB-6: no executed known-good notebook exists yet (the
pipeline ships unexecuted notebooks until slice 2.3), so the pass side is a
synthetic executed notebook and the real june9 artifact pins the
`unprobeable` behavior instead.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parent.parent
ZOO = REPO / "tests" / "fixtures" / "zoo"
EVIDENCE = REPO / "tests" / "fixtures" / "evidence"

from probes.fixtures import make_planning_fixture  # noqa: E402
from probes.package_loader import load_module_from_path  # noqa: E402
from probes.universal import (  # noqa: E402
    descent_verdict,
    extract_loss_series,
    parse_return_shape_claim,
    probe_callable_sensitivity,
    probe_executed_notebook,
    probe_loss_descent,
    probe_seed_threading,
    probe_shape_claim,
)


def _verdict_of(verdicts, probe_id):
    matching = [v for v in verdicts if v.probe_id == probe_id]
    assert matching, f"no {probe_id} verdict in {[v.probe_id for v in verdicts]}"
    return matching[0]


# ---------------------------------------------------------------------------
# UB-6: executed-notebook sanity
# ---------------------------------------------------------------------------


def test_ub6_fails_the_chance_level_gbald_notebook():
    verdicts = probe_executed_notebook(
        ZOO / "gbald-lr74-never-learns" / "notebook.ipynb"
    )
    main = _verdict_of(verdicts, "UB-6")
    assert main.verdict == "fail"
    assert "chance" in main.message
    # And the degenerate 1.0000-x3 prior demo surfaces as a warn.
    warns = [v for v in verdicts if v.verdict == "warn"]
    assert any("1.0000" in v.message for v in warns), warns


def test_ub6_detects_n_classes_from_outputs():
    # "Classes: 10" appears in the gbald outputs, so no explicit n_classes
    # should be needed for the chance computation.
    verdicts = probe_executed_notebook(
        ZOO / "gbald-lr74-never-learns" / "notebook.ipynb", n_classes=None
    )
    assert _verdict_of(verdicts, "UB-6").verdict == "fail"


def test_ub6_unexecuted_notebook_is_unprobeable_not_pass():
    verdicts = probe_executed_notebook(
        EVIDENCE / "june9-gbald-run" / "notebook.ipynb"
    )
    assert [v.verdict for v in verdicts] == ["unprobeable"]


def test_ub6_malformed_notebook_is_unprobeable_not_crash(tmp_path):
    p = tmp_path / "notebook.ipynb"
    p.write_text("not json", encoding="utf-8")

    verdicts = probe_executed_notebook(p)

    assert [v.verdict for v in verdicts] == ["unprobeable"]
    assert "could not be read" in verdicts[0].message


def test_ub6_passes_a_genuinely_learning_notebook(tmp_path):
    nb = {
        "cells": [{
            "cell_type": "code", "id": "x", "source": [],
            "outputs": [{
                "output_type": "stream", "name": "stdout",
                "text": [
                    "Classes: 4\n",
                    "Round 0: labels=20, test_acc=0.31\n",
                    "Round 1: labels=30, test_acc=0.55\n",
                    "Round 2: labels=40, test_acc=0.78\n",
                ],
            }],
        }],
    }
    p = tmp_path / "nb.ipynb"
    p.write_text(json.dumps(nb))
    verdicts = probe_executed_notebook(p)
    main = _verdict_of(verdicts, "UB-6")
    assert main.verdict == "pass", main.message
    # The integer `labels` counter is excluded (floats only), so the
    # metric-agnostic arms never see it; the vetted accuracy raises no flag.
    assert not [v for v in verdicts if v.verdict in ("warn", "flag_for_researcher")]


def test_ub6_passes_a_spaced_abbreviated_accuracy_label(tmp_path):
    # The BADGE clean-baseline regression: the notebook prints a spaced,
    # abbreviated "test acc:" label (not "test_acc="), and a "Classes : N" line
    # with a space before the colon. The chance arm must still see the climbing
    # accuracy series and pass, never fall through to the unvetted flag.
    nb = {
        "cells": [{
            "cell_type": "code", "id": "x", "source": [],
            "outputs": [{
                "output_type": "stream", "name": "stdout",
                "text": [
                    "Classes : 10\n",
                    "Initial  -- labeled:   100  |  test acc: 0.5640\n",
                    "Round  1 -- labeled:   200  |  test acc: 0.5880\n",
                    "Round  2 -- labeled:   300  |  test acc: 0.6570\n",
                    "Round  3 -- labeled:   400  |  test acc: 0.7570\n",
                ],
            }],
        }],
    }
    p = tmp_path / "nb.ipynb"
    p.write_text(json.dumps(nb))
    verdicts = probe_executed_notebook(p)
    main = _verdict_of(verdicts, "UB-6")
    assert main.verdict == "pass", main.message
    assert "beats chance" in main.message
    assert not [v for v in verdicts if v.verdict == "flag_for_researcher"]


def test_metric_regex_matches_bare_space_separator_not_prose():
    # The Rethinking 2026-07-20 matrix shape: epoch lines separate the label
    # from the value with bare whitespace ("acc 0.0977"), no = or :. Those
    # lines carried the whole headline series, and missing them left the
    # beats-chance demo verdict undecidable at one matched point. Prose that
    # merely mentions accuracy near a number must still not match.
    from probes.universal import _METRIC_RE

    line = "Epoch   0 | Train loss 2.6362  acc 0.0977 | Test loss  2.2984  acc 0.1719"
    assert [m.group(1) for m in _METRIC_RE.finditer(line)] == ["0.0977", "0.1719"]
    assert _METRIC_RE.search("Final test accuracy: 0.1875").group(1) == "0.1875"
    assert _METRIC_RE.search("test_acc=0.31").group(1) == "0.31"
    assert _METRIC_RE.search("accuracy improved 0.9x over baseline") is None
    assert _METRIC_RE.search("accuracy of the 0.5 threshold") is None


# ---------------------------------------------------------------------------
# UB-6 §1.2 hardening: metric-agnostic flatness + unvetted-number disclosure.
# The chance arm only understands accuracy; these arms catch a degenerate run
# whose metric is a loss/RMSE/F1 curve or whose class count is undetectable,
# which today slips through as a silent `unprobeable`.
# ---------------------------------------------------------------------------


def _nb_from_lines(tmp_path, *lines, name="nb.ipynb"):
    nb = {"cells": [{
        "cell_type": "code", "id": "c", "source": [],
        "outputs": [{"output_type": "stream", "name": "stdout",
                     "text": [ln if ln.endswith("\n") else ln + "\n"
                              for ln in lines]}],
    }]}
    p = tmp_path / name
    p.write_text(json.dumps(nb))
    return p


def test_numeric_series_extractor_excludes_integer_counters():
    # `labels=1000` and `Classes: 10` are integer counters, not metrics; only
    # the decimal `test_acc` progression is a series.
    from probes.universal import extract_numeric_series
    nb = {"cells": [{"cell_type": "code", "outputs": [{
        "output_type": "stream", "name": "stdout", "text": [
            "Classes: 10\n",
            "Round 0: labels=1000, test_acc=0.0900\n",
            "Round 1: labels=1100, test_acc=0.0940\n",
            "Round 2: labels=1200, test_acc=0.0990\n",
        ]}]}]}
    key, series = extract_numeric_series(nb)
    assert key == "test_acc"
    assert series == [0.0900, 0.0940, 0.0990]


def test_ub6_flat_non_accuracy_loss_fails_metric_agnostically(tmp_path):
    # A flat val_loss the accuracy regex never matches, and no class line, so
    # the chance arm cannot run. The flatness arm must still FAIL it.
    p = _nb_from_lines(
        tmp_path,
        "Epoch 0: val_loss=2.3026",
        "Epoch 1: val_loss=2.3025",
        "Epoch 2: val_loss=2.3027",
        "Epoch 3: val_loss=2.3026",
    )
    main = _verdict_of(probe_executed_notebook(p), "UB-6")
    assert main.verdict == "fail", main.message
    assert "flat" in main.message and "val_loss" in main.message


def test_ub6_classless_accuracy_is_unvetted_flag_not_silent_unprobeable(tmp_path):
    # accuracy [0.090, 0.105, 0.116] with NO class line: cannot chance-check,
    # ~25% spread so not flat — the design's canonical unvetted-number case.
    p = _nb_from_lines(
        tmp_path,
        "Round 0: accuracy=0.090",
        "Round 1: accuracy=0.105",
        "Round 2: accuracy=0.116",
    )
    main = _verdict_of(probe_executed_notebook(p), "UB-6")
    assert main.verdict == "flag_for_researcher", main.message
    assert "unvetted" in main.message


def test_ub6_classless_loss_curve_is_unvetted_flag(tmp_path):
    # A descending loss with no class count: not flat, not chance-checkable.
    # UB-6 discloses it as unvetted; descent (UB-9) is what actually vets it.
    p = _nb_from_lines(
        tmp_path,
        "Step 0: loss=1.8",
        "Step 1: loss=1.2",
        "Step 2: loss=0.6",
    )
    main = _verdict_of(probe_executed_notebook(p), "UB-6")
    assert main.verdict == "flag_for_researcher", main.message


def test_ub6_number_free_output_stays_unprobeable(tmp_path):
    p = _nb_from_lines(
        tmp_path, "Model built.", "Training complete.", "Done.")
    main = _verdict_of(probe_executed_notebook(p), "UB-6")
    assert main.verdict == "unprobeable", main.message


def test_ub6_flat_config_echo_beside_moving_metric_is_not_degenerate(tmp_path):
    # A healthy run that echoes a flat lr each round AND a moving loss, with no
    # class count. Degeneracy means NOTHING moved; loss moves, so this must be
    # the unvetted disclosure on the moving series, never a flat degenerate-fail.
    p = _nb_from_lines(
        tmp_path,
        "Epoch 0: lr=0.001, val_loss=1.80",
        "Epoch 1: lr=0.001, val_loss=1.10",
        "Epoch 2: lr=0.001, val_loss=0.50",
    )
    main = _verdict_of(probe_executed_notebook(p), "UB-6")
    assert main.verdict == "flag_for_researcher", main.message
    assert "val_loss" in main.message


def test_ub6_all_series_flat_fails_degenerate(tmp_path):
    # Nothing moves anywhere: a genuine degenerate executed output.
    p = _nb_from_lines(
        tmp_path,
        "Epoch 0: lr=0.001, val_loss=2.3026",
        "Epoch 1: lr=0.001, val_loss=2.3025",
        "Epoch 2: lr=0.001, val_loss=2.3027",
    )
    main = _verdict_of(probe_executed_notebook(p), "UB-6")
    assert main.verdict == "fail", main.message
    assert "flat" in main.message


def test_ub6_flat_metric_beside_moving_config_is_surfaced_not_masked(tmp_path):
    # The masking guard (red-team FN1): a flat, degenerate non-loss metric
    # (auc=0.5) printed beside a MOVING config echo (an lr schedule), no class.
    # The flat auc must be NAMED in the disclosure, never silently dropped just
    # because the lr schedule moved.
    p = _nb_from_lines(
        tmp_path,
        "epoch 0: auc=0.500, lr=1.000",
        "epoch 1: auc=0.500, lr=0.500",
        "epoch 2: auc=0.500, lr=0.333",
        "epoch 3: auc=0.500, lr=0.250",
    )
    main = _verdict_of(probe_executed_notebook(p), "UB-6")
    assert main.verdict == "flag_for_researcher", main.message
    assert "auc" in main.message and "flat" in main.message, main.message


def test_ub6_nan_divergence_fails_not_unprobeable(tmp_path):
    # NaN prints as the text "nan", which the float-series parser cannot see.
    # A non-finite metric must FAIL, never read as a silent unprobeable.
    p = _nb_from_lines(
        tmp_path,
        "epoch 0: loss=2.30",
        "epoch 1: loss=nan",
        "epoch 2: loss=nan",
    )
    main = _verdict_of(probe_executed_notebook(p), "UB-6")
    assert main.verdict == "fail", main.message
    assert main.reason == "output_diverged"
    assert "printed 'nan'" in main.message
    # R2C-071's rider: divergence is a candidate source, not the asserted
    # diagnosis. The 2026-08-06 delivery's NaN lived entirely in the ground
    # truth while every weight and prediction was finite, and the confident
    # wording sent the reviewer hunting a config bug.
    assert "the run diverged" not in main.message
    assert "training divergence" in main.message


def test_ub6_inf_divergence_fails(tmp_path):
    p = _nb_from_lines(tmp_path, "step 0: cost=inf", "step 1: cost=inf")
    main = _verdict_of(probe_executed_notebook(p), "UB-6")
    assert main.verdict == "fail" and main.reason == "output_diverged"
    assert "printed 'inf'" in main.message
    assert "the run diverged" not in main.message


def test_ub6_prose_with_info_does_not_false_trip_divergence(tmp_path):
    # "info"/"inference" must not match the NaN/inf scan (value-position only).
    p = _nb_from_lines(
        tmp_path,
        "Running inference on the validation set",
        "Classes: 4",
        "Round 0: test_acc=0.31",
        "Round 1: test_acc=0.55",
        "Round 2: test_acc=0.78",
    )
    main = _verdict_of(probe_executed_notebook(p), "UB-6")
    assert main.verdict == "pass", main.message


def test_ub6_chance_fail_stays_primary_no_extra_series_verdict(tmp_path):
    # Regression: the gbald chance fail must remain the single, first series
    # verdict — the new arms are in mutually-exclusive elif branches, so they
    # never pile a second UB-6 series verdict onto a chance-checked notebook.
    verdicts = probe_executed_notebook(
        ZOO / "gbald-lr74-never-learns" / "notebook.ipynb")
    series_verdicts = [v for v in verdicts
                       if v.verdict in ("fail", "flag_for_researcher", "pass")]
    assert len(series_verdicts) == 1
    assert series_verdicts[0].verdict == "fail"
    assert "chance" in series_verdicts[0].message


# ---------------------------------------------------------------------------
# UB-9 §1.3: loss descent / best-so-far monotone. A NEW probe (NOT UB-5).
# UB-6 tests peak-beats-chance, not descent, so a sign-flipped loss that still
# peaks above chance slips past it; UB-9 catches the flat and the rising curve.
# ---------------------------------------------------------------------------


def test_ub9_descending_loss_passes():
    v = descent_verdict([1.8, 1.2, 0.6], "val_loss")
    assert v.verdict == "pass", v.message
    assert "descends" in v.message


def test_ub9_noisy_but_descending_loss_passes():
    # Bumpy trajectory whose best-so-far still falls a lot — must not false-fail.
    v = descent_verdict([1.8, 2.0, 1.1, 1.3, 0.5], "train_loss")
    assert v.verdict == "pass", v.message


def test_ub9_flat_loss_fails():
    v = descent_verdict([2.3026, 2.3025, 2.3027, 2.3026], "val_loss")
    assert v.verdict == "fail", v.message
    assert "does not descend" in v.message


def test_ub9_rising_loss_fails_sign_flip_class():
    # A maximized / sign-flipped objective rises — best-so-far never improves.
    v = descent_verdict([0.5, 1.1, 1.8], "loss")
    assert v.verdict == "fail", v.message


def test_ub9_too_short_series_is_unprobeable():
    assert descent_verdict([1.0, 0.5], "loss").verdict == "unprobeable"


def test_ub9_nan_loss_fails_diverged():
    v = descent_verdict([1.0, float("nan"), 0.3], "loss")
    assert v.verdict == "fail" and "NaN" in v.message


def _structured_history(values: list[float]) -> dict:
    from tests.test_time_series_training_history import _record

    losses = [
        {
            "index": index,
            "value": value,
            "sample_weight": 8,
            "checkpoint_id": f"c{index}",
        }
        for index, value in enumerate(values)
    ]
    return _record(
        checkpoint_id=f"c{len(losses) - 1}",
        loss_observations=losses,
        selection_range=None,
        selection={
            "status": "not_performed",
            "reason": "the fitting entry returns its final checkpoint",
        },
    )


def _history_run(tmp_path, values: list[float]):
    run = tmp_path / "run"
    pipeline = run / ".pipeline"
    pipeline.mkdir(parents=True)
    (pipeline / "training_history.json").write_text(
        json.dumps(_structured_history(values)), encoding="utf-8"
    )
    return run


def test_extract_loss_series_reads_only_structured_loss_observations():
    key, series = extract_loss_series(_structured_history([1.8, 1.2, 0.6]))
    assert key == "training_loss"
    assert series == [1.8, 1.2, 0.6]


def test_ub9_on_structured_descending_loss_passes(tmp_path):
    run = _history_run(tmp_path, [2.10, 1.20, 0.45])
    # Contradictory notebook prose cannot replace the validated artifact.
    (run / "notebook.ipynb").write_text(
        json.dumps({"cells": [{"cell_type": "code", "outputs": [{
            "output_type": "stream", "name": "stdout",
            "text": "train_loss=0.1\ntrain_loss=9.9\n",
        }]}]}),
        encoding="utf-8",
    )
    v = probe_loss_descent(run)
    assert v.verdict == "pass", v.message
    assert v.probe_id == "UB-9"
    assert "record_digest=" in v.evidence


def test_ub9_on_flat_structured_loss_fails(tmp_path):
    run = _history_run(tmp_path, [2.3026, 2.3025, 2.3027, 2.3026])
    assert probe_loss_descent(run).verdict == "fail"


def test_ub9_notebook_loss_without_structured_artifact_is_unprobeable(tmp_path):
    notebook = _nb_from_lines(
        tmp_path,
        "Epoch 0: train_loss=2.10",
        "Epoch 1: train_loss=1.20",
        "Epoch 2: train_loss=0.45",
    )
    run = tmp_path / "run-with-prose-only"
    run.mkdir()
    (run / "notebook.ipynb").write_bytes(notebook.read_bytes())

    verdict = probe_loss_descent(run)
    assert verdict.verdict == "unprobeable"
    assert verdict.reason == "training_history_missing"


def test_ub9_invalid_structured_artifact_is_unprobeable(tmp_path):
    run = tmp_path / "invalid-history"
    (run / ".pipeline").mkdir(parents=True)
    (run / ".pipeline" / "training_history.json").write_text(
        '{"status":"recorded"}', encoding="utf-8"
    )

    verdict = probe_loss_descent(run)
    assert verdict.verdict == "unprobeable"
    assert verdict.reason == "training_history_invalid"


def test_ub9_explicit_nontraining_history_is_not_applicable(tmp_path):
    from scripts.time_series_training_history import (
        not_applicable_training_history,
    )

    run = tmp_path / "nontraining"
    (run / ".pipeline").mkdir(parents=True)
    record = not_applicable_training_history(
        reason="the method has no fitting phase"
    )
    (run / ".pipeline" / "training_history.json").write_text(
        json.dumps(record), encoding="utf-8"
    )

    verdict = probe_loss_descent(run)
    assert verdict.verdict == "not_applicable"
    assert verdict.reason == "training_not_applicable"


# ---------------------------------------------------------------------------
# UB-3: decision-variable sensitivity on the real pdwa planner (M-005)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def pdwa():
    return load_module_from_path(ZOO / "pdwa-omega-degenerate" / "method.py")


def _pdwa_objective_kwargs(fixture):
    return {
        "v": 0.8,
        "omega": 0.3,
        "robot_state": fixture.start,
        "goal_position": fixture.goal,
        "obstacle_states": fixture.obstacle_states_at(0),
        "dt": fixture.dt,
        "v_max": 1.5,
        "r_robot": 0.1,
        "r_obs": 0.15,
        "alpha_h": 1.0,
        "alpha_d": 1.0,
        "alpha_c": 1.0,
        "alpha_f": 1.0,
        "beta": [0.1, 0.5, 1.0, 2.0, 5.0],
    }


def test_ub3_flags_pdwa_omega_degeneracy(pdwa):
    fixture = make_planning_fixture(seed=4)
    verdict = probe_callable_sensitivity(
        pdwa.evaluate_objective_function,
        base_kwargs=_pdwa_objective_kwargs(fixture),
        vary_param="omega",
        candidates=list(fixture.omega_candidates),
        flag_not_fail=True,          # faithful transcription of degenerate
        finding_class="M-005",       # paper math -> researcher, not auto-fail
    )
    assert verdict.verdict == "flag_for_researcher", verdict.message
    assert "omega" in verdict.message


def test_ub3_passes_when_the_input_actually_matters(pdwa):
    # Same machinery, varying v instead: the objective DOES read v
    # (false-positive guard for the sensitivity probe itself).
    fixture = make_planning_fixture(seed=4)
    verdict = probe_callable_sensitivity(
        pdwa.evaluate_objective_function,
        base_kwargs=_pdwa_objective_kwargs(fixture),
        vary_param="v",
        candidates=[0.2, 0.5, 0.9, 1.2],
    )
    assert verdict.verdict == "pass", verdict.message


# ---------------------------------------------------------------------------
# UB-8: seed threading on the real pdwa planner
# ---------------------------------------------------------------------------


def test_ub8_pdwa_select_velocity_threads_seed(pdwa):
    fixture = make_planning_fixture(seed=4)
    verdict = probe_seed_threading(
        pdwa.select_velocity,
        kwargs={
            "robot_state": fixture.start,
            "obstacle_states": fixture.obstacle_states_at(0),
            "goal_position": fixture.goal,
            "dt": fixture.dt,
        },
        seed_param="seed",
        expect_stochastic=True,  # its omega is a seeded random tie-break
    )
    assert verdict.verdict in ("pass", "warn"), verdict.message
    # Same-seed determinism is the hard requirement; it must not FAIL.
    assert verdict.verdict != "fail"


# ---------------------------------------------------------------------------
# UB-1: shape claims vs execution (M-001, on the KD mutant)
# ---------------------------------------------------------------------------


def test_ub1_parses_the_kd_mutant_claim():
    pytest.importorskip("torch")  # the zoo module imports it at load
    src_doc = load_module_from_path(
        ZOO / "kd-wrong-op-reconstructed" / "method.py"
    ).compute_instance_weights.__doc__
    assert parse_return_shape_claim(src_doc) == ("B", "N_queries", "1")


def test_ub1_fails_the_kd_wrong_op_mutant():
    torch = pytest.importorskip("torch")
    mod = load_module_from_path(ZOO / "kd-wrong-op-reconstructed" / "method.py")
    verdict = probe_shape_claim(
        mod.compute_instance_weights,
        call_args=(torch.randn(2, 5, 8), torch.randn(2, 5, 8)),
        dims={"B": 2, "N_queries": 5},
    )
    assert verdict.verdict == "fail", verdict.message
    assert "(2, 5, 5, 5)" in verdict.evidence


def test_ub1_passes_a_correct_implementation():
    def correct(x):
        """Compute row sums.

        Returns:
            sums: (B, 1) per-row totals.
        """
        return x.sum(axis=1, keepdims=True)

    verdict = probe_shape_claim(
        correct, call_args=(np.ones((3, 7)),), dims={"B": 3}
    )
    assert verdict.verdict == "pass", verdict.message


def test_ub1_unparseable_claim_is_unprobeable():
    def undocumented(x):
        return x

    verdict = probe_shape_claim(undocumented, call_args=(np.ones(2),))
    assert verdict.verdict == "unprobeable"


# ---------------------------------------------------------------------------
# UB-6 series-targeting fixes (GBALD 2026-06-30 + bev-distill 2026-07-02, the
# two-concrete-cases pair): count-first class phrasing feeds the chance arm,
# a runaway loss fails with the defect named, and the residual disclosure
# path prefers the accuracy-named series over a longer generic one.
# ---------------------------------------------------------------------------


def _nb_from_cells(tmp_path, cells_lines, name="nb.ipynb"):
    nb = {"cells": [{
        "cell_type": "code", "id": f"c{i}", "source": [],
        "outputs": [{"output_type": "stream", "name": "stdout",
                     "text": [ln if ln.endswith("\n") else ln + "\n"
                              for ln in lines]}],
    } for i, lines in enumerate(cells_lines)]}
    p = tmp_path / name
    p.write_text(json.dumps(nb))
    return p


def test_ub6_detects_count_first_class_phrasing(tmp_path):
    # The GBALD notebook printed "…784 features, 10 classes" and no labeled
    # "Classes: 10" line, so the chance arm never ran and a healthy curve
    # shipped as an unvetted flag.
    p = _nb_from_lines(
        tmp_path,
        "Pool: 1750 samples, 784 features, 10 classes",
        "test acc: 0.461", "test acc: 0.729", "test acc: 0.878",
    )
    v = _verdict_of(probe_executed_notebook(p), "UB-6")
    assert v.verdict == "pass", v.message
    assert "chance 0.10" in v.message


def test_ub6_runaway_loss_fails_with_defect_named(tmp_path):
    p = _nb_from_lines(
        tmp_path,
        "Epoch 1/10: avg loss = -0.0508",
        "Epoch 3/10: avg loss = -0.7203",
        "Epoch 6/10: avg loss = -5.4719",
        "Epoch 9/10: avg loss = -14.6442",
        "Training complete. Final avg loss: -18.9361",
    )
    v = _verdict_of(probe_executed_notebook(p), "UB-6")
    assert v.verdict == "fail", v.message
    assert "runs away" in v.message


def test_ub6_runaway_is_per_cell_not_masked_by_demo_cells(tmp_path):
    # bev-distill shape: small bounded demo losses print in earlier cells;
    # the training loop's runaway sits in its own cell. The cross-cell
    # concatenation is non-monotonic, so the check must group per cell.
    p = _nb_from_cells(tmp_path, [
        ["Dense feature distillation loss: 0.0050",
         "Instance distillation loss: 0.0000",
         "Combined distillation loss: 0.0050"],
        ["avg loss = -0.0508", "avg loss = -0.7203",
         "avg loss = -5.4719", "avg loss = -18.9361"],
    ])
    v = _verdict_of(probe_executed_notebook(p), "UB-6")
    assert v.verdict == "fail", v.message
    assert "runs away" in v.message


def test_ub6_runaway_skips_improving_and_non_loss_series(tmp_path):
    # An improving log-likelihood-style loss (shrinking magnitude) and a
    # legitimately growing series not named loss must not read as runaway.
    p = _nb_from_lines(
        tmp_path,
        "loss = -100.0", "loss = -60.0", "loss = -30.0", "loss = -20.0",
        "reward = 0.5", "reward = 5.0", "reward = 50.0", "reward = 500.0",
    )
    v = _verdict_of(probe_executed_notebook(p), "UB-6")
    assert v.verdict != "fail", v.message


def test_ub6_flag_prefers_accuracy_named_series(tmp_path):
    # No class count: the disclosure must name the accuracy series even when
    # a generic series (GBALD's 'score') is longer.
    p = _nb_from_lines(
        tmp_path,
        "score = 0.91", "score = 0.87", "score = 0.71", "score = 0.55",
        "score = 0.41", "score = 0.32",
        "acc = 0.31", "acc = 0.52", "acc = 0.74",
    )
    v = _verdict_of(probe_executed_notebook(p), "UB-6")
    assert v.verdict == "flag_for_researcher", v.message
    assert "'acc'" in v.message
