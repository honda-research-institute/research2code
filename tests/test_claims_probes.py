"""CT-1 contribution-floor (AL arm): degenerate-null identity fails, real
selectors pass, unmappable selectors stay honest."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from probes.claims import (  # noqa: E402
    CONTRADICTED,
    build_ledger,
    check_discriminating_power_al,
    contradiction_flag_verdicts,
    contradiction_gate,
    detect_level_claim,
    executed_metric_observations,
    probe_contribution_floor_al,
    random_acquisition,
    synthesize_behavioral_rows,
)
from probes.package_loader import load_module_from_path  # noqa: E402

EVIDENCE = Path(__file__).parent / "fixtures" / "evidence"


def _module(tmp_path, source, name="selector_under_test.py"):
    path = tmp_path / name
    path.write_text(source, encoding="utf-8")
    return load_module_from_path(path)


def _executed_notebook(*lines: str) -> dict:
    return {
        "cells": [{
            "cell_type": "code",
            "outputs": [{
                "output_type": "stream",
                "name": "stdout",
                "text": [ln if ln.endswith("\n") else ln + "\n"
                         for ln in lines],
            }],
        }],
        "metadata": {},
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def test_ct1_fails_first_k_placeholder(tmp_path):
    mod = _module(tmp_path, (
        "def select_batch(model, x_unlabeled, batch_size, seed):\n"
        "    # TODO implement BALD; placeholder for now\n"
        "    return list(range(batch_size))\n"))
    v = probe_contribution_floor_al(mod, "select_batch")
    assert v.verdict == "fail"
    assert v.finding_class == "M-004"
    assert "first_k" in v.message


def test_ct1_fails_seeded_random_placeholder(tmp_path):
    mod = _module(tmp_path, (
        "import numpy as np\n"
        "\n"
        "\n"
        "def select_batch(model, x_unlabeled, batch_size, seed):\n"
        "    rng = np.random.default_rng(seed)\n"
        "    return rng.choice(len(x_unlabeled), batch_size,"
        " replace=False).tolist()\n"), name="random_placeholder.py")
    v = probe_contribution_floor_al(mod, "select_batch")
    assert v.verdict == "fail"
    assert "rng_choice(seed)" in v.message


def test_ct1_passes_model_driven_selector(tmp_path):
    # Uncertainty-style selector: scores depend on the pool content, so
    # selections track the data, not a degenerate null.
    mod = _module(tmp_path, (
        "import torch\n"
        "\n"
        "\n"
        "def select_batch(model, x_unlabeled, batch_size, seed):\n"
        "    model.eval()\n"
        "    with torch.no_grad():\n"
        "        probs = torch.softmax(model(x_unlabeled), dim=1)\n"
        "    margin = probs.max(dim=1).values\n"
        "    return torch.argsort(margin)[:batch_size].tolist()\n"),
        name="margin_selector.py")
    v = probe_contribution_floor_al(mod, "select_batch")
    assert v.verdict == "pass", v.message


def test_ct1_passes_june9_gbald():
    pytest.importorskip("sklearn")
    mod = load_module_from_path(
        EVIDENCE / "june9-gbald-run" / "method" / "method.py")
    # CT-1 and UB-7 separate here by design: UB-7 flags the dead prior
    # channel; CT-1 passes because BALD ordering still drives selection.
    v = probe_contribution_floor_al(mod, "select_batch")
    assert v.verdict == "pass", v.message


def test_ct1_unprobeable_on_unmappable_signature(tmp_path):
    mod = _module(tmp_path, (
        "def select_batch(model, x_unlabeled, batch_size, seed,"
        " oracle_handle):\n"
        "    return list(range(batch_size))\n"), name="unmappable.py")
    v = probe_contribution_floor_al(mod, "select_batch")
    assert v.verdict == "unprobeable"
    assert "oracle_handle" in v.message


# ---------------------------------------------------------------------------
# CT-5 discriminating power (§1.3): an invariant must FAIL on the
# contribution-ablated baseline (uniform random acquisition) to earn verified.
# ---------------------------------------------------------------------------

_MARGIN_SELECTOR = (
    "import torch\n"
    "\n"
    "\n"
    "def select_batch(model, x_unlabeled, batch_size, seed):\n"
    "    model.eval()\n"
    "    with torch.no_grad():\n"
    "        probs = torch.softmax(model(x_unlabeled), dim=1)\n"
    "    margin = probs.max(dim=1).values\n"
    "    return torch.argsort(margin)[:batch_size].tolist()\n")


def _no_duplicates(selection, kit):
    return len(selection) == len(set(selection))


def _differs_from_random_null(selection, kit):
    null = sorted(random_acquisition(
        kit["pool_size"], kit["batch_size"], kit["seed"]))
    return sorted(set(selection)) != null


def test_ct5_no_duplicates_is_non_discriminating(tmp_path):
    # A real uncertainty selector has no duplicate selections — but so does
    # random acquisition, so "no duplicates" verifies nothing about the
    # contribution. The check must say so (fail = non-discriminating).
    pytest.importorskip("torch")
    mod = _module(tmp_path, _MARGIN_SELECTOR, name="margin_for_ct5.py")
    v = check_discriminating_power_al(mod, "select_batch", _no_duplicates)
    assert v.verdict == "fail", v.message
    assert "non-discriminating" in v.message.lower()


def test_ct5_differs_from_null_is_discriminating(tmp_path):
    # The real selector differs from random acquisition; the ablated baseline
    # (random acquisition itself) does not, so the invariant discriminates.
    pytest.importorskip("torch")
    mod = _module(tmp_path, _MARGIN_SELECTOR, name="margin_for_ct5b.py")
    v = check_discriminating_power_al(
        mod, "select_batch", _differs_from_random_null)
    assert v.verdict == "pass", v.message
    assert "discriminating" in v.message.lower()


def test_ct5_invariant_false_on_real_selector_fails(tmp_path):
    # An invariant the real selector itself violates cannot verify anything.
    pytest.importorskip("torch")
    mod = _module(tmp_path, _MARGIN_SELECTOR, name="margin_for_ct5c.py")
    v = check_discriminating_power_al(
        mod, "select_batch", lambda sel, kit: False)
    assert v.verdict == "fail", v.message
    assert "does not even hold" in v.message


def test_ct5_unprobeable_on_unmappable_signature(tmp_path):
    mod = _module(tmp_path, (
        "def select_batch(model, x_unlabeled, batch_size, seed,"
        " oracle_handle):\n"
        "    return list(range(batch_size))\n"), name="unmappable_ct5.py")
    v = check_discriminating_power_al(mod, "select_batch", _no_duplicates)
    assert v.verdict == "unprobeable"


# ---------------------------------------------------------------------------
# CT-3 claims ledger (v1 extractor)
# ---------------------------------------------------------------------------


def test_ledger_rows_from_experiment_elements(tmp_path):
    from probes.claims import build_claims_ledger
    pm = {"elements": [
        {"id": "exp-main", "type": "experiment", "name": "main comparison",
         "section": "Section 5",
         "source_text": "GBALD outperforms BALD on MNIST, reaching 0.99 "
                        "accuracy with 600 acquired samples."},
        {"id": "eq-1", "type": "equation", "source_text": "x = argmax ..."},
        {"id": "exp-trials", "type": "experiment", "name": "trials",
         "section": "Section 6",
         "source_text": "A total of 40 trials were conducted."},
    ]}
    rows = build_claims_ledger(pm)
    assert [r["claim_id"] for r in rows] == ["exp-main", "exp-trials"]
    main = rows[0]
    assert main["comparative"] is True
    assert main["direction_phrase"].lower().startswith("outperform")
    assert "0.99" in main["numbers"] and "600" in main["numbers"]
    assert main["status"] == "untested_at_this_scale"
    trials = rows[1]
    assert trials["comparative"] is False
    assert trials["numbers"] == ["40"]


def test_ledger_probe_writes_artifact_and_passes(tmp_path):
    import json
    from probes.claims import probe_claims_ledger
    run = tmp_path / "run"
    (run / ".pipeline").mkdir(parents=True)
    (run / ".pipeline" / "paper_map.json").write_text(json.dumps({
        "elements": [{"id": "exp-1", "type": "experiment",
                      "source_text": "Method improves accuracy by 4%."}]}))
    v = probe_claims_ledger(run)
    assert v.verdict == "pass"
    ledger = json.loads(
        (run / ".pipeline" / "claims_ledger.json").read_text())
    assert ledger["claims"][0]["claim_id"] == "exp-1"
    assert ledger["claims"][0]["status"] == "untested_at_this_scale"


def test_ledger_on_real_paper_maps():
    import json
    from probes.claims import build_claims_ledger
    # Every matrix paper's map must yield a non-empty, fully-statused ledger
    # with a comparative claim (the three-paper survey's conclusion, pinned on a
    # real artifact). Read the COMMITTED fixture snapshot, not r2c_runs/ — the
    # live run dir is mutable and gets overwritten by each pipeline launch (a
    # 2026-06-15 BADGE run wrote a fresh map with no comparative experiment and
    # flipped this test red), exactly the coupling the zoo discipline avoids.
    real = (Path(__file__).parent.parent / "tests" / "fixtures" / "delivery"
            / "badge-run-20260610" / ".pipeline" / "paper_map.json")
    if not real.is_file():
        import pytest
        pytest.skip("committed BADGE fixture snapshot not present")
    rows = build_claims_ledger(json.loads(real.read_text()))
    assert len(rows) >= 4
    assert all(r["status"] == "untested_at_this_scale" for r in rows)
    assert any(r["comparative"] for r in rows)


# ---------------------------------------------------------------------------
# CT-3 step 2: scale classification + synthesized behavioral rows. Verified
# can appear ONLY on a synthesized behavioral row whose check is discriminating
# (fails for the contribution-removed baseline); lifted headline rows stay
# scale-bound and untested, so a headline number never earns a verified badge.
# ---------------------------------------------------------------------------


def test_detect_level_claim():
    assert detect_level_claim(
        {"claim_text": "reaches 0.99 accuracy", "numbers": ["0.99"]}) is True
    # A named full dataset anchors a scale even with no extracted number.
    assert detect_level_claim(
        {"claim_text": "evaluated on CIFAR-10", "numbers": []}) is True
    # A purely qualitative claim carries no absolute level.
    assert detect_level_claim(
        {"claim_text": "outperforms the baselines", "numbers": []}) is False


def test_lifted_rows_are_scale_bound_and_untested():
    pm = {"elements": [{
        "id": "exp-main", "type": "experiment",
        "source_text": "GBALD outperforms BALD on MNIST, reaching 0.99 "
                       "accuracy with 600 acquired samples."}]}
    row = build_ledger(pm, [], "active_learning/bayesian")["claims"][0]
    assert row["kind"] == "lifted"
    assert row["scale_free"] is False        # never prose-inferred as scale-free
    assert row["level_claim"] is True        # numbers + a named dataset
    assert row["status"] == "untested_at_this_scale"
    assert row["verdict_trace"]["blocked_by"].startswith("S0")


def test_lifted_row_records_matched_executed_number_but_stays_untested():
    pm = {"elements": [{
        "id": "exp-main", "type": "experiment",
        "source_text": "Method reaches 0.99 accuracy on MNIST."}]}
    executed = [{
        "metric": "accuracy",
        "value": 0.73,
        "raw": "0.73",
        "source": "notebook.ipynb",
        "scale": "executed_smoke",
        "temporal": "final",
    }]

    row = build_ledger(
        pm, [], "active_learning", executed_metrics=executed)["claims"][0]

    assert row["paper_side"]["value"] == 0.99
    assert row["executed_side"]["value"] == 0.73
    assert row["executed_side"]["aggregation"] == "final"
    assert row["status"] == "untested_at_this_scale"
    assert row["verdict_trace"]["blocked_by"].startswith("G1")
    assert row["verdict_trace"]["gate"]["G6"]["held"] is True
    assert "rather than a verdict about the paper" in row["reasoning"]


def test_relative_improvement_number_is_not_matched_to_absolute_metric():
    pm = {"elements": [{
        "id": "exp-main", "type": "experiment",
        "source_text": "Method improves accuracy by 4% over the baseline."}]}
    executed = [{"metric": "accuracy", "value": 0.73, "raw": "0.73"}]

    row = build_ledger(
        pm, [], "active_learning", executed_metrics=executed)["claims"][0]

    assert row["paper_side"] is None
    assert row["executed_side"] is None
    assert row["status"] == "untested_at_this_scale"


def test_multi_number_metric_claim_is_left_unmatched():
    pm = {"elements": [{
        "id": "exp-main", "type": "experiment",
        "source_text": "The method reaches 70%/80%/90% accuracy thresholds."}]}
    executed = [{"metric": "accuracy", "value": 0.73, "raw": "0.73"}]

    row = build_ledger(
        pm, [], "active_learning", executed_metrics=executed)["claims"][0]

    assert row["paper_side"] is None
    assert row["executed_side"] is None


def test_spaced_multi_number_metric_claim_is_left_unmatched():
    pm = {"elements": [{
        "id": "exp-main", "type": "experiment",
        "source_text": "The method reaches 70% / 80% / 90% accuracy thresholds."}]}
    executed = [{"metric": "accuracy", "value": 0.73, "raw": "0.73"}]

    row = build_ledger(
        pm, [], "active_learning", executed_metrics=executed)["claims"][0]

    assert row["paper_side"] is None
    assert row["executed_side"] is None


def test_improvement_noun_number_is_not_absolute_metric():
    pm = {"elements": [{
        "id": "exp-main", "type": "experiment",
        "source_text": "BADGE gives a 4% accuracy improvement over baseline."}]}
    executed = [{"metric": "accuracy", "value": 0.73, "raw": "0.73"}]

    row = build_ledger(
        pm, [], "active_learning", executed_metrics=executed)["claims"][0]

    assert row["paper_side"] is None
    assert row["executed_side"] is None


def test_metric_looking_counts_are_not_paper_side_numbers():
    for text in ("The paper reports 5 error cases.",
                 "The objective has 10 loss terms."):
        pm = {"elements": [{"id": "exp-main", "type": "experiment",
                            "source_text": text}]}
        row = build_ledger(pm, [], "active_learning")["claims"][0]
        assert row["paper_side"] is None, text
        assert row["executed_side"] is None, text


def test_final_test_metric_preferred_over_final_train_metric():
    pm = {"elements": [{
        "id": "exp-main", "type": "experiment",
        "source_text": "Method reaches 0.90 accuracy on MNIST."}]}
    nb = _executed_notebook(
        "Final test accuracy: 0.73",
        "Final train accuracy: 0.99",
    )

    row = build_ledger(
        pm, [], "active_learning",
        executed_metrics=executed_metric_observations(nb))["claims"][0]

    assert row["executed_side"]["raw"] == "0.73"
    assert row["executed_side"]["split"] == "test"


def test_round_fallback_ignores_later_non_round_best_metric():
    pm = {"elements": [{
        "id": "exp-main", "type": "experiment",
        "source_text": "Method reaches 0.90 accuracy on MNIST."}]}
    nb = _executed_notebook(
        "Round 0: test accuracy=0.30",
        "Round 1: test accuracy=0.50",
        "Best validation accuracy=0.80",
    )

    row = build_ledger(
        pm, [], "active_learning",
        executed_metrics=executed_metric_observations(nb))["claims"][0]

    assert row["executed_side"]["raw"] == "0.50"
    assert row["executed_side"]["aggregation"] == "last_observed_round"


def test_round_fallback_prefers_test_over_train_within_latest_round():
    pm = {"elements": [{
        "id": "exp-main", "type": "experiment",
        "source_text": "Method reaches 0.90 accuracy on MNIST."}]}
    nb = _executed_notebook(
        "Round 0: test accuracy=0.30 train accuracy=0.92",
        "Round 1: test accuracy=0.50 train accuracy=0.99",
    )

    row = build_ledger(
        pm, [], "active_learning",
        executed_metrics=executed_metric_observations(nb))["claims"][0]

    assert row["executed_side"]["raw"] == "0.50"
    assert row["executed_side"]["split"] == "test"
    assert row["executed_side"]["round"] == 1


def test_probe_claims_ledger_reads_executed_notebook_outputs(tmp_path):
    import json
    from probes.claims import probe_claims_ledger
    run = tmp_path / "run"
    (run / ".pipeline").mkdir(parents=True)
    (run / ".pipeline" / "paper_map.json").write_text(json.dumps({
        "elements": [{"id": "exp-1", "type": "experiment",
                      "source_text": "Method reaches 0.90 accuracy on MNIST."}]
    }))
    (run / "notebook.ipynb").write_text(json.dumps(_executed_notebook(
        "Round 0: labels=20, test accuracy = 0.31",
        "Round 1: labels=30, test accuracy = 0.55",
        "Final accuracy: 0.78",
    )))

    v = probe_claims_ledger(run)

    assert v.verdict == "pass"
    assert "1 lifted number comparison" in v.message
    ledger = json.loads(
        (run / ".pipeline" / "claims_ledger.json").read_text())
    row = ledger["claims"][0]
    assert row["paper_side"]["raw"] == "0.90"
    assert row["executed_side"]["raw"] == "0.78"
    assert row["executed_side"]["aggregation"] == "final"
    assert row["status"] == "untested_at_this_scale"


def test_structured_demo_claim_numbers_require_valid_demonstrated_skill(tmp_path):
    import json
    from probes.claims import probe_claims_ledger

    run = tmp_path / "run"
    pipeline = run / ".pipeline"
    pipeline.mkdir(parents=True)
    (pipeline / "paper_map.json").write_text(json.dumps({
        "elements": [{
            "id": "exp-1",
            "type": "experiment",
            "source_text": "Method reaches 0.90 accuracy on MNIST.",
        }],
    }))
    (run / "notebook.ipynb").write_text(json.dumps(_executed_notebook(
        "Final accuracy: 0.78",
    )))
    artifact = {
        "schema_version": "2.0.0",
        "verdict": "undetermined",
        "evidence_status": {
            "execution": {"status": "completed", "reasons": []},
            "evaluation_validity": {"status": "invalid", "reasons": []},
            "mechanism": {"status": "undetermined", "reasons": []},
            "skill": {"status": "undetermined", "reasons": []},
            "paper_benchmark": {"status": "not_assessed", "reasons": []},
        },
    }
    (pipeline / "demo_verdict.json").write_text(json.dumps(artifact))

    verdict = probe_claims_ledger(run)
    ledger = json.loads((pipeline / "claims_ledger.json").read_text())
    assert "0 lifted number comparison" in verdict.message
    assert ledger["claims"][0]["executed_side"] is None

    artifact["verdict"] = "succeeded"
    artifact["evidence_status"]["evaluation_validity"]["status"] = "valid"
    artifact["evidence_status"]["skill"]["status"] = "demonstrated"
    (pipeline / "demo_verdict.json").write_text(json.dumps(artifact))

    verdict = probe_claims_ledger(run)
    ledger = json.loads((pipeline / "claims_ledger.json").read_text())
    assert "1 lifted number comparison" in verdict.message
    assert ledger["claims"][0]["executed_side"]["raw"] == "0.78"


def test_required_structured_demo_claim_metrics_fail_closed_without_schema2(
    tmp_path,
):
    from probes.claims import _read_executed_metrics

    run = tmp_path / "run"
    pipeline = run / ".pipeline"
    pipeline.mkdir(parents=True)
    (pipeline / "method_spec.json").write_text(json.dumps({
        "comparison": {
            "classification": {"id": "time_series_forecasting"},
        },
    }))
    (run / "notebook.ipynb").write_text(json.dumps(_executed_notebook(
        "Final RMSE: 0.78",
    )))
    artifact = pipeline / "demo_verdict.json"

    assert _read_executed_metrics(run) == []

    artifact.write_text("{malformed")
    assert _read_executed_metrics(run) == []

    artifact.write_text(json.dumps({
        "schema_version": "2.0.0",
        "verdict": "succeeded",
        "evidence_status": {"skill": {"status": "demonstrated"}},
    }))
    assert _read_executed_metrics(run) == []


def test_required_structured_demo_claim_metrics_preserve_schema1_compatibility(
    tmp_path,
):
    from probes.claims import _read_executed_metrics

    run = tmp_path / "run"
    pipeline = run / ".pipeline"
    pipeline.mkdir(parents=True)
    (pipeline / "method_spec.json").write_text(json.dumps({
        "comparison": {
            "classification": {"id": "time_series_forecasting"},
        },
    }))
    (pipeline / "demo_verdict.json").write_text(json.dumps({
        "schema_version": "1.0.0",
        "verdict": "succeeded",
        "evidence_line": "Demo verdict: PASS",
    }))
    (run / "notebook.ipynb").write_text(json.dumps(_executed_notebook(
        "Final RMSE: 0.78",
    )))

    metrics = _read_executed_metrics(run)

    assert len(metrics) == 1
    assert metrics[0]["raw"] == "0.78"


def test_family_without_demo_skill_keeps_absent_artifact_claim_behavior(tmp_path):
    from probes.claims import _read_executed_metrics

    run = tmp_path / "run"
    pipeline = run / ".pipeline"
    pipeline.mkdir(parents=True)
    (pipeline / "method_spec.json").write_text(json.dumps({
        "comparison": {
            "classification": {"id": "active_learning"},
        },
    }))
    (run / "notebook.ipynb").write_text(json.dumps(_executed_notebook(
        "Final accuracy: 0.78",
    )))

    metrics = _read_executed_metrics(run)

    assert len(metrics) == 1
    assert metrics[0]["raw"] == "0.78"


def test_behavioral_verified_on_discriminating_pass():
    verdicts = [{"probe_id": "CT-1", "verdict": "pass",
                 "message": "differs from the degenerate nulls",
                 "element_ids": ["exp-main"]}]
    rows = synthesize_behavioral_rows(verdicts, "active_learning/bayesian")
    diff = [r for r in rows if r["claim_id"] == "behavioral:differs-from-null"]
    assert diff and diff[0]["status"] == "verified_at_scale", rows
    assert diff[0]["scale_free"] is True and diff[0]["level_claim"] is False
    assert diff[0]["depends_on"] == ["exp-main"]          # the 1.1 join key flows
    assert diff[0]["verifiable_aspect"]


def test_behavioral_untested_on_nondiscriminating_pass():
    # A passing loop-invariant check (AL-1) is real but a random baseline also
    # satisfies it, so it cannot earn verified.
    rows = synthesize_behavioral_rows(
        [{"probe_id": "AL-1", "verdict": "pass", "message": "invariants hold"}],
        "active_learning")
    nd = [r for r in rows if r["claim_id"] == "behavioral:no-duplicate-selections"]
    assert nd and nd[0]["status"] == "untested_at_this_scale"
    assert nd[0]["verdict_trace"]["blocked_by"] == "S2:not-discriminating"
    assert "does not confirm" in nd[0]["reasoning"]


def test_behavioral_suspect_on_flagged_check():
    # Step 3 branch F: a flagged behavioral check (a dead scoring term) is a
    # suspect-our-implementation signal, not a silent untested. With inertness
    # the ONLY signal, the method-intrinsic split applies (neutral wording).
    rows = synthesize_behavioral_rows(
        [{"probe_id": "UB-7", "verdict": "flag_for_researcher",
          "message": "dead term channel"}], "active_learning")
    st = [r for r in rows if r["claim_id"] == "behavioral:scoring-terms-live"]
    assert st and st[0]["status"] == "suspect_our_implementation"
    assert st[0]["suspected_cause"] == "method_appears_inert_as_published"
    # Never blames the paper.
    assert "paper being wrong" not in st[0]["reasoning"]
    assert "judgment" in st[0]["reasoning"]


def test_unprobeable_predicate_gets_no_row():
    rows = synthesize_behavioral_rows(
        [{"probe_id": "CT-1", "verdict": "unprobeable", "message": "no torch"}],
        "active_learning")
    assert not [r for r in rows
                if r["claim_id"] == "behavioral:differs-from-null"]


def test_not_applicable_predicate_stays_distinct_and_gets_no_claim_row():
    from probes.claims import _aggregate_status

    verdicts = [{
        "probe_id": "CT-1",
        "verdict": "not_applicable",
        "message": "the conditional mechanism is not declared",
    }]

    assert _aggregate_status(verdicts) == "not_applicable"
    rows = synthesize_behavioral_rows(verdicts, "active_learning")
    assert not [
        row for row in rows
        if row["claim_id"] == "behavioral:differs-from-null"
    ]


def test_kd_teacher_signal_verified():
    rows = synthesize_behavioral_rows(
        [{"probe_id": "KD-1", "verdict": "pass",
          "message": "teacher influences loss"}], "knowledge_distillation")
    assert rows and rows[0]["status"] == "verified_at_scale"


def test_build_ledger_lifted_first_then_behavioral_with_tally():
    pm = {"elements": [{"id": "exp-1", "type": "experiment",
                        "source_text": "improves accuracy by 4%."}]}
    verdicts = [{"probe_id": "CT-1", "verdict": "pass", "message": "differs"}]
    ledger = build_ledger(pm, verdicts, "active_learning")
    assert ledger["claims"][0]["kind"] == "lifted"
    assert ledger["claims"][0]["claim_id"] == "exp-1"
    assert any(r["kind"] == "behavioral" and r["status"] == "verified_at_scale"
               for r in ledger["claims"])
    assert ledger["tally"]["verified_at_scale"] == 1
    assert ledger["tally"]["untested_at_this_scale"] >= 1


def test_gbald_dead_prior_demotes_to_suspect_never_verified():
    # The GBALD firewall (step 3). GBALD's selector genuinely differs from
    # random (CT-1 passes), so step 2 would verify differs-from-null. But its
    # dead prior channel (UB-7 flag) is a same-path fidelity defect, so branch F
    # DEMOTES that would-be-verified row to suspect_our_implementation — a
    # structural pass cannot paint over a same-path defect. Nothing about the
    # GBALD selector earns verified, and nothing points at the paper.
    pytest.importorskip("sklearn")
    from probes.term_ablation import probe_al_selector_terms
    mod = load_module_from_path(
        EVIDENCE / "june9-gbald-run" / "method" / "method.py")
    ct1 = probe_contribution_floor_al(mod, "select_batch")
    ub7 = probe_al_selector_terms(mod, "select_batch")
    assert ct1.verdict == "pass", ct1.message            # really does differ
    rows = synthesize_behavioral_rows(
        [ct1.to_dict(), ub7.to_dict()], "active_learning/bayesian")
    assert rows, "expected synthesized behavioral rows"
    # No GBALD behavioral row verifies (the path carries a fidelity defect).
    assert not [r for r in rows if r["status"] == "verified_at_scale"]
    # The would-be-verified differs-from-null row is demoted to suspect, and
    # since the dead prior is the only signal, the method-intrinsic split routes
    # it to a human with neutral wording (our wiring or the method as published).
    diff = [r for r in rows if r["claim_id"] == "behavioral:differs-from-null"]
    assert diff and diff[0]["status"] == "suspect_our_implementation"
    assert "may be our wiring" in diff[0]["reasoning"]
    # Never an accusation of the paper: contradicted does not exist here.
    assert all(r["status"] != "contradicted" for r in rows)


def test_real_badge_paper_map_lifted_rows_never_verify():
    # The committed BADGE map's headline experiment rows are all scale-bound and
    # untested even when behavioral rows verify — no headline number is ever
    # stamped verified.
    real = (Path(__file__).parent.parent / "tests" / "fixtures" / "delivery"
            / "badge-run-20260610" / ".pipeline" / "paper_map.json")
    if not real.is_file():
        pytest.skip("committed BADGE fixture snapshot not present")
    import json
    pm = json.loads(real.read_text())
    verdicts = [{"probe_id": "CT-1", "verdict": "pass", "message": "differs"}]
    ledger = build_ledger(pm, verdicts, "active_learning")
    lifted = [r for r in ledger["claims"] if r["kind"] == "lifted"]
    assert lifted and all(r["status"] == "untested_at_this_scale"
                          for r in lifted)
    assert all(not r["scale_free"] for r in lifted)
    assert ledger["tally"]["verified_at_scale"] >= 1   # the behavioral row


# ---------------------------------------------------------------------------
# CT-3 step 3: branch F — the fidelity firewall. A would-be-verified row whose
# path is not fidelity-clean is demoted to suspect_our_implementation (never a
# paper verdict); a failed/flagged check is itself suspect. Deterministic cause.
# ---------------------------------------------------------------------------


def test_us4_fail_demotes_would_be_verified_to_suspect_r0_class():
    # The canonical GBALD attribution: a scale-mismatch fail (R_0) on the path
    # demotes the differs-from-null would-be-verified row to suspect, cause =
    # the scale-dependent parameter (US-4 outranks the downstream dead term).
    verdicts = [
        {"probe_id": "CT-1", "verdict": "pass", "message": "differs"},
        {"probe_id": "US-4", "verdict": "fail",
         "message": "R_0=2000 unrescaled on [0,1] data"},
    ]
    rows = synthesize_behavioral_rows(verdicts, "active_learning/bayesian")
    diff = [r for r in rows
            if r["claim_id"] == "behavioral:differs-from-null"][0]
    assert diff["status"] == "suspect_our_implementation"
    assert diff["suspected_cause"] == "scale_dependent_param_unrescaled"
    assert diff["linked_finding_id"] == "US-4"
    # Framed as our bug, with the explicit do-not-blame-the-paper reassurance.
    assert "most likely our implementation" in diff["reasoning"]
    assert "do not read it as the paper being wrong" in diff["reasoning"]


def test_cause_priority_scale_param_outranks_dead_term():
    verdicts = [
        {"probe_id": "CT-1", "verdict": "pass"},
        {"probe_id": "US-4", "verdict": "fail"},
        {"probe_id": "UB-7", "verdict": "flag_for_researcher"},
    ]
    rows = synthesize_behavioral_rows(verdicts, "active_learning")
    suspect = [r for r in rows
               if r["status"] == "suspect_our_implementation"]
    assert suspect
    assert all(r["suspected_cause"] == "scale_dependent_param_unrescaled"
               for r in suspect)


def test_implementing_code_finding_demotes_but_offpath_finding_does_not():
    base = [{"probe_id": "CT-1", "verdict": "pass", "message": "differs"}]
    # A finding on method.py is on the selector's path -> demote to suspect.
    on = synthesize_behavioral_rows(base, "active_learning", findings=[{
        "id": "F010", "severity": "important",
        "resolution_status": "pending", "file": "method/method.py"}])
    d_on = [r for r in on if r["claim_id"] == "behavioral:differs-from-null"][0]
    assert d_on["status"] == "suspect_our_implementation"
    assert d_on["suspected_cause"] == "open_fidelity_finding"
    assert d_on["linked_finding_id"] == "F010"
    # The same finding on requirements.txt is OFF the implementing-code path
    # (the BADGE case): the verified row survives.
    off = synthesize_behavioral_rows(base, "active_learning", findings=[{
        "id": "F001", "severity": "important",
        "resolution_status": "pending", "file": "requirements.txt"}])
    d_off = [r for r in off
             if r["claim_id"] == "behavioral:differs-from-null"][0]
    assert d_off["status"] == "verified_at_scale"


def test_resolved_finding_does_not_demote():
    rows = synthesize_behavioral_rows(
        [{"probe_id": "CT-1", "verdict": "pass"}], "active_learning",
        findings=[{"id": "F0", "severity": "critical",
                   "resolution_status": "applied", "file": "method/method.py"}])
    d = [r for r in rows if r["claim_id"] == "behavioral:differs-from-null"][0]
    assert d["status"] == "verified_at_scale"


def test_nondiscriminating_own_check_fail_is_suspect():
    # AL-1 fail (a loop index bug) -> the no-dups row is suspect (our bug), not
    # silently untested.
    rows = synthesize_behavioral_rows(
        [{"probe_id": "AL-1", "verdict": "fail",
          "message": "reacquires already-labeled points"}], "active_learning")
    nd = [r for r in rows
          if r["claim_id"] == "behavioral:no-duplicate-selections"][0]
    assert nd["status"] == "suspect_our_implementation"


def test_lifted_rows_never_become_suspect():
    # Branch F applies only to behavioral rows; a lifted headline row stays
    # untested even when the run carries fidelity signals.
    pm = {"elements": [{"id": "exp-1", "type": "experiment",
                        "source_text": "reaches 97% on MNIST"}]}
    verdicts = [{"probe_id": "US-4", "verdict": "fail"},
                {"probe_id": "CT-1", "verdict": "pass"}]
    ledger = build_ledger(pm, verdicts, "active_learning")
    lifted = [r for r in ledger["claims"] if r["kind"] == "lifted"]
    assert lifted and all(r["status"] == "untested_at_this_scale"
                          for r in lifted)


def test_implementing_code_match_is_case_insensitive():
    # A finding recorded with non-canonical casing on the implementing file must
    # still demote (the red-team's case-sensitivity miss on a case-insensitive fs).
    rows = synthesize_behavioral_rows(
        [{"probe_id": "CT-1", "verdict": "pass"}], "active_learning",
        findings=[{"id": "F9", "severity": "critical",
                   "resolution_status": "pending", "file": "method/Method.py"}])
    d = [r for r in rows if r["claim_id"] == "behavioral:differs-from-null"][0]
    assert d["status"] == "suspect_our_implementation"


def test_path_fidelity_signals_unit():
    from probes.claims import path_fidelity_signals
    sigs = path_fidelity_signals(
        [{"probe_id": "US-4", "verdict": "fail"},
         {"probe_id": "UB-6", "verdict": "fail"}],
        [{"id": "F1", "severity": "important", "resolution_status": "pending",
          "file": "method/method.py"}])
    keys = {s["cause_key"] for s in sigs}
    assert keys == {"scale_dependent_param_unrescaled",
                    "executed_output_degenerate", "open_fidelity_finding"}


# ---------------------------------------------------------------------------
# CT-3 step 5: the contradiction gate (§3.3) — the almost-unreachable bar, the
# ONE place CT-3 may point at the paper. The gate is a hard seven-clause
# conjunction that is fail-closed everywhere; today it is structurally
# unreachable (run_mode=smoke, single-seed, model veto unavailable), so every
# failing row still lands at suspect via branch F. These tests force each clause
# to confirm the conjunction logic, the routing wire, the delivery hook, and the
# structural-unreachability guarantee.
# ---------------------------------------------------------------------------

_CLEAN = []  # a provably clean path: no fidelity-not-clean signal
_K3 = {"k": 3, "reproduced": True}


def _gate(row=None, signals=None, seed_set=None, run_mode="smoke"):
    return contradiction_gate(row or {"level_claim": False}, signals or [],
                              seed_set=seed_set, run_mode=run_mode)


def test_gate_clean_path_single_seed_blocks_at_reproduction_routes_untested():
    # A clean-path scale-free invariant violation on one seed: no own-bug clause
    # fails, so it is NOT branch F; the reproduction clause (G5) blocks it, and
    # S3 routes that to untested (a one-seed wobble is not a verdict), never the
    # paper.
    g = _gate(seed_set={"k": 1, "reproduced": False})
    assert not g["held"]
    assert g["own_bug_block"] is False
    assert g["blocked_by"].startswith("G5")


def test_gate_reproduced_seeds_still_blocked_closed_at_model_veto():
    # Even a clean path reproduced across k>=3 seeds cannot reach contradicted:
    # the model cross-examination veto is unavailable and fails closed (G7), the
    # humility default that keeps the bar almost-unreachable.
    g = _gate(seed_set=_K3)
    assert not g["held"]
    assert g["own_bug_block"] is False
    assert g["blocked_by"].startswith("G7")


def test_gate_own_bug_signal_blocks_and_routes_branch_f():
    # An unrescaled scale param (the R_0 class) fails the scale-params clause
    # (G4): an own-bug block, so S3 routes to branch F (suspect), never the
    # paper. This is the literal GBALD firewall at the gate.
    g = _gate(signals=[{"cause_key": "scale_dependent_param_unrescaled",
                        "priority": 1}], seed_set=_K3)
    assert not g["held"]
    assert g["own_bug_block"] is True
    assert g["blocked_by"].startswith("G4")


def test_gate_degenerate_output_blocks_at_output_sane_clause():
    g = _gate(signals=[{"cause_key": "executed_output_degenerate",
                        "priority": 2}], seed_set=_K3)
    assert g["own_bug_block"] is True
    assert g["blocked_by"].startswith("G2")


def test_gate_level_claim_needs_explicit_full_run_mode_never_inferred():
    # A level/headline claim can only be scale-matched (G1b) by an explicit
    # run_mode=full assertion; at smoke scale it is blocked at G1, so headline
    # numbers can never be contradicted today.
    smoke = _gate({"level_claim": True}, seed_set=_K3, run_mode="smoke")
    assert smoke["blocked_by"].startswith("G1")
    assert smoke["own_bug_block"] is False
    # With run_mode=full the scale clause passes; the veto still fails closed.
    full = _gate({"level_claim": True}, seed_set=_K3, run_mode="full")
    assert full["clauses"]["G1"]["held"] is True
    assert full["blocked_by"].startswith("G7")


def test_gate_g6_margin_within_is_not_a_contradiction():
    # A matched number within the margin is a magnitude shortfall, never a
    # contradiction (G6). Beyond the margin, G6 holds and the veto is the block.
    within = _gate({"level_claim": True, "paper_side": 0.99,
                    "executed_side": 0.95}, seed_set=_K3, run_mode="full")
    assert within["clauses"]["G6"]["held"] is False
    assert within["blocked_by"].startswith("G6")
    beyond = _gate({"level_claim": True, "paper_side": 0.99,
                    "executed_side": 0.50}, seed_set=_K3, run_mode="full")
    assert beyond["clauses"]["G6"]["held"] is True
    assert beyond["blocked_by"].startswith("G7")


def test_gate_holds_only_when_every_clause_passes(monkeypatch):
    # The conjunction is exhaustive: the ONLY thing keeping a clean, full-scale,
    # reproduced, beyond-margin disagreement out of contradicted is the
    # fail-closed model veto. With the veto available and not objecting, the
    # gate holds — proof the machinery is correct and merely inert today.
    import probes.claims as claims
    monkeypatch.setattr(claims, "_g7_cross_examine",
                        lambda row, signals: (True, "no implementation, scale, "
                                              "data, or variance explanation"))
    g = claims.contradiction_gate(
        {"level_claim": True, "paper_side": 0.99, "executed_side": 0.50},
        _CLEAN, seed_set=_K3, run_mode="full")
    assert g["held"] is True
    assert g["blocked_by"] is None
    # Drop ANY single clause and it no longer holds.
    assert not claims.contradiction_gate(
        {"level_claim": True, "paper_side": 0.99, "executed_side": 0.50},
        _CLEAN, seed_set={"k": 1, "reproduced": False}, run_mode="full")["held"]


# --- S3 routing wire: the three-way branch on the gate verdict ---------------


def _force_gate(monkeypatch, *, held, own_bug_block):
    import probes.claims as claims
    monkeypatch.setattr(claims, "contradiction_gate",
                        lambda *a, **k: {"held": held,
                                         "own_bug_block": own_bug_block,
                                         "blocked_by": None if held else "Gx:test",
                                         "summary": "forced", "clauses": {}})


def test_s3_routes_to_contradicted_when_gate_holds(monkeypatch):
    # When the gate holds, the failing behavioral row is contradicted with the
    # careful-human-look wording and a clean trace. (Unreachable via the live
    # pipeline; this proves the wire from a held gate to the verdict.)
    _force_gate(monkeypatch, held=True, own_bug_block=False)
    rows = synthesize_behavioral_rows(
        [{"probe_id": "AL-1", "verdict": "fail"}], "active_learning",
        seed_set=_K3)
    nd = [r for r in rows
          if r["claim_id"] == "behavioral:no-duplicate-selections"][0]
    assert nd["status"] == CONTRADICTED
    assert "careful human look" in nd["reasoning"]
    assert nd["verdict_trace"]["result"] == CONTRADICTED
    assert nd["verdict_trace"]["blocked_by"] is None
    assert nd["seed_set"] == _K3


def test_s3_routes_to_untested_on_clean_path_blocked_by_scale_clause(monkeypatch):
    # A clean path blocked only by a scale/seed/veto clause is untested, never a
    # phantom own-bug (§3.3 G5). One seed is not a verdict.
    _force_gate(monkeypatch, held=False, own_bug_block=False)
    rows = synthesize_behavioral_rows(
        [{"probe_id": "AL-1", "verdict": "fail"}], "active_learning",
        seed_set={"k": 1, "reproduced": False})
    nd = [r for r in rows
          if r["claim_id"] == "behavioral:no-duplicate-selections"][0]
    assert nd["status"] == "untested_at_this_scale"
    assert "not treating one seed as a verdict" in nd["reasoning"]


def test_s3_records_gate_trace_on_a_live_suspect_row():
    # The live default: a failing check carries an own-bug signal, so the gate
    # is own-bug-blocked and the row is suspect — unchanged status. What step 5
    # adds is the trace: the gate evaluation and exactly what kept the row out
    # of contradicted ("blocked_by").
    rows = synthesize_behavioral_rows(
        [{"probe_id": "AL-1", "verdict": "fail",
          "message": "reacquires labeled points"}], "active_learning")
    nd = [r for r in rows
          if r["claim_id"] == "behavioral:no-duplicate-selections"][0]
    assert nd["status"] == "suspect_our_implementation"
    assert "S3" in nd["verdict_trace"] and "gate" in nd["verdict_trace"]
    assert nd["verdict_trace"]["blocked_by"].startswith("G3")  # own check failed


# --- delivery wiring + structural-unreachability guarantee -------------------


def test_contradiction_flag_verdicts_emit_demoting_probe_verdict():
    # A contradicted row emits a synthetic flag_for_researcher CT-3 verdict that
    # demotes through the existing probe-reasons path (§10.1), carrying the
    # claim's path as the join key.
    ledger = {"claims": [
        {"status": CONTRADICTED, "name": "the invariant",
         "claim_id": "behavioral:x", "depends_on": ["e1", "e2"]},
        {"status": "untested_at_this_scale", "name": "other"}]}
    vs = contradiction_flag_verdicts(ledger)
    assert len(vs) == 1
    assert vs[0].probe_id == "CT-3"
    assert vs[0].verdict == "flag_for_researcher"
    assert vs[0].element_ids == ["e1", "e2"]
    assert "careful human look" in vs[0].message


def test_no_contradiction_no_flag_verdict():
    ledger = build_ledger(
        {"elements": []},
        [{"probe_id": "CT-1", "verdict": "pass"}], "active_learning")
    assert contradiction_flag_verdicts(ledger) == []


def test_live_smoke_default_never_contradicts():
    # The structural guarantee: with the real defaults (smoke run, single seed,
    # no veto), NO failing behavioral row reaches contradicted, across every
    # AL invariant. The bar is unreachable in the current pipeline.
    verdicts = [
        {"probe_id": "CT-1", "verdict": "fail"},
        {"probe_id": "UB-7", "verdict": "flag_for_researcher"},
        {"probe_id": "AL-1", "verdict": "fail"},
        {"probe_id": "UB-9", "verdict": "fail"},
    ]
    rows = synthesize_behavioral_rows(verdicts, "active_learning")
    assert rows
    assert all(r["status"] != CONTRADICTED for r in rows)


def test_read_run_mode_and_seed_set_default_and_parse(tmp_path):
    import json
    from probes.claims import _read_run_mode, _read_seed_set
    run = tmp_path / "run"
    (run / ".pipeline").mkdir(parents=True)
    # No driver_state -> conservative defaults.
    assert _read_run_mode(run) == "smoke"
    assert _read_seed_set(run) == {"k": 1, "reproduced": False}
    # An explicit full-scale, multi-seed run state is read through; an unknown
    # run_mode falls back to smoke (never inferred).
    (run / ".pipeline" / "driver_state.json").write_text(json.dumps(
        {"run_mode": "full", "seed_set": {"k": 5, "reproduced": True}}))
    assert _read_run_mode(run) == "full"
    assert _read_seed_set(run) == {"k": 5, "reproduced": True}
    (run / ".pipeline" / "driver_state.json").write_text(json.dumps(
        {"run_mode": "bogus"}))
    assert _read_run_mode(run) == "smoke"


def test_seed_set_reader_fails_closed_on_malformed_entries(tmp_path):
    # Red-team hardening: the seed-set reader must fail CLOSED on a malformed
    # but valid-JSON state file, never open and never crash.
    from probes.claims import _read_seed_set
    run = tmp_path / "run"
    (run / ".pipeline").mkdir(parents=True)
    ds = run / ".pipeline" / "driver_state.json"
    # A non-finite k (json accepts the bare token Infinity) must not crash the
    # battery — it falls back to the conservative single-seed default.
    ds.write_text('{"seed_set": {"k": Infinity, "reproduced": true}}')
    assert _read_seed_set(run) == {"k": 1, "reproduced": False}
    ds.write_text('{"seed_set": {"k": -Infinity, "reproduced": true}}')
    assert _read_seed_set(run) == {"k": 1, "reproduced": False}
    # `reproduced` is true only for a real JSON bool — a stringy/numeric value
    # (incl. the string "false") fails closed to False, never coerced open.
    for bad in ('"false"', '"no"', '"0"', '1', '[1]'):
        ds.write_text('{"seed_set": {"k": 3, "reproduced": %s}}' % bad)
        assert _read_seed_set(run) == {"k": 3, "reproduced": False}, bad
    # Only a real JSON true reads as reproduced.
    ds.write_text('{"seed_set": {"k": 3, "reproduced": true}}')
    assert _read_seed_set(run) == {"k": 3, "reproduced": True}


def test_battery_survives_malformed_seed_set_state(tmp_path):
    # The battery entry point must not crash on a malformed driver_state.json:
    # a non-finite k once raised an uncaught OverflowError through to
    # probe_claims_ledger. It must build the ledger as if single-seed.
    import json
    from probes.claims import probe_claims_ledger
    run = tmp_path / "run"
    (run / ".pipeline").mkdir(parents=True)
    (run / ".pipeline" / "paper_map.json").write_text(json.dumps(
        {"elements": [{"id": "exp-1", "type": "experiment",
                       "source_text": "selects diverse points."}]}))
    (run / ".pipeline" / "driver_state.json").write_text(
        '{"run_mode": "smoke", "seed_set": {"k": Infinity, "reproduced": true}}')
    v = probe_claims_ledger(run, report=None, write=False)
    assert v.verdict == "pass"
