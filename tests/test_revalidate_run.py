"""Post-delivery re-validation gate + provenance delta (R2C-019).

The design's validation sketch, pinned deterministically: an innocuous
edit keeps all verdicts, a planted claim-breaking edit flips the right
verdict and the delta names it, delivery-time evidence stays
byte-identical, and every record carries its provenance (adapted commit,
battery versions, not-delivery-truth marker). Battery drift is bucketed
separately so a newer battery can never masquerade as a regression.
"""

from __future__ import annotations

import hashlib
import json

import pytest
import subprocess
from pathlib import Path

from revalidate_run import (
    compute_delta,
    main as revalidate_main,
    next_results_dir,
    summary_line,
)
from run_probes import main as run_probes_main


def _make_delivered_run(tmp_path: Path) -> Path:
    """A minimal delivered AL run: battery baseline written in default
    mode (delivery-time truth), then git-baselined as delivered."""
    pytest.importorskip("torch")  # the delivered selector below imports it
    run = tmp_path / "delivered-run"
    (run / ".pipeline").mkdir(parents=True)
    (run / "method").mkdir()
    # A selector that genuinely passes at delivery: entropy-based, so it
    # responds to model weights (the acquisition contract's sensitivity
    # arm) and beats the degenerate nulls (the contribution floor).
    (run / "method" / "method.py").write_text(
        "import torch\n\n\n"
        "def select_batch(model, x_unlabeled, batch_size, seed):\n"
        "    model.eval()\n"
        "    with torch.no_grad():\n"
        "        probs = torch.softmax(model(x_unlabeled), dim=1)\n"
        "    entropy = -(probs * torch.log(probs + 1e-9)).sum(dim=1)\n"
        "    return torch.topk(entropy, batch_size).indices.tolist()\n")
    (run / "method" / "__init__.py").write_text(
        "from .method import select_batch\n")
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
    # Delivery-time battery, default mode: report + ledger under .pipeline/.
    assert run_probes_main(["--run-dir", str(run)]) in (0, 1)
    _git(run, "init")
    _git(run, "add", "-A")
    _git(run, "commit", "-m", "as delivered")
    return run


def _git(run: Path, *args: str) -> None:
    subprocess.run(
        ["git", "-C", str(run), "-c", "user.email=t@t", "-c",
         "user.name=t", *args],
        check=True, capture_output=True, text=True)


def _hashes_outside_results(run: Path) -> dict[str, str]:
    results_home = run / "details" / "post_delivery_validation"
    return {
        p.relative_to(run).as_posix():
            hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(run.rglob("*"))
        if p.is_file() and ".git" not in p.parts
        and not p.is_relative_to(results_home)
    }


def test_innocuous_edit_keeps_verdicts_and_freezes_delivery_truth(tmp_path):
    run = _make_delivered_run(tmp_path)
    method = run / "method" / "method.py"
    method.write_text(method.read_text() + "\n# researcher note\n")

    before = _hashes_outside_results(run)
    rc = revalidate_main(["--run-dir", str(run)])
    assert rc == 0, "an innocuous edit must not gate"

    # Everything outside the new results directory is byte-identical —
    # including the delivery-time report and ledger under .pipeline/.
    assert _hashes_outside_results(run) == before

    results = run / "details" / "post_delivery_validation" / "001"
    assert (results / "probe_report.json").is_file()
    assert (results / "claims_ledger.json").is_file()
    record = json.loads((results / "revalidation.json").read_text())

    # Provenance: not delivery truth, adapted commit, both battery stamps.
    assert record["delivery_truth"] is False
    assert record["record_type"] == "post_delivery_revalidation"
    assert record["adapted_tree"]["commit"]
    assert record["adapted_tree"]["dirty"] is True  # gate runs pre-commit
    assert record["battery"]["at_delivery"]
    assert record["battery"]["now"] == record["battery"]["at_delivery"]
    assert record["baseline_report"] == ".pipeline/probe_report.json"

    delta = record["delta"]
    assert delta["regressions"] == []
    assert delta["battery_drift"]["checks_added"] == []
    assert delta["battery_drift"]["checks_removed"] == []
    assert delta["battery_drift"]["attribution"] == "adaptation"
    assert "no regressions" in record["summary"]
    assert "details/post_delivery_validation/001" in record["summary"]


def test_claim_breaking_edit_flips_the_right_verdict_and_gates(tmp_path):
    run = _make_delivered_run(tmp_path)
    baseline = json.loads(
        (run / ".pipeline" / "probe_report.json").read_text())
    passed_at_delivery = {
        v["probe_id"] for v in baseline["verdicts"]
        if v["verdict"] == "pass"}
    (run / "method" / "method.py").write_text(
        "def select_batch(model, x_unlabeled, batch_size, seed):\n"
        "    raise RuntimeError('adaptation broke the selector')\n")

    rc = revalidate_main(["--run-dir", str(run)])
    assert rc == 1, "a regression must gate"

    results = run / "details" / "post_delivery_validation" / "001"
    record = json.loads((results / "revalidation.json").read_text())
    regressed = {e["check"] for e in record["delta"]["regressions"]}
    assert regressed, "the planted break must appear as a regression"
    assert regressed <= passed_at_delivery, (
        "regressions must be checks that passed at delivery")
    for check in regressed:
        assert check in record["summary"], "the delta names the flip"
    # The delivery-time baseline is still the frozen comparison point.
    assert json.loads(
        (run / ".pipeline" / "probe_report.json").read_text()) == baseline


def test_results_directories_are_sequenced(tmp_path):
    run = _make_delivered_run(tmp_path)
    assert revalidate_main(["--run-dir", str(run)]) == 0
    assert revalidate_main(["--run-dir", str(run)]) == 0
    home = run / "details" / "post_delivery_validation"
    assert sorted(p.name for p in home.iterdir()) == ["001", "002"]
    assert next_results_dir(run).name == "003"


def test_refusals_are_honest_and_specific(tmp_path, capsys):
    # No run dir at all.
    assert revalidate_main(["--run-dir", str(tmp_path / "nope")]) == 2

    run = _make_delivered_run(tmp_path)

    # A live run is read-only.
    progress = run / ".pipeline" / "progress.json"
    progress.write_text(json.dumps({"run_status": "running"}))
    assert revalidate_main(["--run-dir", str(run)]) == 2
    assert "live" in capsys.readouterr().err
    progress.unlink()

    # No delivery-time baseline: honest absence, never a fabricated one.
    baseline = run / ".pipeline" / "probe_report.json"
    saved = baseline.read_text()
    baseline.unlink()
    assert revalidate_main(["--run-dir", str(run)]) == 2
    assert "predates the battery record" in capsys.readouterr().err
    baseline.write_text(saved)

    # No git baseline: the adapted state would be unrecordable.
    no_git = tmp_path / "no-git-run"
    no_git.mkdir()
    (no_git / ".pipeline").mkdir()
    (no_git / ".pipeline" / "probe_report.json").write_text(saved)
    assert revalidate_main(["--run-dir", str(no_git)]) == 2
    assert "finalize_run_git" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# Delta semantics as pure-function contracts (no battery run needed).
# ---------------------------------------------------------------------------

def _report(version, **verdicts_by_id):
    return {
        "battery_version": version,
        "verdicts": [
            {"probe_id": pid, "verdict": verdict, "message": f"{pid} msg"}
            for pid, verdict in verdicts_by_id.items()
        ],
    }


def test_delta_classifies_flips_and_still_failing():
    baseline = _report("v1", **{"US-10": "pass", "UB-6": "fail",
                                "AL-5": "pass", "CT-1": "warn"})
    current = _report("v1", **{"US-10": "pass", "UB-6": "fail",
                               "AL-5": "fail", "CT-1": "pass"})
    delta = compute_delta(baseline, current)
    assert [e["check"] for e in delta["regressions"]] == ["AL-5"]
    assert delta["regressions"][0]["at_delivery"] == "pass"
    assert delta["regressions"][0]["now"] == "fail"
    assert [e["check"] for e in delta["improvements"]] == ["CT-1"]
    assert [e["check"] for e in
            delta["still_failing_from_delivery"]] == ["UB-6"]
    assert delta["unchanged"] == 1  # US-10


def test_delta_aggregates_multi_row_checks_by_worst_verdict():
    baseline = {"battery_version": "v1", "verdicts": [
        {"probe_id": "US-10", "verdict": "pass", "message": "clean"}]}
    current = {"battery_version": "v1", "verdicts": [
        {"probe_id": "US-10", "verdict": "fail", "message": "row 1"},
        {"probe_id": "US-10", "verdict": "fail", "message": "row 2"}]}
    delta = compute_delta(baseline, current)
    assert len(delta["regressions"]) == 1, "checks compare, not rows"


def test_not_applicable_is_distinct_and_frozen_revalidation_tracks_coverage():
    became_conditional = compute_delta(
        _report("v1", **{"TSF-2": "pass"}),
        _report("v1", **{"TSF-2": "not_applicable"}),
    )
    assert [row["check"] for row in became_conditional["regressions"]] == [
        "TSF-2"
    ]
    assert became_conditional["improvements"] == []
    assert became_conditional["unchanged"] == 0

    became_applicable = compute_delta(
        _report("v1", **{"TSF-2": "not_applicable"}),
        _report("v1", **{"TSF-2": "pass"}),
    )
    assert became_applicable["regressions"] == []
    assert [row["check"] for row in became_applicable["improvements"]] == [
        "TSF-2"
    ]
    assert became_applicable["unchanged"] == 0

    # A multi-row conditional check is applicable when any row actually ran;
    # list order must not let an earlier not-applicable row hide that pass.
    current = {
        "battery_version": "v1",
        "verdicts": [
            {"probe_id": "TSF-2", "verdict": "not_applicable",
             "message": "graph-free branch"},
            {"probe_id": "TSF-2", "verdict": "pass",
             "message": "autoregressive branch passed"},
        ],
    }
    mixed = compute_delta(_report("v1", **{"TSF-2": "pass"}), current)
    assert mixed["regressions"] == []
    assert mixed["unchanged"] == 1

    # Not-applicable is neutral, not an amnesty for a later applicable fail.
    failed = compute_delta(
        _report("v1", **{"TSF-2": "not_applicable"}),
        _report("v1", **{"TSF-2": "fail"}),
    )
    assert [row["check"] for row in failed["regressions"]] == ["TSF-2"]


def test_battery_drift_bucketing_by_version_stamp():
    # Same version: check-set change is attributable to the adaptation.
    same = compute_delta(_report("v1", **{"US-10": "pass"}),
                         _report("v1", **{"US-10": "pass",
                                          "AL-6": "pass"}))
    assert same["battery_drift"]["checks_added"] == ["AL-6"]
    assert same["battery_drift"]["attribution"] == "adaptation"

    # Different version: drift bucket, never a regression/improvement.
    diff = compute_delta(_report("v1", **{"US-10": "pass",
                                          "MP-1": "pass"}),
                         _report("v2", **{"US-10": "pass",
                                          "AL-6": "fail"}))
    assert diff["battery_drift"]["checks_added"] == ["AL-6"]
    assert diff["battery_drift"]["checks_removed"] == ["MP-1"]
    assert diff["battery_drift"]["attribution"] == (
        "battery_changed_since_delivery")
    assert diff["regressions"] == []  # AL-6 is drift, not a flip

    # Unstamped baseline (pre-stamp delivery): honest unknown.
    unknown = compute_delta({"verdicts": []},
                            _report("v2", **{"US-10": "pass"}))
    assert unknown["battery_drift"]["attribution"] == "unknown"


def test_summary_line_names_flips_drift_and_results_home():
    delta = compute_delta(
        _report("v1", **{"US-10": "pass", "AL-5": "pass"}),
        _report("v1", **{"US-10": "pass", "AL-5": "fail",
                         "AL-6": "pass"}))
    line = summary_line(delta, "details/post_delivery_validation/004")
    assert "1 regression(s): AL-5" in line
    assert "check set drifted (+1/-0, adaptation)" in line
    assert "details/post_delivery_validation/004" in line
