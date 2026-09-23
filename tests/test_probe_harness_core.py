"""Slice 1.2 harness-core tests: fixtures, loader, stub kit, verdict schema.

The fixture tests double as the harness's SELF-CALIBRATION (the v3
smoke_method_harness pattern): a fixture that a trivial classifier can't ace,
or whose scale knob lies, would make every downstream probe untrustworthy —
so those properties are pinned here before any probe exists.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parent.parent
ZOO = REPO / "tests" / "fixtures" / "zoo"

from probes import ProbeReport, ProbeVerdict  # noqa: E402
from probes.fixtures import (  # noqa: E402
    make_classification_fixture,
    make_kd_logits_fixture,
    make_planning_fixture,
    nearest_prototype_accuracy,
)
from probes.package_loader import (  # noqa: E402
    ProbeLoadError,
    StubModel,
    imported_method_package,
    load_module_from_path,
    make_al_stub_kit,
)


# ---------------------------------------------------------------------------
# Fixtures: determinism, scale knob, guaranteed signal (self-calibration)
# ---------------------------------------------------------------------------


def test_classification_fixture_is_seed_deterministic():
    a = make_classification_fixture(seed=7)
    b = make_classification_fixture(seed=7)
    c = make_classification_fixture(seed=8)
    np.testing.assert_array_equal(a.x_pool, b.x_pool)
    np.testing.assert_array_equal(a.y_test, b.y_test)
    assert not np.array_equal(a.x_pool, c.x_pool)


@pytest.mark.parametrize("scale,lo,hi", [
    ("zero_one", 0.0, 1.0),
    ("raw255", 0.0, 255.0),
])
def test_scale_knob_produces_advertised_ranges(scale, lo, hi):
    f = make_classification_fixture(scale=scale, seed=3)
    vmin, vmax = f.value_range
    assert lo <= vmin <= vmax <= hi
    # The knob must meaningfully span its range (US-4 measures against this).
    assert vmax - vmin > 0.5 * (hi - lo)


@pytest.mark.parametrize("scale", ["standardized", "zero_one", "raw255"])
def test_fixture_has_guaranteed_signal_at_every_scale(scale):
    # Self-calibration: a nearest-prototype classifier must ace the fixture;
    # otherwise above-chance probes (UB-5/UB-6) can't be trusted to gate.
    f = make_classification_fixture(scale=scale, seed=11)
    assert nearest_prototype_accuracy(f) >= 0.9


def test_planning_fixture_dynamic_vs_frozen_contrast():
    dyn = make_planning_fixture(dynamic=True, seed=5)
    frozen = make_planning_fixture(dynamic=False, seed=5)
    # Dynamic tracks move; frozen tracks don't — the MP-1 (M-004) contrast.
    assert not np.allclose(dyn.obstacle_tracks[:, 0], dyn.obstacle_tracks[:, -1])
    assert np.allclose(frozen.obstacle_tracks[:, 0], frozen.obstacle_tracks[:, -1])
    # Dict-shape adapter carries motion into the x/y vs x_prev/y_prev pairs.
    states = dyn.obstacle_states_at(3)
    assert any(s["y"] != s["y_prev"] for s in states)
    states0 = frozen.obstacle_states_at(3)
    assert all(s["y"] == s["y_prev"] for s in states0)


def test_kd_fixture_teacher_is_mostly_correct():
    f = make_kd_logits_fixture(seed=2)
    teacher_pred = f.teacher_logits.argmax(axis=1)
    assert (teacher_pred == f.labels).mean() >= 0.8


# ---------------------------------------------------------------------------
# Loader: lone files, full packages, missing deps → unprobeable material
# ---------------------------------------------------------------------------


def test_load_module_from_path_runs_real_zoo_artifact():
    # pdwa method.py is numpy-only — loads anywhere the suite runs.
    mod = load_module_from_path(ZOO / "pdwa-omega-degenerate" / "method.py")
    assert callable(mod.select_velocity)
    assert callable(mod.evaluate_objective_function)


def test_load_module_missing_dependency_is_named(tmp_path):
    f = tmp_path / "method.py"
    f.write_text("import definitely_not_installed_xyz\n")
    with pytest.raises(ProbeLoadError) as exc:
        load_module_from_path(f)
    assert exc.value.missing_dependency == "definitely_not_installed_xyz"


def test_imported_method_package_isolates_and_restores(tmp_path):
    run = tmp_path / "run"
    pkg = run / "method"
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text("from .core import answer\n")
    (pkg / "core.py").write_text("def answer():\n    return 42\n")
    import sys
    with imported_method_package(run) as method:
        assert method.answer() == 42
        assert "method" in sys.modules
    assert "method" not in sys.modules  # restored


def test_imported_method_package_requires_method_dir(tmp_path):
    with pytest.raises(ProbeLoadError):
        with imported_method_package(tmp_path):
            pass


# ---------------------------------------------------------------------------
# Stub kit: recording, adversarial positions, fresh-model fingerprints
# ---------------------------------------------------------------------------


def test_al_stub_kit_records_and_adversarial_positions_vary_by_round():
    kit = make_al_stub_kit(batch_size=3)
    r0 = kit["select_batch"](object(), np.zeros((10, 2)), 3, 0)
    r1 = kit["select_batch"](object(), np.zeros((10, 2)), 3, 1)
    assert r0 == [0, 1, 2] and r1 == [1, 2, 3]  # shifted per round
    assert kit["select_batch"].call_count == 2


def test_stub_model_fingerprints_distinguish_fresh_from_reused():
    kit = make_al_stub_kit(batch_size=2)
    m1 = kit["build_model"]()
    m2 = kit["build_model"]()
    assert m1.fingerprint != m2.fingerprint
    kit["train_from_scratch"](m1)
    kit["train_from_scratch"](m1)
    # A warm-started loop shows one fingerprint with multiple train marks —
    # exactly the AL-4 (M-003) signal.
    assert m1.train_marks == 2
    assert isinstance(m1, StubModel)


# ---------------------------------------------------------------------------
# Verdict schema
# ---------------------------------------------------------------------------


def test_probe_report_buckets_and_serializes():
    report = ProbeReport(target="zoo/example")
    report.add(ProbeVerdict("UB-2", "fail", "constant scores", evidence="prior=1.0 x6"))
    report.add(ProbeVerdict("UB-3", "flag_for_researcher", "ω-degenerate",
                            finding_class="M-005"))
    report.add(ProbeVerdict("AL-1", "unprobeable", "loop not extractable"))
    report.add(ProbeVerdict("UB-8", "pass", "seed threads"))
    assert [v.probe_id for v in report.gating_failures] == ["UB-2"]
    assert {v.probe_id for v in report.needs_review} == {"UB-3", "AL-1"}
    d = report.to_dict()
    assert d["counts"]["pass"] == 1 and d["counts"]["fail"] == 1


def test_unknown_verdict_rejected():
    with pytest.raises(ValueError):
        ProbeVerdict("UB-1", "maybe", "nope")


def test_not_applicable_is_closed_non_demoting_verdict():
    report = ProbeReport(target="graph-free-control")
    verdict = ProbeVerdict(
        "UB-1", "not_applicable", "conditional graph probe does not apply",
    )
    report.add(verdict)

    assert report.gating_failures == []
    assert report.needs_review == []
    assert report.counts()["not_applicable"] == 1


def test_probe_ref_is_additive_stable_and_roundtrips():
    archived = ProbeVerdict("UB-1", "pass", "legacy")
    assert archived.probe_ref == ""
    assert archived.to_dict()["probe_ref"] == ""

    current = ProbeVerdict(
        "UB-1",
        "pass",
        "sample spread is genuine",
        probe_ref="probes.time_series.sample_genuineness",
    )
    assert current.to_dict()["probe_ref"] == (
        "probes.time_series.sample_genuineness"
    )


def test_probe_ref_normalizes_whitespace_and_rejects_non_string():
    normalized = ProbeVerdict(
        "UB-1",
        "pass",
        "sample spread is genuine",
        probe_ref="  probes.time_series.sample_genuineness  ",
    )
    assert normalized.probe_ref == "probes.time_series.sample_genuineness"
    assert normalized.to_dict()["probe_ref"] == (
        "probes.time_series.sample_genuineness"
    )
    assert ProbeVerdict("UB-1", "pass", "legacy", probe_ref="   ").probe_ref == ""

    with pytest.raises(ValueError, match="probe_ref must be a string"):
        ProbeVerdict(
            "UB-1",
            "pass",
            "invalid ref type",
            probe_ref=123,  # type: ignore[arg-type]
        )


# ---------------------------------------------------------------------------
# CT-3 §1.1 path join key: element_ids on ProbeVerdict, related_elements on
# Finding. Both additive, both default empty (the fail-closed degenerate
# state), both round-trip — so the claims ledger CAN ask "is this claim's path
# fidelity-clean?" against a shared key, never just a file-plus-prose match.
# ---------------------------------------------------------------------------


def test_probe_verdict_element_ids_default_empty_and_roundtrip():
    bare = ProbeVerdict("UB-7", "pass", "all channels live")
    assert bare.element_ids == []                      # default: empty join key
    assert bare.to_dict()["element_ids"] == []         # serializes
    keyed = ProbeVerdict("UB-7", "fail", "dead prior channel",
                         element_ids=["exp-main", "eq-10"])
    assert keyed.to_dict()["element_ids"] == ["exp-main", "eq-10"]
    # Independent instances never share the mutable default.
    bare.element_ids.append("leak")
    assert ProbeVerdict("UB-1", "pass", "x").element_ids == []


def test_finding_related_elements_default_empty_and_additive():
    from schemas.review_report import Finding
    bare = Finding(
        id="F001", severity="important", target_agent="method-coder",
        issue_type="algorithm_divergence", file="method/method.py",
        location="select_batch, lines 30-50",
        description="transposed cost matrix",
    )
    assert bare.related_elements == []                 # default: empty join key
    keyed = Finding(
        id="F002", severity="critical", target_agent="human",
        issue_type="algorithm_divergence", file="method/method.py",
        location="cell 21", description="sign flip",
        related_elements=["exp-main"],
    )
    assert keyed.related_elements == ["exp-main"]


def test_committed_review_reports_still_validate_without_the_join_key():
    # The field is additive + optional, so every committed review_report.json
    # (written before the join key existed) must still parse against the model.
    #
    # Scope: `r2c_runs/` is unversioned live-run scratch and always skipped
    # so local artifacts cannot break the suite. A curated example whose own
    # KNOWN_ISSUES.md machine-readably discloses that its review report
    # failed validation is also skipped — the 2026-07-07 pdwa example was
    # curated exactly for that honestly-disclosed degrade (issue_type
    # outside the schema vocabulary), and this test must not force the
    # disclosure to be papered over.
    import json
    from schemas.review_report import ReviewReport
    seen = 0
    for path in sorted((REPO / "tests" / "fixtures").rglob("review_report.json")):
        rel_parts = path.relative_to(REPO).parts
        if rel_parts[0] == "r2c_runs":
            continue
        known_issues = path.parent.parent / "details" / "KNOWN_ISSUES.md"
        if known_issues.is_file():
            disclosures = [
                line for line in known_issues.read_text(encoding="utf-8").splitlines()
                if "issue_signature:" in line and "review_report.json" in line
            ]
            if disclosures:
                continue
        ReviewReport.model_validate(json.loads(path.read_text(encoding="utf-8")))
        seen += 1
    assert seen >= 1, "expected at least one committed review_report.json"


def test_pluggable_from_spec_reads_the_june9_spec():
    from probes.package_loader import pluggable_from_spec
    name, signature = pluggable_from_spec(
        REPO / "tests" / "fixtures" / "evidence" / "june9-gbald-run"
        / "pipeline" / "method_spec.json"
    )
    assert name and name in signature
