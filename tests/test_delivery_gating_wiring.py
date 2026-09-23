"""Slice 1.5 driver wiring: battery at delivery, label into manifest+README.

The battery subprocess is patched everywhere — these tests exercise the
wiring contracts: a crashed battery delivers draft (never halts), the
label lands in final_manifest.json's delivery section, and the README
banner is idempotent.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import run_layout
import run_pipeline
from generate_method_md import UNAVAILABLE
from run_pipeline import finalize_delivery_banner, run_delivery_gating
from tests.helpers.state import make_state

CLEAN_REPORT = {"target": "x", "counts": {},
                "verdicts": [{"probe_id": "US-1", "verdict": "pass",
                              "message": "ok"},
                             {"probe_id": "CT-1", "verdict": "pass",
                              "message": "contribution evidence"}]}
UNIVERSAL_ONLY_REPORT = {"target": "x", "counts": {},
                         "verdicts": [{"probe_id": "US-1", "verdict": "pass",
                                       "message": "ok"}]}
FLAGGED_REPORT = {"target": "x", "counts": {},
                  "verdicts": [{"probe_id": "UB-7",
                                "verdict": "flag_for_researcher",
                                "message": "dead term"}]}


def _patch_battery(monkeypatch, state, report: dict | None,
                   exc: Exception | None = None,
                   *, install_legacy_spec: bool = True):
    spec_path = state.paths.pipeline_dir / "method_spec.json"
    if install_legacy_spec and not spec_path.is_file():
        spec_path.write_text(json.dumps({
            "schema_version": "1.11.0",
            "comparison": {"classification": {"id": "active_learning"}},
            "methodology_replication_contract": {"elements": []},
        }))

    def fake_run(cmd, **kwargs):
        if exc is not None:
            raise exc
        if report is not None:
            (state.paths.pipeline_dir / "probe_report.json").write_text(
                json.dumps(report))
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
    monkeypatch.setattr(run_pipeline.subprocess, "run", fake_run)


def _events(state) -> list[dict]:
    path = state.paths.pipeline_dir / "run_events.jsonl"
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text().splitlines()]


def _append_smoke_event(state, event_type, details):
    run_pipeline._append_run_event(
        state.paths, event_type, stage_id="stage_3c",
        status="passed" if event_type == "smoke_passed" else "failed",
        summary=f"smoke_run_notebook.py {event_type}", details=details)


def _sha256_of(path) -> str:
    import hashlib
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_notebook_digest_mismatch_demotes_to_draft(tmp_path, monkeypatch):
    # R2C-039 block 2: a delivered notebook that differs from the one the
    # smoke gate verified cannot present itself as verified.
    state = make_state(tmp_path / "run")
    nb = state.paths.run_dir / "notebook.ipynb"
    nb.write_text('{"cells": []}', encoding="utf-8")
    _append_smoke_event(state, "smoke_passed",
                        {"exit_code": 0, "notebook_sha256": _sha256_of(nb)})
    nb.write_text('{"cells": [], "modified_after_smoke": true}',
                  encoding="utf-8")
    _patch_battery(monkeypatch, state, CLEAN_REPORT)

    delivery = run_delivery_gating(state)

    assert delivery["label"] == "draft"
    ids = [r.get("id") for r in delivery["reasons"]]
    assert "notebook_digest_mismatch" in ids


def test_notebook_digest_match_stays_verified(tmp_path, monkeypatch):
    state = make_state(tmp_path / "run")
    nb = state.paths.run_dir / "notebook.ipynb"
    nb.write_text('{"cells": []}', encoding="utf-8")
    _append_smoke_event(state, "smoke_passed",
                        {"exit_code": 0, "notebook_sha256": _sha256_of(nb)})
    _patch_battery(monkeypatch, state, CLEAN_REPORT)

    delivery = run_delivery_gating(state)

    assert delivery["label"] == "verified"


def test_graph_delivery_receives_reconstructed_execution_plan(
    tmp_path, monkeypatch,
):
    import build_probe_harness
    import delivery_label

    state = make_state(tmp_path / "run")
    graph_spec = {
        "schema_version": "1.14.0",
        "comparison": {"classification": {"id": "graph_fixture"}},
        "methodology_replication_contract": {
            "elements": [],
            "homogeneous_graph_mechanism": {},
        },
    }
    (state.paths.pipeline_dir / "method_spec.json").write_text(
        json.dumps(graph_spec), encoding="utf-8"
    )
    _patch_battery(
        monkeypatch, state, UNIVERSAL_ONLY_REPORT,
        install_legacy_spec=False,
    )
    sentinel_plan = {"schema_version": "1.0", "status": "ready"}
    observed_context = {}

    def fake_freezer(run_dir, declared_context):
        assert run_dir == state.paths.run_dir
        observed_context.update(declared_context)
        return sentinel_plan

    monkeypatch.setattr(
        build_probe_harness,
        "_frozen_graph_mechanism_execution_plan",
        fake_freezer,
    )
    original = delivery_label.derive_delivery_label
    observed = {}

    def spy(*args, **kwargs):
        observed["plan"] = kwargs.get("graph_execution_plan")
        return original(*args, **kwargs)

    monkeypatch.setattr(delivery_label, "derive_delivery_label", spy)

    run_delivery_gating(state)

    assert observed["plan"] is sentinel_plan
    assert "graph_mechanism.alignment_prerequisite" in observed_context


def test_digest_check_exempts_smoke_failed_deliveries(tmp_path, monkeypatch):
    # The environmental-degrade path (smoke exit 3, recorded as
    # smoke_failed) is already disclosed as rendered-but-not-verified;
    # the digest invariant must not double-demote it against a stale
    # earlier pass.
    state = make_state(tmp_path / "run")
    nb = state.paths.run_dir / "notebook.ipynb"
    nb.write_text('{"cells": []}', encoding="utf-8")
    _append_smoke_event(state, "smoke_passed",
                        {"exit_code": 0, "notebook_sha256": _sha256_of(nb)})
    _append_smoke_event(state, "smoke_failed", {"exit_code": 3})
    nb.write_text('{"cells": [], "rerendered": true}', encoding="utf-8")
    _patch_battery(monkeypatch, state, CLEAN_REPORT)

    delivery = run_delivery_gating(state)

    assert all(r.get("id") != "notebook_digest_mismatch"
               for r in delivery["reasons"])


def test_digest_check_skips_pre_invariant_smoke_events(tmp_path, monkeypatch):
    # A smoke_passed recorded before the digest field existed carries
    # nothing to compare; resumed older runs must not demote.
    state = make_state(tmp_path / "run")
    nb = state.paths.run_dir / "notebook.ipynb"
    nb.write_text('{"cells": []}', encoding="utf-8")
    _append_smoke_event(state, "smoke_passed", {"exit_code": 0})
    _patch_battery(monkeypatch, state, CLEAN_REPORT)

    delivery = run_delivery_gating(state)

    assert delivery["label"] == "verified"


def test_smoke_gate_records_verified_notebook_digest(tmp_path, fake_subprocess):
    state = make_state(tmp_path / "run")
    nb = state.paths.run_dir / "notebook.ipynb"
    nb.write_text('{"cells": []}', encoding="utf-8")
    fake_subprocess.expect_script(returncode=0)

    exit_code, _, _ = run_pipeline._run_smoke_gate(state)

    assert exit_code == 0
    smoke_events = [e for e in _events(state)
                    if e["event_type"] == "smoke_passed"]
    assert smoke_events[-1]["details"]["notebook_sha256"] == _sha256_of(nb)


def test_clean_battery_yields_verified_and_logs_event(tmp_path, monkeypatch):
    state = make_state(tmp_path / "run")
    _patch_battery(monkeypatch, state, CLEAN_REPORT)
    delivery = run_delivery_gating(state)
    assert delivery["label"] == "verified"
    events = [e for e in _events(state)
              if e["event_type"] == "delivery_label_derived"]
    assert len(events) == 1
    assert events[0]["details"]["label"] == "verified"


def test_missing_method_spec_cannot_activate_legacy_contribution_evidence(
    tmp_path, monkeypatch,
):
    state = make_state(tmp_path / "run")
    _patch_battery(
        monkeypatch,
        state,
        CLEAN_REPORT,
        install_legacy_spec=False,
    )

    delivery = run_delivery_gating(state)

    assert delivery["label"] == "draft"
    assert delivery["reasons"][0]["source"] == "method_spec"
    assert "is missing" in delivery["reasons"][0]["message"]


def test_malformed_method_spec_cannot_activate_legacy_contribution_evidence(
    tmp_path, monkeypatch,
):
    state = make_state(tmp_path / "run")
    (state.paths.pipeline_dir / "method_spec.json").write_text("{bad json")
    _patch_battery(monkeypatch, state, CLEAN_REPORT)

    delivery = run_delivery_gating(state)

    assert delivery["label"] == "draft"
    assert delivery["reasons"][0]["source"] == "method_spec"
    assert "malformed" in delivery["reasons"][0]["message"]


def test_unreadable_method_spec_demotes_instead_of_escaping(
    tmp_path, monkeypatch,
):
    state = make_state(tmp_path / "run")
    _patch_battery(monkeypatch, state, CLEAN_REPORT)
    original_read_text = Path.read_text

    def fail_spec_read(path, *args, **kwargs):
        if path == state.paths.method_spec:
            raise OSError("synthetic read failure")
        return original_read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", fail_spec_read)

    delivery = run_delivery_gating(state)

    assert delivery["label"] == "draft"
    assert delivery["reasons"][0]["source"] == "method_spec"
    assert "unreadable" in delivery["reasons"][0]["message"]


def test_universal_only_passes_yield_uncertified_with_family(tmp_path,
                                                             monkeypatch):
    # The false-verified hole, closed at the wiring layer: zero demoters
    # without contribution evidence delivers uncertified — new territory,
    # and the taxonomy node id from method_spec.json lands as the
    # growth-engine demand key.
    state = make_state(tmp_path / "run")
    (state.paths.pipeline_dir / "method_spec.json").write_text(json.dumps({
        "comparison": {"classification": {"id": "trajectory_forecasting"}}}))
    _patch_battery(monkeypatch, state, UNIVERSAL_ONLY_REPORT)
    delivery = run_delivery_gating(state)
    assert delivery["label"] == "uncertified_new_territory"
    assert delivery["missing_probe_family"] == "trajectory_forecasting"


def test_flag_yields_draft(tmp_path, monkeypatch):
    state = make_state(tmp_path / "run")
    _patch_battery(monkeypatch, state, FLAGGED_REPORT)
    delivery = run_delivery_gating(state)
    assert delivery["label"] == "draft"
    assert delivery["reasons"][0]["id"] == "UB-7"


def test_unresolved_important_review_finding_yields_draft(tmp_path,
                                                          monkeypatch):
    state = make_state(tmp_path / "run")
    (state.paths.pipeline_dir / "review_report.json").write_text(json.dumps({
        "review_status": "issues_found",
        "findings": [{"id": "F003", "severity": "important",
                      "description": "provenance imprecise"}]}))
    _patch_battery(monkeypatch, state, CLEAN_REPORT)
    delivery = run_delivery_gating(state)
    assert delivery["label"] == "draft"
    assert delivery["reasons"][0]["id"] == "F003"


def test_battery_crash_delivers_draft_never_raises(tmp_path, monkeypatch):
    state = make_state(tmp_path / "run")
    _patch_battery(monkeypatch, state, None,
                   exc=subprocess.TimeoutExpired(cmd="x", timeout=900))
    delivery = run_delivery_gating(state)
    assert delivery["label"] == "draft"
    assert delivery["reasons"][0]["source"] == "battery"
    assert "timed out" in delivery["reasons"][0]["message"]


def test_no_report_written_delivers_draft(tmp_path, monkeypatch):
    state = make_state(tmp_path / "run")
    _patch_battery(monkeypatch, state, None)  # battery "ran", wrote nothing
    delivery = run_delivery_gating(state)
    assert delivery["label"] == "draft"
    assert any(r["source"] == "battery" for r in delivery["reasons"])


# ---------------------------------------------------------------------------
# Neutral-plan label cap (R2C-032, approved 2026-07-28): a run built on the
# generic provisional build plan caps at uncertified — new territory, however
# clean the battery; a pack whose plan inherits a committed ancestor keeps
# the full label range. Fixture id re-pointed at promotion (R2C-029,
# 2026-07-28): the SRL id these tests were written with is now a committed
# node (whose exact-match plan would defeat the gap shape under test), so
# the fixtures follow test_provisional_pack_overlay's synthetic gap id.
# ---------------------------------------------------------------------------


def _write_gap_spec(state):
    from tests.test_provisional_pack_overlay import GAP_PARADIGM_ID

    (state.paths.pipeline_dir / "method_spec.json").write_text(json.dumps({
        "comparison": {
            "classification": {"id": GAP_PARADIGM_ID},
            "pluggable_component": {
                "name": "plan",
                "signature": "plan(start, goal, environment, dynamics, seed=0)",
            },
        },
    }))


def _install_gap_pack(state, pack):
    import yaml

    install_dir = state.paths.provisional_packs_dir / "20260728-srl"
    install_dir.mkdir(parents=True)
    (install_dir / "pack.yaml").write_text(yaml.safe_dump(pack),
                                           encoding="utf-8")


def test_neutral_plan_run_delivers_uncertified_not_verified(tmp_path,
                                                            monkeypatch):
    from delivery_label import NEUTRAL_PLAN_CAP_ID
    from tests.test_provisional_pack_overlay import PACK_NEUTRAL

    state = make_state(tmp_path / "run")
    _write_gap_spec(state)
    _install_gap_pack(state, PACK_NEUTRAL)
    _patch_battery(monkeypatch, state, CLEAN_REPORT)
    delivery = run_delivery_gating(state)
    assert delivery["label"] == "uncertified_new_territory"
    assert any(d["id"] == NEUTRAL_PLAN_CAP_ID
               for d in delivery["disclosures"])
    from tests.test_provisional_pack_overlay import GAP_PARADIGM_ID
    assert delivery["missing_probe_family"] == GAP_PARADIGM_ID
    # The run event records the capped label, never the pre-cap one.
    events = [e for e in _events(state)
              if e["event_type"] == "delivery_label_derived"]
    assert events[0]["details"]["label"] == "uncertified_new_territory"


def test_inherited_plan_run_keeps_full_range(tmp_path, monkeypatch):
    # Control: a pack inheriting a fitting committed parent plan keeps the
    # full label range (the committed no-pack baseline is
    # test_clean_battery_yields_verified_and_logs_event, unmodified).
    from tests.test_provisional_pack_overlay import PACK_INHERIT

    state = make_state(tmp_path / "run")
    _write_gap_spec(state)
    _install_gap_pack(state, PACK_INHERIT)
    _patch_battery(monkeypatch, state, CLEAN_REPORT)
    delivery = run_delivery_gating(state)
    assert delivery["label"] == "verified"


def test_battery_crash_on_neutral_run_stays_capped(tmp_path, monkeypatch):
    # The post-hoc draft demotions must not RAISE a capped label: a battery
    # error on a neutral-plan run stays uncertified, with the error fully
    # disclosed in `reasons`.
    from tests.test_provisional_pack_overlay import PACK_NEUTRAL

    state = make_state(tmp_path / "run")
    _write_gap_spec(state)
    _install_gap_pack(state, PACK_NEUTRAL)
    _patch_battery(monkeypatch, state, None,
                   exc=subprocess.TimeoutExpired(cmd="x", timeout=900))
    delivery = run_delivery_gating(state)
    assert delivery["label"] == "uncertified_new_territory"
    assert any(r["source"] == "battery" for r in delivery["reasons"])


def test_manifest_accepts_capped_uncertified_delivery(tmp_path):
    # Pins the no-schema-change claim: a capped delivery (reasons-bearing
    # uncertified + the build_plan-source disclosure) round-trips through
    # DeliveryVerdict untouched.
    from delivery_label import derive_delivery_label
    from final_manifest import build_final_manifest

    state = make_state(tmp_path / "run")
    state.paths.method_spec.write_text(json.dumps({
        "schema_version": "1.14.0",
        "methodology_replication_contract": {"elements": []},
    }))
    capped = derive_delivery_label(
        {"verdicts": [
            {"probe_id": "UB-7", "verdict": "flag_for_researcher",
             "message": "dead term"},
            {"probe_id": "CT-1", "verdict": "pass", "message": "ok"},
        ]},
        None, paradigm="motion_planning/synthetic_gap_variant",
        neutral_build_plan=True)
    manifest = build_final_manifest(state.paths, stage_results=[],
                                    delivery=capped)
    dumped = manifest.model_dump()
    assert dumped["delivery"]["label"] == "uncertified_new_territory"
    assert dumped["delivery"]["reasons"], "demoting reasons must stay visible"
    assert any(d["id"] == "neutral_plan_cap"
               for d in dumped["delivery"]["disclosures"])


# ---------------------------------------------------------------------------
# README banner
# ---------------------------------------------------------------------------


def _draft_delivery(n=2):
    return {"label": "draft",
            "reasons": [{"source": "probe", "id": f"P{i}", "message": "m"}
                        for i in range(n)],
            "disclosures": [], "probe_counts": {}}


def test_banner_prepends_and_replaces_idempotently(tmp_path):
    state = make_state(tmp_path / "run")
    readme = run_layout.run_path(state.paths.run_dir, run_layout.PACKAGE_README)
    readme.write_text("# Package\n\nbody\n")
    finalize_delivery_banner(state, _draft_delivery(2))
    first = readme.read_text()
    assert first.startswith("> **Delivery: draft** - 2 findings")
    assert "[`REPORT.md`](../REPORT.md)" in first
    assert "# Package" in first
    # Re-running with a different verdict replaces, never stacks.
    finalize_delivery_banner(
        state, {"label": "verified", "reasons": [], "disclosures": [],
                "probe_counts": {}})
    second = readme.read_text()
    assert second.count("**Delivery: ") == 1
    assert second.startswith("> **Delivery: verified**")
    assert "[`REPORT.md`](../REPORT.md)" in second
    assert "# Package" in second


def test_banner_singular_grammar(tmp_path):
    state = make_state(tmp_path / "run")
    run_layout.run_path(state.paths.run_dir,
                        run_layout.PACKAGE_README).write_text("body\n")
    finalize_delivery_banner(state, _draft_delivery(1))
    assert "1 finding needs your attention" in \
        (state.paths.run_dir / run_layout.PACKAGE_README).read_text()


def test_explanation_only_banner(tmp_path):
    state = make_state(tmp_path / "run")
    run_layout.run_path(state.paths.run_dir,
                        run_layout.PACKAGE_README).write_text("# Package\n")
    finalize_delivery_banner(state, {
        "label": "explanation_only",
        "reasons": [{"source": "halt", "id": "stage_1", "message": "gap"}],
        "disclosures": [],
        "probe_counts": {},
    })

    out = (state.paths.run_dir / run_layout.PACKAGE_README).read_text()
    assert out.startswith("> **Delivery: explanation only**")
    assert "method explanation was not produced before the halt" in out
    assert "METHOD.md" not in out
    assert "KNOWN_ISSUES.md" in out

    (state.paths.run_dir / "METHOD.md").write_text("# Method\n")
    finalize_delivery_banner(state, {
        "label": "explanation_only",
        "reasons": [{"source": "halt", "id": "stage_1", "message": "gap"}],
        "disclosures": [],
        "probe_counts": {},
    })
    out = (state.paths.run_dir / run_layout.PACKAGE_README).read_text()
    assert out.count("**Delivery: ") == 1
    assert "[`METHOD.md`](../METHOD.md)" in out

    (state.paths.run_dir / "METHOD.md").write_text(UNAVAILABLE)
    finalize_delivery_banner(state, {
        "label": "explanation_only",
        "reasons": [{"source": "halt", "id": "stage_1", "message": "gap"}],
        "disclosures": [],
        "probe_counts": {},
    })
    out = (state.paths.run_dir / run_layout.PACKAGE_README).read_text()
    assert "contains a decomposition source index" in out
    assert "not a substantive method explanation" in out

    (state.paths.run_dir / "METHOD.md").write_text(" \n")
    finalize_delivery_banner(state, {
        "label": "explanation_only",
        "reasons": [{"source": "halt", "id": "stage_1", "message": "gap"}],
        "disclosures": [],
        "probe_counts": {},
    })
    out = (state.paths.run_dir / run_layout.PACKAGE_README).read_text()
    assert "exists but is empty" in out
    assert "decomposition source index" not in out


def test_uncertified_new_territory_banner(tmp_path):
    # Wording guardrail (design note §5): the full phrase, the two-sided
    # truth (nothing failed AND nobody verified the mechanism), never a
    # bare "uncertified" that reads as "probably fine".
    state = make_state(tmp_path / "run")
    run_layout.run_path(state.paths.run_dir,
                        run_layout.PACKAGE_README).write_text("# Package\n")
    finalize_delivery_banner(state, {
        "label": "uncertified_new_territory",
        "reasons": [], "disclosures": [], "probe_counts": {"pass": 5},
        "missing_probe_family": "trajectory_forecasting",
    })
    out = (state.paths.run_dir / run_layout.PACKAGE_README).read_text()
    assert out.startswith("> **Delivery: uncertified — new territory**")
    assert "every universal check passed" in out.lower()
    assert "don't exist yet" in out
    assert "nobody has verified" in out
    assert "[`REPORT.md`](../REPORT.md)" in out


def test_known_issues_banner_replaces_legacy_copy(tmp_path):
    from run_pipeline import finalize_run_report

    state = make_state(tmp_path / "run")
    readme = run_layout.run_path(state.paths.run_dir, run_layout.PACKAGE_README)
    readme.write_text(
        "> **This package has 1 known issue.** See [`KNOWN_ISSUES.md`](KNOWN_ISSUES.md).\n\n"
        "# Package\n",
        encoding="utf-8",
    )
    run_layout.run_path(state.paths.run_dir, run_layout.KNOWN_ISSUES_MD).write_text(
        "# Known issues\n\n## stage_3c - smoke failed\n\nbody\n",
        encoding="utf-8",
    )

    finalize_run_report(state)
    out = readme.read_text(encoding="utf-8")

    assert out.startswith("> **Known issues:**")
    assert "[`REPORT.md`](../REPORT.md)" in out
    assert "See [`KNOWN_ISSUES.md`](KNOWN_ISSUES.md)." not in out
    assert "[`KNOWN_ISSUES.md`](../details/KNOWN_ISSUES.md)" in out


def test_clean_stage_record_prunes_stale_known_issue(tmp_path):
    state = make_state(tmp_path / "run")

    run_pipeline.degrade(
        state.paths,
        "stage_1",
        reason="old reviewer contract issue",
        what_failed="the analyzer artifacts failed semantic review",
        what_to_do="review Stage 1",
    )
    run_pipeline.degrade(
        state.paths,
        "stage_2c",
        reason="method validator issue",
        what_failed="method.py failed validation",
        what_to_do="review method.py",
    )

    state.record(run_pipeline.StageResult(status="completed", stage_id="stage_1"))

    issues = (state.paths.run_dir / run_layout.KNOWN_ISSUES_MD).read_text(encoding="utf-8")
    assert "## stage_1" not in issues
    assert "## stage_2c" in issues


def test_clean_stage_record_removes_empty_known_issues_file(tmp_path):
    state = make_state(tmp_path / "run")
    run_pipeline.degrade(
        state.paths,
        "stage_1",
        reason="old reviewer contract issue",
        what_failed="the analyzer artifacts failed semantic review",
        what_to_do="review Stage 1",
    )

    state.record(run_pipeline.StageResult(status="completed", stage_id="stage_1"))

    assert not (state.paths.run_dir / run_layout.KNOWN_ISSUES_MD).exists()


def test_known_issue_writer_deduplicates_same_failure_signature(tmp_path):
    state = make_state(tmp_path / "run")
    for _ in range(2):
        run_pipeline.degrade(
            state.paths,
            "stage_2d",
            reason="validate_arch_contract_runtime.py: load_data returned 4 values",
            what_failed="arch contract runtime validation failed",
            where=".pipeline/arch_contract.json",
            what_to_do="reconcile arch_contract.json with method/data.py",
        )

    issues = (state.paths.run_dir / run_layout.KNOWN_ISSUES_MD).read_text(encoding="utf-8")
    assert issues.count("## stage_2d") == 1


def test_known_issue_writer_keeps_distinct_signatures_same_stage(tmp_path):
    state = make_state(tmp_path / "run")
    run_pipeline.degrade(
        state.paths,
        "stage_2d",
        reason="validate_arch_contract.py: missing x_test",
        what_failed="arch contract validation failed",
        where=".pipeline/arch_contract.json",
        what_to_do="add the missing loader return",
    )
    run_pipeline.degrade(
        state.paths,
        "stage_2d",
        reason="validate_arch_contract_runtime.py: load_data returned 4 values",
        what_failed="arch contract runtime validation failed",
        where=".pipeline/arch_contract.json",
        what_to_do="reconcile arch_contract.json with method/data.py",
    )

    issues = (state.paths.run_dir / run_layout.KNOWN_ISSUES_MD).read_text(encoding="utf-8")
    assert issues.count("## stage_2d") == 2


def test_known_issue_writer_deduplicates_reworded_judge_rationale(tmp_path):
    # The fedavg/ROMAN 2026-07-17 shape: a resume re-hits the same wall and
    # the judge rewords its free-form rationale — same class, same
    # confidence, same failing validator. One entry, not two.
    state = make_state(tmp_path / "run")
    for rationale in (
        "oscillation: iteration 0 already dispatched a fix for the dict "
        "return and it did not land",
        "prior decisions show the identical dict-return failure; "
        "re-dispatching the same agent has already failed once",
    ):
        run_pipeline.degrade(
            state.paths,
            "stage_2d",
            reason=("halt-judge decided to halt (unclear/medium) at "
                    f"validate_arch_contract_runtime.py: {rationale}"),
            what_failed="architecture-contract validation could not be satisfied",
            where=".pipeline/arch_contract.json",
            what_to_do="inspect the architecture-coder prompt",
        )

    issues = (state.paths.run_dir / run_layout.KNOWN_ISSUES_MD).read_text(encoding="utf-8")
    assert issues.count("## stage_2d") == 1


def test_known_issue_writer_keeps_distinct_judge_verdicts_same_stage(tmp_path):
    # Adjacent-good: a different judge classification or a different failing
    # validator is a different issue and must land as its own entry.
    state = make_state(tmp_path / "run")
    run_pipeline.degrade(
        state.paths,
        "stage_2d",
        reason=("halt-judge decided to halt (pipeline_bug/high) at "
                "validate_arch_contract.py: the schema lacks ndarray"),
        what_failed="architecture-contract validation could not be satisfied",
        where=".pipeline/arch_contract.json",
        what_to_do="fix the schema",
    )
    run_pipeline.degrade(
        state.paths,
        "stage_2d",
        reason=("halt-judge decided to halt (pipeline_bug/high) at "
                "validate_arch_contract_runtime.py: the dry-run synthesizes "
                "torch tensors for a numpy paradigm"),
        what_failed="architecture-contract validation could not be satisfied",
        where=".pipeline/arch_contract.json",
        what_to_do="fix the dry-run synthesizer",
    )

    issues = (state.paths.run_dir / run_layout.KNOWN_ISSUES_MD).read_text(encoding="utf-8")
    assert issues.count("## stage_2d") == 2


# ---------------------------------------------------------------------------
# Manifest integration
# ---------------------------------------------------------------------------


def test_manifest_carries_delivery_section(tmp_path, monkeypatch):
    from final_manifest import build_final_manifest
    state = make_state(tmp_path / "run")
    manifest = build_final_manifest(
        state.paths, stage_results=[],
        delivery=_draft_delivery(1))
    dumped = manifest.model_dump()
    assert dumped["delivery"]["label"] == "draft"
    assert dumped["delivery"]["reasons"][0]["id"] == "P0"
    # Pre-1.1.0 manifests (no delivery) must still validate.
    legacy = build_final_manifest(state.paths, stage_results=[])
    assert legacy.model_dump()["delivery"] is None


def test_manifest_accepts_explanation_only_delivery(tmp_path):
    from final_manifest import build_final_manifest
    state = make_state(tmp_path / "run")
    manifest = build_final_manifest(
        state.paths,
        stage_results=[],
        delivery={
            "schema_version": "1.0.0",
            "label": "explanation_only",
            "reasons": [{"source": "halt", "id": "stage_1", "message": "gap"}],
            "disclosures": [],
            "probe_counts": {},
        },
    )

    assert manifest.delivery is not None
    assert manifest.delivery.label == "explanation_only"


def test_halt_writes_explanation_only_package(tmp_path, monkeypatch):
    state = make_state(tmp_path / "run")
    monkeypatch.setattr(run_pipeline, "_post_halt_to_session",
                        lambda *args, **kwargs: None)

    result = run_pipeline.halt(
        state.paths,
        "stage_1",
        reason="no supported paradigm matched this paper",
        user_message="This paper needs a new field guide.",
        state=state,
    )

    assert result.status == "halted"
    assert (state.paths.run_dir / "REPORT.md").is_file()
    assert (state.paths.run_dir / run_layout.KNOWN_ISSUES_MD).is_file()
    assert (state.paths.run_dir / run_layout.PACKAGE_README).is_file()
    manifest = json.loads(
        (state.paths.run_dir / run_layout.FINAL_MANIFEST_JSON).read_text(encoding="utf-8")
    )
    assert manifest["delivery"]["label"] == "explanation_only"
    # The required-artifact set keys off the delivery label (2026-07-02
    # batch: honest explanation-only packages were failing validation over
    # code artifacts a halted run cannot have). With the explanation tier
    # present, the halt degrades the manifest instead of blocking it.
    assert manifest["run_status"] == "degraded"
    assert any(s["stage_id"] == "stage_1" and s["status"] == "halted"
               for s in manifest["stage_summary"])
    report = (state.paths.run_dir / "REPORT.md").read_text(encoding="utf-8")
    assert "Delivery label: **explanation only" in report
    assert "method understanding" in report
    assert "stage_1" in report
    assert "halted" in report
    assert (
        "No method understanding, code generation, notebook execution, or review failures were recorded."
        not in report
    )
    issues = (state.paths.run_dir / run_layout.KNOWN_ISSUES_MD).read_text(encoding="utf-8")
    assert "could not produce a reliable code package" in issues
    assert "stopped before R2C could deliver a reliable code package" in issues
    assert "produced end-to-end" not in issues
    assert issues.count("## stage_1") == 1
    readme = (state.paths.run_dir / run_layout.PACKAGE_README).read_text(encoding="utf-8")
    assert "method explanation was not produced before the halt" in readme
    assert "METHOD.md" not in readme

    run_pipeline.halt(
        state.paths,
        "stage_1",
        reason="no supported paradigm matched this paper",
        user_message="This paper needs a new field guide.",
        state=state,
    )
    issues = (state.paths.run_dir / run_layout.KNOWN_ISSUES_MD).read_text(encoding="utf-8")
    assert issues.count("## stage_1") == 1


def test_explanation_only_halt_rewrites_preexisting_degrade_intro(
    tmp_path, monkeypatch,
):
    """A later halt must correct an end-to-end preamble already on disk."""
    state = make_state(tmp_path / "run")
    monkeypatch.setattr(
        run_pipeline, "_post_halt_to_session", lambda *args, **kwargs: None
    )
    run_pipeline.degrade(
        state.paths,
        "stage_2d",
        reason="runtime contract needs review",
        what_failed="runtime contract validation failed",
        what_to_do="review the runtime contract",
    )
    issues_path = run_layout.run_path(
        state.paths.run_dir, run_layout.KNOWN_ISSUES_MD
    )
    assert "This package was produced end-to-end" in issues_path.read_text()

    recorded_action = "Reject this document from the current method pipeline."
    run_pipeline.halt(
        state.paths,
        "stage_1",
        reason="document is outside the current method scope",
        user_message="The contribution could not be routed.",
        halt_class="paradigm_mismatch",
        context={"paradigm_gap_report": {
            "decision": "unsupported_or_unclear",
            "recommended_next_action": recorded_action,
        }},
        state=state,
    )

    issues = issues_path.read_text(encoding="utf-8")
    assert "This package was produced end-to-end" not in issues
    assert "stopped before R2C could deliver a reliable code package" in issues
    assert recorded_action in issues
    assert "Do not resume the unchanged run" in issues
    assert "## stage_2d" in issues
    assert "## stage_1" in issues

    state.record(run_pipeline.StageResult(status="completed", stage_id="stage_1"))
    run_pipeline.finalize_run_report(state)
    resumed_issues = issues_path.read_text(encoding="utf-8")
    assert "This package was produced end-to-end" in resumed_issues
    assert "stopped before R2C could deliver" not in resumed_issues
    assert "## stage_2d" in resumed_issues
    assert "## stage_1" not in resumed_issues


def test_terminal_halt_with_empty_gap_action_never_suggests_resume(
    tmp_path, monkeypatch,
):
    state = make_state(tmp_path / "run")
    monkeypatch.setattr(
        run_pipeline, "_post_halt_to_session", lambda *args, **kwargs: None
    )

    run_pipeline.halt(
        state.paths,
        "stage_1",
        reason="contribution is outside the current method scope",
        user_message="Resume the run.",
        halt_class="paradigm_mismatch",
        context={"paradigm_gap_report": {
            "decision": "unsupported_or_unclear",
            "recommended_next_action": "",
        }},
        state=state,
    )

    issues = run_layout.run_path(
        state.paths.run_dir, run_layout.KNOWN_ISSUES_MD
    ).read_text(encoding="utf-8")
    assert "Scope-decision guidance" in issues
    assert "gap report did not record a next action" in issues
    assert "recorded next action is" not in issues
    assert "Do not resume the unchanged run" in issues
    assert "until the halt is resolved and the pipeline is resumed" not in issues


def test_uncertified_banner_qualifies_floor_when_a_universal_check_did_not_run(tmp_path):
    """The bev-distill 2026-07-04 shape: a smoke cap-degrade shipped the
    notebook unexecuted, the executed-notebook check read unprobeable, and
    the old fixed banner sentence overclaimed. The banner now derives the
    floor clause from the delivery's own disclosures."""
    state = make_state(tmp_path / "run")
    run_layout.run_path(state.paths.run_dir,
                        run_layout.PACKAGE_README).write_text("# Package\n")
    finalize_delivery_banner(state, {
        "label": "uncertified_new_territory",
        "reasons": [],
        "disclosures": [
            {"id": "UB-6", "source": "probe", "verdict": "unprobeable",
             "message": "notebook has no executed outputs"},
        ],
        "probe_counts": {"pass": 4, "unprobeable": 1},
        "missing_probe_family": "knowledge_distillation.bev",
    })
    out = (state.paths.run_dir / run_layout.PACKAGE_README).read_text()
    assert out.startswith("> **Delivery: uncertified — new territory**")
    assert "every universal check passed" not in out.lower()
    assert "could run passed" in out
    assert "1 universal check could not run" in out
    assert "nobody has verified" in out


# ---------------------------------------------------------------------------
# Block 8 — dead_attempt_finding_leaks_into_delivery (detr 2026-07-15): a
# successful resumed delivery kept the previous invocation's terminal-halt
# entry in KNOWN_ISSUES.md ("treat any code artifacts as absent or
# unreliable") next to the current, different degrade for the same stage.
# PipelineState.record now retires exactly that halt-shaped entry when its
# stage re-runs to a degraded result. Everything else stays conservative:
# degrade entries are pruned only by the existing completed/skipped rule,
# other stages' entries are untouched, and legacy heading variants survive.
# ---------------------------------------------------------------------------


def _halt_known_issue(state, stage_id):
    run_pipeline._append_known_issue(
        state.paths,
        stage_id=stage_id,
        what_failed=run_pipeline._EXPLANATION_ONLY_WHAT_FAILED,
        where=f".pipeline/{stage_id}.halt",
        what_to_do=(
            "Treat any code artifacts as absent or unreliable until the "
            "halt is resolved and the pipeline is resumed."),
        reason="smoke gate failed but could not parse failing cell index",
    )


def _degrade_known_issue(state, stage_id, what_failed, reason="detail"):
    run_pipeline._append_known_issue(
        state.paths, stage_id=stage_id, what_failed=what_failed,
        what_to_do="Review the delivered artifact.", reason=reason,
    )


def _known_issues_text(state) -> str:
    path = state.paths.run_dir / run_layout.KNOWN_ISSUES_MD
    return path.read_text(encoding="utf-8") if path.is_file() else ""


def test_degraded_rerun_retires_dead_halt_entry_keeps_current_issue(run_dir):
    """The detr known-bad, exact shape: stage_3c halted in the prior
    invocation, re-ran here, and degraded for a different reason. The dead
    halt entry goes; the current degrade entry stays; the append-only event
    log is untouched."""
    from run_pipeline import StageResult

    state = make_state(run_dir)
    events_path = state.paths.pipeline_dir / "run_events.jsonl"
    events_before = '{"event_type": "run_started"}\n'
    events_path.write_text(events_before)

    _halt_known_issue(state, "stage_3c")
    _degrade_known_issue(state, "stage_3c", "smoke gate timed out",
                         reason="training cell exceeded the per-cell budget")
    state.record(StageResult(status="degraded", stage_id="stage_3c"))

    text = _known_issues_text(state)
    assert run_pipeline._EXPLANATION_ONLY_WHAT_FAILED not in text
    assert "smoke gate timed out" in text
    assert events_path.read_text() == events_before


def test_degraded_result_leaves_other_stages_halt_entry(run_dir):
    """Failed-then-still-failed support: an unrelated degraded stage must not
    erase the terminal-halt entry that belongs to another stage."""
    from run_pipeline import StageResult

    state = make_state(run_dir)
    _halt_known_issue(state, "stage_3c")
    state.record(StageResult(status="degraded", stage_id="stage_2c"))

    text = _known_issues_text(state)
    assert f"## stage_3c — {run_pipeline._EXPLANATION_ONLY_WHAT_FAILED}" in text


def test_degraded_rerun_keeps_still_applicable_prior_degrades(run_dir):
    """Still-applicable prior issues: earlier degrade entries for the same
    stage survive a new degraded result. Only the halt-shaped entry is
    invocation-scoped truth."""
    from run_pipeline import StageResult

    state = make_state(run_dir)
    _degrade_known_issue(state, "stage_2b",
                         "architecture contract schema rejected a component")
    _degrade_known_issue(state, "stage_2b", "seed declared but never used")
    state.record(StageResult(status="degraded", stage_id="stage_2b"))

    text = _known_issues_text(state)
    assert "architecture contract schema rejected a component" in text
    assert "seed declared but never used" in text


def test_completed_rerun_still_removes_all_stage_entries(run_dir):
    """Failed-then-successful, existing behavior pinned: a stage that later
    completes cleanly removes its halt AND degrade entries (and the file
    itself when nothing remains)."""
    from run_pipeline import StageResult

    state = make_state(run_dir)
    _halt_known_issue(state, "stage_3c")
    _degrade_known_issue(state, "stage_3c", "smoke gate timed out")
    state.record(StageResult(status="completed", stage_id="stage_3c"))

    assert _known_issues_text(state) == ""


def test_single_invocation_degrade_leaves_file_byte_identical(run_dir):
    """Single-invocation adjacent good: with no prior halt entry, recording a
    degraded result changes nothing in KNOWN_ISSUES.md."""
    from run_pipeline import StageResult

    state = make_state(run_dir)
    _degrade_known_issue(state, "stage_3c", "smoke gate timed out")
    before = _known_issues_text(state)
    state.record(StageResult(status="degraded", stage_id="stage_3c"))

    assert _known_issues_text(state) == before


def test_legacy_halt_heading_variant_is_left_alone(run_dir):
    """Legacy conservatism: an entry whose heading does not exactly match the
    current terminal-halt heading is never removed by the degraded-result
    reconciliation — without a proven match, do nothing."""
    from run_pipeline import StageResult

    state = make_state(run_dir)
    _degrade_known_issue(state, "stage_3c",
                         "R2C stopped before delivering a package")
    state.record(StageResult(status="degraded", stage_id="stage_3c"))

    text = _known_issues_text(state)
    assert "R2C stopped before delivering a package" in text
