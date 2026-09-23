"""The three consumers of the probe-to-contract join (R2C-047 steps 3-5).

The join primitive itself is covered by tests/test_probe_spec_join.py. This
file covers what the three consumers DO with it:

- the claims table's label per branch, including the contradiction that
  existed before any branch was known (a "needs a fix on our side" verdict
  cell sitting over a note that refuses to say whose fault it is);
- the stage 2.c tier upgrade, where a flag on a must-replicate mechanism
  becomes a hard error the running fix loop can act on;
- the unprobeable core-novelty label cap, where a run with no evidence about
  the paper's central mechanism can no longer present at full label.
"""

from __future__ import annotations

import pytest

import render_claims_report as rcr
from delivery_label import (
    LABEL_DRAFT,
    LABEL_UNCERTIFIED,
    LABEL_VERIFIED,
    UNPROBEABLE_CORE_CAP_ID,
    apply_unprobeable_core_novelty_cap,
    derive_delivery_label,
    unprobeable_core_novelty,
)
from probe_spec_join import (
    CONTRACT_CONSISTENT,
    CONTRACT_CONTRADICTS,
    CONTRACT_SILENT,
)
from probes.claims import (
    CONTRACT_REQUIRES_EXACTLY_CAUSE,
    SUSPECT_OUR_IMPLEMENTATION,
    UNATTRIBUTED_CAUSE,
    UNTESTED_AT_THIS_SCALE,
    build_ledger,
)
from render_claims_report import render_claims_report

PARADIGM = "active_learning/bayesian"

# The shaping case: UB-7 flags a dead scoring term. Inertness is the only
# signal, so the cause picker lands on the wording that will not attribute the
# problem to either side.
_DEAD_TERM_ROW = "behavioral:scoring-terms-live"


def _element(element_id: str, *, role="core_methodology",
             status="must_replicate", approximations=()) -> dict:
    return {
        "element_id": element_id,
        "role": role,
        "replication_status": status,
        "technical_concept": "geometric representativeness ranking",
        "required_behavior": "rank candidates by the ellipsoid geodesic rescale",
        "paper_section": "Section 4.3, Eq. (12)",
        "acceptable_approximations": list(approximations),
        "forbidden_substitutions": [],
    }


def _spec(*elements: dict) -> dict:
    return {"methodology_replication_contract": {"elements": list(elements)}}


def _verdict(probe_id: str, verdict: str, *, branch="", element_ids=()) -> dict:
    return {"probe_id": probe_id, "verdict": verdict,
            "message": f"{probe_id} message", "spec_branch": branch,
            "element_ids": list(element_ids)}


def _row(ledger: dict, claim_id: str) -> dict:
    for row in ledger["claims"]:
        if row.get("claim_id") == claim_id:
            return row
    raise AssertionError(f"no {claim_id} row in {[r.get('claim_id') for r in ledger['claims']]}")


# ---------------------------------------------------------------------------
# Consumer 1: the claims-table label per branch
# ---------------------------------------------------------------------------


def test_unattributed_suspect_row_does_not_render_as_ours_to_fix():
    # The shape the record was opened on: the note says the mechanism may be
    # inert as the paper published it, and the verdict cell said we need to fix
    # it. Both cannot be true in one row.
    ledger = build_ledger({"elements": []},
                          [_verdict("UB-7", "flag_for_researcher")], PARADIGM)
    row = _row(ledger, _DEAD_TERM_ROW)
    assert row["status"] == SUSPECT_OUR_IMPLEMENTATION
    assert row["suspected_cause"] == UNATTRIBUTED_CAUSE
    report = render_claims_report(ledger)
    assert "Unresolved (needs your judgment)" in report
    assert "1 unresolved (needs your judgment)" in report
    # And the fix-on-our-side count does not also claim it.
    assert "0 need a fix on our side" in report


def test_attributed_suspect_row_still_renders_as_ours_to_fix():
    # The negative control: a scale-mismatch fail attributes cleanly, so the
    # existing label is correct and must not move.
    ledger = build_ledger({"elements": []},
                          [_verdict("US-4", "fail"),
                           _verdict("CT-1", "fail")], PARADIGM)
    report = render_claims_report(ledger)
    assert "Needs a fix on our side" in report
    assert "Unresolved (needs your judgment)" not in report


def test_contract_contradicts_drops_the_hedge_and_keeps_the_fix_label():
    ledger = build_ledger(
        {"elements": []},
        [_verdict("UB-7", "flag_for_researcher", branch=CONTRACT_CONTRADICTS,
                  element_ids=["ranking"])],
        PARADIGM)
    row = _row(ledger, _DEAD_TERM_ROW)
    assert row["spec_branch"] == CONTRACT_CONTRADICTS
    assert row["status"] == SUSPECT_OUR_IMPLEMENTATION
    assert row["suspected_cause"] == CONTRACT_REQUIRES_EXACTLY_CAUSE
    assert "ours to fix" in row["reasoning"]
    assert "may be our wiring" not in row["reasoning"]
    report = render_claims_report(ledger)
    assert "Needs a fix on our side" in report
    assert "Unresolved (needs your judgment)" not in report


def test_contract_consistent_stops_calling_an_approved_approximation_a_defect():
    ledger = build_ledger(
        {"elements": []},
        [_verdict("UB-7", "flag_for_researcher", branch=CONTRACT_CONSISTENT,
                  element_ids=["ranking"])],
        PARADIGM)
    row = _row(ledger, _DEAD_TERM_ROW)
    assert row["status"] == UNTESTED_AT_THIS_SCALE
    assert row["suspected_cause"] == "contract_approved_approximation"
    assert "known limit" in row["reasoning"]
    report = render_claims_report(ledger)
    assert "Needs a fix on our side" not in report
    assert "Untested at this scale" in report


def test_contract_silent_row_is_unchanged_from_todays_behavior():
    silent = build_ledger(
        {"elements": []},
        [_verdict("UB-7", "flag_for_researcher", branch=CONTRACT_SILENT)],
        PARADIGM)
    unstamped = build_ledger({"elements": []},
                             [_verdict("UB-7", "flag_for_researcher")],
                             PARADIGM)
    a, b = _row(silent, _DEAD_TERM_ROW), _row(unstamped, _DEAD_TERM_ROW)
    assert a["status"] == b["status"] == SUSPECT_OUR_IMPLEMENTATION
    assert a["reasoning"] == b["reasoning"]
    assert a["spec_branch"] == b["spec_branch"] == CONTRACT_SILENT


def test_contradicts_wins_over_consistent_across_a_predicates_verdicts():
    # One predicate, two backing verdicts adjudicated differently. There is no
    # honest average, and the strict reading is the one that routes a real
    # defect to a fix.
    ledger = build_ledger(
        {"elements": []},
        [_verdict("UB-7", "flag_for_researcher", branch=CONTRACT_CONSISTENT),
         _verdict("UB-7", "flag_for_researcher", branch=CONTRACT_CONTRADICTS)],
        PARADIGM)
    row = _row(ledger, _DEAD_TERM_ROW)
    assert row["spec_branch"] == CONTRACT_CONTRADICTS
    assert row["status"] == SUSPECT_OUR_IMPLEMENTATION


def test_renderer_and_ledger_agree_on_the_unattributed_cause_key():
    # The renderer stays stdlib-only and cannot import probes.claims, so the
    # two copies of the key are pinned together here instead.
    assert rcr.UNATTRIBUTED_CAUSE == UNATTRIBUTED_CAUSE


def test_passing_rows_carry_the_branch_without_changing_status():
    ledger = build_ledger(
        {"elements": []},
        [_verdict("CT-1", "pass", branch=CONTRACT_CONTRADICTS)], PARADIGM)
    row = _row(ledger, "behavioral:differs-from-null")
    assert row["status"] == "verified_at_scale"
    assert row["spec_branch"] == CONTRACT_CONTRADICTS


# ---------------------------------------------------------------------------
# Consumer 2: the stage 2.c tier upgrade
# ---------------------------------------------------------------------------


class _FakeVerdict:
    def __init__(self, verdict: str, element_ids=()):
        self.probe_id = "AL-5"
        self.verdict = verdict
        self.message = "the selector ignores the model entirely"
        self.element_ids = list(element_ids)
        self.spec_branch = ""
        self.bound_callables = []
        self.probe_ref = ""


@pytest.fixture()
def early_arm(monkeypatch, tmp_path):
    """Drive the early AL-5 arm without a loadable generated package: the
    branch under test is the tier decision, not the probe itself."""
    import validate_method_coder_output as vmco

    (tmp_path / "method").mkdir()
    (tmp_path / "method" / "method.py").write_text("", encoding="utf-8")

    def _run(verdict: str, spec, element_ids=("ranking",)):
        import probes.al_loop as al_loop
        import probes.package_loader as loader
        monkeypatch.setattr(loader, "load_module_from_path",
                            lambda *a, **k: object())
        monkeypatch.setattr(
            al_loop, "probe_acquisition_contract",
            lambda *a, **k: _FakeVerdict(verdict, element_ids))
        return vmco._active_learning_selector_behavioral_errors(
            run_dir=tmp_path, pluggable_name="select_batch", spec=spec)

    return _run


def test_flag_on_a_must_replicate_mechanism_becomes_a_hard_error(early_arm):
    errors, warnings = early_arm("flag_for_researcher", _spec(_element("ranking")))
    assert len(errors) == 1
    assert "CONTRADICTS" in errors[0]
    assert "geometric representativeness ranking" in errors[0]
    assert "rather than deferring it to the researcher" in errors[0]
    assert warnings == []


def test_flag_on_an_approximable_mechanism_keeps_its_warning(early_arm):
    errors, warnings = early_arm(
        "flag_for_researcher",
        _spec(_element("ranking", status="faithful_approximation_allowed",
                       approximations=["fewer MC samples at demo scale"])))
    assert errors == []
    assert len(warnings) == 1
    assert "CONSISTENT" in warnings[0]


def test_flag_with_a_silent_contract_keeps_todays_warning(early_arm):
    errors, warnings = early_arm("flag_for_researcher", _spec())
    assert errors == []
    assert len(warnings) == 1
    assert "flag_for_researcher" in warnings[0]
    assert "CONTRADICTS" not in warnings[0]


def test_no_spec_at_all_leaves_the_arm_exactly_as_it_was(early_arm):
    errors, warnings = early_arm("flag_for_researcher", None)
    assert errors == []
    assert len(warnings) == 1


def test_an_unprobeable_early_arm_is_never_upgraded(early_arm):
    # Missing evidence is not an observed defect, whatever the contract says,
    # so there is nothing for the fix loop to act on.
    errors, warnings = early_arm("unprobeable", _spec(_element("ranking")))
    assert errors == []
    assert len(warnings) == 1


def test_a_hard_fail_carries_the_contract_sentence_too(early_arm):
    errors, _ = early_arm("fail", _spec(_element("ranking")))
    assert len(errors) == 1
    assert "must return exactly the requested" in errors[0]
    assert "CONTRADICTS" in errors[0]


# ---------------------------------------------------------------------------
# Consumer 3: the unprobeable core-novelty label cap
# ---------------------------------------------------------------------------


def _clean_report(*extra: dict) -> dict:
    # CT-1 passing is the contribution evidence that would otherwise verify.
    return {"verdicts": [_verdict("CT-1", "pass"), *extra]}


def test_unprobed_core_novelty_caps_a_would_be_verified_delivery():
    report = _clean_report(_verdict("AL-5", "unprobeable", element_ids=["ranking"]))
    out = derive_delivery_label(report, None, paradigm=PARADIGM,
                                spec=_spec(_element("ranking")))
    assert out["label"] == LABEL_UNCERTIFIED
    caps = [d for d in out["disclosures"] if d["id"] == UNPROBEABLE_CORE_CAP_ID]
    assert len(caps) == 1
    assert "geometric representativeness ranking" in caps[0]["message"]
    assert out["missing_probe_family"] == PARADIGM


def test_unprobed_peripheral_mechanism_stays_label_neutral():
    report = _clean_report(_verdict("AL-5", "unprobeable", element_ids=["ranking"]))
    out = derive_delivery_label(
        report, None, paradigm=PARADIGM,
        spec=_spec(_element("ranking", role="supporting_component")))
    assert out["label"] == LABEL_VERIFIED
    assert not [d for d in out["disclosures"]
                if d["id"] == UNPROBEABLE_CORE_CAP_ID]


def test_unprobed_approximable_core_stays_label_neutral_here():
    # An approximated core is already the approximated_core rule's business.
    # This cap is about MISSING EVIDENCE on a mechanism required exactly, so it
    # must not fire a second time on the same fact.
    report = _clean_report(_verdict("AL-5", "unprobeable", element_ids=["ranking"]))
    out = derive_delivery_label(
        report, None, paradigm=PARADIGM,
        spec=_spec(_element("ranking",
                            status="faithful_approximation_allowed",
                            approximations=["fewer MC samples"])))
    assert not [d for d in out["disclosures"]
                if d["id"] == UNPROBEABLE_CORE_CAP_ID]


def test_a_probed_core_novelty_is_not_capped():
    report = _clean_report(_verdict("AL-5", "pass", element_ids=["ranking"]))
    out = derive_delivery_label(report, None, paradigm=PARADIGM,
                                spec=_spec(_element("ranking")))
    assert out["label"] == LABEL_VERIFIED


def test_no_spec_derives_exactly_as_before():
    report = _clean_report(_verdict("AL-5", "unprobeable", element_ids=["ranking"]))
    with_spec = derive_delivery_label(report, None, paradigm=PARADIGM, spec=None)
    assert with_spec["label"] == LABEL_VERIFIED
    assert unprobeable_core_novelty(report, None) == []


def test_verdicts_with_no_element_ids_cannot_cap():
    report = _clean_report(_verdict("AL-5", "unprobeable"))
    out = derive_delivery_label(report, None, paradigm=PARADIGM,
                                spec=_spec(_element("ranking")))
    assert out["label"] == LABEL_VERIFIED


def test_the_cap_is_idempotent_and_never_raises_a_capped_label():
    # The driver re-applies the cap after its post-hoc draft demotions. A
    # second application must neither duplicate the disclosure nor lift the
    # label from uncertified (rank 2) back to draft (rank 3).
    elements = unprobeable_core_novelty(
        _clean_report(_verdict("AL-5", "unprobeable", element_ids=["ranking"])),
        _spec(_element("ranking")))
    delivery = {"label": LABEL_UNCERTIFIED, "reasons": [], "disclosures": []}
    apply_unprobeable_core_novelty_cap(delivery, elements)
    delivery["label"] = LABEL_DRAFT  # what a post-hoc battery error would set
    apply_unprobeable_core_novelty_cap(delivery, elements)
    assert delivery["label"] == LABEL_UNCERTIFIED
    assert len([d for d in delivery["disclosures"]
                if d["id"] == UNPROBEABLE_CORE_CAP_ID]) == 1


def test_the_cap_keeps_every_demoting_reason_visible():
    report = _clean_report(_verdict("AL-5", "unprobeable", element_ids=["ranking"]),
                           _verdict("UB-6", "fail"))
    out = derive_delivery_label(report, None, paradigm=PARADIGM,
                                spec=_spec(_element("ranking")))
    assert out["label"] == LABEL_UNCERTIFIED
    assert [r["id"] for r in out["reasons"]], "demoting reasons must stay recorded"


def test_two_unprobed_core_mechanisms_are_named_once_each():
    report = _clean_report(
        _verdict("AL-5", "unprobeable", element_ids=["ranking", "scoring"]),
        _verdict("UB-7", "unprobeable", element_ids=["ranking"]))
    elements = unprobeable_core_novelty(
        report, _spec(_element("ranking"), _element("scoring")))
    assert [e.element_id for e in elements] == ["ranking", "scoring"]


# ---------------------------------------------------------------------------
# The binding, end to end (R2C-072). A probe declares the callables it
# exercised; the battery resolves those through the generated code's own
# annotations into paper-map ids; the crosswalk carries them to the contract.
# ---------------------------------------------------------------------------


_ANNOTATED_SELECTOR = '''\
def select_batch(x_unlabeled, batch_size, model=None, seed=0):
    """Pick a batch.

    # paper-element: alg-acquisition
    """
    return list(range(batch_size))
'''


def _bound_run(
    tmp_path,
    *,
    source=_ANNOTATED_SELECTOR,
    crosswalk=("alg-acquisition",),
    verification_probe_refs=None,
):
    run = tmp_path / "run"
    (run / "method").mkdir(parents=True)
    (run / ".pipeline").mkdir(parents=True)
    (run / "method" / "method.py").write_text(source, encoding="utf-8")
    element = dict(_element("ranking"), paper_element_ids=list(crosswalk))
    if verification_probe_refs is not None:
        element["verification_probe_refs"] = list(verification_probe_refs)
    spec = _spec(element)
    (run / ".pipeline" / "method_spec.json").write_text(
        __import__("json").dumps(spec), encoding="utf-8")
    return run, spec


def test_a_declared_callable_reaches_the_contract_through_the_crosswalk(tmp_path):
    from probes import ProbeReport, ProbeVerdict
    from run_probes import _apply_spec_conditioning

    run, _ = _bound_run(tmp_path)
    report = ProbeReport(target=str(run))
    verdict = ProbeVerdict("AL-5", "flag_for_researcher",
                           "the selection ignores the model")
    verdict.bound_callables = ["select_batch"]
    # A current battery stamps this even when replaying an archived spec whose
    # contract predates verification_probe_refs. Adjudication remains legacy;
    # certification remains explicitly non-reference-qualified.
    verdict.probe_ref = "al_loop.acquisition_contract"
    report.add(verdict)

    _apply_spec_conditioning(run, report)

    assert verdict.element_ids == ["alg-acquisition"]
    assert verdict.spec_branch == CONTRACT_CONTRADICTS
    assert "CONTRADICTS" in verdict.message


def test_declaring_nothing_leaves_the_verdict_unbound(tmp_path):
    from probes import ProbeReport, ProbeVerdict
    from run_probes import _apply_spec_conditioning

    run, _ = _bound_run(tmp_path)
    report = ProbeReport(target=str(run))
    verdict = ProbeVerdict("AL-5", "flag_for_researcher", "observation")
    report.add(verdict)

    _apply_spec_conditioning(run, report)

    assert verdict.element_ids == []
    assert verdict.spec_branch == CONTRACT_SILENT


def test_an_unannotated_callable_stays_unbound(tmp_path):
    # No recorded link between that code and the paper, so no adjudication.
    from probes import ProbeReport, ProbeVerdict
    from run_probes import _apply_spec_conditioning

    run, _ = _bound_run(
        tmp_path, source="def select_batch(x, batch_size):\n    return []\n")
    report = ProbeReport(target=str(run))
    verdict = ProbeVerdict("AL-5", "fail", "observation")
    verdict.bound_callables = ["select_batch"]
    report.add(verdict)

    _apply_spec_conditioning(run, report)

    assert verdict.element_ids == []
    assert verdict.spec_branch == CONTRACT_SILENT


def test_ids_a_probe_set_itself_are_never_overruled(tmp_path):
    from probes import ProbeReport, ProbeVerdict
    from run_probes import _apply_spec_conditioning

    run, _ = _bound_run(tmp_path)
    report = ProbeReport(target=str(run))
    verdict = ProbeVerdict("AL-5", "fail", "observation",
                           element_ids=["ranking"])
    verdict.bound_callables = ["select_batch"]
    report.add(verdict)

    _apply_spec_conditioning(run, report)

    assert verdict.element_ids == ["ranking", "alg-acquisition"]


def test_the_acquisition_probe_declares_the_selector_it_exercised():
    # Stamped in a wrapper so a future arm cannot ship unbound by forgetting.
    import probes.al_loop as al_loop

    verdict = al_loop.probe_acquisition_contract(object(), "select_batch")
    assert verdict.bound_callables == ["select_batch"]
    assert verdict.probe_ref == "al_loop.acquisition_contract"


def test_the_early_arm_binds_the_same_way_the_battery_does(tmp_path, monkeypatch):
    # Both enforcement points must reach the same adjudication, or a finding
    # routed at stage 2.c would be judged differently from the identical
    # finding at delivery.
    import probes.al_loop as al_loop
    import probes.package_loader as loader
    import validate_method_coder_output as vmco

    run, spec = _bound_run(
        tmp_path,
        verification_probe_refs=["al_loop.acquisition_contract"],
    )
    monkeypatch.setattr(loader, "load_module_from_path", lambda *a, **k: object())

    def _fake(*a, **k):
        v = _FakeVerdict("flag_for_researcher", ())
        v.bound_callables = ["select_batch"]
        v.probe_ref = "al_loop.acquisition_contract"
        return v

    monkeypatch.setattr(al_loop, "probe_acquisition_contract", _fake)
    errors, warnings = vmco._active_learning_selector_behavioral_errors(
        run_dir=run, pluggable_name="select_batch", spec=spec)

    assert len(errors) == 1
    assert "CONTRADICTS" in errors[0]
    assert warnings == []

    from probes import ProbeReport
    from run_probes import _apply_spec_conditioning

    verdict = _fake()
    report = ProbeReport(target=str(run))
    report.add(verdict)
    _apply_spec_conditioning(run, report)

    assert verdict.element_ids == ["alg-acquisition"]
    assert verdict.spec_branch == CONTRACT_CONTRADICTS
    assert "CONTRADICTS" in verdict.message


def test_early_and_full_seams_both_reject_a_mismatched_exact_ref(
    tmp_path, monkeypatch,
):
    import probes.al_loop as al_loop
    import probes.package_loader as loader
    import validate_method_coder_output as vmco
    from probes import ProbeReport
    from run_probes import _apply_spec_conditioning

    run, spec = _bound_run(
        tmp_path,
        verification_probe_refs=["al_loop.some_other_contract"],
    )
    monkeypatch.setattr(loader, "load_module_from_path", lambda *a, **k: object())

    def _fake(*a, **k):
        verdict = _FakeVerdict("flag_for_researcher", ())
        verdict.bound_callables = ["select_batch"]
        verdict.probe_ref = "al_loop.acquisition_contract"
        return verdict

    monkeypatch.setattr(al_loop, "probe_acquisition_contract", _fake)
    errors, warnings = vmco._active_learning_selector_behavioral_errors(
        run_dir=run, pluggable_name="select_batch", spec=spec,
    )

    assert errors == []
    assert len(warnings) == 1
    assert "declares no such verification ref" in warnings[0]

    verdict = _fake()
    report = ProbeReport(target=str(run))
    report.add(verdict)
    _apply_spec_conditioning(run, report)

    assert verdict.element_ids == ["alg-acquisition"]
    assert verdict.spec_branch == CONTRACT_SILENT
    assert "declares no such verification ref" in verdict.message
