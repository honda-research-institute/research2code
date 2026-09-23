"""R2C-047 step 2 — the shared join primitive.

The failure class: probe findings and the methodology contract never join, so a
producer-fixable finding and a paper-intrinsic finding get identical treatment
in routing, wording, and labeling.

The shaping case is the delivered active-learning selector. Its
model-insensitivity arm says "legitimate only if the paper's method is purely
geometric (verify against the paper)" — on a run whose own spec declares the
scoring must-replicate with a forbidden-substitution list, which answers that
question. The neutral wording is right for a genuinely geometric selector and
wrong here, and only the contract can tell the two apart.

Design confirmed by the maintainer 2026-08-05: a deterministic set intersection on
element identifiers with NO fuzzy fallback, three disclosure branches, and the
unprobeable-core-novelty effect as a CAP rather than a deduction.
"""

from __future__ import annotations

from scripts.probe_spec_join import (
    CONTRACT_CONSISTENT, CONTRACT_CONTRADICTS, CONTRACT_SILENT,
    contract_elements, disclosure, join_verdict, unbound_gap_note,
)


def _element(element_id: str, *, role: str, status: str,
             approximations: list[str] | None = None, **over) -> dict:
    base = {
        "element_id": element_id,
        "role": role,
        "replication_status": status,
        "paper_section": "Section 3.2",
        "paper_evidence": "the paper states it",
        "technical_concept": "uncertainty-weighted scoring",
        "required_behavior": "the score must consume the model's predictive "
                             "uncertainty",
        "demo_scale_implementation": "MC-dropout over the pool",
        "acceptable_approximations": approximations or [],
        "forbidden_substitutions": ["a purely geometric distance score"],
        "required_controls": [],
        "fairness_checks": [],
        "feasibility_rationale": "replicable at demo scale",
        "verification_expectations": ["the selection responds to weights"],
        "blockers": [],
    }
    base.update(over)
    return base


def _spec(*elements: dict) -> dict:
    return {"methodology_replication_contract": {"elements": list(elements)}}


class _Verdict:
    """The ProbeVerdict surface the join reads, without the catalog
    registration a real one requires."""

    def __init__(
        self,
        verdict: str,
        element_ids: list[str],
        probe_id="AL-5",
        *,
        probe_ref: str = "",
        bound_callables: list[str] | None = None,
    ):
        self.probe_id = probe_id
        self.verdict = verdict
        self.element_ids = element_ids
        self.probe_ref = probe_ref
        self.bound_callables = bound_callables or []


# --- The intersection -------------------------------------------------------


def test_the_shaping_case_binds_and_reads_as_ours_to_fix():
    spec = _spec(_element("acquisition-scoring", role="core_methodology",
                          status="must_replicate"))
    join = join_verdict(_Verdict("flag_for_researcher", ["acquisition-scoring"]),
                        spec)

    assert join.bound
    assert join.branch == CONTRACT_CONTRADICTS
    assert join.producer_fixable
    assert [e.element_id for e in join.must_replicate_elements] \
        == ["acquisition-scoring"]
    assert join.core_novelty_elements  # core_methodology AND must_replicate


def test_a_geometric_selector_keeps_the_researcher_question():
    """The negative control the acceptance criteria name: a package whose
    contract does not demand model-sensitivity must keep the neutral wording,
    because there the probe genuinely cannot tell our miswiring from the
    paper's design."""
    spec = _spec(_element("acquisition-scoring", role="supporting_mechanism",
                          status="faithful_approximation_allowed",
                          approximations=["a geometric core-set score"]))
    join = join_verdict(_Verdict("flag_for_researcher", ["acquisition-scoring"]),
                        spec)

    assert join.branch == CONTRACT_CONSISTENT
    assert join.producer_fixable is False


def test_no_element_ids_means_no_adjudication():
    """A probe finding without ids is a probe-side gap, never a matcher to
    soften. It joins to nothing and every consumer falls back to neutral."""
    spec = _spec(_element("acquisition-scoring", role="core_methodology",
                          status="must_replicate"))
    join = join_verdict(_Verdict("fail", []), spec)

    assert join.bound is False
    assert join.branch == CONTRACT_SILENT
    assert join.producer_fixable is False
    assert unbound_gap_note(join) is None  # nothing was named, nothing to report


def test_an_unmatched_id_is_reported_not_matched():
    """An id the contract does not carry must NOT fall back to prose matching
    against a similarly-named element."""
    spec = _spec(_element("acquisition-scoring", role="core_methodology",
                          status="must_replicate"))
    join = join_verdict(
        _Verdict("fail", ["acquisition_scoring", "acquisition-scoring-v2"]),
        spec)

    assert join.bound is False
    assert join.branch == CONTRACT_SILENT
    assert join.contract_present is True  # the contract exists, the id is wrong
    note = unbound_gap_note(join)
    assert note is not None
    assert "`acquisition_scoring`" in note
    assert "naming gap to close" in note
    assert "not a judgment" in note


def test_a_run_without_a_contract_joins_to_nothing():
    """And says nothing about it: with no contract, ids that went nowhere are
    not a naming gap, they mean this paradigm's analyzer path emits no
    contract."""
    for spec in ({}, None, {"methodology_replication_contract": None}):
        join = join_verdict(_Verdict("fail", ["x"]), spec)
        assert join.branch == CONTRACT_SILENT
        assert join.unmatched_ids == ("x",)
        assert join.contract_present is False
        assert unbound_gap_note(join) is None


def test_a_dict_verdict_joins_the_same_as_an_object():
    """The battery holds objects; a delivered probe report holds dicts. Both
    must adjudicate identically, so a re-read of a delivered run reaches the
    same branch."""
    spec = _spec(_element("acquisition-scoring", role="core_methodology",
                          status="must_replicate"))
    as_object = join_verdict(_Verdict("fail", ["acquisition-scoring"]), spec)
    as_dict = join_verdict({"probe_id": "AL-5", "verdict": "fail",
                            "element_ids": ["acquisition-scoring"]}, spec)

    assert as_object == as_dict


def test_duplicate_ids_bind_once():
    spec = _spec(_element("acquisition-scoring", role="core_methodology",
                          status="must_replicate"))
    join = join_verdict(
        _Verdict("fail", ["acquisition-scoring", "acquisition-scoring"]), spec)

    assert len(join.matched) == 1


def test_any_must_replicate_match_decides_the_branch():
    """Mixed binding: one element approximable, one exact. The exact one wins,
    because a finding that contradicts any must-replicate element is a defect
    regardless of what else it touches."""
    spec = _spec(
        _element("batching", role="supporting_mechanism",
                 status="faithful_approximation_allowed",
                 approximations=["one batch per round"]),
        _element("scoring", role="core_methodology", status="must_replicate"),
    )
    join = join_verdict(_Verdict("fail", ["batching", "scoring"]), spec)

    assert join.branch == CONTRACT_CONTRADICTS
    assert [e.element_id for e in join.must_replicate_elements] == ["scoring"]


# --- What counts as fixable, and what caps the label ------------------------


def test_a_passing_verdict_is_never_producer_fixable():
    spec = _spec(_element("scoring", role="core_methodology",
                          status="must_replicate"))
    join = join_verdict(_Verdict("pass", ["scoring"]), spec)

    assert join.branch == CONTRACT_CONTRADICTS  # the contract still says exact
    assert join.producer_fixable is False       # but there is nothing to fix


def test_unprobeable_core_novelty_caps_the_label():
    """The bayesian shape: a run shipped at full label with its core novelty
    unprobed, because unprobeable never demoted."""
    spec = _spec(_element("scoring", role="core_methodology",
                          status="must_replicate"))
    join = join_verdict(_Verdict("unprobeable", ["scoring"]), spec)

    assert join.caps_the_label is True
    # Missing evidence is not an observed defect, so it routes no fix.
    assert join.producer_fixable is False


def test_unprobeable_peripheral_mechanism_stays_label_neutral():
    """The other half of the acceptance criterion: only core novelty moves the
    label."""
    for role, status in (("supporting_mechanism", "must_replicate"),
                         ("core_methodology", "faithful_approximation_allowed")):
        spec = _spec(_element("scoring", role=role, status=status,
                              approximations=(["approx"] if status
                                              == "faithful_approximation_allowed"
                                              else [])))
        join = join_verdict(_Verdict("unprobeable", ["scoring"]), spec)
        assert join.caps_the_label is False, (role, status)


def test_unprobeable_without_a_binding_never_caps():
    join = join_verdict(_Verdict("unprobeable", []), _spec())
    assert join.caps_the_label is False


# --- The three-way disclosure ----------------------------------------------


def test_the_contradicts_branch_says_it_is_ours_to_fix():
    spec = _spec(_element("scoring", role="core_methodology",
                          status="must_replicate"))
    join = join_verdict(_Verdict("flag_for_researcher", ["scoring"]), spec)

    text = disclosure(join, neutral="verify against the paper")

    assert "CONTRADICTS" in text
    assert "defect on our side to fix" in text
    # Names the mechanism in the paper's own words, from the contract.
    assert "uncertainty-weighted scoring (Section 3.2)" in text
    assert "must consume the model's predictive uncertainty" in text
    # And does not fall back to asking the researcher.
    assert "verify against the paper" not in text


def test_the_consistent_branch_says_no_fix_is_implied():
    spec = _spec(_element("scoring", role="supporting_mechanism",
                          status="faithful_approximation_allowed",
                          approximations=["a geometric core-set score"]))
    join = join_verdict(_Verdict("flag_for_researcher", ["scoring"]), spec)

    text = disclosure(join, neutral="verify against the paper")

    assert "CONSISTENT" in text
    assert "faithful_approximation_allowed" in text
    assert "No fix is implied" in text
    assert "verify against the paper" not in text


def test_the_silent_branch_preserves_todays_wording():
    """The recorded neutrality decision survives untouched where it belongs:
    the probe cannot tell our miswiring from the paper's design, so the
    researcher adjudicates."""
    join = join_verdict(_Verdict("flag_for_researcher", []), _spec())

    assert disclosure(join, neutral="verify against the paper") \
        == "verify against the paper"
    assert disclosure(join) == ""


def test_not_replicable_reads_as_a_known_limit_not_a_defect():
    spec = _spec(_element("scoring", role="core_methodology",
                          status="not_replicable",
                          blockers=["needs the paper's proprietary corpus"]))
    join = join_verdict(_Verdict("fail", ["scoring"]), spec)

    assert join.branch == CONTRACT_CONSISTENT
    assert join.producer_fixable is False
    assert "not_replicable" in disclosure(join)


def test_the_disclosure_names_every_contradicted_element():
    spec = _spec(
        _element("scoring", role="core_methodology", status="must_replicate",
                 technical_concept="uncertainty scoring"),
        _element("sampling", role="core_methodology", status="must_replicate",
                 technical_concept="ancestral sampling",
                 paper_section="Algorithm 1"),
    )
    join = join_verdict(_Verdict("fail", ["scoring", "sampling"]), spec)

    text = disclosure(join)
    assert "2 elements must-replicate" in text
    assert "uncertainty scoring" in text
    assert "ancestral sampling (Algorithm 1)" in text


# --- The reader -------------------------------------------------------------


def test_contract_elements_skips_malformed_entries():
    spec = {"methodology_replication_contract": {"elements": [
        _element("good", role="core_methodology", status="must_replicate"),
        {"element_id": ""},          # no id
        "not a dict",
        {"role": "core_methodology"},  # no id key at all
    ]}}

    elements = contract_elements(spec)

    assert list(elements) == ["good"]
    assert len(elements["good"]) == 1
    assert elements["good"][0].must_replicate
    assert elements["good"][0].forbidden_substitutions == (
        "a purely geometric distance score",)


def test_the_schema_invariant_the_branch_rule_rests_on():
    """The branch rule reads `must_replicate` as "no approved approximation"
    without checking the approximations list, because the spec schema enforces
    the bi-implication. If that ever loosens, this rule needs a second
    condition, so the invariant is asserted here rather than assumed."""
    import pytest  # noqa: PLC0415
    from pydantic import ValidationError  # noqa: PLC0415

    from schemas.method_spec import MethodologyContractElement  # noqa: PLC0415

    with pytest.raises(ValidationError):
        MethodologyContractElement(**_element(
            "scoring", role="supporting_mechanism", status="must_replicate",
            approximations=["a geometric score"]))


# --- The battery seam (step 3, wired for every family at once) --------------


def test_the_battery_stamps_every_verdict_with_its_branch(tmp_path):
    """The conditioning runs as one pass over the finished verdicts rather than
    a spec threaded through every probe signature, so a family that knows
    nothing about the contract still gets adjudicated."""
    import json  # noqa: PLC0415

    from probes import ProbeReport, ProbeVerdict  # noqa: PLC0415
    from run_probes import _apply_spec_conditioning  # noqa: PLC0415

    run = tmp_path / "run"
    (run / ".pipeline").mkdir(parents=True)
    (run / ".pipeline" / "method_spec.json").write_text(json.dumps(_spec(
        _element("scoring", role="core_methodology", status="must_replicate"),
        _element("batching", role="supporting_mechanism",
                 status="faithful_approximation_allowed",
                 approximations=["one batch per round"],
                 technical_concept="synchronized batching"),
    )), encoding="utf-8")

    report = ProbeReport(target=str(run))
    contradicted = ProbeVerdict("UB-6", "fail", "the selection ignores weights",
                                element_ids=["scoring"])
    excused = ProbeVerdict("UB-6", "flag_for_researcher",
                           "batching is vacuous", element_ids=["batching"])
    unbound = ProbeVerdict("UB-6", "fail", "something else", element_ids=[])
    passing = ProbeVerdict("UB-6", "pass", "all good", element_ids=["scoring"])
    for verdict in (contradicted, excused, unbound, passing):
        report.add(verdict)

    _apply_spec_conditioning(run, report)

    assert contradicted.spec_branch == CONTRACT_CONTRADICTS
    assert "defect on our side to fix" in contradicted.message
    assert excused.spec_branch == CONTRACT_CONSISTENT
    assert "No fix is implied" in excused.message
    # Silent stays silent in prose, and still records that it was adjudicated.
    assert unbound.spec_branch == CONTRACT_SILENT
    assert unbound.message == "something else"
    # A passing verdict has nothing for the contract to contradict or excuse,
    # so its prose is untouched even though it binds.
    assert passing.spec_branch == CONTRACT_CONTRADICTS
    assert passing.message == "all good"
    # The branch survives serialization into the delivered probe report.
    assert contradicted.to_dict()["spec_branch"] == CONTRACT_CONTRADICTS


def test_the_battery_seam_is_idempotent_and_survives_a_missing_spec(tmp_path):
    from probes import ProbeReport, ProbeVerdict  # noqa: PLC0415
    from run_probes import _apply_spec_conditioning  # noqa: PLC0415

    run = tmp_path / "run"
    (run / ".pipeline").mkdir(parents=True)

    report = ProbeReport(target=str(run))
    verdict = ProbeVerdict("UB-6", "fail", "observation",
                           element_ids=["scoring"])
    report.add(verdict)

    # No spec on disk: everything reads silent, nothing raises.
    _apply_spec_conditioning(run, report)
    assert verdict.spec_branch == CONTRACT_SILENT
    assert verdict.message == "observation"

    # A second pass must not append a second sentence.
    _apply_spec_conditioning(run, report)
    assert verdict.message == "observation"


def test_an_unmatched_id_reports_its_gap_in_the_delivered_message(tmp_path):
    """The probe-side gap is visible on the surface a researcher reads, rather
    than passing as a contract that happens to be silent."""
    import json  # noqa: PLC0415

    from probes import ProbeReport, ProbeVerdict  # noqa: PLC0415
    from run_probes import _apply_spec_conditioning  # noqa: PLC0415

    run = tmp_path / "run"
    (run / ".pipeline").mkdir(parents=True)
    (run / ".pipeline" / "method_spec.json").write_text(json.dumps(_spec(
        _element("scoring", role="core_methodology", status="must_replicate"),
    )), encoding="utf-8")

    report = ProbeReport(target=str(run))
    verdict = ProbeVerdict("UB-6", "fail", "observation",
                           element_ids=["scoring-v2"])
    report.add(verdict)

    _apply_spec_conditioning(run, report)

    assert verdict.spec_branch == CONTRACT_SILENT
    assert "naming gap to close" in verdict.message


# ---------------------------------------------------------------------------
# The crosswalk (R2C-072). Generated code is annotated with paper_map ids, so a
# behavioral verdict can only ever name those; the contract names its own. The
# two spaces are disjoint on every delivered run, so without this the join binds
# nothing in production however correct the rest of it is.
# ---------------------------------------------------------------------------


def test_a_verdict_carrying_paper_map_ids_binds_through_the_crosswalk():
    spec = _spec(_element("bald-uncertainty-scoring", role="core_methodology",
                          status="must_replicate",
                          paper_element_ids=["concept-bald-scoring",
                                             "eq-mutual-information"]))
    join = join_verdict(_Verdict("flag_for_researcher",
                                 ["eq-mutual-information"]), spec)
    assert join.bound
    assert [e.element_id for e in join.matched] == ["bald-uncertainty-scoring"]
    assert join.branch == CONTRACT_CONTRADICTS
    assert join.producer_fixable


def test_the_contract_id_still_binds_alongside_the_crosswalk():
    spec = _spec(_element("scoring", role="core_methodology",
                          status="must_replicate",
                          paper_element_ids=["concept-scoring"]))
    assert join_verdict(_Verdict("fail", ["scoring"]), spec).bound
    assert join_verdict(_Verdict("fail", ["concept-scoring"]), spec).bound


def test_naming_one_obligation_in_both_spaces_binds_it_once():
    spec = _spec(_element("scoring", role="core_methodology",
                          status="must_replicate",
                          paper_element_ids=["concept-scoring"]))
    join = join_verdict(_Verdict("fail", ["scoring", "concept-scoring"]), spec)
    assert [e.element_id for e in join.matched] == ["scoring"]


def test_an_empty_crosswalk_leaves_the_join_exactly_as_it_was():
    # Every spec written before the field existed. Paper-map ids bind nothing,
    # which is honest: there is no recorded link to bind them through.
    spec = _spec(_element("scoring", role="core_methodology",
                          status="must_replicate"))
    join = join_verdict(_Verdict("fail", ["concept-scoring"]), spec)
    assert not join.bound
    assert join.unmatched_ids == ("concept-scoring",)


def test_a_crosswalked_id_does_not_leak_between_elements():
    spec = _spec(
        _element("scoring", role="core_methodology", status="must_replicate",
                 paper_element_ids=["concept-scoring"]),
        _element("batching", role="supporting_component",
                 status="faithful_approximation_allowed",
                 approximations=["smaller batches at demo scale"],
                 paper_element_ids=["concept-batching"]))
    join = join_verdict(_Verdict("fail", ["concept-batching"]), spec)
    assert [e.element_id for e in join.matched] == ["batching"]
    assert join.branch == CONTRACT_CONSISTENT


def test_a_contract_element_id_wins_a_collision_with_a_crosswalk_id():
    # Pathological but cheap to pin: if the analyzer ever names a contract
    # element the same as some other element's paper-map grounding, identity
    # beats grounding.
    spec = _spec(
        _element("shared", role="core_methodology", status="must_replicate"),
        _element("other", role="supporting_component",
                 status="faithful_approximation_allowed",
                 approximations=["a demo-scale stand-in"],
                 paper_element_ids=["shared"]))
    join = join_verdict(_Verdict("fail", ["shared"]), spec)
    assert [e.element_id for e in join.matched] == ["shared"]


def test_contract_elements_exposes_both_keys_for_one_element():
    spec = _spec(_element("scoring", role="core_methodology",
                          status="must_replicate",
                          paper_element_ids=["concept-scoring"]))
    table = contract_elements(spec)
    assert table["scoring"] == table["concept-scoring"]
    assert table["scoring"][0] is table["concept-scoring"][0]
    assert table["scoring"][0].paper_element_ids == ("concept-scoring",)


# ---------------------------------------------------------------------------
# R2C-087: one-to-many anchors qualified by an exact stable probe ref.
# ---------------------------------------------------------------------------


SAMPLE_REF = "probes.time_series.sample_genuineness"
HISTORY_REF = "probes.time_series.own_history_sensitivity"
SHARED_REF = "probes.time_series.deliberate_shared_control"


def _shared_graphdeepar_spec() -> dict:
    """The current pdfgnn four-way `alg-graphdeepar` grounding."""
    return _spec(
        _element(
            "gnn-encoder-neighborhood-aggregation",
            role="core_methodology",
            status="must_replicate",
            paper_element_ids=["alg-graphdeepar"],
            verification_probe_refs=[HISTORY_REF],
        ),
        _element(
            "synchronized-batching",
            role="core_methodology",
            status="must_replicate",
            paper_element_ids=["alg-graphdeepar"],
            verification_probe_refs=[
                "probes.time_series.synchronized_batching",
            ],
        ),
        _element(
            "students-t-distribution-output",
            role="core_methodology",
            status="must_replicate",
            paper_element_ids=["alg-graphdeepar"],
            verification_probe_refs=[SAMPLE_REF],
        ),
        _element(
            "end-to-end-training",
            role="supporting_mechanism",
            status="must_replicate",
            paper_element_ids=["alg-graphdeepar"],
            verification_probe_refs=["probes.time_series.end_to_end_fit"],
        ),
    )


def test_shared_pdfgnn_anchor_keeps_four_candidates_and_ref_selects_one():
    spec = _shared_graphdeepar_spec()
    join = join_verdict(
        _Verdict(
            "pass",
            # The battery adds this id from the callable's paper-element
            # anchor while preserving any explicit ids on the verdict.
            ["alg-graphdeepar"],
            probe_id="TSF-1",
            probe_ref=SAMPLE_REF,
            bound_callables=["forecast"],
        ),
        spec,
    )

    assert [element.element_id for element in join.candidates] == [
        "gnn-encoder-neighborhood-aggregation",
        "synchronized-batching",
        "students-t-distribution-output",
        "end-to-end-training",
    ]
    assert [element.element_id for element in join.matched] == [
        "students-t-distribution-output",
    ]
    assert join.probe_ref_known is True
    assert join.reference_qualified is True


def test_deliberately_shared_ref_binds_each_declaring_obligation_once():
    spec = _spec(
        _element(
            "first",
            role="core_methodology",
            status="must_replicate",
            paper_element_ids=["paper-shared"],
            verification_probe_refs=[SHARED_REF],
        ),
        _element(
            "middle",
            role="supporting_mechanism",
            status="must_replicate",
            paper_element_ids=["paper-shared"],
            verification_probe_refs=[HISTORY_REF],
        ),
        _element(
            "last",
            role="supporting_mechanism",
            status="must_replicate",
            paper_element_ids=["paper-shared"],
            verification_probe_refs=[SHARED_REF],
        ),
    )

    join = join_verdict(
        _Verdict("pass", ["paper-shared"], probe_ref=SHARED_REF), spec,
    )

    assert [element.element_id for element in join.matched] == ["first", "last"]
    assert join.reference_qualified


def test_archived_no_ref_read_preserves_every_shared_candidate_in_order():
    archived = _shared_graphdeepar_spec()
    archived["schema_version"] = "1.11.0"
    for element in archived["methodology_replication_contract"]["elements"]:
        del element["verification_probe_refs"]
    join = join_verdict(
        _Verdict("fail", ["alg-graphdeepar"]),
        archived,
    )

    assert [element.element_id for element in join.matched] == [
        "gnn-encoder-neighborhood-aggregation",
        "synchronized-batching",
        "students-t-distribution-output",
        "end-to-end-training",
    ]
    assert join.bound
    assert join.reference_qualified is False


def test_new_battery_ref_over_archived_contract_keeps_legacy_adjudication_only():
    legacy = _spec(_element(
        "acquisition-scoring",
        role="core_methodology",
        status="must_replicate",
        paper_element_ids=["paper-selector"],
    ))

    join = join_verdict(
        _Verdict(
            "flag_for_researcher",
            ["paper-selector"],
            probe_ref="al_loop.acquisition_contract",
        ),
        legacy,
    )

    assert [element.element_id for element in join.matched] == [
        "acquisition-scoring"
    ]
    assert join.branch == CONTRACT_CONTRADICTS
    assert join.reference_contract is False
    assert join.reference_qualified is False


def test_current_schema_omission_fails_closed_as_reference_aware():
    current = _spec(_element(
        "acquisition-scoring",
        role="core_methodology",
        status="must_replicate",
        paper_element_ids=["paper-selector"],
    ))
    current["schema_version"] = "1.12.0"

    join = join_verdict(
        _Verdict(
            "pass",
            ["paper-selector"],
            probe_ref="al_loop.acquisition_contract",
        ),
        current,
    )

    assert join.candidates
    assert join.matched == ()
    assert join.reference_contract is True
    assert join.reference_qualified is False


def test_current_shared_anchor_without_ref_stays_diagnostic_not_bound():
    join = join_verdict(
        _Verdict(
            "flag_for_researcher",
            ["alg-graphdeepar"],
            probe_id="TSF-1",
        ),
        _shared_graphdeepar_spec(),
    )

    assert len(join.candidates) == 4
    assert join.matched == ()
    assert join.bound is False
    assert join.branch == CONTRACT_SILENT
    assert join.producer_fixable is False
    assert "omits the exact probe ref" in (unbound_gap_note(join) or "")


def test_raw_join_normalizes_refs_the_same_way_as_method_spec_validation():
    spec = _spec(_element(
        "sample-obligation",
        role="core_methodology",
        status="must_replicate",
        paper_element_ids=["paper-sample"],
        verification_probe_refs=[f"  {SAMPLE_REF}  "],
    ))

    join = join_verdict(
        _Verdict("pass", ["paper-sample"], probe_ref=SAMPLE_REF), spec,
    )

    assert join.known_probe_refs == (SAMPLE_REF,)
    assert [element.element_id for element in join.matched] == [
        "sample-obligation"
    ]
    assert join.reference_qualified


def test_unknown_ref_and_known_ref_without_candidate_intersection_stay_unbound():
    spec = _spec(
        _element(
            "sample-obligation",
            role="core_methodology",
            status="must_replicate",
            paper_element_ids=["paper-sample"],
            verification_probe_refs=[SAMPLE_REF],
        ),
        _element(
            "history-obligation",
            role="core_methodology",
            status="must_replicate",
            paper_element_ids=["paper-history"],
            verification_probe_refs=[HISTORY_REF],
        ),
    )

    unknown = join_verdict(
        _Verdict("pass", ["paper-sample"], probe_ref="probes.unknown"), spec,
    )
    assert unknown.candidates and not unknown.bound
    assert unknown.probe_ref_known is False
    assert "declares no such verification ref" in (unbound_gap_note(unknown) or "")

    no_intersection = join_verdict(
        _Verdict("pass", ["paper-sample"], probe_ref=HISTORY_REF), spec,
    )
    assert [e.element_id for e in no_intersection.candidates] == [
        "sample-obligation",
    ]
    assert no_intersection.probe_ref_known is True
    assert no_intersection.bound is False
    assert "none declares its exact probe ref" in (
        unbound_gap_note(no_intersection) or ""
    )


def test_missing_anchor_with_known_ref_remains_a_visible_naming_gap():
    spec = _shared_graphdeepar_spec()
    join = join_verdict(
        _Verdict("pass", ["paper-id-missing"], probe_ref=SAMPLE_REF), spec,
    )

    assert join.candidates == ()
    assert join.unmatched_ids == ("paper-id-missing",)
    assert join.probe_ref_known is True
    assert "naming gap to close" in (unbound_gap_note(join) or "")


def test_known_ref_with_unresolved_callable_is_visible_as_unanchored():
    join = join_verdict(
        _Verdict(
            "pass",
            [],
            probe_id="TSF-1",
            probe_ref=SAMPLE_REF,
            bound_callables=["forecast"],
        ),
        _shared_graphdeepar_spec(),
    )

    assert join.probe_ref_known
    assert join.declared_callables == ("forecast",)
    assert not join.bound
    note = unbound_gap_note(join) or ""
    assert "resolved to no paper-element anchor" in note
    assert "unanchored coverage" in note


def test_unknown_ref_without_anchor_is_named_instead_of_disappearing():
    join = join_verdict(
        _Verdict(
            "pass",
            [],
            probe_id="TSF-1",
            probe_ref="probes.time_series.unknown",
            bound_callables=["forecast"],
        ),
        _shared_graphdeepar_spec(),
    )

    assert not join.bound
    assert "declares no such verification ref" in (
        unbound_gap_note(join) or ""
    )


def test_known_ref_without_any_grounding_is_visible_as_unanchored():
    join = join_verdict(
        _Verdict(
            "pass",
            [],
            probe_id="TSF-1",
            probe_ref=SAMPLE_REF,
        ),
        _shared_graphdeepar_spec(),
    )

    assert not join.bound
    note = unbound_gap_note(join) or ""
    assert "declares no callable or element anchor" in note
    assert "unanchored coverage" in note


def test_direct_id_precedence_blocks_crosswalk_collision_even_when_ref_matches_other():
    spec = _spec(
        _element(
            "shared",
            role="core_methodology",
            status="must_replicate",
            verification_probe_refs=[SAMPLE_REF],
        ),
        _element(
            "other",
            role="supporting_mechanism",
            status="must_replicate",
            paper_element_ids=["shared"],
            verification_probe_refs=[HISTORY_REF],
        ),
    )

    join = join_verdict(
        _Verdict("pass", ["shared"], probe_ref=HISTORY_REF), spec,
    )

    assert [element.element_id for element in join.candidates] == ["shared"]
    assert join.matched == ()
    assert join.probe_ref_known is True


def test_direct_and_crosswalk_aliases_deduplicate_after_ref_intersection():
    spec = _spec(_element(
        "sample-obligation",
        role="core_methodology",
        status="must_replicate",
        paper_element_ids=["paper-sample"],
        verification_probe_refs=[SAMPLE_REF],
    ))

    join = join_verdict(
        _Verdict(
            "pass",
            ["paper-sample", "sample-obligation", "paper-sample"],
            probe_ref=SAMPLE_REF,
        ),
        spec,
    )

    assert [element.element_id for element in join.candidates] == [
        "sample-obligation",
    ]
    assert [element.element_id for element in join.matched] == [
        "sample-obligation",
    ]


def test_contract_crosswalk_is_an_order_preserving_multimap():
    table = contract_elements(_shared_graphdeepar_spec())

    assert [element.element_id for element in table["alg-graphdeepar"]] == [
        "gnn-encoder-neighborhood-aggregation",
        "synchronized-batching",
        "students-t-distribution-output",
        "end-to-end-training",
    ]


# --- R2C-092: derived graph-ref wiring ---------------------------------------


def _graph_mechanism(*, alignment: str, construction: str, message: str,
                     ablation: str | None = None) -> dict:
    mechanism = {
        "schema_version": "1.0",
        "alignment_element_id": alignment,
        "construction": {"element_id": construction},
        "message_passing": {"element_id": message},
        "permutation_applicability": "equivariant",
        "contribution_ablation": (
            {"kind": "empty_graph", "element_id": ablation}
            if ablation else None
        ),
        "probe_refs": {
            "parameter_agreement": "graph_mechanism.parameter_agreement",
            "construction": "graph_mechanism.construction_semantics",
            "topology": "graph_mechanism.topology_sensitivity",
            "neighbor_signal": "graph_mechanism.neighbor_sensitivity",
            "permutation": "graph_mechanism.permutation_equivalence",
            "contribution_ablation": (
                "graph_mechanism.contribution_ablation" if ablation else None
            ),
        },
    }
    return mechanism


def test_join_sees_derived_graph_refs_the_stored_json_omits():
    # The 2026-08-10/11 pdfgnn loops dropped hand-transcribed graph refs on
    # every fix iteration. The join derives ownership from the block, so a
    # graph verdict stays reference-qualified without transcription.
    spec = _spec(
        _element("graph-align", role="supporting_mechanism",
                 status="must_replicate"),
        _element("graph-build", role="supporting_mechanism",
                 status="must_replicate"),
        _element("graph-message", role="core_methodology",
                 status="must_replicate"),
    )
    spec["methodology_replication_contract"]["homogeneous_graph_mechanism"] = (
        _graph_mechanism(alignment="graph-align", construction="graph-build",
                         message="graph-message")
    )

    elements = contract_elements(spec)
    assert "graph_mechanism.alignment_prerequisite" in (
        elements["graph-align"][0].verification_probe_refs
    )
    assert "graph_mechanism.construction_semantics" in (
        elements["graph-build"][0].verification_probe_refs
    )
    assert "graph_mechanism.permutation_equivalence" in (
        elements["graph-message"][0].verification_probe_refs
    )


def test_join_strips_a_misplaced_graph_ref_under_a_mechanism():
    spec = _spec(
        _element("graph-align", role="supporting_mechanism",
                 status="must_replicate",
                 verification_probe_refs=[
                     "graph_mechanism.construction_semantics",
                 ]),
        _element("graph-build", role="supporting_mechanism",
                 status="must_replicate"),
        _element("graph-message", role="core_methodology",
                 status="must_replicate"),
    )
    spec["methodology_replication_contract"]["homogeneous_graph_mechanism"] = (
        _graph_mechanism(alignment="graph-align", construction="graph-build",
                         message="graph-message")
    )

    elements = contract_elements(spec)
    assert "graph_mechanism.construction_semantics" not in (
        elements["graph-align"][0].verification_probe_refs
    )
    assert "graph_mechanism.construction_semantics" in (
        elements["graph-build"][0].verification_probe_refs
    )


def test_join_without_a_mechanism_keeps_declared_refs_verbatim():
    spec = _spec(
        _element("scoring", role="core_methodology", status="must_replicate",
                 verification_probe_refs=[
                     "graph_mechanism.construction_semantics",
                     "active_learning.batch_composition",
                 ]),
    )

    elements = contract_elements(spec)
    assert list(elements["scoring"][0].verification_probe_refs) == [
        "graph_mechanism.construction_semantics",
        "active_learning.batch_composition",
    ]
