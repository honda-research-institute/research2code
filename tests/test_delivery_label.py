"""Delivery label rules: the three-state policy.

The two escapes under test: a probe fail/flag can no longer ship as
`verified`, and a critical/important fidelity finding can no longer
defer its way past the label. warn/unprobeable disclose, never demote.

Plus the third state (design note 2026-07-02): `verified` additionally
requires contribution evidence — zero demoters with no passing
contribution-tier probe delivers `uncertified_new_territory`, closing
the false-verified hole where a paper whose contribution probes all
came back unprobeable read `verified` on universal passes alone.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from delivery_label import (
    LABEL_DRAFT,
    LABEL_UNCERTIFIED,
    LABEL_VERIFIED,
    _graph_probe_axis,
    derive_delivery_label,
    is_contribution_probe,
)

# Calibration anchors are FROZEN copies of real run artifacts under version
# control, NOT the mutable gitignored r2c_runs/. Reading r2c_runs/ made these
# tests break every time a run was regenerated (the BADGE run's 2026-06-15
# re-run dropped the US-3 finding the test pinned). Frozen fixtures decouple the
# calibration from whatever happens to be sitting in r2c_runs/.
FIXTURES = Path(__file__).resolve().parent / "fixtures" / "delivery"


def _report(*verdicts):
    return {"verdicts": [
        {"probe_id": pid, "verdict": v, "message": f"{pid} message"}
        for pid, v in verdicts]}


def _review(*findings):
    return {"review_status": "issues_found",
            "findings": [dict(f) for f in findings]}


def _reference_spec(*refs: str) -> dict:
    return {"methodology_replication_contract": {"elements": [{
        "element_id": "contribution",
        "role": "core_methodology",
        "replication_status": "must_replicate",
        "technical_concept": "contribution",
        "required_behavior": "the contribution changes the result",
        "paper_section": "Section 3",
        "acceptable_approximations": [],
        "forbidden_substitutions": [],
        "paper_element_ids": ["paper-contribution"],
        "verification_probe_refs": list(refs),
    }]}}


def test_clean_battery_and_review_is_verified():
    # The CT-1 pass is the contribution evidence — universal passes alone
    # no longer certify (see the new-territory tests below).
    out = derive_delivery_label(
        _report(("US-1", "pass"), ("UB-7", "pass"), ("CT-1", "pass")),
        _review())
    assert out["label"] == LABEL_VERIFIED
    assert out["reasons"] == []
    assert out["probe_counts"] == {"pass": 3}
    assert "missing_probe_family" not in out


def test_probe_fail_demotes():
    out = derive_delivery_label(
        _report(("US-3", "fail"), ("US-1", "pass")), None)
    assert out["label"] == LABEL_DRAFT
    assert [r["id"] for r in out["reasons"]] == ["US-3"]


def test_researcher_flag_demotes():
    out = derive_delivery_label(_report(("UB-7", "flag_for_researcher")),
                                None)
    assert out["label"] == LABEL_DRAFT
    assert out["reasons"][0]["verdict"] == "flag_for_researcher"


def test_warn_and_unprobeable_disclose_but_do_not_demote():
    out = derive_delivery_label(
        _report(("US-2a", "warn"), ("UB-6", "unprobeable"),
                ("US-1", "pass"), ("AL-5", "pass")), None)
    assert out["label"] == LABEL_VERIFIED
    assert {d["id"] for d in out["disclosures"]} == {"US-2a", "UB-6"}


def test_not_applicable_is_counted_separately_and_never_demotes():
    out = derive_delivery_label(
        _report(("US-1", "pass"), ("MP-1", "not_applicable"),
                ("CT-1", "pass")),
        None,
    )

    assert out["label"] == LABEL_VERIFIED
    assert out["reasons"] == []
    assert out["probe_counts"] == {"pass": 2, "not_applicable": 1}


def test_missing_battery_report_demotes():
    out = derive_delivery_label(None, None)
    assert out["label"] == LABEL_DRAFT
    assert out["reasons"][0]["source"] == "battery"


@pytest.mark.parametrize("severity", ["critical", "important"])
@pytest.mark.parametrize("status", [None, "pending", "failed", "needs_user"])
def test_unresolved_gating_finding_demotes(severity, status):
    finding = {"id": "F003", "severity": severity, "description": "d"}
    if status is not None:
        finding["resolution_status"] = status
    out = derive_delivery_label(_report(("US-1", "pass")), _review(finding))
    assert out["label"] == LABEL_DRAFT
    assert out["reasons"][0]["source"] == "fidelity_review"


def test_applied_resolution_and_nice_to_have_do_not_demote():
    out = derive_delivery_label(
        _report(("US-1", "pass"), ("KD-1", "pass")), _review(
            {"id": "F1", "severity": "important", "description": "d",
             "resolution_status": "applied"},
            {"id": "F2", "severity": "nice-to-have", "description": "d"},
        ))
    assert out["label"] == LABEL_VERIFIED


# ---------------------------------------------------------------------------
# The third state: uncertified — new territory (false-verified hole closed).
# ---------------------------------------------------------------------------


def test_universal_only_passes_deliver_uncertified_not_verified():
    # THE false-verified hole (2026-06-18): every universal check passes,
    # nothing demotes, but no contribution-tier probe certified anything.
    out = derive_delivery_label(
        _report(("US-1", "pass"), ("UB-5", "pass"), ("UB-7", "pass")),
        _review(), paradigm="trajectory_forecasting.diffusion")
    assert out["label"] == LABEL_UNCERTIFIED
    assert out["reasons"] == []
    assert out["missing_probe_family"] == "trajectory_forecasting.diffusion"


def test_all_unprobeable_contribution_tier_is_uncertified():
    # Contribution probes EXIST for the family but none could bind to this
    # artifact — unprobeable is not certification (zoo case a).
    out = derive_delivery_label(
        _report(("US-1", "pass"), ("UB-6", "pass"),
                ("AL-1", "unprobeable"), ("CT-1", "unprobeable")),
        None)
    assert out["label"] == LABEL_UNCERTIFIED
    assert {d["id"] for d in out["disclosures"]} == {"AL-1", "CT-1"}


def test_uncertified_without_paradigm_records_unknown_family():
    out = derive_delivery_label(_report(("US-1", "pass")), None)
    assert out["label"] == LABEL_UNCERTIFIED
    assert out["missing_probe_family"] == "unknown"


def test_one_contribution_pass_is_sufficient_evidence():
    # Zoo case b: one paradigm-tier pass + zero demoters → verified, even
    # with other contribution probes unprobeable.
    out = derive_delivery_label(
        _report(("US-1", "pass"), ("MP-4", "pass"), ("MP-1", "unprobeable")),
        None)
    assert out["label"] == LABEL_VERIFIED
    assert "missing_probe_family" not in out


def test_reference_aware_contract_requires_a_qualified_spec_join():
    spec = _reference_spec("claims.contribution_floor")
    base = {
        "probe_id": "CT-1",
        "verdict": "pass",
        "message": "contribution differs from null",
        "element_ids": ["paper-contribution"],
    }

    missing_ref = derive_delivery_label(
        {"verdicts": [base]}, None, paradigm="active_learning", spec=spec)
    assert missing_ref["label"] == LABEL_UNCERTIFIED
    assert missing_ref["disclosures"][0]["verdict"] == "binding_gap"
    assert "omits the exact probe ref" in missing_ref["disclosures"][0][
        "message"
    ]

    wrong_ref = derive_delivery_label(
        {"verdicts": [{**base, "probe_ref": "al_loop.microharness"}]},
        None, paradigm="active_learning", spec=spec)
    assert wrong_ref["label"] == LABEL_UNCERTIFIED
    assert "declares no such verification ref" in wrong_ref["disclosures"][
        0
    ]["message"]

    dangling_id = derive_delivery_label(
        {"verdicts": [{**base,
                       "probe_ref": "claims.contribution_floor",
                       "element_ids": ["not-in-the-contract"]}]},
        None, paradigm="active_learning", spec=spec)
    assert dangling_id["label"] == LABEL_UNCERTIFIED
    assert "names element(s) the methodology contract does not carry" in (
        dangling_id["disclosures"][0]["message"]
    )

    qualified = derive_delivery_label(
        {"verdicts": [{**base, "probe_ref": "claims.contribution_floor"}]},
        None, paradigm="active_learning", spec=spec)
    assert qualified["label"] == LABEL_VERIFIED
    assert qualified["disclosures"] == []


def test_ungrounded_contribution_pass_discloses_generic_binding_gap():
    spec = _reference_spec("claims.contribution_floor")
    out = derive_delivery_label(
        {"verdicts": [{
            "probe_id": "KD-1",
            "verdict": "pass",
            "message": "distillation behavior passed",
        }]},
        None,
        paradigm="knowledge_distillation",
        spec=spec,
    )

    assert out["label"] == LABEL_UNCERTIFIED
    assert out["disclosures"][0]["verdict"] == "binding_gap"
    assert "carries neither the exact probe ref nor methodology grounding" in (
        out["disclosures"][0]["message"]
    )


def test_tsf_prefix_counts_only_with_exact_reference_qualified_binding():
    ref = "time_series_forecasting.sample_genuineness"
    spec = _reference_spec(ref)
    row = {
        "probe_id": "TSF-1",
        "probe_ref": ref,
        "verdict": "pass",
        "message": "genuine Student-t samples",
        "bound_callables": ["forecast"],
    }

    unbound = derive_delivery_label(
        {"verdicts": [row]},
        None,
        paradigm="time_series_forecasting",
        spec=spec,
    )
    assert unbound["label"] == LABEL_UNCERTIFIED
    assert unbound["disclosures"][0]["verdict"] == "binding_gap"

    qualified = derive_delivery_label(
        {"verdicts": [{**row, "element_ids": ["paper-contribution"]}]},
        None,
        paradigm="time_series_forecasting",
        spec=spec,
    )
    assert qualified["label"] == LABEL_VERIFIED


def test_archived_contract_without_probe_ref_field_keeps_legacy_read():
    spec = _reference_spec("claims.contribution_floor")
    del spec["methodology_replication_contract"]["elements"][0][
        "verification_probe_refs"
    ]

    out = derive_delivery_label(
        _report(("CT-1", "pass")), None, spec=spec)

    assert out["label"] == LABEL_VERIFIED


def test_current_schema_cannot_omit_carrier_into_legacy_certification():
    spec = _reference_spec("claims.contribution_floor")
    spec["schema_version"] = "1.12.0"
    del spec["methodology_replication_contract"]["elements"][0][
        "verification_probe_refs"
    ]

    out = derive_delivery_label(
        _report(("CT-1", "pass")), None,
        paradigm="active_learning", spec=spec,
    )

    assert out["label"] == LABEL_UNCERTIFIED


def test_flag_demotes_even_when_contribution_probes_pass():
    # Zoo case c: draft wins over both other states.
    out = derive_delivery_label(
        _report(("CT-1", "pass"), ("AL-5", "pass"),
                ("UB-7", "flag_for_researcher")), None)
    assert out["label"] == LABEL_DRAFT
    assert "missing_probe_family" not in out


def test_claims_ledger_probe_is_not_contribution_evidence():
    # CT-3 is the claims LEDGER (bookkeeping); only CT-1/CT-5 certify the
    # contribution itself.
    out = derive_delivery_label(
        _report(("US-1", "pass"), ("CT-3", "pass")), None)
    assert out["label"] == LABEL_UNCERTIFIED


def test_only_discriminating_graph_ablation_is_contribution_evidence():
    assert is_contribution_probe("HG-7")
    for probe_id in ("HG-1", "HG-2", "HG-3", "HG-4", "HG-5", "HG-6"):
        assert not is_contribution_probe(probe_id)


_GRAPH_REFS = {
    "HG-1": "graph_mechanism.alignment_prerequisite",
    "HG-2": "graph_mechanism.parameter_agreement",
    "HG-3": "graph_mechanism.construction_semantics",
    "HG-4": "graph_mechanism.topology_sensitivity",
    "HG-5": "graph_mechanism.neighbor_sensitivity",
    "HG-6": "graph_mechanism.permutation_equivalence",
    "HG-7": "graph_mechanism.contribution_ablation",
}


def _graph_element(element_id: str, *refs: str) -> dict:
    return {
        "element_id": element_id,
        "role": "core_methodology",
        "replication_status": "must_replicate",
        "technical_concept": element_id,
        "required_behavior": f"exercise {element_id}",
        "paper_section": "Section 3",
        "acceptable_approximations": [],
        "forbidden_substitutions": [],
        "paper_element_ids": [f"paper-{element_id}"],
        "verification_probe_refs": list(refs),
    }


def _graph_spec() -> dict:
    return {
        "schema_version": "1.14.0",
        "methodology_replication_contract": {
            "elements": [
                _graph_element("graph-alignment", _GRAPH_REFS["HG-1"]),
                _graph_element(
                    "graph-construction",
                    _GRAPH_REFS["HG-2"],
                    _GRAPH_REFS["HG-3"],
                ),
                _graph_element(
                    "message-passing",
                    _GRAPH_REFS["HG-4"],
                    _GRAPH_REFS["HG-5"],
                    _GRAPH_REFS["HG-6"],
                ),
                _graph_element("graph-ablation", _GRAPH_REFS["HG-7"]),
            ],
            "homogeneous_graph_mechanism": {
                "schema_version": "1.0",
                "alignment_element_id": "graph-alignment",
                "construction": {
                    "element_id": "graph-construction",
                    "callable": {
                        "module": "method.model",
                        "qualname": "construct_graph_exact",
                    },
                    "feature_input_root": "batch.features",
                    "feature_parameter": "features_exact",
                    "output_selector": {"kind": "return_value"},
                    "threshold": {
                        "metric": "cosine_similarity",
                        "comparison": "greater_than_or_equal",
                        "parameter": {
                            "params_name": "similarity_threshold",
                            "callable_parameter": "threshold_exact",
                        },
                    },
                    "cap": {"kind": "none"},
                    "self_loop_policy": "forbidden",
                    "direction_policy": "undirected_bidirectional",
                },
                "message_passing": {
                    "element_id": "message-passing",
                    "callable": {
                        "module": "method.model",
                        "qualname": "execute_graph_exact",
                    },
                    "graph_parameter": "graph_exact",
                    "neighbor_signal_root": "batch.signals",
                    "neighbor_signal_parameter": "signals_exact",
                    "output_root": "batch.outputs",
                },
                "permutation_applicability": "equivariant",
                "contribution_ablation": {
                    "kind": "empty_graph",
                    "element_id": "graph-ablation",
                    "discriminating_probe_ref": _GRAPH_REFS["HG-5"],
                },
                "probe_refs": {
                    "parameter_agreement": _GRAPH_REFS["HG-2"],
                    "construction": _GRAPH_REFS["HG-3"],
                    "topology": _GRAPH_REFS["HG-4"],
                    "neighbor_signal": _GRAPH_REFS["HG-5"],
                    "permutation": _GRAPH_REFS["HG-6"],
                    "contribution_ablation": _GRAPH_REFS["HG-7"],
                },
            },
        },
    }


def _graph_plan() -> dict:
    groundings = {
        probe_ref: {
            "element_ids": [owner],
            "bound_callables": callables,
        }
        for probe_ref, owner, callables in (
            (_GRAPH_REFS["HG-1"], "graph-alignment",
             ["method.training:_prepare_graph_batch"]),
            (_GRAPH_REFS["HG-2"], "graph-construction",
             ["method.model:construct_graph_exact"]),
            (_GRAPH_REFS["HG-3"], "graph-construction",
             ["method.model:construct_graph_exact"]),
            (_GRAPH_REFS["HG-4"], "message-passing",
             ["method.model:execute_graph_exact"]),
            (_GRAPH_REFS["HG-5"], "message-passing",
             ["method.model:execute_graph_exact"]),
            (_GRAPH_REFS["HG-6"], "message-passing",
             ["method.model:execute_graph_exact"]),
            (_GRAPH_REFS["HG-7"], "graph-ablation",
             ["method.model:execute_graph_exact"]),
        )
    }
    return {
        "schema_version": "1.0",
        "status": "ready",
        "representation": "sparse_edge_index",
        "alignment": {
            "status": "pass",
            "preparation_callable": {
                "module": "method.training",
                "name": "_prepare_graph_batch",
            },
        },
        "alignment_runtime_receipt": {
            "status": "pass",
            "verified": True,
            "reason": "stage_2d_runtime_alignment_verified",
            "authority_digests": {
                ".pipeline/arch_contract.json": f"sha256:{'c' * 64}",
            },
        },
        "callable_liveness_receipt": {
            "status": "pass",
            "verified": True,
            "reason": "graph_callable_liveness_verified",
        },
        "construction": {
            "callable": {
                "module": "method.model",
                "qualname": "construct_graph_exact",
            },
            "feature_input_root": "batch.features",
            "threshold": {
                "metric": "cosine_similarity",
                "comparison": "greater_than_or_equal",
                "value": 0.5,
                "parameter": {
                    "params_name": "similarity_threshold",
                    "callable_parameter": "threshold_exact",
                },
            },
            "self_loop_policy": "forbidden",
            "direction_policy": "undirected_bidirectional",
            "cap": {"kind": "none"},
        },
        "parameter_authority": {
            "status": "pass",
            "bindings": {
                "similarity_threshold": {
                    "carrier_value": 0.5,
                    "params_value": 0.5,
                    "paper_value": None,
                    "source": "paper",
                    "suppressed": False,
                    "duplicates": [],
                }
            },
        },
        "execution": {
            "callable": {
                "module": "method.model",
                "qualname": "execute_graph_exact",
            },
            "neighbor_signal_root": "batch.signals",
            "output_root": "batch.outputs",
            "seed": 1729,
            "tolerance": 1e-8,
        },
        "ablation": {
            "kind": "empty_graph",
            "element_id": "graph-ablation",
            "discriminating_probe_ref": _GRAPH_REFS["HG-5"],
        },
        "groundings": groundings,
        "fixture": {
            "scope": "delivery-label-graph-fixture",
            "entity_ids": ["a", "b", "c"],
            "expected_graph": [[0, 1], [1, 0]],
            "comparison_witness": {
                "metric": "cosine_similarity",
                "comparison": "greater_than_or_equal",
                "threshold_value": 1.0,
                "entity_axis": 0,
                "feature_input": [
                    [1.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                ],
                "expected_graph": [[0, 1], [1, 0]],
            },
            "parameter_interventions": {
                "similarity_threshold": {
                    "nominal_value": 0.5,
                    "alternate_value": 0.75,
                    "value": 0.75,
                    "callable_parameter": "threshold_exact",
                    "expected_graph": [[], []],
                }
            },
            "topology_intervention": {
                "remove_edges": [[0, 1]],
                "targets": [1],
            },
            "neighbor_intervention": {
                "source": 0,
                "target": 1,
                "delta": 1.0,
                "control_targets": [2],
            },
            "permutation": [1, 0, 2],
        },
    }


def _graph_pass_evidence(probe_id: str, trace: dict) -> dict:
    evidence = {"trace": trace}
    if probe_id == "HG-1":
        evidence.update({
            "validator": "validate_arch_contract_runtime.py",
            "authority_paths": [".pipeline/arch_contract.json"],
        })
    elif probe_id == "HG-2":
        evidence["parameters"] = [{
            "role": "threshold",
            "params_name": "similarity_threshold",
            "carrier_value": 0.5,
            "params_value": 0.5,
            "paper_value": None,
            "planned_value": 0.5,
            "runtime_value": 0.5,
            "source": "paper",
            "suppressed": False,
            "duplicates": [],
        }]
        evidence["parameter_interventions"] = [{
            "role": "threshold",
            "params_name": "similarity_threshold",
            "callable_parameter": "threshold_exact",
            "nominal_value": 0.5,
            "intervention_value": 0.75,
            "runtime_value": 0.75,
            "expected_nominal_edge_symmetric_difference": 2,
            "observed_nominal_edge_symmetric_difference": 2,
            "expected_edge_count": 0,
            "constructed_edge_count": 0,
        }]
    elif probe_id == "HG-3":
        evidence.update({
            "representation": "sparse_edge_index",
            "node_count": 3,
            "expected_edge_count": 2,
            "constructed_edge_count": 2,
            "observed_edge_symmetric_difference": 0,
            "self_loop_policy": "forbidden",
            "direction_policy": "undirected_bidirectional",
            "cap_kind": "none",
            "comparison_witness": {
                "metric": "cosine_similarity",
                "comparison": "greater_than_or_equal",
                "threshold_value": 1.0,
                "identical_entity_positions": [0, 1],
                "runtime_threshold_value": 1.0,
                "expected_edge_count": 2,
                "constructed_edge_count": 2,
                "observed_edge_symmetric_difference": 0,
            },
        })
    elif probe_id == "HG-4":
        evidence.update({
            "effect_threshold": 1e-8,
            "baseline_max_abs": 1.0,
            "changed_max_abs": 1.0,
            "effect_scale": 1.0,
            "seed": 1729,
            "tolerance": 1e-8,
            "removed_edges": [[0, 1], [1, 0]],
            "target_entities": [1],
            "target_max_abs_deltas": [1.0],
            "maximum_target_delta": 1.0,
        })
    elif probe_id == "HG-5":
        evidence.update({
            "effect_threshold": 1e-8,
            "baseline_max_abs": 1.0,
            "changed_max_abs": 1.0,
            "effect_scale": 1.0,
            "seed": 1729,
            "tolerance": 1e-8,
            "source_entity": 0,
            "target_entity": 1,
            "signal_delta": 1.0,
            "target_max_abs_delta": 1.0,
            "control_max_abs_deltas": {"2": 0.0},
        })
    elif probe_id == "HG-6":
        evidence.update({
            "effect_threshold": 1e-8,
            "baseline_max_abs": 1.0,
            "changed_max_abs": 1.0,
            "effect_scale": 1.0,
            "seed": 1729,
            "tolerance": 1e-8,
            "permutation_new_to_old": [1, 0, 2],
            "restored_max_abs_delta": 0.0,
        })
    else:
        real = _graph_pass_evidence("HG-5", trace)
        real.pop("trace")
        null = dict(real)
        null["target_max_abs_delta"] = 0.0
        evidence.update({
            "seed": 1729,
            "tolerance": 1e-8,
            "ablation_kind": "empty_graph",
            "discriminating_probe_ref": _GRAPH_REFS["HG-5"],
            "real_status": "pass",
            "real_evidence": real,
            "null_status": "fail",
            "null_reason_code": "neighbor_response_absent",
            "null_evidence": null,
        })
    return evidence


def _graph_report(plan: dict | None = None) -> dict:
    from probes.graph_mechanism import graph_plan_trace

    plan = plan or _graph_plan()
    owners = {
        "HG-1": "graph-alignment",
        "HG-2": "graph-construction",
        "HG-3": "graph-construction",
        "HG-4": "message-passing",
        "HG-5": "message-passing",
        "HG-6": "message-passing",
        "HG-7": "graph-ablation",
    }
    callables = {
        "HG-1": ["method.training:_prepare_graph_batch"],
        "HG-2": ["method.model:construct_graph_exact"],
        "HG-3": ["method.model:construct_graph_exact"],
        "HG-4": ["method.model:execute_graph_exact"],
        "HG-5": ["method.model:execute_graph_exact"],
        "HG-6": ["method.model:execute_graph_exact"],
        "HG-7": ["method.model:execute_graph_exact"],
    }
    trace = graph_plan_trace(plan)
    return {"verdicts": [
        {
            "probe_id": probe_id,
            "probe_ref": probe_ref,
            "verdict": "pass",
            "message": f"{probe_id} passed",
            "element_ids": [owners[probe_id]],
            "bound_callables": callables[probe_id],
            "evidence": json.dumps(_graph_pass_evidence(probe_id, trace)),
        }
        for probe_id, probe_ref in _GRAPH_REFS.items()
    ]}


def _structured_graph_demo_verdict() -> dict:
    return {
        "schema_version": "2.0.0",
        "verdict": "succeeded",
        "evidence_status": {
            "execution": {"status": "completed", "reasons": []},
            "evaluation_validity": {"status": "valid", "reasons": []},
            "mechanism": {"status": "undetermined", "reasons": []},
            "skill": {"status": "undetermined", "reasons": []},
            "paper_benchmark": {"status": "not_assessed", "reasons": []},
        },
    }


def test_graph_ladder_certifies_only_exact_plan_bound_rows():
    plan = _graph_plan()
    out = derive_delivery_label(
        _graph_report(plan), None, spec=_graph_spec(),
        graph_execution_plan=plan,
        demo_verdict=_structured_graph_demo_verdict(),
    )

    assert out["label"] == LABEL_VERIFIED
    assert out["graph_evidence_applicability"] == "homogeneous_graph_v1"
    axes = out["demo_verdict"]["evidence_status"]
    assert axes["graph_alignment"]["status"] == "demonstrated"
    assert axes["graph_construction"]["status"] == "demonstrated"
    assert axes["mechanism"]["status"] == "demonstrated"
    assert axes["contribution"]["status"] == "demonstrated"
    assert axes["skill"]["status"] == "undetermined"
    assert axes["paper_benchmark"]["status"] == "not_assessed"


def test_graph_ladder_missing_structured_demo_cannot_verify():
    plan = _graph_plan()

    out = derive_delivery_label(
        _graph_report(plan), None, spec=_graph_spec(),
        graph_execution_plan=plan,
    )

    assert out["label"] == LABEL_DRAFT
    assert {reason["id"] for reason in out["reasons"]} == {
        "structured_demo_evidence_missing"
    }
    assert out["graph_evidence_applicability"] == "homogeneous_graph_v1"
    axes = out["demo_verdict"]["evidence_status"]
    assert axes["graph_alignment"]["status"] == "demonstrated"
    assert axes["graph_construction"]["status"] == "demonstrated"
    assert axes["mechanism"]["status"] == "demonstrated"
    assert axes["contribution"]["status"] == "demonstrated"
    assert axes["skill"]["status"] == "undetermined"
    assert axes["paper_benchmark"]["status"] == "undetermined"


def test_graph_ladder_requires_external_plan_authority():
    out = derive_delivery_label(
        _graph_report(), None, spec=_graph_spec(),
        demo_verdict=_structured_graph_demo_verdict(),
    )

    assert out["label"] == LABEL_UNCERTIFIED


def test_graph_ladder_rejects_wrong_callables_even_with_seven_passes():
    report = _graph_report()
    for row in report["verdicts"]:
        row["bound_callables"] = ["method.model:definitely_wrong"]

    out = derive_delivery_label(
        report, None, spec=_graph_spec(), graph_execution_plan=_graph_plan(),
        demo_verdict=_structured_graph_demo_verdict(),
    )

    assert out["label"] == LABEL_UNCERTIFIED


@pytest.mark.parametrize(
    "mutation", ["wrong-owner", "missing-plan-digest"],
)
def test_graph_ladder_rejects_wrong_owner_or_missing_digest(mutation):
    report = _graph_report()
    if mutation == "wrong-owner":
        report["verdicts"][3]["element_ids"] = ["graph-construction"]
    else:
        # JSON evidence is immutable text at the delivery boundary; make the
        # missing-digest mutation explicit after decoding.
        evidence = json.loads(report["verdicts"][2]["evidence"])
        evidence["trace"].pop("execution_plan_digest")
        report["verdicts"][2]["evidence"] = json.dumps(evidence)

    out = derive_delivery_label(
        report, None, spec=_graph_spec(), graph_execution_plan=_graph_plan(),
        demo_verdict=_structured_graph_demo_verdict(),
    )

    assert out["label"] == LABEL_UNCERTIFIED


def test_graph_ladder_rejects_cross_row_digest_drift():
    report = _graph_report()
    evidence = json.loads(report["verdicts"][4]["evidence"])
    evidence["trace"]["fixture_digest"] = f"sha256:{'c' * 64}"
    report["verdicts"][4]["evidence"] = json.dumps(evidence)

    out = derive_delivery_label(
        report, None, spec=_graph_spec(), graph_execution_plan=_graph_plan(),
        demo_verdict=_structured_graph_demo_verdict(),
    )

    assert out["label"] == LABEL_UNCERTIFIED


def test_graph_ladder_rejects_digest_only_or_incomplete_pass_evidence():
    report = _graph_report()
    evidence = json.loads(report["verdicts"][3]["evidence"])
    evidence.pop("maximum_target_delta")
    report["verdicts"][3]["evidence"] = json.dumps(evidence)

    out = derive_delivery_label(
        report, None, spec=_graph_spec(), graph_execution_plan=_graph_plan(),
        demo_verdict=_structured_graph_demo_verdict(),
    )

    assert out["label"] == LABEL_UNCERTIFIED


@pytest.mark.parametrize(
    "mutation",
    ["missing-plan-witness", "missing-observation", "forged-runtime-threshold"],
)
def test_graph_ladder_rejects_missing_or_forged_comparison_witness(mutation):
    plan = _graph_plan()
    if mutation == "missing-plan-witness":
        plan["fixture"].pop("comparison_witness")
        report = _graph_report(plan)
    else:
        report = _graph_report(plan)
        evidence = json.loads(report["verdicts"][2]["evidence"])
        if mutation == "missing-observation":
            evidence.pop("comparison_witness")
        else:
            evidence["comparison_witness"]["runtime_threshold_value"] = 0.5
        report["verdicts"][2]["evidence"] = json.dumps(evidence)

    out = derive_delivery_label(
        report,
        None,
        spec=_graph_spec(),
        graph_execution_plan=plan,
        demo_verdict=_structured_graph_demo_verdict(),
    )

    assert out["label"] == LABEL_UNCERTIFIED


def test_graph_ladder_rejects_forged_hg6_effect_threshold():
    report = _graph_report()
    evidence = json.loads(report["verdicts"][5]["evidence"])
    evidence["effect_threshold"] = 1_000_000.0
    report["verdicts"][5]["evidence"] = json.dumps(evidence)

    out = derive_delivery_label(
        report, None, spec=_graph_spec(), graph_execution_plan=_graph_plan(),
        demo_verdict=_structured_graph_demo_verdict(),
    )

    assert out["label"] == LABEL_UNCERTIFIED


def test_graph_ladder_rejects_forged_hg7_null_effect_threshold():
    report = _graph_report()
    evidence = json.loads(report["verdicts"][6]["evidence"])
    evidence["null_evidence"]["target_max_abs_delta"] = 5.0
    evidence["null_evidence"]["effect_threshold"] = 10.0
    report["verdicts"][6]["evidence"] = json.dumps(evidence)

    out = derive_delivery_label(
        report, None, spec=_graph_spec(), graph_execution_plan=_graph_plan(),
        demo_verdict=_structured_graph_demo_verdict(),
    )

    assert out["label"] == LABEL_UNCERTIFIED


@pytest.mark.parametrize("mutation", ["missing-parameters", "forged-carrier"])
def test_graph_ladder_rejects_incomplete_nominal_parameter_authority(mutation):
    report = _graph_report()
    evidence = json.loads(report["verdicts"][1]["evidence"])
    if mutation == "missing-parameters":
        evidence.pop("parameters")
    else:
        evidence["parameters"][0]["carrier_value"] = 999.0
    report["verdicts"][1]["evidence"] = json.dumps(evidence)

    out = derive_delivery_label(
        report, None, spec=_graph_spec(), graph_execution_plan=_graph_plan(),
        demo_verdict=_structured_graph_demo_verdict(),
    )

    assert out["label"] == LABEL_UNCERTIFIED


def test_graph_ladder_requires_verified_callable_liveness_receipt():
    plan = _graph_plan()
    plan["callable_liveness_receipt"]["verified"] = False
    report = _graph_report(plan)

    out = derive_delivery_label(
        report, None, spec=_graph_spec(), graph_execution_plan=plan,
        demo_verdict=_structured_graph_demo_verdict(),
    )

    assert out["label"] == LABEL_UNCERTIFIED


def test_hg1_alignment_does_not_require_callable_liveness_receipt():
    plan = _graph_plan()
    plan.pop("callable_liveness_receipt")
    report = _graph_report(plan)
    report["verdicts"] = [report["verdicts"][0]]

    axis = _graph_probe_axis(
        report, _graph_spec(), "graph_alignment", plan
    )

    assert axis["status"] == "demonstrated"


def test_hg1_alignment_still_requires_verified_alignment_receipt():
    plan = _graph_plan()
    plan.pop("callable_liveness_receipt")
    plan["alignment_runtime_receipt"]["verified"] = False
    report = _graph_report(plan)
    report["verdicts"] = [report["verdicts"][0]]

    axis = _graph_probe_axis(
        report, _graph_spec(), "graph_alignment", plan
    )

    assert axis["status"] == "undetermined"


def test_hg2_and_higher_still_require_callable_liveness_receipt():
    plan = _graph_plan()
    plan.pop("callable_liveness_receipt")
    report = _graph_report(plan)
    report["verdicts"] = report["verdicts"][:3]

    axis = _graph_probe_axis(
        report, _graph_spec(), "graph_construction", plan
    )

    assert axis["status"] == "undetermined"


def test_graph_ladder_rejects_a_different_external_plan():
    report = _graph_report()
    different_plan = _graph_plan()
    different_plan["execution"]["seed"] += 1

    out = derive_delivery_label(
        report, None, spec=_graph_spec(),
        graph_execution_plan=different_plan,
        demo_verdict=_structured_graph_demo_verdict(),
    )

    assert out["label"] == LABEL_UNCERTIFIED


def test_graph_ladder_rejects_malformed_duplicate_raw_row():
    report = _graph_report()
    duplicate = dict(report["verdicts"][-1])
    duplicate["probe_ref"] = "graph_mechanism.neighbor_sensitivity"
    report["verdicts"].append(duplicate)

    out = derive_delivery_label(
        report, None, spec=_graph_spec(), graph_execution_plan=_graph_plan(),
        demo_verdict=_structured_graph_demo_verdict(),
    )

    assert out["label"] == LABEL_UNCERTIFIED


def test_missing_battery_report_is_draft_not_uncertified():
    # Absence of verification evidence stays draft (fail-closed), never
    # "new territory".
    out = derive_delivery_label(None, None)
    assert out["label"] == LABEL_DRAFT


def test_probe_evidence_propagates_into_reasons():
    # The proximity-to-correct summary groups demoters by evidence trace.
    report = {"verdicts": [
        {"probe_id": "UB-5", "verdict": "fail", "message": "m",
         "evidence": "method/train.py:42"},
    ]}
    out = derive_delivery_label(report, None)
    assert out["reasons"][0]["evidence"] == "method/train.py:42"


# ---------------------------------------------------------------------------
# Calibration against the real 2026-06-12 run artifacts (both correctly
# draft today: GBALD's dead geometric term + important provenance finding;
# BADGE's embellished learning-rate quote).
# ---------------------------------------------------------------------------


def _live(slug, with_review=False):
    pipeline = FIXTURES / slug
    probe = json.loads((pipeline / "probe_report.json").read_text())
    review = None
    if with_review and (pipeline / "review_report.json").is_file():
        review = json.loads((pipeline / "review_report.json").read_text())
    return derive_delivery_label(probe, review)


def test_live_bayesian_is_draft_for_the_flag_and_the_provenance_finding():
    # Frozen bayesian-active-learning (GBALD) run: a flag-for-researcher probe
    # (UB-7) plus an unresolved important fidelity finding (F003) both demote.
    out = _live("bayesian-active-learning", with_review=True)
    assert out["label"] == LABEL_DRAFT
    sources = {(r["source"], r["id"]) for r in out["reasons"]}
    assert ("probe", "UB-7") in sources
    assert ("fidelity_review", "F003") in sources


def test_live_badge_is_draft_for_the_fidelity_finding():
    # Frozen deep-batch-active-learning (BADGE) run: the probe sweep is clean, so
    # probe-only it would be verified; the draft label comes from an important
    # fidelity finding (F001 — the notebook imports sklearn, missing from
    # requirements.txt). The probe-driven demotion rule is covered separately by
    # test_probe_fail_demotes.
    probe_only = _live("deep-batch-active-learning")
    assert probe_only["label"] == LABEL_VERIFIED
    assert probe_only["reasons"] == []

    out = _live("deep-batch-active-learning", with_review=True)
    assert out["label"] == LABEL_DRAFT
    assert {(r["source"], r["id"]) for r in out["reasons"]} == {("fidelity_review", "F001")}


# ---------------------------------------------------------------------------
# Approximated core (maintainer-approved 2026-07-05): approved approximations on a
# core_methodology element demote — a contribution-tier pass certifies a
# family behavior, not an approximated core (the SRL re-roll #1 case, where
# MP-1 passed while the learned value network shipped approximated).
# ---------------------------------------------------------------------------


def _approx(eid="value-network", status="faithful_approximation_allowed"):
    return {"element_id": eid, "replication_status": status}


def test_approximated_core_demotes_despite_contribution_evidence():
    out = derive_delivery_label(
        _report(("US-1", "pass"), ("MP-1", "pass")), _review(),
        core_approximations=[_approx()])
    assert out["label"] == LABEL_DRAFT
    assert [r["source"] for r in out["reasons"]] == ["approximated_core"]
    assert "value-network" in out["reasons"][0]["message"]
    assert "approved approximation" in out["reasons"][0]["message"]


def test_approximated_core_without_evidence_stays_uncertified_and_discloses():
    out = derive_delivery_label(
        _report(("US-1", "pass")), None, paradigm="motion_planning",
        core_approximations=[_approx()])
    assert out["label"] == LABEL_UNCERTIFIED
    assert out["reasons"] == []
    assert any(d["source"] == "approximated_core" for d in out["disclosures"])
    assert out["missing_probe_family"] == "motion_planning"


def test_approximated_core_rides_along_with_other_demoters():
    out = derive_delivery_label(
        _report(("US-3", "fail"), ("CT-1", "pass")), None,
        core_approximations=[_approx()])
    assert out["label"] == LABEL_DRAFT
    sources = [r["source"] for r in out["reasons"]]
    assert sources[0] == "approximated_core"
    assert "probe" in sources


def test_no_approximations_leaves_derivation_unchanged():
    out = derive_delivery_label(_report(("CT-1", "pass")), None)
    assert out["label"] == LABEL_VERIFIED


def test_schema_refuses_verified_with_completeness_causes():
    import pytest as _pytest
    from schemas.final_manifest import DeliveryVerdict

    with _pytest.raises(ValueError, match="completeness cause"):
        DeliveryVerdict(
            label="verified",
            reasons=[{"source": "approximated_core", "id": "x",
                      "message": "m"}])


# ---------------------------------------------------------------------------
# Neutral-plan label cap (R2C-032, approved 2026-07-28): a delivery built on
# the generic provisional build plan was never held to any family's build
# conventions, so it caps at uncertified — new territory whatever its probe
# verdicts. Under this rule the mid-July gap drafts (fedavg, DomIndOnto)
# would have read uncertified.
# ---------------------------------------------------------------------------


def test_neutral_plan_caps_would_be_verified():
    from delivery_label import NEUTRAL_PLAN_CAP_ID

    report = _report(("US-1", "pass"), ("CT-1", "pass"))
    # Control: the same inputs without the provenance fact stay verified.
    control = derive_delivery_label(
        report, _review(), paradigm="motion_planning/rl_collision_avoidance")
    assert control["label"] == LABEL_VERIFIED

    out = derive_delivery_label(
        report, _review(), paradigm="motion_planning/rl_collision_avoidance",
        neutral_build_plan=True)
    assert out["label"] == LABEL_UNCERTIFIED
    assert out["reasons"] == []
    cap = out["disclosures"][0]
    assert cap["id"] == NEUTRAL_PLAN_CAP_ID
    assert cap["source"] == "build_plan"
    assert "provisional build plan" in cap["message"]
    assert out["missing_probe_family"] == "motion_planning/rl_collision_avoidance"


def test_neutral_plan_caps_draft_and_keeps_reasons():
    from delivery_label import NEUTRAL_PLAN_CAP_ID

    # The mid-July sentence made executable: a demoting verdict on a
    # neutral-plan run reads uncertified, with the demoting reason still
    # recorded — nothing hidden.
    out = derive_delivery_label(
        _report(("UB-7", "flag_for_researcher")), None,
        neutral_build_plan=True)
    assert out["label"] == LABEL_UNCERTIFIED
    assert [r["id"] for r in out["reasons"]] == ["UB-7"]
    assert any(d["id"] == NEUTRAL_PLAN_CAP_ID for d in out["disclosures"])


def test_neutral_plan_cap_is_idempotent_and_leaves_low_labels_alone():
    from delivery_label import NEUTRAL_PLAN_CAP_ID, apply_neutral_plan_cap

    out = derive_delivery_label(
        _report(("US-1", "pass"), ("CT-1", "pass")), None,
        neutral_build_plan=True)
    # Re-application (the driver's post-demotion guard) adds nothing.
    again = apply_neutral_plan_cap(out, "motion_planning")
    assert sum(1 for d in again["disclosures"]
               if d["id"] == NEUTRAL_PLAN_CAP_ID) == 1

    # An already-uncertified delivery is left alone: label unchanged and the
    # recorded family is never overwritten.
    low = derive_delivery_label(_report(("US-1", "pass")), None,
                                paradigm="widget_family")
    assert low["label"] == LABEL_UNCERTIFIED
    apply_neutral_plan_cap(low, "other_family")
    assert low["label"] == LABEL_UNCERTIFIED
    assert low["missing_probe_family"] == "widget_family"

    # An explanation_only delivery dict is untouched.
    explanation = {"label": "explanation_only", "reasons": [],
                   "disclosures": []}
    assert apply_neutral_plan_cap(dict(explanation)) == explanation


def test_universal_floor_clause_reports_failures_under_cap():
    from delivery_label import universal_floor_clause

    # A capped delivery can carry genuine universal failures in `reasons`
    # while reading uncertified — the floor clause must never render "Every
    # universal check passed." over a failed check (the bev-distill
    # 2026-07-04 overclaim class, one layer deeper).
    capped = derive_delivery_label(
        _report(("US-3", "fail"), ("CT-1", "pass")), None,
        neutral_build_plan=True)
    assert capped["label"] == LABEL_UNCERTIFIED
    clause = universal_floor_clause(capped)
    assert "FAILED" in clause
    assert "Every universal check passed" not in clause

    # Unchanged shapes keep their exact sentences.
    clean = {"disclosures": [], "reasons": []}
    assert universal_floor_clause(clean) == "Every universal check passed."


def test_universal_floor_clause_derives_from_what_ran():
    """Maintainer-approved wording option (2026-07-04): the uncertified floor
    sentence may only claim a full pass when every universal check RAN.
    Universal-tier gaps are counted by the sanctioned id-prefix sets;
    contribution-tier unprobeables don't qualify the sentence (the label's
    other half already covers them)."""
    from delivery_label import universal_floor_clause

    clean = {"disclosures": [], "probe_counts": {"pass": 5}}
    assert universal_floor_clause(clean) == "Every universal check passed."

    contribution_gap_only = {"disclosures": [
        {"id": "KD-1", "verdict": "unprobeable", "message": "m"},
    ]}
    assert universal_floor_clause(contribution_gap_only) == (
        "Every universal check passed.")

    universal_gap = {"disclosures": [
        {"id": "UB-6", "verdict": "unprobeable",
         "message": "notebook has no executed outputs"},
        {"id": "KD-1", "verdict": "unprobeable", "message": "m"},
    ]}
    clause = universal_floor_clause(universal_gap)
    assert "could run passed" in clause
    assert "1 universal check could not run" in clause
    assert "not paperwork" in clause
