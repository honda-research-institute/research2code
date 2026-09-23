"""Schema tests for the paradigm-aware DataSetup validator on MethodSpec.

The 2026-05-26 pdwa run halted at stage 1 because schemas/method_spec.py's
DataSetup required four AL-specific integer fields (`initial_labeled`,
`batch_size`, `total_budget`, `num_rounds`) as non-Optional ints, even
though the motion_planning paradigm genuinely has no values for them.
The fix:
  - Relaxed the four fields to Optional[int] = None.
  - Added a MethodSpec-level model_validator that requires non-null for
    paradigms whose id starts with `active_learning/` (or is exactly
    `active_learning`).

These tests pin down both directions of the new behavior: non-AL
paradigms with nulls must validate; AL paradigms with nulls must fail.
"""

from __future__ import annotations

import copy
import json
import subprocess
import sys
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator
from pydantic import ValidationError

from schemas.method_spec import (
    SCHEMA_VERSION,
    HOMOGENEOUS_GRAPH_ALIGNMENT_PROBE_REF,
    HOMOGENEOUS_GRAPH_MECHANISM_SCHEMA_VERSION,
    VERIFICATION_PROBE_REFS_SCHEMA_VERSION,
    MethodSpec,
    methodology_contract_paper_values_for_param,
)
from scripts.validate_method_spec import (
    cross_check_paper_map,
    cross_check_promise_contract_consistency,
    cross_check_verification_probe_refs,
    methodology_core_detail_errors,
)


ROOT = Path(__file__).resolve().parents[1]


def _minimal_valid_spec(*, paradigm_id: str) -> dict:
    """A spec that satisfies ALL non-DataSetup schema requirements.

    Built explicitly (not loaded from r2c_runs/) so the tests don't depend
    on run artifacts that may be wiped between runs.
    """
    return {
        "schema_version": "1.2.0",
        "paper": {
            "title": "Test Paper",
            "authors": "Test Author",
            "repo_url": None,
        },
        "core_method": {
            "name": "Test method",
            "summary": "A test method.",
            "type": "algorithm",
            "paper_sections": ["Section 1"],
            "key_elements": ["alg-test"],
        },
        "paper_claims": {
            "method_description": "Test method description.",
            "claimed_results": "Test results.",
            "benchmark_scale": "Test scale.",
        },
        "try_it_out": {
            "definition": "Test definition.",
            "user_provides": [
                {"name": "Input", "description": "Test input.", "type": "data"},
            ],
            "system_provides": [
                {"name": "Output", "description": "Test output.", "type": "code"},
            ],
        },
        "data_requirements": {
            "format": "Test format.",
            "example_datasets": ["Test"],
            "auto_download_feasible": False,
            "auto_download_details": None,
            "synthetic_feasible": True,
            "synthetic_description": "Test synthetic.",
        },
        "dependencies": {
            "frameworks": ["numpy"],
            "pretrained_models": [],
            "compute": "CPU sufficient",
            "estimated_time": "< 60s",
        },
        "repo": {
            "url": None,
            "cloned": False,
            "clone_path": None,
            "key_files": [],
        },
        "comparison": {
            "classification": {
                "id": paradigm_id,
                "detection_reasoning": "Test paradigm detection.",
            },
            "description": "Test comparison.",
            "pluggable_component": {
                "name": "do_thing",
                "signature": "do_thing(arg, seed) -> Result",
                "seed_param": "seed",
                "description": "Test pluggable component.",
            },
            "controlled_variables": {"env": "fixed"},
            "independent_variable": "method",
            "evaluation_checkpoints": "per-step",
            "primary_metric": "accuracy",
            "visualization": "plot",
        },
        "critical_requirements": {
            "model": {
                "architecture": "Test architecture",
                "paper_section": "Section 1",
                "specific_features": [],
            },
            "training": {
                "optimizer": "Adam",
                "learning_rate": "0.001",
                "protocol": "test",
                "mc_samples": 0,
                "num_epochs": None,
                "seed": None,
                "paper_section": "Section 1",
            },
            "data_setup": {
                "initial_labeled": None,
                "batch_size": None,
                "total_budget": None,
                "num_rounds": None,
                "benchmark_name": "test benchmark",
                "special_protocol": None,
                "paper_section": "Section 1",
            },
            "blockers": [],
            "scale_dependent_hyperparameters": [],
            "required_model_methods": [],
        },
        "feasibility": "reproducible",
    }


def _al_spec_with_data_setup_ints() -> dict:
    spec = _minimal_valid_spec(
        paradigm_id="active_learning/batch_acquisition",
    )
    spec["critical_requirements"]["data_setup"].update({
        "initial_labeled": 100,
        "batch_size": 100,
        "total_budget": 35000,
        "num_rounds": 350,
    })
    return spec


def _methodology_element(
    *,
    element_id: str = "gradient-embedding-acquisition",
    role: str = "core_methodology",
    status: str = "must_replicate",
    concept: str = "Last-layer gradient embedding acquisition",
) -> dict:
    return {
        "element_id": element_id,
        "role": role,
        "replication_status": status,
        "paper_section": "Section 3, Algorithm 1",
        "paper_evidence": "The paper defines the acquisition function around this mechanism.",
        "technical_concept": concept,
        "required_behavior": (
            "Compute per-sample last-layer gradient embeddings and select a diverse batch."
        ),
        "demo_scale_implementation": (
            "Run the same gradient-embedding mechanism on the smoke-scale unlabeled pool."
        ),
        "acceptable_approximations": (
            ["Use a smaller unlabeled pool while preserving the same acquisition computation."]
            if status == "faithful_approximation_allowed"
            else []
        ),
        "forbidden_substitutions": [
            "Do not replace gradient embeddings with entropy or softmax-probability scores."
        ],
        "required_controls": ["Use the same model snapshot for all acquisition scores."],
        "fairness_checks": ["Only the acquisition logic differs from the surrounding harness."],
        "feasibility_rationale": (
            "The algorithm is fully specified in the paper and can run at demo scale."
            if status != "not_replicable"
            else "The method depends on an unavailable proprietary training corpus."
        ),
        "verification_expectations": [
            "method.py calls model.forward_with_embedding before batch selection."
        ],
        "verification_probe_refs": [],
        "blockers": (
            ["Unavailable proprietary training corpus."]
            if status == "not_replicable"
            else []
        ),
    }


def _add_methodology_contract(
    spec: dict,
    *,
    elements: list[dict],
    verdict_override: str | None = None,
) -> dict:
    spec = copy.deepcopy(spec)
    core_ids = [
        element["element_id"]
        for element in elements
        if element["role"] == "core_methodology"
    ]
    blockers = [
        {
            "element_id": element["element_id"],
            "technical_concept": element["technical_concept"],
            "reason": element["feasibility_rationale"],
        }
        for element in elements
        if (
            element["role"] == "core_methodology"
            and element["replication_status"] == "not_replicable"
        )
    ]
    approximations = [
        {
            "element_id": element["element_id"],
            "technical_concept": element["technical_concept"],
            "rationale": element["feasibility_rationale"],
        }
        for element in elements
        if element["replication_status"] == "faithful_approximation_allowed"
    ]
    if blockers:
        verdict = "not_replicable"
    elif approximations:
        verdict = "feasible_with_approved_approximations"
    else:
        verdict = "feasible"

    spec["methodology_replication_contract"] = {
        "schema_version": "1.0",
        "elements": elements,
    }
    spec["methodology_contract_pack"] = {
        "schema_version": "1.0",
        "summary": "Demo-scale implementation must preserve the core acquisition mechanism.",
        "core_methodology_element_ids": core_ids,
        "implementation_obligations": [
            element["required_behavior"] for element in elements
        ],
        "approved_approximations": [
            item
            for element in elements
            for item in element["acceptable_approximations"]
        ],
        "forbidden_substitutions": [
            item
            for element in elements
            for item in element["forbidden_substitutions"]
        ],
        "required_controls": [
            item
            for element in elements
            for item in element["required_controls"]
        ],
        "verification_expectations": [
            item
            for element in elements
            for item in element["verification_expectations"]
        ],
    }
    spec["replication_feasibility"] = {
        "schema_version": "1.0",
        "verdict": verdict_override or verdict,
        "blockers": blockers,
        "approved_approximations": approximations,
    }
    return spec


def _homogeneous_graph_elements_and_mechanism() -> tuple[list[dict], dict]:
    alignment = _methodology_element(
        element_id="graph-relational-alignment",
        role="supporting_mechanism",
        concept="Stable entity and graph index alignment",
    )
    construction = _methodology_element(
        element_id="graph-similarity-construction",
        role="supporting_mechanism",
        concept="Cosine-similarity graph construction",
    )
    message_passing = _methodology_element(
        element_id="graph-neighbor-message-passing",
        concept="Neighbor-aware graph message passing",
    )
    ablation = _methodology_element(
        element_id="graph-free-decoder-control",
        role="evaluation_control",
        concept="Paper-justified non-graph decoder control",
    )

    for index, element in enumerate((alignment, construction, message_passing)):
        element["paper_element_ids"] = [f"paper-graph-{index}"]
        element["relational_structure"] = {"kind": "homogeneous_graph"}
    alignment["verification_probe_refs"] = [
        HOMOGENEOUS_GRAPH_ALIGNMENT_PROBE_REF,
    ]
    construction["verification_probe_refs"] = [
        "graph_mechanism.parameter_agreement",
        "graph_mechanism.construction_semantics",
    ]
    message_passing["verification_probe_refs"] = [
        "graph_mechanism.topology_sensitivity",
        "graph_mechanism.neighbor_sensitivity",
        "graph_mechanism.permutation_equivalence",
    ]
    ablation["verification_probe_refs"] = [
        "graph_mechanism.contribution_ablation",
    ]

    mechanism = {
        "schema_version": "1.0",
        "alignment_element_id": alignment["element_id"],
        "construction": {
            "element_id": construction["element_id"],
            "callable": {
                "module": "method.model",
                "qualname": "build_article_graph",
            },
            "feature_input_root": "batch.static_features",
            "feature_parameter": "static_features",
            "output_selector": {"kind": "tuple_item", "index": 0},
            "threshold": {
                "metric": "cosine_similarity",
                "comparison": "greater_than_or_equal",
                "parameter": {
                    "params_name": "similarity_threshold",
                    "callable_parameter": "similarity_threshold",
                },
            },
            "cap": {"kind": "none"},
            "self_loop_policy": "required",
            "direction_policy": "undirected_bidirectional",
        },
        "message_passing": {
            "element_id": message_passing["element_id"],
            "callable": {
                "module": "method.model",
                "qualname": "execute_graph_message_passing",
            },
            "graph_parameter": "edge_index",
            "neighbor_signal_root": "batch.demand",
            "neighbor_signal_parameter": "demand_history",
            "output_root": "outputs.graph_embeddings",
        },
        "permutation_applicability": "equivariant",
        "contribution_ablation": {
            "kind": "non_graph_decoder",
            "element_id": ablation["element_id"],
            "discriminating_probe_ref": (
                "graph_mechanism.neighbor_sensitivity"
            ),
            "callable": {
                "module": "method.model",
                "qualname": "execute_non_graph_decoder",
            },
            "graph_parameter": "edge_index",
            "neighbor_signal_parameter": "demand_history",
            "output_root": "outputs.graph_embeddings",
        },
        "probe_refs": {
            "parameter_agreement": "graph_mechanism.parameter_agreement",
            "construction": "graph_mechanism.construction_semantics",
            "topology": "graph_mechanism.topology_sensitivity",
            "neighbor_signal": "graph_mechanism.neighbor_sensitivity",
            "permutation": "graph_mechanism.permutation_equivalence",
            "contribution_ablation": "graph_mechanism.contribution_ablation",
        },
    }
    return [alignment, construction, message_passing, ablation], mechanism


def _homogeneous_graph_spec() -> dict:
    spec = _minimal_valid_spec(paradigm_id="time_series_forecasting/global")
    spec["schema_version"] = SCHEMA_VERSION
    elements, mechanism = _homogeneous_graph_elements_and_mechanism()
    spec = _add_methodology_contract(spec, elements=elements)
    spec["methodology_replication_contract"][
        "homogeneous_graph_mechanism"
    ] = mechanism
    return spec


def test_non_al_paradigm_accepts_null_data_setup_ints():
    """motion_planning spec with null AL-specific ints must validate.

    Pins down the pdwa 2026-05-26 fix: before the change, this same spec
    halted stage 1 with 4 'int_type' validation errors.
    """
    spec = _minimal_valid_spec(
        paradigm_id="motion_planning",
    )
    # All four AL-specific fields are null — pdwa's actual shape.
    validated = MethodSpec.model_validate(spec)
    ds = validated.critical_requirements.data_setup
    assert ds.initial_labeled is None
    assert ds.batch_size is None
    assert ds.total_budget is None
    assert ds.num_rounds is None


def test_al_paradigm_accepts_populated_data_setup_ints():
    """AL spec with all four ints populated must validate (the happy path —
    pins down that the relaxation didn't break valid AL specs)."""
    spec = _al_spec_with_data_setup_ints()
    validated = MethodSpec.model_validate(spec)
    ds = validated.critical_requirements.data_setup
    assert ds.initial_labeled == 100
    assert ds.batch_size == 100
    assert ds.total_budget == 35000
    assert ds.num_rounds == 350


def test_data_setup_accepts_batch_returns_for_two_stage_selector():
    """Two-stage selectors carry the paper's preselection count b in
    data_setup.batch_returns (GBALD: rank b=300, select b'=100)."""
    spec = _al_spec_with_data_setup_ints()
    spec["critical_requirements"]["data_setup"]["batch_returns"] = 300
    validated = MethodSpec.model_validate(spec)
    assert validated.critical_requirements.data_setup.batch_returns == 300


def test_data_setup_batch_returns_optional_for_single_stage_selector():
    """batch_returns is optional: a single-stage AL selector (no preselection,
    e.g. BADGE) leaves it null and the four-field requirement is unaffected."""
    spec = _al_spec_with_data_setup_ints()  # no batch_returns set
    validated = MethodSpec.model_validate(spec)
    assert validated.critical_requirements.data_setup.batch_returns is None


@pytest.mark.parametrize("nulled_field", [
    "initial_labeled", "batch_size", "total_budget", "num_rounds",
])
def test_al_paradigm_rejects_null_data_setup_int(nulled_field):
    """For an AL paradigm, every one of the four ints being null is
    individually a validation error (each field is independently required)."""
    spec = _al_spec_with_data_setup_ints()
    spec["critical_requirements"]["data_setup"][nulled_field] = None
    with pytest.raises(ValidationError) as exc_info:
        MethodSpec.model_validate(spec)
    msgs = [err.get("msg", "") for err in exc_info.value.errors()]
    matched = [m for m in msgs if "Active-learning paradigm" in m
               and nulled_field in m]
    assert matched, (
        f"Expected the AL-paradigm-aware validator to flag {nulled_field!r}; "
        f"got: {msgs}"
    )


def test_al_paradigm_rejects_all_nulls():
    """When all four AL ints are null on an AL spec, the validator names
    all four in a single error (callers don't have to re-validate four
    times to learn the full set of missing fields)."""
    spec = _minimal_valid_spec(
        paradigm_id="active_learning/batch_acquisition",
    )
    # Leave all four at their default null state.
    with pytest.raises(ValidationError) as exc_info:
        MethodSpec.model_validate(spec)
    msgs = [err.get("msg", "") for err in exc_info.value.errors()]
    matched = [m for m in msgs if "Active-learning paradigm" in m]
    assert matched, f"Expected validator error message; got: {msgs}"
    # All four field names appear in the single error message.
    for name in ("initial_labeled", "batch_size", "total_budget", "num_rounds"):
        assert name in matched[0], (
            f"Expected {name!r} to be named in the validator error; got: {matched[0]}"
        )


def test_al_parent_paradigm_also_gated():
    """The validator gates on `active_learning` exactly as well as
    `active_learning/<sub>`. Pins down that a spec written against the
    parent paradigm (rare but allowed) still enforces the four ints."""
    spec = _minimal_valid_spec(
        paradigm_id="active_learning",
    )
    with pytest.raises(ValidationError) as exc_info:
        MethodSpec.model_validate(spec)
    msgs = [err.get("msg", "") for err in exc_info.value.errors()]
    assert any("Active-learning paradigm" in m for m in msgs), (
        f"Expected validator to fire on parent active_learning paradigm; got: {msgs}"
    )


def test_paradigm_id_substring_does_not_falsely_match():
    """Defensive: a future paradigm id that happens to contain
    `active_learning` as a substring (but not as a path prefix) is NOT
    treated as AL. Example: `meta_learning/active_learning_inspired`
    is hypothetical but the matcher uses startswith, not `in`."""
    spec = _minimal_valid_spec(
        paradigm_id="meta_learning/active_learning_inspired",
    )
    # Null ints should validate — this paradigm id doesn't start with
    # `active_learning/`.
    MethodSpec.model_validate(spec)


# ---------------------------------------------------------------------------
# mc_samples: Optional[int] (sweep finding 2026-05-26, same shape as DataSetup
# ints — MC-dropout is a bayesian-AL concept, non-MC-dropout paradigms should
# write null, not the prior "0 if not applicable" workaround).
# ---------------------------------------------------------------------------


def test_training_strings_accept_null_for_non_training_paradigm():
    """motion_planning spec with null optimizer/learning_rate/protocol must
    validate (pdwa 2026-07-28: the analyzer honestly wrote null for a paper
    with no training phase and the then-required str halted a paper that
    had delivered the day before on invented 'n/a' prose)."""
    spec = _minimal_valid_spec(paradigm_id="motion_planning")
    training = spec["critical_requirements"]["training"]
    training["optimizer"] = None
    training["learning_rate"] = None
    training["protocol"] = None
    validated = MethodSpec.model_validate(spec)
    assert validated.critical_requirements.training.optimizer is None
    assert validated.critical_requirements.training.learning_rate is None
    assert validated.critical_requirements.training.protocol is None


def test_training_strings_still_accept_prose_for_training_paradigm():
    """The happy path stays: a training paradigm's populated strings
    validate exactly as before the fields became optional."""
    spec = _minimal_valid_spec(paradigm_id="knowledge_distillation")
    training = spec["critical_requirements"]["training"]
    training["optimizer"] = "SGD with momentum 0.9"
    training["learning_rate"] = "0.02, decayed 10x at epochs 16 and 22"
    training["protocol"] = "retrain from scratch"
    validated = MethodSpec.model_validate(spec)
    assert validated.critical_requirements.training.optimizer.startswith("SGD")


def test_mc_samples_accepts_null_for_non_mc_paradigm():
    """motion_planning spec with null mc_samples must validate (pdwa's case
    pre-fix had mc_samples=0 as a workaround; null is now the right value)."""
    spec = _minimal_valid_spec(
        paradigm_id="motion_planning",
    )
    spec["critical_requirements"]["training"]["mc_samples"] = None
    validated = MethodSpec.model_validate(spec)
    assert validated.critical_requirements.training.mc_samples is None


def test_mc_samples_accepts_int_for_bayesian_paradigm():
    """bayesian-AL spec with mc_samples populated must validate (the happy
    path — pins down that relaxing the field didn't break valid AL specs)."""
    spec = _al_spec_with_data_setup_ints()
    spec["comparison"]["classification"]["id"] = "active_learning/bayesian"
    spec["critical_requirements"]["training"]["mc_samples"] = 100
    validated = MethodSpec.model_validate(spec)
    assert validated.critical_requirements.training.mc_samples == 100


def test_mc_samples_zero_workaround_still_validates():
    """Backwards compat: prior method_specs that wrote `mc_samples: 0` as a
    'not applicable' workaround should still parse cleanly (no schema
    regression). The relaxation is purely additive — `0` and `null` are
    both legal now."""
    spec = _minimal_valid_spec(
        paradigm_id="motion_planning",
    )
    spec["critical_requirements"]["training"]["mc_samples"] = 0
    validated = MethodSpec.model_validate(spec)
    assert validated.critical_requirements.training.mc_samples == 0


# ---------------------------------------------------------------------------
# Methodology replication contract (main/v3 integration Step 1)
# ---------------------------------------------------------------------------


def test_methodology_contract_is_optional_for_migration():
    spec = _minimal_valid_spec(
        paradigm_id="motion_planning",
    )
    validated = MethodSpec.model_validate(spec)
    assert validated.methodology_replication_contract is None
    assert validated.replication_feasibility is None


def test_badge_like_core_gradient_embedding_contract_validates_and_derives_feasible():
    spec = _al_spec_with_data_setup_ints()
    spec = _add_methodology_contract(spec, elements=[_methodology_element()])

    validated = MethodSpec.model_validate(spec)
    assert validated.methodology_replication_contract is not None
    element = validated.methodology_replication_contract.elements[0]
    assert element.role.value == "core_methodology"
    assert element.replication_status.value == "must_replicate"
    assert "gradient embeddings" in element.required_behavior

    derived = validated.derive_replication_feasibility()
    assert derived is not None
    assert derived.verdict.value == "feasible"
    assert derived.blockers == []


def test_methodology_contract_approved_approximation_derives_run_verdict():
    spec = _al_spec_with_data_setup_ints()
    spec = _add_methodology_contract(
        spec,
        elements=[_methodology_element(status="faithful_approximation_allowed")],
    )

    validated = MethodSpec.model_validate(spec)
    derived = validated.derive_replication_feasibility()
    assert derived is not None
    assert derived.verdict.value == "feasible_with_approved_approximations"
    assert derived.approved_approximations[0].element_id == "gradient-embedding-acquisition"


def test_methodology_contract_core_not_replicable_derives_run_blocker():
    spec = _al_spec_with_data_setup_ints()
    spec = _add_methodology_contract(
        spec,
        elements=[_methodology_element(status="not_replicable")],
    )

    validated = MethodSpec.model_validate(spec)
    derived = validated.derive_replication_feasibility()
    assert derived is not None
    assert derived.verdict.value == "not_replicable"
    assert derived.blockers[0].element_id == "gradient-embedding-acquisition"


def test_methodology_contract_fields_must_be_provided_together():
    spec = _al_spec_with_data_setup_ints()
    spec["methodology_replication_contract"] = {
        "schema_version": "1.0",
        "elements": [_methodology_element()],
    }

    with pytest.raises(ValidationError) as exc_info:
        MethodSpec.model_validate(spec)
    assert "methodology fidelity fields must be provided together" in str(exc_info.value)


def test_methodology_contract_rejects_declared_verdict_drift():
    spec = _al_spec_with_data_setup_ints()
    spec = _add_methodology_contract(
        spec,
        elements=[_methodology_element(status="not_replicable")],
        verdict_override="feasible",
    )

    with pytest.raises(ValidationError) as exc_info:
        MethodSpec.model_validate(spec)
    assert "replication_feasibility.verdict must match" in str(exc_info.value)


def test_methodology_contract_rejects_approximations_on_must_replicate_element():
    """Approved approximations are keyed by element replication_status.

    Regression for GBALD Stage 1: a non-core evaluation-control element kept
    status=must_replicate while its approximation also appeared in
    replication_feasibility.approved_approximations, causing a second validator
    failure after the judge's narrow fix.
    """
    spec = _al_spec_with_data_setup_ints()
    element = _methodology_element(
        element_id="batch-selection-protocol",
        role="evaluation_control",
        status="must_replicate",
        concept="Batch returns/outputs ratio",
    )
    element["acceptable_approximations"] = [
        "Smaller b and b_prime values acceptable for demo scale."
    ]
    spec = _add_methodology_contract(
        spec,
        elements=[_methodology_element(), element],
    )

    with pytest.raises(ValidationError) as exc_info:
        MethodSpec.model_validate(spec)
    msg = str(exc_info.value)
    assert "batch-selection-protocol" in msg
    assert "declares acceptable_approximations" in msg
    assert "faithful_approximation_allowed" in msg


def test_methodology_contract_rejects_demo_mc_samples_as_paper_truth():
    """training.mc_samples is paper truth, not the demo/signature default.

    Regression for GBALD Stage 2.x: the analyzer wrote
    critical_requirements.training.mc_samples=100 while the methodology
    contract correctly said the paper uses 2000 and approves reducing it to
    100 for demos. Stage 2.x then propagated 100 as params.mc_samples.paper_value.
    """
    spec = _al_spec_with_data_setup_ints()
    spec["comparison"]["classification"]["id"] = "active_learning/bayesian"
    spec["critical_requirements"]["training"]["mc_samples"] = 100
    mc_element = _methodology_element(
        element_id="mc_sample_count",
        role="supporting_mechanism",
        status="faithful_approximation_allowed",
        concept="MC dropout sample count",
    )
    mc_element["paper_evidence"] = "Paper uses 2000 MC dropout samples."
    mc_element["required_behavior"] = (
        "Use sufficient MC samples for posterior approximation."
    )
    mc_element["acceptable_approximations"] = [
        "Reducing mc_samples from 2000 to 100 for demo runs."
    ]
    spec = _add_methodology_contract(
        spec,
        elements=[_methodology_element(), mc_element],
    )

    with pytest.raises(ValidationError) as exc_info:
        MethodSpec.model_validate(spec)
    msg = str(exc_info.value)
    assert "critical_requirements.training.mc_samples" in msg
    assert "paper value 2000" in msg
    assert "is 100" in msg
    assert "demo-scale defaults" in msg


def test_methodology_contract_accepts_paper_mc_samples_with_demo_approximation():
    spec = _al_spec_with_data_setup_ints()
    spec["comparison"]["classification"]["id"] = "active_learning/bayesian"
    spec["critical_requirements"]["training"]["mc_samples"] = 2000
    mc_element = _methodology_element(
        element_id="mc_sample_count",
        role="supporting_mechanism",
        status="faithful_approximation_allowed",
        concept="MC dropout sample count",
    )
    mc_element["paper_evidence"] = "Paper uses 2000 MC dropout samples."
    mc_element["acceptable_approximations"] = [
        "Reducing mc_samples from 2000 to 100 for demo runs."
    ]
    spec = _add_methodology_contract(
        spec,
        elements=[_methodology_element(), mc_element],
    )

    validated = MethodSpec.model_validate(spec)
    assert validated.critical_requirements.training.mc_samples == 2000


def test_methodology_contract_extracts_parameter_equals_value_paper_truth():
    contract = {
        "elements": [
            {
                "paper_evidence": "The paper sets eta=0.9 for the geometric probability update.",
                "acceptable_approximations": [
                    "Use eta=0.9 at demo scale; this parameter is not rescaled."
                ],
            }
        ]
    }

    values = methodology_contract_paper_values_for_param(contract, "eta")
    assert values == [0.9]


def test_methodology_contract_extracts_symbol_alias_and_spaced_scientific_value():
    contract = {
        "elements": [
            {
                "paper_evidence": (
                    "The paper settings are $R_0 = 2.0e + 3$ and $\\eta = 0.9$."
                ),
                "acceptable_approximations": [],
            }
        ]
    }

    assert methodology_contract_paper_values_for_param(contract, "R_0") == [2000]
    assert methodology_contract_paper_values_for_param(contract, "eta") == [0.9]


def test_methodology_contract_ignores_bare_demo_assignment_as_paper_truth():
    """A bare ``param=N`` in a demo-scale field is the DEMO value, not paper truth.

    Regression for GBALD Stage 1: the analyzer wrote
    demo_scale_implementation="Use mc_samples=20 (demo) vs paper 2000". The
    extractor matched the bare assignment and treated 20 as the paper value,
    then rejected the correct structured mc_samples=2000 and halted the run on
    a false positive. The demo assignment must not be extracted; the genuine
    paper value, phrased without an explicit "paper uses N" cue here, is simply
    not extracted, so the cross-check stays silent rather than false-rejecting.
    """
    contract = {
        "elements": [
            {
                "demo_scale_implementation": (
                    "Use mc_samples=20 (demo) vs paper 2000. The geometric "
                    "ranking works identically at any scale."
                ),
                "acceptable_approximations": [],
            }
        ]
    }
    assert methodology_contract_paper_values_for_param(contract, "mc_samples") == []


def test_methodology_contract_extracts_explicit_paper_value_from_demo_field():
    """The fix only suppresses the bare-assignment form in demo fields. An
    explicitly-named paper value in a demo field ("from 2000 to 20", "paper
    uses 2000") is still extracted, so a genuinely-wrong structured value is
    still caught.
    """
    contract = {
        "elements": [
            {
                "demo_scale_implementation": (
                    "Reduce mc_samples from 2000 to 20 for the demo."
                ),
                "acceptable_approximations": [
                    "Paper uses 2000 mc samples; the demo uses fewer."
                ],
            }
        ]
    }
    assert methodology_contract_paper_values_for_param(contract, "mc_samples") == [2000]


def test_methodology_contract_rejects_core_element_without_verification_controls():
    spec = _al_spec_with_data_setup_ints()
    element = _methodology_element()
    element["verification_expectations"] = []
    spec = _add_methodology_contract(spec, elements=[element])

    with pytest.raises(ValidationError) as exc_info:
        MethodSpec.model_validate(spec)
    assert "requires non-empty verification_expectations" in str(exc_info.value)


def test_check_feasibility_halts_on_not_replicable_contract(tmp_path: Path):
    spec = _al_spec_with_data_setup_ints()
    spec = _add_methodology_contract(
        spec,
        elements=[_methodology_element(status="not_replicable")],
    )
    spec_path = tmp_path / "method_spec.json"
    output_path = tmp_path / "feasibility_gate.json"
    spec_path.write_text(json.dumps(spec, indent=2), encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "check_feasibility.py"),
            str(spec_path),
            "--output",
            str(output_path),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0
    halt_path = output_path.with_suffix(output_path.suffix + ".halt")
    assert halt_path.exists()
    halt = json.loads(halt_path.read_text(encoding="utf-8"))
    assert halt["status"] == "halted"
    assert halt["replication_feasibility_at_spec"] == "not_replicable"
    assert halt["not_replicable_core_methodology"][0]["element_id"] == (
        "gradient-embedding-acquisition"
    )


def test_check_feasibility_records_replication_verdict_on_pass(tmp_path: Path):
    spec = _al_spec_with_data_setup_ints()
    spec = _add_methodology_contract(
        spec,
        elements=[_methodology_element(status="faithful_approximation_allowed")],
    )
    spec_path = tmp_path / "method_spec.json"
    output_path = tmp_path / "feasibility_gate.json"
    spec_path.write_text(json.dumps(spec, indent=2), encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "check_feasibility.py"),
            str(spec_path),
            "--output",
            str(output_path),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0
    assert not output_path.with_suffix(output_path.suffix + ".halt").exists()
    summary = json.loads(output_path.read_text(encoding="utf-8"))
    assert summary["status"] == "passed"
    assert (
        summary["replication_feasibility_at_spec"]
        == "feasible_with_approved_approximations"
    )


def test_validate_method_spec_default_allows_missing_methodology_contract(
    tmp_path: Path,
):
    spec = _al_spec_with_data_setup_ints()
    spec_path = tmp_path / "method_spec.json"
    spec_path.write_text(json.dumps(spec, indent=2), encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "validate_method_spec.py"),
            str(spec_path),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0
    assert "validates against schema" in result.stdout


def test_validate_method_spec_can_require_methodology_contract(tmp_path: Path):
    spec = _al_spec_with_data_setup_ints()
    spec_path = tmp_path / "method_spec.json"
    spec_path.write_text(json.dumps(spec, indent=2), encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "validate_method_spec.py"),
            str(spec_path),
            "--require-methodology-contract",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 2
    assert "methodology-contract requirement failed" in result.stderr
    assert "methodology_replication_contract" in result.stderr
    assert "methodology_contract_pack" in result.stderr
    assert "replication_feasibility" in result.stderr


def test_validate_method_spec_require_methodology_contract_passes_when_present(
    tmp_path: Path,
):
    spec = _al_spec_with_data_setup_ints()
    spec = _add_methodology_contract(spec, elements=[_methodology_element()])
    spec_path = tmp_path / "method_spec.json"
    spec_path.write_text(json.dumps(spec, indent=2), encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "validate_method_spec.py"),
            str(spec_path),
            "--require-methodology-contract",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0
    assert "methodology-contract requirement passed" in result.stdout


def test_methodology_core_detail_errors_reports_all_missing_support_fields():
    spec = _al_spec_with_data_setup_ints()
    element_0 = _methodology_element(element_id="geometric-prior-model")
    element_0["required_controls"] = []
    element_1 = _methodology_element(element_id="ellipsoid-geodesic-rescaling")
    element_1["fairness_checks"] = []
    element_2 = _methodology_element(element_id="geometric-ranking")
    element_2["required_controls"] = []
    element_2["fairness_checks"] = []
    spec = _add_methodology_contract(
        spec,
        elements=[element_0, element_1, element_2],
    )

    errors = methodology_core_detail_errors(spec)

    assert errors == [
        "methodology_replication_contract.elements.0.required_controls: "
        "core methodology element 'geometric-prior-model' requires non-empty "
        "required_controls",
        "methodology_replication_contract.elements.1.fairness_checks: "
        "core methodology element 'ellipsoid-geodesic-rescaling' requires "
        "non-empty fairness_checks",
        "methodology_replication_contract.elements.2.required_controls: "
        "core methodology element 'geometric-ranking' requires non-empty "
        "required_controls",
        "methodology_replication_contract.elements.2.fairness_checks: "
        "core methodology element 'geometric-ranking' requires non-empty "
        "fairness_checks",
    ]


def test_validate_method_spec_reports_all_core_detail_gaps_before_pydantic_mask(
    tmp_path: Path,
):
    spec = _al_spec_with_data_setup_ints()
    element_0 = _methodology_element(element_id="geometric-prior-model")
    element_0["required_controls"] = []
    element_1 = _methodology_element(element_id="ellipsoid-geodesic-rescaling")
    element_1["fairness_checks"] = []
    element_2 = _methodology_element(element_id="geometric-ranking")
    element_2["required_controls"] = []
    element_2["fairness_checks"] = []
    spec = _add_methodology_contract(
        spec,
        elements=[element_0, element_1, element_2],
    )
    spec_path = tmp_path / "method_spec.json"
    spec_path.write_text(json.dumps(spec, indent=2), encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "validate_method_spec.py"),
            str(spec_path),
            "--strict",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    assert "core detail completeness failed (4 missing field(s))" in result.stderr
    assert "elements.0.required_controls" in result.stderr
    assert "elements.1.fairness_checks" in result.stderr
    assert "elements.2.required_controls" in result.stderr
    assert "elements.2.fairness_checks" in result.stderr
    assert "schema validation failed" in result.stderr


# ---------------------------------------------------------------------------
# Naming bridge: the optional `symbol` on try_it_out provide entries
# (the naming bridge design note (internal, not shipped), schema v1.7.0). The
# field is the machine-checkable identifier a system-provides promise
# resolves to; Stage 2.b binds it to a public name in the generated package.
# ---------------------------------------------------------------------------


def test_provide_symbol_accepts_valid_identifier():
    spec = _minimal_valid_spec(paradigm_id="motion_planning")
    spec["try_it_out"]["system_provides"] = [
        {
            "name": "AC classifier (avoidance gating)",
            "description": "Gates candidate trajectories.",
            "type": "model",
            "symbol": "AvoidanceClassifier",
        },
    ]
    validated = MethodSpec.model_validate(spec)
    assert validated.try_it_out.system_provides[0].symbol == "AvoidanceClassifier"


def test_provide_symbol_is_optional_and_legacy_specs_stay_valid():
    """The field is additive: every existing spec (no symbol anywhere)
    validates unchanged, and entries without a symbol default to None —
    conservative by construction, no bridge enforcement without an explicit
    declaration."""
    spec = _minimal_valid_spec(paradigm_id="motion_planning")
    validated = MethodSpec.model_validate(spec)
    assert validated.try_it_out.system_provides[0].symbol is None
    assert validated.try_it_out.user_provides[0].symbol is None


def test_provide_symbol_rejects_leading_underscore():
    """A promised component is user-facing by construction — an
    underscore-private symbol is exactly the ICRA defect the bridge exists
    to catch, so the schema rejects it at Stage 1."""
    spec = _minimal_valid_spec(paradigm_id="motion_planning")
    spec["try_it_out"]["system_provides"] = [
        {
            "name": "AC classifier",
            "description": "x",
            "type": "model",
            "symbol": "_AvoidanceClassifier",
        },
    ]
    with pytest.raises(ValidationError, match="leading underscore"):
        MethodSpec.model_validate(spec)


@pytest.mark.parametrize("bad_symbol", [
    "AC classifier (avoidance gating)",  # prose, not an identifier
    "method.AvoidanceClassifier",        # dotted path, not one identifier
    "select-batch",                      # kebab-case
    "3Model",                            # leading digit
    "class",                             # Python keyword — not importable
    "",                                  # empty string
])
def test_provide_symbol_rejects_non_identifiers(bad_symbol):
    spec = _minimal_valid_spec(paradigm_id="motion_planning")
    spec["try_it_out"]["system_provides"] = [
        {"name": "Entry", "description": "x", "type": "code", "symbol": bad_symbol},
    ]
    with pytest.raises(ValidationError, match="not a valid Python identifier"):
        MethodSpec.model_validate(spec)

# ---------------------------------------------------------------------------
# Promise kind: the optional `symbol_kind` on declared symbols (schema
# v1.8.0, plan-of-record item 10). Two concrete cases shaped it: detr's KD
# roll (same-kind name mismatch, producer-fixable) and ADAM (class-shaped
# promises over a pure-function delivery, upstream). The kind is the
# machine-checkable HALF the name alone could not carry.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kind", ["class", "function"])
def test_provide_symbol_kind_accepts_declared_kinds(kind):
    spec = _minimal_valid_spec(paradigm_id="motion_planning")
    spec["try_it_out"]["system_provides"] = [
        {
            "name": "AC classifier (avoidance gating)",
            "description": "Gates candidate trajectories.",
            "type": "model",
            "symbol": "AvoidanceClassifier",
            "symbol_kind": kind,
        },
    ]
    validated = MethodSpec.model_validate(spec)
    assert validated.try_it_out.system_provides[0].symbol_kind == kind


def test_provide_symbol_kind_is_optional_and_legacy_specs_stay_valid():
    """Additive like `symbol` itself: a spec declaring a symbol with no kind
    validates unchanged and gets name-only bridge enforcement downstream."""
    spec = _minimal_valid_spec(paradigm_id="motion_planning")
    spec["try_it_out"]["system_provides"] = [
        {
            "name": "AC classifier",
            "description": "x",
            "type": "model",
            "symbol": "AvoidanceClassifier",
        },
    ]
    validated = MethodSpec.model_validate(spec)
    assert validated.try_it_out.system_provides[0].symbol_kind is None


def test_provide_symbol_kind_without_symbol_is_rejected():
    """The kind qualifies a declared importable name — a kind with no symbol
    has nothing to bind to, so the schema rejects it at Stage 1 rather than
    letting downstream gates silently ignore it."""
    spec = _minimal_valid_spec(paradigm_id="motion_planning")
    spec["try_it_out"]["system_provides"] = [
        {
            "name": "AC classifier",
            "description": "x",
            "type": "model",
            "symbol_kind": "class",
        },
    ]
    with pytest.raises(ValidationError, match="without a symbol"):
        MethodSpec.model_validate(spec)


def test_provide_symbol_kind_rejects_unknown_values():
    spec = _minimal_valid_spec(paradigm_id="motion_planning")
    spec["try_it_out"]["system_provides"] = [
        {
            "name": "AC classifier",
            "description": "x",
            "type": "model",
            "symbol": "AvoidanceClassifier",
            "symbol_kind": "module",
        },
    ]
    with pytest.raises(ValidationError):
        MethodSpec.model_validate(spec)


# ---------------------------------------------------------------------------
# Promise/contract consistency: the promise-SET counterpart of symbol_kind's
# per-symbol kind-awareness (overnight-0721-log.md fix proposal 5). The ADAM
# 2026-07-21 roll declared TWO function promises (`optimize`,
# `optimize_adamax`) while its own pluggable_component contract declared ONE
# `optimize()` entrypoint with a `variant: str = 'adam'` kwarg — a
# contradiction between two analyzer-authored surfaces, visible in the spec
# alone, that previously survived to the 2.d naming-bridge re-check after a
# full generation pass. cross_check_promise_contract_consistency fails it at
# Stage 1.
# ---------------------------------------------------------------------------


# Byte-exact from r2c_runs/ADAM/.pipeline/method_spec.json (overnight
# 2026-07-21/22 roll, the item-10 acceptance run), baked in as literals:
# r2c_runs/ is deletable and tests must never reference it.
_ADAM_OVERPROMISE_SYSTEM_PROVIDES = [
    {
        "name": "Adam optimizer implementation",
        "description": (
            "The core Adam algorithm (Algorithm 1) implemented as a callable "
            "function with the taxonomy pluggable signature, maintaining "
            "moment state across steps and returning an OptimizationResult."
        ),
        "type": "code",
        "symbol": "optimize",
        "symbol_kind": "function",
    },
    {
        "name": "AdaMax variant",
        "description": (
            "The AdaMax variant (Algorithm 2) using the L-infinity norm "
            "instead of L2 for the second moment estimate."
        ),
        "type": "code",
        "symbol": "optimize_adamax",
        "symbol_kind": "function",
    },
]

_ADAM_PLUGGABLE_COMPONENT = {
    "name": "optimize",
    "signature": (
        "optimize(objective, initial_params, data, seed, *, num_steps: int "
        "= 100, step_size: float = 0.001, beta1: float = 0.9, beta2: float "
        "= 0.999, epsilon: float = 1e-8, variant: str = 'adam') -> "
        "OptimizationResult"
    ),
    "seed_param": "seed",
    "description": (
        "The core Adam optimization function. Takes a differentiable "
        "objective, initial parameters, data for minibatch sampling, and "
        "hyperparameters. The variant parameter switches between 'adam' "
        "(Algorithm 1, L2 second moment) and 'ada_max' (Algorithm 2, "
        "L-infinity second moment)."
    ),
}


def _adam_overpromise_spec() -> dict:
    spec = _minimal_valid_spec(paradigm_id="motion_planning")
    spec["try_it_out"]["system_provides"] = copy.deepcopy(
        _ADAM_OVERPROMISE_SYSTEM_PROVIDES
    )
    spec["comparison"]["pluggable_component"] = copy.deepcopy(
        _ADAM_PLUGGABLE_COMPONENT
    )
    return spec


def test_promise_contract_consistency_fails_adam_overpromise():
    """The known-bad fixture: the second function promise name-extends the
    single declared entrypoint and the contract never declares it. One error,
    naming the owner (Stage 1 analyzer) and both resolutions the halt judge
    diagnosed (drop the promise, or declare the second entrypoint)."""
    spec = MethodSpec.model_validate(_adam_overpromise_spec())

    errors = cross_check_promise_contract_consistency(spec)

    assert len(errors) == 1
    message = errors[0]
    assert "'AdaMax variant'" in message
    assert "`optimize_adamax`" in message
    assert "`optimize`" in message
    assert "internally contradicts itself" in message
    assert "Stage 1 analyzer" in message
    # Both judge-diagnosed resolutions, in the analyzer's vocabulary:
    assert "drop the `optimize_adamax` system_provides entry" in message
    assert "second entrypoint" in message
    # The contract's own variant switch is named so the fix-mode analyzer
    # sees WHERE the contract already routes the variant.
    assert "`variant`" in message


def test_promise_contract_consistency_cli_fails_strict_with_exit_2(
    tmp_path: Path,
):
    """End-to-end through the Stage 1 validator CLI: --strict fails the ADAM
    fixture with the strict exit code (2) before any taxonomy or paper_map
    cross-check runs, and the stderr carries the analyzer-routing message the
    judge reads. Default mode stays structural-only (exit 0), matching how
    the pipeline scopes the gate."""
    spec_path = tmp_path / "method_spec.json"
    spec_path.write_text(
        json.dumps(_adam_overpromise_spec(), indent=2), encoding="utf-8"
    )

    strict = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "validate_method_spec.py"),
            str(spec_path),
            "--strict",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert strict.returncode == 2
    assert "promise/contract consistency cross-check failed" in strict.stderr
    assert "optimize_adamax" in strict.stderr
    assert "Stage 1 analyzer" in strict.stderr

    default = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "validate_method_spec.py"),
            str(spec_path),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert default.returncode == 0


def test_promise_contract_consistency_passes_own_stem_helpers():
    """The green-run shape (ROMAN25 2026-07-21): additional function promises
    with their own name stems beside the entrypoint are helper promises, not
    entrypoint variants — untouched."""
    spec = _minimal_valid_spec(paradigm_id="motion_planning")
    spec["comparison"]["pluggable_component"] = {
        "name": "plan",
        "signature": (
            "plan(start, goal, other_agents, static_objects, seed, "
            "v_max: float = 1.0) -> PlanResult"
        ),
        "seed_param": "seed",
        "description": "Plans a trajectory and returns a subgoal.",
    }
    spec["try_it_out"]["system_provides"] = [
        {
            "name": "Planner",
            "description": "The planner entrypoint.",
            "type": "code",
            "symbol": "plan",
            "symbol_kind": "function",
        },
        {
            "name": "Trajectory creation",
            "description": "Creates candidate trajectories.",
            "type": "code",
            "symbol": "create_candidate_trajectories",
            "symbol_kind": "function",
        },
        {
            "name": "Risk evaluator",
            "description": "Evaluates trajectories.",
            "type": "code",
            "symbol": "evaluate_trajectories",
            "symbol_kind": "function",
        },
    ]
    validated = MethodSpec.model_validate(spec)

    assert cross_check_promise_contract_consistency(validated) == []


def test_promise_contract_consistency_passes_declared_second_entrypoint():
    """A genuinely multi-entrypoint contract passes: when the component's own
    surface declares the sibling symbol, the promise is reconcilable — this
    is judge-diagnosed resolution (b) expressed in the spec."""
    spec = _adam_overpromise_spec()
    spec["comparison"]["pluggable_component"]["signature"] = (
        "optimize(objective, initial_params, data, seed, *, num_steps: int "
        "= 100) -> OptimizationResult; optimize_adamax(objective, "
        "initial_params, data, seed, *, num_steps: int = 100) -> "
        "OptimizationResult"
    )
    validated = MethodSpec.model_validate(spec)

    assert cross_check_promise_contract_consistency(validated) == []


def test_promise_contract_consistency_skips_placeholder_component_name():
    """A component whose name is not one importable identifier (placeholder
    prose) gets no enforcement — the same conservative tolerance the 2.b
    bridge grants invalid declared symbols. Closest realizable analog of "no
    pluggable component": the schema requires the block itself."""
    spec = _adam_overpromise_spec()
    spec["comparison"]["pluggable_component"]["name"] = (
        "the optimizer (see Algorithm 1)"
    )
    validated = MethodSpec.model_validate(spec)

    assert cross_check_promise_contract_consistency(validated) == []


def test_promise_contract_consistency_skips_legacy_specs_without_kind():
    """Legacy posture: promises without symbol_kind (and specs without
    symbols at all) get no set-consistency enforcement, exactly like the
    per-symbol kind checks downstream — additive, never retroactive."""
    spec = _adam_overpromise_spec()
    for entry in spec["try_it_out"]["system_provides"]:
        del entry["symbol_kind"]
    validated = MethodSpec.model_validate(spec)
    assert cross_check_promise_contract_consistency(validated) == []

    bare = MethodSpec.model_validate(
        _minimal_valid_spec(paradigm_id="motion_planning")
    )
    assert cross_check_promise_contract_consistency(bare) == []


def test_promise_contract_consistency_ignores_class_kind_extensions():
    """A class-kind promise never trips the set check — wrong-kind promises
    are the 2.b kind-awareness check's jurisdiction, and a class name that
    happens to extend the entrypoint stem is not an entrypoint variant."""
    spec = _adam_overpromise_spec()
    spec["try_it_out"]["system_provides"][1]["symbol_kind"] = "class"
    validated = MethodSpec.model_validate(spec)

    assert cross_check_promise_contract_consistency(validated) == []


# ---------------------------------------------------------------------------
# seed_param null semantics (DomIndOnto KBP 2026-07-29, stage 2c): a
# deterministic component has no seed parameter. The field folds the
# string sentinels of "no seed" into a real null, and raw-JSON consumers
# share the fold via normalized_seed_param.
# ---------------------------------------------------------------------------

def test_seed_param_none_string_folds_to_null():
    spec = _minimal_valid_spec(paradigm_id="motion_planning")
    spec["comparison"]["pluggable_component"]["seed_param"] = "None"
    parsed = MethodSpec.model_validate(spec)
    assert parsed.comparison.pluggable_component.seed_param is None


def test_seed_param_real_name_passes_through():
    spec = _minimal_valid_spec(paradigm_id="motion_planning")
    parsed = MethodSpec.model_validate(spec)
    assert parsed.comparison.pluggable_component.seed_param == "seed"


def test_seed_param_may_be_omitted_for_deterministic_components():
    spec = _minimal_valid_spec(paradigm_id="motion_planning")
    del spec["comparison"]["pluggable_component"]["seed_param"]
    parsed = MethodSpec.model_validate(spec)
    assert parsed.comparison.pluggable_component.seed_param is None


def test_normalized_seed_param_shared_fold():
    from schemas.method_spec import normalized_seed_param

    assert normalized_seed_param("None") is None
    assert normalized_seed_param("null") is None
    assert normalized_seed_param("") is None
    assert normalized_seed_param(None) is None
    assert normalized_seed_param("seed") == "seed"


# ---------------------------------------------------------------------------
# Typed calibration contexts (R2C-091)
# ---------------------------------------------------------------------------


def _scale_hyperparameter() -> dict:
    return {
        "name": "R_0",
        "paper_value": 2000.0,
        "formula": None,
        "description": "A threshold calibrated against the paper input.",
        "paper_section": "Section 3",
    }


@pytest.mark.parametrize(
    "scale",
    [
        "raw_pixel_unnormalized",
        "pixel_zero_one",
        "pixel_centered",
        "standardized",
        "unit_norm",
    ],
)
def test_schema_113_accepts_closed_feature_magnitude_contexts(scale):
    spec = _minimal_valid_spec(paradigm_id="motion_planning/sampling_based")
    spec["schema_version"] = "1.13.0"
    entry = _scale_hyperparameter()
    entry["calibration_context"] = {
        "kind": "feature_magnitude",
        "scale": scale,
    }
    spec["critical_requirements"]["scale_dependent_hyperparameters"] = [entry]

    validated = MethodSpec.model_validate(spec)
    context = validated.critical_requirements.scale_dependent_hyperparameters[
        0
    ].calibration_context

    assert context is not None
    assert context.kind == "feature_magnitude"
    assert context.scale == scale


@pytest.mark.parametrize(
    "context",
    [
        {
            "kind": "representation_convention",
            "convention": "target_box_grid",
        },
        {
            "kind": "other",
            "label": "raw sensor distance in meters",
            "reason": "No supported observer measures this physical-unit context.",
        },
    ],
)
def test_schema_113_accepts_representation_and_other_contexts(context):
    spec = _minimal_valid_spec(paradigm_id="motion_planning/sampling_based")
    spec["schema_version"] = SCHEMA_VERSION
    entry = _scale_hyperparameter()
    entry["calibration_context"] = context
    spec["critical_requirements"]["scale_dependent_hyperparameters"] = [entry]

    validated = MethodSpec.model_validate(spec)
    got = validated.critical_requirements.scale_dependent_hyperparameters[
        0
    ].calibration_context

    assert got is not None
    assert got.kind == context["kind"]


@pytest.mark.parametrize(
    "context",
    [
        {"kind": "feature_magnitude", "scale": "meters"},
        {
            "kind": "representation_convention",
            "convention": "normalized_boxes",
        },
        {"kind": "graph_statistic", "statistic": "degree"},
        {"kind": "protocol", "role": "forecast_call_horizon"},
        {"kind": "dataset", "name": "MNIST"},
        {"kind": "other", "label": " ", "reason": "Evidence-backed."},
        {"kind": "other", "label": "physical units", "reason": "\t"},
    ],
)
def test_typed_calibration_context_rejects_unsupported_grammar(context):
    spec = _minimal_valid_spec(paradigm_id="motion_planning/sampling_based")
    spec["schema_version"] = SCHEMA_VERSION
    entry = _scale_hyperparameter()
    entry["calibration_context"] = context
    spec["critical_requirements"]["scale_dependent_hyperparameters"] = [entry]

    with pytest.raises(ValidationError):
        MethodSpec.model_validate(spec)


def test_explicit_schema_113_requires_typed_calibration_context():
    spec = _minimal_valid_spec(paradigm_id="motion_planning/sampling_based")
    spec["schema_version"] = SCHEMA_VERSION
    entry = _scale_hyperparameter()
    entry["assumes_data_scale"] = "raw_pixel_unnormalized"
    spec["critical_requirements"]["scale_dependent_hyperparameters"] = [entry]

    with pytest.raises(ValidationError) as exc_info:
        MethodSpec.model_validate(spec)

    assert "explicit v1.13+ specs must emit typed calibration_context" in str(
        exc_info.value
    )


def test_explicit_schema_112_retains_legacy_scale_carrier():
    spec = _minimal_valid_spec(paradigm_id="motion_planning/sampling_based")
    spec["schema_version"] = "1.12.0"
    entry = _scale_hyperparameter()
    entry["assumes_data_scale"] = "raw_pixel_unnormalized"
    spec["critical_requirements"]["scale_dependent_hyperparameters"] = [entry]

    validated = MethodSpec.model_validate(spec)
    got = validated.critical_requirements.scale_dependent_hyperparameters[0]

    assert got.assumes_data_scale == "raw_pixel_unnormalized"
    assert got.calibration_context is None


def test_omitted_schema_version_retains_legacy_scale_carrier():
    spec = _minimal_valid_spec(paradigm_id="motion_planning/sampling_based")
    del spec["schema_version"]
    entry = _scale_hyperparameter()
    entry["assumes_data_scale"] = "raw_pixel_unnormalized"
    spec["critical_requirements"]["scale_dependent_hyperparameters"] = [entry]

    validated = MethodSpec.model_validate(spec)

    assert validated.schema_version == SCHEMA_VERSION
    assert "schema_version" not in validated.model_fields_set


@pytest.mark.parametrize("carrier_shape", ["mixed", "neither"])
def test_scale_hyperparameter_requires_exactly_one_context_carrier(carrier_shape):
    spec = _minimal_valid_spec(paradigm_id="motion_planning/sampling_based")
    spec["schema_version"] = SCHEMA_VERSION
    entry = _scale_hyperparameter()
    if carrier_shape == "mixed":
        entry["assumes_data_scale"] = "raw_pixel_unnormalized"
        entry["calibration_context"] = {
            "kind": "feature_magnitude",
            "scale": "raw_pixel_unnormalized",
        }
    spec["critical_requirements"]["scale_dependent_hyperparameters"] = [entry]

    with pytest.raises(ValidationError) as exc_info:
        MethodSpec.model_validate(spec)

    assert "provide exactly one" in str(exc_info.value)


def test_exported_calibration_context_is_a_strict_discriminated_union():
    schema = MethodSpec.model_json_schema()
    hyperparameter = schema["$defs"]["ScaleDependentHyperparam"]
    carrier = hyperparameter["properties"][
        "calibration_context"
    ]["anyOf"][0]

    assert carrier["discriminator"]["propertyName"] == "kind"
    assert set(carrier["discriminator"]["mapping"]) == {
        "feature_magnitude",
        "representation_convention",
        "other",
    }
    assert schema["$defs"]["FeatureMagnitudeCalibrationContext"][
        "additionalProperties"
    ] is False
    assert schema["$defs"]["RepresentationConventionCalibrationContext"][
        "additionalProperties"
    ] is False
    assert schema["$defs"]["OtherCalibrationContext"][
        "additionalProperties"
    ] is False
    assert hyperparameter["oneOf"] == [
        {
            "properties": {
                "assumes_data_scale": {"type": "string"},
                "calibration_context": {"type": "null"},
            },
            "required": ["assumes_data_scale"],
        },
        {
            "properties": {
                "assumes_data_scale": {"type": "null"},
                "calibration_context": {"not": {"type": "null"}},
            },
            "required": ["calibration_context"],
        },
    ]


def test_exported_schema_preserves_one_active_context_after_model_dump():
    spec = _minimal_valid_spec(paradigm_id="motion_planning/sampling_based")
    spec["schema_version"] = SCHEMA_VERSION
    entry = _scale_hyperparameter()
    entry["calibration_context"] = {
        "kind": "feature_magnitude",
        "scale": "raw_pixel_unnormalized",
    }
    spec["critical_requirements"]["scale_dependent_hyperparameters"] = [entry]

    payload = MethodSpec.model_validate(spec).model_dump(mode="json")
    validator = Draft202012Validator(MethodSpec.model_json_schema())

    assert list(validator.iter_errors(payload)) == []

    mixed = copy.deepcopy(payload)
    mixed["critical_requirements"]["scale_dependent_hyperparameters"][0][
        "assumes_data_scale"
    ] = "raw_pixel_unnormalized"
    assert list(validator.iter_errors(mixed))

    neither = copy.deepcopy(payload)
    neither["critical_requirements"]["scale_dependent_hyperparameters"][0][
        "calibration_context"
    ] = None
    assert list(validator.iter_errors(neither))


def test_portable_schema_enforces_current_typed_carrier_version_boundary():
    validator = Draft202012Validator(MethodSpec.model_json_schema())
    legacy = _minimal_valid_spec(
        paradigm_id="motion_planning/sampling_based",
    )
    entry = _scale_hyperparameter()
    entry["assumes_data_scale"] = "raw_pixel_unnormalized"
    legacy["critical_requirements"]["scale_dependent_hyperparameters"] = [entry]

    archived = copy.deepcopy(legacy)
    archived["schema_version"] = "1.12.0"
    assert list(validator.iter_errors(archived)) == []

    current = copy.deepcopy(legacy)
    current["schema_version"] = SCHEMA_VERSION
    errors = list(validator.iter_errors(current))
    assert errors
    assert any(
        "calibration_context" in "/".join(str(part) for part in error.path)
        or "calibration_context" in error.message
        for error in errors
    )


# ---------------------------------------------------------------------------
# Typed homogeneous-graph mechanism contract (R2C-088)
# ---------------------------------------------------------------------------


def test_current_homogeneous_graph_mechanism_round_trips_closed_v1_grammar():
    spec = _homogeneous_graph_spec()

    validated = MethodSpec.model_validate(spec)
    mechanism = (
        validated.methodology_replication_contract.homogeneous_graph_mechanism
    )
    assert mechanism is not None
    assert mechanism.alignment_element_id == "graph-relational-alignment"
    assert mechanism.construction.callable.qualname == "build_article_graph"
    assert mechanism.construction.output_selector.kind == "tuple_item"
    assert mechanism.construction.threshold.parameter.params_name == (
        "similarity_threshold"
    )
    assert mechanism.construction.cap.kind == "none"
    assert mechanism.message_passing.neighbor_signal_root == "batch.demand"
    assert mechanism.contribution_ablation.kind == "non_graph_decoder"
    assert mechanism.probe_refs.contribution_ablation == (
        "graph_mechanism.contribution_ablation"
    )

    portable_errors = list(
        Draft202012Validator(MethodSpec.model_json_schema()).iter_errors(spec)
    )
    assert portable_errors == []


def test_graph_mechanism_schema_references_r2c084_without_duplicating_identity():
    schema = MethodSpec.model_json_schema()
    properties = schema["$defs"]["HomogeneousGraphMechanismContract"][
        "properties"
    ]

    assert set(properties) == {
        "schema_version",
        "alignment_element_id",
        "construction",
        "message_passing",
        "permutation_applicability",
        "contribution_ablation",
        "probe_refs",
    }
    for identity_field in (
        "representation",
        "entity_axis",
        "graph_root",
        "phase_batch_modes",
        "degree_semantics",
        "output_order",
    ):
        assert identity_field not in properties


def test_current_homogeneous_graph_marker_requires_mechanism_block():
    spec = _homogeneous_graph_spec()
    spec["methodology_replication_contract"].pop("homogeneous_graph_mechanism")

    with pytest.raises(
        ValidationError,
        match=r"explicit v1.14\+ homogeneous graph specs must provide",
    ):
        MethodSpec.model_validate(spec)

    portable_errors = list(
        Draft202012Validator(MethodSpec.model_json_schema()).iter_errors(spec)
    )
    assert portable_errors
    assert any(
        "homogeneous_graph_mechanism" in error.message
        for error in portable_errors
    )


def test_archived_homogeneous_graph_spec_may_omit_mechanism_block():
    spec = _homogeneous_graph_spec()
    spec["schema_version"] = "1.13.0"
    spec["methodology_replication_contract"].pop("homogeneous_graph_mechanism")

    validated = MethodSpec.model_validate(spec)
    assert (
        validated.methodology_replication_contract.homogeneous_graph_mechanism
        is None
    )
    assert list(
        Draft202012Validator(MethodSpec.model_json_schema()).iter_errors(spec)
    ) == []


@pytest.mark.parametrize("relational_structure", [None, {
    "kind": "unsupported",
    "unsupported_kind": "heterogeneous_graph",
}])
def test_current_graph_free_and_unsupported_specs_do_not_invent_mechanism(
    relational_structure,
):
    spec = _minimal_valid_spec(paradigm_id="time_series_forecasting/global")
    spec["schema_version"] = SCHEMA_VERSION
    element = _methodology_element()
    if relational_structure is not None:
        element["paper_element_ids"] = ["paper-relational-form"]
        element["relational_structure"] = relational_structure
    spec = _add_methodology_contract(spec, elements=[element])

    validated = MethodSpec.model_validate(spec)
    assert (
        validated.methodology_replication_contract.homogeneous_graph_mechanism
        is None
    )
    assert list(
        Draft202012Validator(MethodSpec.model_json_schema()).iter_errors(spec)
    ) == []


@pytest.mark.parametrize(
    "kind",
    ["empty_graph", "identity_graph", "permuted_graph"],
)
def test_paper_grounded_graph_input_controls_use_the_closed_ablation_arm(kind):
    spec = _homogeneous_graph_spec()
    mechanism = spec["methodology_replication_contract"][
        "homogeneous_graph_mechanism"
    ]
    mechanism["contribution_ablation"] = {
        "kind": kind,
        "element_id": "graph-free-decoder-control",
        "discriminating_probe_ref": (
            "graph_mechanism.neighbor_sensitivity"
        ),
    }

    validated = MethodSpec.model_validate(spec)
    got = validated.methodology_replication_contract.homogeneous_graph_mechanism
    assert got.contribution_ablation.kind == kind


def test_deterministic_per_source_cap_carries_its_own_parameter_identity():
    spec = _homogeneous_graph_spec()
    construction = spec["methodology_replication_contract"][
        "homogeneous_graph_mechanism"
    ]["construction"]
    construction["cap"] = {
        "kind": "per_source_top_similarity",
        "parameter": {
            "params_name": "max_neighbors",
            "callable_parameter": "max_neighbors",
        },
    }

    validated = MethodSpec.model_validate(spec)
    got = validated.methodology_replication_contract.homogeneous_graph_mechanism
    assert got.construction.cap.parameter.params_name == "max_neighbors"


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        (
            ("construction", "self_loop_policy"),
            "infer",
            "self_loop_policy",
        ),
        (
            ("construction", "direction_policy"),
            "undirected",
            "direction_policy",
        ),
        (
            ("construction", "cap"),
            {"kind": "sampled_neighbors"},
            "cap",
        ),
        (
            ("contribution_ablation", "kind"),
            "convenient_baseline",
            "contribution_ablation",
        ),
    ],
)
def test_homogeneous_graph_mechanism_rejects_open_ended_v1_semantics(
    path, value, message,
):
    spec = _homogeneous_graph_spec()
    target = spec["methodology_replication_contract"][
        "homogeneous_graph_mechanism"
    ]
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value

    with pytest.raises(ValidationError, match=message):
        MethodSpec.model_validate(spec)


def test_homogeneous_graph_mechanism_allows_shared_exact_element_owners():
    spec = _homogeneous_graph_spec()
    mechanism = spec["methodology_replication_contract"][
        "homogeneous_graph_mechanism"
    ]
    mechanism["message_passing"]["element_id"] = mechanism[
        "construction"
    ]["element_id"]
    construction = next(
        element
        for element in spec["methodology_replication_contract"]["elements"]
        if element["element_id"] == mechanism["construction"]["element_id"]
    )
    original_message = next(
        element
        for element in spec["methodology_replication_contract"]["elements"]
        if element["element_id"] == "graph-neighbor-message-passing"
    )
    # The original message element is no longer bound by the block, so its
    # marker and refs must go with it (markers and graph refs are derived
    # from the block under R2C-092; a leftover marker is a stray).
    original_message["relational_structure"] = None
    original_message["verification_probe_refs"] = []

    validated = MethodSpec.model_validate(spec)
    contract = validated.methodology_replication_contract
    validated_mechanism = contract.homogeneous_graph_mechanism
    assert validated_mechanism is not None
    assert validated_mechanism.message_passing.element_id == (
        validated_mechanism.construction.element_id
    )
    shared_owner = next(
        element
        for element in contract.elements
        if element.element_id == validated_mechanism.construction.element_id
    )
    # The shared owner receives the union of both roles' derived refs.
    assert {
        "graph_mechanism.parameter_agreement",
        "graph_mechanism.construction_semantics",
        "graph_mechanism.topology_sensitivity",
        "graph_mechanism.neighbor_sensitivity",
        "graph_mechanism.permutation_equivalence",
    } <= set(shared_owner.verification_probe_refs)


def test_homogeneous_graph_mechanism_rejects_unknown_element_owner():
    spec = _homogeneous_graph_spec()

    spec["methodology_replication_contract"]["homogeneous_graph_mechanism"][
        "alignment_element_id"
    ] = "unknown-alignment"
    with pytest.raises(ValidationError, match="unknown methodology element"):
        MethodSpec.model_validate(spec)


def test_graph_mechanism_allows_explicitly_absent_contribution_control():
    spec = _homogeneous_graph_spec()
    contract = spec["methodology_replication_contract"]
    mechanism = contract["homogeneous_graph_mechanism"]
    mechanism["contribution_ablation"] = None
    mechanism["probe_refs"]["contribution_ablation"] = None
    contract["elements"] = [
        element
        for element in contract["elements"]
        if element["element_id"] != "graph-free-decoder-control"
    ]

    validated = MethodSpec.model_validate(spec)
    got = validated.methodology_replication_contract.homogeneous_graph_mechanism

    assert got is not None
    assert got.contribution_ablation is None
    assert got.probe_refs.contribution_ablation is None
    assert list(
        Draft202012Validator(MethodSpec.model_json_schema()).iter_errors(spec)
    ) == []


@pytest.mark.parametrize(
    "path",
    [
        ("contribution_ablation",),
        ("probe_refs", "contribution_ablation"),
    ],
)
def test_graph_contribution_nullable_fields_are_still_required(path):
    spec = _homogeneous_graph_spec()
    mechanism = spec["methodology_replication_contract"][
        "homogeneous_graph_mechanism"
    ]
    target = mechanism
    for key in path[:-1]:
        target = target[key]
    target.pop(path[-1])

    with pytest.raises(ValidationError, match=path[-1]):
        MethodSpec.model_validate(spec)

    assert list(
        Draft202012Validator(MethodSpec.model_json_schema()).iter_errors(spec)
    )


@pytest.mark.parametrize(
    ("ablation_is_null", "probe_ref_is_null"),
    [(True, False), (False, True)],
)
def test_graph_contribution_control_and_probe_ref_must_agree(
    ablation_is_null, probe_ref_is_null,
):
    spec = _homogeneous_graph_spec()
    mechanism = spec["methodology_replication_contract"][
        "homogeneous_graph_mechanism"
    ]
    if ablation_is_null:
        mechanism["contribution_ablation"] = None
    if probe_ref_is_null:
        mechanism["probe_refs"]["contribution_ablation"] = None

    with pytest.raises(ValidationError, match="contribution"):
        MethodSpec.model_validate(spec)


@pytest.mark.parametrize(
    "criterion",
    [
        "graph_mechanism.topology_sensitivity",
        "graph_mechanism.neighbor_sensitivity",
    ],
)
def test_graph_contribution_control_names_exact_discriminating_probe(criterion):
    spec = _homogeneous_graph_spec()
    mechanism = spec["methodology_replication_contract"][
        "homogeneous_graph_mechanism"
    ]
    mechanism["contribution_ablation"][
        "discriminating_probe_ref"
    ] = criterion

    validated = MethodSpec.model_validate(spec)
    got = validated.methodology_replication_contract.homogeneous_graph_mechanism

    assert got.contribution_ablation.discriminating_probe_ref == criterion


def test_graph_contribution_control_rejects_invented_discriminator():
    spec = _homogeneous_graph_spec()
    mechanism = spec["methodology_replication_contract"][
        "homogeneous_graph_mechanism"
    ]
    mechanism["contribution_ablation"][
        "discriminating_probe_ref"
    ] = "graph_mechanism.permutation_equivalence"

    with pytest.raises(ValidationError, match="discriminating_probe_ref"):
        MethodSpec.model_validate(spec)


def test_duplicated_graph_refs_are_normalized_to_their_derived_owners():
    # 2026-08-10 pdfgnn Attempt 1: two duplicate-ownership findings surfaced
    # one fix iteration apart and exhausted the cap with a single error left.
    # Under R2C-092 the same shape is not an error at all: graph refs are
    # derived from the block, so the hand-copied duplicates are stripped and
    # every graph ref ends with exactly one owner.
    spec = _homogeneous_graph_spec()
    elements = spec["methodology_replication_contract"]["elements"]
    by_id = {element["element_id"]: element for element in elements}
    by_id["graph-relational-alignment"]["verification_probe_refs"].append(
        "graph_mechanism.construction_semantics"
    )
    by_id["graph-free-decoder-control"]["verification_probe_refs"].append(
        "graph_mechanism.topology_sensitivity"
    )

    validated = MethodSpec.model_validate(spec)
    contract = validated.methodology_replication_contract
    for probe_ref in (
        "graph_mechanism.construction_semantics",
        "graph_mechanism.topology_sensitivity",
    ):
        owners = [
            element.element_id
            for element in contract.elements
            if probe_ref in element.verification_probe_refs
        ]
        assert len(owners) == 1, (probe_ref, owners)


def test_null_graph_contribution_strips_the_stray_hg7_ref():
    spec = _homogeneous_graph_spec()
    mechanism = spec["methodology_replication_contract"][
        "homogeneous_graph_mechanism"
    ]
    mechanism["contribution_ablation"] = None
    mechanism["probe_refs"]["contribution_ablation"] = None

    validated = MethodSpec.model_validate(spec)
    contract = validated.methodology_replication_contract
    assert not any(
        "graph_mechanism.contribution_ablation"
        in element.verification_probe_refs
        for element in contract.elements
    )


def test_per_source_cap_schema_records_closed_mutual_top_k_semantics():
    schema = MethodSpec.model_json_schema()
    description = schema["$defs"]["PerSourceTopSimilarityCap"]["description"]

    assert "descending cosine similarity" in description
    assert "canonical target-index" in description
    assert "both endpoints select each other" in description
    assert "Self loops" in description


@pytest.mark.parametrize(
    "path",
    [
        ("construction", "callable"),
        ("message_passing", "callable"),
        ("contribution_ablation", "callable"),
    ],
)
def test_graph_callable_qualname_requires_a_top_level_probeable_helper(path):
    spec = _homogeneous_graph_spec()
    mechanism = spec["methodology_replication_contract"][
        "homogeneous_graph_mechanism"
    ]
    mechanism[path[0]][path[1]]["qualname"] = "GraphForecaster.forward"

    with pytest.raises(ValidationError, match="string_pattern_mismatch"):
        MethodSpec.model_validate(spec)

    portable_errors = list(
        Draft202012Validator(MethodSpec.model_json_schema()).iter_errors(spec)
    )
    assert portable_errors


@pytest.mark.parametrize("qualname", ["_private_graph_helper", "for"])
def test_graph_callable_qualname_must_be_public_and_not_a_keyword(qualname):
    spec = _homogeneous_graph_spec()
    mechanism = spec["methodology_replication_contract"][
        "homogeneous_graph_mechanism"
    ]
    mechanism["construction"]["callable"]["qualname"] = qualname

    with pytest.raises(ValidationError, match="qualname"):
        MethodSpec.model_validate(spec)
    assert list(
        Draft202012Validator(MethodSpec.model_json_schema()).iter_errors(spec)
    )


def test_graph_mechanism_element_roles_stay_validated_and_refs_derive():
    # The ablation element's role is producer-owned and stays validated.
    spec = _homogeneous_graph_spec()
    elements = spec["methodology_replication_contract"]["elements"]
    ablation = next(
        element for element in elements
        if element["element_id"] == "graph-free-decoder-control"
    )
    ablation["role"] = "supporting_mechanism"
    with pytest.raises(ValidationError, match="role must be 'evaluation_control'"):
        MethodSpec.model_validate(spec)

    # Omitted or dropped graph refs are derived back from the block instead
    # of failing the producer (R2C-092): the Attempt 4/5 fix loops dropped a
    # ref every time they fixed an adjacent error.
    spec = _homogeneous_graph_spec()
    elements = spec["methodology_replication_contract"]["elements"]
    alignment = next(
        element for element in elements
        if element["element_id"] == "graph-relational-alignment"
    )
    alignment["verification_probe_refs"] = []
    message = next(
        element for element in elements
        if element["element_id"] == "graph-neighbor-message-passing"
    )
    message["verification_probe_refs"].remove(
        "graph_mechanism.neighbor_sensitivity"
    )

    validated = MethodSpec.model_validate(spec)
    contract = validated.methodology_replication_contract
    by_id = {element.element_id: element for element in contract.elements}
    assert "graph_mechanism.alignment_prerequisite" in (
        by_id["graph-relational-alignment"].verification_probe_refs
    )
    assert "graph_mechanism.neighbor_sensitivity" in (
        by_id["graph-neighbor-message-passing"].verification_probe_refs
    )


def test_permutation_ref_is_present_exactly_when_equivariance_is_applicable():
    spec = _homogeneous_graph_spec()
    mechanism = spec["methodology_replication_contract"][
        "homogeneous_graph_mechanism"
    ]
    mechanism["probe_refs"]["permutation"] = None
    with pytest.raises(ValidationError, match="equivariant.*requires"):
        MethodSpec.model_validate(spec)

    spec = _homogeneous_graph_spec()
    mechanism = spec["methodology_replication_contract"][
        "homogeneous_graph_mechanism"
    ]
    mechanism["permutation_applicability"] = "not_applicable"
    with pytest.raises(ValidationError, match="not_applicable.*null"):
        MethodSpec.model_validate(spec)


def test_graph_threshold_and_cap_must_bind_distinct_params_entries():
    spec = _homogeneous_graph_spec()
    construction = spec["methodology_replication_contract"][
        "homogeneous_graph_mechanism"
    ]["construction"]
    construction["cap"] = {
        "kind": "per_source_top_similarity",
        "parameter": {
            "params_name": "similarity_threshold",
            "callable_parameter": "max_neighbors",
        },
    }

    with pytest.raises(ValidationError, match="different params entries"):
        MethodSpec.model_validate(spec)


@pytest.mark.parametrize("collision", ["feature_threshold", "threshold_cap"])
def test_graph_constructor_callable_parameter_roles_are_pairwise_distinct(
    collision,
):
    spec = _homogeneous_graph_spec()
    construction = spec["methodology_replication_contract"][
        "homogeneous_graph_mechanism"
    ]["construction"]
    if collision == "feature_threshold":
        construction["feature_parameter"] = construction["threshold"][
            "parameter"
        ]["callable_parameter"]
    else:
        construction["cap"] = {
            "kind": "per_source_top_similarity",
            "parameter": {
                "params_name": "max_neighbors",
                "callable_parameter": construction["threshold"]["parameter"][
                    "callable_parameter"
                ],
            },
        }

    with pytest.raises(ValidationError, match="pairwise-distinct"):
        MethodSpec.model_validate(spec)


def test_graph_message_and_callable_null_bind_distinct_inputs_and_same_output():
    spec = _homogeneous_graph_spec()
    mechanism = spec["methodology_replication_contract"][
        "homogeneous_graph_mechanism"
    ]
    null = mechanism["contribution_ablation"]
    validated = MethodSpec.model_validate(spec)
    got = validated.methodology_replication_contract.homogeneous_graph_mechanism
    assert got.contribution_ablation.graph_parameter == null["graph_parameter"]
    assert got.contribution_ablation.output_root == (
        got.message_passing.output_root
    )

    mechanism["message_passing"]["neighbor_signal_parameter"] = (
        mechanism["message_passing"]["graph_parameter"]
    )
    with pytest.raises(ValidationError, match="distinct callable parameters"):
        MethodSpec.model_validate(spec)

    spec = _homogeneous_graph_spec()
    mechanism = spec["methodology_replication_contract"][
        "homogeneous_graph_mechanism"
    ]
    mechanism["contribution_ablation"]["output_root"] = "outputs.forecasts"
    with pytest.raises(ValidationError, match="same coindexed output root"):
        MethodSpec.model_validate(spec)


@pytest.mark.parametrize("duplicate_role", ["message_passing", "ablation"])
def test_graph_callable_roles_require_distinct_exact_identities(duplicate_role):
    spec = _homogeneous_graph_spec()
    mechanism = spec["methodology_replication_contract"][
        "homogeneous_graph_mechanism"
    ]
    construction_callable = copy.deepcopy(
        mechanism["construction"]["callable"]
    )
    if duplicate_role == "message_passing":
        mechanism["message_passing"]["callable"] = construction_callable
    else:
        mechanism["contribution_ablation"]["callable"] = construction_callable

    with pytest.raises(ValidationError, match="distinct callable identities"):
        MethodSpec.model_validate(spec)


# ---------------------------------------------------------------------------
# The exact probe-to-obligation carrier (R2C-087)
# ---------------------------------------------------------------------------


def test_contract_element_carries_unique_verification_probe_refs():
    spec = _al_spec_with_data_setup_ints()
    element = _methodology_element()
    element["verification_probe_refs"] = [
        "al_loop.acquisition_contract",
        "al_loop.microharness",
    ]
    spec = _add_methodology_contract(spec, elements=[element])

    validated = MethodSpec.model_validate(spec)
    got = validated.methodology_replication_contract.elements[0]
    assert got.verification_probe_refs == [
        "al_loop.acquisition_contract",
        "al_loop.microharness",
    ]


def test_exported_probe_ref_carrier_declares_nonblank_unique_items():
    schema = MethodSpec.model_json_schema()
    carrier = schema["$defs"]["MethodologyContractElement"]["properties"][
        "verification_probe_refs"
    ]

    assert carrier["uniqueItems"] is True
    assert carrier["items"]["type"] == "string"
    assert carrier["items"]["minLength"] == 1
    assert carrier["items"]["pattern"] == r"\S"


def test_verification_probe_refs_are_optional_for_archived_graph_free_specs():
    spec = _minimal_valid_spec(paradigm_id="motion_planning/sampling_based")
    spec["schema_version"] = "1.11.0"
    element = _methodology_element()
    del element["verification_probe_refs"]
    spec = _add_methodology_contract(spec, elements=[element])

    validated = MethodSpec.model_validate(spec)
    assert (
        validated.methodology_replication_contract.elements[
            0
        ].verification_probe_refs
        == []
    )
    assert cross_check_verification_probe_refs(validated, ROOT) == []


def test_probe_ref_activation_remains_schema_112_after_later_schema_bump():
    assert VERIFICATION_PROBE_REFS_SCHEMA_VERSION == "1.12.0"
    assert HOMOGENEOUS_GRAPH_MECHANISM_SCHEMA_VERSION == "1.14.0"
    assert HOMOGENEOUS_GRAPH_ALIGNMENT_PROBE_REF == (
        "graph_mechanism.alignment_prerequisite"
    )

    spec = _al_spec_with_data_setup_ints()
    spec["schema_version"] = "1.12.0"
    element = _methodology_element()
    del element["verification_probe_refs"]
    spec = _add_methodology_contract(spec, elements=[element])

    errors = cross_check_verification_probe_refs(
        MethodSpec.model_validate(spec), ROOT
    )

    assert len(errors) == 1
    assert "omits verification_probe_refs under schema 1.12.0" in errors[0]
    assert "use []" in errors[0]


def test_fresh_cli_rejects_a_self_declared_archived_schema(tmp_path):
    spec = _al_spec_with_data_setup_ints()
    spec["schema_version"] = "1.11.0"
    spec_path = tmp_path / "method_spec.json"
    spec_path.write_text(json.dumps(spec, indent=2), encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "validate_method_spec.py"),
            str(spec_path),
            "--require-current-schema",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 2
    assert "current-schema requirement failed" in result.stderr
    assert repr(SCHEMA_VERSION) in result.stderr


def test_fresh_cli_rejects_an_omitted_schema_version_instead_of_defaulting_it(
    tmp_path,
):
    spec = _al_spec_with_data_setup_ints()
    del spec["schema_version"]
    spec_path = tmp_path / "method_spec.json"
    spec_path.write_text(json.dumps(spec, indent=2), encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "validate_method_spec.py"),
            str(spec_path),
            "--require-current-schema",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 2
    assert "raw value is None" in result.stderr


def test_archived_cli_read_still_accepts_an_older_schema_without_fresh_gate(
    tmp_path,
):
    spec = _al_spec_with_data_setup_ints()
    spec["schema_version"] = "1.11.0"
    spec_path = tmp_path / "method_spec.json"
    spec_path.write_text(json.dumps(spec, indent=2), encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "validate_method_spec.py"),
            str(spec_path),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0
    assert "validates against schema v1.11.0" in result.stdout


@pytest.mark.parametrize(
    "refs",
    [
        [""],
        ["   "],
        ["al_loop.microharness", "al_loop.microharness"],
        ["al_loop.microharness", " al_loop.microharness "],
    ],
)
def test_verification_probe_refs_reject_blank_or_duplicate_values(refs):
    spec = _al_spec_with_data_setup_ints()
    element = _methodology_element()
    element["verification_probe_refs"] = refs
    spec = _add_methodology_contract(spec, elements=[element])

    with pytest.raises(
        ValidationError, match="verification_probe_refs must be non-blank and unique"
    ):
        MethodSpec.model_validate(spec)


def test_verification_probe_refs_accept_an_inherited_effective_probe():
    # The batch-acquisition variant inherits this ref from active_learning.
    spec = _al_spec_with_data_setup_ints()
    element = _methodology_element()
    element["verification_probe_refs"] = ["al_loop.acquisition_contract"]
    spec = _add_methodology_contract(spec, elements=[element])

    assert (
        cross_check_verification_probe_refs(MethodSpec.model_validate(spec), ROOT)
        == []
    )


def test_verification_probe_refs_reject_unknown_exact_identity():
    spec = _al_spec_with_data_setup_ints()
    element = _methodology_element()
    element["verification_probe_refs"] = ["al_loop.acquisition-contract"]
    spec = _add_methodology_contract(spec, elements=[element])

    errors = cross_check_verification_probe_refs(
        MethodSpec.model_validate(spec), ROOT
    )
    assert len(errors) == 1
    assert "al_loop.acquisition-contract" in errors[0]
    assert "al_loop.acquisition_contract" in errors[0]
    assert element["element_id"] in errors[0]


def test_verification_probe_refs_resolve_registered_taxonomy_aliases():
    spec = _al_spec_with_data_setup_ints()
    spec["comparison"]["classification"]["id"] = "pool_based_active_learning"
    element = _methodology_element()
    element["verification_probe_refs"] = ["al_loop.acquisition_contract"]
    spec = _add_methodology_contract(spec, elements=[element])

    assert (
        cross_check_verification_probe_refs(MethodSpec.model_validate(spec), ROOT)
        == []
    )


def test_verification_probe_refs_resolve_run_local_overlay(tmp_path):
    gap_id = "motion_planning/synthetic_probe_binding"
    probe_ref = "synthetic_gap.binding_contract"
    pack = {
        "schema_version": "1.0",
        "status": "provisional",
        "legacy_paradigm": gap_id,
        "extends": "motion_planning",
        "taxonomy_id": "CLC-MP/synthetic_probe_binding",
        "fingerprint": {
            "what_it_is": "A synthetic planner used only to test probe binding.",
            "not_this": [],
        },
        "scaffold_hints": {
            "interface_hint": "plan(start, goal, environment, seed) -> PlanResult",
        },
        "semantic_checks": [
            {
                "id": "SYNTH-binding-contract",
                "stage": "stage_2c_method",
                "check": "The synthetic planner exercises its binding contract.",
                "silent_failure": "The probe passes without exercising the contract.",
                "severity": "error",
                "status": "observed",
                "probe": probe_ref,
            }
        ],
        "smoke_bugs": [],
    }
    packs_dir = tmp_path / "provisional_packs"
    install_dir = packs_dir / "20260809-r2c087"
    install_dir.mkdir(parents=True)
    (install_dir / "pack.yaml").write_text(json.dumps(pack), encoding="utf-8")

    spec = _minimal_valid_spec(paradigm_id=gap_id)
    element = _methodology_element()
    element["verification_probe_refs"] = [probe_ref]
    spec = _add_methodology_contract(spec, elements=[element])
    validated = MethodSpec.model_validate(spec)

    assert cross_check_verification_probe_refs(validated, ROOT) != []
    assert (
        cross_check_verification_probe_refs(
            validated, ROOT, provisional_packs_dir=packs_dir
        )
        == []
    )


def test_verification_probe_refs_fail_the_strict_cli_by_exact_identity(tmp_path):
    spec = _al_spec_with_data_setup_ints()
    element = _methodology_element()
    element["verification_probe_refs"] = ["al_loop.acquisition-contract"]
    spec = _add_methodology_contract(spec, elements=[element])
    spec_path = tmp_path / "method_spec.json"
    spec_path.write_text(json.dumps(spec, indent=2), encoding="utf-8")

    strict = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "validate_method_spec.py"),
            str(spec_path),
            "--strict",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert strict.returncode == 2
    assert "methodology verification-probe cross-check failed" in strict.stderr
    assert "al_loop.acquisition-contract" in strict.stderr


# ---------------------------------------------------------------------------
# The contract-to-paper-map crosswalk (R2C-072)
# ---------------------------------------------------------------------------


def test_contract_element_carries_a_paper_map_crosswalk():
    spec = _al_spec_with_data_setup_ints()
    element = _methodology_element()
    element["paper_element_ids"] = ["concept-gradient-embedding", "alg-badge"]
    spec = _add_methodology_contract(spec, elements=[element])

    validated = MethodSpec.model_validate(spec)
    got = validated.methodology_replication_contract.elements[0]
    assert got.paper_element_ids == ["concept-gradient-embedding", "alg-badge"]


def test_the_crosswalk_is_optional_so_older_specs_still_validate():
    # Every spec written before the field existed. It costs the element its
    # contract adjudication and must not fail validation.
    spec = _al_spec_with_data_setup_ints()
    spec = _add_methodology_contract(spec, elements=[_methodology_element()])
    validated = MethodSpec.model_validate(spec)
    assert validated.methodology_replication_contract.elements[0].paper_element_ids == []


def _write_paper_map(tmp_path: Path, *ids: str) -> Path:
    pipeline = tmp_path / ".pipeline"
    pipeline.mkdir(parents=True, exist_ok=True)
    (pipeline / "paper_map.json").write_text(
        json.dumps({"elements": [{"id": i} for i in ids]}), encoding="utf-8")
    return pipeline / "method_spec.json"


def test_a_dangling_crosswalk_id_fails_the_paper_map_cross_check(tmp_path):
    # Worse than a missing crosswalk: it looks like a working link and binds
    # nothing, which is the silent failure the field exists to remove.
    spec = _al_spec_with_data_setup_ints()
    element = _methodology_element()
    element["paper_element_ids"] = ["concept-real", "concept-typo"]
    spec = _add_methodology_contract(spec, elements=[element])
    spec_path = _write_paper_map(tmp_path, "alg-test", "concept-real")

    errors = cross_check_paper_map(MethodSpec.model_validate(spec), spec_path)
    assert len(errors) == 1
    assert "concept-typo" in errors[0]
    assert "concept-real" not in errors[0].split("valid IDs")[0]
    assert element["element_id"] in errors[0]


def test_a_resolvable_crosswalk_passes_the_cross_check(tmp_path):
    spec = _al_spec_with_data_setup_ints()
    element = _methodology_element()
    element["paper_element_ids"] = ["concept-real"]
    spec = _add_methodology_contract(spec, elements=[element])
    spec_path = _write_paper_map(tmp_path, "alg-test", "concept-real")

    assert cross_check_paper_map(MethodSpec.model_validate(spec), spec_path) == []


def test_an_absent_crosswalk_is_not_a_cross_check_error(tmp_path):
    spec = _al_spec_with_data_setup_ints()
    spec = _add_methodology_contract(spec, elements=[_methodology_element()])
    spec_path = _write_paper_map(tmp_path, "alg-test")

    assert cross_check_paper_map(MethodSpec.model_validate(spec), spec_path) == []


def test_a_spec_with_no_contract_at_all_still_cross_checks(tmp_path):
    spec = _al_spec_with_data_setup_ints()
    spec_path = _write_paper_map(tmp_path, "alg-test")

    assert cross_check_paper_map(MethodSpec.model_validate(spec), spec_path) == []


# ---------------------------------------------------------------------------
# R2C-092: the graph-mechanism block is authorable in one pass. Markers and
# graph-ref ownership are derived from the block's element ids; residual
# producer-owned violations report together in one message. Known-bad shapes
# are the 2026-08-10/11 pdfgnn Attempt 4/5 halts: the analyzer relocated the
# marker and dropped refs while fixing adjacent errors, one revelation per
# dispatch, and never converged.
# ---------------------------------------------------------------------------


def test_attempt5_dropped_refs_shape_passes_with_derived_wiring():
    # Attempt 5 final halt: correct markers on the bound elements, but the
    # alignment ref and both construction refs dropped during the fix loop.
    # Under derivation this shape is simply valid, and the refs come back.
    spec = _homogeneous_graph_spec()
    mechanism = spec["methodology_replication_contract"][
        "homogeneous_graph_mechanism"
    ]
    # Attempt 5 also shared one element between alignment and message
    # passing (gnn-encoder-construction); reproduce that composition.
    shared_id = mechanism["message_passing"]["element_id"]
    mechanism["alignment_element_id"] = shared_id
    for element in spec["methodology_replication_contract"]["elements"]:
        element["verification_probe_refs"] = []
    unbound_alignment = next(
        element
        for element in spec["methodology_replication_contract"]["elements"]
        if element["element_id"] == "graph-relational-alignment"
    )
    unbound_alignment["relational_structure"] = None

    validated = MethodSpec.model_validate(spec)
    contract = validated.methodology_replication_contract
    by_id = {element.element_id: element for element in contract.elements}
    assert set(by_id[shared_id].verification_probe_refs) == {
        "graph_mechanism.alignment_prerequisite",
        "graph_mechanism.topology_sensitivity",
        "graph_mechanism.neighbor_sensitivity",
        "graph_mechanism.permutation_equivalence",
    }
    assert set(
        by_id["graph-similarity-construction"].verification_probe_refs
    ) == {
        "graph_mechanism.parameter_agreement",
        "graph_mechanism.construction_semantics",
    }
    assert set(
        by_id["graph-free-decoder-control"].verification_probe_refs
    ) == {"graph_mechanism.contribution_ablation"}


def test_missing_markers_on_bound_elements_are_derived_not_failed():
    # The relocation loop's other half: a bound element lost its marker while
    # the producer fixed an adjacent error. Derivation restores it.
    spec = _homogeneous_graph_spec()
    for element in spec["methodology_replication_contract"]["elements"]:
        element["relational_structure"] = None

    validated = MethodSpec.model_validate(spec)
    contract = validated.methodology_replication_contract
    by_id = {element.element_id: element for element in contract.elements}
    for element_id in (
        "graph-relational-alignment",
        "graph-similarity-construction",
        "graph-neighbor-message-passing",
    ):
        marker = by_id[element_id].relational_structure
        assert marker is not None and marker.kind.value == "homogeneous_graph"
    assert by_id["graph-free-decoder-control"].relational_structure is None


def test_relocated_marker_reports_the_stray_in_one_message():
    # The Attempt 4/5 relocation shape: the marker moved onto an element the
    # block does not bind. The bound element gets its marker derived; the
    # stray is the one remaining producer-owned violation and the message
    # says the wiring is derived.
    spec = _homogeneous_graph_spec()
    elements = spec["methodology_replication_contract"]["elements"]
    by_id = {element["element_id"]: element for element in elements}
    by_id["graph-similarity-construction"]["relational_structure"] = None
    by_id["graph-free-decoder-control"]["relational_structure"] = {
        "kind": "homogeneous_graph"
    }
    by_id["graph-free-decoder-control"]["paper_element_ids"] = ["paper-graph-9"]

    with pytest.raises(ValidationError) as excinfo:
        MethodSpec.model_validate(spec)
    message = str(excinfo.value)
    assert "derived" in message
    assert "graph-free-decoder-control" in message
    # The bound element whose marker went missing is NOT an error under
    # derivation, so it must not be named as one.
    assert "graph-similarity-construction' is bound" not in message


def test_all_marker_violations_report_in_one_message():
    # sequential_gate_reveal known-bad: two independent marker violations
    # (a stray marker and a bound element without paper grounding) must land
    # in ONE validation message, not one per fix dispatch.
    spec = _homogeneous_graph_spec()
    elements = spec["methodology_replication_contract"]["elements"]
    by_id = {element["element_id"]: element for element in elements}
    by_id["graph-similarity-construction"]["relational_structure"] = None
    by_id["graph-similarity-construction"]["paper_element_ids"] = []
    stray = by_id["graph-free-decoder-control"]
    stray["relational_structure"] = {"kind": "homogeneous_graph"}
    stray["paper_element_ids"] = ["paper-graph-8"]

    with pytest.raises(ValidationError) as excinfo:
        MethodSpec.model_validate(spec)
    message = str(excinfo.value)
    assert message.count("Value error") == 1
    assert "no paper_element_ids" in message
    assert stray["element_id"] in message


def test_bound_element_with_unsupported_marker_contradicts_the_block():
    spec = _homogeneous_graph_spec()
    elements = spec["methodology_replication_contract"]["elements"]
    construction = next(
        element for element in elements
        if element["element_id"] == "graph-similarity-construction"
    )
    construction["relational_structure"] = {
        "kind": "unsupported",
        "unsupported_kind": "heterogeneous_graph",
    }

    with pytest.raises(ValidationError, match="contradicting the block"):
        MethodSpec.model_validate(spec)


def test_specs_without_a_mechanism_keep_declared_refs_untouched():
    # Graph-free control: without a block, no derivation runs and declared
    # refs pass through byte-identical (archived-spec behavior preserved).
    spec = _al_spec_with_data_setup_ints()
    element = _methodology_element()
    element["verification_probe_refs"] = [
        "active_learning.batch_composition",
    ]
    spec = _add_methodology_contract(spec, elements=[element])

    validated = MethodSpec.model_validate(spec)
    got = validated.methodology_replication_contract.elements[0]
    assert list(got.verification_probe_refs) == [
        "active_learning.batch_composition",
    ]


def test_derived_refs_do_not_mark_the_field_declared():
    # The strict validator's explicit-emission floor keys on the producer
    # having written the field; derivation must not silently satisfy it.
    spec = _homogeneous_graph_spec()
    elements = spec["methodology_replication_contract"]["elements"]
    alignment = next(
        element for element in elements
        if element["element_id"] == "graph-relational-alignment"
    )
    alignment.pop("verification_probe_refs", None)

    validated = MethodSpec.model_validate(spec)
    contract = validated.methodology_replication_contract
    got = next(
        element for element in contract.elements
        if element.element_id == "graph-relational-alignment"
    )
    assert "graph_mechanism.alignment_prerequisite" in (
        got.verification_probe_refs
    )
    assert "verification_probe_refs" not in got.model_fields_set


def test_derived_graph_refs_cross_check_clean_on_the_registered_family():
    # R2C-092: with zero transcribed refs, the derived graph wiring passes
    # the strict taxonomy cross-check on the committed forecasting family.
    spec = _homogeneous_graph_spec()
    spec["comparison"]["classification"]["id"] = "time_series_forecasting"
    for element in spec["methodology_replication_contract"]["elements"]:
        element["verification_probe_refs"] = []

    validated = MethodSpec.model_validate(spec)
    assert cross_check_verification_probe_refs(validated, ROOT) == []


def test_unregistered_family_reports_derived_refs_as_a_pipeline_gap():
    # A taxonomy node without graph probe registrations cannot blame the
    # producer for refs the pipeline derived onto the elements.
    spec = _homogeneous_graph_spec()
    spec["comparison"]["classification"]["id"] = "motion_planning"
    for element in spec["methodology_replication_contract"]["elements"]:
        element["verification_probe_refs"] = []

    validated = MethodSpec.model_validate(spec)
    errors = cross_check_verification_probe_refs(validated, ROOT)
    assert errors
    assert all(
        error.startswith("pipeline coverage gap (not a producer error)")
        for error in errors
    )


# ---------------------------------------------------------------------------
# Symbol-kind vs the taxonomy build plan (bayesian-active-learning
# 2026-09-01): a spec that promises a symbol under one kind while the
# matched build plan declares the other is a contradiction no stage-2
# producer can fix — the coder follows the build plan, so the 2.d kind
# re-check halts the run after generation. Stage-1 strict validation now
# rejects it at the producing stage.
# ---------------------------------------------------------------------------

from scripts.validate_method_spec import (  # noqa: E402
    cross_check_symbol_kinds_against_build_plan,
)


def _al_data_setup(spec: dict) -> dict:
    # AL specs must populate the four benchmark data_setup fields.
    spec["critical_requirements"]["data_setup"].update(
        {"initial_labeled": 20, "batch_size": 100,
         "total_budget": 620, "num_rounds": 6})
    return spec


def _kind_crosscheck(spec_dict: dict, tmp_path) -> list[str]:
    spec_path = tmp_path / "method_spec.json"
    spec_path.write_text(json.dumps(spec_dict), encoding="utf-8")
    return cross_check_symbol_kinds_against_build_plan(
        MethodSpec.model_validate(spec_dict), spec_path, ROOT)


def test_symbol_kind_contradicting_the_build_plan_fails_strict(tmp_path):
    # The live halt shape: build_model promised as a class while the AL
    # build plan declares it a function factory in method/training.py.
    spec = _al_data_setup(_minimal_valid_spec(paradigm_id="active_learning/bayesian"))
    spec["try_it_out"]["system_provides"] = [
        {"name": "model and training", "description": "Builds the model.",
         "type": "code", "symbol": "build_model", "symbol_kind": "class"},
    ]
    errors = _kind_crosscheck(spec, tmp_path)
    assert len(errors) == 1
    assert "build_model" in errors[0]
    assert "kind='function'" in errors[0]
    assert "method/training.py" in errors[0]
    assert "Fix the SPEC" in errors[0]


def test_symbol_kind_agreeing_with_the_build_plan_passes(tmp_path):
    spec = _al_data_setup(_minimal_valid_spec(paradigm_id="active_learning/bayesian"))
    spec["try_it_out"]["system_provides"] = [
        {"name": "model and training", "description": "Builds the model.",
         "type": "code", "symbol": "build_model", "symbol_kind": "function"},
    ]
    assert _kind_crosscheck(spec, tmp_path) == []


def test_symbols_outside_the_build_plan_are_untouched(tmp_path):
    # Conservative posture: a symbol the plan does not declare, an entry
    # without symbol_kind, and an unserved paradigm all stay unenforced.
    spec = _al_data_setup(_minimal_valid_spec(paradigm_id="active_learning/bayesian"))
    spec["try_it_out"]["system_provides"] = [
        {"name": "helper", "description": "Method-owned helper.",
         "type": "code", "symbol": "construct_core_set",
         "symbol_kind": "function"},
        {"name": "kindless", "description": "Name-only promise.",
         "type": "code", "symbol": "build_model"},
    ]
    assert _kind_crosscheck(spec, tmp_path) == []
    unserved = _minimal_valid_spec(paradigm_id="not/a/real/paradigm")
    unserved["try_it_out"]["system_provides"] = [
        {"name": "model", "description": "d.", "type": "code",
         "symbol": "build_model", "symbol_kind": "class"},
    ]
    assert _kind_crosscheck(unserved, tmp_path) == []
