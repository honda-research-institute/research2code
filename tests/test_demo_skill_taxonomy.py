"""Family-owned demo-skill taxonomy contract (R2C-086)."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from scripts import taxonomy as tx
from scripts.validate_taxonomy import lint


ROOT = Path(__file__).resolve().parents[1]

FORECASTING_CONTRACT = {
    "schema_version": "1.0.0",
    "primary_metric": {
        "id": "rmse",
        "direction": "lower_is_better",
        "aggregation": "root_mean_squared_error",
        "units": "target_units",
    },
    "eligibility": {
        "minimum_finite_rows": 1,
        "require_nonzero_actuals": True,
    },
    "decision_policy": "all_required",
    "comparators": [
        {
            "id": "predict_zero",
            "role": "degeneracy_floor",
            "implementation": "predict_zero",
            "required": True,
            "absolute_margin": 0.0,
            "tolerance": 1e-12,
        },
        {
            "id": "repeat_last",
            "role": "minimum_competence",
            "implementation": "repeat_last_pre_window",
            "required": True,
            "absolute_margin": 0.0,
            "tolerance": 1e-12,
        },
    ],
}


def _taxonomy_with(contract: object) -> tx.Taxonomy:
    raw = {
        "schema_version": "2.0",
        "method_roots": {
            "TE": {
                "name": "Train",
                "families": {
                    "TE-DEMO": {
                        "name": "Demo",
                        "variants": {
                            "demo_method": {
                                "name": "Demo",
                                "taxonomy_id": "TE-DEMO/demo_method",
                                "status": "populated",
                                "legacy_paradigm": "demo_method",
                                "fingerprint": {"what_it_is": "A demo method."},
                                "demo_skill": contract,
                            }
                        },
                    }
                },
            }
        },
        "task_domains": {},
    }
    return tx.load_taxonomy_uncached(ROOT, raw)


def _codes(contract: object) -> set[str]:
    return {diagnostic.code for diagnostic in lint(_taxonomy_with(contract))}


def test_forecasting_demo_skill_contract_is_exact_and_alias_served():
    canonical = tx.load_demo_skill("time_series_forecasting")

    assert canonical == FORECASTING_CONTRACT
    assert tx.load_demo_skill("graph_time_series_forecasting") == canonical

    node = tx.serves("time_series_forecasting")
    assert node is not None
    assert tx.node_implementation(node)["demo_skill"] == canonical
    assert tx.node_testing(node)["demo_skill"] == canonical


def test_adjacent_families_do_not_inherit_forecasting_comparators():
    # Active learning keeps its chance-based success semantics. Adding the
    # forecasting contract must not silently replace that family-owned rule.
    assert tx.load_demo_skill("active_learning") == {}
    assert tx.load_demo_success("active_learning")["checks"] == [
        {
            "kind": "beats_chance",
            "gloss": (
                "the headline accuracy series must beat chance "
                "(1/num_classes) by the shared margin"
            ),
        }
    ]

    # A served family with no meaningful declared comparator remains absent,
    # rather than receiving predict-zero or persistence as a default.
    assert tx.load_demo_skill("domain_adaptation") == {}
    assert tx.load_demo_skill("no_such_family") == {}
    assert tx.load_demo_skill(None) == {}


def test_well_formed_demo_skill_contract_lints_clean():
    assert not {code for code in _codes(FORECASTING_CONTRACT) if code.startswith("demo_skill")}


def test_family_declared_constant_comparator_lints_clean():
    contract = deepcopy(FORECASTING_CONTRACT)
    contract["comparators"][0] = {
        **contract["comparators"][0],
        "implementation": "constant_prediction",
        "constant_value": 0.0,
    }

    assert not {
        code for code in _codes(contract) if code.startswith("demo_skill")
    }


@pytest.mark.parametrize(
    ("mutate", "expected_code"),
    [
        (lambda contract: [], "demo_skill_shape"),
        (
            lambda contract: {**contract, "schema_version": "2.0.0"},
            "demo_skill_schema_version",
        ),
        (
            lambda contract: {
                **contract,
                "primary_metric": {
                    **contract["primary_metric"],
                    "direction": "smaller",
                },
            },
            "demo_skill_metric_direction",
        ),
        (
            lambda contract: {
                **contract,
                "primary_metric": {
                    **contract["primary_metric"],
                    "direction": ["lower_is_better"],
                },
            },
            "demo_skill_metric_direction",
        ),
        (
            lambda contract: {
                **contract,
                "eligibility": {
                    **contract["eligibility"],
                    "minimum_finite_rows": 0,
                },
            },
            "demo_skill_minimum_finite_rows",
        ),
        (
            lambda contract: {**contract, "decision_policy": "any_required"},
            "demo_skill_decision_policy",
        ),
        (
            lambda contract: {
                **contract,
                "decision_policy": ["all_required"],
            },
            "demo_skill_decision_policy",
        ),
        (
            lambda contract: {**contract, "comparators": []},
            "demo_skill_comparators",
        ),
        (
            lambda contract: {
                **contract,
                "comparators": [
                    contract["comparators"][0],
                    {
                        **contract["comparators"][1],
                        "id": contract["comparators"][0]["id"],
                    },
                ],
            },
            "demo_skill_comparator_id_duplicate",
        ),
        (
            lambda contract: {
                **contract,
                "comparators": [
                    {
                        **contract["comparators"][0],
                        "tolerance": float("inf"),
                    }
                ],
            },
            "demo_skill_comparator_threshold",
        ),
        (
            lambda contract: {
                **contract,
                "comparators": [{
                    **contract["comparators"][0],
                    "implementation": "invented_baseline",
                }],
            },
            "demo_skill_comparator_implementation",
        ),
        (
            lambda contract: {
                **contract,
                "comparators": [{
                    **contract["comparators"][0],
                    "role": ["degeneracy_floor"],
                }],
            },
            "demo_skill_comparator_role",
        ),
        (
            lambda contract: {
                **contract,
                "comparators": [{
                    **contract["comparators"][0],
                    "implementation": ["predict_zero"],
                }],
            },
            "demo_skill_comparator_implementation",
        ),
        (
            lambda contract: {
                **contract,
                "comparators": [{
                    **contract["comparators"][0],
                    "implementation": "constant_prediction",
                }],
            },
            "demo_skill_comparator_constant",
        ),
        (
            lambda contract: {
                **contract,
                "comparators": [{
                    **contract["comparators"][0],
                    "implementation": "constant_prediction",
                    "constant_value": float("nan"),
                }],
            },
            "demo_skill_comparator_constant",
        ),
        (
            lambda contract: {
                **contract,
                "comparators": [
                    {**entry, "required": False}
                    for entry in contract["comparators"]
                ],
            },
            "demo_skill_required_comparator_missing",
        ),
        (
            lambda contract: {**contract, "invented_policy": "magic"},
            "demo_skill_shape",
        ),
    ],
)
def test_demo_skill_lint_rejects_malformed_contracts(mutate, expected_code):
    contract = mutate(deepcopy(FORECASTING_CONTRACT))

    assert expected_code in _codes(contract)


def test_demo_skill_is_served_from_a_provisional_overlay(tmp_path: Path):
    from tests.test_provisional_pack_overlay import GAP_SHAPED_PACK, _install_pack

    pack = deepcopy(GAP_SHAPED_PACK)
    pack["demo_skill"] = deepcopy(FORECASTING_CONTRACT)
    packs_dir = _install_pack(tmp_path, pack, proposal_id="demo-skill")
    overlay = tx.load_taxonomy(ROOT, provisional_packs_dir=packs_dir)

    assert tx.load_demo_skill(
        pack["legacy_paradigm"], taxonomy=overlay
    ) == FORECASTING_CONTRACT
    assert tx.load_demo_skill(pack["legacy_paradigm"]) == {}


def test_forecasting_role_views_carry_producer_and_reviewer_policy():
    from scripts import taxonomy_role_views

    rendered = taxonomy_role_views.render_role_views(ROOT)
    base = Path("docs/generated/field-guides/time_series_forecasting")

    for role in ("producer", "reviewer", "notebook", "diagnostician"):
        text = rendered[base / f"{role}.md"]
        assert "## demo_skill" in text
        assert "implementation: repeat_last_pre_window" in text

    for role in ("producer", "reviewer", "notebook"):
        text = rendered[base / f"{role}.md"]
        normalized = " ".join(text.split())
        assert "literal schema_version `2.0.0`" in text
        assert "evaluation_protocol_role" in text
        assert "do not infer" in text
        assert "a role from the window length" in normalized
