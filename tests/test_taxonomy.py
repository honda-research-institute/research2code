"""Taxonomy SSOT + loader tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
REVIEW_STAGES = (
    "stage_1_analyzer",
    "stage_2b_architecture",
    "stage_2c_method",
    "stage_2x_params",
    "stage_3a_notebook",
)

# The 13 compatibility-facing paradigm ids the SSOT maps. Legacy guide files are
# retired; these ids remain because run specs and researchers still use the
# human-readable slugs.
AL_POPULATED_PARADIGMS = {
    "active_learning",
    "active_learning/batch_acquisition",
    "active_learning/bayesian",
}
PHASE3_POPULATED_PARADIGMS = {
    "knowledge_distillation",
    "knowledge_distillation/detection",
    "knowledge_distillation/detection/cross_modal",
    "vision_transformer",
    "domain_adaptation",
    "motion_planning",
    "motion_planning/sampling_based",
    "motion_planning/optimization_based",
    # Promoted from the SRL run's provisional pack (R2C-029, 2026-07-28).
    "motion_planning/rl_collision_avoidance",
    "stochastic_optimization",
    # Committed directly from five rolls of pdfgnn audit evidence
    # (R2C-070, 2026-08-06), superseding the two run-authored packs.
    "time_series_forecasting",
}
POPULATED_PARADIGMS = AL_POPULATED_PARADIGMS | PHASE3_POPULATED_PARADIGMS
RESERVED_PARADIGMS: set[str] = set()
EXPECTED_LEGACY_PARADIGMS = POPULATED_PARADIGMS | RESERVED_PARADIGMS


# ---------------------------------------------------------------------------
# 0.2 / 0.6 — the SSOT parses and lints clean.
# ---------------------------------------------------------------------------
def test_ssot_parses_and_lints_clean():
    from scripts.taxonomy import load_taxonomy
    from scripts.validate_taxonomy import lint

    tax = load_taxonomy(ROOT)
    errors = [d.as_dict() for d in lint(tax)]
    errors = [d for d in errors if d["severity"] == "error"]
    assert errors == [], f"SSOT lint errors: {errors}"
    # All 9 method roots present and valid.
    assert set(tax.roots) == {"TE", "CLC", "SC", "AD", "PP", "OM", "GEN", "DA", "INF"}


def test_every_legacy_paradigm_maps_to_a_node():
    from scripts.taxonomy import load_taxonomy

    tax = load_taxonomy(ROOT)
    for pid in EXPECTED_LEGACY_PARADIGMS:
        assert tax.node_for_legacy(pid) is not None, f"no node maps legacy paradigm {pid!r}"


# ---------------------------------------------------------------------------
# 0.3 — lookup, aliases, classification-tuple validation.
# ---------------------------------------------------------------------------
def test_alias_resolution_and_lookup():
    from scripts.taxonomy import load_taxonomy

    tax = load_taxonomy(ROOT)
    assert tax.resolve_alias("vit") == "vision_transformer"
    assert tax.variant("vit").taxonomy_id == "TE-AR/vision_transformer"
    # Lookup by taxonomy_id, slug, and alias all resolve to the same node.
    assert tax.variant("TE-TS/active_learning") is tax.variant("active_learning")
    assert tax.variant("pool_based_active_learning") is tax.variant("active_learning")


def test_classification_tuple_rejects_unknown_slug():
    from scripts.taxonomy import TaxonomyError, assert_classification_tuple, load_taxonomy

    tax = load_taxonomy(ROOT)
    # Valid tuple resolves.
    variant, domain = assert_classification_tuple(tax, "active_learning", "CV")
    assert variant.taxonomy_id == "TE-TS/active_learning"
    assert domain == ("CV", None)
    # Leaf slug resolves to its domain.
    _, domain2 = assert_classification_tuple(tax, "active_learning", "image_classification")
    assert domain2 == ("CV", "image_classification")
    # Unknown method variant rejected.
    with pytest.raises(TaxonomyError):
        assert_classification_tuple(tax, "not_a_real_method", "CV")
    # Unknown task domain rejected.
    with pytest.raises(TaxonomyError):
        assert_classification_tuple(tax, "active_learning", "not_a_domain")


# ---------------------------------------------------------------------------
# 0.3 — intra-file inheritance (replaces cross-file extends:).
# ---------------------------------------------------------------------------
def test_intra_file_inheritance_accumulates_checks():
    from scripts.taxonomy import effective_node, load_taxonomy

    tax = load_taxonomy(ROOT)
    al = effective_node(tax, "active_learning")
    al_ids = {c["id"] for c in al["semantic_checks"]}
    bayes = effective_node(tax, "bayesian")
    bayes_ids = {c["id"] for c in bayes["semantic_checks"]}
    # Sub-variant inherits the family/variant checks and adds its own specialization.
    assert al_ids.issubset(bayes_ids)
    assert "AL-bayes-mc-samples" in bayes_ids
    assert "AL-bayes-mc-samples" not in al_ids


def test_domain_overlay_appends_domain_checks():
    from scripts.taxonomy import effective_node, load_taxonomy_uncached

    raw = {
        "schema_version": "2.0",
        "method_roots": {
            "TE": {
                "name": "Train and Evaluate",
                "families": {
                    "TE-TS": {
                        "name": "Training Strategy",
                        "variants": {
                            "demo_method": {
                                "name": "Demo",
                                "taxonomy_id": "TE-TS/demo_method",
                                "semantic_checks": [
                                    {"id": "D-1", "stage": "stage_2c_method", "check": "x",
                                     "silent_failure": "y", "severity": "error", "status": "seed"}
                                ],
                                "domain_checks": {
                                    "CV": [
                                        {"id": "D-CV-1", "stage": "stage_3a_notebook", "check": "z",
                                         "silent_failure": "w", "severity": "warning", "status": "seed"}
                                    ]
                                },
                            }
                        },
                    }
                },
            }
        },
        "task_domains": {"CV": {"name": "Computer Vision", "leaves": {}}},
    }
    tax = load_taxonomy_uncached(ROOT, raw)
    base = {c["id"] for c in effective_node(tax, "demo_method")["semantic_checks"]}
    with_cv = {c["id"] for c in effective_node(tax, "demo_method", "CV")["semantic_checks"]}
    assert base == {"D-1"}
    assert with_cv == {"D-1", "D-CV-1"}


# ---------------------------------------------------------------------------
# 0.5 / 4.1 — taxonomy serving by compatibility id.
# ---------------------------------------------------------------------------
def test_serves_routes_only_populated_nodes():
    """Populated legacy nodes are taxonomy-served; no legacy paradigm is still reserved."""
    from scripts import taxonomy as tx

    tax = tx.load_taxonomy(ROOT)
    for pid in RESERVED_PARADIGMS:
        assert tx.serves(pid, tax) is None, f"{pid} unexpectedly taxonomy-served while reserved"
    for pid in POPULATED_PARADIGMS:
        node = tx.serves(pid, tax)
        assert node is not None, f"{pid} should be taxonomy-served after Phase 3"
        assert node.legacy_paradigm == pid


def test_serves_flips_when_node_is_populated():
    """When a node is `populated` and claims the paradigm, serves() returns it."""
    from scripts import taxonomy as tx

    raw = {
        "schema_version": "2.0",
        "method_roots": {
            "TE": {
                "name": "Train and Evaluate",
                "families": {
                    "TE-TS": {
                        "name": "Training Strategy",
                        "variants": {
                            "active_learning": {
                                "name": "Active Learning",
                                "status": "populated",
                                "legacy_paradigm": "active_learning",
                                "taxonomy_id": "TE-TS/active_learning",
                                "fingerprint": {"what_it_is": "selects informative points"},
                            }
                        },
                    }
                },
            }
        },
    }
    tax = tx.load_taxonomy_uncached(ROOT, raw)
    node = tx.serves("active_learning", tax)
    assert node is not None and node.taxonomy_id == "TE-TS/active_learning"
    assert tx.serves("knowledge_distillation", tax) is None


def test_serves_resolves_registered_aliases_to_the_populated_node():
    """A7 decision execution (the B-02 spin-off): a runtime classification
    may carry any registered taxonomy alias, not only a legacy_paradigm id.
    serves() owns the alias fallback (previously a private copy inside
    load_demo_skill), and the build plan keys the static table on the
    resolved node's canonical legacy id — before this, an alias-id spec
    reached no build plan at all."""
    from pathlib import Path

    from scripts import taxonomy as tx
    from scripts.build_plan import load_build_plan

    tax = tx.load_taxonomy(ROOT)
    node = tx.serves("pool_based_active_learning", tax)
    assert node is not None and node.slug == "active_learning"

    alias_plan = load_build_plan(
        {"comparison": {"classification": {"id": "pool_based_active_learning"}}},
        ROOT)
    canonical_plan = load_build_plan(
        {"comparison": {"classification": {"id": "active_learning"}}}, ROOT)
    assert alias_plan is not None and canonical_plan is not None
    assert alias_plan["plan_key"] == canonical_plan["plan_key"] == "active_learning"
    # Identical mechanism; only the provenance field keeps the spec's own id.
    assert alias_plan["paradigm_id"] == "pool_based_active_learning"
    scrub = lambda p: {k: v for k, v in p.items() if k != "paradigm_id"}
    assert scrub(alias_plan) == scrub(canonical_plan)

    # Negatives hold: unknown ids and unpopulated variants stay unserved.
    assert tx.serves("no_such_family", tax) is None
    assert tx.serves("semi_supervised_learning", tax) is None
    assert load_build_plan(
        {"comparison": {"classification": {"id": "no_such_family"}}}, ROOT) is None


def test_legacy_fallback_is_retired_for_migrated_paradigms():
    """Phase 4.1: all migrated legacy paradigms are served from taxonomy."""
    from scripts import taxonomy as tx

    present = set(tx.registered_paradigm_ids(ROOT))
    missing = EXPECTED_LEGACY_PARADIGMS - present
    assert not missing, f"expected paradigms missing from catalog: {missing}"

    for pid in EXPECTED_LEGACY_PARADIGMS:
        assert tx.serves(pid) is not None, f"{pid} is no longer served"


# ---------------------------------------------------------------------------
# Served-node review checks. Parity is at the COVERAGE / BEHAVIOUR level
# (the node carries decomposed checks and all compatibility ids route through
# the node), NOT byte-identity (the check shape is the
# intended redesign: id/stage/check/silent_failure/severity/status).
# ---------------------------------------------------------------------------
def test_al_node_is_served_through_the_adapter():
    """merged_* for populated paradigms route through taxonomy nodes."""
    from scripts import taxonomy as tx

    for pid in POPULATED_PARADIGMS:
        node_view = tx.merged_stage_review_focus(None, pid, "stage_3a_notebook")
        assert node_view.get("semantic_checks"), f"{pid} did not serve stage_3a checks"
        # Node checks carry the new schema shape.
        for check in node_view["semantic_checks"]:
            assert {"id", "stage", "check", "silent_failure", "severity", "status"} <= set(check)


def test_al_node_preserves_review_coverage_across_stages():
    """The populated AL node declares checks for every review stage."""
    from scripts import taxonomy as tx

    for stage in REVIEW_STAGES:
        node = tx.merged_stage_review_focus(None, "active_learning", stage)
        assert node.get("semantic_checks"), f"node has no checks at {stage}"


def test_al_sub_variant_checks_and_economics_specialize():
    """bayesian/batch inherit AL's full check set and add their own; the
    smoke-economics floor specializes per sub-variant (parity with the legacy
    demo_scale ratios: AL=0.2, batch=0.05, bayesian=0.4)."""
    from scripts import taxonomy as tx

    tax = tx.load_taxonomy(ROOT)
    al = {c["id"] for c in tx.effective_node(tax, "active_learning")["semantic_checks"]}
    bayes = tx.effective_node(tax, "bayesian")
    batch = tx.effective_node(tax, "batch_acquisition")
    assert al <= {c["id"] for c in bayes["semantic_checks"]}
    assert al <= {c["id"] for c in batch["semantic_checks"]}
    assert tx.effective_node(tax, "active_learning")["smoke_economics"]["max_budget_to_pool_ratio"] == 0.2
    assert batch["smoke_economics"]["max_budget_to_pool_ratio"] == 0.05
    assert bayes["smoke_economics"]["max_budget_to_pool_ratio"] == 0.4


def test_al_demo_scale_threshold_is_served_by_paradigm_id_only():
    """The `derive_params` demo-scale floor resolves from taxonomy id."""
    from scripts import taxonomy as tx

    for pid, expected in (("active_learning", 0.2),
                          ("active_learning/batch_acquisition", 0.05),
                          ("active_learning/bayesian", 0.4)):
        assert tx.demo_scale_threshold(ROOT, paradigm_id=pid) == expected, \
            f"node fast-path floor drift for {pid}"

    assert tx.demo_scale_threshold(ROOT, paradigm_id="unknown/paradigm") is None


def test_registered_paradigm_ids_are_taxonomy_served_catalog_parity():
    """2.6/4.2: the orchestrator's registered-id seam reads through taxonomy."""
    from scripts import taxonomy as tx
    from run_pipeline import _registered_paradigm_ids

    assert tx.registered_paradigm_ids(ROOT) == sorted(EXPECTED_LEGACY_PARADIGMS)
    assert _registered_paradigm_ids(ROOT) == sorted(EXPECTED_LEGACY_PARADIGMS)


def test_smoke_economics_serves_ratio_and_param_floors():
    """5.7c: smoke_economics is a structured params/testing context, not only
    the legacy max_budget_to_pool_ratio scalar."""
    from scripts import taxonomy as tx

    batch = tx.load_smoke_economics("active_learning/batch_acquisition")
    assert batch is not None
    assert batch["max_budget_to_pool_ratio"] == 0.05
    assert batch["floors"]["batch_size"]["absolute_min"] == 50

    bayes = tx.load_smoke_economics("active_learning/bayesian")
    assert bayes is not None
    assert bayes["max_budget_to_pool_ratio"] == 0.4
    assert bayes["floors"]["mc_samples"]["absolute_min"] == 20

    assert tx.load_smoke_economics("knowledge_distillation") is None


def test_load_pack_exposes_effective_taxonomy_node_pack():
    """recentering §5.2: packs are effective taxonomy nodes. The loader is
    additive (parent AL fields inherited by bayesian) and now available for
    every legacy paradigm migrated in Phase 3."""
    from scripts import taxonomy as tx

    pack = tx.load_pack("active_learning/bayesian", ROOT)
    assert pack is not None
    assert pack["pack"] == "active_learning/bayesian"
    assert pack["taxonomy_id"] == "TE-TS/active_learning/bayesian"
    assert pack["fingerprint"]["what_it_is"]
    # Parent AL model default inherited; bayesian floor/extra specialized.
    assert pack["model_defaults"]["hidden_dim"]["value"] == 256
    assert pack["model_defaults"]["dropout_rate"]["value"] == 0.5
    assert pack["paradigm_extras"]["mc_samples"]["value"] == 20
    assert pack["smoke_economics"]["floors"]["mc_samples"]["absolute_min"] == 20

    assert tx.load_pack("knowledge_distillation", ROOT) is not None
    assert tx.load_pack("motion_planning", ROOT) is not None
    assert tx.load_pack("motion_planning/sampling_based", ROOT) is not None
    assert tx.load_pack("stochastic_optimization", ROOT) is not None


def _served_al_spec(
    paradigm_id: str = "active_learning",
    *,
    required_model_methods: list[dict] | None = None,
) -> dict:
    classification = {"id": paradigm_id}
    return {
        "comparison": {
            "classification": classification,
            "pluggable_component": {
                "name": "select_batch",
                "signature": "select_batch(model, x_unlabeled, batch_size, seed) -> List[int]",
                "seed_param": "seed",
            },
        },
        "critical_requirements": {
            "model": {"specific_features": []},
            "required_model_methods": required_model_methods or [],
        },
    }


def _served_spec(
    paradigm_id: str,
    *,
    pluggable_name: str | None = None,
    signature: str | None = None,
    task_domain: str | None = None,
) -> dict:
    defaults = {
        "knowledge_distillation": (
            "compute_distillation_loss",
            "compute_distillation_loss(student, teacher, batch, seed) -> torch.Tensor",
        ),
        "knowledge_distillation/detection": (
            "compute_distillation_loss",
            "compute_distillation_loss(student, teacher, batch, seed) -> torch.Tensor",
        ),
        "knowledge_distillation/detection/cross_modal": (
            "compute_distillation_loss",
            "compute_distillation_loss(student, teacher, batch, seed) -> torch.Tensor",
        ),
        "vision_transformer": (
            "group_tokens",
            "group_tokens(tokens, P=None, num_groups=4, *, seed=0) -> list",
        ),
        "domain_adaptation": (
            "generate_pseudo_labels",
            "generate_pseudo_labels(detectors, target_data, seed) -> PseudoLabelSet",
        ),
        "motion_planning": (
            "plan",
            "plan(start, goal, environment, dynamics, seed) -> PlanResult",
        ),
        "motion_planning/sampling_based": (
            "plan",
            "plan(start, goal, environment, dynamics, seed) -> PlanResult",
        ),
        "motion_planning/optimization_based": (
            "plan",
            "plan(start, goal, environment, dynamics, seed) -> PlanResult",
        ),
        "stochastic_optimization": (
            "optimize",
            (
                "optimize(objective, initial_params, data, seed, *, num_steps: int = 100, "
                "step_size: float = 0.001, beta1: float = 0.9, beta2: float = 0.999, "
                "epsilon: float = 1e-8) -> OptimizationResult"
            ),
        ),
    }
    default_name, default_sig = defaults.get(paradigm_id, ("select_batch", "select_batch(model, x_unlabeled, batch_size, seed) -> List[int]"))
    classification = {"id": paradigm_id}
    if task_domain:
        classification["task_domain"] = task_domain
    return {
        "comparison": {
            "classification": classification,
            "pluggable_component": {
                "name": pluggable_name or default_name,
                "signature": signature or default_sig,
                "seed_param": "seed",
            },
        },
        "critical_requirements": {
            "model": {"specific_features": []},
            "required_model_methods": [],
        },
    }


def test_pack_role_view_and_build_plan_parity_for_migrated_al_surfaces():
    """recentering §5.2/5.3 evidence: the effective pack carries the migrated
    role-view surfaces, while the spec-derived build plan preserves the
    producer ownership split without copying package_manifest into the node."""
    from scripts import taxonomy as tx
    from scripts.build_plan import load_build_plan

    for pid in AL_POPULATED_PARADIGMS:
        pack = tx.load_pack(pid, ROOT)
        assert pack is not None

        node_pluggable = pack["pluggable_component"]
        assert {"name", "return_type", "signature_template", "contract"} <= set(node_pluggable)
        assert pack["notebook_layout"]["sections"]

        plan = load_build_plan(_served_al_spec(pid), ROOT)
        assert plan is not None
        plan_manifest = plan["package_manifest"]
        plan_owner = {
            entry["path"]: entry.get("produced_by")
            for entry in plan_manifest["files"]
        }
        assert plan_owner["method/method.py"] == "method_coder"
        assert plan_owner["method/model.py"] == "architecture_coder"
        assert plan_owner["requirements.txt"] == "init_finalizer"


def _symbol_names(manifest: dict, path: str) -> list[str]:
    for entry in manifest.get("files") or []:
        if entry.get("path") == path:
            return [
                sym.get("name")
                for sym in (entry.get("public_symbols") or [])
                if isinstance(sym, dict) and sym.get("name")
            ]
    return []


def _symbol_count(manifest: dict, path: str, kind: str) -> int:
    for entry in manifest.get("files") or []:
        if entry.get("path") == path:
            return sum(
                1
                for sym in (entry.get("public_symbols") or [])
                if isinstance(sym, dict) and sym.get("kind") == kind
            )
    return 0


def test_phase3_pluggable_and_build_plan_parity_for_served_nodes():
    """Phase 3.1-3.5: the new served nodes expose the runtime contract from
    taxonomy/build_plan while preserving guide-era producer ownership and
    structural schema expectations."""
    from scripts import taxonomy as tx
    from scripts.build_plan import load_build_plan

    pluggable_keys = {"name", "return_type", "signature_template", "contract"}
    for pid in PHASE3_POPULATED_PARADIGMS:
        node_pluggable = tx.load_pluggable_component_contract(paradigm_id=pid)
        assert pluggable_keys <= set(node_pluggable), f"{pid} pluggable contract incomplete"

        plan = load_build_plan(_served_spec(pid), ROOT)
        assert plan is not None
        plan_manifest = plan["package_manifest"]
        paths = [f["path"] for f in plan_manifest["files"]]
        assert "method/method.py" in paths
        assert "method/model.py" in paths
        owners = {f["path"]: f.get("produced_by") for f in plan_manifest["files"]}
        if "requirements.txt" in owners:
            assert owners["requirements.txt"] == "init_finalizer"
        assert _symbol_names(plan_manifest, "method/method.py")
        assert _symbol_count(plan_manifest, "method/model.py", "class") >= 1

        plan_required = plan["arch_contract_requirements"]["required_blocks"]
        assert plan_required, f"{pid} missing arch-contract requirements"


def test_phase3_task_domain_overlay_for_cross_modal_kd():
    """Phase 3.3: cross-modal KD proves method-family lookup and task-domain
    overlays are independent axes."""
    from scripts import taxonomy as tx

    tax = tx.load_taxonomy(ROOT)
    variant, domain = tx.assert_classification_tuple(
        tax, "TE-TS/knowledge_distillation/detection/cross_modal", "AD"
    )
    assert variant.taxonomy_id == "TE-TS/knowledge_distillation/detection/cross_modal"
    assert domain == ("AD", None)

    _, pc_domain = tx.assert_classification_tuple(
        tax, "cross_modal", "point_cloud_detection"
    )
    assert pc_domain == ("PC", "point_cloud_detection")

    ad_ids = {
        c["id"]
        for c in tx.effective_node(tax, "cross_modal", "AD")["semantic_checks"]
    }
    pc_ids = {
        c["id"]
        for c in tx.effective_node(tax, "cross_modal", "PC")["semantic_checks"]
    }
    assert "KD-xmodal-AD-scene-pairing" in ad_ids
    assert "KD-xmodal-PC-teacher-input" in pc_ids
    assert "KD-xmodal-AD-scene-pairing" not in pc_ids


def test_phase3_served_nodes_no_longer_require_field_guide_for_build_plan():
    from scripts.build_plan import load_build_plan

    for pid in PHASE3_POPULATED_PARADIGMS:
        spec = _served_spec(pid)
        assert "field_guide_path" not in spec["comparison"]["classification"]
        assert load_build_plan(spec, ROOT) is not None


def test_build_plan_derives_required_model_methods_from_spec():
    """recentering §5.3: the binding model interface is spec-derived, with the
    node's interface_hint only acting as the prior."""
    from scripts.build_plan import load_build_plan

    plan = load_build_plan(_served_al_spec(
        "active_learning/batch_acquisition",
        required_model_methods=[{
            "name": "forward_with_embedding",
            "signature": "forward_with_embedding(self, x: Tensor) -> tuple[Tensor, Tensor]",
        }],
    ), ROOT)
    assert plan is not None
    model_file = next(
        entry for entry in plan["package_manifest"]["files"]
        if entry["path"] == "method/model.py"
    )
    required_methods = model_file["public_symbols"][0]["required_methods"]
    assert "forward(self, x: torch.Tensor) -> torch.Tensor" in required_methods
    assert "forward_with_embedding(self, x: Tensor) -> tuple[Tensor, Tensor]" in required_methods


def test_served_al_validators_do_not_require_field_guide_path(tmp_path):
    """Phase 2.5/2.6: served AL consumers use the build plan and node id, so a
    missing/bogus field_guide_path no longer blocks the deterministic package
    validators. Reserved paradigms still fall back to legacy guide paths."""
    from scripts.finalize_package_init import finalize
    from scripts.validate_arch_contract import validate as validate_arch_contract
    from scripts.validate_architecture_coder_output import validate as validate_architecture
    from scripts.validate_method_coder_output import validate as validate_method

    run_dir = tmp_path / "run"
    method_dir = run_dir / "method"
    pipeline_dir = run_dir / ".pipeline"
    method_dir.mkdir(parents=True)
    pipeline_dir.mkdir()
    (pipeline_dir / "paper_map.json").write_text(
        '{"schema_version": "1.0", "elements": []}\n',
        encoding="utf-8",
    )
    def _tensor(dtype: str, *dimensions: str) -> dict:
        return {
            "kind": "tensor",
            "dtype": dtype,
            "dimensions": [{"dimension": name} for name in dimensions],
        }

    selected_indices = {
        "kind": "opaque",
        "type_description": "list[int] of selected pool indices",
        "reason": "Schema 2 does not yet synthesize Python list containers.",
    }
    contract = {
        "schema_version": "2.0.0",
        "paradigm_id": "active_learning",
        "dimensions": {
            name: {"expression": {"kind": "literal", "value": value}}
            for name, value in {
                "fixture_batch_size": 2,
                "training_example_count": 4,
                "pool_example_count": 5,
                "test_example_count": 3,
                "feature_count": 4,
                "class_count": 2,
            }.items()
        },
        "data_loader": {"load_data_returns": {
            "x_pool": _tensor("float32", "pool_example_count", "feature_count"),
            "y_pool": _tensor("int64", "pool_example_count"),
            "x_test": _tensor("float32", "test_example_count", "feature_count"),
            "y_test": _tensor("int64", "test_example_count"),
        }},
        "architecture": {"model": {
            "class_name": "TinyModel",
            "constructor_args": {
                "input_dim": {"dimension": {"dimension": "feature_count"}},
                "n_classes": {"dimension": {"dimension": "class_count"}},
            },
            "forward": {
                "input": {"x": _tensor(
                    "float32", "fixture_batch_size", "feature_count"
                )},
                "output": _tensor(
                    "float32", "fixture_batch_size", "class_count"
                ),
            },
        }},
        "pluggable_component": {
            "name": "select_batch",
            "input": {"x_unlabeled": _tensor(
                "float32", "pool_example_count", "feature_count"
            )},
            "output": selected_indices,
        },
        "training_loop": {
            "function_name": "train_from_scratch",
            "input": {
                "x_train": _tensor(
                    "float32", "training_example_count", "feature_count"
                ),
                "y_train": _tensor("int64", "training_example_count"),
            },
        },
    }
    (pipeline_dir / "arch_contract.json").write_text(
        json.dumps(contract), encoding="utf-8"
    )
    (method_dir / "data.py").write_text(
        """import os
from typing import Tuple
import torch

def load_data(path: str | os.PathLike | None = None, *, pool_size: int = 5000, n_test: int = 1000, seed: int = 0) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return torch.zeros(pool_size, 4), torch.zeros(pool_size), torch.zeros(n_test, 4), torch.zeros(n_test)
""",
        encoding="utf-8",
    )
    (method_dir / "model.py").write_text(
        """import torch
import torch.nn as nn

class TinyModel(nn.Module):
    def __init__(self, input_dim: int = 4, n_classes: int = 2):
        super().__init__()
        self.fc = nn.Linear(input_dim, n_classes)

    def forward(self, x):
        return self.fc(x)
""",
        encoding="utf-8",
    )
    (method_dir / "training.py").write_text(
        """import torch
from .model import TinyModel

def build_model(input_dim: int, n_classes: int, **arch_kwargs) -> TinyModel:
    return TinyModel(input_dim, n_classes)

def train_from_scratch(model: TinyModel, x_train: torch.Tensor, y_train: torch.Tensor, *, learning_rate: float, max_epochs: int, train_until_accuracy: float, seed: int) -> TinyModel:
    torch.manual_seed(seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    return model
""",
        encoding="utf-8",
    )
    (method_dir / "method.py").write_text(
        """import numpy as np

def select_batch(model, x_unlabeled, batch_size, seed):
    rng = np.random.default_rng(seed)
    if len(x_unlabeled) == 0:
        return []
    size = min(batch_size, len(x_unlabeled))
    return list(rng.choice(len(x_unlabeled), size=size, replace=False))
""",
        encoding="utf-8",
    )

    spec = _served_al_spec("active_learning")
    assert validate_architecture(spec, run_dir, ROOT) == []
    method_errors, method_warnings = validate_method(spec, run_dir, ROOT)
    assert method_errors == []
    assert validate_arch_contract(spec, run_dir) == []
    assert finalize(tmp_path / "missing-spec.json", run_dir) == 1

    spec_path = pipeline_dir / "method_spec.json"
    spec_path.write_text(json.dumps(spec), encoding="utf-8")
    assert finalize(spec_path, run_dir) == 0


# ---------------------------------------------------------------------------
# Configurable-system: context bucket views + per-bucket lint
# (the configurable node schema design note (internal, not shipped))
# ---------------------------------------------------------------------------
def test_node_bucket_views_assemble_al_context():
    """The four bucket accessors surface the AL node's fields; the two
    cross-bucket fields (semantic_checks, smoke_economics) appear in two views."""
    from scripts import taxonomy as tx

    node = tx.serves("active_learning")
    assert node is not None

    impl = tx.node_implementation(node)
    assert "pluggable_component" in impl
    assert "templates_dir" in impl            # surfaced from scaffold_hints

    params = tx.node_parameters(node)
    assert "smoke_economics" in params

    expertise = tx.node_expertise(node)
    assert "fingerprint" in expertise
    assert "semantic_checks" in expertise     # cross-bucket: also in testing

    testing = tx.node_testing(node)
    assert "semantic_checks" in testing
    assert "smoke_economics" in testing       # cross-bucket: also in parameters

    # The cross-bucket fields are the same content in both views (not duplicated).
    assert expertise["semantic_checks"] == testing["semantic_checks"]
    assert params["smoke_economics"] == testing["smoke_economics"]


def test_bucket_coverage_reports_al_provision():
    """bucket_coverage reports which context AL provides per bucket; every bucket
    has at least one field, and notebook_layout joins implementation in Phase 1.6."""
    from scripts import taxonomy as tx

    cov = tx.bucket_coverage(tx.serves("active_learning"))
    assert set(cov) == {"expertise", "implementation", "parameters", "testing"}
    for bucket in cov:
        assert cov[bucket], f"AL provides no {bucket} context"
    assert "pluggable_component" in cov["implementation"]
    assert "notebook_layout" in cov["implementation"]  # migrated in Phase 1.6


def _populated_node_raw(extra_fields: dict) -> dict:
    """A minimal lint-clean populated node, plus `extra_fields` to exercise lint."""
    body = {
        "name": "Demo", "taxonomy_id": "TE-TS/demo_method",
        "status": "populated", "legacy_paradigm": "demo_method",
        "fingerprint": {"what_it_is": "x"},
    }
    body.update(extra_fields)
    return {
        "schema_version": "2.0",
        "method_roots": {"TE": {"name": "T", "families": {
            "TE-TS": {"name": "TS", "variants": {"demo_method": body}}}}},
        "task_domains": {},
    }


def test_notebook_layout_lint_rejects_missing_sections():
    from scripts.taxonomy import load_taxonomy_uncached
    from scripts.validate_taxonomy import lint

    tax = load_taxonomy_uncached(ROOT, _populated_node_raw(
        {"notebook_layout": {"description": "no sections here"}}))
    assert "notebook_layout_sections_missing" in {d.code for d in lint(tax)}


def test_ssot_lint_rejects_the_retired_arch_contract_schema_key():
    """B-11 item 6: the SSOT refuses scaffold_hints.arch_contract_schema
    through the SAME rule the pack-authoring validator enforces (called, not
    copied — two authorities over one rule is the R2C-049 pattern). The
    committed SSOT carries zero occurrences, so the guard starts green."""
    from scripts.taxonomy import load_taxonomy_uncached
    from scripts.validate_taxonomy import lint

    tax = load_taxonomy_uncached(ROOT, _populated_node_raw(
        {"scaffold_hints": {"arch_contract_schema": {"blocks": {}}}}))
    assert "retired_arch_contract_schema" in {d.code for d in lint(tax)}


def test_paradigm_extras_lint_rejects_bad_shape():
    from scripts.taxonomy import load_taxonomy_uncached
    from scripts.validate_taxonomy import lint

    tax = load_taxonomy_uncached(ROOT, _populated_node_raw(
        {"paradigm_extras": {"mc_samples": {"value": 20}}}))  # missing reasoning_template
    assert "paradigm_extras_entry_shape" in {d.code for d in lint(tax)}


def test_scenario_assumption_dimension_lint_requires_capture_guidance():
    from scripts.taxonomy import load_taxonomy_uncached
    from scripts.validate_taxonomy import lint

    tax = load_taxonomy_uncached(ROOT, _populated_node_raw({
        "scenario_assumption_dimensions": [{
            "id": "obstacle_geometry",
            "display_name": "Obstacle geometry",
            "description": "Permitted obstacle shapes.",
            # normalized_value_guidance deliberately absent
        }],
    }))
    assert "scenario_assumption_dimension_field" in {
        d.code for d in lint(tax)
    }


def test_well_formed_bucket_fields_lint_clean():
    """Correctly-shaped notebook_layout + paradigm_extras introduce no errors."""
    from scripts.taxonomy import load_taxonomy_uncached
    from scripts.validate_taxonomy import lint

    tax = load_taxonomy_uncached(ROOT, _populated_node_raw({
        "notebook_layout": {"sections": [{"id": "title", "title": "T", "kind": "agnostic"}]},
        "paradigm_extras": {"mc_samples": {"value": 20, "reasoning_template": "use {value}"}},
    }))
    new_codes = {"notebook_layout_malformed", "notebook_layout_sections_missing",
                 "notebook_layout_section_id_missing", "paradigm_extras_malformed",
                 "paradigm_extras_entry_shape"}
    assert not ({d.code for d in lint(tax)} & new_codes)


def test_effective_blocks_are_served_from_taxonomy_node():
    """Phase 4.1: effective consumer blocks render from taxonomy only."""
    from scripts import taxonomy as tx

    blocks = tx.effective_blocks(None, "active_learning/bayesian")
    assert blocks["pluggable_component"]["name"] == "select_batch"
    assert blocks["notebook_layout"]["sections"]
    assert blocks["stage_review_focus"]


# ---------------------------------------------------------------------------
# Phase 1.5 — the AL build-contract surface (pluggable_component + templates_dir)
# is node-served.
# ---------------------------------------------------------------------------
def test_al_pluggable_component_contract_parity():
    """1.5.3: load_pluggable_component_contract serves the node's block for AL."""
    from scripts import taxonomy as tx

    for pid in AL_POPULATED_PARADIGMS:
        node = tx.load_pluggable_component_contract(paradigm_id=pid)
        assert node, f"{pid} served an empty pluggable_component"
        assert {"name", "return_type", "signature_template", "contract"} <= set(node)
        assert node.get("rationale"), f"{pid} lost its pluggable_component rationale"


def test_al_pluggable_component_threads_x_labeled_only_for_bayesian():
    """bayesian overrides the contract to thread x_labeled; batch_acquisition
    inherits the parent (no x_labeled) — the override-vs-inherit proof."""
    from scripts import taxonomy as tx

    parent = tx.load_pluggable_component_contract(paradigm_id="active_learning")
    batch = tx.load_pluggable_component_contract(paradigm_id="active_learning/batch_acquisition")
    bayes = tx.load_pluggable_component_contract(paradigm_id="active_learning/bayesian")
    assert parent["contract"]["fixed_positional_args"] == ["model", "x_unlabeled", "batch_size"]
    assert batch["contract"]["fixed_positional_args"] == ["model", "x_unlabeled", "batch_size"]
    assert bayes["contract"]["fixed_positional_args"] == ["model", "x_unlabeled", "x_labeled", "batch_size"]
    assert "x_labeled" in bayes["signature_template"]
    assert "x_labeled" not in parent["signature_template"]
    assert "x_labeled" not in batch["signature_template"]


def test_templates_dir_resolves_through_served_node():
    """resolve_templates_dir returns a real templates/ path from served nodes."""
    from scripts import taxonomy as tx

    for pid in POPULATED_PARADIGMS:
        via_node = tx.resolve_templates_dir(paradigm_id=pid, repo_root=ROOT)
        assert via_node is not None and via_node.is_dir(), f"no templates dir for {pid}"


# ---------------------------------------------------------------------------
# Phase 1.6 — the AL `notebook_layout` build-contract surface is node-served.
# Newer Phase 3 nodes serve the validator-critical section contract too.
# ---------------------------------------------------------------------------
def test_al_notebook_layout_parity():
    """1.6.3: load_notebook_layout serves the node's block for AL."""
    from scripts import taxonomy as tx

    for pid in AL_POPULATED_PARADIGMS:
        node = tx.load_notebook_layout(paradigm_id=pid)
        assert node, f"{pid} served an empty notebook_layout"
        assert node.get("sections"), f"{pid} notebook_layout has no sections"


def test_load_notebook_layout_resolves_by_taxonomy_id_only():
    """A served paradigm resolves its notebook_layout directly from the node."""
    from scripts import taxonomy as tx

    by_id = tx.load_notebook_layout(paradigm_id="active_learning")
    assert by_id.get("sections")


def _section_contract(layout: dict) -> list[tuple[str | None, str | None, tuple[tuple[str | None, str | None], ...]]]:
    return [
        (
            section.get("id"),
            section.get("title"),
            tuple(
                (sub.get("id"), sub.get("title"))
                for sub in section.get("subsections") or []
                if isinstance(sub, dict)
            ),
        )
        for section in layout.get("sections") or []
        if isinstance(section, dict)
    ]


def test_phase3_new_served_notebook_layout_section_contracts():
    """Phase 3.6/3.8: newly served non-AL nodes expose the notebook section
    contract from the taxonomy node, so validator paths do not need guide-path
    carriers for those nodes."""
    from scripts import taxonomy as tx

    for pid in (
        "motion_planning",
        "motion_planning/sampling_based",
        "motion_planning/optimization_based",
        "stochastic_optimization",
    ):
        node = tx.load_notebook_layout(paradigm_id=pid)
        assert node and node.get("sections"), f"{pid} served no notebook_layout"
        assert _section_contract(node)


# ---------------------------------------------------------------------------
# Phase 5.7 — AL's `paradigm_extras` smoke defaults are now node-served.
# load_paradigm_extra serves the AL sub-variants' (value, reasoning_template) from
# the node; `derive_params.KNOWN_EXTRAS_BY_PARADIGM` is now only a defensive
# fallback for provisional/future ids. The
# reasoning_template keeps its `{value}`/`{paper_value}` .format() slots verbatim.
# ---------------------------------------------------------------------------
# The exact (value, reasoning_template) tuples the AL nodes must serve — snapshotted
# from the taxonomy SSOT so this test catches accidental wording/value drift.
_EXPECTED_AL_EXTRAS = {
    ("active_learning/bayesian", "mc_samples"): (
        20,
        "Paper uses {paper_value} MC dropout samples. We use {value} — the "
        "bayesian-AL taxonomy minimum for BALD's posterior estimate to be "
        "meaningful (Gal et al. recommend T ≥ 20). Larger T at paper scale "
        "gives a tighter estimate but dominates runtime at smoke scale.",
    ),
    ("active_learning/batch_acquisition", "core_set_size"): (
        100,
        "Paper uses a {paper_value}-point core-set. We use {value} — smaller "
        "core-set runs Stage 1 fast and still seeds the per-round acquisition "
        "with a diverse initial labeled set.",
    ),
    ("active_learning/batch_acquisition", "batch_returns"): (
        200,
        "Paper sets b at ~3 × b' (the batch_size). We use 2 × 100 = {value} — "
        "enough to give the geometric ranking room without spending compute "
        "on candidates the ranker will reject.",
    ),
}
_EXPECTED_KD_EXTRAS = {
    ("knowledge_distillation", "temperature"): (
        4.0,
        "Paper uses T={paper_value} for logit softening. We use T={value} — "
        "Hinton et al.'s typical KD temperature; reasonable default when the "
        "smoke run doesn't otherwise depend on the exact value.",
    ),
    ("knowledge_distillation", "alpha"): (
        0.5,
        "Paper uses α={paper_value} for the supervised-vs-distillation loss "
        "weight. We use α={value} — balanced (50% supervised + 50% KD), a "
        "common default that lets the distillation signal show on a small "
        "dataset without overwhelming the supervised term.",
    ),
}


def test_al_paradigm_extras_served_verbatim():
    """5.7: load_paradigm_extra returns the AL sub-variants' smoke defaults from the
    node, byte-identical (incl. {value}/{paper_value} .format() slots) to the values
    that live in the taxonomy SSOT."""
    from scripts import taxonomy as tx

    for (pid, name), expected in _EXPECTED_AL_EXTRAS.items():
        assert tx.load_paradigm_extra(pid, name) == expected, f"extra drift for {pid}::{name}"

    for (pid, name), expected in _EXPECTED_KD_EXTRAS.items():
        assert tx.load_paradigm_extra(pid, name) == expected, f"extra drift for {pid}::{name}"


def test_paradigm_extras_negative_space():
    """5.7: node inheritance reproduces the legacy longest-prefix-match exactly —
    a sub-variant never sees a sibling's extra, the parent declares none, and a
    reserved paradigm (or an unknown name) returns None so the consumer falls back."""
    from scripts import taxonomy as tx

    assert tx.load_paradigm_extra("active_learning/bayesian", "core_set_size") is None
    assert tx.load_paradigm_extra("active_learning/batch_acquisition", "mc_samples") is None
    assert tx.load_paradigm_extra("active_learning", "mc_samples") is None      # parent has none
    assert tx.load_paradigm_extra("knowledge_distillation", "temperature") is not None
    assert tx.load_paradigm_extra("motion_planning", "temperature") is None  # no such node extra
    assert tx.load_paradigm_extra("active_learning/bayesian", "no_such_extra") is None
    assert tx.load_paradigm_extra("", "mc_samples") is None
    assert tx.load_paradigm_extra(None, "mc_samples") is None


def test_lookup_known_extra_prefers_node_over_python_table():
    """5.7: derive_params._lookup_known_extra resolves served paradigms from the node;
    no migrated entries remain in the Python fallback table."""
    from scripts import taxonomy as tx
    from scripts.derive_params import KNOWN_EXTRAS_BY_PARADIGM, _lookup_known_extra

    # AL extras no longer live in the Python table (node is the SSOT)...
    assert not any(k.startswith("active_learning") for k in KNOWN_EXTRAS_BY_PARADIGM)
    # ...yet _lookup_known_extra still resolves them (via the node).
    assert _lookup_known_extra("active_learning/bayesian", "mc_samples") == (
        tx.load_paradigm_extra("active_learning/bayesian", "mc_samples")
    )
    # KD also resolves from the node after Phase 3.
    assert _lookup_known_extra("knowledge_distillation", "temperature") == (
        tx.load_paradigm_extra("knowledge_distillation", "temperature")
    )


# ---------------------------------------------------------------------------
# Phase 5.7b — AL's `model_defaults` (the system_inferred fallback arm of
# derive_params._add_model_params) are now node-served. load_model_default serves
# (value, reasoning_template); the node owns ONLY the fallback (the source=paper /
# CNN-substitution arms stay evidence-driven in Python). hidden_dim is AL-wide
# (parent active_learning, inherited by both sub-variants); dropout_rate is
# bayesian-only. The reasoning_template keeps its {value} .format() slot verbatim.
# ---------------------------------------------------------------------------
_EXPECTED_AL_MODEL_DEFAULTS = {
    ("active_learning", "hidden_dim"): (
        256,
        "Paper states no data-type-keyed MLP width. We use hidden_dim={value} "
        "as the taxonomy-typical AL convention.",
    ),
    ("active_learning/bayesian", "dropout_rate"): (
        0.5,
        "Bayesian-AL taxonomy recommends dropout in [0.25, 0.5] "
        "(Gal & Ghahramani 2016); {value} is the taxonomy-typical "
        "default. Paper does not specify a dropout rate.",
    ),
}

# The exact strings derive_params._add_model_params should produce from the
# node-served templates.
_EXPECTED_HIDDEN_REASONING = (
    "Paper states no data-type-keyed MLP width. We use hidden_dim=256 "
    "as the taxonomy-typical AL convention."
)
_EXPECTED_DROPOUT_REASONING = (
    "Bayesian-AL taxonomy recommends dropout in [0.25, 0.5] "
    "(Gal & Ghahramani 2016); 0.5 is the taxonomy-typical "
    "default. Paper does not specify a dropout rate."
)


def test_al_model_defaults_served_verbatim():
    """5.7b: load_model_default returns the AL nodes' system_inferred fallbacks
    byte-identically (incl. the {value} slot), and formatting reproduces the
    expected rendered prose."""
    from scripts import taxonomy as tx

    for (pid, name), expected in _EXPECTED_AL_MODEL_DEFAULTS.items():
        assert tx.load_model_default(pid, name) == expected, f"model_default drift for {pid}::{name}"

    # Formatting the served template reproduces the expected rendered reasoning.
    hv, ht = tx.load_model_default("active_learning", "hidden_dim")
    assert ht.format(value=hv) == _EXPECTED_HIDDEN_REASONING
    dv, dt = tx.load_model_default("active_learning/bayesian", "dropout_rate")
    assert dt.format(value=dv) == _EXPECTED_DROPOUT_REASONING


def test_model_defaults_inheritance_and_negative_space():
    """5.7b: hidden_dim is AL-wide (parent + both sub-variants inherit the same
    256), dropout_rate is bayesian-only (batch_acquisition never sees it), and a
    non-AL paradigm / unknown name / empty id returns None so the consumer falls
    back to the Python literal."""
    from scripts import taxonomy as tx

    # hidden_dim inherited identically by every neural AL sub-variant.
    for pid in ("active_learning", "active_learning/bayesian", "active_learning/batch_acquisition"):
        assert tx.load_model_default(pid, "hidden_dim") == (
            tx.load_model_default("active_learning", "hidden_dim")
        )
    # dropout_rate lives only on the bayesian sub-variant.
    assert tx.load_model_default("active_learning/bayesian", "dropout_rate") is not None
    assert tx.load_model_default("active_learning/batch_acquisition", "dropout_rate") is None
    assert tx.load_model_default("active_learning", "dropout_rate") is None  # parent declares none
    # no node default / unknown name / empty id -> None (Python fallback owns these).
    assert tx.load_model_default("knowledge_distillation", "hidden_dim") is None
    assert tx.load_model_default("motion_planning", "hidden_dim") is None
    assert tx.load_model_default("active_learning", "no_such_param") is None
    assert tx.load_model_default("", "hidden_dim") is None
    assert tx.load_model_default(None, "hidden_dim") is None


# ---------------------------------------------------------------------------
# Phase 2.2 / 4.1 — the taxonomy API's `effective_blocks` composite renders the
# migrated surfaces (common_smoke_bugs + stage_review_focus) from served nodes.
# ---------------------------------------------------------------------------
def test_effective_blocks_node_aware_for_smoke_bugs_and_stage_review():
    from scripts import taxonomy as tx

    al_node = tx.effective_blocks(None, "active_learning")
    assert "AL-SB-labels-out-of-range" in al_node["common_smoke_bugs"]
    assert all(
        isinstance(v, dict) and v.get("status") in {"seed", "observed"}
        for v in al_node["common_smoke_bugs"].values()
    )
    assert al_node["stage_review_focus"]

    for pid in POPULATED_PARADIGMS:
        blocks = tx.effective_blocks(None, pid)
        assert blocks.get("stage_review_focus"), f"{pid} lost stage-review focus"


# ---------------------------------------------------------------------------
# Phase 2.1 — the 14 pipeline-wide checks live ONCE in the top-level
# universal_checks block. _node_stage_review_focus composes them into a served
# node's stage-review (node-wins by id): AL overrides the ones it specializes
# (keeping probes/observed status) and inherits the rest from the block.
# ---------------------------------------------------------------------------
EXPECTED_UNIVERSAL_IDS = {
    "core_method_summary_matches_paper", "critical_requirements_essential_features",
    "essential_features_implemented_not_just_annotated", "architecture_substitution_documented",
    "training_protocol_matches_paper", "paper_element_implementations_match_paper",
    "paper_scale_faithful_implementation", "simplifications_flagged",
    "pluggable_function_actually_does_the_method", "random_seeds_threaded",
    "paper_source_values_match_paper", "system_default_reasoning_explains_deviation",
    "notebook_narrative_matches_code", "paper_citations_accurate",
}


def test_universal_checks_block_is_the_single_canonical_set():
    from scripts import taxonomy as tx

    tax = tx.load_taxonomy()
    ids = [c["id"] for c in tax.universal_checks]
    assert set(ids) == EXPECTED_UNIVERSAL_IDS  # exactly the 14, defined once
    assert len(ids) == len(set(ids)) == 14
    # every entry satisfies the node semantic_check shape the lint enforces
    for c in tax.universal_checks:
        assert {"id", "stage", "check", "silent_failure", "severity", "status"} <= set(c)
        assert c["severity"] in {"error", "warning"} and c["status"] in {"seed", "observed"}


def test_node_stage_review_composes_universals_with_node_override():
    from scripts import taxonomy as tx

    # Across all review stages, the served AL block = universals (for that stage) +
    # AL-specific, with node-wins on id collisions and NO duplicate ids.
    composed_ids: set[str] = set()
    for stage in REVIEW_STAGES:
        block = tx.merged_stage_review_focus(None, "active_learning", stage)
        ids = [c["id"] for c in block.get("semantic_checks", [])]
        assert len(ids) == len(set(ids)), f"duplicate ids at {stage}: {ids}"
        composed_ids.update(ids)
    # All 14 universals surface somewhere in AL's composed review.
    assert EXPECTED_UNIVERSAL_IDS <= composed_ids
    # AL's specialization of a universal OVERRIDES the generic (keeps its probe + observed).
    s2c = tx.merged_stage_review_focus(None, "active_learning", "stage_2c_method")
    pluggable = next(c for c in s2c["semantic_checks"] if c["id"] == "pluggable_function_actually_does_the_method")
    assert pluggable.get("probe") == "claims.contribution_floor"
    assert pluggable.get("status") == "observed"
    # A genuinely AL-specific check (no universal id) is retained.
    assert any(c["id"] == "AL-method-pluggable-no-hidden-state" for c in s2c["semantic_checks"])


def test_universal_checks_inherited_by_a_node_that_does_not_override():
    """A populated node that does NOT declare a universal's id inherits the generic
    block entry verbatim — AL never declared `random_seeds_threaded`, so it surfaces
    from the block."""
    from scripts import taxonomy as tx

    s2c = tx.merged_stage_review_focus(None, "active_learning", "stage_2c_method")
    rs = next(c for c in s2c["semantic_checks"] if c["id"] == "random_seeds_threaded")
    block_rs = next(c for c in tx.load_taxonomy().universal_checks if c["id"] == "random_seeds_threaded")
    assert rs["check"] == block_rs["check"]  # inherited generic text, not a node override
    assert rs["severity"] == "error"


# ---------------------------------------------------------------------------
# bev-distill 2026-07-02 F001/F002: static-plan paradigms (KD family, ViT,
# DA, MP) never merged the spec's paper-derived required_model_methods into
# the manifest's model classes, so the coder implemented only the paradigm
# conventions, the spec interface was enforced nowhere, and the stage-4
# fidelity review demoted the delivery. The first fix broadcast the spec
# methods onto EVERY class — which corrupted multi-class requirements
# (overnight 2026-07-08: iDb-RRT's step_jacobian demanded of the collision
# model; detr-distill's probe methods demanded of student AND teacher; both
# runs failed 2.b on correct-per-manifest output). Current semantics:
# single-class manifests merge into the class (legacy AL behavior);
# multi-class manifests route spec methods to the entry-level
# `any_class_required_methods` bucket, which the 2.b gate enforces as
# "at least one public class defines it".
# ---------------------------------------------------------------------------


def _kd_spec_with_required_methods(methods):
    return {
        "comparison": {
            "classification": {
                "id": "knowledge_distillation/detection/cross_modal"},
            "pluggable_component": {
                "name": "compute_distillation_loss",
                "signature": ("compute_distillation_loss(student, teacher, "
                              "batch, seed) -> torch.Tensor"),
            },
        },
        "critical_requirements": {
            "required_model_methods": [
                {"name": s.split("(", 1)[0], "signature": s} for s in methods
            ],
        },
    }


def _model_class_symbols(plan):
    for entry in plan["package_manifest"]["files"]:
        if entry["path"] == "method/model.py":
            return entry["public_symbols"]
    raise AssertionError("no model.py entry in manifest")


def _model_entry(plan):
    for entry in plan["package_manifest"]["files"]:
        if entry["path"] == "method/model.py":
            return entry
    raise AssertionError("no model.py entry in manifest")


def test_static_plan_routes_spec_methods_to_any_class_bucket_for_multi_class():
    from scripts.build_plan import load_build_plan

    spec = _kd_spec_with_required_methods([
        "get_bev_features(self, batch) -> Tensor",
        "get_instance_features(self, batch) -> Tuple[Tensor, Tensor, Tensor]",
    ])
    plan = load_build_plan(spec, ROOT)
    entry = _model_entry(plan)
    bucket_names = [
        s.split("(", 1)[0] for s in entry["any_class_required_methods"]
    ]
    assert "get_bev_features" in bucket_names
    assert "get_instance_features" in bucket_names
    # Per-class requirements carry ONLY the paradigm conventions — the spec
    # methods must not be broadcast onto every class (the overnight
    # 2026-07-08 iDb-RRT / detr-distill corruption).
    for symbol in _model_class_symbols(plan):
        names = [s.split("(", 1)[0] for s in symbol["required_methods"]]
        assert "get_bev_features" not in names, symbol["name"]
        assert "get_instance_features" not in names, symbol["name"]
        # The paradigm conventions stay.
        assert "forward_with_bev_features" in names, symbol["name"]


def test_static_plan_single_class_merges_into_the_class():
    from scripts.build_plan import load_build_plan

    # vision_transformer's manifest declares exactly one model class, so the
    # legacy merge-into-the-class behavior is pinned unchanged.
    spec = _kd_spec_with_required_methods([
        "get_token_groups(self, x) -> Tensor",
    ])
    spec["comparison"]["classification"]["id"] = "vision_transformer"
    plan = load_build_plan(spec, ROOT)
    symbols = _model_class_symbols(plan)
    assert len(symbols) == 1
    names = [s.split("(", 1)[0] for s in symbols[0]["required_methods"]]
    assert "get_token_groups" in names
    assert "any_class_required_methods" not in _model_entry(plan)


def test_static_plan_bucket_dedupes_against_class_conventions():
    from scripts.build_plan import load_build_plan

    # A spec that re-declares a convention method (different signature text)
    # must not re-require it via the bucket.
    spec = _kd_spec_with_required_methods([
        "forward_with_bev_features(self, batch) -> Tuple[Dict, Tensor]",
    ])
    plan = load_build_plan(spec, ROOT)
    entry = _model_entry(plan)
    bucket = entry.get("any_class_required_methods") or []
    assert bucket == []
    for symbol in _model_class_symbols(plan):
        names = [s.split("(", 1)[0] for s in symbol["required_methods"]]
        assert names.count("forward_with_bev_features") == 1


def test_static_plan_without_spec_methods_keeps_templates_verbatim():
    from scripts.build_plan import load_build_plan

    spec = _kd_spec_with_required_methods([])
    plan = load_build_plan(spec, ROOT)
    assert "any_class_required_methods" not in _model_entry(plan)
    for symbol in _model_class_symbols(plan):
        names = [s.split("(", 1)[0] for s in symbol["required_methods"]]
        assert names == ["forward", "forward_with_bev_features"]

def test_params_derivation_lint_rejects_bad_entries():
    from scripts.taxonomy import load_taxonomy_uncached
    from scripts.validate_taxonomy import lint

    tax = load_taxonomy_uncached(ROOT, _populated_node_raw(
        {"params_derivation": {
            "a": {"kind": "config_path"},                      # no reasoning
            "b": {"kind": "derived_statistic", "formula": "x"},  # no inputs
            "c": {"kind": "suppress"},                         # no reason
            "d": {"kind": "mystery"},                          # unknown kind
            "e": {"kind": "config_path", "reasoning": "cfg.",
                  "protocol_role": "wrong_role"},
            "f": {"kind": "suppress", "reason": "no.",
                  "protocol_role": "test_span"},
            "g": {"kind": "config_path", "reasoning": "cfg.",
                  "protocol_role": "context_length"},
            "h": {"kind": "config_path", "reasoning": "cfg.",
                  "protocol_role": "context_length"},
        }}))
    codes = {d.code for d in lint(tax)}
    assert "params_derivation_config_path_reasoning" in codes
    assert "params_derivation_inputs_missing" in codes
    assert "params_derivation_suppress_reason" in codes
    assert "params_derivation_kind" in codes
    assert "params_derivation_protocol_role" in codes
    assert "params_derivation_protocol_role_kind" in codes
    assert "params_derivation_protocol_role_duplicate" in codes


def test_params_derivation_lint_accepts_well_formed_block():
    from scripts.taxonomy import load_taxonomy_uncached
    from scripts.validate_taxonomy import lint

    tax = load_taxonomy_uncached(ROOT, _populated_node_raw(
        {"params_derivation": {
            "ontology_path": {
                "kind": "config_path", "reasoning": "cfg.",
                "protocol_role": "context_length",
            },
            "u": {"kind": "derived_statistic", "formula": "n * E",
                  "inputs": {"n": "spec.a.b", "E": "params.E"}},
            "hidden_dim": {"kind": "suppress", "reason": "no model."},
        }}))
    assert not {d.code for d in lint(tax)
                if d.code.startswith("params_derivation")}
