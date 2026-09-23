"""Arch-contract headroom: pack-declared `family_components` blocks.

Failure class: `family_component_rejected_by_universal_schema` — the
universal ArchContract skeleton (`extra='forbid'`) rejects a structurally
valid family-specific top-level component the family's own pack requires.
Two concrete cases fixed the abstraction
(the arch contract headroom design note (internal, not shipped), approved
2026-07-16):

  - ADAM (2026-07-08): stochastic_optimization's manifest required an
    `optimizer_state` block the schema forbade; the coder could not win, and
    the fix at the time hard-added a universal optional field.
  - SRL (2026-07-15): the RL family needs a `reward_function` component; the
    coder correctly emitted it, the schema rejected it, and the run halted
    at Stage 2.d twice (halt-judge: pipeline bug, this exact class).

The fix under test: one typed extension container
(`ArchContract.family_components`), legal names declared by the pack
(committed taxonomy node or run-local provisional pack), enforced in BOTH
directions by `scripts/validate_arch_contract.py`. `optimizer_state` is
deliberately NOT migrated onto the container.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from schemas.arch_contract import (
    ArchContract,
    FamilyComponentBlock,
    FamilyComponentEntry,
)
from scripts.validate_arch_contract import validate as validate_arch_contract

ROOT = Path(__file__).resolve().parents[1]

SRL_PARADIGM_ID = "single_agent_rl"


def _v2_literal_dimension(value: int) -> dict:
    return {"expression": {"kind": "literal", "value": value}}


def _v2_dimension(identity: str, symbol: str | None = None) -> dict:
    use = {"dimension": identity}
    if symbol is not None:
        use["display_symbol"] = symbol
    return use


def _v2_tensor(dtype: str, *dimensions: tuple[str, str | None]) -> dict:
    return {
        "kind": "tensor",
        "dtype": dtype,
        "dimensions": [_v2_dimension(identity, symbol)
                       for identity, symbol in dimensions],
    }


def _v2_opaque(type_description: str, reason: str) -> dict:
    return {
        "kind": "opaque",
        "type_description": type_description,
        "reason": reason,
    }


# ---------------------------------------------------------------------------
# Schema layer: the container model itself.
# ---------------------------------------------------------------------------


def test_family_component_block_single_slot_shape():
    """SRL's reward_function shape: type/shape/description on the block."""
    block = FamilyComponentBlock.model_validate({
        "type": "callable",
        "shape": "scalar",
        "description": "Norm-inducing reward the policy optimizes.",
    })
    assert block.entries is None


def test_family_component_block_multi_slot_entries():
    """The optimizer_state shape: a named entries map, one slot per entry
    (the container must be able to express the ADAM 2026-07-08 case even
    though optimizer_state itself stays a universal field)."""
    block = FamilyComponentBlock.model_validate({
        "description": "Adam-style optimizer state.",
        "entries": {
            "step": {"type": "int", "description": "iteration counter"},
            "params": {"shape": "(P,)"},
        },
    })
    assert set(block.entries) == {"step", "params"}
    assert isinstance(block.entries["step"], FamilyComponentEntry)


def test_family_component_block_rejects_unknown_keys():
    """`extra='forbid'` typo safety inside the container's typed values."""
    with pytest.raises(Exception):
        FamilyComponentBlock.model_validate({"descripton": "typo"})
    with pytest.raises(Exception):
        FamilyComponentBlock.model_validate(
            {"entries": {"step": {"shpae": "(P,)"}}}
        )


def _srl_contract(family_components: dict | None) -> dict:
    """The SRL 2026-07-15 contract shape: a gap-family contract whose
    family-specific component rides in the extension container."""
    contract = {
        "schema_version": "1.0.0",
        "paradigm_id": SRL_PARADIGM_ID,
        "data_loader": {"load_data_returns": {
            "observations": "(N, obs_dim)", "env": "(opaque)"}},
        "architecture": {
            "policy_network": {
                "class_name": "PolicyNetwork",
                "forward": {"input": {"observation": "(B, obs_dim)"},
                            "output_type": "tensor",
                            "output_shape": "(B, action_dim)"},
            },
        },
        "pluggable_component": {
            "name": "compute_reward",
            "input_shapes": {"state": "(obs_dim,)", "action": "(action_dim,)"},
            "output_shape": "scalar",
        },
    }
    if family_components is not None:
        contract["family_components"] = family_components
    return contract


REWARD_FUNCTION_BLOCK = {
    "type": "callable",
    "shape": "scalar",
    "description": "Norm-inducing reward computed from (state, action).",
}

TYPED_REWARD_FUNCTION_COMPONENT = {
    "value": _v2_opaque(
        "callable(state, action) -> scalar reward",
        "Callable values are intentionally outside the bounded v2 synthesizer.",
    ),
}


def test_arch_contract_accepts_declared_container():
    contract = ArchContract.model_validate(
        _srl_contract({"reward_function": REWARD_FUNCTION_BLOCK}))
    assert set(contract.family_components) == {"reward_function"}


def test_arch_contract_still_rejects_undeclared_top_level_key():
    """The universal skeleton stays static: the exact pre-fix SRL emission (a
    NEW top-level key) is still a pydantic error. The container is the only
    extension point."""
    raw = _srl_contract(None)
    raw["reward_function"] = REWARD_FUNCTION_BLOCK
    with pytest.raises(Exception):
        ArchContract.model_validate(raw)


def test_container_invisible_when_unused():
    """Adjacent-good at the schema layer: a contract that never uses the
    container round-trips without the key (exclude_none)."""
    contract = ArchContract.model_validate(_srl_contract(None))
    assert "family_components" not in contract.model_dump(exclude_none=True)


# ---------------------------------------------------------------------------
# Declaration surface: provisional pack -> overlay -> build plan.
# ---------------------------------------------------------------------------

# The SRL 2026-07-13/15 shape: a gap pack populating the committed reserved
# `single_agent_rl` placeholder, now also declaring the component the family
# needs at the top level of its architecture contract.
SRL_SHAPED_PACK = {
    "schema_version": "1.0",
    "status": "provisional",
    "legacy_paradigm": SRL_PARADIGM_ID,
    "extends": None,
    "taxonomy_id": "PROVISIONAL/single_agent_rl",
    "fingerprint": {
        "what_it_is": "Model-free single-agent RL for collision-avoidance "
        "control with a learned value network over ego observations.",
        "not_this": [],
    },
    "scaffold_hints": {
        "interface_hint": "compute_reward(state, action, seed) -> float",
    },
    "family_components": {
        "reward_function": {
            "required": True,
            "description": "Reward the policy optimizes, as method.py computes it.",
        },
    },
    "semantic_checks": [],
    "smoke_bugs": [],
}


def _install_run_pack(run_dir: Path, pack: dict) -> Path:
    install_dir = run_dir / ".pipeline" / "provisional_packs" / "20260715-srl"
    install_dir.mkdir(parents=True)
    (install_dir / "pack.yaml").write_text(
        yaml.safe_dump(pack), encoding="utf-8")
    return run_dir / ".pipeline" / "provisional_packs"


def _srl_spec() -> dict:
    return {
        "comparison": {
            "classification": {"id": SRL_PARADIGM_ID},
            "pluggable_component": {
                "name": "compute_reward",
                "signature": "compute_reward(state, action, seed=0) -> float",
            },
        },
    }


def _write_contract(run_dir: Path, contract: dict) -> None:
    (run_dir / ".pipeline").mkdir(parents=True, exist_ok=True)
    (run_dir / ".pipeline" / "arch_contract.json").write_text(
        json.dumps(contract), encoding="utf-8")


def test_pack_declaration_flows_into_build_plan(tmp_path):
    from scripts.build_plan import load_build_plan

    run_dir = tmp_path / "run"
    packs_dir = _install_run_pack(run_dir, SRL_SHAPED_PACK)
    plan = load_build_plan(_srl_spec(), ROOT, provisional_packs_dir=packs_dir)
    assert plan is not None
    declared = plan["arch_contract_requirements"]["family_components"]
    assert set(declared) == {"reward_function"}
    assert declared["reward_function"]["required"] is True

    # Leak guard: without the overlay the id is unserved, exactly as before.
    assert load_build_plan(_srl_spec(), ROOT) is None


def test_committed_plans_carry_no_family_components(monkeypatch):
    """Adjacent-good: NON-DECLARING committed families stay untouched. Since
    R2C-029, motion_planning/rl_collision_avoidance is the first committed
    declarer (variant-level, so the motion_planning family plan swept here
    stays clean); every non-declaring static plan is WHOLE-PLAN identical to
    what it would be with the family-components merge disabled (not merely
    missing the key — a merge that mutated any other plan part for
    non-declaring nodes must fail here; review finding 2026-07-17)."""
    import scripts.build_plan as bp

    for pid in ("active_learning", "knowledge_distillation",
                "motion_planning", "stochastic_optimization"):
        assert pid in bp.STATIC_PLAN_BY_PARADIGM
        spec = {"comparison": {"classification": {"id": pid},
                               "pluggable_component": {"name": "x"}}}
        live = bp.load_build_plan(spec, ROOT)
        assert live is not None, f"{pid} should be taxonomy-served"
        assert "family_components" not in live["arch_contract_requirements"], pid

        with monkeypatch.context() as mp:
            mp.setattr(bp, "_merge_node_family_components",
                       lambda *a, **k: None)
            baseline = bp.load_build_plan(spec, ROOT)
        assert live == baseline, (
            f"{pid}: family-components merge changed the plan for a "
            f"non-declaring node")


def test_committed_node_declaration_surface_merges_into_plan():
    """The same declaration surface works for a committed taxonomy node (the
    promotion target of any gap pack): a node-declared `family_components`
    mapping lands in the derived plan's requirements."""
    from scripts import taxonomy as tx
    from scripts.build_plan import derive_static_build_plan

    raw = {
        "schema_version": "2.0",
        "method_roots": {
            "CLC": {
                "name": "Closed-Loop Control",
                "families": {
                    "CLC-RL": {
                        "name": "Reinforcement Learning",
                        "variants": {
                            "demo_rl": {
                                "name": "Demo RL",
                                "status": "populated",
                                "legacy_paradigm": "demo_rl",
                                "taxonomy_id": "CLC-RL/demo_rl",
                                "family_components": {
                                    "reward_function": {"required": True},
                                },
                            }
                        },
                    }
                },
            }
        },
    }
    tax = tx.load_taxonomy_uncached(ROOT, raw)
    node = tax.variant("demo_rl")
    spec = {"comparison": {"classification": {"id": "demo_rl"},
                           "pluggable_component": {"name": "compute_reward"}}}
    plan = derive_static_build_plan(
        spec, "demo_rl", node, plan_key="_provisional", tax=tax)
    assert set(plan["arch_contract_requirements"]["family_components"]) == {
        "reward_function"}


def test_promoted_rl_ca_is_the_first_live_committed_declarer():
    """R2C-029: the promoted motion_planning/rl_collision_avoidance node
    declares reward_function on the real SSOT, and the declaration reaches
    the derived plan with no overlay — the committed version of the exact
    declaration the SRL run's provisional pack was missing when Stage 2.b
    halted (stage_2b.halt, 2026-07-28)."""
    from scripts.build_plan import load_build_plan

    spec = {"comparison": {
        "classification": {"id": "motion_planning/rl_collision_avoidance"},
        "pluggable_component": {"name": "train_and_plan"},
    }}
    plan = load_build_plan(spec, ROOT)
    assert plan is not None
    declared = plan["arch_contract_requirements"]["family_components"]
    assert declared["reward_function"]["required"] is True
    assert declared["reward_function"]["required_entries"] == []
    # The optional alternate placements stay optional.
    assert declared["value_network"]["required"] is False
    assert declared["policy"]["required"] is False


def test_family_components_inherit_and_child_overrides():
    """The declaration is an inheritable mapping field: shallow-merged by
    component name down the chain, child wins per name."""
    from scripts import taxonomy as tx

    raw = {
        "schema_version": "2.0",
        "method_roots": {
            "CLC": {
                "name": "Closed-Loop Control",
                "families": {
                    "CLC-RL": {
                        "name": "Reinforcement Learning",
                        "family_components": {
                            "reward_function": {"required": True},
                            "replay_buffer": {"required": False},
                        },
                        "variants": {
                            "demo_rl": {
                                "name": "Demo RL",
                                "status": "populated",
                                "legacy_paradigm": "demo_rl",
                                "taxonomy_id": "CLC-RL/demo_rl",
                                "family_components": {
                                    "replay_buffer": {"required": True},
                                },
                            }
                        },
                    }
                },
            }
        },
    }
    tax = tx.load_taxonomy_uncached(ROOT, raw)
    effective = tx.effective_node(tax, "demo_rl")["family_components"]
    assert effective["reward_function"] == {"required": True}
    assert effective["replay_buffer"] == {"required": True}  # child wins
    node = tax.variant("demo_rl")
    # Surfaced through the implementation bucket (the build/codegen context),
    # resolving against the taxonomy view the node came from.
    assert "family_components" in tx.node_implementation(node, tax)


# ---------------------------------------------------------------------------
# Stage 2.d static validator: both directions, SRL-shaped.
# ---------------------------------------------------------------------------


def _validate_srl_run(tmp_path: Path, contract: dict, pack: dict = SRL_SHAPED_PACK):
    run_dir = tmp_path / "run"
    _install_run_pack(run_dir, pack)
    _write_contract(run_dir, contract)
    return validate_arch_contract(_srl_spec(), run_dir)


def test_declared_and_emitted_component_accepted_end_to_end(tmp_path):
    """Known-good replay of the SRL halt: the pack declares reward_function,
    the coder emits it under the container, and Stage 2.d passes — the exact
    emission that halted the 2026-07-15 run twice."""
    errs = _validate_srl_run(
        tmp_path, _srl_contract({"reward_function": REWARD_FUNCTION_BLOCK}))
    assert errs == [], errs


def test_undeclared_component_rejected_with_actionable_message(tmp_path):
    """Known-bad: a component the pack never declared fails naming the
    unknown key, the declared set, and both remedies (declare it in the
    pack, or remove it from the contract). The pack is named as a possible
    owner — the fedavg 2026-07-16 lesson: when the producer follows the
    pack, pack-side errors must point at the pack."""
    errs = _validate_srl_run(
        tmp_path,
        _srl_contract({
            "reward_function": REWARD_FUNCTION_BLOCK,
            "reward_shaping": {"description": "never declared"},
        }),
    )
    [err] = [e for e in errs if "reward_shaping" in e]
    assert "family_components.reward_shaping" in err
    assert "['reward_function']" in err  # the declared set
    assert "declare" in err and "remove it from the contract" in err
    assert "provisional pack" in err  # the pack named as a possible owner


def test_declared_required_component_missing_rejected(tmp_path):
    """Known-bad: omitting a declared-required component fails loudly, same
    as any required block today."""
    errs = _validate_srl_run(tmp_path, _srl_contract(None))
    [err] = [e for e in errs if "reward_function" in e]
    assert "declared required" in err and "missing" in err


def test_typo_in_component_name_fails_both_directions(tmp_path):
    """Typo safety for the container: a misspelled component name can never
    pass as an ignored extra — it fires missing-required AND undeclared."""
    errs = _validate_srl_run(
        tmp_path, _srl_contract({"reward_funtcion": REWARD_FUNCTION_BLOCK}))
    assert any("reward_function" in e and "missing" in e for e in errs), errs
    assert any("reward_funtcion" in e and "not declared" in e for e in errs), errs


def test_optional_declared_component_may_be_omitted(tmp_path):
    pack = dict(SRL_SHAPED_PACK)
    pack["family_components"] = {
        "reward_function": {"required": False},
    }
    errs = _validate_srl_run(tmp_path, _srl_contract(None), pack=pack)
    assert errs == [], errs
    # ...but when emitted, it is still a legal (declared) name.
    errs = _validate_srl_run(
        tmp_path / "emitted",
        _srl_contract({"reward_function": REWARD_FUNCTION_BLOCK}), pack=pack)
    assert errs == [], errs


def test_declared_required_entries_enforced(tmp_path):
    """A declaration may pin entry names for multi-slot components (the
    optimizer_state shape); a missing entry is named exactly."""
    pack = dict(SRL_SHAPED_PACK)
    pack["family_components"] = {
        "reward_function": {
            "required": True,
            "required_entries": ["state", "action"],
        },
    }
    errs = _validate_srl_run(
        tmp_path,
        _srl_contract({
            "reward_function": {"entries": {"state": {"shape": "(obs_dim,)"}}},
        }),
        pack=pack,
    )
    [err] = errs
    assert "family_components.reward_function.entries.action" in err
    contract = _srl_contract({
        "reward_function": {"entries": {
            "state": {"shape": "(obs_dim,)"},
            "action": {"shape": "(action_dim,)"},
        }},
    })
    assert _validate_srl_run(tmp_path / "good", contract, pack=pack) == []


# ---------------------------------------------------------------------------
# Adjacent-goods: committed families are untouched by the container.
# ---------------------------------------------------------------------------


def _validate_committed(tmp_path: Path, contract: dict, spec: dict):
    run_dir = tmp_path / "run"
    _write_contract(run_dir, contract)
    return validate_arch_contract(spec, run_dir)


def test_stochastic_optimization_optimizer_state_untouched(tmp_path):
    """optimizer_state stays a universal optional field (deliberately NOT
    migrated onto the container): the ADAM-shaped contract validates exactly
    as before, and its required_subblocks enforcement still fires."""
    contract = {
        "schema_version": "1.0.0",
        "paradigm_id": "stochastic_optimization",
        "data_loader": {"load_data_returns": {"x": "(B, n_features)", "y": "(B,)"}},
        "architecture": {
            "objective_model": {
                "class_name": "ObjectiveModel",
                "forward": {"input": {"x": "(B, n_features)"},
                            "output_type": "tensor", "output_shape": "(B, 1)"},
            },
        },
        "pluggable_component": {
            "name": "optimize",
            "input_shapes": {"initial_params": "(P,)"},
            "output_shape": "dict of traces",
        },
        "optimizer_state": {
            "step": {"type": "int"},
            "params": {"shape": "(P,)"},
            "first_moment": {"shape": "(P,)"},
            "second_moment": {"shape": "(P,)"},
        },
    }
    spec = {"comparison": {"classification": {"id": "stochastic_optimization"},
                           "pluggable_component": {"name": "optimize"}}}
    assert _validate_committed(tmp_path, contract, spec) == []

    # The ADAM 2026-07-08 known-bad is still caught through required_blocks.
    broken = dict(contract)
    broken.pop("optimizer_state")
    errs = _validate_committed(tmp_path / "broken", broken, spec)
    assert any("optimizer_state" in e for e in errs), errs


def test_motion_planning_contract_validates_byte_identically(tmp_path):
    """Adjacent-good: a committed motion-planning contract (whose extra parts
    live INSIDE the open architecture dict) validates cleanly, and the parsed
    model never grows a family_components key."""
    contract = {
        "schema_version": "1.0.0",
        "paradigm_id": "motion_planning",
        "data_loader": {"load_data_returns": {"start": "(state_dim,)",
                                              "environment": "(opaque)"}},
        "architecture": {
            "dynamics": {
                "class_name": "UnicycleDynamics",
                "forward": {"input": {"state": "(state_dim,)",
                                      "control": "(control_dim,)", "dt": "float"},
                            "output_type": "tensor", "output_shape": "(state_dim,)"},
            },
            "collision_model": {
                "class_name": "CircularCollisionModel",
                "forward": {"input": {"state": "(state_dim,)"}, "output_type": "bool"},
            },
        },
        "pluggable_component": {
            "name": "plan",
            "input_shapes": {"start": "(state_dim,)", "seed": "int"},
            "output_shape": "PlanResult",
        },
    }
    spec = {"comparison": {"classification": {"id": "motion_planning"},
                           "pluggable_component": {"name": "plan"}}}
    assert _validate_committed(tmp_path, contract, spec) == []
    parsed = ArchContract.model_validate(contract)
    assert "family_components" not in parsed.model_dump(exclude_none=True)


def test_active_learning_contract_validates_byte_identically(tmp_path):
    """Adjacent-good: the supervised-ML/AL shape is untouched."""
    contract = {
        "schema_version": "1.0.0",
        "paradigm_id": "active_learning",
        "data_loader": {"load_data_returns": {
            "x_pool": "(N_pool, n_features)", "y_pool": "(N_pool,)",
            "x_test": "(N_test, n_features)", "y_test": "(N_test,)"}},
        "architecture": {
            "model": {
                "class_name": "Net",
                "forward": {"input": {"x": "(B, n_features)"},
                            "output_type": "tensor",
                            "output_shape": "(B, n_classes)"},
            },
        },
        "pluggable_component": {
            "name": "select_batch",
            "input_shapes": {"x_unlabeled": "(N_pool, n_features)"},
            "output_shape": "list[int] of length batch_size",
        },
        "training_loop": {
            "function_name": "train_from_scratch",
            "input_shapes": {"x_train": "(N, n_features)", "y_train": "(N,)"},
        },
    }
    spec = {"comparison": {"classification": {"id": "active_learning"},
                           "pluggable_component": {"name": "select_batch"}}}
    assert _validate_committed(tmp_path, contract, spec) == []
    parsed = ArchContract.model_validate(contract)
    assert "family_components" not in parsed.model_dump(exclude_none=True)


# ---------------------------------------------------------------------------
# Proposal authoring: the gap path gets the same declaration surface.
# ---------------------------------------------------------------------------


def test_proposal_pack_shape_accepts_family_components():
    from scripts.validate_paradigm_proposal import _check_pack_shape

    errors: list[str] = []
    warnings: list[str] = []
    pack = dict(SRL_SHAPED_PACK)
    _check_pack_shape(pack, errors, warnings)
    assert not [e for e in errors if "family_components" in e], errors


def test_proposal_pack_shape_rejects_malformed_declarations():
    """Authoring-time typo safety: a malformed declaration fails inside the
    proposal retry loop, not three stages later in a live run."""
    from scripts.validate_paradigm_proposal import _check_pack_shape

    def _errors_for(family_components) -> list[str]:
        errors: list[str] = []
        pack = dict(SRL_SHAPED_PACK)
        pack["family_components"] = family_components
        _check_pack_shape(pack, errors, [])
        return [e for e in errors if "family_components" in e]

    # Not a mapping.
    assert any("must be a mapping" in e for e in _errors_for(["reward_function"]))
    # Non-identifier component name.
    assert any("python-identifier" in e
               for e in _errors_for({"reward function": {}}))
    # Typo'd declaration subkey (extra='forbid' on the declaration model).
    assert any("family_components.reward_function invalid" in e
               for e in _errors_for({"reward_function": {"requird": True}}))
    # required_entries must be a list of names.
    assert any("family_components.reward_function invalid" in e
               for e in _errors_for({"reward_function": {"required_entries": "state"}}))


def test_conformance_walker_can_resolve_family_component_paths():
    """A future pack may require dotted paths into the container (e.g. via
    required_blocks); the item-21 conformance walker must be able to resolve
    them so such a requirement never becomes an inexpressible trap."""
    from tests.test_arch_contract_conformance import _terminal_types

    assert _terminal_types(
        ArchContract, "family_components.reward_function".split("."))
    assert _terminal_types(
        ArchContract,
        "family_components.reward_function.entries.state.shape".split("."))


# ---------------------------------------------------------------------------
# Review-round additions (2026-07-17)
# ---------------------------------------------------------------------------


def test_undeclared_component_fails_at_the_2b_gate(tmp_path):
    """The typo class fails at the producer's own Stage 2.b gate, not two
    stages later at 2.d. Before the container existed, a stray top-level
    key died at 2.b via `extra='forbid'`; the container must not move that
    failure downstream (review finding, 2026-07-17)."""
    from scripts.validate_architecture_coder_output import validate
    from tests.test_validator_manifest_driven import (
        _make_spec,
        _write_arch_contract,
        _write_minimal_paper_map,
    )

    run_dir = tmp_path / "run"
    method_dir = run_dir / "method"
    pipeline_dir = run_dir / ".pipeline"
    method_dir.mkdir(parents=True)

    (method_dir / "model.py").write_text("""
import torch.nn as nn

class MLPClassifier(nn.Module):
    def __init__(self, input_dim, n_classes):
        super().__init__()
        self.fc = nn.Linear(input_dim, n_classes)
    def forward(self, x):
        return self.fc(x)
""")
    (method_dir / "training.py").write_text("""
import torch
from .model import MLPClassifier

def build_model(input_dim, n_classes):
    return MLPClassifier(input_dim, n_classes)

def train_from_scratch(x_train, y_train, *, learning_rate=1e-3, num_epochs=10, batch_size=32, seed=0):
    torch.manual_seed(seed)
    model = MLPClassifier(x_train.shape[1], int(y_train.max()) + 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    return model
""")
    _write_arch_contract(pipeline_dir, paradigm_id="active_learning")
    _write_minimal_paper_map(pipeline_dir)

    # Inject an undeclared component (active_learning declares none).
    contract_path = pipeline_dir / "arch_contract.json"
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    contract["family_components"] = {
        "reward_function": {
            "value": _v2_opaque(
                "callable reward function",
                "The undeclared-name gate runs independently of synthesis.",
            ),
        },
    }
    contract_path.write_text(json.dumps(contract), encoding="utf-8")

    errors = validate(_make_spec(paradigm_id="active_learning"), run_dir, ROOT)
    hits = [e for e in errors
            if "family_components.reward_function" in e
            and "not declared" in e]
    assert hits, f"expected the undeclared-name error at 2.b, got: {errors}"
    # The error keeps the message bar: both remedies named.
    assert "declare" in hits[0] and "remove it from the contract" in hits[0]


def test_new_subparadigm_pack_must_declare_family_components():
    """R2C-032 authoring floor: a new-subparadigm pack must DECLARE its
    family components — absence is an error, an explicit empty mapping
    (`family_components: {}`) is the deliberate "the universal skeleton
    covers this method" answer. The 2026-07-28 SRL run cleared
    classification on a pack with no declaration and died at stage 2b when
    the contract's reward_function was rejected as undeclared."""
    from scripts.validate_paradigm_proposal import _check_pack_shape

    def _errors_for(pack, decision):
        errors: list[str] = []
        _check_pack_shape(pack, errors, [], decision=decision)
        return [e for e in errors if "family_components" in e]

    undeclared = dict(SRL_SHAPED_PACK)
    del undeclared["family_components"]

    # Absence is an authoring error for a new sub-paradigm, and the error
    # names the explicit-empty escape hatch.
    errs = _errors_for(dict(undeclared), "new_subparadigm_needed")
    assert errs, "absence must be an error for new_subparadigm_needed"
    assert "must declare family_components" in errs[0]
    assert "family_components: {}" in errs[0]

    # The explicit empty mapping is the legitimate opt-out.
    empty = dict(undeclared)
    empty["family_components"] = {}
    assert _errors_for(empty, "new_subparadigm_needed") == []

    # Top-level packs and decision-less callers keep today's contract.
    assert _errors_for(dict(undeclared), "new_top_level_needed") == []
    assert _errors_for(dict(undeclared), None) == []


# The SRL 2026-07-28 stage-2b halt shape, made executable (R2C-032): the
# run's pack extended the committed motion_planning family, the implicit
# ancestor walk handed the RL package the motion-planning manifest, and all
# four validation failures traced to that inherited build context. The
# neutral-plan declaration is the fix; the undeclared pack must keep failing
# exactly as today (honest, deterministic — the acceptance criterion).
# Fixture id re-pointed at promotion (R2C-029, 2026-07-28): the halt's
# original id is now a committed node, and a provisional pack for a
# committed id is shadow-skipped, so the gap shape under test needs an id
# the SSOT does not serve.

MP_GAP_PARADIGM_ID = "motion_planning/synthetic_gap_variant"

MP_GAP_PACK_UNDECLARED = {
    "schema_version": "1.0",
    "status": "provisional",
    "legacy_paradigm": MP_GAP_PARADIGM_ID,
    "extends": "motion_planning",
    "taxonomy_id": "CLC-MP/synthetic_gap_variant",
    "fingerprint": {
        "what_it_is": "Multiagent collision avoidance via deep RL with a "
        "norm-inducing reward and weight-shared value network.",
        "not_this": [],
    },
    "scaffold_hints": {
        "interface_hint": "plan(start, goal, environment, dynamics, seed) "
                          "-> PlanResult",
    },
    "semantic_checks": [],
    "smoke_bugs": [],
}

MP_GAP_PACK_NEUTRAL = {
    **MP_GAP_PACK_UNDECLARED,
    "build_plan": {
        "source": "neutral",
        "reasoning": "SA-CADRL is an RL method; the motion-planning "
                     "manifest does not describe its build shape.",
    },
    "family_components": {
        "reward_function": {
            "required": False,
            "description": "Norm-inducing reward the policy optimizes.",
        },
    },
}


def test_neutral_plan_2b_gate_accepts_paper_shaped_classes(tmp_path):
    from scripts.validate_architecture_coder_output import validate
    from tests.test_validator_manifest_driven import (
        _make_spec,
        _write_minimal_paper_map,
    )

    def _run_2b(run_dir, pack):
        method_dir = run_dir / "method"
        pipeline_dir = run_dir / ".pipeline"
        method_dir.mkdir(parents=True)
        install_dir = pipeline_dir / "provisional_packs" / "20260728-srl"
        install_dir.mkdir(parents=True)
        (install_dir / "pack.yaml").write_text(
            yaml.safe_dump(pack), encoding="utf-8")

        # The architecture-coder output the halt shows was CORRECT for the
        # paper: RL-shaped classes, no classical planning class structure.
        (method_dir / "model.py").write_text("""
import torch.nn as nn


class MultiAgentValueNet(nn.Module):
    def __init__(self, obs_dim=14, hidden_dim=64):
        super().__init__()
        self.fc = nn.Linear(obs_dim, 1)

    def forward(self, x):
        return self.fc(x)


class SA_CADRLPolicy(nn.Module):
    def __init__(self, obs_dim=14, action_dim=11):
        super().__init__()
        self.fc = nn.Linear(obs_dim, action_dim)

    def forward(self, x):
        return self.fc(x)
""")
        (method_dir / "training.py").write_text("""
import torch

from .model import MultiAgentValueNet


def build_value_net(obs_dim=14, **arch_kwargs):
    return MultiAgentValueNet(obs_dim=obs_dim, **arch_kwargs)


def train_and_plan(value_net, episodes, *, learning_rate=1e-3, seed=0):
    torch.manual_seed(seed)
    return value_net
""")
        contract = {
            "schema_version": "2.0.0",
            "paradigm_id": MP_GAP_PARADIGM_ID,
            "dimensions": {
                "fixture_batch_size": _v2_literal_dimension(2),
                "observation_feature_count": _v2_literal_dimension(14),
                "action_count": _v2_literal_dimension(11),
                "state_coordinate_count": _v2_literal_dimension(2),
                "scalar_output_width": _v2_literal_dimension(1),
            },
            "data_loader": {"load_data_returns": {
                "observations": _v2_tensor(
                    "float32",
                    ("fixture_batch_size", "N"),
                    ("observation_feature_count", "obs_dim"),
                ),
                "env": _v2_opaque(
                    "collision-avoidance environment",
                    "Environment objects are structured runtime values.",
                ),
            }},
            "architecture": {
                "value_net": {
                    "class_name": "MultiAgentValueNet",
                    "constructor_args": {
                        "obs_dim": {
                            "dimension": _v2_dimension(
                                "observation_feature_count", "obs_dim"
                            ),
                        },
                    },
                    "forward": {
                        "input": {
                            "x": _v2_tensor(
                                "float32",
                                ("fixture_batch_size", "B"),
                                ("observation_feature_count", "obs_dim"),
                            ),
                        },
                        "output": _v2_tensor(
                            "float32",
                            ("fixture_batch_size", "B"),
                            ("scalar_output_width", "1"),
                        ),
                    },
                },
                "policy": {
                    "class_name": "SA_CADRLPolicy",
                    "constructor_args": {
                        "obs_dim": {
                            "dimension": _v2_dimension(
                                "observation_feature_count", "obs_dim"
                            ),
                        },
                        "action_dim": {
                            "dimension": _v2_dimension(
                                "action_count", "action_dim"
                            ),
                        },
                    },
                    "forward": {
                        "input": {
                            "x": _v2_tensor(
                                "float32",
                                ("fixture_batch_size", "B"),
                                ("observation_feature_count", "obs_dim"),
                            ),
                        },
                        "output": _v2_tensor(
                            "float32",
                            ("fixture_batch_size", "B"),
                            ("action_count", "action_dim"),
                        ),
                    },
                },
            },
            "pluggable_component": {
                "name": "plan",
                "input": {
                    "start": _v2_tensor(
                        "float32", ("state_coordinate_count", "state_dim")
                    ),
                },
                "output": _v2_opaque(
                    "PlanResult",
                    "PlanResult is a structured planner result.",
                ),
            },
            "family_components": {
                "reward_function": TYPED_REWARD_FUNCTION_COMPONENT,
            },
        }
        (pipeline_dir / "arch_contract.json").write_text(
            json.dumps(contract), encoding="utf-8")
        _write_minimal_paper_map(pipeline_dir)
        spec = _make_spec(paradigm_id=MP_GAP_PARADIGM_ID)
        spec["comparison"]["pluggable_component"]["name"] = "plan"
        return validate(spec, run_dir, ROOT)

    # Known-good: under the declared-neutral pack the flexible provisional
    # manifest accepts the paper-shaped classes and the pack-declared
    # reward_function.
    errors = _run_2b(tmp_path / "neutral", MP_GAP_PACK_NEUTRAL)
    assert not any("<SystemDynamics>" in e for e in errors), errors
    assert not any("<CollisionModel>" in e for e in errors), errors
    assert not any("precompute_motion_primitives" in e for e in errors), errors
    assert not any("family_components.reward_function" in e for e in errors), errors

    # Known-bad, unchanged: the undeclared pack reproduces the SRL halt's
    # four failure shapes exactly as today.
    errors = _run_2b(tmp_path / "undeclared", MP_GAP_PACK_UNDECLARED)
    assert any("family_components.reward_function" in e
               and "not declared" in e for e in errors), errors
    assert any("<SystemDynamics>" in e for e in errors), errors
    assert any("<CollisionModel>" in e for e in errors), errors
    assert any("precompute_motion_primitives" in e for e in errors), errors


def test_committed_node_declarations_are_shape_linted():
    """Committed-side declaration lint (review finding, 2026-07-17): any
    `family_components` block on a committed taxonomy node must validate
    against FamilyComponentDeclaration, so the first SRL-style promotion
    cannot land a malformed shape that the build-plan merge would silently
    drop or under-enforce."""
    from pydantic import ValidationError

    from schemas.paradigm_proposal import FamilyComponentDeclaration

    ssot = yaml.safe_load(
        (ROOT / "docs/ssot/taxonomies.yaml").read_text(encoding="utf-8"))

    def _walk(node, path):
        if not isinstance(node, dict):
            return
        declared = node.get("family_components")
        if declared is not None:
            assert isinstance(declared, dict), (
                f"{path}: family_components must be a mapping, "
                f"got {type(declared).__name__}")
            for name, body in declared.items():
                assert str(name).isidentifier(), f"{path}: bad name {name!r}"
                FamilyComponentDeclaration.model_validate(body or {})
        for key, child in node.items():
            if isinstance(child, dict):
                _walk(child, f"{path}.{key}")

    _walk(ssot, "taxonomies")

    # The lint has teeth: the malformed shapes the review named are
    # rejected by the declaration model itself.
    with pytest.raises(ValidationError):
        FamilyComponentDeclaration.model_validate(
            {"required_entires": ["state"]})  # typo'd subkey
    with pytest.raises(ValidationError):
        FamilyComponentDeclaration.model_validate(["not", "a", "mapping"])
