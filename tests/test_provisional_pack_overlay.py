"""Gap-path serves() overlay tests (gap-path-serves-overlay-design.md).

The SRL run (rl_collision_avoidance, a genuine taxonomy gap) authored and
installed a provisional pack exactly as designed, then halted because the
strict spec validator's cross-check resolves `serves()` against the
committed SSOT only. These tests pin the fix: `load_taxonomy` accepts a
run-local `provisional_packs_dir` overlay, grafted nodes serve their
provisional classification, and — the leak guard, asserted both ways —
nothing provisional is ever reachable without the overlay argument.

The original fixture id (motion_planning/rl_collision_avoidance) was
promoted to a committed node (R2C-029, 2026-07-28); the never-committed
fixtures now use the synthetic GAP_PARADIGM_ID so the leak guards keep
testing an id the SSOT does not serve, and the promoted id gets its own
positive committed-serve tests below.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from schemas.method_spec import MethodSpec
from scripts import run_layout, taxonomy
from scripts.taxonomy import VariantNode, effective_node, load_taxonomy, serves
from scripts.validate_method_spec import cross_check_field_guide
from tests.test_method_spec_schema import _minimal_valid_spec

ROOT = Path(__file__).resolve().parents[1]

# Synthetic never-committed gap id. The original fixture id
# (motion_planning/rl_collision_avoidance) was promoted to a committed node
# (R2C-029, 2026-07-28); this id preserves the exact overlay shape the SRL
# run exercised — a sub-variant pack extending the committed motion_planning
# family — without colliding with the SSOT.
GAP_PARADIGM_ID = "motion_planning/synthetic_gap_variant"

GAP_SHAPED_PACK = {
    "schema_version": "1.0",
    "status": "provisional",
    "legacy_paradigm": GAP_PARADIGM_ID,
    "extends": "motion_planning",
    "taxonomy_id": "CLC-MP/synthetic_gap_variant",
    "fingerprint": {
        "what_it_is": "Multiagent collision avoidance via deep RL with a "
        "norm-inducing reward and weight-shared value network.",
        "not_this": [],
    },
    "scaffold_hints": {
        "interface_hint": "plan(start, goal, environment, dynamics, seed) -> PlanResult",
    },
    "semantic_checks": [],
    "smoke_bugs": [],
}


def _install_pack(tmp_path: Path, pack: dict, proposal_id: str = "20260703-srl") -> Path:
    packs_dir = tmp_path / "provisional_packs"
    install_dir = packs_dir / proposal_id
    install_dir.mkdir(parents=True)
    (install_dir / "pack.yaml").write_text(yaml.safe_dump(pack), encoding="utf-8")
    return packs_dir


def test_overlay_serves_the_provisional_classification(tmp_path):
    packs_dir = _install_pack(tmp_path, GAP_SHAPED_PACK)
    tax = load_taxonomy(ROOT, provisional_packs_dir=packs_dir)

    node = serves(GAP_PARADIGM_ID, tax)
    assert isinstance(node, VariantNode)
    assert node.provisional is True
    assert node.is_populated

    # The grafted node resolves inheritance through its extends-family: the
    # effective view carries both the pack's own fields and every inheritable
    # field the committed motion_planning family declares.
    effective = effective_node(tax, node.slug)
    assert effective["fingerprint"]["what_it_is"].startswith("Multiagent")
    committed_family = load_taxonomy(ROOT).node_for_legacy("motion_planning")
    for key in committed_family.fields:
        if key in taxonomy._INHERITABLE_MAPPING_FIELDS:
            assert key in effective


def test_overlay_never_leaks_into_the_committed_view(tmp_path):
    packs_dir = _install_pack(tmp_path, GAP_SHAPED_PACK)
    overlay = load_taxonomy(ROOT, provisional_packs_dir=packs_dir)
    assert serves(GAP_PARADIGM_ID, overlay) is not None

    # The same id, resolved without the overlay argument, stays unserved —
    # on the default view, on a fresh committed load, and on the committed
    # instance the overlay was built from.
    assert serves(GAP_PARADIGM_ID) is None
    committed = load_taxonomy(ROOT)
    assert serves(GAP_PARADIGM_ID, committed) is None
    assert GAP_PARADIGM_ID not in committed._node_by_legacy

    # An empty/missing pack dir returns the committed view unchanged.
    assert load_taxonomy(ROOT, provisional_packs_dir=tmp_path / "absent") is committed


def test_stub_pack_still_reads_unserved(tmp_path):
    stub = dict(GAP_SHAPED_PACK)
    stub["scaffold_hints"] = {}  # no interface_hint: the gap path's contract floor
    packs_dir = _install_pack(tmp_path, stub)
    tax = load_taxonomy(ROOT, provisional_packs_dir=packs_dir)

    # The node grafts (visible, marked provisional) but is_populated stays
    # honest, so serves() keeps answering None and the consumer gate fires.
    node = tax.node_for_legacy(GAP_PARADIGM_ID)
    assert node is not None and node.provisional
    assert serves(GAP_PARADIGM_ID, tax) is None


def test_overlay_grafts_under_a_top_level_variant_parent(tmp_path):
    # The repo's own gap fixture shape (active_learning/geometric): `extends`
    # resolves to a committed top-level VariantNode, so the pack grafts as
    # its sub-variant and inherits through the full parent chain.
    pack = {
        "schema_version": "1.0",
        "status": "provisional",
        "legacy_paradigm": "active_learning/geometric",
        "extends": "active_learning",
        "taxonomy_id": "TE-TS/active_learning/geometric",
        "fingerprint": {"what_it_is": "Geometric active-learning selection."},
        "scaffold_hints": {
            "interface_hint": "select_batch(model, x_unlabeled, batch_size, seed)",
        },
        "semantic_checks": [],
        "smoke_bugs": [],
    }
    packs_dir = _install_pack(tmp_path, pack, proposal_id="20260704-geometric")
    tax = load_taxonomy(ROOT, provisional_packs_dir=packs_dir)

    node = serves("active_learning/geometric", tax)
    assert isinstance(node, VariantNode)
    assert node.provisional and node.parent_slug == "active_learning"

    # Inherits the committed active_learning variant's effective fields
    # (e.g. its pluggable-component contract) beneath the pack's own.
    effective = effective_node(tax, node.slug)
    parent_effective = effective_node(load_taxonomy(ROOT), "active_learning")
    for key in parent_effective:
        assert key in effective
    # The committed parent is untouched on the plain view.
    assert serves("active_learning/geometric") is None


def test_pack_shadowing_a_committed_id_is_skipped(tmp_path):
    shadow = dict(GAP_SHAPED_PACK)
    shadow["legacy_paradigm"] = "motion_planning"  # committed family id
    shadow["taxonomy_id"] = "CLC-MP/shadow"
    packs_dir = _install_pack(tmp_path, shadow)
    tax = load_taxonomy(ROOT, provisional_packs_dir=packs_dir)

    committed_node = load_taxonomy(ROOT).node_for_legacy("motion_planning")
    assert tax.node_for_legacy("motion_planning") is committed_node
    assert any(d.code == "provisional_shadows_committed" for d in tax.diagnostics)


# The 2026-07-13 SRL halt shape verbatim: the gap path authored a correct
# pack for `single_agent_rl`, but the committed SSOT reserves that name as
# an empty placeholder and the unconditional shadow skip blocked the graft,
# so serves() found only the stub and the strict cross-check halted the run.
RESERVED_SHAPED_PACK = {
    "schema_version": "1.0",
    "status": "provisional",
    "legacy_paradigm": "single_agent_rl",
    "extends": None,
    "taxonomy_id": "PROVISIONAL/single_agent_rl",
    "fingerprint": {
        "what_it_is": "Model-free single-agent RL for collision-avoidance "
        "control with a learned value network over ego observations.",
        "not_this": [],
    },
    "scaffold_hints": {
        "interface_hint": "act(observation, seed) -> Action",
    },
    "semantic_checks": [],
    "smoke_bugs": [],
}


def test_pack_populates_a_committed_reserved_placeholder(tmp_path):
    packs_dir = _install_pack(tmp_path, RESERVED_SHAPED_PACK,
                              proposal_id="20260713-single_agent_rl")
    tax = load_taxonomy(ROOT, provisional_packs_dir=packs_dir)

    node = serves("single_agent_rl", tax)
    assert isinstance(node, VariantNode)
    assert node.provisional is True
    assert node.is_populated
    # The graft adopts the COMMITTED identity, not the PROVISIONAL/* one:
    # downstream consumers see the real taxonomy id and family position.
    assert node.taxonomy_id == "CLC-RL/single_agent_rl"
    assert any(d.code == "provisional_populates_reserved"
               for d in tax.diagnostics)

    # Leak guard, both ways: the committed view keeps its reserved stub and
    # never serves the id without the overlay argument.
    committed = load_taxonomy(ROOT)
    assert committed._variants_by_slug["single_agent_rl"].status == "reserved"
    assert serves("single_agent_rl", committed) is None


def test_stub_pack_on_a_reserved_placeholder_stays_unserved(tmp_path):
    stub = dict(RESERVED_SHAPED_PACK)
    stub["scaffold_hints"] = {}  # below the gap path's contract floor
    packs_dir = _install_pack(tmp_path, stub)
    tax = load_taxonomy(ROOT, provisional_packs_dir=packs_dir)

    # The placeholder is populated-in-name only: the grafted node is
    # provisional and NOT populated, so serves() stays honest.
    node = tax._variants_by_slug["single_agent_rl"]
    assert node.provisional and node.status == "provisional"
    assert serves("single_agent_rl", tax) is None


def test_populated_committed_nodes_stay_shadow_protected(tmp_path):
    # The carve-out is for reserved placeholders ONLY: a collision with a
    # populated committed node still skips, and the skip message now names
    # the colliding node's status so the mechanism is legible.
    shadow = dict(RESERVED_SHAPED_PACK)
    shadow["legacy_paradigm"] = "motion_planning"  # committed populated id
    shadow["taxonomy_id"] = "PROVISIONAL/motion_planning"
    packs_dir = _install_pack(tmp_path, shadow)
    tax = load_taxonomy(ROOT, provisional_packs_dir=packs_dir)

    committed_node = load_taxonomy(ROOT).node_for_legacy("motion_planning")
    assert tax.node_for_legacy("motion_planning") is committed_node
    skip = next(d for d in tax.diagnostics
                if d.code == "provisional_shadows_committed")
    assert "populate a single reserved placeholder" in skip.message


def test_strict_cross_check_names_skipped_packs(tmp_path):
    # Truthfulness: when a pack was present but did not graft, "not served"
    # alone hides the mechanism from the halt stderr and the judge.
    shadow = dict(RESERVED_SHAPED_PACK)
    shadow["legacy_paradigm"] = "motion_planning"
    shadow["taxonomy_id"] = "PROVISIONAL/motion_planning"
    packs_dir = _install_pack(tmp_path, shadow)

    spec = MethodSpec.model_validate(
        _minimal_valid_spec(paradigm_id="some_unserved_paradigm"))
    errors = cross_check_field_guide(
        spec, ROOT, provisional_packs_dir=packs_dir)
    assert errors and "not served" in errors[0]
    assert "did NOT graft" in errors[0]
    assert "provisional_shadows_committed" in errors[0]

    # Without a packs dir the message stays exactly as before.
    errors_plain = cross_check_field_guide(spec, ROOT)
    assert errors_plain and "did NOT graft" not in errors_plain[0]


def test_driver_passes_pack_dir_only_when_installed(tmp_path, monkeypatch):
    # The one driver call site: _run_spec_validator forwards the run's
    # provisional_packs_dir to the strict validator exactly when the gap
    # path installed something; a non-gap run's command is unchanged.
    import run_pipeline
    from tests.helpers.state import make_state

    captured: list[list[str]] = []

    def _fake_run_script(stage_id, cmd, timeout=None):
        captured.append(list(cmd))
        from types import SimpleNamespace

        return SimpleNamespace(returncode=0, stderr="")

    monkeypatch.setattr(run_pipeline, "run_script", _fake_run_script)
    state = make_state(tmp_path / "run")

    run_pipeline._run_spec_validator(state)
    assert "--provisional-packs-dir" not in captured[-1]
    assert "--require-current-schema" not in captured[-1]
    assert "--require-methodology-contract" not in captured[-1]

    run_pipeline._run_fresh_spec_validator(state)
    assert "--require-current-schema" in captured[-1]
    assert "--require-methodology-contract" in captured[-1]

    state.paths.provisional_packs_dir.mkdir(parents=True)
    run_pipeline._run_spec_validator(state)
    cmd = captured[-1]
    # Position-independent: the command may also carry --paper-md (the
    # param-glossary meaning-quote floor, 2026-07-21).
    assert "--provisional-packs-dir" in cmd
    assert cmd[cmd.index("--provisional-packs-dir") + 1] == str(
        state.paths.provisional_packs_dir)


def test_cross_check_leak_guard_both_ways(tmp_path):
    packs_dir = _install_pack(tmp_path, GAP_SHAPED_PACK)
    spec = MethodSpec.model_validate(_minimal_valid_spec(paradigm_id=GAP_PARADIGM_ID))

    without_overlay = cross_check_field_guide(spec, ROOT)
    assert any("not served" in e for e in without_overlay)

    with_overlay = cross_check_field_guide(
        spec, ROOT, provisional_packs_dir=packs_dir
    )
    assert with_overlay == []


# ---------------------------------------------------------------------------
# Stage-2a + build-plan consumer threading (queue item 3, 2026-07-04
# overnight): the overlay reaches the scaffolder, the build-plan deriver,
# and the deterministic consumers, keyed on the run's own pack dir. SRL's
# 17:10 terminal named the exact wall these pin: scaffold_package.py called
# the taxonomy WITHOUT the overlay and exited "no taxonomy node serves".
# ---------------------------------------------------------------------------


def _install_run_pack(run_dir: Path, pack: dict | None = None) -> Path:
    """Install a pack at the RUN-LOCAL location `run_overlay_dir` keys on."""
    install_dir = run_dir / ".pipeline" / "provisional_packs" / "20260703-srl"
    install_dir.mkdir(parents=True)
    (install_dir / "pack.yaml").write_text(
        yaml.safe_dump(pack or GAP_SHAPED_PACK), encoding="utf-8")
    return run_dir / ".pipeline" / "provisional_packs"


def _gap_spec_dict() -> dict:
    return {
        "paper": {"title": "SRL", "authors": "Test et al."},
        "core_method": {"name": "SRL", "summary": "Deep RL collision avoidance."},
        "comparison": {
            "classification": {"id": GAP_PARADIGM_ID},
            "pluggable_component": {
                "name": "plan",
                "signature": "plan(start, goal, environment, dynamics, seed=0)",
            },
        },
        "critical_requirements": {"scale_dependent_hyperparameters": []},
    }


def test_run_overlay_dir_keys_on_installed_pack(tmp_path):
    run_dir = tmp_path / "run"
    assert taxonomy.run_overlay_dir(None) is None
    assert taxonomy.run_overlay_dir(run_dir) is None  # no .pipeline at all
    (run_dir / ".pipeline").mkdir(parents=True)
    assert taxonomy.run_overlay_dir(run_dir) is None  # no pack installed
    packs_dir = _install_run_pack(run_dir)
    assert taxonomy.run_overlay_dir(run_dir) == packs_dir


def test_scaffolder_serves_gap_paper_through_overlay(tmp_path):
    import json

    from scripts.scaffold_package import scaffold

    run_dir = tmp_path / "run"
    _install_run_pack(run_dir)
    spec_path = run_dir / ".pipeline" / "method_spec.json"
    spec_path.write_text(json.dumps(_gap_spec_dict()), encoding="utf-8")

    # With the run-local pack: the provisional node serves, templates_dir
    # inherits from the committed motion_planning family, files render.
    rc = scaffold(spec_path, run_dir, ROOT)
    assert rc == 0
    assert (run_dir / run_layout.PACKAGE_README).is_file()

    # Leak guard: the same spec in a run WITHOUT an installed pack still
    # halts at the serves() gate (deterministic, exit 1).
    bare_run = tmp_path / "bare"
    (bare_run / ".pipeline").mkdir(parents=True)
    bare_spec = bare_run / ".pipeline" / "method_spec.json"
    bare_spec.write_text(json.dumps(_gap_spec_dict()), encoding="utf-8")
    assert scaffold(bare_spec, bare_run, ROOT) == 1


def test_build_plan_resolves_provisional_node_via_ancestor_plan(tmp_path):
    from scripts.build_plan import load_build_plan

    run_dir = tmp_path / "run"
    packs_dir = _install_run_pack(run_dir)
    spec = _gap_spec_dict()

    plan = load_build_plan(spec, ROOT, provisional_packs_dir=packs_dir)
    assert plan is not None
    # The plan binds the RUN's provisional identity, not the ancestor's...
    assert plan["paradigm_id"] == GAP_PARADIGM_ID
    assert plan["taxonomy_id"] == "CLC-MP/synthetic_gap_variant"
    # ...while the structural base comes from the committed motion_planning
    # ancestor, and the interface hint comes from the pack itself.
    assert plan["pluggable_component"]["name"] == "plan"
    assert plan["interface_hint"].startswith("plan(")

    # Leak guard: no overlay, no plan.
    assert load_build_plan(spec, ROOT) is None


def test_static_plan_key_pins_committed_exact_match():
    from scripts.build_plan import STATIC_PLAN_BY_PARADIGM, _static_plan_key

    # Provisional nodes fall back to the nearest committed ancestor.
    assert _static_plan_key(GAP_PARADIGM_ID, provisional=True) == "motion_planning"
    # Committed nodes keep exact-match-only resolution — their no-plan
    # behavior is pinned (a served-but-unplanned committed id still fails
    # deterministically at the consumer gate, exactly as before).
    assert _static_plan_key(GAP_PARADIGM_ID, provisional=False) is None
    for committed_key in STATIC_PLAN_BY_PARADIGM:
        assert _static_plan_key(committed_key, provisional=False) == committed_key


# ---------------------------------------------------------------------------
# R2C-029: the SRL run's provisional identity is now a COMMITTED node. These
# are the positive halves of the leak guards above — the promoted id serves
# with no overlay, resolves its own static plan (no ancestor fallback), and
# carries the family_components declaration the stage_2b halt asked for.
# ---------------------------------------------------------------------------

PROMOTED_RL_CA_ID = "motion_planning/rl_collision_avoidance"


def test_promoted_rl_collision_avoidance_serves_committed():
    tax = load_taxonomy(ROOT)
    node = serves(PROMOTED_RL_CA_ID, tax)
    assert isinstance(node, VariantNode)
    assert not getattr(node, "provisional", False)
    assert node.taxonomy_id == "CLC-MP/rl_collision_avoidance"
    # Exact-match static plan, no provisional ancestor walk.
    from scripts.build_plan import _static_plan_key
    assert _static_plan_key(PROMOTED_RL_CA_ID, provisional=False) == PROMOTED_RL_CA_ID


def test_promoted_rl_collision_avoidance_build_plan_is_rl_shaped():
    """The stage_2b halt's three manifest failures, inverted: the committed
    plan declares the value-network/policy classes and a real training row
    instead of the planner classes and precompute_motion_primitives."""
    from scripts.build_plan import load_build_plan

    spec = {
        "paper": {"title": "SRL", "authors": "Test et al."},
        "core_method": {"name": "SA-CADRL", "summary": "Deep RL collision avoidance."},
        "comparison": {
            "classification": {"id": PROMOTED_RL_CA_ID},
            "pluggable_component": {
                "name": "train_and_plan",
                "signature": "train_and_plan(n_agents: int = 2, n_episodes: int = 700, seed: int = 42) -> PlanResult",
            },
        },
        "critical_requirements": {"scale_dependent_hyperparameters": []},
    }
    plan = load_build_plan(spec, ROOT)  # NO overlay argument
    assert plan is not None
    assert plan["plan_key"] == PROMOTED_RL_CA_ID
    files = {f["path"]: f for f in plan["package_manifest"]["files"]}
    model_symbols = {s["name"] for s in files["method/model.py"]["public_symbols"]}
    assert model_symbols == {"<ValueNetwork>", "<Policy>"}
    training_symbols = {s["name"] for s in files["method/training.py"]["public_symbols"]}
    assert "train_policy" in training_symbols
    assert "precompute_motion_primitives" not in training_symbols
    # The halt's fourth failure: reward_function is now a declared legal name.
    declared = plan["arch_contract_requirements"]["family_components"]
    assert declared["reward_function"]["required"] is True


def test_stale_srl_pack_now_shadows_the_committed_node(tmp_path):
    """R2C-031 interplay: a resumed/old run that still carries the SRL
    provisional pack for the now-committed id hits the shadow guard — the
    committed node wins and the skip is diagnosable."""
    stale = dict(GAP_SHAPED_PACK)
    stale["legacy_paradigm"] = PROMOTED_RL_CA_ID
    stale["taxonomy_id"] = "CLC-MP/rl_collision_avoidance"
    packs_dir = _install_pack(tmp_path, stale, proposal_id="20260728-stale-srl")
    tax = load_taxonomy(ROOT, provisional_packs_dir=packs_dir)

    committed_node = load_taxonomy(ROOT).node_for_legacy(PROMOTED_RL_CA_ID)
    assert tax.node_for_legacy(PROMOTED_RL_CA_ID) is committed_node
    assert any(d.code == "provisional_shadows_committed" for d in tax.diagnostics)


def test_bucket_loaders_resolve_overlay_view(tmp_path):
    packs_dir = _install_pack(tmp_path, GAP_SHAPED_PACK)
    tax = load_taxonomy(ROOT, provisional_packs_dir=packs_dir)

    # templates_dir + notebook_layout inherit through the pack's extends
    # chain when the overlay taxonomy is passed...
    assert taxonomy.resolve_templates_dir(
        GAP_PARADIGM_ID, ROOT, taxonomy=tax) is not None
    layout = taxonomy.load_notebook_layout(GAP_PARADIGM_ID, taxonomy=tax)
    assert layout.get("sections")
    # ...and stay sealed on the committed view (leak guard).
    assert taxonomy.resolve_templates_dir(GAP_PARADIGM_ID, ROOT) is None
    assert taxonomy.load_notebook_layout(GAP_PARADIGM_ID) == {}


def test_derive_params_gap_run_derives_via_overlay(tmp_path):
    from scripts.derive_params import derive

    run_dir = tmp_path / "run"
    _install_run_pack(run_dir)

    out = derive(_gap_spec_dict(), ROOT, run_dir=run_dir)
    assert out["schema_version"]
    assert isinstance(out["params"], dict)


def test_scaffolder_validator_sees_overlay_build_plan(tmp_path):
    import json as _json

    from scripts.scaffold_package import scaffold
    from scripts.validate_scaffolder_output import validate as validate_scaffold

    run_dir = tmp_path / "run"
    _install_run_pack(run_dir)
    spec = _gap_spec_dict()
    spec_path = run_dir / ".pipeline" / "method_spec.json"
    spec_path.write_text(_json.dumps(spec), encoding="utf-8")
    assert scaffold(spec_path, run_dir, ROOT) == 0

    # With the pack installed the validator resolves a build plan (whatever
    # else it finds, the halt is no longer "no taxonomy build plan").
    errors = validate_scaffold(spec, run_dir, ROOT)
    assert not any("no taxonomy build plan" in e for e in errors)

    # Leak guard: a bare run with the same spec fails at the plan gate.
    bare_run = tmp_path / "bare"
    (bare_run / ".pipeline").mkdir(parents=True)
    bare_errors = validate_scaffold(spec, bare_run, ROOT)
    assert any("no taxonomy build plan" in e for e in bare_errors)


def test_gap_family_training_params_derive_from_spec_evidence(tmp_path):
    """SRL 2026-07-05 (matrix row 5): the from-scratch gap run halted at the
    2.x review because params.json was EMPTY — the motion_planning branch
    assumes planners have no training knobs, and this family trains a
    network. For provisional (gap-pack) nodes the deriver now tops up from
    the spec's own structured training evidence, values only, no
    conventions. Committed families are byte-identical (the gate is
    node.provisional)."""
    from scripts.derive_params import derive

    run_dir = tmp_path / "run"
    _install_run_pack(run_dir)
    spec = _gap_spec_dict()
    spec["critical_requirements"]["training"] = {
        "optimizer": "RMSprop",
        "learning_rate": "0.001",
        "num_epochs": 3000,
        "mc_samples": None,
        "paper_section": "Section III.B, Algorithm 1",
    }

    out = derive(spec, ROOT, run_dir=run_dir)
    params = out["params"]
    # Numeric evidence lands with paper provenance and the analyzer's section.
    assert params["max_epochs"]["value"] == 3000
    assert params["max_epochs"]["source"] == "paper"
    assert params["max_epochs"]["paper_section"] == "Section III.B, Algorithm 1"
    assert params["learning_rate"]["value"] == 0.001
    # Nulls are skipped, never fabricated.
    assert "mc_samples" not in params

    # Prose in a numeric field is skipped (the 05:27 SRL draw's shape).
    spec_prose = _gap_spec_dict()
    spec_prose["critical_requirements"]["training"] = {
        "learning_rate": "Paper does not explicitly state the RMSprop rate.",
        "num_epochs": None,
        "paper_section": "Section III.B",
    }
    out2 = derive(spec_prose, ROOT, run_dir=run_dir)
    assert "max_epochs" not in out2["params"]
    assert "learning_rate" not in out2["params"]

    # Committed-family pin: the SAME spec without the pack (a committed
    # motion_planning id) derives exactly as before — no training params.
    committed_spec = _gap_spec_dict()
    committed_spec["comparison"]["classification"]["id"] = "motion_planning"
    committed_spec["critical_requirements"]["training"] = {
        "learning_rate": "0.001", "num_epochs": 3000,
        "paper_section": "Section III.B",
    }
    bare_run = tmp_path / "bare"
    (bare_run / ".pipeline").mkdir(parents=True)
    out3 = derive(committed_spec, ROOT, run_dir=bare_run)
    assert "max_epochs" not in out3["params"]
    assert "learning_rate" not in out3["params"]


def test_promoted_rl_ca_params_derivation_replaces_provisional_topup(tmp_path):
    """R2C-029 regression guard: the provisional-only spec-evidence top-up is
    gated on node.provisional and turns OFF at promotion. The committed node's
    params_derivation block must keep the training knobs flowing (via
    taxonomy.load_params_derivation, merged last in derive()).

    Provenance note (decision-log open judgment 5): params_derivation emits
    source: system_inferred with the formula documented (schema section 3.17),
    NOT source: paper as the provisional top-up did; widening the deriver's
    top-up gate instead is the flagged alternative. The formula inputs are
    numeric-only (`_resolve_derivation_input`), so this spec carries a float
    learning rate where the gap-path top-up test above uses the string shape.
    """
    from scripts.derive_params import derive

    spec = _gap_spec_dict()
    spec["comparison"]["classification"]["id"] = PROMOTED_RL_CA_ID
    spec["comparison"]["pluggable_component"] = {
        "name": "train_and_plan",
        "signature": "train_and_plan(n_agents=2, n_episodes=700, seed=42)",
    }
    spec["critical_requirements"]["training"] = {
        "optimizer": "RMSprop",
        "learning_rate": 0.001,
        "num_epochs": 3000,
        "paper_section": "Section III-B, Algorithm 1",
    }
    bare_run = tmp_path / "bare"
    (bare_run / ".pipeline").mkdir(parents=True)

    params = derive(spec, ROOT, run_dir=bare_run)["params"]
    assert params["max_epochs"]["value"] == 3000
    assert params["learning_rate"]["value"] == 0.001
    assert params["max_epochs"]["source"] == "system_inferred"
    assert params["learning_rate"]["source"] == "system_inferred"

    # Honest-null half (the SRL spec's own shape): prose/null training
    # evidence emits documented nulls, never guesses.
    spec_prose = _gap_spec_dict()
    spec_prose["comparison"]["classification"]["id"] = PROMOTED_RL_CA_ID
    spec_prose["critical_requirements"]["training"] = {
        "learning_rate": "Not explicitly reported; RMSprop per Algorithm 1.",
        "num_epochs": None,
        "paper_section": "Section III-B",
    }
    null_run = tmp_path / "null"
    (null_run / ".pipeline").mkdir(parents=True)
    params_null = derive(spec_prose, ROOT, run_dir=null_run)["params"]
    assert params_null["max_epochs"]["value"] is None
    assert "inputs unavailable" in params_null["max_epochs"]["reasoning"]
    assert params_null["learning_rate"]["value"] is None


def test_analyzer_contract_carries_gap_family_structuring_rule():
    """The load-bearing half of the SRL params fix is analyzer-side: on a
    gap-family run the structured fields are the only parameter carriers,
    so the dispatch summaries (canonical + retry) and the agent file all
    state the structuring rule."""
    from pathlib import Path as _P

    from dispatch_templates import STAGE_TASK_SUMMARIES

    for key in ("stage_1_analyzer", "stage_1_analyzer_retry"):
        text = STAGE_TASK_SUMMARIES[key]
        assert "ONLY parameter carriers" in text, key
        assert "scale_dependent_hyperparameters" in text, key
        assert "never prose" in text, key
        assert "Use exactly ONE paper-value carrier" in text, key
        assert "SCALE-FREE stated constant" in text, key
        assert "DATA-SCALE-DEPENDENT value" in text, key
        assert "meaning-only with paper_value=null" in text, key
        assert "Never repeat one parameter's paper value" in text, key
    agent_md = _P(".opencode/agents/r2c-method-analyzer.md").read_text()
    assert "ONLY parameter carriers" in agent_md
    assert "number or null, never a prose sentence" in agent_md
    assert "Use one paper-value carrier, never two" in agent_md
    assert "One blessed paper-value carrier per parameter" in agent_md
    assert "scale-free stated constant belongs in the glossary only" in agent_md
    assert "Data-scale-dependent values live in the lane only" in agent_md
    assert "Preserve case: `K` and `k` are distinct" in agent_md


def test_analyzer_contract_requires_role_typed_tsf_evaluation_protocol():
    """R2C-082: both analyzer dispatch modes and the durable agent contract
    carry the same role/provenance rule; a retry must not recreate K=1/K=26
    after the canonical attempt was rejected."""
    from pathlib import Path as _P

    from dispatch_templates import STAGE_TASK_SUMMARIES

    contracts = {
        key: STAGE_TASK_SUMMARIES[key]
        for key in ("stage_1_analyzer", "stage_1_analyzer_retry")
    }
    contracts["agent"] = _P(
        ".opencode/agents/r2c-method-analyzer.md"
    ).read_text(encoding="utf-8")

    for name, raw in contracts.items():
        text = " ".join(raw.split())
        assert "comparison.evaluation_protocol" in text, name
        for role in (
            "context_length",
            "forecast_call_horizon",
            "validation_span",
            "test_span",
        ):
            assert role in text, (name, role)
        for field in (
            "parameter_name",
            "unit",
            "granularity",
            "paper_value_status",
            "evidence_quote",
            "paper_element_ids",
        ):
            assert field in text, (name, field)
        assert "positive numeric" in text, name
        assert "VERBATIM" in text, name
        assert "paper_unspecified" in text, name
        assert "protocol_role" in text, name
        assert "T+1" in text, name
        assert (
            "MUST NOT appear in "
            "`critical_requirements.scale_dependent_hyperparameters`"
        ) in text or (
            "MUST NOT appear in "
            "critical_requirements.scale_dependent_hyperparameters"
        ) in text, name


# ---------------------------------------------------------------------------
# Top-level packs (extends: null) — the fedavg 2026-07-06 gap run. The
# halt-judge's diagnosis, verbatim class: "the pack is skipped and never
# grafted, so serves('federated_learning') returns None. The analyzer ...
# cannot fix a taxonomy overlay mechanism limitation."
# ---------------------------------------------------------------------------

FEDAVG_SHAPED_PACK = {
    "schema_version": "1.0",
    "status": "provisional",
    "legacy_paradigm": "federated_learning",
    "extends": None,
    "taxonomy_id": "PROVISIONAL/federated_learning",
    "fingerprint": {
        "what_it_is": "Client-server distributed training: local SGD on "
        "private client data, server aggregates by data-size-weighted "
        "model averaging.",
        "not_this": [],
    },
    "scaffold_hints": {
        "interface_hint": "train_federated(model_fn, clients, rounds, "
        "client_fraction, local_epochs, seed) -> TrainedModel",
    },
    "semantic_checks": [],
    "smoke_bugs": [],
}


def test_overlay_grafts_top_level_pack_under_provisional_group(tmp_path):
    packs_dir = _install_pack(tmp_path, FEDAVG_SHAPED_PACK, "20260707-fedavg")
    tax = load_taxonomy(ROOT, provisional_packs_dir=packs_dir)

    node = serves("federated_learning", tax)
    assert isinstance(node, VariantNode)
    assert node.provisional is True
    assert node.is_populated
    assert node.root_id == taxonomy.PROVISIONAL_GROUP_ID
    # The synthetic group never leaks into the committed view.
    committed = load_taxonomy(ROOT)
    assert taxonomy.PROVISIONAL_GROUP_ID not in committed.roots
    assert serves("federated_learning", committed) is None
    # And the effective node is self-contained (no inherited surprises).
    fields = effective_node(tax, node.slug)
    assert isinstance(fields, dict)


def test_top_level_pack_with_placeholder_contract_stays_unserved(tmp_path):
    """The LIVE fedavg pack shape: the field-guide author left the
    template TODO as the interface hint. A placeholder is not a
    contract — the pack grafts (visible, diagnosable) but must not read
    populated, so serves() refuses and the consumer gate fails honestly
    instead of scaffolding against a TODO."""
    pack = {**FEDAVG_SHAPED_PACK,
            "scaffold_hints": {"interface_hint": "TODO: describe the "
                               "pluggable interface this paper needs."}}
    packs_dir = _install_pack(tmp_path, pack, "20260707-fedavg-todo")
    tax = load_taxonomy(ROOT, provisional_packs_dir=packs_dir)

    assert serves("federated_learning", tax) is None
    grafted = tax._node_by_legacy.get("federated_learning")
    assert grafted is not None and not grafted.is_populated

# ---------------------------------------------------------------------------
# R2C-032 — authoring-time build-plan choice (approved 2026-07-28). A
# provisional pack may declare `build_plan: {source, reasoning}`: `neutral`
# routes to the generic provisional plan (the SRL 2026-07-28 stage-2b halt:
# the implicitly inherited motion-planning manifest demanded SystemDynamics/
# CollisionModel classes an RL method cannot have); `inherit_parent` and an
# absent declaration both keep today's ancestor walk, byte-identically.
# Every derived plan now carries `plan_key` provenance for the delivery-label
# cap.
# ---------------------------------------------------------------------------

PACK_NEUTRAL = {
    **GAP_SHAPED_PACK,
    "build_plan": {
        "source": "neutral",
        "reasoning": "SA-CADRL is an RL method; the motion-planning manifest "
                     "(SystemDynamics/CollisionModel/"
                     "precompute_motion_primitives) does not describe its "
                     "build shape.",
    },
    "family_components": {
        "reward_function": {
            "required": True,
            "description": "Norm-inducing reward the policy optimizes.",
        },
    },
}

PACK_INHERIT = {
    **GAP_SHAPED_PACK,
    "build_plan": {
        "source": "inherit_parent",
        "reasoning": "planner-shaped child; the parent manifest fits",
    },
}

GEOMETRIC_SHAPED_PACK = {
    "schema_version": "1.0",
    "status": "provisional",
    "legacy_paradigm": "active_learning/geometric",
    "extends": "active_learning",
    "taxonomy_id": "TE-TS/active_learning/geometric",
    "fingerprint": {"what_it_is": "Geometric active-learning selection."},
    "scaffold_hints": {
        "interface_hint": "select_batch(model, x_unlabeled, batch_size, seed)",
    },
    "semantic_checks": [],
    "smoke_bugs": [],
}


def _geometric_spec_dict() -> dict:
    return {
        "comparison": {
            "classification": {"id": "active_learning/geometric"},
            "pluggable_component": {
                "name": "select_batch",
                "signature": "select_batch(model, x_unlabeled, batch_size, seed=0)",
            },
        },
    }


def _manifest_symbol_names(plan: dict) -> set[str]:
    return {
        s.get("name")
        for entry in plan["package_manifest"]["files"]
        for s in entry.get("public_symbols") or []
    }


PACK_NEUTRAL_EMPTY_TRAINING = {
    **PACK_NEUTRAL,
    "package_manifest": {
        "description": "KBP-shaped ownership: no training functions.",
        "files": [
            {"path": "method/training.py", "produced_by": None,
             "public_symbols": []},
            # A NON-empty entry is authoring documentation: it must not
            # rewrite the plan's enforced symbols (the merge's non-empty
            # shape is owed its own second concrete case).
            {"path": "method/model.py", "produced_by": "architecture_coder",
             "public_symbols": ["build_extraction_pipeline"]},
        ],
    },
}


def test_pack_declared_empty_file_drops_neutral_placeholder(tmp_path):
    # The DomIndOnto KBP stage-2d halt (2026-07-29): the pack declared
    # training.py producer-less and symbol-less, the neutral plan's
    # <training_functions> placeholder survived the derivation, and the
    # finalizer demanded functions of a deliberately docstring-only file.
    from scripts.build_plan import load_build_plan

    run_dir = tmp_path / "run"
    packs_dir = _install_run_pack(run_dir, PACK_NEUTRAL_EMPTY_TRAINING)
    plan = load_build_plan(_gap_spec_dict(), ROOT,
                           provisional_packs_dir=packs_dir)
    files = {e["path"]: e for e in plan["package_manifest"]["files"]}
    # The explicit empty declaration is binding: no placeholder row
    # survives for training.py.
    assert files["method/training.py"]["public_symbols"] == []
    # The non-empty model.py declaration stays unconsumed: the neutral
    # placeholder is untouched.
    assert [s["name"] for s in files["method/model.py"]["public_symbols"]] \
        == ["<MethodComponentClass>"]


def test_pack_without_manifest_keeps_placeholders_byte_identically(tmp_path):
    from scripts.build_plan import load_build_plan

    run_dir = tmp_path / "run"
    packs_dir = _install_run_pack(run_dir, PACK_NEUTRAL)
    plan = load_build_plan(_gap_spec_dict(), ROOT,
                           provisional_packs_dir=packs_dir)
    files = {e["path"]: e for e in plan["package_manifest"]["files"]}
    assert [s["name"] for s in files["method/training.py"]["public_symbols"]] \
        == ["<training_functions>"]


def test_finalizer_accepts_declared_empty_training(tmp_path):
    # End-to-end over the halted shape: a docstring-only training.py
    # finalizes cleanly when the pack declares the file symbol-less,
    # and WITHOUT the declaration the same package still fails R2 —
    # both sides of the contract.
    import json
    import subprocess
    import sys as _sys

    def _seed(run_dir):
        method_dir = run_dir / "method"
        method_dir.mkdir(parents=True)
        (method_dir / "model.py").write_text(
            '"""Model."""\n\n\nclass PipelineComponent:\n'
            '    """Component."""\n')
        (method_dir / "training.py").write_text(
            '"""No training functions: KBP-shaped paradigm."""\n')
        (method_dir / "method.py").write_text(
            '"""Method."""\n\n\n'
            'def plan(start, goal, environment, dynamics, seed=0):\n'
            '    """Plan."""\n    return None\n')
        (method_dir / "data.py").write_text(
            '"""Data."""\n\n\ndef load_data(path=None, *, seed=0):\n'
            '    """Load."""\n    return {}\n')
        spec_path = run_dir / ".pipeline" / "method_spec.json"
        spec_path.write_text(json.dumps(_gap_spec_dict()),
                             encoding="utf-8")
        return spec_path

    def _finalize(run_dir, spec_path):
        return subprocess.run(
            [_sys.executable, "scripts/finalize_package_init.py",
             "--spec", str(spec_path), "--run-dir", str(run_dir)],
            cwd=ROOT, capture_output=True, text=True, timeout=60)

    declared = tmp_path / "declared"
    _install_run_pack(declared, PACK_NEUTRAL_EMPTY_TRAINING)
    proc = _finalize(declared, _seed(declared))
    assert proc.returncode == 0, proc.stderr
    init_text = (declared / "method" / "__init__.py").read_text(
        encoding="utf-8")
    assert "plan" in init_text

    undeclared = tmp_path / "undeclared"
    _install_run_pack(undeclared, PACK_NEUTRAL)
    proc = _finalize(undeclared, _seed(undeclared))
    assert proc.returncode == 2
    assert "no public top-level functions" in proc.stderr


def test_declared_neutral_takes_provisional_plan(tmp_path):
    from scripts.build_plan import PROVISIONAL_PLAN_KEY, load_build_plan

    run_dir = tmp_path / "run"
    packs_dir = _install_run_pack(run_dir, PACK_NEUTRAL)
    plan = load_build_plan(_gap_spec_dict(), ROOT,
                           provisional_packs_dir=packs_dir)
    assert plan is not None
    assert plan["plan_key"] == PROVISIONAL_PLAN_KEY

    # The manifest is the neutral provisional shape, not the parent's: none
    # of the classes the SRL halt shows the parent demanding, a flexible
    # model.py, and producer-defined training functions.
    names = _manifest_symbol_names(plan)
    assert "<SystemDynamics>" not in names
    assert "<CollisionModel>" not in names
    assert "precompute_motion_primitives" not in names
    files = {e["path"]: e for e in plan["package_manifest"]["files"]}
    assert files["method/model.py"]["class_count"] == "flexible"
    assert [s["name"] for s in files["method/training.py"]["public_symbols"]] \
        == ["<training_functions>"]

    # The pack's own family_components still merge — the declaration surface
    # is plan-key independent (the load-bearing composition of pieces 1+2).
    declared = plan["arch_contract_requirements"]["family_components"]
    assert set(declared) == {"reward_function"}
    assert declared["reward_function"]["required"] is True

    # Identity fields still bind the run's provisional node.
    assert plan["paradigm_id"] == GAP_PARADIGM_ID
    assert plan["taxonomy_id"] == "CLC-MP/synthetic_gap_variant"
    assert plan["interface_hint"].startswith("plan(")


def test_declared_inherit_parent_keeps_ancestor_plan(tmp_path):
    from scripts.build_plan import load_build_plan

    inherit_run = tmp_path / "inherit"
    packs_inherit = _install_run_pack(inherit_run, PACK_INHERIT)
    plan_inherit = load_build_plan(_gap_spec_dict(), ROOT,
                                   provisional_packs_dir=packs_inherit)
    assert plan_inherit is not None
    assert plan_inherit["plan_key"] == "motion_planning"
    names = _manifest_symbol_names(plan_inherit)
    assert "<SystemDynamics>" in names
    assert "precompute_motion_primitives" in names

    # Declaring inherit_parent is a no-op relative to the walk: byte-equal
    # to the plan the undeclared pack gets.
    undeclared_run = tmp_path / "undeclared"
    packs_undeclared = _install_run_pack(undeclared_run, GAP_SHAPED_PACK)
    plan_undeclared = load_build_plan(_gap_spec_dict(), ROOT,
                                      provisional_packs_dir=packs_undeclared)
    assert plan_inherit == plan_undeclared


def test_undeclared_pack_resolves_exactly_as_today(tmp_path):
    from scripts.build_plan import PROVISIONAL_PLAN_KEY, load_build_plan

    # Committed-ancestor shape (today's installed SRL pack): the walk lands
    # on motion_planning, now recorded as plan-key provenance.
    run_dir = tmp_path / "run"
    packs_dir = _install_run_pack(run_dir, GAP_SHAPED_PACK)
    plan = load_build_plan(_gap_spec_dict(), ROOT,
                           provisional_packs_dir=packs_dir)
    assert plan["plan_key"] == "motion_planning"

    # No-committed-ancestor shape (the fedavg/DomIndOnto mid-July shape):
    # today's fallback is the generic provisional plan, and the provenance
    # says so — this is exactly the shape the delivery-label cap keys on.
    reserved_run = tmp_path / "reserved"
    packs_reserved = _install_run_pack(reserved_run, RESERVED_SHAPED_PACK)
    spec = {
        "comparison": {
            "classification": {"id": "single_agent_rl"},
            "pluggable_component": {
                "name": "act",
                "signature": "act(observation, seed=0) -> Action",
            },
        },
    }
    plan_reserved = load_build_plan(spec, ROOT,
                                    provisional_packs_dir=packs_reserved)
    assert plan_reserved is not None
    assert plan_reserved["plan_key"] == PROVISIONAL_PLAN_KEY


def test_neutral_declaration_wins_over_al_family_branch(tmp_path):
    from scripts.build_plan import PROVISIONAL_PLAN_KEY, load_build_plan

    # A gap pack under an active-learning parent that opts out of family
    # build conventions must actually get the neutral plan — routing it into
    # the AL-derived plan anyway would reintroduce the SRL failure shape one
    # family over.
    neutral_run = tmp_path / "neutral"
    pack = {**GEOMETRIC_SHAPED_PACK,
            "build_plan": {"source": "neutral",
                           "reasoning": "the AL build shape does not fit"}}
    packs_dir = _install_run_pack(neutral_run, pack)
    plan = load_build_plan(_geometric_spec_dict(), ROOT,
                           provisional_packs_dir=packs_dir)
    assert plan is not None
    assert plan["plan_key"] == PROVISIONAL_PLAN_KEY

    # Mirror: an undeclared AL descendant resolves through the static
    # ancestor walk (B-02 migration). DELIBERATE behavior change, decided at
    # landing: plan_key records the committed ancestor entry the plan was
    # based on ("active_learning"), no longer the descendant's own id — the
    # same provenance rule every other family's gap descendants follow.
    plain_run = tmp_path / "plain"
    packs_plain = _install_run_pack(plain_run, GEOMETRIC_SHAPED_PACK)
    plan_plain = load_build_plan(_geometric_spec_dict(), ROOT,
                                 provisional_packs_dir=packs_plain)
    assert plan_plain is not None
    assert plan_plain["plan_key"] == "active_learning"
    assert plan_plain["paradigm_id"] == "active_learning/geometric"
    data_entry = next(e for e in plan_plain["package_manifest"]["files"]
                      if e["path"] == "method/data.py")
    assert "pool_size" in data_entry["public_symbols"][0]["signature"]


def test_committed_resolution_ignores_build_plan_reads(monkeypatch):
    # Byte-identical pin for committed nodes: the declared-source read is
    # gated on node.provisional, so it must be unreachable for every
    # committed plan — proven by making the read explode and comparing
    # whole plans.
    import scripts.build_plan as bp

    def _boom(*args, **kwargs):
        raise AssertionError(
            "the declared build-plan read reached a committed node")

    committed_ids = [pid for pid in bp.STATIC_PLAN_BY_PARADIGM
                     if pid != bp.PROVISIONAL_PLAN_KEY]
    for pid in committed_ids:
        spec = {"comparison": {"classification": {"id": pid},
                               "pluggable_component": {"name": "x"}}}
        live = bp.load_build_plan(spec, ROOT)
        with monkeypatch.context() as mp:
            mp.setattr(bp, "_declared_build_plan_source", _boom)
            patched = bp.load_build_plan(spec, ROOT)
        assert live == patched, pid
        if live is not None:
            # Committed provenance is always the node's own id — never the
            # provisional key, so the label cap can never touch it.
            assert live["plan_key"] == pid


def test_overlay_carries_build_plan_declaration(tmp_path):
    # Pins the _INHERITABLE_MAPPING_FIELDS + implementation-bucket plumbing
    # so a future field-list refactor cannot silently drop the key (the
    # family_components mirror of this test lives in
    # test_family_components.py::test_family_components_inherit_and_child_overrides).
    packs_dir = _install_pack(tmp_path, PACK_NEUTRAL)
    tax = load_taxonomy(ROOT, provisional_packs_dir=packs_dir)
    node = serves(GAP_PARADIGM_ID, tax)
    assert node is not None
    block = taxonomy.node_implementation(node, tax).get("build_plan")
    assert isinstance(block, dict)
    assert block["source"] == "neutral"
