"""B-02: active_learning on the static build-plan mechanism (decision 6).

Golden plans were captured from the retired bespoke path
(derive_active_learning_build_plan) at migration time and committed under
tests/fixtures/build_plans/. The static path must reproduce them
byte-identically — the red-teamed output-diff gate, made durable. No frozen
copy of the deleted function and no git-show pinning: the goldens are plain
tracked JSON.
"""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

from scripts.build_plan import PROVISIONAL_PLAN_KEY, load_build_plan

ROOT = Path(__file__).resolve().parent.parent
GOLDEN = ROOT / "tests" / "fixtures" / "build_plans"


def _golden(name: str) -> dict:
    return json.loads((GOLDEN / name).read_text(encoding="utf-8"))


def _spec(name: str) -> dict:
    # method_spec.json snapshots of the two starter papers' runs.
    return json.loads((GOLDEN / f"spec_{name}.json").read_text(encoding="utf-8"))


def test_bayesian_spec_reproduces_the_bespoke_plan():
    plan = load_build_plan(_spec("bayesian-active-learning"))
    assert plan == _golden("golden_bayesian-active-learning.json")


def test_batch_acquisition_spec_reproduces_the_bespoke_plan():
    plan = load_build_plan(_spec("deep-batch-active-learning"))
    assert plan == _golden("golden_deep-batch-active-learning.json")


def test_bare_family_id_reproduces_the_bespoke_plan():
    plan = load_build_plan({
        "comparison": {"classification": {"id": "active_learning"},
                       "pluggable_component": {"name": "select_batch"}}})
    assert plan == _golden("golden_bare-family.json")


def test_same_name_different_signature_collision_prefers_the_paper():
    """Maintainer decision 2026-08-18: a spec restating a manifest-declared
    method with a DIFFERENT signature is flagged as a structured collision
    and the paper-declared (spec) signature wins. The bespoke path used to
    append a duplicate; plain static name-dedup used to silently DROP the
    spec-declared interface — both wrong."""
    spec = {
        "comparison": {"classification": {"id": "active_learning"},
                       "pluggable_component": {"name": "select_batch"}},
        "critical_requirements": {"required_model_methods": [
            {"signature":
             "forward(self, x: torch.Tensor, *, return_embedding: bool = False)"
             " -> torch.Tensor"},
        ]},
    }
    plan = load_build_plan(spec)
    model_entry = next(e for e in plan["package_manifest"]["files"]
                       if e["path"] == "method/model.py")
    methods = model_entry["public_symbols"][0]["required_methods"]
    assert methods == [
        "forward(self, x: torch.Tensor, *, return_embedding: bool = False)"
        " -> torch.Tensor"
    ], "paper-declared signature must replace the manifest default"
    assert plan["signature_collisions"] == [{
        "method": "forward",
        "manifest_signature":
            "forward(self, x: torch.Tensor) -> torch.Tensor",
        "spec_signature":
            "forward(self, x: torch.Tensor, *, return_embedding: bool = False)"
            " -> torch.Tensor",
        "resolution": "paper_declared",
    }]


def test_agreeing_restated_signature_is_not_a_collision():
    spec = {
        "comparison": {"classification": {"id": "active_learning"},
                       "pluggable_component": {"name": "select_batch"}},
        "critical_requirements": {"required_model_methods": [
            {"signature": "forward(self, x: torch.Tensor) -> torch.Tensor"},
        ]},
    }
    plan = load_build_plan(spec)
    assert "signature_collisions" not in plan


# ---------------------------------------------------------------------------
# Gap-path corpus: the four provisional-pack declaration shapes (B-02 rev 2).
# The live R2C-032 route, not synthetic: an AL descendant now resolves
# through the static ancestor walk instead of the deleted family branch.
# ---------------------------------------------------------------------------

_BASE_PACK = {
    "schema_version": "1.0",
    "status": "provisional",
    "legacy_paradigm": "active_learning/corpus_case",
    "extends": "active_learning",
    "taxonomy_id": "TE-TS/active_learning/corpus_case",
    "fingerprint": {"what_it_is": "Gap-path corpus case."},
    "scaffold_hints": {
        "interface_hint": "select_batch(model, x_unlabeled, batch_size, seed)",
    },
    "semantic_checks": [],
    "smoke_bugs": [],
}

_CORPUS_SPEC = {
    "comparison": {
        "classification": {"id": "active_learning/corpus_case"},
        "pluggable_component": {
            "name": "select_batch",
            "signature": "select_batch(model, x_unlabeled, batch_size, seed=0)",
        },
    },
}


def _install(run_dir: Path, pack: dict) -> Path:
    from tests.test_provisional_pack_overlay import _install_run_pack

    return _install_run_pack(run_dir, pack)


def _plan_for(tmp_path: Path, pack: dict) -> dict:
    packs_dir = _install(tmp_path / "run", pack)
    plan = load_build_plan(
        deepcopy(_CORPUS_SPEC), ROOT, provisional_packs_dir=packs_dir)
    assert plan is not None
    return plan


def test_gap_path_undeclared_walks_to_the_family_plan(tmp_path):
    plan = _plan_for(tmp_path, deepcopy(_BASE_PACK))
    assert plan["plan_key"] == "active_learning"
    assert plan["paradigm_id"] == "active_learning/corpus_case"
    training = next(e for e in plan["package_manifest"]["files"]
                    if e["path"] == "method/training.py")
    names = [s["name"] for s in training["public_symbols"]]
    assert names == ["build_model", "train_from_scratch"]


def test_gap_path_inherit_parent_matches_undeclared(tmp_path):
    pack = deepcopy(_BASE_PACK)
    pack["build_plan"] = {"source": "inherit_parent",
                          "reasoning": "family shape fits"}
    plan = _plan_for(tmp_path, pack)
    undeclared = _plan_for(tmp_path / "b", deepcopy(_BASE_PACK))
    assert plan == undeclared


def test_gap_path_neutral_takes_the_provisional_plan(tmp_path):
    pack = deepcopy(_BASE_PACK)
    pack["build_plan"] = {"source": "neutral",
                          "reasoning": "family conventions do not fit"}
    plan = _plan_for(tmp_path, pack)
    assert plan["plan_key"] == PROVISIONAL_PLAN_KEY


def test_gap_path_empty_file_declaration_now_reaches_al_descendants(tmp_path):
    """KNOWN LIMITATION, pinned deliberately: after the migration,
    _merge_pack_declared_empty_files applies to AL descendants, so a pack
    declaring empty public symbols for training.py empties that manifest
    entry while AL_ARCH_CONTRACT_REQUIREMENTS still demands
    train_from_scratch — an internally contradictory plan the 2.b gate
    would fail. The contradiction is visible here rather than silent; a
    coherence check across the two plan halves is future work, not this
    migration's scope."""
    pack = deepcopy(_BASE_PACK)
    pack["package_manifest"] = {
        "files": [{"path": "method/training.py", "public_symbols": []}],
    }
    plan = _plan_for(tmp_path, pack)
    training = next(e for e in plan["package_manifest"]["files"]
                    if e["path"] == "method/training.py")
    assert training.get("public_symbols") == []
    required = plan["arch_contract_requirements"]["required_blocks"]
    assert required["training_loop.function_name"] == {
        "must_equal": "train_from_scratch"}


# ---------------------------------------------------------------------------
# Driver-side recording (the assumptions.md half of the decision)
# ---------------------------------------------------------------------------


def test_driver_records_collisions_in_assumptions(tmp_path):
    import run_pipeline
    from tests.helpers.state import make_state

    state = make_state(tmp_path / "run")
    state.paths.method_spec.write_text(json.dumps({
        "comparison": {"classification": {"id": "active_learning"},
                       "pluggable_component": {"name": "select_batch"}},
        "critical_requirements": {"required_model_methods": [
            {"signature":
             "forward(self, x: torch.Tensor, *, return_embedding: bool = "
             "False) -> torch.Tensor"},
        ]},
    }), encoding="utf-8")
    state.paths.repo_root = ROOT

    run_pipeline._record_build_plan_signature_collisions(state, "stage_2b")

    import run_layout

    assumptions = run_layout.run_path(
        state.paths.run_dir, run_layout.ASSUMPTIONS_MD).read_text(
        encoding="utf-8")
    assert "signature collision" in assumptions.lower()
    assert "return_embedding" in assumptions


def test_driver_recording_is_silent_without_collisions_or_spec(tmp_path):
    import run_pipeline
    import run_layout
    from tests.helpers.state import make_state

    state = make_state(tmp_path / "run")
    state.paths.repo_root = ROOT
    # No spec on disk at all: must not raise.
    state.paths.method_spec.unlink(missing_ok=True)
    run_pipeline._record_build_plan_signature_collisions(state, "stage_2b")
    # Clean spec: no assumptions entry.
    state.paths.method_spec.write_text(json.dumps({
        "comparison": {"classification": {"id": "active_learning"},
                       "pluggable_component": {"name": "select_batch"}},
    }), encoding="utf-8")
    run_pipeline._record_build_plan_signature_collisions(state, "stage_2b")
    assert not run_layout.run_path(
        state.paths.run_dir, run_layout.ASSUMPTIONS_MD).exists()
