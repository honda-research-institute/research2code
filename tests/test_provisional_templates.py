"""Item 4 — stage-2a scaffolding for provisional paradigms.

Design: the stage 2a provisional templates design note (internal, not shipped)
(tests-to-pass form). The neutral committed skeleton at
paradigms/_provisional/templates/ closes the gap path's last unbuilt seam:
a run-local provisional pack with no committed-ancestor templates scaffolds
through stage 2a instead of halting (the fedavg 2026-07-07 00:03 halt and
the SRL 2026-07-14 stage_2a.halt).
"""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from scripts import taxonomy
from scripts.taxonomy import load_taxonomy, serves

ROOT = Path(__file__).resolve().parents[1]

# The SRL 2026-07-14 shape: a pack populating the reserved single_agent_rl
# stub, whose family (CLC-RL) carries no scaffold_hints — no templates_dir
# anywhere on the inheritance chain.
GAP_PACK = {
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
        "interface_hint": "train_policy(env, value_network, seed) -> TrainedPolicy",
    },
    "pluggable_component": {
        "name": "train_policy",
        "signature_template": (
            "train_policy(env, value_network, seed, *, "
            "num_episodes: int = 3000) -> TrainedPolicy"
        ),
        "contract": {
            "fixed_positional_args": ["env", "value_network"],
            "seed_param": "seed",
            "forbidden_param_names": [],
        },
    },
    "semantic_checks": [],
    "smoke_bugs": [],
}


# The DomIndOnto 2026-07-21 roll-1 interface hint, verbatim (extracted from
# the overnight batch log's slots dump — the run dir itself is disposable).
# The pack shipped the hint as a signature PLUS a full triple-quoted NumPy
# docstring and declared no pluggable_component.signature_template, so
# `_build_slots`'s hint fallback interpolated the docstring terminators
# inside data.py's module docstring and the module stopped parsing
# (stage-2a scaffold-validator halt: invalid syntax, data.py line 17).
DOMINDONTO_0721_BAD_HINT = '''populate_knowledge_base(
    texts,
    ontology_path,
    mapping_rules_path,
    disambiguation_reference_path,
    seed=None
) -> List[Triple]
"""Extract entities, events, and relations from text and produce
ontology-mapped triples in a knowledge base.

Parameters
----------
texts : List[str]
    Input documents to extract from.
ontology_path : str
    Path to the OWL ontology file that defines the target domain schema.
mapping_rules_path : str
    Path to the mapping rules file that maps extracted types to
    ontology classes (target_type, triggers, argument_mappings per rule).
disambiguation_reference_path : str
    Path to the abbreviation/equivalence reference file used for
    cross-document entity deduplication.
seed : int, optional
    Random seed for stochastic components (classifier-based mapping).

Returns
-------
List[Triple]
    List of (subject, predicate, object) triples mapped to the ontology.
"""'''


def _install_run_pack(run_dir: Path, pack: dict | None = None) -> Path:
    install_dir = run_dir / ".pipeline" / "provisional_packs" / "20260713-single_agent_rl"
    install_dir.mkdir(parents=True)
    (install_dir / "pack.yaml").write_text(
        yaml.safe_dump(pack or GAP_PACK), encoding="utf-8")
    return run_dir / ".pipeline" / "provisional_packs"


def _gap_spec_dict() -> dict:
    return {
        "paper": {"title": "SRL-shaped gap paper", "authors": "Test et al."},
        "core_method": {
            "name": "SRL",
            "summary": "Single-agent deep RL for collision avoidance.",
        },
        "comparison": {
            "classification": {"id": "single_agent_rl"},
            "pluggable_component": {
                "name": "train_policy",
                "signature": "train_policy(env, value_network, seed, *, num_episodes=3000)",
            },
        },
        "critical_requirements": {"scale_dependent_hyperparameters": []},
    }


def _typed_gap_arch_contract(*, class_name: str = "ValueNetwork") -> dict:
    """Current Stage-2b output for the provisional single-agent-RL shape."""
    opaque_policy = {
        "kind": "opaque",
        "type_description": "TrainedPolicy",
        "reason": "The policy is a structured generated-package object.",
    }
    seed = {
        "kind": "scalar",
        "dtype": "int64",
        "source": {"literal": 0, "display_symbol": "seed"},
    }
    return {
        "schema_version": "2.0.0",
        "paradigm_id": "single_agent_rl",
        "dimensions": {
            "observation_feature_count": {
                "expression": {"kind": "literal", "value": 14},
            },
        },
        "data_loader": {"load_data_returns": {}},
        "architecture": {
            "value_network": {
                "class_name": class_name,
                "constructor_args": {},
                "forward": {
                    "input": {
                        "observation": {
                            "kind": "tensor",
                            "dtype": "float32",
                            "dimensions": [{
                                "dimension": "observation_feature_count",
                                "display_symbol": "obs_dim",
                            }],
                        },
                    },
                    "output": {
                        "kind": "scalar",
                        "dtype": "float32",
                        "source": {"literal": 0.0},
                    },
                },
            },
        },
        "pluggable_component": {
            "name": "train_policy",
            "input": {"seed": seed},
            "output": opaque_policy,
        },
        "training_loop": {
            "function_name": "run_training_episodes",
            "input": {"seed": seed},
            "output": opaque_policy,
        },
    }


def _scaffold_gap_run(tmp_path: Path) -> Path:
    from scripts.scaffold_package import scaffold

    run_dir = tmp_path / "run"
    _install_run_pack(run_dir)
    spec_path = run_dir / ".pipeline" / "method_spec.json"
    spec_path.write_text(json.dumps(_gap_spec_dict()), encoding="utf-8")
    rc = scaffold(spec_path, run_dir, ROOT)
    assert rc == 0, f"scaffold exited {rc}"
    return run_dir


def test_gap_pack_without_templates_scaffolds_neutral_skeleton(tmp_path):
    """Design test 1 — the 00:03 halt, closed: no halt, standard files
    present at the consolidated locations."""
    run_dir = _scaffold_gap_run(tmp_path)
    for rel in ("method/README.md", "method/data.py",
                "method/example_data/README.md"):
        assert (run_dir / rel).is_file(), f"missing {rel}"


def test_rendered_files_carry_the_authored_interface(tmp_path):
    """Design test 2 — slots carry the pack's declared interface; no slot
    literals, no TODO placeholders in any rendered file."""
    run_dir = _scaffold_gap_run(tmp_path)
    rendered = {
        rel: (run_dir / rel).read_text(encoding="utf-8")
        for rel in ("method/README.md", "method/data.py",
                    "method/example_data/README.md")
    }
    readme = rendered["method/README.md"]
    assert "train_policy" in readme
    assert "SRL-shaped gap paper" in readme
    assert "num_episodes" in readme  # the full signature_template landed
    assert "train_policy" in rendered["method/data.py"]
    for rel, text in rendered.items():
        assert "{{" not in text, f"{rel} has unrendered slot literals"
        assert "todo" not in text.lower(), f"{rel} carries TODO placeholders"


def test_docstring_bearing_hint_scaffolds_parseable_module(tmp_path):
    """The DomIndOnto 2026-07-21 roll-1 halt, closed at the interpolation
    site: a pack whose interface_hint carries its own triple-quoted
    docstring (and no pluggable_component to shadow the hint fallback)
    must still scaffold a data.py that parses — the sanitizer drops the
    embedded docstring, keeps the signature, and the deterministic
    stage-2a validator accepts the tree."""
    import ast

    from scripts.scaffold_package import scaffold
    from scripts.validate_scaffolder_output import validate

    bad_pack = {k: v for k, v in GAP_PACK.items() if k != "pluggable_component"}
    bad_pack["scaffold_hints"] = {"interface_hint": DOMINDONTO_0721_BAD_HINT}

    run_dir = tmp_path / "run"
    _install_run_pack(run_dir, bad_pack)
    spec_path = run_dir / ".pipeline" / "method_spec.json"
    spec_path.write_text(json.dumps(_gap_spec_dict()), encoding="utf-8")
    rc = scaffold(spec_path, run_dir, ROOT)
    assert rc == 0, f"scaffold exited {rc}"

    data_py = (run_dir / "method" / "data.py").read_text(encoding="utf-8")
    ast.parse(data_py)  # the roll-1 failure mode: invalid syntax at line 17
    # The signature survived; the embedded docstring did not.
    assert "populate_knowledge_base(" in data_py
    assert "Extract entities, events" not in data_py
    # The hint-derived interface name rendered clean into the README.
    readme = (run_dir / "method" / "README.md").read_text(encoding="utf-8")
    assert "populate_knowledge_base" in readme
    assert validate(_gap_spec_dict(), run_dir, ROOT) == []


def test_multiline_signature_only_hint_survives_interpolation(tmp_path):
    """The known-good multi-line shape (signature spanning lines, no
    docstring) is untouched by the sanitizer and lands verbatim in the
    scaffolded module."""
    import ast

    from scripts.scaffold_package import scaffold

    hint = (
        "select_batch(\n"
        "    model,\n"
        "    x_unlabeled,\n"
        "    batch_size,\n"
        "    seed=None,\n"
        ") -> SelectedIndices"
    )
    pack = {k: v for k, v in GAP_PACK.items() if k != "pluggable_component"}
    pack["scaffold_hints"] = {"interface_hint": hint}

    run_dir = tmp_path / "run"
    _install_run_pack(run_dir, pack)
    spec_path = run_dir / ".pipeline" / "method_spec.json"
    spec_path.write_text(json.dumps(_gap_spec_dict()), encoding="utf-8")
    rc = scaffold(spec_path, run_dir, ROOT)
    assert rc == 0, f"scaffold exited {rc}"

    data_py = (run_dir / "method" / "data.py").read_text(encoding="utf-8")
    ast.parse(data_py)
    assert hint in data_py  # interpolated unchanged


def test_sanitize_interface_hint_shapes():
    """Unit coverage for the sanitizer itself: no-op on the known-good
    shapes, docstring-stripping on the DomIndOnto roll-1 fixture, and
    terminator removal when the value LEADS with a docstring (nothing to
    truncate to)."""
    from scripts.scaffold_package import _sanitize_interface_hint

    # Roll 2's actual clean one-line hint: untouched.
    one_line = ("run_pipeline(documents, ontology_path, mapping_rules, "
                "seed, *paradigm_extras_by_name) -> KnowledgeBase")
    assert _sanitize_interface_hint(one_line) == one_line

    # Multi-line signature-only: untouched.
    multi_line = "select_batch(\n    model,\n    seed=None,\n) -> Indices"
    assert _sanitize_interface_hint(multi_line) == multi_line

    # The roll-1 fixture: truncated at the embedded docstring, signature kept.
    cleaned = _sanitize_interface_hint(DOMINDONTO_0721_BAD_HINT)
    assert cleaned == DOMINDONTO_0721_BAD_HINT.split('"""')[0].strip()
    assert '"""' not in cleaned and "'''" not in cleaned

    # Leading terminator (both spellings): content survives, terminators go.
    assert _sanitize_interface_hint('"""select_batch(model, seed)"""') == (
        "select_batch(model, seed)")
    assert _sanitize_interface_hint("'''select_batch(model, seed)'''") == (
        "select_batch(model, seed)")


def test_neutral_skeleton_unreachable_for_committed_paradigms():
    """Design test 3 — every committed populated paradigm resolves its own
    templates dir, never the neutral skeleton (committed behavior is
    byte-identical: the fallback only fires on provisional nodes)."""
    neutral = (ROOT / taxonomy.PROVISIONAL_TEMPLATES_DIR).resolve()
    assert neutral.is_dir()
    tax = load_taxonomy(ROOT)
    for pid in taxonomy.registered_paradigm_ids(ROOT):
        node = serves(pid, tax)
        if node is None:
            continue
        resolved = taxonomy.resolve_templates_dir(pid, ROOT, taxonomy=tax)
        if resolved is not None:
            assert resolved.resolve() != neutral, (
                f"committed paradigm {pid} resolved the provisional skeleton"
            )


def test_committed_node_without_templates_still_resolves_none(monkeypatch):
    """Design test 4 — silence on a committed node is a repo bug, not a new
    family: the neutral fallback must not mask it."""
    tax = load_taxonomy(ROOT)
    assert serves("active_learning", tax) is not None
    monkeypatch.setattr(
        taxonomy, "_node_effective_field", lambda *a, **k: {})
    assert taxonomy.resolve_templates_dir(
        "active_learning", ROOT, taxonomy=tax) is None


def test_generic_build_plan_serves_no_ancestor_packs(tmp_path):
    """Design test 5a — a provisional pack with no committed ancestor gets
    the generic plan: shared scaffolder rows, the pack's pluggable component
    substituted into method/method.py."""
    from scripts.build_plan import PROVISIONAL_PLAN_KEY, _static_plan_key, load_build_plan

    assert _static_plan_key("single_agent_rl", provisional=True) == PROVISIONAL_PLAN_KEY
    # Committed ids never reach the generic plan.
    assert _static_plan_key("single_agent_rl", provisional=False) is None

    run_dir = tmp_path / "run"
    packs_dir = _install_run_pack(run_dir)
    plan = load_build_plan(_gap_spec_dict(), ROOT, provisional_packs_dir=packs_dir)
    assert plan is not None
    assert plan["pluggable_component"]["name"] == "train_policy"
    files = {f["path"]: f for f in plan["package_manifest"]["files"]}
    assert files["method/data.py"]["produced_by"] == "package_scaffolder"
    assert files["method/model.py"]["class_count"] == "flexible"
    method_symbols = {s.get("name") for s in files["method/method.py"]["public_symbols"]}
    assert "train_policy" in method_symbols


def test_scaffolder_validator_accepts_neutral_tree(tmp_path):
    """Design test 5b — the stage-2a validator accepts the scaffolded tree
    against the generic manifest (file presence, rendered slots, Python
    parses, load_data signature matches)."""
    from scripts.validate_scaffolder_output import validate

    run_dir = _scaffold_gap_run(tmp_path)
    assert validate(_gap_spec_dict(), run_dir, ROOT) == []


def test_2b_validator_honors_generic_manifest_allowances(tmp_path):
    """The generic plan's flexible class count and producer-defined training
    functions pass the 2b gate for a multi-class provisional package."""
    from scripts.validate_architecture_coder_output import validate

    run_dir = tmp_path / "run"
    _install_run_pack(run_dir)
    method = run_dir / "method"
    method.mkdir(parents=True)
    (method / "model.py").write_text(
        "class ValueNetwork:\n"
        "    def evaluate(self, observation):\n"
        "        return 0.0\n"
        "class TargetNetwork:\n"
        "    def evaluate(self, observation):\n"
        "        return 0.0\n",
        encoding="utf-8",
    )
    (method / "training.py").write_text(
        "def run_training_episodes(env, value_network, seed, *, num_episodes=3000):\n"
        "    return value_network\n",
        encoding="utf-8",
    )
    contract = _typed_gap_arch_contract()
    (run_dir / ".pipeline").mkdir(parents=True, exist_ok=True)
    (run_dir / ".pipeline" / "arch_contract.json").write_text(
        json.dumps(contract), encoding="utf-8")

    errors = validate(_gap_spec_dict(), run_dir, ROOT)
    assert not [e for e in errors if "class(es)" in e], errors
    assert not [e for e in errors if "missing top-level" in e], errors


def test_init_finalizer_handles_generic_plan_placeholders(tmp_path):
    """The SRL 2026-07-15 stage-2d halt, closed: the generic plan's
    `<training_functions>` placeholder is producer-defined — the finalizer
    re-exports what training.py actually defines and accepts a flexible
    class count, instead of demanding a literal `<training_functions>`."""
    from scripts.finalize_package_init import finalize

    run_dir = tmp_path / "run"
    _install_run_pack(run_dir)
    spec_path = run_dir / ".pipeline" / "method_spec.json"
    spec_path.write_text(json.dumps(_gap_spec_dict()), encoding="utf-8")
    method = run_dir / "method"
    method.mkdir(parents=True)
    (method / "model.py").write_text(
        "class ValueNetwork:\n    pass\n\n"
        "class TargetNetwork:\n    pass\n",
        encoding="utf-8",
    )
    (method / "training.py").write_text(
        "def run_training_episodes(env, value_network, seed):\n"
        "    return value_network\n",
        encoding="utf-8",
    )
    (method / "method.py").write_text(
        "def train_policy(env, value_network, seed, *, num_episodes=3000):\n"
        "    return None\n",
        encoding="utf-8",
    )
    (method / "data.py").write_text(
        "def load_data(path=None, *, seed=0):\n    return {}\n",
        encoding="utf-8",
    )

    rc = finalize(spec_path, run_dir)
    assert rc == 0, f"finalize exited {rc}"
    init_text = (method / "__init__.py").read_text(encoding="utf-8")
    assert "run_training_episodes" in init_text
    assert "ValueNetwork" in init_text and "TargetNetwork" in init_text
    assert "train_policy" in init_text
    assert "<" not in init_text
