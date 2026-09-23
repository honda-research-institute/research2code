"""Paradigm-declared parameter derivation (plan item 9, 2026-07-21).

Root cause pinned by these tests: a taxonomy node / provisional pack can
declare stage_2x_params review checks that require specific params.json
entries, but the deterministic deriver had no machine-readable view of
them, so the 2.x reviewer halted runs on entries no producer could emit.
Two live cases shape the surface (the two-concrete-cases rule):

- DomIndOnto (knowledge_base_population): required pipeline configuration
  paths (ontology_path, mapping_rules_path, dictionary_paths) never
  entered params.json — they are fixed POSITIONAL signature args, and the
  extras pass only reads kwarg defaults. Plus a meaningless hidden_dim on
  a paradigm with no neural model.
- fedavg (federated_learning): the pack requires the derived statistic
  u = nE/(KB) documented; the deriver had no notion of derived
  statistics.

The node's `params_derivation` block is the machine-readable half of the
review focus; `derive_params._apply_params_derivation` consumes it
generically (kinds: config_path / derived_statistic / suppress).
"""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from scripts import taxonomy
from scripts.derive_params import (
    _apply_params_derivation,
    _resolve_derivation_input,
    _safe_eval_formula,
    derive,
)
from scripts.validate_paradigm_proposal import _check_params_derivation
from schemas.params import Params

ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# Fixtures — paradigm-neutral shapes modeled on the two live cases.
# ---------------------------------------------------------------------------


def _kbp_shaped_pack() -> dict:
    """A top-level provisional pack shaped like DomIndOnto's KBP pack."""
    return {
        "schema_version": "1.0",
        "status": "provisional",
        "legacy_paradigm": "widget_pipeline",
        "extends": None,
        "taxonomy_id": "PROVISIONAL/widget_pipeline",
        "fingerprint": {"what_it_is": "A non-ML extraction pipeline.",
                        "not_this": []},
        "scaffold_hints": {
            "interface_hint": "populate_widgets(documents, ontology_path, "
                              "mapping_rules, seed=None) -> PopulatedKB",
        },
        "semantic_checks": [],
        "smoke_bugs": [],
        "params_derivation": {
            "ontology_path": {
                "kind": "config_path",
                "reasoning": "Path to the domain ontology the pipeline loads.",
            },
            "mapping_rules_path": {
                "kind": "config_path",
                "reasoning": "External mapping-rules configuration file.",
                "paper_section": "Section 3.2",
            },
            "hidden_dim": {
                "kind": "suppress",
                "reason": "No neural model in this paradigm; hidden_dim is "
                          "meaningless.",
            },
        },
    }


def _fl_shaped_derivation() -> dict:
    """A declaration shaped like the fedavg pack's u = nE/(KB) requirement."""
    return {
        "u_expected_updates": {
            "kind": "derived_statistic",
            "formula": "n * E / (K * B)",
            "inputs": {
                "E": "params.E",
                "B": "params.B",
                "n": "spec.critical_requirements.data_setup.dataset_size",
                "K": "spec.critical_requirements.data_setup.num_clients",
            },
            "paper_section": "Section 2 (eq-expected-updates)",
        },
    }


def _install_pack(tmp_path: Path, pack: dict) -> Path:
    packs_dir = tmp_path / "provisional_packs"
    install_dir = packs_dir / "20260721-widget"
    install_dir.mkdir(parents=True)
    (install_dir / "pack.yaml").write_text(yaml.safe_dump(pack),
                                           encoding="utf-8")
    return packs_dir


def _spec_for(paradigm_id: str) -> dict:
    return {
        "comparison": {
            "classification": {"id": paradigm_id},
            "pluggable_component": {
                "name": "populate_widgets",
                "signature": "populate_widgets(documents, ontology_path, "
                             "mapping_rules, seed=None) -> PopulatedKB",
            },
        },
        "critical_requirements": {},
    }


# ---------------------------------------------------------------------------
# Known-bad case 1 (DomIndOnto shape): config paths + suppression.
# ---------------------------------------------------------------------------


def test_declared_config_paths_are_emitted_and_hidden_dim_suppressed(tmp_path):
    packs_dir = _install_pack(tmp_path, _kbp_shaped_pack())
    tax = taxonomy.load_taxonomy(ROOT, provisional_packs_dir=packs_dir)

    params: dict = {
        # What the pre-change deriver actually produced for DomIndOnto:
        # a taxonomy-convention model param on a paradigm with no model.
        "hidden_dim": {"value": 256, "source": "system_inferred",
                       "reasoning": "taxonomy-typical default width."},
    }
    _apply_params_derivation(params, _spec_for("widget_pipeline"),
                             "widget_pipeline", tax)

    assert "hidden_dim" not in params
    for name in ("ontology_path", "mapping_rules_path"):
        entry = params[name]
        assert entry["source"] == "system_inferred"
        assert "Pipeline configuration" in entry["reasoning"]
        assert entry["used_in_notebook"] is False
        assert entry["unused_reason"]
    assert params["mapping_rules_path"]["paper_section"] == "Section 3.2"
    # Every emitted entry passes the params schema.
    Params.model_validate({"schema_version": "1.1.0", "params": params})


def test_config_path_with_demo_value_is_notebook_usable(monkeypatch):
    monkeypatch.setattr(taxonomy, "load_params_derivation", lambda *_: {
        "rules_path": {"kind": "config_path",
                       "reasoning": "Rules file the demo ships.",
                       "demo_value": "method/example_data/rules.yaml"},
    })
    params: dict = {}
    _apply_params_derivation(params, {}, "widget_pipeline", None)
    entry = params["rules_path"]
    assert entry["value"] == "method/example_data/rules.yaml"
    assert entry.get("used_in_notebook", True) is True
    assert "unused_reason" not in entry


def test_existing_entries_are_never_overwritten(monkeypatch):
    monkeypatch.setattr(taxonomy, "load_params_derivation", lambda *_: {
        "ontology_path": {"kind": "config_path", "reasoning": "Declared."},
    })
    params = {"ontology_path": {"value": "real.owl", "source": "paper",
                                "note": "Paper names the ontology file."}}
    _apply_params_derivation(params, {}, "widget_pipeline", None)
    assert params["ontology_path"]["value"] == "real.owl"
    assert params["ontology_path"]["source"] == "paper"


# ---------------------------------------------------------------------------
# Known-bad case 2 (fedavg shape): derived statistics.
# ---------------------------------------------------------------------------


def test_derived_statistic_computes_from_params_and_spec(monkeypatch):
    monkeypatch.setattr(taxonomy, "load_params_derivation",
                        lambda *_: _fl_shaped_derivation())
    params = {
        "E": {"value": 5, "source": "spec_default", "reasoning": "sig default."},
        "B": {"value": 10, "source": "spec_default", "reasoning": "sig default."},
    }
    spec = {"critical_requirements": {"data_setup": {
        "dataset_size": 60000, "num_clients": 100}}}
    _apply_params_derivation(params, spec, "fed_stat", None)

    entry = params["u_expected_updates"]
    # u = 60000 * 5 / (100 * 10) = 300, collapsed to int.
    assert entry["value"] == 300
    assert entry["source"] == "system_inferred"
    assert "n * E / (K * B)" in entry["reasoning"]
    assert entry["paper_section"] == "Section 2 (eq-expected-updates)"
    assert entry["used_in_notebook"] is False
    Params.model_validate({"schema_version": "1.1.0", "params": params})


def test_derived_statistic_with_unresolvable_inputs_documents_honestly(
        monkeypatch):
    """The fedavg reality: the spec does NOT structurally carry n and K.
    The entry must still exist — documented formula, named gaps, no guess."""
    monkeypatch.setattr(taxonomy, "load_params_derivation",
                        lambda *_: _fl_shaped_derivation())
    params = {
        "E": {"value": 5, "source": "spec_default", "reasoning": "sig default."},
        "B": {"value": 10, "source": "spec_default", "reasoning": "sig default."},
    }
    _apply_params_derivation(params, {"critical_requirements": {}},
                             "fed_stat", None)

    entry = params["u_expected_updates"]
    assert entry["value"] is None
    assert "could not be computed" in entry["reasoning"]
    assert "dataset_size" in entry["reasoning"]  # names the missing input
    assert "num_clients" in entry["reasoning"]
    Params.model_validate({"schema_version": "1.1.0", "params": params})


# ---------------------------------------------------------------------------
# Adjacent-good: paradigms that declare nothing are untouched.
# ---------------------------------------------------------------------------


def test_no_declaration_is_a_no_op():
    params = {"batch_size": {"value": 8, "source": "paper", "note": "stated."}}
    before = json.dumps(params, sort_keys=True)
    _apply_params_derivation(params, {}, "active_learning", None)
    assert json.dumps(params, sort_keys=True) == before


def test_committed_paradigms_declare_no_params_derivation_today():
    tax = taxonomy.load_taxonomy(ROOT)
    for pid in ("active_learning", "knowledge_distillation",
                "motion_planning"):
        assert taxonomy.load_params_derivation(pid, tax) == {}


def test_loader_resolves_overlay_and_filters_malformed(tmp_path):
    pack = _kbp_shaped_pack()
    pack["params_derivation"]["broken"] = {"no_kind_here": True}
    packs_dir = _install_pack(tmp_path, pack)
    tax = taxonomy.load_taxonomy(ROOT, provisional_packs_dir=packs_dir)

    declared = taxonomy.load_params_derivation("widget_pipeline", tax)
    assert set(declared) == {"ontology_path", "mapping_rules_path",
                             "hidden_dim"}
    # Without the overlay the paradigm is unserved: nothing leaks.
    assert taxonomy.load_params_derivation(
        "widget_pipeline", taxonomy.load_taxonomy(ROOT)) == {}


def test_end_to_end_derive_with_overlay_pack(tmp_path):
    """The full DomIndOnto reproduction through derive(): a run-local
    overlay pack's declarations reach params.json via run_dir alone."""
    run_dir = tmp_path / "run"
    pipeline = run_dir / ".pipeline"
    pipeline.mkdir(parents=True)
    packs_dir = pipeline / "provisional_packs"
    install_dir = packs_dir / "20260721-widget"
    install_dir.mkdir(parents=True)
    (install_dir / "pack.yaml").write_text(
        yaml.safe_dump(_kbp_shaped_pack()), encoding="utf-8")

    output = derive(_spec_for("widget_pipeline"), ROOT, run_dir=run_dir)
    params = output["params"]
    assert "ontology_path" in params
    assert "mapping_rules_path" in params
    assert "hidden_dim" not in params
    Params.model_validate(output)


# ---------------------------------------------------------------------------
# Formula machinery.
# ---------------------------------------------------------------------------


def test_safe_eval_formula_arithmetic_and_integral_collapse():
    assert _safe_eval_formula("n * E / (K * B)",
                              {"n": 60000, "E": 5, "K": 100, "B": 10}) == 300
    assert _safe_eval_formula("a / b", {"a": 1, "b": 3}) == 1 / 3
    assert _safe_eval_formula("-a + 2 ** 3", {"a": 1}) == 7


def test_safe_eval_formula_rejects_code_and_bad_math():
    assert _safe_eval_formula("__import__('os')", {}) is None
    assert _safe_eval_formula("a.b", {"a": 1}) is None
    assert _safe_eval_formula("f(1)", {"f": 1}) is None
    assert _safe_eval_formula("unknown + 1", {}) is None
    assert _safe_eval_formula("1 / 0", {}) is None
    assert _safe_eval_formula("", {}) is None


def test_resolve_derivation_input_forms():
    params = {"E": {"value": 5, "source": "paper", "note": "x"},
              "flag": {"value": True, "source": "paper", "note": "x"}}
    spec = {"critical_requirements": {"data_setup": {"num_clients": 100,
                                                     "name": "MNIST"}}}
    assert _resolve_derivation_input("params.E", params, spec) == 5
    assert _resolve_derivation_input(
        "spec.critical_requirements.data_setup.num_clients", params, spec) == 100
    # Booleans, strings, missing targets, and unknown forms are all None.
    assert _resolve_derivation_input("params.flag", params, spec) is None
    assert _resolve_derivation_input(
        "spec.critical_requirements.data_setup.name", params, spec) is None
    assert _resolve_derivation_input("params.absent", params, spec) is None
    assert _resolve_derivation_input("env.HOME", params, spec) is None


# ---------------------------------------------------------------------------
# Authoring-time lints (the proposal validator is the gap-pack entry path).
# ---------------------------------------------------------------------------


def test_proposal_lint_accepts_well_formed_block():
    errors: list[str] = []
    warnings: list[str] = []
    pack = _kbp_shaped_pack()
    pack["params_derivation"].update(_fl_shaped_derivation())
    _check_params_derivation(pack, errors, warnings)
    assert errors == []
    assert warnings == []


def test_proposal_lint_rejects_malformed_entries():
    errors: list[str] = []
    _check_params_derivation({"params_derivation": {
        "a": {"kind": "config_path"},                       # no reasoning
        "b": {"kind": "derived_statistic", "inputs": {}},   # no formula/inputs
        "c": {"kind": "derived_statistic", "formula": "x",
              "inputs": {"x": "env.HOME"}},                 # bad ref
        "d": {"kind": "suppress"},                          # no reason
        "e": {"kind": "mystery"},                           # unknown kind
        "not an identifier": {"kind": "suppress", "reason": "r"},
    }}, errors, [])
    joined = "\n".join(errors)
    assert "must carry `reasoning`" in joined
    assert "must carry `formula`" in joined
    assert "non-empty `inputs`" in joined
    assert "must reference `params.<name>`" in joined
    assert "must carry `reason`" in joined
    assert "'mystery' is not one of" in joined
    assert "python-identifier param name" in joined


def test_proposal_lint_rejects_2x_check_without_declaration():
    """The exact producer-blind combination that halted both 2026-07-21
    runs: a stage_2x_params check with no params_derivation block.

    Was a non-blocking warning until 2026-08-05. It then halted a third
    live run (pdfgnn, similarity_cutoff), so R2C-055 promoted it to an
    authoring error to keep the fix in the proposal retry loop."""
    errors: list[str] = []
    _check_params_derivation({"semantic_checks": [
        {"id": "X-params", "stage": "stage_2x_params", "check": "..."}]},
        errors, [])
    assert len(errors) == 1
    assert "params_derivation" in errors[0]


# ---------------------------------------------------------------------------
# Notebook provenance table: the "Where in paper" column (researcher feedback 2026-07,
# Japan-feedback plan §2 — per-parameter paper location in the notebook).
# ---------------------------------------------------------------------------


def test_params_table_renders_paper_location_column():
    from scripts.render_notebook import _render_params_table_cell

    table = _render_params_table_cell({"params": {
        "C": {"value": 0.1, "source": "paper",
              "paper_section": "Section 3 (Increasing parallelism)",
              "note": "Paper states 'we fix C=0.1'."},
        "B": {"value": 10, "source": "spec_default",
              "reasoning": "signature default."},
    }})
    header, _, row_c, row_b = table.splitlines()
    assert "| Where in paper |" in header
    assert "| Section 3 (Increasing parallelism) |" in row_c
    assert "| — |" in row_b  # no recorded location renders an honest dash


# ---------------------------------------------------------------------------
# Param glossary carriage (param-glossary-design.md, 2026-07-21).
# ---------------------------------------------------------------------------


def test_glossary_quotes_stamp_matching_params_by_name_and_alias():
    from scripts.derive_params import _apply_param_glossary

    spec = {"critical_requirements": {"param_glossary": [
        {"name": "E", "aliases": ["local_epochs"],
         "meaning_quote": "E, the number of training passes each client "
                          "makes over its local dataset on each round",
         "paper_section": "Section 2"},
        {"name": "Z", "aliases": [], "meaning_quote": "unused",
         "paper_section": "S9"},
    ]}}
    params = {
        "E": {"value": 5, "source": "spec_default", "reasoning": "sig."},
        "local_epochs": {"value": 5, "source": "spec_default",
                         "reasoning": "sig.", "paper_section": "Table 2"},
        "B": {"value": 10, "source": "spec_default", "reasoning": "sig."},
    }
    _apply_param_glossary(params, spec)
    assert params["E"]["paper_says"].startswith("E, the number")
    assert params["E"]["paper_section"] == "Section 2"
    # Alias hit; its OWN paper_section is never overwritten.
    assert params["local_epochs"]["paper_says"].startswith("E, the number")
    assert params["local_epochs"]["paper_section"] == "Table 2"
    assert "paper_says" not in params["B"]  # no fuzzy matching
    Params.model_validate({"schema_version": "1.1.0", "params": params})


def test_spec_validator_meaning_quote_floor(tmp_path):
    import subprocess, sys
    from tests.test_method_spec_schema import _minimal_valid_spec

    spec = _minimal_valid_spec(paradigm_id="motion_planning")
    spec["critical_requirements"]["param_glossary"] = [
        {"name": "E", "aliases": [],
         "meaning_quote": "the number of local epochs per round",
         "paper_section": "Section 2"}]
    spec_path = tmp_path / "method_spec.json"
    spec_path.write_text(json.dumps(spec), encoding="utf-8")
    paper = tmp_path / "paper.md"

    paper.write_text("We define E as the number of local\nepochs per "
                     "round of communication.", encoding="utf-8")
    ok = subprocess.run([sys.executable, "scripts/validate_method_spec.py",
                         str(spec_path), "--paper-md", str(paper)],
                        capture_output=True, text=True)
    assert ok.returncode == 0, ok.stderr

    paper.write_text("The paper never says that sentence.",
                     encoding="utf-8")
    bad = subprocess.run([sys.executable, "scripts/validate_method_spec.py",
                          str(spec_path), "--paper-md", str(paper)],
                         capture_output=True, text=True)
    assert bad.returncode == 1
    assert "meaning-quote floor failed" in bad.stderr
    assert "param_glossary['E']" in bad.stderr


def test_glossary_cap_enforced_by_schema(tmp_path):
    import pytest
    from pydantic import ValidationError
    from tests.test_method_spec_schema import _minimal_valid_spec
    from schemas.method_spec import MethodSpec

    spec = _minimal_valid_spec(paradigm_id="motion_planning")
    spec["critical_requirements"]["param_glossary"] = [
        {"name": f"p{i}", "aliases": [], "meaning_quote": "q",
         "paper_section": "s"} for i in range(13)]
    with pytest.raises(ValidationError):
        MethodSpec.model_validate(spec)


def test_notebook_notes_prefer_paper_says():
    from scripts.render_notebook import _render_params_table_cell

    table = _render_params_table_cell({"params": {
        "E": {"value": 5, "source": "spec_default", "reasoning": "sig.",
              "paper_says": "the number of local epochs per round"},
    }})
    assert "the number of local epochs per round" in table


# ---------------------------------------------------------------------------
# Known-bad case 3 (SRL 2026-07-21 stage-2x halt): declared parameter
# quality. F002 — a declared config_path consulted no paper truth at all,
# so episodes shipped null while the run's own methodology contract
# recorded the paper's 3000. F003 — declared names an earlier pass emitted
# as bare signature defaults kept the boilerplate "no paper statement was
# found" reasoning the declaration itself contradicts.
# ---------------------------------------------------------------------------


SRL_PAPER = ("Offline training (Algorithm 1) took approximately nine hours "
             "to complete 3,000 episodes on the four-agent network.")


def _srl_shaped_spec() -> dict:
    """The SRL evidence shape: paper truth lives in the methodology
    contract's approved approximation, not in a structured field."""
    return {
        "comparison": {"classification": {"id": "widget_rl"}},
        "critical_requirements": {},
        "methodology_replication_contract": {"elements": [{
            "name": "training-loop",
            "paper_evidence": "The training loop runs episodic V-learning.",
            "acceptable_approximations": [
                "Number of training episodes can be reduced from 3000 to "
                "200-500 for demo scale"],
        }]},
    }


def test_config_path_carries_contract_paper_value(monkeypatch):
    monkeypatch.setattr(taxonomy, "load_params_derivation", lambda *_: {
        "episodes": {"kind": "config_path",
                     "reasoning": "Total number of training episodes.",
                     "paper_section": "Algorithm 1, Section IV-A"},
    })
    params: dict = {}
    _apply_params_derivation(params, _srl_shaped_spec(), "widget_rl", None,
                             paper_text=SRL_PAPER)
    entry = params["episodes"]
    assert entry["value"] == 3000
    assert entry["source"] == "paper"
    assert entry["paper_section"] == "Algorithm 1, Section IV-A"
    assert "methodology contract" in entry["note"]
    # Still a runtime-supplied configuration path: usage flags unchanged.
    assert entry["used_in_notebook"] is False
    assert entry["unused_reason"]
    Params.model_validate({"schema_version": "1.1.0", "params": params})


def test_config_path_unverifiable_paper_value_is_kept_for_audit(monkeypatch):
    # The contract records 3000, but the paper text does not render the
    # value (the comma-fold covers "3,000"; this paper genuinely lacks it):
    # never stamp source=paper on a value the US-3 arm would reject — keep
    # the honest null with the paper value preserved for audit.
    monkeypatch.setattr(taxonomy, "load_params_derivation", lambda *_: {
        "episodes": {"kind": "config_path",
                     "reasoning": "Total number of training episodes.",
                     "paper_section": "Algorithm 1, Section IV-A"},
    })
    params: dict = {}
    _apply_params_derivation(params, _srl_shaped_spec(), "widget_rl", None,
                             paper_text="The paper text renders no number.")
    entry = params["episodes"]
    assert entry["value"] is None
    assert entry["source"] == "system_inferred"
    assert entry["paper_value"] == 3000
    assert "preserved in paper_value for audit" in entry["reasoning"]
    Params.model_validate({"schema_version": "1.1.0", "params": params})


def test_config_path_demo_value_keeps_paper_value_for_audit(monkeypatch):
    monkeypatch.setattr(taxonomy, "load_params_derivation", lambda *_: {
        "episodes": {"kind": "config_path",
                     "reasoning": "Total number of training episodes.",
                     "demo_value": 200},
    })
    params: dict = {}
    _apply_params_derivation(params, _srl_shaped_spec(), "widget_rl", None,
                             paper_text=SRL_PAPER)
    entry = params["episodes"]
    assert entry["value"] == 200
    assert entry["source"] == "system_inferred"
    assert entry["paper_value"] == 3000
    Params.model_validate({"schema_version": "1.1.0", "params": params})


def test_declared_signature_default_gains_declared_provenance(monkeypatch):
    # F003 (SRL gamma/q_norm): the declaration says the paper defines this
    # parameter, so the earlier pass's spec_default boilerplate ("no paper
    # statement was found") is re-labeled with the declaration's own
    # provenance. Value untouched.
    monkeypatch.setattr(taxonomy, "load_params_derivation", lambda *_: {
        "gamma": {"kind": "config_path",
                  "reasoning": "Discount factor in [0, 1) used in the "
                               "value function.",
                  "paper_section": "Section II-A, Equation (5)"},
    })
    params = {"gamma": {
        "value": 0.9, "source": "spec_default",
        "reasoning": "Value from the pluggable_component signature default "
                     "(gamma=0.9). No structured spec source ... was found.",
    }}
    _apply_params_derivation(params, {}, "widget_rl", None)
    entry = params["gamma"]
    assert entry["value"] == 0.9
    assert entry["source"] == "system_inferred"
    assert entry["reasoning"].startswith("Discount factor in [0, 1)")
    assert "signature default" in entry["reasoning"]
    assert entry["paper_section"] == "Section II-A, Equation (5)"
    Params.model_validate({"schema_version": "1.1.0", "params": params})


def test_declared_paper_sourced_entry_is_untouched(monkeypatch):
    monkeypatch.setattr(taxonomy, "load_params_derivation", lambda *_: {
        "gamma": {"kind": "config_path", "reasoning": "Declared."},
    })
    params = {"gamma": {"value": 0.95, "source": "paper",
                        "note": "Paper sets gamma directly."}}
    before = json.dumps(params, sort_keys=True)
    _apply_params_derivation(params, {}, "widget_rl", None)
    assert json.dumps(params, sort_keys=True) == before


def test_glossary_hit_relabels_bare_signature_default():
    # F003 (SRL epsilon_f): a glossary quote IS a paper statement of the
    # parameter, so the spec_default boilerplate is re-labeled to an honest
    # system_inferred; entries with real provenance are untouched.
    from scripts.derive_params import _apply_param_glossary

    spec = {"critical_requirements": {"param_glossary": [
        {"name": "epsilon_f", "aliases": [],
         "meaning_quote": "with prob ϵf, mirror every traj in the x-axis",
         "paper_section": "Algorithm 1, line 8"},
        {"name": "d_safe", "aliases": [],
         "meaning_quote": "the safety margin", "paper_section": "Eq. 4"},
    ]}}
    params = {
        "epsilon_f": {"value": 0.01, "source": "spec_default",
                      "reasoning": "Value from the pluggable_component "
                                   "signature default (epsilon_f=0.01)."},
        "d_safe": {"value": 0.375, "source": "paper",
                   "paper_section": "Eq. 4", "note": "Paper-stated."},
    }
    _apply_param_glossary(params, spec)
    entry = params["epsilon_f"]
    assert entry["source"] == "system_inferred"
    assert "paper defines this parameter" in entry["reasoning"]
    assert entry["paper_says"].startswith("with prob")
    assert params["d_safe"]["source"] == "paper"  # real provenance untouched
    Params.model_validate({"schema_version": "1.1.0", "params": params})
