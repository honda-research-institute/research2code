"""Provenance is evidence-driven in BOTH directions (the A-001 mirror pair).

A-001 (2026-05-26): dropout_rate OVER-claimed as paper-sourced when it was a
field-guide convention. The 2026-06-10 GBALD stage-2x halt was the mirror:
UNDER-claimed as system_inferred when the spec's specific_features carried
explicit paper evidence ("Dropout rate 0.5", Section 7.4) — the reviewer
correctly refused the dishonest-conservative label and the run halted because
the fix needed a producer change. These tests pin the producer consulting the
spec's structured evidence first.
"""

from __future__ import annotations

from pathlib import Path

from scripts.derive_params import (_add_kd_training_params, _add_model_params,
                                   _add_paradigm_extras,
                                   _add_training_params,
                                   _mark_runtime_unused_model_params,
                                   _parse_learning_rate,
                                   _specific_feature_evidence,
                                   _training_threshold_evidence)
from scripts.validate_params_provenance import (check_params,
                                                paper_claim_findable)


def _spec(features):
    return {
        "critical_requirements": {
            "model": {"specific_features": features},
            "data_setup": {},
        },
    }


PAPER_DROPOUT = {
    "feature": "Dropout rate 0.3",
    "why_essential": "controls MC variance",
    "severity": "important",
    "paper_section": "Section 5.1",
}
NON_NUMERIC_DROPOUT = {
    "feature": "Dropout layers active at inference for MC dropout",
    "why_essential": "MC sampling needs dropout at inference",
    "severity": "essential",
    "paper_section": "Section 4.3",
}


def test_paper_stated_dropout_gets_paper_provenance_and_paper_value():
    params: dict = {}
    _add_model_params(params, _spec([NON_NUMERIC_DROPOUT, PAPER_DROPOUT]),
                      "active_learning/bayesian")
    entry = params["dropout_rate"]
    # The paper's value (0.3), not the convention default (0.5).
    assert entry["value"] == 0.3
    assert entry["source"] == "paper"
    assert entry["paper_section"] == "Section 5.1"


def test_without_paper_evidence_the_convention_label_stays_honest():
    params: dict = {}
    # Only the non-numeric feature: "active at inference" is a behavior
    # requirement, not a stated value — must NOT be treated as paper evidence
    # (its '4.3' lives in paper_section, not the feature text... and even the
    # text has no number).
    _add_model_params(params, _spec([NON_NUMERIC_DROPOUT]),
                      "active_learning/bayesian")
    entry = params["dropout_rate"]
    assert entry["value"] == 0.5
    assert entry["source"] == "system_inferred"
    assert "paper_section" not in entry


def test_evidence_helper_requires_number_and_section():
    assert _specific_feature_evidence(
        _spec([PAPER_DROPOUT]), "dropout") == (0.3, "Section 5.1",
                                               "Dropout rate 0.3")
    assert _specific_feature_evidence(
        _spec([NON_NUMERIC_DROPOUT]), "dropout") is None
    no_section = {**PAPER_DROPOUT, "paper_section": ""}
    assert _specific_feature_evidence(_spec([no_section]), "dropout") is None


def test_non_bayesian_paradigms_get_no_dropout_param():
    params: dict = {}
    _add_model_params(params, _spec([PAPER_DROPOUT]),
                      "active_learning/batch_acquisition")
    assert "dropout_rate" not in params


# ---------------------------------------------------------------------------
# train_until_accuracy: the over-claim direction (fabricated-99% class).
# Caught live 2026-06-10: the fresh GBALD run's quote-match probe (US-3)
# failed on a "Paper: train until training accuracy exceeds 99%." note the
# producer stamped unconditionally — the same fabricated quote the june9
# manual audit found, while the stage-2x LLM reviewer passed it both times.
# ---------------------------------------------------------------------------


def _training_spec(protocol: str, section: str = "Section 5 (Training)"):
    return {
        "critical_requirements": {
            "model": {},
            "data_setup": {},
            "training": {
                "optimizer": "Adam",
                "learning_rate": "Not explicitly stated.",
                "protocol": protocol,
                "paper_section": section,
            },
        },
    }


def test_stated_threshold_gets_paper_provenance_and_paper_value():
    params: dict = {}
    _add_training_params(params, _training_spec(
        "Retrain from scratch; train until training accuracy exceeds 98%."))
    entry = params["train_until_accuracy"]
    assert entry["source"] == "paper"
    assert entry["value"] == 0.98  # value follows the paper, not the 0.99 convention
    assert entry["paper_section"] == "Section 5 (Training)"


def test_unstated_threshold_is_a_convention_not_a_paper_claim():
    params: dict = {}
    # GBALD's real spec shape: protocol says only "retrain from scratch".
    _add_training_params(params, _training_spec("retrain from scratch"))
    entry = params["train_until_accuracy"]
    assert entry["source"] == "system_inferred"
    assert entry["value"] == 0.99
    assert "convention" in entry["reasoning"]
    # max_epochs reasoning must not fabricate a paper protocol either.
    assert "Paper trains" not in params["max_epochs"]["reasoning"]


def test_threshold_evidence_helper_parses_percent_and_fraction():
    spec = _training_spec(
        "Each round trains until training accuracy reaches 0.97.")
    assert _training_threshold_evidence(spec) == (
        0.97, "Section 5 (Training)",
        "Each round trains until training accuracy reaches 0.97.")
    assert _training_threshold_evidence(
        _training_spec("retrain from scratch")) is None
    no_section = _training_spec(
        "train until training accuracy exceeds 99%.", section="")
    assert _training_threshold_evidence(no_section) is None


# ---------------------------------------------------------------------------
# BADGE stage-2x halt (2026-06-10, the third provenance direction case):
# the paper states the threshold but the ANALYZER never extracted it into
# the spec, so spec-only evidence under-claimed. The paper text is now the
# fallback evidence source — it is the same ground truth the reviewer
# checks. Second finding, same halt: hidden_dim's old template hardcoded
# "256"/"1024" string checks and under-claimed BADGE's directly-stated MLP
# width.
# ---------------------------------------------------------------------------

BADGE_PAPER_TEXT = """\
## 4 EXPERIMENTS

We now describe the experiments. We fit
models using cross-entropy loss and the Adam variant of SGD until training
accuracy exceeds 99%. We use a learning rate of 0.001 for image data.
"""

BADGE_ARCH = (
    "Neural network classifier with a softmax output layer. The paper uses "
    "a 2-layer MLP with ReLU (hidden dim 256 for image data, 1024 for "
    "OpenML tabular data), an 18-layer ResNet, or an 11-layer VGG.")
MAY_GBALD_ARCH = (
    "MLP with 1024 for tabular data and 256 for image data with "
    "ResNet-18 / VGG-11.")
FRESH_GBALD_ARCH = (
    "MLP with three blocks of [convolution, dropout, maxpooling, relu] "
    "using 32, 64, 128 3x3 convolution filters.")


def _model_spec(arch, benchmark="MNIST"):
    return {
        "critical_requirements": {
            "model": {"architecture": arch,
                      "paper_section": "Section 4 (Experiments)",
                      "specific_features": []},
            "data_setup": {"benchmark_name": benchmark},
        },
    }


def test_paper_text_fallback_recovers_unextracted_threshold():
    spec = _training_spec("retrain from scratch")
    value, section, quote = _training_threshold_evidence(
        spec, paper_text=BADGE_PAPER_TEXT)
    assert value == 0.99
    assert section == "Section 4 EXPERIMENTS"
    assert "exceeds 99%" in quote


def test_paper_text_without_header_stays_unclaimed():
    headerless = "We train until training accuracy exceeds 99%."
    assert _training_threshold_evidence(
        _training_spec("retrain from scratch"), paper_text=headerless) is None


def test_real_gbald_paper_text_never_matches_the_threshold_pattern():
    # The fabricated-99% guard: GBALD's only 0.99s are figure-caption
    # accuracy readings; the fallback must not turn them into a protocol.
    paper = (Path(__file__).parent / "fixtures" / "evidence"
             / "june9-gbald-run" / ".pipeline" / "paper.md")
    if not paper.is_file():
        paper = (Path(__file__).parent / "fixtures" / "evidence"
                 / "june9-gbald-run" / "pipeline" / "paper.md")
    import pytest
    if not paper.is_file():
        pytest.skip("june9 paper.md not in evidence fixture")
    assert _training_threshold_evidence(
        _training_spec("retrain from scratch"),
        paper_text=paper.read_text(encoding="utf-8")) is None


def test_hidden_dim_paper_when_mlp_owns_the_width():
    params: dict = {}
    _add_model_params(params, _model_spec(BADGE_ARCH), "active_learning")
    entry = params["hidden_dim"]
    assert entry["source"] == "paper"
    assert entry["value"] == 256
    assert entry["paper_section"] == "Section 4 (Experiments)"


def test_hidden_dim_inferred_when_cnn_owns_the_width():
    params: dict = {}
    _add_model_params(params, _model_spec(MAY_GBALD_ARCH), "active_learning")
    entry = params["hidden_dim"]
    assert entry["source"] == "system_inferred"
    assert entry["value"] == 256
    assert "CNN" in entry["reasoning"]


def test_hidden_dim_convention_when_no_width_stated():
    params: dict = {}
    _add_model_params(params, _model_spec(FRESH_GBALD_ARCH),
                      "active_learning")
    entry = params["hidden_dim"]
    assert entry["source"] == "system_inferred"
    assert "convention" in entry["reasoning"]


def test_hidden_dim_tabular_paper_width():
    params: dict = {}
    _add_model_params(params, _model_spec(BADGE_ARCH, benchmark="OpenML"),
                      "active_learning")
    entry = params["hidden_dim"]
    assert entry["source"] == "paper"
    assert entry["value"] == 1024


def test_hidden_dim_marked_unused_when_runtime_cannot_consume_it(tmp_path):
    run = tmp_path / "run"
    method = run / "method"
    method.mkdir(parents=True)
    (method / "model.py").write_text(
        "import torch.nn as nn\n"
        "class Net(nn.Module):\n"
        "    def __init__(self, input_dim, n_classes):\n"
        "        super().__init__()\n"
        "        self.fc = nn.Linear(input_dim, n_classes)\n",
        encoding="utf-8",
    )
    (method / "training.py").write_text(
        "from .model import Net\n"
        "def build_model(input_dim, n_classes):\n"
        "    return Net(input_dim, n_classes)\n",
        encoding="utf-8",
    )
    params = {"hidden_dim": {"value": 11, "source": "system_inferred",
                             "reasoning": "taxonomy convention"}}
    _mark_runtime_unused_model_params(params, run)
    assert params["hidden_dim"]["used_in_notebook"] is False
    assert "does not expose or read" in params["hidden_dim"]["unused_reason"]


def test_hidden_dim_stays_live_when_runtime_accepts_it(tmp_path):
    run = tmp_path / "run"
    method = run / "method"
    method.mkdir(parents=True)
    (method / "model.py").write_text(
        "import torch.nn as nn\n"
        "class Net(nn.Module):\n"
        "    def __init__(self, input_dim, n_classes, hidden_dim=256):\n"
        "        super().__init__()\n"
        "        self.fc1 = nn.Linear(input_dim, hidden_dim)\n"
        "        self.fc2 = nn.Linear(hidden_dim, n_classes)\n",
        encoding="utf-8",
    )
    (method / "training.py").write_text(
        "from .model import Net\n"
        "def build_model(input_dim, n_classes, hidden_dim=256):\n"
        "    return Net(input_dim, n_classes, hidden_dim=hidden_dim)\n",
        encoding="utf-8",
    )
    params = {"hidden_dim": {"value": 256, "source": "system_inferred",
                             "reasoning": "taxonomy convention"}}
    _mark_runtime_unused_model_params(params, run)
    assert "used_in_notebook" not in params["hidden_dim"]


def test_lr_note_attributes_spec_text_not_a_paper_quote():
    # The BADGE matrix-row-3 audit: the note quoted the spec's summary
    # string (with dataset parentheticals the paper never wrote) as
    # "Paper: '...'", which the quote-match probe verifies verbatim and
    # correctly failed. Spec text is attributed as spec text.
    params: dict = {}
    spec = _training_spec("retrain from scratch")
    spec["critical_requirements"]["training"]["learning_rate"] = (
        "0.001 for image data (MNIST, CIFAR-10), 0.0001 for non-image data")
    spec["critical_requirements"]["data_setup"]["benchmark_name"] = "MNIST"
    _add_training_params(params, spec)
    note = params["learning_rate"]["note"]
    assert "Spec training.learning_rate:" in note
    assert "Paper: '" not in note
    assert params["learning_rate"]["value"] == 0.001


def test_lr_last_resort_skips_citation_year_for_plausible_value():
    # The 2026-06-30 GBALD fresh-run halt: the LR free-text "follows Gal et al.
    # 2017 conventions (typically 0.001 Adam ...)" has no "<num> for <descriptor>"
    # match and is not a bare float, so it hits the last-resort number scan.
    # That scan used to take the FIRST number — the citation year 2017 — and
    # emit learning_rate=2017.0, which the US-1 plausible-range gate rejected and
    # Stage 2.x halted on. The scan now skips numbers outside the plausible LR
    # range (1e-6..1) and picks 0.001. Citation years, epoch counts, and sample
    # sizes can never be mistaken for a learning rate.
    value, note = _parse_learning_rate(
        "Paper does not specify explicit LR; follows Gal et al. 2017 "
        "conventions (typically 0.001 Adam for AL benchmarks)",
        is_image=True,
    )
    assert value == 0.001, f"citation year leaked into LR: got {value}"
    assert "first plausible" in note
    # Other citation-year phrasings and a stray epoch count likewise resolve to
    # the real LR, not the year/count.
    assert _parse_learning_rate("follows Devlin et al. 2019, typically 5e-5",
                                is_image=False)[0] == 5e-5
    assert _parse_learning_rate("trained for 200 epochs at 0.005",
                                is_image=True)[0] == 0.005


def test_lr_citation_year_resolves_to_inferred_not_paper():
    # End-to-end through _add_training_params: the recovered 0.001 is a
    # heuristic extraction from prose, so its source stays system_inferred
    # (not falsely stamped source=paper), and US-1 / US-2b now pass.
    params: dict = {}
    spec = _training_spec("retrain from scratch")
    spec["critical_requirements"]["training"]["learning_rate"] = (
        "Paper does not specify explicit LR; follows Gal et al. 2017 "
        "conventions (typically 0.001 Adam for AL benchmarks)")
    spec["critical_requirements"]["data_setup"]["benchmark_name"] = "MNIST"
    _add_training_params(params, spec)
    assert params["learning_rate"]["value"] == 0.001
    assert params["learning_rate"]["source"] == "system_inferred"


# ---------------------------------------------------------------------------
# bev-distill 2026-07-01 stage-2x halt (F001): the spec's LR field led with the
# paper's stated value ("2e-4 initial, cyclic policy"), but the parse ladder
# only reached it via the last-resort heuristic scan, so the entry shipped
# system_inferred with a reasoning that FALSELY claimed "Paper does not
# explicitly specify the learning rate" — the stage-2x reviewer correctly
# halted on the false claim. Pins: (1) a leading stated value parses clean,
# (2) a clean + paper-findable value stamps source=paper through the KD path
# (whose inline copy had drifted from AL and lacked the findability arm),
# (3) a genuinely heuristic extraction stays system_inferred with reasoning
# that says so instead of claiming the paper is silent.
# ---------------------------------------------------------------------------


def test_lr_leading_value_parses_clean():
    value, note = _parse_learning_rate("2e-4 initial, cyclic policy",
                                       is_image=True)
    assert value == 2e-4
    assert "Leading stated value" in note
    assert "using first" not in note
    # A leading number outside the plausible LR range (an epoch count) is not
    # the value — the ladder falls through to the heuristic scan.
    value, note = _parse_learning_rate("24 epochs at 0.005, cyclic",
                                       is_image=True)
    assert value == 0.005
    assert "first plausible" in note


def test_kd_lr_paper_stated_value_stamps_paper_source():
    params: dict = {}
    spec = _training_spec("fixed epochs")
    spec["critical_requirements"]["training"]["learning_rate"] = (
        "2e-4 initial, cyclic policy")
    paper = ("During the distillation phase, the batch size is set to 1 per "
             "GPU with an initial learning rate of 2e-4.")
    _add_kd_training_params(params, spec, paper_text=paper)
    entry = params["learning_rate"]
    assert entry["value"] == 2e-4
    assert entry["source"] == "paper", entry
    assert entry["paper_section"] == "Section 5 (Training)"


def test_kd_lr_clean_value_absent_from_paper_stays_inferred():
    # The findability arm the KD copy used to lack: a cleanly-parsed spec
    # value whose rendering is absent from the paper text is an inference,
    # never asserted as a paper claim.
    params: dict = {}
    spec = _training_spec("fixed epochs")
    spec["critical_requirements"]["training"]["learning_rate"] = "0.000237"
    _add_kd_training_params(params, spec,
                            paper_text="The paper never states that value.")
    assert params["learning_rate"]["source"] == "system_inferred"


def test_lr_heuristic_reasoning_never_claims_paper_silent():
    params: dict = {}
    spec = _training_spec("fixed epochs")
    spec["critical_requirements"]["training"]["learning_rate"] = (
        "follows Gal et al. 2017 conventions (typically 0.001 Adam)")
    _add_kd_training_params(params, spec,
                            paper_text="No learning rate stated anywhere.")
    entry = params["learning_rate"]
    assert entry["value"] == 0.001
    assert entry["source"] == "system_inferred"
    assert "does not explicitly specify" not in entry["reasoning"]
    assert "heuristic" in entry["reasoning"]


def test_model_default_literal_fallback_is_paradigm_neutral():
    # bev-distill F002: the literal hidden_dim fallback (paradigms whose
    # taxonomy node serves no model_defaults entry) shipped "taxonomy-typical
    # AL convention" into a KD run's params.json. The fallback must never
    # cite a convention from a paradigm the paper doesn't belong to.
    params: dict = {}
    spec = {"critical_requirements": {"model": {"architecture": ""},
                                      "data_setup": {}}}
    _add_model_params(params, spec, "knowledge_distillation/detection/cross_modal")
    entry = params["hidden_dim"]
    assert entry["source"] == "system_inferred"
    assert "AL convention" not in entry["reasoning"]
    assert "smoke-scale networks" in entry["reasoning"]


# ---------------------------------------------------------------------------
# US-3b consistency: the deriver must not stamp a "Paper states name=value"
# claim (from a heuristic text scan) that the US-3b provenance validator would
# then reject and halt on. The 2026-07-01 iDb-RRT run halted at Stage 2.x with
# a fabricated goal_bias=3 paper claim; the fix gates the textual value with
# the same predicate US-3b uses.
# ---------------------------------------------------------------------------


def test_paper_claim_findable_matches_us3b():
    paper = "The planner expands a search tree toward the goal using primitives."
    # Value not present near the name -> would be rejected as fabricated.
    assert paper_claim_findable("goal_bias", 3, paper) is False
    # No paper text -> cannot disprove, treated as findable (matches US-3b).
    assert paper_claim_findable("goal_bias", 3, None) is True
    # Non-numeric value -> US-3b only checks numeric claims.
    assert paper_claim_findable("motion_primitives", None, paper) is True
    # Name words and value co-occurring -> findable.
    present = "We set the goal bias to 3 for all runs in the tree extension."
    assert paper_claim_findable("goal_bias", 3, present) is True


def _mp_spec():
    return {
        "comparison": {
            "classification": {"id": "motion_planning/sampling_based"},
            "pluggable_component": {
                "signature": (
                    "plan(start, goal, environment, dynamics, seed, "
                    "goal_bias: float = 0.1) -> PlanResult"
                )
            },
        },
        "critical_requirements": {},
    }


def test_paradigm_extra_drops_unfindable_textual_paper_claim():
    # A heuristic text scan associates a stray "3" with goal_bias, but the paper
    # never states it near that name. The deriver must NOT stamp "Paper states
    # goal_bias=3" (which halted the iDb-RRT run); it keeps the runtime default
    # without a fabricated paper claim, and the US-3b validator stays clean.
    import json
    paper_map = {"elements": [
        {"name": "goal bias",
         "description": "In the extension step, goal_bias = 3 controls sampling."}
    ]}
    paper_text = "The planner expands a tree toward the goal with motion primitives."
    params: dict = {}
    _add_paradigm_extras(params, _mp_spec(), "motion_planning/sampling_based",
                         paper_map=paper_map, paper_text=paper_text)
    gb = params["goal_bias"]
    assert gb["value"] == 0.1
    assert gb["source"] != "paper"
    assert "Paper states goal_bias" not in json.dumps(gb)
    assert not [f for f in check_params(params, paper_text)
                if f["severity"] == "error"]


def test_paradigm_extra_symbol_mismatch_paper_value_is_not_a_claim():
    # bayesian-active-learning 2026-08-31: the paper states the knob under
    # its own symbol (b = 300), so "Paper states batch_returns=300" is a
    # sentence the US-3b probe rejects on every re-derivation — unfixable
    # by any analyzer retry (six dispatches, converged only by splitting
    # the knob into two entries). The deriver now stamps the claim wording
    # only when the shared findability predicate blesses it; otherwise the
    # paper value is preserved for audit under neutral wording and the
    # probes stay clean.
    spec = _mp_spec()
    spec["critical_requirements"]["scale_dependent_hyperparameters"] = [
        {"name": "goal_bias", "paper_value": 0.3,
         "description": "Goal sampling probability.",
         "paper_section": "Section 4"}]
    paper_text = ("The sampler steers toward the target with probability "
                  "0.3, denoted g_b in the planner loop.")
    params: dict = {}
    _add_paradigm_extras(params, spec, "motion_planning/sampling_based",
                         paper_text=paper_text)
    gb = params["goal_bias"]
    assert gb["paper_value"] == 0.3  # paper truth preserved for audit
    assert "Paper states goal_bias" not in gb["reasoning"]
    assert not [f for f in check_params(params, paper_text)
                if f["severity"] == "error"]


def test_paradigm_extra_findable_paper_value_keeps_the_claim_wording():
    spec = _mp_spec()
    spec["critical_requirements"]["scale_dependent_hyperparameters"] = [
        {"name": "goal_bias", "paper_value": 0.3,
         "description": "Goal sampling probability.",
         "paper_section": "Section 4"}]
    paper_text = "We set the goal bias to 0.3 in every benchmark run."
    params: dict = {}
    _add_paradigm_extras(params, spec, "motion_planning/sampling_based",
                         paper_text=paper_text)
    assert "Paper states goal_bias=0.3" in params["goal_bias"]["reasoning"]


def test_extractor_anchoring_ignores_unrelated_range_phrase():
    # Root cause of the iDb-RRT halt: over a WHOLE document, the param-unanchored
    # "from N to M" phrasing mis-attributed a stray "from 3 to 14" to every param
    # merely mentioned in the paper. With require_param_anchor the number must be
    # tied to the param name, so the stray range is ignored; anchored assignments
    # still parse; and the default (short contract-field) behavior is unchanged.
    from schemas.method_spec import extract_param_paper_values_from_text
    doc = ("The planner uses a goal bias to steer sampling toward the goal. "
           "Separately, the benchmark scales the horizon from 3 to 14 seconds.")
    assert extract_param_paper_values_from_text(
        doc, "goal_bias", require_param_anchor=True) == []
    assert extract_param_paper_values_from_text(
        "We set goal_bias = 0.1 in every run.",
        "goal_bias", require_param_anchor=True) == [0.1]
    # Default contract-field behavior (unanchored allowed) is preserved.
    assert 3 in extract_param_paper_values_from_text(doc, "goal_bias")


# ---------------------------------------------------------------------------
# bev-distill 2026-07-02 F003/F004: paper-stated values the anchored ladder
# could not bind. (F004) the paper states the loss weights as a PAIRED
# assignment ("alpha, beta are ... set to 1.0 and 0.25") under bare greek
# names, while the generated params are role-suffixed (alpha_loss/beta_loss);
# (F003) KD batch_size shipped a reasoning claiming "not extracted from spec"
# while the paper says "batch size is set to 1 per GPU". Pins: the suffix
# head is a name anchor, the paired assignment binds by position, "set to"
# is an assignment verb, and the KD fallback acknowledges a findable paper
# value instead of denying extraction.
# ---------------------------------------------------------------------------

from schemas.method_spec import (extract_param_paper_values_from_text,
                                 param_text_aliases)
from scripts.derive_params import _add_kd_data_params


def test_param_text_aliases_strip_role_suffix():
    assert "alpha" in param_text_aliases("alpha_loss")
    assert "beta" in param_text_aliases("beta_weight")
    # Not a role suffix: no head variant.
    assert param_text_aliases("gamma_qs") == ("gamma_qs", "gamma qs")
    # A degenerate head is never an anchor.
    assert "l" not in param_text_aliases("l_loss")


def test_paired_assignment_binds_by_position():
    text = ("alpha, beta are reweighted factors to balance the supervision "
            "(set to 1.0 and 0.25 by default).")
    assert extract_param_paper_values_from_text(
        text, "alpha_loss", require_param_anchor=True) == [1]
    assert extract_param_paper_values_from_text(
        text, "beta_loss", require_param_anchor=True) == [0.25]
    # A stray pair of numbers with no adjacent param names never binds.
    stray = "The two thresholds were set to 1.0 and 0.25 by default."
    assert extract_param_paper_values_from_text(
        stray, "alpha_loss", require_param_anchor=True) == []


def test_set_to_is_an_assignment_verb():
    text = "During distillation, the batch size is set to 1 per GPU."
    assert extract_param_paper_values_from_text(
        text, "batch_size", require_param_anchor=True) == [1]


def test_paper_claim_findable_accepts_alias_and_integral_renderings():
    # The paper states the value under the bare greek name and as "1.0";
    # the extracted value coerces to int 1 under the generated name.
    paper = ("$\\alpha$, $\\beta$ are reweighted factors to balance the "
             "supervision (set to 1.0 and 0.25 by default).")
    assert paper_claim_findable("alpha_loss", 1, paper) is True
    assert paper_claim_findable("beta_loss", 0.25, paper) is True
    assert paper_claim_findable("gamma_qs", 0.5, paper) is False


def test_kd_batch_size_acknowledges_findable_paper_value():
    params: dict = {}
    spec = {"critical_requirements": {"data_setup": {}}}
    paper = "During the distillation phase, the batch size is set to 1 per GPU."
    _add_kd_data_params(params, spec, paper_map=None, paper_text=paper)
    entry = params["batch_size"]
    assert entry["value"] == 32  # smoke override stands
    assert entry["paper_value"] == 1
    assert "not extracted" not in entry["reasoning"]
    assert "batch_size=1" in entry["reasoning"]


def test_kd_batch_size_honest_fallback_without_evidence():
    params: dict = {}
    spec = {"critical_requirements": {"data_setup": {}}}
    _add_kd_data_params(params, spec, paper_map=None,
                        paper_text="No batch size appears anywhere here.")
    entry = params["batch_size"]
    assert entry["value"] == 32
    assert "no anchored paper statement" in entry["reasoning"].lower()
    assert "not extracted" not in entry["reasoning"]


def test_extras_bind_paired_loss_weights_with_section(tmp_path):
    from scripts.derive_params import _add_paradigm_extras

    params: dict = {}
    spec = {"comparison": {"pluggable_component": {
        "signature": ("compute_distillation_loss(student, teacher, batch, "
                      "seed, alpha_loss: float = 1.0, "
                      "beta_loss: float = 0.25) -> Tensor")}}}
    paper_map = {"elements": [{
        "id": "hyp-loss-weights",
        "section": "Section 3.2.2",
        "source_text": ("alpha, beta are reweighted factors to balance the "
                        "supervision (set to 1.0 and 0.25 by default)."),
    }]}
    paper = ("$\\alpha$, $\\beta$ are reweighted factors to balance the "
             "supervision (set to 1.0 and 0.25 by default).")
    _add_paradigm_extras(params, spec, "knowledge_distillation/detection",
                         paper_map=paper_map, paper_text=paper)
    assert params["alpha_loss"]["source"] == "paper"
    assert params["alpha_loss"]["paper_section"] == "Section 3.2.2"
    assert params["beta_loss"]["source"] == "paper"
    assert params["beta_loss"]["value"] == 0.25


# ---------------------------------------------------------------------------
# SRL 2026-07-21 stage-2x halt F001 (critical): the spec's LR free-text said
# the paper does NOT state the value ("Not explicitly stated in the paper;
# RMSprop is named in Algorithm 1 line 13"), but the old phrase list missed
# the "stated" variant, so the string fell to the heuristic prose scan —
# which extracted the "1" of "Algorithm 1" as a fabricated learning rate of
# 1.0 (immediate divergence for RMSprop). Pins: (1) prose that declares
# silence is never value-scanned, (2) bare integers in prose never parse as
# a learning rate even inside the plausible range, (3) the entry ships as an
# honest system_inferred default.
# ---------------------------------------------------------------------------

SRL_LR_TEXT = ("Not explicitly stated in the paper; RMSprop is named in "
               "Algorithm 1 line 13")


def test_lr_unspecified_statement_is_never_prose_scanned():
    value, note = _parse_learning_rate(SRL_LR_TEXT, is_image=False)
    assert value == 0.001, f"fabricated LR from silence-declaring prose: {value}"
    assert "not stated" in note
    assert "first plausible" not in note


def test_lr_unspecified_statement_ships_honest_inferred_entry():
    params: dict = {}
    spec = _training_spec("episodic deep V-learning")
    spec["critical_requirements"]["training"]["learning_rate"] = SRL_LR_TEXT
    _add_training_params(params, spec)
    entry = params["learning_rate"]
    assert entry["value"] == 0.001
    assert entry["source"] == "system_inferred"
    assert "does not explicitly specify" in entry["reasoning"]


def test_lr_bare_integer_prose_never_parses_as_lr():
    # No silence-declaring phrase, so the string reaches the heuristic scan;
    # the only in-range numbers are bare integers (a line number and an
    # algorithm number), which are never learning rates.
    value, note = _parse_learning_rate(
        "RMSprop with the settings of Algorithm 1 line 13", is_image=False)
    assert value == 0.001
    assert "not parseable" in note
    # A bare-integer LEAD is a count, not a learning rate ("1 epoch..."),
    # even though 1 sits inside the plausible LR range.
    value, note = _parse_learning_rate(
        "1 cycle policy at 0.004 peak", is_image=False)
    assert value == 0.004, f"bare integer lead leaked as LR: {value}"


def test_lr_known_good_phrasings_still_parse():
    # Adjacent-good coverage for the silence regex: none of these carry a
    # silence-declaring phrase, and each still parses to its stated value.
    assert _parse_learning_rate("0.001", is_image=False)[0] == 0.001
    assert _parse_learning_rate("2e-4 initial, cyclic policy",
                                is_image=True)[0] == 2e-4
    assert _parse_learning_rate(
        "0.01 for image data, 0.0001 for non-image data",
        is_image=True)[0] == 0.01


# ---------------------------------------------------------------------------
# ROMAN25 2026-07-21 disclosure: the generated signature renames paper
# symbols (the paper's `beta` becomes `beta_sigma`), so the exact-name
# anchored scan cannot bind the paper's `beta=3.0` and four paper-stated
# values shipped labeled spec_default. The analyzer's param glossary is the
# declared link (name <-> aliases, quote-floor validated); the textual
# ladder now anchors under glossary-linked names too. Without the declared
# link nothing binds — suffix guessing would misattribute v_max to
# v_max_CCA in the same paper.
# ---------------------------------------------------------------------------

ROMAN_PAPER_MAP = {"elements": [{
    "id": "hyp-sigma-params",
    "type": "hyperparameter",
    "name": "Sigma Growth Parameters (beta, gamma, delta, sigma_0)",
    "section": "Section IV-C (Risk calculation)",
    "pseudocode": "sigma_0 = 0.1666; beta = 3.0; gamma = 0.4; delta = 0.015",
}]}
ROMAN_PAPER = ("In order to let sigma stay constant over time at standstill, "
               "we choose beta=3.0, gamma=0.4, and delta=0.015 for all "
               "agents. The maximum velocity v_max is 1.0 m/s.")


def _mp_renamed_symbol_spec(glossary: list) -> dict:
    return {
        "comparison": {
            "classification": {"id": "motion_planning"},
            "pluggable_component": {
                "signature": ("plan(start, goal, environment, dynamics, seed, "
                              "beta_sigma: float = 3.0, "
                              "v_max_CCA: float = 1.0) -> PlanResult"),
            },
        },
        "critical_requirements": {"param_glossary": glossary},
    }


def test_glossary_link_binds_renamed_symbol_from_paper_map():
    spec = _mp_renamed_symbol_spec([
        {"name": "beta", "aliases": ["beta_sigma"],
         "meaning_quote": "grow to three times the initial value",
         "paper_section": "Section IV-C"},
    ])
    params: dict = {}
    _add_paradigm_extras(params, spec, "motion_planning",
                         paper_map=ROMAN_PAPER_MAP, paper_text=ROMAN_PAPER)
    entry = params["beta_sigma"]
    assert entry["source"] == "paper", entry
    assert entry["value"] == 3.0
    assert entry["paper_section"] == "Section IV-C (Risk calculation)"
    # The unlinked renamed symbol in the SAME spec stays a signature
    # default: the paper's v_max belongs to v_max, not v_max_CCA.
    assert params["v_max_CCA"]["source"] == "spec_default"


def test_unlinked_renamed_symbol_never_binds():
    params: dict = {}
    _add_paradigm_extras(params, _mp_renamed_symbol_spec([]),
                         "motion_planning",
                         paper_map=ROMAN_PAPER_MAP, paper_text=ROMAN_PAPER)
    assert params["beta_sigma"]["source"] == "spec_default"


def test_contract_scan_drops_unanchored_value_in_multi_param_text():
    # GBALD 2026-09-01 stage-2x halt: one contract clause names two params —
    # "with the reduced mc_samples (taxonomy floor 20, paper value 2000
    # recorded); score ... the top batch_returns by BALD score" — and the
    # unanchored "paper value 2000" was attributed to batch_returns,
    # fabricating a paper claim the US-3b provenance probe then halted on.
    # Same failure class as the iDb-RRT whole-document case: unanchored
    # patterns are only safe when the text mentions no sibling param.
    from schemas.method_spec import (
        _extract_param_paper_values_from_text,
        methodology_contract_paper_values_for_param,
    )
    clause = ("Run select_batch with the reduced mc_samples (taxonomy floor "
              "20, paper value 2000 recorded); score the full demo pool and "
              "take the top batch_returns by BALD score.")
    siblings = ("mc_samples", "batch_returns", "eta", "R_0")
    # Known-bad: scanning for batch_returns must NOT pick up 2000.
    assert _extract_param_paper_values_from_text(
        clause, "batch_returns",
        other_param_names=siblings) == []
    # The same clause yields nothing for mc_samples too, correctly: it is
    # multi-param text and states no ANCHORED mc_samples value ("paper value
    # 2000 recorded" never names the param it belongs to).
    assert _extract_param_paper_values_from_text(
        clause, "mc_samples",
        other_param_names=siblings) == []
    # Known-good: single-param text keeps the unanchored contract behavior.
    single = "mc_samples: the paper uses 2000 forward passes."
    assert _extract_param_paper_values_from_text(
        single, "mc_samples",
        other_param_names=siblings) == [2000]
    # End-to-end through the contract scanner.
    contract = {"elements": [{
        "element_id": "core-1",
        "paper_evidence": clause,
    }]}
    assert methodology_contract_paper_values_for_param(
        contract, "batch_returns",
        other_param_names=siblings) == []
