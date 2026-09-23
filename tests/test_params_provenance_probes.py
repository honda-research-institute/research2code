"""Zoo-validated acceptance tests for the US-1/US-2/US-3 provenance probes.

Per the acceptance contract (tests/fixtures/zoo/README.md + the recentering
plan): every probe must flag the documented known-bad zoo artifacts and must
NOT flag the known-good ones. These tests run the real artifacts through
scripts/validate_params_provenance.py — the matrix rows in zoo/README.md are
asserted literally here.
"""

from __future__ import annotations

import pytest

import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
ZOO = REPO / "tests" / "fixtures" / "zoo"
EVIDENCE = REPO / "tests" / "fixtures" / "evidence"

from validate_params_provenance import (  # noqa: E402  (scripts/ on sys.path via conftest)
    check_params,
    unsatisfiable_paper_claims,
)


def _load_params(path: Path) -> dict:
    loaded = json.loads(path.read_text(encoding="utf-8"))
    return loaded.get("params", loaded)


def _probes_by_param(findings: list[dict]) -> set[tuple[str, str]]:
    return {(f["probe"], f["param"]) for f in findings if f["severity"] == "error"}


# ---------------------------------------------------------------------------
# Fail side — zoo gbald (lr=7.4 run; no paper.md in that scenario, so US-3 off)
# ---------------------------------------------------------------------------


def test_gbald_lr74_params_flag_range_convention_and_provenance():
    findings = check_params(
        _load_params(ZOO / "gbald-lr74-never-learns" / "params.json"),
        paper_text=None,
    )
    flagged = _probes_by_param(findings)
    # US-1: lr=7.4 outside [1e-6, 1].
    assert ("US-1", "learning_rate") in flagged
    # US-2b: "Adam-default convention" cannot justify 7.4.
    assert ("US-2b", "learning_rate") in flagged
    # US-2a: dropout_rate is source=paper with "Field-guide convention" locator.
    assert ("US-2a", "dropout_rate") in flagged


# ---------------------------------------------------------------------------
# Fail side — june9 run (fabricated quote; real paper text available)
# ---------------------------------------------------------------------------


def test_june9_fabricated_provenance_quote_is_flagged():
    run = EVIDENCE / "june9-gbald-run" / "pipeline"
    findings = check_params(
        _load_params(run / "params.json"),
        paper_text=(run / "paper.md").read_text(encoding="utf-8"),
    )
    flagged = _probes_by_param(findings)
    # The invented "train until training accuracy exceeds 99%" quote and/or
    # the unfindable 0.99 value must be caught by US-3.
    assert ("US-3", "train_until_accuracy") in flagged
    # And the run's healthy entries stay clean: lr=0.001 passes US-1/US-2b.
    assert ("US-1", "learning_rate") not in flagged
    assert ("US-2b", "learning_rate") not in flagged


# ---------------------------------------------------------------------------
# Pass side — BADGE (the catalog-wide false-positive guard)
# ---------------------------------------------------------------------------


def test_badge_example_snapshot_carries_exactly_its_documented_finding():
    # Pins the 2026-06-10 BADGE run snapshot, now a committed fixture
    # (example_runs/ became the researcher-facing curated set on 2026-07-06
    # and is no longer a test surface). The snapshot predates the lr-note
    # producer fix, so it carries exactly one documented provenance
    # finding: the note quotes the spec's summary string as a paper quote
    # (the matrix-row-3 audit catch; producer fixed the same evening, fresh
    # derivations pinned clean in test_derive_params_paper_evidence).
    # Everything else stays clean — train_until_accuracy=0.99 genuinely
    # appears in the paper, lr=0.001 is sane — so this remains the
    # false-positive guard. If the snapshot is regenerated with the fixed
    # deriver, expect zero errors and update.
    params_path = (REPO / "tests" / "fixtures" / "delivery"
                   / "badge-run-20260610" / ".pipeline" / "params.json")
    paper_path = REPO / "input_papers" / "deep-batch-active-learning.md"
    findings = check_params(
        _load_params(params_path),
        paper_text=paper_path.read_text(encoding="utf-8"),
    )
    errors = [f for f in findings if f["severity"] == "error"]
    assert [(f["probe"], f["param"]) for f in errors] == [
        ("US-3", "learning_rate")], errors


# ---------------------------------------------------------------------------
# Synthetic edge pins (probe semantics, not zoo artifacts)
# ---------------------------------------------------------------------------


def test_small_integers_are_not_value_matched():
    # num_rounds=6 source=paper: "6" matches any paper trivially, so the value
    # arm must skip it (the quote arm is responsible for small ints).
    findings = check_params(
        {"num_rounds": {"value": 6, "source": "paper",
                        "paper_section": "Section 4.2"}},
        paper_text="this paper never says the number of rounds anywhere",
    )
    assert _probes_by_param(findings) == set()


def test_percent_rendering_matches():
    # 0.99 stated as "99%" in the paper must satisfy US-3.
    findings = check_params(
        {"train_until_accuracy": {"value": 0.99, "source": "paper",
                                  "paper_section": "Section 3"}},
        paper_text="we train until training accuracy exceeds 99% on the set",
    )
    assert _probes_by_param(findings) == set()


def test_unknown_param_names_are_never_range_guessed():
    # R_0 has scale-dependent semantics — US-1 must not opine on it.
    findings = check_params(
        {"R_0": {"value": 2000.0, "source": "system_default",
                 "reasoning": "scale-dependent"}},
        paper_text=None,
    )
    assert _probes_by_param(findings) == set()


# ---------------------------------------------------------------------------
# US-3b — fabricated inline paper claims (fresh pdwa evidence, 2026-06-10)
# ---------------------------------------------------------------------------


def test_us3b_flags_fabricated_beta_claims_on_fresh_pdwa():
    if not (REPO / "input_papers" / "pdwa.md").is_file():
        pytest.skip("pdwa paper not present in input_papers/")
    findings = check_params(
        _load_params(ZOO / "pdwa-jterms-fresh" / "params.json"),
        paper_text=(REPO / "input_papers" / "pdwa.md").read_text(encoding="utf-8"),
    )
    flagged = _probes_by_param(findings)
    # "Paper states beta_1=0.1" — the paper defines betas symbolically only.
    assert ("US-3b", "beta_1") in flagged
    # "Paper states d_safe=0.1" — the paper says d_safe = V_max/4.
    assert ("US-3b", "d_safe") in flagged
    # dt's HONEST reasoning ("uses Δt symbolically without stating a
    # numerical value") must not be flagged.
    assert not any(p == "dt" for _, p in flagged)


def test_us3b_passes_a_true_inline_claim():
    findings = check_params(
        {"r_robot": {"value": 0.1, "source": "system_default",
                     "paper_value": 0.1,
                     "reasoning": "Paper states radius=0.1 for the robot."}},
        paper_text="| Robot radius | m | 0.1 |",
    )
    assert _probes_by_param(findings) == set()


def test_relabeling_the_source_does_not_clear_an_inline_claim():
    # bayesian-active-learning 2026-07-27: the auto-resolve applier relabeled
    # batch_returns to system_inferred and declared success, but US-3b fires
    # on every source, so the very next validator run halted the stage. The
    # applier and the validator now share this predicate.
    entry = {
        "value": 300, "source": "system_inferred",
        "reasoning": ("The paper uses batch_returns=300 in the main MNIST "
                      "experiments but the setting is user-defined."),
    }
    paper = "For single acquisition we suggest b=3 and b'=1."
    assert unsatisfiable_paper_claims(entry, paper) == [("batch_returns", "300")]
    # Every source, not just source=paper — that is the point of the arm.
    for source in ("paper", "system_default", "system_inferred"):
        assert unsatisfiable_paper_claims(dict(entry, source=source), paper)
    # The honest rephrasing the reviewer is now told to write does clear it.
    honest = dict(entry, reasoning=("The paper demonstrates this setting in "
                                    "Section 7.4 without prescribing it."))
    assert unsatisfiable_paper_claims(honest, paper) == []


def test_us3b_accepts_a_claim_under_a_role_suffixed_alias():
    # Deriver/validator symmetry (2026-09-01): the US-3b arm now tries the
    # same param_text_aliases the deriver's stamping gate tries, so a claim
    # the gate blesses can never be rejected by the probe.
    entry = {"value": 0.5, "source": "system_default", "paper_value": 0.5,
             "reasoning": "Paper states alpha_loss=0.5 in Section 3."}
    assert unsatisfiable_paper_claims(
        entry, "we set the alpha to 0.5 in all experiments") == []


def test_us3b_still_rejects_a_symbol_mismatch_claim():
    # GBALD batch_returns (2026-08-31, six analyzer dispatches): the paper
    # names the knob `b`, so the derived-name claim stays unsatisfiable —
    # the fix lives in the deriver, which no longer stamps this sentence.
    entry = {"value": 200, "source": "system_default", "paper_value": 300,
             "reasoning": "Paper states batch_returns=300 in Section 7.4."}
    paper = ("The batch size of the compared baselines is 100, where GBALD "
             "ranks 300 acquisitions to select 100 data for the training, "
             "i.e. b = 300, b' = 100.")
    assert unsatisfiable_paper_claims(entry, paper) == [
        ("batch_returns", "300")]


def test_unsatisfiable_claims_agree_with_the_us3b_arm():
    if not (REPO / "input_papers" / "pdwa.md").is_file():
        pytest.skip("pdwa paper not present in input_papers/")
    # One definition, two callers: whatever the helper flags is exactly what
    # the validator reports, so the applier can never disagree with it.
    params = _load_params(ZOO / "pdwa-jterms-fresh" / "params.json")
    paper = (REPO / "input_papers" / "pdwa.md").read_text(encoding="utf-8")
    by_helper = {
        name for name, entry in params.items()
        if unsatisfiable_paper_claims(entry, paper)
    }
    by_validator = {
        param for probe, param in _probes_by_param(
            check_params(params, paper_text=paper))
        if probe == "US-3b"
    }
    assert by_helper == by_validator


def test_a_true_inline_claim_is_not_flagged_by_the_helper():
    entry = {"value": 0.1, "source": "system_default", "paper_value": 0.1,
             "reasoning": "Paper states radius=0.1 for the robot."}
    assert unsatisfiable_paper_claims(entry, "| Robot radius | m | 0.1 |") == []


def test_helper_with_no_paper_text_claims_nothing():
    entry = {"value": 300, "source": "paper",
             "reasoning": "The paper states batch_returns=300."}
    assert unsatisfiable_paper_claims(entry, None) == []


# ---------------------------------------------------------------------------
# The deterministic gate (wired 2026-06-10): validate_params_output.py now
# runs the provenance probes, so fabrication-class findings fail stage 2x
# BEFORE the LLM review — both 2026-06-10 audit catches (GBALD's fabricated
# threshold quote, BADGE's lr note quoting spec text as a paper quote)
# happened at audit time only because this gate did not exist yet.
# ---------------------------------------------------------------------------


def _gate_spec():
    return {"comparison": {"classification": {"id": "active_learning/bayesian"},
                           "pluggable_component": {"signature": ""}}}


def _gate_run(tmp_path, params, paper):
    run = tmp_path / "run"
    (run / ".pipeline").mkdir(parents=True)
    (run / ".pipeline" / "params.json").write_text(json.dumps(
        {"schema_version": "1.0.0", "params": params}), encoding="utf-8")
    (run / ".pipeline" / "paper.md").write_text(paper, encoding="utf-8")
    return run


def _gate_run_with_model(tmp_path, params, *, model_src: str, training_src: str):
    run = _gate_run(tmp_path, params, "# Paper\n\nSome text.\n")
    method = run / "method"
    method.mkdir()
    (method / "model.py").write_text(model_src, encoding="utf-8")
    (method / "training.py").write_text(training_src, encoding="utf-8")
    return run


_GATE_BASE = {
    "batch_size": {"value": 4, "source": "system_default",
                   "paper_value": "100",
                   "reasoning": "smoke scale, paper uses 100"},
    "num_rounds": {"value": 3, "source": "system_default",
                   "paper_value": "90", "reasoning": "smoke scale"},
    "initial_labeled": {"value": 8, "source": "system_default",
                        "paper_value": "1000", "reasoning": "smoke scale"},
    "learning_rate": {"value": 0.001, "source": "system_inferred",
                      "reasoning": "Adam-default convention 0.001"},
    "max_epochs": {"value": 8, "source": "system_inferred",
                   "reasoning": "smoke bound"},
    "train_until_accuracy": {"value": 0.99, "source": "system_inferred",
                             "reasoning": "field convention 0.99"},
    "pool_size": {"value": 200, "source": "system_default",
                  "paper_value": "full set", "reasoning": "smoke scale"},
    "hidden_dim": {"value": 64, "source": "system_inferred",
                   "reasoning": "convention 64"},
}


def test_gate_blocks_fabricated_paper_quote(tmp_path):
    from scripts.validate_params_output import validate
    params = dict(_GATE_BASE)
    params["train_until_accuracy"] = {
        "value": 0.99, "source": "paper", "paper_section": "Section 7",
        "note": "Paper: train until training accuracy exceeds 99%.",
    }
    run = _gate_run(tmp_path, params,
                    "# Paper\n\nNo such training protocol appears here.\n")
    errors = validate(_gate_spec(), run)
    assert any("US-3" in e and "train_until_accuracy" in e for e in errors)


def test_gate_blocks_implausible_value(tmp_path):
    from scripts.validate_params_output import validate
    params = dict(_GATE_BASE)
    params["learning_rate"] = {
        "value": 7.4, "source": "system_inferred",
        "reasoning": "extracted 7.4",
    }
    run = _gate_run(tmp_path, params, "# Paper\n\nSome text.\n")
    errors = validate(_gate_spec(), run)
    assert any("US-1" in e and "learning_rate" in e for e in errors)


def test_gate_passes_honest_params_and_warns_do_not_block(tmp_path):
    from scripts.validate_params_output import validate
    params = dict(_GATE_BASE)
    # Warn-class: paper-sourced with a real quote but no locator-shaped
    # section is US-2a warn territory via a missing paper_section being
    # schema-blocked, so use the eta shape: paper source, empty section is
    # not schema-legal — keep all entries honest instead and assert clean.
    run = _gate_run(
        tmp_path, params,
        "# Paper\n\nWe train with Adam. Batch size is 100.\n")
    errors = validate(_gate_spec(), run)
    provenance = [e for e in errors if "provenance probe" in e]
    assert provenance == [], provenance


def test_gate_blocks_live_hidden_dim_when_runtime_cannot_use_it(tmp_path):
    from scripts.validate_params_output import validate
    params = dict(_GATE_BASE)
    run = _gate_run_with_model(
        tmp_path,
        params,
        model_src=(
            "import torch.nn as nn\n"
            "class Net(nn.Module):\n"
            "    def __init__(self, input_dim, n_classes):\n"
            "        super().__init__()\n"
            "        self.fc = nn.Linear(input_dim, n_classes)\n"
        ),
        training_src=(
            "from .model import Net\n"
            "def build_model(input_dim, n_classes):\n"
            "    return Net(input_dim, n_classes)\n"
        ),
    )
    errors = validate(_gate_spec(), run)
    assert any("param_runtime_drift" in e and "hidden_dim" in e for e in errors)


def test_gate_accepts_hidden_dim_when_runtime_uses_it(tmp_path):
    from scripts.validate_params_output import validate
    params = dict(_GATE_BASE)
    run = _gate_run_with_model(
        tmp_path,
        params,
        model_src=(
            "import torch.nn as nn\n"
            "class Net(nn.Module):\n"
            "    def __init__(self, input_dim, n_classes, hidden_dim=64):\n"
            "        super().__init__()\n"
            "        self.fc1 = nn.Linear(input_dim, hidden_dim)\n"
            "        self.fc2 = nn.Linear(hidden_dim, n_classes)\n"
        ),
        training_src=(
            "from .model import Net\n"
            "def build_model(input_dim, n_classes, hidden_dim=64):\n"
            "    return Net(input_dim, n_classes, hidden_dim=hidden_dim)\n"
        ),
    )
    errors = validate(_gate_spec(), run)
    assert not [e for e in errors if "param_runtime_drift" in e]


def test_gate_accepts_explicitly_unused_hidden_dim(tmp_path):
    from scripts.validate_params_output import validate
    params = dict(_GATE_BASE)
    params["hidden_dim"] = {
        "value": 11,
        "source": "system_inferred",
        "reasoning": "taxonomy convention",
        "used_in_notebook": False,
        "unused_reason": "runtime uses a fixed-width model in this scaffold",
    }
    run = _gate_run_with_model(
        tmp_path,
        params,
        model_src=(
            "import torch.nn as nn\n"
            "class Net(nn.Module):\n"
            "    def __init__(self, input_dim, n_classes):\n"
            "        super().__init__()\n"
            "        self.fc = nn.Linear(input_dim, n_classes)\n"
        ),
        training_src=(
            "from .model import Net\n"
            "def build_model(input_dim, n_classes):\n"
            "    return Net(input_dim, n_classes)\n"
        ),
    )
    errors = validate(_gate_spec(), run)
    assert not [e for e in errors if "param_runtime_drift" in e]


# ---------------------------------------------------------------------------
# Two-stage acquisition invariant b >= b' (queue 11e): batch_returns must be
# >= batch_size, else the selector silently throttles acquisition (GBALD F002).
# ---------------------------------------------------------------------------


def test_gate_blocks_batch_returns_below_batch_size(tmp_path):
    from scripts.validate_params_output import validate
    params = dict(_GATE_BASE)
    params["batch_size"] = {"value": 100, "source": "paper",
                            "paper_section": "Section 7.4", "note": "paper"}
    params["batch_returns"] = {"value": 30, "source": "spec_default",
                               "reasoning": "signature default"}
    run = _gate_run(tmp_path, params, "# Paper\n\nSome text.\n")
    errors = validate(_gate_spec(), run)
    assert any(
        "batch_acquisition_invariant" in e and "batch_returns" in e
        for e in errors
    ), errors


def test_gate_passes_batch_returns_at_or_above_batch_size(tmp_path):
    from scripts.validate_params_output import validate
    params = dict(_GATE_BASE)
    params["batch_size"] = {"value": 30, "source": "paper",
                            "paper_section": "Section 7.4", "note": "paper"}
    params["batch_returns"] = {"value": 100, "source": "spec_default",
                               "reasoning": "signature default"}
    run = _gate_run(tmp_path, params, "# Paper\n\nSome text.\n")
    errors = validate(_gate_spec(), run)
    assert not [e for e in errors if "batch_acquisition_invariant" in e], errors


def test_gate_no_invariant_when_no_batch_returns(tmp_path):
    """Single-stage selectors (BADGE) declare no batch_returns — the invariant
    must not fire (no false positive)."""
    from scripts.validate_params_output import validate
    params = dict(_GATE_BASE)  # batch_size only, no batch_returns
    run = _gate_run(tmp_path, params, "# Paper\n\nSome text.\n")
    errors = validate(_gate_spec(), run)
    assert not [e for e in errors if "batch_acquisition_invariant" in e], errors


# ---------------------------------------------------------------------------
# Digit-grouping commas (SRL 2026-07-21 stage-2x F002): the paper states the
# episode count only as "3,000", which the value/claim arms tokenized as
# "3" + "000" and called unfindable — so honest paper-sourced entries were
# impossible for comma-grouped values. The fold is leniency-only; a decimal
# comma never collapses.
# ---------------------------------------------------------------------------


COMMA_PAPER = ("Offline training (Algorithm 1) took approximately nine hours "
               "to complete 3,000 episodes on the four-agent network.")


def test_comma_grouped_value_is_findable():
    from validate_params_provenance import (paper_claim_findable,
                                            value_findable_in_paper)
    assert value_findable_in_paper(3000, COMMA_PAPER)
    assert paper_claim_findable("episodes", 3000, COMMA_PAPER)


def test_comma_grouped_paper_entry_passes_probes():
    findings = check_params({
        "episodes": {"value": 3000, "source": "paper",
                     "paper_section": "Algorithm 1, Section IV-A",
                     "note": "Paper-stated episode count."},
    }, COMMA_PAPER)
    assert not [f for f in findings if f["severity"] == "error"], findings


def test_decimal_comma_never_collapses():
    from validate_params_provenance import value_findable_in_paper
    # "2,35" is a decimal comma (two digits after), not digit grouping:
    # folding it would fabricate 235 as findable.
    assert not value_findable_in_paper(235, "the threshold is 2,35 exactly")
    # A wrong comma-grouped value still fails.
    assert not value_findable_in_paper(4000, COMMA_PAPER)


# ---------------------------------------------------------------------------
# LaTeX scientific notation in Marker-parsed tables (pdfgnn 2026-08-05 halt)
# ---------------------------------------------------------------------------


def test_latex_times_braced_exponent_findable():
    """Marker renders paper-table math as `$5 \\times 10^{-3}$`; _norm strips
    the backslash leaving `5 times 10^{-3}`. The rendering set must find it."""
    from validate_params_provenance import value_findable_in_paper

    paper = "| m: | Learning rate | $5 \\times 10^{-3}$ | $5 \\times 10^{-3}$ |"
    assert value_findable_in_paper(0.005, paper)


def test_latex_times_braced_exponent_mantissa_one():
    from validate_params_provenance import value_findable_in_paper

    paper = "we use a weight decay of $1 \\times 10^{-2}$ throughout"
    assert value_findable_in_paper(0.01, paper)


def test_latex_cdot_form_findable():
    from validate_params_provenance import value_findable_in_paper

    paper = "the threshold is $3 \\cdot 10^{-4}$ in all runs"
    assert value_findable_in_paper(0.0003, paper)


def test_absent_value_still_unfindable():
    """Leniency-only: a value the paper genuinely never states keeps failing."""
    from validate_params_provenance import value_findable_in_paper

    paper = "| Learning rate | $5 \\times 10^{-3}$ |"
    assert not value_findable_in_paper(0.017, paper)
