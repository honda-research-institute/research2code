"""US-8 notebook narrative-vs-params probe tests."""

from __future__ import annotations

import json

from probes.narrative import probe_narrative_vs_params


def _seed_run(tmp_path, *, markdown: str, params: dict) -> object:
    run = tmp_path / "run"
    pipeline = run / ".pipeline"
    pipeline.mkdir(parents=True)
    (pipeline / "params.json").write_text(
        json.dumps({"params": params}, indent=2) + "\n",
        encoding="utf-8",
    )
    (run / "notebook.ipynb").write_text(
        json.dumps({
            "cells": [{"cell_type": "markdown", "source": markdown}],
            "metadata": {},
            "nbformat": 4,
            "nbformat_minor": 5,
        }) + "\n",
        encoding="utf-8",
    )
    return run


def test_us8_flags_stale_markdown_param_claim(tmp_path):
    run = _seed_run(
        tmp_path,
        markdown="This notebook uses `R_0=2000.0` for the geometric prior.",
        params={"R_0": {"value": 7.843}},
    )

    verdict = probe_narrative_vs_params(run)

    assert verdict.probe_id == "US-8"
    assert verdict.verdict == "flag_for_researcher"
    assert "R_0 is described as 2000.0" in verdict.message
    assert "7.843" in verdict.message


def test_us8_allows_rescaled_param_with_paper_original_nearby(tmp_path):
    run = _seed_run(
        tmp_path,
        markdown=(
            "This notebook uses `R_0=7.843`, rescaled from the paper's "
            "2000.0 value to match normalized pixels."
        ),
        params={"R_0": {"value": 7.843}},
    )

    verdict = probe_narrative_vs_params(run)

    assert verdict.verdict == "pass"


def test_us8_allows_explicit_paper_original_context(tmp_path):
    run = _seed_run(
        tmp_path,
        markdown=(
            "The paper's `R_0=2000.0` was calibrated for raw pixels. "
            "This notebook uses `R_0=7.843` after rescaling."
        ),
        params={"R_0": {"value": 7.843}},
    )

    verdict = probe_narrative_vs_params(run)

    assert verdict.verdict == "pass"
    assert "1 checked" in verdict.message


def test_us8_allows_rounded_runtime_value(tmp_path):
    run = _seed_run(
        tmp_path,
        markdown="This notebook uses `R_0=7.84` after rescaling.",
        params={"R_0": {"value": 7.843}},
    )

    verdict = probe_narrative_vs_params(run)

    assert verdict.verdict == "pass"


def test_us8_allows_thousands_separator_runtime_value(tmp_path):
    run = _seed_run(
        tmp_path,
        markdown="This demo uses `pool_size = 12 000` examples.",
        params={"pool_size": {"value": 12000}},
    )

    verdict = probe_narrative_vs_params(run)

    assert verdict.verdict == "pass"


def test_us8_allows_compact_k_runtime_value(tmp_path):
    run = _seed_run(
        tmp_path,
        markdown="This demo uses `pool_size=12 k` examples.",
        params={"pool_size": {"value": 12000}},
    )

    verdict = probe_narrative_vs_params(run)

    assert verdict.verdict == "pass"


def test_us8_still_flags_stale_compact_runtime_value(tmp_path):
    run = _seed_run(
        tmp_path,
        markdown="This demo still says `pool_size=10k` examples.",
        params={"pool_size": {"value": 12000}},
    )

    verdict = probe_narrative_vs_params(run)

    assert verdict.verdict == "flag_for_researcher"
    assert "pool_size is described as 10k" in verdict.message


def test_us8_catches_common_r0_alias(tmp_path):
    run = _seed_run(
        tmp_path,
        markdown="This notebook still says `R0=2000.0`.",
        params={"R_0": {"value": 7.843}},
    )

    verdict = probe_narrative_vs_params(run)

    assert verdict.verdict == "flag_for_researcher"
    assert "R_0 is described as 2000.0" in verdict.message


def test_us8_reports_unprobeable_when_inputs_are_missing(tmp_path):
    run = tmp_path / "run"
    (run / ".pipeline").mkdir(parents=True)

    verdict = probe_narrative_vs_params(run)

    assert verdict.verdict == "unprobeable"
    assert "params.json" in verdict.message


# ---------------------------------------------------------------------------
# R2C-041 claim guards (RCA 2026-08-03 finding 3a): the three observed
# false-positive shapes pass clean, the motivating true positives keep
# firing.
# ---------------------------------------------------------------------------


def test_us8_instructional_sentence_is_not_a_claim(tmp_path):
    # SRL: "set qn=0.0 to train a norm-free policy" flagged against the
    # shipped 1.0 — an instruction, not a claim about the run.
    run = _seed_run(
        tmp_path,
        markdown="Set `qn=0.0` to train a norm-free policy variant.",
        params={"qn": {"value": 1.0}},
    )

    assert probe_narrative_vs_params(run).verdict == "pass"


def test_us8_counterfactual_sentence_is_not_a_claim(tmp_path):
    run = _seed_run(
        tmp_path,
        markdown="If you prefer a lighter demo, you can work with "
                 "`pool_size=500` instead.",
        params={"pool_size": {"value": 12000}},
    )

    assert probe_narrative_vs_params(run).verdict == "pass"


def test_us8_declarative_we_set_still_flags(tmp_path):
    # The floor that must hold: the imperative guard cannot hide the
    # motivating declarative shape.
    run = _seed_run(
        tmp_path,
        markdown="We set `qn=0.0` throughout this notebook.",
        params={"qn": {"value": 1.0}},
    )

    verdict = probe_narrative_vs_params(run)

    assert verdict.verdict == "flag_for_researcher"
    assert "qn is described as 0.0" in verdict.message


def test_us8_meters_suffix_is_not_a_multiplier(tmp_path):
    # pdwa: "0.15 m" (meters) parsed as a times-a-million multiplier.
    run = _seed_run(
        tmp_path,
        markdown="The demo keeps `r_robot=0.15` m for the robot radius.",
        params={"r_robot": {"value": 0.15}},
    )

    assert probe_narrative_vs_params(run).verdict == "pass"


def test_us8_arithmetic_tail_is_not_a_claim(tmp_path):
    # pdwa: "r_robot + r_obs = 0.25" read as a claim about r_obs.
    run = _seed_run(
        tmp_path,
        markdown="The combined radius is r_robot + r_obs = 0.25 in the "
                 "collision check.",
        params={"r_obs": {"value": 0.1}, "r_robot": {"value": 0.15}},
    )

    assert probe_narrative_vs_params(run).verdict == "pass"


def test_us8_bullet_dash_is_not_arithmetic(tmp_path):
    # A markdown '-' bullet is not an operator; a stale bullet claim still
    # flags.
    run = _seed_run(
        tmp_path,
        markdown="Key settings:\n\n- `R_0 = 2000.0` (geometric prior)\n",
        params={"R_0": {"value": 7.843}},
    )

    assert probe_narrative_vs_params(run).verdict == "flag_for_researcher"


def test_us8_rendered_params_table_is_exempt(tmp_path):
    # bayesian: a match inside the machine-rendered provenance table, where
    # "Section 7.1" self-destructed the old context window. The table holds
    # paper values beside run values by construction.
    table = (
        "| Parameter | Variable from paper | Value from paper | Paper value "
        "| System value | Where in paper | Used? | Notes |\n"
        "|---|:-:|:-:|---|---|---|:-:|---|\n"
        "| learning_rate | eta | eta=0.01 | 0.01 | 0.001 | Section 7.1 | "
        "yes | demo-scale |\n"
    )
    run = _seed_run(
        tmp_path,
        markdown=table,
        params={"learning_rate": {"value": 0.001}},
    )

    assert probe_narrative_vs_params(run).verdict == "pass"


def test_us8_paper_context_after_the_match_counts(tmp_path):
    # The context window is symmetric now: "as reported in the paper" AFTER
    # the literal is as exculpatory as before it.
    run = _seed_run(
        tmp_path,
        markdown="Training used `R_0=2000.0` as reported in the paper; "
                 "this demo rescales it.",
        params={"R_0": {"value": 7.843}},
    )

    assert probe_narrative_vs_params(run).verdict == "pass"


def test_us8_digit_safe_window_keeps_section_numbers(tmp_path):
    # "Section 7.1" must not split the sentence and orphan the paper cue.
    run = _seed_run(
        tmp_path,
        markdown="Following Section 7.1 of the paper, `R_0=2000.0` was the "
                 "published raw-pixel prior; the demo uses the rescaled "
                 "value.",
        params={"R_0": {"value": 7.843}},
    )

    assert probe_narrative_vs_params(run).verdict == "pass"
