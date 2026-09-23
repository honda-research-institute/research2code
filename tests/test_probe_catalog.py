"""Verification-explainability catalog and rendering contract tests."""

from __future__ import annotations

import json
import re

import pytest

from probes import ProbeVerdict
from probes.catalog import (
    CATALOG_FRAGMENTS,
    PROBE_CATALOG,
    failure_reading,
)
from probes.universal import probe_executed_notebook
from render_run_report import _probe_summary
from run_probes import (
    _node_declared_probe_context,
    _stamp_pack_context,
)


def _write_report(run_dir, verdicts: list[dict]) -> None:
    pipeline = run_dir / ".pipeline"
    pipeline.mkdir(parents=True)
    (pipeline / "probe_report.json").write_text(
        json.dumps({"target": "fixture", "verdicts": verdicts}) + "\n",
        encoding="utf-8",
    )


def test_catalog_fragments_have_one_owner_and_complete_authored_fields():
    fragment_keys = [set(fragment) for fragment in CATALOG_FRAGMENTS]
    assert set(PROBE_CATALOG) == set().union(*fragment_keys)
    assert sum(len(keys) for keys in fragment_keys) == len(PROBE_CATALOG)

    for probe_id, record in PROBE_CATALOG.items():
        assert record["status"] in {"implemented", "designed"}, probe_id
        assert str(record["what"]).strip(), probe_id
        assert str(record["why"]).strip(), probe_id
        readings = record["on_failure"]
        assert all(
            str(readings[outcome]).strip()
            for outcome in (
                "fail",
                "flag_for_researcher",
                "warn",
                "unprobeable",
            )
        ), probe_id


def test_verdict_rejects_uncataloged_id_and_roundtrips_additive_context():
    with pytest.raises(ValueError, match="no authored catalog entry"):
        ProbeVerdict("XX-999", "fail", "unknown")

    verdict = ProbeVerdict(
        "UB-6",
        "fail",
        "loss diverged",
        evidence="notebook.ipynb: loss=nan",
        reason="output_diverged",
        pack_check_id="AL-nb-above-chance",
        pack_check="The notebook result must beat chance.",
        pack_why="A flat curve can conceal an inert acquisition loop.",
    )
    payload = verdict.to_dict()
    assert payload["reason"] == "output_diverged"
    assert payload["pack_check_id"] == "AL-nb-above-chance"
    assert payload["pack_check"].startswith("The notebook")
    assert payload["pack_why"].startswith("A flat curve")

    legacy = ProbeVerdict("UB-6", "fail", "legacy row")
    assert legacy.reason == ""
    assert legacy.pack_check_id == ""
    assert legacy.probe_ref == ""


def test_reason_specific_reading_and_legacy_verdict_fallback():
    nan_reading = failure_reading("UB-6", "fail", "output_diverged")
    generic_reading = failure_reading("UB-6", "fail")

    assert "NaN or infinite" in nan_reading
    assert "paper's stated settings" in nan_reading
    assert "concrete evidence" in generic_reading


def test_nan_emitter_records_the_specific_reason(tmp_path):
    notebook = tmp_path / "notebook.ipynb"
    notebook.write_text(json.dumps({
        "cells": [{
            "cell_type": "code",
            "source": "print('loss=nan')",
            "outputs": [{
                "output_type": "stream",
                "name": "stdout",
                "text": "loss=nan\n",
            }],
        }],
        "metadata": {},
        "nbformat": 4,
        "nbformat_minor": 5,
    }) + "\n", encoding="utf-8")

    verdicts = probe_executed_notebook(notebook)
    failed = [verdict for verdict in verdicts if verdict.verdict == "fail"]

    assert len(failed) == 1
    assert failed[0].reason == "output_diverged"
    assert failed[0].evidence == "notebook.ipynb: loss=nan"


def test_report_groups_checks_and_expands_failures_with_evidence(tmp_path):
    run = tmp_path / "run"
    _write_report(run, [
        {
            "probe_id": "UB-6",
            "verdict": "fail",
            "message": "executed loss printed nan",
            "evidence": "notebook.ipynb: loss=nan",
            "reason": "output_diverged",
            "pack_check": "The executed result must beat chance.",
            "pack_why": "A flat curve can hide an inert method.",
        },
        {
            "probe_id": "UB-6",
            "verdict": "fail",
            "message": "accuracy never beat chance",
            "evidence": "notebook.ipynb: series=[0.10, 0.10]",
            "reason": "below_chance",
        },
        {
            "probe_id": "KD-1",
            "verdict": "unprobeable",
            "message": "teacher builder missing",
        },
        {
            "probe_id": "KD-1",
            "verdict": "unprobeable",
            "message": "loss signature unavailable",
        },
        {
            "probe_id": "US-8",
            "verdict": "warn",
            "message": "notebook prose uses a stale value",
        },
    ])

    text = _probe_summary(run, delivery=None)

    assert text.count("### executed notebook outputs are sane") == 1
    assert "**What this checks.** executed notebook outputs are sane" in text
    assert "**Why it matters.**" in text
    assert "**Recorded instances.**" in text
    assert "notebook.ipynb: loss=nan" in text
    assert "NaN or infinite" in text
    assert "conservative chance baseline" in text
    assert "**Paper-family focus.** The executed result must beat chance." in text
    assert "**Why this matters for this family.** A flat curve can hide" in text
    assert "### Checks that could not run" in text
    assert "**the teacher signal influences the distillation loss (2 instances).**" in text
    assert "### Advisory checks" in text
    assert "notebook prose numbers match the params table" in text
    assert re.search(r"\b(?:US|UB|AL|MP|KD|CT)-\d+[a-z]?\b", text) is None


def test_report_reads_legacy_rows_without_reason_or_catalog_context(tmp_path):
    run = tmp_path / "legacy"
    _write_report(run, [{
        "probe_id": "UB-6",
        "verdict": "fail",
        "message": "legacy flat output",
        "evidence": "notebook.ipynb: series=[1, 1, 1]",
    }])

    text = _probe_summary(run, delivery=None)

    assert "executed notebook outputs are sane" in text
    assert "concrete evidence" in text
    assert "notebook.ipynb: series=[1, 1, 1]" in text


def test_taxonomy_gated_verdict_carries_family_authored_context():
    context = _node_declared_probe_context("active_learning")
    assert context is not None
    assert all(
        record["id"] and record["check"] and record["why"]
        for record in context.values()
    )

    verdict = _stamp_pack_context(
        ProbeVerdict("AL-6", "fail", "pool is empty"),
        "al_loop.demo_config_reachability",
        context,
    )
    payload = verdict.to_dict()

    assert payload["probe_ref"] == "al_loop.demo_config_reachability"
    assert payload["pack_check_id"] == "AL-params-demo-reachability"
    assert "initial_labeled" in payload["pack_check"].lower()
    assert "pool" in payload["pack_why"].lower()


def test_taxonomy_wrapper_refuses_to_overwrite_a_different_probe_ref():
    context = _node_declared_probe_context("active_learning")
    verdict = ProbeVerdict(
        "AL-6",
        "fail",
        "pool is empty",
        probe_ref="al_loop.acquisition_contract",
    )

    with pytest.raises(ValueError, match="attempted to stamp different ref"):
        _stamp_pack_context(
            verdict,
            "al_loop.demo_config_reachability",
            context,
        )


def test_universal_verdict_remains_unbound_without_family_wrapper():
    payload = ProbeVerdict("UB-6", "pass", "notebook executed").to_dict()

    assert payload["probe_ref"] == ""
