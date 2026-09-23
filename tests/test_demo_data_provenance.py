"""Tier-aware demo-data disclosure across report, notebook, and templates."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from demo_data_provenance import (
    PUBLIC_TIER,
    SYNTHETIC_TIER,
    UNKNOWN_TIER,
    disclosure_for,
    provenance_tier,
)
from render_run_report import render_run_report
from validate_notebook_output import (
    _bundle_consumption_errors,
    _bundle_provenance_honesty_errors,
)


def _write_provenance(run_dir: Path, manifest: dict) -> None:
    path = run_dir / "method" / "example_data" / "PROVENANCE.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(manifest), encoding="utf-8")


def _synthetic_manifest() -> dict:
    return {
        "schema_version": 1,
        "schema_id": "time_series_panel_v1",
        "tier": SYNTHETIC_TIER,
        "generator": {"name": "tsf_fixture", "version": 1, "seed": 17},
        "files": {
            "series.csv": {"sha256": "a" * 64},
            "observations.csv": {"sha256": "b" * 64},
        },
        "bundle_digest": "c" * 64,
        "entity_ids": ["series_000", "series_001"],
        "cadence": {"unit": "week", "step": 1},
        "protocol_capacity": {
            "family_fitting_prefix_steps": 24,
            "context_steps": 10,
            "validation_steps": 13,
            "test_steps": 26,
            "static_axis_floor_steps": 49,
            "usable_steps": 73,
        },
        "generation_mechanics": {
            "target": "seasonal trend plus seeded noise",
        },
        # Fixed renderer copy must win even over a contradictory note.
        "honesty_note": "REAL data from the paper",
    }


def test_tier_classifier_supports_explicit_and_legacy_public_manifests():
    assert provenance_tier({"tier": PUBLIC_TIER}) == PUBLIC_TIER
    assert provenance_tier({"tier": SYNTHETIC_TIER}) == SYNTHETIC_TIER
    assert provenance_tier({"provenance_tier": SYNTHETIC_TIER}) == SYNTHETIC_TIER
    assert provenance_tier({
        "source": {"cited_url": "https://example.test/paper-data"}
    }) == PUBLIC_TIER
    assert provenance_tier({"source": {"download_url": "https://example.test"}}) == UNKNOWN_TIER
    assert provenance_tier({
        "tier": "unsupported",
        "provenance_tier": SYNTHETIC_TIER,
    }) == UNKNOWN_TIER


def test_synthetic_disclosure_is_fixed_and_never_trusts_manifest_hype():
    disclosure = disclosure_for(_synthetic_manifest())

    assert disclosure.is_synthetic
    assert "family-owned synthetic" in disclosure.tier_label.lower()
    assert "not the paper's dataset" in disclosure.honesty_statement
    assert "not paper-comparable or benchmark evidence" in disclosure.honesty_statement
    assert "REAL data from the paper" not in disclosure.honesty_statement


def test_report_names_synthetic_tier_and_generation_details_without_origin_upgrade(
    tmp_path,
):
    _write_provenance(tmp_path, _synthetic_manifest())

    report = render_run_report(tmp_path)

    assert "## Demo data provenance" in report
    assert "Family-owned synthetic fallback" in report
    assert "not the paper's dataset" in report
    assert "mechanics only" in report
    assert "time_series_panel_v1" in report
    assert "series_000" in report
    assert "REAL data from the paper" not in report


def test_report_preserves_public_source_disclosure_for_explicit_and_legacy_tiers(
    tmp_path,
):
    for tier in (PUBLIC_TIER, None):
        run_dir = tmp_path / (tier or "legacy")
        manifest = {
            "source": {"cited_url": "https://example.test/paper-data"},
            "honesty_note": "Demo-scale numbers are not paper benchmarks.",
        }
        if tier:
            manifest["tier"] = tier
        _write_provenance(run_dir, manifest)

        report = render_run_report(run_dir)

        assert "Paper-cited public data" in report
        assert "https://example.test/paper-data" in report
        assert "Demo-scale numbers are not paper benchmarks." in report


def test_report_explains_when_public_and_family_fallback_both_refuse(tmp_path):
    pipeline = tmp_path / ".pipeline"
    pipeline.mkdir(parents=True)
    (pipeline / "dataset_acquisition.json").write_text(json.dumps({
        "status": "fallback_refused",
        "offline": True,
        "attempts": [{
            "host": "kaggle_dataset",
            "cited_url": "https://example.test/paper-data",
            "status": "refused_offline",
            "reason": "network disabled",
            "http_code": None,
        }],
        "offline_fallback": {
            "status": "refused",
            "code": "offline_fallback_family_unsupported",
            "reason": "the family has no declared offline contract",
        },
    }), encoding="utf-8")

    report = render_run_report(tmp_path)

    assert "## Demo data provenance" in report
    assert "No provisioned demo data" in report
    assert "fallback_refused" in report
    assert "offline_fallback_family_unsupported" in report
    assert "the family has no declared offline contract" in report
    assert "refused_offline" in report
    assert "No demo-data results should be interpreted" in report


def test_report_does_not_call_missing_generated_manifest_a_refusal(tmp_path):
    pipeline = tmp_path / ".pipeline"
    pipeline.mkdir(parents=True)
    (pipeline / "dataset_acquisition.json").write_text(json.dumps({
        "status": "fallback_generated",
        "offline_fallback": {
            "status": "generated",
            "code": "offline_fallback_generated",
            "reason": "recorded generation",
        },
    }), encoding="utf-8")

    report = render_run_report(tmp_path)

    assert "Offline fallback completion" in report
    assert "completion marker is missing" in report
    assert "recorded generation as incomplete" in report
    assert "Offline fallback refusal" not in report
    assert "Neither a paper-cited public bundle" not in report
    assert "unknown_refusal" not in report


def test_synthetic_notebook_requires_tier_and_evidence_limit(tmp_path):
    _write_provenance(tmp_path, _synthetic_manifest())

    errors = _bundle_provenance_honesty_errors(
        ["# Data\nBundled demo data — REAL, from the paper's own cited source."],
        tmp_path,
    )

    assert any("real or from-paper" in error for error in errors)
    assert any("family-owned synthetic fallback" in error for error in errors)


def test_synthetic_notebook_passes_explicit_mechanics_only_disclosure(tmp_path):
    _write_provenance(tmp_path, _synthetic_manifest())

    errors = _bundle_provenance_honesty_errors([
        "# Data provenance\n"
        "This demo uses the family-owned synthetic fallback. It is not the "
        "paper's dataset; its numbers demonstrate mechanics only and are not "
        "benchmark evidence."
    ], tmp_path)

    assert errors == []


def test_synthetic_notebook_origin_only_disclosure_does_not_state_evidence_limit(
    tmp_path,
):
    _write_provenance(tmp_path, _synthetic_manifest())

    errors = _bundle_provenance_honesty_errors([
        "This demo uses the family-owned synthetic fallback and is not the "
        "paper's dataset."
    ], tmp_path)

    assert any("not paper-comparable or benchmark evidence" in error
               for error in errors)


@pytest.mark.parametrize(
    "contradiction",
    [
        "This fixture is the paper's dataset.",
        "These data are actual paper data.",
        "This is not a toy; this fixture is real paper data.",
    ],
)
def test_synthetic_notebook_refuses_direct_from_paper_contradictions(
    tmp_path, contradiction,
):
    _write_provenance(tmp_path, _synthetic_manifest())
    prose = (
        "This demo uses the family-owned synthetic fallback. Its numbers "
        "demonstrate mechanics only and are not benchmark evidence. "
        f"{contradiction}"
    )

    errors = _bundle_provenance_honesty_errors([prose], tmp_path)

    assert any("real or from-paper" in error for error in errors)


def test_unrelated_negation_does_not_hide_a_positive_origin_claim(tmp_path):
    _write_provenance(tmp_path, _synthetic_manifest())
    prose = (
        "This demo uses the family-owned synthetic fallback. Its numbers "
        "demonstrate mechanics only. We do not omit provenance, and this "
        "fixture is real paper data."
    )

    errors = _bundle_provenance_honesty_errors([prose], tmp_path)

    assert any("real or from-paper" in error for error in errors)


@pytest.mark.parametrize(
    "claim",
    [
        "These results are benchmark evidence.",
        "These numbers match the paper benchmark.",
        "The numbers are not paper-comparable but are benchmark evidence.",
    ],
)
def test_synthetic_notebook_refuses_positive_benchmark_claims_after_disclosure(
    tmp_path, claim,
):
    _write_provenance(tmp_path, _synthetic_manifest())
    prose = (
        "This demo uses the family-owned synthetic fallback. Its numbers "
        "demonstrate mechanics only and are not benchmark evidence. "
        f"{claim}"
    )

    errors = _bundle_provenance_honesty_errors([prose], tmp_path)

    assert any("benchmark evidence" in error for error in errors)


@pytest.mark.parametrize(
    "prose",
    [
        (
            "This demo uses the family-owned synthetic fallback. Results are "
            "not discussed; benchmark evidence is outside scope."
        ),
        (
            "This demo uses the family-owned synthetic fallback. Results are "
            "not shown, but benchmark evidence could be collected later."
        ),
    ],
)
def test_synthetic_notebook_requires_one_clause_to_bind_results_to_limit(
    tmp_path, prose,
):
    _write_provenance(tmp_path, _synthetic_manifest())

    errors = _bundle_provenance_honesty_errors([prose], tmp_path)

    assert any("not paper-comparable or benchmark evidence" in error
               for error in errors)


def test_synthetic_notebook_accepts_result_scoped_never_limit(tmp_path):
    _write_provenance(tmp_path, _synthetic_manifest())
    prose = (
        "This demo uses the family-owned synthetic fallback. Its results can "
        "never be interpreted as benchmark evidence."
    )

    assert _bundle_provenance_honesty_errors([prose], tmp_path) == []


@pytest.mark.parametrize(
    "claim",
    [
        "We do not deny this fixture is real paper data.",
        "This fixture is genuinely real paper data.",
        "These results are strong benchmark evidence.",
        "These numbers agree with the paper's benchmark.",
    ],
)
def test_synthetic_notebook_refuses_adversarial_evidence_laundering(
    tmp_path, claim,
):
    _write_provenance(tmp_path, _synthetic_manifest())
    prose = (
        "This demo uses the family-owned synthetic fallback. Its results are "
        "not paper-comparable. " + claim
    )

    errors = _bundle_provenance_honesty_errors([prose], tmp_path)

    assert any("positive evidence claim" in error for error in errors)


@pytest.mark.parametrize(
    ("prose", "expected"),
    [
        (
            "This demo is not a family-owned synthetic fallback. Its results "
            "are not paper-comparable.",
            "family-owned synthetic fallback",
        ),
        (
            "This demo uses the family-owned synthetic fallback. Its results "
            "are not paper-comparable. The dataset comes from the paper.",
            "positive evidence claim",
        ),
        (
            "This demo uses the family-owned synthetic fallback. Its results "
            "are not paper-comparable. The benchmark evidence here is strong.",
            "positive evidence claim",
        ),
        (
            "This demo uses the family-owned synthetic fallback. Its results "
            "are not paper-comparable. These results confirm the paper's "
            "benchmark.",
            "positive evidence claim",
        ),
    ],
)
def test_synthetic_notebook_refuses_negated_tier_and_grammar_variants(
    tmp_path, prose, expected,
):
    _write_provenance(tmp_path, _synthetic_manifest())

    errors = _bundle_provenance_honesty_errors([prose], tmp_path)

    assert any(expected in error for error in errors)


def test_public_notebook_behavior_remains_unrestricted_by_synthetic_gate(tmp_path):
    _write_provenance(tmp_path, {
        "tier": PUBLIC_TIER,
        "source": {"cited_url": "https://example.test/paper-data"},
    })

    assert _bundle_provenance_honesty_errors([], tmp_path) == []


def test_synthetic_bundle_is_still_required_loader_input(tmp_path):
    _write_provenance(tmp_path, _synthetic_manifest())

    errors = _bundle_consumption_errors(
        ["from method import load_data  # imported but never called"],
        tmp_path,
    )

    assert errors and "family-owned synthetic fallback" in errors[0]
    assert "bundled real tables" not in errors[0]
    assert "notebook_data_binding" in errors[0]


def test_forecasting_templates_are_tier_aware_and_make_no_real_axis_claim():
    root = Path(__file__).resolve().parents[1]
    method_readme = (
        root / "paradigms/time_series_forecasting/templates/method/README.md.template"
    ).read_text(encoding="utf-8")
    data_readme = (
        root
        / "paradigms/time_series_forecasting/templates/method/example_data/README.md.template"
    ).read_text(encoding="utf-8")

    assert "over the real time axis" not in method_readme
    assert SYNTHETIC_TIER in method_readme
    assert PUBLIC_TIER in data_readme
    assert SYNTHETIC_TIER in data_readme
    assert "not paper-comparable or benchmark" in data_readme
