"""The gap-recovery path detects a gap claim that shadows a committed family.

Failure class gap_claim_shadows_committed_family (pdfgnn Attempt 2,
2026-08-10): the analyzer nondeterministically missed the committed
time_series_forecasting match, claimed new_top_level_needed with that exact
id, and the driver burned three pack-author dispatches on a proposal the
validator had to reject as a duplicate. The guard is deterministic and runs
before any authoring dispatch.
"""

from __future__ import annotations

from pathlib import Path

from run_pipeline import _gap_claim_shadowed_committed_family

ROOT = Path(__file__).resolve().parents[1]


def _gap_report(**overrides) -> dict:
    report = {
        "schema_version": "1.0.0",
        "decision": "new_top_level_needed",
        "proposed_new_paradigm_id": "time_series_forecasting",
        "paper_slug": "probabilistic-demand-forecasting-with-graph-neural-networks",
    }
    report.update(overrides)
    return report


def test_committed_family_id_is_flagged_as_shadowed() -> None:
    # The Attempt 2 reproducer: time_series_forecasting is a committed
    # taxonomy family, so a new-top-level claim proposing it is a shadow.
    assert _gap_claim_shadowed_committed_family(
        _gap_report(), ROOT
    ) == "time_series_forecasting"


def test_genuinely_new_family_id_is_not_flagged() -> None:
    report = _gap_report(
        proposed_new_paradigm_id="underwater_basket_weaving_forecasting"
    )
    assert _gap_claim_shadowed_committed_family(report, ROOT) is None


def test_subparadigm_claims_are_out_of_scope_for_the_guard() -> None:
    # One concrete case shaped this guard (two-cases discipline): the
    # subparadigm arm resolves its target during authoring, so the guard
    # deliberately does not judge it.
    report = _gap_report(decision="new_subparadigm_needed")
    assert _gap_claim_shadowed_committed_family(report, ROOT) is None


def test_missing_or_blank_proposed_id_is_not_flagged() -> None:
    assert _gap_claim_shadowed_committed_family(
        _gap_report(proposed_new_paradigm_id=None), ROOT
    ) is None
    assert _gap_claim_shadowed_committed_family(
        _gap_report(proposed_new_paradigm_id=""), ROOT
    ) is None
