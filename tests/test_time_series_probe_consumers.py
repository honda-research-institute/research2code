"""Consumer seams for the schema-2 forecasting probe kit (R2C-087)."""

from __future__ import annotations

from pathlib import Path

from probes import ProbeVerdict
from probes.time_series_forecasting import PROBE_REFS
from run_probes import run_battery
from tests.test_time_series_forecasting_probes import (
    _execution_plan,
    _write_generated_package,
)


SAMPLE_REF = "time_series_forecasting.sample_genuineness"
PATH_REF = "time_series_forecasting.path_dependence"


def _declared_context(*refs: str) -> dict[str, dict[str, str]]:
    return {
        ref: {
            "id": f"check-{index}",
            "check": f"execute {ref}",
            "why": "exact frozen test obligation",
        }
        for index, ref in enumerate(refs, start=1)
    }


def _frozen(*refs: str) -> dict:
    return {
        "schema_version": "1.0",
        "battery_version": "test-version",
        "paradigm": "time_series_forecasting",
        "declared_context": _declared_context(*refs),
        "forecasting_execution_plan": _execution_plan(autoregressive=True),
        "demo_skill_evidence": None,
    }


def _tsf_rows(report) -> list[ProbeVerdict]:
    return [row for row in report.verdicts if row.probe_id.startswith("TSF-")]


def test_full_battery_dispatches_only_exact_declared_forecasting_refs(tmp_path):
    run = _write_generated_package(tmp_path / "run", mode="ancestral")
    frozen = _frozen(SAMPLE_REF, PATH_REF)

    report = run_battery(
        run,
        ledger_path=tmp_path / "results" / "claims_ledger.json",
        frozen_gating=frozen,
    )

    rows = _tsf_rows(report)
    assert [row.probe_ref for row in rows] == [SAMPLE_REF, PATH_REF]
    assert [row.probe_id for row in rows] == ["TSF-1", "TSF-2"]
    assert [row.verdict for row in rows] == ["pass", "pass"]
    assert [row.pack_check_id for row in rows] == ["check-1", "check-2"]
    assert all(row.bound_callables == ["forecast"] for row in rows)


def test_unrelated_declared_context_does_not_activate_forecasting_kit(tmp_path):
    run = _write_generated_package(tmp_path / "run")
    frozen = {
        **_frozen(),
        "declared_context": _declared_context("claims.contribution_floor"),
    }

    report = run_battery(
        run,
        ledger_path=tmp_path / "results" / "claims_ledger.json",
        frozen_gating=frozen,
    )

    assert _tsf_rows(report) == []
    assert not (run / "method" / "probe_calls.jsonl").exists()


def test_early_and_full_seams_share_refs_bindings_and_responsibility_routing(
    tmp_path,
    monkeypatch,
):
    """A supported defect consumes the producer retry; a coverage gap does not."""
    import probes.time_series_forecasting as tsf
    import validate_method_coder_output as method_gate

    run = _write_generated_package(tmp_path / "run")
    calls: list[tuple[frozenset[str], object, object]] = []

    def fake_runner(
        run_dir: Path,
        *,
        enabled_refs,
        execution_plan=None,
        demo_skill_evidence=None,
        probe_groundings=None,
    ):
        del probe_groundings
        del run_dir
        calls.append((
            frozenset(enabled_refs),
            execution_plan,
            demo_skill_evidence,
        ))
        return [
            ProbeVerdict(
                PROBE_REFS[ref],
                "fail" if ref == SAMPLE_REF else "unprobeable",
                (
                    "generated samples disagree with the supported contract"
                    if ref == SAMPLE_REF
                    else "relational grammar is not yet supported"
                ),
                bound_callables=["forecast"],
                reason=(
                    "contract_code_disagreement"
                    if ref == SAMPLE_REF
                    else "unsupported_forecasting_runtime_grammar"
                ),
                probe_ref=ref,
            )
            for ref in PROBE_REFS
            if ref in enabled_refs
        ]

    monkeypatch.setattr(tsf, "run_time_series_forecasting_probes", fake_runner)
    report = run_battery(
        run,
        ledger_path=tmp_path / "results" / "claims_ledger.json",
        frozen_gating=_frozen(SAMPLE_REF, PATH_REF),
    )
    errors, warnings = method_gate._forecasting_behavioral_errors(
        run_dir=run,
        spec={},
    )

    expected_refs = frozenset({SAMPLE_REF, PATH_REF})
    assert [call[0] for call in calls] == [expected_refs, expected_refs]
    assert calls[0][1] is not None  # full battery replays its delivery freeze
    assert calls[1][1] is None  # producer seam resolves the same live contract
    rows = _tsf_rows(report)
    assert [row.bound_callables for row in rows] == [["forecast"], ["forecast"]]
    assert any("TSF-1" in error and SAMPLE_REF in error for error in errors)
    assert not any("TSF-2" in error for error in errors)
    assert any("TSF-2" in warning and PATH_REF in warning for warning in warnings)
