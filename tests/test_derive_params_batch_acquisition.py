"""Two-stage AL acquisition invariant b >= b' (batch_returns >= batch_size).

GBALD Stage-2.x halt (2026-06-30): the paper states b=300, b'=100, and the
spec recorded it ("rank b=300, select b'=100"), but `batch_returns` was derived
from the method signature's stub default of 30 (< batch_size=100), so
validate_params_output's b>=b' gate halted the run. The fix:

  1. Provenance: the paper's b is a structured field (data_setup.batch_returns),
     registered in derive_params._PAPER_VALUE_SPEC_PATHS so the deriver reads it.
  2. Reconciliation: _reconcile_batch_acquisition_invariant lifts the runtime
     value to the paper's b (or at minimum batch_size) so b >= b' holds.

These pin both halves and the mirror to the validator gate.
"""

from __future__ import annotations

from scripts.derive_params import (
    _PAPER_VALUE_SPEC_PATHS,
    _lookup_structured_paper_value,
    _reconcile_batch_acquisition_invariant,
)
from scripts.validate_params_output import _batch_acquisition_invariant_errors

AL = "active_learning/bayesian"


# --- provenance: the structured paper value is wired into the lookup map -----

def test_batch_returns_registered_as_structured_paper_value():
    assert _PAPER_VALUE_SPEC_PATHS["batch_returns"] == [
        "critical_requirements", "data_setup", "batch_returns",
    ]


def test_lookup_reads_paper_b_from_data_setup():
    spec = {"critical_requirements": {"data_setup": {"batch_returns": 300}}}
    assert _lookup_structured_paper_value("batch_returns", spec) == 300


def test_lookup_returns_none_when_batch_returns_absent():
    spec = {"critical_requirements": {"data_setup": {"batch_size": 100}}}
    assert _lookup_structured_paper_value("batch_returns", spec) is None


# --- reconciliation: the runtime value is lifted to satisfy b >= b' ----------

def test_reconcile_uses_paper_b_when_present_and_valid():
    """Paper b=300 present (>= batch_size) -> ship 300, paper-faithful."""
    params = {
        "batch_size": {"value": 100, "source": "paper"},
        "batch_returns": {"value": 30, "source": "system_default", "paper_value": 300},
    }
    _reconcile_batch_acquisition_invariant(params, AL)
    assert params["batch_returns"]["value"] == 300
    assert params["batch_returns"]["source"] == "paper"
    # the resulting params satisfy the validator gate
    assert _batch_acquisition_invariant_errors(params, AL) == []


def test_reconcile_lift_to_paper_emits_schema_valid_provenance():
    """2026-07-03 GBALD re-roll halt: the lift-to-paper branch emitted
    source=paper with only `reasoning` — no locator — and the deterministic
    validator halted the run on the deriver's own output (the one
    source=paper site in the deriver without `note`/`paper_section`). The
    lifted entry must validate against the Params model."""
    from schemas.params import Params

    params = {
        "batch_size": {"value": 100, "source": "paper",
                       "paper_section": "Section 5"},
        "batch_returns": {"value": 30, "source": "spec_default",
                          "paper_value": 300,
                          "reasoning": "Value from the signature default."},
    }
    _reconcile_batch_acquisition_invariant(params, AL)
    entry = params["batch_returns"]
    assert entry["source"] == "paper"
    assert entry.get("note"), "paper-sourced lift must carry a locator note"
    Params.model_validate({"params": params})


def test_reconcile_falls_back_to_batch_size_without_paper_b():
    """No paper b recorded -> at minimum batch_size, so the invariant holds."""
    params = {
        "batch_size": {"value": 100, "source": "paper"},
        "batch_returns": {"value": 30, "source": "spec_default"},
    }
    _reconcile_batch_acquisition_invariant(params, AL)
    assert params["batch_returns"]["value"] == 100
    assert params["batch_returns"]["source"] == "system_inferred"
    assert _batch_acquisition_invariant_errors(params, AL) == []


def test_reconcile_ignores_paper_b_below_batch_size():
    """A recorded paper b that is itself < batch_size must not be used; fall
    back to batch_size so the invariant still holds."""
    params = {
        "batch_size": {"value": 100, "source": "paper"},
        "batch_returns": {"value": 30, "source": "system_default", "paper_value": 50},
    }
    _reconcile_batch_acquisition_invariant(params, AL)
    assert params["batch_returns"]["value"] == 100
    assert params["batch_returns"]["source"] == "system_inferred"


def test_reconcile_noop_when_already_valid():
    params = {
        "batch_size": {"value": 100, "source": "paper"},
        "batch_returns": {"value": 300, "source": "paper"},
    }
    _reconcile_batch_acquisition_invariant(params, AL)
    assert params["batch_returns"]["value"] == 300
    assert params["batch_returns"]["source"] == "paper"


def test_reconcile_noop_for_single_stage_selector():
    """Single-stage selectors (e.g. BADGE) declare no batch_returns; no-op."""
    params = {"batch_size": {"value": 100, "source": "paper"}}
    _reconcile_batch_acquisition_invariant(params, AL)
    assert "batch_returns" not in params


def test_reconcile_noop_for_non_al_paradigm():
    params = {
        "batch_size": {"value": 100, "source": "paper"},
        "batch_returns": {"value": 30, "source": "spec_default"},
    }
    _reconcile_batch_acquisition_invariant(params, "motion_planning/sampling")
    assert params["batch_returns"]["value"] == 30
