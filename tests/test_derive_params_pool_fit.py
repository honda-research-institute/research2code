"""The smoke pool must fit the demo protocol's own arithmetic.

The 2026-06-10 GBALD validation re-run shipped initial_labeled=1000 (paper-
faithful) against the fixed pool_size=800 smoke default: the core-set
swallowed the whole pool, the unlabeled set was empty before round 1, every
select_batch call early-returned nothing, and smoke "passed" a notebook
whose acquisition stage never executed once — which also hid a selector that
crashes on every non-empty pool (the negative-stride class, see
tests/fixtures/zoo/gbald-negstride-selector/). pool_size is system-owned, so
it grows to fit the paper-faithful protocol values, never the reverse.
"""

from __future__ import annotations

from pathlib import Path

from scripts.derive_params import _add_data_setup_params, _add_paradigm_extras

REPO_ROOT = Path(__file__).resolve().parent.parent


def _spec(initial_labeled, batch_size=100, num_rounds=90):
    return {
        "critical_requirements": {
            "data_setup": {
                "initial_labeled": initial_labeled,
                "batch_size": batch_size,
                "num_rounds": num_rounds,
                "paper_section": "Section 7.4",
            },
        },
    }


def _with_classification(spec, paradigm_id):
    spec = dict(spec)
    spec["comparison"] = {
        "classification": {"id": paradigm_id},
        "pluggable_component": (
            (spec.get("comparison") or {}).get("pluggable_component") or {}
        ),
    }
    return spec


def test_pool_grows_to_fit_gbald_shaped_protocol():
    # GBALD shape: 1000 initial + 5 smoke rounds x 100 = 1500 > 800 default.
    params: dict = {}
    _add_data_setup_params(params, _spec(initial_labeled=1000), REPO_ROOT)
    pool = params["pool_size"]
    assert pool["value"] == 1500
    assert pool["source"] == "system_default"
    assert "Raised above the 800 smoke default" in pool["reasoning"]
    # The loop is reachable at the derived config.
    assert params["initial_labeled"]["value"] < pool["value"]


def test_pool_stays_at_default_for_badge_shaped_protocol():
    # BADGE shape: 100 initial + 500 budget = 600 <= 800 — unchanged, and
    # the raised-pool sentence must not appear (no-regression direction).
    params: dict = {}
    _add_data_setup_params(params, _spec(initial_labeled=100), REPO_ROOT)
    pool = params["pool_size"]
    assert pool["value"] == 800
    assert "Raised above" not in pool["reasoning"]


def test_ratio_reflects_the_grown_pool():
    params: dict = {}
    _add_data_setup_params(params, _spec(initial_labeled=1000), REPO_ROOT)
    # 1500 budget / 1500 pool — the honest ratio at the grown pool size.
    assert "≈ 1.00" in params["pool_size"]["reasoning"]


def test_budget_uses_delivered_round_count():
    # Paper num_rounds=8 is small enough to keep as-is (paper-faithful),
    # so the pool must fit 8 rounds, never the 5-round smoke constant
    # (2026-06-11 adversarial-review catch).
    params: dict = {}
    _add_data_setup_params(
        params, _spec(initial_labeled=1000, num_rounds=8), REPO_ROOT)
    assert params["num_rounds"]["value"] == 8
    assert params["pool_size"]["value"] == 1800  # 1000 + 8x100


def test_budget_covers_core_set_signature_bootstrap():
    # Core-set methods bootstrap via construct_core_set(core_set_size); a
    # signature default larger than initial_labeled is the effective
    # bootstrap and the pool must fit it (too-big pools are the safe
    # direction).
    spec = _spec(initial_labeled=100)
    spec["comparison"] = {"pluggable_component": {
        "signature": "select_batch(model, x_unlabeled, batch_size, seed, "
                     "core_set_size=1200)"}}
    params: dict = {}
    _add_data_setup_params(params, spec, REPO_ROOT)
    assert params["pool_size"]["value"] == 1700  # 1200 + 5x100


def test_served_ratio_floor_raises_pool_deterministically():
    # With a served taxonomy classification, the node's smoke_economics ratio
    # floor is a deterministic pool_size input, not advisory prose.
    params: dict = {}
    spec = _with_classification(
        _spec(initial_labeled=100, batch_size=100, num_rounds=90),
        "active_learning/batch_acquisition",
    )
    _add_data_setup_params(params, spec, REPO_ROOT)
    pool = params["pool_size"]
    # Delivered budget is 100 bootstrap + 5 smoke rounds x 100 = 600.
    # Batch-acquisition floor is max_budget_to_pool_ratio=0.05, so pool >= 12,000.
    assert pool["value"] == 12000
    assert "satisfy the taxonomy smoke_economics max_budget_to_pool_ratio floor" in (
        pool["reasoning"]
    )
    assert "≈ 0.05" in pool["reasoning"]


def test_served_ratio_floor_stacks_after_budget_fit_floor():
    params: dict = {}
    spec = _with_classification(
        _spec(initial_labeled=1000, batch_size=100, num_rounds=8),
        "active_learning/bayesian",
    )
    _add_data_setup_params(params, spec, REPO_ROOT)
    # Budget-fit floor is 1,800; bayesian ratio floor 0.4 raises to 4,500.
    assert params["pool_size"]["value"] == 4500


def test_legacy_batch_output_is_demoted_to_batch_size_smoke_value():
    params: dict = {}
    spec = _with_classification(
        _spec(initial_labeled=20, batch_size=100, num_rounds=6),
        "active_learning/bayesian",
    )
    spec["comparison"]["pluggable_component"] = {
        "signature": (
            "select_batch(model, x_unlabeled, x_labeled, batch_size, seed, "
            "batch_returns: int = 30, batch_output: int = 10) -> List[int]"
        )
    }
    _add_data_setup_params(params, spec, REPO_ROOT)
    _add_paradigm_extras(params, spec, "active_learning/bayesian")

    assert params["batch_size"]["value"] == 10
    assert params["batch_size"]["source"] == "system_default"
    assert params["batch_size"]["paper_value"] == 100
    assert "batch_output=10" in params["batch_size"]["reasoning"]
    assert "batch_returns" in params
    assert "batch_output" not in params
    assert params["pool_size"]["value"] == 800
