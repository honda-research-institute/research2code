"""R2C-055: a paper-stated parameter needs a carrier its own path reads.

The pdfgnn night roll of 2026-08-05 halted at stage 2x because
`params.json` had no `similarity_cutoff` while the run's own provisional
pack required it in (0, 1) and the paper states 0.95 in Appendix A
Table 5. Three independent gaps produced that one halt:

  1. The gap-family analyzer contract names
     `critical_requirements.scale_dependent_hyperparameters` as the
     mandatory carrier for a paper's scale values, and `derive_params`
     read that lane only inside the motion-planning branch. A fully
     compliant analyzer's value would have been dropped too.
  2. `param_glossary` had the parameter with a verbatim meaning quote and
     no value, because the glossary schema had no value field. Its
     documented job is stamping `paper_says` onto params that already
     exist, so it can annotate and never create.
  3. The pack validator already WARNED at install time that a
     stage_2x_params check with no `params_derivation` block would halt
     the 2.x review on entries no producer can emit. The warning
     predicted this halt verbatim and did not block.

maintainer-approved both arms on 2026-08-05: the deterministic gate and the
deriver source. These tests pin all three pieces in both directions.
"""

from __future__ import annotations

import json
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from schemas.method_spec import MethodSpec, ParamGlossaryEntry
from scripts import derive_params as derive_params_module
from scripts.derive_params import (
    _add_glossary_value_params,
    _add_scale_dependent_params,
    _consume_parameter_carriers,
    _glossary_param_name,
    derive,
)
from scripts.validate_paradigm_proposal import (
    _check_params_derivation,
    _named_params_in_check,
)
from scripts.validate_method_spec import cross_check_parameter_carrier_exclusivity
from scripts.validate_method_spec import cross_check_parameter_carrier_consumability
from tests.test_method_spec_schema import _minimal_valid_spec


def _spec(*, lane=None, glossary=None) -> dict:
    return {"critical_requirements": {
        "scale_dependent_hyperparameters": lane or [],
        "param_glossary": glossary or [],
    }}


# ---------------------------------------------------------------------------
# Piece 1: the scale-dependent lane reader, now shared with the gap path.
# ---------------------------------------------------------------------------


def test_scale_dependent_lane_creates_paper_sourced_params():
    params: dict = {}
    added = _add_scale_dependent_params(params, _spec(lane=[
        {"name": "similarity_cutoff", "paper_value": 0.95,
         "paper_section": "Appendix A, Table 5",
         "description": "Cosine-similarity threshold for an article edge."},
    ]))
    assert added == ["similarity_cutoff"]
    entry = params["similarity_cutoff"]
    assert entry["value"] == 0.95
    assert entry["source"] == "paper"
    assert entry["paper_section"] == "Appendix A, Table 5"


def test_scale_dependent_lane_skips_valueless_and_malformed_entries():
    """A formula-only or runtime-computed entry is not a tunable param."""
    params: dict = {}
    added = _add_scale_dependent_params(params, _spec(lane=[
        {"name": "clearance_obs", "paper_value": None, "formula": "2*sqrt(d)"},
        {"name": "", "paper_value": 1.0},
        "not a mapping",
    ]))
    assert added == []
    assert params == {}


def test_scale_dependent_lane_can_top_up_without_clobbering():
    """The gap path tops up an existing param set, so it must never
    overwrite a value another reader already produced."""
    params = {"learning_rate": {"value": 0.005, "source": "paper"}}
    added = _add_scale_dependent_params(params, _spec(lane=[
        {"name": "learning_rate", "paper_value": 0.01},
        {"name": "similarity_cutoff", "paper_value": 0.95},
    ]), overwrite=False)
    assert added == ["similarity_cutoff"]
    assert params["learning_rate"]["value"] == 0.005


def test_motion_planning_still_overwrites_by_default():
    """The extraction preserved the motion-planning branch's behavior."""
    params = {"d_safe": {"value": 0.1, "source": "spec_default"}}
    _add_scale_dependent_params(params, _spec(lane=[
        {"name": "d_safe", "paper_value": 0.375},
    ]))
    assert params["d_safe"]["value"] == 0.375
    assert params["d_safe"]["source"] == "paper"


def test_typed_and_legacy_calibration_contexts_derive_byte_identically():
    """R2C-091 types observer choice without changing parameter derivation."""
    base = _minimal_valid_spec(paradigm_id="motion_planning")
    entry = {
        "name": "d_safe",
        "paper_value": 0.375,
        "description": "Safety distance calibrated to the observed features.",
        "paper_section": "Section 4",
    }
    legacy = deepcopy(base)
    legacy["schema_version"] = "1.12.0"
    legacy["critical_requirements"]["scale_dependent_hyperparameters"] = [
        {**entry, "assumes_data_scale": "raw_pixel_unnormalized"},
    ]
    typed = deepcopy(base)
    typed["schema_version"] = "1.13.0"
    typed["critical_requirements"]["scale_dependent_hyperparameters"] = [
        {**entry, "calibration_context": {
            "kind": "feature_magnitude",
            "scale": "raw_pixel_unnormalized",
        }},
    ]

    legacy_bytes = json.dumps(
        derive(legacy), sort_keys=True, separators=(",", ":"),
    ).encode()
    typed_bytes = json.dumps(
        derive(typed), sort_keys=True, separators=(",", ":"),
    ).encode()
    assert typed_bytes == legacy_bytes


# ---------------------------------------------------------------------------
# Piece 2: the glossary value field as the slip-tolerant second source.
# ---------------------------------------------------------------------------


def test_glossary_entry_accepts_an_optional_paper_value():
    with_value = ParamGlossaryEntry(
        name="similarity cutoff", aliases=["similarity_cutoff"],
        meaning_quote="a specified threshold", paper_section="Section 3.2",
        paper_value=0.95)
    assert with_value.paper_value == 0.95
    # Absent stays the honest common case.
    without = ParamGlossaryEntry(
        name="mu", aliases=[], meaning_quote="the mean",
        paper_section="Section 3.3")
    assert without.paper_value is None


def test_glossary_value_creates_param_under_the_derivation_alias():
    """The pdfgnn shape: the paper's own name is not an identifier and the
    alias is the runtime name the pack check requires."""
    params: dict = {}
    added = _add_glossary_value_params(params, _spec(glossary=[
        {"name": "similarity cutoff",
         "aliases": ["similarity_cutoff", "edge_threshold"],
         "meaning_quote": "a specified threshold",
         "paper_section": "Appendix A, Table 5",
         "paper_value": 0.95},
    ]))
    assert added == ["similarity_cutoff"]
    assert params["similarity_cutoff"]["value"] == 0.95
    assert params["similarity_cutoff"]["source"] == "paper"
    assert params["similarity_cutoff"]["paper_section"] == "Appendix A, Table 5"


def test_glossary_value_never_clobbers_a_typed_lane_result():
    """The legacy helper never clobbers before strict validation runs.

    Fresh authoring rejects this duplicate-carrier shape below; retaining the
    helper's non-clobbering behavior keeps old readable artifacts deterministic.
    """
    params = {"similarity_cutoff": {"value": 0.9, "source": "paper",
                                    "note": "from the typed lane"}}
    added = _add_glossary_value_params(params, _spec(glossary=[
        {"name": "similarity cutoff", "aliases": ["similarity_cutoff"],
         "meaning_quote": "q", "paper_section": "S3", "paper_value": 0.95},
    ]))
    assert added == []
    assert params["similarity_cutoff"]["value"] == 0.9


def test_glossary_value_skips_when_any_alias_already_has_a_param():
    """A param already derived under a DIFFERENT alias of the same entry
    must not gain a duplicate under a second name."""
    params = {"context_length": {"value": 10, "source": "paper"}}
    added = _add_glossary_value_params(params, _spec(glossary=[
        {"name": "P", "aliases": ["context_length", "num_lags"],
         "meaning_quote": "P demand lags", "paper_section": "Section 3.2",
         "paper_value": 12},
    ]))
    assert added == []
    assert set(params) == {"context_length"}


def test_backstop_param_yields_when_a_later_producer_makes_the_same_value():
    """The night3 duplicate: the backstop created the entry's alias, a
    later primary pass (the pluggable-signature extras) created the
    entry's name with the same value, and params.json carried 0.95
    twice. The backstop copy yields; the primary lane's name stays."""
    from scripts.derive_params import _drop_shadowed_glossary_params

    spec = _spec(glossary=[
        {"name": "similarity_threshold", "aliases": ["similarity_cutoff"],
         "meaning_quote": "a specified threshold",
         "paper_section": "Section 3.2", "paper_value": 0.95},
    ])
    params = {
        "similarity_cutoff": {"value": 0.95, "source": "paper",
                              "paper_section": "Section 3.2"},
        "similarity_threshold": {"value": 0.95, "source": "paper",
                                 "paper_section": "Appendix A, Table 5"},
    }
    dropped = _drop_shadowed_glossary_params(
        params, spec, ["similarity_cutoff"])
    assert dropped == ["similarity_cutoff"]
    assert set(params) == {"similarity_threshold"}


def test_backstop_param_stays_when_the_values_disagree():
    """A value disagreement is a real conflict and must stay visible on
    both entries, never resolved by silently dropping one."""
    from scripts.derive_params import _drop_shadowed_glossary_params

    spec = _spec(glossary=[
        {"name": "similarity_threshold", "aliases": ["similarity_cutoff"],
         "meaning_quote": "q", "paper_section": "S3", "paper_value": 0.95},
    ])
    params = {
        "similarity_cutoff": {"value": 0.95, "source": "paper"},
        "similarity_threshold": {"value": 0.9, "source": "paper"},
    }
    dropped = _drop_shadowed_glossary_params(
        params, spec, ["similarity_cutoff"])
    assert dropped == []
    assert set(params) == {"similarity_cutoff", "similarity_threshold"}


def test_backstop_param_stays_when_it_is_the_only_carrier():
    from scripts.derive_params import _drop_shadowed_glossary_params

    spec = _spec(glossary=[
        {"name": "similarity_threshold", "aliases": ["similarity_cutoff"],
         "meaning_quote": "q", "paper_section": "S3", "paper_value": 0.95},
    ])
    params = {"similarity_cutoff": {"value": 0.95, "source": "paper"}}
    dropped = _drop_shadowed_glossary_params(
        params, spec, ["similarity_cutoff"])
    assert dropped == []
    assert set(params) == {"similarity_cutoff"}


def test_glossary_value_absent_creates_nothing():
    params: dict = {}
    assert _add_glossary_value_params(params, _spec(glossary=[
        {"name": "mu", "aliases": [], "meaning_quote": "the mean",
         "paper_section": "S3"},
    ])) == []
    assert params == {}


def test_glossary_integral_value_stays_an_int():
    params: dict = {}
    _add_glossary_value_params(params, _spec(glossary=[
        {"name": "num_lags", "aliases": [], "meaning_quote": "q",
         "paper_section": "S3", "paper_value": 10.0},
    ]))
    assert params["num_lags"]["value"] == 10
    assert isinstance(params["num_lags"]["value"], int)


# ---------------------------------------------------------------------------
# R2C-083: promotion must not switch off valid parameter carriers.
# ---------------------------------------------------------------------------


def _carrier_consumer_spec(
    *,
    paradigm_id: str = "time_series_forecasting",
    glossary: list[dict] | None = None,
    lane: list[dict] | None = None,
) -> dict:
    return {
        "comparison": {
            "classification": {"id": paradigm_id},
            "pluggable_component": {
                "name": "forecast",
                "signature": (
                    "forecast(model, history, graph, seed, "
                    "num_samples: int = 100) -> ForecastResult"
                ),
            },
        },
        "critical_requirements": {
            "param_glossary": glossary or [],
            "scale_dependent_hyperparameters": lane or [],
        },
    }


def test_committed_family_consumes_a_glossary_only_paper_value():
    """The R2C-064-valid reduction of the committed pdfgnn failure."""
    spec = _carrier_consumer_spec(glossary=[{
        "name": "Similarity cutoff",
        "aliases": ["similarity_threshold"],
        "meaning_quote": "an edge exists when similarity exceeds a threshold",
        "paper_section": "Section 3.2",
        "paper_value": 0.95,
    }])

    params = derive(spec)["params"]

    assert params["similarity_threshold"]["value"] == 0.95
    assert params["similarity_threshold"]["source"] == "paper"
    assert "similarity_cutoff" not in params
    assert sum(name in params for name in (
        "Similarity cutoff", "similarity_threshold", "similarity_cutoff"
    )) == 1


@pytest.mark.parametrize("shape", ["signature_collision", "glossary_only"])
def test_provisional_carrier_outputs_stay_byte_identical(
    shape, tmp_path, monkeypatch,
):
    """Reduced `_6` and `_8` controls after their archived packs were shadowed.

    Both historical provisional paths had already emitted one paper-sourced
    ``similarity_cutoff``.  The final generic pass must be a byte-for-byte no-op
    whether the provisional signature also named the cutoff (`_6`) or did not
    (`_8`).
    """
    paradigm_id = f"test_only_provisional_graph_forecasting_{shape}"
    spec = _carrier_consumer_spec(
        paradigm_id=paradigm_id,
        glossary=[{
            "name": "similarity cutoff",
            "aliases": ["similarity_cutoff"],
            "meaning_quote": "a specified threshold",
            "paper_section": "Section 3.2",
            "paper_value": 0.95,
        }],
    )
    if shape == "signature_collision":
        spec["comparison"]["pluggable_component"]["signature"] = (
            "forecast(model, history, graph, seed, *, "
            "similarity_cutoff: float = 0.95) -> ForecastResult"
        )
    run_dir = tmp_path / shape
    pack_dir = run_dir / ".pipeline" / "provisional_packs" / "r2c083"
    pack_dir.mkdir(parents=True)
    pack = {
        "schema_version": "1.0",
        "status": "provisional",
        "legacy_paradigm": paradigm_id,
        "extends": None,
        "taxonomy_id": f"PROVISIONAL/{paradigm_id}",
        "fingerprint": {
            "what_it_is": "Graph-conditioned probabilistic forecasting.",
            "not_this": [],
        },
        "scaffold_hints": {
            "interface_hint": spec["comparison"]["pluggable_component"][
                "signature"
            ],
        },
        "semantic_checks": [],
        "smoke_bugs": [],
        "params_derivation": {
            "similarity_cutoff": {
                "kind": "config_path",
                "reasoning": "Cosine threshold used by graph construction.",
            },
        },
    }
    (pack_dir / "pack.yaml").write_text(
        yaml.safe_dump(pack), encoding="utf-8",
    )

    consumer = derive_params_module._consume_parameter_carriers
    monkeypatch.setattr(
        derive_params_module,
        "_consume_parameter_carriers",
        lambda *_args, **_kwargs: {},
    )
    before = derive(spec, run_dir=run_dir)
    monkeypatch.setattr(
        derive_params_module, "_consume_parameter_carriers", consumer,
    )
    after = derive(spec, run_dir=run_dir)

    assert json.dumps(after, sort_keys=True, separators=(",", ":")) == (
        json.dumps(before, sort_keys=True, separators=(",", ":"))
    )
    assert after["params"]["similarity_cutoff"]["value"] == 0.95
    assert after["params"]["similarity_cutoff"]["source"] == "paper"


def test_bayesian_scaled_runtime_and_paper_truth_stay_byte_identical():
    params = {
        "R_0": {
            "value": 7.8431372549019605,
            "source": "system_default",
            "paper_value": 2000.0,
            "reasoning": "Paper radius rescaled from raw pixels to [0, 1].",
        }
    }
    spec = _carrier_consumer_spec(lane=[{
        "name": "R_0",
        "paper_value": 2000.0,
        "paper_section": "Section 4.2",
        "description": "Radius calibrated for raw pixels.",
    }])
    before = json.dumps(params, sort_keys=True, separators=(",", ":"))

    _consume_parameter_carriers(params, spec, {})

    assert json.dumps(params, sort_keys=True, separators=(",", ":")) == before


def test_stronger_existing_paper_truth_outranks_a_stale_carrier():
    params = {
        "forecast_horizon": {
            "value": 4,
            "source": "system_inferred",
            "paper_value": 12,
            "reasoning": "Typed protocol authority resolves twelve steps.",
        }
    }
    spec = _carrier_consumer_spec(glossary=[{
        "name": "forecast_horizon",
        "aliases": [],
        "meaning_quote": "A legacy table labels K as twenty-six steps.",
        "paper_section": "Legacy fixture",
        "paper_value": 26,
    }])
    before = json.dumps(params, sort_keys=True, separators=(",", ":"))

    _consume_parameter_carriers(params, spec, {})

    assert json.dumps(params, sort_keys=True, separators=(",", ":")) == before


def test_explicit_suppression_outranks_a_populated_carrier():
    spec = _carrier_consumer_spec(glossary=[{
        "name": "hidden_dim",
        "aliases": [],
        "meaning_quote": "the decoder hidden width",
        "paper_section": "Appendix A",
        "paper_value": 128,
    }])
    params: dict = {}
    declarations = {
        "hidden_dim": {
            "kind": "suppress",
            "reason": "This family owns decoder widths paper by paper.",
        }
    }

    suppressed = _consume_parameter_carriers(params, spec, declarations)

    assert params == {}
    assert suppressed == {
        "hidden_dim": "This family owns decoder widths paper by paper."
    }


def test_suppression_matches_an_exact_declared_alias():
    params: dict = {}
    spec = _carrier_consumer_spec(glossary=[{
        "name": "decoder width",
        "aliases": ["hidden_dim"],
        "meaning_quote": "the decoder hidden width",
        "paper_section": "Appendix A",
        "paper_value": 128,
    }])

    suppressed = _consume_parameter_carriers(params, spec, {
        "hidden_dim": {
            "kind": "suppress",
            "reason": "Family-specific decoder widths are not generic MLP width.",
        },
    })

    assert params == {}
    assert suppressed == {
        "hidden_dim": "Family-specific decoder widths are not generic MLP width."
    }


@pytest.mark.parametrize("source", ["spec_default", "system_inferred"])
def test_stronger_runtime_entry_keeps_authority_and_distinct_paper_truth(source):
    params = {
        "similarity_threshold": {
            "value": 0.9,
            "source": source,
            "reasoning": "Runtime authority selected this demo value.",
        }
    }
    spec = _carrier_consumer_spec(glossary=[{
        "name": "Similarity cutoff",
        "aliases": ["similarity_threshold"],
        "meaning_quote": "a specified threshold",
        "paper_section": "Section 3.2",
        "paper_value": 0.95,
    }])

    _consume_parameter_carriers(params, spec, {})

    assert params["similarity_threshold"]["value"] == 0.9
    assert params["similarity_threshold"]["source"] == source
    assert params["similarity_threshold"]["paper_value"] == 0.95


def test_one_carrier_with_two_aliases_creates_one_deterministic_entry():
    params: dict = {}
    spec = _carrier_consumer_spec(glossary=[{
        "name": "Similarity cutoff",
        "aliases": ["similarity_threshold", "similarity_cutoff"],
        "meaning_quote": "a specified threshold",
        "paper_section": "Section 3.2",
        "paper_value": 0.95,
    }])

    _consume_parameter_carriers(params, spec, {})

    assert list(params) == ["similarity_threshold"]
    assert params["similarity_threshold"]["value"] == 0.95


def test_generic_consumer_uses_exact_declared_identity_only():
    params = {
        "similarity_threshold": {
            "value": 0.9,
            "source": "system_inferred",
            "reasoning": "Unrelated exact runtime name.",
        }
    }
    spec = _carrier_consumer_spec(glossary=[{
        "name": "similarity threshold",
        "aliases": ["edge_threshold"],
        "meaning_quote": "a specified threshold",
        "paper_section": "Section 3.2",
        "paper_value": 0.95,
    }])

    _consume_parameter_carriers(params, spec, {})

    assert params["similarity_threshold"]["value"] == 0.9
    assert params["edge_threshold"]["value"] == 0.95


def test_strict_validation_rejects_a_populated_unnameable_carrier():
    spec = _carrier_spec(
        lane=[],
        glossary=[_glossary_carrier(
            "similarity cutoff", aliases=["edge threshold"], paper_value=0.95,
        )],
    )

    errors = cross_check_parameter_carrier_consumability(spec)

    assert len(errors) == 1
    assert "paper_value_carrier_unnameable" in errors[0]
    assert "critical_requirements.param_glossary[0]" in errors[0]
    assert "valid Python identifier" in errors[0]


def test_unnameable_refusal_is_wired_only_after_strict_exclusivity(tmp_path):
    raw_spec = _minimal_valid_spec(paradigm_id="motion_planning")
    raw_spec["critical_requirements"]["param_glossary"] = [
        _glossary_carrier(
            "similarity cutoff",
            aliases=["edge threshold"],
            paper_value=0.95,
        )
    ]
    spec_path = tmp_path / "method_spec.json"
    spec_path.write_text(json.dumps(raw_spec), encoding="utf-8")
    validator = Path("scripts/validate_method_spec.py")

    ordinary = subprocess.run(
        [sys.executable, str(validator), str(spec_path)],
        check=False,
        capture_output=True,
        text=True,
    )
    strict = subprocess.run(
        [sys.executable, str(validator), str(spec_path), "--strict"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert ordinary.returncode == 0, ordinary.stderr
    assert strict.returncode == 2
    assert "parameter-carrier exclusivity cross-check passed" in strict.stdout
    assert "parameter-carrier consumability cross-check failed" in strict.stderr
    assert "paper_value_carrier_unnameable" in strict.stderr


@pytest.mark.parametrize("entry,expected", [
    ({"name": "similarity cutoff", "aliases": ["similarity_cutoff"]},
     "similarity_cutoff"),
    ({"name": "P", "aliases": ["context_length"]}, "context_length"),
    ({"name": "seed", "aliases": []}, "seed"),
    ({"name": "s^2", "aliases": ["variance"]}, "variance"),
    ({"name": "s^2", "aliases": []}, None),
    ({"name": "mu", "aliases": ["not an identifier"]}, "mu"),
])
def test_glossary_param_name_prefers_the_derivation_alias(entry, expected):
    assert _glossary_param_name(entry) == expected


# ---------------------------------------------------------------------------
# R2C-064: one blessed paper-value carrier per parameter.
# ---------------------------------------------------------------------------


def _scale_carrier(
    name: str,
    *,
    paper_value: float | None,
    formula: str | None = None,
    assumes_data_scale: str = "raw_pixel_unnormalized",
) -> dict:
    return {
        "name": name,
        "paper_value": paper_value,
        "formula": formula,
        "assumes_data_scale": assumes_data_scale,
        "description": f"{name} is calibrated to the input data scale.",
        "paper_section": "Section 4",
    }


def _glossary_carrier(
    name: str,
    *,
    aliases: list[str] | None = None,
    paper_value: float | None,
) -> dict:
    return {
        "name": name,
        "aliases": aliases or [],
        "meaning_quote": f"The paper defines {name}.",
        "paper_section": "Section 4",
        "paper_value": paper_value,
    }


def _carrier_spec(*, lane: list[dict], glossary: list[dict]) -> MethodSpec:
    spec = _minimal_valid_spec(paradigm_id="motion_planning")
    spec["critical_requirements"]["scale_dependent_hyperparameters"] = lane
    spec["critical_requirements"]["param_glossary"] = glossary
    return MethodSpec.model_validate(spec)


def test_exclusivity_rejects_exact_name_with_the_same_value():
    spec = _carrier_spec(
        lane=[_scale_carrier("similarity_cutoff", paper_value=0.95)],
        glossary=[_glossary_carrier(
            "similarity_cutoff", paper_value=0.95
        )],
    )

    errors = cross_check_parameter_carrier_exclusivity(spec)

    assert len(errors) == 1
    message = errors[0]
    assert "paper_value_carrier_duplication" in message
    assert "scale_dependent_hyperparameters[0]" in message
    assert "param_glossary[0]" in message
    assert "stripped labels=['similarity_cutoff']" in message
    assert "paper_value=0.95" in message
    assert "formula=None" in message
    assert "scale-free constant belongs in the glossary only" in message
    assert "paper_value=null" in message


def test_exclusivity_rejects_a_declared_alias_match():
    spec = _carrier_spec(
        lane=[_scale_carrier("similarity_cutoff", paper_value=0.95)],
        glossary=[_glossary_carrier(
            "similarity threshold",
            aliases=["similarity_cutoff"],
            paper_value=0.95,
        )],
    )

    errors = cross_check_parameter_carrier_exclusivity(spec)

    assert len(errors) == 1
    assert (
        "stripped labels=['similarity threshold', 'similarity_cutoff']"
        in errors[0]
    )


def test_exclusivity_rejects_different_values_for_the_same_parameter():
    spec = _carrier_spec(
        lane=[_scale_carrier("similarity_cutoff", paper_value=0.90)],
        glossary=[_glossary_carrier(
            "similarity_cutoff", paper_value=0.95
        )],
    )

    errors = cross_check_parameter_carrier_exclusivity(spec)

    assert len(errors) == 1
    assert "paper_value=0.9" in errors[0]
    assert "param_glossary[0].paper_value=0.95" in errors[0]


def test_exclusivity_rejects_a_formula_lane_beside_a_glossary_value():
    spec = _carrier_spec(
        lane=[_scale_carrier(
            "d_safe", paper_value=None, formula="V_max / 4"
        )],
        glossary=[_glossary_carrier("d_safe", paper_value=0.375)],
    )

    errors = cross_check_parameter_carrier_exclusivity(spec)

    assert len(errors) == 1
    assert "paper_value=None" in errors[0]
    assert "formula='V_max / 4'" in errors[0]
    assert "param_glossary[0].paper_value=0.375" in errors[0]


@pytest.mark.parametrize(
    ("name", "value", "assumes_data_scale"),
    [
        ("R_0", 2000.0, "raw_pixel_unnormalized"),
        ("d_safe", 0.375, "metres"),
    ],
)
def test_scale_lane_with_meaning_only_glossary_is_valid(
    name: str,
    value: float,
    assumes_data_scale: str,
):
    """The real GBALD and PDWA shapes retain their typed lane semantics."""
    spec = _carrier_spec(
        lane=[_scale_carrier(
            name,
            paper_value=value,
            assumes_data_scale=assumes_data_scale,
        )],
        glossary=[_glossary_carrier(name, paper_value=None)],
    )

    assert cross_check_parameter_carrier_exclusivity(spec) == []


def test_scale_free_glossary_only_control_is_valid():
    spec = _carrier_spec(
        lane=[],
        glossary=[_glossary_carrier(
            "similarity_cutoff", paper_value=0.95
        )],
    )

    assert cross_check_parameter_carrier_exclusivity(spec) == []


def test_scale_dependent_glossary_only_backstop_remains_valid():
    """R2C-055 still prevents a missed GBALD value from disappearing."""
    spec = _carrier_spec(
        lane=[],
        glossary=[_glossary_carrier("R_0", paper_value=2000.0)],
    )

    assert cross_check_parameter_carrier_exclusivity(spec) == []


def test_same_numeric_value_on_unrelated_parameters_is_valid():
    spec = _carrier_spec(
        lane=[_scale_carrier("d_safe", paper_value=0.375)],
        glossary=[_glossary_carrier("safety_margin", paper_value=0.375)],
    )

    assert cross_check_parameter_carrier_exclusivity(spec) == []


def test_token_similar_names_are_not_a_fuzzy_identity_join():
    spec = _carrier_spec(
        lane=[_scale_carrier("similarity_cutoff", paper_value=0.95)],
        glossary=[_glossary_carrier(
            "similarity cutoff",
            aliases=["similarity_threshold"],
            paper_value=0.95,
        )],
    )

    assert cross_check_parameter_carrier_exclusivity(spec) == []


def test_exact_matching_preserves_case_and_keeps_K_distinct_from_k():
    spec = _carrier_spec(
        lane=[_scale_carrier("K", paper_value=26)],
        glossary=[_glossary_carrier("k", paper_value=26)],
    )

    assert cross_check_parameter_carrier_exclusivity(spec) == []


def test_exclusivity_match_strips_boundary_whitespace_only():
    spec = _carrier_spec(
        lane=[_scale_carrier("  R_0 ", paper_value=2000)],
        glossary=[_glossary_carrier(
            "radius",
            aliases=[" R_0  "],
            paper_value=2000,
        )],
    )

    errors = cross_check_parameter_carrier_exclusivity(spec)

    assert len(errors) == 1
    assert "stripped labels=['radius', 'R_0']" in errors[0]


def test_carrier_exclusivity_is_wired_only_in_strict_validation(
    tmp_path: Path,
):
    raw_spec = _minimal_valid_spec(paradigm_id="motion_planning")
    raw_spec["critical_requirements"][
        "scale_dependent_hyperparameters"
    ] = [_scale_carrier("d_safe", paper_value=0.375)]
    raw_spec["critical_requirements"]["param_glossary"] = [
        _glossary_carrier("d_safe", paper_value=0.375)
    ]
    spec_path = tmp_path / "method_spec.json"
    spec_path.write_text(json.dumps(raw_spec), encoding="utf-8")
    validator = Path("scripts/validate_method_spec.py")

    ordinary = subprocess.run(
        [sys.executable, str(validator), str(spec_path)],
        check=False,
        capture_output=True,
        text=True,
    )
    strict = subprocess.run(
        [sys.executable, str(validator), str(spec_path), "--strict"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert ordinary.returncode == 0, ordinary.stderr
    assert strict.returncode == 2
    assert (
        "strict: parameter-carrier exclusivity cross-check failed"
        in strict.stderr
    )
    assert "paper_value_carrier_duplication" in strict.stderr
    assert "taxonomy contract cross-check" not in strict.stderr


# ---------------------------------------------------------------------------
# Piece 3: the pack-authoring refusal (the gate).
# ---------------------------------------------------------------------------


def _pack(*, checks, derivation=None) -> dict:
    pack: dict = {"semantic_checks": checks}
    if derivation is not None:
        pack["params_derivation"] = derivation
    return pack


def test_pack_with_2x_check_and_no_declaration_is_refused():
    errors: list[str] = []
    _check_params_derivation(_pack(checks=[
        {"id": "GTSF-params-similarity-cutoff-valid",
         "stage": "stage_2x_params",
         "check": "similarity_cutoff is in (0, 1)."}]), errors, [])
    joined = "\n".join(errors)
    assert "no params_derivation block" in joined


def test_pack_naming_an_undeclared_param_is_refused_even_with_empty_mapping():
    """The pdfgnn case with the trivial escape closed: an explicit empty
    mapping answers the declaration floor and cannot answer a check that
    names its own requirement."""
    errors: list[str] = []
    _check_params_derivation(_pack(
        checks=[{"id": "GTSF-params-similarity-cutoff-valid",
                 "stage": "stage_2x_params",
                 "check": "similarity_cutoff is in (0, 1) so the graph "
                          "construction produces a meaningful sparse graph."}],
        derivation={}), errors, [])
    joined = "\n".join(errors)
    assert "GTSF-params-similarity-cutoff-valid" in joined
    assert "similarity_cutoff" in joined
    assert "no params_derivation block" not in joined


def test_pack_declaring_the_named_param_passes():
    errors: list[str] = []
    _check_params_derivation(_pack(
        checks=[{"id": "GTSF-params-similarity-cutoff-valid",
                 "stage": "stage_2x_params",
                 "check": "similarity_cutoff is in (0, 1)."}],
        derivation={"similarity_cutoff": {
            "kind": "config_path",
            "reasoning": "Cosine threshold the graph builder consumes.",
            "paper_section": "Appendix A, Table 5"}}), errors, [])
    assert errors == []


def test_a_deliberate_suppress_answers_the_named_param_floor():
    """Dropping a param is a declared decision, not an absent producer."""
    errors: list[str] = []
    _check_params_derivation(_pack(
        checks=[{"id": "X-params", "stage": "stage_2x_params",
                 "check": "hidden_dim must not appear."}],
        derivation={"hidden_dim": {
            "kind": "suppress", "reason": "No neural model here."}}),
        errors, [])
    assert errors == []


def test_a_generically_derived_param_needs_no_declaration():
    """The fedavg shape, found by sweeping every run-local pack: its check
    references learning_rate while declaring only its five bespoke params.
    The deriver emits learning_rate from a generic reader, so demanding a
    declaration would false-fail a pack whose 2.x stage really passed."""
    errors: list[str] = []
    _check_params_derivation(_pack(
        checks=[{"id": "FL-params-fedavg-hyperparams",
                 "stage": "stage_2x_params",
                 "check": "params.json contains client_fraction (C) and "
                          "learning_rate with provenance."}],
        derivation={"client_fraction": {
            "kind": "config_path", "reasoning": "Client sample fraction."}}),
        errors, [])
    assert errors == []


def test_a_keyword_extra_on_the_packs_own_signature_needs_no_declaration():
    """A name the pack's declared interface carries as a keyword extra is
    derived from that signature, so it already has a producer."""
    errors: list[str] = []
    pack = _pack(
        checks=[{"id": "FL-params", "stage": "stage_2x_params",
                 "check": "params.json contains local_epochs with provenance."}],
        derivation={})
    pack["scaffold_hints"] = {"interface_hint": (
        "federated_train(model, client_datasets, num_rounds, seed, *, "
        "local_epochs: int = 5) -> FederatedTrainingResult")}
    _check_params_derivation(pack, errors, [])
    assert errors == []


def test_a_provenance_label_is_not_a_required_param():
    """The other sweep false positive: a params-hygiene check naming only
    the provenance vocabulary requires no parameter entry."""
    errors: list[str] = []
    _check_params_derivation(_pack(
        checks=[{"id": "paper_source_values_match_paper",
                 "stage": "stage_2x_params",
                 "check": "Every param with source: paper is actually stated "
                          "in the paper; values pulled from conventions are "
                          "source: system_inferred."}],
        derivation={}), errors, [])
    assert errors == []


def test_explicit_empty_mapping_answers_a_check_naming_no_param():
    """A general params-hygiene check cannot false-fail a good pack."""
    errors: list[str] = []
    _check_params_derivation(_pack(
        checks=[{"id": "X-params", "stage": "stage_2x_params",
                 "check": "Every declared value traces to the paper."}],
        derivation={}), errors, [])
    assert errors == []


def test_checks_at_other_stages_are_untouched():
    errors: list[str] = []
    _check_params_derivation(_pack(checks=[
        {"id": "X-nb", "stage": "stage_3a_notebook",
         "check": "The notebook calls load_data and plots similarity_cutoff."}]),
        errors, [])
    assert errors == []


@pytest.mark.parametrize("text,expected", [
    ("similarity_cutoff is in (0, 1).", ["similarity_cutoff"]),
    ("The value sits in critical_requirements.training.", []),
    ("params_derivation declares scale_dependent_hyperparameters.", []),
    ("Both max_epochs and learning_rate trace to the paper.",
     ["max_epochs", "learning_rate"]),
    ("similarity_cutoff and similarity_cutoff again.", ["similarity_cutoff"]),
    ("No snake case here at all.", []),
    ("values pulled from conventions are source: system_inferred.", []),
    ("a bare spec_default entry keeps its used_in_notebook flag", []),
])
def test_named_param_extraction_ignores_pipeline_surface_names(text, expected):
    assert _named_params_in_check(text) == expected


def test_the_real_pdfgnn_pack_is_refused_and_the_real_fedavg_pack_is_not():
    """Known-bad and known-good, embedded verbatim from the real artifacts
    (2026-08-05): the pack whose 2.x review halted is refused naming the
    exact parameter, and the fedavg pack the maintainer weighed for promotion stays
    clean. Embedded rather than read from r2c_runs/ because run dirs are
    mutable by design: the 2026-08-05 day roll replaced the halted pack
    with a compliant one and archives get deleted by default, so a test
    reading the live dirs breaks on every re-roll."""
    # The pdfgnn night pack (archived as _5): a stage_2x_params check
    # naming similarity_cutoff, and NO params_derivation block at all.
    halted_pack = {
        "scaffold_hints": {"interface_hint": (
            "forecast(model, past_demand, static_features, "
            "time_varying_features, adjacency_matrix, prediction_length, "
            "seed, *paradigm_extras_by_name) -> ForecastResult")},
        "semantic_checks": [{
            "id": "GTSF-params-similarity-cutoff-valid",
            "stage": "stage_2x_params",
            "check": "similarity_cutoff is in (0, 1) so the graph "
                     "construction produces a meaningful sparse graph. "
                     "Values <= 0 connect everything; values >= 1 connect "
                     "nothing (unless features are identical).",
        }],
    }
    errors: list[str] = []
    _check_params_derivation(halted_pack, errors, [])
    assert any("similarity_cutoff" in e for e in errors)

    # The fedavg pack: its check references learning_rate, which its own
    # interface carries as a keyword extra, plus five declared params.
    fedavg_pack = {
        "scaffold_hints": {"interface_hint": (
            "federated_train(model, client_datasets, num_rounds, seed, *, "
            "client_fraction: float = 0.1, local_epochs: int = 5, "
            "local_batch_size: int = 10, learning_rate: float = 0.001) "
            "-> FederatedTrainingResult")},
        "params_derivation": {
            "client_fraction": {"kind": "config_path", "reasoning": "C."},
            "local_epochs": {"kind": "config_path", "reasoning": "E."},
            "local_batch_size": {"kind": "config_path", "reasoning": "B."},
            "num_clients": {"kind": "config_path", "reasoning": "K."},
            "expected_updates_per_round": {
                "kind": "derived_statistic", "formula": "n * E / (K * B)",
                "inputs": {"E": "params.local_epochs",
                           "B": "params.local_batch_size",
                           "n": "spec.critical_requirements.data_setup.dataset_size",
                           "K": "params.num_clients"}},
        },
        "semantic_checks": [{
            "id": "FL-params-fedavg-hyperparams",
            "stage": "stage_2x_params",
            "check": "params.json contains client_fraction (C), "
                     "local_epochs (E), local_batch_size (B), num_clients "
                     "(K), and learning_rate with provenance. The derived "
                     "statistic expected_updates_per_round = n*E/(K*B) is "
                     "also present.",
        }],
    }
    errors = []
    _check_params_derivation(fedavg_pack, errors, [])
    assert errors == []


# ---------------------------------------------------------------------------
# The drift guard: the checked-in JSON exports must match the models.
# ---------------------------------------------------------------------------


def test_exported_json_schemas_match_the_pydantic_models():
    """`schemas/method_spec.schema.json` had been stale since two earlier
    commits changed the models without re-running `schemas/export.py`
    (seed_param and the training nulls). Nothing enforced the match, so
    the drift was invisible. Regenerate in memory and compare."""
    import importlib
    import schemas.export as export

    importlib.reload(export)
    root = export.ROOT
    for model_cls, version, out_name, slug in (
        (export.ArchContractV2, export.ARCH_CONTRACT_V2_VERSION,
         "arch_contract.schema.json", "arch_contract"),
        (export.MethodSpec, export.METHOD_SPEC_VERSION,
         "method_spec.schema.json", "method_spec"),
        (export.PaperMap, export.PAPER_MAP_VERSION,
         "paper_map.schema.json", "paper_map"),
        (export.FinalManifest, export.FINAL_MANIFEST_VERSION,
         "final_manifest.schema.json", "final_manifest"),
        (export.ParadigmGapReport, export.PARADIGM_GAP_VERSION,
         "paradigm_gap.schema.json", "paradigm_gap"),
        (export.ProposalValidationResult, export.PARADIGM_PROPOSAL_VERSION,
         "paradigm_proposal.schema.json", "paradigm_proposal"),
    ):
        expected = model_cls.model_json_schema()
        expected["$schema"] = "https://json-schema.org/draft/2020-12/schema"
        expected["$id"] = f"https://r2c.local/schemas/{slug}/v{version}"
        on_disk = json.loads(
            (root / "schemas" / out_name).read_text(encoding="utf-8"))
        assert on_disk == expected, (
            f"schemas/{out_name} is stale — run `python3 schemas/export.py` "
            f"after changing schemas/*.py")


def test_a_protocol_bound_glossary_entry_needs_no_runtime_alias():
    # pdfgnn 2026-08-11 attempt-6 halt: the analyzer's 'context length'
    # glossary entry agreed with the typed protocol carrier (schema-checked)
    # but exposed no Python identifier, and the consumability floor demanded
    # a second runtime path for a value the protocol's parameter_name
    # already delivers deterministically. A glossary entry whose labels join
    # a bound protocol quantity is consumable through that carrier.
    from tests.test_evaluation_protocol import _pdfgnn_spec

    spec = _pdfgnn_spec()
    spec["critical_requirements"]["param_glossary"] = [{
        "name": "context length",
        "aliases": [],
        "meaning_quote": "we limit node features to P demand lags",
        "paper_section": "Section 3.2",
        "paper_value": 10,
    }]

    validated = MethodSpec.model_validate(spec)
    assert cross_check_parameter_carrier_consumability(validated) == []

    # Guardrail: an unnameable value NOT joined to any protocol carrier
    # still fails; the exemption is the protocol join, not a blanket pass.
    spec["critical_requirements"]["param_glossary"].append({
        "name": "similarity cutoff",
        "aliases": ["edge threshold"],
        "meaning_quote": "a specified threshold",
        "paper_section": "Section 3.2",
        "paper_value": 0.95,
    })
    validated = MethodSpec.model_validate(spec)
    errors = cross_check_parameter_carrier_consumability(validated)
    assert len(errors) == 1
    assert "similarity cutoff" in errors[0]
