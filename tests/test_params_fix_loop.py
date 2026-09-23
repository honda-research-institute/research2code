"""Stage 2.x provenance fix loop (R2C-033, approved 2026-07-28).

The pdwa 2026-07-28 halt: the analyzer labeled a computed value
(d_safe = V_max/4 = 0.375) as paper-stated, the deterministic provenance
probes caught it, and stage 2.x halted with retry_count 0 — the whole
runnable delivery degraded to explanation-only on a one-turn-fixable
labeling error. The fix loop routes the probe findings mechanically back
to the method analyzer (bounded, two iterations), enforces a scoped-edit
diff so the retry channel cannot become a provenance-laundering channel,
re-runs the spec validator + derive_params + the probes as the terminal
gate, and makes every surviving provenance change loud (assumptions.md
entry + params_provenance_repaired run event).

Fixtures per the approved design: the known-good pdwa d_safe shape
converges; genuine fabrication exhausts and degrades exactly as today;
a scoped-edit violation is rejected; the disclosure entries fire; and
non-probe validator failures keep today's immediate halt (no dispatch).
"""

from __future__ import annotations

import json

import run_events
import run_layout
import run_pipeline
from tests.helpers.state import make_state

# The live pdwa 2026-07-28 stderr, verbatim shape (stage_2x.halt context).
PDWA_STDERR = (
    "FAIL: 2 validation error(s):\n"
    "  - provenance probe US-3b: d_safe: d_safe claims the paper states "
    "V_max=1.5, but no such value appears near 'V_max' in the paper text "
    "(fabricated paper claim)\n"
    "  - provenance probe US-3: d_safe: d_safe=0.375 is source=paper but "
    "no rendering of the value (0.375, 37.5%, 37.5 %, 3.75e-1, 3.75e-01) "
    "appears in the paper text"
)

SCHEMA_STDERR = (
    "FAIL: 1 validation error(s):\n"
    "  - params.json fails Pydantic validation: value must be a number"
)

TPLUS_STDERR = (
    "FAIL: 1 validation error(s):\n"
    "  - provenance probe US-3b: forecast_horizon: forecast_horizon "
    "claims paper_value=1 from T+1 notation, but the paper defines K "
    "symbolically and does not state a numeric one-call horizon"
)


def _spec(*, scale_description: str, paper_value: float | None = 0.375,
          extra_glossary_quote: str = "safety distance for collision "
          "avoidance") -> dict:
    """A pdwa-d_safe-shaped spec: the fabricated paper claim lives in the
    scale_dependent_hyperparameters entry, exactly as on the live run."""
    scale_entry: dict = {
        "name": "d_safe",
        "formula": "V_max / 4",
        "assumes_data_scale": "simulation_velocity_m_per_s",
        "description": scale_description,
        "paper_section": "Section 3.3, Eq. (9)",
    }
    if paper_value is not None:
        scale_entry["paper_value"] = paper_value
    return {
        "schema_version": "1.0.0",
        "paper": {"title": "PDWA", "authors": ["A"], "year": 2026},
        "core_method": {"description": "Predictive DWA."},
        "comparison": {
            "classification": {"id": "motion_planning/dwa"},
            "pluggable_component": {
                "signature": "plan(start, goal, environment, "
                             "d_safe=0.375, r_obs=0.3)",
            },
        },
        "critical_requirements": {
            "param_glossary": [
                {"name": "d_safe", "aliases": [],
                 "meaning_quote": extra_glossary_quote,
                 "paper_section": "Section 3.3, Eq. (9)"},
                {"name": "r_obs", "aliases": [],
                 "meaning_quote": "Each obstacle has a radius of 0.3 m.",
                 "paper_section": "Section 2.2"},
            ],
            "scale_dependent_hyperparameters": [scale_entry],
            "training": {"learning_rate": None},
            "data_setup": {"batch_size": None},
        },
        "methodology_replication_contract": {
            "schema_version": "1.0",
            "elements": [
                {"element_id": "dwa-window",
                 "paper_evidence": "The dynamic window filters on d_safe.",
                 "required_behavior": "Filter on predicted positions."},
            ],
        },
    }


BAD_SPEC = _spec(
    scale_description=(
        "Safety distance threshold. Derived via d_safe = V_max / 4. The "
        "paper uses V_max = 1.5 m/s giving d_safe = 0.375 m."))

# The honest fix: paper_value claim dropped, description states the
# formula with an assumed input; ONLY d_safe surfaces changed.
FIXED_SPEC = _spec(
    scale_description=(
        "Safety distance threshold. The paper defines it only by the "
        "formula d_safe = V_max / 4; V_max is an assumed demo value."),
    paper_value=None)


def _typed_protocol_spec(*, stated_horizon: bool) -> dict:
    """Pdfgnn-shaped T+1 control on the existing fix-loop fixture."""
    spec = json.loads(json.dumps(BAD_SPEC))
    spec["comparison"]["evaluation_protocol"] = {
        "scheme": {
            "kind": "single_holdout",
            "paper_value_status": "paper_stated",
            "description": "One chronological holdout.",
            "evidence_quote": "Validation precedes the test interval.",
            "paper_section": "Section 4",
            "paper_element_ids": ["split"],
        },
        "quantities": [
            {
                "role": "context_length",
                "parameter_name": "context_length",
                "paper_names": ["Context length"],
                "value": 10,
                "unit": "week",
                "granularity": 1,
                "paper_value_status": "paper_stated",
                "evidence_quote": "Context length is 10.",
                "paper_section": "Section 4",
                "paper_element_ids": ["context"],
            },
            {
                "role": "forecast_call_horizon",
                "parameter_name": "forecast_horizon",
                "paper_names": ["K"],
                "value": 1 if stated_horizon else None,
                "unit": "week",
                "granularity": 1,
                "paper_value_status": (
                    "paper_stated" if stated_horizon else "paper_unspecified"
                ),
                "evidence_quote": (
                    "The forecast target begins at T+1."
                    if stated_horizon
                    else "K denotes the number of future forecast steps."
                ),
                "paper_section": "Section 3",
                "paper_element_ids": ["horizon"],
            },
            {
                "role": "validation_span",
                "parameter_name": None,
                "paper_names": ["Validation"],
                "value": 13,
                "unit": "week",
                "granularity": 1,
                "paper_value_status": "paper_stated",
                "evidence_quote": "Validation covers 13 weeks.",
                "paper_section": "Section 4",
                "paper_element_ids": ["split"],
            },
            {
                "role": "test_span",
                "parameter_name": None,
                "paper_names": ["Test"],
                "value": 26,
                "unit": "week",
                "granularity": 1,
                "paper_value_status": "paper_stated",
                "evidence_quote": "Test covers 26 weeks.",
                "paper_section": "Section 4",
                "paper_element_ids": ["split"],
            },
        ],
    }
    return spec


BAD_PROTOCOL_SPEC = _typed_protocol_spec(stated_horizon=True)
FIXED_PROTOCOL_SPEC = _typed_protocol_spec(stated_horizon=False)


def _params(*, d_safe_entry: dict, r_obs_entry: dict | None = None) -> str:
    return json.dumps({
        "params": {
            "d_safe": d_safe_entry,
            "r_obs": r_obs_entry or {
                "value": 0.3,
                "source": "spec_default",
                "reasoning": "Value from the pluggable signature default.",
            },
        },
    }, indent=2)


BAD_PARAMS = _params(d_safe_entry={
    "value": 0.375,
    "source": "paper",
    "paper_section": "Section 3.3, Eq. (9)",
    "note": "The paper uses V_max = 1.5 m/s giving d_safe = 0.375 m.",
})

FIXED_PARAMS = _params(d_safe_entry={
    "value": 0.375,
    "source": "system_inferred",
    "reasoning": "Derived via the paper's formula d_safe = V_max / 4 "
                 "with the assumed demo V_max = 1.5 m/s (logged "
                 "assumption); the paper never prints 0.375.",
})

# A laundering shape: d_safe honestly fixed AND r_obs quietly relabeled.
LAUNDERED_PARAMS = _params(
    d_safe_entry=json.loads(FIXED_PARAMS)["params"]["d_safe"],
    r_obs_entry={
        "value": 0.3,
        "source": "paper",
        "paper_section": "Section 2.2",
    },
)

BAD_PROTOCOL_PARAMS = _params(d_safe_entry={
    "value": 0.375,
    "source": "system_inferred",
    "reasoning": "Unrelated control.",
})
_bad_protocol_payload = json.loads(BAD_PROTOCOL_PARAMS)
_bad_protocol_payload["params"]["forecast_horizon"] = {
    "value": 4,
    "source": "system_default",
    "paper_value": 1,
    "reasoning": "Runtime K=4 differs from the alleged paper K=1.",
    "protocol_role": "forecast_call_horizon",
    "protocol_unit": "week",
    "protocol_granularity": 1,
    "paper_value_status": "paper_stated",
    "paper_says": "The forecast target begins at T+1.",
    "paper_section": "Section 3",
    "paper_element_ids": ["horizon"],
}
BAD_PROTOCOL_PARAMS = json.dumps(_bad_protocol_payload, indent=2)

_fixed_protocol_payload = json.loads(BAD_PROTOCOL_PARAMS)
_fixed_horizon = _fixed_protocol_payload["params"]["forecast_horizon"]
_fixed_horizon.pop("paper_value")
_fixed_horizon.update({
    "source": "system_inferred",
    "reasoning": "K is paper-unspecified; runtime K=4 is system-owned.",
    "paper_value_status": "paper_unspecified",
    "paper_says": "K denotes the number of future forecast steps.",
})
FIXED_PROTOCOL_PARAMS = json.dumps(_fixed_protocol_payload, indent=2)


def _review(findings: list[dict]) -> str:
    # Full seven-key shape: the read path validates stage reviews against
    # schemas/stage_review_report.py (B-01), and every real reviewer output
    # carries these keys.
    return json.dumps({
        "schema_version": "1.0.0",
        "stage_id": "stage_2x_params",
        "review_status": "issues_found" if findings else "passed",
        "summary": "fabricated test review",
        "findings": findings,
        "checks_summary": [],
        "caveats": [],
    }, indent=2)


def _setup_stage(monkeypatch, run_dir, fake_subprocess, spec: dict):
    state = make_state(run_dir)
    state.paths.method_spec.write_text(
        json.dumps(spec, indent=2), encoding="utf-8")
    # Orthogonal 2x machinery (binding disclosure, scale calibration,
    # budget sufficiency) is not under test here.
    for name in ("_log_per_dataset_binding", "_apply_scale_calibration"):
        monkeypatch.setattr(run_pipeline, name, lambda *a, **k: None)
    monkeypatch.setattr(run_pipeline, "_check_training_budget_sufficiency",
                        lambda *a, **k: None)
    fake_subprocess.set_run_dir(run_dir)
    return state


# -- unit: the probe-bullet parser -------------------------------------------


def test_provenance_bullets_parse_the_live_pdwa_stderr():
    bullets = run_pipeline._params_provenance_bullets(PDWA_STDERR)
    assert bullets is not None
    assert [b["probe"] for b in bullets] == ["US-3b", "US-3"]
    assert {b["param"] for b in bullets} == {"d_safe"}
    assert bullets[0]["bullet"].startswith(
        "provenance probe US-3b: d_safe:")


def test_provenance_bullets_reject_non_probe_classes():
    # Schema-class stderr, empty stderr, and a traceback (validator crash)
    # all decline the loop — today's halt path owns those.
    assert run_pipeline._params_provenance_bullets(SCHEMA_STDERR) is None
    assert run_pipeline._params_provenance_bullets("") is None
    assert run_pipeline._params_provenance_bullets(
        "Traceback (most recent call last): ...") is None


def test_provenance_bullets_reject_mixed_probe_and_schema():
    mixed = (PDWA_STDERR + "\n  - params.json missing required parameters "
             "for paradigm 'x': ['batch_size']")
    assert run_pipeline._params_provenance_bullets(mixed) is None


# -- unit: glossary alias expansion -------------------------------------------


def test_glossary_expansion_links_paper_symbol_and_derived_name():
    spec = {"critical_requirements": {"param_glossary": [
        {"name": "E", "aliases": ["local_epochs"]},
        {"name": "C", "aliases": []},
    ]}}
    out = run_pipeline._glossary_linked_param_names(spec, {"local_epochs"})
    assert out == {"local_epochs", "E"}


# -- unit: the scoped-edit guardrail ------------------------------------------


def test_scope_allows_named_param_surfaces_only():
    assert run_pipeline._params_spec_scope_violations(
        BAD_SPEC, FIXED_SPEC, {"d_safe"}) == []


def test_scope_rejects_an_unnamed_glossary_edit():
    tampered = json.loads(json.dumps(FIXED_SPEC))
    tampered["critical_requirements"]["param_glossary"][1][
        "meaning_quote"] = "Each obstacle has a radius of 0.5 m."
    violations = run_pipeline._params_spec_scope_violations(
        BAD_SPEC, tampered, {"d_safe"})
    assert any("r_obs" in v for v in violations)


def test_scope_rejects_an_unnamed_training_field_edit():
    tampered = json.loads(json.dumps(FIXED_SPEC))
    tampered["critical_requirements"]["training"]["learning_rate"] = 0.01
    violations = run_pipeline._params_spec_scope_violations(
        BAD_SPEC, tampered, {"d_safe"})
    assert any("training.learning_rate" in v for v in violations)


def test_scope_rejects_edits_outside_the_parameter_surfaces():
    tampered = json.loads(json.dumps(FIXED_SPEC))
    tampered["core_method"]["description"] = "Rewritten narrative."
    violations = run_pipeline._params_spec_scope_violations(
        BAD_SPEC, tampered, {"d_safe"})
    assert any("core_method" in v for v in violations)


def test_scope_contract_elements_attributed_by_mention():
    changed = json.loads(json.dumps(BAD_SPEC))
    changed["methodology_replication_contract"]["elements"][0][
        "paper_evidence"] = ("The dynamic window filters on d_safe, "
                             "defined only as V_max / 4.")
    assert run_pipeline._params_spec_scope_violations(
        BAD_SPEC, changed, {"d_safe"}) == []
    # The same element edited when d_safe is NOT the named param: rejected.
    violations = run_pipeline._params_spec_scope_violations(
        BAD_SPEC, changed, {"r_obs"})
    assert any("dwa-window" in v for v in violations)


def test_scope_allows_only_named_typed_protocol_provenance_fields():
    assert run_pipeline._params_spec_scope_violations(
        BAD_PROTOCOL_SPEC, FIXED_PROTOCOL_SPEC, {"forecast_horizon"}
    ) == []

    changed_identity = json.loads(json.dumps(FIXED_PROTOCOL_SPEC))
    changed_identity["comparison"]["evaluation_protocol"]["quantities"][1][
        "unit"
    ] = "day"
    violations = run_pipeline._params_spec_scope_violations(
        BAD_PROTOCOL_SPEC, changed_identity, {"forecast_horizon"}
    )
    assert any("immutable protocol identity" in item for item in violations)

    changed_other_role = json.loads(json.dumps(FIXED_PROTOCOL_SPEC))
    changed_other_role["comparison"]["evaluation_protocol"]["quantities"][2][
        "value"
    ] = 14
    violations = run_pipeline._params_spec_scope_violations(
        BAD_PROTOCOL_SPEC, changed_other_role, {"forecast_horizon"}
    )
    assert any("parameter_name=None" in item for item in violations)


def test_output_scope_rejects_an_unnamed_params_entry_change():
    violations = run_pipeline._params_output_scope_violations(
        json.loads(BAD_PARAMS), json.loads(LAUNDERED_PARAMS), {"d_safe"})
    assert any("r_obs" in v for v in violations)
    assert run_pipeline._params_output_scope_violations(
        json.loads(BAD_PARAMS), json.loads(FIXED_PARAMS), {"d_safe"}) == []


# -- unit: the mechanical fix finding -----------------------------------------


def test_fix_finding_carries_bullets_names_and_resolution_kinds():
    bullets = run_pipeline._params_provenance_bullets(PDWA_STDERR)
    finding = run_pipeline._params_provenance_fix_finding(bullets, 1)
    assert finding["severity"] == "critical"
    assert "d_safe" in finding["description"]
    # Each probe's complaint rides along verbatim.
    for b in bullets:
        assert b["bullet"] in finding["description"]
    # The three honest resolution kinds, and the spec-only edit scope.
    fix = finding["proposed_fix"]
    assert "derived statistic" in fix
    assert "demo assumption" in fix
    assert "corrected quote" in fix
    assert "method_spec.json" in fix
    assert "params.json" in finding["description"]
    assert "comparison.evaluation_protocol" in finding["description"]
    assert "paper_value_status" in fix


def test_fix_finding_inlines_acceptance_contracts_and_spec_slices():
    """Agent dispatch optimization (2026-09-01): the finding is
    self-sufficient — failing probes' acceptance contracts plus verbatim
    spec/params slices for the probe-named parameters ride along, so the
    retry needs zero pipeline-source reads and no whole-spec hunt."""
    bullets = run_pipeline._params_provenance_bullets(PDWA_STDERR)
    allowed = run_pipeline._glossary_linked_param_names(BAD_SPEC, {"d_safe"})
    finding = run_pipeline._params_provenance_fix_finding(
        bullets, 1, spec=BAD_SPEC, params=json.loads(BAD_PARAMS),
        allowed=allowed)
    desc = finding["description"]
    # The failing probes' contracts are inlined; a probe that did not fire
    # is not.
    assert "US-3b inline paper claims" in desc
    assert "US-3 paper findability" in desc
    assert "US-1 plausibility" not in desc
    assert "do not read pipeline source" in desc
    # The d_safe surfaces are inlined; unrelated params are not.
    assert "param_glossary entry `d_safe`" in desc
    assert '"V_max / 4"' in desc  # the scale-lane entry rides verbatim
    assert "params.json entry `d_safe`" in desc
    assert "dwa-window" in desc  # contract element mentioning d_safe
    assert "param_glossary entry `r_obs`" not in desc


def test_fix_finding_without_spec_context_keeps_contracts_only():
    bullets = run_pipeline._params_provenance_bullets(PDWA_STDERR)
    finding = run_pipeline._params_provenance_fix_finding(bullets, 1)
    assert "Acceptance contracts" in finding["description"]
    assert "spec/params slices" not in finding["description"]


def test_fix_finding_slices_are_capped():
    bullets = run_pipeline._params_provenance_bullets(PDWA_STDERR)
    huge = json.loads(json.dumps(BAD_SPEC))
    huge["critical_requirements"]["param_glossary"][0][
        "meaning_quote"] = "d_safe " * 4000
    finding = run_pipeline._params_provenance_fix_finding(
        bullets, 1, spec=huge, params=json.loads(BAD_PARAMS),
        allowed={"d_safe"})
    assert "slices truncated" in finding["description"]
    assert len(finding["description"]) < 20000


# -- integration: known-good pdwa shape converges ------------------------------


def test_pdwa_d_safe_shape_converges_in_one_iteration(
        monkeypatch, run_dir, fake_dispatch, fake_subprocess):
    state = _setup_stage(monkeypatch, run_dir, fake_subprocess, BAD_SPEC)
    fake_subprocess.expect_script(  # derive_params.py (initial)
        returncode=0, writes={".pipeline/params.json": BAD_PARAMS})
    fake_subprocess.expect_stage2_script(ok=False, err=PDWA_STDERR)
    fake_dispatch.expect(  # the analyzer performs the honest fix
        agent="r2c-method-analyzer",
        writes={".pipeline/method_spec.json":
                json.dumps(FIXED_SPEC, indent=2)})
    fake_subprocess.expect_script(returncode=0)  # validate_method_spec.py
    fake_subprocess.expect_script(  # derive_params.py (re-derivation)
        returncode=0, writes={".pipeline/params.json": FIXED_PARAMS})
    fake_subprocess.expect_stage2_script(ok=True, err="")  # terminal gate
    fake_dispatch.expect(  # stage reviewer: clean
        agent="r2c-stage-reviewer",
        writes={".pipeline/stage_review_stage_2x_params.json":
                _review([])})

    result = run_pipeline.run_stage_2x(state)

    assert result.status == "completed"
    params = json.loads(
        (run_dir / ".pipeline" / "params.json").read_text(encoding="utf-8"))
    assert params["params"]["d_safe"]["source"] == "system_inferred"
    assert params["params"]["r_obs"]["source"] == "spec_default"
    assert [c.agent for c in fake_dispatch.calls] == [
        "r2c-method-analyzer", "r2c-stage-reviewer"]
    # The analyzer's fix prompt carried the probe bullets verbatim.
    assert "provenance probe US-3b: d_safe" in fake_dispatch.calls[0].prompt
    # No halt artifact.
    assert not (run_dir / ".pipeline" / "stage_2x.halt").exists()


def test_converged_repair_is_loud_assumption_and_run_event(
        monkeypatch, run_dir, fake_dispatch, fake_subprocess):
    state = _setup_stage(monkeypatch, run_dir, fake_subprocess, BAD_SPEC)
    fake_subprocess.expect_script(
        returncode=0, writes={".pipeline/params.json": BAD_PARAMS})
    fake_subprocess.expect_stage2_script(ok=False, err=PDWA_STDERR)
    fake_dispatch.expect(
        agent="r2c-method-analyzer",
        writes={".pipeline/method_spec.json":
                json.dumps(FIXED_SPEC, indent=2)})
    fake_subprocess.expect_script(returncode=0)
    fake_subprocess.expect_script(
        returncode=0, writes={".pipeline/params.json": FIXED_PARAMS})
    fake_subprocess.expect_stage2_script(ok=True, err="")
    fake_dispatch.expect(
        agent="r2c-stage-reviewer",
        writes={".pipeline/stage_review_stage_2x_params.json":
                _review([])})

    result = run_pipeline.run_stage_2x(state)
    assert result.status == "completed"

    # assumptions.md names the parameter, both provenances, and the probe.
    assumptions = (run_dir / run_layout.ASSUMPTIONS_MD).read_text(
        encoding="utf-8")
    assert "Parameter provenance repaired — d_safe" in assumptions
    assert "source: paper -> system_inferred" in assumptions
    assert "provenance probe US-3b: d_safe" in assumptions

    # The run event fired with the same facts.
    events = run_events.load_events(run_dir / ".pipeline")
    repaired = [e for e in events
                if e["event_type"] == "params_provenance_repaired"]
    assert len(repaired) == 1
    details = repaired[0]["details"]
    assert details["param"] == "d_safe"
    assert details["old_source"] == "paper"
    assert details["new_source"] == "system_inferred"
    assert any("US-3b" in b for b in details["probe_findings"])


def test_t_plus_one_protocol_repair_converges_through_analyzer_seam(
        monkeypatch, run_dir, fake_dispatch, fake_subprocess):
    """The real failure class: T+1 must become symbolic K, not K=1."""
    state = _setup_stage(
        monkeypatch, run_dir, fake_subprocess, BAD_PROTOCOL_SPEC
    )
    fake_subprocess.expect_script(
        returncode=0, writes={".pipeline/params.json": BAD_PROTOCOL_PARAMS}
    )
    fake_subprocess.expect_stage2_script(ok=False, err=TPLUS_STDERR)
    fake_dispatch.expect(
        agent="r2c-method-analyzer",
        writes={
            ".pipeline/method_spec.json": json.dumps(
                FIXED_PROTOCOL_SPEC, indent=2
            )
        },
    )
    fake_subprocess.expect_script(returncode=0)  # strict spec validator
    fake_subprocess.expect_script(
        returncode=0, writes={".pipeline/params.json": FIXED_PROTOCOL_PARAMS}
    )
    fake_subprocess.expect_stage2_script(ok=True, err="")
    fake_dispatch.expect(
        agent="r2c-stage-reviewer",
        writes={
            ".pipeline/stage_review_stage_2x_params.json": _review([])
        },
    )

    result = run_pipeline.run_stage_2x(state)

    assert result.status == "completed"
    spec = json.loads(state.paths.method_spec.read_text(encoding="utf-8"))
    horizon = spec["comparison"]["evaluation_protocol"]["quantities"][1]
    assert horizon["role"] == "forecast_call_horizon"
    assert horizon["parameter_name"] == "forecast_horizon"
    assert horizon["paper_value_status"] == "paper_unspecified"
    assert horizon["value"] is None
    params = json.loads(
        (run_dir / ".pipeline" / "params.json").read_text(encoding="utf-8")
    )["params"]["forecast_horizon"]
    assert params["value"] == 4
    assert params["source"] == "system_inferred"
    assert "paper_value" not in params
    assert "comparison.evaluation_protocol" in fake_dispatch.calls[0].prompt


# -- integration: genuine fabrication exhausts and degrades --------------------


def test_genuine_fabrication_exhausts_and_halts_exactly_as_today(
        monkeypatch, run_dir, fake_dispatch, fake_subprocess):
    """Two honest-attempt iterations (in-scope spec edits) that never
    satisfy the probes: the loop exhausts and the stage halts with the
    pre-loop class — producer_output_invalid — which is what routes the
    delivery to the explanation-only package."""
    state = _setup_stage(monkeypatch, run_dir, fake_subprocess, BAD_SPEC)
    attempt_1 = _spec(scale_description="Safety distance. Attempt one.")
    attempt_2 = _spec(scale_description="Safety distance. Attempt two.")
    still_bad_1 = _params(d_safe_entry={
        "value": 0.375, "source": "paper",
        "paper_section": "Section 3.3, Eq. (9)", "note": "Attempt one."})
    still_bad_2 = _params(d_safe_entry={
        "value": 0.375, "source": "paper",
        "paper_section": "Section 3.3, Eq. (9)", "note": "Attempt two."})

    fake_subprocess.expect_script(
        returncode=0, writes={".pipeline/params.json": BAD_PARAMS})
    fake_subprocess.expect_stage2_script(ok=False, err=PDWA_STDERR)
    for spec, still_bad in ((attempt_1, still_bad_1),
                            (attempt_2, still_bad_2)):
        fake_dispatch.expect(
            agent="r2c-method-analyzer",
            writes={".pipeline/method_spec.json":
                    json.dumps(spec, indent=2)})
        fake_subprocess.expect_script(returncode=0)  # spec validator
        fake_subprocess.expect_script(  # derive_params re-run
            returncode=0, writes={".pipeline/params.json": still_bad})
        fake_subprocess.expect_stage2_script(ok=False, err=PDWA_STDERR)

    result = run_pipeline.run_stage_2x(state)

    assert result.status == "halted"
    halt_artifact = json.loads(
        (run_dir / ".pipeline" / "stage_2x.halt").read_text(
            encoding="utf-8"))
    assert halt_artifact["halt_class"] == "producer_output_invalid"
    assert halt_artifact["retry_count"] == 2
    assert "validate_params_output.py failed" in halt_artifact["reason"]
    assert "provenance probe US-3" in halt_artifact["context"]["stderr"]
    # Nothing shipped, so nothing was disclosed as repaired.
    events = run_events.load_events(run_dir / ".pipeline")
    assert not [e for e in events
                if e["event_type"] == "params_provenance_repaired"]
    assumptions_path = run_dir / run_layout.ASSUMPTIONS_MD
    if assumptions_path.exists():
        assert "provenance repaired" not in assumptions_path.read_text(
            encoding="utf-8")


# -- integration: the laundering guardrails reject the retry -------------------


def test_scoped_edit_violation_rejects_the_retry_and_restores_the_spec(
        monkeypatch, run_dir, fake_dispatch, fake_subprocess):
    """A retry that also edits an UNRELATED parameter (r_obs) is rejected
    outright: the spec is restored, the iteration counts as failed, and
    no spec validator / derive_params run ever fires for it."""
    state = _setup_stage(monkeypatch, run_dir, fake_subprocess, BAD_SPEC)
    tampered = json.loads(json.dumps(FIXED_SPEC))
    tampered["critical_requirements"]["param_glossary"][1][
        "meaning_quote"] = "Each obstacle has a radius of 0.5 m."

    fake_subprocess.expect_script(
        returncode=0, writes={".pipeline/params.json": BAD_PARAMS})
    fake_subprocess.expect_stage2_script(ok=False, err=PDWA_STDERR)
    for _ in range(2):  # both iterations violate the edit scope
        fake_dispatch.expect(
            agent="r2c-method-analyzer",
            writes={".pipeline/method_spec.json":
                    json.dumps(tampered, indent=2)})

    result = run_pipeline.run_stage_2x(state)

    assert result.status == "halted"
    halt_artifact = json.loads(
        (run_dir / ".pipeline" / "stage_2x.halt").read_text(
            encoding="utf-8"))
    assert halt_artifact["halt_class"] == "producer_output_invalid"
    assert halt_artifact["retry_count"] == 2
    # The rejected spec never sticks: the original is back on disk.
    spec_on_disk = json.loads(
        state.paths.method_spec.read_text(encoding="utf-8"))
    assert spec_on_disk == BAD_SPEC
    # Only the initial derive_params ran — a rejected iteration never
    # reaches the spec validator or the re-derivation.
    assert len(fake_subprocess.script_calls) == 1
    assert len(fake_subprocess.stage2_calls) == 1
    # Each rejection is audit-visible in the event stream (the maintainer's
    # 2026-08-03 call: a rejected laundering attempt emits a run event,
    # not just a driver-log line).
    rejections = [e for e in run_events.load_events(run_dir / ".pipeline")
                  if e["event_type"] == "params_fix_scope_rejected"]
    assert len(rejections) == 2  # both iterations violated the scope
    assert all(e["details"]["surface"] == "method_spec"
               for e in rejections)
    assert rejections[0]["details"]["probe_named_parameters"] == ["d_safe"]
    assert rejections[0]["details"]["violations"]


def test_output_side_laundering_rejects_and_restores_params(
        monkeypatch, run_dir, fake_dispatch, fake_subprocess):
    """A spec edit that passes the spec-side diff but whose re-derivation
    changes an UNNAMED parameter's entry (the prose-channel laundering
    shape) is rejected: spec AND params.json are restored."""
    state = _setup_stage(monkeypatch, run_dir, fake_subprocess, BAD_SPEC)
    fake_subprocess.expect_script(
        returncode=0, writes={".pipeline/params.json": BAD_PARAMS})
    fake_subprocess.expect_stage2_script(ok=False, err=PDWA_STDERR)
    for _ in range(2):
        fake_dispatch.expect(
            agent="r2c-method-analyzer",
            writes={".pipeline/method_spec.json":
                    json.dumps(FIXED_SPEC, indent=2)})
        fake_subprocess.expect_script(returncode=0)  # spec validator
        fake_subprocess.expect_script(  # derive_params: r_obs also moved
            returncode=0, writes={".pipeline/params.json": LAUNDERED_PARAMS})

    result = run_pipeline.run_stage_2x(state)

    assert result.status == "halted"
    # params.json was restored to the pre-retry state: the laundered
    # r_obs relabel never survives.
    params = json.loads(
        (run_dir / ".pipeline" / "params.json").read_text(encoding="utf-8"))
    assert params["params"]["r_obs"]["source"] == "spec_default"
    spec_on_disk = json.loads(
        state.paths.method_spec.read_text(encoding="utf-8"))
    assert spec_on_disk == BAD_SPEC
    # The output-side rejection is audit-visible too, with its own
    # surface marker distinguishing it from a spec-side rejection.
    rejections = [e for e in run_events.load_events(run_dir / ".pipeline")
                  if e["event_type"] == "params_fix_scope_rejected"]
    assert len(rejections) == 2
    assert all(e["details"]["surface"] == "params_output"
               for e in rejections)


# -- integration: spec-invalid-after-retry counts as a failed iteration --------


def test_spec_invalid_after_retry_counts_as_failed_iteration_then_converges(
        monkeypatch, run_dir, fake_dispatch, fake_subprocess):
    state = _setup_stage(monkeypatch, run_dir, fake_subprocess, BAD_SPEC)
    fake_subprocess.expect_script(
        returncode=0, writes={".pipeline/params.json": BAD_PARAMS})
    fake_subprocess.expect_stage2_script(ok=False, err=PDWA_STDERR)
    # Iteration 1: in-scope fix, but the spec fails strict validation.
    fake_dispatch.expect(
        agent="r2c-method-analyzer",
        writes={".pipeline/method_spec.json":
                json.dumps(FIXED_SPEC, indent=2)})
    fake_subprocess.expect_script(
        returncode=1, stderr="strict validation failed: bad quote floor")
    # Iteration 2: same fix, now valid; converges.
    fake_dispatch.expect(
        agent="r2c-method-analyzer",
        writes={".pipeline/method_spec.json":
                json.dumps(FIXED_SPEC, indent=2)})
    fake_subprocess.expect_script(returncode=0)
    fake_subprocess.expect_script(
        returncode=0, writes={".pipeline/params.json": FIXED_PARAMS})
    fake_subprocess.expect_stage2_script(ok=True, err="")
    fake_dispatch.expect(
        agent="r2c-stage-reviewer",
        writes={".pipeline/stage_review_stage_2x_params.json":
                _review([])})

    result = run_pipeline.run_stage_2x(state)

    assert result.status == "completed"
    assert [c.agent for c in fake_dispatch.calls] == [
        "r2c-method-analyzer", "r2c-method-analyzer", "r2c-stage-reviewer"]


# -- integration: non-probe failures keep today's behavior (noop) --------------


def test_non_probe_validator_failure_keeps_todays_immediate_halt(
        monkeypatch, run_dir, fake_dispatch, fake_subprocess):
    """The no-findings noop: a validator failure with no provenance-probe
    bullets (schema class, crash class) never enters the loop — no
    analyzer dispatch, and the halt is byte-compatible with today's
    (producer_output_invalid, retry_count 0)."""
    state = _setup_stage(monkeypatch, run_dir, fake_subprocess, BAD_SPEC)
    fake_subprocess.expect_script(
        returncode=0, writes={".pipeline/params.json": BAD_PARAMS})
    fake_subprocess.expect_stage2_script(ok=False, err=SCHEMA_STDERR)

    result = run_pipeline.run_stage_2x(state)

    assert result.status == "halted"
    halt_artifact = json.loads(
        (run_dir / ".pipeline" / "stage_2x.halt").read_text(
            encoding="utf-8"))
    assert halt_artifact["halt_class"] == "producer_output_invalid"
    assert halt_artifact["retry_count"] == 0
    assert halt_artifact["reason"] == "validate_params_output.py failed"
    assert fake_dispatch.calls == []


# -- unit: the disclosure helper -----------------------------------------------


def test_repair_logger_skips_unchanged_entries(run_dir):
    state = make_state(run_dir)
    repaired = run_pipeline._log_params_provenance_repairs(
        state, "stage_2x",
        baseline_params=json.loads(BAD_PARAMS),
        final_params=json.loads(BAD_PARAMS),
        bullets_by_param={"d_safe": ["provenance probe US-3: d_safe: ..."]},
        iteration=1)
    assert repaired == []
    assert not (run_dir / run_layout.ASSUMPTIONS_MD).exists()


def test_repair_logger_names_param_provenances_and_probe(run_dir):
    state = make_state(run_dir)
    repaired = run_pipeline._log_params_provenance_repairs(
        state, "stage_2x",
        baseline_params=json.loads(BAD_PARAMS),
        final_params=json.loads(FIXED_PARAMS),
        bullets_by_param={
            "d_safe": ["provenance probe US-3: d_safe: d_safe=0.375 is "
                       "source=paper but ..."]},
        iteration=2)
    assert repaired == ["d_safe"]
    assumptions = (run_dir / run_layout.ASSUMPTIONS_MD).read_text(
        encoding="utf-8")
    assert "d_safe" in assumptions
    assert "paper" in assumptions and "system_inferred" in assumptions
    events = run_events.load_events(run_dir / ".pipeline")
    assert [e["event_type"] for e in events] == [
        "params_provenance_repaired"]
    assert events[0]["details"]["iteration"] == 2
