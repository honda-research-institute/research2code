"""Scenario-fidelity slice A: capture, quote floor, and visibility."""

from __future__ import annotations

import copy
import json
import subprocess
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

from schemas.method_spec import MethodSpec
from scripts import taxonomy
from scripts.render_run_report import render_run_report
from scripts.validate_method_spec import cross_check_scenario_assumptions
from tests.test_method_spec_schema import _minimal_valid_spec


ROOT = Path(__file__).resolve().parents[1]


def _motion_spec() -> dict:
    spec = _minimal_valid_spec(paradigm_id="motion_planning")
    spec["scenario_assumptions"] = {
        "obstacle_geometry": {
            "normalized_value": ["circle"],
            "evidence_quote": "We assume all obstacles are circular.",
            "paper_location": "Section IV-A",
        },
        "agent_population": {
            "normalized_value": {
                "agent_type": "pedestrian",
                "minimum_present": True,
            },
            "evidence_quote": (
                "The robot navigates through a crowd of pedestrians."
            ),
            "paper_location": "Section V",
        },
    }
    return spec


def test_motion_planning_scenario_assumptions_validate_and_join():
    spec = MethodSpec.model_validate(_motion_spec())

    assert spec.scenario_assumptions is not None
    assert (
        spec.scenario_assumptions["agent_population"].normalized_value
        == {"agent_type": "pedestrian", "minimum_present": True}
    )
    assert cross_check_scenario_assumptions(spec, ROOT) == []


def test_scenario_assumption_rejects_empty_normalized_value():
    spec = _motion_spec()
    spec["scenario_assumptions"]["obstacle_geometry"]["normalized_value"] = []

    with pytest.raises(ValidationError, match="normalized_value cannot be empty"):
        MethodSpec.model_validate(spec)


def test_strict_join_rejects_undeclared_dimension_and_family():
    unknown = _motion_spec()
    unknown["scenario_assumptions"] = {
        "weather": {
            "normalized_value": "rain",
            "evidence_quote": "We evaluate the planner in rain.",
            "paper_location": "Section V",
        }
    }
    errors = cross_check_scenario_assumptions(
        MethodSpec.model_validate(unknown), ROOT
    )
    assert "not declared" in errors[0]
    assert "weather" in errors[0]

    other_family = copy.deepcopy(unknown)
    other_family["comparison"]["classification"]["id"] = (
        "active_learning/batch_acquisition"
    )
    data_setup = other_family["critical_requirements"]["data_setup"]
    data_setup.update({
        "initial_labeled": 10,
        "batch_size": 5,
        "total_budget": 20,
        "num_rounds": 2,
    })
    errors = cross_check_scenario_assumptions(
        MethodSpec.model_validate(other_family), ROOT
    )
    assert "declares no scenario_assumption_dimensions" in errors[0]


def test_scenario_evidence_quote_floor(tmp_path):
    spec_path = tmp_path / "method_spec.json"
    paper_path = tmp_path / "paper.md"
    spec_path.write_text(json.dumps(_motion_spec()), encoding="utf-8")
    paper_path.write_text(
        "We assume all obstacles are\ncircular. The robot navigates "
        "through a crowd of pedestrians.",
        encoding="utf-8",
    )

    ok = subprocess.run(
        [
            sys.executable,
            "scripts/validate_method_spec.py",
            str(spec_path),
            "--paper-md",
            str(paper_path),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert ok.returncode == 0, ok.stderr

    paper_path.write_text(
        "This text contains neither recorded assumption.",
        encoding="utf-8",
    )
    bad = subprocess.run(
        [
            sys.executable,
            "scripts/validate_method_spec.py",
            str(spec_path),
            "--paper-md",
            str(paper_path),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert bad.returncode == 1
    assert "scenario-assumption evidence-quote floor failed" in bad.stderr
    assert "scenario_assumptions['obstacle_geometry']" in bad.stderr


def test_both_quote_floors_report_together_in_one_pass(tmp_path):
    """A floor that fail-fasts hides the next floor's failures and burns
    one fix dispatch per revelation (SRL 2026-07-28: three iterations
    each surfaced a new layer and the oscillation judge halted the run).
    Both floors must report in the same validator invocation."""
    spec = _motion_spec()
    spec["critical_requirements"]["param_glossary"] = [
        {
            "name": "eps_f",
            "meaning_quote": "A paraphrased definition of the threshold.",
            "paper_section": "Section III",
        }
    ]
    spec_path = tmp_path / "method_spec.json"
    paper_path = tmp_path / "paper.md"
    spec_path.write_text(json.dumps(spec), encoding="utf-8")
    paper_path.write_text(
        "This paper text contains none of the recorded quotes.",
        encoding="utf-8",
    )

    bad = subprocess.run(
        [
            sys.executable,
            "scripts/validate_method_spec.py",
            str(spec_path),
            "--paper-md",
            str(paper_path),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert bad.returncode == 1
    assert "param-glossary meaning-quote floor failed" in bad.stderr
    assert "param_glossary['eps_f']" in bad.stderr
    assert "scenario-assumption evidence-quote floor failed" in bad.stderr
    assert "scenario_assumptions['obstacle_geometry']" in bad.stderr
    assert "scenario_assumptions['agent_population']" in bad.stderr


def test_motion_planning_dimension_vocabulary_is_family_scoped_and_inherited():
    family_ids = {
        entry["id"]
        for entry in taxonomy.load_scenario_assumption_dimensions(
            "motion_planning"
        )
    }
    child_ids = {
        entry["id"]
        for entry in taxonomy.load_scenario_assumption_dimensions(
            "motion_planning/sampling_based"
        )
    }

    assert family_ids == {
        "obstacle_geometry",
        "agent_population",
        "obstacle_dynamics",
    }
    assert child_ids == family_ids
    assert (
        taxonomy.load_scenario_assumption_dimensions(
            "active_learning/batch_acquisition"
        )
        == []
    )


def test_report_renders_assumptions_beside_demo_description(tmp_path):
    run = tmp_path / "planner"
    pipeline = run / ".pipeline"
    pipeline.mkdir(parents=True)
    (pipeline / "method_spec.json").write_text(
        json.dumps(_motion_spec()), encoding="utf-8"
    )

    report = render_run_report(run)

    assert "## Demo setup" in report
    assert "### The paper's stated demo assumptions" in report
    assert "The notebook demonstrates **Test method** at smoke scale." in report
    assert "**Obstacle geometry**" in report
    assert 'Normalized paper constraint: `["circle"]`' in report
    assert "Evidence from Section IV-A" in report
    assert "We assume all obstacles are circular." in report
    assert "obstacle_geometry" not in report


def test_report_is_byte_identical_when_no_assumptions_are_declared(tmp_path):
    run = tmp_path / "baseline"
    pipeline = run / ".pipeline"
    pipeline.mkdir(parents=True)

    before = render_run_report(run)
    spec = _minimal_valid_spec(
        paradigm_id="active_learning/batch_acquisition"
    )
    spec["critical_requirements"]["data_setup"].update({
        "initial_labeled": 10,
        "batch_size": 5,
        "total_budget": 20,
        "num_rounds": 2,
    })
    (pipeline / "method_spec.json").write_text(
        json.dumps(spec), encoding="utf-8"
    )
    after = render_run_report(run)

    assert after == before


def test_agent_contracts_keep_slice_a_guidance_only():
    analyzer = (
        ROOT / ".opencode" / "agents" / "r2c-method-analyzer.md"
    ).read_text(encoding="utf-8")
    notebook = (
        ROOT / ".opencode" / "agents" / "r2c-notebook-generator.md"
    ).read_text(encoding="utf-8")

    assert "scenario_assumption_dimensions" in analyzer
    assert "Figure-only implications are excluded" in analyzer
    assert "not prove that the generated setup complies" in analyzer
    assert "scenario_assumptions" in notebook
    assert "guidance, not proof" in notebook
    assert "does not claim detector-backed compliance" in notebook
