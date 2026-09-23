"""Scenario-fidelity slice B (R2C-025): detectors, comparators, the bounded
setup-slice harness, taxonomy gating, and delivery-label semantics.

The five acceptance shapes from the approved design's validation sketch:
a circles-only assumption with a polygon setup fails and demotes, the
matching setup passes, captured-but-unbindable flags and demotes, nothing
captured reads as an undetermined disclosure without demotion, and a
non-declaring family stays byte-identical (no bindings, no rows)."""

from __future__ import annotations

import json
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parent.parent
ZOO = REPO / "tests" / "fixtures" / "zoo"

from delivery_label import is_contribution_probe  # noqa: E402
from delivery_label import _probe_reasons  # noqa: E402
from probes.catalog import PROBE_CATALOG  # noqa: E402
from probes.motion_planning import (  # noqa: E402
    SCENARIO_DETECTORS,
    compare_scenario_dynamics,
    compare_scenario_geometry,
    compare_scenario_population,
    observe_scenario_dynamics,
    observe_scenario_geometry,
    observe_scenario_population,
)
from probes.scenario_setup import (  # noqa: E402
    resolve_scenario_detector,
    scenario_fidelity_verdicts,
    setup_code_cells,
)
from run_probes import (  # noqa: E402
    _node_declared_probe_context,
    _scenario_dimension_bindings,
)

MP_BINDINGS = _scenario_dimension_bindings(
    _node_declared_probe_context("motion_planning"))


def _fixture_verdicts(name: str) -> dict[str, object]:
    pairs = scenario_fidelity_verdicts(ZOO / name, MP_BINDINGS)
    return {v.probe_id: v for v, _ in pairs}


# ---------------------------------------------------------------------------
# The five acceptance shapes, on the zoo fixtures
# ---------------------------------------------------------------------------


def test_geometry_mismatch_fixture_fails_and_names_both_shapes():
    verdicts = _fixture_verdicts("mp-scenario-geometry-mismatch-reconstructed")
    sc1 = verdicts["SC-1"]
    assert sc1.verdict == "fail", sc1.message
    assert sc1.reason == "scenario_mismatch"
    assert sc1.finding_class == "M-006"
    assert "polygon" in sc1.message and "circle" in sc1.message
    # The other two dimensions were not captured: disclosures, never demoters.
    assert verdicts["SC-2"].verdict == "unprobeable"
    assert verdicts["SC-2"].reason == "no_captured_assumption"
    assert verdicts["SC-3"].verdict == "unprobeable"


def test_geometry_match_fixture_passes_and_demo_sections_never_execute():
    # Sections 4 and 5 of the fixture notebook are poison cells (raise
    # RuntimeError): executing them would set setup_error, which demotes a
    # pass to flag_for_researcher — so this pass also proves the slicing.
    verdicts = _fixture_verdicts("mp-scenario-geometry-match-reconstructed")
    sc1 = verdicts["SC-1"]
    assert sc1.verdict == "pass", sc1.message
    assert sc1.reason == "scenario_match"


def test_unbindable_fixture_flags_for_researcher():
    verdicts = _fixture_verdicts("mp-scenario-unbindable-reconstructed")
    sc1 = verdicts["SC-1"]
    assert sc1.verdict == "flag_for_researcher", sc1.message
    assert sc1.reason == "setup_unbindable"
    assert "researcher judgment" in sc1.message


def test_absent_assumption_fixture_discloses_all_dimensions():
    verdicts = _fixture_verdicts("mp-scenario-absent-assumption-reconstructed")
    assert {v.verdict for v in verdicts.values()} == {"unprobeable"}
    assert {v.reason for v in verdicts.values()} == {"no_captured_assumption"}


def test_non_declaring_family_yields_no_bindings():
    context = _node_declared_probe_context("active_learning/batch_acquisition")
    assert _scenario_dimension_bindings(context) == []
    # And an unserved paradigm (full-battery fallback) has no bindings either.
    assert _scenario_dimension_bindings(None) == []


# ---------------------------------------------------------------------------
# Delivery-label semantics: which arms demote
# ---------------------------------------------------------------------------


def test_fail_and_flag_demote_while_unprobeable_only_discloses():
    mismatch = _fixture_verdicts("mp-scenario-geometry-mismatch-reconstructed")
    unbindable = _fixture_verdicts("mp-scenario-unbindable-reconstructed")
    report = {"verdicts": [
        mismatch["SC-1"].to_dict(),     # fail
        unbindable["SC-1"].to_dict(),   # flag_for_researcher
        mismatch["SC-2"].to_dict(),     # unprobeable
    ]}
    reasons, disclosures = _probe_reasons(report)
    assert [r["id"] for r in reasons] == ["SC-1", "SC-1"]
    assert [d["id"] for d in disclosures] == ["SC-2"]


def test_scenario_probes_are_not_contribution_tier():
    # A scenario match is evidence about the demo scene, not the method
    # contribution — SC ids must never satisfy the verified label's
    # contribution-evidence requirement.
    for probe_id in ("SC-1", "SC-2", "SC-3"):
        assert not is_contribution_probe(probe_id)


# ---------------------------------------------------------------------------
# Structural: pack declarations resolve, and the walker matches the SSOT
# ---------------------------------------------------------------------------


def _declared_dimensions() -> list[dict]:
    doc = yaml.safe_load(
        (REPO / "docs" / "ssot" / "taxonomies.yaml").read_text(encoding="utf-8"))

    def walk(node):
        if isinstance(node, dict):
            for key, value in node.items():
                if key == "scenario_assumption_dimensions" and isinstance(value, list):
                    yield from (v for v in value if isinstance(v, dict))
                else:
                    yield from walk(value)
        elif isinstance(node, list):
            for item in node:
                yield from walk(item)

    return list(walk(doc))


def test_every_declared_detector_resolves_to_a_cataloged_executor():
    dimensions = _declared_dimensions()
    assert dimensions, "the taxonomy declares no scenario dimensions"
    for entry in dimensions:
        ref = entry.get("detector")
        detector = resolve_scenario_detector(ref)
        assert detector is not None, (
            f"dimension {entry.get('id')!r} declares detector {ref!r} "
            f"with no registered executor")
        assert detector.probe_id in PROBE_CATALOG
        assert entry.get("check") and entry.get("silent_failure")


def test_registry_refs_match_their_family_module():
    for ref, detector in SCENARIO_DETECTORS.items():
        assert ref.startswith("motion_planning.")
        assert resolve_scenario_detector(ref) is detector


def test_setup_slice_walker_agrees_with_the_demo_verdict_slicer():
    # probes/scenario_setup.py reimplements the section walk dependency-free
    # (the vendored harness cannot import scripts/); this pins the two
    # implementations to each other on a committed layout.
    from demo_verdict import (run_demo_context, setup_section_boundary,
                              setup_section_source_blob)

    run = ZOO / "mp-scenario-geometry-match-reconstructed"
    nb = json.loads((run / "notebook.ipynb").read_text(encoding="utf-8"))
    _, demo_success, layout, _ = run_demo_context(run)
    bound = setup_section_boundary(layout, demo_success)
    mine = "\n# --- cell ---\n".join(
        src for _, _, src in setup_code_cells(nb, bound))
    assert mine == setup_section_source_blob(run)


def test_frozen_context_round_trip_preserves_bindings():
    # The vendored harness reads the declared context from frozen JSON; the
    # dimension join key must survive the round trip.
    context = _node_declared_probe_context("motion_planning")
    thawed = json.loads(json.dumps(context))
    assert _scenario_dimension_bindings(thawed) == MP_BINDINGS
    assert MP_BINDINGS == [
        ("agent_population", "motion_planning.scenario_population"),
        ("obstacle_dynamics", "motion_planning.scenario_dynamics_setup"),
        ("obstacle_geometry", "motion_planning.scenario_geometry"),
    ]


# ---------------------------------------------------------------------------
# Harness edge arms: partial setup, uninspectable setup
# ---------------------------------------------------------------------------


def _write_run(tmp_path: Path, setup_cells: list[str],
               assumptions: dict | None) -> Path:
    run = tmp_path / "run"
    (run / ".pipeline").mkdir(parents=True)
    spec: dict = {"comparison": {"classification": {"id": "motion_planning"}}}
    if assumptions is not None:
        spec["scenario_assumptions"] = assumptions
    (run / ".pipeline" / "method_spec.json").write_text(
        json.dumps(spec), encoding="utf-8")
    cells = [{"cell_type": "markdown", "metadata": {},
              "source": ["## 3. The setup pieces\n"]}]
    cells += [{"cell_type": "code", "metadata": {}, "outputs": [],
               "execution_count": None, "source": src.splitlines(keepends=True)}
              for src in setup_cells]
    (run / "notebook.ipynb").write_text(json.dumps(
        {"cells": cells, "metadata": {}, "nbformat": 4, "nbformat_minor": 5}),
        encoding="utf-8")
    return run


GEOMETRY_ONLY = {"obstacle_geometry": {
    "normalized_value": "circle",
    "evidence_quote": "We model all obstacles as circles.",
    "paper_location": "Section III-B",
}}
GEOMETRY_BINDING = [("obstacle_geometry", "motion_planning.scenario_geometry")]


def test_partial_setup_demotes_a_match_to_researcher_judgment(tmp_path):
    run = _write_run(tmp_path, [
        'obstacles = [{"type": "circle", "x": 1.0, "y": 1.0, "radius": 0.3}]\n',
        'raise ValueError("the rest of the setup is broken")\n',
    ], GEOMETRY_ONLY)
    (verdict, _), = scenario_fidelity_verdicts(run, GEOMETRY_BINDING)
    assert verdict.verdict == "flag_for_researcher", verdict.message
    assert verdict.reason == "setup_incomplete"
    assert "did not finish executing" in verdict.message


def test_partial_setup_keeps_an_affirmative_contradiction(tmp_path):
    run = _write_run(tmp_path, [
        'obstacles = [{"type": "polygon", "vertices": [[0, 0], [1, 0], [1, 1]]}]\n',
        'raise ValueError("the rest of the setup is broken")\n',
    ], GEOMETRY_ONLY)
    (verdict, _), = scenario_fidelity_verdicts(run, GEOMETRY_BINDING)
    assert verdict.verdict == "fail", verdict.message
    assert verdict.reason == "scenario_mismatch"
    assert "before the slice finished" in verdict.message


def test_uninspectable_setup_flags_every_captured_dimension(tmp_path):
    run = _write_run(tmp_path, [
        "import time\ntime.sleep(30)\n",
    ], GEOMETRY_ONLY)
    (verdict, _), = scenario_fidelity_verdicts(
        run, GEOMETRY_BINDING, timeout_s=2)
    assert verdict.verdict == "flag_for_researcher", verdict.message
    assert verdict.reason == "setup_uninspectable"
    assert "could not be inspected" in verdict.message


def test_missing_notebook_flags_instead_of_passing(tmp_path):
    run = tmp_path / "run"
    (run / ".pipeline").mkdir(parents=True)
    (run / ".pipeline" / "method_spec.json").write_text(json.dumps({
        "comparison": {"classification": {"id": "motion_planning"}},
        "scenario_assumptions": GEOMETRY_ONLY,
    }), encoding="utf-8")
    (verdict, _), = scenario_fidelity_verdicts(run, GEOMETRY_BINDING)
    assert verdict.verdict == "flag_for_researcher"
    assert verdict.reason == "setup_uninspectable"


# ---------------------------------------------------------------------------
# Pure comparator arms (no subprocess)
# ---------------------------------------------------------------------------


def test_geometry_aliases_and_list_values():
    obs = observe_scenario_geometry({
        "obstacles": [{"type": "circles", "x": 0, "y": 0, "radius": 1}]})
    assert compare_scenario_geometry("circle", obs).verdict == "pass"
    assert compare_scenario_geometry(["rectangle"], obs).verdict == "fail"
    assert compare_scenario_geometry({"weird": 1}, obs).reason == (
        "assumption_uninterpretable")


def test_geometry_unclassified_records_flag():
    obs = observe_scenario_geometry({"obstacles": [{"blob": 42}]})
    comparison = compare_scenario_geometry("circle", obs)
    assert comparison.verdict == "flag_for_researcher"
    assert comparison.reason == "setup_unbindable"


def test_geometry_reads_environment_contract_and_local_lists():
    class Env:
        def __init__(self):
            self.obstacles = [{"xmin": 0, "xmax": 1, "ymin": 0, "ymax": 1}]

    obs = observe_scenario_geometry({
        "env": Env(),
        "obstacle_states": [{"x": 0.0, "y": 0.0, "radius": 0.5}],
    })
    assert obs["bound"]
    assert obs["geometry_types"] == ["circle", "rectangle"]
    assert set(obs["sources"]) == {"env.obstacles", "obstacle_states"}


def test_population_count_match_mismatch_and_ambiguity():
    exact = observe_scenario_population({"pedestrians": [1, 2, 3]})
    assert compare_scenario_population(3, exact).verdict == "pass"
    assert compare_scenario_population(5, exact).verdict == "fail"
    ambiguous = observe_scenario_population({
        "pedestrians": [1, 2, 3], "agents": [1]})
    comparison = compare_scenario_population(3, ambiguous)
    assert comparison.verdict == "flag_for_researcher"
    assert comparison.reason == "ambiguous_binding"


def test_population_mapping_presence_and_zero():
    zero = observe_scenario_population({"pedestrians": []})
    mapping = {"agent_type": "pedestrian", "minimum_present": True}
    assert compare_scenario_population(mapping, zero).verdict == "fail"
    some = observe_scenario_population({"pedestrians": [object(), object()]})
    assert compare_scenario_population(mapping, some).verdict == "pass"
    unbound = observe_scenario_population({"unrelated": 3})
    assert compare_scenario_population(mapping, unbound).verdict == (
        "flag_for_researcher")


def test_population_category_presence():
    some = observe_scenario_population({"pedestrians": [object()]})
    assert compare_scenario_population("crowded", some).verdict == "pass"
    zero = observe_scenario_population({"pedestrians": []})
    assert compare_scenario_population("crowded", zero).verdict == "fail"
    assert compare_scenario_population("none", zero).verdict == "pass"
    assert compare_scenario_population("none", some).verdict == "fail"


def test_dynamics_motion_state_arms():
    moving = observe_scenario_dynamics({
        "obstacle_states": [{"x": 0, "y": 0, "x_prev": 0, "y_prev": 0}]})
    assert compare_scenario_dynamics("dynamic", moving).verdict == "pass"
    static_paper = compare_scenario_dynamics("static", moving)
    assert static_paper.verdict == "flag_for_researcher"
    assert static_paper.reason == "static_claim_motion_state"

    still = observe_scenario_dynamics({
        "obstacles": [{"type": "rectangle", "xmin": 0, "xmax": 1,
                       "ymin": 0, "ymax": 1}]})
    assert compare_scenario_dynamics("static", still).verdict == "pass"
    # Deliberate ownership split: setup-time state cannot decide loop
    # motion, so the dynamic claim defers to MP-1 as a disclosure instead of
    # wrongly demoting loop-mutating deliveries.
    deferred = compare_scenario_dynamics("dynamic", still)
    assert deferred.verdict == "unprobeable"
    assert deferred.reason == "setup_state_motion_undecidable"
    assert "loop check" in deferred.message
    assert compare_scenario_dynamics("sideways", still).reason == (
        "assumption_uninterpretable")


# ---------------------------------------------------------------------------
# Battery integration: gating and pack-context stamping
# ---------------------------------------------------------------------------


def test_battery_stamps_pack_context_on_scenario_verdicts():
    from run_probes import run_battery

    report = run_battery(ZOO / "mp-scenario-geometry-mismatch-reconstructed")
    scenario = {v.probe_id: v for v in report.verdicts
                if v.probe_id.startswith("SC-")}
    assert set(scenario) == {"SC-1", "SC-2", "SC-3"}
    sc1 = scenario["SC-1"]
    assert sc1.verdict == "fail"
    assert sc1.pack_check_id == "scenario-obstacle_geometry"
    assert "obstacle shapes stay within" in sc1.pack_check
    assert "polygon obstacles" in sc1.pack_why
