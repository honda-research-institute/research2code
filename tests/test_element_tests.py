"""Deterministic core of per-element generated tests (R2C-024 block 1).

Pins the approved guards: the join table bounds eligibility honestly,
the vacuity floor rejects green-but-empty tests, the runner leaves the
run tree untouched, the advisory mutation check labels weak tests
instead of dropping them, and the coverage README discloses every
omission by construction.
"""

from __future__ import annotations

import json
from pathlib import Path

from element_tests import (
    eligible_elements,
    main as element_tests_main,
    mutate_function_source,
    mutation_check,
    render_coverage_readme,
    run_test_module,
    validate_test_module,
)

_METHOD_PY = '''\
def affine(x):
    # paper-element: eq-affine
    return 2 * x + 5


# paper-element: alg-pipeline


def demo_curve(x):
    # paper-element: eq-demo-curve
    return x * x


def largest(a, b):
    # paper-element: alg-largest
    return a if a > b else b


def identity(x):
    # paper-element: concept-identity
    return x
'''

_ELEMENTS = [
    {"id": "eq-affine", "type": "equation", "name": "Affine map",
     "section": "Sec. 3, Eq. (2)",
     "source_text": "y = 2x + 5",
     "pseudocode": "affine(x) = 2 * x + 5; worked example: x=3 -> y=11",
     "code_role": "implement"},
    {"id": "alg-pipeline", "type": "algorithm", "name": "Pipeline",
     "section": "Sec. 4", "source_text": "", "pseudocode": "",
     "code_role": "implement"},
    {"id": "eq-demo-curve", "type": "equation", "name": "Demo curve",
     "section": "Sec. 5", "source_text": "y = x^2", "pseudocode": "",
     "code_role": "demonstrate"},
    {"id": "alg-largest", "type": "algorithm", "name": "Largest",
     "section": "Sec. 4.1", "source_text": "",
     "pseudocode": "largest(a, b): return a if a > b else b",
     "code_role": "implement"},
    {"id": "concept-identity", "type": "concept", "name": "Identity",
     "section": "Sec. 2", "source_text": "", "pseudocode": "",
     "code_role": "implement"},
]


def _make_run(tmp_path: Path) -> Path:
    run = tmp_path / "run"
    (run / ".pipeline").mkdir(parents=True)
    (run / "method").mkdir()
    (run / "method" / "method.py").write_text(_METHOD_PY)
    (run / "method" / "__init__.py").write_text(
        "from .method import affine, demo_curve, largest, identity\n")
    (run / ".pipeline" / "paper_map.json").write_text(json.dumps({
        "schema_version": "1.0.0", "title": "t", "elements": _ELEMENTS}))
    return run


def _write_test(run: Path, name: str, body: str) -> Path:
    tests_dir = run / "method" / "tests"
    tests_dir.mkdir(exist_ok=True)
    module = tests_dir / name
    module.write_text(body)
    return module


_GOOD_TEST = '''\
# paper-element: eq-affine
from method.method import affine


def test_affine_worked_example():
    assert affine(3) == 11
'''


def test_eligibility_is_bounded_by_role_and_function_mapping(tmp_path):
    plan = eligible_elements(_make_run(tmp_path))
    # Implemented + function-mapped elements are in; the module-level
    # anchor (no single function) and the demonstrate-role element are out.
    assert set(plan) == {"eq-affine", "alg-largest", "concept-identity"}
    assert plan["eq-affine"]["qualname"] == "affine"
    assert plan["eq-affine"]["file"] == "method/method.py"
    assert "worked example" in plan["eq-affine"]["pseudocode"]


def test_floor_accepts_a_grounded_value_level_test(tmp_path):
    run = _make_run(tmp_path)
    module = _write_test(run, "test_eq_affine.py", _GOOD_TEST)
    element_id, problems = validate_test_module(
        module, eligible_elements(run))
    assert element_id == "eq-affine"
    assert problems == []


def test_floor_rejects_the_vacuous_shapes(tmp_path):
    run = _make_run(tmp_path)
    plan = eligible_elements(run)

    # No anchor at all.
    m = _write_test(run, "test_a.py",
                    "def test_x():\n    assert 11 == 11\n")
    _, problems = validate_test_module(m, plan)
    assert any("exactly one" in p for p in problems)

    # Anchored to an ineligible element.
    m = _write_test(run, "test_b.py",
                    "# paper-element: eq-demo-curve\n"
                    "def test_x():\n    assert 4 == 4\n")
    element_id, problems = validate_test_module(m, plan)
    assert element_id == "eq-demo-curve"
    assert any("not an eligible" in p for p in problems)

    # Never calls the mapped function.
    m = _write_test(run, "test_c.py",
                    "# paper-element: eq-affine\n"
                    "def test_x():\n    assert 11 == 11\n")
    _, problems = validate_test_module(m, plan)
    assert any("never calls 'affine'" in p for p in problems)

    # Shape-only assertions do not verify a claim.
    m = _write_test(run, "test_d.py",
                    "# paper-element: eq-affine\n"
                    "import numpy as np\n"
                    "from method.method import affine\n"
                    "def test_x():\n"
                    "    y = np.array([affine(3)])\n"
                    "    assert y.shape == (1,)\n"
                    "    assert len(y) == 1\n")
    _, problems = validate_test_module(m, plan)
    assert any("value-level" in p for p in problems)

    # Constants untraceable to the element's stated numbers.
    m = _write_test(run, "test_e.py",
                    "# paper-element: eq-affine\n"
                    "from method.method import affine\n"
                    "def test_x():\n    assert affine(10) == 25\n")
    _, problems = validate_test_module(m, plan)
    assert any("traces" in p for p in problems)

    # An element whose pseudocode states no usable numbers demands none.
    m = _write_test(run, "test_f.py",
                    "# paper-element: alg-largest\n"
                    "from method.method import largest\n"
                    "def test_x():\n    assert largest(10, 25) == 25\n")
    _, problems = validate_test_module(m, plan)
    assert problems == []


def test_runner_reports_honestly_and_writes_nothing(tmp_path):
    run = _make_run(tmp_path)
    good = _write_test(run, "test_eq_affine.py", _GOOD_TEST)
    assert run_test_module(run, good)["status"] == "passed"

    bad = _write_test(run, "test_bad.py",
                      "# paper-element: eq-affine\n"
                      "from method.method import affine\n"
                      "def test_x():\n    assert affine(3) == 12\n")
    result = run_test_module(run, bad)
    assert result["status"] == "failed"
    assert "test_x" in result["detail"]

    leftovers = [p for p in run.rglob("*")
                 if p.name in ("__pycache__", ".pytest_cache")]
    assert leftovers == [], "runner littered the run tree"


def test_mutation_check_separates_tested_from_weak(tmp_path):
    run = _make_run(tmp_path)
    plan = eligible_elements(run)

    # A value-pinning test catches the perturbed constant.
    good = _write_test(run, "test_eq_affine.py", _GOOD_TEST)
    verdict = mutation_check(run, plan["eq-affine"], good)
    assert verdict["status"] == "caught"
    assert "numeric constant" in verdict["detail"]

    # A monotonicity-only test survives the same mutant: floor-clean,
    # but it cannot distinguish correct from broken code.
    weak = _write_test(run, "test_weak.py",
                       "# paper-element: eq-affine\n"
                       "from method.method import affine\n"
                       "def test_x():\n    assert affine(3) > affine(2)\n")
    assert mutation_check(run, plan["eq-affine"], weak)["status"] == "survived"

    # No constant in the function: the comparison operator flips instead.
    source = (run / "method" / "method.py").read_text()
    mutated = mutate_function_source(source, "largest")
    assert mutated is not None and "comparison" in mutated[1]
    flip = _write_test(run, "test_largest.py",
                       "# paper-element: alg-largest\n"
                       "from method.method import largest\n"
                       "def test_x():\n    assert largest(2, 5) == 5\n")
    assert mutation_check(run, plan["alg-largest"], flip)["status"] == "caught"

    # Neither mutation class applies to a bare pass-through.
    assert mutate_function_source(source, "identity") is None
    ident = _write_test(run, "test_ident.py",
                        "# paper-element: concept-identity\n"
                        "from method.method import identity\n"
                        "def test_x():\n    assert identity(4) == 4\n")
    assert mutation_check(
        run, plan["concept-identity"], ident)["status"] == "not_applicable"


def test_mutation_check_never_touches_the_run_tree(tmp_path):
    run = _make_run(tmp_path)
    plan = eligible_elements(run)
    module = _write_test(run, "test_eq_affine.py", _GOOD_TEST)
    before = {p: p.read_bytes() for p in run.rglob("*") if p.is_file()}
    mutation_check(run, plan["eq-affine"], module)
    after = {p: p.read_bytes() for p in run.rglob("*") if p.is_file()}
    assert after == before


def test_coverage_readme_discloses_every_eligible_element(tmp_path):
    plan = eligible_elements(_make_run(tmp_path))
    readme = render_coverage_readme(plan, {
        "eq-affine": {"status": "tested", "module": "test_eq_affine.py"},
        "alg-largest": {"status": "weak", "module": "test_weak.py"},
    })
    assert "`eq-affine`" in readme and "distinguishes correct" in readme
    assert "Weak test" in readme
    # The element nobody shipped a test for is disclosed, not dropped.
    assert "`concept-identity`" in readme
    assert "not shipped" in readme
    # the researcher's two-level framing.
    assert "integration test" in readme

    empty = render_coverage_readme({}, {})
    assert "No implemented element" in empty


# ---------------------------------------------------------------------------
# Block 2a registration surface: the test-generator dispatch vocabulary.
# (The judge literal/table consistency itself is pinned in test_judge.py.)
# ---------------------------------------------------------------------------

def test_generator_dispatch_surface_is_registered(tmp_path):
    from dispatch_templates import (
        JUDGE_ELEMENT_TEST_TEMPLATE,
        STAGE_TASK_SUMMARIES,
        WRITEABLE_PATHS,
    )
    from run_pipeline import (
        JUDGE_FIX_DISPATCHERS,
        _dispatch_test_generator_fix,
        _element_plan_sections,
        _element_tests_written,
    )
    from tests.helpers.state import make_state

    assert WRITEABLE_PATHS["r2c-test-generator"] == ["method/tests/test_*.py"]
    assert "value-level" in STAGE_TASK_SUMMARIES["stage_2d_tests"].lower()
    # pdwa 2026-07-27 dropped 19 of 21 modules because the generator wrote
    # abbreviated anchors (`eq-j-c` for `eq-j-clearance`). The summary must
    # state that the anchor id is copied exactly and that a mismatch drops
    # the module, not merely that an anchor comment is required.
    summary_2d = STAGE_TASK_SUMMARIES["stage_2d_tests"].lower()
    assert "character-for-character" in summary_2d
    assert "abbreviate" in summary_2d
    assert "drops the whole module" in summary_2d
    assert JUDGE_FIX_DISPATCHERS["r2c-test-generator"] is (
        _dispatch_test_generator_fix)

    # The judge template formats with the ownership tie-breaker intact.
    rendered = JUDGE_ELEMENT_TEST_TEMPLATE.format(
        stage_id="stage_2d", iteration=0,
        validator_label="element_tests:test_eq_affine.py",
        decision_output_path="/x/decision.json",
        element_id="eq-affine", element_block="y = 2x + 5",
        test_module_path="method/tests/test_eq_affine.py",
        qualname="affine", target_file="method/method.py",
        failure_tail="assert 12 == 11", prior_decisions_block="[]")
    assert "ARBITER" in rendered and "ONE regeneration" in rendered
    assert "r2c-method-coder" in rendered
    assert "r2c-test-generator" in rendered

    # Plan sections carry the element statement the assertions come from.
    run = _make_run(tmp_path)
    plan = eligible_elements(run)
    (section,) = _element_plan_sections(plan)
    assert "`eq-affine`" in section and "worked example" in section
    assert "`affine`" in section

    # Recovery check: parseable module present vs absent/broken.
    state = make_state(run)
    assert _element_tests_written(state) is False
    _write_test(run, "test_eq_affine.py", _GOOD_TEST)
    assert _element_tests_written(state) is True
    _write_test(run, "test_eq_affine.py", "def broken(:\n")
    assert _element_tests_written(state) is False


def test_review_vocabulary_carries_the_test_branch():
    import typing

    from schemas.judge_decision import TargetAgent as JudgeTarget
    from schemas.review_report import IssueType, TargetAgent

    assert "test_defect" in typing.get_args(IssueType)
    assert "test-generator" in typing.get_args(TargetAgent)
    assert "r2c-test-generator" in typing.get_args(JudgeTarget)


# ---------------------------------------------------------------------------
# Block 3: delivery registration + the real-artifact eligibility arm.
# ---------------------------------------------------------------------------

def test_delivery_registries_carry_the_test_surface():
    from final_manifest import ARTIFACT_SPECS
    from fleet_state import ARTIFACT_WHITELIST
    from run_history import _TERMINAL_ARTIFACTS

    by_path = {s.rel_path: s for s in ARTIFACT_SPECS}
    readme = by_path["method/tests/README.md"]
    results = by_path[".pipeline/element_tests.json"]
    # Optional by construction: absence must never block a delivery.
    assert readme.required_for_delivery is False
    assert results.required_for_delivery is False
    assert readme.producer_stage == "stage_2d"
    assert ARTIFACT_WHITELIST["TESTS.md"] == "method/tests/README.md"
    assert (".pipeline/element_tests.json"
            in {src for _, src in _TERMINAL_ARTIFACTS})


def test_report_row_summarizes_coverage_honestly(tmp_path):
    from render_run_report import _issue_summary as build_issues_section

    run = _make_run(tmp_path)
    (run / ".pipeline" / "element_tests.json").write_text(json.dumps({
        "schema_version": "1.0.0",
        "eligible": ["eq-affine", "alg-largest", "concept-identity"],
        "statuses": {
            "eq-affine": {"status": "tested", "module": "test_eq_affine.py"},
            "alg-largest": {"status": "weak", "module": "test_largest.py"},
        },
        "dropped": {"test_ident.py": "vacuity floor"},
    }))
    section = build_issues_section(run)
    assert "Per-element tests" in section
    assert "1 of 3 eligible elements" in section
    assert "1 weak test" in section
    assert "1 generated test could not be verified" in section
    # No results record, no row — never an implied claim.
    bare = _make_run(tmp_path / "bare")
    assert "Per-element tests" not in build_issues_section(bare)


def test_eligibility_binds_on_a_real_delivered_run(tmp_path):
    """The gbald evidence fixture carries a real paper map (36 elements)
    and a real anchored method package — the plan must bind real anchors
    to real functions, not just the synthetic fixture's."""
    import shutil

    evidence = Path(__file__).parent / "fixtures" / "evidence" / "june9-gbald-run"
    run = tmp_path / "gbald"
    run.mkdir()
    shutil.copytree(evidence / "method", run / "method")
    (run / ".pipeline").mkdir()
    shutil.copy2(evidence / "pipeline" / "paper_map.json",
                 run / ".pipeline" / "paper_map.json")

    plan = eligible_elements(run)
    assert plan, "a real green run must yield a non-empty plan"
    for entry in plan.values():
        assert (run / entry["file"]).is_file()
        assert entry["qualname"]
    assert "eq-bald-score" in plan
    assert plan["eq-bald-score"]["pseudocode"]


def test_main_end_to_end_over_a_good_module(tmp_path, capsys):
    run = _make_run(tmp_path)
    _write_test(run, "test_eq_affine.py", _GOOD_TEST)
    rc = element_tests_main(["--run-dir", str(run), "--write-readme"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "1 tested / 1 shipped / 3 eligible" in out
    readme = (run / "method" / "tests" / "README.md").read_text()
    assert "`eq-affine`" in readme and "`alg-largest`" in readme
