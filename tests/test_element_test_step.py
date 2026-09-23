"""Stage 2.d per-element test generation step (R2C-024 block 2).

Driver-level contracts over `_run_element_test_step` with FakeDispatch:
the generator is dispatched once with the eligibility plan, the vacuity
floor gets its one corrective dispatch and then drops, failing tests
route through the ownership judge (code vs test, verbatim element text
as arbiter), the test side gets exactly one regeneration, and every
outcome lands in the coverage README plus the driver-owned results
artifact — the step never halts the stage over its own optional surface.
"""

from __future__ import annotations

import json
from pathlib import Path

from tests.helpers.state import make_state
import pytest

pytestmark = pytest.mark.probe_runtime

_METHOD_PY = '''\
def affine(x):
    # paper-element: eq-affine
    return 2 * x + 5
'''

_ELEMENTS = [
    {"id": "eq-affine", "type": "equation", "name": "Affine map",
     "section": "Sec. 3, Eq. (2)", "source_text": "y = 2x + 5",
     "pseudocode": "affine(x) = 2 * x + 5; worked example: x=3 -> y=11",
     "code_role": "implement"},
]

_GOOD_TEST = '''\
# paper-element: eq-affine
from method.method import affine


def test_affine_worked_example():
    assert affine(3) == 11
'''

_VACUOUS_TEST = '''\
# paper-element: eq-affine
from method.method import affine


def test_shape_only():
    assert len([affine(3)]) == 1
'''

# Floor-clean but asserts a value the element does not state: the code
# returns 11, the test demands 12 — the ownership judge decides who owns it.
_FAILING_TEST = '''\
# paper-element: eq-affine
from method.method import affine


def test_wrong_expectation():
    assert affine(3) == 11 + 1
'''


def _make_run(run_dir: Path) -> None:
    (run_dir / "method").mkdir(exist_ok=True)
    (run_dir / "method" / "method.py").write_text(_METHOD_PY)
    (run_dir / "method" / "__init__.py").write_text(
        "from .method import affine\n")
    (run_dir / ".pipeline" / "paper_map.json").write_text(json.dumps({
        "schema_version": "1.0.0", "title": "t", "elements": _ELEMENTS}))


def _decision(*, iteration: int, module: str, target: str | None,
              action: str = "dispatch_fix") -> dict:
    finding = None
    if action == "dispatch_fix":
        finding = {"id": "JUDGE001", "severity": "critical",
                   "description": "ownership decision",
                   "proposed_fix": "fix it"}
    return {
        "schema_version": "1.0.0",
        "stage_id": "stage_2d",
        "iteration": iteration,
        "validator_label": f"element_tests:{module}",
        "classification": "producer_fixable" if action == "dispatch_fix"
                          else "unclear",
        "action": action,
        "target_agent": target,
        "finding": finding,
        "rationale": "test rationale",
        "confidence": "high",
        "files_examined": ["method/tests/" + module],
    }


def _decision_write(*, iteration: int, module: str, decision: dict) -> dict:
    from run_pipeline import _judge_decision_scratch_relpath

    rel = _judge_decision_scratch_relpath(
        stage_id="stage_2d", iteration=iteration,
        validator_label=f"element_tests:{module}")
    return {rel: json.dumps(decision)}


def _run_step(run_dir: Path):
    from run_pipeline import _run_element_test_step

    return _run_element_test_step(make_state(run_dir))


def test_happy_path_ships_a_tested_element(run_dir, fake_dispatch):
    _make_run(run_dir)
    fake_dispatch.expect(
        agent="r2c-test-generator",
        writes={"method/tests/test_eq_affine.py": _GOOD_TEST})

    assert _run_step(run_dir) is None
    assert fake_dispatch.calls[0].agent == "r2c-test-generator"
    assert "eq-affine" in fake_dispatch.calls[0].prompt
    assert "worked example" in fake_dispatch.calls[0].prompt

    readme = (run_dir / "method" / "tests" / "README.md").read_text()
    assert "distinguishes correct" in readme  # mutant caught -> tested
    results = json.loads(
        (run_dir / ".pipeline" / "element_tests.json").read_text())
    assert results["statuses"]["eq-affine"]["status"] == "tested"
    assert results["dropped"] == {}


def test_no_eligible_elements_discloses_without_dispatch(run_dir,
                                                         fake_dispatch):
    _make_run(run_dir)
    (run_dir / ".pipeline" / "paper_map.json").write_text(json.dumps({
        "schema_version": "1.0.0", "title": "t", "elements": [dict(
            _ELEMENTS[0], code_role="demonstrate")]}))

    assert _run_step(run_dir) is None
    assert fake_dispatch.calls == []
    readme = (run_dir / "method" / "tests" / "README.md").read_text()
    assert "No implemented element" in readme


def test_floor_failure_gets_one_corrective_dispatch_then_ships(
        run_dir, fake_dispatch):
    _make_run(run_dir)
    fake_dispatch.expect(
        agent="r2c-test-generator",
        writes={"method/tests/test_eq_affine.py": _VACUOUS_TEST})
    fake_dispatch.expect(
        agent="r2c-test-generator",
        writes={"method/tests/test_eq_affine.py": _GOOD_TEST})

    assert _run_step(run_dir) is None
    assert [c.agent for c in fake_dispatch.calls] == [
        "r2c-test-generator", "r2c-test-generator"]
    assert "vacuity floor" in fake_dispatch.calls[1].prompt
    results = json.loads(
        (run_dir / ".pipeline" / "element_tests.json").read_text())
    assert results["statuses"]["eq-affine"]["status"] == "tested"


def test_floor_failure_persisting_drops_and_discloses(run_dir,
                                                      fake_dispatch):
    _make_run(run_dir)
    fake_dispatch.expect(
        agent="r2c-test-generator",
        writes={"method/tests/test_eq_affine.py": _VACUOUS_TEST})
    fake_dispatch.expect(agent="r2c-test-generator")  # fix writes nothing

    assert _run_step(run_dir) is None
    assert not (run_dir / "method" / "tests" / "test_eq_affine.py").exists()
    readme = (run_dir / "method" / "tests" / "README.md").read_text()
    assert "not shipped" in readme
    results = json.loads(
        (run_dir / ".pipeline" / "element_tests.json").read_text())
    assert "test_eq_affine.py" in results["dropped"]
    assert "vacuity floor" in results["dropped"]["test_eq_affine.py"]


def test_judged_test_defect_regenerates_once_then_ships(run_dir,
                                                        fake_dispatch):
    _make_run(run_dir)
    module = "test_eq_affine.py"
    fake_dispatch.expect(
        agent="r2c-test-generator",
        writes={f"method/tests/{module}": _FAILING_TEST})
    fake_dispatch.expect(
        agent="r2c-halt-judge",
        writes=_decision_write(
            iteration=0, module=module,
            decision=_decision(iteration=0, module=module,
                               target="r2c-test-generator")))
    fake_dispatch.expect(
        agent="r2c-test-generator",
        writes={f"method/tests/{module}": _GOOD_TEST})

    assert _run_step(run_dir) is None
    assert [c.agent for c in fake_dispatch.calls] == [
        "r2c-test-generator", "r2c-halt-judge", "r2c-test-generator"]
    results = json.loads(
        (run_dir / ".pipeline" / "element_tests.json").read_text())
    assert results["statuses"]["eq-affine"]["status"] == "tested"


def test_second_test_defect_ruling_drops_without_a_second_regen(
        run_dir, fake_dispatch):
    _make_run(run_dir)
    module = "test_eq_affine.py"
    fake_dispatch.expect(
        agent="r2c-test-generator",
        writes={f"method/tests/{module}": _FAILING_TEST})
    fake_dispatch.expect(
        agent="r2c-halt-judge",
        writes=_decision_write(
            iteration=0, module=module,
            decision=_decision(iteration=0, module=module,
                               target="r2c-test-generator")))
    fake_dispatch.expect(
        agent="r2c-test-generator",
        writes={f"method/tests/{module}": _FAILING_TEST})  # regen still bad
    fake_dispatch.expect(
        agent="r2c-halt-judge",
        writes=_decision_write(
            iteration=1, module=module,
            decision=_decision(iteration=1, module=module,
                               target="r2c-test-generator")))

    assert _run_step(run_dir) is None
    # Exactly one regeneration: the second test-side ruling drops instead.
    assert [c.agent for c in fake_dispatch.calls] == [
        "r2c-test-generator", "r2c-halt-judge", "r2c-test-generator",
        "r2c-halt-judge"]
    assert not (run_dir / "method" / "tests" / module).exists()
    results = json.loads(
        (run_dir / ".pipeline" / "element_tests.json").read_text())
    assert "one regeneration" in results["dropped"][module]
    readme = (run_dir / "method" / "tests" / "README.md").read_text()
    assert "not shipped" in readme


def test_judged_code_defect_routes_to_method_coder_and_reruns(
        run_dir, fake_dispatch):
    _make_run(run_dir)
    module = "test_eq_affine.py"
    # The delivered code is broken (returns 2x+6); the shipped GOOD test
    # correctly fails; the judge rules the code owns it.
    (run_dir / "method" / "method.py").write_text(
        "def affine(x):\n"
        "    # paper-element: eq-affine\n"
        "    return 2 * x + 6\n")
    fake_dispatch.expect(
        agent="r2c-test-generator",
        writes={f"method/tests/{module}": _GOOD_TEST})
    fake_dispatch.expect(
        agent="r2c-halt-judge",
        writes=_decision_write(
            iteration=0, module=module,
            decision=_decision(iteration=0, module=module,
                               target="r2c-method-coder")))
    fake_dispatch.expect(
        agent="r2c-method-coder",
        writes={"method/method.py": _METHOD_PY})  # the fix

    assert _run_step(run_dir) is None
    assert [c.agent for c in fake_dispatch.calls] == [
        "r2c-test-generator", "r2c-halt-judge", "r2c-method-coder"]
    results = json.loads(
        (run_dir / ".pipeline" / "element_tests.json").read_text())
    assert results["statuses"]["eq-affine"]["status"] == "tested"
    assert results["dropped"] == {}


# ---------------------------------------------------------------------------
# Per-element dispatch (2026-08-24): one dispatch per eligible element,
# failure isolation, and per-element resume.
# ---------------------------------------------------------------------------

_METHOD_TWO = '''\
def affine(x):
    # paper-element: eq-affine
    return 2 * x + 5


def cube(x):
    # paper-element: eq-cube
    return x ** 3
'''

_ELEMENTS_TWO = [
    _ELEMENTS[0],
    {"id": "eq-cube", "type": "equation", "name": "Cube map",
     "section": "Sec. 3, Eq. (3)", "source_text": "z = x^3",
     "pseudocode": "cube(x) = x ** 3; worked example: x=2 -> z=8",
     "code_role": "implement"},
]

_GOOD_CUBE_TEST = '''\
# paper-element: eq-cube
from method.method import cube


def test_cube_worked_example():
    assert cube(2) == 8
'''


def _make_two_element_run(run_dir: Path) -> None:
    (run_dir / "method").mkdir(exist_ok=True)
    (run_dir / "method" / "method.py").write_text(_METHOD_TWO)
    (run_dir / "method" / "__init__.py").write_text(
        "from .method import affine, cube\n")
    (run_dir / ".pipeline" / "paper_map.json").write_text(json.dumps({
        "schema_version": "1.0.0", "title": "t",
        "elements": _ELEMENTS_TWO}))


def test_each_element_gets_its_own_dispatch(run_dir, fake_dispatch):
    """The batch shape is gone: two eligible elements mean two dispatches,
    each carrying ONLY its own element's section (a timeout can then cost
    one element, never the plan)."""
    _make_two_element_run(run_dir)
    fake_dispatch.expect(
        agent="r2c-test-generator",
        writes={"method/tests/test_eq_affine.py": _GOOD_TEST})
    fake_dispatch.expect(
        agent="r2c-test-generator",
        writes={"method/tests/test_eq_cube.py": _GOOD_CUBE_TEST})

    assert _run_step(run_dir) is None
    gen_calls = [c for c in fake_dispatch.calls
                 if c.agent == "r2c-test-generator"]
    assert len(gen_calls) == 2
    assert "eq-affine" in gen_calls[0].prompt
    assert "eq-cube" not in gen_calls[0].prompt
    assert "eq-cube" in gen_calls[1].prompt
    assert "`eq-affine`" not in gen_calls[1].prompt
    results = json.loads(
        (run_dir / ".pipeline" / "element_tests.json").read_text())
    assert results["statuses"]["eq-affine"]["status"] == "tested"
    assert results["statuses"]["eq-cube"]["status"] == "tested"


def test_one_failed_dispatch_drops_one_element_not_the_stage(
        run_dir, fake_dispatch, monkeypatch):
    """Isolation: eq-affine's dispatch dies on transport, eq-cube still
    ships, the stage does not halt, and the missing element renders as
    not shipped in the coverage README."""
    import run_pipeline
    from opencode_client import OpencodeClientError

    _make_two_element_run(run_dir)
    fake_dispatch.expect(
        agent="r2c-test-generator",
        writes={"method/tests/test_eq_cube.py": _GOOD_CUBE_TEST})

    real_fake = run_pipeline.dispatch_agent

    def _raising(*, state, agent, prompt, timeout_s):
        if agent == "r2c-test-generator" and "eq-affine" in prompt:
            raise OpencodeClientError("dispatch POST timed out after 900s")
        return real_fake(state=state, agent=agent, prompt=prompt,
                         timeout_s=timeout_s)

    monkeypatch.setattr(run_pipeline, "dispatch_agent", _raising)

    assert _run_step(run_dir) is None  # no halt
    results = json.loads(
        (run_dir / ".pipeline" / "element_tests.json").read_text())
    assert results["statuses"]["eq-cube"]["status"] == "tested"
    assert "eq-affine" not in results["statuses"]
    readme = (run_dir / "method" / "tests" / "README.md").read_text()
    assert "eq-affine" in readme  # visible by construction, not shipped


def test_all_dispatches_failing_halts_as_systemic(run_dir, fake_dispatch,
                                                  monkeypatch):
    import run_pipeline
    from opencode_client import OpencodeClientError

    _make_two_element_run(run_dir)

    def _always_raising(*, state, agent, prompt, timeout_s):
        raise OpencodeClientError("connection refused")

    monkeypatch.setattr(run_pipeline, "dispatch_agent", _always_raising)

    result = _run_step(run_dir)
    assert result is not None and result.status == "halted"
    assert "all 2 elements" in (result.halt_artifact or {}).get("reason", "")


def test_resume_skips_elements_with_floor_passing_modules(run_dir,
                                                          fake_dispatch):
    """Per-element resume: a floor-passing module left by a prior roll is
    kept and its element never re-dispatched; only the missing element
    dispatches."""
    _make_two_element_run(run_dir)
    tests_dir = run_dir / "method" / "tests"
    tests_dir.mkdir(parents=True)
    (tests_dir / "test_eq_affine.py").write_text(_GOOD_TEST)
    fake_dispatch.expect(
        agent="r2c-test-generator",
        writes={"method/tests/test_eq_cube.py": _GOOD_CUBE_TEST})

    assert _run_step(run_dir) is None
    gen_calls = [c for c in fake_dispatch.calls
                 if c.agent == "r2c-test-generator"]
    assert len(gen_calls) == 1
    assert "eq-cube" in gen_calls[0].prompt
    results = json.loads(
        (run_dir / ".pipeline" / "element_tests.json").read_text())
    assert results["statuses"]["eq-affine"]["status"] == "tested"
    assert results["statuses"]["eq-cube"]["status"] == "tested"


def test_resume_regenerates_element_whose_module_is_below_floor(
        run_dir, fake_dispatch):
    """A leftover module below the vacuity floor is deleted up front and
    its element regenerated fresh, instead of riding the corrective-fix
    path with a stale module in place."""
    _make_two_element_run(run_dir)
    tests_dir = run_dir / "method" / "tests"
    tests_dir.mkdir(parents=True)
    (tests_dir / "test_eq_affine.py").write_text(_VACUOUS_TEST)
    fake_dispatch.expect(
        agent="r2c-test-generator",
        writes={"method/tests/test_eq_affine.py": _GOOD_TEST})
    fake_dispatch.expect(
        agent="r2c-test-generator",
        writes={"method/tests/test_eq_cube.py": _GOOD_CUBE_TEST})

    assert _run_step(run_dir) is None
    gen_calls = [c for c in fake_dispatch.calls
                 if c.agent == "r2c-test-generator"]
    assert len(gen_calls) == 2
    results = json.loads(
        (run_dir / ".pipeline" / "element_tests.json").read_text())
    assert results["statuses"]["eq-affine"]["status"] == "tested"
