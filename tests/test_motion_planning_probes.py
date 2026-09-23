"""Zoo acceptance tests for the motion-planning probes: the static
frozen-obstacles check (MP-1, finding class M-004) and the behavioral
steering-responsiveness (MP-3) + goal-progress (MP-4) pair."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
ZOO = REPO / "tests" / "fixtures" / "zoo"

from probes.motion_planning import (  # noqa: E402
    notebook_claims_dynamic_obstacles,
    probe_scenario_dynamics_static,
)

pytestmark = pytest.mark.probe_runtime


def test_mp1_passes_the_moving_obstacle_snapshot():
    v = probe_scenario_dynamics_static(
        ZOO / "pdwa-omega-degenerate" / "notebook.ipynb"
    )
    assert v.verdict == "pass", v.message
    assert "updates" in v.message


def test_mp1_fails_the_frozen_obstacles_mutant():
    v = probe_scenario_dynamics_static(
        ZOO / "pdwa-frozen-obstacles-reconstructed" / "notebook.ipynb"
    )
    assert v.verdict == "fail", v.message
    assert v.finding_class == "M-004"


def test_mp1_auto_detects_the_dynamic_claim():
    nb = json.loads(
        (ZOO / "pdwa-frozen-obstacles-reconstructed" / "notebook.ipynb").read_text()
    )
    # The mutant keeps the narrative (that's the claims-vs-behavior gap).
    assert notebook_claims_dynamic_obstacles(nb)


def test_mp1_static_scenario_without_claim_is_consistent(tmp_path):
    nb = {"cells": [{
        "cell_type": "code", "id": "a",
        "source": [
            "for step in range(5):\n",
            "    v, w = select_velocity(robot_state=rs, obstacle_states=obs,"
            " goal_position=g, dt=0.5, seed=step)\n",
        ],
        "outputs": [],
    }]}
    p = tmp_path / "nb.ipynb"
    p.write_text(json.dumps(nb))
    v = probe_scenario_dynamics_static(p)  # no dynamic claim anywhere
    assert v.verdict == "pass"
    assert "no dynamic-obstacle claim" in v.message


def test_mp1_no_planner_call_is_unprobeable(tmp_path):
    nb = {"cells": [
        {"cell_type": "markdown", "source": ["dynamic obstacles ahead\n"]},
        {"cell_type": "code", "id": "a", "source": ["x = 1\n"], "outputs": []},
    ]}
    p = tmp_path / "nb.ipynb"
    p.write_text(json.dumps(nb))
    v = probe_scenario_dynamics_static(p)
    assert v.verdict == "unprobeable"


def test_mp1_environment_object_shape_flags_for_researcher(tmp_path):
    # The fresh-pdwa shape (2026-06-10): plan(start, goal, environment,
    # dynamics, ...) called once, no per-step loop, with a dynamic claim in
    # the narrative. Not statically decidable, so flag, not fail.
    nb = {"cells": [
        {"cell_type": "markdown",
         "source": ["Demo with dynamic obstacles moving toward the robot.\n"]},
        {"cell_type": "code", "id": "a", "outputs": [], "source": [
            "env = load_environment('two_rooms_simple')\n",
            "result = plan(start, goal, environment=env, dynamics=dyn,"
            " seed=0)\n",
        ]},
    ]}
    p = tmp_path / "nb.ipynb"
    p.write_text(json.dumps(nb))
    v = probe_scenario_dynamics_static(p, planner_names=("plan",))
    assert v.verdict == "flag_for_researcher", v.message
    assert v.finding_class == "M-004"


# ---------------------------------------------------------------------------
# MP-3 steering-responsiveness + MP-4 goal-progress (behavioral)
# ---------------------------------------------------------------------------

KNOWN_GOOD_PLANNER = '''
"""Greedy goal-seeking planner: picks the control minimizing next-step
heading error + distance, returns a trajectory of (state, control) pairs."""
import torch


class PlanResult:
    def __init__(self, trajectory, status):
        self.trajectory = trajectory
        self.status = status


def plan(start, goal, environment, dynamics, seed, dt=0.1):
    state = torch.as_tensor(start, dtype=torch.float64).clone()
    goal = torch.as_tensor(goal, dtype=torch.float64)
    trajectory = []
    for _ in range(300):
        dist = torch.hypot(goal[0] - state[0], goal[1] - state[1])
        if float(dist) < 0.3:
            return PlanResult(trajectory, "success")
        best, best_control = None, None
        for v in (0.2, 0.6, 1.0):
            for omega in (-1.0, -0.5, 0.0, 0.5, 1.0):
                control = torch.tensor([v, omega], dtype=torch.float64)
                nxt = dynamics.step(state, control, dt)
                bearing = torch.atan2(goal[1] - nxt[1], goal[0] - nxt[0])
                err = torch.atan2(torch.sin(bearing - nxt[2]),
                                  torch.cos(bearing - nxt[2]))
                d = torch.hypot(goal[0] - nxt[0], goal[1] - nxt[1])
                j = -abs(float(err)) - 0.5 * float(d)
                if best is None or j > best:
                    best, best_control = j, control
        trajectory.append((state.clone(), best_control.clone()))
        state = dynamics.step(state, best_control, dt)
    return PlanResult(trajectory, "timeout")
'''


def _module(tmp_path, source, name):
    path = tmp_path / name
    path.write_text(source, encoding="utf-8")
    from probes.package_loader import load_module_from_path
    return load_module_from_path(path)


def test_mp3_mp4_pass_on_goal_seeking_planner(tmp_path):
    pytest.importorskip("torch")
    from probes.motion_planning import (probe_goal_progress,
                                        probe_steering_responsiveness)
    mod = _module(tmp_path, KNOWN_GOOD_PLANNER, "good_planner.py")
    v3 = probe_steering_responsiveness(mod, "plan")
    assert v3.verdict == "pass", v3.message
    v4 = probe_goal_progress(mod, "plan")
    assert v4.verdict == "pass", v4.message
    assert "success" in v4.message


def test_mp3_flags_tie_break_steering_on_pdwa_jterms():
    pytest.importorskip("torch")
    from probes.motion_planning import probe_steering_responsiveness
    from probes.package_loader import load_module_from_path
    mod = load_module_from_path(ZOO / "pdwa-jterms-fresh" / "method.py")
    # With no obstacles, four of six terms are constant across candidates and
    # the obstacle terms are exactly zero, so the argmax is a tie-break: the
    # chosen turn rate ignores the goal side (the May steering class). One
    # seed and no timeout keep the test bounded (~10s: this planner burns
    # its full internal iteration cap per call); a single non-improving seed
    # is conclusive under the at-most-one-slack predicate.
    v = probe_steering_responsiveness(mod, "plan", seeds=(0,),
                                      per_call_timeout_s=None)
    assert v.verdict == "flag_for_researcher", f"{v.verdict}: {v.message}"
    assert v.finding_class == "M-005"


def test_mp4_fails_no_progress_on_pdwa_jterms():
    pytest.importorskip("torch")
    from probes.motion_planning import probe_goal_progress
    from probes.package_loader import load_module_from_path
    mod = load_module_from_path(ZOO / "pdwa-jterms-fresh" / "method.py")
    v = probe_goal_progress(mod, "plan")
    assert v.verdict == "fail", f"{v.verdict}: {v.message}"


def test_mp4_unprobeable_on_bare_control_selector(tmp_path):
    pytest.importorskip("torch")
    from probes.motion_planning import (probe_goal_progress,
                                        probe_steering_responsiveness)
    mod = _module(tmp_path, (
        "import torch\n"
        "\n"
        "\n"
        "def select_velocity(robot_state, goal, dynamics, dt=0.1, seed=0):\n"
        "    bearing = torch.atan2(goal[1] - robot_state[1],\n"
        "                          goal[0] - robot_state[0])\n"
        "    err = torch.atan2(torch.sin(bearing - robot_state[2]),\n"
        "                      torch.cos(bearing - robot_state[2]))\n"
        "    omega = max(-1.0, min(1.0, float(err)))\n"
        "    return torch.tensor([0.5, omega], dtype=torch.float64)\n"),
        "bare_control.py")
    # MP-3 handles the bare-control shape; MP-4 honestly cannot.
    v3 = probe_steering_responsiveness(mod, "select_velocity")
    assert v3.verdict == "pass", v3.message
    v4 = probe_goal_progress(mod, "select_velocity")
    assert v4.verdict == "unprobeable"
    assert "trajectory" in v4.message


# ---------------------------------------------------------------------------
# Contract-driven planner kit (mp_planner_kit, queue item 12, 2026-07-04)
# ---------------------------------------------------------------------------

_TEMPLATE_CLASSES = '''
"""Mini package in the paradigm-template interface shape (the iDb-RRT case):
Environment with bounds/is_in_collision, UnicycleDynamics without V_max,
paradigm-contract Trajectory with parallel states/controls lists."""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch


@dataclass
class Environment:
    name: str
    state_bounds_lower: torch.Tensor
    state_bounds_upper: torch.Tensor
    obstacles: List[Dict[str, Any]]

    def bounds(self):
        return self.state_bounds_lower, self.state_bounds_upper

    def is_in_collision(self, state):
        return False


class UnicycleDynamics:
    state_dim: int = 3
    control_dim: int = 2

    def step(self, state, control, dt):
        x, y, theta = state[0], state[1], state[2]
        v, omega = control[0], control[1]
        return torch.stack([x + v * torch.cos(theta) * dt,
                            y + v * torch.sin(theta) * dt,
                            theta + omega * dt])


@dataclass
class Trajectory:
    states: List[torch.Tensor]
    controls: List[torch.Tensor]
    cost: float


@dataclass
class PlanResult:
    trajectory: Optional[Trajectory]
    status: str
    stats: Dict[str, Any] = field(default_factory=dict)
'''

_KIT_GOOD_STATE_GOAL_PLANNER = _TEMPLATE_CLASSES + '''

def plan(start, goal, environment, dynamics, seed, *, max_iterations=10000):
    """Greedy goal-seeker in the iDb-RRT interface shape: goal is a full
    STATE (a 2-dim position raises the live size-mismatch error, so the
    probe's goal ladder must advance), the package classes are exercised,
    and the result is the paradigm-contract Trajectory object."""
    if goal.shape[0] != dynamics.state_dim:
        raise RuntimeError(
            "The size of tensor a (3) must match the size of tensor b (2)"
            " at non-singleton dimension 0")
    state = torch.as_tensor(start, dtype=torch.float64).clone()
    states, controls = [state], []
    for _ in range(60):
        if environment.is_in_collision(state):
            return PlanResult(None, "infeasible", {})
        dist = torch.hypot(goal[0] - state[0], goal[1] - state[1])
        if float(dist) < 0.3:
            return PlanResult(Trajectory(states, controls, 0.1 * len(controls)),
                              "success", {"iterations": len(controls)})
        bearing = torch.atan2(goal[1] - state[1], goal[0] - state[0])
        err = torch.atan2(torch.sin(bearing - state[2]),
                          torch.cos(bearing - state[2]))
        control = torch.tensor(
            [1.0, max(-1.0, min(1.0, float(err)))], dtype=torch.float64)
        state = dynamics.step(state, control, 0.1)
        controls.append(control)
        states.append(state)
    return PlanResult(Trajectory(states, controls, 6.0), "timeout",
                      {"iterations": len(controls)})
'''

_KIT_FALSE_SUCCESS_PLANNER = _TEMPLATE_CLASSES + '''

def plan(start, goal, environment, dynamics, seed):
    return PlanResult(None, "success", {})
'''


def test_kit_binds_package_classes_and_probes_pass(tmp_path):
    pytest.importorskip("torch")
    from probes.motion_planning import (mp_planner_kit, probe_goal_progress,
                                        probe_steering_responsiveness)
    mod = _module(tmp_path, _KIT_GOOD_STATE_GOAL_PLANNER, "state_goal.py")
    kit = mp_planner_kit(mod, "plan")
    assert isinstance(kit, dict), kit
    assert type(kit["dynamics"]).__name__ == "UnicycleDynamics"
    assert type(kit["make_environment"]()).__name__ == "Environment"
    assert kit["make_environment"]().obstacles == []
    v3 = probe_steering_responsiveness(mod, "plan", kit=kit)
    assert v3.verdict == "pass", v3.message
    v4 = probe_goal_progress(mod, "plan", kit=kit)
    assert v4.verdict == "pass", v4.message


def test_kit_probes_are_deterministic(tmp_path):
    pytest.importorskip("torch")
    from probes.motion_planning import (mp_planner_kit, probe_goal_progress,
                                        probe_steering_responsiveness)
    verdicts = []
    for _ in range(2):
        mod = _module(tmp_path, _KIT_GOOD_STATE_GOAL_PLANNER, "state_goal.py")
        kit = mp_planner_kit(mod, "plan")
        v3 = probe_steering_responsiveness(mod, "plan", kit=kit)
        v4 = probe_goal_progress(mod, "plan", kit=kit)
        verdicts.append((v3.verdict, v3.message, v4.verdict, v4.message))
    assert verdicts[0] == verdicts[1]


def test_kit_zoo_no_path_mutant_flags_goal_progress():
    pytest.importorskip("torch")
    from probes.motion_planning import (mp_planner_kit, probe_goal_progress,
                                        probe_steering_responsiveness)
    from probes.package_loader import load_module_from_path
    mod = load_module_from_path(
        ZOO / "idbrrt-no-path-reconstructed" / "method.py")
    kit = mp_planner_kit(mod, "plan")
    assert isinstance(kit, dict), kit
    v4 = probe_goal_progress(mod, "plan", kit=kit)
    assert v4.verdict == "flag_for_researcher", f"{v4.verdict}: {v4.message}"
    assert "no trajectory" in v4.message
    assert "timeout" in v4.message
    v3 = probe_steering_responsiveness(mod, "plan", kit=kit)
    assert v3.verdict == "unprobeable"
    # Routes to the goal-progress check without a bare code (the pdwa
    # 2026-07-05 gloss-splice class).
    assert "goal-progress check" in v3.message


def test_kit_mp4_fails_success_claim_without_trajectory(tmp_path):
    pytest.importorskip("torch")
    from probes.motion_planning import mp_planner_kit, probe_goal_progress
    mod = _module(tmp_path, _KIT_FALSE_SUCCESS_PLANNER, "false_success.py")
    v = probe_goal_progress(mod, "plan", kit=mp_planner_kit(mod, "plan"))
    assert v.verdict == "fail", f"{v.verdict}: {v.message}"
    assert "consistency" in v.message


def test_kit_unprobeable_names_unmapped_environment_param(tmp_path):
    pytest.importorskip("torch")
    from probes.motion_planning import mp_planner_kit
    src = _KIT_GOOD_STATE_GOAL_PLANNER.replace(
        "    obstacles: List[Dict[str, Any]]",
        "    obstacles: List[Dict[str, Any]]\n    mesh_file: str")
    mod = _module(tmp_path, src, "mesh_env.py")
    kit = mp_planner_kit(mod, "plan")
    assert isinstance(kit, str) and "mesh_file" in kit


def test_kit_unprobeable_names_nonplanar_state_dim(tmp_path):
    pytest.importorskip("torch")
    from probes.motion_planning import mp_planner_kit
    src = _KIT_GOOD_STATE_GOAL_PLANNER.replace(
        "    state_dim: int = 3", "    state_dim: int = 12")
    mod = _module(tmp_path, src, "quadrotor.py")
    kit = mp_planner_kit(mod, "plan")
    assert isinstance(kit, str) and "state_dim=12" in kit


def test_kit_falls_back_to_stubs_without_declared_classes(tmp_path):
    pytest.importorskip("torch")
    from probes.motion_planning import _StubEnvironment, mp_planner_kit
    from probes.term_ablation import _UnicycleDynamics
    mod = _module(tmp_path, KNOWN_GOOD_PLANNER, "good_planner.py")
    kit = mp_planner_kit(mod, "plan")
    assert isinstance(kit, dict), kit
    assert isinstance(kit["dynamics"], _UnicycleDynamics)
    assert isinstance(kit["make_environment"](), _StubEnvironment)


def test_kit_live_pin_pdwa_reference_package():
    """Live pin on the committed pdwa reference artifact (the 2026-06-10
    degenerate-steering delivery, preserved as a fixture when example_runs/
    became the curated researcher set on 2026-07-06): the kit binds the
    package's own Environment/UnicycleDynamics and MP-4 delivers a REAL fail
    (no goal progress), where the pre-kit stubs produced a harness-shaped
    unprobeable. The fresh 2026-07-05 pdwa delivery cannot replace this pin:
    its planner blows the per-call budget on the trivial fixture, so its
    goal-progress probe reads timeout, not the clean fail pinned here."""
    pytest.importorskip("torch")
    run_dir = (REPO / "tests" / "fixtures" / "delivery"
               / "pdwa-degenerate-steering-20260610")
    if not (run_dir / "method").is_dir():
        pytest.skip("pdwa reference artifact not present")
    from probes.motion_planning import mp_planner_kit, probe_goal_progress
    from probes.package_loader import imported_method_package
    with imported_method_package(run_dir) as method:
        kit = mp_planner_kit(method, "plan")
        assert isinstance(kit, dict), kit
        assert type(kit["make_environment"]()).__name__ == "Environment"
        v4 = probe_goal_progress(method, "plan", kit=kit)
    assert v4.verdict == "fail", f"{v4.verdict}: {v4.message}"
    assert "no goal progress" in v4.message
