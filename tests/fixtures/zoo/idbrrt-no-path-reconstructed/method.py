"""Reconstruction — the iDb-RRT no-path delivery shape (2026-07-04 live).

The delivered iDb-RRT package (r2c_runs/iDb-RRT, breadth run 2026-07-01,
gitignored) returned ``PlanResult(trajectory=None, status="timeout")`` with
``optimization_attempts: 0`` from its OWN executed demo notebook AND from the
probe kit's trivial obstacle-free fixture — the planner never demonstrated a
successful plan anywhere, and nothing bound on that until the contract-driven
kit (mp_planner_kit) landed. This mutant reconstructs that interface shape
compactly: the paradigm-template Environment/dynamics classes plus a plan()
that searches, never connects, and honestly reports timeout with no
trajectory.

Expected verdicts: MP-4 flag_for_researcher (the planner_no_path_found
class at probe budgets), MP-3 unprobeable pointing at MP-4 (no first
decision to judge). A false-success variant (status "success" with no
trajectory) must FAIL MP-4 — that arm is exercised inline in
test_motion_planning_probes.py.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch


@dataclass
class Environment:
    """Paradigm-template environment: bounds + collision geometry."""

    name: str
    state_bounds_lower: torch.Tensor
    state_bounds_upper: torch.Tensor
    obstacles: List[Dict[str, Any]]

    def bounds(self):
        return self.state_bounds_lower, self.state_bounds_upper

    def is_in_collision(self, state: torch.Tensor) -> bool:
        x = state[0].item()
        y = state[1].item() if state.numel() > 1 else 0.0
        for obs in self.obstacles:
            if obs.get("type") == "circle":
                if (x - obs["x"]) ** 2 + (y - obs["y"]) ** 2 <= obs["radius"] ** 2:
                    return True
        return False


class UnicycleDynamics:
    """Paradigm-template dynamics: step + step_jacobian, no V_max attr."""

    state_dim: int = 3
    control_dim: int = 2

    def step(self, state, control, dt):
        x, y, theta = state[0], state[1], state[2]
        v, omega = control[0], control[1]
        return torch.stack([x + v * torch.cos(theta) * dt,
                            y + v * torch.sin(theta) * dt,
                            theta + omega * dt])

    def step_jacobian(self, state, control, dt):
        raise NotImplementedError("not exercised by the mutant")


@dataclass
class Trajectory:
    """Paradigm-contract trajectory: parallel states/controls lists."""

    states: List[torch.Tensor]
    controls: List[torch.Tensor]
    cost: float


@dataclass
class PlanResult:
    trajectory: Optional[Trajectory]
    status: str  # "success" | "timeout" | "infeasible" | "error"
    stats: Dict[str, Any] = field(default_factory=dict)


def plan(
    start: torch.Tensor,
    goal: torch.Tensor,
    environment,
    dynamics: UnicycleDynamics,
    seed: int,
    *,
    goal_bias: float = 0.1,
    max_iterations: int = 10000,
    to_max_iterations: int = 50,
) -> PlanResult:
    """Search that never connects — the live no-path class.

    Faithful to the live shape: a goal STATE of shape (state_dim,) (a 2-dim
    goal position raises the size-mismatch RuntimeError the real delivery
    raised), a bounded search loop that burns its budget, and an honest
    timeout result with zero optimization attempts and no trajectory.
    """
    if goal.shape[0] != dynamics.state_dim:
        raise RuntimeError(
            f"The size of tensor a ({dynamics.state_dim}) must match the "
            f"size of tensor b ({int(goal.shape[0])}) at non-singleton "
            f"dimension 0")
    torch.manual_seed(seed)
    state = start
    for _ in range(min(int(max_iterations), 200)):
        # Expansion whose acceptance radius shrinks to nothing: candidates
        # are generated but never connect to the goal region.
        control = torch.rand(dynamics.control_dim, dtype=torch.float64) * 1e-6
        state = dynamics.step(state, control, 0.1)
    return PlanResult(
        trajectory=None,
        status="timeout",
        stats={"iterations": int(to_max_iterations),
               "optimization_attempts": 0,
               "search_attempts": int(to_max_iterations)},
    )
