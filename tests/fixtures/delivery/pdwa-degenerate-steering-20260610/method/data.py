"""Environment + planning-problem loading.

Default behavior: `load_problem()` (no args) reads from `method/example_data/`
— the directory bundled inside this package. Named problems are defined in
this module's `_NAMED_PROBLEMS` registry; the loader supports both bundled
named problems and user-defined problems dropped into `example_data/`
(see `example_data/README.md` for the format).

For motion planning, "data" means a planning-problem instance:
    (start, goal, environment)
not a dataset of training examples. Use `load_environment(name)` to load
just the environment (for visualization), or `load_problem(name)` to load
start + goal + environment. (The robot dynamics model is separate — it lives
in method/model.py and is constructed by the caller, not returned here.)

Custom problems: drop a `.json` or `.py` file into `example_data/` per the
format in `example_data/README.md`, then pass its name (without extension)
to `load_problem`. To use a system not in `method/model.py`, add a new
SystemDynamics subclass there (see model.py's docstring for the contract).
"""

# NOTE: this template is paradigm-fixed. The scaffolder renders it with
# Enhanced DWA with Dynamic Obstacle Behavior Prediction (PDWA), motion_planning, and other slots. The architecture-coder
# writes the Dynamics + CollisionModel in method/model.py; the method-coder
# writes the planner in method/method.py. data.py stays paradigm-shaped AND
# decoupled from model.py — it loads environments + planning problems
# (start/goal/env) only; the dynamics model lives in model.py and is built by
# the notebook/caller.

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch


# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------


@dataclass
class Environment:
    """The planning environment: state-space bounds + collision geometry.

    Exposes `.is_in_collision(state) -> bool` and `.visualize(ax)` for
    matplotlib rendering. The concrete CollisionModel implementation in
    `method/model.py` consumes this Environment when checking collisions.
    """

    name: str
    state_bounds_lower: torch.Tensor
    state_bounds_upper: torch.Tensor
    obstacles: List[Dict[str, Any]]  # paradigm-shaped — see example_data/README.md

    def bounds(self) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.state_bounds_lower, self.state_bounds_upper

    def is_in_collision(self, state: torch.Tensor) -> bool:
        """Default collision check using axis-aligned rectangles and circles.

        Override / extend in method/model.py's CollisionModel for paper-
        specific collision geometry (signed-distance fields, meshes, etc.).
        """
        # Only checks position (first 2-3 dimensions). Method-coder may
        # replace this with a more sophisticated check.
        x = state[0].item()
        y = state[1].item() if state.numel() > 1 else 0.0
        for obs in self.obstacles:
            kind = obs.get("type")
            if kind == "rectangle":
                if (obs["xmin"] <= x <= obs["xmax"]
                        and obs["ymin"] <= y <= obs["ymax"]):
                    return True
            elif kind == "circle":
                dx = x - obs["x"]
                dy = y - obs["y"]
                if dx * dx + dy * dy <= obs["radius"] ** 2:
                    return True
            # polygon support omitted from the default — method-coder adds
            # it when the paper's environments use polygonal obstacles.
        return False

    def visualize(self, ax) -> None:
        """Render obstacles + bounds on a matplotlib axis."""
        import matplotlib.patches as patches

        for obs in self.obstacles:
            kind = obs.get("type")
            if kind == "rectangle":
                rect = patches.Rectangle(
                    (obs["xmin"], obs["ymin"]),
                    obs["xmax"] - obs["xmin"],
                    obs["ymax"] - obs["ymin"],
                    linewidth=1, edgecolor="black", facecolor="gray", alpha=0.6,
                )
                ax.add_patch(rect)
            elif kind == "circle":
                circle = patches.Circle(
                    (obs["x"], obs["y"]),
                    obs["radius"],
                    linewidth=1, edgecolor="black", facecolor="gray", alpha=0.6,
                )
                ax.add_patch(circle)
        # Set viewport from state bounds (first two dims = x, y)
        ax.set_xlim(self.state_bounds_lower[0].item(),
                    self.state_bounds_upper[0].item())
        ax.set_ylim(self.state_bounds_lower[1].item() if len(self.state_bounds_lower) > 1 else -1,
                    self.state_bounds_upper[1].item() if len(self.state_bounds_upper) > 1 else 1)
        ax.set_aspect("equal")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title(f"Environment: {self.name}")


# ---------------------------------------------------------------------------
# Bundled named problems
# ---------------------------------------------------------------------------

# Method-coder may extend this registry with paper-specific named problems.
# Each entry is a dict of (json-loadable) problem-definition fields per the
# format in example_data/README.md (Option 1).
_NAMED_PROBLEMS: Dict[str, Dict[str, Any]] = {
    "two_rooms_simple": {
        "name": "two_rooms_simple",
        "system": "default",   # method.model.py's primary SystemDynamics subclass
        "start": [0.0, 0.0, 0.0],
        "goal": [4.0, 0.0, 0.0],
        "state_bounds_lower": [-5.0, -3.0, -3.14159],
        "state_bounds_upper": [5.0, 3.0, 3.14159],
        "obstacles": [
            {"type": "rectangle", "xmin": 1.5, "xmax": 2.5, "ymin": -2.0, "ymax": 0.5},
            {"type": "rectangle", "xmin": 1.5, "xmax": 2.5, "ymin": 1.0, "ymax": 3.0},
        ],
        "planning_time_budget_seconds": 30.0,
    },
}


def list_named_problems() -> List[str]:
    return sorted(_NAMED_PROBLEMS.keys())


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------


def _resolve_example_data_dir() -> Path:
    """method/example_data/ relative to this file."""
    return Path(__file__).resolve().parent / "example_data"


def load_environment(
    name: str = "two_rooms_simple", *, seed: int = 0
) -> Environment:
    """Load an environment by name. Checks bundled named problems first;
    falls back to looking up `name + '.json'` in example_data/."""
    if name in _NAMED_PROBLEMS:
        spec = _NAMED_PROBLEMS[name]
    else:
        # Try to load as a JSON file from example_data/
        path = _resolve_example_data_dir() / f"{name}.json"
        if not path.exists():
            raise FileNotFoundError(
                f"environment {name!r} not in bundled named-problems "
                f"({list_named_problems()}) and not found at {path}"
            )
        spec = json.loads(path.read_text(encoding="utf-8"))
    return Environment(
        name=spec.get("name", name),
        state_bounds_lower=torch.tensor(spec["state_bounds_lower"], dtype=torch.float32),
        state_bounds_upper=torch.tensor(spec["state_bounds_upper"], dtype=torch.float32),
        obstacles=spec["obstacles"],
    )


def load_problem(
    name: str = "two_rooms_simple", *, seed: int = 0
) -> Tuple[torch.Tensor, torch.Tensor, Environment]:
    """Load a (start, goal, environment) tuple by name.

    Looks up bundled named problems first; falls back to example_data/<name>.json
    or example_data/<name>.py (programmatic).

    The robot's dynamics model is NOT returned here — it lives in
    `method/model.py` (a SystemDynamics subclass), which `data.py` deliberately
    does not import. The notebook/caller constructs the dynamics from model.py
    and passes it to the planner.
    """
    # 1. Try programmatic Python override
    py_path = _resolve_example_data_dir() / f"{name}.py"
    if py_path.exists():
        import importlib.util
        spec_mod = importlib.util.spec_from_file_location(name, py_path)
        if spec_mod is None or spec_mod.loader is None:
            raise ImportError(f"could not load {py_path}")
        module = importlib.util.module_from_spec(spec_mod)
        spec_mod.loader.exec_module(module)
        if not hasattr(module, "build_problem"):
            raise AttributeError(
                f"{py_path} defines a problem but is missing the required "
                "module-level `build_problem(seed) -> (start, goal, env)`."
            )
        return module.build_problem(seed=seed)  # type: ignore[no-any-return]

    # 2. Named or JSON
    env = load_environment(name, seed=seed)
    if name in _NAMED_PROBLEMS:
        spec = _NAMED_PROBLEMS[name]
    else:
        spec = json.loads((_resolve_example_data_dir() / f"{name}.json").read_text(encoding="utf-8"))
    start = torch.tensor(spec["start"], dtype=torch.float32)
    goal = torch.tensor(spec["goal"], dtype=torch.float32)
    return start, goal, env
