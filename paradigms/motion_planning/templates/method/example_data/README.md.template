# example_data/

This directory is where the notebook reads environments and planning problems
from. The package ships with one or more smoke environments + problems (named
in `method/data.py`'s `load_problem` function); the notebook's `load_problem()`
call defaults to the simplest.

## Use your own problem

`load_problem(name)` returns `(start, goal, environment)`. The robot dynamics
model is separate — it lives in `method/model.py` and is constructed in your
own code (see `notebook.ipynb` §3.2), not described in the problem file. The
simplest way to define a custom problem is to drop a single JSON or Python file
into this directory matching one of the conventions below.

### Option 1 — `.json` (declarative; recommended)

A single `.json` file (any name) defining a problem. The `load_problem` loader
reads it and constructs the corresponding Environment object.

```json
{
  "name": "my_custom_problem",
  "start": [0.0, 0.0, 0.0],
  "goal": [5.0, 3.0, 1.57],
  "state_bounds_lower": [-10.0, -10.0, -3.14159],
  "state_bounds_upper": [10.0, 10.0, 3.14159],
  "obstacles": [
    {"type": "rectangle", "xmin": 1.0, "xmax": 2.0, "ymin": -1.0, "ymax": 4.0},
    {"type": "circle", "x": 3.5, "y": 1.5, "radius": 0.8}
  ],
  "planning_time_budget_seconds": 30.0
}
```

Supported obstacle types: `rectangle` (axis-aligned), `circle`, `polygon` (vertex
list). The robot dynamics is chosen in your code from `method/model.py` (see
notebook §3.2), independent of the problem file. To add a new robot type,
subclass `SystemDynamics` in `method/model.py`; see Option 3 below.

### Option 2 — `.py` (programmatic; for complex environments)

A single `.py` file in this directory that defines a module-level callable
`build_problem(seed: int) -> Tuple[start, goal, env]`. The loader will
import the module and call this function.

```python
# example_data/my_problem.py
import torch

def build_problem(seed: int = 0):
    start = torch.tensor([0.0, 0.0, 0.0])
    goal = torch.tensor([5.0, 3.0, 1.57])
    # ...build env however you like (programmatic obstacles, loaded from
    # a custom format, etc.)
    env = ...
    return start, goal, env
```

### Option 3 — Custom dynamics

If your system isn't in `method/model.py`, add it there as a new subclass of
`SystemDynamics`. The required interface is `step(state, control, dt) -> state`
and `step_jacobian(state, control, dt) -> (df/dx, df/du)`. See the existing
subclasses for reference. Optimization-based planners require correct
Jacobians; finite-difference your implementation to sanity-check.

## Format requirements (all options)

- `start` and `goal` must be 1D tensors matching the system's state dimension.
- `state_bounds_lower` and `state_bounds_upper` must each match state dim.
- Obstacle geometry is paradigm-defined; the package's `CollisionModel`
  uses simple primitives (rectangles, circles, polygons). For more complex
  environments (mesh-based, SDF-based), subclass `CollisionModel` in
  `method/model.py`.
- Planning time budget is in seconds. The planner may finish earlier (if it
  finds a solution) but won't run longer.

## What's bundled

By default, this package ships with one or more named problems referenced from
`method/data.py`'s `load_problem` function. They're chosen to exercise the
paper's planner on a problem instance that the planner is supposed to solve
within the smoke time budget. See `notebook.ipynb` §3.3 for the smoke problem
description.
