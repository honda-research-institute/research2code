"""Motion-planning probes.

MP-1 static arm (finding class M-004): when the notebook's narrative claims a
DYNAMIC obstacle scenario, the obstacle state passed to the planner inside the
demo loop must actually change across steps. The pdwa case shipped a loop that
set `obstacle_states` once and never moved anything, so the paper's headline
contribution (avoiding MOVING obstacles via prediction) was never exercised,
and every review passed it.

Detection is AST-based over the loop cell:
- direct mutation/rebinding of the obstacle variable inside the loop body, or
- the alias-iteration pattern the real notebooks use:
      for obs in obstacle_states: obs['x'] += dx

MP-3 steering-responsiveness (behavioral, flag-tier): with the goal 90° off
the robot's heading and no obstacles, the planner's first chosen control must
reduce the heading error toward the goal, consistently across seeds and on
BOTH sides (a left goal and a right goal — an always-turn-left tie-break
passes one side and fails the other). The May pdwa case is the target class:
an objective insensitive to the steering variable makes the chosen turn rate
an argmax tie-break artifact. Routed flag_for_researcher (the insensitivity
can be faithful transcription of degenerate paper math, the M-005 class).

MP-4 goal-progress (behavioral): on an obstacle-free fixture the planner must
end closer to the goal than it started, and a result that *claims* success
must actually end near the goal (v3's solved-implies-consistency check).

The behavioral probes accept the three real planner output shapes: a
trajectory-returning planner with (state, control) pairs (the 2026-06-10
pdwa shape), the paradigm-contract Trajectory object with parallel
``.states``/``.controls`` lists (the iDb-RRT shape), and a bare-control
selector (the May select_velocity shape — goal progress is then honestly
unprobeable).

Contract-driven kit (``mp_planner_kit``, 2026-07-04, queue item 12): real
deliveries build their problem instance from the package's OWN classes —
the paradigm-template ``Environment`` (bounds + ``is_in_collision``) in
data.py and a ``*Dynamics`` class (``step``/``step_jacobian``) in model.py.
The probes' duck-typed stubs lack those methods, so both behavioral probes
came back unprobeable on the delivered iDb-RRT package (the honest-draft
cause, 2026-07-01). The kit resolves the planner's problem-instance objects
from the package itself, exactly like kd_loss_kit builds student/teacher
via the package's own builders: package classes when declared, the probe
stubs only when the package declares none, and a named unprobeable reason
when a declared class cannot be built (never a fabricated stand-in).
"""

from __future__ import annotations

import ast
import inspect
import json
import math
import re
from pathlib import Path
from types import ModuleType

from probes import ProbeVerdict
from probes.catalogs.motion_planning import PROBE_CATALOG as _PROBE_CATALOG

PROBE_CATALOG = _PROBE_CATALOG
from probes.trainability import _fill_kwargs

DEFAULT_PLANNER_NAMES = ("select_velocity", "plan", "plan_step", "select_action")
_DYNAMIC_CLAIM_RE = re.compile(r"dynamic\s+obstacle|moving\s+obstacle", re.IGNORECASE)


def _code_cells(nb: dict) -> list[str]:
    return ["".join(c["source"]) for c in nb.get("cells", [])
            if c.get("cell_type") == "code"]


def _all_text(nb: dict) -> str:
    return "\n".join("".join(c["source"]) for c in nb.get("cells", []))


def notebook_claims_dynamic_obstacles(nb: dict) -> bool:
    return bool(_DYNAMIC_CLAIM_RE.search(_all_text(nb)))


def _call_name(node: ast.Call) -> str:
    fn = node.func
    if isinstance(fn, ast.Name):
        return fn.id
    if isinstance(fn, ast.Attribute):
        return fn.attr
    return ""


def _obstacle_arg_var(call: ast.Call) -> str | None:
    """The variable name passed as the planner's obstacle state."""
    for kw in call.keywords:
        if kw.arg and "obstacle" in kw.arg and isinstance(kw.value, ast.Name):
            return kw.value.id
    for arg in call.args:
        if isinstance(arg, ast.Name) and "obstacle" in arg.id:
            return arg.id
    return None


def _mutates_name(body: list[ast.stmt], name: str) -> bool:
    """True if any statement in `body` mutates/rebinds `name`, directly or via
    the for-alias pattern (for X in name: mutate X)."""
    for node in ast.walk(ast.Module(body=body, type_ignores=[])):
        # Direct rebinding or subscript/attribute mutation: name = / name[i] = /
        # name[i] += / name.attr = ...
        targets: list[ast.expr] = []
        if isinstance(node, ast.Assign):
            targets = node.targets
        elif isinstance(node, (ast.AugAssign, ast.AnnAssign)):
            targets = [node.target]
        for t in targets:
            base = t
            while isinstance(base, (ast.Subscript, ast.Attribute)):
                base = base.value
            if isinstance(base, ast.Name) and base.id == name:
                return True
        # Mutating method calls: name.append(...), name.extend(...), etc.
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            base = node.func.value
            if isinstance(base, ast.Name) and base.id == name and node.func.attr in (
                "append", "extend", "insert", "pop", "remove", "clear", "update",
            ):
                return True
        # Alias-iteration: for X in name: <body mutates X>.
        if (
            isinstance(node, ast.For)
            and isinstance(node.iter, ast.Name)
            and node.iter.id == name
            and isinstance(node.target, ast.Name)
        ):
            if _mutates_name(node.body, node.target.id):
                return True
    return False


def probe_scenario_dynamics_static(
    nb_path: Path,
    planner_names: tuple[str, ...] = DEFAULT_PLANNER_NAMES,
    claims_dynamic: bool | None = None,
) -> ProbeVerdict:
    """MP-1 (static arm): dynamic-obstacle claims require obstacle motion in
    the demo loop. `claims_dynamic=None` auto-detects from the notebook text."""
    nb = json.loads(Path(nb_path).read_text(encoding="utf-8"))
    name = Path(nb_path).name
    if claims_dynamic is None:
        claims_dynamic = notebook_claims_dynamic_obstacles(nb)
    if not claims_dynamic:
        return ProbeVerdict(
            "MP-1", "pass",
            "notebook makes no dynamic-obstacle claim — static scenario is "
            "consistent", evidence=name, finding_class="M-004",
        )

    for src in _code_cells(nb):
        try:
            tree = ast.parse(src)
        except SyntaxError:
            continue
        for loop in [n for n in ast.walk(tree) if isinstance(n, (ast.For, ast.While))]:
            for call in [n for n in ast.walk(loop) if isinstance(n, ast.Call)]:
                if _call_name(call) not in planner_names:
                    continue
                obstacle_var = _obstacle_arg_var(call)
                if obstacle_var is None:
                    continue
                if _mutates_name(loop.body, obstacle_var):
                    return ProbeVerdict(
                        "MP-1", "pass",
                        f"demo loop updates {obstacle_var!r} between planner "
                        f"calls — the dynamic scenario is actually dynamic",
                        evidence=name, finding_class="M-004",
                    )
                return ProbeVerdict(
                    "MP-1", "fail",
                    f"narrative claims dynamic obstacles but {obstacle_var!r} "
                    f"is never updated inside the demo loop — the paper's "
                    f"moving-obstacle behavior is never exercised",
                    evidence=f"{name}: planner call "
                             f"{_call_name(call)}(...{obstacle_var}...)",
                    finding_class="M-004",
                )
    # Environment-object arm (2026-06-10, from the fresh pdwa run): planners
    # like plan(start, goal, environment, dynamics, ...) consume obstacles
    # inside an environment object, often in a single call with no per-step
    # loop. Whether that environment's obstacles actually move is not
    # statically decidable in general — so a dynamic CLAIM plus this shape is
    # a researcher flag, not a fail and not a silent unprobeable. (The pdwa
    # methodology contract caught exactly this as 'obstacles must have
    # non-zero velocity to exercise prediction'.)
    for src in _code_cells(nb):
        try:
            tree = ast.parse(src)
        except SyntaxError:
            continue
        for call in (n for n in ast.walk(tree) if isinstance(n, ast.Call)):
            if _call_name(call) not in planner_names:
                continue
            arg_names = [a.id for a in call.args if isinstance(a, ast.Name)]
            arg_names += [kw.arg for kw in call.keywords if kw.arg]
            if any(n and ("environment" in n or n == "env" or "dynamics" in n)
                   for n in arg_names):
                return ProbeVerdict(
                    "MP-1", "flag_for_researcher",
                    f"narrative claims dynamic obstacles and "
                    f"{_call_name(call)}(...) consumes an environment object "
                    f"— verify the environment's obstacles actually move "
                    f"during the demo (static environments never exercise "
                    f"the paper's moving-obstacle behavior)",
                    evidence=name, finding_class="M-004",
                )
    return ProbeVerdict(
        "MP-1", "unprobeable",
        f"no planner call ({'/'.join(planner_names)}) with an obstacle-state "
        f"argument found inside a loop",
        evidence=name, finding_class="M-004",
    )

# ---------------------------------------------------------------------------
# MP-3 / MP-4 — behavioral probes on the method package's planner
# ---------------------------------------------------------------------------


class _StubEnvironment:
    """Environment object in the generated-package convention: planners read
    `environment.obstacles` as a list of geometry dicts."""

    def __init__(self, obstacles: list[dict] | None = None):
        self.obstacles = list(obstacles or [])


# ---------------------------------------------------------------------------
# Contract-driven planner kit (queue item 12; the KD-kit pattern)
# ---------------------------------------------------------------------------

# The planar fixtures below (goal 90° off heading, straight-ahead goal)
# assume the paradigm template's state convention [x, y, theta].
_PLANAR_STATE_DIM = 3


def _package_dynamics_class(package, fn):
    """The package's own dynamics class for the planner, or None.

    Ladder: (a) the class annotating the planner's dynamics-named parameter;
    (b) the unique ``*Dynamics`` class with a callable ``step`` exposed by
    the package (top level or the template submodules) — filtered by the
    annotation's name when annotations are postponed strings. None falls
    back to the probe-owned planar stub."""
    annotation = None
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        params = {}
    for name, p in params.items():
        if "dynamics" in name.lower() and p.annotation is not p.empty:
            annotation = p.annotation
            break
    if inspect.isclass(annotation):
        return annotation

    candidates: list[type] = []
    modules = [package] + [m for m in (getattr(package, s, None)
                                       for s in ("method", "model", "data"))
                           if m is not None]
    for mod in modules:
        for obj in vars(mod).values():
            if (inspect.isclass(obj) and obj.__name__.endswith("Dynamics")
                    and callable(getattr(obj, "step", None))
                    and obj not in candidates):
                candidates.append(obj)
    if isinstance(annotation, str):
        named = [c for c in candidates if c.__name__ == annotation.strip()]
        if named:
            return named[0]
    if len(candidates) == 1:
        return candidates[0]
    return None


def mp_planner_kit(package, planner_name: str, seed: int = 0) -> dict | str:
    """Resolve the planner's problem-instance objects from the run's own
    package and return ``{fn, dynamics, make_environment, ...}``, or the
    unprobeable-reason string.

    Mirrors ``kd_loss_kit``: the package's own Environment/dynamics classes
    when it declares them (the paradigm-template contract — real planners
    call ``environment.is_in_collision``/``bounds`` and
    ``dynamics.step_jacobian``, which the duck-typed probe stubs lack), the
    probe stubs only when the package declares none, and a named reason when
    a declared class cannot be built. Never a fabricated stand-in for a
    declared contract."""
    try:
        import torch  # noqa: PLC0415
    except ImportError:
        return "torch required"
    from probes.package_loader import resolve_symbol  # noqa: PLC0415
    from probes.term_ablation import _UnicycleDynamics  # noqa: PLC0415

    fn = resolve_symbol(package, planner_name)
    if fn is None or not callable(fn):
        return f"planner {planner_name!r} not found in package"

    dynamics_cls = _package_dynamics_class(package, fn)
    if dynamics_cls is None:
        dynamics = _UnicycleDynamics()
    else:
        kwargs = _fill_kwargs(dynamics_cls, {"dt": 0.1, "seed": seed})
        if isinstance(kwargs, str):
            return (f"{dynamics_cls.__name__} requires unmapped constructor "
                    f"parameter {kwargs!r} — cannot build the package's own "
                    f"dynamics")
        try:
            dynamics = dynamics_cls(**kwargs)
        except Exception as e:  # noqa: BLE001 — generated classes fail arbitrarily
            return (f"{dynamics_cls.__name__}() raised "
                    f"{type(e).__name__}: {e}")

    try:
        state_dim = int(getattr(dynamics, "state_dim", _PLANAR_STATE_DIM))
    except (TypeError, ValueError):
        state_dim = _PLANAR_STATE_DIM
    if state_dim != _PLANAR_STATE_DIM:
        return (f"package dynamics declares state_dim={state_dim}; the "
                f"planar steering/goal fixtures assume [x, y, theta] — "
                f"probe this planner by hand or extend the kit's fixtures")

    env_cls = resolve_symbol(package, "Environment")
    if not inspect.isclass(env_cls):
        make_environment = _StubEnvironment
    else:
        bound = 10.0 * torch.ones(state_dim, dtype=torch.float64)
        env_kwargs = _fill_kwargs(env_cls, {
            "name": "probe-empty-scenario",
            "state_bounds_lower": -bound, "state_bounds_upper": bound,
            "obstacles": [], "seed": seed,
        })
        if isinstance(env_kwargs, str):
            return (f"Environment requires unmapped constructor parameter "
                    f"{env_kwargs!r} — cannot build an obstacle-free fixture "
                    f"from the package's own class")
        try:
            env_cls(**env_kwargs)
        except Exception as e:  # noqa: BLE001
            return (f"Environment(...) raised on the obstacle-free fixture: "
                    f"{type(e).__name__}: {e}")

        def make_environment(cls=env_cls, kw=env_kwargs):
            return cls(**{k: (list(v) if isinstance(v, list) else v)
                          for k, v in kw.items()})

    return {
        "fn": fn, "planner_name": planner_name,
        "dynamics": dynamics, "make_environment": make_environment,
        "state_dim": state_dim, "seed": seed,
    }


def _planner_kwargs(fn, start, goal, dynamics, dt: float, seed: int,
                    make_environment=None):
    """Name-mapped kwargs for a planner callable; returns the kwargs dict or
    the name of an unmappable required parameter (the _fill_kwargs contract).
    ``make_environment`` is the kit's factory for the package's own
    Environment class; the duck-typed stub remains the no-kit fallback.
    Package dynamics classes carry no V_max (probe-stub convention only), so
    the d_safe candidate guards the attribute."""
    env_factory = make_environment or _StubEnvironment
    return _fill_kwargs(fn, {
        # Probe-scale search budgets for sampling-based planners: the node's
        # calibration priors sacrifice time budget / primitive count /
        # iteration caps first, and these are the delivered iDb-RRT
        # notebook's own demo-rescale values (its defaults blow the probe
        # timeout even on an empty map). Only bind when the planner names
        # them; the probes need the planner's DECISION, not convergence.
        "max_iterations": 500, "to_max_iterations": 5,
        "initial_primitive_count": 50,
        "start": start, "robot_state": start, "state": start,
        "goal": goal, "goal_position": goal,
        "environment": env_factory(), "env": env_factory(),
        "obstacle_positions": [], "obstacles": [],
        "obstacle_current": [], "obstacle_previous": [],
        "dynamics": dynamics, "dt": dt, "seed": seed,
        "d_safe": getattr(dynamics, "V_max", 1.0) / 4,
    })


def _first_decision(result, start):
    """(state_before, control) of the planner's first decision, or None.
    Accepts a trajectory-returning result (.trajectory of (state, control)
    pairs, or a bare list of pairs) and a bare-control result."""
    traj = _trajectory_of(result)
    if traj:
        return traj[0]
    candidate = getattr(result, "control", result)
    try:
        if len(candidate) == 2:
            float(candidate[0]), float(candidate[1])
            return start, candidate
    except (TypeError, ValueError):
        pass
    return None


def _trajectory_of(result):
    """[(state, control), ...] or None. Accepts the pairs-list shape (pdwa)
    and the paradigm-contract Trajectory object with parallel ``.states``/
    ``.controls`` lists (iDb-RRT) — states has one trailing entry more than
    controls, so zip yields (state_i, control_i) up to the last control,
    matching the pairs convention."""
    traj = getattr(result, "trajectory", result)
    if traj is None:
        return None
    states = getattr(traj, "states", None)
    controls = getattr(traj, "controls", None)
    if (isinstance(states, (list, tuple)) and states
            and isinstance(controls, (list, tuple)) and controls):
        return list(zip(states, controls))
    if not isinstance(traj, (list, tuple)) or not traj:
        return None
    pairs = []
    for entry in traj:
        if isinstance(entry, (list, tuple)) and len(entry) == 2:
            pairs.append((entry[0], entry[1]))
        else:
            return None
    return pairs


def _abs_heading_error(state, goal) -> float:
    bearing = math.atan2(float(goal[1]) - float(state[1]),
                         float(goal[0]) - float(state[0]))
    err = bearing - float(state[2])
    return abs(math.atan2(math.sin(err), math.cos(err)))


def _call_with_timeout(fn, kwargs: dict, timeout_s: float | None):
    """("ok", result) | ("timeout", None); exceptions re-raise in the caller.
    The worker is a daemon thread — a timed-out planner keeps spinning in the
    background until its own iteration cap, which is acceptable for a probe
    process but the reason the budget exists at all (a planner that cannot
    leave the origin on an empty map burns its full cap on every call)."""
    if timeout_s is None:
        return "ok", fn(**kwargs)
    import threading  # noqa: PLC0415

    box: dict = {}

    def _worker():
        try:
            box["result"] = fn(**kwargs)
        except Exception as e:  # noqa: BLE001
            box["error"] = e

    t = threading.Thread(target=_worker, daemon=True)
    t.start()
    t.join(timeout_s)
    if t.is_alive():
        return "timeout", None
    if "error" in box:
        raise box["error"]
    return "ok", box["result"]


def _goal_variants(start, goal_xy, state_dim: int) -> list:
    """The two real goal conventions (the two concrete cases, 2026-07-04):
    a 2-dim goal POSITION (pdwa's Eq-7/Eq-8 shape) and a state_dim goal
    STATE (iDb-RRT's ``x_g``, template docstring "goal state, shape
    (state_dim,)") — position plus the arrival bearing from start, zeros for
    any higher state dims. Tried in that order and pinned on first success
    (the KD kit's try-shapes precedent)."""
    import torch  # noqa: PLC0415

    variants = [goal_xy]
    if state_dim and state_dim > 2:
        bearing = math.atan2(float(goal_xy[1]) - float(start[1]),
                             float(goal_xy[0]) - float(start[0]))
        padded = [float(goal_xy[0]), float(goal_xy[1]), bearing]
        padded += [0.0] * (state_dim - 3)
        variants.append(torch.tensor(padded, dtype=torch.float64))
    return variants


def _goal_ladder_caller(kit, start, dt: float, per_call_timeout_s):
    """A ``plan(goal_xy, seed)`` closure over the kit that walks the goal
    ladder on the first call and pins the winning shape for the rest of the
    probe. Returns ("ok", result) | ("timeout", None) |
    ("unmapped", param_name) | ("error", exception). Only an exception
    advances the ladder — a timeout means the goal shape bound and the
    planner ran."""
    pin: dict = {}

    def _plan(goal_xy, seed: int):
        variants = _goal_variants(start, goal_xy, kit["state_dim"])
        indices = [pin["i"]] if "i" in pin else range(len(variants))
        last: Exception | None = None
        for i in indices:
            kwargs = _planner_kwargs(
                kit["fn"], start, variants[i], kit["dynamics"], dt, seed,
                make_environment=kit["make_environment"])
            if isinstance(kwargs, str):
                return ("unmapped", kwargs)
            try:
                outcome, result = _call_with_timeout(
                    kit["fn"], kwargs, per_call_timeout_s)
            except Exception as e:  # noqa: BLE001 — generated planners fail freely
                last = e
                continue
            pin["i"] = i
            return (outcome, result)
        return ("error", last)

    return _plan


def probe_steering_responsiveness(
    module: ModuleType,
    planner_name: str,
    seeds: tuple[int, ...] = (0, 1, 2),
    dt: float = 0.1,
    per_call_timeout_s: float | None = 10.0,
    kit: dict | str | None = None,
) -> ProbeVerdict:
    """MP-3: first chosen control must reduce heading error toward a goal 90°
    off heading, on both sides, across seeds. A side conclusively failing
    short-circuits to the flag (no need to spend budget on the other side);
    a per-call timeout keeps pathological planners from stalling the
    battery. ``kit`` is a prebuilt ``mp_planner_kit`` (or its reason string);
    None builds one from ``module``."""
    try:
        import torch  # noqa: PLC0415
    except ImportError:
        return ProbeVerdict("MP-3", "unprobeable", "torch required")

    if kit is None:
        kit = mp_planner_kit(module, planner_name)
    if isinstance(kit, str):
        return ProbeVerdict("MP-3", "unprobeable", kit)
    dynamics = kit["dynamics"]
    start = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float64)
    plan = _goal_ladder_caller(kit, start, dt, per_call_timeout_s)

    side_summary = []
    for side, label in ((1.0, "left"), (-1.0, "right")):
        goal = torch.tensor([0.0, 3.0 * side], dtype=torch.float64)
        improved = 0
        tried = 0
        for seed in seeds:
            outcome, result = plan(goal, seed)
            if outcome == "unmapped":
                return ProbeVerdict(
                    "MP-3", "unprobeable",
                    f"{planner_name} requires unmapped parameter {result!r}")
            if outcome == "error":
                return ProbeVerdict(
                    "MP-3", "unprobeable",
                    f"planner raised {type(result).__name__}: {result}")
            if outcome == "timeout":
                return ProbeVerdict(
                    "MP-3", "unprobeable",
                    f"planner exceeded the {per_call_timeout_s:.0f}s per-call "
                    f"budget on a trivial obstacle-free fixture — "
                    f"pathological runtime is itself a signal (see the "
                    f"goal-progress check below); rerun by hand with "
                    f"per_call_timeout_s=None to adjudicate steering")
            decision = _first_decision(result, start)
            if decision is None:
                status = str(getattr(result, "status", "") or "")
                if status:
                    return ProbeVerdict(
                        "MP-3", "unprobeable",
                        f"planner produced no plan on the trivial fixture "
                        f"(status={status!r}) — no first decision to judge; "
                        f"the no-plan behavior itself is adjudicated by the "
                        f"goal-progress check below")
                return ProbeVerdict(
                    "MP-3", "unprobeable",
                    f"{planner_name} returned neither a trajectory of "
                    f"(state, control) pairs nor a bare control")
            state0, control = decision
            state0_t = torch.as_tensor(state0, dtype=torch.float64)
            control_t = torch.as_tensor(control, dtype=torch.float64)
            nxt = dynamics.step(state0_t, control_t, dt)
            tried += 1
            if _abs_heading_error(nxt, goal) < _abs_heading_error(
                    state0_t, goal) - 1e-9:
                improved += 1
        side_summary.append(f"{label} goal: {improved}/{tried}")
        if improved < max(1, tried - 1):  # at most one non-improving seed,
            # and a single tried seed must improve
            return ProbeVerdict(
                "MP-3", "flag_for_researcher",
                f"first chosen control does not consistently turn toward a "
                f"goal 90° off heading ({'; '.join(side_summary)}) — "
                f"steering may be an argmax tie-break artifact of an "
                f"objective insensitive to the turn-rate variable (the May "
                f"pdwa class); verify against the paper's objective "
                f"definition",
                finding_class="M-005")
    return ProbeVerdict(
        "MP-3", "pass",
        f"first control reduces heading error on both sides across "
        f"{len(seeds)} seeds ({'; '.join(side_summary)})")


def probe_goal_progress(
    module: ModuleType,
    planner_name: str,
    seed: int = 0,
    dt: float = 0.1,
    per_call_timeout_s: float | None = 20.0,
    kit: dict | str | None = None,
) -> ProbeVerdict:
    """MP-4: on an obstacle-free fixture the planner ends closer to the goal
    than it started; a success-claiming result must end near the goal. The
    per-call budget is generous (one conclusive call): a spinning planner
    burning its full internal iteration cap still finishes within it.
    ``kit`` is a prebuilt ``mp_planner_kit`` (or its reason string); None
    builds one from ``module``."""
    try:
        import torch  # noqa: PLC0415
    except ImportError:
        return ProbeVerdict("MP-4", "unprobeable", "torch required")

    if kit is None:
        kit = mp_planner_kit(module, planner_name)
    if isinstance(kit, str):
        return ProbeVerdict("MP-4", "unprobeable", kit)
    dynamics = kit["dynamics"]
    start = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float64)
    goal = torch.tensor([4.0, 0.0], dtype=torch.float64)
    plan = _goal_ladder_caller(kit, start, dt, per_call_timeout_s)

    outcome, result = plan(goal, seed)
    if outcome == "unmapped":
        return ProbeVerdict(
            "MP-4", "unprobeable",
            f"{planner_name} requires unmapped parameter {result!r}")
    if outcome == "error":
        return ProbeVerdict("MP-4", "unprobeable",
                            f"planner raised {type(result).__name__}: {result}")
    if outcome == "timeout":
        return ProbeVerdict(
            "MP-4", "unprobeable",
            f"planner exceeded the {per_call_timeout_s:.0f}s per-call budget "
            f"on a trivial obstacle-free fixture — pathological runtime is "
            f"itself a signal; rerun by hand with per_call_timeout_s=None")
    traj = _trajectory_of(result)
    status = str(getattr(result, "status", "") or "")
    if not traj and status:
        # A planner RESULT (it carries a status) with no trajectory: the
        # planner ran within budget and decided it could not plan — on a
        # trivial obstacle-free fixture. That is behavioral evidence, not
        # unprobeability (the delivered iDb-RRT's own demo also returned
        # status=timeout with zero optimization attempts, 2026-07-04).
        if status.lower() in ("success", "solved", "converged"):
            return ProbeVerdict(
                "MP-4", "fail",
                f"result claims status={status!r} but carries no trajectory "
                f"— solved-implies-a-path consistency violated")
        return ProbeVerdict(
            "MP-4", "flag_for_researcher",
            f"planner returned status={status!r} with no trajectory on a "
            f"trivial obstacle-free fixture — either it cannot solve "
            f"trivial instances (the planner_no_path_found smoke-bug class) "
            f"or it needs more than the probe-scale search budget "
            f"(max_iterations=500, 5 outer iterations); run the package's "
            f"own demo at the paper's budget to adjudicate",
            evidence=f"stats={str(getattr(result, 'stats', ''))[:200]}")
    if not traj:
        return ProbeVerdict(
            "MP-4", "unprobeable",
            f"{planner_name} returns no trajectory — goal progress needs "
            f"the executed path (bare-control selectors are probed by MP-3)")

    last_state = torch.as_tensor(traj[-1][0], dtype=torch.float64)
    last_control = torch.as_tensor(traj[-1][1], dtype=torch.float64)
    end = dynamics.step(last_state, last_control, dt)
    d_start = float(torch.hypot(goal[0] - start[0], goal[1] - start[1]))
    d_end = float(torch.hypot(goal[0] - end[0], goal[1] - end[1]))
    status = str(getattr(result, "status", "") or "")

    if d_end >= d_start - 1e-6:
        return ProbeVerdict(
            "MP-4", "fail",
            f"no goal progress on an obstacle-free fixture: start "
            f"{d_start:.2f} m from goal, end {d_end:.2f} m after "
            f"{len(traj)} steps (status={status or 'n/a'})",
            evidence=f"trajectory_length={len(traj)}")
    if status.lower() in ("success", "solved", "converged") and d_end > 1.0:
        return ProbeVerdict(
            "MP-4", "fail",
            f"result claims status={status!r} but ends {d_end:.2f} m from "
            f"the goal — solved-implies-near-goal consistency violated")
    return ProbeVerdict(
        "MP-4", "pass",
        f"goal distance {d_start:.1f} → {d_end:.2f} m over {len(traj)} "
        f"steps (status={status or 'n/a'})")


# ---------------------------------------------------------------------------
# Scenario-fidelity detectors (R2C-025 slice B; SC-1/2/3, finding class M-006)
# ---------------------------------------------------------------------------
# Family-owned observers read the EXECUTED setup namespace (built by
# probes/scenario_setup.py's bounded child subprocess) and pure comparators
# judge the observation against the captured paper value from
# method_spec.scenario_assumptions. Two harvest channels on purpose: the
# paradigm Environment contract (`.obstacles` on an instantiated object) AND
# obstacle-named notebook-local lists — the real pdwa delivery keeps its
# moving obstacles in `obstacle_states` dicts while its Environment holds two
# static rectangles, so an Environment-only detector would inspect the wrong
# scene.

from probes.scenario_setup import (  # noqa: E402 — grouped with its detectors
    ScenarioComparison, ScenarioDetector)

_GEOMETRY_ALIASES = {
    "circle": "circle", "circles": "circle", "circular": "circle",
    "disc": "circle", "disk": "circle",
    "rectangle": "rectangle", "rectangles": "rectangle",
    "rectangular": "rectangle", "rect": "rectangle",
    "box": "rectangle", "boxes": "rectangle",
    "square": "rectangle", "squares": "rectangle", "aabb": "rectangle",
    "polygon": "polygon", "polygons": "polygon", "polygonal": "polygon",
    "poly": "polygon",
}

_AGENT_NAME_TOKENS = {
    "pedestrian", "pedestrians", "crowd", "human", "humans",
    "person", "people", "agent", "agents",
}

_MOTION_FIELD_NAMES = {
    "vx", "vy", "velocity", "velocities", "vel", "speed", "heading",
    "x_prev", "y_prev", "prev_x", "prev_y", "dx", "dy", "omega",
}


def _normalize_geometry(token: object) -> str | None:
    if not isinstance(token, str) or not token.strip():
        return None
    token = token.strip().lower()
    return _GEOMETRY_ALIASES.get(token, token)


def _name_tokens(name: str) -> set[str]:
    return set(re.split(r"[^a-z]+", str(name).lower())) - {""}


def _record_fields(entry: object) -> dict | None:
    """A record's field mapping: dict entries as-is, plain objects via their
    instance dict. Modules, classes, and callables are never records."""
    if isinstance(entry, dict):
        return {str(k): v for k, v in entry.items()}
    if (entry is None or inspect.ismodule(entry) or inspect.isclass(entry)
            or inspect.isroutine(entry)):
        return None
    fields = getattr(entry, "__dict__", None)
    if isinstance(fields, dict) and fields:
        return {str(k): v for k, v in fields.items()}
    return None


def obstacle_records(namespace: dict) -> list[tuple[str, dict]]:
    """(source_name, fields) for every obstacle-shaped record reachable from
    the executed setup namespace: entries of lists bound to obstacle-named
    variables, plus entries of any object's ``.obstacles`` list (the paradigm
    Environment contract)."""
    records: list[tuple[str, dict]] = []
    seen: set[int] = set()

    def _collect(source: str, entries) -> None:
        for entry in entries:
            if id(entry) in seen:
                continue
            fields = _record_fields(entry)
            if fields is not None:
                seen.add(id(entry))
                records.append((source, fields))

    for name, value in namespace.items():
        if name.startswith("_"):
            continue
        if inspect.ismodule(value) or inspect.isclass(value) or inspect.isroutine(value):
            continue
        if "obstacle" in name.lower() and isinstance(value, (list, tuple)):
            _collect(name, value)
            continue
        try:
            nested = getattr(value, "obstacles", None)
        except Exception:  # noqa: BLE001 — arbitrary generated objects
            nested = None
        if isinstance(nested, (list, tuple)):
            _collect(f"{name}.obstacles", nested)
    return records


def _record_geometry(fields: dict) -> str | None:
    """The geometry family of one obstacle record: an explicit type-ish field
    wins; otherwise the template contract's key shapes (x/y/radius circles,
    xmin..ymax rectangles, vertex-list polygons)."""
    lower = {str(k).lower(): v for k, v in fields.items()}
    for key in ("type", "kind", "shape", "geometry"):
        value = lower.get(key)
        if isinstance(value, str) and value.strip():
            return _normalize_geometry(value)
    keys = set(lower)
    if {"xmin", "xmax", "ymin", "ymax"} <= keys:
        return "rectangle"
    if {"x", "y", "width", "height"} <= keys:
        return "rectangle"
    if "radius" in keys and ({"x", "y"} <= keys or "center" in keys
                             or {"cx", "cy"} <= keys):
        return "circle"
    for key in ("vertices", "points", "corners"):
        value = lower.get(key)
        if isinstance(value, (list, tuple)) and len(value) >= 3:
            return "polygon"
    return None


def observe_scenario_geometry(namespace: dict) -> dict:
    """SC-1 observer: geometry families across every setup obstacle record."""
    records = obstacle_records(namespace)
    if not records:
        return {"bound": False,
                "reason": "no obstacle records found in the executed setup"}
    classified = [_record_geometry(fields) for _, fields in records]
    return {
        "bound": True,
        "geometry_types": sorted({g for g in classified if g is not None}),
        "unclassified_records": sum(1 for g in classified if g is None),
        "record_count": len(records),
        "sources": sorted({name for name, _ in records}),
    }


def _paper_geometry_set(paper_value: object) -> set[str] | None:
    if isinstance(paper_value, str):
        normalized = _normalize_geometry(paper_value)
        return {normalized} if normalized else None
    if isinstance(paper_value, (list, tuple)):
        out = {g for v in paper_value if (g := _normalize_geometry(v))}
        return out or None
    return None


def _join(items) -> str:
    return ", ".join(str(i) for i in items)


def compare_scenario_geometry(paper_value: object, obs: dict) -> ScenarioComparison:
    if not obs.get("bound"):
        return ScenarioComparison(
            "flag_for_researcher",
            f"could not bind the executed setup "
            f"({obs.get('reason', 'no observation')}) — researcher judgment "
            f"needed on the paper's stated obstacle geometry",
            reason="setup_unbindable")
    allowed = _paper_geometry_set(paper_value)
    if allowed is None:
        return ScenarioComparison(
            "flag_for_researcher",
            f"the captured obstacle-geometry value {paper_value!r} has a "
            f"shape this detector cannot interpret — researcher judgment "
            f"needed",
            reason="assumption_uninterpretable")
    observed = set(obs.get("geometry_types") or [])
    sources = _join(obs.get("sources") or [])
    total = int(obs.get("record_count") or 0)
    violations = sorted(observed - allowed)
    if violations:
        return ScenarioComparison(
            "fail",
            f"the executed setup builds {_join(violations)} obstacles "
            f"although the paper's stated geometry is "
            f"{_join(sorted(allowed))} — the demo contradicts the paper's "
            f"scenario assumption",
            evidence=f"setup objects: {sources}",
            reason="scenario_mismatch")
    unclassified = int(obs.get("unclassified_records") or 0)
    if unclassified:
        return ScenarioComparison(
            "flag_for_researcher",
            f"{unclassified} of {total} setup obstacle records carry no "
            f"recognizable geometry — confirm the demo stays within the "
            f"paper's stated {_join(sorted(allowed))} obstacles",
            evidence=f"setup objects: {sources}",
            reason="setup_unbindable")
    return ScenarioComparison(
        "pass",
        f"all {total} setup obstacle records use "
        f"{_join(sorted(observed))} geometry, within the paper's stated "
        f"{_join(sorted(allowed))}",
        evidence=f"setup objects: {sources}",
        reason="scenario_match")


def _population_counts(namespace: dict) -> dict[str, int]:
    counts: dict[str, int] = {}
    for name, value in namespace.items():
        if name.startswith("_"):
            continue
        if not (_name_tokens(name) & _AGENT_NAME_TOKENS):
            continue
        if isinstance(value, (list, tuple, set)):
            counts[name] = len(value)
        elif isinstance(value, int) and not isinstance(value, bool):
            counts[name] = value
    for source, fields in obstacle_records(namespace):
        lower = {str(k).lower(): v for k, v in fields.items()}
        for key in ("type", "kind", "agent_type"):
            value = lower.get(key)
            if isinstance(value, str) and value.strip().lower() in _AGENT_NAME_TOKENS:
                label = f"{source} records typed {value.strip().lower()!r}"
                counts[label] = counts.get(label, 0) + 1
                break
    return counts


def observe_scenario_population(namespace: dict) -> dict:
    """SC-2 observer: sizes of every agent-like population in the setup."""
    counts = _population_counts(namespace)
    if not counts:
        return {"bound": False,
                "reason": ("no pedestrian- or agent-like entities found in "
                           "the executed setup")}
    return {"bound": True, "counts": counts}


_EMPTY_POPULATION_CATEGORIES = {"none", "empty", "zero"}


def compare_scenario_population(paper_value: object, obs: dict) -> ScenarioComparison:
    if not obs.get("bound"):
        return ScenarioComparison(
            "flag_for_researcher",
            f"{obs.get('reason', 'could not bind the executed setup')} — "
            f"cannot confirm the paper's stated population; researcher "
            f"judgment needed",
            reason="setup_unbindable")
    counts = {str(k): int(v) for k, v in (obs.get("counts") or {}).items()}
    described = "; ".join(f"{k}: {v}" for k, v in sorted(counts.items()))

    def _single_count() -> int | None:
        distinct = set(counts.values())
        return distinct.pop() if len(distinct) == 1 else None

    if isinstance(paper_value, int) and not isinstance(paper_value, bool):
        count = _single_count()
        if count is None:
            return ScenarioComparison(
                "flag_for_researcher",
                f"the setup binds several agent-like populations with "
                f"differing sizes ({described}) — cannot attribute the "
                f"paper's stated count of {paper_value}; researcher judgment "
                f"needed",
                reason="ambiguous_binding")
        if count == paper_value:
            return ScenarioComparison(
                "pass",
                f"the executed setup contains {count} agent-like entities, "
                f"matching the paper's stated count ({described})",
                reason="scenario_match")
        return ScenarioComparison(
            "fail",
            f"the executed setup contains {count} agent-like entities "
            f"although the paper states {paper_value} ({described})",
            reason="scenario_mismatch")

    if isinstance(paper_value, str) and paper_value.strip():
        category = paper_value.strip().lower()
        peak = max(counts.values())
        expects_empty = category in _EMPTY_POPULATION_CATEGORIES
        if expects_empty:
            if peak == 0:
                return ScenarioComparison(
                    "pass",
                    f"the executed setup contains no agent-like entities, "
                    f"matching the paper's stated {category!r} scenario "
                    f"({described})",
                    reason="scenario_match")
            return ScenarioComparison(
                "fail",
                f"the paper states a {category!r} scenario but the executed "
                f"setup contains agent-like entities ({described})",
                reason="scenario_mismatch")
        if peak == 0:
            return ScenarioComparison(
                "fail",
                f"the paper states a {category!r} scenario but the executed "
                f"setup contains zero agent-like entities ({described})",
                reason="scenario_mismatch")
        return ScenarioComparison(
            "pass",
            f"the executed setup contains agent-like entities ({described}); "
            f"the paper's stated category is {category!r} — presence is "
            f"confirmed, the category itself is not numerically checkable",
            reason="scenario_match")

    if isinstance(paper_value, dict):
        agent_type = str(paper_value.get("agent_type") or "").strip().lower()
        if not agent_type:
            return ScenarioComparison(
                "flag_for_researcher",
                f"the captured agent-population value {paper_value!r} names "
                f"no agent_type — researcher judgment needed",
                reason="assumption_uninterpretable")
        matching = {
            k: v for k, v in counts.items()
            if agent_type in k.lower()
            or (_name_tokens(k) & {agent_type, agent_type + "s"})
        }
        if not matching:
            return ScenarioComparison(
                "flag_for_researcher",
                f"no {agent_type}-like entities were found in the executed "
                f"setup ({described or 'no agent-like entities bound'}) — "
                f"cannot confirm the paper's stated {agent_type} population; "
                f"researcher judgment needed",
                reason="setup_unbindable")
        matched_desc = "; ".join(f"{k}: {v}" for k, v in sorted(matching.items()))
        explicit = paper_value.get("count", paper_value.get("minimum_count"))
        peak = max(matching.values())
        if isinstance(explicit, int) and not isinstance(explicit, bool):
            distinct = set(matching.values())
            if len(distinct) > 1:
                return ScenarioComparison(
                    "flag_for_researcher",
                    f"several {agent_type} populations with differing sizes "
                    f"({matched_desc}) — cannot attribute the paper's stated "
                    f"count of {explicit}; researcher judgment needed",
                    reason="ambiguous_binding")
            count = distinct.pop()
            if count == explicit:
                return ScenarioComparison(
                    "pass",
                    f"the executed setup contains {count} {agent_type}(s), "
                    f"matching the paper's stated count ({matched_desc})",
                    reason="scenario_match")
            return ScenarioComparison(
                "fail",
                f"the executed setup contains {count} {agent_type}(s) "
                f"although the paper states {explicit} ({matched_desc})",
                reason="scenario_mismatch")
        expects_presence = bool(paper_value.get("minimum_present", True))
        if expects_presence:
            if peak > 0:
                return ScenarioComparison(
                    "pass",
                    f"the executed setup contains {agent_type}-like entities "
                    f"({matched_desc}), satisfying the paper's stated "
                    f"presence requirement",
                    reason="scenario_match")
            return ScenarioComparison(
                "fail",
                f"the paper requires {agent_type}s to be present but the "
                f"executed setup contains zero ({matched_desc})",
                reason="scenario_mismatch")
        if peak == 0:
            return ScenarioComparison(
                "pass",
                f"the executed setup contains no {agent_type}s "
                f"({matched_desc}), matching the paper's stated absence",
                reason="scenario_match")
        return ScenarioComparison(
            "fail",
            f"the paper states {agent_type}s are absent but the executed "
            f"setup contains them ({matched_desc})",
            reason="scenario_mismatch")

    return ScenarioComparison(
        "flag_for_researcher",
        f"the captured agent-population value {paper_value!r} has a shape "
        f"this detector cannot interpret — researcher judgment needed",
        reason="assumption_uninterpretable")


def observe_scenario_dynamics(namespace: dict) -> dict:
    """SC-3 observer: motion-state evidence on setup-time obstacle records."""
    records = obstacle_records(namespace)
    velocity_sources = sorted(
        name for name, value in namespace.items()
        if not name.startswith("_")
        and "obstacle" in name.lower()
        and (_name_tokens(name) & {"velocity", "velocities", "vel",
                                   "speed", "speeds"})
        and isinstance(value, (list, tuple)) and len(value) > 0)
    if not records and not velocity_sources:
        return {"bound": False,
                "reason": "no obstacle records found in the executed setup"}
    motion_fields = sorted({
        str(k) for _, fields in records for k in fields
        if str(k).lower() in _MOTION_FIELD_NAMES})
    return {
        "bound": True,
        "motion_state": bool(motion_fields or velocity_sources),
        "motion_fields": motion_fields,
        "velocity_sources": velocity_sources,
        "record_count": len(records),
        "sources": sorted({name for name, _ in records}
                          | set(velocity_sources)),
    }


def compare_scenario_dynamics(paper_value: object, obs: dict) -> ScenarioComparison:
    if not obs.get("bound"):
        return ScenarioComparison(
            "flag_for_researcher",
            f"could not bind the executed setup "
            f"({obs.get('reason', 'no observation')}) — researcher judgment "
            f"needed on the paper's stated obstacle dynamics",
            reason="setup_unbindable")
    stated = paper_value.strip().lower() if isinstance(paper_value, str) else ""
    if stated not in ("static", "dynamic", "mixed"):
        return ScenarioComparison(
            "flag_for_researcher",
            f"the captured obstacle-dynamics value {paper_value!r} is not "
            f"one of static/dynamic/mixed — researcher judgment needed",
            reason="assumption_uninterpretable")
    motion = bool(obs.get("motion_state"))
    fields = _join((obs.get("motion_fields") or [])
                   + (obs.get("velocity_sources") or []))
    sources = _join(obs.get("sources") or [])
    if stated in ("dynamic", "mixed"):
        if motion:
            return ScenarioComparison(
                "pass",
                f"setup-time obstacle state carries motion fields ({fields}) "
                f"— consistent with the paper's {stated} scenario",
                evidence=f"setup objects: {sources}",
                reason="scenario_match")
        # Deliberately NOT a researcher flag: motion realized by direct
        # mutation inside the demo loop leaves no setup-time trace, and that
        # loop is exactly what the dynamic-scenario loop check (MP-1)
        # adjudicates — flagging here would double-judge and wrongly demote
        # loop-mutating deliveries MP-1 passes.
        return ScenarioComparison(
            "unprobeable",
            f"setup-time obstacle state carries no motion fields; whether "
            f"the demo loop actually moves the obstacles is adjudicated by "
            f"the dynamic-scenario loop check",
            evidence=f"setup objects: {sources}",
            reason="setup_state_motion_undecidable")
    if motion:
        return ScenarioComparison(
            "flag_for_researcher",
            f"the setup carries obstacle motion state ({fields}) although "
            f"the paper states a static scenario — confirm the demo does not "
            f"move obstacles",
            evidence=f"setup objects: {sources}",
            reason="static_claim_motion_state")
    return ScenarioComparison(
        "pass",
        f"setup-time obstacle state carries no motion fields — consistent "
        f"with the paper's static scenario",
        evidence=f"setup objects: {sources}",
        reason="scenario_match")


SCENARIO_DETECTORS = {
    "motion_planning.scenario_geometry": ScenarioDetector(
        "SC-1", observe_scenario_geometry, compare_scenario_geometry),
    "motion_planning.scenario_population": ScenarioDetector(
        "SC-2", observe_scenario_population, compare_scenario_population),
    "motion_planning.scenario_dynamics_setup": ScenarioDetector(
        "SC-3", observe_scenario_dynamics, compare_scenario_dynamics),
}
