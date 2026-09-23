"""UB-4 direction-consistency + UB-7 argmax-term-ablation (objective probes).

Two probes over the *composition* of a scoring objective, grounded in the two
concrete dead/inverted-term cases:

- pdwa 2026-06-10: six `compute_j_*` terms summed into a maximized objective.
  Four of six consume no decision variable (can never change the per-round
  argmax — the M-005 recurrence), and the exponential obstacle penalties are
  ADDED, so the maximized objective strictly prefers obstacle-adjacent worlds.
- gbald 2026-06-09: `select_batch` composes prior/BALD/ranking/core-set
  channels; the geometric prior was ≡1.0 at the live scale, so its channel
  could never change the selection (the inert-contribution / M-004 class).

Mechanisms (shared AST machinery, two composition styles):

- **Additive-chain analysis** (UB-4 + UB-7 case 1): discover module-level term
  functions called by the scorer, parse the `j = a + b - c` chain for signs,
  then (UB-4) evaluate obstacle-consuming terms on matched near/far obstacle
  world pairs — a maximized objective must not strictly prefer the known-worse
  world on EVERY pair (strictness is the false-positive guard) — and (UB-7)
  check each chain term can vary across a candidate grid at all.
- **Output-perturbation ablation** (UB-7 case 2): for composite selectors,
  perturb each term channel's live output — cyclic shifts (order) plus a
  dup-drop substitution (set membership) — all deterministic; if the
  composite's selection never changes under any of them, that channel is dead
  or discarded. Constant outputs map to themselves under every mode, so the
  ≡1.0-prior class is caught by construction. Channels are discovered
  TRANSITIVELY (june9's dead prior is two calls deep: select_batch →
  geometric_ranking → compute_geometric_prior) and patched at the module
  namespace that resolves each call. UB-7 is flag-tier defense in depth, not
  an exhaustive deadness proof.

Verdicts follow the catalog: UB-4 fails only on strict inversion; UB-7 always
routes to flag_for_researcher (legitimately-weak terms exist; M-005 routing,
never auto-fix). Composition we cannot parse is `unprobeable`, never guessed —
nonadditive composites are CT-4 LLM-review territory.

torch is imported inside probe bodies (fixtures.py convention): absent torch
degrades to `unprobeable`, never a crash and never a silent pass.
"""

from __future__ import annotations

import ast
import inspect
import textwrap
from pathlib import Path
from types import ModuleType

import numpy as np

from probes import ProbeVerdict
from probes.catalogs.term_ablation import PROBE_CATALOG as _PROBE_CATALOG

PROBE_CATALOG = _PROBE_CATALOG
from probes.fixtures import PlanningFixture, make_classification_fixture, \
    make_planning_fixture
from probes.trainability import _fill_kwargs

EPS = 1e-9

# Parameter-name pools for signature mapping (the trainability _fill_kwargs
# pattern): unmapped REQUIRED params make a term untestable, never guessed.
CANDIDATE_KEYS = ("control", "action", "candidate", "u", "cmd",
                  "control_input", "velocity_command")
OBSTACLE_KEYS = ("obstacle_positions", "obstacles", "obstacle_current",
                 "obstacle_list", "obstacle_previous", "obstacle_prev",
                 "obstacles_previous", "obstacle_future", "obstacle_predicted")


# ---------------------------------------------------------------------------
# Shared AST machinery: term discovery + additive-sign parsing
# ---------------------------------------------------------------------------


def _fn_ast(fn) -> ast.AST | None:
    try:
        return ast.parse(textwrap.dedent(inspect.getsource(fn)))
    except (OSError, TypeError, SyntaxError):
        return None


def _module_functions(module: ModuleType) -> dict[str, object]:
    return {
        name: obj for name, obj in vars(module).items()
        if inspect.isfunction(obj) and obj.__module__ == module.__name__
    }


def discover_term_calls(fn, module: ModuleType) -> list[str]:
    """Module-level functions (same module) called by name inside fn."""
    tree = _fn_ast(fn)
    if tree is None:
        return []
    own = _module_functions(module)
    found: list[str] = []
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id in own and node.func.id != fn.__name__
                and node.func.id not in found):
            found.append(node.func.id)
    return found


def discover_term_graph(fn, module: ModuleType) -> dict[str, tuple]:
    """Transitive closure of same-module term calls: {name: (ref_module,
    live_fn)} where ref_module is the namespace that resolves the call (the
    patch target — callers look names up in their own module globals).

    june9's dead geometric prior is two calls deep (select_batch →
    geometric_ranking → compute_geometric_prior); direct discovery alone
    never sees it."""
    seen: dict[str, tuple] = {}
    frontier: list[tuple] = [(fn, module)]
    while frontier:
        cur_fn, cur_mod = frontier.pop()
        for name in discover_term_calls(cur_fn, cur_mod):
            if name in seen:
                continue
            target = getattr(cur_mod, name)
            seen[name] = (cur_mod, target)
            frontier.append((target, inspect.getmodule(target) or cur_mod))
    return seen


def parse_additive_signs(fn, term_names: list[str]) -> dict[str, int] | None:
    """{term: +1|-1} from the largest +/- chain over term results in fn.

    Handles `v = term(...)` locals summed later, direct `term(...) + ...`
    operands, and positive-constant weights (`0.5 * j_x`). Returns None when
    no chain maps >= 2 terms — nonadditive composition, route to CT-4.
    """
    tree = _fn_ast(fn)
    if tree is None:
        return None
    terms = set(term_names)

    var_to_term: dict[str, str] = {}
    for node in ast.walk(tree):
        if (isinstance(node, ast.Assign) and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
                and isinstance(node.value, ast.Call)
                and isinstance(node.value.func, ast.Name)
                and node.value.func.id in terms):
            var_to_term[node.targets[0].id] = node.value.func.id

    def _operand_term(node: ast.AST, sign: int) -> tuple[str, int] | None:
        if isinstance(node, ast.Name) and node.id in var_to_term:
            return var_to_term[node.id], sign
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id in terms):
            return node.func.id, sign
        if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Mult,
                                                                ast.Div)):
            for side, other in ((node.left, node.right),
                                (node.right, node.left)):
                mapped = _operand_term(side, sign)
                if mapped and isinstance(other, ast.Constant) \
                        and isinstance(other.value, (int, float)):
                    if other.value < 0:
                        return mapped[0], -mapped[1]
                    if other.value > 0:
                        return mapped
        return None

    def _flatten(node: ast.AST, sign: int, out: list) -> None:
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
            _flatten(node.left, sign, out)
            _flatten(node.right, sign, out)
        elif isinstance(node, ast.BinOp) and isinstance(node.op, ast.Sub):
            _flatten(node.left, sign, out)
            _flatten(node.right, -sign, out)
        elif isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            _flatten(node.operand, -sign, out)
        elif isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.UAdd):
            _flatten(node.operand, sign, out)
        else:
            out.append((node, sign))

    best: dict[str, int] = {}
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Assign, ast.Return)):
            continue
        value = node.value
        if value is None or not isinstance(value, ast.BinOp):
            continue
        operands: list = []
        _flatten(value, 1, operands)
        chain: dict[str, int] = {}
        for operand, sign in operands:
            mapped = _operand_term(operand, sign)
            if mapped and mapped[0] not in chain:
                chain[mapped[0]] = mapped[1]
        if len(chain) > len(best):
            best = chain
    return best if len(best) >= 2 else None


# ---------------------------------------------------------------------------
# Motion-planning term kit: worlds, candidates, kwargs pool
# ---------------------------------------------------------------------------


class _UnicycleDynamics:
    """Minimal dynamics object matching the generated-package convention."""

    def __init__(self, v_max: float = 1.0, omega_max: float = 1.0,
                 r_robot: float = 0.1):
        self.V_max = v_max
        self.omega_max = omega_max
        self.r_robot = r_robot

    def step(self, state, control, dt):
        import torch  # noqa: PLC0415
        x, y, theta = state[0], state[1], state[2]
        v, omega = control[0], control[1]
        return torch.stack([x + v * torch.cos(theta) * dt,
                            y + v * torch.sin(theta) * dt,
                            theta + omega * dt])


def _mp_kwargs_pool(start, goal, control, world: dict, dt: float,
                    d_safe: float, dynamics) -> dict:
    pool = {
        "robot_state": start, "state": start, "x_robot": start,
        "goal": goal, "goal_position": goal,
        "dt": dt, "d_safe": d_safe, "dynamics": dynamics, "seed": 0,
    }
    for key in CANDIDATE_KEYS:
        pool[key] = control
    current, previous = world["current"], world["previous"]
    for key in OBSTACLE_KEYS:
        pool[key] = previous if "prev" in key else current
    return pool


def _direction_pairs(fixture: PlanningFixture, dynamics, d_safe: float):
    """Matched (near_world, far_world, control) triples: identical except the
    single obstacle sits adjacent to vs far from the robot's next position,
    dead ahead along its post-step heading. previous == current so the pair
    isolates proximity, not motion."""
    import torch  # noqa: PLC0415

    start = torch.tensor(fixture.start, dtype=torch.float64)
    pairs = []
    for v, omega in ((0.6, 0.0), (0.3, 0.4)):
        control = torch.tensor([v, omega], dtype=torch.float64)
        nxt = dynamics.step(start, control, fixture.dt)
        ahead = torch.stack([torch.cos(nxt[2]), torch.sin(nxt[2])])
        for near_d, far_d in ((0.6 * d_safe, 30.0), (0.9 * d_safe, 60.0)):
            def _world(dist: float) -> dict:
                pos = (nxt[:2] + dist * ahead).clone()
                return {"current": [pos], "previous": [pos.clone()]}
            pairs.append((_world(near_d), _world(far_d), control))
    return start, pairs


def probe_direction_consistency(
    module: ModuleType,
    scorer_name: str,
    fixture: PlanningFixture | None = None,
) -> ProbeVerdict:
    """UB-4: the maximized objective must not strictly prefer a known-worse
    (obstacle-adjacent) world on every matched pair."""
    try:
        import torch  # noqa: PLC0415
    except ImportError:
        return ProbeVerdict("UB-4", "unprobeable", "torch required")

    scorer = getattr(module, scorer_name, None)
    if scorer is None:
        return ProbeVerdict("UB-4", "unprobeable",
                            f"scorer {scorer_name!r} not found in module")
    terms = discover_term_calls(scorer, module)
    if not terms:
        return ProbeVerdict("UB-4", "unprobeable",
                            f"{scorer_name} calls no module-level term "
                            f"functions — nothing to direction-test")
    signs = parse_additive_signs(scorer, terms)
    if signs is None:
        return ProbeVerdict(
            "UB-4", "unprobeable",
            f"no additive +/- chain over term results found in "
            f"{scorer_name} — nonadditive composition is reviewer "
            f"territory (CT-4), not guessed here")

    fixture = fixture or make_planning_fixture(seed=0)
    dynamics = _UnicycleDynamics()
    d_safe = dynamics.V_max / 4
    start, pairs = _direction_pairs(fixture, dynamics, d_safe)
    goal = torch.tensor(fixture.goal, dtype=torch.float64)

    inverted: list[tuple[str, float]] = []
    mixed: list[str] = []
    consistent: list[str] = []
    untestable: list[str] = []
    for term, sign in signs.items():
        fn = getattr(module, term, None)
        if fn is None:
            continue
        try:
            param_names = set(inspect.signature(fn).parameters)
        except (TypeError, ValueError):
            untestable.append(term)
            continue
        if not param_names & set(OBSTACLE_KEYS):
            continue  # no obstacle input — direction pairs cannot bind
        diffs: list[float] = []
        try:
            for near_world, far_world, control in pairs:
                kw_n = _fill_kwargs(fn, _mp_kwargs_pool(
                    start, goal, control, near_world, fixture.dt, d_safe,
                    dynamics))
                kw_f = _fill_kwargs(fn, _mp_kwargs_pool(
                    start, goal, control, far_world, fixture.dt, d_safe,
                    dynamics))
                if isinstance(kw_n, str) or isinstance(kw_f, str):
                    untestable.append(term)
                    break
                diffs.append(sign * (float(fn(**kw_n)) - float(fn(**kw_f))))
            else:
                if all(d > EPS for d in diffs):
                    inverted.append((term, max(diffs)))
                elif any(d > EPS for d in diffs):
                    mixed.append(term)
                else:
                    consistent.append(term)
        except Exception as e:  # noqa: BLE001 — generated terms fail arbitrarily
            untestable.append(f"{term} ({type(e).__name__})")

    note = f" (untestable: {', '.join(untestable)})" if untestable else ""
    if inverted:
        worst = ", ".join(f"{t} ({d:+.3g})" for t, d in inverted)
        return ProbeVerdict(
            "UB-4", "fail",
            f"maximized objective strictly prefers the obstacle-adjacent "
            f"world on every pair via: {worst} — penalty terms added (or "
            f"bonuses subtracted) in a maximized objective{note}",
            evidence=f"signs={signs}; pairs={len(pairs)}")
    if mixed:
        return ProbeVerdict(
            "UB-4", "warn",
            f"non-strict direction inconsistency on some pairs: "
            f"{', '.join(mixed)} — boundary effects possible, not a proven "
            f"sign inversion{note}",
            evidence=f"signs={signs}")
    if consistent:
        return ProbeVerdict(
            "UB-4", "pass",
            f"direction-consistent over {len(pairs)} matched pairs: "
            f"{', '.join(consistent)}{note}")
    return ProbeVerdict(
        "UB-4", "unprobeable",
        f"no obstacle-consuming chain term could be evaluated{note}")


def probe_term_ranking_power(
    module: ModuleType,
    scorer_name: str,
    fixture: PlanningFixture | None = None,
    max_candidates: int = 12,
) -> ProbeVerdict:
    """UB-7 (additive-chain case): every claimed scoring term must be able to
    change the per-round argmax — i.e. vary across the candidate grid."""
    try:
        import torch  # noqa: PLC0415
    except ImportError:
        return ProbeVerdict("UB-7", "unprobeable", "torch required")

    scorer = getattr(module, scorer_name, None)
    if scorer is None:
        return ProbeVerdict("UB-7", "unprobeable",
                            f"scorer {scorer_name!r} not found in module")
    terms = discover_term_calls(scorer, module)
    signs = parse_additive_signs(scorer, terms) if terms else None
    if not signs:
        return ProbeVerdict(
            "UB-7", "unprobeable",
            f"no additive scoring chain found in {scorer_name} — "
            f"term-ranking power needs the chain (composite selectors are "
            f"covered by the perturbation arm; the rest is reviewer "
            f"territory, CT-4)")

    fixture = fixture or make_planning_fixture(seed=0)
    dynamics = _UnicycleDynamics()
    d_safe = dynamics.V_max / 4
    start = torch.tensor(fixture.start, dtype=torch.float64)
    goal = torch.tensor(fixture.goal, dtype=torch.float64)
    ahead = torch.tensor([1.0, 0.0], dtype=torch.float64)
    pos = start[:2] + 2 * d_safe * ahead
    world = {"current": [pos], "previous": [pos.clone()]}

    controls = [torch.tensor([v, w], dtype=torch.float64)
                for v in fixture.v_candidates for w in fixture.omega_candidates]
    stride = max(1, len(controls) // max_candidates)
    controls = controls[::stride]

    structural: list[str] = []
    grid_inert: list[str] = []
    live: list[str] = []
    untestable: list[str] = []
    for term in signs:
        fn = getattr(module, term, None)
        if fn is None:
            continue
        try:
            param_names = set(inspect.signature(fn).parameters)
        except (TypeError, ValueError):
            untestable.append(term)
            continue
        if not param_names & set(CANDIDATE_KEYS):
            structural.append(term)
            continue
        values: list[float] = []
        try:
            for control in controls:
                kw = _fill_kwargs(fn, _mp_kwargs_pool(
                    start, goal, control, world, fixture.dt, d_safe, dynamics))
                if isinstance(kw, str):
                    untestable.append(term)
                    break
                values.append(float(fn(**kw)))
            else:
                if max(values) - min(values) <= EPS:
                    grid_inert.append(term)
                else:
                    live.append(term)
        except Exception as e:  # noqa: BLE001
            untestable.append(f"{term} ({type(e).__name__})")

    note = f"; untestable: {', '.join(untestable)}" if untestable else ""
    if structural or grid_inert:
        parts = []
        if structural:
            parts.append(f"{', '.join(structural)} consume no decision "
                         f"variable ({'/'.join(CANDIDATE_KEYS[:3])}...)")
        if grid_inert:
            parts.append(f"{', '.join(grid_inert)} constant over "
                         f"{len(controls)} candidates")
        return ProbeVerdict(
            "UB-7", "flag_for_researcher",
            f"ranking-inert scoring terms — cannot change the argmax within "
            f"an evaluation round: {'; '.join(parts)}{note}",
            evidence=f"signs={signs}; live={live}",
            finding_class="M-005")
    if live:
        return ProbeVerdict(
            "UB-7", "pass",
            f"all {len(live)} chain terms vary across the candidate grid"
            f"{note}")
    return ProbeVerdict("UB-7", "unprobeable",
                        f"no chain term could be evaluated{note}")


# ---------------------------------------------------------------------------
# UB-7 case 2: composite selectors (output-perturbation ablation)
# ---------------------------------------------------------------------------


def _perturb(value, mode: int, index_domain: int | None = None):
    """Deterministic perturbation along the leading axis. Modes >= 1 are
    cyclic shifts (order change, multiset preserved); mode 0 is dup-drop
    (value[0] := value[1] — set change, shape preserved) so index-set outputs
    whose consumers are order-insensitive still register (june9's core-set
    channel false-flagged dead under shifts alone).

    Two membership modes close the gate-then-rerank blind spot (GBALD
    2026-06-30 M-004: a cyclic shift of a top-b INDEX vector permutes the
    same SET, and dup-drop nicks one element, so a "gate the candidate set,
    rerank within it" composite never registers its gate term):
      - mode -1: min-max value swap for 1-D FLOAT vectors — the argmax and
        argmin positions trade values, so any top-k membership the consumer
        induces from the vector changes.
      - mode -2: membership displacement for 1-D INTEGER index vectors when
        `index_domain` (the valid index range, e.g. pool size) is known —
        the first half of the entries is replaced by in-domain indices NOT
        currently present, the honest "does gate membership matter" ablation.

    Constant SCORE outputs map to themselves under every applicable mode —
    that is how the ≡1.0-prior class is caught (min = max makes the swap an
    identity, rolls/dup-drop are identities as before, and displacement
    does not apply to floats). An index vector of duplicated entries is a
    different defect (a degenerate gate, owned by the contract/diversity
    probes) and may legitimately register under displacement. Returns None
    for values a mode does not apply to."""
    if type(value).__module__.startswith("torch"):
        import torch  # noqa: PLC0415
        if isinstance(value, torch.Tensor) and value.ndim >= 1 \
                and value.shape[0] > 1:
            if mode == -1:
                if value.ndim != 1 or not torch.is_floating_point(value):
                    return None
                i = int(torch.argmax(value))
                j = int(torch.argmin(value))
                out = value.clone()
                out[i] = value[j].clone()
                out[j] = value[i].clone()
                return out
            if mode == -2:
                if (value.ndim != 1 or index_domain is None
                        or value.dtype == torch.bool
                        or torch.is_floating_point(value)):
                    return None
                present = {int(v) for v in value.tolist()}
                if not all(0 <= v < index_domain for v in present):
                    return None
                complement = [k for k in range(index_domain)
                              if k not in present]
                if not complement:
                    return None
                out = value.clone()
                n_replace = min((value.shape[0] + 1) // 2, len(complement))
                for k in range(n_replace):
                    out[k] = complement[k]
                return out
            if mode == 0:
                out = value.clone()
                out[0] = value[1]
                return out
            return torch.roll(value, shifts=mode % value.shape[0], dims=0)
        return None
    if isinstance(value, np.ndarray) and value.ndim >= 1 and value.shape[0] > 1:
        if mode == -1:
            if value.ndim != 1 or not np.issubdtype(value.dtype, np.floating):
                return None
            i = int(np.argmax(value))
            j = int(np.argmin(value))
            out = value.copy()
            out[i], out[j] = value[j], value[i]
            return out
        if mode == -2:
            if (value.ndim != 1 or index_domain is None
                    or not np.issubdtype(value.dtype, np.integer)):
                return None
            present = {int(v) for v in value.tolist()}
            if not all(0 <= v < index_domain for v in present):
                return None
            complement = [k for k in range(index_domain) if k not in present]
            if not complement:
                return None
            out = value.copy()
            n_replace = min((value.shape[0] + 1) // 2, len(complement))
            out[:n_replace] = complement[:n_replace]
            return out
        if mode == 0:
            out = value.copy()
            out[0] = value[1]
            return out
        return np.roll(value, mode % value.shape[0], axis=0)
    if isinstance(value, list) and len(value) > 1:
        if mode == -1:
            if not all(isinstance(v, float) for v in value):
                return None
            i = value.index(max(value))
            j = value.index(min(value))
            out = list(value)
            out[i], out[j] = value[j], value[i]
            return out
        if mode == -2:
            if index_domain is None or not all(
                    isinstance(v, (int, np.integer))
                    and not isinstance(v, bool) for v in value):
                return None
            present = {int(v) for v in value}
            if not all(0 <= v < index_domain for v in present):
                return None
            complement = [k for k in range(index_domain) if k not in present]
            if not complement:
                return None
            out = list(value)
            n_replace = min((len(value) + 1) // 2, len(complement))
            out[:n_replace] = complement[:n_replace]
            return out
        if mode == 0:
            return [value[1]] + list(value[1:])
        k = mode % len(value)
        return value[-k:] + value[:-k] if k else list(value)
    if isinstance(value, tuple):
        shifted = [_perturb(v, mode, index_domain) for v in value]
        if all(s is None for s in shifted):
            return None
        return tuple(s if s is not None else v
                     for s, v in zip(shifted, value))
    return None


def _same(a, b) -> bool:
    if type(a).__module__.startswith("torch") \
            or type(b).__module__.startswith("torch"):
        import torch  # noqa: PLC0415
        if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
            return a.shape == b.shape and a.dtype == b.dtype \
                and bool(torch.equal(a, b))
        return False
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        return isinstance(a, np.ndarray) and isinstance(b, np.ndarray) \
            and np.array_equal(a, b)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b))
    try:
        return bool(a == b)
    except Exception:  # noqa: BLE001
        return repr(a) == repr(b)


def probe_composite_term_perturbation(
    module: ModuleType,
    composite_name: str,
    invoke,
    n_shifts: int = 3,
    index_domain: int | None = None,
) -> ProbeVerdict:
    """UB-7 (composite case): perturb each term channel's output; the
    composite's selection must be able to change. `invoke` is a zero-arg
    callable running the composite with fixed inputs AND fixed seeds —
    determinism is verified before any verdict is trusted. `index_domain`
    (when the harness knows it, e.g. the candidate-pool size) enables the
    membership-displacement mode for index-vector channels — without it a
    gate-then-rerank composite's gate term is invisible to perturbation
    (the GBALD 2026-06-30 M-004 blind spot)."""
    composite = getattr(module, composite_name, None)
    if composite is None:
        return ProbeVerdict("UB-7", "unprobeable",
                            f"composite {composite_name!r} not found")
    terms = discover_term_graph(composite, module)
    if not terms:
        return ProbeVerdict(
            "UB-7", "unprobeable",
            f"{composite_name} calls no module-level term functions")

    try:
        baseline = invoke()
        if not _same(invoke(), baseline):
            return ProbeVerdict(
                "UB-7", "unprobeable",
                f"{composite_name} is not deterministic under fixed inputs "
                f"and seeds — per-term verdicts would be noise (seed "
                f"threading is UB-8's territory)")
    except Exception as e:  # noqa: BLE001
        module_file = getattr(module, "__file__", None)
        origin, site = attribute_invocation_crash(
            e, Path(module_file).parent if module_file else None)
        if origin == "subject":
            return ProbeVerdict(
                "UB-7", "unprobeable",
                f"{composite_name} crashed inside the method package's own "
                f"code at {site} ({type(e).__name__}: {e}) — term ablation "
                f"needs a working baseline; selector crashes are "
                f"adjudicated by the acquisition-contract probe's crash "
                f"arm (AL-5)")
        return ProbeVerdict("UB-7", "unprobeable",
                            f"baseline invocation raised "
                            f"{type(e).__name__}: {e}")

    dead: list[str] = []
    alive: list[str] = []
    unperturbable: list[str] = []
    # cyclic shifts, dup-drop, then the membership modes (min-max swap +
    # index displacement) — order matters only for early exit on alive.
    modes = list(range(1, n_shifts)) + [0, -1, -2]
    for term, (ref_mod, live_fn) in terms.items():
        influenced = False
        touched = {"n": 0}
        for mode in modes:
            def _wrapper(*args, _live=live_fn, _mode=mode, _t=touched,
                         **kwargs):
                out = _live(*args, **kwargs)
                shifted = _perturb(out, _mode, index_domain)
                if shifted is None:
                    return out
                _t["n"] += 1
                return shifted

            setattr(ref_mod, term, _wrapper)
            try:
                probe_out = invoke()
            except Exception:  # noqa: BLE001 — perturbed output broke the
                # composite: the channel is consumed downstream, hence live
                influenced = True
                break
            finally:
                setattr(ref_mod, term, live_fn)
            if not _same(probe_out, baseline):
                influenced = True
                break
        if influenced:
            alive.append(term)
        elif touched["n"] == 0:
            unperturbable.append(term)
        else:
            dead.append(term)

    note = (f" (unperturbable output shapes: {', '.join(unperturbable)})"
            if unperturbable else "")
    if dead:
        return ProbeVerdict(
            "UB-7", "flag_for_researcher",
            f"dead term channels in {composite_name}: {', '.join(dead)} — "
            f"output perturbation (cyclic shifts, dup-drop, min-max swap, "
            f"membership displacement) never changes the selection; "
            f"constant or discarded contribution{note}",
            evidence=f"dead={dead}; alive={alive}",
            finding_class="M-004")
    if alive:
        return ProbeVerdict(
            "UB-7", "pass",
            f"all {len(alive)} term channels influence {composite_name}'s "
            f"selection{note}",
            evidence=f"alive={alive}")
    return ProbeVerdict(
        "UB-7", "unprobeable",
        f"no term channel of {composite_name} could be perturbed{note}")


def attribute_invocation_crash(exc: BaseException,
                               package_dir) -> tuple[str, str]:
    """Attribute an exception raised during a kit-conformant invocation.

    Returns (origin, site) where origin is:
    - "subject" — the deepest classifiable traceback frame is inside the
      method package's own directory: the generated code crashed on inputs
      it accepted (behavioral evidence, NOT a harness limitation);
    - "harness" — the deepest classifiable frame is inside scripts/probes/
      (stub or fixture raised mid-call): genuinely unprobeable;
    - "unknown" — no frame falls in either tree (e.g. argument binding
      failed before entering subject code).

    Deepest-frame-wins matters: a broken harness stub invoked FROM subject
    code leaves a subject frame above the stub frame, and only the stub
    frame tells the truth. `site` is "file.py:line" of the deciding frame.
    """
    import traceback  # noqa: PLC0415

    probes_dir = Path(__file__).resolve().parent
    try:
        pkg_dir = Path(package_dir).resolve()
    except (TypeError, ValueError):
        return "unknown", ""
    origin, site = "unknown", ""
    for fr in traceback.extract_tb(exc.__traceback__):  # shallow → deep
        try:
            frame_path = Path(fr.filename).resolve()
        except (TypeError, ValueError):
            continue
        if pkg_dir == frame_path.parent or pkg_dir in frame_path.parents:
            origin, site = "subject", f"{frame_path.name}:{fr.lineno}"
        elif probes_dir == frame_path.parent \
                or probes_dir in frame_path.parents:
            origin, site = "harness", f"{frame_path.name}:{fr.lineno}"
    return origin, site


# Crash signatures that are never an undeclared input-domain assumption —
# within this registry a reproducible subject-internal crash is a definite
# defect (hard fail). Everything else routes flag_for_researcher: the
# harness cannot distinguish broken code from a domain assumption its
# fixture violates (e.g. an image-only selector reshaping to 28x28).
# First member: torch rejecting reversed (negative-stride) numpy index
# arrays — the 2026-06-10 GBALD `np.argsort(...)[::-1]` class.
_CERTAIN_DEFECT_SIGNATURES: tuple[tuple[type, str, str], ...] = (
    # Needle pinned to torch's stride wording: a bare "negative" also
    # matches numpy's "negative dimensions are not allowed" and
    # "probabilities are not non-negative" — pool-size-dependent crashes
    # that belong in the flag tier (2026-06-11 adversarial-review catch).
    (ValueError, "negative stride",
     "reversed (negative-stride) numpy arrays cannot index torch tensors — "
     "make the index contiguous (e.g. `np.argsort(...)[::-1].copy()`) or "
     "sort descending in torch (`torch.topk` / `argsort(descending=True)`)"),
    (NameError, "",
     "an undefined name is reachable on contract-conformant inputs"),
)


def classify_subject_crash(exc: BaseException) -> str | None:
    """Return researcher-facing guidance when the crash signature is in the
    certain-defect registry, else None (uncertain — flag, don't fail).

    Exact-type match, never isinstance: UnboundLocalError subclasses
    NameError, and an unassigned loop variable is input-shape-dependent
    (a zero-iteration loop on the kit's small fixture), which belongs in
    the flag tier (2026-06-11 adversarial-review catch)."""
    for exc_type, needle, guidance in _CERTAIN_DEFECT_SIGNATURES:
        if type(exc) is exc_type and needle.lower() in str(exc).lower():
            return guidance
    return None


def al_selector_kit(
    package: ModuleType,
    pluggable_name: str,
    seed: int = 0,
    fixture_seed: int | None = None,
) -> dict | str:
    """Shared AL-selector harness (UB-7 composite arm + CT-1 floor): the
    separable classification fixture, an MC-dropout stub model (dropout makes
    the stub MC-sampling-live in train mode, so uncertainty channels have
    harness-side signal and can't be false-flagged dead), name-mapped kwargs,
    and a seed-pinned zero-arg invoke. Budget knobs (mc_samples/core_set_size/
    batch_returns) are probe-owned when present, per the UB-5 knob taxonomy.

    Returns the kit dict, or the unprobeable-reason string."""
    try:
        import torch  # noqa: PLC0415
        from torch import nn  # noqa: PLC0415
    except ImportError:
        return "torch required"

    fn = getattr(package, pluggable_name, None)
    if fn is None:
        return f"pluggable {pluggable_name!r} not found in method package"
    module = inspect.getmodule(fn)
    if module is None:
        return f"defining module of {pluggable_name!r} unresolvable"

    fixture = make_classification_fixture(
        scale="zero_one",
        seed=seed if fixture_seed is None else fixture_seed)
    x = torch.tensor(fixture.x_pool, dtype=torch.float32)
    y = torch.tensor(fixture.y_pool, dtype=torch.long)
    n_labeled = 3 * fixture.n_classes

    # Manifest-conformant stub: plain forward (the GBALD interface) PLUS the
    # penultimate-embedding hook (BADGE's spec declares
    # `forward_with_embedding` as the pluggable's model interface — a bare
    # nn.Sequential left every selector probe unprobeable on the 2026-06-10
    # matrix-row-3 audit). Dropout keeps MC-sampling channels live in train
    # mode so uncertainty terms can't be false-flagged dead.
    class _StubNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.body = nn.Sequential(
                nn.Linear(x.shape[1], 16), nn.ReLU(), nn.Dropout(0.5))
            self.head = nn.Linear(16, fixture.n_classes)
            # Introspection attributes generated selectors read off the
            # model object — each fresh generation surfaces another one
            # (2026-06-10: BADGE read forward_with_embedding; the GBALD
            # validation re-run read model.n_classes). Carry the manifest
            # conventions so the harness never false-unprobeables on them.
            self.n_classes = fixture.n_classes
            self.num_classes = fixture.n_classes
            self.input_dim = int(x.shape[1])
            self.n_features = int(x.shape[1])
            self.hidden_dim = 16

        def forward(self, batch):
            return self.head(self.body(batch))

        def forward_with_embedding(self, batch):
            hidden = self.body(batch)
            return self.head(hidden), hidden

    def make_model(model_seed: int):
        torch.manual_seed(model_seed)
        return _StubNet()

    def make_trained_model(model_seed: int):
        """A stub trained on the fixture's labeled split.

        An UNTRAINED stub yields ~uniform logits, so an uncertainty score
        (BALD / predictive entropy) collapses to ~0 and a faithful
        uncertainty-based selector looks model-insensitive purely for lack of
        signal. Training gives the model confident, weight-dependent
        predictions, so a sensitivity test exercises the selector rather than
        the stub's degeneracy. The labeled split is separable by construction,
        so a short full-batch budget fits it."""
        m = make_model(model_seed)
        torch.manual_seed(model_seed)
        opt = torch.optim.Adam(m.parameters(), lr=1e-2)
        loss_fn = nn.CrossEntropyLoss()
        x_lab, y_lab = x[:n_labeled], y[:n_labeled]
        m.train()
        for _ in range(200):
            opt.zero_grad()
            loss_fn(m(x_lab), y_lab).backward()
            opt.step()
        return m

    stub_model = make_model(seed)

    batch_size = 4
    kwargs = _fill_kwargs(fn, {
        "model": stub_model,
        "x_unlabeled": x[n_labeled:], "x_pool": x[n_labeled:],
        "x_labeled": x[:n_labeled], "y_labeled": y[:n_labeled],
        "batch_size": batch_size, "seed": seed,
        "mc_samples": 8, "core_set_size": 6, "batch_returns": 16,
    })
    if isinstance(kwargs, str):
        return f"{pluggable_name} requires unmapped parameter {kwargs!r}"

    def invoke():
        torch.manual_seed(seed)
        np.random.seed(seed)
        return fn(**kwargs)

    module_file = getattr(module, "__file__", None)
    return {
        "fn": fn, "module": module, "invoke": invoke, "kwargs": kwargs,
        "batch_size": batch_size,
        "pool_size": int(x.shape[0]) - n_labeled, "seed": seed,
        "n_features": int(x.shape[1]), "n_classes": fixture.n_classes,
        "make_model": make_model,
        "make_trained_model": make_trained_model,
        "package_dir": (Path(module_file).parent if module_file else None),
    }


def probe_al_selector_terms(
    package: ModuleType,
    pluggable_name: str,
    seed: int = 0,
) -> ProbeVerdict:
    """Battery entry: UB-7 perturbation arm over an AL selector pluggable."""
    kit = al_selector_kit(package, pluggable_name, seed)
    if isinstance(kit, str):
        return ProbeVerdict("UB-7", "unprobeable", kit)
    return probe_composite_term_perturbation(
        kit["module"], pluggable_name, kit["invoke"],
        index_domain=kit["pool_size"])
