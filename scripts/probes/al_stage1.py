"""AL Stage-1 probes — the core-set construction family (queue item 9).

The GBALD-class active-learning composition has a one-time Stage-1 step:
construct a core-set that seeds the labeled pool before the acquisition
loop runs. The verified GBALD delivery executes it in the demo and, until
this kit, NOTHING verified it — the acquisition-loop probes (AL-1/4/5/6)
and the UB-7 term perturbations only see `select_batch`'s call graph, and
Stage 1 is invoked from the notebook outside it.

Binding is by the family's naming convention (a module-level callable whose
name contains ``core_set``/``coreset``), mirrored from six independent GBALD
rolls that all produced ``construct_core_set``. No convention match, an
ambiguous match, or an unbindable signature returns a reason string —
per-probe unprobeable, never a fabricated stand-in.

Three refutation-only probes over a probe-owned synthetic pool (G
well-separated clusters, rows SORTED by cluster so position-degeneracy is
geometrically visible):

- **AL-S1-1 output contract** (fail): returned indices must be unique,
  in-range integers, non-empty. The duplicate/corrupt-index class.
- **AL-S1-2 selection non-degeneracy** (fail, M-004): greedy picks must not
  concentrate in the leading cluster block of the sorted pool. The frozen
  zoo defect (gbald-lr74): with R_0 huge relative to the data, the
  geometric prior is constant, ``argmax`` returns first-position ties, and
  the "selection" carries no information. Seeding picks the kit itself
  supplied (a ``kmeans_centers``-style parameter) are excluded from the
  measurement so a covered seeding cannot mask a dead greedy criterion.
- **AL-S1-3 ellipsoid-channel liveness** (flag_for_researcher, M-004): the
  paper's core novelty is the ellipsoid geodesic rescale, controlled by
  ``eta``. Two invocations identical except for eta must differ somewhere;
  identical selections mean the claimed mechanism is inert. Same seed both
  runs, so rng cannot confound the comparison. No eta parameter exposed →
  unprobeable, stated.
"""

from __future__ import annotations

import inspect

from probes import ProbeVerdict
from probes.catalogs.al_stage1 import PROBE_CATALOG as _PROBE_CATALOG
from probes.trainability import _fill_kwargs

PROBE_CATALOG = _PROBE_CATALOG
_NAME_TOKENS = ("core_set", "coreset")

# Synthetic pool geometry (probe-owned; generic, no paper values).
_N_CLUSTERS = 6
_PER_CLUSTER = 30
_N_FEATURES = 4
_CLUSTER_SPACING = 10.0
_CLUSTER_STD = 0.5
_CORE_SET_SIZE = 12


def _owned_by(owner: str | None, package_name: str) -> bool:
    """Ownership = the probed module itself or any of its submodules.

    The production runner hands the imported PACKAGE (named `method`) while
    every function lives in `method.method` and is re-exported through
    `__init__` — exact equality rejected all of them, so the kit never bound
    on any production run (RCA 2026-08-03, finding 7). Prefix match keeps
    genuinely foreign callables (numpy, torch, another local package) out."""
    return owner is not None and (
        owner == package_name or owner.startswith(package_name + "."))


def _find_coreset_fn(module):
    """The family naming convention, package-owned callables only.
    Returns (fn, name) or a reason string."""
    matches = []
    rejected = []
    for name in dir(module):
        if name.startswith("_"):
            continue
        if not any(tok in name.lower() for tok in _NAME_TOKENS):
            continue
        obj = getattr(module, name)
        if not callable(obj):
            continue
        owner = getattr(obj, "__module__", None)
        if _owned_by(owner, module.__name__):
            matches.append((obj, name))
        else:
            rejected.append((name, owner))
    if not matches:
        if rejected:
            named = "; ".join(
                f"found `{n}` but rejected: owned by module "
                f"`{owner or '<unknown>'}`, not `{module.__name__}` or its "
                f"submodules" for n, owner in sorted(rejected))
            return (f"no package-owned callable matches the core-set naming "
                    f"convention ({named}) — the Stage-1 probes cannot bind")
        return ("no module-level callable matches the core-set naming "
                "convention (name containing 'core_set'/'coreset') — the "
                "Stage-1 probes cannot bind")
    if len(matches) > 1:
        exact = [m for m in matches if m[1] == "construct_core_set"]
        if len(exact) == 1:
            return exact[0]
        return ("ambiguous core-set convention match: "
                + ", ".join(sorted(n for _, n in matches)))
    return matches[0]


def _wants_numpy(param: inspect.Parameter) -> bool:
    """True when the parameter's annotation names a numpy array type.

    Generated code carries `from __future__ import annotations`, so the
    annotation is usually the string "np.ndarray"; evaluated annotations
    stringify to "<class 'numpy.ndarray'>". Unannotated parameters keep the
    kit's torch default."""
    ann = param.annotation
    if ann is inspect.Parameter.empty:
        return False
    text = ann if isinstance(ann, str) else str(ann)
    return "ndarray" in text or "numpy" in text or text.startswith("np.")


def _sorted_cluster_pool(torch, seed: int):
    """(pool, cluster_ids, centers): rows sorted by cluster, well separated."""
    g = torch.Generator().manual_seed(seed)
    rows, ids, centers = [], [], []
    for c in range(_N_CLUSTERS):
        center = torch.zeros(_N_FEATURES)
        center[c % _N_FEATURES] = _CLUSTER_SPACING * (1 + c // _N_FEATURES)
        centers.append(center)
        rows.append(center + _CLUSTER_STD * torch.randn(
            (_PER_CLUSTER, _N_FEATURES), generator=g))
        ids.extend([c] * _PER_CLUSTER)
    return torch.cat(rows), ids, torch.stack(centers)


def al_stage1_kit(module, *, seed: int = 1729):
    """Bind the Stage-1 core-set constructor. Returns a dict kit (with a
    seed-pinned ``invoke(**overrides)``) or a reason string."""
    try:
        import torch  # noqa: PLC0415
    except ImportError:
        return "torch required for the Stage-1 core-set probes"
    found = _find_coreset_fn(module)
    if isinstance(found, str):
        return found
    fn, fn_name = found
    pool, cluster_ids, centers = _sorted_cluster_pool(torch, seed)
    # Nearest pool row per center — supplied when the signature asks for
    # precomputed centers, and excluded from the degeneracy measurement.
    d = ((pool[:, None, :] - centers[None, :, :]) ** 2).sum(-1)
    seeded_idx = sorted(set(int(i) for i in d.argmin(dim=0)))
    candidates = {
        "x_pool": pool, "x_unlabeled": pool, "pool": pool, "x": pool,
        "data": pool,
        "x_initial": pool[seeded_idx[:2]], "x_labeled": pool[seeded_idx[:2]],
        "kmeans_centers": centers,
        "core_set_size": _CORE_SET_SIZE, "n_clusters": _N_CLUSTERS,
        "k": _N_CLUSTERS, "n_samples": _CORE_SET_SIZE,
        "eta": 0.9, "R_0": float(_CLUSTER_SPACING), "seed": seed,
    }
    kwargs = _fill_kwargs(fn, candidates)
    if isinstance(kwargs, str):
        return (f"core-set constructor `{fn_name}` has a required parameter "
                f"the probes cannot supply: `{kwargs}`")
    signature = inspect.signature(fn)
    sig_params = set(signature.parameters)
    # Signature-aware pool typing: a numpy-annotated parameter gets a numpy
    # array, not a torch tensor. Without this the first real bind reports a
    # spurious crash-class fail instead of the truth (the delivered bayesian
    # constructor is `x_unlabeled: np.ndarray`; RCA 2026-08-03, finding 7).
    for name, value in list(kwargs.items()):
        if isinstance(value, torch.Tensor) and _wants_numpy(
                signature.parameters[name]):
            kwargs[name] = value.numpy()

    def invoke(**overrides):
        kw = dict(kwargs)
        kw.update({k: v for k, v in overrides.items() if k in sig_params})
        return fn(**kw)

    return {
        "fn": fn, "fn_name": fn_name, "invoke": invoke, "kwargs": kwargs,
        "pool": pool, "cluster_ids": cluster_ids,
        "pool_size": int(pool.shape[0]),
        "seeded_idx": seeded_idx if "kmeans_centers" in sig_params else [],
        "core_set_size": _CORE_SET_SIZE, "seed": seed,
        "has_eta": "eta" in sig_params,
    }


def _as_index_list(result, pool_size: int):
    """Normalize a selection to a list of ints, or a reason string."""
    try:
        idx = [int(i) for i in list(result)]
    except (TypeError, ValueError):
        return f"selection is not an index sequence: {type(result).__name__}"
    if not idx:
        return "selection is empty"
    bad = [i for i in idx if i < 0 or i >= pool_size]
    if bad:
        return f"out-of-range indices (pool_size={pool_size}): {bad[:5]}"
    return idx


def probe_coreset_output_contract(kit) -> ProbeVerdict:
    """AL-S1-1: unique in-range indices, non-empty."""
    pid = "AL-S1-1"
    if isinstance(kit, str):
        return ProbeVerdict(pid, "unprobeable", kit)
    try:
        result = kit["invoke"]()
    except Exception as e:  # noqa: BLE001 — a crash is a finding, not a probe bug
        return ProbeVerdict(
            pid, "fail",
            f"core-set constructor `{kit['fn_name']}` raised on a synthetic "
            f"pool: {type(e).__name__}: {e}",
            finding_class="M-002")
    idx = _as_index_list(result, kit["pool_size"])
    if isinstance(idx, str):
        return ProbeVerdict(pid, "fail", idx, finding_class="M-002")
    if len(set(idx)) != len(idx):
        dupes = sorted({i for i in idx if idx.count(i) > 1})
        return ProbeVerdict(
            pid, "fail",
            f"duplicate indices in the core-set (snap-to-unused violated): "
            f"{dupes[:5]}",
            finding_class="M-002")
    return ProbeVerdict(
        pid, "pass",
        f"{len(idx)} unique in-range indices from `{kit['fn_name']}`")


def probe_coreset_selection_nondegeneracy(kit) -> ProbeVerdict:
    """AL-S1-2: on a cluster-sorted pool, non-seeding picks must not
    concentrate in the leading cluster block (the constant-score argmax
    signature)."""
    pid = "AL-S1-2"
    if isinstance(kit, str):
        return ProbeVerdict(pid, "unprobeable", kit)
    try:
        result = kit["invoke"]()
    except Exception as e:  # noqa: BLE001
        return ProbeVerdict(
            pid, "unprobeable",
            f"constructor raised (adjudicated by AL-S1-1): "
            f"{type(e).__name__}")
    idx = _as_index_list(result, kit["pool_size"])
    if isinstance(idx, str):
        return ProbeVerdict(
            pid, "unprobeable",
            f"selection unreadable (adjudicated by AL-S1-1): {idx}")
    seeded = set(kit["seeded_idx"])
    picks = [i for i in dict.fromkeys(idx) if i not in seeded]
    if len(picks) < 3:
        return ProbeVerdict(
            pid, "unprobeable",
            f"only {len(picks)} non-seeding picks — too few to judge "
            f"concentration")
    clusters = {kit["cluster_ids"][i] for i in picks}
    leading = sum(1 for i in picks if kit["cluster_ids"][i] == 0)
    frac = leading / len(picks)
    if len(clusters) == 1 or frac >= 0.8:
        return ProbeVerdict(
            pid, "fail",
            f"selection is position-degenerate: {leading}/{len(picks)} "
            f"non-seeding picks fall in the leading cluster block of a "
            f"{_N_CLUSTERS}-cluster pool sorted by cluster — the selection "
            f"criterion carries no data signal (constant-score argmax "
            f"class)",
            finding_class="M-004")
    return ProbeVerdict(
        pid, "pass",
        f"picks span {len(clusters)} of {_N_CLUSTERS} clusters "
        f"(leading-block fraction {frac:.2f})")


def probe_coreset_ellipsoid_channel(kit) -> ProbeVerdict:
    """AL-S1-3: selections at eta=0.15 vs eta=0.9 (same seed, same pool)
    must differ somewhere, or the paper's ellipsoid mechanism is inert."""
    pid = "AL-S1-3"
    if isinstance(kit, str):
        return ProbeVerdict(pid, "unprobeable", kit)
    if not kit["has_eta"]:
        return ProbeVerdict(
            pid, "unprobeable",
            f"`{kit['fn_name']}` exposes no `eta` parameter — the ellipsoid "
            f"rescale channel cannot be exercised from the interface")
    try:
        low = kit["invoke"](eta=0.15)
        high = kit["invoke"](eta=0.9)
    except Exception as e:  # noqa: BLE001
        return ProbeVerdict(
            pid, "unprobeable",
            f"constructor raised under an eta override: {type(e).__name__}")
    low_i = _as_index_list(low, kit["pool_size"])
    high_i = _as_index_list(high, kit["pool_size"])
    if isinstance(low_i, str) or isinstance(high_i, str):
        return ProbeVerdict(pid, "unprobeable",
                            "selection unreadable under an eta override")
    if low_i == high_i:
        return ProbeVerdict(
            pid, "flag_for_researcher",
            "the selection is byte-identical at eta=0.15 and eta=0.9 (same "
            "seed, same pool) — the ellipsoid geodesic rescale, the paper's "
            "central Stage-1 mechanism, appears inert in this "
            "implementation",
            finding_class="M-004")
    diff = sum(1 for a, b in zip(low_i, high_i) if a != b) \
        + abs(len(low_i) - len(high_i))
    return ProbeVerdict(
        pid, "pass",
        f"eta moves the selection ({diff} position(s) differ between "
        f"eta=0.15 and eta=0.9)")


def run_al_stage1_probes(module, *, seed: int = 1729) -> list[ProbeVerdict]:
    """The battery entry point: bind once, run all three probes."""
    kit = al_stage1_kit(module, seed=seed)
    return [
        probe_coreset_output_contract(kit),
        probe_coreset_selection_nondegeneracy(kit),
        probe_coreset_ellipsoid_channel(kit),
    ]
