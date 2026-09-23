"""Arm A of the numeric-coherence pass (queue 11d) — training-budget sufficiency.

The GBALD `verified` blocker. The parameter-deriver picks `max_epochs` and the
batch size without reference to the architecture-coder's convergence behavior,
so a 0.5-dropout MLP can get ~16 gradient updates per round and never leave
chance — a degenerate notebook with no crash. Nothing reconciled the two
stages before the notebook ran.

This check runs at Stage 2.x, right after derivation, where both the
fully-built importable package and the live budget exist (params.json is
derived at 2.x, after the 2.d package finalization). It builds a separable
classification fixture sized to the round-one labeled set, runs the LIVE
`build_model` + `train_from_scratch` with the LIVE budget AND the LIVE dropout,
and asks whether the configuration can fit the fixture to clearly above chance.

This is the deliberate complement to UB-5 (`probes.trainability`): UB-5 OWNS the
budget knobs (it overrides max_epochs and uses a small batch) so it can test the
learning rate in isolation, and it CLAMPS dropout. Arm A keeps the live budget
and the live dropout so it can test sufficiency. Same fixture and package
machinery, opposite knob policy, different question.

Fidelity is empirical (2026-06-30): at a 64-feature fixture with dropout 0.5 and
~110 labeled samples, the GBALD live budget (max_epochs=8, batch=100) sits at
chance across seeds while raising max_epochs to ~50 fits it. The auto-raise
lever is `max_epochs` ALONE — never `batch_size`, which is the load-bearing
acquisition batch, not a training knob.

When the live budget is insufficient the assessment names the smallest
`max_epochs` on a bounded grid that fits (the caller raises params + logs an
assumption). When nothing on the grid fits, the problem is not just the budget
(learning rate or architecture), and the assessment says so (the caller halts).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))


# Fixed moderate fixture dimensionality: high enough that a starved budget
# cannot fit it (12-dim is too easy and lets marginal budgets pass), low enough
# that a healthy budget can (784-dim synthetic is too hard even when healthy).
_FIXTURE_FEATURES = 64
_FIXTURE_CLASSES = 3
_SEED = 0
# Bounded escalation grid for the auto-raise. Verified: 50 epochs fits the
# 64-dim/dropout-0.5/~110-sample fixture across seeds. 400 is the ceiling above
# which "still cannot fit" means the lr/architecture is the problem, not budget.
_EPOCH_GRID = (50, 100, 200, 400)
_BATCH_FALLBACK = 32


def labeled_set_size(params: dict) -> int:
    """Best estimate of the round-one labeled-set size from params.

    GBALD bootstraps with a small random seed (~10) plus a core-set, so the
    round-one labeled set is `core_set_size` + a seed count. Pool-bootstrap
    methods use `initial_labeled`. Clamped to keep the fixture small and fast.
    """
    def _val(name):
        e = params.get(name)
        if isinstance(e, dict):
            v = e.get("value")
            return v if isinstance(v, (int, float)) and not isinstance(v, bool) else None
        return None

    core = _val("core_set_size")
    init = _val("initial_labeled")
    if core is not None:
        n = int(core) + 10  # bootstrap seed the notebook adds before the core-set
    elif init is not None:
        n = int(init)
    else:
        n = 100
    return max(30, min(300, n))


def _fixture_fit_acc(method, *, n_classes, n_labeled, dropout, max_epochs, batch,
                     lr) -> tuple[float, float] | None:
    """Build the fixture, run live build_model + train_from_scratch at the given
    budget, return (train_acc, chance). None when the package does not expose the
    active-learning training shape (build_model + train_from_scratch)."""
    import numpy as np  # noqa: PLC0415
    import torch  # noqa: PLC0415

    from probes.fixtures import make_classification_fixture  # noqa: PLC0415
    from probes.trainability import _fill_kwargs  # noqa: PLC0415

    build_model = getattr(method, "build_model", None)
    train = getattr(method, "train_from_scratch", None)
    if build_model is None or train is None:
        return None

    fx = make_classification_fixture(
        n_classes=n_classes, n_features=_FIXTURE_FEATURES,
        n_per_class_pool=max(1, n_labeled // n_classes), scale="zero_one",
        seed=_SEED)
    x = torch.tensor(fx.x_pool, dtype=torch.float32)
    y = torch.tensor(fx.y_pool, dtype=torch.long)
    torch.manual_seed(_SEED)
    np.random.seed(_SEED)

    build_kwargs = _fill_kwargs(build_model, {
        "input_dim": x.shape[1], "n_classes": n_classes,
        "num_classes": n_classes, "dropout_rate": dropout, "seed": _SEED,
    })
    if isinstance(build_kwargs, str):
        return None
    model = build_model(**build_kwargs)

    train_kwargs = _fill_kwargs(train, {
        "model": model, "x_train": x, "y_train": y, "x": x, "y": y,
        "learning_rate": lr, "lr": lr,
        "max_epochs": int(max_epochs),
        # Honor the live batch. If the training fn takes no batch arg it
        # full-batches; the fallback only applies when a batch arg exists but
        # the live value is unknown.
        "batch_train_size": int(batch), "batch_size": int(batch), "batch": int(batch),
        "train_until_accuracy": 0.99, "seed": _SEED,
    })
    if isinstance(train_kwargs, str):
        return None
    model = train(**train_kwargs)
    model.eval()
    with torch.no_grad():
        acc = float((model(x).argmax(dim=1) == y).float().mean())
    return acc, fx.chance


def assess_budget(run_dir: Path) -> dict:
    """Assess whether the live training budget can fit a labeled-set-sized
    fixture. Returns a JSON-serializable assessment (see module docstring)."""
    run_dir = Path(run_dir)
    from probes.trainability import CHANCE_MARGIN, flat_live_params  # noqa: PLC0415

    def na(reason: str) -> dict:
        return {"applicable": False, "reason": reason}

    try:
        import torch  # noqa: F401,PLC0415
    except ImportError:
        return na("torch unavailable")

    if not (run_dir / "method").is_dir():
        return na("no method/ package in run dir")

    params_path = run_dir / ".pipeline" / "params.json"
    if not params_path.is_file():
        return na("no params.json")
    try:
        doc = json.loads(params_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        return na(f"params.json invalid JSON: {e}")
    params = doc.get("params", doc)
    live = flat_live_params(params)

    live_epochs = int(live.get("max_epochs", 0) or 0)
    if live_epochs <= 0:
        return na("no usable live max_epochs in params")
    live_batch = int(live.get("batch_size") or live.get("batch_train_size") or _BATCH_FALLBACK)
    live_lr = float(live.get("learning_rate", 1e-3) or 1e-3)
    dropout = float(live.get("dropout_rate", 0.5) or 0.5)
    n_labeled = labeled_set_size(params)

    from probes.package_loader import ProbeLoadError, imported_method_package  # noqa: PLC0415

    def fit(max_epochs: int) -> tuple[float, float] | None:
        try:
            with imported_method_package(run_dir) as method:
                return _fixture_fit_acc(
                    method, n_classes=_FIXTURE_CLASSES, n_labeled=n_labeled,
                    dropout=dropout, max_epochs=max_epochs, batch=live_batch,
                    lr=live_lr)
        except ProbeLoadError as e:
            raise RuntimeError(f"package import failed: {e}") from e

    try:
        live_result = fit(live_epochs)
    except RuntimeError as e:
        return na(str(e))
    if live_result is None:
        return na("package does not expose the build_model + train_from_scratch "
                  "training shape (non-AL-training paradigm or non-manifest names)")

    live_acc, chance = live_result
    target = chance + CHANCE_MARGIN
    base = {
        "applicable": True,
        "live_max_epochs": live_epochs,
        "live_batch": live_batch,
        "live_dropout": dropout,
        "n_labeled_fixture": n_labeled,
        "live_acc": round(live_acc, 4),
        "chance": round(chance, 4),
        "target": round(target, 4),
    }
    if live_acc >= target:
        return {**base, "sufficient_at_live": True, "suggested_max_epochs": None,
                "reason": (f"live budget (max_epochs={live_epochs}, batch={live_batch}) "
                           f"fits the fixture to {live_acc:.3f} >= {target:.3f}")}

    # Live budget starves. Search the grid (max_epochs only) for the smallest
    # value that fits, keeping batch and lr fixed.
    for cand in _EPOCH_GRID:
        if cand <= live_epochs:
            continue
        result = fit(cand)
        if result is None:
            break
        acc, _ = result
        if acc >= target:
            return {**base, "sufficient_at_live": False, "suggested_max_epochs": cand,
                    "reason": (f"live budget starves (acc {live_acc:.3f} < {target:.3f}); "
                               f"max_epochs={cand} fits to {acc:.3f} with batch and lr "
                               f"unchanged")}

    return {**base, "sufficient_at_live": False, "suggested_max_epochs": None,
            "reason": (f"the live config cannot fit a separable {n_labeled}-sample fixture "
                       f"even at max_epochs={_EPOCH_GRID[-1]} (acc stays below {target:.3f}). "
                       f"This is not just a budget shortfall — the learning rate "
                       f"({live_lr}) or the architecture (dropout {dropout}) needs attention.")}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=None,
                        help="where to write the assessment JSON "
                             "(default <run-dir>/.pipeline/budget_assessment.json)")
    args = parser.parse_args()

    assessment = assess_budget(args.run_dir)
    out = args.out or (args.run_dir / ".pipeline" / "budget_assessment.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(assessment, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(assessment, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
