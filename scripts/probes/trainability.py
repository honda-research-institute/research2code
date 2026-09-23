"""UB-5 `trains-on-synthetic`: the live training config must actually learn.

The lr=7.4 class, caught behaviorally: build the package's own model with the
package's own training function using the RUN'S LIVE params.json values, on a
guaranteed-separable synthetic fixture, and require final TRAIN accuracy to
beat chance by a clear margin. Any pathological optimizer config (leaked
learning rates, frozen weights, sign errors) fails in seconds — no notebook
execution, no real dataset.

Train accuracy (not test) is deliberate: this is a capacity/optimization
check, not a generalization claim. The fixture is separable by construction
and pinned by the harness self-calibration tests, so "can't fit the training
set" always means "training is broken", never "task too hard".

Conventions: build_model/train_from_scratch are the AL package-manifest names;
signature filling is name-mapped (v3's _invoke_with_kwargs pattern), with
unmapped REQUIRED parameters making the package `unprobeable`, never guessed.
"""

from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np

from probes import ProbeVerdict
from probes.catalogs.trainability import PROBE_CATALOG as _PROBE_CATALOG
from probes.fixtures import ClassificationFixture, make_classification_fixture
from probes.package_loader import ProbeLoadError, imported_method_package

PROBE_CATALOG = _PROBE_CATALOG
CHANCE_MARGIN = 0.25


def _fill_kwargs(fn, candidates: dict) -> dict | str:
    """Map known values onto fn's signature; return the kwargs, or the name
    of a required parameter we can't supply."""
    sig = inspect.signature(fn)
    kwargs = {}
    for name, param in sig.parameters.items():
        if name in candidates:
            kwargs[name] = candidates[name]
        elif param.default is inspect.Parameter.empty and param.kind in (
            param.POSITIONAL_OR_KEYWORD, param.KEYWORD_ONLY,
        ):
            return name
    return kwargs


def probe_trains_on_synthetic(
    run_dir: Path,
    live_params: dict | None = None,
    fixture: ClassificationFixture | None = None,
    epochs_cap: int = 100,
    seed: int = 0,
) -> ProbeVerdict:
    """Run the package's build_model + train_from_scratch with live params."""
    try:
        import torch  # noqa: PLC0415
    except ImportError:
        return ProbeVerdict("UB-5", "unprobeable", "torch required")

    live = dict(live_params or {})
    fixture = fixture or make_classification_fixture(
        n_classes=3, n_features=12, n_per_class_pool=40, scale="zero_one",
        seed=seed)
    # Model init draws from torch's GLOBAL rng (build_model signatures rarely
    # take a seed): without this the probe's verdict varied per process.
    torch.manual_seed(seed)
    np.random.seed(seed)

    try:
        with imported_method_package(Path(run_dir)) as method:
            build_model = getattr(method, "build_model", None)
            train = getattr(method, "train_from_scratch", None)
            if build_model is None or train is None:
                return ProbeVerdict(
                    "UB-5", "unprobeable",
                    "package exposes no build_model/train_from_scratch "
                    "(non-trainable paradigm or non-manifest names)")

            x = torch.tensor(fixture.x_pool, dtype=torch.float32)
            y = torch.tensor(fixture.y_pool, dtype=torch.long)

            build_kwargs = _fill_kwargs(build_model, {
                "input_dim": fixture.x_pool.shape[1],
                "n_classes": fixture.n_classes,
                "num_classes": fixture.n_classes,
                "hidden_dim": int(min(live.get("hidden_dim", 32), 64)),
                # Dropout is CLAMPED, never honored as-is: heavy live dropout
                # (GBALD's 0.5) legitimately needs many more epochs on a tiny
                # set — v3's expertise pack: "dropout 0.5 at n<=200 prevents
                # learning". UB-5 probes the OPTIMIZER config; learning_rate
                # is the subject and is never clamped.
                "dropout_rate": float(min(live.get("dropout_rate", 0.1), 0.2)),
                "seed": seed,
            })
            if isinstance(build_kwargs, str):
                return ProbeVerdict(
                    "UB-5", "unprobeable",
                    f"build_model requires unmapped parameter "
                    f"{build_kwargs!r}")
            model = build_model(**build_kwargs)

            lr = float(live.get("learning_rate", 1e-3))
            # Batch size is a BUDGET knob, same class as max_epochs, so it is
            # probe-owned and set small. A training fn names its batch arg one
            # of several ways (train_from_scratch took `batch_train_size`,
            # other packages use `batch_size`/`batch`); supplying all the
            # aliases is what lets _fill_kwargs bind ANY of them rather than
            # returning unprobeable. Without this the GBALD run's required
            # `batch_train_size` made UB-5 unprobeable — the one check that
            # would have flagged its training starvation never ran. A small
            # batch on the separable toy fixture maximizes gradient steps so
            # the only thing that can fail UB-5 is the SUBJECT knob (lr).
            small_batch = int(min(32, len(fixture.y_pool)))
            train_kwargs = _fill_kwargs(train, {
                "model": model, "x_train": x, "y_train": y,
                "x": x, "y": y,
                "learning_rate": lr, "lr": lr,
                # Knob taxonomy (calibrated on the real june9 package):
                # SUBJECT knobs are honored as-is (learning_rate — the thing
                # under test). CAPACITY knobs are clamped (dropout, hidden).
                # BUDGET knobs are probe-owned (max_epochs, batch size): live
                # smoke values like max_epochs=8 / batch=100 are wall-clock
                # economics, and 8 near-full-batch steps cannot fit even a
                # separable toy set — a guaranteed false positive if honored.
                "max_epochs": int(epochs_cap),
                "batch_train_size": small_batch,
                "batch_size": small_batch,
                "batch": small_batch,
                "train_until_accuracy": float(
                    min(live.get("train_until_accuracy", 0.99), 0.99)),
                "seed": seed,
            })
            if isinstance(train_kwargs, str):
                return ProbeVerdict(
                    "UB-5", "unprobeable",
                    f"train_from_scratch requires unmapped parameter "
                    f"{train_kwargs!r}")
            model = train(**train_kwargs)

            model.eval()
            with torch.no_grad():
                acc = float((model(x).argmax(dim=1) == y).float().mean())
    except ProbeLoadError as e:
        return ProbeVerdict("UB-5", "unprobeable", str(e))
    except Exception as e:  # noqa: BLE001 — generated code fails arbitrarily
        return ProbeVerdict(
            "UB-5", "unprobeable",
            f"training raised {type(e).__name__}: {e}")

    threshold = fixture.chance + CHANCE_MARGIN
    if not np.isfinite(acc) or acc < threshold:
        return ProbeVerdict(
            "UB-5", "fail",
            f"live training config cannot fit a separable synthetic set: "
            f"train accuracy {acc:.3f} < chance+margin {threshold:.2f} "
            f"(learning_rate={lr!r}) — the configuration does not learn",
            evidence=f"acc={acc:.3f} on {len(fixture.y_pool)} samples")
    return ProbeVerdict(
        "UB-5", "pass",
        f"live config trains to {acc:.3f} train accuracy "
        f"(chance {fixture.chance:.2f}, lr={lr!r})")


def flat_live_params(params_json: dict) -> dict:
    """{name: value} view of a params.json mapping (entries or raw values)."""
    flat = {}
    for name, entry in params_json.items():
        flat[name] = entry.get("value") if isinstance(entry, dict) else entry
    return {k: v for k, v in flat.items() if v is not None}
