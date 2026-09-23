"""Deterministic scale calibration at Stage 2.x (hardening item 1, from the
2026-07-02 GBALD acceptance run).

The spec can declare a hyperparameter scale-dependent
(`critical_requirements.scale_dependent_hyperparameters`: paper_value +
assumes_data_scale). Before this step, nothing deterministic reconciled that
declaration with the data scale the delivered `data.py` actually produces:
the 2026-06-30 run was saved by a reviewer finding that happened to carry a
rescale resolution, and the 2026-07-02 run shipped R_0=2000 onto
[0,1]-normalized MNIST because that roll's reviewer instead CERTIFIED
preservation ("R_0 preserved at paper values" is one of its pass criteria).
The review layer is structurally biased toward preservation, so it cannot be
the safety net. The smoke-diagnostician eventually self-corrected (A002 =
2000/255), but two failed smoke executions late.

This step runs at Stage 2.x, right after derivation and validation, where
both params.json and the fully-built importable package exist (the same
placement argument as Arm A / budget_sufficiency.py). It measures the scale
of the data the package's own `load_data` returns (one small call; the cache
it writes is the same one the smoke run reuses later) and, per declared
entry:

- measured scale == assumed scale       -> at calibration, leave the value;
- raw-PIXEL calibration on [0,1] data   -> deterministic rescale value/255
                                           (the caller edits params + logs an
                                           assumption — the A002 pattern);
- value already differs from the paper  -> already rescaled upstream, no-op;
- any other scale pair                  -> needs_attention (no deterministic
                                           conversion is defined; the US-4
                                           static probe at stage 5 stays the
                                           backstop).

The 255 factor is applied ONLY when the assumes string names pixels (or 255
itself): raw non-pixel units (e.g. meters) have no universal factor, and
guessing one would be a fabrication.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from probes.calibration_context import (  # noqa: E402
    FEATURE_MAGNITUDE_OBSERVER,
    TARGET_BOX_GRID_OBSERVER,
    dispatch_calibration_context,
)

_SAMPLE_ROWS = 64
_LOAD_CANDIDATES = {
    "pool_size": 256, "n_test": 64, "n_samples": 256, "seed": 0,
}


def _feature_array(loaded):
    """Pull the pool/train feature array out of load_data's return value.
    Handles the two delivered conventions: a dict with x_* keys, or a
    tuple/list whose first element is the pool."""
    if isinstance(loaded, dict):
        for key in ("x_pool", "x_train", "x", "student_inputs_train"):
            if key in loaded:
                return loaded[key]
        return None
    if isinstance(loaded, (tuple, list)) and loaded:
        return loaded[0]
    return loaded


def _measure_scale(x) -> tuple[str | None, str]:
    """Classify a feature sample into the canonical scale categories."""
    from probes.scale_mismatch import classify_data_scale  # noqa: PLC0415

    import numpy as np  # noqa: PLC0415

    detach = getattr(x, "detach", None)
    if callable(detach):
        x = detach().cpu().numpy()
    try:
        arr = np.asarray(x, dtype=float)
    except (TypeError, ValueError) as e:
        return None, f"load_data output not array-like: {e}"
    flat = arr.reshape(arr.shape[0], -1)[:_SAMPLE_ROWS].ravel()
    if flat.size == 0:
        return None, "empty feature sample from load_data"
    min_val = float(flat.min())
    max_abs = float(abs(flat).max())
    frac_neg = float((flat < -1e-9).mean())
    scale = classify_data_scale(min_val, max_abs, frac_neg)
    detail = (f"load_data sample: min={min_val:.3g}, max|·|={max_abs:.3g}, "
              f"neg_frac={frac_neg:.2f}")
    if scale is None:
        return None, f"sample did not match a known scale ({detail})"
    return scale, detail


def _numeric_box_values(value: object) -> list[float] | None:
    """Flatten one declared boxes carrier without guessing tuple positions."""
    detach = getattr(value, "detach", None)
    if callable(detach):
        value = detach().cpu()
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        value = tolist()
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return [float(value)]
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return None
    result: list[float] = []
    for item in value:
        values = _numeric_box_values(item)
        if values is None:
            return None
        result.extend(values)
    return result


def _target_box_coordinates(loaded: object) -> list[float] | None:
    """Read only named target-bearing ``load_data`` output surfaces."""
    if not isinstance(loaded, Mapping):
        return None
    coordinates: list[float] = []
    for key in ("targets_train", "targets", "targets_test"):
        targets = loaded.get(key)
        if isinstance(targets, Mapping):
            samples: Sequence[object] = [targets]
        elif isinstance(targets, Sequence) \
                and not isinstance(targets, (str, bytes)):
            samples = targets
        else:
            continue
        for sample in samples[:_SAMPLE_ROWS]:
            if not isinstance(sample, Mapping) or "boxes" not in sample:
                continue
            values = _numeric_box_values(sample["boxes"])
            if values is None:
                return None
            coordinates.extend(values)
    return coordinates or None


def _measure_target_box_grid(loaded: object) -> tuple[str | None, str]:
    """Observe grid-vs-normalized magnitude on named target boxes only."""
    coordinates = _target_box_coordinates(loaded)
    if not coordinates:
        return None, (
            "load_data output has no named targets[_train|_test][*].boxes "
            "coordinate carrier"
        )
    max_abs = max(abs(value) for value in coordinates)
    frac_negative = (
        sum(1 for value in coordinates if value < -1e-9) / len(coordinates)
    )
    detail = (
        f"load_data target boxes: max|·|={max_abs:.3g}, "
        f"neg_frac={frac_negative:.2f}"
    )
    if frac_negative >= 0.05:
        return None, f"target boxes contain appreciable negative coordinates ({detail})"
    if max_abs <= 1.5:
        return "zero_one", detail
    return "grid_coordinates", detail


def assess(run_dir: Path) -> dict:
    from probes.scale_mismatch import values_equal  # noqa: PLC0415

    spec_path = run_dir / ".pipeline" / "method_spec.json"
    params_path = run_dir / ".pipeline" / "params.json"
    try:
        spec = json.loads(spec_path.read_text(encoding="utf-8"))
        params_doc = json.loads(params_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        return {"applicable": False, "reason": f"artifact unreadable: {e}"}
    params = params_doc.get("params", params_doc)

    declared = [
        e for e in ((spec.get("critical_requirements") or {})
                    .get("scale_dependent_hyperparameters") or [])
        if isinstance(e, dict) and e.get("name")
        and e.get("paper_value") is not None
        and (
            e.get("calibration_context") is not None
            or e.get("assumes_data_scale") is not None
        )
    ]
    if not declared:
        return {"applicable": False,
                "reason": "no scale-dependent hyperparameters with a "
                          "paper_value + calibration context declared"}

    entries: list[dict] = []
    pending: list[tuple[dict, object]] = []
    required_observers: set[str] = set()
    for decl in declared:
        name = decl["name"]
        paper_value = decl["paper_value"]
        route = dispatch_calibration_context(decl)
        declared_context = (
            decl.get("calibration_context")
            if "calibration_context" in decl
            else decl.get("assumes_data_scale")
        )
        entry = {
            "name": name,
            "paper_value": paper_value,
            "assumes": route.expected_observation,
            "assumes_declared": declared_context,
            "calibration_dispatch": route.as_dict(),
            "measured": None,
        }
        param_entry = params.get(name)
        live = (param_entry or {}).get("value") if isinstance(param_entry, dict) else None
        if not isinstance(param_entry, dict) or live is None:
            entry.update(status="needs_attention",
                         reason=f"{name} declared scale-dependent but absent "
                                f"from params.json")
        elif route.observer is None:
            entry.update(
                status="unprobeable",
                reason=route.reason or (
                    "calibration context has no supported runtime observer"
                ),
            )
        else:
            required_observers.add(route.observer)
            pending.append((entry, route))
        entries.append(entry)

    # Other/unit-norm/already-rescaled entries need no observation.  Avoiding
    # an irrelevant loader call is part of the contract: failure to load
    # feature rows cannot turn unknown physical units into a fake scale issue.
    if not required_observers:
        return {
            "applicable": True,
            "measured_scale": None,
            "measure_detail": "no supported observer required",
            "observations": {},
            "entries": entries,
        }

    # One small load_data call supplies the declared structural observation
    # surfaces.  No tuple position is guessed for target boxes.
    from probes.package_loader import (  # noqa: PLC0415
        ProbeLoadError, imported_method_package)
    from probes.trainability import _fill_kwargs  # noqa: PLC0415

    try:
        with imported_method_package(run_dir) as package:
            load_data = getattr(package, "load_data", None)
            if load_data is None:
                return {"applicable": False,
                        "reason": "package exposes no load_data to measure"}
            kwargs = _fill_kwargs(load_data, _LOAD_CANDIDATES)
            if isinstance(kwargs, str):
                return {"applicable": False,
                        "reason": f"load_data requires unmapped parameter "
                                  f"{kwargs!r}"}
            loaded = load_data(**kwargs)
    except ProbeLoadError as e:
        return {"applicable": False, "reason": f"package import failed: {e}"}
    except Exception as e:  # noqa: BLE001 — generated loaders fail arbitrarily
        return {"applicable": False,
                "reason": f"load_data raised {type(e).__name__}: {e}"}

    observations: dict[str, dict[str, str | None]] = {}
    if FEATURE_MAGNITUDE_OBSERVER in required_observers:
        features = _feature_array(loaded)
        if features is None:
            observation = (None, "no feature array in load_data output")
        else:
            observation = _measure_scale(features)
        observations[FEATURE_MAGNITUDE_OBSERVER] = {
            "measured": observation[0], "detail": observation[1],
        }
    if TARGET_BOX_GRID_OBSERVER in required_observers:
        observation = _measure_target_box_grid(loaded)
        observations[TARGET_BOX_GRID_OBSERVER] = {
            "measured": observation[0], "detail": observation[1],
        }

    for entry, route in pending:
        observed = observations[route.observer]
        measured = observed["measured"]
        detail = observed["detail"]
        entry["measured"] = measured
        entry["measure_detail"] = detail
        if measured is None:
            entry.update(
                status="unprobeable",
                reason=f"could not observe {route.observer}: {detail}",
            )
        elif measured == route.expected_observation:
            entry.update(
                status="at_calibration",
                reason=f"delivered {route.observer} is {measured}, matching "
                       "the declared calibration",
            )
        elif route.scale == "raw_pixel_unnormalized" \
                and measured == "zero_one":
            converted = float(entry["paper_value"]) / 255.0
            live_entry = params.get(entry["name"])
            live = (
                live_entry.get("value")
                if isinstance(live_entry, dict) else None
            )
            if values_equal(live, converted):
                entry.update(
                    status="already_rescaled",
                    reason=f"live value {live} equals the exact supported "
                           f"conversion {entry['paper_value']} / 255 = "
                           f"{converted}",
                )
            elif values_equal(live, entry["paper_value"]):
                entry.update(
                    status="rescale",
                    new_value=converted,
                    derivation=f"{entry['paper_value']} * (1/255) — the value is "
                               "calibrated for raw [0, 255] pixel distances and "
                               "the delivered data is [0, 1]-normalized "
                               "(per-channel /255), so L2 distances shrink by "
                               "255x and the threshold must shrink with them",
                )
            else:
                entry.update(
                    status="needs_attention",
                    reason=f"live value {live} matches neither the paper "
                           f"value {entry['paper_value']} nor its exact "
                           f"raw-pixel to zero-one conversion {converted}",
                )
        else:
            entry.update(
                status="needs_attention",
                reason=f"no deterministic conversion defined for "
                       f"{route.expected_observation} (declared) -> {measured} "
                       f"(measured by {route.observer}); the stage-5 scale "
                       "probe is the backstop",
            )

    primary = (
        observations.get(FEATURE_MAGNITUDE_OBSERVER)
        or observations.get(TARGET_BOX_GRID_OBSERVER)
        or {"measured": None, "detail": ""}
    )
    return {
        "applicable": True,
        "measured_scale": primary["measured"],
        "measure_detail": primary["detail"],
        "observations": observations,
        "entries": entries,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__.strip().splitlines()[0])
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    result = assess(args.run_dir)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    if not result.get("applicable"):
        print(f"scale calibration N/A: {result.get('reason')}")
        return 0
    for entry in result["entries"]:
        print(f"  {entry['name']}: {entry['status']}"
              + (f" -> {entry['new_value']}" if entry.get("new_value") is not None else "")
              + f" ({entry.get('reason') or entry.get('derivation', '')[:80]})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
