"""US-4 — typed calibration-context mismatch.

A method-specific hyperparameter whose meaning depends on the input-data scale
(a distance threshold, similarity radius, norm bound) is only correct when the
data is at the scale the value was calibrated for. The canonical case is
GBALD's ``R_0 = 2000``: calibrated for raw ``[0, 255]`` pixel distances, but the
delivered loader normalizes to ``[0, 1]`` where every pairwise distance is
≤ ~28, so the geometric prior ``R_0 / ||x - D_j||`` saturates to 1.0 for every
candidate and ``select_batch`` silently degenerates to plain BALD. The notebook
author rescales (``R_0_scaled``); the imported-package path with the raw
``params.json`` value is the degenerate one.

Each entry selects its observation surface through the shared R2C-091
``calibration_context`` dispatcher.  Feature magnitude and target-box grid
coordinates are distinct contexts; legacy ``assumes_data_scale`` text is
accepted only through the dispatcher's closed compatibility map.  Unknown,
physical-unit, temporal, graph-statistic, scale-free, and unit-norm contexts
remain explicit and unprobeable instead of selecting a nearby observer.

This is the detection half of US-4. The prevention half — the deterministic
rescale at parameter-derivation time (implementing the schema's documented
formula / preset-match / mismatch contract in ``derive_params.py``) is the
driver-touching follow-on that lands in an idle window.

The only conversion this version can verify is raw-pixel to zero-one feature
magnitude (paper value divided by exactly 255).  A target-box grid match may
pass, but a representation mismatch is unprobeable because no grid-resolution
conversion formula is declared.  No observer or conversion is guessed.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

from probes import ProbeVerdict
from probes.calibration_context import (
    FEATURE_MAGNITUDE_OBSERVER,
    TARGET_BOX_GRID_OBSERVER,
    dispatch_calibration_context,
)
from probes.catalogs.scale_mismatch import PROBE_CATALOG as _PROBE_CATALOG

PROBE_CATALOG = _PROBE_CATALOG
PROBE_ID = "US-4"
TIER = "static"

# Canonical scale categories. assumes_data_scale prose and the measured data
# both normalize into one of these so they can be compared.
RAW = "raw"               # raw pixels / unnormalized, magnitudes >> 1 ([0, 255])
ZERO_ONE = "zero_one"     # min-max to [0, 1]
STANDARDIZED = "centered"  # z-score or [-1, 1] — appreciable negative values
GRID = "grid_coordinates"  # detection-target coordinates in grid-cell units
                           # (bev-distill's sigma: Gaussian spread in BEV
                           # cells). Measured over the targets' boxes, not
                           # feature rows — recognized only since the KD
                           # stub kit gave it an exercisable fixture shape.

# How many feature rows to sample when measuring the data scale.
_SAMPLE_ROWS = 64


# ---------------------------------------------------------------------------
# Pure helpers (unit-tested directly)
# ---------------------------------------------------------------------------


def normalize_scale_name(name: object) -> str | None:
    """Compatibility wrapper over the shared closed legacy dispatcher."""
    route = dispatch_calibration_context({"assumes_data_scale": name})
    if route.observer == FEATURE_MAGNITUDE_OBSERVER:
        return route.expected_observation
    if route.observer == TARGET_BOX_GRID_OBSERVER:
        return GRID
    return None


def classify_data_scale(min_val: float, max_abs: float, frac_negative: float) -> str | None:
    """Classify a measured feature sample into a canonical scale category.

    * appreciable negatives ⇒ centered preprocessing (z-score or [-1, 1]); raw
      pixels and [0, 1] are both non-negative, so the negative fraction is the
      cleanest discriminator and is checked first.
    * otherwise magnitude decides: ≤ ~1.5 is [0, 1], larger is raw/unnormalized.
    """
    if frac_negative >= 0.05:
        return STANDARDIZED
    if max_abs <= 1.5:
        return ZERO_ONE
    if max_abs > 1.5:
        return RAW
    return None


def values_equal(a: object, b: object) -> bool:
    if isinstance(a, bool) or isinstance(b, bool):
        return a == b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return math.isclose(float(a), float(b), rel_tol=1e-6, abs_tol=1e-9)
    return a == b


# ---------------------------------------------------------------------------
# Data-scale measurement (reads the delivered data sample; runs no method code)
# ---------------------------------------------------------------------------


def _iter_feature_rows_from_json(path: Path):
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    if not isinstance(obj, dict):
        return None
    for key in ("x_pool", "x_train", "x_test"):
        rows = obj.get(key)
        if isinstance(rows, list) and rows and isinstance(rows[0], list):
            return rows
    return None


def _iter_feature_rows_from_csv(path: Path):
    try:
        with path.open(newline="") as fh:
            reader = csv.reader(fh)
            rows = list(reader)
    except OSError:
        return None
    if len(rows) < 2:
        return None
    data_rows = rows[1:]  # drop header
    out = []
    for r in data_rows:
        # last column is the integer label; all others are float features
        try:
            out.append([float(c) for c in r[:-1]])
        except ValueError:
            return None
    return out or None


def _iter_feature_rows_from_pt(path: Path):
    try:
        import torch  # noqa: PLC0415
    except ImportError:
        return "no_torch"
    try:
        d = torch.load(path, weights_only=False)
    except Exception:  # noqa: BLE0001 — any unpickling failure is "unreadable"
        return None
    if not isinstance(d, dict):
        return None
    for key in ("x_pool", "x_train", "x_test"):
        t = d.get(key)
        if t is not None:
            try:
                return t.float().tolist()
            except Exception:  # noqa: BLE0001
                return None
    return None


def _find_data_file(run_dir: Path) -> Path | None:
    """Locate the delivered data sample. Prefers .pt (the cached form), then
    .json, then a pool/train CSV — the same conventions ``method/data.py``
    reads, searched recursively under ``method/example_data``."""
    base = run_dir / "method" / "example_data"
    if not base.is_dir():
        base = run_dir / "method"
    if not base.is_dir():
        return None
    for ext in (".pt", ".json"):
        hits = sorted(base.rglob(f"*{ext}"))
        if hits:
            return hits[0]
    for cand in sorted(base.rglob("*.csv")):
        if cand.stem.lower() in ("pool", "train"):
            return cand
    return None


def measure_data_scale(run_dir: Path) -> tuple[str | None, str]:
    """Return (canonical_scale_or_None, human_reason).

    A None scale always carries a reason so the verdict can name why it could
    not measure rather than silently passing.
    """
    data_file = _find_data_file(run_dir)
    if data_file is None:
        return None, "no data sample found under method/example_data"
    if data_file.suffix == ".pt":
        rows = _iter_feature_rows_from_pt(data_file)
        if rows == "no_torch":
            return None, f"torch unavailable to read {data_file.name}"
    elif data_file.suffix == ".json":
        rows = _iter_feature_rows_from_json(data_file)
    else:
        rows = _iter_feature_rows_from_csv(data_file)
    if not rows:
        return None, f"could not read feature rows from {data_file.name}"

    sample = rows[:_SAMPLE_ROWS]
    flat = [float(v) for row in sample for v in row]
    if not flat:
        return None, f"empty feature sample in {data_file.name}"
    min_val = min(flat)
    max_abs = max(abs(v) for v in flat)
    frac_negative = sum(1 for v in flat if v < -1e-9) / len(flat)
    scale = classify_data_scale(min_val, max_abs, frac_negative)
    if scale is None:
        return None, (f"data range [min={min_val:.3g}, max|·|={max_abs:.3g}] in "
                      f"{data_file.name} did not match a known scale")
    return scale, (f"{data_file.name}: min={min_val:.3g}, max|·|={max_abs:.3g}, "
                   f"neg_frac={frac_negative:.2f}")


def _iter_box_coords_from_dict(obj) -> list[float] | None:
    """Flatten box coordinates from a targets-convention data dict: any
    ``targets*`` key holding a list of per-sample dicts with a ``boxes``
    array. Returns the flat coordinate list, or None."""
    if not isinstance(obj, dict):
        return None
    coords: list[float] = []
    for key in ("targets_train", "targets", "targets_test"):
        samples = obj.get(key)
        if not isinstance(samples, list):
            continue
        for sample in samples[:_SAMPLE_ROWS]:
            if not isinstance(sample, dict):
                continue
            boxes = sample.get("boxes")
            if boxes is None:
                continue
            tolist = getattr(boxes, "tolist", None)
            rows = tolist() if callable(tolist) else boxes
            if not isinstance(rows, list):
                continue
            for row in rows:
                vals = row if isinstance(row, list) else [row]
                try:
                    coords.extend(float(v) for v in vals)
                except (TypeError, ValueError):
                    return None
        if coords:
            return coords
    return None


def measure_box_coordinate_scale(run_dir: Path) -> tuple[str | None, str]:
    """Measure the scale of the delivered targets' box coordinates —
    ``GRID`` (grid-cell units, magnitudes past 1) vs ``ZERO_ONE``
    (normalized). The counterpart of :func:`measure_data_scale` for
    grid-calibrated params (bev-distill's sigma), whose assumed scale lives
    in detection targets, not feature rows."""
    data_file = _find_data_file(run_dir)
    if data_file is None:
        return None, "no data sample found under method/example_data"
    if data_file.suffix == ".pt":
        try:
            import torch  # noqa: PLC0415
        except ImportError:
            return None, f"torch unavailable to read {data_file.name}"
        try:
            obj = torch.load(data_file, weights_only=False)
        except Exception:  # noqa: BLE001 — any unpickling failure is unreadable
            return None, f"could not unpickle {data_file.name}"
    elif data_file.suffix == ".json":
        try:
            obj = json.loads(data_file.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None, f"could not read {data_file.name}"
    else:
        return None, (f"{data_file.name} is not a targets-convention sample "
                      f"(need .pt or .json with a targets key)")

    coords = _iter_box_coords_from_dict(obj)
    if not coords:
        return None, f"no targets[*].boxes coordinates in {data_file.name}"
    max_abs = max(abs(v) for v in coords)
    frac_negative = sum(1 for v in coords if v < -1e-9) / len(coords)
    detail = (f"{data_file.name} boxes: max|·|={max_abs:.3g}, "
              f"neg_frac={frac_negative:.2f}")
    if frac_negative >= 0.05:
        return None, (f"appreciable negative box coordinates — not "
                      f"grid-cell units ({detail})")
    if max_abs <= 1.5:
        return ZERO_ONE, detail
    return GRID, detail


# ---------------------------------------------------------------------------
# Probe entry point
# ---------------------------------------------------------------------------


def _load_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


# ---------------------------------------------------------------------------
# US-4b — geometric-ranking scale-invariance (queue 11d Arm B, the folded 11f)
# ---------------------------------------------------------------------------
#
# US-4 above is static: it checks the data scale against a value's declared
# calibration and certifies only the exact raw-pixel /255 conversion. The
# GBALD audit called the historical rounded rescale a band-aid (R_0 rescaled
# 2000 -> 7.843 but still ~76% saturated). The deeper investigation found the
# real Stage-2 root
# cause: rank_by_representativeness misapplied Eq-5's PROBABILITY cap (p=1 within
# R_0) to Eq-13's RANKING score, assigning +inf to within-radius candidates.
# That ties them and collapses the ranking to input order — GBALD silently
# degenerates to plain BALD.
#
# The paper's Eq-13 ranks by the RAW ratio max_j R_0/||x-D_j||. A ratio with a
# constant numerator sorts identically for ANY positive R_0, so a FAITHFUL
# ranking is scale-invariant: select_batch's output must not change when R_0
# changes. A CAPPED ranking is R_0-sensitive (the within-radius mask, hence the
# ties, move with R_0). So US-4b perturbs the scale-dependent parameter across a
# wide range through the actual pluggable and checks invariance:
#   * select_batch invariant to R_0  -> pass (raw-ratio ranking, as Eq-13 wants)
#   * select_batch changes with R_0  -> fail (the ranking caps; degenerate)
#
# This directly detects the cap bug and, unlike a within-radius saturation
# fraction, does NOT over-flag a correctly uncapped ranking (high saturation is
# fine when the ranking ranks by the raw ratio). Scoped to a distance-radius
# scale-dependent param that is a kwarg of the pluggable; other kinds skip.

PROBE_ID_RANK = "US-4b"


def _is_distance_radius_param(entry: dict) -> bool:
    """True when a scale-dependent hyperparameter is a distance threshold or
    radius (the geometric-ranking case), inferred from its declared name and
    description. Pure norm bounds skip — they are not an inverse-distance
    ranking constant."""
    text = f"{entry.get('name', '')} {entry.get('description', '')}".lower()
    return "distance" in text or "radius" in text


def _ranking_sweep_values(live: float | None) -> list[float]:
    """Values spanning well below and well above a selector fixture's O(1)
    distance scale, anchored at the live value when known. A faithful
    ratio-ranking is invariant across all of them; a capped one is not."""
    vals = {0.01, 1.0, 100.0}
    if isinstance(live, (int, float)) and not isinstance(live, bool) and live > 0:
        vals |= {float(live), float(live) * 0.01, float(live) * 100.0}
    return sorted(vals)


def _select_at(kit: dict, name: str, value: float):
    """Run the pluggable with the scale param set to `value`, under the kit's
    fixed seeds, and return the selection as a comparable tuple."""
    import numpy as np  # noqa: PLC0415
    import torch  # noqa: PLC0415

    kwargs = dict(kit["kwargs"])
    kwargs[name] = float(value)
    torch.manual_seed(kit["seed"])
    np.random.seed(kit["seed"])
    return tuple(int(p) for p in kit["fn"](**kwargs))


def probe_geometric_saturation(run_dir: Path) -> list[ProbeVerdict]:
    """US-4b — one verdict per declared distance-radius scale-dependent
    parameter that is a pluggable kwarg: pass when the pluggable's selection is
    invariant to the parameter (faithful raw-ratio ranking, Eq-13), fail (M-004)
    when it changes (the ranking caps within-radius scores and degenerates),
    []/unprobeable otherwise."""
    run_dir = Path(run_dir)
    spec = _load_json(run_dir / ".pipeline" / "method_spec.json")
    if not spec:
        return []  # US-4 owns the no-spec disclosure
    entries = ((spec.get("critical_requirements") or {})
               .get("scale_dependent_hyperparameters") or [])
    radius_entries = [
        entry for entry in entries
        if isinstance(entry, dict)
        and dispatch_calibration_context(entry).observer
        == FEATURE_MAGNITUDE_OBSERVER
        and _is_distance_radius_param(entry)
    ]
    if not radius_entries:
        return []  # no distance-radius scale-dependent param — N/A
    if not (run_dir / "method").is_dir():
        return []  # no method package to exercise

    try:
        import torch  # noqa: F401,PLC0415
    except ImportError:
        return [ProbeVerdict(PROBE_ID_RANK, "unprobeable", "torch required",
                             tier="behavioral")]

    import inspect  # noqa: PLC0415

    from probes.package_loader import (  # noqa: PLC0415
        ProbeLoadError, imported_method_package)
    from probes.term_ablation import al_selector_kit  # noqa: PLC0415

    pluggable = (((spec.get("comparison") or {}).get("pluggable_component") or {})
                 .get("name") or "select_batch")
    params_doc = _load_json(run_dir / ".pipeline" / "params.json") or {}
    params = (params_doc.get("params", params_doc)
              if isinstance(params_doc, dict) else {})

    def _live(name: str):
        e = params.get(name) if isinstance(params, dict) else None
        v = e.get("value") if isinstance(e, dict) else None
        return v if isinstance(v, (int, float)) and not isinstance(v, bool) else None

    verdicts: list[ProbeVerdict] = []
    try:
        with imported_method_package(run_dir) as method:
            kit = al_selector_kit(method, pluggable)
            if isinstance(kit, str):
                return [ProbeVerdict(
                    PROBE_ID_RANK, "unprobeable",
                    f"could not build the selector kit for {pluggable!r}: {kit}",
                    tier="behavioral")]
            sig_params = set(inspect.signature(kit["fn"]).parameters)

            for entry in radius_entries:
                name = entry.get("name") or "?"
                if name not in sig_params:
                    verdicts.append(ProbeVerdict(
                        PROBE_ID_RANK, "unprobeable",
                        f"{name} is not a parameter of {pluggable} — cannot test "
                        f"ranking scale-invariance", tier="behavioral"))
                    continue
                sweep = _ranking_sweep_values(_live(name))
                try:
                    selections = [_select_at(kit, name, v) for v in sweep]
                except Exception as e:  # noqa: BLE001 — generated code fails arbitrarily
                    verdicts.append(ProbeVerdict(
                        PROBE_ID_RANK, "unprobeable",
                        f"{name}: {pluggable} raised under perturbation "
                        f"({type(e).__name__}: {e})", tier="behavioral"))
                    continue
                invariant = all(s == selections[0] for s in selections)
                lo, hi = sweep[0], sweep[-1]
                if invariant:
                    verdicts.append(ProbeVerdict(
                        PROBE_ID_RANK, "pass",
                        f"{name}: {pluggable}'s selection is invariant across "
                        f"{name} in [{lo:g}, {hi:g}] — the representativeness ranking "
                        f"uses the raw {name}/distance ratio (Eq-13), which is "
                        f"scale-invariant as the paper intends.",
                        tier="behavioral", evidence=f"sweep={sweep}"))
                else:
                    verdicts.append(ProbeVerdict(
                        PROBE_ID_RANK, "fail",
                        f"{name}: {pluggable}'s selection CHANGES with {name} (tested "
                        f"[{lo:g}, {hi:g}]). A faithful representativeness ranking "
                        f"(Eq-13, raw {name}/distance) is scale-invariant — a constant "
                        f"numerator cannot reorder candidates. Sensitivity means the "
                        f"ranking caps within-radius scores at 1.0/inf (the Eq-5 "
                        f"probability), tying the saturated candidates and collapsing "
                        f"the ranking to input order (the method degenerates to plain "
                        f"uncertainty sampling). Rank by the raw {name}/distance ratio; "
                        f"do NOT rescale {name} (the ranking does not depend on its "
                        f"magnitude).",
                        tier="behavioral", evidence=f"sweep={sweep}",
                        finding_class="M-004"))
    except ProbeLoadError as e:
        return [ProbeVerdict(PROBE_ID_RANK, "unprobeable",
                             f"package import failed: {e}", tier="behavioral")]
    return verdicts


def probe_scale_dependent_mismatch(run_dir: Path) -> list[ProbeVerdict]:
    """Run US-4 over a run dir. Returns one verdict per declared
    scale-dependent hyperparameter; returns [] (silently N/A) when the spec
    declares none."""
    run_dir = Path(run_dir)
    spec = _load_json(run_dir / ".pipeline" / "method_spec.json")
    if not spec:
        return [ProbeVerdict(PROBE_ID, "unprobeable",
                             "no method_spec.json", tier=TIER)]
    entries = ((spec.get("critical_requirements") or {})
               .get("scale_dependent_hyperparameters") or [])
    entries = [e for e in entries if isinstance(e, dict)]
    if not entries:
        return []  # no scale-dependent hyperparameters declared — N/A, not a verdict

    params_doc = _load_json(run_dir / ".pipeline" / "params.json") or {}
    params = params_doc.get("params", params_doc) if isinstance(params_doc, dict) else {}

    # Observers are lazy and route-specific.  An ``other`` or ``unit_norm``
    # entry must not touch a feature loader merely because it shares this list.
    feature_measured: tuple[str | None, str] | None = None
    box_measured: tuple[str | None, str] | None = None

    verdicts: list[ProbeVerdict] = []
    for entry in entries:
        name = entry.get("name") or "?"
        paper_value = entry.get("paper_value")
        if paper_value is None:
            # formula-only entry: no literal value to check against. The
            # deterministic-rescale half owns formula evaluation; nothing to
            # detect statically here.
            continue
        route = dispatch_calibration_context(entry)
        if route.observer is None:
            label = route.label if route.label is not None else route.scale
            verdicts.append(ProbeVerdict(
                PROBE_ID, "unprobeable",
                f"{name}: calibration context {label!r} is not recognized by "
                f"a supported observer ({route.reason})",
                tier=TIER))
            continue
        param_entry = params.get(name) if isinstance(params, dict) else None
        if not isinstance(param_entry, dict) or "value" not in param_entry:
            verdicts.append(ProbeVerdict(
                PROBE_ID, "unprobeable",
                f"{name}: declared scale-dependent but absent from params.json",
                tier=TIER))
            continue
        live = param_entry.get("value")
        if route.observer == TARGET_BOX_GRID_OBSERVER:
            if box_measured is None:
                box_measured = measure_box_coordinate_scale(run_dir)
            entry_measured, entry_reason = box_measured
        else:
            if feature_measured is None:
                feature_measured = measure_data_scale(run_dir)
            entry_measured, entry_reason = feature_measured
        if entry_measured is None:
            verdicts.append(ProbeVerdict(
                PROBE_ID, "unprobeable",
                f"{name}: could not measure data scale/representation with "
                f"the declared {route.observer} observer ({entry_reason})",
                tier=TIER))
            continue

        expected = route.expected_observation
        scale_matches = entry_measured == expected
        value_unrescaled = values_equal(live, paper_value)

        if scale_matches:
            verdicts.append(ProbeVerdict(
                PROBE_ID, "pass",
                f"{name}: the {route.observer} observer measured "
                f"{entry_measured}, matching the declared calibration "
                f"context; paper value {paper_value} applies directly.",
                tier=TIER,
                evidence=entry_reason))
            continue

        if route.observer == TARGET_BOX_GRID_OBSERVER:
            verdicts.append(ProbeVerdict(
                PROBE_ID, "unprobeable",
                f"{name}: target boxes measure as {entry_measured}, not the "
                f"declared target_box_grid representation. No grid-resolution "
                "conversion formula is declared, so this probe will not "
                "invent one.",
                tier=TIER,
                evidence=(
                    f"paper_value={paper_value}, expected={expected}; "
                    f"{entry_reason}"
                )))
            continue

        if expected == RAW and entry_measured == ZERO_ONE:
            if not isinstance(paper_value, (int, float)) \
                    or isinstance(paper_value, bool):
                verdicts.append(ProbeVerdict(
                    PROBE_ID, "unprobeable",
                    f"{name}: raw-pixel to zero-one calibration requires a "
                    "numeric paper value for the exact /255 comparison",
                    tier=TIER,
                    evidence=entry_reason))
                continue
            converted = float(paper_value) / 255.0
            if values_equal(live, converted):
                verdicts.append(ProbeVerdict(
                    PROBE_ID, "pass",
                    f"{name}={live} was rescaled by the only supported "
                    "conversion: "
                    f"paper value {paper_value} / 255 = {converted} for "
                    "zero-one pixel features.",
                    tier=TIER,
                    evidence=entry_reason))
            elif value_unrescaled:
                verdicts.append(ProbeVerdict(
                    PROBE_ID, "fail",
                    f"{name}={live} is the unrescaled raw-pixel paper value on "
                    f"zero-one features; rescale to the exact supported value "
                    f"{paper_value} / 255 = {converted}.",
                    tier=TIER,
                    evidence=entry_reason,
                    finding_class="M-004"))
            else:
                verdicts.append(ProbeVerdict(
                    PROBE_ID, "fail",
                    f"{name}={live} matches neither the paper value "
                    f"{paper_value} nor its exact raw-pixel to zero-one "
                    f"conversion {converted}.",
                    tier=TIER,
                    evidence=entry_reason,
                    finding_class="M-004"))
            continue

        if value_unrescaled:
            verdicts.append(ProbeVerdict(
                PROBE_ID, "fail",
                f"{name}={live} is the unrescaled paper value, but the delivered "
                f"{route.observer} observer measured {entry_measured} while "
                f"the declared context expects {expected}.",
                tier=TIER,
                evidence=(
                    f"paper_value={paper_value}, expected={expected}; "
                    f"{entry_reason}"
                ),
                finding_class="M-004"))
        else:
            verdicts.append(ProbeVerdict(
                PROBE_ID, "unprobeable",
                f"{name} differs from its paper value, but calibration v1 "
                f"defines no conversion from expected {expected} to observed "
                f"{entry_measured}; the probe cannot certify that change.",
                tier=TIER,
                evidence=entry_reason))
    return verdicts
