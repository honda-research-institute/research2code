"""Fix-loop data-signal check (demo-success-semantics design, 2026-07-16).

The Rethinking-Grouping 2026-07-14 burn: the iteration-0 timeout fix swapped
LABEL-FREE random data into the demo, which made the beats-chance gate
unwinnable for ANY model — two fix iterations (one of them a real,
correctly-diagnosed model bug) burned against an impossible gate before the
judge noticed the data itself. This check asks the deterministic question
BEFORE the loop re-enters: **can the data the fix just installed carry the
signal the gate needs?** For classification shapes, labels must correlate
with features above chance BY CONSTRUCTION (class-conditional mean
separation). The gate stays; the data has to be winnable.

Scope (the maintainer's 2026-07-16 decision): fix loop ONLY — the driver invokes this
after a Stage 3.c fix dispatch changed the demo-data surface (method/data.py
or method/example_data/). It never runs at first notebook authoring, and it
never blocks anything itself: a `fail` verdict is injected into the next fix
dispatch's prompt so the producer fixes the DATA instead of the model.

Applicability is taxonomy-declared, not guessed: the check runs only when the
run's family declares the `beats_chance` demo check (`demo_success.checks`)
— that is the gate whose winnability is being tested. Everything else is
`not_applicable`; data we cannot reach is `unevaluable` (logged, never
blocking, matching the probe layer's never-silent/never-crash rule).

Two data channels, notebook first: fix-mode producers can write nothing but
`.pipeline/notebook_draft.py`, so a fix-loop data swap lives in the notebook
source — the check executes the freshly RENDERED notebook's own setup
sections (boundary resolved from the layout SSOT via
`demo_verdict.setup_section_boundary`, magics skipped, demo/method sections
structurally out of reach) and scans the namespace for the largest
(features, labels) pair. When that channel cannot run, the package's own
`load_data()` is the fallback.

The class-signal rule is split-half nearest-class-mean: fit per-class means
on one half of the samples, classify the other half, and require accuracy
above chance plus the shared margin (probes.universal.CHANCE_MARGIN — one
rule, shared enforcement points). Fitting and scoring on the SAME samples
would overfit noise means and pass random labels at small N; the split makes
label-free/random data score at chance, while class-conditional separated
synthetic data (the pattern-based fix data the design blesses) passes with
huge margin.

Usage (the driver runs this as a bounded subprocess):

    python scripts/fix_data_signal.py --run-dir r2c_runs/<slug>

Output: one JSON object on stdout — {"verdict": "pass"|"fail"|
"not_applicable"|"unevaluable", "message": ..., "stats": {...}}.
Exit 0 for every verdict (the verdict is advisory input to the fix loop);
exit 2 only for CLI misuse.
"""

from __future__ import annotations

import argparse
import inspect
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

# Labels/feature detection bounds: a label array is 1-D integer-typed with a
# plausible class count. Generous ceiling (ImageNet-scale) — the point is to
# exclude index arrays that enumerate every sample.
_MAX_PLAUSIBLE_CLASSES = 1000
_MIN_SAMPLES = 8


def declares_beats_chance(run_dir: Path) -> bool:
    """Does the run's family declare the beats-chance demo check?

    Resolves through the run's provisional-pack overlay so gap-path families
    that declare the check on their pack get the same protection."""
    import taxonomy  # noqa: PLC0415

    spec_path = Path(run_dir) / ".pipeline" / "method_spec.json"
    if not spec_path.is_file():
        return False
    try:
        spec = json.loads(spec_path.read_text(encoding="utf-8"))
        paradigm = str(((spec.get("comparison") or {}).get("classification")
                        or {}).get("id") or "")
    except (OSError, json.JSONDecodeError, AttributeError):
        return False
    if not paradigm:
        return False
    tax = taxonomy.load_taxonomy(
        provisional_packs_dir=taxonomy.run_overlay_dir(Path(run_dir)))
    demo_success = taxonomy.load_demo_success(paradigm, tax)
    return any(
        isinstance(c, dict) and c.get("kind") == "beats_chance"
        for c in demo_success.get("checks") or []
    )


# ---------------------------------------------------------------------------
# Class-signal rule (pure)
# ---------------------------------------------------------------------------


def class_signal_verdict(
    x: np.ndarray, y: np.ndarray, margin: float | None = None
) -> dict[str, Any]:
    """Split-half nearest-class-mean verdict over (features, labels).

    pass: labels correlate with features above chance by construction.
    fail: they do not (random/constant labels — the beats-chance gate is
    unwinnable on this data whatever the model does)."""
    from probes.universal import CHANCE_MARGIN  # noqa: PLC0415

    margin = CHANCE_MARGIN if margin is None else margin
    x = np.asarray(x, dtype=np.float64).reshape(len(y), -1)
    y = np.asarray(y).astype(np.int64).ravel()
    classes = np.unique(y)
    n = len(y)
    if n < _MIN_SAMPLES:
        return {"verdict": "unevaluable",
                "message": f"only {n} labeled samples — too few to assess "
                           f"class signal (need >= {_MIN_SAMPLES})"}
    if len(classes) < 2:
        return {
            "verdict": "fail",
            "message": (
                f"labels are constant (single class {classes[0]!r} across "
                f"{n} samples) — the data carries no class signal, so the "
                f"beats-chance gate can never pass on it"),
            "stats": {"n": int(n), "n_classes": int(len(classes))},
        }
    # Deterministic split-half: even positions fit the class means, odd
    # positions are scored. No RNG — the verdict must be reproducible.
    fit_idx = np.arange(n) % 2 == 0
    eval_idx = ~fit_idx
    means = {}
    for c in classes:
        member = fit_idx & (y == c)
        if member.any():
            means[int(c)] = x[member].mean(axis=0)
    if len(means) < 2 or not eval_idx.any():
        return {"verdict": "unevaluable",
                "message": "the deterministic split leaves fewer than two "
                           "fitted classes to score against"}
    mean_ids = sorted(means)
    stack = np.stack([means[c] for c in mean_ids])
    dists = ((x[eval_idx][:, None, :] - stack[None, :, :]) ** 2).sum(axis=2)
    pred = np.asarray(mean_ids)[dists.argmin(axis=1)]
    acc = float((pred == y[eval_idx]).mean())
    chance = 1.0 / len(classes)
    stats = {"n": int(n), "n_classes": int(len(classes)),
             "split_half_accuracy": round(acc, 4), "chance": round(chance, 4)}
    if acc >= chance + margin:
        return {
            "verdict": "pass",
            "message": (
                f"labels correlate with features by construction: split-half "
                f"nearest-class-mean accuracy {acc:.3f} vs chance "
                f"{chance:.2f} across {len(classes)} classes"),
            "stats": stats,
        }
    return {
        "verdict": "fail",
        "message": (
            f"labels do not correlate with features: split-half "
            f"nearest-class-mean accuracy {acc:.3f} is at chance "
            f"({chance:.2f} for {len(classes)} classes) — label-free/random "
            f"data, so the beats-chance gate can never pass on it whatever "
            f"the model does"),
        "stats": stats,
    }


# ---------------------------------------------------------------------------
# Package data loading (best-effort, never raises out)
# ---------------------------------------------------------------------------


def _to_ndarray(obj: Any) -> np.ndarray | None:
    """numpy view of an array-like (numpy / torch / nested lists), else None."""
    if obj is None or isinstance(obj, (str, bytes, dict, int, float, bool)):
        return None
    if hasattr(obj, "detach"):  # torch tensor without importing torch
        try:
            obj = obj.detach().cpu().numpy()
        except Exception:  # noqa: BLE001 — generated objects fail arbitrarily
            return None
    try:
        arr = np.asarray(obj)
    except Exception:  # noqa: BLE001
        return None
    return arr if arr.dtype != object and arr.size else None


def _flatten_candidates(returned: Any) -> list[np.ndarray]:
    """Array candidates from whatever shape load_data returned."""
    if isinstance(returned, dict):
        items = list(returned.values())
    elif isinstance(returned, (tuple, list)):
        items = list(returned)
    else:
        items = [returned]
    out = []
    for item in items:
        arr = _to_ndarray(item)
        if arr is not None:
            out.append(arr)
    return out


def _plausible_class_count(values: np.ndarray, n: int) -> bool:
    # An index array enumerates ~every sample; a label array repeats values.
    return len(values) <= min(_MAX_PLAUSIBLE_CLASSES, max(2, n // 2))


def _coerce_labels(arr: np.ndarray) -> np.ndarray | None:
    """Interpret an array as class labels when that reading is unambiguous.

    Three encodings coerce (the review's finding 2 — BCE-style float 0.0/1.0
    targets and one-hot matrices must never read as "label-free"):
      - 1-D integer arrays (returned AS-IS, identity preserved);
      - 1-D float arrays whose values are all integral (rounded to int);
      - 2-D strict one-hot rows (exactly one 1 per row, argmax'd).
    Anything else returns None — a 1-D continuous float array might be a
    regression target or a loss curve, and guessing either way misdirects
    the fix loop, so the caller reports `unevaluable` for those."""
    if arr.ndim == 1 and len(arr) >= _MIN_SAMPLES:
        if np.issubdtype(arr.dtype, np.integer):
            return arr if _plausible_class_count(np.unique(arr), len(arr)) else None
        if np.issubdtype(arr.dtype, np.floating):
            if not np.isfinite(arr).all():
                return None
            rounded = np.round(arr)
            if not np.allclose(arr, rounded, atol=1e-6):
                return None
            as_int = rounded.astype(np.int64)
            return as_int if _plausible_class_count(np.unique(as_int), len(as_int)) else None
        return None
    if (arr.ndim == 2 and len(arr) >= _MIN_SAMPLES
            and 2 <= arr.shape[1] <= _MAX_PLAUSIBLE_CLASSES
            and np.issubdtype(arr.dtype, np.number)):
        try:
            one_hot = bool(np.isin(arr, (0, 1)).all()
                           and (arr.sum(axis=1) == 1).all())
        except TypeError:
            return None
        if one_hot:
            return arr.argmax(axis=1).astype(np.int64)
    return None


def find_labeled_pair(
    candidates: list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray] | None:
    """Largest (features, labels) pair among the returned arrays, with the
    label side coerced via `_coerce_labels` (int / integral-float / one-hot)."""
    best: tuple[np.ndarray, np.ndarray] | None = None
    for raw in candidates:
        y = _coerce_labels(raw)
        if y is None:
            continue
        for x in candidates:
            if x is raw or x.ndim < 2 or len(x) != len(y):
                continue
            if best is None or len(y) > len(best[1]):
                best = (x, y)
    return best


def classify_candidates(candidates: list[np.ndarray]) -> tuple[str, Any]:
    """Tagged reading of the harvested arrays, shared by both channels:
      ("pair", (x, y))       — a coercible labeled pair exists
      ("uncoercible", n)     — 1-D label-shaped arrays exist but none coerce
                               (continuous floats, index-like) → the caller
                               reports `unevaluable`, never `fail`
      ("label_free", n)      — arrays exist and nothing is even label-shaped
                               (the Rethinking swap) → `fail`
      ("empty", None)        — nothing array-like harvested
    """
    if not candidates:
        return "empty", None
    pair = find_labeled_pair(candidates)
    if pair is not None:
        return "pair", pair
    one_d = [c for c in candidates if c.ndim == 1 and len(c) >= _MIN_SAMPLES]
    if one_d:
        return "uncoercible", len(one_d)
    return "label_free", len(candidates)


def _harvest_from_rendered_notebook(run_dir: Path) -> tuple[str, Any]:
    """Harvest (features, labels) by executing the rendered notebook's OWN
    setup sections (from §1 up to the layout-declared setup boundary — never
    §0 install and never the demo/method sections).

    This is the channel that sees a notebook-inline data swap: fix-mode
    producers can only write .pipeline/notebook_draft.py, so the recorded
    Rethinking 2026-07-14 label-free swap lived in the notebook source, not
    in method/data.py. Executing the freshly rendered notebook's data cells
    (magic lines skipped) and scanning the resulting namespace evaluates
    exactly the data the demo will train on.

    Returns a `classify_candidates` tag ("pair" / "uncoercible" /
    "label_free" / "empty") or ("error", message) when the channel could not
    run (the caller falls back to the package channel).
    """
    from demo_verdict import (  # noqa: PLC0415
        cell_source, heading_section_number, run_demo_context,
        setup_section_boundary)

    nb_path = run_dir / "notebook.ipynb"
    if not nb_path.is_file():
        return "error", "no rendered notebook.ipynb to execute data cells from"
    try:
        nb = json.loads(nb_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        return "error", f"notebook.ipynb unreadable: {type(e).__name__}"

    # The setup boundary comes from the layout SSOT (never a hardcoded
    # section list), and is capped below the declared headline demo section
    # — demo cells are structurally out of reach even in a compact layout.
    _, demo_success, layout, _ = run_demo_context(run_dir)
    bound = setup_section_boundary(layout, demo_success)

    namespace: dict[str, Any] = {"__name__": "__r2c_data_signal__"}
    section: int | None = None
    cwd_before = Path.cwd()
    sys.path.insert(0, str(run_dir))
    try:
        os.chdir(run_dir)  # notebook-relative paths (method/example_data/)
        for cell in nb.get("cells") or []:
            if cell.get("cell_type") == "markdown":
                for line in cell_source(cell).splitlines():
                    number = heading_section_number(line)
                    if number is not None:
                        section = number
                continue
            if (cell.get("cell_type") != "code" or section is None
                    or not 1 <= section < bound):
                continue
            src = "\n".join(
                ln for ln in cell_source(cell).splitlines()
                if not ln.lstrip().startswith(("%", "!")))
            try:
                exec(compile(src, "<notebook-data-cell>", "exec"), namespace)  # noqa: S102
            except Exception as e:  # noqa: BLE001 — generated cells fail arbitrarily
                return "error", f"notebook data cell raised {type(e).__name__}: {e}"
    finally:
        os.chdir(cwd_before)
        try:
            sys.path.remove(str(run_dir))
        except ValueError:
            pass
    return classify_candidates(_flatten_candidates(
        {k: v for k, v in namespace.items() if not k.startswith("__")}))


def _call_load_data(method: Any, run_dir: Path) -> tuple[Any, str]:
    """(returned value, error message). Fills only safe/derivable kwargs;
    an unfillable REQUIRED parameter makes the data unevaluable, never
    guessed (the probe layer's 2026-06-10 lesson)."""
    load_data = getattr(method, "load_data", None)
    if load_data is None:
        return None, "package exposes no load_data"
    params_path = run_dir / ".pipeline" / "params.json"
    live: dict[str, Any] = {}
    if params_path.is_file():
        try:
            loaded = json.loads(params_path.read_text(encoding="utf-8"))
            for name, entry in (loaded.get("params", loaded) or {}).items():
                live[name] = entry.get("value") if isinstance(entry, dict) else entry
        except (OSError, json.JSONDecodeError, AttributeError):
            live = {}
    try:
        sig = inspect.signature(load_data)
    except (TypeError, ValueError):
        sig = None
    kwargs: dict[str, Any] = {}
    if sig is not None:
        for name, param in sig.parameters.items():
            if param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
                continue
            if name in live:
                kwargs[name] = live[name]
            elif param.default is inspect.Parameter.empty:
                return None, (f"load_data requires parameter {name!r} that "
                              f"params.json does not carry")
    try:
        return load_data(**kwargs), ""
    except Exception as e:  # noqa: BLE001 — generated code fails arbitrarily
        return None, f"load_data raised {type(e).__name__}: {e}"


_LABEL_FREE_MESSAGE = (
    "the demo data exposes no label array alongside its features — "
    "label-free data cannot carry the class signal the beats-chance gate "
    "needs, so the gate can never pass on it")


def _package_channel(run_dir: Path) -> dict[str, Any]:
    """Fallback channel: the package's own load_data()."""
    try:
        from probes.package_loader import (  # noqa: PLC0415
            ProbeLoadError, imported_method_package)
    except ImportError as e:
        return {"verdict": "unevaluable",
                "message": f"probe package loader unavailable: {e}"}
    try:
        with imported_method_package(run_dir) as method:
            returned, err = _call_load_data(method, run_dir)
            if err:
                return {"verdict": "unevaluable", "message": err}
            tag, value = classify_candidates(_flatten_candidates(returned))
            if tag == "pair":
                x, y = value
                return class_signal_verdict(x, y)
            if tag == "label_free":
                return {"verdict": "fail",
                        "message": f"{_LABEL_FREE_MESSAGE} (load_data channel)"}
            if tag == "uncoercible":
                return {"verdict": "unevaluable",
                        "message": f"load_data returned {value} label-shaped "
                                   f"array(s) that could not be confidently "
                                   f"read as class labels"}
            return {"verdict": "unevaluable",
                    "message": "load_data returned nothing array-like to "
                               "assess"}
    except ProbeLoadError as e:
        return {"verdict": "unevaluable", "message": str(e)}
    except Exception as e:  # noqa: BLE001 — never crash the fix loop
        return {"verdict": "unevaluable",
                "message": f"data-signal check raised {type(e).__name__}: {e}"}


def evaluate_run(run_dir: Path) -> dict[str, Any]:
    """The full check over a run dir. Returns the verdict payload.

    Channel order: the rendered notebook's own data cells first (the only
    surface a fix-mode producer can swap data through), then the package's
    load_data() as fallback when the notebook channel could not run."""
    run_dir = Path(run_dir)
    if not declares_beats_chance(run_dir):
        return {"verdict": "not_applicable",
                "message": "the run's family declares no beats_chance demo "
                           "check, so data-signal winnability does not apply"}
    try:
        tag, value = _harvest_from_rendered_notebook(run_dir)
    except Exception as e:  # noqa: BLE001 — never crash the fix loop
        tag, value = "error", f"{type(e).__name__}: {e}"
    if tag == "pair":
        x, y = value
        out = class_signal_verdict(x, y)
        out["channel"] = "notebook"
        return out
    if tag == "label_free":
        # Feature arrays in the demo's own namespace with NO label-shaped
        # array — exactly the Rethinking shape, decided on the notebook's
        # own data.
        return {"verdict": "fail", "channel": "notebook",
                "message": f"{_LABEL_FREE_MESSAGE} (the notebook's own setup "
                           f"cells produced {value} array(s), none label-like)"}
    if tag == "uncoercible":
        # Label-shaped arrays exist but none read unambiguously as class
        # labels (continuous floats, index-like ints). Guessing "fail" here
        # is the same misdirection the check exists to prevent, pointed the
        # other way — stay honest.
        return {"verdict": "unevaluable", "channel": "notebook",
                "message": f"the notebook's setup cells produced {value} "
                           f"label-shaped array(s) that could not be "
                           f"confidently read as class labels"}
    out = _package_channel(run_dir)
    out["channel"] = "package"
    if tag == "error" and out["verdict"] == "unevaluable":
        out["message"] = f"notebook channel: {value}; package channel: " \
                         f"{out['message']}"
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--run-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    if not args.run_dir.is_dir():
        print(f"FAIL: run dir not found: {args.run_dir}", file=sys.stderr)
        return 2
    payload = evaluate_run(args.run_dir)
    print(json.dumps(payload))
    return 0


if __name__ == "__main__":
    sys.exit(main())
