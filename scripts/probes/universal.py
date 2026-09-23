"""Universal behavioral probes (recentering slice 1.3, catalog tier ii).

Implemented here:
- UB-1  probe_shape_claim          (M-001 class: docstring shape claims vs execution)
- UB-2/3 probe_callable_sensitivity (degenerate scores / decision-variable
                                     insensitivity, M-005 routing supported)
- UB-6  probe_executed_notebook    (chance-level curves, metric-agnostic
                                    flatness, unvetted-number disclosure,
                                    repeated constants)
- UB-8  probe_seed_threading       (same seed = same output, seeds differ)

Grounded against the zoo (tests/test_universal_probes.py asserts the
acceptance matrix). Parsers are built from the REAL executed-output formats
(`Round 0: labels=1000, test_acc=0.0900`, `Classes: 10`,
`tensor([1., 1., 1., 1., 1.])`), per the US-3 lesson: design matchers from
artifacts, not from imagination.

UB-4/5/7 live in the per-paradigm slice (they need paradigm context or a full
trainable package).
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Callable

import numpy as np

from probes import ProbeVerdict
from probes.catalogs.universal import PROBE_CATALOG as _PROBE_CATALOG

PROBE_CATALOG = _PROBE_CATALOG

# ---------------------------------------------------------------------------
# UB-6: executed-notebook output sanity
# ---------------------------------------------------------------------------

# Accuracy-named series for the chance arm (1/n_classes is the chance baseline
# for accuracy only, so this stays accuracy-specific by design). The optional
# qualifier (test/val) and the acc/accuracy token are joined by a flexible
# separator so the spaced, abbreviated labels notebook authors actually print
# ("test acc: 0.56") match alongside the underscored forms ("test_acc=0.56").
# The leading \b keeps "acc" from matching inside another word.
# Separator: an explicit = / : or bare whitespace. The bare-whitespace arm
# is the Rethinking 2026-07-20 matrix shape ("Train loss 2.6362  acc 0.0977")
# — those epoch lines carried the whole headline series, and requiring =/:
# left the beats-chance verdict undecidable at one matched point. The float
# must follow the separator immediately, so "accuracy improved 0.9x" style
# prose still does not match.
_METRIC_RE = re.compile(
    r"\b(?:(?:test|val|validation)[ _]+)?acc(?:uracy)?(?:\s*[=:]\s*|\s+)([0-9]*\.[0-9]+)",
    re.IGNORECASE,
)
_N_CLASSES_RE = re.compile(
    r"\b(?:n[_ ]?|num[_ ]?)?classes\s*[:=]\s*(\d+)", re.IGNORECASE)
# Count-first phrasing ("784 features, 10 classes") — the GBALD 2026-06-30
# notebook printed the class count only this way, so the chance arm never ran
# and a healthy 0.46→0.88 accuracy curve shipped as an unvetted flag.
_N_CLASSES_COUNT_FIRST_RE = re.compile(r"\b(\d+)\s+classes\b", re.IGNORECASE)
_CONST_LINE_RE = re.compile(r"=\s*([0-9]+\.[0-9]+)\s*$", re.MULTILINE)
_TENSOR_LITERAL_RE = re.compile(r"tensor\(\[([0-9.,\sеe+-]+)\]")

# Metric-agnostic series (claims-ledger CT-3, P0 instrumentation §1.2). A
# `key=float` / `key: float` pattern, FLOATS ONLY so integer counters like
# `labels=1000` or `Classes: 10` are excluded — a metric is a decimal
# progression, a counter is not. The identifier must sit immediately before
# the separator, so `Round 0: labels=...` never binds "Round" to a value.
_GENERIC_KEYVAL_RE = re.compile(
    r"\b([A-Za-z_][A-Za-z0-9_]*)\s*[=:]\s*(-?[0-9]+\.[0-9]+)\b")

# A recurring numeric series whose relative spread (range over |mean|) falls
# below this is degenerate/flat regardless of metric name or class count — the
# metric-agnostic backstop for the chance arm, which only understands accuracy.
# 2% sits well below a genuine learning curve's movement (the gbald chance
# curve itself spans ~30%) AND below the ~25% spread of the class-less accuracy
# [0.090, 0.105, 0.116] case, which must read as unvetted, never flat.
_FLATNESS_EPS = 0.02

# A diverged run prints NaN/inf as literal text ("loss=nan"), which the float
# regex above never captures — so without this the divergence reads as a silent
# `unprobeable`, not a fail. Matches a NaN/inf only in metric-value position
# (`key = nan`), so the word "info"/"inference" in prose never trips it.
_NAN_INF_RE = re.compile(
    r"\b([A-Za-z_][A-Za-z0-9_]*)\s*[=:]\s*([+-]?(?:nan|inf))\b", re.IGNORECASE)


def _cell_output_texts(nb: dict) -> list[str]:
    """One concatenated text blob per code cell that produced output."""
    blobs: list[str] = []
    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        parts: list[str] = []
        for out in cell.get("outputs", []):
            if out.get("output_type") == "stream":
                parts.append("".join(out.get("text", [])))
            else:
                parts.append("".join(out.get("data", {}).get("text/plain", [])))
        if parts:
            blobs.append("\n".join(parts))
    return blobs


# The beats-chance margin — ONE rule, two enforcement points (the UB-6
# battery verdict at stage 5 and the smoke-time trainability pre-check in
# the driver, queue item 2026-07-03).
CHANCE_MARGIN = 0.15


def extract_metric_series_with_cell(nb: dict) -> tuple[list[float], int | None]:
    """Longest per-cell accuracy series in the executed outputs, plus the
    index of the code cell that printed it (the smoke-time trainability
    check routes its synthetic failure to that cell; UB-6 only needs the
    values)."""
    best: list[float] = []
    best_cell: int | None = None
    for idx, cell in enumerate(nb.get("cells", [])):
        if cell.get("cell_type") != "code":
            continue
        parts: list[str] = []
        for out in cell.get("outputs", []) or []:
            if out.get("output_type") == "stream":
                parts.append("".join(out.get("text", [])))
            else:
                parts.append("".join((out.get("data") or {}).get("text/plain", [])))
        series = [float(m) for m in _METRIC_RE.findall("\n".join(parts))]
        if len(series) > len(best):
            best, best_cell = series, idx
    return best, best_cell


def extract_metric_series(nb: dict) -> list[float]:
    """Longest per-cell metric series found in the executed outputs."""
    return extract_metric_series_with_cell(nb)[0]


def detect_n_classes(nb: dict) -> int | None:
    for blob in _cell_output_texts(nb):
        m = _N_CLASSES_RE.search(blob)
        if m:
            return int(m.group(1))
    for blob in _cell_output_texts(nb):
        m = _N_CLASSES_COUNT_FIRST_RE.search(blob)
        if m:
            return int(m.group(1))
    return None


def recurring_numeric_series(nb: dict) -> dict[str, list[float]]:
    """Every recurring ``key=float`` series across the executed outputs (a key
    seen at least three times), metric-agnostic (loss / rmse / f1 / accuracy
    alike). Floats only, so integer counters are excluded. The flatness arm
    needs ALL series — degeneracy means nothing moved, so a flat config echo
    beside a moving metric must not read as a degenerate run."""
    groups: dict[str, list[float]] = {}
    for blob in _cell_output_texts(nb):
        for key, raw in _GENERIC_KEYVAL_RE.findall(blob):
            groups.setdefault(key, []).append(float(raw))
    return {k: v for k, v in groups.items() if len(v) >= 3}


def extract_numeric_series(nb: dict) -> tuple[str, list[float]]:
    """Longest recurring ``key=float`` series; ``("", [])`` when none. The
    chance arm needs exactly the accuracy series (``extract_metric_series``);
    these arms and the descent checker (UB-9) need any series, hence the
    separate metric-agnostic extractor."""
    best_key, best = "", []
    for key, series in recurring_numeric_series(nb).items():
        if len(series) > len(best):
            best_key, best = key, series
    return best_key, best


def relative_spread(series: list[float]) -> float:
    """Range over |mean| — a scale-free measure of how much a series moves."""
    mean = sum(series) / len(series)
    return (max(series) - min(series)) / (abs(mean) + 1e-9)


def is_flat_series(series: list[float], eps: float = _FLATNESS_EPS) -> bool:
    """A recurring numeric series with negligible relative spread: a degenerate
    executed output (no movement) under any metric name or class count."""
    return len(series) >= 3 and relative_spread(series) < eps


_LOSS_KEY_RE = re.compile(r"loss", re.IGNORECASE)
_RUNAWAY_RATIO = 20.0
_ACC_KEY_RE = re.compile(
    r"(?:(?:test|val|validation)[ _]*)?acc(?:uracy)?", re.IGNORECASE)


def runaway_loss_series(nb: dict) -> tuple[str, list[float]] | None:
    """A loss-family series whose magnitude explodes monotonically within one
    cell: the optimizer is descending an unbounded objective, not learning
    (bev-distill 2026-07-02: a bbox-format mismatch inverted the foreground
    mask and the executed distillation loss ran -0.05 → -18.9; UB-6 could
    only call it "unvetted"). Magnitude-based and per-cell, so improving
    log-likelihoods (shrinking |v|), noisy-but-bounded losses, and small demo
    losses printed in OTHER cells never trip it. Loss-named keys only: a
    growing reward/score series is legitimate."""
    for blob in _cell_output_texts(nb):
        groups: dict[str, list[float]] = {}
        for key, raw in _GENERIC_KEYVAL_RE.findall(blob):
            groups.setdefault(key, []).append(float(raw))
        for key, series in groups.items():
            if not _LOSS_KEY_RE.search(key) or len(series) < 4:
                continue
            mags = [abs(v) for v in series]
            if any(later < earlier for earlier, later in zip(mags, mags[1:])):
                continue
            if mags[-1] >= _RUNAWAY_RATIO * max(mags[0], 1e-9):
                return key, series
    return None


def first_nan_inf(blobs: list[str]) -> tuple[str, str] | None:
    """The first ``key = nan/inf`` in the executed outputs, or None — a run
    whose metric went non-finite, which the float-series parser cannot see
    (NaN/inf print as text)."""
    for blob in blobs:
        m = _NAN_INF_RE.search(blob)
        if m:
            return m.group(1), m.group(2)
    return None


def first_nan_inf_with_cell(nb: dict) -> tuple[str, str, int] | None:
    """The first ``key = nan/inf`` plus the index of the code cell that printed
    it, or None.

    Same rule as `first_nan_inf`, one rule with two enforcement points
    (R2C-071): the stage-5 battery needs only the values, while the smoke-time
    pre-check has to route a synthetic failure at the cell that printed them,
    exactly as the beats-chance rule already does through
    `extract_metric_series_with_cell`."""
    for idx, cell in enumerate(nb.get("cells", [])):
        if cell.get("cell_type") != "code":
            continue
        parts: list[str] = []
        for out in cell.get("outputs", []) or []:
            if out.get("output_type") == "stream":
                parts.append("".join(out.get("text", [])))
            else:
                parts.append("".join((out.get("data") or {}).get("text/plain", [])))
        m = _NAN_INF_RE.search("\n".join(parts))
        if m:
            return m.group(1), m.group(2), idx
    return None


# ---------------------------------------------------------------------------
# Attributing a non-finite metric (R2C-071's misdiagnosis rider). A NaN in a
# metric has more than one source, and the delivered wording used to assert
# the model-side one: "the run diverged". On the 2026-08-06 pdfgnn delivery
# nothing diverged — 0 of 207,563 weights were NaN and every forecast was
# finite — and the NaN came from a held-out window whose ground truth was
# missing. The confident wrong explanation sent the reviewer hunting a config
# bug. Where the run's own artifacts carry a competing explanation, the
# message must name it instead of asserting divergence.


def withheld_target_note(run_dir: Path) -> str | None:
    """The bundle's own recorded reason a metric could be NaN, or None.

    Reads the live-extent facts the materializer records (R2C-069): a public
    forecasting dataset normally withholds its target over a final period, so
    an evaluation window inside that region scores predictions against missing
    values. Best-effort and read-only — no bundle, no note."""
    provenance = Path(run_dir) / "method" / "example_data" / "PROVENANCE.json"
    try:
        manifest = json.loads(provenance.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, ValueError):
        return None
    for entry in manifest.get("files") or []:
        if not isinstance(entry, dict):
            continue
        axis = entry.get("time_axis")
        if not isinstance(axis, dict):
            continue
        dead = [d for d in (axis.get("dead_tail_columns") or [])
                if isinstance(d, dict) and d.get("column")]
        if not dead:
            continue
        detail = ", ".join(
            f"`{d.get('column')}` after {d.get('last_live_step')} "
            f"({d.get('dead_tail_steps')} step(s))"
            for d in dead
        )
        return (
            f"the bundled `{entry.get('file')}` carries no values for "
            f"{detail}, so an evaluation window inside that region scores "
            f"against missing ground truth"
        )
    return None


def nonfinite_metric_message(
    key: str, token: str, run_dir: Path | None = None,
) -> str:
    """What to say about a metric that printed NaN or inf.

    States the observation, then the candidate sources. Divergence is one of
    them, never the asserted diagnosis, unless nothing else can be named — and
    even then it stays a candidate, because the outputs alone cannot tell a
    diverged model from a missing target."""
    note = withheld_target_note(run_dir) if run_dir is not None else None
    sources = (
        f"the run's own bundle provenance offers one: {note}. "
        "The other candidates are training divergence and an undefended "
        "division by zero in the metric itself"
        if note else
        "the candidates are missing or all-zero ground truth in the "
        "evaluated window, training divergence, and an undefended division "
        "by zero in the metric itself"
    )
    return (
        f"executed metric {key!r} printed {token!r} — a non-finite value "
        f"reached the metric computation. Diagnose before repairing: "
        f"{sources}. Check whether the model's own weights and predictions "
        f"are finite before treating this as divergence."
    )


def probe_executed_notebook(
    nb_path: Path,
    n_classes: int | None = None,
    margin: float = CHANCE_MARGIN,
) -> list[ProbeVerdict]:
    """UB-6: the executed outputs must show the method working.

    One primary series verdict, by priority:
    - a NaN/inf printed in metric-value position (`loss=nan`) fails as a diverged
      run (the float parser cannot see NaN/inf text, so this is a literal scan);
    - an accuracy series must beat chance + margin somewhere (chance =
      1/n_classes, auto-detected from a "Classes: N" or "10 classes" output
      line when not supplied). The chance arm is accuracy-named ON PURPOSE:
      1/n_classes is the chance baseline only for accuracy, so a non-accuracy
      metric (auc, dice, score) routes to the arms below rather than a wrong
      chance fail. A per-metric chance baseline is a named follow-on;
    - a loss-named series whose magnitude explodes monotonically within one
      cell fails as a runaway objective (an unbounded loss the optimizer
      exploits — the executed run is descending, not learning);
    - failing the chance arm (a non-accuracy metric, or no class count), a run
      whose EVERY recurring numeric series is near-FLAT fails as a degenerate
      executed output, metric-agnostic per §1.2 (a flat val_loss the accuracy
      regex never saw);
    - a recurring series that moves but is not chance-checkable is
      `flag_for_researcher` ("unvetted executed number present"), an active
      disclosure, never a silent `unprobeable`; it NAMES any co-located flat
      series so a flat degenerate metric beside a moving config echo is surfaced,
      never dropped;
    - only a genuinely number-free executed output stays `unprobeable`.

    Plus, on every cell: >= 3 identical printed values, or an all-identical
    tensor literal, is a degeneracy WARN (UB-2's output-side shadow). An
    unexecuted notebook is `unprobeable`, never a silent pass.
    """
    name = Path(nb_path).name
    try:
        nb = json.loads(Path(nb_path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        return [ProbeVerdict(
            "UB-6", "unprobeable",
            f"notebook could not be read as executed notebook JSON "
            f"({type(e).__name__}: {e})",
            evidence=name,
            reason="notebook_unreadable",
        )]
    blobs = _cell_output_texts(nb)
    if not blobs:
        return [ProbeVerdict(
            "UB-6", "unprobeable",
            "notebook has no executed outputs (delivered unexecuted)",
            evidence=name,
            reason="outputs_missing",
        )]

    verdicts: list[ProbeVerdict] = []

    series = extract_metric_series(nb)                # accuracy-named (chance)
    n_classes = n_classes or detect_n_classes(nb)
    recurring = recurring_numeric_series(nb)           # all metric-agnostic series
    gen_key, gen_series = extract_numeric_series(nb)   # the longest of them
    diverged = first_nan_inf(blobs)
    # One primary series verdict, by priority: divergence (NaN/inf in the
    # output), then the accuracy chance check (1/n_classes is the chance
    # baseline only for an accuracy-named series, so a non-accuracy metric
    # routes to the arms below, never a wrong chance fail), then the
    # metric-agnostic flatness arm (degenerate ONLY when nothing moved — every
    # recurring series flat — so a flat config echo beside a moving metric never
    # false-fails), then the unvetted-number disclosure for a moving series we
    # could not sanity-check (which NAMES any co-located flat series so a
    # degenerate metric beside a moving echo is surfaced, never dropped), and
    # only a number-free output stays unprobeable.
    if diverged:
        dkey, dtok = diverged
        verdicts.append(ProbeVerdict(
            "UB-6", "fail",
            nonfinite_metric_message(dkey, dtok, Path(nb_path).parent),
            evidence=f"{name}: {dkey}={dtok}",
            reason="output_diverged",
        ))
    elif series and n_classes:
        chance = 1.0 / n_classes
        peak = max(series)
        if peak < chance + margin:
            verdicts.append(ProbeVerdict(
                "UB-6", "fail",
                f"metric series never beats chance+margin "
                f"(peak {peak:.3f} vs chance {chance:.2f} + {margin:.2f}) — "
                f"the method shows no sign of working at any round",
                evidence=f"{name}: series={['%.3f' % s for s in series]}",
                reason="below_chance",
            ))
        else:
            verdicts.append(ProbeVerdict(
                "UB-6", "pass",
                f"metric series beats chance (peak {peak:.3f}, "
                f"chance {chance:.2f})",
                evidence=name,
                reason="above_chance",
            ))
    elif runaway := runaway_loss_series(nb):
        rkey, rser = runaway
        verdicts.append(ProbeVerdict(
            "UB-6", "fail",
            f"executed loss series {rkey!r} runs away "
            f"({rser[0]:.4g} → {rser[-1]:.4g}, magnitude x"
            f"{abs(rser[-1]) / max(abs(rser[0]), 1e-9):.0f} and growing every "
            f"step) — the optimizer is descending an unbounded objective, "
            f"not learning",
            evidence=f"{name}: {rkey}={['%.4g' % s for s in rser]}",
            reason="runaway_series",
        ))
    elif recurring:
        flat = {k: s for k, s in recurring.items() if is_flat_series(s)}
        varying = {k: s for k, s in recurring.items() if not is_flat_series(s)}
        if not varying:
            verdicts.append(ProbeVerdict(
                "UB-6", "fail",
                f"every executed metric series is flat ({gen_key!r} sits at "
                f"~{gen_series[0]:.4g} across {len(gen_series)} points, relative "
                f"spread {relative_spread(gen_series):.1%}) — the run shows no "
                f"movement anywhere, a degenerate executed output regardless of "
                f"metric name or class count",
                evidence=f"{name}: {gen_key}={['%.4g' % s for s in gen_series]}",
                reason="flat_series",
            ))
        else:
            # Prefer an accuracy-named series for the disclosure: the GBALD
            # 2026-06-30 flag reported a longer series literally named
            # 'score' while the meaningful test-accuracy curve sat unnamed
            # beside it. Length breaks ties within a family only.
            acc_named = {k: s for k, s in varying.items()
                         if _ACC_KEY_RE.fullmatch(k)}
            pool = acc_named or varying
            key, ser = max(pool.items(), key=lambda kv: len(kv[1]))
            # Surface any co-located flat series too: a flat metric beside a
            # moving config echo must never be silently dropped (it may be a
            # degenerate result, or a constant by design — the researcher rules).
            flat_note = ""
            if flat:
                flat_note = ("; also flat and surfaced for a look (degenerate "
                             "result, or constant by design): "
                             + ", ".join(f"{k}~{flat[k][0]:.4g}"
                                         for k in sorted(flat)))
            verdicts.append(ProbeVerdict(
                "UB-6", "flag_for_researcher",
                f"executed metric series {key!r} moves but is unvetted here (no "
                f"class count for a chance comparison){flat_note} — surfaced "
                f"rather than passed silently",
                evidence=(f"{name}: {key}={['%.4g' % s for s in ser]}"
                          + (f"; flat={sorted(flat)}" if flat else "")),
                reason="unvetted_series",
            ))
    else:
        verdicts.append(ProbeVerdict(
            "UB-6", "unprobeable",
            "no recognizable metric series in executed outputs",
            evidence=name,
            reason="metric_unrecognized",
        ))

    # Repeated-constant scans (warn tier).
    for idx, blob in enumerate(blobs):
        line_vals = _CONST_LINE_RE.findall(blob)
        for val in set(line_vals):
            if line_vals.count(val) >= 3:
                verdicts.append(ProbeVerdict(
                    "UB-6", "warn",
                    f"the same value ({val}) printed {line_vals.count(val)}x "
                    f"in one cell — possible degenerate score/prior",
                    evidence=f"{name}: output cell #{idx}",
                    reason="repeated_constant",
                ))
        for literal in _TENSOR_LITERAL_RE.findall(blob):
            vals = [v for v in re.split(r"[,\s]+", literal.strip()) if v]
            try:
                floats = [float(v) for v in vals]
            except ValueError:
                continue
            if len(floats) >= 4 and len(set(floats)) == 1:
                verdicts.append(ProbeVerdict(
                    "UB-6", "warn",
                    f"tensor literal with {len(floats)} identical values "
                    f"({floats[0]}) — possible degenerate scores",
                    evidence=f"{name}: output cell #{idx}",
                    reason="constant_tensor",
                ))
    return verdicts


# ---------------------------------------------------------------------------
# UB-9: loss descent / best-so-far monotone (claims-ledger CT-3, §1.3)
# ---------------------------------------------------------------------------

# Best-so-far must improve by more than this fraction of the starting value for
# the series to count as descending. Conservative: a learning method's loss
# should fall clearly, and a flat or sign-flipped (rising) curve improves ~0.
_DESCENT_MIN_IMPROVEMENT = 0.05


def extract_loss_series(
    training_history: Mapping[str, object],
) -> tuple[str, list[float]]:
    """Read the validated structured training-loss series, never notebook text."""
    if training_history.get("status") != "recorded":
        return "", []
    observations = training_history.get("loss_observations")
    if not isinstance(observations, list):
        return "", []
    return "training_loss", [
        float(row["value"])
        for row in observations
        if isinstance(row, Mapping)
    ]


def descent_verdict(
    series: list[float],
    key: str = "loss",
    min_improvement: float = _DESCENT_MIN_IMPROVEMENT,
) -> ProbeVerdict:
    """UB-9 core: a loss series must descend (best-so-far falls meaningfully
    from its start). Noise-tolerant via the running minimum, so a bumpy but
    improving curve passes, while a FLAT curve (no learning) and a RISING curve
    (a sign-flipped / maximized objective) both fail. A NEW probe per the design
    note (NOT UB-5, which is `trains-on-synthetic`); UB-5 already partly covers
    the sign-flip by failing to learn, and this is the refinement on the loss
    trajectory itself.

    Known limitation: because the criterion is best-so-far, a run that descends
    once then explodes to a huge FINITE value still passes (it did learn at some
    point). A NaN/inf divergence is caught — here via the isfinite guard on a
    trajectory passed directly, and on the notebook path by UB-6's literal
    NaN/inf scan. A finite-explosion guard would be scale-dependent and
    false-positive-prone, so it is left to the level-comparison machinery."""
    if any(not np.isfinite(v) for v in series):
        return ProbeVerdict(
            "UB-9", "fail",
            f"loss series {key!r} contains NaN/inf — the run diverged",
            evidence=f"{key}={series}",
            reason="output_diverged")
    if len(series) < 3:
        return ProbeVerdict(
            "UB-9", "unprobeable",
            f"loss series {key!r} has {len(series)} point(s) — too short to "
            f"assess descent (need >= 3)",
            reason="series_too_short")
    start = series[0]
    best = min(series)
    improvement = (start - best) / (abs(start) + 1e-9)
    if improvement > min_improvement:
        return ProbeVerdict(
            "UB-9", "pass",
            f"loss {key!r} descends: best-so-far falls from {start:.4g} to "
            f"{best:.4g} ({improvement:.1%} improvement) across "
            f"{len(series)} points",
            evidence=f"{key}={['%.4g' % s for s in series]}",
            reason="loss_descends")
    return ProbeVerdict(
        "UB-9", "fail",
        f"loss {key!r} does not descend: best-so-far moves from {start:.4g} to "
        f"only {best:.4g} ({improvement:.1%}) — a flat curve (no learning) or a "
        f"rising one (the objective may be sign-flipped / maximized)",
        evidence=f"{key}={['%.4g' % s for s in series]}",
        reason="loss_not_descending")


def probe_loss_descent(run_dir: Path) -> ProbeVerdict:
    """UB-9 over the pipeline-owned structured training-history artifact.

    Notebook prose and arbitrary stdout are never evidence.  Missing or
    invalid structured identity is unprobeable; an explicit no-training arm is
    not applicable; only a validated record can pass or fail descent.
    """
    candidate = Path(run_dir)
    artifact = (
        candidate
        if candidate.name == "training_history.json" and candidate.is_file()
        else candidate / ".pipeline" / "training_history.json"
    )
    if not artifact.is_file():
        return ProbeVerdict(
            "UB-9", "unprobeable",
            "no pipeline-validated structured training history is available",
            evidence=artifact.as_posix(),
            reason="training_history_missing",
        )
    try:
        from scripts.time_series_training_history import (  # noqa: PLC0415
            validate_training_history,
        )

        raw = json.loads(artifact.read_text(encoding="utf-8"))
        history = validate_training_history(raw)
    except Exception as exc:  # noqa: BLE001 - invalid evidence is not a verdict
        return ProbeVerdict(
            "UB-9", "unprobeable",
            "structured training history is unreadable or invalid; loss "
            f"descent cannot be assessed ({type(exc).__name__}: {exc})",
            evidence=artifact.as_posix(),
            reason="training_history_invalid",
        )
    if history["status"] == "not_applicable":
        return ProbeVerdict(
            "UB-9", "not_applicable",
            "the structured history declares that this method has no training "
            f"phase: {history['reason']}",
            evidence=f"{artifact.name}: record_digest={history['record_digest']}",
            reason="training_not_applicable",
        )
    key, series = extract_loss_series(history)
    if not series:
        return ProbeVerdict(
            "UB-9", "unprobeable",
            "validated training history contains no loss observations",
            evidence=artifact.as_posix(),
            reason="training_history_loss_missing",
        )
    v = descent_verdict(series, key)
    binding = (
        f"{artifact.name}: model={history['model_id']}, "
        f"checkpoint={history['checkpoint_id']}, "
        f"record_digest={history['record_digest']}"
    )
    return ProbeVerdict("UB-9", v.verdict, v.message,
                        evidence=f"{binding}; {v.evidence}",
                        reason=v.reason)


# ---------------------------------------------------------------------------
# UB-2 / UB-3: callable sensitivity (degenerate scores / unused decision vars)
# ---------------------------------------------------------------------------


def _to_comparable(value: Any) -> np.ndarray:
    return np.asarray(value, dtype=float).ravel()


def probe_callable_sensitivity(
    fn: Callable,
    base_kwargs: dict,
    vary_param: str,
    candidates: list,
    probe_id: str = "UB-3",
    flag_not_fail: bool = False,
    finding_class: str = "",
    tol: float = 1e-9,
) -> ProbeVerdict:
    """Vary ONE input over candidates (others fixed): output must vary.

    `flag_not_fail=True` routes a degenerate result to `flag_for_researcher`
    (the M-005 rule: a faithful transcription of degenerate paper math is
    surfaced to the researcher, never auto-failed or auto-fixed).
    """
    outputs: list[np.ndarray] = []
    for cand in candidates:
        kwargs = dict(base_kwargs)
        kwargs[vary_param] = cand
        try:
            outputs.append(_to_comparable(fn(**kwargs)))
        except Exception as e:  # noqa: BLE001 — generated code fails arbitrarily
            return ProbeVerdict(
                probe_id, "unprobeable",
                f"{fn.__name__}({vary_param}={cand!r}) raised {type(e).__name__}: {e}",
                finding_class=finding_class,
            )
    degenerate = all(
        out.shape == outputs[0].shape and np.allclose(out, outputs[0], atol=tol)
        for out in outputs[1:]
    )
    if degenerate:
        verdict = "flag_for_researcher" if flag_not_fail else "fail"
        return ProbeVerdict(
            probe_id, verdict,
            f"{fn.__name__} output is constant across {len(candidates)} values "
            f"of {vary_param!r} — the input has no effect on the result",
            evidence=f"constant output {outputs[0][:4]}... for "
                     f"{vary_param} in {candidates[:4]}...",
            finding_class=finding_class,
        )
    return ProbeVerdict(
        probe_id, "pass",
        f"{fn.__name__} responds to {vary_param!r} "
        f"({len(candidates)} candidates)",
        finding_class=finding_class,
    )


# ---------------------------------------------------------------------------
# UB-8: seed threading
# ---------------------------------------------------------------------------


def probe_seed_threading(
    fn: Callable,
    kwargs: dict,
    seed_param: str = "seed",
    expect_stochastic: bool = True,
    seeds: tuple[int, ...] = (0, 1, 2, 3),
) -> ProbeVerdict:
    """Same seed twice gives identical output. Different seeds should differ
    when the method is stochastic (warn, not fail, if they never do — a
    tie-break-free deterministic path can be legitimate)."""
    def call(seed):
        return _to_comparable(fn(**{**kwargs, seed_param: seed}))

    try:
        first = call(seeds[0])
        repeat = call(seeds[0])
    except Exception as e:  # noqa: BLE001
        return ProbeVerdict(
            "UB-8", "unprobeable",
            f"{fn.__name__} raised {type(e).__name__}: {e}",
        )
    if first.shape != repeat.shape or not np.allclose(first, repeat):
        return ProbeVerdict(
            "UB-8", "fail",
            f"{fn.__name__} is nondeterministic under a fixed "
            f"{seed_param}={seeds[0]} — seed is not threaded through",
            evidence=f"{first[:4]} vs {repeat[:4]}",
        )
    if expect_stochastic:
        for alt in seeds[1:]:
            other = call(alt)
            if other.shape != first.shape or not np.allclose(other, first):
                return ProbeVerdict(
                    "UB-8", "pass",
                    f"{fn.__name__} threads {seed_param} (deterministic per "
                    f"seed, varies across seeds)",
                )
        return ProbeVerdict(
            "UB-8", "warn",
            f"{fn.__name__} returned identical output for seeds "
            f"{list(seeds)} — stochasticity expected but not observed",
        )
    return ProbeVerdict(
        "UB-8", "pass", f"{fn.__name__} deterministic under fixed seed",
    )


# ---------------------------------------------------------------------------
# UB-1: docstring shape claims vs execution (M-001)
# ---------------------------------------------------------------------------

_SHAPE_CLAIM_RE = re.compile(r"\(\s*([A-Za-z_][A-Za-z0-9_]*(?:\s*[*+]\s*[A-Za-z0-9_]+)?"
                             r"(?:\s*,\s*[A-Za-z0-9_*+ ]+)+)\s*\)")


def parse_return_shape_claim(docstring: str | None) -> tuple[str, ...] | None:
    """First shape-tuple claim in the Returns section, e.g. ('B','N_queries','1')."""
    if not docstring or "Returns" not in docstring:
        return None
    returns_block = docstring.split("Returns", 1)[1]
    m = _SHAPE_CLAIM_RE.search(returns_block)
    if not m:
        return None
    return tuple(part.strip() for part in m.group(1).split(","))


def _resolve_dim(symbol: str, dims: dict[str, int]) -> int | None:
    symbol = symbol.strip()
    if symbol.isdigit():
        return int(symbol)
    if symbol in dims:
        return dims[symbol]
    # Simple products like "n_classes * hidden_dim".
    if "*" in symbol:
        parts = [p.strip() for p in symbol.split("*")]
        resolved = [_resolve_dim(p, dims) for p in parts]
        if all(r is not None for r in resolved):
            out = 1
            for r in resolved:
                out *= r
            return out
    return None


def probe_shape_claim(
    fn: Callable,
    call_args: tuple = (),
    call_kwargs: dict | None = None,
    dims: dict[str, int] | None = None,
    claim: tuple[str, ...] | None = None,
) -> ProbeVerdict:
    """Execute the function and compare the ACTUAL output shape against the
    docstring's CLAIMED shape — never trust the docstring (M-001)."""
    claim = claim or parse_return_shape_claim(fn.__doc__)
    if claim is None:
        return ProbeVerdict(
            "UB-1", "unprobeable",
            f"{fn.__name__} has no parseable Returns shape claim",
            finding_class="M-001",
        )
    dims = dims or {}
    expected = tuple(_resolve_dim(s, dims) for s in claim)
    if any(e is None for e in expected):
        unresolved = [s for s, e in zip(claim, expected) if e is None]
        return ProbeVerdict(
            "UB-1", "unprobeable",
            f"cannot bind claimed dims {unresolved} for {fn.__name__} "
            f"(supply them via dims=...)",
            finding_class="M-001",
        )
    try:
        result = fn(*call_args, **(call_kwargs or {}))
    except Exception as e:  # noqa: BLE001
        return ProbeVerdict(
            "UB-1", "unprobeable",
            f"{fn.__name__} raised {type(e).__name__}: {e}",
            finding_class="M-001",
        )
    actual = tuple(getattr(result, "shape", ()) or ())
    if not actual:
        return ProbeVerdict(
            "UB-1", "unprobeable",
            f"{fn.__name__} returned a shapeless {type(result).__name__}",
            finding_class="M-001",
        )
    actual = tuple(int(d) for d in actual)
    if actual != expected:
        return ProbeVerdict(
            "UB-1", "fail",
            f"{fn.__name__} docstring claims shape {claim} -> {expected} "
            f"but execution produced {actual} — the op does not do what the "
            f"text says",
            evidence=f"claimed {expected}, actual {actual}",
            finding_class="M-001",
        )
    return ProbeVerdict(
        "UB-1", "pass",
        f"{fn.__name__} actual shape {actual} matches its docstring claim",
        finding_class="M-001",
    )
