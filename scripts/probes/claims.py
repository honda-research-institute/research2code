"""CT-1 contribution-floor probe, AL arm (claims tier).

The lowest-FP form of "the headline contribution is never exercised": the
generated selector is behaviorally IDENTICAL to a degenerate null on every
trial pool. The nulls are the placeholder idioms generated code actually
produces (v3's `range(N)` placeholder detector, reborn at the claims tier):

- first-k (`list(range(batch_size))` — also what stable topk over an
  all-constant score vector degrades to at the whole-pool level)
- the two seeded-random placeholder idioms (`default_rng(seed).choice` /
  `default_rng(seed).permutation[:k]`), which reproduce exactly because the
  selector receives the same seed kwarg the null uses

Identity must hold on ALL trial pools to fail — a real method matching a
null on three different pools by chance is numerically negligible, so a
fail here is a placeholder or a collapsed composite, never bad luck.

Complementarity with UB-7 (why both exist): UB-7 asks whether each internal
channel CAN influence the selection; CT-1 asks whether the whole method
differs from a null baseline. june9 GBALD separates them — UB-7 flags its
dead prior channel while CT-1 passes (BALD ordering still drives selection).
A selector whose channels are all live but whose composite collapses to
first-k fails CT-1 and passes UB-7.

Per the design note (claims-tier-design.md): the MP/KD arms land with the
paradigm-pack `contribution_ablation` field; the AL arm needs no pack data
because the degenerate-null family is paradigm-universal for selectors.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

import numpy as np

from probe_spec_join import (
    CONTRACT_CONSISTENT,
    CONTRACT_CONTRADICTS,
    CONTRACT_SILENT,
)
from probes import ProbeVerdict
from probes.catalogs.claims import PROBE_CATALOG as _PROBE_CATALOG
from probes.term_ablation import al_selector_kit, attribute_invocation_crash

PROBE_CATALOG = _PROBE_CATALOG

def _null_selections(pool_size: int, batch_size: int, seed: int) -> dict:
    return {
        "first_k": set(range(batch_size)),
        "rng_choice(seed)": set(
            np.random.default_rng(seed)
            .choice(pool_size, batch_size, replace=False).tolist()),
        "rng_permutation(seed)": set(
            np.random.default_rng(seed)
            .permutation(pool_size)[:batch_size].tolist()),
    }


def probe_contribution_floor_al(
    package: ModuleType,
    pluggable_name: str,
    seed: int = 0,
    trials: int = 3,
) -> ProbeVerdict:
    """CT-1 (AL): selector output must not be identical to a degenerate
    null on every trial pool."""
    matches: dict[str, int] = {}
    live_sets: list[set] = []
    for t in range(trials):
        kit = al_selector_kit(package, pluggable_name, seed,
                              fixture_seed=seed + t)
        if isinstance(kit, str):
            return ProbeVerdict("CT-1", "unprobeable", kit)
        try:
            out = kit["invoke"]()
            live = {int(i) for i in out}
        except Exception as e:  # noqa: BLE001 — generated selectors fail freely
            origin, site = attribute_invocation_crash(e, kit["package_dir"])
            if origin == "subject":
                return ProbeVerdict(
                    "CT-1", "unprobeable",
                    f"{pluggable_name} crashed inside the method package's "
                    f"own code at {site} ({type(e).__name__}: {e}) — no "
                    f"null comparison possible; selector crashes are "
                    f"adjudicated by the acquisition-contract probe's "
                    f"crash arm (AL-5)")
            return ProbeVerdict(
                "CT-1", "unprobeable",
                f"selector invocation raised {type(e).__name__}: {e}")
        live_sets.append(live)
        nulls = _null_selections(kit["pool_size"], kit["batch_size"],
                                 kit["seed"])
        for name, null_set in nulls.items():
            if live == null_set:
                matches[name] = matches.get(name, 0) + 1

    degenerate = sorted(n for n, c in matches.items() if c == trials)
    if degenerate:
        return ProbeVerdict(
            "CT-1", "fail",
            f"{pluggable_name} is behaviorally identical to the degenerate "
            f"null {', '.join(degenerate)} on all {trials} trial pools — "
            f"the claimed contribution is never exercised",
            evidence=f"selections={[sorted(s) for s in live_sets]}",
            finding_class="M-004")
    return ProbeVerdict(
        "CT-1", "pass",
        f"selection differs from the degenerate nulls (first-k, "
        f"seeded-random idioms) across {trials} trial pools")


# ---------------------------------------------------------------------------
# CT-5 — discriminating power (claims-ledger §1.3): a verified invariant must
# FAIL when the contribution is ablated, or it verifies nothing. An invariant a
# trivial baseline also satisfies (a monotone loss on a separable blob holds
# even with the contribution coefficient at 0.0; "no duplicate selections"
# holds for random acquisition) carries zero verification weight.
# ---------------------------------------------------------------------------


def random_acquisition(pool_size: int, batch_size: int, seed: int) -> set:
    """The AL contribution-ablation baseline: uniform random acquisition — the
    selector with its selection contribution removed (catalog CT-1, "AL vs
    random acquisition"). This is the ONE concrete ablation built now;
    coefficient-zeroing for a declared scoring term is the named follow-on
    (it needs the paradigm-pack contribution_ablation field plus a second
    concrete case, per the two-concrete-cases discipline)."""
    return set(
        np.random.default_rng(seed)
        .choice(pool_size, batch_size, replace=False).tolist())


def check_discriminating_power_al(
    package: ModuleType,
    pluggable_name: str,
    invariant,
    *,
    seed: int = 0,
    trials: int = 3,
) -> ProbeVerdict:
    """CT-5 (AL arm): does a behavioral invariant actually discriminate the
    paper's contribution from its absence? The invariant earns verification
    weight only if it HOLDS for the real selector AND FAILS for the
    contribution-ablated baseline (uniform random acquisition) on the same
    fixture. The step-2 verified gate composes this with the invariant's own
    checker and CT-1 passing.

    ``invariant(selection: list[int], kit: dict) -> bool`` receives the
    selection as a SORTED list (so "no duplicates" stays checkable while the real
    and ablated sides are presented in the same order — an order-sensitive
    invariant would otherwise spuriously discriminate) plus the selector kit
    (``pool_size`` / ``batch_size`` / ``seed``) so it can compute its own
    reference. Invariants must be set-semantic (about which points were selected,
    count, uniqueness, membership), not position-sensitive, and are trusted gate
    code — an exception inside an invariant propagates rather than being
    laundered into a verdict.

    Conservative bar: discriminating requires the ablated baseline to fail on
    EVERY trial, so a baseline that satisfies the invariant even once reads as
    non-discriminating. The ablation baseline is deterministic in the kit's seed,
    so the per-trial robustness is on the real-selector side (its fixture varies
    by trial); an independent per-trial ablation draw is a follow-on for the
    step-2 verified gate that consumes this.

    Verdicts (tier ``claims``):
    - ``pass``: holds on the real selector, fails on the ablated baseline on
      every trial (discriminating).
    - ``fail``: the ablated baseline satisfies it on some trial — the invariant
      verifies nothing (non-discriminating).
    - ``fail``: does not hold on the real selector — the claimed invariant is
      false here, so there is nothing to verify.
    - ``unprobeable``: the kit or selector could not run (named reason, never a
      silent pass).
    """
    real_holds: list[bool] = []
    ablated_holds: list[bool] = []
    for t in range(trials):
        kit = al_selector_kit(package, pluggable_name, seed,
                              fixture_seed=seed + t)
        if isinstance(kit, str):
            return ProbeVerdict("CT-5", "unprobeable", kit, tier="claims")
        try:
            real_sel = sorted(int(i) for i in kit["invoke"]())
        except Exception as e:  # noqa: BLE001 — generated selectors fail freely
            origin, site = attribute_invocation_crash(e, kit["package_dir"])
            where = (f" inside the method package at {site}"
                     if origin == "subject" else "")
            return ProbeVerdict(
                "CT-5", "unprobeable",
                f"{pluggable_name} crashed{where} ({type(e).__name__}: {e}) — "
                f"no discriminating-power comparison possible",
                tier="claims")
        ablated_sel = sorted(random_acquisition(
            kit["pool_size"], kit["batch_size"], kit["seed"]))
        # Both sides sorted so a position-sensitive invariant cannot fake
        # discrimination off the order asymmetry alone.
        real_holds.append(bool(invariant(real_sel, kit)))
        ablated_holds.append(bool(invariant(ablated_sel, kit)))

    if not all(real_holds):
        return ProbeVerdict(
            "CT-5", "fail",
            f"the claimed invariant does not even hold for the real "
            f"{pluggable_name} across {trials} trials — there is nothing to "
            f"verify",
            tier="claims")
    if any(ablated_holds):
        return ProbeVerdict(
            "CT-5", "fail",
            f"invariant is NON-discriminating: the contribution-ablated "
            f"baseline (uniform random acquisition) satisfies it on at least "
            f"one of {trials} trials too, so holding it verifies nothing about "
            f"the paper's contribution",
            tier="claims")
    return ProbeVerdict(
        "CT-5", "pass",
        f"invariant is discriminating: it holds for the real {pluggable_name} "
        f"and FAILS for the contribution-ablated baseline on every one of "
        f"{trials} trials, so the check is meaningful",
        tier="claims")


# ---------------------------------------------------------------------------
# CT-3 — the claims ledger (v1 extractor, per the 2026-06-10 survey)
# ---------------------------------------------------------------------------

_DIRECTION_RE = re.compile(
    r"\b(outperform\w*|exceed\w*|improv\w*|higher|faster|better than|"
    r"superior|surpass\w*|reduc\w*|lower(?:s|ed)? than)\b",
    re.IGNORECASE)
_NUMBER_RE = re.compile(r"(?<![\w.])(\d+(?:\.\d+)?%?)(?![\w.])")
_METRIC_WORD = (
    r"accuracy|accuracies|acc|error|loss|rmse|mse|mae|auc|f1(?:[-_ ]score)?|"
    r"score"
)
_PAPER_NUM_BEFORE_METRIC_RE = re.compile(
    rf"(?P<num>(?<![\w.])\d+(?:\.\d+)?%?(?![\w.]))\s+"
    rf"(?P<metric>{_METRIC_WORD})\b",
    re.IGNORECASE)
_PAPER_METRIC_BEFORE_NUM_RE = re.compile(
    rf"\b(?P<metric>{_METRIC_WORD})\b\s*"
    rf"(?:=|:|of|at|to|is|was|were|exceeds?|reaches?|reaching|achieves?|"
    rf"achieving|above|below)\s*"
    rf"(?P<num>(?<![\w.])\d+(?:\.\d+)?%?(?![\w.]))",
    re.IGNORECASE)
_EXECUTED_METRIC_RE = re.compile(
    rf"\b(?P<label>(?:test|validation|val|train|training|final|initial)?"
    rf"[\s_]*(?:acc(?:uracy)?|accuracy)|(?:test|validation|val|train|"
    rf"training|final|initial)?[\s_]*(?:loss|error|rmse|mse|mae)|auc|"
    rf"f1(?:[-_ ]score)?|score)\b\s*[=:]\s*"
    rf"(?P<num>-?\d+(?:\.\d+)?%?)",
    re.IGNORECASE)
_RELATIVE_CLAIM_RE = re.compile(
    r"\b(improv\w*|gain\w*|increas\w*|decreas\w*|reduc\w*|drop\w*|boost\w*|"
    r"delta|difference|relative|over\s+(?:the\s+)?baseline|compared\s+to)\b",
    re.IGNORECASE)
_LISTED_NUMBER_RE = re.compile(
    r"\d+(?:\.\d+)?%?\s*(?:/|,|\band\b|\bor\b)\s*\d+(?:\.\d+)?%?",
    re.IGNORECASE)
_COUNT_NOUN_RE = re.compile(
    r"\b(cases?|terms?|samples?|examples?|classes?|trials?|runs?|rounds?|"
    r"epochs?|seeds?|labels?|points?|datasets?|breakpoints?|counts?)\b",
    re.IGNORECASE)


def _numeric_value(raw: str) -> float | None:
    """Parse a printed paper/executed number to a comparable float.

    Percentages normalize to fractions, so paper-side "99%" and executed-side
    "0.99" compare on the same scale. Malformed values fail closed to None.
    """
    token = str(raw or "").strip()
    is_percent = token.endswith("%")
    try:
        value = float(token.rstrip("%"))
    except ValueError:
        return None
    return value / 100.0 if is_percent else value


def _normalize_metric(label: str) -> str:
    text = str(label or "").lower().replace("_", " ")
    if "acc" in text or "accurac" in text:
        return "accuracy"
    if "loss" in text:
        return "loss"
    if "error" in text:
        return "error"
    if "rmse" in text:
        return "rmse"
    if "mse" in text:
        return "mse"
    if "mae" in text:
        return "mae"
    if "auc" in text:
        return "auc"
    if "f1" in text:
        return "f1"
    return "score"


def _metric_split(label: str) -> str | None:
    text = str(label or "").lower().replace("_", " ")
    if re.search(r"\b(test|testing)\b", text):
        return "test"
    if re.search(r"\b(val|validation)\b", text):
        return "validation"
    if re.search(r"\b(train|training)\b", text):
        return "train"
    return None


def _valid_metric_value(metric: str, raw: str, value: float | None) -> bool:
    """Guard against treating counters or budgets as comparable metrics."""
    if value is None:
        return False
    if metric in {"accuracy", "auc", "f1", "score"}:
        # Normalized metrics are expected on [0, 1], unless written as a
        # percentage. A bare "99 accuracy" is more likely a count/scale typo
        # than a comparable metric target.
        return 0.0 <= value <= 1.0
    return True


def _number_is_range_or_list(text: str, start: int, end: int) -> bool:
    before = text[start - 1] if start > 0 else ""
    after = text[end] if end < len(text) else ""
    return before in "/±+-" or after in "/±+-"


def _relative_delta_context(text: str, start: int) -> bool:
    window = text[max(0, start - 48):start + 96]
    return bool(
        re.search(r"\bby\s*$", text[max(0, start - 12):start].lower())
        or _RELATIVE_CLAIM_RE.search(window)
    )


def _listed_number_context(text: str, start: int, end: int) -> bool:
    window = text[max(0, start - 48):min(len(text), end + 48)]
    return bool(_LISTED_NUMBER_RE.search(window))


def _count_noun_after_metric(text: str, metric_end: int) -> bool:
    return bool(_COUNT_NOUN_RE.match(text[metric_end:].lstrip()))


def _paper_side_from_claim(row: dict) -> dict | None:
    """Conservatively extract ONE level metric from a lifted paper claim.

    The matcher intentionally ignores relative deltas ("improves accuracy by
    4%"), slash/range lists ("70%/80%/90% accuracy"), and rows with multiple
    plausible metric numbers. That leaves ambiguous headline prose untested
    rather than matching the wrong paper number to a smoke-run output.
    """
    text = str(row.get("claim_text") or "")
    candidates: dict[tuple[int, int, str], dict] = {}
    for pattern in (_PAPER_NUM_BEFORE_METRIC_RE, _PAPER_METRIC_BEFORE_NUM_RE):
        for match in pattern.finditer(text):
            raw = match.group("num")
            start, end = match.span("num")
            metric_start, metric_end = match.span("metric")
            metric = _normalize_metric(match.group("metric"))
            value = _numeric_value(raw)
            if _number_is_range_or_list(text, start, end):
                continue
            if _relative_delta_context(text, start):
                continue
            if _listed_number_context(text, start, end):
                continue
            if _count_noun_after_metric(text, metric_end):
                continue
            if not _valid_metric_value(metric, raw, value):
                continue
            candidates[(start, end, metric)] = {
                "value": value,
                "raw": raw,
                "metric": metric,
                "split": _metric_split(text[max(0, metric_start - 16):
                                            min(len(text), metric_end + 16)]),
                "source": "paper_map.claim_text",
                "scale": "paper",
            }
    if len(candidates) != 1:
        return None
    return next(iter(candidates.values()))


def _cell_output_texts(nb: dict) -> list[str]:
    """One concatenated text blob per executed code cell."""
    blobs: list[str] = []
    if not isinstance(nb, dict):
        return blobs
    for cell in nb.get("cells", []):
        if not isinstance(cell, dict) or cell.get("cell_type") != "code":
            continue
        parts: list[str] = []
        for out in cell.get("outputs", []) or []:
            if not isinstance(out, dict):
                continue
            if out.get("output_type") == "stream":
                text = out.get("text", [])
            else:
                text = out.get("data", {}).get("text/plain", [])
            if isinstance(text, str):
                parts.append(text)
            else:
                parts.append("".join(text or []))
        if parts:
            blobs.append("\n".join(parts))
    return blobs


def executed_metric_observations(nb: dict) -> list[dict]:
    """Extract comparable metric observations from executed notebook outputs.

    This is intentionally narrower than the universal sanity probe: it only
    returns metric-labeled values suitable for a paper/executed comparison, with
    counters such as labels/classes ignored by construction.
    """
    observations: list[dict] = []
    for blob in _cell_output_texts(nb):
        for line_no, line in enumerate(blob.splitlines()):
            for match in _EXECUTED_METRIC_RE.finditer(line):
                raw = match.group("num")
                label = match.group("label")
                metric = _normalize_metric(label)
                value = _numeric_value(raw)
                if not _valid_metric_value(metric, raw, value):
                    continue
                lower_line = line.lower()
                round_match = re.search(r"\bround\s+(\d+)", lower_line)
                if "final" in lower_line:
                    temporal = "final"
                elif "initial" in lower_line:
                    temporal = "initial"
                elif round_match:
                    temporal = "round"
                else:
                    temporal = "point"
                observations.append({
                    "value": value,
                    "raw": raw,
                    "metric": metric,
                    "split": _metric_split(label),
                    "label": " ".join(label.split()),
                    "source": "notebook.ipynb",
                    "scale": "executed_smoke",
                    "line": line.strip(),
                    "line_no": line_no,
                    "temporal": temporal,
                    "round": (int(round_match.group(1))
                              if round_match else None),
                    "index": len(observations),
                })
    return observations


def _match_executed_side(paper_side: dict | None,
                         observations: list[dict]) -> dict | None:
    if not paper_side:
        return None
    metric = paper_side.get("metric")
    matching = [o for o in observations if o.get("metric") == metric]
    paper_split = paper_side.get("split")
    if paper_split:
        matching = [o for o in matching if o.get("split") == paper_split]
    if not matching:
        return None
    split_rank = {"test": 0, "validation": 1, None: 2, "train": 3}

    def _rank(obs: dict) -> tuple:
        return (split_rank.get(obs.get("split"), 2), obs.get("index", 0))

    finals = [o for o in matching if o.get("temporal") == "final"]
    if finals:
        chosen = dict(sorted(finals, key=_rank)[0])
        chosen["aggregation"] = "final"
        chosen["n_matched_observations"] = len(matching)
        return chosen
    if len(matching) == 1:
        chosen = dict(matching[0])
        chosen["aggregation"] = "single"
        chosen["n_matched_observations"] = 1
        return chosen
    rounds = [o for o in matching if o.get("temporal") == "round"]
    if rounds:
        numeric_rounds = [o for o in rounds if o.get("round") is not None]
        if numeric_rounds:
            latest_round = max(o.get("round") for o in numeric_rounds)
            rounds = [o for o in numeric_rounds if o.get("round") == latest_round]
        chosen = dict(sorted(rounds, key=_rank)[0])
        chosen["aggregation"] = "last_observed_round"
        chosen["n_matched_observations"] = len(matching)
        return chosen
    return None


def build_claims_ledger(paper_map: dict) -> list[dict]:
    """One ledger row per `experiment` element (the claims source per the
    three-paper survey: comparative claims dominate, bare numbers mostly
    live in hyperparameter elements the provenance probes already guard).

    Every row gets an explicit status. v1 defaults everything to
    `untested_at_this_scale` — the honest visible gap is the product here;
    `verified_at_scale` / `directionally_checked` / `contradicted` start
    flowing when the executed-notebook comparison lands with REPORT.md
    (slice 2.2)."""
    rows = []
    for element in paper_map.get("elements", []):
        if element.get("type") != "experiment":
            continue
        text = str(element.get("source_text", "") or
                   element.get("description", "")).strip()
        direction = _DIRECTION_RE.search(text)
        rows.append({
            "claim_id": element.get("id", "?"),
            "name": element.get("name", ""),
            "claim_text": text[:400],
            "section": element.get("section", ""),
            "comparative": bool(direction),
            "direction_phrase": direction.group(0) if direction else None,
            "numbers": _NUMBER_RE.findall(text)[:12],
            "status": "untested_at_this_scale",
            "status_note": (
                "Smoke-scale execution cannot test this claim; full-scale "
                "data/compute required. The gap is recorded, not silent."),
        })
    return rows


# ---------------------------------------------------------------------------
# CT-3 step 2 — scale classification, synthesized behavioral rows, and the
# verified-at-scale path of the decision tree (§2, §3 S0–S2, §4 schema).
#
# Two row kinds. LIFTED experiment rows (the v1 output) are scale-bound by
# construction and never carry a prose-parsed scale-free predicate, so they
# stay untested. The only place `verified_at_scale` can appear is a separate
# SYNTHESIZED behavioral row, one per scale-free predicate the run's probes
# actually evaluated — and only when that predicate's check is DISCRIMINATING
# (it fails for the contribution-removed baseline), so a property a trivial
# baseline also satisfies never earns a verified badge. Branch F
# (suspect_our_implementation) and the contradiction gate are later steps; a
# non-clean checker lands at untested here, and the existing probe-fail path
# still demotes delivery unchanged.
# ---------------------------------------------------------------------------

VERIFIED_AT_SCALE = "verified_at_scale"
UNTESTED_AT_THIS_SCALE = "untested_at_this_scale"
SUSPECT_OUR_IMPLEMENTATION = "suspect_our_implementation"
CONTRADICTED = "contradicted"

# Branch F (§3.2) — fidelity attribution. A behavioral row can only stay
# verified if the path it was checked on is fidelity-clean; an unclean path
# demotes a would-be-verified row to suspect_our_implementation (never a paper
# verdict), and a failed/flagged behavioral check is itself a suspect signal.
#
# Path-fidelity signals, in deterministic cause-priority order (lower number
# wins). Per the locked recommendation the gate is scoped to the algorithm-
# behavior probes plus implementing-code findings; provenance and other static
# label checks demote delivery on their own paths but do not demote a
# behavioral verified row (a mislabeled param source does not make "the selector
# picks differently from random" false). For a single-pluggable smoke run these
# probes are all on the one behavioral path; fine-grained per-element matching
# is the deferred follow-on for multi-path papers.
_SIGNAL_SPECS = (
    (1, "US-4", "fail", "scale_dependent_param_unrescaled",
     "a scale-dependent parameter on this path was shipped unrescaled for the "
     "data, so the term degenerates at runtime (the R_0 class)"),
    (2, "UB-6", "fail", "executed_output_degenerate",
     "the executed output is degenerate here (a chance-level, flat, or diverged "
     "metric series)"),
    (3, "UB-7", "flag_for_researcher", "claimed_mechanism_inert",
     "a claimed scoring term is inert on this run (it cannot change the "
     "selection)"),
    (4, "CT-1", "fail", "contribution_collapsed_to_null",
     "the method behaves identically to its degenerate null baseline"),
)
_FINDING_PRIORITY = 5
_OWN_CHECK_PRIORITY = 6

# The one cause key whose wording deliberately REFUSES to attribute: the
# contribution did not change behavior, and we cannot tell our wiring from the
# method being inert as published (locked §10.6). Public because the report
# renderer keys the Verdict cell on it — a row that will not attribute its
# cause must not render as "Needs a fix on our side" (R2C-047).
UNATTRIBUTED_CAUSE = "method_appears_inert_as_published"

# What the unattributed cause becomes once the run's own contract has settled
# the question: the mechanism is required exactly, so an inert one is ours.
CONTRACT_REQUIRES_EXACTLY_CAUSE = "contract_requires_mechanism_exactly"
_CONTRACT_SETTLED_WORDING = (
    "the contribution did not change behavior on this run, and the contract "
    "leaves no approximation that would explain that away")
# Inertness causes get the method-intrinsic split: when inertness is the ONLY
# signal (an otherwise-faithful path), we do not launder it to "our bug" — it
# may be our wiring OR the published method as written. Shipped neutral now
# (locked decision §10.6), routed to a human, still demoting.
_INERTNESS_CAUSES = frozenset({"claimed_mechanism_inert",
                               "contribution_collapsed_to_null"})
_GATING_FINDING_SEVERITIES = ("critical", "important")
# Implementing code: a reviewer finding bears on the selector's behavioral path
# only when it sits on the algorithm implementation, never on packaging
# (requirements.txt), data loading, docs, or the notebook. This is what keeps
# BADGE's requirements finding from wrongly demoting its verified row.
_IMPLEMENTING_CODE = ("method.py", "model.py", "training.py")

# Contradiction gate (§3.3) — the seven-clause hard conjunction, the ONE place
# CT-3 may point at the paper. Conservative and fail-closed: a clause with no
# supporting evidence FAILS (absence of evidence is failure). Today every clause
# that could promote a row is inert — run_mode is "smoke" (the full-scale clause
# stays shut for level claims), the smoke run is single-seed (the reproduction
# clause fails), and the model veto is unavailable (fail-closed). So
# `contradicted` is structurally unreachable in the current pipeline and the gate
# only enriches the verdict trace; every failing row still lands at suspect via
# branch F. The machinery activates without re-plumbing once a full-scale,
# multi-seed runner sets run_mode/seed_set (the deferred compute story, §10.4).
_FULL_RUN_MODE = "full"
_DEFAULT_RUN_MODE = "smoke"
_G5_MIN_SEEDS = 3                     # locked default (§10.3); never silently lowered
_DEFAULT_CONTRADICTION_MARGIN = 0.10  # G6 direction-and-margin, wide default (§10.2)

# Which gate clause each fidelity-not-clean signal fails. The own-bug clauses
# (G2 output sane, G4 scale-params clean) route a failure to branch F; any other
# signal (an open finding, a failed own check, a generic not-clean) fails G3
# (the path is not provably clean) and also routes to branch F. Only when NO
# signal is present (a provably clean path) can the scale/seed/margin/veto
# clauses be the block, which routes to untested.
_G2_CAUSES = frozenset({"executed_output_degenerate",
                        "contribution_collapsed_to_null",
                        "claimed_mechanism_inert",
                        "method_appears_inert_as_published"})
_G4_CAUSES = frozenset({"scale_dependent_param_unrescaled"})


def _default_seed_set() -> dict:
    """A single, unreproduced seed — the honest degenerate state of the
    single-seed smoke run, so the reproduction clause (G5) fails closed until a
    multi-seed runner records a real seed set (§10.3 bars dropping below k=3
    silently, so the default can never itself satisfy the clause)."""
    return {"k": 1, "reproduced": False}

# Named full datasets — a dataset name anchors a claim to a specific scale (the
# gazetteer the design names). Lowercased substring match on the claim text.
_DATASET_GAZETTEER = (
    "mnist", "fashion-mnist", "fashionmnist", "cifar", "cifar-10", "cifar-100",
    "imagenet", "tiny-imagenet", "svhn", "coco", "kitti", "celeba", "stl-10",
    "stl10", "places", "pascal", "voc", "ade20k", "cityscapes", "nuscenes",
)


def detect_level_claim(row: dict) -> bool:
    """True when a lifted row carries an absolute, scale-anchored target: a
    named full dataset, or any extracted number (a percentage, an error value,
    an epoch/seed/round budget — a number attached to an experiment claim is
    almost always a level or budget anchor). Conservative by design: a level
    claim can never earn a whole-row verified, so leaning toward true is the
    fail-closed direction. Behavioral rows set level_claim=False explicitly."""
    text = str(row.get("claim_text", "")).lower()
    if any(ds in text for ds in _DATASET_GAZETTEER):
        return True
    return bool(row.get("numbers"))


@dataclass(frozen=True)
class _Predicate:
    """A scale-free behavioral predicate on the paradigm-keyed allowlist.

    `discriminating` marks a check whose very form is an ablation contrast (it
    fails for the contribution-removed baseline), so a clean pass earns
    verified. The non-discriminating predicates (a trivial baseline also passes
    them) can never earn verified and land untested with a plain reason; their
    honest ablation is a named follow-on per the two-concrete-cases discipline.
    """

    claim_id: str
    name: str
    probe_ids: tuple
    discriminating: bool
    what_would_verify: str


# Each predicate is backed by a probe that runs in the battery today; a
# predicate whose probe returns unprobeable or not-applicable (or never ran)
# gets no row, so the ledger only ever claims to have checked what was actually
# evaluated (§2.1).
_ALLOWLIST: dict[str, tuple[_Predicate, ...]] = {
    "active_learning": (
        _Predicate(
            "behavioral:differs-from-null",
            "the method selects differently from random acquisition",
            ("CT-1",), True,
            "an ablation showing the selection collapses to the baseline "
            "without the contribution"),
        _Predicate(
            "behavioral:scoring-terms-live",
            "every claimed scoring term can change the selection",
            ("UB-7",), True,
            "a term-ablation showing each scoring term moves the result"),
        _Predicate(
            "behavioral:no-duplicate-selections",
            "no point is acquired twice across rounds",
            ("AL-1",), False,
            "an ablation that re-selects, since uniqueness alone is met by "
            "random acquisition too"),
        _Predicate(
            "behavioral:labeled-set-grows",
            "the labeled set grows by the batch size each round",
            ("AL-1",), False,
            "a baseline loop also grows the labeled set, so growth alone is "
            "not evidence of the contribution"),
        _Predicate(
            "behavioral:seeds-thread",
            "the same seed reproduces the same selection",
            ("UB-8",), False,
            "determinism holds for any seeded baseline, so it is not evidence "
            "of the contribution"),
        _Predicate(
            "behavioral:loss-descends",
            "the training loss goes down",
            ("UB-9",), False,
            "a contribution-ablated run whose loss does NOT descend (the "
            "coefficient-zeroing ablation, a named follow-on)"),
    ),
    "motion_planning": (
        _Predicate(
            "behavioral:obstacles-move",
            "obstacles actually move in a scenario the paper calls dynamic",
            ("MP-1",), False,
            "a check that the planner's contribution, not the scenario setup, "
            "drives the behavior"),
        _Predicate(
            "behavioral:steers-to-goal",
            "steering consistently turns toward the goal",
            ("MP-3",), False,
            "a baseline planner also steers to the goal on an open map"),
        _Predicate(
            "behavioral:goal-progress",
            "the robot ends closer to the goal than it started",
            ("MP-4",), False,
            "a baseline planner also makes goal progress on an open map"),
    ),
    "knowledge_distillation": (
        _Predicate(
            "behavioral:teacher-signal",
            "the teacher's signal influences the distillation loss",
            ("KD-1",), True,
            "an ablation turning the teacher term off and showing the loss "
            "stops responding"),
    ),
}


def _allowlist_for(paradigm: str) -> tuple[_Predicate, ...]:
    for key, preds in _ALLOWLIST.items():
        if paradigm.startswith(key):
            return preds
    return ()


def _aggregate_status(matching: list[dict]) -> str:
    """Reduce a predicate's backing probe verdicts to one of
    pass / fail / flag / unprobeable / not_applicable / none
    (none == the probe never ran).

    Precedence fail > flag > pass: any failing or flagged sub-verdict blocks a
    verified row even when another sub-verdict passes (so two UB-7 verdicts, one
    passing and one flagging a dead term, never verify). `warn` is advisory and
    non-blocking by catalog rule, and the verdict vocabulary is closed and
    enforced at ProbeVerdict construction, so the final fall-through (warn-only
    or any blocked/unknown outcome -> "unprobeable" -> no row) is the safe
    fail-closed default: an unrecognized verdict can never manufacture a pass.
    An all-``not_applicable`` result remains distinct from missing evidence and
    likewise produces no claim row."""
    if not matching:
        return "none"
    verdicts = [v.get("verdict") for v in matching]
    if "fail" in verdicts:
        return "fail"
    if "flag_for_researcher" in verdicts:
        return "flag"
    if "pass" in verdicts:
        return "pass"
    if verdicts and all(v == "not_applicable" for v in verdicts):
        return "not_applicable"
    return "unprobeable"


def _aggregate_spec_branch(matching: list[dict]) -> str:
    """The strongest contract adjudication across a predicate's backing verdicts
    (R2C-047).

    Precedence contradicts > consistent > silent, mirroring `_aggregate_status`:
    one bound must-replicate element decides the row even when a second verdict
    on the same predicate bound to an approximable one. There is no honest way
    to average the two, and the strict reading is the one that routes a real
    defect to a fix.

    A verdict that never went through the battery's conditioning pass carries no
    stamp, and the empty stamp reads as silent, which is today's behavior.
    """
    branches = {str(v.get("spec_branch") or "") for v in matching}
    if CONTRACT_CONTRADICTS in branches:
        return CONTRACT_CONTRADICTS
    if CONTRACT_CONSISTENT in branches:
        return CONTRACT_CONSISTENT
    return CONTRACT_SILENT


# ---------------------------------------------------------------------------
# Branch F (§3.2): fidelity-cleanliness gate + deterministic cause attribution.
# ---------------------------------------------------------------------------


def _is_implementing_code(file_path: str) -> bool:
    # Case-insensitive basename match: a finding recorded as "Method.py" on a
    # case-insensitive filesystem must still demote (the red-team's miss).
    base = str(file_path or "").replace("\\", "/").rsplit("/", 1)[-1].lower()
    return base in _IMPLEMENTING_CODE


def path_fidelity_signals(verdicts: list[dict],
                          findings: list[dict] | None = None) -> list[dict]:
    """The fidelity-not-clean signals on the (single-pluggable) behavioral path:
    a scale-mismatch fail, a degenerate executed output, a dead scoring term, a
    contribution collapsed to the null, and unresolved critical/important
    findings on implementing code. Each is a dict {priority, cause_key, wording,
    linked} the cause picker ranks. Empty list == the path is provably clean."""
    signals: list[dict] = []
    for priority, probe_id, want, cause_key, wording in _SIGNAL_SPECS:
        for v in verdicts:
            if v.get("probe_id") == probe_id and v.get("verdict") == want:
                signals.append({"priority": priority, "cause_key": cause_key,
                                "wording": wording, "linked": probe_id})
                break
    for f in findings or []:
        if (str(f.get("severity") or "").lower() in _GATING_FINDING_SEVERITIES
                and str(f.get("resolution_status") or "pending") != "applied"
                and (_is_implementing_code(f.get("file", ""))
                     or f.get("related_elements"))):
            signals.append({
                "priority": _FINDING_PRIORITY,
                "cause_key": "open_fidelity_finding",
                "wording": (f"an unresolved {f.get('severity')} fidelity finding "
                            f"on this path ({f.get('file', '?')})"),
                "linked": f.get("id")})
    return signals


def pick_cause(signals: list[dict]) -> dict:
    """The deterministic suspected cause for a suspect row: the highest-priority
    fidelity signal. When inertness is the ONLY kind of signal (an otherwise-
    faithful path), apply the method-intrinsic split — neutral wording that does
    not auto-blame our wiring, routed to a human (locked §10.6)."""
    if not signals:
        return {"cause_key": "fidelity_not_provably_clean",
                "wording": "this path is not provably clean (a check on it did "
                           "not pass, or a path element went unchecked)",
                "linked": None}
    top = min(signals, key=lambda s: s["priority"])
    only_inertness = all(s["cause_key"] in _INERTNESS_CAUSES for s in signals)
    if top["cause_key"] in _INERTNESS_CAUSES and only_inertness:
        return {"cause_key": UNATTRIBUTED_CAUSE,
                "wording": "the contribution did not change behavior here. This "
                           "may be our wiring, or the method as published may be "
                           "inert under these conditions — a question for your "
                           "judgment, not a problem we can attribute to the "
                           "paper",
                "linked": top["linked"]}
    return {"cause_key": top["cause_key"], "wording": top["wording"],
            "linked": top["linked"]}


def _own_check_signal(pred: "_Predicate", status: str,
                      matching: list[dict]) -> dict | None:
    """A behavioral row whose OWN check failed/flagged is itself a suspect
    signal. Registry probes (CT-1/UB-7) are already covered by
    path_fidelity_signals; this adds a generic, lower-priority signal for the
    other backing probes (a failed loop-invariant or descent check) so the row
    is still attributed rather than silently untested."""
    if status not in ("fail", "flag"):
        return None
    linked = matching[0].get("probe_id") if matching else "/".join(pred.probe_ids)
    return {"priority": _OWN_CHECK_PRIORITY,
            "cause_key": "behavioral_check_failed",
            "wording": (f"the underlying check for “{pred.name}” did "
                        f"not pass on our implementation ({status})"),
            "linked": linked}


# ---------------------------------------------------------------------------
# The contradiction gate (§3.3): S3's seven-clause hard conjunction.
# ---------------------------------------------------------------------------


def _direction_disagrees(paper_side, executed_side, margin: float) -> bool:
    """G6 helper: the executed result disagrees with the paper by more than the
    margin. A within-margin shortfall is never a contradiction (§10.2). Defensive
    — an unparseable pair fails closed (treated as no disagreement). This is only
    ever reached on the matched-number path (executed-side matching, the deferred
    §2.3 slice); behavioral invariant rows carry no numbers and hold G6 vacuously."""
    def _val(side):
        return side.get("value") if isinstance(side, dict) else side
    try:
        p, e = float(_val(paper_side)), float(_val(executed_side))
    except (TypeError, ValueError):
        return False
    return abs(p - e) > margin


def _g7_cross_examine(row: dict, signals: list[dict]) -> tuple[bool, str]:
    """G7, the model cross-examination veto (§3.3, §6): a final review that may
    only VETO a contradiction, never create one. It is unavailable in the
    deterministic battery, so it FAILS CLOSED (demote). The real call is deferred
    and coupled to the run_mode="full" slot (§10.4): G7 can only ever fire on a
    row that already passed the full-scale and multi-seed clauses, which is
    impossible today, so this stub never gates a live verdict — it records the
    humility default."""
    return (False, "model cross-examination unavailable — failing closed (the "
            "veto can demote a contradiction, never manufacture one)")


def contradiction_gate(row: dict, signals: list[dict], *,
                       seed_set: dict | None, run_mode: str,
                       margin: float = _DEFAULT_CONTRADICTION_MARGIN) -> dict:
    """The §3.3 seven-clause hard conjunction — the ONE path to `contradicted`.

    Every clause must hold; a clause with no supporting evidence fails. The
    own-bug clauses (G2 output sane / G3 fidelity clean / G4 scale-params clean)
    are the GBALD firewall: any fidelity-not-clean signal on the path fails them
    and routes the row to branch F (suspect), never the paper. The remaining
    clauses (G1 scale matched, G5 reproduces across k>=3 seeds, G6 direction and
    margin, G7 model veto) fail closed today, so `contradicted` is structurally
    unreachable in the current pipeline.

    Returns the per-clause verdicts, whether all held, the first blocking clause
    (`blocked_by`, the design's "exactly what kept a row out of contradicted"),
    and `own_bug_block` — whether an own-bug clause did the blocking, which is
    what decides branch F vs untested in S3."""
    causes = {s["cause_key"] for s in signals}
    clauses: dict[str, tuple[bool, str]] = {}

    # G1 scale matched, operationally. A scale-free invariant violation is
    # scale-free by construction (G1a). A level/number claim needs an explicit
    # full-scale run (G1b), never inferred — inert until run_mode is plumbed.
    if row.get("level_claim"):
        g1 = run_mode == _FULL_RUN_MODE
        clauses["G1"] = (g1, "level claim tested at the paper's scale "
                         "(run_mode=full)" if g1 else
                         "level claim with no run_mode=full assertion — headline "
                         "numbers cannot be contradicted at smoke scale")
    else:
        clauses["G1"] = (True, "scale-free invariant, violated in a scale-free way")

    # G2 executed output sane: no degenerate / collapsed / inert signal.
    g2 = not (causes & _G2_CAUSES)
    clauses["G2"] = (g2, "executed output sane" if g2 else
                     "executed output is degenerate, collapsed, or inert — a "
                     "self-evident own-bug signal")

    # G4 scale-dependent params clean: no US-4 fail on the path.
    g4 = not (causes & _G4_CAUSES)
    clauses["G4"] = (g4, "scale-dependent params clean" if g4 else
                     "a scale-dependent parameter was shipped unrescaled (the "
                     "R_0 class)")

    # G3 fidelity provably clean: NO fidelity-not-clean signal at all (§3.2).
    # Absence of any signal is the clean state; any signal — an open finding, a
    # failed own check, an unchecked element — fails it.
    g3 = not signals
    clauses["G3"] = (g3, "path provably clean" if g3 else
                     "path not provably clean (a check on it did not pass, or a "
                     "path element went unchecked)")

    # G5 reproduces across seeds: k>=3 and the disagreement reproduced.
    k = int((seed_set or {}).get("k", 1) or 1)
    reproduced = bool((seed_set or {}).get("reproduced", False))
    g5 = k >= _G5_MIN_SEEDS and reproduced
    clauses["G5"] = (g5, f"reproduced across {k} seeds" if g5 else
                     f"only {k} seed(s) (need >= {_G5_MIN_SEEDS} reproducing) — a "
                     "single-seed wobble is not a verdict")

    # G6 direction and margin: only bites a matched level comparison. With no
    # executed-side number to compare (today's reality), it is vacuously held.
    paper_side, executed_side = row.get("paper_side"), row.get("executed_side")
    if paper_side is None or executed_side is None:
        clauses["G6"] = (True, "no matched number to compare (a behavioral "
                         "invariant); the margin clause is not applicable")
    else:
        g6 = _direction_disagrees(paper_side, executed_side, margin)
        clauses["G6"] = (g6, f"disagrees by more than the {margin} margin" if g6
                         else "within the margin — a magnitude shortfall is never "
                         "contradicted")

    # G7 model cross-examination (veto only), fail-closed when unavailable.
    clauses["G7"] = _g7_cross_examine(row, signals)

    held = all(ok for ok, _ in clauses.values())
    own_bug_block = not (clauses["G2"][0] and clauses["G3"][0]
                         and clauses["G4"][0])
    blocked_by = None
    if not held:
        # Own-bug clauses first, so the trace names the strongest (most
        # attributable) block before the scale/seed/margin/veto ones.
        for key in ("G2", "G4", "G3", "G1", "G5", "G6", "G7"):
            if not clauses[key][0]:
                blocked_by = f"{key}:{clauses[key][1]}"
                break
    summary = ("contradiction gate held — all seven clauses pass" if held else
               f"contradiction gate blocked at {blocked_by}")
    return {
        "held": held,
        "own_bug_block": own_bug_block,
        "blocked_by": blocked_by,
        "summary": summary,
        "clauses": {key: {"held": ok, "reason": why}
                    for key, (ok, why) in clauses.items()},
    }


def _blank_row() -> dict:
    """A ledger row with every v1 + §4 field defaulted, so lifted and
    behavioral rows share one schema the REPORT can render uniformly."""
    return {
        # v1 fields (preserved — the committed-map test reads these)
        "claim_id": "", "name": "", "claim_text": "", "section": "",
        "comparative": False, "direction_phrase": None, "numbers": [],
        "status": UNTESTED_AT_THIS_SCALE, "status_note": "",
        # §4 additions
        "kind": "lifted", "scale_free": False, "level_claim": False,
        "verifiable_aspect": None, "depends_on": [],
        "suspected_cause": None, "linked_finding_id": None,
        "paper_side": None, "executed_side": None, "seed_set": None,
        "verdict_trace": {}, "reasoning": "", "what_would_verify": "",
        "researcher_override": None,
        # R2C-047: which way the run's own methodology contract adjudicated the
        # backing verdicts. Lifted rows keep the silent default because they are
        # built from paper numbers rather than from probe verdicts.
        "spec_branch": CONTRACT_SILENT,
    }


def _format_side(side: dict | None) -> str:
    if not isinstance(side, dict):
        return "no value"
    raw = side.get("raw")
    metric = side.get("metric") or "metric"
    return f"{metric} {raw}"


def _mark_lifted_contradicted(row: dict, trace: dict,
                              seed_set: dict | None) -> None:
    """A matched level claim cleared every contradiction-gate clause.

    This is unreachable in the smoke pipeline because G1/G5/G7 fail closed, but
    keeping the branch wired proves the executed-side matcher feeds the same
    gate as behavioral failures once a real full-scale, multi-seed runner exists.
    """
    k = int((seed_set or {}).get("k", _G5_MIN_SEEDS) or _G5_MIN_SEEDS)
    row["status"] = CONTRADICTED
    row["seed_set"] = dict(seed_set) if seed_set else _default_seed_set()
    trace["result"] = CONTRADICTED
    trace["blocked_by"] = None
    row["reasoning"] = (
        f"At the paper's own scale, the matched executed value "
        f"({_format_side(row.get('executed_side'))}) disagrees with the paper "
        f"value ({_format_side(row.get('paper_side'))}) beyond the contradiction "
        f"margin, and the implementation, scale, data, variance, and review "
        f"checks did not find another explanation. This warrants careful human "
        f"review before drawing conclusions about the paper.")
    row["what_would_verify"] = (
        "an independent full-scale, multi-seed re-run confirming the same "
        "matched-number disagreement")
    row["status_note"] = row["reasoning"]


def _enrich_lifted_row(v1_row: dict, executed_metrics: list[dict] | None = None,
                       path_signals: list[dict] | None = None,
                       run_mode: str = _DEFAULT_RUN_MODE,
                       seed_set: dict | None = None) -> dict:
    """A v1 lifted experiment row, classified and run through S0. It is
    scale-bound by construction (we never infer scale-freeness from prose). When
    the executed notebook exposes one unambiguous matching metric, the row records
    both sides and evaluates the contradiction gate for traceability, but the live
    smoke run still lands untested because G1/G5/G7 fail closed."""
    row = _blank_row()
    row.update(v1_row)
    row["kind"] = "lifted"
    row["scale_free"] = False
    row["level_claim"] = detect_level_claim(v1_row)
    row["paper_side"] = _paper_side_from_claim(row)
    row["executed_side"] = _match_executed_side(
        row["paper_side"], executed_metrics or [])
    row["status"] = UNTESTED_AT_THIS_SCALE
    trace = {
        "S1": "scale-bound (a lifted experiment claim, never prose-inferred "
              "as scale-free)",
    }
    if row["paper_side"] is None:
        trace["S0"] = "no unambiguous paper-side metric number for this claim"
        trace["result"] = UNTESTED_AT_THIS_SCALE
        trace["blocked_by"] = "S0:no-paper-side-number"
        row["reasoning"] = (
            "We did not verify this claim. It is a scale-bound experiment claim, "
            "and the paper-map text does not contain one unambiguous metric value "
            "that can be matched to the executed notebook.")
    elif row["executed_side"] is None:
        trace["S0"] = (
            f"paper-side {_format_side(row['paper_side'])} found, but no "
            "matching executed notebook metric was found")
        trace["result"] = UNTESTED_AT_THIS_SCALE
        trace["blocked_by"] = "S0:no-matched-executed-observation"
        row["reasoning"] = (
            f"We did not verify this claim. The paper-map text names "
            f"{_format_side(row['paper_side'])}, but the executed notebook did "
            f"not expose one matching metric value for this claim.")
    else:
        trace["S0"] = (
            f"matched paper-side {_format_side(row['paper_side'])} to executed "
            f"{_format_side(row['executed_side'])}")
        gate = contradiction_gate(row, path_signals or [],
                                  seed_set=seed_set or _default_seed_set(),
                                  run_mode=run_mode)
        trace["S3"] = gate["summary"]
        trace["gate"] = gate["clauses"]
        if not gate["held"]:
            trace["blocked_by"] = gate["blocked_by"]
        if gate["held"]:
            _mark_lifted_contradicted(row, trace, seed_set)
        else:
            trace["result"] = UNTESTED_AT_THIS_SCALE
            run_label = "smoke-run" if run_mode == _DEFAULT_RUN_MODE else run_mode
            if run_mode == _DEFAULT_RUN_MODE:
                row["reasoning"] = (
                    f"We found a {run_label} executed value "
                    f"({_format_side(row['executed_side'])}) for the paper-side "
                    f"value ({_format_side(row['paper_side'])}), but this remains "
                    f"a scale-bound claim. A smoke run does not test the paper's "
                    f"full-scale result, so this is recorded as an untested "
                    f"matched comparison rather than a verdict about the paper.")
            else:
                row["reasoning"] = (
                    f"We found an executed value "
                    f"({_format_side(row['executed_side'])}) for the paper-side "
                    f"value ({_format_side(row['paper_side'])}), but the "
                    f"contradiction gate did not clear. This stays untested "
                    f"rather than becoming a verdict about the paper.")
    if row["status"] != CONTRADICTED:
        row["what_would_verify"] = (
            "a full-scale run on the paper's dataset, training budget, and seed "
            "count")
    row["verdict_trace"] = trace
    row["status_note"] = row["reasoning"]
    return row


def _mark_suspect(row: dict, trace: dict, pred: "_Predicate",
                  signals: list[dict], contract_contradicts: bool = False) -> None:
    """Branch F: set suspect_our_implementation with a deterministic cause, a
    plain reason that never blames the paper, and a link to the demoting signal
    (we link, we do not create a new finding — delivery already demotes through
    the existing probe/review paths, so the row is a second view of one fact).

    `contract_contradicts` (R2C-047) is the run's own methodology contract
    marking the bound mechanism must-replicate with no approved approximation.
    It settles the one question the unattributed wording leaves open: the
    contract says the mechanism has to be there exactly, so an inert mechanism
    is ours to fix and the row stops hedging about the paper.
    """
    cause = pick_cause(signals)
    row["status"] = SUSPECT_OUR_IMPLEMENTATION
    hedged = cause["cause_key"] == UNATTRIBUTED_CAUSE
    unattributed = hedged and not contract_contradicts
    row["suspected_cause"] = (CONTRACT_REQUIRES_EXACTLY_CAUSE
                              if hedged and contract_contradicts
                              else cause["cause_key"])
    row["linked_finding_id"] = cause["linked"]
    trace["branch_F"] = (f"path not provably clean -> {row['suspected_cause']} "
                         f"(linked: {cause['linked']})")
    trace["result"] = SUSPECT_OUR_IMPLEMENTATION
    if unattributed:
        row["reasoning"] = (
            f"We could not confirm that {pred.name} on this run: {cause['wording']}.")
    elif contract_contradicts:
        trace["spec_branch"] = ("this run's own contract marks the bound "
                                "mechanism must-replicate with no approved "
                                "approximation")
        # The hedged wording's whole content is the open question the contract
        # has now answered, so it is replaced rather than quoted. An attributed
        # cause keeps its own wording, which was never in doubt.
        wording = (_CONTRACT_SETTLED_WORDING if hedged else cause["wording"])
        row["reasoning"] = (
            f"Our run did not support {pred.name}. This run's own methodology "
            f"contract requires that mechanism exactly and approves no "
            f"demo-scale approximation of it, so this is ours to fix rather "
            f"than a question about the paper's design: {wording}.")
    else:
        row["reasoning"] = (
            f"Our run did not support {pred.name}. This is most likely our "
            f"implementation, not a problem with the paper: {cause['wording']}. "
            f"We are flagging it for a fix and do not read it as the paper being "
            f"wrong.")
    row["what_would_verify"] = (
        "resolve the linked fidelity issue, then re-run the behavioral check")
    row["status_note"] = row["reasoning"]


def _mark_contract_consistent(row: dict, trace: dict, pred: "_Predicate",
                              signals: list[dict]) -> None:
    """R2C-047: the run's own contract is CONSISTENT with the failing check.

    Every bound element carries an approved demo-scale approximation or is
    recorded as not replicable at this scale, so the behavior the probe saw is
    a known limit the contract already sanctioned. Rendering it as "needs a fix
    on our side" would tell the researcher to repair something the run
    deliberately approximated, so the row lands at untested-at-this-scale with
    the approximation named."""
    cause = pick_cause(signals)
    row["status"] = UNTESTED_AT_THIS_SCALE
    row["suspected_cause"] = "contract_approved_approximation"
    row["linked_finding_id"] = cause["linked"]
    trace["spec_branch"] = ("this run's own contract approves the bound "
                            "mechanism's demo-scale approximation")
    trace["result"] = UNTESTED_AT_THIS_SCALE
    row["reasoning"] = (
        f"We could not confirm that {pred.name} at smoke scale. This run's own "
        f"methodology contract already approves a demo-scale approximation of "
        f"the bound mechanism, so this is a known limit of the demo rather than "
        f"a defect to repair. Read the number with that approximation in view.")
    row["what_would_verify"] = (
        "a full-scale run implementing the mechanism exactly, without the "
        "approved demo-scale approximation")
    row["status_note"] = row["reasoning"]


def _mark_contradicted(row: dict, trace: dict, pred: "_Predicate",
                       seed_set: dict | None) -> None:
    """S3: the contradiction gate held on every clause. The single status that
    points at the paper, and by the time it fires the path is fidelity-clean,
    scale-cleared, and reproduced, so it is a genuine human-must-look (§3.3,
    §10.1). Structurally unreachable in the current pipeline; reached only by a
    forced synthetic row in the tests."""
    k = int((seed_set or {}).get("k", _G5_MIN_SEEDS) or _G5_MIN_SEEDS)
    row["status"] = CONTRADICTED
    row["seed_set"] = dict(seed_set) if seed_set else _default_seed_set()
    trace["result"] = CONTRADICTED
    trace["blocked_by"] = None
    row["reasoning"] = (
        f"At the paper's own scale, with our implementation of this path verified "
        f"clean and our run producing a sane result across {k} seeds, our result "
        f"disagrees with the paper by more than the contradiction margin. We could "
        f"not find an implementation, scale, data, or variance explanation. This "
        f"warrants a careful human look before drawing conclusions about the paper.")
    row["what_would_verify"] = (
        "an independent re-run at the paper's scale and seeds confirming the same "
        "disagreement")
    row["status_note"] = row["reasoning"]


def _mark_untested_after_gate(row: dict, trace: dict, pred: "_Predicate",
                              seed_set: dict | None) -> None:
    """S3: a scale-free invariant failed on a provably clean path, but a scale,
    seed, or veto clause blocked contradiction (today, always the single-seed
    clause). The honest call is untested, not a phantom own-bug — one seed is not
    a verdict (§3.3 G5, §5 single-seed template). Not reached in the current
    pipeline (a failing behavioral check always carries an own-bug signal that
    routes to branch F); wired for the day the gate's other clauses can pass."""
    k = int((seed_set or {}).get("k", 1) or 1)
    row["status"] = UNTESTED_AT_THIS_SCALE
    row["seed_set"] = dict(seed_set) if seed_set else _default_seed_set()
    trace["result"] = UNTESTED_AT_THIS_SCALE
    seen = "one seed" if k == 1 else f"{k} seeds"
    row["reasoning"] = (
        f"We checked this on {seen} and saw it not hold, within normal "
        f"single-seed variation. The paper's result is the multi-seed picture, so "
        f"we are not treating {seen} as a verdict.")
    row["what_would_verify"] = (
        f"re-run across at least {_G5_MIN_SEEDS} seeds at the paper's scale and "
        f"confirm the disagreement reproduces")
    row["status_note"] = row["reasoning"]


def _synthesize_behavioral_row(pred: _Predicate, status: str,
                               matching: list[dict],
                               path_signals: list[dict],
                               run_mode: str = _DEFAULT_RUN_MODE,
                               seed_set: dict | None = None) -> dict:
    """One synthesized behavioral row, run through S1, S2, and branch F.

    check fail/flag, gate held (unreachable today)   -> contradicted (S3)
    check fail/flag, contract approves the approx    -> untested (R2C-047)
    check fail/flag, own-bug clause blocked          -> suspect (branch F)
    check fail/flag, only scale/seed clause blocked  -> untested (S3, clean path)
    discriminating pass, path clean                  -> verified_at_scale
    discriminating pass, path NOT clean              -> suspect (demoted, §6)
    non-discriminating pass                          -> untested (not the
        place the path's suspect signal rides — that is the discriminating /
        failed-check rows; a passing non-discriminating property stays untested)
    """
    row = _blank_row()
    row.update({
        "kind": "behavioral",
        "claim_id": pred.claim_id,
        "name": pred.name,
        "claim_text": pred.name,
        "scale_free": True,
        "level_claim": False,
        "depends_on": sorted({e for v in matching
                              for e in (v.get("element_ids") or [])}),
    })
    trace = {
        "S0": f"a probe evaluated this predicate ({'/'.join(pred.probe_ids)})",
        "S1": "scale-free behavioral predicate (paradigm allowlist, not "
              "prose-inferred)",
    }

    branch = _aggregate_spec_branch(matching)
    row["spec_branch"] = branch

    if status in ("fail", "flag"):
        # The row's own behavioral check did not pass: a suspect signal in
        # itself, combined with any other path signal for root-cause attribution.
        signals = list(path_signals)
        own = _own_check_signal(pred, status, matching)
        if own and not any(s["linked"] in pred.probe_ids for s in path_signals):
            signals.append(own)
        trace["S2"] = f"the backing check did not cleanly pass (status={status})"
        # S3: a scale-free invariant failed -> evaluate the contradiction gate.
        # Only if every clause holds is the row contradicted (impossible today).
        # An own-bug clause blocking -> branch F (suspect); a clean path blocked
        # only by a scale/seed/margin/veto clause -> untested (a one-seed wobble
        # is not a verdict). Today a failing check always carries an own-bug
        # signal, so this stays branch F, but the trace now records why the row
        # did not reach contradicted.
        gate = contradiction_gate(row, signals, seed_set=seed_set,
                                  run_mode=run_mode)
        trace["S3"] = gate["summary"]
        trace["gate"] = gate["clauses"]
        if not gate["held"]:
            trace["blocked_by"] = gate["blocked_by"]
        if gate["held"]:
            _mark_contradicted(row, trace, pred, seed_set)
        elif branch == CONTRACT_CONSISTENT:
            # R2C-047: the contract already approves an approximation of every
            # bound mechanism, so the failing check is a sanctioned demo-scale
            # limit. Evaluated ahead of branch F because branch F's whole job is
            # attributing a DEFECT, and there is no defect to attribute here.
            _mark_contract_consistent(row, trace, pred, signals)
        elif gate["own_bug_block"]:
            _mark_suspect(row, trace, pred, signals,
                          contract_contradicts=branch == CONTRACT_CONTRADICTS)
        else:
            _mark_untested_after_gate(row, trace, pred, seed_set)
    elif status == "pass" and pred.discriminating and not path_signals:
        # S2: clean pass + discriminating + clean path + no level -> verified.
        row["status"] = VERIFIED_AT_SCALE
        row["verifiable_aspect"] = pred.name
        trace["S2"] = ("checker passed; the check is an ablation contrast (it "
                       "fails for the contribution-removed baseline), so it is "
                       "discriminating; no level claim")
        trace["G3"] = "path provably clean (no fidelity-not-clean signal)"
        trace["result"] = VERIFIED_AT_SCALE
        row["reasoning"] = (
            f"We verified at smoke scale that {pred.name}. This check fails for "
            f"the contribution-removed baseline, so it is meaningful. It is a "
            f"scale-free behavioral property and does not confirm the paper's "
            f"full-scale headline numbers.")
        row["what_would_verify"] = ""
    elif status == "pass" and pred.discriminating:
        # Would verify, but the path is not fidelity-clean -> demote (§6:
        # fidelity-cleanliness is a precondition of verified, never only of
        # contradicted; a structural pass cannot paint over a same-path defect).
        trace["S2"] = ("checker passed and is discriminating, but the path is "
                       "not provably clean")
        _mark_suspect(row, trace, pred, path_signals)
    else:
        # Non-discriminating pass: holds, but a trivial baseline also satisfies
        # it, so it cannot verify and is not the row the suspect signal rides.
        row["status"] = UNTESTED_AT_THIS_SCALE
        trace["S2"] = ("checker passed but the check is NOT discriminating — a "
                       "trivial baseline also satisfies it")
        trace["blocked_by"] = "S2:not-discriminating"
        trace["result"] = UNTESTED_AT_THIS_SCALE
        row["reasoning"] = (
            f"{pred.name[:1].upper()}{pred.name[1:]} holds, but a trivial "
            f"baseline (such as random acquisition) also satisfies it, so on "
            f"its own it does not confirm the paper's contribution.")
        row["what_would_verify"] = pred.what_would_verify

    row["verdict_trace"] = trace
    row["status_note"] = row["reasoning"]
    return row


def synthesize_behavioral_rows(verdicts: list[dict], paradigm: str,
                               findings: list[dict] | None = None,
                               run_mode: str = _DEFAULT_RUN_MODE,
                               seed_set: dict | None = None) -> list[dict]:
    """One behavioral row per allowlist predicate the run's probes actually
    evaluated (§2.3). A predicate whose backing probe never ran, could not run,
    or was outside the declared method shape gets no row — the ledger never
    claims to have checked something it did not. The run's path-fidelity
    signals (branch F) are computed once and gate every row's verified
    eligibility; `run_mode` and `seed_set` are the run-level inputs to the
    contradiction gate (§3.3)."""
    path_signals = path_fidelity_signals(verdicts, findings)
    seed_set = seed_set or _default_seed_set()
    rows = []
    for pred in _allowlist_for(paradigm):
        matching = [v for v in verdicts
                    if v.get("probe_id") in pred.probe_ids]
        status = _aggregate_status(matching)
        if status in ("none", "unprobeable", "not_applicable"):
            continue
        rows.append(_synthesize_behavioral_row(
            pred, status, matching, path_signals,
            run_mode=run_mode, seed_set=seed_set))
    return rows


def _tally(claims: list[dict]) -> dict:
    out = {VERIFIED_AT_SCALE: 0, "contradicted": 0,
           "suspect_our_implementation": 0, UNTESTED_AT_THIS_SCALE: 0,
           "directionally_checked": 0}
    for row in claims:
        out[row["status"]] = out.get(row["status"], 0) + 1
    return out


def build_ledger(paper_map: dict, verdicts: list[dict] | None = None,
                 paradigm: str = "",
                 findings: list[dict] | None = None,
                 run_mode: str = _DEFAULT_RUN_MODE,
                 seed_set: dict | None = None,
                 executed_metrics: list[dict] | None = None) -> dict:
    """The full CT-3 ledger: enriched lifted rows (always first, so the
    committed-map test still finds the first experiment row at claims[0]) plus
    synthesized behavioral rows, with the honest tally the REPORT leads with.
    `findings` are the reviewer's unresolved findings, which feed the branch-F
    fidelity gate alongside the probe verdicts. `run_mode` and `seed_set` are the
    run-level inputs to the contradiction gate (default smoke / single-seed, so
    the gate stays closed and contradicted is unreachable). `executed_metrics`
    are metric observations extracted from the executed notebook."""
    verdicts = verdicts or []
    seed_set = seed_set or _default_seed_set()
    path_signals = path_fidelity_signals(verdicts, findings)
    lifted = [_enrich_lifted_row(
        r, executed_metrics or [], path_signals,
        run_mode=run_mode, seed_set=seed_set)
        for r in build_claims_ledger(paper_map)]
    behavioral = synthesize_behavioral_rows(
        verdicts, paradigm, findings, run_mode=run_mode, seed_set=seed_set)
    claims = lifted + behavioral
    return {
        "schema_version": "2.0.0",
        "paradigm": paradigm,
        "tally": _tally(claims),
        "claims": claims,
    }


def _read_paradigm(run_dir: Path) -> str:
    spec_path = run_dir / ".pipeline" / "method_spec.json"
    if not spec_path.is_file():
        return ""
    try:
        spec = json.loads(spec_path.read_text(encoding="utf-8"))
        return (spec.get("comparison", {}).get("classification", {})
                .get("id", "") or "")
    except (json.JSONDecodeError, AttributeError):
        return ""


def _read_run_mode(run_dir: Path) -> str:
    """The run mode, read from run state (driver_state.json). Defaults to
    "smoke" and is NEVER inferred (the §10.4 rule: a full-scale run sets
    run_mode="full" explicitly). Today nothing writes "full", so the level-claim
    arm of the contradiction gate stays shut and headline numbers can never be
    contradicted. An unknown value falls back to "smoke" (fail-closed)."""
    state = run_dir / ".pipeline" / "driver_state.json"
    if not state.is_file():
        return _DEFAULT_RUN_MODE
    try:
        mode = json.loads(state.read_text(encoding="utf-8")).get("run_mode")
    except (json.JSONDecodeError, OSError, AttributeError):
        return _DEFAULT_RUN_MODE
    return mode if mode in (_DEFAULT_RUN_MODE, _FULL_RUN_MODE) else _DEFAULT_RUN_MODE


def _read_seed_set(run_dir: Path) -> dict:
    """The smoke run's seed set {k, reproduced}, read from run state. Defaults to
    a single, unreproduced seed, so the gate's reproduction clause (G5) fails
    closed: the smoke run is single-seed today and §10.3 bars silently dropping
    below k=3, so until a multi-seed runner writes this nothing can be called
    reproduced. Defensive on shape — a malformed entry falls back to the default."""
    state = run_dir / ".pipeline" / "driver_state.json"
    default = _default_seed_set()
    if not state.is_file():
        return default
    try:
        ss = json.loads(state.read_text(encoding="utf-8")).get("seed_set")
    except (json.JSONDecodeError, OSError, AttributeError):
        return default
    if not isinstance(ss, dict):
        return default
    try:
        # `reproduced` is true ONLY for a real JSON `true` — a stringy or numeric
        # value fails closed to False (the fail-closed-toward-humility stance,
        # §10.3), never coerced open through Python truthiness. OverflowError
        # guards a non-finite k (json accepts the bare token `Infinity`, and
        # int(inf) raises), so a malformed state file falls back to the default
        # rather than crashing the battery.
        return {"k": max(1, int(ss.get("k", 1) or 1)),
                "reproduced": ss.get("reproduced") is True}
    except (TypeError, ValueError, OverflowError):
        return default


def contradiction_flag_verdicts(ledger: dict) -> list[ProbeVerdict]:
    """One synthetic `flag_for_researcher` probe verdict per contradicted row
    (§10.1): a contradicted claim must demote delivery to draft, and it does so
    through the EXISTING probe-reasons path (delivery_label reads
    flag_for_researcher as a demoting reason), never a new parallel gate. Empty
    today — contradicted is structurally unreachable — and exercised by a forced
    synthetic row in the tests."""
    rows = ledger.get("claims") if isinstance(ledger, dict) else None
    out = []
    for row in rows or []:
        if isinstance(row, dict) and row.get("status") == CONTRADICTED:
            out.append(ProbeVerdict(
                "CT-3", "flag_for_researcher",
                "a claim was contradicted at the paper's scale and needs a "
                "careful human look before any conclusion about the paper: "
                f"{row.get('name') or row.get('claim_id') or 'claim'}",
                tier="claims",
                element_ids=list(row.get("depends_on") or [])))
    return out


def _report_verdicts(report) -> list[dict]:
    """Normalize a ProbeReport, a {"verdicts": [...]} dict, or a raw list of
    verdict dicts into a list of verdict dicts."""
    if report is None:
        return []
    if hasattr(report, "verdicts"):
        return [v.to_dict() for v in report.verdicts]
    if isinstance(report, dict):
        return list(report.get("verdicts", []))
    return list(report)


def _read_findings(run_dir: Path) -> list[dict]:
    """The reviewer's findings (Stage 4), which feed the branch-F fidelity gate.
    Absent or unparseable -> no findings (the probe signals still gate)."""
    path = run_dir / ".pipeline" / "review_report.json"
    if not path.is_file():
        return []
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return []
    findings = report.get("findings") if isinstance(report, dict) else None
    return [f for f in (findings or []) if isinstance(f, dict)]


def _read_executed_metrics(run_dir: Path) -> list[dict]:
    """Executed-side metric observations from the run-root notebook.

    Missing, unexecuted, or malformed notebooks produce no observations. The
    ledger still builds and records an explicit unmatched gap.  For a
    schema-2 structured demo artifact (R2C-086), executed numbers are eligible
    for claims use only when evaluation validity is valid AND every required
    comparator passed.  Legacy/non-structured families retain their existing
    behavior; printed PASS prose is never consulted here.
    """
    structured_required = False
    try:
        from demo_verdict import structured_demo_requirement  # noqa: PLC0415

        _, structured_required = structured_demo_requirement(run_dir)
    except Exception:  # noqa: BLE001 - ordinary legacy families remain unchanged
        structured_required = False

    verdict: dict | None = None
    verdict_path = run_dir / ".pipeline" / "demo_verdict.json"
    if verdict_path.is_file():
        try:
            verdict = json.loads(verdict_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            verdict = None
    if structured_required and not isinstance(verdict, dict):
        return []

    if isinstance(verdict, dict):
        version = str(verdict.get("schema_version") or "")
        # Schema-less/schema-1 records are delivered historical evidence and
        # retain their legacy claims behavior even when today's effective
        # taxonomy has since adopted demo_skill.
        legacy = (not version or version.startswith("1.")) and (
            verdict.get("verdict") in {"succeeded", "failed", "undetermined"}
        )
        axes = verdict.get("evidence_status")
        if structured_required and not legacy:
            complete_axes = isinstance(axes, dict) and all(
                isinstance(axes.get(name), dict)
                for name in (
                    "execution", "evaluation_validity", "mechanism", "skill",
                    "paper_benchmark",
                )
            )
            if not version.startswith("2.") or not complete_axes:
                return []
        if version.startswith("2.") and isinstance(axes, dict):
            evaluation = axes.get("evaluation_validity")
            skill = axes.get("skill")
            eligible = (
                isinstance(evaluation, dict)
                and evaluation.get("status") == "valid"
                and isinstance(skill, dict)
                and skill.get("status") == "demonstrated"
            )
            if not eligible:
                return []
    path = run_dir / "notebook.ipynb"
    if not path.is_file():
        return []
    try:
        nb = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return []
    return executed_metric_observations(nb)


def probe_claims_ledger(run_dir, report=None, write: bool = True,
                        ledger_path=None) -> ProbeVerdict:
    """CT-3 battery entry: build the ledger from the run's own paper_map plus,
    when the battery passes its accumulated `report`, the other probes'
    verdicts (which feed the synthesized behavioral rows). Persists the ledger
    to .pipeline/claims_ledger.json, or to `ledger_path` when given (the
    isolated output mode: a post-delivery battery run must never rewrite the
    delivery-time ledger it needs as its comparison baseline — R2C-019). A
    `pass` here means the ledger BUILT — the gap is visible — it never means
    the claims verified; verified rows are the separate behavioral rows,
    gated on discriminating power."""
    run_dir = Path(run_dir)
    pm_path = run_dir / ".pipeline" / "paper_map.json"
    if not pm_path.is_file():
        return ProbeVerdict("CT-3", "unprobeable", "no paper_map.json",
                            tier="claims")
    try:
        paper_map = json.loads(pm_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        return ProbeVerdict("CT-3", "unprobeable",
                            f"paper_map.json unparseable: {e}", tier="claims")
    paradigm = _read_paradigm(run_dir)
    ledger = build_ledger(paper_map, _report_verdicts(report), paradigm,
                          findings=_read_findings(run_dir),
                          run_mode=_read_run_mode(run_dir),
                          seed_set=_read_seed_set(run_dir),
                          executed_metrics=_read_executed_metrics(run_dir))
    # Delivery wiring (§10.1): a contradicted row emits a synthetic
    # flag_for_researcher verdict onto the battery report, so it demotes through
    # the existing probe-reasons path (no new gating channel). Inert today —
    # contradicted is unreachable — and a no-op when called standalone (no report).
    if hasattr(report, "add"):
        for v in contradiction_flag_verdicts(ledger):
            report.add(v)
    if not ledger["claims"]:
        return ProbeVerdict(
            "CT-3", "warn",
            "paper_map carries no experiment elements — no claims ledger "
            "to build (unusual; check the decomposition)", tier="claims")
    if write:
        out = (Path(ledger_path) if ledger_path is not None
               else run_dir / ".pipeline" / "claims_ledger.json")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(ledger, indent=2) + "\n", encoding="utf-8")
    tally = ledger["tally"]
    n_behavioral = sum(1 for r in ledger["claims"] if r["kind"] == "behavioral")
    n_lifted = len(ledger["claims"]) - n_behavioral
    n_matched = sum(1 for r in ledger["claims"]
                    if r.get("kind") == "lifted" and r.get("executed_side"))
    return ProbeVerdict(
        "CT-3", "pass",
        f"claims ledger built: {n_lifted} lifted claim(s) + {n_behavioral} "
        f"behavioral row(s); {n_matched} lifted number comparison(s) matched "
        f"to executed notebook output; "
        f"{tally[VERIFIED_AT_SCALE]} verified at smoke "
        f"scale, {tally[SUSPECT_OUR_IMPLEMENTATION]} suspect-our-implementation, "
        f"{tally[UNTESTED_AT_THIS_SCALE]} untested — the verification gap is "
        f"visible, never silent",
        evidence=f"verified={[r['claim_id'] for r in ledger['claims'] if r['status'] == VERIFIED_AT_SCALE]}",
        tier="claims")
