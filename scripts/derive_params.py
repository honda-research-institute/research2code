"""Stage 2.x — parameter-deriver (replaces the old calibrator).

Reads the spec + taxonomy build context. Produces a structured params dict with
provenance for every parameter the notebook embeds. Output:

    <RUN_DIR>/.pipeline/params.json

For each parameter, the script picks one of four sources:

  - "paper":           paper specifies the value AND we use it (smoke-faithful).
  - "system_default":  paper specifies a value but we use a smaller one for smoke.
  - "system_inferred": paper doesn't specify; we picked based on taxonomy convention.
  - "spec_default":    the value comes from the analyzer's pluggable signature
                       default and no unambiguous paper value was found.

Method-specific paradigm-extras (mc_samples, core_set_size, R_0, eta, batch_returns,
etc.) are extracted from the spec's `comparison.pluggable_component.signature`
keyword-argument defaults.

The script is fully deterministic — no LLM calls. The reasoning text it
produces is template-based; an optional future LLM-refinement step could
polish the prose, but the templates are paper-faithful enough for smoke runs.

Usage:

    python scripts/derive_params.py --spec <method_spec.json> --run-dir <output_dir>

Exit codes:
  0  params.json written successfully
  1  setup error (missing spec / unparseable spec / missing required fields)
"""

from __future__ import annotations

import argparse
import ast
import json
import keyword
import math
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from schemas.method_spec import (  # noqa: E402
    evaluation_protocol_quantity_for_param,
    extract_param_paper_values_from_text,
    methodology_contract_paper_values_for_param,
)
from schemas.params import SCHEMA_VERSION  # noqa: E402
from scripts import taxonomy  # noqa: E402
from scripts.bundle_axis_floor import (  # noqa: E402
    ProtocolAxisUnresolved,
    load_bundle_manifest,
    protocol_axis_param_shrinks,
    protocol_quantity_steps,
)
from scripts.validate_params_provenance import (  # noqa: E402
    paper_claim_findable,
    value_findable_in_paper,
)


# ---------------------------------------------------------------------------
# Spec helpers
# ---------------------------------------------------------------------------

# Analyzer free-text stating the paper does NOT give the value ("Not
# explicitly stated in the paper", "not specified", "no value given").
# One shared definition for the parser and the provenance entry: the old
# phrase list missed the "stated" variant, so the SRL 2026-07-21 spec text
# ("Not explicitly stated in the paper; RMSprop is named in Algorithm 1
# line 13") fell through to the heuristic prose scan, which extracted the
# "1" of "Algorithm 1" as a fabricated learning rate of 1.0 (stage-2x F001,
# critical). Prose that declares silence must never be value-scanned.
_LR_UNSPECIFIED_RE = re.compile(
    r"\b(?:not|no)\s+(?:\w+[\s-]+){0,2}"
    r"(?:specified|stated|given|reported|provided)\b"
    r"|\bunspecified\b",
    re.IGNORECASE,
)


def _parse_learning_rate(lr_str: str, is_image: bool) -> tuple[float, str]:
    """Robustly extract a learning rate from a spec's free-text learning_rate field.

    Handles four common phrasings:
      - "0.001 for image data, 0.0001 for non-image data" → pick by is_image
      - "0.01 for image datasets" → pick the image value if is_image
      - bare numeric: "0.001" → use directly
      - leading value + qualifier: "2e-4 initial, cyclic policy" → 2e-4

    Returns (numeric_value, note_suffix). Falls back to 0.001 if nothing parseable.
    """
    if not lr_str:
        return 0.001, " Paper learning rate not specified; using 0.001 as paradigm-default."

    # Find "<number> for <descriptor>" patterns.
    pattern = re.compile(
        r"(\d+\.?\d*(?:[eE][-+]?\d+)?)\s*for\s+([^,.;]+?)(?=[,.;]|$)",
        re.IGNORECASE,
    )
    matches = pattern.findall(lr_str)

    image_lr: float | None = None
    non_image_lr: float | None = None
    for value_str, descriptor in matches:
        try:
            value = float(value_str)
        except ValueError:
            continue
        desc = descriptor.lower()
        if "non-image" in desc or "non image" in desc or "tabular" in desc:
            non_image_lr = value
        elif "image" in desc:
            image_lr = value

    if is_image and image_lr is not None:
        # Attribute the spec's summary string as spec text, never as a
        # paper quote: the quote-match probe (US-3) verifies "Paper: ..."
        # citations verbatim against paper.md, and the analyzer's
        # paraphrase (with dataset parentheticals) is not in the paper —
        # the 2026-06-10 BADGE matrix-row audit caught exactly that. The
        # paper-stated VALUE is still verified by US-3's value arm.
        return image_lr, (f" Spec training.learning_rate: '{lr_str[:80]}'."
                          f" Selected image-data value.")
    if not is_image and non_image_lr is not None:
        return non_image_lr, (f" Spec training.learning_rate: '{lr_str[:80]}'."
                              f" Selected non-image value.")
    if image_lr is not None:  # is_image=False but only image LR specified
        return image_lr, " Paper specifies image-data value; using it as the default."
    if non_image_lr is not None:
        return non_image_lr, " Paper specifies non-image value; using it as the default."

    # The field says the paper does not state the value. Stop here — a
    # heuristic scan of such prose is exactly the fabrication vector (the
    # SRL case above), and any number it finds is a section/line/citation
    # artifact, not a learning rate.
    if _LR_UNSPECIFIED_RE.search(lr_str):
        return 0.001, (" Spec says the learning rate is not stated in the "
                       "paper; using 0.001 as paradigm-default.")

    # No "for descriptor" patterns. Try parsing as a single float.
    try:
        return float(lr_str.strip()), " Paper-stated learning rate."
    except ValueError:
        pass

    # A leading numeric ("2e-4 initial, cyclic policy") is the field's stated
    # value followed by a qualifier — a direct statement, not a prose scan
    # (bev-distill 2026-07-01: this phrasing fell to the heuristic scan below
    # and shipped a false "paper does not specify" reasoning). Range-gate it
    # like the last-resort scan so a field that leads with an epoch count or
    # decay step ("24 epochs, cyclic") never parses as a learning rate, and
    # require a decimal point or exponent — a bare integer lead ("1 epoch
    # then decay") is a count, and 1 sits inside the plausible range.
    lead = re.match(r"\s*(\d+\.?\d*(?:[eE][-+]?\d+)?)(?![\w.])", lr_str)
    if lead and re.search(r"[.eE]", lead.group(1)):
        try:
            lead_value = float(lead.group(1))
        except ValueError:
            lead_value = None
        if lead_value is not None and 1e-6 <= lead_value <= 1:
            return lead_value, (f" Spec training.learning_rate: '{lr_str[:80]}'."
                                f" Leading stated value.")

    # Last resort: extract the first PLAUSIBLE learning rate from the prose.
    # Remove bracketed citations ("[1]") first, then scan the remaining numbers
    # and pick the first that falls in the plausible LR range (1e-6..1, the same
    # bound the US-1 provenance gate enforces). A bare integer — a citation year
    # ("Gal et al. 2017"), epoch count, line number ("Algorithm 1 line 13"),
    # or sample size — is never a learning rate and is skipped: the range gate
    # alone let "Algorithm 1" parse to a fabricated lr=1.0 (SRL 2026-07-21
    # stage-2x F001), because 1 sits inside [1e-6, 1]. A real learning rate in
    # prose always carries a decimal point or an exponent.
    lr_clean = re.sub(r"\[\d+\]", "", lr_str)
    nums = re.findall(r"\d+\.?\d*(?:[eE][-+]?\d+)?", lr_clean)
    for num in nums:
        if not re.search(r"[.eE]", num):
            continue
        try:
            value = float(num)
        except ValueError:
            continue
        if 1e-6 <= value <= 1:
            preview = lr_str[:60] + ("…" if len(lr_str) > 60 else "")
            return value, (f" Paper says '{preview}' — using first plausible "
                           f"learning rate ({value}).")

    return 0.001, " Paper learning rate not parseable; using 0.001 as paradigm-default."


def _lr_provenance_entry(lr_str: str, lr_value: float, note_suffix: str,
                         paper_section: str,
                         paper_text: str | None) -> dict:
    """Source-decision for the learning_rate param, shared by the AL and KD
    training-param builders. One definition: the two paths had drifted, and
    the KD copy lacked the findability arm (bev-distill 2026-07-01).

    Three signals mark the value system_inferred rather than paper-stated:
      1. The spec's free-text field explicitly says the paper doesn't
         specify the LR (analyzer prose like "Not explicitly specified in
         paper; likely 0.001 (standard for Adam)")
      2. _parse_learning_rate fell through to a default
         ("not parseable" / "not specified")
      3. The numeric was extracted heuristically as a last resort (the
         "using first plausible" path) — a number found in prose is not a
         direct paper statement. The reasoning must then say exactly that;
         it must NOT claim the paper is silent (the stage-2x reviewer
         correctly halts on that claim when the paper does state the value).

    Even a cleanly-parsed spec value is only a *paper* claim if the value is
    actually findable in the paper text: PDF→markdown conversion mangles
    numerals (the GBALD paper rendered 0.001 as "10K3"/"103"). This uses the
    SAME findability predicate as the US-3 provenance validator, so the
    deriver never stamps a source="paper" that validate_params_output.py
    would then reject and halt on.
    """
    spec_says_unspecified = bool(_LR_UNSPECIFIED_RE.search(lr_str))
    note_says_default = (
        "not parseable" in note_suffix
        or "not specified" in note_suffix
        or "not stated" in note_suffix
    )
    extracted_heuristically = "using first" in note_suffix

    if spec_says_unspecified or note_says_default:
        return {
            "value": lr_value,
            "source": "system_inferred",
            "reasoning": (
                f"Paper does not explicitly specify the learning rate "
                f"(spec.training.learning_rate: '{lr_str[:120]}'). Selected "
                f"{lr_value} as the Adam-default convention."
            ),
        }
    if extracted_heuristically:
        return {
            "value": lr_value,
            "source": "system_inferred",
            "reasoning": (
                f"Extracted {lr_value} as the first plausible learning rate "
                f"in spec.training.learning_rate ('{lr_str[:120]}') — a "
                f"heuristic scan of prose, not a direct paper statement, so "
                f"it is recorded as a system inference."
            ),
        }
    if not value_findable_in_paper(lr_value, paper_text):
        return {
            "value": lr_value,
            "source": "system_inferred",
            "reasoning": (
                f"spec.training.learning_rate parses to {lr_value} "
                f"('{lr_str[:120]}'), but no rendering of that value appears in "
                f"the paper text (PDF→markdown conversion can mangle numerals), "
                f"so it is adopted as an Adam-default-consistent inference "
                f"rather than asserted as a paper claim."
            ),
        }
    return {
        "value": lr_value,
        "source": "paper",
        "paper_section": paper_section,
        "note": f"Adam optimizer at {lr_value}.{note_suffix}",
    }


def _is_image_data(spec: dict) -> bool:
    """Heuristic: data is image-shaped if benchmark_name mentions a known image dataset."""
    data_setup = (spec.get("critical_requirements") or {}).get("data_setup") or {}
    benchmark = (data_setup.get("benchmark_name") or "").lower()
    return any(name in benchmark for name in ("mnist", "cifar", "svhn", "imagenet"))


def _parse_signature_kw_defaults(signature: str) -> dict[str, object]:
    """Parse a signature string and return {param_name: default_value, ...} for
    keyword-arg parameters with literal defaults. Skips fixed positional args
    declared in the taxonomy build plan's `pluggable_component.contract.fixed_positional_args`,
    plus `seed` (which is the harness-injected per-round seed).

    Today's known paradigm-fixed args:
      - active_learning parent / batch_acquisition: [model, x_unlabeled, batch_size]
      - active_learning bayesian (override): [model, x_unlabeled, x_labeled, batch_size]
      - knowledge_distillation: [student, teacher, batch]
      - motion_planning: [start, goal, environment, dynamics]

    The union covers every position any paradigm might use; an unused name in
    the union is harmless (it just doesn't match anything in the method's
    actual signature). Adding new paradigms with new fixed args means
    extending this set. (Eventual cleaner form: read
    `pluggable_component.contract.fixed_positional_args` from the matched
    taxonomy build plan rather than maintaining a union here — PA-D8.)
    """
    SKIP = {"model", "x_unlabeled", "x_labeled", "batch_size", "seed",
            "student", "teacher", "batch",  # KD positionals
            "start", "goal", "environment", "dynamics"}  # motion_planning positionals
    try:
        stub = f"def {signature}:\n    pass\n"
        tree = ast.parse(stub)
    except (SyntaxError, ValueError):
        return {}
    if not tree.body or not isinstance(tree.body[0], (ast.FunctionDef, ast.AsyncFunctionDef)):
        return {}
    func = tree.body[0]
    args = func.args

    out: dict[str, object] = {}

    # Regular positional/positional-or-keyword args
    n_pos = len(args.args)
    n_pos_no_default = n_pos - len(args.defaults)
    for i, arg in enumerate(args.args):
        if arg.arg in SKIP:
            continue
        default_idx = i - n_pos_no_default
        if default_idx < 0:
            continue
        try:
            out[arg.arg] = ast.literal_eval(args.defaults[default_idx])
        except (ValueError, SyntaxError):
            pass

    # Keyword-only args
    for i, arg in enumerate(args.kwonlyargs):
        if arg.arg in SKIP:
            continue
        default = args.kw_defaults[i]
        if default is None:
            continue
        try:
            out[arg.arg] = ast.literal_eval(default)
        except (ValueError, SyntaxError):
            pass

    return out


def _smoke_economics_param_floor(
    paradigm_id: str, name: str, tax: "taxonomy.Taxonomy | None" = None
) -> tuple[float, str] | None:
    """Return a deterministic smoke floor for a param, if the served node declares one."""
    econ = taxonomy.load_smoke_economics(paradigm_id, tax) or {}
    floors = econ.get("floors")
    if not isinstance(floors, dict):
        return None
    entry = floors.get(name)
    if not isinstance(entry, dict):
        return None
    value = (
        entry.get("absolute_min")
        if entry.get("absolute_min") is not None
        else entry.get("minimum")
        if entry.get("minimum") is not None
        else entry.get("absolute")
    )
    if value is None:
        return None
    try:
        floor = float(value)
    except (TypeError, ValueError):
        return None
    return floor, str(entry.get("reason") or f"taxonomy smoke_economics floor for {name}")


def _apply_numeric_floor(value: object, floor: float) -> object:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return value
    if float(value) >= floor:
        return value
    return int(floor) if float(floor).is_integer() else floor


# ---------------------------------------------------------------------------
# Parameter builders
# ---------------------------------------------------------------------------


def _detect_demo_dataset(
    candidate_datasets: list[str], run_dir: Path | None
) -> tuple[str | None, str]:
    """Which of the paper's own per-dataset names does the generated package
    actually mention? Deterministic case-insensitive text scan of method/*.py
    — at stage 2.x the package exists, and its data path (bundled example
    data, dataset-specific fallback download) is what the demo will run.
    The candidate list comes from the spec's own per_dataset_values keys, so
    no dataset vocabulary is hard-coded here. Returns (dataset, evidence) on
    an unambiguous single match, else (None, reason) — ambiguity falls back
    to the scalar, never a guess."""
    if run_dir is None:
        return None, "deriver ran without --run-dir (no generated package to consult)"
    method_dir = Path(run_dir) / "method"
    if not method_dir.is_dir():
        return None, "generated method package not found at method/"
    mentioned: dict[str, str] = {}
    for path in sorted(method_dir.glob("*.py")):
        try:
            text = path.read_text(encoding="utf-8").lower()
        except OSError:
            continue
        for name in candidate_datasets:
            if name.lower() in text:
                mentioned.setdefault(name, path.name)
    if len(mentioned) == 1:
        name, fname = next(iter(mentioned.items()))
        return name, f"method/{fname} names {name}"
    if not mentioned:
        return None, "no per-dataset map key is named by the generated package sources"
    return None, ("more than one map dataset is named by the generated package "
                  f"sources ({', '.join(sorted(mentioned))}); binding would be a guess")


def _resolve_per_dataset_bindings(
    ds: dict, run_dir: Path | None
) -> tuple[dict, dict | None]:
    """Bind data_setup scalars to the demo dataset's paper values.

    Returns (effective_data_setup, binding_record). When the spec carries no
    per_dataset_values the input is returned untouched with record None.
    Bound params get the demo dataset's value; params whose dataset cannot be
    determined keep the spec's primary-dataset scalar and are recorded as
    fallbacks (the driver logs one assumptions.md entry per fallback —
    honest disclosure, never a halt)."""
    pdv = ds.get("per_dataset_values") or {}
    if not isinstance(pdv, dict) or not pdv:
        return ds, None
    candidates = sorted({
        key
        for dataset_map in pdv.values()
        if isinstance(dataset_map, dict)
        for key in dataset_map
    })
    demo_dataset, evidence = _detect_demo_dataset(candidates, run_dir)
    effective = dict(ds)
    record: dict = {"demo_dataset": demo_dataset, "bound": [], "fallbacks": []}
    for name, dataset_map in pdv.items():
        if not isinstance(dataset_map, dict) or not dataset_map:
            continue
        match = None
        if demo_dataset is not None:
            for key, value in dataset_map.items():
                if key.strip().lower() == demo_dataset.strip().lower():
                    match = (key, value)
                    break
        if match is not None:
            key, value = match
            effective[name] = value
            record["bound"].append({
                "param": name,
                "dataset": key,
                "value": value,
                "evidence": evidence,
            })
        else:
            record["fallbacks"].append({
                "param": name,
                "dataset_map": dict(dataset_map),
                "scalar_value": ds.get(name),
                "reason": (
                    f"the paper states per-dataset values for {name} but the "
                    f"demo's dataset could not be bound ({evidence}); using "
                    f"the spec's primary-dataset scalar {ds.get(name)}"
                ),
            })
    return effective, record


def _annotate_bound_datasets(params: dict, binding_record: dict | None) -> None:
    """Attach bound_dataset + a note suffix to each bound param entry."""
    if not binding_record:
        return
    for row in binding_record["bound"]:
        entry = params.get(row["param"])
        if not isinstance(entry, dict):
            continue
        entry["bound_dataset"] = row["dataset"]
        suffix = (
            f" Bound to the {row['dataset']} paper value via "
            f"data_setup.per_dataset_values ({row['evidence']})."
        )
        if entry.get("note"):
            entry["note"] += suffix
        elif entry.get("reasoning"):
            entry["reasoning"] += suffix
        else:
            entry["note"] = suffix.strip()


def _add_data_setup_params(
    params: dict, spec: dict, repo_root: Path,
    tax: "taxonomy.Taxonomy | None" = None,
) -> None:
    cr = spec.get("critical_requirements") or {}
    ds = cr.get("data_setup") or {}
    paper_section = ds.get("paper_section") or "Section 4 (data_setup)"
    sig_defaults = _parse_signature_kw_defaults(
        ((spec.get("comparison") or {}).get("pluggable_component") or {})
        .get("signature") or "")

    # batch_size — the selector's only runtime output-count contract.
    # Legacy analyzer drafts sometimes emitted `batch_output` as a separate
    # smaller smoke value. Collapse that into batch_size provenance instead of
    # surfacing a second output-count parameter.
    batch_size_paper = ds.get("batch_size")
    legacy_batch_output_name = (
        "batch_output"
        if sig_defaults.get("batch_output") is not None
        else "batch_outputs"
        if sig_defaults.get("batch_outputs") is not None
        else None
    )
    legacy_batch_output = (
        sig_defaults.get(legacy_batch_output_name)
        if legacy_batch_output_name is not None
        else None
    )
    if batch_size_paper is not None:
        if (
            isinstance(legacy_batch_output, (int, float))
            and legacy_batch_output != batch_size_paper
        ):
            params["batch_size"] = {
                "value": legacy_batch_output,
                "source": "system_default",
                "paper_value": batch_size_paper,
                "reasoning": (
                    f"Paper uses batch_size={batch_size_paper} per data_setup. "
                    f"The generated signature also carried legacy "
                    f"{legacy_batch_output_name}={legacy_batch_output}; R2C treats that as a "
                    "smoke-scale runtime batch_size so the notebook, params table, "
                    "and acquisition loop share one output-count source of truth. "
                    "`batch_returns` remains the larger candidate prefilter."
                ),
            }
        else:
            params["batch_size"] = {
                "value": batch_size_paper,
                "source": "paper",
                "paper_section": paper_section,
                "note": f"Paper uses batch_size={batch_size_paper} per data_setup.",
            }
    delivered_batch_size = (
        (params.get("batch_size") or {}).get("value")
        if isinstance(params.get("batch_size"), dict)
        else None
    ) or batch_size_paper or 100

    # num_rounds — system_default if paper >> smoke
    num_rounds_paper = ds.get("num_rounds")
    smoke_num_rounds = 5
    if num_rounds_paper:
        if num_rounds_paper > smoke_num_rounds * 2:
            paper_bs = batch_size_paper or delivered_batch_size
            labels_paper = num_rounds_paper * paper_bs
            labels_smoke = smoke_num_rounds * delivered_batch_size
            params["num_rounds"] = {
                "value": smoke_num_rounds,
                "source": "system_default",
                "paper_value": num_rounds_paper,
                "reasoning": (
                    f"Paper runs {num_rounds_paper} rounds (~{labels_paper:,} labels at "
                    f"batch_size={paper_bs}). At smoke scale we run {smoke_num_rounds} rounds "
                    f"at runtime batch_size={delivered_batch_size} "
                    f"(~{labels_smoke:,} labels) — enough to see a learning curve emerge "
                    f"while keeping the acquisition loop (which re-trains and re-scores the "
                    f"pool every round) cheap to run on a laptop CPU at smoke scale."
                ),
            }
        else:
            params["num_rounds"] = {
                "value": num_rounds_paper,
                "source": "paper",
                "paper_section": paper_section,
                "note": f"Paper uses {num_rounds_paper} rounds.",
            }

    # initial_labeled — paper-faithful
    initial_labeled_paper = ds.get("initial_labeled")
    if initial_labeled_paper is not None:
        params["initial_labeled"] = {
            "value": initial_labeled_paper,
            "source": "paper",
            "paper_section": paper_section,
            "note": (
                f"Paper bootstraps with {initial_labeled_paper} uniformly-random labeled examples."
            ),
        }

    # pool_size — system_default; paper uses full dataset, we subsample for smoke
    smoke_pool_default = 800
    pool_paper_explicit = ds.get("pool_size")
    if pool_paper_explicit is not None:
        paper_value_str = pool_paper_explicit
        paper_summary = f"{pool_paper_explicit:,}-sample"
    else:
        paper_value_str = "full training set"
        paper_summary = "full training set (typically 60k–73k samples)"

    # The demo protocol must FIT inside the pool. The 2026-06-10 GBALD
    # validation re-run shipped initial_labeled=1000 against the 800 smoke
    # default: the unlabeled pool was empty before round 1, every acquisition
    # call returned nothing, and smoke "passed" a notebook whose selection
    # stage never executed once. pool_size is system-owned (smoke economics),
    # so it grows to fit the paper-faithful protocol values — never the
    # reverse. The smoke gate still verifies the per-cell runtime budget.
    # Budget arithmetic uses the DELIVERED round count (which is the paper
    # value when the paper's count is small enough to keep), and the
    # bootstrap term covers core-set methods whose signature carries a
    # core_set_size default larger than initial_labeled — too-big pools are
    # the safe direction, too-small pools are the no-op-demo bug.
    delivered_rounds = (params.get("num_rounds") or {}).get("value") or smoke_num_rounds
    core_set_default = sig_defaults.get("core_set_size")
    bootstrap = max(
        initial_labeled_paper or 100,
        core_set_default if isinstance(core_set_default, (int, float)) else 0,
    )
    total_budget = bootstrap + delivered_rounds * delivered_batch_size
    pool_needed_for_budget = max(smoke_pool_default, total_budget)

    # Cross-check the smoke ratio against the matched paradigm's
    # `smoke_economics.max_budget_to_pool_ratio` threshold. For taxonomy-served
    # paradigms this is now a deterministic floor on pool_size; the advisory
    # branch remains only when no deterministic threshold is declared.
    comparison_paradigm = (spec.get("comparison") or {}).get("classification") or {}
    threshold = taxonomy.demo_scale_threshold(
        repo_root, paradigm_id=comparison_paradigm.get("id"), taxonomy=tax
    )
    pool_needed_for_ratio: int | None = None
    if threshold is not None and threshold > 0:
        pool_needed_for_ratio = math.ceil(total_budget / threshold)
    smoke_pool = max(
        pool_needed_for_budget,
        pool_needed_for_ratio if pool_needed_for_ratio is not None else 0,
    )
    ratio = total_budget / smoke_pool
    raised_for_budget = pool_needed_for_budget > smoke_pool_default
    raised_for_ratio = (
        pool_needed_for_ratio is not None and pool_needed_for_ratio > pool_needed_for_budget
    )

    if threshold is not None and threshold > 0:
        ratio_caveat = (
            f"This is at or below the taxonomy smoke_economics "
            f"`max_budget_to_pool_ratio` of {threshold} — diversity baselines "
            f"are still meaningfully tested at this scale."
        )
    elif threshold is not None:
        ratio_caveat = (
            f"The paradigm declares a nonpositive max_budget_to_pool_ratio ({threshold}); "
            "ignoring that malformed floor and interpreting diversity-baseline results "
            "with caution at smoke scale."
        )
    else:
        ratio_caveat = (
            "The paradigm does not declare a max_budget_to_pool_ratio; "
            "interpret diversity-baseline results with caution at smoke scale."
        )

    params["pool_size"] = {
        "value": smoke_pool,
        "source": "system_default",
        "paper_value": paper_value_str,
        "reasoning": (
            f"Paper uses the {paper_summary} as the unlabeled pool. We subsample to "
            f"{smoke_pool:,} to keep each round's acquisition-scoring pass over the whole "
            f"pool cheap (the pool is re-scored every round, so this is the dominant cost of "
            f"the acquisition loop)."
            + (
                f" Raised above the {smoke_pool_default:,} smoke default so the "
                f"paper-faithful initial labels plus the full acquisition budget "
                f"({total_budget:,}) fit inside the pool — a smaller pool would leave "
                f"the acquisition loop with nothing to acquire."
                if raised_for_budget else ""
            )
            + (
                f" Raised further to {smoke_pool:,} to satisfy the taxonomy "
                f"smoke_economics max_budget_to_pool_ratio floor ({threshold}); "
                f"a smaller pool would over-consume the unlabeled set and weaken "
                f"diversity-sensitive acquisition checks."
                if raised_for_ratio else ""
            )
            + f" Note: this is a smoke-scale default, not a runtime "
            f"estimate — the smoke gate is what actually verifies the notebook fits its "
            f"per-cell time budget. Pool ratio (total_budget / pool_size) ≈ {ratio:.2f}. "
            f"{ratio_caveat}"
        ),
    }


def _add_training_params(params: dict, spec: dict,
                         paper_text: str | None = None) -> None:
    cr = spec.get("critical_requirements") or {}
    training = cr.get("training") or {}
    paper_section = training.get("paper_section") or "Section 4 (Training)"
    is_image = _is_image_data(spec)

    # learning_rate — parse from spec; paper-faithful. Source decision lives
    # in _lr_provenance_entry (shared with the KD path).
    lr_str = str(training.get("learning_rate") or "")
    lr_value, note_suffix = _parse_learning_rate(lr_str, is_image)
    params["learning_rate"] = _lr_provenance_entry(
        lr_str, lr_value, note_suffix, paper_section, paper_text)

    # max_epochs + train_until_accuracy — provenance follows spec evidence
    # (the dropout-fix pattern, opposite direction: the old code stamped
    # source=paper unconditionally for the BALD-line 99% convention, which
    # fabricated provenance for papers that never state it).
    threshold = _training_threshold_evidence(spec, paper_text)
    protocol_clause = (
        "Paper trains to an explicit training-accuracy threshold with no "
        "stated epoch cap"
        if threshold is not None else
        "The paper states no epoch cap; per-round training stops at the "
        "convention accuracy threshold below"
    )
    params["max_epochs"] = {
        "value": 8,
        "source": "system_inferred",
        "reasoning": (
            f"{protocol_clause}. At smoke scale the labeled sets are tiny "
            "(≤ ~500 examples) and `train_until_accuracy` usually stops well "
            "before the cap; 8 is a low safety bound that keeps per-round "
            "training fast at smoke scale. Increase it for full-scale training."
        ),
    }

    if threshold is not None:
        value, section, quote = threshold
        params["train_until_accuracy"] = {
            "value": value,
            "source": "paper",
            "paper_section": section,
            "note": f"Paper-stated training threshold: {quote[:140]!r}.",
        }
    else:
        params["train_until_accuracy"] = {
            "value": 0.99,
            "source": "system_inferred",
            "reasoning": (
                "Deep-AL convention: BALD-line papers train each acquisition "
                "round to ≥99% training accuracy (Gal et al. 2017). This "
                "paper's spec states no explicit threshold, so the value is "
                "a field convention rather than a paper claim."
            ),
        }


def _add_provisional_family_training_params(params: dict, spec: dict) -> None:
    """Evidence-only training params for gap (provisional-pack) families.

    A provisional family has no derivation branch and no field conventions
    by definition, so the ONLY honest sources are the spec's own structured
    evidence. This reader takes numeric values from
    `critical_requirements.training` verbatim (source=paper, the analyzer's
    paper_section attached) and skips nulls and prose. It never injects a
    convention (the PA-D3 lesson: fabricated family defaults are worse than
    absent params) — an entry the analyzer did not structure numerically
    stays underivable and surfaces at the 2.x review, which is the correct
    pressure on the analyzer side (SRL 2026-07-05: the paper states values
    the analyzer left null; see the gap-family structuring rule in the
    analyzer contract)."""
    cr = spec.get("critical_requirements") or {}
    training = cr.get("training") or {}
    paper_section = training.get("paper_section") or "the paper's training section"

    def _numeric(v):
        if isinstance(v, bool):
            return None
        if isinstance(v, (int, float)):
            return v
        if isinstance(v, str):
            m = re.fullmatch(r"\s*(\d+(?:\.\d+)?(?:[eE]-?\d+)?)\s*", v)
            if m:
                f = float(m.group(1))
                return int(f) if f.is_integer() else f
        return None

    note = (
        "Gap-family derivation: read directly from the spec's structured "
        "training evidence. No family conventions exist for a provisional "
        "pack, so only paper-evidenced values are emitted."
    )
    for spec_key, param_name in (
        ("num_epochs", "max_epochs"),
        ("mc_samples", "mc_samples"),
    ):
        if param_name in params:
            continue
        val = _numeric(training.get(spec_key))
        if val is None:
            continue
        params[param_name] = {
            "value": val,
            "source": "paper",
            "paper_section": paper_section,
            "note": note,
        }
    if "learning_rate" not in params:
        lr_str = str(training.get("learning_rate") or "")
        if re.search(r"\d", lr_str):
            lr_value, note_suffix = _parse_learning_rate(
                lr_str, _is_image_data(spec))
            params["learning_rate"] = _lr_provenance_entry(
                lr_str, lr_value, note_suffix, paper_section, None)


def _add_model_params(
    params: dict, spec: dict, paradigm_id: str,
    tax: "taxonomy.Taxonomy | None" = None,
) -> None:
    cr = spec.get("critical_requirements") or {}
    model = cr.get("model") or {}
    arch_str = str(model.get("architecture") or "").lower()
    paper_section = model.get("paper_section") or "Section 4 (Model)"

    # hidden_dim — provenance follows the spec's own architecture evidence
    # (the 2026-06-10 BADGE halt's second finding: the old template claimed
    # "neither paper config matches exactly" off hardcoded "256"/"1024"
    # string checks, under-claiming BADGE's directly-stated MLP width).
    smoke_hidden = 256
    is_image = _is_image_data(spec)
    data_label = "image" if is_image else "non-image"
    dim_evidence = _mlp_dim_evidence(arch_str, want_image=is_image)
    if dim_evidence and dim_evidence[1]:
        params["hidden_dim"] = {
            "value": dim_evidence[0],
            "source": "paper",
            "paper_section": paper_section,
            "note": (
                f"Paper-stated MLP hidden dim for {data_label} data "
                f"(critical_requirements.model.architecture)."
            ),
        }
    elif dim_evidence:
        params["hidden_dim"] = {
            "value": dim_evidence[0],
            "source": "system_inferred",
            "reasoning": (
                f"The paper's {data_label}-data width belongs to a CNN "
                f"family per critical_requirements.model.architecture; we "
                f"run an MLP substitute at smoke scale, so the width is "
                f"adopted rather than paper-prescribed for this architecture."
            ),
        }
    else:
        # system_inferred fallback — node-served for migrated paradigms, with a
        # Python literal fallback only for provisional/future ids.
        node_default = taxonomy.load_model_default(paradigm_id, "hidden_dim", tax)
        if node_default is not None:
            value, template = node_default
            params["hidden_dim"] = {
                "value": value,
                "source": "system_inferred",
                "reasoning": template.format(value=value),
            }
        else:
            # This literal fires for paradigms whose taxonomy node serves no
            # hidden_dim default (bev-distill/KD 2026-07-01, F002), so it must
            # stay paradigm-neutral — never cite a convention from a paradigm
            # the paper doesn't belong to.
            params["hidden_dim"] = {
                "value": smoke_hidden,
                "source": "system_inferred",
                "reasoning": (
                    f"Paper states no data-type-keyed MLP width. We use "
                    f"hidden_dim={smoke_hidden} as the taxonomy-typical "
                    f"default width for smoke-scale networks."
                ),
            }

    # bayesian sub-paradigm: dropout_rate. Provenance is evidence-driven, in
    # both directions: A-001 (gbald 2026-05-26) was this value OVER-claimed as
    # paper-sourced when it was a taxonomy convention; the 2026-06-10 GBALD
    # halt was the mirror image — UNDER-claimed as system_inferred when the
    # fresh spec's specific_features carried explicit paper evidence
    # ("Dropout rate 0.5", Section 7.4), and the stage-2x reviewer correctly
    # refused the dishonest-conservative label. So: consult the spec's
    # structured evidence first; fall back to the convention label only when
    # the paper genuinely doesn't state a value.
    if "bayesian" in paradigm_id:
        evidence = _specific_feature_evidence(spec, "dropout")
        if evidence is not None:
            value, section, feature_text = evidence
            params["dropout_rate"] = {
                "value": value,
                "source": "paper",
                "paper_section": section,
                "note": (
                    f"Paper-stated dropout rate (spec specific_features: "
                    f"{feature_text!r})."
                ),
            }
        else:
            # system_inferred fallback — node-served for bayesian AL (Phase 5.7b),
            # with a Python literal fallback only for provisional/future ids.
            node_default = taxonomy.load_model_default(paradigm_id, "dropout_rate", tax)
            if node_default is not None:
                value, template = node_default
                params["dropout_rate"] = {
                    "value": value,
                    "source": "system_inferred",
                    "reasoning": template.format(value=value),
                }
            else:
                params["dropout_rate"] = {
                    "value": 0.5,
                    "source": "system_inferred",
                    "reasoning": (
                        "Bayesian-AL taxonomy recommends dropout in [0.25, 0.5] "
                        "(Gal & Ghahramani 2016); 0.5 is the taxonomy-typical "
                        "default. Paper does not specify a dropout rate."
                    ),
                }


# Smoke-default templates for known paradigm-extras, keyed by paradigm prefix.
# Each leaf entry: (smoke_value, reasoning_template). Template slots: {value},
# {paper_value}.
#
# LEGACY FALLBACK ONLY (Phase 5.7). This table is the read path for paradigms that
# are still `reserved` in the taxonomy SSOT. The active_learning extras
# (`mc_samples`, `core_set_size`, `batch_returns`) migrated to the AL nodes'
# `paradigm_extras` block in docs/ssot/taxonomies.yaml and are served by
# `taxonomy.load_paradigm_extra`, which `_lookup_known_extra` consults first; the
# knowledge_distillation entries migrated to the KD node in Phase 3.
#
# Keyed by paradigm because the same param name can mean different things in
# different paradigms (e.g., `mc_samples` is BALD's posterior-estimate sample
# count in bayesian-AL, but a different sampling concept in KD methods that
# include MC sampling). Resolution uses longest-prefix-match against
# `paradigm_id` (see `_lookup_known_extra`).
KNOWN_EXTRAS_BY_PARADIGM: dict[str, dict[str, tuple[object, str]]] = {}


def _lookup_known_extra(
    paradigm_id: str, name: str, tax: "taxonomy.Taxonomy | None" = None
) -> tuple[object, str] | None:
    """Resolve a smoke-default `(value, reasoning_template)` for (paradigm, name).

    Dual read path (Phase 5.7): a `populated` taxonomy node's `paradigm_extras`
    wins; otherwise fall back to the legacy hardcoded `KNOWN_EXTRAS_BY_PARADIGM`
    table (longest-prefix-match) for the still-`reserved` paradigms.

    Returns None if neither source owns `name`; callers treat that as "fall through
    to the signature-default path with source=spec_default".
    """
    node_entry = taxonomy.load_paradigm_extra(paradigm_id, name, tax)
    if node_entry is not None:
        return node_entry
    for prefix in sorted(KNOWN_EXTRAS_BY_PARADIGM, key=len, reverse=True):
        if paradigm_id.startswith(prefix):
            entry = KNOWN_EXTRAS_BY_PARADIGM[prefix].get(name)
            if entry is not None:
                return entry
    return None


# Map paradigm-extra names to their structured paper-truth location in the spec.
# Kwarg defaults in the signature are runtime conveniences (chosen by the analyzer
# for "the function defaults to a sensible value if the caller doesn't pass anything"),
# NOT the paper-stated value. paper_value belongs in a structured field; this map
# tells us where to look. For names not in this map, fall back to methodology
# contract / paper-map / paper-text evidence before treating the signature
# default as `spec_default`.
_PAPER_VALUE_SPEC_PATHS: dict[str, list[str]] = {
    # Bayesian-AL: paper specifies T (mc_samples) via critical_requirements.training
    "mc_samples": ["critical_requirements", "training", "mc_samples"],
    # Two-stage AL: paper specifies the preselection count b (batch_returns)
    # alongside batch_size (b') in data_setup. The b >= b' invariant is then
    # enforced on the runtime value by _reconcile_batch_acquisition_invariant.
    "batch_returns": ["critical_requirements", "data_setup", "batch_returns"],
}


def _specific_feature_evidence(
    spec: dict, keyword: str
) -> tuple[float, str, str] | None:
    """(value, paper_section, feature_text) when a
    critical_requirements.model.specific_features entry mentions `keyword`
    with a parseable numeric value AND a paper locator. Entries without a
    number (e.g. "Dropout layers active at inference") are skipped — only
    explicit paper-stated values qualify as paper provenance."""
    features = (
        (spec.get("critical_requirements") or {}).get("model") or {}
    ).get("specific_features") or []
    for feature in features:
        if not isinstance(feature, dict):
            continue
        text = str(feature.get("feature", ""))
        if keyword not in text.lower():
            continue
        section = str(feature.get("paper_section", "") or "")
        match = re.search(r"(\d+(?:\.\d+)?)", text)
        if match and section:
            return float(match.group(1)), section, text
    return None


_TRAIN_THRESHOLD_RE = re.compile(
    r"(?:train|training)[^.]{0,80}?accuracy[^.]{0,40}?"
    r"(?:≥|>=|>|exceeds?|above|reaches|of)\s*"
    r"(\d{2,3}(?:\.\d+)?\s*%|0?\.\d+)",
    re.IGNORECASE)


def _nearest_section_header(text: str, pos: int) -> str | None:
    """The markdown header most recently preceding `pos`, shaped into a
    paper locator ("## 4 EXPERIMENTS" → "Section 4 EXPERIMENTS")."""
    headers = re.findall(r"^#{1,6}\s+(.+)$", text[:pos], re.MULTILINE)
    if not headers:
        return None
    header = headers[-1].strip().rstrip("#").strip()
    if re.search(r"section|appendix|chapter|table", header, re.IGNORECASE):
        return header
    return f"Section {header}"


def _threshold_from_match(match: re.Match) -> float:
    raw = match.group(1).replace(" ", "")
    return float(raw.rstrip("%")) / 100.0 if raw.endswith("%") else float(raw)


def _training_threshold_evidence(
    spec: dict, paper_text: str | None = None
) -> tuple[float, str, str] | None:
    """(value, paper_section, quote) when the run's own evidence states a
    train-to-accuracy threshold. Evidence sources, in order:

    1. Spec strings (critical_requirements.training fields +
       model.specific_features) — carries the analyzer's section locator.
    2. The paper text itself, with the locator recovered from the nearest
       markdown header. Added after the 2026-06-10 BADGE stage-2x halt: the
       paper states "until training accuracy exceeds 99%" (Section 4) but
       the analyzer never extracted it into the spec, so spec-only evidence
       under-claimed and the reviewer correctly refused the label. The
       paper is the same ground truth the reviewer checks.

    No stated threshold anywhere (GBALD's shape: protocol says only "retrain
    from scratch", and its paper's 0.99s are figure-caption accuracy
    readings the pattern does not match) returns None — the convention value
    then ships as honest system_inferred, never as a paper claim (the
    fabricated-99% finding from the june9 audit, reproduced live by the
    quote-match probe (US-3) on 2026-06-10's fresh GBALD run)."""
    cr = spec.get("critical_requirements") or {}
    training = cr.get("training") or {}
    t_section = str(training.get("paper_section") or "")
    candidates: list[tuple[str, str]] = [
        (val, t_section) for key, val in training.items()
        if isinstance(val, str) and key != "paper_section"
    ]
    for feature in ((cr.get("model") or {}).get("specific_features") or []):
        if isinstance(feature, dict):
            candidates.append((str(feature.get("feature", "")),
                               str(feature.get("paper_section", "") or "")))
    for text, section in candidates:
        match = _TRAIN_THRESHOLD_RE.search(text)
        if match and section:
            return _threshold_from_match(match), section, text

    if paper_text:
        match = _TRAIN_THRESHOLD_RE.search(paper_text)
        if match:
            section = _nearest_section_header(paper_text, match.start())
            if section:
                start = paper_text.rfind(".", 0, match.start()) + 1
                end = paper_text.find(".", match.end())
                sentence = paper_text[start:end if end != -1
                                      else match.end()].strip()
                return _threshold_from_match(match), section, sentence[:160]
    return None


_DIM_FOR_DATA_RE = re.compile(
    r"(\d{2,5})\s+for\s+(?:the\s+)?[\w\- ]{0,24}?"
    r"(image|non-image|tabular|openml)",
    re.IGNORECASE)
_CNN_FAMILY_RE = re.compile(r"resnet|vgg|\bcnn\b|convolutional", re.IGNORECASE)


def _mlp_dim_evidence(arch_str: str,
                      want_image: bool) -> tuple[int, bool] | None:
    """(hidden_dim, owned_by_mlp) for the wanted data type from the spec's
    architecture string, or None when no data-type-keyed width exists.

    The ownership question is what discriminates the two concrete cases
    (the 2026-06-10 BADGE halt's hidden_dim finding vs GBALD): BADGE's arch
    gives the MLP itself "hidden dim 256 for image data, 1024 for OpenML
    tabular data" — the matched width IS the paper's MLP value. The May-era
    GBALD wording tied its image-side width to ResNet/VGG, so an MLP demo is
    a substitution and the convention label stays honest. Ownership: "mlp"
    appears before the match and no CNN-family name inside the match's own
    clause (up to the next , ; . ) or "and")."""
    for match in _DIM_FOR_DATA_RE.finditer(arch_str):
        is_image_assoc = match.group(2).lower() == "image"
        if is_image_assoc != want_image:
            continue
        tail = arch_str[match.end():]
        boundary = re.search(r"[,;.)]|\band\b", tail)
        clause = tail[:boundary.start()] if boundary else tail
        cnn_owns = bool(_CNN_FAMILY_RE.search(clause))
        mlp_before = "mlp" in arch_str[:match.start()].lower()
        return int(match.group(1)), (mlp_before and not cnn_owns)
    return None


def _lookup_structured_paper_value(name: str, spec: dict) -> object | None:
    """Return the paper-truth value for a paradigm-extra from a structured field, or None."""
    path = _PAPER_VALUE_SPEC_PATHS.get(name)
    if not path:
        return None
    cursor: object = spec
    for key in path:
        if not isinstance(cursor, dict) or key not in cursor:
            return None
        cursor = cursor[key]
    return cursor


def _lookup_scale_dependent_paper_entry(name: str, spec: dict) -> dict | None:
    """Return the matching scale-dependent hyperparameter entry, if it states a value."""
    cr = spec.get("critical_requirements") or {}
    for entry in cr.get("scale_dependent_hyperparameters") or []:
        if not isinstance(entry, dict):
            continue
        if entry.get("name") == name and entry.get("paper_value") is not None:
            return entry
    return None


def _values_equivalent(left: object, right: object) -> bool:
    if isinstance(left, bool) or isinstance(right, bool):
        # Temporal paper values are numeric quantities, never truth values.
        # Python considers True == 1, which would otherwise stamp a boolean
        # runtime default with source=paper.
        return False
    if not isinstance(left, (int, float)) or not isinstance(right, (int, float)):
        return left == right
    return float(left) == float(right)


def _dedupe_values(values: list[object]) -> list[object]:
    unique: list[object] = []
    for value in values:
        if not any(_values_equivalent(value, existing) for existing in unique):
            unique.append(value)
    return unique


def _lookup_contract_paper_value(
    name: str, spec: dict, other_params: tuple[str, ...] = (),
) -> object | None:
    """Recover explicit paper truth from the methodology contract if present.

    This is a compatibility path for already-generated Stage 1 specs. New
    Stage 1 outputs are schema-validated so paper-truth fields and methodology
    contract evidence cannot disagree, but a resumed run may carry an older
    complete sentinel. In that case, prefer the contract only when it states one
    unambiguous paper value for this parameter.
    """
    values = methodology_contract_paper_values_for_param(
        spec.get("methodology_replication_contract"),
        name,
        other_param_names=other_params,
    )
    unique = _dedupe_values(values)
    if len(unique) == 1:
        return unique[0]
    return None


def _glossary_anchor_names(spec: dict, name: str) -> tuple[str, ...]:
    """Other names under which the paper may state this param's value.

    The analyzer's param glossary links a paper symbol to the generated
    signature name via `aliases` (quote-floor-validated evidence, so this
    is a declared link, never a fuzzy guess). A generated `beta_sigma`
    whose glossary entry is {name: "beta", aliases: ["beta_sigma"]} may be
    stated in the paper as plain `beta = 3.0` — the ROMAN25 2026-07-21
    disclosure: four paper-stated values shipped labeled spec_default
    because the exact-name anchor could not bind the renamed symbol."""
    glossary = (spec.get("critical_requirements") or {}).get(
        "param_glossary") or []
    anchors: list[str] = []
    for g in glossary:
        if not isinstance(g, dict):
            continue
        names = [str(g.get("name"))] + [
            str(a) for a in (g.get("aliases") or [])]
        if name in names:
            anchors.extend(n for n in names if n != name and n not in anchors)
    return tuple(anchors)


def _lookup_textual_paper_value(
    name: str,
    *,
    paper_map: dict | None,
    paper_text: str | None,
    anchor_names: tuple[str, ...] = (),
) -> tuple[object, str, str | None] | None:
    """Recover one explicit paper value for a signature extra from paper text.

    This fallback is intentionally conservative: if paper-derived text contains
    multiple distinct values for the same parameter, return None and let the
    safer spec_default/stage-review path handle it. `anchor_names` extends the
    anchored scan with glossary-declared paper names for this param (see
    `_glossary_anchor_names`); values from all anchors pool into the same
    uniqueness check.
    """
    # Both arms scan whole source documents, so require the number to be
    # anchored to the param name — an unanchored phrase ("from 3 to 14")
    # elsewhere in the paper must not be attributed to this param (the
    # 2026-07-01 iDb-RRT goal_bias/delta_0/motion_planning=3 halt). The
    # paper_map arm walks elements individually so a binding can carry the
    # element's section locator (US-2a wants a paper location on
    # source=paper entries — bev-distill 2026-07-02 F004).
    scan_names = tuple(dict.fromkeys((name, *anchor_names)))
    map_values: list[object] = []
    section_by_value: dict[object, str] = {}
    elements = (paper_map or {}).get("elements")
    if isinstance(elements, list):
        for element in elements:
            if not isinstance(element, dict):
                continue
            section = element.get("section") or element.get("paper_section")
            for key in ("name", "description", "source_text", "pseudocode"):
                element_text = element.get(key)
                if not isinstance(element_text, str):
                    continue
                for scan_name in scan_names:
                    for extracted in extract_param_paper_values_from_text(
                            element_text, scan_name,
                            require_param_anchor=True):
                        map_values.append(extracted)
                        if isinstance(section, str) \
                                and extracted not in section_by_value:
                            section_by_value[extracted] = section
    unique_map_values = _dedupe_values(map_values)
    if len(unique_map_values) == 1:
        value = unique_map_values[0]
        return value, "paper_map", section_by_value.get(value)
    if len(unique_map_values) > 1:
        return None

    text_values: list[object] = []
    if paper_text:
        for scan_name in scan_names:
            extracted = extract_param_paper_values_from_text(
                paper_text, scan_name, require_param_anchor=True)
            if extracted:
                text_values.extend(extracted)

    unique = _dedupe_values(text_values)
    if len(unique) != 1:
        return None
    return unique[0], "paper.md", None


def _add_paradigm_extras(
    params: dict,
    spec: dict,
    paradigm_id: str,
    *,
    paper_map: dict | None = None,
    paper_text: str | None = None,
    tax: "taxonomy.Taxonomy | None" = None,
) -> None:
    """Parse pluggable_component.signature for kwarg defaults; add each as a param entry.

    For names with a structured paper-truth source (per _PAPER_VALUE_SPEC_PATHS),
    `paper_value` comes from the structured field. The signature default is a runtime
    convenience and is NOT used as paper_value when a structured source exists.

    KNOWN_EXTRAS lookup is paradigm-scoped so the same param name (e.g.,
    `mc_samples`) can have different reasoning across paradigms.
    """
    pc = (spec.get("comparison") or {}).get("pluggable_component") or {}
    sig = pc.get("signature") or ""
    defaults = _parse_signature_kw_defaults(sig)

    for name, signature_default in defaults.items():
        if paradigm_id.startswith("active_learning") and name in {"batch_output", "batch_outputs"}:
            # Historical analyzer drift: the AL selector output-count contract is
            # `batch_size`; a smaller smoke output count is represented by
            # params["batch_size"] with paper_value retained.
            continue
        # Skip params already added (e.g., if the signature redundantly lists batch_size)
        if name in params:
            continue

        # A role-typed protocol fact is authoritative for a bound temporal
        # carrier. In particular, paper_unspecified is a terminal fact rather
        # than permission to scan nearby notation (pdfgnn's T+1 -> K=1 bug).
        protocol_fact = evaluation_protocol_quantity_for_param(spec, name)

        # Prefer structured paper-truth over signature default for paper_value.
        # If a pre-existing spec has stale structured paper truth but the
        # methodology contract explicitly records a different paper value, use
        # the contract; Stage 1 validation rejects that conflict going forward.
        if protocol_fact is not None:
            protocol_status = protocol_fact.get("paper_value_status")
            protocol_value = protocol_fact.get("value")
            structured = (
                protocol_value if protocol_status == "paper_stated" else None
            )
            contract_value = None
            scale_entry = None
        else:
            structured = _lookup_structured_paper_value(name, spec)
            contract_value = _lookup_contract_paper_value(
                name, spec,
                other_params=tuple(n for n in defaults if n != name))
            scale_entry = _lookup_scale_dependent_paper_entry(name, spec)
        scale_value = scale_entry.get("paper_value") if scale_entry else None
        anchor_names = _glossary_anchor_names(spec, name)
        textual = (
            None
            if protocol_fact is not None
            else _lookup_textual_paper_value(
                name,
                paper_map=paper_map,
                paper_text=paper_text,
                anchor_names=anchor_names,
            )
        )
        # US-3b consistency: the textual scan is heuristic and can associate a
        # stray number near the param name (iDb-RRT extracted goal_bias=3, a
        # value that is not the paper's). Stamping it produces a "Paper states
        # name=value" claim the US-3b provenance validator then rejects as
        # fabricated, halting Stage 2.x. Drop a textual value that would not
        # survive that exact check, so the deriver never manufactures a claim
        # the validator will halt on — the US-3b analog of the value_findable
        # guard the learning-rate path uses. Glossary anchor names count: a
        # value stated under the paper's own name for this param is findable
        # under that name. Structured/contract/scale sources are explicit
        # paper-truth and are left untouched.
        if textual is not None and not any(
            paper_claim_findable(scan_name, textual[0], paper_text)
            for scan_name in (name, *anchor_names)
        ):
            textual = None
        textual_value = textual[0] if textual else None
        textual_source = textual[1] if textual else None
        textual_section = textual[2] if textual else None
        paper_value = (
            contract_value
            if contract_value is not None
            else structured
            if structured is not None
            else scale_value
            if scale_value is not None
            else textual_value
            if textual_value is not None
            else signature_default
        )

        known_extra = _lookup_known_extra(paradigm_id, name, tax)
        floor_entry = _smoke_economics_param_floor(paradigm_id, name, tax)
        if known_extra is not None:
            smoke_value, template = known_extra
            floor_note = ""
            if floor_entry is not None:
                floor, floor_reason = floor_entry
                floored = _apply_numeric_floor(smoke_value, floor)
                if floored != smoke_value:
                    smoke_value = floored
                    floor_note = (
                        f" Raised to the taxonomy smoke_economics floor "
                        f"({floor:g}) because {floor_reason}."
                    )
            params[name] = {
                "value": smoke_value,
                "source": "system_default",
                "paper_value": paper_value,
                "reasoning": (
                    template.format(value=smoke_value, paper_value=paper_value)
                    + floor_note
                ),
            }
        elif floor_entry is not None:
            floor, floor_reason = floor_entry
            floored = _apply_numeric_floor(signature_default, floor)
            if floored != signature_default:
                params[name] = {
                    "value": floored,
                    "source": "system_inferred",
                    "reasoning": (
                        f"The signature default for {name} is {signature_default!r}, "
                        f"below the taxonomy smoke_economics floor ({floor:g}). "
                        f"Using {floored!r} because {floor_reason}."
                    ),
                }
                continue
        elif (
            structured is not None
            or contract_value is not None
            or scale_entry is not None
            or textual_value is not None
        ):
            # Paradigm-extra with a structured paper-truth source — use it.
            if structured is not None:
                note = "Paper-stated value (from spec.critical_requirements structured field)."
                paper_section = None
            elif contract_value is not None:
                note = "Paper-stated value recovered from methodology_replication_contract."
                paper_section = None
            elif scale_entry is not None:
                note = (
                    scale_entry.get("description")
                    or "Paper-stated scale-dependent hyperparameter."
                )
                paper_section = scale_entry.get("paper_section")
            elif textual_value is not None:
                note = f"Paper-stated value recovered from {textual_source}."
                paper_section = textual_section
            if _values_equivalent(signature_default, paper_value):
                params[name] = {
                    "value": signature_default,
                    "source": "paper",
                    "note": note,
                }
                if paper_section:
                    params[name]["paper_section"] = paper_section
            else:
                # The "Paper states name=value" wording is exactly what the
                # US-3b probe re-checks against the paper text, so stamp it
                # only when the shared findability predicate blesses it.
                # When the paper states the value under a different symbol
                # (bayesian-active-learning: `batch_returns` is the paper's
                # `b`), an ungated claim here is unfixable by any analyzer
                # retry — the deriver re-stamps it on every re-derivation.
                if paper_claim_findable(name, paper_value, paper_text):
                    claim = f"Paper states {name}={paper_value!r}"
                else:
                    claim = (
                        f"The spec's structured paper truth records "
                        f"{paper_value!r} for this parameter (the paper may "
                        f"name it by another symbol)"
                    )
                params[name] = {
                    "value": signature_default,
                    "source": "system_default",
                    "paper_value": paper_value,
                    "reasoning": (
                        f"{claim}, but the generated "
                        f"method signature default is {signature_default!r}. Treating "
                        "the signature default as the smoke/runtime value while "
                        "preserving the paper value for audit."
                    ),
                }
        else:
            # No structured paper source and no KNOWN_EXTRAS entry. The signature
            # kwarg default is a runtime convenience, NOT a paper-stated value
            # (cf. the note above _PAPER_VALUE_SPEC_PATHS). Tag accordingly so
            # downstream consumers don't claim paper attribution.
            params[name] = {
                "value": signature_default,
                "source": "spec_default",
                "reasoning": (
                    f"Value from the pluggable_component signature default "
                    f"({name}={signature_default!r}). No structured spec source, "
                    f"methodology-contract value, or anchored paper statement "
                    f"of {name} was found, so it is not asserted as a paper "
                    f"value. If the paper does state one, add a structured "
                    f"source to the spec or a clear {name}=<value> evidence "
                    f"entry to paper_map.json."
                ),
            }


# ---------------------------------------------------------------------------
# Paradigm-declared parameter derivation (plan item 9, 2026-07-21).
#
# Root cause this closes (DomIndOnto + fedavg, 2026-07-21 overnight): the
# taxonomy node / provisional pack declares stage_2x_params review checks
# ("all configuration paths must have a documented source", "u = nE/(KB)
# must be derived and documented") that this deterministic script had no
# way to see, so the reviewer failed params.json for entries the producer
# could never have emitted, with no auto-resolution path. The node's
# `params_derivation` block is the machine-readable half of that review
# focus; this pass consumes it generically — no paradigm names in code.
# ---------------------------------------------------------------------------

_FORMULA_BINOPS = {
    ast.Add: lambda a, b: a + b,
    ast.Sub: lambda a, b: a - b,
    ast.Mult: lambda a, b: a * b,
    ast.Div: lambda a, b: a / b,
    ast.FloorDiv: lambda a, b: a // b,
    ast.Mod: lambda a, b: a % b,
    ast.Pow: lambda a, b: a ** b,
}


def _safe_eval_formula(formula: str, values: dict[str, float]) -> object | None:
    """Evaluate a declared arithmetic formula over named inputs.

    Supports numbers, names bound in `values`, + - * / // % **, and
    unary +/-. Anything else (calls, attributes, unknown names) returns
    None — a pack can only declare arithmetic, never code. An integral
    float result collapses to int so params.json stays clean."""
    try:
        tree = ast.parse(formula, mode="eval")
    except (SyntaxError, ValueError):
        return None

    def _eval(node: ast.AST) -> float:
        if isinstance(node, ast.Expression):
            return _eval(node.body)
        if isinstance(node, ast.BinOp) and type(node.op) in _FORMULA_BINOPS:
            return _FORMULA_BINOPS[type(node.op)](_eval(node.left),
                                                  _eval(node.right))
        if isinstance(node, ast.UnaryOp) and isinstance(
                node.op, (ast.UAdd, ast.USub)):
            operand = _eval(node.operand)
            return operand if isinstance(node.op, ast.UAdd) else -operand
        if isinstance(node, ast.Constant) and isinstance(
                node.value, (int, float)) and not isinstance(node.value, bool):
            return node.value
        if isinstance(node, ast.Name) and node.id in values:
            return values[node.id]
        raise ValueError(f"unsupported formula node: {ast.dump(node)[:60]}")

    try:
        result = _eval(tree)
    except (ValueError, ZeroDivisionError, OverflowError, KeyError):
        return None
    if isinstance(result, float) and result.is_integer():
        return int(result)
    return result


def _resolve_derivation_input(ref: object, params: dict, spec: dict) -> object | None:
    """Resolve one declared formula input to a number.

    Two reference forms: ``params.<name>`` reads an already-derived
    param's value; ``spec.<dotted.path>`` reads a structured spec field.
    Non-numeric or missing targets resolve to None — the caller documents
    the gap honestly instead of guessing."""
    ref = str(ref or "")
    target: object = None
    if ref.startswith("params."):
        entry = params.get(ref[len("params."):])
        target = entry.get("value") if isinstance(entry, dict) else None
    elif ref.startswith("spec."):
        target = spec
        for part in ref[len("spec."):].split("."):
            target = target.get(part) if isinstance(target, dict) else None
    if isinstance(target, (int, float)) and not isinstance(target, bool):
        return target
    return None


def _lookup_declared_paper_value(
    name: str,
    spec: dict,
    *,
    paper_map: dict | None = None,
    paper_text: str | None = None,
) -> tuple[object, str | None, str] | None:
    """One unambiguous paper value for a declared param, or None.

    The same evidence ladder `_add_paradigm_extras` walks for signature
    extras (methodology contract, structured spec paths, scale-dependent
    entries, then the anchored textual scan with the US-3b findability
    guard), packaged for the params_derivation pass: a pack/node that
    declares a config_path name the signature does not carry previously
    got NO paper-truth consultation at all, so SRL 2026-07-21 shipped
    episodes=null while the run's own methodology contract recorded the
    paper's 3000 (stage-2x F002). Returns (value, paper_section_or_None,
    origin_label)."""
    protocol_fact = evaluation_protocol_quantity_for_param(spec, name)
    if protocol_fact is not None:
        status = protocol_fact.get("paper_value_status")
        value = protocol_fact.get("value")
        section = protocol_fact.get("paper_section")
        if status == "paper_stated":
            return value, section, "comparison.evaluation_protocol"
        # Deliberately return a present result with no value. Callers can
        # distinguish an explicit paper-unspecified fact from no structured
        # fact and must not fall through to glossary/prose number mining.
        return None, section, "comparison.evaluation_protocol (paper-unspecified)"

    contract_value = _lookup_contract_paper_value(name, spec)
    if contract_value is not None:
        return contract_value, None, "the spec's methodology contract"
    structured = _lookup_structured_paper_value(name, spec)
    if structured is not None:
        return structured, None, "the spec's structured requirements"
    scale_entry = _lookup_scale_dependent_paper_entry(name, spec)
    if scale_entry is not None:
        return (scale_entry.get("paper_value"),
                scale_entry.get("paper_section"),
                "the spec's scale-dependent hyperparameters")
    anchor_names = _glossary_anchor_names(spec, name)
    textual = _lookup_textual_paper_value(
        name, paper_map=paper_map, paper_text=paper_text,
        anchor_names=anchor_names,
    )
    if textual is not None and any(
        paper_claim_findable(scan_name, textual[0], paper_text)
        for scan_name in (name, *anchor_names)
    ):
        return textual[0], textual[2], textual[1]
    return None


def _apply_params_derivation(
    params: dict,
    spec: dict,
    paradigm_id: str,
    tax: "taxonomy.Taxonomy | None" = None,
    *,
    paper_map: dict | None = None,
    paper_text: str | None = None,
) -> None:
    """Emit / drop the paradigm's declared parameter entries.

    Runs after the legacy and signature producers so `suppress` can drop
    anything those passes added and so `derived_statistic` inputs can read
    their final values.  The precedence-safe generic carrier consumer runs
    immediately afterward; it reads this declaration again so an explicit
    suppression still outranks a populated weak carrier.
    Existing entries keep their value; a declared name an earlier pass
    emitted as a bare signature default (source=spec_default) is
    re-labeled with the declaration's own provenance — the declaration
    says the paper defines this parameter, so the boilerplate "no paper
    statement was found" reasoning misrepresents the evidence (SRL
    2026-07-21 stage-2x F003)."""
    declared = taxonomy.load_params_derivation(paradigm_id, tax)
    for name, entry in sorted(declared.items()):
        kind = str(entry.get("kind") or "")
        if kind == "suppress":
            if name in params:
                params.pop(name)
            continue
        prose = str(entry.get("reasoning") or "").strip()
        paper_section = entry.get("paper_section")
        if name in params:
            existing = params[name]
            if (kind == "config_path" and isinstance(existing, dict)
                    and existing.get("source") == "spec_default"):
                existing["source"] = "system_inferred"
                existing["reasoning"] = (f"{prose} " if prose else "") + (
                    f"The runtime value comes from the pluggable-component "
                    f"signature default ({name}="
                    f"{existing.get('value')!r}). The paradigm declares "
                    f"this name as a required configuration path and the "
                    f"paper defines the parameter without pinning the "
                    f"exact runtime value, so the default is adopted as a "
                    f"reasonable choice within the paper's constraints."
                )
                if paper_section and not existing.get("paper_section"):
                    existing["paper_section"] = str(paper_section)
            continue
        if kind == "config_path":
            demo_value = entry.get("demo_value")
            found = _lookup_declared_paper_value(
                name, spec, paper_map=paper_map, paper_text=paper_text)
            used = bool(entry.get("used_in_notebook", demo_value is not None))
            if (demo_value is None and found is not None
                    and found[0] is not None
                    and value_findable_in_paper(found[0], paper_text)):
                # The paper states the value and the run has no demo
                # override — carry paper truth instead of null (F002).
                value, found_section, origin = found
                new = {
                    "value": value,
                    "source": "paper",
                    "note": (f"{prose} " if prose else "") + (
                        f"Paper-stated value recovered from {origin}. "
                        "Pipeline configuration required by this "
                        "paradigm's pluggable component (declared in the "
                        "taxonomy node's params_derivation)."
                    ),
                }
                section = paper_section or found_section
                if section:
                    new["paper_section"] = str(section)
            else:
                new = {
                    "value": demo_value,
                    "source": "system_inferred",
                    "reasoning": (f"{prose} " if prose else "") + (
                        "Pipeline configuration required by this paradigm's "
                        "pluggable component (declared in the taxonomy node's "
                        "params_derivation), not an ML hyperparameter."
                    ),
                }
                if found is not None and found[0] is not None:
                    # Paper truth exists but could not be verified against
                    # the paper text, or a demo override applies — preserve
                    # it for audit instead of dropping it.
                    new["paper_value"] = found[0]
                    new["reasoning"] += (
                        f" The run's evidence ({found[2]}) records the "
                        f"paper value {found[0]!r}; it is preserved in "
                        f"paper_value for audit."
                    )
                if paper_section:
                    new["paper_section"] = str(paper_section)
            if not used:
                new["used_in_notebook"] = False
                new["unused_reason"] = (
                    "The demo supplies this configuration at runtime; the "
                    "entry exists so the required configuration path has a "
                    "documented source."
                )
            params[name] = new
        elif kind == "derived_statistic":
            formula = str(entry.get("formula") or "").strip()
            inputs = entry.get("inputs") or {}
            resolved: dict[str, float] = {}
            missing: list[str] = []
            for sym, ref in sorted(inputs.items()):
                value = _resolve_derivation_input(ref, params, spec)
                if value is None:
                    missing.append(f"{sym} ({ref})")
                else:
                    resolved[str(sym)] = value
            computed = (_safe_eval_formula(formula, resolved)
                        if formula and not missing else None)
            if computed is not None:
                bound = ", ".join(f"{k}={v!r}" for k, v in sorted(resolved.items()))
                detail = (f"Derived statistic declared by the paradigm: "
                          f"{name} = {formula} = {computed!r} (with {bound}).")
            else:
                gap = ("inputs unavailable: " + ", ".join(missing)
                       if missing else "the formula could not be evaluated")
                detail = (f"Derived statistic declared by the paradigm "
                          f"({name} = {formula}) could not be computed "
                          f"deterministically — {gap}. Documented so the "
                          f"required derivation is auditable.")
            new = {
                "value": computed,
                "source": "system_inferred",
                "reasoning": (f"{prose} " if prose else "") + detail,
                "used_in_notebook": False,
                "unused_reason": (
                    "Derived diagnostic statistic documented for review and "
                    "audit; not a runtime argument."
                ),
            }
            if paper_section:
                new["paper_section"] = str(paper_section)
            params[name] = new
        # Unknown kinds are dropped by load_params_derivation's shape
        # filter or ignored here; the taxonomy/proposal lints reject them
        # at authoring time.


def _apply_evaluation_protocol_params(params: dict, spec: dict) -> None:
    """Make typed temporal protocol facts authoritative on bound params.

    Stage-1 strict validation has already checked each ``parameter_name``
    against the taxonomy's independent ``protocol_role`` declaration. This
    pass runs after legacy producers but before measured-axis reconciliation,
    so that consumer can require the complete typed tuple rather than trusting
    a coincidentally matching params key. It never creates validation/test
    params: spec-only quantities have ``parameter_name=null`` and are rendered
    from the protocol itself.
    """
    comparison = spec.get("comparison") or {}
    protocol = comparison.get("evaluation_protocol") or {}
    quantities = protocol.get("quantities") or []
    for index, quantity in enumerate(quantities):
        if not isinstance(quantity, dict):
            continue
        name = quantity.get("parameter_name")
        if not isinstance(name, str) or not name:
            continue
        entry = params.get(name)
        if not isinstance(entry, dict):
            continue

        role = str(quantity.get("role") or "")
        status = str(quantity.get("paper_value_status") or "")
        paper_value = quantity.get("value")
        unit = str(quantity.get("unit") or "")
        granularity = quantity.get("granularity")
        evidence = str(quantity.get("evidence_quote") or "")
        section = str(quantity.get("paper_section") or "")
        element_ids = [
            str(item) for item in (quantity.get("paper_element_ids") or [])
        ]
        axis_evidence = str(quantity.get("axis_evidence_quote") or "")
        axis_section = str(quantity.get("axis_paper_section") or "")
        axis_element_ids = [
            str(item)
            for item in (quantity.get("axis_paper_element_ids") or [])
        ]

        entry.update({
            "protocol_role": role,
            "protocol_value": paper_value,
            "protocol_unit": unit,
            "protocol_granularity": granularity,
            "protocol_axis_says": axis_evidence,
            "protocol_axis_section": axis_section,
            "protocol_axis_element_ids": axis_element_ids,
            "paper_value_status": status,
            "paper_says": evidence,
            "paper_section": section,
            "paper_element_ids": element_ids,
        })

        if status == "paper_unspecified":
            entry["source"] = "system_inferred"
            entry.pop("paper_value", None)
            entry.pop("note", None)
            entry["reasoning"] = (
                f"The paper defines the {role} role on a {unit} axis "
                f"(granularity {granularity!r}) but leaves its numeric value "
                f"unspecified. Runtime {name}={entry.get('value')!r} is a "
                "system-owned demo choice; nearby notation and validation or "
                "test spans are not paper values for this carrier."
            )
            continue

        if status != "paper_stated":
            # The schema rejects unknown statuses. Defensive no-op keeps a
            # direct legacy caller from manufacturing provenance.
            continue

        path = f"comparison.evaluation_protocol.quantities[{index}]"
        try:
            paper_steps = protocol_quantity_steps(quantity, path=path)
        except ProtocolAxisUnresolved as exc:
            raise ValueError(
                "evaluation protocol carrier cannot bind a physical paper "
                f"quantity to step-count parameter {name!r}: {exc}"
            ) from exc

        if entry.get("value") is None:
            entry["value"] = paper_steps
        runtime_value = entry.get("value")
        runtime_is_numeric = (
            isinstance(runtime_value, (int, float))
            and not isinstance(runtime_value, bool)
            and runtime_value > 0
        )
        if (
            runtime_is_numeric
            and _values_equivalent(runtime_value, paper_steps)
        ):
            entry["source"] = "paper"
            entry.pop("paper_value", None)
            entry.pop("reasoning", None)
            entry["note"] = (
                f"Paper-stated {path} role={role!r} quantity "
                f"{paper_value!r} {unit} / {granularity!r} {unit}/step = "
                f"{paper_steps} runtime steps; bound through the exact "
                "parameter_name carrier."
            )
        else:
            entry["source"] = "system_default"
            entry["paper_value"] = paper_steps
            entry.pop("note", None)
            entry["reasoning"] = (
                f"The paper states {path} role={role!r} quantity "
                f"{paper_value!r} {unit} / {granularity!r} {unit}/step = "
                f"{paper_steps} runtime steps, while the demo invokes "
                f"{name}={runtime_value!r}. The converted paper step count "
                "remains separate from the system-owned runtime choice."
            )


def _glossary_param_name(entry: dict) -> str | None:
    """The runtime param name a glossary entry should produce, or None.

    `aliases` is documented as "other names derivation may use for the same
    parameter", so an alias is the derivation-facing name and `name` is the
    paper's own symbol ("P", "similarity cutoff"). Prefer the first alias
    that is a legal identifier, fall back to `name` when it is one, and skip
    the entry when neither can name a keyword argument.
    """
    for candidate in (entry.get("aliases") or []):
        text = str(candidate).strip()
        if text.isidentifier() and not keyword.iskeyword(text):
            return text
    name = str(entry.get("name") or "").strip()
    if name.isidentifier() and not keyword.iskeyword(name):
        return name
    return None


def _add_glossary_value_params(params: dict, spec: dict) -> list[str]:
    """Create params from glossary entries that state a value (R2C-055).

    The slip-tolerant second source, gap-family only. `param_glossary` is a
    MEANING surface whose documented job is stamping `paper_says` onto params
    that already exist, so it can annotate and never create. When the
    analyzer routes a paper-stated value there instead of into the typed lane
    that owns it, the value reaches no consumer and the run halts several
    stages later on a pack check demanding the param (pdfgnn 2026-08-05,
    similarity_cutoff = 0.95 from Appendix A Table 5).

    Scoped to the gap path deliberately: a committed family has a derivation
    branch with conventions this could contradict, and no existing spec
    carries the field at all, so a universal read would be speculative.

    Never clobbers a param another reader produced, under the entry's own
    name or any of its aliases — the typed lane stays the destination and
    this is only the backstop.
    """
    glossary = (spec.get("critical_requirements") or {}).get(
        "param_glossary") or []
    added: list[str] = []
    for entry in glossary:
        if not isinstance(entry, dict):
            continue
        value = entry.get("paper_value")
        if value is None or isinstance(value, bool):
            continue
        known = {str(entry.get("name") or "")}
        known.update(str(a) for a in (entry.get("aliases") or []))
        if known & set(params):
            continue
        name = _glossary_param_name(entry)
        if not name:
            continue
        if isinstance(value, float) and value.is_integer():
            value = int(value)
        params[name] = {
            "value": value,
            "source": "paper",
            "paper_section": entry.get("paper_section") or "param_glossary",
            "note": (
                "Paper-stated value recovered from the spec's parameter "
                "glossary. The typed parameter lane that owns this value "
                "carried no entry for it, so the glossary's stated value is "
                "used rather than dropping a value the paper gives."
            ),
        }
        added.append(name)
    return added


def _drop_shadowed_glossary_params(params: dict, spec: dict,
                                   glossary_added: list[str]) -> list[str]:
    """One param per glossary ENTRY, not one per name (night3 finding).

    The glossary value arm is the backstop and runs before the
    signature-extras and derivation passes, so a later primary producer
    can create the SAME concept under another of the entry's names
    (night3: the backstop made `similarity_cutoff` from the entry's
    alias, the pluggable-signature pass then made `similarity_threshold`
    from the entry's name, and params.json carried the one 0.95 twice).
    When that happens the backstop copy yields — the typed lane stays
    the destination — but ONLY when the values agree: a disagreement is
    a real conflict that must stay visible on both entries rather than
    be resolved by silently dropping one."""
    glossary = (spec.get("critical_requirements") or {}).get(
        "param_glossary") or []
    dropped: list[str] = []
    for entry in glossary:
        if not isinstance(entry, dict):
            continue
        known = {str(entry.get("name") or "")}
        known.update(str(a) for a in (entry.get("aliases") or []))
        present = sorted(known & set(params))
        if len(present) < 2:
            continue
        for name in present:
            if name not in glossary_added:
                continue
            value = (params[name] or {}).get("value")
            siblings = [p for p in present
                        if p != name and p not in glossary_added]
            if siblings and all(
                    (params[s] or {}).get("value") == value
                    for s in siblings):
                del params[name]
                dropped.append(name)
    return dropped


def _consume_parameter_carriers(
    params: dict,
    spec: dict,
    params_derivation: dict[str, dict],
) -> dict[str, str]:
    """Consume remaining paper-value carriers after named authorities.

    Signature extras and ``params_derivation`` are the stronger runtime
    producers.  This final pass therefore creates only a still-unclaimed exact
    identity.  If a stronger entry already owns a carrier name or declared
    alias, its runtime value and source stay intact; distinct paper truth is
    retained separately in ``paper_value``.

    Identity is deliberately narrow: the exact boundary-stripped carrier name
    or a declared glossary alias.  Numeric equality, token similarity, and
    spelling normalization never join parameters.  Fresh strict validation
    rejects populated unnameable carriers, and this consumer raises the same
    class defensively for direct callers that bypass that gate.

    Returns the exact suppression reasons that prevented otherwise populated
    carriers from being emitted.  The taxonomy declaration remains the
    durable record; the return value gives tests and callers a deterministic
    audit surface without changing ``params.json`` for existing families.
    """
    cr = spec.get("critical_requirements") or {}
    suppressed_by_name = {
        str(name).strip(): str(entry.get("reason") or "").strip()
        for name, entry in params_derivation.items()
        if isinstance(entry, dict) and entry.get("kind") == "suppress"
    }
    suppressed: dict[str, str] = {}

    carriers: list[tuple[str, dict, list[str], object]] = []
    for entry in cr.get("scale_dependent_hyperparameters") or []:
        if not isinstance(entry, dict):
            continue
        value = entry.get("paper_value")
        if value is None or isinstance(value, bool):
            continue
        label = str(entry.get("name") or "").strip()
        carriers.append(("scale-dependent", entry, [label], value))
    for entry in cr.get("param_glossary") or []:
        if not isinstance(entry, dict):
            continue
        value = entry.get("paper_value")
        if value is None or isinstance(value, bool):
            continue
        identities: list[str] = []
        for raw in (entry.get("name"), *(entry.get("aliases") or [])):
            label = str(raw or "").strip()
            if label and label not in identities:
                identities.append(label)
        carriers.append(("glossary", entry, identities, value))

    for carrier_kind, entry, identities, paper_value in carriers:
        if carrier_kind == "glossary":
            canonical = _glossary_param_name(entry)
        else:
            candidate = identities[0] if identities else ""
            canonical = (
                candidate
                if candidate.isidentifier() and not keyword.iskeyword(candidate)
                else None
            )
        if canonical is None:
            if carrier_kind == "scale-dependent":
                # Legacy scale-lane entries may use a descriptive paper label.
                # That older lane has no alias field and retains its existing
                # family-specific consumers.  The fresh unnameable-carrier
                # refusal is scoped to the glossary, whose contract explicitly
                # offers derivation aliases.
                continue
            raise ValueError(
                "paper_value_carrier_unnameable: populated "
                f"{carrier_kind} carrier has stripped identities "
                f"{identities!r}, but none is a valid Python identifier; "
                "declare an exact runtime name or glossary alias"
            )

        suppression_names = [
            name for name in identities if name in suppressed_by_name
        ]
        if suppression_names:
            suppression_name = suppression_names[0]
            suppressed[canonical] = suppressed_by_name[suppression_name]
            continue

        present = [name for name in identities if name in params]
        if not present:
            if isinstance(paper_value, float) and paper_value.is_integer():
                paper_value = int(paper_value)
            if carrier_kind == "glossary":
                section = entry.get("paper_section") or "param_glossary"
                note = (
                    "Paper-stated value recovered from the spec's parameter "
                    "glossary. No stronger signature or taxonomy producer "
                    "claimed its exact declared identity."
                )
            else:
                section = (
                    entry.get("paper_section")
                    or "scale_dependent_hyperparameters"
                )
                note = (
                    entry.get("description")
                    or "Scale-dependent hyperparameter from the paper."
                )
            params[canonical] = {
                "value": paper_value,
                "source": "paper",
                "paper_section": section,
                "note": note,
            }
            continue

        owner_name = canonical if canonical in present else present[0]
        owner = params.get(owner_name)
        if not isinstance(owner, dict):
            raise ValueError(
                f"parameter carrier identity {owner_name!r} is already "
                "claimed by a non-object params entry"
            )
        runtime_value = owner.get("value")
        existing_paper_value = owner.get("paper_value")
        if existing_paper_value is not None:
            # The named producer has already resolved both runtime authority
            # and its distinct paper truth.  A weak carrier cannot replace
            # either field, even when it is stale or contradictory.  Fresh
            # authoring gates own cross-carrier contradictions; legacy specs
            # retain the stronger result byte-for-byte.
            continue
        if not _values_equivalent(runtime_value, paper_value):
            owner["paper_value"] = paper_value

    return suppressed


def _apply_param_glossary(params: dict, spec: dict) -> None:
    """Stamp the paper's own per-parameter explanation onto derived
    entries (param-glossary design 2026-07-21, the researcher's ask second half).

    Deterministic carriage only: the analyzer extracted
    `critical_requirements.param_glossary` (verbatim quotes the spec
    validator checked against paper.md); this pass copies each entry's
    quote onto the derived param matching its name or a declared alias.
    Exact name match only — a miss is honest, never fuzzy. A glossary
    paper_section fills in ONLY when the entry has none of its own.

    A glossary hit on a bare signature-default entry also re-labels it:
    spec_default's boilerplate says NO paper statement of the parameter
    was found, but the glossary quote IS a paper statement of the
    parameter (its meaning, without an exact value), so the honest label
    is system_inferred — a default chosen within the paper's stated
    meaning (SRL 2026-07-21 stage-2x F003, the epsilon_f shape)."""
    glossary = (spec.get("critical_requirements") or {}).get(
        "param_glossary") or []
    by_name: dict[str, dict] = {}
    for g in glossary:
        if not isinstance(g, dict) or not g.get("meaning_quote"):
            continue
        by_name[str(g.get("name"))] = g
        for alias in g.get("aliases") or []:
            by_name.setdefault(str(alias), g)
    for name, entry in params.items():
        g = by_name.get(name)
        if g is None or not isinstance(entry, dict):
            continue
        entry["paper_says"] = str(g["meaning_quote"])
        if not entry.get("paper_section") and g.get("paper_section"):
            entry["paper_section"] = str(g["paper_section"])
        if entry.get("source") == "spec_default":
            entry["source"] = "system_inferred"
            entry["reasoning"] = (
                f"The runtime value comes from the pluggable-component "
                f"signature default ({name}={entry.get('value')!r}). The "
                f"paper defines this parameter (see paper_says) without "
                f"pinning the exact runtime value, so the default is "
                f"adopted as a reasonable choice within the paper's "
                f"stated meaning."
            )


def _reconcile_protocol_params_with_bundle_axis(
    params: dict, spec: dict, run_dir: Path | None,
) -> None:
    """Make the deterministic deriver obey the measured axis authority.

    This consumes only the role-typed metadata stamped immediately before it.
    A compatible measured axis may authorize adoption of a smaller explicit
    forecast-call horizon. Missing or incompatible cadence produces no
    decision; it is not evidence for an insufficient-data trade-off.
    """
    if run_dir is None:
        return
    manifest = load_bundle_manifest(run_dir)
    if manifest is None:
        return
    for shrink in protocol_axis_param_shrinks(spec, params, manifest):
        entry = params[shrink.param]
        entry["paper_value"] = shrink.paper_value
        if shrink.feasible:
            entry["value"] = shrink.paper_value
            entry["source"] = "paper"
            entry.pop("paper_value", None)
            entry.pop("reasoning", None)
            entry["note"] = (
                "Converted paper forecast-call horizon adopted because every "
                f"role-compatible boundary is satisfied ({shrink.arithmetic()})."
            )
            continue
        prior = str(entry.get("reasoning") or entry.get("note") or "").strip()
        entry["source"] = "system_default"
        entry.pop("note", None)
        entry["reasoning"] = " ".join(
            part for part in (prior, shrink.shrink_reason()) if part)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _mark_unused_params(params: dict, spec: dict) -> None:
    """Flag params the spec implies but the method's runtime path doesn't actually use.

    Provenance + usage together means the user can audit the params dict at a glance
    and never wonder "why is this defined but never referenced?"

    Heuristics (paradigm-specific; extend as new paradigms / methods land):

    - **`initial_labeled` is unused when the method has its own bootstrap.** Detected
      by `core_set_size` being a kwarg in the pluggable_component signature (GBALD-like).
      The bootstrap is `construct_core_set(core_set_size)`; there's no separate
      uniform-random initial selection.
    """
    pc = (spec.get("comparison") or {}).get("pluggable_component") or {}
    sig = pc.get("signature") or ""
    sig_kwargs = _parse_signature_kw_defaults(sig)

    # Heuristic 1: GBALD-like methods bootstrap via construct_core_set, not initial_labeled.
    if "core_set_size" in sig_kwargs and "initial_labeled" in params:
        params["initial_labeled"]["used_in_notebook"] = False
        params["initial_labeled"]["unused_reason"] = (
            "This method bootstraps the labeled set via `construct_core_set(core_set_size)` "
            "(Stage 1 of the algorithm) — the initial labeled set IS the core-set. The "
            "spec's data_setup mentions `initial_labeled` for paradigm-fixed reasons but "
            "the algorithm's runtime path doesn't reference it. Kept in the dict for spec "
            "fidelity; ignore at runtime."
        )


def _source_mentions_name(path: Path, name: str) -> bool:
    if not path.is_file():
        return False
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except SyntaxError:
        return False
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id == name:
            return True
        if isinstance(node, ast.arg) and node.arg == name:
            return True
    return False


def _mark_runtime_unused_model_params(params: dict, run_dir: Path) -> None:
    """Mark generated model knobs unused when the runtime cannot consume them.

    The deriver always knows the spec/taxonomy side; at Stage 2.x it also has
    the generated model/training code. If a model-shape param such as
    `hidden_dim` is not accepted or read by that runtime, it must not remain a
    live notebook knob.
    """
    method_dir = run_dir / "method"
    runtime_files = [method_dir / "model.py", method_dir / "training.py"]
    reasons = {
        "hidden_dim": (
            "The generated model/training runtime does not expose or read a "
            "`hidden_dim` argument. Kept in params for taxonomy/spec audit, "
            "but the notebook must not pass it into runtime calls."
        ),
        "dropout_rate": (
            "The generated model/training runtime does not expose or read a "
            "`dropout_rate` argument. Kept in params for taxonomy/spec audit, "
            "but the notebook must not pass it into runtime calls."
        ),
    }
    for name, reason in reasons.items():
        if name not in params:
            continue
        if any(_source_mentions_name(path, name) for path in runtime_files):
            continue
        params[name]["used_in_notebook"] = False
        params[name]["unused_reason"] = reason


_METHOD_CONSUMER_SCAN_NAMES = (
    "initial_labeled",
    "batch_size",
    "total_budget",
    "num_rounds",
    "batch_returns",
    "pool_size",
)


def _mark_unused_by_method(params: dict, spec: dict, run_dir: Path) -> None:
    """Disclose data-setup params the method itself never consumes.

    per-dataset-value-binding-design.md item 5: derivation, not reviewer
    mercy, marks the GBALD-class case (initial_labeled carried as a paper
    claim while the framework's own core-set bootstrap ignores it). Consumer
    surfaces the deriver can check deterministically at 2.x: the pluggable
    component's signature and the generated method package's sources. Purely
    a disclosure — no value, source, or label changes."""
    method_dir = Path(run_dir) / "method"
    if not method_dir.is_dir():
        return
    signature = str(
        (((spec.get("comparison") or {}).get("pluggable_component") or {})
         .get("signature")) or ""
    )
    sources = sorted(method_dir.glob("*.py"))
    for name in _METHOD_CONSUMER_SCAN_NAMES:
        entry = params.get(name)
        if not isinstance(entry, dict):
            continue
        if re.search(rf"\b{re.escape(name)}\b", signature):
            continue
        if any(_source_mentions_name(path, name) for path in sources):
            continue
        entry["unused_by_method"] = True


def _add_kd_data_params(params: dict, spec: dict, *,
                        paper_map: dict | None = None,
                        paper_text: str | None = None) -> None:
    """KD-specific data parameters: batch_size, train_size, n_test.

    The AL `_add_data_setup_params` adds num_rounds/initial_labeled/pool_size,
    which are AL-shaped. KD's notebook §3.1 uses batch_size + train_size + n_test
    instead (no labeled/unlabeled split; standard supervised train/test).
    """
    cr = spec.get("critical_requirements") or {}
    ds = cr.get("data_setup") or {}
    paper_section = ds.get("paper_section") or "Section 4 (data_setup)"

    # batch_size — paper-faithful when available. When the spec carries no
    # structured field, consult the anchored textual ladder before writing a
    # fallback: the old reasoning ("not extracted from spec") read as a
    # paper-silence claim while the paper stated "batch size is set to 1 per
    # GPU" (bev-distill 2026-07-02 F003). A known paper value is preserved
    # for audit even when the smoke run overrides it.
    smoke_batch = 32
    batch_size_paper = ds.get("batch_size")
    if batch_size_paper is not None:
        params["batch_size"] = {
            "value": batch_size_paper,
            "source": "paper",
            "paper_section": paper_section,
            "note": f"Paper uses batch_size={batch_size_paper}.",
        }
    else:
        textual = _lookup_textual_paper_value(
            "batch_size", paper_map=paper_map, paper_text=paper_text)
        if textual is not None and not paper_claim_findable(
                "batch_size", textual[0], paper_text):
            textual = None
        if textual is not None:
            value, source_label, section = textual
            params["batch_size"] = {
                "value": smoke_batch,
                "source": "system_default",
                "paper_value": value,
                "reasoning": (
                    f"Paper-stated batch_size={value!r} (anchored in "
                    f"{source_label}{f', {section}' if section else ''}). "
                    f"Using {smoke_batch} as the smoke-scale demo batch size; "
                    f"the paper value is preserved for audit."
                ),
            }
        else:
            params["batch_size"] = {
                "value": smoke_batch,
                "source": "system_inferred",
                "reasoning": (
                    f"The spec carries no structured data_setup.batch_size and "
                    f"no anchored paper statement of a batch size was found. "
                    f"Using {smoke_batch} as the smoke-scale demo default."
                ),
            }

    # train_size — smoke-scale; paper uses full training set
    params["train_size"] = {
        "value": 1000,
        "source": "system_default",
        "paper_value": "full training set",
        "reasoning": (
            "Paper trains on the full training set (typically tens of thousands to "
            "millions of examples). We subsample to 1,000 for smoke-scale demos to keep "
            "wall-time under ~5 min on CPU. The student won't reach paper accuracy at "
            "this scale; the smoke run verifies the distillation code path runs "
            "end-to-end, not benchmark-quality results."
        ),
    }

    # n_test — smoke; paper uses full test set
    params["n_test"] = {
        "value": 200,
        "source": "system_default",
        "paper_value": "full test set",
        "reasoning": (
            "Smoke evaluation size; sufficient to compute a stable per-epoch accuracy "
            "estimate without dominating runtime. Paper uses the full test set."
        ),
    }


def _add_kd_training_params(params: dict, spec: dict,
                            paper_text: str | None = None) -> None:
    """KD-specific training parameters: learning_rate + num_epochs.

    AL's `_add_training_params` adds max_epochs + train_until_accuracy (AL trains
    each round to a fixed accuracy threshold). KD trains for a fixed number of
    epochs instead.
    """
    cr = spec.get("critical_requirements") or {}
    training = cr.get("training") or {}
    paper_section = training.get("paper_section") or "Section 4 (Training)"
    is_image = _is_image_data(spec)

    # learning_rate — same parsing + source-decision logic as AL, via the
    # shared _lr_provenance_entry helper. The old inline copy here had
    # drifted: it lacked the findability arm, and its heuristic branch
    # falsely claimed "paper does not specify" (bev-distill 2026-07-01,
    # stage-2x halt F001).
    lr_str = str(training.get("learning_rate") or "")
    lr_value, note_suffix = _parse_learning_rate(lr_str, is_image)
    params["learning_rate"] = _lr_provenance_entry(
        lr_str, lr_value, note_suffix, paper_section, paper_text)

    # num_epochs — KD trains for a fixed number of epochs. Read only from
    # training.num_epochs. Do NOT fall back to data_setup.num_rounds: that's an
    # AL-specific field (acquisition rounds) and for KD it is either absent or
    # spuriously populated (e.g., =1), which silently misreads as "1 epoch".
    num_epochs_paper = training.get("num_epochs")
    smoke_num_epochs = 10
    if num_epochs_paper is not None and num_epochs_paper > smoke_num_epochs * 2:
        params["num_epochs"] = {
            "value": smoke_num_epochs,
            "source": "system_default",
            "paper_value": num_epochs_paper,
            "reasoning": (
                f"Paper trains for {num_epochs_paper} epochs. At smoke scale we train "
                f"{smoke_num_epochs} epochs — enough to see the distillation signal "
                f"emerge, while keeping wall-time under ~5 min on CPU."
            ),
        }
    elif num_epochs_paper is not None:
        params["num_epochs"] = {
            "value": num_epochs_paper,
            "source": "paper",
            "paper_section": paper_section,
            "note": f"Paper trains for {num_epochs_paper} epochs.",
        }
    else:
        params["num_epochs"] = {
            "value": smoke_num_epochs,
            "source": "system_inferred",
            "reasoning": (
                f"Paper epoch count not extracted. Using {smoke_num_epochs} epochs "
                "for smoke runs — enough to see distillation effect on a small dataset."
            ),
        }


def _add_scale_dependent_params(
    params: dict, spec: dict, *, overwrite: bool = True,
) -> list[str]:
    """Create params from `critical_requirements.scale_dependent_hyperparameters`.

    The lane's entries carry a paper-stated `paper_value`, so an entry that
    states one is a paper-sourced tunable param. Entries whose `paper_value`
    is null (formula-only or runtime-computed, e.g. clearance_obs) are NOT
    tunable params and are skipped.

    `overwrite=False` skips a name another reader already produced, so a
    caller that tops up an existing param set can never clobber it.

    Extracted from the motion-planning branch on 2026-08-05 (R2C-055) so the
    gap-family path can read the same lane.  R2C-083 later added a universal
    consumer at a different seam, after signature and taxonomy authorities.
    Keeping this earlier provisional reader preserves already-issued overlay
    behavior while the final pass can only top up an unclaimed identity.
    """
    cr = spec.get("critical_requirements") or {}
    added: list[str] = []
    for sdh in (cr.get("scale_dependent_hyperparameters") or []):
        if not isinstance(sdh, dict):
            continue
        name = sdh.get("name")
        val = sdh.get("paper_value")
        if not name or val is None:
            continue
        if not overwrite and name in params:
            continue
        params[name] = {
            "value": val,
            "source": "paper",
            "paper_section": sdh.get("paper_section") or "scale_dependent_hyperparameters",
            "note": (sdh.get("description") or "Scale-dependent hyperparameter from the paper."),
        }
        added.append(str(name))
    return added


def _add_motion_planning_params(params: dict, spec: dict, repo_root: Path) -> None:
    """Motion-planning params.

    A planner has no labeled-data acquisition loop (no initial_labeled /
    batch_size / num_rounds) and no trained-model knobs (no learning_rate /
    max_epochs / hidden_dim). Its tunable params are:
      - scale-dependent hyperparameters the paper states with a concrete value
        (e.g. d_safe = 0.375 m) — surfaced here with paper provenance;
      - the pluggable `plan(...)` signature's keyword extras (mc_samples,
        risk_weights, etc.) — added separately by `_add_paradigm_extras`.
    """
    _add_scale_dependent_params(params, spec)


def _reconcile_batch_acquisition_invariant(params: dict, paradigm_id: str) -> None:
    """Enforce the two-stage AL invariant batch_returns (b) >= batch_size (b').

    A two-stage selector preselects `batch_returns` candidates by the first-stage
    score, then ranks down to `batch_size` outputs, so b >= b'. Unlike a
    cost-driven knob (mc_samples), the preselection count has no runtime-cost
    reason to shrink and is hard-coupled to the output count, so it must not be
    demo-reduced below batch_size. The base derivation can leave batch_returns at
    the method signature's stub default (a standalone literal that cannot see
    batch_size), which is exactly how GBALD shipped b=30 < b'=100. Reconcile here
    where both runtime values are known: lift batch_returns to the paper's b when
    the spec recorded one that satisfies the invariant, otherwise to batch_size.
    Mirrors the validate_params_output b >= b' gate so the run does not halt on a
    value the deriver itself produced. No-op for single-stage selectors (no
    batch_returns) and non-AL paradigms.
    """
    if not paradigm_id.startswith("active_learning"):
        return
    br = params.get("batch_returns")
    bs = params.get("batch_size")
    if not isinstance(br, dict) or not isinstance(bs, dict):
        return
    br_val, bs_val = br.get("value"), bs.get("value")
    numeric = (int, float)
    if isinstance(br_val, bool) or isinstance(bs_val, bool):
        return
    if not isinstance(br_val, numeric) or not isinstance(bs_val, numeric):
        return
    if br_val >= bs_val:
        return

    paper_b = br.get("paper_value")
    use_paper = (
        isinstance(paper_b, numeric)
        and not isinstance(paper_b, bool)
        and paper_b >= bs_val
    )
    target = paper_b if use_paper else bs_val
    if use_paper:
        br["value"] = target
        br["source"] = "paper"
        # source=paper requires a locator (`note` or `paper_section`) — the
        # 2026-07-03 GBALD re-roll halted at the deterministic validator
        # because this branch emitted the provenance as `reasoning` only,
        # the one source=paper site in the deriver without a locator.
        br.pop("reasoning", None)
        br["note"] = (
            f"batch_returns (two-stage preselection count b) reconciled from the "
            f"signature stub default {br_val!r} to the paper value {target!r} to "
            f"satisfy the acquisition invariant b >= b' (batch_size={bs_val})."
        )
    else:
        br["value"] = target
        br["source"] = "system_inferred"
        br["reasoning"] = (
            f"batch_returns (two-stage preselection count b) reconciled from the "
            f"signature stub default {br_val!r} up to batch_size {bs_val} to "
            f"satisfy the acquisition invariant b >= b'. The paper preselection "
            f"size was not recorded in the spec, so batch_size is the minimum "
            f"coherent value."
        )


def derive(
    spec: dict,
    repo_root: Path = ROOT,
    *,
    paper_map: dict | None = None,
    paper_text: str | None = None,
    run_dir: Path | None = None,
) -> dict:
    paradigm_id = (spec.get("comparison") or {}).get("classification", {}).get("id", "")

    # Gap-path serves() overlay: derive against the run's own provisional
    # pack when stage 1 installed one (run_dir is the key); committed runs
    # resolve the committed view byte-identically (overlay dir is None).
    # Loaded from this repo's SSOT (ROOT), exactly like the helpers'
    # previous default-argument behavior -- `repo_root` here is a template/
    # data root, not a taxonomy source.
    tax = taxonomy.load_taxonomy(
        ROOT,
        provisional_packs_dir=taxonomy.run_overlay_dir(run_dir),
    )

    is_kd = paradigm_id.startswith("knowledge_distillation")
    is_al = paradigm_id.startswith("active_learning")
    is_mp = paradigm_id.startswith("motion_planning")

    params: dict = {}
    binding_record: dict | None = None
    if is_kd:
        _add_kd_data_params(params, spec, paper_map=paper_map,
                            paper_text=paper_text)
        _add_kd_training_params(params, spec, paper_text=paper_text)
    elif is_al:
        # Per-dataset value binding (per-dataset-value-binding-design.md):
        # when the spec's data_setup carries per_dataset_values, rebind the
        # affected scalars to the demo's dataset BEFORE the ordinary
        # derivation reads them, so every downstream consumer sees exactly
        # one bound value, as today.
        cr = spec.get("critical_requirements") or {}
        ds = cr.get("data_setup") or {}
        effective_ds, binding_record = _resolve_per_dataset_bindings(ds, run_dir)
        if binding_record is not None:
            spec = {**spec,
                    "critical_requirements": {**cr, "data_setup": effective_ds}}
        _add_data_setup_params(params, spec, repo_root, tax)
        _annotate_bound_datasets(params, binding_record)
        _add_training_params(params, spec, paper_text=paper_text)
    elif is_mp:
        # Planner: scale-dependent hyperparameters only; no AL data/training
        # params, no learned-model params. The plan() signature extras come
        # from _add_paradigm_extras below.
        _add_motion_planning_params(params, spec, repo_root)
    else:
        # Unknown paradigm. Do NOT fabricate AL data/training params (the
        # prior behavior — a catch-all `else` that injected num_rounds /
        # pool_size / train_until_accuracy for every non-KD paradigm — was the
        # PA-D3 bug). A new paradigm with bespoke data/training knobs should
        # add an explicit branch here; until then it gets only model params
        # (below, if it has a learned model) + pluggable-signature extras.
        pass

    # Gap (provisional-pack) families: the family branch above ran on the
    # committed parent's assumptions, which may not fit (SRL 2026-07-05: the
    # motion_planning branch assumes planners have no training knobs, and
    # the RL collision-avoidance family trains a network). Top up with the
    # spec's own structured training evidence, values only, no conventions.
    # Committed families are untouched (the gate is node.provisional).
    _node = taxonomy.serves(paradigm_id, tax) if paradigm_id else None
    glossary_added: list[str] = []
    if _node is not None and getattr(_node, "provisional", False):
        _add_provisional_family_training_params(params, spec)
        # R2C-055: the gap-family analyzer contract names
        # scale_dependent_hyperparameters as the mandatory carrier for a
        # paper's physical/scale values (dispatch_templates: "the structured
        # parameter fields are the ONLY parameter carriers downstream"), and
        # before this the lane had no reader outside the motion-planning
        # branch. A compliant analyzer's value was dropped with no error
        # until a pack check demanded the param three stages later (pdfgnn
        # 2026-08-05, similarity_cutoff). Top up, never clobber.
        _add_scale_dependent_params(params, spec, overwrite=False)
        glossary_added = _add_glossary_value_params(params, spec)

    # Model params (hidden_dim) only apply to paradigms with a learned neural
    # model. motion_planning has no neural net, so skip it there (PA-D4).
    if not is_mp:
        _add_model_params(params, spec, paradigm_id, tax)

    _add_paradigm_extras(
        params,
        spec,
        paradigm_id,
        paper_map=paper_map,
        paper_text=paper_text,
        tax=tax,
    )
    _reconcile_batch_acquisition_invariant(params, paradigm_id)
    _mark_unused_params(params, spec)
    _apply_params_derivation(params, spec, paradigm_id, tax,
                             paper_map=paper_map, paper_text=paper_text)
    _drop_shadowed_glossary_params(params, spec, glossary_added)
    _consume_parameter_carriers(
        params,
        spec,
        taxonomy.load_params_derivation(paradigm_id, tax),
    )
    _apply_param_glossary(params, spec)
    _apply_evaluation_protocol_params(params, spec)
    _reconcile_protocol_params_with_bundle_axis(params, spec, run_dir)

    output: dict = {"schema_version": SCHEMA_VERSION, "params": params}
    if binding_record is not None:
        output["per_dataset_binding"] = binding_record
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    args = parser.parse_args()

    if not args.spec.is_file():
        print(f"error: spec not found: {args.spec}", file=sys.stderr)
        return 1

    try:
        spec = json.loads(args.spec.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        print(f"error: spec is not valid JSON ({e})", file=sys.stderr)
        return 1

    pipeline_dir = args.run_dir / ".pipeline"
    paper_map = None
    paper_map_path = pipeline_dir / "paper_map.json"
    if paper_map_path.is_file():
        try:
            paper_map = json.loads(paper_map_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            paper_map = None
    paper_text = None
    paper_md_path = pipeline_dir / "paper.md"
    if paper_md_path.is_file():
        paper_text = paper_md_path.read_text(encoding="utf-8")

    output = derive(
        spec,
        args.repo_root,
        paper_map=paper_map,
        paper_text=paper_text,
        run_dir=args.run_dir,
    )
    _mark_runtime_unused_model_params(output["params"], args.run_dir)
    _mark_unused_by_method(output["params"], spec, args.run_dir)

    pipeline_dir.mkdir(parents=True, exist_ok=True)
    output_path = pipeline_dir / "params.json"
    output_path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")

    print(f"wrote {output_path}")
    print(f"  {len(output['params'])} parameters: {list(output['params'].keys())}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
