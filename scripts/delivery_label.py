"""Delivery label derivation — the three-state policy (third label signed
off 2026-07-02, the third delivery label design note (internal, not shipped)).

The probe battery and the fidelity review produce verdicts; this module
turns them into the one researcher-facing word on the box: `verified`,
`draft`, or `uncertified_new_territory`. The rules close the two escapes
Section 1 exists to close:

- **Critical findings can't defer their way into `verified`.** A
  fidelity finding of severity critical/important whose resolution was
  not actually APPLIED (pending, failed, needs_user, or absent) demotes
  to draft — the R_0-shipped class, where an important provenance
  finding was routed to a report nobody is forced to read while the
  artifact shipped looking blessed.
- **Smoke is an execution check, not a correctness authority.** The
  label never comes from "the notebook ran"; it comes from the battery's
  behavioral verdicts. A probe `fail` or `flag_for_researcher` demotes:
  a researcher flag means a human must look before trusting, and that is
  exactly what draft means.

`warn` and `unprobeable` verdicts do NOT demote — verified means "what
we could check passed", and the unchecked surface is DISCLOSED instead
of silently blocking (the delivered-unexecuted notebook (UB-6) would
otherwise mark every run draft until slice 2.3 lands). A missing battery
report demotes: absence of verification evidence is not verification.

One exception, added 2026-08-06 (R2C-047): an `unprobeable` verdict bound
to a core_methodology element the run's own contract marks must_replicate
CAPS the label at `uncertified_new_territory`. The disclose-don't-demote
rule was written for peripheral coverage, and it let a delivered run
present at full label with the paper's central mechanism unprobed. This
is a cap rather than a deduction, and peripheral or approximable
mechanisms are untouched, so it costs an ordinary unprobeable nothing.

Partial delivery (design note 2026-07-05, §3.3 + §3.5): a package that
ships with stubbed components can NEVER read `verified`, whatever its
probe verdicts. Stubs are a distinct recorded cause (`partial_delivery`),
not a generic failure: with contribution evidence they demote to `draft`
with one reason per stub; without contribution evidence the label stays
`uncertified_new_territory` (the evidence axis is orthogonal to the
completeness axis — a gap-family paper with stubs records both facts).
The verdict always carries the structured `stubbed_elements` list and a
`partial` flag so downstream tooling can never lose the partial state in
rewording.

The third state closes the false-verified hole (2026-06-18 checkpoint):
zero demoters is necessary but not sufficient for `verified` — the run
must also carry CONTRIBUTION EVIDENCE, at least one passing verdict from
the contribution tier (the family-specific paradigm probes or the
claims-tier contribution probes). Universal-tier passes alone are the
floor every family gets, not certification of the paper's mechanism; a
paper whose contribution probes all came back unprobeable (or whose
family has no contribution probes yet) delivers as
`uncertified_new_territory`, never as a hollow `verified`. The label
encodes EVIDENCE, never quality.

Demo-success verdict (design approved 2026-07-16): the deterministic
post-smoke pass (`scripts/demo_verdict.py`) answers "did the headline demo
demonstrably work" — the question smoke never asks. Its `failed` is a
first-class demoter here (source `demo_verdict`, reason id `demo_failed`)
with a plain-language disclosure, so the label never again lands on draft
through ADJACENT demoters by luck (the iDb-RRT 2026-07-14 shape).
`undetermined` discloses honestly (plus the kit-coverage finding the verdict
pass recorded on a committed family); `succeeded` adds nothing. Where the
verdict and the executed-notebook probe (UB-6) both bound and disagree, the
disagreement is recorded as a probe-bug disclosure we own — probes are the
deeper layer and stay unchanged. A run with NO recorded verdict (pre-feature
runs, smoke never clean) derives as before, identical output modulo the
schema version string.

Structured demo evidence (R2C-086) versions that artifact without rewriting
legacy runs. Invalid evaluation and valid-but-not-demonstrated skill are
distinct demoters; unresolved evidence is disclosed. Execution, mechanism,
skill, and paper-benchmark status remain separate in the compact manifest
record, and mechanism is populated only from bound behavioral probes.

Pure functions, no I/O — the driver supplies parsed JSON.
"""

from __future__ import annotations

import json
import keyword
import math

SCHEMA_VERSION = "1.7.0"
LABEL_VERIFIED = "verified"
LABEL_DRAFT = "draft"
LABEL_UNCERTIFIED = "uncertified_new_territory"
GRAPH_EVIDENCE_NOT_APPLICABLE = "not_applicable"
GRAPH_EVIDENCE_HOMOGENEOUS_V1 = "homogeneous_graph_v1"

# Neutral-plan label cap (R2C-032, approved 2026-07-28): a delivery built on
# the generic provisional build plan (build_plan.py PROVISIONAL_PLAN_KEY)
# was never held to any family's build conventions, so certifying it as
# draft or verified overstates what was checked. The cap holds such a
# delivery at uncertified — new territory (label scale rank 2), whatever
# its probe verdicts; demoting reasons stay recorded so nothing is hidden.
# Under this rule the mid-July gap drafts (fedavg, DomIndOnto) would have
# read uncertified — the honest reading of what they were.
NEUTRAL_PLAN_CAP_ID = "neutral_plan_cap"

# Unprobeable core-novelty cap (R2C-047, policy maintainer decision 2026-08-03, shape
# 2026-08-05). `unprobeable` does not demote, by the rule above: the unchecked
# surface is disclosed rather than silently blocking. That rule was written for
# peripheral coverage, and it let the delivered bayesian run present at full
# label with the paper's central mechanism unprobed. Coverage loss on the
# paper's OWN core novelty is different in kind, so it enters as a CAP (the
# label cannot exceed uncertified — new territory) rather than a point
# deduction: the run has no evidence about the thing the paper is for.
# Peripheral and approximable mechanisms are untouched, so an unprobeable
# supporting check still costs a delivery nothing.
UNPROBEABLE_CORE_CAP_ID = "unprobeable_core_novelty_cap"

_GATING_SEVERITIES = ("critical", "important")
_RESOLVED = "applied"

# The contribution tier, keyed by probe id (the design note's sanctioned
# fallback: verdict `tier` values are not uniformly set across probe
# modules, so the explicit prefix set is the reliable, self-documenting
# key). AL-/MP-/KD-/TSF- are the family-specific paradigm probes; CT-1 and
# CT-5 are the claims-tier probes that certify the contribution itself
# (CT-3 is the claims LEDGER probe — bookkeeping, not certification).
_CONTRIBUTION_PREFIXES = ("AL-", "MP-", "KD-", "TSF-")
_CONTRIBUTION_IDS = frozenset({"CT-1", "CT-5", "HG-7"})


def is_contribution_probe(probe_id: str) -> bool:
    pid = str(probe_id or "")
    return pid in _CONTRIBUTION_IDS or pid.startswith(_CONTRIBUTION_PREFIXES)


_UNIVERSAL_PREFIXES = ("US-", "UB-")


def universal_floor_clause(delivery: dict | None) -> str:
    """The honest 'what passed' clause for uncertified surfaces.

    "Every universal check passed" is only true when every universal check
    actually RAN. The bev-distill 2026-07-04 roll shipped its notebook
    unexecuted after a smoke cap-degrade, the executed-notebook check read
    unprobeable, and the old fixed sentence overclaimed. The clause now
    derives from the delivery's own disclosures (maintainer-approved wording
    option, 2026-07-04): universal-tier ids that could not run are counted
    and named as a gap, keyed by the same sanctioned id-prefix sets the
    label derivation uses (verdict tier fields are inconsistent across
    probe modules)."""
    n_gap = sum(
        1 for d in (delivery or {}).get("disclosures") or []
        if isinstance(d, dict)
        and str(d.get("verdict") or "") == "unprobeable"
        and str(d.get("id") or "").startswith(_UNIVERSAL_PREFIXES)
    )
    # Neutral-plan cap (R2C-032): a capped delivery can carry genuine
    # demoting verdicts in `reasons` while reading uncertified. The floor
    # clause must never claim a pass that did not happen (the bev-distill
    # 2026-07-04 overclaim class, one layer deeper).
    n_fail = sum(
        1 for d in (delivery or {}).get("reasons") or []
        if isinstance(d, dict)
        and str(d.get("verdict") or "") in ("fail", "flag_for_researcher")
        and str(d.get("id") or "").startswith(_UNIVERSAL_PREFIXES)
    )
    if n_fail:
        return (f"{n_fail} universal check{'' if n_fail == 1 else 's'} "
                f"FAILED or was flagged (listed in the reasons below and "
                f"in REPORT.md) — this package has known problems on top "
                f"of its unchecked new-territory surface.")
    if n_gap == 0:
        return "Every universal check passed."
    return (f"Every universal check that could run passed, but "
            f"{n_gap} universal check{'' if n_gap == 1 else 's'} could "
            f"not run (disclosed below and in REPORT.md) — that gap is "
            f"real, not paperwork.")


def contribution_checks_blocked(delivery: dict | None) -> int:
    """Contribution checks that produced no bindable package evidence."""
    return sum(
        1 for d in (delivery or {}).get("disclosures") or []
        if isinstance(d, dict)
        and str(d.get("verdict") or "") in {
            "unprobeable", "binding_gap", "not_applicable",
        }
        and is_contribution_probe(d.get("id"))
    )


def contribution_binding_gaps(delivery: dict | None) -> int:
    """Contribution checks that ran but failed exact obligation binding."""
    return sum(
        1 for d in (delivery or {}).get("disclosures") or []
        if isinstance(d, dict)
        and str(d.get("verdict") or "") == "binding_gap"
        and is_contribution_probe(d.get("id"))
    )


def contribution_not_applicable(delivery: dict | None) -> int:
    """Contribution checks that completed but do not apply to this package."""
    return sum(
        1 for d in (delivery or {}).get("disclosures") or []
        if isinstance(d, dict)
        and str(d.get("verdict") or "") == "not_applicable"
        and is_contribution_probe(d.get("id"))
    )


def contribution_gap_clause(delivery: dict | None) -> str:
    """The honest 'why no certification' clause for uncertified surfaces.

    Two machine-distinguishable cases behind `uncertified_new_territory`:
    the family kit genuinely does not exist yet, or it exists and its
    probes ran but none could bind to this package (all unprobeable).
    The old fixed sentence claimed "checks don't exist yet" either way —
    on the 2026-07-05 detr-distill delivery (a knowledge-distillation
    paper, a family whose kit exists) it rendered one screen above a
    probe table full of KD rows, so the report contradicted itself. The
    clause now derives from the delivery's own disclosures, the same
    pattern as `universal_floor_clause`."""
    n_blocked = contribution_checks_blocked(delivery)
    if n_blocked == 0:
        return ("This paper type is new to the system, so behavioral "
                "checks for its core contribution don't exist yet.")
    n_binding_gap = contribution_binding_gaps(delivery)
    if n_binding_gap:
        return (f"Behavioral checks for this paper type's core contribution "
                f"exist and ran, but {n_binding_gap} passing result"
                f"{'' if n_binding_gap == 1 else 's'} could not bind to the "
                "exact methodology obligation declared by this package. "
                "Each disclosure names the missing, unknown, or mismatched "
                "binding; the results are not certification evidence.")
    n_not_applicable = contribution_not_applicable(delivery)
    if n_not_applicable:
        if n_not_applicable == n_blocked:
            return (
                "Behavioral checks for this paper type's core contribution "
                f"exist and completed, but {n_not_applicable} check"
                f"{'' if n_not_applicable == 1 else 's'} did not apply to "
                "this package's declared mechanism. That is a completed "
                "conditional result, not a missing kit, and it supplies no "
                "certification evidence."
            )
        return (
            "Behavioral checks for this paper type's core contribution "
            f"exist, but none established applicable evidence: "
            f"{n_not_applicable} check"
            f"{'' if n_not_applicable == 1 else 's'} did not apply and "
            f"{n_blocked - n_not_applicable} could not run or bind."
        )
    return (f"Behavioral checks for this paper type's core contribution "
            f"exist, but none could run against this package: "
            f"{n_blocked} check{'' if n_blocked == 1 else 's'} could not "
            f"bind to the delivered interfaces (each disclosure names "
            f"what could not be synthesized).")


def _uses_reference_qualified_contract(spec: dict | None) -> bool:
    """Whether this spec opted into reference-qualified probe bindings.

    Archived contracts predate ``verification_probe_refs``.  Their stored
    reports retain the legacy prefix-based evidence reading.  Presence of the
    field (including an intentionally empty list) marks the new contract and
    turns missing, dangling, or mismatched refs into non-certifying evidence.
    """
    from probe_spec_join import reference_contract_enabled  # noqa: PLC0415

    return reference_contract_enabled(spec)


def _has_contribution_evidence(
    probe_report: dict | None,
    spec: dict | None = None,
    graph_execution_plan: dict | None = None,
) -> bool:
    """At least one contribution-tier probe bound to the artifact AND passed.

    Reference-aware contracts require the verdict's exact family executor ref
    to intersect the methodology obligations reached by its recorded element
    ids/callable anchors.  Legacy contracts remain readable under the older
    prefix rule; this compatibility arm disappears family by family as their
    producers emit ``verification_probe_refs``.
    """
    if not probe_report or not isinstance(probe_report.get("verdicts"), list):
        return False
    if _has_homogeneous_graph_contract(spec):
        # The graph family has an explicit evidence ladder: HG-7 alone cannot
        # certify when alignment, construction, liveness, typed permutation
        # applicability, or the paper-grounded null is absent or incoherent.
        return _graph_probe_axis(
            probe_report, spec, "contribution", graph_execution_plan
        ).get("status") == "demonstrated"
    candidates = [
        verdict for verdict in probe_report["verdicts"]
        if isinstance(verdict, dict)
        and verdict.get("verdict") == "pass"
        and is_contribution_probe(verdict.get("probe_id"))
    ]
    if not _uses_reference_qualified_contract(spec):
        return bool(candidates)

    from probe_spec_join import join_verdict  # noqa: PLC0415

    for verdict in candidates:
        if not str(verdict.get("probe_ref") or "").strip():
            continue
        if join_verdict(verdict, spec).reference_qualified:
            return True
    return False


def _probe_reasons(
    probe_report: dict | None,
    spec: dict | None = None,
) -> tuple[list[dict], list[dict]]:
    """(demoting reasons, non-demoting disclosures) from a battery report."""
    if not probe_report or not isinstance(probe_report.get("verdicts"), list):
        return [{
            "source": "battery",
            "id": "missing",
            "message": "probe battery report missing — verification "
                       "evidence absent, delivering as draft",
        }], []
    reasons, disclosures = [], []
    for v in probe_report["verdicts"]:
        verdict = v.get("verdict")
        entry = {
            "source": "probe",
            "id": str(v.get("probe_id", "?")),
            "verdict": verdict,
            "message": str(v.get("message", "")),
        }
        if v.get("evidence"):
            # File/cell/line trace — the proximity-to-correct summary
            # (failure-path spec §4) groups demoters by this.
            entry["evidence"] = str(v["evidence"])
        if verdict == "fail":
            reasons.append(entry)
        elif verdict == "flag_for_researcher":
            reasons.append(entry)
        elif verdict in ("warn", "unprobeable"):
            disclosures.append(entry)
        elif verdict == "not_applicable" and is_contribution_probe(entry["id"]):
            # A family check existed and completed its applicability decision.
            # Carry that fact into the delivery so the report never calls the
            # kit missing merely because no applicable row could certify.
            disclosures.append(entry)
        elif (
            verdict == "pass"
            and is_contribution_probe(entry["id"])
            and _uses_reference_qualified_contract(spec)
        ):
            # A probe can execute successfully yet fail the second half of
            # R2C-087's evidence contract: exact obligation binding. Keep the
            # observed pass non-demoting, but disclose why it cannot certify.
            from probe_spec_join import (  # noqa: PLC0415
                join_verdict,
                unbound_gap_note,
            )

            join = join_verdict(v, spec)
            gap = unbound_gap_note(join)
            if not join.reference_qualified and not gap:
                gap = (
                    "This passing contribution check carries neither the "
                    "exact probe ref nor methodology grounding required by "
                    "this reference-aware contract. It ran, but remains an "
                    "unbound coverage result rather than certification "
                    "evidence."
                )
            if gap:
                observed = entry["message"]
                disclosures.append({
                    **entry,
                    "verdict": "binding_gap",
                    "observed_verdict": "pass",
                    "message": (
                        observed
                        if gap in observed
                        else f"{observed} — {gap}"
                    ),
                })
    return reasons, disclosures


def _review_reasons(review_report: dict | None) -> list[dict]:
    """Unresolved critical/important fidelity findings demote to draft."""
    if not review_report:
        return []
    reasons = []
    for f in review_report.get("findings") or []:
        if not isinstance(f, dict):
            continue
        severity = str(f.get("severity") or "").lower()
        if severity not in _GATING_SEVERITIES:
            continue
        if str(f.get("resolution_status") or "pending") == _RESOLVED:
            continue
        description = str(f.get("description", ""))
        reasons.append({
            "source": "fidelity_review",
            "id": str(f.get("id", "?")),
            "severity": severity,
            # Cap keeps the label record readable; the ellipsis says the
            # full text lives in the review report (a bare cut reads as a
            # rendering bug on the researcher surface).
            "message": description[:300] + ("…" if len(description) > 300 else ""),
        })
    return reasons


def _ub6_verdicts(probe_report: dict | None) -> set[str]:
    """Every verdict the executed-notebook probe (UB-6) recorded."""
    out: set[str] = set()
    for v in (probe_report or {}).get("verdicts") or []:
        if isinstance(v, dict) and str(v.get("probe_id") or "") == "UB-6":
            out.add(str(v.get("verdict") or ""))
    return out


_GRAPH_PROBE_REFS = {
    "HG-1": "graph_mechanism.alignment_prerequisite",
    "HG-2": "graph_mechanism.parameter_agreement",
    "HG-3": "graph_mechanism.construction_semantics",
    "HG-4": "graph_mechanism.topology_sensitivity",
    "HG-5": "graph_mechanism.neighbor_sensitivity",
    "HG-6": "graph_mechanism.permutation_equivalence",
    "HG-7": "graph_mechanism.contribution_ablation",
}

_GRAPH_CONTRACT_REF_FIELDS = {
    "HG-2": "parameter_agreement",
    "HG-3": "construction",
    "HG-4": "topology",
    "HG-5": "neighbor_signal",
    "HG-6": "permutation",
    "HG-7": "contribution_ablation",
}


def _has_homogeneous_graph_contract(spec: dict | None) -> bool:
    """Whether this fresh spec declares the R2C-088 graph contract.

    Graph scope is never inferred from a callable, parameter, or class name.
    Archived graph specs, graph-free methods, and unsupported relational
    grammars omit this typed block and retain their existing delivery read.
    """
    if not isinstance(spec, dict):
        return False
    methodology = spec.get("methodology_replication_contract")
    return (
        isinstance(methodology, dict)
        and isinstance(methodology.get("homogeneous_graph_mechanism"), dict)
    )


def _graph_callable_key(value: object) -> str | None:
    if not isinstance(value, dict) or set(value) != {"module", "qualname"}:
        return None
    module = value.get("module")
    qualname = value.get("qualname")
    if module not in {"method.model", "method.training", "method.method"}:
        return None
    if (
        not isinstance(qualname, str)
        or not qualname.isidentifier()
        or qualname.startswith("_")
        or keyword.iskeyword(qualname)
    ):
        return None
    return f"{module}:{qualname}"


def _expected_graph_bindings(
    mechanism: dict,
) -> dict[str, tuple[str, list[str] | None]] | None:
    """Spec-owned element and callable identities for every HG obligation."""
    construction = mechanism.get("construction")
    message = mechanism.get("message_passing")
    ablation = mechanism.get("contribution_ablation")
    if not isinstance(construction, dict) or not isinstance(message, dict):
        return None
    construction_callable = _graph_callable_key(construction.get("callable"))
    message_callable = _graph_callable_key(message.get("callable"))
    alignment_owner = mechanism.get("alignment_element_id")
    construction_owner = construction.get("element_id")
    message_owner = message.get("element_id")
    if (
        construction_callable is None
        or message_callable is None
        or construction_callable == message_callable
        or any(
            not isinstance(value, str)
            or not value
            or value != value.strip()
            for value in (alignment_owner, construction_owner, message_owner)
        )
    ):
        return None
    expected: dict[str, tuple[str, list[str] | None]] = {
        # The R2C-084 architecture contract owns this exact preparation
        # callable, so MethodSpec can independently constrain only its owner.
        # The shared execution-plan digest binds the concrete callable used.
        "HG-1": (alignment_owner, None),
        "HG-2": (construction_owner, [construction_callable]),
        "HG-3": (construction_owner, [construction_callable]),
        "HG-4": (message_owner, [message_callable]),
        "HG-5": (message_owner, [message_callable]),
        "HG-6": (message_owner, [message_callable]),
    }
    if isinstance(ablation, dict):
        ablation_owner = ablation.get("element_id")
        if (
            not isinstance(ablation_owner, str)
            or not ablation_owner
            or ablation_owner != ablation_owner.strip()
        ):
            return None
        hg7_callables = [message_callable]
        if ablation.get("kind") in {
            "removed_message_passing", "non_graph_decoder",
        }:
            ablation_callable = _graph_callable_key(ablation.get("callable"))
            if ablation_callable is None:
                return None
            if ablation_callable in {construction_callable, message_callable}:
                return None
            hg7_callables.append(ablation_callable)
        expected["HG-7"] = (ablation_owner, hg7_callables)
    return expected


def _graph_evidence(row: dict) -> dict | None:
    evidence = row.get("evidence")
    if isinstance(evidence, str):
        try:
            evidence = json.loads(evidence)
        except json.JSONDecodeError:
            return None
    return evidence if isinstance(evidence, dict) else None


def _is_sha256_digest(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 71
        and value.startswith("sha256:")
        and all(character in "0123456789abcdef" for character in value[7:])
    )


def _finite_number(value: object) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
    )


def _graph_edges(value: object, representation: object) -> set[tuple[int, int]] | None:
    if not isinstance(value, list):
        return None
    if representation == "dense_adjacency":
        if not value or any(not isinstance(row, list) for row in value):
            return None
        node_count = len(value)
        if any(len(row) != node_count for row in value):
            return None
        return {
            (source, destination)
            for source, row in enumerate(value)
            for destination, weight in enumerate(row)
            if _finite_number(weight) and float(weight) != 0.0
        }
    if representation == "sparse_edge_index" and len(value) == 2:
        sources, destinations = value
        if (
            not isinstance(sources, list)
            or not isinstance(destinations, list)
            or len(sources) != len(destinations)
            or any(
                isinstance(index, bool) or not isinstance(index, int)
                for index in (*sources, *destinations)
            )
        ):
            return None
        return set(zip(sources, destinations))
    return None


def _comparison_witness_evidence_problem(
    evidence: dict,
    construction: dict,
    fixture: dict,
    representation: object,
    *,
    node_count: int,
) -> str | None:
    """Recheck the exact equality-boundary witness behind a passing HG-3."""

    witness = fixture.get("comparison_witness")
    row = evidence.get("comparison_witness")
    threshold = construction.get("threshold")
    if (
        not isinstance(witness, dict)
        or not isinstance(row, dict)
        or not isinstance(threshold, dict)
    ):
        return "HG-3 lacks its exact comparison-boundary witness"
    comparison = threshold.get("comparison")
    if (
        threshold.get("metric") != "cosine_similarity"
        or comparison not in {"greater_than", "greater_than_or_equal"}
        or witness.get("metric") != "cosine_similarity"
        or witness.get("comparison") != comparison
        or not _finite_number(witness.get("threshold_value"))
        or float(witness["threshold_value"]) != 1.0
    ):
        return "HG-3 comparison witness disagrees with the frozen declaration"

    entity_axis = witness.get("entity_axis")
    raw_features = witness.get("feature_input")
    if (
        node_count < 2
        or isinstance(entity_axis, bool)
        or not isinstance(entity_axis, int)
        or not isinstance(raw_features, list)
        or len(raw_features) != node_count
        or any(not isinstance(values, list) for values in raw_features)
        or any(len(values) != node_count for values in raw_features)
    ):
        return "HG-3 comparison witness has invalid feature shape"
    normalized_axis = entity_axis + 2 if entity_axis < 0 else entity_axis
    if normalized_axis not in {0, 1}:
        return "HG-3 comparison witness has invalid entity axis"
    matrix = (
        raw_features
        if normalized_axis == 0
        else [
            [raw_features[column][row] for column in range(node_count)]
            for row in range(node_count)
        ]
    )
    expected_features = [[0.0] * node_count for _ in range(node_count)]
    expected_features[0][0] = 1.0
    expected_features[1][0] = 1.0
    for index in range(2, node_count):
        expected_features[index][index - 1] = 1.0
    if any(
        not _finite_number(observed)
        or float(observed) != expected_features[row_index][column_index]
        for row_index, values in enumerate(matrix)
        for column_index, observed in enumerate(values)
    ):
        return "HG-3 comparison witness is not the exact one-hot boundary fixture"

    expected_edges = _graph_edges(witness.get("expected_graph"), representation)
    oracle_edges: set[tuple[int, int]] = set()
    if comparison == "greater_than_or_equal":
        oracle_edges.update({(0, 1), (1, 0)})
    self_loop_policy = construction.get("self_loop_policy")
    if self_loop_policy == "required":
        oracle_edges.update((index, index) for index in range(node_count))
    elif self_loop_policy != "forbidden":
        return "HG-3 comparison witness has unsupported self-loop authority"
    if expected_edges is None or expected_edges != oracle_edges:
        return "HG-3 comparison witness has a forged boundary oracle"

    expected_count = len(oracle_edges)
    if (
        row.get("metric") != "cosine_similarity"
        or row.get("comparison") != comparison
        or not _finite_number(row.get("threshold_value"))
        or float(row["threshold_value"]) != 1.0
        or row.get("identical_entity_positions") != [0, 1]
        or not _finite_number(row.get("runtime_threshold_value"))
        or float(row["runtime_threshold_value"]) != 1.0
        or isinstance(row.get("expected_edge_count"), bool)
        or row.get("expected_edge_count") != expected_count
        or isinstance(row.get("constructed_edge_count"), bool)
        or row.get("constructed_edge_count") != expected_count
        or isinstance(row.get("observed_edge_symmetric_difference"), bool)
        or row.get("observed_edge_symmetric_difference") != 0
        or bool(row.get("problems"))
    ):
        return "HG-3 comparison-boundary observation is forged or incomplete"
    return None


def _execution_seed_tolerance_problem(
    evidence: dict,
    graph_execution_plan: dict,
) -> str | None:
    execution = graph_execution_plan.get("execution")
    if not isinstance(execution, dict):
        return "canonical graph plan lacks execution evidence authority"
    if (
        evidence.get("seed") != execution.get("seed")
        or evidence.get("tolerance") != execution.get("tolerance")
    ):
        return "graph evidence seed or tolerance disagrees with the plan"
    return None


def _effect_threshold_authority_problem(
    evidence: dict,
    graph_execution_plan: dict,
) -> str | None:
    execution = graph_execution_plan.get("execution")
    tolerance = (
        execution.get("tolerance") if isinstance(execution, dict) else None
    )
    threshold = evidence.get("effect_threshold")
    baseline_max_abs = evidence.get("baseline_max_abs")
    changed_max_abs = evidence.get("changed_max_abs")
    effect_scale = evidence.get("effect_scale")
    values = (tolerance, threshold, baseline_max_abs, changed_max_abs, effect_scale)
    if not all(_finite_number(value) for value in values):
        return "graph effect threshold authority is missing or non-finite"
    if (
        float(tolerance) <= 0
        or float(threshold) < 0
        or float(baseline_max_abs) < 0
        or float(changed_max_abs) < 0
        or float(effect_scale) < 1
    ):
        return "graph effect threshold authority is outside its valid domain"
    expected_scale = max(
        1.0, float(baseline_max_abs), float(changed_max_abs)
    )
    expected_threshold = float(tolerance) * expected_scale
    if (
        not math.isfinite(expected_threshold)
        or float(effect_scale) != expected_scale
        or float(threshold) != expected_threshold
    ):
        return "graph effect threshold disagrees with its exact output scale"
    return None


def _effect_evidence_problem(
    evidence: object,
    *,
    kind: str,
    expect_effect: bool,
    graph_execution_plan: dict,
) -> str | None:
    if not isinstance(evidence, dict):
        return f"{kind} evidence is not a structured mapping"
    execution_problem = _execution_seed_tolerance_problem(
        evidence, graph_execution_plan
    )
    if execution_problem is not None:
        return execution_problem
    threshold_problem = _effect_threshold_authority_problem(
        evidence, graph_execution_plan
    )
    if threshold_problem is not None:
        return threshold_problem
    threshold = evidence["effect_threshold"]
    if kind == "topology":
        removed = evidence.get("removed_edges")
        targets = evidence.get("target_entities")
        deltas = evidence.get("target_max_abs_deltas")
        maximum = evidence.get("maximum_target_delta")
        if (
            not isinstance(removed, list)
            or not removed
            or not isinstance(targets, list)
            or not targets
            or not isinstance(deltas, list)
            or len(deltas) != len(targets)
            or not all(_finite_number(value) for value in deltas)
            or any(float(value) < 0 for value in deltas)
            or not _finite_number(maximum)
            or float(maximum) < 0
            or float(maximum) != max(float(value) for value in deltas)
        ):
            return "topology evidence lacks its exact intervention and deltas"
        fixture = graph_execution_plan.get("fixture")
        intervention = (
            fixture.get("topology_intervention")
            if isinstance(fixture, dict) else None
        )
        construction = graph_execution_plan.get("construction")
        requested = (
            intervention.get("remove_edges")
            if isinstance(intervention, dict) else None
        )
        requested_targets = (
            intervention.get("targets")
            if isinstance(intervention, dict) else None
        )
        valid_requested = (
            isinstance(requested, list)
            and all(
                isinstance(edge, list)
                and len(edge) == 2
                and all(
                    not isinstance(index, bool) and isinstance(index, int)
                    for index in edge
                )
                for edge in requested
            )
        )
        expected_removed = (
            {(edge[0], edge[1]) for edge in requested}
            if valid_requested else set()
        )
        if (
            isinstance(construction, dict)
            and construction.get("direction_policy")
            == "undirected_bidirectional"
        ):
            expected_removed.update(
                (destination, source)
                for source, destination in tuple(expected_removed)
            )
        valid_removed = all(
            isinstance(edge, list)
            and len(edge) == 2
            and all(
                not isinstance(index, bool) and isinstance(index, int)
                for index in edge
            )
            for edge in removed
        )
        observed_removed = (
            {(edge[0], edge[1]) for edge in removed}
            if valid_removed else set()
        )
        expected_removed_rows = [
            [source, destination]
            for source, destination in sorted(expected_removed)
        ]
        if (
            not expected_removed
            or not valid_removed
            or removed != expected_removed_rows
            or observed_removed != expected_removed
            or targets != requested_targets
        ):
            return "topology evidence disagrees with the frozen intervention"
        observed = float(maximum)
    else:
        source = evidence.get("source_entity")
        target = evidence.get("target_entity")
        signal_delta = evidence.get("signal_delta")
        observed = evidence.get("target_max_abs_delta")
        controls = evidence.get("control_max_abs_deltas")
        if (
            isinstance(source, bool)
            or not isinstance(source, int)
            or isinstance(target, bool)
            or not isinstance(target, int)
            or source == target
            or not _finite_number(signal_delta)
            or float(signal_delta) == 0
            or not _finite_number(observed)
            or float(observed) < 0
            or not isinstance(controls, dict)
            or not controls
            or not all(_finite_number(value) for value in controls.values())
            or any(float(value) < 0 for value in controls.values())
            or any(
                float(value) > float(threshold) for value in controls.values()
            )
        ):
            return "neighbor evidence lacks its isolated intervention and controls"
        fixture = graph_execution_plan.get("fixture")
        intervention = (
            fixture.get("neighbor_intervention")
            if isinstance(fixture, dict) else None
        )
        expected_controls = (
            intervention.get("control_targets")
            if isinstance(intervention, dict) else None
        )
        if (
            not isinstance(intervention, dict)
            or source != intervention.get("source")
            or target != intervention.get("target")
            or signal_delta != intervention.get("delta")
            or not isinstance(expected_controls, list)
            or set(controls) != {str(value) for value in expected_controls}
        ):
            return "neighbor evidence disagrees with the frozen intervention"
        observed = float(observed)
    effect_present = float(observed) > float(threshold)
    if effect_present != expect_effect:
        return f"{kind} evidence disagrees with its claimed effect disposition"
    return None


def _graph_pass_evidence_problem(
    probe_id: str,
    evidence: dict,
    mechanism: dict,
    graph_execution_plan: dict,
) -> str | None:
    if probe_id == "HG-1":
        authority_paths = evidence.get("authority_paths")
        receipt = graph_execution_plan.get("alignment_runtime_receipt")
        authority_digests = (
            receipt.get("authority_digests")
            if isinstance(receipt, dict) else None
        )
        if (
            evidence.get("validator") != "validate_arch_contract_runtime.py"
            or not isinstance(authority_paths, list)
            or not authority_paths
            or any(
                not isinstance(path, str) or not path for path in authority_paths
            )
            or not isinstance(authority_digests, dict)
            or authority_paths != sorted(authority_digests)
        ):
            return "HG-1 lacks its runtime alignment authority evidence"
        return None
    if probe_id == "HG-2":
        interventions = evidence.get("parameter_interventions")
        fixture = graph_execution_plan.get("fixture")
        expected_interventions = (
            fixture.get("parameter_interventions")
            if isinstance(fixture, dict) else None
        )
        if not isinstance(interventions, list) or not interventions:
            return "HG-2 lacks graph-parameter interventions"
        if (
            not isinstance(expected_interventions, dict)
            or len(interventions) != len(expected_interventions)
        ):
            return "HG-2 interventions disagree with the frozen fixture"
        construction = graph_execution_plan.get("construction")
        representation = graph_execution_plan.get("representation")
        nominal_edges = _graph_edges(
            fixture.get("expected_graph") if isinstance(fixture, dict) else None,
            representation,
        )
        if not isinstance(construction, dict) or nominal_edges is None:
            return "HG-2 frozen construction authority is incoherent"
        roles: dict[str, tuple[str, dict, dict]] = {}
        for role in ("threshold", "cap"):
            declaration = construction.get(role)
            binding = (
                declaration.get("parameter")
                if isinstance(declaration, dict) else None
            )
            params_name = (
                binding.get("params_name") if isinstance(binding, dict) else None
            )
            if isinstance(params_name, str):
                roles[params_name] = (role, binding, declaration)
        authority = graph_execution_plan.get("parameter_authority")
        authority_bindings = (
            authority.get("bindings") if isinstance(authority, dict) else None
        )
        parameters = evidence.get("parameters")
        if (
            not isinstance(authority_bindings, dict)
            or set(authority_bindings) != set(roles)
            or not isinstance(parameters, list)
            or len(parameters) != len(roles)
        ):
            return "HG-2 lacks exact nominal parameter-authority rows"
        parameter_names: set[str] = set()
        for parameter in parameters:
            params_name = (
                parameter.get("params_name")
                if isinstance(parameter, dict) else None
            )
            role_binding = (
                roles.get(params_name) if isinstance(params_name, str) else None
            )
            authority_binding = (
                authority_bindings.get(params_name)
                if isinstance(params_name, str) else None
            )
            if (
                not isinstance(parameter, dict)
                or not isinstance(params_name, str)
                or params_name in parameter_names
                or role_binding is None
                or not isinstance(authority_binding, dict)
                or parameter.get("role") != role_binding[0]
                or parameter.get("carrier_value")
                != authority_binding.get("carrier_value")
                or parameter.get("params_value")
                != authority_binding.get("params_value")
                or parameter.get("paper_value")
                != authority_binding.get("paper_value")
                or parameter.get("source") != authority_binding.get("source")
                or parameter.get("suppressed") is not False
                or authority_binding.get("suppressed") is not False
                or parameter.get("duplicates") != []
                or authority_binding.get("duplicates") != []
                or parameter.get("planned_value")
                != role_binding[2].get("value")
                or parameter.get("planned_value")
                != authority_binding.get("params_value")
                or parameter.get("runtime_value")
                != authority_binding.get("params_value")
                or (
                    authority_binding.get("paper_value") is None
                    and authority_binding.get("params_value")
                    != authority_binding.get("carrier_value")
                )
                or (
                    authority_binding.get("paper_value") is not None
                    and authority_binding.get("paper_value")
                    != authority_binding.get("carrier_value")
                )
            ):
                return "HG-2 nominal parameter authority is forged or incomplete"
            parameter_names.add(params_name)
        if parameter_names != set(roles):
            return "HG-2 nominal parameter rows do not cover the plan"
        observed_names: set[str] = set()
        for intervention in interventions:
            if not isinstance(intervention, dict):
                return "HG-2 carries a malformed parameter intervention"
            expected = intervention.get(
                "expected_nominal_edge_symmetric_difference"
            )
            observed = intervention.get(
                "observed_nominal_edge_symmetric_difference"
            )
            params_name = intervention.get("params_name")
            expected_intervention = (
                expected_interventions.get(params_name)
                if isinstance(params_name, str) else None
            )
            role_binding = (
                roles.get(params_name) if isinstance(params_name, str) else None
            )
            expected_edges = _graph_edges(
                expected_intervention.get("expected_graph")
                if isinstance(expected_intervention, dict) else None,
                representation,
            )
            expected_difference = (
                len(expected_edges.symmetric_difference(nominal_edges))
                if expected_edges is not None else None
            )
            if (
                not isinstance(expected_intervention, dict)
                or not isinstance(params_name, str)
                or params_name in observed_names
                or role_binding is None
                or intervention.get("role") != role_binding[0]
                or intervention.get("nominal_value")
                != expected_intervention.get("nominal_value")
                or intervention.get("intervention_value")
                != expected_intervention.get("value")
                or intervention.get("intervention_value")
                != expected_intervention.get("alternate_value")
                or intervention.get("callable_parameter")
                != expected_intervention.get("callable_parameter")
                or expected_intervention.get("callable_parameter")
                != role_binding[1].get("callable_parameter")
                or expected_edges is None
                or intervention.get("expected_edge_count")
                != len(expected_edges)
                or intervention.get("constructed_edge_count")
                != len(expected_edges)
                or expected != expected_difference
                or isinstance(expected, bool)
                or not isinstance(expected, int)
                or expected <= 0
                or isinstance(observed, bool)
                or not isinstance(observed, int)
                or observed <= 0
                or observed != expected
                or intervention.get("runtime_value")
                != intervention.get("intervention_value")
            ):
                return "HG-2 intervention lacks an observed expected graph change"
            observed_names.add(params_name)
        if observed_names != set(expected_interventions):
            return "HG-2 does not exactly cover the frozen interventions"
        return None
    if probe_id == "HG-3":
        expected = evidence.get("expected_edge_count")
        constructed = evidence.get("constructed_edge_count")
        fixture = graph_execution_plan.get("fixture")
        construction = graph_execution_plan.get("construction")
        representation = graph_execution_plan.get("representation")
        expected_edges = _graph_edges(
            fixture.get("expected_graph") if isinstance(fixture, dict) else None,
            representation,
        )
        entity_ids = fixture.get("entity_ids") if isinstance(fixture, dict) else None
        if (
            not isinstance(construction, dict)
            or expected_edges is None
            or not isinstance(entity_ids, list)
            or evidence.get("representation") != representation
            or evidence.get("node_count") != len(entity_ids)
            or evidence.get("observed_edge_symmetric_difference") != 0
            or isinstance(expected, bool)
            or not isinstance(expected, int)
            or isinstance(constructed, bool)
            or not isinstance(constructed, int)
            or expected != constructed
            or expected != len(expected_edges)
            or evidence.get("self_loop_policy")
            != construction.get("self_loop_policy")
            or evidence.get("direction_policy")
            != construction.get("direction_policy")
            or evidence.get("cap_kind")
            != (
                construction.get("cap", {}).get("kind")
                if isinstance(construction.get("cap"), dict) else None
            )
        ):
            return "HG-3 lacks exact expected-topology agreement evidence"
        return _comparison_witness_evidence_problem(
            evidence,
            construction,
            fixture,
            representation,
            node_count=len(entity_ids),
        )
    if probe_id == "HG-4":
        return _effect_evidence_problem(
            evidence,
            kind="topology",
            expect_effect=True,
            graph_execution_plan=graph_execution_plan,
        )
    if probe_id == "HG-5":
        return _effect_evidence_problem(
            evidence,
            kind="neighbor",
            expect_effect=True,
            graph_execution_plan=graph_execution_plan,
        )
    if probe_id == "HG-6":
        permutation = evidence.get("permutation_new_to_old")
        restored = evidence.get("restored_max_abs_delta")
        threshold = evidence.get("effect_threshold")
        execution_problem = _execution_seed_tolerance_problem(
            evidence, graph_execution_plan
        )
        threshold_problem = _effect_threshold_authority_problem(
            evidence, graph_execution_plan
        )
        if (
            execution_problem is not None
            or threshold_problem is not None
            or not isinstance(permutation, list)
            or len(permutation) < 2
            or any(
                isinstance(value, bool) or not isinstance(value, int)
                for value in permutation
            )
            or sorted(permutation) != list(range(len(permutation)))
            or permutation == list(range(len(permutation)))
            or not isinstance(graph_execution_plan.get("fixture"), dict)
            or permutation
            != graph_execution_plan["fixture"].get("permutation")
            or not _finite_number(restored)
            or float(restored) < 0
            or not _finite_number(threshold)
            or float(restored) > float(threshold)
        ):
            return "HG-6 lacks coherent restored permutation evidence"
        return None
    ablation = mechanism.get("contribution_ablation")
    plan_ablation = graph_execution_plan.get("ablation")
    discriminator = (
        ablation.get("discriminating_probe_ref")
        if isinstance(ablation, dict) else None
    )
    kind = (
        "topology"
        if discriminator == _GRAPH_PROBE_REFS["HG-4"] else "neighbor"
    )
    expected_null_code = (
        "topology_response_absent"
        if kind == "topology" else "neighbor_response_absent"
    )
    if (
        not isinstance(plan_ablation, dict)
        or _execution_seed_tolerance_problem(
            evidence, graph_execution_plan
        ) is not None
        or evidence.get("ablation_kind") != plan_ablation.get("kind")
        or evidence.get("discriminating_probe_ref") != discriminator
        or evidence.get("discriminating_probe_ref")
        != plan_ablation.get("discriminating_probe_ref")
        or evidence.get("real_status") != "pass"
        or evidence.get("null_status") != "fail"
        or evidence.get("null_reason_code") != expected_null_code
    ):
        return "HG-7 lacks the declared real-versus-null disposition"
    return _effect_evidence_problem(
        evidence.get("real_evidence"),
        kind=kind,
        expect_effect=True,
        graph_execution_plan=graph_execution_plan,
    ) or _effect_evidence_problem(
        evidence.get("null_evidence"),
        kind=kind,
        expect_effect=False,
        graph_execution_plan=graph_execution_plan,
    )


def _bound_graph_probe_rows(
    probe_report: dict | None,
    spec: dict | None,
    probe_ids: tuple[str, ...],
    graph_execution_plan: dict | None,
) -> tuple[list[dict], str | None]:
    """Validate exactly one complete raw row per obligated graph probe."""
    methodology = (
        spec.get("methodology_replication_contract")
        if isinstance(spec, dict) else None
    )
    mechanism = (
        methodology.get("homogeneous_graph_mechanism")
        if isinstance(methodology, dict) else None
    )
    expected_bindings = (
        _expected_graph_bindings(mechanism)
        if isinstance(mechanism, dict) else None
    )
    if expected_bindings is None:
        return [], "typed graph owner or callable bindings are incoherent"
    if (
        not isinstance(graph_execution_plan, dict)
        or graph_execution_plan.get("status") != "ready"
    ):
        return [], "canonical graph execution plan authority is unavailable"
    try:
        from probes.graph_mechanism import graph_plan_trace  # noqa: PLC0415

        expected_trace = graph_plan_trace(graph_execution_plan)
    except (TypeError, ValueError):
        return [], "canonical graph execution plan cannot be digested"
    plan_construction = graph_execution_plan.get("construction")
    plan_execution = graph_execution_plan.get("execution")
    plan_alignment = graph_execution_plan.get("alignment")
    if not all(
        isinstance(value, dict)
        for value in (plan_construction, plan_execution, plan_alignment)
    ):
        return [], "canonical graph execution plan lacks callable authority"
    alignment_receipt = graph_execution_plan.get("alignment_runtime_receipt")
    liveness_receipt = graph_execution_plan.get("callable_liveness_receipt")
    requires_callable_liveness = any(
        probe_id != "HG-1" for probe_id in probe_ids
    )
    if (
        plan_alignment.get("status") != "pass"
        or not isinstance(alignment_receipt, dict)
        or alignment_receipt.get("status") != "pass"
        or alignment_receipt.get("verified") is not True
        or alignment_receipt.get("reason")
        != "stage_2d_runtime_alignment_verified"
    ):
        return [], "canonical graph alignment receipt is not verified"
    if requires_callable_liveness and (
        not isinstance(liveness_receipt, dict)
        or liveness_receipt.get("status") != "pass"
        or liveness_receipt.get("verified") is not True
        or liveness_receipt.get("reason")
        != "graph_callable_liveness_verified"
    ):
        return [], "canonical graph callable-liveness receipt is not verified"
    if (
        _graph_callable_key(plan_construction.get("callable"))
        != expected_bindings["HG-2"][1][0]
        or _graph_callable_key(plan_execution.get("callable"))
        != expected_bindings["HG-4"][1][0]
        or expected_trace.get("construction_feature_root")
        != mechanism["construction"].get("feature_input_root")
        or expected_trace.get("neighbor_signal_root")
        != mechanism["message_passing"].get("neighbor_signal_root")
        or expected_trace.get("output_root")
        != mechanism["message_passing"].get("output_root")
    ):
        return [], "canonical graph plan disagrees with MethodSpec bindings"
    preparation = plan_alignment.get("preparation_callable")
    if not isinstance(preparation, dict):
        return [], "canonical graph plan lacks alignment callable authority"
    prep_module = preparation.get("module")
    prep_name = preparation.get("name")
    alignment_callable = (
        f"{prep_module}:{prep_name}"
        if prep_module == "method.training"
        and isinstance(prep_name, str)
        and prep_name.isidentifier()
        and not keyword.iskeyword(prep_name)
        else None
    )
    if alignment_callable is None:
        return [], "canonical graph alignment callable is incoherent"
    expected_bindings["HG-1"] = (
        expected_bindings["HG-1"][0], [alignment_callable]
    )
    groundings = graph_execution_plan.get("groundings")
    if not isinstance(groundings, dict):
        return [], "canonical graph plan lacks exact obligation groundings"
    applicable_plan_ids = {"HG-1", "HG-2", "HG-3", "HG-4", "HG-5"}
    if mechanism.get("permutation_applicability") == "equivariant":
        applicable_plan_ids.add("HG-6")
    if isinstance(mechanism.get("contribution_ablation"), dict):
        applicable_plan_ids.add("HG-7")
    for probe_id in applicable_plan_ids:
        owner, callables = expected_bindings[probe_id]
        grounding = groundings.get(_GRAPH_PROBE_REFS[probe_id])
        if (
            not isinstance(grounding, dict)
            or grounding.get("element_ids") != [owner]
            or grounding.get("bound_callables") != callables
        ):
            return [], (
                f"canonical graph plan grounding disagrees for {probe_id}"
            )

    verdicts = (probe_report or {}).get("verdicts") or []
    if not isinstance(verdicts, list):
        return [], "probe report verdicts are not a list"

    rows: list[dict] = []
    from probe_spec_join import join_verdict  # noqa: PLC0415

    for probe_id in probe_ids:
        raw_rows = [
            row for row in verdicts
            if isinstance(row, dict) and row.get("probe_id") == probe_id
        ]
        if len(raw_rows) != 1:
            return [], f"{probe_id} requires exactly one raw verdict row"
        row = raw_rows[0]
        expected_ref = _GRAPH_PROBE_REFS[probe_id]
        expected_owner, expected_callables = expected_bindings[probe_id]
        if row.get("probe_ref") != expected_ref:
            return [], f"{probe_id} carries the wrong exact probe ref"
        if row.get("element_ids") != [expected_owner]:
            return [], f"{probe_id} carries the wrong exact owner element"
        bound_callables = row.get("bound_callables")
        if bound_callables != expected_callables:
            return [], f"{probe_id} carries the wrong exact callable bindings"
        if not join_verdict(row, spec).reference_qualified:
            return [], f"{probe_id} is not reference-qualified by MethodSpec"
        evidence = _graph_evidence(row)
        trace = evidence.get("trace") if isinstance(evidence, dict) else None
        if not isinstance(trace, dict):
            return [], f"{probe_id} carries no structured graph plan trace"
        plan_digest = trace.get("execution_plan_digest")
        fixture_digest = trace.get("fixture_digest")
        if not _is_sha256_digest(plan_digest) or not _is_sha256_digest(
            fixture_digest
        ):
            return [], f"{probe_id} carries invalid plan or fixture digests"
        if trace != expected_trace:
            return [], f"{probe_id} trace disagrees with the canonical graph plan"
        if row.get("verdict") == "pass":
            assert evidence is not None
            evidence_problem = _graph_pass_evidence_problem(
                probe_id, evidence, mechanism, graph_execution_plan
            )
            if evidence_problem is not None:
                return [], evidence_problem
        rows.append(row)
    return rows, None


def _graph_probe_axis(
    probe_report: dict | None,
    spec: dict | None,
    axis: str,
    graph_execution_plan: dict | None = None,
) -> dict:
    """Derive one rung of the graph evidence ladder without implication.

    HG-2/3 establish parameter/construction truth, HG-1 establishes the reused
    relational prerequisite, HG-4/5/6 establish mechanism liveness, and only
    HG-7 can establish contribution.  A pass at one rung never fills another.
    """
    methodology = (
        spec.get("methodology_replication_contract")
        if isinstance(spec, dict) else None
    )
    mechanism = (
        methodology.get("homogeneous_graph_mechanism")
        if isinstance(methodology, dict) else None
    )
    if not isinstance(mechanism, dict):
        return {
            "status": "undetermined",
            "reasons": [{
                "code": "graph_contract_missing",
                "message": "the typed homogeneous graph contract is missing",
            }],
        }

    applicability = mechanism.get("permutation_applicability")
    refs = mechanism.get("probe_refs")
    if (
        applicability not in {"equivariant", "not_applicable"}
        or not isinstance(refs, dict)
    ):
        return {
            "status": "undetermined",
            "reasons": [{
                "code": "graph_contract_incoherent",
                "message": (
                    "the graph contract does not carry an exact permutation "
                    "applicability and probe-ref block"
                ),
            }],
        }

    expected_by_axis = {
        "graph_alignment": ("HG-1",),
        "graph_construction": ("HG-1", "HG-2", "HG-3"),
        "mechanism": ("HG-1", "HG-2", "HG-3", "HG-4", "HG-5"),
        "contribution": (
            "HG-1", "HG-2", "HG-3", "HG-4", "HG-5", "HG-7"
        ),
    }
    expected = expected_by_axis[axis]
    if applicability == "equivariant" and axis in {
        "mechanism", "contribution",
    }:
        expected = (
            (*expected[:-1], "HG-6", expected[-1])
            if axis == "contribution"
            else (*expected, "HG-6")
        )

    required_contract_ids = tuple(
        probe_id for probe_id in expected if probe_id != "HG-1"
    )
    for probe_id in required_contract_ids:
        field = _GRAPH_CONTRACT_REF_FIELDS[probe_id]
        if refs.get(field) != _GRAPH_PROBE_REFS[probe_id]:
            return {
                "status": "undetermined",
                "reasons": [{
                    "code": f"incoherent_{axis}_contract",
                    "message": (
                        f"the typed graph contract does not declare the exact "
                        f"{_GRAPH_PROBE_REFS[probe_id]} obligation"
                    ),
                }],
            }
    expected_permutation_ref = (
        _GRAPH_PROBE_REFS["HG-6"] if applicability == "equivariant" else None
    )
    if refs.get("permutation") != expected_permutation_ref:
        return {
            "status": "undetermined",
            "reasons": [{
                "code": f"incoherent_{axis}_permutation_contract",
                "message": (
                    "permutation applicability and its exact graph probe ref "
                    "disagree"
                ),
            }],
        }
    if applicability == "not_applicable":
        permutation_rows = [
            row for row in (probe_report or {}).get("verdicts") or []
            if isinstance(row, dict) and row.get("probe_id") == "HG-6"
        ]
        if permutation_rows and not (
            len(permutation_rows) == 1
            and permutation_rows[0].get("probe_ref")
            == _GRAPH_PROBE_REFS["HG-6"]
            and permutation_rows[0].get("verdict") == "not_applicable"
            and permutation_rows[0].get("element_ids") == []
            and permutation_rows[0].get("bound_callables") == []
        ):
            return {
                "status": "undetermined",
                "reasons": [{
                    "code": f"incoherent_{axis}_permutation_evidence",
                    "message": (
                        "a not-applicable permutation contract cannot carry "
                        "bound, passing, duplicated, or otherwise mixed HG-6 "
                        "evidence"
                    ),
                }],
            }
    if axis == "contribution":
        ablation = mechanism.get("contribution_ablation")
        ablation_kind = (
            ablation.get("kind") if isinstance(ablation, dict) else None
        )
        discriminator = (
            ablation.get("discriminating_probe_ref")
            if isinstance(ablation, dict) else None
        )
        input_kinds = {"empty_graph", "identity_graph", "permuted_graph"}
        callable_kinds = {"removed_message_passing", "non_graph_decoder"}
        expected_ablation_keys = {
            "kind", "element_id", "discriminating_probe_ref",
        }
        callable_ref = None
        if ablation_kind in callable_kinds and isinstance(ablation, dict):
            expected_ablation_keys.update({
                "callable",
                "graph_parameter",
                "neighbor_signal_parameter",
                "output_root",
            })
            callable_ref = ablation.get("callable")
        callable_is_exact = (
            ablation_kind in input_kinds
            or (
                isinstance(callable_ref, dict)
                and set(callable_ref) == {"module", "qualname"}
                and callable_ref.get("module") in {
                    "method.model", "method.training", "method.method",
                }
                and isinstance(callable_ref.get("qualname"), str)
                and callable_ref["qualname"].isidentifier()
                and "." not in callable_ref["qualname"]
            )
        )
        if (
            not isinstance(ablation, dict)
            or set(ablation) != expected_ablation_keys
            or ablation_kind not in input_kinds | callable_kinds
            or not isinstance(ablation.get("element_id"), str)
            or not ablation["element_id"].strip()
            or ablation["element_id"] != ablation["element_id"].strip()
            or not callable_is_exact
            or (
                ablation_kind in callable_kinds
                and (
                    not isinstance(mechanism.get("message_passing"), dict)
                    or ablation.get("output_root")
                    != mechanism["message_passing"].get("output_root")
                )
            )
            or refs.get("contribution_ablation")
            != _GRAPH_PROBE_REFS["HG-7"]
            or discriminator not in {
                _GRAPH_PROBE_REFS["HG-4"], _GRAPH_PROBE_REFS["HG-5"]
            }
        ):
            return {
                "status": "undetermined",
                "reasons": [{
                    "code": "graph_contribution_control_missing",
                    "message": (
                        "contribution requires a paper-grounded null with an "
                        "exact topology or neighbor-sensitivity discriminator"
                    ),
                }],
            }
    rows, evidence_problem = _bound_graph_probe_rows(
        probe_report, spec, expected, graph_execution_plan
    )
    if evidence_problem is not None:
        return {
            "status": "undetermined",
            "reasons": [{
                "code": f"incoherent_{axis}_evidence",
                "message": evidence_problem,
            }],
        }
    if not rows:
        return {
            "status": "undetermined",
            "reasons": [{
                "code": f"no_bound_{axis}_probe",
                "message": (
                    f"no exact contract-bound graph probe established "
                    f"{axis.replace('_', ' ')} evidence"
                ),
            }],
        }

    by_id = {
        probe_id: [
            str(row.get("verdict") or "")
            for row in rows
            if str(row.get("probe_id") or "") == probe_id
        ]
        for probe_id in expected
    }
    all_statuses = {status for statuses in by_id.values() for status in statuses}
    if all_statuses & {"fail", "flag_for_researcher"}:
        return {
            "status": "not_demonstrated",
            "reasons": [{
                "code": f"bound_{axis}_probe_rejected",
                "message": (
                    f"at least one exact contract-bound graph probe rejected "
                    f"{axis.replace('_', ' ')}"
                ),
            }],
        }

    if all_statuses == {"not_applicable"}:
        return {
            "status": "not_applicable",
            "reasons": [{
                "code": f"bound_{axis}_probe_not_applicable",
                "message": (
                    f"every exact contract-bound {axis.replace('_', ' ')} "
                    "probe was not applicable"
                ),
            }],
        }

    # Every required rung has exactly one passing row.  Duplicate or mixed
    # rows are inconclusive rather than letting one selected pass certify a
    # tampered report.  A typed not-applicable permutation creates no HG-6
    # obligation; an equivariant contract requires HG-6 to pass.
    complete = all(statuses == ["pass"] for statuses in by_id.values())
    if not complete:
        return {
            "status": "undetermined",
            "reasons": [{
                "code": f"inconclusive_{axis}_probe",
                "message": (
                    f"exact contract-bound graph probes produced incomplete "
                    f"or inconclusive {axis.replace('_', ' ')} evidence"
                ),
            }],
        }

    messages = {
        "graph_construction": (
            "parameter authority and graph construction semantics both passed"
        ),
        "graph_alignment": (
            "the reused relational contract established entity/index alignment"
        ),
        "mechanism": (
            "topology and neighbor interventions demonstrated graph mechanism "
            "liveness under the declared permutation rule"
        ),
        "contribution": (
            "the real graph mechanism passed a discriminating check that its "
            "paper-justified null failed"
        ),
    }
    return {
        "status": "demonstrated",
        "reasons": [{
            "code": f"bound_{axis}_probe_result",
            "message": messages[axis],
        }],
    }


def _mechanism_evidence_axis(
    probe_report: dict | None,
    spec: dict | None = None,
    graph_execution_plan: dict | None = None,
) -> dict:
    """Mechanism status from bound behavioral probes only.

    A metric or universal hygiene pass is never mechanism evidence.  R2C-087
    supplies exact element binding; until then the honest result is
    undetermined rather than inferred from execution or skill.
    """
    if _has_homogeneous_graph_contract(spec):
        return _graph_probe_axis(
            probe_report, spec, "mechanism", graph_execution_plan
        )

    candidates = [
        row for row in (probe_report or {}).get("verdicts") or []
        if isinstance(row, dict)
        and row.get("tier") == "behavioral"
        and isinstance(row.get("element_ids"), list)
        and bool(row["element_ids"])
    ]
    if _uses_reference_qualified_contract(spec):
        from probe_spec_join import join_verdict  # noqa: PLC0415

        bound = [
            row for row in candidates
            if join_verdict(row, spec).reference_qualified
        ]
    else:
        # Archived contract/report reads retain R2C-086's element-id behavior.
        bound = candidates
    if not bound:
        return {
            "status": "undetermined",
            "reasons": [{
                "code": "no_bound_behavioral_probe",
                "message": "no bound behavioral probe established mechanism evidence",
            }],
        }
    verdicts = {str(row.get("verdict") or "") for row in bound}
    if verdicts & {"fail", "flag_for_researcher"}:
        status = "not_demonstrated"
        message = "at least one bound behavioral probe rejected the mechanism"
    elif "pass" in verdicts:
        status = "demonstrated"
        message = "one or more bound behavioral probes demonstrated the mechanism"
    elif verdicts == {"not_applicable"}:
        status = "not_applicable"
        message = "every bound conditional mechanism probe was not applicable"
    else:
        status = "undetermined"
        message = "bound behavioral probes produced no conclusive mechanism result"
    return {
        "status": status,
        "reasons": [{"code": "bound_behavioral_probe_result",
                     "message": message}],
    }


def _structured_demo_entries(
    demo_verdict: dict,
    probe_report: dict | None,
    spec: dict | None = None,
    graph_execution_plan: dict | None = None,
) -> tuple[list[dict], list[dict], dict]:
    """Four-axis R2C-086 delivery projection for schema-2 artifacts."""
    raw_status = demo_verdict.get("evidence_status")
    axes = {
        key: dict(value)
        for key, value in (raw_status.items() if isinstance(raw_status, dict)
                           else [])
        if isinstance(value, dict)
    }
    axes["mechanism"] = _mechanism_evidence_axis(
        probe_report, spec, graph_execution_plan
    )
    if _has_homogeneous_graph_contract(spec):
        # Preserve the evidence ladder as orthogonal fields.  Their order is
        # normalized below so the manifest and REPORT remain easy to compare.
        axes["graph_construction"] = _graph_probe_axis(
            probe_report, spec, "graph_construction", graph_execution_plan)
        axes["graph_alignment"] = _graph_probe_axis(
            probe_report, spec, "graph_alignment", graph_execution_plan)
        axes["contribution"] = _graph_probe_axis(
            probe_report, spec, "contribution", graph_execution_plan)
        axes = {
            key: axes[key]
            for key in (
                "execution", "evaluation_validity", "graph_construction",
                "graph_alignment", "mechanism", "contribution", "skill",
                "paper_benchmark",
            )
            if key in axes
        }
    record = {
        "schema_version": str(demo_verdict.get("schema_version") or "2.0.0"),
        "verdict": str(demo_verdict.get("verdict") or "undetermined"),
        "evidence_status": axes,
    }
    for key in ("evidence_line", "evidence_cell", "decided_by", "reason"):
        if demo_verdict.get(key) is not None:
            record[key] = demo_verdict[key]

    evaluation = axes.get("evaluation_validity") or {}
    skill = axes.get("skill") or {}
    evaluation_status = str(evaluation.get("status") or "unresolved")
    skill_status = str(skill.get("status") or "undetermined")
    reasons: list[dict] = []
    disclosures: list[dict] = []
    if evaluation_status == "invalid":
        reasons.append({
            "source": "demo_verdict",
            "id": "demo_evaluation_invalid",
            "message": (
                "the notebook ran end to end, but the scored evaluation is "
                "invalid; execution is retained as a separate fact and no "
                "task-skill conclusion is allowed"
            ),
        })
    elif evaluation_status == "unresolved":
        disclosures.append({
            "source": "demo_verdict",
            "id": "demo_evaluation_unresolved",
            "message": (
                "the notebook ran end to end, but evaluation validity is "
                "unresolved, so demonstrated task skill remains undetermined"
            ),
        })

    if skill_status == "not_demonstrated":
        detail = str(demo_verdict.get("evidence_line") or "").strip()
        reasons.append({
            "source": "demo_verdict",
            "id": "demo_skill_not_demonstrated",
            "message": (
                "the valid demo evaluation did not demonstrate task skill "
                "against every required family comparator"
                + (f": {detail}" if detail else "")
            ),
        })
    elif skill_status == "undetermined" and evaluation_status == "valid":
        disclosures.append({
            "source": "demo_verdict",
            "id": "demo_skill_undetermined",
            "message": (
                "the evaluation is valid, but missing, non-finite, or "
                "tolerance-inconclusive comparison evidence leaves task "
                "skill undetermined"
            ),
        })

    marker = demo_verdict.get("presentation_marker")
    marker_line = str(marker.get("line") or "") if isinstance(marker, dict) else ""
    if "demo verdict: pass" in marker_line.lower() \
            and skill_status != "demonstrated":
        disclosures.append({
            "source": "demo_verdict",
            "id": "demo_presentation_disagreement",
            "message": (
                "the notebook printed a PASS presentation line, but the "
                "pipeline-owned structured evidence did not demonstrate "
                "skill; the structured result is authoritative"
            ),
        })
    return reasons, disclosures, record


def _demo_verdict_entries(
    demo_verdict: dict | None,
    probe_report: dict | None,
    spec: dict | None = None,
    graph_execution_plan: dict | None = None,
) -> tuple[list[dict], list[dict], dict | None]:
    """(demoting reasons, disclosures, compact manifest record) from the
    post-smoke demo verdict artifact.

    A missing/unreadable artifact means the pass never ran — ([], [], None),
    so pre-feature runs derive as before (identical output modulo the schema
    version string). The verdict's own semantics:
    `failed` demotes with the design's plain-language pattern and the quoted
    evidence line; `undetermined` discloses (never demotes — absence of a
    marker match is our coverage, not artifact evidence); `succeeded` adds
    nothing. A verdict/UB-6 disagreement in either direction is disclosed as
    a probe-bug finding we own (design: where both exist they should agree).
    """
    if not isinstance(demo_verdict, dict):
        return [], [], None
    if (str(demo_verdict.get("schema_version") or "").startswith("2.")
            and isinstance(demo_verdict.get("evidence_status"), dict)):
        return _structured_demo_entries(
            demo_verdict, probe_report, spec, graph_execution_plan
        )
    verdict = str(demo_verdict.get("verdict") or "")
    if verdict not in ("succeeded", "failed", "undetermined"):
        return [], [], None

    record: dict = {"verdict": verdict}
    for key in ("evidence_line", "evidence_cell", "decided_by", "reason"):
        if demo_verdict.get(key) is not None:
            record[key] = demo_verdict[key]

    reasons: list[dict] = []
    disclosures: list[dict] = []
    evidence_line = str(demo_verdict.get("evidence_line") or "").strip()
    evidence_cell = demo_verdict.get("evidence_cell")

    if verdict == "failed":
        detail = evidence_line or str(demo_verdict.get("marker_gloss") or "")
        reason = {
            "source": "demo_verdict",
            "id": "demo_failed",
            "message": (
                "the notebook runs end to end, but its demonstration does "
                "not succeed — "
                + (f"its own executed output says: \"{detail}\"" if detail
                   else "the headline demo's executed output reports failure")),
        }
        if isinstance(evidence_cell, int):
            reason["evidence"] = f"notebook.ipynb: output of cell {evidence_cell}"
        reasons.append(reason)
    elif verdict == "undetermined":
        disclosures.append({
            "source": "demo_verdict",
            "id": "demo_not_checked",
            "message": (
                "the headline demo outcome was not machine-checked: "
                + str(demo_verdict.get("reason")
                      or "no demo marker could bind to this notebook")),
        })
        finding = demo_verdict.get("kit_coverage_finding")
        if isinstance(finding, dict) and finding.get("message"):
            disclosures.append({
                "source": "demo_verdict",
                "id": str(finding.get("id") or "demo_markers_missing"),
                "message": str(finding["message"]),
            })

    # Probe-bug disclosure: UB-6 reads the same executed outputs, so a hard
    # disagreement means one of OUR layers is wrong — surfaced, never silent.
    ub6 = _ub6_verdicts(probe_report)
    if verdict == "failed" and "pass" in ub6:
        disclosures.append({
            "source": "demo_verdict",
            "id": "demo_probe_disagreement",
            "message": (
                "probe-bug finding we own: the demo verdict reads failed "
                "while the executed-notebook sanity check passed — one of "
                "the two layers misread this notebook's outputs; the "
                "demotion stands until a human rules"),
        })
    elif verdict == "succeeded" and "fail" in ub6:
        disclosures.append({
            "source": "demo_verdict",
            "id": "demo_probe_disagreement",
            "message": (
                "probe-bug finding we own: the demo verdict reads succeeded "
                "while the executed-notebook sanity check failed — one of "
                "the two layers misread this notebook's outputs; the "
                "probe's demotion stands until a human rules"),
        })
    return reasons, disclosures, record


def _required_structured_demo_reason(
    demo_verdict: dict | None,
    *,
    structured_demo_required: bool,
    homogeneous_graph_required: bool = False,
) -> dict | None:
    """Demote absent or malformed schema-2 evidence when its contract requires it.

    Schema-less and schema-1 records are historical compatibility artifacts;
    their marker semantics remain unchanged for graph-free families. A typed
    homogeneous graph contract is fresh and always requires schema-2 evidence,
    so it cannot take that compatibility arm.
    """
    if not structured_demo_required and not homogeneous_graph_required:
        return None
    if not isinstance(demo_verdict, dict):
        subject = (
            "this homogeneous graph contract"
            if homogeneous_graph_required else "this family"
        )
        return {
            "source": "demo_verdict",
            "id": "structured_demo_evidence_missing",
            "message": (
                f"{subject} requires structured demo evidence, but no readable "
                "schema-2 demo verdict artifact was recorded"
            ),
        }

    version = str(demo_verdict.get("schema_version") or "")
    legacy = (not version or version.startswith("1.")) and (
        demo_verdict.get("verdict") in {"succeeded", "failed", "undetermined"}
    )
    if legacy and not homogeneous_graph_required:
        return None

    axes = demo_verdict.get("evidence_status")
    allowed = {
        "execution": {"completed", "failed", "undetermined"},
        "evaluation_validity": {
            "valid", "invalid", "unresolved", "not_applicable",
        },
        "mechanism": {
            "demonstrated", "not_demonstrated", "undetermined",
            "not_applicable",
        },
        "skill": {
            "demonstrated", "not_demonstrated", "undetermined",
            "not_applicable",
        },
        "paper_benchmark": {
            "reproduced", "not_reproduced", "not_assessed", "undetermined",
        },
    }
    well_formed = version.startswith("2.") and isinstance(axes, dict) and all(
        isinstance(axes.get(name), dict)
        and axes[name].get("status") in statuses
        for name, statuses in allowed.items()
    )
    if well_formed:
        if demo_verdict.get("decided_by") == "structured_demo_skill_error":
            return {
                "source": "demo_verdict",
                "id": "structured_demo_evidence_error",
                "message": (
                    "the pipeline-owned structured demo-evidence pass failed, "
                    "so evaluation validity and task skill could not be derived"
                ),
            }
        return None
    subject = (
        "this homogeneous graph contract"
        if homogeneous_graph_required else "this family"
    )
    return {
        "source": "demo_verdict",
        "id": "structured_demo_evidence_malformed",
        "message": (
            f"{subject} requires structured demo evidence, but its demo "
            "verdict artifact is not a complete supported schema-2 record"
        ),
    }


def _missing_graph_demo_record(
    probe_report: dict | None,
    spec: dict | None,
    graph_execution_plan: dict | None,
    problem: dict,
) -> dict:
    """Keep the graph ladder visible when its schema-2 demo record is absent.

    The missing or malformed demo artifact still demotes delivery.  Graph axes
    come from their independent probe authority, while execution, evaluation,
    skill, and paper-scale status stay unresolved rather than being inferred.
    This gives the manifest and REPORT a complete, non-conflated disclosure
    surface even on the failure path.
    """

    unresolved_reason = {
        "code": str(problem.get("id") or "structured_demo_evidence_missing"),
        "message": str(
            problem.get("message")
            or "the homogeneous graph delivery has no structured demo evidence"
        ),
    }
    return {
        "schema_version": "2.0.0",
        "verdict": "undetermined",
        "decided_by": "delivery_graph_evidence_fallback",
        "reason": unresolved_reason["message"],
        "evidence_status": {
            "execution": {
                "status": "undetermined",
                "reasons": [dict(unresolved_reason)],
            },
            "evaluation_validity": {
                "status": "unresolved",
                "reasons": [dict(unresolved_reason)],
            },
            "graph_construction": _graph_probe_axis(
                probe_report, spec, "graph_construction", graph_execution_plan
            ),
            "graph_alignment": _graph_probe_axis(
                probe_report, spec, "graph_alignment", graph_execution_plan
            ),
            "mechanism": _mechanism_evidence_axis(
                probe_report, spec, graph_execution_plan
            ),
            "contribution": _graph_probe_axis(
                probe_report, spec, "contribution", graph_execution_plan
            ),
            "skill": {
                "status": "undetermined",
                "reasons": [dict(unresolved_reason)],
            },
            "paper_benchmark": {
                "status": "undetermined",
                "reasons": [dict(unresolved_reason)],
            },
        },
    }


def _normalize_stubs(stubbed_elements: list | None) -> list[dict]:
    """Defensive copy of the stub records the driver loaded (the artifact
    reader owns schema validation; the label stays pure and tolerant)."""
    records = []
    for s in stubbed_elements or []:
        if not isinstance(s, dict) or not s.get("element_id"):
            continue
        role = "core" if str(s.get("role") or "") == "core" else "supporting"
        rec = {
            "element_id": str(s["element_id"]),
            "role": role,
            "work_order": str(s.get("work_order") or ""),
        }
        if s.get("stub_path"):
            rec["stub_path"] = str(s["stub_path"])
        records.append(rec)
    # Core stubs first: the design's §3.2 rule — the missing core mechanism
    # is the headline, never buried under supporting-component noise.
    records.sort(key=lambda r: (r["role"] != "core", r["element_id"]))
    return records


def _stub_reason(rec: dict) -> dict:
    what = ("the paper's core mechanism" if rec["role"] == "core"
            else "supporting component")
    pointer = f" — work order: `{rec['work_order']}`" if rec["work_order"] else ""
    return {
        "source": "partial_delivery",
        "id": rec["element_id"],
        "message": (f"PARTIAL delivery: {what} `{rec['element_id']}` is NOT "
                    f"implemented; it ships as a self-identifying stub"
                    f"{pointer}"),
    }


def _normalize_core_approximations(core_approximations: list | None) -> list[dict]:
    records = []
    for a in core_approximations or []:
        if not isinstance(a, dict) or not a.get("element_id"):
            continue
        records.append({
            "element_id": str(a["element_id"]),
            "replication_status": str(a.get("replication_status") or ""),
        })
    records.sort(key=lambda r: r["element_id"])
    return records


def _approximation_reason(rec: dict) -> dict:
    return {
        "source": "approximated_core",
        "id": rec["element_id"],
        "message": (f"the paper's core mechanism `{rec['element_id']}` is "
                    f"implemented as an approved approximation "
                    f"({rec['replication_status'] or 'not must_replicate'}) — "
                    f"the core cannot be certified as replicated, so the "
                    f"package delivers as draft however the probes came back"),
    }


def apply_neutral_plan_cap(delivery: dict, paradigm: str | None = None) -> dict:
    """Cap a neutral-build-plan delivery at `uncertified_new_territory`.

    In place and idempotent (keyed on the disclosure id), so the driver can
    re-apply it after its post-derivation demotions — an infra failure on a
    neutral-plan run must not RAISE the label from uncertified (rank 2) to
    draft (rank 3). Demoting reasons are left in `reasons`: the schema puts
    no reasons-empty constraint on uncertified labels, and hiding a failure
    to satisfy a wording convention would be the R_0-shipped move in
    reverse."""
    if delivery.get("label") not in (LABEL_VERIFIED, LABEL_DRAFT):
        return delivery
    delivery["label"] = LABEL_UNCERTIFIED
    disclosures = delivery.setdefault("disclosures", [])
    if not any(isinstance(d, dict) and d.get("id") == NEUTRAL_PLAN_CAP_ID
               for d in disclosures):
        disclosures.insert(0, {
            "source": "build_plan",
            "id": NEUTRAL_PLAN_CAP_ID,
            "message": (
                "this package was built on the generic provisional build "
                "plan — no family-specific build conventions were enforced "
                "on it — so the delivery caps at uncertified — new "
                "territory regardless of probe outcomes; an inherited or "
                "promoted family build plan restores the full label range"),
        })
    if not delivery.get("missing_probe_family"):
        delivery["missing_probe_family"] = str(paradigm) if paradigm else "unknown"
    return delivery


def unprobeable_core_novelty(probe_report: dict | None,
                             spec: dict | None) -> list:
    """Every joined core-novelty element left unprobed by this run (R2C-047).

    Returns the `JoinedElement`s behind the cap, in report order and
    deduplicated by element id, so the disclosure can name the mechanism in the
    paper's own words rather than pointing at a probe id the researcher has
    never seen.

    Empty whenever the join cannot fire: no spec, no methodology contract, no
    element ids on the verdicts, or the join module unavailable. The cap can
    only ever narrow a label, so failing open here fails toward today's
    behavior."""
    if not spec or not isinstance(probe_report, dict):
        return []
    try:
        from probe_spec_join import join_verdict  # noqa: PLC0415
    except Exception:  # noqa: BLE001 — gating must not die on an import
        return []
    out: list = []
    seen: set = set()
    for verdict in probe_report.get("verdicts") or []:
        if not isinstance(verdict, dict):
            continue
        try:
            join = join_verdict(verdict, spec)
        except Exception:  # noqa: BLE001 — a malformed verdict is not a cap
            continue
        if not join.caps_the_label:
            continue
        for element in join.core_novelty_elements:
            if element.element_id in seen:
                continue
            seen.add(element.element_id)
            out.append(element)
    return out


def apply_unprobeable_core_novelty_cap(delivery: dict, elements: list) -> dict:
    """Cap a delivery whose core novelty went unprobed at
    `uncertified_new_territory`.

    In place and idempotent (keyed on the disclosure id), same as
    `apply_neutral_plan_cap`, so the driver can re-apply it after its post-hoc
    demotions without an infra failure RAISING a capped label from uncertified
    (rank 2) to draft (rank 3). A cap, never a deduction: existing demoting
    reasons stay in `reasons`, fully disclosed."""
    if not elements:
        return delivery
    if delivery.get("label") in (LABEL_VERIFIED, LABEL_DRAFT):
        delivery["label"] = LABEL_UNCERTIFIED
    disclosures = delivery.setdefault("disclosures", [])
    if any(isinstance(d, dict) and d.get("id") == UNPROBEABLE_CORE_CAP_ID
           for d in disclosures):
        return delivery
    named = "; ".join(e.in_the_papers_words() for e in elements)
    plural = "mechanisms" if len(elements) > 1 else "mechanism"
    disclosures.insert(0, {
        "source": "probe_battery",
        "id": UNPROBEABLE_CORE_CAP_ID,
        "message": (
            f"the behavioral checks could not bind to {len(elements)} core "
            f"{plural} this run's own methodology contract requires exactly, so "
            f"the delivery carries no evidence about the paper's central "
            f"contribution and caps at uncertified — new territory: {named}"),
    })
    return delivery


def derive_delivery_label(probe_report: dict | None,
                          review_report: dict | None,
                          paradigm: str | None = None,
                          stubbed_elements: list | None = None,
                          core_approximations: list | None = None,
                          demo_verdict: dict | None = None,
                          structured_demo_required: bool = False,
                          neutral_build_plan: bool = False,
                          spec: dict | None = None,
                          graph_execution_plan: dict | None = None) -> dict:
    """The delivery verdict: label + every reason and disclosure behind it.

    Returns a versioned label record with explicit graph-evidence
    applicability, completeness state, reasons, disclosures, and probe counts.
    Reasons demote to draft. Disclosures are the honest unchecked or imperfect
    surface that ships with a verified label.

    Evaluated in order (design notes §2 + partial delivery §3.3 + the
    approximated-core rule, maintainer-approved 2026-07-05 afternoon):

    1. Any demoting reason → `draft` (unchanged), completeness causes
       (stubs, core approximations) recorded alongside so they stay
       visible.
    2. Otherwise, stubbed elements or approved CORE approximations
       present → never `verified`: `draft` with the distinct cause
       (`partial_delivery` / `approximated_core`) when contribution
       evidence exists, else `uncertified_new_territory` (causes
       disclosed, both facts recorded). The SRL 2026-07-05 case:
       a contribution-tier probe pass certifies a family behavior, not
       an approximated core — when the core mechanism itself is an
       approximation, certifying the paper's mechanism is impossible by
       construction. Supporting and demo-scale approximations never
       reach this function (the driver filters on role).
    3. Otherwise, contribution evidence present → `verified`.
    4. Otherwise → `uncertified_new_territory`, with
       `missing_probe_family` recording the taxonomy node id (`paradigm`)
       that lacked bindable contribution probes — the growth-engine
       demand signal.

    `demo_verdict` is the post-smoke demo-success artifact
    (.pipeline/demo_verdict.json): `failed` joins the demoters with its own
    reason id (`demo_failed`), `undetermined` joins the disclosures, and the
    compact record rides in the output as `demo_verdict` so the manifest can
    never lose which way the headline demo went. None (no artifact) changes
    nothing for legacy families; `structured_demo_required` below makes
    absence fail closed for a family that declares the schema-2 contract. A
    typed homogeneous graph contract also requires schema-2 evidence directly,
    independent of taxonomy routing.

    `structured_demo_required` is resolved from the run's effective taxonomy.
    It changes only the absence/malformed-artifact path: a family declaring
    `demo_skill` fails closed, while schema-1 artifacts and families without
    the contract retain their legacy behavior.

    `neutral_build_plan` is the build-context provenance fact (R2C-032) —
    the run's plan resolved to the generic provisional plan. True caps the
    result at `uncertified_new_territory` via `apply_neutral_plan_cap`;
    False (every committed family, every pack whose plan inherits a
    committed ancestor) changes nothing — existing runs derive as before,
    identical output modulo the schema version string.

    `spec` is the run's own method spec, read only for the unprobeable
    core-novelty cap (R2C-047): an `unprobeable` verdict bound to a
    core_methodology element the contract marks must_replicate caps the result
    at `uncertified_new_territory`. None, or a spec with no methodology
    contract, or verdicts carrying no element ids all change nothing.

    `graph_execution_plan` is the independently reconstructed, receipt-bound
    frozen plan for a typed homogeneous-graph contract. Graph rows can certify
    only when their full trace and per-probe observations match this authority.
    It is ignored for graph-free and archived contracts.
    """
    homogeneous_graph_required = _has_homogeneous_graph_contract(spec)
    probe_reasons, disclosures = _probe_reasons(probe_report, spec)
    demo_reasons, demo_disclosures, demo_record = _demo_verdict_entries(
        demo_verdict, probe_report, spec, graph_execution_plan)
    required_structured_reason = _required_structured_demo_reason(
        demo_verdict,
        structured_demo_required=structured_demo_required,
        homogeneous_graph_required=homogeneous_graph_required,
    )
    if required_structured_reason is not None:
        demo_reasons.append(required_structured_reason)
        # Never project a malformed schema-2 payload into the strict final
        # manifest. Graph-bearing deliveries replace it with an explicitly
        # unresolved record whose graph axes still come from independent probe
        # authority. Graph-free compatibility paths omit the invalid record.
        if required_structured_reason["id"] in {
            "structured_demo_evidence_missing",
            "structured_demo_evidence_malformed",
        }:
            demo_record = (
                _missing_graph_demo_record(
                    probe_report,
                    spec,
                    graph_execution_plan,
                    required_structured_reason,
                )
                if homogeneous_graph_required else None
            )
    demoters = probe_reasons + _review_reasons(review_report) + demo_reasons
    disclosures = disclosures + demo_disclosures
    stubs = _normalize_stubs(stubbed_elements)
    approximations = _normalize_core_approximations(core_approximations)
    completeness_reasons = ([_stub_reason(rec) for rec in stubs]
                            + [_approximation_reason(rec)
                               for rec in approximations])
    counts: dict[str, int] = {}
    for v in (probe_report or {}).get("verdicts") or []:
        verdict = str(v.get("verdict", "?"))
        counts[verdict] = counts.get(verdict, 0) + 1
    reasons = list(demoters)
    if demoters:
        label = LABEL_DRAFT
        reasons = completeness_reasons + demoters
    elif completeness_reasons:
        # Nothing partial or core-approximated can read verified.
        if _has_contribution_evidence(
            probe_report, spec, graph_execution_plan
        ):
            label = LABEL_DRAFT
            reasons = completeness_reasons
        else:
            # Evidence axis unchanged: no contribution evidence stays
            # uncertified; the causes are disclosed, never silently
            # dropped (reasons stay empty — non-empty reasons means draft).
            label = LABEL_UNCERTIFIED
            disclosures = completeness_reasons + disclosures
    elif _has_contribution_evidence(
        probe_report, spec, graph_execution_plan
    ):
        label = LABEL_VERIFIED
    else:
        label = LABEL_UNCERTIFIED
    out = {
        "schema_version": SCHEMA_VERSION,
        "graph_evidence_applicability": (
            GRAPH_EVIDENCE_HOMOGENEOUS_V1
            if homogeneous_graph_required
            else GRAPH_EVIDENCE_NOT_APPLICABLE
        ),
        "partial": bool(stubs),
        "label": label,
        "reasons": reasons,
        "disclosures": disclosures,
        "probe_counts": counts,
        "stubbed_elements": stubs,
    }
    if demo_record is not None:
        out["demo_verdict"] = demo_record
    if label == LABEL_UNCERTIFIED:
        out["missing_probe_family"] = str(paradigm) if paradigm else "unknown"
    unprobed_core = unprobeable_core_novelty(probe_report, spec)
    if unprobed_core:
        apply_unprobeable_core_novelty_cap(out, unprobed_core)
        if out["label"] == LABEL_UNCERTIFIED and not out.get("missing_probe_family"):
            out["missing_probe_family"] = str(paradigm) if paradigm else "unknown"
    if neutral_build_plan:
        apply_neutral_plan_cap(out, paradigm)
    return out
