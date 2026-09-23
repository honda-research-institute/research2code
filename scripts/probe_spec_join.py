"""Does the run's own contract already adjudicate this probe finding? (R2C-047)

A behavioral probe observes what the delivered code DOES. The methodology
replication contract, written by the analyzer from the paper, states what it
MUST do. Until these two joined, a producer-fixable finding and a
paper-intrinsic finding got identical treatment in routing, in wording, and in
labeling: the active-learning selector's model-insensitivity arm asked the
researcher whether "the paper's method is purely geometric" on a run whose own
spec declared the scoring must-replicate with a forbidden-substitution list.

## The join

A deterministic set intersection on element identifiers, with NO fuzzy
fallback (maintainer decision 2026-08-05). A probe finding that carries no element ids is a
probe-side gap to fix, never a matcher to soften: `bound` is False and every
consumer falls back to its neutral behavior.

## Why the branch rule needs no prose matching

`schemas/method_spec.py` enforces a bi-implication on every contract element:
`acceptable_approximations` is non-empty if and only if `replication_status`
is `faithful_approximation_allowed`. So a `must_replicate` element has, by
construction, no approved demo-scale approximation to hide behind, and a
behavioral divergence on one is a defect on our side. The three branches fall
straight out of the status field:

- `contract_contradicts` — a matched element is must_replicate. The contract
  and the observation cannot both be right, so this is ours to fix.
- `contract_consistent` — matched elements exist and none is must_replicate:
  the paper's mechanism is either approximable (approximations approved) or
  not replicable at demo scale, so the finding describes a known limit.
- `contract_silent` — nothing matched. The recorded neutrality decision
  applies unchanged: the probe cannot tell our miswiring from the paper's
  design, so the researcher adjudicates.

## Core novelty

`core_novelty` is the conjunction the label cap keys on: role
`core_methodology` AND status `must_replicate`. An `unprobeable` verdict there
means the delivery has no evidence about the paper's central claim, which is
the coverage loss the maintainer ruled must stop being label-neutral (2026-08-03).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

# The graph-mechanism wiring derivation (R2C-092) lives in the
# dependency-free schemas/graph_wiring.py; the bare fallback keeps the
# vendored portable harness working, where this file and graph_wiring.py
# sit side by side with no schemas package (R2C-019).
try:
    from schemas.graph_wiring import (  # noqa: E402
        derived_graph_probe_refs,
        merge_derived_graph_refs,
    )
except ImportError:  # pragma: no cover - exercised only in the vendored tree
    from graph_wiring import (  # noqa: E402
        derived_graph_probe_refs,
        merge_derived_graph_refs,
    )

# Branch names. Public because the disclosure templates and the claims-table
# label alignment both key on them, and a typo would silently pick a branch.
CONTRACT_CONTRADICTS = "contract_contradicts"
CONTRACT_CONSISTENT = "contract_consistent"
CONTRACT_SILENT = "contract_silent"

MUST_REPLICATE = "must_replicate"
CORE_METHODOLOGY = "core_methodology"
REFERENCE_CONTRACT_SCHEMA = (1, 12, 0)


def _version_tuple(value: object) -> tuple[int, ...]:
    parts = str(value or "").split(".")
    if not parts or any(not part.isdigit() for part in parts):
        return ()
    return tuple(int(part) for part in parts)


def reference_contract_enabled(spec: dict | None) -> bool:
    """Whether exact probe-ref qualification governs this raw contract.

    Schema 1.12 introduced the producer carrier, so a fresh spec cannot omit
    it and fall back to prefix-only certification. Older specs opt in only by
    explicitly carrying the field; otherwise current batteries preserve their
    historical id-only adjudication without gaining certification authority.
    """
    if not isinstance(spec, dict):
        return False
    version = _version_tuple(spec.get("schema_version"))
    if version and version >= REFERENCE_CONTRACT_SCHEMA:
        return True
    contract = spec.get("methodology_replication_contract")
    if not isinstance(contract, dict):
        return False
    return any(
        isinstance(element, dict) and "verification_probe_refs" in element
        for element in (contract.get("elements") or [])
    )


@dataclass(frozen=True)
class JoinedElement:
    """One methodology contract element a probe finding binds to."""

    element_id: str
    role: str
    replication_status: str
    technical_concept: str
    required_behavior: str
    paper_section: str
    acceptable_approximations: tuple[str, ...] = ()
    forbidden_substitutions: tuple[str, ...] = ()
    # The paper_map elements this contract element was built from (R2C-072).
    # Empty on a spec written before the crosswalk existed, which costs the
    # element its adjudication and nothing else.
    paper_element_ids: tuple[str, ...] = ()
    # Exact taxonomy executor refs this obligation permits as verification
    # evidence (R2C-087). Optional for archived specs; an empty tuple means a
    # new reference-qualified verdict cannot certify this element.
    verification_probe_refs: tuple[str, ...] = ()
    # Presence is distinct from an empty declared list. Archived contracts
    # omitted the carrier entirely and retain id-only adjudication when a new
    # battery is replayed over them; migrated contracts may deliberately emit
    # [] and thereby refuse reference-qualified binding for this obligation.
    verification_probe_refs_declared: bool = False

    @property
    def must_replicate(self) -> bool:
        return self.replication_status == MUST_REPLICATE

    @property
    def core_novelty(self) -> bool:
        """The paper's central claim, required exactly. What the label cap
        protects: a core element the paper leans on AND that cannot be
        approximated away."""
        return self.role == CORE_METHODOLOGY and self.must_replicate

    def in_the_papers_words(self) -> str:
        """The mechanism named as the paper names it, for a disclosure that
        quotes the contract instead of paraphrasing it."""
        return (f"{self.technical_concept} ({self.paper_section}): "
                f"{self.required_behavior}")


@dataclass(frozen=True)
class SpecJoin:
    """The result of joining one probe verdict to the contract."""

    probe_id: str
    verdict: str
    probe_ref: str = ""
    declared_callables: tuple[str, ...] = ()
    # Elements reached by the verdict's explicit ids and callable-resolved
    # paper anchors, before the exact probe-ref intersection. Exposed so a
    # known ref with no intersection is distinguishable from no anchor at all.
    candidates: tuple[JoinedElement, ...] = ()
    matched: tuple[JoinedElement, ...] = ()
    unmatched_ids: tuple[str, ...] = ()
    # Whether the run HAS a methodology contract at all. An unmatched id means
    # two different things either side of this: with a contract present it is a
    # naming gap worth reporting, and with no contract it just means this
    # paradigm's analyzer path emits none, which is nobody's defect.
    contract_present: bool = False
    known_probe_refs: tuple[str, ...] = ()
    reference_contract: bool = False

    @property
    def bound(self) -> bool:
        return bool(self.matched)

    @property
    def probe_ref_known(self) -> bool:
        return bool(self.probe_ref) and self.probe_ref in self.known_probe_refs

    @property
    def reference_qualified(self) -> bool:
        """True only when an exact nonblank ref selected the final bindings.

        Archived verdicts with no ref retain deterministic id-only joins for
        disclosure/routing compatibility, but callers can use this property to
        prevent those legacy reads from certifying new family evidence.
        """
        return (
            self.reference_contract
            and self.probe_ref_known
            and bool(self.matched)
        )

    @property
    def branch(self) -> str:
        if not self.matched:
            return CONTRACT_SILENT
        if any(element.must_replicate for element in self.matched):
            return CONTRACT_CONTRADICTS
        return CONTRACT_CONSISTENT

    @property
    def must_replicate_elements(self) -> tuple[JoinedElement, ...]:
        return tuple(e for e in self.matched if e.must_replicate)

    @property
    def core_novelty_elements(self) -> tuple[JoinedElement, ...]:
        return tuple(e for e in self.matched if e.core_novelty)

    @property
    def producer_fixable(self) -> bool:
        """A failing or flagged observation the contract contradicts.

        The tier upgrade's precondition. A passing verdict is nothing to fix,
        and an unprobeable one is missing evidence rather than an observed
        defect, so neither qualifies however the contract reads."""
        return (self.branch == CONTRACT_CONTRADICTS
                and self.verdict in ("fail", "flag_for_researcher"))

    @property
    def caps_the_label(self) -> bool:
        """An unprobeable verdict on the paper's core novelty.

        The delivery derivation reads this as a CAP (the label cannot exceed
        the disclosed tier), never a point deduction (maintainer decision 2026-08-05)."""
        return self.verdict == "unprobeable" and bool(self.core_novelty_elements)


def _stable_nonblank(values: object) -> tuple[str, ...]:
    """Normalized nonblank strings, deduplicated in declaration order.

    MethodSpec strips probe refs during validation but the validator does not
    rewrite the stored JSON. Normalize the raw read identically so a
    strict-green ref cannot become unbound merely because the producer left
    surrounding whitespace in the serialized artifact.
    """
    if not isinstance(values, (list, tuple)):
        return ()
    out: list[str] = []
    seen: set[str] = set()
    for raw in values:
        value = str(raw or "").strip()
        if not value or value in seen:
            continue
        seen.add(value)
        out.append(value)
    return tuple(out)


def contract_elements(spec: dict | None) -> dict[str, tuple[JoinedElement, ...]]:
    """Every methodology element, indexed by an order-preserving id multimap.

    TWO id spaces reach this function, and they are genuinely different sets
    (R2C-072). The contract names its own elements (`gnn-encoder-mean-pooling`).
    The paper map names the paper's parts (`concept-gnn-encoder`), and those are
    the ids generated code is annotated with, so those are the ids a behavioral
    probe can actually produce. Five delivered runs showed zero overlap between
    the two, which is why the join bound nothing before the crosswalk existed.

    So each element is indexed under its own id AND under every id in its
    `paper_element_ids` crosswalk. Paper ids are one-to-many: every grounded
    obligation survives in contract order. Direct methodology ids take
    precedence over a colliding paper id and resolve to their one exact
    element. A verdict may carry either kind, and naming the same obligation
    through both is deduplicated later by methodology `element_id`.

    Tolerant of a spec that has no contract (an older run, a paradigm whose
    analyzer path does not emit one): the result is empty and every join is
    unbound, which is the neutral branch.
    """
    if not isinstance(spec, dict):
        return {}
    contract = spec.get("methodology_replication_contract")
    if not isinstance(contract, dict):
        return {}
    # R2C-092: under a declared graph-mechanism block, graph_mechanism.*
    # ownership is derived from the block, not transcribed by the producer.
    # Merge the same derived wiring the schema normalizes in, so a verdict's
    # reference qualification cannot depend on hand-copied refs the stored
    # JSON may lack. Specs without the block keep their declared refs as-is.
    mechanism = contract.get("homogeneous_graph_mechanism")
    derived_graph_refs = (
        derived_graph_probe_refs(mechanism)
        if isinstance(mechanism, dict) else {}
    )
    ordered: list[JoinedElement] = []
    for raw in contract.get("elements") or []:
        if not isinstance(raw, dict):
            continue
        element_id = str(raw.get("element_id") or "")
        if not element_id:
            continue
        element = JoinedElement(
            element_id=element_id,
            role=str(raw.get("role") or ""),
            replication_status=str(raw.get("replication_status") or ""),
            technical_concept=str(raw.get("technical_concept") or ""),
            required_behavior=str(raw.get("required_behavior") or ""),
            paper_section=str(raw.get("paper_section") or ""),
            acceptable_approximations=tuple(
                str(a) for a in (raw.get("acceptable_approximations") or [])),
            forbidden_substitutions=tuple(
                str(f) for f in (raw.get("forbidden_substitutions") or [])),
            paper_element_ids=tuple(
                _stable_nonblank(raw.get("paper_element_ids")),
            ),
            verification_probe_refs=tuple(
                merge_derived_graph_refs(
                    _stable_nonblank(raw.get("verification_probe_refs")),
                    derived_graph_refs.get(element_id, ()),
                )
                if derived_graph_refs
                else _stable_nonblank(raw.get("verification_probe_refs")),
            ),
            verification_probe_refs_declared=(
                "verification_probe_refs" in raw
            ),
        )
        ordered.append(element)

    # Build direct ids first so collision precedence is independent of list
    # position and paper-id insertion order.
    out: dict[str, tuple[JoinedElement, ...]] = {
        element.element_id: (element,) for element in ordered
    }
    crosswalk: dict[str, list[JoinedElement]] = {}
    for element in ordered:
        for paper_id in element.paper_element_ids:
            if paper_id in out:
                continue  # direct methodology identity wins every collision
            bucket = crosswalk.setdefault(paper_id, [])
            if all(existing.element_id != element.element_id for existing in bucket):
                bucket.append(element)
    out.update({key: tuple(value) for key, value in crosswalk.items()})
    return out


def join_verdict(verdict: Any, spec: dict | None) -> SpecJoin:
    """Join one probe verdict by anchor candidates, then exact probe ref.

    `verdict` is a ProbeVerdict or the dict form a probe report carries, so
    the same primitive serves the live battery and a re-read of a delivered
    probe report. The battery resolves ``bound_callables`` through generated
    ``# paper-element:`` anchors into additive ``element_ids`` before this
    call; explicit ids and callable-derived ids therefore share one candidate
    vocabulary without guessing callable names here.
    """
    if isinstance(verdict, dict):
        probe_id = str(verdict.get("probe_id") or "")
        outcome = str(verdict.get("verdict") or "")
        ids = verdict.get("element_ids") or []
        probe_ref = str(verdict.get("probe_ref") or "").strip()
        declared_callables = _stable_nonblank(verdict.get("bound_callables"))
    else:
        probe_id = str(getattr(verdict, "probe_id", "") or "")
        outcome = str(getattr(verdict, "verdict", "") or "")
        ids = getattr(verdict, "element_ids", None) or []
        probe_ref = str(getattr(verdict, "probe_ref", "") or "").strip()
        declared_callables = _stable_nonblank(
            getattr(verdict, "bound_callables", None)
        )
    return _join_ids(
        probe_id,
        outcome,
        ids,
        spec,
        probe_ref=probe_ref,
        declared_callables=declared_callables,
    )


def _join_ids(
    probe_id: str, outcome: str, ids: Iterable[Any], spec: dict | None,
    *, probe_ref: str = "", declared_callables: Iterable[str] = (),
) -> SpecJoin:
    elements = contract_elements(spec)
    unmatched: list[str] = []
    seen_keys: set[str] = set()
    candidate_ids: set[str] = set()
    for raw in ids:
        element_id = str(raw or "")
        if not element_id or element_id in seen_keys:
            continue
        seen_keys.add(element_id)
        candidates = elements.get(element_id, ())
        if not candidates:
            # An id the contract does not carry. NOT softened into a prose
            # match: it is either a paper_map id outside the contract or a
            # probe-side naming gap, and both are visible here rather than
            # guessed at.
            unmatched.append(element_id)
            continue
        candidate_ids.update(element.element_id for element in candidates)

    # Direct entries were inserted in contract order before crosswalk keys.
    # Flattening the multimap this way recovers that order, independent of the
    # verdict's id order, while deduplicating direct/crosswalk aliases.
    ordered_contract: list[JoinedElement] = []
    seen_elements: set[str] = set()
    for bucket in elements.values():
        for element in bucket:
            if element.element_id in seen_elements:
                continue
            seen_elements.add(element.element_id)
            ordered_contract.append(element)
    candidates = tuple(
        element for element in ordered_contract
        if element.element_id in candidate_ids
    )
    known_probe_refs = tuple(dict.fromkeys(
        ref
        for element in ordered_contract
        for ref in element.verification_probe_refs
    ))
    reference_contract = reference_contract_enabled(spec)
    # Archived contracts retain every candidate deterministically. Current
    # reference-aware contracts require an exact nonblank ref before anything
    # is matched: candidates remain visible for diagnosis, but a missing ref
    # must not recreate the old bind-all behavior on a shared paper anchor.
    matched = (
        candidates
        if not reference_contract
        else tuple(
            element for element in candidates
            if probe_ref in element.verification_probe_refs
        )
    )
    return SpecJoin(
        probe_id=probe_id,
        verdict=outcome,
        probe_ref=probe_ref,
        declared_callables=tuple(declared_callables),
        candidates=candidates,
        matched=matched,
        unmatched_ids=tuple(unmatched),
        contract_present=bool(elements),
        known_probe_refs=known_probe_refs,
        reference_contract=reference_contract,
    )


# ---------------------------------------------------------------------------
# The three-way disclosure. One sentence appended to the probe's own message,
# naming the mechanism in the paper's own words and stating which branch
# applies, so the claims-table label and its note cannot contradict each other.


def disclosure(join: SpecJoin, *, neutral: str = "") -> str:
    """The spec-conditioned sentence for this finding, or `neutral` when the
    contract is silent.

    Deliberately additive: the probe still authors its own observation, and
    this states what the run's contract says about it. A probe message that
    asked a question the contract answers now carries the answer beside it.
    """
    if join.branch == CONTRACT_SILENT:
        return neutral
    if join.branch == CONTRACT_CONTRADICTS:
        elements = join.must_replicate_elements
        named = "; ".join(e.in_the_papers_words() for e in elements)
        plural = "elements" if len(elements) > 1 else "element"
        return (
            f"This run's own methodology contract CONTRADICTS the "
            f"observation: it marks {len(elements)} {plural} must-replicate "
            f"with no approved approximation — {named}. So this is a defect "
            f"on our side to fix, not a question about the paper's design."
        )
    named = "; ".join(e.in_the_papers_words() for e in join.matched)
    statuses = ", ".join(sorted({e.replication_status for e in join.matched}))
    return (
        f"This run's own methodology contract is CONSISTENT with the "
        f"observation: the bound element(s) carry status {statuses}, so the "
        f"behavior is within what the contract already approves for demo "
        f"scale — {named}. No fix is implied; the disclosure exists so the "
        f"number is read with the approximation in view."
    )


def unbound_gap_note(join: SpecJoin) -> str | None:
    """Why a finding could not be adjudicated, when it carried ids that went
    nowhere. Surfaces the probe-side gap instead of letting it read as a
    contract that happens to be silent.

    Silent when the run has no contract at all: there the ids went nowhere
    because this paradigm's analyzer path emits no contract, which is not a
    naming gap and must not be reported as one.
    """
    if join.bound or not join.contract_present:
        return None
    if join.reference_contract and join.candidates and not join.probe_ref:
        named = ", ".join(f"`{e.element_id}`" for e in join.candidates)
        return (
            f"This finding reached candidate obligation(s) {named}, but it "
            "omits the exact probe ref required by this reference-aware "
            "methodology contract. The contract therefore stays unbound "
            "rather than guessing from a shared callable or paper anchor."
        )
    if join.probe_ref and not join.probe_ref_known:
        return (
            f"This finding carries probe ref `{join.probe_ref}`, but the "
            "methodology contract declares no such verification ref. The "
            "finding remains unbound; this is a probe-side or analyzer-side "
            "coverage gap, not evidence about the paper."
        )
    if join.candidates and join.probe_ref:
        named = ", ".join(f"`{e.element_id}`" for e in join.candidates)
        return (
            f"This finding reached candidate obligation(s) {named}, but none "
            f"declares its exact probe ref `{join.probe_ref}`. The contract "
            "therefore stays unbound rather than guessing from a shared "
            "callable or paper anchor."
        )
    if (
        join.reference_contract
        and join.probe_ref
        and not join.candidates
        and not join.unmatched_ids
    ):
        if not join.declared_callables:
            return (
                f"This finding carries known probe ref `{join.probe_ref}`, "
                "but declares no callable or element anchor that could reach "
                "a methodology obligation. The result is unanchored coverage, "
                "not certification evidence."
            )
        named = ", ".join(f"`{name}`" for name in join.declared_callables)
        return (
            f"This finding carries known probe ref `{join.probe_ref}` and "
            f"declares callable(s) {named}, but those interfaces resolved to "
            "no paper-element anchor and therefore no methodology obligation. "
            "The result is unanchored coverage, not certification evidence."
        )
    if not join.unmatched_ids:
        return None
    named = ", ".join(f"`{i}`" for i in join.unmatched_ids)
    return (
        f"This finding names element(s) the methodology contract does not "
        f"carry ({named}), so the contract could not adjudicate it. That is a "
        f"probe-side or analyzer-side naming gap to close, not a judgment "
        f"that the paper is silent."
    )
