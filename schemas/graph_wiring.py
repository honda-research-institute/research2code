"""Derived wiring for the homogeneous-graph mechanism block (R2C-092).

The mechanism block's element ids fully determine which methodology
elements carry the ``relational_structure.kind='homogeneous_graph'``
marker and which exact ``graph_mechanism.*`` probe refs each element
owns.  Grading a producer on hand-transcribing those cross-references
burned five live pdfgnn attempts (2026-08-10/11): every fix risked
relocating a marker or dropping a ref, so error counts moved but never
converged.  The failure class is ``derived_fact_transcription_burn``.

This module is the single derivation surface.  The pydantic schema, the
strict spec validator, the probe-spec join, the graph-mechanism runtime
plan, and the architecture-contract validator all consume the same pure
functions.  The stored method_spec.json is never rewritten: each read
surface normalizes identically (the probe_spec_join convention).

Producer-owned semantics stay validated, never derived: which elements
the block binds, callable identities, ablation choice and its
discriminator, and paper grounding.  Only the mechanical wiring (marker
placement and graph-ref ownership) is derived here.

Dependency-free on purpose (stdlib only): raw-dict consumers, including
vendored portable harness code, must be able to import it without
pydantic.  Helpers accept both raw mappings and typed pydantic models,
reading keys or attributes uniformly.
"""

from collections.abc import Mapping
from typing import Any


ALIGNMENT_PROBE_REF = "graph_mechanism.alignment_prerequisite"
PARAMETER_AGREEMENT_PROBE_REF = "graph_mechanism.parameter_agreement"
CONSTRUCTION_PROBE_REF = "graph_mechanism.construction_semantics"
TOPOLOGY_PROBE_REF = "graph_mechanism.topology_sensitivity"
NEIGHBOR_SIGNAL_PROBE_REF = "graph_mechanism.neighbor_sensitivity"
PERMUTATION_PROBE_REF = "graph_mechanism.permutation_equivalence"
CONTRIBUTION_ABLATION_PROBE_REF = "graph_mechanism.contribution_ablation"

GRAPH_PROBE_REF_VOCABULARY = frozenset(
    {
        ALIGNMENT_PROBE_REF,
        PARAMETER_AGREEMENT_PROBE_REF,
        CONSTRUCTION_PROBE_REF,
        TOPOLOGY_PROBE_REF,
        NEIGHBOR_SIGNAL_PROBE_REF,
        PERMUTATION_PROBE_REF,
        CONTRIBUTION_ABLATION_PROBE_REF,
    }
)


def _read(obj: Any, name: str) -> Any:
    if isinstance(obj, Mapping):
        return obj.get(name)
    return getattr(obj, name, None)


def _exact_element_id(value: Any) -> str | None:
    if isinstance(value, str) and value and value == value.strip():
        return value
    return None


def _mechanism_role_ids(
    mechanism: Any,
) -> tuple[str | None, str | None, str | None, str | None]:
    """(alignment, construction, message_passing, ablation) element ids.

    Tolerant of malformed raw input (a missing or misshapen sub-block reads
    as None); the pydantic schema separately rejects malformed fresh specs.
    """
    if mechanism is None:
        return None, None, None, None
    alignment = _exact_element_id(_read(mechanism, "alignment_element_id"))
    construction_block = _read(mechanism, "construction")
    message_block = _read(mechanism, "message_passing")
    ablation_block = _read(mechanism, "contribution_ablation")
    construction = (
        _exact_element_id(_read(construction_block, "element_id"))
        if construction_block is not None
        else None
    )
    message = (
        _exact_element_id(_read(message_block, "element_id"))
        if message_block is not None
        else None
    )
    ablation = (
        _exact_element_id(_read(ablation_block, "element_id"))
        if ablation_block is not None
        else None
    )
    return alignment, construction, message, ablation


def mechanism_marker_element_ids(mechanism: Any) -> list[str]:
    """Element ids the block requires to carry ``kind='homogeneous_graph'``.

    Alignment, construction, and message passing participate in the graph.
    The contribution-ablation control never carries the marker: it may be a
    non-graph decoder.  Role order, deduplicated for shared elements.
    """
    alignment, construction, message, _ = _mechanism_role_ids(mechanism)
    out: list[str] = []
    for element_id in (alignment, construction, message):
        if element_id is not None and element_id not in out:
            out.append(element_id)
    return out


def derived_graph_probe_refs(mechanism: Any) -> dict[str, list[str]]:
    """element_id -> exact graph refs it owns, fully determined by the block.

    The permutation ref exists only when the block declares an equivariant
    permutation probe; the ablation ref only when a contribution ablation
    and its probe ref are both declared (the schema enforces both-or-neither
    for fresh specs).  An element serving several roles owns the union.
    """
    alignment, construction, message, ablation = _mechanism_role_ids(mechanism)
    if alignment is None or construction is None or message is None:
        return {}
    refs = _read(mechanism, "probe_refs")
    out: dict[str, list[str]] = {}

    def _own(element_id: str, ref: str) -> None:
        bucket = out.setdefault(element_id, [])
        if ref not in bucket:
            bucket.append(ref)

    _own(alignment, ALIGNMENT_PROBE_REF)
    _own(construction, PARAMETER_AGREEMENT_PROBE_REF)
    _own(construction, CONSTRUCTION_PROBE_REF)
    _own(message, TOPOLOGY_PROBE_REF)
    _own(message, NEIGHBOR_SIGNAL_PROBE_REF)
    if refs is not None and _read(refs, "permutation") is not None:
        _own(message, PERMUTATION_PROBE_REF)
    if (
        ablation is not None
        and refs is not None
        and _read(refs, "contribution_ablation") is not None
    ):
        _own(ablation, CONTRIBUTION_ABLATION_PROBE_REF)
    return out


def merge_derived_graph_refs(
    declared: Any, derived: Any
) -> list[str]:
    """Declared refs outside the graph vocabulary, then the derived refs.

    Under a declared mechanism block the graph vocabulary is pipeline-owned:
    a hand-transcribed graph ref is neither trusted (it may sit on the wrong
    element) nor required (the owner gets it derived), so the merge drops
    every declared graph-vocabulary ref and appends the derived ones.  Refs
    outside the vocabulary remain producer-owned and pass through in
    declaration order.  Only call this when the spec declares a mechanism;
    archived specs without the block keep their declared refs untouched.
    """
    out: list[str] = []
    for ref in declared or ():
        value = str(ref or "").strip()
        if (
            value
            and value not in GRAPH_PROBE_REF_VOCABULARY
            and value not in out
        ):
            out.append(value)
    for ref in derived or ():
        if ref not in out:
            out.append(ref)
    return out


def homogeneous_graph_element_ids(
    elements: Any, mechanism: Any
) -> list[str]:
    """Declared homogeneous-marker ids plus mechanism-bound ids.

    Declared markers keep element order (archived graph specs have markers
    and no block); mechanism-bound ids the producer did not mark are
    appended in role order, so fresh specs resolve to the derived set even
    when the stored JSON omits the markers.
    """
    out: list[str] = []
    for element in elements or ():
        marker = _read(element, "relational_structure")
        if marker is None:
            continue
        kind = _read(marker, "kind")
        kind_value = getattr(kind, "value", kind)
        if kind_value != "homogeneous_graph":
            continue
        element_id = _exact_element_id(_read(element, "element_id"))
        if element_id is not None and element_id not in out:
            out.append(element_id)
    for element_id in mechanism_marker_element_ids(mechanism):
        if element_id not in out:
            out.append(element_id)
    return out
