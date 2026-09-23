#!/usr/bin/env python3
"""Sliced review bundles and acceptance-record loading for the insight
semantic reviewer.

Design: the insight semantic reviewer design note (internal, not shipped)
(approved 2026-07-16). Same standalone discipline as generic_insights.py:
never imports run_pipeline, never discovers a run, never contacts a model
or the network. The driver hook is a later, separately reviewed change.

Two halves:

- **Bundle building.** The dispatcher hands the reviewer per-record
  bundles — the record's statement or structured normative fields, its
  citations with exact quotes plus a bounded context window, and the
  coverage entries the record touches — never the whole paper. Input
  hygiene by construction (the cap-burn lesson).
- **Record loading.** Semantic status moves off `unreviewed` only through
  a hash-matching acceptance record. Missing, unparseable, schema-invalid,
  wrong-major-version, hash-stale, or id-incomplete records are all
  ABSENT, and absence is never demotion and never promotion. A
  `needs_human` overall verdict also leaves the status unreviewed.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:  # pragma: no cover - import plumbing
    sys.path.insert(0, str(_REPO_ROOT))

from pydantic import ValidationError  # noqa: E402

from schemas.generic_insights import GenericInsights  # noqa: E402
from schemas.semantic_review import (  # noqa: E402
    SCHEMA_MAJOR,
    SemanticReviewRecord,
)

from generic_insights import build_region_inventory  # noqa: E402

# The acceptance record's canonical run-relative path, beside the three
# internal shadow artifacts.
RECORD_RELPATH = ".pipeline/generic_insights.semantic_review.json"

# Bytes of source context included on each side of a cited span. Bounded
# and clamped to the citation's region body so a bundle can never smuggle
# in the whole document.
CONTEXT_WINDOW_BYTES = 400


def _decode(data: bytes) -> str:
    return data.decode("utf-8", errors="replace")


def _citation_context(
    source: bytes, start: int, end: int, clamp: tuple[int, int]
) -> tuple[str, str]:
    """Bounded context before and after a cited span, clamped to the
    enclosing region body (or the whole document when no regions exist)."""
    lo = max(clamp[0], start - CONTEXT_WINDOW_BYTES)
    hi = min(clamp[1], end + CONTEXT_WINDOW_BYTES)
    return _decode(source[lo:start]), _decode(source[end:hi])


def build_review_bundles(source: bytes, model: GenericInsights) -> dict:
    """Per-record review bundles for a validated candidate.

    The caller passes the exact source bytes the candidate was validated
    against and the parsed model; this function does no I/O. Returns
    {document, declared_gaps, records:[...]} where each record bundle
    carries the fields the reviewer needs to answer entailment — and
    nothing else."""
    regions = {r.region_id: r for r in build_region_inventory(source)}
    coverage_by_region = {h.region_id: h for h in model.coverage.headings}
    citation_by_id = {c.id: c for c in model.citations}

    def _citation_bundle(cid: str) -> dict:
        cit = citation_by_id[cid]
        region = regions.get(cit.region_id)
        clamp = region.body_span if region else (0, len(source))
        before, after = _citation_context(
            source, cit.span.start, cit.span.end, clamp)
        decl = coverage_by_region.get(cit.region_id)
        return {
            "id": cit.id,
            "region_id": cit.region_id,
            "heading": region.text if region else None,
            "quote": cit.quote,
            "context_before": before,
            "context_after": after,
            "coverage_disposition":
                decl.disposition.value if decl else None,
        }

    records = []
    for rec in model.insights:
        records.append({
            "record_id": rec.id,
            "kind": rec.kind.value,
            "statement": rec.statement,
            "normative": rec.normative.model_dump(mode="json")
            if rec.normative else None,
            "steps": list(rec.steps),
            "facets": [f.value for f in rec.facets],
            "conceptual_only": rec.conceptual_only,
            "confidence": rec.confidence.value,
            "qualifies_insight_id": rec.qualifies_insight_id,
            "citations": [_citation_bundle(cid) for cid in rec.citation_ids],
        })
    return {
        "document": {
            "kind": model.document.kind.value,
            "contribution_kinds": list(model.document.contribution_kinds),
        },
        "declared_gaps": {
            "missing": list(model.coverage.missing),
            "unresolved_dependencies":
                list(model.coverage.unresolved_dependencies),
            "vocabulary_none_identified_reason":
                model.vocabulary_none_identified_reason,
        },
        "records": records,
    }


def build_case_bundle(source: bytes, case: dict) -> dict:
    """A bundle for one golden-corpus case (statement + citations). Same
    shape as build_review_bundles so the reviewer prompt is identical in
    calibration and production; the corpus's minimal cases have no
    document metadata, coverage entries, or structured fields."""
    records = [{
        "record_id": case["case_id"],
        "kind": None,
        "statement": case["statement"],
        "normative": None,
        "steps": [],
        "facets": [],
        "conceptual_only": None,
        "confidence": None,
        "qualifies_insight_id": None,
        "citations": [
            {
                "id": c["id"],
                "region_id": None,
                "heading": None,
                "quote": c["quote"],
                "context_before": _citation_context(
                    source, c["span"]["start"], c["span"]["end"],
                    (0, len(source)))[0],
                "context_after": _citation_context(
                    source, c["span"]["start"], c["span"]["end"],
                    (0, len(source)))[1],
                "coverage_disposition": None,
            }
            for c in case["citations"]
        ],
    }]
    return {"document": None, "declared_gaps": None, "records": records}


# ---------------------------------------------------------------------------
# Acceptance-record loading — stale reads as absent
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RecordResolution:
    """Outcome of resolving an acceptance record against the live hash
    pair. `record` is non-None only for a present, valid, hash-matching,
    id-complete record; `reason` says why it is absent otherwise (for
    shadow diagnostics — absence itself is never an error)."""

    record: SemanticReviewRecord | None
    reason: str

    @property
    def present(self) -> bool:
        return self.record is not None


def _absent(reason: str) -> RecordResolution:
    return RecordResolution(record=None, reason=reason)


def load_acceptance_record(
    record_path: Path,
    *,
    source_sha256: str,
    candidate_sha256: str,
    expected_record_ids: set[str] | None = None,
) -> RecordResolution:
    """Load the acceptance record valid for exactly this source/candidate
    hash pair. Every failure mode reads as absent:

    - no file, unreadable, or not JSON;
    - schema-invalid or unknown schema MAJOR;
    - hash mismatch on either side (a stale record from before a
      regeneration must never certify the new bytes);
    - when `expected_record_ids` is given, any missing or extra record id
      across verdicts + needs_human (a partial review certifies nothing).
    """
    path = Path(record_path)
    try:
        raw = path.read_bytes()
    except OSError as exc:
        return _absent(f"no readable record at {path.name}: {exc}")
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        return _absent(f"record is not valid JSON: {exc}")
    try:
        record = SemanticReviewRecord.model_validate(payload)
    except ValidationError as exc:
        summary = "; ".join(
            f"{'.'.join(str(p) for p in err['loc'])}: {err['msg']}"
            for err in exc.errors())
        return _absent(f"record failed schema validation: {summary}")

    declared_major = record.schema_version.split(".", 1)[0]
    if declared_major != SCHEMA_MAJOR:
        return _absent(
            f"record schema_version {record.schema_version!r} has unknown "
            f"major version (this loader understands major {SCHEMA_MAJOR})")
    if record.source_sha256 != source_sha256:
        return _absent(
            "stale record: source_sha256 does not match the live source")
    if record.candidate_sha256 != candidate_sha256:
        return _absent(
            "stale record: candidate_sha256 does not match the live candidate")
    if expected_record_ids is not None:
        covered = {v.record_id for v in record.verdicts} | \
                  {e.record_id for e in record.needs_human}
        if covered != set(expected_record_ids):
            missing = sorted(set(expected_record_ids) - covered)
            extra = sorted(covered - set(expected_record_ids))
            return _absent(
                "incomplete record: verdicts + needs_human must cover the "
                f"candidate's record ids exactly (missing {missing}, "
                f"extra {extra})")
    return RecordResolution(record=record, reason="present")


def resolve_semantic_status(resolution: RecordResolution) -> str:
    """The semantic status implied by a record resolution. Absent records
    and needs_human verdicts leave it `unreviewed`; only a decisive
    hash-matching record moves it."""
    if resolution.record is None:
        return "unreviewed"
    if resolution.record.overall_verdict == "needs_human":
        return "unreviewed"
    return resolution.record.overall_verdict


