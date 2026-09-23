#!/usr/bin/env python3
"""Deterministic validator and renderer for the generic insight artifact.

Block 7 pure tranche (contract:
the generic insight artifact contract note (internal, not shipped), approved
2026-07-16). This module is standalone by design: it never imports
`run_pipeline`, never discovers a run, never contacts a model or the
network, and mutates nothing beyond the output paths its caller passes
explicitly. The driver hook is a later, separately reviewed change.

What the deterministic layer proves (and all it proves):
- the candidate is schema-valid with no unknown fields;
- the source digest matches the exact input bytes;
- every citation's quote equals the named byte slice inside its
  validator-owned source region;
- every substantive record is cited;
- when the source declares the BCP 14 framework, every operative keyword
  occurrence maps to exactly one normative record with matching strength;
- every source heading has a declared coverage disposition, and every
  paragraph/list unit under a mandatory-risk heading is covered by a
  dedicated record.

Whether a statement FOLLOWS from its quote is semantic fidelity — a
separate, externally owned review. This layer always reports
`semantic_status: "unreviewed"`; a producer cannot self-certify meaning.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:  # pragma: no cover - import plumbing
    sys.path.insert(0, str(_REPO_ROOT))

from pydantic import ValidationError

from schemas.generic_insights import (  # noqa: E402
    SCHEMA_MAJOR,
    SCHEMA_VERSION,
    GenericInsights,
    InsightKind,
    NormativeStrength,
)


# ---------------------------------------------------------------------------
# Source regions — validator-owned heading inventory
# ---------------------------------------------------------------------------

_ATX_HEADING = re.compile(rb"^(#{1,6})[ \t]+(.*?)[ \t#]*$", re.MULTILINE)


@dataclass
class SourceRegion:
    """One validator-owned coverage region: `h000` is the synthetic preamble
    (bytes before the first ATX heading, when non-whitespace) or, for a
    source with no headings at all, the whole document. `h001`, `h002`, ...
    are ATX headings in source order, any level. `body_span` is half-open,
    from just past the heading line's terminator to the first byte of the
    next ATX heading of any level (or EOF); regions never overlap."""

    region_id: str
    level: int  # 0 for the synthetic preamble/whole-document region
    text: str  # exact heading text ("" for synthetic regions)
    normalized: str
    occurrence: int  # 1-based among same-normalized-text headings
    body_span: tuple[int, int]


def _normalize_heading(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def _strip_section_number(normalized: str) -> str:
    """Strip a leading section number and punctuation for risk matching
    ("3.1 Security Considerations" -> "security considerations")."""
    return re.sub(r"^[0-9]+(\.[0-9]+)*[.):\s-]*\s*", "", normalized).strip()


def build_region_inventory(source: bytes) -> list[SourceRegion]:
    regions: list[SourceRegion] = []
    matches = list(_ATX_HEADING.finditer(source))
    if not matches:
        return [SourceRegion("h000", 0, "", "", 1, (0, len(source)))]
    counter = 1
    first_start = matches[0].start()
    if source[:first_start].strip():
        regions.append(SourceRegion("h000", 0, "", "", 1, (0, first_start)))
    seen: dict[str, int] = {}
    for i, m in enumerate(matches):
        text = m.group(2).decode("utf-8", errors="replace")
        normalized = _normalize_heading(text)
        seen[normalized] = seen.get(normalized, 0) + 1
        line_end = source.find(b"\n", m.end(0))
        body_start = len(source) if line_end < 0 else line_end + 1
        body_end = matches[i + 1].start() if i + 1 < len(matches) else len(source)
        regions.append(SourceRegion(
            region_id=f"h{counter:03d}",
            level=len(m.group(1)),
            text=text,
            normalized=normalized,
            occurrence=seen[normalized],
            body_span=(body_start, max(body_start, body_end)),
        ))
        counter += 1
    return regions


# ---------------------------------------------------------------------------
# Mandatory-risk coverage units
# ---------------------------------------------------------------------------

# Closed v1 set; a semantically similar heading may be PROPOSED for review
# but is never silently promoted into this deterministic gate.
MANDATORY_RISK_HEADINGS = frozenset({
    "security",
    "security considerations",
    "limitations",
    "limitations and future work",
    "safety considerations",
    "privacy considerations",
    "ethical considerations",
    "ethics",
    "threat model",
})

_LIST_ITEM = re.compile(rb"^[ \t]*(?:[-*+]|[0-9]+[.)])[ \t]+", re.MULTILINE)


@dataclass
class RiskUnit:
    unit_id: str
    region_id: str
    span: tuple[int, int]


def _trimmed_span(source: bytes, start: int, end: int) -> tuple[int, int] | None:
    chunk = source[start:end]
    stripped = chunk.strip()
    if not stripped:
        return None
    lead = len(chunk) - len(chunk.lstrip())
    trail = len(chunk) - len(chunk.rstrip())
    return (start + lead, end - trail)


def build_risk_units(source: bytes, regions: list[SourceRegion]) -> list[RiskUnit]:
    """Every non-empty prose paragraph and list item under a mandatory-risk
    heading becomes one coverage unit that demands its own dedicated record.
    This is what mechanically prevents one generic wide citation from
    marking a five-paragraph security section passed."""
    units: list[RiskUnit] = []
    n = 0
    for region in regions:
        if _strip_section_number(region.normalized) not in MANDATORY_RISK_HEADINGS:
            continue
        start, end = region.body_span
        body = source[start:end]
        # Paragraph blocks split on blank lines; list blocks split per item.
        for block_match in re.finditer(rb"(?s)\S.*?(?=\n[ \t]*\n|\Z)", body):
            b_start, b_end = start + block_match.start(), start + block_match.end()
            block = source[b_start:b_end]
            item_starts = [m.start() for m in _LIST_ITEM.finditer(block)]
            if item_starts:
                bounds = item_starts + [len(block)]
                pieces = [
                    (b_start + bounds[i], b_start + bounds[i + 1])
                    for i in range(len(item_starts))
                ]
                # Prose before the first list item is its own unit.
                if item_starts[0] > 0:
                    pieces.insert(0, (b_start, b_start + item_starts[0]))
            else:
                pieces = [(b_start, b_end)]
            for p_start, p_end in pieces:
                span = _trimmed_span(source, p_start, p_end)
                if span is None:
                    continue
                n += 1
                units.append(RiskUnit(f"risk-{n:03d}", region.region_id, span))
    return units


# ---------------------------------------------------------------------------
# BCP 14 normative occurrences
# ---------------------------------------------------------------------------

# Longest token first so "MUST NOT" is never consumed as "MUST" + stray NOT.
_BCP14_TOKENS: tuple[str, ...] = (
    "MUST NOT", "SHALL NOT", "SHOULD NOT", "NOT RECOMMENDED",
    "MUST", "REQUIRED", "SHALL", "SHOULD", "RECOMMENDED", "MAY", "OPTIONAL",
)

_TOKEN_TO_STRENGTH = {t: NormativeStrength(t.replace(" ", "_")) for t in _BCP14_TOKENS}

_BCP14_PATTERN = re.compile(
    rb"\b(" + b"|".join(t.replace(" ", r" ").encode() for t in _BCP14_TOKENS) + rb")\b"
)

# The keyword-definition boilerplate ("The key words \"MUST\" ... are to be
# interpreted as described in BCP 14 [RFC2119] [RFC8174] ...") both DECLARES
# the framework and is the only scanner exclusion: its token mentions define
# vocabulary rather than imposing requirements.
_BCP14_DECLARATION = re.compile(
    rb"(?is)key\s*words?\b.{0,400}?\binterpreted\s+as\s+described\s+in\b"
    rb".{0,120}?\b(BCP\s*14|RFC\s*2119)\b"
)


@dataclass
class NormativeOccurrence:
    occurrence_id: str
    token: str
    token_span: tuple[int, int]
    clause_span: tuple[int, int]


def find_bcp14_declaration(source: bytes) -> tuple[int, int] | None:
    """The declaration paragraph's span, or None when the source never
    declares the framework. Natural-language lowercase modals are NOT
    normative rules; without a declaration the scanner does not apply."""
    m = _BCP14_DECLARATION.search(source)
    if not m:
        return None
    para_start = source.rfind(b"\n\n", 0, m.start())
    para_start = 0 if para_start < 0 else para_start + 2
    para_end = source.find(b"\n\n", m.end())
    para_end = len(source) if para_end < 0 else para_end
    return (para_start, para_end)


def _clause_span(source: bytes, token_start: int, token_end: int) -> tuple[int, int]:
    """The minimal containing clause: back to the previous sentence
    terminator or paragraph break, forward to the next. Two modals in one
    sentence share a clause span but remain distinct occurrences."""
    window_start = max(source.rfind(b"\n\n", 0, token_start), 0)
    starts = [window_start]
    for sep in (b". ", b".\n", b"! ", b"? ", b":\n"):
        pos = source.rfind(sep, window_start, token_start)
        if pos >= 0:
            starts.append(pos + len(sep))
    start = max(starts)
    para_end = source.find(b"\n\n", token_end)
    end = len(source) if para_end < 0 else para_end
    for sep in (b". ", b".\n", b"! ", b"? "):
        pos = source.find(sep, token_end, end)
        if pos >= 0:
            end = min(end, pos + 1)
    span = _trimmed_span(source, start, end)
    return span if span else (token_start, token_end)


def scan_normative_occurrences(source: bytes) -> list[NormativeOccurrence]:
    """Every operative BCP 14 occurrence, longest-token-first, with a stable
    id in source order. Empty when the framework is not declared. The only
    exclusion is the declaration boilerplate itself — a producer reason can
    never excuse an operative clause."""
    declaration = find_bcp14_declaration(source)
    if declaration is None:
        return []
    occurrences: list[NormativeOccurrence] = []
    n = 0
    for m in _BCP14_PATTERN.finditer(source):
        if declaration[0] <= m.start() < declaration[1]:
            continue
        n += 1
        occurrences.append(NormativeOccurrence(
            occurrence_id=f"bcp-{n:04d}",
            token=m.group(1).decode("ascii"),
            token_span=(m.start(1), m.end(1)),
            clause_span=_clause_span(source, m.start(1), m.end(1)),
        ))
    return occurrences


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


@dataclass
class ValidationResult:
    deterministic_status: str  # passed | partial | invalid
    semantic_status: str
    findings: dict[str, list[str]]
    regions: list[SourceRegion]
    risk_units: list[RiskUnit]
    occurrences: list[NormativeOccurrence]
    model: GenericInsights | None
    source_sha256: str
    candidate_sha256: str
    partial_reasons: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.deterministic_status in {"passed", "partial"}


def _candidate_digest(candidate_bytes: bytes) -> str:
    return hashlib.sha256(candidate_bytes).hexdigest()


def validate_candidate(
    paper_path: Path,
    candidate_path: Path,
    *,
    expected_source_path: str | None = None,
) -> ValidationResult:
    """Validate one candidate against one source. Both paths are explicit;
    nothing else is read. `expected_source_path` (when given) is compared
    against the candidate's inert `source.path` metadata — the file that is
    actually read is always `paper_path`."""
    findings: dict[str, list[str]] = {
        "schema": [], "source_grounding": [], "coverage": [],
        "normative": [], "risk_coverage": [],
    }
    partial_reasons: list[str] = []

    source = Path(paper_path).read_bytes()
    source_sha = hashlib.sha256(source).hexdigest()
    candidate_bytes = Path(candidate_path).read_bytes()
    candidate_sha = _candidate_digest(candidate_bytes)

    regions = build_region_inventory(source)
    risk_units = build_risk_units(source, regions)
    occurrences = scan_normative_occurrences(source)

    def _result(status: str, model: GenericInsights | None) -> ValidationResult:
        return ValidationResult(
            deterministic_status=status,
            semantic_status="unreviewed",
            findings=findings,
            regions=regions,
            risk_units=risk_units,
            occurrences=occurrences,
            model=model,
            source_sha256=source_sha,
            candidate_sha256=candidate_sha,
            partial_reasons=partial_reasons,
        )

    # --- schema ---
    try:
        payload = json.loads(candidate_bytes)
    except json.JSONDecodeError as exc:
        findings["schema"].append(f"candidate is not valid JSON: {exc}")
        return _result("invalid", None)
    try:
        model = GenericInsights.model_validate(payload)
    except ValidationError as exc:
        findings["schema"].extend(
            f"{'.'.join(str(p) for p in err['loc'])}: {err['msg']}"
            for err in exc.errors())
        return _result("invalid", None)

    declared_major = model.schema_version.split(".", 1)[0]
    if declared_major != SCHEMA_MAJOR:
        findings["schema"].append(
            f"schema_version {model.schema_version!r} has unknown major "
            f"version (this validator understands {SCHEMA_VERSION})")
        return _result("invalid", model)

    # --- source binding ---
    if model.source.sha256 != source_sha:
        findings["source_grounding"].append(
            f"source.sha256 {model.source.sha256} does not match the input "
            f"bytes ({source_sha})")
        return _result("invalid", model)
    if expected_source_path is not None and model.source.path != expected_source_path:
        findings["source_grounding"].append(
            f"source.path {model.source.path!r} does not match the expected "
            f"relative path {expected_source_path!r}")

    region_by_id = {r.region_id: r for r in regions}

    # --- citations ---
    for cit in model.citations:
        region = region_by_id.get(cit.region_id)
        if region is None:
            findings["source_grounding"].append(
                f"citation {cit.id}: unknown region {cit.region_id!r}")
            continue
        start, end = cit.span.start, cit.span.end
        if end > len(source):
            findings["source_grounding"].append(
                f"citation {cit.id}: span [{start}, {end}) exceeds the "
                f"source ({len(source)} bytes)")
            continue
        if not (region.body_span[0] <= start and end <= region.body_span[1]):
            findings["source_grounding"].append(
                f"citation {cit.id}: span [{start}, {end}) is outside "
                f"region {cit.region_id} body "
                f"[{region.body_span[0]}, {region.body_span[1]})")
            continue
        if source[start:end] != cit.quote.encode("utf-8"):
            findings["source_grounding"].append(
                f"citation {cit.id}: quote does not equal the source bytes "
                f"at [{start}, {end})")

    # --- normative occurrences ---
    occurrence_by_id = {o.occurrence_id: o for o in occurrences}
    citation_by_id = {c.id: c for c in model.citations}
    mapped: dict[str, str] = {}
    for rec in model.insights:
        if rec.kind is not InsightKind.normative_rule:
            continue
        norm = rec.normative
        assert norm is not None  # schema-enforced
        occ = occurrence_by_id.get(norm.occurrence_id)
        if occ is None:
            findings["normative"].append(
                f"insight {rec.id}: unknown normative occurrence "
                f"{norm.occurrence_id!r}")
            continue
        if norm.occurrence_id in mapped:
            findings["normative"].append(
                f"occurrence {norm.occurrence_id} is mapped by both "
                f"{mapped[norm.occurrence_id]} and {rec.id}")
            continue
        mapped[norm.occurrence_id] = rec.id
        if norm.strength.source_token != occ.token:
            findings["normative"].append(
                f"insight {rec.id}: strength {norm.strength.source_token!r} "
                f"does not match occurrence {norm.occurrence_id} token "
                f"{occ.token!r} — weakening or strengthening a source "
                f"modality is rejected deterministically")
        # The cited quote must enclose the token occurrence and its clause.
        enclosed = any(
            (c := citation_by_id.get(cid)) is not None
            and c.span.start <= occ.clause_span[0]
            and occ.clause_span[1] <= c.span.end
            for cid in rec.citation_ids)
        if not enclosed:
            findings["normative"].append(
                f"insight {rec.id}: no citation encloses occurrence "
                f"{norm.occurrence_id}'s token and minimal clause "
                f"[{occ.clause_span[0]}, {occ.clause_span[1]})")
    for occ in occurrences:
        if occ.occurrence_id not in mapped:
            findings["normative"].append(
                f"operative occurrence {occ.occurrence_id} ({occ.token!r} at "
                f"[{occ.token_span[0]}, {occ.token_span[1]})) is not mapped "
                f"by any normative record")

    # --- heading coverage ---
    declared = {h.region_id: h for h in model.coverage.headings}
    for region in regions:
        if region.region_id not in declared:
            findings["coverage"].append(
                f"region {region.region_id} "
                f"({region.text or 'preamble'!s}) has no coverage disposition")
    for region_id in declared:
        if region_id not in region_by_id:
            findings["coverage"].append(
                f"coverage declares unknown region {region_id!r}")
    # `covered` must be computed, not asserted: at least one substantive
    # record cites a span inside the heading body.
    cited_spans_by_region: dict[str, list[tuple[int, int]]] = {}
    substantive_citation_ids = set(model.summary.citation_ids)
    substantive_citation_ids.update(model.intuition.citation_ids)
    for v in model.vocabulary:
        substantive_citation_ids.update(v.citation_ids)
    for rec in model.insights:
        substantive_citation_ids.update(rec.citation_ids)
    for cid in substantive_citation_ids:
        cit = citation_by_id.get(cid)
        if cit is not None:
            cited_spans_by_region.setdefault(cit.region_id, []).append(
                (cit.span.start, cit.span.end))
    for region_id, decl in declared.items():
        if decl.disposition.value == "covered" and \
                not cited_spans_by_region.get(region_id):
            findings["coverage"].append(
                f"region {region_id} is declared covered but no substantive "
                f"record cites a span inside it")

    # --- mandatory-risk units ---
    unit_owner: dict[str, str] = {}
    for rec in model.insights:
        for cid in rec.citation_ids:
            cit = citation_by_id.get(cid)
            if cit is None:
                continue
            for unit in risk_units:
                if (cit.span.start, cit.span.end) == unit.span:
                    if rec.id in unit_owner.values() and \
                            unit_owner.get(unit.unit_id) != rec.id:
                        # A record may own at most one unit.
                        prior = [u for u, r in unit_owner.items() if r == rec.id]
                        findings["risk_coverage"].append(
                            f"record {rec.id} claims risk units "
                            f"{prior[0]} and {unit.unit_id}; each unit needs "
                            f"its own dedicated record")
                        continue
                    if unit.unit_id in unit_owner:
                        findings["risk_coverage"].append(
                            f"risk unit {unit.unit_id} is claimed by both "
                            f"{unit_owner[unit.unit_id]} and {rec.id}")
                        continue
                    unit_owner[unit.unit_id] = rec.id
    for unit in risk_units:
        if unit.unit_id not in unit_owner:
            findings["risk_coverage"].append(
                f"mandatory-risk unit {unit.unit_id} in region "
                f"{unit.region_id} (bytes [{unit.span[0]}, {unit.span[1]})) "
                f"has no dedicated covering record")

    # --- status ---
    if any(findings.values()):
        return _result("invalid", model)
    if not model.vocabulary:
        partial_reasons.append(
            "vocabulary: none identified — "
            + (model.vocabulary_none_identified_reason or ""))
    for h in model.coverage.headings:
        if h.disposition.value == "missing":
            partial_reasons.append(
                f"heading {h.region_id} declared missing: {h.reason}")
    for item in model.coverage.missing:
        partial_reasons.append(f"declared missing: {item}")
    return _result("partial" if partial_reasons else "passed", model)


def validation_sidecar_payload(result: ValidationResult) -> dict:
    """The validator-owned sidecar (`generic_insights.validation.json`).
    No timestamp and no absolute path — identical inputs produce identical
    bytes. The candidate can never set these verdicts."""
    return {
        "schema_version": SCHEMA_VERSION,
        "source_sha256": result.source_sha256,
        "candidate_sha256": result.candidate_sha256,
        "deterministic_status": result.deterministic_status,
        "semantic_status": result.semantic_status,
        "findings": result.findings,
        "partial_reasons": result.partial_reasons,
        "region_inventory": [
            {
                "region_id": r.region_id, "level": r.level, "text": r.text,
                "normalized": r.normalized, "occurrence": r.occurrence,
                "body_span": list(r.body_span),
            }
            for r in result.regions
        ],
        "risk_unit_inventory": [
            {"unit_id": u.unit_id, "region_id": u.region_id,
             "span": list(u.span)}
            for u in result.risk_units
        ],
        "normative_occurrence_inventory": [
            {"occurrence_id": o.occurrence_id, "token": o.token,
             "token_span": list(o.token_span),
             "clause_span": list(o.clause_span)}
            for o in result.occurrences
        ],
    }


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

_AFFORDANCE_CAVEAT = (
    "conceptual opportunity only — no code, conformance, or verification "
    "evidence was produced for this item")

_KIND_HEADINGS: tuple[tuple[InsightKind, str], ...] = (
    (InsightKind.behavior, "Behavior"),
    (InsightKind.normative_rule, "Normative rules"),
    (InsightKind.implementation_affordance, "Implementation affordances"),
    (InsightKind.limitation, "Limitations"),
    (InsightKind.security_consideration, "Security considerations"),
    (InsightKind.empirical_observation, "Empirical observations"),
)


def _render_normative(rec) -> str:
    norm = rec.normative
    parts = [f"{norm.subject} {norm.strength.source_token} {norm.action}"]
    if norm.conditions:
        parts.append("Conditions: " + "; ".join(norm.conditions) + ".")
    if norm.exceptions:
        parts.append("Exceptions: " + "; ".join(norm.exceptions) + ".")
    parts.append(f"(occurrence `{norm.occurrence_id}`)")
    return " ".join(parts)


def render_insights_md(result: ValidationResult) -> str:
    """Deterministic Markdown for a validated (`passed` or `partial`) model.
    Raises ValueError on an invalid result — rendering invalid content is
    the caller's bug, and the CLI never does it."""
    if not result.ok or result.model is None:
        raise ValueError("refusing to render an invalid insight candidate")
    m = result.model
    lines: list[str] = []
    lines.append("# Document insights")
    lines.append("")
    lines.append(
        f"<!-- generic_insights schema {m.schema_version} | "
        f"source sha256 {result.source_sha256} | "
        f"candidate sha256 {result.candidate_sha256} | "
        f"deterministic {result.deterministic_status} | "
        f"semantic {result.semantic_status} -->")
    lines.append("")
    lines.append(
        f"Document kind: {m.document.kind.value} "
        f"({', '.join(m.document.contribution_kinds)}). "
        f"Source grounding was machine-checked against the parsed source; "
        f"whether each statement faithfully represents the source is a "
        f"separate review and is currently **{result.semantic_status}**.")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append(m.summary.text)
    lines.append("")
    lines.append("## Intuition")
    lines.append("")
    lines.append(m.intuition.text)
    lines.append("")
    lines.append("## Vocabulary (paper-local)")
    lines.append("")
    if m.vocabulary:
        for v in m.vocabulary:
            rels = "".join(
                f" Related: {r.kind} -> {r.to_id}." for r in v.relationships)
            dep = (f" External definition not ingested: "
                   f"{v.unresolved_dependency}." if v.unresolved_dependency else "")
            lines.append(
                f"- **{v.term}** ({v.domain}, confidence {v.confidence.value}): "
                f"{v.meaning}{rels}{dep}")
    else:
        lines.append(
            f"- None identified: {m.vocabulary_none_identified_reason}")
    lines.append("")
    for kind, heading in _KIND_HEADINGS:
        records = [r for r in m.insights if r.kind is kind]
        if not records:
            continue
        lines.append(f"## {heading}")
        lines.append("")
        for rec in records:
            if kind is InsightKind.normative_rule:
                body = _render_normative(rec)
            else:
                body = rec.statement or ""
            if kind is InsightKind.implementation_affordance:
                body = f"{body} ({_AFFORDANCE_CAVEAT})"
            if rec.steps:
                body += " Steps: " + " -> ".join(rec.steps) + "."
            lines.append(f"- {body} [confidence: {rec.confidence.value}]")
        lines.append("")
    # Always present, even when empty: a partial artifact is useful, but it
    # cannot look exhaustive by omission.
    lines.append("## Missing or unresolved coverage")
    lines.append("")
    gaps: list[str] = []
    for h in m.coverage.headings:
        if h.disposition.value != "covered":
            gaps.append(f"- {h.region_id}: {h.disposition.value} — {h.reason}")
    gaps.extend(f"- declared missing: {item}" for item in m.coverage.missing)
    gaps.extend(
        f"- unresolved external dependency: {item}"
        for item in m.coverage.unresolved_dependencies)
    if gaps:
        lines.extend(gaps)
    else:
        lines.append("- none declared")
    lines.append("")
    return "\n".join(lines)


def atomic_write_text(path: Path, text: str) -> None:
    """Write via a same-directory temp file + rename so a failed render can
    never leave a truncated researcher-facing file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
        os.replace(tmp_name, path)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


# ---------------------------------------------------------------------------
# Fixture-oriented CLI (explicit paths only; never discovers a run)
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Validate a generic-insight candidate against its source and "
            "optionally render the internal INSIGHTS.md. Standalone and "
            "network-free; all paths are explicit."))
    parser.add_argument("--paper", required=True, type=Path,
                        help="authoritative parsed source (paper.md bytes)")
    parser.add_argument("--candidate", required=True, type=Path,
                        help="generic_insights.json candidate")
    parser.add_argument("--out-validation", type=Path, default=None,
                        help="write the validation sidecar JSON here")
    parser.add_argument("--render", type=Path, default=None,
                        help="render INSIGHTS.md here (only when valid; an "
                             "invalid candidate never creates or overwrites "
                             "rendered output)")
    parser.add_argument("--expected-source-path", default=None,
                        help="expected inert source.path metadata value")
    args = parser.parse_args(argv)

    try:
        result = validate_candidate(
            args.paper, args.candidate,
            expected_source_path=args.expected_source_path)
    except OSError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if args.out_validation is not None:
        atomic_write_text(
            args.out_validation,
            json.dumps(validation_sidecar_payload(result), indent=2,
                       sort_keys=True) + "\n")
    if args.render is not None and result.ok:
        atomic_write_text(args.render, render_insights_md(result))

    print(f"deterministic_status: {result.deterministic_status}")
    print(f"semantic_status: {result.semantic_status}")
    for dimension, items in result.findings.items():
        for item in items:
            print(f"[{dimension}] {item}")
    return 0 if result.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
