"""
Pydantic model for the generic insight artifact (Block 7, pure tranche).

The generic understanding tier's canonical candidate: a source-grounded
account of a document — summary, intuition, paper-local vocabulary, and
insight records — where every substantive statement carries exact byte-span
citations into the authoritative parsed source (`.pipeline/paper.md` in a
run). The contract is the generic insight artifact contract note
(internal, not shipped); this module is the schema half only.
Deterministic validation (source grounding, coverage, normative occurrences)
lives in scripts/generic_insights.py — schema validity here never implies
source fidelity.

Discipline mirrors `paper_map.py` / `method_spec.py`:
- `extra="forbid"` everywhere — typos and drift are validation errors.
- Cross-reference invariants (citation ids, vocabulary ids) are enforced.
- `document.kind` is informational, never a control-flow signal.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, model_validator


SCHEMA_VERSION = "1.0.0"

# The version family this code understands. An unknown MAJOR fails closed in
# the validator; minor/patch drift is tolerated at parse time.
SCHEMA_MAJOR = SCHEMA_VERSION.split(".", 1)[0]


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class DocumentKind(str, Enum):
    research_method = "research_method"
    technical_standard = "technical_standard"
    other = "other"


class Confidence(str, Enum):
    high = "high"
    medium = "medium"
    low = "low"


class InsightKind(str, Enum):
    behavior = "behavior"
    normative_rule = "normative_rule"
    implementation_affordance = "implementation_affordance"
    limitation = "limitation"
    security_consideration = "security_consideration"
    empirical_observation = "empirical_observation"


class Facet(str, Enum):
    security = "security"
    privacy = "privacy"
    limitation = "limitation"


class NormativeStrength(str, Enum):
    """BCP 14 tokens. Underscored values render with their source spelling
    (`MUST_NOT` renders as `MUST NOT`)."""

    MUST = "MUST"
    MUST_NOT = "MUST_NOT"
    REQUIRED = "REQUIRED"
    SHALL = "SHALL"
    SHALL_NOT = "SHALL_NOT"
    SHOULD = "SHOULD"
    SHOULD_NOT = "SHOULD_NOT"
    RECOMMENDED = "RECOMMENDED"
    NOT_RECOMMENDED = "NOT_RECOMMENDED"
    MAY = "MAY"
    OPTIONAL = "OPTIONAL"

    @property
    def source_token(self) -> str:
        return self.value.replace("_", " ")


class ByteSpan(_Strict):
    """Zero-based UTF-8 byte span into the source, half-open [start, end)."""

    start: int = Field(ge=0)
    end: int = Field(ge=0)

    @model_validator(mode="after")
    def _ordered(self) -> "ByteSpan":
        if self.end <= self.start:
            raise ValueError(f"byte span must be non-empty: [{self.start}, {self.end})")
        return self


class Source(_Strict):
    """Inert logical metadata about the authoritative source. `path` is
    checked against the caller's expected relative path and is never
    resolved or opened by the validator — the caller passes the real file."""

    path: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class DocumentInfo(_Strict):
    kind: DocumentKind
    contribution_kinds: list[str] = Field(min_length=1)


class Citation(_Strict):
    """One exact source anchor. `region_id` is validator-owned (`h000`,
    `h001`, ...); the validator checks the quote's bytes equal the named
    slice and the slice falls inside the region body."""

    id: str
    region_id: str
    span: ByteSpan
    quote: str


class CitedText(_Strict):
    text: str = Field(min_length=1)
    citation_ids: list[str] = Field(min_length=1)


class VocabularyRelationship(_Strict):
    to_id: str
    kind: str = Field(min_length=1)


class VocabularyRecord(_Strict):
    id: str
    term: str = Field(min_length=1)
    meaning: str = Field(min_length=1)
    domain: str = Field(min_length=1)
    confidence: Confidence
    citation_ids: list[str] = Field(min_length=1)
    relationships: list[VocabularyRelationship] = Field(default_factory=list)
    # Cross-document definition the source uses but does not define
    # (e.g. "idempotent" pointing at [HTTP] Section 9.2.2). Recorded, never
    # improvised.
    unresolved_dependency: str | None = None


class NormativeContent(_Strict):
    """Structured normative statement. Rendering is derived from these
    fields so free prose cannot silently replace the modality. The
    `occurrence_id` binds to exactly one validator-scanned token
    occurrence."""

    framework: str = Field(min_length=1)
    strength: NormativeStrength
    subject: str = Field(min_length=1)
    action: str = Field(min_length=1)
    conditions: list[str] = Field(default_factory=list)
    exceptions: list[str] = Field(default_factory=list)
    occurrence_id: str = Field(min_length=1)


class InsightRecord(_Strict):
    id: str
    kind: InsightKind
    # Non-normative records carry prose here; normative records derive their
    # rendered statement from `normative` instead (enforced below).
    statement: str | None = None
    normative: NormativeContent | None = None
    citation_ids: list[str] = Field(min_length=1)
    vocabulary_ids: list[str] = Field(default_factory=list)
    confidence: Confidence
    # Closed cross-cutting facets; a facet never creates a second rendered
    # record. A normative rule tagged `security` can satisfy both its
    # occurrence and one mandatory-risk unit.
    facets: list[Facet] = Field(default_factory=list)
    # Behaviors may carry ordered steps.
    steps: list[str] = Field(default_factory=list)
    # Affordances must be explicitly conceptual; the renderer adds the fixed
    # no-code caveat. Meaningless (and forbidden) on other kinds.
    conceptual_only: bool | None = None
    # A limitation may point at the insight it qualifies.
    qualifies_insight_id: str | None = None

    @model_validator(mode="after")
    def _kind_shape(self) -> "InsightRecord":
        if self.kind is InsightKind.normative_rule:
            if self.normative is None:
                raise ValueError(f"{self.id}: normative_rule requires `normative`")
            if self.statement is not None:
                raise ValueError(
                    f"{self.id}: normative_rule renders from structured fields; "
                    "free-prose `statement` is not allowed")
        else:
            if self.normative is not None:
                raise ValueError(f"{self.id}: only normative_rule carries `normative`")
            if not (self.statement or "").strip():
                raise ValueError(f"{self.id}: non-normative records require `statement`")
        if self.kind is InsightKind.implementation_affordance:
            if self.conceptual_only is not True:
                raise ValueError(
                    f"{self.id}: implementation_affordance must set "
                    "conceptual_only=true — an insight artifact is never "
                    "evidence that code exists")
        elif self.conceptual_only is not None:
            raise ValueError(f"{self.id}: conceptual_only is affordance-only")
        return self


class HeadingDisposition(str, Enum):
    covered = "covered"
    not_applicable = "not_applicable"
    missing = "missing"


class HeadingCoverage(_Strict):
    region_id: str
    disposition: HeadingDisposition
    # Required for not_applicable and missing (enforced below).
    reason: str | None = None

    @model_validator(mode="after")
    def _reason_required(self) -> "HeadingCoverage":
        if self.disposition is not HeadingDisposition.covered and \
                not (self.reason or "").strip():
            raise ValueError(
                f"{self.region_id}: `{self.disposition.value}` requires a reason")
        return self


class Coverage(_Strict):
    headings: list[HeadingCoverage] = Field(min_length=1)
    # Explicit gaps the candidate declares (topics it could not ground,
    # sections it could not interpret). Rendered visibly even when empty.
    missing: list[str] = Field(default_factory=list)
    unresolved_dependencies: list[str] = Field(default_factory=list)


class GenericInsights(_Strict):
    """The canonical candidate (`.pipeline/generic_insights.json`)."""

    schema_version: str
    source: Source
    document: DocumentInfo
    summary: CitedText
    intuition: CitedText
    citations: list[Citation] = Field(min_length=1)
    vocabulary: list[VocabularyRecord] = Field(default_factory=list)
    # Required (with a visible reason) when vocabulary is empty; forbidden
    # otherwise. An empty list without reasoning is a validation error, and
    # even with reasoning the deterministic status caps at `partial`.
    vocabulary_none_identified_reason: str | None = None
    insights: list[InsightRecord] = Field(min_length=1)
    coverage: Coverage

    @model_validator(mode="after")
    def _cross_refs_resolve(self) -> "GenericInsights":
        problems: list[str] = []

        citation_ids = [c.id for c in self.citations]
        if len(set(citation_ids)) != len(citation_ids):
            problems.append("citation ids are not unique")
        known_citations = set(citation_ids)

        vocab_ids = [v.id for v in self.vocabulary]
        if len(set(vocab_ids)) != len(vocab_ids):
            problems.append("vocabulary ids are not unique")
        known_vocab = set(vocab_ids)

        insight_ids = [i.id for i in self.insights]
        if len(set(insight_ids)) != len(insight_ids):
            problems.append("insight ids are not unique")
        known_insights = set(insight_ids)

        def _check_citations(owner: str, ids: list[str]) -> None:
            for cid in ids:
                if cid not in known_citations:
                    problems.append(f"{owner} references unknown citation {cid!r}")

        _check_citations("summary", self.summary.citation_ids)
        _check_citations("intuition", self.intuition.citation_ids)
        for v in self.vocabulary:
            _check_citations(f"vocabulary {v.id}", v.citation_ids)
            for rel in v.relationships:
                if rel.to_id not in known_vocab:
                    problems.append(
                        f"vocabulary {v.id} relationship targets unknown id {rel.to_id!r}")
        for rec in self.insights:
            _check_citations(f"insight {rec.id}", rec.citation_ids)
            for vid in rec.vocabulary_ids:
                if vid not in known_vocab:
                    problems.append(
                        f"insight {rec.id} references unknown vocabulary {vid!r}")
            if rec.qualifies_insight_id is not None and \
                    rec.qualifies_insight_id not in known_insights:
                problems.append(
                    f"insight {rec.id} qualifies unknown insight "
                    f"{rec.qualifies_insight_id!r}")

        if self.vocabulary and self.vocabulary_none_identified_reason is not None:
            problems.append(
                "vocabulary_none_identified_reason is only allowed when the "
                "vocabulary list is empty")
        if not self.vocabulary and not \
                (self.vocabulary_none_identified_reason or "").strip():
            problems.append(
                "empty vocabulary requires vocabulary_none_identified_reason")

        coverage_regions = [h.region_id for h in self.coverage.headings]
        if len(set(coverage_regions)) != len(coverage_regions):
            problems.append("coverage declares a region more than once")

        if problems:
            raise ValueError("; ".join(problems))
        return self
