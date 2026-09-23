"""Schema for Stage 1 paradigm-gap reports.

When the analyzer cannot classify a paper into an existing taxonomy node, the
pipeline should halt with structured evidence instead of a prose-only sidecar.
This artifact is the input to later taxonomy-pack proposal tooling.
"""

from __future__ import annotations

from enum import Enum
import re
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


SCHEMA_VERSION = "1.0.0"
_PARADIGM_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_-]*(/[a-z0-9][a-z0-9_-]*)*$")


class GapDecision(str, Enum):
    existing_exact_match = "existing_exact_match"
    new_subparadigm_needed = "new_subparadigm_needed"
    new_top_level_needed = "new_top_level_needed"
    unsupported_or_unclear = "unsupported_or_unclear"


class EvidenceQuote(BaseModel):
    model_config = ConfigDict(extra="forbid")

    paper_section: str = Field(description="Paper section, heading, or paper_map element id.")
    quote_or_observation: str = Field(description="Short paper-grounded evidence for the decision.")
    relevance: str = Field(description="Why this evidence supports the paradigm decision.")


class ParadigmCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    paradigm_id: str
    taxonomy_id: Optional[str] = None
    fit: str = Field(description="Why this existing taxonomy node was considered.")
    decision: str = Field(description="Accepted/rejected/partial fit rationale.")


class ParadigmGapReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = SCHEMA_VERSION
    decision: GapDecision
    confidence: float = Field(ge=0.0, le=1.0)
    paper_slug: Optional[str] = None
    paper_title: Optional[str] = None
    paper_paradigm_summary: str
    # The three id fields below use the LOWERCASE legacy paradigm-id
    # vocabulary (what `serves()`/`load_pack()` resolve and the proposal
    # tooling consumes), never the canonical taxonomy ids with uppercase
    # family codes (`TE-TS/...`). The 2026-07-06 fedavg gap report halted
    # on exactly that confusion; the analyzer prompt now pins the form.
    # Unifying the two vocabularies at ingest is a named Opus-queue item.
    matched_existing_paradigm: Optional[str] = Field(
        default=None,
        description="Legacy paradigm id, lowercase path form "
                    "(e.g. 'active_learning/bayesian').")
    proposed_parent_paradigm: Optional[str] = Field(
        default=None,
        description="Legacy paradigm id of the parent, lowercase path form. "
                    "No taxonomy family codes, no prose glosses.")
    proposed_new_paradigm_id: Optional[str] = Field(
        default=None,
        description="Proposed id in lowercase legacy path form "
                    "(e.g. 'federated_learning' for a new top level).")
    candidate_matches: list[ParadigmCandidate] = Field(default_factory=list)
    rejected_matches: list[ParadigmCandidate] = Field(default_factory=list)
    registered_paradigms: list[str] = Field(default_factory=list)
    paper_evidence: list[EvidenceQuote] = Field(min_length=1)
    recommended_next_action: str

    @field_validator(
        "matched_existing_paradigm",
        "proposed_parent_paradigm",
        "proposed_new_paradigm_id",
    )
    @classmethod
    def _valid_paradigm_id(cls, value: str | None) -> str | None:
        if value is None:
            return None
        if not _PARADIGM_ID_RE.match(value):
            # Name the offending value and the expected shape: this text
            # surfaces in the stage-1 halt artifact, and a vague rule with
            # no example is how the 2026-07-06 fedavg run died.
            hint = ""
            if value != value.lower():
                hint = (" This looks like a canonical taxonomy id "
                        "(uppercase family code); use the lowercase legacy "
                        "paradigm-id vocabulary from registered_paradigms "
                        "instead.")
            raise ValueError(
                f"paradigm ids must be lowercase path-form strings using "
                f"letters, numbers, underscores, hyphens, and slashes — "
                f"got {value!r}, expected a form like "
                f"'active_learning/bayesian'.{hint}"
            )
        return value

    @model_validator(mode="after")
    def _decision_consistency(self) -> "ParadigmGapReport":
        if self.decision == GapDecision.existing_exact_match:
            if not self.matched_existing_paradigm:
                raise ValueError("existing_exact_match requires matched_existing_paradigm")
            if self.proposed_new_paradigm_id:
                raise ValueError("existing_exact_match must not propose a new paradigm id")
        elif self.decision == GapDecision.new_subparadigm_needed:
            if not self.proposed_parent_paradigm:
                raise ValueError("new_subparadigm_needed requires proposed_parent_paradigm")
            if not self.proposed_new_paradigm_id:
                raise ValueError("new_subparadigm_needed requires proposed_new_paradigm_id")
            parent_prefix = f"{self.proposed_parent_paradigm}/"
            if not self.proposed_new_paradigm_id.startswith(parent_prefix):
                raise ValueError(
                    "new sub-paradigm id must be below proposed_parent_paradigm"
                )
            if self.proposed_new_paradigm_id == self.proposed_parent_paradigm:
                raise ValueError("new sub-paradigm id must be more specific than parent")
        elif self.decision == GapDecision.new_top_level_needed:
            if not self.proposed_new_paradigm_id:
                raise ValueError("new_top_level_needed requires proposed_new_paradigm_id")
            if "/" in self.proposed_new_paradigm_id:
                raise ValueError("new top-level paradigm id must not contain slash")
            if self.proposed_parent_paradigm:
                raise ValueError("new_top_level_needed must not set proposed_parent_paradigm")
        elif self.decision == GapDecision.unsupported_or_unclear:
            if self.matched_existing_paradigm or self.proposed_new_paradigm_id:
                raise ValueError("unsupported_or_unclear must not select or propose a paradigm")
        return self
