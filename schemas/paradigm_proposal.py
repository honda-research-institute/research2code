"""Schemas for candidate taxonomy-pack proposal packets."""

from __future__ import annotations

import re
from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

SCHEMA_VERSION = "1.0.0"
_PARADIGM_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_-]*(/[a-z0-9][a-z0-9_-]*)*$")

ProposalDecision = Literal["new_subparadigm_needed", "new_top_level_needed"]


class ProposalEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid")

    paper_section: str
    quote_or_observation: str
    relevance: str


class FamilyComponentDeclaration(BaseModel):
    """One legal `arch_contract.family_components` name a pack declares.

    Arch-contract headroom (approved 2026-07-16): the universal ArchContract
    skeleton stays `extra='forbid'`, so a family that needs a top-level
    component beyond it (single-agent RL's reward_function, SRL 2026-07-15;
    the shape stochastic_optimization's optimizer_state was hard-added for,
    ADAM 2026-07-08) declares the name in its pack's `family_components`
    mapping. The declaration flows through the build plan into the Stage 2.d
    static validator, which enforces the declared set in both directions.
    `validate_paradigm_proposal.py` type-checks gap-pack declarations against
    this model at authoring time so a malformed declaration fails the
    proposal loop, not a live run."""

    model_config = ConfigDict(extra="forbid")

    required: bool = Field(
        default=True,
        description=(
            "Whether the contract MUST carry this component (default). "
            "False declares the name as legal but optional."
        ),
    )
    description: Optional[str] = Field(
        default=None,
        description="What the component is, in the family's terms.",
    )
    required_entries: list[str] = Field(
        default_factory=list,
        description=(
            "Entry names the contract's component block must carry in its "
            "`entries` map (multi-slot components only, e.g. optimizer "
            "state's ['step', 'params']). Empty for single-slot components."
        ),
    )


# The pack-declared build-plan routing vocabulary (R2C-032, approved
# 2026-07-28). Owned here (the schemas module imports nothing from scripts)
# so the authoring validator and the build-plan resolver cannot drift.
BUILD_PLAN_SOURCE_INHERIT = "inherit_parent"
BUILD_PLAN_SOURCE_NEUTRAL = "neutral"


class BuildPlanChoice(BaseModel):
    """The pack author's build-plan routing choice (R2C-032).

    A provisional pack extending a committed family used to inherit the
    parent's static build plan implicitly (the ancestor walk in
    `scripts/build_plan.py::_static_plan_key`). The 2026-07-28 SRL run
    showed why that must be a declared choice: the parent motion-planning
    manifest demanded dynamics/collision classes an RL method can never
    satisfy, and the architecture coder and the validator could not both be
    right. `inherit_parent` keeps today's walk (and the full delivery-label
    range); `neutral` takes the generic provisional plan (and caps the
    delivery at uncertified — new territory). Packs that declare nothing
    resolve exactly as before — the walk — so already-installed packs are
    untouched."""

    model_config = ConfigDict(extra="forbid")

    source: Literal["inherit_parent", "neutral"]
    reasoning: Optional[str] = Field(
        default=None,
        description=(
            "Why this pack fits (or cannot fit) the parent's build shape, "
            "in the paper's terms."
        ),
    )


class ParadigmProposal(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = SCHEMA_VERSION
    proposal_id: str
    source_paper_slug: str
    source_gap_report: str = ".pipeline/paradigm_gap_report.json"
    decision: ProposalDecision
    target_paradigm_id: str
    target_taxonomy_id: Optional[str] = Field(
        default=None,
        description=(
            "Optional proposed taxonomy id. Maintainers may adjust it before "
            "promotion into docs/ssot/taxonomies.yaml."
        ),
    )
    extends: Optional[str] = None
    title: str
    scope_summary: str
    pack: dict[str, Any] = Field(
        default_factory=dict,
        description="Small provisional taxonomy-node pack authored for review.",
    )
    evidence: list[ProposalEvidence] = Field(min_length=1)
    coupling_warnings: list[str] = Field(default_factory=list)
    validation_command: str = "python3 scripts/validate_paradigm_proposal.py <proposal_dir>"

    @field_validator("target_paradigm_id", "extends")
    @classmethod
    def _valid_paradigm_id(cls, value: str | None) -> str | None:
        if value is None:
            return None
        if not _PARADIGM_ID_RE.match(value):
            raise ValueError(
                "paradigm ids must be lowercase path-form strings using letters, "
                "numbers, underscores, hyphens, and slashes"
            )
        return value

    @model_validator(mode="after")
    def _decision_consistency(self) -> "ParadigmProposal":
        if self.decision == "new_subparadigm_needed":
            if not self.extends:
                raise ValueError("new sub-paradigm proposals require extends")
            if not self.target_paradigm_id.startswith(f"{self.extends}/"):
                raise ValueError("target_paradigm_id must be below extends")
        if self.decision == "new_top_level_needed" and self.extends is not None:
            raise ValueError("new top-level proposals must set extends to null")
        if self.decision == "new_top_level_needed" and "/" in self.target_paradigm_id:
            raise ValueError("new top-level proposals must use a top-level target_paradigm_id")
        return self


class ProposalValidationResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = SCHEMA_VERSION
    proposal_dir: str
    valid: bool
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    proposal: ParadigmProposal | None = None
