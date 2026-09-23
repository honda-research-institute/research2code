"""Pydantic model for `<RUN_DIR>/.pipeline/review_report.json`.

The paper-fidelity reviewer (Stage 4) produces this file after reading the
generated method/ package + notebook.ipynb + spec + paper + paper map. It's
the **semantic-review** counterpart of the deterministic validators that ran
during Stages 2–3.

The reviewer ONLY produces findings — it never edits files. The orchestrator
(Phase C) reads this report and either auto-resolves findings (where the
reviewer attached a `proposed_resolution`) or routes to the appropriate
producer agent for re-dispatch:

  - `proposed_resolution` set → driver applies the expert default and logs to
    `<run_dir>/assumptions.md`; the researcher reviews assumptions later
  - `severity: critical`, no resolution → auto-route to `target_agent` for re-dispatch
  - `severity: important` findings → surface to user via `deferred_findings.md`
  - `severity: nice-to-have` findings → surface only; never auto-route
  - `target_agent: human` → always surface; user decides

Same separation-of-concerns we use for the deterministic validators (validators
emit errors; orchestrator re-dispatches). Reviewer = semantic validator.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal, Union

from pydantic import BaseModel, ConfigDict, Field, model_validator


SCHEMA_VERSION = "1.2.0"


# Issue types — these are the categories the reviewer agent classifies findings into.
# New types can be added; the orchestrator routes by severity + target_agent, not by type.
IssueType = Literal[
    # Math / paper citation issues
    "math_citation_wrong",            # Prose cites Eq. X but the actual eq is different
    "math_citation_imprecise",        # Citation is roughly right but the framing misleads
    "invalid_paper_element_id",       # # paper-element: <id> doesn't exist in paper_map.json
    # Algorithm fidelity issues
    "algorithm_divergence",           # Code does something different from what the paper specifies
    "simplification_unflagged",       # Code simplifies the algorithm but doesn't say so
    "essential_feature_missing",      # spec marks a feature essential but it's not implemented
    "essential_annotation_missing",   # Implemented but no `# essential:` marker — reviewer can't trace
    # Documentation issues
    "undocumented_assumption",        # Closed-form / shortcut whose precondition isn't documented
    "stale_documentation",            # Comment / docstring references the wrong file/symbol/value
    "narrative_inaccuracy",           # Notebook markdown claim doesn't match the code below
    # Provenance issues
    "provenance_reasoning_inaccurate",  # params.json reasoning text doesn't match paper values
    # Generated per-element test issues (R2C-024)
    "test_defect",                    # A generated test misreads its paper element or asserts something the element does not state
    # Catch-all
    "other",
]


# Target-agent — who should fix the finding. The orchestrator uses this to route.
TargetAgent = Literal[
    "method-coder",        # Issues in method.py
    "architecture-coder",  # Issues in model.py / training.py
    "notebook-generator",  # Issues in notebook.ipynb
    "parameter-deriver",   # Issues in params.json
    "package-scaffolder",  # Issues in data.py / example_data/README.md / top-level README.md
    "analyzer",            # Issues in method_spec.json (Stage 1 bug)
    "test-generator",      # Issues in method/tests/ generated element tests
    "human",               # Cannot be auto-fixed; needs human judgment
]


class RescaleParamResolution(BaseModel):
    """Auto-resolution: rescale a numeric parameter to match the actual data
    scale used in the run. Applied deterministically by the driver — it edits
    `params.json` directly and logs to `assumptions.md`. No re-dispatch.

    Used for Pass 0.A (scale_dependent_hyperparameters mismatch). The
    canonical case: paper's `R_0=2000` calibrated for [0, 255] pixels, but
    data.py applies `ToTensor()` normalizing to [0, 1] — multiply by 1/255.
    """

    model_config = ConfigDict(extra="forbid")

    kind: Literal["rescale_param"] = "rescale_param"
    param: str = Field(
        description='Param name as it appears in params.json["params"] (e.g., "R_0").'
    )
    new_value: float = Field(
        description=(
            "Final value the driver should set. The reviewer does the math; "
            "the driver just writes the value."
        )
    )
    factor_derivation: str = Field(
        description=(
            "Short ASCII derivation, for the assumptions log. Example: "
            '"2000.0 * (1/255) = 7.843 — ratio between calibrated [0,255] '
            'range and actual [0,1] range."'
        )
    )
    expert_reasoning: str = Field(
        description=(
            "1-3 sentences for assumptions.md explaining *why* this rescaling "
            "is the expert default (vs. alternatives like skipping "
            "normalization). The researcher reads this to audit the decision."
        )
    )
    alternative: str | None = Field(
        default=None,
        description=(
            "Optional: alternative resolution an expert might prefer in some "
            'contexts (e.g., "skip ToTensor() in data.py to preserve raw '
            'pixel values"). Logged to assumptions.md so the researcher can '
            "override if needed."
        ),
    )


class RegenerateWithRequirementResolution(BaseModel):
    """Auto-resolution: re-dispatch a producer agent with explicit guidance
    bolted into the prompt. Used for Pass 0.B (required_model_methods that
    are missing or have the wrong body). The driver re-runs the producer's
    fix-mode dispatcher with `guidance` injected; if the post-regen re-review
    still has the same finding, `resolution_status` flips to `failed` and the
    finding surfaces to assumptions.md as needs-user.
    """

    model_config = ConfigDict(extra="forbid")

    kind: Literal["regenerate_with_requirement"] = "regenerate_with_requirement"
    target_agent: "TargetAgent" = Field(
        description="Which producer to re-dispatch with the guidance."
    )
    guidance: str = Field(
        description=(
            "Explicit, prescriptive guidance bolted into the producer's "
            "fix-mode prompt. Should reference the exact spec requirement "
            'and what the body must return. Example: "Re-implement '
            "forward_with_embedding(x) to return (logits, embedding) where "
            "embedding is the activation from the layer immediately before "
            "the classifier head. Currently returns (logits, logits). See "
            'spec.critical_requirements.required_model_methods[0]."'
        )
    )
    expected_outcome: str = Field(
        description=(
            "What the re-review should confirm after the regen. Used by the "
            "incremental re-reviewer to decide pass/fail. Example: "
            '"method/model.py defines forward_with_embedding returning a '
            "tuple where the second element is shape (B, hidden_dim) and "
            'sourced from the penultimate layer, not the final layer."'
        )
    )
    expert_reasoning: str = Field(
        description="1-3 sentences for assumptions.md explaining the choice."
    )
    alternative: str | None = Field(
        default=None,
        description="Optional alternative resolution, logged for the researcher.",
    )


class RelabelParamSourceResolution(BaseModel):
    """Auto-resolution: edit a parameter's provenance metadata in
    `params.json`. Used by the stage-reviewer (`stage_2x_params`) when it
    detects a `paper_source_values_match_paper` violation — a param labeled
    `source: paper` whose value the paper doesn't actually state.

    The driver applies this deterministically: load `params.json`, find the
    named param entry, set `source` to `new_source`, delete the keys in
    `remove_fields`, set the keys in `add_fields`, write back. Re-validate
    against the Params pydantic model afterward to catch any schema
    mismatch (e.g., new_source=system_inferred requires a `reasoning`
    field, so add_fields must include it).

    The canonical case: GBALD's `train_until_accuracy=0.99` labeled as
    `source: paper` with a `paper_section` citation, but Section 7.4 of
    the paper discusses MC dropout sample counts, not training accuracy
    thresholds. Correct labeling is `system_inferred` with `reasoning`.
    """

    model_config = ConfigDict(extra="forbid")

    kind: Literal["relabel_param_source"] = "relabel_param_source"
    param: str = Field(
        description=(
            'Param key in params.json (e.g., "train_until_accuracy", '
            '"learning_rate"). Driver finds the entry at '
            "`params.json[\"params\"][param]`."
        )
    )
    new_source: Literal["paper", "system_default", "system_inferred", "spec_default"] = Field(
        description=(
            "The correct provenance label. The driver also enforces the "
            "schema's per-source field requirements (`system_default` needs "
            "`paper_value` + `reasoning`; `system_inferred` needs `reasoning`; "
            "etc.) by re-validating after the edit."
        )
    )
    remove_fields: list[str] = Field(
        default_factory=list,
        description=(
            "Names of keys to delete from the param entry. Used when the "
            "new source is less-restrictive than the old one — e.g., "
            'switching from `paper` to `system_inferred` removes the now-'
            "misleading `paper_section` and `note` fields. Driver no-ops "
            "on keys that don't exist."
        ),
    )
    add_fields: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Keys to set on the param entry (overwriting if present). Used "
            "to add fields the new source requires, such as `reasoning` or "
            "numeric/structured `paper_value` metadata. This resolution kind "
            "does not touch the param's runtime `value`."
        )
    )
    expert_reasoning: str = Field(
        description=(
            "1-3 sentences for assumptions.md explaining why this relabel "
            "is the correct fix. The researcher reads this to audit the "
            "decision. Reference the paper section the reviewer checked "
            "and what it actually says vs. what the param claimed."
        )
    )
    alternative: str | None = Field(
        default=None,
        description=(
            "Optional: alternative resolution the researcher might prefer "
            "(e.g., \"if you have a paper reference for this value, "
            'restore source=paper with the correct paper_section\"). '
            "Logged to assumptions.md."
        ),
    )


ProposedResolution = Annotated[
    Union[
        RescaleParamResolution,
        RegenerateWithRequirementResolution,
        RelabelParamSourceResolution,
    ],
    Field(discriminator="kind"),
]


ResolutionStatus = Literal[
    "pending",     # reviewer wrote a proposed_resolution; driver hasn't acted yet
    "applied",     # driver applied the resolution successfully (deterministic, or post-regen converged)
    "failed",      # driver tried but the resolution didn't converge (regen still finds the issue)
    "needs_user",  # reviewer determined no expert default exists for this case; surfaces to user
]


class DisclosureRequest(BaseModel):
    """Attach to a nice-to-have finding or caveat that must reach a surface the
    researcher actually reads (assumptions.md), instead of only landing in
    deferred_findings.md where the 07-06 audit found the GBALD Eq. 13
    disclosure buried. The driver promotes it during stage-5 routing
    (`_promote_reviewer_disclosures`). Disclosure only: promotion never changes
    severity or the delivery label."""

    model_config = ConfigDict(extra="forbid")

    surface: Literal["assumptions"] = Field(
        description="Researcher surface to route to. v1 supports assumptions.md only."
    )
    note: str = Field(
        description="One-line, researcher-facing summary of what is being disclosed."
    )


class Finding(BaseModel):
    """One reviewer finding."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(description="Stable ID for this finding (e.g., 'F001', 'F002')")
    severity: Literal["critical", "important", "nice-to-have"]
    target_agent: TargetAgent
    issue_type: IssueType

    file: str = Field(description="Relative path of the offending file (e.g., 'method/method.py')")
    location: str = Field(
        description=(
            "Where in the file (e.g., 'compute_gradient_embeddings, lines 30-50' or "
            "'cell 21' for notebook cells)"
        )
    )
    related_elements: list[str] = Field(
        default_factory=list,
        description=(
            "The path join key (claims-ledger CT-3, P0 instrumentation §1.1): "
            "paper_map element ids this finding bears on. Lets the claims "
            "ledger ask whether a claim's code path is fidelity-clean by "
            "matching this finding against the SAME elements a probe verdict "
            "keys on (ProbeVerdict.element_ids), instead of the file-plus-prose "
            "match a transposed matrix or sign flip never trips. Additive and "
            "optional; empty until the verdict machinery populates it, and an "
            "empty key fails closed (a path with no matching finding is treated "
            "as not-provably-clean, never a silent pass)."
        ),
    )

    description: str = Field(description="What's wrong, in plain prose. 1-3 sentences.")
    proposed_fix: str | None = Field(
        default=None,
        description=(
            "Optional hint the producer agent can use when re-dispatched. Not binding — "
            "the producer makes the final call on how to fix."
        ),
    )

    proposed_resolution: ProposedResolution | None = Field(
        default=None,
        description=(
            "If set, the driver applies this resolution automatically and "
            "logs the decision to `<run_dir>/assumptions.md`. Reserved for "
            "Pass 0 (spec-asserted structured invariants) where the expert "
            "default is well-defined. For other findings (Pass 1-5), leave "
            "null and the driver falls back to traditional routing."
        ),
    )
    resolution_status: ResolutionStatus = Field(
        default="pending",
        description=(
            "Lifecycle state of the resolution. Reviewer sets `pending` (or "
            "`needs_user` if no expert default exists). Driver updates to "
            "`applied` after success or `failed` after a regen attempt did "
            "not converge."
        ),
    )

    disclosure_request: DisclosureRequest | None = Field(
        default=None,
        description=(
            "Set on a nice-to-have finding (or caveat) whose content the "
            "researcher must see, so the driver routes it to assumptions.md "
            "during stage-5 routing instead of leaving it only in "
            "deferred_findings.md. Prefer this over prose-asking ('document "
            "in assumptions.md'); the driver also has a text bridge that "
            "catches the prose form. Disclosure only, never adjudication."
        ),
    )


class ReviewReport(BaseModel):
    """Top-level review_report.json schema."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = SCHEMA_VERSION
    review_status: Literal["passed", "issues_found"]
    summary: str = Field(description="Plain-prose summary of the review pass. 1-3 sentences.")
    findings: list[Finding] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def _normalize_legacy_report(cls, data: Any) -> Any:
        """Accept committed pre-1.0 review reports without relaxing v1 output."""
        if not isinstance(data, dict) or "review_status" in data:
            return data
        status = data.get("status")
        if status not in {"passed", "issues_found"}:
            return data
        methodology = data.get("methodology_fidelity")
        summary = None
        if isinstance(methodology, dict):
            summary = methodology.get("summary")
        if not isinstance(summary, str) or not summary.strip():
            summary = f"Legacy review report migrated from schema_version {data.get('schema_version', 'unknown')}."
        return {
            "schema_version": data.get("schema_version", SCHEMA_VERSION),
            "review_status": status,
            "summary": summary,
            "findings": data.get("findings") or [],
        }
