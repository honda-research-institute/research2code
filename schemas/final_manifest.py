"""Pydantic model for `<RUN_DIR>/final_manifest.json`.

The final manifest is an audit index for a delivered R2C run. It records the
stage outcomes, validation evidence, delivered artifacts, and content hashes.
It is intentionally generated from main's existing artifacts and control-plane
state; it does not replace stage sentinels or `driver_state.json`.
"""

from __future__ import annotations

from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    model_serializer,
    model_validator,
)


SCHEMA_VERSION = "1.9.0"
SEMVER_PATTERN = r"^(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)$"

ManifestStatus = Literal["passed", "degraded", "blocked"]
DeliveryLabel = Literal[
    "verified", "draft", "explanation_only", "uncertified_new_territory"]
GraphEvidenceApplicability = Literal[
    "not_applicable", "homogeneous_graph_v1"]
ArtifactStatus = Literal["present", "missing", "hash_mismatch"]
ValidationStatus = Literal[
    "passed",
    "failed",
    "missing",
    "skipped",
    "degraded",
    "blocked",
    "unknown",
]
StageStatus = Literal["completed", "skipped", "halted", "degraded", "missing"]


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class StageSummary(_Strict):
    stage_id: str
    stage_label: str
    stage_category: str | None = None
    status: StageStatus
    notes: str = ""
    started_at: str | None = None
    completed_at: str | None = None
    last_event_sequence: int | None = None


class ArtifactRecord(_Strict):
    path: str = Field(description="Run-directory-relative artifact path.")
    kind: str
    display_name: str
    producer_stage: str
    producer_stage_label: str
    required_for_delivery: bool
    status: ArtifactStatus
    validation_status: ValidationStatus
    sha256: str | None = None
    byte_size: int | None = None


class ValidationRecord(_Strict):
    stage_id: str | None = None
    stage_label: str | None = None
    validator: str
    status: ValidationStatus
    summary: str = ""
    artifacts: list[str] = Field(default_factory=list)
    event_sequence: int | None = None


class ManifestReferences(_Strict):
    report_md: str | None = None
    method_spec: str | None = None
    assumptions_md: str | None = None
    deferred_findings_md: str | None = None
    known_issues_md: str | None = None
    probe_report: str | None = None
    claims_ledger: str | None = None
    review_report: str | None = None
    run_events: str | None = None
    driver_state: str | None = None
    final_manifest: str = "final_manifest.json"


class DeliveryReason(_Strict):
    """One demoting reason or non-demoting disclosure behind the label."""

    source: str
    id: str
    message: str
    verdict: str | None = None
    # What the check itself reported before label derivation re-classified
    # the entry (e.g. a passing contribution check demoted to
    # `verdict="binding_gap"` keeps `observed_verdict="pass"`). Written by
    # delivery_label.py's binding-gap path since 12a890616.
    observed_verdict: str | None = None
    severity: str | None = None
    # File/cell/line trace from the probe verdict, when it carried one —
    # the proximity-to-correct summary groups demoters by this.
    evidence: str | None = None


class StubbedElement(_Strict):
    """One component the pipeline could not build, shipped as an explicit
    self-identifying stub with a researcher-consumable work order (partial
    delivery design §3.1/§3.3). Structured on purpose: the digest and the
    recurrence counters aggregate these without string parsing."""

    element_id: str = Field(
        description="Methodology-contract element id the stub stands in for.")
    role: Literal["core", "supporting"] = Field(
        description="Contract ranking: `core` (core_methodology) dominates "
                    "the surfaces; everything else is `supporting`.")
    work_order: str = Field(
        description="Run-directory-relative path to the work-order markdown.")
    stub_path: str | None = Field(
        default=None,
        description="Run-directory-relative path to the stub module, when "
                    "the stub is a code file.")


class DemoEvidenceReason(_Strict):
    code: str
    message: str


class ExecutionEvidenceAxis(_Strict):
    status: Literal["completed", "failed", "undetermined"]
    reasons: list[DemoEvidenceReason] = Field(default_factory=list)


class EvaluationValidityAxis(_Strict):
    status: Literal["valid", "invalid", "unresolved", "not_applicable"]
    reasons: list[DemoEvidenceReason] = Field(default_factory=list)


class MechanismEvidenceAxis(_Strict):
    status: Literal[
        "demonstrated", "not_demonstrated", "undetermined", "not_applicable"
    ]
    reasons: list[DemoEvidenceReason] = Field(default_factory=list)


class SkillEvidenceAxis(_Strict):
    status: Literal[
        "demonstrated", "not_demonstrated", "undetermined", "not_applicable"
    ]
    reasons: list[DemoEvidenceReason] = Field(default_factory=list)


class PaperBenchmarkEvidenceAxis(_Strict):
    status: Literal[
        "reproduced", "not_reproduced", "not_assessed", "undetermined"
    ]
    reasons: list[DemoEvidenceReason] = Field(default_factory=list)


class DemoEvidenceStatus(_Strict):
    """Orthogonal evidence axes introduced by R2C-086 and R2C-088.

    The graph-only axes are omitted from serialization when absent so a
    graph-free evidence payload keeps its pre-v1.8 byte shape. They reuse the
    mechanism status grammar because each asks whether one graph obligation
    was demonstrated, while keeping construction, alignment, and contribution
    separate.
    """

    execution: ExecutionEvidenceAxis
    evaluation_validity: EvaluationValidityAxis
    mechanism: MechanismEvidenceAxis
    skill: SkillEvidenceAxis
    paper_benchmark: PaperBenchmarkEvidenceAxis
    graph_construction: MechanismEvidenceAxis | None = None
    graph_alignment: MechanismEvidenceAxis | None = None
    contribution: MechanismEvidenceAxis | None = None

    @model_serializer(mode="wrap")
    def _omit_absent_graph_axes(self, handler):
        """Preserve the graph-free serialized shape on Pydantic 2.6+."""
        payload = handler(self)
        for field_name in (
            "graph_construction", "graph_alignment", "contribution",
        ):
            if getattr(self, field_name) is None:
                payload.pop(field_name, None)
        return payload


class DemoVerdictRecord(_Strict):
    """Compact record of the post-smoke demo-success verdict (1.7.0):
    did the headline demonstration demonstrably work, per the family's
    taxonomy-declared markers. `failed` always coexists with a
    `demo_verdict`-sourced demoting reason (the derivation guarantees it,
    the validator below re-checks) — the demo failure demotes through its
    OWN reason, never through adjacent-demoter luck."""

    schema_version: str | None = None
    verdict: Literal["succeeded", "failed", "undetermined"]
    evidence_line: str | None = None
    evidence_cell: int | None = None
    decided_by: str | None = None
    reason: str | None = None
    evidence_status: DemoEvidenceStatus | None = None


class DeliveryVerdict(_Strict):
    """The three-state delivery decision: `draft` on any demoting reason;
    `verified` needs zero demoters AND contribution evidence (a passing
    contribution-tier probe); zero demoters WITHOUT contribution evidence
    is `uncertified_new_territory` (the false-verified hole, closed). A
    verified label still carries its disclosures (warn/unprobeable).
    Readers stay tolerant of pre-1.1.0 two-value verdicts — old manifests
    remain valid.

    Partial delivery (1.2.0): `partial` is declared FIRST so the word
    PARTIAL lands in the first line of the rendered delivery block (§3.5
    criterion 1), and `stubbed_elements` is the structured record no
    rewording can lose (§3.5 criterion 5). Nothing partial can read
    `verified` — the derivation guarantees it, the validator re-checks."""

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "allOf": [
                {
                    "if": {
                        "properties": {
                            "schema_version": {
                                "pattern": (
                                    r"^(?:1\.(?:[7-9]|[1-9][0-9]+)\."
                                    r"[0-9]+|(?:[2-9]|[1-9][0-9]+)\."
                                    r"[0-9]+\.[0-9]+)$"
                                ),
                            },
                        },
                        "required": ["schema_version"],
                    },
                    "then": {
                        "required": ["graph_evidence_applicability"],
                    },
                },
                {
                    "if": {
                        "properties": {
                            "graph_evidence_applicability": {
                                "const": "homogeneous_graph_v1",
                            },
                        },
                        "required": ["graph_evidence_applicability"],
                    },
                    "then": {
                        "required": ["demo_verdict"],
                        "properties": {
                            "demo_verdict": {
                                "type": "object",
                                "required": [
                                    "schema_version", "evidence_status",
                                ],
                                "properties": {
                                    "schema_version": {
                                        "type": "string",
                                        "pattern": r"^2\.",
                                    },
                                    "evidence_status": {
                                        "type": "object",
                                        "required": [
                                            "graph_construction",
                                            "graph_alignment",
                                            "contribution",
                                        ],
                                    },
                                },
                            },
                        },
                    },
                },
                {
                    "if": {
                        "properties": {
                            "graph_evidence_applicability": {
                                "const": "homogeneous_graph_v1",
                            },
                            "label": {"const": "verified"},
                        },
                        "required": [
                            "graph_evidence_applicability", "label",
                        ],
                    },
                    "then": {
                        "properties": {
                            "demo_verdict": {
                                "properties": {
                                    "evidence_status": {
                                        "properties": {
                                            axis: {
                                                "type": "object",
                                                "properties": {
                                                    "status": {
                                                        "const": "demonstrated",
                                                    },
                                                },
                                                "required": ["status"],
                                            }
                                            for axis in (
                                                "graph_construction",
                                                "graph_alignment",
                                                "mechanism",
                                                "contribution",
                                            )
                                        },
                                    },
                                },
                            },
                        },
                    },
                },
            ],
        },
    )

    # Declaration order is load-bearing: `partial` must render first.
    partial: bool = False
    schema_version: str = Field(default="1.1.0", pattern=SEMVER_PATTERN)
    label: DeliveryLabel
    reasons: list[DeliveryReason] = Field(default_factory=list)
    disclosures: list[DeliveryReason] = Field(default_factory=list)
    probe_counts: dict[str, int] = Field(default_factory=dict)
    stubbed_elements: list[StubbedElement] = Field(default_factory=list)
    # Required on delivery schema 1.7+. This discriminator lets a standalone
    # final manifest distinguish graph-free evidence from a graph-bearing
    # record whose graph axes were removed or never projected.
    graph_evidence_applicability: GraphEvidenceApplicability | None = Field(
        default=None,
        description=(
            "Required on delivery schema 1.7+. This discriminator lets a "
            "standalone final manifest distinguish graph-free evidence from a "
            "graph-bearing record whose graph axes were removed or never "
            "projected."
        ),
    )
    # Set only on uncertified_new_territory: the taxonomy node id that
    # lacked bindable contribution probes (the growth-engine demand key).
    missing_probe_family: str | None = None
    # Post-smoke demo-success verdict (1.6.0). Absent on runs where the
    # verdict pass never ran (pre-feature manifests stay valid).
    demo_verdict: DemoVerdictRecord | None = None

    @model_validator(mode="after")
    def _nothing_partial_reads_verified(self) -> "DeliveryVerdict":
        version = tuple(int(part) for part in self.schema_version.split("."))
        if version >= (1, 7, 0) and self.graph_evidence_applicability is None:
            raise ValueError(
                "delivery schema 1.7+ requires explicit graph evidence "
                "applicability"
            )
        if (self.stubbed_elements or self.partial) and self.label == "verified":
            raise ValueError(
                "a delivery with stubbed elements can never be verified "
                "(partial delivery design §3.5)")
        if self.stubbed_elements and not self.partial:
            raise ValueError(
                "stubbed_elements present but partial flag unset — the "
                "partial state must be unmistakable (§3.5)")
        if self.label == "verified" and any(
            r.source in ("partial_delivery", "approximated_core")
            for r in self.reasons
        ):
            raise ValueError(
                "a completeness cause (partial_delivery / approximated_core) "
                "can never coexist with a verified label")
        if (self.demo_verdict is not None
                and self.demo_verdict.verdict == "failed"
                and self.label == "verified"):
            raise ValueError(
                "a failed demo verdict can never coexist with a verified "
                "label — the headline demonstration visibly failed in its "
                "own output (demo-success semantics, 2026-07-16)")
        if (self.demo_verdict is not None
                and self.demo_verdict.evidence_status is not None
                and self.label == "verified"
                and (
                    self.demo_verdict.evidence_status.evaluation_validity.status
                    == "invalid"
                    or self.demo_verdict.evidence_status.skill.status
                    == "not_demonstrated"
                )):
            raise ValueError(
                "invalid demo evaluation or not-demonstrated task skill can "
                "never coexist with a verified label (R2C-086)")
        evidence = (
            self.demo_verdict.evidence_status
            if self.demo_verdict is not None else None
        )
        graph_axes = (
            evidence.graph_alignment,
            evidence.graph_construction,
            evidence.contribution,
        ) if evidence is not None else (None, None, None)
        graph_bearing = (
            self.graph_evidence_applicability == "homogeneous_graph_v1"
        )
        if graph_bearing:
            if (
                self.demo_verdict is None
                or not str(self.demo_verdict.schema_version or "").startswith("2.")
                or evidence is None
                or any(axis is None for axis in graph_axes)
            ):
                raise ValueError(
                    "a homogeneous-graph delivery requires one complete schema-2 "
                    "record with separate alignment, construction, mechanism, "
                    "contribution, skill, and paper-benchmark axes"
                )
        elif (
            self.graph_evidence_applicability == "not_applicable"
            and any(axis is not None for axis in graph_axes)
        ):
            raise ValueError(
                "a graph-not-applicable delivery cannot carry graph evidence axes"
            )
        if self.label == "verified" and (
            graph_bearing or any(axis is not None for axis in graph_axes)
        ):
            if (
                any(
                    axis is None or axis.status != "demonstrated"
                    for axis in graph_axes
                )
                or evidence.mechanism.status != "demonstrated"
            ):
                raise ValueError(
                    "a graph-bearing verified delivery requires demonstrated "
                    "alignment, construction, mechanism, and contribution axes "
                    "(R2C-088)"
                )
        return self


class FinalManifest(_Strict):
    schema_version: str = SCHEMA_VERSION
    run_id: str
    paper_slug: str
    generated_at: str
    git_commit: str
    config_hash: str
    run_status: ManifestStatus
    stage_summary: list[StageSummary]
    artifacts: list[ArtifactRecord]
    validation_summary: list[ValidationRecord]
    references: ManifestReferences
    diagnostics: list[str] = Field(default_factory=list)
    # Optional for manifests before the delivery-label slice; the driver
    # always supplies it now.
    delivery: DeliveryVerdict | None = None
