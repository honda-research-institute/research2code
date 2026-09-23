"""
Pydantic model for the canonical method spec produced by Stage 1 (Step 3).

Source of truth for what `method_spec.json` must contain. Every downstream pipeline
stage (calibration, generation, comparison, summary) consumes this artifact via this
model. Any field added or changed here is a schema change; bump SCHEMA_VERSION and
ensure all consumers are updated.

DESIGN DECISIONS — settled 2026-05-01, see docs/phases/stage-1-paper-to-spec.md
"Resolved decisions" section for full rationale.

  Q1. Pydantic, with JSON Schema exported via `model_json_schema()` for non-Python
      consumers. Export lives at schemas/method_spec.schema.json.
  Q2. Paradigm detection is a sub-step of Step 3 (the analyzer). The spec carries
      `comparison.classification` as a structured block with taxonomy-node
      detection metadata; if no taxonomy node matches, the analyzer halts before
      writing this file.
  Q3. The pluggable component records `seed_param` as a separate field (the named
      parameter that consumes the seed). The `signature` string is human-readable
      reference; `seed_param` is the machine-checkable contract.
  Q4. Paradigms are keyed by the taxonomy `classification.id` exposed from
      docs/ssot/taxonomies.yaml. The old `field_guide_path` carrier was retired
      in schema v1.5.0.
  Q5. `standard_baselines[].source` records provenance ("paper",
      "field_guide_required", "field_guide_recommended"). The validator script
      cross-checks every field-guide `required_baselines` entry against this list,
      matched by `family` not by name.
  Q6. `comparison.classification.id` is a path-form string
      ("active_learning/batch_acquisition"). Not an enum in v1.
  Q7. `paper.authors` is a free string. Authors don't drive pipeline logic.
  Q8. Structured fields are authoritative; prose fields (descriptions, summaries)
      are human-readable annotations only. Validation never cross-checks prose.
  Q9. Scale-dependent hyperparameters get their own structured field
      (`critical_requirements.scale_dependent_hyperparameters`) — a surgical
      fix for the silent-misconfiguration failure mode (paper's R_0=2000 used
      against normalized data) without restructuring how all method-specific
      hyperparameters are represented. Non-scale-dependent hyperparameters
      continue to live in prose. Settled 2026-05-05; see docs/phases/
      stage-3-method-to-comparison.md "Stage 3 prerequisite — Item G".
  Q10. Required model methods beyond `forward()` get their own structured
       field (`critical_requirements.required_model_methods`). Same shape of
       fix as Q9: papers that demand non-`forward()` hooks (penultimate-
       embedding access, MC-dropout-aware prediction, etc.) frequently
       satisfy the *interface* (a method is defined with that name) without
       satisfying the *substance* (the body actually returns what the
       algorithm needs). The structured field lets the Stage 4 fidelity
       reviewer iterate hook-by-hook instead of relying on free-form
       judgment to remember the check. Settled 2026-05-15; first concrete
       case: GBALD's `forward_with_embedding` for geometric ranking in
       feature space.
  Q11. Methodology replication gets its own optional strict contract
       (`methodology_replication_contract`, `methodology_contract_pack`, and
       `replication_feasibility`). This contract records which paper
       mechanisms must be replicated, which demo-scale approximations are
       approved, and which core methodology blockers halt Stage 1 before
       generation.
  Q12. `try_it_out` provide entries carry an optional `symbol` — the exact
       importable Python identifier the promise resolves to (schema v1.7.0,
       the Stage 1 naming bridge; see the naming bridge design note,
       internal, not shipped). Prose `name`/`description` stay
       human-readable annotations (Q8); `symbol` is the machine-checkable
       contract the Stage 2.b validator resolves to a public top-level name
       in the generated package. Optional and additive: legacy specs stay
       valid, and an entry without a symbol gets no bridge enforcement
       (conservative by construction — omission is legal, never guessed).
  Q13. Declared symbols carry an optional `symbol_kind` (`class` or
       `function`, schema v1.8.0) — the KIND of surface the promise
       resolves to, not just its name. Two concrete cases shaped this
       (plan-of-record item 10): the detr KD roll promised a function that
       shipped under a different name (same kind, producer-fixable one
       alias away), while ADAM promised class-shaped optimizer symbols for
       a method honestly delivered as a pure-function API (wrong kind — no
       rename can fix it, and the 2.d halt correctly classified it
       upstream, but only after full generation). With the kind declared,
       Stage 2.b checks the resolved definition's AST kind and stops
       deferring class promises to function-only ownership buckets, so the
       ADAM shape fails at 2.b with judge routing instead of a late 2.d
       hard halt. Optional and additive like Q12: `symbol_kind` without
       `symbol` is rejected, omission is legal, and an entry without a
       kind gets name-only enforcement (today's behavior, byte-identical).
  Q14. Motion-planning papers may carry `scenario_assumptions` (schema
       v1.9.0): a dimension-keyed capture of paper-stated demo constraints.
       Each entry preserves a normalized value, a verbatim paper quote, and
       its paper location. The matched taxonomy node owns the legal dimension
       ids, and strict validation checks that join. The field is optional and
       additive so non-declaring families and legacy specs remain unchanged.
       This is the visibility-only slice: detector bindings and delivery
       demotion belong to the later scenario-fidelity enforcement slice.
  Q15. Temporal evaluation facts use a role-typed protocol block (schema
       v1.10.0). Forecast-call horizon, context length, validation span, and
       test span each carry their own unit, evidence, and explicit
       paper-stated versus paper-unspecified status. Numeric agreement never
       authorizes a cross-role join, and heuristic prose scans cannot populate
       paper truth downstream.
  Q16. Paper-grounded homogeneous graph methods carry a relational-structure
       marker (schema v1.11.0). The marker activates a conditional architecture
       contract for stable entity identity, graph endpoint index spaces, and
       jointly transformed node-aligned roots. Unsupported relation forms keep
       their concrete kind instead of being coerced into homogeneous semantics.
  Q17. Methodology obligations may name the exact taxonomy probes that verify
       them (`verification_probe_refs`, schema v1.12.0). The list is optional
       for archived-spec compatibility, while fresh strict validation checks
       each nonblank unique ref against the effective taxonomy node's closed
       `semantic_checks[].probe` set. Exact refs supplement paper grounding;
       they are not inferred from prose, element names, or shared callables.
  Q18. Scale-dependent hyperparameters carry a typed calibration context
       (schema v1.13.0) instead of an open-ended scale string. Feature
       magnitude, representation convention, and evidence-preserving `other`
       contexts are strict discriminated arms. Explicit v1.13+ specs must use
       the typed carrier; archived specs and specs that omitted their version
       may retain the legacy `assumes_data_scale` carrier, but one entry may
       never contain both carriers or neither carrier.
  Q19. Homogeneous graph methods carry one typed graph-mechanism block (schema
       v1.14.0). It binds exact construction, alignment, message-passing, and
       paper-justified control elements to a closed similarity-graph grammar
       and exact probe refs. Relational identity, representation, batching,
       degree semantics, and output order remain solely in the R2C-084
       architecture contract. Archived graph specs remain readable without
       the block; explicit v1.14+ graph specs must emit it.
"""

from enum import Enum
import keyword
import re
from typing import Annotated, Literal, Optional

from pydantic import (BaseModel, ConfigDict, Field, StrictInt, field_validator,
                      model_validator)

# The graph-mechanism wiring derivation lives in the dependency-free
# schemas/graph_wiring.py so raw-dict consumers (probe-spec join, runtime
# plans, the arch-contract validator, vendored harness code) share the one
# implementation without importing pydantic (R2C-092).
from schemas.graph_wiring import (
    ALIGNMENT_PROBE_REF as HOMOGENEOUS_GRAPH_ALIGNMENT_PROBE_REF,
    derived_graph_probe_refs,
    mechanism_marker_element_ids,
    merge_derived_graph_refs,
)


SCHEMA_VERSION = "1.14.0"
VERIFICATION_PROBE_REFS_SCHEMA_VERSION = "1.12.0"
CALIBRATION_CONTEXT_SCHEMA_VERSION = "1.13.0"
HOMOGENEOUS_GRAPH_MECHANISM_SCHEMA_VERSION = "1.14.0"


def _schema_version_tuple(value: object) -> tuple[int, ...]:
    """Comparable numeric schema version, or empty for an unknown spelling."""
    parts = str(value or "").split(".")
    if not parts or any(not part.isdigit() for part in parts):
        return ()
    parsed = tuple(int(part) for part in parts)
    return parsed + (0,) * max(0, 3 - len(parsed))


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class MethodType(str, Enum):
    """Classification of what kind of contribution the paper makes."""

    algorithm = "algorithm"
    loss_function = "loss_function"
    training_methodology = "training_methodology"
    pipeline = "pipeline"
    architecture = "architecture"


class ProvideKind(str, Enum):
    code = "code"
    data = "data"
    model = "model"


class Severity(str, Enum):
    """How load-bearing a model feature is. `essential` = method breaks without it."""

    essential = "essential"
    important = "important"
    nice_to_have = "nice-to-have"


class BlockerStatus(str, Enum):
    must_implement = "must_implement"
    can_approximate = "can_approximate"
    cannot_implement = "cannot_implement"


class Feasibility(str, Enum):
    reproducible = "reproducible"
    approximate = "approximate"
    infeasible = "infeasible"


class ComputeRequirement(str, Enum):
    cpu = "CPU sufficient"
    single_gpu = "Single GPU required"
    multi_gpu = "Multi-GPU required"


class BaselineSource(str, Enum):
    """Provenance of an entry in `comparison.standard_baselines`.

    Used by the validator to confirm that field-guide-required baselines were
    not silently dropped — every required family must have at least one
    baseline whose source is `field_guide_required` (or `paper`, since a
    paper-mentioned baseline that happens to satisfy the family also counts).
    """

    paper = "paper"
    field_guide_required = "field_guide_required"
    field_guide_recommended = "field_guide_recommended"


class BaselineFamily(str, Enum):
    """Standard families used to cross-check field-guide `required_baselines`.

    Match between field guide and spec is by family, not by name — a paper
    that compares against "Entropy Sampling" (family=uncertainty) satisfies
    a field guide requiring `uncertainty_family: at_least_one_of [...]`.
    """

    random = "random"
    uncertainty = "uncertainty"
    diversity = "diversity"
    bayesian = "bayesian"
    other = "other"


class MethodologyElementRole(str, Enum):
    """What role a methodology contract element plays in paper fidelity."""

    core_methodology = "core_methodology"
    supporting_mechanism = "supporting_mechanism"
    paper_scale_detail = "paper_scale_detail"
    evaluation_control = "evaluation_control"


class ReplicationStatus(str, Enum):
    """Whether a methodology element must be exact, approximated, or blocked."""

    must_replicate = "must_replicate"
    faithful_approximation_allowed = "faithful_approximation_allowed"
    not_replicable = "not_replicable"


class RelationalStructureKind(str, Enum):
    """Whether a methodology element activates relational index contracts.

    Version one deliberately supports only one homogeneous graph.  The
    ``unsupported`` arm lets the analyzer preserve paper truth for a
    heterogeneous, bipartite, hypergraph, dynamic, or sampled-neighbor
    mechanism without pretending it fits the homogeneous contract.
    """

    homogeneous_graph = "homogeneous_graph"
    unsupported = "unsupported"


class ReplicationFeasibilityVerdict(str, Enum):
    """Run-level verdict derived from the methodology replication contract."""

    feasible = "feasible"
    feasible_with_approved_approximations = "feasible_with_approved_approximations"
    not_replicable = "not_replicable"


class EvaluationProtocolRole(str, Enum):
    """Scientific role of one quantity on a temporal evaluation axis."""

    context_length = "context_length"
    forecast_call_horizon = "forecast_call_horizon"
    validation_span = "validation_span"
    test_span = "test_span"


class PaperValueStatus(str, Enum):
    """Whether the paper pins a numeric value for a typed quantity."""

    paper_stated = "paper_stated"
    paper_unspecified = "paper_unspecified"


class EvaluationSchemeKind(str, Enum):
    """Chronological evaluation shapes whose span arithmetic differs."""

    single_holdout = "single_holdout"
    rolling_origin = "rolling_origin"
    expanding_window = "expanding_window"
    sliding_window = "sliding_window"
    cross_validation = "cross_validation"


# ---------------------------------------------------------------------------
# Sub-models (in the order they appear in the top-level MethodSpec)
# ---------------------------------------------------------------------------


class _Strict(BaseModel):
    """Base for all spec sub-models. Forbids unknown fields so typos and schema
    drift are caught at validation time rather than silently ignored."""

    model_config = ConfigDict(extra="forbid")


class RelationalStructure(_Strict):
    """Paper-grounded relational shape carried by one methodology element."""

    kind: RelationalStructureKind
    unsupported_kind: str | None = Field(
        default=None,
        min_length=1,
        description=(
            "Concrete relational form when kind='unsupported' (for example, "
            "'heterogeneous_graph'). Null for the supported homogeneous_graph "
            "shape. This is a pipeline-coverage declaration, not permission "
            "to approximate the paper as a homogeneous graph."
        ),
    )

    @model_validator(mode="after")
    def _unsupported_kind_matches_discriminator(self) -> "RelationalStructure":
        if self.kind == RelationalStructureKind.unsupported:
            if not (self.unsupported_kind or "").strip():
                raise ValueError(
                    "relational_structure kind='unsupported' requires a "
                    "non-empty unsupported_kind"
                )
            self.unsupported_kind = self.unsupported_kind.strip()
        elif self.unsupported_kind is not None:
            raise ValueError(
                "relational_structure unsupported_kind must be null when "
                "kind='homogeneous_graph'"
            )
        return self


_PYTHON_IDENTIFIER_PATTERN = r"^[A-Za-z_][A-Za-z0-9_]*$"


def _exact_nonblank(value: str, *, field_name: str) -> str:
    if not value or value != value.strip():
        raise ValueError(f"{field_name} must be non-blank and already stripped")
    return value


class GraphCallableReference(_Strict):
    """Exact generated-package callable, without importing generated code."""

    module: Literal["method.model", "method.training", "method.method"]
    qualname: str = Field(
        min_length=1,
        pattern=r"^[A-Za-z][A-Za-z0-9_]*$",
        json_schema_extra={"not": {"enum": keyword.kwlist}},
        description=(
            "Exact public top-level helper function within the declared "
            "generated-package module. Dotted class or instance methods are "
            "outside v1 because the probe plan carries no model-construction "
            "authority."
        ),
    )

    @field_validator("qualname")
    @classmethod
    def _qualname_is_public_identifier(cls, value: str) -> str:
        if keyword.iskeyword(value) or value.startswith("_"):
            raise ValueError(
                "graph callable qualname must name one public, non-keyword "
                "top-level helper"
            )
        return value


class GraphParameterBinding(_Strict):
    """One exact params entry bound to one exact callable parameter."""

    params_name: str = Field(
        min_length=1,
        pattern=r"^[A-Za-z_][A-Za-z0-9_]*$",
        description="Exact key in .pipeline/params.json.",
    )
    callable_parameter: str = Field(
        min_length=1,
        pattern=r"^[A-Za-z_][A-Za-z0-9_]*$",
        description="Exact keyword parameter on the graph-construction callable.",
    )

    @field_validator("callable_parameter")
    @classmethod
    def _callable_parameter_is_not_keyword(cls, value: str) -> str:
        if keyword.iskeyword(value):
            raise ValueError("graph callable parameter cannot be a Python keyword")
        return value


class ReturnValueGraphOutput(_Strict):
    kind: Literal["return_value"]


class TupleItemGraphOutput(_Strict):
    kind: Literal["tuple_item"]
    index: StrictInt = Field(ge=0)


class MappingItemGraphOutput(_Strict):
    kind: Literal["mapping_item"]
    key: str = Field(min_length=1)

    @field_validator("key")
    @classmethod
    def _key_is_exact(cls, value: str) -> str:
        return _exact_nonblank(value, field_name="graph output mapping key")


GraphOutputSelector = Annotated[
    ReturnValueGraphOutput | TupleItemGraphOutput | MappingItemGraphOutput,
    Field(discriminator="kind"),
]


class SimilarityThresholdContract(_Strict):
    metric: Literal["cosine_similarity"]
    comparison: Literal["greater_than", "greater_than_or_equal"]
    parameter: GraphParameterBinding


class NoNeighborhoodCap(_Strict):
    kind: Literal["none"]


class PerSourceTopSimilarityCap(_Strict):
    """Closed v1 per-source cap with deterministic mutual undirected selection.

    Candidates rank by descending cosine similarity with canonical target-index
    tie-breaks.  For ``undirected_bidirectional`` construction, a non-self pair
    survives only when both endpoints select each other; both directions are
    then emitted.  Self loops are applied after this cap and do not consume it.
    """

    kind: Literal["per_source_top_similarity"]
    parameter: GraphParameterBinding


NeighborhoodCapContract = Annotated[
    NoNeighborhoodCap | PerSourceTopSimilarityCap,
    Field(discriminator="kind"),
]


class HomogeneousGraphConstructionContract(_Strict):
    """Closed v1 construction grammar for a paper-grounded similarity graph."""

    element_id: str = Field(min_length=1)
    callable: GraphCallableReference
    feature_input_root: str = Field(
        min_length=1,
        description=(
            "Exact R2C-084 coindexed logical root supplied to graph construction."
        ),
    )
    feature_parameter: str = Field(
        min_length=1,
        pattern=r"^[A-Za-z_][A-Za-z0-9_]*$",
    )
    output_selector: GraphOutputSelector
    threshold: SimilarityThresholdContract
    cap: NeighborhoodCapContract
    self_loop_policy: Literal["required", "forbidden"]
    direction_policy: Literal["directed", "undirected_bidirectional"]

    @field_validator("element_id", "feature_input_root")
    @classmethod
    def _references_are_exact(cls, value: str, info) -> str:
        return _exact_nonblank(value, field_name=info.field_name)

    @field_validator("feature_parameter")
    @classmethod
    def _feature_parameter_is_not_keyword(cls, value: str) -> str:
        if keyword.iskeyword(value):
            raise ValueError("graph feature parameter cannot be a Python keyword")
        return value

    @model_validator(mode="after")
    def _parameter_roles_are_distinct(self) -> "HomogeneousGraphConstructionContract":
        if (
            isinstance(self.cap, PerSourceTopSimilarityCap)
            and self.cap.parameter.params_name == self.threshold.parameter.params_name
        ):
            raise ValueError(
                "homogeneous graph threshold and neighborhood cap must bind "
                "different params entries"
            )
        callable_parameters = {
            "feature": self.feature_parameter,
            "threshold": self.threshold.parameter.callable_parameter,
        }
        if isinstance(self.cap, PerSourceTopSimilarityCap):
            callable_parameters["cap"] = self.cap.parameter.callable_parameter
        if len(set(callable_parameters.values())) != len(callable_parameters):
            raise ValueError(
                "homogeneous graph feature, threshold, and cap roles must "
                "bind pairwise-distinct callable parameters"
            )
        return self


class HomogeneousGraphMessagePassingContract(_Strict):
    """Exact graph-consuming callable and isolated neighbor-signal seam."""

    element_id: str = Field(min_length=1)
    callable: GraphCallableReference
    graph_parameter: str = Field(
        min_length=1,
        pattern=r"^[A-Za-z_][A-Za-z0-9_]*$",
    )
    neighbor_signal_root: str = Field(
        min_length=1,
        description="Exact R2C-084 coindexed root changed by the neighbor probe.",
    )
    neighbor_signal_parameter: str = Field(
        min_length=1,
        pattern=r"^[A-Za-z_][A-Za-z0-9_]*$",
    )
    output_root: str = Field(
        min_length=1,
        description=(
            "Exact R2C-084 coindexed logical root returned by this probeable "
            "message-passing helper. It may differ from the final inference "
            "output root."
        ),
    )

    @field_validator("element_id", "neighbor_signal_root", "output_root")
    @classmethod
    def _references_are_exact(cls, value: str, info) -> str:
        return _exact_nonblank(value, field_name=info.field_name)

    @field_validator("graph_parameter", "neighbor_signal_parameter")
    @classmethod
    def _input_parameter_is_not_keyword(cls, value: str) -> str:
        if keyword.iskeyword(value):
            raise ValueError("graph message parameter cannot be a Python keyword")
        return value

    @model_validator(mode="after")
    def _input_parameter_roles_are_distinct(
        self,
    ) -> "HomogeneousGraphMessagePassingContract":
        if self.graph_parameter == self.neighbor_signal_parameter:
            raise ValueError(
                "graph and neighbor-signal roles must bind distinct callable "
                "parameters"
            )
        return self


class GraphInputContributionAblation(_Strict):
    """A paper-justified graph-input control executed by the real callable."""

    kind: Literal["empty_graph", "identity_graph", "permuted_graph"]
    element_id: str = Field(min_length=1)
    discriminating_probe_ref: Literal[
        "graph_mechanism.topology_sensitivity",
        "graph_mechanism.neighbor_sensitivity",
    ]

    @field_validator("element_id")
    @classmethod
    def _element_id_is_exact(cls, value: str) -> str:
        return _exact_nonblank(value, field_name="ablation element_id")


class CallableContributionAblation(_Strict):
    """A paper-justified removed-message or non-graph callable control."""

    kind: Literal["removed_message_passing", "non_graph_decoder"]
    element_id: str = Field(min_length=1)
    callable: GraphCallableReference
    graph_parameter: str = Field(
        min_length=1,
        pattern=r"^[A-Za-z_][A-Za-z0-9_]*$",
        description=(
            "Exact keyword on the null callable receiving the same graph "
            "intervention as the real message-passing arm."
        ),
    )
    neighbor_signal_parameter: str = Field(
        min_length=1,
        pattern=r"^[A-Za-z_][A-Za-z0-9_]*$",
        description=(
            "Exact keyword on the null callable receiving the same neighbor "
            "signal fixture as the real arm."
        ),
    )
    output_root: str = Field(
        min_length=1,
        description=(
            "Exact R2C-084 coindexed output root returned by the null arm; "
            "it must equal the real message helper output root."
        ),
    )
    discriminating_probe_ref: Literal[
        "graph_mechanism.topology_sensitivity",
        "graph_mechanism.neighbor_sensitivity",
    ]

    @field_validator("element_id", "output_root")
    @classmethod
    def _reference_is_exact(cls, value: str, info) -> str:
        return _exact_nonblank(value, field_name=f"ablation {info.field_name}")

    @field_validator("graph_parameter", "neighbor_signal_parameter")
    @classmethod
    def _input_parameter_is_not_keyword(cls, value: str) -> str:
        if keyword.iskeyword(value):
            raise ValueError("graph ablation parameter cannot be a Python keyword")
        return value

    @model_validator(mode="after")
    def _input_parameter_roles_are_distinct(
        self,
    ) -> "CallableContributionAblation":
        if self.graph_parameter == self.neighbor_signal_parameter:
            raise ValueError(
                "callable contribution-ablation graph and neighbor-signal "
                "roles must bind distinct parameters"
            )
        return self


GraphContributionAblation = Annotated[
    GraphInputContributionAblation | CallableContributionAblation,
    Field(discriminator="kind"),
]


class HomogeneousGraphProbeRefs(_Strict):
    """Exact graph evidence obligations registered in the taxonomy."""

    parameter_agreement: Literal["graph_mechanism.parameter_agreement"]
    construction: Literal["graph_mechanism.construction_semantics"]
    topology: Literal["graph_mechanism.topology_sensitivity"]
    neighbor_signal: Literal["graph_mechanism.neighbor_sensitivity"]
    permutation: Literal["graph_mechanism.permutation_equivalence"] | None
    contribution_ablation: (
        Literal["graph_mechanism.contribution_ablation"] | None
    )


class HomogeneousGraphMechanismContract(_Strict):
    """Paper-grounded mechanism semantics layered on R2C-084 identity."""

    schema_version: Literal["1.0"] = "1.0"
    alignment_element_id: str = Field(
        min_length=1,
        description=(
            "Exact methodology element grounding the reused R2C-084 relational "
            "alignment prerequisite."
        ),
    )
    construction: HomogeneousGraphConstructionContract
    message_passing: HomogeneousGraphMessagePassingContract
    permutation_applicability: Literal["equivariant", "not_applicable"]
    contribution_ablation: GraphContributionAblation | None
    probe_refs: HomogeneousGraphProbeRefs

    @field_validator("alignment_element_id")
    @classmethod
    def _alignment_id_is_exact(cls, value: str) -> str:
        return _exact_nonblank(value, field_name="alignment_element_id")

    @model_validator(mode="after")
    def _permutation_ref_matches_applicability(self) -> "HomogeneousGraphMechanismContract":
        if (
            self.permutation_applicability == "equivariant"
            and self.probe_refs.permutation is None
        ):
            raise ValueError(
                "permutation_applicability='equivariant' requires "
                "graph_mechanism.permutation_equivalence"
            )
        if (
            self.permutation_applicability == "not_applicable"
            and self.probe_refs.permutation is not None
        ):
            raise ValueError(
                "permutation_applicability='not_applicable' requires a null "
                "permutation probe ref"
            )
        if (
            self.contribution_ablation is None
            and self.probe_refs.contribution_ablation is not None
        ):
            raise ValueError(
                "a null contribution_ablation requires a null contribution "
                "probe ref"
            )
        if (
            self.contribution_ablation is not None
            and self.probe_refs.contribution_ablation is None
        ):
            raise ValueError(
                "a declared contribution_ablation requires "
                "graph_mechanism.contribution_ablation"
            )
        callable_roles = {
            "construction": self.construction.callable,
            "message_passing": self.message_passing.callable,
        }
        if isinstance(self.contribution_ablation, CallableContributionAblation):
            callable_roles["contribution_ablation"] = (
                self.contribution_ablation.callable
            )
            if (
                self.contribution_ablation.output_root
                != self.message_passing.output_root
            ):
                raise ValueError(
                    "callable contribution ablation must return the same "
                    "coindexed output root as message passing"
                )
        identities = [
            (reference.module, reference.qualname)
            for reference in callable_roles.values()
        ]
        if len(identities) != len(set(identities)):
            raise ValueError(
                "homogeneous graph construction, message passing, and "
                "callable contribution ablation must use distinct callable "
                "identities"
            )
        return self


class Paper(_Strict):
    title: str
    authors: str = Field(
        description="Free-form author list. Affiliations may be embedded in parentheses."
    )
    repo_url: Optional[str] = None


class CoreMethod(_Strict):
    name: str
    summary: str = Field(description="One-sentence description of what the method does.")
    type: MethodType
    paper_sections: list[str]
    key_elements: list[str] = Field(
        description="paper_map element IDs that describe the core method."
    )


class PaperClaims(_Strict):
    method_description: str = Field(
        description="2-3 sentence researcher-facing description of the method's mechanism (not its results)."
    )
    claimed_results: str = Field(
        description="2-3 sentences on what the paper reports — baselines beaten, datasets, where the advantage shows."
    )
    benchmark_scale: str = Field(
        description="Paper's main benchmark in concrete numbers — used to contextualize demo-scale results."
    )


class Provide(_Strict):
    name: str
    description: str
    type: ProvideKind
    symbol: Optional[str] = Field(
        default=None,
        description=(
            "The exact importable Python identifier this promise resolves to "
            "in the generated package (e.g., \"AvoidanceClassifier\"). Set it "
            "when a code/model entry promises ONE concrete importable "
            "artifact; omit it when the promise is not a single importable "
            "name (a dataset, a behavior). Never guessed — an entry without "
            "a symbol simply gets no naming-bridge enforcement. Stage 2.b "
            "resolves every declared symbol to a public top-level name in "
            "the generated package (defined public in a module, or "
            "re-exported by method/__init__.py)."
        ),
        # The exported JSON Schema carries the syntactic rule for non-Python
        # consumers. Deliberately json_schema_extra rather than pydantic's
        # `pattern=`, so the model validator below stays the single
        # enforcement point (it also rejects keywords, which no regex can)
        # with its actionable error text.
        json_schema_extra={"pattern": "^[A-Za-z][A-Za-z0-9_]*$"},
    )
    symbol_kind: Optional[Literal["class", "function"]] = Field(
        default=None,
        description=(
            "The KIND of surface the declared symbol resolves to: \"class\" "
            "when the promise is a class the researcher instantiates, "
            "\"function\" when it is a callable they invoke. Set it whenever "
            "you set `symbol` and the delivered surface's kind is knowable "
            "from the build shape; derive it from how the method is actually "
            "delivered, never from what the concept is called in the paper "
            "(an optimizer delivered as a pure-function API is a function "
            "promise, whatever the paper names it). Requires `symbol`; "
            "omission is legal and means name-only bridge enforcement."
        ),
    )

    @model_validator(mode="after")
    def _symbol_is_public_identifier(self) -> "Provide":
        if self.symbol is None:
            if self.symbol_kind is not None:
                raise ValueError(
                    f"try_it_out provide entry {self.name!r}: symbol_kind "
                    f"{self.symbol_kind!r} is declared without a symbol. The "
                    "kind qualifies a declared importable name; declare the "
                    "symbol too, or omit both."
                )
            return self
        if not self.symbol.isidentifier() or keyword.iskeyword(self.symbol):
            raise ValueError(
                f"try_it_out provide entry {self.name!r}: symbol "
                f"{self.symbol!r} is not a valid Python identifier. The "
                "symbol must be the exact importable name the promise "
                "resolves to (one identifier, no dots, no spaces); omit the "
                "field when the promise is not one importable name."
            )
        if self.symbol.startswith("_"):
            raise ValueError(
                f"try_it_out provide entry {self.name!r}: symbol "
                f"{self.symbol!r} starts with an underscore. Promised "
                "components are user-facing by construction, so the symbol "
                "must be a public name (no leading underscore)."
            )
        return self


class TryItOut(_Strict):
    definition: str
    user_provides: list[Provide]
    system_provides: list[Provide]


class DataRequirements(_Strict):
    format: str
    example_datasets: list[str]
    auto_download_feasible: bool
    auto_download_details: Optional[str] = None
    synthetic_feasible: bool
    synthetic_description: Optional[str] = None


class PretrainedModel(_Strict):
    name: str
    trained_on: str
    auto_download: bool
    url: Optional[str] = None
    size_mb: Optional[float] = None
    license: str = "unknown"


class Dependencies(_Strict):
    frameworks: list[str]
    pretrained_models: list[PretrainedModel] = Field(default_factory=list)
    compute: ComputeRequirement
    estimated_time: str


class RepoFile(_Strict):
    path: str
    role: str


class Repo(_Strict):
    url: Optional[str] = None
    cloned: bool = False
    clone_path: Optional[str] = None
    key_files: list[RepoFile] = Field(default_factory=list)


class Paradigm(_Strict):
    """Result of the paradigm-detection sub-step inside the analyzer.

    The analyzer resolves the paper to a taxonomy node and records the node's
    legacy-compatible id plus detection reasoning. If no taxonomy node matches,
    the analyzer halts and writes a paradigm-gap report instead of emitting
    method_spec.json.
    """

    id: str = Field(
        description=(
            'Taxonomy-backed path-form identifier. Examples: "active_learning", '
            '"active_learning/batch_acquisition".'
        )
    )
    detection_reasoning: str = Field(
        description=(
            "One or two sentences explaining why the analyzer classified the paper "
            "into this paradigm. Auditable trail for sniff-test review."
        )
    )


def normalized_seed_param(value: object) -> object:
    """Fold the string sentinels of "no seed" into a real null.

    `seed_param` was a required string until 2026-07-29, so a producer
    describing a DETERMINISTIC component satisfied the schema with the
    literal string "None" — and the method-coder validator then demanded
    a parameter literally named None of a function that cannot have one
    (DomIndOnto knowledge-base population, stage 2c). "None", "null",
    and "" cannot be real Python parameter names, so folding them is
    lossless. Exported because the stage validators and the build plan
    read the spec ARTIFACT as raw JSON without parsing it through
    MethodSpec; they must fold the same sentinels the schema does."""
    if isinstance(value, str) and value.strip().lower() in (
            "none", "null", ""):
        return None
    return value


class PluggableComponent(_Strict):
    """The minimal piece of code that represents the method's novel contribution.

    The signature MUST include a seed parameter so the caller (notebook §5.1
    acquisition loop in v2) threads a per-round seed through to the method
    for reproducibility. The `seed_param` field is the machine-checkable name
    of that parameter; the `signature` string is the human-readable form.
    """

    name: str = Field(description='Function/class name (e.g., "select_batch").')
    signature: str = Field(
        description=(
            "Full signature for human reference, e.g., "
            '"select_batch(model, x_unlabeled, batch_size, seed) -> List[int]".'
        )
    )
    seed_param: str | None = Field(
        default=None,
        description=(
            "Named parameter that consumes the random seed. The verifier "
            "confirms the seed gets threaded through to this parameter from "
            'config. Example: "seed". null when the component is '
            "deterministic and takes no seed (a rule-based knowledge-base "
            "population pipeline has nothing to seed)."
        ),
    )
    description: str

    @field_validator("seed_param", mode="before")
    @classmethod
    def _seed_param_none_sentinel(cls, v: object) -> object:
        return normalized_seed_param(v)


class Baseline(_Strict):
    name: str
    family: BaselineFamily = Field(
        description=(
            "Used to cross-check against field-guide `required_baselines`. "
            "Match is by family, not by name."
        )
    )
    source: BaselineSource = Field(
        description=(
            "Where this baseline came from: paper-mentioned, field-guide-required, "
            "or field-guide-recommended."
        )
    )
    description: str
    implementation: str = Field(
        description="Short prose or snippet describing how to implement the baseline."
    )


class Comparison(_Strict):
    # Renamed from `paradigm` -> `classification` (Phase 2.7, 2026-06-23): the
    # spec references its taxonomy node by `classification.id`. The legacy
    # `field_guide_path` carrier was removed in Phase 4.3.
    classification: Paradigm
    description: str = Field(
        description="Plain-English: what a fair comparison looks like for this method."
    )
    pluggable_component: PluggableComponent
    controlled_variables: dict[str, str] = Field(
        description=(
            "Variables held constant across methods. Key = variable name; "
            "value = why it must be held constant."
        )
    )
    independent_variable: str
    evaluation_checkpoints: str
    evaluation_protocol: Optional["EvaluationProtocol"] = Field(
        default=None,
        description=(
            "Role-typed temporal evaluation facts. Required by strict "
            "validation for time_series_forecasting; omitted for families "
            "without a temporal forecast/evaluation axis. This is the "
            "authority for context, one-call horizon, validation span, test "
            "span, units, and whether the paper actually pins each value."
        ),
    )
    primary_metric: str
    visualization: str
    # standard_baselines is a v1 (comparison-driver) artifact. v2 produces
    # a single-method package + notebook with no baseline implementations,
    # so the analyzer SHOULD omit this field. Kept Optional/empty-default
    # so legacy specs still parse without error.
    standard_baselines: list[Baseline] = Field(default_factory=list)


class ModelFeature(_Strict):
    feature: str
    why_essential: str
    severity: Severity
    paper_section: str


class ModelRequirements(_Strict):
    architecture: str
    paper_section: str
    specific_features: list[ModelFeature]


class TrainingRequirements(_Strict):
    optimizer: Optional[str] = Field(
        default=None,
        description=(
            "Optimizer the paper trains with. MUST be null for paradigms "
            "with no training phase (motion_planning, classical "
            "optimization, etc.) — do NOT write an 'n/a' prose workaround; "
            "null is the semantically correct value (pdwa 2026-07-28: the "
            "analyzer honestly wrote null here and the then-required str "
            "halted a paper that had delivered the day before on invented "
            "'n/a — algorithmic method' filler)."
        ),
    )
    learning_rate: Optional[str] = Field(
        default=None,
        description=(
            "Free string — papers often report multiple LRs (e.g., per-dataset). "
            "The calibrator extracts a numeric value from this prose. "
            "MUST be null for paradigms with no training phase, same rule "
            "as `optimizer`; the calibrator treats null as absent."
        ),
    )
    protocol: Optional[str] = Field(
        default=None,
        description=(
            "Retrain from scratch | warm start | fine-tune | etc. "
            "MUST be null for paradigms with no training phase, same rule "
            "as `optimizer`."
        ),
    )
    mc_samples: Optional[int] = Field(
        default=None,
        description=(
            "Number of MC dropout forward passes the paper specifies for "
            "Bayesian uncertainty estimation. Required (non-null) for "
            "bayesian-AL papers that use MC dropout (e.g., BALD, BatchBALD, "
            "GBALD). MUST be null for paradigms that don't use MC dropout "
            "(non-bayesian AL, motion_planning, optimization, etc.) — do "
            "NOT write 0 as a 'not applicable' workaround; null is the "
            "semantically correct value, and derive_params.py falls back to "
            "the signature default when this field is null."
        ),
    )
    num_epochs: Optional[int] = Field(
        default=None,
        description=(
            "Number of training epochs the paper trains for. Required for "
            "fixed-epoch paradigms (knowledge_distillation, supervised "
            "classification, etc.); null for paradigms whose training count "
            "is expressed differently (e.g., active learning uses "
            "`data_setup.num_rounds` instead). If the paper expresses epochs "
            "via a schedule label (e.g., '2x schedule = 24 epochs'), record "
            "the integer count, not the schedule label."
        ),
    )
    seed: Optional[int] = Field(
        default=None,
        description=(
            "Random seed reported by the paper, if any. Null if the paper does "
            "not specify a single seed (e.g., reports mean over multiple seeds — "
            "in which case the harness chooses seeds and records them per-run)."
        ),
    )
    paper_section: str


class DataSetup(_Strict):
    """Values from the paper's MAIN benchmark — not toy demos or ablations.

    The four numeric fields (`initial_labeled`, `batch_size`, `total_budget`,
    `num_rounds`) are active-learning paradigm concepts: an iterative
    label-acquisition loop where each round draws `batch_size` labels from a
    pool until `total_budget` is exhausted, starting from `initial_labeled`
    bootstrap labels over `num_rounds` total. Paradigms that don't have this
    structure (motion_planning, optimization, RL, etc.) genuinely have no
    values here — for those paradigms, all four fields MUST be null.

    `batch_returns` is an optional fifth field: the two-stage preselection
    count b for selectors that prefilter a candidate pool before ranking down
    to `batch_size` (b'). It is null for single-stage selectors and non-AL
    paradigms, so it is not part of the four-field non-null requirement.

    For AL paradigms, the four fields MUST come from a single benchmark
    configuration. Mixing values across benchmarks (e.g., initial_labeled
    from MNIST + total_budget from CIFAR-10) is forbidden — it produces
    internally-incoherent specs that fail arithmetic sanity checks
    (total_budget should approximate initial_labeled + num_rounds * batch_size).

    The cross-paradigm null/non-null requirement is enforced by a
    `MethodSpec`-level model validator (paradigm-aware) rather than here,
    because this model can't see `comparison.classification.id`.
    """

    initial_labeled: Optional[int] = Field(
        default=None,
        description=(
            "Initial labeled pool size at AL bootstrap. Required (non-null) "
            "for active-learning paradigms; MUST be null for non-AL paradigms."
        ),
    )
    batch_size: Optional[int] = Field(
        default=None,
        description=(
            "Labels acquired per AL round. Required (non-null) for "
            "active-learning paradigms; MUST be null for non-AL paradigms."
        ),
    )
    total_budget: Optional[int] = Field(
        default=None,
        description=(
            "Total label budget across all AL rounds (≈ initial_labeled + "
            "num_rounds * batch_size). Required (non-null) for active-learning "
            "paradigms; MUST be null for non-AL paradigms."
        ),
    )
    num_rounds: Optional[int] = Field(
        default=None,
        description=(
            "Number of AL acquisition rounds. Required (non-null) for "
            "active-learning paradigms; MUST be null for non-AL paradigms."
        ),
    )
    batch_returns: Optional[int] = Field(
        default=None,
        description=(
            "Two-stage selectors only: the paper's first-stage preselection "
            "count b — the candidate pool scored/ranked before the final batch "
            "is chosen (e.g. GBALD's 'rank b=300, select b'=100'). The paper "
            "invariant is b >= batch_size (b >= b'), so this must be >= "
            "batch_size when present. Null for single-stage selectors (e.g. "
            "BADGE) that return batch_size directly with no preselection, and "
            "for non-AL paradigms. This is the paper-truth value; the method "
            "signature's batch_returns default is a runtime convenience, not "
            "paper truth."
        ),
    )
    benchmark_name: Optional[str] = Field(
        default=None,
        description=(
            "Identifier of the benchmark configuration these values came from "
            '(e.g., "MNIST main benchmark", "SVHN, batch_size=100"). Required '
            "when the paper reports multiple benchmarks; downstream stages use "
            "this to disambiguate which configuration was captured. May be a "
            "non-AL identifier for non-AL paradigms (e.g., "
            '"Structured + randomized obstacle scenarios" for motion planning).'
        ),
    )
    per_dataset_values: Optional[dict[str, dict[str, int]]] = Field(
        default=None,
        description=(
            "When the paper states DIFFERENT literal values for the same "
            "data_setup parameter per dataset (GBALD §7.4: initial labeled "
            'sets of "20, 1000, 1000 random samples" for MNIST/SVHN/CIFAR-10), '
            "map parameter name -> {dataset name -> value} instead of "
            "silently picking one. The scalar field keeps the PRIMARY "
            "evaluation dataset's entry so existing consumers work "
            "unchanged; the parameter deriver (stage 2.x) binds by the "
            "dataset the demo actually uses and records `bound_dataset` in "
            "params.json. Keys must be the data_setup field names "
            "(initial_labeled, batch_size, total_budget, num_rounds, "
            "batch_returns). A scalar that contradicts every entry in its "
            "map is a validation error. This models only paper-stated "
            "literals — dataset-conditional formulas and scale rules stay "
            "with the scale-calibration layer."
        ),
    )
    special_protocol: Optional[str] = Field(
        default=None,
        description='Free-text — e.g., "rank 300 candidates, select 100".',
    )
    paper_section: str

    _PER_DATASET_BINDABLE_FIELDS = (
        "initial_labeled",
        "batch_size",
        "total_budget",
        "num_rounds",
        "batch_returns",
    )

    @model_validator(mode="after")
    def _per_dataset_values_coherent(self) -> "DataSetup":
        if not self.per_dataset_values:
            return self
        for name, dataset_map in self.per_dataset_values.items():
            if name not in self._PER_DATASET_BINDABLE_FIELDS:
                raise ValueError(
                    f"per_dataset_values names unknown parameter {name!r}; "
                    f"keys must be one of {list(self._PER_DATASET_BINDABLE_FIELDS)}"
                )
            if not dataset_map:
                raise ValueError(
                    f"per_dataset_values[{name!r}] is empty — name the "
                    "datasets and their paper values, or omit the entry"
                )
            scalar = getattr(self, name)
            if scalar is not None and scalar not in dataset_map.values():
                raise ValueError(
                    f"data_setup.{name}={scalar} contradicts "
                    f"per_dataset_values[{name!r}]={dataset_map}: the scalar "
                    "must be the map entry for the paper's primary "
                    "evaluation dataset"
                )
        return self


class Blocker(_Strict):
    requirement: str
    status: BlockerStatus
    reason: str
    resolution: str


class MethodologyContractElement(_Strict):
    """One paper-grounded methodology obligation.

    Elements are more specific than the older high-level feasibility string.
    Downstream producers and reviewers consume these entries as explicit
    identity constraints for the generated demo-scale method.
    """

    element_id: str = Field(
        min_length=1,
        description="Stable kebab-case ID unique within the methodology contract.",
    )
    role: MethodologyElementRole
    replication_status: ReplicationStatus
    paper_section: str = Field(
        min_length=1,
        description="Section, algorithm, figure, or equation where the paper grounds this element.",
    )
    paper_evidence: str = Field(
        min_length=1,
        description="Short paper-grounded evidence for why this element belongs in the contract.",
    )
    technical_concept: str = Field(
        min_length=1,
        description="Short noun phrase naming the method mechanism or control.",
    )
    required_behavior: str = Field(
        min_length=1,
        description="Runtime behavior the implementation must preserve.",
    )
    demo_scale_implementation: str = Field(
        min_length=1,
        description="How the behavior can be implemented in the smoke/demo-scale artifact.",
    )
    acceptable_approximations: list[str] = Field(
        description="Approved demo-scale approximations. Empty when exact replication is required.",
    )
    forbidden_substitutions: list[str] = Field(
        description="Concrete substitutions that would break methodology identity.",
    )
    required_controls: list[str] = Field(
        description="Controls that keep the demo evaluation or comparison fair.",
    )
    fairness_checks: list[str] = Field(
        description="Checks that ensure only allowed variables differ.",
    )
    feasibility_rationale: str = Field(
        min_length=1,
        description="Why this element is replicable, approximable, or blocked.",
    )
    verification_expectations: list[str] = Field(
        description="Expected downstream code/review checks for this element.",
    )
    verification_probe_refs: list[str] = Field(
        default_factory=list,
        json_schema_extra={
            "items": {"type": "string", "minLength": 1, "pattern": r"\S"},
            "uniqueItems": True,
        },
        description=(
            "Exact taxonomy semantic_checks[].probe refs that genuinely verify "
            "this obligation. Refs are nonblank and unique; strict spec "
            "validation checks them against the matched effective taxonomy "
            "node. Optional for archived specs, where an empty list means no "
            "probe-to-obligation binding is declared."
        ),
    )
    blockers: list[str] = Field(
        description="Reasons this element cannot be replicated. Empty unless status is not_replicable.",
    )
    paper_element_ids: list[str] = Field(
        default_factory=list,
        description=(
            "IDs of the paper_map elements this contract element was built "
            "from. The machine-readable form of the grounding paper_section "
            "and paper_evidence already state in prose. Generated code is "
            "annotated with paper_map IDs, so this is the ONLY link that lets "
            "a behavioral finding about a piece of code reach the contract "
            "obligation it bears on (R2C-072). Every ID is cross-checked "
            "against paper_map.json's closed ID set at spec validation. "
            "Optional for backward compatibility with specs written before "
            "the field existed; an empty list means findings on this element "
            "cannot be adjudicated by the contract."
        ),
    )
    relational_structure: RelationalStructure | None = Field(
        default=None,
        description=(
            "Typed activation for a paper-grounded relational mechanism. "
            "Use kind='homogeneous_graph' only for a single homogeneous "
            "graph. Use kind='unsupported' plus unsupported_kind for other "
            "relational forms so downstream validators report a pipeline "
            "coverage gap instead of inferring graph behavior from names or "
            "prose. Null for non-relational elements."
        ),
    )

    @field_validator("verification_probe_refs")
    @classmethod
    def _verification_probe_refs_are_unique_and_nonblank(
        cls, refs: list[str]
    ) -> list[str]:
        normalized = [ref.strip() for ref in refs]
        if any(not ref for ref in normalized) or len(set(normalized)) != len(
            normalized
        ):
            raise ValueError(
                "verification_probe_refs must be non-blank and unique"
            )
        return normalized

    @model_validator(mode="after")
    def _status_requires_supporting_detail(self) -> "MethodologyContractElement":
        if self.relational_structure is not None and not self.paper_element_ids:
            raise ValueError(
                f"methodology element {self.element_id!r} declares "
                "relational_structure but has no paper_element_ids; the graph "
                "activation must be grounded in the paper map"
            )

        if self.role == MethodologyElementRole.core_methodology:
            for field_name in (
                "forbidden_substitutions",
                "required_controls",
                "fairness_checks",
                "verification_expectations",
            ):
                if not getattr(self, field_name):
                    raise ValueError(
                        f"core methodology element {self.element_id!r} requires "
                        f"non-empty {field_name}"
                    )

        if (
            self.replication_status
            == ReplicationStatus.faithful_approximation_allowed
            and not self.acceptable_approximations
        ):
            raise ValueError(
                f"methodology element {self.element_id!r} is marked "
                "faithful_approximation_allowed but has no acceptable_approximations"
            )

        if (
            self.acceptable_approximations
            and self.replication_status
            != ReplicationStatus.faithful_approximation_allowed
        ):
            raise ValueError(
                f"methodology element {self.element_id!r} declares "
                "acceptable_approximations but replication_status is "
                f"{self.replication_status.value!r}; use "
                "'faithful_approximation_allowed' for approved demo-scale "
                "approximations, or clear acceptable_approximations for exact "
                "requirements"
            )

        if (
            self.replication_status == ReplicationStatus.not_replicable
            and not self.blockers
        ):
            raise ValueError(
                f"methodology element {self.element_id!r} is marked not_replicable "
                "but has no blockers"
            )

        return self


class MethodologyReplicationContract(_Strict):
    schema_version: Literal["1.0"] = "1.0"
    elements: list[MethodologyContractElement] = Field(min_length=1)
    homogeneous_graph_mechanism: HomogeneousGraphMechanismContract | None = Field(
        default=None,
        description=(
            "Typed construction, liveness, permutation, and paper-justified "
            "ablation contract for the one supported homogeneous graph. Null "
            "for graph-free and unsupported relational methods, and optional "
            "when reading archived pre-v1.14 method specs. Relational identity "
            "and representation remain in arch_contract.relational_indexing."
        ),
    )

    def core_elements(self) -> list[MethodologyContractElement]:
        return [
            element
            for element in self.elements
            if element.role == MethodologyElementRole.core_methodology
        ]

    @model_validator(mode="after")
    def _require_unique_ids_and_core_element(self) -> "MethodologyReplicationContract":
        element_ids = [element.element_id for element in self.elements]
        duplicates = sorted({element_id for element_id in element_ids if element_ids.count(element_id) > 1})
        if duplicates:
            raise ValueError(
                "methodology_replication_contract.elements contains duplicate "
                f"element_id value(s): {duplicates}"
            )
        if not self.core_elements():
            raise ValueError(
                "methodology_replication_contract must include at least one "
                "role='core_methodology' element"
            )

        mechanism = self.homogeneous_graph_mechanism
        if mechanism is None:
            return self

        # R2C-092: the block's element ids fully determine the relational
        # markers and graph probe-ref ownership, so both are DERIVED here
        # (via schemas/graph_wiring, the surface every raw-dict consumer
        # shares) instead of graded as hand-transcription.  Five 2026-08-10/11
        # live attempts showed a producer relocating the marker and dropping
        # refs while fixing adjacent errors, one revelation per dispatch.
        # What stays validated is producer-owned: which elements the block
        # binds, the ablation element's role, paper grounding, and a marker
        # that contradicts the block.  Every residual violation reports in
        # ONE message (the SRL 2026-07-28 all-floors-report-together rule).
        by_id = {element.element_id: element for element in self.elements}
        issues: list[str] = []
        owned_ids = {
            "alignment_element_id": mechanism.alignment_element_id,
            "construction.element_id": mechanism.construction.element_id,
            "message_passing.element_id": mechanism.message_passing.element_id,
        }
        if mechanism.contribution_ablation is not None:
            owned_ids["contribution_ablation.element_id"] = (
                mechanism.contribution_ablation.element_id
            )
        missing_ids = sorted(
            {
                element_id
                for element_id in owned_ids.values()
                if element_id not in by_id
            }
        )
        if missing_ids:
            issues.append(
                "homogeneous_graph_mechanism references unknown methodology "
                f"element ID(s): {missing_ids}"
            )

        if mechanism.contribution_ablation is not None:
            ablation_element = by_id.get(
                mechanism.contribution_ablation.element_id
            )
            if (
                ablation_element is not None
                and ablation_element.role
                != MethodologyElementRole.evaluation_control
            ):
                issues.append(
                    "homogeneous_graph_mechanism.contribution_ablation."
                    f"element_id references {ablation_element.element_id!r}, "
                    "whose role must be 'evaluation_control'"
                )

        bound_marker_ids = [
            element_id
            for element_id in mechanism_marker_element_ids(mechanism)
            if element_id in by_id
        ]
        for element_id in bound_marker_ids:
            element = by_id[element_id]
            marker = element.relational_structure
            if marker is None:
                if not element.paper_element_ids:
                    issues.append(
                        f"methodology element {element_id!r} is bound by "
                        "homogeneous_graph_mechanism but has no "
                        "paper_element_ids; the derived relational marker "
                        "must stay grounded in the paper map"
                    )
                    continue
                declared_before = "relational_structure" in element.model_fields_set
                element.relational_structure = RelationalStructure(
                    kind=RelationalStructureKind.homogeneous_graph
                )
                if not declared_before:
                    element.model_fields_set.discard("relational_structure")
            elif marker.kind != RelationalStructureKind.homogeneous_graph:
                issues.append(
                    f"methodology element {element_id!r} is bound by "
                    "homogeneous_graph_mechanism but declares "
                    f"relational_structure.kind={marker.kind.value!r}, "
                    "contradicting the block"
                )
        stray_marker_ids = [
            element.element_id
            for element in self.elements
            if (
                element.relational_structure is not None
                and element.relational_structure.kind
                == RelationalStructureKind.homogeneous_graph
                and element.element_id not in bound_marker_ids
            )
        ]
        if stray_marker_ids:
            issues.append(
                "relational_structure.kind='homogeneous_graph' is derived "
                "from homogeneous_graph_mechanism and belongs only to its "
                "alignment, construction, and message-passing elements; "
                f"remove it from {stray_marker_ids} or bind those elements "
                "in the block"
            )
        if issues:
            raise ValueError("; ".join(issues))

        derived_refs = derived_graph_probe_refs(mechanism)
        for element in self.elements:
            merged = merge_derived_graph_refs(
                element.verification_probe_refs,
                derived_refs.get(element.element_id, ()),
            )
            if merged != element.verification_probe_refs:
                declared_before = (
                    "verification_probe_refs" in element.model_fields_set
                )
                element.verification_probe_refs = merged
                # Assignment marks the field declared; the strict validator's
                # explicit-emission floor must keep seeing what the producer
                # actually wrote.
                if not declared_before:
                    element.model_fields_set.discard("verification_probe_refs")
        return self


_NUMBER_TOKEN = r"(\d[\d,]*(?:\.\d+)?(?:\s*[eE]\s*[+-]?\s*\d+)?)"

# The name-anchor vocabulary lives in the dependency-free
# schemas/param_text.py (the portable check harness vendors the params
# provenance validator, which needs it without the pydantic models here).
# Re-imported so existing consumers keep one implementation.
from schemas.param_text import (  # noqa: E402,F401
    _PARAM_ROLE_SUFFIXES,
    _PARAM_TEXT_ALIASES,
    param_text_aliases,
)


def _read_attr_or_key(obj: object, name: str) -> object | None:
    if isinstance(obj, dict):
        return obj.get(name)
    return getattr(obj, name, None)


def _coerce_numeric_token(token: str) -> int | float:
    value = float(re.sub(r"\s+", "", token.replace(",", "")))
    if value.is_integer():
        return int(value)
    return value


def _numeric_values_equal(left: object, right: object) -> bool:
    if isinstance(left, bool) or isinstance(right, bool):
        return left == right
    if not isinstance(left, (int, float)) or not isinstance(right, (int, float)):
        return left == right
    return float(left) == float(right)


def evaluation_protocol_quantity_for_param(
    spec: object, param_name: str,
) -> object | None:
    """Return the explicitly bound protocol quantity for ``param_name``.

    No name similarity or prose scan participates in this join. Stage-1
    strict validation independently checks ``parameter_name`` against the
    matched taxonomy node's ``params_derivation.protocol_role`` declaration.
    This helper merely consumes that already-validated binding downstream.
    """
    comparison = _read_attr_or_key(spec, "comparison")
    protocol = _read_attr_or_key(comparison, "evaluation_protocol")
    quantities = _read_attr_or_key(protocol, "quantities")
    if not isinstance(quantities, list):
        return None
    for quantity in quantities:
        if _read_attr_or_key(quantity, "parameter_name") == param_name:
            return quantity
    return None


def evaluation_protocol_identity_labels(quantity: object) -> set[str]:
    """Exact paper/runtime labels for conflict detection, never value lookup.

    ``paper_names`` carries prose identities while ``paper_symbols`` carries
    atomic notation.  Keeping those two producer declarations separate avoids
    guessing that an arbitrary capital letter is a parameter and also keeps
    lowercase or multi-character symbols from disappearing at downstream
    joins.
    """
    labels = {
        str(item).strip()
        for field in ("paper_names", "paper_symbols")
        for item in (_read_attr_or_key(quantity, field) or [])
        if isinstance(item, str) and item.strip()
    }
    parameter_name = _read_attr_or_key(quantity, "parameter_name")
    if isinstance(parameter_name, str) and parameter_name.strip():
        labels.add(parameter_name.strip())
    return labels


def normalize_evaluation_protocol_label(label: str) -> str:
    """Case/whitespace-insensitive identity key used only for joins."""
    return " ".join(label.casefold().replace("_", " ").split())


def evaluation_protocol_label_matches(quantity: object, label: str) -> bool:
    """Join one external label to a typed identity without inferring a role."""
    candidate = label.strip()
    if not candidate:
        return False
    symbols = {
        str(item).strip()
        for item in (_read_attr_or_key(quantity, "paper_symbols") or [])
        if isinstance(item, str) and item.strip()
    }
    if candidate in symbols:
        return True
    names = {
        str(item).strip()
        for item in (_read_attr_or_key(quantity, "paper_names") or [])
        if isinstance(item, str) and item.strip()
    }
    parameter_name = _read_attr_or_key(quantity, "parameter_name")
    if isinstance(parameter_name, str) and parameter_name.strip():
        names.add(parameter_name.strip())
    key = normalize_evaluation_protocol_label(candidate)
    return key in {normalize_evaluation_protocol_label(name) for name in names}


def _literal_spans(text: str, literal: str) -> list[tuple[int, int]]:
    escaped = re.escape(literal).replace(r"\ ", r"\s+")
    prefix = r"(?<![A-Za-z0-9_])" if literal[0].isalnum() else ""
    suffix = r"(?![A-Za-z0-9_])" if literal[-1].isalnum() else ""
    return [
        match.span()
        for match in re.finditer(prefix + escaped + suffix, text)
    ]


def _protocol_identity_pattern(quantity: object) -> str:
    names = {
        str(item).strip()
        for item in (_read_attr_or_key(quantity, "paper_names") or [])
        if isinstance(item, str) and item.strip()
    }
    parameter_name = _read_attr_or_key(quantity, "parameter_name")
    if isinstance(parameter_name, str) and parameter_name.strip():
        names.add(parameter_name.strip())
    symbols = {
        str(item).strip()
        for item in (_read_attr_or_key(quantity, "paper_symbols") or [])
        if isinstance(item, str) and item.strip()
    }
    # Prose identities are case-insensitive; mathematical symbols are exact.
    # In particular K, k, and K_pred remain distinct paper notation.
    # Every alternate carries its own word-boundary lookarounds (the
    # _literal_spans convention): a single-letter symbol like P otherwise
    # matches inside WMAPE or GraphDeepAR, and the candidate scan then
    # attaches unrelated nearby numbers as protocol claims (pdfgnn
    # 2026-08-11 halt, judge-confirmed pipeline bug).
    def _bounded(pattern: str, literal: str) -> str:
        prefix = r"(?<![A-Za-z0-9_])" if literal[0].isalnum() else ""
        suffix = r"(?![A-Za-z0-9_])" if literal[-1].isalnum() else ""
        return prefix + pattern + suffix

    parts = [
        _bounded(
            "(?i:" + re.escape(label).replace(r"\ ", r"\s+") + ")", label
        )
        for label in sorted(names, key=len, reverse=True)
    ] + [
        _bounded(re.escape(symbol), symbol)
        for symbol in sorted(symbols, key=len, reverse=True)
    ]
    return "(?:" + "|".join(parts) + ")" if parts else r"(?!)"


def _protocol_number_tokens(text: str) -> list[tuple[int | float, tuple[int, int]]]:
    values: list[tuple[int | float, tuple[int, int]]] = []
    for match in re.finditer(_NUMBER_TOKEN, text):
        # Mathematical indexing such as T+1/T+K is a boundary, never a
        # paper-stated value for the forecast-call horizon.
        prefix = text[max(0, match.start() - 4):match.start()]
        if re.search(r"[A-Za-z]\s*\+\s*$", prefix):
            continue
        values.append((_coerce_numeric_token(match.group(1)), match.span(1)))
    return values


def evaluation_protocol_bound_values(
    quantity: object, text: str | None = None,
) -> list[int | float]:
    """Strict, role-compatible numeric assertions in one evidence passage.

    This is deliberately narrower than free-text extraction.  A number is
    authority only when an explicit paper name/symbol is grammatically bound
    to it, or when validation/test roles and their paired values are ordered
    explicitly.  Merely being the only number near role prose is insufficient.
    """
    evidence = text if text is not None else _read_attr_or_key(
        quantity, "evidence_quote"
    )
    if not isinstance(evidence, str):
        return []
    role = _read_attr_or_key(quantity, "role")
    if isinstance(role, EvaluationProtocolRole):
        role = role.value
    identity = _protocol_identity_pattern(quantity)
    number = r"\d[\d,]*(?:\.\d+)?(?:\s*[eE]\s*[+-]?\s*\d+)?"
    temporal_unit = (
        r"(?:seconds?|minutes?|hours?|days?|weeks?|months?|quarters?|years?|"
        r"time[ -]?steps?|points?|samples?)"
    )
    role_pattern = {
        "context_length": (
            r"(?:context(?:\s+(?:length|window))?|lookback|history\s+window)"
        ),
        "forecast_call_horizon": (
            r"(?:(?:forecast(?:[ -]call)?|prediction)\s+horizon|"
            r"future\s+(?:time\s+)?steps?)"
        ),
        "validation_span": (
            r"(?:validation(?:\s+(?:set|span|window|length))?)"
        ),
        "test_span": (
            r"(?:(?:test|holdout|held-out)"
            r"(?:\s+(?:set|span|window|length))?)"
        ),
    }.get(str(role), r"(?!)")
    direct_patterns = (
        rf"(?<![A-Za-z0-9_]){identity}(?![A-Za-z0-9_])"
        rf"\s*(?i:\||=|:|is(?:\s+(?:fixed\s+to|set\s+to))?|"
        rf"set\s+to|of)\s*(?P<value>{number})",
        rf"(?i:\b(?:length|size|span)\s+of\s+(?:the\s+)?)"
        rf"{identity}\b[^.\n]{{0,40}}?"
        rf"(?i:\b(?:is\s+)?(?:fixed\s+)?(?:set\s+)?to\s+)"
        rf"(?P<value>{number})",
        # Common compact declarations used in captions and summaries.
        rf"(?<![A-Za-z0-9_]){identity}(?![A-Za-z0-9_])\s*\(\s*"
        rf"(?P<value>{number})(?:\s*{temporal_unit})?\s*\)",
        rf"(?P<value>{number})\s*[- ]\s*{temporal_unit}\s+"
        rf"(?i:{role_pattern})[^.\n]{{0,20}}?"
        rf"(?<![A-Za-z0-9_]){identity}(?![A-Za-z0-9_])",
        rf"(?i:\b(?:predicts?|forecasts?)\s+)"
        rf"(?P<value>{number})\s+(?:future\s+)?{temporal_unit}"
        rf"[^.\n]{{0,30}}?(?<![A-Za-z0-9_]){identity}"
        rf"(?![A-Za-z0-9_])",
        rf"(?i:{role_pattern})\s*(?:of|=|:|is(?:\s+set\s+to)?)\s*"
        rf"(?P<value>{number})(?:\s*{temporal_unit})?"
        rf"[^.\n]{{0,24}}?(?<![A-Za-z0-9_]){identity}"
        rf"(?![A-Za-z0-9_])",
    )
    values: list[int | float] = []
    for pattern in direct_patterns:
        for match in re.finditer(pattern, evidence):
            value = _coerce_numeric_token(match.group("value"))
            if not any(_numeric_values_equal(value, item) for item in values):
                values.append(value)

    subject_pattern = re.compile(
        rf"(?<![A-Za-z0-9_]){identity}(?![A-Za-z0-9_])"
        rf"(?P<bridge>[^.\n]{{0,48}}?)(?i:\b(?:covers?|spans?|lasts?)\s+"
        rf"(?:the\s+)?(?:last\s+)?)(?P<value>{number})"
    )
    competing_terms = {
        "context_length": ("forecast", "horizon", "future", "validation", "test", "holdout"),
        "forecast_call_horizon": ("context", "lookback", "history", "validation", "test", "holdout"),
        "validation_span": ("context", "forecast", "horizon", "future", "test", "holdout"),
        "test_span": ("context", "forecast", "horizon", "future", "validation"),
    }.get(str(role), ())
    for match in subject_pattern.finditer(evidence):
        bridge = match.group("bridge").casefold()
        if any(re.search(rf"\b{re.escape(term)}\b", bridge) for term in competing_terms):
            continue
        value = _coerce_numeric_token(match.group("value"))
        if not any(_numeric_values_equal(value, item) for item in values):
            values.append(value)

    validation = r"validation(?:\s+(?:set|span|window))?"
    test = r"(?:test(?:\s+(?:set|span|window))?|holdout|held-out)"
    for first_role, first_pattern, second_role, second_pattern in (
        ("validation_span", validation, "test_span", test),
        ("test_span", test, "validation_span", validation),
    ):
        pair_pattern = re.compile(
            rf"\b{first_pattern}\b\s*(?:,|and)\s*"
            rf"\b{second_pattern}\b[^.\n]{{0,120}}?"
            rf"(?P<first>{number})\s*(?:,|and)\s*"
            rf"(?P<second>{number})"
            rf"(?:[^.\n]{{0,40}}?respectively)?",
            flags=re.IGNORECASE,
        )
        match = pair_pattern.search(evidence)
        if match and role in {first_role, second_role}:
            key = "first" if role == first_role else "second"
            value = _coerce_numeric_token(match.group(key))
            if not any(_numeric_values_equal(value, item) for item in values):
                values.append(value)
            break
    return values


def evaluation_protocol_candidate_values(
    quantity: object, text: str,
) -> list[int | float]:
    """Broad possible claims for fail-closed prose conflict detection.

    Unlike ``evaluation_protocol_bound_values``, these candidates never grant
    paper authority.  They let notebook/legacy checks halt on a long or wrapped
    sentence that combines a protocol identity with a numeric assertion.
    """
    identity_pattern = _protocol_identity_pattern(quantity)
    identity_matches = list(re.finditer(identity_pattern, text))
    strict = evaluation_protocol_bound_values(quantity, text)
    values = list(strict)
    role = _read_attr_or_key(quantity, "role")
    if isinstance(role, EvaluationProtocolRole):
        role = role.value
    number = r"\d[\d,]*(?:\.\d+)?"
    temporal_unit = (
        r"(?:seconds?|minutes?|hours?|days?|weeks?|months?|quarters?|years?|"
        r"time[ -]?steps?|points?|samples?)"
    )
    role_pattern = {
        "context_length": (
            r"(?:context(?:\s+(?:length|window))?|lookback|history\s+window)"
        ),
        "forecast_call_horizon": (
            r"(?:(?:forecast(?:[ -]call)?|prediction)\s+horizon|"
            r"one[ -]call\s+horizon|future\s+(?:time\s+)?steps?)"
        ),
        "validation_span": (
            r"(?:validation(?:\s+(?:set|span|window|length))?)"
        ),
        "test_span": (
            r"(?:(?:test|holdout|held-out)"
            r"(?:\s+(?:set|span|window|length))?)"
        ),
    }.get(str(role), r"(?!)")
    role_claim_patterns = (
        rf"(?i:{role_pattern})\s*(?:=|:|of|is(?:\s+set\s+to)?)\s*"
        rf"(?P<value>{number})(?:\s*{temporal_unit})?",
        rf"(?P<value>{number})\s*[- ]\s*{temporal_unit}\s+"
        rf"(?i:{role_pattern})",
    )
    if role == "forecast_call_horizon":
        role_claim_patterns += (
            rf"(?i:\b(?:predicts?|forecasts?)\s+)"
            rf"(?P<value>{number})\s+(?:future\s+)?{temporal_unit}"
            r"[^.\n]{0,24}\bper\s+(?:call|forecast)\b",
        )
    for pattern in role_claim_patterns:
        for match in re.finditer(pattern, text):
            value = _coerce_numeric_token(match.group("value"))
            if not any(_numeric_values_equal(value, item) for item in values):
                values.append(value)
    if not identity_matches:
        return values
    competing_terms = {
        "context_length": (
            "forecast", "horizon", "future", "validation", "test", "holdout",
        ),
        "forecast_call_horizon": (
            "context", "lookback", "history", "validation", "test", "holdout",
        ),
        "validation_span": (
            "context", "forecast", "horizon", "future", "test", "holdout",
        ),
        "test_span": (
            "context", "forecast", "horizon", "future", "validation",
        ),
    }.get(str(role), ())
    assertion_cue = re.compile(
        r"[=:()]|\b(?:is|are|of|at|to|equals?|uses?|sets?|configured|"
        r"chosen|fixed|limited|predicts?|forecasts?)\b",
        flags=re.IGNORECASE,
    )
    for identity_match in identity_matches:
        # An identity's claim scope ends once a captured number is followed
        # by a closing parenthesis, comma boundary, or semicolon before the
        # next number: in "(context length = 10), and maximum number of
        # neighbors (10 for retail, 5 for e-commerce)" the 5 belongs to the
        # neighbors clause, not to context length (pdfgnn Attempt 4 halt,
        # 2026-08-10). Compound claims without such a boundary ("10 in
        # retail and 12 in e-commerce") still capture every value.
        claim_cursor: int | None = None
        for value, number_span in _protocol_number_tokens(text):
            if number_span[0] <= identity_match.end():
                continue
            if claim_cursor is not None and re.search(
                r"[),;]", text[claim_cursor:number_span[0]]
            ):
                break
            bridge = text[identity_match.end():number_span[0]]
            if len(bridge) > 180 or not assertion_cue.search(bridge):
                continue
            folded_bridge = bridge.casefold()
            if any(
                re.search(rf"\b{re.escape(term)}\b", folded_bridge)
                for term in competing_terms
            ):
                continue
            claim_cursor = number_span[1]
            if not any(_numeric_values_equal(value, item) for item in values):
                values.append(value)
    return values


def _mentions_param(text: str, param_name: str) -> bool:
    normalized = text.lower().replace("_", " ")
    return any(alias.lower().replace("_", " ") in normalized
               for alias in param_text_aliases(param_name))


def _param_alias_regex(param_name: str) -> str:
    parts = []
    for alias in param_text_aliases(param_name):
        escaped = re.escape(alias).replace(r"\ ", r"\s+").replace("_", r"[_\s]")
        parts.append(escaped)
    return "(?:" + "|".join(parts) + ")"


def _extract_param_paper_values_from_text(
    text: str, param_name: str, *, allow_bare_assignment: bool = True,
    require_param_anchor: bool = False,
    other_param_names: tuple[str, ...] = (),
) -> list[int | float]:
    """Extract explicit paper values from contract prose for one param.

    This intentionally recognizes only high-signal phrasings such as
    "paper uses 2000" or "reducing mc_samples from 2000 to 100". It does not
    treat arbitrary numbers in demo prose as paper truth.

    ``allow_bare_assignment`` governs the bare ``param = N`` form. That form is
    paper truth in a field that describes the paper (``paper_evidence``,
    ``required_behavior``), but in a demo-scale field a bare assignment is the
    DEMO value, not the paper value. A demo line like
    "Use mc_samples=20 (demo) vs paper 2000" must not yield 20 as paper truth;
    the paper value in those fields is always named explicitly ("from 2000 to
    20", "paper uses 2000"), which the remaining patterns still capture.

    ``require_param_anchor`` drops the param-UNANCHORED phrasings ("from N to
    M", "paper uses N", "paper's value is N"). Those are safe in a short,
    param-specific contract field where the whole text is about one param, but
    over a WHOLE document they mis-attribute any stray phrase to every param
    that happens to be mentioned: the 2026-07-01 iDb-RRT paper says "from 3 to
    14" once, and every planner knob (goal_bias, delta_0, motion_primitives)
    picked up 3 as its paper value. Callers scanning source documents (paper.md,
    paper_map) set this True so only param-anchored patterns fire.
    """
    if not _mentions_param(text, param_name):
        return []

    # Unanchored patterns assume the whole text is about this one param. That
    # assumption also fails INSIDE a short contract field when the prose names
    # two params in one clause: the GBALD 2026-09-01 halt came from "with the
    # reduced mc_samples (taxonomy floor 20, paper value 2000 recorded); score
    # ... the top batch_returns by BALD score" — scanning for batch_returns,
    # the unanchored "paper value 2000" fired and fabricated a paper claim
    # (same class as the iDb-RRT whole-document case that introduced
    # require_param_anchor). If the caller names sibling params and any of
    # them is mentioned here too, this is multi-param text: anchored only.
    if not require_param_anchor and any(
        _mentions_param(text, other)
        for other in other_param_names if other != param_name
    ):
        require_param_anchor = True

    param_pattern = _param_alias_regex(param_name)
    bare_assignment = (
        (rf"\b{param_pattern}\s*(?:=|:|is|of|(?:is\s+)?set\s+to)\s*{_NUMBER_TOKEN}",)
        if allow_bare_assignment
        else ()
    )
    # Anchored patterns tie the number to the param name; safe over any text.
    anchored_patterns = (
        *bare_assignment,
        rf"\bpaper\s+(?:uses|reports|specifies|sets)\s+{param_pattern}\s*(?:=|:)?\s*{_NUMBER_TOKEN}",
    )
    # Unanchored patterns rely on the whole text being about this one param.
    unanchored_patterns = (
        rf"\bfrom\s+{_NUMBER_TOKEN}\s+(?:to|->)\s+{_NUMBER_TOKEN}",
        rf"\bpaper\s+(?:uses|reports|specifies|sets)\s+(?:T\s*=\s*)?{_NUMBER_TOKEN}",
        rf"\bpaper(?:'s)?\s+(?:value|count)\s*(?:is|=|:)?\s*\(?{_NUMBER_TOKEN}\)?",
    )
    patterns = (
        anchored_patterns if require_param_anchor
        else (*anchored_patterns, *unanchored_patterns)
    )
    values: list[int | float] = []
    for pattern in patterns:
        for match in re.finditer(pattern, text, flags=re.IGNORECASE):
            value = _coerce_numeric_token(match.group(1))
            if value not in values:
                values.append(value)
    for value in _pair_assignment_values(text, param_pattern):
        if value not in values:
            values.append(value)
    return values


def _pair_assignment_values(text: str, param_pattern: str) -> list[int | float]:
    """Anchored paired assignment: "alpha, beta are ... set to 1.0 and 0.25".

    Papers commonly state two hyperparameters in one clause; the param's
    position among the named pair selects its value (first name takes the
    first number, second name the second). Both names and both numbers sit
    in one period-free span, so this stays name-anchored — a stray "set to
    1.0 and 0.25" elsewhere never binds (bev-distill 2026-07-02 F004: the
    loss weights were stated exactly this way and the extractor could not
    bind them)."""
    other = r"[\w\\]+"
    joiner = r"\s*(?:,\s*|\s+and\s+)"
    tail = rf"[^.\n]{{0,80}}?(?:set\s+to|are|=|:)\s*{_NUMBER_TOKEN}\s+and\s+{_NUMBER_TOKEN}"
    values: list[int | float] = []
    first = rf"\b{param_pattern}{joiner}{other}{tail}"
    for match in re.finditer(first, text, flags=re.IGNORECASE):
        values.append(_coerce_numeric_token(match.group(1)))
    second = rf"\b{other}{joiner}{param_pattern}\b{tail}"
    for match in re.finditer(second, text, flags=re.IGNORECASE):
        values.append(_coerce_numeric_token(match.group(2)))
    return [v for i, v in enumerate(values) if v not in values[:i]]


def extract_param_paper_values_from_text(
    text: str, param_name: str, *, require_param_anchor: bool = False,
) -> list[int | float]:
    """Public wrapper for deterministic consumers that scan paper-derived text.

    Consumers scanning a WHOLE source document (paper.md, paper_map) should pass
    ``require_param_anchor=True`` so an unrelated phrase elsewhere in the text is
    not attributed to this param.
    """
    return _extract_param_paper_values_from_text(
        text, param_name, require_param_anchor=require_param_anchor)


def methodology_contract_paper_values_for_param(
    contract: object | None, param_name: str, *, exact_label: bool = False,
    other_param_names: tuple[str, ...] = (),
) -> list[int | float]:
    """Return paper-truth values explicitly stated in a methodology contract.

    Accepts either the pydantic contract model or the raw JSON/dict shape so
    deterministic scripts can reuse the same extraction without validating an
    already-generated spec.
    """
    if contract is None:
        return []
    elements = _read_attr_or_key(contract, "elements")
    if not isinstance(elements, list):
        return []

    values: list[int | float] = []

    def _extract(text: str, *, allow_bare_assignment: bool) -> list[int | float]:
        if not exact_label:
            return _extract_param_paper_values_from_text(
                text,
                param_name,
                allow_bare_assignment=allow_bare_assignment,
                other_param_names=other_param_names,
            )
        number = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
        escaped = re.escape(param_name)
        boundary = (
            r"(?<![A-Za-z0-9_])" + escaped + r"(?![A-Za-z0-9_])"
        )
        bare = (
            rf"{boundary}\s*(?:=|:|is|(?:is\s+)?set\s+to)\s*"
            rf"(?P<value>{number})",
        ) if allow_bare_assignment else ()
        patterns = (
            *bare,
            rf"\bpaper\s+(?:uses|reports|specifies|sets)\s+{boundary}"
            rf"\s*(?:=|:)?\s*(?P<value>{number})",
        )
        found: list[int | float] = []
        for pattern in patterns:
            for match in re.finditer(pattern, text):
                value = _coerce_numeric_token(match.group("value"))
                if value not in found:
                    found.append(value)
        return found

    for element in elements:
        # A bare ``param = N`` is paper truth only in fields that describe the
        # PAPER. In demo-scale fields a bare assignment is the demo value, so
        # the paper value there must be named explicitly ("from 2000 to 20",
        # "paper uses 2000") — handled by the remaining patterns. Splitting the
        # scan stops a demo default like "mc_samples=20 (demo)" from being
        # mistaken for paper truth.
        paper_truth_texts: list[str] = []
        for field_name in ("paper_evidence", "required_behavior"):
            value = _read_attr_or_key(element, field_name)
            if isinstance(value, str):
                paper_truth_texts.append(value)

        demo_texts: list[str] = []
        for field_name in ("demo_scale_implementation", "feasibility_rationale"):
            value = _read_attr_or_key(element, field_name)
            if isinstance(value, str):
                demo_texts.append(value)
        for list_name in ("acceptable_approximations", "verification_expectations"):
            items = _read_attr_or_key(element, list_name)
            if isinstance(items, list):
                demo_texts.extend(item for item in items if isinstance(item, str))

        for text in paper_truth_texts:
            for value in _extract(text, allow_bare_assignment=True):
                if value not in values:
                    values.append(value)
        for text in demo_texts:
            for value in _extract(text, allow_bare_assignment=False):
                if value not in values:
                    values.append(value)
    return values


class MethodologyContractPack(_Strict):
    """Flattened downstream summary of the methodology contract."""

    schema_version: Literal["1.0"] = "1.0"
    summary: str = Field(min_length=1)
    core_methodology_element_ids: list[str] = Field(min_length=1)
    implementation_obligations: list[str]
    approved_approximations: list[str]
    forbidden_substitutions: list[str]
    required_controls: list[str]
    verification_expectations: list[str]


class ReplicationFeasibilityBlocker(_Strict):
    element_id: str = Field(min_length=1)
    technical_concept: str = Field(min_length=1)
    reason: str = Field(min_length=1)


class ReplicationFeasibilityApproximation(_Strict):
    element_id: str = Field(min_length=1)
    technical_concept: str = Field(min_length=1)
    rationale: str = Field(min_length=1)


class ReplicationFeasibility(_Strict):
    schema_version: Literal["1.0"] = "1.0"
    verdict: ReplicationFeasibilityVerdict
    blockers: list[ReplicationFeasibilityBlocker]
    approved_approximations: list[ReplicationFeasibilityApproximation]


class FeatureMagnitudeCalibrationContext(_Strict):
    """Numeric feature scale against which a paper value was calibrated."""

    kind: Literal["feature_magnitude"]
    scale: Literal[
        "raw_pixel_unnormalized",
        "pixel_zero_one",
        "pixel_centered",
        "standardized",
        "unit_norm",
    ]


class RepresentationConventionCalibrationContext(_Strict):
    """Non-magnitude representation convention used by a paper value."""

    kind: Literal["representation_convention"]
    convention: Literal["target_box_grid"]


class OtherCalibrationContext(_Strict):
    """Evidence-backed calibration context outside the supported observers."""

    kind: Literal["other"]
    label: str = Field(min_length=1)
    reason: str = Field(min_length=1)

    @field_validator("label", "reason")
    @classmethod
    def _reject_blank_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must contain non-whitespace text")
        return value


CalibrationContext = Annotated[
    FeatureMagnitudeCalibrationContext
    | RepresentationConventionCalibrationContext
    | OtherCalibrationContext,
    Field(discriminator="kind"),
]


class ScaleDependentHyperparam(_Strict):
    """A method-specific hyperparameter whose meaningful value depends on a
    declared calibration context (e.g., a feature-distance threshold or a
    target-box-grid convention).

    This block exists to prevent silent misconfiguration when the demo's data
    preprocessing differs from the paper's. The canonical example is GBALD's
    `R_0 = 2000` distance threshold — calibrated for raw [0, 255] pixel
    distances; nonsense for normalized [-1, 1] data without rescaling.

    Stage 2's generator consults this list at config-write time:

      - If `formula` is provided, the generator evaluates it in the chosen
        preset's context (with symbols `d` for input dimension, `n_classes`,
        and `data_norm` for the typical L2 norm under the preset's
        preprocessing).
      - Else, the typed `calibration_context` selects the relevant observer
        without treating every calibration convention as feature magnitude.
      - Archived specs may retain the legacy `assumes_data_scale` carrier.
        A single entry never mixes the legacy and typed carriers.

    Temporal counts are not data-scale hyperparameters: context length,
    forecast-call horizon, validation span, and test span live only in
    ``comparison.evaluation_protocol``. Other non-scale-dependent
    hyperparameters continue to live in prose (model.specific_features,
    blockers.resolution); this structured field is the surgical fix for the
    scale-dependence failure mode only. A broader hyperparameters restructure
    is deferred (see Stage 3 plan).
    """

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "oneOf": [
                {
                    "required": ["assumes_data_scale"],
                    "properties": {
                        "assumes_data_scale": {"type": "string"},
                        "calibration_context": {"type": "null"},
                    },
                },
                {
                    "required": ["calibration_context"],
                    "properties": {
                        "calibration_context": {"not": {"type": "null"}},
                        "assumes_data_scale": {"type": "null"},
                    },
                },
            ]
        },
    )

    name: str = Field(description='Hyperparameter name as it appears in the paper (e.g., "R_0", "sigma").')
    paper_value: Optional[float] = Field(
        default=None,
        description=(
            "Literal value reported in the paper, if a single number is given. "
            "Use null when the paper expresses the value only as a formula or "
            "leaves it implicit. At least one of `paper_value` or `formula` "
            "must be present."
        ),
    )
    formula: Optional[str] = Field(
        default=None,
        description=(
            "Symbolic derivation given by the paper, e.g., \"2 * sqrt(d)\". "
            "When present, the generator evaluates it instead of using "
            "`paper_value`. Available symbols: `d` (input dimension), "
            "`n_classes`, `data_norm`. Use null if the paper provides no "
            "derivation."
        ),
    )
    assumes_data_scale: Optional[str] = Field(
        default=None,
        description=(
            "Legacy pre-v1.13 carrier for the data preprocessing this "
            "hyperparameter assumes. Archived specs and specs with an omitted "
            "schema_version may retain it, but explicit v1.13+ specs must use "
            "calibration_context instead. Examples: "
            '"raw_pixel_unnormalized" (e.g., MNIST in [0, 255]), '
            '"pixel_zero_one" (e.g., [0, 1] after /255), '
            '"pixel_centered" (e.g., [-1, 1] or [-2, 2]), '
            '"unit_norm" (L2-normalized), "standardized" (zero-mean unit-variance). '
            "Never combine this field with calibration_context."
        )
    )
    calibration_context: Optional[CalibrationContext] = Field(
        default=None,
        description=(
            "Typed context against which the paper value is calibrated. "
            "feature_magnitude uses a closed numeric-scale vocabulary; "
            "representation_convention currently supports target_box_grid; "
            "other preserves an evidence-backed context that has no supported "
            "observer and must not be silently converted."
        ),
    )
    description: str = Field(
        description=(
            "What the hyperparameter does and why it is scale-dependent. "
            'Example: "Distance threshold for the geometric probability '
            'model. Calibrated for unnormalized pixel distances; needs '
            'rescaling for normalized data."'
        )
    )
    paper_section: str

    @model_validator(mode="after")
    def _require_value_or_formula(self) -> "ScaleDependentHyperparam":
        if self.paper_value is None and self.formula is None:
            raise ValueError(
                f"ScaleDependentHyperparam '{self.name}': at least one of "
                "`paper_value` or `formula` must be provided."
            )
        has_legacy_context = self.assumes_data_scale is not None
        has_typed_context = self.calibration_context is not None
        if has_legacy_context == has_typed_context:
            raise ValueError(
                f"ScaleDependentHyperparam '{self.name}': provide exactly one "
                "of legacy `assumes_data_scale` or typed "
                "`calibration_context`."
            )
        return self


class RequiredModelMethod(_Strict):
    """A method (in the Python class sense) the architecture must expose
    beyond `forward()`, because the algorithm explicitly calls it.

    Canonical examples: `forward_with_embedding(x) -> (logits, embedding)`
    for methods that rank in feature space (GBALD geometric representativeness,
    BADGE last-layer-gradient diversity); `predict_with_uncertainty(x)` for
    methods that need calibrated uncertainty estimates; `forward_features(x)`
    for any pipeline that operates on penultimate-layer activations.

    This block exists for the same reason as `scale_dependent_hyperparameters`:
    a structured catalog the Stage 4 fidelity reviewer can iterate as an
    explicit checklist. The interface alone is easy to satisfy (the
    architecture-coder can stub the method); the substance — does the
    body actually return what the algorithm needs? — is what frequently
    drifts.

    The reviewer's per-entry check is:

      - Does `method/model.py` define a method with this `name`?
      - Does its signature match `signature` (or a documented superset)?
      - Does `method/method.py` actually call it (no orphan hooks)?
      - Does the body return values consistent with `returns_description`?

    Non-required model methods (helpers, optional convenience methods,
    methods used only by training) do NOT belong here. This list is
    *strictly* the methods the algorithm body calls.
    """

    name: str = Field(
        description=(
            'Method name as it must appear on the model class. Examples: '
            '"forward_with_embedding", "predict_with_uncertainty", '
            '"forward_features".'
        )
    )
    signature: str = Field(
        description=(
            "Signature in plain ASCII, e.g., "
            '"forward_with_embedding(self, x: Tensor) -> tuple[Tensor, Tensor]". '
            "The reviewer cross-checks the implemented signature against this."
        )
    )
    returns_description: str = Field(
        description=(
            "What the return value MUST be, semantically. Example: "
            '"Tuple of (logits, penultimate_embedding) where embedding is '
            "the activation from the layer just before the classifier head, "
            'shape (B, hidden_dim)." This is what frequently drifts; '
            "the reviewer reads the body and confirms the returned object "
            "matches this description."
        )
    )
    purpose: str = Field(
        description=(
            "Why the algorithm needs this method. One sentence. Example: "
            '"Geometric representativeness ranking (Eq. 13) computes '
            "L2 distances in feature space, not input space; the algorithm "
            'requires access to penultimate-layer activations."'
        )
    )
    called_from: str = Field(
        description=(
            "Where in the algorithm this method is called, in plain prose. "
            'Example: "method/method.py::geometric_ranking and select_batch '
            '(both pass x_candidates and x_labeled through it before '
            'computing distances)."'
        )
    )
    paper_section: str = Field(
        description="Section / equation reference where the requirement is stated."
    )


class ParamGlossaryEntry(_Strict):
    """The paper's own explanation of one parameter (param-glossary
    design, 2026-07-21 — the researcher's traceability ask, second half).

    `meaning_quote` is the paper's VERBATIM definition sentence; the
    spec validator checks it against paper.md (whitespace-normalized
    substring, the equation-floor check), so a paraphrase fails at
    extraction time. Missing entries are honest — the analyzer only
    records parameters the paper actually explains."""

    name: str = Field(description=(
        'Parameter name as the paper uses it (e.g., "E", "C").'))
    aliases: list[str] = Field(default_factory=list, description=(
        "Other names derivation may use for the same parameter "
        '(e.g., "local_epochs"). Empty when the paper name is the '
        "derived name."))
    meaning_quote: str = Field(description=(
        "The paper's own words for what the parameter represents, "
        "copied verbatim from the paper."))
    paper_section: str = Field(description=(
        "Where the quote lives (section, equation, or table)."))
    paper_value: Optional[float] = Field(
        default=None,
        description=(
            "The literal value the paper states for this parameter, when it "
            "states one (e.g. a threshold given in an appendix table). Null "
            "when the paper defines the parameter's meaning without pinning a "
            "value, which is the common case and stays honest.\n\n"
            "This field plays two roles, split by whether the value depends "
            "on the data's scale (R2C-064).\n\n"
            "For a SCALE-FREE stated constant — a normalized threshold, a "
            "cutoff in (0,1), a probability, a ratio — this IS the blessed "
            "carrier. scale_dependent_hyperparameters is defined by "
            "data-scale dependence and its consumer (the scale-calibration "
            "step) dispatches entries by typed calibration_context, which a "
            "scale-free constant has no meaningful value for, so routing one "
            "there would fill the lane with entries its own consumer must "
            "reject or leave unprobeable.\n\n"
            "For a SCALE-DEPENDENT value — a physical distance or radius — "
            "critical_requirements.scale_dependent_hyperparameters is the "
            "blessed carrier. When that lane entry exists, a corresponding "
            "glossary entry is meaning-only and this field MUST be null; "
            "putting the value in both carriers is rejected during fresh "
            "strict validation (R2C-064). The slip-tolerant R2C-055 backstop "
            "still applies when the typed lane is absent: a glossary-only "
            "paper_value can create the paper-sourced parameter instead of "
            "letting a misrouted value vanish."
        ),
    )


class EvaluationProtocolQuantity(_Strict):
    """One paper-grounded quantity with one scientific protocol role."""

    role: EvaluationProtocolRole
    parameter_name: str | None = Field(
        default=None,
        description=(
            "Exact params.json carrier name when this protocol fact has a "
            "runtime parameter. Strict validation checks the name against "
            "the matched taxonomy node's independent protocol_role "
            "declaration. Leave null when the fact is spec-only; validation "
            "and test spans do not gain invented runtime params."
        ),
    )
    paper_names: list[str] = Field(
        min_length=1,
        description=(
            "Exact paper-facing prose names for this quantity (for example "
            "['forecast horizon'] or ['context length']). Symbols belong in "
            "paper_symbols. These names are an identity bridge for conflict "
            "detection and never authorize a numeric value by themselves."
        ),
    )
    paper_symbols: list[str] = Field(
        description=(
            "Every exact atomic paper symbol for this quantity (for example "
            "['K'] or ['K_pred']). Keep empty only when the evidence uses no "
            "symbol. Symbols identify the typed fact but never supply its "
            "numeric value."
        ),
    )
    value: int | float | None = Field(
        default=None,
        description=(
            "Positive paper-stated quantity for this role, or null when the "
            "paper defines the role but leaves its numeric value unspecified."
        ),
    )
    unit: str = Field(
        min_length=1,
        description=(
            "Singular scientific unit for this quantity (for example week, "
            "day, time_step, sample). Unit compatibility is checked only "
            "within the same role; sharing a unit never merges roles."
        ),
    )
    granularity: int | float = Field(
        gt=0,
        description=(
            "Positive size of one protocol step in the declared unit. For "
            "weekly observations use unit='week', granularity=1. Value, unit, "
            "and granularity travel together so a daily bundle cannot be "
            "silently treated as a weekly axis."
        ),
    )
    axis_evidence_quote: str = Field(
        min_length=1,
        description=(
            "Verbatim paper passage grounding unit and granularity. This may "
            "equal evidence_quote when one passage states both, or point to "
            "a separate cadence passage when the role/value quote uses only "
            "abstract time steps."
        ),
    )
    axis_paper_section: str = Field(min_length=1)
    axis_paper_element_ids: list[str] = Field(
        description=(
            "Paper-map element IDs grounding unit and granularity. Kept "
            "separate from role/value IDs so non-contiguous evidence is not "
            "stitched into a fabricated quote. An empty list is the honest "
            "state when no paper-map element covers the quoted passage; it "
            "costs the record downstream adjudication, never invents a link."
        ),
    )
    paper_value_status: PaperValueStatus
    evidence_quote: str = Field(
        min_length=1,
        description=(
            "Verbatim paper passage grounding the quantity's role and, when "
            "paper_value_status is paper_stated, its value. For an unspecified "
            "value, quote the definition of the symbolic quantity; do not use "
            "nearby notation as a numeric assertion. When the role wording, "
            "notation, and stated value live in different paper locations "
            "(e.g. prose notation plus an appendix table row), join the "
            "contiguous passages with the elision marker ' [...] '; each "
            "fragment is verbatim-checked against the paper on its own, and "
            "grammatical bindings never cross the marker."
        ),
    )
    paper_section: str = Field(min_length=1)
    paper_element_ids: list[str] = Field(
        description=(
            "Paper-map element IDs grounding this role and value status. "
            "Unit/granularity IDs are carried separately so non-contiguous "
            "paper passages stay distinct. An empty list is the honest state "
            "when no paper-map element covers the quoted passage; it costs "
            "the record downstream adjudication, never invents a link."
        ),
    )

    @field_validator(
        "unit", "axis_evidence_quote", "axis_paper_section",
        "evidence_quote", "paper_section", mode="before",
    )
    @classmethod
    def _required_protocol_text_is_not_whitespace(cls, value: object) -> object:
        # ``min_length`` runs before the after-model normalization below, so
        # whitespace-only evidence would otherwise survive and make the
        # normalized verbatim-quote check vacuously succeed (``"" in text``).
        return value.strip() if isinstance(value, str) else value

    @field_validator("value", "granularity", mode="before")
    @classmethod
    def _temporal_numbers_are_not_booleans(cls, value: object) -> object:
        # Pydantic otherwise coerces True and numeric strings to numbers before
        # an after-validator can distinguish the raw producer error. Downstream
        # scripts consume the original JSON, not this normalized model, so the
        # raw carrier must already be numeric.
        if isinstance(value, bool):
            raise ValueError(
                "evaluation protocol value and granularity must be numeric "
                "or null as applicable, not booleans"
            )
        if value is not None and not isinstance(value, (int, float)):
            raise ValueError(
                "evaluation protocol value and granularity must be raw "
                "numeric JSON values, not numeric strings"
            )
        return value

    @model_validator(mode="after")
    def _status_matches_value(self) -> "EvaluationProtocolQuantity":
        value = self.value
        if isinstance(value, bool):
            raise ValueError(
                f"evaluation protocol role {self.role.value!r} value must be "
                "numeric or null, not a boolean"
            )
        if self.paper_value_status == PaperValueStatus.paper_stated:
            if value is None:
                raise ValueError(
                    f"evaluation protocol role {self.role.value!r} is "
                    "paper_stated, so value must be present"
                )
            if isinstance(value, bool) or value <= 0:
                raise ValueError(
                    f"evaluation protocol role {self.role.value!r} must "
                    f"have a positive numeric value, got {value!r}"
                )
        elif value is not None:
            raise ValueError(
                f"evaluation protocol role {self.role.value!r} is "
                "paper_unspecified, so value must be null; a nearby number "
                "cannot be promoted into paper truth"
            )
        if isinstance(self.granularity, bool):
            raise ValueError(
                f"evaluation protocol role {self.role.value!r} granularity "
                "must be a positive number, not a boolean"
            )
        if self.parameter_name is not None:
            self.parameter_name = self.parameter_name.strip()
            if not self.parameter_name:
                raise ValueError("evaluation protocol parameter_name cannot be blank")
        self.paper_names = [item.strip() for item in self.paper_names]
        if any(not item for item in self.paper_names):
            raise ValueError("evaluation protocol paper_names cannot be blank")
        notation_shaped_names = [
            item for item in self.paper_names
            if (
                len(item) == 1
                or "_" in item
                or any(char.isdigit() for char in item)
                or (
                    item.isidentifier()
                    and any(char.isupper() for char in item[1:])
                )
            )
        ]
        if notation_shaped_names:
            raise ValueError(
                "evaluation protocol paper_names contains notation-shaped "
                f"identity value(s) {notation_shaped_names!r}; preserve exact "
                "case in paper_symbols instead of laundering symbols through "
                "the case-insensitive prose lane"
            )
        normalized_names = [" ".join(item.casefold().split()) for item in self.paper_names]
        if len(set(normalized_names)) != len(normalized_names):
            raise ValueError("evaluation protocol paper_names must be unique")
        self.paper_symbols = [item.strip() for item in self.paper_symbols]
        if any(not item for item in self.paper_symbols):
            raise ValueError("evaluation protocol paper_symbols cannot be blank")
        if len(set(self.paper_symbols)) != len(self.paper_symbols):
            raise ValueError("evaluation protocol paper_symbols must be unique")
        if any(re.search(r"\s", symbol) for symbol in self.paper_symbols):
            raise ValueError(
                "evaluation protocol paper_symbols must be atomic tokens; "
                "put prose phrases in paper_names"
            )
        if set(self.paper_symbols) & set(self.paper_names):
            raise ValueError(
                "evaluation protocol identities cannot appear in both "
                "paper_names and paper_symbols"
            )
        self.unit = self.unit.strip()
        if not re.fullmatch(r"[a-z][a-z0-9_]*", self.unit):
            raise ValueError(
                "evaluation protocol unit must be a canonical lowercase "
                "identifier such as 'week' or 'time_step'"
            )
        self.axis_evidence_quote = self.axis_evidence_quote.strip()
        self.axis_paper_section = self.axis_paper_section.strip()
        self.axis_paper_element_ids = [
            item.strip() for item in self.axis_paper_element_ids
        ]
        if any(not item for item in self.axis_paper_element_ids):
            raise ValueError(
                "evaluation protocol axis_paper_element_ids cannot be blank"
            )
        if len(set(self.axis_paper_element_ids)) != len(
            self.axis_paper_element_ids
        ):
            raise ValueError(
                "evaluation protocol axis_paper_element_ids must be unique"
            )
        self.evidence_quote = self.evidence_quote.strip()
        if any(
            not fragment.strip()
            for fragment in self.evidence_quote.split("[...]")
        ):
            raise ValueError(
                "evaluation protocol evidence_quote fragments joined by "
                "'[...]' must each be a non-empty verbatim passage; remove "
                "leading, trailing, or doubled elision markers"
            )
        self.paper_section = self.paper_section.strip()
        self.paper_element_ids = [item.strip() for item in self.paper_element_ids]
        if any(not item for item in self.paper_element_ids):
            raise ValueError("evaluation protocol paper_element_ids cannot be blank")
        if len(set(self.paper_element_ids)) != len(self.paper_element_ids):
            raise ValueError("evaluation protocol paper_element_ids must be unique")

        role_terms = {
            EvaluationProtocolRole.context_length: (
                # "lag"/"lags" is standard forecasting vocabulary for the
                # autoregressive context window ("number of lags", "P demand
                # lags", "lag order") and can never mean a horizon or an
                # evaluation span, so it states this role as unambiguously
                # as "lookback" (pdfgnn 2026-08-11 attempt-5 halt: the
                # paper's own Section 3.2 wording failed the floor).
                "context", "lookback", "look-back", "history", "historical",
                "input window", "lag", "lags",
            ),
            EvaluationProtocolRole.forecast_call_horizon: (
                "forecast", "horizon", "future", "predict", "prediction",
                "lead time",
            ),
            EvaluationProtocolRole.validation_span: (
                "validation", "valid set", "validation set",
            ),
            EvaluationProtocolRole.test_span: (
                "test", "test set", "holdout", "held-out",
            ),
        }
        # The three evidence floors below constrain ONE field jointly: the
        # role/value evidence_quote must simultaneously contain a declared
        # paper name, wording that states the scientific role, and every
        # declared exact symbol. Report every unmet floor in one message —
        # revealing them one fix dispatch at a time whipsawed the 2026-08-11
        # pdfgnn roll (name added -> role wording lost -> symbol lost -> role
        # lost again, 3 retries burned on one quote). A single passage often
        # cannot satisfy all three; contiguous verbatim fragments joined by
        # ' [...] ' are the sanctioned form for that.
        evidence_issues: list[str] = []
        if self.role == EvaluationProtocolRole.validation_span and not any(
            "valid" in name.casefold() for name in self.paper_names
        ):
            evidence_issues.append(
                "validation_span paper_names must explicitly identify the "
                "validation quantity"
            )
        if self.role == EvaluationProtocolRole.test_span and not any(
            any(term in name.casefold() for term in ("test", "holdout", "held-out"))
            for name in self.paper_names
        ):
            evidence_issues.append(
                "test_span paper_names must explicitly identify the test or "
                "holdout quantity"
            )
        paper_name_spans = [
            span
            for paper_name in self.paper_names
            for span in _literal_spans(
                self.evidence_quote.casefold(), paper_name.casefold()
            )
        ]
        if not paper_name_spans:
            evidence_issues.append(
                f"evaluation protocol role {self.role.value!r} evidence_quote "
                f"does not contain any declared paper_names {self.paper_names!r}; "
                "the paper-facing identity must be verbatim, not inferred"
            )
        role_spans = [
            span
            for term in role_terms[self.role]
            for span in _literal_spans(
                self.evidence_quote.casefold(), term.casefold()
            )
        ]
        if not role_spans:
            evidence_issues.append(
                f"evaluation protocol role {self.role.value!r} evidence_quote "
                "does not state that scientific role; quote enough surrounding "
                "paper text to distinguish context, one-call forecast, "
                "validation, and test quantities"
            )
        missing_symbols = [
            symbol
            for symbol in self.paper_symbols
            if not _literal_spans(self.evidence_quote, symbol)
        ]
        if missing_symbols:
            evidence_issues.append(
                f"evaluation protocol paper symbol(s) {missing_symbols!r} are "
                "absent from evidence_quote; symbols must preserve exact "
                "paper spelling and case"
            )
        # The value-binding floor constrains the SAME quote, so it batches
        # here rather than waiting in the later validator: on the 2026-08-11
        # attempt-5 roll it was revealed only after the role floor passed,
        # one more fix dispatch late (the paper states the value only in an
        # appendix table, so the producer must learn that the join has to
        # carry the number in the same message that demands the join).
        if self.paper_value_status == PaperValueStatus.paper_stated:
            bound_values = evaluation_protocol_bound_values(self)
            if not bound_values or not any(
                _numeric_values_equal(self.value, candidate)
                for candidate in bound_values
            ):
                evidence_issues.append(
                    f"evaluation protocol role {self.role.value!r} declares "
                    f"value={self.value!r}, but strict name/role evidence "
                    f"binds {bound_values!r}; an only/nearby number, T+1 "
                    "index, or another role's span cannot authorize paper "
                    "truth"
                )
        if evidence_issues:
            raise ValueError(
                "; ".join(evidence_issues)
                + "; one evidence_quote must satisfy every floor above at "
                "once — when no single contiguous passage does, join "
                "contiguous verbatim fragments with ' [...] '"
            )
        return self

    def role_adjacent_symbol_candidates(self) -> set[str]:
        """Exact-notation symbols the evidence_quote uses in a role-signature
        position. Ownership is enforced at the protocol level, not here: a
        quote that carries enough surrounding paper text (which the role-term
        gate demands) legitimately mentions sibling roles' symbols, so a
        candidate only becomes a finding when no quantity in the protocol
        declares it."""
        role_pattern = {
            EvaluationProtocolRole.context_length: (
                r"(?:context(?:\s+(?:length|window))?|lookback|history\s+window)"
            ),
            EvaluationProtocolRole.forecast_call_horizon: (
                r"(?:(?:forecast(?:[ -]call)?|prediction)\s+horizon|"
                r"future\s+(?:time\s+)?steps?)"
            ),
            EvaluationProtocolRole.validation_span: (
                r"(?:validation(?:\s+(?:set|span|window|length))?)"
            ),
            EvaluationProtocolRole.test_span: (
                r"(?:(?:test|holdout|held-out)(?:\s+(?:set|span|window|length))?)"
            ),
        }[self.role]
        positional_symbol_patterns = (
            rf"\bfuture\s+(?P<symbol>[A-Za-z][A-Za-z0-9_]*)\s+"
            r"(?:time[ -]?)?steps?\b",
            rf"\b{role_pattern}\s+(?P<symbol>[A-Za-z][A-Za-z0-9_]*)\b"
            r"(?=\s*(?:=|:|\||\)|,|denotes\b|represents\b|is\b))",
        )
        definition_symbol_patterns = (
            rf"\b{role_pattern}\s*\(\s*"
            r"(?P<symbol>[A-Za-z][A-Za-z0-9_]*)\s*\)",
            rf"\b{role_pattern}\b[^.\n]{{0,30}}?\b"
            r"(?:written|denoted|represented)\s+as\s+"
            r"(?P<symbol>[A-Za-z][A-Za-z0-9_]*)\b",
            rf"\b(?P<symbol>[A-Za-z][A-Za-z0-9_]*)\s+"
            rf"(?:denotes|represents|is)\s+(?:the\s+)?{role_pattern}\b",
            rf"\b{role_pattern}\s+(?:is\s+)?(?:denoted|represented)\s+by\s+"
            r"(?P<symbol>[A-Za-z][A-Za-z0-9_]*)\b",
            r"\b(?P<symbol>[A-Za-z][A-Za-z0-9_]*)\s+(?:denotes|represents)\s+"
            r"(?:that|this|the)\s+(?:window|span|horizon|length)\b",
        )
        if self.role == EvaluationProtocolRole.context_length:
            positional_symbol_patterns += (
                r"\b(?P<symbol>[A-Za-z][A-Za-z0-9_]*)\s+"
                r"(?:demand\s+)?lags?\b",
            )
        elif self.role == EvaluationProtocolRole.forecast_call_horizon:
            positional_symbol_patterns += (
                r"\bover\s+(?P<symbol>[A-Za-z][A-Za-z0-9_]*)\s+"
                r"(?:future\s+)?(?:weeks?|time[ -]?steps?|points?)\b",
                r"\b(?P<symbol>[A-Za-z][A-Za-z0-9_]*)\s+future\s+"
                r"(?:weeks?|time[ -]?steps?|points?)\b",
            )
        stopwords = {
            "a", "an", "the", "time", "step", "steps", "is", "set",
            "uses", "covers", "span", "spans", "length", "window",
            "validation", "test", "holdout", "horizon", "forecast",
        }
        positional_candidates = {
            match.group("symbol")
            for pattern in positional_symbol_patterns
            for match in re.finditer(
                pattern, self.evidence_quote, flags=re.IGNORECASE
            )
            if match.group("symbol").casefold() not in stopwords
            and (
                len(match.group("symbol")) == 1
                or "_" in match.group("symbol")
                or any(char.isdigit() for char in match.group("symbol"))
                or any(char.isupper() for char in match.group("symbol"))
                or match.group("symbol") in self.paper_names
            )
        }
        definition_candidates = {
            match.group("symbol")
            for pattern in definition_symbol_patterns
            for match in re.finditer(
                pattern, self.evidence_quote, flags=re.IGNORECASE
            )
            if match.group("symbol").casefold() not in stopwords
        }
        # Re-run exact spelling capture for each candidate. The case-insensitive
        # grammar above recognizes sentence capitalization, but notation itself
        # remains byte-exact in paper_symbols.
        exact_candidates = {
            match.group(0)
            for candidate in positional_candidates | definition_candidates
            for match in re.finditer(
                r"(?<![A-Za-z0-9_])" + re.escape(candidate)
                + r"(?![A-Za-z0-9_])",
                self.evidence_quote,
            )
        }
        return exact_candidates

    @model_validator(mode="after")
    def _role_identity_names_are_explicit(self) -> "EvaluationProtocolQuantity":
        bound_values = evaluation_protocol_bound_values(self)
        if bound_values:
            concrete_unit_patterns = {
                "second": r"seconds?|secs?",
                "minute": r"minutes?|mins?",
                "hour": r"hours?",
                "day": r"days?",
                "week": r"weeks?",
                "month": r"months?",
                "quarter": r"quarters?",
                "year": r"years?",
            }
            explicit_units: set[str] = set()
            for number_value, number_span in _protocol_number_tokens(
                self.evidence_quote
            ):
                if not any(
                    _numeric_values_equal(number_value, candidate)
                    for candidate in bound_values
                ):
                    continue
                suffix = self.evidence_quote[number_span[1]:number_span[1] + 20]
                for canonical, pattern in concrete_unit_patterns.items():
                    if re.match(
                        rf"\s*[- ]?\s*(?:{pattern})\b",
                        suffix,
                        flags=re.IGNORECASE,
                    ):
                        explicit_units.add(canonical)
            if explicit_units and explicit_units != {self.unit}:
                raise ValueError(
                    f"evaluation protocol role {self.role.value!r} binds "
                    f"value(s) {bound_values!r} to explicit unit(s) "
                    f"{sorted(explicit_units)!r} in evidence_quote, but the "
                    f"typed axis declares unit={self.unit!r}; preserve the "
                    "unit/value conflict instead of publishing a converted "
                    "fact without an explicit relationship"
                )
        if self.paper_value_status == PaperValueStatus.paper_stated:
            # The no-binding arm batches with the quote evidence floors in
            # _status_matches_value; only contradiction shapes remain here.
            conflicts = [
                candidate for candidate in bound_values
                if not _numeric_values_equal(self.value, candidate)
            ]
            if conflicts:
                raise ValueError(
                    f"evaluation protocol role {self.role.value!r} evidence "
                    f"binds conflicting values {bound_values!r}; preserve the "
                    "conflict instead of selecting one"
                )
        elif bound_values:
            raise ValueError(
                f"evaluation protocol role {self.role.value!r} is marked "
                "paper_unspecified, but its evidence explicitly binds value(s) "
                f"{bound_values!r}; a stated value cannot be hidden as an "
                "unspecified runtime choice"
            )

        axis_nouns = (
            r"(?:demand|targets?|observations?|measurements?|samples?|data|"
            r"time[ -]?series|series|cadence|frequency|intervals?)"
        )
        unit_terms = {
            "second": r"(?:seconds?|sec(?:ond)?s?|secondly)",
            "minute": r"(?:minutes?|mins?)",
            "hour": r"(?:hours?|hourly)",
            "day": r"(?:days?|daily)",
            "week": r"(?:weeks?|weekly)",
            "month": r"(?:months?|monthly)",
            "quarter": r"(?:quarters?|quarterly)",
            "year": r"(?:years?|yearly|annual(?:ly)?)",
            "time_step": r"(?:time[ -]?steps?|timesteps?)",
        }
        unit_term = unit_terms.get(
            self.unit, re.escape(self.unit.replace("_", " ")) + r"s?"
        )
        # Shared by the axis-binding pattern and the abstract-step cadence
        # arm below; both must accept the same temporal-window qualifiers.
        window_qualifiers = (
            r"(?:future|historical|context|forecast|previous|following|"
            r"preceding|next|upcoming)"
        )
        unit_patterns = (
            rf"\b{unit_term}\b\s+{axis_nouns}\b",
            rf"\b{axis_nouns}\b[^.\n]{{0,45}}\b"
            rf"(?:aggregate|sample|observe|record|measure)(?:d|s|ing)?\b"
            rf"[^.\n]{{0,30}}\b{unit_term}\b",
            rf"\b(?:per|each|every)\s+(?:\d+(?:\.\d+)?\s+)?"
            rf"{unit_term}\b",
            rf"\b\d+(?:\.\d+)?[ -]{unit_term}\s+intervals?\b",
            # Abstract protocol steps (for example future K time steps) are a
            # legitimate axis even when a later task maps them to bundle
            # time. The qualifier vocabulary covers both directions of the
            # temporal window: "the previous P and the following K time
            # steps" states the axis exactly as "future K time steps" does
            # (pdfgnn 2026-08-11 attempt 7).
            rf"\b{window_qualifiers}\b[^.\n]{{0,35}}\b{unit_term}\b",
        )
        matching_unit_patterns = [
            pattern for pattern in unit_patterns
            if re.search(pattern, self.axis_evidence_quote, flags=re.IGNORECASE)
        ]
        if not matching_unit_patterns:
            # Name the units the quote DOES state, so the fixer resolves the
            # unit/quote disagreement in one dispatch instead of guessing
            # (pdfgnn 2026-08-11 attempt 7: unit='time_step' declared against
            # the weekly-aggregation axis quote burned the loop's last
            # retry).
            stated_units = sorted(
                canonical
                for canonical, term in unit_terms.items()
                if canonical != self.unit
                and re.search(
                    rf"\b{term}\b", self.axis_evidence_quote,
                    flags=re.IGNORECASE,
                )
            )
            unit_appears_unbound = re.search(
                rf"\b{unit_term}\b", self.axis_evidence_quote,
                flags=re.IGNORECASE,
            )
            if stated_units:
                hint = (
                    f"; the quote itself states unit term(s) "
                    f"{stated_units!r} — declare that unit or quote the "
                    f"passage stating {self.unit!r} cadence"
                )
            elif unit_appears_unbound:
                hint = (
                    f"; the quote mentions {self.unit!r} wording but not in "
                    "an axis-binding pattern — quote the passage that ties "
                    "the unit to the observed or predicted series"
                )
            else:
                hint = ""
            raise ValueError(
                f"evaluation protocol role {self.role.value!r} declares "
                f"unit={self.unit!r}, but axis_evidence_quote contains no "
                f"compatible unit term; unit provenance must be explicit{hint}"
            )
        cadence_values: list[int | float] = []

        def _add_cadence(raw: str | None) -> None:
            value = 1 if raw is None else _coerce_numeric_token(raw)
            if not any(
                _numeric_values_equal(value, existing)
                for existing in cadence_values
            ):
                cadence_values.append(value)

        number_token = r"\d[\d,]*(?:\.\d+)?"
        for match in re.finditer(
            rf"\b(?:per|each|every)\s+"
            rf"(?:(?P<value>{number_token})\s+)?{unit_term}\b",
            self.axis_evidence_quote,
            flags=re.IGNORECASE,
        ):
            _add_cadence(match.group("value"))
        for match in re.finditer(
            rf"\b(?P<value>{number_token})[ -]{unit_term}\s+"
            r"(?:cadence|frequency|intervals?)\b",
            self.axis_evidence_quote,
            flags=re.IGNORECASE,
        ):
            _add_cadence(match.group("value"))
        cadence_adjectives = {
            "second": r"\bsecondly\b",
            "hour": r"\bhourly\b",
            "day": r"\bdaily\b",
            "week": r"\bweekly\b",
            "month": r"\bmonthly\b",
            "quarter": r"\bquarterly\b",
            "year": r"\b(?:yearly|annual(?:ly)?)\b",
        }.get(self.unit)
        if cadence_adjectives and re.search(
            rf"{cadence_adjectives}\s+{axis_nouns}\b",
            self.axis_evidence_quote,
            flags=re.IGNORECASE,
        ):
            _add_cadence(None)
        if self.unit == "time_step" and re.search(
            rf"\b{window_qualifiers}\b[^.\n]{{0,35}}"
            r"\b(?:time[ -]?steps?|timesteps?)\b",
            self.axis_evidence_quote,
            flags=re.IGNORECASE,
        ):
            _add_cadence(None)
        if not cadence_values or not any(
            _numeric_values_equal(self.granularity, candidate)
            for candidate in cadence_values
        ) or any(
            not _numeric_values_equal(self.granularity, candidate)
            for candidate in cadence_values
        ):
            raise ValueError(
                f"evaluation protocol role {self.role.value!r} declares "
                f"granularity={self.granularity!r}, but axis evidence binds "
                f"cadence value(s) {cadence_values!r}; a window duration "
                "cannot supply sampling granularity"
            )
        return self


class EvaluationProtocolScheme(_Strict):
    """How forecast origins and held-out spans form the evaluation."""

    kind: EvaluationSchemeKind | None = Field(
        default=None,
        description=(
            "Paper-stated evaluation shape, or null when the paper does not "
            "specify how forecast origins are traversed."
        ),
    )
    paper_value_status: PaperValueStatus
    description: str = Field(min_length=1)
    evidence_quote: str = Field(
        min_length=1,
        description=(
            "Verbatim paper passage establishing the chronological evaluation "
            "shape (single holdout, rolling origin, expanding window, etc.)."
        ),
    )
    paper_section: str = Field(min_length=1)
    paper_element_ids: list[str] = Field(
        description=(
            "Paper-map element IDs grounding the scheme statement. An empty "
            "list is the honest state when no paper-map element covers the "
            "quoted passage."
        ),
    )

    @field_validator(
        "description", "evidence_quote", "paper_section", mode="before"
    )
    @classmethod
    def _required_protocol_text_is_not_whitespace(cls, value: object) -> object:
        return value.strip() if isinstance(value, str) else value

    @model_validator(mode="after")
    def _status_matches_kind(self) -> "EvaluationProtocolScheme":
        if (
            self.paper_value_status == PaperValueStatus.paper_stated
            and self.kind is None
        ):
            raise ValueError(
                "evaluation protocol scheme is paper_stated, so kind must be present"
            )
        if (
            self.paper_value_status == PaperValueStatus.paper_unspecified
            and self.kind is not None
        ):
            raise ValueError(
                "evaluation protocol scheme is paper_unspecified, so kind must be null"
            )
        self.description = self.description.strip()
        self.evidence_quote = self.evidence_quote.strip()
        self.paper_section = self.paper_section.strip()
        self.paper_element_ids = [item.strip() for item in self.paper_element_ids]
        if any(not item for item in self.paper_element_ids):
            raise ValueError(
                "evaluation protocol scheme paper_element_ids cannot be blank"
            )
        if len(set(self.paper_element_ids)) != len(self.paper_element_ids):
            raise ValueError(
                "evaluation protocol scheme paper_element_ids must be unique"
            )
        evidence_patterns = {
            EvaluationSchemeKind.single_holdout: (
                r"\b(?:single|one)\s+(?:chronological\s+)?"
                r"(?:hold[ -]?out|held[ -]?out|split)\b",
                r"\bsplit\s+each\s+[^.\n]{0,30}\s+into\s+"
                r"(?:three|3)\s+subsets\b",
                r"\btraining\s*,\s*validation\s*,\s*and\s+test\s+sets\b",
            ),
            EvaluationSchemeKind.rolling_origin: (
                r"\brolling[ -]origin\b",
                r"\brolling\s+(?:forecast\s+)?origins?\b",
                r"\bforecast\s+origins?\b[^.\n]{0,60}\b"
                r"(?:advance|move|step)(?:s|d|ing)?\b",
            ),
            EvaluationSchemeKind.expanding_window: (
                r"\bexpanding\s+window\b", r"\bgrowing\s+window\b",
            ),
            EvaluationSchemeKind.sliding_window: (
                r"\bsliding\s+window\b", r"\bmoving\s+window\b",
            ),
            EvaluationSchemeKind.cross_validation: (
                r"\b(?:time[ -]series\s+)?cross[ -]?validation\b",
                r"\b(?:cross[ -]?validation|evaluation)\s+folds?\b",
            ),
        }
        recognized = {
            kind for kind, patterns in evidence_patterns.items()
            if any(
                re.search(pattern, self.evidence_quote, flags=re.IGNORECASE)
                for pattern in patterns
            )
        }
        description_recognized = {
            kind for kind, patterns in evidence_patterns.items()
            if any(
                re.search(pattern, self.description, flags=re.IGNORECASE)
                for pattern in patterns
            )
        }
        if self.paper_value_status == PaperValueStatus.paper_unspecified:
            if recognized:
                raise ValueError(
                    "evaluation protocol scheme is marked paper_unspecified, "
                    f"but evidence explicitly identifies {sorted(kind.value for kind in recognized)!r}; "
                    "a stated scheme cannot be hidden as unspecified"
                )
        elif recognized != {self.kind}:
            raise ValueError(
                f"evaluation protocol scheme kind={self.kind.value!r} is not "
                "exclusively grounded by its evidence_quote; recognized "
                f"scheme cue(s) are {sorted(kind.value for kind in recognized)!r}. "
                "Preserve missing or conflicting scheme evidence instead of "
                "relabeling it."
            )
        if description_recognized and description_recognized != {self.kind}:
            raise ValueError(
                "evaluation protocol scheme description contains conflicting "
                f"scheme cue(s) {sorted(kind.value for kind in description_recognized)!r} "
                f"for kind={self.kind.value if self.kind is not None else None!r}; "
                "researcher-facing summary prose cannot contradict the typed "
                "scheme"
            )
        return self


class EvaluationProtocol(_Strict):
    """Role-typed temporal evaluation facts, independent of demo choices."""

    scheme: EvaluationProtocolScheme
    quantities: list[EvaluationProtocolQuantity] = Field(
        min_length=4,
        max_length=4,
        description=(
            "Exactly one fact for each role: context_length, "
            "forecast_call_horizon, validation_span, and test_span. Use "
            "paper_unspecified with null value rather than omitting a role."
        ),
    )

    @model_validator(mode="after")
    def _roles_are_complete_and_unique(self) -> "EvaluationProtocol":
        by_role: dict[EvaluationProtocolRole, list[EvaluationProtocolQuantity]] = {}
        for quantity in self.quantities:
            by_role.setdefault(quantity.role, []).append(quantity)
        duplicates = {
            role.value: [
                {
                    "value": item.value,
                    "unit": item.unit,
                    "granularity": item.granularity,
                    "paper_section": item.paper_section,
                }
                for item in items
            ]
            for role, items in by_role.items()
            if len(items) > 1
        }
        if duplicates:
            raise ValueError(
                "evaluation_protocol.quantities contains conflicting duplicate "
                f"role facts: {duplicates}; preserve the conflict for analyzer "
                "review instead of selecting one value"
            )
        missing = sorted(
            role.value for role in EvaluationProtocolRole if role not in by_role
        )
        if missing:
            raise ValueError(
                "evaluation_protocol.quantities is missing role(s) "
                f"{missing}; use paper_unspecified with value null when the "
                "paper leaves a quantity symbolic"
            )
        paper_name_owners: dict[str, EvaluationProtocolRole] = {}
        for quantity in self.quantities:
            for paper_name in quantity.paper_names:
                normalized = normalize_evaluation_protocol_label(paper_name)
                prior = paper_name_owners.get(normalized)
                if prior is not None and prior != quantity.role:
                    raise ValueError(
                        f"evaluation protocol paper name {paper_name!r} is "
                        f"assigned to both {prior.value!r} and "
                        f"{quantity.role.value!r}; paper-facing identities "
                        "must not cross protocol roles"
                    )
                paper_name_owners[normalized] = quantity.role
        paper_symbol_owners: dict[str, EvaluationProtocolRole] = {}
        for quantity in self.quantities:
            for symbol in quantity.paper_symbols:
                prose_owner = paper_name_owners.get(
                    normalize_evaluation_protocol_label(symbol)
                )
                if prose_owner is not None:
                    raise ValueError(
                        f"evaluation protocol identity {symbol!r} is a paper "
                        f"symbol for {quantity.role.value!r} but a prose name "
                        f"for {prose_owner.value!r}; one paper identity cannot "
                        "cross roles or identity lanes"
                    )
                prior = paper_symbol_owners.get(symbol)
                if prior is not None and prior != quantity.role:
                    raise ValueError(
                        f"evaluation protocol paper symbol {symbol!r} is "
                        f"assigned to both {prior.value!r} and "
                        f"{quantity.role.value!r}; notation identities must "
                        "not cross protocol roles"
                    )
                paper_symbol_owners[symbol] = quantity.role
        declared_symbols = {
            symbol
            for quantity in self.quantities
            for symbol in quantity.paper_symbols
        }
        for quantity in self.quantities:
            omitted = sorted(
                candidate
                for candidate in quantity.role_adjacent_symbol_candidates()
                if candidate not in declared_symbols
            )
            if omitted:
                raise ValueError(
                    f"evaluation protocol role {quantity.role.value!r} "
                    f"paper_symbols omit role-adjacent paper symbol(s) "
                    f"{omitted!r} that no protocol quantity declares; list "
                    "exact notation on the quantity that owns each symbol so "
                    "lowercase and multi-character carriers cannot bypass "
                    "typed ownership"
                )
        return self


ScenarioAssumptionScalar = str | int | float | bool
ScenarioAssumptionValue = (
    ScenarioAssumptionScalar
    | list[ScenarioAssumptionScalar]
    | dict[str, ScenarioAssumptionScalar]
)


class ScenarioAssumption(_Strict):
    """One paper-stated constraint on the demonstration scenario.

    The mapping key in ``MethodSpec.scenario_assumptions`` is the matched
    taxonomy node's dimension id. ``normalized_value`` is intentionally
    small JSON data rather than prose so the later detector slice can bind
    it without reparsing a sentence. ``evidence_quote`` keeps the source
    auditable and is checked verbatim against paper.md by the Stage 1
    validator.
    """

    normalized_value: ScenarioAssumptionValue = Field(
        description=(
            "A compact normalized value following the matched taxonomy "
            "dimension's guidance. Scalars, flat scalar lists, and flat "
            "scalar mappings are supported."
        ),
    )
    evidence_quote: str = Field(
        min_length=1,
        description=(
            "A verbatim text passage from the parsed paper that states the "
            "scenario assumption. Figure-only evidence is not accepted."
        ),
    )
    paper_location: str = Field(
        min_length=1,
        description="Section, remark, example, equation, or table containing the quote.",
    )

    @model_validator(mode="after")
    def _normalized_value_is_substantive(self) -> "ScenarioAssumption":
        value = self.normalized_value
        if isinstance(value, str) and not value.strip():
            raise ValueError("scenario assumption normalized_value cannot be blank")
        if isinstance(value, (list, dict)) and not value:
            raise ValueError("scenario assumption normalized_value cannot be empty")
        if isinstance(value, list):
            for item in value:
                if isinstance(item, str) and not item.strip():
                    raise ValueError(
                        "scenario assumption normalized_value list cannot "
                        "contain blank strings"
                    )
        if isinstance(value, dict):
            for key, item in value.items():
                if not key.strip():
                    raise ValueError(
                        "scenario assumption normalized_value mapping cannot "
                        "contain blank keys"
                    )
                if isinstance(item, str) and not item.strip():
                    raise ValueError(
                        "scenario assumption normalized_value mapping cannot "
                        "contain blank string values"
                    )
        return self


class CriticalRequirements(_Strict):
    model: ModelRequirements
    training: TrainingRequirements
    data_setup: DataSetup
    blockers: list[Blocker]
    param_glossary: list[ParamGlossaryEntry] = Field(
        default_factory=list,
        max_length=12,
        description=(
            "The paper's own per-parameter definitions, capped at 12 "
            "entries (the analyzer-reliability guard). Populated by the "
            "analyzer while reading the sections it already reads; "
            "consumed deterministically by derive_params to stamp "
            "`paper_says` onto matching derived parameters. Empty when "
            "the paper explains no parameters explicitly."
        ),
    )
    scale_dependent_hyperparameters: list[ScaleDependentHyperparam] = Field(
        default_factory=list,
        description=(
            "Method-specific hyperparameters whose value depends on a typed "
            "calibration context: feature magnitude, the supported target-box "
            "grid representation, or preserved unsupported physical/mixed/"
            "unknown evidence. Empty list when the method has no such "
            "hyperparameters. Populated by the analyzer; consumed by Stage "
            "2.x and the fidelity reviewer through the same observer-routing "
            "contract."
        ),
    )
    required_model_methods: list[RequiredModelMethod] = Field(
        default_factory=list,
        description=(
            "Methods (in the Python class sense) the architecture must "
            "expose beyond `forward()`, because the algorithm body in "
            "method.py explicitly calls them. Empty list when the method "
            "only needs `forward()` (e.g., entropy / margin / random "
            "acquisition in active learning). Populated by the analyzer; "
            "iterated by the Stage 4 fidelity reviewer as an explicit "
            "per-entry checklist (does the method exist, does its body "
            "match `returns_description`, does method.py actually call "
            "it). Exists to catch the failure mode where an interface "
            "is satisfied but the body silently returns the wrong thing."
        ),
    )


# ---------------------------------------------------------------------------
# Top-level
# ---------------------------------------------------------------------------


class MethodSpec(_Strict):
    """The canonical structured representation of a paper's method, claim, and
    comparison setup. Produced by Stage 1 (Step 3); consumed by every downstream
    stage."""

    # The nested carrier union cannot see the top-level schema version.  Keep
    # the portable JSON Schema aligned with the Pydantic version boundary:
    # archived payloads may use the legacy string, while explicit output for
    # the current producer schema must use the typed carrier.  The ordinary
    # field schemas still validate the complete objects referenced here.
    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "allOf": [
                {
                    "if": {
                        "required": ["schema_version"],
                        "properties": {
                            "schema_version": {"const": SCHEMA_VERSION},
                        },
                    },
                    "then": {
                        "properties": {
                            "critical_requirements": {
                                "properties": {
                                    "scale_dependent_hyperparameters": {
                                        "items": {
                                            "required": ["calibration_context"],
                                            "properties": {
                                                "calibration_context": {
                                                    "not": {"type": "null"},
                                                },
                                                "assumes_data_scale": {
                                                    "type": "null",
                                                },
                                            },
                                        },
                                    },
                                },
                            },
                        },
                    },
                },
                {
                    "if": {
                        "required": [
                            "schema_version",
                            "methodology_replication_contract",
                        ],
                        "properties": {
                            "schema_version": {"const": SCHEMA_VERSION},
                            "methodology_replication_contract": {
                                "type": "object",
                                "required": ["elements"],
                                "properties": {
                                    "elements": {
                                        "contains": {
                                            "type": "object",
                                            "required": ["relational_structure"],
                                            "properties": {
                                                "relational_structure": {
                                                    "type": "object",
                                                    "required": ["kind"],
                                                    "properties": {
                                                        "kind": {
                                                            "const": "homogeneous_graph"
                                                        },
                                                    },
                                                },
                                            },
                                        },
                                    },
                                },
                            },
                        },
                    },
                    "then": {
                        "properties": {
                            "methodology_replication_contract": {
                                "required": ["homogeneous_graph_mechanism"],
                                "properties": {
                                    "homogeneous_graph_mechanism": {
                                        "not": {"type": "null"}
                                    },
                                },
                            },
                        },
                    },
                },
            ],
        },
    )

    schema_version: str = Field(
        default=SCHEMA_VERSION,
        description=(
            "Schema version this spec was written against. Validators reject "
            "specs whose version they do not understand."
        ),
    )

    paper: Paper
    core_method: CoreMethod
    paper_claims: PaperClaims
    try_it_out: TryItOut
    data_requirements: DataRequirements
    dependencies: Dependencies
    repo: Repo
    comparison: Comparison
    critical_requirements: CriticalRequirements
    scenario_assumptions: Optional[dict[str, ScenarioAssumption]] = Field(
        default=None,
        min_length=1,
        description=(
            "Paper-stated demo-scenario constraints keyed by dimension ids "
            "declared by the matched taxonomy node. Each entry carries a "
            "normalized value, verbatim evidence quote, and paper location. "
            "Omit when the node declares no capture dimensions or the paper "
            "states no assumption on them."
        ),
    )
    methodology_replication_contract: Optional[MethodologyReplicationContract] = None
    methodology_contract_pack: Optional[MethodologyContractPack] = None
    replication_feasibility: Optional[ReplicationFeasibility] = None
    feasibility: Feasibility

    def derive_replication_feasibility(self) -> Optional[ReplicationFeasibility]:
        """Derive the run-level replication verdict from contract elements.

        The halt rule is intentionally narrow: only a core methodology element
        marked `not_replicable` blocks generation. Approved approximations on
        any element move the run to the explicit approximation verdict.
        """
        contract = self.methodology_replication_contract
        if contract is None:
            return None

        blockers: list[ReplicationFeasibilityBlocker] = []
        approved_approximations: list[ReplicationFeasibilityApproximation] = []

        for element in contract.elements:
            if (
                element.role == MethodologyElementRole.core_methodology
                and element.replication_status == ReplicationStatus.not_replicable
            ):
                blockers.append(ReplicationFeasibilityBlocker(
                    element_id=element.element_id,
                    technical_concept=element.technical_concept,
                    reason=element.feasibility_rationale,
                ))
            if element.replication_status == ReplicationStatus.faithful_approximation_allowed:
                approved_approximations.append(ReplicationFeasibilityApproximation(
                    element_id=element.element_id,
                    technical_concept=element.technical_concept,
                    rationale=element.feasibility_rationale,
                ))

        if blockers:
            verdict = ReplicationFeasibilityVerdict.not_replicable
        elif approved_approximations:
            verdict = ReplicationFeasibilityVerdict.feasible_with_approved_approximations
        else:
            verdict = ReplicationFeasibilityVerdict.feasible

        return ReplicationFeasibility(
            verdict=verdict,
            blockers=blockers,
            approved_approximations=approved_approximations,
        )

    @model_validator(mode="after")
    def _require_typed_calibration_context_for_fresh_specs(self) -> "MethodSpec":
        """Keep archived scale carriers readable without weakening fresh specs."""
        if "schema_version" not in self.model_fields_set:
            return self

        declared = _schema_version_tuple(self.schema_version)
        activation = _schema_version_tuple(CALIBRATION_CONTEXT_SCHEMA_VERSION)
        if not declared or declared < activation:
            return self

        for index, entry in enumerate(
            self.critical_requirements.scale_dependent_hyperparameters
        ):
            if entry.calibration_context is None:
                raise ValueError(
                    "critical_requirements.scale_dependent_hyperparameters["
                    f"{index}] {entry.name!r} uses legacy "
                    "assumes_data_scale under schema "
                    f"{self.schema_version}; explicit v1.13+ specs must emit "
                    "typed calibration_context"
                )
        return self

    @model_validator(mode="after")
    def _require_graph_mechanism_for_fresh_graph_specs(self) -> "MethodSpec":
        """Keep archived graph specs readable while closing fresh graph grammar."""
        if "schema_version" not in self.model_fields_set:
            return self

        declared = _schema_version_tuple(self.schema_version)
        activation = _schema_version_tuple(
            HOMOGENEOUS_GRAPH_MECHANISM_SCHEMA_VERSION
        )
        if not declared or declared < activation:
            return self

        contract = self.methodology_replication_contract
        if contract is None:
            return self
        graph_ids = [
            element.element_id
            for element in contract.elements
            if (
                element.relational_structure is not None
                and element.relational_structure.kind
                == RelationalStructureKind.homogeneous_graph
            )
        ]
        if graph_ids and contract.homogeneous_graph_mechanism is None:
            raise ValueError(
                "explicit v1.14+ homogeneous graph specs must provide "
                "methodology_replication_contract.homogeneous_graph_mechanism; "
                f"graph element IDs: {sorted(graph_ids)}"
            )
        return self

    @model_validator(mode="after")
    def _evaluation_protocol_carrier_consistency(self) -> "MethodSpec":
        """Reject contradictory typed and legacy protocol carriers.

        The protocol block is the only surface allowed to authorize temporal
        paper values. Its ``parameter_name`` binding is checked independently
        against taxonomy in strict validation; here we keep the spec's own
        glossary and scale-calibration lanes consistent with that declaration.
        Name normalization below is used only to detect a duplicate legacy
        lane and halt. It never authorizes a value or chooses a role.
        """
        protocol = self.comparison.evaluation_protocol
        if protocol is None:
            return self

        bound: dict[str, EvaluationProtocolQuantity] = {}
        quantity_index: dict[str, int] = {}
        for index, quantity in enumerate(protocol.quantities):
            name = quantity.parameter_name
            if name is None:
                continue
            if name in bound:
                prior = bound[name]
                raise ValueError(
                    "comparison.evaluation_protocol binds parameter_name "
                    f"{name!r} to multiple roles: quantities["
                    f"{quantity_index[name]}].role={prior.role.value!r} and "
                    f"quantities[{index}].role={quantity.role.value!r}; one "
                    "runtime carrier cannot represent two protocol roles"
                )
            bound[name] = quantity
            quantity_index[name] = index

        glossary_labels: dict[str, set[str]] = {
            name: evaluation_protocol_identity_labels(quantity)
            for name, quantity in bound.items()
        }

        def _label_appears_in_protocol_evidence(
            label: str, quantity: EvaluationProtocolQuantity
        ) -> bool:
            """Conservative duplicate detection, never value authority.

            If an analyzer omits a paper symbol such as ``K`` from
            ``paper_names`` but puts a legacy K carrier beside the typed
            horizon, the exact symbol still appears in the typed evidence.
            Halt on that unresolved identity overlap; do not use it to select
            a role or number.
            """
            label = label.strip()
            if not label:
                return False
            escaped = re.escape(label).replace(r"\ ", r"\s+")
            prefix = r"(?<![A-Za-z0-9_])" if label[0].isalnum() else ""
            suffix = r"(?![A-Za-z0-9_])" if label[-1].isalnum() else ""
            return re.search(
                prefix + escaped + suffix,
                quantity.evidence_quote,
                flags=re.IGNORECASE,
            ) is not None

        for index, entry in enumerate(self.critical_requirements.param_glossary):
            labels = {entry.name, *entry.aliases}
            matched = sorted(
                name for name, quantity in bound.items()
                if any(
                    evaluation_protocol_label_matches(quantity, label)
                    for label in labels
                )
            )
            if not matched:
                unresolved = sorted(
                    name for name, quantity in bound.items()
                    if any(
                        _label_appears_in_protocol_evidence(label, quantity)
                        for label in labels
                    )
                )
                if unresolved:
                    raise ValueError(
                        "critical_requirements.param_glossary["
                        f"{index}] labels {sorted(labels)!r} appear verbatim "
                        "in evaluation-protocol evidence for carrier(s) "
                        f"{unresolved!r}, but are absent from their "
                        "paper_names identity bridge; preserve this unresolved "
                        "legacy/typed overlap for analyzer correction"
                    )
            if len(matched) > 1:
                roles = [bound[name].role.value for name in matched]
                raise ValueError(
                    "critical_requirements.param_glossary["
                    f"{index}] names {sorted(labels)!r}, which joins multiple "
                    "comparison.evaluation_protocol carriers "
                    f"{matched!r} with roles {roles!r}; preserve the role "
                    "conflict instead of selecting a value"
                )
            if not matched:
                continue
            parameter_name = matched[0]
            quantity = bound[parameter_name]
            glossary_labels[parameter_name].update(labels)
            if (
                quantity.paper_value_status == PaperValueStatus.paper_unspecified
                and entry.paper_value is not None
            ):
                raise ValueError(
                    "critical_requirements.param_glossary["
                    f"{index}].paper_value={entry.paper_value!r} conflicts "
                    "with comparison.evaluation_protocol.quantities["
                    f"{quantity_index[parameter_name]}] role="
                    f"{quantity.role.value!r}, unit={quantity.unit!r}, "
                    f"granularity={quantity.granularity!r}, "
                    "paper_value_status='paper_unspecified', value=null; "
                    "nearby notation cannot populate this carrier"
                )
            if (
                quantity.paper_value_status == PaperValueStatus.paper_stated
                and entry.paper_value is not None
                and not _numeric_values_equal(entry.paper_value, quantity.value)
            ):
                raise ValueError(
                    "critical_requirements.param_glossary["
                    f"{index}].paper_value={entry.paper_value!r} conflicts "
                    "with comparison.evaluation_protocol.quantities["
                    f"{quantity_index[parameter_name]}] role="
                    f"{quantity.role.value!r}, value={quantity.value!r}, "
                    f"unit={quantity.unit!r}, granularity="
                    f"{quantity.granularity!r}; role-compatible structured "
                    "facts must agree exactly"
                )

        def _lane_name(value: str) -> str:
            prefix = value.split("(", 1)[0]
            return re.sub(r"[^a-z0-9]+", "_", prefix.lower()).strip("_")

        normalized_labels = {
            name: {
                _lane_name(label)
                for label in labels
                if label not in bound[name].paper_symbols
            }
            for name, labels in glossary_labels.items()
        }
        for index, entry in enumerate(
            self.critical_requirements.scale_dependent_hyperparameters
        ):
            lane_name = _lane_name(entry.name)
            matches = sorted({
                name
                for name, quantity in bound.items()
                if evaluation_protocol_label_matches(quantity, entry.name)
                or (
                    lane_name
                    and lane_name in normalized_labels[name]
                )
            })
            if len(matches) > 1:
                raise ValueError(
                    "critical_requirements.scale_dependent_hyperparameters["
                    f"{index}].name={entry.name!r} joins multiple temporal "
                    f"protocol carriers {matches!r}; preserve the identity "
                    "conflict instead of selecting one"
                )
            if not matches:
                unresolved = sorted(
                    name for name, quantity in bound.items()
                    if _label_appears_in_protocol_evidence(entry.name, quantity)
                )
                if unresolved:
                    raise ValueError(
                        "critical_requirements.scale_dependent_hyperparameters["
                        f"{index}].name={entry.name!r} appears verbatim in "
                        "evaluation-protocol evidence for carrier(s) "
                        f"{unresolved!r}, but is absent from their paper_names "
                        "identity bridge; temporal/scale ownership is "
                        "unresolved and must halt"
                    )
                continue
            parameter_name = matches[0]
            quantity = bound[parameter_name]
            raise ValueError(
                "critical_requirements.scale_dependent_hyperparameters["
                f"{index}].name={entry.name!r} duplicates "
                "comparison.evaluation_protocol.quantities["
                f"{quantity_index[parameter_name]}] parameter_name="
                f"{parameter_name!r}, role={quantity.role.value!r}, "
                f"unit={quantity.unit!r}, granularity="
                f"{quantity.granularity!r}; temporal protocol counts belong "
                "only in the role-typed evaluation protocol, never in the "
                "data-scale calibration lane"
            )

        contract = self.methodology_replication_contract
        if contract is not None:
            assigned_labels: set[str] = set()
            for element in contract.elements:
                for text in (element.paper_evidence, element.required_behavior):
                    for match in re.finditer(
                        r"(?<![A-Za-z0-9_])"
                        r"(?P<label>[A-Za-z][A-Za-z0-9_]*)"
                        r"(?![A-Za-z0-9_])\s*"
                        r"(?:=|:|is\s+(?:set\s+)?to)\s*"
                        + _NUMBER_TOKEN,
                        text,
                    ):
                        assigned_labels.add(match.group("label"))
            for label in sorted(assigned_labels):
                unresolved = [
                    name for name, quantity in bound.items()
                    if _literal_spans(quantity.evidence_quote, label)
                    and not evaluation_protocol_label_matches(quantity, label)
                ]
                if unresolved:
                    raise ValueError(
                        "methodology_replication_contract assigns paper "
                        f"value to label {label!r}, which appears exactly in "
                        "evaluation-protocol evidence for carrier(s) "
                        f"{unresolved!r} but is absent from their explicit "
                        "paper_names/paper_symbols identity bridge; preserve "
                        "the unresolved structured ownership conflict"
                    )
            for parameter_name, quantity in bound.items():
                contract_values: list[int | float] = []
                for paper_name in glossary_labels[parameter_name]:
                    for value in methodology_contract_paper_values_for_param(
                        contract,
                        paper_name,
                        exact_label=paper_name in quantity.paper_symbols,
                    ):
                        if not any(
                            _numeric_values_equal(value, existing)
                            for existing in contract_values
                        ):
                            contract_values.append(value)
                if not contract_values:
                    continue
                if (
                    quantity.paper_value_status
                    == PaperValueStatus.paper_unspecified
                ):
                    raise ValueError(
                        "methodology_replication_contract claims paper value(s) "
                        f"{contract_values!r} for protocol carrier "
                        f"{parameter_name!r}, role={quantity.role.value!r}, "
                        "but comparison.evaluation_protocol marks that value "
                        "paper_unspecified; preserve the structured conflict "
                        "for analyzer correction"
                    )
                conflicting = [
                    value for value in contract_values
                    if not _numeric_values_equal(value, quantity.value)
                ]
                if conflicting:
                    raise ValueError(
                        "methodology_replication_contract contains conflicting "
                        f"paper value(s) {conflicting!r} for protocol carrier "
                        f"{parameter_name!r}, role={quantity.role.value!r}, "
                        f"while comparison.evaluation_protocol declares "
                        f"value={quantity.value!r}; no contract value is "
                        "selected silently"
                    )

        return self

    @model_validator(mode="after")
    def _data_setup_required_for_active_learning(self) -> "MethodSpec":
        """Cross-paradigm enforcement: the four AL-specific numeric fields in
        `critical_requirements.data_setup` must be non-null for any paradigm
        whose id starts with `active_learning/` (or is exactly
        `active_learning`), and SHOULD be null for paradigms that don't have
        an iterative label-acquisition concept.

        The "must be null for non-AL" half is not enforced here because some
        future paradigm might co-opt one of the field names with different
        semantics (e.g., a planner that runs `num_rounds` re-planning passes
        would not be AL but might want the field). The null-for-non-AL
        guidance lives in the analyzer agent's prompt and the field's
        description; this validator only enforces the strict "AL needs all
        four populated" half — which is what stage 1 needs to halt on.
        """
        # classification.id is the canonical taxonomy-served legacy id
        # ("active_learning", "active_learning/bayesian", "motion_planning")
        # — NOT the old filesystem path. See class Paradigm.
        paradigm_id = (self.comparison.classification.id or "").strip()
        is_al = (
            paradigm_id == "active_learning"
            or paradigm_id.startswith("active_learning/")
        )
        if not is_al:
            return self
        ds = self.critical_requirements.data_setup
        missing = [
            name for name in
            ("initial_labeled", "batch_size", "total_budget", "num_rounds")
            if getattr(ds, name) is None
        ]
        if missing:
            raise ValueError(
                f"Active-learning paradigm ({paradigm_id!r}) requires "
                f"critical_requirements.data_setup.{{{', '.join(missing)}}} "
                f"to be non-null integer(s). These fields describe the "
                f"paper's main benchmark configuration; AL specs must "
                f"populate all four from a single benchmark."
            )
        return self

    @model_validator(mode="after")
    def _methodology_contract_consistency(self) -> "MethodSpec":
        """Keep the new methodology-fidelity fields coherent when present."""
        contract_fields = {
            "methodology_replication_contract": self.methodology_replication_contract,
            "methodology_contract_pack": self.methodology_contract_pack,
            "replication_feasibility": self.replication_feasibility,
        }
        present = [name for name, value in contract_fields.items() if value is not None]
        if not present:
            return self
        missing = [name for name, value in contract_fields.items() if value is None]
        if missing:
            raise ValueError(
                "methodology fidelity fields must be provided together when "
                f"any are present; missing: {', '.join(missing)}"
            )

        assert self.methodology_replication_contract is not None
        assert self.methodology_contract_pack is not None
        assert self.replication_feasibility is not None

        core_ids = {element.element_id for element in self.methodology_replication_contract.core_elements()}
        pack_core_ids = set(self.methodology_contract_pack.core_methodology_element_ids)
        if pack_core_ids != core_ids:
            raise ValueError(
                "methodology_contract_pack.core_methodology_element_ids must "
                "match methodology_replication_contract core element IDs "
                f"(expected {sorted(core_ids)}, got {sorted(pack_core_ids)})"
            )

        derived = self.derive_replication_feasibility()
        assert derived is not None
        if self.replication_feasibility.verdict != derived.verdict:
            expected_approx_ids = {
                approx.element_id for approx in derived.approved_approximations
            }
            actual_approx_ids = {
                approx.element_id
                for approx in self.replication_feasibility.approved_approximations
            }
            raise ValueError(
                "replication_feasibility.verdict must match the verdict derived "
                "from methodology_replication_contract element statuses "
                f"(expected verdict {derived.verdict.value}, got "
                f"{self.replication_feasibility.verdict.value}; expected "
                "approved_approximations element IDs "
                f"{sorted(expected_approx_ids)}, got {sorted(actual_approx_ids)})"
            )

        expected_blocker_ids = {blocker.element_id for blocker in derived.blockers}
        actual_blocker_ids = {blocker.element_id for blocker in self.replication_feasibility.blockers}
        if actual_blocker_ids != expected_blocker_ids:
            raise ValueError(
                "replication_feasibility.blockers must match core not_replicable "
                f"elements (expected {sorted(expected_blocker_ids)}, got {sorted(actual_blocker_ids)})"
            )

        expected_approx_ids = {approx.element_id for approx in derived.approved_approximations}
        actual_approx_ids = {
            approx.element_id for approx in self.replication_feasibility.approved_approximations
        }
        if actual_approx_ids != expected_approx_ids:
            raise ValueError(
                "replication_feasibility.approved_approximations must match "
                "faithful_approximation_allowed elements "
                f"(expected {sorted(expected_approx_ids)}, got {sorted(actual_approx_ids)})"
            )

        return self

    @model_validator(mode="after")
    def _paper_truth_training_values_match_methodology_contract(self) -> "MethodSpec":
        """Keep structured paper-truth fields aligned with contract evidence.

        `critical_requirements.training.mc_samples` is consumed by Stage 2.x as
        the paper value. If the methodology contract explicitly says the paper
        uses one value but allows a demo reduction to another value, the
        critical requirement must retain the paper value; demo defaults belong
        in the pluggable signature/default params and approved approximations.
        """
        contract = self.methodology_replication_contract
        if contract is None:
            return self

        training = self.critical_requirements.training
        param_values = {
            "mc_samples": training.mc_samples,
        }
        for param_name, structured_value in param_values.items():
            if structured_value is None:
                continue
            contract_values = methodology_contract_paper_values_for_param(
                contract, param_name,
            )
            unique_values: list[int | float] = []
            for value in contract_values:
                if not any(_numeric_values_equal(value, existing) for existing in unique_values):
                    unique_values.append(value)
            if not unique_values:
                continue
            if len(unique_values) > 1:
                raise ValueError(
                    "methodology_replication_contract contains conflicting "
                    f"paper values for {param_name}: {unique_values}"
                )
            contract_value = unique_values[0]
            if not _numeric_values_equal(structured_value, contract_value):
                raise ValueError(
                    f"critical_requirements.training.{param_name} must contain "
                    "the paper-stated value because Stage 2.x uses it as "
                    f"params.{param_name}.paper_value; methodology contract "
                    f"evidence indicates paper value {contract_value!r}, but "
                    f"critical_requirements.training.{param_name} is "
                    f"{structured_value!r}. Put demo-scale defaults in "
                    "comparison.pluggable_component.signature or "
                    "acceptable_approximations, not in the paper-truth "
                    "training field."
                )

        return self
