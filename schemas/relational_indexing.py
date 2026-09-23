"""Relational-indexing contract machinery (R2C-084 lineage).

Extracted from schemas/arch_contract.py (Track A decision 5, 2026-08-19):
this block is CURRENT v2 schema shared by both contract versions — it only
happened to live in the v1 file. schemas/arch_contract_v2.py imports it
from here; schemas/arch_contract.py re-exports it so existing v1-era
importers keep working until the v1-only remainder retires.
"""

from __future__ import annotations

from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictInt,
    model_validator,
)


class RelationalPreparationCallable(BaseModel):
    """Pure graph-batch preparation seam owned by the architecture producer.

    The module is fixed in v1 because ``method/training.py`` owns batching.
    The callable accepts five keyword arguments (``source_entity_ids``,
    ``batch_entity_ids``, ``coindexed``, ``graph``, and ``degrees``) and
    returns the mapping-and-batch dictionary documented on
    :class:`RelationalIndexing`.
    """

    model_config = ConfigDict(extra="forbid")

    module: Literal["method.training"] = "method.training"
    name: str = Field(
        min_length=1,
        pattern=r"^[A-Za-z_][A-Za-z0-9_]*$",
        description=(
            "Exact top-level callable name in method/training.py. It may be "
            "private, but the declared training-loop entry point must reach it."
        ),
    )
    tensor_backend: Literal["torch", "numpy"] = Field(
        default="torch",
        description="Array family used by the callable and its graph tensors.",
    )


class RelationalPhaseBatchModes(BaseModel):
    """Entity batching policy at each scientifically distinct phase."""

    model_config = ConfigDict(extra="forbid")

    fitting: Literal[
        "canonical_full_graph",
        "coherent_full_graph_permutation",
        "induced_subgraph",
    ]
    model_selection: Literal[
        "canonical_full_graph",
        "coherent_full_graph_permutation",
        "induced_subgraph",
    ]
    inference: Literal[
        "canonical_full_graph",
        "coherent_full_graph_permutation",
        "induced_subgraph",
    ]
    reported_evaluation: Literal[
        "canonical_full_graph",
        "coherent_full_graph_permutation",
        "induced_subgraph",
    ]


def _validate_relational_binding_text(value: str, *, field_name: str) -> None:
    """Reject absent or normalized-by-guessing execution-binding names."""
    if not value or value != value.strip():
        raise ValueError(
            f"relational_indexing.execution.{field_name} must be non-blank "
            "and already stripped"
        )


def _validate_relational_binding_root(
    value: str, *, field_name: str, prefix: str
) -> None:
    _validate_relational_binding_text(value, field_name=field_name)
    if not value.startswith(prefix) or value == prefix:
        raise ValueError(
            f"relational_indexing.execution.{field_name} must be a full "
            f"canonical root below {prefix!r}; got {value!r}"
        )


class RelationalFittingCallBinding(BaseModel):
    """Exact schema-2 training-call roots used by the relational smoke."""

    model_config = ConfigDict(extra="forbid")

    model_input_root: str = Field(
        min_length=1,
        description=(
            "Full typed root for the fitting callable's constructed-model "
            "parameter, for example training_loop.input.model."
        ),
    )
    source_entity_ids_input_root: str = Field(
        min_length=1,
        description=(
            "Full typed training input root that receives canonical stable "
            "entity ids."
        ),
    )
    batch_entity_ids_input_root: str = Field(
        min_length=1,
        description=(
            "Full typed training input root that receives the requested batch's "
            "stable entity ids."
        ),
    )
    graph_input_root: str = Field(
        min_length=1,
        description=(
            "Full typed training input root that receives the canonical source "
            "graph passed into the declared preparation callable."
        ),
    )
    degree_input_root: str | None = Field(
        default=None,
        description=(
            "Full typed training input root that receives source degrees passed "
            "into the declared preparation callable, or null when "
            "relational_indexing has no degree root."
        ),
    )
    coindexed_input_roots: dict[str, str] = Field(
        description=(
            "Mapping from ordinary logical node-aligned input roots to full "
            "typed training roots carrying source-domain values. Stable ids "
            "and degrees use their dedicated fitting roots. May be empty when "
            "the fitting entry has no other direct node-aligned inputs."
        ),
    )
    one_epoch_input_root: str | None = Field(
        default=None,
        description=(
            "Full typed integer-scalar training input root that Stage 2.d "
            "overrides to one, when the fitting entry exposes an epoch count."
        ),
    )
    fitting_entry_is_one_epoch: Literal[True] | None = Field(
        default=None,
        description=(
            "Set true only when the declared fitting entry is intrinsically one "
            "epoch and therefore has no epoch-count input."
        ),
    )

    @model_validator(mode="after")
    def _fitting_roots_are_exact(self) -> "RelationalFittingCallBinding":
        for field_name in (
            "model_input_root",
            "source_entity_ids_input_root",
            "batch_entity_ids_input_root",
            "graph_input_root",
        ):
            _validate_relational_binding_root(
                getattr(self, field_name),
                field_name=f"fitting.{field_name}",
                prefix="training_loop.input.",
            )
        if self.degree_input_root is not None:
            _validate_relational_binding_root(
                self.degree_input_root,
                field_name="fitting.degree_input_root",
                prefix="training_loop.input.",
            )
        if self.one_epoch_input_root is not None:
            _validate_relational_binding_root(
                self.one_epoch_input_root,
                field_name="fitting.one_epoch_input_root",
                prefix="training_loop.input.",
            )
        for logical_root, typed_root in self.coindexed_input_roots.items():
            _validate_relational_binding_text(
                logical_root,
                field_name="fitting.coindexed_input_roots logical root",
            )
            _validate_relational_binding_root(
                typed_root,
                field_name=(
                    "fitting.coindexed_input_roots"
                    f"[{logical_root!r}]"
                ),
                prefix="training_loop.input.",
            )
        if (self.one_epoch_input_root is None) == (
            self.fitting_entry_is_one_epoch is None
        ):
            raise ValueError(
                "relational_indexing.execution.fitting must declare exactly "
                "one one-epoch control: one_epoch_input_root or "
                "fitting_entry_is_one_epoch=true"
            )
        return self


class RelationalInferenceCallBinding(BaseModel):
    """Exact schema-2 architecture-forward roots used for inference."""

    model_config = ConfigDict(extra="forbid")

    architecture_block: str = Field(
        min_length=1,
        description="Exact key of the constructed architecture block.",
    )
    graph_input_root: str = Field(
        min_length=1,
        description="Full typed architecture forward root for the inference graph.",
    )
    degree_input_root: str | None = Field(
        default=None,
        description=(
            "Full typed architecture forward root for inference degrees, or "
            "null when relational_indexing has no degree root."
        ),
    )
    coindexed_input_roots: dict[str, str] = Field(
        description=(
            "Mapping from ordinary logical node-aligned input roots to full "
            "typed architecture forward roots. Degrees use their dedicated "
            "inference root; output identity remains preparation metadata. May "
            "be empty for a graph-only forward surface."
        ),
    )
    output_coindexed_root: str = Field(
        min_length=1,
        description=(
            "Logical relational root naming the architecture forward output's "
            "entity axis. The preparation result supplies its ordered stable "
            "output ids."
        ),
    )

    @model_validator(mode="after")
    def _inference_roots_are_exact(self) -> "RelationalInferenceCallBinding":
        _validate_relational_binding_text(
            self.architecture_block,
            field_name="inference.architecture_block",
        )
        prefix = f"architecture.{self.architecture_block}.forward.input."
        _validate_relational_binding_root(
            self.graph_input_root,
            field_name="inference.graph_input_root",
            prefix=prefix,
        )
        if self.degree_input_root is not None:
            _validate_relational_binding_root(
                self.degree_input_root,
                field_name="inference.degree_input_root",
                prefix=prefix,
            )
        for logical_root, typed_root in self.coindexed_input_roots.items():
            _validate_relational_binding_text(
                logical_root,
                field_name="inference.coindexed_input_roots logical root",
            )
            _validate_relational_binding_root(
                typed_root,
                field_name=(
                    "inference.coindexed_input_roots"
                    f"[{logical_root!r}]"
                ),
                prefix=prefix,
            )
        _validate_relational_binding_text(
            self.output_coindexed_root,
            field_name="inference.output_coindexed_root",
        )
        return self


class RelationalExecutionBinding(BaseModel):
    """Closed crosswalk from relational semantics to typed runtime calls."""

    model_config = ConfigDict(extra="forbid")

    entity_dimension: str = Field(
        min_length=1,
        description=(
            "Schema-2 semantic dimension identity for the canonical source "
            "entity count."
        ),
    )
    fitting: RelationalFittingCallBinding
    inference: RelationalInferenceCallBinding

    @model_validator(mode="after")
    def _entity_dimension_is_exact(self) -> "RelationalExecutionBinding":
        _validate_relational_binding_text(
            self.entity_dimension,
            field_name="entity_dimension",
        )
        return self


class RelationalIndexing(BaseModel):
    """Identity and graph-preparation contract for one homogeneous graph.

    The preparation callable is deliberately model-free. Given canonical
    source arrays and requested stable entity ids, it returns a dictionary
    with exactly these semantic entries (additional diagnostic entries are
    allowed):

    ``local_to_source``
        Integer vector mapping each prepared row to its canonical source-row
        position.
    ``source_to_local``
        Integer vector over the source domain; selected rows map to local
        positions and excluded rows are ``-1``.
    ``coindexed``
        Every declared co-indexed root, transformed along its declared axis.
    ``graph``
        The induced/relabelled sparse edge index or dense adjacency.
    ``degrees``
        The transformed degree vector, or ``None`` when no degree root exists.
    ``output_entity_ids``
        Stable ids naming model-output rows under ``output_order``.

    This fixed seam makes identity observable without requiring a graph
    library and lets Stage 2.d test the batching code before constructing the
    model. Heterogeneous, bipartite, hypergraph, dynamic, and stochastic
    neighbor-sampling forms are outside v1.
    """

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1.0"] = "1.0"
    methodology_element_ids: list[str] = Field(
        min_length=1,
        description=(
            "Exact methodology_replication_contract element ids whose "
            "relational_structure.kind is homogeneous_graph."
        ),
    )
    entity_axis: str = Field(
        min_length=1,
        description="Paper/domain name for the shared node/entity axis.",
    )
    stable_entity_id_root: str = Field(
        min_length=1,
        description="Root carrying stable ids for the entity axis.",
    )
    graph_root: str = Field(
        min_length=1,
        description="Root carrying the source graph representation.",
    )
    representation: Literal["sparse_edge_index", "dense_adjacency"]
    source_endpoint_index_space: Literal["source_entity_axis_positions"] = (
        "source_entity_axis_positions"
    )
    prepared_endpoint_index_space: Literal["local_batch_positions"] = (
        "local_batch_positions"
    )
    edge_orientation: Literal["source_to_destination"] = (
        "source_to_destination"
    )
    coindexed_roots: dict[str, StrictInt] = Field(
        min_length=2,
        description=(
            "Root-to-axis-index map for every node-aligned tensor or row set. "
            "It must include stable_entity_id_root and, when present, "
            "degree_root. Targets, features, covariates, masks, embeddings, "
            "forecasts, actuals, baselines, and per-entity metric rows belong "
            "here whenever the package carries them."
        ),
    )
    phase_batch_modes: RelationalPhaseBatchModes
    degree_root: str | None = Field(
        default=None,
        description="Node-aligned degree/normalization root, when one exists.",
    )
    degree_kind: Literal["in_degree", "out_degree"] | None = None
    degree_semantics: Literal["source_graph", "induced_graph"] | None = None
    output_order: Literal[
        "prepared_entity_order", "canonical_source_order"
    ] = "prepared_entity_order"
    preparation_callable: RelationalPreparationCallable
    execution: RelationalExecutionBinding | None = Field(
        default=None,
        description=(
            "Optional exact schema-2 crosswalk from the model-free relational "
            "fixture to fitting and inference calls. Archived schema-1 contracts "
            "remain valid without it."
        ),
    )

    @model_validator(mode="after")
    def _relational_fields_are_coherent(self) -> "RelationalIndexing":
        ids = [item.strip() for item in self.methodology_element_ids]
        if any(not item for item in ids) or len(set(ids)) != len(ids):
            raise ValueError(
                "relational_indexing.methodology_element_ids must be non-blank "
                "and unique"
            )
        self.methodology_element_ids = ids

        normalized_roots = {root.strip(): axis for root, axis in self.coindexed_roots.items()}
        if any(not root for root in normalized_roots):
            raise ValueError("relational_indexing.coindexed_roots cannot contain blank roots")
        if any(axis < 0 for axis in normalized_roots.values()):
            raise ValueError(
                "relational_indexing.coindexed_roots axis indexes must be "
                "non-negative integers"
            )
        if len(normalized_roots) != len(self.coindexed_roots):
            raise ValueError(
                "relational_indexing.coindexed_roots collide after whitespace "
                "normalization"
            )
        self.coindexed_roots = normalized_roots

        self.entity_axis = self.entity_axis.strip()
        self.stable_entity_id_root = self.stable_entity_id_root.strip()
        self.graph_root = self.graph_root.strip()
        for field_name in (
            "entity_axis", "stable_entity_id_root", "graph_root"
        ):
            if not getattr(self, field_name):
                raise ValueError(
                    f"relational_indexing.{field_name} cannot be blank or "
                    "whitespace-only"
                )
        if self.stable_entity_id_root not in self.coindexed_roots:
            raise ValueError(
                "relational_indexing.stable_entity_id_root must appear in "
                "coindexed_roots"
            )
        if self.coindexed_roots[self.stable_entity_id_root] != 0:
            raise ValueError(
                "relational_indexing.stable_entity_id_root is a one-dimensional "
                "identity vector and must use entity axis 0"
            )
        if self.graph_root in self.coindexed_roots:
            raise ValueError(
                "relational_indexing.graph_root is structural, not a node-row "
                "root, and must not appear in coindexed_roots"
            )

        degree_fields = (
            self.degree_root,
            self.degree_kind,
            self.degree_semantics,
        )
        if any(value is not None for value in degree_fields) and not all(
            value is not None for value in degree_fields
        ):
            raise ValueError(
                "relational_indexing degree_root, degree_kind, and "
                "degree_semantics must be supplied together or all be null"
            )
        if self.degree_root is not None:
            self.degree_root = self.degree_root.strip()
            if self.degree_root not in self.coindexed_roots:
                raise ValueError(
                    "relational_indexing.degree_root must appear in "
                    "coindexed_roots"
                )
            if self.coindexed_roots[self.degree_root] != 0:
                raise ValueError(
                    "relational_indexing.degree_root is a one-dimensional "
                    "node vector and must use entity axis 0"
                )
        if self.execution is not None:
            fitting_roots = set(self.execution.fitting.coindexed_input_roots)
            inference_roots = set(self.execution.inference.coindexed_input_roots)
            unknown_fitting = sorted(fitting_roots - set(self.coindexed_roots))
            unknown_inference = sorted(inference_roots - set(self.coindexed_roots))
            unknown_output = (
                []
                if self.execution.inference.output_coindexed_root
                in self.coindexed_roots
                else [self.execution.inference.output_coindexed_root]
            )
            if unknown_fitting or unknown_inference or unknown_output:
                raise ValueError(
                    "relational_indexing.execution coindexed logical roots must "
                    "exist in relational_indexing.coindexed_roots; "
                    f"unknown fitting roots={unknown_fitting}, unknown inference "
                    f"roots={unknown_inference}, unknown output roots="
                    f"{unknown_output}"
                )

            dedicated_roots = {self.stable_entity_id_root}
            if self.degree_root is not None:
                dedicated_roots.add(self.degree_root)
            duplicate_fitting_roles = sorted(fitting_roots & dedicated_roots)
            duplicate_inference_roles = sorted(inference_roots & dedicated_roots)
            if duplicate_fitting_roles or duplicate_inference_roles:
                raise ValueError(
                    "relational_indexing.execution coindexed_input_roots must "
                    "not repeat stable-id or degree roles that have dedicated "
                    "bindings; duplicate fitting roots="
                    f"{duplicate_fitting_roles}, duplicate inference roots="
                    f"{duplicate_inference_roles}"
                )

            has_degree = self.degree_root is not None
            fitting_has_degree = (
                self.execution.fitting.degree_input_root is not None
            )
            inference_has_degree = (
                self.execution.inference.degree_input_root is not None
            )
            if fitting_has_degree != has_degree or inference_has_degree != has_degree:
                raise ValueError(
                    "relational_indexing.execution fitting.degree_input_root and "
                    "inference.degree_input_root must both be non-null exactly "
                    "when relational_indexing.degree_root is non-null"
                )
        return self
