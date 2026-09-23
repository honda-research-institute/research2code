"""Pydantic model for `<RUN_DIR>/.pipeline/arch_contract.json`.

The architecture-coder produces this artifact at Stage 2.b. Method-coder
(Stage 2.c), parameter-deriver (Stage 2.x), and notebook-generator (Stage
3.a) consume it as the canonical record of what shapes flow through the
generated package. The Stage 2.d dry-run validator
(`scripts/validate_arch_contract_runtime.py`) verifies the code is
internally consistent with the declared shapes — catching the class of
bug where a `forward()` signature says "tensor in, tensor out" but the
function actually only handles a specific shape that downstream callers
don't satisfy.

## Why the schema is intentionally permissive

The CONTRACT'S existence is paradigm-driven (every paradigm has shapes);
the CONTRACT'S concrete fields are paper-driven (each paper picks specific
shapes). The OUTER STRUCTURE is universal:

  - data_loader  (every paradigm has a data layer)
  - architecture (every paradigm has at least one model-shaped class)
  - pluggable_component (every paradigm has a paper-specific contribution)
  - training_loop (most paradigms have a training/optimization step)

The INTERIOR varies by paradigm:

  - architecture.student + architecture.teacher       (knowledge_distillation)
  - architecture.model                                (active_learning)
  - architecture.dynamics + architecture.collision    (motion_planning)
  - pluggable_component.batch_dict_shape              (loss-fn style)
  - pluggable_component.input_shapes + output_shape   (acquisition-fn style)

Per-paradigm completeness is checked by the static build plan's
`arch_contract_requirements` (scripts/build_plan.py) +
`scripts/validate_arch_contract.py`; the old `arch_contract_schema` block
is RETIRED (R2C-049) and refused at authoring and SSOT lint time. This
file just pins the universal outer skeleton.

Families whose contract needs a top-level component the skeleton cannot
know in advance declare it in their pack and emit it under the single
typed extension container `family_components` (see FamilyComponentBlock).
The skeleton itself stays static and `extra='forbid'` stays on every
model.

Shape strings are kept as opaque strings (e.g., `"(B, 3, H, W)"`). The
dry-run validator parses them at test time, binding symbols to concrete
small integers (see `validate_arch_contract_runtime.py`'s DRY_RUN_BINDINGS).
"""

from __future__ import annotations

from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    model_validator,
)


LEGACY_SCHEMA_VERSION = "1.0.0"
SCHEMA_VERSION = "1.1.0"


ShapeString = str
"""A shape string. Conventionally `(SYM1, SYM2, ...)` where each `SYMN` is
either a positive integer literal (`3`, `9`) or a symbolic name (`B`, `H`,
`N_points`). Free-form alternatives like `'list[int] of length batch_size'`
or `'scalar int'` are also accepted for shapes that aren't tensor-shaped.
The dry-run validator parses tensor-shaped strings; it skips free-form ones."""


class ForwardSignature(BaseModel):
    """Shape contract for a single method's call signature + return.

    Despite the name (kept for back-compat with supervised-ML paradigms whose
    primary method is `forward`), this models ANY callable method's contract —
    e.g. motion_planning's `step` / `step_jacobian` / `is_in_collision`.

    `output_type` discriminates which of `output_shape` / `output_keys` /
    `output_shapes` is meaningful:
      - `tensor` → `output_shape` is set
      - `dict`   → `output_keys` is set (name → shape per key)
      - `tuple`  → `output_shapes` is set (ordered list of shapes)
      - `list`   → `output_shape` is set, free-form (e.g.
        `'list[BoundingBox3D] of length N'`). Detection-family methods
        honestly return lists of structured objects; the enum previously
        could not say so and a correct ms3d contract failed 2.d
        (overnight 2026-07-08).
      - `scalar` / `bool` / `float` / `int` → a non-tensor scalar return;
        none of the shape fields apply. Used by non-ML paradigms whose
        methods return plain Python scalars (e.g. a collision check returns
        `bool`, a distance query returns `float`).
      - `ndarray` → `output_shape` is set; the return is a numpy array.
        Pure-numpy paradigms (motion planning, classic optimization)
        honestly return np.ndarray, and the enum previously could not say
        so — a correct ROMAN25 contract was rejected at stage 2.d twice
        (researcher request, 2026-07-14). The runtime dry-run also synthesizes numpy
        inputs for ndarray-returning methods instead of torch tensors.
    """

    model_config = ConfigDict(extra="forbid")

    input: dict[str, ShapeString] = Field(
        description="Mapping from parameter name to its expected shape string."
    )
    output_type: Literal["tensor", "ndarray", "dict", "tuple", "list", "scalar", "bool", "float", "int"]
    output_shape: ShapeString | None = Field(
        default=None,
        description="Shape of the return value (required when output_type='tensor' or 'ndarray').",
    )
    output_keys: dict[str, ShapeString] | None = Field(
        default=None,
        description="Mapping from output dict key to its shape (required when output_type='dict').",
    )
    output_shapes: list[ShapeString | dict[str, ShapeString]] | None = Field(
        default=None,
        description=(
            "Ordered shapes of tuple elements (required when output_type='tuple'). "
            "Each element is either a shape string for a tensor element, or a "
            "dict mapping key→shape for a dict-typed tuple element — common for "
            "methods returning (pred_dict, features) in one forward pass."
        ),
    )


class ConstructorArg(BaseModel):
    """One exact keyword argument used to construct an architecture block.

    ``literal`` accepts any JSON value, including an explicit ``null`` or a
    list such as a hidden-layer layout. ``dimension`` names one resolvable
    integer dimension. Presence, rather than non-nullness, discriminates the
    two arms so ``{"literal": null}`` remains an honest declaration.
    """

    model_config = ConfigDict(extra="forbid")

    literal: JsonValue | None = Field(
        default=None,
        description="Exact JSON value passed to the builder keyword.",
    )
    dimension: str | None = Field(
        default=None,
        description=(
            "Exact semantic dimension name resolved before generated code is "
            "imported. Unknown dimensions fail; they never fall back to 8."
        ),
    )

    @model_validator(mode="after")
    def _exactly_one_source(self) -> "ConstructorArg":
        present = self.model_fields_set & {"literal", "dimension"}
        if len(present) != 1:
            raise ValueError(
                "constructor argument must declare exactly one of literal or "
                "dimension"
            )
        if "dimension" in present:
            if self.dimension is None or not self.dimension.strip():
                raise ValueError(
                    "constructor argument dimension must be a non-blank "
                    "semantic name"
                )
            self.dimension = self.dimension.strip()
        return self


class ArchBlock(BaseModel):
    """One model-shaped class in the architecture (student, teacher, model,
    dynamics, collision, etc.). Which keys exist under `architecture` is
    paradigm-defined; this schema is permissive."""

    model_config = ConfigDict(extra="forbid")

    class_name: str = Field(
        description="The Python class name as defined in method/model.py."
    )
    constructor_args: dict[str, ConstructorArg] | None = Field(
        default=None,
        description=(
            "Exact keyword mapping used to construct this block in schema "
            "1.1.0. An empty mapping explicitly declares a no-argument "
            "constructor. None is reserved for legacy schema 1.0.0 reads."
        ),
    )
    forward: ForwardSignature = Field(
        description="Shape contract for the class's primary forward pass."
    )
    additional_methods: dict[str, ForwardSignature] = Field(
        default_factory=dict,
        description=(
            "Optional additional callable methods with their own shape "
            "contracts (e.g., `forward_with_embedding` for batch_acquisition AL)."
        ),
    )

    @model_validator(mode="after")
    def _constructor_names_are_keywords(self) -> "ArchBlock":
        if self.constructor_args is None:
            return self
        invalid = sorted(
            name for name in self.constructor_args if not name.isidentifier()
        )
        if invalid:
            raise ValueError(
                "constructor_args keys must be exact Python keyword names; "
                f"invalid: {invalid}"
            )
        return self


class OptimizerStateEntry(BaseModel):
    """One optimizer-state slot (e.g. Adam's step / params / first_moment /
    second_moment). Slot names come from the paradigm manifest's
    `optimizer_state.required_subblocks`; each entry declares what the slot
    holds. Added for the stochastic_optimization paradigm, whose manifest
    required this block while the schema forbade it (ADAM, overnight
    2026-07-08 — the coder could not win)."""

    model_config = ConfigDict(extra="forbid")

    type: str | None = Field(
        default=None,
        description="Python/tensor type of the slot (e.g. 'int', 'Tensor').",
    )
    shape: ShapeString | None = Field(
        default=None,
        description="Shape of the slot when tensor-valued (e.g. '(P,)').",
    )
    description: str | None = Field(
        default=None,
        description="What the slot stores, in the paper's terms.",
    )


class FamilyComponentEntry(BaseModel):
    """One named slot inside a family component block (same shape as
    OptimizerStateEntry): what the slot holds, in the paper's terms."""

    model_config = ConfigDict(extra="forbid")

    type: str | None = Field(
        default=None,
        description="Python/tensor type of the slot (e.g. 'int', 'Tensor').",
    )
    shape: ShapeString | None = Field(
        default=None,
        description="Shape of the slot when tensor-valued (e.g. '(B,)').",
    )
    description: str | None = Field(
        default=None,
        description="What the slot stores, in the paper's terms.",
    )


class FamilyComponentBlock(BaseModel):
    """One pack-declared family-specific component block.

    The universal skeleton cannot know every family's top-level components
    in advance: stochastic_optimization needed `optimizer_state` (ADAM,
    2026-07-08 — hard-added as a universal field after a halt) and the
    single-agent-RL family needed `reward_function` (SRL, 2026-07-15 — the
    coder emitted it correctly and the schema rejected it twice; the failure
    class is `family_component_rejected_by_universal_schema`). This block is
    the typed value of the one extension container (`family_components`);
    the LEGAL component names come from the family's pack, enforced in both
    directions by `scripts/validate_arch_contract.py`.

    The diff between the two concrete cases fixes the shape: a component is
    either single-slot (SRL's reward_function — `type`/`shape`/`description`
    on the block itself suffice) or multi-slot (the optimizer_state shape —
    a named `entries` map, one FamilyComponentEntry per slot). Both arms are
    typed; `extra='forbid'` keeps top-level typos inside a block loud."""

    model_config = ConfigDict(extra="forbid")

    type: str | None = Field(
        default=None,
        description="Python/tensor type for a single-slot component.",
    )
    shape: ShapeString | None = Field(
        default=None,
        description="Shape for a single-slot, tensor-valued component.",
    )
    description: str | None = Field(
        default=None,
        description="What the component is, in the paper's terms.",
    )
    entries: dict[str, FamilyComponentEntry] | None = Field(
        default=None,
        description=(
            "Named slots for a multi-slot component (the optimizer_state "
            "shape: {'step': ..., 'params': ...}). None for single-slot "
            "components. Required entry names may be pinned by the pack's "
            "declaration (`required_entries`)."
        ),
    )


class DataLoaderBlock(BaseModel):
    """Shape contract for what `method/data.py`'s `load_data()` returns."""

    model_config = ConfigDict(extra="forbid")

    load_data_returns: dict[str, ShapeString] = Field(
        description=(
            "Mapping from each returned variable name to its shape. Example "
            "for classification KD: "
            "{'x_train': '(N, 3, H, W)', 'y_train': '(N,)', ...}."
        )
    )


class PluggableComponentBlock(BaseModel):
    """Shape contract for the paper's pluggable component (method.py's
    primary public function). Paradigm-shape-varying — one or the other of
    `batch_dict_shape` (loss-fn paradigms like KD) or `input_shapes` +
    `output_shape` (selection-fn paradigms like AL) is populated."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(
        description="Function name; must match spec.comparison.pluggable_component.name."
    )
    batch_dict_shape: dict[str, ShapeString] | None = Field(
        default=None,
        description=(
            "For paradigms where the pluggable takes a batch dict (KD's "
            "compute_distillation_loss receives {'inputs': ..., "
            "'teacher_inputs': ..., 'gt_boxes': ...}): map each key to its "
            "shape. None for paradigms that don't use a batch dict."
        ),
    )
    input_shapes: dict[str, ShapeString] | None = Field(
        default=None,
        description=(
            "For paradigms where the pluggable takes named tensor inputs "
            "(AL's select_batch receives x_unlabeled): map each parameter to "
            "its shape. None for batch-dict-style paradigms."
        ),
    )
    output_shape: ShapeString | None = Field(
        default=None,
        description=(
            "Output shape (single tensor or descriptive string). Examples: "
            "'list[int] of length batch_size' for AL's select_batch, "
            "'scalar tensor' for KD's compute_distillation_loss."
        ),
    )


class TrainingLoopBlock(BaseModel):
    """Shape contract for `method/training.py`'s training-loop entry point.
    Required when the paradigm has a training step the notebook calls
    (knowledge_distillation, active_learning); optional for paradigms with
    no training loop (e.g., motion_planning's solve-once shape)."""

    model_config = ConfigDict(extra="forbid")

    function_name: str = Field(
        description=(
            "The training-loop function name in method/training.py (e.g., "
            "`train_from_scratch` for AL, `train_with_distillation` for KD)."
        )
    )
    input_shapes: dict[str, ShapeString] = Field(
        description="Mapping from each input parameter name to its shape."
    )


# The relational-indexing machinery is CURRENT v2 schema shared by both
# contract versions; it lives in schemas/relational_indexing.py (extracted
# 2026-08-19, Track A decision 5). Re-exported here so v1-era importers keep
# working until the v1-only remainder of this file retires.
from schemas.relational_indexing import (  # noqa: E402,F401
    RelationalExecutionBinding,
    RelationalFittingCallBinding,
    RelationalIndexing,
    RelationalInferenceCallBinding,
    RelationalPhaseBatchModes,
    RelationalPreparationCallable,
)



class ArchContract(BaseModel):
    """Stage 2.b architecture-coder's structured output. The canonical
    record of what shapes flow through the generated method/ package.

    Paradigm-specific completeness is verified by
    `scripts/validate_arch_contract.py` against the taxonomy build plan's
    architecture contract block. The dry-run validator
    (`scripts/validate_arch_contract_runtime.py`) verifies the package's
    code is consistent with the declared shapes at runtime."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1.0.0", "1.1.0"] = Field(
        default=LEGACY_SCHEMA_VERSION,
        description=(
            "1.0.0 remains readable through the legacy construction adapter. "
            "1.1.0 records exact constructor_args for archived or resumed "
            "contracts. Current producers emit the separate typed 2.0.0 "
            "schema. A missing version retains legacy behavior."
        ),
    )
    paradigm_id: str = Field(
        description=(
            "Paradigm classifier; must match "
            "`spec.comparison.classification.id`. Used by the validator to look "
            "up the matched taxonomy build plan's architecture contract block."
        )
    )
    data_loader: DataLoaderBlock
    architecture: dict[str, ArchBlock] = Field(
        description=(
            "Paradigm-defined keys. Examples: {'student': ..., 'teacher': ...} "
            "for KD; {'model': ...} for AL. At least one key is required."
        ),
        min_length=1,
    )
    pluggable_component: PluggableComponentBlock
    training_loop: TrainingLoopBlock | None = None
    optimizer_state: dict[str, OptimizerStateEntry] | None = Field(
        default=None,
        description=(
            "Optimizer-state slots for stochastic-optimization paradigms "
            "(e.g. {'step': ..., 'params': ..., 'first_moment': ..., "
            "'second_moment': ...} for Adam). Slot names must match the "
            "paradigm manifest's optimizer_state.required_subblocks. None "
            "for paradigms without optimizer state."
        ),
    )
    family_components: dict[str, FamilyComponentBlock] | None = Field(
        default=None,
        description=(
            "Family-specific top-level component blocks beyond the universal "
            "skeleton (e.g. {'reward_function': ...} for single-agent RL). "
            "The legal component names are declared by the family's taxonomy "
            "or provisional pack and enforced in both directions by "
            "scripts/validate_arch_contract.py: a declared-required name must "
            "be present, and a present name must be declared. None for "
            "families that need no extension blocks."
        ),
    )
    relational_indexing: RelationalIndexing | None = Field(
        default=None,
        description=(
            "Conditional homogeneous-graph identity and batch-preparation "
            "contract. Required exactly when the method spec contains a "
            "paper-grounded homogeneous_graph methodology element; otherwise "
            "null."
        ),
    )

    @model_validator(mode="after")
    def _constructor_contract_matches_version(self) -> "ArchContract":
        present = [
            name
            for name, block in self.architecture.items()
            if "constructor_args" in block.model_fields_set
        ]
        if self.schema_version == SCHEMA_VERSION:
            missing = sorted(
                name
                for name, block in self.architecture.items()
                if name not in present or block.constructor_args is None
            )
            if missing:
                raise ValueError(
                    "schema 1.1.0 requires constructor_args on every "
                    f"architecture block; missing: {missing}. Use {{}} for an "
                    "intentional no-argument constructor"
                )
        elif present:
            raise ValueError(
                "constructor_args requires schema_version 1.1.0; schema "
                f"1.0.0 uses the legacy construction adapter (blocks: "
                f"{sorted(present)})"
            )
        return self
