"""Pure semantic resolution for architecture-contract version 2.

This module deliberately imports no generated package.  It resolves semantic
dimensions, checked arithmetic, exact bundle facts, and typed value shapes
before Stage 2.d can execute producer code.  Legacy 1.0/1.1 contracts are read
through their existing model and are never normalized into invented v2
identities.
"""

from __future__ import annotations

import json
from functools import reduce
from operator import mul
from typing import Any, Iterable, Iterator, Literal, Mapping

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    StrictInt,
    field_validator,
    model_validator,
)

from schemas.arch_contract import ArchContract
from schemas.arch_contract_v2 import (
    ArchContractV2,
    ClassIdConstraint,
    ConstructorValue,
    DType,
    LiteralConstructorValue,
    DimensionExpression,
    DimensionUse,
    IndexConstraint,
    MaskConstraint,
    MultiTypedFamilyComponent,
    NDArrayDescriptor,
    OpaqueDescriptor,
    ScalarDimensionSource,
    ScalarDescriptor,
    SingleTypedFamilyComponent,
    TensorDescriptor,
    TypedValueDescriptor,
)


IssueCode = Literal[
    "incomplete_generated_contract",
    "unsupported_validator_feature",
    "contract_code_disagreement",
    "bundle_contract_disagreement",
]
IssueOwner = Literal["producer", "pipeline"]
PIPELINE_OWNED_CODES = frozenset({"unsupported_validator_feature"})
DEFAULT_MAX_SYNTHETIC_ELEMENTS = 1_000_000
DEFAULT_MAX_SYNTHETIC_RANK = 32
SEMANTIC_ISSUE_PREFIX = "R2C_SEMANTIC_ISSUE:"


class SemanticIssue(BaseModel):
    """Closed issue envelope whose owner is derived from its code."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1.0"] = "1.0"
    code: IssueCode
    owner: IssueOwner | None = None
    message: str = Field(min_length=1)
    roots: list[str] = Field(min_length=1)
    values: dict[str, JsonValue] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _owner_follows_code(self) -> "SemanticIssue":
        expected: IssueOwner = (
            "pipeline" if self.code in PIPELINE_OWNED_CODES else "producer"
        )
        if self.owner is not None and self.owner != expected:
            raise ValueError(
                f"issue code {self.code!r} is owned by {expected!r}, not "
                f"{self.owner!r}"
            )
        self.owner = expected
        if any(not root or root != root.strip() for root in self.roots):
            raise ValueError("semantic issue roots must be non-blank and stripped")
        return self


def format_semantic_issue(issue: SemanticIssue) -> str:
    """Serialize one issue on the validator-to-driver wire."""
    return SEMANTIC_ISSUE_PREFIX + issue.model_dump_json(exclude_none=True)


def parse_semantic_issue(message: str) -> SemanticIssue | None:
    """Parse one exact wire message; arbitrary surrounding text is untrusted."""
    if not message.startswith(SEMANTIC_ISSUE_PREFIX):
        return None
    try:
        payload = json.loads(message[len(SEMANTIC_ISSUE_PREFIX):])
        return SemanticIssue.model_validate(payload)
    except (json.JSONDecodeError, ValueError):
        return None


def semantic_issue_exit_code(
    messages: Iterable[str], *, trust_pipeline_ownership: bool = True
) -> int:
    """Return 0 for clean, 3 for trusted pipeline-only, otherwise 1.

    Callers that can contain generated text must disable pipeline ownership
    unless they first wrap that text in a validator-created producer issue.
    """
    materialized = list(messages)
    if not materialized:
        return 0
    if not trust_pipeline_ownership:
        return 1
    parsed = [parse_semantic_issue(message) for message in materialized]
    if all(issue is not None for issue in parsed) and all(
        issue.owner == "pipeline" for issue in parsed if issue is not None
    ):
        return 3
    return 1


class BundleFact(BaseModel):
    model_config = ConfigDict(extra="forbid")

    value: StrictInt = Field(gt=0)
    root: str = Field(min_length=1)

    @field_validator("root")
    @classmethod
    def _root_is_exact(cls, value: str) -> str:
        if value != value.strip():
            raise ValueError("bundle fact root must be non-blank and stripped")
        return value


class ResolvedDimension(BaseModel):
    model_config = ConfigDict(extra="forbid")

    value: StrictInt = Field(gt=0)
    source: Literal["declared", "bundle"]
    evidence_roots: list[str] = Field(min_length=1)

    @field_validator("evidence_roots")
    @classmethod
    def _evidence_roots_are_exact(cls, values: list[str]) -> list[str]:
        if any(not value or value != value.strip() for value in values):
            raise ValueError(
                "resolved dimension evidence roots must be non-blank and stripped"
            )
        return values


class ResolutionResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dimensions: dict[str, ResolvedDimension]
    issues: list[SemanticIssue]


class DescriptorResolution(BaseModel):
    model_config = ConfigDict(extra="forbid")

    shape: tuple[int, ...] | None
    issues: list[SemanticIssue]


class ResolvedFixtureSpec(BaseModel):
    """Portable, fully resolved plan for one typed contract value.

    ``constraint_upper_bound`` is exclusive: index and class-id values must be
    greater than or equal to zero and strictly less than this value.  Opaque
    values remain in the plan so traversal is auditable, but are intentionally
    not synthesizable.
    """

    model_config = ConfigDict(extra="forbid")

    kind: Literal["tensor", "ndarray", "scalar", "opaque"]
    dtype: DType | None = None
    device: Literal["cpu"] | None = None
    shape: tuple[int, ...] | None = None
    scalar_value: JsonValue | None = None
    opaque: bool = False
    synthesizable: bool
    constraint: dict[str, JsonValue] | None = None
    constraint_upper_bound: StrictInt | None = Field(default=None, gt=0)


class ContractSemanticResolution(BaseModel):
    """One semantic pass shared by every version-2 runtime consumer."""

    model_config = ConfigDict(extra="forbid")

    dimensions: dict[str, ResolvedDimension]
    constructor_args: dict[str, dict[str, JsonValue]]
    fixtures: dict[str, ResolvedFixtureSpec]
    issues: list[SemanticIssue]


AnyArchContract = ArchContract | ArchContractV2


def load_arch_contract(raw: Mapping[str, Any]) -> AnyArchContract:
    """Read v1.0/v1.1 or v2 without laundering legacy shape strings.

    A missing version remains the historical 1.0 read path.  Unknown versions
    fail explicitly instead of being accepted by the nearest model.
    """
    version = raw.get("schema_version", "1.0.0")
    if not isinstance(version, str):
        raise ValueError(f"unsupported arch_contract schema_version {version!r}")
    if version in {"1.0.0", "1.1.0"}:
        return ArchContract.model_validate(dict(raw))
    if version == "2.0.0":
        return ArchContractV2.model_validate(dict(raw))
    raise ValueError(f"unsupported arch_contract schema_version {version!r}")


def bundle_dimension_facts(manifest: Mapping[str, Any]) -> dict[str, BundleFact]:
    """Extract only bundle facts the current provenance actually measures.

    Today that vocabulary has one identity: ``time_axis_steps``.  Feature
    widths are intentionally absent; callers cannot infer them from column
    counts, display symbols, or filenames.
    """
    best: tuple[int, str] | None = None
    files = manifest.get("files")
    if not isinstance(files, list):
        return {}
    for index, entry in enumerate(files):
        if not isinstance(entry, Mapping):
            continue
        axis = entry.get("time_axis")
        if not isinstance(axis, Mapping):
            continue
        value = axis.get("steps_kept")
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            continue
        root = f"PROVENANCE.json#files[{index}].time_axis.steps_kept"
        if best is None or value > best[0]:
            best = (value, root)
    if best is None:
        return {}
    return {"time_axis_steps": BundleFact(value=best[0], root=best[1])}


def _issue(
    code: IssueCode,
    message: str,
    *roots: str,
    values: Mapping[str, JsonValue] | None = None,
) -> SemanticIssue:
    return SemanticIssue(
        code=code,
        message=message,
        roots=list(roots),
        values=dict(values or {}),
    )


class _DimensionResolver:
    def __init__(
        self,
        contract: ArchContractV2,
        bundle_facts: Mapping[str, BundleFact],
    ) -> None:
        self.contract = contract
        self.bundle_facts = bundle_facts
        self.resolved: dict[str, ResolvedDimension] = {}
        self.issues: list[SemanticIssue] = []
        self._failed: set[str] = set()

    def resolve_all(self) -> ResolutionResult:
        for identity in self.contract.dimensions:
            self._resolve_identity(identity, ())
        return ResolutionResult(dimensions=self.resolved, issues=self.issues)

    def _resolve_identity(
        self,
        identity: str,
        stack: tuple[str, ...],
        reference_root: str | None = None,
    ) -> ResolvedDimension | None:
        if identity in self.resolved:
            return self.resolved[identity]
        if identity in self._failed:
            if reference_root is not None:
                self.issues.append(
                    _issue(
                        "incomplete_generated_contract",
                        f"{reference_root} references unresolved semantic dimension "
                        f"{identity!r}",
                        reference_root,
                        f"dimensions.{identity}",
                    )
                )
            return None
        root = f"dimensions.{identity}"
        if identity in stack:
            cycle = [*stack[stack.index(identity):], identity]
            self.issues.append(
                _issue(
                    "incomplete_generated_contract",
                    "dimension registry contains a cycle: " + " -> ".join(cycle),
                    root,
                    values={"cycle": cycle},
                )
            )
            self._failed.update(cycle)
            return None
        definition = self.contract.dimensions.get(identity)
        if definition is None:
            roots = (
                (reference_root, root)
                if reference_root is not None and reference_root != root
                else (root,)
            )
            self.issues.append(
                _issue(
                    "incomplete_generated_contract",
                    f"dimension reference {identity!r} is absent from the registry",
                    *roots,
                )
            )
            self._failed.add(identity)
            return None

        declared = self._eval_expression(
            definition.expression, (*stack, identity), f"{root}.expression"
        )
        if declared is None:
            self._failed.add(identity)
            return None

        binding = definition.bundle_binding
        if binding is None:
            result = ResolvedDimension(
                value=declared,
                source="declared",
                evidence_roots=[f"{root}.expression"],
            )
            self.resolved[identity] = result
            return result

        fact = self.bundle_facts.get(identity)
        binding_root = f"{root}.bundle_binding"
        if fact is None:
            self.issues.append(
                _issue(
                    "incomplete_generated_contract",
                    f"dimension {identity!r} requires bundle fact "
                    f"{identity!r}, but that exact measurement is absent",
                    root,
                    binding_root,
                )
            )
            self._failed.add(identity)
            return None
        if binding.policy == "must_match" and fact.value != declared:
            self.issues.append(
                _issue(
                    "bundle_contract_disagreement",
                    f"dimension {identity!r} declares {declared} but exact "
                    f"bundle fact {identity!r} is {fact.value}",
                    root,
                    fact.root,
                    values={"declared": declared, "measured": fact.value},
                )
            )
            self._failed.add(identity)
            return None
        result = ResolvedDimension(
            value=fact.value,
            source="bundle",
            evidence_roots=[f"{root}.expression", binding_root, fact.root],
        )
        self.resolved[identity] = result
        return result

    def _eval_expression(
        self,
        expression: DimensionExpression,
        stack: tuple[str, ...],
        root: str,
    ) -> int | None:
        if expression.kind == "literal":
            return expression.value
        if expression.kind == "reference":
            referenced = self._resolve_identity(
                expression.dimension, stack, reference_root=root
            )
            return referenced.value if referenced is not None else None
        if expression.kind in {"add", "multiply"}:
            values: list[int] = []
            for index, operand in enumerate(expression.operands or []):
                value = self._eval_expression(
                    operand, stack, f"{root}.operands[{index}]"
                )
                if value is None:
                    return None
                values.append(value)
            result = sum(values) if expression.kind == "add" else reduce(mul, values, 1)
        else:
            numerator = self._eval_expression(
                expression.numerator, stack, f"{root}.numerator"  # type: ignore[arg-type]
            )
            divisor = self._eval_expression(
                expression.divisor, stack, f"{root}.divisor"  # type: ignore[arg-type]
            )
            if numerator is None or divisor is None:
                return None
            quotient, remainder = divmod(numerator, divisor)
            if remainder:
                self.issues.append(
                    _issue(
                        "incomplete_generated_contract",
                        "exact_divide is not integral: "
                        f"{numerator} / {divisor} leaves remainder {remainder}",
                        root,
                        values={
                            "numerator": numerator,
                            "divisor": divisor,
                            "remainder": remainder,
                        },
                    )
                )
                return None
            result = quotient
        if result <= 0:
            self.issues.append(
                _issue(
                    "incomplete_generated_contract",
                    f"dimension expression resolved to non-positive value {result}",
                    root,
                    values={"resolved": result},
                )
            )
            return None
        return result


def resolve_dimensions(
    contract: ArchContractV2,
    bundle_facts: Mapping[str, BundleFact] | None = None,
) -> ResolutionResult:
    return _DimensionResolver(contract, bundle_facts or {}).resolve_all()


def _dimension_value(
    use: DimensionUse,
    resolved: Mapping[str, ResolvedDimension],
    *,
    root: str,
) -> tuple[int | None, list[SemanticIssue]]:
    value = resolved.get(use.dimension)
    if value is not None:
        return value.value, []
    return None, [
        _issue(
            "incomplete_generated_contract",
            f"{root} references unresolved semantic dimension {use.dimension!r}; "
            f"display symbol {use.display_symbol!r} has no binding authority",
            root,
            f"dimensions.{use.dimension}",
        )
    ]


def resolve_constructor_value(
    value: ConstructorValue,
    resolved: Mapping[str, ResolvedDimension],
    *,
    root: str,
) -> tuple[JsonValue | int | None, list[SemanticIssue]]:
    if isinstance(value, LiteralConstructorValue):
        return value.literal, []
    return _dimension_value(value.dimension, resolved, root=root)


def resolve_descriptor(
    descriptor: TypedValueDescriptor,
    resolved: Mapping[str, ResolvedDimension],
    *,
    root: str,
    max_elements: int = DEFAULT_MAX_SYNTHETIC_ELEMENTS,
    max_rank: int = DEFAULT_MAX_SYNTHETIC_RANK,
) -> DescriptorResolution:
    """Resolve a descriptor's concrete shape and narrow role constraints."""
    if isinstance(descriptor, OpaqueDescriptor):
        return DescriptorResolution(shape=None, issues=[])
    if isinstance(descriptor, ScalarDescriptor):
        if isinstance(descriptor.source, ScalarDimensionSource):
            value, issues = _dimension_value(
                descriptor.source.dimension,
                resolved,
                root=f"{root}.source.dimension",
            )
            if value is not None:
                upper = 2**31 - 1 if descriptor.dtype == "int32" else 2**63 - 1
                if value > upper:
                    issues.append(
                        _issue(
                            "incomplete_generated_contract",
                            f"dimension-backed scalar value {value} is outside "
                            f"{descriptor.dtype} range",
                            root,
                            f"dimensions.{descriptor.source.dimension.dimension}",
                            values={"resolved": value, "upper_bound": upper},
                        )
                    )
            return DescriptorResolution(shape=(), issues=issues)
        return DescriptorResolution(shape=(), issues=[])

    assert isinstance(descriptor, (TensorDescriptor, NDArrayDescriptor))
    issues: list[SemanticIssue] = []
    shape: list[int] = []
    for index, use in enumerate(descriptor.dimensions):
        value, use_issues = _dimension_value(
            use, resolved, root=f"{root}.dimensions[{index}]"
        )
        issues.extend(use_issues)
        if value is not None:
            shape.append(value)

    constraint = descriptor.constraint
    if isinstance(constraint, IndexConstraint):
        if descriptor.dtype not in {"int32", "int64"}:
            issues.append(
                _issue(
                    "incomplete_generated_contract",
                    f"index descriptor requires int32 or int64, got {descriptor.dtype}",
                    root,
                )
            )
        if constraint.indexed_dimension not in resolved:
            issues.append(
                _issue(
                    "incomplete_generated_contract",
                    "index constraint names unresolved semantic axis "
                    f"{constraint.indexed_dimension!r}",
                    f"{root}.constraint",
                    f"dimensions.{constraint.indexed_dimension}",
                )
            )
    elif isinstance(constraint, ClassIdConstraint):
        if descriptor.dtype not in {"int32", "int64"}:
            issues.append(
                _issue(
                    "incomplete_generated_contract",
                    f"class-id descriptor requires int32 or int64, got {descriptor.dtype}",
                    root,
                )
            )
        if constraint.class_count_dimension not in resolved:
            issues.append(
                _issue(
                    "incomplete_generated_contract",
                    "class-id constraint names unresolved class count "
                    f"{constraint.class_count_dimension!r}",
                    f"{root}.constraint",
                    f"dimensions.{constraint.class_count_dimension}",
                )
            )
    elif isinstance(constraint, MaskConstraint) and descriptor.dtype != "bool":
        issues.append(
            _issue(
                "incomplete_generated_contract",
                f"mask descriptor requires bool dtype, got {descriptor.dtype}",
                root,
            )
        )

    if issues:
        return DescriptorResolution(shape=None, issues=issues)
    if len(shape) > max_rank:
        return DescriptorResolution(
            shape=None,
            issues=[
                _issue(
                    "unsupported_validator_feature",
                    f"typed fixture at {root} has rank {len(shape)}, above "
                    f"validator cap {max_rank}",
                    root,
                    values={"rank": len(shape), "cap": max_rank},
                )
            ],
        )
    elements = reduce(mul, shape, 1)
    if elements > max_elements:
        return DescriptorResolution(
            shape=None,
            issues=[
                _issue(
                    "unsupported_validator_feature",
                    f"typed fixture at {root} requires {elements} elements, "
                    f"above validator cap {max_elements}",
                    root,
                    values={"elements": elements, "cap": max_elements},
                )
            ],
        )
    return DescriptorResolution(shape=tuple(shape), issues=[])


def iter_contract_descriptors(
    contract: ArchContractV2,
) -> Iterator[tuple[str, TypedValueDescriptor]]:
    """Yield every typed descriptor once under its canonical contract root."""

    for name, descriptor in contract.data_loader.load_data_returns.items():
        yield f"data_loader.load_data_returns.{name}", descriptor

    for block_name, block in contract.architecture.items():
        for name, descriptor in block.forward.input.items():
            yield f"architecture.{block_name}.forward.input.{name}", descriptor
        yield f"architecture.{block_name}.forward.output", block.forward.output
        for method_name, signature in block.additional_methods.items():
            for name, descriptor in signature.input.items():
                yield (
                    f"architecture.{block_name}.additional_methods."
                    f"{method_name}.input.{name}",
                    descriptor,
                )
            yield (
                f"architecture.{block_name}.additional_methods."
                f"{method_name}.output",
                signature.output,
            )

    for name, descriptor in contract.pluggable_component.input.items():
        yield f"pluggable_component.input.{name}", descriptor
    yield "pluggable_component.output", contract.pluggable_component.output

    if contract.training_loop is not None:
        for name, descriptor in contract.training_loop.input.items():
            yield f"training_loop.input.{name}", descriptor
        if contract.training_loop.output is not None:
            yield "training_loop.output", contract.training_loop.output

    for name, descriptor in (contract.optimizer_state or {}).items():
        yield f"optimizer_state.{name}", descriptor

    for component_name, component in contract.family_components.items():
        if isinstance(component, SingleTypedFamilyComponent):
            yield f"family_components.{component_name}.value", component.value
            continue
        assert isinstance(component, MultiTypedFamilyComponent)
        for entry_name, descriptor in component.entries.items():
            yield (
                f"family_components.{component_name}.entries.{entry_name}",
                descriptor,
            )


def _fixture_spec(
    descriptor: TypedValueDescriptor,
    descriptor_resolution: DescriptorResolution,
    resolved: Mapping[str, ResolvedDimension],
) -> ResolvedFixtureSpec:
    if isinstance(descriptor, OpaqueDescriptor):
        return ResolvedFixtureSpec(
            kind="opaque",
            opaque=True,
            synthesizable=False,
        )

    constraint = getattr(descriptor, "constraint", None)
    constraint_payload = (
        constraint.model_dump(exclude_none=True) if constraint is not None else None
    )
    constraint_upper_bound: int | None = None
    if isinstance(constraint, IndexConstraint):
        bound = resolved.get(constraint.indexed_dimension)
        if bound is not None:
            constraint_upper_bound = bound.value
    elif isinstance(constraint, ClassIdConstraint):
        bound = resolved.get(constraint.class_count_dimension)
        if bound is not None:
            constraint_upper_bound = bound.value

    if isinstance(descriptor, ScalarDescriptor):
        if isinstance(descriptor.source, ScalarDimensionSource):
            source = resolved.get(descriptor.source.dimension.dimension)
            scalar_value: JsonValue | None = (
                source.value if source is not None else None
            )
        else:
            scalar_value = descriptor.source.literal
        return ResolvedFixtureSpec(
            kind="scalar",
            dtype=descriptor.dtype,
            shape=descriptor_resolution.shape,
            scalar_value=scalar_value,
            synthesizable=not descriptor_resolution.issues,
            constraint=constraint_payload,
            constraint_upper_bound=constraint_upper_bound,
        )

    assert isinstance(descriptor, (TensorDescriptor, NDArrayDescriptor))
    return ResolvedFixtureSpec(
        kind=descriptor.kind,
        dtype=descriptor.dtype,
        device=(descriptor.device if isinstance(descriptor, TensorDescriptor) else None),
        shape=descriptor_resolution.shape,
        synthesizable=not descriptor_resolution.issues,
        constraint=constraint_payload,
        constraint_upper_bound=constraint_upper_bound,
    )


def resolve_contract(
    contract: ArchContractV2,
    bundle_facts: Mapping[str, BundleFact] | None = None,
    *,
    max_elements: int = DEFAULT_MAX_SYNTHETIC_ELEMENTS,
    max_rank: int = DEFAULT_MAX_SYNTHETIC_RANK,
) -> ContractSemanticResolution:
    """Resolve all constructor and value surfaces in one deterministic pass."""

    dimension_result = resolve_dimensions(contract, bundle_facts)
    dimensions = dimension_result.dimensions
    issues = list(dimension_result.issues)

    constructor_args: dict[str, dict[str, JsonValue]] = {}
    for block_name, block in contract.architecture.items():
        block_args: dict[str, JsonValue] = {}
        for arg_name, declaration in block.constructor_args.items():
            root = f"architecture.{block_name}.constructor_args.{arg_name}"
            value, value_issues = resolve_constructor_value(
                declaration,
                dimensions,
                root=root,
            )
            issues.extend(value_issues)
            if not value_issues:
                block_args[arg_name] = value
        constructor_args[block_name] = block_args

    fixtures: dict[str, ResolvedFixtureSpec] = {}
    for root, descriptor in iter_contract_descriptors(contract):
        descriptor_resolution = resolve_descriptor(
            descriptor,
            dimensions,
            root=root,
            max_elements=max_elements,
            max_rank=max_rank,
        )
        issues.extend(descriptor_resolution.issues)
        fixtures[root] = _fixture_spec(
            descriptor,
            descriptor_resolution,
            dimensions,
        )

    return ContractSemanticResolution(
        dimensions=dimensions,
        constructor_args=constructor_args,
        fixtures=fixtures,
        issues=issues,
    )
