"""Typed architecture-contract schema used by new Stage-2 producers.

Version 2 separates semantic dimension identity from local display symbols,
represents derived dimensions as a closed expression tree, and describes each
runtime value's container and dtype.  The legacy 1.0/1.1 schema remains in
``schemas.arch_contract`` for archived reads and resumable runs.
"""

from __future__ import annotations

import math

from typing import Annotated, Literal, TypeAlias

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    StrictInt,
    field_validator,
    model_validator,
)

from schemas.relational_indexing import RelationalIndexing


SCHEMA_VERSION = "2.0.0"


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


def _nonblank(value: str, *, field_name: str) -> str:
    if not value or value != value.strip():
        raise ValueError(f"{field_name} must be non-blank and already stripped")
    return value


class BundleDimensionBinding(_Strict):
    """Bind the registry identity to the same-named exact bundle fact."""

    policy: Literal["override_fixture", "must_match"]


class LiteralDimensionExpression(_Strict):
    kind: Literal["literal"]
    value: StrictInt = Field(gt=0)


class ReferenceDimensionExpression(_Strict):
    kind: Literal["reference"]
    dimension: str

    @field_validator("dimension")
    @classmethod
    def _dimension_is_exact(cls, value: str) -> str:
        return _nonblank(value, field_name="dimension reference")


class AddDimensionExpression(_Strict):
    kind: Literal["add"]
    operands: list["DimensionExpression"] = Field(min_length=2)


class MultiplyDimensionExpression(_Strict):
    kind: Literal["multiply"]
    operands: list["DimensionExpression"] = Field(min_length=2)


class ExactDivideDimensionExpression(_Strict):
    kind: Literal["exact_divide"]
    numerator: "DimensionExpression"
    divisor: "DimensionExpression"


DimensionExpression: TypeAlias = Annotated[
    LiteralDimensionExpression
    | ReferenceDimensionExpression
    | AddDimensionExpression
    | MultiplyDimensionExpression
    | ExactDivideDimensionExpression,
    Field(discriminator="kind"),
]

for _expression_model in (
    AddDimensionExpression,
    MultiplyDimensionExpression,
    ExactDivideDimensionExpression,
):
    _expression_model.model_rebuild(
        _types_namespace={"DimensionExpression": DimensionExpression}
    )


class DimensionDefinition(_Strict):
    expression: DimensionExpression
    bundle_binding: BundleDimensionBinding | None = None


class DimensionUse(_Strict):
    """A semantic reference plus optional presentation-only local symbol."""

    dimension: str
    display_symbol: str | None = None

    @field_validator("dimension")
    @classmethod
    def _dimension_is_exact(cls, value: str) -> str:
        return _nonblank(value, field_name="dimension use")

    @field_validator("display_symbol")
    @classmethod
    def _display_symbol_is_not_blank(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _nonblank(value, field_name="display_symbol")


DType: TypeAlias = Literal["float32", "float64", "int32", "int64", "bool"]


class IndexConstraint(_Strict):
    kind: Literal["index"]
    indexed_dimension: str

    @field_validator("indexed_dimension")
    @classmethod
    def _indexed_dimension_is_exact(cls, value: str) -> str:
        return _nonblank(value, field_name="indexed_dimension")


class ClassIdConstraint(_Strict):
    kind: Literal["class_id"]
    class_count_dimension: str

    @field_validator("class_count_dimension")
    @classmethod
    def _class_dimension_is_exact(cls, value: str) -> str:
        return _nonblank(value, field_name="class_count_dimension")


class MaskConstraint(_Strict):
    kind: Literal["mask"]


ValueConstraint: TypeAlias = Annotated[
    IndexConstraint | ClassIdConstraint | MaskConstraint,
    Field(discriminator="kind"),
]


class TensorDescriptor(_Strict):
    kind: Literal["tensor"]
    dtype: DType
    dimensions: list[DimensionUse]
    device: Literal["cpu"] = "cpu"
    constraint: ValueConstraint | None = None


class NDArrayDescriptor(_Strict):
    kind: Literal["ndarray"]
    dtype: DType
    dimensions: list[DimensionUse]
    constraint: ValueConstraint | None = None


class ScalarLiteralSource(_Strict):
    literal: JsonValue
    display_symbol: str | None = None

    @field_validator("display_symbol")
    @classmethod
    def _display_symbol_is_not_blank(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _nonblank(value, field_name="display_symbol")

class ScalarDimensionSource(_Strict):
    dimension: DimensionUse


ScalarSource: TypeAlias = ScalarLiteralSource | ScalarDimensionSource


class ScalarDescriptor(_Strict):
    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "allOf": [
                {
                    "if": {
                        "properties": {"dtype": {"const": "bool"}},
                        "required": ["dtype"],
                    },
                    "then": {
                        "properties": {
                            "source": {
                                "type": "object",
                                "properties": {
                                    "literal": {"type": "boolean"},
                                    "display_symbol": {
                                        "anyOf": [
                                            {"type": "string", "minLength": 1},
                                            {"type": "null"},
                                        ],
                                        "default": None,
                                    },
                                },
                                "required": ["literal"],
                                "additionalProperties": False,
                            }
                        }
                    },
                },
                *(
                    {
                        "if": {
                            "properties": {"dtype": {"const": dtype}},
                            "required": ["dtype"],
                        },
                        "then": {
                            "properties": {
                                "source": {
                                    "oneOf": [
                                        {
                                            "type": "object",
                                            "properties": {
                                                "literal": {
                                                    "type": "integer",
                                                    "minimum": lower,
                                                    "maximum": upper,
                                                },
                                                "display_symbol": {
                                                    "anyOf": [
                                                        {
                                                            "type": "string",
                                                            "minLength": 1,
                                                        },
                                                        {"type": "null"},
                                                    ],
                                                    "default": None,
                                                },
                                            },
                                            "required": ["literal"],
                                            "additionalProperties": False,
                                        },
                                        {
                                            "type": "object",
                                            "properties": {
                                                "dimension": {
                                                    "type": "object",
                                                    "properties": {
                                                        "dimension": {
                                                            "type": "string",
                                                            "minLength": 1,
                                                        },
                                                        "display_symbol": {
                                                            "anyOf": [
                                                                {
                                                                    "type": "string",
                                                                    "minLength": 1,
                                                                },
                                                                {"type": "null"},
                                                            ],
                                                            "default": None,
                                                        },
                                                    },
                                                    "required": ["dimension"],
                                                    "additionalProperties": False,
                                                }
                                            },
                                            "required": ["dimension"],
                                            "additionalProperties": False,
                                        },
                                    ]
                                }
                            }
                        },
                    }
                    for dtype, lower, upper in (
                        ("int32", -(2**31), 2**31 - 1),
                        ("int64", -(2**63), 2**63 - 1),
                    )
                ),
                *(
                    {
                        "if": {
                            "properties": {"dtype": {"const": dtype}},
                            "required": ["dtype"],
                        },
                        "then": {
                            "properties": {
                                "source": {
                                    "type": "object",
                                    "properties": {
                                        "literal": {
                                            "type": "number",
                                            "minimum": -limit,
                                            "maximum": limit,
                                        },
                                        "display_symbol": {
                                            "anyOf": [
                                                {
                                                    "type": "string",
                                                    "minLength": 1,
                                                },
                                                {"type": "null"},
                                            ],
                                            "default": None,
                                        },
                                    },
                                    "required": ["literal"],
                                    "additionalProperties": False,
                                }
                            }
                        },
                    }
                    for dtype, limit in (
                        ("float32", 3.4028235e38),
                        ("float64", 1.7976931348623157e308),
                    )
                ),
            ]
        },
    )

    kind: Literal["scalar"]
    dtype: DType
    source: ScalarSource

    @model_validator(mode="after")
    def _source_matches_dtype(self) -> "ScalarDescriptor":
        if isinstance(self.source, ScalarDimensionSource):
            if self.dtype in {"int32", "int64"}:
                return self
            raise ValueError(
                "dimension-backed scalar descriptor requires int32 or int64 dtype"
            )
        literal = self.source.literal
        if literal is None:
            raise ValueError("typed scalar literal cannot be null")
        if self.dtype == "bool" and type(literal) is not bool:
            raise ValueError("bool scalar descriptor requires a boolean literal")
        if self.dtype in {"int32", "int64"}:
            if type(literal) is not int:
                raise ValueError(
                    f"{self.dtype} scalar descriptor requires an integer literal"
                )
            lower, upper = (
                (-2**31, 2**31 - 1)
                if self.dtype == "int32"
                else (-2**63, 2**63 - 1)
            )
            if not lower <= literal <= upper:
                raise ValueError(
                    f"scalar literal {literal} is outside {self.dtype} range"
                )
        if self.dtype in {"float32", "float64"}:
            limit = (
                3.4028235e38
                if self.dtype == "float32"
                else 1.7976931348623157e308
            )
            if type(literal) not in {int, float}:
                raise ValueError(
                    f"{self.dtype} scalar descriptor requires a finite numeric literal"
                )
            if isinstance(literal, float) and not math.isfinite(literal):
                raise ValueError(
                    f"{self.dtype} scalar descriptor requires a finite numeric literal"
                )
            if abs(literal) > limit:
                raise ValueError(
                    f"scalar literal {literal} is outside {self.dtype} finite range"
                )
        return self


class OpaqueDescriptor(_Strict):
    kind: Literal["opaque"]
    type_description: str
    reason: str

    @field_validator("type_description", "reason")
    @classmethod
    def _opaque_text_is_exact(cls, value: str) -> str:
        return _nonblank(value, field_name="opaque descriptor text")


TypedValueDescriptor: TypeAlias = Annotated[
    TensorDescriptor | NDArrayDescriptor | ScalarDescriptor | OpaqueDescriptor,
    Field(discriminator="kind"),
]


class SingleTypedFamilyComponent(_Strict):
    value: TypedValueDescriptor


class MultiTypedFamilyComponent(_Strict):
    entries: dict[str, TypedValueDescriptor] = Field(min_length=1)


TypedFamilyComponentBlock: TypeAlias = (
    SingleTypedFamilyComponent | MultiTypedFamilyComponent
)


class LiteralConstructorValue(_Strict):
    literal: JsonValue


class DimensionConstructorValue(_Strict):
    dimension: DimensionUse


ConstructorValue: TypeAlias = LiteralConstructorValue | DimensionConstructorValue


class TypedCallableSignature(_Strict):
    input: dict[str, TypedValueDescriptor]
    output: TypedValueDescriptor


class ArchBlockV2(_Strict):
    class_name: str = Field(min_length=1)
    constructor_args: dict[str, ConstructorValue]
    forward: TypedCallableSignature
    additional_methods: dict[str, TypedCallableSignature] = Field(
        default_factory=dict
    )

    @model_validator(mode="after")
    def _constructor_names_are_keywords(self) -> "ArchBlockV2":
        invalid = sorted(
            name for name in self.constructor_args if not name.isidentifier()
        )
        if invalid:
            raise ValueError(
                "constructor_args keys must be exact Python keyword names; "
                f"invalid: {invalid}"
            )
        return self


class DataLoaderBlockV2(_Strict):
    load_data_returns: dict[str, TypedValueDescriptor]


class PluggableComponentBlockV2(_Strict):
    name: str = Field(min_length=1)
    input: dict[str, TypedValueDescriptor]
    output: TypedValueDescriptor


class TrainingLoopBlockV2(_Strict):
    function_name: str = Field(min_length=1)
    input: dict[str, TypedValueDescriptor]
    output: TypedValueDescriptor | None = None


class ArchContractV2(_Strict):
    """Stage-2 architecture contract for typed semantic execution."""

    schema_version: Literal["2.0.0"]
    paradigm_id: str = Field(min_length=1)
    dimensions: dict[str, DimensionDefinition] = Field(default_factory=dict)
    data_loader: DataLoaderBlockV2
    architecture: dict[str, ArchBlockV2] = Field(min_length=1)
    pluggable_component: PluggableComponentBlockV2
    training_loop: TrainingLoopBlockV2 | None = None
    optimizer_state: dict[str, TypedValueDescriptor] | None = None
    family_components: dict[str, TypedFamilyComponentBlock] = Field(
        default_factory=dict
    )
    relational_indexing: RelationalIndexing | None = None

    @model_validator(mode="after")
    def _dimension_ids_are_exact(self) -> "ArchContractV2":
        invalid = sorted(
            identity
            for identity in self.dimensions
            if not identity or identity != identity.strip()
        )
        if invalid:
            raise ValueError(
                "dimension registry identities must be non-blank and already "
                f"stripped; invalid: {invalid}"
            )
        return self
