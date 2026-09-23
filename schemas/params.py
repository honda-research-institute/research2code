"""Pydantic model for `<RUN_DIR>/.pipeline/params.json`.

The parameter-deriver (Stage 2.x) produces this file; the notebook-generator
(Stage 3) embeds its contents into `notebook.ipynb` as the `params` cell.

Each parameter records both its **provenance** AND its **runtime usage**:

Provenance — one of four sources:

  - `paper`: paper specifies both the variable's existence AND the value we
             use. Requires `paper_section` (where in the paper) and/or `note`.
  - `system_default`: paper specifies the variable and a numeric value, but
             runtime uses a different value. Requires `paper_value` (the paper's
             actual value) and `reasoning` (justification of the smoke choice).
  - `system_inferred`: the runtime numeric value is system-owned. The paper may
             omit the variable entirely or define it symbolically without
             fixing a number; typed protocol metadata distinguishes those
             cases. Requires `reasoning` (justification).
  - `spec_default`: the value comes from the analyzer's
             `pluggable_component.signature` kwarg default. The signature
             default is a runtime convenience chosen by the analyzer when no
             paper-stated value was pinned to a structured field — it may or
             may not match the paper. Requires `reasoning`. Use this rather
             than `paper` to avoid claiming paper attribution for an
             analyzer-chosen default.

Runtime usage — `used_in_notebook` flags whether the notebook actually wires
this parameter into a code cell:

  - `True` (default): the param flows into a runtime call (e.g., the AL loop
             passes `batch_size` to `select_batch`).
  - `False`: the spec implies the param but the method's bootstrap or
             algorithm doesn't need it (e.g., GBALD's `initial_labeled` —
             GBALD's bootstrap is `construct_core_set(core_set_size)`, so the
             notebook never uses initial_labeled). Requires `unused_reason`
             explaining why the param is in the dict but inert. The notebook
             still SHOWS the param so the user can audit; the provenance
             table marks it ❌ in the "Used?" column. The user should never
             have to wonder why a param is defined but never referenced —
             unused_reason answers that.

The schema enforces both the "required field per source" invariants AND the
"unused_reason required when used_in_notebook=False" invariant. A reviewer
agent later judges whether the reasoning text is paper-faithful.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


SCHEMA_VERSION = "1.3.0"


class ParamEntry(BaseModel):
    """One parameter's value, provenance, and runtime-usage status."""

    model_config = ConfigDict(extra="forbid")

    value: Any
    source: Literal["paper", "system_default", "system_inferred", "spec_default"]
    paper_section: str | None = None
    paper_value: Any | None = None
    note: str | None = None
    reasoning: str | None = None

    # The paper's own explanation of what this parameter represents,
    # copied verbatim from the spec's param_glossary (which the spec
    # validator checked against paper.md). Param-glossary design
    # 2026-07-21 — the researcher's traceability ask, second half.
    paper_says: str | None = None

    # Role-typed temporal protocol provenance (schema 1.3.0). These fields
    # travel as one unit and are populated only when the matched taxonomy
    # independently declares this params key as a protocol carrier. The
    # paper's verbatim evidence remains in paper_says; paper_section and
    # paper_element_ids keep its location and identity machine-checkable.
    protocol_role: Literal[
        "context_length",
        "forecast_call_horizon",
        "validation_span",
        "test_span",
    ] | None = None
    # Exact physical quantity from the typed protocol. ``value`` and
    # ``paper_value`` are runtime step counts; keeping the source quantity
    # separate prevents q at g units/step from masquerading as q steps.
    protocol_value: int | float | None = None
    protocol_unit: str | None = None
    protocol_granularity: int | float | None = None
    protocol_axis_says: str | None = None
    protocol_axis_section: str | None = None
    protocol_axis_element_ids: list[str] = Field(default_factory=list)
    paper_value_status: Literal[
        "paper_stated", "paper_unspecified"
    ] | None = None
    paper_element_ids: list[str] = Field(default_factory=list)

    @field_validator("protocol_granularity", mode="before")
    @classmethod
    def _protocol_granularity_is_not_boolean(cls, value: object) -> object:
        # Pydantic otherwise coerces booleans and numeric strings before the
        # model-level invariant can inspect the raw generated JSON.
        if isinstance(value, bool):
            raise ValueError(
                "protocol_granularity must be a positive number, not a boolean"
            )
        if value is not None and not isinstance(value, (int, float)):
            raise ValueError(
                "protocol_granularity must be a raw numeric JSON value, not "
                "a numeric string"
            )
        return value

    @field_validator("protocol_value", mode="before")
    @classmethod
    def _protocol_value_is_raw_numeric_or_null(cls, value: object) -> object:
        if isinstance(value, bool):
            raise ValueError(
                "protocol_value must be a positive number, not a boolean"
            )
        if value is not None and not isinstance(value, (int, float)):
            raise ValueError(
                "protocol_value must be a raw numeric JSON value or null, "
                "not a numeric string"
            )
        return value

    # Per-dataset binding (schema 1.1.0). When the spec's data_setup carries
    # per_dataset_values for this parameter and the deriver could determine
    # the demo's dataset, `value` is that dataset's paper value and
    # bound_dataset names it. None means no map applied (or the honest
    # scalar fallback — the entry's note says which).
    bound_dataset: str | None = None

    # Method-consumption disclosure (schema 1.1.0). True when derivation
    # found no consumer for this parameter in the generated method package
    # or the pluggable signature (GBALD's initial_labeled: the framework's
    # own core-set bootstrap ignores it). Purely a disclosure — never
    # demotes, never blocks.
    unused_by_method: bool | None = None

    # Runtime usage. Default True; False means the param is included for spec
    # completeness but the notebook never references it in a code cell.
    used_in_notebook: bool = True
    unused_reason: str | None = None

    @model_validator(mode="after")
    def _validate_provenance_fields(self) -> "ParamEntry":
        if self.source == "paper":
            if not self.note and not self.paper_section:
                raise ValueError(
                    "source='paper' requires either `note` or `paper_section` (or both)"
                )
        elif self.source == "system_default":
            if self.paper_value is None:
                raise ValueError(
                    "source='system_default' requires `paper_value` (the value the paper uses)"
                )
            if not self.reasoning:
                raise ValueError(
                    "source='system_default' requires `reasoning` (justification of smoke choice)"
                )
        elif self.source == "system_inferred":
            if not self.reasoning:
                raise ValueError(
                    "source='system_inferred' requires `reasoning` (justification of choice)"
                )
        elif self.source == "spec_default":
            if not self.reasoning:
                raise ValueError(
                    "source='spec_default' requires `reasoning` "
                    "(why the signature default is being propagated, since it "
                    "may not match the paper's value)"
                )

        protocol_fields = {
            "protocol_role": self.protocol_role,
            "protocol_unit": self.protocol_unit,
            "protocol_granularity": self.protocol_granularity,
            "protocol_axis_says": self.protocol_axis_says,
            "protocol_axis_section": self.protocol_axis_section,
            "paper_value_status": self.paper_value_status,
        }
        populated = [name for name, value in protocol_fields.items() if value is not None]
        if populated and len(populated) != len(protocol_fields):
            missing = [
                name for name, value in protocol_fields.items() if value is None
            ]
            raise ValueError(
                "typed protocol metadata must be provided together; missing "
                + ", ".join(missing)
            )
        if populated:
            assert self.protocol_unit is not None
            assert self.protocol_granularity is not None
            assert self.paper_value_status is not None
            if not self.protocol_unit.strip():
                raise ValueError("protocol_unit cannot be blank")
            if (
                isinstance(self.value, bool)
                or not isinstance(self.value, (int, float))
                or self.value <= 0
            ):
                raise ValueError(
                    "typed protocol runtime value must be a positive numeric "
                    "quantity, not a boolean or non-numeric value"
                )
            if (
                isinstance(self.protocol_granularity, bool)
                or self.protocol_granularity <= 0
            ):
                raise ValueError(
                    "protocol_granularity must be a positive number, not a boolean"
                )
            if (
                not self.paper_says
                or not self.paper_says.strip()
                or not self.paper_section
                or not self.paper_section.strip()
            ):
                raise ValueError(
                    "typed protocol metadata requires paper_says and "
                    "paper_section so role and evidence remain auditable"
                )
            # An EMPTY id list is the honest state when the decomposer's map
            # covers none of the quoted passage (the 2026-08-10 protocol
            # decision, mirrored from the methodology-element rule). Stage 1
            # validates that shape, so this schema must carry it: requiring
            # non-empty ids here made derive_params emit entries its own
            # validator rejected (pdfgnn 2026-08-11 stage-2x halt).
            # Auditability stays with the required paper_says/paper_section.
            if any(not item.strip() for item in self.paper_element_ids):
                raise ValueError("paper_element_ids cannot contain blank values")
            if len(set(self.paper_element_ids)) != len(self.paper_element_ids):
                raise ValueError("paper_element_ids must be unique")
            if (
                not self.protocol_axis_says
                or not self.protocol_axis_says.strip()
                or not self.protocol_axis_section
                or not self.protocol_axis_section.strip()
            ):
                raise ValueError(
                    "typed protocol metadata requires protocol_axis_says and "
                    "protocol_axis_section so unit/granularity provenance "
                    "remains auditable"
                )
            # Same honest-empty rule for the axis citation list.
            if any(
                not item.strip() for item in self.protocol_axis_element_ids
            ):
                raise ValueError(
                    "protocol_axis_element_ids cannot contain blank values"
                )
            if len(set(self.protocol_axis_element_ids)) != len(
                self.protocol_axis_element_ids
            ):
                raise ValueError("protocol_axis_element_ids must be unique")
            if self.paper_value_status == "paper_unspecified":
                if self.protocol_value is not None:
                    raise ValueError(
                        "paper_value_status='paper_unspecified' requires "
                        "protocol_value=null"
                    )
                if self.paper_value is not None:
                    raise ValueError(
                        "paper_value_status='paper_unspecified' requires "
                        "paper_value=null"
                    )
                if self.source not in {"system_inferred", "spec_default"}:
                    raise ValueError(
                        "paper-unspecified protocol carriers must remain "
                        "system_inferred or spec_default"
                    )
            else:
                if (
                    isinstance(self.protocol_value, bool)
                    or not isinstance(self.protocol_value, (int, float))
                    or self.protocol_value <= 0
                ):
                    raise ValueError(
                        "paper-stated protocol carriers require a positive "
                        "numeric protocol_value"
                    )
                raw_steps = (
                    float(self.protocol_value)
                    / float(self.protocol_granularity)
                )
                rounded = round(raw_steps)
                if rounded <= 0 or abs(raw_steps - rounded) > 1e-9:
                    raise ValueError(
                        "paper-stated protocol_value/granularity must produce "
                        f"positive integral runtime steps; got "
                        f"{self.protocol_value!r}/{self.protocol_granularity!r}"
                    )
                paper_steps = int(rounded)
                if self.source not in {"paper", "system_default"}:
                    raise ValueError(
                        "paper-stated protocol carriers must use source='paper' "
                        "when the runtime value matches or "
                        "source='system_default' with paper_value when it "
                        "differs"
                    )
                if self.source == "paper" and self.paper_value is not None:
                    raise ValueError(
                        "a paper-sourced protocol carrier must not retain a "
                        "separate paper_value; that stale field can publish a "
                        "second, conflicting protocol value"
                    )
                if self.source == "paper" and self.value != paper_steps:
                    raise ValueError(
                        "a paper-sourced protocol carrier value must equal "
                        "protocol_value/protocol_granularity runtime steps"
                    )
                if self.source == "system_default" and self.paper_value is None:
                    raise ValueError(
                        "a system-default protocol carrier with a paper-stated "
                        "value must preserve its converted runtime step count "
                        "separately in paper_value"
                    )
                if (
                    self.source == "system_default"
                    and self.paper_value != paper_steps
                ):
                    raise ValueError(
                        "paper_value must equal the positive integral runtime "
                        "step count derived from protocol_value/granularity"
                    )

        # Usage: when not used, must say why.
        if not self.used_in_notebook and not self.unused_reason:
            raise ValueError(
                "used_in_notebook=False requires `unused_reason` — the user should never "
                "have to wonder why a param is in the dict but never referenced."
            )
        if self.used_in_notebook and self.unused_reason:
            raise ValueError(
                "unused_reason should only be set when used_in_notebook=False"
            )
        return self


class PerDatasetBoundValue(BaseModel):
    """One parameter bound to the demo dataset's paper value."""

    model_config = ConfigDict(extra="forbid")

    param: str
    dataset: str
    value: Any
    evidence: str  # which generated source named the dataset


class PerDatasetFallback(BaseModel):
    """One parameter whose map could not be bound; the scalar shipped."""

    model_config = ConfigDict(extra="forbid")

    param: str
    dataset_map: dict[str, Any]
    scalar_value: Any
    reason: str


class PerDatasetBindingRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    demo_dataset: str | None = None
    bound: list[PerDatasetBoundValue] = []
    fallbacks: list[PerDatasetFallback] = []


class Params(BaseModel):
    """Top-level params.json schema."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = SCHEMA_VERSION
    params: dict[str, ParamEntry]

    # Per-dataset binding record (schema 1.1.0, optional — absent for specs
    # without per_dataset_values). The driver reads `fallbacks` after the
    # deriver runs and logs one assumptions.md entry per item; `bound` rows
    # are run-event material only (a clean bind is paper-faithful).
    per_dataset_binding: PerDatasetBindingRecord | None = None
