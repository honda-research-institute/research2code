"""Compare a bundled time axis with a role-typed evaluation protocol.

R2C-065's first floor used ``context + 2 * forecast_horizon``.  That proxy
conflated a one-call forecast horizon with the validation and test spans; the
pdfgnn paper actually states context 10, validation 13 weeks, test 26 weeks,
and leaves numeric K unspecified.  For the currently evidenced
``single_holdout`` scheme the static coverage floor is therefore exactly
``context + validation + test``.  K belongs to a separate per-call check once
the run has chosen a concrete demo value.

Counts are comparable only on one axis.  Bundle provenance now carries an
exact source cadence when all distinct timestamps parse and have one regular
interval.  This module accepts only an identity join: source and protocol
must have the same unit and granularity.  A daily source under a weekly
protocol is UNRESOLVED until a later contract records and realizes the
loader's aggregation/resampling output; raw daily rows are never divided by
seven or treated as weekly observations by assumption.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class AxisRequirement:
    """The protocol arithmetic the bundle has to satisfy."""

    context_steps: int
    validation_steps: int
    test_steps: int
    unit: str
    granularity: int | float

    @property
    def floor(self) -> int:
        return self.context_steps + self.validation_steps + self.test_steps

    def arithmetic(self) -> str:
        return (
            f"{self.context_steps} context steps + "
            f"{self.validation_steps} validation steps + "
            f"{self.test_steps} test steps = {self.floor} minimum "
            f"({self.granularity:g} {self.unit}/step; forecast-call horizon "
            "is a separate per-call quantity)"
        )


class ProtocolAxisUnresolved(ValueError):
    """The typed protocol cannot yet produce exact static axis arithmetic."""


@dataclass(frozen=True)
class AxisVerdict:
    # None is an explicit unresolved join.  It must never be interpreted as
    # an infeasible bundle: absence of conversion evidence is not evidence
    # that the bundle is too short.
    feasible: bool | None
    realized_steps: int
    requirement: AxisRequirement
    table: str
    column: str
    # R2C-069: the axis the table nominally spans, and the columns that stop
    # carrying values before it ends. `realized_steps` is the USABLE extent,
    # so these two exist to keep the message honest about the difference.
    nominal_steps: int = 0
    dead_tail_columns: tuple[str, ...] = ()
    source_unit: str | None = None
    source_granularity: int | float | None = None
    unresolved_reason: str | None = None

    def _live_extent_note(self) -> str:
        if not self.dead_tail_columns or self.nominal_steps <= self.realized_steps:
            return ""
        columns = ", ".join(f"`{c}`" for c in self.dead_tail_columns)
        return (
            f" (measured on the USABLE axis: the table spans "
            f"{self.nominal_steps} steps, but {columns} stop carrying values "
            f"after step {self.realized_steps}, so the last "
            f"{self.nominal_steps - self.realized_steps} hold no target)"
        )

    def message(self) -> str:
        if self.feasible is None:
            return (
                "bundle time-axis feasibility is UNRESOLVED: "
                f"`{self.table}`.`{self.column}` has "
                f"{self.realized_steps} usable source steps, but "
                f"{self.unresolved_reason or 'its source cadence is unknown'}; "
                f"the typed single-holdout protocol needs "
                f"{self.requirement.arithmetic()}. No raw-count conversion "
                "was assumed"
                + self._live_extent_note()
            )
        if self.feasible:
            return (
                f"bundle time axis is protocol-feasible: {self.realized_steps} "
                f"usable steps on `{self.table}`.`{self.column}` against "
                f"{self.requirement.arithmetic()}"
                + self._live_extent_note()
            )
        return (
            f"bundle time axis cannot support the paper's protocol: the "
            f"longest usable bundled axis is {self.realized_steps} steps "
            f"(`{self.table}`.`{self.column}`) and the protocol needs "
            f"{self.requirement.arithmetic()}"
            + self._live_extent_note()
        )


@dataclass(frozen=True)
class AxisParamShrink:
    """One typed forecast-call parameter below its paper step equivalent."""

    param: str
    resolved_value: int
    paper_steps: int
    paper_quantity: int | float
    role: str
    unit: str
    granularity: int | float
    quantity_path: str
    context_steps: int
    validation_steps: int
    test_steps: int
    realized_steps: int
    table: str
    column: str
    feasible: bool

    @property
    def paper_value(self) -> int:
        """Backward-compatible name for the runtime-step paper equivalent."""
        return self.paper_steps

    def arithmetic(self) -> str:
        call_steps = self.context_steps + self.paper_steps
        validation_ok = self.paper_steps <= self.validation_steps
        test_ok = self.paper_steps <= self.test_steps
        axis_ok = call_steps <= self.realized_steps
        return (
            f"{self.quantity_path} role={self.role!r}: paper quantity "
            f"{self.paper_quantity:g} {self.unit} / {self.granularity:g} "
            f"{self.unit}/step = {self.paper_steps} runtime steps; "
            f"K={self.paper_steps} <= validation_span="
            f"{self.validation_steps} is {validation_ok}; "
            f"K={self.paper_steps} <= test_span={self.test_steps} is "
            f"{test_ok}; context_length={self.context_steps} + "
            f"K={self.paper_steps} = {call_steps} <= compatible realized "
            f"axis={self.realized_steps} at "
            "method/example_data/PROVENANCE.json "
            f"(`{self.table}`.`{self.column}`) is {axis_ok}"
        )

    def validation_message(self) -> str:
        return (
            "typed_authority_consumer_bypass: resolved protocol parameter "
            f"params.{self.param}.value={self.resolved_value} is below the "
            f"typed paper equivalent {self.paper_steps} runtime steps, and "
            f"the role-compatible ranges support it ({self.arithmetic()}). "
            "Resolve the exact parameter_name carrier at the converted paper "
            "step count; glossary and scale-lane values are not authority."
        )

    def shrink_reason(self) -> str:
        return (
            f"The demo keeps {self.param}={self.resolved_value} below the "
            f"typed paper equivalent {self.paper_steps} runtime steps because "
            f"at least one role-compatible boundary is insufficient "
            f"({self.arithmetic()})."
        )


def _positive_int(value: object) -> int | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if isinstance(value, float) and not value.is_integer():
        return None
    return int(value) if value > 0 else None


def protocol_quantity_steps(quantity: dict, *, path: str) -> int:
    """Convert one typed physical quantity to positive integral steps.

    ``value`` is the paper's physical quantity and ``granularity`` is the
    amount of that same unit represented by one runtime step.  Keeping this
    conversion shared prevents the producer and validator from comparing a
    raw paper number with a step-count parameter.
    """
    role = quantity.get("role")
    value = quantity.get("value")
    unit = quantity.get("unit")
    granularity = quantity.get("granularity")
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or value <= 0
        or not isinstance(unit, str)
        or not unit
        or isinstance(granularity, bool)
        or not isinstance(granularity, (int, float))
        or granularity <= 0
    ):
        raise ProtocolAxisUnresolved(
            f"{path} role={role!r} needs positive numeric value and "
            "granularity plus a non-empty unit before it can bind a "
            "step-count runtime carrier"
        )
    raw_steps = float(value) / float(granularity)
    rounded = round(raw_steps)
    if rounded <= 0 or abs(raw_steps - rounded) > 1e-9:
        raise ProtocolAxisUnresolved(
            f"{path} role={role!r} paper quantity {value!r} {unit} / "
            f"{granularity!r} {unit}/step = {raw_steps:g} runtime steps, "
            "which is not an integral number of runtime steps (a positive "
            "integer is required)"
        )
    return int(rounded)


def protocol_axis_floor(spec: dict) -> AxisRequirement | None:
    """Build the strict typed ``single_holdout`` coverage requirement.

    Legacy glossary numbers are intentionally ignored.  None is the
    cross-family scope guard when no typed protocol exists.  A typed protocol
    whose scheme or quantities cannot produce exact arithmetic is explicit
    unresolved state rather than a guessed floor.
    """
    protocol = (spec.get("comparison") or {}).get("evaluation_protocol")
    if not isinstance(protocol, dict):
        return None
    scheme = protocol.get("scheme")
    scheme_kind = scheme.get("kind") if isinstance(scheme, dict) else None
    if scheme_kind != "single_holdout":
        raise ProtocolAxisUnresolved(
            "evaluation scheme is not a stated single_holdout; rolling, "
            "expanding, sliding, cross-validation, and unspecified schemes "
            "need typed origin/fold/window controls before their coverage "
            "arithmetic can be derived"
        )
    quantities = protocol.get("quantities")
    if not isinstance(quantities, list):
        raise ProtocolAxisUnresolved(
            "comparison.evaluation_protocol.quantities is unavailable"
        )
    role_items = [
        item for item in quantities
        if isinstance(item, dict) and isinstance(item.get("role"), str)
    ]
    by_role = {item["role"]: item for item in role_items}
    all_roles = (
        "context_length", "forecast_call_horizon",
        "validation_span", "test_span",
    )
    missing = [role for role in all_roles if role not in by_role]
    if missing:
        raise ProtocolAxisUnresolved(
            f"typed single_holdout protocol is missing role(s) {missing!r}"
        )
    duplicates = sorted({
        role for role in all_roles
        if sum(item.get("role") == role for item in role_items) > 1
    })
    if duplicates:
        raise ProtocolAxisUnresolved(
            f"typed single_holdout protocol duplicates role(s) {duplicates!r}"
        )

    shared_unit: str | None = None
    shared_granularity: int | float | None = None
    for role in all_roles:
        quantity = by_role[role]
        granularity = quantity.get("granularity")
        unit = quantity.get("unit")
        if (
            isinstance(granularity, bool)
            or not isinstance(granularity, (int, float))
            or granularity <= 0
            or not isinstance(unit, str)
            or not unit
        ):
            raise ProtocolAxisUnresolved(
                f"typed role {role!r} must have positive numeric granularity "
                "plus a non-empty unit"
            )
        if shared_unit is None:
            shared_unit = unit
            shared_granularity = granularity
        elif unit != shared_unit or float(granularity) != float(
            shared_granularity
        ):
            raise ProtocolAxisUnresolved(
                "all evaluation-protocol roles must share one exact "
                "unit/granularity before their step counts can be "
                f"added; saw {shared_unit!r}/{shared_granularity!r} and "
                f"{unit!r}/{granularity!r}"
            )

    converted: dict[str, int] = {}
    for role in ("context_length", "validation_span", "test_span"):
        quantity = by_role[role]
        if quantity.get("paper_value_status") != "paper_stated":
            raise ProtocolAxisUnresolved(
                f"typed role {role!r} is paper-unspecified; no static "
                "evaluation-coverage value can be inferred"
            )
        role_index = next(
            index for index, item in enumerate(quantities)
            if item is quantity
        )
        converted[role] = protocol_quantity_steps(
            quantity,
            path=f"comparison.evaluation_protocol.quantities[{role_index}]",
        )

    assert shared_unit is not None and shared_granularity is not None
    return AxisRequirement(
        context_steps=converted["context_length"],
        validation_steps=converted["validation_span"],
        test_steps=converted["test_span"],
        unit=shared_unit,
        granularity=shared_granularity,
    )


def _usable_steps(axis: dict) -> tuple[int, int, tuple[str, ...]]:
    """The table's usable axis length, its nominal length, and the columns
    that end early.

    The usable length is the live extent the materializer measured (R2C-069):
    a nominal span is not a usable one when the source withholds its target
    over a final period, which is the normal layout for a public forecasting
    dataset because that period is the competition's own prediction window.
    Falls back to the nominal count for provenance written before live extent
    was measured. Such a legacy axis still receives an unresolved cadence
    verdict rather than a feasibility claim.
    """
    nominal = _positive_int(axis.get("steps_kept")) or 0
    live = _positive_int(axis.get("live_steps"))
    dead = axis.get("dead_tail_columns")
    columns = tuple(
        str(d.get("column"))
        for d in (dead if isinstance(dead, list) else [])
        if isinstance(d, dict) and d.get("column")
    )
    if live is None or live > nominal:
        return nominal, nominal, columns
    return live, nominal, columns


def check_bundle_axis(
    manifest: dict, requirement: AxisRequirement,
) -> AxisVerdict | None:
    """Compare the bundle's longest USABLE time axis against the floor.

    None when no bundled table carries a time axis at all, which is not a
    verdict: a paradigm whose data has no time column (image classification)
    must be untouched by this check, and a temporal paper whose bundle has
    no detected time column is a detection gap to fix rather than a bundle
    to refuse.

    The comparison is against the usable extent rather than the nominal step
    count, because the 2026-08-06 pdfgnn bundle cleared the floor 25 times
    over on its nominal 1,092 steps while its last 59 held no target at all.
    Compatible-cadence tables outrank incompatible ones, then usable extent
    ranks candidates within that set. The longest raw axis is not necessarily
    an axis the protocol can consume."""
    candidates: list[
        tuple[
            int, str, str, int, tuple[str, ...],
            str | None, int | float | None,
        ]
    ] = []
    for entry in manifest.get("files") or []:
        axis = entry.get("time_axis") if isinstance(entry, dict) else None
        if not isinstance(axis, dict):
            continue
        steps, nominal, dead_columns = _usable_steps(axis)
        cadence = axis.get("source_cadence")
        source_unit = cadence.get("unit") if isinstance(cadence, dict) else None
        source_granularity = (
            cadence.get("granularity") if isinstance(cadence, dict) else None
        )
        if (
            not isinstance(source_unit, str)
            or isinstance(source_granularity, bool)
            or not isinstance(source_granularity, (int, float))
            or source_granularity <= 0
        ):
            source_unit = None
            source_granularity = None
        candidates.append((
            steps, str(entry.get("file") or "?"),
            str(axis.get("column") or "?"), nominal, dead_columns,
            source_unit, source_granularity,
        ))
    if not candidates:
        return None
    compatible = [
        candidate for candidate in candidates
        if candidate[5] == requirement.unit
        and candidate[6] is not None
        and float(candidate[6]) == float(requirement.granularity)
    ]
    best = max(compatible or candidates, key=lambda candidate: candidate[0])
    (
        steps, table, column, nominal, dead_columns,
        source_unit, source_granularity,
    ) = best
    if not compatible:
        if source_unit is None:
            reason = (
                "provenance carries no exact regular parseable source cadence"
            )
        else:
            reason = (
                f"source cadence is {source_granularity:g} "
                f"{source_unit}/step while the protocol cadence is "
                f"{requirement.granularity:g} {requirement.unit}/step, and "
                "no realized aggregation/resampling carrier joins them"
            )
        return AxisVerdict(
            feasible=None,
            realized_steps=steps,
            requirement=requirement,
            table=table,
            column=column,
            nominal_steps=nominal,
            dead_tail_columns=dead_columns,
            source_unit=source_unit,
            source_granularity=source_granularity,
            unresolved_reason=reason,
        )
    return AxisVerdict(feasible=steps >= requirement.floor,
                       realized_steps=steps, requirement=requirement,
                       table=table, column=column,
                       nominal_steps=nominal,
                       dead_tail_columns=dead_columns,
                       source_unit=source_unit,
                       source_granularity=source_granularity)


def load_bundle_manifest(run_dir: Path) -> dict | None:
    """The run's measured bundle provenance, or None when unavailable."""
    path = run_dir / "method" / "example_data" / "PROVENANCE.json"
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return manifest if isinstance(manifest, dict) else None


def protocol_axis_param_shrinks(
    spec: dict, params: dict, manifest: dict,
) -> list[AxisParamShrink]:
    """Return typed, exact-name forecast-horizon reconciliation decisions.

    Authority requires all of the following: a paper-stated
    ``forecast_call_horizon`` bound by its exact ``parameter_name``; producer
    metadata copied from that same quantity; a positive integral physical to
    step conversion; and a bundle axis with the exact protocol cadence.  A
    missing or incompatible cadence is absence of authority, never evidence
    that the data is insufficient.  Glossary names and scale lanes are not
    read anywhere in this join.
    """
    try:
        requirement = protocol_axis_floor(spec)
    except ProtocolAxisUnresolved:
        return []
    if requirement is None:
        return []
    axis = check_bundle_axis(manifest, requirement)
    # Stage 2a owns the static C+V+T coverage verdict. Only a bundle that
    # conclusively clears that earlier floor is a live input to this Stage-2x
    # reconciliation; a statically infeasible or unresolved manifest cannot
    # authorize either adoption or a shrink explanation here.
    if axis is None or axis.feasible is not True:
        return []

    protocol = (spec.get("comparison") or {}).get("evaluation_protocol") or {}
    quantities = protocol.get("quantities") or []
    if not isinstance(quantities, list):
        return []

    shrinks: list[AxisParamShrink] = []
    for index, quantity in enumerate(quantities):
        if (
            not isinstance(quantity, dict)
            or quantity.get("role") != "forecast_call_horizon"
            or quantity.get("paper_value_status") != "paper_stated"
        ):
            continue
        name = quantity.get("parameter_name")
        if not isinstance(name, str) or not name:
            continue
        entry = params.get(name)
        if not isinstance(entry, dict):
            continue
        path = f"comparison.evaluation_protocol.quantities[{index}]"
        try:
            paper_steps = protocol_quantity_steps(quantity, path=path)
        except ProtocolAxisUnresolved:
            return []
        resolved = _positive_int(entry.get("value"))
        if resolved is None or resolved >= paper_steps:
            continue

        # The producer runs first and stamps this tuple. Requiring it here
        # prevents a direct caller or stale params record from bypassing the
        # typed authority with a coincidentally matching key.
        typed_fields = {
            "protocol_role": quantity.get("role"),
            "protocol_value": quantity.get("value"),
            "protocol_unit": quantity.get("unit"),
            "protocol_granularity": quantity.get("granularity"),
            "paper_value_status": quantity.get("paper_value_status"),
        }
        if any(
            entry.get(field) != expected
            for field, expected in typed_fields.items()
        ):
            continue

        call_steps = requirement.context_steps + paper_steps
        feasible = (
            paper_steps <= requirement.validation_steps
            and paper_steps <= requirement.test_steps
            and call_steps <= axis.realized_steps
        )
        shrinks.append(AxisParamShrink(
            param=name,
            resolved_value=resolved,
            paper_steps=paper_steps,
            paper_quantity=quantity["value"],
            role="forecast_call_horizon",
            unit=requirement.unit,
            granularity=requirement.granularity,
            quantity_path=path,
            context_steps=requirement.context_steps,
            validation_steps=requirement.validation_steps,
            test_steps=requirement.test_steps,
            realized_steps=axis.realized_steps,
            table=axis.table,
            column=axis.column,
            feasible=feasible,
        ))
    return shrinks
