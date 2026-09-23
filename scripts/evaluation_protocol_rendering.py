"""Deterministic researcher-facing view of a typed evaluation protocol.

The method spec owns paper truth; params.json owns the values the demo will
actually use.  Keeping those two columns separate prevents a validation or
test span from being presented as a one-call forecast horizon, and makes an
explicit paper-unspecified value visible instead of replacing it with nearby
notation.

Legacy specs without ``comparison.evaluation_protocol`` produce no block.
"""

from __future__ import annotations

import json
from typing import Any

from pydantic import ValidationError

from schemas.method_spec import EvaluationProtocol


_ROLE_ORDER = (
    "context_length",
    "forecast_call_horizon",
    "validation_span",
    "test_span",
)

_ROLE_LABELS = {
    "context_length": "Context length",
    "forecast_call_horizon": "One-call forecast horizon",
    "validation_span": "Validation span",
    "test_span": "Test span",
}


def _clean(value: object) -> str:
    """Whitespace-normalize and escape a Markdown table cell."""
    return " ".join(str(value or "").split()).replace("|", "\\|")


def _display_value(value: object) -> str:
    if value is None:
        return "—"
    return _clean(json.dumps(value, ensure_ascii=False, sort_keys=True))


def _params_entries(params: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(params, dict):
        return {}
    entries = params.get("params", params)
    return entries if isinstance(entries, dict) else {}


def _evidence_location(section: object, element_ids: object) -> str:
    location = _clean(section) or "paper location unavailable"
    if isinstance(element_ids, list):
        ids = ", ".join(
            f"`{_clean(item)}`" for item in element_ids if str(item).strip()
        )
        if ids:
            location += f"; paper IDs: {ids}"
    return location


def _scheme_line(scheme: dict[str, Any]) -> str:
    status = str(scheme.get("paper_value_status") or "")
    section = _evidence_location(
        scheme.get("paper_section"), scheme.get("paper_element_ids")
    )
    if status == "paper_unspecified":
        declaration = "**paper-unspecified**"
    else:
        kind = _clean(scheme.get("kind")).replace("_", " ")
        declaration = f"**{kind or 'paper-stated'}**"

    suffix = f" — {section.rstrip('.')}" if section else ""
    return f"**Paper evaluation scheme:** {declaration}{suffix}."


def _runtime_cell(
    quantity: dict[str, Any], entries: dict[str, Any],
) -> str:
    parameter_name = quantity.get("parameter_name")
    if not isinstance(parameter_name, str) or not parameter_name:
        return "spec-only fact; no runtime parameter"
    entry = entries.get(parameter_name)
    if not isinstance(entry, dict):
        return f"`{_clean(parameter_name)}`: not derived"
    value = _display_value(entry.get("value"))
    source = _clean(entry.get("source")).replace("_", " ") or "source unknown"
    return f"`{_clean(parameter_name)}` = {value} ({source})"


def render_evaluation_protocol_block(
    spec: dict[str, Any] | None,
    params: dict[str, Any] | None = None,
    *,
    heading: str = "## Evaluation protocol",
) -> str:
    """Render paper protocol facts beside, never merged with, demo values.

    Returns an empty string for a missing/legacy protocol block.  Inputs are
    raw JSON dictionaries because all three consumers already load their own
    artifacts and this renderer must stay usable for partial deliveries.
    """
    if not isinstance(spec, dict):
        return ""
    comparison = spec.get("comparison")
    if not isinstance(comparison, dict):
        return ""
    protocol = comparison.get("evaluation_protocol")
    if not isinstance(protocol, dict):
        return ""
    try:
        # Researcher-facing renderers also run for partial/halted deliveries
        # and therefore consume raw JSON rather than a pre-validated model.
        # Validate here before selecting by role: a duplicate role or an
        # unknown status must never silently win and become asserted truth.
        protocol = EvaluationProtocol.model_validate(protocol).model_dump(
            mode="json"
        )
    except ValidationError:
        return "\n".join([
            heading,
            "",
            "**Evaluation protocol unavailable:** the structured protocol "
            "record is invalid or internally conflicting, so no paper or "
            "runtime values are shown. Inspect and repair "
            "`.pipeline/method_spec.json` before relying on this section.",
        ])
    scheme = protocol["scheme"]
    quantities = protocol["quantities"]

    by_role = {
        str(quantity.get("role")): quantity
        for quantity in quantities
        if isinstance(quantity, dict) and quantity.get("role")
    }
    entries = _params_entries(params)
    rows = [
        heading,
        "",
        _scheme_line(scheme),
        "",
        "| protocol role | paper declaration | unit / granularity | runtime / demo | role/value evidence | axis evidence |",
        "|---|---|---|---|---|---|",
    ]
    for role in _ROLE_ORDER:
        quantity = by_role.get(role)
        if not isinstance(quantity, dict):
            continue
        status = str(quantity.get("paper_value_status") or "")
        if status == "paper_unspecified":
            paper_declaration = "**paper-unspecified**"
        else:
            paper_declaration = f"{_display_value(quantity.get('value'))} (paper-stated)"
        unit = _clean(quantity.get("unit")) or "unit unavailable"
        granularity = _display_value(quantity.get("granularity"))
        unit_cell = f"{unit}; {granularity} {unit} per protocol step"
        value_evidence = _evidence_location(
            quantity.get("paper_section"), quantity.get("paper_element_ids")
        )
        axis_evidence = _evidence_location(
            quantity.get("axis_paper_section"),
            quantity.get("axis_paper_element_ids"),
        )
        rows.append(
            f"| **{_ROLE_LABELS[role]}** | {paper_declaration} | "
            f"{unit_cell} | {_runtime_cell(quantity, entries)} | "
            f"{value_evidence} | {axis_evidence} |"
        )
    return "\n".join(rows)
