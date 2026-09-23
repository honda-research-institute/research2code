"""Shared constructors for dependency-free probe catalog entries."""

from __future__ import annotations

from typing import Any


DEFAULT_READINGS = {
    "fail": (
        "The check found concrete evidence that the delivered behavior does "
        "not satisfy this expectation. Review the recorded evidence and the "
        "implementation path before relying on this part of the demo."
    ),
    "flag_for_researcher": (
        "The result needs domain judgment. It may reflect a faithful but "
        "degenerate method, a demo-scale limitation, or a delivery problem. "
        "Compare the evidence with the paper before deciding."
    ),
    "warn": (
        "This is advisory evidence. It does not fail the delivery, though the "
        "named risk remains worth checking."
    ),
    "unprobeable": (
        "This check could not reach a conclusion. The recorded message names "
        "the missing artifact, environment capability, harness vocabulary, "
        "or method shape that blocked it."
    ),
}


def entry(
    what: str,
    why: str,
    *,
    status: str = "implemented",
    readings: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Return one complete catalog entry with optional specialized readings."""
    on_failure = dict(DEFAULT_READINGS)
    if readings:
        on_failure.update(readings)
    return {
        "what": what,
        "why": why,
        "on_failure": on_failure,
        "status": status,
    }
