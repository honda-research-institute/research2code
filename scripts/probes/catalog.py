"""Aggregated authored explanations for every probe verdict id.

This module is dependency-free so researcher-facing renderers and driver
notices can use the same authority without importing probe executors.
"""

from __future__ import annotations

from typing import Any

from probes.catalogs.al_loop import PROBE_CATALOG as AL_LOOP
from probes.catalogs.al_stage1 import PROBE_CATALOG as AL_STAGE1
from probes.catalogs.claims import PROBE_CATALOG as CLAIMS
from probes.catalogs.graph_mechanism import PROBE_CATALOG as GRAPH_MECHANISM
from probes.catalogs.knowledge_distillation import (
    PROBE_CATALOG as KNOWLEDGE_DISTILLATION,
)
from probes.catalogs.motion_planning import PROBE_CATALOG as MOTION_PLANNING
from probes.catalogs.narrative import PROBE_CATALOG as NARRATIVE
from probes.catalogs.run_probes import PROBE_CATALOG as RUN_PROBES
from probes.catalogs.scale_mismatch import PROBE_CATALOG as SCALE_MISMATCH
from probes.catalogs.term_ablation import PROBE_CATALOG as TERM_ABLATION
from probes.catalogs.time_series_forecasting import (
    PROBE_CATALOG as TIME_SERIES_FORECASTING,
)
from probes.catalogs.trainability import PROBE_CATALOG as TRAINABILITY
from probes.catalogs.universal import PROBE_CATALOG as UNIVERSAL


CATALOG_FRAGMENTS = (
    RUN_PROBES,
    SCALE_MISMATCH,
    NARRATIVE,
    UNIVERSAL,
    TRAINABILITY,
    TERM_ABLATION,
    AL_LOOP,
    AL_STAGE1,
    MOTION_PLANNING,
    TIME_SERIES_FORECASTING,
    GRAPH_MECHANISM,
    KNOWLEDGE_DISTILLATION,
    CLAIMS,
)


def _aggregate() -> dict[str, dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    for fragment in CATALOG_FRAGMENTS:
        overlap = set(merged).intersection(fragment)
        if overlap:
            raise ValueError(
                "probe catalog ids have multiple owners: "
                + ", ".join(sorted(overlap))
            )
        merged.update(fragment)
    return merged


PROBE_CATALOG = _aggregate()
PROBE_GLOSS = {
    probe_id: str(record["what"])
    for probe_id, record in PROBE_CATALOG.items()
    if probe_id != "battery"
}


def probe_record(probe_id: object) -> dict[str, Any] | None:
    return PROBE_CATALOG.get(str(probe_id or ""))


def probe_label(probe_id: object) -> str:
    record = probe_record(probe_id)
    return str(record["what"]) if record else "verification check"


def failure_reading(
    probe_id: object,
    verdict: object,
    reason: object = "",
) -> str:
    record = probe_record(probe_id)
    if not record:
        return (
            "The check recorded a result without an authored explanation. "
            "Use the result message and evidence as the current authority."
        )
    readings = record.get("on_failure")
    if not isinstance(readings, dict):
        return ""
    reason_key = str(reason or "")
    verdict_key = str(verdict or "")
    return str(
        readings.get(reason_key)
        or readings.get(verdict_key)
        or readings.get("default")
        or ""
    )
