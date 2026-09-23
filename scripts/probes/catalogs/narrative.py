"""Catalog text for checks emitted by ``probes/narrative.py``."""

from __future__ import annotations

from probes.catalogs._shared import entry


PROBE_CATALOG = {
    "US-8": entry(
        "notebook prose numbers match the params table",
        "Stale narrative values can contradict the configuration that the "
        "executed demo actually used.",
    ),
}
