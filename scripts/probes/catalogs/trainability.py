"""Catalog text for checks emitted by ``probes/trainability.py``."""

from __future__ import annotations

from probes.catalogs._shared import entry


PROBE_CATALOG = {
    "UB-5": entry(
        "the live training config learns a toy dataset",
        "A tiny separable dataset isolates whether the delivered model and "
        "training configuration can learn at all.",
    ),
}
