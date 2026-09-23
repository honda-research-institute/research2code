"""Catalog text for checks emitted by ``probes/term_ablation.py``."""

from __future__ import annotations

from probes.catalogs._shared import entry


PROBE_CATALOG = {
    "UB-4": entry(
        "the objective prefers known-better states",
        "A sign error can make a maximized objective reward collisions or "
        "distance from the goal.",
    ),
    "UB-7": entry(
        "claimed scoring terms can change the result",
        "A term that never changes the selected ranking is absent in behavior "
        "even when the source code contains its formula.",
    ),
}
