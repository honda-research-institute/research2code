"""Catalog text for checks emitted by ``probes/al_stage1.py``."""

from __future__ import annotations

from probes.catalogs._shared import entry


PROBE_CATALOG = {
    "AL-S1-1": entry(
        "the selector returns valid, unique picks",
        "Invalid or duplicate core-set indices corrupt the labeled pool before "
        "the iterative acquisition loop begins.",
    ),
    "AL-S1-2": entry(
        "selections are not stuck in the first cluster",
        "Position-concentrated selections are a practical signal that the "
        "core-set score collapsed to ties.",
    ),
    "AL-S1-3": entry(
        "the paper's selection parameter has a live effect",
        "If changing the parameter that controls the paper's geometric "
        "mechanism changes nothing, the claimed contribution may be inert.",
    ),
}
