"""Catalog text for knowledge-distillation probes."""

from __future__ import annotations

from probes.catalogs._shared import entry


PROBE_CATALOG = {
    "KD-1": entry(
        "the teacher signal influences the distillation loss",
        "A distillation objective that ignores the teacher collapses to an "
        "ordinary student loss while retaining convincing terminology.",
    ),
    "KD-2": entry(
        "temperature changes the distillation loss",
        "An inert temperature parameter means the delivered loss does not "
        "implement the claimed scaling behavior.",
    ),
    "KD-3": entry(
        "distillation tensor shapes behave as claimed",
        "Rank and axis mistakes can silently apply the distillation equation "
        "over the wrong dimensions.",
        status="designed",
    ),
}
