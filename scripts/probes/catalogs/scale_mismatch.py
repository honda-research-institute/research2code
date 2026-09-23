"""Catalog text for checks emitted by ``probes/scale_mismatch.py``."""

from __future__ import annotations

from probes.catalogs._shared import entry


PROBE_CATALOG = {
    "US-4": entry(
        "scale-dependent parameters match the data scale",
        "A paper value calibrated for one scale can become inert or dominant "
        "when the delivered data uses another scale.",
    ),
    "US-4b": entry(
        "the geometric ranking is scale-invariant (not a capped probability)",
        "Capping a ratio as if it were a probability can flatten every score "
        "and turn an informative ranking into position order.",
    ),
}
