"""Catalog text for claims-tier probes."""

from __future__ import annotations

from probes.catalogs._shared import entry


PROBE_CATALOG = {
    "CT-1": entry(
        "the method differs from its null baseline",
        "A contribution that behaves identically to a trivial baseline was "
        "not meaningfully exercised.",
    ),
    "CT-2": entry(
        "paper comparisons are directionally checked",
        "Even at smoke scale, a comparison moving opposite to the paper's "
        "claim deserves explicit researcher attention.",
        status="designed",
    ),
    "CT-3": entry(
        "paper claims are recorded in the claims ledger",
        "Every extracted claim needs an explicit verification status so that "
        "missing evidence cannot be mistaken for support.",
    ),
    "CT-4": entry(
        "equations are checked for internal consistency",
        "A transcription can preserve symbols while reversing an optimization "
        "direction or making a claimed term constant.",
        status="designed",
    ),
    "CT-5": entry(
        "a behavioral check distinguishes the contribution",
        "A property that a trivial baseline also satisfies provides no "
        "evidence for the paper's contribution.",
    ),
}
