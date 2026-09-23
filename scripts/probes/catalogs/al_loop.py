"""Catalog text for checks emitted by ``probes/al_loop.py``."""

from __future__ import annotations

from probes.catalogs._shared import entry


PROBE_CATALOG = {
    "AL-1": entry(
        "active-learning loop bookkeeping is correct",
        "Index-space mistakes can acquire the wrong samples while producing a "
        "plausible learning curve.",
    ),
    "AL-2": entry(
        "pool-index arithmetic avoids risky offsets",
        "Offset arithmetic often confuses positions in the unlabeled pool "
        "with indices in the original dataset.",
    ),
    "AL-3": entry(
        "the model is retrained fresh each round",
        "Warm-starting changes the experimental protocol when the paper "
        "requires training from scratch.",
    ),
    "AL-4": entry(
        "fresh model weights are verified behaviorally",
        "Object construction alone does not prove that each round begins from "
        "independent initial weights.",
    ),
    "AL-5": entry(
        "the acquisition function returns a valid, responsive batch",
        "Duplicate, out-of-range, fixed, or model-insensitive selections make "
        "the acquisition loop invalid.",
    ),
    "AL-6": entry(
        "the demo configuration actually exercises acquisition",
        "A full initial labeled set or exhausted pool can make the notebook "
        "skip the paper's contribution while still completing successfully.",
    ),
    "AL-7": entry(
        "learning-curve points evaluate the matching model",
        "A curve can look plausible while each label is attached to metrics "
        "from a different training round.",
    ),
}
