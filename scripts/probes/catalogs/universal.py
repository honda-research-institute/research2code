"""Catalog text for checks emitted by ``probes/universal.py``."""

from __future__ import annotations

from probes.catalogs._shared import entry


PROBE_CATALOG = {
    "UB-1": entry(
        "shape claims hold when the code runs",
        "Tensor-rank and axis mistakes often execute successfully while "
        "implementing a different equation.",
    ),
    "UB-2": entry(
        "score-producing functions produce varied scores",
        "A constant score cannot rank candidates and usually collapses to a "
        "position-based tie break.",
    ),
    "UB-3": entry(
        "the objective responds to decision variables",
        "A decision variable that cannot affect the objective cannot affect "
        "the selected action.",
    ),
    "UB-6": entry(
        "executed notebook outputs are sane",
        "The notebook is the integration test. Diverged, chance-level, or "
        "flat outputs mean that successful execution alone is insufficient.",
        readings={
            "output_diverged": (
                "A NaN or infinite value means the executed computation "
                "became numerically invalid somewhere. Where is the question "
                "to answer first: a non-finite metric can come from the "
                "ground truth it scores against (a held-out window with no "
                "data), from the model's own weights or predictions, or from "
                "an undefended division inside the metric. Check whether the "
                "weights and predictions are finite before treating this as "
                "training divergence, then compare the first invalid value "
                "with the paper's stated settings to separate a delivery "
                "error from settings that are unstable at demo scale."
            ),
            "below_chance": (
                "The measured accuracy never cleared a conservative chance "
                "baseline. This usually means the delivered training or "
                "evaluation path is ineffective at demo scale."
            ),
            "runaway_series": (
                "The recorded loss grows consistently and substantially. "
                "Check update direction, learning rate, and objective signs."
            ),
            "flat_series": (
                "Every recurring result series stayed effectively constant. "
                "Check whether training, acquisition, or evaluation is "
                "actually updating the state it reports."
            ),
            "unvetted_series": (
                "A numeric series moved, though this check lacks the context "
                "needed to decide whether that movement is good. A researcher "
                "should compare it with the metric definition in the paper."
            ),
        },
    ),
    "UB-8": entry(
        "random seeds thread through the method",
        "Ignored or fixed seeds make variability claims and repeated "
        "verification unreliable.",
    ),
    "UB-9": entry(
        "training loss descends",
        "A flat or rising loss is direct evidence that the delivered "
        "optimization path is not learning as intended.",
    ),
}
