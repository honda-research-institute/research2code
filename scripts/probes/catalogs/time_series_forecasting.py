"""Catalog text for time-series forecasting probes."""

from __future__ import annotations

from probes.catalogs._shared import entry


PROBE_CATALOG = {
    "TSF-1": entry(
        "forecast samples are genuine distribution draws",
        "Conditional means copied across the sample axis can look like "
        "probabilistic forecasts while understating uncertainty by orders "
        "of magnitude.",
    ),
    "TSF-2": entry(
        "autoregressive samples carry path dependence",
        "Independent horizon marginals cannot represent a decoder whose "
        "sampled value at one step conditions the next step.",
    ),
    "TSF-3": entry(
        "each series forecast responds to its own history",
        "A severed self-signal path can leave a series forecast unchanged "
        "when that series' eligible history changes dramatically.",
    ),
    "TSF-4": entry(
        "held-out forecasts beat both required naive baselines",
        "A runnable model that loses to predict-zero or repeat-last has not "
        "demonstrated useful forecasting skill on the valid held-out window.",
    ),
    "TSF-5": entry(
        "forecast magnitude has not collapsed toward zero",
        "Unscaled targets can produce decreasing training loss while the "
        "forecast remains orders of magnitude smaller than its own history.",
    ),
}
