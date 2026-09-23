"""Catalog text for checks emitted by ``scripts/run_probes.py``."""

from __future__ import annotations

from probes.catalogs._shared import entry


PROBE_CATALOG = {
    "US-1": entry(
        "parameter values are physically plausible",
        "Implausible configuration values can make correct-looking code fail "
        "or produce meaningless results.",
    ),
    "US-2": entry(
        "provenance labels are internally consistent",
        "A parameter labeled as paper-sourced or inferred must agree with the "
        "evidence and reasoning attached to it.",
        status="designed",
    ),
    "US-2a": entry(
        "paper-sourced parameters name a paper location",
        "A paper-sourced value without a location cannot be independently "
        "checked by a researcher.",
    ),
    "US-2b": entry(
        "parameter values match their stated conventions",
        "A value attributed to a named convention must actually fit that "
        "convention, or its provenance is misleading.",
    ),
    "US-3": entry(
        "paper-sourced values are findable in the paper",
        "This catches fabricated or mistranscribed values before they shape "
        "the generated implementation.",
    ),
    "US-3b": entry(
        "inline paper-value claims are findable in the paper",
        "Numbers repeated inside reasoning text need the same source support "
        "as the structured parameter value.",
    ),
    "US-5": entry(
        "declared decision parameters are read by the code",
        "An unused decision parameter can make a claimed algorithmic term "
        "silently inert.",
        status="designed",
    ),
    "US-6": entry(
        "loaded artifacts reach the code that should use them",
        "A dataset, environment, or model artifact can be loaded successfully "
        "and still never influence the demonstrated method.",
        status="designed",
    ),
    "US-7": entry(
        "printed result claims come from computation",
        "Literal success text can make a failed or unevaluated demo look "
        "successful.",
        status="designed",
    ),
    "US-9": entry(
        "notebook validator checks passed",
        "The notebook must honor the package, parameter, and execution "
        "contracts before its narrative can be trusted.",
        status="designed",
    ),
    "US-10": entry(
        "generated code imports and lints cleanly",
        "Syntax, undefined-name, and import failures prevent later behavioral "
        "checks from exercising the delivery at all.",
    ),
    "battery": entry(
        "paradigm-specific checks were selected",
        "Without a paradigm classification the battery can run universal "
        "checks only, leaving the paper's method family under-checked.",
    ),
}
