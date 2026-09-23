"""Catalog text for motion-planning probes."""

from __future__ import annotations

from probes.catalogs._shared import entry


# Specialized readings for the scenario-fidelity outcome arms, keyed by the
# emitter's `reason` field (probes/scenario_setup.py and the family
# comparators above the default per-verdict readings).
_SCENARIO_READINGS = {
    "scenario_mismatch": (
        "The executed demo setup affirmatively contradicts a scenario "
        "assumption the paper states in text. The demo can still run, but "
        "it is exercising a different scenario than the paper describes."
    ),
    "no_captured_assumption": (
        "The paper's parsed text states no assumption on this scenario "
        "dimension, so there is nothing to compare the demo against. This "
        "is a disclosure, not a defect."
    ),
    "setup_unbindable": (
        "The paper states an assumption on this dimension but the detector "
        "could not bind the executed setup's state to check it. Confirm the "
        "demo scene by eye before relying on it."
    ),
    "setup_uninspectable": (
        "The notebook's setup slice could not be executed or inspected at "
        "all, so no scenario comparison was possible. Confirm the demo "
        "scene by eye before relying on it."
    ),
    "setup_incomplete": (
        "The setup slice stopped on an error before finishing, so only the "
        "records constructed before the failure could be checked."
    ),
    "setup_state_motion_undecidable": (
        "Setup-time state alone cannot decide whether obstacles move; the "
        "dynamic-scenario loop check owns that judgment."
    ),
    "static_claim_motion_state": (
        "The setup carries motion machinery a static-scenario paper does "
        "not state. Confirm the demo keeps obstacles still."
    ),
    "ambiguous_binding": (
        "More than one candidate population was found and they disagree, "
        "so the count cannot be attributed automatically."
    ),
    "assumption_uninterpretable": (
        "The captured paper value does not follow the dimension's "
        "normalization guidance, so the detector cannot compare it."
    ),
}


PROBE_CATALOG = {
    "MP-1": entry(
        "dynamic obstacles actually move",
        "A dynamic-obstacle method demonstrated on frozen obstacles never "
        "exercises its headline capability.",
    ),
    "MP-2": entry(
        "collision claims are recomputed from trajectories",
        "Printed collision-free claims need an independent replay against the "
        "delivered collision model.",
        status="designed",
    ),
    "MP-3": entry(
        "steering turns toward the goal",
        "A steering-insensitive objective can choose turns by arbitrary tie "
        "break while plots still look plausible.",
    ),
    "MP-4": entry(
        "the robot makes progress toward the goal",
        "A planner that returns or claims success without reducing goal "
        "distance has not demonstrated useful planning behavior.",
    ),
    # Scenario-fidelity checks (SC-*, finding class M-006): the executed demo
    # setup versus the paper's captured scenario assumptions. Deliberately
    # NOT MP-prefixed — a scenario match is evidence about the demo scene,
    # not about the method contribution, so these ids must stay outside the
    # delivery label's contribution tier.
    "SC-1": entry(
        "the demo's obstacle shapes stay within the paper's stated geometry",
        "A circles-only paper demonstrated on polygon obstacles never tests "
        "the geometry its collision model actually assumes.",
        readings=_SCENARIO_READINGS,
    ),
    "SC-2": entry(
        "the demo contains the agent population the paper's scenario states",
        "A crowded-pedestrian paper demonstrated with zero pedestrians never "
        "exercises its headline interaction behavior.",
        readings=_SCENARIO_READINGS,
    ),
    "SC-3": entry(
        "setup-time obstacle motion state matches the paper's "
        "static-or-dynamic assumption",
        "A dynamic-obstacle paper whose setup cannot express motion may "
        "demonstrate on a frozen scene without anyone noticing.",
        readings=_SCENARIO_READINGS,
    ),
}
