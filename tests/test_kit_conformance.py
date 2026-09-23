"""The kit conformance suite — every verification kit passes the same
contract, mechanically.

Built 2026-07-06 (pulled forward from the Day-2 plan). Registry and
rationale live in `kit_conformance_registry.py`. The suite is
parametrized over registered kits, so its tests are the acceptance bar a
NEW kit meets by registering — and the growth gate at the bottom makes
registration mandatory for any prefix the delivery label derivation
counts as certification evidence.
"""

from __future__ import annotations

import pytest

from probes import VERDICTS
from tests.kit_conformance_registry import (DEMOTING_VERDICTS, KIT_REGISTRY,
                                            Case, Expect, KitConformance)


def _case_params(bucket: str):
    return [
        pytest.param(kit, case, id=f"{kit.kit_id}:{case.case_id}")
        for kit in KIT_REGISTRY
        for case in getattr(kit, bucket)
    ]


def _by_id(verdicts, probe_id):
    hits = [v for v in verdicts if v.probe_id == probe_id]
    assert hits, (
        f"probe {probe_id} emitted no verdict at all — silence is not an "
        f"allowed outcome (got: {[(v.probe_id, v.verdict) for v in verdicts]})")
    return hits[0]


def _check_expectations(case: Case, verdicts) -> None:
    for exp in case.expect:
        v = _by_id(verdicts, exp.probe_id)
        assert v.verdict == exp.verdict, (
            f"{case.case_id}: {exp.probe_id} expected {exp.verdict!r}, got "
            f"{v.verdict!r}: {v.message}")
        if exp.message_contains is not None:
            assert exp.message_contains in v.message, (
                f"{case.case_id}: {exp.probe_id} message must name "
                f"{exp.message_contains!r}, got: {v.message}")
        if exp.finding_class is not None:
            assert v.finding_class == exp.finding_class, (
                f"{case.case_id}: {exp.probe_id} expected finding class "
                f"{exp.finding_class!r}, got {v.finding_class!r}")


# ---------------------------------------------------------------------------
# Invariant 1 — the kit catches its family's zoo defect
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kit,case", _case_params("defects"))
def test_defect_case_is_caught(kit: KitConformance, case: Case, tmp_path):
    verdicts = case.run(tmp_path)
    _check_expectations(case, verdicts)
    flagged = [v for v in verdicts if v.verdict in DEMOTING_VERDICTS]
    assert flagged, (
        f"{case.case_id}: a defect case produced zero demoting verdicts — "
        f"the kit missed its own family's known-bad artifact")


# ---------------------------------------------------------------------------
# Invariant 2 — the kit never false-fails the healthy fixture
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kit,case", _case_params("healthy"))
def test_healthy_case_is_not_flagged(kit: KitConformance, case: Case,
                                     tmp_path):
    verdicts = case.run(tmp_path)
    _check_expectations(case, verdicts)
    demoters = [(v.probe_id, v.verdict, v.message) for v in verdicts
                if v.verdict in DEMOTING_VERDICTS]
    assert not demoters, (
        f"{case.case_id}: healthy fixture drew demoting verdicts: {demoters}")


# ---------------------------------------------------------------------------
# Invariant 3 — the kit never fabricates stand-ins
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kit,case", _case_params("unbindable"))
def test_unbindable_case_is_all_unprobeable(kit: KitConformance, case: Case,
                                            tmp_path):
    verdicts = case.run(tmp_path)
    assert verdicts, (
        f"{case.case_id}: an unbindable package must SAY unprobeable — "
        f"an empty verdict list is silence, not honesty")
    wrong = [(v.probe_id, v.verdict, v.message) for v in verdicts
             if v.verdict != "unprobeable"]
    assert not wrong, (
        f"{case.case_id}: a package the kit cannot bind produced verdicts "
        f"other than unprobeable — a pass or fail here was computed against "
        f"fabricated inputs: {wrong}")
    for v in verdicts:
        assert v.message.strip(), (
            f"{case.case_id}: {v.probe_id} unprobeable without a reason")


# ---------------------------------------------------------------------------
# Invariant 4 — determinism (same case, same verdicts)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kit,case",
    [pytest.param(kit, kit.defects[0], id=kit.kit_id) for kit in KIT_REGISTRY])
def test_first_defect_case_is_deterministic(kit: KitConformance, case: Case,
                                            tmp_path):
    a = [(v.probe_id, v.verdict, v.message)
         for v in case.run(tmp_path / "a")]
    b = [(v.probe_id, v.verdict, v.message)
         for v in case.run(tmp_path / "b")]
    assert a == b


# ---------------------------------------------------------------------------
# Mechanical hygiene — vocabulary, prefixes, non-empty messages
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kit,case",
    _case_params("defects") + _case_params("healthy"))
def test_verdict_vocabulary_and_prefix(kit: KitConformance, case: Case,
                                       tmp_path):
    verdicts = case.run(tmp_path)
    for v in verdicts:
        assert v.verdict in VERDICTS, f"{v.probe_id}: {v.verdict!r}"
        assert v.message.strip(), f"{v.probe_id}: empty message"
        assert v.probe_id.startswith(kit.prefix), (
            f"{case.case_id}: verdict {v.probe_id} does not carry the "
            f"kit's {kit.prefix!r} prefix — family attribution would break "
            f"label derivation")


# ---------------------------------------------------------------------------
# The growth gate — no certification-bearing prefix without conformance
# ---------------------------------------------------------------------------


def test_every_contribution_prefix_has_a_conformance_entry():
    """delivery_label counts these prefixes as contribution evidence: a
    passing probe under one of them can mint a `verified` label. Any prefix
    with that power MUST have a registered conformance entry, so a new kit
    cannot quietly certify packages without passing this suite."""
    from delivery_label import _CONTRIBUTION_PREFIXES  # noqa: PLC0415

    registered = {kit.prefix for kit in KIT_REGISTRY}
    missing = set(_CONTRIBUTION_PREFIXES) - registered
    assert not missing, (
        f"contribution prefixes without kit conformance coverage: "
        f"{sorted(missing)} — register a KitConformance entry (zoo defect "
        f"case, healthy case, unbindable case) before this prefix may "
        f"certify deliveries")


def test_registered_kits_have_all_three_case_buckets():
    for kit in KIT_REGISTRY:
        assert kit.defects, f"{kit.kit_id}: no zoo-defect case"
        assert kit.healthy, f"{kit.kit_id}: no healthy fixture case"
        assert kit.unbindable, f"{kit.kit_id}: no unbindable case"


def test_expectations_use_the_sanctioned_vocabulary():
    for kit in KIT_REGISTRY:
        for bucket in (kit.defects, kit.healthy, kit.unbindable):
            for case in bucket:
                for exp in case.expect:
                    assert isinstance(exp, Expect)
                    assert exp.verdict in VERDICTS
