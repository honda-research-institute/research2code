"""AL-5 crash arm + crash attribution + AL-6 demo-config reachability.

Zoo acceptance for the 2026-06-10 negative-stride adjudication
(`gbald-negstride-selector/`): a selector that crashes inside its own code
on contract-conformant inputs is a FINDING, never a silent unprobeable —
and the demo config that hid the crash (empty unlabeled pool from round 0)
is deterministically flagged from params.json alone.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from probes.al_loop import probe_demo_config_reachability

ZOO = Path(__file__).parent / "fixtures" / "zoo"
EVIDENCE = Path(__file__).parent / "fixtures" / "evidence"
NEGSTRIDE = ZOO / "gbald-negstride-selector"


# ---------------------------------------------------------------------------
# AL-6 — pure params arithmetic, no torch needed
# ---------------------------------------------------------------------------


def _params(path: Path) -> dict:
    loaded = json.loads(path.read_text(encoding="utf-8"))
    return loaded.get("params", loaded)


def test_al6_fails_unreachable_loop_on_zoo_params():
    # The shipped config: initial_labeled=1000 >= pool_size=800 — the
    # unlabeled pool is empty before round 1, Stage 2 never executes.
    v = probe_demo_config_reachability(_params(NEGSTRIDE / "params.json"))
    assert v.verdict == "fail"
    assert v.finding_class == "M-004"
    assert "unreachable" in v.message
    assert "initial_labeled=1000" in v.message


def test_al6_passes_badge_shaped_config():
    v = probe_demo_config_reachability({
        "initial_labeled": {"value": 100},
        "pool_size": {"value": 800},
        "num_rounds": {"value": 5},
        "batch_size": {"value": 100},
    })
    assert v.verdict == "pass", v.message


def test_al6_warns_on_early_exhaustion():
    # 700 unlabeled remain; 5 rounds x 200 want 1000 — rounds 4-5 starve.
    v = probe_demo_config_reachability({
        "initial_labeled": {"value": 100},
        "pool_size": {"value": 800},
        "num_rounds": {"value": 5},
        "batch_size": {"value": 200},
    })
    assert v.verdict == "warn"
    assert "exhausts early" in v.message


def test_al6_uses_batch_size_not_legacy_batch_outputs():
    # The runtime acquisition count is batch_size. Historical GBALD drafts
    # carried batch_outputs, but that no longer overrides the contract.
    v = probe_demo_config_reachability({
        "initial_labeled": {"value": 100},
        "pool_size": {"value": 800},
        "num_rounds": {"value": 5},
        "batch_size": {"value": 1000},
        "batch_outputs": {"value": 100},
    })
    assert v.verdict == "warn", v.message
    assert "exhausts early" in v.message


def test_al6_unprobeable_without_keys():
    v = probe_demo_config_reachability({"learning_rate": {"value": 0.001}})
    assert v.verdict == "unprobeable"
    assert "pool_size" in v.message and "initial_labeled" in v.message


def test_al6_unused_initial_falls_back_to_core_set_size():
    # Core-set bootstrap papers: the deriver marks initial_labeled unused
    # and the notebook bootstraps via construct_core_set(core_set_size) —
    # reachability must key on the knob the loop actually uses.
    v = probe_demo_config_reachability({
        "initial_labeled": {"value": 1000, "used_in_notebook": False},
        "core_set_size": {"value": 50},
        "pool_size": {"value": 800},
    })
    assert v.verdict == "pass", v.message
    assert "core_set_size=50" in v.message


def test_al6_unused_initial_without_core_set_is_unprobeable():
    v = probe_demo_config_reachability({
        "initial_labeled": {"value": 1000, "used_in_notebook": False},
        "pool_size": {"value": 800},
    })
    assert v.verdict == "unprobeable"
    assert "core_set_size" in v.message


# ---------------------------------------------------------------------------
# Crash attribution + AL-5 crash arm (torch needed from here down)
# ---------------------------------------------------------------------------


torch = pytest.importorskip("torch")

from probes.al_loop import probe_acquisition_contract  # noqa: E402
from probes.claims import probe_contribution_floor_al  # noqa: E402
from probes.package_loader import load_module_from_path  # noqa: E402
from probes.term_ablation import (  # noqa: E402
    attribute_invocation_crash, probe_al_selector_terms)


def _module(tmp_path, source, name="selector_under_test.py"):
    path = tmp_path / name
    path.write_text(source, encoding="utf-8")
    return load_module_from_path(path)


def test_attribution_subject_vs_harness_vs_unknown(tmp_path):
    mod = _module(tmp_path, (
        "def select_batch(model, x_unlabeled, batch_size, seed):\n"
        "    raise RuntimeError('inside subject')\n"))
    try:
        mod.select_batch(None, None, 4, 0)
    except RuntimeError as e:
        origin, site = attribute_invocation_crash(e, tmp_path)
        assert origin == "subject"
        assert site == "selector_under_test.py:2"
    # No traceback frames in either tree → unknown.
    origin, _ = attribute_invocation_crash(ValueError("bare"), tmp_path)
    assert origin == "unknown"


def test_al5_fails_negstride_selector_on_every_pool():
    # The adjudicated case: `np.argsort(...)[::-1]` indexing a torch tensor
    # crashes on EVERY legal input; signature is in the certain-defect
    # registry → hard fail naming the site and the contiguity fix.
    mod = load_module_from_path(NEGSTRIDE / "method.py")
    v = probe_acquisition_contract(mod, "select_batch")
    assert v.verdict == "fail"
    assert "method.py:402" in v.message
    assert "negative-stride" in v.message
    assert "copy()" in v.message


def test_al5_flags_uncertain_signature_instead_of_failing(tmp_path):
    # A crash the registry cannot certify (could be an undeclared
    # input-domain assumption, e.g. an image-only selector) → flag, not
    # fail. The researcher adjudicates.
    mod = _module(tmp_path, (
        "def select_batch(model, x_unlabeled, batch_size, seed):\n"
        "    if x_unlabeled.shape[1] != 784:\n"
        "        raise RuntimeError('expects flattened 28x28 images')\n"
        "    return list(range(batch_size))\n"), name="image_only.py")
    v = probe_acquisition_contract(mod, "select_batch")
    assert v.verdict == "flag_for_researcher"
    assert "every contract-conformant invocation" in v.message
    assert "input-domain assumption" in v.message


def test_al5_unrelated_negative_valueerror_flags_not_fails(tmp_path):
    # 2026-06-11 adversarial-review catch: the certain-defect needle was
    # the bare word "negative", which also matches numpy's "negative
    # dimensions are not allowed" — a pool-size-dependent crash (healthy
    # at 800, crashes on the kit's small fixture) that must route to the
    # researcher, never hard-fail with stride guidance.
    mod = _module(tmp_path, (
        "import numpy as np\n"
        "\n"
        "\n"
        "def select_batch(model, x_unlabeled, batch_size, seed):\n"
        "    buffer = np.zeros(len(x_unlabeled) - 100)\n"
        "    return list(range(batch_size))\n"), name="offset_buffer.py")
    v = probe_acquisition_contract(mod, "select_batch")
    assert v.verdict == "flag_for_researcher", v.message
    assert "negative-stride" not in v.message


def test_al5_unbound_local_flags_not_fails(tmp_path):
    # UnboundLocalError subclasses NameError; a loop variable unassigned on
    # a zero-iteration loop is input-shape-dependent, so the certain-defect
    # registry must not promote it past the flag tier.
    mod = _module(tmp_path, (
        "def select_batch(model, x_unlabeled, batch_size, seed):\n"
        "    for i in range(len(x_unlabeled) - 100):\n"
        "        last = i\n"
        "    return list(range(batch_size))[: last + batch_size]\n"),
        name="zero_iteration.py")
    v = probe_acquisition_contract(mod, "select_batch")
    assert v.verdict == "flag_for_researcher", v.message


def test_al5_calltime_missing_dep_is_unprobeable(tmp_path):
    # A lazy import of a missing dependency raises in subject frames, but
    # it's an environment limitation — same routing as load-time deps.
    mod = _module(tmp_path, (
        "def select_batch(model, x_unlabeled, batch_size, seed):\n"
        "    import nonexistent_acquisition_lib\n"
        "    return list(range(batch_size))\n"), name="lazy_dep.py")
    v = probe_acquisition_contract(mod, "select_batch")
    assert v.verdict == "unprobeable"
    assert "missing dependency" in v.message


def test_al5_stub_interface_gap_does_not_fail(tmp_path):
    # A selector reading a model attribute the harness stub lacks raises
    # AttributeError inside subject frames — but that signature is NOT in
    # the certain-defect registry (it was a harness gap twice on the
    # 2026-06-10 audits), so it must never hard-fail.
    mod = _module(tmp_path, (
        "def select_batch(model, x_unlabeled, batch_size, seed):\n"
        "    return list(range(batch_size))[:model.attr_the_stub_lacks]\n"),
        name="stub_gap.py")
    v = probe_acquisition_contract(mod, "select_batch")
    assert v.verdict != "fail"


def test_al5_known_goods_still_pass():
    pytest.importorskip("sklearn")
    good = load_module_from_path(
        EVIDENCE / "june9-gbald-run" / "method" / "method.py")
    assert probe_acquisition_contract(good, "select_batch").verdict == "pass"


def test_ct1_and_ub7_report_subject_crash_precisely():
    mod = load_module_from_path(NEGSTRIDE / "method.py")
    for verdict in (probe_contribution_floor_al(mod, "select_batch"),
                    probe_al_selector_terms(mod, "select_batch")):
        assert verdict.verdict == "unprobeable"
        assert "method.py:402" in verdict.message
        assert "crash arm (AL-5)" in verdict.message
