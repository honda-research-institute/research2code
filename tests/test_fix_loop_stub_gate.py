"""Fix-loop third option (queue item 7) — the deterministic gate module.

The reachability gates G1-G5 decide, without any agent judgment, whether the
smoke loop's terminal give-up may convert the single failing component to a
stub. Unit level: the gates, the G4 plausibility screen, the maintainer-approved
file-to-element mapping default, and the draft-cell marker edit. The wired
adversarial fixtures live in test_stage_3c.py.
"""

from __future__ import annotations

from fix_loop_stub_gate import (GateVerdict, diagnosis_plausibility_screen,
                                evaluate_stub_gates,
                                map_failing_file_to_element,
                                replace_draft_cell_with_stub_marker)


def _iter(target="method/training.py", frame="method/training.py:12 in fn",
          root_cause="Uncatalogued bug.", n=0):
    return {
        "iteration": n,
        "failing_cell": n,
        "diagnosis": {"target_agent": "r2c-architecture-coder",
                      "target_file": target,
                      "root_cause": root_cause,
                      "proposed_fix": "fix it"},
        "deepest_owned_frame": frame,
    }


def _evaluate(iters, *, current_frame="method/training.py:12 in fn",
              cell_source="import torch\nx = model(batch)\n",
              existing=None, contract=None):
    return evaluate_stub_gates(
        give_up_point="test give-up",
        prior_iterations=iters,
        current_frame=current_frame,
        cell_source=cell_source,
        existing_stubs=existing or [],
        contract_elements=contract or [],
    )


# --- the mapping default (maintainer-approved 2026-07-10) --------------------------


def test_core_file_maps_to_unique_contract_core_element():
    contract = [
        {"element_id": "geodesic-core-set", "role": "core_methodology"},
        {"element_id": "acq-scoring", "role": "supporting_mechanism"},
    ]
    eid, role, basis = map_failing_file_to_element(contract, "method/method.py")
    assert (eid, role) == ("geodesic-core-set", "core")
    assert "unique" in basis


def test_core_file_without_unique_core_element_synthesizes():
    eid, role, basis = map_failing_file_to_element([], "method/method.py")
    assert (eid, role) == ("method-method", "core")
    assert "synthesized" in basis


def test_supporting_file_synthesizes_supporting():
    eid, role, basis = map_failing_file_to_element(
        [{"element_id": "x", "role": "core_methodology"}],
        "method/training.py")
    assert (eid, role) == ("method-training", "supporting")


# --- G2 breadth ---------------------------------------------------------------


def test_g2_single_target_never_fires():
    iters = [_iter(n=0), _iter(n=1), _iter(n=2)]  # same target thrice
    v = _evaluate(iters)
    assert not v.fire and not v.halt_on_g4
    assert any("G2 FAIL" in r for r in v.reasons)


# --- G3 isolation ---------------------------------------------------------------


def test_g3_wandering_frames_never_fire():
    iters = [_iter(n=0), _iter(target="method/method.py", n=1,
                               frame="method/method.py:5 in f")]
    v = _evaluate(iters)
    assert not v.fire and not v.halt_on_g4
    assert any("G3 FAIL" in r for r in v.reasons)


def test_g3_missing_frame_never_fires():
    iters = [_iter(n=0), _iter(target="method/method.py", n=1, frame=None)]
    v = _evaluate(iters)
    assert not v.fire
    assert any("G3 FAIL" in r for r in v.reasons)


# --- G5 one per run -------------------------------------------------------------


def test_g5_existing_fix_loop_stub_blocks():
    iters = [_iter(n=0), _iter(target="method/method.py", n=1)]
    v = _evaluate(iters, existing=[{
        "element_id": "prior", "role": "supporting",
        "work_order": "work_orders/prior.md",
        "stub_path": "method/utils.py"}])
    assert not v.fire
    assert any("G5 FAIL" in r for r in v.reasons)


def test_g5_gate_producer_record_without_stub_path_does_not_block():
    iters = [_iter(n=0), _iter(target="method/method.py", n=1)]
    v = _evaluate(iters, existing=[{
        "element_id": "gate-obligation", "role": "supporting",
        "work_order": "work_orders/gate-obligation.md"}])
    assert v.fire, v.summary()


# --- G4 plausibility screen -----------------------------------------------------


def test_g4_complexity_diagnosis_on_plot_cell_halts():
    """The detr audit shape: cap burned 'fixing' a bare plot cell diagnosed
    as pathological complexity — the diagnosis chain is broken, HALT."""
    iters = [_iter(n=0),
             _iter(target="method/method.py", n=1,
                   root_cause="Pathological complexity: an infinite loop "
                              "in the scoring path.")]
    v = _evaluate(iters, cell_source="plt.plot(losses)\nplt.show()\n")
    assert v.halt_on_g4 and not v.fire
    assert any("G4 MISMATCH" in r for r in v.reasons)


def test_g4_complexity_diagnosis_with_a_loop_passes():
    iters = [_iter(n=0),
             _iter(target="method/method.py", n=1,
                   root_cause="Runaway complexity in the loop.")]
    v = _evaluate(iters,
                  cell_source="for epoch in range(3):\n    train(model)\n")
    assert v.fire, v.summary()


def test_g4_shape_mismatch_needs_tensor_ops():
    ok, verdict = diagnosis_plausibility_screen(
        "shape mismatch between boxes and labels",
        "print('hello world')")
    assert not ok and "G4 MISMATCH" in verdict
    ok, _ = diagnosis_plausibility_screen(
        "shape mismatch between boxes and labels",
        "x = torch.zeros(3).reshape(1, 3)")
    assert ok


def test_g4_uncataloged_class_passes_with_no_opinion():
    ok, verdict = diagnosis_plausibility_screen(
        "Uncatalogued bug in the io path.", "print('x')")
    assert ok and "no opinion" in verdict


# --- all gates hold -------------------------------------------------------------


def test_all_gates_fire_with_mapping_and_evidence_trail():
    iters = [_iter(n=0), _iter(target="method/method.py", n=1)]
    v = _evaluate(iters)
    assert v.fire and not v.halt_on_g4
    assert v.component_file == "method/training.py"
    assert (v.element_id, v.role) == ("method-training", "supporting")
    # The full evidence trail is recorded for the assumptions entry.
    assert sum(1 for r in v.reasons if "pass" in r) >= 5
    assert isinstance(v, GateVerdict) and v.summary()


# --- draft marker edit ----------------------------------------------------------


DRAFT = """\
# %% [markdown]
# # Title

# %%
print('cell 0')

# %%
result = run_component(x)
print(result)

# %%
print('cell 2')
"""


def test_replace_locates_unique_cell_and_inserts_marker():
    out = replace_draft_cell_with_stub_marker(
        DRAFT, "result = run_component(x)\nprint(result)\n", "my-elem")
    assert out is not None
    assert "# %% PLACEHOLDER: component_stub:my-elem" in out
    assert "run_component" not in out
    assert "print('cell 0')" in out and "print('cell 2')" in out


def test_replace_refuses_ambiguous_or_missing_cells():
    dup = DRAFT + "\n# %%\nresult = run_component(x)\nprint(result)\n"
    assert replace_draft_cell_with_stub_marker(
        dup, "result = run_component(x)\nprint(result)\n", "e") is None
    assert replace_draft_cell_with_stub_marker(
        DRAFT, "never_in_draft()\n", "e") is None
    assert replace_draft_cell_with_stub_marker(DRAFT, "", "e") is None
