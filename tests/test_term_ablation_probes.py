"""UB-4 direction-consistency + UB-7 term-ablation, pinned on the two
concrete cases: pdwa's inverted/inert j-terms and june9 GBALD's dead
geometric-prior channel."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from probes.package_loader import load_module_from_path  # noqa: E402
from probes.term_ablation import (  # noqa: E402
    discover_term_calls, parse_additive_signs, probe_al_selector_terms,
    probe_composite_term_perturbation, probe_direction_consistency,
    probe_term_ranking_power)

ZOO = Path(__file__).parent / "fixtures" / "zoo"
EVIDENCE = Path(__file__).parent / "fixtures" / "evidence"

KNOWN_GOOD_MP = textwrap.dedent('''
    """Correct DWA-style composite: terms consume the candidate control and
    the obstacle penalty is SUBTRACTED from the maximized objective."""
    import torch


    def g_heading(robot_state, control, dt, dynamics, goal):
        nxt = dynamics.step(robot_state, control, dt)
        angle = torch.atan2(goal[1] - nxt[1], goal[0] - nxt[0]) - nxt[2]
        angle = torch.atan2(torch.sin(angle), torch.cos(angle))
        return -torch.abs(angle).item()


    def g_velocity(control):
        return float(control[0])


    def g_obstacle_penalty(robot_state, control, dt, dynamics,
                           obstacle_positions, d_safe):
        nxt = dynamics.step(robot_state, control, dt)
        penalty = 0.0
        for obs in obstacle_positions:
            d = torch.sqrt((obs[0] - nxt[0]) ** 2 + (obs[1] - nxt[1]) ** 2)
            penalty += torch.exp(0.5 * (d_safe - d)).item()
        return penalty


    def plan(robot_state, goal, dynamics, obstacle_positions, d_safe,
             dt=0.5):
        best, best_control = None, None
        for v in (0.2, 0.5, 0.8):
            for omega in (-0.5, 0.0, 0.5):
                control = torch.tensor([v, omega], dtype=torch.float64)
                j_h = g_heading(robot_state, control, dt, dynamics, goal)
                j_v = g_velocity(control)
                j_o = g_obstacle_penalty(robot_state, control, dt, dynamics,
                                         obstacle_positions, d_safe)
                j = j_h + 0.3 * j_v - j_o
                if best is None or j > best:
                    best, best_control = j, control
        return best_control
''')


def _write_module(tmp_path, source, name="mod_under_test.py"):
    path = tmp_path / name
    path.write_text(source, encoding="utf-8")
    return load_module_from_path(path)


# ---------------------------------------------------------------------------
# Discovery + sign parsing
# ---------------------------------------------------------------------------


def test_discovery_and_signs_on_pdwa_jterms_fresh():
    mod = load_module_from_path(ZOO / "pdwa-jterms-fresh" / "method.py")
    terms = discover_term_calls(mod.plan, mod)
    j_terms = {t for t in terms if t.startswith("compute_j_")}
    assert j_terms == {
        "compute_j_heading", "compute_j_distance", "compute_j_clearance",
        "compute_j_future_obs", "compute_j_risk_obs", "compute_j_count_obs",
    }
    signs = parse_additive_signs(mod.plan, terms)
    # The mutant's documented property: every term ADDED, penalties included.
    assert signs == {t: 1 for t in j_terms}


def test_signs_capture_subtraction_and_weights(tmp_path):
    mod = _write_module(tmp_path, KNOWN_GOOD_MP)
    terms = discover_term_calls(mod.plan, mod)
    signs = parse_additive_signs(mod.plan, terms)
    assert signs == {"g_heading": 1, "g_velocity": 1, "g_obstacle_penalty": -1}


def test_signs_none_when_composition_not_additive(tmp_path):
    mod = _write_module(tmp_path, textwrap.dedent("""
        def score_a(x):
            return x
        def score_b(x):
            return x + 1
        def combined(x):
            return max(score_a(x), score_b(x))
    """), name="nonadditive.py")
    terms = discover_term_calls(mod.combined, mod)
    assert set(terms) == {"score_a", "score_b"}
    assert parse_additive_signs(mod.combined, terms) is None


# ---------------------------------------------------------------------------
# UB-4 direction consistency
# ---------------------------------------------------------------------------


def test_ub4_fails_on_pdwa_added_penalties():
    mod = load_module_from_path(ZOO / "pdwa-jterms-fresh" / "method.py")
    v = probe_direction_consistency(mod, "plan")
    assert v.verdict == "fail"
    # The exponential proximity penalty added into a maximized objective is
    # the canonical inversion; it must be named.
    assert "compute_j_clearance" in v.message


def test_ub4_passes_on_corrected_composition(tmp_path):
    mod = _write_module(tmp_path, KNOWN_GOOD_MP)
    v = probe_direction_consistency(mod, "plan")
    assert v.verdict == "pass", v.message
    assert "g_obstacle_penalty" in v.message


def test_ub4_unprobeable_without_additive_chain(tmp_path):
    mod = _write_module(tmp_path, textwrap.dedent("""
        def score_a(x):
            return x
        def score_b(x):
            return -x
        def combined(x):
            return max(score_a(x), score_b(x))
    """), name="nonadditive2.py")
    v = probe_direction_consistency(mod, "combined")
    assert v.verdict == "unprobeable"
    assert "CT-4" in v.message


# ---------------------------------------------------------------------------
# UB-7 case 1: ranking power over the candidate grid
# ---------------------------------------------------------------------------


def test_ub7_flags_pdwa_decision_variable_free_terms():
    mod = load_module_from_path(ZOO / "pdwa-jterms-fresh" / "method.py")
    v = probe_term_ranking_power(mod, "plan")
    assert v.verdict == "flag_for_researcher"
    assert v.finding_class == "M-005"
    # The four state-only terms (the stage-4 audit's 4-of-6 finding).
    for inert in ("compute_j_heading", "compute_j_distance",
                  "compute_j_risk_obs", "compute_j_count_obs"):
        assert inert in v.message
    # The two control-consuming terms are live and must not be flagged.
    assert "compute_j_clearance" not in v.message
    assert "compute_j_future_obs" not in v.message


def test_ub7_passes_when_all_terms_rank(tmp_path):
    mod = _write_module(tmp_path, KNOWN_GOOD_MP)
    v = probe_term_ranking_power(mod, "plan")
    assert v.verdict == "pass", v.message


# ---------------------------------------------------------------------------
# UB-7 case 2: composite output-perturbation
# ---------------------------------------------------------------------------


def test_ub7_perturbation_flags_dead_channel(tmp_path):
    mod = _write_module(tmp_path, textwrap.dedent("""
        import numpy as np


        def informative_scores(x):
            return x.sum(axis=1)


        def dead_prior(x):
            # The geometric-prior class: constant at the live scale.
            return np.ones(len(x))


        def discarded_scores(x):
            return x[:, 0] * 2.0


        def select(x, batch_size, seed):
            base = informative_scores(x)
            prior = dead_prior(x)
            _ = discarded_scores(x)  # computed then discarded
            return list(np.argsort(base * prior)[-batch_size:])
    """), name="composite_dead.py")
    import numpy as np
    x = np.random.default_rng(0).normal(size=(20, 4))
    v = probe_composite_term_perturbation(
        mod, "select", lambda: mod.select(x, 4, 0))
    assert v.verdict == "flag_for_researcher"
    assert v.finding_class == "M-004"
    assert "dead_prior" in v.evidence
    assert "discarded_scores" in v.evidence
    assert "informative_scores" not in v.evidence.split("alive=")[0]


def test_ub7_perturbation_unprobeable_when_nondeterministic(tmp_path):
    mod = _write_module(tmp_path, textwrap.dedent("""
        _calls = [0]


        def scores(x):
            # Ordering flips on every invocation (unseeded-RNG stand-in,
            # made deterministic for the test itself).
            _calls[0] += 1
            sign = 1 if _calls[0] % 2 else -1
            return [sign * (i + 1) for i in range(len(x))]


        def select(x, batch_size):
            s = scores(x)
            return sorted(range(len(s)), key=lambda i: s[i])[-batch_size:]
    """), name="nondeterministic.py")
    v = probe_composite_term_perturbation(
        mod, "select", lambda: mod.select([1.0, 2.0, 3.0, 4.0], 2))
    assert v.verdict == "unprobeable"
    assert "deterministic" in v.message


def test_ub7_on_june9_gbald_flags_dead_geometric_prior():
    pytest.importorskip("sklearn")
    mod = load_module_from_path(
        EVIDENCE / "june9-gbald-run" / "method" / "method.py")
    v = probe_al_selector_terms(mod, "select_batch")
    assert v.verdict == "flag_for_researcher", f"{v.verdict}: {v.message}"
    # R_0=2000 on zero_one-scaled data makes the geometric prior ≡ 1.0:
    # its channel cannot change the selection. The run's actual bug class.
    assert "compute_geometric_prior" in v.evidence.split("alive=")[0]
    # The BALD channel orders the candidate set and must read as live.
    assert "compute_bald_scores" not in v.evidence.split("alive=")[0]


def test_kit_serves_embedding_hook_selectors(tmp_path):
    # BADGE's spec declares forward_with_embedding as the model interface;
    # a bare Sequential stub left every selector probe unprobeable on the
    # 2026-06-10 matrix-row-3 audit. The kit's stub must serve both
    # interfaces (plain forward stays the GBALD path).
    mod = _write_module(tmp_path, textwrap.dedent("""
        import torch


        def gradient_scores(model, x):
            model.eval()
            with torch.no_grad():
                logits, emb = model.forward_with_embedding(x)
            probs = torch.softmax(logits, dim=1)
            margin = probs.max(dim=1).values
            return emb.norm(dim=1) * (1 - margin)


        def select_batch(model, x_unlabeled, batch_size, seed):
            scores = gradient_scores(model, x_unlabeled)
            return torch.argsort(scores)[-batch_size:].tolist()
    """), name="embedding_selector.py")
    v = probe_al_selector_terms(mod, "select_batch")
    assert v.verdict == "pass", f"{v.verdict}: {v.message}"


# ---------------------------------------------------------------------------
# UB-7 membership modes — the gate-then-rerank blind spot (GBALD 2026-06-30
# M-004: a top-b INDEX-vector gate is invisible to cyclic shifts, which
# permute the same set, and dup-drop nicks a single element; the membership-
# displacement mode replaces half the gate with in-domain strangers so the
# gate's contribution is observable). Real-selector validation: the delivered
# GBALD selector flips flag→pass with index_domain, the delivered BADGE
# selector stays pass (2026-07-02).
# ---------------------------------------------------------------------------

GATE_THEN_RERANK = textwrap.dedent("""
    import numpy as np


    def uncertainty_gate(x, n_top):
        # Top-n_top candidate INDICES by an informative score (the GBALD
        # compute_bald_scores shape: an index vector, set-semantic consumer).
        # Scores read ONLY feature 0 while the rerank reads features 1:, so
        # the two signals are decorrelated — dropping the single top-scored
        # point (dup-drop) rarely moves the geometric picks, which is the
        # live GBALD blind-spot shape.
        return np.argsort(-x[:, 0])[:n_top]


    def diversity_rerank(candidates, n_select):
        # Order-insensitive max-min rerank within the gated subset: returns
        # POSITIONS of the selected candidates (farthest-first).
        d = np.linalg.norm(candidates[:, 1:], axis=1)
        chosen = [int(np.argmax(d))]
        while len(chosen) < n_select:
            rest = [i for i in range(len(candidates)) if i not in chosen]
            gaps = [min(np.linalg.norm(candidates[i, 1:] - candidates[j, 1:])
                        for j in chosen) for i in rest]
            chosen.append(rest[int(np.argmax(gaps))])
        return np.asarray(sorted(chosen))


    def select(x, batch_size, seed):
        gate = uncertainty_gate(x, n_top=8)
        positions = diversity_rerank(x[gate], n_select=batch_size)
        return sorted(int(i) for i in gate[positions])
""")


def test_ub7_gate_membership_dead_without_domain_hint(tmp_path):
    # Documents the blind spot: without index_domain the gate term reads
    # dead even though it fully determines the candidate set.
    mod = _write_module(tmp_path, GATE_THEN_RERANK, name="gate_rerank_a.py")
    import numpy as np
    x = np.random.default_rng(0).normal(size=(20, 4))
    v = probe_composite_term_perturbation(
        mod, "select", lambda: mod.select(x, 3, 0))
    assert v.verdict == "flag_for_researcher", v.message
    assert "uncertainty_gate" in v.evidence.split("alive=")[0]


def test_ub7_gate_membership_alive_with_domain_hint(tmp_path):
    mod = _write_module(tmp_path, GATE_THEN_RERANK, name="gate_rerank_b.py")
    import numpy as np
    x = np.random.default_rng(0).normal(size=(20, 4))
    v = probe_composite_term_perturbation(
        mod, "select", lambda: mod.select(x, 3, 0), index_domain=len(x))
    assert v.verdict == "pass", v.message
    assert "uncertainty_gate" in (v.evidence or "")


def test_ub7_constant_prior_stays_dead_under_membership_modes(tmp_path):
    # The ≡1.0-prior catch must survive the new modes: min-max swap is an
    # identity on constants and displacement does not apply to floats.
    mod = _write_module(tmp_path, textwrap.dedent("""
        import numpy as np


        def informative_scores(x):
            return x.sum(axis=1)


        def dead_prior(x):
            return np.ones(len(x))


        def select(x, batch_size, seed):
            return list(np.argsort(
                informative_scores(x) * dead_prior(x))[-batch_size:])
    """), name="composite_dead_hinted.py")
    import numpy as np
    x = np.random.default_rng(0).normal(size=(20, 4))
    v = probe_composite_term_perturbation(
        mod, "select", lambda: mod.select(x, 4, 0), index_domain=len(x))
    assert v.verdict == "flag_for_researcher"
    assert "dead_prior" in v.evidence.split("alive=")[0]


def test_perturb_membership_mode_edge_cases():
    import numpy as np

    from probes.term_ablation import _perturb

    # Displacement needs a domain and integer dtype.
    idx = np.array([4, 2, 7])
    assert _perturb(idx, -2, index_domain=None) is None
    assert _perturb(np.array([0.5, 0.2]), -2, index_domain=10) is None
    # A vector covering its whole domain has no strangers to displace with.
    assert _perturb(np.array([1, 0, 2]), -2, index_domain=3) is None
    # Out-of-domain values disqualify the vector (not an index vector).
    assert _perturb(np.array([4, 99]), -2, index_domain=10) is None
    # Displacement replaces the first half with in-domain strangers.
    out = _perturb(np.array([4, 2, 7, 1]), -2, index_domain=8)
    assert out is not None and set(out.tolist()) != {4, 2, 7, 1}
    assert all(0 <= v < 8 for v in out.tolist())
    # Min-max swap trades the extremes and is an identity on constants.
    swapped = _perturb(np.array([1.0, 5.0, 3.0]), -1)
    assert swapped.tolist() == [5.0, 1.0, 3.0]
    assert _perturb(np.array([2.0, 2.0, 2.0]), -1).tolist() == [2.0, 2.0, 2.0]
