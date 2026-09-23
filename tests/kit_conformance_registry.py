"""Kit conformance registry — the contract every verification kit must pass.

Three kits exist (active learning, knowledge distillation, motion
planning), which meets the two-concrete-cases bar for this abstraction.
Each kit registers a declarative `KitConformance` entry; the parametrized
suite in `test_kit_conformance.py` enforces the same four invariants for
every entry, so a future kit gets the whole contract by registering, not
by remembering to hand-write the right tests:

1. **Catches the family's zoo defect.** At least one case built from a
   known-bad zoo artifact (or the sanctioned synthetic reconstruction
   when no artifact was ever captured) where the kit's probes must flag.
2. **Never false-fails the healthy fixture.** Known-good cases where the
   named probes must pass and NOTHING in the kit may emit a demoting
   verdict.
3. **Never fabricates stand-ins.** A case the kit genuinely cannot bind
   must come back all-`unprobeable` — never a pass or fail computed
   against invented inputs, and never silence.
4. **Deterministic.** Same case, same verdicts, twice.

Plus the growth gate in the suite: every contribution-probe prefix the
delivery label derivation counts as certification evidence must have a
registered entry here. A kit that can mint `verified` labels without
conformance coverage is exactly the "subtly wrong kit quietly mislabels
a package" failure this file exists to prevent.

Adapters take a scratch directory and return the kit's verdict list,
mirroring how `run_probes.py` invokes each kit (including its convention
of converting a reason-string kit into per-probe unprobeable verdicts).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import pytest

torch = pytest.importorskip("torch")

REPO = Path(__file__).resolve().parent.parent
ZOO = REPO / "tests" / "fixtures" / "zoo"
TSF_ZOO_CASES = json.loads((
    ZOO / "tsf-probe-reconstructions" / "cases.json"
).read_text(encoding="utf-8"))

from probes import ProbeVerdict  # noqa: E402


# ---------------------------------------------------------------------------
# Declarative shapes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Expect:
    """One expected verdict row within a case."""

    probe_id: str
    verdict: str
    message_contains: str | None = None
    finding_class: str | None = None


@dataclass(frozen=True)
class Case:
    case_id: str
    run: Callable[[Path], list[ProbeVerdict]]
    expect: tuple[Expect, ...] = ()


@dataclass(frozen=True)
class KitConformance:
    kit_id: str
    prefix: str                    # the contribution probe-id prefix
    defects: tuple[Case, ...]      # invariant 1 — each must flag as declared
    healthy: tuple[Case, ...]      # invariant 2 — passes, zero demoters
    unbindable: tuple[Case, ...]   # invariant 3 — all unprobeable, non-empty


# Verdicts that demote a delivery — a healthy fixture must never see one.
DEMOTING_VERDICTS = frozenset({"fail", "flag_for_researcher"})


# ---------------------------------------------------------------------------
# Dual-loader module cases (RCA 2026-08-03 finding 7 prevention)
# ---------------------------------------------------------------------------
#
# The stage-1 core-set kit shipped dead on arrival because its only
# conformance ran through the single-file loader, where the binder's
# module-ownership bug is invisible: production hands the imported PACKAGE
# and every function lives in a submodule. Every module-shaped case now runs
# under BOTH loaders, so a binder that works only on the friendly shape can
# never register green again.


def _packageize(method_body: str, tmp: Path) -> Path:
    """Build a production-shaped run dir from a single-file fixture body:
    method/method.py plus the generated-style re-exporting __init__."""
    run_dir = tmp / "pkg_run"
    pkg = run_dir / "method"
    pkg.mkdir(parents=True)
    (pkg / "method.py").write_text(method_body, encoding="utf-8")
    (pkg / "__init__.py").write_text(
        '"""Conformance mirror of the generated re-exporting __init__."""\n'
        "from .method import *  # noqa: F401,F403\n",
        encoding="utf-8")
    return run_dir


def _module_case(case_id: str, verdicts_fn, *, method_py: Path | None = None,
                 source=None, expect: tuple = ()) -> tuple:
    """Two Cases for one module-shaped fixture: the single-file loader and
    the production package loader. `source` is a str or a zero-arg callable
    (kept lazy so registry import stays light); `method_py` is a zoo path."""

    def _body() -> str:
        if source is not None:
            return source() if callable(source) else source
        return method_py.read_text(encoding="utf-8")

    def run_single(tmp: Path):
        from probes.package_loader import load_module_from_path  # noqa: PLC0415

        if method_py is not None:
            py = method_py
        else:
            py = tmp / "single_file.py"
            py.write_text(_body(), encoding="utf-8")
        return verdicts_fn(load_module_from_path(py))

    def run_packaged(tmp: Path):
        from probes.package_loader import imported_method_package  # noqa: PLC0415

        run_dir = _packageize(_body(), tmp)
        with imported_method_package(run_dir) as method:
            return verdicts_fn(method)

    return (
        Case(f"{case_id} [single-file loader]", run_single, expect),
        Case(f"{case_id} [production package loader]", run_packaged, expect),
    )


# ---------------------------------------------------------------------------
# Active learning (AL-)
# ---------------------------------------------------------------------------


def _al_zoo(nb_dir: str):
    def run(_tmp: Path) -> list[ProbeVerdict]:
        from probes.al_loop import run_al_loop_microharness  # noqa: PLC0415

        return run_al_loop_microharness(ZOO / nb_dir / "notebook.ipynb")

    return run


def _al_loopless(tmp: Path) -> list[ProbeVerdict]:
    from probes.al_loop import run_al_loop_microharness  # noqa: PLC0415

    nb = {"cells": [{"cell_type": "code", "id": "a",
                     "source": ["x = 1\n"], "outputs": []}]}
    path = tmp / "loopless.ipynb"
    path.write_text(json.dumps(nb), encoding="utf-8")
    return run_al_loop_microharness(path)


def _al_stage1_verdicts(mod) -> list[ProbeVerdict]:
    from probes.al_stage1 import run_al_stage1_probes  # noqa: PLC0415

    return run_al_stage1_probes(mod)


# A core-set constructor whose signature exposes nothing bindable: the
# kit must return its reason and every probe reads unprobeable — never
# a selection computed against invented inputs.
_AL_OPAQUE_SRC = (
    "def construct_core_set(problem_spec):\n"
    "    return []\n"
)

# A foreign callable wearing the family name: the binder must refuse it and
# the disclosure must name the rejected owner instead of claiming nothing
# matched (RCA 2026-08-03 finding 7's misleading-skip half).
_AL_FOREIGN_ALIAS_SRC = (
    "from numpy import argsort as coreset_argsort\n"
)


AL_CONFORMANCE = KitConformance(
    kit_id="active_learning",
    prefix="AL-",
    defects=(
        Case(
            "badge-badloop (zoo, frozen-evidence M-002/M-003)",
            _al_zoo("badge-badloop-reconstructed"),
            (
                Expect("AL-1", "fail", finding_class="M-002"),
                Expect("AL-4", "fail", message_contains="warm-start"),
            ),
        ),
        *_module_case(
            "gbald-lr74 core-set (zoo, frozen-evidence): constant-score "
            "argmax — the Stage-1 selection carries no data signal",
            _al_stage1_verdicts,
            method_py=ZOO / "gbald-lr74-never-learns" / "method.py",
            expect=(
                Expect("AL-S1-2", "fail",
                       message_contains="position-degenerate",
                       finding_class="M-004"),
                # The defect is selection-level, not contract-level: the
                # indices themselves are well-formed.
                Expect("AL-S1-1", "pass"),
            ),
        ),
    ),
    healthy=(
        Case(
            "badge-goodmethod (zoo, corrected loop)",
            _al_zoo("badge-goodmethod-badnotebook"),
            (Expect("AL-1", "pass"), Expect("AL-4", "pass")),
        ),
        Case(
            "gbald-double-retrain (zoo, the false-positive guard: growth "
            "steps of 0 are loop-shape variation)",
            _al_zoo("gbald-double-retrain-loop"),
            (Expect("AL-1", "pass"), Expect("AL-4", "pass")),
        ),
        *_module_case(
            "gbald-verified-coreset (zoo, harvested verbatim from the "
            "2026-07-03 verified delivery): the known-good Stage-1 shape",
            _al_stage1_verdicts,
            method_py=ZOO / "gbald-verified-coreset" / "method.py",
            expect=(
                Expect("AL-S1-1", "pass"),
                Expect("AL-S1-2", "pass"),
                Expect("AL-S1-3", "pass"),
            ),
        ),
    ),
    unbindable=(
        Case("notebook without a recognizable loop", _al_loopless),
        *_module_case("opaque single-argument core-set signature",
                      _al_stage1_verdicts, source=_AL_OPAQUE_SRC),
        *_module_case("foreign numpy alias wearing the core-set name",
                      _al_stage1_verdicts, source=_AL_FOREIGN_ALIAS_SRC),
    ),
)


# ---------------------------------------------------------------------------
# Knowledge distillation (KD-)
# ---------------------------------------------------------------------------


def _kd_teacher_ignoring(_tmp: Path) -> list[ProbeVerdict]:
    # Sanctioned synthetic: the bev-distill quality-no-op class was never
    # captured as an artifact (see test_kd_probes module docstring), so the
    # reconstruction is a loss that looks like distillation and never reads
    # the teacher's content.
    from probes.kd import probe_teacher_signal_influence  # noqa: PLC0415

    def fake_distillation_loss(student_feats, teacher_feats):
        return (student_feats ** 2).mean() + 0.0 * teacher_feats.sum() * 0

    return [probe_teacher_signal_influence(fake_distillation_loss)]


def _kd_loss_verdicts(mod) -> list[ProbeVerdict]:
    from probes.kd import (probe_teacher_signal_influence,  # noqa: PLC0415
                           probe_temperature_sensitivity)

    out = [probe_teacher_signal_influence(mod.weighted_feature_loss)]
    kd2 = probe_temperature_sensitivity(mod.weighted_feature_loss)
    if kd2 is not None:
        out.append(kd2)
    return out


def _kd_ragged_targets(tmp: Path) -> list[ProbeVerdict]:
    # A model-and-batch loss whose contract declares a batch key no
    # synthesizer can honor. The kit must return its reason string and the
    # battery convention (run_probes.py) converts that into per-probe
    # unprobeable rows — mirrored here.
    from probes.kd import kd_loss_kit  # noqa: PLC0415
    from tests.test_kd_probes import _KIT_LOSS_GOOD  # noqa: PLC0415

    py = tmp / "kd_ragged.py"
    py.write_text(_KIT_LOSS_GOOD, encoding="utf-8")
    from probes.package_loader import load_module_from_path  # noqa: PLC0415

    mod = load_module_from_path(py)
    contract = {"pluggable_component": {"batch_dict_shape": {
        "student_inputs": "(B, 3, H_img, W_img)",
        "teacher_inputs": "(B, N_points, 3)",
        "annotations": "ragged per-image record list, keys vary by dataset",
    }}}
    kit = kd_loss_kit(mod, "compute_distillation_loss",
                      arch_contract=contract)
    assert isinstance(kit, str), (
        f"conformance harness error: expected a reason string, got {kit!r}")
    return [ProbeVerdict("KD-1", "unprobeable", kit),
            ProbeVerdict("KD-2", "unprobeable", kit)]


KD_CONFORMANCE = KitConformance(
    kit_id="knowledge_distillation",
    prefix="KD-",
    defects=(
        Case(
            "teacher-ignoring loss (sanctioned synthetic, quality-no-op "
            "class)",
            _kd_teacher_ignoring,
            (Expect("KD-1", "fail", message_contains="no effect"),),
        ),
    ),
    healthy=(
        *_module_case(
            "kd-wrong-op (zoo) weighted_feature_loss — teacher-responsive",
            _kd_loss_verdicts,
            method_py=ZOO / "kd-wrong-op-reconstructed" / "method.py",
            expect=(Expect("KD-1", "pass"),),
        ),
    ),
    unbindable=(
        Case("contract declares an unsynthesizable batch key",
             _kd_ragged_targets),
    ),
)


# ---------------------------------------------------------------------------
# Motion planning (MP-)
# ---------------------------------------------------------------------------


def _mp_verdicts(mod) -> list[ProbeVerdict]:
    from probes.motion_planning import (mp_planner_kit,  # noqa: PLC0415
                                        probe_goal_progress,
                                        probe_steering_responsiveness)

    kit = mp_planner_kit(mod, "plan")
    return [probe_steering_responsiveness(mod, "plan", kit=kit),
            probe_goal_progress(mod, "plan", kit=kit)]


def _state_goal_planner_src() -> str:
    from tests.test_motion_planning_probes import \
        _KIT_GOOD_STATE_GOAL_PLANNER  # noqa: PLC0415

    return _KIT_GOOD_STATE_GOAL_PLANNER


# A planner whose signature exposes nothing the kit can bind a problem
# instance to. The kit must say so, never invent a start/goal/world.
_MP_OPAQUE_SRC = (
    "def plan(problem_spec):\n"
    "    return None\n"
)


MP_CONFORMANCE = KitConformance(
    kit_id="motion_planning",
    prefix="MP-",
    defects=(
        *_module_case(
            "idbrrt-no-path (zoo reconstruction) — planner never reaches "
            "the goal",
            _mp_verdicts,
            method_py=ZOO / "idbrrt-no-path-reconstructed" / "method.py",
            expect=(
                Expect("MP-4", "flag_for_researcher",
                       message_contains="no trajectory"),
                # MP-3 routing to the goal-progress check is DESIGNED here
                # (no first decision exists to judge), so the defect case
                # pins it as unprobeable-with-routing, not as a flag.
                Expect("MP-3", "unprobeable",
                       message_contains="goal-progress check"),
            ),
        ),
    ),
    healthy=(
        *_module_case(
            "state-goal greedy planner (paradigm-template interface shape)",
            _mp_verdicts,
            source=_state_goal_planner_src,
            expect=(Expect("MP-3", "pass"), Expect("MP-4", "pass")),
        ),
    ),
    unbindable=(
        *_module_case("opaque single-argument planner signature",
                      _mp_verdicts, source=_MP_OPAQUE_SRC),
    ),
)


# ---------------------------------------------------------------------------
# Time-series forecasting (TSF-)
# ---------------------------------------------------------------------------


def _tsf_runtime_case(
    tmp: Path,
    *,
    mode: str = "healthy",
    representation: str | None = None,
    autoregressive: bool = False,
    heldout: bool = True,
    unsupported: bool = False,
) -> list[ProbeVerdict]:
    """Production-shaped schema-2 adapter used by the shared kit contract."""

    from probes.time_series_forecasting import (  # noqa: PLC0415
        PROBE_REFS,
        run_time_series_forecasting_probes,
    )
    from tests.test_time_series_forecasting_probes import (  # noqa: PLC0415
        _execution_plan,
        _ready_heldout_evidence,
        _write_generated_package,
    )

    run_dir = _write_generated_package(tmp / "run", mode=mode)
    plan = _execution_plan(
        representation=representation,
        autoregressive=autoregressive,
    )
    if unsupported:
        plan = dict(plan)
        plan["source_contract_schema"] = "1.0.0"
    evidence = _ready_heldout_evidence()
    if mode == "known_bad":
        evidence = _ready_heldout_evidence(
            model=[1.0e-12, -1.0e-12], repeat=[10.0, 20.0]
        )
    refs = set(PROBE_REFS)
    return run_time_series_forecasting_probes(
        run_dir,
        enabled_refs=refs,
        execution_plan=plan,
        demo_skill_evidence=evidence if heldout else None,
    )


def _tsf_zoo_case(tmp: Path, case_id: str) -> list[ProbeVerdict]:
    case = TSF_ZOO_CASES[case_id]
    return _tsf_runtime_case(
        tmp,
        mode=case["mode"],
        representation=case["representation"],
        autoregressive=case["autoregressive"],
    )


def _tsf_output_bad(tmp: Path) -> list[ProbeVerdict]:
    return _tsf_zoo_case(tmp, "audited_output_bad")


def _tsf_relational_self_severed(tmp: Path) -> list[ProbeVerdict]:
    return _tsf_zoo_case(tmp, "relational_self_path_severed")


def _tsf_graph_free_good(tmp: Path) -> list[ProbeVerdict]:
    return _tsf_zoo_case(tmp, "graph_free_adjacent_good")


def _tsf_sparse_ancestral_good(tmp: Path) -> list[ProbeVerdict]:
    return _tsf_zoo_case(tmp, "synthetic_ancestral_good")


def _tsf_sparse_good(tmp: Path) -> list[ProbeVerdict]:
    return _tsf_zoo_case(tmp, "archived_11_sparse_good")


def _tsf_dense_good(tmp: Path) -> list[ProbeVerdict]:
    return _tsf_zoo_case(tmp, "archived_7_dense_good")


def _tsf_schema1_unbindable(tmp: Path) -> list[ProbeVerdict]:
    return _tsf_runtime_case(tmp, heldout=False, unsupported=True)


def _tsf_independent_autoregressive_bad(tmp: Path) -> list[ProbeVerdict]:
    return _tsf_runtime_case(tmp, mode="healthy", autoregressive=True)


def _tsf_heldout_skill_bad(tmp: Path) -> list[ProbeVerdict]:
    from probes.time_series_forecasting import (  # noqa: PLC0415
        PROBE_REFS,
        run_time_series_forecasting_probes,
    )
    from tests.test_time_series_forecasting_probes import (  # noqa: PLC0415
        _execution_plan,
        _ready_heldout_evidence,
        _write_generated_package,
    )

    run_dir = _write_generated_package(tmp / "run")
    return run_time_series_forecasting_probes(
        run_dir,
        enabled_refs=set(PROBE_REFS),
        execution_plan=_execution_plan(),
        demo_skill_evidence=_ready_heldout_evidence(
            model=[9.0, 19.0], repeat=[10.0, 20.0]
        ),
    )


TSF_CONFORMANCE = KitConformance(
    kit_id="time_series_forecasting",
    prefix="TSF-",
    defects=(
        Case(
            "audited pdfgnn output classes reconstructed: broadcast samples, "
            "severed history response, and collapsed magnitude",
            _tsf_output_bad,
            (
                Expect("TSF-1", "fail"),
                Expect("TSF-3", "fail"),
                Expect("TSF-4", "fail"),
                Expect("TSF-5", "fail"),
            ),
        ),
        Case(
            "night3/_7 executed relational known-bad class: isolated own "
            "history path severed on the asymmetric graph",
            _tsf_relational_self_severed,
            (Expect("TSF-3", "fail"),),
        ),
        Case(
            "autoregressive contract served by independent horizon marginals",
            _tsf_independent_autoregressive_bad,
            (Expect("TSF-2", "fail"),),
        ),
        Case(
            "valid held-out model loses to required repeat-last comparator",
            _tsf_heldout_skill_bad,
            (Expect("TSF-4", "fail"),),
        ),
    ),
    healthy=(
        Case(
            "graph-free forecasting adjacent control",
            _tsf_graph_free_good,
            (
                Expect("TSF-1", "pass"),
                Expect("TSF-2", "not_applicable"),
                Expect("TSF-3", "pass"),
                Expect("TSF-4", "pass"),
                Expect("TSF-5", "pass"),
            ),
        ),
        Case(
            "synthetic sparse ancestral-path positive control",
            _tsf_sparse_ancestral_good,
            tuple(Expect(f"TSF-{index}", "pass") for index in range(1, 6)),
        ),
        Case(
            "archived _11 sparse representation control (output path not "
            "claimed ancestral)",
            _tsf_sparse_good,
            (
                Expect("TSF-1", "pass"),
                Expect("TSF-2", "not_applicable"),
                Expect("TSF-3", "pass"),
                Expect("TSF-4", "pass"),
                Expect("TSF-5", "pass"),
            ),
        ),
        Case(
            "archived _7 dense representation with restored self path",
            _tsf_dense_good,
            (
                Expect("TSF-1", "pass"),
                Expect("TSF-2", "not_applicable"),
                Expect("TSF-3", "pass"),
                Expect("TSF-4", "pass"),
                Expect("TSF-5", "pass"),
            ),
        ),
    ),
    unbindable=(
        Case(
            "legacy schema-1 runtime and absent typed evaluation role",
            _tsf_schema1_unbindable,
        ),
    ),
)


# ---------------------------------------------------------------------------
# The registry
# ---------------------------------------------------------------------------

KIT_REGISTRY: tuple[KitConformance, ...] = (
    AL_CONFORMANCE,
    KD_CONFORMANCE,
    MP_CONFORMANCE,
    TSF_CONFORMANCE,
)
