"""AL Stage-1 core-set probe kit (queue item 9) — binding-level unit tests.

The behavioral invariants (defect flags, healthy no-false-fail, unbindable
all-unprobeable, determinism) live in the kit conformance suite via
AL_CONFORMANCE's Stage-1 cases; this file pins the binding rules the suite
does not exercise directly.
"""

from __future__ import annotations

import sys
import textwrap
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent

pytest.importorskip("torch")

from probes.al_stage1 import al_stage1_kit, run_al_stage1_probes  # noqa: E402
from probes.package_loader import load_module_from_path  # noqa: E402


def _mod(tmp_path, source, name="m.py"):
    p = tmp_path / name
    p.write_text(textwrap.dedent(source), encoding="utf-8")
    return load_module_from_path(p)


def test_no_convention_match_is_unprobeable_with_the_convention_named(tmp_path):
    mod = _mod(tmp_path, """
        def select_batch(model, x_unlabeled, batch_size):
            return list(range(batch_size))
        """)
    kit = al_stage1_kit(mod)
    assert isinstance(kit, str) and "naming convention" in kit
    verdicts = run_al_stage1_probes(mod)
    assert [v.verdict for v in verdicts] == ["unprobeable"] * 3
    assert [v.probe_id for v in verdicts] == ["AL-S1-1", "AL-S1-2", "AL-S1-3"]


def test_ambiguous_match_prefers_the_exact_family_name(tmp_path):
    mod = _mod(tmp_path, """
        def coreset_helper(x_pool):
            return [0]

        def construct_core_set(x_pool, core_set_size, eta=0.9, seed=0):
            return list(range(core_set_size))
        """)
    kit = al_stage1_kit(mod)
    assert isinstance(kit, dict) and kit["fn_name"] == "construct_core_set"


def test_ambiguous_match_without_the_exact_name_is_unprobeable(tmp_path):
    mod = _mod(tmp_path, """
        def build_coreset(x_pool):
            return [0]

        def coreset_indices(x_pool):
            return [1]
        """)
    kit = al_stage1_kit(mod)
    assert isinstance(kit, str) and "ambiguous" in kit


def test_duplicate_indices_fail_the_output_contract(tmp_path):
    mod = _mod(tmp_path, """
        def construct_core_set(x_pool, core_set_size, seed=0):
            return [0] * core_set_size
        """)
    verdicts = {v.probe_id: v for v in run_al_stage1_probes(mod)}
    assert verdicts["AL-S1-1"].verdict == "fail"
    assert "duplicate" in verdicts["AL-S1-1"].message


def test_missing_eta_channel_is_unprobeable_not_a_guess(tmp_path):
    mod = _mod(tmp_path, """
        def construct_core_set(x_pool, core_set_size, seed=0):
            import torch
            d = ((x_pool[None, 0] - x_pool) ** 2).sum(-1)
            return [int(i) for i in d.argsort(descending=True)[:core_set_size]]
        """)
    verdicts = {v.probe_id: v for v in run_al_stage1_probes(mod)}
    assert verdicts["AL-S1-3"].verdict == "unprobeable"
    assert "eta" in verdicts["AL-S1-3"].message


def test_dead_eta_channel_is_flagged_for_researcher(tmp_path):
    mod = _mod(tmp_path, """
        def construct_core_set(x_pool, core_set_size, eta=0.9, seed=0):
            # eta accepted, never used: the paper's mechanism is inert.
            import torch
            d = ((x_pool - x_pool.mean(0)) ** 2).sum(-1)
            return [int(i) for i in d.argsort(descending=True)[:core_set_size]]
        """)
    verdicts = {v.probe_id: v for v in run_al_stage1_probes(mod)}
    assert verdicts["AL-S1-3"].verdict == "flag_for_researcher"
    assert verdicts["AL-S1-3"].finding_class == "M-004"


# ---------------------------------------------------------------------------
# Package ownership (RCA 2026-08-03 finding 7): the binder must accept the
# package's own re-exported symbols and refuse genuinely foreign callables,
# naming what it rejected.
# ---------------------------------------------------------------------------


_PKG_CORESET_BODY = textwrap.dedent("""
    def construct_core_set(x_pool, core_set_size, eta=0.9, seed=0):
        import torch
        anchor = eta * x_pool.mean(0)
        d = ((x_pool - anchor) ** 2).sum(-1)
        return [int(i) for i in d.argsort(descending=True)[:core_set_size]]
    """)


def _package(tmp_path, method_body):
    from probes.package_loader import imported_method_package
    pkg = tmp_path / "run" / "method"
    pkg.mkdir(parents=True)
    (pkg / "method.py").write_text(method_body, encoding="utf-8")
    (pkg / "__init__.py").write_text(
        "from .method import *  # noqa: F401,F403\n", encoding="utf-8")
    return imported_method_package(tmp_path / "run")


def test_package_reexport_binds(tmp_path):
    # The production shape: the runner hands the package `method`, the
    # constructor lives in `method.method`. Exact-equality ownership
    # rejected this on every production run.
    with _package(tmp_path, _PKG_CORESET_BODY) as method:
        kit = al_stage1_kit(method)
        assert isinstance(kit, dict), kit
        assert kit["fn_name"] == "construct_core_set"
        verdicts = {v.probe_id: v for v in run_al_stage1_probes(method)}
    assert verdicts["AL-S1-1"].verdict == "pass"


def test_foreign_alias_refuses_with_named_owner(tmp_path):
    mod = _mod(tmp_path, """
        from numpy import argsort as coreset_argsort
        """)
    kit = al_stage1_kit(mod)
    assert isinstance(kit, str)
    assert "coreset_argsort" in kit and "owned by module" in kit
    assert "numpy" in kit


def test_name_matching_import_from_another_module_refuses(tmp_path):
    (tmp_path / "otherlib.py").write_text(textwrap.dedent("""
        def construct_core_set(x_pool, core_set_size, seed=0):
            return list(range(core_set_size))
        """), encoding="utf-8")
    mod = _mod(tmp_path, f"""
        import sys
        sys.path.insert(0, {str(tmp_path)!r})
        from otherlib import construct_core_set
        """)
    sys.path.remove(str(tmp_path))
    kit = al_stage1_kit(mod)
    assert isinstance(kit, str)
    assert "construct_core_set" in kit and "owned by module" in kit
    assert "otherlib" in kit


def test_numpy_annotated_pool_gets_numpy_not_torch(tmp_path):
    # The delivered bayesian constructor is `x_unlabeled: np.ndarray`; the
    # kit's torch pool produced a spurious crash-class fail on the first
    # real (read-only) bind. Annotation-aware typing must hand numpy in.
    mod = _mod(tmp_path, """
        from __future__ import annotations
        import numpy as np

        def construct_core_set(x_unlabeled: np.ndarray, core_set_size: int = 12,
                               eta: float = 0.9, seed=0):
            if not isinstance(x_unlabeled, np.ndarray):
                raise TypeError(f"expected ndarray, got {type(x_unlabeled)}")
            d = ((x_unlabeled - eta * x_unlabeled.mean(0)) ** 2).sum(-1)
            return [int(i) for i in np.argsort(-d)[:core_set_size]]
        """)
    verdicts = {v.probe_id: v for v in run_al_stage1_probes(mod)}
    assert verdicts["AL-S1-1"].verdict == "pass", verdicts["AL-S1-1"].message
    assert verdicts["AL-S1-2"].verdict == "pass", verdicts["AL-S1-2"].message
    assert verdicts["AL-S1-3"].verdict == "pass", verdicts["AL-S1-3"].message
