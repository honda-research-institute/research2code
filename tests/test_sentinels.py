"""Phase 1 — Stage complete-sentinel system tests.

Covers the partial-completion-on-resume failure mode (bev-distill 2026-05-19):
stage 2.d's producer wrote `__init__.py` + `requirements.txt`, the arch_contract
validator halted, the user cleared the halt and resumed — but `_skip_if_done`
saw the producer outputs and skipped the validators entirely, letting an
upstream BatchNorm bug ride through to stage 3.c.

The sentinel system (`<stage_id>.complete`) requires whole-stage completion,
not just output-file presence. This test file pins that behavior.
"""

from __future__ import annotations

import json
from pathlib import Path

from tests.helpers.assertions import (
    assert_sentinel_absent,
    assert_sentinel_present,
    assert_stage_skipped,
)
from tests.helpers.state import make_paths


# ---------------------------------------------------------------------------
# Helper unit tests (these test infrastructure, not failure modes)
# ---------------------------------------------------------------------------


def test_sentinel_path_anchored_at_pipeline_dir(run_dir):
    """`_stage_complete_sentinel_path` returns `.pipeline/<stage_id>.complete`."""
    from run_pipeline import _stage_complete_sentinel_path
    paths = make_paths(run_dir)
    p = _stage_complete_sentinel_path(paths, "stage_2d")
    assert p == run_dir / ".pipeline" / "stage_2d.complete"


def test_sentinel_write_creates_file(run_dir):
    """`_write_stage_complete_sentinel` creates an empty marker file."""
    from run_pipeline import _write_stage_complete_sentinel
    paths = make_paths(run_dir)
    _write_stage_complete_sentinel(paths, "stage_2d")
    assert_sentinel_present(run_dir, "stage_2d")


def test_sentinel_write_is_idempotent(run_dir):
    """Calling write twice updates mtime but doesn't fail."""
    from run_pipeline import _write_stage_complete_sentinel
    paths = make_paths(run_dir)
    _write_stage_complete_sentinel(paths, "stage_2d")
    _write_stage_complete_sentinel(paths, "stage_2d")
    assert_sentinel_present(run_dir, "stage_2d")


def test_sentinel_clear_removes_file(run_dir):
    """`_clear_stage_complete_sentinel` deletes the marker."""
    from run_pipeline import (
        _clear_stage_complete_sentinel,
        _write_stage_complete_sentinel,
    )
    paths = make_paths(run_dir)
    _write_stage_complete_sentinel(paths, "stage_2d")
    assert_sentinel_present(run_dir, "stage_2d")
    _clear_stage_complete_sentinel(paths, "stage_2d")
    assert_sentinel_absent(run_dir, "stage_2d")


def test_sentinel_clear_is_idempotent_on_missing(run_dir):
    """Clearing when the sentinel doesn't exist is a no-op (no error)."""
    from run_pipeline import _clear_stage_complete_sentinel
    paths = make_paths(run_dir)
    _clear_stage_complete_sentinel(paths, "stage_2d")  # should not raise
    assert_sentinel_absent(run_dir, "stage_2d")


# ---------------------------------------------------------------------------
# _skip_if_done behavior matrix — the actual bev-distill regression test
# ---------------------------------------------------------------------------


def test_skip_if_done_outputs_missing_runs_stage(run_dir):
    """Outputs missing + sentinel missing → returns None (run the stage)."""
    from run_pipeline import _skip_if_done
    paths = make_paths(run_dir)
    out1 = run_dir / "out1.json"
    out2 = run_dir / "out2.json"
    # outputs don't exist
    result = _skip_if_done(paths, "stage_test", [out1, out2])
    assert result is None


def test_skip_if_done_outputs_present_but_no_sentinel_runs_stage(run_dir):
    """**The bev-distill 2026-05-19 regression**: outputs alone don't trigger
    skip. Without the complete-sentinel, the validators must re-run."""
    from run_pipeline import _skip_if_done
    paths = make_paths(run_dir)
    out1 = run_dir / "out1.json"
    out2 = run_dir / "out2.json"
    out1.write_text("{}")
    out2.write_text("{}")
    # sentinel is missing — should NOT skip
    result = _skip_if_done(paths, "stage_test", [out1, out2])
    assert result is None, (
        "outputs present without sentinel must trigger run, not skip. "
        "This is the bev-distill 2026-05-19 cascade: partial-completion "
        "from a halted prior run left outputs on disk that looked 'done', "
        "so the validators were skipped on resume."
    )


def test_skip_if_done_outputs_and_sentinel_present_skips(run_dir):
    """Outputs + sentinel present + no halt → skip."""
    from run_pipeline import _skip_if_done, _write_stage_complete_sentinel
    paths = make_paths(run_dir)
    out1 = run_dir / "out1.json"
    out2 = run_dir / "out2.json"
    out1.write_text("{}")
    out2.write_text("{}")
    _write_stage_complete_sentinel(paths, "stage_test")
    result = _skip_if_done(paths, "stage_test", [out1, out2])
    assert_stage_skipped(result, "stage_test")


def test_skip_if_done_halt_present_forces_rerun_and_clears_halt(run_dir):
    """Outputs + sentinel + halt → run (with halt artifact cleared)."""
    from run_pipeline import _skip_if_done, _write_stage_complete_sentinel
    paths = make_paths(run_dir)
    out1 = run_dir / "out1.json"
    out1.write_text("{}")
    _write_stage_complete_sentinel(paths, "stage_test")
    halt_path = run_dir / ".pipeline" / "stage_test.halt"
    halt_path.write_text(json.dumps({"status": "halted"}))

    result = _skip_if_done(paths, "stage_test", [out1])
    assert result is None
    assert not halt_path.exists(), (
        "_skip_if_done should clear the stale halt artifact when it forces "
        "a re-run, so the next halt() call can write a fresh one"
    )


def test_skip_if_done_returns_paths_written(run_dir):
    """The skipped StageResult carries the expected outputs as paths_written."""
    from run_pipeline import _skip_if_done, _write_stage_complete_sentinel
    paths = make_paths(run_dir)
    out1 = run_dir / "out1.json"
    out2 = run_dir / "out2.json"
    out1.write_text("{}")
    out2.write_text("{}")
    _write_stage_complete_sentinel(paths, "stage_test")
    result = _skip_if_done(paths, "stage_test", [out1, out2])
    assert result is not None
    assert list(result.paths_written) == [out1, out2]


# ---------------------------------------------------------------------------
# Driver-managed marker: sentinel files are exempt from out-of-scope checks
# ---------------------------------------------------------------------------


def test_sentinel_is_driver_managed():
    """Sentinels are written by the driver, not by agents — must be exempt
    from `_detect_out_of_scope_writes`."""
    from run_pipeline import _is_driver_managed
    assert _is_driver_managed(".pipeline/stage_2d.complete") is True
    assert _is_driver_managed(".pipeline/stage_3c.complete") is True
    # Sanity: agent-owned files are NOT driver-managed
    assert _is_driver_managed("method/model.py") is False
    assert _is_driver_managed("method/method.py") is False
