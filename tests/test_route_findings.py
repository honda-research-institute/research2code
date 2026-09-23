"""B-07 step 2: route_findings' self-test body, migrated to pytest.

The 48 asserts are carried verbatim, split into per-surface test
functions. The two blocks that were silently gated on gitignored
r2c_runs/ artifacts (`if <path>.exists():` — invisible permanent skips in
CI) are now explicit manual_only tests that pytest.skip when the live
fleet is absent, per the A1 fleet-test convention.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from route_findings import (
    cell_index_to_section,
    critical_findings,
    filter_by_severity,
    group_by_target_agent,
    important_findings,
    nice_to_have_findings,
    severity_rank,
    should_halt_stage,
    smoke_cell_to_producer,
    smoke_traceback_deepest_owned_frame,
    smoke_traceback_to_producer,
)

REPO = Path(__file__).resolve().parent.parent

FINDINGS = [
    {"id": "F1", "severity": "critical"},
    {"id": "F2", "severity": "important"},
    {"id": "F3", "severity": "nice-to-have"},
    {"id": "F4", "severity": "unknown"},
]


def test_severity_rank():
    assert severity_rank("critical") == 2
    assert severity_rank("important") == 1
    assert severity_rank("nice-to-have") == 0
    assert severity_rank("garbage") == -1


def test_filter_by_severity_and_partitions():
    assert [f["id"] for f in filter_by_severity(FINDINGS, min_severity="critical")] == ["F1"]
    assert [f["id"] for f in filter_by_severity(FINDINGS, min_severity="important")] == ["F1", "F2"]
    assert [f["id"] for f in filter_by_severity(FINDINGS, min_severity="nice-to-have")] == ["F1", "F2", "F3"]
    assert [f["id"] for f in critical_findings(FINDINGS)] == ["F1"]
    assert [f["id"] for f in important_findings(FINDINGS)] == ["F2"]
    assert [f["id"] for f in nice_to_have_findings(FINDINGS)] == ["F3"]


def test_group_by_target_agent():
    grouped = group_by_target_agent([
        {"id": "F1", "target_agent": "method-coder"},
        {"id": "F2", "target_agent": "method-coder"},
        {"id": "F3", "target_agent": "architecture-coder"},
        {"id": "F4"},  # missing target_agent
    ])
    assert set(grouped.keys()) == {"method-coder", "architecture-coder", "unknown"}
    assert [f["id"] for f in grouped["method-coder"]] == ["F1", "F2"]
    assert [f["id"] for f in grouped["unknown"]] == ["F4"]


def test_smoke_cell_to_producer():
    # The routing table from the archived v2 orchestrator spec (internal, not shipped) Stage 3.c
    assert smoke_cell_to_producer(0) == "notebook-generator"
    assert smoke_cell_to_producer(1) == "notebook-generator"
    assert smoke_cell_to_producer(2) == "notebook-generator"
    assert smoke_cell_to_producer(3) == "architecture-coder"
    assert smoke_cell_to_producer(4) == "method-coder"
    assert smoke_cell_to_producer(5) == "method-coder"
    assert smoke_cell_to_producer(6) == "notebook-generator"
    assert smoke_cell_to_producer(99) == "notebook-generator"  # default


TB_METHOD = (
    'Cell In[6], line 12\n'
    'File "/Users/x/r2c_runs/foo/method/method.py", line 88, in select_batch\n'
    '    raise ValueError("x")\n'
    'ValueError: x'
)
TB_WITH_ANSI = (
    '\x1b[31mTraceback\x1b[0m\n'
    '\x1b[36mFile\x1b[0m \x1b[32m"/runs/q/method/method.py"\x1b[0m, line 1, in foo\n'
)
TB_DEEPEST = (
    'Cell In[8], line 12\n'
    'File "/runs/q/method/method.py", line 50, in select_batch\n'
    'File "/runs/q/method/model.py", line 22, in forward\n'
    'RuntimeError: x'
)
TB_NOTEBOOK_INLINE = (
    'Cell In[6], line 4\n'
    'File numpy/random/mtrand.pyx:860, in numpy.random.mtrand.RandomState.choice()\n'
    "TypeError: choice() got an unexpected keyword argument 'rng'"
)


def test_smoke_traceback_to_producer():
    # The deepest method/*.py frame wins; absent any, "Cell In[" marks
    # notebook-written code.
    assert smoke_traceback_to_producer(TB_METHOD) == "method-coder"

    tb_model = (
        'Cell In[4], line 3\n'
        'File "/runs/bar/method/model.py", line 22, in forward\n'
        '    return self.fc(x)\n'
        'RuntimeError: shape mismatch'
    )
    assert smoke_traceback_to_producer(tb_model) == "architecture-coder"

    tb_training = (
        'Cell In[4], line 3\n'
        'File "/runs/bar/method/training.py", line 40, in train_from_scratch\n'
        '    return model\n'
        'TypeError: bad arg'
    )
    assert smoke_traceback_to_producer(tb_training) == "architecture-coder"

    # data.py / __init__.py: the call site is in the notebook, not in the
    # producer-owned file — route to notebook-generator.
    tb_data = (
        'Cell In[2], line 1\n'
        'File "/runs/baz/method/data.py", line 5, in load_data\n'
        '    raise FileNotFoundError("missing")\n'
    )
    assert smoke_traceback_to_producer(tb_data) == "notebook-generator"

    # No method/ frame, but the failing context is a Jupyter cell — bug is
    # in the notebook code itself. The GBALD `np.random.choice` case.
    assert smoke_traceback_to_producer(TB_NOTEBOOK_INLINE) == "notebook-generator"

    # ANSI color escapes (nbclient rich tracebacks) must be stripped before
    # matching — otherwise the regex misses `method/` paths.
    assert smoke_traceback_to_producer(TB_WITH_ANSI) == "method-coder"

    # Deepest frame wins: outer method.py + inner model.py → architecture-coder.
    assert smoke_traceback_to_producer(TB_DEEPEST) == "architecture-coder"

    # Truly ambiguous: no Cell tag, no method/ frame → None, fall back to
    # section-based routing.
    assert smoke_traceback_to_producer("RuntimeError: opaque") is None


def test_smoke_traceback_deepest_owned_frame():
    method_owned = ["method/method.py"]
    arch_owned = ["method/model.py", "method/training.py"]

    anchor = smoke_traceback_deepest_owned_frame(TB_METHOD, method_owned)
    assert anchor == "method/method.py:88 in select_batch", anchor
    assert smoke_traceback_deepest_owned_frame(TB_METHOD, arch_owned) is None

    # Mixed traceback: each allowlist gets its own deepest matching frame.
    assert smoke_traceback_deepest_owned_frame(TB_DEEPEST, method_owned) == "method/method.py:50 in select_batch"
    assert smoke_traceback_deepest_owned_frame(TB_DEEPEST, arch_owned) == "method/model.py:22 in forward"

    assert smoke_traceback_deepest_owned_frame(TB_WITH_ANSI, method_owned) == "method/method.py:1 in foo"
    assert smoke_traceback_deepest_owned_frame(TB_NOTEBOOK_INLINE, method_owned) is None

    # bev-distill regression: the actual failing traceback shape from the
    # 2026-05-14 run. method-coder's allowlist matches method/method.py:175.
    # NOTE (carried from the in-body TODO, B-07 rev 2): the bev traceback
    # uses `~/` shorthand and `File <path>:N, in <fn>` (colon-separated, no
    # quotes — IPython's rich formatting), not the standard quoted
    # `File "..."` form; the frame regex accepts both forms.
    tb_bev = (
        'IndexError                                Traceback (most recent call last)\n'
        'Cell In[9], line 10\n'
        '     10 quality_scores = compute_quality_scores(\n'
        'File ~/Documents/Work/research2code-mvp/r2c_runs/bev-distill/method/method.py:255, in compute_quality_scores\n'
        '    255     ious_matrix = compute_3d_box_iou(teacher_boxes, gt_boxes)\n'
        'File ~/Documents/Work/research2code-mvp/r2c_runs/bev-distill/method/method.py:175, in compute_3d_box_iou\n'
        '    175     yaw1 = torch.atan2(boxes1[:, 8], boxes1[:, 9])\n'
        'IndexError: index 8 is out of bounds for dimension of size 8'
    )
    anchor_bev = smoke_traceback_deepest_owned_frame(tb_bev, ["method/method.py"])
    assert anchor_bev == "method/method.py:175 in compute_3d_box_iou", anchor_bev


def test_should_halt_stage():
    assert should_halt_stage([{"severity": "critical"}]) is True
    assert should_halt_stage([{"severity": "important"}]) is True
    assert should_halt_stage([{"severity": "nice-to-have"}]) is False
    assert should_halt_stage([]) is False


def test_cell_index_to_section():
    nb = {
        "cells": [
            {"cell_type": "markdown", "source": ["## 0. Install\n"]},
            {"cell_type": "code", "source": ["%pip install foo"]},  # cell 1, §0
            {"cell_type": "markdown", "source": ["## 1. Imports\n"]},
            {"cell_type": "code", "source": ["import foo"]},        # cell 3, §1
            {"cell_type": "markdown", "source": ["## 3. Model build\n"]},
            {"cell_type": "code", "source": ["model = build_model()"]},  # cell 5, §3
        ]
    }
    assert cell_index_to_section(1, nb) == 0
    assert cell_index_to_section(3, nb) == 1
    assert cell_index_to_section(5, nb) == 3


# -- live-fleet fixtures: were `if path.exists():` blocks inside the
# -- self-test, i.e. permanently-invisible skips in CI. Now explicit.


@pytest.mark.manual_only
def test_bev_distill_halt_fixture_buckets_as_unknown():
    halt = REPO / "r2c_runs" / "bev-distill" / ".pipeline" / "stage_2x_params.halt"
    if not halt.exists():
        pytest.skip("live fleet absent")
    findings = json.loads(halt.read_text()).get("findings", [])
    assert len(findings) == 3, f"expected 3 historical findings, got {len(findings)}"
    assert should_halt_stage(findings) is True
    grouped = group_by_target_agent(findings)
    # Halt artifacts don't carry target_agent → all bucket as 'unknown'
    assert "unknown" in grouped
    assert len(grouped["unknown"]) == 3


@pytest.mark.manual_only
def test_bev_distill_stage_review_fixture_api():
    # Contents change across rolls — exercise the API on real data, don't
    # pin a roll-dependent verdict.
    review_path = (REPO / "r2c_runs" / "bev-distill" / ".pipeline"
                   / "stage_review_stage_2x_params.json")
    if not review_path.exists():
        pytest.skip("live fleet absent")
    findings = json.loads(review_path.read_text()).get("findings", [])
    assert isinstance(should_halt_stage(findings), bool)
