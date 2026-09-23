"""Shared fixture run for the note-4/5 correspondence tests.

One fixture package + paper map + notebook draft, modeled on the anchor
and call shapes the design review verified on the real ms3d delivery:

- a dedicated small implementation (`kde_mode_1d`, leading anchor);
- a fusion function carrying its own id plus a shared one, docstring and
  leading anchors (`kbf_fuse`: primary for concept-kbf, secondary site
  for eq-kde);
- two small refinement functions (`refine_a` / `refine_b`) so a single
  overview cell can resolve to MORE implementing functions than the
  per-cell cap allows;
- an un-anchored helper (`plot_helper`) that must never associate;
- a >60-line orchestrator (`generate_pipeline`) that is primary for the
  algorithm id and for a call-site-only id (concept-tracking), while
  merely touching eq-kde / concept-kbf at call sites — the cap-to-primary
  case AND the size-cap case in one function.

The draft mirrors the real drafts' shapes: a setup cell opening with
IPython magics, per-component demo cells under LaTeX markdown, an
overview cell, and an end-to-end orchestrator cell.

Used by tests/test_notebook_implementation_notes.py and
tests/test_method_md_correspondence.py.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

# 56 padding statements push generate_pipeline well past the 60-line
# inline-source cap while keeping the fixture readable.
_ORCHESTRATOR_PADDING = "\n".join(
    f"    step_{i} = {i}  # padding so the size cap engages" for i in range(56))

FIXTURE_METHOD_PY = f'''\
"""Fixture package — anchor shapes modeled on the real ms3d delivery."""


def kde_mode_1d(values, weights):
    """Weighted KDE mode (the dedicated implementation)."""
    # paper-element: eq-kde
    return values[0]


def kbf_fuse(sources):
    """KBF fusion.

    # paper-element: eq-kde
    # paper-element: concept-kbf
    """
    # paper-element: eq-kde
    # paper-element: concept-kbf
    return [kde_mode_1d(s, [1.0]) for s in sources]


def refine_a(tracks):
    """Refinement A."""
    # paper-element: concept-refine_a
    return tracks


def refine_b(tracks):
    """Refinement B."""
    # paper-element: concept-refine_b
    return tracks


def plot_helper(x):
    """Un-anchored helper — must never associate."""
    return x


def generate_pipeline(data):
    """Orchestrator carrying call-site anchors for the dedicated
    implementations plus its own algorithm id.

    # paper-element: alg-pipeline
    """
    # paper-element: alg-pipeline
    fused = kbf_fuse(data)
    # paper-element: eq-kde
    # paper-element: concept-kbf
    # paper-element: concept-tracking
{_ORCHESTRATOR_PADDING}
    return fused
'''

FIXTURE_PAPER_MAP = {
    "title": "Correspondence Fixture Paper",
    "elements": [
        {"id": "eq-kde", "type": "equation", "name": "Weighted KDE mode",
         "section": "Section 4.2", "code_role": "implement",
         "source_text": "The KDE mode equation."},
        {"id": "concept-kbf", "type": "concept", "name": "KBF fusion",
         "section": "Section 4.3", "code_role": "implement",
         "source_text": "Fuse boxes via KDE."},
        {"id": "alg-pipeline", "type": "algorithm",
         "name": "Full pipeline", "section": "Algorithm 1",
         "code_role": "implement",
         "description": "Run every step end to end."},
        {"id": "concept-tracking", "type": "concept", "name": "Tracking",
         "section": "Section 4.4", "code_role": "implement",
         "source_text": "Track across frames."},
        {"id": "concept-refine_a", "type": "concept", "name": "Refinement A",
         "section": "Section 4.5", "code_role": "implement",
         "source_text": "Refine A."},
        {"id": "concept-refine_b", "type": "concept", "name": "Refinement B",
         "section": "Section 4.6", "code_role": "implement",
         "source_text": "Refine B."},
        {"id": "concept-unanchored", "type": "concept",
         "name": "Explained-only concept", "section": "Section 2",
         "code_role": "explain",
         "source_text": "Background the code never claims."},
    ],
}

FIXTURE_DRAFT = '''\
# %% [markdown]
# # Correspondence fixture notebook

# %% [markdown]
# ## 1. Setup

# %%
%matplotlib inline
from method import (
    generate_pipeline,
    kbf_fuse,
    kde_mode_1d,
    plot_helper,
    refine_a,
    refine_b,
)

# %% [markdown]
# ### 4.2 The KDE building block
#
# $$\\hat{f}(x) = \\sum_i w_i K(x - x_i)$$

# %%
x = kde_mode_1d([1.0, 2.0], [1.0, 1.0])
print(x)

# %% [markdown]
# ### 4.3 KBF fusion

# %%
fused = kbf_fuse([[1.0], [2.0]])
rechecked = kde_mode_1d([3.0], [1.0])

# %%
plot_helper(fused)

# %% [markdown]
# ### 4.9 Overview of every refinement

# %%
a = refine_a([1])
b = refine_b([2])
f2 = kbf_fuse([[1.0]])
k2 = kde_mode_1d([1.0], [1.0])

# %% [markdown]
# ## 5. End to end

# %%
out = generate_pipeline([[1.0]])
'''

_ANCHOR_COMMENT_RE = re.compile(r"^\s*# paper-element:.*\n", re.MULTILINE)


def make_correspondence_run(tmp_path: Path, *, anchors: bool = True,
                            draft: bool = True) -> Path:
    """A run dir carrying the fixture package, paper map, params, and
    (optionally) the notebook draft. `anchors=False` strips every anchor
    comment from the delivered code — the anchor-free no-regression case."""
    run = tmp_path / "run"
    (run / ".pipeline").mkdir(parents=True)
    (run / ".pipeline" / "paper_map.json").write_text(
        json.dumps(FIXTURE_PAPER_MAP), encoding="utf-8")
    (run / ".pipeline" / "params.json").write_text(
        json.dumps({"params": {}}), encoding="utf-8")
    method_py = FIXTURE_METHOD_PY
    if not anchors:
        method_py = _ANCHOR_COMMENT_RE.sub("", method_py)
        assert "paper-element" not in method_py
    (run / "method").mkdir()
    (run / "method" / "method.py").write_text(method_py, encoding="utf-8")
    if draft:
        (run / ".pipeline" / "notebook_draft.py").write_text(
            FIXTURE_DRAFT, encoding="utf-8")
    return run


def def_line(needle: str, text: str = FIXTURE_METHOD_PY) -> int:
    """1-based line of the fixture line starting with `needle` (e.g.
    'def kde_mode_1d') — keeps line-number assertions robust against
    fixture edits."""
    for i, line in enumerate(text.split("\n"), start=1):
        if line.startswith(needle):
            return i
    raise AssertionError(f"{needle!r} not found in fixture")
