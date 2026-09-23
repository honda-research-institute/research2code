"""B-06 golden-membership pin for the stage-2d/3c upstream digests.

Lands BEFORE the digest-twin merge: the two stages' upstream sets are
deliberately different (2d excludes its own __init__.py output and reads
the contract pair; 3c includes 2d's outputs plus requirements.txt and the
notebook surfaces), and unifying them would make 2d re-run on every
resume after the requirements reconcile rewrites requirements.txt. This
test asserts the EXACT relative-path key sets for both stages on one
seeded run dir, so any merge that drifts either set fails here byte-for-
byte before it fails in a live resume.
"""

from __future__ import annotations

import re
from pathlib import Path

from run_pipeline import (
    _compute_stage_2d_upstream_digest,
    _compute_stage_3c_upstream_digest,
)
from tests.helpers.state import make_state


def _seed_both_stage_upstreams(run_dir: Path) -> None:
    (run_dir / "method" / "nested").mkdir(parents=True, exist_ok=True)
    for name in ("model.py", "training.py", "method.py", "data.py",
                 "__init__.py"):
        (run_dir / "method" / name).write_text(f"# {name}\n")
    # rglob must reach nested local dependencies for BOTH stages
    (run_dir / "method" / "nested" / "helper.py").write_text("# helper\n")
    (run_dir / ".pipeline").mkdir(parents=True, exist_ok=True)
    for name in ("arch_contract.json", "method_spec.json", "params.json"):
        (run_dir / ".pipeline" / name).write_text("{}\n")
    (run_dir / ".pipeline" / "notebook_draft.py").write_text("# nb\n")
    (run_dir / "notebook.ipynb").write_text("{}\n")
    (run_dir / "requirements.txt").write_text("# reqs\n")


def test_stage_2d_and_3c_digest_key_sets_are_the_pinned_goldens(run_dir):
    state = make_state(run_dir)
    _seed_both_stage_upstreams(run_dir)

    d2d = _compute_stage_2d_upstream_digest(state.paths)
    d3c = _compute_stage_3c_upstream_digest(state.paths)

    # 2d: every method source EXCEPT its own __init__.py output, plus the
    # contract pair. NEVER requirements.txt (reconcile rewrites it on every
    # resume) and never the notebook surfaces.
    assert set(d2d) == {
        "method/data.py",
        "method/method.py",
        "method/model.py",
        "method/nested/helper.py",
        "method/training.py",
        ".pipeline/arch_contract.json",
        ".pipeline/method_spec.json",
    }

    # 3c: every method source INCLUDING 2d's __init__.py output, plus the
    # notebook surfaces, params, and requirements.txt. Never the contract
    # pair (the smoke gate does not read them).
    assert set(d3c) == {
        "method/__init__.py",
        "method/data.py",
        "method/method.py",
        "method/model.py",
        "method/nested/helper.py",
        "method/training.py",
        ".pipeline/notebook_draft.py",
        "notebook.ipynb",
        ".pipeline/params.json",
        "requirements.txt",
    }

    for digest in (d2d, d3c):
        for rel, value in digest.items():
            assert re.fullmatch(r"sha256:[0-9a-f]{64}", value), (rel, value)

    # Shared files hash identically across the two computations — the merge
    # must not change how a file's content is keyed or hashed.
    for rel in set(d2d) & set(d3c):
        assert d2d[rel] == d3c[rel]
