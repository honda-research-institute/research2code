"""requirements.txt must cover the NOTEBOOK's third-party imports, not only the
method package's.

Root cause (2026-06-15): `requirements.txt` is built at Stage 2.d from the
method package alone, but the notebook is authored one stage later (3.a) and can
import a library the package never used — e.g. scikit-learn for a plot. Left
unreconciled, `pip install -r requirements.txt` then run hits an ImportError
(the F001 draft-demotion the BADGE validation run surfaced). Stage 3.a now runs
`reconcile_requirements_for_notebook` to close the gap at the source.
"""

from __future__ import annotations

import json
from pathlib import Path

from finalize_package_init import (
    _build_requirements_txt,
    _detect_notebook_third_party_imports,
    _detect_third_party_imports,
    reconcile_requirements_for_notebook,
)

PID = "active_learning"


def _notebook(cells: list[tuple[str, str]]) -> str:
    """Build a minimal nbformat-v4 notebook JSON from (cell_type, source) pairs."""
    return json.dumps({
        "cells": [
            {"cell_type": ct, "source": src, "metadata": {},
             **({"outputs": [], "execution_count": None} if ct == "code" else {})}
            for ct, src in cells
        ],
        "metadata": {}, "nbformat": 4, "nbformat_minor": 5,
    })


def _make_run(tmp_path: Path, *, notebook_json: str | None, initial_reqs: set[str]) -> Path:
    run = tmp_path / "run"
    method = run / "method"
    method.mkdir(parents=True)
    (method / "__init__.py").write_text("from .method import select_batch\n")
    (method / "method.py").write_text("import torch\nimport numpy as np\n\ndef select_batch():\n    return []\n")
    (method / "data.py").write_text("import torchvision\n\ndef load_data():\n    return None\n")
    (run / "requirements.txt").write_text(_build_requirements_txt(initial_reqs, PID))
    (run / "method_spec.json").write_text(json.dumps(
        {"comparison": {"classification": {"id": PID}}}))
    if notebook_json is not None:
        (run / "notebook.ipynb").write_text(notebook_json)
    return run


def test_notebook_scan_keeps_third_party_drops_local_stdlib_and_magics(tmp_path):
    nb = _notebook([
        ("code", "%pip install -r requirements.txt\n"),
        ("code", "from sklearn.decomposition import PCA\nimport torch\nfrom method import select_batch\nimport os\n"),
        ("code", "!echo hi\n%matplotlib inline\nimport numpy as np\n"),
        ("markdown", "# not code: import nothing"),
    ])
    # tmp_path, never a fixed /tmp name: two concurrent pytest runs collided
    # on the shared path during the Session A timing run (W3 verdicts §2.4.3).
    p = tmp_path / "_nb_scan_test.ipynb"
    p.write_text(nb)
    detected = _detect_notebook_third_party_imports(p)
    # sklearn/torch/numpy kept; os is stdlib; method is the local package; magics ignored.
    assert detected == {"sklearn", "torch", "numpy"}


def test_method_scan_excludes_local_package_self_import(tmp_path):
    """Generated method code may self-import absolutely (`from method.model
    import ...` — legal producer variance, DomIndOnto 2026-07-21) instead of
    relatively. The method-dir scan must exclude the local package name
    exactly like the notebook scan does, or `method` lands in
    requirements.txt and pip install fails on a nonexistent distribution."""
    method = tmp_path / "method"
    method.mkdir()
    (method / "method.py").write_text(
        "from method.model import Model\n"
        "import method.data\n"
        "import torch\n"
        "import os\n\n"
        "def run():\n    return Model()\n",
        encoding="utf-8",
    )
    # torch kept; method (both self-import forms) is the local package;
    # os is stdlib.
    assert _detect_third_party_imports(method) == {"torch"}


def test_method_scan_excludes_bare_sibling_module_imports(tmp_path):
    """A producer may import a sibling module by bare name (`import model`
    for method/model.py — DomIndOnto 2026-07-28) instead of through the
    package. The sibling must not become a pip requirement: `model`
    reached pip and halted stage 2.d with an internal contract violation
    on the nonexistent distribution."""
    method = tmp_path / "method"
    method.mkdir()
    (method / "model.py").write_text("import numpy\n", encoding="utf-8")
    (method / "training.py").write_text(
        "import model\n"
        "import numpy\n\n"
        "def train():\n    return model\n",
        encoding="utf-8",
    )
    # numpy kept; model is a sibling file of the package itself.
    assert _detect_third_party_imports(method) == {"numpy"}


def test_reconcile_adds_notebook_only_dependency(tmp_path):
    # 2.d would have written requirements from the method package only:
    # torch + numpy (base) + torchvision (data.py). No scikit-learn yet.
    run = _make_run(
        tmp_path,
        notebook_json=_notebook([
            ("code", "from sklearn.decomposition import PCA\nimport torch\nfrom method import select_batch\n"),
        ]),
        initial_reqs={"torch", "numpy", "torchvision"},
    )
    spec = run / "method_spec.json"

    changed, added = reconcile_requirements_for_notebook(run, spec)
    assert changed is True
    assert added == ["scikit-learn"]  # the only new line, mapped via PIP_NAME_OVERRIDES
    reqs = (run / "requirements.txt").read_text()
    assert "scikit-learn" in reqs
    assert "torchvision" in reqs  # method-derived extra preserved

    # Idempotent: a second pass is a no-op.
    changed2, added2 = reconcile_requirements_for_notebook(run, spec)
    assert changed2 is False and added2 == []


def test_reconcile_noop_when_notebook_uses_only_declared_deps(tmp_path):
    run = _make_run(
        tmp_path,
        notebook_json=_notebook([
            ("code", "import torch\nimport matplotlib.pyplot as plt\nfrom method import select_batch\n"),
        ]),
        initial_reqs={"torch", "numpy", "torchvision"},
    )
    changed, added = reconcile_requirements_for_notebook(run, run / "method_spec.json")
    assert changed is False and added == []


def test_reconcile_best_effort_when_notebook_missing(tmp_path):
    run = _make_run(tmp_path, notebook_json=None, initial_reqs={"torch", "numpy"})
    changed, added = reconcile_requirements_for_notebook(run, run / "method_spec.json")
    assert changed is False and added == []


# ---------------------------------------------------------------------------
# R2C-050 — unresolvable-requirement parsing and the producer trace
# ---------------------------------------------------------------------------


def test_unresolvable_names_parse_both_pip_phrases_once():
    from finalize_package_init import unresolvable_requirement_names

    out = unresolvable_requirement_names(
        "ERROR: Could not find a version that satisfies the requirement dgl "
        "(from versions: none)\n"
        "ERROR: No matching distribution found for dgl\n"
    )
    assert out == ["dgl"]


def test_unresolvable_names_strip_specifiers_and_extras():
    from finalize_package_init import unresolvable_requirement_names

    assert unresolvable_requirement_names(
        "No matching distribution found for torch>=99.0"
    ) == ["torch"]
    assert unresolvable_requirement_names(
        "Could not find a version that satisfies the requirement dgl[cuda]==2.0"
    ) == ["dgl"]


def test_unresolvable_names_empty_on_transport_noise():
    from finalize_package_init import unresolvable_requirement_names

    assert unresolvable_requirement_names(
        "WARNING: connection broken by ProxyError\n"
    ) == []
    assert unresolvable_requirement_names("") == []


_TRACE_MANIFEST = [
    {"path": "method/model.py", "produced_by": "architecture_coder"},
    {"path": "method/training.py", "produced_by": "architecture_coder"},
    {"path": "method/method.py", "produced_by": "method_coder"},
    {"path": "method/data.py", "produced_by": "package_scaffolder"},
]


def test_trace_finds_the_importing_file_and_its_producer(tmp_path):
    """The pdfgnn dgl shape: the architecture coder's model.py imports it."""
    from finalize_package_init import trace_requirement_to_producers

    md = tmp_path / "method"
    md.mkdir()
    (md / "model.py").write_text("import dgl\nfrom dgl.nn import GraphConv\n")
    (md / "method.py").write_text("import torch\n")

    trace = trace_requirement_to_producers("dgl", md, _TRACE_MANIFEST)

    assert trace == {
        "requirement": "dgl",
        "files": ["method/model.py"],
        "producers": ["architecture_coder"],
    }


def test_trace_maps_pip_name_back_to_import_name(tmp_path):
    """opencv-python is imported as cv2; the reverse mapping must hold."""
    from finalize_package_init import trace_requirement_to_producers

    md = tmp_path / "method"
    md.mkdir()
    (md / "method.py").write_text("import cv2\n")

    trace = trace_requirement_to_producers("opencv-python", md, _TRACE_MANIFEST)

    assert trace is not None
    assert trace["files"] == ["method/method.py"]
    assert trace["producers"] == ["method_coder"]


def test_trace_handles_dash_underscore_convention(tmp_path):
    from finalize_package_init import trace_requirement_to_producers

    md = tmp_path / "method"
    md.mkdir()
    (md / "training.py").write_text("import torch_geometric\n")

    trace = trace_requirement_to_producers("torch-geometric", md, _TRACE_MANIFEST)

    assert trace is not None
    assert trace["files"] == ["method/training.py"]


def test_trace_returns_none_for_finalizer_artifacts(tmp_path):
    """The DomIndOnto 2026-07-21 leak: `method` in requirements.txt traces
    to no producer import, so the internal-bug classification is kept."""
    from finalize_package_init import trace_requirement_to_producers

    md = tmp_path / "method"
    md.mkdir()
    (md / "model.py").write_text("import torch\n")

    assert trace_requirement_to_producers("method", md, _TRACE_MANIFEST) is None
    assert trace_requirement_to_producers("dgl", md, _TRACE_MANIFEST) is None


def test_trace_without_manifest_reports_files_but_no_producers(tmp_path):
    """No build-plan manifest → the file is named but ownership is empty,
    and the caller's routability filter keeps the internal halt."""
    from finalize_package_init import trace_requirement_to_producers

    md = tmp_path / "method"
    md.mkdir()
    (md / "model.py").write_text("import dgl\n")

    trace = trace_requirement_to_producers("dgl", md, None)

    assert trace is not None
    assert trace["files"] == ["method/model.py"]
    assert trace["producers"] == []
