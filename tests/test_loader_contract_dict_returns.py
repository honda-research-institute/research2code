"""loader_contract_mismatch — the dry-run honors dict returns and the generic
template can load the bundle it ships beside.

The pdfgnn 2026-08-04 stage-2d degrade: the generic gap-family data.py
template returns a dict of named arrays BY DESIGN, while the stage 2.d
runtime dry-run accepted only tuple returns, so the pipeline's own scaffold
contradicted its own gate. The contradiction stayed latent while gap runs
had no data (load_data raised FileNotFoundError and the dry-run skipped);
R2C-052's demo bundling made load_data succeed for the first time and the
degrade surfaced. `load_data_returns` is already a name->shape map, so no
schema change: the dry-run now checks a dict return by KEY and a tuple
return by position.

Second arm, same run, found by the delivery quality review: the template's
csv loader float-cast every cell and json outranked csv, so on a real
bundle (header rows, string ids, PROVENANCE.json sitting beside the tables)
`load_data()` returned the provenance METADATA as the dataset, silently.
The template now skips bundle metadata, parses header tables into named
columns, and keys every table by file stem.
"""

from __future__ import annotations

import json
import subprocess
import sys
import textwrap
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

import pytest

from tests.test_provisional_templates import _scaffold_gap_run

pytestmark = pytest.mark.probe_runtime


# ---------------------------------------------------------------------------
# Dry-run: dict-return semantics
# ---------------------------------------------------------------------------

_DICT_CONTRACT = {
    "schema_version": "1.0.0",
    "paradigm_id": "graph_forecasting_test",
    "data_loader": {"load_data_returns": {
        "sales": "(N_rows, N_cols)",
        "product_hierarchy": "(N_rows, N_cols)",
    }},
    "architecture": {
        "model": {
            "class_name": "TinyNet",
            "forward": {
                "input": {"x": "(B, 4)"},
                "output_type": "tensor",
                "output_shape": "(B, 2)",
            },
        },
    },
    "pluggable_component": {"name": "forecast",
                            "input_shapes": {"seed": "int"},
                            "output_shape": "Forecast"},
    "training_loop": {"function_name": "train_from_scratch",
                      "input_shapes": {"seed": "int"}},
}

_TINY_MODEL = textwrap.dedent(
    """
    import torch
    import torch.nn as nn
    class TinyNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = nn.Linear(4, 2)
        def forward(self, x):
            return self.lin(x)
    """
)
_TINY_TRAINING = textwrap.dedent(
    """
    from .model import TinyNet
    def build_model(**kw):
        return TinyNet()
    def train_from_scratch(*a, seed=0, **kw):
        return TinyNet()
    """
)
_TINY_INIT = (
    "from .model import TinyNet\n"
    "from .training import build_model, train_from_scratch\n"
    "from .data import load_data\n"
    "def forecast(*a, **k):\n    return None\n"
)


def _write_runtime_pkg(tmp: Path, *, data_src: str, contract: dict) -> Path:
    run = tmp / "run"
    method = run / "method"
    pipe = run / ".pipeline"
    method.mkdir(parents=True)
    pipe.mkdir(parents=True)
    (method / "model.py").write_text(_TINY_MODEL)
    (method / "training.py").write_text(_TINY_TRAINING)
    (method / "data.py").write_text(data_src)
    (method / "__init__.py").write_text(_TINY_INIT)
    (pipe / "arch_contract.json").write_text(json.dumps(contract))
    (pipe / "method_spec.json").write_text("{}")
    return run


def _loader_errors(errors: list[str]) -> list[str]:
    return [e for e in errors if "load_data" in e]


def test_dict_loader_with_declared_names_passes(tmp_path):
    """The repaired pdfgnn shape: dict return whose keys carry the declared
    names satisfies the contract."""
    from scripts.validate_arch_contract_runtime import validate

    run = _write_runtime_pkg(
        tmp_path,
        data_src=(
            "import numpy as np\n"
            "def load_data(path=None, *, seed=0):\n"
            "    return {'sales': np.zeros((5, 3)),\n"
            "            'product_hierarchy': np.zeros((4, 2))}\n"
        ),
        contract=_DICT_CONTRACT,
    )
    assert _loader_errors(validate({}, run)) == []


def test_dict_loader_missing_declared_name_fails(tmp_path):
    """The as-delivered pdfgnn shape: the contract declares a name the dict
    does not carry. The finding names both sides so the fix loop can act."""
    from scripts.validate_arch_contract_runtime import validate

    contract = json.loads(json.dumps(_DICT_CONTRACT))
    contract["data_loader"]["load_data_returns"] = {"data": "(N_rows, N_cols)"}
    run = _write_runtime_pkg(
        tmp_path,
        data_src=(
            "import numpy as np\n"
            "def load_data(path=None, *, seed=0):\n"
            "    return {'sales': np.zeros((5, 3))}\n"
        ),
        contract=contract,
    )
    errs = _loader_errors(validate({}, run))
    assert errs, "missing declared name must fail"
    assert "'data'" in errs[0] and "sales" in errs[0]


def test_dict_loader_nested_tables_skip_shape_check(tmp_path):
    """A dict-of-columns table (the template's header-csv shape) has no
    .shape; the rank check skips it instead of erroring."""
    from scripts.validate_arch_contract_runtime import validate

    run = _write_runtime_pkg(
        tmp_path,
        data_src=(
            "import numpy as np\n"
            "def load_data(path=None, *, seed=0):\n"
            "    return {'sales': {'qty': np.zeros(5), 'store': np.array(['a']*5)},\n"
            "            'product_hierarchy': np.zeros((4, 2))}\n"
        ),
        contract=_DICT_CONTRACT,
    )
    assert _loader_errors(validate({}, run)) == []


def test_tuple_loader_semantics_unchanged(tmp_path):
    """Committed-family behavior is pinned: positional tuple returns keep
    passing, and a count mismatch keeps failing."""
    from scripts.validate_arch_contract_runtime import validate

    contract = json.loads(json.dumps(_DICT_CONTRACT))
    contract["data_loader"]["load_data_returns"] = {
        "x_train": "(N, 4)", "y_train": "(N,)"}
    good = _write_runtime_pkg(
        tmp_path / "good",
        data_src=(
            "import numpy as np\n"
            "def load_data(path=None, *, seed=0):\n"
            "    return np.zeros((6, 4)), np.zeros(6)\n"
        ),
        contract=contract,
    )
    assert _loader_errors(validate({}, good)) == []

    bad = _write_runtime_pkg(
        tmp_path / "bad",
        data_src=(
            "import numpy as np\n"
            "def load_data(path=None, *, seed=0):\n"
            "    return (np.zeros((6, 4)),)\n"
        ),
        contract=contract,
    )
    errs = _loader_errors(validate({}, bad))
    assert errs and "declares 2 returns" in errs[0]


def test_non_tuple_non_dict_return_fails(tmp_path):
    """A bare array or list still fails, with wording that names both
    accepted structures."""
    from scripts.validate_arch_contract_runtime import validate

    contract = json.loads(json.dumps(_DICT_CONTRACT))
    contract["data_loader"]["load_data_returns"] = {"data": "(N, 4)"}
    run = _write_runtime_pkg(
        tmp_path,
        data_src=(
            "import numpy as np\n"
            "def load_data(path=None, *, seed=0):\n"
            "    return np.zeros((6, 4))\n"
        ),
        contract=contract,
    )
    errs = _loader_errors(validate({}, run))
    assert errs and "tuple" in errs[0] and "dict" in errs[0]


# ---------------------------------------------------------------------------
# Template: the generic loader reads the bundle it ships beside
# ---------------------------------------------------------------------------


def _load_data_via_subprocess(run_dir: Path, expr: str) -> str:
    proc = subprocess.run(
        [sys.executable, "-c",
         "import json\n"
         "from method.data import load_data\n"
         "d = load_data()\n"
         f"print(json.dumps({expr}))"],
        cwd=str(run_dir), capture_output=True, text=True, timeout=30,
    )
    assert proc.returncode == 0, proc.stderr
    return json.loads(proc.stdout.strip())


def test_template_loads_real_bundle_tables(tmp_path):
    """The pdfgnn bundle shape against the real scaffolded template: three
    header csvs with string and blank cells plus PROVENANCE.json and
    README.md. The loader must return one named table per csv, skip the
    metadata, type numeric columns as floats, and keep string columns."""
    run_dir = _scaffold_gap_run(tmp_path)
    example = run_dir / "method" / "example_data"
    example.mkdir(parents=True, exist_ok=True)
    (example / "sales.csv").write_text(
        "product_id,store_id,qty,price\n"
        "P0001,S01,3,9.99\n"
        "P0002,S01,,4.50\n"
        "P0003,S02,7,1.25\n",
        encoding="utf-8",
    )
    (example / "product_hierarchy.csv").write_text(
        "product_id,category\nP0001,food\nP0002,drink\n", encoding="utf-8")
    (example / "store_cities.csv").write_text(
        "store_id,city\nS01,Kyoto\nS02,Osaka\n", encoding="utf-8")
    (example / "PROVENANCE.json").write_text(
        json.dumps({"source": {"cited_url": "https://example.test"},
                    "files": [], "honesty_note": "metadata, not data"}),
        encoding="utf-8",
    )
    (example / "README.md").write_text("bundle docs\n", encoding="utf-8")

    keys = _load_data_via_subprocess(run_dir, "sorted(d.keys())")
    assert keys == ["product_hierarchy", "sales", "store_cities"], keys

    sales_cols = _load_data_via_subprocess(run_dir, "sorted(d['sales'].keys())")
    assert sales_cols == ["price", "product_id", "qty", "store_id"], sales_cols

    qty = _load_data_via_subprocess(run_dir, "[str(v) for v in d['sales']['qty']]")
    assert qty[0] == "3.0" and qty[1] == "nan" and qty[2] == "7.0", qty

    ids = _load_data_via_subprocess(run_dir, "list(d['sales']['product_id'])")
    assert ids == ["P0001", "P0002", "P0003"], ids


def test_template_headerless_numeric_csv_keeps_legacy_shape(tmp_path):
    """A single headerless all-numeric csv keeps the legacy contract: one
    2-D float array under the key 'data'."""
    run_dir = _scaffold_gap_run(tmp_path)
    example = run_dir / "method" / "example_data"
    example.mkdir(parents=True, exist_ok=True)
    (example / "table.csv").write_text("1,2\n3,4\n", encoding="utf-8")

    keys = _load_data_via_subprocess(run_dir, "sorted(d.keys())")
    assert keys == ["data"], keys
    shape = _load_data_via_subprocess(run_dir, "list(d['data'].shape)")
    assert shape == [2, 2], shape
