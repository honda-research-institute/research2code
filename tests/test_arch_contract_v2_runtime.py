"""R2C-085 -- typed Stage-2d fixture synthesis and runtime agreement."""

from __future__ import annotations

import copy
import json
from pathlib import Path

from scripts.arch_contract_semantics import SemanticIssue, format_semantic_issue
from scripts.arch_contract_semantics import parse_semantic_issue
from scripts.validate_arch_contract_runtime import validate
import pytest

pytestmark = pytest.mark.probe_runtime


def _lit(value: int) -> dict:
    return {"kind": "literal", "value": value}


def _use(identity: str) -> dict:
    return {"dimension": identity}


def _array(
    kind: str,
    dtype: str,
    *dimensions: str,
    constraint: dict | None = None,
) -> dict:
    descriptor = {
        "kind": kind,
        "dtype": dtype,
        "dimensions": [_use(identity) for identity in dimensions],
    }
    if constraint is not None:
        descriptor["constraint"] = constraint
    return descriptor


def _scalar(dtype: str, value: object) -> dict:
    return {"kind": "scalar", "dtype": dtype, "source": {"literal": value}}


def _opaque(label: str) -> dict:
    return {
        "kind": "opaque",
        "type_description": label,
        "reason": "No bounded generic synthesis exists for this value.",
    }


def _contract() -> dict:
    features = _array("tensor", "float32", "batch_count", "feature_width")
    edge_index = _array(
        "tensor",
        "int64",
        "endpoint_count",
        "edge_count",
        constraint={"kind": "index", "indexed_dimension": "node_count"},
    )
    mask = _array(
        "ndarray",
        "bool",
        "batch_count",
        constraint={"kind": "mask"},
    )
    logits = _array("tensor", "float64", "batch_count", "class_count")
    labels = _array(
        "tensor",
        "int64",
        "batch_count",
        constraint={
            "kind": "class_id",
            "class_count_dimension": "class_count",
        },
    )
    forecast = _array("ndarray", "float64", "batch_count", "horizon")
    return {
        "schema_version": "2.0.0",
        "paradigm_id": "typed_mixed_test",
        "dimensions": {
            "batch_count": {"expression": _lit(2)},
            "feature_width": {"expression": _lit(3)},
            "node_count": {"expression": _lit(5)},
            "endpoint_count": {"expression": _lit(2)},
            "edge_count": {"expression": _lit(4)},
            "class_count": {"expression": _lit(3)},
            "horizon": {"expression": _lit(2)},
        },
        "data_loader": {
            "load_data_returns": {
                "features": features,
                "edge_index": edge_index,
                "mask": mask,
            }
        },
        "architecture": {
            "model": {
                "class_name": "Net",
                "constructor_args": {
                    "width": {"dimension": _use("feature_width")},
                    "mode": {"literal": None},
                },
                "forward": {
                    "input": {
                        "features": features,
                        "edge_index": edge_index,
                        "mask": mask,
                        "temperature": _scalar("float32", 0.5),
                    },
                    "output": logits,
                },
                "additional_methods": {
                    "inspect": {
                        "input": {"values": forecast},
                        "output": _scalar("int32", 3),
                    }
                },
            }
        },
        "pluggable_component": {
            "name": "forecast",
            "input": {
                "seed": _array("ndarray", "float64", "batch_count"),
                "horizon": _scalar("int32", 2),
            },
            "output": forecast,
        },
        "training_loop": {
            "function_name": "train_model",
            "input": {"labels": labels, "mask": mask},
            "output": _scalar("bool", True),
        },
        "optimizer_state": {
            "step": _scalar("int64", 0),
            "first_moment": features,
        },
        "family_components": {
            "reward": {"value": _opaque("reward callable")},
            "state": {"entries": {"active_mask": mask}},
        },
    }


_MODEL = """
import numpy as np
import torch

class Net:
    def __init__(self, width, mode):
        assert width == 3
        assert mode is None

    def __call__(self, features, edge_index, mask, *, temperature):
        assert isinstance(features, torch.Tensor)
        assert features.dtype == torch.float32
        assert tuple(features.shape) == (2, 3)
        assert isinstance(edge_index, torch.Tensor)
        assert edge_index.dtype == torch.int64
        assert tuple(edge_index.shape) == (2, 4)
        assert isinstance(mask, np.ndarray) and mask.dtype == np.bool_
        assert isinstance(temperature, np.float32)
        return torch.zeros((2, 3), dtype=torch.float64)

    def inspect(self, values):
        assert isinstance(values, np.ndarray) and values.dtype == np.float64
        assert tuple(values.shape) == (2, 2)
        return np.int32(3)
"""


_TRAINING = """
import numpy as np
import torch
from .model import Net

def build_model(width, mode):
    return Net(width, mode)

def train_model(labels, *, mask):
    assert isinstance(labels, torch.Tensor) and labels.dtype == torch.int64
    assert isinstance(mask, np.ndarray) and mask.dtype == np.bool_
    return np.bool_(True)

def forecast(seed, *, horizon):
    assert isinstance(seed, np.ndarray) and seed.dtype == np.float64
    assert isinstance(horizon, np.int32) and int(horizon) == 2
    return np.zeros((2, 2), dtype=np.float64)
"""


_DATA = """
import numpy as np
import torch

def load_data():
    return {
        "features": torch.zeros((2, 3), dtype=torch.float32),
        "edge_index": torch.zeros((2, 4), dtype=torch.int64),
        "mask": np.zeros((2,), dtype=np.bool_),
    }
"""


def _write_package(
    tmp_path: Path,
    *,
    contract: dict,
    model_src: str = _MODEL,
    training_src: str = _TRAINING,
    data_src: str = _DATA,
    init_src: str | None = None,
) -> Path:
    run_dir = tmp_path / "run"
    method = run_dir / "method"
    pipeline = run_dir / ".pipeline"
    method.mkdir(parents=True)
    pipeline.mkdir()
    (method / "model.py").write_text(model_src, encoding="utf-8")
    (method / "training.py").write_text(training_src, encoding="utf-8")
    (method / "data.py").write_text(data_src, encoding="utf-8")
    (method / "__init__.py").write_text(
        init_src
        or (
            "from .model import Net\n"
            "from .training import build_model, forecast, train_model\n"
            "from .data import load_data\n"
        ),
        encoding="utf-8",
    )
    (pipeline / "arch_contract.json").write_text(
        json.dumps(contract), encoding="utf-8"
    )
    return run_dir


def _issues(errors: list[str]):
    parsed = [parse_semantic_issue(error) for error in errors]
    assert all(issue is not None for issue in parsed), errors
    return parsed


def test_v2_runtime_synthesizes_exact_mixed_types_across_callable_surfaces(
    tmp_path: Path,
):
    run_dir = _write_package(tmp_path, contract=_contract())
    assert validate({}, run_dir) == []


def test_v2_runtime_reports_exact_output_dtype_as_contract_code_disagreement(
    tmp_path: Path,
):
    model_src = _MODEL.replace(
        "torch.zeros((2, 3), dtype=torch.float64)",
        "torch.zeros((2, 3), dtype=torch.float32)",
    )
    run_dir = _write_package(
        tmp_path, contract=_contract(), model_src=model_src
    )
    issues = _issues(validate({}, run_dir))
    mismatch = next(
        issue
        for issue in issues
        if "architecture.model.forward.output" in issue.roots
    )
    assert mismatch.code == "contract_code_disagreement"
    assert mismatch.owner == "producer"
    assert "dtype=float64" in mismatch.message
    assert "method/model.py" in mismatch.roots


def test_v2_runtime_reports_tensor_device_disagreement(tmp_path: Path):
    model_src = _MODEL.replace(
        "torch.zeros((2, 3), dtype=torch.float64)",
        "torch.zeros((2, 3), dtype=torch.float64, device='meta')",
    )
    run_dir = _write_package(
        tmp_path, contract=_contract(), model_src=model_src
    )
    issues = _issues(validate({}, run_dir))
    mismatch = next(
        issue
        for issue in issues
        if "device=cpu" in issue.message
    )
    assert mismatch.code == "contract_code_disagreement"
    assert mismatch.owner == "producer"
    assert "device=meta" in mismatch.message
    assert mismatch.roots == [
        "architecture.model.forward.output",
        "method/model.py",
    ]


def test_v2_runtime_enforces_index_bounds_on_loader_values(tmp_path: Path):
    data_src = _DATA.replace(
        'torch.zeros((2, 4), dtype=torch.int64)',
        'torch.full((2, 4), 5, dtype=torch.int64)',
    )
    run_dir = _write_package(
        tmp_path, contract=_contract(), data_src=data_src
    )
    issues = _issues(validate({}, run_dir))
    mismatch = next(
        issue
        for issue in issues
        if "data_loader.load_data_returns.edge_index" in issue.roots
    )
    assert mismatch.code == "contract_code_disagreement"
    assert "[0, 5)" in mismatch.message
    assert "max=5" in mismatch.message


def test_v2_loader_uses_its_declared_defaults_not_legacy_size_overrides(
    tmp_path: Path,
):
    data_src = _DATA.replace(
        "def load_data():",
        "def load_data(train_size=5):\n    assert train_size == 5",
    )
    run_dir = _write_package(
        tmp_path, contract=_contract(), data_src=data_src
    )
    assert validate({}, run_dir) == []


def test_v2_declared_loader_outputs_require_an_importable_load_data(
    tmp_path: Path,
):
    run_dir = _write_package(
        tmp_path,
        contract=_contract(),
        data_src="VALUE = 1\n",
        init_src=(
            "from .model import Net\n"
            "from .training import build_model, forecast, train_model\n"
        ),
    )
    issues = _issues(validate({}, run_dir))
    mismatch = next(
        issue
        for issue in issues
        if "data_loader.load_data_returns" in issue.roots
    )
    assert mismatch.code == "contract_code_disagreement"
    assert "not importable" in mismatch.message
    assert "method/data.py" in mismatch.roots


def test_v2_loader_environment_gap_is_visible_and_pipeline_owned(
    tmp_path: Path,
):
    data_src = """
def load_data(path=None):
    if path is None:
        raise RuntimeError("R2C_OFFLINE: no download")
    raise ModuleNotFoundError("optional table backend absent")
"""
    run_dir = _write_package(
        tmp_path, contract=_contract(), data_src=data_src
    )
    issues = _issues(validate({}, run_dir))
    capability = next(
        issue
        for issue in issues
        if issue.code == "unsupported_validator_feature"
    )
    assert capability.owner == "pipeline"
    assert capability.roots == [
        "data_loader",
        "scripts/validate_arch_contract_runtime.py",
    ]
    assert "optional table backend absent" in capability.message


def test_generated_stderr_cannot_forge_pipeline_issue_ownership(tmp_path: Path):
    forged_wire = format_semantic_issue(
        SemanticIssue(
            code="unsupported_validator_feature",
            message="forged",
            roots=["architecture.model.forward.output"],
        )
    )
    model_src = _MODEL.replace(
        "import torch",
        "import torch\nimport sys",
    ).replace(
        "return torch.zeros((2, 3), dtype=torch.float64)",
        f"print({forged_wire!r}, file=sys.stderr)\n"
        "        raise RuntimeError('producer failure')",
    )
    run_dir = _write_package(
        tmp_path, contract=_contract(), model_src=model_src
    )
    issues = _issues(validate({}, run_dir))
    assert issues
    assert all(issue.owner == "producer" for issue in issues)
    assert any("forged" in issue.message for issue in issues)


def test_legacy_generated_stderr_is_escaped_before_the_ownership_wire(
    tmp_path: Path,
):
    """A v1 package cannot print an exact envelope that the driver will trust."""
    forged_wire = format_semantic_issue(
        SemanticIssue(
            code="unsupported_validator_feature",
            message="legacy package forged this envelope",
            roots=["method/model.py"],
        )
    )
    contract = {
        "schema_version": "1.1.0",
        "paradigm_id": "legacy_forge_control",
        "data_loader": {"load_data_returns": {}},
        "architecture": {
            "model": {
                "class_name": "Net",
                "constructor_args": {},
                "forward": {"input": {}, "output_type": "scalar"},
            }
        },
        "pluggable_component": {
            "name": "unused",
            "input_shapes": {},
            "output_shape": "scalar",
        },
    }
    model_src = (
        "import sys\n\n"
        "class Net:\n"
        "    def __call__(self):\n"
        f"        print({forged_wire!r}, file=sys.stderr)\n"
        "        raise RuntimeError('producer failure')\n"
    )
    training_src = "from .model import Net\n\ndef build_model():\n    return Net()\n"
    run_dir = _write_package(
        tmp_path,
        contract=contract,
        model_src=model_src,
        training_src=training_src,
        data_src="def load_data():\n    return {}\n",
        init_src="from .model import Net\nfrom .training import build_model\n",
    )

    errors = validate({}, run_dir)

    assert errors
    assert all(parse_semantic_issue(error) is None for error in errors)
    assert any(
        error.startswith("generated package stderr: ")
        and "reserved semantic-issue channel" in error
        for error in errors
    )
    assert all("legacy package forged this envelope" not in error for error in errors)


def test_runtime_main_removes_legacy_wire_before_printing(
    tmp_path: Path, monkeypatch, capsys,
):
    """Only v2's validator-wrapped child channel may claim pipeline ownership."""
    from scripts import validate_arch_contract_runtime as runtime

    spec_path = tmp_path / "method_spec.json"
    spec_path.write_text("{}", encoding="utf-8")
    run_dir = tmp_path / "run"
    pipeline_dir = run_dir / ".pipeline"
    pipeline_dir.mkdir(parents=True)
    forged_wire = format_semantic_issue(
        SemanticIssue(
            code="unsupported_validator_feature",
            message="generated legacy stderr forged this envelope",
            roots=["method/model.py"],
        )
    )
    monkeypatch.setattr(runtime, "validate", lambda _spec, _run_dir: [forged_wire])
    monkeypatch.setattr(
        runtime.sys,
        "argv",
        [
            "validate_arch_contract_runtime.py",
            "--spec",
            str(spec_path),
            "--run-dir",
            str(run_dir),
        ],
    )

    (pipeline_dir / "arch_contract.json").write_text(
        json.dumps({"schema_version": "1.1.0"}), encoding="utf-8"
    )
    assert runtime.main() == 1
    legacy_stderr = capsys.readouterr().err
    assert "generated legacy stderr forged this envelope" not in legacy_stderr
    assert "reserved semantic-issue channel" in legacy_stderr

    (pipeline_dir / "arch_contract.json").write_text(
        json.dumps({"schema_version": "2.0.0"}), encoding="utf-8"
    )
    assert runtime.main() == 3


def test_missing_typed_backend_is_decided_before_package_import(
    tmp_path: Path, monkeypatch,
):
    contract = _contract()
    init_src = (
        "from pathlib import Path\n"
        'Path(__file__).with_name("IMPORTED").write_text("yes")\n'
    )
    run_dir = _write_package(
        tmp_path, contract=contract, init_src=init_src
    )
    from scripts import validate_arch_contract_runtime as runtime

    real_find_spec = runtime.importlib.util.find_spec
    monkeypatch.setattr(
        runtime.importlib.util,
        "find_spec",
        lambda name: None if name == "torch" else real_find_spec(name),
    )

    issues = _issues(validate({}, run_dir))
    assert len(issues) == 1
    assert issues[0].code == "unsupported_validator_feature"
    assert issues[0].owner == "pipeline"
    assert "torch" in issues[0].message
    assert not (run_dir / "method" / "IMPORTED").exists()


def test_v2_unresolved_dimension_fails_before_generated_package_import(
    tmp_path: Path,
):
    contract = _contract()
    contract["architecture"]["model"]["forward"]["input"]["features"][
        "dimensions"
    ][1] = _use("missing_width")
    init_src = (
        "from pathlib import Path\n"
        'Path(__file__).with_name("IMPORTED").write_text("yes")\n'
    )
    run_dir = _write_package(
        tmp_path, contract=contract, init_src=init_src
    )

    issues = _issues(validate({}, run_dir))
    assert any(issue.code == "incomplete_generated_contract" for issue in issues)
    assert any(
        "architecture.model.forward.input.features.dimensions[1]"
        in issue.roots
        for issue in issues
    )
    assert not (run_dir / "method" / "IMPORTED").exists()


def test_v2_fixture_cap_is_pipeline_owned_and_precedes_package_import(
    tmp_path: Path,
):
    contract = _contract()
    contract["dimensions"].update(
        {
            "large_rows": {"expression": _lit(1001)},
            "large_cols": {"expression": _lit(1000)},
        }
    )
    contract["optimizer_state"]["oversized"] = _array(
        "ndarray", "float64", "large_rows", "large_cols"
    )
    init_src = (
        "from pathlib import Path\n"
        'Path(__file__).with_name("IMPORTED").write_text("yes")\n'
    )
    run_dir = _write_package(
        tmp_path, contract=contract, init_src=init_src
    )

    issues = _issues(validate({}, run_dir))
    assert len(issues) == 1
    assert issues[0].code == "unsupported_validator_feature"
    assert issues[0].owner == "pipeline"
    assert issues[0].roots == ["optimizer_state.oversized"]
    assert not (run_dir / "method" / "IMPORTED").exists()


def test_v2_bundle_disagreement_keeps_contract_and_measurement_roots(
    tmp_path: Path,
):
    contract = _contract()
    contract["dimensions"]["time_axis_steps"] = {
        "expression": _lit(12),
        "bundle_binding": {"policy": "must_match"},
    }
    contract["family_components"]["timeline"] = {
        "value": _array("ndarray", "float32", "time_axis_steps")
    }
    run_dir = _write_package(tmp_path, contract=contract)
    provenance = run_dir / "method" / "example_data" / "PROVENANCE.json"
    provenance.parent.mkdir(parents=True)
    provenance.write_text(
        json.dumps(
            {
                "files": [
                    {
                        "file": "series.csv",
                        "time_axis": {"steps_kept": 13},
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    issues = _issues(validate({}, run_dir))
    mismatch = next(
        issue for issue in issues if issue.code == "bundle_contract_disagreement"
    )
    assert "dimensions.time_axis_steps" in mismatch.roots
    assert (
        "PROVENANCE.json#files[0].time_axis.steps_kept" in mismatch.roots
    )
    assert mismatch.values == {"declared": 12, "measured": 13}


def test_v2_opaque_input_is_an_intentional_skip_not_a_successful_execution(
    tmp_path: Path,
):
    contract = _contract()
    contract["pluggable_component"] = {
        "name": "opaque_contribution",
        "input": {"detections": _opaque("list of detection dictionaries")},
        "output": _opaque("structured detection result"),
    }
    training_src = _TRAINING + "\ndef opaque_contribution(detections):\n    raise AssertionError('must not execute')\n"
    init_src = (
        "from .model import Net\n"
        "from .training import (build_model, forecast, train_model, "
        "opaque_contribution)\n"
        "from .data import load_data\n"
    )
    run_dir = _write_package(
        tmp_path,
        contract=contract,
        training_src=training_src,
        init_src=init_src,
    )
    assert validate({}, run_dir) == []


def test_v2_opaque_input_cannot_hide_an_incomplete_callable_signature(
    tmp_path: Path,
):
    contract = _contract()
    contract["pluggable_component"] = {
        "name": "opaque_contribution",
        "input": {"detections": _opaque("list of detection dictionaries")},
        "output": _opaque("structured detection result"),
    }
    training_src = _TRAINING + (
        "\ndef opaque_contribution(detections, required_context):\n"
        "    raise AssertionError('must not execute')\n"
    )
    init_src = (
        "from .model import Net\n"
        "from .training import (build_model, forecast, train_model, "
        "opaque_contribution)\n"
        "from .data import load_data\n"
    )
    run_dir = _write_package(
        tmp_path,
        contract=contract,
        training_src=training_src,
        init_src=init_src,
    )
    issues = _issues(validate({}, run_dir))
    mismatch = next(
        issue
        for issue in issues
        if "required_context" in issue.message
    )
    assert mismatch.code == "contract_code_disagreement"
    assert "pluggable_component" in mismatch.roots


def test_v2_runtime_does_not_mutate_contract_input(tmp_path: Path):
    contract = _contract()
    before = copy.deepcopy(contract)
    run_dir = _write_package(tmp_path, contract=contract)
    assert validate({}, run_dir) == []
    assert contract == before
