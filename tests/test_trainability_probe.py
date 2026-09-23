"""UB-5 trains-on-synthetic: both sides validated on the REAL june9 package.

Pass side: the run's actual healthy config (lr=0.001). Fail side: the same
real package with lr=7.4 injected — the exact shipped failure, now caught
behaviorally in seconds.
"""

from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

REPO = Path(__file__).resolve().parent.parent
JUNE9 = REPO / "tests" / "fixtures" / "evidence" / "june9-gbald-run"

from probes.trainability import flat_live_params, probe_trains_on_synthetic  # noqa: E402


def test_ub5_passes_the_real_june9_config():
    import json
    live = flat_live_params(json.loads(
        (JUNE9 / "pipeline" / "params.json").read_text())["params"])
    assert live["learning_rate"] == 0.001
    # Budget knobs are probe-owned: the live max_epochs=8 is smoke
    # economics, not an optimizer property (full-batch = 1 step/epoch).
    v = probe_trains_on_synthetic(JUNE9, live_params=live)
    assert v.verdict == "pass", v.message


def test_ub5_fails_the_lr74_class_on_the_same_real_package():
    v = probe_trains_on_synthetic(
        JUNE9, live_params={"learning_rate": 7.4}, epochs_cap=15)
    assert v.verdict == "fail", v.message
    assert "7.4" in v.message


def test_ub5_unprobeable_without_training_functions(tmp_path):
    pkg = tmp_path / "method"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("VALUE = 1\n")
    v = probe_trains_on_synthetic(tmp_path)
    assert v.verdict == "unprobeable"


_PKG_WITH_REQUIRED_BATCH_ARG = '''
import torch
import torch.nn as nn


def build_model(input_dim, n_classes, dropout_rate=0.5):
    return nn.Sequential(
        nn.Linear(input_dim, 32), nn.ReLU(), nn.Linear(32, n_classes)
    )


def train_from_scratch(model, x_train, y_train, max_epochs, learning_rate,
                       batch_train_size, train_until_accuracy=None, seed=0):
    """GBALD-shaped signature: batch_train_size is REQUIRED with no default."""
    torch.manual_seed(seed)
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.CrossEntropyLoss()
    ds = torch.utils.data.TensorDataset(x_train, y_train)
    loader = torch.utils.data.DataLoader(ds, batch_size=batch_train_size, shuffle=True)
    for _ in range(max_epochs):
        for xb, yb in loader:
            opt.zero_grad()
            loss_fn(model(xb), yb).backward()
            opt.step()
    return model
'''


def _write_pkg(tmp_path, body: str):
    pkg = tmp_path / "method"
    pkg.mkdir()
    (pkg / "training.py").write_text(body)
    (pkg / "__init__.py").write_text(
        "from .training import build_model, train_from_scratch\n"
    )
    return tmp_path


def test_ub5_binds_when_training_fn_has_required_batch_arg(tmp_path):
    """Regression for the GBALD binding gap (queue 11b): a training function
    that declares a REQUIRED batch-size argument used to make UB-5
    unprobeable ('train_from_scratch requires unmapped parameter
    batch_train_size'), hiding the training-starvation class. Batch size is a
    probe-owned BUDGET knob now, so the probe binds and actually trains."""
    run_dir = _write_pkg(tmp_path, _PKG_WITH_REQUIRED_BATCH_ARG)
    v = probe_trains_on_synthetic(run_dir, live_params={"learning_rate": 1e-2})
    assert v.verdict == "pass", v.message


def test_ub5_still_fails_lr74_with_required_batch_arg(tmp_path):
    """The batch-knob binding does not mask a genuinely broken optimizer
    config: the lr=7.4 class still fails behaviorally even though the function
    now binds."""
    run_dir = _write_pkg(tmp_path, _PKG_WITH_REQUIRED_BATCH_ARG)
    v = probe_trains_on_synthetic(
        run_dir, live_params={"learning_rate": 7.4}, epochs_cap=15)
    assert v.verdict == "fail", v.message
