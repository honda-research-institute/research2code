"""Arm A of the numeric-coherence pass (queue 11d) — training-budget sufficiency.

assess_budget builds a labeled-set-sized fixture and runs the live build_model +
train_from_scratch with the LIVE budget and LIVE dropout. A starved budget (the
GBALD shape: low max_epochs on a high-dropout model) lands near chance and the
assessment names a max_epochs that fits; a healthy budget passes; a non-training
package is N/A; a broken learning rate cannot be fixed by raising epochs.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from scripts.budget_sufficiency import assess_budget, labeled_set_size  # noqa: E402


# A 3-hidden-layer 0.5-dropout MLP (GBALD shape) whose train_from_scratch takes
# a REQUIRED batch_train_size (the drift 11b now prevents) and runs its
# convergence check under eval() (the 11c-correct pattern). With heavy dropout
# on a few-hundred-sample set, a low epoch budget starves it.
_DROPOUT_MLP_PKG = '''
import torch
import torch.nn as nn


class Net(nn.Module):
    def __init__(self, input_dim, n_classes, dropout_rate=0.5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256), nn.ReLU(), nn.Dropout(dropout_rate),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(dropout_rate),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(dropout_rate),
            nn.Linear(64, n_classes),
        )

    def forward(self, x):
        return self.net(x)


def build_model(input_dim, n_classes, dropout_rate=0.5):
    return Net(input_dim, n_classes, dropout_rate=dropout_rate)


def train_from_scratch(model, x_train, y_train, max_epochs, learning_rate,
                       batch_train_size, train_until_accuracy=None, seed=0):
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
        if train_until_accuracy is not None:
            model.eval()
            with torch.no_grad():
                acc = (model(x_train).argmax(1) == y_train).float().mean().item()
            model.train()
            if acc >= train_until_accuracy:
                break
    return model
'''

_NON_TRAINING_PKG = "VALUE = 1\n"


def _make_run(tmp_path: Path, *, package: str, params: dict) -> Path:
    run = tmp_path / "run"
    (run / "method").mkdir(parents=True)
    (run / ".pipeline").mkdir(parents=True)
    (run / "method" / "__init__.py").write_text(package, encoding="utf-8")
    (run / ".pipeline" / "params.json").write_text(
        json.dumps({"schema_version": "1.0.0", "params": params}), encoding="utf-8")
    return run


def _params(max_epochs: int, lr: float = 1e-3) -> dict:
    return {
        "max_epochs": {"value": max_epochs, "source": "system_inferred",
                       "reasoning": "smoke bound"},
        "batch_size": {"value": 100, "source": "paper",
                       "paper_section": "S7", "note": "paper batch"},
        "learning_rate": {"value": lr, "source": "system_inferred",
                          "reasoning": "Adam default"},
        "dropout_rate": {"value": 0.5, "source": "paper",
                         "paper_section": "S7", "note": "dropout 0.5"},
        "core_set_size": {"value": 100, "source": "system_default",
                          "paper_value": 1000, "reasoning": "smoke"},
    }


def test_labeled_set_size_from_core_set_plus_seed():
    assert labeled_set_size(_params(8)) == 110  # core_set_size 100 + seed 10


def test_starved_budget_flagged_with_suggested_raise(tmp_path):
    run = _make_run(tmp_path, package=_DROPOUT_MLP_PKG, params=_params(8))
    a = assess_budget(run)
    assert a["applicable"] is True
    assert a["sufficient_at_live"] is False, a
    assert a["suggested_max_epochs"] is not None and a["suggested_max_epochs"] > 8, a
    assert a["live_acc"] < a["target"]


def test_healthy_budget_passes(tmp_path):
    run = _make_run(tmp_path, package=_DROPOUT_MLP_PKG, params=_params(200))
    a = assess_budget(run)
    assert a["applicable"] is True
    assert a["sufficient_at_live"] is True, a
    assert a["suggested_max_epochs"] is None


def test_non_training_package_is_not_applicable(tmp_path):
    run = _make_run(tmp_path, package=_NON_TRAINING_PKG, params=_params(8))
    a = assess_budget(run)
    assert a["applicable"] is False


def test_broken_lr_cannot_be_fixed_by_raising_epochs(tmp_path):
    # lr=7.4 diverges at any epoch count, so the grid is exhausted and no
    # suggestion is offered (the caller flags rather than auto-raises).
    run = _make_run(tmp_path, package=_DROPOUT_MLP_PKG, params=_params(8, lr=7.4))
    a = assess_budget(run)
    assert a["applicable"] is True
    assert a["sufficient_at_live"] is False
    assert a["suggested_max_epochs"] is None, a


# --- orchestrator auto-raise -------------------------------------------------


def _valid_params_doc(max_epochs: int) -> dict:
    return {"schema_version": "1.0.0", "params": _params(max_epochs)}


def test_apply_budget_raise_writes_params_and_logs_assumption(tmp_path):
    from tests.helpers.state import make_state
    from run_pipeline import _apply_budget_raise

    run = tmp_path / "run"
    (run / ".pipeline").mkdir(parents=True)
    (run / ".pipeline" / "params.json").write_text(
        json.dumps(_valid_params_doc(8)), encoding="utf-8")
    state = make_state(run)

    assessment = {"live_acc": 0.33, "target": 0.58, "live_batch": 100,
                  "n_labeled_fixture": 110, "live_dropout": 0.5}
    ok, aid = _apply_budget_raise(state, suggested_max_epochs=50, assessment=assessment)
    assert ok, aid

    data = json.loads((run / ".pipeline" / "params.json").read_text())
    assert data["params"]["max_epochs"]["value"] == 50
    assert data["params"]["batch_size"]["value"] == 100  # untouched
    import run_layout
    assumptions = (run / run_layout.ASSUMPTIONS_MD).read_text(encoding="utf-8")
    assert aid in assumptions
    assert "max_epochs raised from 8 to 50" in assumptions
