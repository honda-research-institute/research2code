"""KD-1/KD-2 probe tests.

Pass side: the kd-wrong-op mutant's loss IS teacher-responsive (its defect is
shape, not signal — UB-1's job). Fail side: a synthetic reconstruction of the
bev-distill quality-no-op class (loss that ignores the teacher); no artifact
of that run was ever captured, so synthetic is the only fail-side path.
"""

from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

REPO = Path(__file__).resolve().parent.parent
ZOO = REPO / "tests" / "fixtures" / "zoo"

from probes.kd import (  # noqa: E402
    probe_teacher_signal_influence,
    probe_temperature_sensitivity,
)
from probes.package_loader import load_module_from_path  # noqa: E402


def test_kd1_passes_a_teacher_responsive_loss():
    mod = load_module_from_path(ZOO / "kd-wrong-op-reconstructed" / "method.py")
    v = probe_teacher_signal_influence(mod.weighted_feature_loss)
    assert v.verdict == "pass", v.message


def test_kd1_fails_a_teacher_ignoring_loss():
    # Synthetic reconstruction of the quality-no-op class: looks like a
    # distillation loss, never reads the teacher signal's content.
    def fake_distillation_loss(student_feats, teacher_feats):
        return (student_feats ** 2).mean() + 0.0 * teacher_feats.sum() * 0

    v = probe_teacher_signal_influence(fake_distillation_loss)
    assert v.verdict == "fail", v.message
    assert "no effect" in v.message


def test_kd2_varies_temperature_on_a_standard_kd_loss():
    import torch.nn.functional as F

    def kd_loss(student_logits, teacher_logits, temperature=1.0):
        t = temperature
        return F.kl_div(
            F.log_softmax(student_logits / t, dim=1),
            F.softmax(teacher_logits / t, dim=1),
            reduction="batchmean",
        ) * (t * t)

    v = probe_temperature_sensitivity(kd_loss)
    assert v is not None and v.verdict == "pass", v and v.message


def test_kd2_flags_inert_temperature():
    def kd_loss(student_logits, teacher_logits, temperature=1.0):
        del temperature  # accepted, never used — the inert-knob class
        return ((student_logits - teacher_logits) ** 2).mean()

    v = probe_temperature_sensitivity(kd_loss)
    assert v is not None and v.verdict == "fail", v and v.message


def test_kd2_not_applicable_without_temperature_param():
    def plain_loss(student_logits, teacher_logits):
        return ((student_logits - teacher_logits) ** 2).mean()

    assert probe_temperature_sensitivity(plain_loss) is None


def test_kd1_unprobeable_without_recognizable_params():
    def opaque(a, b):
        return (a - b).mean()

    v = probe_teacher_signal_influence(opaque)
    assert v.verdict == "unprobeable"


# ---------------------------------------------------------------------------
# Contract-driven kit (model-and-batch losses — the bev-distill shape)
# ---------------------------------------------------------------------------

from probes.kd import (  # noqa: E402
    kd_loss_kit,
    probe_kd_teacher_influence,
    probe_kd_temperature,
    requires_model_batch_kit,
)

# A miniature cross-modal package in the bev-distill interface shape:
# builders + a loss taking (student, teacher, batch, seed). The models emit
# the contract-declared pred_* keys so target seeding is exercised.
_KIT_PKG_COMMON = '''
import torch
from torch import nn


class _Det(nn.Module):
    def __init__(self, num_classes=3, hidden_dim=8, num_queries=5):
        super().__init__()
        self.lin = nn.Linear(3, hidden_dim)
        self.cls = nn.Linear(hidden_dim, num_classes)
        self.box = nn.Linear(hidden_dim, 4)
        self.num_queries = num_queries

    def forward(self, x):
        if x.dim() == 4:  # (B, C, H, W) images -> (B, H*W, C) rows
            x = x.flatten(2).transpose(1, 2)
        h = torch.tanh(self.lin(x[:, : self.num_queries, :]))
        return {"pred_logits": self.cls(h),
                "pred_boxes": self.box(h).abs() + 0.1,
                "pred_features": h}


def build_student(num_classes, hidden_dim=8, num_queries=5):
    return _Det(num_classes, hidden_dim, num_queries)


def build_teacher(num_classes, hidden_dim=8, num_queries=5):
    return _Det(num_classes, hidden_dim, num_queries)
'''

_KIT_LOSS_GOOD = _KIT_PKG_COMMON + '''
def compute_distillation_loss(student, teacher, batch, seed, tau=0.07):
    s = student(batch["student_inputs"])
    with torch.no_grad():
        t = teacher(batch["teacher_inputs"])
    feat = ((s["pred_features"] - t["pred_features"]) ** 2).mean()
    a = s["pred_features"].flatten(0, 1)
    b = t["pred_features"].flatten(0, 1)
    sim = (a @ b.t()) / tau
    nce = torch.logsumexp(sim, dim=-1).mean() - sim.diag().mean()
    return feat + nce
'''

_KIT_LOSS_CONSTANT_TEACHER = _KIT_PKG_COMMON + '''
def compute_distillation_loss(student, teacher, batch, seed, tau=0.07):
    del teacher, tau  # accepted, never read — the C1 no-op class
    s = student(batch["student_inputs"])
    return (s["pred_features"] ** 2).mean()
'''

_KIT_LOSS_INERT_TAU = _KIT_PKG_COMMON + '''
def compute_distillation_loss(student, teacher, batch, seed, tau=0.07):
    del tau  # accepted, never used
    s = student(batch["student_inputs"])
    with torch.no_grad():
        t = teacher(batch["teacher_inputs"])
    return ((s["pred_features"] - t["pred_features"]) ** 2).mean()
'''

_KIT_LOSS_UNDECLARED_KEY = _KIT_PKG_COMMON + '''
def compute_distillation_loss(student, teacher, batch, seed):
    return batch["calibration_matrix"].sum()
'''

_KIT_ARCH_CONTRACT = {
    "pluggable_component": {
        "name": "compute_distillation_loss",
        "batch_dict_shape": {
            "student_inputs": "(B, 3, H_img, W_img)",
            "teacher_inputs": "(B, N_points, 3)",
            "targets": "list[dict] of length B, keys: boxes (N_obj, 4), "
                       "labels (N_obj,)",
        },
    },
}


def _kit_module(tmp_path, source, name="kit_pkg"):
    py = tmp_path / f"{name}.py"
    py.write_text(source)
    return load_module_from_path(py)


def test_requires_model_batch_kit_discriminates_the_two_shapes():
    def tensor_loss(student_logits, teacher_logits, temperature=1.0):
        return ((student_logits - teacher_logits) ** 2).mean()

    def model_batch_loss(student, teacher, batch, seed, tau=0.07):
        return batch

    assert not requires_model_batch_kit(tensor_loss)
    assert requires_model_batch_kit(model_batch_loss)


def test_kit_tensor_mode_keeps_the_fast_path():
    mod = load_module_from_path(ZOO / "kd-wrong-op-reconstructed" / "method.py")
    kit = kd_loss_kit(mod, "weighted_feature_loss")
    assert isinstance(kit, dict) and kit["mode"] == "tensor"
    v = probe_kd_teacher_influence(kit)
    assert v.verdict == "pass", v.message  # unchanged zoo verdict


def test_kit_probes_pass_a_live_model_batch_loss(tmp_path):
    mod = _kit_module(tmp_path, _KIT_LOSS_GOOD)
    kit = kd_loss_kit(mod, "compute_distillation_loss",
                      arch_contract=_KIT_ARCH_CONTRACT)
    assert isinstance(kit, dict), kit
    assert kit["mode"] == "model_batch"
    v1 = probe_kd_teacher_influence(kit)
    assert v1.verdict == "pass", v1.message
    v2 = probe_kd_temperature(kit)
    assert v2 is not None and v2.verdict == "pass", v2 and v2.message


def test_kit_probes_are_deterministic(tmp_path):
    verdicts = []
    for _ in range(2):
        mod = _kit_module(tmp_path, _KIT_LOSS_GOOD)
        kit = kd_loss_kit(mod, "compute_distillation_loss",
                          arch_contract=_KIT_ARCH_CONTRACT)
        v1 = probe_kd_teacher_influence(kit)
        v2 = probe_kd_temperature(kit)
        verdicts.append((v1.verdict, v1.message, v2.verdict, v2.message))
    assert verdicts[0] == verdicts[1]


def test_kit_kd1_fails_a_constant_teacher_model_batch_loss(tmp_path):
    mod = _kit_module(tmp_path, _KIT_LOSS_CONSTANT_TEACHER)
    kit = kd_loss_kit(mod, "compute_distillation_loss",
                      arch_contract=_KIT_ARCH_CONTRACT)
    assert isinstance(kit, dict), kit
    v = probe_kd_teacher_influence(kit)
    assert v.verdict == "fail", v.message
    assert "no effect" in v.message


def test_kit_kd2_fails_an_inert_tau_model_batch_loss(tmp_path):
    mod = _kit_module(tmp_path, _KIT_LOSS_INERT_TAU)
    kit = kd_loss_kit(mod, "compute_distillation_loss",
                      arch_contract=_KIT_ARCH_CONTRACT)
    assert isinstance(kit, dict), kit
    v = probe_kd_temperature(kit)
    assert v is not None and v.verdict == "fail", v and v.message


def test_kit_unprobeable_names_the_missing_builder(tmp_path):
    src = _KIT_LOSS_GOOD.replace("def build_teacher", "def _hidden_builder")
    mod = _kit_module(tmp_path, src)
    kit = kd_loss_kit(mod, "compute_distillation_loss",
                      arch_contract=_KIT_ARCH_CONTRACT)
    assert isinstance(kit, str) and "build_teacher" in kit


def test_kit_unprobeable_names_an_undeclared_batch_key(tmp_path):
    mod = _kit_module(tmp_path, _KIT_LOSS_UNDECLARED_KEY)
    kit = kd_loss_kit(mod, "compute_distillation_loss",
                      arch_contract=_KIT_ARCH_CONTRACT)
    assert isinstance(kit, str) and "calibration_matrix" in kit


def test_kit_unprobeable_without_batch_shape_contract(tmp_path):
    mod = _kit_module(tmp_path, _KIT_LOSS_GOOD)
    kit = kd_loss_kit(mod, "compute_distillation_loss", arch_contract=None)
    assert isinstance(kit, str) and "batch_dict_shape" in kit


_KIT_LOSS_DICT_TARGETS = _KIT_PKG_COMMON + '''
def compute_distillation_loss(student, teacher, batch, seed, tau=0.07):
    s = student(batch["student_inputs"])
    with torch.no_grad():
        t = teacher(batch["teacher_inputs"])
    # Consumes the dict-of-batched-tensors targets form (the bev-distill
    # 2026-07-04 roll's declared shape).
    gt_boxes = batch["targets"]["boxes"]
    gt_labels = batch["targets"]["labels"]
    feat = ((s["pred_features"] - t["pred_features"]) ** 2).mean()
    anchor = (gt_boxes.sum() * 0 + gt_labels.sum() * 0)
    a = s["pred_features"].flatten(0, 1)
    b = t["pred_features"].flatten(0, 1)
    sim = (a @ b.t()) / tau
    nce = torch.logsumexp(sim, dim=-1).mean() - sim.diag().mean()
    return feat + nce + anchor
'''

_KIT_ARCH_CONTRACT_DICT_TARGETS = {
    "pluggable_component": {
        "name": "compute_distillation_loss",
        "batch_dict_shape": {
            "student_inputs": "(B, 3, H_img, W_img)",
            "teacher_inputs": "(B, N_points, 3)",
            "targets": "dict — {'boxes': (B, max_objects, 4), "
                       "'labels': (B, max_objects)}",
        },
    },
}


def test_kit_synthesizes_dict_form_targets(tmp_path):
    """The second concrete targets declaration (bev-distill 2026-07-04):
    dict of batched tensors instead of list[dict]. The kit must synthesize
    that form so the probes bind instead of reporting unparseable-shape."""
    import torch
    mod = _kit_module(tmp_path, _KIT_LOSS_DICT_TARGETS)
    kit = kd_loss_kit(mod, "compute_distillation_loss",
                      arch_contract=_KIT_ARCH_CONTRACT_DICT_TARGETS)
    assert isinstance(kit, dict), kit
    targets = kit["batch"]["targets"]
    assert isinstance(targets, dict)
    assert tuple(targets["boxes"].shape) == (4, 3, 4)
    assert tuple(targets["labels"].shape) == (4, 3)
    v1 = probe_kd_teacher_influence(kit)
    assert v1.verdict == "pass", v1.message
    v2 = probe_kd_temperature(kit)
    assert v2 is not None and v2.verdict == "pass", v2 and v2.message


def test_kit_list_dict_targets_form_unchanged(tmp_path):
    """Regression guard: the original list[dict] declaration keeps its
    per-element form after the dict-form addition."""
    mod = _kit_module(tmp_path, _KIT_LOSS_GOOD)
    kit = kd_loss_kit(mod, "compute_distillation_loss",
                      arch_contract=_KIT_ARCH_CONTRACT)
    assert isinstance(kit, dict), kit
    targets = kit["batch"]["targets"]
    assert isinstance(targets, list) and len(targets) == 4
    assert set(targets[0]) == {"boxes", "labels"}


def test_kit_capitalized_list_dict_decl_synthesizes(tmp_path):
    """The detr-distill 2026-07-05 declaration, verbatim: 'List[dict]' with
    a capital L. The case-sensitive form match sent this recognized targets
    shape to the tensor parser, every KD probe read unprobeable, and the
    delivery dropped to new-territory — the whole 'KD structured-targets
    gap' was this one character. Form matching is now case-insensitive."""
    contract = {
        "pluggable_component": {
            "name": "compute_distillation_loss",
            "batch_dict_shape": {
                "student_inputs": "(B, 3, H_img, W_img)",
                "teacher_inputs": "(B, N_points, 3)",
                "targets": "List[dict] of length B, each dict has "
                           "'boxes': (N_obj, 4) and 'labels': (N_obj)",
            },
        },
    }
    mod = _kit_module(tmp_path, _KIT_LOSS_GOOD)
    kit = kd_loss_kit(mod, "compute_distillation_loss",
                      arch_contract=contract)
    assert isinstance(kit, dict), kit
    targets = kit["batch"]["targets"]
    assert isinstance(targets, list) and len(targets) == 4
    assert set(targets[0]) == {"boxes", "labels"}
    v1 = probe_kd_teacher_influence(kit)
    assert v1.verdict == "pass", v1.message
