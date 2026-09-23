"""_import_alias_for_class sees imports anywhere in the tree.

Regression for the ACC 2026-07-14 stage-2b degrade: training.py held a
perfectly valid `if TYPE_CHECKING: from .model import ...` and the old
tree.body-only loop could not see it, so the validator reported a missing
import that was present (false negative, judge-classified pipeline_bug).
"""

from __future__ import annotations

import ast

from validate_architecture_coder_output import _import_alias_for_class


def test_type_checking_guarded_import_is_seen():
    tree = ast.parse(
        "from typing import TYPE_CHECKING\n"
        "if TYPE_CHECKING:\n"
        "    from .model import DoubleIntegratorDynamics\n"
    )
    assert _import_alias_for_class(tree, "DoubleIntegratorDynamics")


def test_top_level_and_aliased_imports_still_seen():
    tree = ast.parse("from .model import Foo as Bar\n")
    assert _import_alias_for_class(tree, "Bar")
    assert _import_alias_for_class(tree, "Foo")


def test_missing_import_still_fails():
    tree = ast.parse("import numpy as np\n")
    assert not _import_alias_for_class(tree, "Foo")


# ---------------------------------------------------------------------------
# dead_code_masking — constructed-but-never-read nn.* submodules
# ---------------------------------------------------------------------------


def test_dead_constructed_submodule_flagged():
    """The pdfgnn 2026-08-04 shape: decoder_projection built in __init__ to
    carry the covariate concatenation, never called anywhere, masking the
    missing mechanism."""
    import ast
    from validate_architecture_coder_output import dead_module_members

    src = """
import torch.nn as nn
class GraphDeepAR(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(9, 128)
        self.decoder_projection = nn.Linear(25, 9)
    def forward(self, x):
        out, _ = self.lstm(x)
        return out
"""
    hits = dead_module_members(ast.parse(src))
    assert [(cls, attr) for _, cls, attr in hits] == [("GraphDeepAR", "decoder_projection")]


def test_used_submodules_not_flagged():
    import ast
    from validate_architecture_coder_output import dead_module_members

    src = """
import torch.nn as nn
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(25, 9)
        self.lstm = nn.LSTM(9, 128)
    def forward(self, x):
        return self.lstm(self.proj(x))
"""
    assert dead_module_members(ast.parse(src)) == []


def test_dynamic_attribute_access_skips_class():
    import ast
    from validate_architecture_coder_output import dead_module_members

    src = """
import torch.nn as nn
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.head_a = nn.Linear(4, 2)
    def forward(self, x, name):
        return getattr(self, name)(x)
"""
    assert dead_module_members(ast.parse(src)) == []


def test_non_nn_assignments_ignored():
    import ast
    from validate_architecture_coder_output import dead_module_members

    src = """
import torch.nn as nn
class Net(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = dict(cfg)
        self.window = compute_window(cfg)
    def forward(self, x):
        return x
"""
    assert dead_module_members(ast.parse(src)) == []


def test_submodule_used_from_training_module_not_flagged():
    """The bev-distill shape: projection heads constructed on the model
    classes and applied from the training module's distillation loop are
    live, not dead."""
    import ast
    from validate_architecture_coder_output import dead_module_members

    model_src = """
import torch.nn as nn
class BEVFormerStudent(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Linear(4, 8)
        self.proj_head_2d = nn.Linear(8, 16)
    def forward(self, x):
        return self.backbone(x)
"""
    training_src = """
def train_with_distillation(student, teacher, batch):
    feat = student(batch)
    projected = student.proj_head_2d(feat)
    return projected
"""
    hits = dead_module_members(
        ast.parse(model_src), (ast.parse(training_src),))
    assert hits == []
