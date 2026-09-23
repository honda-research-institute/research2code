"""Blocker #3 regression: the architecture-coder validator must recognize every
torch dropout layer — including the dimensional variants `nn.Dropout2d/1d/3d`.

The live bug (validate_architecture_coder_output.py:241, unchanged since 2026-05-08)
was `name == "Dropout" or name.endswith("Dropout")`, which is False for `Dropout2d`
(it ends with "2d"). For an image-data MC-dropout CNN — the standard
`nn.Dropout2d` case — the architecture-coder produced CORRECT code and the validator
spuriously rejected it ("no architecture class has an nn.Dropout layer"), forcing the
stage-2b degrade in the 2026-06-22 bayesian-active-learning run.

These tests exercise the exact gate `validate()` runs (line ~538): a spec that flags
MC-dropout as essential -> `needs_dropout_layers`, then
`any(_class_has_dropout_layer(c) for c in arch_classes)`. Known-good (Dropout2d /
plain Dropout) must pass; known-bad (no dropout at all) must still flag.
"""
import ast

from scripts.validate_architecture_coder_output import (
    _class_has_dropout_layer,
    _find_public_top_level_classes,
    _required_architecture_hooks,
)

MC_DROPOUT_SPEC = {
    "critical_requirements": {
        "model": {
            "specific_features": [
                {"feature": "MC dropout active at inference", "severity": "essential"},
            ]
        }
    }
}


def _classes(src: str):
    return _find_public_top_level_classes(ast.parse(src))


def _gate_passes(model_src: str, spec: dict) -> bool:
    """Reproduce validate()'s dropout gate: True == no spurious rejection.

    Mirrors the 2026-07-06 gate: ALL top-level classes, private included —
    dropout-at-inference is a module property, not a public-interface one
    (the ICRA21_HICA two-guard trap below)."""
    hooks = _required_architecture_hooks(spec)
    if not hooks["needs_dropout_layers"]:
        return True
    all_classes = [
        node for node in ast.parse(model_src).body
        if isinstance(node, ast.ClassDef)
    ]
    return any(_class_has_dropout_layer(c) for c in all_classes)


# --- forward_with_embedding heuristic: scoped to active learning -------------
#
# detr-distill 2026-07-03: a knowledge-distillation spec's essential feature
# "Teacher query embeddings accessible after refinement" (served by the
# spec's own required_model_methods) tripped the keyword heuristic and
# halted 2.b demanding the AL-only forward_with_embedding hook. The
# halt-judge classified it pipeline_bug/high; the heuristic now applies only
# to active_learning paradigms, with an explicit required_model_methods
# declaration winning for any paradigm.

def _spec(paradigm_id, feature, required_methods=None):
    model = {
        "specific_features": [{"feature": feature, "severity": "essential"}],
    }
    if required_methods is not None:
        model["required_model_methods"] = required_methods
    return {
        "comparison": {"classification": {"id": paradigm_id}},
        "critical_requirements": {"model": model},
    }


def test_embedding_keyword_does_not_fire_for_non_al_paradigms():
    hooks = _required_architecture_hooks(_spec(
        "knowledge_distillation/detection",
        "Teacher query embeddings accessible after refinement",
    ))
    assert hooks["needs_forward_with_embedding"] is False


def test_embedding_keyword_still_fires_for_active_learning():
    hooks = _required_architecture_hooks(_spec(
        "active_learning/batch_acquisition",
        "penultimate-layer embedding for gradient computation",
    ))
    assert hooks["needs_forward_with_embedding"] is True


def test_explicit_required_model_method_declaration_wins_for_any_paradigm():
    hooks = _required_architecture_hooks(_spec(
        "knowledge_distillation/detection",
        "Teacher query embeddings accessible after refinement",
        required_methods=[{"name": "forward_with_embedding",
                           "signature": "forward_with_embedding(self, x)"}],
    ))
    assert hooks["needs_forward_with_embedding"] is True


# --- the detector itself: every torch dropout variant counts -----------------

def test_class_has_dropout_layer_matches_every_torch_variant():
    for layer in (
        "nn.Dropout(p=0.5)",
        "nn.Dropout2d(0.5)",      # the blocker #3 case (image-data MC dropout)
        "nn.Dropout1d(0.5)",
        "nn.Dropout3d(0.5)",
        "nn.AlphaDropout(0.3)",
        "nn.FeatureAlphaDropout(0.3)",
    ):
        src = f"class M(nn.Module):\n    def __init__(self):\n        super().__init__()\n        self.d = {layer}\n"
        assert _class_has_dropout_layer(_classes(src)[0]), f"{layer} not detected as a dropout layer"


def test_class_without_dropout_is_not_detected():
    src = "class M(nn.Module):\n    def __init__(self):\n        super().__init__()\n        self.fc = nn.Linear(10, 2)\n"
    assert not _class_has_dropout_layer(_classes(src)[0])


# --- the gate: known-good passes, known-bad flags ----------------------------

def test_dropout2d_cnn_passes_the_mc_dropout_gate():
    # known-GOOD (regression guard): a CNN using nn.Dropout2d for image-data MC
    # dropout. Correct code — must NOT be flagged. This is what the old
    # endswith("Dropout") check spuriously rejected.
    src = (
        "class MCDropoutCNN(nn.Module):\n"
        "    def __init__(self):\n"
        "        super().__init__()\n"
        "        self.conv = nn.Conv2d(1, 16, 3)\n"
        "        self.drop = nn.Dropout2d(p=0.5)  # active at inference for MC dropout\n"
    )
    assert _gate_passes(src, MC_DROPOUT_SPEC)


def test_plain_dropout_mlp_passes_the_mc_dropout_gate():
    # known-GOOD: plain nn.Dropout (the june9 MCDropoutMLP shape) — passed before
    # and must keep passing.
    src = (
        "class MCDropoutMLP(nn.Module):\n"
        "    def __init__(self):\n"
        "        super().__init__()\n"
        "        self.fc = nn.Linear(784, 256)\n"
        "        self.drop = nn.Dropout(p=0.5)\n"
    )
    assert _gate_passes(src, MC_DROPOUT_SPEC)


def test_no_dropout_still_flagged_when_mc_dropout_required():
    # known-BAD: spec requires MC dropout but the model has no dropout of any kind.
    # Broadening the match must NOT disable the real check — the gate must still fail.
    src = (
        "class PlainCNN(nn.Module):\n"
        "    def __init__(self):\n"
        "        super().__init__()\n"
        "        self.conv = nn.Conv2d(1, 16, 3)\n"
        "        self.fc = nn.Linear(16, 2)\n"
    )
    assert not _gate_passes(src, MC_DROPOUT_SPEC)


def test_private_dropout_backbone_passes_the_mc_dropout_gate():
    # ICRA21_HICA 2026-07-06 two-guard trap: a motion_planning manifest
    # declares exactly two PUBLIC plain-Python classes (dynamics +
    # collision), so the producer's dropout-carrying torch backbone must be
    # private — and the old public-only scan could never see it. The gate
    # must accept dropout in a private class; the halt-judge diagnosed the
    # trap as pipeline_bug/high.
    src = (
        "class _SharedBackbone(nn.Module):\n"
        "    def __init__(self):\n"
        "        super().__init__()\n"
        "        self.conv = nn.Conv2d(1, 16, 3)\n"
        "        self.drop = nn.Dropout(p=0.5)\n"
        "\n"
        "class SystemDynamics:\n"
        "    def step(self, x, u, dt):\n"
        "        return x\n"
        "\n"
        "class CollisionModel:\n"
        "    def is_in_collision(self, x):\n"
        "        return False\n"
    )
    assert _gate_passes(src, MC_DROPOUT_SPEC)
    # And the public-only view genuinely cannot see it (the trap's shape).
    assert not any(_class_has_dropout_layer(c) for c in _classes(src))
