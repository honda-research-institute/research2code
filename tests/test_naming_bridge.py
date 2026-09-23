"""Stage 1 naming bridge — spec promises resolve to importable public names.

Design: the naming bridge design note (internal, not shipped) (approved
2026-07-16). Completes the general fix the public-symbol enforcement build
recorded as its honest gap (the public symbol enforcement design note,
internal, not shipped, "Build record"): system_provides
entries were prose, so the 2.b spec-surface check had nothing to bind.
Failure class: spec_promise_not_importable (ICRA 2026-07-13: two
spec-promised classifiers defined underscore-private with no re-export).

Covered here (the zoo gate, fixtures before any live-run claim):

- ICRA-shaped known-bad: declared symbols defined underscore-private with
  no re-export fail 2.b with the remedy-naming error;
- both accepted remedies pass: a re-export in method/__init__.py, and a
  public rename — the latter exercising the class-count allowance for the
  spec-declared extra classes (the second ICRA trap: without the
  allowance, renaming public converts the private-name failure into a
  count failure);
- declared symbol missing entirely fails naming it;
- declared symbol defined in two files fails with the ownership-collision
  error naming both sites (the maintainer's 2026-07-16 fail-loud decision, no
  silent dedup);
- symbols owned by a producer that runs AFTER the architecture coder
  (the pluggable function, method helpers in method/method.py) are
  deferred at 2.b, not failed;
- adjacent-good: active-learning and knowledge-distillation specs without
  symbol fields validate identically to a spec with no try_it_out at all
  (no behavior change for committed families).

Adversarial-review round (2026-07-17) additions:

- phantom re-exports do not resolve (an __init__ import only counts when
  the source module actually binds the name);
- model promises never defer, even on a name coincidence with a
  later-producer manifest symbol;
- the count allowance is assignment-based, so a symbol naming a manifest
  role class frees no slot for an undeclared stray;
- attribute mutation is not a definition site (no spurious collisions),
  while a genuine top-level rebind still collides;
- the Stage 2.d finalizer is bridge-aware end to end: both remedies
  survive the REAL finalize (and re-finalize), deferred symbols are
  re-checked there with method-coder attribution, and the same count
  allowance applies.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from scripts.validate_architecture_coder_output import validate
from tests.test_validator_manifest_driven import (
    _MP_TRAINING,
    _MP_TWO_CLASS_MODEL,
    _make_spec,
    _write_arch_contract,
    _write_minimal_paper_map,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------


_PRIVATE_CLASSIFIERS = """

class _AvoidanceClassifier:
    def predict(self, features):
        return 0


class _HCAMClassifier:
    def predict(self, features):
        return 1
"""

_PUBLIC_CLASSIFIERS = _PRIVATE_CLASSIFIERS.replace("_AvoidanceClassifier", "AvoidanceClassifier").replace("_HCAMClassifier", "HCAMClassifier")


def _mp_run(tmp_path, model_src: str, training_src: str = _MP_TRAINING) -> Path:
    run_dir = tmp_path / "run"
    method_dir = run_dir / "method"
    pipeline_dir = run_dir / ".pipeline"
    method_dir.mkdir(parents=True)
    (method_dir / "model.py").write_text(model_src)
    (method_dir / "training.py").write_text(training_src)
    _write_arch_contract(
        pipeline_dir,
        paradigm_id="motion_planning",
        training_function="precompute_motion_primitives",
    )
    _write_minimal_paper_map(pipeline_dir)
    return run_dir


def _mp_spec_with_provides(entries: list[dict]) -> dict:
    spec = _make_spec(paradigm_id="motion_planning")
    spec["try_it_out"] = {
        "definition": "Try the avoidance-gated planner on a synthetic scene.",
        "user_provides": [],
        "system_provides": entries,
    }
    return spec


def _classifier_entries() -> list[dict]:
    """The ICRA shape: two model promises, each with a declared symbol."""
    return [
        {
            "name": "AC classifier (avoidance gating)",
            "description": "Gates candidate trajectories on avoidance risk.",
            "type": "model",
            "symbol": "AvoidanceClassifier",
        },
        {
            "name": "HCAM classifier",
            "description": "Scores human-centered attention maps.",
            "type": "model",
            "symbol": "HCAMClassifier",
        },
    ]


# ---------------------------------------------------------------------------
# Known-bad: the ICRA shape (private definitions, no re-export)
# ---------------------------------------------------------------------------


def test_icra_shape_private_classes_fail_with_remedy_naming_errors(tmp_path):
    """Spec declares two symbols; the package defines them underscore-private
    with no re-export. 2.b must fail once per symbol, name the private
    variant, and name both accepted remedies (rename public, or add the
    re-export)."""
    run_dir = _mp_run(tmp_path, _MP_TWO_CLASS_MODEL + _PRIVATE_CLASSIFIERS)
    spec = _mp_spec_with_provides(_classifier_entries())

    errors = validate(spec, run_dir, REPO_ROOT)
    bridge_errors = [e for e in errors if "promises the importable symbol" in e]
    assert len(bridge_errors) == 2, errors
    for symbol, private in (
        ("AvoidanceClassifier", "_AvoidanceClassifier"),
        ("HCAMClassifier", "_HCAMClassifier"),
    ):
        (err,) = [e for e in bridge_errors if f"`{symbol}`" in e]
        assert f"`{private}`" in err, err
        assert "rename the component public" in err, err
        assert "method/__init__.py" in err, err
    # The private classes stay out of the public class count — no count error.
    assert not any("top-level public class(es)" in e for e in errors), errors


def test_declared_symbol_missing_entirely_fails_naming_it(tmp_path):
    """A declared symbol nothing in the package defines (no private variant
    either) fails with the same remedy-naming error, naming the symbol."""
    run_dir = _mp_run(tmp_path, _MP_TWO_CLASS_MODEL)
    spec = _mp_spec_with_provides([
        {
            "name": "Phantom classifier",
            "description": "Promised but never generated.",
            "type": "model",
            "symbol": "PhantomClassifier",
        },
    ])

    errors = validate(spec, run_dir, REPO_ROOT)
    (err,) = [e for e in errors if "promises the importable symbol" in e]
    assert "`PhantomClassifier`" in err, err
    assert "rename the component public" in err, err
    # Third attribution route: when the package genuinely has no such
    # artifact, the spec's symbol itself may be wrong — the error must point
    # the halt judge at the upstream (Stage 1) classification instead of
    # inviting another coder dispatch (the fedavg oscillation shape).
    assert "upstream (Stage 1)" in err, err


def test_declared_symbol_defined_in_two_files_is_ownership_collision(tmp_path):
    """Two files defining the same declared symbol fail loud with an
    ownership-collision error naming both definition sites — never a silent
    dedup (maintainer decision 2026-07-16)."""
    run_dir = _mp_run(
        tmp_path,
        _MP_TWO_CLASS_MODEL + _PUBLIC_CLASSIFIERS,
        _MP_TRAINING + (
            "\n\nclass AvoidanceClassifier:\n"
            "    def predict(self, features):\n"
            "        return 0\n"
        ),
    )
    spec = _mp_spec_with_provides(_classifier_entries())

    errors = validate(spec, run_dir, REPO_ROOT)
    (collision,) = [e for e in errors if "ambiguous ownership" in e]
    assert "`AvoidanceClassifier`" in collision, collision
    assert "method/model.py" in collision, collision
    assert "method/training.py" in collision, collision
    # The collision replaces the unresolved-symbol error for that symbol; the
    # cleanly-defined second symbol raises no error at all.
    assert not any("promises the importable symbol" in e for e in errors), errors


# ---------------------------------------------------------------------------
# Known-good: both accepted remedies
# ---------------------------------------------------------------------------


def test_icra_shape_reexport_remedy_passes(tmp_path):
    """Same fixture with the re-export remedy: the classes stay private in
    model.py and method/__init__.py re-exports them under the declared
    public names. 2.b passes clean."""
    run_dir = _mp_run(tmp_path, _MP_TWO_CLASS_MODEL + _PRIVATE_CLASSIFIERS)
    (run_dir / "method" / "__init__.py").write_text(
        "from .model import _AvoidanceClassifier as AvoidanceClassifier\n"
        "from .model import _HCAMClassifier as HCAMClassifier\n"
        '\n__all__ = ["AvoidanceClassifier", "HCAMClassifier"]\n'
    )
    spec = _mp_spec_with_provides(_classifier_entries())

    errors = validate(spec, run_dir, REPO_ROOT)
    assert errors == [], errors


def test_icra_shape_rename_public_remedy_passes_with_count_allowance(tmp_path):
    """Same fixture with the rename-public remedy: model.py now carries FOUR
    public classes against the motion-planning manifest's declared two. The
    class-count check must admit the two extras because spec-declared bridge
    symbols resolve to them — the second ICRA trap (renaming public used to
    convert the private-name failure into a count failure)."""
    run_dir = _mp_run(tmp_path, _MP_TWO_CLASS_MODEL + _PUBLIC_CLASSIFIERS)
    spec = _mp_spec_with_provides(_classifier_entries())

    errors = validate(spec, run_dir, REPO_ROOT)
    assert errors == [], errors


def test_bridge_symbol_naming_a_manifest_role_class_is_not_an_extra(tmp_path):
    """A declared symbol may resolve to one of the manifest's own declared
    classes (the analyzer promises the main model). That must not inflate
    the expected count or fail anything."""
    run_dir = _mp_run(tmp_path, _MP_TWO_CLASS_MODEL)
    spec = _mp_spec_with_provides([
        {
            "name": "System dynamics model",
            "description": "The unicycle dynamics used by the planner.",
            "type": "model",
            "symbol": "UnicycleDynamics",
        },
    ])

    errors = validate(spec, run_dir, REPO_ROOT)
    assert errors == [], errors


# ---------------------------------------------------------------------------
# Typo safety: the count allowance admits ONLY spec-declared extras
# ---------------------------------------------------------------------------


def test_undeclared_extra_class_still_fails_the_count_check(tmp_path):
    """An extra public class no spec symbol declares still fails the count
    check, and the error names the undeclared class(es)."""
    run_dir = _mp_run(
        tmp_path,
        _MP_TWO_CLASS_MODEL + _PUBLIC_CLASSIFIERS + (
            "\n\nclass StrayHelper:\n"
            "    def helper(self):\n"
            "        return None\n"
        ),
    )
    spec = _mp_spec_with_provides(_classifier_entries())

    errors = validate(spec, run_dir, REPO_ROOT)
    (count_err,) = [e for e in errors if "top-level public class(es)" in e]
    assert "StrayHelper" in count_err, count_err
    assert "not declared by any spec symbol" in count_err, count_err


# ---------------------------------------------------------------------------
# Deferral: symbols owned by producers that run after the architecture coder
# ---------------------------------------------------------------------------


def test_symbols_owned_by_later_producers_are_deferred_at_2b(tmp_path):
    """method/method.py does not exist when 2.b runs (the method coder runs
    after this gate). A declared symbol the manifest assigns to that file —
    the pluggable function by concrete name, or any code promise the
    open-ended method-helpers bucket may own — must be deferred, not failed:
    failing it here would mis-attribute a not-yet-generated file to the
    architecture coder. Model promises are never deferred (the tests above
    pin that they fail)."""
    run_dir = _mp_run(tmp_path, _MP_TWO_CLASS_MODEL)
    spec = _mp_spec_with_provides([
        {
            "name": "Planner algorithm",
            "description": "The pluggable planning function.",
            "type": "code",
            # _make_spec's pluggable_component.name — the manifest's concrete
            # method/method.py public symbol after build-plan substitution.
            "symbol": "compute_distillation_loss",
        },
        {
            "name": "Trajectory scoring helper",
            "description": "A method helper the method coder will write.",
            "type": "code",
            "symbol": "score_trajectory",
        },
    ])

    errors = validate(spec, run_dir, REPO_ROOT)
    assert errors == [], errors


def test_code_symbol_resolving_in_present_files_passes_without_deferral(tmp_path):
    """A code promise the architecture coder already satisfied (the training
    entry point) resolves directly — deferral is only the fallback for
    unresolved symbols."""
    run_dir = _mp_run(tmp_path, _MP_TWO_CLASS_MODEL)
    spec = _mp_spec_with_provides([
        {
            "name": "Motion-primitive precomputation",
            "description": "Precomputes the primitive library.",
            "type": "code",
            "symbol": "precompute_motion_primitives",
        },
    ])

    errors = validate(spec, run_dir, REPO_ROOT)
    assert errors == [], errors


# ---------------------------------------------------------------------------
# Adjacent-good: committed families without symbols see no behavior change
# ---------------------------------------------------------------------------


_AL_MODEL = """
import torch.nn as nn

class MLPClassifier(nn.Module):
    def __init__(self, input_dim, n_classes):
        super().__init__()
        self.fc = nn.Linear(input_dim, n_classes)
    def forward(self, x):
        return self.fc(x)
"""

_AL_TRAINING = """
import torch
from .model import MLPClassifier

def build_model(input_dim, n_classes):
    return MLPClassifier(input_dim, n_classes)

def train_from_scratch(x_train, y_train, *, learning_rate=1e-3, num_epochs=10, batch_size=32, seed=0):
    torch.manual_seed(seed)
    model = MLPClassifier(x_train.shape[1], int(y_train.max()) + 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    return model
"""

_KD_MODEL = """
import torch.nn as nn

class StudentNet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.fc = nn.Linear(10, num_classes)
    def forward(self, x):
        return self.fc(x)

class TeacherNet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.fc = nn.Linear(10, num_classes)
    def forward(self, x):
        return self.fc(x)
"""

_KD_TRAINING = """
import torch
from .model import StudentNet, TeacherNet

def build_student(input_dim, n_classes, **kw):
    return StudentNet(n_classes)

def build_teacher(input_dim, n_classes, **kw):
    return TeacherNet(n_classes)

def train_with_distillation(student, teacher, x_train, y_train, *,
                             distillation_loss_fn, learning_rate=1e-3,
                             num_epochs=10, batch_size=32, seed=0, **kw):
    torch.manual_seed(seed)
    optimizer = torch.optim.Adam(student.parameters(), lr=learning_rate)
    return student
"""


def _run_with_sources(tmp_path, *, paradigm_id: str, model_src: str, training_src: str) -> Path:
    run_dir = tmp_path / "run"
    method_dir = run_dir / "method"
    pipeline_dir = run_dir / ".pipeline"
    method_dir.mkdir(parents=True)
    (method_dir / "model.py").write_text(model_src)
    (method_dir / "training.py").write_text(training_src)
    _write_arch_contract(pipeline_dir, paradigm_id=paradigm_id)
    _write_minimal_paper_map(pipeline_dir)
    return run_dir


def test_adjacent_good_specs_without_symbols_validate_unchanged(tmp_path):
    """An active-learning and a knowledge-distillation spec whose
    system_provides entries carry NO symbol field validate exactly like a
    spec with no try_it_out block at all: the bridge is conservative by
    construction, and committed families that never use it see identical
    validation."""
    for paradigm_id, model_src, training_src in (
        ("active_learning", _AL_MODEL, _AL_TRAINING),
        ("knowledge_distillation", _KD_MODEL, _KD_TRAINING),
    ):
        run_dir = _run_with_sources(
            tmp_path / paradigm_id,
            paradigm_id=paradigm_id,
            model_src=model_src,
            training_src=training_src,
        )
        legacy_spec = _make_spec(paradigm_id=paradigm_id)
        symbol_less_spec = _make_spec(paradigm_id=paradigm_id)
        symbol_less_spec["try_it_out"] = {
            "definition": "Try the method on a small demo pool.",
            "user_provides": [],
            "system_provides": [
                {"name": "Algorithm implementation", "description": "x", "type": "code"},
                {"name": "Training loop", "description": "x", "type": "code"},
            ],
        }

        legacy_errors = validate(legacy_spec, run_dir, REPO_ROOT)
        symbol_less_errors = validate(symbol_less_spec, run_dir, REPO_ROOT)
        assert legacy_errors == [], (paradigm_id, legacy_errors)
        assert symbol_less_errors == legacy_errors, (paradigm_id, symbol_less_errors)


# ---------------------------------------------------------------------------
# Adversarial round: phantom re-exports, deferral scope, allowance leak,
# definition-site precision
# ---------------------------------------------------------------------------


def test_phantom_reexport_does_not_resolve_the_bridge(tmp_path):
    """An __init__.py import only counts as a re-export when the SOURCE
    module actually binds the imported name. A phantom
    `from .model import AvoidanceClassifier` (model.py defines only the
    private variant) is dead at import time — with or without a try/except
    wrapper — and must not satisfy the bridge."""
    run_dir = _mp_run(tmp_path, _MP_TWO_CLASS_MODEL + _PRIVATE_CLASSIFIERS)
    (run_dir / "method" / "__init__.py").write_text(
        "from .model import AvoidanceClassifier\n"
        "try:\n"
        "    from .model import HCAMClassifier\n"
        "except ImportError:\n"
        "    HCAMClassifier = None\n"
    )
    spec = _mp_spec_with_provides(_classifier_entries())

    errors = validate(spec, run_dir, REPO_ROOT)
    bridge_errors = [e for e in errors if "promises the importable symbol" in e]
    assert any("`AvoidanceClassifier`" in e for e in bridge_errors), errors
    # The try/except-wrapped variant fails too: the phantom import verifies
    # against the source module's bindings, and the `HCAMClassifier = None`
    # fallback sits inside the except block, not the module's top-level body
    # — a None-valued fallback is a phantom promise either way, so the
    # conservative scan rejecting it is the intended behavior.
    assert any("`HCAMClassifier`" in e for e in bridge_errors), errors
    assert len(bridge_errors) == 2, errors


def test_phantom_reexport_without_fallback_fails_both_symbols(tmp_path):
    """Both phantom imports, no fallback binding anywhere: both symbols
    stay unresolved."""
    run_dir = _mp_run(tmp_path, _MP_TWO_CLASS_MODEL + _PRIVATE_CLASSIFIERS)
    (run_dir / "method" / "__init__.py").write_text(
        "from .model import AvoidanceClassifier\n"
        "from .model import HCAMClassifier\n"
    )
    spec = _mp_spec_with_provides(_classifier_entries())

    errors = validate(spec, run_dir, REPO_ROOT)
    bridge_errors = [e for e in errors if "promises the importable symbol" in e]
    assert len(bridge_errors) == 2, errors


def test_model_promise_never_defers_even_on_pluggable_name_match(tmp_path):
    """Deferral is verified by promise type: a `model` promise is
    architecture surface no matter what its symbol is named, so a model
    promise whose symbol coincides with the manifest's pluggable-function
    name still fails at 2.b when unresolved."""
    run_dir = _mp_run(tmp_path, _MP_TWO_CLASS_MODEL)
    spec = _mp_spec_with_provides([
        {
            "name": "Learned loss model",
            "description": "Promised as a model, named like the pluggable.",
            "type": "model",
            "symbol": "compute_distillation_loss",
        },
    ])

    errors = validate(spec, run_dir, REPO_ROOT)
    (err,) = [e for e in errors if "promises the importable symbol" in e]
    assert "`compute_distillation_loss`" in err, err


def test_role_class_symbol_frees_no_allowance_slot(tmp_path):
    """The allowance-leak known-bad: a declared symbol resolving to one of
    the manifest's own role classes (UnicycleDynamics) must not free a count
    slot for an undeclared stray class."""
    run_dir = _mp_run(
        tmp_path,
        _MP_TWO_CLASS_MODEL + (
            "\n\nclass StrayHelper:\n"
            "    def helper(self):\n"
            "        return None\n"
        ),
    )
    spec = _mp_spec_with_provides([
        {
            "name": "System dynamics model",
            "description": "The unicycle dynamics used by the planner.",
            "type": "model",
            "symbol": "UnicycleDynamics",
        },
    ])

    errors = validate(spec, run_dir, REPO_ROOT)
    (count_err,) = [e for e in errors if "top-level public class(es)" in e]
    assert "StrayHelper" in count_err, count_err


def test_attribute_mutation_is_not_a_definition_site(tmp_path):
    """`X.THRESHOLD = 0.5` at the top level of training.py mutates an
    imported object — it defines nothing, so it must not register
    training.py as a second definition site (no spurious ownership
    collision)."""
    run_dir = _mp_run(
        tmp_path,
        _MP_TWO_CLASS_MODEL + _PUBLIC_CLASSIFIERS,
        _MP_TRAINING + (
            "\nfrom .model import AvoidanceClassifier\n"
            "AvoidanceClassifier.THRESHOLD = 0.5\n"
        ),
    )
    spec = _mp_spec_with_provides(_classifier_entries())

    errors = validate(spec, run_dir, REPO_ROOT)
    assert errors == [], errors


def test_genuine_top_level_rebind_is_still_a_collision(tmp_path):
    """The complement: a real top-level rebind of the declared symbol in a
    second file remains an ownership collision."""
    run_dir = _mp_run(
        tmp_path,
        _MP_TWO_CLASS_MODEL + _PUBLIC_CLASSIFIERS,
        _MP_TRAINING + '\nAvoidanceClassifier = "rebound"\n',
    )
    spec = _mp_spec_with_provides(_classifier_entries())

    errors = validate(spec, run_dir, REPO_ROOT)
    (collision,) = [e for e in errors if "ambiguous ownership" in e]
    assert "method/model.py" in collision and "method/training.py" in collision, collision


# ---------------------------------------------------------------------------
# Stage 2.d finalizer integration: both remedies survive the REAL finalize
# (and the Stage 5 re-finalization, which calls the same code path), the
# count allowance is ported, and deferred symbols are re-checked.
# ---------------------------------------------------------------------------

import json

from scripts.finalize_package_init import finalize

_MP_DATA = """
def load_environment(name="two_rooms_simple", *, seed=0):
    return object()

def load_problem(name="two_rooms_simple", *, seed=0):
    return None, None, object()
"""

_MP_METHOD = """
def compute_distillation_loss(dynamics, collision_model, *, seed=0):
    return []
"""


def _mp_finalize_run(
    tmp_path, model_src: str, spec: dict, *,
    training_src: str = _MP_TRAINING, method_src: str = _MP_METHOD,
) -> tuple:
    run_dir = tmp_path / "run"
    method_dir = run_dir / "method"
    pipeline_dir = run_dir / ".pipeline"
    method_dir.mkdir(parents=True)
    pipeline_dir.mkdir(parents=True)
    (method_dir / "model.py").write_text(model_src)
    (method_dir / "training.py").write_text(training_src)
    (method_dir / "method.py").write_text(method_src)
    (method_dir / "data.py").write_text(_MP_DATA)
    spec_path = pipeline_dir / "method_spec.json"
    spec_path.write_text(json.dumps(spec, indent=2))
    return run_dir, spec_path


def test_finalize_emits_alias_reexports_and_refinalize_keeps_them(tmp_path):
    """The re-export remedy survives the REAL finalizer: with the promised
    classes underscore-private, finalize() itself emits the deterministic
    alias re-exports in the regenerated __init__.py — and a second finalize
    pass (the Stage 5 re-finalization shape) re-derives instead of erasing
    them."""
    spec = _mp_spec_with_provides(_classifier_entries())
    run_dir, spec_path = _mp_finalize_run(
        tmp_path, _MP_TWO_CLASS_MODEL + _PRIVATE_CLASSIFIERS, spec)

    for round_name in ("initial finalize", "stage-5 style re-finalize"):
        rc = finalize(spec_path, run_dir)
        assert rc == 0, round_name
        init = (run_dir / "method" / "__init__.py").read_text()
        assert (
            "from .model import _AvoidanceClassifier as AvoidanceClassifier"
            in init
        ), (round_name, init)
        assert (
            "from .model import _HCAMClassifier as HCAMClassifier" in init
        ), (round_name, init)
        assert '"AvoidanceClassifier",' in init, (round_name, init)
        assert '"HCAMClassifier",' in init, (round_name, init)


def test_finalize_rename_public_remedy_passes_count_and_exports(tmp_path):
    """The rename-public remedy survives the REAL finalizer: four public
    classes against the manifest's two pass finalize's own count check via
    the ported allowance, and both promised classes are re-exported."""
    spec = _mp_spec_with_provides(_classifier_entries())
    run_dir, spec_path = _mp_finalize_run(
        tmp_path, _MP_TWO_CLASS_MODEL + _PUBLIC_CLASSIFIERS, spec)

    rc = finalize(spec_path, run_dir)
    assert rc == 0
    init = (run_dir / "method" / "__init__.py").read_text()
    assert "AvoidanceClassifier" in init, init
    assert "HCAMClassifier" in init, init
    assert '"AvoidanceClassifier",' in init, init
    assert '"HCAMClassifier",' in init, init


def test_finalize_recheck_fails_deferred_symbol_with_method_coder_attribution(
    tmp_path, capsys,
):
    """Deferral is not terminal: a code promise 2.b deferred to the method
    coder is re-checked at finalize time, and if the complete package still
    does not define it, finalize fails naming the symbol, the method coder,
    and the upstream (Stage 1) route for a wrong spec symbol."""
    spec = _mp_spec_with_provides([
        {
            "name": "Trajectory scoring helper",
            "description": "Promised, deferred at 2.b, never written.",
            "type": "code",
            "symbol": "score_trajectory",
        },
    ])
    run_dir, spec_path = _mp_finalize_run(tmp_path, _MP_TWO_CLASS_MODEL, spec)

    rc = finalize(spec_path, run_dir)
    err = capsys.readouterr().err
    assert rc == 2
    assert "`score_trajectory`" in err, err
    assert "method_coder" in err, err
    assert "upstream (Stage 1)" in err, err
    assert not (run_dir / "method" / "__init__.py").is_file()


def test_finalize_count_check_keeps_typo_safety_and_leak_freedom(tmp_path, capsys):
    """The ported count allowance keeps both safety properties: an
    undeclared stray fails even when declared extras are present, and a
    symbol naming a manifest role class frees no slot."""
    # Undeclared stray next to two declared (public) extras.
    spec = _mp_spec_with_provides(_classifier_entries())
    run_dir, spec_path = _mp_finalize_run(
        tmp_path / "stray",
        _MP_TWO_CLASS_MODEL + _PUBLIC_CLASSIFIERS + (
            "\n\nclass StrayHelper:\n"
            "    def helper(self):\n"
            "        return None\n"
        ),
        spec,
    )
    rc = finalize(spec_path, run_dir)
    err = capsys.readouterr().err
    assert rc == 2
    assert "StrayHelper" in err, err

    # The leak shape: role-class symbol + stray.
    spec = _mp_spec_with_provides([
        {
            "name": "System dynamics model",
            "description": "x",
            "type": "model",
            "symbol": "UnicycleDynamics",
        },
    ])
    run_dir, spec_path = _mp_finalize_run(
        tmp_path / "leak",
        _MP_TWO_CLASS_MODEL + (
            "\n\nclass StrayHelper:\n"
            "    def helper(self):\n"
            "        return None\n"
        ),
        spec,
    )
    rc = finalize(spec_path, run_dir)
    err = capsys.readouterr().err
    assert rc == 2
    assert "StrayHelper" in err, err


def test_finalize_reexports_promised_public_extra_function(tmp_path):
    """A promised symbol defined public in a module the standard collection
    skips (an extra training.py function the manifest never declared) is
    added to the re-export surface instead of silently dropped."""
    spec = _mp_spec_with_provides([
        {
            "name": "Candidate scoring",
            "description": "Extra public training-side utility.",
            "type": "code",
            "symbol": "score_candidates",
        },
    ])
    run_dir, spec_path = _mp_finalize_run(
        tmp_path,
        _MP_TWO_CLASS_MODEL,
        spec,
        training_src=_MP_TRAINING + (
            "\n\ndef score_candidates(candidates, *, seed=0):\n"
            "    return list(candidates)\n"
        ),
    )

    rc = finalize(spec_path, run_dir)
    assert rc == 0
    init = (run_dir / "method" / "__init__.py").read_text()
    training_lines = [
        line for line in init.splitlines()
        if line.startswith("from .training import")
    ]
    assert training_lines and "score_candidates" in training_lines[0], init
    assert '"score_candidates",' in init, init


# ---------------------------------------------------------------------------
# Promise kind-awareness (plan-of-record item 10, schema v1.8.0). Two
# concrete cases shaped the abstraction: detr's KD roll promised a function
# that shipped under a different name (same kind, producer-fixable one alias
# away) and ADAM promised class-shaped optimizer symbols (`Adam`, `AdaMax`)
# for a method honestly delivered as a pure-function API (wrong kind — no
# rename can fix it; the 2.d halt correctly classified it upstream but only
# after full generation). With `symbol_kind` declared, the ADAM shape fails
# at 2.b with judge routing instead: a class promise never defers to the
# open-ended function bucket, and a resolved definition's AST kind must not
# contradict the declared kind.
# ---------------------------------------------------------------------------


_PUBLIC_GATE_FUNCTION = """

def avoidance_gate(candidates, features):
    return candidates
"""

_ONE_PUBLIC_CLASSIFIER = """

class AvoidanceClassifier:
    def predict(self, features):
        return 0
"""


def test_kind_mismatch_class_promised_function_defined_fails_2b(tmp_path):
    """A promise declared `class` resolving to a public FUNCTION definition
    fails 2.b with the kind-mismatch error: no rename or re-export can fix a
    kind mismatch, and the error routes the judge at the upstream (Stage 1)
    classification when the delivered shape is the honest one."""
    run_dir = _mp_run(tmp_path, _MP_TWO_CLASS_MODEL + _PUBLIC_GATE_FUNCTION)
    spec = _mp_spec_with_provides([
        {
            "name": "Avoidance gate",
            "description": "Gates candidate trajectories.",
            "type": "code",
            "symbol": "avoidance_gate",
            "symbol_kind": "class",
        },
    ])

    errors = validate(spec, run_dir, REPO_ROOT)
    (err,) = [e for e in errors if "promised KIND of surface" in e]
    assert "`avoidance_gate`" in err, err
    assert "as a class" in err, err
    assert "defines it as a function" in err, err
    assert "upstream (Stage 1)" in err, err


def test_kind_mismatch_function_promised_class_defined_fails_2b_and_frees_no_slot(
    tmp_path,
):
    """The symmetric mismatch: a promise declared `function` resolving to a
    public CLASS fails the kind check — and the count allowance excludes
    function-kind symbols, so the extra class also stays an undeclared stray
    for the count check (a function promise's name must not launder a stray
    class past it)."""
    run_dir = _mp_run(tmp_path, _MP_TWO_CLASS_MODEL + _ONE_PUBLIC_CLASSIFIER)
    spec = _mp_spec_with_provides([
        {
            "name": "AC scoring routine",
            "description": "Scores avoidance candidates.",
            "type": "code",
            "symbol": "AvoidanceClassifier",
            "symbol_kind": "function",
        },
    ])

    errors = validate(spec, run_dir, REPO_ROOT)
    (mismatch,) = [e for e in errors if "promised KIND of surface" in e]
    assert "as a function" in mismatch, mismatch
    assert "defines it as a class" in mismatch, mismatch
    assert any("top-level public class(es)" in e for e in errors), errors


def test_kind_match_class_promise_passes_with_count_allowance(tmp_path):
    """The matching declaration changes nothing: a `class` promise resolving
    to a public class passes, including the count allowance for the extra
    class (the rename-public remedy with the kind declared)."""
    run_dir = _mp_run(tmp_path, _MP_TWO_CLASS_MODEL + _ONE_PUBLIC_CLASSIFIER)
    spec = _mp_spec_with_provides([
        {
            "name": "AC classifier",
            "description": "Gates candidate trajectories.",
            "type": "model",
            "symbol": "AvoidanceClassifier",
            "symbol_kind": "class",
        },
    ])

    errors = validate(spec, run_dir, REPO_ROOT)
    assert errors == [], errors


@pytest.mark.parametrize("declared_kind", ["class", "function"])
def test_assignment_binding_satisfies_any_declared_kind(tmp_path, declared_kind):
    """An assignment binding can alias anything, so it satisfies either
    declared kind — the kind check only fails on a DEFINITE class-vs-function
    mismatch (conservative by construction, same posture as the rest of the
    bridge)."""
    run_dir = _mp_run(
        tmp_path,
        _MP_TWO_CLASS_MODEL,
        _MP_TRAINING + "\n\nscore_trajectory = len\n",
    )
    spec = _mp_spec_with_provides([
        {
            "name": "Trajectory scorer",
            "description": "Scores trajectories.",
            "type": "code",
            "symbol": "score_trajectory",
            "symbol_kind": declared_kind,
        },
    ])

    errors = validate(spec, run_dir, REPO_ROOT)
    assert errors == [], errors


def test_adam_shape_class_promise_fails_at_2b_instead_of_deferring(tmp_path):
    """The ADAM genus, moved to its earliest gate: a `class` promise nothing
    on the architecture surface defines must NOT defer to the open-ended
    method-helpers function bucket (a function-only producer cannot deliver
    a class). 2.b fails it with the kind note pointing the judge at the
    upstream (Stage 1) case — instead of the late 2.d hard halt ADAM
    actually hit."""
    run_dir = _mp_run(tmp_path, _MP_TWO_CLASS_MODEL)
    spec = _mp_spec_with_provides([
        {
            "name": "Adam optimizer",
            "description": "The optimizer as a researcher-facing object.",
            "type": "code",
            "symbol": "AdamOptimizer",
            "symbol_kind": "class",
        },
    ])

    errors = validate(spec, run_dir, REPO_ROOT)
    (err,) = [e for e in errors if "promises the importable symbol" in e]
    assert "`AdamOptimizer`" in err, err
    assert "class surface" in err, err
    assert "pure-function API" in err, err
    assert "upstream (Stage 1)" in err, err


def test_function_kind_promise_still_defers_to_function_bucket(tmp_path):
    """The kind-aware deferral keeps today's behavior for function promises:
    a `function` promise the method-helpers bucket may own is deferred at
    2.b exactly as an undeclared-kind one is."""
    run_dir = _mp_run(tmp_path, _MP_TWO_CLASS_MODEL)
    spec = _mp_spec_with_provides([
        {
            "name": "Trajectory scoring helper",
            "description": "A method helper the method coder will write.",
            "type": "code",
            "symbol": "score_trajectory",
            "symbol_kind": "function",
        },
    ])

    errors = validate(spec, run_dir, REPO_ROOT)
    assert errors == [], errors


def test_class_promise_does_not_defer_on_contradicting_manifest_kind(tmp_path):
    """A concrete later-producer name match only defers when the manifest
    entry's own `kind` does not contradict the promise: the pluggable
    function's manifest kind is `function`, so promising its name as a
    `class` fails at 2.b instead of deferring into a guaranteed 2.d death."""
    run_dir = _mp_run(tmp_path, _MP_TWO_CLASS_MODEL)
    spec = _mp_spec_with_provides([
        {
            "name": "Planner algorithm",
            "description": "The pluggable planning entry point.",
            "type": "code",
            "symbol": "compute_distillation_loss",
            "symbol_kind": "class",
        },
    ])

    errors = validate(spec, run_dir, REPO_ROOT)
    (err,) = [e for e in errors if "promises the importable symbol" in e]
    assert "`compute_distillation_loss`" in err, err
    assert "class surface" in err, err


def test_reexport_resolution_stays_conservative_about_kind(tmp_path):
    """A promise resolved through a method/__init__.py re-export passes
    without a kind check — an alias may bind anything, and failing the
    architecture coder on an unverifiable kind would be a mis-attribution.
    Pinned as designed conservatism, not an oversight."""
    run_dir = _mp_run(tmp_path, _MP_TWO_CLASS_MODEL + _PRIVATE_CLASSIFIERS)
    (run_dir / "method" / "__init__.py").write_text(
        "from .model import _AvoidanceClassifier as AvoidanceClassifier\n"
        '\n__all__ = ["AvoidanceClassifier"]\n'
    )
    spec = _mp_spec_with_provides([
        {
            "name": "AC classifier",
            "description": "Gates candidate trajectories.",
            "type": "model",
            "symbol": "AvoidanceClassifier",
            "symbol_kind": "function",
        },
    ])

    errors = validate(spec, run_dir, REPO_ROOT)
    assert errors == [], errors


# ---------------------------------------------------------------------------
# Kind-awareness at the Stage 2.d finalizer: the re-check point sees the
# COMPLETE package, so kind mismatches in later-producer files surface here,
# the alias remedy refuses a wrong-kind private definition, and an absent
# class promise is attributed to the architecture surface.
# ---------------------------------------------------------------------------


def test_finalize_kind_mismatch_on_method_coder_function_fails(tmp_path, capsys):
    """A `class` promise 2.b deferred (concrete manifest name owned by the
    method coder is exempt from the bucket rule only when kinds agree;
    here the definition exists) that the complete package delivers as a
    FUNCTION fails finalize with the kind-mismatch error."""
    spec = _mp_spec_with_provides([
        {
            "name": "Trajectory scoring helper",
            "description": "Promised as a class, delivered as a function.",
            "type": "code",
            "symbol": "score_trajectory",
            "symbol_kind": "class",
        },
    ])
    run_dir, spec_path = _mp_finalize_run(
        tmp_path, _MP_TWO_CLASS_MODEL, spec,
        method_src=_MP_METHOD + "\n\ndef score_trajectory(traj):\n    return 0.0\n",
    )

    rc = finalize(spec_path, run_dir)
    err = capsys.readouterr().err
    assert rc == 2
    assert "promised KIND of surface" in err, err
    assert "`score_trajectory`" in err, err
    assert "as a class" in err, err
    assert "as a function" in err, err
    assert not (run_dir / "method" / "__init__.py").is_file()


def test_finalize_refuses_alias_remedy_for_wrong_kind_private_variant(
    tmp_path, capsys,
):
    """The alias remedy re-exports a private definition under the promised
    name — a wrong-kind private definition would keep the name and break the
    promise's shape, so finalize refuses to emit the alias and fails with
    the kind-mismatch error naming the private variant."""
    spec = _mp_spec_with_provides([
        {
            "name": "Trajectory scoring helper",
            "description": "Promised as a class, defined private as a function.",
            "type": "code",
            "symbol": "score_trajectory",
            "symbol_kind": "class",
        },
    ])
    run_dir, spec_path = _mp_finalize_run(
        tmp_path, _MP_TWO_CLASS_MODEL, spec,
        method_src=_MP_METHOD + "\n\ndef _score_trajectory(traj):\n    return 0.0\n",
    )

    rc = finalize(spec_path, run_dir)
    err = capsys.readouterr().err
    assert rc == 2
    assert "promised KIND of surface" in err, err
    assert "`_score_trajectory`" in err, err
    assert not (run_dir / "method" / "__init__.py").is_file()


def test_finalize_absent_class_promise_attributed_to_architecture(tmp_path, capsys):
    """An absent symbol promised as a `class` is an architecture surface
    (same axis as the existing `type: model` rule), so the finalize failure
    attributes the architecture coder and carries the kind note for the
    judge."""
    spec = _mp_spec_with_provides([
        {
            "name": "Adam optimizer",
            "description": "Promised as a class, never delivered.",
            "type": "code",
            "symbol": "AdamOptimizer",
            "symbol_kind": "class",
        },
    ])
    run_dir, spec_path = _mp_finalize_run(tmp_path, _MP_TWO_CLASS_MODEL, spec)

    rc = finalize(spec_path, run_dir)
    err = capsys.readouterr().err
    assert rc == 2
    assert "`AdamOptimizer`" in err, err
    assert "architecture_coder" in err, err
    assert "class surface" in err, err


def test_finalize_assignment_binding_satisfies_declared_kind(tmp_path):
    """The finalizer's kind re-check keeps the same conservatism as 2.b: an
    assignment binding satisfies any declared kind, so a deliberate alias
    delivery (`score_trajectory = _impl`) finalizes clean."""
    spec = _mp_spec_with_provides([
        {
            "name": "Trajectory scoring helper",
            "description": "Delivered as a deliberate alias binding.",
            "type": "code",
            "symbol": "score_trajectory",
            "symbol_kind": "class",
        },
    ])
    run_dir, spec_path = _mp_finalize_run(
        tmp_path, _MP_TWO_CLASS_MODEL, spec,
        method_src=_MP_METHOD
        + "\n\ndef _impl(traj):\n    return 0.0\n\nscore_trajectory = _impl\n",
    )

    rc = finalize(spec_path, run_dir)
    assert rc == 0
    init = (run_dir / "method" / "__init__.py").read_text()
    assert "score_trajectory" in init, init
