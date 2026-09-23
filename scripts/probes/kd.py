"""Knowledge-distillation probes (KD-1/KD-2).

KD-1 `teacher-signal-influence`: the distillation loss must respond when the
teacher's outputs change. The bev-distill founding case (M-001's sibling,
C1/quality-score no-op) was a weighting term that collapsed to a constant, so
the "distillation" loss ignored the teacher signal entirely while looking
faithful. No artifact of that run was ever captured, so the fail side is a
synthetic reconstruction in the tests.

KD-2 `temperature-sensitivity`: when the loss exposes a temperature-like
parameter, varying it must change the loss (temperature scaling that does
nothing is the same no-op class). Not applicable (returns None) when no such
parameter exists — the caller decides whether N/A matters.

Both introspect the callable's signature to find teacher/student/temperature
parameters by name pattern (never positional guessing), and go `unprobeable`
when the signature doesn't expose what the probe needs.

Two loss interface shapes exist (the two concrete cases, 2026-07-02):

1. **Tensor-taking** — ``loss(student_logits, teacher_logits, ...)``. The
   original fast path below handles these; it is untouched by the kit.
2. **Model-and-batch-taking** — ``loss(student, teacher, batch, seed, ...)``
   (bev-distill's cross-modal shape). ``student``/``teacher`` are the run's
   real ``nn.Module``s and ``batch`` carries the loader-contract keys.
   ``kd_loss_kit`` builds those FROM the run's own artifacts: models via the
   package's ``build_student``/``build_teacher`` at probe-tiny widths, the
   batch synthesized from ``arch_contract.json``'s
   ``pluggable_component.batch_dict_shape``. Never a fabricated stand-in:
   a missing builder or an undeclared batch key is an unprobeable with the
   missing piece named.
"""

from __future__ import annotations

import inspect
from typing import Callable

import numpy as np

from probes import ProbeVerdict
from probes.catalogs.knowledge_distillation import (
    PROBE_CATALOG as _PROBE_CATALOG,
)
from probes.fixtures import KDLogitsFixture, make_kd_logits_fixture
from probes.trainability import _fill_kwargs
from probes.universal import probe_callable_sensitivity

PROBE_CATALOG = _PROBE_CATALOG
_TEACHER_HINTS = ("teacher",)
_STUDENT_HINTS = ("student",)
_TEMPERATURE_HINTS = ("temperature", "temp", "tau")


def _find_param(fn: Callable, hints: tuple[str, ...]) -> str | None:
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return None
    for name in params:
        lowered = name.lower()
        if any(h in lowered for h in hints):
            # Exact-word-ish guard: 'temp' should not match 'template'.
            if "template" in lowered:
                continue
            return name
    return None


def _first_working_tensors(fn, student_param, teacher_param,
                           fixture=None, seed=0):
    """Generated KD losses come in two shapes: logit distillation (B, C) and
    feature distillation (B, N_queries, D). Try both (v3's rank-aware
    hook_smoke pattern) and return the first (student, teacher) pair the
    callable accepts, or an unprobeable verdict."""
    import torch  # noqa: PLC0415

    fixture = fixture or make_kd_logits_fixture(seed=seed)
    logits_t = torch.tensor(fixture.teacher_logits, dtype=torch.float32)
    logits_s = torch.tensor(fixture.student_logits, dtype=torch.float32)
    g = torch.Generator().manual_seed(seed)
    feats_t = torch.randn(4, 5, 8, generator=g)
    feats_s = feats_t + 0.5 * torch.randn(4, 5, 8, generator=g)

    last_error: Exception | None = None
    for student, teacher in ((logits_s, logits_t), (feats_s, feats_t)):
        try:
            fn(**{student_param: student, teacher_param: teacher})
            return student, teacher
        except Exception as e:  # noqa: BLE001 — generated code fails arbitrarily
            last_error = e
    return ProbeVerdict(
        "KD-1", "unprobeable",
        f"{getattr(fn, '__name__', fn)!r} rejected both logit (B,C) and "
        f"feature (B,N,D) shapes — last error: "
        f"{type(last_error).__name__}: {last_error}")


def probe_teacher_signal_influence(
    fn: Callable,
    fixture: KDLogitsFixture | None = None,
    seed: int = 0,
) -> ProbeVerdict:
    """KD-1: perturbing the teacher tensor must change the loss."""
    try:
        import torch  # noqa: PLC0415
    except ImportError:
        return ProbeVerdict("KD-1", "unprobeable", "torch required")

    teacher_param = _find_param(fn, _TEACHER_HINTS)
    student_param = _find_param(fn, _STUDENT_HINTS)
    if teacher_param is None or student_param is None:
        return ProbeVerdict(
            "KD-1", "unprobeable",
            f"{getattr(fn, '__name__', fn)!r} exposes no teacher/student "
            f"parameters by name — adapt the probe call by hand")

    shaped = _first_working_tensors(fn, student_param, teacher_param,
                                    fixture=fixture, seed=seed)
    if isinstance(shaped, ProbeVerdict):
        return shaped
    student, teacher = shaped
    g = torch.Generator().manual_seed(seed)
    variants = [
        teacher,
        teacher[torch.randperm(teacher.shape[0], generator=g)],  # shuffled
        teacher * 3.0,                                            # sharpened
    ]
    return probe_callable_sensitivity(
        fn,
        base_kwargs={student_param: student, teacher_param: teacher},
        vary_param=teacher_param,
        candidates=variants,
        probe_id="KD-1",
        finding_class="C1",
    )


def probe_temperature_sensitivity(
    fn: Callable,
    fixture: KDLogitsFixture | None = None,
    seed: int = 0,
) -> ProbeVerdict | None:
    """KD-2: varying a temperature-like parameter must change the loss.
    Returns None when the callable exposes no temperature parameter."""
    try:
        import torch  # noqa: PLC0415
    except ImportError:
        return ProbeVerdict("KD-2", "unprobeable", "torch required")

    temp_param = _find_param(fn, _TEMPERATURE_HINTS)
    if temp_param is None:
        return None
    teacher_param = _find_param(fn, _TEACHER_HINTS)
    student_param = _find_param(fn, _STUDENT_HINTS)
    if teacher_param is None or student_param is None:
        return ProbeVerdict(
            "KD-2", "unprobeable",
            f"{getattr(fn, '__name__', fn)!r} has a temperature param but no "
            f"recognizable teacher/student params")

    shaped = _first_working_tensors(fn, student_param, teacher_param,
                                    fixture=fixture, seed=seed)
    if isinstance(shaped, ProbeVerdict):
        return ProbeVerdict("KD-2", "unprobeable", shaped.message)
    student, teacher = shaped
    return probe_callable_sensitivity(
        fn,
        base_kwargs={student_param: student, teacher_param: teacher,
                     temp_param: 1.0},
        vary_param=temp_param,
        candidates=[1.0, 2.0, 4.0, 8.0],
        probe_id="KD-2",
        finding_class="C1",
    )


# ---------------------------------------------------------------------------
# Contract-driven kit for model-and-batch-taking losses (bev-distill shape)
# ---------------------------------------------------------------------------

# Probe-owned tiny widths for the package's own builders — the loss needs the
# real classes' method behavior (forward_with_bev_features, projection heads),
# not their paper-scale capacity. Names mirror the manifest conventions the
# way al_selector_kit's _StubNet carries n_classes/num_classes.
_TINY_N_CLASSES = 3
_TINY_BUILDER_CANDIDATES = {
    "num_classes": _TINY_N_CLASSES, "n_classes": _TINY_N_CLASSES,
    "bev_channels": 8, "bev_h": 8, "bev_w": 8,
    "hidden_dim": 8, "embed_dim": 8, "feature_dim": 8, "channels": 8,
    "num_queries": 5, "n_queries": 5,
    "seed": 0,
}
_BATCH_SMOKE = 4          # batch elements, the design note's smoke scale
_KNOB_MARKERS = ("size", "num", "count", "return")  # batch_size is a knob,
                                                    # not the batch


def _batch_param(names) -> str | None:
    for name in names:
        low = name.lower()
        if low == "batch" or (low.startswith("batch") and
                              not any(m in low for m in _KNOB_MARKERS)):
            return name
    return None


def requires_model_batch_kit(fn: Callable) -> bool:
    """True when the loss demands more than the two named tensors — i.e. it
    has required parameters beyond its student/teacher params, so the plain
    tensor fast path cannot bind it and the contract-driven kit must."""
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return False
    student = _find_param(fn, _STUDENT_HINTS)
    teacher = _find_param(fn, _TEACHER_HINTS)
    for name, p in params.items():
        if p.default is not inspect.Parameter.empty:
            continue
        if p.kind not in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY):
            continue
        if name not in (student, teacher):
            return True
    return False


def _resolve_dim(token: str, n_classes: int) -> int:
    """Map one symbolic dim from a contract shape string to a smoke value."""
    token = token.strip()
    try:
        return int(token)
    except ValueError:
        pass
    low = token.lower()
    if low == "b" or "batch" in low:
        return _BATCH_SMOKE
    if "point" in low:
        return 64
    if "img" in low or low in ("h", "w") or low.startswith(("h_", "w_")):
        return 32
    if "obj" in low:
        return 3
    if "class" in low:
        return n_classes
    if "quer" in low:
        return 5
    return 8


def _parse_tensor_dims(decl: str, n_classes: int) -> list[int] | None:
    """Parse a contract shape string like ``(B, 3, H_img, W_img)`` into
    concrete smoke dims; None when the decl is not a plain tensor shape."""
    decl = decl.strip()
    start, end = decl.find("("), decl.find(")")
    if start != 0 or end < 0:
        return None
    tokens = [t for t in decl[start + 1:end].split(",") if t.strip()]
    if not tokens:
        return None
    return [_resolve_dim(t, n_classes) for t in tokens]


def _pick_plausible_boxes(boxes) -> list[int]:
    """Rank predicted boxes by geometric validity, best first.

    Target seeding copies the teacher's own predicted boxes as ground truth
    so IoU-gated quality weighting sees perfect overlap — but an untrained
    box head emits many boxes with non-positive extents, and a zero-area box
    has IoU 0 against EVERYTHING (including its own copy), which silently
    zeroes the instance branch again. The box format is method-owned and
    undeclared, so validity is scored under both conventions: positive
    width/height (center format) and positive corner deltas (xyxy format).
    """
    scored = []
    for i in range(boxes.shape[0]):
        x1, y1, a, b = (float(v) for v in boxes[i][:4])
        valid_center = a > 1e-3 and b > 1e-3
        valid_xyxy = (a - x1) > 1e-3 and (b - y1) > 1e-3
        scored.append((-(int(valid_center) + int(valid_xyxy)), i))
    return [i for _score, i in sorted(scored)]


def _synthesize_batch(shape_decl: dict, teacher, n_classes: int,
                      seed: int) -> tuple[dict, str | None] | str:
    """Build the batch dict from the loader contract's declared shapes.

    Tensor-shaped keys become seeded ``torch.rand`` tensors. A targets key
    is recognized in BOTH declared forms seen live (the two concrete cases,
    2026-07-02 and 2026-07-04): ``list[dict] ... boxes ... labels`` (one
    dict per batch element) and ``dict — {'boxes': (B, N, 4), 'labels':
    (B, N)}`` (batched tensors). Either way it is seeded from the TEACHER'S
    OWN detached predictions when its forward returns the contract-declared
    ``pred_boxes``/``pred_logits`` keys — random ground truth has ~zero IoU
    with untrained predicted boxes, so IoU-gated quality weighting
    (Eq-6-style) zeroes the instance branch and a live temperature term
    reads falsely inert. Falls back to seeded random boxes/labels.

    Returns ``(batch, teacher_input_key)`` or an unprobeable-reason string
    naming the first key whose declaration the synthesizer cannot honor.
    """
    import torch  # noqa: PLC0415

    g = torch.Generator().manual_seed(seed)
    batch: dict = {}
    targets_keys: list[tuple[str, str]] = []
    teacher_key: str | None = None
    for key, decl in shape_decl.items():
        if not isinstance(decl, str):
            return (f"batch key {key!r} has a non-string shape declaration "
                    f"in the arch contract — cannot synthesize it")
        # Case-insensitive form match: the arch contract is agent-authored
        # prose, and the 2026-07-05 detr-distill contract wrote
        # "List[dict] of length B, ..." — the capital L fell through the
        # exact match, the recognized targets form went to the tensor
        # parser, and every KD probe read unprobeable. That single
        # character was the whole "KD structured-targets gap".
        decl_l = decl.lower()
        if "boxes" in decl_l and "labels" in decl_l and (
                "list[dict]" in decl_l or decl_l.lstrip().startswith("dict")):
            targets_keys.append((key, decl))
            continue
        dims = _parse_tensor_dims(decl, n_classes)
        if dims is None:
            return (f"batch key {key!r} declares unparseable shape {decl!r} "
                    f"— cannot synthesize it")
        batch[key] = torch.rand(*dims, generator=g)
        if "teacher" in key.lower():
            teacher_key = key

    # Teacher-prediction-seeded targets (detached; fall back to random).
    preds = None
    if targets_keys and teacher is not None and teacher_key is not None:
        try:
            with torch.no_grad():
                out = teacher(batch[teacher_key])
            if isinstance(out, dict) and "pred_boxes" in out:
                preds = out
        except Exception:  # noqa: BLE001 — fall back to random targets
            preds = None

    for key, decl in targets_keys:
        n_obj = 3
        per_element: list[tuple] = []
        for b in range(_BATCH_SMOKE):
            if preds is not None:
                pick = _pick_plausible_boxes(preds["pred_boxes"][b])[:n_obj]
                boxes = preds["pred_boxes"][b][pick].detach().clone()
                if "pred_logits" in preds:
                    labels = (preds["pred_logits"][b][pick]
                              .argmax(-1).detach().clone())
                else:
                    labels = torch.randint(0, n_classes, (n_obj,), generator=g)
            else:
                boxes = torch.rand(n_obj, 4, generator=g) * 6 + 1
                labels = torch.randint(0, n_classes, (n_obj,), generator=g)
            per_element.append((boxes, labels))
        if "list[dict]" in decl.lower():
            batch[key] = [{"boxes": b, "labels": l} for b, l in per_element]
        else:
            # dict-of-batched-tensors form: {'boxes': (B, N, 4),
            # 'labels': (B, N)}.
            batch[key] = {
                "boxes": torch.stack([b for b, _ in per_element]),
                "labels": torch.stack([l for _, l in per_element]),
            }

    return batch, teacher_key


def _resolve_symbol(package, name: str):
    """Shared resolver, see probes.package_loader.resolve_symbol."""
    from probes.package_loader import resolve_symbol  # noqa: PLC0415

    return resolve_symbol(package, name)


def kd_loss_kit(package, pluggable_name: str, arch_contract: dict | None = None,
                seed: int = 0) -> dict | str:
    """Shared KD-loss harness: resolve the loss's invocation contract from
    the run's own artifacts and return a seed-pinned zero-arg-invokable kit,
    or the unprobeable-reason string (never a fabricated stand-in).

    Tensor-taking losses return a ``{"mode": "tensor", "fn": fn}`` kit and
    keep the original fast path. Model-and-batch losses return
    ``{"mode": "model_batch", ...}`` with the package's own built models and
    a contract-synthesized batch.
    """
    try:
        import torch  # noqa: PLC0415
    except ImportError:
        return "torch required"

    fn = _resolve_symbol(package, pluggable_name)
    if fn is None:
        return f"pluggable {pluggable_name!r} not found in method package"
    student_param = _find_param(fn, _STUDENT_HINTS)
    teacher_param = _find_param(fn, _TEACHER_HINTS)
    if student_param is None or teacher_param is None:
        return (f"{pluggable_name!r} exposes no teacher/student parameters "
                f"by name")

    if not requires_model_batch_kit(fn):
        return {"mode": "tensor", "fn": fn}

    # --- model-and-batch mode -------------------------------------------
    sig = inspect.signature(fn)
    required = [n for n, p in sig.parameters.items()
                if p.default is inspect.Parameter.empty
                and p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)]
    extras = [n for n in required if n not in (student_param, teacher_param)]
    batch_param = _batch_param(extras)

    builders = {}
    for role, builder_name in (("student", "build_student"),
                               ("teacher", "build_teacher")):
        builder = _resolve_symbol(package, builder_name)
        if builder is None:
            return (f"{pluggable_name} takes {role} as a model, but the "
                    f"package exposes no {builder_name}() to construct it")
        kwargs = _fill_kwargs(builder, _TINY_BUILDER_CANDIDATES)
        if isinstance(kwargs, str):
            return (f"{builder_name} requires unmapped parameter {kwargs!r} "
                    f"— cannot build a probe-scale {role}")
        builders[role] = (builder, kwargs)

    def _make(role: str, model_seed: int):
        builder, kwargs = builders[role]
        torch.manual_seed(model_seed)
        model = builder(**kwargs)
        eval_fn = getattr(model, "eval", None)
        if callable(eval_fn):
            eval_fn()  # dropout off: the probe compares loss VALUES
        return model

    try:
        student = _make("student", seed)
        teacher = _make("teacher", seed)
    except Exception as e:  # noqa: BLE001 — generated builders fail arbitrarily
        return (f"build_student/build_teacher raised at probe-tiny widths: "
                f"{type(e).__name__}: {e}")

    batch = None
    teacher_batch_key = None
    if batch_param is not None:
        shape_decl = ((arch_contract or {}).get("pluggable_component") or {}
                      ).get("batch_dict_shape")
        if not isinstance(shape_decl, dict) or not shape_decl:
            return (f"{pluggable_name} takes {batch_param!r} but the arch "
                    f"contract declares no pluggable_component."
                    f"batch_dict_shape to synthesize it from")
        synthesized = _synthesize_batch(shape_decl, teacher,
                                        _TINY_N_CLASSES, seed)
        if isinstance(synthesized, str):
            return synthesized
        batch, teacher_batch_key = synthesized

    base_kwargs = _fill_kwargs(fn, {
        student_param: student, teacher_param: teacher,
        **({batch_param: batch} if batch_param else {}),
        "seed": seed, "random_seed": seed, "rng_seed": seed,
        "num_classes": _TINY_N_CLASSES, "n_classes": _TINY_N_CLASSES,
    })
    if isinstance(base_kwargs, str):
        return f"{pluggable_name} requires unmapped parameter {base_kwargs!r}"

    def invoke(**overrides):
        torch.manual_seed(seed)
        np.random.seed(seed)
        kwargs = dict(base_kwargs)
        kwargs.update(overrides)
        return fn(**kwargs)

    # Trial invocation: surface undeclared-batch-key reads and other
    # contract-conformant-input crashes as named unprobeables now, so the
    # probes never report a half-broken kit as behavioral evidence.
    try:
        invoke()
    except KeyError as e:
        return (f"{pluggable_name} reads batch key {e} that the loader "
                f"contract does not declare")
    except Exception as e:  # noqa: BLE001 — generated code fails arbitrarily
        return (f"{pluggable_name} raised on contract-synthesized inputs: "
                f"{type(e).__name__}: {e}")

    return {
        "mode": "model_batch", "fn": fn, "invoke": invoke,
        "base_kwargs": base_kwargs, "batch": batch,
        "student": student, "teacher": teacher,
        "student_param": student_param, "teacher_param": teacher_param,
        "batch_param": batch_param, "teacher_batch_key": teacher_batch_key,
        "make_teacher": lambda s: _make("teacher", s),
        "seed": seed, "n_classes": _TINY_N_CLASSES,
        "remake": lambda s: kd_loss_kit(
            package, pluggable_name, arch_contract=arch_contract, seed=s),
    }


# How many model/batch compositions a sensitivity arm tries before calling a
# term inert. Untrained models make branch-gating a coin flip per seed —
# bev-distill's IoU-gated quality weighting zeroes the whole instance branch
# whenever the seed's box head emits only non-positive extents (2 of the
# first 6 seeds are live). One witness composition where the term moves the
# loss proves it live; constant across ALL compositions is the C1 inert class.
_N_COMPOSITIONS = 4


def _compositions(kit: dict):
    """Yield the base kit then deterministically re-seeded rebuilds."""
    yield kit
    for i in range(1, _N_COMPOSITIONS):
        rebuilt = kit["remake"](kit["seed"] + i)
        if isinstance(rebuilt, dict):
            yield rebuilt


def _comparable(out) -> "np.ndarray":
    """Loss output → flat float array. Detaches torch tensors first (a
    model-batch loss carries grad; np.asarray refuses grad-bearing tensors)."""
    detach = getattr(out, "detach", None)
    if callable(detach):
        out = detach().cpu().numpy()
    return np.asarray(out, dtype=float).ravel()


def _outputs_differ(kit: dict, vary_param: str,
                    candidates: list) -> tuple[bool, str] | str:
    """Invoke the kit's loss varying ONE argument; report whether any output
    differs BIT-EXACTLY from the first.

    Exact comparison is deliberate: a genuinely inert term yields
    bit-identical floats, while a weak-but-live channel (bev-distill's tau
    moves a 5.5-magnitude loss by ~1e-6) drowns inside np.allclose's relative
    tolerance and would false-fail. The trade-off — cancellation-style
    inertness like ``(x/t)*t`` may escape by one ulp — is accepted: the C1
    founding class is unused/constant terms, which are always bit-identical.

    Returns ``(moved, evidence)`` or an error-reason string.
    """
    outs = []
    for cand in candidates:
        try:
            outs.append(_comparable(kit["invoke"](**{vary_param: cand})))
        except Exception as e:  # noqa: BLE001 — generated code fails arbitrarily
            return (f"{kit['fn'].__name__}({vary_param}=<variant>) raised "
                    f"{type(e).__name__}: {e}")
    moved = any(o.shape != outs[0].shape or not np.array_equal(o, outs[0])
                for o in outs[1:])
    return moved, f"outputs={[float(o[0]) if o.size else None for o in outs]}"


def _teacher_influence_once(kit: dict) -> tuple[str, str]:
    """One composition's KD-1 arms. Returns (state, detail) where state is
    "moved" / "constant" / "error"."""
    import torch  # noqa: PLC0415

    seed = kit["seed"]
    errors = []
    if kit["batch_param"] is not None and kit["teacher_batch_key"] is not None:
        key, batch = kit["teacher_batch_key"], kit["batch"]
        t = batch[key]
        g = torch.Generator().manual_seed(seed + 1)
        perm = torch.randperm(t.shape[0], generator=g)
        if int((perm == torch.arange(t.shape[0])).all()):
            perm = torch.roll(perm, 1)
        shuffled = dict(batch)
        shuffled[key] = t[perm]
        scaled = dict(batch)
        scaled[key] = t * 3.0
        res = _outputs_differ(kit, kit["batch_param"],
                              [batch, shuffled, scaled])
        if isinstance(res, str):
            errors.append(res)
        elif res[0]:
            return "moved", f"teacher inputs in the batch ({key!r} shuffled and ×3)"

    try:
        teacher_alt = kit["make_teacher"](seed + 1)
        res = _outputs_differ(kit, kit["teacher_param"],
                              [kit["teacher"], teacher_alt])
        if isinstance(res, str):
            errors.append(res)
        elif res[0]:
            return "moved", "the teacher model's weights (re-seeded build)"
    except Exception as e:  # noqa: BLE001
        errors.append(f"re-seeded build_teacher raised {type(e).__name__}: {e}")

    if errors and len(errors) == (2 if kit["teacher_batch_key"] else 1):
        return "error", errors[-1]
    return "constant", ""


def probe_kd_teacher_influence(kit: dict) -> ProbeVerdict:
    """KD-1 over a kit: teacher-side perturbations must change the loss.

    Tensor kits route to the original fast path. Model-batch kits perturb
    the teacher's inputs in the batch (shuffle across the batch dim, scale
    ×3) and, independently, the teacher model's weights (a re-seeded build)
    — over up to ``_N_COMPOSITIONS`` model/batch compositions. One witness
    composition where the loss moves is a pass; constant across ALL of them
    means the teacher signal is ignored (C1).
    """
    if kit["mode"] == "tensor":
        return probe_teacher_signal_influence(kit["fn"])

    fn_name = kit["fn"].__name__
    tried, errors = 0, []
    for comp in _compositions(kit):
        state, detail = _teacher_influence_once(comp)
        if state == "moved":
            return ProbeVerdict(
                "KD-1", "pass",
                f"{fn_name} responds to {detail}"
                + (f" (composition seed {comp['seed']})"
                   if comp["seed"] != kit["seed"] else ""),
                finding_class="C1")
        if state == "error":
            errors.append(detail)
        tried += 1
    if errors and len(errors) == tried:
        return ProbeVerdict("KD-1", "unprobeable", errors[0])
    return ProbeVerdict(
        "KD-1", "fail",
        f"{fn_name} output is constant across teacher-side perturbations "
        f"(batch inputs and model weights, {tried} model/batch "
        f"compositions) — the teacher signal has no effect on the result",
        finding_class="C1")


def probe_kd_temperature(kit: dict) -> ProbeVerdict | None:
    """KD-2 over a kit: varying the temperature-like kwarg must change the
    loss. Returns None when the loss exposes no temperature parameter.

    Candidates anchor at the signature's own default (bev-distill's
    tau=0.07 regime differs from softmax-T's 1.0-8.0) and spread ×64.
    Tries up to ``_N_COMPOSITIONS`` model/batch compositions: gated branch
    structures (IoU-weighted instance terms) can zero the temperature's
    whole branch on an unlucky seed, and one live witness settles it.
    """
    if kit["mode"] == "tensor":
        return probe_temperature_sensitivity(kit["fn"])

    fn = kit["fn"]
    temp_param = _find_param(fn, _TEMPERATURE_HINTS)
    if temp_param is None:
        return None
    default = inspect.signature(fn).parameters[temp_param].default
    base = float(default) if isinstance(default, (int, float)) \
        and not isinstance(default, bool) else 1.0
    candidates = [base, base * 4.0, base * 16.0, base * 64.0]

    tried, errors = 0, []
    for comp in _compositions(kit):
        res = _outputs_differ(comp, temp_param, candidates)
        if isinstance(res, str):
            errors.append(res)
            tried += 1
            continue
        moved, evidence = res
        if moved:
            return ProbeVerdict(
                "KD-2", "pass",
                f"{fn.__name__} responds to {temp_param!r} "
                f"({len(candidates)} candidates from {base:g})"
                + (f" (composition seed {comp['seed']})"
                   if comp["seed"] != kit["seed"] else ""),
                evidence=evidence, finding_class="C1")
        tried += 1
    if errors and len(errors) == tried:
        return ProbeVerdict("KD-2", "unprobeable", errors[0])
    return ProbeVerdict(
        "KD-2", "fail",
        f"{fn.__name__} output is constant across {len(candidates)} values "
        f"of {temp_param!r} ({tried} model/batch compositions) — the input "
        f"has no effect on the result",
        finding_class="C1")
