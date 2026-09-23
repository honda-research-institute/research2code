"""Method-package loading + recording stubs for behavioral probes (slice 1.2).

Two loading paths, both isolation-careful:

- `load_module_from_path(...)` — import a lone generated file (e.g. a zoo
  scenario that snapshots only method.py) under a unique module name.
- `imported_method_package(run_dir)` — context-manage `import method` against
  a full run dir, with sys.path/sys.modules restored afterward so multiple
  artifacts can be probed in one process.

Missing third-party dependencies (torch/sklearn on a probe host) raise
ProbeLoadError with the dependency named — the probe runner maps that to an
`unprobeable` verdict instead of crashing or silently passing (v3's
skip-collapses-to-pass lesson, inverted).

The stub kit supports the loop-microharness probes (AL-1/AL-4/MP-1): recording
fakes that stand in for package symbols inside an extracted notebook loop,
plus a StubModel whose identity/fingerprint changes on every build so
warm-start detection (M-003) has something behavioral to grip.
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.util
import sys
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Any, Callable


class ProbeLoadError(RuntimeError):
    """Artifact could not be loaded; `missing_dependency` names the module
    when the cause is an absent third-party package."""

    def __init__(self, message: str, missing_dependency: str | None = None):
        super().__init__(message)
        self.missing_dependency = missing_dependency


def _missing_dep_from(exc: ImportError) -> str | None:
    name = getattr(exc, "name", None)
    if name:
        return name.split(".")[0]
    return None


def load_module_from_path(py_path: Path) -> ModuleType:
    """Import a single .py file under a content-unique module name."""
    py_path = Path(py_path)
    if not py_path.is_file():
        raise ProbeLoadError(f"no such file: {py_path}")
    digest = hashlib.sha256(str(py_path.resolve()).encode()).hexdigest()[:12]
    mod_name = f"_probe_target_{py_path.stem}_{digest}"
    spec = importlib.util.spec_from_file_location(mod_name, py_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module
    try:
        spec.loader.exec_module(module)
    except ImportError as e:
        del sys.modules[mod_name]
        dep = _missing_dep_from(e)
        raise ProbeLoadError(
            f"{py_path.name} requires {dep or 'a missing module'}: {e}",
            missing_dependency=dep,
        ) from e
    except Exception as e:  # noqa: BLE001 — generated code can fail arbitrarily
        del sys.modules[mod_name]
        raise ProbeLoadError(f"{py_path.name} failed at import time: {e}") from e
    return module


@contextmanager
def imported_method_package(run_dir: Path):
    """`import method` against run_dir, restoring interpreter state after.

    Yields the imported package. Any previously-imported `method*` modules
    are stashed and restored so consecutive probes on different runs don't
    bleed into each other.
    """
    run_dir = Path(run_dir)
    if not (run_dir / "method").is_dir():
        raise ProbeLoadError(f"no method/ package under {run_dir}")

    stashed = {k: sys.modules.pop(k) for k in list(sys.modules)
               if k == "method" or k.startswith("method.")}
    sys.path.insert(0, str(run_dir))
    try:
        importlib.invalidate_caches()
        try:
            yield importlib.import_module("method")
        except ImportError as e:
            dep = _missing_dep_from(e)
            raise ProbeLoadError(
                f"method package under {run_dir} requires "
                f"{dep or 'a missing module'}: {e}",
                missing_dependency=dep,
            ) from e
    finally:
        sys.path.remove(str(run_dir))
        for k in [k for k in sys.modules
                  if k == "method" or k.startswith("method.")]:
            del sys.modules[k]
        sys.modules.update(stashed)


def resolve_symbol(package, name: str,
                   subs: tuple[str, ...] = ("method", "training", "model",
                                            "data")):
    """Resolve ``name`` from the package top level, then from the
    conventional submodules (the build plan re-exports through __init__, but
    a hand-rolled package may not). Shared by the probe kits (KD losses live
    in method/training, motion-planning Environment/dynamics classes live in
    data/model per the paradigm templates)."""
    obj = getattr(package, name, None)
    if obj is not None:
        return obj
    for sub in subs:
        mod = getattr(package, sub, None)
        if mod is not None:
            obj = getattr(mod, name, None)
            if obj is not None:
                return obj
    return None


def pluggable_from_spec(spec_path: Path) -> tuple[str, str]:
    """(name, signature) of the pluggable component from method_spec.json.

    Probe target names MUST come from the spec, never from defaults: fresh
    runs rename the entire public API (the 2026-06-10 pdwa run produced
    plan/compute_j_* where May's produced select_velocity/
    evaluate_objective_function).
    """
    import json  # noqa: PLC0415

    spec = json.loads(Path(spec_path).read_text(encoding="utf-8"))
    pc = spec.get("comparison", {}).get("pluggable_component", {})
    name = pc.get("name", "")
    if not name:
        raise ProbeLoadError(f"no pluggable_component.name in {spec_path}")
    return name, pc.get("signature", "")


# ---------------------------------------------------------------------------
# Stub kit
# ---------------------------------------------------------------------------


@dataclass
class RecordedCall:
    args: tuple
    kwargs: dict


@dataclass
class RecordingStub:
    """Callable stand-in that records every invocation.

    `returns` may be a value or a callable receiving (*args, **kwargs).
    """

    name: str
    returns: Any = None
    calls: list[RecordedCall] = field(default_factory=list)

    def __call__(self, *args, **kwargs):
        self.calls.append(RecordedCall(args=args, kwargs=dict(kwargs)))
        if callable(self.returns):
            return self.returns(*args, **kwargs)
        return self.returns

    @property
    def call_count(self) -> int:
        return len(self.calls)


class StubModel:
    """Identity-distinct model stand-in: every construction gets a fresh
    fingerprint, and `train_marks` counts in-place training calls — the pair
    of signals the fresh-weights probe (AL-4 / M-003) asserts on."""

    _counter = 0

    def __init__(self):
        StubModel._counter += 1
        self.fingerprint = StubModel._counter
        self.train_marks = 0


def make_al_stub_kit(
    batch_size: int,
    select_positions: Callable[[int], list[int]] | None = None,
) -> dict[str, RecordingStub]:
    """Stubs for the §5.1 active-learning loop microharness (AL-1).

    `select_batch` returns ADVERSARIAL positions by default — round r yields
    positions [r*1 .. r*1+batch_size) mod pool — so positional-offset
    arithmetic visibly corrupts bookkeeping instead of coinciding with it.
    """
    def _default_positions(round_idx: int) -> list[int]:
        return [round_idx + i for i in range(batch_size)]

    positions_for = select_positions or _default_positions
    select_calls = {"n": 0}

    def _select(*_args, **_kwargs):
        out = positions_for(select_calls["n"])
        select_calls["n"] += 1
        return out

    def _build(*_args, **_kwargs):
        return StubModel()

    def _train(model, *_args, **_kwargs):
        model.train_marks += 1
        return model

    return {
        "build_model": RecordingStub("build_model", returns=_build),
        "train_from_scratch": RecordingStub("train_from_scratch", returns=_train),
        "select_batch": RecordingStub("select_batch", returns=_select),
        "evaluate": RecordingStub("evaluate", returns=lambda *a, **k: 0.5),
    }
