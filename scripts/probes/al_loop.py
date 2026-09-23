"""AL acquisition-loop microharness (AL-1/AL-2/AL-4; finding classes M-002/M-003).

The pipeline's silent-failure capital is the notebook's §5 acquisition loop:
index-space bugs (M-002) and warm-starting (M-003) run fine, look right, and
were waved through by LLM reviewers WITH the failure mode named in the check.
This harness extracts the loop cell and EXECUTES it in a sandbox where the
method package is replaced by recording stubs and the data is a tiny synthetic
pool with unique rows — then asserts value-level invariants that hold for any
correct loop regardless of its variable names or bookkeeping style:

- AL-1a  the unlabeled set passed to the pluggable shrinks by batch_size/round
- AL-1b  the trained set grows by batch_size/round
- AL-1c  the rows selected at round r are exactly rows added to training
- AL-1d  rows selected once never reappear in a later unlabeled set
- AL-4   a FRESH model is built for every training call (fingerprinted stubs)
- AL-2   (static, warn-tier) positional-offset arithmetic near indexing

Adversarial stubs are the trick: `select_batch` returns a different shifted
position window each round, so offset arithmetic that would coincidentally
work with round-stable positions visibly corrupts the bookkeeping.

Seeding is name-agnostic: the cell is executed and every NameError is
resolved through a heuristic table (pool/test tensors, index lists that are
deliberately NON-contiguous, cfg built from the cell's own subscripts, stub
models). Unknown names make the artifact `unprobeable` — never guessed.

Requires torch (the loops use torch ops); absent torch => unprobeable.
"""

from __future__ import annotations

import ast
import json
import re
import runpy
import tempfile
from pathlib import Path

import numpy as np

from probes import ProbeVerdict
from probes.catalogs.al_loop import PROBE_CATALOG as _PROBE_CATALOG
from probes.package_loader import RecordingStub, StubModel

PROBE_CATALOG = _PROBE_CATALOG
_OFFSET_ARITH_RE = re.compile(r"[-+]\s*len\(")


# ---------------------------------------------------------------------------
# Loop-cell location + cfg-key discovery
# ---------------------------------------------------------------------------


def find_acquisition_loop_cell(nb: dict, pluggable_name: str) -> tuple[str, str] | None:
    """(cell_id, source) of the first code cell whose loop body calls the
    pluggable component."""
    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        src = "".join(cell["source"])
        if pluggable_name not in src:
            continue
        try:
            tree = ast.parse(src)
        except SyntaxError:
            continue
        for loop in (n for n in ast.walk(tree) if isinstance(n, (ast.For, ast.While))):
            for call in (n for n in ast.walk(loop) if isinstance(n, ast.Call)):
                fn = call.func
                name = fn.id if isinstance(fn, ast.Name) else getattr(fn, "attr", "")
                if name == pluggable_name:
                    return cell.get("id", "?"), src
    return None


def _cfg_keys(src: str) -> set[str]:
    keys: set[str] = set()
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return keys
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Subscript)
            and isinstance(node.value, ast.Name)
            and node.value.id == "cfg"
            and isinstance(node.slice, ast.Constant)
            and isinstance(node.slice.value, str)
        ):
            keys.add(node.slice.value)
    return keys


# ---------------------------------------------------------------------------
# Sandbox
# ---------------------------------------------------------------------------


class _Silent:
    """No-op stand-in for plotting modules and print."""

    def __call__(self, *a, **k):
        return self

    def __getattr__(self, _name):
        return self


def _sandbox_model_cls(torch, n_classes: int):
    class _SandboxModel(StubModel):
        # Real nn.Modules always carry the `training` bool; loop cells that
        # save/restore mode (`was_train = model.training`) made AL-1
        # unprobeable on the 2026-07-02 GBALD run. Kept consistent by
        # train()/eval() below, initialized True (the nn.Module default).
        training = True

        def eval(self):
            self.training = False
            return self

        def train(self, mode=True, *_a, **_k):
            self.training = bool(mode)
            return self

        def __call__(self, x):
            n = len(x)
            return torch.zeros((n, n_classes))

    return _SandboxModel


class _SeededIndexList(list):
    """Initial label-index container that satisfies BOTH idioms generated loop
    cells use for it: plain-list ops (`append`, slicing, `list(...)`, `.copy()`)
    AND the tensor/ndarray API (`.tolist()`, `.numpy()`) for notebooks that keep
    the label indices as a torch/numpy object. Seeding as a bare list made AL-1
    unprobeable on the 2026-06-30 BADGE run (`set(idx.tolist())` → AttributeError);
    seeding as a bare ndarray would instead break the `.append()`/`.copy()` idiom.
    This hybrid keeps AL-1 probeable across both."""

    def tolist(self):
        return list(self)

    def numpy(self):
        return np.asarray(self)

    def copy(self):
        return _SeededIndexList(self)


class ALLoopSandbox:
    """One attempt-loop execution environment for an acquisition-loop cell."""

    def __init__(self, torch, n_pool=60, n_features=6, n_classes=3,
                 batch_size=4, num_rounds=3, seed=0):
        rng = np.random.default_rng(seed)
        # Unique rows so row-identity equals point-identity.
        base = rng.normal(0.0, 1.0, size=(n_pool, n_features))
        base += np.arange(n_pool)[:, None] * 1e-3
        self.torch = torch
        self.n_classes = n_classes
        self.batch_size = batch_size
        self.num_rounds = num_rounds
        self.seed = seed
        self.x_pool = torch.tensor(base, dtype=torch.float32)
        self.y_pool = torch.tensor(rng.integers(0, n_classes, n_pool))
        self.x_test = torch.tensor(
            rng.normal(0.0, 1.0, size=(8, n_features)), dtype=torch.float32)
        self.y_test = torch.tensor(rng.integers(0, n_classes, 8))
        # Deliberately NON-contiguous initial labels: exposes the
        # contiguous-prefix assumption (M-002's enabling belief).
        self.init_indices = sorted(
            rng.choice(n_pool, size=8, replace=False).tolist())

        model_cls = _sandbox_model_cls(torch, n_classes)
        # Ordered event log: loop styles differ (badge acquires then trains
        # within a round; gbald trains then acquires, so its FINAL selection
        # never reaches training) — invariants must respect call order.
        self.timeline: list[tuple] = []

        def _positions(round_idx: int) -> list[int]:
            return [round_idx + i for i in range(batch_size)]

        select_count = {"n": 0}

        def _select(*a, **k):
            # The unlabeled pool is the pluggable's second positional arg
            # (`select_batch(model, x_unlabeled, ...)` for every AL variant),
            # but faithful notebooks may pass it by keyword — the 2026-06-30
            # GBALD run called select_batch with ALL keyword args, so a bare
            # `a[1]` raised IndexError and made AL-1 unprobeable. Resolve the
            # pool from keyword-or-position.
            pool = k.get("x_unlabeled", k.get("x_pool"))
            if pool is None and len(a) > 1:
                pool = a[1]
            out = _positions(select_count["n"])
            # Positions must index the pool the loop actually passed — a
            # shrinking pool made the unbounded [round, round+batch) pattern
            # raise IndexError on the 2026-07-02 GBALD run. Wrap and dedupe;
            # an exhausted pool selects nothing.
            try:
                n_avail = len(pool) if pool is not None else 0
            except TypeError:
                n_avail = 0
            if n_avail:
                out = list(dict.fromkeys(p % n_avail for p in out))
            elif pool is not None:
                out = []
            select_count["n"] += 1
            self.timeline.append(("select", _rows(pool), list(out)))
            return out

        def _train(model, *a, **_k):
            model.train_marks += 1
            self.timeline.append(("train", _rows(a[0]), model))
            return model

        self.stubs = {
            "build_model": RecordingStub("build_model",
                                         returns=lambda *a, **k: model_cls()),
            "train_from_scratch": RecordingStub("train_from_scratch", returns=_train),
        }
        self.pluggable_stub = RecordingStub("pluggable", returns=_select)

    def namespace_value(self, name: str, cfg_keys: set[str]):
        """Heuristic seeding for an unbound name; raises KeyError if unknown."""
        t = self.torch
        exact = {
            "x_pool": self.x_pool, "y_pool": self.y_pool,
            "x_test": self.x_test, "y_test": self.y_test,
            "SEED": self.seed, "seed": self.seed,
            "rng": np.random.default_rng(self.seed),
            "np": np, "numpy": np, "torch": t,
            "input_dim": self.x_pool.shape[1],
            "n_features": self.x_pool.shape[1],
            "n_classes": self.n_classes, "num_classes": self.n_classes,
            "plt": _Silent(), "print": _Silent(),
        }
        if name in exact:
            return exact[name]
        if name in self.stubs:
            return self.stubs[name]
        if name == "cfg":
            defaults = {"batch_size": self.batch_size,
                        "num_rounds": self.num_rounds}
            for key in cfg_keys:
                defaults.setdefault(key, 2)
            return defaults
        if re.search(r"idx|index|indices", name):
            # Index-name variants appear anywhere in the identifier, not just
            # as a suffix (GBALD 2026-07-03 roll: `labeled_idx_bootstrap` fell
            # through the old `endswith("_idx")` rule → AL-1 unprobeable).
            # Labeled/unlabeled index names must seed DISJOINT (the 2026-07-02
            # GBALD run seeded bootstrap_labeled_idx == bootstrap_unlabeled_idx,
            # so the loop's union never grew and the shrinking pool ran out).
            if "unlabeled" in name:
                labeled = set(self.init_indices)
                return _SeededIndexList(
                    i for i in range(len(self.x_pool)) if i not in labeled)
            return _SeededIndexList(self.init_indices)
        if "model" in name:
            return _sandbox_model_cls(self.torch, self.n_classes)()
        raise KeyError(name)


# ---------------------------------------------------------------------------
# The microharness
# ---------------------------------------------------------------------------


def _rows(tensor_like) -> list[bytes]:
    arr = np.asarray(tensor_like.detach() if hasattr(tensor_like, "detach")
                     else tensor_like, dtype=np.float32)
    return [row.tobytes() for row in arr]


def run_al_loop_microharness(
    nb_path: Path,
    pluggable_name: str = "select_batch",
    num_rounds: int = 3,
    batch_size: int = 4,
    seed: int = 0,
) -> list[ProbeVerdict]:
    try:
        import torch  # noqa: PLC0415
    except ImportError:
        return [ProbeVerdict("AL-1", "unprobeable",
                             "torch required for the loop microharness")]

    nb = json.loads(Path(nb_path).read_text(encoding="utf-8"))
    name = Path(nb_path).name
    located = find_acquisition_loop_cell(nb, pluggable_name)
    if located is None:
        return [ProbeVerdict(
            "AL-1", "unprobeable",
            f"no loop cell calling {pluggable_name!r} found", evidence=name)]
    cell_id, src = located
    cfg_keys = _cfg_keys(src)

    verdicts: list[ProbeVerdict] = []
    if _OFFSET_ARITH_RE.search(src):
        verdicts.append(ProbeVerdict(
            "AL-2", "warn",
            "positional-offset arithmetic (`± len(...)`) present in the loop "
            "cell — forbidden by the AL taxonomy contract; AL-1 is the authority",
            evidence=f"{name}:{cell_id}", finding_class="M-002"))

    # Execute with NameError-driven seeding. Fresh sandbox per attempt so
    # only the final (successful) attempt's recordings are inspected. The
    # cell runs from a scratch file via runpy so the stubs land in its
    # globals through init_globals.
    seeded: dict = {}
    with tempfile.TemporaryDirectory(prefix="r2c-al-cell-") as tmp:
        cell_file = Path(tmp) / "cell.py"
        cell_file.write_text(src, encoding="utf-8")
        for _attempt in range(25):
            sandbox = ALLoopSandbox(torch, batch_size=batch_size,
                                    num_rounds=num_rounds, seed=seed)
            ns = {"__builtins__": __builtins__}
            ns[pluggable_name] = sandbox.pluggable_stub
            for known, value in seeded.items():
                ns[known] = sandbox.namespace_value(known, cfg_keys)
                del value
            try:
                runpy.run_path(str(cell_file), init_globals=ns)
                break
            except NameError as e:
                missing = getattr(e, "name", None) or str(e).split("'")[1]
                try:
                    sandbox.namespace_value(missing, cfg_keys)
                except KeyError:
                    return verdicts + [ProbeVerdict(
                        "AL-1", "unprobeable",
                        f"loop cell needs unknown name {missing!r} — extend "
                        f"the seeding table or probe by hand",
                        evidence=f"{name}:{cell_id}")]
                seeded[missing] = True
            except Exception as e:  # noqa: BLE001 — generated code fails arbitrarily
                return verdicts + [ProbeVerdict(
                    "AL-1", "unprobeable",
                    f"loop cell raised {type(e).__name__}: {e}",
                    evidence=f"{name}:{cell_id}")]
        else:
            return verdicts + [ProbeVerdict(
                "AL-1", "unprobeable", "seeding did not converge",
                evidence=f"{name}:{cell_id}")]

    select_calls = sandbox.pluggable_stub.calls
    train_calls = sandbox.stubs["train_from_scratch"].calls
    if not select_calls or not train_calls:
        return verdicts + [ProbeVerdict(
            "AL-1", "unprobeable",
            f"loop ran but recorded {len(select_calls)} pluggable / "
            f"{len(train_calls)} training calls", evidence=f"{name}:{cell_id}")]

    failures: list[str] = []
    timeline = sandbox.timeline
    selects = [(i, rows, pos) for i, (kind, rows, pos) in enumerate(timeline)
               if kind == "select"]
    trains = [(i, rows, model) for i, (kind, rows, model) in enumerate(timeline)
              if kind == "train"]

    # AL-1a: unlabeled pool shrinks by batch_size per acquisition round.
    sizes = [len(rows) for _, rows, _ in selects]
    expected = [sizes[0] - batch_size * i for i in range(len(sizes))]
    if sizes != expected:
        failures.append(
            f"AL-1a: unlabeled sizes {sizes} (expected {expected}) — the "
            f"pool is not set-differenced after acquisition")

    # AL-1b: training sizes never shrink, every step is one acquired batch
    # (or a same-set retrain), and the loop nets at least
    # (rounds-1)*batch_size of growth. Steps of 0 are legitimate loop-shape
    # variation, NOT a defect: the fresh 2026-06-10 GBALD notebook retrains
    # from scratch both before acquisition (scoring model) and after it
    # (learning-curve model), so consecutive train events alternate 0 and
    # +batch_size — the strict every-step-grows form false-positived on it.
    # Shrinkage, off-by-something jumps, and stagnation all still fail.
    tsizes = [len(rows) for _, rows, _ in trains]
    growth = [b - a for a, b in zip(tsizes, tsizes[1:])]
    n_rounds = len(selects)
    min_net = max(0, (n_rounds - 1) * batch_size)
    if any(g not in (0, batch_size) for g in growth) \
            or sum(growth) < min_net:
        failures.append(
            f"AL-1b: training-set growth {growth} (each step must be 0 or "
            f"{batch_size}, with net growth >= {min_net})")

    # AL-1c: rows selected at round r are exactly the rows that enter a
    # SUBSEQUENT training call. A final selection with no training after it
    # is legitimate loop-shape variation (the unused-final-acquisition case
    # has its own static check) — skipped here, not failed.
    for r, (t_idx, pool_rows, positions) in enumerate(selects):
        if max(positions) >= len(pool_rows):
            continue
        later_training = set().union(*[set(rows) for i, rows, _ in trains
                                       if i > t_idx]) if any(
            i > t_idx for i, _, _ in trains) else None
        if later_training is None:
            continue
        chosen = {pool_rows[p] for p in positions}
        missing = chosen - later_training
        if missing:
            failures.append(
                f"AL-1c: round {r}: {len(missing)}/{batch_size} selected "
                f"rows never reach training — selections are mapped to the "
                f"wrong pool points")
            break

    # AL-1d: selected rows must not reappear in a later unlabeled set.
    for r, (t_idx, pool_rows, positions) in enumerate(selects):
        if max(positions) >= len(pool_rows):
            continue
        chosen = {pool_rows[p] for p in positions}
        leaked = False
        for later_r, (_, later_pool, _) in enumerate(
                selects[r + 1:], start=r + 1):
            leak = chosen & set(later_pool)
            if leak:
                failures.append(
                    f"AL-1d: {len(leak)} row(s) selected at round {r} are "
                    f"still selectable at round {later_r} — already-labeled "
                    f"points leak back into the pool")
                leaked = True
                break
        if leaked:
            break

    verdicts.append(
        ProbeVerdict(
            "AL-1", "fail",
            "; ".join(failures),
            evidence=f"{name}:{cell_id}", finding_class="M-002",
        ) if failures else ProbeVerdict(
            "AL-1", "pass",
            f"loop bookkeeping invariants hold over {len(select_calls)} "
            f"adversarial rounds", evidence=f"{name}:{cell_id}",
            finding_class="M-002",
        )
    )

    # AL-4: fresh model per training call (fingerprint diversity).
    trained_models = [c.args[0] for c in train_calls
                      if c.args and isinstance(c.args[0], StubModel)]
    if trained_models:
        fingerprints = {m.fingerprint for m in trained_models}
        if len(fingerprints) < len(trained_models):
            reused = max(m.train_marks for m in trained_models)
            verdicts.append(ProbeVerdict(
                "AL-4", "fail",
                f"one model object trained {reused}x across rounds — the "
                f"loop warm-starts where the protocol says retrain from "
                f"scratch", evidence=f"{name}:{cell_id}",
                finding_class="M-003"))
        else:
            verdicts.append(ProbeVerdict(
                "AL-4", "pass",
                f"{len(trained_models)} training calls used "
                f"{len(fingerprints)} distinct fresh models",
                evidence=f"{name}:{cell_id}", finding_class="M-003"))
    else:
        verdicts.append(ProbeVerdict(
            "AL-4", "unprobeable",
            "no stub models observed in training calls",
            evidence=f"{name}:{cell_id}", finding_class="M-003"))

    return verdicts


# ---------------------------------------------------------------------------
# AL-5 — acquisition-output contract (selector-level, no notebook needed)
# ---------------------------------------------------------------------------


def _crash_arm_verdict(package, pluggable_name: str, seed: int,
                       first_exc: Exception, first_site: str) -> ProbeVerdict:
    """AL-5 crash arm: the selector crashed inside its own code on a
    contract-conformant invocation. Reproduce on two more trial pools
    (CT-1's all-trials rule — a fail is never bad luck):

    - crashes on EVERY pool with a certain-defect signature → fail (the
      2026-06-10 GBALD class: `np.argsort(...)[::-1]` indexing a torch
      tensor crashes on every legal input; the delivered notebook hid it
      by never running the loop);
    - crashes on every pool, signature uncertain → flag_for_researcher
      (broken contract and undeclared input-domain assumption — an
      image-only selector on a generic fixture — are indistinguishable
      from the harness side);
    - crashes only on some pools → flag_for_researcher (fragile input
      contract).
    """
    from probes.term_ablation import (  # noqa: PLC0415
        al_selector_kit, attribute_invocation_crash, classify_subject_crash)

    # A call-time lazy import of a missing dependency is an environment
    # limitation, not a method defect — same routing as load-time deps
    # (ProbeLoadError → unprobeable), even though it raises in subject code.
    if isinstance(first_exc, ImportError):
        return ProbeVerdict(
            "AL-5", "unprobeable",
            f"missing dependency at selector call time: {first_exc}")

    crashes = [(type(first_exc).__name__, str(first_exc), first_site)]
    clean_trials = 0
    attempted = 1
    for t in (1, 2):
        kit_t = al_selector_kit(package, pluggable_name, seed,
                                fixture_seed=seed + t)
        if isinstance(kit_t, str):
            continue
        attempted += 1
        try:
            # Consume exactly like the contract check does — a generator
            # selector defers its crash to iteration time.
            [int(i) for i in kit_t["invoke"]()]
            clean_trials += 1
        except Exception as e_t:  # noqa: BLE001
            origin_t, site_t = attribute_invocation_crash(
                e_t, kit_t["package_dir"])
            if origin_t == "subject":
                crashes.append((type(e_t).__name__, str(e_t), site_t))

    exc_name, exc_text, site = crashes[0]
    detail = (f"{exc_name} at {site}: "
              f"{exc_text[:200]}{'…' if len(exc_text) > 200 else ''}")
    sites = f"crash_sites={sorted({c[2] for c in crashes})}"
    if clean_trials:
        return ProbeVerdict(
            "AL-5", "flag_for_researcher",
            f"fragile input contract: {pluggable_name} crashed on "
            f"{len(crashes)} of {attempted} contract-conformant synthetic "
            f"pools ({detail}) — some legal inputs crash it; verify against "
            f"the inputs your data will actually produce",
            evidence=sites)

    # The every-pool claim requires every attempted trial to have crashed
    # in subject code — a single observed crash plus unattributable trials
    # must not masquerade as full reproduction.
    if len(crashes) == attempted == 3:
        guidance = classify_subject_crash(first_exc)
        if guidance is not None:
            return ProbeVerdict(
                "AL-5", "fail",
                f"{pluggable_name} crashes inside the method package on "
                f"every contract-conformant invocation ({attempted} "
                f"distinct synthetic pools matching its declared "
                f"signature): {detail}. {guidance}",
                evidence=sites)
        return ProbeVerdict(
            "AL-5", "flag_for_researcher",
            f"{pluggable_name} crashes inside the method package on every "
            f"contract-conformant invocation ({detail}) — either the input "
            f"contract is broken (the crash is reachable on your data) or "
            f"the method holds an input-domain assumption its signature "
            f"does not declare; call it once on real data to adjudicate",
            evidence=sites)
    return ProbeVerdict(
        "AL-5", "flag_for_researcher",
        f"{pluggable_name} crashed inside the method package on "
        f"{len(crashes)} of {attempted} reproduction attempts ({detail}); "
        f"the remaining attempts could not be attributed — treat as a "
        f"fragile input contract and verify on real data",
        evidence=sites)


def probe_acquisition_contract(package, pluggable_name: str,
                               seed: int = 0) -> ProbeVerdict:
    """AL-5 with its binding declared (R2C-072).

    Every arm of this check exercises exactly one callable, the selector
    pluggable, so the verdict says so on the way out and the battery resolves
    that to the paper elements the selector implements. Stamped in a wrapper
    rather than at each of the nine return sites, so a future arm cannot ship
    unbound by forgetting.
    """
    verdict = _acquisition_contract(package, pluggable_name, seed)
    verdict.bound_callables = [pluggable_name]
    # This wrapper is shared by the early producer seam and the full battery.
    # Stamp the stable taxonomy executor identity here so both paths intersect
    # the same methodology obligation before the battery adds authored pack
    # context.
    verdict.probe_ref = "al_loop.acquisition_contract"
    return verdict


def _acquisition_contract(package, pluggable_name: str,
                          seed: int = 0) -> ProbeVerdict:
    """AL-5: the selector's output is batch_size unique in-range indices, and
    the selection responds to the model's weights.

    Contract violations (wrong count, duplicates, out-of-range) are hard
    fails — no paper makes them faithful. Model-insensitivity is routed
    flag_for_researcher instead: a purely geometric selector can be exactly
    what the paper specifies (core-set methods), so the researcher adjudicates
    against the paper's claim. Reuses the shared selector harness (fixture +
    MC-dropout stub + name-mapped kwargs)."""
    try:
        import torch  # noqa: PLC0415
    except ImportError:
        return ProbeVerdict("AL-5", "unprobeable", "torch required")
    from probes.term_ablation import (  # noqa: PLC0415
        al_selector_kit, attribute_invocation_crash)

    kit = al_selector_kit(package, pluggable_name, seed)
    if isinstance(kit, str):
        return ProbeVerdict("AL-5", "unprobeable", kit)

    try:
        out = kit["invoke"]()
        selection = [int(i) for i in out]
    except Exception as e:  # noqa: BLE001 — generated selectors fail freely
        origin, site = attribute_invocation_crash(e, kit["package_dir"])
        if origin != "subject":
            return ProbeVerdict(
                "AL-5", "unprobeable",
                f"selector invocation raised {type(e).__name__}: {e}")
        return _crash_arm_verdict(package, pluggable_name, seed, e, site)

    problems = []
    if len(selection) != kit["batch_size"]:
        problems.append(f"returned {len(selection)} indices, "
                        f"batch_size is {kit['batch_size']}")
    if len(set(selection)) != len(selection):
        problems.append(f"duplicate indices in one batch: {sorted(selection)}")
    out_of_range = [i for i in selection
                    if i < 0 or i >= kit["pool_size"]]
    if out_of_range:
        problems.append(f"indices outside the unlabeled pool "
                        f"[0, {kit['pool_size']}): {sorted(out_of_range)}")
    if problems:
        return ProbeVerdict(
            "AL-5", "fail",
            f"acquisition contract violated: {'; '.join(problems)}",
            evidence=f"selection={sorted(selection)[:12]}",
            finding_class="M-002")

    if "model" not in kit["kwargs"]:
        return ProbeVerdict(
            "AL-5", "pass",
            f"contract holds ({kit['batch_size']} unique in-range indices); "
            "selector takes no model — sensitivity arm not applicable")

    # Sensitivity arm: same pool, same seeds, different model weights — but the
    # weights must carry a real signal. An UNTRAINED stub yields ~uniform
    # predictions, so an uncertainty score (BALD / predictive entropy)
    # collapses to ~0 and a faithful uncertainty selector looks
    # model-insensitive purely for lack of signal. (The max-min direction fix
    # made this acute: with degenerate uncertainty the geometric stage fully
    # determines the pick, which is weight-independent.) So compare two stubs
    # TRAINED on the fixture labels. A selector that genuinely ignores the
    # model still selects identically (correctly flagged); one that consumes
    # uncertainty now responds. Uses the kit's own factory so the stub
    # interface (embedding hook included) is identical across both sides.
    make_sensitivity_model = kit.get("make_trained_model", kit["make_model"])

    def _run_with(model) -> list[int]:
        torch.manual_seed(seed)
        np.random.seed(seed)
        return [int(i) for i in kit["fn"](**{**kit["kwargs"], "model": model})]

    try:
        base_sel = _run_with(make_sensitivity_model(seed))
        other_out = _run_with(make_sensitivity_model(seed + 101))
    except Exception as e:  # noqa: BLE001
        origin, site = attribute_invocation_crash(e, kit["package_dir"])
        if origin == "subject":
            return ProbeVerdict(
                "AL-5", "flag_for_researcher",
                f"fragile input contract: {pluggable_name} passed the "
                f"baseline pool but crashed under different model weights "
                f"({type(e).__name__} at {site}: {e}) — some legal inputs "
                f"crash it",
                evidence=f"crash_sites=['{site}']")
        return ProbeVerdict(
            "AL-5", "unprobeable",
            f"sensitivity arm raised {type(e).__name__}: {e}")

    if set(other_out) == set(base_sel):
        return ProbeVerdict(
            "AL-5", "flag_for_researcher",
            "contract holds, but the selection is identical under different "
            "model weights (compared with trained stubs, so the uncertainty "
            "signal is real) — the selector may not consume model uncertainty "
            "at all; legitimate only if the paper's method is purely geometric "
            "(verify against the paper)",
            evidence=f"selection={sorted(base_sel)[:12]}",
            finding_class="M-004")
    return ProbeVerdict(
        "AL-5", "pass",
        f"contract holds ({kit['batch_size']} unique in-range indices) and "
        f"the selection responds to model weights")


# ---------------------------------------------------------------------------
# AL-6 — demo-config reachability (static, params.json only)
# ---------------------------------------------------------------------------


def probe_demo_config_reachability(params: dict) -> ProbeVerdict:
    """AL-6: the delivered demo configuration must actually exercise the
    acquisition loop.

    The 2026-06-10 GBALD validation re-run shipped initial_labeled=1000
    against pool_size=800: the core-set swallowed the whole pool, the
    unlabeled set was empty at round 0, and every select_batch call
    early-returned [] — so the smoke gate "passed" a notebook whose Stage-2
    contribution never executed once (and whose selector crashes on every
    non-empty pool, see the AL-5 crash arm). Pure params.json arithmetic:

    - initial_labeled >= pool_size → fail: the loop is unreachable, the
      delivered demo demonstrates nothing of the acquisition method
      (inert-mechanism class, M-004);
    - the per-round budget exhausts the pool before num_rounds complete →
      warn: trailing rounds silently acquire nothing;
    - missing/non-numeric keys → unprobeable, never guessed.
    """
    def value_of(name):
        entry = params.get(name)
        v = entry.get("value") if isinstance(entry, dict) else entry
        return v if isinstance(v, (int, float)) else None

    pool = value_of("pool_size")
    initial = value_of("initial_labeled")
    initial_name = "initial_labeled"

    # When the deriver marked initial_labeled unused (core-set bootstrap
    # methods: the initial labeled set IS construct_core_set's output), the
    # effective bootstrap size is core_set_size — comparing the unused knob
    # against the pool would false-fail a reachable loop (2026-06-11
    # adversarial-review catch).
    initial_entry = params.get("initial_labeled")
    if isinstance(initial_entry, dict) \
            and initial_entry.get("used_in_notebook") is False:
        if value_of("core_set_size") is not None:
            initial = value_of("core_set_size")
            initial_name = "core_set_size"
        else:
            return ProbeVerdict(
                "AL-6", "unprobeable",
                "initial_labeled is marked unused (core-set bootstrap) and "
                "no numeric core_set_size entry exists — the effective "
                "bootstrap size is not derivable from params.json",
                tier="static")

    if pool is None or initial is None:
        missing = [n for n, v in (("pool_size", pool), (initial_name, initial))
                   if v is None]
        return ProbeVerdict(
            "AL-6", "unprobeable",
            f"params.json lacks numeric {' and '.join(missing)} — "
            f"loop reachability not derivable", tier="static")

    if initial >= pool:
        return ProbeVerdict(
            "AL-6", "fail",
            f"the acquisition loop is unreachable at the delivered demo "
            f"config: {initial_name}={initial:g} >= pool_size={pool:g}, so "
            f"the unlabeled pool is empty before round 1 and every "
            f"acquisition call returns nothing — the demo never exercises "
            f"the method's selection stage",
            tier="static",
            evidence=f"{initial_name}={initial:g}, pool_size={pool:g}",
            finding_class="M-004")

    # Per-round acquisition: the delivered runtime contract has exactly one
    # output-count knob, batch_size. Historical GBALD drafts used
    # batch_outputs/batch_output for b', but current params should demote any
    # smoke-scale reduction into batch_size with paper_value retained.
    per_round = value_of("batch_size")
    legacy_output = value_of("batch_output")
    if legacy_output is None:
        legacy_output = value_of("batch_outputs")
    if per_round is None and legacy_output is not None:
        return ProbeVerdict(
            "AL-6", "fail",
            "params.json uses a legacy batch_output(s) output-count knob without "
            "a runtime batch_size. Active-learning demos must put the actual "
            "per-round acquisition count in batch_size; batch_returns may remain "
            "a larger candidate prefilter.",
            tier="static",
            finding_class="M-002")
    rounds = value_of("num_rounds")
    if per_round and rounds:
        remaining = pool - initial
        if rounds * per_round > remaining:
            full_rounds = int(remaining // per_round)
            return ProbeVerdict(
                "AL-6", "warn",
                f"the demo pool exhausts early: {rounds:g} rounds x "
                f"{per_round:g} acquisitions need {rounds * per_round:g} "
                f"unlabeled samples but only {remaining:g} exist after the "
                f"initial labels — rounds beyond ~{full_rounds + 1} acquire "
                f"nothing",
                tier="static",
                evidence=(f"pool_size={pool:g}, {initial_name}={initial:g}, "
                          f"num_rounds={rounds:g}, per_round={per_round:g}"))

    return ProbeVerdict(
        "AL-6", "pass",
        f"the delivered demo config exercises the acquisition loop "
        f"({initial_name}={initial:g} < pool_size={pool:g}"
        + (f"; budget {rounds * per_round:g} fits the remaining "
           f"{pool - initial:g})" if per_round and rounds else ")"),
        tier="static")


# ---------------------------------------------------------------------------
# AL-3 — fresh-retrain gate, registered under its catalog id (static)
# ---------------------------------------------------------------------------


def probe_fresh_retrain_static(nb_path, protocol: str = "") -> ProbeVerdict | None:
    """AL-3: the acquisition loop must build a FRESH model each round.

    Subsumes, never duplicates: calls the stage validator's own AST gate
    (probes.nb_ast._al_loop_warm_starts), which the driver also
    runs at the notebook stages — this registers the same check under its
    catalog id so battery reports and the zoo matrix can key on it. AL-4 is
    the behavioral complement (fingerprinted stub models in the sandbox).

    Returns None (not applicable) when the paper's training protocol
    deliberately warm-starts — the same rare exception the validator skips.
    """
    if "warm" in (protocol or "").lower():
        return None
    from probes.nb_ast import _al_loop_warm_starts  # noqa: PLC0415

    nb_path = Path(nb_path)
    if not nb_path.is_file():
        return ProbeVerdict("AL-3", "unprobeable", "no notebook.ipynb",
                            tier="static")
    try:
        nb = json.loads(nb_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        return ProbeVerdict("AL-3", "unprobeable",
                            f"notebook unparseable: {e}", tier="static")
    code_cells = ["".join(c.get("source", [])) for c in nb.get("cells", [])
                  if c.get("cell_type") == "code"]
    warm = _al_loop_warm_starts(code_cells, train_fn="train_from_scratch",
                                build_fn="build_model")
    if warm is True:
        return ProbeVerdict(
            "AL-3", "fail",
            "the acquisition loop calls train_from_scratch(...) inside the "
            "round loop but never build_model(...) there — one model object "
            "is reused across rounds (warm-starting), while "
            "retrain-from-scratch active learning requires a fresh model "
            "every round",
            tier="static", evidence=str(nb_path), finding_class="M-003")
    if warm is False:
        return ProbeVerdict(
            "AL-3", "pass",
            "in-loop training builds a fresh model each round "
            "(fresh-weights behavior is AL-4's complementary check)",
            tier="static")
    return ProbeVerdict(
        "AL-3", "unprobeable",
        "no in-loop training found in the notebook's code cells — the "
        "fresh-retrain gate has nothing to check (bootstrap-only training "
        "is legitimate)", tier="static")


def probe_eval_label_alignment_static(
    nb_path,
    *,
    pluggable_name: str = "select_batch",
) -> ProbeVerdict:
    """AL-7: learning-curve budget points must evaluate the matching model.

    Subsumes the notebook validator's AST gate: a point at post-acquisition
    label count must come from a model retrained after that acquisition, not
    from the pre-merge scoring model. This failure is silent at smoke time
    because the curve still has the right shape.
    """
    from probes.nb_ast import (  # noqa: PLC0415
        _al_loop_has_postmerge_count_for_premerge_eval,
    )

    nb_path = Path(nb_path)
    if not nb_path.is_file():
        return ProbeVerdict("AL-7", "unprobeable", "no notebook.ipynb",
                            tier="static")
    try:
        nb = json.loads(nb_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        return ProbeVerdict("AL-7", "unprobeable",
                            f"notebook unparseable: {e}", tier="static")
    code_cells = ["".join(c.get("source", [])) for c in nb.get("cells", [])
                  if c.get("cell_type") == "code"]
    if _al_loop_has_postmerge_count_for_premerge_eval(
        code_cells,
        pluggable_name=pluggable_name,
    ):
        return ProbeVerdict(
            "AL-7", "fail",
            "the acquisition loop appends a learning-curve point after "
            "merging new labels but before retraining/evaluating on that "
            "merged labeled set, so the plotted label budget belongs to a "
            "different model than the recorded accuracy",
            tier="static", evidence=str(nb_path),
            finding_class="M-002")
    return ProbeVerdict(
        "AL-7", "pass",
        "post-acquisition learning-curve points are retrained/evaluated "
        "after the label merge", tier="static")
