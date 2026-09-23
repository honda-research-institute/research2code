"""Zoo acceptance tests for the AL loop microharness (AL-1/AL-2/AL-4).

The matrix rows: the corrected badge loop and the gbald loop must PASS the
invariants under adversarial stub selections; the badge-badloop mutant must
FAIL them (M-002) and FAIL fresh-model detection (M-003). Requires torch
(the real loop cells use torch ops); CI skips.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

REPO = Path(__file__).resolve().parent.parent
ZOO = REPO / "tests" / "fixtures" / "zoo"

from probes.al_loop import run_al_loop_microharness  # noqa: E402

pytestmark = pytest.mark.probe_runtime


def _by_id(verdicts, probe_id):
    matching = [v for v in verdicts if v.probe_id == probe_id]
    assert matching, f"no {probe_id} in {[(v.probe_id, v.verdict) for v in verdicts]}"
    return matching[0]


def test_corrected_badge_loop_passes_invariants():
    verdicts = run_al_loop_microharness(
        ZOO / "badge-goodmethod-badnotebook" / "notebook.ipynb")
    al1 = _by_id(verdicts, "AL-1")
    assert al1.verdict == "pass", al1.message
    al4 = _by_id(verdicts, "AL-4")
    assert al4.verdict == "pass", al4.message


def test_badloop_mutant_fails_m002_and_m003():
    verdicts = run_al_loop_microharness(
        ZOO / "badge-badloop-reconstructed" / "notebook.ipynb")
    al1 = _by_id(verdicts, "AL-1")
    assert al1.verdict == "fail", al1.message
    assert al1.finding_class == "M-002"
    al4 = _by_id(verdicts, "AL-4")
    assert al4.verdict == "fail", al4.message
    assert "warm-start" in al4.message
    al2 = _by_id(verdicts, "AL-2")
    assert al2.verdict == "warn"


def test_gbald_loop_passes_invariants():
    # The lr=7.4 run's loop BOOKKEEPING is correct (zoo README) — different
    # variable names (core_set_indices, remaining_indices), GBALD-specific
    # pluggable kwargs. The harness must adapt via name-agnostic seeding.
    verdicts = run_al_loop_microharness(
        ZOO / "gbald-lr74-never-learns" / "notebook.ipynb")
    al1 = _by_id(verdicts, "AL-1")
    assert al1.verdict == "pass", al1.message


def test_double_retrain_loop_shape_is_not_a_growth_failure():
    # The fresh 2026-06-10 GBALD notebook retrains from scratch BEFORE
    # acquisition (scoring model) and AFTER it (learning-curve model), so
    # consecutive train events alternate growth 0 and +batch_size. The
    # strict every-step-grows invariant false-positived on this known-good
    # loop during the matrix-row audit; growth steps of 0 are loop-shape
    # variation, not a defect. Also needs `n_features` in the seeding table
    # (this cell builds models with it).
    verdicts = run_al_loop_microharness(
        ZOO / "gbald-double-retrain-loop" / "notebook.ipynb")
    al1 = _by_id(verdicts, "AL-1")
    assert al1.verdict == "pass", al1.message
    al4 = _by_id(verdicts, "AL-4")
    assert al4.verdict == "pass", al4.message


def test_loop_with_prior_cell_rng_passes_invariants(tmp_path):
    # A notebook loop cell may legitimately use `rng` initialized in an earlier
    # setup cell. The microharness seeds that conventional cross-cell state so
    # runnable notebooks do not become unprobeable only because the loop cell is
    # extracted in isolation.
    nb = _nb(tmp_path, """
labeled_idx = rng.choice(len(x_pool), size=8, replace=False)
unlabeled_idx = np.setdiff1d(np.arange(len(x_pool)), labeled_idx)
model = build_model(input_dim=input_dim, n_classes=n_classes)
model = train_from_scratch(model, x_pool[labeled_idx], y_pool[labeled_idx])
for round_idx in range(cfg["num_rounds"]):
    x_unlabeled = x_pool[unlabeled_idx]
    positions = select_batch(
        model,
        x_unlabeled,
        batch_size=cfg["batch_size"],
        seed=SEED + round_idx,
    )
    chosen = unlabeled_idx[np.asarray(positions)]
    labeled_idx = np.sort(np.concatenate([labeled_idx, chosen]))
    unlabeled_idx = np.setdiff1d(unlabeled_idx, chosen)
    model = build_model(input_dim=input_dim, n_classes=n_classes)
    model = train_from_scratch(model, x_pool[labeled_idx], y_pool[labeled_idx])
""")

    verdicts = run_al_loop_microharness(nb)

    al1 = _by_id(verdicts, "AL-1")
    assert al1.verdict == "pass", al1.message
    al4 = _by_id(verdicts, "AL-4")
    assert al4.verdict == "pass", al4.message


def test_loop_with_all_keyword_pluggable_call_is_probeable(tmp_path):
    # A faithful loop may call the pluggable with ALL keyword arguments. The
    # sandbox stub used to read the unlabeled pool as the second POSITIONAL arg,
    # so an all-keyword call left the positional tuple empty and raised
    # IndexError -> AL-1 unprobeable (the 2026-06-30 GBALD acceptance run). The
    # stub now resolves the pool from keyword-or-position. Same bookkeeping as
    # the passing loop above, only the call convention differs.
    nb = _nb(tmp_path, """
labeled_idx = rng.choice(len(x_pool), size=8, replace=False)
unlabeled_idx = np.setdiff1d(np.arange(len(x_pool)), labeled_idx)
model = build_model(input_dim=input_dim, n_classes=n_classes)
model = train_from_scratch(model, x_pool[labeled_idx], y_pool[labeled_idx])
for round_idx in range(cfg["num_rounds"]):
    x_unlabeled = x_pool[unlabeled_idx]
    positions = select_batch(
        model=model,
        x_unlabeled=x_unlabeled,
        batch_size=cfg["batch_size"],
        seed=SEED + round_idx,
    )
    chosen = unlabeled_idx[np.asarray(positions)]
    labeled_idx = np.sort(np.concatenate([labeled_idx, chosen]))
    unlabeled_idx = np.setdiff1d(unlabeled_idx, chosen)
    model = build_model(input_dim=input_dim, n_classes=n_classes)
    model = train_from_scratch(model, x_pool[labeled_idx], y_pool[labeled_idx])
""")

    verdicts = run_al_loop_microharness(nb)

    al1 = _by_id(verdicts, "AL-1")
    assert al1.verdict == "pass", al1.message


def test_loop_calling_tolist_on_index_seed_is_probeable(tmp_path):
    # A faithful loop may keep its label indices as a tensor/array and call
    # `.tolist()` on the seeded index name. The sandbox seeded `*_idx` names as
    # a plain list, so `.tolist()` raised AttributeError -> AL-1 unprobeable
    # (the 2026-06-30 BADGE acceptance run). Index names now seed as a list
    # subclass that also answers `.tolist()`. Same bookkeeping as the passing
    # loop above, only the first line exercises the tensor idiom on the seed.
    nb = _nb(tmp_path, """
labeled_idx = np.array(bootstrap_labeled_idx.tolist())
unlabeled_idx = np.setdiff1d(np.arange(len(x_pool)), labeled_idx)
model = build_model(input_dim=input_dim, n_classes=n_classes)
model = train_from_scratch(model, x_pool[labeled_idx], y_pool[labeled_idx])
for round_idx in range(cfg["num_rounds"]):
    x_unlabeled = x_pool[unlabeled_idx]
    positions = select_batch(model, x_unlabeled, batch_size=cfg["batch_size"], seed=SEED + round_idx)
    chosen = unlabeled_idx[np.asarray(positions)]
    labeled_idx = np.sort(np.concatenate([labeled_idx, chosen]))
    unlabeled_idx = np.setdiff1d(unlabeled_idx, chosen)
    model = build_model(input_dim=input_dim, n_classes=n_classes)
    model = train_from_scratch(model, x_pool[labeled_idx], y_pool[labeled_idx])
""")

    verdicts = run_al_loop_microharness(nb)

    al1 = _by_id(verdicts, "AL-1")
    assert al1.verdict == "pass", al1.message


def test_loop_reading_model_training_attr_is_probeable(tmp_path):
    # A faithful loop may save/restore the model's train mode via the standard
    # nn.Module `training` attribute (`was_train = model.training`). The
    # sandbox model exposed train()/eval() methods but no `training` bool, so
    # the read raised AttributeError -> AL-1 unprobeable (the 2026-07-02 GBALD
    # acceptance run). The stub now carries `training`, kept consistent by
    # train()/eval(). Same bookkeeping as the passing loop above, only the
    # mode save/restore lines differ.
    nb = _nb(tmp_path, """
labeled_idx = rng.choice(len(x_pool), size=8, replace=False)
unlabeled_idx = np.setdiff1d(np.arange(len(x_pool)), labeled_idx)
model = build_model(input_dim=input_dim, n_classes=n_classes)
model = train_from_scratch(model, x_pool[labeled_idx], y_pool[labeled_idx])
for round_idx in range(cfg["num_rounds"]):
    was_train = model.training
    model.eval()
    x_unlabeled = x_pool[unlabeled_idx]
    positions = select_batch(model, x_unlabeled, batch_size=cfg["batch_size"], seed=SEED + round_idx)
    if was_train:
        model.train()
    chosen = unlabeled_idx[np.asarray(positions)]
    labeled_idx = np.sort(np.concatenate([labeled_idx, chosen]))
    unlabeled_idx = np.setdiff1d(unlabeled_idx, chosen)
    model = build_model(input_dim=input_dim, n_classes=n_classes)
    model = train_from_scratch(model, x_pool[labeled_idx], y_pool[labeled_idx])
""")

    verdicts = run_al_loop_microharness(nb)

    al1 = _by_id(verdicts, "AL-1")
    assert al1.verdict == "pass", al1.message


def test_loop_with_unknown_sampler_remains_unprobeable(tmp_path):
    nb = _nb(tmp_path, """
labeled_idx = sampler.choice(len(x_pool), size=8, replace=False)
unlabeled_idx = np.setdiff1d(np.arange(len(x_pool)), labeled_idx)
model = build_model(input_dim=input_dim, n_classes=n_classes)
for round_idx in range(cfg["num_rounds"]):
    positions = select_batch(model, x_pool[unlabeled_idx], cfg["batch_size"], SEED + round_idx)
    chosen = unlabeled_idx[np.asarray(positions)]
    labeled_idx = np.concatenate([labeled_idx, chosen])
    unlabeled_idx = np.setdiff1d(unlabeled_idx, chosen)
    model = train_from_scratch(model, x_pool[labeled_idx], y_pool[labeled_idx])
""")

    verdicts = run_al_loop_microharness(nb)

    al1 = _by_id(verdicts, "AL-1")
    assert al1.verdict == "unprobeable"
    assert "sampler" in al1.message


def test_notebook_without_loop_is_unprobeable(tmp_path):
    nb = {"cells": [{"cell_type": "code", "id": "a",
                     "source": ["x = 1\n"], "outputs": []}]}
    p = tmp_path / "nb.ipynb"
    p.write_text(json.dumps(nb))
    verdicts = run_al_loop_microharness(p)
    assert verdicts[0].verdict == "unprobeable"


# ---------------------------------------------------------------------------
# AL-7 evaluation/label-budget alignment (static)
# ---------------------------------------------------------------------------


def _nb(tmp_path, source: str):
    path = tmp_path / "notebook.ipynb"
    path.write_text(json.dumps({
        "cells": [{"cell_type": "code", "id": "loop",
                   "source": [source], "outputs": []}]
    }), encoding="utf-8")
    return path


def test_al7_fails_premerge_eval_with_postmerge_label_count(tmp_path):
    from probes.al_loop import probe_eval_label_alignment_static

    v = probe_eval_label_alignment_static(_nb(tmp_path, """
learning_curve = []
for r in range(cfg["num_rounds"]):
    model = build_model(input_dim=input_dim, n_classes=n_classes)
    model = train_from_scratch(model, x_pool[labeled_idx], y_pool[labeled_idx], seed=SEED + r)
    test_acc = evaluate(model, x_test, y_test)
    selected = select_batch(model, x_pool[unlabeled_idx], cfg["batch_size"], SEED + r)
    chosen = unlabeled_idx[np.asarray(selected)]
    labeled_idx = np.union1d(labeled_idx, chosen)
    unlabeled_idx = np.setdiff1d(unlabeled_idx, chosen)
    learning_curve.append((len(labeled_idx), test_acc))
"""))
    assert v.verdict == "fail"
    assert v.probe_id == "AL-7"
    assert v.finding_class == "M-002"


def test_al7_passes_postmerge_retrain_before_eval(tmp_path):
    from probes.al_loop import probe_eval_label_alignment_static

    v = probe_eval_label_alignment_static(_nb(tmp_path, """
learning_curve = []
for r in range(cfg["num_rounds"]):
    selected = select_batch(model, x_pool[unlabeled_idx], cfg["batch_size"], SEED + r)
    chosen = unlabeled_idx[np.asarray(selected)]
    labeled_idx = np.union1d(labeled_idx, chosen)
    unlabeled_idx = np.setdiff1d(unlabeled_idx, chosen)
    model = build_model(input_dim=input_dim, n_classes=n_classes)
    model = train_from_scratch(model, x_pool[labeled_idx], y_pool[labeled_idx], seed=SEED + r)
    test_acc = evaluate(model, x_test, y_test)
    learning_curve.append((len(labeled_idx), test_acc))
"""))
    assert v.verdict == "pass", v.message


# ---------------------------------------------------------------------------
# AL-5 acquisition-output contract (selector-level)
# ---------------------------------------------------------------------------


def _selector_module(tmp_path, body: str, name: str):
    from probes.package_loader import load_module_from_path
    path = tmp_path / name
    path.write_text(body, encoding="utf-8")
    return load_module_from_path(path)


def test_al5_passes_june9_gbald_selector():
    pytest.importorskip("sklearn")
    from probes.al_loop import probe_acquisition_contract
    from probes.package_loader import load_module_from_path
    mod = load_module_from_path(
        REPO / "tests" / "fixtures" / "evidence" / "june9-gbald-run"
        / "method" / "method.py")
    v = probe_acquisition_contract(mod, "select_batch")
    assert v.verdict == "pass", f"{v.verdict}: {v.message}"


def test_al5_fails_on_duplicate_indices(tmp_path):
    from probes.al_loop import probe_acquisition_contract
    mod = _selector_module(tmp_path, (
        "def select_batch(model, x_unlabeled, batch_size, seed):\n"
        "    return [0] * batch_size\n"), "dupes.py")
    v = probe_acquisition_contract(mod, "select_batch")
    assert v.verdict == "fail"
    assert v.finding_class == "M-002"
    assert "duplicate" in v.message


def test_al5_fails_on_out_of_range_indices(tmp_path):
    from probes.al_loop import probe_acquisition_contract
    mod = _selector_module(tmp_path, (
        "def select_batch(model, x_unlabeled, batch_size, seed):\n"
        "    n = len(x_unlabeled)\n"
        "    return [n + i for i in range(batch_size)]\n"), "oob.py")
    v = probe_acquisition_contract(mod, "select_batch")
    assert v.verdict == "fail"
    assert "outside the unlabeled pool" in v.message


def test_al5_fails_when_selector_returns_batch_output(tmp_path):
    from probes.al_loop import probe_acquisition_contract
    mod = _selector_module(tmp_path, (
        "def select_batch(model, x_unlabeled, batch_size, seed, batch_returns=16, batch_output=10):\n"
        "    return list(range(batch_output))\n"), "batch_output.py")
    v = probe_acquisition_contract(mod, "select_batch")
    assert v.verdict == "fail"
    assert "returned 10 indices" in v.message
    assert "batch_size is 4" in v.message


def test_al5_passes_when_selector_returns_exact_batch_size(tmp_path):
    from probes.al_loop import probe_acquisition_contract
    mod = _selector_module(tmp_path, (
        "import torch\n"
        "def select_batch(model, x_unlabeled, batch_size, seed, batch_returns=16):\n"
        "    with torch.no_grad():\n"
        "        scores = model(x_unlabeled).sum(dim=1)\n"
        "    return torch.topk(scores, k=batch_size).indices.tolist()\n"), "exact_batch.py")
    v = probe_acquisition_contract(mod, "select_batch")
    assert v.verdict == "pass", v.message


def test_al5_flags_model_insensitive_selector(tmp_path):
    from probes.al_loop import probe_acquisition_contract
    # Contract-clean but ignores the model entirely: unique in-range indices,
    # same selection under different weights — even with TRAINED stubs, since it
    # never reads the model. Researcher adjudicates (a purely geometric method
    # can be paper-faithful).
    mod = _selector_module(tmp_path, (
        "def select_batch(model, x_unlabeled, batch_size, seed):\n"
        "    return list(range(batch_size))\n"), "geometric.py")
    v = probe_acquisition_contract(mod, "select_batch")
    assert v.verdict == "flag_for_researcher"
    assert v.finding_class == "M-004"
    assert "model" in v.message


def test_al5_uncertainty_selector_passes_with_trained_stub(tmp_path):
    """An uncertainty-based selector (predictive entropy) is degenerate under an
    UNTRAINED stub (uniform predictions -> ~equal entropy -> identical pick) but
    model-sensitive once the stub is trained. The sensitivity arm now trains the
    stub, so a faithful uncertainty selector passes instead of being
    false-flagged as model-insensitive (the GBALD acceptance-run case)."""
    pytest.importorskip("torch")
    from probes.al_loop import probe_acquisition_contract
    mod = _selector_module(tmp_path, (
        "import torch\n"
        "import torch.nn.functional as F\n"
        "def select_batch(model, x_unlabeled, batch_size, seed, batch_returns=16):\n"
        "    model.train()\n"
        "    with torch.no_grad():\n"
        "        probs = torch.stack([F.softmax(model(x_unlabeled), dim=1)\n"
        "                             for _ in range(8)]).mean(0)\n"
        "    ent = -(probs * probs.clamp_min(1e-12).log()).sum(1)\n"
        "    return torch.topk(ent, k=batch_size).indices.tolist()\n"),
        "uncertainty.py")
    v = probe_acquisition_contract(mod, "select_batch")
    assert v.verdict == "pass", f"{v.verdict}: {v.message}"


def test_al5_make_trained_model_is_nondegenerate_and_distinct(tmp_path):
    """The sensitivity arm's trained stubs must be (a) non-degenerate (confident,
    not uniform) and (b) distinct across seeds. Otherwise the arm can't tell a
    model-consuming selector from a model-ignoring one."""
    torch = pytest.importorskip("torch")
    import torch.nn.functional as F
    from probes.term_ablation import al_selector_kit
    mod = _selector_module(tmp_path, (
        "def select_batch(model, x_unlabeled, batch_size, seed, batch_returns=16):\n"
        "    return list(range(batch_size))\n"), "stub_sel.py")
    kit = al_selector_kit(mod, "select_batch", 0)
    assert "make_trained_model" in kit
    xu = kit["kwargs"]["x_unlabeled"]
    ma, mb = kit["make_trained_model"](0), kit["make_trained_model"](101)
    ma.eval(); mb.eval()
    with torch.no_grad():
        pa, pb = F.softmax(ma(xu), dim=1), F.softmax(mb(xu), dim=1)
    uniform = 1.0 / kit["n_classes"]
    assert pa.max(dim=1).values.mean().item() > uniform + 0.05  # confident, not uniform
    assert not torch.allclose(pa, pb, atol=1e-3)  # distinct across seeds


def test_infix_index_names_are_probeable(tmp_path):
    # GBALD 2026-07-03 roll: the notebook's bootstrap indices were named
    # `labeled_idx_bootstrap` / `unlabeled_idx_bootstrap` — "idx" in the
    # MIDDLE of the identifier fell through the old endswith("_idx") rule
    # and AL-1 came back unprobeable on naming variance alone. Index-name
    # seeding now matches idx/index/indices anywhere in the name, keeping
    # the labeled/unlabeled disjointness rule.
    nb = _nb(tmp_path, """
labeled_idx = np.array(labeled_idx_bootstrap)
unlabeled_idx = np.array(unlabeled_idx_bootstrap)
model = build_model(input_dim=input_dim, n_classes=n_classes)
model = train_from_scratch(model, x_pool[labeled_idx], y_pool[labeled_idx])
for round_idx in range(cfg["num_rounds"]):
    x_unlabeled = x_pool[unlabeled_idx]
    positions = select_batch(model, x_unlabeled, batch_size=cfg["batch_size"], seed=SEED + round_idx)
    chosen = unlabeled_idx[np.asarray(positions)]
    labeled_idx = np.sort(np.concatenate([labeled_idx, chosen]))
    unlabeled_idx = np.setdiff1d(unlabeled_idx, chosen)
    model = build_model(input_dim=input_dim, n_classes=n_classes)
    model = train_from_scratch(model, x_pool[labeled_idx], y_pool[labeled_idx])
""")

    verdicts = run_al_loop_microharness(nb)

    al1 = _by_id(verdicts, "AL-1")
    assert al1.verdict == "pass", al1.message
