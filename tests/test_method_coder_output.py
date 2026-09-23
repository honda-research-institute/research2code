from __future__ import annotations

import ast

from validate_method_coder_output import (
    _active_learning_selector_static_errors,
    _dead_remaining_quota_loops,
    _find_function_def,
    _global_seed_mutation_calls,
    _hidden_unreturned_pluggable_acquisitions,
    _inverse_distance_descending_prose_contradictions,
    _pluggable_signature_contract_errors,
    _quadratic_kmeanspp_distance_broadcasts,
)
from validate_architecture_coder_output import _extract_paper_element_ids


def test_dead_remaining_quota_loop_detected():
    src = """
def construct_core_set(n_coreset):
    selected = []
    for _ in range(n_coreset):
        selected.append(0)
    n_remaining_to_acquire = n_coreset - len(selected)
    for _ in range(n_remaining_to_acquire):
        selected.append(1)
    return selected
"""
    hits = _dead_remaining_quota_loops(ast.parse(src))
    assert hits
    assert "construct_core_set" in hits[0][1]
    assert "n_remaining_to_acquire" in hits[0][1]


def test_paper_element_parser_accepts_plus_in_ids(tmp_path):
    path = tmp_path / "method.py"
    path.write_text("def f():\n    # paper-element: alg-kmeans++\n    pass\n")

    assert _extract_paper_element_ids(path) == [(2, "alg-kmeans++")]


def test_pluggable_signature_contract_rejects_case_mismatch():
    src = """
def select_batch(model, X_unlabeled, batch_size, seed):
    return []
"""
    tree = ast.parse(src)
    fn = _find_function_def(tree, "select_batch")

    errors = _pluggable_signature_contract_errors(
        pluggable_name="select_batch",
        expected_sig="select_batch(model, x_unlabeled, batch_size, seed) -> List[int]",
        actual_fn=fn,
    )

    assert any("missing spec parameter" in error for error in errors)
    assert any("extra required parameter" in error for error in errors)
    assert any("X_unlabeled" in error for error in errors)


def test_pluggable_signature_contract_allows_optional_extras():
    src = """
def select_batch(model, x_unlabeled, batch_size, seed, temperature=1.0):
    return []
"""
    tree = ast.parse(src)
    fn = _find_function_def(tree, "select_batch")

    assert _pluggable_signature_contract_errors(
        pluggable_name="select_batch",
        expected_sig="select_batch(model, x_unlabeled, batch_size, seed) -> List[int]",
        actual_fn=fn,
    ) == []


def test_remaining_quota_loop_allows_distinct_initialization_quota():
    src = """
def construct_core_set(n_initial, n_coreset):
    selected = []
    for _ in range(n_initial):
        selected.append(0)
    n_remaining_to_acquire = n_coreset - len(selected)
    for _ in range(n_remaining_to_acquire):
        selected.append(1)
    return selected
"""
    assert _dead_remaining_quota_loops(ast.parse(src)) == []


def test_remaining_quota_loop_allows_conditional_fill_shape():
    src = """
def construct_core_set(n_coreset):
    selected = []
    for _ in range(n_coreset):
        if should_add():
            selected.append(0)
    n_remaining_to_acquire = n_coreset - len(selected)
    for _ in range(n_remaining_to_acquire):
        selected.append(1)
    return selected
"""
    assert _dead_remaining_quota_loops(ast.parse(src)) == []


def test_inverse_distance_descending_farther_prose_detected():
    src = '''
def rank_representative(x_pool, centers, R_0):
    """Farther samples receive higher scores and are preferred."""
    distances = pairwise_distance(x_pool, centers)
    min_distance = distances.min(axis=1)
    scores = R_0 / (min_distance + 1e-8)
    return np.argsort(scores)[::-1]
'''
    hits = _inverse_distance_descending_prose_contradictions(ast.parse(src), src)
    assert hits
    assert "rank_representative" in hits[0][1]


def test_inverse_distance_descending_far_from_prose_detected():
    src = '''
def rank_representative(x_pool, centers, R_0):
    """Select samples far from centers with high R_0 / distance scores."""
    distances = pairwise_distance(x_pool, centers)
    min_distance = distances.min(axis=1)
    scores = R_0 / (min_distance + 1e-8)
    return np.argsort(scores)[::-1]
'''
    hits = _inverse_distance_descending_prose_contradictions(ast.parse(src), src)
    assert hits
    assert "rank_representative" in hits[0][1]


def test_inverse_distance_descending_closer_prose_passes():
    src = '''
def rank_representative(x_pool, centers, R_0):
    """Closer samples receive higher scores under inverse distance."""
    distances = pairwise_distance(x_pool, centers)
    min_distance = distances.min(axis=1)
    scores = R_0 / (min_distance + 1e-8)
    return np.argsort(scores)[::-1]
'''
    assert _inverse_distance_descending_prose_contradictions(ast.parse(src), src) == []


def test_hidden_unreturned_pluggable_acquisitions_detected():
    src = """
def select_batch(model, x_unlabeled, x_labeled, batch_size, seed):
    labeled_len = len(x_labeled)
    x_pool = torch.cat([x_labeled, x_unlabeled], dim=0)
    core_set_indices = construct_core_set(x_pool, seed)
    new_labeled_indices = [
        idx - labeled_len for idx in core_set_indices if idx >= labeled_len
    ]
    x_labeled = torch.cat([x_labeled, x_unlabeled[new_labeled_indices]], dim=0)
    remaining_unlabeled_indices = [
        i for i in range(len(x_unlabeled)) if i not in new_labeled_indices
    ]
    x_unlabeled = x_unlabeled[remaining_unlabeled_indices]
    selected_positions = rank_candidates(model, x_unlabeled, batch_size)
    return [remaining_unlabeled_indices[i] for i in selected_positions]
"""
    tree = ast.parse(src)
    fn = _find_function_def(tree, "select_batch")
    hits = _hidden_unreturned_pluggable_acquisitions(fn)
    assert hits
    assert "new_labeled_indices" in hits[0][1]


def test_hidden_pluggable_acquisitions_allowed_when_returned():
    src = """
def select_batch(model, x_unlabeled, x_labeled, batch_size, seed):
    labeled_len = len(x_labeled)
    x_pool = torch.cat([x_labeled, x_unlabeled], dim=0)
    core_set_indices = construct_core_set(x_pool, seed)
    new_labeled_indices = [
        idx - labeled_len for idx in core_set_indices if idx >= labeled_len
    ]
    x_labeled = torch.cat([x_labeled, x_unlabeled[new_labeled_indices]], dim=0)
    selected_positions = rank_candidates(model, x_unlabeled, batch_size)
    all_positions = new_labeled_indices + selected_positions
    return all_positions
"""
    tree = ast.parse(src)
    fn = _find_function_def(tree, "select_batch")
    assert _hidden_unreturned_pluggable_acquisitions(fn) == []


def test_plain_pluggable_acquisition_passes_hidden_acquisition_check():
    src = """
def select_batch(model, x_unlabeled, x_labeled, batch_size, seed):
    scores = score_candidates(model, x_unlabeled, x_labeled)
    return torch.topk(scores, batch_size).indices.tolist()
"""
    tree = ast.parse(src)
    fn = _find_function_def(tree, "select_batch")
    assert _hidden_unreturned_pluggable_acquisitions(fn) == []


def test_active_learning_selector_rejects_batch_output_split():
    src = """
def select_batch(model, x_unlabeled, batch_size, seed, batch_returns=30, batch_output=10):
    scores = score_candidates(model, x_unlabeled)
    top = torch.topk(scores, batch_returns).indices
    return top[:batch_output].tolist()
"""
    tree = ast.parse(src)
    fn = _find_function_def(tree, "select_batch")
    errors = _active_learning_selector_static_errors(fn)
    assert any("batch_output" in error for error in errors)


def test_active_learning_selector_must_reference_batch_size():
    src = """
def select_batch(model, x_unlabeled, batch_size, seed):
    return [0, 1, 2, 3]
"""
    tree = ast.parse(src)
    fn = _find_function_def(tree, "select_batch")
    errors = _active_learning_selector_static_errors(fn)
    assert any("never references" in error for error in errors)


def test_kmeanspp_full_center_broadcast_detected():
    src = """
def kmeans_plus_plus_seeding(embeddings, k, rng):
    selected = [rng.integers(0, len(embeddings))]
    for _ in range(1, k):
        centers = embeddings[selected]
        diffs = embeddings[:, np.newaxis, :] - centers
        sq_dists = np.sum(diffs ** 2, axis=2)
        probs = sq_dists.min(axis=1)
        selected.append(rng.choice(len(embeddings), p=probs / probs.sum()))
    return selected
"""

    hits = _quadratic_kmeanspp_distance_broadcasts(ast.parse(src))

    assert hits
    assert "kmeans_plus_plus_seeding" in hits[0][1]


def test_kmeanspp_incremental_distance_update_passes():
    src = """
def kmeans_plus_plus_seeding(embeddings, k, rng):
    selected = [rng.integers(0, len(embeddings))]
    min_sq = np.sum((embeddings - embeddings[selected[0]]) ** 2, axis=1)
    for _ in range(1, k):
        new_sq = np.sum((embeddings - embeddings[selected[-1]]) ** 2, axis=1)
        min_sq = np.minimum(min_sq, new_sq)
        selected.append(rng.choice(len(embeddings), p=min_sq / min_sq.sum()))
    return selected
"""

    assert _quadratic_kmeanspp_distance_broadcasts(ast.parse(src)) == []


def test_ranking_saturation_cap_detected_where_shape():
    """R11 gate, the 2026-07-02 fresh-roll shape: a rank-named function
    capping within-radius scores at a constant via np.where collapses the
    ranking to input order (M-004)."""
    from validate_method_coder_output import _saturation_capped_rankings
    src = '''
import numpy as np
from scipy.spatial.distance import cdist

def compute_geometric_ranking(x_candidates, x_labeled, R_0=2000.0):
    dists = cdist(x_candidates, x_labeled, metric="euclidean")
    within = (dists <= R_0)
    prob_per_center = np.where(within, 1.0, R_0 / np.maximum(dists, 1e-12))
    geometric_prob = np.max(prob_per_center, axis=1)
    return -geometric_prob
'''
    hits = _saturation_capped_rankings(ast.parse(src))
    assert len(hits) == 1
    assert "compute_geometric_ranking" in hits[0][1]


def test_ranking_saturation_cap_detected_mask_assign_shape():
    """The 2026-06-30 audited shape: constant assigned through a
    comparison-derived mask inside a rank-named function."""
    from validate_method_coder_output import _saturation_capped_rankings
    src = '''
import numpy as np

def rank_by_representativeness(x, centers, R_0=2000.0):
    d = ((x[:, None] - centers[None]) ** 2).sum(-1) ** 0.5
    prob = R_0 / d
    within = d <= R_0
    prob[within] = 1.0
    return prob.max(1)
'''
    hits = _saturation_capped_rankings(ast.parse(src))
    assert len(hits) == 1
    assert "rank_by_representativeness" in hits[0][1]


def test_ranking_raw_ratio_passes_cap_gate():
    """The faithful shape (rank by raw min-distances, R_0 unused in the
    ordering) must pass."""
    from validate_method_coder_output import _saturation_capped_rankings
    src = '''
import numpy as np
from scipy.spatial.distance import cdist

def geometric_ranking(x_candidates, x_labeled, R_0=2000.0):
    dists = cdist(x_candidates, x_labeled)
    min_distances = dists.min(axis=1)
    return np.argsort(-min_distances)
'''
    assert _saturation_capped_rankings(ast.parse(src)) == []


def test_ranking_exclusion_mask_passes_cap_gate():
    """A validity/exclusion mask (where with a constant branch whose
    condition is NOT a parameter comparison) is a legit idiom, not a cap."""
    from validate_method_coder_output import _saturation_capped_rankings
    src = '''
import numpy as np

def rank_candidates(scores, valid):
    masked = np.where(valid, scores, -np.inf)
    return np.argsort(-masked)
'''
    assert _saturation_capped_rankings(ast.parse(src)) == []


def test_non_ranking_probability_cap_passes_cap_gate():
    """The Eq-5 cap is FAITHFUL where an actual probability is computed —
    only rank-named functions are gated."""
    from validate_method_coder_output import _saturation_capped_rankings
    src = '''
import numpy as np

def geometric_probability(dists, R_0=2000.0):
    return np.where(dists <= R_0, 1.0, R_0 / np.maximum(dists, 1e-12))
'''
    assert _saturation_capped_rankings(ast.parse(src)) == []

def test_bare_sibling_from_import_detected():
    """R2C-036 known-bad: the DomIndOnto 0728c shape — a bare `from model
    import ...` in method.py, which resolves in no real consumer."""
    from validate_method_coder_output import _bare_sibling_imports
    src = """
from model import build_model

def populate_kb(document, seed):
    return build_model(document)
"""
    hits = _bare_sibling_imports(ast.parse(src), {"model", "training", "data"})
    assert len(hits) == 1
    assert hits[0][0] == 2
    assert "from .model import build_model" in hits[0][1]


def test_bare_sibling_plain_import_detected():
    from validate_method_coder_output import _bare_sibling_imports
    src = """
import training

def run(seed):
    return training.train_model(seed)
"""
    hits = _bare_sibling_imports(ast.parse(src), {"model", "training"})
    assert len(hits) == 1
    assert "import training" in hits[0][1]


def test_relative_and_qualified_sibling_imports_pass():
    """R2C-036 known-good: both sanctioned forms pass untouched."""
    from validate_method_coder_output import _bare_sibling_imports
    src = """
from .model import build_model
from method.training import train_model
from . import data

def select_batch(model, x_unlabeled, batch_size, seed):
    return []
"""
    assert _bare_sibling_imports(ast.parse(src), {"model", "training", "data"}) == []


def test_third_party_imports_pass_sibling_gate():
    """A canonical-path module shape: third-party and stdlib imports never
    collide with the sibling set."""
    from validate_method_coder_output import _bare_sibling_imports
    src = """
import numpy as np
import torch
from collections import defaultdict

def compute_distillation_loss(batch, temperature, seed):
    return torch.zeros(())
"""
    assert _bare_sibling_imports(ast.parse(src), {"model", "training", "data"}) == []


def test_dotted_bare_sibling_import_detected():
    from validate_method_coder_output import _bare_sibling_imports
    src = "import tests.test_elements\n"
    hits = _bare_sibling_imports(ast.parse(src), {"model", "tests"})
    assert len(hits) == 1
    assert "tests" in hits[0][1]


# --- R2C-079: the package's own name is a sibling stem too ---------------------
#
# Every generated package contains method/method.py, because that is where the
# spec's pluggable lives, so a real run's sibling set ALWAYS contains "method".
# The known-good above omits it, which is exactly why the false positive
# shipped: the sanctioned `from method.model import X` was reported as a bare
# sibling import, and the suggested replacement (`from .method.model import X`)
# resolves to method.method.model, which raises. Observed live 2026-08-07.


def _REAL_SIBLINGS():
    """The sibling set a real run produces, package stem included."""
    return {"method", "model", "training", "data"}


def test_package_qualified_import_passes_with_the_package_stem_in_the_set():
    """The exact shape from the 2026-08-07 forecasting roll."""
    from validate_method_coder_output import _bare_sibling_imports
    src = "from method.model import ForecastResult\n"
    assert _bare_sibling_imports(ast.parse(src), _REAL_SIBLINGS()) == []


def test_package_qualified_plain_import_passes_with_the_package_stem_in_the_set():
    from validate_method_coder_output import _bare_sibling_imports
    src = "import method.training\n"
    assert _bare_sibling_imports(ast.parse(src), _REAL_SIBLINGS()) == []


def test_genuine_bare_sibling_still_fails_with_the_package_stem_in_the_set():
    """The guard must not blanket-pass: a truly bare sibling is still wrong."""
    from validate_method_coder_output import _bare_sibling_imports
    src = "from model import ForecastResult\n"
    hits = _bare_sibling_imports(ast.parse(src), _REAL_SIBLINGS())
    assert len(hits) == 1


def test_the_suggested_replacement_is_importable():
    """The MESSAGE is under test, not just the verdict.

    The suggestion is what the producer acts on, so a wrong one costs a fix
    iteration exactly as a wrong verdict does. No test asserted the suggested
    text before, which is how `from .method.model import ...` shipped."""
    from validate_method_coder_output import _bare_sibling_imports
    src = "from model import ForecastResult\n"
    (_, msg), = _bare_sibling_imports(ast.parse(src), _REAL_SIBLINGS())
    assert "`from .model import ForecastResult`" in msg
    # The broken suggestion this record exists for must never reappear.
    assert ".method.model" not in msg


def test_package_name_is_taken_from_the_directory_not_hardcoded():
    """A package dir named something other than `method` still recognises its
    own qualified form, so the guard is not a `method`-shaped special case."""
    from validate_method_coder_output import _bare_sibling_imports
    src = "from pkg.model import Thing\n"
    siblings = {"pkg", "model"}
    assert _bare_sibling_imports(ast.parse(src), siblings, "pkg") == []
    # ...and with a different package name, the same line IS a bare sibling.
    assert len(_bare_sibling_imports(ast.parse(src), siblings, "method")) == 1


def test_sibling_module_names_from_package_dir(tmp_path):
    from validate_method_coder_output import _sibling_module_names
    pkg = tmp_path / "method"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    (pkg / "model.py").write_text("")
    (pkg / "training.py").write_text("")
    subpkg = pkg / "tests"
    subpkg.mkdir()
    (subpkg / "__init__.py").write_text("")
    plain_dir = pkg / "artifacts"
    plain_dir.mkdir()

    assert _sibling_module_names(pkg) == {"model", "training", "tests"}
    assert _sibling_module_names(tmp_path / "missing") == set()


def _tree(src):
    return ast.parse(src)


def test_duplicate_public_class_across_producers_detected():
    """R2C-053, the pdfgnn shape: the forecast container dataclass defined in
    both model.py and method.py, with __init__.py exporting only one."""
    from validate_method_coder_output import duplicate_public_definitions

    duplicates = duplicate_public_definitions({
        "method/model.py": _tree(
            "class ProbabilisticForecast:\n"
            "    mean: int\n"
            "class GNNEncoder:\n    pass\n"
        ),
        "method/method.py": _tree(
            "class ProbabilisticForecast:\n"
            "    mean: float\n"
            "def forecast(model, data, horizon, seed):\n    return None\n"
        ),
    })

    assert list(duplicates) == ["ProbabilisticForecast"]
    assert duplicates["ProbabilisticForecast"] == [
        "method/method.py",
        "method/model.py",
    ]


def test_duplicate_public_function_across_producers_detected():
    from validate_method_coder_output import duplicate_public_definitions

    duplicates = duplicate_public_definitions({
        "method/model.py": _tree("def build_graph(x):\n    return x\n"),
        "method/method.py": _tree("def build_graph(x):\n    return x\n"),
    })
    assert list(duplicates) == ["build_graph"]


def test_importing_a_sibling_type_is_not_a_duplicate_definition():
    """The honest fix must pass: define once, import elsewhere."""
    from validate_method_coder_output import duplicate_public_definitions

    assert duplicate_public_definitions({
        "method/model.py": _tree("class ProbabilisticForecast:\n    mean: int\n"),
        "method/method.py": _tree(
            "from .model import ProbabilisticForecast\n"
            "def forecast(model, data, horizon, seed):\n"
            "    return ProbabilisticForecast()\n"
        ),
    }) == {}


def test_private_and_assignment_duplicates_are_not_flagged():
    """Two modules binding the same module-level CONSTANT is ordinary and
    carries no identity semantics; flagging it would burn a fix loop on noise.
    Private helpers are likewise each module's own business."""
    from validate_method_coder_output import duplicate_public_definitions

    assert duplicate_public_definitions({
        "method/model.py": _tree(
            "DEFAULT_DIM = 8\n"
            "def _helper(x):\n    return x\n"
            "class _Cache:\n    pass\n"
        ),
        "method/method.py": _tree(
            "DEFAULT_DIM = 8\n"
            "def _helper(x):\n    return x\n"
            "class _Cache:\n    pass\n"
        ),
    }) == {}


def test_same_name_in_one_module_only_is_not_a_duplicate():
    from validate_method_coder_output import duplicate_public_definitions

    assert duplicate_public_definitions({
        "method/model.py": _tree("class Model:\n    pass\n"),
        "method/training.py": _tree("def train_model(m):\n    return m\n"),
    }) == {}


def test_fedavg_delivered_shape_is_the_second_concrete_case():
    """R2C-053's second case, found in a DELIVERED run rather than constructed.
    fedavg defines federated_train in both method.py (returning TrainedModel,
    the spec's promised pluggable) and training.py (returning a plain dict),
    and __init__.py exports the training.py one. Two signatures and two return
    types under one public name."""
    from validate_method_coder_output import duplicate_public_definitions

    duplicates = duplicate_public_definitions({
        "method/method.py": _tree(
            "def federated_train(model, client_simulator, num_rounds, C, E, B,\n"
            "                    learning_rate, seed):\n"
            "    return TrainedModel()\n"
        ),
        "method/training.py": _tree(
            "def federated_train(model, client_simulator, num_rounds, C=0.1,\n"
            "                    E=5, B=10, learning_rate=0.01, seed=42):\n"
            "    return {}\n"
        ),
    })

    assert duplicates == {
        "federated_train": ["method/method.py", "method/training.py"]
    }


# ---------------------------------------------------------------------------
# rng_threading — global RNG mutation in the library module (check 7b)
# ---------------------------------------------------------------------------


def test_global_seed_mutation_flagged_in_library_function():
    """The pdfgnn 2026-08-04 fidelity demoter: torch.manual_seed inside a
    per-call sampling function derails the caller's global RNG."""
    src = """
import torch
def sample_from_t_distribution(mu, scale, df, *, seed):
    torch.manual_seed(seed)
    return torch.randn(3)
"""
    hits = _global_seed_mutation_calls(ast.parse(src))
    assert [chain for _, chain in hits] == ["torch.manual_seed"]


def test_global_seed_mutation_covers_numpy_and_stdlib():
    src = """
import random
import numpy as np
def f(seed):
    np.random.seed(seed)
    random.seed(seed)
    return 1
"""
    hits = _global_seed_mutation_calls(ast.parse(src))
    assert sorted(chain for _, chain in hits) == ["np.random.seed", "random.seed"]


def test_global_seed_inside_fork_rng_is_exempt():
    """MC-dropout-style ops have no generator argument; seeding inside
    torch.random.fork_rng() restores caller state and is the sanctioned
    pattern."""
    src = """
import torch
def mc_dropout_predict(model, x, *, seed):
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        return model(x)
"""
    assert _global_seed_mutation_calls(ast.parse(src)) == []


def test_local_generator_not_flagged():
    src = """
import torch
import numpy as np
def sample(mu, *, seed):
    g = torch.Generator()
    g.manual_seed(seed)
    rng = np.random.default_rng(seed)
    return torch.randn(3, generator=g), rng.random()
"""
    assert _global_seed_mutation_calls(ast.parse(src)) == []


# -- B-04: the merged signature reconstructor ---------------------------------


def test_signature_string_emits_positional_only_marker():
    # The two drifted reconstructors were merged onto the scaffolder
    # validator's version, whose ONLY behavioral difference was emitting the
    # `/` positional-only marker. This fixture pins that exact difference:
    # a generic signature would pass under either ancestor.
    from signature_ast import _signature_string

    fn = ast.parse("def f(a, b, /, c, *, d=1): ...").body[0]
    assert _signature_string(fn) == "f(a, b, /, c, *, d=1)"
