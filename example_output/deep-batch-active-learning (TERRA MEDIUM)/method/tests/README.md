# Per-element tests

Two levels of verification ship with this delivery. The tests in
this directory verify individual implemented paper elements
against the paper's own statements, one small deterministic test
per element, at fixture scale. The notebook demo is the
integration test that runs everything together.

Run them from the run directory:

    python -m pytest method/tests -q

Every eligible element (implemented in code and mapped to a
function) is listed below, including the ones no shipped test
covers — coverage gaps are disclosed, never silent.

- **BADGE batch active learning** (`alg-badge`, Algorithm 1)
  - Weak test — passes, but ALSO passed against a mechanically broken copy of its function, so it could not demonstrate it distinguishes correct from broken code.
  - Module: `test_alg_badge.py`, exercises `select_batch`.
- **Hallucinated gradient embedding computation** (`alg-gradient-embedding`, Algorithm 1 and Section 3)
  - Tested (no mutant applicable) — passes; the function offers no bounded mutation to check the test against.
  - Module: `test_alg_gradient_embedding.py`, exercises `GradientEmbeddingMLP.forward_with_embedding`.
- **k-MEANS++ seeding over gradient embeddings** (`alg-kmeanspp-seeding`, Section 3 and Appendix A, Algorithm 2)
  - Tested — passes, and fails on a mechanically broken copy of its function, so it distinguishes correct from broken code.
  - Module: `test_alg_kmeanspp_seeding.py`, exercises `kmeans_plus_plus_seeding`.
- **Joint diversity and uncertainty acquisition** (`concept-diverse-uncertain-batches`, Section 3)
  - Weak test — passes, but ALSO passed against a mechanically broken copy of its function, so it could not demonstrate it distinguishes correct from broken code.
  - Module: `test_concept_diverse_uncertain_batches.py`, exercises `select_batch`.
- **Pool-based active learning setting** (`concept-pool-based-active-learning`, Section 2)
  - Weak test — passes, but ALSO passed against a mechanically broken copy of its function, so it could not demonstrate it distinguishes correct from broken code.
  - Module: `test_concept_pool_based_active_learning.py`, exercises `select_batch`.
- **Neural classifier prediction** (`eq-classifier-prediction`, Section 2)
  - Tested (no mutant applicable) — passes; the function offers no bounded mutation to check the test against.
  - Module: `test_eq_classifier_prediction.py`, exercises `GradientEmbeddingMLP.forward`.
- **Cross-entropy training loss** (`eq-cross-entropy`, Section 2)
  - Weak test — passes, but ALSO passed against a mechanically broken copy of its function, so it could not demonstrate it distinguishes correct from broken code.
  - Module: `test_eq_cross_entropy.py`, exercises `train_from_scratch`.
- **BADGE gradient embedding** (`eq-gradient-embedding`, Algorithm 1)
  - Tested — passes, and fails on a mechanically broken copy of its function, so it distinguishes correct from broken code.
  - Module: `test_eq_gradient_embedding.py`, exercises `compute_gradient_embeddings`.
- **Nearest-center distance** (`eq-kmeanspp-distance`, Appendix A, Algorithm 2)
  - Tested — passes, and fails on a mechanically broken copy of its function, so it distinguishes correct from broken code.
  - Module: `test_eq_kmeanspp_distance.py`, exercises `kmeans_plus_plus_seeding`.
- **k-MEANS++ sampling probability** (`eq-kmeanspp-probability`, Appendix A, Algorithm 2)
  - Tested — passes, and fails on a mechanically broken copy of its function, so it distinguishes correct from broken code.
  - Module: `test_eq_kmeanspp_probability.py`, exercises `kmeans_plus_plus_seeding`.
- **Softmax cross-entropy at the final layer** (`eq-softmax-cross-entropy`, Section 3, multiclass softmax example)
  - Tested — passes, and fails on a mechanically broken copy of its function, so it distinguishes correct from broken code.
  - Module: `test_eq_softmax_cross_entropy.py`, exercises `compute_gradient_embeddings`.
- **Per-class gradient embedding block** (`eq-softmax-gradient-block`, Section 3, Equation (1))
  - Tested (no mutant applicable) — passes; the function offers no bounded mutation to check the test against.
  - Module: `test_eq_softmax_gradient_block.py`, exercises `GradientEmbeddingMLP.forward_with_embedding`.
