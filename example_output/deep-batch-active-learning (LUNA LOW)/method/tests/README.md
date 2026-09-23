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
  - Tested (no mutant applicable) — passes; the function offers no bounded mutation to check the test against.
  - Module: `test_alg_badge.py`, exercises `select_batch`.
- **k-MEANS++ seeding sampler** (`alg-kmeans-plus-plus`, Appendix A, Algorithm 2)
  - Tested — passes, and fails on a mechanically broken copy of its function, so it distinguishes correct from broken code.
  - Module: `test_alg_kmeans_plus_plus.py`, exercises `kmeans_plus_plus_seeding`.
- **Neural classifier and cross-entropy training objective** (`eq-classifier-cross-entropy`, Section 2)
  - Weak test — passes, but ALSO passed against a mechanically broken copy of its function, so it could not demonstrate it distinguishes correct from broken code.
  - Module: `test_eq_classifier_cross_entropy.py`, exercises `BADGEClassifier.forward`.
- **Hallucinated final-layer gradient embedding** (`eq-gradient-embedding`, Section 3, Equation (1))
  - Tested — passes, and fails on a mechanically broken copy of its function, so it distinguishes correct from broken code.
  - Module: `test_eq_gradient_embedding.py`, exercises `compute_gradient_embeddings`.
- **Softmax cross-entropy in last-layer parameterization** (`eq-softmax-cross-entropy`, Section 3)
  - Generated but could not be verified — not shipped.
