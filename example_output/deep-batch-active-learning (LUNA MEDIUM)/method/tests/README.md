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
- **Hallucinated-label uncertainty** (`concept-hallucinated-label`, Section 3)
  - Generated but could not be verified — not shipped.
- **Gradient embedding block formula** (`eq-gradient-block`, Section 3, Equation 1)
  - Generated but could not be verified — not shipped.
- **Hallucinated last-layer gradient embedding** (`eq-gradient-embedding`, Algorithm 1 and Section 3)
  - Generated but could not be verified — not shipped.
- **Softmax activation** (`eq-softmax-definition`, Section 3)
  - Generated but could not be verified — not shipped.
- **Softmax output model** (`eq-softmax-model`, Section 3)
  - Generated but could not be verified — not shipped.
