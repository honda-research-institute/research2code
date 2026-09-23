# Golden semantic-review corpus (Block 7 continuation)

Test-only fixtures for the generic-insight contract's semantic layer.
Each case directory holds:

- `candidate.json` — the reviewed interpretation: a statement plus exact
  byte-span citations into one committed source.
- `expectations.json` — the recorded verdict and its oracle: source and
  candidate SHA-256 bindings, required facts, prohibited
  interpretations, required ambiguity disclosures, rationale, and
  reviewer/sign-off provenance.

Verdicts transcribe the approved contract
(the generic insight artifact contract note (internal, not shipped)):
the RFC 10008 known-bad corruption table becomes the rejected cases,
and the BADGE/PDWA accepted-account sections become the accepted cases.
Semantic acceptance is externally owned. Production validation never
reads this corpus and always reports `semantic_status: "unreviewed"`;
the fixture-only helper (`tests/semantic_corpus.py`) proves corpus
integrity and does not classify arbitrary paraphrases.

Sources cited (committed, hash-bound):

- `tests/fixtures/generic_insights/rfc10008/paper.md`
- `input_papers/deep-batch-active-learning.md` (BADGE)

Adding a case is a reviewed act: update `EXPECTED_CASES` in
`tests/test_generic_insights_semantic_corpus.py` in the same change.
