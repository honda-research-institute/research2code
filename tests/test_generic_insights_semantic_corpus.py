"""Block 7 continuation: the golden semantic-review fixture corpus.

Proves the reviewed corpus is intact and hash-bound, that known-bad RFC
corruptions stay recorded as rejected, that the accepted BADGE/PDWA
accounts carry their required facts and disclosures, and that production
validation never upgrades a semantically corrupt but exactly cited
candidate beyond `semantic_status: "unreviewed"`. Semantic verdicts are
externally owned; nothing here becomes a production heuristic.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from generic_insights import validate_candidate, validation_sidecar_payload
from schemas.generic_insights import SCHEMA_VERSION

from tests.semantic_corpus import CORPUS_DIR, load_cases, verify_case

REPO_ROOT = Path(__file__).resolve().parent.parent

# The frozen corpus. Adding a case is a deliberate, reviewed act that
# updates this table; silently dropping or renaming one is a failure.
EXPECTED_CASES = {
    "rfc10008-idempotence-strengthening": "rejected",
    "rfc10008-safety-cache-conflation": "rejected",
    "rfc10008-get-post-strengthening": "rejected",
    "rfc10008-advisory-status-mandating": "rejected",
    "rfc10008-redirect-imperative": "rejected",
    "rfc10008-cache-key-invented-algorithm": "rejected",
    "rfc10008-pagination-invented-rule": "rejected",
    "rfc10008-accepted-core-account": "accepted",
    "badge-accepted-directional-account": "accepted",
}


def test_corpus_is_complete_and_frozen():
    cases = load_cases()
    found = {exp["case_id"]: exp["verdict"] for _, _, exp in cases}
    assert found == EXPECTED_CASES


def test_every_case_verifies_exactly():
    problems = []
    for cdir, candidate, expectations in load_cases():
        problems.extend(verify_case(cdir, candidate, expectations))
    assert problems == []


def test_corpus_sources_are_the_committed_fixtures():
    """Every case cites one of the two committed sources, by hash."""
    allowed = {
        "tests/fixtures/generic_insights/rfc10008/paper.md",
        "input_papers/deep-batch-active-learning.md",
    }
    for _, candidate, expectations in load_cases():
        path = candidate["source"]["path"]
        assert path in allowed, path
        digest = hashlib.sha256((REPO_ROOT / path).read_bytes()).hexdigest()
        assert digest == expectations["source_sha256"]


def test_corrupt_but_exactly_cited_candidate_stays_unreviewed(tmp_path):
    """The contract's traceability/entailment split, on the production
    path: a candidate embedding a corpus-recorded corrupt statement with
    an exact citation passes deterministic validation and still reports
    semantic_status=unreviewed — never accepted."""
    case_dir = CORPUS_DIR / "rfc10008-idempotence-strengthening"
    case = json.loads((case_dir / "candidate.json").read_text("utf-8"))
    corrupt_statement = case["statement"]
    cited_sentence = case["citations"][0]["quote"]

    source = (
        "# Corpus Probe\n\n## Behavior\n\n"
        f"{cited_sentence}\n\n## Results\n\nRetries succeeded in tests.\n"
    )
    data = source.encode("utf-8")
    start = data.find(cited_sentence.encode("utf-8"))
    assert start >= 0
    paper = tmp_path / "paper.md"
    paper.write_text(source, encoding="utf-8")

    candidate = {
        "schema_version": SCHEMA_VERSION,
        "source": {"path": ".pipeline/paper.md",
                   "sha256": hashlib.sha256(data).hexdigest()},
        "document": {"kind": "technical_standard",
                     "contribution_kinds": ["corpus_probe"]},
        "summary": {"text": "Describes QUERY retry behavior.",
                    "citation_ids": ["c-idem"]},
        "intuition": {"text": "Retries are safe to repeat.",
                      "citation_ids": ["c-idem"]},
        "citations": [
            {"id": "c-idem", "region_id": "h002",
             "span": {"start": start,
                      "end": start + len(cited_sentence.encode("utf-8"))},
             "quote": cited_sentence},
        ],
        "vocabulary": [
            {"id": "v-idem", "term": "idempotent", "domain": "http",
             "meaning": "Retryable per the cited source.",
             "confidence": "high", "citation_ids": ["c-idem"]},
        ],
        "insights": [
            {"id": "i-corrupt", "kind": "behavior",
             "statement": corrupt_statement,
             "citation_ids": ["c-idem"], "confidence": "high"},
        ],
        "coverage": {
            "headings": [
                {"region_id": "h001", "disposition": "not_applicable",
                 "reason": "title heading with no body text"},
                {"region_id": "h002", "disposition": "covered"},
                {"region_id": "h003", "disposition": "not_applicable",
                 "reason": "test-scaffold filler sentence"},
            ],
            "missing": [],
            "unresolved_dependencies": [],
        },
    }
    candidate_path = tmp_path / "generic_insights.json"
    candidate_path.write_text(json.dumps(candidate), encoding="utf-8")

    result = validate_candidate(paper, candidate_path,
                                expected_source_path=".pipeline/paper.md")
    # Exactly cited and structurally complete: deterministic gates pass.
    assert result.deterministic_status == "passed", result.problems
    # And the corrupt meaning is still never self-certified.
    assert result.semantic_status == "unreviewed"
    sidecar = validation_sidecar_payload(result)
    assert sidecar["semantic_status"] == "unreviewed"


def test_helper_is_not_imported_by_production():
    """The corpus helper is test infrastructure only."""
    for module in ("scripts/generic_insights.py",
                   "schemas/generic_insights.py"):
        text = (REPO_ROOT / module).read_text(encoding="utf-8")
        assert "semantic_corpus" not in text
