"""Insight semantic reviewer build: acceptance-record schema, sliced
review bundles, hash-bound record loading (stale reads as absent), the
reviewer dispatch prompt, and the corpus calibration harness bar.

Design: the insight semantic reviewer design note (internal, not shipped).
Nothing here runs a model: the calibration harness is exercised with fake
reviewers; the golden corpus remains the recorded hiring bar the live
calibration run must clear before the reviewer owns acceptance.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from calibrate_insight_reviewer import (
    DEFAULT_CORPUS,
    RECORD_FILENAME,
    REPORT_FILENAME,
    CalibrationCase,
    agent_definition_binding,
    classify_outcome,
    load_corpus_cases,
    run_calibration,
)
from dispatch_templates import (
    PRODUCER_ANTI_REVIEW_CLAUSE,
    WRITEABLE_PATHS,
    _writes_only_review_artifacts,
    build_insight_review_prompt,
)
from insight_semantic_review import (
    CONTEXT_WINDOW_BYTES,
    RECORD_RELPATH,
    build_case_bundle,
    build_review_bundles,
    load_acceptance_record,
    resolve_semantic_status,
)
from schemas.generic_insights import GenericInsights
from schemas.semantic_review import (
    SCHEMA_VERSION,
    SemanticReviewRecord,
    derive_overall_verdict,
)

REPO_ROOT = Path(__file__).resolve().parent.parent

SRC_SHA = "a" * 64
CAND_SHA = "b" * 64


def _record_payload(
    *,
    record_id: str = "i-001",
    verdict: str = "accepted",
    failure_class: str | None = None,
    needs_human: list[dict] | None = None,
    overall: str | None = None,
    source_sha256: str = SRC_SHA,
    candidate_sha256: str = CAND_SHA,
    schema_version: str = SCHEMA_VERSION,
) -> dict:
    verdicts = []
    if verdict:
        entry: dict = {"record_id": record_id, "verdict": verdict,
                       "reason": "the quote decides it"}
        if failure_class:
            entry["failure_class"] = failure_class
        verdicts.append(entry)
    if overall is None:
        overall = derive_overall_verdict(
            [v["verdict"] for v in verdicts], bool(needs_human))
    return {
        "schema_version": schema_version,
        "reviewer": "r2c-insight-semantic-reviewer",
        "source_sha256": source_sha256,
        "candidate_sha256": candidate_sha256,
        "verdicts": verdicts,
        "needs_human": needs_human or [],
        "overall_verdict": overall,
    }


# ---------------------------------------------------------------------------
# Acceptance-record schema
# ---------------------------------------------------------------------------


def test_valid_record_roundtrips():
    rec = SemanticReviewRecord.model_validate(_record_payload())
    assert rec.overall_verdict == "accepted"
    assert rec.verdicts[0].record_id == "i-001"


def test_rejection_requires_failure_class():
    with pytest.raises(ValidationError, match="failure class"):
        SemanticReviewRecord.model_validate(
            _record_payload(verdict="rejected", overall="rejected"))


def test_acceptance_forbids_failure_class():
    with pytest.raises(ValidationError, match="cannot carry"):
        SemanticReviewRecord.model_validate(_record_payload(
            verdict="accepted", failure_class="unsupported_strengthening"))


def test_failure_class_is_a_closed_set():
    with pytest.raises(ValidationError):
        SemanticReviewRecord.model_validate(_record_payload(
            verdict="rejected", failure_class="sounded_wrong",
            overall="rejected"))


def test_empty_record_is_invalid():
    payload = _record_payload()
    payload["verdicts"] = []
    payload["overall_verdict"] = "accepted"
    with pytest.raises(ValidationError, match="no verdicts"):
        SemanticReviewRecord.model_validate(payload)


def test_duplicate_record_ids_rejected_across_lists():
    payload = _record_payload(
        needs_human=[{"record_id": "i-001", "reason": "ambiguous span"}],
        overall="needs_human")
    with pytest.raises(ValidationError, match="more than once"):
        SemanticReviewRecord.model_validate(payload)


def test_overall_verdict_is_derived_not_chosen():
    # A reviewer cannot hand-pick "accepted" over a rejection.
    payload = _record_payload(
        verdict="rejected", failure_class="invented_algorithm_or_rule",
        overall="accepted")
    with pytest.raises(ValidationError, match="inconsistent"):
        SemanticReviewRecord.model_validate(payload)


def test_rejection_dominates_escalation_in_derivation():
    assert derive_overall_verdict(["rejected", "accepted"], True) == "rejected"
    assert derive_overall_verdict(["accepted"], True) == "needs_human"
    assert derive_overall_verdict(["accepted"], False) == "accepted"
    assert derive_overall_verdict([], True) == "needs_human"
    assert derive_overall_verdict([], False) is None


def test_unknown_fields_forbidden():
    payload = _record_payload()
    payload["comment"] = "extra"
    with pytest.raises(ValidationError):
        SemanticReviewRecord.model_validate(payload)


# ---------------------------------------------------------------------------
# Record loading — every failure mode reads as absent
# ---------------------------------------------------------------------------


def _write_record(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_loader_accepts_hash_matching_record(tmp_path):
    path = tmp_path / RECORD_FILENAME
    _write_record(path, _record_payload())
    res = load_acceptance_record(
        path, source_sha256=SRC_SHA, candidate_sha256=CAND_SHA,
        expected_record_ids={"i-001"})
    assert res.present
    assert resolve_semantic_status(res) == "accepted"


def test_loader_missing_file_is_absent(tmp_path):
    res = load_acceptance_record(
        tmp_path / "nope.json",
        source_sha256=SRC_SHA, candidate_sha256=CAND_SHA)
    assert not res.present
    assert resolve_semantic_status(res) == "unreviewed"


def test_loader_treats_drifted_candidate_hash_as_absent(tmp_path):
    """The design's stale-record gate: the candidate regenerates after
    review, so the record's hashes no longer match — it must read as
    absent, never as certification of the new bytes."""
    path = tmp_path / RECORD_FILENAME
    _write_record(path, _record_payload())
    res = load_acceptance_record(
        path, source_sha256=SRC_SHA, candidate_sha256="c" * 64)
    assert not res.present
    assert "stale" in res.reason
    assert resolve_semantic_status(res) == "unreviewed"


def test_loader_treats_drifted_source_hash_as_absent(tmp_path):
    path = tmp_path / RECORD_FILENAME
    _write_record(path, _record_payload())
    res = load_acceptance_record(
        path, source_sha256="d" * 64, candidate_sha256=CAND_SHA)
    assert not res.present
    assert "stale" in res.reason


def test_loader_unknown_major_version_is_absent(tmp_path):
    path = tmp_path / RECORD_FILENAME
    _write_record(path, _record_payload(schema_version="2.0.0"))
    res = load_acceptance_record(
        path, source_sha256=SRC_SHA, candidate_sha256=CAND_SHA)
    assert not res.present
    assert "major" in res.reason


def test_loader_non_json_is_absent(tmp_path):
    path = tmp_path / RECORD_FILENAME
    path.write_text("not json", encoding="utf-8")
    res = load_acceptance_record(
        path, source_sha256=SRC_SHA, candidate_sha256=CAND_SHA)
    assert not res.present


def test_loader_incomplete_id_coverage_is_absent(tmp_path):
    """A partial review certifies nothing: verdicts + needs_human must
    cover the candidate's record ids exactly."""
    path = tmp_path / RECORD_FILENAME
    _write_record(path, _record_payload(record_id="i-001"))
    res = load_acceptance_record(
        path, source_sha256=SRC_SHA, candidate_sha256=CAND_SHA,
        expected_record_ids={"i-001", "i-002"})
    assert not res.present
    assert "incomplete" in res.reason
    res = load_acceptance_record(
        path, source_sha256=SRC_SHA, candidate_sha256=CAND_SHA,
        expected_record_ids={"i-999"})
    assert not res.present


def test_needs_human_leaves_status_unreviewed(tmp_path):
    path = tmp_path / RECORD_FILENAME
    _write_record(path, _record_payload(
        verdict="", needs_human=[
            {"record_id": "i-001", "reason": "context window too small"}]))
    res = load_acceptance_record(
        path, source_sha256=SRC_SHA, candidate_sha256=CAND_SHA,
        expected_record_ids={"i-001"})
    assert res.present
    assert resolve_semantic_status(res) == "unreviewed"


def test_rejected_record_resolves_rejected(tmp_path):
    path = tmp_path / RECORD_FILENAME
    _write_record(path, _record_payload(
        verdict="rejected", failure_class="imperative_from_suggestion"))
    res = load_acceptance_record(
        path, source_sha256=SRC_SHA, candidate_sha256=CAND_SHA)
    assert resolve_semantic_status(res) == "rejected"


# ---------------------------------------------------------------------------
# Review bundles — sliced, never the whole paper
# ---------------------------------------------------------------------------


_BUNDLE_SOURCE = (
    "# Probe Paper\n\n"
    "## Behavior\n\n"
    "Clients MAY retry idempotent requests after a connection failure.\n"
    "Retries preserve the original request body.\n\n"
    "## Notes\n\n"
    "This section exists so the context clamp has a boundary to respect. "
    + "Filler sentence. " * 40 + "\n"
)


def _bundle_model() -> tuple[bytes, GenericInsights]:
    source = _BUNDLE_SOURCE.encode("utf-8")
    quote = "Clients MAY retry idempotent requests after a connection failure."
    start = source.find(quote.encode("utf-8"))
    assert start > 0
    candidate = {
        "schema_version": "1.0.0",
        "source": {"path": ".pipeline/paper.md",
                   "sha256": hashlib.sha256(source).hexdigest()},
        "document": {"kind": "technical_standard",
                     "contribution_kinds": ["bundle_probe"]},
        "summary": {"text": "Describes retry behavior.",
                    "citation_ids": ["c-1"]},
        "intuition": {"text": "Retries are safe for idempotent requests.",
                      "citation_ids": ["c-1"]},
        "citations": [
            {"id": "c-1", "region_id": "h002",
             "span": {"start": start, "end": start + len(quote)},
             "quote": quote},
        ],
        "vocabulary": [],
        "vocabulary_none_identified_reason":
            "probe fixture with no paper-local terms",
        "insights": [
            {"id": "i-behavior", "kind": "behavior",
             "statement": "Clients can retry after connection failures.",
             "citation_ids": ["c-1"], "confidence": "high"},
            {"id": "i-rule", "kind": "normative_rule",
             "normative": {
                 "framework": "BCP 14", "strength": "MAY",
                 "subject": "clients", "action": "retry idempotent requests",
                 "conditions": ["after a connection failure"],
                 "exceptions": [], "occurrence_id": "bcp-0001"},
             "citation_ids": ["c-1"], "confidence": "high"},
        ],
        "coverage": {
            "headings": [
                {"region_id": "h001", "disposition": "not_applicable",
                 "reason": "title with no body"},
                {"region_id": "h002", "disposition": "covered"},
                {"region_id": "h003", "disposition": "not_applicable",
                 "reason": "test filler"},
            ],
            "missing": ["figure 2 could not be grounded"],
            "unresolved_dependencies": [],
        },
    }
    return source, GenericInsights.model_validate(candidate)


def test_bundles_cover_every_insight_record():
    source, model = _bundle_model()
    bundle = build_review_bundles(source, model)
    assert [r["record_id"] for r in bundle["records"]] == \
        ["i-behavior", "i-rule"]
    assert bundle["document"]["kind"] == "technical_standard"
    assert bundle["declared_gaps"]["missing"] == \
        ["figure 2 could not be grounded"]


def test_bundle_carries_exact_quote_and_bounded_context():
    source, model = _bundle_model()
    bundle = build_review_bundles(source, model)
    cit = bundle["records"][0]["citations"][0]
    assert cit["quote"].startswith("Clients MAY retry")
    assert cit["heading"] == "Behavior"
    assert cit["coverage_disposition"] == "covered"
    assert len(cit["context_before"].encode()) <= CONTEXT_WINDOW_BYTES
    assert len(cit["context_after"].encode()) <= CONTEXT_WINDOW_BYTES
    # Clamped to the region body: the next heading and its filler never
    # leak into the context window.
    assert "## Notes" not in cit["context_after"]
    assert "Filler sentence" not in cit["context_after"]


def test_normative_record_bundles_structured_fields_not_prose():
    source, model = _bundle_model()
    bundle = build_review_bundles(source, model)
    rule = bundle["records"][1]
    assert rule["statement"] is None
    assert rule["normative"]["strength"] == "MAY"
    assert rule["normative"]["subject"] == "clients"


def test_bundle_never_carries_the_whole_source():
    source, model = _bundle_model()
    text = json.dumps(build_review_bundles(source, model))
    # The far tail of the source (Notes filler) must not appear anywhere.
    assert "Filler sentence" not in text


def test_case_bundle_matches_production_shape():
    case_dir = DEFAULT_CORPUS / "rfc10008-idempotence-strengthening"
    case = json.loads((case_dir / "candidate.json").read_text("utf-8"))
    source = (REPO_ROOT / case["source"]["path"]).read_bytes()
    bundle = build_case_bundle(source, case)
    assert len(bundle["records"]) == 1
    rec = bundle["records"][0]
    assert rec["record_id"] == "rfc10008-idempotence-strengthening"
    assert rec["statement"] == case["statement"]
    cit = rec["citations"][0]
    assert cit["quote"] == case["citations"][0]["quote"]
    assert 0 < len(cit["context_before"].encode()) <= CONTEXT_WINDOW_BYTES
    assert 0 < len(cit["context_after"].encode()) <= CONTEXT_WINDOW_BYTES


# ---------------------------------------------------------------------------
# Dispatch prompt + writeable paths
# ---------------------------------------------------------------------------


def test_review_prompt_carries_bindings_and_no_producer_framing():
    prompt = build_insight_review_prompt(
        bundle_json='{"records": []}',
        record_path="/tmp/x/generic_insights.semantic_review.json",
        source_sha256=SRC_SHA,
        candidate_sha256=CAND_SHA,
    )
    assert SRC_SHA in prompt and CAND_SHA in prompt
    assert "/tmp/x/generic_insights.semantic_review.json" in prompt
    assert "USE YOUR WRITE TOOL EXACTLY ONCE" in prompt
    assert "needs_human" in prompt
    assert "unsupported_strengthening" in prompt
    # The single-method package scope contract has nothing to review and
    # must not reach this agent.
    assert "SCOPE CONTRACT" not in prompt
    # The producer-framed anti-review clause inverted a reviewer's task
    # once (2026-06-15); the acceptance record path must read as a review
    # artifact.
    assert PRODUCER_ANTI_REVIEW_CLAUSE not in prompt


def test_semantic_review_path_is_a_review_artifact():
    assert _writes_only_review_artifacts([RECORD_RELPATH])
    assert WRITEABLE_PATHS["r2c-insight-semantic-reviewer"] == [RECORD_RELPATH]


def test_agent_definition_is_subagent_and_bash_denied():
    text = (REPO_ROOT /
            ".opencode/agents/r2c-insight-semantic-reviewer.md").read_text(
        encoding="utf-8")
    assert "mode: subagent" in text
    assert "bash: deny" in text
    assert "judge of meaning" in text


# ---------------------------------------------------------------------------
# Calibration harness — the corpus is the hiring bar
# ---------------------------------------------------------------------------


def _fake_reviewer(behavior):
    """A fake dispatch function. `behavior(case)` returns the payload to
    write at the record path (or None to write nothing)."""
    def dispatch(case: CalibrationCase, prompt: str, record_path: Path):
        payload = behavior(case)
        if payload is not None:
            record_path.write_text(json.dumps(payload), encoding="utf-8")
    return dispatch


def _correct_payload(case: CalibrationCase) -> dict:
    return _record_payload(
        record_id=case.case_id,
        verdict=case.expected_verdict,
        failure_class="unsupported_strengthening"
        if case.expected_verdict == "rejected" else None,
        source_sha256=case.source_sha256,
        candidate_sha256=case.candidate_sha256,
    )


def test_corpus_cases_load_hash_verified():
    cases = load_corpus_cases()
    assert len(cases) == 9
    assert sum(1 for c in cases if c.expected_verdict == "rejected") == 7
    assert sum(1 for c in cases if c.expected_verdict == "accepted") == 2


def test_perfect_reviewer_passes_the_bar(tmp_path):
    cases = load_corpus_cases()
    report = run_calibration(
        cases, _fake_reviewer(_correct_payload), tmp_path,
        binding=agent_definition_binding())
    summary = report["summary"]
    assert summary["bar_passed"] is True
    assert summary["zero_known_bad_accepts"] is True
    assert summary["matches"] == summary["total"] == 9
    assert summary["disqualifying_cases"] == []
    # The report is a kept artifact with per-case verdicts and bindings.
    on_disk = json.loads((tmp_path / REPORT_FILENAME).read_text("utf-8"))
    assert on_disk["summary"] == summary
    assert on_disk["binding"]["agent_definition_sha256"]
    assert all(c["prompt_sha256"] for c in on_disk["cases"])


def test_accepting_one_known_bad_is_disqualifying(tmp_path):
    cases = load_corpus_cases()
    bad_id = next(c.case_id for c in cases if c.expected_verdict == "rejected")

    def behavior(case):
        payload = _correct_payload(case)
        if case.case_id == bad_id:
            payload = _record_payload(
                record_id=case.case_id, verdict="accepted",
                source_sha256=case.source_sha256,
                candidate_sha256=case.candidate_sha256)
        return payload

    report = run_calibration(cases, _fake_reviewer(behavior), tmp_path)
    summary = report["summary"]
    assert summary["bar_passed"] is False
    assert summary["zero_known_bad_accepts"] is False
    assert summary["known_bad_accepted"] == 1
    assert bad_id in summary["disqualifying_cases"]


def test_needs_human_on_accepted_case_is_miss_not_disqualifying(tmp_path):
    cases = load_corpus_cases()
    good_id = next(
        c.case_id for c in cases if c.expected_verdict == "accepted")

    def behavior(case):
        if case.case_id == good_id:
            return _record_payload(
                record_id=case.case_id, verdict="",
                needs_human=[{"record_id": case.case_id,
                              "reason": "cannot decide from the bundle"}],
                source_sha256=case.source_sha256,
                candidate_sha256=case.candidate_sha256)
        return _correct_payload(case)

    report = run_calibration(cases, _fake_reviewer(behavior), tmp_path)
    summary = report["summary"]
    assert summary["bar_passed"] is True
    assert summary["non_disqualifying_misses"] == [good_id]


def test_needs_human_on_known_bad_fails_the_bar(tmp_path):
    cases = load_corpus_cases()
    bad_id = next(c.case_id for c in cases if c.expected_verdict == "rejected")

    def behavior(case):
        if case.case_id == bad_id:
            return _record_payload(
                record_id=case.case_id, verdict="",
                needs_human=[{"record_id": case.case_id,
                              "reason": "unsure"}],
                source_sha256=case.source_sha256,
                candidate_sha256=case.candidate_sha256)
        return _correct_payload(case)

    report = run_calibration(cases, _fake_reviewer(behavior), tmp_path)
    assert report["summary"]["bar_passed"] is False
    case = next(c for c in report["cases"] if c["case_id"] == bad_id)
    assert case["outcome"] == "undecided_known_bad"


def test_stale_hash_record_reads_as_no_record(tmp_path):
    """The design's drift gate at the harness level: a reviewer that
    mangles the hash binding produced nothing certifiable."""
    cases = load_corpus_cases()[:1]

    def behavior(case):
        payload = _correct_payload(case)
        payload["candidate_sha256"] = "e" * 64
        return payload

    report = run_calibration(cases, _fake_reviewer(behavior), tmp_path)
    case = report["cases"][0]
    assert case["record_present"] is False
    assert "stale" in case["record_resolution"]
    assert case["outcome"] == "no_record"
    assert report["summary"]["bar_passed"] is False


def test_dispatch_error_is_recorded_not_raised(tmp_path):
    cases = load_corpus_cases()[:1]

    def dispatch(case, prompt, record_path):
        raise ConnectionError("server unreachable")

    report = run_calibration(cases, dispatch, tmp_path)
    case = report["cases"][0]
    assert "server unreachable" in case["dispatch_error"]
    assert case["outcome"] == "no_record"
    assert report["summary"]["bar_passed"] is False


def test_cli_refuses_out_dir_outside_repo(tmp_path, capsys):
    """An out-of-repo --out makes every reviewer write hang on a
    permission ask no headless session can answer (each case burns its
    full timeout and reads as no_record) — the CLI refuses up front."""
    from calibrate_insight_reviewer import main
    with pytest.raises(SystemExit) as excinfo:
        main(["--out", str(tmp_path / "cal")])
    assert excinfo.value.code == 2
    assert "outside the repo" in capsys.readouterr().err


def test_classify_outcome_table():
    assert classify_outcome("rejected", "rejected") == "match"
    assert classify_outcome("rejected", "accepted") == "accepted_known_bad"
    assert classify_outcome("rejected", "needs_human") == \
        "undecided_known_bad"
    assert classify_outcome("rejected", None) == "no_record"
    assert classify_outcome("accepted", "accepted") == "match"
    assert classify_outcome("accepted", "rejected") == "rejected_known_good"
    assert classify_outcome("accepted", "needs_human") == \
        "undecided_known_good"
    assert classify_outcome("accepted", None) == "no_record"


# ---------------------------------------------------------------------------
# Shadow boundary — review stays nonblocking, production stays untouched
# ---------------------------------------------------------------------------


def test_review_modules_are_not_imported_by_production():
    """The calibration harness must stay unreachable from the driver, and
    the deterministic validator keeps sole ownership of its statuses.

    The old string-grep of run_pipeline.py alone passed on a technicality
    once the shadow hook landed (run_pipeline imports insight_shadow_hook,
    which imports insight_semantic_review — one hop the grep never saw), so
    this guard walks the TRANSITIVE import closure over scripts/ from the
    driver. insight_semantic_review is reachable by design through the
    shadow hook; the enforced invariant is that calibrate_insight_reviewer
    (the offline calibration harness) never becomes production code."""
    import ast

    scripts_dir = REPO_ROOT / "scripts"
    local_modules = {p.stem for p in scripts_dir.glob("*.py")}

    def _imports(module: str) -> set[str]:
        tree = ast.parse(
            (scripts_dir / f"{module}.py").read_text(encoding="utf-8")
        )
        found: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                found.update(a.name.split(".")[0] for a in node.names)
                found.update(
                    a.name.split(".")[1] for a in node.names
                    if a.name.startswith("scripts.")
                )
            elif isinstance(node, ast.ImportFrom) and node.module:
                parts = node.module.split(".")
                found.add(parts[0])
                if parts[0] == "scripts" and len(parts) > 1:
                    found.add(parts[1])
        return found & local_modules

    reachable: set[str] = set()
    frontier = ["run_pipeline"]
    while frontier:
        module = frontier.pop()
        if module in reachable:
            continue
        reachable.add(module)
        frontier.extend(_imports(module))

    assert "calibrate_insight_reviewer" not in reachable, (
        "the offline calibration harness became reachable from the driver"
    )
    validator = (REPO_ROOT / "scripts/generic_insights.py").read_text("utf-8")
    assert "insight_semantic_review" not in validator
    assert "schemas.semantic_review" not in validator
    assert "schemas/semantic_review" not in validator
