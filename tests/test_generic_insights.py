"""Block 7 pure tranche: generic insight validator + renderer.

Deterministic gates only — semantic fidelity is externally owned and the
validator always reports `semantic_status: "unreviewed"`. Fixtures: two
synthetic sources with hand-computable spans, the committed RFC 10008 full
source (tests/fixtures/generic_insights/rfc10008/), the committed BADGE and
PDWA parsed sources (example_runs/), and copied-run write-boundary checks
against the ICRA explanation-only parity control.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from generic_insights import (
    MANDATORY_RISK_HEADINGS,
    atomic_write_text,
    build_region_inventory,
    build_risk_units,
    find_bcp14_declaration,
    render_insights_md,
    scan_normative_occurrences,
    validate_candidate,
    validation_sidecar_payload,
)
from schemas.generic_insights import SCHEMA_VERSION

REPO_ROOT = Path(__file__).resolve().parent.parent
RFC_FIXTURE = REPO_ROOT / "tests/fixtures/generic_insights/rfc10008/paper.md"
BADGE_SOURCE = REPO_ROOT / "input_papers/deep-batch-active-learning.md"
# Finished run directories used as write-boundary parity controls. They are
# not part of the repo; the tests that copy them skip when they are absent.
ICRA_RUN = REPO_ROOT / "example_runs/old/ICRA21_HICA"
GBALD_RUN = REPO_ROOT / "example_runs/old/bayesian-active-learning"


# ---------------------------------------------------------------------------
# Synthetic sources with hand-computable spans
# ---------------------------------------------------------------------------

SIMPLE_SOURCE = """# Tiny Method

## Approach

We batch gradients and cluster the embeddings.

## Results

Accuracy improved on the benchmark.
"""

SPEC_SOURCE = """# Tiny Spec

The key words "MUST", "MUST NOT", "SHOULD", and "MAY" in this document are
to be interpreted as described in BCP 14 [RFC2119] [RFC8174].

## Rules

A server MUST reject malformed frames. A proxy MUST NOT cache responses and
SHOULD revalidate stale entries.

## Security Considerations

Attackers can replay tokens.

- Log every access attempt.
- Rotate signing keys.
"""


def _span_of(source: str, needle: str) -> tuple[int, int]:
    """Byte span of `needle` inside `source` (must occur exactly once)."""
    data = source.encode("utf-8")
    target = needle.encode("utf-8")
    start = data.find(target)
    assert start >= 0, f"needle not found: {needle!r}"
    assert data.find(target, start + 1) < 0, f"needle not unique: {needle!r}"
    return (start, start + len(target))


def _write_source(tmp_path: Path, text: str) -> tuple[Path, str]:
    paper = tmp_path / "paper.md"
    paper.write_text(text, encoding="utf-8")
    return paper, hashlib.sha256(text.encode("utf-8")).hexdigest()


def _citation(cid: str, region_id: str, source: str, needle: str) -> dict:
    start, end = _span_of(source, needle)
    return {"id": cid, "region_id": region_id,
            "span": {"start": start, "end": end}, "quote": needle}


def _simple_candidate(sha256: str) -> dict:
    """A fully valid candidate for SIMPLE_SOURCE. Regions: h001 title (empty
    body -> not_applicable), h002 Approach, h003 Results."""
    return {
        "schema_version": SCHEMA_VERSION,
        "source": {"path": ".pipeline/paper.md", "sha256": sha256},
        "document": {"kind": "research_method",
                     "contribution_kinds": ["toy_batch_method"]},
        "summary": {"text": "Batches gradients, then clusters embeddings.",
                    "citation_ids": ["c-approach"]},
        "intuition": {"text": "Clustering spreads the selected batch.",
                      "citation_ids": ["c-approach"]},
        "citations": [
            _citation("c-approach", "h002", SIMPLE_SOURCE,
                      "We batch gradients and cluster the embeddings."),
            _citation("c-results", "h003", SIMPLE_SOURCE,
                      "Accuracy improved on the benchmark."),
        ],
        "vocabulary": [
            {"id": "v-embedding", "term": "embedding", "domain": "toy_method",
             "meaning": "The vector the method clusters.",
             "confidence": "high", "citation_ids": ["c-approach"]},
        ],
        "insights": [
            {"id": "i-obs", "kind": "empirical_observation",
             "statement": "The paper reports improved benchmark accuracy.",
             "citation_ids": ["c-results"], "confidence": "medium"},
        ],
        "coverage": {
            "headings": [
                {"region_id": "h001", "disposition": "not_applicable",
                 "reason": "title heading with no body text"},
                {"region_id": "h002", "disposition": "covered"},
                {"region_id": "h003", "disposition": "covered"},
            ],
            "missing": [],
            "unresolved_dependencies": [],
        },
    }


def _spec_candidate(sha256: str) -> dict:
    """A fully valid candidate for SPEC_SOURCE: three normative records
    (MUST / MUST NOT / SHOULD), three dedicated risk-unit records, full
    heading coverage."""
    rules_sentence_1 = "A server MUST reject malformed frames."
    rules_sentence_2 = ("A proxy MUST NOT cache responses and\n"
                        "SHOULD revalidate stale entries.")
    return {
        "schema_version": SCHEMA_VERSION,
        "source": {"path": ".pipeline/paper.md", "sha256": sha256},
        "document": {"kind": "technical_standard",
                     "contribution_kinds": ["toy_protocol"]},
        "summary": {"text": "Defines frame handling rules for a toy protocol.",
                    "citation_ids": ["c-rule1"]},
        "intuition": {"text": "Strict rejection keeps proxies consistent.",
                      "citation_ids": ["c-rule2"]},
        "citations": [
            _citation("c-rule1", "h002", SPEC_SOURCE, rules_sentence_1),
            _citation("c-rule2", "h002", SPEC_SOURCE, rules_sentence_2),
            _citation("c-risk1", "h003", SPEC_SOURCE,
                      "Attackers can replay tokens."),
            _citation("c-risk2", "h003", SPEC_SOURCE,
                      "- Log every access attempt."),
            _citation("c-risk3", "h003", SPEC_SOURCE,
                      "- Rotate signing keys."),
        ],
        "vocabulary": [
            {"id": "v-frame", "term": "frame", "domain": "toy_protocol",
             "meaning": "The protocol's message unit.",
             "confidence": "medium", "citation_ids": ["c-rule1"]},
        ],
        "insights": [
            {"id": "i-must", "kind": "normative_rule",
             "normative": {"framework": "BCP 14", "strength": "MUST",
                           "subject": "a server", "action": "reject malformed frames",
                           "occurrence_id": "bcp-0001"},
             "citation_ids": ["c-rule1"], "confidence": "high"},
            {"id": "i-mustnot", "kind": "normative_rule",
             "normative": {"framework": "BCP 14", "strength": "MUST_NOT",
                           "subject": "a proxy", "action": "cache responses",
                           "occurrence_id": "bcp-0002"},
             "citation_ids": ["c-rule2"], "confidence": "high"},
            {"id": "i-should", "kind": "normative_rule",
             "normative": {"framework": "BCP 14", "strength": "SHOULD",
                           "subject": "a proxy", "action": "revalidate stale entries",
                           "occurrence_id": "bcp-0003"},
             "citation_ids": ["c-rule2"], "confidence": "high"},
            {"id": "i-replay", "kind": "security_consideration",
             "statement": "The source names token replay as an attack vector.",
             "citation_ids": ["c-risk1"], "confidence": "high"},
            {"id": "i-log", "kind": "security_consideration",
             "statement": "The source calls for logging access attempts.",
             "citation_ids": ["c-risk2"], "confidence": "high"},
            {"id": "i-rotate", "kind": "security_consideration",
             "statement": "The source calls for rotating signing keys.",
             "citation_ids": ["c-risk3"], "confidence": "high"},
        ],
        "coverage": {
            "headings": [
                {"region_id": "h001", "disposition": "not_applicable",
                 "reason": "title body holds only the framework "
                           "declaration boilerplate"},
                {"region_id": "h002", "disposition": "covered"},
                {"region_id": "h003", "disposition": "covered"},
            ],
            "missing": [],
            "unresolved_dependencies": [],
        },
    }


def _validate(tmp_path: Path, source: str, candidate: dict):
    paper, _ = _write_source(tmp_path, source)
    candidate_path = tmp_path / "generic_insights.json"
    candidate_path.write_text(json.dumps(candidate), encoding="utf-8")
    return validate_candidate(paper, candidate_path,
                              expected_source_path=".pipeline/paper.md")


# ---------------------------------------------------------------------------
# Region inventory
# ---------------------------------------------------------------------------


def test_region_inventory_preamble_ordering_and_body_spans():
    src = SPEC_SOURCE.encode("utf-8")
    regions = build_region_inventory(src)
    # SPEC_SOURCE starts with the title heading, so no preamble region —
    # but the declaration paragraph lives between h001 and h002... it is
    # inside h001's body. Wait: h001 is "# Tiny Spec" and the declaration
    # paragraph is its body.
    ids = [r.region_id for r in regions]
    assert ids == ["h001", "h002", "h003"]
    h001 = regions[0]
    assert h001.text == "Tiny Spec"
    body = src[h001.body_span[0]:h001.body_span[1]]
    assert b"key words" in body and b"## Rules" not in body
    # Regions never overlap and cover in order.
    for earlier, later in zip(regions, regions[1:]):
        assert earlier.body_span[1] <= later.body_span[0] + len("### ") + len(later.text.encode()) + 2


def test_region_inventory_preamble_and_duplicate_headings():
    src = b"intro text before any heading\n\n# A\n\nbody\n\n## Same\n\nx\n\n## Same\n\ny\n"
    regions = build_region_inventory(src)
    assert regions[0].region_id == "h000" and regions[0].level == 0
    assert src[regions[0].body_span[0]:regions[0].body_span[1]].strip() == \
        b"intro text before any heading"
    same = [r for r in regions if r.normalized == "same"]
    assert [r.occurrence for r in same] == [1, 2]


def test_region_inventory_no_headings_is_whole_document():
    src = b"just prose, no headings at all\n"
    regions = build_region_inventory(src)
    assert len(regions) == 1
    assert regions[0].region_id == "h000"
    assert regions[0].body_span == (0, len(src))


# ---------------------------------------------------------------------------
# Risk units + BCP 14 scanning
# ---------------------------------------------------------------------------


def test_risk_units_split_paragraphs_and_list_items():
    src = SPEC_SOURCE.encode("utf-8")
    units = build_risk_units(src, build_region_inventory(src))
    assert len(units) == 3
    texts = [src[u.span[0]:u.span[1]] for u in units]
    assert texts[0] == b"Attackers can replay tokens."
    assert texts[1] == b"- Log every access attempt."
    assert texts[2] == b"- Rotate signing keys."


def test_risk_headings_match_after_section_number_strip():
    src = (b"# T\n\n## 5.2 Security Considerations\n\nRisk paragraph.\n\n"
           b"## 6 Evaluation\n\nNot a risk section.\n")
    units = build_risk_units(src, build_region_inventory(src))
    assert len(units) == 1
    assert src[units[0].span[0]:units[0].span[1]] == b"Risk paragraph."
    # The closed set is the deterministic gate; similar headings are not
    # silently promoted.
    assert "evaluation" not in MANDATORY_RISK_HEADINGS


def test_bcp14_scan_requires_declaration_and_excludes_boilerplate():
    spec = SPEC_SOURCE.encode("utf-8")
    assert find_bcp14_declaration(spec) is not None
    occurrences = scan_normative_occurrences(spec)
    # The declaration's own token mentions are excluded; three operative
    # occurrences remain, longest-token-first so MUST NOT is never MUST.
    assert [(o.occurrence_id, o.token) for o in occurrences] == [
        ("bcp-0001", "MUST"), ("bcp-0002", "MUST NOT"), ("bcp-0003", "SHOULD"),
    ]
    # Two modals in one sentence are two occurrences sharing a clause.
    assert occurrences[1].clause_span == occurrences[2].clause_span

    # Lowercase modals without a declared framework are not rules.
    undeclared = b"# Doc\n\nYou must be careful. Clients MAY retry.\n"
    assert find_bcp14_declaration(undeclared) is None
    assert scan_normative_occurrences(undeclared) == []


# ---------------------------------------------------------------------------
# Validation matrix — synthetic known-good and corruptions
# ---------------------------------------------------------------------------


def test_simple_candidate_passes(tmp_path):
    result = _validate(tmp_path, SIMPLE_SOURCE,
                       _simple_candidate(_sha(SIMPLE_SOURCE)))
    assert result.deterministic_status == "passed", result.findings
    assert result.semantic_status == "unreviewed"


def test_spec_candidate_passes_with_normative_and_risk_floors(tmp_path):
    result = _validate(tmp_path, SPEC_SOURCE,
                       _spec_candidate(_sha(SPEC_SOURCE)))
    assert result.deterministic_status == "passed", result.findings


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def test_digest_mismatch_is_invalid_before_anything_else(tmp_path):
    candidate = _simple_candidate("0" * 64)
    result = _validate(tmp_path, SIMPLE_SOURCE, candidate)
    assert result.deterministic_status == "invalid"
    assert any("sha256" in f for f in result.findings["source_grounding"])


def test_quote_byte_mismatch_is_invalid(tmp_path):
    candidate = _simple_candidate(_sha(SIMPLE_SOURCE))
    candidate["citations"][0]["quote"] = "We batch gradients and cluster embeddings."
    result = _validate(tmp_path, SIMPLE_SOURCE, candidate)
    assert result.deterministic_status == "invalid"
    assert any("quote does not equal" in f
               for f in result.findings["source_grounding"])


def test_span_outside_named_region_is_invalid(tmp_path):
    candidate = _simple_candidate(_sha(SIMPLE_SOURCE))
    # Claim the Results sentence belongs to the Approach region.
    bad = _citation("c-results", "h002", SIMPLE_SOURCE,
                    "Accuracy improved on the benchmark.")
    candidate["citations"][1] = bad
    result = _validate(tmp_path, SIMPLE_SOURCE, candidate)
    assert result.deterministic_status == "invalid"
    assert any("outside region" in f for f in result.findings["source_grounding"])


def test_unmapped_operative_occurrence_is_invalid(tmp_path):
    candidate = _spec_candidate(_sha(SPEC_SOURCE))
    candidate["insights"] = [r for r in candidate["insights"]
                             if r["id"] != "i-should"]
    result = _validate(tmp_path, SPEC_SOURCE, candidate)
    assert result.deterministic_status == "invalid"
    assert any("bcp-0003" in f and "not mapped" in f
               for f in result.findings["normative"])


def test_strength_mismatch_is_rejected_deterministically(tmp_path):
    candidate = _spec_candidate(_sha(SPEC_SOURCE))
    for rec in candidate["insights"]:
        if rec["id"] == "i-should":
            rec["normative"]["strength"] = "MUST"  # strengthening SHOULD
    result = _validate(tmp_path, SPEC_SOURCE, candidate)
    assert result.deterministic_status == "invalid"
    assert any("does not match occurrence" in f
               for f in result.findings["normative"])


def test_double_mapped_occurrence_is_invalid(tmp_path):
    candidate = _spec_candidate(_sha(SPEC_SOURCE))
    for rec in candidate["insights"]:
        if rec["id"] == "i-mustnot":
            rec["normative"]["occurrence_id"] = "bcp-0001"
    result = _validate(tmp_path, SPEC_SOURCE, candidate)
    assert result.deterministic_status == "invalid"
    assert any("mapped by both" in f for f in result.findings["normative"])


def test_risk_unit_without_dedicated_record_is_invalid(tmp_path):
    candidate = _spec_candidate(_sha(SPEC_SOURCE))
    candidate["insights"] = [r for r in candidate["insights"]
                             if r["id"] != "i-rotate"]
    result = _validate(tmp_path, SPEC_SOURCE, candidate)
    assert result.deterministic_status == "invalid"
    assert any("has no dedicated covering record" in f
               for f in result.findings["risk_coverage"])


def test_wide_citation_cannot_blanket_risk_units(tmp_path):
    """One generic wide citation spanning the whole security section must
    not mark its units covered — each unit's record cites exactly the unit
    span."""
    candidate = _spec_candidate(_sha(SPEC_SOURCE))
    whole = ("Attackers can replay tokens.\n\n- Log every access attempt.\n"
             "- Rotate signing keys.")
    wide = _citation("c-wide", "h003", SPEC_SOURCE, whole)
    candidate["citations"] = [c for c in candidate["citations"]
                              if not c["id"].startswith("c-risk")] + [wide]
    for rec in candidate["insights"]:
        if rec["id"] in {"i-replay", "i-log", "i-rotate"}:
            rec["citation_ids"] = ["c-wide"]
    result = _validate(tmp_path, SPEC_SOURCE, candidate)
    assert result.deterministic_status == "invalid"
    assert len([f for f in result.findings["risk_coverage"]
                if "no dedicated covering record" in f]) == 3


def test_covered_disposition_requires_a_real_citation(tmp_path):
    candidate = _simple_candidate(_sha(SIMPLE_SOURCE))
    candidate["insights"] = [
        {"id": "i-obs", "kind": "empirical_observation",
         "statement": "The paper reports improved benchmark accuracy.",
         "citation_ids": ["c-approach"], "confidence": "medium"},
    ]
    result = _validate(tmp_path, SIMPLE_SOURCE, candidate)
    assert result.deterministic_status == "invalid"
    assert any("declared covered but no substantive record cites" in f
               for f in result.findings["coverage"])


def test_undeclared_heading_is_invalid_and_missing_is_partial(tmp_path):
    candidate = _simple_candidate(_sha(SIMPLE_SOURCE))
    candidate["coverage"]["headings"] = candidate["coverage"]["headings"][:2]
    result = _validate(tmp_path, SIMPLE_SOURCE, candidate)
    assert result.deterministic_status == "invalid"
    assert any("has no coverage disposition" in f
               for f in result.findings["coverage"])

    candidate = _simple_candidate(_sha(SIMPLE_SOURCE))
    candidate["coverage"]["headings"][2] = {
        "region_id": "h003", "disposition": "missing",
        "reason": "results table could not be grounded"}
    candidate["insights"][0]["citation_ids"] = ["c-approach"]
    result = _validate(tmp_path, SIMPLE_SOURCE, candidate)
    assert result.deterministic_status == "partial"
    assert any("h003" in r for r in result.partial_reasons)


def test_empty_vocabulary_needs_reason_and_caps_at_partial(tmp_path):
    candidate = _simple_candidate(_sha(SIMPLE_SOURCE))
    candidate["vocabulary"] = []
    result = _validate(tmp_path, SIMPLE_SOURCE, candidate)
    assert result.deterministic_status == "invalid"  # schema: reason required

    candidate["vocabulary_none_identified_reason"] = \
        "the toy paper defines no specialized terms"
    result = _validate(tmp_path, SIMPLE_SOURCE, candidate)
    assert result.deterministic_status == "partial"
    assert any("none identified" in r for r in result.partial_reasons)


def test_schema_rejects_unknown_fields_and_unknown_major_version(tmp_path):
    candidate = _simple_candidate(_sha(SIMPLE_SOURCE))
    candidate["surprise"] = True
    result = _validate(tmp_path, SIMPLE_SOURCE, candidate)
    assert result.deterministic_status == "invalid"
    assert result.findings["schema"]

    candidate = _simple_candidate(_sha(SIMPLE_SOURCE))
    candidate["schema_version"] = "9.0.0"
    result = _validate(tmp_path, SIMPLE_SOURCE, candidate)
    assert result.deterministic_status == "invalid"
    assert any("unknown major" in f for f in result.findings["schema"])


def test_affordance_must_be_conceptual_only(tmp_path):
    candidate = _simple_candidate(_sha(SIMPLE_SOURCE))
    candidate["insights"].append({
        "id": "i-aff", "kind": "implementation_affordance",
        "statement": "The clustering step could be a standalone helper.",
        "citation_ids": ["c-approach"], "confidence": "low"})
    result = _validate(tmp_path, SIMPLE_SOURCE, candidate)
    assert result.deterministic_status == "invalid"
    assert any("conceptual_only" in f for f in result.findings["schema"])


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def test_render_is_deterministic_and_names_gaps(tmp_path):
    result = _validate(tmp_path, SPEC_SOURCE, _spec_candidate(_sha(SPEC_SOURCE)))
    first = render_insights_md(result)
    second = render_insights_md(result)
    assert first == second
    assert "MUST NOT" in first  # token rendered verbatim from structure
    assert "unreviewed" in first
    assert "## Missing or unresolved coverage" in first
    # Fully covered spec: the section is present but explicitly says so...
    # h000/h001 are not_applicable, so they appear as gap lines.
    assert "not_applicable" in first


def test_render_affordance_carries_fixed_no_code_caveat(tmp_path):
    candidate = _simple_candidate(_sha(SIMPLE_SOURCE))
    candidate["insights"].append({
        "id": "i-aff", "kind": "implementation_affordance",
        "statement": "The clustering step could be a standalone helper.",
        "conceptual_only": True,
        "citation_ids": ["c-approach"], "confidence": "low"})
    result = _validate(tmp_path, SIMPLE_SOURCE, candidate)
    assert result.deterministic_status == "passed"
    text = render_insights_md(result)
    assert "conceptual opportunity only" in text
    assert "no code, conformance, or verification evidence" in text


def test_render_refuses_invalid_result(tmp_path):
    candidate = _simple_candidate("0" * 64)
    result = _validate(tmp_path, SIMPLE_SOURCE, candidate)
    with pytest.raises(ValueError):
        render_insights_md(result)


def test_atomic_write_replaces_never_truncates(tmp_path):
    target = tmp_path / "INSIGHTS.md"
    atomic_write_text(target, "first\n")
    atomic_write_text(target, "second\n")
    assert target.read_text() == "second\n"
    leftovers = [p for p in tmp_path.iterdir() if p.name != "INSIGHTS.md"]
    assert leftovers == []


# ---------------------------------------------------------------------------
# Committed real sources — deterministic inventories
# ---------------------------------------------------------------------------


def test_rfc10008_fixture_matches_contract_inventories():
    """The contract's RFC facts, machine-checked on the committed full
    source: the framework declaration is found, exactly nine operative
    BCP 14 occurrences exist, and Security Considerations yields exactly
    five mandatory-risk units. The fixture digest is bound in PROVENANCE.md."""
    src = RFC_FIXTURE.read_bytes()
    recorded = (RFC_FIXTURE.parent / "PROVENANCE.md").read_text()
    assert hashlib.sha256(src).hexdigest() in recorded

    assert find_bcp14_declaration(src) is not None
    occurrences = scan_normative_occurrences(src)
    assert len(occurrences) == 9
    assert [o.token for o in occurrences] == [
        "MUST", "MUST", "MAY", "MUST", "MAY", "MAY", "MUST NOT", "MUST",
        "SHOULD"]

    regions = build_region_inventory(src)
    security = [r for r in regions
                if "security considerations" in r.normalized]
    assert len(security) == 1
    units = build_risk_units(src, regions)
    assert len(units) == 5
    assert {u.region_id for u in units} == {security[0].region_id}


def test_badge_source_inventories_cleanly():
    """A source-valid method paper: regions resolve, no BCP 14 framework is
    declared (lowercase modals are prose, not rules), and quotes cut from
    any region body validate by construction."""
    for path in (BADGE_SOURCE,):
        src = path.read_bytes()
        regions = build_region_inventory(src)
        assert len(regions) > 3
        assert scan_normative_occurrences(src) == []
        spans = [r.body_span for r in regions]
        assert all(0 <= a <= b <= len(src) for a, b in spans)


# ---------------------------------------------------------------------------
# Write boundary — parity controls on copied runs
# ---------------------------------------------------------------------------


def _tree_state(root: Path) -> dict[str, str]:
    state = {}
    for p in sorted(root.rglob("*")):
        if p.is_file():
            state[str(p.relative_to(root))] = hashlib.sha256(
                p.read_bytes()).hexdigest()
    return state


def _run_cli(paper: Path, candidate: Path, out_validation: Path,
             render: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts/generic_insights.py"),
         "--paper", str(paper), "--candidate", str(candidate),
         "--out-validation", str(out_validation), "--render", str(render)],
        capture_output=True, text=True, timeout=120)


@pytest.mark.parametrize("run_dir", [ICRA_RUN, GBALD_RUN],
                         ids=["icra-explanation-only", "gbald-verified"])
def test_invalid_candidate_writes_only_sidecar_on_copied_run(tmp_path, run_dir):
    """Parity controls: on a copied explanation-only (ICRA) and verified
    (GBALD) run, an invalid candidate writes exactly the requested sidecar,
    never creates the render, never overwrites an existing render, and
    leaves every other byte of the run tree identical. The runs' legacy
    paper maps are irrelevant — this route never reads them."""
    if not run_dir.is_dir():
        pytest.skip(f"parity control run not present: {run_dir.name}")
    copy = tmp_path / "run"
    shutil.copytree(run_dir, copy)
    sentinel = copy / ".pipeline" / "INSIGHTS.md"
    sentinel.write_text("pre-existing render must survive\n", encoding="utf-8")
    before = _tree_state(copy)

    candidate = tmp_path / "candidate.json"
    candidate.write_text(json.dumps({"schema_version": SCHEMA_VERSION}),
                         encoding="utf-8")
    sidecar = copy / ".pipeline" / "generic_insights.validation.json"
    proc = _run_cli(copy / ".pipeline" / "paper.md", candidate, sidecar,
                    sentinel)
    assert proc.returncode == 1, proc.stderr

    after = _tree_state(copy)
    assert sentinel.read_text() == "pre-existing render must survive\n"
    payload = json.loads(sidecar.read_text())
    assert payload["deterministic_status"] == "invalid"
    assert payload["semantic_status"] == "unreviewed"
    changed = {k for k in before if before[k] != after.get(k)}
    created = set(after) - set(before)
    assert changed == set()
    assert created == {str(sidecar.relative_to(copy))}


def test_valid_candidate_writes_exactly_the_requested_outputs(tmp_path):
    """The pure tranche's write boundary: a valid candidate produces the
    sidecar and the render at the explicit paths, and nothing else in the
    surrounding tree changes."""
    run = tmp_path / "run"
    (run / ".pipeline").mkdir(parents=True)
    (run / "METHOD.md").write_text("untouched\n", encoding="utf-8")
    (run / ".pipeline" / "unrelated.json").write_text("{}\n", encoding="utf-8")
    paper = run / ".pipeline" / "paper.md"
    paper.write_text(SPEC_SOURCE, encoding="utf-8")
    candidate_path = run / ".pipeline" / "generic_insights.json"
    candidate_path.write_text(json.dumps(_spec_candidate(_sha(SPEC_SOURCE))),
                              encoding="utf-8")
    before = _tree_state(run)

    sidecar = run / ".pipeline" / "generic_insights.validation.json"
    render = run / ".pipeline" / "INSIGHTS.md"
    proc = _run_cli(paper, candidate_path, sidecar, render)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "deterministic_status: passed" in proc.stdout

    after = _tree_state(run)
    created = set(after) - set(before)
    changed = {k for k in before if before[k] != after.get(k)}
    assert changed == set()
    assert created == {
        str(sidecar.relative_to(run)), str(render.relative_to(run))}
    # Identical inputs -> identical sidecar bytes (no timestamps, no
    # absolute paths).
    first = sidecar.read_bytes()
    proc = _run_cli(paper, candidate_path, sidecar, render)
    assert proc.returncode == 0
    assert sidecar.read_bytes() == first
    payload = json.loads(first)
    assert str(tmp_path) not in first.decode("utf-8")
    assert payload["deterministic_status"] == "passed"


def test_module_never_imports_the_driver():
    """Standalone by contract: importing the module must not pull in
    run_pipeline (no driver coupling, no run discovery)."""
    proc = subprocess.run(
        [sys.executable, "-c",
         "import sys; sys.path.insert(0, 'scripts'); "
         "import generic_insights; "
         "assert 'run_pipeline' not in sys.modules"],
        capture_output=True, text=True, cwd=REPO_ROOT, timeout=60)
    assert proc.returncode == 0, proc.stderr
