"""Insight shadow hook: default-off gating, deterministic candidate
materialization (source binding + quote re-anchoring), and the contract's
full-tree parity gates on copied control run trees across all six
outcomes (passed / partial / invalid / exception / timeout / absent).

Contract: the generic insight artifact contract note (internal, not shipped),
section "Later shadow hook: required failure semantics". Nothing here runs
a model: producer and reviewer dispatches are injected stubs; the live
transport is exercised only by real shadow batches.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import pytest

from generic_insights import (
    build_region_inventory,
    build_risk_units,
    scan_normative_occurrences,
)
from insight_shadow_hook import (
    CANDIDATE_RELPATH,
    EVENT_TYPE,
    PAPER_RELPATH,
    PLACEHOLDER_SHA256,
    PLACEHOLDER_SPAN,
    RECORD_RELPATH,
    RENDER_RELPATH,
    SHADOW_RELPATHS,
    VALIDATION_RELPATH,
    build_insight_producer_prompt,
    insight_shadow_enabled,
    materialize_candidate,
    maybe_run_insight_shadow,
    run_insight_shadow,
)
from run_events import EVENT_TYPES, load_events
from schemas.semantic_review import SCHEMA_VERSION as REVIEW_SCHEMA_VERSION

REPO_ROOT = Path(__file__).resolve().parent.parent
CONTROL_RUNS = [
    # Explanation-only halted control and verified-package control — the
    # contract's two named parity fixtures.
    REPO_ROOT / "example_runs" / "old" / "ICRA21_HICA",
    REPO_ROOT / "example_runs" / "old" / "bayesian-active-learning",
]
CONTROL_IDS = [p.name for p in CONTROL_RUNS]
pytestmark = pytest.mark.skipif(
    not all(p.is_dir() for p in CONTROL_RUNS),
    reason="control run directories (finished pipeline runs) not present")


class FakeDispatchTimeout(Exception):
    """Name-matched by the hook's timeout detection (same class name as
    opencode_client.DispatchTimeout, no client import needed)."""


FakeDispatchTimeout.__name__ = "DispatchTimeout"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def snapshot_tree(root: Path) -> dict[str, str]:
    """Every file under root (including lock dirs' contents) to its
    content hash. Directories are represented implicitly; empty dirs that
    appear/disappear are caught by the lock assertions."""
    snap: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        if path.is_file():
            rel = path.relative_to(root).as_posix()
            snap[rel] = hashlib.sha256(path.read_bytes()).hexdigest()
    return snap


def copy_control(src: Path, tmp_path: Path) -> Path:
    dst = tmp_path / src.name
    shutil.copytree(src, dst)
    return dst


def first_quotable_line(source: bytes, body_span: tuple[int, int]) -> str | None:
    """A verbatim, reasonably long line from a region body — unique enough
    to anchor and definitely present in the source bytes."""
    body = source[body_span[0]:body_span[1]]
    for raw in body.split(b"\n"):
        line = raw.strip(b"\r")
        if len(line) >= 40 and not line.startswith(b"#"):
            return line.decode("utf-8", errors="strict")
    return None


def build_valid_candidate(source: bytes, *, empty_vocab: bool = False) -> dict:
    """A schema-valid, source-grounded candidate for arbitrary paper
    bytes, exactly as the producer agent would emit it: verbatim quotes,
    placeholder spans, placeholder source digest. The hook's
    materialization step supplies the real bindings."""
    regions = build_region_inventory(source)
    risk_units = build_risk_units(source, regions)
    assert not scan_normative_occurrences(source), (
        "control paper unexpectedly declares BCP 14; fixture builder "
        "does not model normative records")

    citations: list[dict] = []
    covered_regions: set[str] = set()

    def add_citation(region_id: str, quote: str) -> str:
        cid = f"c{len(citations) + 1:03d}"
        citations.append({
            "id": cid,
            "region_id": region_id,
            "span": dict(PLACEHOLDER_SPAN),
            "quote": quote,
        })
        covered_regions.add(region_id)
        return cid

    # Two general-purpose citations from the first quotable regions.
    general: list[str] = []
    for region in regions:
        quote = first_quotable_line(source, region.body_span)
        if quote is not None:
            general.append(add_citation(region.region_id, quote))
        if len(general) == 2:
            break
    assert general, "control paper has no quotable region"
    c_main = general[0]
    c_alt = general[-1]

    insights: list[dict] = [{
        "id": "i-behavior",
        "kind": "behavior",
        "statement": "The document describes its mechanism as quoted.",
        "citation_ids": [c_main],
        "vocabulary_ids": [] if empty_vocab else ["v-term"],
        "confidence": "medium",
    }]
    # One dedicated record per mandatory-risk unit, quoting the unit text
    # exactly and completely (span equality comes from re-anchoring).
    for unit in risk_units:
        quote = source[unit.span[0]:unit.span[1]].decode("utf-8")
        cid = add_citation(unit.region_id, quote)
        insights.append({
            "id": f"i-{unit.unit_id}",
            "kind": "limitation",
            "statement": f"Declared boundary from source unit {unit.unit_id}.",
            "citation_ids": [cid],
            "vocabulary_ids": [],
            "confidence": "high",
        })

    candidate: dict = {
        "schema_version": "1.0.0",
        "source": {"path": PAPER_RELPATH, "sha256": PLACEHOLDER_SHA256},
        "document": {
            "kind": "research_method",
            "contribution_kinds": ["fixture_control_account"],
        },
        "summary": {
            "text": "Fixture summary of the document's stated mechanism.",
            "citation_ids": [c_main],
        },
        "intuition": {
            "text": "Fixture intuition grounded in the same passage.",
            "citation_ids": [c_alt],
        },
        "citations": citations,
        "vocabulary": [] if empty_vocab else [{
            "id": "v-term",
            "term": "fixture term",
            "meaning": "paper-local meaning for the fixture",
            "domain": "fixture_domain",
            "confidence": "medium",
            "citation_ids": [c_main],
            "relationships": [],
        }],
        "insights": insights,
        "coverage": {
            "headings": [
                {"region_id": r.region_id, "disposition": "covered"}
                if r.region_id in covered_regions else
                {"region_id": r.region_id, "disposition": "not_applicable",
                 "reason": "not cited by this fixture control candidate"}
                for r in regions
            ],
            "missing": [],
            "unresolved_dependencies": [],
        },
    }
    if empty_vocab:
        candidate["vocabulary_none_identified_reason"] = (
            "fixture models the empty-vocabulary partial outcome")
    return candidate


def producer_stub(run_dir: Path, payload) -> callable:
    """Writes the payload (dict → JSON, str → raw) to the candidate path,
    exactly as a completed producer dispatch would leave it."""
    def dispatch(prompt: str) -> None:
        path = run_dir / CANDIDATE_RELPATH
        if isinstance(payload, dict):
            path.write_text(json.dumps(payload), encoding="utf-8")
        else:
            path.write_text(payload, encoding="utf-8")
    return dispatch


def reviewer_stub(run_dir: Path, *, write: bool = True) -> callable:
    """Writes a hash-bound all-accepted record computed from the live
    materialized candidate — what an ideal reviewer dispatch leaves."""
    def dispatch(prompt: str) -> None:
        if not write:
            return
        source = (run_dir / PAPER_RELPATH).read_bytes()
        candidate_bytes = (run_dir / CANDIDATE_RELPATH).read_bytes()
        record_ids = [
            rec["id"] for rec in json.loads(candidate_bytes)["insights"]]
        record = {
            "schema_version": REVIEW_SCHEMA_VERSION,
            "reviewer": "r2c-insight-semantic-reviewer",
            "source_sha256": hashlib.sha256(source).hexdigest(),
            "candidate_sha256": hashlib.sha256(candidate_bytes).hexdigest(),
            "verdicts": [
                {"record_id": rid, "verdict": "accepted",
                 "reason": "fixture: entailed by its quoted span"}
                for rid in record_ids
            ],
            "needs_human": [],
            "overall_verdict": "accepted",
        }
        (run_dir / RECORD_RELPATH).write_text(
            json.dumps(record), encoding="utf-8")
    return dispatch


def raising_dispatch(exc: Exception) -> callable:
    def dispatch(prompt: str) -> None:
        raise exc
    return dispatch


def noop_dispatch(prompt: str) -> None:
    return None


def assert_parity(
    before: dict[str, str],
    after: dict[str, str],
    *,
    run_rel_prefix: str = "",
) -> set[str]:
    """Byte-hash equality outside the exact namespaced shadow paths plus
    the append-only event log. Returns the changed/created set for
    outcome-specific assertions."""
    allowed = {run_rel_prefix + p for p in SHADOW_RELPATHS}
    allowed.add(run_rel_prefix + ".pipeline/run_events.jsonl")
    removed = set(before) - set(after)
    assert not removed, f"hook removed files outside its scope: {removed}"
    changed = {
        rel for rel in after
        if before.get(rel) != after[rel]
    }
    illegal = changed - allowed
    assert not illegal, f"hook touched files outside the allowlist: {illegal}"
    return changed


def assert_event_delta(run_dir: Path, before_lines: list[str], report: dict) -> None:
    """Exactly one append-only namespaced record: identical prefix, one
    new line, valid sequence, our event type and status."""
    after_lines = (run_dir / ".pipeline/run_events.jsonl").read_text(
        encoding="utf-8").splitlines()
    assert after_lines[:len(before_lines)] == before_lines
    assert len(after_lines) == len(before_lines) + 1
    event = json.loads(after_lines[-1])
    assert event["event_type"] == EVENT_TYPE
    assert event["sequence"] == len(after_lines)
    assert event["stage_id"] == "insight_shadow"
    assert event["status"] == report["outcome"]
    assert event["details"]["outcome"] == report["outcome"]
    # The full log still replays cleanly through the strict loader.
    events = load_events(run_dir / ".pipeline")
    assert events[-1]["event_type"] == EVENT_TYPE


def assert_locks_clean(run_dir: Path, had_run_lock: bool) -> None:
    assert not (run_dir / ".pipeline/run_events.jsonl.lock").exists(), (
        "event append lock left behind")
    assert (run_dir / ".pipeline/_lock").exists() == had_run_lock, (
        "run lock presence changed")


# ---------------------------------------------------------------------------
# Flag gating
# ---------------------------------------------------------------------------


def test_event_type_is_registered() -> None:
    assert EVENT_TYPE in EVENT_TYPES


@pytest.mark.parametrize("value,expected", [
    (None, False), ("", False), ("0", False), ("true", False),
    ("yes", False), ("1", True),
])
def test_enabled_flag_fails_closed(value, expected) -> None:
    env = {} if value is None else {"R2C_INSIGHT_SHADOW": value}
    assert insight_shadow_enabled(env) is expected


def test_default_off_is_a_true_noop(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("R2C_INSIGHT_SHADOW", raising=False)
    run_dir = copy_control(CONTROL_RUNS[0], tmp_path)
    before = snapshot_tree(run_dir)
    assert maybe_run_insight_shadow(run_dir, run_id="t") is None
    assert snapshot_tree(run_dir) == before


def test_maybe_run_never_raises(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("R2C_INSIGHT_SHADOW", "1")
    # A nonexistent run dir must come back as a report, not an exception.
    report = maybe_run_insight_shadow(
        tmp_path / "no-such-run", run_id="t", server_url=None, port=0)
    assert report is not None
    assert report["outcome"] == "absent"


# ---------------------------------------------------------------------------
# Materialization: source binding + re-anchoring
# ---------------------------------------------------------------------------

SOURCE = (
    b"# Title\n\nPreamble is empty here.\n\n"
    b"## Methods\n\nThe quick brown fox jumps over the lazy dog tonight.\n"
    b"A repeated sentence appears here. Filler between the copies.\n"
    b"A repeated sentence appears here. And then the section ends.\n\n"
    b"## Limitations\n\nThis approach only works on parsed Markdown.\n"
)


def _regions_by_text(source: bytes) -> dict[str, object]:
    return {r.text: r for r in build_region_inventory(source)}


def _minimal_payload(citations: list[dict]) -> dict:
    return {"source": {}, "citations": citations, "insights": []}


def test_materialize_fills_source_binding() -> None:
    payload, stats = materialize_candidate(SOURCE, _minimal_payload([]))
    assert payload["source"]["path"] == PAPER_RELPATH
    assert payload["source"]["sha256"] == hashlib.sha256(SOURCE).hexdigest()
    assert stats["source_binding_filled"] is True


def test_materialize_reanchors_unique_quote() -> None:
    methods = _regions_by_text(SOURCE)["Methods"]
    quote = "The quick brown fox jumps over the lazy dog tonight."
    payload, stats = materialize_candidate(SOURCE, _minimal_payload([{
        "id": "c1", "region_id": methods.region_id,
        "span": dict(PLACEHOLDER_SPAN), "quote": quote,
    }]))
    span = payload["citations"][0]["span"]
    assert SOURCE[span["start"]:span["end"]] == quote.encode("utf-8")
    assert methods.body_span[0] <= span["start"]
    assert span["end"] <= methods.body_span[1]
    assert stats["citations_reanchored"] == 1
    assert not stats["citations_unresolved"]
    assert not stats["citations_ambiguous"]


def test_materialize_ambiguous_quote_takes_first_and_reports() -> None:
    methods = _regions_by_text(SOURCE)["Methods"]
    quote = "A repeated sentence appears here."
    payload, stats = materialize_candidate(SOURCE, _minimal_payload([{
        "id": "c1", "region_id": methods.region_id,
        "span": dict(PLACEHOLDER_SPAN), "quote": quote,
    }]))
    span = payload["citations"][0]["span"]
    assert span["start"] == SOURCE.find(quote.encode("utf-8"))
    assert stats["citations_ambiguous"] == ["c1"]


def test_materialize_unresolved_quote_keeps_placeholder() -> None:
    methods = _regions_by_text(SOURCE)["Methods"]
    payload, stats = materialize_candidate(SOURCE, _minimal_payload([{
        "id": "c1", "region_id": methods.region_id,
        "span": dict(PLACEHOLDER_SPAN),
        "quote": "This text is nowhere in the source.",
    }]))
    assert payload["citations"][0]["span"] == PLACEHOLDER_SPAN
    assert stats["citations_unresolved"] == ["c1"]


def test_materialize_quote_outside_named_region_is_unresolved() -> None:
    limitations = _regions_by_text(SOURCE)["Limitations"]
    payload, stats = materialize_candidate(SOURCE, _minimal_payload([{
        "id": "c1", "region_id": limitations.region_id,
        "span": dict(PLACEHOLDER_SPAN),
        "quote": "The quick brown fox jumps over the lazy dog tonight.",
    }]))
    assert payload["citations"][0]["span"] == PLACEHOLDER_SPAN
    assert stats["citations_unresolved"] == ["c1"]


def test_materialize_normative_occurrence_disambiguates() -> None:
    source = (
        b"# Spec\n\nThe key words \"MUST\", \"SHOULD\", and \"MAY\" in this "
        b"document are to be interpreted as described in BCP 14 [RFC2119] "
        b"[RFC8174].\n\n"
        b"## Rules\n\nA server MUST reject bad input. Unrelated filler.\n"
        b"A server MUST reject bad input. Second copy of the clause.\n"
    )
    occurrences = scan_normative_occurrences(source)
    assert len(occurrences) >= 2
    target = occurrences[-1]  # the second copy's MUST
    rules = _regions_by_text(source)["Rules"]
    quote = "A server MUST reject bad input."
    payload, stats = materialize_candidate(source, {
        "source": {},
        "citations": [{
            "id": "c1", "region_id": rules.region_id,
            "span": dict(PLACEHOLDER_SPAN), "quote": quote,
        }],
        "insights": [{
            "id": "i1", "kind": "normative_rule",
            "citation_ids": ["c1"],
            "normative": {"occurrence_id": target.occurrence_id},
        }],
    })
    span = payload["citations"][0]["span"]
    assert span["start"] <= target.token_span[0]
    assert target.token_span[1] <= span["end"]
    # Disambiguated by the bound occurrence, not reported ambiguous.
    assert stats["citations_ambiguous"] == []


def test_materialize_tolerates_malformed_shapes() -> None:
    for junk in (["not", "a", "dict"], {"citations": "nope"},
                 {"citations": [42], "insights": {"x": 1}}):
        payload, _ = materialize_candidate(SOURCE, junk)
        assert payload == junk  # untouched; schema validation reports it


# ---------------------------------------------------------------------------
# Producer prompt
# ---------------------------------------------------------------------------


def test_producer_prompt_contents() -> None:
    regions = build_region_inventory(SOURCE)
    risk_units = build_risk_units(SOURCE, regions)
    assert risk_units, "fixture source must have a Limitations unit"
    unit_texts = [
        (u, SOURCE[u.span[0]:u.span[1]].decode("utf-8")) for u in risk_units]
    prompt = build_insight_producer_prompt(
        paper_path="/abs/paper.md",
        candidate_path="/abs/generic_insights.json",
        regions=regions,
        risk_units=unit_texts,
        occurrences=[],
    )
    assert "/abs/generic_insights.json" in prompt
    assert "/abs/paper.md" in prompt
    assert "Think-class procedural anchor" in prompt
    assert PLACEHOLDER_SHA256 in prompt
    assert "VERBATIM" in prompt
    for r in regions:
        assert f"`{r.region_id}`" in prompt
    for u in risk_units:
        assert u.unit_id in prompt
    assert "This approach only works on parsed Markdown." in prompt
    assert "Writeable paths" in prompt
    # No BCP 14 section when the source declares none.
    assert "Normative BCP 14 occurrences" not in prompt
    assert "{expected_output_path}" not in prompt, "anchor placeholder leaked"


# ---------------------------------------------------------------------------
# Full-tree parity on the two control runs, all six outcomes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("control", CONTROL_RUNS, ids=CONTROL_IDS)
@pytest.mark.parametrize("outcome", [
    "passed", "partial", "invalid", "exception", "timeout", "absent"])
def test_parity_across_outcomes(tmp_path, control, outcome) -> None:
    run_dir = copy_control(control, tmp_path)
    had_run_lock = (run_dir / ".pipeline/_lock").exists()
    before = snapshot_tree(run_dir)
    before_events = (run_dir / ".pipeline/run_events.jsonl").read_text(
        encoding="utf-8").splitlines()
    source = (run_dir / PAPER_RELPATH).read_bytes()

    if outcome == "passed":
        producer = producer_stub(run_dir, build_valid_candidate(source))
    elif outcome == "partial":
        producer = producer_stub(
            run_dir, build_valid_candidate(source, empty_vocab=True))
    elif outcome == "invalid":
        producer = producer_stub(run_dir, "{not json at all")
    elif outcome == "exception":
        producer = raising_dispatch(RuntimeError("backend fell over"))
    elif outcome == "timeout":
        producer = raising_dispatch(FakeDispatchTimeout("900s ceiling"))
    else:  # absent — dispatch completes but writes nothing
        producer = noop_dispatch

    report = run_insight_shadow(
        run_dir, run_id="parity-test",
        producer_dispatch=producer,
        reviewer_dispatch=reviewer_stub(run_dir),
    )

    assert report["outcome"] == outcome
    assert report["event_appended"] is True, report["event_error"]
    changed = assert_parity(before, snapshot_tree(run_dir))
    assert_event_delta(run_dir, before_events, report)
    assert_locks_clean(run_dir, had_run_lock)

    candidate = run_dir / CANDIDATE_RELPATH
    validation = run_dir / VALIDATION_RELPATH
    render = run_dir / RENDER_RELPATH
    record = run_dir / RECORD_RELPATH
    if outcome in {"passed", "partial"}:
        assert candidate.exists() and validation.exists() and render.exists()
        assert record.exists()
        assert report["review"] == {
            "status": "present", "reason": "present",
            "overall_verdict": "accepted"}
        sidecar = json.loads(validation.read_text(encoding="utf-8"))
        assert sidecar["deterministic_status"] == outcome
        assert sidecar["semantic_status"] == "unreviewed"
        # Re-anchoring resolved every fixture quote.
        stats = report["materialization"]
        assert stats["citations_unresolved"] == []
    elif outcome == "invalid":
        assert candidate.exists() and validation.exists()
        assert not render.exists(), "invalid output must never render"
        assert not record.exists(), "invalid output must never be reviewed"
        assert report["review"]["status"] == "not_dispatched"
    else:  # exception, timeout, absent
        assert not candidate.exists()
        assert not validation.exists()
        assert not render.exists()
        assert not record.exists()
        # Only the event log changed at all.
        assert changed == {".pipeline/run_events.jsonl"}


@pytest.mark.parametrize("control", CONTROL_RUNS, ids=CONTROL_IDS)
def test_stale_artifacts_removed_before_generation(tmp_path, control) -> None:
    """A failed invocation leaves NO stale shadow output from an earlier
    invocation behind — absence, never a stale result read as current."""
    run_dir = copy_control(control, tmp_path)
    for rel in SHADOW_RELPATHS:
        path = run_dir / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("stale artifact from an earlier invocation")
    report = run_insight_shadow(
        run_dir, run_id="stale-test",
        producer_dispatch=raising_dispatch(RuntimeError("boom")),
        reviewer_dispatch=reviewer_stub(run_dir),
    )
    assert report["outcome"] == "exception"
    assert sorted(report["stale_removed"]) == sorted(SHADOW_RELPATHS)
    for rel in SHADOW_RELPATHS:
        assert not (run_dir / rel).exists()


@pytest.mark.parametrize("reviewer_behavior", ["silent", "raises"])
def test_review_failure_never_demotes_outcome(
        tmp_path, reviewer_behavior) -> None:
    """A reviewer that writes nothing (or dies) leaves the deterministic
    outcome untouched; absence of review is never demotion."""
    run_dir = copy_control(CONTROL_RUNS[1], tmp_path)
    source = (run_dir / PAPER_RELPATH).read_bytes()
    reviewer = (reviewer_stub(run_dir, write=False)
                if reviewer_behavior == "silent"
                else raising_dispatch(FakeDispatchTimeout("reviewer hung")))
    report = run_insight_shadow(
        run_dir, run_id="review-fail-test",
        producer_dispatch=producer_stub(
            run_dir, build_valid_candidate(source)),
        reviewer_dispatch=reviewer,
    )
    assert report["outcome"] == "passed"
    assert (run_dir / RENDER_RELPATH).exists()
    if reviewer_behavior == "silent":
        assert report["review"]["status"] == "absent"
    else:
        assert report["review"]["status"] == "timeout"
    assert report["event_appended"] is True


def test_event_failure_is_swallowed(tmp_path, monkeypatch) -> None:
    """An event-log failure is recorded in the report and swallowed —
    best-effort in every outcome."""
    run_dir = copy_control(CONTROL_RUNS[0], tmp_path)
    # A pre-existing foreign lock makes the append fail deterministically.
    (run_dir / ".pipeline/run_events.jsonl.lock").mkdir()
    report = run_insight_shadow(
        run_dir, run_id="event-fail-test",
        producer_dispatch=raising_dispatch(RuntimeError("boom")),
        reviewer_dispatch=reviewer_stub(run_dir),
    )
    assert report["outcome"] == "exception"
    assert report["event_appended"] is False
    assert "locked" in report["event_error"]


# ---------------------------------------------------------------------------
# Render-equivalent re-anchoring (queue item 11 consumed insight-side) and
# the targeted corrective nudge (queue item 12). Evidence: Think producers
# emit render-equivalent but byte-different quotes (math delimiters,
# emphasis, whitespace) and cannot transcribe the difference on retry; a
# quote neither layer resolves previously left the candidate honest-invalid
# with no recovery path short of regenerating unrelated analysis.
# ---------------------------------------------------------------------------

MATH_SOURCE = (
    b"# Title\n\nPreamble is empty here.\n\n"
    b"## Methods\n\nThe update rule $\\sigma_i = \\min(\\sigma_{max}, "
    b"\\sigma_{i-1} + \\delta v_i)$ caps the growth per step.\n"
    b"Ambiguous flavor text repeats now. Ambiguous flavor text repeats "
    b"now.\n"
)


def test_materialize_render_equivalent_quote_adopts_source_bytes() -> None:
    methods = _regions_by_text(MATH_SOURCE)["Methods"]
    # MathJax delimiters instead of the source's dollars — byte-different,
    # render-equivalent (delta class B).
    quote = ("The update rule \\(\\sigma_i = \\min(\\sigma_{max}, "
             "\\sigma_{i-1} + \\delta v_i)\\) caps the growth per step.")
    payload, stats = materialize_candidate(MATH_SOURCE, _minimal_payload([{
        "id": "c1", "region_id": methods.region_id,
        "span": dict(PLACEHOLDER_SPAN), "quote": quote,
    }]))
    cit = payload["citations"][0]
    span = cit["span"]
    # The quote now IS the source's own bytes at the resolved span.
    assert MATH_SOURCE[span["start"]:span["end"]] == \
        cit["quote"].encode("utf-8")
    assert "$\\sigma_i" in cit["quote"]  # the source's delimiters, not ours
    assert stats["citations_render_reanchored"] == ["c1"]
    assert stats["citations_reanchored"] == 1
    assert not stats["citations_unresolved"]


def test_materialize_render_ambiguous_quote_stays_unresolved() -> None:
    methods = _regions_by_text(MATH_SOURCE)["Methods"]
    # Folded form occurs twice; normalized matching never chooses among
    # occurrences, so the honest failure path keeps the placeholder.
    payload, stats = materialize_candidate(MATH_SOURCE, _minimal_payload([{
        "id": "c1", "region_id": methods.region_id,
        "span": dict(PLACEHOLDER_SPAN),
        "quote": "Ambiguous  flavor text repeats now.",
    }]))
    assert payload["citations"][0]["span"] == PLACEHOLDER_SPAN
    assert stats["citations_unresolved"] == ["c1"]
    assert stats["citations_render_reanchored"] == []


def test_apply_quote_corrections_touches_only_listed_unresolved_ids() -> None:
    from insight_shadow_hook import apply_quote_corrections

    payload = _minimal_payload([
        {"id": "c1", "region_id": "h001", "span": dict(PLACEHOLDER_SPAN),
         "quote": "wrong quote"},
        {"id": "c2", "region_id": "h001", "span": {"start": 5, "end": 9},
         "quote": "already resolved"},
    ])
    corrections = {"corrections": [
        {"id": "c1", "quote": "fixed verbatim quote"},
        {"id": "c2", "quote": "attempted overwrite of a resolved citation"},
        {"id": "c9", "quote": "unknown id"},
        "not-a-dict",
        {"id": "c1"},  # missing quote
    ]}
    applied = apply_quote_corrections(payload, corrections, ["c1"])
    assert applied == ["c1"]
    assert payload["citations"][0]["quote"] == "fixed verbatim quote"
    assert payload["citations"][1]["quote"] == "already resolved"
    # Malformed container shapes apply nothing.
    assert apply_quote_corrections(payload, ["nope"], ["c1"]) == []
    assert apply_quote_corrections(payload, {"corrections": "x"}, ["c1"]) == []
    # An identical re-copy resolves nothing and is not counted.
    assert apply_quote_corrections(
        payload, {"corrections": [{"id": "c1",
                                   "quote": "fixed verbatim quote"}]},
        ["c1"]) == []


def test_correction_prompt_contents() -> None:
    from insight_shadow_hook import build_insight_correction_prompt

    prompt = build_insight_correction_prompt(
        paper_path="/run/.pipeline/paper.md",
        corrections_path="/run/.pipeline/generic_insights.corrections.json",
        unresolved=[{"id": "c7", "region_id": "h002",
                     "region_text": "Methods",
                     "quote": "the quote that failed"}],
    )
    assert "/run/.pipeline/generic_insights.corrections.json" in prompt
    assert "`c7`" in prompt
    assert "the quote that failed" in prompt
    assert "ONLY the citation ids listed below" in prompt
    assert "OMIT that id" in prompt


def _correcting_producer(run_dir: Path, first_payload: dict,
                         corrections) -> callable:
    """First call writes the candidate; the corrective call writes the
    corrections file (dict → JSON, str → raw, None → nothing)."""
    calls = {"n": 0}

    def dispatch(prompt: str) -> None:
        calls["n"] += 1
        if calls["n"] == 1:
            (run_dir / CANDIDATE_RELPATH).write_text(
                json.dumps(first_payload), encoding="utf-8")
        elif corrections is not None:
            from insight_shadow_hook import CORRECTIONS_RELPATH
            path = run_dir / CORRECTIONS_RELPATH
            if isinstance(corrections, dict):
                path.write_text(json.dumps(corrections), encoding="utf-8")
            else:
                path.write_text(corrections, encoding="utf-8")
    return dispatch


@pytest.mark.parametrize("control", CONTROL_RUNS[:1], ids=CONTROL_IDS[:1])
def test_corrective_nudge_recovers_unresolved_quote(tmp_path, control) -> None:
    run_dir = copy_control(control, tmp_path)
    source = (run_dir / PAPER_RELPATH).read_bytes()
    candidate = build_valid_candidate(source)
    # Break one citation's quote beyond render-equivalence.
    good_quote = candidate["citations"][0]["quote"]
    candidate["citations"][0]["quote"] = (
        "A paraphrase that appears nowhere in the source document at all.")
    before = snapshot_tree(run_dir)
    events_before = (run_dir / ".pipeline/run_events.jsonl").read_text(
        encoding="utf-8").splitlines()

    report = run_insight_shadow(
        run_dir, run_id=run_dir.name,
        producer_dispatch=_correcting_producer(
            run_dir, candidate,
            {"corrections": [
                {"id": candidate["citations"][0]["id"],
                 "quote": good_quote}]}),
        reviewer_dispatch=reviewer_stub(run_dir),
    )
    assert report["correction"]["status"] == "applied"
    assert report["correction"]["unresolved_before"] == [
        candidate["citations"][0]["id"]]
    assert report["correction"]["unresolved_after"] == []
    assert report["outcome"] in ("passed", "partial"), report
    # The merged candidate carries the corrected quote verbatim.
    merged = json.loads((run_dir / CANDIDATE_RELPATH).read_bytes())
    assert merged["citations"][0]["quote"] == good_quote
    # Full-tree parity still holds (the corrections file is a namespaced
    # shadow write) and exactly one event was appended.
    assert_parity(before, snapshot_tree(run_dir))
    assert_event_delta(run_dir, events_before, report)
    event = json.loads((run_dir / ".pipeline/run_events.jsonl").read_text(
        encoding="utf-8").splitlines()[-1])
    assert event["details"]["correction"]["status"] == "applied"


@pytest.mark.parametrize("mode", ["error", "timeout", "absent", "no_effect"])
def test_corrective_nudge_failure_never_demotes_outcome(
        tmp_path, mode) -> None:
    run_dir = copy_control(CONTROL_RUNS[0], tmp_path)
    source = (run_dir / PAPER_RELPATH).read_bytes()
    candidate = build_valid_candidate(source)
    bad_id = candidate["citations"][0]["id"]
    candidate["citations"][0]["quote"] = (
        "A paraphrase that appears nowhere in the source document at all.")
    before = snapshot_tree(run_dir)

    if mode == "error":
        calls = {"n": 0}

        def dispatch(prompt: str) -> None:
            calls["n"] += 1
            if calls["n"] == 1:
                (run_dir / CANDIDATE_RELPATH).write_text(
                    json.dumps(candidate), encoding="utf-8")
            else:
                raise RuntimeError("corrective dispatch failed")
    elif mode == "timeout":
        calls = {"n": 0}

        def dispatch(prompt: str) -> None:
            calls["n"] += 1
            if calls["n"] == 1:
                (run_dir / CANDIDATE_RELPATH).write_text(
                    json.dumps(candidate), encoding="utf-8")
            else:
                raise FakeDispatchTimeout("corrective dispatch timed out")
    elif mode == "absent":
        dispatch = _correcting_producer(run_dir, candidate, None)
    else:  # no_effect: correction names a citation that is not unresolved
        dispatch = _correcting_producer(
            run_dir, candidate,
            {"corrections": [{"id": "not-a-real-id", "quote": "whatever"}]})

    report = run_insight_shadow(
        run_dir, run_id=run_dir.name,
        producer_dispatch=dispatch,
        reviewer_dispatch=reviewer_stub(run_dir),
    )
    expected_status = {"error": "error", "timeout": "timeout",
                       "absent": "absent", "no_effect": "no_effect"}[mode]
    assert report["correction"]["status"] == expected_status
    assert report["correction"]["unresolved_before"] == [bad_id]
    # The outcome stays whatever the first validation said — a correction
    # failure never raises and never demotes further.
    assert report["outcome"] == "invalid"
    assert_parity(before, snapshot_tree(run_dir))


def test_corrective_nudge_not_dispatched_when_all_quotes_resolve(
        tmp_path) -> None:
    run_dir = copy_control(CONTROL_RUNS[0], tmp_path)
    source = (run_dir / PAPER_RELPATH).read_bytes()
    calls = {"n": 0}
    inner = producer_stub(run_dir, build_valid_candidate(source))

    def counting(prompt: str) -> None:
        calls["n"] += 1
        inner(prompt)

    report = run_insight_shadow(
        run_dir, run_id=run_dir.name,
        producer_dispatch=counting,
        reviewer_dispatch=reviewer_stub(run_dir),
    )
    assert calls["n"] == 1  # exactly one producer dispatch, no nudge
    assert report["correction"]["status"] == "not_needed"
    assert report["outcome"] in ("passed", "partial")
