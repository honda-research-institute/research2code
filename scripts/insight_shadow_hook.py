#!/usr/bin/env python3
"""Insight shadow hook — in-run generic-insight candidate generation plus
semantic-reviewer dispatch, best-effort and non-blocking in every outcome.

Contract: the generic insight artifact contract note (internal, not shipped)
(section "Later shadow hook: required failure semantics") and
the insight semantic reviewer design note (internal, not shipped). The driver
calls exactly one function here (`maybe_run_insight_shadow`); everything
else is standalone so fixtures exercise the full flow with injected
dispatch functions and no live model, server, or network.

Failure semantics (the load-bearing part):

- Default OFF. The hook is inert unless `R2C_INSIGHT_SHADOW=1`.
- Missing, partial, invalid, timed-out, or exception outcomes never raise
  into the driver, never change stage status, halts, delivery, claims, or
  any pre-existing file.
- The hook's writes are exactly the five namespaced shadow paths
  (candidate, validation sidecar, internal INSIGHTS.md render, semantic
  acceptance record, corrective-quote file) plus one append-only
  namespaced run_events record.
- Stale shadow artifacts from an earlier invocation are removed up front,
  so whatever exists afterwards belongs to THIS invocation; absence is
  never mistaken for a result.
- Only a source-digest-matched `deterministic_status` of passed/partial
  renders internally; an invalid candidate never renders.

Candidate materialization: the producer agent copies verbatim quotes but
cannot compute byte offsets or digests, so it emits placeholder spans and a
placeholder source digest. This module re-anchors every citation quote to
its exact byte span inside the named validator-owned region (the
contract's "candidate interpretations must be re-anchored to the
authoritative text") and fills the source binding deterministically. A
quote that does not occur verbatim in its region keeps the placeholder and
is reported by the validator — re-anchoring never edits meaning, only
resolves exact bytes the producer already committed to. Two bounded
recovery layers sit between an inexact quote and the honest failure
path (queue items 11 and 12): a unique render-equivalent fold match
replaces the quote with the source's own bytes deterministically, and
quotes that still fail get one targeted corrective re-ask whose answers
merge back as quote fields only.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Callable

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:  # pragma: no cover - import plumbing
    sys.path.insert(0, str(_REPO_ROOT))
if str(_REPO_ROOT / "scripts") not in sys.path:  # pragma: no cover
    sys.path.insert(0, str(_REPO_ROOT / "scripts"))

from dispatch_templates import (  # noqa: E402
    THINK_CLASS_ANCHOR,
    build_insight_review_prompt,
    format_writeable_paths_block,
)
from generic_insights import (  # noqa: E402
    atomic_write_text,
    build_region_inventory,
    build_risk_units,
    render_insights_md,
    scan_normative_occurrences,
    validate_candidate,
    validation_sidecar_payload,
)
from insight_semantic_review import (  # noqa: E402
    RECORD_RELPATH,
    build_review_bundles,
    load_acceptance_record,
)
from quote_reanchor import find_unique_span  # noqa: E402
from schemas.generic_insights import SCHEMA_VERSION  # noqa: E402

# Enable flag: the committed hook is inert until a batch opts in.
SHADOW_ENV = "R2C_INSIGHT_SHADOW"

PRODUCER_AGENT = "r2c-insight-producer"
REVIEWER_AGENT = "r2c-insight-semantic-reviewer"

PAPER_RELPATH = ".pipeline/paper.md"
CANDIDATE_RELPATH = ".pipeline/generic_insights.json"
VALIDATION_RELPATH = ".pipeline/generic_insights.validation.json"
RENDER_RELPATH = ".pipeline/INSIGHTS.md"
CORRECTIONS_RELPATH = ".pipeline/generic_insights.corrections.json"
# RECORD_RELPATH (acceptance record) imported above; the five together are
# the hook's complete file-write surface inside a run tree.
SHADOW_RELPATHS = (
    CANDIDATE_RELPATH, VALIDATION_RELPATH, RENDER_RELPATH, RECORD_RELPATH,
    CORRECTIONS_RELPATH,
)

# The one namespaced event this hook appends (registered in
# scripts/run_events.py EVENT_TYPES).
EVENT_TYPE = "insight_shadow_recorded"
EVENT_STAGE_ID = "insight_shadow"

# Per-dispatch wall ceiling. Generation reads a whole paper and writes a
# large JSON object; review reads sliced bundles. Both match the driver's
# standard producer window.
DEFAULT_TIMEOUT_S = 900.0

PLACEHOLDER_SHA256 = "0" * 64
PLACEHOLDER_SPAN = {"start": 0, "end": 1}

# A dispatch function receives the fully built prompt and must leave the
# agent's artifact at the expected path (or not, on failure). Injected by
# fixtures; live runs use _make_live_dispatch.
DispatchFn = Callable[[str], None]


def insight_shadow_enabled(env: dict[str, str] | None = None) -> bool:
    """True only on the exact opt-in value. Anything else — unset, empty,
    "0", "true" — leaves the hook inert (default OFF, fail closed)."""
    source = os.environ if env is None else env
    return source.get(SHADOW_ENV) == "1"


# ---------------------------------------------------------------------------
# Producer prompt
# ---------------------------------------------------------------------------

_PRODUCER_TASK_TEMPLATE = (
    "## Task: source-grounded generic-insight candidate\n"
    "\n"
    "Read the authoritative parsed source at:\n"
    "\n"
    "    {paper_path}\n"
    "\n"
    "and write ONE JSON candidate (schema_version \"{schema_version}\", "
    "single object, no wrapper key, no Markdown fences) to exactly:\n"
    "\n"
    "    {candidate_path}\n"
    "\n"
    "The candidate is a source-grounded account of what this document says "
    "— NOT an implementation, NOT a claim that code exists. A deterministic "
    "validator re-anchors your quotes and judges the result after you "
    "return; a separate reviewer judges semantic entailment. Your quotes "
    "must therefore be VERBATIM copies of source bytes (same characters, "
    "whitespace, and Markdown emphasis). Set every citation \"span\" to the "
    "placeholder {{\"start\": 0, \"end\": 1}} and \"source\" to "
    "{{\"path\": \"{source_relpath}\", \"sha256\": \"{placeholder_sha}\"}} "
    "— the validator fills the real offsets and digest from your exact "
    "quote text.\n"
    "\n"
    "### Shape (all fields required unless noted)\n"
    "\n"
    "```json\n"
    "{{\n"
    "  \"schema_version\": \"{schema_version}\",\n"
    "  \"source\": {{\"path\": \"{source_relpath}\", "
    "\"sha256\": \"{placeholder_sha}\"}},\n"
    "  \"document\": {{\"kind\": \"research_method | technical_standard | "
    "other\", \"contribution_kinds\": [\"<paper-local slug>\"]}},\n"
    "  \"summary\": {{\"text\": \"what the document does, 2-6 sentences\", "
    "\"citation_ids\": [\"c1\"]}},\n"
    "  \"intuition\": {{\"text\": \"plain-language mechanism/rationale\", "
    "\"citation_ids\": [\"c2\"]}},\n"
    "  \"citations\": [{{\"id\": \"c1\", \"region_id\": \"h001\", "
    "\"span\": {{\"start\": 0, \"end\": 1}}, "
    "\"quote\": \"<verbatim source text>\"}}],\n"
    "  \"vocabulary\": [{{\"id\": \"v1\", \"term\": \"...\", "
    "\"meaning\": \"paper-local meaning\", \"domain\": \"<paper-local "
    "domain slug>\", \"confidence\": \"high|medium|low\", "
    "\"citation_ids\": [\"c1\"], \"relationships\": "
    "[{{\"to_id\": \"v2\", \"kind\": \"uses\"}}]}}],\n"
    "  \"insights\": [{{\"id\": \"i1\", \"kind\": \"behavior | "
    "normative_rule | implementation_affordance | limitation | "
    "security_consideration | empirical_observation\", "
    "\"statement\": \"...\", \"citation_ids\": [\"c1\"], "
    "\"vocabulary_ids\": [], \"confidence\": \"high|medium|low\"}}],\n"
    "  \"coverage\": {{\"headings\": [{{\"region_id\": \"h001\", "
    "\"disposition\": \"covered | not_applicable | missing\", "
    "\"reason\": \"required unless covered\"}}], \"missing\": [], "
    "\"unresolved_dependencies\": []}}\n"
    "}}\n"
    "```\n"
    "\n"
    "Field rules the validator enforces:\n"
    "\n"
    "- every id is unique; every referenced id resolves;\n"
    "- every substantive item (summary, intuition, vocabulary meanings, "
    "insight records) carries at least one citation that actually supports "
    "it;\n"
    "- `normative_rule` records carry a structured `normative` object "
    "instead of `statement` (only when the prompt lists BCP 14 "
    "occurrences); all other kinds carry prose `statement`;\n"
    "- `implementation_affordance` records set `\"conceptual_only\": true` "
    "(the field is forbidden on other kinds);\n"
    "- a `behavior` may carry ordered `steps`; a `limitation` may point at "
    "the insight it qualifies via `qualifies_insight_id`;\n"
    "- empty vocabulary requires `vocabulary_none_identified_reason`;\n"
    "- reported results stay attributed observations "
    "(`empirical_observation`), never upgraded into guarantees; ambiguous "
    "or contradictory passages become `limitation` records, never silently "
    "resolved.\n"
    "\n"
    "Aim for 4-15 vocabulary records and 6-20 insight records — substantive "
    "but bounded; depth beats bulk."
)

_REGION_INVENTORY_HEADER = (
    "### Coverage regions (validator-owned; classify EVERY id below "
    "exactly once in coverage.headings)\n"
    "\n"
    "`covered` only where one of your records cites inside that section; "
    "otherwise `not_applicable` or `missing` with a plain reason. Cite each "
    "quote with the region id whose section actually contains it."
)

_RISK_UNITS_HEADER = (
    "### Mandatory-risk units (each needs its own dedicated record)\n"
    "\n"
    "For EACH unit below, write one dedicated record (kind `limitation` or "
    "`security_consideration`, or a matching facet) whose citation quotes "
    "the unit's text EXACTLY and COMPLETELY as printed here:"
)

_OCCURRENCES_HEADER = (
    "### Normative BCP 14 occurrences (each maps to exactly one "
    "`normative_rule` record)\n"
    "\n"
    "This source declares the BCP 14 framework. Map EACH occurrence id "
    "below to exactly one `normative_rule` record: set `normative` to "
    "{\"framework\": \"BCP 14\", \"strength\": \"<the token, underscored "
    "form>\", \"subject\": ..., \"action\": ..., \"conditions\": [...], "
    "\"exceptions\": [...], \"occurrence_id\": \"<the id>\"} and make the "
    "record's quote enclose that token's clause verbatim. Never merge two "
    "occurrences into one record."
)


def build_insight_producer_prompt(
    *,
    paper_path: str,
    candidate_path: str,
    regions: list,
    risk_units: list,
    occurrences: list,
) -> str:
    """Assemble the producer's dispatch prompt: task + think anchor +
    validator-owned inventories (region ids, risk units, BCP 14
    occurrences) + writeable paths. Inventories carry ids and exact text,
    never byte offsets — offsets are the deterministic layer's job."""
    sections = [
        _PRODUCER_TASK_TEMPLATE.format(
            paper_path=paper_path,
            candidate_path=candidate_path,
            schema_version=SCHEMA_VERSION,
            source_relpath=PAPER_RELPATH,
            placeholder_sha=PLACEHOLDER_SHA256,
        ),
        THINK_CLASS_ANCHOR.format(expected_output_path=candidate_path),
    ]

    region_lines = []
    for r in regions:
        label = r.text if r.text else "(synthetic preamble/whole-document)"
        region_lines.append(f"- `{r.region_id}` (level {r.level}): {label}")
    sections.append(_REGION_INVENTORY_HEADER + "\n\n" + "\n".join(region_lines))

    if risk_units:
        unit_lines = []
        for u, unit_text in risk_units:
            unit_lines.append(
                f"- unit `{u.unit_id}` in region `{u.region_id}`:\n"
                f"  ```\n  {unit_text}\n  ```")
        sections.append(_RISK_UNITS_HEADER + "\n\n" + "\n".join(unit_lines))

    if occurrences:
        occ_lines = []
        for o, clause_text in occurrences:
            occ_lines.append(
                f"- occurrence `{o.occurrence_id}` (token {o.token}): "
                f"clause `{clause_text}`")
        sections.append(_OCCURRENCES_HEADER + "\n\n" + "\n".join(occ_lines))

    sections.append(format_writeable_paths_block([candidate_path]))
    return "\n\n".join(sections)


# ---------------------------------------------------------------------------
# Targeted corrective nudge (queue item 12, 2026-07-27)
#
# Root cause this closes: a producer quote that fails re-anchoring left the
# whole candidate honest-invalid with no recovery path — the only options
# were accepting the invalid candidate (never) or regenerating everything
# (which re-rolls unrelated analysis and usually reproduces the same quote
# defect). The nudge re-asks for EXACTLY the quotes that failed, and the
# deterministic layer merges only those quote fields back — unrelated
# analysis is untouched mechanically, not by trust.
# ---------------------------------------------------------------------------

_CORRECTION_TASK_TEMPLATE = (
    "## Task: corrective quote re-copy (targeted)\n"
    "\n"
    "Your earlier insight candidate for this document is kept as-is; only "
    "the citation quote(s) listed below could not be located verbatim in "
    "the authoritative parsed source at:\n"
    "\n"
    "    {paper_path}\n"
    "\n"
    "For EACH citation below, re-read its region in that file and re-copy "
    "the supporting passage as an EXACT byte-for-byte span: same "
    "characters, whitespace, Markdown emphasis, and the source's own math "
    "delimiters. Copy from the file, never from memory.\n"
    "\n"
    "Write ONE JSON object (no Markdown fences) to exactly:\n"
    "\n"
    "    {corrections_path}\n"
    "\n"
    "```json\n"
    "{{\n"
    "  \"corrections\": [\n"
    "    {{\"id\": \"<citation id>\", \"quote\": \"<verbatim source "
    "text>\"}}\n"
    "  ]\n"
    "}}\n"
    "```\n"
    "\n"
    "Rules:\n"
    "\n"
    "- include ONLY the citation ids listed below — a deterministic merge "
    "applies your quotes to those citations and nothing else;\n"
    "- the quote must support the same statement the citation already "
    "backs — you are re-copying evidence, never changing meaning;\n"
    "- if the region genuinely contains no verbatim passage supporting "
    "the claim, OMIT that id — an honest omission is valid."
)


def build_insight_correction_prompt(
    *,
    paper_path: str,
    corrections_path: str,
    unresolved: list[dict],
) -> str:
    """Assemble the corrective dispatch prompt: task + think anchor + the
    exact failing citations (id, region, current quote) + writeable
    paths. Only the listed ids are mergeable, so the prompt and the merge
    enforce the same boundary."""
    lines = []
    for item in unresolved:
        region_label = item.get("region_text") or "(untitled region)"
        lines.append(
            f"- citation `{item['id']}` in region `{item['region_id']}` "
            f"({region_label}); its current quote was NOT found verbatim:\n"
            f"  ```\n  {item['quote']}\n  ```")
    sections = [
        _CORRECTION_TASK_TEMPLATE.format(
            paper_path=paper_path, corrections_path=corrections_path),
        THINK_CLASS_ANCHOR.format(expected_output_path=corrections_path),
        "### Citations to re-copy\n\n" + "\n".join(lines),
        format_writeable_paths_block([corrections_path]),
    ]
    return "\n\n".join(sections)


def _unresolved_citation_details(
    payload: Any, regions: list, unresolved_ids: list[str],
) -> list[dict]:
    """(id, region_id, region_text, quote) for each unresolved citation
    that carries enough shape to be re-asked."""
    region_text = {r.region_id: (r.text or "") for r in regions}
    details: list[dict] = []
    citations = payload.get("citations") if isinstance(payload, dict) else None
    if not isinstance(citations, list):
        return details
    for cit in citations:
        if not isinstance(cit, dict) or cit.get("id") not in unresolved_ids:
            continue
        quote = cit.get("quote")
        if not isinstance(quote, str) or not quote:
            continue  # nothing to re-copy against; schema reports it
        details.append({
            "id": cit["id"],
            "region_id": str(cit.get("region_id") or ""),
            "region_text": region_text.get(cit.get("region_id"), ""),
            "quote": quote,
        })
    return details


def apply_quote_corrections(
    payload: Any, corrections: Any, unresolved_ids: list[str],
) -> list[str]:
    """Merge corrective quotes into the candidate, quote fields only.

    Only citations named in `unresolved_ids` are touchable — the nudge can
    never modify a citation that already resolved, and it can never touch
    any field but `quote`. Returns the applied citation ids. Malformed
    corrections shapes apply nothing."""
    if not isinstance(payload, dict) or not isinstance(corrections, dict):
        return []
    entries = corrections.get("corrections")
    citations = payload.get("citations")
    if not isinstance(entries, list) or not isinstance(citations, list):
        return []
    by_id = {
        cit.get("id"): cit for cit in citations
        if isinstance(cit, dict) and isinstance(cit.get("id"), str)
    }
    applied: list[str] = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        cid = entry.get("id")
        quote = entry.get("quote")
        if (cid not in unresolved_ids or cid not in by_id
                or not isinstance(quote, str) or not quote):
            continue
        if by_id[cid].get("quote") == quote:
            continue  # identical re-copy cannot resolve anything new
        by_id[cid]["quote"] = quote
        applied.append(cid)
    return applied


# ---------------------------------------------------------------------------
# Deterministic candidate materialization (source binding + re-anchoring)
# ---------------------------------------------------------------------------


def _find_all(haystack: bytes, needle: bytes) -> list[int]:
    positions: list[int] = []
    start = 0
    while True:
        idx = haystack.find(needle, start)
        if idx < 0:
            return positions
        positions.append(idx)
        start = idx + 1


def _render_reanchor_citation(
    source: bytes,
    source_text: str,
    region: Any,
    quote: str,
) -> tuple[int, int] | None:
    """Unique render-equivalent byte span for `quote` inside `region`, or
    None. `find_unique_span` works in the character domain; the region's
    byte span converts through decoded prefixes (region boundaries come
    from the same source bytes, so they sit on valid UTF-8 boundaries)."""
    lo, hi = region.body_span
    try:
        char_lo = len(source[:lo].decode("utf-8"))
        char_hi = char_lo + len(source[lo:hi].decode("utf-8"))
    except UnicodeDecodeError:
        return None
    char_span = find_unique_span(quote, source_text,
                                 region=(char_lo, char_hi))
    if char_span is None:
        return None
    byte_start = len(source_text[:char_span[0]].encode("utf-8"))
    byte_end = byte_start + len(
        source_text[char_span[0]:char_span[1]].encode("utf-8"))
    if byte_start < lo or byte_end > hi:
        # The delimiter expansion may step just outside the region body;
        # a quote that cannot sit inside its named region fails honestly.
        return None
    return byte_start, byte_end


def materialize_candidate(source: bytes, payload: Any) -> tuple[Any, dict]:
    """Fill the deterministic bindings the producer cannot compute: the
    exact source digest/path, and each citation's byte span located by its
    verbatim quote inside its named region body.

    Exact-bytes resolution first. Multiple exact matches prefer the one
    enclosing a bound normative occurrence's token span when that
    disambiguates; otherwise the first match, counted as ambiguous.

    A quote with no exact match gets ONE deterministic fallback before the
    honest failure path: render-equivalent re-anchoring (the queue item 11
    module — Think producers emit render-equivalent but byte-different
    quotes and demonstrably cannot transcribe the difference on retry).
    When the folded quote occurs exactly once inside its named region, the
    quote is REPLACED by the source's own bytes and the span filled — the
    quote is a locator, the source is the authority, and the validator
    remains the terminal gate. Ambiguity under the fold still fails
    honestly: normalized matching never chooses among occurrences (the
    contract's rule). A quote that neither form resolves keeps the
    placeholder and is counted unresolved — meaning is never edited here.

    Returns (payload, stats). Tolerates any malformed shape by leaving it
    untouched (schema validation reports it downstream)."""
    stats: dict[str, Any] = {
        "source_binding_filled": False,
        "citations_total": 0,
        "citations_reanchored": 0,
        "citations_render_reanchored": [],
        "citations_unresolved": [],
        "citations_ambiguous": [],
    }
    if not isinstance(payload, dict):
        return payload, stats

    # Character-domain view for the render-equivalence fold (byte spans
    # stay the artifact contract; conversion happens per resolved quote).
    try:
        source_text: str | None = source.decode("utf-8")
    except UnicodeDecodeError:
        source_text = None  # fold fallback unavailable; exact-bytes only

    source_sha = hashlib.sha256(source).hexdigest()
    src = payload.get("source")
    if not isinstance(src, dict):
        src = {}
        payload["source"] = src
    src["path"] = PAPER_RELPATH
    src["sha256"] = source_sha
    stats["source_binding_filled"] = True

    regions = {r.region_id: r for r in build_region_inventory(source)}

    # Citation id -> normative occurrences bound by records citing it, for
    # deterministic disambiguation of repeated exact quotes.
    occurrence_by_id = {
        o.occurrence_id: o for o in scan_normative_occurrences(source)}
    cited_occurrences: dict[str, list] = {}
    insights = payload.get("insights")
    if isinstance(insights, list):
        for rec in insights:
            if not isinstance(rec, dict):
                continue
            norm = rec.get("normative")
            if not isinstance(norm, dict):
                continue
            occ = occurrence_by_id.get(norm.get("occurrence_id"))
            if occ is None:
                continue
            citation_ids = rec.get("citation_ids")
            if isinstance(citation_ids, list):
                for cid in citation_ids:
                    if isinstance(cid, str):
                        cited_occurrences.setdefault(cid, []).append(occ)

    citations = payload.get("citations")
    if not isinstance(citations, list):
        return payload, stats
    for cit in citations:
        if not isinstance(cit, dict):
            continue
        stats["citations_total"] += 1
        cit_id = cit.get("id") if isinstance(cit.get("id"), str) else "?"
        quote = cit.get("quote")
        region = regions.get(cit.get("region_id"))
        if not isinstance(quote, str) or not quote or region is None:
            stats["citations_unresolved"].append(cit_id)
            continue
        needle = quote.encode("utf-8")
        lo, hi = region.body_span
        matches = [lo + m for m in _find_all(source[lo:hi], needle)]
        # A match must fit entirely inside the region body.
        matches = [m for m in matches if m + len(needle) <= hi]
        if not matches:
            # Render-equivalent fallback: unique folded match inside the
            # named region replaces the quote with the source's own bytes.
            resolved = (
                _render_reanchor_citation(source, source_text, region, quote)
                if source_text is not None else None
            )
            if resolved is None:
                stats["citations_unresolved"].append(cit_id)
                continue
            byte_start, byte_end = resolved
            cit["quote"] = source[byte_start:byte_end].decode("utf-8")
            cit["span"] = {"start": byte_start, "end": byte_end}
            stats["citations_render_reanchored"].append(cit_id)
            stats["citations_reanchored"] += 1
            continue
        chosen: int | None = None
        if len(matches) > 1:
            for occ in cited_occurrences.get(cit_id, []):
                enclosing = [
                    m for m in matches
                    if m <= occ.token_span[0]
                    and occ.token_span[1] <= m + len(needle)
                ]
                if len(enclosing) == 1:
                    chosen = enclosing[0]
                    break
            if chosen is None:
                chosen = matches[0]
                stats["citations_ambiguous"].append(cit_id)
        else:
            chosen = matches[0]
        cit["span"] = {"start": chosen, "end": chosen + len(needle)}
        stats["citations_reanchored"] += 1
    return payload, stats


# ---------------------------------------------------------------------------
# Live dispatch (fresh session per dispatch; injected away in fixtures)
# ---------------------------------------------------------------------------


def _make_live_dispatch(
    *,
    agent: str,
    title: str,
    server_url: str | None,
    port: int,
    timeout_s: float,
    directory: str,
) -> DispatchFn:
    """One fresh opencode session per dispatch (the per-dispatch-session
    transport) — no shared conversation state, same as the calibration
    harness."""
    import opencode_client  # noqa: PLC0415 - keep the module import-light

    if server_url:
        opencode_client.set_server_url(server_url)

    def dispatch(prompt: str) -> None:
        session_id = opencode_client.create_session(
            title=title, directory=directory, port=port)
        opencode_client.dispatch_and_wait(
            session_id=session_id, agent=agent, prompt=prompt,
            port=port, timeout_s=opencode_client.scaled_timeout_s(timeout_s))

    return dispatch


def _is_dispatch_timeout(exc: Exception) -> bool:
    """DispatchTimeout without importing opencode_client at module load
    (fixtures may run with no client available)."""
    for klass in type(exc).__mro__:
        if klass.__name__ == "DispatchTimeout":
            return True
    return False


# ---------------------------------------------------------------------------
# The hook
# ---------------------------------------------------------------------------


def run_insight_shadow(
    run_dir: Path,
    *,
    run_id: str,
    server_url: str | None = None,
    port: int = 4096,
    timeout_s: float = DEFAULT_TIMEOUT_S,
    producer_dispatch: DispatchFn | None = None,
    reviewer_dispatch: DispatchFn | None = None,
) -> dict:
    """Generate, validate, render, and review one shadow insight candidate
    for a run. Returns an outcome report; NEVER raises (every internal
    failure lands in the report and the namespaced event instead).

    outcome is one of: passed | partial | invalid | absent | timeout |
    exception — the contract's six fixture classes. The semantic review's
    sub-outcome rides in the report's `review` field and the event details.
    """
    started = time.monotonic()
    run_dir = Path(run_dir)
    report: dict[str, Any] = {
        "outcome": "exception",
        "phase": "setup",
        "reason": None,
        "deterministic_status": None,
        "review": {"status": "not_dispatched", "reason": None,
                   "overall_verdict": None},
        "correction": {"status": "not_needed", "reason": None,
                       "unresolved_before": [], "unresolved_after": [],
                       "applied": []},
        "materialization": None,
        "stale_removed": [],
        "event_appended": False,
        "event_error": None,
    }
    try:
        _run_insight_shadow_inner(
            run_dir, report,
            run_id=run_id, server_url=server_url, port=port,
            timeout_s=timeout_s,
            producer_dispatch=producer_dispatch,
            reviewer_dispatch=reviewer_dispatch,
        )
    except Exception as exc:  # noqa: BLE001 - non-blocking in every outcome
        report["outcome"] = "exception"
        report["reason"] = f"{type(exc).__name__}: {exc}"
    report["elapsed_s"] = round(time.monotonic() - started, 3)
    _append_shadow_event(run_dir, run_id, report)
    return report


def _run_insight_shadow_inner(
    run_dir: Path,
    report: dict,
    *,
    run_id: str,
    server_url: str | None,
    port: int,
    timeout_s: float,
    producer_dispatch: DispatchFn | None,
    reviewer_dispatch: DispatchFn | None,
) -> None:
    paper_path = run_dir / PAPER_RELPATH
    candidate_path = run_dir / CANDIDATE_RELPATH
    validation_path = run_dir / VALIDATION_RELPATH
    render_path = run_dir / RENDER_RELPATH
    record_path = run_dir / RECORD_RELPATH

    # Invocation safety: remove every shadow artifact from any earlier
    # invocation first, so a stale result can never read as current.
    for relpath in SHADOW_RELPATHS:
        path = run_dir / relpath
        if path.exists():
            path.unlink()
            report["stale_removed"].append(relpath)

    if not paper_path.is_file():
        report["outcome"] = "absent"
        report["phase"] = "source"
        report["reason"] = f"no parsed source at {PAPER_RELPATH}"
        return
    source = paper_path.read_bytes()

    regions = build_region_inventory(source)
    risk_units = build_risk_units(source, regions)
    occurrences = scan_normative_occurrences(source)

    def _slice(span: tuple[int, int]) -> str:
        return source[span[0]:span[1]].decode("utf-8", errors="replace")

    prompt = build_insight_producer_prompt(
        paper_path=str(paper_path),
        candidate_path=str(candidate_path),
        regions=regions,
        risk_units=[(u, _slice(u.span)) for u in risk_units],
        occurrences=[(o, _slice(o.clause_span)) for o in occurrences],
    )

    if producer_dispatch is None:
        producer_dispatch = _make_live_dispatch(
            agent=PRODUCER_AGENT,
            title=f"r2c insight-shadow producer: {run_dir.name}",
            server_url=server_url, port=port, timeout_s=timeout_s,
            directory=str(_REPO_ROOT))
    report["phase"] = "producer_dispatch"
    try:
        producer_dispatch(prompt)
    except Exception as exc:  # noqa: BLE001
        report["outcome"] = (
            "timeout" if _is_dispatch_timeout(exc) else "exception")
        report["reason"] = f"producer dispatch: {type(exc).__name__}: {exc}"
        return

    if not candidate_path.is_file():
        report["outcome"] = "absent"
        report["phase"] = "producer_output"
        report["reason"] = "producer dispatch completed without a candidate"
        return

    # Materialize: fill source binding, re-anchor quotes to exact spans.
    report["phase"] = "materialization"
    try:
        payload = json.loads(candidate_path.read_bytes())
    except json.JSONDecodeError:
        payload = None  # leave the raw bytes for the validator to report
    if payload is not None:
        payload, stats = materialize_candidate(source, payload)
        report["materialization"] = stats
        atomic_write_text(
            candidate_path,
            json.dumps(payload, indent=2, sort_keys=True,
                       ensure_ascii=False) + "\n")

    report["phase"] = "validation"
    result = validate_candidate(
        paper_path, candidate_path, expected_source_path=PAPER_RELPATH)
    report["deterministic_status"] = result.deterministic_status
    atomic_write_text(
        validation_path,
        json.dumps(validation_sidecar_payload(result), indent=2,
                   sort_keys=True) + "\n")
    report["outcome"] = result.deterministic_status

    # Targeted corrective nudge (item 12): quotes that survived neither
    # exact-bytes nor render-equivalent re-anchoring get ONE producer
    # re-ask for exactly those quotes; the deterministic merge applies
    # quote fields only, then re-materializes and re-validates. Every
    # failure mode lands in the report — the outcome then simply stays
    # whatever the first validation said.
    unresolved = list(
        (report.get("materialization") or {}).get("citations_unresolved")
        or [])
    correction = report["correction"]
    if (payload is not None and unresolved
            and result.deterministic_status != "passed"):
        correction["unresolved_before"] = list(unresolved)
        correction["unresolved_after"] = list(unresolved)
        details = _unresolved_citation_details(payload, regions, unresolved)
        corrections_path = run_dir / CORRECTIONS_RELPATH
        if not details:
            correction["status"] = "skipped"
            correction["reason"] = (
                "no unresolved citation carries a re-askable quote")
        else:
            report["phase"] = "correction_dispatch"
            correction["status"] = "dispatched"
            correction_prompt = build_insight_correction_prompt(
                paper_path=str(paper_path),
                corrections_path=str(corrections_path),
                unresolved=details,
            )
            try:
                producer_dispatch(correction_prompt)
            except Exception as exc:  # noqa: BLE001
                correction["status"] = (
                    "timeout" if _is_dispatch_timeout(exc) else "error")
                correction["reason"] = (
                    f"correction dispatch: {type(exc).__name__}: {exc}")
            else:
                corrections_payload = None
                if corrections_path.is_file():
                    try:
                        corrections_payload = json.loads(
                            corrections_path.read_bytes())
                    except json.JSONDecodeError:
                        corrections_payload = None
                if corrections_payload is None:
                    correction["status"] = "absent"
                    correction["reason"] = (
                        "correction dispatch completed without a parseable "
                        "corrections file")
                else:
                    applied = apply_quote_corrections(
                        payload, corrections_payload, unresolved)
                    correction["applied"] = applied
                    if not applied:
                        correction["status"] = "no_effect"
                        correction["reason"] = (
                            "no correction matched an unresolved citation")
                    else:
                        report["phase"] = "correction_materialization"
                        payload, stats = materialize_candidate(
                            source, payload)
                        report["materialization"] = stats
                        atomic_write_text(
                            candidate_path,
                            json.dumps(payload, indent=2, sort_keys=True,
                                       ensure_ascii=False) + "\n")
                        report["phase"] = "correction_validation"
                        result = validate_candidate(
                            paper_path, candidate_path,
                            expected_source_path=PAPER_RELPATH)
                        report["deterministic_status"] = (
                            result.deterministic_status)
                        atomic_write_text(
                            validation_path,
                            json.dumps(validation_sidecar_payload(result),
                                       indent=2, sort_keys=True) + "\n")
                        report["outcome"] = result.deterministic_status
                        correction["status"] = "applied"
                        correction["unresolved_after"] = list(
                            stats.get("citations_unresolved") or [])

    if not result.ok or result.model is None:
        return  # invalid: never rendered, never reviewed

    atomic_write_text(render_path, render_insights_md(result))

    # Semantic review: sliced bundles, one fresh-session dispatch, verdict
    # loaded back under full hash binding. Sub-outcome only — the overall
    # outcome stays the deterministic status.
    report["phase"] = "review_dispatch"
    bundle = build_review_bundles(source, result.model)
    review_prompt = build_insight_review_prompt(
        bundle_json=json.dumps(bundle, indent=1, sort_keys=True),
        record_path=str(record_path),
        source_sha256=result.source_sha256,
        candidate_sha256=result.candidate_sha256,
    )
    if reviewer_dispatch is None:
        reviewer_dispatch = _make_live_dispatch(
            agent=REVIEWER_AGENT,
            title=f"r2c insight-shadow review: {run_dir.name}",
            server_url=server_url, port=port, timeout_s=timeout_s,
            directory=str(_REPO_ROOT))
    try:
        reviewer_dispatch(review_prompt)
    except Exception as exc:  # noqa: BLE001
        report["review"] = {
            "status": "timeout" if _is_dispatch_timeout(exc) else "error",
            "reason": f"{type(exc).__name__}: {exc}",
            "overall_verdict": None,
        }
        report["phase"] = "done"
        return

    resolution = load_acceptance_record(
        record_path,
        source_sha256=result.source_sha256,
        candidate_sha256=result.candidate_sha256,
        expected_record_ids={rec.id for rec in result.model.insights},
    )
    report["review"] = {
        "status": "present" if resolution.present else "absent",
        "reason": resolution.reason,
        "overall_verdict":
            resolution.record.overall_verdict if resolution.record else None,
    }
    report["phase"] = "done"


def _append_shadow_event(run_dir: Path, run_id: str, report: dict) -> None:
    """One append-only namespaced event, best-effort: an event failure is
    recorded in the report and swallowed — it can never fail the run."""
    try:
        from run_events import append_event  # noqa: PLC0415

        artifacts = [
            relpath for relpath in SHADOW_RELPATHS
            if (Path(run_dir) / relpath).is_file()
        ]
        details = {
            "outcome": report["outcome"],
            "phase": report["phase"],
            "reason": report["reason"],
            "deterministic_status": report["deterministic_status"],
            "review": report["review"],
            "correction": report.get("correction"),
            "materialization": report["materialization"],
            "stale_removed": report["stale_removed"],
            "elapsed_s": report.get("elapsed_s"),
        }
        append_event(
            Path(run_dir) / ".pipeline",
            event_type=EVENT_TYPE,
            run_id=run_id,
            stage_id=EVENT_STAGE_ID,
            stage_label="Insight shadow (generic understanding tier)",
            status=report["outcome"],
            summary=(
                f"insight shadow outcome={report['outcome']} "
                f"review={report['review']['status']}"),
            artifacts=artifacts,
            details=details,
        )
        report["event_appended"] = True
    except Exception as exc:  # noqa: BLE001 - best-effort in every outcome
        report["event_error"] = f"{type(exc).__name__}: {exc}"


def maybe_run_insight_shadow(
    run_dir: Path,
    *,
    run_id: str,
    server_url: str | None = None,
    port: int = 4096,
) -> dict | None:
    """The driver's single entry point. Returns None (and does nothing)
    unless R2C_INSIGHT_SHADOW=1; otherwise runs the shadow flow and returns
    its report. Never raises."""
    try:
        if not insight_shadow_enabled():
            return None
        return run_insight_shadow(
            run_dir, run_id=run_id, server_url=server_url, port=port)
    except Exception as exc:  # noqa: BLE001 - the outer belt
        return {"outcome": "exception",
                "reason": f"{type(exc).__name__}: {exc}"}
