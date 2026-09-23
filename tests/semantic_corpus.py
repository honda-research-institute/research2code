"""Fixture-only helper for the Block 7 golden semantic-review corpus.

Verifies the exact recorded expectations of each corpus case under
`tests/fixtures/generic_insights/semantic_review/`: hash bindings to the
committed source and to the candidate bytes, citation-quote byte
equality, verdict validity, required facts, prohibited interpretations,
required ambiguity disclosures, and reviewer/sign-off provenance.

This module is test infrastructure only. It is never imported by
production code, and it does not claim to classify arbitrary
paraphrases — it proves the reviewed corpus is intact, nothing more.
Semantic verdicts belong to external review; production validation
always reports `semantic_status: "unreviewed"`.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
CORPUS_DIR = REPO_ROOT / "tests/fixtures/generic_insights/semantic_review"

VALID_VERDICTS = {"accepted", "rejected"}

REQUIRED_EXPECTATION_FIELDS = (
    "case_id",
    "source_sha256",
    "candidate_sha256",
    "verdict",
    "required_facts",
    "prohibited_interpretations",
    "required_disclosures",
    "rationale",
    "provenance",
)

REQUIRED_PROVENANCE_FIELDS = (
    "verdict_basis",
    "signed_off_by",
    "recorded_by",
    "semantic_owner",
)


def load_cases() -> list[tuple[Path, dict, dict]]:
    """All corpus cases as (case_dir, candidate, expectations), sorted."""
    cases = []
    for cdir in sorted(p for p in CORPUS_DIR.iterdir() if p.is_dir()):
        candidate = json.loads(
            (cdir / "candidate.json").read_text(encoding="utf-8"))
        expectations = json.loads(
            (cdir / "expectations.json").read_text(encoding="utf-8"))
        cases.append((cdir, candidate, expectations))
    return cases


def verify_case(cdir: Path, candidate: dict, expectations: dict) -> list[str]:
    """Every recorded expectation, checked exactly. Returns problems."""
    problems: list[str] = []
    cid = expectations.get("case_id")

    for field in REQUIRED_EXPECTATION_FIELDS:
        if field not in expectations:
            problems.append(f"{cid}: expectations missing field {field!r}")
    for field in REQUIRED_PROVENANCE_FIELDS:
        if not (expectations.get("provenance") or {}).get(field):
            problems.append(f"{cid}: provenance missing field {field!r}")

    if candidate.get("case_id") != cid or cdir.name != cid:
        problems.append(
            f"{cid}: case id mismatch (dir {cdir.name!r}, "
            f"candidate {candidate.get('case_id')!r})")

    verdict = expectations.get("verdict")
    if verdict not in VALID_VERDICTS:
        problems.append(f"{cid}: invalid verdict {verdict!r}")

    # Candidate bytes are hash-bound to the expectations record.
    cand_bytes = (cdir / "candidate.json").read_bytes()
    cand_sha = hashlib.sha256(cand_bytes).hexdigest()
    if cand_sha != expectations.get("candidate_sha256"):
        problems.append(f"{cid}: candidate.json hash drifted")

    # Source is hash-bound in both records and must exist in the repo.
    src_rel = (candidate.get("source") or {}).get("path") or ""
    src_path = REPO_ROOT / src_rel
    if not src_path.is_file():
        problems.append(f"{cid}: source {src_rel!r} not found")
        return problems
    src = src_path.read_bytes()
    src_sha = hashlib.sha256(src).hexdigest()
    if src_sha != (candidate.get("source") or {}).get("sha256"):
        problems.append(f"{cid}: candidate source sha mismatch")
    if src_sha != expectations.get("source_sha256"):
        problems.append(f"{cid}: expectations source sha mismatch")

    # Every citation quote must equal the exact source bytes at its span.
    for c in candidate.get("citations") or []:
        span = c.get("span") or {}
        start, end = span.get("start"), span.get("end")
        sliced = src[start:end].decode("utf-8", errors="replace")
        if sliced != c.get("quote"):
            problems.append(
                f"{cid}: citation {c.get('id')} quote does not match "
                f"source bytes [{start}, {end})")

    statement = candidate.get("statement") or ""
    if not statement.strip():
        problems.append(f"{cid}: empty statement")

    if verdict == "accepted":
        for fact in expectations.get("required_facts") or []:
            if fact not in statement:
                problems.append(f"{cid}: required fact missing: {fact!r}")
        for phrase in expectations.get("prohibited_interpretations") or []:
            if phrase in statement:
                problems.append(
                    f"{cid}: prohibited interpretation present: {phrase!r}")
        for disclosure in expectations.get("required_disclosures") or []:
            if disclosure not in statement:
                problems.append(
                    f"{cid}: required disclosure missing: {disclosure!r}")
    elif verdict == "rejected":
        # A rejected case records the corrupt interpretation itself; the
        # statement must actually carry the corruption it documents.
        recorded = [
            phrase for phrase in
            (expectations.get("prohibited_interpretations") or [])
            if phrase in statement
        ]
        if not recorded:
            problems.append(
                f"{cid}: rejected statement carries none of its recorded "
                f"prohibited interpretations")
        if not (expectations.get("rationale") or "").strip():
            problems.append(f"{cid}: rejected case has no rationale")

    return problems
