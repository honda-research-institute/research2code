"""B-01 items 1-3: the stage-review contract and the schema-link check.

Three surfaces under test:

1. `schemas/stage_review_report.py` — the read-path contract for
   stage_review_<stage_id>.json, shaped from the full recorded artifact
   history. Known-good corpus: every tracked historical review validates.
   Known-bad: a review with no findings key fails.
2. The wiring in `_read_review_json_or_err` — stage-reviewer reads
   (expected_stage_id set) validate against the model; the stage-5
   review_report.json reads (no expected_stage_id) are untouched.
3. The repo-wide schema-link check — every `schemas/<name>.py` string in
   scripts/ and .opencode/agents/*.md resolves to a real file, so a prompt
   can never again cite a module that does not exist (the
   stage_review_report.py phantom this block closes, and the halt-judge's
   dead method_package pointer before it).
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
from pydantic import ValidationError

from run_pipeline import _read_review_json_or_err
from schemas.stage_review_report import StageReviewReport

REPO_ROOT = Path(__file__).resolve().parent.parent

TRACKED_CORPUS_ROOTS = (
    REPO_ROOT / "tests" / "fixtures" / "evidence",
)


def _stage_reviews_under(root: Path) -> list[Path]:
    # os.walk-style manual walk: glob's ** does not enter hidden dirs like
    # .pipeline, where these artifacts live.
    found = []
    for p in sorted(root.rglob("*.json")):
        if p.name.startswith("stage_review_"):
            found.append(p)
    return found


# ---------------------------------------------------------------------------
# 1. The model, against history
# ---------------------------------------------------------------------------

def test_known_good_corpus_all_tracked_reviews_validate():
    files = [f for root in TRACKED_CORPUS_ROOTS for f in _stage_reviews_under(root)]
    assert len(files) >= 5, f"tracked corpus unexpectedly small: {files}"
    for f in files:
        data = json.loads(f.read_text(encoding="utf-8"))
        StageReviewReport.model_validate(data)  # raises on the first regression


@pytest.mark.manual_only
def test_known_good_corpus_live_fleet_validates():
    files = _stage_reviews_under(REPO_ROOT / "r2c_runs")
    if not files:
        pytest.skip("live fleet absent")
    for f in files:
        data = json.loads(f.read_text(encoding="utf-8"))
        StageReviewReport.model_validate(data)


VALID_REVIEW = {
    "schema_version": "1.0.0",
    "stage_id": "stage_2c_method",
    "review_status": "passed",
    "summary": "All checks pass.",
    "findings": [],
    "checks_summary": [{"check_id": "c1", "status": "pass", "note": "ok"}],
    "caveats": [],
}


def test_known_bad_missing_findings_fails():
    bad = {k: v for k, v in VALID_REVIEW.items() if k != "findings"}
    with pytest.raises(ValidationError):
        StageReviewReport.model_validate(bad)


def test_permissive_where_history_is_permissive():
    review = dict(VALID_REVIEW)
    review["review_status"] = "issues_found"
    review["findings"] = [{
        # "warning" severity, off-vocabulary issue_type, no
        # resolution_status, plus an extra key: all recorded shapes.
        "id": "F001", "severity": "warning", "target_agent": "method-coder",
        "issue_type": "paper_fidelity", "file": "method/method.py",
        "location": "lines 1-2", "description": "x", "proposed_fix": None,
        "check_id": "c1", "extra_key_from_the_future": True,
    }]
    review["caveats"] = ["a string", {"surface": "assumptions", "note": "a dict"}]
    StageReviewReport.model_validate(review)


# ---------------------------------------------------------------------------
# 2. The read-path wiring
# ---------------------------------------------------------------------------

def _write(tmp_path: Path, data) -> Path:
    p = tmp_path / "stage_review_stage_2c_method.json"
    p.write_text(json.dumps(data), encoding="utf-8")
    return p


def test_read_path_accepts_valid_stage_review(tmp_path):
    p = _write(tmp_path, VALID_REVIEW)
    review, err = _read_review_json_or_err(
        p, kind="stage_2c_method review", expected_stage_id="stage_2c_method")
    assert err is None
    assert review == VALID_REVIEW


def test_read_path_rejects_contract_violation(tmp_path):
    bad = dict(VALID_REVIEW)
    bad["review_status"] = "done"  # not in the passed|issues_found contract
    p = _write(tmp_path, bad)
    review, err = _read_review_json_or_err(
        p, kind="stage_2c_method review", expected_stage_id="stage_2c_method")
    assert review is None
    assert err is not None and "schemas/stage_review_report.py" in err


def test_read_path_without_stage_id_is_not_stage_review_gated(tmp_path):
    # The stage-5 review_report.json reads pass no expected_stage_id and
    # must keep today's looser findings-key contract.
    p = tmp_path / "review_report.json"
    p.write_text(json.dumps({"findings": []}), encoding="utf-8")
    review, err = _read_review_json_or_err(p, kind="stage_5 routing review")
    assert err is None
    assert review == {"findings": []}


# ---------------------------------------------------------------------------
# 3. The repo-wide schema-link check
# ---------------------------------------------------------------------------

SCHEMA_REF_RE = re.compile(r"schemas/([A-Za-z0-9_]+\.py)")


def test_every_schema_module_reference_resolves():
    sources = list((REPO_ROOT / "scripts").rglob("*.py"))
    sources += sorted((REPO_ROOT / ".opencode" / "agents").glob("*.md"))
    assert sources, "link-check found no source files to scan"
    missing = []
    referenced = set()
    for src in sources:
        try:
            text = src.read_text(encoding="utf-8")
        except OSError:
            continue
        for name in SCHEMA_REF_RE.findall(text):
            referenced.add(name)
            if not (REPO_ROOT / "schemas" / name).is_file():
                missing.append(f"{src.relative_to(REPO_ROOT)} -> schemas/{name}")
    assert not missing, "dangling schema references:\n" + "\n".join(missing)
    # The check must actually be exercising real references, including the
    # two this block repaired.
    assert "stage_review_report.py" in referenced
    assert "review_report.py" in referenced
