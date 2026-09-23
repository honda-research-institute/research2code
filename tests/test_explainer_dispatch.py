"""Explainer dispatch builder + part merge (slice 2.1 redesign).

The contract that came out of the failed first dispatch: ground truth is
INLINED in every prompt (no read-the-paper honor system), work is
chunked so one dispatch is minutes not an hour, and parts merge into the
one sidecar the validator judges for coverage.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from explainer_dispatch import (augment_prompt_with_rejections,
                                build_dispatches, chunk_elements,
                                merge_sidecar_parts, reconcile_landed_parts,
                                _paper_front)
from validate_method_explanations import check_explanations


def _eq(i: int, role: str = "implement") -> dict:
    return {"id": f"eq-{i}", "type": "equation", "code_role": role,
            "name": f"Equation {i}", "section": "Section 3",
            "source_text": f"the paper's verbatim statement number {i}",
            "description": f"decomposition note {i}",
            "pseudocode": f"step_{i}()"}


PAPER_MAP = {"title": "Test Paper",
             "elements": [_eq(i) for i in range(1, 10)]}
PAPER_TEXT = "Title and abstract.\n\nWe claim novelty in batching.\n\n" + \
    "Body paragraph. " * 400


def _dispatches(chunk_size=4):
    return build_dispatches(PAPER_MAP, PAPER_TEXT, paper_path="/x/paper.md",
                            output_dir="/x/.pipeline",
                            chunk_size=chunk_size)


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------


def test_chunks_cover_everything_in_order_without_overlap():
    chunks = chunk_elements(PAPER_MAP["elements"], 4)
    assert [len(c) for c in chunks] == [4, 4, 1]
    flat = [e["id"] for c in chunks for e in c]
    assert flat == [e["id"] for e in PAPER_MAP["elements"]]


def test_chunk_size_must_be_positive():
    with pytest.raises(ValueError):
        chunk_elements(PAPER_MAP["elements"], 0)


# ---------------------------------------------------------------------------
# Dispatch prompts: the inlining contract
# ---------------------------------------------------------------------------


def test_every_prompt_inlines_its_elements_ground_truth():
    for d in _dispatches():
        for eid in d["element_ids"]:
            i = eid.split("-")[1]
            assert f"the paper's verbatim statement number {i}" in d["prompt"]
            assert f"decomposition note {i}" in d["prompt"]
            assert f"step_{i}()" in d["prompt"]


def test_prompt_lists_exactly_its_chunk_ids_and_part_path():
    dispatches = _dispatches()
    assert [d["part"] for d in dispatches] == [1, 2, 3]
    for d in dispatches:
        assert d["output_path"] == \
            f"/x/.pipeline/method_explanations.part-{d['part']}.json"
        assert d["output_path"] in d["prompt"]
        for eid in d["element_ids"]:
            assert f"- {eid}" in d["prompt"]
    # An id from another chunk must not appear in this chunk's id list.
    assert "- eq-5" not in dispatches[0]["prompt"]


def test_front_matter_inlined_once_per_dispatch():
    for d in _dispatches():
        assert "We claim novelty in batching." in d["prompt"]
        assert "Test Paper" in d["prompt"]


def test_dispatches_are_deterministic():
    a = [d["prompt"] for d in _dispatches()]
    b = [d["prompt"] for d in _dispatches()]
    assert a == b


def test_front_matter_cuts_at_paragraph_boundary():
    front = _paper_front(PAPER_TEXT, max_chars=60)
    assert front == "Title and abstract.\n\nWe claim novelty in batching."
    # Short papers pass through whole.
    assert _paper_front("tiny paper") == "tiny paper"


# ---------------------------------------------------------------------------
# Part merge
# ---------------------------------------------------------------------------


def _write_part(path, ids):
    entry = {"what": "Computes the verbatim statement's quantity for each "
                     "candidate point in the pool.",
             "why_novel": "Standard machinery per the paper's own "
                          "positioning, included for completeness.",
             "intuition": "Bigger values mean the model wants the label "
                          "more, so ranking by it surfaces useful points."}
    path.write_text(json.dumps(
        {"schema_version": "1.0.0",
         "explanations": {i: dict(entry) for i in ids}}))
    return path


def test_merge_then_validate_round_trip(tmp_path):
    dispatches = _dispatches()
    parts = [_write_part(tmp_path / f"part-{d['part']}.json",
                         d["element_ids"]) for d in dispatches]
    merged, skipped = merge_sidecar_parts(parts)
    assert skipped == []
    assert merged["schema_version"] == "1.0.0"
    assert len(merged["explanations"]) == 9
    # The merged sidecar is what the validator judges — full coverage,
    # no fabricated ids, substantive fields.
    assert check_explanations(merged, PAPER_MAP, PAPER_TEXT) == []


def test_lost_part_fails_merged_coverage(tmp_path):
    dispatches = _dispatches()
    parts = [_write_part(tmp_path / f"part-{d['part']}.json",
                         d["element_ids"]) for d in dispatches[:-1]]
    merged, _ = merge_sidecar_parts(parts)
    findings = check_explanations(merged, PAPER_MAP, PAPER_TEXT)
    assert any(f["check"] == "coverage" and "eq-9" in f["message"]
               for f in findings)


def test_overlapping_parts_are_an_error(tmp_path):
    a = _write_part(tmp_path / "a.json", ["eq-1", "eq-2"])
    b = _write_part(tmp_path / "b.json", ["eq-2", "eq-3"])
    with pytest.raises(ValueError, match="eq-2"):
        merge_sidecar_parts([a, b])


def test_retry_prompt_carries_rejection_notes():
    # An informed retry, not a blind re-roll: the counterexample text from a
    # refuted claim (or any validator finding message) is appended to the
    # retry prompt so the explainer can correct the specific defect.
    base = _dispatches()[0]["prompt"]
    notes = ["eq-2: a mechanism claim is refuted by a numeric counterexample "
             "— at x=0.9 the value decreases. Fix the explanation."]
    augmented = augment_prompt_with_rejections(base, notes)
    assert augmented.startswith(base)  # inlined material untouched
    assert "previous attempt rejected" in augmented
    assert "at x=0.9 the value decreases" in augmented


def test_no_rejection_notes_leaves_prompt_unchanged():
    base = _dispatches()[0]["prompt"]
    assert augment_prompt_with_rejections(base, []) == base


def test_rejection_notes_are_truncated():
    # The cap is sized so a note can carry its candidate paper passage
    # (item 23 part 3) without truncating the passage mid-quote; a
    # runaway note still cannot flood the prompt.
    base = _dispatches()[0]["prompt"]
    augmented = augment_prompt_with_rejections(base, ["x" * 5000])
    assert len(augmented) < len(base) + 2000


# ---------------------------------------------------------------------------
# Landed-part reconciliation (the mixed-layout resume bug, A8 interlude)
# ---------------------------------------------------------------------------


def _plan(tmp_path, chunk_size):
    return build_dispatches(PAPER_MAP, PAPER_TEXT, paper_path="/x/paper.md",
                            output_dir=str(tmp_path), chunk_size=chunk_size)


def test_reconcile_deletes_parts_from_a_different_chunk_layout(tmp_path):
    # Parts landed under the old four-equation layout, then the chunk size
    # changed to 1: every part number now means a different equation set.
    # Left in place they collide with freshly dispatched parts at merge.
    for d in _plan(tmp_path, chunk_size=4):
        _write_part(Path(d["output_path"]), d["element_ids"])
    plan = _plan(tmp_path, chunk_size=1)
    deleted = reconcile_landed_parts(plan)
    assert deleted  # the stale four-equation parts went away
    survivors = sorted(tmp_path.glob("method_explanations.part-*.json"))
    planned = {d["output_path"]: set(d["element_ids"]) for d in plan}
    for part in survivors:
        keys = set(json.loads(part.read_text())["explanations"])
        assert keys == planned[str(part)]


def test_reconcile_keeps_parts_matching_the_current_plan(tmp_path):
    plan = _plan(tmp_path, chunk_size=1)
    for d in plan[:3]:
        _write_part(Path(d["output_path"]), d["element_ids"])
    assert reconcile_landed_parts(plan) == []
    assert len(list(tmp_path.glob("method_explanations.part-*.json"))) == 3


def test_reconcile_deletes_stray_parts_beyond_the_plan(tmp_path):
    # The reverse relayout (chunk 1 -> chunk 4): old high-numbered parts
    # have no planned counterpart at all.
    for d in _plan(tmp_path, chunk_size=1):
        _write_part(Path(d["output_path"]), d["element_ids"])
    plan = _plan(tmp_path, chunk_size=4)
    reconcile_landed_parts(plan)
    survivors = {str(p) for p in tmp_path.glob("method_explanations.part-*.json")}
    assert survivors <= {d["output_path"] for d in plan}


def test_reconcile_leaves_unparseable_parts_for_the_merge_machinery(tmp_path):
    plan = _plan(tmp_path, chunk_size=1)
    bad = Path(plan[0]["output_path"])
    bad.write_text("{not json")
    assert reconcile_landed_parts(plan) == []
    assert bad.is_file()  # merge's skip/requeue path owns this case


def test_reconcile_with_empty_plan_is_a_noop(tmp_path):
    assert reconcile_landed_parts([]) == []


def test_missing_and_unparseable_parts_are_skipped_not_fatal(tmp_path):
    # E1 (audit 2026-06-25): one malformed part must NOT wipe the others. A
    # missing or unparseable part (e.g. an unescaped inner quote breaking JSON)
    # is skipped and reported, and the good parts still merge — so METHOD.md
    # keeps the explanations that did parse instead of rendering all pending.
    good = _write_part(tmp_path / "good.json", ["eq-1", "eq-2"])
    bad = tmp_path / "bad.json"
    bad.write_text('{"explanations": {"eq-3": {"what": "has an "inner" quote"}}}')
    missing = tmp_path / "nope.json"
    merged, skipped = merge_sidecar_parts([good, bad, missing])
    assert sorted(merged["explanations"]) == ["eq-1", "eq-2"]
    assert set(skipped) == {bad, missing}
