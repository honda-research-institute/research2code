"""Phase 4a — Stage 1 (decomposer + analyzer + feasibility + reviewer).

Tests run_stage_1's two-phase structure:
  1.a — decomposer → paper_map → validate_paper_map → fix-loop
  1.b — analyzer → method_spec → validate_method_spec → feasibility gate
        → stage-reviewer → critical-findings fix-loop

Both phases are stitched together in a single `run_stage_1` function, so
these tests treat them as one stage but distinguish leaves by step name.

Most failures route through the halt-judge via `_validator_retry(use_judge=True)`
or `_invoke_judge` directly. Decomposer/analyzer halt sidecars (paper_map.json.halt
/ method_spec.json.halt) are agent-written failure signals — the fake
dispatch writes them when simulating "paper is non-decomposable / paradigm
mismatch" cases.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace

from tests.helpers.assertions import (
    assert_dispatched,
    assert_not_dispatched,
    assert_stage_completed,
    assert_stage_degraded,
    assert_stage_halted,
)
from tests.helpers.inject import (
    minimal_method_spec,
    minimal_paper_map,
    queue_judge_decision,
)
from tests.helpers.state import make_state
import pytest

pytestmark = pytest.mark.probe_runtime


def _paper_map_json() -> str:
    return json.dumps(minimal_paper_map(), indent=2)


def _method_spec_json(**kw) -> str:
    return json.dumps(minimal_method_spec(**kw), indent=2)


def _paper_map_part_writes() -> dict[str, str]:
    paper_map = minimal_paper_map()
    elements = list(paper_map["elements"])
    element_files = [f"elements/{index:03d}-{element['id']}.json"
                     for index, element in enumerate(elements, start=1)]
    writes = {
        ".pipeline/paper_map_parts/manifest.json": json.dumps({
            "schema_version": paper_map.get("schema_version", "1.0.0"),
            "title": paper_map.get("title", "Test Paper"),
            "element_files": element_files,
        }, indent=2),
    }
    for rel_path, element in zip(element_files, elements, strict=True):
        writes[f".pipeline/paper_map_parts/{rel_path}"] = json.dumps(element, indent=2)
    return writes


def _method_spec_part_writes() -> dict[str, str]:
    spec = minimal_method_spec()
    # Write ALL required MethodSpec sections — the assembler now enforces a
    # completeness contract (refuses to assemble a spec missing a required
    # field), so a partial part set no longer assembles. minimal_method_spec()
    # carries only a subset, so stub the remaining required sections; the
    # validator is faked in these tests and the assembler checks key PRESENCE,
    # not schema validity. (DRV-C1)
    sections = {
        "paper": spec["paper"],
        "core_method": spec["core_method"],
        "comparison": spec["comparison"],
        "critical_requirements": spec["critical_requirements"],
        "paper_claims": spec.get("paper_claims", []),
        "try_it_out": spec.get("try_it_out", {}),
        "data_requirements": spec.get("data_requirements", {}),
        "dependencies": spec.get("dependencies", {}),
        "repo": spec.get("repo", {}),
        "feasibility": "reproducible",
    }
    part_files = [f"{key}.json" for key in sections]
    writes = {
        ".pipeline/method_spec_parts/manifest.json": json.dumps({
            "schema_version": spec.get("schema_version", "1.0.0"),
            "part_files": part_files,
        }, indent=2),
    }
    for key, value in sections.items():
        writes[f".pipeline/method_spec_parts/{key}.json"] = json.dumps(
            {key: value}, indent=2
        )
    return writes


def _malformed_method_spec_part_writes() -> dict[str, str]:
    writes = _method_spec_part_writes()
    writes[".pipeline/method_spec_parts/comparison.json"] = (
        '{"comparison": {"classification": {"id": "x"}}},\n'
        '"pluggable_component": {"name": "select_batch"}\n'
    )
    return writes


def _write_run_files(run_dir: Path, writes: dict[str, str]) -> None:
    for rel_path, content in writes.items():
        target = run_dir / rel_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")


def _supersession_events(run_dir: Path) -> list[dict]:
    event_log = run_dir / ".pipeline" / "run_events.jsonl"
    if not event_log.is_file():
        return []
    events = [
        json.loads(line)
        for line in event_log.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return [
        event for event in events
        if event.get("event_type") == "artifact_halt_sidecar_superseded"
    ]


def _proposal_dir_for_recovery(run_dir: Path) -> Path:
    proposal_dir = run_dir / ".pipeline" / "paradigm_proposals" / "manual-proposal"
    proposal_dir.mkdir(parents=True, exist_ok=True)
    (proposal_dir / "proposal.json").write_text(
        json.dumps(
            {
                "schema_version": "1.0.0",
                "proposal_id": "manual-proposal",
                "source_paper_slug": "test-paper",
                "source_gap_report": ".pipeline/paradigm_gap_report.json",
                "decision": "new_subparadigm_needed",
                "target_paradigm_id": "active_learning/geometric",
                "target_taxonomy_id": "TE-TS/active_learning/geometric",
                "extends": "active_learning",
                "title": "Geometric Active Learning",
                "scope_summary": "Geometric active-learning selection.",
                "pack": {
                    "schema_version": "1.0",
                    "status": "provisional",
                    "legacy_paradigm": "active_learning/geometric",
                    "extends": "active_learning",
                    "taxonomy_id": "TE-TS/active_learning/geometric",
                    "fingerprint": {
                        "what_it_is": "Geometric active-learning selection.",
                    },
                    "scaffold_hints": {
                        "interface_hint": "select_batch(model, x_unlabeled, batch_size, seed)",
                    },
                    "semantic_checks": [],
                    "smoke_bugs": [],
                },
                "evidence": [
                    {
                        "paper_section": "method",
                        "quote_or_observation": "Geometric selector.",
                        "relevance": "Needs a child active-learning pack.",
                    }
                ],
                "coupling_warnings": [],
                "validation_command": "python3 scripts/validate_paradigm_proposal.py <proposal_dir>",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (proposal_dir / "pack.yaml").write_text(
        """schema_version: "1.0"
status: provisional
legacy_paradigm: active_learning/geometric
extends: active_learning
taxonomy_id: TE-TS/active_learning/geometric
fingerprint:
  what_it_is: Geometric active-learning selection.
scaffold_hints:
  interface_hint: select_batch(model, x_unlabeled, batch_size, seed)
semantic_checks: []
smoke_bugs: []
""",
        encoding="utf-8",
    )
    (proposal_dir / "validation_report.json").write_text(
        json.dumps({"valid": True, "errors": [], "warnings": []}, indent=2),
        encoding="utf-8",
    )
    return proposal_dir


def _new_subparadigm_gap_json() -> str:
    return json.dumps(
        {
            "schema_version": "1.0.0",
            "decision": "new_subparadigm_needed",
            "confidence": 0.9,
            "paper_slug": "test-paper",
            "paper_title": "Geometric Active Learning",
            "paper_paradigm_summary": "An active-learning method with a new geometric selector.",
            "matched_existing_paradigm": None,
            "proposed_parent_paradigm": "active_learning",
            "proposed_new_paradigm_id": "active_learning/geometric",
            "candidate_matches": [],
            "rejected_matches": [],
            "registered_paradigms": ["active_learning"],
            "paper_evidence": [
                {
                    "paper_section": "method",
                    "quote_or_observation": "The paper defines a geometric selector.",
                    "relevance": "Parent active_learning is close but incomplete.",
                }
            ],
            "recommended_next_action": "Draft a new sub-paradigm pack.",
        },
        indent=2,
    )


def _clean_stage_review() -> str:
    return json.dumps({
        "schema_version": "1.0.0",
        "stage_id": "stage_1_analyzer",
        "review_status": "passed",
        "summary": "All checks pass.",
        "findings": [],
        "checks_summary": [],
        "caveats": [],
    }, indent=2)


def _queue_happy_stage_1b(fake_dispatch, fake_subprocess) -> None:
    fake_dispatch.expect(
        agent="r2c-method-analyzer",
        writes={".pipeline/method_spec.json": _method_spec_json()},
    )
    _queue_happy_after_method_spec(fake_dispatch, fake_subprocess)


def _queue_happy_after_method_spec(fake_dispatch, fake_subprocess) -> None:
    fake_subprocess.expect_script(returncode=0)  # method-spec validator
    fake_subprocess.expect_script(returncode=0)  # feasibility gate
    fake_dispatch.expect(
        agent="r2c-stage-reviewer",
        writes={
            ".pipeline/stage_review_stage_1_analyzer.json": _clean_stage_review()
        },
    )


def _stage_review_with_critical(finding_id: str = "F001") -> str:
    return json.dumps({
        "schema_version": "1.0.0",
        "stage_id": "stage_1_analyzer",
        "review_status": "issues_found",
        "summary": "Critical finding.",
        "findings": [{
            "id": finding_id,
            "check_id": "test_check",
            "severity": "critical",
            "target_agent": "analyzer",
            "issue_type": "other",
            "file": "method_spec.json",
            "location": "comparison.classification",
            "description": "test finding",
            "proposed_fix": "fix it",
            "proposed_resolution": None,
            "resolution_status": "pending",
        }],
        "checks_summary": [],
        "caveats": [],
    }, indent=2)


def _stage_review_missing_stage_id() -> str:
    return json.dumps({
        "schema_version": "1.2.0",
        "review_status": "passed",
        "summary": "Analyzer outputs are faithful to the paper.",
        "findings": [],
    }, indent=2)


def _judge_decision(
    *, iteration: int = 0, action: str = "dispatch_fix",
    target_agent: str = "r2c-decomposer",
    validator_label: str = "validate_paper_map.py",
) -> dict:
    return {
        "schema_version": "1.0.0",
        "stage_id": "stage_1",
        "iteration": iteration,
        "validator_label": validator_label,
        "classification": "producer_fixable",
        "action": action,
        "target_agent": target_agent if action == "dispatch_fix" else None,
        "finding": {
            "id": "JUDGE001", "severity": "critical",
            "description": "test", "proposed_fix": "fix",
        } if action == "dispatch_fix" else None,
        "rationale": "test rationale",
        "confidence": "high",
        "files_examined": ["scripts/run_pipeline.py"],
    }


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


def test_stage_1_happy_path_completes(fake_dispatch, fake_subprocess, run_dir):
    """All steps pass on first iteration: decomposer + analyzer + feasibility
    + reviewer with no critical findings."""
    from run_pipeline import run_stage_1
    state = make_state(run_dir)

    fake_dispatch.expect(agent="r2c-decomposer",
                         writes={".pipeline/paper_map.json": _paper_map_json()})
    fake_subprocess.expect_script(returncode=0)  # validate_paper_map.py
    fake_dispatch.expect(agent="r2c-method-analyzer",
                         writes={".pipeline/method_spec.json": _method_spec_json()})
    fake_subprocess.expect_script(returncode=0)  # validate_method_spec.py
    fake_subprocess.expect_script(returncode=0)  # check_feasibility.py
    fake_dispatch.expect(agent="r2c-stage-reviewer",
                         writes={".pipeline/stage_review_stage_1_analyzer.json":
                                 _clean_stage_review()})

    result = run_stage_1(state)
    assert_stage_completed(result, "stage_1")
    assert "OUTPUT MODE: CANONICAL ONLY" in fake_dispatch.calls[0].prompt
    assert "OUTPUT MODE: CANONICAL ONLY" in fake_dispatch.calls[1].prompt


# ---------------------------------------------------------------------------
# Stage 1.a — decomposer
# ---------------------------------------------------------------------------


def test_stage_1a_decomposer_halt_sidecar_halts_with_diagnosis(
    fake_dispatch, fake_subprocess, run_dir
):
    """When the decomposer writes paper_map.json.halt (paper non-decomposable),
    the stage halts with the agent's diagnosis."""
    from run_pipeline import run_stage_1
    state = make_state(run_dir)

    halt_content = json.dumps({"reason": "paper not decomposable"})
    fake_dispatch.expect(
        agent="r2c-decomposer",
        writes={".pipeline/paper_map.json.halt": halt_content},
    )

    result = run_stage_1(state)
    assert_stage_halted(result, stage_id="stage_1",
                        reason_contains="decomposer halted")
    # No analyzer or reviewer dispatched (we never got past 1.a).
    assert_not_dispatched(fake_dispatch, "r2c-method-analyzer")


def test_stage_1a_decomposer_reconsidered_artifact_reaches_validator(
    fake_dispatch, fake_subprocess, run_dir
):
    """A halt written before a completed paper map in one dispatch is stale."""
    from run_events import replay_run_dir
    from run_pipeline import run_stage_1

    state = make_state(run_dir)
    fake_dispatch.expect(
        agent="r2c-decomposer",
        writes={
            ".pipeline/paper_map.json.halt": json.dumps(
                {"reason": "paper not decomposable"}
            ),
            ".pipeline/paper_map.json": _paper_map_json(),
        },
    )
    fake_subprocess.expect_script(returncode=0)  # paper-map validator reached
    fake_dispatch.expect(
        agent="r2c-method-analyzer",
        writes={".pipeline/method_spec.json": _method_spec_json()},
    )
    fake_subprocess.expect_script(returncode=0)
    fake_subprocess.expect_script(returncode=0)
    fake_dispatch.expect(
        agent="r2c-stage-reviewer",
        writes={
            ".pipeline/stage_review_stage_1_analyzer.json": _clean_stage_review()
        },
    )

    result = run_stage_1(state)

    assert_stage_completed(result, "stage_1")
    assert not state.paths.paper_map_halt.exists()
    events = _supersession_events(run_dir)
    assert len(events) == 1
    assert events[0]["stage_id"] == "stage_1"
    assert events[0]["details"]["agent"] == "r2c-decomposer"
    assert events[0]["details"]["attempt"] == "initial"
    assert "continue to validation" in events[0]["summary"]
    replay = replay_run_dir(run_dir)
    assert [event["event_type"] for event in replay["artifacts"]] == [
        "artifact_halt_sidecar_superseded"
    ]


def test_stage_1_halt_precedence_rejects_unrelated_future_dated_artifact(
    run_dir,
):
    """Absolute freshness cannot make an unchanged artifact mask a new halt."""
    import run_pipeline

    state = make_state(run_dir)
    artifact = state.paths.method_spec
    sidecar = state.paths.method_spec_halt
    artifact.write_text(_method_spec_json(), encoding="utf-8")
    future_ns = 2_000_000_000_000_000_000
    os.utime(artifact, ns=(future_ns, future_ns))
    before = run_pipeline._snapshot_stage_1_artifact_halt(artifact, sidecar)
    sidecar.write_text(json.dumps({"reason": "current halt"}), encoding="utf-8")

    authoritative = run_pipeline._stage_1_halt_sidecar_is_authoritative(
        state,
        artifact=artifact,
        halt_sidecar=sidecar,
        before_dispatch=before,
        agent="r2c-method-analyzer",
        attempt="initial",
    )

    assert authoritative is True
    assert sidecar.exists()
    assert _supersession_events(run_dir) == []


def test_stage_1_halt_precedence_fails_closed_on_write_time_tie(run_dir):
    """Ambiguous same-dispatch ordering keeps the halt authoritative."""
    import run_pipeline

    state = make_state(run_dir)
    artifact = state.paths.paper_map
    sidecar = state.paths.paper_map_halt
    before = run_pipeline._snapshot_stage_1_artifact_halt(artifact, sidecar)
    sidecar.write_text(json.dumps({"reason": "ambiguous"}), encoding="utf-8")
    artifact.write_text(_paper_map_json(), encoding="utf-8")
    tied_ns = 1_800_000_000_000_000_000
    os.utime(sidecar, ns=(tied_ns, tied_ns))
    os.utime(artifact, ns=(tied_ns, tied_ns))

    authoritative = run_pipeline._stage_1_halt_sidecar_is_authoritative(
        state,
        artifact=artifact,
        halt_sidecar=sidecar,
        before_dispatch=before,
        agent="r2c-decomposer",
        attempt="initial",
    )

    assert authoritative is True
    assert sidecar.exists()
    assert _supersession_events(run_dir) == []


def test_stage_1_halt_precedence_keeps_later_current_halt(run_dir):
    """Artifact-then-halt in one dispatch is a deliberate final halt."""
    import run_pipeline

    state = make_state(run_dir)
    artifact = state.paths.paper_map
    sidecar = state.paths.paper_map_halt
    before = run_pipeline._snapshot_stage_1_artifact_halt(artifact, sidecar)
    artifact.write_text(_paper_map_json(), encoding="utf-8")
    sidecar.write_text(json.dumps({"reason": "final halt"}), encoding="utf-8")
    artifact_ns = 1_800_000_000_000_000_000
    sidecar_ns = artifact_ns + 1
    os.utime(artifact, ns=(artifact_ns, artifact_ns))
    os.utime(sidecar, ns=(sidecar_ns, sidecar_ns))

    authoritative = run_pipeline._stage_1_halt_sidecar_is_authoritative(
        state,
        artifact=artifact,
        halt_sidecar=sidecar,
        before_dispatch=before,
        agent="r2c-decomposer",
        attempt="initial",
    )

    assert authoritative is True
    assert sidecar.exists()
    assert _supersession_events(run_dir) == []


def test_stage_1a_preexisting_paper_map_cannot_mask_halt_sidecar(
    fake_dispatch, run_dir
):
    """A skipped decomposer has no current-dispatch artifact that can win."""
    from run_pipeline import run_stage_1

    state = make_state(run_dir)
    state.paths.paper_map.write_text(_paper_map_json(), encoding="utf-8")
    state.paths.paper_map_halt.write_text(
        json.dumps({"reason": "paper not decomposable"}), encoding="utf-8"
    )

    result = run_stage_1(state)

    assert_stage_halted(
        result,
        stage_id="stage_1",
        reason_contains="not written by a current dispatch",
    )
    assert_not_dispatched(fake_dispatch, "r2c-decomposer")
    assert_not_dispatched(fake_dispatch, "r2c-method-analyzer")
    assert _supersession_events(run_dir) == []


def test_stage_1a_decomposer_missing_output_retry_recovers(
    fake_dispatch, fake_subprocess, run_dir
):
    """When the decomposer's first dispatch writes nothing, a stronger-prompt
    retry fires and writes the paper map; the stage continues."""
    from run_pipeline import run_stage_1
    state = make_state(run_dir)

    # 1st decomposer dispatch — writes nothing
    fake_dispatch.expect(agent="r2c-decomposer", writes={})
    # Retry dispatch — writes chunk parts
    fake_dispatch.expect(agent="r2c-decomposer", writes=_paper_map_part_writes())
    # Validator passes
    fake_subprocess.expect_script(returncode=0)
    # Continue happy path through stage 1.b
    fake_dispatch.expect(agent="r2c-method-analyzer",
                         writes={".pipeline/method_spec.json": _method_spec_json()})
    fake_subprocess.expect_script(returncode=0)
    fake_subprocess.expect_script(returncode=0)
    fake_dispatch.expect(agent="r2c-stage-reviewer",
                         writes={".pipeline/stage_review_stage_1_analyzer.json":
                                 _clean_stage_review()})

    result = run_stage_1(state)
    assert_stage_completed(result, "stage_1")
    # Decomposer ran twice (initial + retry)
    assert_dispatched(fake_dispatch, "r2c-decomposer", times=2)


def test_stage_1a_reconsidered_artifact_wins_on_missing_output_retry(
    fake_dispatch, fake_subprocess, run_dir
):
    """The shared precedence rule also covers the first chunk-mode retry."""
    from run_pipeline import run_stage_1

    state = make_state(run_dir)
    fake_dispatch.expect(agent="r2c-decomposer", writes={})
    fake_dispatch.expect(
        agent="r2c-decomposer",
        writes={
            ".pipeline/paper_map.json.halt": json.dumps({"reason": "halt first"}),
            ".pipeline/paper_map.json": _paper_map_json(),
        },
    )
    fake_subprocess.expect_script(returncode=0)
    _queue_happy_stage_1b(fake_dispatch, fake_subprocess)

    result = run_stage_1(state)

    assert_stage_completed(result, "stage_1")
    events = _supersession_events(run_dir)
    assert len(events) == 1
    assert events[0]["details"]["attempt"] == "retry"


def test_stage_1a_retry_resolves_reconsideration_before_scope_halt(
    fake_dispatch, run_dir
):
    """A stray write still halts, but cannot leave a superseded sidecar."""
    from run_pipeline import run_stage_1

    state = make_state(run_dir)
    fake_dispatch.expect(agent="r2c-decomposer", writes={})
    fake_dispatch.expect(
        agent="r2c-decomposer",
        writes={
            ".pipeline/paper_map.json.halt": json.dumps({"reason": "halt first"}),
            ".pipeline/paper_map.json": _paper_map_json(),
            "method/stray.py": "# out of scope\n",
        },
    )

    result = run_stage_1(state)

    assert_stage_halted(
        result,
        stage_id="stage_1",
        reason_contains="outside its allowlist",
    )
    assert not state.paths.paper_map_halt.exists()
    events = _supersession_events(run_dir)
    assert len(events) == 1
    assert events[0]["details"]["attempt"] == "retry"
    assert_not_dispatched(fake_dispatch, "r2c-method-analyzer")


def test_stage_1a_reconsidered_artifact_wins_on_chunk_assembly_retry(
    fake_dispatch, fake_subprocess, run_dir
):
    """The final paper-map chunk retry uses the same precedence contract."""
    from run_pipeline import run_stage_1

    state = make_state(run_dir)
    malformed_parts = _paper_map_part_writes()
    element_path = next(
        path for path in malformed_parts if "/elements/" in path
    )
    malformed_parts[element_path] = '{"id": "truncated"'
    fake_dispatch.expect(agent="r2c-decomposer", writes={})
    fake_dispatch.expect(agent="r2c-decomposer", writes=malformed_parts)
    fake_dispatch.expect(
        agent="r2c-decomposer",
        writes={
            ".pipeline/paper_map.json.halt": json.dumps({"reason": "halt first"}),
            ".pipeline/paper_map.json": _paper_map_json(),
        },
    )
    fake_subprocess.expect_script(returncode=0)
    _queue_happy_stage_1b(fake_dispatch, fake_subprocess)

    result = run_stage_1(state)

    assert_stage_completed(result, "stage_1")
    events = _supersession_events(run_dir)
    assert len(events) == 1
    assert events[0]["details"]["attempt"] == "chunk_retry"


def test_stage_1a_decomposer_chunked_parts_assemble_and_continue(
    fake_dispatch, fake_subprocess, run_dir
):
    """When canonical output is missing, the decomposer retries in chunk mode;
    the driver assembles the canonical paper_map.json and continues."""
    from run_pipeline import run_stage_1
    state = make_state(run_dir)

    fake_dispatch.expect(agent="r2c-decomposer", writes={})
    fake_dispatch.expect(agent="r2c-decomposer", writes=_paper_map_part_writes())
    fake_subprocess.expect_script(returncode=0)  # validate_paper_map.py
    fake_dispatch.expect(agent="r2c-method-analyzer",
                         writes={".pipeline/method_spec.json": _method_spec_json()})
    fake_subprocess.expect_script(returncode=0)
    fake_subprocess.expect_script(returncode=0)
    fake_dispatch.expect(agent="r2c-stage-reviewer",
                         writes={".pipeline/stage_review_stage_1_analyzer.json":
                                 _clean_stage_review()})

    result = run_stage_1(state)
    assert_stage_completed(result, "stage_1")
    assembled = json.loads((run_dir / ".pipeline" / "paper_map.json").read_text(
        encoding="utf-8"
    ))
    assert assembled["title"] == "Test Paper"
    assert assembled["elements"] == minimal_paper_map()["elements"]
    assert_dispatched(fake_dispatch, "r2c-decomposer", times=2)
    assert "OUTPUT MODE: CHUNK ONLY" in fake_dispatch.calls[1].prompt


def test_stage_1a_decomposer_chunks_in_canonical_mode_retry_as_chunk_only(
    fake_dispatch, fake_subprocess, run_dir
):
    """If the decomposer writes chunk files during the canonical-only attempt,
    the driver clears them and redispatches with an explicit chunk-only prompt."""
    from run_pipeline import run_stage_1
    state = make_state(run_dir)

    fake_dispatch.expect(agent="r2c-decomposer", writes=_paper_map_part_writes())
    fake_dispatch.expect(agent="r2c-decomposer", writes=_paper_map_part_writes())
    fake_subprocess.expect_script(returncode=0)
    fake_dispatch.expect(agent="r2c-method-analyzer",
                         writes={".pipeline/method_spec.json": _method_spec_json()})
    fake_subprocess.expect_script(returncode=0)
    fake_subprocess.expect_script(returncode=0)
    fake_dispatch.expect(agent="r2c-stage-reviewer",
                         writes={".pipeline/stage_review_stage_1_analyzer.json":
                                 _clean_stage_review()})

    result = run_stage_1(state)
    assert_stage_completed(result, "stage_1")
    assert_dispatched(fake_dispatch, "r2c-decomposer", times=2)
    assert "OUTPUT MODE: CANONICAL ONLY" in fake_dispatch.calls[0].prompt
    assert "OUTPUT MODE: CHUNK ONLY" in fake_dispatch.calls[1].prompt


def test_stage_1a_decomposer_missing_output_retry_also_fails_halts(
    fake_dispatch, fake_subprocess, run_dir
):
    """When both the initial dispatch AND the retry produce no paper_map.json,
    the stage halts with the missing-output reason."""
    from run_pipeline import run_stage_1
    state = make_state(run_dir)

    fake_dispatch.expect(agent="r2c-decomposer", writes={})
    fake_dispatch.expect(agent="r2c-decomposer", writes={})

    result = run_stage_1(state)
    assert_stage_halted(result, stage_id="stage_1",
                        reason_contains="decomposer did not produce paper_map.json")


def test_stage_1a_validate_paper_map_failure_routes_to_judge(
    fake_dispatch, fake_subprocess, run_dir
):
    """validate_paper_map.py fails → judge invoked → decomposer fix-mode
    dispatched → re-validation passes → continue."""
    from run_pipeline import run_stage_1
    state = make_state(run_dir)

    fake_dispatch.expect(agent="r2c-decomposer",
                         writes={".pipeline/paper_map.json": _paper_map_json()})
    # Validator fails iter 0
    fake_subprocess.expect_script(returncode=1, stderr="schema error")
    # Judge invoked with default cap=1; decides dispatch_fix to decomposer
    queue_judge_decision(
        fake_dispatch,
        stage_id="stage_1",
        iteration=0,
        validator_label="validate_paper_map.py",
        classification="producer_fixable",
        action="dispatch_fix",
        target_agent="r2c-decomposer",
        finding=_judge_decision(
            iteration=0, target_agent="r2c-decomposer",
        )["finding"],
    )
    # Decomposer fix-mode re-dispatched
    fake_dispatch.expect(agent="r2c-decomposer",
                         writes={".pipeline/paper_map.json": _paper_map_json()})
    # Validator passes
    fake_subprocess.expect_script(returncode=0)
    # Continue happy path through 1.b
    fake_dispatch.expect(agent="r2c-method-analyzer",
                         writes={".pipeline/method_spec.json": _method_spec_json()})
    fake_subprocess.expect_script(returncode=0)
    fake_subprocess.expect_script(returncode=0)
    fake_dispatch.expect(agent="r2c-stage-reviewer",
                         writes={".pipeline/stage_review_stage_1_analyzer.json":
                                 _clean_stage_review()})

    result = run_stage_1(state)
    assert_stage_completed(result, "stage_1")
    assert_dispatched(fake_dispatch, "r2c-halt-judge", times=1)


# ---------------------------------------------------------------------------
# Stage 1.b — analyzer
# ---------------------------------------------------------------------------


def test_stage_1b_analyzer_halt_sidecar_paradigm_mismatch_halts(
    fake_dispatch, fake_subprocess, run_dir
):
    """When the analyzer writes method_spec.json.halt (paradigm not in tree),
    the stage halts and materializes a structured paradigm-gap report."""
    from run_pipeline import run_stage_1
    state = make_state(run_dir)

    # Decomposer phase passes
    fake_dispatch.expect(agent="r2c-decomposer",
                         writes={".pipeline/paper_map.json": _paper_map_json()})
    fake_subprocess.expect_script(returncode=0)
    # Analyzer writes the halt sidecar
    halt_content = json.dumps({"reason": "paradigm mismatch"})
    fake_dispatch.expect(
        agent="r2c-method-analyzer",
        writes={".pipeline/method_spec.json.halt": halt_content},
    )

    result = run_stage_1(state)
    assert_stage_halted(result, stage_id="stage_1",
                        reason_contains="analyzer halted (paradigm mismatch)")
    gap_path = run_dir / ".pipeline" / "paradigm_gap_report.json"
    gap_md = run_dir / ".pipeline" / "paradigm_gap_report.md"
    assert gap_path.is_file()
    assert gap_md.is_file()
    gap_report = json.loads(gap_path.read_text(encoding="utf-8"))
    assert gap_report["decision"] == "unsupported_or_unclear"
    context_report = result.halt_artifact["context"]["paradigm_gap_report"]
    assert context_report["synthesized_by_driver"] is True
    assert context_report["path"] == str(gap_path)
    user_message = result.halt_artifact["user_message"]
    assert "understood the document" in user_message
    assert "coverage or routing" in user_message
    assert "active learning, knowledge distillation" not in user_message
    assert "Paradigm Gap Report" in gap_md.read_text(encoding="utf-8")


def test_stage_1b_analyzer_reconsidered_artifact_enters_validator_fix_loop(
    fake_dispatch, fake_subprocess, run_dir
):
    """The fedavg shape reaches validation even when the artifact needs fixes."""
    from run_pipeline import run_stage_1

    state = make_state(run_dir)
    fake_dispatch.expect(
        agent="r2c-decomposer",
        writes={".pipeline/paper_map.json": _paper_map_json()},
    )
    fake_subprocess.expect_script(returncode=0)
    fake_dispatch.expect(
        agent="r2c-method-analyzer",
        writes={
            ".pipeline/method_spec.json.halt": json.dumps(
                {"reason": "new top-level paradigm needed"}
            ),
            # Deliberately schema-invalid: precedence must not pre-validate.
            ".pipeline/method_spec.json": json.dumps(
                {"schema_version": "1.0.0", "field_name_slip": True}
            ),
        },
    )
    fake_subprocess.expect_script(
        returncode=1,
        stderr="field_name_slip is not a valid MethodSpec field",
    )
    queue_judge_decision(
        fake_dispatch,
        stage_id="stage_1",
        iteration=0,
        validator_label="validate_method_spec.py --strict",
        classification="producer_fixable",
        action="dispatch_fix",
        target_agent="r2c-method-analyzer",
        finding=_judge_decision(
            iteration=0,
            target_agent="r2c-method-analyzer",
            validator_label="validate_method_spec.py --strict",
        )["finding"],
    )
    fake_dispatch.expect(
        agent="r2c-method-analyzer",
        writes={".pipeline/method_spec.json": _method_spec_json()},
    )
    fake_subprocess.expect_script(returncode=0)
    fake_subprocess.expect_script(returncode=0)
    fake_dispatch.expect(
        agent="r2c-stage-reviewer",
        writes={
            ".pipeline/stage_review_stage_1_analyzer.json": _clean_stage_review()
        },
    )

    result = run_stage_1(state)

    assert_stage_completed(result, "stage_1")
    assert not state.paths.method_spec_halt.exists()
    assert_dispatched(fake_dispatch, "r2c-halt-judge", times=1)
    events = _supersession_events(run_dir)
    assert len(events) == 1
    assert events[0]["details"]["agent"] == "r2c-method-analyzer"
    assert events[0]["details"]["attempt"] == "initial"
    assert events[0]["details"]["artifact_changed_in_dispatch"] is True
    assert events[0]["details"]["halt_changed_in_dispatch"] is True


def test_stage_1b_structured_paradigm_gap_authors_provisional_pack_and_retries(
    fake_dispatch, fake_subprocess, run_dir, monkeypatch
):
    """A structured new-subparadigm gap should author a run-local pack,
    retry the analyzer with that pack, and continue without promoting into
    paradigms/."""
    import run_pipeline
    from run_pipeline import run_stage_1

    state = make_state(run_dir)
    proposal_dir = _proposal_dir_for_recovery(run_dir)
    provisional_rel = "run/.pipeline/provisional_packs/manual-proposal/pack.yaml"

    def fake_author_proposal(**kwargs):
        assert kwargs["run_dir"] == run_dir
        return SimpleNamespace(
            proposal_dir=proposal_dir,
            valid=True,
            iterations=1,
            errors=[],
            warnings=[],
            validation_report_path=proposal_dir / "validation_report.json",
            authoring_log_path=proposal_dir / "proposal_authoring_log.md",
        )

    monkeypatch.setattr(run_pipeline, "author_proposal", fake_author_proposal)

    fake_dispatch.expect(
        agent="r2c-decomposer",
        writes={".pipeline/paper_map.json": _paper_map_json()},
    )
    fake_subprocess.expect_script(returncode=0)
    fake_dispatch.expect(
        agent="r2c-method-analyzer",
        writes={
            ".pipeline/method_spec.json.halt": json.dumps({"reason": "new pack needed"}),
            ".pipeline/paradigm_gap_report.json": _new_subparadigm_gap_json(),
        },
    )
    fake_dispatch.expect(
        agent="r2c-method-analyzer",
        writes={
            ".pipeline/method_spec.json": _method_spec_json(
                paradigm_id="active_learning/geometric",
            )
        },
    )
    fake_subprocess.expect_script(returncode=0)
    fake_subprocess.expect_script(returncode=0)
    fake_dispatch.expect(
        agent="r2c-stage-reviewer",
        writes={".pipeline/stage_review_stage_1_analyzer.json": _clean_stage_review()},
    )

    result = run_stage_1(state)

    assert_stage_completed(result, "stage_1")
    manifest_path = run_dir / ".pipeline" / "provisional_pack.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["pack_path"] == provisional_rel
    assert manifest["promotion_required_for_reuse"] is True
    assert (run_dir / ".pipeline" / "provisional_packs" / "manual-proposal" / "pack.yaml").is_file()
    assert "Taxonomy/build-plan source: `run/.pipeline/provisional_packs/manual-proposal/pack.yaml`" in fake_dispatch.calls[2].prompt
    assert not (run_dir.parent / "paradigms" / "supervised_ml" / "active_learning" / "geometric").exists()
    # R2C-031: the retry receives classification as driver-stated fact.
    # The first analyzer pass decides freely (no block); the post-install
    # retry is bound to the installed pack's target and told the
    # reserved-target rule no longer applies, with the honest-halt escape
    # kept for a pack that cannot describe the paper.
    assert "Classification is resolved for this run" not in fake_dispatch.calls[1].prompt
    retry_prompt = fake_dispatch.calls[2].prompt
    assert "Classification is resolved for this run" in retry_prompt
    assert "exactly `active_learning/geometric`" in retry_prompt
    assert "Do NOT re-evaluate `route_elsewhere`" in retry_prompt
    assert "cannot describe this paper's method" in retry_prompt


def test_pack_installed_block_states_manifest_fact_or_stays_out(run_dir):
    """The resolved-classification block (R2C-031) exists exactly when a
    usable installed-pack manifest does. No manifest means no block, a
    manifest missing its target id means no block (failing toward the old
    first-pass contract rather than binding to a blank), and a valid
    manifest binds the block to the manifest's own target id and pack
    path — whatever branch that target lives in."""
    from run_pipeline import _pack_installed_classification_block

    state = make_state(run_dir)
    manifest_path = state.paths.provisional_pack_manifest

    assert _pack_installed_classification_block(state.paths) is None

    manifest_path.write_text(json.dumps(
        {"pack_path": "run/.pipeline/provisional_packs/p/pack.yaml"}
    ), encoding="utf-8")
    assert _pack_installed_classification_block(state.paths) is None

    # The SRL shape: a pack whose target sits in a different branch than
    # the route_elsewhere signal's named home must bind all the same.
    manifest_path.write_text(json.dumps({
        "target_paradigm_id": "motion_planning/rl_collision_avoidance",
        "pack_path": "run/.pipeline/provisional_packs/p/pack.yaml",
    }), encoding="utf-8")
    block = _pack_installed_classification_block(state.paths)
    assert block is not None
    assert "exactly `motion_planning/rl_collision_avoidance`" in block
    assert "run/.pipeline/provisional_packs/p/pack.yaml" in block
    assert "cannot describe this paper's method" in block


def _gap_json_with_malformed_evidence() -> str:
    """The SRL re-roll #2 shape (2026-07-05): a correct new-subparadigm
    decision whose evidence list carries one entry with a typo'd key."""
    raw = json.loads(_new_subparadigm_gap_json())
    raw["paper_evidence"].append({
        "paper_section": "experiments",
        # `quote_or_obsorption` — the literal live typo class.
        "quote_or_obsorption": "policy deployed at nominal speed",
        "relevance": "supports the gap decision",
    })
    return json.dumps(raw, indent=2)


def _gap_json_with_uniformly_drifted_evidence() -> str:
    """The pdfgnn shape (2026-07-27): a correct, complete gap decision whose
    EVERY evidence entry carries the same drifted key. A model that gets a
    field name wrong gets it wrong everywhere, so the drop-and-keep-the-rest
    salvage had nothing left to keep and refused."""
    raw = json.loads(_new_subparadigm_gap_json())
    raw["paper_evidence"] = [
        {
            "paper_section": f"Section {n}",
            "quote_or_relevance": f"verbatim analyzer evidence {n}",
            "relevance": "supports the gap decision",
        }
        for n in (1, 2, 3)
    ]
    return json.dumps(raw, indent=2)


def test_uniformly_drifted_evidence_field_is_renamed_not_dropped():
    """R2C gap-path root cause (pdfgnn 2026-07-27): the unambiguous
    one-to-one key drift is renamed, so a correct decision survives even
    when no entry validates as written. The analyzer's own text is moved,
    never invented."""
    from run_pipeline import _salvage_gap_report

    raw = json.loads(_gap_json_with_uniformly_drifted_evidence())

    report, dropped, repaired = _salvage_gap_report(raw)

    assert report is not None
    assert (dropped, repaired) == (0, 3)
    assert report.decision.value == "new_subparadigm_needed"
    # Every quote is carried over verbatim under the right key.
    assert [e.quote_or_observation for e in report.paper_evidence] == [
        "verbatim analyzer evidence 1",
        "verbatim analyzer evidence 2",
        "verbatim analyzer evidence 3",
    ]


def test_evidence_repair_refuses_ambiguous_and_broken_entries():
    """The repair is bounded: it applies only to exactly one unknown key
    standing in for exactly one missing required field, with a string to
    carry. Anything less clear-cut falls through to the drop path rather
    than guessing which field the content belongs to."""
    from run_pipeline import _repair_evidence_field_alias

    good = {"paper_section": "s", "quote_or_relevance": "q", "relevance": "r"}
    assert _repair_evidence_field_alias(good) is not None

    # Two unknown keys and two missing fields: which goes where is a guess.
    assert _repair_evidence_field_alias(
        {"paper_section": "s", "quote_or_relevance": "q", "why": "r"}) is None
    # Unknown key present but nothing missing: nothing to rename into.
    assert _repair_evidence_field_alias(
        {"paper_section": "s", "quote_or_observation": "q", "relevance": "r",
         "extra": "x"}) is None
    # Missing field but no unknown key to supply it.
    assert _repair_evidence_field_alias(
        {"paper_section": "s", "relevance": "r"}) is None
    # A non-string payload is not a quote.
    assert _repair_evidence_field_alias(
        {"paper_section": "s", "quote_or_relevance": {"nested": 1},
         "relevance": "r"}) is None
    assert _repair_evidence_field_alias("not a dict") is None


def test_salvage_still_refuses_defects_outside_paper_evidence():
    """The load-bearing fields are never salvaged: a bad decision value
    halts, with or without repairable evidence."""
    from run_pipeline import _salvage_gap_report

    raw = json.loads(_gap_json_with_uniformly_drifted_evidence())
    raw["decision"] = "not_a_real_decision"

    report, _dropped, repaired = _salvage_gap_report(raw)

    assert report is None
    assert repaired == 3  # the evidence was repairable; the decision was not


def test_salvage_refuses_when_no_evidence_entry_survives():
    """An evidence list where nothing is readable leaves the report with no
    grounding: refuse rather than ship a decision with zero evidence."""
    from run_pipeline import _salvage_gap_report

    raw = json.loads(_new_subparadigm_gap_json())
    raw["paper_evidence"] = [{"unrelated": "shape"}, {"also": "wrong"}]

    report, dropped, repaired = _salvage_gap_report(raw)

    assert report is None
    assert (dropped, repaired) == (2, 0)


def _gap_json_with_taxonomy_vocabulary(*, parent="TE-TS") -> str:
    """The fedavg shape (2026-07-06, BOTH attempts): a correct gap
    decision whose id fields speak the canonical taxonomy vocabulary
    (uppercase family code; attempt 1 added a prose gloss on the
    parent) instead of legacy paradigm ids."""
    raw = json.loads(_new_subparadigm_gap_json())
    raw["proposed_parent_paradigm"] = parent
    raw["proposed_new_paradigm_id"] = "TE-TS/federated_learning"
    return json.dumps(raw, indent=2)


def test_translate_gap_report_vocabulary_translates_both_live_shapes():
    """Unit pin on the deterministic vocabulary translation: the two
    live fedavg reports translate to the documented legacy equivalence
    (family-code parent means new top level, placement note appended);
    everything outside the narrow provable class refuses."""
    from run_pipeline import _translate_gap_report_vocabulary

    # Three live draws of the same semantic statement (fedavg attempts
    # 1-3): glossed parent + prefixed id, bare parent + prefixed id, and
    # bare parent + BARE id. All translate identically.
    live_shapes = [
        ("TE-TS (Training Strategy family)", "TE-TS/federated_learning"),
        ("TE-TS", "TE-TS/federated_learning"),
        ("TE-TS", "federated_learning"),
    ]
    for parent, proposed in live_shapes:
        raw = json.loads(_gap_json_with_taxonomy_vocabulary(parent=parent))
        raw["proposed_new_paradigm_id"] = proposed
        translation = _translate_gap_report_vocabulary(raw)
        assert translation is not None, (parent, proposed)
        translated, notes = translation
        assert translated["decision"] == "new_top_level_needed"
        assert translated["proposed_parent_paradigm"] is None
        assert translated["proposed_new_paradigm_id"] == "federated_learning"
        assert "TE-TS" in translated["recommended_next_action"]
        assert any("judgment change" in n for n in notes)

    base = json.loads(_gap_json_with_taxonomy_vocabulary())
    # Unknown group code: not our defect class, refuse.
    assert _translate_gap_report_vocabulary(dict(
        base, proposed_parent_paradigm="ZZ-XX",
        proposed_new_paradigm_id="ZZ-XX/federated_learning")) is None
    # Deeper-than-one-segment proposals have no live case: refuse.
    assert _translate_gap_report_vocabulary(dict(
        base, proposed_new_paradigm_id="TE-TS/federated/averaging")) is None
    # Parent code and proposal prefix must agree: refuse.
    assert _translate_gap_report_vocabulary(dict(
        base, proposed_parent_paradigm="OM-OPT")) is None
    # Only sub-paradigm proposals carry the family-parent equivalence.
    assert _translate_gap_report_vocabulary(dict(
        base, decision="unsupported_or_unclear")) is None


def test_stage_1b_malformed_evidence_gap_report_is_salvaged_and_recovers(
    fake_dispatch, fake_subprocess, run_dir, monkeypatch
):
    """A typo'd key in ONE supporting-evidence entry must not discard a
    correct gap decision: the driver salvages the report (raw preserved as
    .rejected), records the salvage event, and the pack recovery runs."""
    import run_pipeline
    from run_pipeline import run_stage_1

    state = make_state(run_dir)
    proposal_dir = _proposal_dir_for_recovery(run_dir)

    def fake_author_proposal(**kwargs):
        return SimpleNamespace(
            proposal_dir=proposal_dir,
            valid=True,
            iterations=1,
            errors=[],
            warnings=[],
            validation_report_path=proposal_dir / "validation_report.json",
            authoring_log_path=proposal_dir / "proposal_authoring_log.md",
        )

    monkeypatch.setattr(run_pipeline, "author_proposal", fake_author_proposal)

    fake_dispatch.expect(
        agent="r2c-decomposer",
        writes={".pipeline/paper_map.json": _paper_map_json()},
    )
    fake_subprocess.expect_script(returncode=0)
    fake_dispatch.expect(
        agent="r2c-method-analyzer",
        writes={
            ".pipeline/method_spec.json.halt": json.dumps({"reason": "new pack needed"}),
            ".pipeline/paradigm_gap_report.json": _gap_json_with_malformed_evidence(),
        },
    )
    fake_dispatch.expect(
        agent="r2c-method-analyzer",
        writes={
            ".pipeline/method_spec.json": _method_spec_json(
                paradigm_id="active_learning/geometric",
            )
        },
    )
    fake_subprocess.expect_script(returncode=0)
    fake_subprocess.expect_script(returncode=0)
    fake_dispatch.expect(
        agent="r2c-stage-reviewer",
        writes={".pipeline/stage_review_stage_1_analyzer.json": _clean_stage_review()},
    )

    result = run_stage_1(state)

    assert_stage_completed(result, "stage_1")
    # The raw report survives for audit (the canonical repaired file is
    # consumed by the recovery and then cleared with the other stale halt
    # artifacts before the analyzer retry — pinned by the stale-artifacts
    # test below).
    rejected = run_dir / ".pipeline" / "paradigm_gap_report.json.rejected"
    assert rejected.is_file()
    assert "quote_or_obsorption" in rejected.read_text(encoding="utf-8")
    # The salvage is recorded as a run event, never silent.
    events = [
        json.loads(line)
        for line in (run_dir / ".pipeline" / "run_events.jsonl").read_text(
            encoding="utf-8").splitlines()
    ]
    salvage_events = [e for e in events
                      if e["event_type"] == "gap_report_salvaged"]
    assert salvage_events
    # The typo'd entry is RENAMED, not discarded: the unambiguous one-to-one
    # drift preserves the analyzer's evidence instead of dropping readable
    # text (the pdfgnn 2026-07-27 root cause, fixed for both shapes).
    assert salvage_events[-1]["details"]["repaired_entries"] == 1
    assert salvage_events[-1]["details"]["dropped_entries"] == 0
    # And the recovery actually ran: the provisional pack is installed.
    assert (run_dir / ".pipeline" / "provisional_pack.json").is_file()


def test_stage_1b_taxonomy_vocabulary_gap_report_is_translated_and_recovers(
    fake_dispatch, fake_subprocess, run_dir, monkeypatch
):
    """The fedavg 2026-07-06 defect, end to end: a gap report whose id
    fields speak the canonical taxonomy vocabulary must be translated
    deterministically (raw preserved, loud run event, analyzer judgment
    unchanged) and the pack recovery must run — this exact shape killed
    the gap-path proof twice in one night."""
    import run_pipeline
    from run_pipeline import run_stage_1

    state = make_state(run_dir)
    proposal_dir = _proposal_dir_for_recovery(run_dir)

    def fake_author_proposal(**kwargs):
        return SimpleNamespace(
            proposal_dir=proposal_dir,
            valid=True,
            iterations=1,
            errors=[],
            warnings=[],
            validation_report_path=proposal_dir / "validation_report.json",
            authoring_log_path=proposal_dir / "proposal_authoring_log.md",
        )

    monkeypatch.setattr(run_pipeline, "author_proposal", fake_author_proposal)

    fake_dispatch.expect(
        agent="r2c-decomposer",
        writes={".pipeline/paper_map.json": _paper_map_json()},
    )
    fake_subprocess.expect_script(returncode=0)
    fake_dispatch.expect(
        agent="r2c-method-analyzer",
        writes={
            ".pipeline/method_spec.json.halt": json.dumps({"reason": "new pack needed"}),
            ".pipeline/paradigm_gap_report.json": _gap_json_with_taxonomy_vocabulary(
                parent="TE-TS (Training Strategy family)",
            ),
        },
    )
    fake_dispatch.expect(
        agent="r2c-method-analyzer",
        writes={
            ".pipeline/method_spec.json": _method_spec_json(
                paradigm_id="active_learning/geometric",
            )
        },
    )
    fake_subprocess.expect_script(returncode=0)
    fake_subprocess.expect_script(returncode=0)
    fake_dispatch.expect(
        agent="r2c-stage-reviewer",
        writes={".pipeline/stage_review_stage_1_analyzer.json": _clean_stage_review()},
    )

    result = run_stage_1(state)

    assert_stage_completed(result, "stage_1")
    # Raw taxonomy-vocabulary report preserved for audit.
    rejected = run_dir / ".pipeline" / "paradigm_gap_report.json.rejected"
    assert rejected.is_file()
    assert "TE-TS/federated_learning" in rejected.read_text(encoding="utf-8")
    # Loud, never silent.
    events = [
        json.loads(line)
        for line in (run_dir / ".pipeline" / "run_events.jsonl").read_text(
            encoding="utf-8").splitlines()
    ]
    assert any(e["event_type"] == "gap_report_vocabulary_translated"
               for e in events)
    # And the recovery actually ran on the translated report.
    assert (run_dir / ".pipeline" / "provisional_pack.json").is_file()


def test_stage_1b_unsalvageable_gap_report_halts_with_the_validation_error(
    fake_dispatch, fake_subprocess, run_dir
):
    """A defect OUTSIDE supporting evidence (a bad decision value) is not
    salvageable: the run halts, and the halt reason names the schema
    failure instead of a generic mismatch."""
    from run_pipeline import run_stage_1

    state = make_state(run_dir)
    raw = json.loads(_new_subparadigm_gap_json())
    raw["decision"] = "brand_new_universe"

    fake_dispatch.expect(
        agent="r2c-decomposer",
        writes={".pipeline/paper_map.json": _paper_map_json()},
    )
    fake_subprocess.expect_script(returncode=0)
    fake_dispatch.expect(
        agent="r2c-method-analyzer",
        writes={
            ".pipeline/method_spec.json.halt": json.dumps({"reason": "new pack needed"}),
            ".pipeline/paradigm_gap_report.json": json.dumps(raw),
        },
    )

    result = run_stage_1(state)
    assert_stage_halted(result, stage_id="stage_1",
                        reason_contains="failed schema validation")


def test_stage_1b_fresh_halt_beats_preexisting_method_spec(
    fake_dispatch, fake_subprocess, run_dir
):
    """Coexistence alone cannot let an unrelated stale spec mask a new halt."""
    from run_pipeline import run_stage_1

    state = make_state(run_dir)
    pipeline = run_dir / ".pipeline"
    (pipeline / "paper_map.json").write_text(_paper_map_json(), encoding="utf-8")
    (pipeline / "method_spec.json").write_text(
        _method_spec_json(), encoding="utf-8"
    )
    future_ns = 2_000_000_000_000_000_000
    os.utime(pipeline / "method_spec.json", ns=(future_ns, future_ns))
    fake_dispatch.expect(
        agent="r2c-method-analyzer",
        writes={
            ".pipeline/method_spec.json.halt": json.dumps(
                {"reason": "current paradigm mismatch"}
            )
        },
    )

    result = run_stage_1(state)

    assert_stage_halted(
        result,
        stage_id="stage_1",
        reason_contains="analyzer halted (paradigm mismatch)",
    )
    assert state.paths.method_spec_halt.exists()
    assert _supersession_events(run_dir) == []


def test_stage_1b_clears_stale_analyzer_halt_artifacts_before_rerun(
    fake_dispatch, fake_subprocess, run_dir
):
    """A prior analyzer paradigm halt should not poison a fresh Stage 1 rerun."""
    from run_pipeline import run_stage_1
    state = make_state(run_dir)

    pipeline = run_dir / ".pipeline"
    (pipeline / "paper_map.json").write_text(_paper_map_json(), encoding="utf-8")
    (pipeline / "stage_1.halt").write_text(
        json.dumps({"reason": "previous paradigm mismatch"}),
        encoding="utf-8",
    )
    (pipeline / "method_spec.json.halt").write_text(
        json.dumps({"reason": "previous paradigm mismatch"}),
        encoding="utf-8",
    )
    (pipeline / "paradigm_gap_report.json").write_text(
        json.dumps({"schema_version": "1.0.0", "stale": True}),
        encoding="utf-8",
    )
    (pipeline / "paradigm_gap_report.md").write_text("stale\n", encoding="utf-8")

    fake_dispatch.expect(
        agent="r2c-method-analyzer",
        writes={".pipeline/method_spec.json": _method_spec_json()},
    )
    fake_subprocess.expect_script(returncode=0)
    fake_subprocess.expect_script(returncode=0)
    fake_dispatch.expect(
        agent="r2c-stage-reviewer",
        writes={".pipeline/stage_review_stage_1_analyzer.json": _clean_stage_review()},
    )

    result = run_stage_1(state)
    assert_stage_completed(result, "stage_1")
    assert_not_dispatched(fake_dispatch, "r2c-decomposer")
    assert_dispatched(fake_dispatch, "r2c-method-analyzer", times=1)
    assert not (pipeline / "stage_1.halt").exists()
    assert not (pipeline / "method_spec.json.halt").exists()
    assert not (pipeline / "paradigm_gap_report.json").exists()
    assert not (pipeline / "paradigm_gap_report.md").exists()


def test_stage_1b_analyzer_missing_output_retry_recovers(
    fake_dispatch, fake_subprocess, run_dir
):
    """Analyzer first dispatch writes nothing → retry writes method_spec.json
    → stage continues."""
    from run_pipeline import run_stage_1
    state = make_state(run_dir)

    fake_dispatch.expect(agent="r2c-decomposer",
                         writes={".pipeline/paper_map.json": _paper_map_json()})
    fake_subprocess.expect_script(returncode=0)
    fake_dispatch.expect(agent="r2c-method-analyzer", writes={})  # missing
    fake_dispatch.expect(agent="r2c-method-analyzer",
                         writes=_method_spec_part_writes())
    fake_subprocess.expect_script(returncode=0)
    fake_subprocess.expect_script(returncode=0)
    fake_dispatch.expect(agent="r2c-stage-reviewer",
                         writes={".pipeline/stage_review_stage_1_analyzer.json":
                                 _clean_stage_review()})

    result = run_stage_1(state)
    assert_stage_completed(result, "stage_1")
    assert_dispatched(fake_dispatch, "r2c-method-analyzer", times=2)


def test_stage_1b_reconsidered_artifact_wins_on_missing_output_retry(
    fake_dispatch, fake_subprocess, run_dir
):
    """The fedavg precedence rule applies on the first analyzer retry."""
    from run_pipeline import run_stage_1

    state = make_state(run_dir)
    fake_dispatch.expect(
        agent="r2c-decomposer",
        writes={".pipeline/paper_map.json": _paper_map_json()},
    )
    fake_subprocess.expect_script(returncode=0)
    fake_dispatch.expect(agent="r2c-method-analyzer", writes={})
    fake_dispatch.expect(
        agent="r2c-method-analyzer",
        writes={
            ".pipeline/method_spec.json.halt": json.dumps(
                {"reason": "halt first"}
            ),
            ".pipeline/method_spec.json": _method_spec_json(),
        },
    )
    _queue_happy_after_method_spec(fake_dispatch, fake_subprocess)

    result = run_stage_1(state)

    assert_stage_completed(result, "stage_1")
    events = _supersession_events(run_dir)
    assert len(events) == 1
    assert events[0]["details"]["attempt"] == "retry"


def test_stage_1b_reconsidered_artifact_wins_on_chunk_assembly_retry(
    fake_dispatch, fake_subprocess, run_dir
):
    """The final analyzer chunk retry uses the same precedence contract."""
    from run_pipeline import run_stage_1

    state = make_state(run_dir)
    fake_dispatch.expect(
        agent="r2c-decomposer",
        writes={".pipeline/paper_map.json": _paper_map_json()},
    )
    fake_subprocess.expect_script(returncode=0)
    fake_dispatch.expect(agent="r2c-method-analyzer", writes={})
    fake_dispatch.expect(
        agent="r2c-method-analyzer",
        writes=_malformed_method_spec_part_writes(),
    )
    fake_dispatch.expect(
        agent="r2c-method-analyzer",
        writes={
            ".pipeline/method_spec.json.halt": json.dumps(
                {"reason": "halt first"}
            ),
            ".pipeline/method_spec.json": _method_spec_json(),
        },
    )
    _queue_happy_after_method_spec(fake_dispatch, fake_subprocess)

    result = run_stage_1(state)

    assert_stage_completed(result, "stage_1")
    events = _supersession_events(run_dir)
    assert len(events) == 1
    assert events[0]["details"]["attempt"] == "chunk_retry"


def test_stage_1b_analyzer_chunked_parts_assemble_and_continue(
    fake_dispatch, fake_subprocess, run_dir
):
    """When canonical output is missing, the analyzer retries in chunk mode;
    the driver assembles the canonical method_spec.json and continues."""
    from run_pipeline import run_stage_1
    state = make_state(run_dir)

    fake_dispatch.expect(agent="r2c-decomposer",
                         writes={".pipeline/paper_map.json": _paper_map_json()})
    fake_subprocess.expect_script(returncode=0)
    fake_dispatch.expect(agent="r2c-method-analyzer", writes={})
    fake_dispatch.expect(agent="r2c-method-analyzer",
                         writes=_method_spec_part_writes())
    fake_subprocess.expect_script(returncode=0)
    fake_subprocess.expect_script(returncode=0)
    fake_dispatch.expect(agent="r2c-stage-reviewer",
                         writes={".pipeline/stage_review_stage_1_analyzer.json":
                                 _clean_stage_review()})

    result = run_stage_1(state)
    assert_stage_completed(result, "stage_1")
    assembled = json.loads((run_dir / ".pipeline" / "method_spec.json").read_text(
        encoding="utf-8"
    ))
    assert assembled["paper"] == minimal_method_spec()["paper"]
    assert assembled["core_method"] == minimal_method_spec()["core_method"]
    assert assembled["feasibility"] == "reproducible"
    assert_dispatched(fake_dispatch, "r2c-method-analyzer", times=2)
    assert "OUTPUT MODE: CHUNK ONLY" in fake_dispatch.calls[2].prompt


def test_stage_1b_malformed_chunk_retry_prompt_includes_part_error(
    fake_dispatch, fake_subprocess, run_dir
):
    """If analyzer writes malformed method_spec_parts, the missing-output
    retry includes the assembly error so the agent can rewrite the bad part."""
    from run_pipeline import run_stage_1
    state = make_state(run_dir)

    fake_dispatch.expect(agent="r2c-decomposer",
                         writes={".pipeline/paper_map.json": _paper_map_json()})
    fake_subprocess.expect_script(returncode=0)
    fake_dispatch.expect(agent="r2c-method-analyzer", writes={})
    fake_dispatch.expect(agent="r2c-method-analyzer",
                         writes=_malformed_method_spec_part_writes())
    fake_dispatch.expect(agent="r2c-method-analyzer",
                         writes=_method_spec_part_writes())
    fake_subprocess.expect_script(returncode=0)
    fake_subprocess.expect_script(returncode=0)
    fake_dispatch.expect(agent="r2c-stage-reviewer",
                         writes={".pipeline/stage_review_stage_1_analyzer.json":
                                 _clean_stage_review()})

    result = run_stage_1(state)
    assert_stage_completed(result, "stage_1")
    assert_dispatched(fake_dispatch, "r2c-method-analyzer", times=3)
    retry_prompt = fake_dispatch.calls[3].prompt
    assert "Previous output error" in retry_prompt
    assert "comparison.json" in retry_prompt
    assert "invalid JSON" in retry_prompt
    assert "raw unquoted text" in retry_prompt
    assert "OUTPUT MODE: CHUNK ONLY" in retry_prompt


def test_stage_1b_method_spec_parts_require_wrapper_objects(run_dir):
    """Method-spec chunk files must wrap exactly one top-level spec key."""
    from stage1_artifact_parts import assemble_method_spec_parts

    writes = _method_spec_part_writes()
    writes[".pipeline/method_spec_parts/feasibility.json"] = json.dumps(
        "reproducible"
    )
    _write_run_files(run_dir, writes)

    result = assemble_method_spec_parts(
        run_dir / ".pipeline",
        run_dir / ".pipeline" / "method_spec.json",
    )

    assert not result.assembled
    assert "wrapper object" in result.message
    assert "feasibility.json" in result.message


def test_stage_1a_paper_map_element_parts_reject_full_envelope(run_dir):
    """Paper-map element chunks must contain one element, not full paper_map."""
    from stage1_artifact_parts import assemble_paper_map_parts

    writes = _paper_map_part_writes()
    first_part = next(
        rel for rel in writes if rel.startswith(".pipeline/paper_map_parts/elements/")
    )
    writes[first_part] = _paper_map_json()
    _write_run_files(run_dir, writes)

    result = assemble_paper_map_parts(
        run_dir / ".pipeline",
        run_dir / ".pipeline" / "paper_map.json",
    )

    assert not result.assembled
    assert "not a full paper_map envelope" in result.message
    assert first_part.split("/")[-1] in result.message


def test_stage_1b_validate_method_spec_failure_routes_to_judge(
    fake_dispatch, fake_subprocess, run_dir
):
    """validate_method_spec.py fails → judge → analyzer fix → re-validate
    → continue."""
    from run_pipeline import run_stage_1
    state = make_state(run_dir)

    fake_dispatch.expect(agent="r2c-decomposer",
                         writes={".pipeline/paper_map.json": _paper_map_json()})
    fake_subprocess.expect_script(returncode=0)
    fake_dispatch.expect(agent="r2c-method-analyzer",
                         writes={".pipeline/method_spec.json": _method_spec_json()})
    # validate_method_spec.py fails
    aggregate_error = (
        "methodology_replication_contract core detail completeness failed "
        "(2 missing field(s)):\n"
        "  - methodology_replication_contract.elements.0.required_controls: "
        "core methodology element 'a' requires non-empty required_controls\n"
        "  - methodology_replication_contract.elements.1.fairness_checks: "
        "core methodology element 'b' requires non-empty fairness_checks\n"
    )
    fake_subprocess.expect_script(returncode=1, stderr=aggregate_error)
    # Judge → analyzer fix-mode
    queue_judge_decision(
        fake_dispatch,
        stage_id="stage_1",
        iteration=0,
        validator_label="validate_method_spec.py --strict",
        classification="producer_fixable",
        action="dispatch_fix",
        target_agent="r2c-method-analyzer",
        finding=_judge_decision(
            iteration=0, target_agent="r2c-method-analyzer",
            validator_label="validate_method_spec.py --strict",
        )["finding"],
    )
    fake_dispatch.expect(agent="r2c-method-analyzer",
                         writes={".pipeline/method_spec.json": _method_spec_json()})
    fake_subprocess.expect_script(returncode=0)
    # Continue happy path
    fake_subprocess.expect_script(returncode=0)
    fake_dispatch.expect(agent="r2c-stage-reviewer",
                         writes={".pipeline/stage_review_stage_1_analyzer.json":
                                 _clean_stage_review()})

    result = run_stage_1(state)
    assert_stage_completed(result, "stage_1")
    assert aggregate_error in fake_dispatch.calls[2].prompt


def test_stage_1b_validate_method_spec_fix_no_write_gets_one_retry(
    fake_dispatch, fake_subprocess, run_dir
):
    """If judge-routed analyzer fix-mode returns without writing the spec, the
    validator loop gives that no-write turn one direct retry before treating
    the unchanged validator output as fixation."""
    from run_pipeline import run_stage_1
    state = make_state(run_dir)

    fake_dispatch.expect(agent="r2c-decomposer",
                         writes={".pipeline/paper_map.json": _paper_map_json()})
    fake_subprocess.expect_script(returncode=0)
    fake_dispatch.expect(agent="r2c-method-analyzer",
                         writes={".pipeline/method_spec.json": _method_spec_json()})
    fake_subprocess.expect_script(returncode=1, stderr="schema error")
    queue_judge_decision(
        fake_dispatch,
        stage_id="stage_1",
        iteration=0,
        validator_label="validate_method_spec.py --strict",
        classification="producer_fixable",
        action="dispatch_fix",
        target_agent="r2c-method-analyzer",
        finding=_judge_decision(
            iteration=0, target_agent="r2c-method-analyzer",
            validator_label="validate_method_spec.py --strict",
        )["finding"],
    )
    fake_dispatch.expect(agent="r2c-method-analyzer", writes={})
    fake_subprocess.expect_script(returncode=1, stderr="schema error")
    fake_dispatch.expect(agent="r2c-method-analyzer",
                         writes={".pipeline/method_spec.json": _method_spec_json()})
    fake_subprocess.expect_script(returncode=0)
    fake_subprocess.expect_script(returncode=0)
    fake_dispatch.expect(agent="r2c-stage-reviewer",
                         writes={".pipeline/stage_review_stage_1_analyzer.json":
                                 _clean_stage_review()})

    result = run_stage_1(state)

    assert_stage_completed(result, "stage_1")
    assert_dispatched(fake_dispatch, "r2c-method-analyzer", times=3)
    assert "completed without writing" in fake_dispatch.calls[4].prompt


def test_stage_1b_feasibility_gate_blocks_halts(
    fake_dispatch, fake_subprocess, run_dir
):
    """When check_feasibility.py writes a halt sidecar, the stage halts
    with the gate's reason."""
    from run_pipeline import run_stage_1
    state = make_state(run_dir)

    fake_dispatch.expect(agent="r2c-decomposer",
                         writes={".pipeline/paper_map.json": _paper_map_json()})
    fake_subprocess.expect_script(returncode=0)
    fake_dispatch.expect(agent="r2c-method-analyzer",
                         writes={".pipeline/method_spec.json": _method_spec_json()})
    fake_subprocess.expect_script(returncode=0)
    # The real check_feasibility.py signals "blocked" by writing a halt
    # sidecar at paths.feasibility_halt (not via exit code). The fake
    # script materializes that side-effect file via `writes=`.
    fake_subprocess.set_run_dir(run_dir)
    fake_subprocess.expect_script(
        returncode=0,
        writes={".pipeline/feasibility_gate.json.halt":
                json.dumps({"reason": "paradigm not supported"})},
    )

    result = run_stage_1(state)
    assert_stage_halted(result, stage_id="stage_1",
                        reason_contains="feasibility gate blocked")
    # Reviewer never dispatched (feasibility blocked before reviewer step)
    assert_not_dispatched(fake_dispatch, "r2c-stage-reviewer")


# ---------------------------------------------------------------------------
# Stage 1.b — stage reviewer
# ---------------------------------------------------------------------------


def test_stage_1b_reviewer_missing_findings_retry_recovers(
    fake_dispatch, fake_subprocess, run_dir
):
    """Stage reviewer's first dispatch writes no findings file → retry → ok."""
    from run_pipeline import run_stage_1
    state = make_state(run_dir)

    fake_dispatch.expect(agent="r2c-decomposer",
                         writes={".pipeline/paper_map.json": _paper_map_json()})
    fake_subprocess.expect_script(returncode=0)
    fake_dispatch.expect(agent="r2c-method-analyzer",
                         writes={".pipeline/method_spec.json": _method_spec_json()})
    fake_subprocess.expect_script(returncode=0)
    fake_subprocess.expect_script(returncode=0)
    # Initial reviewer dispatch — writes nothing
    fake_dispatch.expect(agent="r2c-stage-reviewer", writes={})
    # Retry — writes the review
    fake_dispatch.expect(agent="r2c-stage-reviewer",
                         writes={".pipeline/stage_review_stage_1_analyzer.json":
                                 _clean_stage_review()})

    result = run_stage_1(state)
    assert_stage_completed(result, "stage_1")
    assert_dispatched(fake_dispatch, "r2c-stage-reviewer", times=2)


def test_stage_1b_reviewer_missing_stage_id_degrades_with_known_issue(
    fake_dispatch, fake_subprocess, run_dir
):
    """Validated Stage 1 producer artifacts should not hard-halt solely
    because the semantic reviewer omitted its contract echo field."""
    from run_pipeline import run_stage_1
    state = make_state(run_dir)

    fake_dispatch.expect(agent="r2c-decomposer",
                         writes={".pipeline/paper_map.json": _paper_map_json()})
    fake_subprocess.expect_script(returncode=0)
    fake_dispatch.expect(agent="r2c-method-analyzer",
                         writes={".pipeline/method_spec.json": _method_spec_json()})
    fake_subprocess.expect_script(returncode=0)
    fake_subprocess.expect_script(returncode=0)
    fake_dispatch.expect(
        agent="r2c-stage-reviewer",
        writes={".pipeline/stage_review_stage_1_analyzer.json":
                _stage_review_missing_stage_id()},
    )

    result = run_stage_1(state)

    assert_stage_degraded(
        result,
        stage_id="stage_1",
        run_dir=run_dir,
        known_issue_contains="missing required field 'stage_id'",
    )


def test_stage_1b_reviewer_critical_findings_dispatch_analyzer_fix_recovers(
    fake_dispatch, fake_subprocess, run_dir
):
    """Reviewer finds critical findings → analyzer fix-mode dispatched →
    re-validate → re-feasibility → re-review (clean) → stage completes."""
    from run_pipeline import run_stage_1
    from schemas.method_spec import SCHEMA_VERSION  # ensure import works
    state = make_state(run_dir)

    fake_dispatch.expect(agent="r2c-decomposer",
                         writes={".pipeline/paper_map.json": _paper_map_json()})
    fake_subprocess.expect_script(returncode=0)
    fake_dispatch.expect(agent="r2c-method-analyzer",
                         writes={".pipeline/method_spec.json": _method_spec_json()})
    fake_subprocess.expect_script(returncode=0)
    fake_subprocess.expect_script(returncode=0)
    # Iter 0: reviewer reports critical
    fake_dispatch.expect(agent="r2c-stage-reviewer",
                         writes={".pipeline/stage_review_stage_1_analyzer.json":
                                 _stage_review_with_critical()})
    # Analyzer fix-mode dispatched
    fake_dispatch.expect(agent="r2c-method-analyzer",
                         writes={".pipeline/method_spec.json": _method_spec_json()})
    # Post-reviewer-fix validator passes
    fake_subprocess.expect_script(returncode=0)
    # Post-reviewer-fix feasibility passes
    fake_subprocess.expect_script(returncode=0)
    # Iter 1: reviewer clean
    fake_dispatch.expect(agent="r2c-stage-reviewer",
                         writes={".pipeline/stage_review_stage_1_analyzer.json":
                                 _clean_stage_review()})

    result = run_stage_1(state)
    assert_stage_completed(result, "stage_1")
    # Analyzer ran twice (initial + fix); reviewer ran twice (initial + post-fix)
    assert_dispatched(fake_dispatch, "r2c-method-analyzer", times=2)
    assert_dispatched(fake_dispatch, "r2c-stage-reviewer", times=2)


def test_stage_1b_reviewer_findings_persist_after_cap_halts(
    fake_dispatch, fake_subprocess, run_dir
):
    """Stage 1 reviewer cap=1: if findings persist after one fix-mode round,
    the stage halts with the persistent-findings reason."""
    from run_pipeline import run_stage_1
    state = make_state(run_dir)

    fake_dispatch.expect(agent="r2c-decomposer",
                         writes={".pipeline/paper_map.json": _paper_map_json()})
    fake_subprocess.expect_script(returncode=0)
    fake_dispatch.expect(agent="r2c-method-analyzer",
                         writes={".pipeline/method_spec.json": _method_spec_json()})
    fake_subprocess.expect_script(returncode=0)
    fake_subprocess.expect_script(returncode=0)
    # Iter 0: reviewer reports critical
    fake_dispatch.expect(agent="r2c-stage-reviewer",
                         writes={".pipeline/stage_review_stage_1_analyzer.json":
                                 _stage_review_with_critical()})
    # Analyzer fix-mode
    fake_dispatch.expect(agent="r2c-method-analyzer",
                         writes={".pipeline/method_spec.json": _method_spec_json()})
    fake_subprocess.expect_script(returncode=0)
    fake_subprocess.expect_script(returncode=0)
    # Iter 1: reviewer STILL reports critical
    fake_dispatch.expect(agent="r2c-stage-reviewer",
                         writes={".pipeline/stage_review_stage_1_analyzer.json":
                                 _stage_review_with_critical(finding_id="F002")})

    result = run_stage_1(state)
    assert_stage_halted(result, stage_id="stage_1",
                        reason_contains="critical finding(s) persist")


# ---------------------------------------------------------------------------
# Stage 1.b — feasibility re-ask (analyzer-proposed surrogate on a
# cannot_implement halt; 2026-07-03 bev-distill roll-variance case)
# ---------------------------------------------------------------------------


def _cannot_implement_halt(n=2, with_surrogate=True) -> dict:
    blockers = []
    for i in range(n):
        blockers.append({
            "requirement": f"licensed dataset {i}",
            "reason": "requires a license agreement; not auto-downloadable",
            "resolution_proposed_by_analyzer": (
                "Demo uses synthetic feature maps; the loss computation "
                "is identical, only the data source differs."
                if with_surrogate else ""),
        })
    return {
        "status": "halted",
        "reason": f"Feasibility gate: {n} blocker(s) marked `cannot_implement`.",
        "cannot_implement_blockers": blockers,
    }


def _seed_method_spec(state) -> str:
    content = json.dumps({"schema_version": "1.5.0", "seed": "original"})
    state.paths.method_spec.write_text(content, encoding="utf-8")
    return content


def test_feasibility_reask_commit_path_proceeds_with_assumption(
    run_dir, monkeypatch
):
    """Analyzer commits to its surrogate → spec revalidates, gate passes,
    the run proceeds, and the approximation lands in assumptions.md."""
    import run_pipeline

    state = make_state(run_dir)
    _seed_method_spec(state)
    calls = {"dispatch": 0}

    def fake_dispatch(_state, blockers):
        calls["dispatch"] += 1
        assert len(blockers) == 2
        _state.paths.method_spec.write_text(
            json.dumps({"schema_version": "1.5.0", "seed": "edited"}),
            encoding="utf-8")

    monkeypatch.setattr(
        run_pipeline, "_dispatch_analyzer_feasibility_reask", fake_dispatch)
    monkeypatch.setattr(
        run_pipeline, "_run_fresh_spec_validator", lambda s: (True, ""))
    monkeypatch.setattr(
        run_pipeline, "_run_feasibility_gate", lambda s: (True, None))

    ok, halt_record = run_pipeline._reask_feasibility_cannot_implement(
        state, "stage_1", _cannot_implement_halt())

    assert ok is True
    assert halt_record is None
    assert calls["dispatch"] == 1
    import run_layout
    md = (state.paths.run_dir / run_layout.ASSUMPTIONS_MD).read_text(encoding="utf-8")
    assert "approved approximations" in md
    assert "licensed dataset 0" in md
    # The analyzer's edit survives (no restore on the success path).
    assert "edited" in state.paths.method_spec.read_text(encoding="utf-8")


def test_feasibility_reask_confirmed_halt_uses_fresh_record(
    run_dir, monkeypatch
):
    """Analyzer keeps cannot_implement → gate halts again → the halt stands
    as a deliberate decision with the FRESH record; no assumption logged."""
    import run_pipeline

    state = make_state(run_dir)
    _seed_method_spec(state)
    fresh_halt = {"status": "halted", "reason": "confirmed",
                  "cannot_implement_blockers": []}

    monkeypatch.setattr(
        run_pipeline, "_dispatch_analyzer_feasibility_reask",
        lambda _s, _b: None)
    monkeypatch.setattr(
        run_pipeline, "_run_fresh_spec_validator", lambda s: (True, ""))
    monkeypatch.setattr(
        run_pipeline, "_run_feasibility_gate", lambda s: (False, fresh_halt))

    ok, halt_record = run_pipeline._reask_feasibility_cannot_implement(
        state, "stage_1", _cannot_implement_halt())

    assert ok is False
    assert halt_record == fresh_halt
    import run_layout
    assert not (state.paths.run_dir / run_layout.ASSUMPTIONS_MD).exists()


def test_feasibility_reask_skips_when_any_blocker_lacks_surrogate(
    run_dir, monkeypatch
):
    """A blocker without a proposed surrogate keeps the halt without
    burning a dispatch — the re-ask can only ever clear the whole gate."""
    import run_pipeline

    state = make_state(run_dir)
    _seed_method_spec(state)

    def boom(_s, _b):
        raise AssertionError("must not dispatch")

    monkeypatch.setattr(
        run_pipeline, "_dispatch_analyzer_feasibility_reask", boom)

    original = _cannot_implement_halt(n=2, with_surrogate=False)
    ok, halt_record = run_pipeline._reask_feasibility_cannot_implement(
        state, "stage_1", original)
    assert ok is False
    assert halt_record is original


def test_feasibility_reask_dispatch_failure_restores_spec(
    run_dir, monkeypatch
):
    """Best-effort: a dispatch failure restores the snapshotted spec and
    falls through to the exact old halt."""
    import run_pipeline
    from opencode_client import OpencodeClientError

    state = make_state(run_dir)
    original_spec = _seed_method_spec(state)

    def failing_dispatch(_state, _blockers):
        _state.paths.method_spec.write_text("{corrupt", encoding="utf-8")
        raise OpencodeClientError("dispatch timed out")

    monkeypatch.setattr(
        run_pipeline, "_dispatch_analyzer_feasibility_reask", failing_dispatch)

    original = _cannot_implement_halt()
    ok, halt_record = run_pipeline._reask_feasibility_cannot_implement(
        state, "stage_1", original)

    assert ok is False
    assert halt_record is original
    assert state.paths.method_spec.read_text(encoding="utf-8") == original_spec


def test_feasibility_reask_invalid_spec_restores_and_halts(
    run_dir, monkeypatch
):
    """The re-asked spec failing strict validation restores the original
    and halts as before (never ships a half-edited spec)."""
    import run_pipeline

    state = make_state(run_dir)
    original_spec = _seed_method_spec(state)

    def fake_dispatch(_state, _blockers):
        _state.paths.method_spec.write_text(
            json.dumps({"schema_version": "1.5.0", "seed": "bad-edit"}),
            encoding="utf-8")

    monkeypatch.setattr(
        run_pipeline, "_dispatch_analyzer_feasibility_reask", fake_dispatch)
    monkeypatch.setattr(
        run_pipeline, "_run_fresh_spec_validator",
        lambda s: (False, "extra_forbidden: surprise field"))

    original = _cannot_implement_halt()
    ok, halt_record = run_pipeline._reask_feasibility_cannot_implement(
        state, "stage_1", original)

    assert ok is False
    assert halt_record is original
    assert state.paths.method_spec.read_text(encoding="utf-8") == original_spec


# ---------------------------------------------------------------------------
# Item 27 — monotone-progress bonus iteration (SRL overnight 07-08:
# 14 → 8 → 3 failing equation quotes, and the cap halted the run 3 short)
# ---------------------------------------------------------------------------


def _quote_check_stderr(n: int, salt: str = "") -> str:
    """A validate_paper_map-shaped stderr with a countable failing list and
    a per-count-distinct signature."""
    bullets = "\n".join(
        f"  - eq-{salt}{i}: source_text is not a verbatim passage of the paper"
        for i in range(n))
    return f"equation verbatim-quote check failed ({n} element(s)):\n{bullets}"


def _queue_bonus_iteration(fake_dispatch, *, iteration: int):
    """One judge-routed fix iteration: judge decides dispatch_fix to the
    decomposer, decomposer fix writes the map."""
    queue_judge_decision(
        fake_dispatch,
        stage_id="stage_1",
        iteration=iteration,
        validator_label="validate_paper_map.py",
        classification="producer_fixable",
        action="dispatch_fix",
        target_agent="r2c-decomposer",
        finding={"id": f"JUDGE{iteration}", "severity": "critical",
                 "description": "fix the next batch of quotes"},
    )
    fake_dispatch.expect(
        agent="r2c-decomposer",
        writes={".pipeline/paper_map.json": f'{{"iter": {iteration}}}'},
    )


def test_validator_retry_monotone_progress_earns_bonus_and_converges(
    fake_dispatch, run_dir,
):
    """The SRL trajectory shape: every iteration strictly reduces the
    failing count, so the cap halt is preceded by one judge-routed bonus
    iteration — which clears the last failures here."""
    import run_pipeline

    state = make_state(run_dir)
    outcomes = [(False, _quote_check_stderr(14)),
                (False, _quote_check_stderr(8)),
                (False, _quote_check_stderr(3)),
                (True, "")]

    def validator_fn(_state):
        return outcomes.pop(0)

    for i in range(3):
        _queue_bonus_iteration(fake_dispatch, iteration=i)

    result = run_pipeline._validator_retry(
        state, stage_id="stage_1",
        validator_fn=validator_fn,
        fix_dispatch_fn=run_pipeline._dispatch_decomposer_fix,
        validator_label="validate_paper_map.py",
        cap=2, use_judge=True,
    )
    assert result is None  # converged on the bonus iteration
    assert not outcomes  # all four validator outcomes consumed


def test_validator_retry_bonus_exhausted_halts_with_trajectory(
    fake_dispatch, run_dir,
):
    """When the blessed bonus iteration still fails, the halt reason names
    the bonus and the failing-count trajectory truthfully."""
    import run_pipeline

    state = make_state(run_dir)
    outcomes = [(False, _quote_check_stderr(14)),
                (False, _quote_check_stderr(8)),
                (False, _quote_check_stderr(3)),
                (False, _quote_check_stderr(1))]

    def validator_fn(_state):
        return outcomes.pop(0)

    for i in range(3):
        _queue_bonus_iteration(fake_dispatch, iteration=i)

    result = run_pipeline._validator_retry(
        state, stage_id="stage_1",
        validator_fn=validator_fn,
        fix_dispatch_fn=run_pipeline._dispatch_decomposer_fix,
        validator_label="validate_paper_map.py",
        cap=2, use_judge=True,
    )
    assert result is not None and result.status == "halted"
    reason = result.halt_artifact["reason"]
    assert "plus one bonus iteration granted for monotone progress" in reason
    assert "14 → 8 → 3 → 1" in reason
    assert result.halt_artifact["halt_class"] == "fix_loop_exhausted"


def test_validator_retry_no_bonus_on_plateaued_count(fake_dispatch, run_dir):
    """A changed signature with a NON-decreasing count (the 8 → 8 plateau)
    earns no bonus: the cap halt fires exactly as before."""
    import run_pipeline

    state = make_state(run_dir)
    outcomes = [(False, _quote_check_stderr(14)),
                (False, _quote_check_stderr(8, salt="a")),
                (False, _quote_check_stderr(8, salt="b"))]

    def validator_fn(_state):
        return outcomes.pop(0)

    for i in range(2):
        _queue_bonus_iteration(fake_dispatch, iteration=i)

    result = run_pipeline._validator_retry(
        state, stage_id="stage_1",
        validator_fn=validator_fn,
        fix_dispatch_fn=run_pipeline._dispatch_decomposer_fix,
        validator_label="validate_paper_map.py",
        cap=2, use_judge=True,
    )
    assert result is not None and result.status == "halted"
    reason = result.halt_artifact["reason"]
    assert "bonus" not in reason
    # Trajectory-driven wording (iDb-RRT 2026-07-13: the old categorical
    # "signature changed each iteration" claim was false for that run and
    # shipped into the researcher-facing report).
    assert "failing-count trajectory" in reason
    assert "did not converge" in reason


def test_validator_retry_no_write_iteration_does_not_poison_bonus(
    fake_dispatch, run_dir,
):
    """The iDb-RRT 2026-07-13 shape: a cap-burned fix dispatch writes
    nothing, the re-validation re-observes the identical error set, and the
    act-first retry then lands. The flat re-observation must NOT enter the
    failing-count trajectory — the burn-down here is 7 → 3 → 2 with one
    no-write iteration in the middle, and the monotone bonus must still be
    granted at the cap."""
    import run_pipeline

    state = make_state(run_dir)
    seven = _quote_check_stderr(7)
    outcomes = [(False, seven),
                (False, seven),  # identical: the fix wrote nothing
                (False, _quote_check_stderr(3)),
                (False, _quote_check_stderr(2)),
                (True, "")]

    def validator_fn(_state):
        return outcomes.pop(0)

    # Iteration 0: judge routes a fix that writes NOTHING (the cap burn).
    queue_judge_decision(
        fake_dispatch,
        stage_id="stage_1",
        iteration=0,
        validator_label="validate_paper_map.py",
        classification="producer_fixable",
        action="dispatch_fix",
        target_agent="r2c-decomposer",
        finding={"id": "JUDGE0", "severity": "critical",
                 "description": "fix the quotes"},
    )
    fake_dispatch.expect(agent="r2c-decomposer", writes={})
    # Iteration 1: identical signature + no write → act-first retry, no
    # judge involved, and this one lands.
    fake_dispatch.expect(
        agent="r2c-decomposer",
        writes={".pipeline/paper_map.json": '{"iter": "retry"}'},
    )
    # Iterations 2 and 3 (3 is the bonus at cap=3): judge-routed fixes.
    for i in (2, 3):
        _queue_bonus_iteration(fake_dispatch, iteration=i)

    result = run_pipeline._validator_retry(
        state, stage_id="stage_1",
        validator_fn=validator_fn,
        fix_dispatch_fn=run_pipeline._dispatch_decomposer_fix,
        validator_label="validate_paper_map.py",
        cap=3, use_judge=True,
    )
    assert result is None  # bonus granted, converged on it
    assert not outcomes


def test_stderr_excerpt_keeps_count_header_and_early_findings():
    """Head+tail excerpting: the "(N element(s))" header and the first
    failing items survive truncation (a plain tail slice hid 2 of 7 findings
    from the judge and corrupted the bonus counts, iDb-RRT 2026-07-13)."""
    from run_pipeline import _stderr_excerpt, _validator_failing_count

    stderr = _quote_check_stderr(7) * 1  # ~7 bullets, small
    assert _stderr_excerpt(stderr) == stderr  # under the cap: unchanged

    # Pad each bullet so the full output far exceeds the cap.
    long_stderr = "equation verbatim-quote check failed (7 element(s)):\n" + \
        "\n".join(f"  - eq-{i}: source_text is not verbatim " + "x" * 400
                  for i in range(7))
    excerpt = _stderr_excerpt(long_stderr)
    assert len(excerpt) <= 2000
    assert "(7 element(s))" in excerpt
    assert "eq-0" in excerpt          # earliest finding survives
    assert "eq-6" in excerpt          # freshest finding survives
    assert "elided" in excerpt        # the cut is marked, not silent
    assert _validator_failing_count(excerpt) == 7


def test_validator_failing_count_parses_headers_and_bullets():
    from run_pipeline import _validator_failing_count

    assert _validator_failing_count(_quote_check_stderr(14)) == 14
    # Header truncated away from the tail: bullets still count.
    assert _validator_failing_count(
        "  - eq-1: bad quote\n  - eq-2: bad quote\n") == 2
    assert _validator_failing_count("schema error, no list shape") is None
    assert _validator_failing_count("") is None


def test_validator_retry_plateau_to_one_earns_the_bonus(
    fake_dispatch, run_dir,
):
    """The pdfgnn 2026-08-11 attempt 5-7 shape: the failing count never
    grows and burns down to exactly one remaining error (3 -> 3 -> 2 -> 1).
    One judge-routed bonus iteration is granted, and here it converges."""
    import run_pipeline

    state = make_state(run_dir)
    outcomes = [(False, _quote_check_stderr(3)),
                (False, _quote_check_stderr(3, salt="b")),
                (False, _quote_check_stderr(2)),
                (False, _quote_check_stderr(1)),
                (True, "")]

    def validator_fn(_state):
        return outcomes.pop(0)

    for i in range(4):
        _queue_bonus_iteration(fake_dispatch, iteration=i)

    result = run_pipeline._validator_retry(
        state, stage_id="stage_1",
        validator_fn=validator_fn,
        fix_dispatch_fn=run_pipeline._dispatch_decomposer_fix,
        validator_label="validate_paper_map.py",
        cap=3, use_judge=True,
    )
    assert result is None  # converged on the bonus iteration
    assert not outcomes


def test_validator_retry_no_bonus_for_a_plateau_above_one(
    fake_dispatch, run_dir,
):
    """A trajectory that plateaus at a higher count (14 -> 8 -> 8) still
    earns nothing: the second qualifying shape requires burning down to
    exactly one remaining error."""
    import run_pipeline

    state = make_state(run_dir)
    outcomes = [(False, _quote_check_stderr(14)),
                (False, _quote_check_stderr(8, salt="a")),
                (False, _quote_check_stderr(8, salt="b"))]

    def validator_fn(_state):
        return outcomes.pop(0)

    for i in range(2):
        _queue_bonus_iteration(fake_dispatch, iteration=i)

    result = run_pipeline._validator_retry(
        state, stage_id="stage_1",
        validator_fn=validator_fn,
        fix_dispatch_fn=run_pipeline._dispatch_decomposer_fix,
        validator_label="validate_paper_map.py",
        cap=2, use_judge=True,
    )
    assert result is not None and result.status == "halted"
    assert "bonus" not in result.halt_artifact["reason"]
