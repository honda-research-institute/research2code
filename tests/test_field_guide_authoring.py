from __future__ import annotations

import json
from pathlib import Path

import yaml

from scripts.opencode_client import DispatchResult


ROOT = Path(__file__).resolve().parent.parent


def _write_taxonomy(repo_root: Path) -> None:
    from scripts import taxonomy

    taxonomy.load_taxonomy.cache_clear()
    path = repo_root / "docs" / "ssot" / "taxonomies.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(
            {
                "schema_version": "2.0",
                "universal_checks": [],
                "method_roots": {
                    "TE": {
                        "name": "Technique",
                        "archetype": "method",
                        "families": {
                            "TS": {
                                "name": "Training Strategy",
                                "description": "Methods that change training/data flow.",
                                "variants": {
                                    "active_learning": {
                                        "name": "Active Learning",
                                        "status": "populated",
                                        "legacy_paradigm": "active_learning",
                                        "taxonomy_id": "TE-TS/active_learning",
                                        "fingerprint": {
                                            "what_it_is": "Iteratively selects examples to label.",
                                        },
                                        "scaffold_hints": {
                                            "interface_hint": "select_batch(model, x_unlabeled, batch_size, seed)",
                                        },
                                        "pluggable_component": {
                                            "name": "select_batch",
                                            "signature_template": (
                                                "select_batch(model, x_unlabeled, batch_size, seed) -> list[int]"
                                            ),
                                            "contract": {"seed_param": "seed"},
                                        },
                                    }
                                },
                            }
                        },
                    }
                },
                "task_domains": {},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )


def _write_gap_report(
    run_dir: Path,
    *,
    decision: str,
    proposed_id: str,
    parent: str | None = None,
) -> None:
    pipeline = run_dir / ".pipeline"
    pipeline.mkdir(parents=True, exist_ok=True)
    (pipeline / "paper.md").write_text("# Demo Paper\n\nMethod details.\n", encoding="utf-8")
    (pipeline / "paper_map.json").write_text("{}\n", encoding="utf-8")
    payload = {
        "schema_version": "1.0.0",
        "decision": decision,
        "confidence": 0.9,
        "paper_slug": "demo-paper",
        "paper_title": "Demo Paper",
        "paper_paradigm_summary": "The paper proposes a new method family.",
        "matched_existing_paradigm": None,
        "proposed_parent_paradigm": parent,
        "proposed_new_paradigm_id": proposed_id,
        "candidate_matches": [],
        "rejected_matches": [],
        "registered_paradigms": [parent] if parent else [],
        "paper_evidence": [
            {
                "paper_section": "method",
                "quote_or_observation": "The method defines a new procedure.",
                "relevance": "Supports the proposed paradigm.",
            }
        ],
        "recommended_next_action": "Draft a candidate taxonomy pack.",
    }
    (pipeline / "paradigm_gap_report.json").write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )


def _complete_pack(paradigm_id: str, taxonomy_id: str, extends: str | None) -> str:
    return yaml.safe_dump(
        {
            "schema_version": "1.0",
            "status": "provisional",
            "legacy_paradigm": paradigm_id,
            "extends": extends,
            "taxonomy_id": taxonomy_id,
            "fingerprint": {"what_it_is": f"{paradigm_id} methods."},
            "scaffold_hints": {
                "interface_hint": "run_method(data, seed) -> dict",
            },
            "semantic_checks": [],
            "smoke_bugs": [],
        },
        sort_keys=False,
    )


def test_author_prompt_names_scope_inputs_and_pack_targets(tmp_path):
    from scripts.author_field_guide_proposal import (
        build_author_prompt,
        resolve_or_create_proposal_dir,
    )

    repo_root = tmp_path / "repo"
    _write_taxonomy(repo_root)
    run_dir = repo_root / "r2c_runs" / "demo-paper"
    _write_gap_report(
        run_dir,
        decision="new_top_level_needed",
        proposed_id="control_theory",
    )
    proposal_dir = resolve_or_create_proposal_dir(run_dir, repo_root)

    prompt = build_author_prompt(
        proposal_dir=proposal_dir,
        run_dir=run_dir,
        repo_root=repo_root,
    )

    assert "USE YOUR WRITE TOOL" in prompt
    assert "Forbidden write targets" in prompt
    assert "paradigms/**" in prompt
    assert "pack.yaml" in prompt
    assert "FIELD_GUIDE.md" not in prompt
    assert "templates/README.md.template" not in prompt
    assert "Do not promote the proposal" in prompt


def test_authoring_stops_when_inherited_subproposal_already_valid(tmp_path):
    from scripts.author_field_guide_proposal import author_proposal
    from scripts.propose_field_guide import create_proposal_packet

    repo_root = tmp_path / "repo"
    _write_taxonomy(repo_root)
    run_dir = repo_root / "r2c_runs" / "demo-paper"
    _write_gap_report(
        run_dir,
        decision="new_subparadigm_needed",
        proposed_id="active_learning/geometric",
        parent="active_learning",
    )

    # A bare inherited scaffold is deliberately NOT valid anymore: the
    # R2C-032 authoring floors demand the author's explicit build_plan
    # routing choice and family_components declaration (the 2026-07-28 SRL
    # stage-2b halt). This test pins the no-dispatch contract for a
    # proposal that already carries both.
    proposal_dir = create_proposal_packet(run_dir, repo_root)
    pack_path = proposal_dir / "pack.yaml"
    pack = yaml.safe_load(pack_path.read_text(encoding="utf-8"))
    pack["build_plan"] = {
        "source": "inherit_parent",
        "reasoning": "the parent active-learning manifest fits",
    }
    pack["family_components"] = {}
    pack_path.write_text(yaml.safe_dump(pack, sort_keys=False),
                         encoding="utf-8")

    def fail_dispatch(prompt: str) -> DispatchResult:
        raise AssertionError("valid inherited proposal should not dispatch")

    result = author_proposal(
        run_dir=run_dir,
        repo_root=repo_root,
        dispatch_fn=fail_dispatch,
    )

    assert result.valid
    assert result.iterations == 0
    assert result.authoring_log_path.is_file()


def test_authoring_dispatches_until_validation_passes(tmp_path):
    from scripts.author_field_guide_proposal import (
        author_proposal,
        resolve_or_create_proposal_dir,
    )

    repo_root = tmp_path / "repo"
    _write_taxonomy(repo_root)
    run_dir = repo_root / "r2c_runs" / "demo-paper"
    _write_gap_report(
        run_dir,
        decision="new_top_level_needed",
        proposed_id="control_theory",
    )
    proposal_dir = resolve_or_create_proposal_dir(run_dir, repo_root)
    pack_path = proposal_dir / "pack.yaml"
    pack = yaml.safe_load(pack_path.read_text(encoding="utf-8"))
    pack["taxonomy_id"] = ""
    pack_path.write_text(yaml.safe_dump(pack, sort_keys=False), encoding="utf-8")

    calls: list[str] = []

    def fake_dispatch(prompt: str) -> DispatchResult:
        calls.append(prompt)
        pack_path.write_text(
            _complete_pack("control_theory", "PROVISIONAL/control_theory", None),
            encoding="utf-8",
        )
        return DispatchResult(
            session_id="ses_test",
            user_message_id="user_msg",
            assistant_message_id="assistant_msg",
            completed=True,
            elapsed_s=0.1,
        )

    result = author_proposal(
        run_dir=run_dir,
        repo_root=repo_root,
        proposal_dir=proposal_dir,
        dispatch_fn=fake_dispatch,
    )

    assert result.valid
    assert result.iterations == 1
    assert len(calls) == 1
    report = json.loads(result.validation_report_path.read_text(encoding="utf-8"))
    assert report["valid"] is True
    log_text = result.authoring_log_path.read_text(encoding="utf-8")
    assert "Validation After Iteration 1" in log_text


def test_r2c_paradigms_slash_command_invokes_plugin_tool():
    # Launch-path migration (2026-06-11): the command is a thin pointer at
    # the r2c_paradigms plugin tool — the detached spawn is deterministic
    # plugin code now, not a model-followed bash block. The pin keeps the
    # foreground-deadlock warning and the manual-promotion contract, and
    # keeps the obsolete bash plumbing OUT.
    command = (ROOT / ".opencode" / "command" / "r2c-paradigms.md").read_text(
        encoding="utf-8"
    )

    assert "`r2c_paradigms` tool" in command
    assert "Do NOT run `scripts/author_field_guide_proposal.py` in the foreground" in command
    assert "scripts/apply_paradigm_proposal.py <proposal_dir>" in command
    assert "nohup" not in command
    assert "--port" not in command


def test_r2c_run_slash_command_invokes_plugin_tool():
    command = (ROOT / ".opencode" / "command" / "r2c-run.md").read_text(
        encoding="utf-8"
    )

    assert "`r2c_run` tool" in command
    assert "scripts/run_pipeline.py` in the foreground" in command
    assert "watch command" in command
    assert "side panel is convenient but not authoritative" in command
    assert "nohup" not in command
    assert "R2C_PORT" not in command
    # Item 29 phase 2 regression (2026-07-09): `--fresh` reached the model only
    # if the command interpolates ALL arguments. The original `paper="$1"` form
    # captured just the first token, so `--fresh` (as $2) was silently dropped
    # and the model reported "no --fresh flag was mentioned". Pin the full-args
    # placeholder and the flag-parsing instruction so it cannot regress.
    assert "$ARGUMENTS" in command
    assert 'paper="$1"' not in command   # the token-losing form
    assert "--fresh" in command
    assert "fresh=true" in command


def test_r2c_plugin_carries_the_launch_contract():
    # The plugin owns what r2c-start.sh used to do: the address handoff,
    # the loopback proxy exemption, and detached file-logged spawns (the
    # deadlock guard's sanctioned shape). Pinned so a refactor can't
    # silently drop a job.
    plugin = (ROOT / ".opencode" / "plugin" / "r2c.ts").read_text(
        encoding="utf-8"
    )

    assert "R2C_SERVER_URL" in plugin
    assert "R2C_PORT" in plugin                 # legacy strangler channel
    assert "no_proxy" in plugin
    assert "detached: true" in plugin
    assert "--server-url" in plugin
    assert "unref()" in plugin


def test_scope_guard_exempts_driver_managed_artifacts_in_repo_relative_form():
    """The 2026-07-06 fedavg fresh run, verbatim violation shape: the
    authoring was killed for "writing" r2c_runs/fedavg/.pipeline/
    progress.json and run_events.jsonl — files the DRIVER appends during
    any dispatch. The exemption matched run-relative paths only (never
    the repo-relative keys the snapshot actually uses) and missed
    progress.json entirely. Both forms must exempt; genuinely
    out-of-scope writes must still flag."""
    from scripts.author_field_guide_proposal import (
        FileState,
        _is_driver_managed_run_artifact,
        _scope_violations,
    )

    # The live repo-relative shape plus the historical run-relative one.
    for rel in (
        "r2c_runs/fedavg/.pipeline/run_events.jsonl",
        "r2c_runs/fedavg/.pipeline/progress.json",
        "r2c_runs/fedavg/.pipeline/driver_state.json",
        "r2c_runs/fedavg/.pipeline/run_events.jsonl.lock",
        "r2c_runs/fedavg/.pipeline/_lock/owner.json",
        ".pipeline/run_events.jsonl",
        ".pipeline/progress.json",
    ):
        assert _is_driver_managed_run_artifact(rel), rel

    for rel in (
        "scripts/run_pipeline.py",
        "r2c_runs/fedavg/.pipeline/paper_map.json",
        "r2c_runs/fedavg/METHOD.md",
        "docs/ssot/taxonomies.yaml",
    ):
        assert not _is_driver_managed_run_artifact(rel), rel

    before = {
        "r2c_runs/fedavg/.pipeline/run_events.jsonl": FileState(1, 1),
        "r2c_runs/fedavg/.pipeline/progress.json": FileState(1, 1),
        "r2c_runs/fedavg/.pipeline/paper_map.json": FileState(1, 1),
    }
    after = {
        "r2c_runs/fedavg/.pipeline/run_events.jsonl": FileState(2, 2),
        "r2c_runs/fedavg/.pipeline/progress.json": FileState(2, 2),
        "r2c_runs/fedavg/.pipeline/paper_map.json": FileState(1, 1),
    }
    assert _scope_violations(before, after) == []
    # A real out-of-scope edit still flags.
    after["r2c_runs/fedavg/.pipeline/paper_map.json"] = FileState(3, 3)
    assert _scope_violations(before, after) == [
        "r2c_runs/fedavg/.pipeline/paper_map.json"]
