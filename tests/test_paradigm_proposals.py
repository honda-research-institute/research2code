from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError


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


def _pack_payload(**overrides) -> dict:
    payload = {
        "schema_version": "1.0",
        "status": "provisional",
        "legacy_paradigm": "active_learning/geometric",
        "extends": "active_learning",
        "taxonomy_id": "TE-TS/active_learning/geometric",
        "fingerprint": {"what_it_is": "Geometric active-learning batch selection."},
        "scaffold_hints": {
            "interface_hint": "select_batch(model, x_unlabeled, batch_size, seed)",
        },
        "semantic_checks": [],
        "smoke_bugs": [],
        # R2C-032 authoring floors (approved 2026-07-28): a new-subparadigm
        # pack must declare its build-plan routing and its family-components
        # surface explicitly — silence is an authoring error, so the default
        # fixture carries both.
        "build_plan": {
            "source": "inherit_parent",
            "reasoning": "The parent active-learning manifest describes "
                         "this selector's build shape.",
        },
        "family_components": {},
    }
    payload.update(overrides)
    return payload


def _proposal_payload(**overrides) -> dict:
    payload = {
        "schema_version": "1.0.0",
        "proposal_id": "manual-proposal",
        "source_paper_slug": "demo-paper",
        "source_gap_report": ".pipeline/paradigm_gap_report.json",
        "decision": "new_subparadigm_needed",
        "target_paradigm_id": "active_learning/geometric",
        "target_taxonomy_id": "TE-TS/active_learning/geometric",
        "extends": "active_learning",
        "title": "Geometric Active Learning",
        "scope_summary": "Selects batches using a geometric objective.",
        "pack": _pack_payload(),
        "evidence": [
            {
                "paper_section": "method",
                "quote_or_observation": "The method optimizes geometric diversity.",
                "relevance": "Identifies the new child family.",
            }
        ],
        "coupling_warnings": [],
        "validation_command": "python3 scripts/validate_paradigm_proposal.py <proposal_dir>",
    }
    payload.update(overrides)
    return payload


def _write_subproposal(proposal_dir: Path, **proposal_overrides) -> Path:
    proposal_dir.mkdir(parents=True, exist_ok=True)
    payload = _proposal_payload(**proposal_overrides)
    pack = payload.get("pack") or _pack_payload()
    (proposal_dir / "proposal.json").write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )
    (proposal_dir / "pack.yaml").write_text(
        yaml.safe_dump(pack, sort_keys=False),
        encoding="utf-8",
    )
    return proposal_dir


def _write_gap_report(path: Path, **overrides) -> None:
    payload = {
        "schema_version": "1.0.0",
        "decision": "new_subparadigm_needed",
        "confidence": 0.86,
        "paper_slug": "demo-paper",
        "paper_title": "Geometric Batch Selection",
        "paper_paradigm_summary": "The paper proposes a geometric active-learning batch selector.",
        "matched_existing_paradigm": None,
        "proposed_parent_paradigm": "active_learning",
        "proposed_new_paradigm_id": "active_learning/geometric",
        "candidate_matches": [
            {
                "paradigm_id": "active_learning",
                "taxonomy_id": "TE-TS/active_learning",
                "fit": "Parent family fits.",
                "decision": "Needs a child node.",
            }
        ],
        "rejected_matches": [],
        "registered_paradigms": ["active_learning"],
        "paper_evidence": [
            {
                "paper_section": "method",
                "quote_or_observation": "The selector optimizes geometric diversity.",
                "relevance": "It fits active learning but needs a new child pack.",
            }
        ],
        "recommended_next_action": "Draft a candidate taxonomy pack.",
    }
    payload.update(overrides)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def test_gap_report_schema_enforces_decision_consistency_and_taxonomy_candidates():
    from schemas.paradigm_gap import ParadigmGapReport

    report = ParadigmGapReport.model_validate(
        {
            "decision": "new_subparadigm_needed",
            "confidence": 0.5,
            "paper_paradigm_summary": "summary",
            "proposed_parent_paradigm": "active_learning",
            "proposed_new_paradigm_id": "active_learning/geometric",
            "candidate_matches": [
                {
                    "paradigm_id": "active_learning",
                    "taxonomy_id": "TE-TS/active_learning",
                    "fit": "parent",
                    "decision": "child needed",
                }
            ],
            "paper_evidence": [
                {
                    "paper_section": "method",
                    "quote_or_observation": "observation",
                    "relevance": "relevance",
                }
            ],
            "recommended_next_action": "review",
        }
    )
    assert report.candidate_matches[0].taxonomy_id == "TE-TS/active_learning"

    with pytest.raises(ValidationError, match="Extra inputs"):
        ParadigmGapReport.model_validate(
            {
                "decision": "new_subparadigm_needed",
                "confidence": 0.5,
                "paper_paradigm_summary": "summary",
                "proposed_parent_paradigm": "active_learning",
                "proposed_new_paradigm_id": "active_learning/geometric",
                "candidate_matches": [
                    {
                        "paradigm_id": "active_learning",
                        "field_guide_path": "paradigms/active_learning/FIELD_GUIDE.md",
                        "fit": "parent",
                        "decision": "child needed",
                    }
                ],
                "paper_evidence": [
                    {
                        "paper_section": "method",
                        "quote_or_observation": "observation",
                        "relevance": "relevance",
                    }
                ],
                "recommended_next_action": "review",
            }
        )

    with pytest.raises(ValidationError, match="new sub-paradigm id"):
        ParadigmGapReport.model_validate(
            {
                "decision": "new_subparadigm_needed",
                "confidence": 0.5,
                "paper_paradigm_summary": "summary",
                "proposed_parent_paradigm": "active_learning",
                "proposed_new_paradigm_id": "knowledge_distillation/detection",
                "paper_evidence": [
                    {
                        "paper_section": "method",
                        "quote_or_observation": "observation",
                        "relevance": "relevance",
                    }
                ],
                "recommended_next_action": "review",
            }
        )


def _gap_report_payload(**overrides):
    payload = {
        "decision": "new_subparadigm_needed",
        "confidence": 0.9,
        "paper_paradigm_summary": "summary",
        "proposed_parent_paradigm": "active_learning",
        "proposed_new_paradigm_id": "active_learning/geometric",
        "paper_evidence": [
            {
                "paper_section": "method",
                "quote_or_observation": "observation",
                "relevance": "relevance",
            }
        ],
        "recommended_next_action": "review",
    }
    payload.update(overrides)
    return payload


def test_gap_report_rejects_taxonomy_vocabulary_with_a_named_hint():
    """The 2026-07-06 fedavg gap report, verbatim id values: the analyzer
    proposed in the canonical taxonomy vocabulary (uppercase family code,
    plus a prose gloss on the parent) and the run halted with a message
    that named the rule but not the value or the fix. The error must now
    carry the offending value, an example of the right form, and — for
    uppercase inputs — the taxonomy-vocabulary hint, because this text is
    what the stage-1 halt artifact shows a researcher."""
    from schemas.paradigm_gap import ParadigmGapReport

    with pytest.raises(ValidationError) as exc_info:
        ParadigmGapReport.model_validate(_gap_report_payload(
            proposed_parent_paradigm="TE-TS (Training Strategy family)",
            proposed_new_paradigm_id="TE-TS/federated_learning",
        ))
    message = str(exc_info.value)
    assert "'TE-TS/federated_learning'" in message
    assert "active_learning/bayesian" in message  # the right-form example
    assert "taxonomy id" in message               # the vocabulary hint
    assert "registered_paradigms" in message      # where to look instead

    # The hint is uppercase-specific: a plain malformed lowercase value
    # gets the form error without the taxonomy-vocabulary diagnosis.
    with pytest.raises(ValidationError) as exc_info:
        ParadigmGapReport.model_validate(_gap_report_payload(
            proposed_parent_paradigm="active learning",
            proposed_new_paradigm_id="active learning/geometric",
        ))
    message = str(exc_info.value)
    assert "'active learning'" in message
    assert "taxonomy id" not in message

    # Positive control: the fedavg proposal expressed in the legal
    # vocabulary (a new top level, family placement deferred to prose)
    # validates clean.
    report = ParadigmGapReport.model_validate(_gap_report_payload(
        decision="new_top_level_needed",
        proposed_parent_paradigm=None,
        proposed_new_paradigm_id="federated_learning",
        recommended_next_action="author a provisional pack; the natural "
                                "taxonomy placement is the training-strategy "
                                "family, decided at promotion time",
    ))
    assert report.proposed_new_paradigm_id == "federated_learning"


def test_proposal_schema_enforces_parent_and_top_level_shape():
    from schemas.paradigm_proposal import ParadigmProposal

    with pytest.raises(ValidationError, match="below extends"):
        ParadigmProposal.model_validate(
            _proposal_payload(target_paradigm_id="knowledge_distillation/detection")
        )

    with pytest.raises(ValidationError, match="top-level"):
        ParadigmProposal.model_validate(
            _proposal_payload(
                decision="new_top_level_needed",
                target_paradigm_id="control/barrier",
                target_taxonomy_id="PROVISIONAL/control/barrier",
                extends=None,
            )
        )


def test_propose_field_guide_scaffolds_pack_proposal_without_promoting(tmp_path):
    from scripts.propose_field_guide import create_proposal_packet

    repo_root = tmp_path / "repo"
    _write_taxonomy(repo_root)
    run_dir = repo_root / "r2c_runs" / "demo-paper"
    gap_path = run_dir / ".pipeline" / "paradigm_gap_report.json"
    _write_gap_report(gap_path)

    proposal_dir = create_proposal_packet(
        run_dir,
        repo_root,
        proposal_id="manual-proposal",
    )

    proposal = json.loads((proposal_dir / "proposal.json").read_text(encoding="utf-8"))
    pack = yaml.safe_load((proposal_dir / "pack.yaml").read_text(encoding="utf-8"))
    assert proposal["target_paradigm_id"] == "active_learning/geometric"
    assert proposal["target_taxonomy_id"] == "TE-TS/active_learning/geometric"
    assert proposal["extends"] == "active_learning"
    assert pack["legacy_paradigm"] == "active_learning/geometric"
    assert pack["taxonomy_id"] == "TE-TS/active_learning/geometric"
    assert pack["scaffold_hints"]["interface_hint"] == (
        "select_batch(model, x_unlabeled, batch_size, seed)"
    )
    assert (proposal_dir / "coupling_report.md").is_file()
    assert not (repo_root / "paradigms").exists()


def test_validate_subproposal_checks_pack_shape_and_parent(tmp_path):
    from scripts.validate_paradigm_proposal import validate_proposal

    repo_root = tmp_path / "repo"
    _write_taxonomy(repo_root)
    proposal_dir = _write_subproposal(tmp_path / "proposal")

    result = validate_proposal(proposal_dir, repo_root)

    assert result.valid
    assert result.errors == []


def test_validate_proposal_rejects_missing_pack_taxonomy_id(tmp_path):
    from scripts.validate_paradigm_proposal import validate_proposal

    repo_root = tmp_path / "repo"
    _write_taxonomy(repo_root)
    proposal_dir = _write_subproposal(
        tmp_path / "proposal",
        pack=_pack_payload(taxonomy_id=""),
    )

    result = validate_proposal(proposal_dir, repo_root)

    assert not result.valid
    assert any("pack.yaml taxonomy_id missing" in error for error in result.errors)


def test_apply_proposal_writes_reviewable_proposed_pack(tmp_path):
    from scripts.apply_paradigm_proposal import apply_proposal

    repo_root = tmp_path / "repo"
    _write_taxonomy(repo_root)
    proposal_dir = _write_subproposal(tmp_path / "proposal")

    written = apply_proposal(proposal_dir, repo_root)

    target = repo_root / "docs" / "ssot" / "proposed_packs" / "manual-proposal.yaml"
    assert written == ["docs/ssot/proposed_packs/manual-proposal.yaml"]
    assert target.is_file()
    payload = yaml.safe_load(target.read_text(encoding="utf-8"))
    assert payload["proposal"]["target_paradigm_id"] == "active_learning/geometric"
    assert payload["proposal"]["target_taxonomy_id"] == "TE-TS/active_learning/geometric"
    assert payload["pack"]["legacy_paradigm"] == "active_learning/geometric"
    assert not (repo_root / "paradigms").exists()


def test_validate_proposal_rejects_docstring_bearing_interface_hint(tmp_path):
    """The DomIndOnto 2026-07-21 roll-1 pack shipped its interface hint as a
    signature PLUS a full triple-quoted docstring. The scaffolder interpolates
    the hint inside the generated module's docstring, so the embedded quotes
    closed it early and stage 2a halted on invalid syntax — with no LLM in the
    loop to fix it. The authoring floor must hard-fail such a hint so the
    proposal retry loop iterates while an author is still on the hook."""
    from scripts.validate_paradigm_proposal import validate_proposal

    from tests.test_provisional_templates import DOMINDONTO_0721_BAD_HINT

    repo_root = tmp_path / "repo"
    _write_taxonomy(repo_root)
    proposal_dir = _write_subproposal(
        tmp_path / "proposal",
        pack=_pack_payload(scaffold_hints={
            "interface_hint": DOMINDONTO_0721_BAD_HINT,
        }),
    )

    result = validate_proposal(proposal_dir, repo_root)

    assert not result.valid
    assert any("interface_hint" in e and "docstring terminator" in e
               for e in result.errors)

    # Same floor for the single-quote spelling.
    proposal_dir = _write_subproposal(
        tmp_path / "proposal-single-quotes",
        pack=_pack_payload(scaffold_hints={
            "interface_hint": "select_batch(model, seed)\n'''pick a batch'''",
        }),
    )
    result = validate_proposal(proposal_dir, repo_root)
    assert not result.valid
    assert any("interface_hint" in e and "docstring terminator" in e
               for e in result.errors)


def test_validate_proposal_accepts_multiline_signature_only_hint(tmp_path):
    """Positive control for the docstring-terminator floor: a multi-line
    signature WITHOUT a docstring is a legal hint shape (the default payload
    already covers the one-line shape)."""
    from scripts.validate_paradigm_proposal import validate_proposal

    repo_root = tmp_path / "repo"
    _write_taxonomy(repo_root)
    proposal_dir = _write_subproposal(
        tmp_path / "proposal",
        pack=_pack_payload(scaffold_hints={
            "interface_hint": (
                "select_batch(\n"
                "    model,\n"
                "    x_unlabeled,\n"
                "    batch_size,\n"
                "    seed=None,\n"
                ") -> SelectedIndices"
            ),
        }),
    )

    result = validate_proposal(proposal_dir, repo_root)

    assert result.valid
    assert result.errors == []


# ---------------------------------------------------------------------------
# R2C-032 authoring floors (approved 2026-07-28): a new-subparadigm pack must
# declare `build_plan.source` (the build-context routing choice the SRL
# 2026-07-28 stage-2b halt shows was silently inherited) and the
# `family_components` key (explicit empty mapping allowed).
# ---------------------------------------------------------------------------


def test_validate_subproposal_requires_build_plan_choice(tmp_path):
    from scripts.validate_paradigm_proposal import validate_proposal

    repo_root = tmp_path / "repo"
    _write_taxonomy(repo_root)

    # Known-bad: the pack omits build_plan entirely (family_components stays
    # the explicit empty mapping, isolating the build_plan error).
    pack = _pack_payload()
    del pack["build_plan"]
    proposal_dir = _write_subproposal(tmp_path / "proposal", pack=pack)
    result = validate_proposal(proposal_dir, repo_root)
    assert not result.valid
    [err] = [e for e in result.errors if "build_plan" in e]
    assert "build_plan.source" in err
    assert "inherit_parent" in err and "neutral" in err

    # Known-good: a declared neutral choice validates (missing reasoning is
    # a warning, never a blocker).
    proposal_dir = _write_subproposal(
        tmp_path / "proposal-neutral",
        pack=_pack_payload(build_plan={"source": "neutral"}),
    )
    result = validate_proposal(proposal_dir, repo_root)
    assert result.valid, result.errors
    assert any("no reasoning" in w for w in result.warnings)


def test_validate_subproposal_requires_family_components_key(tmp_path):
    from scripts.validate_paradigm_proposal import validate_proposal

    repo_root = tmp_path / "repo"
    _write_taxonomy(repo_root)

    # Known-bad: the key is missing entirely. The error names the SRL
    # failure shape and the explicit-empty escape hatch.
    pack = _pack_payload()
    del pack["family_components"]
    proposal_dir = _write_subproposal(tmp_path / "proposal", pack=pack)
    result = validate_proposal(proposal_dir, repo_root)
    assert not result.valid
    [err] = [e for e in result.errors if "family_components" in e]
    assert "family_components: {}" in err
    assert "stage 2.b" in err

    # Known-good: the explicit empty mapping (the default fixture) passes.
    proposal_dir = _write_subproposal(tmp_path / "proposal-empty",
                                      pack=_pack_payload())
    result = validate_proposal(proposal_dir, repo_root)
    assert result.valid, result.errors


def test_build_plan_choice_vocabulary_and_shape(tmp_path):
    from scripts.validate_paradigm_proposal import (
        _check_build_plan,
        validate_proposal,
    )

    repo_root = tmp_path / "repo"
    _write_taxonomy(repo_root)

    def _errors(workdir, pack, **proposal_overrides):
        proposal_dir = _write_subproposal(tmp_path / workdir, pack=pack,
                                          **proposal_overrides)
        return validate_proposal(proposal_dir, repo_root).errors

    # Bad vocabulary.
    errs = _errors("p-vocab", _pack_payload(build_plan={"source": "parent"}))
    assert any("build_plan invalid" in e for e in errs), errs

    # Typo'd subkey (extra='forbid' on the choice model).
    errs = _errors("p-typo", _pack_payload(
        build_plan={"source": "neutral", "resoning": "typo"}))
    assert any("build_plan invalid" in e for e in errs), errs

    # Not a mapping.
    errs = _errors("p-list", _pack_payload(build_plan=["neutral"]))
    assert any("must be a mapping" in e for e in errs), errs

    # inherit_parent on a top-level pack: no parent plan to inherit.
    top_level = dict(
        decision="new_top_level_needed",
        target_paradigm_id="federated_learning",
        target_taxonomy_id="PROVISIONAL/federated_learning",
        extends=None,
    )
    errs = _errors(
        "p-top-inherit",
        _pack_payload(legacy_paradigm="federated_learning", extends=None,
                      taxonomy_id="PROVISIONAL/federated_learning",
                      build_plan={"source": "inherit_parent"}),
        **top_level,
    )
    assert any("extends nothing" in e for e in errs), errs

    # A top-level pack may omit build_plan entirely: with no parent there is
    # nothing to inherit and the runtime default is already the neutral plan.
    top_pack = _pack_payload(legacy_paradigm="federated_learning",
                             extends=None,
                             taxonomy_id="PROVISIONAL/federated_learning")
    del top_pack["build_plan"]
    proposal_dir = _write_subproposal(tmp_path / "p-top-silent",
                                      pack=top_pack, **top_level)
    result = validate_proposal(proposal_dir, repo_root)
    assert result.valid, result.errors

    # inherit_parent on a sub-paradigm whose ancestry carries no static
    # plan: the declaration would silently degrade to the neutral plan at
    # runtime, so the author must say `neutral` and take the cap honestly.
    # (Unit-level: the check needs only the pack — no taxonomy graft.)
    errors: list[str] = []
    _check_build_plan(
        {"legacy_paradigm": "widget_family/child", "extends": "widget_family",
         "build_plan": {"source": "inherit_parent", "reasoning": "r"}},
        errors, [], decision="new_subparadigm_needed")
    assert any("silently degrade" in e for e in errors), errors


def test_validate_proposal_rejects_placeholder_contract_values(tmp_path):
    """The fedavg 2026-07-06 pack shipped a literal template TODO as its
    interface hint — non-empty, so the old emptiness floor passed it, the
    overlay read it as a carried contract, and the run died three stages
    later at the strict spec cross-check. Placeholders must fail the
    authoring floor exactly like missing values so the authoring loop
    iterates until the agent writes the real interface."""
    from scripts.validate_paradigm_proposal import validate_proposal

    repo_root = tmp_path / "repo"
    _write_taxonomy(repo_root)
    proposal_dir = _write_subproposal(
        tmp_path / "proposal",
        pack=_pack_payload(scaffold_hints={
            # Verbatim from the live fedavg pack.
            "interface_hint": "TODO: describe the pluggable interface "
                              "this paper needs.",
        }),
    )

    result = validate_proposal(proposal_dir, repo_root)

    assert not result.valid
    assert any("interface_hint" in e and "placeholder" in e
               for e in result.errors)


def test_validate_proposal_refuses_retired_arch_contract_schema(tmp_path):
    """R2C-049. The 2026-08-03 probabilistic-demand-forecasting pack declared
    an `arch_contract_schema` block requiring `data_loader.{path, columns,
    split_config}` and `pluggable_component.{signature, return_type}`. The
    universal ArchContract skeleton declares none of those and is
    `extra='forbid'` on every model, so the architecture coder could satisfy
    the pack or pydantic and never both. Stage 2b degraded with 12
    `extra_forbidden` errors. Refuse it where it is authored, while the
    authoring retry loop can still act."""
    from scripts.validate_paradigm_proposal import validate_proposal

    repo_root = tmp_path / "repo"
    _write_taxonomy(repo_root)
    proposal_dir = _write_subproposal(
        tmp_path / "proposal",
        pack=_pack_payload(scaffold_hints={
            "interface_hint": "select_batch(model, x_unlabeled, batch_size, seed)",
            # The real block, verbatim in shape, from that run's pack.
            "arch_contract_schema": {
                "data_loader": {
                    "required": True,
                    "required_entries": ["path", "columns", "split_config"],
                },
                "pluggable_component": {
                    "required": True,
                    "required_entries": ["name", "signature", "return_type"],
                },
            },
        }),
    )

    result = validate_proposal(proposal_dir, repo_root)

    assert not result.valid
    offending = [e for e in result.errors if "arch_contract_schema" in e]
    assert offending, result.errors
    # The refusal must route the author to the surviving surfaces, otherwise a
    # gap run has nowhere to go.
    assert any("RETIRED" in e for e in offending)
    assert any("family_components" in e for e in offending)
    assert any("params_derivation" in e for e in offending)


def test_validate_proposal_accepts_a_pack_without_the_retired_block(tmp_path):
    """Positive control: the SRL-shaped pack (real family_components, no
    retired block) still installs untouched."""
    from scripts.validate_paradigm_proposal import validate_proposal

    repo_root = tmp_path / "repo"
    _write_taxonomy(repo_root)
    proposal_dir = _write_subproposal(
        tmp_path / "proposal",
        pack=_pack_payload(family_components={
            "reward_function": {
                "required": True,
                "description": "The paper's reward shaping term.",
                "required_entries": ["shape", "components"],
            },
        }),
    )

    result = validate_proposal(proposal_dir, repo_root)

    assert result.valid, result.errors
    assert not any("arch_contract_schema" in e for e in result.errors)


def test_retired_block_refusal_ignores_packs_with_no_scaffold_hints_mapping():
    """A malformed scaffold_hints must not crash the new check; the existing
    interface_hint floor is what reports that shape."""
    from scripts.validate_paradigm_proposal import (
        _check_retired_arch_contract_schema,
    )

    errors: list[str] = []
    _check_retired_arch_contract_schema({"scaffold_hints": "not-a-mapping"}, errors)
    _check_retired_arch_contract_schema({}, errors)
    assert errors == []
