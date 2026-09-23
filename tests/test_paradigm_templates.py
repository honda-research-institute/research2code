"""Regression test: every taxonomy-served paradigm with a `templates/` dir must
produce output that satisfies its own `package_manifest`.

The 2026-05-26 pdwa stage 2.a halt traced to this bug: I authored the
motion_planning build-plan manifest and its data.py template in the
same session, and they drifted (manifest declared `load_environment(name:
str = 'two_rooms', ...)` while the template emitted `name: str =
'two_rooms_simple'`). The validator caught it at the first paper run
that exercised motion_planning, but the regression would have been
prevented if we'd sweep-tested templates against manifests as part of
the test suite.

This test runs `scaffold_package.scaffold()` + `validate_scaffolder_output.validate()`
on every paradigm with a templates/ dir, against a minimal synthetic
spec. Any drift between what the template renders and what the
manifest declares fails here at CI time, not at the next paper's stage 2.a.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent

from scripts import taxonomy
from scripts.scaffold_package import scaffold
from scripts.validate_scaffolder_output import validate


def _paradigm_ids_with_templates() -> list[str]:
    """Taxonomy-served paradigm ids that resolve a templates/ dir.

    Sub-paradigms that inherit templates from a parent are still listed when the
    effective taxonomy node resolves a template directory; the generated package
    is validated against that node's build plan.
    """
    tax = taxonomy.load_taxonomy(ROOT)
    ids: list[str] = []
    for node in tax._node_by_legacy.values():
        paradigm_id = getattr(node, "legacy_paradigm", None)
        if not paradigm_id:
            continue
        if taxonomy.serves(paradigm_id, tax) is None:
            continue
        if taxonomy.resolve_templates_dir(paradigm_id=paradigm_id, repo_root=ROOT):
            ids.append(paradigm_id)
    return sorted(set(ids))


def _min_spec(paradigm_id: str) -> dict:
    """Smallest spec that satisfies scaffold + validate_scaffolder_output.
    The scaffolder reads classification id, `paper.{title, authors}`, and
    `core_method.{name, summary}`. The validator reads classification id and
    dispatches through the taxonomy build plan."""
    return {
        "schema_version": "1.2.0",
        "paper": {"title": "Sweep Test", "authors": "test", "repo_url": None},
        "core_method": {
            "name": "test",
            "summary": "test",
            "type": "algorithm",
            "paper_sections": [],
            "key_elements": [],
        },
        "comparison": {
            "classification": {
                "id": paradigm_id,
                "detection_reasoning": "test sweep",
            },
        },
    }


@pytest.mark.parametrize(
    "paradigm_id",
    _paradigm_ids_with_templates(),
)
def test_scaffolder_output_matches_manifest(paradigm_id: str, tmp_path: Path):
    """For each paradigm with templates, render the templates and confirm
    the rendered files satisfy the paradigm's own `package_manifest`.

    Catches manifest/template drift at CI time. If this test fails, the
    paradigm's taxonomy build-plan manifest and its templates have disagreed on a
    signature, file name, or symbol kind — fix whichever side is wrong
    (template usually wins for signatures because it's the actual code;
    manifest wins for structure because it's the contract)."""
    run_dir = tmp_path / "run"
    (run_dir / ".pipeline").mkdir(parents=True)
    spec_path = run_dir / ".pipeline" / "method_spec.json"
    spec = _min_spec(paradigm_id)
    spec_path.write_text(json.dumps(spec, indent=2))

    rc = scaffold(spec_path, run_dir, ROOT)
    assert rc == 0, (
        f"scaffold returned {rc} for {paradigm_id}; "
        f"check the taxonomy templates dir and slot references."
    )

    errors = validate(spec, run_dir, ROOT)
    assert not errors, (
        f"Manifest/template drift in {paradigm_id}:\n"
        + "\n".join(f"  - {e}" for e in errors)
    )


@pytest.mark.parametrize(
    "paradigm_id",
    _paradigm_ids_with_templates(),
)
def test_scaffolded_code_has_no_orphaned_placeholder(paradigm_id: str, tmp_path: Path):
    """Scaffolder-owned .py files must not ship a `raise NotImplementedError`
    placeholder. Such a placeholder could only be filled by a producer that can
    WRITE the file — but scaffolder-owned files (`kind: paradigm_fixed`,
    `produced_by: package_scaffolder`) are outside every producer's writeable
    paths, so the placeholder is orphaned: nothing fills it, and the notebook
    crashes at runtime the first time it hits the unfilled path.

    This is the 2026-05-27 pdwa stage 3.c halt: motion_planning's data.py
    template had `_build_dynamics()` raising NotImplementedError with a comment
    saying 'the method-coder fills this in' — but data.py is scaffolder-owned and
    the method-coder can only write method.py. It surfaced only at smoke runtime,
    on the first motion_planning run ever to reach the smoke gate. A static gate
    here catches the whole class at CI time."""
    run_dir = tmp_path / "run"
    (run_dir / ".pipeline").mkdir(parents=True)
    spec_path = run_dir / ".pipeline" / "method_spec.json"
    spec_path.write_text(json.dumps(_min_spec(paradigm_id), indent=2))

    rc = scaffold(spec_path, run_dir, ROOT)
    assert rc == 0

    offenders = [
        str(py.relative_to(run_dir))
        for py in sorted((run_dir / "method").rglob("*.py"))
        if "raise NotImplementedError" in py.read_text(encoding="utf-8")
    ]
    assert not offenders, (
        f"scaffolder-owned file(s) in {paradigm_id} ship a "
        f"`raise NotImplementedError` placeholder that no producer can fill (orphaned): "
        f"{offenders}. Either drop the placeholder (the value should come from a "
        f"producer-written file like model.py, built by the caller) or move the code "
        f"into a producer-owned file."
    )


def test_downloading_templates_carry_offline_guard():
    """Every template data loader that can download honors R2C_OFFLINE (the
    2026-07-13 stage-2d network-stall fix): a validation harness that sets it
    must get a loud refusal, never an unbounded download."""
    downloading = [
        t for t in (ROOT / "paradigms").rglob("*.template")
        if "download=True" in t.read_text(encoding="utf-8")
    ]
    assert downloading, "expected at least one downloading template (sweep is broken)"
    missing_guard = [
        str(t.relative_to(ROOT)) for t in downloading
        if "R2C_OFFLINE" not in t.read_text(encoding="utf-8")
    ]
    assert not missing_guard, f"templates download without an offline guard: {missing_guard}"


# ---------------------------------------------------------------------------
# B-11 class-closing assertions: scaffolder-manifest coverage asymmetry.
# Declared-to-rendered was always checked; rendered-to-declared was not,
# which is how cross_modal's misplaced run-root README sat unnoticed.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("paradigm_id", _paradigm_ids_with_templates())
def test_every_rendered_file_is_declared_in_the_manifest(
        paradigm_id: str, tmp_path: Path):
    from scripts.build_plan import load_build_plan

    run_dir = tmp_path / "run"
    (run_dir / ".pipeline").mkdir(parents=True)
    spec = _min_spec(paradigm_id)
    spec_path = run_dir / ".pipeline" / "method_spec.json"
    spec_path.write_text(json.dumps(spec, indent=2))
    assert scaffold(spec_path, run_dir, ROOT) == 0

    plan = load_build_plan(spec, ROOT)
    assert plan is not None, paradigm_id
    declared = {e["path"] for e in plan["package_manifest"]["files"]}
    rendered = {
        p.relative_to(run_dir).as_posix()
        for p in run_dir.rglob("*")
        if p.is_file() and not p.relative_to(run_dir).as_posix()
        .startswith(".pipeline")
    }
    undeclared = rendered - declared
    assert not undeclared, (
        f"{paradigm_id}: scaffolder rendered files the manifest never "
        f"declares (nothing validates them): {sorted(undeclared)}"
    )


def test_every_template_lives_under_method():
    strays = [
        p.as_posix() for p in (ROOT / "paradigms").rglob("*.template")
        if "templates/method/" not in p.as_posix()
    ]
    assert not strays, (
        "templates outside templates/method/ render outside the package "
        f"(the cross_modal run-root README class): {strays}"
    )


def test_readme_conditioners_reach_the_scaffolded_cross_modal_readme(
        tmp_path: Path):
    """B-11 mandatory gate: moving cross_modal's README template under
    templates/method/ switches the driver's README conditioners from no-op
    to live on this paradigm — none of the scaffolder tests reached them.
    Scaffold a real cross_modal tree and prove two conditioners find and
    edit method/README.md."""
    import run_layout
    import run_pipeline
    from tests.helpers.state import make_state

    paradigm_id = "knowledge_distillation/detection/cross_modal"
    run_dir = tmp_path / "run"
    (run_dir / ".pipeline").mkdir(parents=True)
    spec_path = run_dir / ".pipeline" / "method_spec.json"
    spec_path.write_text(json.dumps(_min_spec(paradigm_id), indent=2))
    assert scaffold(spec_path, run_dir, ROOT) == 0

    readme = run_layout.run_path(run_dir, run_layout.PACKAGE_README)
    assert readme.is_file(), "the moved template must render method/README.md"
    assert "Run All" in readme.read_text(encoding="utf-8")

    state = make_state(run_dir)
    run_pipeline._condition_readme_run_invitation(state)
    conditioned = readme.read_text(encoding="utf-8")
    assert "Do not click Run All yet" in conditioned

    run_pipeline.finalize_delivery_banner(state, {
        "label": "draft",
        "reasons": [{"source": "test", "id": "x", "message": "m"}],
        "disclosures": [],
    })
    banner = readme.read_text(encoding="utf-8")
    assert banner.startswith("> **Delivery: draft**")
