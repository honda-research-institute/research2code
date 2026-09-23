"""Partial delivery — label cause, manifest field, and the stub formats.

Design: the partial delivery design note (internal, not shipped) (§3.3 labels, §3.5
unambiguity constraint). The invariants under test:

- nothing partial can ever read `verified` (derivation AND schema),
- stubs are a distinct recorded cause (`partial_delivery`), never a
  generic failure, and the structured `stubbed_elements` list survives
  into the verdict,
- the gap-family composition holds: stubs on an uncertified — new
  territory delivery record both facts without inventing a fourth label,
- stub files and work orders self-identify (criterion 3), and the
  delivery-time verifier catches every mute surface.
"""

from __future__ import annotations

import importlib.util
import json

import pytest

import run_layout
from delivery_label import (LABEL_DRAFT, LABEL_UNCERTIFIED, LABEL_VERIFIED,
                            derive_delivery_label)
from partial_delivery import (load_stubbed_elements, record_stubbed_element,
                              render_stub_module, render_work_order,
                              verify_stub_surfaces, write_stub)


def _report(*verdicts):
    return {"verdicts": [
        {"probe_id": pid, "verdict": v, "message": f"{pid} message"}
        for pid, v in verdicts]}


def _stub(eid="hungarian-matching", role="core", **kw):
    rec = {"element_id": eid, "role": role,
           "work_order": f"work_orders/{eid}.md"}
    rec.update(kw)
    return rec


# ---------------------------------------------------------------------------
# Label derivation (§3.3): the partial_delivery cause inside existing states.
# ---------------------------------------------------------------------------


def test_supporting_stub_with_contribution_evidence_is_draft_partial():
    out = derive_delivery_label(
        _report(("US-1", "pass"), ("CT-1", "pass")), None,
        stubbed_elements=[_stub("aux-logging-head", role="supporting")])
    assert out["label"] == LABEL_DRAFT
    assert out["partial"] is True
    assert [r["source"] for r in out["reasons"]] == ["partial_delivery"]
    assert "NOT implemented" in out["reasons"][0]["message"]
    assert "supporting component" in out["reasons"][0]["message"]
    assert out["stubbed_elements"][0]["element_id"] == "aux-logging-head"


def test_stub_can_never_read_verified():
    # The §3.5 floor: a probe sweep that would certify a complete package
    # cannot certify a partial one.
    clean = derive_delivery_label(
        _report(("US-1", "pass"), ("CT-1", "pass")), None)
    assert clean["label"] == LABEL_VERIFIED
    stubbed = derive_delivery_label(
        _report(("US-1", "pass"), ("CT-1", "pass")), None,
        stubbed_elements=[_stub()])
    assert stubbed["label"] == LABEL_DRAFT


def test_core_stub_message_names_the_core_mechanism():
    out = derive_delivery_label(
        _report(("CT-1", "pass")), None, stubbed_elements=[_stub()])
    assert out["label"] == LABEL_DRAFT
    assert "core mechanism" in out["reasons"][0]["message"]
    assert "hungarian-matching" in out["reasons"][0]["message"]
    assert "work_orders/hungarian-matching.md" in out["reasons"][0]["message"]


def test_core_stubs_sort_before_supporting():
    out = derive_delivery_label(
        _report(("CT-1", "pass")), None,
        stubbed_elements=[_stub("zz-aux", role="supporting"),
                          _stub("matcher", role="core")])
    assert [s["element_id"] for s in out["stubbed_elements"]] == [
        "matcher", "zz-aux"]
    assert out["reasons"][0]["id"] == "matcher"


def test_stub_reasons_ride_along_with_other_demoters():
    out = derive_delivery_label(
        _report(("US-3", "fail"), ("CT-1", "pass")), None,
        stubbed_elements=[_stub()])
    assert out["label"] == LABEL_DRAFT
    sources = [r["source"] for r in out["reasons"]]
    assert "partial_delivery" in sources and "probe" in sources
    # The stub cause leads — the missing core is the headline.
    assert sources[0] == "partial_delivery"


def test_gap_family_stubs_compose_with_uncertified():
    # Table row 4 (§3.3): no contribution evidence keeps the evidence-axis
    # label; the stubs are disclosed and recorded, never silently dropped,
    # and no fourth label appears.
    out = derive_delivery_label(
        _report(("US-1", "pass")), None, paradigm="motion_planning.srl",
        stubbed_elements=[_stub()])
    assert out["label"] == LABEL_UNCERTIFIED
    assert out["partial"] is True
    assert out["reasons"] == []
    assert out["missing_probe_family"] == "motion_planning.srl"
    assert any(d["source"] == "partial_delivery" for d in out["disclosures"])
    assert out["stubbed_elements"][0]["element_id"] == "hungarian-matching"


def test_no_stubs_keeps_existing_shape():
    out = derive_delivery_label(_report(("CT-1", "pass")), None)
    assert out["label"] == LABEL_VERIFIED
    assert out["partial"] is False
    assert out["stubbed_elements"] == []


# ---------------------------------------------------------------------------
# Manifest schema (§3.5 criteria 1 + 5 + 6).
# ---------------------------------------------------------------------------


def test_schema_refuses_partial_verified():
    from schemas.final_manifest import DeliveryVerdict

    with pytest.raises(ValueError, match="never be verified"):
        DeliveryVerdict(label="verified", partial=True,
                        stubbed_elements=[_stub()])
    with pytest.raises(ValueError, match="partial flag unset"):
        DeliveryVerdict(label="draft", stubbed_elements=[_stub()])


def test_schema_accepts_pre_partial_verdicts():
    from schemas.final_manifest import DeliveryVerdict

    v = DeliveryVerdict(label="verified")
    assert v.partial is False
    assert v.stubbed_elements == []


def test_partial_flag_renders_first_in_the_delivery_block():
    # §3.5 criterion 1 for the manifest surface: field order is
    # declaration order, and `partial` is declared first on purpose.
    from schemas.final_manifest import DeliveryVerdict

    v = DeliveryVerdict(label="draft", partial=True,
                        stubbed_elements=[_stub()])
    dumped = json.dumps(v.model_dump(), indent=2)
    first_field_line = dumped.splitlines()[1]
    assert '"partial": true' in first_field_line


def test_derived_verdict_validates_against_the_schema():
    from schemas.final_manifest import DeliveryVerdict

    out = derive_delivery_label(
        _report(("CT-1", "pass")), None,
        stubbed_elements=[_stub(stub_path="method/matcher.py")])
    v = DeliveryVerdict.model_validate(out)
    assert v.partial is True
    assert v.stubbed_elements[0].stub_path == "method/matcher.py"


# ---------------------------------------------------------------------------
# The canonical artifact: .pipeline/stubbed_elements.json.
# ---------------------------------------------------------------------------


def test_missing_artifact_means_no_stubs(tmp_path):
    assert load_stubbed_elements(tmp_path) == ([], None)


def test_unreadable_artifact_is_an_error_not_a_clean_read(tmp_path):
    (tmp_path / "stubbed_elements.json").write_text("{nope", encoding="utf-8")
    stubs, error = load_stubbed_elements(tmp_path)
    assert stubs == []
    assert "unreadable" in error


def test_bad_records_fail_schema_validation(tmp_path):
    (tmp_path / "stubbed_elements.json").write_text(
        json.dumps({"schema_version": "1.0",
                    "stubbed_elements": [{"element_id": "x", "role": "boss"}]}),
        encoding="utf-8")
    stubs, error = load_stubbed_elements(tmp_path)
    assert stubs == []
    assert "schema validation" in error


def test_record_roundtrip_replaces_on_element_id_and_sorts_core_first(tmp_path):
    record_stubbed_element(tmp_path, element_id="aux", role="supporting",
                           work_order="work_orders/aux.md")
    record_stubbed_element(tmp_path, element_id="matcher",
                           role="core_methodology",
                           work_order="work_orders/matcher.md",
                           stub_path="method/matcher.py")
    record_stubbed_element(tmp_path, element_id="aux", role="supporting",
                           work_order="work_orders/aux-v2.md")
    stubs, error = load_stubbed_elements(tmp_path)
    assert error is None
    assert [(s["element_id"], s["role"]) for s in stubs] == [
        ("matcher", "core"), ("aux", "supporting")]
    assert stubs[1]["work_order"] == "work_orders/aux-v2.md"


# ---------------------------------------------------------------------------
# The work-order format (§3.1) and the stub-module format (criterion 3).
# ---------------------------------------------------------------------------


def test_work_order_first_line_says_partial_and_sections_render():
    md = render_work_order(
        element_id="hungarian-matching",
        role="core_methodology",
        interface={
            "signature": "def match(cost: \"Tensor\") -> \"Tensor\"",
            "arguments": ["cost: [B, N, M] float32 pairwise costs"],
            "returns": "index tensor [B, N] int64",
            "call_sites": ["method/method.py:injects into loss assembly"],
        },
        paper_anchor={
            "section": "Section 3.2, Eq. 4",
            "quotes": ["we search for a permutation of N elements"],
        },
        why_not_built="fix loop exhausted: diagnostician died on output cap",
        verified_neighborhood=["loss assembly: KD-1 pass"],
        fix_history="iteration 3: same failing cell",
    )
    first = md.splitlines()[0]
    assert "PARTIAL" in first and "NOT implemented" in first
    for needle in ("CORE mechanism", "## The precise interface",
                   "def match", "[B, N, M]", "## The paper anchor",
                   "> we search for a permutation of N elements",
                   "## Why R2C could not build it", "fix loop exhausted",
                   "## What was verified around it", "KD-1 pass",
                   "## Fix-loop history"):
        assert needle in md, needle


def test_work_order_renders_honest_fallbacks_when_inputs_missing():
    md = render_work_order(element_id="aux", role="supporting",
                           why_not_built="dataset unobtainable")
    assert "PARTIAL" in md.splitlines()[0]
    assert "interface was not recorded" in md
    assert "No verbatim quotes were recorded" in md
    assert "treat the boundary as unverified" in md
    assert "## Fix-loop history" not in md


def _import_stub(tmp_path, src):
    path = tmp_path / "stub_mod.py"
    path.write_text(src, encoding="utf-8")
    spec = importlib.util.spec_from_file_location("stub_mod", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_stub_module_self_identifies_and_raises(tmp_path):
    src = render_stub_module(
        element_id="hungarian-matching",
        work_order="work_orders/hungarian-matching.md",
        signatures=["def match(cost) -> list"])
    mod = _import_stub(tmp_path, src)
    assert "NOT IMPLEMENTED" in mod.__doc__
    assert "PARTIAL" in mod.__doc__
    assert "work_orders/hungarian-matching.md" in mod.__doc__
    with pytest.raises(NotImplementedError, match="work_orders/"):
        mod.match([[1.0]])
    with pytest.raises(NotImplementedError):
        mod.anything_else
    # Dunder lookups keep normal semantics (inspection tooling survives).
    with pytest.raises(AttributeError):
        mod.__wrapped__


def test_write_stub_writes_both_surfaces_and_records(tmp_path):
    (tmp_path / ".pipeline").mkdir()
    write_stub(
        tmp_path,
        element_id="matcher",
        role="core_methodology",
        stub_rel_path="method/matcher.py",
        work_order_markdown=render_work_order(
            element_id="matcher", role="core_methodology",
            why_not_built="fix loop exhausted"),
        signatures=["def match(cost)"],
    )
    stubs, error = load_stubbed_elements(tmp_path / ".pipeline")
    assert error is None and len(stubs) == 1
    assert verify_stub_surfaces(tmp_path, stubs) == []


# ---------------------------------------------------------------------------
# Delivery-time surface verification (criterion 3, enforced).
# ---------------------------------------------------------------------------


def test_verifier_flags_missing_and_mute_surfaces(tmp_path):
    stubs = [_stub("matcher", stub_path="method/matcher.py")]
    problems = verify_stub_surfaces(tmp_path, stubs)
    assert any("work order" in p and "missing" in p for p in problems)
    assert any("stub file" in p and "missing" in p for p in problems)

    (tmp_path / "work_orders").mkdir()
    (tmp_path / "work_orders" / "matcher.md").write_text(
        "# a normal-looking file\n", encoding="utf-8")
    (tmp_path / "method").mkdir()
    (tmp_path / "method" / "matcher.py").write_text(
        "def match(cost):\n    return None\n", encoding="utf-8")
    problems = verify_stub_surfaces(tmp_path, stubs)
    assert any("does not say PARTIAL in its first line" in p
               for p in problems)
    assert any("does not say NOT IMPLEMENTED" in p for p in problems)
    assert any("does not raise" in p for p in problems)
    assert any("does not point at" in p for p in problems)


def test_verifier_passes_a_clean_stub(tmp_path):
    (tmp_path / ".pipeline").mkdir()
    rec = write_stub(
        tmp_path, element_id="aux", role="supporting",
        stub_rel_path="method/aux.py",
        work_order_markdown=render_work_order(
            element_id="aux", role="supporting", why_not_built="x"),
    )
    assert verify_stub_surfaces(tmp_path, [rec]) == []


# ---------------------------------------------------------------------------
# Surfaces (§3.5 criteria 1, 2, 4, 6): PARTIAL in the first line of every
# entry surface, numbers not adjectives, stubs named before anything else.
# ---------------------------------------------------------------------------


def _write_contract(pipeline_dir, n_elements=4):
    elements = [{"element_id": f"e{i}"} for i in range(n_elements)]
    (pipeline_dir / "method_spec.json").write_text(
        json.dumps({"methodology_replication_contract":
                    {"elements": elements}}),
        encoding="utf-8")


def _partial_delivery(stubs, label="draft"):
    reasons = [] if label == "uncertified_new_territory" else [
        {"source": "partial_delivery", "id": s["element_id"], "message": "m"}
        for s in stubs]
    out = {"schema_version": "1.2.0", "partial": True, "label": label,
           "reasons": reasons, "disclosures": [], "probe_counts": {},
           "stubbed_elements": stubs}
    if label == "uncertified_new_territory":
        out["missing_probe_family"] = "x.y"
    return out


def test_readme_banner_partial_first_line(tmp_path):
    from tests.helpers.state import make_state

    state = make_state(tmp_path / "run")
    _write_contract(state.paths.pipeline_dir)
    run_layout.run_path(state.paths.run_dir, run_layout.PACKAGE_README).write_text(
        "# Package\n\nbody\n", encoding="utf-8")
    delivery = _partial_delivery(
        [_stub("matcher", role="core"), _stub("aux", role="supporting")])
    from run_pipeline import finalize_delivery_banner
    finalize_delivery_banner(state, delivery)
    text = (state.paths.run_dir / run_layout.PACKAGE_README).read_text(encoding="utf-8")
    first = text.splitlines()[0]
    assert first.startswith("> **PARTIAL delivery — draft**")
    assert "2 of 4 components implemented, 2 stubbed" in first
    assert "core mechanism (`matcher`) is NOT implemented" in first
    assert "work_orders/" in first
    assert "# Package" in text

    # Idempotent: re-running replaces, never stacks; and a stale complete
    # banner is replaced by the partial one.
    finalize_delivery_banner(state, delivery)
    text = (state.paths.run_dir / run_layout.PACKAGE_README).read_text(encoding="utf-8")
    assert text.count("PARTIAL delivery — draft") == 1
    finalize_delivery_banner(state, {"schema_version": "1.2.0",
                                     "partial": False, "label": "verified",
                                     "reasons": [], "disclosures": [],
                                     "probe_counts": {},
                                     "stubbed_elements": []})
    text = (state.paths.run_dir / run_layout.PACKAGE_README).read_text(encoding="utf-8")
    assert "PARTIAL" not in text.splitlines()[0]
    finalize_delivery_banner(state, delivery)
    text = (state.paths.run_dir / run_layout.PACKAGE_README).read_text(encoding="utf-8")
    assert text.splitlines()[0].startswith("> **PARTIAL delivery")
    assert "**Delivery: verified**" not in text


def test_end_of_run_notice_partial_first_line(tmp_path):
    from tests.helpers.state import make_paths
    from run_pipeline import _render_end_of_run_notice

    paths = make_paths(tmp_path / "run")
    _write_contract(paths.pipeline_dir)
    notice = _render_end_of_run_notice(
        paths, run_status="completed", manifest_status="degraded",
        delivery=_partial_delivery([_stub("matcher", role="core")]))
    first = notice.splitlines()[0]
    assert "PARTIAL delivery" in first
    assert "3 of 4 components implemented, 1 stubbed" in first

    complete = _render_end_of_run_notice(
        paths, run_status="completed", manifest_status="passed",
        delivery={"label": "verified", "reasons": [], "disclosures": []})
    assert "PARTIAL" not in complete.splitlines()[0]


def test_report_leads_with_partial_and_numbers(tmp_path):
    from render_run_report import render_run_report

    run_dir = tmp_path / "run"
    (run_dir / ".pipeline").mkdir(parents=True)
    _write_contract(run_dir / ".pipeline")
    report = render_run_report(
        run_dir,
        delivery=_partial_delivery(
            [_stub("matcher", role="core"), _stub("aux", role="supporting")]),
        stage_results=[])
    lines = report.splitlines()
    assert lines[0] == "# PARTIAL delivery — Run Report"
    # Criterion 2: numbers + the stub names before anything else.
    head = "\n".join(lines[:12])
    assert "2 of 4 components implemented, 2 stubbed" in head
    assert "`matcher` — the paper's core mechanism" in head
    assert "`aux` — supporting component" in head
    assert head.find("stubbed") < report.find("Run: `")
    # §3.2: the core-gap headline is present and bold.
    assert "core mechanism (`matcher`) is NOT implemented" in head

    complete = render_run_report(run_dir, delivery={
        "label": "verified", "reasons": [], "disclosures": [],
        "probe_counts": {}}, stage_results=[])
    assert complete.splitlines()[0] == "# Run Report"


def _notebook_fixture(tmp_path, draft, stubs=None):
    run_dir = tmp_path / "run"
    pipeline = run_dir / ".pipeline"
    pipeline.mkdir(parents=True)
    (pipeline / "notebook_draft.py").write_text(draft, encoding="utf-8")
    (pipeline / "params.json").write_text(
        json.dumps({"params": {}}), encoding="utf-8")
    if stubs is not None:
        (pipeline / "stubbed_elements.json").write_text(
            json.dumps({"schema_version": "1.0",
                        "stubbed_elements": stubs}), encoding="utf-8")
    return run_dir


_DRAFT = """\
# %% [markdown]
# # Demo Title

# %% PLACEHOLDER: component_stub:matcher

# %%
print("hello")
"""


def test_notebook_renders_stub_pair_and_partial_banner(tmp_path):
    import nbformat
    from render_notebook import render

    run_dir = _notebook_fixture(
        tmp_path, _DRAFT,
        stubs=[_stub("matcher", role="core", stub_path="method/matcher.py")])
    _write_contract(run_dir / ".pipeline")
    assert render(run_dir) == 0
    nb = nbformat.read(run_dir / "notebook.ipynb", as_version=4)
    # Banner directly under the title, ahead of everything else.
    banner = nb.cells[1]
    assert banner.cell_type == "markdown"
    assert "PARTIAL delivery" in banner.source.splitlines()[0]
    assert "3 of 4 components implemented, 1 stubbed" in banner.source
    assert "core mechanism (`matcher`) is NOT implemented" in banner.source
    # The stub pair: markdown notice then raising code cell.
    md_idx = next(i for i, c in enumerate(nb.cells)
                  if "the next cell raises" in c.source)
    assert nb.cells[md_idx].cell_type == "markdown"
    assert "`matcher`" in nb.cells[md_idx].source
    code = nb.cells[md_idx + 1]
    assert code.cell_type == "code"
    assert "raise NotImplementedError" in code.source
    assert "work_orders/matcher.md" in code.source


def test_notebook_banner_injected_even_without_markers(tmp_path):
    import nbformat
    from render_notebook import render

    draft = "# %% [markdown]\n# # Demo Title\n\n# %%\nprint('hi')\n"
    run_dir = _notebook_fixture(
        tmp_path, draft, stubs=[_stub("matcher", role="core")])
    assert render(run_dir) == 0
    nb = nbformat.read(run_dir / "notebook.ipynb", as_version=4)
    assert "PARTIAL delivery" in nb.cells[1].source


def test_notebook_unknown_stub_marker_fails_render(tmp_path):
    from render_notebook import render

    run_dir = _notebook_fixture(tmp_path, _DRAFT, stubs=[])
    assert render(run_dir) == 2


def test_notebook_without_stubs_renders_unchanged(tmp_path):
    import nbformat
    from render_notebook import render

    draft = "# %% [markdown]\n# # Demo Title\n\n# %%\nprint('hi')\n"
    run_dir = _notebook_fixture(tmp_path, draft)
    assert render(run_dir) == 0
    nb = nbformat.read(run_dir / "notebook.ipynb", as_version=4)
    assert all("PARTIAL" not in c.source for c in nb.cells)


def test_notebook_validator_criterion_four(tmp_path):
    from validate_notebook_output import _partial_stub_errors

    run_dir = tmp_path / "run"
    pipeline = run_dir / ".pipeline"
    pipeline.mkdir(parents=True)
    (pipeline / "stubbed_elements.json").write_text(
        json.dumps({"schema_version": "1.0",
                    "stubbed_elements": [
                        _stub("matcher", role="core",
                              stub_path="method/matcher.py")]}),
        encoding="utf-8")

    silent = _partial_stub_errors(run_dir, ["# Demo"], ["print('hi')"])
    assert any("no markdown cell says PARTIAL" in e for e in silent)
    assert any("no code cell raises" in e for e in silent)

    spoken = _partial_stub_errors(
        run_dir,
        ["> PARTIAL delivery: `matcher` is NOT implemented"],
        ["raise NotImplementedError('matcher stub, see work_orders/')"])
    assert spoken == []

    # A gate-recorded obligation with NO code representation requires the
    # notice only — there is no cell that could exercise it, so no raising
    # cell is demanded.
    (pipeline / "stubbed_elements.json").write_text(
        json.dumps({"schema_version": "1.0",
                    "stubbed_elements": [_stub("proprietary-sensor-sim",
                                               role="supporting")]}),
        encoding="utf-8")
    notice_only = _partial_stub_errors(
        run_dir,
        ["> PARTIAL delivery: `proprietary-sensor-sim` is NOT implemented"],
        ["print('hi')"])
    assert notice_only == []

    (pipeline / "stubbed_elements.json").write_text("{bad", encoding="utf-8")
    broken = _partial_stub_errors(run_dir, [], [])
    assert any("unreadable" in e for e in broken)


def test_run_delivery_gating_wires_stubs_into_the_verdict(tmp_path, monkeypatch):
    """End-to-end through the driver's gate: recorded stubs surface in the
    verdict (partial, stubbed_elements, partial_delivery reasons) and a
    mute stub file demotes via the surface verifier."""
    import subprocess as sp
    from tests.helpers.state import make_state
    from run_pipeline import run_delivery_gating

    state = make_state(tmp_path / "run")
    pipeline = state.paths.pipeline_dir
    # A current reference-qualified battery row with contribution evidence,
    # so the complete-package label would have been verified.  The delivery
    # gate intentionally fails closed when method_spec.json is missing; this
    # fixture exercises partial-delivery demotion rather than that independent
    # artifact-integrity arm.
    (pipeline / "probe_report.json").write_text(json.dumps({
        "verdicts": [{"probe_id": "CT-1", "verdict": "pass",
                      "message": "ok",
                      "probe_ref": "claims.contribution_floor",
                      "element_ids": ["contribution"]}]}), encoding="utf-8")
    (pipeline / "method_spec.json").write_text(json.dumps({
        "comparison": {
            "classification": {"id": "adaptive_moment_optimizer"},
        },
        "methodology_replication_contract": {
            "elements": [{
                "element_id": "contribution",
                "role": "core_methodology",
                "replication_status": "must_replicate",
                "technical_concept": "the paper contribution",
                "required_behavior": "the contribution changes the result",
                "paper_section": "Section 3",
                "acceptable_approximations": [],
                "forbidden_substitutions": [],
                "paper_element_ids": ["paper-contribution"],
                "verification_probe_refs": ["claims.contribution_floor"],
            }],
        },
    }), encoding="utf-8")

    def fake_run(*args, **kwargs):
        class P:
            returncode = 0
            stdout = ""
            stderr = ""
        return P()

    monkeypatch.setattr(sp, "run", fake_run)
    write_stub(
        state.paths.run_dir, element_id="matcher", role="core_methodology",
        stub_rel_path="method/matcher.py",
        work_order_markdown=render_work_order(
            element_id="matcher", role="core_methodology",
            why_not_built="fix loop exhausted"),
    )
    delivery = run_delivery_gating(state)
    assert delivery["partial"] is True
    assert delivery["label"] == LABEL_DRAFT
    assert delivery["stubbed_elements"][0]["element_id"] == "matcher"
    assert all(r["source"] == "partial_delivery" for r in delivery["reasons"])

    # Mute the stub file: the surface verifier must demote loudly.
    (state.paths.run_dir / "method" / "matcher.py").write_text(
        "def match(cost):\n    return None\n", encoding="utf-8")
    delivery = run_delivery_gating(state)
    assert delivery["label"] == LABEL_DRAFT
    assert any(r["id"] == "stub_surface" for r in delivery["reasons"])


# ---------------------------------------------------------------------------
# Producer #1: the feasibility gate records not-replicable SUPPORTING
# elements as stubs (maintainer-approved 2026-07-05, option A) — recorded proceed
# instead of today's silent proceed, reconciled against the final spec.
# ---------------------------------------------------------------------------


def _producer_spec(status="not_replicable"):
    return {
        "comparison": {"classification": {"id": "motion_planning"}},
        "methodology_replication_contract": {"elements": [
            {"element_id": "core-mechanism", "role": "core_methodology",
             "replication_status": "must_replicate"},
            {"element_id": "proprietary-sensor-sim",
             "role": "supporting_mechanism",
             "replication_status": status,
             "paper_section": "Section IV-A",
             "paper_evidence": "evaluated with the vendor sensor model",
             "feasibility_rationale": "the vendor simulator is unobtainable",
             "blockers": ["closed-source SDK"]},
            {"element_id": "hardware-demo", "role": "evaluation_control",
             "replication_status": "not_replicable"},
        ]}}


def test_gate_producer_records_supporting_stub_with_work_order(tmp_path):
    from tests.helpers.state import make_state
    from run_pipeline import _record_feasibility_gate_stubs

    state = make_state(tmp_path / "run")
    state.paths.method_spec.write_text(json.dumps(_producer_spec()),
                                       encoding="utf-8")
    _record_feasibility_gate_stubs(state, "stage_1")

    stubs, error = load_stubbed_elements(state.paths.pipeline_dir)
    assert error is None
    # Only the supporting element: core stays the gate's halt business,
    # evaluation_control is not a package component.
    assert [(s["element_id"], s["role"]) for s in stubs] == [
        ("proprietary-sensor-sim", "supporting")]
    assert "stub_path" not in stubs[0]
    wo = (state.paths.run_dir / "work_orders" / "proprietary-sensor-sim.md")
    text = wo.read_text(encoding="utf-8")
    assert "PARTIAL" in text.splitlines()[0]
    assert "vendor simulator is unobtainable" in text
    assert "closed-source SDK" in text
    assert "> evaluated with the vendor sensor model" in text
    assumptions = (state.paths.run_dir / run_layout.ASSUMPTIONS_MD).read_text(
        encoding="utf-8")
    assert "ships as a stub" in assumptions
    events = (state.paths.pipeline_dir / "run_events.jsonl").read_text(
        encoding="utf-8")
    assert "gate_stubs_recorded" in events
    # And the delivery-time surface verifier accepts the recorded shape.
    assert verify_stub_surfaces(state.paths.run_dir, stubs) == []


def test_gate_producer_is_idempotent_and_reconciles(tmp_path):
    from tests.helpers.state import make_state
    from run_pipeline import _record_feasibility_gate_stubs

    state = make_state(tmp_path / "run")
    state.paths.method_spec.write_text(json.dumps(_producer_spec()),
                                       encoding="utf-8")
    _record_feasibility_gate_stubs(state, "stage_1")
    _record_feasibility_gate_stubs(state, "stage_1")
    stubs, _ = load_stubbed_elements(state.paths.pipeline_dir)
    assert len(stubs) == 1
    assumptions = (state.paths.run_dir / run_layout.ASSUMPTIONS_MD).read_text(
        encoding="utf-8")
    assert assumptions.count("ships as a stub") == 1

    # A corrected spec on resume withdraws the stale record, loudly.
    state.paths.method_spec.write_text(
        json.dumps(_producer_spec(status="must_replicate")),
        encoding="utf-8")
    _record_feasibility_gate_stubs(state, "stage_1")
    stubs, error = load_stubbed_elements(state.paths.pipeline_dir)
    assert error is None and stubs == []
    assert not (state.paths.pipeline_dir / "stubbed_elements.json").is_file()
    assumptions = (state.paths.run_dir / run_layout.ASSUMPTIONS_MD).read_text(
        encoding="utf-8")
    assert "withdrawn" in assumptions


def test_gate_producer_feeds_the_delivery_gate_end_to_end(tmp_path, monkeypatch):
    """The full producer-to-label path: gate records the stub, delivery
    reads it, the package is partial draft with the stub cause, and the
    end-of-run surfaces get the PARTIAL treatment."""
    import subprocess as sp
    from tests.helpers.state import make_state
    from run_pipeline import _record_feasibility_gate_stubs, run_delivery_gating

    state = make_state(tmp_path / "run")
    state.paths.method_spec.write_text(json.dumps(_producer_spec()),
                                       encoding="utf-8")
    (state.paths.pipeline_dir / "probe_report.json").write_text(json.dumps({
        "verdicts": [{"probe_id": "MP-1", "verdict": "pass",
                      "message": "ok"}]}), encoding="utf-8")

    def fake_run(*args, **kwargs):
        class P:
            returncode = 0
            stdout = ""
            stderr = ""
        return P()

    monkeypatch.setattr(sp, "run", fake_run)
    _record_feasibility_gate_stubs(state, "stage_1")
    delivery = run_delivery_gating(state)
    assert delivery["partial"] is True
    assert delivery["label"] == LABEL_DRAFT
    assert any(r["source"] == "partial_delivery"
               and r["id"] == "proprietary-sensor-sim"
               for r in delivery["reasons"])


def test_run_delivery_gating_filters_approximations_by_role(tmp_path, monkeypatch):
    """Only core_methodology approximations demote; supporting and
    demo-scale approximations never reach the derivation (maintainer-approved
    role filter, 2026-07-05)."""
    import subprocess as sp
    from tests.helpers.state import make_state
    from run_pipeline import run_delivery_gating

    state = make_state(tmp_path / "run")
    pipeline = state.paths.pipeline_dir
    (pipeline / "probe_report.json").write_text(json.dumps({
        "verdicts": [{"probe_id": "MP-1", "verdict": "pass",
                      "message": "ok"}]}), encoding="utf-8")
    (pipeline / "method_spec.json").write_text(json.dumps({
        "comparison": {"classification": {"id": "motion_planning"}},
        "methodology_replication_contract": {"elements": [
            {"element_id": "value-network", "role": "core_methodology",
             "replication_status": "faithful_approximation_allowed"},
            {"element_id": "core-kept", "role": "core_methodology",
             "replication_status": "must_replicate"},
            {"element_id": "hardware-demo", "role": "evaluation_control",
             "replication_status": "not_replicable"},
            {"element_id": "aux-buffer", "role": "supporting_mechanism",
             "replication_status": "faithful_approximation_allowed"},
        ]}}), encoding="utf-8")

    def fake_run(*args, **kwargs):
        class P:
            returncode = 0
            stdout = ""
            stderr = ""
        return P()

    monkeypatch.setattr(sp, "run", fake_run)
    delivery = run_delivery_gating(state)
    assert delivery["label"] == LABEL_DRAFT
    approx = [r for r in delivery["reasons"]
              if r["source"] == "approximated_core"]
    assert [r["id"] for r in approx] == ["value-network"]
