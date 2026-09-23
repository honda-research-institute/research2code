"""Halt-reason catalog (queue item 12, halt-reason-rewrite-design.md).

Covers the catalog module itself (closed classes, evidence-bounded
mechanism resolution), the artifact fields, both researcher surfaces
consuming the same story, old-artifact tolerance, and the two 7/4 golden
fixtures: detr stage 3c (harvested verbatim — the wrong-mechanism headline
the design exists to kill) and SRL stage 2a (reconstructed from the design
note; the original artifact was overwritten by a later run).
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import halt_catalog
from dispatch_templates import build_halt_artifact
from render_run_report import render_run_report
from run_pipeline import _render_halt_notice

FIXTURES = Path(__file__).parent / "fixtures" / "evidence" / "halt-artifacts"

# The bare-code scrub run_pipeline applies to researcher notices; catalog
# stories must never need it. Same grammar as the live scrub paths (one
# definition, render_claims_report) so this guard covers every id shape
# they do, stage-scoped ids included.
from render_claims_report import CODE_RE as _CODE_RE  # noqa: E402

# The detr 2026-07-04 dispatch evidence, as the run events recorded it:
# completed, well under the 900s budget, zero writes, twice in a row.
DETR_EVIDENCE = {
    "completed": True,
    "elapsed_s": 412.0,
    "timeout_s": 900.0,
    "writes_observed": 0,
    "attempts": 2,
    "step_gloss": "the debugger's analysis step",
    "failing_location": "the distillation loss call, cell 45 of the notebook",
}


class _Stage:
    def __init__(self, stage_id, status, notes=""):
        self.stage_id = stage_id
        self.status = status
        self.notes = notes


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


# ---------------------------------------------------------------------------
# Mechanism resolver: evidence-bounded, never a guess
# ---------------------------------------------------------------------------


def test_mechanism_detr_no_write_is_not_called_a_timeout():
    phrase = halt_catalog.resolve_mechanism(DETR_EVIDENCE)
    assert phrase is not None
    assert "without producing its file" in phrase
    assert "twice" in phrase
    assert "not a timeout" in phrase
    assert "timed out" not in phrase


def test_mechanism_real_timeout_reads_as_timeout():
    phrase = halt_catalog.resolve_mechanism(
        {"completed": False, "elapsed_s": 901.0, "timeout_s": 900.0})
    assert "timeout" in phrase


def test_mechanism_transport_outranks_timing_and_writes():
    phrase = halt_catalog.resolve_mechanism(
        {"error_kind": "transport", "completed": True, "writes_observed": 0,
         "elapsed_s": 950.0, "timeout_s": 900.0})
    assert "connection failed" in phrase


def test_mechanism_unsettled_evidence_resolves_to_none():
    assert halt_catalog.resolve_mechanism(None) is None
    assert halt_catalog.resolve_mechanism({}) is None
    # Completed WITH writes: nothing here proves a mechanism.
    assert halt_catalog.resolve_mechanism(
        {"completed": True, "writes_observed": 3}) is None
    # Elapsed under budget with no other signal: not a timeout, not no-write.
    assert halt_catalog.resolve_mechanism(
        {"elapsed_s": 100.0, "timeout_s": 900.0}) is None


def test_mechanism_single_no_write_has_no_attempt_count():
    phrase = halt_catalog.resolve_mechanism(
        {"completed": True, "writes_observed": 0})
    assert phrase == "finished its turn without producing its file"


# ---------------------------------------------------------------------------
# The catalog: closed, complete, researcher-voiced
# ---------------------------------------------------------------------------


def test_every_class_renders_a_complete_plain_language_story():
    for name in sorted(halt_catalog.HALT_CLASSES):
        story = halt_catalog.render_halt_story(name, stage_id="stage_2b")
        assert story is not None, name
        blob = " ".join(
            [story.what_happened, story.why_stopped, story.what_next])
        assert "{" not in blob and "}" not in blob, (name, blob)
        # No stage ids, no probe/finding codes, no script names on the
        # researcher surface; those live in the technical collapsible.
        assert "stage_2b" not in blob, name
        assert not _CODE_RE.search(blob), (name, blob)
        assert ".py" not in blob, (name, blob)


def test_unknown_or_missing_class_returns_none():
    assert halt_catalog.render_halt_story(None, stage_id="stage_1") is None
    assert halt_catalog.render_halt_story(
        "not_a_class", stage_id="stage_1") is None


def test_stage_activity_tolerates_suffixed_and_unknown_ids():
    assert halt_catalog.stage_activity("stage_3c") == \
        "running the demo notebook end to end"
    assert halt_catalog.stage_activity("stage_2b_arch") == \
        "generating the model and training code"
    assert halt_catalog.stage_activity("stage_99") == "processing this paper"


def test_transport_story_carries_the_pending_finding():
    story = halt_catalog.render_halt_story(
        "transport_failure", stage_id="stage_3c",
        evidence={"error_kind": "transport",
                  "pending_finding": "diagnose why the demo never beat "
                                     "chance accuracy"})
    assert "diagnose why the demo never beat chance accuracy" in \
        story.what_happened
    # The pending upstream question is framed as the part worth reading.
    assert "still open" in story.what_happened


# ---------------------------------------------------------------------------
# Every halt site declares a valid class (design rule 1: a site that fits
# no class is a review finding, not a reason for a catch-all)
# ---------------------------------------------------------------------------


# Class-resolving helpers a site may call instead of a literal.
_CLASS_HELPERS = {
    "_dispatch_error_halt_class",
    "_judge_invocation_halt_class",
    "_pack_authoring_halt_class",
}


def _halt_class_values(node: ast.expr) -> list[str] | None:
    """Literal class strings a halt_class kwarg can evaluate to, or None
    when the expression is an approved helper call."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return [node.value]
    if isinstance(node, ast.Call):
        fn = node.func
        name = fn.id if isinstance(fn, ast.Name) else getattr(fn, "attr", "")
        if name in _CLASS_HELPERS:
            return None
        raise AssertionError(
            f"halt_class computed by unapproved call {ast.dump(fn)}")
    if isinstance(node, ast.IfExp):
        values = []
        for branch in (node.body, node.orelse):
            got = _halt_class_values(branch)
            if got:
                values.extend(got)
        return values
    raise AssertionError(
        f"halt_class is not a literal, helper call, or conditional of "
        f"literals: {ast.dump(node)}")


def test_every_driver_halt_site_declares_a_catalog_class():
    source = (Path(__file__).parent.parent / "scripts" /
              "run_pipeline.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    sites = missing = 0
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "halt"):
            continue
        sites += 1
        kwargs = {kw.arg: kw.value for kw in node.keywords if kw.arg}
        if "halt_class" not in kwargs:
            missing += 1
            continue
        values = _halt_class_values(kwargs["halt_class"])
        for value in values or []:
            assert value in halt_catalog.HALT_CLASSES, (
                f"line {node.lineno}: halt_class {value!r} is not in the "
                "catalog")
    assert sites > 100, f"AST scan found only {sites} halt sites — scan broken?"
    assert missing == 0, (
        f"{missing} halt site(s) declare no halt_class. Every halt() call "
        "must classify itself from scripts/halt_catalog.py; a site that "
        "fits no class is a design review finding, not a skip.")


# ---------------------------------------------------------------------------
# Stage 2.d pip-failure split: a dependency-RESOLUTION failure is a pipeline
# bug (the run shipped an uninstallable requirements.txt), not transport.
# Fixture shape: DomIndOnto 2026-07-21 roll 2 — the finalizer's import scan
# let the local package name `method` into requirements.txt and pip could
# not resolve it, yet the halt was classed as a server-connection problem.
# ---------------------------------------------------------------------------


_DOMINDONTO_RESOLUTION_STDERR = (
    "ERROR: Could not find a version that satisfies the requirement method "
    "(from versions: none)\n"
    "ERROR: No matching distribution found for method\n"
)


def test_pip_resolution_failure_is_a_pipeline_bug_not_transport():
    from run_pipeline import _pip_resolution_failure
    assert _pip_resolution_failure(_DOMINDONTO_RESOLUTION_STDERR) is True


def test_pip_resolver_conflict_is_a_pipeline_bug_too():
    from run_pipeline import _pip_resolution_failure
    conflict = (
        "ERROR: Cannot install -r requirements.txt (line 3) and numpy==1.24.0 "
        "because these package versions have conflicting dependencies.\n"
        "ERROR: ResolutionImpossible: for help visit ...\n"
    )
    assert _pip_resolution_failure(conflict) is True


def test_pip_network_failures_stay_transport():
    from run_pipeline import _pip_resolution_failure
    # An unreachable index ALSO ends in "No matching distribution found
    # (from versions: none)" after the retries — the wire evidence must win.
    offline = (
        "WARNING: Retrying (Retry(total=0, connect=None, read=None)) after "
        "connection broken by 'NewConnectionError(...: Failed to establish "
        "a new connection: [Errno -3] Temporary failure in name "
        "resolution')': /simple/torch/\n"
        "ERROR: Could not find a version that satisfies the requirement "
        "torch (from versions: none)\n"
        "ERROR: No matching distribution found for torch\n"
    )
    assert _pip_resolution_failure(offline) is False
    # The driver's own install-timeout message is transport as well.
    assert _pip_resolution_failure(
        "pip install of the package requirements timed out after 900s"
    ) is False


def test_pip_unrecognized_failures_keep_the_historical_transport_read():
    from run_pipeline import _pip_resolution_failure
    assert _pip_resolution_failure("ERROR: something unrecognized") is False
    assert _pip_resolution_failure("") is False
    assert _pip_resolution_failure(None) is False


# ---------------------------------------------------------------------------
# Artifact fields
# ---------------------------------------------------------------------------


def test_halt_artifact_records_class_and_evidence():
    artifact = build_halt_artifact(
        stage="stage_3c", reason="smoke gate failed after cap=3",
        halt_class="fix_loop_exhausted", evidence=DETR_EVIDENCE)
    assert artifact["halt_class"] == "fix_loop_exhausted"
    assert artifact["evidence"]["writes_observed"] == 0


def test_halt_artifact_without_class_is_unchanged():
    artifact = build_halt_artifact(stage="stage_1", reason="x failed")
    assert "halt_class" not in artifact and "evidence" not in artifact


# ---------------------------------------------------------------------------
# TUI notice (run_pipeline._render_halt_notice)
# ---------------------------------------------------------------------------


def test_notice_classified_halt_renders_catalog_story():
    notice = _render_halt_notice(
        "stage_3c", "smoke gate failed after cap=3",
        Path("/tmp/run/.pipeline/stage_3c.halt"), None,
        halt_class="fix_loop_exhausted", evidence=DETR_EVIDENCE)
    assert "**What happened.**" in notice
    assert "**Why we stopped instead of guessing.**" in notice
    assert "**What to do next.**" in notice
    assert "internal pipeline error" not in notice
    # Technical reason stays, behind the collapsible.
    assert "smoke gate failed after cap=3" in notice.split("<details>")[1]


def test_notice_hand_set_user_message_wins_over_catalog():
    notice = _render_halt_notice(
        "stage_1", "feasibility gate blocked the run",
        Path("/tmp/run/.pipeline/stage_1.halt"),
        "The paper's core mechanism needs hardware we cannot emulate.",
        halt_class="not_feasible")
    assert "hardware we cannot emulate" in notice
    # The catalog story does not double-render under the message.
    assert "**What happened.**" not in notice


def test_terminal_scope_notice_uses_gap_action_instead_of_resume():
    recorded_action = "Reject this document from the current method pipeline."
    context = {"paradigm_gap_report": {
        "decision": "unsupported_or_unclear",
        "recommended_next_action": recorded_action,
    }}

    notice = _render_halt_notice(
        "stage_1",
        "analyzer halted after understanding the protocol",
        Path("/tmp/run/.pipeline/stage_1.halt"),
        "Old static family list; resume the run.",
        halt_class="paradigm_mismatch",
        context=context,
    )

    assert halt_catalog.terminal_gap_action(
        "paradigm_mismatch", context
    ) == recorded_action
    assert "The document was understood" in notice
    assert "coverage or routing" in notice
    assert recorded_action in notice
    assert "Old static family list" not in notice
    assert "Do not resume the unchanged run" in notice
    assert "To resume after addressing the issue" not in notice
    assert "python3 scripts/run_pipeline.py" not in notice


def test_terminal_scope_notice_with_empty_action_still_never_resumes():
    context = {"paradigm_gap_report": {
        "decision": "unsupported_or_unclear",
        "recommended_next_action": "",
    }}

    notice = _render_halt_notice(
        "stage_1",
        "analyzer could not route the contribution",
        Path("/tmp/run/.pipeline/stage_1.halt"),
        "Resume the run.",
        halt_class="paradigm_mismatch",
        context=context,
    )

    assert "gap report did not record a next action" in notice
    assert "choose an in-scope input" in notice
    assert "Resume the run" not in notice
    assert "To resume after addressing the issue" not in notice
    assert "python3 scripts/run_pipeline.py" not in notice


def test_notice_unclassified_halt_keeps_stock_engineering_wording():
    notice = _render_halt_notice(
        "stage_2x", "derive_params.py exit 2",
        Path("/tmp/run/.pipeline/stage_2x.halt"), None)
    assert "internal pipeline error" in notice
    assert "report" in notice and "engineering" in notice


# ---------------------------------------------------------------------------
# REPORT.md halt block: same story as the notice
# ---------------------------------------------------------------------------


def test_report_halt_block_renders_catalog_story(tmp_path):
    run = tmp_path / "halted-run"
    _write_json(run / ".pipeline" / "stage_3c.halt", build_halt_artifact(
        stage="stage_3c", reason="smoke gate failed after cap=3",
        halt_class="fix_loop_exhausted", evidence=DETR_EVIDENCE))
    report = render_run_report(
        run, stage_results=[_Stage("stage_3c", "halted", "cap exhausted")])

    story = halt_catalog.render_halt_story(
        "fix_loop_exhausted", stage_id="stage_3c", evidence=DETR_EVIDENCE)
    # The exact sentences the TUI notice showed appear in the report — the
    # two surfaces never tell different stories.
    assert story.what_happened in report
    assert story.why_stopped in report
    assert story.what_next in report
    # The generic internal-error wording is replaced.
    assert "hit an internal error" not in report
    # Resume command still present and concrete.
    assert "`/r2c-run halted-run`" in report


def test_report_halt_block_user_message_still_wins(tmp_path):
    run = tmp_path / "halted-run"
    _write_json(run / ".pipeline" / "stage_1.halt", build_halt_artifact(
        stage="stage_1", reason="feasibility gate blocked the run",
        user_message="Confirm the value in Section 4.2 and re-run.",
        halt_class="not_feasible"))
    report = render_run_report(
        run, stage_results=[_Stage("stage_1", "halted", "blocked")])
    assert "Confirm the value in Section 4.2" in report
    assert "We stopped while" not in report


# ---------------------------------------------------------------------------
# Diagnostician evidence builder (block 3): claims bounded by what the
# driver observed
# ---------------------------------------------------------------------------


class _FakeResult:
    def __init__(self, completed, elapsed_s):
        self.completed = completed
        self.elapsed_s = elapsed_s


def test_diagnostician_evidence_claims_no_write_only_when_file_absent():
    from run_pipeline import _diagnostician_halt_evidence
    attempts = [("attempt 1", _FakeResult(True, 410.0)),
                ("attempt 2 (write-first retry)", _FakeResult(True, 412.0))]

    absent = _diagnostician_halt_evidence(attempts, diagnosis_file_exists=False)
    assert absent["writes_observed"] == 0 and absent["attempts"] == 2
    phrase = halt_catalog.resolve_mechanism(absent)
    assert "twice" in phrase and "not a timeout" in phrase

    # File present but invalid: no zero-write claim, mechanism unsettled.
    present = _diagnostician_halt_evidence(attempts, diagnosis_file_exists=True)
    assert "writes_observed" not in present
    assert halt_catalog.resolve_mechanism(present) is None

    assert _diagnostician_halt_evidence([], diagnosis_file_exists=False) is None


def test_judge_halt_rationale_stays_out_of_the_headline():
    """Item 12 rule 3 (supersedes R-004 for halts): the judge's rationale —
    the detr fixture's wrong '900s timeout' guess — renders only behind the
    technical collapsible; the headline is the catalog story plus the
    provable mechanism."""
    rationale = "most likely due to another 900s dispatch timeout"
    notice = _render_halt_notice(
        "stage_3c", f"halt-judge decided to halt: {rationale}",
        Path("/tmp/run/.pipeline/stage_3c.halt"), None,
        halt_class="judge_halt", evidence=DETR_EVIDENCE)
    story = notice.split("<details>")[0]
    assert "automated reviewer" in story
    assert "900s" not in story
    assert "finished twice without producing its file" in story
    assert rationale in notice.split("<details>")[1]


# ---------------------------------------------------------------------------
# Golden: detr stage 3c, 2026-07-04 (verbatim fixture)
# ---------------------------------------------------------------------------


def test_detr_verbatim_artifact_predates_catalog_and_renders_stock(tmp_path):
    """Old-artifact tolerance: the harvested 7/4 artifact has no halt_class,
    so both surfaces render exactly the pre-catalog engineering wording."""
    artifact = json.loads(
        (FIXTURES / "detr-3c-20260704.halt.json").read_text(encoding="utf-8"))
    assert "halt_class" not in artifact  # it predates the field

    run = tmp_path / "detr-distill"
    _write_json(run / ".pipeline" / "stage_3c.halt", artifact)
    report = render_run_report(
        run, stage_results=[_Stage("stage_3c", "halted", artifact["reason"])])
    assert "hit an internal error" in report

    notice = _render_halt_notice(
        "stage_3c", artifact["reason"],
        run / ".pipeline" / "stage_3c.halt", artifact.get("user_message"))
    assert "internal pipeline error" in notice


def test_detr_golden_reclassified_story_never_guesses_a_timeout(tmp_path):
    """The design's target rendering: same halt, now classified with the
    evidence the driver held all along. The story reads the provable
    no-write mechanism; the guessed '900s timeout' headline is gone."""
    verbatim = json.loads(
        (FIXTURES / "detr-3c-20260704.halt.json").read_text(encoding="utf-8"))
    artifact = build_halt_artifact(
        stage="stage_3c", reason=verbatim["reason"],
        retry_count=verbatim["retry_count"], context=verbatim["context"],
        halt_class="fix_loop_exhausted", evidence=DETR_EVIDENCE)

    run = tmp_path / "detr-distill"
    _write_json(run / ".pipeline" / "stage_3c.halt", artifact)
    report = render_run_report(
        run, stage_results=[_Stage("stage_3c", "halted", artifact["reason"])])

    block = report.split("## This run stopped")[1].split("<details>")[0]
    assert "running the demo notebook end to end" in block
    assert "finished twice without producing its file" in block
    assert "not a timeout" in block
    assert "cell 45" in block
    # The wrong-mechanism prose stays out of the plain-language story.
    assert "900s" not in block
    assert "timed out" not in block
    # The technical reason (with its 900s claim) is preserved for
    # engineering below the story, not erased.
    assert "900s" in report.split("<details>")[1]


# ---------------------------------------------------------------------------
# Golden: SRL stage 2a, 2026-07-04 (reconstructed from the design note)
# ---------------------------------------------------------------------------


def test_srl_golden_gap_paper_scaffold_failure_reads_as_coverage_gap(tmp_path):
    """Reconstructed 7/4 fixture: 'scaffold_package.py exit 1' on a paper
    running under a newly authored provisional pack. The researcher story
    is the coverage-gap one — no script names, no exit codes."""
    artifact = build_halt_artifact(
        stage="stage_2a", reason="scaffold_package.py exit 1",
        context={"stderr": "KeyError: 'package_manifest'"},
        halt_class="gap_pack_rejected")

    run = tmp_path / "SRL"
    _write_json(run / ".pipeline" / "stage_2a.halt", artifact)
    report = render_run_report(
        run, stage_results=[_Stage("stage_2a", "halted", artifact["reason"])])

    block = report.split("## This run stopped")[1].split("<details>")[0]
    assert "setting up the code package" in block
    assert "new to the system" in block
    assert "Nothing on your side is wrong" in block
    assert "coverage gap" in block
    assert "scaffold_package" not in block
    assert "exit 1" not in block
    # The technical detail keeps the script name and exit code.
    assert "scaffold_package.py exit 1" in report.split("<details>")[1]
