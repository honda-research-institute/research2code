"""Halt-reason catalog: plain-language, evidence-bounded stopped-run stories.

Queue item 12 (halt-reason-rewrite-design.md, drafted 2026-07-04 from the two
7/4 live fixtures). Three rules, in priority order:

1. Halt classes are first-class. Every halt() call site declares a
   `halt_class` from the closed catalog below. The catalog — not the call
   site — owns the researcher-facing story (what happened, why we stopped
   instead of guessing, what to do next). A site that fits no class is a
   review finding, not a reason to add a catch-all.
2. Mechanism claims are evidence-bounded. `resolve_mechanism` may only name
   a mechanism when a structured evidence field proves it (elapsed at/over
   budget → timed out; completed with zero writes → finished without
   producing its file; a transport error → the connection failed). When no
   field settles it, the story says what is KNOWN and stops — the
   smoke-runner honesty rule applied at delivery.
3. Internal prose never headlines. Judge rationales, validator stderr, and
   diagnostician text stay behind the technical collapsible, labeled as
   internal assessments. The plain-language story is template + structured
   evidence only. (The detr 2026-07-04 fixture is the counterexample this
   guards against: the halt-judge's rationale asserted "another 900s
   dispatch timeout" while the dispatch had COMPLETED, fast, with zero
   writes — and that guess went to the researcher as the headline.)

Considered and rejected: an LLM rewrite pass at halt time. The detr fixture
IS the failure mode (a model plausibly asserting an unevidenced mechanism),
and halts frequently fire precisely when dispatch infrastructure is
unhealthy, so the rewrite dispatch would fail exactly when needed most.
Deterministic templates from a small catalog are auditable and testable.

Consumed by BOTH researcher surfaces — `_render_halt_notice` in
run_pipeline.py (the TUI notice) and `_halt_block` in render_run_report.py
(the REPORT.md stopped-run block) — so the two never tell different
stories. Stdlib-only, imports nothing from the driver.

This module changes only what a stopped run says about itself; halt policy
(what halts, when) is untouched.
"""

from __future__ import annotations

from dataclasses import dataclass


# Plain-language activity per stage, for headlines (failure-path spec §3:
# "no stage ids in the headline"). Owned here so the TUI notice and
# REPORT.md share one gloss; render_run_report re-exports it.
STAGE_ACTIVITY = {
    "stage_0": "preparing the paper for processing",
    "stage_1": "analyzing the paper's method",
    "stage_1x": "writing the method explanation",
    "stage_2a": "setting up the code package",
    "stage_2b": "generating the model and training code",
    "stage_2c": "implementing the paper's method",
    "stage_2d": "finalizing the code package",
    "stage_2x": "deriving the paper's parameter values",
    "stage_3a": "writing the demo notebook",
    "stage_3b": "rendering the demo notebook",
    "stage_3c": "running the demo notebook end to end",
    "stage_4": "reviewing the code against the paper",
    "stage_5": "routing the review findings",
}

_DEFAULT_ACTIVITY = "processing this paper"


def stage_activity(stage_id: str) -> str:
    """Plain-language gloss for a stage id, tolerant of suffixed ids
    ("stage_2b_arch") and unknown stages."""
    if stage_id in STAGE_ACTIVITY:
        return STAGE_ACTIVITY[stage_id]
    parts = stage_id.split("_")
    if len(parts) >= 2:
        prefix = "_".join(parts[:2])
        if prefix in STAGE_ACTIVITY:
            return STAGE_ACTIVITY[prefix]
    return _DEFAULT_ACTIVITY


@dataclass(frozen=True)
class HaltStory:
    """The researcher-facing parts the catalog owns.

    `what_happened` and `why_stopped` are complete sentences. For ordinary
    recoverable stops, `what_next` omits the concrete resume command so each
    surface can append its own. A terminal structured gap action replaces that
    generic recovery guidance entirely.

    `engineering_actionable`: True when the researcher's only real move is
    reporting to engineering (the surface may adjust its framing)."""

    halt_class: str
    what_happened: str
    why_stopped: str
    what_next: str
    engineering_actionable: bool


def terminal_gap_action(
    halt_class: str | None,
    context: dict | str | None,
) -> str | None:
    """Return action guidance for a terminal scope decision.

    ``unsupported_or_unclear`` is not a transient failure: its structured gap
    report may say to reject the input, clarify scope, or add coverage.  The
    renderer must preserve that recorded action instead of appending a generic
    resume instruction. If the report omitted the action text, an explicit
    missing-action fallback keeps the decision terminal and honest. Other gap
    decisions remain recoverable and keep the ordinary halt story.
    """
    # Current artifacts name paradigm_mismatch explicitly.  Older Stage-1
    # artifacts predate halt_class but already carry the same structured gap
    # report; that context is sufficient for a presentation-only backfill.
    if (
        halt_class not in {None, "paradigm_mismatch"}
        or not isinstance(context, dict)
    ):
        return None
    report = context.get("paradigm_gap_report")
    if not isinstance(report, dict):
        return None
    if report.get("decision") != "unsupported_or_unclear":
        return None
    action = str(report.get("recommended_next_action") or "").strip()
    if action:
        return action
    return (
        "The gap report did not record a next action. Review its evidence and "
        "choose an in-scope input or explicitly expand coverage before "
        "starting another run."
    )


# ---------------------------------------------------------------------------
# Evidence-bounded mechanism resolution (design rule 2)
# ---------------------------------------------------------------------------

# Evidence keys, all optional, produced by the halt site from values the
# driver already holds (DispatchResult + run events — never from a model's
# inference about infrastructure):
#   completed        bool   session.idle observed within the timeout
#   elapsed_s        float  wall time of the dispatch
#   timeout_s        float  the budget it ran under
#   writes_observed  int    files the agent actually wrote
#   attempts         int    consecutive attempts that failed this same way
#   error_kind       str    "transport" when the client raised on the wire
#   step_gloss       str    plain-language name of the failed step
#                           (e.g. "its analysis step"); default "the step"
#   failing_location str    plain prose locator of the failing spot
#                           (e.g. "the matching step, cell 45 of the
#                           notebook") — used by fix_loop_exhausted
#   pending_finding  str    what the dead dispatch was FOR — the part the
#                           researcher can actually reason about (the maintainer's
#                           07-07 mock-run lesson) — used by
#                           transport_failure


def resolve_mechanism(evidence: dict | None) -> str | None:
    """Map structured dispatch evidence to one provable mechanism phrase.

    Returns a verb phrase for the failed step ("<step_gloss> <phrase>"), or
    None when no evidence field settles the mechanism — in which case the
    story says what is known and stops, never guessing. Priority: a
    transport error outranks timing (the wire failing explains everything
    downstream), timing outranks the no-write read (a turn cut off at the
    budget never got the chance to write)."""
    if not evidence:
        return None
    if str(evidence.get("error_kind") or "") == "transport":
        return "could not reach the model server (the connection failed)"
    elapsed = evidence.get("elapsed_s")
    timeout = evidence.get("timeout_s")
    if (
        isinstance(elapsed, (int, float)) and isinstance(timeout, (int, float))
        and timeout > 0 and elapsed >= timeout
    ):
        return "ran out of its time budget and was cut off (a timeout)"
    if evidence.get("completed") is True and evidence.get("writes_observed") == 0:
        attempts = evidence.get("attempts")
        if isinstance(attempts, int) and attempts >= 2:
            times = "twice" if attempts == 2 else f"{attempts} times in a row"
            return (
                f"finished {times} without producing its file "
                "(a known long-analysis failure mode, not a timeout)"
            )
        return "finished its turn without producing its file"
    return None


_UNSETTLED_MECHANISM = (
    "did not produce its file; the run log has the details"
)


# ---------------------------------------------------------------------------
# The closed class catalog (design rule 1)
# ---------------------------------------------------------------------------

# Each entry: (what_happened_detail, why_stopped, what_next,
# engineering_actionable). `{mechanism}` in why_stopped is filled from
# resolve_mechanism (or the honest unsettled phrase); `{step}` from
# evidence step_gloss. Wording rules: complete sentences, no stage ids, no
# script names, no probe/finding codes — those live in the technical
# collapsible each surface already renders.

_CLASSES: dict[str, tuple[str, str, str, bool]] = {
    "bad_input_file": (
        "The input paper file could not be turned into usable text.",
        "Every later step reads the paper's text, so we stop rather than "
        "analyze a garbled or empty paper.",
        "Check the input file: is it the right PDF, does it open, is it "
        "machine-readable text rather than page scans? Replace it and "
        "re-run.",
        False,
    ),
    "paradigm_mismatch": (
        "The document was understood, but its contribution could not be "
        "routed to a registered code-generation recipe.",
        "This is a coverage or routing decision, not a document-parsing "
        "failure. There is no vetted build-and-check route for this "
        "contribution, and forcing it through the closest wrong route risks "
        "code that silently does not represent the document.",
        "Nothing on your side is wrong. This is a system coverage or routing "
        "decision; the run report carries the gap report's action when one "
        "was recorded.",
        True,
    ),
    "not_feasible": (
        "The paper's core method cannot be faithfully reproduced at demo "
        "scale.",
        "The run records exactly which requirement blocks reproduction. We "
        "stop rather than deliver code that quietly substitutes something "
        "else for the paper's method.",
        "Read the feasibility explanation in the report. If a listed "
        "blocker is genuinely acceptable to approximate, record that and "
        "re-run; otherwise a faithful demo of this paper is out of reach.",
        False,
    ),
    "gap_pack_rejected": (
        # Wording covers BOTH live shapes of this class: a drafted recipe
        # that failed its checks, and a clean recipe the system cannot yet
        # build from (SRL 2026-07-13: the pack validated with zero errors
        # and the missing piece was the unbuilt template half — the old
        # "did not pass its own checks" copy was false for that run, and
        # the promised "draft recipe in the report" never rendered).
        "The paper's method family is new to the system, and the build "
        "recipe the run drafted for it on the fly is not yet something "
        "the system can safely build from.",
        "Building from a recipe the system cannot yet fully vet or "
        "assemble risks silently wrong code, so we stop rather than guess "
        "at how this method family should be put together.",
        "Nothing on your side is wrong. This is a system coverage gap; "
        "the run keeps the drafted recipe on disk for engineering, and "
        "the technical detail names the exact missing piece.",
        True,
    ),
    "driver_exception": (
        # The last-resort stage guard (stage-exception-guard-design.md):
        # an unhandled exception escaped a stage function. BY DEFINITION
        # ours — this class should page engineering like a pipeline_bug
        # judge classification.
        "The run hit an internal error in the pipeline itself and stopped.",
        "This is a pipeline bug, not a problem with your paper. Continuing "
        "past an internal error risks silently wrong output, so the run "
        "packaged what it had and stopped.",
        "Resume the run; completed steps are never redone. If it stops "
        "here twice, send the run folder to engineering. Everything the "
        "run finished before the error is listed below and stays usable.",
        True,
    ),
    "producer_wrote_nothing": (
        "{mechanism}.",
        "The built-in retries were used, and continuing would mean "
        "building on a file that does not exist.",
        "Resume the run; steps that already completed are never redone. If "
        "it stops at the same place twice, report it to engineering with "
        "the technical detail attached.",
        False,
    ),
    "producer_output_invalid": (
        "A generated file kept failing its checks.",
        "The built-in repair attempts were used without producing a "
        "version that passes, and we do not pass known-bad work "
        "downstream.",
        "Resume the run to try this step again. If it stops here twice, "
        "report it to engineering with the technical detail attached.",
        False,
    ),
    "fix_loop_exhausted": (
        "The automated repair attempts for a failing check were used up "
        "without converging.",
        "Repeating the same attempt was unlikely to succeed, so the run "
        "stopped rather than burn more attempts on a stuck loop.",
        "The technical detail names the failing spot. If you (or an AI "
        "assistant) fix or replace that one piece, resuming picks up from "
        "this check; completed steps are never redone.",
        False,
    ),
    "judge_halt": (
        "An automated reviewer examined a repeated failure and judged that "
        "another automatic repair attempt would not help.",
        "Its assessment is preserved in the technical detail, labeled as "
        "an internal note — it is often right about the code while wrong "
        "about the cause, so it is kept out of this summary.",
        "If the internal note points at something in the paper or your "
        "inputs you can adjust, do so and re-run. Otherwise report it to "
        "engineering with the technical detail attached.",
        False,
    ),
    "out_of_scope_write": (
        "An internal step wrote files outside the area it owns.",
        "That can corrupt the run's records, so we stop and protect what "
        "was already produced rather than continue from a possibly "
        "tainted state.",
        "Resume the run — this is usually transient. If it happens again, "
        "report it to engineering.",
        True,
    ),
    "transport_failure": (
        "The connection to the model server failed, so a required step "
        "never ran.{pending}",
        "This is an infrastructure problem, not a problem with your paper; "
        "retrying blindly while the server is unhealthy would only pile up "
        "failures.",
        "Resume the run once the model server is reachable; completed "
        "steps are never redone. If it keeps happening, engineering needs "
        "to look at the server.",
        True,
    ),
    "internal_contract_violation": (
        "An internal step broke one of the run's own rules.",
        "This is a system bug, not anything about your paper. We stop "
        "rather than continue from a state the system does not "
        "understand.",
        "Report this to engineering with the technical detail attached. "
        "Resuming may clear it, but the bug is worth filing either way.",
        True,
    ),
    "needs_user_input": (
        "The run reached a decision only you can make.",
        "Guessing here without you could produce a demo that quietly "
        "answers the wrong question.",
        "Answer the question in the message above, then resume the run; "
        "completed steps are never redone.",
        False,
    ),
}

HALT_CLASSES = frozenset(_CLASSES)

# Classes whose template already asserts the mechanism in prose; appending
# the resolved phrase would say the same thing twice.
_MECHANISM_IN_PROSE = frozenset({"transport_failure"})


def render_halt_story(
    halt_class: str | None,
    *,
    stage_id: str,
    evidence: dict | None = None,
) -> HaltStory | None:
    """Render the four-part story for a classified halt.

    Returns None for an unknown or missing class — callers fall back to
    their existing stock rendering, so old halt artifacts (no `halt_class`
    field) render exactly as before."""
    if not halt_class or halt_class not in _CLASSES:
        return None
    detail, why, what_next, engineering = _CLASSES[halt_class]
    evidence = evidence or {}

    step = str(evidence.get("step_gloss") or "the step")
    resolved = resolve_mechanism(evidence)
    mechanism = resolved or _UNSETTLED_MECHANISM
    pending = ""
    raw_pending = str(evidence.get("pending_finding") or "").strip()
    if raw_pending:
        pending = (
            f" The step that never ran was going to {raw_pending} — that "
            "unresolved question is still open, and it is the part worth "
            "reading before resuming."
        )
    mechanism_sentence = f"{step} {mechanism}"
    mechanism_sentence = mechanism_sentence[0].upper() + mechanism_sentence[1:]
    template_uses_mechanism = "{mechanism}" in detail
    detail = detail.format(mechanism=mechanism_sentence, pending=pending)

    what_happened = (
        f"We stopped while {stage_activity(stage_id)}. {detail}"
    )
    failing_location = str(evidence.get("failing_location") or "").strip()
    if failing_location:
        what_happened += (
            f" The failure sits in one place: {failing_location}."
        )
    # A mechanism the evidence proves is worth stating even when the class
    # template has no slot for it (the detr 2026-07-04 fixture: the fix
    # loop halted because the debugger's analysis step finished twice with
    # zero writes — that provable fact IS the story). Unsettled mechanisms
    # are never appended; the story says what is known and stops.
    if (
        resolved is not None and not template_uses_mechanism
        and halt_class not in _MECHANISM_IN_PROSE
    ):
        what_happened += f" {mechanism_sentence}."
    return HaltStory(
        halt_class=halt_class,
        what_happened=what_happened,
        why_stopped=why,
        what_next=what_next,
        engineering_actionable=engineering,
    )


def _self_test() -> None:
    # Every class renders completely, with no leftover format placeholders
    # and no stage ids in the story.
    for name in sorted(HALT_CLASSES):
        story = render_halt_story(name, stage_id="stage_3c")
        assert story is not None, name
        blob = " ".join(
            [story.what_happened, story.why_stopped, story.what_next])
        assert "{" not in blob and "}" not in blob, (name, blob)
        assert "stage_3c" not in blob, (name, blob)

    # Unknown / missing class → None (old artifacts render as before).
    assert render_halt_story(None, stage_id="stage_1") is None
    assert render_halt_story("no_such_class", stage_id="stage_1") is None

    # The detr 2026-07-04 evidence shape: dispatch completed, well under
    # budget, zero writes, twice. The mechanism must read no-write — not
    # the judge's guessed timeout.
    detr = {"completed": True, "elapsed_s": 412.0, "timeout_s": 900.0,
            "writes_observed": 0, "attempts": 2}
    phrase = resolve_mechanism(detr)
    assert phrase is not None and "without producing its file" in phrase
    assert "twice" in phrase and "not a timeout" in phrase
    assert "timed out" not in phrase

    # A real timeout reads as one.
    assert "timeout" in resolve_mechanism(
        {"completed": False, "elapsed_s": 900.2, "timeout_s": 900.0})

    # Transport outranks everything.
    assert "connection failed" in resolve_mechanism(
        {"error_kind": "transport", "completed": True,
         "writes_observed": 0})

    # No settling evidence → None, never a guess.
    assert resolve_mechanism({"completed": True, "writes_observed": 3}) is None
    assert resolve_mechanism({}) is None
    assert resolve_mechanism(None) is None

    print("halt_catalog self-test OK")


if __name__ == "__main__":
    _self_test()
