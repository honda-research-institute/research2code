"""Deterministic verbatim-quote repair for the method-spec floors.

Every positive case here is the REAL recorded shape from the 0728b
bayesian-active-learning fresh roll: the producer holds the correct
passage but inserts spaces inside math delimiters (`$ R_0 $` for the
paper's `$R_0$`), the whitespace-normalized floor rightly fails, and
the producer demonstrably cannot byte-fix the difference across
retries — the run burned all 3 stage-1 fix-loop retries on exactly
this class and halted. Render-equivalent re-anchoring repairs it
deterministically; a paraphrase must NOT repair and stays on the
honest fix-loop path; near-match adoption (R2C-030's rule, extended
to this surface) lands loud on every surface.
"""

from __future__ import annotations

import json
import subprocess

from tests.helpers.state import make_state

# The recorded bayesian-active-learning meaning-quote shapes (0728b),
# lightly condensed. The paper's own bytes keep the run's double-space
# habit around inline math; the producer's transcription inserts spaces
# INSIDE the math delimiters, which survives whitespace normalization
# and fails the floor.
_R0_PASSAGE = (
    "Eq. (5) has one parameter  $R_0$  which describes the geometric "
    "prior from probability. The default radius of the intern balls  "
    "$R_0$  is used to legalize the prior and has no further influences "
    "on the sampling process."
)
_ETA_PASSAGE = (
    "Ellipsoid geodesic is adjusted by  $\\eta$  which controls how far "
    "of the updates of core-set to the boundaries of distributions."
)
_NM_PASSAGE = (
    "Let b=1000,b'=500 in GBALD,  $\\mathcal{N}_{\\mathcal{M}}$  be the "
    "number of the core-set size, the iteration budget  $\\mathcal{A}$  "
    "of GBALD can then be defined as the remaining acquisition rounds."
)
_SCENARIO_PASSAGE = (
    "The obstacle is assumed to have a circular shape with a defined "
    "radius boundary  $r_{obs}$  around its center point."
)


def _space_math(passage: str, *tokens: str) -> str:
    """The recorded defect class: spaces inserted inside `$...$`."""
    out = passage
    for token in tokens:
        out = out.replace(f"${token}$", f"$ {token} $")
    assert out != passage
    return out


_R0_QUOTE = _space_math(_R0_PASSAGE, "R_0")
_ETA_QUOTE = _space_math(_ETA_PASSAGE, "\\eta")
_NM_QUOTE = _space_math(
    _NM_PASSAGE, "\\mathcal{N}_{\\mathcal{M}}", "\\mathcal{A}")
_SCENARIO_QUOTE = _space_math(_SCENARIO_PASSAGE, "r_{obs}")

# The R2C-030 near-match class on this surface: a word-level
# transcription slip the render-equivalence fold cannot absorb (letter
# identity differs), with exactly one near-perfect candidate.
_R0_SLIPPED = _R0_PASSAGE.replace("geometric prior", "geometrical prior")
assert _R0_SLIPPED != _R0_PASSAGE

_PARAPHRASE = (
    "The radius parameter of the interior balls only normalizes the "
    "geometric prior and does not otherwise affect how samples are "
    "acquired by the algorithm."
)

PAPER = (
    "# 3 Method\n\n"
    "## 3.2 Geometric prior\n\n"
    + _R0_PASSAGE + "\n\n" + _ETA_PASSAGE + "\n\n"
    "## 3.3 Acquisition budget\n\n"
    + _NM_PASSAGE + "\n\n" + _SCENARIO_PASSAGE + "\n\n"
    "## 3.4 Discussion\n\n"
    "Unrelated closing prose about experiments and datasets, long "
    "enough to form its own region for containment checks.\n"
)


def _nws(t: str) -> str:
    return " ".join((t or "").split())


def _glossary_entry(name: str, quote: str, section: str) -> dict:
    return {"name": name, "aliases": [], "meaning_quote": quote,
            "paper_section": section}


def _write_fixtures(state, glossary=None, scenario=None) -> None:
    (state.paths.pipeline_dir / "paper.md").write_text(
        PAPER, encoding="utf-8")
    spec: dict = {"critical_requirements": {
        "param_glossary": glossary if glossary is not None else []}}
    if scenario is not None:
        spec["scenario_assumptions"] = scenario
    state.paths.method_spec.write_text(
        json.dumps(spec, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8")


def _read_spec(state) -> dict:
    return json.loads(state.paths.method_spec.read_text(encoding="utf-8"))


# --- The recorded class: whitespace inside math delimiters ---------------

def test_whitespace_in_math_meaning_quotes_reanchor(tmp_path):
    import run_pipeline

    state = make_state(tmp_path / "run")
    _write_fixtures(state, glossary=[
        _glossary_entry("R_0", _R0_QUOTE, "Section 3.2"),
        _glossary_entry("eta", _ETA_QUOTE, "Section 3.2"),
        _glossary_entry("N_M", _NM_QUOTE, "Section 3.3"),
    ])
    # The fixtures reproduce the halt: all three fail the floor as-is.
    spec = _read_spec(state)
    assert len(run_pipeline._spec_quote_floor_failures(spec, PAPER)) == 3

    reanchored, adopted = run_pipeline._repair_failing_spec_quotes(state)

    assert (reanchored, adopted) == (3, 0)
    fixed = _read_spec(state)
    quotes = [e["meaning_quote"]
              for e in fixed["critical_requirements"]["param_glossary"]]
    assert quotes == [_R0_PASSAGE, _ETA_PASSAGE, _NM_PASSAGE]
    # The repaired quotes pass the floor by construction (paper bytes).
    assert run_pipeline._spec_quote_floor_failures(fixed, PAPER) == []
    for quote in quotes:
        assert _nws(quote) in _nws(PAPER)
    # Re-anchoring is disclosure-light: a run event, no assumption, no
    # adoption sidecar — the same pattern as the equation surface.
    events = (state.paths.pipeline_dir / "run_events.jsonl").read_text(
        encoding="utf-8")
    assert "spec_quotes_reanchored" in events
    assert "spec_quotes_adopted" not in events
    assert not (state.paths.pipeline_dir / "quote_adoptions.json").exists()
    import run_layout
    assert not (run_layout.run_path(
        state.paths.run_dir, run_layout.ASSUMPTIONS_MD)).exists()


def test_paraphrase_does_not_repair_and_reaches_fix_loop(tmp_path):
    import run_pipeline

    state = make_state(tmp_path / "run")
    _write_fixtures(state, glossary=[
        _glossary_entry("R_0", _R0_QUOTE, "Section 3.2"),
        _glossary_entry("R_0_paraphrase", _PARAPHRASE, "Section 3.2"),
    ])

    reanchored, adopted = run_pipeline._repair_failing_spec_quotes(state)

    # The render-equivalent quote repairs; the paraphrase must not —
    # no re-anchor span exists and adoption is far below the floor.
    assert (reanchored, adopted) == (1, 0)
    fixed = _read_spec(state)
    by_name = {e["name"]: e
               for e in fixed["critical_requirements"]["param_glossary"]}
    assert by_name["R_0"]["meaning_quote"] == _R0_PASSAGE
    assert by_name["R_0_paraphrase"]["meaning_quote"] == _PARAPHRASE
    # The paraphrase still fails the floor, so the validator re-run
    # fails and the enriched fix loop stays the honest path.
    remaining = run_pipeline._spec_quote_floor_failures(fixed, PAPER)
    assert len(remaining) == 1
    assert "R_0_paraphrase" in remaining[0]["key"]
    assert not (state.paths.pipeline_dir / "quote_adoptions.json").exists()


def test_pure_paraphrase_leaves_spec_untouched(tmp_path):
    import run_pipeline

    state = make_state(tmp_path / "run")
    _write_fixtures(state, glossary=[
        _glossary_entry("R_0_paraphrase", _PARAPHRASE, "Section 3.2"),
    ])
    before = state.paths.method_spec.read_text(encoding="utf-8")

    assert run_pipeline._repair_failing_spec_quotes(state) == (0, 0)

    # No repairs means no rewrite: the fix loop sees the producer's
    # own bytes, not a partially-normalized file.
    assert state.paths.method_spec.read_text(encoding="utf-8") == before
    assert not (state.paths.pipeline_dir / "run_events.jsonl").exists()


# --- R2C-030 adoption extended to the spec surface ------------------------

def test_adoption_on_spec_quote_writes_loud_surfaces(tmp_path, monkeypatch):
    import run_pipeline
    from quote_adopt import ADOPTION_FLOOR

    # Pinned on explicitly: this test documents the gated behavior and
    # must stay green whichever default the constant ships with.
    monkeypatch.setattr(
        run_pipeline, "SPEC_QUOTE_ADOPTION_ENABLED", True)
    state = make_state(tmp_path / "run")
    _write_fixtures(state, glossary=[
        _glossary_entry("R_0", _R0_SLIPPED, "Section 3.2"),
    ])

    reanchored, adopted = run_pipeline._repair_failing_spec_quotes(state)

    assert (reanchored, adopted) == (0, 1)
    fixed = _read_spec(state)
    entry = fixed["critical_requirements"]["param_glossary"][0]
    assert entry["meaning_quote"] == _R0_PASSAGE
    assert run_pipeline._spec_quote_floor_failures(fixed, PAPER) == []

    # Loud on every surface: run event, assumptions.md entry with the
    # original quote preserved in the sidecar, and the sidecar row
    # carries the surface marker so the report row can tell the
    # artifacts apart.
    events = (state.paths.pipeline_dir / "run_events.jsonl").read_text(
        encoding="utf-8")
    assert "spec_quotes_adopted" in events
    sidecar = json.loads(
        (state.paths.pipeline_dir / "quote_adoptions.json").read_text(
            encoding="utf-8"))
    assert len(sidecar) == 1
    row = sidecar[0]
    assert row["element_id"] == "param_glossary['R_0'].meaning_quote"
    assert row["surface"] == "method_spec"
    assert row["original_quote"] == _R0_SLIPPED
    assert row["adopted_passage"] == _R0_PASSAGE
    assert row["similarity"] >= ADOPTION_FLOOR
    import run_layout
    assumptions = run_layout.run_path(
        state.paths.run_dir, run_layout.ASSUMPTIONS_MD).read_text(
        encoding="utf-8")
    assert "param_glossary['R_0'].meaning_quote" in assumptions
    assert row["assumption_id"] in assumptions


def test_adoption_gate_constant_keeps_surface_reanchor_only(
        tmp_path, monkeypatch):
    import run_pipeline

    monkeypatch.setattr(
        run_pipeline, "SPEC_QUOTE_ADOPTION_ENABLED", False)
    state = make_state(tmp_path / "run")
    _write_fixtures(state, glossary=[
        _glossary_entry("R_0", _R0_SLIPPED, "Section 3.2"),
    ])

    assert run_pipeline._repair_failing_spec_quotes(state) == (0, 0)
    entry = _read_spec(state)["critical_requirements"]["param_glossary"][0]
    assert entry["meaning_quote"] == _R0_SLIPPED
    assert not (state.paths.pipeline_dir / "quote_adoptions.json").exists()


# --- The second floor: scenario-assumption evidence quotes ----------------

def test_scenario_assumption_dict_shape_reanchors(tmp_path):
    import run_pipeline

    state = make_state(tmp_path / "run")
    _write_fixtures(state, scenario={
        "obstacle_geometry": {
            "normalized_value": "circle",
            "evidence_quote": _SCENARIO_QUOTE,
            "paper_location": "Section 3.3",
        },
    })

    assert run_pipeline._repair_failing_spec_quotes(state) == (1, 0)
    fixed = _read_spec(state)
    assert (fixed["scenario_assumptions"]["obstacle_geometry"]
            ["evidence_quote"] == _SCENARIO_PASSAGE)
    events = (state.paths.pipeline_dir / "run_events.jsonl").read_text(
        encoding="utf-8")
    assert "scenario_assumptions['obstacle_geometry'].evidence_quote" \
        in events


def test_scenario_assumption_list_shape_is_handled(tmp_path):
    # The schema says dict (dimension_id -> entry); a list-shaped
    # container from a malformed producer artifact must degrade to a
    # repair keyed by dimension_id, never a crash inside the repair.
    import run_pipeline

    state = make_state(tmp_path / "run")
    _write_fixtures(state, scenario=[
        {"dimension_id": "obstacle_geometry",
         "normalized_value": "circle",
         "evidence_quote": _SCENARIO_QUOTE,
         "paper_location": "Section 3.3"},
    ])

    assert run_pipeline._repair_failing_spec_quotes(state) == (1, 0)
    fixed = _read_spec(state)
    assert (fixed["scenario_assumptions"][0]["evidence_quote"]
            == _SCENARIO_PASSAGE)


# --- Noop and seam behavior ------------------------------------------------

def test_no_failing_quotes_is_a_noop(tmp_path):
    import run_pipeline

    state = make_state(tmp_path / "run")
    _write_fixtures(state, glossary=[
        _glossary_entry("R_0", _R0_PASSAGE, "Section 3.2"),
    ], scenario={
        "obstacle_geometry": {
            "normalized_value": "circle",
            "evidence_quote": _SCENARIO_PASSAGE,
            "paper_location": "Section 3.3",
        },
    })
    before = state.paths.method_spec.read_text(encoding="utf-8")

    assert run_pipeline._repair_failing_spec_quotes(state) == (0, 0)

    assert state.paths.method_spec.read_text(encoding="utf-8") == before
    assert not (state.paths.pipeline_dir / "quote_adoptions.json").exists()
    assert not (state.paths.pipeline_dir / "run_events.jsonl").exists()


def _fake_floor_validator(run_pipeline):
    """A run_script stand-in that applies the REAL floor semantics (via
    the module's own in-process re-check) to the CURRENT file, in the
    real validator's stderr format. Lets the seam test exercise
    validate -> repair -> re-validate without a subprocess."""
    calls: list[int] = []

    def fake_run_script(stage_id, args, *, timeout=300):
        calls.append(1)
        spec = json.loads(run_pipeline.Path(args[1]).read_text(
            encoding="utf-8"))
        paper_text = None
        if "--paper-md" in args:
            paper_text = run_pipeline.Path(
                args[args.index("--paper-md") + 1]).read_text(
                encoding="utf-8")
        failures = (run_pipeline._spec_quote_floor_failures(
            spec, paper_text) if paper_text is not None else [])
        if failures:
            lines = "\n".join(f"  - {f['key']}: not a verbatim passage"
                              for f in failures)
            return subprocess.CompletedProcess(
                args=args, returncode=1, stdout="",
                stderr=("param-glossary meaning-quote floor failed:\n"
                        + lines + "\n"))
        return subprocess.CompletedProcess(
            args=args, returncode=0, stdout="ok\n", stderr="")

    return fake_run_script, calls


def test_spec_validator_seam_repairs_then_passes(tmp_path, monkeypatch):
    import run_pipeline

    state = make_state(tmp_path / "run")
    _write_fixtures(state, glossary=[
        _glossary_entry("R_0", _R0_QUOTE, "Section 3.2"),
        _glossary_entry("eta", _ETA_QUOTE, "Section 3.2"),
        _glossary_entry("N_M", _NM_QUOTE, "Section 3.3"),
    ])
    fake, calls = _fake_floor_validator(run_pipeline)
    monkeypatch.setattr(run_pipeline, "run_script", fake)

    ok, err_tail = run_pipeline._run_spec_validator(state)

    # Validate (fail) -> deterministic repair -> re-validate (pass):
    # the terminal gate is the validator itself, run twice.
    assert ok is True
    assert len(calls) == 2
    fixed = _read_spec(state)
    assert run_pipeline._spec_quote_floor_failures(fixed, PAPER) == []
    events = (state.paths.pipeline_dir / "run_events.jsonl").read_text(
        encoding="utf-8")
    assert "spec_quotes_reanchored" in events
    assert "validation_passed" in events


def test_spec_validator_seam_paraphrase_still_fails_to_fix_loop(
        tmp_path, monkeypatch):
    import run_pipeline

    state = make_state(tmp_path / "run")
    _write_fixtures(state, glossary=[
        _glossary_entry("R_0_paraphrase", _PARAPHRASE, "Section 3.2"),
    ])
    fake, calls = _fake_floor_validator(run_pipeline)
    monkeypatch.setattr(run_pipeline, "run_script", fake)

    ok, err_tail = run_pipeline._run_spec_validator(state)

    # Nothing repaired -> no re-run -> the failure flows to the fix
    # loop exactly as before this change.
    assert ok is False
    assert len(calls) == 1
    assert "floor failed" in err_tail
    assert "R_0_paraphrase" in err_tail
