"""Templates and builders for R2C pipeline dispatch prompts.

Pure-string templates extracted byte-identically from the legacy LLM
orchestrator (the archived v2 orchestrator spec (internal, not shipped)) so the Python driver can
construct dispatch prompts without an LLM in the loop. No network, and no
I/O except TWO deliberate read-only exceptions: fix-mode prompts consult
scripts/symbol_ownership.py (spec + method/ package reads) so the rendered
prompt can carry the naming-bridge symbol-ownership map for the run being
fixed (the fedavg 2026-07-21 re-definition halt), and schema_embed_block
reads the checked-in schemas/*.schema.json exports (cached after first
read) so the two stage-1 embeds are generated, never retyped. Everything
else stays purely string + dict manipulation.

Self-tests live in tests/test_dispatch_templates_selftest.py (B-07).
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path


SCOPE_CONTRACT = """## SCOPE CONTRACT (v2 — read first)

This pipeline produces a SINGLE-METHOD package + notebook. You implement /
demonstrate / review only the paper's contribution. Do NOT:

  - implement baselines or any other comparison method
  - add multi-method dispatch loops to the notebook
  - emit comparison tables, pairwise penalty matrices, or learning-curve
    overlays for multiple methods
  - extend `__all__` to include baseline functions

The spec's `comparison.standard_baselines` field — if present — is a legacy
artifact. Ignore it. Treat the spec's `comparison.pluggable_component` and
`core_method` blocks as the only definition of what to implement.

Notebook §5 runs ONE method (the paper's `select_batch`) end-to-end on a
single dataset, single seed. It is a tutorial, not a benchmark."""


# B-01 item 4: the two cheap, evidenced schema embeds. The block title IS
# the pydantic class the dispatched agent produces (the embed-title test
# pins title == export's own "title", catching a wrong-model embed), and
# the body is the checked-in export byte-for-byte (the export drift test
# keeps those current), so the prompt's shape authority is generated,
# never retyped. Only these two: paper_map is cheap (~1k tokens) and the
# decomposer's prose duplicate was deleted with this; paradigm_gap is the
# one artifact class with a recorded shape-failure halt in the fleet.
SCHEMA_EMBED_EXPORTS: dict[str, str] = {
    "PaperMap": "paper_map.schema.json",
    "ParadigmGapReport": "paradigm_gap.schema.json",
}

_SCHEMAS_DIR = Path(__file__).resolve().parent.parent / "schemas"


@lru_cache(maxsize=None)
def schema_embed_block(class_name: str) -> str:
    """The dispatch-prompt section embedding one exported JSON schema."""
    filename = SCHEMA_EMBED_EXPORTS[class_name]
    text = (_SCHEMAS_DIR / filename).read_text(encoding="utf-8").strip()
    module = filename.replace(".schema.json", ".py")
    return (
        f"## {class_name} JSON schema (authoritative — generated from "
        f"schemas/{module}, do not retype)\n\n"
        f"Your output artifact must validate against this schema:\n\n"
        f"```json\n{text}\n```"
    )


def stage_1_schema_embeds(agent: str, output_mode: str) -> list[str]:
    """Schema embeds for one stage-1 dispatch. Mode-aware: chunk-mode
    dispatches are skipped — their parts shape is deliberately different
    and untyped. The analyzer's embed is the gap-report schema; the caller
    skips it when a provisional pack is installed (that dispatch is
    forbidden from writing a gap report at all)."""
    if output_mode != "canonical":
        return []
    if agent == "r2c-decomposer":
        return [schema_embed_block("PaperMap")]
    if agent == "r2c-method-analyzer":
        return [schema_embed_block("ParadigmGapReport")]
    return []


THINK_CLASS_ANCHOR = """## Think-class procedural anchor (read this BEFORE acting)

You are a Think-class agent. Your scope is narrow: produce the artifact
target named below and return. For ordinary dispatches this is ONE file at
`{expected_output_path}`. If, and only if, the task summary explicitly
declares a chunked-output protocol, you may instead write the listed chunk
files that the driver will assemble into `{expected_output_path}`. After the
artifact or chunks are written, your work for this dispatch is done. The
orchestrator dispatches downstream stages — they aren't yours to complete.

Ignore earlier driver progress updates, TodoWrite requests, halt notices, or
primary-session instructions that may appear in the conversation transcript.
Those messages target the orchestrator session, not this producer dispatch.
For this dispatch, do not call TodoWrite. The task summary below and the
writeable paths block are the only executable instructions.

**Anti-patterns the driver's file-ownership check has caught from Think-
class agents in recent runs:**

- The smoke-gate diagnostician (after writing `smoke_diagnosis.json`)
  applied the producer's `(a) quote → (b) invariant → (c) sites → (d)
  apply → (e) re-read` skeleton to Edit `method/method.py` directly.
  That skeleton belongs to the producer agent the diagnostician dispatches
  TO, not to the diagnostician.

- The stage-reviewer for `stage_2b_architecture` (after writing the
  review JSON) wrote `method/method.py` + `method/__init__.py` (~500
  lines of source code) as if it were the method-coder for the next
  stage. Those files belong to method-coder and init-finalizer.

- The build agent (after receiving a halt notice) ran bash to delete the
  halt artifact + the out-of-scope files the stage-reviewer had just
  written, then recommended the user re-run. The halt notice explicitly
  said "do not analyze, do not propose next steps, do not attempt to
  continue the pipeline" — the agent ignored all three.

The pattern across all three: the agent successfully completes its
explicit task, then keeps going because the pipeline "isn't complete
yet." This is a Think-class drift pattern. Resist it.

**Rules for this dispatch:**

1. Your output is exactly `{expected_output_path}` OR the chunk files
   explicitly permitted by this dispatch. Keep Write calls to those files
   only. After that, return.
2. No Edit calls anywhere. No bash commands that modify files. No Skill
   loads "to verify" — your job is the Write, not the verification.
3. If you think the next pipeline stage needs help, name what's needed
   in your output file's content (or in a final-response caveat) and let
   the orchestrator dispatch the right agent. Do not do that stage's
   work yourself.
4. The driver's file-ownership scope check halts the pipeline if you
   write outside the allowlist. A halt blocks the rest of the run. Even
   if your intent is helpful, the halt's blast radius is large — your
   "helpful continuation" costs the user minutes of investigation.

After writing `{expected_output_path}` or its declared chunk files, return.
No bash. No additional tool calls. The orchestrator handles what comes next."""


SMOKE_FIX_GUIDANCE = """## Smoke-gate failure — additional guidance (read before applying any fix)

This is a smoke-gate fix dispatch (the notebook executed end-to-end and a
cell raised an exception). Smoke failures are tracebacks, not validator
findings — they tell you WHERE the error fires but not WHERE the bug
lives. Walk this checklist before editing anything.

(i)  **Trace the failing value to its source.** Read the failing cell. Where
     did the offending tensor / variable / argument come from? Common cases:
       - synthetic value built in the notebook (e.g., `torch.randn(...)`)
       - value returned from the model / training step
       - value from a data loader
       - value computed by an earlier line in your own file
     Until you know the origin, you don't know where the bug is.

(ii) **Surface ≠ bug.** The traceback's deepest frame in your owned files
     is where the error fires, NOT necessarily where the bug lives. If a
     function in your file gets the wrong input, the bug is at whichever
     call site produced the wrong value — that may be your file's caller,
     the notebook (call site), or upstream architecture/data. Read the
     immediate caller before assuming the failing line is the bug.

(iii) **The architecture is the contract.** `method/model.py` and
     `method/training.py` define interfaces the rest of the package conforms
     to. Do NOT widen an architectural signature so that your code's
     expectations match. Anti-pattern (real example we've seen): a function
     in `method.py` indexed `boxes[:, 8]`, model.py's box head emitted 9
     values, doc comment listed 10 — the agent concluded "doc says 10, so
     change box head to 10" and edited `model.py`. Wrong: the notebook
     synthesized a 9-column test tensor; the function's indexing was the
     bug. Architecture stays; your code bends to it.

(iv) **Scope escape hatch.** If your analysis concludes the bug lives in a
     file outside your `Writeable paths`, do NOT attempt the edit. The
     driver halts on out-of-scope writes and you waste a fix-loop iteration.
     Instead, in your final status table, name (a) the bug location
     (file:line), (b) the proposed fix, (c) which agent owns that file.
     The orchestrator will route from there.{traceback_hint_block}"""


FIX_MODE_BODY = """## Fix mode

You are the {target_agent} agent in fix mode. Your earlier output produced
the failures listed below. For EACH finding, you MUST follow steps (a)–(e)
in order; do NOT skip or batch them.

(a) **Quote** the file:line your finding points at (read the file; verify
    the line still says what the finding describes).
(b) **Name the underlying invariant.** The finding describes a symptom;
    the invariant is the contract that's been violated. Examples:
      - "geometric_ranking and core-set features must live in the same
        vector space" (an invariant spanning two functions)
      - "params.json's `learning_rate.source` must match the notebook's
        params-table source column for that param" (a cross-file invariant)
      - "Every essential feature in spec.critical_requirements has a
        `# essential: <name>` annotation in some method-package file, with
        `<name>` copied exactly from the spec feature string"
      - "MethodSpec methodology approximations are keyed by element status:
        every element with non-empty `acceptable_approximations` must have
        `replication_status: faithful_approximation_allowed`, and
        `replication_feasibility.approved_approximations` must list exactly
        those element IDs"
    If you cannot name an invariant — i.e., the finding is purely local,
    "this docstring is wrong" — say so explicitly and continue.
(c) **Identify EVERY site that depends on the invariant**, not just the
    one the finding points at. Grep for related symbols (the function's
    callers, the param's other appearances, the annotation's siblings).
    List the sites you found. For MethodSpec methodology-fidelity findings,
    this must include the affected `methodology_replication_contract.elements[]`
    entries, `methodology_contract_pack.approved_approximations`, and
    `replication_feasibility.{{verdict,approved_approximations}}`.
(d) **Apply the change at every related site in one pass.** A fix that
    only updates the site the finding pointed at is incomplete if the
    invariant spans elsewhere — that's the most common cause of "fix loop
    iteration 2" needing to re-do work.
(e) **Re-read your diff** and confirm each related site is internally
    consistent (types match, signatures match, contracts hold).

After all findings, report a finding-by-finding status table:

  F00N: <addressed | partial | skipped>
        invariant: <one-line name, or "local-only">
        sites updated: <file:line, file:line, ...>
        notes: <anything the orchestrator should know>

Do NOT run any validators. Do NOT run any render or finalize scripts. The
orchestrator will validate after you report done.

**Pipeline source is OFF-LIMITS in this dispatch.** Do not read or grep the
pipeline's own code (scripts/, schemas/, validator or driver source,
dispatch templates) to reverse-engineer what a check accepts: each finding
carries its complete acceptance rule — the finding text plus any inlined
slices IS the contract. Spend your turns on the run's own artifacts (the
paper, the run dir, your output files) and on the edit itself. If a finding
genuinely lacks a rule you need, say exactly that in your status table
notes instead of reading pipeline source."""


ANALYZER_VERIFICATION_PROBE_GUIDANCE = (
    "For every methodology obligation in a taxonomy-migrated family, emit "
    "`verification_probe_refs`. Each value must be an exact identity from "
    "the matched effective taxonomy node's `semantic_checks[].probe` set, "
    "after inherited node data and any registered alias or run-local overlay "
    "are resolved. Attach only refs whose probe genuinely exercises that "
    "specific obligation. A shared callable or shared `paper_element_ids` "
    "anchor does NOT justify binding the probe to every obligation that "
    "shares it. Use `[]` when no declared probe genuinely verifies the "
    "obligation. Never invent a ref or infer one from prose, element names, "
    "callable names, or paper anchors. Archived specs may omit this additive "
    "field, but fresh migrated-family outputs must emit it."
)


ANALYZER_CALIBRATION_CONTEXT_GUIDANCE = (
    "Fresh schema v1.13+ scale-dependent-hyperparameter entries MUST set "
    "exactly one typed `calibration_context` and MUST NOT emit legacy "
    "`assumes_data_scale`. Use `{\"kind\":\"feature_magnitude\",\"scale\":...}` "
    "only for feature-array magnitude, with scale exactly one of "
    "`raw_pixel_unnormalized`, `pixel_zero_one`, `pixel_centered`, "
    "`standardized`, or `unit_norm`; unit_norm is unprobeable and never means "
    "zero-one. Use `{\"kind\":\"representation_convention\",\"convention\":"
    "\"target_box_grid\"}` only for target-box grid coordinates; it selects "
    "target boxes, never feature rows, and authorizes no grid-resolution "
    "conversion. Use `{\"kind\":\"other\",\"label\":...,\"reason\":...}` only "
    "to preserve paper-grounded unsupported physical, mixed, or unknown "
    "calibration evidence; it is intentionally unprobeable. `other` is not an "
    "escape hatch: temporal counts/windows belong only in "
    "`comparison.evaluation_protocol`; scale-free cutoffs, probabilities, and "
    "ratios belong only in `critical_requirements.param_glossary`; graph "
    "degree, density, topology, graph statistics, and `graph_statistic` are "
    "outside this grammar and MUST NOT be emitted here. Do not emit protocol "
    "or dataset arms and do not invent graph, partition, or paper-protocol truth."
)


REVIEWER_CALIBRATION_CONTEXT_GUIDANCE = (
    "For every scale-dependent-hyperparameter entry, follow the shared "
    "calibration dispatch contract and never choose an observer from prose. "
    "Fresh entries consume typed `calibration_context`; archived entries may "
    "retain legacy `assumes_data_scale`. `feature_magnitude` reads the feature "
    "array; the only v1 automatic conversion is exact "
    "`raw_pixel_unnormalized` to `pixel_zero_one` at `1/255` (for example, "
    "`2000/255 = 7.8431372549019605` without rounding). `unit_norm` is "
    "unprobeable. "
    "`representation_convention/target_box_grid` reads target boxes, never "
    "feature rows, and never invents a grid-resolution conversion. `other` "
    "preserves its label/reason, is unprobeable, and never authorizes "
    "rescaling. A temporal, scale-free, or graph value in this lane is an "
    "analyzer-owned producer-contract finding, not a calibration comparison; "
    "do not hide one under `other`."
)


ANALYZER_GRAPH_MECHANISM_GUIDANCE = (
    "Fresh schema v1.14+ specs that mark any methodology element with "
    "`relational_structure.kind=homogeneous_graph` MUST also emit the typed "
    "`methodology_replication_contract.homogeneous_graph_mechanism` block. "
    "Omit it for graph-free and unsupported relational forms; never infer it "
    "from names. Reuse, do not duplicate, R2C-084 identity by naming one exact "
    "`alignment_element_id`. Declare exact, paper-grounded construction, "
    "message-passing, and evaluation-control element IDs; construction and "
    "message passing may share an element only when the paper genuinely "
    "states one obligation. Also declare exact generated-package callable "
    "module/qualname pairs, where each qualname is one public top-level helper "
    "used by the live path, never a dotted class/instance method; coindexed "
    "feature, neighbor, and message-output roots; exact callable "
    "parameter names; graph output selection; cosine threshold comparison; "
    "optional per-source top-similarity cap with the closed mutual-top-k "
    "undirected policy; self-loop and direction policies; "
    "and whether coherent permutation equivalence applies. Every threshold/cap "
    "must bind an exact params identity whose paper truth remains in its blessed "
    "R2C-083 carrier. When the paper has no justified graph null, emit explicit "
    "null ablation and contribution-ref fields. Otherwise select only a "
    "paper-justified empty/identity/permuted graph, removed message passing, or "
    "non-graph decoder null that the paper's method and comparison story "
    "support; its element must be an `evaluation_control`, and it must name the "
    "exact topology- or neighbor-sensitivity discriminator shared by both arms. "
    "A callable null also declares its exact graph keyword, neighbor-signal "
    "keyword, and the same coindexed output root as the real arm; do not assume "
    "the real helper's parameter names. "
    "Do NOT hand-copy the block's wiring onto elements: the pipeline derives "
    "`relational_structure.kind=homogeneous_graph` markers for the block's "
    "alignment, construction, and message-passing elements, and derives every "
    "per-element `graph_mechanism.*` verification ref from the block's own "
    "element ids (`graph_mechanism.alignment_prerequisite` on the alignment "
    "element; parameter-agreement and construction refs on the construction "
    "element; topology, neighbor-signal, and any permutation ref on the "
    "message-passing element; the contribution ref on the ablation control). "
    "Leave `graph_mechanism.*` "
    "values OUT of `verification_probe_refs` (still emit the field, with its "
    "non-graph refs or `[]`), and do not add the homogeneous marker to any "
    "element the block does not bind — a stray marker is rejected. "
    "`kind=unsupported` markers remain yours to author from paper evidence. "
    "Do not invent a control, callable, parameter, graph form, or probe ref."
)


GRAPH_MECHANISM_IMPLEMENTATION_GUIDANCE = (
    "When `homogeneous_graph_mechanism` is present, implement every exact "
    "callable that belongs to your owned module as a pure public top-level "
    "helper used by the live path, and keep it importable at the declared "
    "module/qualname. Never substitute a dotted class/instance method: the v1 "
    "probe contract has no authority to guess model construction. The graph "
    "constructor is a pure deterministic "
    "surface: consume the declared feature input and threshold/cap through the "
    "declared keyword parameters, return the declared sparse edge-index or "
    "dense adjacency selection, and obey direction and self-loop policy. A "
    "per-source cap ranks admitted non-self candidates by descending cosine "
    "similarity with canonical target-index ties; undirected construction "
    "keeps only mutual selections before adding required self loops. Never "
    "replace a params-bound value with an equal internal literal. The declared "
    "message-passing callable must consume the prepared graph and declared "
    "neighbor-signal parameter on the live model path. If the paper-grounded "
    "training function receives the declared architecture model, preserve and "
    "return that exact model object: weight/optimizer operations may use its "
    "identity-preserving public methods, but never replace its class, forward "
    "dispatch, or model binding, and never pass the model object into an opaque "
    "helper. If the paper-grounded "
    "non-null ablation declares a callable, implement that exact callable too. "
    "Never invent a control when the contract records null. Reuse the "
    "relational preparation/identity mapping; do not independently reorder or "
    "reconstruct entity identity. Unsupported graph grammars remain pipeline "
    "coverage gaps, not producer simplifications."
)


GRAPH_MECHANISM_NOTEBOOK_GUIDANCE = (
    "When `homogeneous_graph_mechanism` is present, make its declared callables "
    "provably live with one direct top-level notebook path. After the rendered "
    "params placeholders define `cfg = unpack(params)`, import and call the exact "
    "public graph constructor once with explicit keywords: pass the exact "
    "R2C-084 feature binding and read every threshold/cap only as "
    "`cfg[<declared params_name>]`. Select its declared graph output into one "
    "local name without transforming or overwriting it. Make exactly one typed "
    "fitting snapshot: a NumPy array uses no-argument `copy()`, while a torch "
    "Tensor uses no-argument `clone()`. Pass only that snapshot and the exact "
    "feature binding through the exact "
    "training keywords. Directly import and instantiate the exact declared "
    "architecture class once at top level after graph construction, and pass "
    "that exact model binding through the fitting model keyword. Keep the "
    "original graph for the exact pluggable inference keyword. Bind the "
    "fitting call's returned model and pass that "
    "exact binding through the pluggable model keyword. Do not use dynamic "
    "namespace or attribute mutation (`globals`, `locals`, `vars`, `exec`, "
    "`eval`, `setattr`, or `delattr`) on this notebook path. Do not hide these "
    "bindings behind wrappers, positional arguments, `**kwargs`, mutations, or "
    "reconstructed equal values. The architecture's declared message helper "
    "must return its exact coindexed `output_root` through a direct local binding "
    "or direct final return on both fitting and inference paths."
)


REVIEWER_GRAPH_MECHANISM_GUIDANCE = (
    "When the spec declares `homogeneous_graph_mechanism`, audit each exact "
    "construction, message-passing, and ablation callable against that block. "
    "Each must be the declared public top-level helper used by the live path, "
    "not a dotted class/instance method that would require guessed model "
    "construction. "
    "Confirm threshold/cap values come from the named params entries (not equal "
    "literals), the fixed mutual-top-k/tie-break/self-loop semantics match, "
    "prepared "
    "identity reaches the live path, neighbor signal is consumed, and a "
    "non-null control is paper-justified and uses its exact declared "
    "topology/neighbor discriminator. A null control must remain unavailable, "
    "not be invented. Keep graph construction, entity "
    "alignment, mechanism liveness, contribution ablation, held-out skill, and "
    "paper-scale uplift as separate conclusions. A clean run, aligned graph, "
    "sensitivity pass, falling loss, or skill result cannot substitute for any "
    "other rung."
)


# Per-stage 1-paragraph task summaries. Producer agents already have their
# full instructions in `.opencode/agents/<agent>.md` (loaded as system prompt
# by the Task tool / message dispatch); these summaries name what to do for
# this run only.
STAGE_TASK_SUMMARIES: dict[str, str] = {
    "stage_1_decomposer": (
        "Stage 1.a — read paper.md (path below), then USE YOUR WRITE TOOL to "
        "produce paper_map.json at the path below. OUTPUT MODE: CANONICAL "
        "ONLY. Do not write `paper_map_parts/` in this dispatch; if the single "
        "artifact cannot be produced, return without writing chunks and the "
        "driver will retry in chunk mode. Do not return until paper_map.json "
        "exists on disk or you have written the halt sidecar allowed by your "
        "agent definition. The file must validate against schemas/paper_map.py. "
        "The analyzer in Stage 1.b consumes paper_map.json; element IDs you "
        "assign here will be referenced from method_spec.json's "
        "core_method.key_elements, so use stable IDs (e.g., alg-*, eq-*, "
        "concept-*) and be thorough."
    ),
    "stage_1_analyzer": (
        "Stage 1.b — read paper.md and paper_map.json (already written at "
        "the paths below), then USE YOUR WRITE TOOL to produce "
        "method_spec.json at the path below. OUTPUT MODE: CANONICAL ONLY. Do "
        "not write `method_spec_parts/` in this dispatch; if the single "
        "artifact cannot be produced, return without writing chunks and the "
        "driver will retry in chunk mode. Do not return until method_spec.json "
        "exists on disk, or until you have written the paradigm-mismatch halt "
        "artifacts allowed by your agent definition. The spec must validate "
        "against schemas/method_spec.py; its "
        "core_method.key_elements MUST be a list of paper_map element IDs "
        "(e.g., \"eq-5\", \"alg-1\") — not prose descriptions — and every "
        "ID must exist in paper_map.json. Also populate the methodology "
        "fidelity fields (`methodology_replication_contract`, "
        "`methodology_contract_pack`, and `replication_feasibility`) so "
        "downstream stages know which paper mechanisms must be replicated, "
        "which approximations are approved, and whether any core methodology "
        "element is `not_replicable`. For those fields, every element with "
        "non-empty `acceptable_approximations` must have "
        "`replication_status: faithful_approximation_allowed`, and "
        "`replication_feasibility.approved_approximations` must list exactly "
        "those element IDs. If the Run paths block lists a taxonomy node or "
        "run-local provisional pack, use that source as classification/build "
        "context for this run. "
        + ANALYZER_VERIFICATION_PROBE_GUIDANCE
        + " "
        + ANALYZER_CALIBRATION_CONTEXT_GUIDANCE
        + " "
        + ANALYZER_GRAPH_MECHANISM_GUIDANCE
        + " "
        "For a run-local provisional pack (a gap-family paper), the structured parameter fields are the ONLY parameter carriers downstream — no derivation conventions exist for a brand-new family — so every paper-stated numeric value MUST land in its role-compatible structured field: physical values calibrated to the numerical scale of input values go in critical_requirements.scale_dependent_hyperparameters, training values go in critical_requirements.training as NUMBERS (a numeric field must hold a number or null, never prose), method knobs go into the pluggable component signature as keyword defaults, and time-series evaluation quantities follow comparison.evaluation_protocol below. "
        "The calibration lane is only for values tied to one of the typed contexts above. A SCALE-FREE stated constant (a normalized threshold, a cutoff in (0,1), a probability, a ratio) goes in critical_requirements.param_glossary with paper_value set. Temporal counts and windows — context length, number of lags, lookback, forecast-call horizon, validation span, and test span — MUST NOT appear in critical_requirements.scale_dependent_hyperparameters. "
        "For every canonical or alias time_series_forecasting spec, comparison.evaluation_protocol is REQUIRED. It has a scheme and exactly one quantity for each role: context_length, forecast_call_horizon, validation_span, and test_span. Each quantity records role, optional parameter_name, exact prose paper_names, an explicitly present exact case-sensitive paper_symbols list (empty only when the paper uses no notation), value, canonical unit, positive numeric granularity, and paper_value_status. Role/value provenance uses a VERBATIM evidence_quote, paper_section, and paper_element_ids naming the map elements that cover the quoted passage; unit/granularity provenance separately uses VERBATIM axis_evidence_quote, axis_paper_section, and axis_paper_element_ids. When no paper-map element covers a quoted passage, an EMPTY id list is the honest state (it costs downstream adjudication); never cite a valid-but-unrelated element to fill the list. Reuse one passage only when it grounds both. When one contiguous passage cannot carry the role wording, the notation, and the stated value together (e.g. prose defines the symbol while an appendix table states the value), join the contiguous passages with the elision marker ' [...] '; each fragment must be verbatim on its own, a paraphrase is never allowed, and value/name bindings never cross the marker, so the fragment that states the value must bind it to a declared name or symbol itself. Symbols another role owns may appear inside your quote's surrounding text; declare each symbol only on the quantity that owns it. Names/symbols are identity bridges, never numeric authority. A stated value requires an explicit name/symbol assignment, a direct role-subject binding, a table row, or paired validation/test ordering such as 13 and 26 respectively; a lone or nearby number is insufficient. Axis evidence must bind cadence to the target/observed series, not merely mention a unit. Set parameter_name only for a taxonomy-declared carrier with the same protocol_role; do not invent validation/test params. A paper-unspecified quantity uses paper_value_status='paper_unspecified' and value=null. Scheme kind is null exactly when paper_value_status='paper_unspecified'. Never infer K=1 from T+1, copy test 26 into K, merge roles, or use unrelated unit prose as axis truth. "
        "Use exactly ONE paper-value carrier for a parameter. A SCALE-FREE stated constant belongs in critical_requirements.param_glossary only, with paper_value set and no scale-dependent lane entry. A DATA-SCALE-DEPENDENT value belongs in critical_requirements.scale_dependent_hyperparameters only; if its definition also earns a glossary entry, keep that glossary entry meaning-only with paper_value=null. Never repeat one parameter's paper value in both carriers. "
        "Declare a paper-grounded relational mechanism through the typed "
        "homogeneous_graph_mechanism block (one homogeneous graph) or a "
        "relational_structure kind=unsupported marker with the concrete "
        "unsupported_kind (any other relational form). The homogeneous "
        "markers on the block's own elements are derived by the pipeline; "
        "never infer graph activation from a graph-like name or prose, and "
        "never simplify an unsupported relational form to make generation "
        "proceed. "
        "Do not write the retired guide-path carrier; "
        "it was removed from the served MethodSpec schema. If no registered "
        "taxonomy node or supplied pack matches, "
        "do not write method_spec.json; instead write method_spec.json.halt "
        "plus `.pipeline/paradigm_gap_report.json` and "
        "`.pipeline/paradigm_gap_report.md` with structured evidence for "
        "whether a new sub-paradigm, new top-level paradigm, or unsupported "
        "status is appropriate. A `route_elsewhere` signal that matches the "
        "paper and points at a node whose status is `reserved` counts as "
        "no-match: that IS a taxonomy gap — never fall back to the closest "
        "populated node the routing signal told you to leave."
    ),
    "stage_2b_architecture": (
        "Stage 2.b — produce three artifacts for this paper: "
        "(1) `method/model.py` (architecture class(es)), "
        "(2) `method/training.py` (build helpers + training-loop entry point), "
        "(3) `.pipeline/arch_contract.json` (structured shape contract method-"
        "coder and notebook-generator will consume). The architecture must "
        "satisfy the spec's `critical_requirements.model.specific_features` "
        "and the spec-derived taxonomy build plan; the shape contract must "
        "match what your code actually "
        "accepts at runtime (Stage 2.d will dry-run your classes against the "
        "contract's shapes). If `methodology_contract_pack` or "
        "`methodology_replication_contract` names model-level obligations or "
        "verification expectations, preserve them in model.py/training.py. "
        "When a methodology element has relational_structure.kind="
        "homogeneous_graph, arch_contract.json must include relational_indexing "
        "and training.py must expose its declared pure preparation callable on "
        "the fitting path. Jointly map every co-indexed node root, relabel "
        "source-position endpoints to integer local positions, preserve "
        "source-to-destination orientation and declared degree semantics, and "
        "return explicit identity/output mappings; do not construct a model in "
        "that helper. For schema 2, relational_indexing.execution must bind "
        "the exact constructed architecture block and full typed fitting and "
        "inference input roots; fitting receives source-domain values, then "
        "its model call consumes the helper's prepared graph, degree, and "
        "shared forward-input objects. Keep fitting-only prepared roots in the "
        "loss path and output ids in final joins; Stage 2.d observes their "
        "preparation boundary while integrated acceptance observes those "
        "downstream consumers. Empty ordinary co-index maps are valid for a "
        "graph-only callable; graph, degree, id, mapped-root, and output-axis "
        "domains must still share exact semantic dimension identities rather "
        "than merely equal counts. Do not "
        "guess these bindings from parameter names. Bind an exact typed epoch "
        "input to one, or declare an intrinsically one-epoch fitting entry. "
        "Unsupported relational "
        "markers are pipeline coverage "
        "gaps, not architecture-coder repair instructions. "
        "New arch_contract.json output MUST use schema_version 2.0.0 with "
        "stable semantic dimensions, structured checked arithmetic, typed "
        "input/output descriptors, and exact callable parameter names. Do "
        "not emit v1 shape fields, infer unmeasured feature widths, or invent "
        "evaluation protocol/partition truth. Structured KD batches remain "
        "one exact opaque parameter because v2 has no dict grammar; opaque "
        "certifies no interior members. "
        + GRAPH_MECHANISM_IMPLEMENTATION_GUIDANCE
        + " "
        "Naming rule (item 31b): components the spec declares as user-facing "
        "(anything `try_it_out.system_provides` describes, and every class the "
        "contract names) are public by construction — never leading-"
        "underscore. When a system_provides entry declares `symbol_kind`, "
        "deliver that surface as the declared kind (a promised class must be "
        "a class, a promised function a function) — the 2.b gate checks the "
        "definition's AST kind against the declaration. Internal helpers may "
        "stay private, but every name your "
        "own files reference must resolve exactly (a class defined `__X` is "
        "not referenceable as `_X`)."
    ),
    "stage_2c_method": (
        "Stage 2.c — produce method.py for this paper. Output "
        "`method/method.py` ONLY (no baseline files). The pluggable "
        "function's name comes from `spec.comparison.pluggable_component.name`; "
        "its signature must match `.signature` byte-for-byte. The body must "
        "NOT enumerate baseline functions. Read "
        "`methodology_replication_contract` and `methodology_contract_pack`; "
        "implement every core `must_replicate` element, document only the "
        "approved approximations, and avoid every forbidden substitution. "
        "If arch_contract.relational_indexing is present, consume its prepared "
        "local graph and explicit identity/output mappings; do not independently "
        "subset, permute, cast, or reconstruct node identity. "
        "Read arch_contract.schema_version first. Current v2 contracts use "
        "the dimensions registry and typed pluggable_component.input/output; "
        "v1 shape fields are only for archived/resumed contracts, and opaque "
        "values (including structured KD batches) carry no interior "
        "certification. "
        "Do not satisfy the contract with annotations on dead code: each "
        "core implementation must be reachable from the pluggable function or "
        "from a helper the notebook is expected to call."
        " "
        + GRAPH_MECHANISM_IMPLEMENTATION_GUIDANCE
    ),
    "stage_2d_tests": (
        "Stage 2.d — generate the per-element test surface. Write ONE "
        "module per element in the Eligible elements section below, at "
        "`method/tests/test_<element id with hyphens as underscores>.py`, "
        "and NOTHING else (the driver renders method/tests/README.md "
        "itself). Each module carries exactly one `# paper-element: <id>` "
        "comment, where `<id>` is the eligible element's id copied "
        "CHARACTER-FOR-CHARACTER from the heading in that section — the "
        "same string the filename is built from. Never abbreviate or "
        "shorten it: the driver joins on this id, and an anchor that is "
        "not an exact eligible id drops the whole module from the "
        "delivered package (pdwa 2026-07-27 lost 19 of 21 modules to "
        "anchors like `eq-j-c` for `eq-j-clearance` and `eq-k` for "
        "`eq-kinematic-model`). Each module also "
        "imports and CALLS the implementing function named for "
        "its element, and asserts at least one VALUE-level property the "
        "element itself states — shape/len/dtype checks alone fail the "
        "deterministic floor. When the element's pseudocode states "
        "concrete numbers or a worked example, pin them (stated input in, "
        "stated output asserted; small tolerance for floats). Fixture "
        "scale, deterministic, no network. Write assertions from the "
        "element's statement, never from what the code computes — each "
        "test is checked against a mechanically broken copy of its "
        "function and labeled weak when it cannot tell the difference."
    ),
    "stage_3a_notebook": (
        "Stage 3.a — write `<PIPELINE_DIR>/notebook_draft.py` (jupytext-percent "
        "format) for this paper. §4 and §5 must reflect the methodology "
        "contract: show the core elements the code implements, disclose "
        "approved demo-scale approximations, and avoid implying forbidden "
        "substitutions are faithful. §5 runs the single method end-to-end on "
        "a single seed; do NOT add comparison cells or multi-method dispatch loops. "
        + GRAPH_MECHANISM_NOTEBOOK_GUIDANCE
        + " "
        "When the notebook discusses approximations or departures from the "
        "paper, enumerate all params whose source is `system_default`, "
        "`system_inferred`, or `spec_default`; never say only one thing differs "
        "unless the params and assumptions artifacts prove that is true. For "
        "active-learning notebooks, keep displayed rounds, acquisition calls, "
        "batch size, and labeled-budget math consistent. "
        "Every metric the notebook names must compute that name's standard "
        "formula (metric_name_integrity — the pdfgnn 2026-08-04 delivery "
        "reported WMAPE while computing plain MAPE, the weighting dropped): "
        "if the implementation differs from the named metric's definition, "
        "rename the metric to what the code actually computes and state the "
        "difference in prose — never report a value under a name whose "
        "formula the code does not satisfy. "
        "Read arch_contract.schema_version first: current v2 handoffs use "
        "dimensions and typed input/output descriptors; v1 shape fields are "
        "archived/resume compatibility only. Never synthesize or claim the "
        "interior of an opaque value, including a structured KD batch."
    ),
    "stage_4_review": (
        "Stage 4 — review the generated package + notebook for paper fidelity. "
        "Begin with Pass 0 (spec-asserted structured invariants): iterate every "
        "entry in `spec.methodology_replication_contract.elements` when that "
        "field is present, confirming core `must_replicate` behavior is "
        "present, approved approximations are documented, and forbidden "
        "substitutions are absent. Then iterate every "
        "entry in `spec.critical_requirements.scale_dependent_hyperparameters` "
        "and every entry in `spec.critical_requirements.required_model_methods`. "
        "These are mechanical, per-entry checks that must be performed "
        "exhaustively before any judgment-based pass; they exist precisely "
        "because LLM judgment misses them when the artifact looks superficially "
        "right. "
        + REVIEWER_CALIBRATION_CONTEXT_GUIDANCE
        + " "
        + REVIEWER_GRAPH_MECHANISM_GUIDANCE
        + " Then run Passes 1-5 as described in the agent contract. "
        "Include a researcher sniff-test pass: verify every core mechanism is "
        "on a live runtime path, objective/ranking direction words in prose "
        "match the code (`highest`, `lowest`, `nearest`, `farthest`, `diverse`, "
        "`representative`), active-learning budgets match actual acquisition "
        "calls, and notebook prose lists all paper departures recorded in "
        "params.json or assumptions.md. "
        "Baselines are out of scope; their presence in the package is itself a "
        "finding the reviewer should raise as `issue_type: scope_creep, "
        "severity: critical`."
    ),
    "stage_1_decomposer_retry": (
        "Stage 1.a RETRY — your previous dispatch did NOT produce paper_map.json "
        "or assemblable paper_map_parts. You read the paper but did not write a "
        "usable artifact. Do NOT read any additional files. Do NOT load any "
        "additional skills. Do NOT analyze the paper further — you have "
        "everything you need.\n\n"
        "**Previous output error:** {previous_output_error}\n\n"
        "If this names a JSON parse error, rewrite the named part file as "
        "standalone valid JSON. Every file ending in `.json` must parse with "
        "`json.loads`; raw unquoted text is not valid JSON. "
        "OUTPUT MODE: CHUNK ONLY. Do not write paper_map.json in this retry. "
        "Your IMMEDIATE next action must be to write chunked parts under "
        "`<run_dir>/.pipeline/paper_map_parts/`: `manifest.json` plus one small "
        "JSON object per element under `paper_map_parts/elements/`. Each "
        "element file must contain exactly one paper-map element object, not a "
        "full paper_map envelope and not `{{\"element\": ...}}`. The schema is "
        "schemas/paper_map.py; element IDs must be stable (alg-*, eq-*, "
        "concept-*); related_equations must reference type=equation elements "
        "only. Write the chunk files NOW."
    ),
    "stage_1_analyzer_retry": (
        "Stage 1.b RETRY — your previous dispatch did NOT produce method_spec.json "
        "or assemblable method_spec_parts. You read inputs but did not write a "
        "usable artifact. Do NOT read any additional files. Do NOT load any "
        "additional skills. Do NOT continue analyzing — you have everything you "
        "need.\n\n"
        "**Previous output error:** {previous_output_error}\n\n"
        "If this names a JSON parse error, rewrite the named part file as "
        "standalone valid JSON. Every file ending in `.json` must parse with "
        "`json.loads`; raw unquoted text such as `reproducible` is not valid "
        "JSON — write `\"reproducible\"` or "
        "`{{\"feasibility\": \"reproducible\"}}`. Do not place sibling "
        "top-level keys after a closed JSON object; one part file must be "
        "exactly one JSON value.\n\n"
        "OUTPUT MODE: CHUNK ONLY. Do not write method_spec.json in this retry. "
        "If the Run paths block lists a taxonomy node or run-local provisional "
        "pack, use that source as classification/build context for this run. "
        + ANALYZER_CALIBRATION_CONTEXT_GUIDANCE
        + " "
        "For a run-local provisional pack (a gap-family paper), the structured parameter fields are the ONLY parameter carriers downstream — no derivation conventions exist for a brand-new family — so every paper-stated numeric value MUST land in its role-compatible structured field: physical values calibrated to the numerical scale of input values go in critical_requirements.scale_dependent_hyperparameters, training values go in critical_requirements.training as NUMBERS (a numeric field must hold a number or null, never prose), method knobs go into the pluggable component signature as keyword defaults, and time-series evaluation quantities follow comparison.evaluation_protocol below. "
        "The calibration lane is only for values tied to one of the typed contexts above. A SCALE-FREE stated constant (a normalized threshold, a cutoff in (0,1), a probability, a ratio) goes in critical_requirements.param_glossary with paper_value set. Temporal counts and windows — context length, number of lags, lookback, forecast-call horizon, validation span, and test span — MUST NOT appear in critical_requirements.scale_dependent_hyperparameters. "
        "For every canonical or alias time_series_forecasting spec, comparison.evaluation_protocol is REQUIRED. It has a scheme and exactly one quantity for each role: context_length, forecast_call_horizon, validation_span, and test_span. Each quantity records role, optional parameter_name, exact prose paper_names, an explicitly present exact case-sensitive paper_symbols list (empty only when the paper uses no notation), value, canonical unit, positive numeric granularity, and paper_value_status. Role/value provenance uses a VERBATIM evidence_quote, paper_section, and paper_element_ids naming the map elements that cover the quoted passage; unit/granularity provenance separately uses VERBATIM axis_evidence_quote, axis_paper_section, and axis_paper_element_ids. When no paper-map element covers a quoted passage, an EMPTY id list is the honest state (it costs downstream adjudication); never cite a valid-but-unrelated element to fill the list. Reuse one passage only when it grounds both. When one contiguous passage cannot carry the role wording, the notation, and the stated value together (e.g. prose defines the symbol while an appendix table states the value), join the contiguous passages with the elision marker ' [...] '; each fragment must be verbatim on its own, a paraphrase is never allowed, and value/name bindings never cross the marker, so the fragment that states the value must bind it to a declared name or symbol itself. Symbols another role owns may appear inside your quote's surrounding text; declare each symbol only on the quantity that owns it. Names/symbols are identity bridges, never numeric authority. A stated value requires an explicit name/symbol assignment, a direct role-subject binding, a table row, or paired validation/test ordering such as 13 and 26 respectively; a lone or nearby number is insufficient. Axis evidence must bind cadence to the target/observed series, not merely mention a unit. Set parameter_name only for a taxonomy-declared carrier with the same protocol_role; do not invent validation/test params. A paper-unspecified quantity uses paper_value_status='paper_unspecified' and value=null. Scheme kind is null exactly when paper_value_status='paper_unspecified'. Never infer K=1 from T+1, copy test 26 into K, merge roles, or use unrelated unit prose as axis truth. "
        "Use exactly ONE paper-value carrier for a parameter. A SCALE-FREE stated constant belongs in critical_requirements.param_glossary only, with paper_value set and no scale-dependent lane entry. A DATA-SCALE-DEPENDENT value belongs in critical_requirements.scale_dependent_hyperparameters only; if its definition also earns a glossary entry, keep that glossary entry meaning-only with paper_value=null. Never repeat one parameter's paper value in both carriers. "
        "Do not write the retired guide-path carrier; it was removed from the "
        "served MethodSpec schema. "
        "Your IMMEDIATE next action must be to write chunked parts under "
        "`<run_dir>/.pipeline/method_spec_parts/`: `manifest.json` plus small "
        "top-level section files such as `paper.json`, `core_method.json`, "
        "`paper_claims.json`, `try_it_out.json`, `data_requirements.json`, "
        "`dependencies.json`, `repo.json`, `comparison.json`, "
        "`critical_requirements.json`, `methodology_replication_contract.json`, "
        "`methodology_contract_pack.json`, `replication_feasibility.json`, and "
        "`feasibility.json`. Every section file must be a wrapper object with "
        "exactly one MethodSpec top-level key, for example "
        "`{{\"comparison\": {{...}}}}` or "
        "`{{\"feasibility\": \"reproducible\"}}`; do not write direct section "
        "values. The assembled spec must validate against schemas/method_spec.py; "
        "core_method.key_elements MUST be a list of paper_map element IDs "
        "(e.g., \"eq-5\", \"alg-1\") — every ID must already exist in "
        "paper_map.json. The methodology fidelity fields "
        "(`methodology_replication_contract`, `methodology_contract_pack`, "
        "`replication_feasibility`) must also be present and internally "
        "consistent: every element with non-empty `acceptable_approximations` "
        "must have `replication_status: faithful_approximation_allowed`, and "
        "`replication_feasibility.approved_approximations` must list exactly "
        "those element IDs. "
        + ANALYZER_VERIFICATION_PROBE_GUIDANCE
        + " "
        + ANALYZER_GRAPH_MECHANISM_GUIDANCE
        + " Write the artifact content NOW."
    ),
}


STAGE_REVIEW_TASK_TEMPLATE = (
    "Per-stage semantic review for {stage_id}. The deterministic validator "
    "just passed; your job is the semantic + integration check. Read the "
    "matched taxonomy node for the stage_review_focus.{stage_id} "
    "block, apply each semantic_check, then USE YOUR "
    "WRITE TOOL EXACTLY ONCE to save findings to "
    "<run_dir>/.pipeline/stage_review_{stage_id}.json. After that file is "
    "written, return. Do NOT write any other file. Do NOT implement, edit, "
    "or create source code — you are the reviewer for {stage_id}, not the "
    "producer for the next stage. Do NOT write review JSONs for stages "
    "other than {stage_id}. The driver halts on out-of-scope writes. The "
    "review file must validate against schemas/stage_review_report.py. "
    "Do NOT invent additional checks beyond what the taxonomy node specifies "
    "OR what your agent procedure mandates as a universal baseline (see "
    "the 'Universal baseline check: equation verification' section of "
    "your agent procedure — that baseline IS required for stage_2c_method "
    "regardless of taxonomy semantic-check enumeration).\n\n"
    "**Equation-verification reminder (stage_2c_method ONLY).** If "
    "{stage_id} == `stage_2c_method`, you MUST run the universal "
    "equation-verification baseline described in your agent procedure "
    "AFTER applying the taxonomy semantic_checks. Scan `method/method.py` "
    "for every `# paper-element: eq-XXX` comment, look up each `eq-XXX` in "
    "`paper_map.json`, and verify symbol-by-symbol that the implementation "
    "matches the paper formula. The bev-distill 2026-05-22 case (quality-"
    "score weighting reduced to a no-op via scalar-vs-vector aggregation "
    "order) is the canonical failure mode this baseline exists to catch. "
    "For any other stage_id, this baseline does NOT apply.\n\n"
    "**Methodology-contract reminder.** If `method_spec.json` contains "
    "`methodology_replication_contract`, treat it as an explicit review "
    "checklist. For `stage_1_analyzer`, verify the contract's core elements, "
    "statuses, blockers, approved approximations, run-level "
    "`replication_feasibility`, and paper-truth structured fields are "
    "paper-grounded and internally consistent. If the contract says a paper "
    "uses one value but approves a demo reduction to another, the structured "
    "paper-truth field must contain the paper value, not the demo/default "
    "value. "
    "For producer-output stages, verify the artifact under review preserves "
    "the relevant core methodology elements, documents only approved "
    "approximations, and does not use a forbidden substitution.\n\n"
    "**Per-dataset binding reminder (stage_2x_params ONLY).** When the "
    "spec's `data_setup.per_dataset_values` names dataset-specific paper "
    "values for a parameter, a derived value that differs from the spec's "
    "scalar is CORRECT — not a finding — when params.json records "
    "`bound_dataset` and the value equals that dataset's map entry (the "
    "deriver bound the paper's value for the dataset the demo actually "
    "uses). Findings on this claim class are limited to: a bound value "
    "that mismatches its named dataset's map entry, or a scalar that "
    "contradicts every map entry (an analyzer error).\n\n"
    "**Runtime-path and semantics reminder.** For producer-output stages, do "
    "not stop at comments, annotations, or imported helpers. Verify the core "
    "paper behavior is on a live execution path from the notebook or the "
    "pluggable component. When prose describes ranking/objective direction "
    "using words like `highest`, `lowest`, `nearest`, `farthest`, `diverse`, "
    "or `representative`, compare that claim against the actual sort order, "
    "distance/probability formula, and selected indices. If a notebook states "
    "an acquisition/training budget, confirm the loop performs that many "
    "method calls rather than only that many plot/evaluation points.\n\n"
    "**CONTRACT FIELD — COPY VERBATIM.** The `stage_id` field in your "
    "output JSON MUST be exactly `{stage_id}` — character-for-character. "
    "Do NOT paraphrase or invent variants (e.g., `{stage_id}` → "
    "`{stage_id}_reviewed`, `stage_2b` → `stage_2b_architecture`). The "
    "driver halts on contract mismatch."
)


STAGE_REVIEW_RESOLUTION_REASK_TEMPLATE = (
    "Structured-resolution re-ask for {stage_id}. Your review at "
    "<run_dir>/.pipeline/stage_review_{stage_id}.json contains blocking "
    "finding(s) WITHOUT a structured `proposed_resolution`. This stage's "
    "producer is a deterministic script — there is no LLM fix dispatch — "
    "so an unresolved blocking finding halts the entire run. For EACH "
    "finding listed below, decide one of:\n"
    "  (a) a safe expert-default resolution exists — attach a structured "
    "`proposed_resolution` per the 'Auto-resolvable check kinds' section "
    "of your agent procedure (today: kind=relabel_param_source for "
    "paper_source_values_match_paper findings; `add_fields` keys must be "
    "ParamEntry schema fields only — put supporting detail inside the "
    "`reasoning` text, never as new keys). When you relabel a param AWAY "
    "from source=paper, the `reasoning` you supply must NOT contain an "
    "inline claim of the form \"the paper states/uses/sets <name>=<value>\" "
    "for a value the paper does not literally state next to that name: the "
    "provenance validator rejects such a claim on EVERY source, so the "
    "relabel alone cannot clear the finding and the driver reverts it. "
    "Describe the paper's context without the name=value form (\"the paper "
    "demonstrates this setting in Section 7.4 without prescribing it\"), or\n"
    "  (b) no safe expert default exists — set `resolution_status` to "
    "\"needs_user\" so the halt is a deliberate decision, not an "
    "omission.\n"
    "Do NOT re-review the stage. Do NOT add, drop, or reword findings; "
    "keep each finding's `id` and content identical, changing ONLY "
    "`proposed_resolution` and `resolution_status`. Some finding ids "
    "below may look synthesized rather than reviewer-style (e.g. "
    "`DEF-c_safe` instead of `F001`) because the driver built them from "
    "validator errors — they are still contract fields. Key each finding "
    "by the EXACT id shown below and never renumber to your own F-style "
    "scheme: a response filed under a different id cannot be merged back "
    "and halts the run. USE YOUR WRITE TOOL "
    "EXACTLY ONCE to save the full review JSON (same top-level shape as "
    "your original review: schema_version, stage_id, review_status, "
    "summary, findings, checks_summary, caveats) containing ONLY the "
    "findings below to "
    "<run_dir>/.pipeline/stage_review_{stage_id}_resolution.json — do NOT "
    "overwrite your original review file.\n\n"
    "**CONTRACT FIELD — COPY VERBATIM.** The `stage_id` field in your "
    "output JSON MUST be exactly `{stage_id}` — character-for-character.\n\n"
    "Findings needing a resolution decision:\n{findings_json}"
)


FEASIBILITY_SURROGATE_REASK_TEMPLATE = (
    "Feasibility re-ask for stage_1. The deterministic feasibility gate "
    "halted this run: {n} blocker(s) in your method_spec.json are marked "
    "`cannot_implement`, yet each carries YOUR OWN `resolution` text "
    "proposing a workable surrogate (a demo-scale substitute). That "
    "combination is internally inconsistent: `cannot_implement` means "
    "'the run must stop', while a faithful surrogate belongs under "
    "`can_approximate` with the approximation contract spelled out in "
    "`resolution`. For EACH blocker listed below, decide one of:\n"
    "  (a) the proposed surrogate IS faithful for a demo-scale "
    "replication — change that blocker's `status` to `can_approximate` "
    "and rewrite its `resolution` as an explicit approximation contract: "
    "what is substituted, what is preserved exactly (the computation the "
    "paper contributes), and what conclusions remain valid at demo "
    "scale, or\n"
    "  (b) the requirement is genuinely unimplementable even as a "
    "surrogate — KEEP `status` as `cannot_implement` and strengthen "
    "`reason` with one sentence on why the surrogate is not faithful.\n"
    "Judge each blocker on faithfulness, not convenience — (b) is a "
    "legitimate answer and halts the run deliberately.\n"
    "Edit `<run_dir>/.pipeline/method_spec.json` IN PLACE, changing ONLY "
    "the `status`, `resolution`, and `reason` fields of the blockers "
    "listed below inside `critical_requirements.blockers`. Do NOT add or "
    "remove blockers, do NOT touch any other part of the spec, and do "
    "NOT relax the methodology replication contract. The edited file "
    "must still validate against schemas/method_spec.py.\n\n"
    "Blockers to decide (requirement text identifies each entry):\n"
    "{blockers_json}"
)


STAGE_REVIEWER_RETRY_TASK_TEMPLATE = (
    "Per-stage semantic review for {stage_id} — RETRY. Your previous "
    "dispatch did NOT produce `<run_dir>/.pipeline/stage_review_{stage_id}.json`. "
    "You read inputs and (likely) applied the taxonomy semantic checks, "
    "but never called Write. Do NOT read any additional files. Do NOT load "
    "any additional skills. Do NOT continue analyzing — you have everything "
    "you need from the prior dispatch's reading. Your IMMEDIATE next action "
    "must be a single Write tool call producing "
    "stage_review_{stage_id}.json at the path below. Top-level keys must "
    "be EXACTLY: schema_version, stage_id, review_status, summary, "
    "findings, checks_summary, caveats — no wrapper key like "
    "'review_report'. Write the full JSON content NOW. The review file must "
    "validate against schemas/stage_review_report.py.\n\n"
    "**CONTRACT FIELD — COPY VERBATIM.** The `stage_id` field in your "
    "output JSON MUST be exactly `{stage_id}` — character-for-character."
)


JUDGE_ELEMENT_TEST_TEMPLATE = (
    "Halt-recovery judgment at {stage_id}, iteration {iteration}. A "
    "generated per-element test just FAILED against the assembled "
    "package. A failing generated test has exactly two candidate "
    "culprits, and your job is to assign ownership: the delivered CODE "
    "(the test correctly caught a defect — action=dispatch_fix with "
    "target_agent `r2c-method-coder` and a normal code finding) or the "
    "TEST itself (action=dispatch_fix with target_agent "
    "`r2c-test-generator` and issue class test_defect in the finding "
    "description). The paper element's own verbatim text below is the "
    "ARBITER: if the failing assertion traces to a property or worked "
    "example the element itself states, the code owns the failure; if "
    "the assertion does not trace to the element's statement, the test "
    "owns it. Read the element text, the test module, and the "
    "implementing function before deciding. The test side gets ONE "
    "regeneration in this loop — if a prior decision below already "
    "routed this module to the test-generator, do not route it there "
    "again; either the code owns it or halt. Write EXACTLY ONE decision "
    "object to `{decision_output_path}`. Do NOT write "
    "`<run_dir>/.pipeline/judge_decisions.json` (driver-owned). Do NOT "
    "edit any producer artifact. Do NOT load skills. The decision must "
    "validate against `schemas/judge_decision.py`.\n\n"
    "**CONTRACT FIELDS — COPY VERBATIM.** Your decision MUST contain:\n"
    "  - `stage_id`: `{stage_id}` (exactly)\n"
    "  - `iteration`: {iteration} (the integer, not a string)\n"
    "  - `validator_label`: `{validator_label}` (exact string)\n"
    "If any of these don't match this prompt, the driver rejects your "
    "scratch decision with a 'contract violation' error.\n\n"
    "**Paper element (id `{element_id}`, the arbiter):**\n"
    "```\n{element_block}\n```\n\n"
    "**Failing test module:** `{test_module_path}` (exercises "
    "`{qualname}` in `{target_file}`)\n\n"
    "**Test failure output tail:**\n```\n{failure_tail}\n```\n\n"
    "**Prior judge decisions in this stage** (oscillation check — the "
    "one-regeneration cap on the test side is enforced against these):\n"
    "```json\n{prior_decisions_block}\n```"
)


JUDGE_TASK_TEMPLATE = (
    "Halt-recovery judgment at {stage_id}, iteration {iteration}. The "
    "validator `{validator_label}` just failed inside the producer "
    "fix-loop. Your job: read the validator script, read the failing "
    "artifact, read the matched taxonomy node/build plan context, then decide "
    "whether the producer can fix this (action=dispatch_fix with a "
    "structured finding) OR whether this is upstream / pipeline-bug / "
    "unclear and should halt. The stderr tail is appended below; "
    "treat it as the SYMPTOM, not the diagnosis. If the failure concerns "
    "MethodSpec methodology-fidelity fields, diagnose the whole invariant: "
    "`acceptable_approximations`, element `replication_status`, "
    "`methodology_contract_pack.approved_approximations`, and "
    "`replication_feasibility.{{verdict,approved_approximations}}` must agree. "
    "If stderr includes a "
    "`methodology_replication_contract core detail completeness failed` "
    "block, treat every listed element/field as the repair scope; do not "
    "route a one-field fix for only the first Pydantic schema error. "
    "Any dispatch_fix finding must name every dependent site, not only the "
    "first field mentioned by stderr. Write EXACTLY ONE "
    "decision object to `{decision_output_path}`. Do NOT write "
    "`<run_dir>/.pipeline/judge_decisions.json`; that canonical history "
    "is driver-owned and the driver will append your validated scratch "
    "decision to it. Do NOT edit any producer artifact. Do NOT load "
    "skills. The decision must validate against "
    "`schemas/judge_decision.py`.\n\n"
    "**CONTRACT FIELDS — COPY VERBATIM.** Your decision MUST contain:\n"
    "  - `stage_id`: `{stage_id}` (exactly — not 'stage_1a', not "
    "'{stage_id}_something', literally `{stage_id}`)\n"
    "  - `iteration`: {iteration} (the integer, not a string)\n"
    "  - `validator_label`: `{validator_label}` (exact string)\n"
    "If any of these don't match what's in this prompt, the driver "
    "will reject your scratch decision with a 'contract violation' "
    "error. If a value here seems wrong, halt instead of rewriting it.\n\n"
    "**Validator stderr tail:**\n```\n{stderr_tail}\n```\n\n"
    "**Prior judge decisions in this stage** (read these for "
    "oscillation detection — if you'd be making the same call as last "
    "iteration on the same class of failure, escalate to halt):\n"
    "```json\n{prior_decisions_block}\n```"
)


JUDGE_REVIEWER_TASK_TEMPLATE = (
    "Halt-recovery judgment at {stage_id}, iteration {iteration}. The "
    "stage-reviewer for `{reviewer_stage_id}` flagged "
    "{n_findings} critical finding(s) that have PERSISTED across the "
    "tier-3 producer fix-loop — each prior iteration dispatched a "
    "producer at the target_agent each finding declares, but the "
    "subsequent re-review surfaced the same findings (or new ones at "
    "the same severity). Your job: read the matched taxonomy node/build "
    "plan context, read the producer artifact(s) under review (declared in "
    "each finding's `file` field), read the persisting findings below, "
    "then decide whether ONE more producer dispatch can fix this "
    "(action=dispatch_fix with a structured finding) OR whether this "
    "is upstream / pipeline-bug / unclear and should halt. Write "
    "EXACTLY ONE decision object to `{decision_output_path}`. Do NOT "
    "write `<run_dir>/.pipeline/judge_decisions.json`; that canonical "
    "history is driver-owned and the driver will append your validated "
    "scratch decision to it. Do NOT edit any producer artifact. Do NOT "
    "load skills. The decision must validate against "
    "`schemas/judge_decision.py`.\n\n"
    "**CONTRACT FIELDS — COPY VERBATIM.** Your decision MUST contain:\n"
    "  - `stage_id`: `{stage_id}` (exactly — not 'stage_1a', not "
    "'{stage_id}_reviewer', literally `{stage_id}`)\n"
    "  - `iteration`: {iteration} (the integer, not a string)\n"
    "  - `validator_label`: `{validator_label}` (exact string)\n"
    "If these don't match what's in this prompt, the driver rejects "
    "your scratch decision with a 'contract violation' error.\n\n"
    "Treat the findings as the SYMPTOM, not the diagnosis: if the "
    "producer has been re-dispatched cap times and the same findings "
    "persist, the diagnostic frame is either wrong, the producer "
    "lacks context, or the fix lives upstream. Be willing to halt "
    "with a clear classification rather than burn another dispatch "
    "on a stuck loop.\n\n"
    "**Persisting critical findings (from stage_review_{reviewer_stage_id}.json):**\n"
    "```json\n{findings_block}\n```\n\n"
    "**Prior judge decisions in this stage** (read these for "
    "oscillation detection — if you'd be making the same call as last "
    "iteration on the same class of failure, escalate to halt):\n"
    "```json\n{prior_decisions_block}\n```"
)


CONTRACT_FIELDS_VERBATIM_CLAUSE = (
    "**CONTRACT FIELDS — COPY VERBATIM.** This prompt provides values for "
    "the following fields: {field_list}. When you write your output, the "
    "corresponding fields in your JSON MUST contain those values EXACTLY as "
    "they appear above — character-for-character. Do NOT paraphrase, "
    "abbreviate, capitalize differently, or invent variants (e.g., "
    "'stage_1' → 'stage_1a', 'stage_2b' → 'stage_2b_arch'). The driver "
    "looks these values up after you write the file; any mismatch will "
    "halt the pipeline. If the prompt's value seems wrong, halt and "
    "explain why — do NOT silently rewrite it."
)


INCREMENTAL_REREVIEW_TEMPLATE = (
    "Stage 4 re-review (iteration {iteration} of fix loop). The prior review "
    "found findings {prior_finding_ids} (see "
    "`<PIPELINE_DIR>/review_report.json` for the prior report and the "
    "producer's fix-mode status table appended below for what was changed). "
    "Your task is focused, but not blind to downstream effects: (a) for EACH "
    "prior finding, verify whether the fix landed — read the named file:line, "
    "confirm the change matches the "
    "proposed_fix, mark `still_present` or `resolved`; (b) scan the files "
    "the producer's status table reports as touched AND any delivered "
    "artifacts deterministically regenerated by the driver afterward "
    "(especially `notebook.ipynb`, `.pipeline/params.json`, and "
    "`assumptions.md`), looking for regressions or stale prose introduced by "
    "the fix; (c) write the new "
    "`review_report.json` with the merged finding list (resolved findings "
    "dropped, still-present findings kept, new regressions added). Do NOT "
    "re-derive findings from unrelated unchanged files; the prior reviewer "
    "already covered them."
)


PRODUCER_CLOSING = (
    "Read the inputs, follow your agent's procedure, then report back: list "
    "of files you wrote, one-line summary of each, any caveats. Do NOT run "
    "the render script or any validator."
)


REVIEWER_CLOSING = (
    "Read the inputs, follow your agent's procedure, then report back: number "
    "of findings by severity, the most important 1-3, any caveats. Do NOT run "
    "any validator."
)


# Every producer/reviewer dispatch declares the paths (relative to run_dir,
# fnmatch-style globs supported) it is allowed to write during that dispatch.
# Files modified or created outside the allowlist are out-of-scope edits;
# the driver halts with attribution so the user can correct the agent's
# prompt/contract instead of letting cross-file invariants silently break.
#
# Surfaced by bev-distill: notebook-generator edited method/model.py to
# rename `_ObjectDGCNNTeacher` → `ObjectDGCNNTeacher` (satisfying a notebook
# validator finding) without touching training.py, leaving the package's
# import chain broken; smoke gate caught it three substages later with a
# generic traceback. The fix is general: every producer is responsible only
# for its own files, and a cross-stage edit is a contract violation worth
# halting on at the point where it happens.
#
# Halt sidecars (`*.halt`) and `driver_state.json` are exempt from the check
# — see run_pipeline.py's `_is_driver_managed`. Agents may legitimately
# write `<artifact>.halt` (decomposer, analyzer, feasibility) to signal a
# hard block; we treat those as protocol traffic, not artifact writes.
WRITEABLE_PATHS: dict[str, list[str]] = {
    "r2c-decomposer": [
        ".pipeline/paper_map.json",
        ".pipeline/paper_map_parts/*.json",
    ],
    "r2c-method-analyzer": [
        ".pipeline/method_spec.json",
        ".pipeline/method_spec_parts/*.json",
        ".pipeline/paradigm_gap_report.json",
        ".pipeline/paradigm_gap_report.md",
    ],
    "r2c-architecture-coder": [
        "method/model.py",
        "method/training.py",
        ".pipeline/arch_contract.json",
    ],
    "r2c-method-coder": [
        "method/method.py",
    ],
    "r2c-notebook-generator": [
        ".pipeline/notebook_draft.py",
    ],
    "r2c-test-generator": [
        "method/tests/test_*.py",
    ],
    "r2c-method-explainer": [
        ".pipeline/method_explanations.part-*.json",
    ],
    "r2c-stage-reviewer": [
        ".pipeline/stage_review_*.json",
    ],
    "r2c-paper-fidelity-reviewer": [
        ".pipeline/review_report.json",
    ],
    "r2c-smoke-diagnostician": [
        ".pipeline/smoke_diagnosis.json",
    ],
    "r2c-halt-judge": [
        ".pipeline/judge_decision_parts/*.json",
    ],
    "r2c-insight-semantic-reviewer": [
        ".pipeline/generic_insights.semantic_review.json",
    ],
    # Shadow-tier candidate producer (Block 7 continuation): dispatched by
    # scripts/insight_shadow_hook.py, never by the stage machinery. Its one
    # artifact is the internal candidate the deterministic layer re-anchors
    # and validates.
    "r2c-insight-producer": [
        ".pipeline/generic_insights.json",
        # Corrective re-ask output (insight_shadow_hook CORRECTIONS_RELPATH).
        # Declared-consistency hygiene: nothing on the hook path reads this
        # entry, but declaring it means another agent's write to the
        # corrections path is a hard halt instead of revert-and-continue,
        # which is correct (B-13 change 2 rider).
        ".pipeline/generic_insights.corrections.json",
    ],
}


STAGE_1_OUTPUT_MODE_WRITEABLE_PATHS: dict[str, dict[str, list[str]]] = {
    "r2c-decomposer": {
        "canonical": [
            ".pipeline/paper_map.json",
        ],
        "chunk": [
            ".pipeline/paper_map_parts/*.json",
        ],
    },
    "r2c-method-analyzer": {
        "canonical": [
            ".pipeline/method_spec.json",
            ".pipeline/paradigm_gap_report.json",
            ".pipeline/paradigm_gap_report.md",
        ],
        "chunk": [
            ".pipeline/method_spec_parts/*.json",
            ".pipeline/paradigm_gap_report.json",
            ".pipeline/paradigm_gap_report.md",
        ],
    },
}


# Producer-only addendum to the writeable-paths block. It tells an agent whose
# job is to PRODUCE code/artifacts that the taxonomy `stage_review_focus`
# is the reviewer's checklist (not its own) and that producing its listed file
# is its whole job — never to ALSO emit a review of its output (B-004: the
# method-coder inlined the reviewer's job, pdwa 2.c halt). Every sentence is
# PRODUCER-framed ("producing your listed file(s) is your entire job; reviewing
# them is a separate agent's dispatch"), so it MUST NOT reach a review-only
# agent, whose listed output IS the review file and whose instruction IS
# stage_review_focus. Sent to the stage_2x reviewer it inverted the task and
# drove it to author a notebook.py (deep-batch run, 2026-06-15 halt) — see
# `_writes_only_review_artifacts`.
PRODUCER_ANTI_REVIEW_CLAUSE = (
    "The taxonomy `stage_review_focus` block is reference material "
    "describing the stage-reviewer's checks — it is NOT an instruction to "
    "you, even if you read the taxonomy/build-plan source for your own contract. Do NOT "
    "write a `stage_review_*.json`, `review_report.json`, or any "
    "review/diagnosis artifact unless it matches a pattern above. "
    "Producing your listed file(s) is your entire job; reviewing them is "
    "a separate agent's dispatch. If you notice a problem in your own "
    "output, fix it in your listed file(s) — do not also emit a review of it."
)


def _writes_only_review_artifacts(writeable_paths: list[str]) -> bool:
    """True when every writeable path is a review artifact (`stage_review_*.json`
    or `review_report.json`) — i.e. the agent IS a reviewer whose product is the
    review. Such agents must not receive the producer-framed anti-review clause,
    which would tell them their job is to produce, not review, and invert it.
    Diagnosis/decision producers (smoke_diagnosis, judge_decision_parts) are NOT
    review-only and keep the clause: producing their listed file is their job."""
    return bool(writeable_paths) and all(
        ("stage_review_" in p) or ("review_report" in p)
        or ("semantic_review" in p)
        for p in writeable_paths
    )


def format_writeable_paths_block(writeable_paths: list[str]) -> str:
    """Render the writeable-paths section of a dispatch prompt.

    Producer agents are also told in their `.opencode/agents/<agent>.md`
    contracts which files they own; this block restates the constraint
    in the dispatch itself so the agent has it in working memory while
    deciding whether to make an out-of-scope edit. Review-only agents get the
    scope constraint WITHOUT the producer-framed anti-review clause (which would
    invert their task — see PRODUCER_ANTI_REVIEW_CLAUSE)."""
    bullet_lines = "\n".join(f"  - {p}" for p in writeable_paths)
    constraint = (
        "**Writeable paths** (relative to the run dir, this dispatch only):\n"
        f"{bullet_lines}\n\n"
        "You MUST only create or modify files matching the patterns above. "
        "Editing any other file under the run dir is a contract violation; "
        "the driver halts on detection. If a fix requires a change outside "
        "your scope, do NOT make the edit — describe the needed change in "
        "your final response so the orchestrator can re-dispatch the agent "
        "that owns that file."
    )
    if _writes_only_review_artifacts(writeable_paths):
        return constraint
    return constraint + "\n\n" + PRODUCER_ANTI_REVIEW_CLAUSE


# Prepended to a dispatch when the driver's file-ownership layer caught an agent
# writing outside its scope, its required output did not validate even after the
# strays were removed (a wander, not a benign extra write), and the driver is
# giving the SAME agent one corrective re-dispatch in a fresh work session before
# halting (run_pipeline `_corrective_redispatch_after_wander`, branch A).
#
# Framing constraints, each load-bearing:
#  - AGENT-NEUTRAL. It must NOT say "producer" / "reviewer" or carry the
#    producer-vs-reviewer anti-review framing. Cross-wiring that framing onto a
#    reviewer is exactly what drove the 2026-06-15 stage-2x reviewer to author a
#    notebook.py and halt the run. This preamble is reused for any opted-in
#    agent, so it speaks only about scope, never role.
#  - PAST-TENSE + SELF-CONTAINED. The corrective dispatch runs in a fresh work
#    session, so the wandered turn is not in context. The note owns that the
#    action happened ("an earlier dispatch of this task created ...") without
#    claiming the prior turn is visible.
#  - NOT MISSING-OUTPUT framed. The agent DID act; it wrote in the wrong place.
#    Telling it "you never produced anything" would be wrong and confusing.
OUT_OF_SCOPE_CORRECTIVE_PREAMBLE = (
    "**Scope correction — read this before doing anything else.**\n\n"
    "An earlier dispatch of this task created file(s) that were "
    "outside the scope you were assigned:\n"
    "{stray_lines}\n\n"
    "The orchestrator has already removed those file(s). They were written "
    "outside your assigned scope, so do NOT recreate them and do NOT assume "
    "they exist. This is not a judgment about the content you put in them — "
    "only about where it was written.\n\n"
    "On THIS turn, the only file(s) you may create or modify are:\n"
    "{allowed_lines}\n\n"
    "Produce exactly that output and nothing else, then end your turn. If you "
    "are convinced a change is genuinely needed outside these path(s), do NOT "
    "make it — describe the needed change in your final response so the "
    "orchestrator can route it to whichever agent owns that file.\n\n"
    "---\n\n"
    "Your original task (unchanged) follows.\n\n"
)


def build_corrective_redispatch_preamble(
    strays: list[str], writeable_paths: list[str],
) -> str:
    """Render OUT_OF_SCOPE_CORRECTIVE_PREAMBLE with the removed strays and the
    positive allowlist filled in. Prepended to the agent's original prompt by
    the driver's branch-A corrective re-dispatch."""
    stray_lines = "\n".join(f"  - {s}" for s in strays) or "  - (the removed file(s))"
    allowed_lines = "\n".join(f"  - {p}" for p in writeable_paths)
    return OUT_OF_SCOPE_CORRECTIVE_PREAMBLE.format(
        stray_lines=stray_lines, allowed_lines=allowed_lines,
    )


# Prepended to a dispatch when the driver re-dispatches an agent whose prior
# attempt at the SAME task completed without producing its artifact (the
# missing-output / no-write turn class, B-002). Generalized from the
# diagnostician's write-first retry (detr 2026-07-04): the dominant cause on
# BOTH model tiers is long pre-write reasoning exhausting the turn's per-step
# output budget — the turn "completes" with nothing on disk. An identical
# re-dispatch of the same prompt tends to die identically (detr: initial
# 677.8s + retry 370.8s, both capped, zero writes); inverting the order so
# the artifact lands first is the proven recovery shape.
#
# Framing constraints, each load-bearing:
#  - ARTIFACT-KIND PARAMETRIC. `artifact_kind` names what the agent's best
#    current content is called (a "decision", a "review report") so the
#    imperfect-but-honest framing reads naturally for deciders and reviewers
#    alike. The diagnostician keeps its bespoke SMOKE_DIAGNOSIS_RETRY_PREAMBLE
#    (schema-field detail); this template is the generalization of its shape.
#  - AGENT-NEUTRAL + ROLE-NEUTRAL, like the corrective preamble: no
#    producer/reviewer framing that could invert a reviewer's task.
#  - Allows ONE refining re-Write so write-first doesn't force the agent to
#    ship a conclusion it immediately finds wrong — but the first Write is
#    non-negotiable and comes before any further reading.
WRITE_FIRST_RETRY_PREAMBLE = (
    "**RETRY — WRITE FIRST.** A previous dispatch of this exact task "
    "completed WITHOUT producing `{artifact_name}`. The most common cause "
    "is long pre-write analysis: the turn's output budget is exhausted by "
    "reasoning and the turn ends with NOTHING on disk. This retry inverts "
    "the order: your FIRST tool call must be the Write that creates "
    "`{artifact_name}` at the path named in the task below, carrying your "
    "best CURRENT {artifact_kind} from the evidence already in this "
    "prompt. An imperfect but honest {artifact_kind} the driver can act "
    "on beats a perfect analysis that never lands. Keep any pre-write "
    "deliberation to a few sentences; do NOT read additional files first; "
    "do NOT load any skills. If brief follow-up reading afterward "
    "materially changes your conclusion, you may re-Write the file once — "
    "but the first Write comes first.\n\n"
    "---\n\n"
    "Your original task (unchanged) follows.\n\n"
)


def build_write_first_retry_preamble(
    *, artifact_name: str, artifact_kind: str,
) -> str:
    """Render WRITE_FIRST_RETRY_PREAMBLE for a missing-output re-dispatch.
    Prepend the result to the agent's original prompt. `artifact_name` is
    the file the prior turn failed to write (basename or path);
    `artifact_kind` is what its content is called (e.g. "decision",
    "review report")."""
    return WRITE_FIRST_RETRY_PREAMBLE.format(
        artifact_name=artifact_name, artifact_kind=artifact_kind,
    )


# Fix-mode sibling of WRITE_FIRST_RETRY_PREAMBLE. Same class, different
# first action: fix mode operates on an EXISTING artifact, so the proven
# recovery is a targeted edit, not a from-scratch write. Position is the
# load-bearing part: overnight 2026-07-08 (ACC2021_MPC_CBF, SRL) four
# fix-turns died at the per-step output cap with this same instruction
# present but BURIED inside the findings block — while the halt-judge's
# retry, which leads with its write-first preamble at the very top of the
# prompt, recovered in 14.5s in the same run. The preamble is triggered by
# the `_act_first_retry` sentinel on a finding (set by the driver's
# no-write retry path) and never rendered into the findings block.
FIX_MODE_ACT_FIRST_RETRY_PREAMBLE = (
    "**RETRY — ACT FIRST.** A previous dispatch of this exact fix task "
    "completed WITHOUT writing anything. The dominant cause is long "
    "pre-write reasoning exhausting the turn's output budget, so the turn "
    "ends with NOTHING on disk. This retry inverts the order — your FIRST "
    "tool call this turn must change your owned artifact:\n"
    "  - Artifact already on disk: apply the FIRST validator-named fix as "
    "ONE targeted edit call, immediately, from the evidence already in "
    "this prompt. Then continue finding-by-finding, one edit per finding. "
    "Do NOT compose the complete fixed file before acting.\n"
    "  - Artifact missing: ONE write call producing a minimal "
    "syntactically-complete skeleton (docstring, imports, the exact "
    "contract signatures, NotImplementedError bodies). Do NOT implement "
    "the algorithm this turn; later fix-loop turns fill it in.\n"
    "Keep pre-edit deliberation to a few sentences, and re-read only the "
    "artifact section you are about to edit.\n\n"
    "---\n\n"
    "Your original fix task (unchanged) follows.\n\n"
)


# Tool-call-first writing discipline for file-producing agents (cap-burn
# measurement note, fix 1 — approved 2026-07-17). Every fatal cap burn in the
# July 13-15 ledger (19 of 19, ~380k output tokens, 22.6% of the window's
# output) was the same shape: the agent composed a large file as prose or
# inline code, never issued the write tool call, and the step died at the
# per-step output cap with nothing on disk. Post-input-hygiene the burn mass
# sits on the coder agents (method-coder 2.c, architecture-coder 2.b)
# composing big source files. The rule is family- and paper-agnostic: land
# the write call early and build large files in bounded pieces so no single
# call (and no pre-write ramble) can approach the cap. Opt-in via
# `build_dispatch_prompt(writing_discipline=True)` — file-producing agents
# only; reviewer/judge dispatches write one small artifact each and keep
# their own write-first instructions.
FILE_WRITING_DISCIPLINE = """## File-writing discipline (tool calls first — read before composing)

Your turn has a finite per-step output budget. Composing a whole file in
prose or inline code BEFORE touching your write tool is the known fatal
pattern: the step exhausts its budget mid-composition, no tool call ever
lands, and the dispatch ends with NOTHING on disk.

1. Issue your first write tool call BEFORE any extended reasoning. As soon
   as you know a file's skeleton (docstring, imports, signatures), write
   that skeleton to disk — then build the file up.
2. Write large files in bounded chunks: create the file with its first
   section, then extend it section by section in separate tool calls
   (append in parts). Never emit one giant call carrying an entire large
   file — an oversized call truncates at the output cap and lands nothing.
3. Keep pre-write deliberation short, and reason BETWEEN tool calls rather
   than in place of them. Code that only exists in your reasoning does not
   exist."""


# Corrective-resume nudge for a fatal cap burn (cap-burn measurement note,
# fix 2 — approved 2026-07-17). Posted by the driver into the SAME work
# session that just died at the per-step output cap without landing a write
# (turn shape cap_burn), exactly once. A plain stage-owned re-roll of the
# same prompt tends to reproduce the same ramble (SRL 2026-07-15: three
# consecutive identical method-coder burns); resuming the session instead
# converts the already-paid composition tokens into a landed artifact.
# Deliberately short, agent-neutral, family- and paper-agnostic: the session
# already holds the full task context.
CAP_BURN_CORRECTIVE_RESUME_NUDGE = (
    "**Corrective resume.** Your last step hit the per-step output cap "
    "without landing a tool call — either the write was never issued, or "
    "its JSON outgrew the cap and arrived cut off. Nothing from that step "
    "reached disk. Do NOT restart your analysis and do NOT read more "
    "files: issue the write tool call NOW for what you have already "
    "composed, in bounded chunks — create the file with its first "
    "section, then extend it in separate calls so no single call "
    "approaches the cap. When your task's listed file(s) are on disk, "
    "end your turn."
)

# The same recovery for the OTHER shape of dead turn (R2C-060): the turn
# ended under the cap without ever issuing its write. The wording must not
# mention the cap (it was not hit) and must not invite more investigation:
# the whole point is that the reads are already done and sitting in this
# session's context.
SHORT_EMPTY_CORRECTIVE_RESUME_NUDGE = (
    "**Corrective resume.** Your turn ended without writing any of the "
    "files your task requires, so nothing reached disk. The investigation "
    "you already did is still in this session's context. Do NOT start over "
    "and do NOT read more files: write your conclusion NOW from what you "
    "have, in bounded chunks — create the file with its first section, then "
    "extend it in separate calls. If your analysis is genuinely incomplete, "
    "write what you DO know and say plainly which part is unresolved; an "
    "honest partial artifact is worth far more than another empty turn. "
    "When your task's listed file(s) are on disk, end your turn."
)


# Post-install classification binding for the stage 1 analyzer retry
# (R2C-031, gap-pack retry contract). When the driver has validated and
# installed a run-local provisional pack, the gap decision has already been
# taken THIS run — a previous analyzer pass took the halt path, its proposal
# validated, and the retry exists only to produce method_spec.json on top of
# the pack. Without this block the retry re-ran first-pass classification
# from scratch in a fresh session: the reserved-target halt rule (2026-07-05)
# carries no installed-pack exception, so whenever the pack's target sat in a
# different branch than the route_elsewhere signal's named home, the analyzer
# re-halted on the rule and the run died with a valid pack on disk (SRL
# 2026-07-28; the mid-July SRL retries passed only because their packs
# populated the exact reserved node the signal named). The driver states the
# resolved classification as fact from the installed manifest, so the retry
# stops being a fresh model judgment.
PACK_INSTALLED_CLASSIFICATION_BLOCK = (
    "**Classification is resolved for this run.** A provisional pack "
    "authored for THIS paper was validated and installed by the driver at "
    "`{pack_path}` (the Taxonomy/build-plan source above). Its target id is "
    "`{target_paradigm_id}`. Produce method_spec.json classifying into "
    "exactly `{target_paradigm_id}`, with the pack as your classification "
    "and build context. Do NOT re-evaluate `route_elsewhere` signals "
    "against the registered taxonomy, and do NOT write paradigm-gap "
    "artifacts: the reserved-routing-target rule governs first-pass "
    "classification with no installed pack, and this dispatch is past that "
    "decision — the gap that rule detects is exactly what this pack "
    "resolves. The only remaining honest halt: if the installed pack itself "
    "cannot describe this paper's method, write method_spec.json.halt "
    "naming the pack and the specific mismatch."
)


@dataclass(frozen=True)
class DispatchPaths:
    """Run-specific paths injected into a dispatch prompt's paths block."""

    spec: str
    paper: str
    paper_map: str
    run_dir: str
    taxonomy_source: str
    params: str | None = None
    notebook: str | None = None
    method_dir: str | None = None
    review_report: str | None = None

    def to_block(self) -> str:
        lines = [
            "**Run paths:**",
            f"- Spec: `{self.spec}`",
            f"- Paper: `{self.paper}`",
            f"- Paper map: `{self.paper_map}`",
            f"- Run dir: `{self.run_dir}`",
            f"- Taxonomy/build-plan source: `{self.taxonomy_source}`",
        ]
        if self.params:
            lines.append(f"- Params: `{self.params}`")
        if self.notebook:
            lines.append(f"- Notebook: `{self.notebook}`")
        if self.method_dir:
            lines.append(f"- Method dir: `{self.method_dir}`")
        if self.review_report:
            lines.append(f"- Prior review report: `{self.review_report}`")
        return "\n".join(lines)


def build_dispatch_prompt(
    *,
    task_summary: str,
    paths: DispatchPaths,
    closing: str = PRODUCER_CLOSING,
    writeable_paths: list[str] | None = None,
    think_anchor_output_path: str | None = None,
    extra_sections: list[str] | None = None,
    writing_discipline: bool = False,
) -> str:
    """Assemble a standard producer-dispatch prompt: scope + task +
    (writing discipline for file-producing agents) + (think anchor for
    Think-class agents) + paths + (writeable_paths) + closing.

    `think_anchor_output_path` enables the shared Think-class procedural-
    anchor block (THINK_CLASS_ANCHOR). Pass the agent's expected single
    output path when dispatching any Think-class agent (decomposer,
    analyzer, stage-reviewer, smoke-diagnostician, paper-fidelity-reviewer).
    The anchor is the cross-cutting fix for the after-write drift pattern
    surfaced in three different Think-class agents (see the constant's
    docstring + process/taxonomy-pack-authoring-process.md).

    `writing_discipline` enables the tool-call-first FILE_WRITING_DISCIPLINE
    block. Opt in for agents that compose large files (architecture-coder,
    method-coder, notebook-generator) — the post-hygiene fatal cap-burn
    sites. Deliberately NOT default-on: reviewer/judge/diagnostician
    dispatches each write one small artifact and carry their own write-first
    instructions; their prompts must not change (see the cap-burn
    measurement note)."""
    sections = [SCOPE_CONTRACT, task_summary]
    if writing_discipline:
        # Early position is load-bearing: the same instruction buried late in
        # a fix prompt did not prevent the overnight 2026-07-08 burns, while
        # the judge's top-of-prompt write-first preamble recovered in 14.5s.
        sections.append(FILE_WRITING_DISCIPLINE)
    if think_anchor_output_path:
        sections.append(THINK_CLASS_ANCHOR.format(
            expected_output_path=think_anchor_output_path
        ))
    sections.append(paths.to_block())
    if extra_sections:
        # Caller-supplied resource context built at the call site (e.g. the
        # stage-2.b/2.c closed-set blocks: valid paper-element IDs + verbatim
        # essential strings — materialized from the artifacts so the producer
        # copies tokens rather than regenerating them from memory). Sits with
        # the resource paths so the agent reads it before the closing.
        sections.extend(s for s in extra_sections if s)
    if writeable_paths:
        sections.append(format_writeable_paths_block(writeable_paths))
    sections.append(closing)
    return "\n\n".join(sections)


def format_symbol_ownership_block(symbol_ownership: dict[str, str]) -> str:
    """Render the naming-bridge symbol-ownership section of a fix-mode prompt.

    `symbol_ownership` maps each spec-promised public symbol to the ONE
    method/ module that defines it right now (derived read-only by
    scripts/symbol_ownership.py from the same helpers the 2.b gate and the
    2.d finalizer resolve ownership with). Motivating case (fedavg
    2026-07-21): a stage-5 fix dispatch to the architecture coder ADDED
    `federated_train` to training.py while method.py had owned it since
    stage 2.c — the ownership re-check then halted the run on "defined in
    2 files — ambiguous ownership". The fix producer could not know it
    should import rather than re-define, because the prompt did not carry
    the map. Callers render this block only when the map is non-empty, so
    runs without bridge data keep byte-identical fix prompts."""
    bullet_lines = "\n".join(
        f"  - `{symbol}` — defined in `{module}`"
        for symbol, module in symbol_ownership.items()
    )
    return (
        "**Symbol-ownership map** (naming bridge: each spec-promised symbol "
        "and the one module that defines it):\n"
        f"{bullet_lines}\n\n"
        "If a fix needs one of these symbols in a file that does not define "
        "it, IMPORT the symbol from its owning module — do NOT write a new "
        "definition. Creating a second definition of an owned symbol is "
        "itself a defect: the pipeline's ownership check fails the package "
        "on \"defined in 2 files — ambiguous ownership\" and halts the run, "
        "even when the fix is otherwise correct."
    )


def _bridge_symbol_ownership(paths: DispatchPaths) -> dict[str, str]:
    """Best-effort naming-bridge ownership lookup for a fix dispatch.

    Delegates to scripts/symbol_ownership.py (lazy import keeps this
    module import-pure and dependency-light). ANY failure — helper module
    unavailable, unreadable run dir, malformed spec — yields {}: prompt
    assembly must never break, and a run without derivable bridge data
    must render byte-identical fix prompts (no block at all)."""
    try:
        try:
            from symbol_ownership import load_bridge_symbol_ownership
        except ImportError:
            from scripts.symbol_ownership import load_bridge_symbol_ownership
        return load_bridge_symbol_ownership(paths.spec, paths.run_dir)
    except Exception:
        return {}


def build_fix_mode_prompt(
    *,
    target_agent: str,
    findings: list[dict],
    paths: DispatchPaths,
    writeable_paths: list[str] | None = None,
    smoke_context: dict | None = None,
    think_anchor_output_path: str | None = None,
    extra_sections: list[str] | None = None,
) -> str:
    """Assemble a fix-mode prompt: scope + fix-mode-skeleton + (smoke
    guidance) + findings + paths + (symbol-ownership map) + (writeable_paths)
    + closing.

    When `smoke_context` is provided, the SMOKE_FIX_GUIDANCE block is
    inserted between the fix-mode skeleton and the findings — used by the
    smoke-gate fix path (`run_stage_3c`) to teach the producer about data-
    flow tracing and the architecture-is-the-contract rule. The dict may
    contain a `traceback_hint` field naming the deepest traceback frame in
    the producer's owned files; if present, it's rendered as an anchor
    line at the end of the guidance block.

    The symbol-ownership map is derived from `paths` (spec + run dir) at
    assembly time — see _bridge_symbol_ownership — and appears only when
    the run has derivable naming-bridge data; otherwise the prompt is
    byte-identical to the pre-map assembly."""
    if not findings:
        raise ValueError("build_fix_mode_prompt requires at least one finding")
    act_first_retry = any(f.get("_act_first_retry") for f in findings)
    findings_block = "**Fix findings from validator/reviewer:**\n\n"
    for i, f in enumerate(findings, start=1):
        fid = f.get("id", f"F{i:03d}")
        sev = f.get("severity", "unknown")
        desc = f.get("description", "")
        fix = f.get("proposed_fix", "")
        findings_block += f"{i}. **{fid} ({sev})**: {desc}\n"
        if fix:
            findings_block += f"   Proposed fix: {fix}\n"
        findings_block += "\n"
    sections = [
        SCOPE_CONTRACT,
        FIX_MODE_BODY.format(target_agent=target_agent),
    ]
    if target_agent == "r2c-method-analyzer":
        # Stage-1 strict failures route back through this generic fix builder.
        # Re-state the exact-ref producer contract here so a repair cannot
        # satisfy one finding by guessing or broadly copying refs across
        # obligations that share a callable/paper anchor.
        sections.append(ANALYZER_VERIFICATION_PROBE_GUIDANCE)
        sections.append(ANALYZER_CALIBRATION_CONTEXT_GUIDANCE)
        sections.append(ANALYZER_GRAPH_MECHANISM_GUIDANCE)
    elif target_agent in {"r2c-architecture-coder", "r2c-method-coder"}:
        sections.append(GRAPH_MECHANISM_IMPLEMENTATION_GUIDANCE)
    elif target_agent == "r2c-notebook-generator":
        sections.append(GRAPH_MECHANISM_NOTEBOOK_GUIDANCE)
    if think_anchor_output_path:
        sections.append(THINK_CLASS_ANCHOR.format(
            expected_output_path=think_anchor_output_path
        ))
    if smoke_context is not None:
        hint = (smoke_context.get("traceback_hint") or "").strip()
        hint_block = (
            f"\n\n**Deepest traceback frame in your owned files:** `{hint}`\n"
            "Start your investigation there; verify the bug's actual location "
            "via the (i)–(iv) checklist before editing."
            if hint
            else ""
        )
        sections.append(SMOKE_FIX_GUIDANCE.format(traceback_hint_block=hint_block))
    sections.extend([
        "---",
        findings_block.rstrip(),
        paths.to_block(),
    ])
    if extra_sections:
        # Same closed-set resource context as build_dispatch_prompt — re-injected
        # in fix mode so a later fix iteration cannot re-introduce an abbreviated
        # ID or paraphrased essential string the first dispatch got right.
        sections.extend(s for s in extra_sections if s)
    symbol_ownership = _bridge_symbol_ownership(paths)
    if symbol_ownership:
        # Naming-bridge ownership context (fedavg 2026-07-21, overnight-0721
        # fix proposal 3): tells the fix producer which module already owns
        # each spec-promised symbol, so the fix imports instead of
        # re-defining. Rendered ONLY when the run has derivable bridge data;
        # runs without it keep byte-identical fix prompts.
        sections.append(format_symbol_ownership_block(symbol_ownership))
    if writeable_paths:
        sections.append(format_writeable_paths_block(writeable_paths))
    sections.append(
        "Read the inputs, apply the fixes, then report back the status "
        "table. Do NOT run the render script or any validator."
    )
    prompt = "\n\n".join(sections)
    if act_first_retry:
        # Top of the prompt, ahead of everything — the position is what
        # separated the judge's healed cap-burn turns from the fix loop's
        # fatal ones (overnight 2026-07-08).
        prompt = FIX_MODE_ACT_FIRST_RETRY_PREAMBLE + prompt
    return prompt


SMOKE_DIAGNOSIS_TASK = (
    "Stage 3.c smoke-gate diagnosis. The notebook executed end-to-end and a "
    "cell raised an exception. Apply Steps 1–7 of your agent procedure: "
    "(1) read spec + matched taxonomy/build-plan source for domain context — "
    "consult the effective `common_smoke_bugs` block from the served node or "
    "run-local pack, "
    "(2) read the failing cell, (3) trace the failing value to its origin "
    "using `.pipeline/data_flow.json` (the driver pre-extracts per-symbol "
    "assignments + reads), (4) distinguish surface from bug + apply the "
    "architecture-as-contract rule, (5) consult prior iterations if present, "
    "(6) pick target_agent + target_file, (7) argue paper-fidelity and "
    "write the diagnosis. USE YOUR WRITE TOOL on "
    "`<run_dir>/.pipeline/smoke_diagnosis.json` ONLY. Do not Edit any source "
    "file. Do not draft a fix-mode status table.\n\n"
    "**Budget your pre-write reasoning.** Your turn has a finite output "
    "budget, and on a hard bug a full differential analysis can exhaust it "
    "BEFORE any Write lands — the turn then ends with nothing on disk and "
    "the fix loop stalls. If your analysis is running long, WRITE the "
    "diagnosis now with your best current hypothesis, then continue only "
    "if needed and re-Write the file at most ONCE if further analysis "
    "materially changes it. A landed imperfect diagnosis beats an "
    "unwritten perfect one.\n\n"
    "**Verify contract vs code before concluding mismatch.** When the "
    "error looks like a shape/dtype/range mismatch against a declared "
    "contract (arch_contract.json, taxonomy build plan), READ the actual receiver "
    "code at the error site BEFORE concluding the input is wrong. Models "
    "often explicitly handle shape variants the contract doesn't enumerate "
    "(e.g., `if x.dim() == 5: x = x.mean(dim=1)` to accept both "
    "`(B, C, H, W)` and `(B, V, C, H, W)`). If the receiver code already "
    "handles the variant being passed, the bug is NOT a contract violation. "
    "Look elsewhere — most often dtype (`torch.randint` defaults to int, but "
    "`nn.Conv2d`/`nn.Linear` require float), value range, normalization, "
    "or a downstream broadcasting issue. The contract is a starting point; "
    "the implementation code is the ground truth.\n\n"
    "**Required fields in the diagnosis JSON (schema-enforced — missing any "
    "of these halts the pipeline at schema validation):**\n"
    "  - `schema_version`: `\"1.0.0\"`\n"
    "  - `target_agent`: one of `r2c-method-coder`, `r2c-architecture-coder`, "
    "`r2c-notebook-generator`\n"
    "  - `target_file`: relative path within run_dir, in target_agent's "
    "writeable_paths\n"
    "  - `bug_shape`: a key from the taxonomy node's `common_smoke_bugs` block "
    "(or "
    "`uncatalogued` if no key matches)\n"
    "  - `root_cause`: 1-3 sentences naming the actual bug (NOT the symptom)\n"
    "  - `proposed_fix`: 1-3 sentences naming a concrete change\n"
    "  - `value_origin_trace`: list of AT LEAST 3 strings, ordered from "
    "the failure site walking upstream to the value's first origin (use "
    "data_flow.json for lookups)\n"
    "  - `reasoning`: a prose summary of the analysis and scope "
    "verification. The step-by-step trace does NOT go here — it goes in "
    "`value_origin_trace` above, and writing it here does not satisfy "
    "that field\n"
    "  - `paper_fidelity_check`: argues the fix is paper-faithful OR "
    "explicitly flags a smoke-scale deviation and defends it"
)


def _smoke_routable_writeable_block(writeable_paths_map: dict[str, list[str]]) -> str:
    """Render the smoke-routable subset of WRITEABLE_PATHS for the diagnostician.

    The diagnostician picks one of three target_agent values; this block
    tells it which file paths each owns, so it can pick a `target_file`
    that the next dispatch's scope check will accept."""
    lines = ["**File-ownership map** (smoke-routable agents only):"]
    for agent in ("r2c-method-coder", "r2c-architecture-coder", "r2c-notebook-generator"):
        owned = writeable_paths_map.get(agent, [])
        joined = ", ".join(f"`{p}`" for p in owned)
        lines.append(f"  - `{agent}` owns: {joined}")
    return "\n".join(lines)


def _format_prior_iterations_block(prior_iterations: list[dict]) -> str:
    """Render the prior-iteration history block for the diagnostician's
    prompt. Each entry must have: iteration, failing_cell, section,
    mechanical_routing, diagnosis (dict), producer_dispatched.

    The diagnostician at iteration N consumes this to decide whether the
    current bug is a regression of a prior fix, a new layer the prior fix
    exposed, or evidence the prior routing was wrong. Without history each
    iteration re-derives context from session conversation, which is the
    contamination path that caused iter-2 of the bev-distill 2026-05-14 run
    to follow the producer's (a)-(e) skeleton instead of its own
    Steps 1-7 procedure."""
    if not prior_iterations:
        return ""
    lines = [
        "**Prior iterations in this smoke fix loop (THIS run):**",
        "",
        "Read this carefully. It tells you whether your job here is to:",
        "  (a) diagnose a regression — the prior fix didn't land",
        "  (b) diagnose a new layer — the prior fix worked and exposed this",
        "  (c) re-route — the prior fix went to the wrong agent",
        "",
    ]
    for entry in prior_iterations:
        diag = entry.get("diagnosis") or {}
        cell_line = f"  - Failing cell: {entry['failing_cell']} (§{entry['section']})"
        if entry.get("exception_class"):
            cell_line += f" raised `{entry['exception_class']}`"
        lines.extend([
            f"Iteration {entry['iteration']}:",
            cell_line,
            f"  - Mechanical routing: {entry['mechanical_routing']}",
            f"  - Diagnostician picked: {diag.get('target_agent', '?')} "
            f"(target_file=`{diag.get('target_file', '?')}`)",
            f"  - Root cause: {diag.get('root_cause', '?')}",
            f"  - Proposed fix: {diag.get('proposed_fix', '?')}",
            f"  - Producer dispatched: {entry.get('producer_dispatched', '?')}",
        ])
        if entry.get("data_signal"):
            # The fix-loop data-signal verdict (demo-success design): the
            # demo data this iteration's fix installed can never pass the
            # beats-chance gate — the diagnosis must target the DATA, not
            # keep re-diagnosing the model against an unwinnable gate.
            indented = "\n".join(
                f"    {ln}" for ln in str(entry["data_signal"]).splitlines())
            lines.append(f"  - Data-signal verdict after this fix:\n{indented}")
        lines.append("")
    return "\n".join(lines).rstrip()


SMOKE_DIAGNOSIS_RETRY_PREAMBLE = (
    "**Stage 3.c diagnostician RETRY.** Your previous dispatch did NOT "
    "produce `smoke_diagnosis.json`: it analyzed at length and never "
    "called Write. Long pre-write analysis exhausts the turn's output "
    "budget and the turn dies with NOTHING on disk. This retry inverts "
    "the order: your FIRST tool call must be the Write that creates "
    "`smoke_diagnosis.json` at the path below, carrying your BEST "
    "CURRENT HYPOTHESIS from the failing cell and traceback. An "
    "imperfect but honest diagnosis the driver can route to a producer "
    "beats a perfect analysis that never lands. Keep any pre-write "
    "deliberation to a few sentences; do NOT read additional files "
    "first; do NOT load any skills. The schema is "
    "schemas/smoke_diagnosis.py and EVERY required field must be in the "
    "Write: schema_version + target_agent + target_file + bug_shape + "
    "root_cause + proposed_fix + value_origin_trace (a list of at least "
    "3 strings) + reasoning + paper_fidelity_check. If brief follow-up "
    "reading afterward materially changes your diagnosis, you may "
    "re-Write the file once — but the first Write comes first."
)
# (The field list above is the FULL required set on purpose: the 07-13
# ICRA write-first retry produced an otherwise-valid file missing exactly
# value_origin_trace, and this preamble's old field list was the one
# place that field went unmentioned.)


SMOKE_DIAGNOSIS_SCHEMA_REPAIR_PREAMBLE = (
    # Pure-merge repair (ICRA 2026-07-13): the old wording asked the agent
    # to READ its prior file first, and the live repair turn read five
    # files, announced the write, and dead-stopped at one output token —
    # nothing landed and the run halted on the stale file. The prior JSON
    # now rides the prompt so the FIRST and only action is the Write.
    "**Stage 3.c diagnostician SCHEMA REPAIR.** Your previous dispatch "
    "DID write `smoke_diagnosis.json`, but the halt-judge inspected it "
    "and the file failed schema validation. The judge's findings are:\n\n"
    "{repair_findings_block}\n\n"
    "Your prior diagnosis is reproduced below IN FULL — do NOT read any "
    "files first. Your FIRST tool call must be a single Write of "
    "`smoke_diagnosis.json` containing this same JSON with the "
    "missing/invalid field(s) corrected. KEEP every other field "
    "**unchanged** unless a finding specifically calls it out; do NOT "
    "re-analyze the smoke failure — the diagnosis content was accepted, "
    "only the schema fields were wrong.\n\n"
    "**Your prior diagnosis (merge the fixes into this):**\n"
    "```json\n{prior_diagnosis_json}\n```"
)


def _format_forbidden_pick_retry_block(forbidden_pick: str) -> str:
    """Render the forbidden-target retry preamble. Used when the driver
    re-dispatches the diagnostician after its prior pick was in the
    failed_targets set. The agent must reconsider — especially via the
    consumer-side hypothesis — and produce a Tier 2 differential
    diagnosis (top-3 hypotheses with source quoting per the agent's
    Step 6 contract)."""
    return (
        "**Stage 3.c diagnostician FORBIDDEN-TARGET RETRY.** Your previous "
        f"dispatch this iteration picked `target_file = {forbidden_pick}`. "
        "That file is in the failed_targets list — it was already tried in "
        "an earlier iteration AND the smoke gate still failed at the same "
        "cell after that fix attempt landed. Picking it again is highly "
        "unlikely to converge.\n\n"
        "**You are now in Tier 2 of the differential diagnosis** (per your "
        "Step 6 contract). REQUIRED for this retry:\n\n"
        "1. Generate AT LEAST three hypotheses for the failure. For each, "
        "quote the specific lines of source code from the file you suspect "
        "that demonstrate the bug. Rank them by evidence strength.\n"
        "2. Explicitly include a **consumer-side hypothesis** — the bug "
        f"may live in a *consumer* of `{forbidden_pick}`, not in "
        f"`{forbidden_pick}` itself. A consumer is any file that imports "
        f"from, calls into, or interprets the output of `{forbidden_pick}`. "
        "The notebook's acquisition / training loop is the most common "
        "consumer; check it carefully for in-place mutations, index "
        "manipulations, or iteration patterns that misuse the producer's "
        "correct return.\n"
        "3. If after the Tier 2 walk your top hypothesis STILL points at "
        f"`{forbidden_pick}`, you MUST quote the specific line(s) that the "
        "previous fix-mode dispatch failed to address. If you cannot quote "
        "such a line, your top hypothesis is wrong — pick a different "
        "target.\n\n"
        "Apply the mutation-and-iteration hazard rule (Step 6) carefully: "
        "if the failure trace points at a line inside a loop that mutates "
        "one of the values it iterates against (`.pop`, `del`, in-place "
        "index manipulation), the bug is almost always in the consumer's "
        "mutation pattern, not in the producer of the iteration values."
    )


def build_smoke_diagnosis_prompt(
    *,
    paths: DispatchPaths,
    stderr_tail: str,
    failing_cell_source: str,
    failing_cell_index: int,
    section: int,
    diagnosis_output_path: str,
    data_flow_path: str | None = None,
    data_flow_slice_path: str | None = None,
    prior_iterations: list[dict] | None = None,
    failed_targets: list[str] | None = None,
    retry_mode: bool = False,
    forbidden_pick_retry: str | None = None,
    repair_findings: list[dict] | None = None,
    prior_diagnosis_json: str | None = None,
) -> str:
    """Assemble the smoke-gate diagnostician dispatch prompt.

    Carries the exact failing cell source, the stderr/traceback tail, the
    WRITEABLE_PATHS map, and (when present) a structured summary of
    prior fix-loop iterations in this same run. The diagnostician runs on
    a Think-class model and writes smoke_diagnosis.json; the driver
    consumes it on return."""
    failing_cell_block = (
        f"**Failing cell:** index {failing_cell_index} "
        f"(§{section} of the notebook layout)\n\n"
        f"**Cell source:**\n```python\n{failing_cell_source.rstrip()}\n```"
    )
    stderr_block = (
        f"**Smoke gate stderr (traceback tail):**\n"
        f"```\n{stderr_tail.rstrip()}\n```"
    )
    output_instruction = (
        f"**Output path:** Use your Write tool to save the diagnosis to "
        f"`{diagnosis_output_path}`."
    )
    sections = [SCOPE_CONTRACT]
    if repair_findings:
        repair_lines = []
        for i, f in enumerate(repair_findings, start=1):
            fid = f.get("id", f"J{i:03d}")
            sev = f.get("severity", "critical")
            desc = f.get("description", "")
            fix = f.get("proposed_fix", "") or ""
            repair_lines.append(f"  {i}. **{fid} ({sev})**: {desc}")
            if fix:
                repair_lines.append(f"     Proposed fix: {fix}")
        sections.append(SMOKE_DIAGNOSIS_SCHEMA_REPAIR_PREAMBLE.format(
            repair_findings_block="\n".join(repair_lines),
            prior_diagnosis_json=(
                prior_diagnosis_json
                or "(prior file unreadable — reconstruct from the findings "
                   "above and the failing cell, then Write immediately)"),
        ))
    elif retry_mode:
        sections.append(SMOKE_DIAGNOSIS_RETRY_PREAMBLE)
    if forbidden_pick_retry:
        sections.append(_format_forbidden_pick_retry_block(forbidden_pick_retry))
    sections.extend([
        SMOKE_DIAGNOSIS_TASK,
        THINK_CLASS_ANCHOR.format(expected_output_path=diagnosis_output_path),
    ])
    if prior_iterations:
        sections.append(_format_prior_iterations_block(prior_iterations))
    if failed_targets:
        sections.append(
            "**Failed targets — DO NOT pick these for this iteration:**\n\n"
            + "\n".join(f"  - `{t}`" for t in failed_targets)
            + "\n\nThe smoke gate kept failing at the same cell after a fix-mode "
            "dispatch targeted each of these files. Either the diagnosis was "
            "wrong (the bug isn't actually in those files) OR the producer's "
            "fix implementation was wrong — in either case, picking the same "
            "target again is highly unlikely to converge in the remaining "
            "fix-loop budget. Your `target_file` for THIS iteration must NOT "
            "be in the list above. Use the data-flow context to find the "
            "next candidate upstream. The driver will halt the pipeline if "
            "your target_file is in this list."
        )
    if data_flow_slice_path:
        sections.append(
            "**Data-flow context (READ THIS FIRST when tracing a value's origin):**\n\n"
            f"The driver pre-sliced the data-flow map to exactly the symbols "
            f"named in the failing cell and the traceback (plus one upstream "
            f"hop through their assignment expressions) at "
            f"`{data_flow_slice_path}`. Read the SLICE, not the full map — "
            f"the full map at `{data_flow_path or 'the .pipeline dir'}` can "
            f"be hundreds of KB and reading it whole burns your output "
            f"budget; look up a symbol there ONLY if it is missing from the "
            f"slice. Each symbol has `assignments` (where it's set, with the "
            f"RHS expression) and `reads` (where it's used).\n\n"
            f"**Use case for THIS dispatch:** when the traceback names a "
            f"variable that's wrong (a label out of range, a tensor with "
            f"unexpected shape, an index out of bounds), look that "
            f"variable up in the slice's `symbols` and walk the "
            f"`assignments` chain to find the upstream derivation. The "
            f"bug almost always lives at the derivation site, NOT at the "
            f"validation site where the error fires."
        )
    elif data_flow_path:
        sections.append(
            "**Data-flow context (READ THIS FIRST when tracing a value's origin):**\n\n"
            f"The driver pre-extracted a structured map of every named "
            f"symbol's assignment and read sites across the notebook draft "
            f"+ method/*.py at `{data_flow_path}`. Use this to answer "
            f"'where did this value come from?' without reading every file. "
            f"Each symbol has `assignments` (where it's set, with the RHS "
            f"expression) and `reads` (where it's used).\n\n"
            f"**Use case for THIS dispatch:** when the traceback names a "
            f"variable that's wrong (a label out of range, a tensor with "
            f"unexpected shape, an index out of bounds), look that "
            f"variable up in data_flow.json's `symbols` and walk the "
            f"`assignments` chain to find the upstream derivation. The "
            f"bug almost always lives at the derivation site, NOT at the "
            f"validation site where the error fires.\n\n"
            f"Example: if `train_from_scratch` raises 'y_train labels out "
            f"of [0, n_classes)', look up `n_classes` in data_flow.json. "
            f"If its assignment is in notebook_draft.py (e.g., from "
            f"`len(y_pool.unique())`), the bug is the notebook's "
            f"derivation — route to `r2c-notebook-generator`, NOT to the "
            f"file where CrossEntropyLoss validated."
        )
    sections.extend([
        failing_cell_block,
        stderr_block,
        _smoke_routable_writeable_block(WRITEABLE_PATHS),
        output_instruction,
        paths.to_block(),
        format_writeable_paths_block(WRITEABLE_PATHS["r2c-smoke-diagnostician"]),
        (
            "Apply Steps 1–7 of your agent procedure. USE YOUR WRITE TOOL "
            "on `smoke_diagnosis.json` only — write it once, with at most "
            "one refining re-Write if late analysis materially changes it; "
            "if your analysis runs long, write your best current hypothesis "
            "BEFORE it exhausts the turn. Do not Edit any "
            "source file. Do not draft a fix-mode status table — that's the "
            "producer agent's procedure, not yours. After the diagnosis is "
            "written, return."
        ),
    ])
    return "\n\n".join(sections)


INSIGHT_REVIEW_TASK_TEMPLATE = (
    "## Task: semantic entailment review of insight records\n"
    "\n"
    "You are the insight semantic reviewer. For EACH record in the review\n"
    "bundle below, answer exactly one question: do the cited source\n"
    "passages support this statement at this strength, with these\n"
    "conditions, without resolving ambiguities the source leaves open?\n"
    "\n"
    "The bundle is your ONLY evidence. Each record carries its statement\n"
    "(or structured normative fields), its citations with the exact\n"
    "quoted source bytes, and a bounded context window around each quote.\n"
    "Do not ask for, look for, or assume access to the full document —\n"
    "judge entailment strictly from the quotes and their context. If the\n"
    "context is insufficient to decide, that is what `needs_human` is for.\n"
    "\n"
    "USE YOUR WRITE TOOL EXACTLY ONCE: write one JSON object to\n"
    "`{record_path}`. After that file is written, return. Do not edit,\n"
    "repair, re-render, or improve any candidate or statement — you judge\n"
    "records, you never fix them. A rewritten statement, however\n"
    "reasonable, is a contract violation.\n"
    "\n"
    "**Verdict discipline per record:**\n"
    "\n"
    "- `accepted` — every claim in the statement is supported by the\n"
    "  cited quotes at the stated strength and scope. Preserved direction,\n"
    "  conditions, exceptions, and uncertainty.\n"
    "- `rejected` — the statement goes beyond, conflates, invents,\n"
    "  strengthens, or resolves what the quotes actually say. Name the\n"
    "  failure class (below) and cite the deciding span content in one\n"
    "  sentence.\n"
    "- `needs_human` (separate list) — you cannot decide from the bundle.\n"
    "  Say what blocked the verdict. Never guess a verdict to avoid\n"
    "  escalating; never escalate to avoid deciding a clear case.\n"
    "\n"
    "Every record_id in the bundle appears EXACTLY ONCE across `verdicts`\n"
    "and `needs_human`. A missing or extra id invalidates the whole record.\n"
    "\n"
    "**Failure classes (closed set — pick the dominant one):**\n"
    "\n"
    "- `unsupported_strengthening`: the statement promises more than the\n"
    "  quote (e.g. \"idempotent\" turned into \"never produces a different\n"
    "  result\").\n"
    "- `safety_scope_conflation`: two distinct source concepts merged into\n"
    "  one claim (e.g. safety conflated with cacheability).\n"
    "- `invented_algorithm_or_rule`: a mechanism, algorithm, or rule the\n"
    "  quotes never state.\n"
    "- `imperative_from_suggestion`: advisory or optional source language\n"
    "  rendered as a requirement or command.\n"
    "- `unresolved_ambiguity_resolution`: the source leaves a question\n"
    "  open and the statement resolves it anyway.\n"
    "- `coverage_omission`: the statement misrepresents what the source\n"
    "  covers or omits.\n"
    "\n"
    "**Output object (exact shape, no wrapper key):**\n"
    "\n"
    "```json\n"
    "{{\n"
    "  \"schema_version\": \"1.0.0\",\n"
    "  \"reviewer\": \"r2c-insight-semantic-reviewer\",\n"
    "  \"source_sha256\": \"{source_sha256}\",\n"
    "  \"candidate_sha256\": \"{candidate_sha256}\",\n"
    "  \"verdicts\": [\n"
    "    {{\"record_id\": \"<id>\", \"verdict\": \"accepted\" | \"rejected\",\n"
    "     \"failure_class\": \"<class>\" (rejected only, omit when accepted),\n"
    "     \"reason\": \"<one sentence citing the deciding span>\"}}\n"
    "  ],\n"
    "  \"needs_human\": [\n"
    "    {{\"record_id\": \"<id>\", \"reason\": \"<what blocked the verdict>\"}}\n"
    "  ],\n"
    "  \"overall_verdict\": \"accepted\" | \"rejected\" | \"needs_human\"\n"
    "}}\n"
    "```\n"
    "\n"
    "Copy `source_sha256` and `candidate_sha256` into your record EXACTLY\n"
    "as given above — they bind your review to the exact bytes you judged,\n"
    "and a mismatch voids the record. `overall_verdict` is derived, not\n"
    "chosen: any rejection makes it `rejected`; otherwise any escalation\n"
    "makes it `needs_human`; otherwise `accepted`."
)


def build_insight_review_prompt(
    *,
    bundle_json: str,
    record_path: str,
    source_sha256: str,
    candidate_sha256: str,
) -> str:
    """Assemble the insight semantic reviewer's dispatch prompt: task +
    think anchor + the sliced review bundle + writeable paths.

    Deliberately NOT built on build_dispatch_prompt: the SCOPE CONTRACT's
    single-method package/notebook framing has nothing to review here and
    would only invite drift. The bundle (built by
    scripts/insight_semantic_review.py) is the entire evidence surface —
    the reviewer never receives run paths or the full paper."""
    return "\n\n".join([
        INSIGHT_REVIEW_TASK_TEMPLATE.format(
            record_path=record_path,
            source_sha256=source_sha256,
            candidate_sha256=candidate_sha256,
        ),
        THINK_CLASS_ANCHOR.format(expected_output_path=record_path),
        "**Review bundle (your only evidence):**\n\n"
        "```json\n" + bundle_json + "\n```",
        format_writeable_paths_block([record_path]),
    ])


def build_halt_artifact(
    *,
    stage: str,
    reason: str,
    retry_count: int = 0,
    context: dict | str | None = None,
    target_script: str | None = None,
    findings: list[dict] | None = None,
    user_message: str | None = None,
    halt_class: str | None = None,
    evidence: dict | None = None,
) -> dict:
    """Build a halt artifact dict for `<PIPELINE_DIR>/<stage>.halt`.

    `reason` stays technical (engineering / debugging / git-blame). When a halt
    is *researcher-actionable* (the paper isn't reproducible, doesn't fit a
    paradigm, the input file is bad, etc.), pass `user_message`: 1-2 plain
    sentences telling the researcher what happened and what to do.

    `halt_class` (halt-reason catalog, queue item 12): the call site's
    declaration of which closed catalog class this halt belongs to — see
    `scripts/halt_catalog.py`. When set (and no `user_message` overrides it),
    both researcher surfaces render the catalog's plain-language four-part
    story instead of the stock "internal error" notice. `evidence` carries
    the structured fields the catalog's mechanism resolver is allowed to
    speak from (completed, elapsed_s, timeout_s, writes_observed, attempts,
    error_kind, step_gloss, failing_location, pending_finding) — values the
    driver already holds, never a model's inference. Both fields are
    optional; old artifacts without them render exactly as before."""
    artifact: dict = {
        "status": "halted",
        "stage": stage,
        "reason": reason,
        "retry_count": retry_count,
    }
    if user_message is not None:
        artifact["user_message"] = user_message
    if halt_class is not None:
        artifact["halt_class"] = halt_class
    if evidence is not None:
        artifact["evidence"] = evidence
    if context is not None:
        artifact["context"] = context
    if target_script is not None:
        artifact["target_script"] = target_script
    if findings is not None:
        artifact["findings"] = findings
    return artifact




if __name__ == "__main__":
    raise SystemExit(
        "self-tests moved to tests/test_dispatch_templates_selftest.py — run: "
        "python3 -m pytest tests/test_dispatch_templates_selftest.py"
    )
