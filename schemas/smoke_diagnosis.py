"""Pydantic model for `<RUN_DIR>/.pipeline/smoke_diagnosis.json`.

The smoke-gate diagnostician (Think-class agent) produces this file before
each smoke-fix-mode dispatch in Stage 3.c. It examines the failing cell +
traceback + relevant files and decides:
  - which agent owns the bug (target_agent)
  - which file the fix should land in (target_file)
  - what the bug actually is (root_cause)
  - what change resolves it (proposed_fix)

The driver consumes this and:
  1. Overrides the mechanical traceback-based producer routing if the
     diagnostician disagrees with it (the diagnostician traces data flow;
     the routing just looks at the deepest method/*.py frame).
  2. Injects root_cause + proposed_fix into the SMOKE001 finding so the
     producer's fix-mode dispatch starts with a real diagnosis, not a
     raw traceback the producer has to re-analyze.

The schema is intentionally small and bounded — the agent has three
target_agent options and is forced to commit to a concrete fix proposal
rather than emitting open-ended analysis.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


SCHEMA_VERSION = "1.0.0"


# Smoke-routable target agents — same three as smoke_cell_to_producer maps to.
# The diagnostician cannot route a smoke fix to analyzer/scaffolder/etc.;
# the smoke gate is downstream of those stages and surfacing a bug there
# requires user intervention (the script halts surface that case).
TargetAgent = Literal[
    "r2c-method-coder",
    "r2c-architecture-coder",
    "r2c-notebook-generator",
]


class SmokeDiagnosis(BaseModel):
    """Structured diagnosis of a smoke-gate failure."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1.0.0"] = Field(
        default=SCHEMA_VERSION,
        description="Schema version. Always '1.0.0' for now.",
    )
    target_agent: TargetAgent = Field(
        description=(
            "Which producer agent should receive the fix-mode dispatch. The "
            "diagnostician picks based on which agent owns the file containing "
            "the bug — not where the error fires."
        )
    )
    target_file: str = Field(
        description=(
            "Relative path within run_dir of the file the fix should land in "
            "(e.g., 'method/method.py'). Must match one of target_agent's "
            "WRITEABLE_PATHS entries."
        )
    )
    root_cause: str = Field(
        description=(
            "1-3 sentences naming the actual bug. NOT the symptom (where the "
            "error fires); the cause. Example: 'compute_3d_box_iou indexes "
            "column 8 of a tensor that should have 9 columns, but the prior "
            "slice at line 162 reduces it to 8 columns.'"
        )
    )
    proposed_fix: str = Field(
        description=(
            "1-3 sentences naming the specific change target_agent should "
            "make. Concrete enough to act on, not abstract. Example: 'Remove "
            "the .narrow(1, 0, 8) call on line 162 so boxes pass through "
            "with their full 9-column shape.'"
        )
    )
    reasoning: str = Field(
        description=(
            "Explanation of how the diagnosis was reached. Walks the (i)–(iv) "
            "checklist from SMOKE_FIX_GUIDANCE: value origin, surface-vs-bug "
            "distinction, architecture-as-contract check, scope verification."
        )
    )
    paper_fidelity_check: str = Field(
        description=(
            "1-3 sentences arguing why the proposed_fix is consistent with the "
            "paper's algorithm. If the fix is a smoke-substitute that deviates "
            "from paper-faithful behavior, say so explicitly and explain why "
            "the deviation is acceptable for smoke-scale demonstration. NEVER "
            "omit this — a fix you can't defend on paper-fidelity grounds is "
            "usually a hack around the wrong layer of the bug. Example of a "
            "valid paper-fidelity argument: 'Paper specifies Object-DGCNN as "
            "the teacher (graph attention over points); this fix replaces the "
            "broken Linear backbone with a per-point Linear, which is a smoke-"
            "scale stand-in but preserves the per-point structure that DGCNN "
            "would have. Acceptable for smoke; paper-scale would need the "
            "actual DGCNN.' Example of an INVALID fix surfaced this way: "
            "'Flatten the 3D point cloud to 2D before the backbone' — destroys "
            "point structure entirely; the fix is in the wrong place."
        )
    )
    bug_shape: str = Field(
        description=(
            "The class of bug this halt fits, drawn from the matched "
            "taxonomy node's `common_smoke_bugs` block (or "
            "`uncatalogued` if no entry matches). Examples for "
            "supervised_ml: `labels_out_of_range`, `model_input_shape_mismatch`, "
            "`stochastic_call_without_seed`. The diagnostician must pick one "
            "BEFORE picking target_agent + target_file — the catalog encodes "
            "where the typical root cause for each bug shape lives, so the "
            "routing is informed by accumulated knowledge of which file "
            "owns each bug class.\n\n"
            "If `uncatalogued`, the diagnostician relies on Step 3's "
            "value-origin trace alone; the user reviewing the halt may "
            "consider adding this case to the taxonomy bug catalog after "
            "the fact."
        )
    )
    value_origin_trace: list[str] = Field(
        min_length=3,
        description=(
            "Ordered list naming every step in the failing value's derivation "
            "chain, starting from the SITE OF FAILURE and walking UPSTREAM to "
            "the value's first origin. Each entry is a short string of the "
            "form `<file:line> in <scope>: <expression or note>`. Required "
            "minimum 3 entries — forces the diagnostician to actually walk "
            "the chain instead of patching locally. The most common "
            "smoke-bug class is 'error surfaces at line X, but value was "
            "derived wrong N steps upstream'; this field makes that walk "
            "structurally mandatory.\n\n"
            "Use the driver-supplied data_flow.json to look up each "
            "symbol's `assignments` field for the previous step in the chain.\n\n"
            "Example (the gbald labels-out-of-range bug):\n"
            "  [\n"
            "    'method/training.py:104: y_train.max() vs n_classes — error site',\n"
            "    'method/training.py: y_train ← x_train, y_train parameters of train_from_scratch (called from notebook)',\n"
            "    'notebook §3.4 cell 17: y_labeled passed to train_from_scratch ← construct_core_set return',\n"
            "    'method/method.py:209 return: y_core ← y_pool[core_set_indices].clone()',\n"
            "    'notebook_draft.py:111: n_classes = int(y_pool.max().item()) + 1 ← derivation site (compare against y_pool values)'\n"
            "  ]\n\n"
            "If this list ends at the validation site, you didn't trace far "
            "enough — the bug is almost never at the validation site itself."
        ),
    )
