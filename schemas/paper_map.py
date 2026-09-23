"""
Pydantic model for the Paper Map produced by Stage 1 Step 2 (decomposer).

The paper map is the structured decomposition of a paper into its technical
elements — algorithms, equations, concepts, experiments, properties,
hyperparameters — each with a stable ID. Step 3 (method analyzer) and later
stages reference these IDs (`core_method.key_elements`, `paper_sections`,
etc.) to ground spec fields in the paper's structure.

This schema mirrors the discipline of `method_spec.py`:
- `extra="forbid"` everywhere — typos and schema drift are validation errors.
- Cross-reference invariants are enforced (dependencies, related_equations).
- `has_pseudocode` and `pseudocode` must be consistent.

ID convention (recommended, not enforced by the schema): kebab-case with
type-prefix — `alg-*`, `eq-*`, `concept-*`, `exp-*`, `prop-*`, `hyp-*`.
The schema only enforces uniqueness and cross-reference resolution.
"""

from __future__ import annotations

import difflib
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, model_validator


SCHEMA_VERSION = "1.0.0"


def nearest_id(bad_id: str, known_ids: list[str] | set[str]) -> str | None:
    """Best existing id for an unknown one, or None when nothing is close.

    The id-mutation genus (queue item 22, overnight 2026-07-08): agents
    regenerate ids from language instead of copying them, then a strict
    consumer rejects the near-miss and a blind retry repeats it. Naming
    the nearest REAL id in the error is what lets the retry converge in
    one pass.

    Score is the max of character similarity (difflib) and hyphen-token
    overlap. Both matter on the two live shapes: ADAM's
    `eq-temporal-averaging` vs `eq-temporal-average` is a character-level
    near-miss (difflib 0.90), while iDb-RRT's `eq-dynamics-continuous` vs
    `equation-continuous-dynamics` reorders tokens and abbreviates the
    prefix, which difflib alone RANKS WRONG (it prefers
    `eq-dynamics-discrete`) but token overlap ranks right. Below the 0.7
    floor returns None — the same run's genuinely dangling `eq-ema-mt`
    scores 0.67 against its closest sibling, and a wrong suggestion is
    worse than none."""
    candidates = sorted(set(known_ids) - {bad_id})
    if not bad_id or not candidates:
        return None

    def _tokens(s: str) -> list[str]:
        return [t for t in s.lower().split("-") if t]

    def _tok_eq(a: str, b: str) -> bool:
        # Equality, or a 2+ char prefix relation (`eq` vs `equation`).
        if a == b:
            return True
        if len(a) >= 2 and b.startswith(a):
            return True
        return len(b) >= 2 and a.startswith(b)

    bad_toks = _tokens(bad_id)
    best_id, best_score = None, 0.0
    for cand in candidates:
        char_score = difflib.SequenceMatcher(None, bad_id, cand).ratio()
        cand_toks = _tokens(cand)
        if bad_toks and cand_toks:
            hits = (
                sum(1 for t in bad_toks if any(_tok_eq(t, c) for c in cand_toks))
                + sum(1 for c in cand_toks if any(_tok_eq(c, t) for t in bad_toks))
            )
            tok_score = hits / (len(bad_toks) + len(cand_toks))
        else:
            tok_score = 0.0
        score = max(char_score, tok_score)
        if score > best_score:
            best_id, best_score = cand, score
    return best_id if best_score >= 0.7 else None


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class ElementType(str, Enum):
    algorithm = "algorithm"
    equation = "equation"
    concept = "concept"
    experiment = "experiment"
    property = "property"
    hyperparameter = "hyperparameter"


class CodeRole(str, Enum):
    """How the element gets used in generated code.

    See `.opencode/skills/code-role-classification` for the full guidance.
    Briefly:
        implement     — produces runnable code
        demonstrate   — visualization or simulation that conveys the idea
        theoretical   — proof scaffolding; not executed
    """

    implement = "implement"
    demonstrate = "demonstrate"
    theoretical = "theoretical"


# ---------------------------------------------------------------------------
# Sub-models
# ---------------------------------------------------------------------------


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Element(_Strict):
    id: str = Field(
        min_length=1,
        description=(
            "Unique within this paper map. Convention: kebab-case prefix per type "
            "(`alg-*`, `eq-*`, `concept-*`, `exp-*`, `prop-*`, `hyp-*`)."
        ),
    )
    type: ElementType
    name: str
    section: str = Field(
        description='Paper section reference, e.g., "Section 3" or "Algorithm 1".'
    )
    description: str
    source_text: str = Field(
        description=(
            "Exact verbatim passage from the paper. For equation elements this is "
            "the equation EXACTLY as the paper writes it, LaTeX and $/$$ delimiters "
            "included (validated as a substring of paper.md); METHOD.md renders it "
            "as display math. Prose passages stay plain text. "
            "May be empty for elements that summarize claims without a single quotable line."
        )
    )
    pseudocode: str = Field(
        default="",
        description=(
            "Plain-ASCII representation of the element's computable content if available; "
            'else "". For algorithms this is the step-by-step pseudocode; for equations '
            "this is the ASCII-form expression (e.g., `g_x = (p_i - I(y_hat=i)) * z(x; V)`); "
            "for other element types it is typically empty. LaTeX commands must be "
            "replaced with ASCII (e.g., `\\sum` → `sum`)."
        ),
    )
    dependencies: list[str] = Field(
        default_factory=list,
        description="IDs of elements this depends on. Must resolve within this paper map.",
    )
    related_equations: list[str] = Field(
        default_factory=list,
        description=(
            "IDs of equation-type elements related to this. Must resolve within "
            "this paper map AND must reference elements whose type is `equation`."
        ),
    )
    code_role: CodeRole


class PaperMap(_Strict):
    schema_version: str = Field(
        default=SCHEMA_VERSION,
        description=(
            "Schema version this paper map was written against. Validators reject "
            "maps whose version they do not understand."
        ),
    )
    title: str
    elements: list[Element]

    @model_validator(mode="after")
    def _ids_unique_and_refs_resolve(self) -> "PaperMap":
        ids: list[str] = [e.id for e in self.elements]
        seen: set[str] = set()
        dups: list[str] = []
        for i in ids:
            if i in seen:
                dups.append(i)
            seen.add(i)
        if dups:
            raise ValueError(f"duplicate element IDs: {sorted(set(dups))}")

        all_ids = set(ids)
        equation_ids = {e.id for e in self.elements if e.type == ElementType.equation}

        def _hint(bad: str) -> str:
            near = nearest_id(bad, all_ids)
            return (
                f" — nearest existing id: `{near}`; if that is the element "
                f"you meant, use it EXACTLY" if near else ""
            )

        errors: list[str] = []
        for e in self.elements:
            for dep in e.dependencies:
                if dep not in all_ids:
                    errors.append(
                        f"element `{e.id}`: dependency `{dep}` does not exist "
                        f"in this paper map{_hint(dep)}"
                    )
            for ref in e.related_equations:
                if ref not in all_ids:
                    errors.append(
                        f"element `{e.id}`: related_equations entry `{ref}` "
                        f"does not exist{_hint(ref)}"
                    )
                elif ref not in equation_ids:
                    errors.append(
                        f"element `{e.id}`: related_equations entry `{ref}` references a "
                        f"non-equation element (related_equations must point at type=equation only)"
                    )
        if errors:
            raise ValueError(
                "paper map cross-reference errors:\n  - " + "\n  - ".join(errors)
            )
        return self
