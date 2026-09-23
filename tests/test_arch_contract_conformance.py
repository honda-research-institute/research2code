"""Paradigm-manifest ↔ contract-schema conformance (queue item 21).

Overnight 2026-07-08 produced three shapes of the same genus — a paradigm's
`arch_contract_requirements` demanding something `schemas/arch_contract.py`
cannot express, so a schema-correct contract fails and the run degrades at
stage 2.d as pipeline_bug:

  - ADAM run 4: stochastic_optimization requires a top-level
    `optimizer_state` block; ArchContract (extra=forbid) had no such field.
    Omit → "required but missing"; comply → "extra inputs not permitted".
  - ms3d run 7: the output_type literal could not express an honest
    list-of-structured-objects detection output (`list[BoundingBox3D]`).
  - Rethinking-Grouping run 9: vision_transformer required subblock
    `forward.output`, a key the schema never had (its vocabulary is
    output_type / output_shape / output_keys).

This test closes the whole drift class: every required block path and
required subblock key declared by ANY paradigm must be constructible under
the pydantic contract models. A future paradigm declaring an inexpressible
requirement goes red here instead of trapping a coder at 2.d.
"""

from __future__ import annotations

import types
import typing

from pydantic import BaseModel

from schemas.arch_contract import ArchContract


def _terminal_types(tp, parts: list[str]) -> list:
    """All types a dotted path can resolve to under the pydantic models.

    - BaseModel: the next part must be a declared field (extra=forbid
      everywhere in this schema family, so undeclared fields are rejected
      at runtime and must be rejected here too).
    - dict[str, X]: any key is constructible; the part is consumed and
      resolution continues into the value type.
    - Optional/Union: each non-None arm is tried.
    - Anything else (str, Literal, ...): terminal; unresolvable if parts
      remain.
    """
    if not parts:
        return [tp]
    origin = typing.get_origin(tp)
    args = typing.get_args(tp)
    if origin in (typing.Union, types.UnionType):
        out = []
        for a in args:
            if a is type(None):
                continue
            out.extend(_terminal_types(a, parts))
        return out
    if origin is dict:
        value_tp = args[1] if len(args) > 1 else object
        return _terminal_types(value_tp, parts[1:])
    if isinstance(tp, type) and issubclass(tp, BaseModel):
        field = tp.model_fields.get(parts[0])
        if field is None:
            return []
        return _terminal_types(field.annotation, parts[1:])
    return []


def _requirements_by_paradigm() -> dict[str, dict]:
    from scripts.build_plan import STATIC_PLAN_BY_PARADIGM

    # active_learning is covered by the loop since B-02 moved it onto the
    # static mechanism (its former spec-derived requirements never read the
    # spec, so nothing is lost by the static read).
    return {
        pid: plan.get("arch_contract_requirements") or {}
        for pid, plan in STATIC_PLAN_BY_PARADIGM.items()
    }


def _conformance_failures() -> list[str]:
    failures: list[str] = []
    for pid, reqs in _requirements_by_paradigm().items():
        for block_path, block_spec in (reqs.get("required_blocks") or {}).items():
            block_types = _terminal_types(ArchContract, block_path.split("."))
            if not block_types:
                failures.append(
                    f"{pid}: required block `{block_path}` is not "
                    f"constructible under ArchContract"
                )
                continue
            for sub in (block_spec or {}).get("required_subblocks", []) or []:
                if not any(
                    _terminal_types(bt, sub.split(".")) for bt in block_types
                ):
                    failures.append(
                        f"{pid}: required subblock `{sub}` of block "
                        f"`{block_path}` is not constructible under "
                        f"ArchContract"
                    )
    return failures


def test_every_paradigm_requirement_is_expressible_under_the_contract_schema():
    failures = _conformance_failures()
    assert failures == [], (
        "paradigm manifests require contract content the schema cannot "
        "express — a schema-correct contract will fail these paradigms at "
        "stage 2.d:\n" + "\n".join(f"  - {f}" for f in failures)
    )
