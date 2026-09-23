"""Id fidelity across dispatch boundaries (queue item 22).

The genus, three live instances in one overnight (2026-07-08): agents
regenerate ids from language instead of copying them, then a strict
consumer rejects the near-miss and a blind retry repeats it. The fixes
here are deterministic and per-surface: unknown-id errors name the
nearest REAL id so retries converge in one pass, same-type elements may
not mix confusable id prefixes (the shape that invites the mutation),
and the resolution re-ask pins synthesized-looking finding ids as
contract fields.

Fixture maps are the harvested overnight evidence verbatim
(tests/fixtures/evidence/overnight-0707/item-22-id-fidelity/).
"""

from __future__ import annotations

import json
from pathlib import Path

from dispatch_templates import STAGE_REVIEW_RESOLUTION_REASK_TEMPLATE
from schemas.paper_map import PaperMap, nearest_id
from validate_method_explanations import check_explanations
from validate_paper_map import check_id_prefix_consistency

FIXTURES = (Path(__file__).parent / "fixtures" / "evidence"
            / "overnight-0707" / "item-22-id-fidelity")


def _fixture_map(name: str) -> dict:
    return json.loads((FIXTURES / f"{name}.paper_map.json")
                      .read_text(encoding="utf-8"))


def _ids(paper_map: dict) -> list[str]:
    return [e["id"] for e in paper_map["elements"]]


# --- nearest_id: the retry-convergence hint ---------------------------------


def test_nearest_id_resolves_the_adam_character_near_miss():
    # ADAM's explainer keyed `eq-temporal-averaging`; the map says
    # `eq-temporal-average`. Character similarity carries this one.
    ids = _ids(_fixture_map("ADAM"))
    assert nearest_id("eq-temporal-averaging", ids) == "eq-temporal-average"


def test_nearest_id_resolves_the_idb_token_reorder():
    # iDb-RRT's decomposer cited `eq-dynamics-continuous` where its own
    # map says `equation-continuous-dynamics` — reordered tokens plus an
    # abbreviated prefix. Pure character similarity ranks the WRONG
    # sibling (`eq-dynamics-discrete`) first; token overlap must win.
    ids = _ids(_fixture_map("iDb-RRT"))
    assert nearest_id("eq-dynamics-continuous", ids) == \
        "equation-continuous-dynamics"


def test_nearest_id_stays_silent_on_a_genuinely_dangling_id():
    # ADAM's `eq-ema-mt` references no real element (closest sibling
    # scores below the floor). A wrong suggestion is worse than none.
    ids = _ids(_fixture_map("ADAM"))
    assert nearest_id("eq-ema-mt", ids) is None


def test_nearest_id_handles_empty_inputs():
    assert nearest_id("eq-anything", []) is None
    assert nearest_id("", ["eq-a"]) is None
    # The bad id itself never counts as its own suggestion.
    assert nearest_id("eq-a", ["eq-a"]) is None


# --- schema cross-reference errors carry the hint ---------------------------


def _minimal_element(el_id: str, **overrides) -> dict:
    base = {"id": el_id, "type": "equation", "name": el_id,
            "section": "Section 1", "description": "d",
            "source_text": "", "code_role": "implement"}
    base.update(overrides)
    return base


def test_cross_reference_error_names_the_nearest_id():
    elements = [
        _minimal_element("eq-temporal-average"),
        _minimal_element("eq-regret-bound",
                         related_equations=["eq-temporal-averaging"]),
    ]
    try:
        PaperMap.model_validate({"title": "T", "elements": elements})
        raise AssertionError("expected a cross-reference validation error")
    except ValueError as exc:
        msg = str(exc)
        assert "eq-temporal-averaging" in msg
        assert "nearest existing id" in msg
        assert "`eq-temporal-average`" in msg


def test_dangling_cross_reference_gets_no_misleading_hint():
    elements = [
        _minimal_element("eq-core"),
        _minimal_element("eq-other", dependencies=["concept-unrelated-thing"]),
    ]
    try:
        PaperMap.model_validate({"title": "T", "elements": elements})
        raise AssertionError("expected a cross-reference validation error")
    except ValueError as exc:
        msg = str(exc)
        assert "concept-unrelated-thing" in msg
        assert "nearest existing id" not in msg


# --- prefix-consistency prophylactic ----------------------------------------


def test_idb_map_fails_the_prefix_consistency_check():
    errors = check_id_prefix_consistency(_fixture_map("iDb-RRT"))
    assert len(errors) == 1
    assert "`eq-` and `equation-`" in errors[0]
    assert "equation-continuous-dynamics" in errors[0]
    # The minority renames toward the majority (six eq-* vs one equation-*).
    assert "to the `eq-` prefix" in errors[0]


def test_single_prefix_map_passes():
    assert check_id_prefix_consistency(_fixture_map("ADAM")) == []


def test_unrelated_prefixes_coexisting_are_not_flagged():
    # Only substring-related prefix pairs invite the mutation; genuinely
    # different prefixes within a type stay legal.
    pm = {"elements": [_minimal_element("eq-main"),
                       _minimal_element("loss-l1")]}
    assert check_id_prefix_consistency(pm) == []


def test_prefix_mixing_across_types_is_not_flagged():
    # `prop-*` properties next to `property`-prefixed CONCEPTS would be
    # odd naming but is not the same-type confusion shape.
    pm = {"elements": [_minimal_element("eq-main"),
                       _minimal_element("equation-notes", type="concept")]}
    assert check_id_prefix_consistency(pm) == []


# --- explainer fabricated-element findings carry the hint -------------------


def test_fabricated_element_finding_names_the_nearest_id():
    paper_map = _fixture_map("ADAM")
    entry = {"what": "x" * 30, "why_novel": "y" * 30, "intuition": "z" * 30}
    sidecar = {"schema_version": "1.0.0",
               "explanations": {"eq-temporal-averaging": dict(entry)}}
    findings = check_explanations(sidecar, paper_map, None)
    fabricated = [f for f in findings if f["check"] == "fabricated_element"]
    assert len(fabricated) == 1
    assert "'eq-temporal-average'" in fabricated[0]["message"]
    assert "EXACTLY" in fabricated[0]["message"]


def test_fabricated_element_without_a_close_sibling_has_no_hint():
    paper_map = _fixture_map("ADAM")
    entry = {"what": "x" * 30, "why_novel": "y" * 30, "intuition": "z" * 30}
    sidecar = {"schema_version": "1.0.0",
               "explanations": {"totally-invented-thing": dict(entry)}}
    findings = check_explanations(sidecar, paper_map, None)
    fabricated = [f for f in findings if f["check"] == "fabricated_element"]
    assert len(fabricated) == 1
    assert "Nearest existing id" not in fabricated[0]["message"]


# --- the resolution re-ask pins synthesized-looking ids ---------------------


def test_reask_template_pins_synthesized_finding_ids():
    # 12b layer 3's live failure: the reviewer answered a DEF-r_e dispatch
    # under its own F001 numbering and the id-keyed merge dropped a
    # correct resolution. The template must name the synthesized shape
    # and forbid renumbering.
    assert "look synthesized" in STAGE_REVIEW_RESOLUTION_REASK_TEMPLATE
    assert "DEF-" in STAGE_REVIEW_RESOLUTION_REASK_TEMPLATE
    assert "never renumber" in STAGE_REVIEW_RESOLUTION_REASK_TEMPLATE
