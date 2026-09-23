"""Role-typed temporal evaluation protocol contract tests (R2C-082).

The pdfgnn fixture is the real-paper regression: its paper states context
length 10, validation span 13 weeks, and test span 26 weeks while leaving the
one-call forecast horizon K symbolic.  The explicit-K and rolling-origin cases
below are synthetic mechanical controls only.  They are not second-paper
evidence and must not be used to broaden the protocol abstraction.
"""

from __future__ import annotations

import copy
import json
import subprocess
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

from schemas.method_spec import EvaluationProtocol, MethodSpec
from schemas.params import Params
from scripts.derive_params import derive
from scripts.evaluation_protocol_rendering import render_evaluation_protocol_block
from scripts.validate_method_spec import (
    cross_check_evaluation_protocol,
    cross_check_paper_map,
)
from scripts.validate_params_output import _evaluation_protocol_errors
from tests.test_method_spec_schema import (
    _add_methodology_contract,
    _methodology_element,
    _minimal_valid_spec,
)


ROOT = Path(__file__).resolve().parents[1]

PDFGNN_CONTEXT_QUOTE = (
    "|            |                    | Context length    | 10                 "
    "| 10                 |"
)
PDFGNN_HORIZON_QUOTE = (
    r"Our task is to predict demand for N articles during the future K time "
    r"steps  $(y_i^{T+1}, y_i^{T+2}, \dots, y_i^{T+K})_{i=1}^N$  given the "
    r"historical demand  $(y_i^1, y_i^2, \dots, y_i^T)_{i=1}^N$ ."
)
PDFGNN_SPLIT_QUOTE = (
    "We split each dataset into three subsets along the time axis: training, "
    "validation, and test sets. The test set is used for model evaluation and "
    "covers the last 26 weeks. The validation set is used to tune model "
    "hyperparameters; the length of the validation set is fixed to 13 weeks "
    "preceding the test set. The training set covers all data available prior "
    "to the validation set."
)
PDFGNN_TEST_QUOTE = (
    "The test set is used for model evaluation and covers the last 26 weeks."
)
PDFGNN_VALIDATION_QUOTE = (
    "The validation set is used to tune model hyperparameters; the length of "
    "the validation set is fixed to 13 weeks preceding the test set."
)
PDFGNN_AXIS_QUOTE = (
    "In each dataset, we aggregate weekly demand across all stores to produce "
    "the target time series."
)
PDFGNN_LAGS_QUOTE = (
    r"we limit node features to P demand lags: $\mathcal{N}_i^t = "
    r"(y_i^{t-P+1}, y_i^{t-P+2}, \dots, y_i^t)$ . The number of lags serves "
    r"as a hyperparameter."
)
PDFGNN_PAPER_TEXT = "\n".join(
    [
        PDFGNN_HORIZON_QUOTE,
        PDFGNN_CONTEXT_QUOTE,
        PDFGNN_LAGS_QUOTE,
        PDFGNN_SPLIT_QUOTE,
        PDFGNN_AXIS_QUOTE,
    ]
)

PROTOCOL_IDS = {
    "context": "table-5-context-length",
    "horizon": "task-forecast-horizon",
    "split": "evaluation-split",
    "axis": "weekly-demand-axis",
}

PAPER_NAMES = {
    "context_length": ["Context length"],
    "forecast_call_horizon": ["future K time steps"],
    "validation_span": ["validation set"],
    "test_span": ["test set"],
}

PAPER_SYMBOLS = {
    "context_length": [],
    "forecast_call_horizon": ["K"],
    "validation_span": [],
    "test_span": [],
}


def _quantity(
    role: str,
    *,
    parameter_name: str | None,
    value: int | float | None,
    status: str,
    quote: str,
    element_id: str,
) -> dict:
    return {
        "role": role,
        "parameter_name": parameter_name,
        "paper_names": PAPER_NAMES[role],
        "paper_symbols": PAPER_SYMBOLS[role],
        "value": value,
        "unit": "week",
        "granularity": 1,
        "axis_evidence_quote": PDFGNN_AXIS_QUOTE,
        "axis_paper_section": "Datasets",
        "axis_paper_element_ids": [PROTOCOL_IDS["axis"]],
        "paper_value_status": status,
        "evidence_quote": quote,
        "paper_section": "Evaluation protocol",
        "paper_element_ids": [element_id],
    }


def _pdfgnn_spec() -> dict:
    spec = _minimal_valid_spec(paradigm_id="time_series_forecasting")
    spec["schema_version"] = "1.11.0"
    spec["comparison"]["evaluation_protocol"] = {
        "scheme": {
            "kind": "single_holdout",
            "paper_value_status": "paper_stated",
            "description": (
                "One chronological training/validation/test split with the "
                "validation interval immediately before the test interval."
            ),
            "evidence_quote": PDFGNN_SPLIT_QUOTE,
            "paper_section": "Evaluation protocol",
            "paper_element_ids": [PROTOCOL_IDS["split"]],
        },
        "quantities": [
            _quantity(
                "context_length",
                parameter_name="context_length",
                value=10,
                status="paper_stated",
                quote=PDFGNN_CONTEXT_QUOTE,
                element_id=PROTOCOL_IDS["context"],
            ),
            _quantity(
                "forecast_call_horizon",
                parameter_name="forecast_horizon",
                value=None,
                status="paper_unspecified",
                quote=PDFGNN_HORIZON_QUOTE,
                element_id=PROTOCOL_IDS["horizon"],
            ),
            _quantity(
                "validation_span",
                parameter_name=None,
                value=13,
                status="paper_stated",
                quote=PDFGNN_VALIDATION_QUOTE,
                element_id=PROTOCOL_IDS["split"],
            ),
            _quantity(
                "test_span",
                parameter_name=None,
                value=26,
                status="paper_stated",
                quote=PDFGNN_TEST_QUOTE,
                element_id=PROTOCOL_IDS["split"],
            ),
        ],
    }
    return spec


def _protocol_quantity(spec: dict, role: str) -> dict:
    return next(
        quantity
        for quantity in spec["comparison"]["evaluation_protocol"]["quantities"]
        if quantity["role"] == role
    )


def _write_paper_map(tmp_path: Path) -> Path:
    pipeline_dir = tmp_path / ".pipeline"
    pipeline_dir.mkdir(parents=True)
    evidence = {
        "alg-test": ("Synthetic algorithm source.", "Method"),
        PROTOCOL_IDS["context"]: (
            PDFGNN_CONTEXT_QUOTE,
            "Evaluation protocol",
        ),
        PROTOCOL_IDS["horizon"]: (
            PDFGNN_HORIZON_QUOTE,
            "Evaluation protocol",
        ),
        PROTOCOL_IDS["split"]: (
            PDFGNN_SPLIT_QUOTE,
            "Evaluation protocol",
        ),
        PROTOCOL_IDS["axis"]: (PDFGNN_AXIS_QUOTE, "Datasets"),
        "hyp-lags": (PDFGNN_LAGS_QUOTE, "Evaluation protocol"),
    }
    (pipeline_dir / "paper_map.json").write_text(
        json.dumps({
            "elements": [
                {"id": item, "source_text": quote, "section": section}
                for item, (quote, section) in evidence.items()
            ]
        }),
        encoding="utf-8",
    )
    return pipeline_dir / "method_spec.json"


def test_pdfgnn_keeps_symbolic_k_separate_from_t_plus_one_and_test_span() -> None:
    """The real pdfgnn regression must keep K system-owned at runtime 4."""
    spec = _pdfgnn_spec()

    validated = MethodSpec.model_validate(spec)
    assert cross_check_evaluation_protocol(validated, ROOT) == []

    protocol = validated.comparison.evaluation_protocol
    assert protocol is not None
    by_role = {quantity.role.value: quantity for quantity in protocol.quantities}
    assert by_role["context_length"].value == 10
    assert by_role["forecast_call_horizon"].value is None
    assert by_role["forecast_call_horizon"].paper_value_status.value == (
        "paper_unspecified"
    )
    assert by_role["validation_span"].value == 13
    assert by_role["test_span"].value == 26

    output = derive(spec, paper_text=PDFGNN_PAPER_TEXT)
    params = output["params"]
    horizon = params["forecast_horizon"]
    context = params["context_length"]

    assert context["value"] == 10
    assert context["source"] == "paper"
    assert horizon["value"] == 4
    assert horizon["source"] == "system_inferred"
    assert "paper_value" not in horizon
    assert horizon["protocol_role"] == "forecast_call_horizon"
    assert horizon["paper_value_status"] == "paper_unspecified"
    assert "validation_span" not in params
    assert "test_span" not in params
    assert "nearby notation and validation or test spans" in horizon["reasoning"]

    stale = copy.deepcopy(output)
    stale["params"]["context_length"]["paper_value"] = 26
    with pytest.raises(ValidationError, match="second, conflicting protocol value"):
        Params.model_validate(stale)
    errors = _evaluation_protocol_errors(spec, stale["params"])
    assert len(errors) == 1
    assert "paper_value=26 is stale" in errors[0]
    assert "paper-stated 'context_length' conversion" in errors[0]
    assert "10 'week' / 1 'week'/step = 10 runtime steps" in errors[0]


def test_boolean_runtime_cannot_equal_numeric_paper_protocol_value() -> None:
    spec = _pdfgnn_spec()
    spec["comparison"]["pluggable_component"]["signature"] = (
        "forecast(data, context_length=True, forecast_horizon=4, seed=0) "
        "-> Result"
    )
    _protocol_quantity(spec, "context_length")["value"] = 1

    output = derive(spec, paper_text=PDFGNN_PAPER_TEXT)
    context = output["params"]["context_length"]
    assert context["value"] is True
    assert context["source"] != "paper"
    assert context["paper_value"] == 1

    with pytest.raises(ValidationError, match="not a boolean"):
        Params.model_validate(output)
    errors = _evaluation_protocol_errors(spec, output["params"])
    assert len(errors) == 1
    assert "booleans cannot equal temporal value 1" in errors[0]


def test_numeric_strings_cannot_enter_typed_protocol_or_runtime() -> None:
    spec = _pdfgnn_spec()
    _protocol_quantity(spec, "context_length")["value"] = "1"
    _protocol_quantity(spec, "context_length")["granularity"] = "1"
    with pytest.raises(ValidationError, match="raw numeric JSON values"):
        MethodSpec.model_validate(spec)

    _protocol_quantity(spec, "context_length")["granularity"] = 1
    spec["comparison"]["pluggable_component"]["signature"] = (
        "forecast(data, context_length='1', forecast_horizon=4, seed=0) "
        "-> Result"
    )
    with pytest.raises(ValueError, match="positive numeric value"):
        derive(spec, paper_text=PDFGNN_PAPER_TEXT)


def test_synthetic_explicit_k_12_preserves_paper_value_without_using_test_26() -> None:
    """Synthetic mechanical control only; this is not second-paper evidence."""
    spec = _pdfgnn_spec()
    horizon_fact = _protocol_quantity(spec, "forecast_call_horizon")
    horizon_fact.update(
        {
            "value": 12,
            "paper_value_status": "paper_stated",
            "paper_names": ["one-call forecast horizon"],
            "evidence_quote": (
                "Synthetic mechanical control: one-call forecast horizon K = 12."
            ),
            "paper_section": "Synthetic mechanical control",
            "paper_element_ids": ["synthetic-explicit-k"],
        }
    )

    validated = MethodSpec.model_validate(spec)
    assert cross_check_evaluation_protocol(validated, ROOT) == []

    output = derive(spec, paper_text=horizon_fact["evidence_quote"])
    horizon = output["params"]["forecast_horizon"]
    assert horizon["value"] == 4
    assert horizon["source"] == "system_default"
    assert horizon["paper_value"] == 12
    assert horizon["protocol_role"] == "forecast_call_horizon"
    assert horizon["paper_value_status"] == "paper_stated"
    assert horizon["paper_value"] != 26


def test_synthetic_rolling_origin_k_4_and_test_26_remain_distinct() -> None:
    """Synthetic mechanical control only; this is not second-paper evidence."""
    spec = _pdfgnn_spec()
    scheme = spec["comparison"]["evaluation_protocol"]["scheme"]
    scheme.update(
        {
            "kind": "rolling_origin",
            "description": "Synthetic rolling-origin evaluation control.",
            "evidence_quote": "Synthetic control uses rolling forecast origins.",
            "paper_section": "Synthetic mechanical control",
            "paper_element_ids": ["synthetic-rolling-origin"],
        }
    )
    horizon_fact = _protocol_quantity(spec, "forecast_call_horizon")
    horizon_fact.update(
        {
            "value": 4,
            "paper_value_status": "paper_stated",
            "paper_names": ["forecast call"],
            "evidence_quote": "Synthetic control states K = 4 per forecast call.",
            "paper_section": "Synthetic mechanical control",
            "paper_element_ids": ["synthetic-k-four"],
        }
    )

    validated = MethodSpec.model_validate(spec)
    protocol = validated.comparison.evaluation_protocol
    assert protocol is not None
    assert protocol.scheme.kind is not None
    assert protocol.scheme.kind.value == "rolling_origin"
    assert cross_check_evaluation_protocol(validated, ROOT) == []

    output = derive(spec, paper_text="\n".join(
        [scheme["evidence_quote"], horizon_fact["evidence_quote"]]
    ))
    horizon = output["params"]["forecast_horizon"]
    assert horizon["value"] == 4
    assert horizon["source"] == "paper"
    assert "paper_value" not in horizon
    assert _protocol_quantity(spec, "test_span")["value"] == 26
    assert "test_span" not in output["params"]


def test_protocol_rejects_duplicate_and_missing_roles() -> None:
    duplicate = _pdfgnn_spec()
    duplicate_quantity = _protocol_quantity(duplicate, "test_span")
    duplicate_quantity.update({
        "role": "validation_span",
        "paper_names": ["validation evaluation window"],
        "evidence_quote": "The validation evaluation window is 26 weeks.",
    })
    with pytest.raises(ValidationError, match="conflicting duplicate role facts"):
        MethodSpec.model_validate(duplicate)

    missing = _pdfgnn_spec()
    missing["comparison"]["evaluation_protocol"]["quantities"].pop()
    with pytest.raises(ValidationError) as exc_info:
        MethodSpec.model_validate(missing)
    assert any(
        error["loc"]
        == ("comparison", "evaluation_protocol", "quantities")
        and error["type"] == "too_short"
        for error in exc_info.value.errors()
    )


def test_role_value_evidence_rejects_swaps_and_accepts_paired_order() -> None:
    swapped = _pdfgnn_spec()
    _protocol_quantity(swapped, "validation_span")["value"] = 26
    _protocol_quantity(swapped, "test_span")["value"] = 13
    with pytest.raises(ValidationError, match="strict name/role evidence"):
        MethodSpec.model_validate(swapped)

    paired = _pdfgnn_spec()
    paired_quote = (
        "The validation and test spans are 13 and 26 weeks, respectively."
    )
    validation = _protocol_quantity(paired, "validation_span")
    test = _protocol_quantity(paired, "test_span")
    validation.update({
        "paper_names": ["validation"],
        "evidence_quote": paired_quote,
    })
    test.update({
        "paper_names": ["test"],
        "evidence_quote": paired_quote,
    })
    MethodSpec.model_validate(paired)

    test["value"] = 13
    with pytest.raises(ValidationError, match="strict name/role evidence"):
        MethodSpec.model_validate(paired)


def test_t_plus_one_index_and_test_span_cannot_become_horizon_values() -> None:
    indexed = _pdfgnn_spec()
    horizon = _protocol_quantity(indexed, "forecast_call_horizon")
    horizon.update({"value": 1, "paper_value_status": "paper_stated"})
    with pytest.raises(ValidationError, match=r"T\+1 index"):
        MethodSpec.model_validate(indexed)

    crossed_subject = _pdfgnn_spec()
    horizon = _protocol_quantity(crossed_subject, "forecast_call_horizon")
    horizon.update({
        "paper_names": ["forecast horizon"],
        "value": 26,
        "paper_value_status": "paper_stated",
        "evidence_quote": (
            "The forecast horizon K is scored on a test set that covers "
            "26 weeks."
        ),
    })
    with pytest.raises(ValidationError, match="another role's span"):
        MethodSpec.model_validate(crossed_subject)


def test_stated_horizon_cannot_hide_behind_unspecified_status() -> None:
    spec = _pdfgnn_spec()
    horizon = _protocol_quantity(spec, "forecast_call_horizon")
    horizon.update({
        "paper_names": ["forecast horizon"],
        "evidence_quote": "The forecast horizon K = 12 future steps.",
    })

    with pytest.raises(ValidationError, match="cannot be hidden"):
        MethodSpec.model_validate(spec)


@pytest.mark.parametrize("symbol", ["k", "K_pred"])
def test_lowercase_and_multichar_symbols_are_exact_typed_identities(
    symbol: str,
) -> None:
    spec = _pdfgnn_spec()
    horizon = _protocol_quantity(spec, "forecast_call_horizon")
    horizon.update({
        "paper_names": ["forecast horizon"],
        "paper_symbols": [symbol],
        "value": 12,
        "paper_value_status": "paper_stated",
        "evidence_quote": f"The forecast horizon {symbol} = 12 future steps.",
    })
    MethodSpec.model_validate(spec)

    omitted = copy.deepcopy(spec)
    omitted_horizon = _protocol_quantity(omitted, "forecast_call_horizon")
    omitted_horizon["paper_symbols"] = []
    # Keep the paper value bound through the prose name so the ownership
    # gate (now protocol-level) is what fires, not value binding.
    omitted_horizon["evidence_quote"] = (
        f"The forecast horizon is 12 future steps. "
        f"{symbol} denotes the forecast horizon."
    )
    with pytest.raises(ValidationError, match="paper_symbols omit"):
        MethodSpec.model_validate(omitted)


def test_symbol_notation_cannot_be_laundered_through_prose_names() -> None:
    spec = _pdfgnn_spec()
    horizon = _protocol_quantity(spec, "forecast_call_horizon")
    horizon["paper_names"] = ["K"]
    horizon["paper_symbols"] = []

    with pytest.raises(ValidationError, match="notation-shaped"):
        MethodSpec.model_validate(spec)


def test_ordinary_role_phrase_needs_no_invented_symbol() -> None:
    spec = _pdfgnn_spec()
    horizon = _protocol_quantity(spec, "forecast_call_horizon")
    horizon.update({
        "paper_names": ["forecast horizon"],
        "paper_symbols": [],
        "evidence_quote": (
            "The forecast horizon remains unspecified for future steps."
        ),
    })

    MethodSpec.model_validate(spec)


def test_parenthetical_multichar_symbol_must_be_declared() -> None:
    spec = _pdfgnn_spec()
    horizon = _protocol_quantity(spec, "forecast_call_horizon")
    horizon.update({
        "paper_names": ["forecast horizon"],
        "paper_symbols": [],
        "evidence_quote": (
            "The forecast horizon (tau) remains unspecified for future steps."
        ),
    })
    with pytest.raises(ValidationError, match="paper_symbols omit.*tau"):
        MethodSpec.model_validate(spec)

    horizon["paper_symbols"] = ["tau"]
    MethodSpec.model_validate(spec)


def test_sibling_owned_symbol_in_surrounding_quote_is_not_omitted() -> None:
    # 2026-08-10 Attempt 0 reproducer: the role-term gate demands enough
    # surrounding paper text, and that text legitimately mentions the
    # horizon's K inside the context quantity's quote. K is owned by the
    # sibling forecast_call_horizon quantity, so typed ownership is intact
    # and the context quantity must not be forced to claim K itself.
    spec = _pdfgnn_spec()
    context = _protocol_quantity(spec, "context_length")
    context.update({
        "paper_names": ["Context length", "demand lags"],
        "paper_symbols": ["P"],
        "evidence_quote": (
            PDFGNN_CONTEXT_QUOTE
            + " [...] we limit node features to P demand lags. [...] "
            + PDFGNN_HORIZON_QUOTE
        ),
    })
    MethodSpec.model_validate(spec)


def test_unowned_role_adjacent_symbol_fails_at_the_protocol_level() -> None:
    # The ownership guarantee survives the protocol-level move: a symbol the
    # quote uses in a role-signature position that NO quantity declares is
    # still a finding.
    spec = _pdfgnn_spec()
    context = _protocol_quantity(spec, "context_length")
    context.update({
        "paper_names": ["Context length"],
        "paper_symbols": [],
        "evidence_quote": (
            PDFGNN_CONTEXT_QUOTE + " ... we predict the future J time steps."
        ),
    })
    with pytest.raises(ValidationError, match="paper_symbols omit.*'J'"):
        MethodSpec.model_validate(spec)


def _run_strict_validator(spec: dict, tmp_path: Path) -> subprocess.CompletedProcess:
    spec_path = tmp_path / "method_spec.json"
    paper_path = tmp_path / "paper.md"
    spec_path.write_text(json.dumps(spec), encoding="utf-8")
    paper_path.write_text(PDFGNN_PAPER_TEXT, encoding="utf-8")
    return subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "validate_method_spec.py"),
            str(spec_path),
            "--paper-md",
            str(paper_path),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def _fragmented_context_spec() -> dict:
    # The 2026-08-10 Attempt 0 shape: notation lives in Section 3.2 prose,
    # the stated value lives in an appendix table row, and no contiguous
    # passage carries both. The elision marker is the honest declaration.
    spec = _pdfgnn_spec()
    context = _protocol_quantity(spec, "context_length")
    context.update({
        "paper_names": ["Context length", "demand lags"],
        "paper_symbols": ["P"],
        "evidence_quote": (
            PDFGNN_CONTEXT_QUOTE
            + " [...] we limit node features to P demand lags"
        ),
    })
    return spec


def test_fragmented_protocol_quote_passes_the_verbatim_floor(
    tmp_path: Path,
) -> None:
    proc = _run_strict_validator(_fragmented_context_spec(), tmp_path)
    assert proc.returncode == 0, proc.stderr


def test_fragment_paraphrase_still_fails_the_verbatim_floor(
    tmp_path: Path,
) -> None:
    spec = _fragmented_context_spec()
    context = _protocol_quantity(spec, "context_length")
    context["evidence_quote"] = (
        PDFGNN_CONTEXT_QUOTE
        + " [...] node features are limited to P demand lags"
    )
    proc = _run_strict_validator(spec, tmp_path)
    assert proc.returncode == 1
    assert "is not a verbatim passage of the paper" in proc.stderr
    assert "node features are limited" in proc.stderr


def test_fragmented_quote_grounds_each_cited_element_per_fragment(
    tmp_path: Path,
) -> None:
    # pdfgnn Attempt 3 halt (2026-08-10): a joined quote is never a
    # substring of any single paper-map element, so the whole-string
    # overlap test failed every fragmented quote. Each cited element must
    # overlap at least one fragment.
    spec = _fragmented_context_spec()
    context = _protocol_quantity(spec, "context_length")
    context["paper_element_ids"] = [PROTOCOL_IDS["context"], "hyp-lags"]
    spec_path = _write_paper_map(tmp_path)

    assert cross_check_paper_map(MethodSpec.model_validate(spec), spec_path) == []


def test_fragmented_quote_still_rejects_an_ungrounded_element(
    tmp_path: Path,
) -> None:
    spec = _fragmented_context_spec()
    context = _protocol_quantity(spec, "context_length")
    context["paper_element_ids"] = [PROTOCOL_IDS["split"]]
    spec_path = _write_paper_map(tmp_path)

    errors = cross_check_paper_map(MethodSpec.model_validate(spec), spec_path)
    assert len(errors) == 1
    assert "does not overlap" in errors[0]


def test_uncovered_passage_declares_honest_empty_element_ids(
    tmp_path: Path,
) -> None:
    # pdfgnn Attempt 3 (2026-08-10): the decomposer's map covered none of
    # the protocol passages, the schema demanded at least one id, and the
    # cross-check rejected everything citable — a cross-producer deadlock.
    # An empty list is the honest state (mirrors the methodology-element
    # rule) and must validate end to end.
    spec = _pdfgnn_spec()
    context = _protocol_quantity(spec, "context_length")
    context["paper_element_ids"] = []
    spec["comparison"]["evaluation_protocol"]["scheme"]["paper_element_ids"] = []
    spec_path = _write_paper_map(tmp_path)

    assert cross_check_paper_map(MethodSpec.model_validate(spec), spec_path) == []


def test_claim_scope_ends_at_a_boundary_after_a_captured_number() -> None:
    # pdfgnn Attempt 4 halt (2026-08-10): the bridge scan bound the
    # neighbors clause's 5 to context length across the closing paren.
    from schemas.method_spec import evaluation_protocol_candidate_values

    spec = _pdfgnn_spec()
    context = MethodSpec.model_validate(spec).comparison.evaluation_protocol.quantities[0]
    text = (
        "Hyperparameters controlling graph construction: similarity cutoff "
        "threshold (0.95 for both public datasets), number of demand lags P "
        "(context length = 10), and maximum number of neighbors for "
        "neighborhood sampling (10 for retail, 5 for e-commerce)."
    )
    assert evaluation_protocol_candidate_values(context, text) == [10]


def test_compound_claims_without_a_boundary_all_capture() -> None:
    from schemas.method_spec import evaluation_protocol_candidate_values

    spec = _pdfgnn_spec()
    context = MethodSpec.model_validate(spec).comparison.evaluation_protocol.quantities[0]
    text = "The context length is 10 in retail and 12 in e-commerce runs."
    assert evaluation_protocol_candidate_values(context, text) == [10, 12]


def test_summary_element_with_shared_run_grounds_the_citation(
    tmp_path: Path,
) -> None:
    # pdfgnn Attempt 4: a decomposer element is itself a summary with its
    # own elisions, so neither containment direction holds even when it
    # shares an eight-word contiguous run with the quote.
    spec = _fragmented_context_spec()
    context = _protocol_quantity(spec, "context_length")
    context["evidence_quote"] = (
        PDFGNN_CONTEXT_QUOTE + " [...] " + PDFGNN_LAGS_QUOTE
    )
    context["paper_element_ids"] = ["hyp-graph-summary"]
    spec_path = _write_paper_map(tmp_path)
    pm_path = spec_path.parent / "paper_map.json"
    pm = json.loads(pm_path.read_text(encoding="utf-8"))
    pm["elements"].append({
        "id": "hyp-graph-summary",
        "source_text": (
            "We define an edge between two articles if similarity exceeds a "
            "threshold... The number of lags serves as a hyperparameter."
        ),
        "section": "Evaluation protocol",
    })
    pm_path.write_text(json.dumps(pm), encoding="utf-8")

    assert cross_check_paper_map(MethodSpec.model_validate(spec), spec_path) == []


def test_elision_markers_cannot_be_empty_fragments() -> None:
    spec = _fragmented_context_spec()
    context = _protocol_quantity(spec, "context_length")
    context["evidence_quote"] = "[...] " + context["evidence_quote"]
    with pytest.raises(ValidationError, match="non-empty verbatim passage"):
        MethodSpec.model_validate(spec)


def test_paper_names_are_complete_unique_role_identities() -> None:
    missing = _pdfgnn_spec()
    _protocol_quantity(missing, "forecast_call_horizon").pop("paper_names")
    with pytest.raises(ValidationError):
        MethodSpec.model_validate(missing)

    crossed = _pdfgnn_spec()
    _protocol_quantity(crossed, "validation_span")["paper_names"] = ["test set"]
    with pytest.raises(ValidationError, match="validation quantity"):
        MethodSpec.model_validate(crossed)

    duplicate = _pdfgnn_spec()
    _protocol_quantity(duplicate, "validation_span")["paper_names"] = [
        "validation set", "shared window"
    ]
    _protocol_quantity(duplicate, "test_span")["paper_names"] = [
        "test set", "shared window"
    ]
    with pytest.raises(ValidationError, match="assigned to both"):
        MethodSpec.model_validate(duplicate)


@pytest.mark.parametrize(
    ("kind", "status", "message"),
    [
        (None, "paper_stated", "paper_stated, so kind must be present"),
        (
            "single_holdout",
            "paper_unspecified",
            "paper_unspecified, so kind must be null",
        ),
    ],
)
def test_scheme_kind_and_paper_value_status_are_consistent(
    kind: str | None,
    status: str,
    message: str,
) -> None:
    spec = _pdfgnn_spec()
    scheme = spec["comparison"]["evaluation_protocol"]["scheme"]
    scheme["kind"] = kind
    scheme["paper_value_status"] = status

    with pytest.raises(ValidationError, match=message):
        MethodSpec.model_validate(spec)


def test_scheme_kind_must_be_grounded_by_its_own_evidence() -> None:
    spec = _pdfgnn_spec()
    spec["comparison"]["evaluation_protocol"]["scheme"]["kind"] = (
        "rolling_origin"
    )

    with pytest.raises(ValidationError, match="not exclusively grounded"):
        MethodSpec.model_validate(spec)


def test_scheme_cannot_hide_a_stated_kind_or_use_an_open_ended_kind() -> None:
    hidden = _pdfgnn_spec()
    scheme = hidden["comparison"]["evaluation_protocol"]["scheme"]
    scheme.update({"kind": None, "paper_value_status": "paper_unspecified"})
    with pytest.raises(ValidationError, match="cannot be hidden"):
        MethodSpec.model_validate(hidden)

    open_ended = _pdfgnn_spec()
    open_ended["comparison"]["evaluation_protocol"]["scheme"]["kind"] = "other"
    with pytest.raises(ValidationError):
        MethodSpec.model_validate(open_ended)


def test_scheme_description_cannot_contradict_typed_kind() -> None:
    spec = _pdfgnn_spec()
    spec["comparison"]["evaluation_protocol"]["scheme"]["description"] = (
        "Rolling-origin evaluation across many forecast origins."
    )

    with pytest.raises(ValidationError, match="description contains conflicting"):
        MethodSpec.model_validate(spec)


@pytest.mark.parametrize(
    ("status", "value", "message"),
    [
        ("paper_stated", None, "paper_stated, so value must be present"),
        ("paper_unspecified", 12, "paper_unspecified, so value must be null"),
    ],
)
def test_quantity_value_and_paper_value_status_are_consistent(
    status: str,
    value: int | None,
    message: str,
) -> None:
    spec = _pdfgnn_spec()
    horizon = _protocol_quantity(spec, "forecast_call_horizon")
    horizon["paper_value_status"] = status
    horizon["value"] = value

    with pytest.raises(ValidationError, match=message):
        MethodSpec.model_validate(spec)


def test_strict_taxonomy_binding_rejects_missing_and_cross_role_carriers() -> None:
    spec = _pdfgnn_spec()
    assert cross_check_evaluation_protocol(MethodSpec.model_validate(spec), ROOT) == []

    missing = copy.deepcopy(spec)
    _protocol_quantity(missing, "forecast_call_horizon")["parameter_name"] = None
    errors = cross_check_evaluation_protocol(
        MethodSpec.model_validate(missing), ROOT
    )
    assert len(errors) == 1
    assert "forecast_horizon" in errors[0]
    assert "no comparison.evaluation_protocol quantity binds" in errors[0]

    crossed = copy.deepcopy(spec)
    _protocol_quantity(crossed, "context_length")["parameter_name"] = (
        "forecast_horizon"
    )
    _protocol_quantity(crossed, "forecast_call_horizon")["parameter_name"] = (
        "context_length"
    )
    errors = cross_check_evaluation_protocol(
        MethodSpec.model_validate(crossed), ROOT
    )
    assert len(errors) == 2
    assert all("cross-role carrier join is forbidden" in error for error in errors)
    assert any(
        "parameter_name='forecast_horizon'" in error
        and "role='context_length'" in error
        and "protocol_role='forecast_call_horizon'" in error
        for error in errors
    )

    invented = copy.deepcopy(spec)
    _protocol_quantity(invented, "validation_span")["parameter_name"] = (
        "validation_span"
    )
    errors = cross_check_evaluation_protocol(
        MethodSpec.model_validate(invented), ROOT
    )
    assert len(errors) == 1
    assert "do not invent validation/test runtime carriers" in errors[0]


def test_protocol_rejects_one_runtime_carrier_bound_to_two_roles() -> None:
    spec = _pdfgnn_spec()
    _protocol_quantity(spec, "validation_span")["parameter_name"] = (
        "forecast_horizon"
    )
    with pytest.raises(ValidationError, match="one runtime carrier"):
        MethodSpec.model_validate(spec)


def test_glossary_entry_cannot_join_two_protocol_carriers() -> None:
    spec = _pdfgnn_spec()
    spec["critical_requirements"]["param_glossary"] = [
        {
            "name": "shared",
            "aliases": ["context_length", "forecast_horizon"],
            "meaning_quote": "A deliberately ambiguous shared quantity.",
            "paper_section": "Synthetic invalid control",
            "paper_value": None,
        }
    ]
    with pytest.raises(ValidationError, match="joins multiple"):
        MethodSpec.model_validate(spec)


def test_protocol_carrier_conflicts_with_legacy_glossary_paper_value() -> None:
    spec = _pdfgnn_spec()
    spec["critical_requirements"]["param_glossary"] = [
        {
            "name": "K",
            "aliases": ["forecast_horizon"],
            "meaning_quote": PDFGNN_HORIZON_QUOTE,
            "paper_section": "Evaluation protocol",
            "paper_value": 26,
        }
    ]

    with pytest.raises(ValidationError) as exc_info:
        MethodSpec.model_validate(spec)
    message = str(exc_info.value)
    assert "param_glossary" in message
    assert "paper_value=26.0 conflicts" in message
    assert "role='forecast_call_horizon'" in message
    assert "paper_value_status='paper_unspecified'" in message


def test_temporal_protocol_carrier_is_exclusive_from_scale_lane() -> None:
    spec = _pdfgnn_spec()
    spec["critical_requirements"]["scale_dependent_hyperparameters"] = [
        {
            "name": "forecast_horizon",
            "paper_value": 26,
            "formula": None,
            "assumes_data_scale": "weekly observations",
            "description": "Intentionally invalid temporal scale-lane duplicate.",
            "paper_section": "Evaluation protocol",
        }
    ]

    with pytest.raises(ValidationError) as exc_info:
        MethodSpec.model_validate(spec)
    message = str(exc_info.value)
    assert "scale_dependent_hyperparameters" in message
    assert "parameter_name='forecast_horizon'" in message
    assert "role='forecast_call_horizon'" in message
    assert "temporal protocol counts belong only" in message


@pytest.mark.parametrize("legacy_lane", ["glossary", "scale"])
def test_unlinked_paper_symbol_cannot_bypass_protocol_exclusivity(
    legacy_lane: str,
) -> None:
    spec = _pdfgnn_spec()
    horizon = _protocol_quantity(spec, "forecast_call_horizon")
    # Deliberately incomplete identity list: K remains verbatim in evidence.
    horizon["paper_names"] = ["future"]
    horizon["paper_symbols"] = []
    if legacy_lane == "glossary":
        spec["critical_requirements"]["param_glossary"] = [
            {
                "name": "K",
                "aliases": [],
                "meaning_quote": PDFGNN_HORIZON_QUOTE,
                "paper_section": "Evaluation protocol",
                "paper_value": 26,
            }
        ]
    else:
        spec["critical_requirements"]["scale_dependent_hyperparameters"] = [
            {
                "name": "K",
                "paper_value": 26,
                "formula": None,
                "assumes_data_scale": "weekly observations",
                "description": "Intentionally invalid temporal scale carrier.",
                "paper_section": "Evaluation protocol",
            }
        ]

    with pytest.raises(ValidationError, match="paper_symbols omit"):
        MethodSpec.model_validate(spec)


def test_protocol_conflict_with_methodology_contract_is_not_silently_selected() -> None:
    spec = _pdfgnn_spec()
    element = _methodology_element()
    element["paper_evidence"] = "The paper sets forecast_horizon = 26."
    element["required_behavior"] = "Use forecast_horizon = 26."
    spec = _add_methodology_contract(spec, elements=[element])

    with pytest.raises(ValidationError, match="marks that value paper_unspecified"):
        MethodSpec.model_validate(spec)


@pytest.mark.parametrize("granularity", [0, True])
def test_protocol_granularity_must_be_a_positive_non_boolean_number(
    granularity: int | bool,
) -> None:
    spec = _pdfgnn_spec()
    _protocol_quantity(spec, "forecast_call_horizon")["granularity"] = granularity

    with pytest.raises(ValidationError):
        MethodSpec.model_validate(spec)


def test_axis_evidence_rejects_unit_and_cadence_laundering() -> None:
    wrong_unit = _pdfgnn_spec()
    horizon = _protocol_quantity(wrong_unit, "forecast_call_horizon")
    horizon.update({
        "paper_names": ["forecast horizon"],
        "value": 13,
        "paper_value_status": "paper_stated",
        "evidence_quote": "The forecast horizon K = 13 days.",
    })
    with pytest.raises(ValidationError, match="explicit unit.*day"):
        MethodSpec.model_validate(wrong_unit)

    unrelated_day = _pdfgnn_spec()
    horizon = _protocol_quantity(unrelated_day, "forecast_call_horizon")
    horizon.update({
        "unit": "day",
        "axis_evidence_quote": (
            "We aggregate weekly demand into the target time series and add "
            "day of month as a feature."
        ),
    })
    with pytest.raises(ValidationError, match="unit provenance must be explicit"):
        MethodSpec.model_validate(unrelated_day)

    duration_is_not_cadence = _pdfgnn_spec()
    horizon = _protocol_quantity(duration_is_not_cadence, "forecast_call_horizon")
    horizon["granularity"] = 13
    horizon["axis_evidence_quote"] = (
        "We aggregate weekly demand into the target time series over a "
        "13-week forecast window."
    )
    with pytest.raises(ValidationError, match="window duration"):
        MethodSpec.model_validate(duration_is_not_cadence)

    explicit_cadence = _pdfgnn_spec()
    horizon = _protocol_quantity(explicit_cadence, "forecast_call_horizon")
    horizon["axis_evidence_quote"] = (
        "We record the target demand every 13 weeks."
    )
    with pytest.raises(ValidationError, match="cadence value.*13"):
        MethodSpec.model_validate(explicit_cadence)


def test_nonunit_paper_unspecified_cadence_remains_system_owned() -> None:
    spec = _pdfgnn_spec()
    horizon = _protocol_quantity(spec, "forecast_call_horizon")
    horizon.update({
        "unit": "hour",
        "granularity": 0.5,
        "axis_evidence_quote": "We record target demand every 0.5 hours.",
    })
    validated = MethodSpec.model_validate(spec)
    assert cross_check_evaluation_protocol(validated, ROOT) == []

    output = derive(spec, paper_text=PDFGNN_PAPER_TEXT)
    runtime = output["params"]["forecast_horizon"]
    assert runtime["value"] == 4
    assert runtime["source"] == "system_inferred"
    assert runtime["protocol_value"] is None
    assert runtime["protocol_unit"] == "hour"
    assert runtime["protocol_granularity"] == 0.5


@pytest.mark.parametrize("value", ["", "   "])
def test_protocol_unit_cannot_be_empty_or_whitespace(value: str) -> None:
    spec = _pdfgnn_spec()
    _protocol_quantity(spec, "forecast_call_horizon")["unit"] = value

    with pytest.raises(ValidationError):
        MethodSpec.model_validate(spec)


@pytest.mark.parametrize("field", ["evidence_quote", "paper_section"])
def test_protocol_quantity_evidence_cannot_be_whitespace(field: str) -> None:
    spec = _pdfgnn_spec()
    _protocol_quantity(spec, "forecast_call_horizon")[field] = "   "

    with pytest.raises(ValidationError):
        MethodSpec.model_validate(spec)


@pytest.mark.parametrize(
    "field", ["description", "evidence_quote", "paper_section"]
)
def test_protocol_scheme_evidence_cannot_be_whitespace(field: str) -> None:
    spec = _pdfgnn_spec()
    spec["comparison"]["evaluation_protocol"]["scheme"][field] = "   "

    with pytest.raises(ValidationError):
        MethodSpec.model_validate(spec)


def test_derived_params_carry_role_unit_granularity_status_and_evidence() -> None:
    spec = _pdfgnn_spec()
    output = derive(spec, paper_text=PDFGNN_PAPER_TEXT)
    validated = Params.model_validate(output)

    horizon = validated.params["forecast_horizon"]
    assert horizon.protocol_role == "forecast_call_horizon"
    assert horizon.protocol_value is None
    assert horizon.protocol_unit == "week"
    assert horizon.protocol_granularity == 1
    assert horizon.protocol_axis_says == PDFGNN_AXIS_QUOTE
    assert horizon.protocol_axis_section == "Datasets"
    assert horizon.protocol_axis_element_ids == [PROTOCOL_IDS["axis"]]
    assert horizon.paper_value_status == "paper_unspecified"
    assert horizon.paper_says == PDFGNN_HORIZON_QUOTE
    assert horizon.paper_section == "Evaluation protocol"
    assert horizon.paper_element_ids == [PROTOCOL_IDS["horizon"]]
    assert _evaluation_protocol_errors(spec, output["params"]) == []

    broken = copy.deepcopy(output)
    broken["params"]["forecast_horizon"]["protocol_granularity"] = 13
    errors = _evaluation_protocol_errors(spec, broken["params"])
    assert len(errors) == 1
    assert "params.forecast_horizon.protocol_granularity=13" in errors[0]
    assert "granularity" in errors[0]

    incomplete = copy.deepcopy(output)
    incomplete["params"]["forecast_horizon"].pop("protocol_unit")
    with pytest.raises(ValidationError, match="typed protocol metadata"):
        Params.model_validate(incomplete)

    boolean_granularity = copy.deepcopy(output)
    boolean_granularity["params"]["forecast_horizon"][
        "protocol_granularity"
    ] = True
    with pytest.raises(ValidationError, match="not a boolean"):
        Params.model_validate(boolean_granularity)

    for field in (
        "paper_says", "paper_section", "protocol_axis_says",
        "protocol_axis_section",
    ):
        whitespace = copy.deepcopy(output)
        whitespace["params"]["forecast_horizon"][field] = "   "
        with pytest.raises(ValidationError, match="auditable"):
            Params.model_validate(whitespace)

    # An EMPTY (or absent) citation list is the honest state when the
    # decomposer's map covers none of the quoted passage — stage 1 blesses
    # it, so the params carrier must hold it (pdfgnn 2026-08-11 stage-2x
    # halt: derive_params emitted entries its own validator rejected).
    missing_axis = copy.deepcopy(output)
    missing_axis["params"]["forecast_horizon"].pop(
        "protocol_axis_element_ids"
    )
    validated_empty = Params.model_validate(missing_axis)
    assert validated_empty.params[
        "forecast_horizon"
    ].protocol_axis_element_ids == []
    honest_empty = copy.deepcopy(output)
    honest_empty["params"]["forecast_horizon"]["paper_element_ids"] = []
    assert Params.model_validate(honest_empty).params[
        "forecast_horizon"
    ].paper_element_ids == []


def test_protocol_paper_element_crosswalk_accepts_real_ids_and_rejects_dangling(
    tmp_path: Path,
) -> None:
    spec = _pdfgnn_spec()
    validated = MethodSpec.model_validate(spec)
    spec_path = _write_paper_map(tmp_path)
    assert cross_check_paper_map(validated, spec_path) == []

    dangling = _pdfgnn_spec()
    _protocol_quantity(dangling, "forecast_call_horizon")[
        "paper_element_ids"
    ] = ["horizon-typo"]
    errors = cross_check_paper_map(MethodSpec.model_validate(dangling), spec_path)
    assert len(errors) == 1
    assert "role='forecast_call_horizon'" in errors[0]
    assert "horizon-typo" in errors[0]
    assert "role, value, unit, granularity, and evidence" in errors[0]


def test_protocol_axis_crosswalk_rejects_dangling_wrong_text_and_wrong_section(
    tmp_path: Path,
) -> None:
    spec_path = _write_paper_map(tmp_path)

    dangling = _pdfgnn_spec()
    _protocol_quantity(dangling, "forecast_call_horizon")[
        "axis_paper_element_ids"
    ] = ["axis-typo"]
    errors = cross_check_paper_map(MethodSpec.model_validate(dangling), spec_path)
    assert len(errors) == 1
    assert ".axis_evidence.axis_paper_element_ids" in errors[0]

    wrong_text = _pdfgnn_spec()
    _protocol_quantity(wrong_text, "forecast_call_horizon")[
        "axis_paper_element_ids"
    ] = [PROTOCOL_IDS["horizon"]]
    errors = cross_check_paper_map(MethodSpec.model_validate(wrong_text), spec_path)
    assert len(errors) == 2
    assert any("valid-but-wrong paper-map ID" in error for error in errors)
    assert any("evidence location and identity must agree" in error for error in errors)

    wrong_section = _pdfgnn_spec()
    _protocol_quantity(wrong_section, "forecast_call_horizon")[
        "axis_paper_section"
    ] = "Figure 3"
    errors = cross_check_paper_map(MethodSpec.model_validate(wrong_section), spec_path)
    assert len(errors) == 1
    assert "evidence location and identity must agree" in errors[0]


@pytest.mark.parametrize(
    ("declared", "actual"),
    [("Data", "Database"), ("Figure 3", "Section 3")],
)
def test_protocol_section_crosswalk_does_not_use_substring_or_number_only(
    tmp_path: Path,
    declared: str,
    actual: str,
) -> None:
    spec = _pdfgnn_spec()
    horizon = _protocol_quantity(spec, "forecast_call_horizon")
    horizon["paper_section"] = declared
    spec_path = _write_paper_map(tmp_path)
    paper_map = json.loads(spec_path.with_name("paper_map.json").read_text())
    next(
        item for item in paper_map["elements"]
        if item["id"] == PROTOCOL_IDS["horizon"]
    )["section"] = actual
    spec_path.with_name("paper_map.json").write_text(json.dumps(paper_map))

    errors = cross_check_paper_map(MethodSpec.model_validate(spec), spec_path)
    assert len(errors) == 1
    assert "evidence location and identity must agree" in errors[0]


@pytest.mark.parametrize("field", ["core_method.summary", "paper_map.name"])
@pytest.mark.parametrize(
    "false_claim",
    [
        "The method uses forecast horizon K = 26 weeks in one call.",
        "The method uses a 26-week forecast horizon K.",
        "The method predicts 26 future weeks per call (K).",
        "The method uses a forecast horizon of 26 weeks.",
        "The forecast horizon of 26 weeks is denoted K.",
        "The one-call horizon K (26 weeks) is evaluated.",
        "Demo forecast horizon K = 26 weeks.",
    ],
)
def test_reader_narratives_cannot_republish_test_span_as_symbolic_horizon(
    tmp_path: Path,
    field: str,
    false_claim: str,
) -> None:
    spec = _pdfgnn_spec()
    spec_path = _write_paper_map(tmp_path)
    if field == "core_method.summary":
        spec["core_method"]["summary"] = false_claim
    else:
        paper_map_path = spec_path.with_name("paper_map.json")
        paper_map = json.loads(paper_map_path.read_text())
        next(
            item for item in paper_map["elements"]
            if item["id"] == PROTOCOL_IDS["split"]
        )["name"] = false_claim
        paper_map_path.write_text(json.dumps(paper_map))

    errors = cross_check_paper_map(MethodSpec.model_validate(spec), spec_path)
    assert len(errors) == 1
    assert "Reader-facing summaries" in errors[0]
    assert "paper_unspecified" in errors[0]


def test_reader_narrative_can_keep_horizon_and_test_span_separate(
    tmp_path: Path,
) -> None:
    spec = _pdfgnn_spec()
    spec["core_method"]["summary"] = (
        "The K-step forecast is scored on a test set that covers 26 weeks."
    )
    spec_path = _write_paper_map(tmp_path)

    assert cross_check_paper_map(MethodSpec.model_validate(spec), spec_path) == []


def test_renderer_withholds_invalid_or_conflicting_raw_protocol() -> None:
    spec = _pdfgnn_spec()
    protocol = spec["comparison"]["evaluation_protocol"]
    protocol["quantities"].append({
        **copy.deepcopy(_protocol_quantity(spec, "forecast_call_horizon")),
        "value": 26,
        "paper_value_status": "paper_stated",
        "paper_names": ["forecast horizon"],
        "evidence_quote": "The forecast horizon K = 26 weeks.",
    })
    rendered = render_evaluation_protocol_block(spec)
    assert "Evaluation protocol unavailable" in rendered
    assert "26 (paper-stated)" not in rendered

    malformed = _pdfgnn_spec()
    _protocol_quantity(malformed, "forecast_call_horizon")[
        "paper_value_status"
    ] = "nonsense"
    _protocol_quantity(malformed, "forecast_call_horizon")["value"] = 99
    rendered = render_evaluation_protocol_block(malformed)
    assert "Evaluation protocol unavailable" in rendered
    assert "99 (paper-stated)" not in rendered


def test_nonforecasting_family_rejects_protocol_contamination() -> None:
    spec = _pdfgnn_spec()
    spec["comparison"]["classification"]["id"] = "motion_planning"
    errors = cross_check_evaluation_protocol(MethodSpec.model_validate(spec), ROOT)
    assert len(errors) == 1
    assert "present for non-forecasting" in errors[0]


def test_protocol_quote_floor_reports_all_failures_in_one_pass(
    tmp_path: Path,
) -> None:
    spec = _pdfgnn_spec()
    protocol = spec["comparison"]["evaluation_protocol"]
    protocol["scheme"]["evidence_quote"] = "We use one chronological split."
    horizon = _protocol_quantity(spec, "forecast_call_horizon")
    horizon["paper_names"] = ["future K time steps"]
    horizon["evidence_quote"] = "We predict future K time steps."
    spec_path = tmp_path / "method_spec.json"
    paper_path = tmp_path / "paper.md"
    spec_path.write_text(json.dumps(spec), encoding="utf-8")
    paper_path.write_text(PDFGNN_PAPER_TEXT, encoding="utf-8")

    proc = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "validate_method_spec.py"),
            str(spec_path),
            "--paper-md",
            str(paper_path),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 1
    assert "evaluation-protocol evidence-quote floor failed" in proc.stderr
    assert "comparison.evaluation_protocol.scheme" in proc.stderr
    assert "role='forecast_call_horizon'" in proc.stderr
    assert "We use one chronological split" in proc.stderr
    assert "We predict future K time steps" in proc.stderr


def test_legacy_non_tsf_spec_without_protocol_remains_readable_and_unblocked() -> None:
    spec = _minimal_valid_spec(paradigm_id="motion_planning")
    validated = MethodSpec.model_validate(spec)

    assert validated.comparison.evaluation_protocol is None
    assert cross_check_evaluation_protocol(validated, ROOT) == []


def test_legacy_tsf_spec_remains_readable_but_fresh_strict_validation_blocks() -> None:
    spec = _minimal_valid_spec(paradigm_id="time_series_forecasting")
    validated = MethodSpec.model_validate(spec)

    assert validated.comparison.evaluation_protocol is None
    errors = cross_check_evaluation_protocol(validated, ROOT)
    assert len(errors) == 1
    assert "required for fresh time-series output" in errors[0]
    assert "paper_value_status='paper_unspecified'" in errors[0]


def test_a_symbol_never_matches_inside_a_word_or_acronym() -> None:
    # pdfgnn 2026-08-11 halt (judge-confirmed pipeline bug): the exact
    # symbol 'P' matched inside 'WMAPE', and the assertion-cue bridge then
    # attached unrelated nearby numbers (a 31.98% RMSE improvement, a
    # 9-feature count) as context-length claims conflicting with the typed
    # value 10. Symbol alternates carry their own word boundaries now.
    from schemas.method_spec import evaluation_protocol_candidate_values

    spec = _pdfgnn_spec()
    quantity = _protocol_quantity(spec, "context_length")
    quantity["paper_symbols"] = ["P"]
    quantity["evidence_quote"] = "number of demand lags P (context length = 10)"
    context = MethodSpec.model_validate(spec).comparison.evaluation_protocol.quantities[0]
    assert context.paper_symbols == ["P"]
    assert context.evidence_quote
    claims_text = (
        "GraphDeepAR consistently outperforms standard DeepAR across three "
        "real-world datasets (retail, e-commerce, adidas) on RMSE, MAE, and "
        "WMAPE metrics. Largest gains on e-commerce (31.98% RMSE "
        "improvement) and on cold-start articles (6.72% RMSE uplift)."
    )
    dataset_text = (
        "Evaluation on the public retail dataset with 629 articles, 148 "
        "weeks, and 19 features. GraphDeepAR vs DeepAR comparison using "
        "RMSE, MAE, and WMAPE metrics. Graph uses 9 article features for "
        "cosine similarity."
    )
    assert evaluation_protocol_candidate_values(context, claims_text) == []
    assert evaluation_protocol_candidate_values(context, dataset_text) == []


def test_a_free_standing_symbol_still_binds_its_stated_value() -> None:
    # The boundary fix must not cost genuine symbol claims.
    from schemas.method_spec import (
        evaluation_protocol_bound_values,
        evaluation_protocol_candidate_values,
    )

    spec = _pdfgnn_spec()
    quantity = _protocol_quantity(spec, "context_length")
    quantity["paper_symbols"] = ["P"]
    quantity["evidence_quote"] = "number of demand lags P (context length = 10)"
    context = MethodSpec.model_validate(spec).comparison.evaluation_protocol.quantities[0]
    text = "We use P = 10 demand lags as encoder context."
    assert evaluation_protocol_bound_values(context, text) == [10]
    assert evaluation_protocol_candidate_values(context, text) == [10]


def test_quote_evidence_floors_report_together_in_one_message() -> None:
    # 2026-08-11 pdfgnn re-roll halt (fix loop exhausted, 5 -> 3 -> 1 -> 1):
    # the name, role-wording, and symbol floors on one evidence_quote were
    # revealed one fix dispatch at a time, so the producer satisfied each in
    # turn while breaking another. All unmet floors must land in ONE error.
    spec = _pdfgnn_spec()
    quantity = _protocol_quantity(spec, "context_length")
    quantity["paper_names"] = ["number of encoder steps"]
    quantity["paper_symbols"] = ["P"]
    # A quote that satisfies none of the floors: no declared name, no role
    # wording, no symbol, and no bindable stated value.
    quantity["evidence_quote"] = (
        "The hyperparameter is selected from the appendix table."
    )

    with pytest.raises(ValidationError) as excinfo:
        MethodSpec.model_validate(spec)
    message = str(excinfo.value)
    assert message.count("Value error") == 1
    assert "does not contain any declared paper_names" in message
    assert "does not state that scientific role" in message
    assert "'P'" in message and "absent from evidence_quote" in message
    assert "strict name/role evidence binds []" in message
    assert "satisfy every floor above at once" in message
    assert "[...]" in message


def test_a_fragment_join_satisfies_all_three_quote_floors() -> None:
    # The sanctioned fix shape for pdfgnn: the symbol lives in Section 3.2
    # prose while the role wording and value live in the appendix table, so
    # the quote joins the two contiguous verbatim fragments.
    spec = _pdfgnn_spec()
    quantity = _protocol_quantity(spec, "context_length")
    quantity["paper_names"] = ["context length", "number of lags"]
    quantity["paper_symbols"] = ["P"]
    quantity["evidence_quote"] = (
        "we limit node features to P demand lags [...] "
        "| Context length | 10 | 10 |"
    )

    validated = MethodSpec.model_validate(spec)
    context = validated.comparison.evaluation_protocol.quantities[0]
    assert context.paper_symbols == ["P"]


def test_grounding_errors_surface_beside_schema_errors(tmp_path: Path) -> None:
    # 2026-08-11 pdfgnn attempt 3 (fix loop exhausted, 2 -> 1 -> 1 -> 5):
    # the paper-map grounding layer ran only after schema validation
    # passed, so the producer burned its whole budget on schema floors and
    # saw five grounding errors with zero retries left. Every
    # individually-parseable protocol record gets its citations checked
    # even while a sibling record still fails schema.
    from scripts.validate_method_spec import (
        best_effort_protocol_grounding_errors,
    )

    spec = _pdfgnn_spec()
    # Quantity 3 (test_span) fails its own schema: quote lacks the name.
    _protocol_quantity(spec, "test_span")["paper_names"] = [
        "a name the quote lacks",
    ]
    # Quantity 2 (validation_span) parses but cites the wrong element.
    _protocol_quantity(spec, "validation_span")["paper_element_ids"] = [
        PROTOCOL_IDS["axis"],
    ]
    spec_path = _write_paper_map(tmp_path)

    errors = best_effort_protocol_grounding_errors(spec, spec_path)
    assert any(
        "validation_span" in error and "does not overlap" in error
        for error in errors
    )
    # The schema-broken record is skipped here; its schema error is
    # reported by the pass this rides along with.
    assert not any("test_span" in error for error in errors)


def test_best_effort_grounding_is_silent_without_a_paper_map(
    tmp_path: Path,
) -> None:
    from scripts.validate_method_spec import (
        best_effort_protocol_grounding_errors,
    )

    spec = _pdfgnn_spec()
    missing = tmp_path / "no_pipeline" / "method_spec.json"
    assert best_effort_protocol_grounding_errors(spec, missing) == []


def test_lag_wording_states_the_context_role_and_the_value_floor_batches() -> None:
    # pdfgnn 2026-08-11 attempt-5 halt: the paper's own Section 3.2 wording
    # ("we limit node features to P demand lags") failed the role floor
    # because "lags" was missing from the context-length vocabulary, and
    # fixing that revealed the value-binding floor one dispatch later. The
    # real attempt-5 quote now fails ONLY on value binding, in one message
    # that carries the fragment-join guidance.
    spec = _pdfgnn_spec()
    quantity = _protocol_quantity(spec, "context_length")
    quantity["paper_names"] = ["context length", "number of lags"]
    quantity["paper_symbols"] = ["P"]
    quantity["evidence_quote"] = (
        "we limit node features to P demand lags: "
        "$\\mathcal{N}_i^t = (y_i^{t-P+1}, \\dots, y_i^t)$ . "
        "The number of lags serves as a hyperparameter."
    )

    with pytest.raises(ValidationError) as excinfo:
        MethodSpec.model_validate(spec)
    message = str(excinfo.value)
    assert message.count("Value error") == 1
    assert "does not state that scientific role" not in message
    assert "strict name/role evidence binds []" in message
    assert "[...]" in message


def test_bidirectional_window_qualifiers_bind_an_abstract_step_axis() -> None:
    # pdfgnn 2026-08-11 attempt 7: the axis quote "for the previous P and
    # the following K time steps" states an abstract step axis exactly as
    # "future K time steps" does, but the qualifier vocabulary accepted
    # only future/historical/context/forecast, so the unit floor (and then
    # the cadence arm) rejected the paper's own formulation.
    spec = _pdfgnn_spec()
    quantity = _protocol_quantity(spec, "context_length")
    quantity["unit"] = "time_step"
    quantity["axis_evidence_quote"] = (
        "Context length | 10 | 10 [...] for the previous P and the "
        "following K time steps"
    )

    validated = MethodSpec.model_validate(spec)
    context = validated.comparison.evaluation_protocol.quantities[0]
    assert context.unit == "time_step"

    # Guardrail: a quote with no axis-binding unit wording still fails,
    # and the message names the unit terms the quote does state.
    bad = _pdfgnn_spec()
    quantity = _protocol_quantity(bad, "context_length")
    quantity["unit"] = "time_step"
    quantity["axis_evidence_quote"] = (
        "we aggregate weekly demand across all stores"
    )
    with pytest.raises(ValidationError) as excinfo:
        MethodSpec.model_validate(bad)
    message = str(excinfo.value)
    assert "no compatible unit term" in message
    assert "'week'" in message
