"""Bounded source-to-marker value-flow coverage for R2C-090."""

from __future__ import annotations

import json

import pytest

from scripts.time_series_training_history import (
    TrainingHistoryCoverageError,
    TrainingHistoryProducerError,
)
from scripts.training_history_notebook_flow import (
    TRAINING_HISTORY_MARKER,
    training_history_notebook_applicable,
    validate_training_history_architecture_contract,
    validate_training_history_notebook_flow,
)


PLAN = {
    "training_history_execution": {
        "schema_version": "1.0.0",
        "training_call": {
            "module": "method.training",
            "callable": "train_model",
            "history_result": {
                "kind": "mapping_key",
                "key": "training_history",
            },
        },
    },
}


def _print(payload: str) -> str:
    return (
        f"print({TRAINING_HISTORY_MARKER!r} + json.dumps(\n"
        f"    {payload}, sort_keys=True, separators=(',', ':'),\n"
        "    ensure_ascii=False, allow_nan=False))\n"
    )


def _good(*extra: str) -> str:
    return (
        "import json\n"
        "from method.training import train_model\n"
        "training_result = train_model(model, targets)\n"
        "training_history = training_result['training_history']\n"
        + "".join(extra)
        + _print("training_history")
    )


def _write_contract(tmp_path, contract: dict) -> None:
    pipeline = tmp_path / ".pipeline"
    pipeline.mkdir(parents=True, exist_ok=True)
    (pipeline / "arch_contract.json").write_text(
        json.dumps(contract), encoding="utf-8"
    )


def test_direct_declared_call_exact_key_and_canonical_marker_are_proven():
    assert validate_training_history_notebook_flow([_good()], PLAN) == {
        "training_module": "method.training",
        "training_callable": "train_model",
        "history_key": "training_history",
        "marker": TRAINING_HISTORY_MARKER,
    }


@pytest.mark.parametrize(
    ("source", "code"),
    [
        (
            "import json\n"
            "from method.training import train_model\n"
            "training_result = train_model(model, targets)\n"
            + _print("{'status': 'recorded'}"),
            "training_history_notebook_marker_literal",
        ),
        (
            "import json\n"
            "from method.training import train_model\n"
            "training_result = train_model(model, targets)\n"
            "training_history = training_result['history']\n"
            + _print("training_history"),
            "training_history_notebook_result_key_disagreement",
        ),
        (
            "import json\n"
            "from method.training import train_model\n"
            "train_model(model, targets)\n"
            + _print("{'status': 'recorded'}"),
            "training_history_notebook_training_result_discarded",
        ),
        (
            "import json\n"
            "from fake.training import train_model\n"
            "training_result = train_model(model, targets)\n"
            "training_history = training_result['training_history']\n"
            + _print("training_history"),
            "training_history_notebook_training_route_disagreement",
        ),
        (
            "import json\n"
            "from method.training import train_model\n"
            "train_model = fake_train_model\n"
            "training_result = train_model(model, targets)\n"
            "training_history = training_result['training_history']\n"
            + _print("training_history"),
            "training_history_notebook_training_binding_rebound",
        ),
        (
            "import json\n"
            "import method.training as training\n"
            "training = fake_training\n"
            "training_result = training.train_model(model, targets)\n"
            "training_history = training_result['training_history']\n"
            + _print("training_history"),
            "training_history_notebook_training_binding_rebound",
        ),
        (
            "import json\n"
            "from method.training import train_model\n"
            "json = FakeSerializer()\n"
            "training_result = train_model(model, targets)\n"
            "training_history = training_result['training_history']\n"
            + _print("training_history"),
            "training_history_notebook_output_binding_rebound",
        ),
        (
            "import fake_json as json\n"
            "from method.training import train_model\n"
            "training_result = train_model(model, targets)\n"
            "training_history = training_result['training_history']\n"
            + _print("training_history"),
            "training_history_notebook_serializer_binding_disagreement",
        ),
        (
            "import json\n"
            "from method.training import train_model\n"
            "print = fake_print\n"
            "training_result = train_model(model, targets)\n"
            "training_history = training_result['training_history']\n"
            + _print("training_history"),
            "training_history_notebook_output_binding_rebound",
        ),
    ],
)
def test_supported_source_disagreements_are_producer_owned(source, code):
    with pytest.raises(TrainingHistoryProducerError) as exc_info:
        validate_training_history_notebook_flow([source], PLAN)
    assert exc_info.value.code == code
    assert exc_info.value.consume_producer_retry is True


def test_local_callable_shadow_cannot_impersonate_declared_import_route():
    source = (
        "import json\n"
        "def train_model(model, targets):\n"
        "    return {'training_history': fake}\n"
        "training_result = train_model(model, targets)\n"
        "training_history = training_result['training_history']\n"
        + _print("training_history")
    )
    with pytest.raises(TrainingHistoryProducerError) as exc_info:
        validate_training_history_notebook_flow([source], PLAN)
    assert exc_info.value.code == (
        "training_history_notebook_training_route_disagreement"
    )


@pytest.mark.parametrize(
    "mutation",
    [
        (
            "training_history['loss_observations'] = fake_losses\n"
            "training_history['record_digest'] = "
            "recompute_digest(training_history)\n"
        ),
        "training_history.update(fake_history)\n",
        "if use_fake:\n    training_history = fake_history\n",
        (
            "history_alias = training_history\n"
            "history_alias['record_digest'] = fake_digest\n"
        ),
    ],
)
def test_result_and_history_alias_mutation_or_rebinding_is_rejected(mutation):
    with pytest.raises(TrainingHistoryProducerError) as exc_info:
        validate_training_history_notebook_flow([_good(mutation)], PLAN)
    assert exc_info.value.code in {
        "training_history_notebook_history_mutated",
        "training_history_notebook_result_alias_rebound",
    }


@pytest.mark.parametrize(
    "escape",
    [
        "box = [training_history]\nbox[0]['record_digest'] = fake_digest\n",
        (
            "box = {'history': training_history}\n"
            "box['history']['record_digest'] = fake_digest\n"
        ),
    ],
)
def test_container_alias_escape_cannot_hide_history_mutation(escape):
    with pytest.raises(TrainingHistoryCoverageError) as exc_info:
        validate_training_history_notebook_flow([_good(escape)], PLAN)
    assert exc_info.value.code == (
        "training_history_notebook_alias_escape_unsupported"
    )
    assert exc_info.value.consume_producer_retry is False


def test_interprocedural_flow_is_pipeline_coverage_without_retry():
    source = (
        "import json\n"
        "from method.training import train_model\n"
        "def fit():\n"
        "    return train_model(model, targets)\n"
        "training_result = fit()\n"
        "training_history = training_result['training_history']\n"
        + _print("training_history")
    )
    with pytest.raises(TrainingHistoryCoverageError) as exc_info:
        validate_training_history_notebook_flow([source], PLAN)
    assert exc_info.value.code == (
        "training_history_notebook_training_grammar_unsupported"
    )
    assert exc_info.value.consume_producer_retry is False


@pytest.mark.parametrize(
    "mutation",
    [
        "setattr(training, 'train_model', fake_train_model)\n",
        "training.__dict__.update({'train_model': fake_train_model})\n",
    ],
)
def test_declared_training_module_dynamic_mutation_is_producer_owned(mutation):
    source = (
        "import json\n"
        "import method.training as training\n"
        + mutation
        + "training_result = training.train_model(model, targets)\n"
        "training_history = training_result['training_history']\n"
        + _print("training_history")
    )
    with pytest.raises(TrainingHistoryProducerError) as exc_info:
        validate_training_history_notebook_flow([source], PLAN)
    assert exc_info.value.code == (
        "training_history_notebook_protected_binding_mutated"
    )


@pytest.mark.parametrize("name", ["train_model", "print"])
def test_globals_cannot_replace_protected_bindings(name):
    source = (
        "import json\n"
        "from method.training import train_model\n"
        f"globals()[{name!r}] = fake\n"
        "training_result = train_model(model, targets)\n"
        "training_history = training_result['training_history']\n"
        + _print("training_history")
    )
    with pytest.raises(TrainingHistoryProducerError) as exc_info:
        validate_training_history_notebook_flow([source], PLAN)
    assert exc_info.value.code == (
        "training_history_notebook_protected_binding_rebound"
    )


@pytest.mark.parametrize(
    "dynamic",
    [
        "exec(\"train_model = fake\")\n",
        "fit = train_model\n",
        "j = json\n",
        (
            "import method.training as training\n"
            "fit = getattr(training, 'train_model')\n"
        ),
    ],
)
def test_indirect_binding_or_dynamic_namespace_is_pipeline_coverage(dynamic):
    import_line = (
        "" if "import method.training as training" in dynamic
        else "from method.training import train_model\n"
    )
    call = "fit" if "fit =" in dynamic else "train_model"
    source = (
        "import json\n"
        + import_line
        + dynamic
        + f"training_result = {call}(model, targets)\n"
        "training_history = training_result['training_history']\n"
        + _print("training_history")
    )
    with pytest.raises(TrainingHistoryCoverageError) as exc_info:
        validate_training_history_notebook_flow([source], PLAN)
    assert exc_info.value.consume_producer_retry is False


def test_stdlib_json_serializer_cannot_be_mutated_via_setattr():
    source = _good("setattr(json, 'dumps', fake_dumps)\n")
    with pytest.raises(TrainingHistoryProducerError) as exc_info:
        validate_training_history_notebook_flow([source], PLAN)
    assert exc_info.value.code == (
        "training_history_notebook_protected_binding_mutated"
    )


@pytest.mark.parametrize("escaped", ["training_history", "training_result"])
def test_for_alias_of_result_or_history_is_pipeline_coverage(escaped):
    source = _good(f"for alias in [{escaped}]:\n    pass\n")
    with pytest.raises(TrainingHistoryCoverageError) as exc_info:
        validate_training_history_notebook_flow([source], PLAN)
    assert exc_info.value.code == (
        "training_history_notebook_alias_escape_unsupported"
    )


@pytest.mark.parametrize("wrapper", ["if enabled:", "for _ in [0]:"])
def test_marker_must_be_one_direct_top_level_statement(wrapper):
    marker = "\n".join("    " + line for line in _print(
        "training_history"
    ).splitlines()) + "\n"
    source = (
        "import json\n"
        "from method.training import train_model\n"
        "training_result = train_model(model, targets)\n"
        "training_history = training_result['training_history']\n"
        + wrapper + "\n" + marker
    )
    with pytest.raises(TrainingHistoryCoverageError) as exc_info:
        validate_training_history_notebook_flow([source], PLAN)
    assert exc_info.value.code == (
        "training_history_notebook_marker_grammar_unsupported"
    )


def test_dormant_real_marker_cannot_authorize_dynamic_fake_output_route():
    dormant = "\n".join("    " + line for line in _print(
        "training_history"
    ).splitlines()) + "\n"
    source = (
        "import json\n"
        "from method.training import train_model\n"
        "training_result = train_model(model, targets)\n"
        "training_history = training_result['training_history']\n"
        "if False:\n"
        + dormant
        + "fake_marker = 'R2C_' + 'TRAINING_HISTORY_JSON: '\n"
        + _print("{'status': 'recorded'}").replace(
            repr(TRAINING_HISTORY_MARKER), "fake_marker"
        )
    )
    with pytest.raises(TrainingHistoryCoverageError) as exc_info:
        validate_training_history_notebook_flow([source], PLAN)
    assert exc_info.value.consume_producer_retry is False


def test_schema2_plan_arm_is_applicable_even_when_required_loop_is_missing(
    tmp_path,
):
    _write_contract(tmp_path, {"schema_version": "2.0.0"})
    assert training_history_notebook_applicable(tmp_path, PLAN) is True
    with pytest.raises(TrainingHistoryProducerError) as exc_info:
        validate_training_history_architecture_contract(tmp_path, PLAN)
    assert exc_info.value.code == "training_history_required_training_loop_missing"


def test_only_schema1_or_plan_without_history_arm_is_not_applicable(tmp_path):
    _write_contract(tmp_path, {
        "schema_version": "1.0.0",
        "training_loop": {"function_name": "train_model"},
    })
    assert training_history_notebook_applicable(tmp_path, PLAN) is False

    _write_contract(tmp_path, {
        "schema_version": "2.0.0",
        "training_loop": {"function_name": "train_model"},
    })
    assert training_history_notebook_applicable(tmp_path, {}) is False
