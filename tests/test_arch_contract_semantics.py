"""R2C-085 dark foundation: semantic dimensions and typed values."""

from __future__ import annotations

import copy

import pytest
from jsonschema import Draft202012Validator
from pydantic import ValidationError

from schemas.arch_contract import ArchContract
from schemas.arch_contract_v2 import ArchContractV2
from scripts.arch_contract_semantics import (
    BundleFact,
    ContractSemanticResolution,
    ResolvedDimension,
    ResolvedFixtureSpec,
    SemanticIssue,
    bundle_dimension_facts,
    format_semantic_issue,
    iter_contract_descriptors,
    load_arch_contract,
    parse_semantic_issue,
    resolve_contract,
    resolve_constructor_value,
    resolve_descriptor,
    resolve_dimensions,
    semantic_issue_exit_code,
)


def _lit(value: int) -> dict:
    return {"kind": "literal", "value": value}


def _ref(identity: str) -> dict:
    return {"kind": "reference", "dimension": identity}


def _opaque(label: str = "fixture") -> dict:
    return {
        "kind": "opaque",
        "type_description": label,
        "reason": "Not executable in the bounded typed fixture.",
    }


def _contract(dimensions: dict | None = None) -> dict:
    return {
        "schema_version": "2.0.0",
        "paradigm_id": "typed_test",
        "dimensions": dimensions or {},
        "data_loader": {"load_data_returns": {}},
        "architecture": {
            "model": {
                "class_name": "Net",
                "constructor_args": {},
                "forward": {"input": {}, "output": _opaque("model output")},
            }
        },
        "pluggable_component": {
            "name": "contribution",
            "input": {},
            "output": _opaque("contribution output"),
        },
    }


def _legacy(version: str | None) -> dict:
    raw = {
        "paradigm_id": "legacy_test",
        "data_loader": {"load_data_returns": {}},
        "architecture": {
            "model": {
                "class_name": "Net",
                "forward": {
                    "input": {"x": "(B, n_features)"},
                    "output_type": "tensor",
                    "output_shape": "(B, n_features)",
                },
            }
        },
        "pluggable_component": {
            "name": "contribution",
            "input_shapes": {},
            "output_shape": "scalar",
        },
    }
    if version is not None:
        raw["schema_version"] = version
    if version == "1.1.0":
        raw["architecture"]["model"]["constructor_args"] = {}
    return raw


def test_versioned_loader_preserves_v1_v11_and_reads_v2():
    assert isinstance(load_arch_contract(_legacy(None)), ArchContract)
    assert load_arch_contract(_legacy(None)).schema_version == "1.0.0"
    assert load_arch_contract(_legacy("1.1.0")).schema_version == "1.1.0"
    assert isinstance(load_arch_contract(_contract()), ArchContractV2)
    missing_version = _contract()
    missing_version.pop("schema_version")
    with pytest.raises(ValidationError):
        ArchContractV2.model_validate(missing_version)
    with pytest.raises(ValueError, match="unsupported arch_contract schema_version"):
        load_arch_contract({"schema_version": "9.0.0"})
    for malformed in ([], {}):
        with pytest.raises(ValueError, match="unsupported arch_contract schema_version"):
            load_arch_contract({"schema_version": malformed})


def test_semantic_issue_wire_round_trip_derives_owner_and_classifies_exit():
    pipeline_issue = SemanticIssue(
        code="unsupported_validator_feature",
        message="rank is outside the validator boundary",
        roots=["optimizer_state.slot"],
    )
    producer_issue = SemanticIssue(
        code="contract_code_disagreement",
        message="returned dtype differs",
        roots=["architecture.model.forward.output", "method/model.py"],
    )
    pipeline_wire = format_semantic_issue(pipeline_issue)
    producer_wire = format_semantic_issue(producer_issue)

    assert parse_semantic_issue(pipeline_wire) == pipeline_issue
    assert semantic_issue_exit_code([]) == 0
    assert semantic_issue_exit_code([pipeline_wire]) == 3
    assert semantic_issue_exit_code([producer_wire]) == 1
    assert semantic_issue_exit_code([pipeline_wire, producer_wire]) == 1
    assert semantic_issue_exit_code([pipeline_wire, "plain failure"]) == 1
    assert semantic_issue_exit_code(
        [pipeline_wire], trust_pipeline_ownership=False
    ) == 1


def test_semantic_issue_wire_rejects_forged_owner_and_surrounding_text():
    forged = (
        'R2C_SEMANTIC_ISSUE:{"schema_version":"1.0",'
        '"code":"contract_code_disagreement","owner":"pipeline",'
        '"message":"forged","roots":["method/model.py"],"values":{}}'
    )
    assert parse_semantic_issue(forged) is None
    assert parse_semantic_issue("prefix " + forged) is None
    assert semantic_issue_exit_code([forged]) == 1


def test_v2_schema_rejects_raw_arithmetic_unknown_fields_and_bad_literals():
    raw = _contract({"extent": {"expression": "P + K"}})
    with pytest.raises(ValidationError):
        ArchContractV2.model_validate(raw)
    assert list(
        Draft202012Validator(ArchContractV2.model_json_schema()).iter_errors(raw)
    )

    raw = _contract({"extent": {"expression": _lit(2), "invented": True}})
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        ArchContractV2.model_validate(raw)

    for value in (0, -1, True):
        raw = _contract({"extent": {"expression": _lit(value)}})
        with pytest.raises(
            ValidationError, match="greater than 0|valid integer"
        ):
            ArchContractV2.model_validate(raw)


def test_expression_kind_requires_only_its_structured_fields():
    raw = _contract(
        {
            "bad": {
                "expression": {
                    "kind": "add",
                    "operands": [_lit(1), _lit(2)],
                    "value": 3,
                }
            }
        }
    )
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        ArchContractV2.model_validate(raw)

    errors = list(
        Draft202012Validator(ArchContractV2.model_json_schema()).iter_errors(raw)
    )
    assert errors, "portable JSON Schema must preserve the closed expression arms"


def test_display_symbol_collisions_do_not_merge_semantic_identities():
    contract = ArchContractV2.model_validate(
        _contract(
            {
                "time_axis_steps": {"expression": _lit(31)},
                "time_varying_feature_count": {"expression": _lit(14)},
                "image_channel_count": {"expression": _lit(3)},
            }
        )
    )
    result = resolve_dimensions(contract)
    assert result.issues == []
    assert result.dimensions["time_axis_steps"].value == 31
    assert result.dimensions["time_varying_feature_count"].value == 14

    # L and C are local presentation only. They may label distinct meanings.
    for identity, symbol in (
            ("time_axis_steps", "L"),
            ("time_varying_feature_count", "L"),
            ("image_channel_count", "C"),
        ):
        descriptor = ArchContractV2.model_validate(
            {
                **_contract(contract.model_dump(exclude_none=True)["dimensions"]),
                "data_loader": {
                    "load_data_returns": {
                        "x": {
                            "kind": "tensor",
                            "dtype": "float32",
                            "dimensions": [
                                {"dimension": identity, "display_symbol": symbol}
                            ],
                        }
                    }
                },
            }
        ).data_loader.load_data_returns["x"]
        resolved = resolve_descriptor(
            descriptor, result.dimensions, root=f"data_loader.{identity}"
        )
        assert resolved.shape == (result.dimensions[identity].value,)

    fedavg = _contract()
    fedavg["data_loader"]["load_data_returns"] = {
        "client_sampling_fraction": {
            "kind": "scalar",
            "dtype": "float32",
            "source": {"literal": 0.25, "display_symbol": "C"},
        }
    }
    parsed_fedavg = ArchContractV2.model_validate(fedavg)
    fraction = parsed_fedavg.data_loader.load_data_returns[
        "client_sampling_fraction"
    ]
    assert fraction.source.display_symbol == "C"
    assert resolve_descriptor(fraction, {}, root="fedavg.C").shape == ()
    assert list(
        Draft202012Validator(ArchContractV2.model_json_schema()).iter_errors(fedavg)
    ) == []

    fractional_dimension = _contract(
        {"client_sampling_fraction": {"expression": _lit(0.25)}}
    )
    with pytest.raises(ValidationError):
        ArchContractV2.model_validate(fractional_dimension)


def test_student_teacher_may_reuse_one_local_symbol_without_sharing_value():
    contract = ArchContractV2.model_validate(
        _contract(
            {
                "student_hidden_width": {"expression": _lit(8)},
                "teacher_hidden_width": {"expression": _lit(16)},
            }
        )
    )
    result = resolve_dimensions(contract)
    for identity, expected in (
        ("student_hidden_width", 8),
        ("teacher_hidden_width", 16),
    ):
        raw = {
            "kind": "tensor",
            "dtype": "float32",
            "dimensions": [{"dimension": identity, "display_symbol": "D"}],
        }
        descriptor = contract.data_loader.__class__.model_validate(
            {"load_data_returns": {"x": raw}}
        ).load_data_returns["x"]
        assert resolve_descriptor(
            descriptor, result.dimensions, root=identity
        ).shape == (expected,)


def test_supported_derived_arithmetic_resolves_exactly():
    contract = ArchContractV2.model_validate(
        _contract(
            {
                "P": {"expression": _lit(10)},
                "K": {"expression": _lit(4)},
                "n_classes": {"expression": _lit(6)},
                "H": {"expression": _lit(8)},
                "W": {"expression": _lit(6)},
                "num_groups": {"expression": _lit(4)},
                "n_obs": {"expression": _lit(5)},
                "context_plus_horizon": {
                    "expression": {"kind": "add", "operands": [_ref("P"), _ref("K")]}
                },
                "logit_count": {
                    "expression": {"kind": "add", "operands": [_ref("n_classes"), _lit(1)]}
                },
                "token_count": {
                    "expression": {"kind": "multiply", "operands": [_ref("H"), _ref("W")]}
                },
                "grouped_tokens": {
                    "expression": {
                        "kind": "exact_divide",
                        "numerator": _ref("token_count"),
                        "divisor": _ref("num_groups"),
                    }
                },
                "joint_state_width": {
                    "expression": {
                        "kind": "add",
                        "operands": [
                            _lit(6),
                            {"kind": "multiply", "operands": [_ref("n_obs"), _lit(8)]},
                        ],
                    }
                },
            }
        )
    )
    result = resolve_dimensions(contract)
    assert result.issues == []
    assert {name: result.dimensions[name].value for name in (
        "context_plus_horizon", "logit_count", "token_count",
        "grouped_tokens", "joint_state_width",
    )} == {
        "context_plus_horizon": 14,
        "logit_count": 7,
        "token_count": 48,
        "grouped_tokens": 12,
        "joint_state_width": 46,
    }


def test_dangling_cycle_and_nonexact_division_fail_without_fallback():
    dangling = ArchContractV2.model_validate(
        _contract({"x": {"expression": _ref("missing")}})
    )
    result = resolve_dimensions(dangling)
    assert "x" not in result.dimensions
    dangling_issue = next(
        issue for issue in result.issues if "absent from the registry" in issue.message
    )
    assert dangling_issue.roots == ["dimensions.x.expression", "dimensions.missing"]

    repeated = ArchContractV2.model_validate(
        _contract(
            {
                "first": {"expression": _ref("missing")},
                "second": {"expression": _ref("missing")},
            }
        )
    )
    repeated_result = resolve_dimensions(repeated)
    assert {tuple(issue.roots) for issue in repeated_result.issues} == {
        ("dimensions.first.expression", "dimensions.missing"),
        ("dimensions.second.expression", "dimensions.missing"),
    }

    cyclic = ArchContractV2.model_validate(
        _contract(
            {
                "a": {"expression": _ref("b")},
                "b": {"expression": _ref("a")},
            }
        )
    )
    result = resolve_dimensions(cyclic)
    assert result.dimensions == {}
    assert any("cycle" in issue.message for issue in result.issues)

    remainder = ArchContractV2.model_validate(
        _contract(
            {
                "ten": {"expression": _lit(10)},
                "three": {"expression": _lit(3)},
                "bad": {
                    "expression": {
                        "kind": "exact_divide",
                        "numerator": _ref("ten"),
                        "divisor": _ref("three"),
                    }
                },
            }
        )
    )
    result = resolve_dimensions(remainder)
    assert "bad" not in result.dimensions
    assert any(issue.values.get("remainder") == 1 for issue in result.issues)


def test_bundle_override_requires_the_same_semantic_identity():
    contract = ArchContractV2.model_validate(
        _contract(
            {
                "time_axis_steps": {
                    "expression": _lit(12),
                    "bundle_binding": {
                        "policy": "override_fixture",
                    },
                },
                "time_varying_feature_count": {"expression": _lit(14)},
            }
        )
    )
    facts = {
        "time_axis_steps": BundleFact(
            value=1092,
            root="PROVENANCE.json#files[0].time_axis.steps_kept",
        )
    }
    result = resolve_dimensions(contract, facts)
    assert result.issues == []
    assert result.dimensions["time_axis_steps"].value == 1092
    assert result.dimensions["time_axis_steps"].source == "bundle"
    assert result.dimensions["time_varying_feature_count"].value == 14

    wrong_identity = _contract(
        {
            "time_varying_feature_count": {
                "expression": _lit(14),
                "bundle_binding": {"policy": "override_fixture"},
            }
        }
    )
    wrong_contract = ArchContractV2.model_validate(wrong_identity)
    wrong_result = resolve_dimensions(wrong_contract, facts)
    assert "time_varying_feature_count" not in wrong_result.dimensions
    assert "requires bundle fact 'time_varying_feature_count'" in (
        wrong_result.issues[0].message
    )

    explicit_cross_binding = copy.deepcopy(wrong_identity)
    explicit_cross_binding["dimensions"]["time_varying_feature_count"][
        "bundle_binding"
    ]["fact"] = "time_axis_steps"
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        ArchContractV2.model_validate(explicit_cross_binding)
    assert list(
        Draft202012Validator(ArchContractV2.model_json_schema()).iter_errors(
            explicit_cross_binding
        )
    )


def test_bundle_binding_absence_and_must_match_disagreement_are_explicit():
    raw = _contract(
        {
            "time_axis_steps": {
                "expression": _lit(12),
                "bundle_binding": {
                    "policy": "must_match",
                },
            }
        }
    )
    contract = ArchContractV2.model_validate(raw)
    absent = resolve_dimensions(contract)
    assert absent.issues[0].code == "incomplete_generated_contract"

    mismatch = resolve_dimensions(
        contract,
        {
            "time_axis_steps": BundleFact(
                value=13, root="PROVENANCE.json#files[0].time_axis.steps_kept"
            )
        },
    )
    assert mismatch.issues[0].code == "bundle_contract_disagreement"
    assert mismatch.issues[0].values == {"declared": 12, "measured": 13}

    matched = resolve_dimensions(
        contract,
        {
            "time_axis_steps": BundleFact(
                value=12, root="PROVENANCE.json#files[0].time_axis.steps_kept"
            )
        },
    )
    assert matched.issues == []
    assert matched.dimensions["time_axis_steps"].evidence_roots == [
        "dimensions.time_axis_steps.expression",
        "dimensions.time_axis_steps.bundle_binding",
        "PROVENANCE.json#files[0].time_axis.steps_kept",
    ]


def test_bundle_facts_do_not_invent_feature_widths():
    facts = bundle_dimension_facts(
        {
            "files": [
                {
                    "file": "series.csv",
                    "columns": ["a", "b", "c"],
                    "time_axis": {"steps_kept": 31},
                }
            ]
        }
    )
    assert set(facts) == {"time_axis_steps"}
    assert facts["time_axis_steps"].value == 31
    with pytest.raises(ValidationError, match="valid integer"):
        BundleFact(value=True, root="PROVENANCE.json#files[0]")
    for bad_root in (" ", " evidence "):
        with pytest.raises(ValidationError, match="non-blank and stripped"):
            BundleFact(value=3, root=bad_root)
    with pytest.raises(ValidationError, match="non-blank and stripped"):
        ResolvedDimension(
            value=3,
            source="declared",
            evidence_roots=[" dimensions.x.expression "],
        )


def test_typed_descriptors_enforce_index_class_and_mask_dtypes():
    contract = ArchContractV2.model_validate(
        _contract(
            {
                "edge_count": {"expression": _lit(4)},
                "node_count": {"expression": _lit(5)},
                "class_count": {"expression": _lit(3)},
            }
        )
    )
    resolved = resolve_dimensions(contract).dimensions
    cases = [
        (
            {"kind": "index", "indexed_dimension": "node_count"},
            "float32",
            "int32 or int64",
        ),
        (
            {"kind": "class_id", "class_count_dimension": "class_count"},
            "float64",
            "int32 or int64",
        ),
        ({"kind": "mask"}, "int64", "bool dtype"),
    ]
    for constraint, dtype, message in cases:
        raw = copy.deepcopy(
            _contract(contract.model_dump(exclude_none=True)["dimensions"])
        )
        raw["data_loader"]["load_data_returns"] = {
            "value": {
                "kind": "tensor",
                "dtype": dtype,
                "dimensions": [{"dimension": "edge_count"}],
                "constraint": constraint,
            }
        }
        descriptor = ArchContractV2.model_validate(raw).data_loader.load_data_returns["value"]
        result = resolve_descriptor(descriptor, resolved, root="data_loader.value")
        assert any(message in issue.message for issue in result.issues)

    good_raw = copy.deepcopy(
        _contract(contract.model_dump(exclude_none=True)["dimensions"])
    )
    good_raw["data_loader"]["load_data_returns"] = {
        "edges": {
            "kind": "tensor",
            "dtype": "int64",
            "dimensions": [{"dimension": "edge_count"}],
            "constraint": {"kind": "index", "indexed_dimension": "node_count"},
        },
        "labels": {
            "kind": "ndarray",
            "dtype": "int32",
            "dimensions": [{"dimension": "node_count"}],
            "constraint": {"kind": "class_id", "class_count_dimension": "class_count"},
        },
        "mask": {
            "kind": "tensor",
            "dtype": "bool",
            "dimensions": [{"dimension": "node_count"}],
            "constraint": {"kind": "mask"},
        },
    }
    parsed = ArchContractV2.model_validate(good_raw)
    for root, descriptor in parsed.data_loader.load_data_returns.items():
        assert resolve_descriptor(descriptor, resolved, root=root).issues == []


def test_mixed_container_scalar_and_opaque_descriptors_remain_distinct():
    raw = _contract({"rows": {"expression": _lit(4)}})
    raw["data_loader"]["load_data_returns"] = {
        "tensor": {
            "kind": "tensor",
            "dtype": "float32",
            "dimensions": [{"dimension": "rows"}],
        },
        "array": {
            "kind": "ndarray",
            "dtype": "float64",
            "dimensions": [{"dimension": "rows"}],
        },
        "seed": {
            "kind": "scalar",
            "dtype": "int64",
            "source": {"literal": 0},
        },
        "records": _opaque("list of detection records"),
    }
    contract = ArchContractV2.model_validate(raw)
    resolved = resolve_dimensions(contract).dimensions
    assert contract.data_loader.load_data_returns["tensor"].kind == "tensor"
    assert contract.data_loader.load_data_returns["array"].kind == "ndarray"
    assert resolve_descriptor(
        contract.data_loader.load_data_returns["seed"], resolved, root="seed"
    ).shape == ()
    assert resolve_descriptor(
        contract.data_loader.load_data_returns["records"], resolved, root="records"
    ).shape is None


def test_optimizer_and_family_extension_values_are_typed():
    raw = _contract({"parameter_count": {"expression": _lit(4)}})
    raw["optimizer_state"] = {
        "step": {
            "kind": "scalar",
            "dtype": "int64",
            "source": {"literal": 0},
        },
        "first_moment": {
            "kind": "tensor",
            "dtype": "float32",
            "dimensions": [{"dimension": "parameter_count"}],
        },
    }
    raw["family_components"] = {
        "reward_function": {"value": _opaque("reward callable")},
        "named_state": {
            "entries": {
                "active": {
                    "kind": "scalar",
                    "dtype": "bool",
                    "source": {"literal": True},
                }
            }
        },
    }
    contract = ArchContractV2.model_validate(raw)
    assert contract.optimizer_state["step"].dtype == "int64"
    assert contract.family_components["reward_function"].value.kind == "opaque"
    assert contract.family_components["named_state"].entries["active"].dtype == "bool"

    raw["family_components"]["invalid"] = {
        "value": _opaque("single"),
        "entries": {"also": _opaque("multiple")},
    }
    with pytest.raises(ValidationError):
        ArchContractV2.model_validate(raw)
    portable_errors = list(
        Draft202012Validator(ArchContractV2.model_json_schema()).iter_errors(raw)
    )
    assert portable_errors


@pytest.mark.parametrize(
    ("dtype", "literal"),
    [
        ("int64", "text"),
        ("bool", 5),
        ("float32", {"not": "a scalar"}),
        ("float64", None),
        ("int32", 2**31),
        ("float32", 1e40),
        ("float64", 10**1000),
    ],
)
def test_scalar_literals_must_match_their_declared_dtype(dtype, literal):
    raw = _contract()
    raw["data_loader"]["load_data_returns"] = {
        "bad": {
            "kind": "scalar",
            "dtype": dtype,
            "source": {"literal": literal},
        }
    }
    with pytest.raises(ValidationError):
        ArchContractV2.model_validate(raw)
    assert list(
        Draft202012Validator(ArchContractV2.model_json_schema()).iter_errors(raw)
    )

    dimension_backed = _contract({"count": {"expression": _lit(3)}})
    dimension_backed["data_loader"]["load_data_returns"] = {
        "bad": {
            "kind": "scalar",
            "dtype": "float32",
            "source": {"dimension": {"dimension": "count"}},
        }
    }
    with pytest.raises(ValidationError, match="dimension-backed scalar"):
        ArchContractV2.model_validate(dimension_backed)
    assert list(
        Draft202012Validator(ArchContractV2.model_json_schema()).iter_errors(
            dimension_backed
        )
    )


def test_scalar_source_is_portably_exclusive_and_dimension_range_is_checked():
    base = _contract()
    for source in ({}, {"literal": 1, "dimension": {"dimension": "count"}}):
        raw = copy.deepcopy(base)
        raw["data_loader"]["load_data_returns"] = {
            "bad": {"kind": "scalar", "dtype": "int32", "source": source}
        }
        with pytest.raises(ValidationError):
            ArchContractV2.model_validate(raw)
        assert list(
            Draft202012Validator(ArchContractV2.model_json_schema()).iter_errors(raw)
        )

    large = _contract({"count": {"expression": _lit(2**31)}})
    large["data_loader"]["load_data_returns"] = {
        "count": {
            "kind": "scalar",
            "dtype": "int32",
            "source": {"dimension": {"dimension": "count"}},
        }
    }
    contract = ArchContractV2.model_validate(large)
    result = resolve_descriptor(
        contract.data_loader.load_data_returns["count"],
        resolve_dimensions(contract).dimensions,
        root="data_loader.count",
    )
    assert result.shape == ()
    assert result.issues[0].code == "incomplete_generated_contract"
    assert "outside int32 range" in result.issues[0].message


def test_allocation_cap_is_pipeline_owned_and_issue_owner_cannot_be_forged():
    raw = _contract({"rows": {"expression": _lit(101)}, "cols": {"expression": _lit(101)}})
    raw["data_loader"]["load_data_returns"] = {
        "x": {
            "kind": "tensor",
            "dtype": "float32",
            "dimensions": [{"dimension": "rows"}, {"dimension": "cols"}],
        }
    }
    contract = ArchContractV2.model_validate(raw)
    resolved = resolve_dimensions(contract).dimensions
    result = resolve_descriptor(
        contract.data_loader.load_data_returns["x"],
        resolved,
        root="data_loader.x",
        max_elements=10_000,
    )
    assert result.issues[0].code == "unsupported_validator_feature"
    assert result.issues[0].owner == "pipeline"

    with pytest.raises(ValidationError, match="owned by 'pipeline'"):
        SemanticIssue(
            code="unsupported_validator_feature",
            owner="producer",
            message="forged",
            roots=["data_loader.x"],
        )


@pytest.mark.parametrize(
    ("code", "owner"),
    [
        ("incomplete_generated_contract", "producer"),
        ("unsupported_validator_feature", "pipeline"),
        ("contract_code_disagreement", "producer"),
        ("bundle_contract_disagreement", "producer"),
    ],
)
def test_semantic_issue_owner_is_derived_from_closed_code(code, owner):
    issue = SemanticIssue(code=code, message="controlled", roots=["root"])
    assert issue.owner == owner


def test_v2_constructor_args_preserve_exact_keyword_names():
    raw = _contract()
    raw["architecture"]["model"]["constructor_args"] = {
        "not-a-keyword": {"literal": 3}
    }
    with pytest.raises(ValidationError, match="exact Python keyword names"):
        ArchContractV2.model_validate(raw)


def test_constructor_values_have_one_portable_source_and_resolve_exactly():
    raw = _contract({"hidden_width": {"expression": _lit(17)}})
    raw["architecture"]["model"]["constructor_args"] = {
        "width": {"dimension": {"dimension": "hidden_width"}},
        "activation": {"literal": "relu"},
        "nullable": {"literal": None},
    }
    contract = ArchContractV2.model_validate(raw)
    dimensions = resolve_dimensions(contract).dimensions
    width, issues = resolve_constructor_value(
        contract.architecture["model"].constructor_args["width"],
        dimensions,
        root="architecture.model.constructor_args.width",
    )
    assert (width, issues) == (17, [])
    activation, issues = resolve_constructor_value(
        contract.architecture["model"].constructor_args["activation"],
        dimensions,
        root="architecture.model.constructor_args.activation",
    )
    assert (activation, issues) == ("relu", [])

    for invalid in ({}, {"literal": 3, "dimension": {"dimension": "hidden_width"}}):
        candidate = copy.deepcopy(raw)
        candidate["architecture"]["model"]["constructor_args"]["bad"] = invalid
        with pytest.raises(ValidationError):
            ArchContractV2.model_validate(candidate)
        assert list(
            Draft202012Validator(ArchContractV2.model_json_schema()).iter_errors(
                candidate
            )
        )


def test_contract_resolution_traverses_every_typed_surface_with_stable_roots():
    dimensions = {
        "batch_count": {"expression": _lit(2)},
        "node_count": {"expression": _lit(5)},
        "edge_count": {"expression": _lit(4)},
        "class_count": {"expression": _lit(3)},
        "feature_width": {"expression": _lit(6)},
        "endpoint_count": {"expression": _lit(2)},
    }
    tensor = {
        "kind": "tensor",
        "dtype": "float32",
        "dimensions": [
            {"dimension": "batch_count"},
            {"dimension": "feature_width"},
        ],
    }
    ndarray = {
        "kind": "ndarray",
        "dtype": "float64",
        "dimensions": [
            {"dimension": "batch_count"},
            {"dimension": "feature_width"},
        ],
    }
    labels = {
        "kind": "tensor",
        "dtype": "int64",
        "dimensions": [{"dimension": "batch_count"}],
        "constraint": {
            "kind": "class_id",
            "class_count_dimension": "class_count",
        },
    }
    mask = {
        "kind": "tensor",
        "dtype": "bool",
        "dimensions": [{"dimension": "batch_count"}],
        "constraint": {"kind": "mask"},
    }
    raw = _contract(dimensions)
    raw["data_loader"]["load_data_returns"] = {
        "features": tensor,
        "edge_index": {
            "kind": "tensor",
            "dtype": "int64",
            "dimensions": [
                {"dimension": "endpoint_count"},
                {"dimension": "edge_count"},
            ],
            "constraint": {
                "kind": "index",
                "indexed_dimension": "node_count",
            },
        },
    }
    raw["architecture"]["model"] = {
        "class_name": "Net",
        "constructor_args": {
            "width": {"dimension": {"dimension": "feature_width"}},
            "activation": {"literal": "relu"},
            "nullable": {"literal": None},
        },
        "forward": {
            "input": {"features": tensor, "mask": mask},
            "output": labels,
        },
        "additional_methods": {
            "inspect": {
                "input": {
                    "array": ndarray,
                    "seed": {
                        "kind": "scalar",
                        "dtype": "int64",
                        "source": {"literal": 0},
                    },
                },
                "output": {
                    "kind": "scalar",
                    "dtype": "int32",
                    "source": {
                        "dimension": {"dimension": "class_count"}
                    },
                },
            }
        },
    }
    raw["pluggable_component"] = {
        "name": "contribution",
        "input": {"candidate": tensor},
        "output": ndarray,
    }
    raw["training_loop"] = {
        "function_name": "train_model",
        "input": {"labels": labels},
        "output": {
            "kind": "scalar",
            "dtype": "bool",
            "source": {"literal": True},
        },
    }
    raw["optimizer_state"] = {
        "step": {
            "kind": "scalar",
            "dtype": "int64",
            "source": {"literal": 0},
        }
    }
    raw["family_components"] = {
        "reward": {"value": _opaque("reward callable")},
        "state": {"entries": {"active": mask}},
    }

    contract = ArchContractV2.model_validate(raw)
    assert [root for root, _ in iter_contract_descriptors(contract)] == [
        "data_loader.load_data_returns.features",
        "data_loader.load_data_returns.edge_index",
        "architecture.model.forward.input.features",
        "architecture.model.forward.input.mask",
        "architecture.model.forward.output",
        "architecture.model.additional_methods.inspect.input.array",
        "architecture.model.additional_methods.inspect.input.seed",
        "architecture.model.additional_methods.inspect.output",
        "pluggable_component.input.candidate",
        "pluggable_component.output",
        "training_loop.input.labels",
        "training_loop.output",
        "optimizer_state.step",
        "family_components.reward.value",
        "family_components.state.entries.active",
    ]

    result = resolve_contract(contract)
    assert isinstance(result, ContractSemanticResolution)
    assert result.issues == []
    assert result.constructor_args == {
        "model": {"width": 6, "activation": "relu", "nullable": None}
    }
    assert list(result.fixtures) == [
        root for root, _ in iter_contract_descriptors(contract)
    ]

    edge = result.fixtures["data_loader.load_data_returns.edge_index"]
    assert edge == ResolvedFixtureSpec(
        kind="tensor",
        dtype="int64",
        device="cpu",
        shape=(2, 4),
        synthesizable=True,
        constraint={"kind": "index", "indexed_dimension": "node_count"},
        constraint_upper_bound=5,
    )
    label = result.fixtures["architecture.model.forward.output"]
    assert label.constraint == {
        "kind": "class_id",
        "class_count_dimension": "class_count",
    }
    assert label.constraint_upper_bound == 3
    masked = result.fixtures["architecture.model.forward.input.mask"]
    assert masked.constraint == {"kind": "mask"}
    assert masked.constraint_upper_bound is None
    seed = result.fixtures[
        "architecture.model.additional_methods.inspect.input.seed"
    ]
    assert (seed.kind, seed.dtype, seed.shape, seed.scalar_value) == (
        "scalar",
        "int64",
        (),
        0,
    )
    assert result.fixtures[
        "architecture.model.additional_methods.inspect.output"
    ].scalar_value == 3
    opaque = result.fixtures["family_components.reward.value"]
    assert opaque.model_dump(exclude_none=True) == {
        "kind": "opaque",
        "opaque": True,
        "synthesizable": False,
    }


def test_contract_resolution_keeps_every_unresolved_consumer_root():
    missing_tensor = {
        "kind": "tensor",
        "dtype": "float32",
        "dimensions": [{"dimension": "missing"}],
    }
    raw = _contract()
    raw["data_loader"]["load_data_returns"] = {"bad": missing_tensor}
    raw["architecture"]["model"] = {
        "class_name": "Net",
        "constructor_args": {
            "width": {"dimension": {"dimension": "missing"}}
        },
        "forward": {
            "input": {"bad": missing_tensor},
            "output": missing_tensor,
        },
        "additional_methods": {
            "inspect": {
                "input": {"bad": missing_tensor},
                "output": missing_tensor,
            }
        },
    }
    raw["pluggable_component"] = {
        "name": "contribution",
        "input": {"bad": missing_tensor},
        "output": missing_tensor,
    }
    raw["training_loop"] = {
        "function_name": "train_model",
        "input": {"bad": missing_tensor},
        "output": missing_tensor,
    }
    raw["optimizer_state"] = {"bad": missing_tensor}
    raw["family_components"] = {
        "single": {"value": missing_tensor},
        "multi": {"entries": {"bad": missing_tensor}},
    }
    contract = ArchContractV2.model_validate(raw)

    result = resolve_contract(contract)
    consumer_roots = [
        "architecture.model.constructor_args.width",
        "data_loader.load_data_returns.bad.dimensions[0]",
        "architecture.model.forward.input.bad.dimensions[0]",
        "architecture.model.forward.output.dimensions[0]",
        "architecture.model.additional_methods.inspect.input.bad.dimensions[0]",
        "architecture.model.additional_methods.inspect.output.dimensions[0]",
        "pluggable_component.input.bad.dimensions[0]",
        "pluggable_component.output.dimensions[0]",
        "training_loop.input.bad.dimensions[0]",
        "training_loop.output.dimensions[0]",
        "optimizer_state.bad.dimensions[0]",
        "family_components.single.value.dimensions[0]",
        "family_components.multi.entries.bad.dimensions[0]",
    ]
    issue_roots = {root for issue in result.issues for root in issue.roots}
    assert set(consumer_roots) <= issue_roots
    assert all(issue.code == "incomplete_generated_contract" for issue in result.issues)
    assert result.constructor_args == {"model": {}}
    assert all(not fixture.synthesizable for fixture in result.fixtures.values())
    assert all(fixture.shape is None for fixture in result.fixtures.values())


def test_contract_resolution_surfaces_pipeline_owned_fixture_cap_at_exact_root():
    raw = _contract(
        {
            "rows": {"expression": _lit(101)},
            "cols": {"expression": _lit(101)},
        }
    )
    raw["optimizer_state"] = {
        "oversized": {
            "kind": "ndarray",
            "dtype": "float64",
            "dimensions": [
                {"dimension": "rows"},
                {"dimension": "cols"},
            ],
        }
    }
    result = resolve_contract(
        ArchContractV2.model_validate(raw),
        max_elements=10_000,
    )
    assert [(issue.code, issue.owner, issue.roots) for issue in result.issues] == [
        (
            "unsupported_validator_feature",
            "pipeline",
            ["optimizer_state.oversized"],
        )
    ]
    assert result.fixtures["optimizer_state.oversized"] == ResolvedFixtureSpec(
        kind="ndarray",
        dtype="float64",
        shape=None,
        synthesizable=False,
    )


def test_contract_resolution_surfaces_pipeline_owned_rank_cap_at_exact_root():
    dimensions = {
        f"axis_{index}": {"expression": _lit(1)}
        for index in range(33)
    }
    raw = _contract(dimensions)
    raw["optimizer_state"] = {
        "too_many_axes": {
            "kind": "ndarray",
            "dtype": "float64",
            "dimensions": [
                {"dimension": f"axis_{index}"} for index in range(33)
            ],
        }
    }

    result = resolve_contract(ArchContractV2.model_validate(raw))
    assert [(issue.code, issue.owner, issue.roots) for issue in result.issues] == [
        (
            "unsupported_validator_feature",
            "pipeline",
            ["optimizer_state.too_many_axes"],
        )
    ]
    assert result.issues[0].values == {"rank": 33, "cap": 32}
    assert not result.fixtures["optimizer_state.too_many_axes"].synthesizable
