"""Producer and downstream prompt activation for architecture contract v2."""

from pathlib import Path

from scripts.dispatch_templates import STAGE_TASK_SUMMARIES


ROOT = Path(__file__).resolve().parents[1]


def _agent(name: str) -> str:
    return (ROOT / ".opencode" / "agents" / name).read_text(encoding="utf-8")


def _flat(text: str) -> str:
    """Whitespace-insensitive form: the agent markdown hard-wraps its
    paragraphs, so asserting on literal embedded newlines turns red on a
    pure reflow (same defense as test_nonfinite_metric_smoke_check)."""
    return " ".join(text.split())


def test_architecture_coder_emits_typed_v2_without_invented_authority():
    prompt = _agent("r2c-architecture-coder.md")

    assert '"schema_version": "2.0.0"' in prompt
    assert "stable semantic identities" in prompt
    assert '"kind": "exact_divide"' in prompt
    assert "exactly matches its callable parameter name" in prompt
    assert "schema 2.0 has no typed dict grammar" in _flat(prompt)
    assert "feature widths have no current bundle measurement" in _flat(prompt)
    assert "does not author evaluation protocol or partition truth" in _flat(prompt)
    for identity in (
        "training_example_count",
        "forward_batch_count",
        "pool_example_count",
        "selection_count",
    ):
        assert f'"{identity}"' in prompt
    assert '"indexed_dimension": "pool_example_count"' in prompt
    assert "training-loop output may be `null`" in _flat(prompt)
    assert "`schema_version` is `1.1.0`" not in prompt


def test_method_and_notebook_branch_on_version_and_limit_opaque_claims():
    for name in ("r2c-method-coder.md", "r2c-notebook-generator.md"):
        prompt = _agent(name)
        assert "Read `schema_version` first" in prompt
        assert "pluggable_component.input/.output" in prompt
        assert "archived" in prompt and "resumed" in prompt
        assert "opaque" in prompt
        assert (
            "no structured-dict grammar" in prompt
            or "no KD structured-dict grammar" in prompt
        )
    method = _agent("r2c-method-coder.md")
    assert "certifies no runtime container" in method
    assert "certifies only its `type_description`" not in _flat(method)


def test_dispatch_summaries_carry_the_v2_handoff_boundary():
    architecture = STAGE_TASK_SUMMARIES["stage_2b_architecture"]
    method = STAGE_TASK_SUMMARIES["stage_2c_method"]
    notebook = STAGE_TASK_SUMMARIES["stage_3a_notebook"]

    assert "schema_version 2.0.0" in architecture
    assert "exact callable parameter names" in architecture
    assert "v1 shape fields are only for archived/resumed" in method
    assert "current v2 handoffs use dimensions" in notebook
    assert "opaque" in architecture and "opaque" in method and "opaque" in notebook
