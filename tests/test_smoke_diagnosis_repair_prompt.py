"""The diagnostician schema-repair dispatch is a pure merge.

ICRA 2026-07-13: the old repair prompt asked the agent to re-read its
prior diagnosis from disk, and the live repair turn read five files,
announced the write, and dead-stopped at one output token — nothing
landed and the run halted on the stale file. The prior JSON now rides
the prompt so the first (and only) action is the corrected Write.
"""

from __future__ import annotations

from dispatch_templates import build_smoke_diagnosis_prompt


class _PathsStub:
    def to_block(self) -> str:
        return "**Paths block**"


_COMMON = dict(
    paths=_PathsStub(),
    stderr_tail="ValueError: boom",
    failing_cell_source="ac(img)",
    failing_cell_index=27,
    section=4,
    diagnosis_output_path="/run/.pipeline/smoke_diagnosis.json",
)

_FINDINGS = [{"id": "J001", "severity": "critical",
              "description": "missing required field value_origin_trace"}]


def test_repair_prompt_inlines_the_prior_diagnosis():
    prior = '{"root_cause": "conv5 kernel too large", "target_agent": "x"}'
    prompt = build_smoke_diagnosis_prompt(
        **_COMMON, repair_findings=_FINDINGS, prior_diagnosis_json=prior)
    assert "SCHEMA REPAIR" in prompt
    assert prior in prompt
    assert "do NOT read any files first" in prompt
    assert "missing required field value_origin_trace" in prompt


def test_repair_prompt_degrades_honestly_without_the_prior_file():
    prompt = build_smoke_diagnosis_prompt(
        **_COMMON, repair_findings=_FINDINGS, prior_diagnosis_json=None)
    assert "SCHEMA REPAIR" in prompt
    assert "prior file unreadable" in prompt


def test_write_first_retry_names_every_required_field():
    # The 07-13 write-first retry produced a file missing exactly
    # value_origin_trace, and the retry preamble's field list was the one
    # place that field went unmentioned.
    prompt = build_smoke_diagnosis_prompt(**_COMMON, retry_mode=True)
    assert "diagnostician RETRY" in prompt
    for field in ("schema_version", "target_agent", "target_file",
                  "bug_shape", "root_cause", "proposed_fix",
                  "value_origin_trace", "reasoning",
                  "paper_fidelity_check"):
        assert field in prompt.split("**Stage 3.c diagnostician RETRY.**")[1]
