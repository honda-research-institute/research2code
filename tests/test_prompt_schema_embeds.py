"""B-01 item 4: the two stage-1 schema embeds (paper_map, paradigm_gap).

Two guards shaped from rev 1's specific defects:
- The embed-title test would have caught rev 1 pairing an agent with the
  WRONG model's export (a validator verdict wrapper instead of the pack
  the agent writes): each embedded export's own "title" must equal the
  class the dispatched agent actually produces, and the block header must
  name it.
- The per-agent prompt-size ceiling pins the rendered canonical prompts
  so no future embed silently multiplies a prompt (per-step billing
  re-sends the whole conversation ~8.8x per dispatch).
"""

from __future__ import annotations

import json

from dispatch_templates import (
    SCHEMA_EMBED_EXPORTS,
    STAGE_TASK_SUMMARIES,
    _SCHEMAS_DIR,
    DispatchPaths,
    build_dispatch_prompt,
    schema_embed_block,
    stage_1_schema_embeds,
)

SAMPLE_PATHS = DispatchPaths(
    spec="/tmp/run/.pipeline/method_spec.json",
    paper="/tmp/run/.pipeline/paper.md",
    paper_map="/tmp/run/.pipeline/paper_map.json",
    run_dir="/tmp/run",
    taxonomy_source="taxonomy:knowledge_distillation",
)

# Which class each embed-bearing agent PRODUCES (the artifact the dispatch
# exists to obtain — for the analyzer's gap arm, the gap report).
AGENT_PRODUCED_CLASS = {
    "r2c-decomposer": "PaperMap",
    "r2c-method-analyzer": "ParadigmGapReport",
}


def _canonical_prompt(agent: str, task_key: str) -> str:
    return build_dispatch_prompt(
        task_summary=STAGE_TASK_SUMMARIES[task_key],
        paths=SAMPLE_PATHS,
        extra_sections=stage_1_schema_embeds(agent, "canonical"),
        think_anchor_output_path="/tmp/run/.pipeline/out.json",
    )


def test_embed_title_is_the_model_class_the_agent_produces():
    for class_name, filename in SCHEMA_EMBED_EXPORTS.items():
        export = json.loads(
            (_SCHEMAS_DIR / filename).read_text(encoding="utf-8")
        )
        assert export.get("title") == class_name, (
            f"{filename} exports {export.get('title')!r}, but the embed "
            f"claims {class_name!r} — wrong-model embed (the rev 1 defect)"
        )
        block = schema_embed_block(class_name)
        assert block.startswith(f"## {class_name} JSON schema"), block[:80]
        # The body is the export byte-for-byte (drift is impossible while
        # the export drift test holds).
        assert export == json.loads(
            block.split("```json\n", 1)[1].rsplit("\n```", 1)[0]
        )


def test_agents_are_paired_with_the_class_they_produce():
    for agent, class_name in AGENT_PRODUCED_CLASS.items():
        embeds = stage_1_schema_embeds(agent, "canonical")
        assert len(embeds) == 1
        assert embeds[0].startswith(f"## {class_name} JSON schema")


def test_embeds_are_mode_aware_and_scoped():
    # Chunk-mode dispatches are skipped: their parts shape is deliberately
    # different and untyped.
    for agent in AGENT_PRODUCED_CLASS:
        assert stage_1_schema_embeds(agent, "chunk") == []
    # No other agent grows an embed by accident.
    assert stage_1_schema_embeds("r2c-method-coder", "canonical") == []


def test_canonical_prompts_carry_the_embed_exactly_once():
    for agent, task_key in [
        ("r2c-decomposer", "stage_1_decomposer"),
        ("r2c-method-analyzer", "stage_1_analyzer"),
    ]:
        prompt = _canonical_prompt(agent, task_key)
        header = f"## {AGENT_PRODUCED_CLASS[agent]} JSON schema"
        assert prompt.count(header) == 1, f"{agent}: {prompt.count(header)}"


# Measured 2026-08-20: decomposer 9,264 bytes / analyzer 20,477 bytes
# rendered canonical, embeds included. Ceilings carry ~30% headroom; a
# future embed (each is 4-5k bytes) blows them and must raise them here
# DELIBERATELY, with the cost recosted.
PROMPT_SIZE_CEILINGS = {
    "r2c-decomposer": 13_000,
    "r2c-method-analyzer": 27_000,
}

# The largest task summary is the analyzer's at ~11.1k bytes. A schema
# pasted INTO a summary (instead of through stage_1_schema_embeds) trips
# this even if the per-agent ceiling above is dodged.
TASK_SUMMARY_CEILING = 13_000


def test_per_agent_prompt_size_ceiling():
    for (agent, task_key), ceiling in zip(
        [("r2c-decomposer", "stage_1_decomposer"),
         ("r2c-method-analyzer", "stage_1_analyzer")],
        [PROMPT_SIZE_CEILINGS["r2c-decomposer"],
         PROMPT_SIZE_CEILINGS["r2c-method-analyzer"]],
    ):
        size = len(_canonical_prompt(agent, task_key))
        assert size < ceiling, (
            f"{agent} canonical prompt is {size} bytes (ceiling {ceiling}) "
            f"— an embed multiplied the prompt; raise the ceiling only "
            f"deliberately"
        )


def test_every_task_summary_stays_prose_sized():
    for key, summary in STAGE_TASK_SUMMARIES.items():
        assert len(summary) < TASK_SUMMARY_CEILING, (
            f"STAGE_TASK_SUMMARIES[{key!r}] is {len(summary)} bytes — "
            f"schema-sized content belongs in stage_1_schema_embeds, not "
            f"in a task summary"
        )
