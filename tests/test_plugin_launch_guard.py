"""The r2c plugin's launch tools (r2c_run / r2c_paradigms) must be researcher-only.

2026-06-15 incident: the Stage-3a notebook-generator agent hallucinated a
"run the paper parser on bayesian-active-learning" task, found r2c_run in its
toolset, and called it — spawning a rogue parallel pipeline that shared the live
run's opencode server and aborted its notebook-authoring turn (deep-batch halted
at 3a). Root cause: dispatched pipeline agents (all named r2c-*) could reach the
researcher-facing launch tools. The plugin now refuses a launch when the caller
is an r2c-* agent (ToolContext.agent), leaving the researcher's primary-agent
launch untouched.

This pins the containment two ways: a source-level guard-present check that
always runs in CI, and a node behavioral test (skipped where the plugin's
node_modules is absent, e.g. CI without an npm install).
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
PLUGIN = REPO / ".opencode" / "plugin" / "r2c.ts"
OPENCODE_PKG = REPO / ".opencode" / "node_modules" / "@opencode-ai" / "plugin"


def test_launch_guard_is_wired_into_both_tools():
    """Source pin: the guard exists, keys on the r2c- caller prefix, and both
    launch tools consult it before spawning a driver."""
    src = PLUGIN.read_text(encoding="utf-8")
    assert 'DISPATCHED_AGENT_PREFIX = "r2c-"' in src
    assert "function launchToolDeniedFor" in src
    assert "startsWith(DISPATCHED_AGENT_PREFIX)" in src
    # both launch tools consult the guard on their caller
    assert src.count("launchToolDeniedFor(context?.agent)") == 2, (
        "both r2c_run and r2c_paradigms must guard their caller")
    # the guard must run BEFORE the driver spawn in each tool
    run_spawn = src.index('"--paper"')                       # r2c_run's spawn args
    para_spawn = src.index("author_field_guide_proposal.py")  # r2c_paradigms' spawn
    for spawn_at in (run_spawn, para_spawn):
        guard_at = src.rfind("launchToolDeniedFor(context?.agent)", 0, spawn_at)
        assert guard_at != -1, "launch guard must precede the driver spawn"
    assert "runSlugFor" in src
    assert "./r2c-watch" in src
    assert "watch='${watch}'" in src


def test_fresh_flag_threads_through_r2c_run():
    """Item 29 phase 2: r2c_run exposes a `fresh` arg and passes --fresh to the
    driver only when set (default resume stays the empty-args path)."""
    src = PLUGIN.read_text(encoding="utf-8")
    assert "fresh: tool.schema" in src
    assert 'fresh ? ["--fresh"] : []' in src


@pytest.mark.skipif(not shutil.which("node"), reason="node not available")
@pytest.mark.skipif(not OPENCODE_PKG.is_dir(), reason="plugin node_modules absent (e.g. CI without install)")
def test_launch_guard_behavior_via_node():
    """Behavioral: drive the guard THROUGH R2CPlugin's real tools and assert
    deny(r2c-*) / allow(primary). Uses a port-0 (unreachable) server so the
    allow path returns the unreachable message instead of spawning a driver.

    Also pins the EXPORT SURFACE to exactly {R2CPlugin}: opencode invokes every
    plugin-module export as a Plugin at load time, so an extra exported helper
    throws and takes the whole plugin down (the r2c_run-unavailable regression,
    2026-06-15). This also proves the module parses and resolves its SDK import."""
    script = r"""
import * as mod from "./.opencode/plugin/r2c.ts";
const names = Object.keys(mod);
if (names.length !== 1 || names[0] !== "R2CPlugin") {
  console.error("BAD EXPORT SURFACE (breaks opencode plugin load): " + names.join(","));
  process.exit(1);
}
const hooks = await mod.R2CPlugin({ serverUrl: new URL("http://127.0.0.1:0"), directory: process.cwd() });
let ok = true;
for (const [name, t] of [["r2c_run", hooks.tool.r2c_run], ["r2c_paradigms", hooks.tool.r2c_paradigms]]) {
  const arg = name === "r2c_run" ? { paper: "x" } : { run: "x" };
  const deny = String(await t.execute(arg, { agent: "r2c-notebook-generator" }));
  const allow = String(await t.execute(arg, { agent: "build" }));
  if (!deny.includes("researcher-only")) { ok = false; console.error(`FAIL ${name}: must deny r2c-*`); }
  if (allow.includes("researcher-only")) { ok = false; console.error(`FAIL ${name}: must allow primary`); }
}
// Reasoning knobs: R2C_REASONING_EFFORT reaches OpenAI requests,
// R2C_THINKING_BUDGET reaches Anthropic ones, and nothing touches others.
const params = async (providerID) => {
  const output = { options: {} };
  await hooks["chat.params"]({ model: { providerID, id: "m" }, agent: "build" }, output);
  return output.options;
};
const openai = await params("openai");
const anthropic = await params("anthropic");
const google = await params("google");
if (openai.reasoningEffort !== "medium") { ok = false; console.error("FAIL openai reasoningEffort: " + JSON.stringify(openai)); }
if (!anthropic.thinking || anthropic.thinking.budgetTokens !== 8000) { ok = false; console.error("FAIL anthropic thinking: " + JSON.stringify(anthropic)); }
if (Object.keys(google).length !== 0) { ok = false; console.error("FAIL google must be untouched: " + JSON.stringify(google)); }
process.exit(ok ? 0 : 1);
"""
    r = subprocess.run(
        ["node", "--input-type=module", "-e", script],
        cwd=str(REPO), capture_output=True, text=True, timeout=60,
        # Plugin init auto-launches the live viewer (job 3); the opt-out
        # keeps this test hermetic — no real server spawned from pytest.
        env={**os.environ, "R2C_NO_WATCH": "1",
             "R2C_REASONING_EFFORT": "medium", "R2C_THINKING_BUDGET": "8000"},
    )
    assert r.returncode == 0, f"stdout={r.stdout}\nstderr={r.stderr}"


def test_live_view_autolaunch_is_wired_and_optional():
    """Source pin for job 3 (the maintainer's auto-launch call, 2026-07-06): the
    plugin starts the read-only viewer at load, behind a port-reuse
    check and an R2C_NO_WATCH opt-out, and a viewer failure can never
    take the plugin down (the launch tools live in the same module)."""
    src = PLUGIN.read_text(encoding="utf-8")
    assert "R2C_NO_WATCH" in src
    assert "R2C_WATCH_PORT" in src
    assert "scripts/r2c_watch.py" in src
    # Reuse-not-double: the port check gates the spawn.
    port_check = src.index("await portOpen(watchPort)")
    viewer_spawn = src.index("scripts/r2c_watch.py")
    assert port_check < viewer_spawn
    # Non-fatal on failure: the viewer spawn is inside a try, and the
    # detached-spawn helper swallows async spawn errors.
    assert "live view failed to start (non-fatal)" in src
    assert 'child.once("error"' in src
