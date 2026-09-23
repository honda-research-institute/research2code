/**
 * R2C launch plugin — the replacement for r2c-start.sh (2026-06-11).
 *
 * Runs inside whatever opencode server hosts this project, in TUI-embedded
 * and `opencode serve` headless modes alike, and does the launch script's
 * two surviving jobs from the inside:
 *
 * 1. Address handoff. The plugin receives the server's own URL and exports
 *    it (R2C_SERVER_URL) into the server process environment, which every
 *    bash-tool child inherits — so backgrounded drivers find their server
 *    without a port convention. R2C_PORT is exported too so the legacy
 *    slash-command bash path keeps working during the strangler window.
 *    Per-window servers get distinct addresses automatically, which kills
 *    the parallel-run port-collision class by construction.
 *
 * 2. Loopback proxy exemption. Corporate WSL environments set
 *    http_proxy/HTTP_PROXY with no localhost exemption (a researcher's proxied machine,
 *    2026-06-11: loopback API calls answered by the corporate Squid).
 *    Loopback names are prepended to no_proxy/NO_PROXY, preserving any
 *    corporate list, for everything the server spawns. The driver's own
 *    client is additionally proxy-immune (opencode_client._OPENER), so
 *    this is defense in depth for other tooling.
 *
 * It also defines the run tools (r2c_run / r2c_paradigms): deterministic
 * detached spawns of the Python drivers, replacing the paste-this-bash-
 * verbatim slash commands whose instruction blocks accumulated three
 * incident postmortems. The spawn redirects stdout to a /tmp log FILE —
 * that is the shape run_pipeline's deadlock guard sanctions (a pipe
 * stdout means a foreground bash tool, which would deadlock the session
 * against the driver's own dispatches; see README.md, "How it works").
 *
 * 3. Live-view auto-launch (the maintainer's call, 2026-07-06): the read-only
 *    viewer (fleet page at /, per-run behavioral timeline at
 *    /run/<slug>) starts with the server so nobody has to remember a
 *    second command — this is the plugin-era home of what r2c-start.sh
 *    does for the legacy path. The viewer is spawned detached, so it
 *    survives server restarts; the next server load finds the port
 *    occupied and reuses it (at most one viewer per port, by
 *    construction). R2C_NO_WATCH=1 skips it, R2C_WATCH_PORT overrides
 *    the port. The browser opens only on a fresh viewer start, never
 *    on reuse, so server restarts do not spray tabs.
 *
 * Readiness probing has no replacement: code running inside the server
 * cannot observe it not-yet-running. That whole class (curl shims,
 * proxy-poisoned probes, startup races) is deleted, not migrated.
 */
import { tool, type Plugin } from "@opencode-ai/plugin"
import { spawn } from "node:child_process"
import { existsSync, mkdtempSync, openSync, readFileSync } from "node:fs"
import { connect } from "node:net"
import { tmpdir } from "node:os"
import { join } from "node:path"

// Containment: r2c_run / r2c_paradigms are RESEARCHER tools. Every dispatched
// pipeline agent is named `r2c-*` (mode: subagent). On 2026-06-15 the Stage-3a
// notebook-generator hallucinated a "run the paper parser on
// bayesian-active-learning" task, found r2c_run in its toolset, and called it —
// spawning a rogue parallel pipeline that shared this server and aborted the
// live run. The researcher's /r2c-run always runs in the primary TUI agent
// (never `r2c-*`), so a denylist on the `r2c-` caller prefix blocks every
// dispatched subagent (current and future) by construction while leaving the
// human launch path untouched. Failure mode is safe: an unidentified caller is
// allowed (the launch keeps working), never silently blocked.
//
// NOT exported: opencode treats every export of a plugin module as a Plugin and
// invokes it at load time, so an exported helper throws and takes R2CPlugin down
// with it (the r2c_run-unavailable regression, 2026-06-15). Only R2CPlugin is
// exported; the guard is exercised through R2CPlugin's tools in the test.
const DISPATCHED_AGENT_PREFIX = "r2c-"

function launchToolDeniedFor(agent?: string): string | null {
  if (agent && agent.startsWith(DISPATCHED_AGENT_PREFIX)) {
    return (
      "ERROR: launch tools (r2c_run / r2c_paradigms) are researcher-only and " +
      `cannot be called by a dispatched pipeline agent (caller: ${agent}). ` +
      "This is not your task — ignore this tool and complete your assigned work."
    )
  }
  return null
}

export const R2CPlugin: Plugin = async ({ serverUrl, directory }) => {
  const server = serverUrl.origin

  // A TUI launched without --port keeps its server in-process: serverUrl
  // carries port 0 and NOTHING external can connect (verified 2026-06-11
  // on binary 1.16.2 — no TCP listener exists). The drivers are external
  // processes, so this window cannot host pipeline runs; the tools below
  // refuse with the fix instead of spawning a driver doomed to fail its
  // health check.
  const reachable = serverUrl.port !== "0" && serverUrl.port !== ""
  const unreachableMsg =
    "ERROR: this opencode window's server is not externally reachable " +
    "(launched without --port, so the embedded server never bound a TCP " +
    "listener). The R2C driver is an external process and cannot connect. " +
    "Relaunch with `opencode --port 4096` (any free port; parallel windows " +
    "need distinct ports), or use the legacy ./r2c-start.sh."

  // Job 1: address handoff to every child the server spawns.
  if (reachable) {
    process.env.R2C_SERVER_URL = server
    process.env.R2C_PORT = serverUrl.port
  }

  // Job 2: proxy exemptions for everything the server spawns, corporate
  // list preserved (tokenized merge: dedupes, drops empty entries from
  // ragged corporate values). Loopback always; R2C_NO_PROXY_EXTRA adds
  // the domains hosting your model endpoints, MCP gateway, or Marker
  // parse service (comma-separated) when a corporate proxy cannot
  // retrieve internal hosts (observed 2026-06-11 on a WSL machine: every
  // provider check answered with a proxy error page). NOTE this only
  // covers child processes — opencode's OWN startup fetches need the
  // exemption in the user's shell (see README's WSL note).
  const extra = (process.env.R2C_NO_PROXY_EXTRA ?? "")
    .split(",")
    .map((t) => t.trim())
    .filter(Boolean)
  const existing = (process.env.no_proxy ?? process.env.NO_PROXY ?? "")
    .split(",")
    .map((t) => t.trim())
    .filter(Boolean)
  const merged = [
    ...new Set(["localhost", "127.0.0.1", ...extra, ...existing]),
  ].join(",")
  process.env.no_proxy = merged
  process.env.NO_PROXY = merged

  console.log(`[r2c] plugin loaded: server=${server} dir=${directory}`)

  const spawnDetached = (args: string[], logPrefix: string) => {
    const log = join(
      mkdtempSync(join(tmpdir(), `${logPrefix}-`)),
      "driver.log",
    )
    const fd = openSync(log, "a")
    const child = spawn("python3", args, {
      cwd: directory,
      detached: true, // own process group: outlives turns and subagents
      stdio: ["ignore", fd, fd], // regular-file stdout (deadlock-guard shape)
      env: { ...process.env, R2C_SERVER_URL: server },
    })
    // An async spawn failure (e.g. python3 missing) emits 'error'; without
    // a listener that is an uncaught exception INSIDE the opencode server.
    child.once("error", (err) => {
      console.log(`[r2c] detached spawn failed (${logPrefix}): ${err}`)
    })
    child.unref()
    return { pid: child.pid, log }
  }

  const runSlugFor = (paper: string) => {
    const trimmed = paper.trim()
    const base = trimmed.split(/[\\/]/).pop() ?? trimmed
    return base.replace(/\.(pdf|md)$/i, "")
  }

  // Job 3: live-view auto-launch. Port-check first so a viewer left by a
  // previous server (or by legacy r2c-start.sh, which composes with this
  // via the same check) is reused, never doubled. Every failure path is
  // swallowed after a log line — the viewer is a convenience and must
  // never take the plugin (and with it the launch tools) down.
  const watchPort = Number(process.env.R2C_WATCH_PORT ?? "8765") || 8765
  const watchUrl = `http://127.0.0.1:${watchPort}/`
  const portOpen = (port: number): Promise<boolean> =>
    new Promise((resolve) => {
      const sock = connect({ port, host: "127.0.0.1" })
      const done = (up: boolean) => { sock.destroy(); resolve(up) }
      sock.once("connect", () => done(true))
      sock.once("error", () => done(false))
      sock.setTimeout(500, () => done(false))
    })
  const openBrowser = (url: string) => {
    const attempts =
      process.platform === "darwin" ? [["open", url]]
      : process.platform === "win32" ? [["cmd", "/c", "start", "", url]]
      : [["xdg-open", url], ["wslview", url]]
    const tryOne = (index: number) => {
      if (index >= attempts.length) return
      try {
        const child = spawn(attempts[index][0], attempts[index].slice(1), {
          stdio: "ignore", detached: true,
        })
        child.once("error", () => tryOne(index + 1))
        child.unref()
      } catch {
        tryOne(index + 1)
      }
    }
    tryOne(0)
  }
  if (!process.env.R2C_NO_WATCH) {
    if (await portOpen(watchPort)) {
      console.log(`[r2c] live view already up at ${watchUrl} (reusing)`)
    } else {
      try {
        const viewer = spawnDetached(
          [
            "scripts/r2c_watch.py",
            "--host", "127.0.0.1",
            "--port", String(watchPort),
          ],
          "r2c-watch",
        )
        console.log(
          `[r2c] live view started at ${watchUrl} (pid=${viewer.pid}, log=${viewer.log})`,
        )
        openBrowser(watchUrl)
      } catch (err) {
        console.log(`[r2c] live view failed to start (non-fatal): ${err}`)
      }
    }
  }

  // Every R2C agent runs on the model named by R2C_MODEL (opencode.json reads
  // it as the default model). Unset means opencode picks its own default,
  // which is rarely what a pipeline run wants, so say so at startup.
  if (!process.env.R2C_MODEL) {
    console.log(
      "[r2c] R2C_MODEL is not set; opencode will use its default model. " +
        "Set R2C_MODEL=<provider>/<model> (see .env.example) before starting.",
    )
  }

  // Reasoning level for the served model, from two env vars (see
  // .env.example): R2C_REASONING_EFFORT (OpenAI: low|medium|high) and
  // R2C_THINKING_BUDGET (Anthropic: extended-thinking budget in tokens).
  // Unset means the provider's default. Applied per request so the same
  // knobs cover every agent; other providers are left untouched.
  const reasoningEffort = (process.env.R2C_REASONING_EFFORT ?? "").trim()
  const thinkingBudget = Number.parseInt((process.env.R2C_THINKING_BUDGET ?? "").trim(), 10)

  return {
    "chat.params": async (input, output) => {
      const provider = input.model?.providerID
      if (provider === "openai" && reasoningEffort) {
        output.options.reasoningEffort = reasoningEffort
      } else if (provider === "anthropic" && thinkingBudget > 0) {
        output.options.thinking = { type: "enabled", budgetTokens: thinkingBudget }
      }
    },

    tool: {
      r2c_run: tool({
        description:
          "Start the R2C pipeline on a paper. Spawns the Python driver " +
          "detached with this server's own address; the session stays " +
          "free for the driver's dispatches. Report the pid and log " +
          "path back to the researcher; the driver runs autonomously " +
          "from there (side-panel todos, halt notices in-session). " +
          "Pass fresh=true to start a clean roll (archives any existing " +
          "delivery for this slug to <slug>_<n>, nothing deleted). " +
          "Pass paper='__probe__' only for launch-path diagnostics: it " +
          "spawns a 60s sleeper instead of the driver.",
        args: {
          paper: tool.schema
            .string()
            .describe(
              "Paper input: a slug (e.g. bayesian-active-learning), or " +
              "a .md/.pdf filename or path under input_papers/",
            ),
          fresh: tool.schema
            .boolean()
            .optional()
            .describe(
              "Start a clean roll: archive an existing delivery for this " +
              "slug to <slug>_<n> (nothing deleted) and run from scratch, " +
              "instead of the default same-dir resume. Set this when the " +
              "researcher typed `/r2c-run <slug> --fresh`. Refused if a run " +
              "is already live on the slug.",
            ),
        },
        async execute({ paper, fresh }, context) {
          const denied = launchToolDeniedFor(context?.agent)
          if (denied) {
            console.log(`[r2c] BLOCKED r2c_run launch by dispatched agent ${context?.agent}`)
            return denied
          }
          if (!reachable) return unreachableMsg
          if (!paper.trim()) return "ERROR: empty paper argument"
          if (paper === "__probe__") {
            const probe = spawnDetached(
              ["-c", "import time; time.sleep(60)"],
              "r2c-probe",
            )
            return `probe sleeper started (pid=${probe.pid}, log=${probe.log}, server=${server})`
          }
          const { pid, log } = spawnDetached(
            [
              "scripts/run_pipeline.py",
              "--paper", paper,
              "--server-url", server,
              ...(fresh ? ["--fresh"] : []),
            ],
            "r2c-run",
          )
          const watch = `./r2c-watch ${runSlugFor(paper)}`
          const view = `http://127.0.0.1:${watchPort}/run/${encodeURIComponent(runSlugFor(paper))}`
          const mode = fresh ? " (fresh roll: prior delivery archived to <slug>_<n>)" : ""
          return `driver started${mode} (pid=${pid}, log=${log}, server=${server}, live view=${view}, watch='${watch}')`
        },
      }),

      r2c_paradigms: tool({
        description:
          "Author a taxonomy-pack proposal for a halted unmatched-paper " +
          "run (one with .pipeline/paradigm_gap_report.json). Spawns " +
          "the authoring driver detached; report the pid, log path and " +
          "run dir back to the researcher. Promotion stays a manual " +
          "maintainer action (scripts/apply_paradigm_proposal.py).",
        args: {
          run: tool.schema
            .string()
            .describe(
              "Run slug under r2c_runs/ (e.g. ADAM) or an explicit run " +
              "directory path",
            ),
        },
        async execute({ run }, context) {
          const denied = launchToolDeniedFor(context?.agent)
          if (denied) {
            console.log(`[r2c] BLOCKED r2c_paradigms launch by dispatched agent ${context?.agent}`)
            return denied
          }
          if (!reachable) return unreachableMsg
          if (!run.trim()) return "ERROR: empty run argument"
          const candidates = [run, join("r2c_runs", run)]
          const runDir = candidates.find((c) =>
            existsSync(join(directory, c)),
          )
          if (!runDir) {
            return `ERROR: run dir not found: ${run} (also checked r2c_runs/${run})`
          }
          const { pid, log } = spawnDetached(
            [
              "scripts/author_field_guide_proposal.py",
              "--run-dir", runDir,
            ],
            "r2c-paradigms",
          )
          return `taxonomy-pack author started (pid=${pid}, log=${log}, run_dir=${runDir}, server=${server})`
        },
      }),
    },
  }
}
