# R2C

R2C turns a research paper (PDF or markdown) into working, explained, importable code, on your laptop, with an honest report about how much to trust the result.

Give it a paper and it delivers, under `r2c_runs/<paper>/`:

- **`REPORT.md`**, start here, always. The delivery label (how much was verified), every finding in plain language, and a claims table with honest verdicts.
- **`METHOD.md`**, the paper's method explained equation by equation.
- **`notebook.ipynb`**, a tutorial that runs the method end to end on a small demo-scale dataset.
- **`method/`**, an importable Python package (`from method import ...`) with its own README.

Supporting evidence (assumptions the system made, known issues, deferred findings, the machine-readable manifest) lives under `details/`, linked from `REPORT.md`. `.pipeline/` is internal working state you never need to open.

## What you need

- **A Unix shell.** macOS and Linux work as-is. On Windows use WSL (recommended) or Git Bash; `r2c-start.sh` is a bash script.
- **Python 3.10 or newer** for R2C itself (3.10 to 3.13 if you plan to feed it PDFs, see Marker below). Heavy method-specific dependencies (torch and friends) are not installed up front; each generated package installs its own during the run.
- **[opencode](https://opencode.ai)**, the agent runtime R2C dispatches through.
- **A model provider API key.** The default model is Google's `gemini-flash-latest`, which has a free tier. See [Getting a free Google API key](#getting-a-free-google-api-key). OpenAI and Anthropic work the same way, see [Choosing a model](#choosing-a-model).
- **Optional: [Marker](https://github.com/datalab-to/marker)** for PDF inputs. Markdown inputs need nothing extra. See [Running your own paper](#running-your-own-paper).

## Quickstart

```bash
# 1. R2C's own (light) dependencies
pip install -r requirements.txt

# 2. Model and key. Copy the template, keep one R2C_MODEL line and the
#    matching key, delete the rest.
cp .env.example .env

# 3. Start the R2C environment: an opencode server plus TUI on one port,
#    and the live view in your browser.
./r2c-start.sh 4096

# 4. In the opencode window, run one of the two starter papers:
/r2c-run bayesian-active-learning
```

The driver detaches and runs on its own, roughly 30 to 120 minutes depending on the paper and the model. The command prints a log path you can `tail -f` from any terminal. When it finishes:

```bash
cd r2c_runs/bayesian-active-learning
# read REPORT.md first, then:
pip install -r requirements.txt
jupyter notebook notebook.ipynb     # then "Run All"
```

`input_papers/` ships two starter papers (`bayesian-active-learning`, `deep-batch-active-learning`), each as the original PDF and its parsed markdown.

## Getting a free Google API key

The default model runs on Google AI Studio's free tier. No credit card is needed.

1. Go to [aistudio.google.com/apikey](https://aistudio.google.com/apikey) and sign in with a Google account.
2. Click **Create API key**. If asked, pick or create a Google Cloud project (any name works, it is only a container for the key).
3. Copy the key. It starts with `AIza`.
4. Put it in `.env`:

   ```bash
   export R2C_MODEL="google/gemini-flash-latest"
   export GOOGLE_GENERATIVE_AI_API_KEY="AIza..."
   ```

The free tier is rate-limited per minute and per day (see the caveats under [Choosing a model](#choosing-a-model)). If a run stalls on quota, wait for the daily reset or add billing to the project, which keeps the same key.

## Choosing a model

Every R2C agent runs on one model, named by `R2C_MODEL` as an opencode `<provider>/<model>` id. `./r2c-start.sh` reads it from `.env`, and the pipeline loads the same file (a variable already exported in your shell always wins). [`.env.example`](.env.example) lists every knob. The ones that matter:

| Variable | What it does |
|---|---|
| `R2C_MODEL` | `google/gemini-flash-latest` (default, free tier), `openai/gpt-5.6`, `anthropic/claude-sonnet-5`, or any id opencode knows |
| `GOOGLE_GENERATIVE_AI_API_KEY`, `OPENAI_API_KEY`, `ANTHROPIC_API_KEY` | The key for the provider you chose (or use `opencode auth login`) |
| `R2C_REASONING_EFFORT` | OpenAI reasoning level: `low`, `medium`, `high` |
| `R2C_THINKING_BUDGET` | Anthropic extended-thinking budget in tokens |
| `R2C_TIMEOUT_SCALE` | Multiplier on every agent timeout, default 3.0. `0` removes the limits |

For a concrete picture of the tradeoff, [example_output/](example_output/) holds three complete deliveries of the same paper on different models, with a README comparing time, cost, and what changed.

Two honest caveats:

- **R2C was built and tuned on a self-hosted Qwen 27B model.** Other models work through the same opencode provider layer, but quality and speed will vary, and the pipeline's agents lean hard on tool calling. A weaker model may stall in the tool-heavy coding stages. The timeout budgets you get by default are three times the ones that model needed.
- **Free tiers are rate-limited and quota-limited.** A single agent turn carries tens of thousands of prompt tokens, and a full run makes hundreds of turns. Expect rate-limit pauses on a free tier, and possibly a daily quota wall. That is a slowdown, not a cost, but the run will take longer than the numbers above.

## Running your own paper

Drop your paper into `input_papers/`, then `/r2c-run <its-filename-without-extension>`.

- A **`.md`** markdown export runs directly, no conversion step.
- A **`.pdf`** is converted to markdown with [Marker](https://github.com/datalab-to/marker), running locally on your machine. Install it once:

  ```bash
  pip install marker-pdf      # needs Python 3.10 to 3.13
  brew install llama.cpp      # Marker's inference backend (Linux: see llama.cpp releases)
  ```

  Marker downloads about 1.5 GB of models on first use, and a CPU parse of a paper takes several minutes. On Python 3.14 install Marker into its own venv and point `R2C_MARKER_BIN` at its `marker_single`. Marker's optional LLM assist is off by default (`R2C_MARKER_USE_LLM`), and the pipeline strips figures from the parsed text anyway.

- When both formats exist under the same name, R2C uses the `.md`. Say `/r2c-run <name>.pdf` to force a fresh parse.

## What happens during a run

Five phases, in plain language:

1. **Understand the paper.** Parse it, map its sections and claims, extract the method into a structured spec.
2. **Explain the method.** Write `METHOD.md`. This ships even if later phases fail.
3. **Write the code.** Generate the model, training, and method code, then validate that the package imports and satisfies its declared architecture contract. On a fresh machine this phase also installs missing heavy dependencies.
4. **Build and smoke-run the notebook.** Generate the tutorial, then execute it end to end. Usually the longest phase.
5. **Verify and report.** Run behavioral probes against the package, route review findings, write `REPORT.md` with the delivery label.

Each phase is gated by checks. When a check cannot be satisfied, the run either continues with the issue logged honestly or stops on purpose. Halts are most common in phases 3 and 4, where the generated code first meets reality.

### Watching progress

- **The live view** at [http://127.0.0.1:8765/](http://127.0.0.1:8765/) opens automatically with `./r2c-start.sh`: every run with its label, and for a live run the current stage and a plain-language timeline. Read-only and local to your machine.
- The opencode side panel ticks off stages (it can lag while a long agent turn runs).
- `tail -f <log path>` is the raw stream, and `r2c_runs/<slug>/.pipeline/progress.json` always holds the current stage.

### When something fails

1. Read the halt notice in the opencode window or the run's `details/KNOWN_ISSUES.md`, or ask: `/r2c-chat <slug>`, then "why did this run stop?"
2. To resume, run `/r2c-run <slug>` again. Completed stages are never redone.
3. To start truly fresh, delete `r2c_runs/<slug>/` first.
4. Most failures are transient (a hung or rate-limited model call). If a resume hits the same wall twice, the halt artifact under `r2c_runs/<slug>/.pipeline/` is the thing to open an issue with.

### Batch runs

For several papers, one at a time with a fast-fail guard that stops when runs die the same way twice:

```bash
# manifest: one paper path per line, optional 'wipe' to start fresh
export R2C_SERVER_URL=http://127.0.0.1:4096
bash scripts/run_batch.sh my-manifest.txt &
```

Progress lands in `r2c_runs/_batch_logs/`.

## Reading a delivery

### The five-minute read

1. **Open `REPORT.md`.** The first line under the title is the delivery label. It tells you how much to trust the package before you read anything else.
2. **Read "Why this label".** Every finding that held the package below a stronger label, in plain language, with what it means for you.
3. **Skim the claims table.** One row per claim the paper makes, with an honest verdict: verified at demo scale, untested at this scale, or "needs a fix on our side". Untested is the honest default. A laptop-scale demo cannot reproduce a paper's full-scale numbers, and R2C never pretends it did.
4. **Check `details/assumptions.md`.** Decisions the system made for you (a rescaled constant, a raised training budget), each with reasoning and a "to override" line.
5. **Then open the notebook.** Section headings walk the method step by step. The package under `method/` is what you import into your own code.

### What the labels mean

- **verified**: the code runs, and at least one behavioral check that would fail for a trivial baseline passed for this package. The strongest label. It does not mean the paper's headline numbers were reproduced.
- **draft**: the package was produced end to end, but specific findings need your attention before you rely on it. Often you are one small fix away.
- **uncertified, new territory**: nothing failed, but nobody verified the core mechanism either. The behavioral checks for this paper's family do not exist yet or could not run. Treat the core-method code like a colleague's first draft.
- **explanation only**: R2C could not produce a reliable code package. `METHOD.md` still stands on its own.

A label can also carry **PARTIAL**: one or more components were deliberately stubbed instead of implemented, each stub naming a work order for finishing it by hand.

### The run folder

| Item | What it is |
|---|---|
| `REPORT.md` | The front door: label, findings, claims ledger |
| `METHOD.md` | The paper's method explained equation by equation |
| `notebook.ipynb` | Tutorial that runs the method on demo-scale data |
| `requirements.txt` | Paper-specific dependencies for the notebook and package |
| `method/` | The importable package, with its own README |
| `details/` | Supporting evidence, linked from `REPORT.md` |
| `.pipeline/` | Internal working state |

Inside `details/`: `assumptions.md` (every decision made for you, with overrides), `KNOWN_ISSUES.md` (what could not be satisfied, and what to try), `deferred_findings.md` (review findings for your judgment), `final_manifest.json` (machine-readable record), and `POST_DELIVERY_CHANGES.md` (appears once you change the run through `/r2c-chat`).

### Ground rules the system keeps

- R2C never edits your code to make a check pass. It reports honestly and stops.
- Changes you request through `/r2c-chat` are the exception, and they come with a paper trail: each run folder is its own git repository, every chat change is one revertable commit, and the delivery label always describes the run as delivered.
- Labels encode evidence, not quality. A `draft` with two precise findings is often more useful than it sounds.
- Demo scale is deliberate. The parameter table in the notebook says which values are paper-faithful and which are demo-scale accommodations, and how to scale them back up.

## Talk to your runs

Once a run finishes (delivered or halted), in the same opencode window:

```
/r2c-chat            # overview of every run you have
/r2c-chat <slug>     # focus on one run
```

The chat is grounded in the run's own artifacts and can **explain** ("why did this run stop?", "is the alpha weighting doing what the paper says?"), **change things** ("rewrite the training loop with vanilla SGD", "port the sampling helper from my other run"), and **plan** ("how would I scale this to my real dataset?"). Every change is re-checked, recorded in `details/POST_DELIVERY_CHANGES.md`, and committed to the run folder's own git history, so "undo that" always works. A run that is currently executing is read-only.

## Network notes

**Proxy environments.** If launching fails with proxy error pages, your environment routes loopback traffic through a proxy. `./r2c-start.sh` exempts `localhost` and `127.0.0.1` for everything it starts. Add your provider's domains through `R2C_NO_PROXY_EXTRA` in `.env` if they are unreachable through the proxy.

**Restricted package downloads.** If pip is blocked, let the run continue to its dependency halt. R2C writes `r2c_runs/<slug>/requirements.txt` before stopping and prints an install command for the exact interpreter it used. Install on a network where pip works, then resume with `/r2c-run <slug>`. R2C checks installed versions locally and skips the install when the environment is already prepared. `R2C_WHEELHOUSE` points it at a local wheel directory instead.

## How it works

The pipeline is paper-agnostic and taxonomy-aware: the same driver runs every paper without code changes. Method and task knowledge lives in the taxonomy (`docs/ssot/taxonomies.yaml`) plus paper-derived specs. A Python driver (`scripts/run_pipeline.py`) sequences the stages and dispatches LLM producer agents over opencode's HTTP API. Every judgment call lives in an agent; all sequencing, validation, retry, and routing is deterministic driver code.

```
Stage 0 — setup + paper parsing                       (script)
Stage 1 — decompose + analyze paper
  1.a   r2c-decomposer                (LLM agent → paper_map.json)
  1.b   r2c-method-analyzer           (LLM agent → method_spec.json)
  1.x   method explanation            (script + LLM explainer → METHOD.md)
Stage 2 — package generation
  2.a   package_scaffolder            (script)
  2.b   r2c-architecture-coder        (LLM agent → model.py + training.py)
  2.c   r2c-method-coder              (LLM agent → method.py)
  2.d   init_finalizer                (script → __init__.py + requirements.txt)
  2.x   parameter_deriver             (script → params.json)
Stage 3 — notebook generation
  3.a   r2c-notebook-generator        (LLM agent → notebook_draft.py)
  3.b   render_notebook               (script → notebook.ipynb)
  3.c   smoke_run_notebook            (script — runtime gate)
Stage 4 — r2c-paper-fidelity-reviewer  (LLM agent — emergent properties only)
Stage 5 — findings routing + delivery gating + REPORT.md (driver)
```

Each producing stage is gated by three checks: existence, mechanical quality, and semantic review (`r2c-stage-reviewer`), with findings routed through one fix-mode dispatch path.

`/r2c-run` invokes the r2c plugin (`.opencode/plugin/r2c.ts`), which spawns the driver detached with the hosting server's own address. The detached spawn is required: the driver POSTs agent dispatches back into the same server, and a foreground driver would deadlock. A window launched without a port cannot host runs; the run tool says so.

For development, partial pipelines run directly from a separate terminal:

```bash
python3 scripts/run_pipeline.py --paper <slug> --stop-after stage_2x
```

Stages whose outputs are already on disk skip cheaply, so re-runs after a fix only re-execute the affected stages and everything downstream.

## Workspace layout

```
.opencode/
  plugin/r2c.ts                        Launch plugin: server-address handoff, run tools, live view
  command/                             /r2c-run, /r2c-chat, /r2c-paradigms slash commands
  agents/                              LLM producer agents the driver dispatches
  opencode.json                        Default model = R2C_MODEL
docs/
  ssot/taxonomies.yaml                 Method/task taxonomy (single source of truth)
  generated/field-guides/              Per-node agent field guides (generated from the taxonomy)
example_output/                      Three finished deliveries of one paper on different models, with a comparison README
input_papers/                          Starter papers (PDF + parsed markdown)
paradigms/                             Node-owned file templates for the scaffolder
schemas/                               Pydantic models, the canonical data contracts
scripts/
  run_pipeline.py                      Pipeline driver
  parse_pdf.py                         PDF → markdown via local Marker
  r2c_watch.py / fleet_state.py        Live-view server
  run_batch.sh                         Batch runner
  ...                                  Per-stage scripts + deterministic validators
tests/                                 Pytest suite + regression fixtures
r2c_runs/<slug>/                       Per-paper pipeline outputs (gitignored)
r2c-start.sh                           Start server + TUI + live view on one port
.env.example                           Every configuration variable, documented
requirements.txt                       R2C's own deps (no torch here)
```

## Contributing

See [CONTRIBUTING](CONTRIBUTING.md) for setup, the test gates, and the ground rules.

## License
 See [LICENSE](LICENSE.pdf).
