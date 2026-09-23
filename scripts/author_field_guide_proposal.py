#!/usr/bin/env python3
"""Orchestrate advisory taxonomy-pack proposal authoring.

This maintainer command sits between the existing scaffold and promotion tools:

1. create or reuse a proposal packet under a run's `.pipeline/` directory,
2. dispatch `r2c-field-guide-author` to author only that proposal packet,
3. validate the packet,
4. redispatch with validation errors until it passes or the retry cap is hit.

It never edits the canonical taxonomy; promotion remains an explicit maintainer
step through `scripts/apply_paradigm_proposal.py`.
"""

from __future__ import annotations

import argparse
import json
import os
import stat
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from schemas.paradigm_proposal import ParadigmProposal, ProposalValidationResult  # noqa: E402
from scripts import taxonomy  # noqa: E402
from scripts.opencode_client import (  # noqa: E402
    DEFAULT_PORT,
    DispatchResult,
    OpencodeClientError,
    SessionNotFound,
    ServerUnreachable,
    discover_session,
    dispatch_and_wait,
    get_agent_model,
    health_check,
)
from scripts.propose_field_guide import PROPOSAL_ROOT, create_proposal_packet  # noqa: E402
from scripts.validate_paradigm_proposal import validate_proposal  # noqa: E402


AUTHOR_AGENT = "r2c-field-guide-author"
DEFAULT_TIMEOUT_S = 1800
DEFAULT_MAX_ITERATIONS = 3
AUTHORING_LOG = "proposal_authoring_log.md"


class AuthoringError(RuntimeError):
    """Raised when proposal authoring cannot continue safely."""


@dataclass(frozen=True)
class AuthoringResult:
    proposal_dir: Path
    valid: bool
    iterations: int
    errors: list[str]
    warnings: list[str]
    validation_report_path: Path
    authoring_log_path: Path
    dry_run_prompt_path: Path | None = None


@dataclass(frozen=True)
class FileState:
    size: int
    mtime_ns: int


DispatchFn = Callable[[str], DispatchResult]


def resolve_or_create_proposal_dir(
    run_dir: Path,
    repo_root: Path = ROOT,
    *,
    proposal_dir: Path | None = None,
    proposal_id: str | None = None,
    overwrite_scaffold: bool = False,
    force_new: bool = False,
) -> Path:
    """Return a proposal directory, creating one from the gap report if needed."""

    run_dir = run_dir.resolve()
    repo_root = repo_root.resolve()
    if proposal_dir is not None:
        resolved = proposal_dir.resolve()
        if not (resolved / "proposal.json").is_file():
            raise AuthoringError(f"proposal.json missing in proposal dir: {resolved}")
        return resolved

    proposals_root = run_dir / PROPOSAL_ROOT
    existing = sorted(
        path
        for path in proposals_root.iterdir()
        if path.is_dir() and (path / "proposal.json").is_file()
    ) if proposals_root.is_dir() else []

    if existing and not force_new and proposal_id is None:
        return existing[-1]

    try:
        return create_proposal_packet(
            run_dir,
            repo_root,
            proposal_id=proposal_id,
            overwrite=overwrite_scaffold,
        )
    except Exception as exc:  # noqa: BLE001
        raise AuthoringError(f"could not create proposal packet: {exc}") from exc


def build_author_prompt(
    *,
    proposal_dir: Path,
    run_dir: Path,
    repo_root: Path = ROOT,
    validation: ProposalValidationResult | None = None,
) -> str:
    """Build the per-dispatch prompt for `r2c-field-guide-author`."""

    proposal = _load_proposal(proposal_dir)
    gap_report_path = run_dir / proposal.source_gap_report
    paper_path = run_dir / ".pipeline" / "paper.md"
    paper_map_path = run_dir / ".pipeline" / "paper_map.json"
    tax = taxonomy.load_taxonomy(repo_root)
    parent = tax.node_for_legacy(proposal.extends) if proposal.extends else None
    parent_pack = taxonomy.load_pack(proposal.extends, repo_root) if proposal.extends else None

    validation_block = _validation_prompt_block(validation)
    parent_taxonomy_id = getattr(parent, "taxonomy_id", getattr(parent, "id", None)) if parent else None
    parent_block = (
        f"- Parent taxonomy node: `{parent_taxonomy_id}` for `{proposal.extends}`"
        if parent_taxonomy_id else "- Parent taxonomy node: `-`"
    )
    inherited_hint = ""
    if parent_pack:
        inherited_hint = json.dumps(
            {
                key: parent_pack.get(key)
                for key in ("fingerprint", "scaffold_hints", "pluggable_component")
                if parent_pack.get(key)
            },
            indent=2,
            sort_keys=True,
        )

    return f"""# Taxonomy-Pack Proposal Authoring Task

USE YOUR WRITE TOOL to author or revise the proposal packet in:

`{proposal_dir}`

Do not return until `{proposal_dir / 'pack.yaml'}` exists and reflects the task below.

## Critical Scope

You are `r2c-field-guide-author`, not a maintainer and not a pipeline runner.
You are authoring a proposal packet only.

Allowed write targets:

- `{proposal_dir / 'pack.yaml'}`
- `{proposal_dir / 'proposal.md'}`
- `{proposal_dir / 'coupling_report.md'}`

Forbidden write targets:

- `paradigms/**`
- `scripts/**`
- `schemas/**`
- `.opencode/**`
- `docs/**`
- `{run_dir / 'method'}**`
- any other run artifact outside `{proposal_dir}`

Do not promote the proposal. Do not run validators. Do not edit runtime source
of truth files. After writing the proposal packet, return a concise summary.

## Proposal

- Proposal dir: `{proposal_dir}`
- Run dir: `{run_dir}`
- Decision: `{proposal.decision}`
- Target paradigm id: `{proposal.target_paradigm_id}`
- Target taxonomy id: `{proposal.target_taxonomy_id or '-'}`
- Extends: `{proposal.extends or '-'}`
- Title: `{proposal.title}`

## Inputs To Read

- Proposal metadata: `{proposal_dir / 'proposal.json'}`
- Pack draft: `{proposal_dir / 'pack.yaml'}`
- Gap report JSON: `{gap_report_path}`
- Gap report Markdown: `{gap_report_path.with_suffix('.md')}`
- Paper: `{paper_path}`
- Paper map: `{paper_map_path}`
- Current validation report: `{proposal_dir / 'validation_report.json'}`
- Coupling report draft: `{proposal_dir / 'coupling_report.md'}`

{parent_block}

Inherited parent context excerpt:

```json
{inherited_hint or '{}'}
```

## Validation Context

{validation_block}

## Authoring Requirements

For `new_top_level_needed`:

- Fill `pack.yaml` with the small provisional taxonomy node: `fingerprint`,
  `scaffold_hints.interface_hint`, known `priors`, and any concrete
  `semantic_checks` / `smoke_bugs` supported by the paper.
- Do not invent a build plan in this proposal unless the gap report gives
  enough evidence to own codegen semantics. Leave open questions explicit.
- In `smoke_bugs`, route `fix:` values only to known agents such as
  `r2c-architecture-coder`, `r2c-method-coder`, `r2c-notebook-generator`,
  `r2c-stage-reviewer`, `r2c-smoke-diagnostician`, or `human`.

For `new_subparadigm_needed`:

- Inherit the parent by default.
- Override only concrete differences supported by the paper and parent-pack
  comparison.
- REQUIRED — declare `build_plan.source`, the build-context routing choice:
  - `inherit_parent` when the parent's package manifest (its model.py
    classes and training.py functions, shown in the parent context above)
    genuinely describes how THIS paper's method is built.
  - `neutral` when it does not (e.g. an RL method under a classical-planning
    parent). The neutral plan is the generic provisional build shape; a
    delivery built on it is labeled uncertified — new territory, which is
    the honest cost of opting out of family build conventions.
  - Add `reasoning:` grounding the choice in the paper. A missing or
    unexamined choice is exactly how a correct implementation dies at the
    architecture gate against a manifest it can never satisfy.
- REQUIRED — declare `family_components` (see the rule under "For every
  proposal"). Write an explicit empty mapping (`family_components: {{}}`)
  only when the universal contract skeleton fully covers the method; if
  your interface implies components like a reward function, a value
  network, or a replay buffer, declare them.
- Add reviewer checks as `semantic_checks` entries with stable `id`, `stage`,
  `check`, `silent_failure`, `severity`, and `status`.
- Keep `open_questions` for anything not proven by this paper.

For every proposal:

- If the family's architecture contract needs a top-level component block
  beyond the universal skeleton (data_loader / architecture /
  pluggable_component / training_loop), declare it under `family_components`:
  a mapping from component name to `{{required, description,
  required_entries}}`. Declared names are the ONLY legal
  `family_components` keys the Stage 2.d validator accepts in
  arch_contract.json (e.g. a single-agent RL pack declares
  `reward_function`). Declare only components the paper actually needs.
  For a new sub-paradigm the key itself is REQUIRED (an explicit
  `family_components: {{}}` is the "nothing beyond the skeleton" answer):
  a pack with no declaration rejects every family-specific component the
  contract names, however correct.
- If any of your `stage_2x_params` semantic checks requires specific
  parameter entries to EXIST in params.json (required configuration paths,
  a derived statistic the paper defines, or the absence of an
  inapplicable convention param), also declare them under
  `params_derivation` — the deterministic parameter deriver reads that
  block and cannot act on check prose alone. A check without the matching
  declaration halts the run at 2.x on entries no producer can emit.
  Mapping from param name to a kind entry:
  - `{{kind: config_path, reasoning: <what the path configures and where
    the method consumes it>}}` — for pipeline configuration the pluggable
    component requires (optional: `demo_value`, `paper_section`,
    `used_in_notebook`).
  - `{{kind: derived_statistic, formula: "n * E / (K * B)", inputs:
    {{E: params.E, B: params.B, n: spec.<dotted.path>, K:
    spec.<dotted.path>}}}}` — for a paper-defined statistic that must be
    derived and documented; arithmetic only, inputs reference other
    params (`params.<name>`) or structured spec fields
    (`spec.<dotted.path>`); optional `paper_section`, `reasoning`.
  - `{{kind: suppress, reason: <why the param is meaningless here>}}` —
    to drop a cross-paradigm convention param this family does not have
    (e.g. `hidden_dim` on a paradigm with no neural model).
- Ground claims in the paper and gap-report evidence.
- Update `coupling_report.md` with concrete analyzer, scaffolder, architecture,
  method, params, notebook, reviewer, and smoke-diagnostician implications.
- Update `proposal.md` if the review notes or next steps changed.
- Stop after the proposal packet is written.
"""


def author_proposal(
    *,
    run_dir: Path,
    repo_root: Path = ROOT,
    proposal_dir: Path | None = None,
    proposal_id: str | None = None,
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
    dispatch_fn: DispatchFn | None = None,
    allow_update_existing: bool = False,
    dry_run_prompt: bool = False,
    validate_only: bool = False,
    force_new: bool = False,
    overwrite_scaffold: bool = False,
) -> AuthoringResult:
    """Author a proposal packet to a valid state, or return validation errors."""

    run_dir = run_dir.resolve()
    repo_root = repo_root.resolve()
    proposal_path = resolve_or_create_proposal_dir(
        run_dir,
        repo_root,
        proposal_dir=proposal_dir,
        proposal_id=proposal_id,
        overwrite_scaffold=overwrite_scaffold,
        force_new=force_new,
    )
    log_path = proposal_path / AUTHORING_LOG
    _append_log(
        log_path,
        [
            f"# Proposal Authoring Log",
            "",
            f"- Started: {_now()}",
            f"- Proposal dir: `{proposal_path}`",
            f"- Run dir: `{run_dir}`",
            "",
        ],
        reset=True,
    )

    validation = _validate_and_write(
        proposal_path,
        repo_root,
        allow_update_existing=allow_update_existing,
    )
    if validation.valid or validate_only:
        _append_validation_to_log(log_path, 0, validation)
        return _result(proposal_path, validation, 0, log_path)

    if dry_run_prompt:
        prompt = build_author_prompt(
            proposal_dir=proposal_path,
            run_dir=run_dir,
            repo_root=repo_root,
            validation=validation,
        )
        prompt_path = proposal_path / "author_prompt.md"
        prompt_path.write_text(prompt, encoding="utf-8")
        _append_log(
            log_path,
            [
                "## Dry Run",
                "",
                f"- Wrote dispatch prompt: `{prompt_path}`",
                "- No agent was dispatched.",
                "",
            ],
        )
        return _result(proposal_path, validation, 0, log_path, prompt_path)

    if dispatch_fn is None:
        raise AuthoringError("dispatch_fn missing; CLI should provide opencode dispatch")

    iterations_run = 0
    for iteration in range(1, max_iterations + 1):
        iterations_run = iteration
        prompt = build_author_prompt(
            proposal_dir=proposal_path,
            run_dir=run_dir,
            repo_root=repo_root,
            validation=validation,
        )
        prompt_path = proposal_path / f"author_prompt_iteration_{iteration}.md"
        prompt_path.write_text(prompt, encoding="utf-8")
        before = _snapshot_guarded_scope(repo_root, run_dir, proposal_path)
        started = time.time()
        _append_log(
            log_path,
            [
                f"## Iteration {iteration}",
                "",
                f"- Prompt: `{prompt_path}`",
                f"- Validation before dispatch: invalid ({len(validation.errors)} error(s), {len(validation.warnings)} warning(s))",
                "",
            ],
        )
        dispatch_result = dispatch_fn(prompt)
        elapsed = time.time() - started
        violations = _scope_violations(
            before,
            _snapshot_guarded_scope(repo_root, run_dir, proposal_path),
        )
        if violations:
            _append_log(
                log_path,
                [
                    "- Dispatch completed, but out-of-scope writes were detected.",
                    *[f"  - `{item}`" for item in violations],
                    "",
                ],
            )
            raise AuthoringError(
                "taxonomy-pack author wrote outside the proposal scope: "
                + ", ".join(violations[:10])
            )
        _append_log(
            log_path,
            [
                f"- Dispatch assistant message: `{dispatch_result.assistant_message_id}`",
                f"- Dispatch elapsed: {elapsed:.1f}s",
                "",
            ],
        )

        validation = _validate_and_write(
            proposal_path,
            repo_root,
            allow_update_existing=allow_update_existing,
        )
        _append_validation_to_log(log_path, iteration, validation)
        if validation.valid:
            break

    return _result(proposal_path, validation, iterations_run, log_path)


def _load_proposal(proposal_dir: Path) -> ParadigmProposal:
    return ParadigmProposal.model_validate(
        json.loads((proposal_dir / "proposal.json").read_text(encoding="utf-8"))
    )


def _validate_and_write(
    proposal_dir: Path,
    repo_root: Path,
    *,
    allow_update_existing: bool,
) -> ProposalValidationResult:
    validation = validate_proposal(
        proposal_dir,
        repo_root,
        allow_update_existing=allow_update_existing,
    )
    (proposal_dir / "validation_report.json").write_text(
        json.dumps(validation.model_dump(mode="json"), indent=2) + "\n",
        encoding="utf-8",
    )
    return validation


def _result(
    proposal_dir: Path,
    validation: ProposalValidationResult,
    iterations: int,
    log_path: Path,
    dry_run_prompt_path: Path | None = None,
) -> AuthoringResult:
    return AuthoringResult(
        proposal_dir=proposal_dir,
        valid=validation.valid,
        iterations=iterations,
        errors=list(validation.errors),
        warnings=list(validation.warnings),
        validation_report_path=proposal_dir / "validation_report.json",
        authoring_log_path=log_path,
        dry_run_prompt_path=dry_run_prompt_path,
    )


def _validation_prompt_block(validation: ProposalValidationResult | None) -> str:
    if validation is None:
        return "No validation report was supplied. Author the packet from the scaffold."
    lines = [
        f"- Current validation status: `{'valid' if validation.valid else 'invalid'}`",
        f"- Error count: {len(validation.errors)}",
        f"- Warning count: {len(validation.warnings)}",
    ]
    if validation.errors:
        lines.extend(["", "Errors to fix:"])
        lines.extend(f"- {error}" for error in validation.errors)
    if validation.warnings:
        lines.extend(["", "Warnings to consider:"])
        lines.extend(f"- {warning}" for warning in validation.warnings)
    return "\n".join(lines)


def _append_log(path: Path, lines: list[str], *, reset: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join(lines).rstrip() + "\n"
    if reset:
        path.write_text(text, encoding="utf-8")
    else:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(text)


def _append_validation_to_log(
    path: Path,
    iteration: int,
    validation: ProposalValidationResult,
) -> None:
    label = "Initial Validation" if iteration == 0 else f"Validation After Iteration {iteration}"
    lines = [
        f"## {label}",
        "",
        f"- Status: `{'valid' if validation.valid else 'invalid'}`",
        f"- Errors: {len(validation.errors)}",
        f"- Warnings: {len(validation.warnings)}",
    ]
    if validation.errors:
        lines.extend(["", "Errors:"])
        lines.extend(f"- {error}" for error in validation.errors)
    if validation.warnings:
        lines.extend(["", "Warnings:"])
        lines.extend(f"- {warning}" for warning in validation.warnings)
    lines.append("")
    _append_log(path, lines)


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _snapshot_guarded_scope(
    repo_root: Path,
    run_dir: Path,
    proposal_dir: Path,
) -> dict[str, FileState]:
    """Snapshot files the author agent is not allowed to edit."""

    roots = [
        repo_root / "docs",
        repo_root / "paradigms",
        repo_root / "schemas",
        repo_root / "scripts",
        repo_root / ".opencode" / "agents",
        run_dir,
    ]
    snapshot: dict[str, FileState] = {}
    for root in roots:
        if not root.exists():
            continue
        for path in _iter_files(root):
            if _is_allowed_proposal_path(path, proposal_dir):
                continue
            try:
                rel = path.relative_to(repo_root).as_posix()
            except ValueError:
                rel = path.as_posix()
            stat_result = path.stat()
            snapshot[rel] = FileState(
                size=stat_result.st_size,
                mtime_ns=stat_result.st_mtime_ns,
            )
    return snapshot


def _iter_files(root: Path):
    excluded_dir_names = {
        ".git",
        ".pytest_cache",
        "__pycache__",
        "node_modules",
    }
    for path in sorted(root.rglob("*")):
        if any(part in excluded_dir_names for part in path.parts):
            continue
        if path.is_file():
            yield path


def _is_allowed_proposal_path(path: Path, proposal_dir: Path) -> bool:
    try:
        path.resolve().relative_to(proposal_dir.resolve())
        return True
    except ValueError:
        return False


def _is_driver_managed_run_artifact(rel_path: str) -> bool:
    """Files the DRIVER (not the author agent) writes during a dispatch.

    The scope snapshot keys are REPO-relative for real runs
    (r2c_runs/<slug>/.pipeline/run_events.jsonl) while this exemption
    originally matched run-relative paths only — so it never matched in
    production, and it also missed progress.json (rewritten by the
    driver on every event append). Found live 2026-07-06: the fedavg
    fresh run's authoring was killed for "writing" the driver's own
    event log the moment the (separately fixed) event registration made
    those appends succeed mid-dispatch. Normalize to the path segment
    after the last '.pipeline/' so both forms match."""
    marker = ".pipeline/"
    idx = rel_path.rfind(marker)
    if idx == -1:
        return False
    tail = rel_path[idx + len(marker):]
    return (
        tail == "driver_state.json"
        or tail == "run_events.jsonl"
        or tail == "progress.json"
        or tail.startswith("run_events.jsonl.lock")
        or tail.startswith("_lock")
    )


def _scope_violations(
    before: dict[str, FileState],
    after: dict[str, FileState],
) -> list[str]:
    before_keys = set(before)
    after_keys = set(after)
    changed = [
        key for key in sorted(before_keys & after_keys)
        if before[key] != after[key]
        and not _is_driver_managed_run_artifact(key)
    ]
    added = [
        f"{key} (created)" for key in sorted(after_keys - before_keys)
        if not _is_driver_managed_run_artifact(key)
    ]
    deleted = [
        f"{key} (deleted)" for key in sorted(before_keys - after_keys)
        if not _is_driver_managed_run_artifact(key)
    ]
    return changed + added + deleted


def _detect_session_deadlock_risk() -> tuple[bool, str]:
    """Best-effort check for foreground invocation inside an opencode bash tool."""

    if os.environ.get("R2C_SKIP_DEADLOCK_CHECK"):
        return False, "R2C_SKIP_DEADLOCK_CHECK set"
    try:
        if any(os.isatty(fd) for fd in (0, 1, 2)):
            return False, "tty present on stdin/stdout/stderr"
    except OSError:
        return False, "tty probe failed"
    if sys.platform == "win32":
        return False, "windows: probe not implemented"

    found_opencode_pid: int | None = None
    pid = os.getppid()
    seen: set[int] = set()
    for _ in range(20):
        if pid in seen or pid <= 1:
            break
        seen.add(pid)
        try:
            proc = subprocess.run(
                ["ps", "-p", str(pid), "-o", "comm=,ppid="],
                capture_output=True,
                text=True,
                timeout=2,
            )
        except (subprocess.SubprocessError, FileNotFoundError, OSError):
            return False, "ps probe failed"
        if proc.returncode != 0:
            break
        line = proc.stdout.strip()
        if not line:
            break
        parts = line.rsplit(None, 1)
        if len(parts) != 2:
            break
        comm, ppid_str = parts
        if "opencode" in comm.lower():
            found_opencode_pid = pid
            break
        try:
            pid = int(ppid_str)
        except ValueError:
            break
    if found_opencode_pid is None:
        return False, "no opencode in ancestor chain"

    try:
        st_out = os.fstat(1)
    except OSError:
        return False, "stdout stat failed"
    if stat.S_ISREG(st_out.st_mode):
        return False, "stdout is a regular file"
    try:
        devnull_st = os.stat(os.devnull)
        if stat.S_ISCHR(st_out.st_mode) and st_out.st_rdev == devnull_st.st_rdev:
            return False, "stdout is /dev/null"
    except OSError:
        pass
    return True, (
        f"opencode ancestor (pid={found_opencode_pid}); no TTY; stdout is a pipe"
    )


def _check_for_session_deadlock_or_exit() -> None:
    is_risk, diag = _detect_session_deadlock_risk()
    if not is_risk:
        return
    sys.stderr.write(
        "\n"
        "ERROR: author_field_guide_proposal.py appears to be running synchronously\n"
        "inside an opencode bash tool. This command dispatches an agent back into\n"
        "the same opencode session and would deadlock while the bash tool holds\n"
        "the session lock.\n\n"
        f"  Detection: {diag}\n\n"
        "Run this command from a separate terminal connected to the same repo, or\n"
        "use --dry-run-prompt to create a prompt without dispatching. If this is a\n"
        "false positive, set R2C_SKIP_DEADLOCK_CHECK=1.\n"
    )
    sys.exit(2)


def _make_opencode_dispatch_fn(
    *,
    session_id: str,
    port: int,
    model: dict | None,
    timeout_s: int,
) -> DispatchFn:
    def _dispatch(prompt: str) -> DispatchResult:
        return dispatch_and_wait(
            session_id=session_id,
            agent=AUTHOR_AGENT,
            prompt=prompt,
            model=model,
            port=port,
            timeout_s=timeout_s,
        )

    return _dispatch


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    parser.add_argument("--proposal-dir", type=Path)
    parser.add_argument("--proposal-id")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--session-id")
    parser.add_argument("--dir", default=os.getcwd(), help="workspace dir for session discovery")
    parser.add_argument("--timeout-s", type=int, default=DEFAULT_TIMEOUT_S)
    parser.add_argument("--max-iterations", type=int, default=DEFAULT_MAX_ITERATIONS)
    parser.add_argument("--allow-update-existing", action="store_true")
    parser.add_argument("--force-new", action="store_true")
    parser.add_argument("--overwrite-scaffold", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--dry-run-prompt", action="store_true")
    args = parser.parse_args(argv)

    try:
        if args.max_iterations < 1:
            raise AuthoringError("--max-iterations must be >= 1")

        dispatch_fn: DispatchFn | None = None
        if not args.validate_only and not args.dry_run_prompt:
            _check_for_session_deadlock_or_exit()
            if not health_check(port=args.port):
                raise ServerUnreachable(f"opencode server not reachable on port {args.port}")
            session_id = args.session_id or discover_session(
                directory=str(Path(args.dir).resolve()),
                port=args.port,
            )
            model = get_agent_model(AUTHOR_AGENT, port=args.port)
            dispatch_fn = _make_opencode_dispatch_fn(
                session_id=session_id,
                port=args.port,
                model=model,
                timeout_s=args.timeout_s,
            )

        result = author_proposal(
            run_dir=args.run_dir,
            repo_root=args.repo_root,
            proposal_dir=args.proposal_dir,
            proposal_id=args.proposal_id,
            max_iterations=args.max_iterations,
            dispatch_fn=dispatch_fn,
            allow_update_existing=args.allow_update_existing,
            dry_run_prompt=args.dry_run_prompt,
            validate_only=args.validate_only,
            force_new=args.force_new,
            overwrite_scaffold=args.overwrite_scaffold,
        )
    except (AuthoringError, OpencodeClientError, SessionNotFound, ServerUnreachable) as exc:
        print(f"authoring FAILED: {exc}", file=sys.stderr)
        return 2

    if result.dry_run_prompt_path is not None:
        print(f"author prompt written: {result.dry_run_prompt_path}")
        print(f"current validation report: {result.validation_report_path}")
        return 0

    if result.valid:
        print(f"proposal OK: {result.proposal_dir}")
        print(f"iterations: {result.iterations}")
        print(f"validation report: {result.validation_report_path}")
        print(f"authoring log: {result.authoring_log_path}")
        print(f"promote only after review: python3 scripts/apply_paradigm_proposal.py {result.proposal_dir}")
        return 0

    print(f"proposal still invalid after {result.iterations} iteration(s): {result.proposal_dir}", file=sys.stderr)
    for error in result.errors:
        print(f"  - {error}", file=sys.stderr)
    for warning in result.warnings:
        print(f"  warning: {warning}", file=sys.stderr)
    print(f"validation report: {result.validation_report_path}", file=sys.stderr)
    print(f"authoring log: {result.authoring_log_path}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
