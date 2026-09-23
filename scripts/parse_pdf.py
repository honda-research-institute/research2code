#!/usr/bin/env python3
"""
Parse a PDF into Markdown with Marker (https://github.com/datalab-to/marker),
run locally on this machine.

Marker is an optional dependency: `pip install marker-pdf` puts the
`marker_single` command on PATH (set R2C_MARKER_BIN to point elsewhere).
Markdown (.md) paper inputs need none of this. Marker's first run downloads
its models (several GB); a CPU parse of a paper takes minutes, so the
per-attempt budget is generous. Marker's own LLM assist (`--use_llm`, which
uses Marker's provider configuration, Gemini via GOOGLE_API_KEY by default)
is controlled by R2C_MARKER_USE_LLM and DEFAULTS TO FALSE: it is slower,
needs a key, and plain Marker output is what the pipeline was built on.

Usage:
    python scripts/parse_pdf.py <input.pdf> --output <output.md>
"""

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict
from pathlib import Path

from trusted import trusted_path, trusted_text
from pdf_parse_quality import (
    ParseQuality,
    assess_parse_quality,
    default_min_words,
    write_quality_report,
)

MARKER_BIN_ENV = "R2C_MARKER_BIN"
DEFAULT_MARKER_BIN = "marker_single"
# Marker's inference backend; Marker itself honors this env var.
LLAMA_BIN_ENV = "LLAMA_CPP_BINARY"
DEFAULT_LLAMA_BIN = "llama-server"
# Exit code for "the parser is not installed", distinct from a bad parse, so
# the driver's halt can say which one it was.
PARSER_MISSING_EXIT = 3
INSTALL_RECIPE = (
    "install it once: `pip install marker-pdf` (Python 3.10-3.13; on a newer "
    "interpreter use a separate venv and set R2C_MARKER_BIN to its "
    "marker_single) and `brew install llama.cpp` (Linux: "
    "https://github.com/ggml-org/llama.cpp/releases), see .env.example. "
    "PDF inputs cannot be parsed without it; markdown (.md) paper inputs "
    "are unaffected."
)
USE_LLM_ENV = "R2C_MARKER_USE_LLM"
DEFAULT_USE_LLM = False
# OCR-ing every page is slow on CPU and papers are digital PDFs; opt in when a
# PDF's text layer is broken.
FORCE_OCR_ENV = "R2C_MARKER_FORCE_OCR"
DEFAULT_FORCE_OCR = False
DEFAULT_RETRIES = 1
DEFAULT_RETRY_DELAY_S = 2.0
# Per-attempt wall clock. Bounds a hung or runaway parse so the retry loop
# gets its turn; the driver's stage-0 budget is derived from these values
# (total_budget_s), so they never disagree. 0 disables.
DEFAULT_ATTEMPT_TIMEOUT_S = 1500.0


class ParsePDFError(RuntimeError):
    def __init__(
        self, message: str, *, quality: ParseQuality | None = None
    ) -> None:
        super().__init__(message)
        self.quality = quality


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return value if value >= 0 else default


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        value = float(raw)
    except ValueError:
        return default
    return value if value >= 0 else default


def _env_flag(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    value = raw.strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    return default


def resolve_use_llm() -> bool:
    """LLM-assist flag from R2C_MARKER_USE_LLM; unset/empty/unrecognized
    falls back to DEFAULT_USE_LLM (False — see the module docstring)."""
    return _env_flag(USE_LLM_ENV, DEFAULT_USE_LLM)


def resolve_force_ocr() -> bool:
    return _env_flag(FORCE_OCR_ENV, DEFAULT_FORCE_OCR)


def resolve_marker_bin() -> str:
    """Absolute path of the Marker CLI, or a ParsePDFError that tells the
    researcher how to install it."""
    wanted = os.environ.get(MARKER_BIN_ENV) or DEFAULT_MARKER_BIN
    found = shutil.which(wanted)
    if not found:
        raise ParsePDFError(
            f"Marker is not installed ({wanted!r} not found); {INSTALL_RECIPE}")
    return found


def preflight() -> str:
    """Everything a local Marker run needs, checked before any work: the
    marker_single CLI and llama.cpp's llama-server (Marker's inference
    backend, which it otherwise reports only minutes into a parse)."""
    binary = resolve_marker_bin()
    llama = os.environ.get(LLAMA_BIN_ENV) or DEFAULT_LLAMA_BIN
    if not shutil.which(llama):
        raise ParsePDFError(
            f"Marker's inference backend is not installed ({llama!r} not "
            f"found); {INSTALL_RECIPE}")
    return binary


def total_budget_s(
    retries: int | None = None, attempt_timeout_s: float | None = None,
) -> float:
    """Wall clock the whole parse (all attempts) may take; the driver uses
    it as the stage-0 subprocess budget. 0 attempt timeout means unbounded,
    reported as a day so a caller can still pass a number."""
    retries = _env_int("R2C_MARKER_RETRIES", DEFAULT_RETRIES) if retries is None else retries
    if attempt_timeout_s is None:
        attempt_timeout_s = _env_float(
            "R2C_MARKER_ATTEMPT_TIMEOUT_S", DEFAULT_ATTEMPT_TIMEOUT_S)
    if attempt_timeout_s <= 0:
        return 24 * 3600.0
    delay = _env_float("R2C_MARKER_RETRY_DELAY_S", DEFAULT_RETRY_DELAY_S)
    return (retries + 1) * attempt_timeout_s + retries * delay + 60.0


def _describe_error(exc: BaseException | None) -> str:
    """Render an exception for a retry/summary message. str() first, repr()
    when str() is empty (a bare exception carries no message)."""
    if exc is None:
        return "unknown error"
    text = str(exc).strip()
    return text or repr(exc)


def _call_marker(pdf_path: str, use_llm: bool, timeout_s: float) -> str:
    """One local Marker run: `marker_single` into a scratch directory,
    bounded by `timeout_s` (0 = unbounded). Returns the markdown text."""
    binary = resolve_marker_bin()
    name = Path(pdf_path).name
    with tempfile.TemporaryDirectory(prefix="r2c-marker-") as out_dir:
        cmd = [
            str(trusted_path(binary)), trusted_text(pdf_path),
            "--output_dir", out_dir,
            "--output_format", "markdown",
            # Figures are not consumed downstream; the pipeline strips
            # embedded image payloads from paper.md anyway.
            "--disable_image_extraction",
        ]
        if resolve_force_ocr():
            cmd.append("--force_ocr")
        if use_llm:
            cmd.append("--use_llm")
        try:
            proc = subprocess.run(
                cmd, capture_output=True, text=True,
                timeout=timeout_s if timeout_s > 0 else None,
            )
        except subprocess.TimeoutExpired as exc:
            raise ParsePDFError(
                f"Marker timed out after {timeout_s:.0f}s on {name} (the "
                f"attempt was killed). Marker's first run also downloads its "
                f"models; rerun, or raise R2C_MARKER_ATTEMPT_TIMEOUT_S."
            ) from exc
        if proc.returncode != 0:
            # Last lines only, on line boundaries: the actionable message
            # (a missing backend, an install hint) sits at the end.
            lines = (proc.stderr or proc.stdout or "").strip().splitlines()
            tail = "\n".join(lines[-12:])
            raise ParsePDFError(
                f"marker_single exited {proc.returncode} on {name}:\n{tail}"
            )
        outputs = sorted(Path(out_dir).rglob("*.md"))
        if not outputs:
            raise ParsePDFError(
                f"Marker produced no markdown for {name} (exit 0, output dir "
                f"empty): {(proc.stderr or '').strip()[-500:]}"
            )
        return outputs[0].read_text(encoding="utf-8")


def parse_pdf(pdf_path: str, *, use_llm: bool | None = None) -> str:
    """Parse a PDF with local Marker and return Markdown (no retries)."""
    resolved_use_llm = resolve_use_llm() if use_llm is None else use_llm
    return _call_marker(pdf_path, resolved_use_llm, 0.0)


def parse_pdf_with_retries(
    pdf_path: str, *,
    retries: int | None = None,
    retry_delay_s: float | None = None,
    min_words: int | None = None,
    attempt_timeout_s: float | None = None,
    use_llm: bool | None = None,
) -> tuple[str, ParseQuality, int]:
    """Parse a PDF, retrying transient failures and unusable parses."""
    resolved_use_llm = resolve_use_llm() if use_llm is None else use_llm
    retries = DEFAULT_RETRIES if retries is None else max(0, retries)
    retry_delay_s = (
        DEFAULT_RETRY_DELAY_S if retry_delay_s is None else max(0.0, retry_delay_s)
    )
    min_words = (
        min_words
        if min_words is not None and min_words > 0
        else default_min_words()
    )
    if attempt_timeout_s is None:
        attempt_timeout_s = _env_float(
            "R2C_MARKER_ATTEMPT_TIMEOUT_S", DEFAULT_ATTEMPT_TIMEOUT_S
        )
    attempts = retries + 1
    last_error: Exception | None = None
    last_quality: ParseQuality | None = None
    for attempt in range(1, attempts + 1):
        try:
            markdown = _call_marker(pdf_path, resolved_use_llm, attempt_timeout_s)
            quality = assess_parse_quality(markdown, min_words=min_words)
            if quality.passed:
                return markdown, quality, attempt
            last_quality = quality
            last_error = ParsePDFError(
                "PDF parse quality gate failed: " + "; ".join(quality.reasons),
                quality=quality,
            )
        except Exception as exc:  # noqa: BLE001 - retries own this boundary
            last_error = exc
        if attempt < attempts:
            print(
                f"Attempt {attempt}/{attempts} failed: "
                f"{_describe_error(last_error)}",
                file=sys.stderr,
            )
            if retry_delay_s:
                time.sleep(retry_delay_s)
    if isinstance(last_error, ParsePDFError):
        raise last_error
    raise ParsePDFError(
        f"PDF parse failed after {attempts} attempt(s): "
        f"{_describe_error(last_error)}",
        quality=last_quality,
    )


def main():
    # Repo-root .env first (never overrides the shell) — standalone runs
    # get the same configuration story as driver-spawned ones.
    from env_file import load_env_file  # local: keep import cost off parse_pdf consumers
    load_env_file()
    parser = argparse.ArgumentParser(
        description="Parse a PDF to Markdown with Marker, run locally."
    )
    parser.add_argument("pdf_path", help="Path to the input PDF file")
    parser.add_argument(
        "--output", "-o", required=True, help="Output Markdown file path"
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=_env_int("R2C_MARKER_RETRIES", DEFAULT_RETRIES),
        help="Retry count after the first parse attempt.",
    )
    parser.add_argument(
        "--retry-delay",
        type=float,
        default=_env_float("R2C_MARKER_RETRY_DELAY_S", DEFAULT_RETRY_DELAY_S),
        help="Seconds to wait between parse attempts.",
    )
    parser.add_argument(
        "--attempt-timeout",
        type=float,
        default=_env_float(
            "R2C_MARKER_ATTEMPT_TIMEOUT_S", DEFAULT_ATTEMPT_TIMEOUT_S
        ),
        help="Per-attempt wall-clock timeout in seconds (0 disables). A hung "
             "Marker run is killed so remaining attempts can run.",
    )
    parser.add_argument(
        "--min-words",
        type=int,
        default=default_min_words(),
        help="Minimum word count required for a usable paper parse.",
    )
    parser.add_argument(
        "--quality-report",
        type=Path,
        help="Write JSON parse-quality report here.",
    )
    args = parser.parse_args()

    pdf_path = Path(args.pdf_path)
    if not pdf_path.exists():
        print(f"ERROR: File not found: {pdf_path}", file=sys.stderr)
        sys.exit(1)

    try:
        marker_bin = preflight()
    except ParsePDFError as exc:
        # The not-installed story must reach stage_0's stderr tail as one
        # plain line, not a traceback (it is rendered to the researcher),
        # and its own exit code so the driver's halt names the cause.
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(PARSER_MISSING_EXIT)
    use_llm = resolve_use_llm()
    retries = max(0, args.retries)
    retry_delay = max(0.0, args.retry_delay)
    attempt_timeout = max(0.0, args.attempt_timeout)
    print(
        f"Parsing: {pdf_path.name} (via {marker_bin}, "
        f"llm_assist={'on' if use_llm else 'off'}, "
        f"force_ocr={'on' if resolve_force_ocr() else 'off'}, "
        f"{retries + 1} attempt(s) max, "
        f"{attempt_timeout:.0f}s per attempt)"
    )
    t0 = time.perf_counter()
    try:
        markdown, quality, attempt = parse_pdf_with_retries(
            str(pdf_path),
            retries=retries,
            retry_delay_s=retry_delay,
            min_words=args.min_words,
            attempt_timeout_s=attempt_timeout,
        )
    except ParsePDFError as exc:
        if args.quality_report and exc.quality is not None:
            write_quality_report(
                args.quality_report,
                exc.quality,
                source_pdf=pdf_path,
            )
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
    parse_time = time.perf_counter() - t0

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(markdown)
    if args.quality_report:
        write_quality_report(args.quality_report, quality, source_pdf=pdf_path)

    char_count = len(markdown)
    print(
        f"  Parsed in {parse_time:.1f}s on attempt {attempt} — "
        f"{quality.word_count} words, {char_count} chars"
    )
    print(f"  Quality: {asdict(quality)}")
    print(f"  Output: {output_path}")


if __name__ == "__main__":
    main()
