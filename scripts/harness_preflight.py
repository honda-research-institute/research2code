"""Capability preflight for the portable check harness (R2C-019).

Dependency-free (stdlib; torch probed, never required). Vendored into
every built harness and runs at run_checks.py start: it states what
paper-scale reproduction needs (the paper's verbatim hardware statement
or an honest absence, plus the spec's compute class and time estimate),
what this machine has, and refuses paper-scale expectations loudly when
the machine cannot meet them. The battery itself always proceeds — it
verifies the delivery at demo scale, which is CPU-sized by design; the
refusal is about the paper-scale claim, not the demo checks.

No inventions: every requirement line is either the paper's own sentence,
the analyzer's structured compute class, or a measurement of THIS
machine. A missing statement renders as "not stated in the paper".
"""

from __future__ import annotations

import json
import os
import platform
import sys
from pathlib import Path

GPU_CLASSES = ("Single GPU required", "Multi-GPU required")


def machine_capabilities() -> dict:
    """What this machine offers, measured, never guessed."""
    caps: dict = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "ram_gb": None,
        "torch": None,
        "cuda_available": None,
    }
    try:
        pages = os.sysconf("SC_PHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
        caps["ram_gb"] = round(pages * page_size / (1024 ** 3), 1)
    except (ValueError, OSError, AttributeError):
        pass
    try:
        import torch  # noqa: PLC0415

        caps["torch"] = str(getattr(torch, "__version__", "unknown"))
        caps["cuda_available"] = bool(torch.cuda.is_available())
    except Exception:  # noqa: BLE001 - torch is optional everywhere here
        pass
    return caps


def render_preflight(hardware_block: dict, caps: dict) -> str:
    """The researcher-facing preflight text, quotes and measurements only."""
    compute_class = str(hardware_block.get("compute_class") or "")
    estimated_time = str(hardware_block.get("estimated_time") or "")
    statement = hardware_block.get("paper_statement")

    lines = ["== Capability preflight =="]
    lines.append(f"Paper-scale compute class: "
                 f"{compute_class or 'not recorded'}.")
    if isinstance(statement, dict) and statement.get("text"):
        section = statement.get("section")
        where = f" ({section})" if section else ""
        lines.append(f"The paper states{where}: "
                     f"\"{statement['text']}\"")
    else:
        lines.append("The paper does not state its hardware.")
    if estimated_time:
        lines.append(f"Paper-scale time estimate: {estimated_time}.")

    machine = (f"This machine: {caps.get('cpu_count')} CPU core(s)"
               + (f", {caps['ram_gb']} GB RAM" if caps.get("ram_gb") else "")
               + (f", torch {caps['torch']}" if caps.get("torch")
                  else ", no torch")
               + (", CUDA GPU available" if caps.get("cuda_available")
                  else ", no CUDA GPU"))
    lines.append(machine + ".")

    needs_gpu = compute_class in GPU_CLASSES
    if needs_gpu and not caps.get("cuda_available"):
        need = (statement.get("text") if isinstance(statement, dict)
                and statement.get("text") else compute_class)
        lines.append(
            f"REFUSED: paper-scale reproduction needs {compute_class} and "
            f"this machine has no CUDA GPU. What it would take: {need}. "
            "The checks below still run — they verify this delivery at "
            "demo scale, which is CPU-sized by design.")
    else:
        lines.append(
            "Note: this deliverable's code runs at demo scale. Paper-scale "
            "reproduction is not included in the delivery and is not "
            "attempted by these checks.")
    return "\n".join(lines)


def run_preflight(harness_dir: Path, output_dir: Path | None) -> str:
    """Load the frozen hardware block, render, persist, and return the
    text. Missing or unparseable blocks degrade to honest absence lines —
    the preflight never blocks the battery."""
    hardware_block: dict = {}
    block_path = Path(harness_dir) / "hardware_block.json"
    if block_path.is_file():
        try:
            loaded = json.loads(block_path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                hardware_block = loaded
        except json.JSONDecodeError:
            pass
    caps = machine_capabilities()
    text = render_preflight(hardware_block, caps)
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "preflight.json").write_text(
            json.dumps({"hardware_block": hardware_block,
                        "machine": caps, "rendered": text},
                       indent=2, sort_keys=True) + "\n",
            encoding="utf-8")
    return text


def main() -> int:  # pragma: no cover - tiny CLI shim for manual use
    harness_dir = Path(__file__).resolve().parent
    print(run_preflight(harness_dir, None))
    return 0


if __name__ == "__main__":
    sys.exit(main())
