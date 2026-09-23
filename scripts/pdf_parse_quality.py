#!/usr/bin/env python3
"""Quality gate for PDF-to-Markdown parse output.

The gate is intentionally simple and deterministic. It catches the cases that
make the rest of the pipeline unusable: empty parses, service-error pages, and
image-only OCR output with too little real text.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path


DEFAULT_MIN_WORDS = 250

_HTML_DATA_IMAGE_RE = re.compile(
    r"<img\b[^>]*\bsrc=[\"']data:image/[^\"']+[\"'][^>]*>",
    re.IGNORECASE,
)
_MD_DATA_IMAGE_RE = re.compile(
    r"!\[[^\]]*\]\(data:image/[^)]+\)",
    re.IGNORECASE,
)
_HTML_TAG_RE = re.compile(r"<[^>]+>")
_WORD_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_'-]*")
_ERROR_SNIPPETS = tuple(
    s.lower()
    for s in (
        "Traceback (most recent call last)",
        "Internal Server Error",
        "Bad Gateway",
        "Application error",
    )
)


@dataclass
class ParseQuality:
    passed: bool
    word_count: int
    char_count: int
    visible_char_count: int
    data_image_count: int
    data_image_chars: int
    min_words: int
    reasons: list[str]


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return value if value > 0 else default


def default_min_words() -> int:
    return _env_int("R2C_PDF_MIN_WORDS", DEFAULT_MIN_WORDS)


def _normalize_min_words(value: int | None) -> int:
    if value is None or value <= 0:
        return default_min_words()
    return value


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


_DATA_URI_RE = re.compile(r"data:image/[^)\"'\s]+", re.IGNORECASE)


def replace_payloads_with_placeholders(markdown: str) -> tuple[str, int]:
    """Swap each data-URI image payload for a figure-N.png placeholder.

    Figure positions survive (tag/alt structure untouched) and the prose is
    byte-identical outside the payload URIs. Idempotent: a stripped document
    has no data-URI left to match.
    """
    count = 0

    def repl(match: re.Match[str]) -> str:
        nonlocal count
        count += 1
        return f"figure-{count}.png"

    return _DATA_URI_RE.sub(repl, markdown), count


def strip_embedded_images(markdown: str) -> tuple[str, int, int]:
    """Remove data-URI image payloads while preserving alt text if present."""
    data_image_chars = 0
    data_image_count = 0

    def replace_html(match: re.Match[str]) -> str:
        nonlocal data_image_chars, data_image_count
        data_image_count += 1
        data_image_chars += len(match.group(0))
        alt = re.search(r"\balt=[\"']([^\"']*)[\"']", match.group(0), re.IGNORECASE)
        label = alt.group(1).strip() if alt else ""
        if label:
            return f"\n{label}\n"
        return "\n\n"

    def replace_markdown(match: re.Match[str]) -> str:
        nonlocal data_image_chars, data_image_count
        data_image_count += 1
        data_image_chars += len(match.group(0))
        alt = re.match(r"!\[([^\]]*)\]", match.group(0))
        label = alt.group(1).strip() if alt else ""
        if label:
            return f"\n{label}\n"
        return "\n\n"

    stripped = _HTML_DATA_IMAGE_RE.sub(replace_html, markdown)
    stripped = _MD_DATA_IMAGE_RE.sub(replace_markdown, stripped)
    return stripped, data_image_count, data_image_chars


def assess_parse_quality(markdown: str, *, min_words: int | None = None) -> ParseQuality:
    min_words = _normalize_min_words(min_words)
    stripped, image_count, image_chars = strip_embedded_images(markdown)
    visible = _HTML_TAG_RE.sub(" ", stripped)
    words = _WORD_RE.findall(visible)
    visible_chars = len("".join(visible.split()))
    reasons: list[str] = []
    if not markdown.strip():
        reasons.append("parse output is empty")
    lower_markdown = markdown.lower()
    starts_like_error = any(
        snippet in lower_markdown.lstrip()[:500]
        for snippet in _ERROR_SNIPPETS
    )
    if starts_like_error and len(words) < min_words:
        reasons.append("parse output looks like a service error page")
    if len(words) < min_words:
        reasons.append(
            f"parse output has {len(words)} words, below the {min_words} word minimum"
        )
    return ParseQuality(
        passed=not reasons,
        word_count=len(words),
        char_count=len(markdown),
        visible_char_count=visible_chars,
        data_image_count=image_count,
        data_image_chars=image_chars,
        min_words=min_words,
        reasons=reasons,
    )


def quality_report_payload(
    quality: ParseQuality, *, source_pdf: Path | None = None
) -> dict[str, object]:
    payload: dict[str, object] = asdict(quality)
    if source_pdf is not None:
        resolved = source_pdf.resolve()
        payload["source_pdf_path"] = str(resolved)
        payload["source_pdf_sha256"] = (
            file_sha256(resolved) if resolved.is_file() else None
        )
    return payload


def write_quality_report(
    path: Path, quality: ParseQuality, *, source_pdf: Path | None = None
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(quality_report_payload(quality, source_pdf=source_pdf), indent=2)
        + "\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("markdown_path", type=Path)
    parser.add_argument("--report", type=Path, help="Write JSON quality report here.")
    parser.add_argument("--source-pdf", type=Path, help="PDF source for provenance.")
    parser.add_argument(
        "--min-words",
        type=int,
        default=default_min_words(),
        help="Minimum word count required for a usable paper parse.",
    )
    args = parser.parse_args()

    if not args.markdown_path.is_file():
        print(f"ERROR: file not found: {args.markdown_path}", file=sys.stderr)
        return 1
    markdown = args.markdown_path.read_text(encoding="utf-8")
    quality = assess_parse_quality(markdown, min_words=args.min_words)
    if args.report:
        write_quality_report(args.report, quality, source_pdf=args.source_pdf)
    print(json.dumps(
        quality_report_payload(quality, source_pdf=args.source_pdf),
        indent=2,
    ))
    if not quality.passed:
        print("ERROR: PDF parse quality gate failed", file=sys.stderr)
        for reason in quality.reasons:
            print(f"- {reason}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
