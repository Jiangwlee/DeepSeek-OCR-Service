from __future__ import annotations

import re

# Matches layout/tag lines like `sub_title[[...]]` or `text[[...]]`
_LAYOUT_LINE_RE = re.compile(r"(?m)^\s*[A-Za-z0-9_]+\[\[[^\n]*?\]\]\s*(?:\n|$)")
# Matches inline DeepSeek reference tokens such as <|ref|>...<|/ref|>
_INLINE_TAG_RE = re.compile(r"<\|/?(?:ref|det)\|>")


def strip_layout_tags(text: str) -> str:
    """Remove DeepSeek layout helper tags from OCR output."""
    without_layout_lines = _LAYOUT_LINE_RE.sub("", text)
    cleaned = _INLINE_TAG_RE.sub("", without_layout_lines)
    return cleaned
