from __future__ import annotations

import re

MAX_QUERY_LENGTH = 1000
DEFAULT_QUERY_CONTENT = "No content available for this document."


def normalize_query_text(text: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9 ]", " ", text or "")
    cleaned = re.sub(r"\s+", " ", cleaned)
    cleaned = cleaned.strip()
    if not cleaned:
        cleaned = DEFAULT_QUERY_CONTENT
    return cleaned[:MAX_QUERY_LENGTH]
