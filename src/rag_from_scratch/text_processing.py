from __future__ import annotations

import re
from collections import Counter


WORD_PATTERN = re.compile(r"[A-Za-z0-9_\-]+")


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip())


def chunk_text(text: str, max_chars: int = 900, overlap: int = 120) -> list[str]:
    text = normalize_text(text)
    if len(text) <= max_chars:
        return [text]

    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(len(text), start + max_chars)
        if end < len(text):
            split_at = text.rfind(" ", start, end)
            if split_at > start + max_chars * 0.6:
                end = split_at
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        start = max(0, end - overlap)
    return chunks


def summarize_chunk(text: str, max_terms: int = 18) -> str:
    words = [token.lower() for token in WORD_PATTERN.findall(text)]
    if not words:
        return ""
    counts = Counter(words)
    top_terms = [term for term, _ in counts.most_common(max_terms)]
    return "Summary terms: " + ", ".join(top_terms)


def token_set(text: str) -> set[str]:
    return {token.lower() for token in WORD_PATTERN.findall(text)}
