"""Rich-transcription special tokens and dense-timestamp guards.

Ported from the UbiTok seed ``special_tokens.py`` (the transcription-relevant
subset). Lives only in the optional ``transcription`` extra; the core
``ubi_tokenizer`` package never imports it.
"""

from __future__ import annotations

import re
from collections.abc import Iterable

DENSE_TIMESTAMP_TOKEN_PATTERNS: list[str] = [
    r"<\|ts_[0-9]+\|>",
    r"<\|timestamp_[0-9]+\|>",
    r"<\|time_[0-9]+\|>",
]

RICH_TRANSCRIPTION_TOKENS: list[str] = [
    "<|transcript_start|>",
    "<|transcript_end|>",
    "<|segment_start|>",
    "<|segment_end|>",
    "<|speaker|>",
    "<|start_time|>",
    "<|end_time|>",
    "<|duration|>",
    "<|content|>",
    "<|non_speech_event|>",
]

_DENSE_TIMESTAMP_REGEXES: tuple[re.Pattern[str], ...] = tuple(
    re.compile(pattern) for pattern in DENSE_TIMESTAMP_TOKEN_PATTERNS
)


def find_dense_timestamp_tokens(tokens: Iterable[str]) -> list[str]:
    """Return tokens that look like dense timestamp special-token ranges."""
    matches: list[str] = []
    for token in tokens:
        if any(pattern.fullmatch(token) for pattern in _DENSE_TIMESTAMP_REGEXES):
            matches.append(token)
    return matches


def validate_no_dense_timestamp_tokens(tokens: Iterable[str]) -> None:
    """Raise ValueError if any token matches a prohibited dense timestamp form."""
    matches = find_dense_timestamp_tokens(tokens)
    if matches:
        preview = ", ".join(matches[:10])
        raise ValueError(f"Dense timestamp tokens are prohibited: {preview}")
