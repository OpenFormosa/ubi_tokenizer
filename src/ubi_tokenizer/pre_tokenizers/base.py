"""Pre-tokenizer base class.

Ported pre-token regex from the UbiTok seed ``pretokenize.py``.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from typing import Any

BYTE_LEVEL_PRETOKEN_PATTERN = re.compile(
    r"['’](?:s|t|re|ve|m|ll|d)|[^\W\d_]+|\d+|[^\s\w]+|_+|\s+",
    flags=re.IGNORECASE | re.UNICODE,
)


def iter_pretokenized_pieces(text: str) -> Iterator[str]:
    """Yield generic pre-tokenized pieces without crossing boundaries."""
    position = 0
    for match in BYTE_LEVEL_PRETOKEN_PATTERN.finditer(text):
        if match.start() > position:
            yield text[position : match.start()]
        yield match.group(0)
        position = match.end()
    if position < len(text):
        yield text[position:]


def iter_pretokenized_with_offsets(text: str) -> Iterator[tuple[str, tuple[int, int]]]:
    """Yield (piece, (start, end)) character offsets for each pre-token piece."""
    position = 0
    for match in BYTE_LEVEL_PRETOKEN_PATTERN.finditer(text):
        if match.start() > position:
            yield text[position : match.start()], (position, match.start())
        yield match.group(0), (match.start(), match.end())
        position = match.end()
    if position < len(text):
        yield text[position:], (position, len(text))


class PreTokenizer:
    """Abstract base for pre-tokenizers."""

    def pre_tokenize_str(self, text: str) -> list[tuple[str, tuple[int, int]]]:
        """Split ``text`` into (piece, char-offset) pairs.

        The returned pieces are in byte-mapped space so they can be fed
        directly into the BPE model.
        """
        raise NotImplementedError

    def to_dict(self) -> dict[str, Any]:  # pragma: no cover - interface
        raise NotImplementedError
