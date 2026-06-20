"""Optional rich-transcription helpers (the ``[transcription]`` extra).

This subpackage is NOT imported by the core ``ubi_tokenizer`` package. Import it
explicitly::

    from ubi_tokenizer.transcription import timestamp_format

It is stdlib-only; the ``[transcription]`` extra exists purely as a namespace
gate, not to pull in third-party dependencies.
"""

from __future__ import annotations

from ubi_tokenizer.transcription.special_tokens import (
    DENSE_TIMESTAMP_TOKEN_PATTERNS,
    RICH_TRANSCRIPTION_TOKENS,
    find_dense_timestamp_tokens,
    validate_no_dense_timestamp_tokens,
)

__all__ = [
    "DENSE_TIMESTAMP_TOKEN_PATTERNS",
    "RICH_TRANSCRIPTION_TOKENS",
    "find_dense_timestamp_tokens",
    "validate_no_dense_timestamp_tokens",
]
