"""Pre-tokenizers."""

from __future__ import annotations

from typing import Any, Optional

from ubi_tokenizer.pre_tokenizers.base import PreTokenizer
from ubi_tokenizer.pre_tokenizers.byte_level import ByteLevel
from ubi_tokenizer.pre_tokenizers.sequence import Sequence

__all__ = ["PreTokenizer", "ByteLevel", "Sequence", "build_pre_tokenizer"]


def build_pre_tokenizer(data: Optional[dict[str, Any]]) -> Optional[PreTokenizer]:
    """Construct a pre-tokenizer from a tokenizer.json block."""
    if data is None:
        return None
    kind = data.get("type")
    if kind == "ByteLevel":
        return ByteLevel.from_dict(data)
    if kind == "Sequence":
        return Sequence.from_dict(data)
    raise ValueError(f"Unsupported pre_tokenizer type: {kind!r}")
