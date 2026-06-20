"""Decoders."""

from __future__ import annotations

from typing import Any, Optional

from ubi_tokenizer.decoders.byte_level import ByteLevel, Decoder

__all__ = ["ByteLevel", "Decoder", "build_decoder"]


def build_decoder(data: Optional[dict[str, Any]]) -> Optional[ByteLevel]:
    """Construct a decoder from a tokenizer.json block."""
    if data is None:
        return None
    kind = data.get("type")
    if kind == "ByteLevel":
        return ByteLevel.from_dict(data)
    raise ValueError(f"Unsupported decoder type: {kind!r}")
