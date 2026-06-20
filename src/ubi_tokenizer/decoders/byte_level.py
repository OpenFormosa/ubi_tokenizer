"""ByteLevel decoder.

Lifts the byte-buffer flush logic from the UbiTok seed ``ByteLevelBPE.decode``
(the byte-token portion only) to turn byte-mapped tokens back into text.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from ubi_tokenizer.byte_level import BYTE_DECODER


class ByteLevel:
    """Decode byte-mapped tokens back into a UTF-8 string."""

    def __init__(
        self,
        add_prefix_space: bool = True,
        use_regex: bool = True,
        trim_offsets: bool = True,
    ) -> None:
        self.add_prefix_space = add_prefix_space
        self.use_regex = use_regex
        self.trim_offsets = trim_offsets

    def decode(self, tokens: Iterable[str]) -> str:
        """Decode a list of byte-mapped tokens into text."""
        byte_buffer = bytearray()
        for token in tokens:
            for char in token:
                if char not in BYTE_DECODER:
                    raise ValueError(f"Token cannot be byte-decoded: {token!r}")
                byte_buffer.append(BYTE_DECODER[char])
        return byte_buffer.decode("utf-8", errors="replace")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ByteLevel":
        return cls(
            add_prefix_space=bool(data.get("add_prefix_space", True)),
            use_regex=bool(data.get("use_regex", True)),
            trim_offsets=bool(data.get("trim_offsets", True)),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "ByteLevel",
            "add_prefix_space": self.add_prefix_space,
            "trim_offsets": self.trim_offsets,
            "use_regex": self.use_regex,
        }


class Decoder:  # pragma: no cover - marker base class
    """Abstract base for decoders."""
