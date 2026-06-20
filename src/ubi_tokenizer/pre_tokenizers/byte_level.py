"""ByteLevel pre-tokenizer.

Wraps the generic pre-token splitter and maps each UTF-8 piece into the
reversible byte-mapped space via :func:`encode_token_bytes`.
"""

from __future__ import annotations

from typing import Any

from ubi_tokenizer.byte_level import encode_token_bytes
from ubi_tokenizer.pre_tokenizers.base import (
    PreTokenizer,
    iter_pretokenized_with_offsets,
)


class ByteLevel(PreTokenizer):
    """Byte-level pre-tokenizer mirroring ``tokenizers.pre_tokenizers.ByteLevel``."""

    def __init__(
        self,
        add_prefix_space: bool = False,
        use_regex: bool = True,
        trim_offsets: bool = True,
    ) -> None:
        self.add_prefix_space = add_prefix_space
        self.use_regex = use_regex
        self.trim_offsets = trim_offsets

    def pre_tokenize_str(self, text: str) -> list[tuple[str, tuple[int, int]]]:
        if self.add_prefix_space and text and not text.startswith(" "):
            # Offsets are computed against the prefixed string then shifted back.
            shifted = " " + text
            pieces = self._split(shifted)
            result: list[tuple[str, tuple[int, int]]] = []
            for piece, (start, end) in pieces:
                new_start = max(start - 1, 0)
                new_end = max(end - 1, 0)
                result.append((piece, (new_start, new_end)))
            return result
        return self._split(text)

    def _split(self, text: str) -> list[tuple[str, tuple[int, int]]]:
        result: list[tuple[str, tuple[int, int]]] = []
        if self.use_regex:
            iterator = iter_pretokenized_with_offsets(text)
        else:
            iterator = iter([(text, (0, len(text)))]) if text else iter([])
        for piece, offset in iterator:
            mapped = encode_token_bytes(piece.encode("utf-8"))
            result.append((mapped, offset))
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ByteLevel":
        return cls(
            add_prefix_space=bool(data.get("add_prefix_space", False)),
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
