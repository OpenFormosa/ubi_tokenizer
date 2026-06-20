"""Encoding: the output of :meth:`Tokenizer.encode`.

Mirrors ``tokenizers.Encoding``: parallel arrays of ids, tokens, offsets,
type ids, attention/special masks, and word ids.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Encoding:
    """A single encoded sequence."""

    ids: list[int] = field(default_factory=list)
    tokens: list[str] = field(default_factory=list)
    type_ids: list[int] = field(default_factory=list)
    offsets: list[tuple[int, int]] = field(default_factory=list)
    attention_mask: list[int] = field(default_factory=list)
    special_tokens_mask: list[int] = field(default_factory=list)
    word_ids: list[Optional[int]] = field(default_factory=list)
    overflowing: list["Encoding"] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.ids)

    @property
    def n_sequences(self) -> int:
        return 1

    def truncate(
        self,
        max_length: int,
        stride: int = 0,
        direction: str = "right",
    ) -> None:
        """Truncate the encoding in place to ``max_length`` tokens.

        ``stride`` overflowing tokens are carried over into ``overflowing``.
        """
        if max_length < 0:
            raise ValueError("max_length must be non-negative")
        if len(self.ids) <= max_length:
            return
        if direction not in ("right", "left"):
            raise ValueError("direction must be 'right' or 'left'")

        if direction == "right":
            kept = slice(0, max_length)
            overflow_start = max(max_length - stride, 0)
            overflow = slice(overflow_start, len(self.ids))
        else:
            cut = len(self.ids) - max_length
            kept = slice(cut, len(self.ids))
            overflow = slice(0, min(cut + stride, len(self.ids)))

        overflow_encoding = self._slice(overflow)
        self._apply_slice(kept)
        if len(overflow_encoding):
            self.overflowing.append(overflow_encoding)

    def pad(
        self,
        length: int,
        *,
        direction: str = "right",
        pad_id: int = 0,
        pad_type_id: int = 0,
        pad_token: str = "[PAD]",
    ) -> None:
        """Pad the encoding in place up to ``length`` tokens."""
        if direction not in ("right", "left"):
            raise ValueError("direction must be 'right' or 'left'")
        missing = length - len(self.ids)
        if missing <= 0:
            return

        pad_ids = [pad_id] * missing
        pad_tokens = [pad_token] * missing
        pad_type_ids = [pad_type_id] * missing
        pad_offsets = [(0, 0)] * missing
        pad_attention = [0] * missing
        pad_special = [1] * missing
        pad_word_ids: list[Optional[int]] = [None] * missing

        if direction == "right":
            self.ids = self.ids + pad_ids
            self.tokens = self.tokens + pad_tokens
            self.type_ids = self.type_ids + pad_type_ids
            self.offsets = self.offsets + pad_offsets
            self.attention_mask = self.attention_mask + pad_attention
            self.special_tokens_mask = self.special_tokens_mask + pad_special
            self.word_ids = self.word_ids + pad_word_ids
        else:
            self.ids = pad_ids + self.ids
            self.tokens = pad_tokens + self.tokens
            self.type_ids = pad_type_ids + self.type_ids
            self.offsets = pad_offsets + self.offsets
            self.attention_mask = pad_attention + self.attention_mask
            self.special_tokens_mask = pad_special + self.special_tokens_mask
            self.word_ids = pad_word_ids + self.word_ids

    def _slice(self, sl: slice) -> "Encoding":
        return Encoding(
            ids=self.ids[sl],
            tokens=self.tokens[sl],
            type_ids=self.type_ids[sl],
            offsets=self.offsets[sl],
            attention_mask=self.attention_mask[sl],
            special_tokens_mask=self.special_tokens_mask[sl],
            word_ids=self.word_ids[sl],
        )

    def _apply_slice(self, sl: slice) -> None:
        self.ids = self.ids[sl]
        self.tokens = self.tokens[sl]
        self.type_ids = self.type_ids[sl]
        self.offsets = self.offsets[sl]
        self.attention_mask = self.attention_mask[sl]
        self.special_tokens_mask = self.special_tokens_mask[sl]
        self.word_ids = self.word_ids[sl]
