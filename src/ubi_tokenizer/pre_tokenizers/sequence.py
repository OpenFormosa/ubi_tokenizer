"""Sequence pre-tokenizer: apply several pre-tokenizers in order."""

from __future__ import annotations

from typing import Any

from ubi_tokenizer.pre_tokenizers.base import PreTokenizer


class Sequence(PreTokenizer):
    """Chain multiple pre-tokenizers, feeding each output piece to the next."""

    def __init__(self, pretokenizers: list[PreTokenizer]) -> None:
        self.pretokenizers = list(pretokenizers)

    def pre_tokenize_str(self, text: str) -> list[tuple[str, tuple[int, int]]]:
        pieces: list[tuple[str, tuple[int, int]]] = [(text, (0, len(text)))]
        for pretok in self.pretokenizers:
            next_pieces: list[tuple[str, tuple[int, int]]] = []
            for piece, (start, _end) in pieces:
                for sub_piece, (sub_start, sub_end) in pretok.pre_tokenize_str(piece):
                    next_pieces.append(
                        (sub_piece, (start + sub_start, start + sub_end))
                    )
            pieces = next_pieces
        return pieces

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Sequence":
        from ubi_tokenizer.pre_tokenizers import build_pre_tokenizer

        return cls([build_pre_tokenizer(item) for item in data.get("pretokenizers", [])])

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "Sequence",
            "pretokenizers": [p.to_dict() for p in self.pretokenizers],
        }
