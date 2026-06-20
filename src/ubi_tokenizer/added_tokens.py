"""AddedToken: a registered (usually special) token with metadata.

Mirrors ``tokenizers.AddedToken`` and round-trips with the ``added_tokens``
entries of the tokenizer.json schema.
"""

from __future__ import annotations

from typing import Any


class AddedToken:
    """A token registered on a tokenizer outside of the learned vocabulary."""

    __slots__ = (
        "content",
        "single_word",
        "lstrip",
        "rstrip",
        "normalized",
        "special",
    )

    def __init__(
        self,
        content: str,
        *,
        single_word: bool = False,
        lstrip: bool = False,
        rstrip: bool = False,
        normalized: bool = False,
        special: bool = False,
    ) -> None:
        self.content = content
        self.single_word = single_word
        self.lstrip = lstrip
        self.rstrip = rstrip
        # Special tokens default to not normalized, matching HF behaviour.
        self.normalized = normalized
        self.special = special

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AddedToken":
        """Build an AddedToken from a tokenizer.json ``added_tokens`` entry."""
        return cls(
            data["content"],
            single_word=bool(data.get("single_word", False)),
            lstrip=bool(data.get("lstrip", False)),
            rstrip=bool(data.get("rstrip", False)),
            normalized=bool(data.get("normalized", False)),
            special=bool(data.get("special", False)),
        )

    def to_dict(self, token_id: int | None = None) -> dict[str, Any]:
        """Serialize as a tokenizer.json ``added_tokens`` entry."""
        payload: dict[str, Any] = {}
        if token_id is not None:
            payload["id"] = token_id
        payload.update(
            {
                "content": self.content,
                "single_word": self.single_word,
                "lstrip": self.lstrip,
                "rstrip": self.rstrip,
                "normalized": self.normalized,
                "special": self.special,
            }
        )
        return payload

    def __eq__(self, other: object) -> bool:
        if isinstance(other, AddedToken):
            return self.content == other.content
        if isinstance(other, str):
            return self.content == other
        return NotImplemented

    def __hash__(self) -> int:
        return hash(self.content)

    def __repr__(self) -> str:
        return (
            f"AddedToken({self.content!r}, single_word={self.single_word}, "
            f"lstrip={self.lstrip}, rstrip={self.rstrip}, "
            f"normalized={self.normalized}, special={self.special})"
        )

    def __str__(self) -> str:
        return self.content
