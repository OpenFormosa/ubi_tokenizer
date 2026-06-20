"""Stdlib-only normalizers (Unicode normalization, case folding, replace)."""

from __future__ import annotations

import re
import unicodedata
from typing import Any

from ubi_tokenizer.normalizers.base import Normalizer


class _UnicodeNormalizer(Normalizer):
    form = ""

    def normalize_str(self, text: str) -> str:
        return unicodedata.normalize(self.form, text)

    def to_dict(self) -> dict[str, Any]:
        return {"type": self.form}


class NFC(_UnicodeNormalizer):
    form = "NFC"


class NFD(_UnicodeNormalizer):
    form = "NFD"


class NFKC(_UnicodeNormalizer):
    form = "NFKC"


class NFKD(_UnicodeNormalizer):
    form = "NFKD"


class Lowercase(Normalizer):
    """Lowercase the text."""

    def normalize_str(self, text: str) -> str:
        return text.lower()

    def to_dict(self) -> dict[str, Any]:
        return {"type": "Lowercase"}


class Replace(Normalizer):
    """Replace occurrences of ``pattern`` with ``content``."""

    def __init__(self, pattern: str, content: str, *, is_regex: bool = False) -> None:
        self.pattern = pattern
        self.content = content
        self.is_regex = is_regex

    def normalize_str(self, text: str) -> str:
        if self.is_regex:
            return re.sub(self.pattern, self.content, text)
        return text.replace(self.pattern, self.content)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Replace":
        pattern = data.get("pattern", {})
        if isinstance(pattern, dict):
            if "Regex" in pattern:
                return cls(pattern["Regex"], data.get("content", ""), is_regex=True)
            return cls(pattern.get("String", ""), data.get("content", ""))
        return cls(str(pattern), data.get("content", ""))

    def to_dict(self) -> dict[str, Any]:
        pattern: dict[str, str] = (
            {"Regex": self.pattern} if self.is_regex else {"String": self.pattern}
        )
        return {"type": "Replace", "pattern": pattern, "content": self.content}


class Sequence(Normalizer):
    """Apply a chain of normalizers in order."""

    def __init__(self, normalizers: list[Normalizer]) -> None:
        self.normalizers = list(normalizers)

    def normalize_str(self, text: str) -> str:
        for normalizer in self.normalizers:
            text = normalizer.normalize_str(text)
        return text

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Sequence":
        from ubi_tokenizer.normalizers import build_normalizer

        built = [build_normalizer(item) for item in data.get("normalizers", [])]
        return cls([n for n in built if n is not None])

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "Sequence",
            "normalizers": [n.to_dict() for n in self.normalizers],
        }
