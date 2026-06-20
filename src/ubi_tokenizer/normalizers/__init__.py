"""Normalizers."""

from __future__ import annotations

from typing import Any, Optional

from ubi_tokenizer.normalizers.base import Normalizer
from ubi_tokenizer.normalizers.unicode import (
    NFC,
    NFD,
    NFKC,
    NFKD,
    Lowercase,
    Replace,
    Sequence,
)

__all__ = [
    "Normalizer",
    "NFC",
    "NFD",
    "NFKC",
    "NFKD",
    "Lowercase",
    "Replace",
    "Sequence",
    "build_normalizer",
]

_SIMPLE = {
    "NFC": NFC,
    "NFD": NFD,
    "NFKC": NFKC,
    "NFKD": NFKD,
    "Lowercase": Lowercase,
}


def build_normalizer(data: Optional[dict[str, Any]]) -> Optional[Normalizer]:
    """Construct a normalizer from a tokenizer.json block."""
    if data is None:
        return None
    kind = data.get("type")
    if kind in _SIMPLE:
        return _SIMPLE[kind]()
    if kind == "Replace":
        return Replace.from_dict(data)
    if kind == "Sequence":
        return Sequence.from_dict(data)
    raise ValueError(f"Unsupported normalizer type: {kind!r}")
