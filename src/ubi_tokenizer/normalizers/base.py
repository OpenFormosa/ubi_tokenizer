"""Normalizer base class."""

from __future__ import annotations

from typing import Any


class Normalizer:
    """Abstract base for normalizers."""

    def normalize_str(self, text: str) -> str:  # pragma: no cover - interface
        raise NotImplementedError

    def to_dict(self) -> dict[str, Any]:  # pragma: no cover - interface
        raise NotImplementedError
