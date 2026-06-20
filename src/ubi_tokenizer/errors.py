"""Exception hierarchy for ubi_tokenizer."""

from __future__ import annotations


class UbiTokenizerError(Exception):
    """Base class for all ubi_tokenizer errors."""


class DeserializationError(UbiTokenizerError):
    """Raised when a tokenizer payload cannot be parsed."""


class ModelError(UbiTokenizerError):
    """Raised for invalid model state (missing tokens, bad merges, etc.)."""
