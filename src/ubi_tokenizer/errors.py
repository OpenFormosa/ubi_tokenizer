"""Exception hierarchy for ubi_tokenizer."""

from __future__ import annotations


class UbiTokenizerError(Exception):
    """Base class for all ubi_tokenizer errors."""


class DeserializationError(UbiTokenizerError):
    """Raised when a tokenizer payload cannot be parsed."""


class ModelError(UbiTokenizerError):
    """Raised for invalid model state (missing tokens, bad merges, etc.)."""


class UnsupportedFeatureError(UbiTokenizerError):
    """Raised when a requested option is recognized but not implemented.

    Distinct from :class:`ModelError`: the request is well-formed, it simply
    cannot be honoured. Raised eagerly so the caller finds out when the
    tokenizer is built rather than getting silently wrong tokenization.
    """
