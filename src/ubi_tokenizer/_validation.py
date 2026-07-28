"""Shared rejection rules for recognized-but-unimplemented options.

The BPE model and the BPE trainer both accept options that they store, and in
the model's case serialize, without ever acting on them. Silently accepting
such an option hands the caller wrong tokenization with no signal, so a value
that would change behaviour is rejected instead.

The rule has to stay identical in both places, so it lives here rather than
being duplicated.
"""

from __future__ import annotations

from typing import Optional

from ubi_tokenizer.errors import ModelError, UnsupportedFeatureError

_HINT = (
    "recognized but not implemented by ubi_tokenizer. Drop the option, or use "
    "huggingface/tokenizers if you need it."
)


def reject_unimplemented_affixes(
    continuing_subword_prefix: Optional[str],
    end_of_word_suffix: Optional[str],
) -> None:
    """Reject affix options that would change tokenization.

    ``None`` and ``""`` are genuine no-ops in HuggingFace and stay accepted --
    GPT-2-lineage ``tokenizer.json`` files ship ``""`` for both, and rejecting
    those would break loading files that tokenize correctly today.
    """
    for name, value in (
        ("continuing_subword_prefix", continuing_subword_prefix),
        ("end_of_word_suffix", end_of_word_suffix),
    ):
        if value:
            raise UnsupportedFeatureError(f"{name}={value!r} is {_HINT}")


def reject_unimplemented_dropout(dropout: Optional[float]) -> None:
    """Reject BPE dropout.

    ``None`` and ``0.0`` both mean "no dropout" and stay accepted. Values
    outside ``[0, 1]`` are invalid rather than unimplemented, which is why they
    raise :class:`ModelError`; HuggingFace rejects them at construction too.
    """
    if dropout is None:
        return
    if not 0.0 <= dropout <= 1.0:
        raise ModelError(f"dropout={dropout!r} must be between 0 and 1, inclusive")
    if dropout > 0.0:
        raise UnsupportedFeatureError(f"dropout={dropout!r} is {_HINT}")
