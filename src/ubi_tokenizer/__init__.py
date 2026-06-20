"""ubi_tokenizer: a ubiquitous, dependency-free, pure-Python tokenizer runtime.

Mirrors the architecture and public API of ``huggingface/tokenizers`` for
byte-level BPE, but is implemented entirely with the Python standard library
(no Rust, no third-party dependencies). It reads and writes the same
``tokenizer.json`` schema produced by UbiTok / HuggingFace.
"""

from __future__ import annotations

from ubi_tokenizer import (
    decoders,
    models,
    normalizers,
    pre_tokenizers,
    processors,
    trainers,
)
from ubi_tokenizer.added_tokens import AddedToken
from ubi_tokenizer.encoding import Encoding
from ubi_tokenizer.errors import UbiTokenizerError
from ubi_tokenizer.tokenizer import Tokenizer

__version__ = "0.1.0"

__all__ = [
    "Tokenizer",
    "Encoding",
    "AddedToken",
    "UbiTokenizerError",
    "models",
    "pre_tokenizers",
    "decoders",
    "normalizers",
    "processors",
    "trainers",
]
