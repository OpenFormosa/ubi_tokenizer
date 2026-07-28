"""Byte-pair-encoding model.

The :class:`BPE` model holds the learned vocabulary and merge ranks and turns a
single byte-mapped pre-token piece into a list of subword tokens. The merge
logic is lifted verbatim from the UbiTok seed ``ByteLevelBPE._apply_bpe`` so
that encoding is byte-identical to the reference implementation.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Optional

from ubi_tokenizer._validation import (
    reject_unimplemented_affixes,
    reject_unimplemented_dropout,
)
from ubi_tokenizer.errors import ModelError


class Model:
    """Abstract base for tokenization models."""

    def tokenize(self, piece: str) -> list[str]:  # pragma: no cover - interface
        raise NotImplementedError

    def token_to_id(self, token: str) -> Optional[int]:  # pragma: no cover
        raise NotImplementedError

    def id_to_token(self, token_id: int) -> Optional[str]:  # pragma: no cover
        raise NotImplementedError

    def get_vocab(self) -> dict[str, int]:  # pragma: no cover - interface
        raise NotImplementedError

    def get_vocab_size(self) -> int:  # pragma: no cover - interface
        raise NotImplementedError


class BPE(Model):
    """Byte-pair-encoding model over a byte-mapped vocabulary."""

    def __init__(
        self,
        vocab: Optional[dict[str, int]] = None,
        merges: Optional[list[tuple[str, str]]] = None,
        *,
        cache_capacity: int = 10000,
        dropout: Optional[float] = None,
        unk_token: Optional[str] = None,
        continuing_subword_prefix: Optional[str] = None,
        end_of_word_suffix: Optional[str] = None,
        fuse_unk: bool = False,
        byte_fallback: bool = False,
        ignore_merges: bool = False,
    ) -> None:
        reject_unimplemented_dropout(dropout)
        reject_unimplemented_affixes(continuing_subword_prefix, end_of_word_suffix)
        self.vocab: dict[str, int] = dict(vocab) if vocab else {}
        self.merges: list[tuple[str, str]] = [tuple(m) for m in (merges or [])]
        # Accepted rather than rejected: a cache is a pure optimization, so its
        # absence cannot change output. HuggingFace does not serialize it into
        # tokenizer.json either, so to_dict() stays faithful by omitting it.
        self.cache_capacity = cache_capacity
        self.dropout = dropout
        self.unk_token = unk_token
        self.continuing_subword_prefix = continuing_subword_prefix
        self.end_of_word_suffix = end_of_word_suffix
        self.fuse_unk = fuse_unk
        self.byte_fallback = byte_fallback
        self.ignore_merges = ignore_merges
        self._rebuild()

    # -- index maintenance -------------------------------------------------
    def _rebuild(self) -> None:
        self.id_to_token_map: dict[int, str] = {
            token_id: token for token, token_id in self.vocab.items()
        }
        self.merge_ranks: dict[tuple[str, str], int] = {
            pair: rank for rank, pair in enumerate(self.merges)
        }

    def set_vocab_and_merges(
        self, vocab: dict[str, int], merges: list[tuple[str, str]]
    ) -> None:
        """Replace vocab + merges in place (used by trainers)."""
        self.vocab = dict(vocab)
        self.merges = [tuple(m) for m in merges]
        self._rebuild()

    # -- tokenization ------------------------------------------------------
    def tokenize(self, piece: str) -> list[str]:
        """Turn one pre-token piece into subword tokens."""
        if not piece:
            return []
        # A whole piece that is already a vocabulary entry is emitted as-is with
        # no merging. GPT-4o / Llama-3 tokenizers rely on this.
        if self.ignore_merges and piece in self.vocab:
            return [piece]
        return self._apply_merges(self._resolve_symbols(piece))

    def _resolve_symbols(self, piece: str) -> list[str]:
        """Map each character onto vocabulary tokens, resolving unknowns.

        Runs *before* merging, because the tokens produced here -- byte-fallback
        tokens and generated unks alike -- take part in merges themselves. A
        vocabulary containing ``<0xE4><0xB8>`` with a merge over its two halves
        tokenizes ``中`` as ``['<0xE4><0xB8>', '<0xAD>']``.

        Unresolvable characters are dropped when there is no unk token, which is
        what HuggingFace does.
        """
        symbols: list[str] = []
        # HuggingFace holds at most one deferred unk, and what flushes it is
        # load-bearing; see the byte-fallback branch below.
        pending_unk = False
        for character in piece:
            if character in self.vocab:
                if pending_unk:
                    symbols.append(self._unk_token_or_raise())
                    pending_unk = False
                symbols.append(character)
                continue
            if self.byte_fallback:
                byte_tokens = [
                    f"<0x{value:02X}>" for value in character.encode("utf-8")
                ]
                # All-or-nothing per character: a single missing byte token sends
                # the whole character down the unk path rather than emitting a
                # partial expansion.
                if all(token in self.vocab for token in byte_tokens):
                    # Deliberately leaves pending_unk alone. HuggingFace defers a
                    # pending unk past a successful byte expansion, so with
                    # <0xE4> missing, '中é' gives ['<0xC3>', '<0xA9>', '<unk>'].
                    # Probably an upstream quirk, but byte-parity is the point of
                    # this package, so it is reproduced rather than corrected.
                    symbols.extend(byte_tokens)
                    continue
            if self.unk_token is None:
                continue
            if pending_unk and not self.fuse_unk:
                symbols.append(self._unk_token_or_raise())
            pending_unk = True
        if pending_unk:
            symbols.append(self._unk_token_or_raise())
        return symbols

    def _unk_token_or_raise(self) -> str:
        """Return the unk token, or fail if the vocabulary has no id for it."""
        if self.unk_token not in self.vocab:
            raise ModelError(
                f"unk token {self.unk_token!r} is not in the vocabulary, so "
                "out-of-vocabulary input cannot be represented"
            )
        return self.unk_token

    # -- core merge logic (lifted from ByteLevelBPE._apply_bpe) ------------
    def _apply_merges(self, symbols: Sequence[str]) -> list[str]:
        """Apply BPE merges to already-resolved symbols."""
        symbols = tuple(symbols)
        if len(symbols) < 2:
            return list(symbols)
        while True:
            pairs = [
                (symbols[index], symbols[index + 1])
                for index in range(len(symbols) - 1)
            ]
            ranked = [
                (self.merge_ranks[pair], pair)
                for pair in pairs
                if pair in self.merge_ranks
            ]
            if not ranked:
                break
            _, best_pair = min(ranked)
            merged: list[str] = []
            index = 0
            while index < len(symbols):
                if (
                    index < len(symbols) - 1
                    and symbols[index] == best_pair[0]
                    and symbols[index + 1] == best_pair[1]
                ):
                    merged.append(symbols[index] + symbols[index + 1])
                    index += 2
                else:
                    merged.append(symbols[index])
                    index += 1
            symbols = tuple(merged)
            if len(symbols) < 2:
                break
        return list(symbols)

    # -- vocabulary access -------------------------------------------------
    def token_to_id(self, token: str) -> Optional[int]:
        return self.vocab.get(token)

    def id_to_token(self, token_id: int) -> Optional[str]:
        return self.id_to_token_map.get(token_id)

    def get_vocab(self) -> dict[str, int]:
        return dict(self.vocab)

    def get_vocab_size(self) -> int:
        return len(self.vocab)

    # -- serialization -----------------------------------------------------
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BPE":
        """Build a BPE model from a tokenizer.json ``model`` block."""
        merges_raw = data.get("merges", [])
        merges: list[tuple[str, str]] = []
        for item in merges_raw:
            if isinstance(item, (list, tuple)):
                left, right = item[0], item[1]
            else:
                left, right = item.split(" ", 1)
            merges.append((left, right))
        return cls(
            vocab=data.get("vocab", {}),
            merges=merges,
            dropout=data.get("dropout"),
            unk_token=data.get("unk_token"),
            continuing_subword_prefix=data.get("continuing_subword_prefix"),
            end_of_word_suffix=data.get("end_of_word_suffix"),
            fuse_unk=bool(data.get("fuse_unk", False)),
            byte_fallback=bool(data.get("byte_fallback", False)),
            ignore_merges=bool(data.get("ignore_merges", False)),
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize as a tokenizer.json ``model`` block."""
        return {
            "type": "BPE",
            "dropout": self.dropout,
            "unk_token": self.unk_token,
            "continuing_subword_prefix": self.continuing_subword_prefix,
            "end_of_word_suffix": self.end_of_word_suffix,
            "fuse_unk": self.fuse_unk,
            "byte_fallback": self.byte_fallback,
            "ignore_merges": self.ignore_merges,
            "vocab": dict(self.vocab),
            "merges": [f"{left} {right}" for left, right in self.merges],
        }
