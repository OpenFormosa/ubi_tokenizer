"""In-memory greedy byte-level BPE trainer.

For large/distributed training use UbiTok; this is for small corpora and tests.

This is a clean-room implementation: it does NOT import ubitok/taiwan_tokenizer.
The output format (byte-mapped vocab + ``"left right"`` merges, byte alphabet
first, then specials, then learned merges) matches the UbiTok export exactly so
that the resulting BPE model is byte-identical to a UbiTok-trained one.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Iterator
from typing import Optional

from ubi_tokenizer._validation import reject_unimplemented_affixes
from ubi_tokenizer.byte_level import BYTE_ENCODER, encode_token_bytes
from ubi_tokenizer.models.bpe import BPE
from ubi_tokenizer.pre_tokenizers.byte_level import ByteLevel


class Trainer:  # pragma: no cover - marker base class
    """Abstract base for trainers."""


class BpeTrainer(Trainer):
    """Greedy in-memory BPE trainer.

    For large/distributed training use UbiTok; this is for small corpora and
    tests.
    """

    def __init__(
        self,
        vocab_size: int = 30000,
        min_frequency: int = 2,
        special_tokens: Optional[list[str]] = None,
        initial_alphabet: Optional[list[str]] = None,
        show_progress: bool = False,
        continuing_subword_prefix: Optional[str] = None,
        end_of_word_suffix: Optional[str] = None,
    ) -> None:
        reject_unimplemented_affixes(continuing_subword_prefix, end_of_word_suffix)
        self.vocab_size = vocab_size
        self.min_frequency = min_frequency
        self.special_tokens = list(special_tokens or [])
        self.initial_alphabet = list(initial_alphabet or [])
        self.show_progress = show_progress
        self.continuing_subword_prefix = continuing_subword_prefix
        self.end_of_word_suffix = end_of_word_suffix

    # -- public training entry points --------------------------------------
    def train(self, files: list[str], model: BPE) -> None:
        """Train on a list of UTF-8 text files, mutating ``model`` in place."""
        self.train_from_iterator(_iter_file_lines(files), model)

    def train_from_iterator(self, iterator: Iterable[str], model: BPE) -> None:
        """Train on an iterable of text strings, mutating ``model`` in place."""
        word_counts = self._count_words(iterator)
        vocab, merges = self._learn(word_counts)
        model.set_vocab_and_merges(vocab, merges)
        if model.unk_token is None and "<|unk|>" in vocab:
            model.unk_token = "<|unk|>"

    # -- internals ---------------------------------------------------------
    def _count_words(self, iterator: Iterable[str]) -> Counter[str]:
        """Pre-tokenize the corpus and count byte-mapped pieces."""
        pretok = ByteLevel(add_prefix_space=False, use_regex=True)
        counts: Counter[str] = Counter()
        for text in iterator:
            if not text:
                continue
            for piece, _offset in pretok.pre_tokenize_str(text):
                if piece:
                    counts[piece] += 1
        return counts

    def _learn(self, word_counts: Counter[str]) -> tuple[dict[str, int], list[tuple[str, str]]]:
        # Seed the 256-byte alphabet (byte value -> byte-mapped char), keeping
        # the deterministic byte-value ordering used by UbiTok exports.
        vocab: dict[str, int] = {}
        next_id = 0
        for value in range(256):
            token = BYTE_ENCODER[value]
            vocab[token] = next_id
            next_id += 1
        for symbol in self.initial_alphabet:
            mapped = encode_token_bytes(symbol.encode("utf-8"))
            if mapped not in vocab:
                vocab[mapped] = next_id
                next_id += 1

        # Represent each unique word as a tuple of byte-mapped symbols.
        words: list[tuple[str, ...]] = []
        freqs: list[int] = []
        for word, count in word_counts.items():
            words.append(tuple(word))
            freqs.append(count)

        merges: list[tuple[str, str]] = []
        # Budget for learned-merge tokens, leaving room for special tokens.
        target = self.vocab_size - len(self.special_tokens)

        while len(vocab) < target:
            pair_counts = self._count_pairs(words, freqs)
            if not pair_counts:
                break
            # Most frequent pair; deterministic tie-break on the pair itself.
            best_pair, best_count = max(
                pair_counts.items(), key=lambda item: (item[1], _neg_key(item[0]))
            )
            if best_count < self.min_frequency:
                break
            merged_token = best_pair[0] + best_pair[1]
            merges.append(best_pair)
            if merged_token not in vocab:
                vocab[merged_token] = len(vocab)
            words = [self._merge_word(word, best_pair) for word in words]

        # Special tokens are appended last, after the learned merges.
        for token in self.special_tokens:
            if token not in vocab:
                vocab[token] = len(vocab)

        return vocab, merges

    @staticmethod
    def _count_pairs(
        words: list[tuple[str, ...]], freqs: list[int]
    ) -> Counter[tuple[str, str]]:
        pair_counts: Counter[tuple[str, str]] = Counter()
        for symbols, freq in zip(words, freqs):
            for index in range(len(symbols) - 1):
                pair_counts[(symbols[index], symbols[index + 1])] += freq
        return pair_counts

    @staticmethod
    def _merge_word(
        symbols: tuple[str, ...], pair: tuple[str, str]
    ) -> tuple[str, ...]:
        if len(symbols) < 2:
            return symbols
        merged: list[str] = []
        index = 0
        left, right = pair
        while index < len(symbols):
            if (
                index < len(symbols) - 1
                and symbols[index] == left
                and symbols[index + 1] == right
            ):
                merged.append(left + right)
                index += 2
            else:
                merged.append(symbols[index])
                index += 1
        return tuple(merged)


def _neg_key(pair: tuple[str, str]) -> tuple[int, ...]:
    """Tie-break key: prefer the lexicographically smallest pair deterministically."""
    return tuple(-ord(char) for char in (pair[0] + " " + pair[1]))


def _iter_file_lines(files: list[str]) -> Iterator[str]:
    for path in files:
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                yield line.rstrip("\n")
