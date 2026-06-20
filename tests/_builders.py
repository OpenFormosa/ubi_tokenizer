"""Shared helpers to build tiny tokenizers for tests."""

from __future__ import annotations

from ubi_tokenizer.byte_level import BYTE_ENCODER, encode_token_bytes
from ubi_tokenizer.decoders.byte_level import ByteLevel as ByteLevelDecoder
from ubi_tokenizer.models.bpe import BPE
from ubi_tokenizer.pre_tokenizers.byte_level import ByteLevel as ByteLevelPreTokenizer
from ubi_tokenizer.tokenizer import Tokenizer

# A few hand-picked merges over common ASCII byte-mapped chars. Each left/right
# side is a byte-mapped token; later merges may reference earlier merged tokens.
_AB = encode_token_bytes(b"a") + encode_token_bytes(b"b")
HAND_MERGES: list[tuple[str, str]] = [
    (encode_token_bytes(b"a"), encode_token_bytes(b"b")),  # "a" + "b" -> "ab"
    (_AB, encode_token_bytes(b"c")),  # "ab" + "c" -> "abc"
    (encode_token_bytes(b"1"), encode_token_bytes(b"2")),  # "1" + "2" -> "12"
]

SPECIAL_TOKENS = ["<|bos|>", "<|eos|>", "<|pad|>", "<|unk|>"]


def build_vocab(merges, special_tokens):
    """Build a byte-level vocab: 256 byte tokens, then merges, then specials."""
    vocab: dict[str, int] = {BYTE_ENCODER[value]: value for value in range(256)}
    next_id = 256
    for left, right in merges:
        token = left + right
        if token not in vocab:
            vocab[token] = next_id
            next_id += 1
    for token in special_tokens:
        if token not in vocab:
            vocab[token] = next_id
            next_id += 1
    return vocab


def build_tiny_tokenizer(merges=None, special_tokens=None):
    """Construct a ubi_tokenizer Tokenizer with ByteLevel pre-tokenizer/decoder."""
    if merges is None:
        merges = HAND_MERGES
    if special_tokens is None:
        special_tokens = SPECIAL_TOKENS
    vocab = build_vocab(merges, special_tokens)
    model = BPE(vocab=vocab, merges=merges, unk_token="<|unk|>")
    tokenizer = Tokenizer(model)
    tokenizer.pre_tokenizer = ByteLevelPreTokenizer(
        add_prefix_space=False, use_regex=True, trim_offsets=True
    )
    tokenizer.decoder = ByteLevelDecoder(
        add_prefix_space=True, use_regex=True, trim_offsets=True
    )
    tokenizer.add_special_tokens(special_tokens)
    return tokenizer, vocab, merges, special_tokens
