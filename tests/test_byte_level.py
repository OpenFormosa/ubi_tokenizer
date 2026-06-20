"""Byte-level mapping reversibility."""

import pytest

from ubi_tokenizer.byte_level import (
    BYTE_DECODER,
    BYTE_ENCODER,
    decode_token_bytes,
    encode_token_bytes,
)


def test_byte_encode_decode_reversible_all_bytes():
    for value in range(256):
        raw = bytes([value])
        mapped = encode_token_bytes(raw)
        assert decode_token_bytes(mapped) == raw


def test_full_byte_string_roundtrip():
    raw = bytes(range(256))
    mapped = encode_token_bytes(raw)
    assert len(mapped) == 256
    assert decode_token_bytes(mapped) == raw


def test_encoder_decoder_are_inverse():
    assert len(BYTE_ENCODER) == 256
    assert len(BYTE_DECODER) == 256
    for value, char in BYTE_ENCODER.items():
        assert BYTE_DECODER[char] == value


def test_decode_rejects_non_byte_char():
    # An emoji code point is well outside the 256-entry byte map.
    bad = chr(0x1F600)
    assert bad not in BYTE_DECODER
    with pytest.raises(ValueError):
        decode_token_bytes(bad)
