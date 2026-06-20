"""End-to-end roundtrip + golden parity against the vendored seed."""

import pytest

from tests._builders import build_tiny_tokenizer
from tests._seed_reference import ByteLevelBPE

SAMPLES = [
    "大家好，歡迎回來。",
    "注音 ㄅㄆㄇ 與台語 tshit-á 混合測試。",
    '{"Start":0.00,"End":3.42,"Content":"測試"}',
    "abc 123 def",
]


@pytest.mark.parametrize("text", SAMPLES)
def test_decode_encode_roundtrip(text):
    tokenizer, _vocab, _merges, _specials = build_tiny_tokenizer()
    enc = tokenizer.encode(text)
    assert tokenizer.decode(enc.ids) == text


@pytest.mark.parametrize("text", SAMPLES)
def test_golden_parity_with_seed(text):
    tokenizer, vocab, merges, specials = build_tiny_tokenizer()
    seed = ByteLevelBPE(vocab=vocab, merges=merges, special_tokens=specials)
    assert tokenizer.encode(text).ids == seed.encode(text)


@pytest.mark.parametrize("text", SAMPLES)
def test_golden_parity_decode(text):
    tokenizer, vocab, merges, specials = build_tiny_tokenizer()
    seed = ByteLevelBPE(vocab=vocab, merges=merges, special_tokens=specials)
    ids = tokenizer.encode(text).ids
    assert tokenizer.decode(ids) == seed.decode(ids)


def test_encoding_arrays_are_parallel_and_consistent():
    tokenizer, _vocab, _merges, _specials = build_tiny_tokenizer()
    enc = tokenizer.encode("abc 123 def")
    n = len(enc)
    assert len(enc.ids) == n
    assert len(enc.tokens) == n
    assert len(enc.offsets) == n
    assert len(enc.type_ids) == n
    assert len(enc.attention_mask) == n
    assert len(enc.special_tokens_mask) == n
    assert len(enc.word_ids) == n
    assert all(mask == 1 for mask in enc.attention_mask)
    assert all(type_id == 0 for type_id in enc.type_ids)
    assert all(mask == 0 for mask in enc.special_tokens_mask)
