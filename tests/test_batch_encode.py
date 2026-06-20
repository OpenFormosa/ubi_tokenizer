"""Batch encode/decode and padding."""

from tests._builders import build_tiny_tokenizer

TEXTS = ["abc", "123 456", "大家好"]


def test_encode_batch_matches_per_item():
    tokenizer, _vocab, _merges, _specials = build_tiny_tokenizer()
    batch = tokenizer.encode_batch(TEXTS)
    for text, enc in zip(TEXTS, batch):
        assert enc.ids == tokenizer.encode(text).ids


def test_decode_batch_matches_per_item():
    tokenizer, _vocab, _merges, _specials = build_tiny_tokenizer()
    batch = tokenizer.encode_batch(TEXTS)
    id_lists = [enc.ids for enc in batch]
    decoded = tokenizer.decode_batch(id_lists)
    assert decoded == [tokenizer.decode(ids) for ids in id_lists]
    assert decoded == TEXTS


def test_enable_padding_pads_ids_and_attention_mask():
    tokenizer, _vocab, _merges, _specials = build_tiny_tokenizer()
    pad_id = tokenizer.token_to_id("<|pad|>")
    tokenizer.enable_padding(pad_id=pad_id, pad_token="<|pad|>")
    batch = tokenizer.encode_batch(TEXTS)
    target = max(len(tokenizer.encode(text)) for text in TEXTS)
    for enc in batch:
        assert len(enc.ids) == target
        assert len(enc.attention_mask) == target
        assert len(enc.attention_mask) == len(enc.ids)
    # Padded positions carry pad_id with attention 0.
    shortest = min(batch, key=lambda enc: sum(enc.attention_mask))
    real_len = sum(shortest.attention_mask)
    assert all(token_id == pad_id for token_id in shortest.ids[real_len:])
    assert all(mask == 0 for mask in shortest.attention_mask[real_len:])


def test_no_padding_disables_padding():
    tokenizer, _vocab, _merges, _specials = build_tiny_tokenizer()
    tokenizer.enable_padding()
    tokenizer.no_padding()
    batch = tokenizer.encode_batch(TEXTS)
    lengths = {len(enc) for enc in batch}
    assert len(lengths) > 1  # not all equal -> no padding applied


def test_fixed_length_padding_on_single_encode():
    tokenizer, _vocab, _merges, _specials = build_tiny_tokenizer()
    pad_id = tokenizer.token_to_id("<|pad|>")
    tokenizer.enable_padding(pad_id=pad_id, pad_token="<|pad|>", length=16)
    enc = tokenizer.encode("abc")
    assert len(enc) == 16
