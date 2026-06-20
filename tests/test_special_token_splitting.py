"""Special-token splitting (longest match first) and skip-on-decode."""

from tests._builders import build_tiny_tokenizer


def test_adjacent_special_tokens_split():
    tokenizer, _vocab, _merges, _specials = build_tiny_tokenizer()
    enc = tokenizer.encode("<|bos|><|eos|>")
    assert enc.tokens == ["<|bos|>", "<|eos|>"]
    assert enc.special_tokens_mask == [1, 1]


def test_special_surrounded_by_text():
    tokenizer, _vocab, _merges, _specials = build_tiny_tokenizer()
    enc = tokenizer.encode("a<|pad|>b")
    assert enc.tokens[0] != "<|pad|>"
    assert "<|pad|>" in enc.tokens
    pad_index = enc.tokens.index("<|pad|>")
    assert enc.special_tokens_mask[pad_index] == 1


def test_longest_match_first():
    # Register an overlapping pair where one token is a prefix of another.
    tokenizer, _vocab, _merges, _specials = build_tiny_tokenizer(
        special_tokens=["<|a|>", "<|ab|>"]
    )
    # build_tiny_tokenizer registers specials; "<|ab|>" must win over "<|a|>".
    enc = tokenizer.encode("<|ab|>")
    assert enc.tokens == ["<|ab|>"]
    assert enc.special_tokens_mask == [1]


def test_decode_skip_special_tokens_drops_them():
    tokenizer, _vocab, _merges, _specials = build_tiny_tokenizer()
    enc = tokenizer.encode("<|bos|>abc<|eos|>")
    assert tokenizer.decode(enc.ids, skip_special_tokens=True) == "abc"


def test_decode_keeps_special_tokens_by_default():
    tokenizer, _vocab, _merges, _specials = build_tiny_tokenizer()
    enc = tokenizer.encode("<|bos|>abc<|eos|>")
    assert tokenizer.decode(enc.ids) == "<|bos|>abc<|eos|>"
