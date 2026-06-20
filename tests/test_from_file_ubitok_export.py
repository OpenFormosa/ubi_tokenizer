"""Loading a committed UbiTok-schema tokenizer.json fixture."""

from pathlib import Path

from ubi_tokenizer.tokenizer import Tokenizer

FIXTURE = Path(__file__).parent / "fixtures" / "tiny_ubitok_export" / "tokenizer.json"


def test_loads_fixture_and_roundtrips():
    tokenizer = Tokenizer.from_file(FIXTURE)
    for text in ["大家好", "abc 123", '{"x":0}']:
        enc = tokenizer.encode(text)
        assert tokenizer.decode(enc.ids) == text


def test_special_tokens_resolve_via_token_to_id():
    tokenizer = Tokenizer.from_file(FIXTURE)
    for special in ["<|pad|>", "<|bos|>", "<|eos|>", "<|unk|>"]:
        token_id = tokenizer.token_to_id(special)
        assert token_id is not None
        assert tokenizer.id_to_token(token_id) == special


def test_special_tokens_mask_is_correct():
    tokenizer = Tokenizer.from_file(FIXTURE)
    enc = tokenizer.encode("<|bos|>abc<|eos|>")
    # First and last tokens are the specials.
    assert enc.special_tokens_mask[0] == 1
    assert enc.special_tokens_mask[-1] == 1
    assert enc.tokens[0] == "<|bos|>"
    assert enc.tokens[-1] == "<|eos|>"
    # The middle byte tokens are not special.
    assert sum(enc.special_tokens_mask) == 2


def test_decode_skip_special_tokens():
    tokenizer = Tokenizer.from_file(FIXTURE)
    enc = tokenizer.encode("<|bos|>abc<|eos|>")
    assert tokenizer.decode(enc.ids, skip_special_tokens=True) == "abc"
    assert tokenizer.decode(enc.ids, skip_special_tokens=False) == "<|bos|>abc<|eos|>"
