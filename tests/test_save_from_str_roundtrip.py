"""Serialization round-trips: to_str/from_str and loaded-export re-emit."""

import json
from pathlib import Path

from tests._builders import build_tiny_tokenizer
from ubi_tokenizer.tokenizer import Tokenizer

FIXTURE = Path(__file__).parent / "fixtures" / "tiny_ubitok_export" / "tokenizer.json"


def test_from_str_to_str_equivalent():
    tokenizer, _vocab, _merges, _specials = build_tiny_tokenizer()
    dumped = tokenizer.to_str()
    reloaded = Tokenizer.from_str(dumped)
    # The re-emitted dict is identical.
    assert reloaded.to_str() == dumped


def test_reloaded_tokenizer_encodes_identically():
    tokenizer, _vocab, _merges, _specials = build_tiny_tokenizer()
    reloaded = Tokenizer.from_str(tokenizer.to_str())
    for text in ["abc 123 def", "大家好", "<|bos|>x<|eos|>"]:
        assert tokenizer.encode(text).ids == reloaded.encode(text).ids
        assert tokenizer.decode(tokenizer.encode(text).ids) == reloaded.decode(
            reloaded.encode(text).ids
        )


def test_loaded_export_to_str_reparses_identically():
    tokenizer = Tokenizer.from_file(FIXTURE)
    emitted = tokenizer.to_str()
    reparsed = Tokenizer.from_str(emitted)
    assert reparsed.to_str() == emitted


def test_loaded_export_to_str_matches_schema_keys():
    tokenizer = Tokenizer.from_file(FIXTURE)
    payload = json.loads(tokenizer.to_str())
    assert set(payload.keys()) == {
        "version",
        "truncation",
        "padding",
        "added_tokens",
        "normalizer",
        "pre_tokenizer",
        "post_processor",
        "decoder",
        "model",
    }
    assert payload["model"]["type"] == "BPE"
    assert payload["pre_tokenizer"]["type"] == "ByteLevel"
    assert payload["decoder"]["type"] == "ByteLevel"


def test_save_writes_file(tmp_path):
    tokenizer, _vocab, _merges, _specials = build_tiny_tokenizer()
    out = tmp_path / "tokenizer.json"
    tokenizer.save(out, pretty=True)
    assert out.exists()
    reloaded = Tokenizer.from_file(out)
    assert reloaded.to_str() == tokenizer.to_str()
