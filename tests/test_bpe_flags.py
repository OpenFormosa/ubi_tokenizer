"""BPE option behaviour, checked against recorded HuggingFace output.

The expectations in ``fixtures/hf_bpe_flags.json`` were measured by running real
``huggingface/tokenizers``; see ``scripts/gen_hf_flag_fixtures.py``. This module
reads that recording using only the standard library, so the suite keeps working
without any third-party package installed.
"""

import json
from pathlib import Path

import pytest

from ubi_tokenizer.errors import ModelError
from ubi_tokenizer.models.bpe import BPE

FIXTURE = Path(__file__).parent / "fixtures" / "hf_bpe_flags.json"


def _load_cases():
    # A missing fixture is a hard failure, never a skip: silently skipping would
    # remove all HuggingFace parity coverage without anyone noticing.
    if not FIXTURE.exists():
        raise AssertionError(
            f"missing ground-truth fixture {FIXTURE}; "
            "regenerate with scripts/gen_hf_flag_fixtures.py"
        )
    return json.loads(FIXTURE.read_text(encoding="utf-8"))["cases"]


CASES = _load_cases()


def _model(case):
    return BPE(
        vocab=dict(case["vocab"]),
        merges=[tuple(pair) for pair in case["merges"]],
        **case["flags"],
    )


def _expectations():
    for case in CASES:
        for text, expected in case["expected"].items():
            yield pytest.param(case, text, expected, id=f"{case['name']}-{text!r}")


@pytest.mark.parametrize(("case", "text", "expected"), list(_expectations()))
def test_matches_huggingface(case, text, expected):
    assert _model(case).tokenize(text) == expected, case["why"]


def test_fixture_records_its_ground_truth_version():
    payload = json.loads(FIXTURE.read_text(encoding="utf-8"))
    assert payload["_ground_truth"].startswith("huggingface tokenizers ")


def test_ignore_merges_is_actually_discriminating():
    # Guards the fixture set itself: if these two cases ever agree, the
    # ignore_merges coverage has quietly become vacuous.
    off = next(c for c in CASES if c["name"] == "ignore_merges_off_control")
    on = next(c for c in CASES if c["name"] == "ignore_merges_discriminating")
    assert off["expected"] != on["expected"]


# -- behaviour with no HuggingFace counterpart ------------------------------


def test_unk_token_absent_from_vocab_raises():
    # HuggingFace raises lazily at tokenize time rather than at construction,
    # and so do we. Constructing must stay quiet.
    model = BPE(vocab={"a": 0}, merges=[], unk_token="<unk>")
    assert model.tokenize("a") == ["a"]
    with pytest.raises(ModelError, match="<unk>"):
        model.tokenize("az")


def test_byte_fallback_hit_still_works_when_unk_is_absent_from_vocab():
    # Only the unk path needs the unk token, so a pure byte-fallback hit must
    # not be dragged down by a missing unk.
    vocab = {f"<0x{value:02X}>": value for value in range(256)}
    model = BPE(vocab=vocab, merges=[], unk_token="<unk>", byte_fallback=True)
    assert model.tokenize("a") == ["<0x61>"]


def test_lone_surrogate_propagates_encoding_error():
    # Pure Python can be handed a lone surrogate where HuggingFace's binding
    # layer would already have rejected it. Propagating is deliberate.
    model = BPE(vocab={"a": 0}, merges=[], byte_fallback=True)
    with pytest.raises(UnicodeEncodeError):
        model.tokenize("\ud800")
