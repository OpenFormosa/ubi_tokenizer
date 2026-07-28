"""Unimplemented BPE / trainer options must fail loudly instead of silently.

``BPE`` and ``BpeTrainer`` accept several HuggingFace-compatible options that are
stored and serialized but never acted on. Accepting them silently means a caller
gets wrong tokenization with no signal, so a non-default value now raises.
Default values stay silent: they are true no-ops in HuggingFace too, and
GPT-2-lineage ``tokenizer.json`` files ship ``""`` for the affix options.
"""

import pytest

from ubi_tokenizer.errors import ModelError, UbiTokenizerError, UnsupportedFeatureError
from ubi_tokenizer.models.bpe import BPE
from ubi_tokenizer.trainers.bpe import BpeTrainer


def test_unsupported_feature_error_is_a_ubi_tokenizer_error():
    assert issubclass(UnsupportedFeatureError, UbiTokenizerError)


# -- dropout ---------------------------------------------------------------


@pytest.mark.parametrize("value", [None, 0.0])
def test_no_op_dropout_is_accepted(value):
    # HuggingFace treats both as "no dropout", so they must not raise.
    assert BPE(dropout=value).dropout == value


@pytest.mark.parametrize("value", [0.1, 0.5, 1.0])
def test_active_dropout_is_rejected(value):
    with pytest.raises(UnsupportedFeatureError, match="dropout"):
        BPE(dropout=value)


@pytest.mark.parametrize("value", [-0.1, 1.5])
def test_out_of_range_dropout_is_rejected_as_invalid_model_state(value):
    # HuggingFace rejects these at construction as invalid rather than
    # unimplemented, so they are ModelError and not UnsupportedFeatureError.
    with pytest.raises(ModelError, match="dropout"):
        BPE(dropout=value)


# -- affix options --------------------------------------------------------


@pytest.mark.parametrize("option", ["continuing_subword_prefix", "end_of_word_suffix"])
@pytest.mark.parametrize("value", [None, ""])
def test_no_op_affix_options_are_accepted(option, value):
    assert getattr(BPE(**{option: value}), option) == value


@pytest.mark.parametrize(
    ("option", "value"),
    [("continuing_subword_prefix", "##"), ("end_of_word_suffix", "</w>")],
)
def test_active_affix_options_are_rejected(option, value):
    with pytest.raises(UnsupportedFeatureError, match=option):
        BPE(**{option: value})


@pytest.mark.parametrize(
    ("option", "value"),
    [("continuing_subword_prefix", "##"), ("end_of_word_suffix", "</w>")],
)
def test_trainer_rejects_active_affix_options(option, value):
    # The trainer stores these two and never reads them either; fixing only the
    # model would leave half of the same defect in place.
    with pytest.raises(UnsupportedFeatureError, match=option):
        BpeTrainer(**{option: value})


@pytest.mark.parametrize("option", ["continuing_subword_prefix", "end_of_word_suffix"])
@pytest.mark.parametrize("value", [None, ""])
def test_trainer_accepts_no_op_affix_options(option, value):
    assert getattr(BpeTrainer(**{option: value}), option) == value


def test_trainer_defaults_are_accepted():
    trainer = BpeTrainer(vocab_size=64, min_frequency=1)
    assert trainer.continuing_subword_prefix is None
    assert trainer.end_of_word_suffix is None


# -- message quality -------------------------------------------------------


def test_rejection_message_names_the_option_and_says_it_is_unimplemented():
    with pytest.raises(UnsupportedFeatureError) as excinfo:
        BPE(end_of_word_suffix="</w>")
    message = str(excinfo.value)
    assert "end_of_word_suffix" in message
    assert "not implemented" in message.lower()


# -- cache_capacity is accepted, not rejected ------------------------------


def test_cache_capacity_is_retained():
    # A cache is a pure optimization: its absence cannot change output, so this
    # is accepted rather than rejected. It was previously discarded entirely.
    assert BPE(cache_capacity=1234).cache_capacity == 1234


def test_cache_capacity_stays_out_of_serialization():
    # HuggingFace does not serialize cache_capacity into tokenizer.json, so
    # omitting it keeps to_dict() faithful rather than lossy.
    assert "cache_capacity" not in BPE(cache_capacity=1234).to_dict()


# -- deserialization path --------------------------------------------------


def test_unsupported_option_in_tokenizer_json_is_rejected_on_load():
    from ubi_tokenizer.tokenizer import Tokenizer

    payload = {
        "version": "1.0",
        "added_tokens": [],
        "model": {"type": "BPE", "vocab": {"a": 0}, "merges": [], "dropout": 0.5},
    }
    with pytest.raises(UnsupportedFeatureError, match="dropout"):
        Tokenizer.from_dict(payload)
