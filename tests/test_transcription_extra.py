"""The optional transcription extra."""

import pytest


def test_core_import_does_not_pull_transcription():
    import sys

    # Importing the core package must not import the transcription subpackage.
    for mod in list(sys.modules):
        if mod.startswith("ubi_tokenizer.transcription"):
            del sys.modules[mod]
    import ubi_tokenizer  # noqa: F401

    assert "ubi_tokenizer.transcription" not in sys.modules


def test_timestamp_format_roundtrip():
    from ubi_tokenizer.transcription.timestamp_format import (
        format_transcript_segments_json,
        parse_transcript_segments_json,
    )

    segments = [
        {"Start": 0.0, "End": 3.42, "Content": "測試"},
        {"Start": 3.42, "End": 6.0, "Content": "second"},
    ]
    serialized = format_transcript_segments_json(segments)
    parsed = parse_transcript_segments_json(serialized)
    assert parsed[0]["Content"] == "測試"
    assert parsed[0]["Start"] == 0.0
    assert parsed[0]["End"] == 3.42
    assert parsed[1]["Content"] == "second"


def test_format_timestamp_precision():
    from ubi_tokenizer.transcription.timestamp_format import format_timestamp_seconds

    assert format_timestamp_seconds(3.425, 2) == "3.43"
    assert format_timestamp_seconds(0, 2) == "0.00"


def test_dense_timestamp_guards():
    from ubi_tokenizer.transcription.special_tokens import (
        find_dense_timestamp_tokens,
        validate_no_dense_timestamp_tokens,
    )

    assert find_dense_timestamp_tokens(["<|ts_12|>", "ok", "<|time_3|>"]) == [
        "<|ts_12|>",
        "<|time_3|>",
    ]
    with pytest.raises(ValueError):
        validate_no_dense_timestamp_tokens(["<|timestamp_5|>"])
    # No prohibited tokens -> no error.
    validate_no_dense_timestamp_tokens(["<|transcript_start|>", "plain"])


def test_rich_transcription_tokens_available():
    from ubi_tokenizer.transcription import RICH_TRANSCRIPTION_TOKENS

    assert "<|transcript_start|>" in RICH_TRANSCRIPTION_TOKENS
    assert "<|segment_start|>" in RICH_TRANSCRIPTION_TOKENS
