# Transcription extra

The optional `transcription` extra provides rich-transcription timestamp
formatting and special-token guards. It is **not** imported by the core package —
`import ubi_tokenizer` never pulls it in — and it remains stdlib-only. The extra
exists as a namespace gate, not to add third-party dependencies.

```bash
pip install "ubi-tokenizer[transcription]"
```

## Timestamp formatting

```python
from ubi_tokenizer.transcription.timestamp_format import (
    format_transcript_segments_json,
    parse_transcript_segments_json,
    format_timestamp_seconds,
)

segments = [{"Start": 0.0, "End": 3.42, "Content": "測試"}]
serialized = format_transcript_segments_json(segments)
# '[{"Start":0.00,"End":3.42,"Content":"測試"}]'
parsed = parse_transcript_segments_json(serialized)

format_timestamp_seconds(3.425, 2)  # "3.43" (ROUND_HALF_UP, fixed decimals)
```

`parse_transcript_segments_json` also accepts text wrapped in
`<|transcript_start|>` / `<|transcript_end|>` delimiters.

## Rich-transcription tokens and dense-timestamp guards

```python
from ubi_tokenizer.transcription import (
    RICH_TRANSCRIPTION_TOKENS,
    find_dense_timestamp_tokens,
    validate_no_dense_timestamp_tokens,
)

RICH_TRANSCRIPTION_TOKENS  # ['<|transcript_start|>', '<|segment_start|>', ...]

# Dense per-frame timestamp tokens (<|ts_12|>, <|timestamp_5|>, <|time_3|>) are
# prohibited; the guard raises ValueError if any are present.
validate_no_dense_timestamp_tokens(my_special_tokens)
```
