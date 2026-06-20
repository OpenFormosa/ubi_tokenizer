# Loading a UbiTok export

[UbiTok](https://github.com/voidful) handles large / distributed tokenizer
*training* and writes an export directory. `ubi_tokenizer` is the matching
dependency-free *runtime*.

## From a `tokenizer.json`

UbiTok writes a `tokenizer.json` in the same schema HuggingFace uses. Load it
directly:

```python
from ubi_tokenizer import Tokenizer

tok = Tokenizer.from_file("ubitok_export/tokenizer.json")
```

The schema's top-level keys are `version`, `truncation`, `padding`,
`added_tokens`, `normalizer`, `pre_tokenizer`, `post_processor`, `decoder`, and
`model`. The `model` block is a `BPE` with a byte-mapped `vocab` (`{token: id}`)
and `merges` (`["left right", ...]`). Vocab tokens and merges are already in
`bytes_to_unicode` (byte-mapped) space.

## From a raw export directory

UbiTok also emits `vocab.json`, `merges.txt`, and `special_tokens_map.json`. Load
that trio with `from_ubitok_dir`, which mirrors `ByteLevelBPE.from_files`:

```python
from ubi_tokenizer import Tokenizer

tok = Tokenizer.from_ubitok_dir("ubitok_export/")
```

This wires up a `ByteLevel` pre-tokenizer and decoder, registers every special
token from `special_tokens_map.json`, and resolves `<|unk|>` if present.

## Re-emitting

A loaded export re-serializes to the same schema:

```python
tok.save("roundtrip/tokenizer.json")
# Tokenizer.from_str(tok.to_str()) re-parses to an equivalent tokenizer.
```
