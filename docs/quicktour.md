# Quick tour

`ubi_tokenizer` mirrors the public API of `huggingface/tokenizers`, so if you have
used that library this will feel familiar.

## Load and encode

```python
from ubi_tokenizer import Tokenizer

tok = Tokenizer.from_file("tokenizer.json")
enc = tok.encode("abc 123 def")

enc.ids                  # list[int]
enc.tokens               # list[str] (byte-mapped subwords)
enc.offsets              # list[(start, end)] character offsets
enc.attention_mask       # all 1s for real tokens
enc.special_tokens_mask  # 1 for special tokens, else 0
enc.type_ids             # all 0 for a single sequence
enc.word_ids             # pre-token index each token came from
```

## Decode

```python
text = tok.decode(enc.ids)                          # round-trips exactly
clean = tok.decode(enc.ids, skip_special_tokens=True)
```

`decode` routes special-token ids straight to their literal content and buffers
runs of byte-mapped tokens through the `ByteLevel` decoder, flushing the byte
buffer to UTF-8 at every special-token boundary.

## Batch

```python
encs = tok.encode_batch(["a", "bb", "ccc"])
texts = tok.decode_batch([e.ids for e in encs])
```

With `enable_padding()` the batch is padded to the longest member; padded
positions get `attention_mask == 0`.

## Pair encoding

```python
enc = tok.encode("question", "context")
# without a post_processor the two sequences are concatenated with
# type_ids 0 (first) and 1 (second).
```

## Serialize

```python
tok.save("out/tokenizer.json", pretty=True)
again = Tokenizer.from_str(tok.to_str())   # equivalent tokenizer
```
