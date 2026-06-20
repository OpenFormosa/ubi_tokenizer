# ubi_tokenizer

**Ubiquitous, dependency-free, pure-Python runtime for HuggingFace-tokenizers-style byte-level BPE.**

`ubi_tokenizer` ("ubi" = *ubiquitous*) mirrors the architecture and public API of
[`huggingface/tokenizers`](https://github.com/huggingface/tokenizers) — `Tokenizer`,
`Encoding`, `AddedToken`, and the `models` / `pre_tokenizers` / `decoders` /
`normalizers` / `processors` / `trainers` submodules — but is implemented **entirely
with the Python standard library**. No Rust, no `torch`, no `transformers`, no
third-party dependencies at all.

It reads and writes the **same `tokenizer.json`** that HuggingFace and
[UbiTok](#sibling-repos) produce, so you can train a tokenizer elsewhere and run it
anywhere Python runs.

## Install

```bash
pip install ubi-tokenizer
# optional rich-transcription helpers (stdlib-only namespace gate)
pip install "ubi-tokenizer[transcription]"
```

## Quick tour

```python
from ubi_tokenizer import Tokenizer

# Load a tokenizer.json (HuggingFace / UbiTok schema).
tok = Tokenizer.from_file("tokenizer.json")

enc = tok.encode("大家好，歡迎回來。")
print(enc.ids)      # -> [..token ids..]
print(enc.tokens)   # -> ['å¤§', 'å®¶', ...]  (byte-mapped subwords)
print(enc.offsets)  # -> [(0, 1), (1, 2), ...] character offsets

print(tok.decode(enc.ids))  # -> "大家好，歡迎回來。"
```

`encode` returns an `Encoding` with parallel arrays: `ids`, `tokens`, `type_ids`,
`offsets`, `attention_mask`, `special_tokens_mask`, `word_ids`.

### Batching and padding

```python
tok.enable_padding(pad_id=tok.token_to_id("<|pad|>"), pad_token="<|pad|>")
batch = tok.encode_batch(["abc", "大家好", "123 456"])
# every encoding is padded to the longest in the batch
texts = tok.decode_batch([e.ids for e in batch])
```

### Special tokens

```python
tok.add_special_tokens(["<|bos|>", "<|eos|>"])
enc = tok.encode("<|bos|>hello<|eos|>")
# special tokens are split out longest-match-first and flagged in
# enc.special_tokens_mask; decode(skip_special_tokens=True) drops them.
```

### Training (small corpora)

```python
from ubi_tokenizer import Tokenizer
from ubi_tokenizer.models import BPE
from ubi_tokenizer.pre_tokenizers import ByteLevel as ByteLevelPre
from ubi_tokenizer.decoders import ByteLevel as ByteLevelDec
from ubi_tokenizer.trainers import BpeTrainer

tok = Tokenizer(BPE())
tok.pre_tokenizer = ByteLevelPre()
tok.decoder = ByteLevelDec()

trainer = BpeTrainer(vocab_size=2000, min_frequency=2, special_tokens=["<|unk|>"])
tok.train_from_iterator(corpus_lines, trainer)
tok.save("tokenizer.json")
```

> For large/distributed training use **UbiTok**; the built-in `BpeTrainer` is for
> small corpora and tests.

## How it compares to HuggingFace `tokenizers`

| | HuggingFace `tokenizers` | `ubi_tokenizer` |
| --- | --- | --- |
| `tokenizer.json` format | yes | **same format** |
| Implementation | Rust core + Python bindings | **pure Python (stdlib only)** |
| Dependencies | Rust toolchain / wheels | **none** |
| Use case | training + inference at scale | **ubiquitous runtime / inference** |

The pipeline is the familiar one: **Normalizer → PreTokenizer → Model →
PostProcessor → Decoder** (see [`docs/components.md`](docs/components.md)).

## Sibling repos

- **UbiTok** — distributed / large-scale tokenizer *training*. Export a tokenizer
  there, run it here. See [`docs/loading-a-ubitok-export.md`](docs/loading-a-ubitok-export.md).
- **PangolinBench** — tokenizer *benchmarks*.

## License

MIT © 2026 voidful
