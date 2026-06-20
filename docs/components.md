# Components

`ubi_tokenizer` uses the same five-stage pipeline as `huggingface/tokenizers`:

```
Normalizer → PreTokenizer → Model → PostProcessor → Decoder
```

Each stage is optional except the **Model**, and each is independently
configurable on a `Tokenizer` instance.

## Normalizer (`ubi_tokenizer.normalizers`)

Cleans raw text before pre-tokenization. Stdlib-backed implementations:

- `NFC`, `NFD`, `NFKC`, `NFKD` — Unicode normalization via `unicodedata.normalize`.
- `Lowercase` — case folding.
- `Replace(pattern, content, is_regex=False)` — literal or regex substitution.
- `Sequence([...])` — apply several in order.

```python
from ubi_tokenizer.normalizers import NFC, Lowercase, Sequence
tok.normalizer = Sequence([NFC(), Lowercase()])
```

## PreTokenizer (`ubi_tokenizer.pre_tokenizers`)

Splits normalized text into word-like pieces and maps each into reversible
byte-mapped space.

- `ByteLevel(add_prefix_space=False, use_regex=True, trim_offsets=True)` — the
  workhorse; returns `(byte_mapped_piece, (char_start, char_end))` pairs.
- `Sequence([...])` — chain pre-tokenizers.

## Model (`ubi_tokenizer.models`)

Turns one byte-mapped piece into subword tokens.

- `BPE(vocab, merges, *, unk_token=None, ...)` — byte-pair encoding. `tokenize`
  greedily applies the lowest-rank applicable merge until none remain (identical
  to the UbiTok reference runtime).

## PostProcessor (`ubi_tokenizer.processors`)

Adds template special tokens and assembles single/pair encodings.

- `PostProcessor` — identity (default).
- `TemplateProcessing(single, pair, special_tokens)` — wrap with literal tokens
  such as `"[CLS] $A [SEP]"`.
- `Sequence([...])` — chain processors.

## Decoder (`ubi_tokenizer.decoders`)

Reconstructs text from tokens.

- `ByteLevel(...)` — buffers byte-mapped tokens and flushes the byte buffer to
  UTF-8, routing special tokens to their literal content.

## Trainers (`ubi_tokenizer.trainers`)

- `BpeTrainer(vocab_size, min_frequency, special_tokens=None, ...)` — in-memory
  greedy trainer. For large/distributed training use UbiTok; this is for small
  corpora and tests.
