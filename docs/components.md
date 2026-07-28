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
  resolves each character onto vocabulary tokens, then greedily applies the
  lowest-rank applicable merge until none remain (identical to the UbiTok
  reference runtime).

Resolution happens **before** merging, because the tokens it produces take part
in merges themselves — a vocabulary containing `<0xE4><0xB8>` plus a merge over
its two halves tokenizes `中` as `['<0xE4><0xB8>', '<0xAD>']`.

| Option | Effect |
| --- | --- |
| `unk_token` | out-of-vocabulary characters become this token. Without it they are **dropped**, matching HuggingFace |
| `fuse_unk` | consecutive unresolvable characters collapse into a single unk |
| `byte_fallback` | an out-of-vocabulary character becomes its UTF-8 bytes as `<0xXX>` tokens (uppercase, zero-padded). All-or-nothing per character: one missing byte token sends the whole character to unk |
| `ignore_merges` | a piece that is already a vocabulary entry is emitted whole, skipping merges. Required by GPT-4o / Llama-3 tokenizers |

Setting `unk_token` to a string that is absent from the vocabulary is allowed at
construction and raises `ModelError` only if unresolvable input actually arrives,
which is when HuggingFace raises too.

### Two behaviours worth knowing

**A pending unk survives a byte-fallback hit.** HuggingFace defers at most one
unresolved character, and a successful byte expansion does not flush it. With
`<0xE4>` missing from the vocabulary, `'中é'` gives
`['<0xC3>', '<0xA9>', '<unk>']` — the unk for `中` lands *after* the bytes of `é`.
This is likely an upstream quirk, but byte-parity with HuggingFace is the point
of this package, so it is reproduced rather than corrected.

**`byte_fallback` is encode-only.** The `ByteLevel` decoder maps each character
through the byte table, so it renders `<0xE4>` as the six ASCII characters of its
own name rather than as part of `中`. Round-tripping byte-fallback output needs a
`ByteFallback` decoder, which does not exist yet.

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

## Unimplemented options

`BPE` and `BpeTrainer` accept the full HuggingFace option set so that a
`tokenizer.json` round-trips unchanged. Options that are not implemented raise
`UnsupportedFeatureError` when set to a value that would change tokenization,
rather than being accepted and ignored:

| Option | Accepted values | Rejected values |
| --- | --- | --- |
| `dropout` | `None`, `0.0` | anything in `(0, 1]` — outside `[0, 1]` raises `ModelError` |
| `continuing_subword_prefix` | `None`, `""` | any non-empty string |
| `end_of_word_suffix` | `None`, `""` | any non-empty string |

The accepted values are genuine no-ops in HuggingFace as well, so files that
tokenize correctly today keep loading — GPT-2-lineage exports ship `""` for both
affix options.

`cache_capacity` is accepted and retained but not yet used. It is not rejected
because a cache is a pure optimization: its absence cannot change output.
HuggingFace does not serialize it into `tokenizer.json`, so `to_dict()` omits it.
