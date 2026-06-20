"""The :class:`Tokenizer` orchestrator.

Mirrors ``tokenizers.Tokenizer``: it wires together a normalizer, a
pre-tokenizer, a model, a post-processor and a decoder, and exposes
``encode``/``decode`` plus tokenizer.json (de)serialization. Special-token
splitting and byte-level encode/decode are lifted from the UbiTok seed
``ByteLevelBPE`` so encoding is byte-identical to the reference.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Optional

from ubi_tokenizer.added_tokens import AddedToken
from ubi_tokenizer.decoders import build_decoder
from ubi_tokenizer.decoders.byte_level import ByteLevel as ByteLevelDecoder
from ubi_tokenizer.encoding import Encoding
from ubi_tokenizer.errors import DeserializationError
from ubi_tokenizer.models.bpe import BPE
from ubi_tokenizer.normalizers import build_normalizer
from ubi_tokenizer.pre_tokenizers import build_pre_tokenizer
from ubi_tokenizer.pre_tokenizers.byte_level import ByteLevel as ByteLevelPreTokenizer
from ubi_tokenizer.processors import build_post_processor


class _PaddingConfig:
    def __init__(
        self,
        *,
        direction: str = "right",
        pad_id: int = 0,
        pad_type_id: int = 0,
        pad_token: str = "[PAD]",
        length: Optional[int] = None,
    ) -> None:
        self.direction = direction
        self.pad_id = pad_id
        self.pad_type_id = pad_type_id
        self.pad_token = pad_token
        self.length = length


class _TruncationConfig:
    def __init__(
        self,
        *,
        max_length: int = 512,
        stride: int = 0,
        direction: str = "right",
    ) -> None:
        self.max_length = max_length
        self.stride = stride
        self.direction = direction


class Tokenizer:
    """A complete tokenization pipeline around a model."""

    def __init__(self, model: BPE) -> None:
        self.model = model
        self._normalizer = None
        self._pre_tokenizer: Any = None
        self._post_processor = None
        self._decoder: Any = None
        self.version = "1.0"

        # Registered tokens added on top of the model's vocabulary.
        self._added_tokens: dict[str, AddedToken] = {}
        self._added_token_ids: dict[str, int] = {}
        self._added_id_to_token: dict[int, str] = {}
        self._special_by_length: list[str] = []

        self._padding: Optional[_PaddingConfig] = None
        self._truncation: Optional[_TruncationConfig] = None

        # Seed registered tokens from the model vocab special-looking entries is
        # not done automatically; callers register via add_special_tokens or the
        # from_str path.

    # -- component properties ---------------------------------------------
    @property
    def normalizer(self):
        return self._normalizer

    @normalizer.setter
    def normalizer(self, value) -> None:
        self._normalizer = value

    @property
    def pre_tokenizer(self):
        return self._pre_tokenizer

    @pre_tokenizer.setter
    def pre_tokenizer(self, value) -> None:
        self._pre_tokenizer = value

    @property
    def post_processor(self):
        return self._post_processor

    @post_processor.setter
    def post_processor(self, value) -> None:
        self._post_processor = value

    @property
    def decoder(self):
        return self._decoder

    @decoder.setter
    def decoder(self, value) -> None:
        self._decoder = value

    # -- added / special tokens -------------------------------------------
    def add_tokens(self, tokens: Iterable) -> int:
        """Register non-special added tokens. Returns the number newly added."""
        return self._register_tokens(tokens, special=False)

    def add_special_tokens(self, tokens: Iterable) -> int:
        """Register special added tokens. Returns the number newly added."""
        return self._register_tokens(tokens, special=True)

    def _register_tokens(self, tokens: Iterable, special: bool) -> int:
        added = 0
        for token in tokens:
            if isinstance(token, AddedToken):
                added_token = token
                if special:
                    added_token.special = True
            else:
                added_token = AddedToken(str(token), special=special, normalized=not special)
            content = added_token.content
            if content in self._added_tokens:
                continue
            token_id = self.model.token_to_id(content)
            if token_id is None:
                token_id = self.get_vocab_size(with_added_tokens=True)
                self.model.vocab[content] = token_id
                self.model._rebuild()
            self._added_tokens[content] = added_token
            self._added_token_ids[content] = token_id
            self._added_id_to_token[token_id] = content
            added += 1
        self._rebuild_special_index()
        return added

    def _rebuild_special_index(self) -> None:
        specials = [
            content
            for content, token in self._added_tokens.items()
            if token.special
        ]
        self._special_by_length = sorted(specials, key=len, reverse=True)

    @property
    def _special_token_set(self) -> set[str]:
        return set(self._special_by_length)

    # -- special-token splitting (lifted from ByteLevelBPE._split_special) -
    def _split_special(self, text: str) -> list[tuple[str, bool]]:
        if not self._special_by_length:
            return [(text, False)] if text else []
        chunks: list[tuple[str, bool]] = []
        position = 0
        while position < len(text):
            matched = None
            for token in self._special_by_length:
                if text.startswith(token, position):
                    matched = token
                    break
            if matched is not None:
                chunks.append((matched, True))
                position += len(matched)
                continue
            next_position = position + 1
            while next_position < len(text):
                if any(
                    text.startswith(token, next_position)
                    for token in self._special_by_length
                ):
                    break
                next_position += 1
            chunks.append((text[position:next_position], False))
            position = next_position
        return chunks

    # -- vocabulary access -------------------------------------------------
    def get_vocab(self, with_added_tokens: bool = True) -> dict[str, int]:
        vocab = self.model.get_vocab()
        if with_added_tokens:
            for content, token_id in self._added_token_ids.items():
                vocab[content] = token_id
        return vocab

    def get_vocab_size(self, with_added_tokens: bool = True) -> int:
        return len(self.get_vocab(with_added_tokens=with_added_tokens))

    def token_to_id(self, token: str) -> Optional[int]:
        if token in self._added_token_ids:
            return self._added_token_ids[token]
        return self.model.token_to_id(token)

    def id_to_token(self, token_id: int) -> Optional[str]:
        if token_id in self._added_id_to_token:
            return self._added_id_to_token[token_id]
        return self.model.id_to_token(token_id)

    # -- padding / truncation ---------------------------------------------
    def enable_padding(
        self,
        *,
        direction: str = "right",
        pad_id: int = 0,
        pad_type_id: int = 0,
        pad_token: str = "[PAD]",
        length: Optional[int] = None,
        pad_to_multiple_of: Optional[int] = None,
    ) -> None:
        self._padding = _PaddingConfig(
            direction=direction,
            pad_id=pad_id,
            pad_type_id=pad_type_id,
            pad_token=pad_token,
            length=length,
        )

    def no_padding(self) -> None:
        self._padding = None

    def enable_truncation(
        self,
        max_length: int,
        *,
        stride: int = 0,
        direction: str = "right",
        strategy: str = "longest_first",
    ) -> None:
        self._truncation = _TruncationConfig(
            max_length=max_length, stride=stride, direction=direction
        )

    def no_truncation(self) -> None:
        self._truncation = None

    # -- encode ------------------------------------------------------------
    def encode(
        self,
        sequence,
        pair=None,
        *,
        add_special_tokens: bool = True,
        is_pretokenized: bool = False,
    ) -> Encoding:
        encoding = self._encode_single(sequence, is_pretokenized=is_pretokenized)
        pair_encoding = (
            self._encode_single(pair, is_pretokenized=is_pretokenized)
            if pair is not None
            else None
        )
        if self._post_processor is not None:
            encoding = self._post_processor.process(
                encoding, pair_encoding, add_special_tokens
            )
        elif pair_encoding is not None:
            encoding = self._concat(encoding, pair_encoding)

        if self._truncation is not None:
            encoding.truncate(
                self._truncation.max_length,
                stride=self._truncation.stride,
                direction=self._truncation.direction,
            )
        if self._padding is not None and self._padding.length is not None:
            self._pad_encoding(encoding, self._padding.length)
        return encoding

    def _encode_single(self, sequence, *, is_pretokenized: bool) -> Encoding:
        encoding = Encoding()
        if is_pretokenized:
            return self._encode_pretokenized(sequence)
        word_index = 0
        for chunk, is_special in self._split_special(sequence):
            if not chunk:
                continue
            if is_special:
                token_id = self.token_to_id(chunk)
                encoding.ids.append(token_id)
                encoding.tokens.append(chunk)
                encoding.type_ids.append(0)
                encoding.offsets.append((0, 0))
                encoding.attention_mask.append(1)
                encoding.special_tokens_mask.append(1)
                encoding.word_ids.append(None)
                continue
            normalized = chunk
            if self._normalizer is not None:
                normalized = self._normalizer.normalize_str(chunk)
            pre = self._pre_tokenize(normalized)
            for piece, offset in pre:
                tokens = self.model.tokenize(piece)
                for token in tokens:
                    token_id = self.token_to_id(token)
                    encoding.ids.append(token_id if token_id is not None else -1)
                    encoding.tokens.append(token)
                    encoding.type_ids.append(0)
                    encoding.offsets.append(offset)
                    encoding.attention_mask.append(1)
                    encoding.special_tokens_mask.append(0)
                    encoding.word_ids.append(word_index)
                word_index += 1
        return encoding

    def _encode_pretokenized(self, words: Iterable[str]) -> Encoding:
        encoding = Encoding()
        for word_index, word in enumerate(words):
            normalized = word
            if self._normalizer is not None:
                normalized = self._normalizer.normalize_str(word)
            pre = self._pre_tokenize(normalized)
            for piece, offset in pre:
                for token in self.model.tokenize(piece):
                    token_id = self.token_to_id(token)
                    encoding.ids.append(token_id if token_id is not None else -1)
                    encoding.tokens.append(token)
                    encoding.type_ids.append(0)
                    encoding.offsets.append(offset)
                    encoding.attention_mask.append(1)
                    encoding.special_tokens_mask.append(0)
                    encoding.word_ids.append(word_index)
        return encoding

    def _pre_tokenize(self, text: str) -> list[tuple[str, tuple[int, int]]]:
        if self._pre_tokenizer is not None:
            return self._pre_tokenizer.pre_tokenize_str(text)
        # No pre-tokenizer: treat the whole string as one (already byte-mapped)
        # piece. This matches HF's "Whitespace-free" default for raw models.
        return [(text, (0, len(text)))] if text else []

    # -- decode ------------------------------------------------------------
    def decode(self, ids: Iterable[int], *, skip_special_tokens: bool = False) -> str:
        """Decode token ids back into text (byte runs via the decoder)."""
        output_parts: list[str] = []
        byte_run: list[str] = []
        decoder = self._decoder if self._decoder is not None else ByteLevelDecoder()
        special_set = self._special_token_set
        for token_id in ids:
            token = self.id_to_token(int(token_id))
            if token is None:
                continue
            if token in special_set or int(token_id) in self._added_id_to_token:
                if byte_run:
                    output_parts.append(decoder.decode(byte_run))
                    byte_run = []
                if token in special_set:
                    if not skip_special_tokens:
                        output_parts.append(token)
                    continue
                # Non-special added token: emit verbatim.
                output_parts.append(token)
                continue
            byte_run.append(token)
        if byte_run:
            output_parts.append(decoder.decode(byte_run))
        return "".join(output_parts)

    # -- batch -------------------------------------------------------------
    def encode_batch(
        self,
        inputs: list,
        *,
        add_special_tokens: bool = True,
        is_pretokenized: bool = False,
    ) -> list[Encoding]:
        encodings = [
            self.encode(
                item if not isinstance(item, tuple) else item[0],
                pair=item[1] if isinstance(item, tuple) and len(item) > 1 else None,
                add_special_tokens=add_special_tokens,
                is_pretokenized=is_pretokenized,
            )
            for item in inputs
        ]
        if self._padding is not None and self._padding.length is None:
            target = max((len(enc) for enc in encodings), default=0)
            for enc in encodings:
                self._pad_encoding(enc, target)
        return encodings

    def decode_batch(
        self, sequences: list[Iterable[int]], *, skip_special_tokens: bool = False
    ) -> list[str]:
        return [
            self.decode(ids, skip_special_tokens=skip_special_tokens)
            for ids in sequences
        ]

    def _pad_encoding(self, encoding: Encoding, length: int) -> None:
        assert self._padding is not None
        encoding.pad(
            length,
            direction=self._padding.direction,
            pad_id=self._padding.pad_id,
            pad_type_id=self._padding.pad_type_id,
            pad_token=self._padding.pad_token,
        )

    @staticmethod
    def _concat(a: Encoding, b: Encoding) -> Encoding:
        result = Encoding(
            ids=a.ids + b.ids,
            tokens=a.tokens + b.tokens,
            type_ids=a.type_ids + [1] * len(b.ids),
            offsets=a.offsets + b.offsets,
            attention_mask=a.attention_mask + b.attention_mask,
            special_tokens_mask=a.special_tokens_mask + b.special_tokens_mask,
            word_ids=a.word_ids + b.word_ids,
        )
        return result

    # -- training ----------------------------------------------------------
    def train(self, files: list[str], trainer) -> None:
        trainer.train(files, self.model)
        self._sync_added_after_train(trainer)

    def train_from_iterator(self, iterator: Iterable[str], trainer) -> None:
        trainer.train_from_iterator(iterator, self.model)
        self._sync_added_after_train(trainer)

    def _sync_added_after_train(self, trainer) -> None:
        # Re-register the trainer's special tokens against new vocab ids.
        self._added_tokens.clear()
        self._added_token_ids.clear()
        self._added_id_to_token.clear()
        specials = getattr(trainer, "special_tokens", []) or []
        self.add_special_tokens(specials)

    # -- serialization -----------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        added_tokens = [
            self._added_tokens[content].to_dict(token_id)
            for content, token_id in sorted(
                self._added_token_ids.items(), key=lambda item: item[1]
            )
        ]
        return {
            "version": self.version,
            "truncation": self._truncation_to_dict(),
            "padding": None,
            "added_tokens": added_tokens,
            "normalizer": self._normalizer.to_dict() if self._normalizer else None,
            "pre_tokenizer": self._pre_tokenizer.to_dict()
            if self._pre_tokenizer
            else None,
            "post_processor": self._post_processor.to_dict()
            if self._post_processor
            else None,
            "decoder": self._decoder.to_dict() if self._decoder else None,
            "model": self.model.to_dict(),
        }

    def _truncation_to_dict(self) -> Optional[dict[str, Any]]:
        if self._truncation is None:
            return None
        return {
            "direction": self._truncation.direction,
            "max_length": self._truncation.max_length,
            "strategy": "LongestFirst",
            "stride": self._truncation.stride,
        }

    def to_str(self, pretty: bool = False) -> str:
        payload = self.to_dict()
        if pretty:
            return json.dumps(payload, ensure_ascii=False, indent=2)
        return json.dumps(payload, ensure_ascii=False)

    def save(self, path: str | Path, pretty: bool = True) -> None:
        Path(path).write_text(self.to_str(pretty=pretty), encoding="utf-8")

    @classmethod
    def from_str(cls, json_str: str) -> "Tokenizer":
        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as exc:
            raise DeserializationError(f"Invalid tokenizer JSON: {exc.msg}") from exc
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Tokenizer":
        model_data = data.get("model")
        if not model_data or model_data.get("type") != "BPE":
            raise DeserializationError("Only BPE models are supported")
        model = BPE.from_dict(model_data)
        tokenizer = cls(model)
        tokenizer.version = data.get("version", "1.0")
        tokenizer._normalizer = build_normalizer(data.get("normalizer"))
        tokenizer._pre_tokenizer = build_pre_tokenizer(data.get("pre_tokenizer"))
        tokenizer._post_processor = build_post_processor(data.get("post_processor"))
        tokenizer._decoder = build_decoder(data.get("decoder"))

        truncation = data.get("truncation")
        if truncation is not None:
            tokenizer._truncation = _TruncationConfig(
                max_length=truncation.get("max_length", 512),
                stride=truncation.get("stride", 0),
                direction=truncation.get("direction", "right").lower(),
            )

        for entry in data.get("added_tokens", []):
            added = AddedToken.from_dict(entry)
            token_id = entry.get("id")
            if token_id is None:
                token_id = model.token_to_id(added.content)
            if token_id is None:
                token_id = tokenizer.get_vocab_size(with_added_tokens=True)
                model.vocab[added.content] = token_id
                model._rebuild()
            tokenizer._added_tokens[added.content] = added
            tokenizer._added_token_ids[added.content] = token_id
            tokenizer._added_id_to_token[token_id] = added.content
        tokenizer._rebuild_special_index()
        return tokenizer

    @classmethod
    def from_file(cls, path: str | Path) -> "Tokenizer":
        text = Path(path).read_text(encoding="utf-8")
        return cls.from_str(text)

    @classmethod
    def from_ubitok_dir(cls, directory: str | Path) -> "Tokenizer":
        """Load a UbiTok export directory (vocab.json + merges.txt + maps).

        Mirrors ``ByteLevelBPE.from_files``.
        """
        path = Path(directory)
        vocab = json.loads((path / "vocab.json").read_text(encoding="utf-8"))
        merges = _read_merges(path / "merges.txt")

        special_tokens: list[str] = []
        special_map_path = path / "special_tokens_map.json"
        if special_map_path.exists():
            special_map = json.loads(special_map_path.read_text(encoding="utf-8"))
            for value in special_map.values():
                if isinstance(value, str):
                    special_tokens.append(value)
                elif isinstance(value, list):
                    special_tokens.extend(
                        item for item in value if isinstance(item, str)
                    )

        unk = "<|unk|>" if "<|unk|>" in vocab else None
        model = BPE(vocab=vocab, merges=merges, unk_token=unk)
        tokenizer = cls(model)
        tokenizer.pre_tokenizer = ByteLevelPreTokenizer(
            add_prefix_space=False, use_regex=True, trim_offsets=True
        )
        tokenizer.decoder = ByteLevelDecoder(
            add_prefix_space=True, use_regex=True, trim_offsets=True
        )
        # Deduplicate while preserving order.
        seen: set[str] = set()
        ordered_specials = [
            token
            for token in special_tokens
            if not (token in seen or seen.add(token))
        ]
        tokenizer.add_special_tokens(ordered_specials)
        return tokenizer


def _read_merges(path: Path) -> list[tuple[str, str]]:
    merges: list[tuple[str, str]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        left, right = stripped.split()
        merges.append((left, right))
    return merges
