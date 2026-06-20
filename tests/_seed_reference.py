"""Vendored copy of the UbiTok seed ByteLevelBPE runtime.

Copied (logic-identical) from
``UbiTok/src/taiwan_tokenizer/tokenizer_model.py`` so the test suite can assert
golden parity between ``ubi_tokenizer`` and the reference implementation without
importing ubitok/taiwan_tokenizer.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Iterator

# --- byte_utils.py (vendored) ---------------------------------------------


def bytes_to_unicode() -> dict[int, str]:
    byte_values = list(range(ord("!"), ord("~") + 1))
    byte_values += list(range(ord("¡"), ord("¬") + 1))
    byte_values += list(range(ord("®"), ord("ÿ") + 1))
    chars = byte_values[:]
    extra = 0
    for value in range(256):
        if value not in byte_values:
            byte_values.append(value)
            chars.append(256 + extra)
            extra += 1
    return dict(zip(byte_values, [chr(value) for value in chars]))


BYTE_ENCODER = bytes_to_unicode()
BYTE_DECODER = {value: key for key, value in BYTE_ENCODER.items()}


def encode_token_bytes(data: bytes) -> str:
    return "".join(BYTE_ENCODER[value] for value in data)


# --- pretokenize.py (vendored) --------------------------------------------

BYTE_LEVEL_PRETOKEN_PATTERN = re.compile(
    r"['’](?:s|t|re|ve|m|ll|d)|[^\W\d_]+|\d+|[^\s\w]+|_+|\s+",
    flags=re.IGNORECASE | re.UNICODE,
)


def iter_pretokenized_pieces(text: str) -> Iterator[str]:
    position = 0
    for match in BYTE_LEVEL_PRETOKEN_PATTERN.finditer(text):
        if match.start() > position:
            yield text[position : match.start()]
        yield match.group(0)
        position = match.end()
    if position < len(text):
        yield text[position:]


# --- tokenizer_model.py (vendored) ----------------------------------------


class ByteLevelBPE:
    def __init__(
        self,
        vocab: dict[str, int],
        merges: list[tuple[str, str]],
        special_tokens: Iterable[str],
    ) -> None:
        self.vocab = dict(vocab)
        self.id_to_token = {token_id: token for token, token_id in vocab.items()}
        self.special_tokens = list(special_tokens)
        self.special_token_set = set(self.special_tokens)
        self.merge_ranks = {pair: rank for rank, pair in enumerate(merges)}
        self._special_by_length = sorted(self.special_tokens, key=len, reverse=True)

    def encode(self, text: str) -> list[int]:
        ids: list[int] = []
        for chunk, is_special in self._split_special(text):
            if not chunk:
                continue
            if is_special:
                ids.append(self.vocab[chunk])
                continue
            for piece in iter_pretokenized_pieces(chunk):
                encoded_piece = encode_token_bytes(piece.encode("utf-8"))
                ids.extend(self.vocab[token] for token in self._apply_bpe(encoded_piece))
        return ids

    def decode(self, ids: Iterable[int], skip_special_tokens: bool = False) -> str:
        output_parts: list[str] = []
        byte_buffer = bytearray()
        for token_id in ids:
            token = self.id_to_token[int(token_id)]
            if token in self.special_token_set:
                if byte_buffer:
                    output_parts.append(byte_buffer.decode("utf-8", errors="replace"))
                    byte_buffer.clear()
                if not skip_special_tokens:
                    output_parts.append(token)
                continue
            for char in token:
                byte_buffer.append(BYTE_DECODER[char])
        if byte_buffer:
            output_parts.append(byte_buffer.decode("utf-8", errors="replace"))
        return "".join(output_parts)

    def _apply_bpe(self, encoded_piece: str) -> list[str]:
        symbols = tuple(encoded_piece)
        if len(symbols) < 2:
            return list(symbols)
        while True:
            pairs = [
                (symbols[index], symbols[index + 1])
                for index in range(len(symbols) - 1)
            ]
            ranked = [
                (self.merge_ranks[pair], pair)
                for pair in pairs
                if pair in self.merge_ranks
            ]
            if not ranked:
                break
            _, best_pair = min(ranked)
            merged: list[str] = []
            index = 0
            while index < len(symbols):
                if (
                    index < len(symbols) - 1
                    and symbols[index] == best_pair[0]
                    and symbols[index + 1] == best_pair[1]
                ):
                    merged.append(symbols[index] + symbols[index + 1])
                    index += 2
                else:
                    merged.append(symbols[index])
                    index += 1
            symbols = tuple(merged)
            if len(symbols) < 2:
                break
        return list(symbols)

    def _split_special(self, text: str) -> list[tuple[str, bool]]:
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
