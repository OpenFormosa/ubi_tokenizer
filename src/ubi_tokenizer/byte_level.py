"""Byte-level vocabulary representation helpers.

Ported verbatim from the UbiTok seed (``byte_utils.py``). Provides the
reversible byte<->Unicode mapping used by byte-level BPE tokenizers.
"""

from __future__ import annotations


def bytes_to_unicode() -> dict[int, str]:
    """Return the reversible byte-to-Unicode map used by byte-level BPE."""
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


BYTE_ENCODER: dict[int, str] = bytes_to_unicode()
BYTE_DECODER: dict[str, int] = {value: key for key, value in BYTE_ENCODER.items()}


def encode_token_bytes(data: bytes) -> str:
    """Encode raw bytes into the reversible byte-level token string."""
    return "".join(BYTE_ENCODER[value] for value in data)


def decode_token_bytes(text: str) -> bytes:
    """Decode a byte-level token string back into raw bytes."""
    try:
        return bytes(BYTE_DECODER[char] for char in text)
    except KeyError as exc:
        raise ValueError(f"token contains non byte-level character: {exc}") from exc
