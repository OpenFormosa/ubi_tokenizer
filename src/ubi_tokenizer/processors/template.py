"""TemplateProcessing and Sequence post-processors.

A small subset of ``tokenizers.processors.TemplateProcessing``: it wraps a
single/pair encoding with literal special tokens described by a template such
as ``"[CLS] $A [SEP]"``.
"""

from __future__ import annotations

from typing import Any, Optional

from ubi_tokenizer.encoding import Encoding
from ubi_tokenizer.processors.base import PostProcessor


class TemplateProcessing(PostProcessor):
    """Wrap encodings with literal special tokens from a template string."""

    def __init__(
        self,
        single: str | list[str] | None = None,
        pair: str | list[str] | None = None,
        special_tokens: Optional[list[tuple[str, int]]] = None,
    ) -> None:
        self.single = _as_pieces(single) if single is not None else ["$A"]
        self.pair = _as_pieces(pair) if pair is not None else None
        self.special_tokens: dict[str, int] = {
            content: token_id for content, token_id in (special_tokens or [])
        }

    def process(
        self,
        encoding: Encoding,
        pair: Optional[Encoding] = None,
        add_special_tokens: bool = True,
    ) -> Encoding:
        if not add_special_tokens:
            if pair is None:
                return encoding
            return _concat(encoding, pair)

        template = self.pair if pair is not None else self.single
        if template is None:
            template = self.single

        result = Encoding()
        for piece in template:
            if piece == "$A" or piece.startswith("$A"):
                _extend(result, encoding, type_id=_type_of(piece, 0))
            elif piece == "$B" or piece.startswith("$B"):
                if pair is not None:
                    _extend(result, pair, type_id=_type_of(piece, 1))
            else:
                self._append_special(result, piece)
        return result

    def _append_special(self, result: Encoding, content: str) -> None:
        token_id = self.special_tokens.get(content)
        if token_id is None:
            raise ValueError(f"Template special token not registered: {content!r}")
        result.ids.append(token_id)
        result.tokens.append(content)
        result.type_ids.append(0)
        result.offsets.append((0, 0))
        result.attention_mask.append(1)
        result.special_tokens_mask.append(1)
        result.word_ids.append(None)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TemplateProcessing":
        single = _template_from_json(data.get("single"))
        pair = _template_from_json(data.get("pair"))
        special: list[tuple[str, int]] = []
        for content, spec in (data.get("special_tokens") or {}).items():
            ids = spec.get("ids", [])
            if ids:
                special.append((content, ids[0]))
        return cls(single=single, pair=pair, special_tokens=special)

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "TemplateProcessing",
            "single": _template_to_json(self.single),
            "pair": _template_to_json(self.pair) if self.pair else [],
            "special_tokens": {
                content: {"id": content, "ids": [token_id], "tokens": [content]}
                for content, token_id in self.special_tokens.items()
            },
        }


class Sequence(PostProcessor):
    """Apply several post-processors in order."""

    def __init__(self, processors: list[PostProcessor]) -> None:
        self.processors = list(processors)

    def process(
        self,
        encoding: Encoding,
        pair: Optional[Encoding] = None,
        add_special_tokens: bool = True,
    ) -> Encoding:
        for processor in self.processors:
            encoding = processor.process(encoding, pair, add_special_tokens)
            pair = None
        return encoding

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Sequence":
        from ubi_tokenizer.processors import build_post_processor

        built = [build_post_processor(item) for item in data.get("processors", [])]
        return cls([p for p in built if p is not None])

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "Sequence",
            "processors": [p.to_dict() for p in self.processors],
        }


# -- helpers ---------------------------------------------------------------
def _as_pieces(template: str | list[str]) -> list[str]:
    if isinstance(template, list):
        return list(template)
    return template.split()


def _type_of(piece: str, default: int) -> int:
    if ":" in piece:
        try:
            return int(piece.split(":", 1)[1])
        except ValueError:
            return default
    return default


def _extend(result: Encoding, src: Encoding, type_id: int) -> None:
    result.ids.extend(src.ids)
    result.tokens.extend(src.tokens)
    result.type_ids.extend([type_id] * len(src.ids))
    result.offsets.extend(src.offsets)
    result.attention_mask.extend(src.attention_mask)
    result.special_tokens_mask.extend(src.special_tokens_mask)
    result.word_ids.extend(src.word_ids)


def _concat(a: Encoding, b: Encoding) -> Encoding:
    result = Encoding()
    _extend(result, a, type_id=0)
    _extend(result, b, type_id=1)
    return result


def _template_from_json(template: Any) -> Optional[list[str]]:
    if template is None:
        return None
    pieces: list[str] = []
    for item in template:
        if "SpecialToken" in item:
            pieces.append(item["SpecialToken"]["id"])
        elif "Sequence" in item:
            seq = item["Sequence"]
            marker = "$A" if seq["id"] == "A" else "$B"
            type_id = seq.get("type_id", 0)
            pieces.append(f"{marker}:{type_id}" if type_id else marker)
    return pieces


def _template_to_json(template: Optional[list[str]]) -> list[dict[str, Any]]:
    if template is None:
        return []
    result: list[dict[str, Any]] = []
    for piece in template:
        if piece.startswith("$A") or piece.startswith("$B"):
            seq_id = "A" if piece.startswith("$A") else "B"
            type_id = _type_of(piece, 0)
            result.append({"Sequence": {"id": seq_id, "type_id": type_id}})
        else:
            result.append({"SpecialToken": {"id": piece, "type_id": 0}})
    return result
