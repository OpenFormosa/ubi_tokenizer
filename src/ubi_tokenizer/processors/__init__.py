"""Post-processors."""

from __future__ import annotations

from typing import Any, Optional

from ubi_tokenizer.processors.base import PostProcessor
from ubi_tokenizer.processors.template import Sequence, TemplateProcessing

__all__ = [
    "PostProcessor",
    "TemplateProcessing",
    "Sequence",
    "build_post_processor",
]


def build_post_processor(data: Optional[dict[str, Any]]) -> Optional[PostProcessor]:
    """Construct a post-processor from a tokenizer.json block."""
    if data is None:
        return None
    kind = data.get("type")
    if kind == "TemplateProcessing":
        return TemplateProcessing.from_dict(data)
    if kind == "Sequence":
        return Sequence.from_dict(data)
    raise ValueError(f"Unsupported post_processor type: {kind!r}")
