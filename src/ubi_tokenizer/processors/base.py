"""Post-processor base class."""

from __future__ import annotations

from typing import Any, Optional

from ubi_tokenizer.encoding import Encoding


class PostProcessor:
    """Abstract base for post-processors. Default is identity."""

    def process(
        self,
        encoding: Encoding,
        pair: Optional[Encoding] = None,
        add_special_tokens: bool = True,
    ) -> Encoding:
        """Return the (possibly modified) encoding. Default: identity."""
        return encoding

    def num_special_tokens_to_add(self, is_pair: bool = False) -> int:
        return 0

    def to_dict(self) -> dict[str, Any]:  # pragma: no cover - interface
        raise NotImplementedError
