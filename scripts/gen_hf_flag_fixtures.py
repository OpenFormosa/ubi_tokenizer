#!/usr/bin/env python3
"""Regenerate the HuggingFace ground-truth fixtures for BPE option behaviour.

``tests/test_bpe_flags.py`` checks ``ubi_tokenizer``'s BPE against recorded
HuggingFace output. This script produces that recording, so the expectations are
measured rather than derived from reading the Rust source.

It is the only part of the test setup that needs a third-party package, and it is
run by hand -- the test suite itself stays standard-library only, which is the
point of the whole project. Install the dev extra first:

    pip install -e ".[dev]"
    python scripts/gen_hf_flag_fixtures.py

The ``tokenizers`` version is pinned in the dev extra and recorded in the output.
Several behaviours captured here are quirky enough (see ``bf_partial_bytes``)
that regenerating against a different version could silently move ground truth.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

try:
    import tokenizers
    from tokenizers import models
except ImportError:  # pragma: no cover - developer-facing script
    raise SystemExit(
        'huggingface tokenizers is required to regenerate fixtures: pip install -e ".[dev]"'
    )

OUTPUT = Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "hf_bpe_flags.json"

BYTE_TOKENS = [f"<0x{value:02X}>" for value in range(256)]


def vocab_of(*groups: list[str]) -> dict[str, int]:
    """Assign ids in listed order, skipping duplicates."""
    vocab: dict[str, int] = {}
    for group in groups:
        for token in group:
            if token not in vocab:
                vocab[token] = len(vocab)
    return vocab


def bytes_except(*excluded: str) -> list[str]:
    return [token for token in BYTE_TOKENS if token not in excluded]


# Inputs reused across the byte-fallback cases. "\t" (0x09), "\n" (0x0A) and
# "\x01" exercise bytes below 0x10, where an implementation that produced
# uppercase-but-not-zero-padded hex would otherwise pass unnoticed -- every
# CJK/accented character has bytes >= 0x80 and so is always two hex digits.
LOW_BYTE_INPUTS = ["\t", "\n", "\x01", "\x0f"]

CASES: list[dict[str, Any]] = [
    {
        "name": "baseline_no_flags",
        "why": "merge loop must not regress",
        "vocab": vocab_of(["a", "b", "c", "ab", "abc"]),
        "merges": [("a", "b"), ("ab", "c")],
        "flags": {},
        "inputs": ["abc", "ab", "a", "abz", "zz", "az", ""],
    },
    {
        "name": "ignore_merges_off_control",
        "why": "control for the case below; proves the flag is what changes output",
        "vocab": vocab_of(["x", "y", "xy", "z", "xz"]),
        "merges": [("x", "z")],
        "flags": {},
        "inputs": ["xy", "xz", "xyz", "xyxy", ""],
    },
    {
        "name": "ignore_merges_discriminating",
        "why": "'xy' is in vocab but no merge reaches it, so the flag alone decides",
        "vocab": vocab_of(["x", "y", "xy", "z", "xz"]),
        "merges": [("x", "z")],
        "flags": {"ignore_merges": True},
        "inputs": ["xy", "xz", "xyz", "xyxy", ""],
    },
    {
        "name": "drop_when_no_unk",
        "why": "out-of-vocab characters are dropped, not turned into an invalid id",
        "vocab": vocab_of(["a", "b"]),
        "merges": [],
        "flags": {},
        "inputs": ["zz", "azb", "az", "za", "中"],
    },
    {
        "name": "unk_only",
        "why": "one unk per out-of-vocab character",
        "vocab": vocab_of(["a", "b", "ab", "<unk>"]),
        "merges": [("a", "b")],
        "flags": {"unk_token": "<unk>"},
        "inputs": ["ab", "az", "zz", "az z", "中", "<unk>"],
    },
    {
        "name": "unk_fused",
        "why": "consecutive pending unks coalesce",
        "vocab": vocab_of(["a", "b", "ab", "<unk>"]),
        "merges": [("a", "b")],
        "flags": {"unk_token": "<unk>", "fuse_unk": True},
        "inputs": ["ab", "az", "zz", "az z", "中", "azza"],
    },
    {
        "name": "unk_token_present_in_vocab",
        "why": "an in-vocab unk string must not swallow a following genuine unk",
        "vocab": vocab_of(["a", "U"]),
        "merges": [],
        "flags": {"unk_token": "U", "fuse_unk": True},
        "inputs": ["Uz", "zU", "aUza", "UU"],
    },
    {
        "name": "byte_fallback_with_unk",
        "why": "byte expansion, including bytes below 0x10",
        "vocab": vocab_of(["a", "<unk>"], BYTE_TOKENS),
        "merges": [],
        "flags": {"unk_token": "<unk>", "byte_fallback": True},
        "inputs": ["a", "az", "中", "é", "abc", "€", *LOW_BYTE_INPUTS],
    },
    {
        "name": "byte_fallback_without_unk",
        "why": "no unk to fall back on; misses are dropped",
        "vocab": vocab_of(["a"], bytes_except("<0xE4>", "<0xB8>", "<0xAD>")),
        "merges": [],
        "flags": {"byte_fallback": True},
        "inputs": ["a", "az", "中", "é", "a中a", *LOW_BYTE_INPUTS],
    },
    {
        "name": "bare_ascii_absent_but_byte_token_present",
        "why": "byte fallback also fires for plain ASCII when the bare char is missing",
        "vocab": vocab_of(["<unk>"], ["<0x61>", "<0x62>", "<0x63>"]),
        "merges": [],
        "flags": {"unk_token": "<unk>", "byte_fallback": True},
        "inputs": ["abc", "a", "abd"],
    },
    {
        "name": "bf_partial_bytes",
        "why": "all-or-nothing per character, and the pending-unk slot is NOT flushed by a byte-fallback hit",
        "vocab": vocab_of(["a", "<unk>"], bytes_except("<0xE4>")),
        "merges": [],
        "flags": {"unk_token": "<unk>", "byte_fallback": True},
        "inputs": ["中", "中é", "é中", "中中é", "中é中", "中中", "aé中", "中a"],
    },
    {
        "name": "bf_partial_bytes_fused",
        "why": "same slot behaviour with coalescing on top",
        "vocab": vocab_of(["a", "<unk>"], bytes_except("<0xE4>")),
        "merges": [],
        "flags": {"unk_token": "<unk>", "byte_fallback": True, "fuse_unk": True},
        "inputs": ["中", "中é", "é中", "中中é", "中é中", "中中", "aé中", "中a"],
    },
    {
        "name": "merge_over_byte_tokens",
        "why": "byte-fallback tokens participate in merges, so resolution precedes merging",
        "vocab": vocab_of(
            ["a", "<unk>", "<0xE4>", "<0xB8>", "<0xAD>", "<0xE4><0xB8>", "a<0xE4>"]
        ),
        "merges": [("<0xE4>", "<0xB8>")],
        "flags": {"unk_token": "<unk>", "byte_fallback": True},
        "inputs": ["中", "a中"],
    },
    {
        "name": "merge_over_byte_tokens_rank_order",
        "why": "merge rank over byte tokens changes the result",
        "vocab": vocab_of(
            ["a", "<unk>", "<0xE4>", "<0xB8>", "<0xAD>", "<0xE4><0xB8>", "a<0xE4>"]
        ),
        "merges": [("a", "<0xE4>"), ("<0xE4>", "<0xB8>")],
        "flags": {"unk_token": "<unk>", "byte_fallback": True},
        "inputs": ["中", "a中"],
    },
    {
        "name": "merge_over_unk_tokens",
        "why": "generated unk tokens participate in merges too",
        "vocab": vocab_of(["a", "<unk>", "<unk><unk>"]),
        "merges": [("<unk>", "<unk>")],
        "flags": {"unk_token": "<unk>"},
        "inputs": ["zz", "azza", "z", "zzz"],
    },
    {
        "name": "fuse_unk_without_unk_token",
        "why": "fuse_unk is inert with no unk token to fuse",
        "vocab": vocab_of(["a", "b"]),
        "merges": [],
        "flags": {"fuse_unk": True},
        "inputs": ["azza", "zz", "ab"],
    },
    {
        "name": "ignore_merges_falls_through_to_unk",
        "why": "a piece absent from vocab still takes the normal resolve path",
        "vocab": vocab_of(["a", "b", "ab", "<unk>"]),
        "merges": [("a", "b")],
        "flags": {"ignore_merges": True, "unk_token": "<unk>"},
        "inputs": ["ab", "abz", "zz", "a"],
    },
]


def build(case: dict[str, Any]) -> dict[str, Any]:
    model = models.BPE(
        vocab=dict(case["vocab"]),
        merges=[tuple(pair) for pair in case["merges"]],
        **case["flags"],
    )
    expected = {}
    for text in case["inputs"]:
        expected[text] = [token.value for token in model.tokenize(text)]
    return {
        "name": case["name"],
        "why": case["why"],
        "vocab": case["vocab"],
        "merges": [list(pair) for pair in case["merges"]],
        "flags": case["flags"],
        "expected": expected,
    }


def main() -> None:
    payload = {
        "_generated_by": "scripts/gen_hf_flag_fixtures.py",
        "_ground_truth": f"huggingface tokenizers {tokenizers.__version__}",
        "_warning": (
            "Do not hand-edit. Regenerate with the pinned tokenizers version; some "
            "recorded behaviour is version-sensitive."
        ),
        "cases": [build(case) for case in CASES],
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    total = sum(len(case["expected"]) for case in payload["cases"])
    print(f"wrote {OUTPUT.relative_to(Path.cwd())}: {len(payload['cases'])} cases, {total} inputs")


if __name__ == "__main__":
    main()
