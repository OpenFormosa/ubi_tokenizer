"""Tiny encode/decode CLI for ubi_tokenizer.

Examples::

    python -m ubi_tokenizer encode --tokenizer tokenizer.json "hello world"
    python -m ubi_tokenizer decode --tokenizer tokenizer.json 72 101 108
"""

from __future__ import annotations

import argparse
import sys

from ubi_tokenizer.tokenizer import Tokenizer


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="ubi-tokenizer")
    sub = parser.add_subparsers(dest="command", required=True)

    encode_parser = sub.add_parser("encode", help="Encode text into token ids")
    encode_parser.add_argument("--tokenizer", required=True, help="tokenizer.json path")
    encode_parser.add_argument("text", help="text to encode")

    decode_parser = sub.add_parser("decode", help="Decode token ids into text")
    decode_parser.add_argument("--tokenizer", required=True, help="tokenizer.json path")
    decode_parser.add_argument("ids", nargs="+", type=int, help="token ids to decode")

    args = parser.parse_args(argv)
    tokenizer = Tokenizer.from_file(args.tokenizer)

    if args.command == "encode":
        encoding = tokenizer.encode(args.text)
        print(" ".join(str(token_id) for token_id in encoding.ids))
        return 0
    if args.command == "decode":
        print(tokenizer.decode(args.ids))
        return 0
    parser.print_help()
    return 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
