"""BPE model merge-order behaviour."""

from ubi_tokenizer.models.bpe import BPE


def _byte_vocab(extra=None):
    from ubi_tokenizer.byte_level import BYTE_ENCODER

    vocab = {BYTE_ENCODER[value]: value for value in range(256)}
    next_id = 256
    for token in extra or []:
        if token not in vocab:
            vocab[token] = next_id
            next_id += 1
    return vocab


def test_single_symbol_piece_is_unchanged():
    model = BPE(vocab=_byte_vocab(), merges=[])
    assert model.tokenize("a") == ["a"]
    assert model.tokenize("") == []


def test_merges_applied_by_rank():
    # Merge "a"+"b" -> "ab", then "ab"+"c" -> "abc".
    vocab = _byte_vocab(["ab", "abc"])
    merges = [("a", "b"), ("ab", "c")]
    model = BPE(vocab=vocab, merges=merges)
    assert model.tokenize("abc") == ["abc"]


def test_lower_rank_merge_wins():
    # Two competing pairs; the earlier-listed merge has the lower rank and wins.
    vocab = _byte_vocab(["ab", "bc"])
    merges = [("a", "b"), ("b", "c")]
    model = BPE(vocab=vocab, merges=merges)
    # "abc": pair (a,b) rank 0 beats (b,c) rank 1, so we get "ab" then "c".
    assert model.tokenize("abc") == ["ab", "c"]


def test_reordered_merges_change_result():
    vocab = _byte_vocab(["ab", "bc"])
    merges = [("b", "c"), ("a", "b")]
    model = BPE(vocab=vocab, merges=merges)
    # Now (b,c) has lower rank, so we get "a" then "bc".
    assert model.tokenize("abc") == ["a", "bc"]


def test_repeated_pair_merges_left_to_right():
    vocab = _byte_vocab(["aa"])
    merges = [("a", "a")]
    model = BPE(vocab=vocab, merges=merges)
    assert model.tokenize("aaaa") == ["aa", "aa"]


def test_no_applicable_merge_keeps_symbols():
    model = BPE(vocab=_byte_vocab(), merges=[("x", "y")])
    assert model.tokenize("abc") == ["a", "b", "c"]
