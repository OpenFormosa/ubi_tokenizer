"""In-memory BpeTrainer behaviour."""

from ubi_tokenizer.decoders.byte_level import ByteLevel as ByteLevelDecoder
from ubi_tokenizer.models.bpe import BPE
from ubi_tokenizer.pre_tokenizers.byte_level import ByteLevel as ByteLevelPreTokenizer
from ubi_tokenizer.tokenizer import Tokenizer
from ubi_tokenizer.trainers.bpe import BpeTrainer

CORPUS = [
    "the quick brown fox jumps over the lazy dog",
    "the lazy dog sleeps while the quick fox runs",
    "quick quick quick brown brown fox fox fox",
    "the the the the dog dog dog cat cat cat",
    "hello world hello world hello tokenizer world",
    "byte level bpe byte level bpe byte level",
    "training tiny tokenizers is fun and quick",
    "the fox the fox the fox the dog the dog",
    "aaa bbb ccc aaa bbb ccc aaa bbb ccc",
    "merge merge merge pair pair pair token token",
    "lorem ipsum dolor sit amet lorem ipsum",
    "data data data model model model train train",
    "alpha beta gamma alpha beta gamma alpha beta",
    "one two three one two three one two three",
    "repeat repeat repeat repeat repeat repeat now",
    "pure python tokenizer pure python tokenizer",
    "ubiquitous ubiquitous tokens tokens runtime",
    "encode decode encode decode encode decode now",
    "small corpus small corpus small corpus test",
    "the end the end the end the end the end here",
]


def _build_tokenizer():
    model = BPE(vocab={}, merges=[])
    tokenizer = Tokenizer(model)
    tokenizer.pre_tokenizer = ByteLevelPreTokenizer(add_prefix_space=False)
    tokenizer.decoder = ByteLevelDecoder()
    return tokenizer


def test_train_respects_vocab_size_cap():
    tokenizer = _build_tokenizer()
    trainer = BpeTrainer(vocab_size=300, min_frequency=2, special_tokens=["<|unk|>"])
    tokenizer.train_from_iterator(CORPUS, trainer)
    assert tokenizer.get_vocab_size() <= 300
    # The 256 byte alphabet + the single special token are always present.
    assert tokenizer.get_vocab_size() >= 257


def test_train_seeds_byte_alphabet_and_specials():
    tokenizer = _build_tokenizer()
    trainer = BpeTrainer(vocab_size=320, min_frequency=2, special_tokens=["<|unk|>", "<|pad|>"])
    tokenizer.train_from_iterator(CORPUS, trainer)
    vocab = tokenizer.get_vocab()
    # All 256 byte-mapped tokens present.
    from ubi_tokenizer.byte_level import BYTE_ENCODER

    for value in range(256):
        assert BYTE_ENCODER[value] in vocab
    assert "<|unk|>" in vocab
    assert "<|pad|>" in vocab


def test_min_frequency_respected():
    tokenizer = _build_tokenizer()
    # With a very high min_frequency no merges should be learned.
    trainer = BpeTrainer(vocab_size=1000, min_frequency=10_000, special_tokens=["<|unk|>"])
    tokenizer.train_from_iterator(CORPUS, trainer)
    # 256 bytes + 1 special and no learned merges.
    assert tokenizer.get_vocab_size() == 257
    assert tokenizer.model.merges == []


def test_trained_tokenizer_roundtrips():
    tokenizer = _build_tokenizer()
    trainer = BpeTrainer(vocab_size=400, min_frequency=2, special_tokens=["<|unk|>"])
    tokenizer.train_from_iterator(CORPUS, trainer)
    for text in ["the quick brown fox", "hello world", "aaa bbb ccc"]:
        enc = tokenizer.encode(text)
        assert tokenizer.decode(enc.ids) == text


def test_train_from_files(tmp_path):
    path = tmp_path / "corpus.txt"
    path.write_text("\n".join(CORPUS), encoding="utf-8")
    tokenizer = _build_tokenizer()
    trainer = BpeTrainer(vocab_size=350, min_frequency=2, special_tokens=["<|unk|>"])
    tokenizer.train([str(path)], trainer)
    assert tokenizer.get_vocab_size() <= 350
    enc = tokenizer.encode("the quick brown fox")
    assert tokenizer.decode(enc.ids) == "the quick brown fox"


def test_learned_merges_actually_compress():
    tokenizer = _build_tokenizer()
    trainer = BpeTrainer(vocab_size=400, min_frequency=2, special_tokens=["<|unk|>"])
    tokenizer.train_from_iterator(CORPUS, trainer)
    assert len(tokenizer.model.merges) > 0
    # A frequent word should encode to fewer tokens than its raw byte length.
    enc = tokenizer.encode("the")
    assert len(enc.ids) < len("the".encode("utf-8"))
