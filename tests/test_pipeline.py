import json
import numpy as np

from src.data.pipeline import RawDocument, build_token_cache, normalize_text, open_token_shard, prepare_documents
from src.tokenization.bpe import ByteLevelBPETokenizer


def test_prepare_documents_normalizes_deduplicates_and_splits_deterministically():
    docs = [
        RawDocument("books", "A\r\n\r\n  useful   document " * 10),
        RawDocument("duplicate", "A\n\n useful document " * 10),
        RawDocument("technical", "Different material. " * 20),
    ]
    first = prepare_documents(docs, validation_fraction=0.5, min_characters=20)
    second = prepare_documents(docs, validation_fraction=0.5, min_characters=20)
    assert normalize_text("a\r\n\r\n b") == "a\n\nb"
    assert [(item.sha256, item.split) for item in first] == [(item.sha256, item.split) for item in second]
    assert len({item.sha256 for item in first}) == 2
    assert {item.split for item in first} == {"train", "validation"}


def test_prepare_documents_chunks_long_documents_for_representative_validation():
    documents = prepare_documents(
        [RawDocument("books", ("A paragraph with useful language. " * 20 + "\n\n") * 8)],
        validation_fraction=0.25,
        min_characters=20,
        chunk_characters=200,
    )
    assert len(documents) >= 4
    assert {item.split for item in documents} == {"train", "validation"}


def test_token_cache_is_persistent_and_memmapped(tmp_path):
    docs = prepare_documents(
        [RawDocument("general", "The cat sat on the mat. " * 30), RawDocument("books", "A long book sentence. " * 30)],
        validation_fraction=0.5,
        min_characters=20,
    )
    manifest = build_token_cache(docs, tmp_path / "cache", vocab_size=280)
    loaded = json.loads((tmp_path / "cache" / "manifest.json").read_text())
    assert manifest == loaded
    train = open_token_shard(loaded, "train")
    validation = open_token_shard(loaded, "validation")
    assert isinstance(train, np.memmap)
    assert train.dtype == np.uint32
    assert len(train) > 0 and len(validation) > 0


def test_byte_tokenizer_reports_lossless_unknown_frequency():
    tokenizer = ByteLevelBPETokenizer()
    tokenizer.train(["General language text with café and emoji 👋"], vocab_size=280)
    stats = tokenizer.analyze_corpus(["General language text with café and emoji 👋"])
    assert stats["unknown_token_count"] == 0
    assert stats["unknown_frequency"] == 0.0
