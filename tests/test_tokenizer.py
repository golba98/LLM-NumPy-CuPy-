import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "src"))
import pytest
import os
from llm_numpy.tokenization.bpe import ByteLevelBPETokenizer

def test_tokenizer_bpe_training_and_roundtrip():
    corpus = "The cat sat on the mat. The dog sat on the rug. The cat slept. The dog slept."
    tokenizer = ByteLevelBPETokenizer()
    tokenizer.train(corpus, vocab_size=280)

    assert len(tokenizer.vocab) == 280

    # Encoding & Decoding roundtrip
    encoded = tokenizer.encode(corpus)
    decoded = tokenizer.decode(encoded)
    assert decoded == corpus

def test_tokenizer_unicode_emoji_and_whitespace_preservation():
    tokenizer = ByteLevelBPETokenizer()
    sample_text = "Hello 👋 World! Cafe\n\t  123 456  世界"

    encoded = tokenizer.encode(sample_text)
    decoded = tokenizer.decode(encoded)
    assert decoded == sample_text

def test_tokenizer_save_load(tmp_path):
    corpus = "Machine learning with NumPy and pure Python."
    tokenizer_orig = ByteLevelBPETokenizer()
    tokenizer_orig.train(corpus, vocab_size=275)

    save_path = str(tmp_path / "bpe_tokenizer.json")
    tokenizer_orig.save(save_path)
    assert os.path.exists(save_path)

    tokenizer_loaded = ByteLevelBPETokenizer()
    tokenizer_loaded.load(save_path)

    text = "Machine learning NumPy!"
    assert tokenizer_orig.encode(text) == tokenizer_loaded.encode(text)

def test_tokenizer_corpus_analysis():
    corpus = "Testing tokenization analysis."
    tokenizer = ByteLevelBPETokenizer()
    stats = tokenizer.analyze_corpus(corpus)

    assert stats["character_count"] == len(corpus)
    assert stats["byte_count"] == len(corpus.encode('utf-8'))
    assert stats["token_count"] > 0
