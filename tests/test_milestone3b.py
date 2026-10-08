import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "src"))
import numpy as np

from llm_numpy.data import DataLoader, LanguageModelDataset, TextCorpus
from llm_numpy.generation.sampler import top_k_sampling, top_p_sampling
from llm_numpy.tokenization import ByteLevelBPETokenizer
from llm_numpy.training.metrics import token_accuracy
from llm_numpy.training.schedules import learning_rate


def test_byte_bpe_is_deterministic_and_lossless(tmp_path):
    text = "Hello 👋 世界 café\n\t 123"
    first = ByteLevelBPETokenizer(); second = ByteLevelBPETokenizer()
    first.train([text, text], vocab_size=300); second.train([text, text], vocab_size=300)
    assert first.merges == second.merges
    assert first.encode(text) == second.encode(text)
    assert first.decode(first.encode(text)) == text
    path = tmp_path / "tokenizer.json"
    first.save(path)
    loaded = ByteLevelBPETokenizer().load(path)
    assert loaded.encode(text) == first.encode(text)


def test_dataset_uses_every_complete_window():
    dataset = LanguageModelDataset([10, 20, 30, 40, 50, 60], context_length=3, stride=1)
    assert len(dataset) == 3
    np.testing.assert_array_equal(dataset[0][0], [10, 20, 30])
    np.testing.assert_array_equal(dataset[-1][1], [40, 50, 60])


def test_dataloader_epoch_order_is_controlled():
    dataset = LanguageModelDataset(range(80), context_length=4, stride=4)
    loader = DataLoader(dataset, batch_size=2, seed=12)
    first = list(loader); second = list(loader)
    assert any(not np.array_equal(a[0], b[0]) for a, b in zip(first, second))
    loader.set_epoch(0)
    again = list(loader)
    for original, repeated in zip(first, again):
        np.testing.assert_array_equal(original[0], repeated[0])


def test_corpus_preserves_document_boundaries():
    tokenizer = ByteLevelBPETokenizer()
    corpus = TextCorpus(["one", "two"])
    assert corpus.token_ids(tokenizer, tokenizer.EOS_ID)[3] == tokenizer.EOS_ID


def test_sampling_validation_and_seeded_behavior():
    logits = np.array([4.0, 3.0, 0.0, -1.0])
    assert top_k_sampling(logits, k=1) == 0
    assert top_p_sampling(logits, p=1.0, rng=np.random.default_rng(1)) in range(4)
    np.testing.assert_raises(ValueError, top_k_sampling, logits, 0)
    np.testing.assert_raises(ValueError, top_p_sampling, logits, 0.0)


def test_schedule_has_warmup_and_cosine_endpoints():
    assert learning_rate(0, 1.0, warmup_steps=4) == 0.0
    assert learning_rate(4, 1.0, warmup_steps=4, decay_steps=8) == 1.0
    assert learning_rate(12, 1.0, warmup_steps=4, decay_steps=8) == 0.0
