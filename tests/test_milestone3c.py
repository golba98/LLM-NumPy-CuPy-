import numpy as np

from src.config import LLMConfig
from src.data import TextCorpus
from src.nn.model import TinyLLM
from src.optim.adamw import AdamW
from src.optim.clip import clip_grad_norm_
from src.training.profiler import StepProfiler
from src.training.reference import generate_reference
from src.training.trainer import Trainer
from src.utils.seed import set_seed


def _model():
    set_seed(42)
    return TinyLLM(LLMConfig(vocab_size=16, max_seq_len=8, dim=8, num_layers=1, num_heads=2, hidden_dim=16))


def test_accumulated_gradients_and_optimizer_match_large_batch():
    inputs = np.array([[1, 2, 3, 4], [4, 5, 6, 7], [7, 8, 9, 10], [10, 11, 12, 13]])
    targets = np.roll(inputs, -1, axis=1)
    large, accumulated = _model(), _model()
    opt_large, opt_acc = AdamW(large.parameters(), lr=.001, weight_decay=0), AdamW(accumulated.parameters(), lr=.001, weight_decay=0)
    opt_large.zero_grad(); _, loss = large(inputs, targets=targets); loss.backward()
    opt_acc.zero_grad()
    for start in (0, 2):
        _, micro_loss = accumulated(inputs[start:start+2], targets[start:start+2])
        (micro_loss / 2).backward()
    for first, second in zip(large.parameters(), accumulated.parameters()):
        np.testing.assert_allclose(first.grad, second.grad, atol=1e-12, rtol=1e-10)
    clip_grad_norm_(large.parameters(), 1.0); clip_grad_norm_(accumulated.parameters(), 1.0)
    opt_large.step(); opt_acc.step()
    for first, second in zip(large.parameters(), accumulated.parameters()):
        np.testing.assert_allclose(first.data, second.data, atol=1e-12, rtol=1e-10)


def test_single_document_split_precedes_windowing():
    train, validation = TextCorpus.from_text("abcdefghij").split(.6)
    assert train.documents == ["abcdef"]
    assert validation.documents == ["ghij"]
    assert not set(train.documents[0]) & set(validation.documents[0])


def test_profiler_records_nonnegative_stages_without_changing_behavior():
    profiler = StepProfiler()
    with profiler.stage("forward"):
        value = sum(range(10))
    assert value == 45
    assert profiler.summary()["forward"]["mean_ms"] >= 0
    disabled = StepProfiler(enabled=False)
    with disabled.stage("forward"):
        pass
    assert disabled.summary() == {}


def test_reference_fixture_generation_is_repeatable(tmp_path):
    first = generate_reference(tmp_path / "one")
    second = generate_reference(tmp_path / "two")
    with np.load(first / "reference_forward.npz") as left, np.load(second / "reference_forward.npz") as right:
        for key in left.files:
            np.testing.assert_array_equal(left[key], right[key])


def test_checked_in_reference_fixture_matches_regeneration(tmp_path):
    generated = generate_reference(tmp_path / "generated")
    fixture_root = __import__("pathlib").Path(__file__).resolve().parents[1] / "reference"
    with np.load(generated / "reference_forward.npz") as fresh, np.load(fixture_root / "reference_forward.npz") as stored:
        for key in fresh.files:
            np.testing.assert_allclose(fresh[key], stored[key], atol=1e-12, rtol=1e-12)
    with np.load(generated / "reference_gradients.npz") as fresh, np.load(fixture_root / "reference_gradients.npz") as stored:
        for key in fresh.files:
            np.testing.assert_allclose(fresh[key], stored[key], atol=1e-12, rtol=1e-12)
