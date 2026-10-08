import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "src"))
import numpy as np
import pytest

from llm_numpy.data.sft import conversation_to_example
from llm_numpy.losses.masked_cross_entropy import masked_cross_entropy_loss
from llm_numpy.tokenization.bpe import ByteLevelBPETokenizer
from llm_numpy.tensor import Tensor


def test_masked_cross_entropy_ignores_user_positions_and_normalizes_selected_tokens():
    logits = Tensor(np.array([[3.0, 0.0], [0.0, 3.0], [2.0, 0.0]], dtype=np.float64), requires_grad=True)
    targets = np.array([0, 1, 1])
    mask = np.array([0.0, 1.0, 1.0])
    loss = masked_cross_entropy_loss(logits, targets, mask)
    expected = -0.5 * (np.log(np.exp(3.0) / (np.exp(3.0) + 1.0)) +
                       np.log(1.0 / (np.exp(2.0) + 1.0)))
    np.testing.assert_allclose(loss.data, expected)
    loss.backward()
    np.testing.assert_allclose(logits.grad[0], 0.0)
    assert not np.allclose(logits.grad[1], 0.0)
    assert not np.allclose(logits.grad[2], 0.0)


def test_masked_cross_entropy_requires_an_assistant_target():
    logits = Tensor(np.zeros((2, 3)), requires_grad=True)
    with pytest.raises(ValueError, match="at least one"):
        masked_cross_entropy_loss(logits, np.array([0, 1]), np.zeros(2))


def test_conversation_formatter_masks_user_and_trains_assistant_eos():
    tokenizer = ByteLevelBPETokenizer()
    tokenizer.train(["User: hi\nAssistant: hello\n"], vocab_size=300)
    example = conversation_to_example(
        [{"role": "user", "content": "hi"},
         {"role": "assistant", "content": "hello"}], tokenizer)
    assert example.input_ids.shape == example.target_ids.shape == example.loss_mask.shape
    assert np.any(example.loss_mask > 0)
    # The assistant response and its newline/EOS are supervised; user text is not.
    assert example.loss_mask[-1] == 1.0
    assert np.any(example.loss_mask == 0.0)
