import numpy as np
import pytest

from src.backend import cuda_available, to_cpu, to_device
from src.config import LLMConfig
from src.nn.model import TinyLLM
from src.optim.adamw import AdamW
from src.training.checkpoint import load_checkpoint, save_checkpoint


pytestmark = pytest.mark.skipif(not cuda_available(), reason="CuPy/CUDA unavailable")


def _model_pair():
    config = LLMConfig(vocab_size=32, max_seq_len=8, dim=16, num_layers=1,
                       num_heads=4, hidden_dim=32)
    cpu = TinyLLM(config).to("cpu", dtype=np.float32)
    gpu = TinyLLM(config).to("cuda", dtype=np.float32)
    for left, right in zip(cpu.parameters(), gpu.parameters()):
        right.data[...] = to_device(left.data, "cuda")
    return cpu, gpu


def test_cpu_cuda_forward_loss_and_gradients_match():
    cpu, gpu = _model_pair()
    inputs = np.array([[1, 2, 3, 4, 5, 6]], dtype=np.int64)
    targets = np.array([[2, 3, 4, 5, 6, 7]], dtype=np.int64)

    cpu_logits, cpu_loss = cpu(inputs, targets=targets)
    gpu_logits, gpu_loss = gpu(inputs, targets=targets)
    cpu_loss.backward()
    gpu_loss.backward()

    np.testing.assert_allclose(cpu_logits.data, to_cpu(gpu_logits.data), rtol=2e-4, atol=2e-5)
    np.testing.assert_allclose(cpu_loss.data, to_cpu(gpu_loss.data), rtol=2e-5, atol=2e-5)
    for cpu_param, gpu_param in zip(cpu.parameters(), gpu.parameters()):
        np.testing.assert_allclose(cpu_param.grad, to_cpu(gpu_param.grad), rtol=4e-4, atol=4e-5)


def test_cuda_adamw_checkpoint_round_trip_to_cpu(tmp_path):
    cpu, gpu = _model_pair()
    cpu_optimizer = AdamW(cpu.parameters(), lr=1e-3, weight_decay=0.0)
    gpu_optimizer = AdamW(gpu.parameters(), lr=1e-3, weight_decay=0.0).to("cuda", dtype=np.float32)
    inputs = np.array([[1, 2, 3, 4, 5, 6]], dtype=np.int64)
    targets = np.array([[2, 3, 4, 5, 6, 7]], dtype=np.int64)
    _, gpu_loss = gpu(inputs, targets=targets)
    gpu_loss.backward()
    gpu_optimizer.step()
    checkpoint = tmp_path / "cuda.npz"
    save_checkpoint(str(checkpoint), gpu, gpu_optimizer, 1, 0, 6, cpu.config)

    restored = TinyLLM(cpu.config).to("cpu", dtype=np.float32)
    restored_optimizer = AdamW(restored.parameters(), lr=1e-3, weight_decay=0.0)
    step, epoch, tokens = load_checkpoint(str(checkpoint), restored, restored_optimizer)
    assert (step, epoch, tokens) == (1, 0, 6)
    for source, target in zip(gpu.parameters(), restored.parameters()):
        np.testing.assert_allclose(to_cpu(source.data), target.data, rtol=1e-6, atol=1e-6)
