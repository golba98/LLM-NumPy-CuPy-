"""Namespace identity, weights-only compatibility and relocated shard paths."""
import importlib
import json
from pathlib import Path
import numpy as np
import pytest
from llm_numpy.assets import resolve_input_path, resolve_output_path
from llm_numpy.config import LLMConfig
from llm_numpy.nn.model import TinyLLM
from llm_numpy.optim.adamw import AdamW
from llm_numpy.training.checkpoint import save_checkpoint, load_model_weights
from llm_numpy.training.reference import generate_reference
from llm_numpy.data.pipeline import open_token_shard
from llm_numpy.utils.seed import set_seed


def test_legacy_imports_share_tensor_and_module_objects():
    for suffix in ("tensor", "parameter", "nn.model", "training.checkpoint", "backend"):
        legacy = importlib.import_module("src." + suffix)
        canonical = importlib.import_module("llm_numpy." + suffix)
        if suffix == "backend":
            assert legacy.get_backend is canonical.get_backend
        else:
            assert legacy is canonical


def test_weights_only_loader_preserves_indexed_checkpoint_and_checks_shapes(tmp_path):
    config = LLMConfig(vocab_size=32, max_seq_len=8, dim=8, num_layers=1, num_heads=2, hidden_dim=16)
    set_seed(42)
    original = TinyLLM(config)
    inputs = np.array([[1, 2, 3, 4]])
    optimizer = AdamW(original.parameters(), lr=.001)
    _, loss = original(inputs, targets=inputs)
    loss.backward()
    optimizer.step()
    path = tmp_path / "checkpoint.npz"
    save_checkpoint(str(path), original, optimizer, 1, 0, 4, config)
    restored = TinyLLM(config)
    load_model_weights(path, restored, expected_model_config=config)
    np.testing.assert_array_equal(original(inputs).data, restored(inputs).data)
    wrong = TinyLLM(LLMConfig(vocab_size=33, max_seq_len=8, dim=8, num_layers=1, num_heads=2, hidden_dim=16))
    with pytest.raises(ValueError, match="configuration"):
        load_model_weights(path, wrong, expected_model_config=wrong.config)


def test_shard_root_does_not_depend_on_working_directory(tmp_path):
    values = np.array([1, 2, 3, 4], dtype=np.uint32)
    values.tofile(tmp_path / "train.bin")
    manifest = {"shards": {"train": {"path": "train.bin", "tokens": 4}}}
    np.testing.assert_array_equal(open_token_shard(manifest, "train", root=tmp_path), values)


def test_protected_storage_rejects_output_and_preserves_old_input_alias(tmp_path, monkeypatch):
    root = tmp_path / "workspace"
    root.mkdir()
    protected = tmp_path / "protected"
    protected.mkdir()
    (root / "artifacts.local.json").write_text(json.dumps({"asset_root": str(protected), "input_aliases": {"/retired/numpy": str(protected)}}))
    monkeypatch.setenv("NUMPY_LLM_WORKSPACE_ROOT", str(root))
    monkeypatch.delenv("NUMPY_LLM_ASSET_ROOT", raising=False)
    monkeypatch.delenv("NUMPY_LLM_OUTPUT_ROOT", raising=False)
    assert resolve_input_path("/retired/numpy/data/train.bin") == protected / "data/train.bin"
    with pytest.raises(ValueError):
        resolve_output_path(protected / "run")


def test_preserved_optimizer_reference_matches_regeneration(tmp_path):
    generated = generate_reference(tmp_path / "generated")
    reference = Path(__file__).resolve().parents[1] / "reference/reference_optimizer_step.npz"
    with np.load(generated / "reference_optimizer_step.npz") as fresh, np.load(reference) as saved:
        for key in fresh.files:
            np.testing.assert_allclose(fresh[key], saved[key], atol=1e-12, rtol=1e-12)


def test_historical_unicode_tokenizer_samples_still_roundtrip():
    from llm_numpy.tokenization.bpe import ByteLevelBPETokenizer
    tokenizer = ByteLevelBPETokenizer()
    sample = "Hello World! Cafe\n\t  123 456  世界 ÄÖÜ"
    assert tokenizer.decode(tokenizer.encode(sample)) == sample
