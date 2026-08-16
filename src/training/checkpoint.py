import json
import os
import hashlib
import numpy as np
from src.backend import array_module, to_cpu
from dataclasses import asdict, is_dataclass
from typing import Optional, Tuple
from src.nn.module import Module
from src.optim.adamw import AdamW

FORMAT_VERSION = 2

def _json_hash(value) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
    return hashlib.sha256(encoded).hexdigest()


def _sha256_file(filepath: str) -> str:
    digest = hashlib.sha256()
    with open(filepath, "rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _jsonable_rng_state(state):
    return [state[0], state[1].tolist(), int(state[2]), int(state[3]), float(state[4])]


def save_checkpoint(filepath: str, model: Module, optimizer: AdamW, step: int,
                    epoch: int, tokens_processed: int, model_config=None,
                    training_config=None, tokenizer_path: Optional[str] = None,
                    best_val_loss: Optional[float] = None,
                    dataset_manifest_sha256: Optional[str] = None) -> None:
    parent = os.path.dirname(filepath)
    if parent:
        os.makedirs(parent, exist_ok=True)
    metadata = {
        "format_version": FORMAT_VERSION,
        "step": int(step), "epoch": int(epoch), "tokens_processed": int(tokens_processed),
        "model_config": asdict(model_config) if is_dataclass(model_config) else model_config,
        "training_config": asdict(training_config) if is_dataclass(training_config) else training_config,
        "tokenizer_path": tokenizer_path,
        "best_val_loss": best_val_loss,
        "dataset_manifest_sha256": dataset_manifest_sha256,
        "numpy_rng_state": _jsonable_rng_state(np.random.get_state()),
    }
    metadata["parameter_count"] = int(sum(parameter.data.size for parameter in model.parameters()))
    metadata["model_config_sha256"] = _json_hash(metadata["model_config"]) if metadata["model_config"] is not None else None
    metadata["training_config_sha256"] = _json_hash(metadata["training_config"]) if metadata["training_config"] is not None else None
    if tokenizer_path and os.path.exists(tokenizer_path):
        metadata["tokenizer_sha256"] = _sha256_file(tokenizer_path)
    arrays = {"metadata": np.array(json.dumps(metadata)), "optimizer_t": np.array(optimizer.t)}
    for index, parameter in enumerate(model.parameters()):
        arrays[f"model_p_{index}"] = to_cpu(parameter.data)
    for group_index, group in enumerate(optimizer.param_groups):
        for param_index, (moment_m, moment_v) in enumerate(zip(group["m"], group["v"])):
            arrays[f"opt_m_{group_index}_{param_index}"] = to_cpu(moment_m)
            arrays[f"opt_v_{group_index}_{param_index}"] = to_cpu(moment_v)
    # A Target checkpoint is several GiB.  Never expose a partially written
    # archive at the canonical path if the process is interrupted during the
    # write or filesystem flush.
    temporary_filepath = filepath + ".tmp.npz"
    temporary_sidecar = filepath + ".tmp.sha256"
    try:
        np.savez(temporary_filepath, **arrays)
        with open(temporary_filepath, "rb") as handle:
            handle.flush() if hasattr(handle, "flush") else None
            os.fsync(handle.fileno())
        os.replace(temporary_filepath, filepath)
    finally:
        if os.path.exists(temporary_filepath):
            os.unlink(temporary_filepath)
    digest = _sha256_file(filepath)
    try:
        with open(temporary_sidecar, "w", encoding="utf-8") as handle:
            handle.write(f"{digest}  {os.path.basename(filepath)}\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_sidecar, filepath + ".sha256")
    finally:
        if os.path.exists(temporary_sidecar):
            os.unlink(temporary_sidecar)


def load_checkpoint(filepath: str, model: Module, optimizer: AdamW, *,
                    expected_model_config=None, expected_tokenizer_path: Optional[str] = None,
                    expected_dataset_manifest_sha256: Optional[str] = None) -> Tuple[int, int, int]:
    sidecar = filepath + ".sha256"
    if os.path.exists(sidecar):
        with open(sidecar, encoding="utf-8") as handle:
            expected = handle.read().split()[0]
        actual = _sha256_file(filepath)
        if expected != actual:
            raise ValueError(f"checkpoint checksum mismatch: {filepath}")
    with np.load(filepath, allow_pickle=False) as data:
        if "metadata" in data:
            metadata = json.loads(str(data["metadata"].item()))
            if expected_model_config is not None:
                expected = asdict(expected_model_config) if is_dataclass(expected_model_config) else expected_model_config
                if metadata.get("model_config_sha256") and metadata["model_config_sha256"] != _json_hash(expected):
                    raise ValueError("checkpoint model configuration is incompatible")
            if expected_tokenizer_path and metadata.get("tokenizer_sha256"):
                if not os.path.exists(expected_tokenizer_path) or metadata["tokenizer_sha256"] != _sha256_file(expected_tokenizer_path):
                    raise ValueError("checkpoint tokenizer is incompatible")
            if expected_dataset_manifest_sha256 and metadata.get("dataset_manifest_sha256") != expected_dataset_manifest_sha256:
                raise ValueError("checkpoint dataset manifest is incompatible")
            step = int(metadata["step"])
            epoch = int(metadata["epoch"])
            tokens_processed = int(metadata["tokens_processed"])
            rng_state = metadata.get("numpy_rng_state")
            if rng_state:
                np.random.set_state((rng_state[0], np.asarray(rng_state[1], dtype=np.uint32),
                                     int(rng_state[2]), int(rng_state[3]), float(rng_state[4])))
        else:
            step, epoch, tokens_processed = int(data["step"]), int(data["epoch"]), int(data["tokens_processed"])
        optimizer.t = int(data["optimizer_t"])
        for index, parameter in enumerate(model.parameters()):
            xp = array_module(parameter.data)
            parameter.data[...] = xp.asarray(data[f"model_p_{index}"], dtype=parameter.data.dtype)
        for group_index, group in enumerate(optimizer.param_groups):
            for param_index, (moment_m, moment_v) in enumerate(zip(group["m"], group["v"])):
                xp = array_module(moment_m)
                moment_m[...] = xp.asarray(data[f"opt_m_{group_index}_{param_index}"], dtype=moment_m.dtype)
                moment_v[...] = xp.asarray(data[f"opt_v_{group_index}_{param_index}"], dtype=moment_v.dtype)
    return step, epoch, tokens_processed
