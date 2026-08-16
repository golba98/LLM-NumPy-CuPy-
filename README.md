# Codexa NumPy LLM

Codexa is a from-scratch decoder-only language-model framework. It owns the
Tensor/autograd implementation, Transformer, attention, RMSNorm, SwiGLU,
cross-entropy, AdamW, tokenizer, dataset pipeline, checkpointing, generation,
and training loop.

The numerical backend is selectable:

- NumPy on CPU
- CuPy/CUDA on NVIDIA GPUs

CuPy provides CUDA array operations only. It does not provide Codexa's model,
autograd, optimizer, tokenizer, or training loop.

## Current verified model

The CUDA implementation has completed numerical parity and the full lifecycle
on the 240.9M benchmark target. The production vocabulary increases the tied
embedding model to 253,284,096 parameters. The known-good benchmark is FP16
with FP32 AdamW moments, batch 4, context 128, and approximately 4,001
training tokens/second on an RTX 4080.

## Install and verify

```powershell
python -m pip install -r requirements.txt
python -m pip install -r requirements-cuda.txt
python scripts/environment_report.py
python scripts/cuda_self_test.py
python -m pytest -q tests
```

Windows-specific setup is documented in [docs/WINDOWS_CUDA_SETUP.md](docs/WINDOWS_CUDA_SETUP.md).

## Before pretraining

Run the bounded preflight first:

```powershell
python scripts/preflight_training.py --config configs/pretrain_250m.json
```

The checked-in local token cache is historical and currently blocked from
production use pending source-provenance cleanup. Do not bypass that check for
real training. `scripts/production_dry_run.py` supports at most three steps and
labels its artifacts as a dry run.

## Layout

```text
src/          model, autograd, backends, data, training, generation
tests/        CPU and conditional CUDA regression tests
benchmarks/   persisted-performance benchmark tools
configs/      explicit training configurations
scripts/      environment, preflight, checkpoint, and dry-run tools
docs/         backend, Windows, pretraining, and evaluation documentation
data/         local datasets, tokenizer caches, manifests, and prompts
runs/         local experiments and benchmark evidence; not source control
```

Shakespeare is smoke/regression data only. General-language pretraining and
later instruction SFT are separate phases.

## Large local artifacts

Training datasets, tokenized shards, logs, checkpoints, and generated model
artifacts are intentionally excluded from Git. GitHub is used for the source,
tests, configurations, documentation, and small reproducibility manifests.
Keep large model files on local or external artifact storage and record their
relative path, training step, configuration, and SHA-256 checksum in a small
text or JSON manifest when an artifact needs to be referenced.

The working tree may therefore be used with an empty `data/` and `runs/`
directory. Run the smoke examples and tests that do not require a local
dataset or checkpoint; restore the corresponding local artifacts before
resuming training or evaluating a saved model.

# LLM-NumPy-CuPy-
