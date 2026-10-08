# Codexa — independent NumPy/CuPy LLM

Codexa owns its Tensor/autograd engine, analytical derivatives, layers, RMSNorm, SwiGLU, RoPE attention, decoder-only Transformer, optimizers, byte-level BPE, training, checkpoints and sampling. NumPy supplies CPU arrays; optional CuPy supplies CUDA arrays. PyTorch is not a runtime dependency.

This project is a modular monorepo. Its Tensor/ops relationship and layer/model dependencies are best maintained within one package, rather than separate Git repositories.

```text
src/llm_numpy/  backend, Tensor, ops, nn, losses, tokenization, optim,
               data, training, generation, utilities and packaged CLIs
src/           temporary legacy import adapters
configs/       explicit run configurations
scripts/       compatibility CLI launchers
examples/      educational demonstrations
benchmarks/    original CPU/CUDA research tools
reference/     preserved small deterministic mathematical fixtures
tests/         CPU, gradient, checkpoint and conditional CUDA checks
docs/          model, data, backend and migration records
outputs/       ignored new experiments
```

## Install and validate

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.lock
.venv/bin/python -m pip install --no-deps -e .
.venv/bin/python -m pytest -q tests
# Optional CUDA dependencies are recorded in requirements-cuda.lock.
codexa-numpy-environment-report
codexa-numpy-train-general --help
codexa-numpy-chat-codexa --help
```

The local validated environment currently inherits existing system NumPy/CuPy libraries and installs this package locally. It is not a hermetic environment; the exact baseline pins and CUDA setup documentation describe reproduction. No mathematical engine or CUDA backend was replaced. Package imports and installed CLI commands work outside the checkout; pass NUMPY_LLM_WORKSPACE_ROOT when selecting another checkout.

## Historical inputs and outputs

Copy artifacts.local.example.json to ignored artifacts.local.json and configure asset_root, historical_roots and output_root. NUMPY_LLM_ASSET_ROOT overrides the current asset root. Historical input data, runs, checkpoints and logs have been relocated to protected temporary storage; they no longer require deprecated project directories. Conflicting historical manifests/tokenizers are retained separately with provenance.

New CLI outputs default beneath outputs/. Historical dataset manifests and checkpoint metadata remain unchanged; path resolution interprets their original relative references. Original data-provenance restrictions remain in force. Do not train a large model as a validation step.

The production tied-embedding target has 253,284,096 parameters and a 16K tokenizer. Existing 240.9M benchmark configurations remain historical records. See [docs/MIGRATION.md](docs/MIGRATION.md), [docs/VALIDATION.md](docs/VALIDATION.md), MODEL_CARD.md and DATASET_CARD.md for evidence and limitations.
