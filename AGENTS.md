# NumPy/CuPy repository guidelines

The canonical implementation is `src/llm_numpy`. Keep Tensor/autograd and explicit derivatives in tensor.py and ops/, layers/attention/RoPE/Transformer in nn/, and backends, losses, tokenization, optimizers, data, training, generation and utilities in their existing named subpackages. This is one modular monorepo. Do not replace the numerical engine with PyTorch or add deep-learning framework dependencies. Generic src imports and old executable paths are compatibility adapters; package internals use llm_numpy imports.

Use `.venv/bin/python -m pytest -q tests` and preserve reference fixtures. All tests isolate generated outputs using temporary directories. Numerical changes require forward/gradient checks and CPU/CUDA comparisons; preserve parameter traversal, weight tying, checkpoint formats, tokenizer IDs and RNG state. CLI implementations live in llm_numpy.cli, model presets in llm_numpy.model_presets, and reusable logic belongs outside examples.

Historical inputs resolve through NUMPY_LLM_ASSET_ROOT or ignored artifacts.local.json. New outputs default beneath outputs/. Keep environments, datasets, checkpoints, private logs and recovery snapshots out of Git. Preserve dataset provenance statuses rather than clearing blocked data implicitly. No expensive training during migrations.

Use focused feature branches and commits; review staged diffs and run tests before PRs. Do not force-push, merge PRs, modify permissions/protection or remove pending-retirement directories until preservation and review gates pass.
