# Repository storage and cleanup

The GitHub repository contains the NumPy/CuPy implementation, tests,
documentation, configuration, and small deterministic reference fixtures.
Generated datasets, logs, checkpoints, and experiment output remain local.

## Keep in Git

- `src/`, `tests/`, `examples/`, `scripts/`, and `benchmarks/`
- `configs/`, `docs/`, and project README files
- `reference/reference_config.json`
- `reference/reference_targets.npy`
- `reference/reference_tokens.npy`
- The three deterministic `.npz` fixtures under `reference/`

The reference fixtures are small and are required by
`tests/test_milestone3c.py`. Regenerate them with:

```bash
python3 tools/generate_reference.py
```

## Keep local only

The following paths are generated or potentially very large and are excluded
from GitHub:

- `runs/` — checkpoints, experiment output, and benchmark artifacts
- `data/` — local datasets and tokenized data
- `logs/` — training logs and CSV metrics
- Python caches such as `__pycache__/` and `.pytest_cache/`

`runs/README.md` and `data/README.md` remain as placeholders explaining the
local artifact policy.

## Before removing a model artifact

Check whether the artifact is needed for resume, evaluation, comparison, or
provenance. When retaining a checkpoint, also retain its configuration,
training metrics, and `.sha256` sidecar when available. Do not treat similarly
named checkpoints as interchangeable without verifying their lineage.

After cleanup, verify the repository with:

```bash
python3 -m pytest -q tests/
git diff --check
```
