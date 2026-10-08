# Consolidation validation — 2026-10-08

Recorded on this machine after migration, not historical extraction results.

| Check | Result |
|---|---|
| PyTorch integration, generative environment | 121 passed, 5 skipped; 8 inherited return-value warnings |
| Specialist package | 39 passed |
| Specialist integration/memory selection | 58 passed |
| Architecture / Tokenizer / Data / Training / Inference / Memory | 5 / 6 / 15 / 10 / 1 / 5 passed |
| NumPy complete suite | 126 passed, including CPU/CuPy and migration checks |
| Component CLI help / installed NumPy CLI help | 33 / 15 passed |
| Built distributions | 7 components + integration + NumPy; no private assets in wheels |
| PyTorch environment dependency checks | Both profiles: no broken requirements |
| Syntax / shell / whitespace | 447 Python files parsed; shell syntax and Git diff checks passed |

The original 934,356,480-parameter PyTorch repair checkpoint loads strictly on CUDA. Two native conversation turns produce exactly the preserved implementation's greedy token sequences (bounded to eight new tokens each). KV-cache consistency and learned-position/RoPE compatibility are covered by regression tests. The full native base checkpoint reader verifies its sidecar and all expected model keys/shapes at recorded optimizer step 10,000; a complete large optimizer-resume run was not performed.

The original 253,284,096-parameter NumPy follow-up checkpoint loads on CuPy/CUDA and produces exactly the preserved implementation's four generated tokens. Fifty-three engine modules have identical ASTs after import namespace and blank-line docstring whitespace normalization; the only engine changes are explicit data roots and an additive inference-only checkpoint reader. All parameter shapes match. A deprecated demo checkpoint also loads and executes on CPU. These bounded comparisons do not prove equality of every possible training trajectory.

Each implementation completed a three-step tiny CUDA training smoke, with finite nonzero gradients, new outputs beneath its canonical outputs/ directory and checkpoint reload on CPU. The PyTorch run used a sample of the relocated corpus and the visible Kitty monitor. No large pretraining or SFT experiment was started.

The pinned, locally cached EmbeddingGemma encoder loads offline: 271,002,624 frozen parameters, 768-dimensional normalized embeddings, deterministic repeated output. A real isolated worker passes scoped-memory retrieval and clean shutdown. Its permissive retrieval threshold tests transport/isolation, not retrieval quality or classifier accuracy.

All 517 relocated assets (138,521,323,134 bytes) passed streaming SHA-256 verification before and after relocation. Later metadata checks remain unchanged. Actual NumPy tokenizer files round-trip Unicode and the relocated follow-up uint32 token shard opens successfully. Ten Git bundles verify and restore into bare repositories. Local evidence lives under validation/ and the protected recovery directory; it is intentionally excluded from public Git.

## Repeat bounded checks

From the sibling PyTorch workspace root: `python run.py test -q`; `python run.py --profile specialist --repo LLM-Specialist test -q`; `python run.py --profile specialist test -q tests/test_specialist.py tests/test_memory.py tests/test_native_memory.py tests/test_general_chat.py`; `python tools/build_packages.py`; `python tools/check_cli.py`; `python tools/verify_preservation.py` (use --full only for streaming asset verification).

From NumPy: `.venv/bin/python -m pytest -q -p no:cacheprovider tests`. Configure ignored artifacts.local.json before private checkpoint/data validation. See each README and environment records. Public-clone tests skip private reference/asset checks when those records are absent.

## Limits

Independent off-disk asset backup is still unavailable. NumPy's local environment inherits system libraries; a fresh isolated install is not validated. A full central suite in the specialist environment requires generative-only pyarrow; the separate generative suite and specialist integration selection above were used instead. Model quality was not improved or promoted by this restructuring.

Fresh GitHub branch clones pass integration 119 tests (7 skipped without private references/assets) and NumPy 126 tests. All seven published submodule pins resolve. Runtime environments were reused.
