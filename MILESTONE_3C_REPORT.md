# Milestone 3C NumPy Reference Report

## Regression and gradient equivalence

The full suite passes with **102 tests**: 97 Milestone 3B regressions plus 5 Milestone 3C tests. Accumulation scales each already-mean microbatch loss by `n_i / N`, where `n_i` is its target-token count and `N` is the total target-token count. Gradients are accumulated without clearing them, clipped once after the final microbatch, and passed to one AdamW update. The deterministic equal-microbatch test passes with maximum gradient and parameter differences below `1e-12` in float64.

## Real-corpus validation

The bundled corpus is a 1,675-character public-domain Shakespeare excerpt. The split is contiguous and deterministic at 90/10, before tokenization and overlapping windows; BPE training uses the training split only.

| Measurement | Result |
| --- | ---: |
| Vocabulary / merges | 512 / 252 |
| Train / validation tokens | 524 / 89 |
| Model parameters | 133,440 |
| Initial validation loss | 6.219234 |
| Best validation loss | 6.156773 at step 20 |
| Final training loss | 5.935016 |

The run completed without NaN or Inf values. Generation before and after training is recorded by `examples/train_real_corpus.py`; generation is qualitative evidence only.

## Performance baseline

Canonical benchmark: batch 4, context 32, vocabulary 256, 2 layers, dimension 64, 4 heads, hidden dimension 176, 128 prediction tokens/step, 10 timed steps after 3 warm-ups.

| Stage | Mean ms | Share |
| --- | ---: | ---: |
| Forward | 2.717 | 39.1% |
| Backward | 3.382 | 48.7% |
| Gradient clipping | 0.141 | 2.0% |
| AdamW | 0.691 | 9.9% |
| Total step | 6.951 | 100% |

Measured throughput was 18,415.69 tokens/sec on Python 3.14.6 and NumPy 2.4.6. Context scaling measured 4.904 ms (T=16), 9.110 ms (T=32), and 17.809 ms (T=64). The dominant measured costs are backward, forward, and AdamW; NumPy matrix multiplication uses the platform's native numerical/BLAS backend.

## Reference fixtures and next port

`reference/` contains deterministic tokens, configuration, forward arrays, gradients, and one-step optimizer arrays. Regenerate them with `python3 tools/generate_reference.py`; `tests/test_milestone3c.py` verifies reproducibility.

The recommended Milestone 4 order is raw C++ storage and strides, elementwise/reduction operations, matrix multiplication, broadcasting, reverse-mode autograd, then Linear, Embedding, RMSNorm, Softmax, RoPE, attention, SwiGLU, TransformerBlock, and TinyLLM. CUDA should wait until the C++ CPU implementation matches these fixtures.
