# NumPy model and compatibility

The independently implemented production target uses vocabulary 16,384, width 768, 34 decoder layers, 12 heads, SwiGLU hidden width 2,048, RoPE, RMSNorm and tied token/output embeddings: 253,284,096 parameters. The current production configuration uses context 128. Other tier configurations and the earlier 240.9M benchmark remain available.

Parameter enumeration and tied-weight identity are serialization contracts: checkpoints store indexed model arrays plus AdamW moments and training counters. The current format also records model/training configuration hashes, tokenizer identity, dataset lineage and NumPy RNG state; legacy checkpoints remain readable. Weight-only inference loading does not restore optimizer/RNG state and must not be described as an exact training resume.

This migration changes package paths and asset resolution, not mathematical operations. Check validation records for tested CPU/CUDA behavior, preserved reference fixtures and actual checkpoint parity. Existing conversational checkpoints and historical base runs remain experimental; restructuring does not promote model quality or change their training lineage.
