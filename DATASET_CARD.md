# Dataset and tokenizer records

Current and historical corpora, tokenized shards, tokenizer files, manifests and SFT masks remain local protected assets. Their manifests preserve document/source hashes, splits, shard dtypes and tokenizer identities. Historical/local-artifact variants are kept separately rather than overwritten by similarly named current files.

The NumPy tokenizer is its own stdlib byte-level BPE with four reserved IDs (0-3); it is not interchangeable with the PyTorch tokenizer's eight reserved IDs. Equal vocabulary sizes do not establish compatibility. Verify tokenizer and dataset hashes before checkpoint resume or inference.

Data provenance/clearance statuses are retained exactly. Historical blocked caches remain blocked for production use. Refer to source manifests for source attribution and usage conditions; this migration does not grant new permissions or silently clear datasets. Runtime roots resolve relocated paths without modifying historical manifest bytes.
