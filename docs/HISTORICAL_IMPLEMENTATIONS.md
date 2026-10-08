# Historical NumPy implementation preservation

The deprecated pre-backend implementation contains 85 substantive files. Fifty match active source paths byte-for-byte; 35 differ, including a checkpoint, an earlier BPE codec and training demonstration. It is preserved as an exact private source snapshot and supplemental nested-data archive in the configured recovery store. Its checkpoint is relocated separately and validated with the active legacy loader.

The local-artifact directory contains 212 files with no byte-identical counterpart in active source/data: corpora, raw educational texts, tokenizers, tokenized shards and provenance manifests. All remain in a separately rooted historical asset tree. Active files with the same relative names are not overwritten. Original directories remain pending retirement; no directory was deleted because of its name.
