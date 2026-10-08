"""Build a tokenizer-compatible follow-up FineWeb token cache."""

from __future__ import annotations
from llm_numpy.assets import workspace_root, normalize_arguments, resolve_input_path, resolve_output_path

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np


from llm_numpy.data.pipeline import RawDocument, prepare_documents, source_statistics, write_documents
from llm_numpy.tokenization import ByteLevelBPETokenizer


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_tokens(path: Path, documents, tokenizer) -> dict:
    values = []
    for document in documents:
        values.extend(tokenizer.encode(document.text))
        values.append(tokenizer.EOS_ID)
    array = np.asarray(values, dtype=np.uint32)
    path.parent.mkdir(parents=True, exist_ok=True)
    array.tofile(path)
    return {"path": str(path), "tokens": int(array.size), "sha256": digest(path)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--fineweb-jsonl", type=Path, required=True)
    parser.add_argument("--fineweb-manifest", type=Path, required=True)
    parser.add_argument("--skip-characters", type=int, required=True)
    parser.add_argument("--max-characters", type=int, default=30_000_000)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--provenance-output", type=Path, required=True)
    args = normalize_arguments(parser.parse_args())

    skipped = 0
    selected = 0
    raw_documents = []
    with args.fineweb_jsonl.open(encoding="utf-8") as handle:
        for line in handle:
            item = json.loads(line)
            text = str(item.get("text", "")).strip()
            if len(text) < 200:
                continue
            if skipped < args.skip_characters:
                skipped += len(text)
                continue
            if selected >= args.max_characters:
                break
            text = text[: args.max_characters - selected]
            raw_documents.append(RawDocument("fineweb_edu", text, f"FineWeb-Edu followup {item.get('document_id', len(raw_documents))}"))
            selected += len(text)

    documents = prepare_documents(raw_documents, validation_fraction=0.1,
                                  min_characters=200, chunk_characters=20_000)
    args.output.mkdir(parents=True, exist_ok=True)
    write_documents(documents, args.output / "documents.jsonl")
    tokenizer = ByteLevelBPETokenizer.from_file(args.tokenizer)
    train = [item for item in documents if item.split == "train"]
    validation = [item for item in documents if item.split == "validation"]
    shards = {
        "train": write_tokens(args.output / "train.bin", train, tokenizer),
        "validation": write_tokens(args.output / "validation.bin", validation, tokenizer),
    }
    manifest = {
        "format": "codexa-token-cache-v1-followup",
        "documents": source_statistics(documents),
        "tokenizer": {
            "path": str(args.tokenizer),
            "sha256": digest(args.tokenizer),
            "vocab_size": len(tokenizer.vocab),
        },
        "shards": shards,
        "token_storage_dtype": "uint32",
        "document_manifest_sha256": hashlib.sha256("\n".join(item.sha256 for item in documents).encode()).hexdigest(),
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    upstream = json.loads(args.fineweb_manifest.read_text(encoding="utf-8"))
    provenance = {
        "format": "codexa-pretraining-source-manifest-v3-followup",
        "status": "cleared_for_bounded_run",
        "tokenizer": str(args.tokenizer),
        "source": "HuggingFaceFW/fineweb-edu",
        "revision": upstream.get("revision"),
        "license": upstream.get("license", "odc-by"),
        "input_manifest": str(args.fineweb_manifest),
        "skip_characters": args.skip_characters,
        "selected_characters": selected,
        "output_manifest": str(args.output / "manifest.json"),
    }
    args.provenance_output.parent.mkdir(parents=True, exist_ok=True)
    args.provenance_output.write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"selected_characters={selected}")
    print(f"documents={len(documents)}")
    print(f"train_tokens={shards['train']['tokens']}")
    print(f"validation_tokens={shards['validation']['tokens']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
