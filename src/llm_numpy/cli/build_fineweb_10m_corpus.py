"""Build a bounded NumPy corpus from the pinned local FineWeb-Edu text cache."""

from __future__ import annotations
from llm_numpy.assets import workspace_root, normalize_arguments, resolve_input_path, resolve_output_path

import argparse
import json
import sys
from pathlib import Path


from llm_numpy.data.pipeline import RawDocument, build_token_cache, prepare_documents, read_documents, write_documents


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-documents", type=Path, required=True)
    parser.add_argument("--fineweb-jsonl", type=Path, required=True)
    parser.add_argument("--fineweb-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--provenance-output", type=Path, required=True)
    parser.add_argument("--max-fineweb-characters", type=int, default=30_000_000)
    parser.add_argument("--vocab-size", type=int, default=16_384)
    parser.add_argument("--tokenizer-sample-characters", type=int, default=2_000_000)
    args = normalize_arguments(parser.parse_args())

    documents = [RawDocument(item.source, item.text, item.title)
                 for item in read_documents(args.base_documents)]
    fineweb_characters = 0
    fineweb_documents = 0
    with args.fineweb_jsonl.open(encoding="utf-8") as handle:
        for line in handle:
            if fineweb_characters >= args.max_fineweb_characters:
                break
            item = json.loads(line)
            text = str(item.get("text", "")).strip()
            if len(text) < 200:
                continue
            remaining = args.max_fineweb_characters - fineweb_characters
            text = text[:remaining]
            documents.append(RawDocument(
                "fineweb_edu",
                text,
                f"FineWeb-Edu {item.get('document_id', fineweb_documents)}",
            ))
            fineweb_characters += len(text)
            fineweb_documents += 1

    prepared = prepare_documents(documents, validation_fraction=0.1,
                                 min_characters=200, chunk_characters=20_000)
    args.output.mkdir(parents=True, exist_ok=True)
    write_documents(prepared, args.output / "documents.jsonl")
    manifest = build_token_cache(
        prepared,
        args.output,
        vocab_size=args.vocab_size,
        tokenizer_sample_characters=args.tokenizer_sample_characters,
    )

    source_manifest = json.loads(args.fineweb_manifest.read_text(encoding="utf-8"))
    provenance = {
        "format": "codexa-pretraining-source-manifest-v3",
        "status": "cleared_for_bounded_run",
        "purpose": "Bounded NumPy 10M-token training corpus.",
        "output_manifest": str(args.output / "manifest.json"),
        "selected_fineweb_documents": fineweb_documents,
        "selected_fineweb_characters": fineweb_characters,
        "sources": [
            {
                "name": "existing-cleared-corpus",
                "manifest": str(args.base_documents),
                "status": "cleared",
            },
            {
                "name": "HuggingFaceFW/fineweb-edu",
                "configuration": source_manifest.get("dataset_configuration", "sample-10BT"),
                "revision": source_manifest.get("revision"),
                "license": source_manifest.get("license", "odc-by"),
                "input_manifest": str(args.fineweb_manifest),
                "input_checksums": source_manifest.get("input_checksums", {}),
                "status": "cleared_for_bounded_run",
            },
        ],
    }
    args.provenance_output.parent.mkdir(parents=True, exist_ok=True)
    args.provenance_output.write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"fineweb_documents={fineweb_documents}")
    print(f"fineweb_characters={fineweb_characters}")
    print(f"documents={len(prepared)}")
    print(f"train_tokens={manifest['shards']['train']['tokens']}")
    print(f"validation_tokens={manifest['shards']['validation']['tokens']}")
    print(f"total_tokens={manifest['shards']['train']['tokens'] + manifest['shards']['validation']['tokens']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
