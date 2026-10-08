"""Add a bounded set of public-domain books to the existing general corpus.

This deliberately avoids the slower encyclopedia endpoints.  It reuses the
already validated multi-source manifest and only adds English Gutenberg texts,
with a hard character budget and per-request timeout.
"""

from __future__ import annotations
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "src"))

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from llm_numpy.data.pipeline import RawDocument, build_token_cache, prepare_documents, read_documents, write_documents
from examples.prepare_general_corpus import fetch


BOOKS = (
    (345, "Dracula"),
    (844, "The Importance of Being Earnest"),
    (996, "Don Quixote"),
    (1184, "The Count of Monte Cristo"),
    (1232, "The Prince"),
    (1399, "Anna Karenina"),
    (161, " sense and sensibility"),
    (2554, "Crime and Punishment"),
    (2591, "Grimms Fairy Tales"),
    (2701, "Moby Dick"),
    (3207, "On the Origin of Species"),
    (4300, "Ulysses"),
    (514, "Little Women"),
    (600, "The Mystery of Edwin Drood"),
    (768, "Wuthering Heights"),
    (8800, "The Divine Comedy"),
    (1268, "The Portrait of Dorian Gray"),
    (1661, "The Adventures of Sherlock Holmes"),
    (2147, "The Works of Edgar Allan Poe"),
    (2852, "The Hound of the Baskervilles"),
)


def fetch_book(book_id: int) -> str:
    urls = (
        f"https://www.gutenberg.org/cache/epub/{book_id}/pg{book_id}.txt",
        f"https://www.gutenberg.org/files/{book_id}/{book_id}-0.txt",
        f"https://www.gutenberg.org/files/{book_id}/{book_id}.txt",
    )
    last_error = None
    for url in urls:
        try:
            return fetch(url, "text")
        except Exception as exc:
            last_error = exc
    raise RuntimeError(f"book {book_id} unavailable: {last_error}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-manifest", type=Path, default=Path("data/tokenized/general-smoke-1m/documents.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("data/tokenized/general-expanded"))
    parser.add_argument("--max-new-characters", type=int, default=8_000_000)
    parser.add_argument("--max-document-characters", type=int, default=500_000)
    parser.add_argument("--vocab-size", type=int, default=16_384)
    parser.add_argument("--tokenizer-sample-characters", type=int, default=500_000)
    from llm_numpy.assets import normalize_arguments
    args = normalize_arguments(parser.parse_args())
    documents = [RawDocument(item.source, item.text, item.title) for item in read_documents(args.base_manifest)]
    new_characters = 0
    raw_dir = args.output.parent.parent / "raw" / "book-expansion"
    for book_id, title in BOOKS:
        if new_characters >= args.max_new_characters:
            break
        try:
            text = fetch_book(book_id)[: min(args.max_document_characters, args.max_new_characters - new_characters)]
        except Exception as exc:
            print(f"warning: unable to fetch {title}: {exc}", file=sys.stderr)
            continue
        if len(text) < 200:
            continue
        raw_path = raw_dir / f"{book_id}.txt"
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        raw_path.write_text(text, encoding="utf-8")
        documents.append(RawDocument("books", text, title.strip()))
        new_characters += len(text)
        print(f"added={title.strip()} characters={len(text)} total_new={new_characters}", flush=True)

    prepared = prepare_documents(documents, validation_fraction=0.1, min_characters=200, chunk_characters=20_000)
    args.output.mkdir(parents=True, exist_ok=True)
    write_documents(prepared, args.output / "documents.jsonl")
    manifest = build_token_cache(prepared, args.output, vocab_size=args.vocab_size,
                                 tokenizer_sample_characters=args.tokenizer_sample_characters)
    print(f"prepared_documents={manifest['documents']['documents']} raw_characters={manifest['documents']['raw_characters']}")
    print(f"train_tokens={manifest['shards']['train']['tokens']} validation_tokens={manifest['shards']['validation']['tokens']}")
    print(f"manifest={args.output / 'manifest.json'}")


if __name__ == "__main__":
    main()
