"""Rebuild a provenance-cleared local corpus from approved document families.

This intentionally keeps only Project Gutenberg books and Python's official
documentation. Wikipedia, OpenStax, Khan Academy, and other educational
captures remain excluded until their exact revision and license metadata are
recorded.
"""

from __future__ import annotations
from llm_numpy.assets import workspace_root, normalize_arguments, resolve_input_path, resolve_output_path

import argparse
import json
import re
import sys
from pathlib import Path


from llm_numpy.data.pipeline import RawDocument, build_token_cache, prepare_documents, read_documents, write_documents


GUTENBERG_BOOKS = {
    "a tale of two cities": 98,
    "alice in wonderland": 11,
    "anna karenina": 1399,
    "adventures of huckleberry finn": 76,
    "crime and punishment": 2554,
    "don quixote": 996,
    "dracula": 345,
    "frankenstein": 84,
    "great expectations": 1400,
    "grimms fairy tales": 2591,
    "jane eyre": 1260,
    "little women": 514,
    "moby dick": 2701,
    "on the origin of species": 3207,
    "pride and prejudice": 1342,
    "sense and sensibility": 161,
    "sherlock holmes": 1661,
    "the adventures of sherlock holmes": 1661,
    "the count of monte cristo": 1184,
    "the divine comedy": 8800,
    "the importance of being earnest": 844,
    "the mystery of edwin drood": 600,
    "the odyssey": 1727,
    "the portrait of dorian gray": 1268,
    "the prince": 1232,
    "the republic": 1497,
    "the time machine": 35,
    "tom sawyer": 74,
    "ulysses": 4300,
    "war and peace": 2600,
    "wuthering heights": 768,
}

PYTHON_DOCS = {
    "Python control flow": "https://docs.python.org/3/tutorial/controlflow.html",
    "Python data structures": "https://docs.python.org/3/tutorial/datastructures.html",
    "Python errors": "https://docs.python.org/3/tutorial/errors.html",
    "Python tutorial": "https://docs.python.org/3/tutorial/introduction.html",
}


def base_title(title: str) -> str:
    return re.sub(r"\s+\[\d+\]$", "", title).strip()


def select_documents(documents):
    selected = []
    selected_titles = set()
    for item in documents:
        title = base_title(item.title)
        if item.source == "books" and title.lower() in GUTENBERG_BOOKS:
            selected.append(RawDocument("books", item.text, title))
            selected_titles.add(title.lower())
        elif item.source == "technical" and title in PYTHON_DOCS:
            selected.append(RawDocument("technical", item.text, title))
    if not selected:
        raise RuntimeError("no approved documents found in the input manifest")
    return selected, selected_titles


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-documents", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--provenance-output", type=Path, required=True)
    parser.add_argument("--vocab-size", type=int, default=16384)
    parser.add_argument("--tokenizer-sample-characters", type=int, default=2_000_000)
    args = normalize_arguments(parser.parse_args())

    source_documents = read_documents(args.base_documents)
    selected, selected_titles = select_documents(source_documents)
    prepared = prepare_documents(selected, validation_fraction=0.1, min_characters=200)
    args.output.mkdir(parents=True, exist_ok=True)
    write_documents(prepared, args.output / "documents.jsonl")
    manifest = build_token_cache(
        prepared,
        args.output,
        vocab_size=args.vocab_size,
        tokenizer_sample_characters=args.tokenizer_sample_characters,
    )

    records = [
        {
            "name": "project-gutenberg-books",
            "status": "cleared_for_this_cache",
            "license": "Project Gutenberg public-domain texts; verify jurisdiction before redistribution.",
            "url_pattern": "https://www.gutenberg.org/ebooks/{id}",
            "books": sorted(
                [{"title": title, "gutenberg_id": GUTENBERG_BOOKS[title.lower()]}
                 for title in selected_titles],
                key=lambda item: item["title"],
            ),
        },
        {
            "name": "python-documentation",
            "status": "cleared_for_this_cache",
            "license": "Python Software Foundation License; retain source attribution.",
            "urls": sorted(PYTHON_DOCS.values()),
        },
    ]
    provenance = {
        "format": "codexa-pretraining-source-manifest-v2",
        "status": "cleared",
        "purpose": "Source-cleared local corpus for bounded calibration; not a redistribution license.",
        "input_manifest": str(args.base_documents),
        "output_manifest": str(args.output / "manifest.json"),
        "selected_documents": len(prepared),
        "selected_titles": len(selected_titles),
        "sources": records,
        "excluded_sources": [
            {"name": "wikipedia", "reason": "exact dump revision and attribution not recorded"},
            {"name": "openstax", "reason": "permission status unresolved in the cached corpus"},
            {"name": "khan-academy", "reason": "cached capture is not validated educational text"},
            {"name": "other educational captures", "reason": "per-source license metadata not recorded"},
        ],
    }
    args.provenance_output.parent.mkdir(parents=True, exist_ok=True)
    args.provenance_output.write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"selected_documents={len(prepared)}")
    print(f"selected_titles={len(selected_titles)}")
    print(f"train_tokens={manifest['shards']['train']['tokens']}")
    print(f"validation_tokens={manifest['shards']['validation']['tokens']}")
    print(f"output={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
