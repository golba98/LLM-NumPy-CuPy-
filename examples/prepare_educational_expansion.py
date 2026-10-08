"""Add fixed public-domain educational and scientific texts to a cache."""

from __future__ import annotations
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "src"))

import argparse
import hashlib
import re
import sys
from pathlib import Path
from urllib.request import Request, urlopen

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from llm_numpy.data.pipeline import RawDocument, build_token_cache, prepare_documents, read_documents, write_documents


# Gutenberg IDs and titles are fixed so the input set remains auditable.
# These are educational, mathematical, scientific, civics, and teacher-text
# works rather than conversational data or additional novels.
WORKS = (
    (33283, "Calculus Made Easy"),
    (21076, "The First Six Books of the Elements of Euclid"),
    (26839, "Mathematical Recreations and Essays"),
    (26752, "The Way To Geometry"),
    (26373, "The Elements of Non-Euclidean Geometry"),
    (22599, "The Hindu-Arabic Numerals"),
    (27635, "The Canterbury Puzzles"),
    (16713, "Amusements in Mathematics"),
    (22627, "German Science Reader"),
    (21016, "Essays on the Theory of Numbers"),
    (38769, "A Course of Pure Mathematics"),
    (39713, "The Foundations of Science"),
    (41568, "An Introduction to Mathematics"),
    (18440, "Logic: Deductive and Inductive"),
    (17366, "Notes on Nursing"),
    (55264, "On Growth and Form"),
    (78009, "A Guide to the History of Physical Education"),
    (21829, "A Treatise on Domestic Economy"),
    (25545, "Children's Literature: A Textbook of Sources"),
    (27742, "Popular Education"),
    (23320, "The Deaf: Their Position in Society"),
    (28036, "Public School Education"),
    (21353, "Civics and Health"),
    (26139, "Ontario Teachers' Manuals: Nature Study"),
    (28335, "How Two Boys Made Their Own Electrical Apparatus"),
    (20871, "Human Foods and Their Nutritive Value"),
    (22766, "Electricity for Boys"),
    (27790, "A Practical Enquiry into the Philosophy of Education"),
    (22251, "The Teacher"),
    (27963, "History of Education"),
    (852, "Democracy and Education"),
    (28097, "English: Composition and Literature"),
    (78050, "Principia Mathematica, Volume I"),
    (39041, "Elementary Illustrations of the Differential and Integral Calculus"),
    (302, "The Fibonacci Number Series"),
    (57359, "The Logic of Chance"),
    (26262, "Utility of Quaternions in Physics"),
    (15114, "An Investigation of the Laws of Thought"),
    (79080, "Curiosa Mathematica"),
    (4763, "The Game of Logic"),
    (23321, "The Choctaw Freedmen and the Story of Oak Hill Industrial Academy"),
    (27793, "Child and Country"),
    (25944, "Essentials of Diseases of the Skin"),
    (27279, "Southern Literature From 1579-1895"),
    (23033, "Classic French Course in English"),
    (22795, "The Ontario High School Reader"),
    (21045, "Education and the Higher Life"),
    (25897, "Aratra Pentelici: Seven Lectures on Sculpture"),
    (22636, "A Middle High German Primer"),
    (2087, "Life and Letters of Charles Darwin"),
    (2739, "More Letters of Charles Darwin"),
    (27509, "The 2006 CIA World Factbook"),
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
            request = Request(url, headers={"User-Agent": "Codexa-NumPy-corpus-prep/1.0"})
            with urlopen(request, timeout=20) as response:
                return response.read(4_000_000).decode("utf-8", errors="replace")
        except Exception as exc:
            last_error = exc
    raise RuntimeError(f"book {book_id} unavailable: {last_error}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-documents", type=Path,
                        default=Path("data/tokenized/general-small-10m/documents.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("data/tokenized/general-small-10m-edu"))
    parser.add_argument("--max-new-characters", type=int, default=8_000_000)
    parser.add_argument("--max-document-characters", type=int, default=500_000)
    parser.add_argument("--vocab-size", type=int, default=16_384)
    parser.add_argument("--tokenizer-sample-characters", type=int, default=2_000_000)
    from llm_numpy.assets import normalize_arguments
    args = normalize_arguments(parser.parse_args())

    documents = [RawDocument(item.source, item.text, item.title)
                 for item in read_documents(args.base_documents)]
    existing_hashes = {hashlib.sha256(item.text.encode("utf-8")).hexdigest()
                       for item in documents}
    existing_titles = {
        re.sub(r"\s+\[\d+\]$", "", item.title).strip()
        for item in documents if item.source == "general_educational"
    }
    raw_dir = args.output.parent.parent / "raw" / "general-small-10m-edu"
    added = 0
    for book_id, title in WORKS:
        if added >= args.max_new_characters:
            break
        if title in existing_titles:
            continue
        try:
            text = fetch_book(book_id)
        except Exception as exc:
            print(f"warning: unable to fetch {title}: {exc}", file=sys.stderr)
            continue
        text = text[:min(args.max_document_characters, args.max_new_characters - added)]
        if len(text) < 500:
            continue
        text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if text_hash in existing_hashes:
            continue
        path = raw_dir / f"{book_id}.txt"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        documents.append(RawDocument("general_educational", text, title))
        existing_hashes.add(text_hash)
        existing_titles.add(title)
        added += len(text)
        print(f"added={title} characters={len(text)} total_new={added}", flush=True)

    prepared = prepare_documents(documents, validation_fraction=0.1,
                                 min_characters=200, chunk_characters=20_000)
    args.output.mkdir(parents=True, exist_ok=True)
    write_documents(prepared, args.output / "documents.jsonl")
    manifest = build_token_cache(
        prepared, args.output, vocab_size=args.vocab_size,
        tokenizer_sample_characters=args.tokenizer_sample_characters)
    print(f"prepared_documents={manifest['documents']['documents']}")
    print(f"raw_characters={manifest['documents']['raw_characters']}")
    print(f"train_tokens={manifest['shards']['train']['tokens']}")
    print(f"validation_tokens={manifest['shards']['validation']['tokens']}")
    print(f"manifest={args.output / 'manifest.json'}")


if __name__ == "__main__":
    main()
