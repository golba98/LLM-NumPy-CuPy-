"""Build a larger, fixed multi-source general-language cache.

The source lists are intentionally explicit and bounded.  Existing prepared
documents are retained, while new material is downloaded into ``data/raw``
and then normalized, deduplicated, deterministically split, BPE-tokenized,
and persisted as memmap-friendly shards.
"""

from __future__ import annotations
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "src"))

import argparse
import html
import re
import sys
from pathlib import Path
from urllib.request import Request, urlopen

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from llm_numpy.data.pipeline import RawDocument, build_token_cache, prepare_documents, read_documents, write_documents
from examples.prepare_general_corpus import _strip_html


# Fixed URLs keep the corpus reproducible and auditable.  Wikipedia and
# OpenStax are CC-licensed educational/general sources; Python documentation
# is PSF-licensed technical source material.
WIKIPEDIA_TOPICS = (
    "Algebra", "Calculus", "Geometry", "Statistics", "Probability",
    "Computer science", "Data structure", "Algorithm", "Operating system",
    "Internet", "Database", "Computer network", "Cryptography",
    "Machine learning", "Deep learning", "Neural network", "Robotics",
    "Astronomy", "Earth science", "Climate change", "Geology", "Chemistry",
    "Organic chemistry", "Cell biology", "Genetics", "Evolution", "Ecology",
    "Human body", "Medicine", "Psychology", "Sociology", "Economics",
    "Macroeconomics", "Microeconomics", "Political science", "Law", "History",
    "Ancient history", "World War II", "Philosophy", "Ethics", "Logic",
    "Linguistics", "Art", "Music", "Architecture", "Geography", "Agriculture",
    "Energy", "Renewable energy", "Materials science", "Physics", "Biology",
)

OPENSTAX_BOOKS = (
    ("biology-2e", "biology"),
    ("physics-2e", "physics"),
    ("principles-economics-3e", "economics"),
    ("introduction-sociology-3e", "sociology"),
    ("psychology-2e", "psychology"),
)

PYTHON_DOCS = (
    "library/collections.html", "library/itertools.html", "library/functools.html",
    "library/os.html", "library/sys.html", "library/pathlib.html",
    "library/json.html", "library/re.html", "library/math.html",
    "library/random.html", "library/datetime.html", "library/asyncio.html",
    "library/typing.html", "library/dataclasses.html", "library/sqlite3.html",
    "howto/regex.html", "howto/sorting.html", "howto/functional.html",
    "tutorial/classes.html", "tutorial/modules.html", "tutorial/inputoutput.html",
    "tutorial/stdlib.html", "tutorial/stdlib2.html", "tutorial/venv.html",
)


def safe_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def fetch_direct(url: str, kind: str) -> str:
    """Fetch one bounded document without the range-retry loop.

    Wikipedia frequently ignores byte ranges, which can make a range-based
    downloader retry the same large page indefinitely.  These source pages
    are individually bounded and are safer to fetch in one request.
    """
    request = Request(url, headers={"User-Agent": "Codexa-NumPy-corpus-prep/1.0"})
    with urlopen(request, timeout=20) as response:
        value = response.read(4_000_000).decode("utf-8", errors="replace")
    return _strip_html(value) if kind == "html" else value


def openstax_pages(book: str, limit: int = 40) -> list[tuple[str, str]]:
    """Discover stable chapter slugs from the book's public table of contents."""
    index_url = f"https://openstax.org/books/{book}/pages/1-introduction"
    try:
        html_text = fetch_direct(index_url, "text")
    except Exception:
        return []
    pattern = re.compile(
        rf"href\s*=\s*[\"']/books/{re.escape(book)}/pages/([a-z0-9][a-z0-9-]*)[\"?#]",
        re.IGNORECASE,
    )
    seen: set[str] = set()
    pages = []
    for slug in pattern.findall(html_text):
        if slug in seen or slug == "1-introduction" or len(slug) < 3:
            continue
        seen.add(slug)
        pages.append((slug.replace("-", " "), f"https://openstax.org/books/{book}/pages/{slug}"))
        if len(pages) >= limit:
            break
    return pages


def add_download(documents: list[RawDocument], raw_dir: Path, source: str,
                 title: str, url: str, kind: str, budget: int) -> int:
    try:
        text = fetch_direct(url, kind)
    except Exception as exc:
        print(f"warning: unable to fetch {title}: {exc}", file=sys.stderr)
        return 0
    text = text[:budget]
    if len(text) < 500:
        return 0
    path = raw_dir / source / f"{safe_name(title)}.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    documents.append(RawDocument(source, text, title))
    print(f"added source={source} title={title} chars={len(text)}", flush=True)
    return len(text)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-documents", type=Path,
                        default=Path("data/tokenized/general-expanded/documents.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("data/tokenized/general-small-10m"))
    parser.add_argument("--max-new-wikipedia-characters", type=int, default=10_000_000)
    parser.add_argument("--max-new-educational-characters", type=int, default=10_000_000)
    parser.add_argument("--max-new-technical-characters", type=int, default=5_000_000)
    parser.add_argument("--per-document-characters", type=int, default=180_000)
    parser.add_argument("--vocab-size", type=int, default=16_384)
    parser.add_argument("--tokenizer-sample-characters", type=int, default=2_000_000)
    from llm_numpy.assets import normalize_arguments
    args = normalize_arguments(parser.parse_args())

    documents = [RawDocument(item.source, item.text, item.title)
                 for item in read_documents(args.base_documents)]
    raw_dir = args.output.parent.parent / "raw" / "general-small-10m"

    wiki_total = 0
    for topic in WIKIPEDIA_TOPICS:
        if wiki_total >= args.max_new_wikipedia_characters:
            break
        budget = min(args.per_document_characters,
                     args.max_new_wikipedia_characters - wiki_total)
        wiki_total += add_download(
            documents, raw_dir, "wikipedia", topic,
            "https://en.wikipedia.org/wiki/" + topic.replace(" ", "_"), "html", budget)

    education_total = 0
    for book, subject in OPENSTAX_BOOKS:
        for slug_title, url in openstax_pages(book):
            if education_total >= args.max_new_educational_characters:
                break
            budget = min(args.per_document_characters,
                         args.max_new_educational_characters - education_total)
            title = f"OpenStax {subject} {slug_title}"
            education_total += add_download(
                documents, raw_dir, "general_educational", title, url, "html", budget)

    technical_total = 0
    for path in PYTHON_DOCS:
        if technical_total >= args.max_new_technical_characters:
            break
        budget = min(args.per_document_characters,
                     args.max_new_technical_characters - technical_total)
        technical_total += add_download(
            documents, raw_dir, "technical", f"Python {path}",
            "https://docs.python.org/3/" + path, "html", budget)

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
