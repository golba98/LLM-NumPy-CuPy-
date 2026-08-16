"""Download a small, reproducible multi-source general-language corpus.

The defaults are intentionally bounded.  This is a preparation tool, not a
large web crawl: it downloads a few public-domain books, encyclopedia pages,
educational pages, and technical documentation, then writes a persistent
tokenized cache under ``data/tokenized/general-small``.
"""

from __future__ import annotations

import argparse
import html
from http.client import IncompleteRead
import re
import signal
import sys
from pathlib import Path
from urllib.request import Request, urlopen

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.pipeline import RawDocument, build_token_cache, prepare_documents, write_documents


SOURCES = (
    ("general_educational", "Khan Academy", "https://www.khanacademy.org/science/biology", "html"),
    ("general_educational", "OpenStax biology", "https://openstax.org/books/biology-2e/pages/1-introduction", "html"),
    ("general_educational", "OpenStax physics", "https://openstax.org/books/physics/pages/1-introduction", "html"),
    ("general_educational", "OpenStax economics", "https://openstax.org/books/principles-economics-3e/pages/1-introduction", "html"),
    ("wikipedia", "Language model", "https://en.wikipedia.org/wiki/Language_model", "html"),
    ("wikipedia", "Natural language processing", "https://en.wikipedia.org/wiki/Natural_language_processing", "html"),
    ("wikipedia", "Artificial intelligence", "https://en.wikipedia.org/wiki/Artificial_intelligence", "html"),
    ("wikipedia", "Computer", "https://en.wikipedia.org/wiki/Computer", "html"),
    ("wikipedia", "Physics", "https://en.wikipedia.org/wiki/Physics", "html"),
    ("wikipedia", "Biology", "https://en.wikipedia.org/wiki/Biology", "html"),
    ("books", "Pride and Prejudice", "https://www.gutenberg.org/files/1342/1342-0.txt", "text"),
    ("books", "Alice in Wonderland", "https://www.gutenberg.org/files/11/11-0.txt", "text"),
    ("books", "Frankenstein", "https://www.gutenberg.org/files/84/84-0.txt", "text"),
    ("books", "Sherlock Holmes", "https://www.gutenberg.org/files/1661/1661-0.txt", "text"),
    ("books", "Moby Dick", "https://www.gutenberg.org/files/2701/2701-0.txt", "text"),
    ("books", "A Tale of Two Cities", "https://www.gutenberg.org/files/98/98-0.txt", "text"),
    ("books", "Tom Sawyer", "https://www.gutenberg.org/files/74/74-0.txt", "text"),
    ("books", "The Time Machine", "https://www.gutenberg.org/files/35/35-0.txt", "text"),
    ("books", "War and Peace", "https://www.gutenberg.org/files/2600/2600-0.txt", "text"),
    ("books", "Great Expectations", "https://www.gutenberg.org/files/1400/1400-0.txt", "text"),
    ("books", "Adventures of Huckleberry Finn", "https://www.gutenberg.org/files/76/76-0.txt", "text"),
    ("books", "The Republic", "https://www.gutenberg.org/files/1497/1497-0.txt", "text"),
    ("books", "Jane Eyre", "https://www.gutenberg.org/files/1260/1260-0.txt", "text"),
    ("books", "Wuthering Heights", "https://www.gutenberg.org/files/768/768-0.txt", "text"),
    ("books", "Little Women", "https://www.gutenberg.org/files/514/514-0.txt", "text"),
    ("books", "The Odyssey", "https://www.gutenberg.org/files/1727/1727-0.txt", "text"),
    ("technical", "Python tutorial", "https://docs.python.org/3/tutorial/introduction.html", "html"),
    ("technical", "Python control flow", "https://docs.python.org/3/tutorial/controlflow.html", "html"),
    ("technical", "Python data structures", "https://docs.python.org/3/tutorial/datastructures.html", "html"),
    ("technical", "Python errors", "https://docs.python.org/3/tutorial/errors.html", "html"),
    ("technical", "NumPy user guide", "https://numpy.org/doc/stable/user/whatisnumpy.html", "html"),
    ("technical", "NumPy absolute beginners", "https://numpy.org/doc/stable/user/absolute_beginners.html", "html"),
)


def _strip_html(value: str) -> str:
    value = re.sub(r"<script\b.*?</script>|<style\b.*?</style>", " ", value, flags=re.I | re.S)
    value = re.sub(r"<[^>]+>", " ", value)
    return html.unescape(re.sub(r"\s+", " ", value))


def _download_bytes(url: str, chunk_size: int = 512 * 1024, retries: int = 4) -> bytes:
    """Download a bounded resource with recovery for incomplete HTTP reads."""
    chunks = bytearray()
    total = None
    while total is None or len(chunks) < total:
        start = len(chunks)
        for attempt in range(retries):
            headers = {"User-Agent": "Codexa-NumPy-corpus-prep/1.0"}
            if start:
                headers["Range"] = f"bytes={start}-{start + chunk_size - 1}"
            try:
                with urlopen(Request(url, headers=headers), timeout=30) as response:
                    payload = response.read(chunk_size)
                    content_range = response.headers.get("Content-Range")
                    if content_range:
                        total = int(content_range.rsplit("/", 1)[1])
                    elif response.headers.get("Content-Length") and not start:
                        total = int(response.headers["Content-Length"])
                if payload:
                    chunks.extend(payload)
                    break
            except IncompleteRead as error:
                if error.partial:
                    chunks.extend(error.partial)
                    break
            except Exception:
                if attempt == retries - 1:
                    raise
        else:
            raise RuntimeError(f"unable to download {url} after {retries} retries")
        if total is None and len(chunks) < chunk_size:
            break
    return bytes(chunks[:total] if total else chunks)


def fetch(url: str, kind: str) -> str:
    def timeout_handler(signum, frame):
        raise TimeoutError(f"download exceeded 45 seconds: {url}")

    previous_handler = signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(45)
    try:
        value = _download_bytes(url).decode("utf-8", errors="replace")
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous_handler)
    return _strip_html(value) if kind == "html" else value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("data/tokenized/general-small"))
    parser.add_argument("--max-total-characters", type=int, default=1_000_000)
    parser.add_argument("--max-document-characters", type=int, default=180_000)
    parser.add_argument("--vocab-size", type=int, default=1024)
    parser.add_argument("--tokenizer-sample-characters", type=int, default=100_000)
    args = parser.parse_args()
    raw_dir = args.output.parent.parent / "raw" / "general"
    documents = []
    total = 0
    per_source_budget = max(1, args.max_total_characters // len(SOURCES))
    for source, title, url, kind in SOURCES:
        try:
            text = fetch(url, kind)[: args.max_document_characters]
        except Exception as exc:
            print(f"warning: unable to fetch {title}: {exc}", file=sys.stderr)
            continue
        remaining = min(args.max_total_characters - total, per_source_budget)
        if remaining <= 0:
            break
        text = text[:remaining]
        raw_path = raw_dir / f"{source}_{re.sub(r'[^a-z0-9]+', '_', title.lower()).strip('_')}.txt"
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        raw_path.write_text(text, encoding="utf-8")
        documents.append(RawDocument(source, text, title))
        total += len(text)

    prepared = prepare_documents(documents, validation_fraction=0.1, min_characters=200)
    manifest_path = args.output / "documents.jsonl"
    write_documents(prepared, manifest_path)
    manifest = build_token_cache(
        prepared, args.output, vocab_size=args.vocab_size,
        tokenizer_sample_characters=args.tokenizer_sample_characters,
    )
    print(f"prepared_documents={len(prepared)} raw_characters={total}")
    print(f"train_tokens={manifest['shards']['train']['tokens']} validation_tokens={manifest['shards']['validation']['tokens']}")
    print(f"vocab_size={manifest['tokenizer']['vocab_size']} bytes_per_token={manifest['tokenizer']['bytes_per_token']:.3f}")
    print(f"manifest={args.output / 'manifest.json'}")


if __name__ == "__main__":
    main()
