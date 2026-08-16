"""Bounded, reproducible corpus preparation and token-cache utilities.

The pipeline deliberately uses only the standard library and NumPy.  Raw
documents are normalized and deduplicated before a deterministic document
split, then token IDs are written as uint32 binary files that can be opened
with ``numpy.memmap`` during training.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np

from src.tokenization.bpe import ByteLevelBPETokenizer


@dataclass(frozen=True)
class RawDocument:
    source: str
    text: str
    title: str = ""


@dataclass(frozen=True)
class PreparedDocument:
    source: str
    title: str
    text: str
    split: str
    sha256: str


_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_SPACE_RE = re.compile(r"[ \t\f\v]+")


def normalize_text(text: str) -> str:
    """Normalize line endings, controls, and excessive horizontal spacing."""
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _CONTROL_RE.sub("", text)
    text = "\n".join(_SPACE_RE.sub(" ", line).strip() for line in text.split("\n"))
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _digest_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def prepare_documents(
    documents: Iterable[RawDocument],
    validation_fraction: float = 0.1,
    min_characters: int = 80,
    max_characters: int | None = None,
    chunk_characters: int = 20_000,
) -> List[PreparedDocument]:
    """Normalize, filter, deduplicate, and deterministically split documents."""
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1")
    if chunk_characters <= 0:
        raise ValueError("chunk_characters must be positive")
    unique: Dict[str, RawDocument] = {}
    for document in documents:
        text = normalize_text(document.text)
        if max_characters is not None:
            text = text[:max_characters]
        paragraphs = [part for part in re.split(r"\n{2,}", text) if part]
        chunks: List[str] = []
        current = ""
        for paragraph in paragraphs or [text]:
            while len(paragraph) > chunk_characters:
                if current:
                    chunks.append(current)
                    current = ""
                chunks.append(paragraph[:chunk_characters])
                paragraph = paragraph[chunk_characters:]
            candidate = f"{current}\n\n{paragraph}" if current else paragraph
            if len(candidate) > chunk_characters and current:
                chunks.append(current)
                current = paragraph
            else:
                current = candidate
        if current:
            chunks.append(current)
        for index, chunk in enumerate(chunks):
            if len(chunk) < min_characters:
                continue
            digest = _digest(chunk)
            title = f"{document.title} [{index + 1}]" if len(chunks) > 1 else document.title
            unique.setdefault(digest, RawDocument(document.source, chunk, title))

    prepared = []
    for digest, document in sorted(unique.items(), key=lambda item: (item[1].source, item[0])):
        # Hash bucketing avoids order-dependent splits while keeping documents intact.
        bucket = int(digest[:8], 16) / 0x100000000
        split = "validation" if bucket < validation_fraction else "train"
        prepared.append(PreparedDocument(document.source, document.title, document.text, split, digest))
    if len(prepared) > 1 and not any(item.split == "validation" for item in prepared):
        item = prepared[-1]
        prepared[-1] = PreparedDocument(item.source, item.title, item.text, "validation", item.sha256)
    if len(prepared) > 1 and not any(item.split == "train" for item in prepared):
        item = prepared[0]
        prepared[0] = PreparedDocument(item.source, item.title, item.text, "train", item.sha256)
    return prepared


def source_statistics(documents: Sequence[PreparedDocument]) -> Dict[str, object]:
    by_source: Dict[str, Dict[str, int]] = {}
    for document in documents:
        entry = by_source.setdefault(document.source, {"documents": 0, "characters": 0})
        entry["documents"] += 1
        entry["characters"] += len(document.text)
    return {
        "documents": len(documents),
        "raw_characters": sum(len(document.text) for document in documents),
        "sources": by_source,
        "train_documents": sum(document.split == "train" for document in documents),
        "validation_documents": sum(document.split == "validation" for document in documents),
    }


def write_documents(documents: Sequence[PreparedDocument], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for document in documents:
            handle.write(json.dumps(asdict(document), ensure_ascii=False, sort_keys=True) + "\n")


def read_documents(path: str | Path) -> List[PreparedDocument]:
    with Path(path).open(encoding="utf-8") as handle:
        return [PreparedDocument(**json.loads(line)) for line in handle if line.strip()]


def _write_tokens(path: Path, token_ids: Sequence[int], storage_dtype: str = "auto") -> Dict[str, object]:
    path.parent.mkdir(parents=True, exist_ok=True)
    if storage_dtype == "auto":
        storage_dtype = "uint16" if max(token_ids, default=0) <= np.iinfo(np.uint16).max else "uint32"
    if storage_dtype not in {"uint16", "uint32"}:
        raise ValueError("storage_dtype must be auto, uint16, or uint32")
    values = np.asarray(token_ids, dtype=getattr(np, storage_dtype))
    values.tofile(path)
    return {"path": str(path), "tokens": int(values.size), "sha256": _digest_bytes(path.read_bytes())}


def build_token_cache(
    documents: Sequence[PreparedDocument],
    output_dir: str | Path,
    vocab_size: int = 1024,
    tokenizer_sample_characters: int = 2_000_000,
    storage_dtype: str = "uint32",
) -> Dict[str, object]:
    """Train our byte BPE on train text and persist tokenizer, shards, and manifest."""
    output = Path(output_dir)
    tokenizer = ByteLevelBPETokenizer()
    train_documents = [item for item in documents if item.split == "train"]
    validation_documents = [item for item in documents if item.split == "validation"]
    if not train_documents or not validation_documents:
        raise ValueError("both train and validation documents are required")
    sample: List[str] = []
    used = 0
    for document in train_documents:
        if used >= tokenizer_sample_characters:
            break
        text = document.text[: tokenizer_sample_characters - used]
        sample.append(text)
        used += len(text)
    tokenizer.train(sample, vocab_size=vocab_size)
    output.mkdir(parents=True, exist_ok=True)
    tokenizer_path = output / "tokenizer.json"
    tokenizer.save(tokenizer_path)

    shards = {}
    for split, split_documents in (("train", train_documents), ("validation", validation_documents)):
        ids: List[int] = []
        for document in split_documents:
            ids.extend(tokenizer.encode(document.text))
            ids.append(tokenizer.EOS_ID)
        shards[split] = _write_tokens(output / f"{split}.bin", ids, storage_dtype)
    stats = tokenizer.analyze_corpus([item.text for item in documents])
    manifest = {
        "format": "codexa-token-cache-v1",
        "documents": source_statistics(documents),
        "tokenizer": {**stats, "path": str(tokenizer_path), "training_characters": used},
        "shards": shards,
        "document_manifest_sha256": _digest("\n".join(item.sha256 for item in documents)),
        "token_storage_dtype": storage_dtype if storage_dtype != "auto" else ("uint16" if vocab_size <= 65536 else "uint32"),
    }
    manifest_path = output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest


def open_token_shard(manifest: Mapping[str, object], split: str) -> np.memmap:
    """Open a persisted shard without loading the corpus into RAM."""
    shard = manifest["shards"][split]  # type: ignore[index]
    path = Path(shard["path"])  # type: ignore[index]
    dtype = manifest.get("token_storage_dtype", shard.get("dtype", "uint32"))  # type: ignore[union-attr]
    return np.memmap(path, dtype=np.dtype(dtype), mode="r", shape=(int(shard["tokens"]),))  # type: ignore[index]
