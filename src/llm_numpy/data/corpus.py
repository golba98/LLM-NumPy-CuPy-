"""Deterministic text-corpus helpers."""

from pathlib import Path
from typing import Iterable, List, Sequence, Union


class TextCorpus:
    def __init__(self, documents: Sequence[str], normalize_newlines: bool = True, drop_empty: bool = False):
        cleaned: List[str] = []
        for document in documents:
            if not isinstance(document, str):
                raise TypeError("documents must contain strings")
            value = document.replace("\r\n", "\n").replace("\r", "\n") if normalize_newlines else document
            if value or not drop_empty:
                cleaned.append(value)
        self.documents = cleaned

    @classmethod
    def from_text(cls, text: str, **kwargs) -> "TextCorpus":
        return cls([text], **kwargs)

    @classmethod
    def from_file(cls, path: Union[str, Path], **kwargs) -> "TextCorpus":
        return cls([Path(path).read_text(encoding="utf-8")], **kwargs)

    def with_eos(self, eos_token_id: int) -> List[int]:
        raise TypeError("with_eos requires tokenized documents; use token_ids()")

    def token_ids(self, tokenizer, eos_token_id: int) -> List[int]:
        result: List[int] = []
        for document in self.documents:
            result.extend(tokenizer.encode(document))
            result.append(int(eos_token_id))
        return result

    def split(self, train_fraction: float = 0.9) -> tuple["TextCorpus", "TextCorpus"]:
        """Split documents before tokenization/window construction."""
        if not 0 < train_fraction < 1:
            raise ValueError("train_fraction must be between 0 and 1")
        if len(self.documents) < 2:
            text = self.documents[0] if self.documents else ""
            boundary = int(len(text) * train_fraction)
            return (TextCorpus([text[:boundary]], drop_empty=False),
                    TextCorpus([text[boundary:]], drop_empty=False))
        boundary = max(1, min(len(self.documents) - 1, round(len(self.documents) * train_fraction)))
        return (TextCorpus(self.documents[:boundary], drop_empty=False),
                TextCorpus(self.documents[boundary:], drop_empty=False))

    def __len__(self) -> int:
        return len(self.documents)
