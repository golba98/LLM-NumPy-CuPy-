"""A small, deterministic byte-level BPE tokenizer implemented with the stdlib."""

from __future__ import annotations

import json
import heapq
import itertools
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple, Union


ByteSymbol = bytes


class ByteLevelBPETokenizer:
    """Lossless UTF-8 byte tokenizer with optional learned BPE merges.

    IDs 0--3 are permanently reserved for special tokens and IDs 4--259
    represent the complete raw-byte fallback vocabulary.
    """

    FORMAT = "byte-level-bpe"
    VERSION = 1
    SPECIAL_TOKENS = {"<pad>": 0, "<bos>": 1, "<eos>": 2, "<unk>": 3}

    def __init__(self) -> None:
        self.special_tokens = dict(self.SPECIAL_TOKENS)
        self.PAD_TOKEN, self.BOS_TOKEN = "<pad>", "<bos>"
        self.EOS_TOKEN, self.UNK_TOKEN = "<eos>", "<unk>"
        self.PAD_ID, self.BOS_ID = 0, 1
        self.EOS_ID, self.UNK_ID = 2, 3
        self.vocab: Dict[int, bytes] = {}
        self.encoder: Dict[bytes, int] = {}
        self.merges: List[Tuple[bytes, bytes]] = []
        self._merge_ranks: Dict[Tuple[int, int], int] = {}
        self._init_base_vocab()

    def _init_base_vocab(self) -> None:
        self.vocab = {
            0: b"<pad>",
            1: b"<bos>",
            2: b"<eos>",
            3: b"<unk>",
            **{i + 4: bytes([i]) for i in range(256)},
        }
        self.encoder = {value: key for key, value in self.vocab.items() if key >= 4}

    @staticmethod
    def _as_documents(texts: Union[str, Iterable[str]]) -> List[str]:
        if isinstance(texts, str):
            return [texts]
        return [str(text) for text in texts]

    def train(self, texts: Union[str, Iterable[str]], vocab_size: int = 512) -> None:
        """Learn deterministic pair merges from one or more text documents."""
        if vocab_size < 260:
            raise ValueError("vocab_size must be at least 260 (4 specials + 256 bytes)")

        sequences = [list(document.encode("utf-8")) for document in self._as_documents(texts)]
        self.merges = []
        self._init_base_vocab()
        # Maintain linked token sequences and an occurrence index.  Updating
        # only pairs adjacent to a merge makes large vocabularies practical
        # without introducing a tokenizer dependency.
        class Node:
            __slots__ = ("value", "previous", "next", "active")

            def __init__(self, value: int):
                self.value = value
                self.previous = None
                self.next = None
                self.active = True

        linked_sequences: List[List[Node]] = []
        pair_nodes: Dict[Tuple[int, int], set[Node]] = {}
        counts: Dict[Tuple[int, int], int] = {}
        heap: List[Tuple[int, bytes, bytes, int, int]] = []

        def add_pair(left, right) -> None:
            if left is None or right is None or not left.active or not right.active:
                return
            key = (left.value, right.value)
            pair_nodes.setdefault(key, set()).add(left)
            counts[key] = counts.get(key, 0) + 1
            heapq.heappush(heap, (-counts[key], self.vocab[key[0]], self.vocab[key[1]], key[0], key[1]))

        def remove_pair(left, right) -> None:
            if left is None or right is None:
                return
            key = (left.value, right.value)
            occurrences = pair_nodes.get(key)
            if occurrences is not None:
                occurrences.discard(left)
            counts[key] = counts.get(key, 0) - 1

        for sequence in sequences:
            nodes = [Node(value + 4) for value in sequence]
            linked_sequences.append(nodes)
            for left, right in zip(nodes, nodes[1:]):
                left.next, right.previous = right, left
                add_pair(left, right)

        while len(self.vocab) < vocab_size and heap:
            _, left_bytes, right_bytes, left_id, right_id = heapq.heappop(heap)
            pair = (left_id, right_id)
            if counts.get(pair, 0) <= 0 or not pair_nodes.get(pair):
                continue
            if left_bytes != self.vocab[left_id] or right_bytes != self.vocab[right_id]:
                continue
            merged = left_bytes + right_bytes
            if merged in self.encoder:
                break
            token_id = len(self.vocab)
            self.vocab[token_id] = merged
            self.encoder[merged] = token_id
            self.merges.append((left_bytes, right_bytes))

            occurrences = list(pair_nodes.get(pair, ()))
            for left in occurrences:
                right = left.next
                if not left.active or right is None or not right.active or (left.value, right.value) != pair:
                    continue
                previous, following = left.previous, right.next
                remove_pair(previous, left)
                remove_pair(left, right)
                remove_pair(right, following)
                left.value = token_id
                left.next = following
                if following is not None:
                    following.previous = left
                right.active = False
                add_pair(previous, left)
                add_pair(left, following)
            pair_nodes.pop(pair, None)
        self._rebuild_merge_index()

    def _rebuild_merge_index(self) -> None:
        self._merge_ranks = {}
        for rank, (left, right) in enumerate(self.merges):
            left_id = self.encoder[left]
            right_id = self.encoder[right]
            self._merge_ranks[(left_id, right_id)] = rank

    def _encode_bytes(self, raw: bytes) -> List[int]:
        """Apply learned merges with a rank-priority heap in near-linear time."""
        if not raw:
            return []

        class Node:
            __slots__ = ("value", "previous", "next", "active")

            def __init__(self, value: int):
                self.value = value
                self.previous = None
                self.next = None
                self.active = True

        nodes = [Node(self.encoder[bytes([value])]) for value in raw]
        heap: List[Tuple[int, int, Node]] = []
        serial = itertools.count()

        def queue_pair(left, right) -> None:
            if left is None or right is None or not left.active or not right.active:
                return
            rank = self._merge_ranks.get((left.value, right.value))
            if rank is not None:
                heapq.heappush(heap, (rank, next(serial), left))

        for left, right in zip(nodes, nodes[1:]):
            left.next, right.previous = right, left
            queue_pair(left, right)

        while heap:
            rank, _, left = heapq.heappop(heap)
            right = left.next
            if not left.active or right is None or not right.active:
                continue
            if self._merge_ranks.get((left.value, right.value)) != rank:
                continue
            merged_bytes = self.vocab[left.value] + self.vocab[right.value]
            merged_id = self.encoder[merged_bytes]
            previous, following = left.previous, right.next
            left.value = merged_id
            left.next = following
            if following is not None:
                following.previous = left
            right.active = False
            queue_pair(previous, left)
            queue_pair(left, following)
        return [node.value for node in nodes if node.active]

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> List[int]:
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        result = [self.BOS_ID] if add_bos else []
        result.extend(self._encode_bytes(text.encode("utf-8")))
        if add_eos:
            result.append(self.EOS_ID)
        return result

    def decode(self, token_ids: Sequence[int], skip_special_tokens: bool = True) -> str:
        special_ids = set(self.special_tokens.values())
        chunks: List[bytes] = []
        for token_id in token_ids:
            token_id = int(token_id)
            if token_id in special_ids:
                if skip_special_tokens:
                    continue
                continue  # special tokens are control symbols, not UTF-8 text
            if token_id not in self.vocab:
                raise ValueError(f"Unknown token ID: {token_id}")
            chunks.append(self.vocab[token_id])
        # A model may sample an arbitrary byte sequence before it has learned
        # valid UTF-8 boundaries; replacement keeps diagnostics printable.
        return b"".join(chunks).decode("utf-8", errors="replace")

    def save(self, filepath: Union[str, Path]) -> None:
        payload = {
            "format": self.FORMAT,
            "version": self.VERSION,
            "special_tokens": self.special_tokens,
            "vocab": {str(token_id): value.hex() for token_id, value in self.vocab.items()},
            "merges": [[left.hex(), right.hex()] for left, right in self.merges],
        }
        with open(filepath, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)

    def load(self, filepath: Union[str, Path]) -> "ByteLevelBPETokenizer":
        with open(filepath, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if payload.get("format") != self.FORMAT:
            raise ValueError("Unsupported tokenizer format")
        if payload.get("special_tokens") != self.SPECIAL_TOKENS:
            raise ValueError("Tokenizer special-token IDs are incompatible")
        self.special_tokens = dict(payload["special_tokens"])
        self.vocab = {int(key): bytes.fromhex(value) for key, value in payload["vocab"].items()}
        self.encoder = {value: key for key, value in self.vocab.items()}
        self.merges = [(bytes.fromhex(left), bytes.fromhex(right)) for left, right in payload["merges"]]
        self._rebuild_merge_index()
        return self

    @classmethod
    def from_file(cls, filepath: Union[str, Path]) -> "ByteLevelBPETokenizer":
        return cls().load(filepath)

    def analyze_corpus(self, texts: Union[str, Iterable[str]]) -> Dict[str, Union[int, float]]:
        documents = self._as_documents(texts)
        text = "".join(documents)
        byte_count = len(text.encode("utf-8"))
        token_count = sum(len(self.encode(doc)) for doc in documents)
        return {
            "vocab_size": len(self.vocab),
            "special_token_count": len(self.special_tokens),
            "base_byte_count": 256,
            "bpe_merge_count": len(self.merges),
            "character_count": len(text),
            "utf8_byte_count": byte_count,
            "byte_count": byte_count,
            "token_count": token_count,
            "chars_per_token": len(text) / max(token_count, 1),
            "characters_per_token": len(text) / max(token_count, 1),
            "bytes_per_token": byte_count / max(token_count, 1),
            "compression_ratio": byte_count / max(token_count, 1),
            "unknown_token_count": sum(self.encode(doc).count(self.UNK_ID) for doc in documents),
            "unknown_frequency": sum(self.encode(doc).count(self.UNK_ID) for doc in documents) / max(token_count, 1),
        }


ByteBPETokenizer = ByteLevelBPETokenizer
