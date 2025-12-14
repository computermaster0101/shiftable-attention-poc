from __future__ import annotations

from collections import Counter
from typing import Dict, List


class SimpleTokenizer:
    """
    Very simple whitespace + lowercasing tokenizer with a trainable vocabulary.

    Special tokens:
        <pad>, <unk>, <bos>, <eos>
    """
    def __init__(
        self,
        stoi: Dict[str, int],
        itos: List[str],
        unk_token: str = "<unk>",
        pad_token: str = "<pad>",
        bos_token: str = "<bos>",
        eos_token: str = "<eos>",
    ) -> None:
        self.stoi = stoi
        self.itos = itos
        self.unk_token = unk_token
        self.pad_token = pad_token
        self.bos_token = bos_token
        self.eos_token = eos_token

        self.vocab_size = len(itos)
        self.pad_id = stoi[pad_token]
        self.unk_id = stoi[unk_token]
        self.bos_id = stoi[bos_token]
        self.eos_id = stoi[eos_token]

    @staticmethod
    def _basic_tokenize(text: str) -> List[str]:
        # Lowercase + whitespace split.
        return text.strip().lower().split()

    @classmethod
    def build_from_files(
        cls,
        paths: List[str],
        min_freq: int = 1,
        max_vocab_size: int | None = None,
    ) -> "SimpleTokenizer":
        counter: Counter[str] = Counter()
        for path in paths:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    tokens = cls._basic_tokenize(line)
                    counter.update(tokens)

        specials = ["<pad>", "<unk>", "<bos>", "<eos>"]
        stoi: Dict[str, int] = {}
        itos: List[str] = []

        # Add specials first
        for sp in specials:
            stoi[sp] = len(itos)
            itos.append(sp)

        # Filter by min_freq and sort by frequency (desc)
        items = [
            (token, freq)
            for token, freq in counter.items()
            if freq >= min_freq and token not in stoi
        ]
        items.sort(key=lambda x: x[1], reverse=True)

        # If we have a max vocab size, truncate
        if max_vocab_size is not None:
            # max number of non-special tokens we can add
            remaining = max_vocab_size - len(itos)
            if remaining < 0:
                remaining = 0
            items = items[:remaining]

        # Add remaining tokens
        for token, freq in items:
            stoi[token] = len(itos)
            itos.append(token)

        print(f"[tokenizer] Built vocab of size {len(itos)} "
              f"(min_freq={min_freq}, max_vocab_size={max_vocab_size})")

        return cls(stoi=stoi, itos=itos)

    def encode(self, text: str, add_specials: bool = True) -> List[int]:
        tokens = self._basic_tokenize(text)
        ids = [self.stoi.get(tok, self.unk_id) for tok in tokens]
        if add_specials:
            return [self.bos_id] + ids + [self.eos_id]
        return ids

    def decode(self, ids: List[int], skip_specials: bool = True) -> str:
        tokens: List[str] = []
        specials = {self.pad_token, self.unk_token, self.bos_token, self.eos_token}
        for idx in ids:
            if idx < 0 or idx >= len(self.itos):
                continue
            tok = self.itos[idx]
            if skip_specials and tok in specials:
                continue
            tokens.append(tok)
        return " ".join(tokens)

    def save(self, path: str) -> None:
        import json

        data = {
            "itos": self.itos,
            "unk_token": self.unk_token,
            "pad_token": self.pad_token,
            "bos_token": self.bos_token,
            "eos_token": self.eos_token,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f)

    @classmethod
    def load(cls, path: str) -> "SimpleTokenizer":
        import json

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        itos: List[str] = data["itos"]
        stoi = {tok: idx for idx, tok in enumerate(itos)}
        return cls(
            stoi=stoi,
            itos=itos,
            unk_token=data["unk_token"],
            pad_token=data["pad_token"],
            bos_token=data["bos_token"],
            eos_token=data["eos_token"],
        )
