from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import torch
from torch.utils.data import Dataset

from .tokenizer import SimpleTokenizer

def list_text_files(data_dir: str) -> List[str]:
    """
    Recursively list all .txt files under data_dir.
    """
    root = Path(data_dir)
    paths: List[str] = []
    for p in root.rglob("*.txt"):
        if p.is_file():
            paths.append(str(p))
    return paths


def load_texts_from_files(paths: List[str]) -> List[str]:
    """
    Load entire contents of each file as a single text example.
    """
    texts: List[str] = []
    for path in paths:
        with open(path, "r", encoding="utf-8") as f:
            texts.append(f.read())
    return texts


class LMDataset(Dataset):
    """
    Next-token prediction dataset.

    For each input sequence of length seq_len, the target is the same sequence
    shifted by one token.
    """
    def __init__(self, texts: List[str], tokenizer, seq_len: int) -> None:

        if not isinstance(tokenizer, SimpleTokenizer):
            raise TypeError("tokenizer must be a SimpleTokenizer")

        self.tokenizer = tokenizer
        self.seq_len = seq_len

        self.samples: List[Tuple[torch.Tensor, torch.Tensor]] = []

        for text in texts:
            token_ids = tokenizer.encode(text)
            if len(token_ids) < 2:
                continue

            start = 0
            while start + seq_len < len(token_ids):
                inp = token_ids[start:start + seq_len]
                tgt = token_ids[start + 1:start + seq_len + 1]
                self.samples.append(
                    (
                        torch.tensor(inp, dtype=torch.long),
                        torch.tensor(tgt, dtype=torch.long),
                    )
                )
                # Non-overlapping chunks; change to smaller step if you want more data
                start += seq_len

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        return self.samples[idx]
