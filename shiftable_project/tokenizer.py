from __future__ import annotations

"""
Drop-in replacement for the previous whitespace SimpleTokenizer.

Goal:
- Keep the SAME class name + public method signatures so the rest of the repo keeps working.
- Replace closed-vocabulary "word list" tokenization with an open-vocabulary *subword* tokenizer
  inspired by SentencePiece Unigram:
    - Learn a vocabulary of pieces from data
    - Encode by Viterbi (best-score) segmentation
    - Decode by concatenating pieces and restoring spaces

Important notes:
- This implementation is dependency-free (no `sentencepiece` wheel required).
- It is NOT a full reproduction of SentencePiece's EM training, but it matches the
  core property you need for GRCLM: **new domains remain representable** without <unk> collapse.
- The tokenizer artifact is still JSON at TOKENIZER_PATH. `load()` remains backward-compatible
  with the previous format that only stored `itos`.

Special tokens (kept identical):
    <pad>, <unk>, <bos>, <eos>
"""

from collections import Counter
from typing import Dict, List, Tuple, Optional
import json
import math
import re


class SimpleTokenizer:
    """
    Subword tokenizer (Unigram-like) with a trainable vocabulary.

    Public API preserved:
      - build_from_files(paths, min_freq, max_vocab_size)
      - encode(text, add_specials=True)
      - decode(ids, skip_specials=True)
      - save(path)
      - load(path)

    Internals:
      - Text is normalized to a SentencePiece-like form using a "word boundary" marker:
            "hello world" -> "▁hello▁world"
      - Vocabulary consists of variable-length pieces (strings) with log-frequency scores.
      - Encoding uses Viterbi DP to find the best (max score) segmentation.
    """

    # SentencePiece uses U+2581 "LOWER ONE EIGHTH BLOCK" as a word-boundary marker.
    _WB = "▁"

    def __init__(
        self,
        stoi: Dict[str, int],
        itos: List[str],
        scores: Optional[List[float]] = None,
        unk_token: str = "<unk>",
        pad_token: str = "<pad>",
        bos_token: str = "<bos>",
        eos_token: str = "<eos>",
        max_piece_length: int = 12,
    ) -> None:
        self.stoi = stoi
        self.itos = itos
        self.scores = scores if scores is not None else [0.0] * len(itos)

        self.unk_token = unk_token
        self.pad_token = pad_token
        self.bos_token = bos_token
        self.eos_token = eos_token

        self.vocab_size = len(itos)
        self.pad_id = stoi[pad_token]
        self.unk_id = stoi[unk_token]
        self.bos_id = stoi[bos_token]
        self.eos_id = stoi[eos_token]

        self.max_piece_length = max_piece_length

        # Fast lookup for piece scores during encoding
        # (specials are included; they just won't appear in normal text).
        self._piece_score: Dict[str, float] = {p: float(s) for p, s in zip(self.itos, self.scores)}

    # ------------------------------------------------------------------ #
    # Normalization
    # ------------------------------------------------------------------ #
    @classmethod
    def _normalize(cls, text: str) -> str:
        """
        SentencePiece-like normalization:
          - lowercase
          - collapse whitespace
          - insert word-boundary marker (▁) before each whitespace-separated token
        """
        text = text.strip().lower()
        if not text:
            return ""
        # collapse whitespace
        parts = re.split(r"\s+", text)
        return cls._WB + cls._WB.join(parts)

    # Kept for compatibility with older code that might call it.
    @staticmethod
    def _basic_tokenize(text: str) -> List[str]:
        return text.strip().lower().split()

    # ------------------------------------------------------------------ #
    # Training (build vocab)
    # ------------------------------------------------------------------ #
    @classmethod
    def build_from_files(
        cls,
        paths: List[str],
        min_freq: int = 1,
        max_vocab_size: int | None = None,
    ) -> "SimpleTokenizer":
        """
        Build a subword vocabulary from the provided files.

        This is a practical Unigram-style vocabulary builder:
          - normalize text to include "▁" word-boundary marker
          - count character n-grams up to MAX_LEN
          - keep all single characters (including ▁)
          - add the most frequent longer pieces until max_vocab_size is reached
          - score pieces by log(freq)

        NOTE: SentencePiece Unigram uses EM to optimize a probabilistic model.
        Here we approximate by frequency-based piece selection + Viterbi decoding,
        which is sufficient to avoid <unk> collapse and supports emergent domains.
        """
        # How long pieces can be during training.
        # You can bump this later if you want fewer fragments for long technical terms.
        MAX_LEN = 12

        ngram_counts: Counter[str] = Counter()

        # Count ngrams on normalized text.
        for path in paths:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    s = cls._normalize(line)
                    if not s:
                        continue
                    L = len(s)
                    # Sliding window n-grams
                    for i in range(L):
                        # Always count single char (len=1)
                        # And up to MAX_LEN
                        max_j = min(L, i + MAX_LEN)
                        for j in range(i + 1, max_j + 1):
                            ngram_counts[s[i:j]] += 1

        specials = ["<pad>", "<unk>", "<bos>", "<eos>"]

        # Always include all single characters we observed (including ▁),
        # plus a basic ASCII fallback set so unseen-but-ASCII inputs won't collapse.
        single_chars = {p for p in ngram_counts.keys() if len(p) == 1}
        ascii_fallback = {chr(i) for i in range(32, 127)}
        base_chars = sorted(single_chars.union(ascii_fallback).union({cls._WB}))

        # Build candidate piece list (exclude specials).
        # Keep pieces with freq >= min_freq.
        # We'll prioritize longer pieces by frequency to reduce fragmentation.
        items: List[Tuple[str, int]] = [
            (piece, freq)
            for piece, freq in ngram_counts.items()
            if freq >= min_freq and piece not in specials
        ]

        # Sort: higher freq first, then longer pieces first (helps reduce fragmentation),
        # then lexicographically for stability.
        items.sort(key=lambda x: (x[1], len(x[0]), x[0]), reverse=True)

        # Start vocab with specials + base chars
        itos: List[str] = []
        stoi: Dict[str, int] = {}
        scores: List[float] = []

        for sp in specials:
            stoi[sp] = len(itos)
            itos.append(sp)
            scores.append(0.0)

        # Add base chars
        for ch in base_chars:
            if ch in stoi:
                continue
            stoi[ch] = len(itos)
            itos.append(ch)
            # Score by log(freq+1) if we saw it, else a small default
            scores.append(math.log(ngram_counts.get(ch, 1) + 1.0))

        # Determine capacity for additional pieces
        if max_vocab_size is not None:
            remaining = max(0, int(max_vocab_size) - len(itos))
        else:
            # Reasonable default: keep a lot of frequent pieces, but not everything.
            remaining = 20000

        # Add the most frequent multi-char pieces
        added = 0
        for piece, freq in items:
            if added >= remaining:
                break
            if piece in stoi:
                continue
            if len(piece) == 1:
                continue  # already covered by base chars
            stoi[piece] = len(itos)
            itos.append(piece)
            scores.append(math.log(float(freq) + 1.0))
            added += 1

        print(
            f"[tokenizer] Built subword vocab of size {len(itos)} "
            f"(min_freq={min_freq}, max_vocab_size={max_vocab_size}, max_piece_len={MAX_LEN})"
        )

        return cls(
            stoi=stoi,
            itos=itos,
            scores=scores,
            max_piece_length=MAX_LEN,
        )

    # ------------------------------------------------------------------ #
    # Encoding / decoding
    # ------------------------------------------------------------------ #
    def encode(self, text: str, add_specials: bool = True) -> List[int]:
        """
        Encode text into token IDs using Viterbi segmentation.
        """
        s = self._normalize(text)

        if not s:
            ids: List[int] = []
            if add_specials:
                return [self.bos_id, self.eos_id]
            return ids

        # Viterbi DP: dp[i] = best score up to position i (exclusive)
        # back[i] = (prev_index, piece)
        n = len(s)
        neg_inf = -1e30
        dp: List[float] = [neg_inf] * (n + 1)
        back: List[Tuple[int, str] | None] = [None] * (n + 1)
        dp[0] = 0.0

        max_len = self.max_piece_length

        for i in range(n):
            if dp[i] <= neg_inf / 2:
                continue
            # Try pieces starting at i
            end_max = min(n, i + max_len)
            # Prefer longer matches first (often yields better segmentation)
            for j in range(end_max, i, -1):
                piece = s[i:j]
                sc = self._piece_score.get(piece)
                if sc is None:
                    continue
                cand = dp[i] + sc
                if cand > dp[j]:
                    dp[j] = cand
                    back[j] = (i, piece)

        # If we couldn't segment (should be rare because single chars exist), fall back to char-level with <unk> as needed
        if back[n] is None:
            piece_ids = []
            for ch in s:
                piece_ids.append(self.stoi.get(ch, self.unk_id))
        else:
            # Reconstruct best path
            pieces: List[str] = []
            idx = n
            while idx > 0 and back[idx] is not None:
                prev, piece = back[idx]
                pieces.append(piece)
                idx = prev
            pieces.reverse()
            piece_ids = [self.stoi.get(p, self.unk_id) for p in pieces]

        if add_specials:
            return [self.bos_id] + piece_ids + [self.eos_id]
        return piece_ids

    def decode(self, ids: List[int], skip_specials: bool = True) -> str:
        """
        Decode token IDs back into text.
        """
        specials = {self.pad_token, self.unk_token, self.bos_token, self.eos_token}
        pieces: List[str] = []
        for idx in ids:
            if idx < 0 or idx >= len(self.itos):
                continue
            tok = self.itos[idx]
            if skip_specials and tok in specials:
                continue
            pieces.append(tok)

        if not pieces:
            return ""

        s = "".join(pieces)

        # Restore spaces from the word-boundary marker.
        # "▁hello▁world" -> "hello world"
        if self._WB in s:
            # leading ▁ creates a leading space; strip it.
            s = s.replace(self._WB, " ").strip()
        return s

    # ------------------------------------------------------------------ #
    # Persistence (JSON)
    # ------------------------------------------------------------------ #
    def save(self, path: str) -> None:
        data = {
            "type": "unigram_like",
            "itos": self.itos,
            "scores": self.scores,
            "unk_token": self.unk_token,
            "pad_token": self.pad_token,
            "bos_token": self.bos_token,
            "eos_token": self.eos_token,
            "max_piece_length": self.max_piece_length,
            "word_boundary": self._WB,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f)

    @classmethod
    def load(cls, path: str) -> "SimpleTokenizer":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        itos: List[str] = data["itos"]
        stoi = {tok: idx for idx, tok in enumerate(itos)}

        # Backward compatibility: old files only had itos + special token names.
        scores = data.get("scores")
        max_piece_length = int(data.get("max_piece_length", 12))

        tok = cls(
            stoi=stoi,
            itos=itos,
            scores=scores,
            unk_token=data.get("unk_token", "<unk>"),
            pad_token=data.get("pad_token", "<pad>"),
            bos_token=data.get("bos_token", "<bos>"),
            eos_token=data.get("eos_token", "<eos>"),
            max_piece_length=max_piece_length,
        )

        # If a different boundary marker was used, respect it.
        wb = data.get("word_boundary")
        if isinstance(wb, str) and wb:
            tok._WB = wb  # type: ignore[attr-defined]
        return tok
