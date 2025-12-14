from __future__ import annotations

import argparse
import os
from typing import Dict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .tokenizer import SimpleTokenizer
from .data import LMDataset, list_text_files, load_texts_from_files
from .models import BaseTransformerLM


def lm_loss(logits: torch.Tensor, targets: torch.Tensor, pad_id: int) -> torch.Tensor:
    vocab_size = logits.size(-1)
    loss_fct = nn.CrossEntropyLoss(ignore_index=pad_id)
    return loss_fct(
        logits.view(-1, vocab_size),
        targets.view(-1),
    )


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    pad_id: int,
) -> float:
    model.train()
    total_loss = 0.0
    total_tokens = 0

    for input_ids, target_ids in dataloader:
        input_ids = input_ids.to(device)
        target_ids = target_ids.to(device)

        optimizer.zero_grad()
        logits = model(input_ids)
        loss = lm_loss(logits, target_ids, pad_id=pad_id)
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            non_pad = (target_ids != pad_id).sum().item()
            total_loss += loss.item() * non_pad
            total_tokens += non_pad

    return total_loss / max(1, total_tokens)


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    pad_id: int,
) -> float:
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for input_ids, target_ids in dataloader:
            input_ids = input_ids.to(device)
            target_ids = target_ids.to(device)

            logits = model(input_ids)
            loss = lm_loss(logits, target_ids, pad_id=pad_id)

            non_pad = (target_ids != pad_id).sum().item()
            total_loss += loss.item() * non_pad
            total_tokens += non_pad

    return total_loss / max(1, total_tokens)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train generalist BaseTransformerLM.")
    parser.add_argument("--data_dir", type=str, required=True, help="Directory with general corpus .txt files.")
    parser.add_argument("--output_dir", type=str, required=True, help="Where to save tokenizer and model.")
    parser.add_argument("--seq_len", type=int, default=128)
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--n_heads", type=int, default=8)
    parser.add_argument("--n_layers", type=int, default=4)
    parser.add_argument("--dim_feedforward", type=int, default=1024)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--min_freq", type=int, default=1, help="Minimum token frequency to keep in vocab.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # 1) Build tokenizer from general corpus
    file_paths = list_text_files(args.data_dir)
    if not file_paths:
        raise RuntimeError(f"No .txt files found in {args.data_dir}")

    tokenizer = SimpleTokenizer.build_from_files(file_paths, min_freq=args.min_freq, max_vocab_size=args.max_vocab_size)
    tokenizer_path = os.path.join(args.output_dir, "tokenizer.json")
    tokenizer.save(tokenizer_path)

    # 2) Build dataset and dataloader
    texts = load_texts_from_files(file_paths)
    dataset = LMDataset(texts, tokenizer, seq_len=args.seq_len)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # 3) Initialize model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BaseTransformerLM(
        vocab_size=tokenizer.vocab_size,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        max_seq_len=args.seq_len,
        pad_id=tokenizer.pad_id,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # 4) Training loop
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, dataloader, optimizer, device, pad_id=tokenizer.pad_id)
        val_loss = evaluate(model, dataloader, device, pad_id=tokenizer.pad_id)
        print(f"[Generalist] Epoch {epoch}/{args.epochs} - train loss: {train_loss:.4f}, val loss: {val_loss:.4f}")

    # 5) Save model checkpoint with config
    ckpt_path = os.path.join(args.output_dir, "generalist.pt")
    config: Dict[str, object] = {
        "vocab_size": tokenizer.vocab_size,
        "d_model": args.d_model,
        "n_heads": args.n_heads,
        "n_layers": args.n_layers,
        "dim_feedforward": args.dim_feedforward,
        "dropout": args.dropout,
        "max_seq_len": args.seq_len,
        "pad_id": tokenizer.pad_id,
    }
    torch.save(
        {
            "config": config,
            "model_state_dict": model.state_dict(),
        },
        ckpt_path,
    )
    print(f"Saved generalist model checkpoint to {ckpt_path}")
    print(f"Saved tokenizer to {tokenizer_path}")


if __name__ == "__main__":
    main()
