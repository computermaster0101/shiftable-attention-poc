from __future__ import annotations

import argparse
import os
from typing import Dict, List

import torch
import torch.nn as nn
from torch.utils.data import ConcatDataset, DataLoader

from .tokenizer import SimpleTokenizer
from .data import LMDataset, list_text_files, load_texts_from_files
from .models import BaseTransformerLM, ShiftableTransformerLM


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


def build_domain_dataset(
    data_dir: str,
    tokenizer: SimpleTokenizer,
    seq_len: int,
) -> LMDataset:
    file_paths = list_text_files(data_dir)
    if not file_paths:
        raise RuntimeError(f"No .txt files found in {data_dir}")
    texts = load_texts_from_files(file_paths)
    return LMDataset(texts, tokenizer, seq_len=seq_len)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train specialist heads on top of a generalist model.")
    parser.add_argument("--tokenizer_path", type=str, required=True, help="Path to tokenizer.json from generalist training.")
    parser.add_argument("--generalist_ckpt", type=str, required=True, help="Path to generalist.pt checkpoint.")
    parser.add_argument("--general_dir", type=str, required=False, help="Optional: general corpus to mix in during specialist training.")
    parser.add_argument("--datascience_dir", type=str, required=True, help="Directory with datascience specialist corpus (.txt).")
    parser.add_argument("--business_dir", type=str, required=True, help="Directory with businessadvisor specialist corpus (.txt).")
    parser.add_argument("--seq_len", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--output_dir", type=str, required=True, help="Where to save the shiftable model.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # 1) Load tokenizer
    tokenizer = SimpleTokenizer.load(args.tokenizer_path)

    # 2) Load generalist base model
    ckpt = torch.load(args.generalist_ckpt, map_location="cpu")
    config: Dict[str, object] = ckpt["config"]
    base_model = BaseTransformerLM(
        vocab_size=int(config["vocab_size"]),
        d_model=int(config["d_model"]),
        n_heads=int(config["n_heads"]),
        n_layers=int(config["n_layers"]),
        dim_feedforward=int(config["dim_feedforward"]),
        dropout=float(config["dropout"]),
        max_seq_len=int(config["max_seq_len"]),
        pad_id=int(config["pad_id"]),
    )
    base_model.load_state_dict(ckpt["model_state_dict"])

    # 3) Build shiftable LM with two specialists: datascience and businessadvisor
    specialist_names: List[str] = ["datascience", "businessadvisor"]
    shift_model = ShiftableTransformerLM.from_base_model(
        base_model,
        num_specialists=2,
        specialist_names=specialist_names,
        use_cls_token_pool=False,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    shift_model = shift_model.to(device)

    # 4) Optionally freeze embeddings + LM head if you want *only* specialist attention to update.
    # Comment these lines out if you want the embeddings/head to keep adapting as well.
    for name, param in shift_model.named_parameters():
        if name.startswith("tok_emb") or name.startswith("pos_emb") or name.startswith("lm_head"):
            param.requires_grad = False

    # 5) Build datasets
    datasets = []

    if args.general_dir:
        print("Including general corpus during specialist training.")
        general_ds = build_domain_dataset(args.general_dir, tokenizer, seq_len=args.seq_len)
        datasets.append(general_ds)

    datascience_ds = build_domain_dataset(args.datascience_dir, tokenizer, seq_len=args.seq_len)
    datasets.append(datascience_ds)

    business_ds = build_domain_dataset(args.business_dir, tokenizer, seq_len=args.seq_len)
    datasets.append(business_ds)

    train_dataset = ConcatDataset(datasets)
    dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, shift_model.parameters()), lr=args.lr)

    # 6) Train specialists + gate
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(shift_model, dataloader, optimizer, device, pad_id=tokenizer.pad_id)
        val_loss = evaluate(shift_model, dataloader, device, pad_id=tokenizer.pad_id)
        print(f"[Specialists] Epoch {epoch}/{args.epochs} - train loss: {train_loss:.4f}, val loss: {val_loss:.4f}")

    # 7) Save shiftable model checkpoint
    ckpt_path = os.path.join(args.output_dir, "shiftable_specialists.pt")
    torch.save(
        {
            "config": {
                "vocab_size": tokenizer.vocab_size,
                "d_model": base_model.d_model,
                "n_heads": base_model.n_heads,
                "n_layers": base_model.n_layers,
                "dim_feedforward": base_model.dim_feedforward,
                "dropout": base_model.dropout,
                "max_seq_len": base_model.max_seq_len,
                "pad_id": tokenizer.pad_id,
                "num_specialists": 2,
                "specialist_names": specialist_names,
            },
            "model_state_dict": shift_model.state_dict(),
        },
        ckpt_path,
    )
    print(f"Saved shiftable specialist model checkpoint to {ckpt_path}")


if __name__ == "__main__":
    main()
