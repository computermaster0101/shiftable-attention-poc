from __future__ import annotations

import json
import time
import os
import logging
import math
import shutil
import threading
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import ConcatDataset, DataLoader

from . import config
from .router import DomainRouter, RoutingResult

# Ensure shiftable_project is importable
from shiftable_project import (
    SimpleTokenizer,
    LMDataset,
    list_text_files,
    load_texts_from_files,
    BaseTransformerLM,
    ShiftableTransformerLM,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def _lm_loss(logits: torch.Tensor, targets: torch.Tensor, pad_id: int) -> torch.Tensor:
    vocab_size = logits.size(-1)
    loss_fct = nn.CrossEntropyLoss(ignore_index=pad_id)
    return loss_fct(
        logits.view(-1, vocab_size),
        targets.view(-1),
    )


def _train_one_epoch(
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
        loss = _lm_loss(logits, target_ids, pad_id=pad_id)
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            non_pad = (target_ids != pad_id).sum().item()
            total_loss += loss.item() * non_pad
            total_tokens += non_pad

    return total_loss / max(1, total_tokens)


def _evaluate(
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
            loss = _lm_loss(logits, target_ids, pad_id=pad_id)

            non_pad = (target_ids != pad_id).sum().item()
            total_loss += loss.item() * non_pad
            total_tokens += non_pad

    return total_loss / max(1, total_tokens)


def _discover_specialist_names() -> List[str]:
    ignore = {"general", "shiftable", "outputs", "__pycache__"}
    names = []
    data_root = config.DATA_ROOT
    if not data_root.exists():
        return names

    for child in data_root.iterdir():
        if not child.is_dir():
            continue
        if child.name in ignore or child.name.startswith("."):
            continue
        # only treat as specialist if it actually has txt files
        if not list_text_files(str(child)):
            continue
        names.append(child.name)

    names.sort()
    return names


def _build_lm_dataset_from_dir(
    data_dir: Path,
    tokenizer: SimpleTokenizer,
    seq_len: int,
) -> LMDataset:
    if not data_dir.exists():
        raise RuntimeError(f"Data directory does not exist: {data_dir}")
    file_paths = list_text_files(str(data_dir))
    if not file_paths:
        raise RuntimeError(f"No .txt files found in {data_dir}")
    texts = load_texts_from_files(file_paths)
    return LMDataset(texts, tokenizer, seq_len)


class ModelManager:
    """
    Manages training and inference for the generalist + shiftable specialist model.

    - On first initialization:
      - Trains a generalist model if needed.
      - Trains a shiftable model for all discovered specialists.
      - Builds domain statistics (centroid + diagonal covariance) for routing.
    - When specialists change:
      - Rebuilds shiftable model and domain stats.

    Also integrates:
      - DomainRouter for geometric routing over domains.
        [primary testing for this phase of the prototype]
      - Automatic emergent expert creation: unknown-domain queries are collected
        into emergent corpora, and once enough samples are gathered, a new
        specialist is registered and the shiftable model is retrained. 
        [this is placeholder logic and not correct. Agentic or Human-in-the-loop specialist creation required]
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._initialized = False

        self.device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer: Optional[SimpleTokenizer] = None
        self.shift_model: Optional[ShiftableTransformerLM] = None
        self.base_model: Optional[BaseTransformerLM] = None
        self.specialist_names: List[str] = []

        # Geometric router (uses DOMAIN_STATS_PATH)
        self.router = DomainRouter(config.DOMAIN_STATS_PATH)

        # In-memory buffer of unknown-domain embeddings used to decide when
        # to spawn new emergent specialists based on the geometric clustering
        # condition (Definition 4 in the GRCLM paper).
        # [this is placeholder logic and not correct. Agentic or Human-in-the-loop specialist creation required]
        self._emergent_buffer: List[torch.Tensor] = []

    # ------------------------------------------------------------------ #
    # Initialization pipeline
    # ------------------------------------------------------------------ #

    def ensure_initialized(self) -> None:
        """
        Runs the initialization pipeline on first call.
        Safe to call multiple times.
        """
        with self._lock:
            if self._initialized:
                return

            logger.info("Starting model initialization.")

            config.GENERAL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            config.SHIFTABLE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

            if not config.TOKENIZER_PATH.exists() or not config.GENERALIST_CKPT_PATH.exists():
                logger.info("Generalist checkpoint or tokenizer not found. Training generalist from scratch.")
                self._train_generalist()
            else:
                logger.info("Found existing generalist checkpoint and tokenizer.")

            # Shiftable artifacts can be either legacy monolithic (shiftable.pt)
            # or modular (shiftable_base.pt + specialists/*.pt)
            if not config.SHIFTABLE_BASE_PATH.exists():
                if config.SHIFTABLE_CKPT_PATH.exists():
                    logger.info("Found legacy shiftable checkpoint. Migrating to modular artifacts.")
                    self._migrate_monolithic_shiftable_to_split()
                else:
                    logger.info("Shiftable artifacts not found. Training shiftable model for all specialists (then exporting modular artifacts).")
                    self._train_shiftable_for_all_specialists()
            else:
                logger.info("Found existing modular shiftable artifacts. Loading model.")
            self._load_tokenizer_and_shiftable_model()
            self.router.reload()
            self._initialized = True
            logger.info("Model initialization complete.")

    def _train_generalist(self) -> None:
        """
        Train the BaseTransformerLM on the general corpus.
        """
        general_dir = config.GENERAL_DATA_DIR
        if not general_dir.exists():
            raise RuntimeError(f"General data directory does not exist: {general_dir}")

        # Build tokenizer from general corpus
        file_paths = list_text_files(str(general_dir))
        if not file_paths:
            raise RuntimeError(f"No .txt files found in general data directory: {general_dir}")

        logger.info("Building tokenizer from general corpus.")
        tokenizer = SimpleTokenizer.build_from_files(
            file_paths,
            min_freq=config.GENERAL_MIN_FREQ,
            max_vocab_size=config.GENERAL_MAX_VOCAB_SIZE,
            )
        logger.info(
            "Tokenizer built. vocab_size=%d (min_freq=%d, max_vocab_size=%d)",
            tokenizer.vocab_size,
            config.GENERAL_MIN_FREQ,
            getattr(config, "GENERAL_MAX_VOCAB_SIZE", 50_000),
        )

        config.TOKENIZER_PATH.parent.mkdir(parents=True, exist_ok=True)
        tokenizer.save(str(config.TOKENIZER_PATH))
        logger.info("Saved tokenizer to %s", config.TOKENIZER_PATH)

        # Build dataset/dataloader
        texts = load_texts_from_files(file_paths)
        dataset = LMDataset(texts, tokenizer, seq_len=config.MAX_SEQ_LEN)
        dataloader = DataLoader(
            dataset, 
            batch_size=config.GENERAL_BATCH_SIZE, 
            shuffle=True,
            )

        # Initialize model
        model = BaseTransformerLM(
            vocab_size=tokenizer.vocab_size,
            d_model=config.D_MODEL,
            n_heads=config.N_HEADS,
            n_layers=config.N_LAYERS,
            dim_feedforward=config.DIM_FEEDFORWARD,
            dropout=config.DROPOUT,
            max_seq_len=config.MAX_SEQ_LEN,
            pad_id=tokenizer.pad_id,
        ).to(self.device)

        optimizer = torch.optim.Adam(model.parameters(), lr=config.GENERAL_LR)

        # Training loop
        logger.info("Beginning training loop")
        for epoch in range(1, config.GENERAL_EPOCHS + 1):
            start = time.time()
            train_loss = _train_one_epoch(model, dataloader, optimizer, self.device, pad_id=tokenizer.pad_id)
            # Commented out val_loss calculation to speed up training
            val_loss = _evaluate(model, dataloader, self.device, pad_id=tokenizer.pad_id)
            # val_loss = 0.0
            epoch_time = time.time() - start
            logger.info(
                "[Generalist] Epoch %d/%d - train loss: %.4f, val loss: %.4f, Duration: %d seconds",
                epoch,
                config.GENERAL_EPOCHS,
                train_loss,
                val_loss,
                epoch_time
            )

        ckpt = {
            "config": {
                "vocab_size": tokenizer.vocab_size,
                "d_model": config.D_MODEL,
                "n_heads": config.N_HEADS,
                "n_layers": config.N_LAYERS,
                "dim_feedforward": config.DIM_FEEDFORWARD,
                "dropout": config.DROPOUT,
                "max_seq_len": config.MAX_SEQ_LEN,
                "pad_id": tokenizer.pad_id,
            },
            "model_state_dict": model.state_dict(),
        }
        torch.save(ckpt, config.GENERALIST_CKPT_PATH)
        logger.info("Saved generalist checkpoint to %s", config.GENERALIST_CKPT_PATH)

    def _load_generalist_model(self) -> Tuple[BaseTransformerLM, Dict[str, object]]:
        """
        Load the generalist model and its config from disk.
        """
        if not config.GENERALIST_CKPT_PATH.exists():
            raise RuntimeError("Generalist checkpoint not found; cannot load generalist model.")

        ckpt = torch.load(config.GENERALIST_CKPT_PATH, map_location="cpu")
        cfg: Dict[str, object] = ckpt["config"]

        model = BaseTransformerLM(
            vocab_size=int(cfg["vocab_size"]),
            d_model=int(cfg["d_model"]),
            n_heads=int(cfg["n_heads"]),
            n_layers=int(cfg["n_layers"]),
            dim_feedforward=int(cfg["dim_feedforward"]),
            dropout=float(cfg["dropout"]),
            max_seq_len=int(cfg["max_seq_len"]),
            pad_id=int(cfg["pad_id"]),
        )
        model.load_state_dict(ckpt["model_state_dict"])
        return model, cfg

    
    def _train_shiftable_for_all_specialists(self) -> None:
        """
        Train ShiftableTransformerLM for ALL discovered specialists.
        This is used both at first initialization and whenever the specialist set changes.

        Key behaviors (v04 goals):
        - Specialist training is "light": we DO NOT include the general corpus by default.
          (You may override by defining config.SHIFTABLE_INCLUDE_GENERAL=True.)
        - We freeze the entire trunk and only train specialist parameters (+ gates / blend_gate).
        - After training, we export modular artifacts:
            outputs/shiftable/shiftable_base.pt
            outputs/shiftable/specialists/<name>.pt  (one per specialist)
        """
        if not config.TOKENIZER_PATH.exists():
            raise RuntimeError("Tokenizer not found; cannot train shiftable model.")

        tokenizer = SimpleTokenizer.load(str(config.TOKENIZER_PATH))
        base_model, base_cfg = self._load_generalist_model()

        specialist_names = _discover_specialist_names()
        num_specialists = len(specialist_names)

        if not specialist_names:
            logger.warning("No specialist directories found. Training shiftable model with 0 specialists.")
        else:
            logger.info("Training shiftable model for specialists: %s", ", ".join(specialist_names))

        shift_model = ShiftableTransformerLM.from_base_model(
            base_model,
            num_specialists=num_specialists,
            specialist_names=specialist_names if specialist_names else [],
            use_cls_token_pool=config.USE_CLS_TOKEN_POOL,
        ).to(self.device)

        # ------------------------------------------------------------------
        # Freeze everything, then unfreeze only specialist-specific params
        # and the gates that control specialist participation.
        # ------------------------------------------------------------------
        for _, p in shift_model.named_parameters():
            p.requires_grad = False

        for name, p in shift_model.named_parameters():
            lname = name.lower()
            if (
                "spec_" in lname
                or ".spec_mhas." in lname
                or "specialist" in lname
                or "gate" in lname   # DomainGate MLPs (including fc2 rows for specialists)
                or "blend_gate" in lname
            ):
                p.requires_grad = True

        # Sanity log
        try:
            trainable = [(n, p.numel()) for n, p in shift_model.named_parameters() if p.requires_grad]
            total = sum(c for _, c in trainable)
            logger.info("Shiftable trainable tensors=%d, params=%d", len(trainable), total)
            for n, c in trainable[:25]:
                logger.info("  trainable: %s (%d)", n, c)
        except Exception:
            pass

        # ------------------------------------------------------------------
        # Build datasets: specialists only by default.
        # ------------------------------------------------------------------
        datasets = []

        include_general = bool(getattr(config, "SHIFTABLE_INCLUDE_GENERAL", False))
        if include_general and config.GENERAL_DATA_DIR.exists():
            logger.info("Including general corpus in shiftable training (SHIFTABLE_INCLUDE_GENERAL=True).")
            general_ds = _build_lm_dataset_from_dir(
                config.GENERAL_DATA_DIR,
                tokenizer,
                seq_len=config.MAX_SEQ_LEN,
            )
            datasets.append(general_ds)
        else:
            logger.info("Skipping general corpus for shiftable training (specialist-only pass).")

        for spec_name in specialist_names:
            spec_dir = config.DATA_ROOT / spec_name
            logger.info("Adding specialist dataset from %s", spec_dir)
            spec_ds = _build_lm_dataset_from_dir(spec_dir, tokenizer, seq_len=config.MAX_SEQ_LEN)
            datasets.append(spec_ds)

        if not datasets:
            raise RuntimeError("No datasets found for shiftable training (no specialists and general excluded).")

        train_dataset = ConcatDataset(datasets)
        dataloader = DataLoader(
            train_dataset,
            batch_size=config.SHIFTABLE_BATCH_SIZE,
            shuffle=True,
        )

        optimizer = torch.optim.Adam(
            [p for p in shift_model.parameters() if p.requires_grad],
            lr=config.SHIFTABLE_LR,
        )

        logger.info("Beginning training loop")
        for epoch in range(1, config.SHIFTABLE_EPOCHS + 1):
            start = time.time()
            train_loss = _train_one_epoch(shift_model, dataloader, optimizer, self.device, pad_id=tokenizer.pad_id)
            # Commented out val_loss calculation to speed up training
            val_loss = _evaluate(shift_model, dataloader, self.device, pad_id=tokenizer.pad_id)
            # val_loss = 0.0
            epoch_time = time.time() - start
            logger.info(
                "[Shiftable] Epoch %d/%d - train loss: %.4f, val loss: %.4f Duration: %d seconds",
                epoch,
                config.SHIFTABLE_EPOCHS,
                train_loss,
                val_loss,
                epoch_time
            )

        # Save legacy monolithic artifact (kept for compatibility / debugging)
        ckpt = {
            "config": {
                "vocab_size": tokenizer.vocab_size,
                "d_model": int(base_cfg["d_model"]),
                "n_heads": int(base_cfg["n_heads"]),
                "n_layers": int(base_cfg["n_layers"]),
                "dim_feedforward": int(base_cfg["dim_feedforward"]),
                "dropout": float(base_cfg["dropout"]),
                "max_seq_len": int(base_cfg["max_seq_len"]),
                "pad_id": tokenizer.pad_id,
                "num_specialists": len(specialist_names),
                "specialist_names": specialist_names,
            },
            "model_state_dict": shift_model.state_dict(),
        }

        config.SHIFTABLE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        torch.save(ckpt, config.SHIFTABLE_CKPT_PATH)
        logger.info("Saved legacy shiftable checkpoint to %s", config.SHIFTABLE_CKPT_PATH)

        # Export split artifacts (base + one .pt per specialist)
        self._save_split_shiftable_artifacts(
            shift_model=shift_model,
            tokenizer=tokenizer,
            specialist_names=specialist_names,
            source_cfg=ckpt.get("config"),
        )

        # Build / refresh domain statistics for routing (general + specialists)
        domain_names = ["general"] + specialist_names
        self._build_domain_stats(tokenizer, base_model, domain_names)
        self.router.reload()

    # ------------------------------------------------------------------ #
    # Split-artifact specialist lifecycle (base + per-specialist)
    # ------------------------------------------------------------------ #

    def _shiftable_split_artifacts_exist(self) -> bool:
        """Return True if the split-artifact layout exists on disk."""
        if not config.SHIFTABLE_BASE_PATH.exists():
            return False
        # specialists dir may legitimately be empty (0 specialists)
        return True

    def _list_specialists_on_disk(self) -> List[str]:
        """
        Source of truth for specialists at runtime.

        We treat each file in outputs/shiftable/specialists/*.pt as a plugin specialist.
        The stem of the filename is the specialist name.
        """
        if not config.SPECIALISTS_DIR.exists():
            return []
        names: List[str] = []
        for p in sorted(config.SPECIALISTS_DIR.glob("*.pt")):
            if p.name.startswith("."):
                continue
            names.append(p.stem)
        return names

    def _migrate_monolithic_shiftable_to_split(self) -> None:
        """
        One-time migration: outputs/shiftable/shiftable.pt -> shiftable_base.pt + specialists/<name>.pt.

        This lets you keep your existing trained model but unlock add/remove specialists
        by simply adding/removing specialist plugin files.
        """
        if not config.SHIFTABLE_CKPT_PATH.exists():
            raise RuntimeError("Legacy shiftable checkpoint not found; cannot migrate.")

        tokenizer = SimpleTokenizer.load(str(config.TOKENIZER_PATH))
        ckpt = torch.load(config.SHIFTABLE_CKPT_PATH, map_location="cpu")
        cfg: Dict[str, object] = ckpt["config"]

        base_model, _ = self._load_generalist_model()
        base_model.eval()

        specialist_names = list(cfg.get("specialist_names", []))
        shift_model = ShiftableTransformerLM.from_base_model(
            base_model,
            num_specialists=int(cfg.get("num_specialists", len(specialist_names))),
            specialist_names=specialist_names,
            use_cls_token_pool=False,
        )
        shift_model.load_state_dict(ckpt["model_state_dict"])
        shift_model.eval()

        self._save_split_shiftable_artifacts(
            shift_model=shift_model,
            tokenizer=tokenizer,
            specialist_names=specialist_names,
            source_cfg=cfg,
        )

    def _save_split_shiftable_artifacts(
        self,
        shift_model: ShiftableTransformerLM,
        tokenizer: SimpleTokenizer,
        specialist_names: List[str],
        source_cfg: Optional[Dict[str, object]] = None,
    ) -> None:
        """
        Save:
          - shiftable_base.pt (all size-stable params)
          - specialists/<name>.pt (params specific to each specialist)
        """
        config.SHIFTABLE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        config.SPECIALISTS_DIR.mkdir(parents=True, exist_ok=True)

        # 1) Build base state dict by removing size-dependent keys
        full_sd = {k: v.detach().cpu() for k, v in shift_model.state_dict().items()}

        # Capture the base (domain=0) rows for all gates, then remove fc2 + spec modules from base_sd
        gate_base_rows = {
            "blend_gate": {
                "weight0": shift_model.blend_gate.fc2.weight[0].detach().cpu(),
                "bias0": shift_model.blend_gate.fc2.bias[0].detach().cpu(),
            },
            "blocks": [],
        }
        for layer_idx in range(shift_model.n_layers):
            gate = shift_model.blocks[layer_idx].self_attn.gate
            gate_base_rows["blocks"].append(
                {
                    "weight0": gate.fc2.weight[0].detach().cpu(),
                    "bias0": gate.fc2.bias[0].detach().cpu(),
                }
            )

        def is_dynamic_key(key: str) -> bool:
            # Specialist MHA parameters
            if ".self_attn.spec_mhas." in key:
                return True
            # Gate fc2 depends on num_domains (=1+num_specialists)
            if key.endswith(".self_attn.gate.fc2.weight") or key.endswith(".self_attn.gate.fc2.bias"):
                return True
            if key.endswith("blend_gate.fc2.weight") or key.endswith("blend_gate.fc2.bias"):
                return True
            return False

        base_sd = {k: v for k, v in full_sd.items() if not is_dynamic_key(k)}

        base_cfg: Dict[str, object] = {
            # Keep what you need to sanity-check compatibility at load time
            "vocab_size": shift_model.vocab_size,
            "d_model": shift_model.d_model,
            "n_heads": shift_model.n_heads,
            "n_layers": shift_model.n_layers,
            "dim_feedforward": shift_model.dim_feedforward,
            "dropout": shift_model.dropout,
            "max_seq_len": shift_model.max_seq_len,
            "pad_id": shift_model.pad_id,
        }
        if source_cfg:
            # preserve any additional training metadata you already had
            for k in ("generalist_ckpt", "trained_at", "notes"):
                if k in source_cfg:
                    base_cfg[k] = source_cfg[k]

        base_ckpt = {
            "config": base_cfg,
            "model_state_dict": base_sd,
            "gate_base_rows": gate_base_rows,
        }
        torch.save(base_ckpt, config.SHIFTABLE_BASE_PATH)
        logger.info("Saved split shiftable base to %s", config.SHIFTABLE_BASE_PATH)

        # 2) Save each specialist plugin
        for idx, name in enumerate(specialist_names):
            spec_payload = self._extract_specialist_payload(shift_model, idx=idx, name=name)
            spec_path = config.SPECIALISTS_DIR / f"{name}.pt"
            torch.save(spec_payload, spec_path)
            logger.info("Saved specialist '%s' to %s", name, spec_path)

    def _extract_specialist_payload(self, shift_model: ShiftableTransformerLM, idx: int, name: str) -> Dict[str, object]:
        """Extract just the parameters that belong to ONE specialist (by index)."""
        mha_layers: List[Dict[str, torch.Tensor]] = []
        gate_rows: List[Dict[str, torch.Tensor]] = []
        for layer_idx in range(shift_model.n_layers):
            spec_mha = shift_model.blocks[layer_idx].self_attn.spec_mhas[idx]
            mha_layers.append({k: v.detach().cpu() for k, v in spec_mha.state_dict().items()})

            gate = shift_model.blocks[layer_idx].self_attn.gate.fc2
            gate_rows.append(
                {
                    "weight": gate.weight[idx + 1].detach().cpu(),
                    "bias": gate.bias[idx + 1].detach().cpu(),
                }
            )

        blend = shift_model.blend_gate.fc2
        payload: Dict[str, object] = {
            "name": name,
            "format_version": 1,
            "n_layers": shift_model.n_layers,
            "d_model": shift_model.d_model,
            "n_heads": shift_model.n_heads,
            "mha_layers": mha_layers,
            "gate_rows": gate_rows,
            "blend_gate_row": {
                "weight": blend.weight[idx + 1].detach().cpu(),
                "bias": blend.bias[idx + 1].detach().cpu(),
            },
        }
        return payload

    def _apply_base_gate_rows(self, shift_model: ShiftableTransformerLM, gate_rows: Dict[str, object]) -> None:
        """Apply the stored domain-0 rows for all gates (blend_gate + per-block SMA gates)."""
        try:
            bg = gate_rows.get("blend_gate", {})
            if bg:
                shift_model.blend_gate.fc2.weight.data[0].copy_(bg["weight0"])
                shift_model.blend_gate.fc2.bias.data[0].copy_(bg["bias0"])

            blocks = gate_rows.get("blocks", [])
            for layer_idx in range(min(len(blocks), shift_model.n_layers)):
                row = blocks[layer_idx]
                gate = shift_model.blocks[layer_idx].self_attn.gate.fc2
                gate.weight.data[0].copy_(row["weight0"])
                gate.bias.data[0].copy_(row["bias0"])
        except Exception:
            logger.exception("Failed applying base gate rows; continuing with initialized weights.")

    def _load_one_specialist_into_model(self, shift_model: ShiftableTransformerLM, idx: int, name: str) -> None:
        """Load a single specialist plugin into the live model at index idx."""
        spec_path = config.SPECIALISTS_DIR / f"{name}.pt"
        if not spec_path.exists():
            raise RuntimeError(f"Specialist file missing: {spec_path}")

        payload = torch.load(spec_path, map_location="cpu")
        if int(payload.get("format_version", 0)) != 1:
            raise RuntimeError(f"Unsupported specialist format for {name}: {payload.get('format_version')}")
        if int(payload.get("n_layers", -1)) != int(shift_model.n_layers):
            raise RuntimeError(f"Specialist {name} incompatible: n_layers mismatch")

        mha_layers = payload["mha_layers"]
        gate_rows = payload["gate_rows"]
        for layer_idx in range(shift_model.n_layers):
            spec_mha = shift_model.blocks[layer_idx].self_attn.spec_mhas[idx]
            spec_mha.load_state_dict(mha_layers[layer_idx])

            gate = shift_model.blocks[layer_idx].self_attn.gate.fc2
            gate.weight.data[idx + 1].copy_(gate_rows[layer_idx]["weight"])
            gate.bias.data[idx + 1].copy_(gate_rows[layer_idx]["bias"])

        blend = payload["blend_gate_row"]
        shift_model.blend_gate.fc2.weight.data[idx + 1].copy_(blend["weight"])
        shift_model.blend_gate.fc2.bias.data[idx + 1].copy_(blend["bias"])
    def _build_domain_stats(
        self,
        tokenizer: SimpleTokenizer,
        base_model: BaseTransformerLM,
        domain_names: List[str],
    ) -> None:
        """
        Build per-domain statistics (centroid + covariance) in TRUNK embedding space.

        Before this patch, stats were built from raw token embeddings (tok_emb),
        which is (a) not the trunk and (b) mismatched against the query embedding
        space in inference. GRCLM assumes a fixed embedding model E(x) for routing.

        We now compute sentence embeddings q = E(x) using the generalist trunk
        encoder (BaseTransformerLM.encode), mean-pooled over non-PAD tokens.

        Output JSON schema remains compatible with DomainRouter:
          - centroid: mean vector
          - var_diag: diagonal of covariance (clamped)
          - cov: full covariance (kept for future upgrades)
          - cov_chol: optional cholesky for stable solves
        """
        base_model = base_model.to(self.device)
        base_model.eval()
        d_model = base_model.d_model

        stats: Dict[str, Dict[str, object]] = {}
        total_domains = 0

        batch_size = int(getattr(config, "ROUTER_STATS_BATCH_SIZE", 32))
        if batch_size <= 0:
            batch_size = 32

        def _encode_text_batch(text_batch: List[str]) -> torch.Tensor:
            # Tokenize and pad to batch max length
            ids_list = [tokenizer.encode(t, add_specials=True) for t in text_batch]
            ids_list = [ids[: config.MAX_SEQ_LEN] for ids in ids_list if ids]
            if not ids_list:
                return torch.empty((0, d_model), device=self.device)

            max_len = max(len(x) for x in ids_list)
            pad = tokenizer.pad_id

            input_ids = torch.full(
                (len(ids_list), max_len),
                pad,
                dtype=torch.long,
                device=self.device,
            )
            attn = torch.zeros(
                (len(ids_list), max_len),
                dtype=torch.long,
                device=self.device,
            )

            for i, ids in enumerate(ids_list):
                n = len(ids)
                input_ids[i, :n] = torch.tensor(ids, dtype=torch.long, device=self.device)
                attn[i, :n] = 1

            return base_model.encode(input_ids, attention_mask=attn)  # [B, D]

        with torch.no_grad():
            for domain in domain_names:
                if domain == "general":
                    data_dir = config.GENERAL_DATA_DIR
                else:
                    data_dir = config.DATA_ROOT / domain

                if not data_dir.exists():
                    logger.warning("Domain '%s' data directory %s does not exist; skipping.", domain, data_dir)
                    continue

                file_paths = list_text_files(str(data_dir))
                if not file_paths:
                    logger.warning("Domain '%s' has no .txt files in %s; skipping.", domain, data_dir)
                    continue

                logger.info("Building TRUNK-space domain stats for '%s' from %d files.", domain, len(file_paths))

                # Now counts *samples* (lines), not tokens
                count_samples = 0
                sum_vec = torch.zeros(d_model, dtype=torch.float64, device=self.device)
                sum_outer = torch.zeros((d_model, d_model), dtype=torch.float64, device=self.device)

                batch: List[str] = []

                def _flush_batch() -> None:
                    nonlocal count_samples, sum_vec, sum_outer, batch
                    if not batch:
                        return
                    embs = _encode_text_batch(batch).double()  # [B,D]
                    if embs.numel() == 0:
                        batch = []
                        return
                    sum_vec += embs.sum(dim=0)
                    sum_outer += embs.T @ embs
                    count_samples += int(embs.shape[0])
                    batch = []

                for path in file_paths:
                    with open(path, "r", encoding="utf-8", errors="ignore") as f:
                        for line in f:
                            text = line.strip()
                            if not text:
                                continue
                            batch.append(text)
                            if len(batch) >= batch_size:
                                _flush_batch()

                _flush_batch()

                if count_samples == 0:
                    logger.warning("Domain '%s' had no valid samples; skipping stats.", domain)
                    continue

                # Mean (centroid)
                mean64 = (sum_vec / count_samples)  # float64
                mean = mean64.float()

                # Covariance with Bessel correction (~ / (N-1))
                denom = max(count_samples - 1, 1)
                ExxT = sum_outer / denom
                mu = mean64.unsqueeze(1)  # [d, 1]
                cov = ExxT - (mu @ mu.T)

                # Regularization term λI (PDF uses + λI for stability)
                lambda_reg = getattr(config, "ROUTER_COV_LAMBDA", 1e-3)
                cov = cov + (lambda_reg * torch.eye(d_model, dtype=torch.float64, device=self.device))

                # Clamp diagonal for numerical stability
                diag = torch.diagonal(cov)
                diag_clamped = torch.clamp(diag, min=config.ROUTER_MIN_VAR)
                cov = cov.clone()
                cov[range(d_model), range(d_model)] = diag_clamped

                try:
                    cov_chol = torch.linalg.cholesky(cov)
                    cov_chol_out = cov_chol.float().cpu().tolist()
                except Exception:
                    cov_chol_out = None

                var_diag_out = diag_clamped.float().cpu().tolist()

                stats[domain] = {
                    "count": int(count_samples),
                    "centroid": mean.cpu().tolist(),
                    "var_diag": var_diag_out,
                    "cov": cov.float().cpu().tolist(),
                    "cov_lambda": float(lambda_reg),
                    "cov_chol": cov_chol_out,
                }

                logger.info(
                    "Domain '%s' stats: samples=%d, mean_norm=%.4f",
                    domain,
                    count_samples,
                    float(mean.norm().item()),
                )
                total_domains += 1

        if total_domains == 0:
            logger.warning("No domain stats were built; DOMAIN_STATS_PATH will not be updated.")
            return

        domain_stats = {
            "dim": d_model,
            "domains": stats,
        }

        config.DOMAIN_STATS_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(config.DOMAIN_STATS_PATH, "w", encoding="utf-8") as f:
            json.dump(domain_stats, f)
        logger.info("Saved domain stats to %s", config.DOMAIN_STATS_PATH)


    def _load_tokenizer_and_shiftable_model(self) -> None:
        """
        Load tokenizer + generalist + shiftable model.

        Preferred: split artifacts
            - shiftable_base.pt
            - specialists/<name>.pt (one per specialist)

        Backward compatible:
            - if legacy shiftable.pt exists and split artifacts do not, migrate once.
        """
        if not config.TOKENIZER_PATH.exists():
            raise RuntimeError("Tokenizer file not found when attempting to load shiftable model.")
        if not config.GENERALIST_CKPT_PATH.exists():
            raise RuntimeError("Generalist checkpoint not found when attempting to load shiftable model.")

        # Ensure split artifacts exist (migrate legacy if needed)
        if not self._shiftable_split_artifacts_exist():
            if config.SHIFTABLE_CKPT_PATH.exists():
                logger.info("Found legacy shiftable.pt. Migrating to split artifacts.")
                self._migrate_monolithic_shiftable_to_split()
            else:
                raise RuntimeError("No shiftable artifacts found. Expected shiftable_base.pt or legacy shiftable.pt.")

        tokenizer = SimpleTokenizer.load(str(config.TOKENIZER_PATH))
        base_model, _ = self._load_generalist_model()
        base_model.eval()

        specialist_names = self._list_specialists_on_disk()
        shift_model = ShiftableTransformerLM.from_base_model(
            base_model=base_model,
            num_specialists=len(specialist_names),
            specialist_names=specialist_names,
            use_cls_token_pool=config.USE_CLS_TOKEN_POOL,
        )

        base_bundle = torch.load(config.SHIFTABLE_BASE_PATH, map_location="cpu")
        base_sd = base_bundle.get("model_state_dict", {})
        gate_rows = base_bundle.get("gate_base_rows", {})

        missing, unexpected = shift_model.load_state_dict(base_sd, strict=False)
        if unexpected:
            logger.warning("Unexpected keys when loading shiftable base: %s", unexpected)
        # Apply domain-0 gate rows (fc2 row 0) for every gate
        self._apply_base_gate_rows(shift_model, gate_rows)

        # Load each specialist plugin by index
        for idx, name in enumerate(specialist_names):
            self._load_one_specialist_into_model(shift_model, idx=idx, name=name)

        shift_model = shift_model.to(self.device)
        shift_model.eval()

        self.tokenizer = tokenizer
        self.base_model = base_model.to(self.device)
        self.base_model.eval()
        self.shift_model = shift_model
        self.specialist_names = specialist_names

        logger.info("Loaded split shiftable artifacts: base=%s, specialists=%d", config.SHIFTABLE_BASE_PATH, len(specialist_names))

    def _load_emergent_state(self) -> Tuple[int, int]:
        """
        Load current emergent expert index and sample count from disk.

        Returns:
            (current_index, current_samples)
        """
        if not config.EMERGENT_STATE_PATH.exists():
            return 1, 0

        try:
            with open(config.EMERGENT_STATE_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            logger.warning("Failed to load emergent state; resetting. Error: %s", e)
            return 1, 0

        idx = int(data.get("current_index", 1))
        samples = int(data.get("current_samples", 0))
        if idx < 1:
            idx = 1
        if samples < 0:
            samples = 0
        return idx, samples

    def _save_emergent_state(self, current_index: int, current_samples: int) -> None:
        """
        Persist current emergent expert index and sample count to disk.
        """
        config.EMERGENT_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "current_index": int(current_index),
            "current_samples": int(current_samples),
        }
        with open(config.EMERGENT_STATE_PATH, "w", encoding="utf-8") as f:
            json.dump(data, f)

    def _log_input_output(
        self,
        prompt: str,
        completion: str,
        query_embedding,
        routing_result,
        domain_prior,
        domain_mask,
    ):
        os.makedirs("logs", exist_ok=True)
        path = os.path.join("logs", "input_output_log.jsonl")

        record = {
            "timestamp": time.time(),
            "prompt": prompt,
            "completion": completion,
            "embedding": query_embedding.tolist() if hasattr(query_embedding, "tolist") else None,
            "routing": None,
            "domain_prior": domain_prior.tolist() if domain_prior is not None else None,
            "domain_mask": domain_mask.tolist() if domain_mask is not None else None,
            "is_unknown_domain": routing_result.is_unknown if routing_result else None,
        }

        if routing_result:
            record["routing"] = {
                "is_unknown": routing_result.is_unknown,
                "metrics": [
                    {
                        "name": m.domain,
                        "similarity": m.similarity,
                        "mahalanobis": m.mahalanobis,
                        "entropy": m.entropy,
                        "support": m.support,
                        "score": m.score,
                    }
                    for m in routing_result.metrics
                ],
            }

        with open(path, "a") as f:
            f.write(json.dumps(record) + "\n")

    def _handle_emergent_query(
        self,
        prompt: str,
        completion: str,
        query_embedding: torch.Tensor,
        routing_result: Optional[RoutingResult],
    ) -> None:
        """
        [This is placeholder logic and not correct. Agentic or Human-in-the-loop specialist creation required]
        Handle a query that appears to be outside existing domain coverage.

        Behavior:
          - Ignore very short prompts.
          - Append prompt+completion to an emergent corpus file for the
            current auto expert (emergent_XXX).
          - Update emergent state (sample count).
          - Also append a JSONL record to EMERGENCE_LOG_PATH for analysis.
          - When enough samples are collected, automatically register a new
            specialist using self.add_specialist(expert_name), then advance
            to the next emergent expert index.
        """
        if len(prompt.strip()) < config.EMERGENT_MIN_PROMPT_LENGTH:
            return

        # Maintain in-memory buffer of unknown-domain embeddings for geometric
        # emergence checks (Definition 4 in the GRCLM paper).
        emb = query_embedding.detach().cpu().view(-1)
        if emb.numel() > 0:
            self._emergent_buffer.append(emb)
            # Keep the buffer bounded in size.
            if len(self._emergent_buffer) > config.EMERGENT_MAX_BUFFER_SIZE:
                overflow = len(self._emergent_buffer) - config.EMERGENT_MAX_BUFFER_SIZE
                if overflow > 0:
                    self._emergent_buffer = self._emergent_buffer[overflow:]

        # Load current emergent state
        current_index, current_samples = self._load_emergent_state()
        expert_name = f"{config.EMERGENT_AUTO_PREFIX}_{current_index:03d}"

        # Append to emergent corpus under shiftable_project/data/emergent_XXX/
        expert_dir = config.DATA_ROOT / expert_name
        expert_dir.mkdir(parents=True, exist_ok=True)
        corpus_path = expert_dir / f"emergent_{expert_name}.txt"

        block = (
            "### PROMPT ###\n"
            f"{prompt}\n\n"
            "### COMPLETION ###\n"
            f"{completion}\n\n"
        )

        with open(corpus_path, "a", encoding="utf-8") as f:
            f.write(block)

        current_samples += 1
        self._save_emergent_state(current_index, current_samples)

        # Also persist a JSONL record for analysis / tooling compatibility
        try:
            record: Dict[str, object] = {
                "prompt": prompt,
                "completion": completion,
                "embedding": query_embedding.detach().cpu().tolist(),
                "specialists": list(self.specialist_names),
            }
            if routing_result is not None:
                record["routing"] = {
                    "metrics": [
                        {
                            "name": m.domain,
                            "similarity": float(m.similarity),
                            "mahalanobis": float(m.mahalanobis),
                            "entropy": float(m.entropy),
                            "support": float(m.support),
                            "score": float(m.score),
                        }
                        for m in routing_result.metrics
                    ],
                    "best_domain": routing_result.best_domain,
                    "is_unknown": routing_result.is_unknown,
                    "reason": routing_result.reason,
                }
            else:
                record["routing"] = {
                    "metrics": [],
                    "best_domain": None,
                    "is_unknown": True,
                    "reason": "no_router",
                }
                try:
                    log_path = Path(getattr(config, "EMERGENCE_LOG_PATH")).expanduser()
                    log_path.parent.mkdir(parents=True, exist_ok=True)
                    with open(config.EMERGENCE_LOG_PATH, "a", encoding="utf-8") as f_log:
                        f_log.write(json.dumps(record) + "\n")
                except Exception as e:
                    logger.warning("Failed to write emergence log: %s", e)

        except Exception as e:
            logger.warning("Failed to write emergence log: %s", e)

        # Decide whether to spawn a new specialist based on either the
        # geometric emergence condition (cluster size + variance) or the
        # legacy count-based threshold.
        should_register = False

        # Geometric emergence condition.
        if (
            len(self._emergent_buffer) >= config.EMERGENT_MIN_CLUSTER_SIZE
            and self._emergent_buffer
        ):
            try:
                buf_stack = torch.stack(self._emergent_buffer, dim=0)
                mean = buf_stack.mean(dim=0)
                diffs = buf_stack - mean
                sq_dists = (diffs * diffs).sum(dim=1)
                variance = float(sq_dists.mean().item())

                if variance <= config.EMERGENT_MAX_CLUSTER_VARIANCE:
                    logger.info(
                        "Emergent buffer satisfied cluster condition: n=%d, variance=%.4f <= %.4f.",
                        len(self._emergent_buffer),
                        variance,
                        config.EMERGENT_MAX_CLUSTER_VARIANCE,
                    )
                    should_register = True
                    # Reset buffer for the next emergent domain.
                    self._emergent_buffer.clear()
                else:
                    logger.debug(
                        "Emergent buffer n=%d has variance %.4f > %.4f; not spawning specialist yet.",
                        len(self._emergent_buffer),
                        variance,
                        config.EMERGENT_MAX_CLUSTER_VARIANCE,
                    )
            except Exception as e:
                logger.warning("Failed to compute emergent cluster variance: %s", e)

        # Fallback: legacy count-based condition.
        if current_samples >= config.EMERGENT_MAX_SAMPLES_PER_SPECIALIST:
            logger.info(
                "Emergent expert '%s' reached %d samples; registering as new specialist (count-based).",
                expert_name,
                current_samples,
            )
            should_register = True

        if should_register:
            # Move to next emergent expert index for future queries
            next_index = current_index + 1
            self._save_emergent_state(next_index, 0)

            # Register new specialist (triggers retrain + router refresh)
            try:
                self.add_specialist(expert_name)
            except Exception as e:
                logger.error(
                    "Failed to auto-register emergent specialist '%s': %s",
                    expert_name,
                    e,
                )

    # ------------------------------------------------------------------ #
    # Public methods used by the API
    # ------------------------------------------------------------------ #

    def get_status(self) -> Dict[str, object]:
        """
        Returns current status info for the API /health endpoint.
        """
        self.ensure_initialized()
        return {
            "initialized": self._initialized,
            "device": str(self.device),
            "tokenizer_path": str(config.TOKENIZER_PATH),
            "generalist_ckpt_path": str(config.GENERALIST_CKPT_PATH),
            "shiftable_ckpt_path": str(config.SHIFTABLE_CKPT_PATH),
            "specialists": list(self.specialist_names),
        }

    def list_specialists(self) -> List[str]:
        self.ensure_initialized()
        return list(self.specialist_names)

    def add_specialist(self, name: str) -> List[str]:
        """
        Add a new specialist domain, given that its corpus exists at:
            shiftable_project/data/<name>/*.txt

        This retrains the shiftable model over all specialists (old + new),
        rebuilds domain stats, and reloads router + model.
        """
        with self._lock:
            self.ensure_initialized()

            name = name.strip()
            if not name:
                raise ValueError("Specialist name must be a non-empty string.")

            if any(c in name for c in "/\\"):
                raise ValueError("Specialist name must not contain path separators.")

            current_specialists = set(self.specialist_names)
            if name in current_specialists:
                logger.info("Specialist '%s' already exists; no retraining needed.", name)
                return list(self.specialist_names)

            # Ensure corpus directory exists and has .txt files
            spec_dir = config.DATA_ROOT / name
            if not spec_dir.exists():
                raise RuntimeError(
                    f"Specialist directory {spec_dir} does not exist. "
                    f"Create it and add .txt files before calling this endpoint."
                )

            txt_files = list_text_files(str(spec_dir))
            if not txt_files:
                raise RuntimeError(
                    f"Specialist directory {spec_dir} contains no .txt files."
                )

            logger.info("Adding new specialist '%s'. Rebuilding shiftable model.", name)

            # Re-train shiftable model over ALL specialists (old + new)
            self._train_shiftable_for_all_specialists()
            self._load_tokenizer_and_shiftable_model()
            self.router.reload()

            logger.info("Successfully added specialist '%s'.", name)
            return list(self.specialist_names)

    def delete_specialist(self, name: str) -> List[str]:
        """
        Delete an existing specialist.

        Process:
        - Removes its corpus directory at shiftable_project/data/<name>
        - Retrains the shiftable model over remaining specialists
        - Rebuilds domain stats and reloads router + model

        At least one specialist must remain.
        """
        with self._lock:
            self.ensure_initialized()

            name = name.strip()
            if not name:
                raise ValueError("Specialist name must be a non-empty string.")

            if name not in self.specialist_names:
                raise RuntimeError(f"Specialist '{name}' does not exist.")

            if len(self.specialist_names) <= 1:
                raise RuntimeError("Cannot delete the last specialist; at least one must remain.")

            spec_dir = config.DATA_ROOT / name
            if spec_dir.exists():
                logger.info("Deleting specialist directory %s", spec_dir)
                shutil.rmtree(spec_dir)

            logger.info("Deleted specialist '%s'. Rebuilding shiftable model for remaining specialists.", name)

            self._train_shiftable_for_all_specialists()
            self._load_tokenizer_and_shiftable_model()
            self.router.reload()

            logger.info("Successfully deleted specialist '%s'. Remaining: %s", name, ", ".join(self.specialist_names))
            return list(self.specialist_names)
        
    def _build_domain_prior_from_routing(
        self,
        routing_result: RoutingResult,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Turn DomainRouter metrics into:
          - domain_prior: soft prior over domains (base + specialists)
          - domain_mask: which domains are allowed (1) vs blocked (0)

        Policy (GRCLM-style, sparse selection + soft cooperation):
          1) Use Softmax(composite_score / selection_temp) to RANK specialists.
             Select a sparse set whose cumulative probability mass reaches TOP_P,
             bounded by [MIN_K, MAX_K].
          2) Use Sigmoid(composite_score / sigmoid_temp) to assign *independent*
             blend strengths for the selected specialists.
          3) Convert strengths into a proper prior distribution (sums to 1)
             with base always included.

        Notes:
          - This function is intentionally lightweight: routing remains geometric.
          - Predictive entropy (token-distribution uncertainty) is handled later
            during generation-time calibration, and only over the allowed set.
        """

        device = self.device
        num_specialists = len(self.specialist_names)
        num_domains = 1 + num_specialists

        # Base-only routing for unknowns (or if no specialists exist)
        if getattr(routing_result, "is_unknown", False) or num_specialists == 0:
            domain_mask = torch.zeros((num_domains,), device=device, dtype=torch.float32)
            domain_mask[0] = 1.0
            domain_prior = torch.zeros((num_domains,), device=device, dtype=torch.float32)
            domain_prior[0] = 1.0
            return domain_prior, domain_mask

        # Resolve config with safe defaults (some configs use slightly different names)
        selection_temp = float(getattr(config, "ROUTER_SELECTION_TEMPERATURE", 1.0) or 1.0)
        sigmoid_temp = float(getattr(config, "ROUTER_SIGMOID_TEMPERATURE", 1.0) or 1.0)
        top_p = float(getattr(config, "ROUTER_TOP_P", 0.6) or 0.6)
        max_k = int(getattr(config, "ROUTER_TOP_K_ADVISORS", 0) or 0)
        min_k = int(getattr(config, "ROUTER_MIN_K_ADVISORS", getattr(config, "ROUTER_MIN_K", 1)) or 1)

        # Clamp/guard values
        if selection_temp <= 0:
            selection_temp = 1.0
        if sigmoid_temp <= 0:
            sigmoid_temp = 1.0
        top_p = max(0.0, min(1.0, top_p))
        if max_k <= 0:
            max_k = num_specialists
        max_k = min(max_k, num_specialists)
        min_k = max(0, min(min_k, max_k))

        # Map specialist names -> domain indices
        name_to_idx: Dict[str, int] = {"general": 0}
        for i, name in enumerate(self.specialist_names, start=1):
            name_to_idx[name] = i

        # Build composite score vector (base + specialists). Base has no router metric; keep it 0.
        scores = torch.full((num_domains,), float("-inf"), device=device, dtype=torch.float32)
        scores[0] = 0.0
        for m in getattr(routing_result, "metrics", []):
            idx = name_to_idx.get(getattr(m, "domain", None))
            if idx is not None:
                scores[idx] = float(getattr(m, "score", float("-inf")))

        # Specialist-only scores for selection (exclude base)
        spec_scores = scores[1:].clone()

        # If no valid specialist scores, fall back to base-only
        if torch.isneginf(spec_scores).all():
            domain_mask = torch.zeros((num_domains,), device=device, dtype=torch.float32)
            domain_mask[0] = 1.0
            domain_prior = torch.zeros((num_domains,), device=device, dtype=torch.float32)
            domain_prior[0] = 1.0
            return domain_prior, domain_mask

        # Softmax ranking probabilities
        rank_logits = spec_scores / selection_temp
        rank_probs = torch.softmax(rank_logits, dim=0)

        # Rank indices (0..num_specialists-1) by descending prob
        order = torch.argsort(rank_probs, descending=True)

        # Select sparse set by cumulative probability mass up to top_p
        selected_spec: List[int] = []
        cum = 0.0
        for j in order.tolist():
            if torch.isneginf(spec_scores[j]):
                continue
            selected_spec.append(j)
            cum += float(rank_probs[j])
            if len(selected_spec) >= max_k:
                break
            if cum >= top_p and len(selected_spec) >= min_k:
                break

        # Ensure minimum selection if possible
        if len(selected_spec) < min_k:
            for j in order.tolist():
                if j in selected_spec:
                    continue
                if torch.isneginf(spec_scores[j]):
                    continue
                selected_spec.append(j)
                if len(selected_spec) >= min_k or len(selected_spec) >= max_k:
                    break

        # Build mask (base always included)
        domain_mask = torch.zeros((num_domains,), device=device, dtype=torch.float32)
        domain_mask[0] = 1.0
        for j in selected_spec:
            domain_mask[1 + j] = 1.0

        # Sigmoid blend strengths (independent)
        strengths = torch.zeros((num_domains,), device=device, dtype=torch.float32)
        strengths[0] = 1.0
        sig = torch.sigmoid(spec_scores / sigmoid_temp)
        for j in selected_spec:
            strengths[1 + j] = sig[j]

        # Convert strengths into a proper prior distribution over allowed domains
        eps = float(getattr(config, "ROUTER_EPS", 1e-8) or 1e-8)
        strengths = strengths * domain_mask
        total = strengths.sum().clamp_min(eps)
        domain_prior = (strengths / total).clamp_min(eps)
        domain_prior = domain_prior / domain_prior.sum().clamp_min(eps)

        return domain_prior, domain_mask

    def _calibrate_domain_weights_with_predictive_entropy(
        self,
        input_ids_tensor: torch.Tensor,
        domain_prior: torch.Tensor,
        domain_mask: torch.Tensor,
        domain_weights: Optional[torch.Tensor],
        temperature: float = 1.0,
        eta: float = 1.0,
        lam: float = 1.0,
        min_conf: float = 0.2,
        eps: float = 1e-9,
    ) -> torch.Tensor:
        """
        Predictive-entropy calibration for generation-time blending.

        We compute a *confidence* per allowed domain by measuring next-token
        predictive entropy under an (almost) isolated one-hot domain weighting.
        Then we reweight the existing blend weights (geometry prior + optional
        learned correction) as:

            w_final ∝ w_in * conf^lam,  where conf = exp(-eta * H_pred)

        This NEVER changes the selection set; it only rescales within domain_mask.
        """
        self.ensure_initialized()
        assert self.shift_model is not None
        assert self.tokenizer is not None

        # Shape normalization: expect [1, num_domains]
        if input_ids_tensor.dim() != 2:
            raise ValueError(f"input_ids_tensor must be [1,S], got {tuple(input_ids_tensor.shape)}")

        if domain_prior.dim() == 1:
            domain_prior = domain_prior.unsqueeze(0)
        if domain_mask.dim() == 1:
            domain_mask = domain_mask.unsqueeze(0)

        num_domains = int(domain_mask.size(-1))

        # Choose the incoming weights (prefer explicit domain_weights, else the geometric prior)
        w_in = domain_weights if domain_weights is not None else domain_prior
        if w_in.dim() == 1:
            w_in = w_in.unsqueeze(0)

        # Restrict to allowed domains
        allowed = (domain_mask[0] > 0.0).nonzero(as_tuple=False).view(-1).tolist()
        if len(allowed) <= 1:
            return w_in

        # Numerical guards
        if temperature <= 0:
            temperature = 1.0
        min_conf = float(max(0.0, min(1.0, min_conf)))
        eps = float(max(1e-12, eps))

        # Build an attention mask once
        attention_mask = (input_ids_tensor != self.tokenizer.pad_id).long()

        conf = torch.ones((num_domains,), device=self.device, dtype=torch.float32)

        self.shift_model.eval()
        with torch.no_grad():
            for idx in allowed:
                # Build a run-specific sparse mask (keep base compute allowed)
                run_mask = torch.zeros((1, num_domains), device=self.device, dtype=torch.float32)
                run_mask[0, 0] = 1.0
                run_mask[0, idx] = 1.0

                # One-hot weighting to isolate the domain contribution
                run_w = torch.zeros((1, num_domains), device=self.device, dtype=torch.float32)
                run_w[0, idx] = 1.0

                logits = self.shift_model(
                    input_ids_tensor,
                    attention_mask=attention_mask,
                    domain_prior=domain_prior,
                    domain_mask=run_mask,
                    domain_weights=run_w,
                )

                next_logits = logits[0, -1, :] / temperature
                logp = torch.log_softmax(next_logits, dim=-1)
                p = torch.exp(logp)
                h = -(p * logp).sum()  # scalar entropy

                # Confidence from entropy
                c = torch.exp(-eta * h).clamp(min=min_conf, max=1.0)
                conf[idx] = c

        # Combine incoming weights with confidence
        w = w_in[0].clone()
        w = w * (conf ** lam)

        # Enforce selection mask and renormalize
        w = w * domain_mask[0]
        s = float(w.sum().item())
        if s > 0:
            w = w / s
        else:
            # fallback uniform over allowed
            w = domain_mask[0] / float(len(allowed))

        return w.unsqueeze(0)
    def _compute_query_embedding(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Compute a query embedding in the SAME space as the domain fingerprints.

        IMPORTANT:
        This uses the GENERALIST TRUNK encoder (BaseTransformerLM.encode), not the
        shift model's token embedding table. That makes cosine/Mahalanobis routing
        geometrically meaningful and consistent with GRCLM.

        Args:
            input_ids: Tensor of shape [seq_len] on self.device.

        Returns:
            1D tensor of shape [d_model].
        """
        assert self.base_model is not None
        assert self.tokenizer is not None

        if input_ids.ndim != 1:
            raise ValueError(f"Expected [seq_len] input_ids, got shape {tuple(input_ids.shape)}")

        ids = input_ids.unsqueeze(0)  # [1, S]
        attn = (ids != self.tokenizer.pad_id).long()

        q = self.base_model.encode(ids, attention_mask=attn)  # [1, D]
        return q.squeeze(0)

    def generate(
        self,
        prompt: str,
        max_new_tokens: int,
        temperature: float,
        top_k: int,
    ) -> Dict[str, str]:
        """
        Autoregressive generation using the current shiftable model, plus
        geometric routing and automatic emergent expert creation.

        Now the geometric router is also used to steer which specialist heads
        participate in answering the query.
        """
        self.ensure_initialized()
        assert self.shift_model is not None
        assert self.tokenizer is not None

        self.shift_model.eval()

        # ---------------------------------------------------------------------
        # 1) Encode prompt
        # ---------------------------------------------------------------------
        # IMPORTANT: for generation we must NOT append <eos> to the prompt.
        prompt_ids = self.tokenizer.encode(prompt, add_specials=False)
        input_ids = [self.tokenizer.bos_id] + prompt_ids
        input_ids_tensor = torch.tensor(
            input_ids,
            dtype=torch.long,
            device=self.device,
        ).unsqueeze(0)

        # ---------------------------------------------------------------------
        # 2) Compute query embedding and run geometric routing over domains
        # ---------------------------------------------------------------------
        base_input_ids_tensor = torch.tensor(
            input_ids,
            dtype=torch.long,
            device=self.device,
        )
        query_embedding = self._compute_query_embedding(base_input_ids_tensor)

        routing_result: Optional[RoutingResult] = None
        if self.specialist_names:
            routing_result = self.router.route(query_embedding)

        domain_prior: Optional[torch.Tensor] = None  # shape [1, 1 + num_specialists]
        domain_mask: Optional[torch.Tensor] = None   # shape [1, 1 + num_specialists]
        domain_weights: Optional[torch.Tensor] = None  # shape [1, 1 + num_specialists]


        # Only build a prior/mask if we have specialists AND the router thinks
        # this query belongs to a known domain.
        if (
            self.specialist_names
            and routing_result is not None
            and not routing_result.is_unknown
        ):
            # Use the GRCLM-style routing policy helper:
            # softmax (selection probs) -> sigmoid (blend strength) -> learned gate (inside SMA)
            prior_vec, mask_vec = self._build_domain_prior_from_routing(routing_result)

            # Shape to [1, num_domains] so it can be broadcast across batch
            domain_prior = prior_vec.unsqueeze(0)
            domain_mask = mask_vec.unsqueeze(0)

            # compute a fixed (per-prompt) domain weight vector using
            # geometry prior + a small learned correction gate.
            domain_weights = None
            try:
                alpha = float(getattr(config, "GATE_GEOMETRY_ALPHA", 6.0))
                beta = float(getattr(config, "GATE_LEARNED_BETA", 1.0))

                # Learned correction gate lives on the shift model (trained lightly when specialists change)
                if hasattr(self.shift_model, "blend_gate") and callable(getattr(self.shift_model, "blend_gate")):
                    pooled = query_embedding.unsqueeze(0)  # [1, d_model]
                    g_logits = self.shift_model.blend_gate(pooled)  # [1, num_domains]

                    prior = torch.clamp(domain_prior, min=1e-9)
                    mix_logits = alpha * torch.log(prior) + beta * g_logits

                    # Enforce sparse selection
                    mix_logits = mix_logits.masked_fill(domain_mask == 0, float("-inf"))
                    domain_weights = torch.softmax(mix_logits, dim=-1)
            except Exception:
                # If anything goes wrong, fall back to prior-driven per-layer gate
                domain_weights = None


        # ---------------------------------------------------------------------
        # 2b) Optional: predictive-entropy calibration of domain_weights
        # ---------------------------------------------------------------------
        # Geometry decides *which* domains are allowed (domain_mask) and provides
        # the prior (domain_prior). Predictive entropy then reweights only the
        # allowed domains based on next-token confidence, without overriding the
        # geometric selection itself.
        if (
            domain_mask is not None
            and domain_prior is not None
            and int(torch.count_nonzero(domain_mask).item()) > 1
            and bool(getattr(config, "USE_PREDICTIVE_ENTROPY", True))
        ):
            try:
                domain_weights = self._calibrate_domain_weights_with_predictive_entropy(
                    input_ids_tensor=input_ids_tensor,
                    domain_prior=domain_prior,
                    domain_mask=domain_mask,
                    domain_weights=domain_weights,
                    temperature=float(getattr(config, "PRED_ENTROPY_TEMPERATURE", 1.0) or 1.0),
                    eta=float(getattr(config, "PRED_ENTROPY_ETA", 1.0) or 1.0),
                    lam=float(getattr(config, "PRED_ENTROPY_LAMBDA", 1.0) or 1.0),
                    min_conf=float(getattr(config, "PRED_ENTROPY_MIN_CONF", 0.2) or 0.2),
                    eps=float(getattr(config, "PRED_ENTROPY_EPS", 1e-9) or 1e-9),
                )
            except Exception:
                # Any failure should never break generation; fall back to geometry-only weights.
                pass

        # ---------------------------------------------------------------------
        # 3) Autoregressive generation, steered by domain_prior/domain_mask
        # ---------------------------------------------------------------------
        generated_ids: List[int] = input_ids.copy()

        for _ in range(max_new_tokens):
            if len(generated_ids) > self.shift_model.max_seq_len:
                context_ids = generated_ids[-self.shift_model.max_seq_len :]
            else:
                context_ids = generated_ids

            context_tensor = torch.tensor(
                context_ids,
                dtype=torch.long,
                device=self.device,
            ).unsqueeze(0)

            attention_mask = (context_tensor != self.tokenizer.pad_id).long()

            with torch.no_grad():
                # NOTE: shift_model must accept domain_prior/domain_mask kwargs
                logits = self.shift_model(
                    context_tensor,
                    attention_mask=attention_mask,
                    domain_prior=domain_prior,
                    domain_mask=domain_mask,
                    domain_weights=domain_weights,
                )
                next_token_logits = logits[0, -1, :]

            if temperature <= 0:
                next_token_id = int(torch.argmax(next_token_logits).item())
            else:
                logits_scaled = next_token_logits / temperature

                if top_k > 0 and top_k < logits_scaled.size(-1):
                    values, indices = torch.topk(logits_scaled, top_k)
                    logits_filtered = torch.full_like(logits_scaled, float("-inf"))
                    logits_filtered[indices] = values
                else:
                    logits_filtered = logits_scaled

                probs = torch.softmax(logits_filtered, dim=-1)
                next_token_id = int(torch.multinomial(probs, num_samples=1).item())
                # What if we use softmax to find which heads to invoke and sigmoid to have them work together for generation?
            generated_ids.append(next_token_id)

            if next_token_id == self.tokenizer.eos_id:
                break

        full_text = self.tokenizer.decode(generated_ids, skip_specials=True)
        completion_ids = generated_ids[len(input_ids) :]
        completion_text = self.tokenizer.decode(completion_ids, skip_specials=True)

        # -----------------------------------------------------------------
        # 3.5) Low-confidence / unknown-domain response policy
        # If routing is unknown (or there are no specialists), we still log
        # the sample for emergence, but we *prepend* an explicit disclaimer.
        # If the model produces an empty completion (e.g., EOS immediately),
        # we provide a minimal fallback string so the user sees something.
        # -----------------------------------------------------------------
        is_low_confidence = (not self.specialist_names) or (
            routing_result is not None and routing_result.is_unknown
        )
        if is_low_confidence:
            disclaimer = (
                "I don’t know, but I will log that I need to learn this. "
                "I have low confidence this is the correct answer:\n\n"
            )
            if completion_text.strip() == "":
                completion_text = "I don't know what to think."
            completion_text = disclaimer + completion_text
            # Keep full_text consistent with what we return
            full_text = prompt + "\n" + completion_text

        self._log_input_output(
            prompt=prompt,
            completion=completion_text,
            query_embedding=query_embedding,
            routing_result=routing_result,
            domain_prior=domain_prior,
            domain_mask=domain_mask,
        )


        # ---------------------------------------------------------------------
        # 4) Emergent handling: unchanged logic
        # [this is placeholder logic and not correct. Agentic or Human-in-the-loop specialist creation required]
        # ---------------------------------------------------------------------
        if (not self.specialist_names) or (
            routing_result is not None and routing_result.is_unknown
        ):
            self._handle_emergent_query(
                prompt=prompt,
                completion=completion_text,
                query_embedding=query_embedding,
                routing_result=routing_result,
            )

        return {
            "prompt": prompt,
            "completion": completion_text,
            "full_text": full_text,
        }
    

# Global singleton manager for the API
model_manager = ModelManager()
