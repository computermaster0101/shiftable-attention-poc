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
    """
    Any directory under shiftable_project/data/ that is not 'general'
    is treated as a specialist corpus.
    """
    names: List[str] = []
    data_root = config.DATA_ROOT
    if not data_root.exists():
        return names

    for child in data_root.iterdir():
        if child.is_dir() and child.name != "general":
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
        self.router = DomainRouter(config.DOMAIN_STATS_PATH, self.device)

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

            if not config.SHIFTABLE_CKPT_PATH.exists():
                logger.info("Shiftable checkpoint not found. Training shiftable model for all specialists.")
                self._train_shiftable_for_all_specialists()
            else:
                logger.info("Found existing shiftable checkpoint. Loading model.")

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
            val_loss = _evaluate(model, dataloader, self.device, pad_id=tokenizer.pad_id)
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
        """
        if not config.TOKENIZER_PATH.exists():
            raise RuntimeError("Tokenizer not found; cannot train shiftable model.")

        tokenizer = SimpleTokenizer.load(str(config.TOKENIZER_PATH))
        base_model, base_cfg = self._load_generalist_model()

        specialist_names = _discover_specialist_names()
        num_specialists = len(specialist_names)

        if not specialist_names:
            logger.warning(
                "No specialist directories found. Training shiftable model with general-only (0 specialists)."
            )
        else:
            logger.info("Training shiftable model for specialists: %s", ", ".join(specialist_names))

        shift_model = ShiftableTransformerLM.from_base_model(
            base_model,
            num_specialists=num_specialists,
            specialist_names=specialist_names if specialist_names else [],
            use_cls_token_pool=config.USE_CLS_TOKEN_POOL,
        )

        shift_model = shift_model.to(self.device)

        # Optionally freeze embeddings and LM head to focus training on specialist branches
        for name, param in shift_model.named_parameters():
            if name.startswith("tok_emb") or name.startswith("pos_emb") or name.startswith("lm_head"):
                param.requires_grad = False

        # Build datasets (general + all specialists)
        datasets = []

        if config.GENERAL_DATA_DIR.exists():
            logger.info("Including general corpus in shiftable training.")
            general_ds = _build_lm_dataset_from_dir(config.GENERAL_DATA_DIR, tokenizer, seq_len=config.MAX_SEQ_LEN)
            datasets.append(general_ds)
        else:
            logger.warning("General data directory %s does not exist; skipping general corpus.", config.GENERAL_DATA_DIR)

        for spec_name in specialist_names:
            spec_dir = config.DATA_ROOT / spec_name
            logger.info("Adding specialist dataset from %s", spec_dir)
            spec_ds = _build_lm_dataset_from_dir(spec_dir, tokenizer, seq_len=config.MAX_SEQ_LEN)
            datasets.append(spec_ds)

        train_dataset = ConcatDataset(datasets)
        dataloader = DataLoader(
            train_dataset, 
            batch_size=config.SHIFTABLE_BATCH_SIZE, 
            shuffle=True,
            )

        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, shift_model.parameters()),
            lr=config.SHIFTABLE_LR,
        )

        for epoch in range(1, config.SHIFTABLE_EPOCHS + 1):
            train_loss = _train_one_epoch(shift_model, dataloader, optimizer, self.device, pad_id=tokenizer.pad_id)
            val_loss = _evaluate(shift_model, dataloader, self.device, pad_id=tokenizer.pad_id)
            logger.info(
                "[Shiftable] Epoch %d/%d - train loss: %.4f, val loss: %.4f",
                epoch,
                config.SHIFTABLE_EPOCHS,
                train_loss,
                val_loss,
            )

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
        logger.info("Saved shiftable checkpoint to %s", config.SHIFTABLE_CKPT_PATH)

        # Build / refresh domain statistics for routing (general + specialists)
        domain_names = ["general"] + specialist_names
        self._build_domain_stats(tokenizer, base_model, domain_names)
        self.router.reload()

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
        Load tokenizer and shiftable model from disk into memory.
        """
        if not config.TOKENIZER_PATH.exists():
            raise RuntimeError("Tokenizer file not found when attempting to load shiftable model.")
        if not config.SHIFTABLE_CKPT_PATH.exists():
            raise RuntimeError("Shiftable checkpoint not found when attempting to load model.")

        tokenizer = SimpleTokenizer.load(str(config.TOKENIZER_PATH))
        ckpt = torch.load(config.SHIFTABLE_CKPT_PATH, map_location="cpu")
        cfg: Dict[str, object] = ckpt["config"]

        base_model, _ = self._load_generalist_model()
        base_model = base_model.to(self.device)
        base_model.eval()
        self.base_model = base_model
        shift_model = ShiftableTransformerLM.from_base_model(
            base_model,
            num_specialists=int(cfg["num_specialists"]),
            specialist_names=list(cfg["specialist_names"]),
            use_cls_token_pool=False,
        )
        shift_model.load_state_dict(ckpt["model_state_dict"])
        shift_model = shift_model.to(self.device)
        shift_model.eval()

        self.tokenizer = tokenizer
        self.shift_model = shift_model
        self.specialist_names = list(cfg["specialist_names"])

        logger.info("Loaded shiftable model with specialists: %s", ", ".join(self.specialist_names))

    # ------------------------------------------------------------------ #
    # Emergent expert state + corpus helpers
    # ------------------------------------------------------------------ #

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
                        "name": m.name,
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
                            "name": m.name,
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

            config.EMERGENCE_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
            with open(config.EMERGENCE_LOG_PATH, "a", encoding="utf-8") as f_log:
                f_log.write(json.dumps(record) + "\n")
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

        Routing policy (GRCLM-style):
          1) Softmax over composite scores is used to *rank/select* specialists.
             We select a set whose cumulative Softmax probability mass reaches
             ROUTER_TOP_P (default 0.6), capped by ROUTER_TOP_K_ADVISORS.
          2) Sigmoid over composite scores provides *independent blend strength*
             for the selected specialists.
          3) The learned gate (inside shiftable attention) still blends, but is
             constrained by domain_mask and biased by domain_prior.

        Domain index convention:
          idx 0: "general" (base)
          idx i>0: self.specialist_names[i-1]
        """
        assert self.specialist_names is not None

        device = self.device
        num_specialists = len(self.specialist_names)
        num_domains = 1 + num_specialists

        # Map names -> indices
        name_to_idx: Dict[str, int] = {"general": 0}
        for i, name in enumerate(self.specialist_names, start=1):
            name_to_idx[name] = i

        # Composite score vector (base + specialists)
        scores = torch.full((num_domains,), float("-inf"), device=device, dtype=torch.float32)

        # Fill with composite scores from router
        for m in routing_result.metrics:
            idx = name_to_idx.get(getattr(m, "domain", None))
            if idx is not None:
                scores[idx] = float(m.score)

        # Always keep base in the game; if it was -inf, give it a neutral score
        if not torch.isfinite(scores[0]):
            scores[0] = 0.0

        # Config knobs (optional)
        sel_temp = float(getattr(config, "ROUTER_SELECTION_TEMPERATURE", 1.0) or 1.0)
        blend_temp = float(getattr(config, "ROUTER_BLEND_TEMPERATURE", 1.0) or 1.0)
        blend_bias = float(getattr(config, "ROUTER_BLEND_BIAS", 0.0) or 0.0)

        # Selection mass (top-p) and caps
        top_p = float(getattr(config, "ROUTER_TOP_P", 0.6) or 0.6)
        top_p = max(0.0, min(1.0, top_p))

        max_k = int(getattr(config, "ROUTER_TOP_K_ADVISORS", 0) or 0)
        if max_k <= 0:
            max_k = num_specialists  # no cap

        min_k = int(getattr(config, "ROUTER_MIN_K", 1) or 1)
        min_k = max(0, min(min_k, num_specialists))

        # Build mask: base always included
        domain_mask = torch.zeros((num_domains,), device=device, dtype=torch.float32)
        domain_mask[0] = 1.0

        # If no specialists, prior is all on base.
        if num_specialists == 0:
            domain_prior = torch.zeros((num_domains,), device=device, dtype=torch.float32)
            domain_prior[0] = 1.0
            return domain_prior, domain_mask

        # Softmax over specialist scores for selection (ignore -inf -> prob 0)
        spec_scores = scores[1:].clone()
        # Temperature scaling (avoid divide-by-zero)
        if sel_temp <= 0:
            sel_temp = 1.0
        spec_probs = torch.softmax(spec_scores / sel_temp, dim=0)

        # Rank specialists by probability
        sorted_probs, sorted_idx = torch.sort(spec_probs, descending=True)

        selected_rel: list[int] = []
        cum = 0.0
        for p, idx_rel in zip(sorted_probs.tolist(), sorted_idx.tolist()):
            # Skip degenerate / invalid specialists
            if p <= 0 or not torch.isfinite(spec_scores[idx_rel]):
                continue
            selected_rel.append(int(idx_rel))
            cum += float(p)
            if len(selected_rel) >= max_k:
                break
            if cum >= top_p and len(selected_rel) >= min_k:
                break

        # Fallback: if nothing selected but we have valid scores, pick best one
        if not selected_rel:
            # pick argmax among finite scores
            finite_mask = torch.isfinite(spec_scores)
            if torch.any(finite_mask):
                best = int(torch.argmax(torch.where(finite_mask, spec_probs, torch.tensor(-1.0, device=device))).item())
                selected_rel = [best]

        # Apply selection to mask (specialists are 1..num_domains-1)
        for idx_rel in selected_rel:
            domain_mask[1 + idx_rel] = 1.0

        # Sigmoid blend strength on selected (and base)
        if blend_temp <= 0:
            blend_temp = 1.0
        strengths = torch.sigmoid((scores - blend_bias) / blend_temp)

        # Build prior = (selection_probs * strengths) for selected specialists, plus base strength.
        domain_prior = torch.zeros((num_domains,), device=device, dtype=torch.float32)
        domain_prior[0] = strengths[0]  # base participation

        for idx_rel in selected_rel:
            dom_idx = 1 + idx_rel
            domain_prior[dom_idx] = spec_probs[idx_rel] * strengths[dom_idx]

        # Normalize prior over allowed domains (avoid all-zero)
        domain_prior = domain_prior * domain_mask
        s = float(domain_prior.sum().item())
        if s > 0:
            domain_prior = domain_prior / s
        else:
            # uniform over allowed domains as last resort
            allowed = torch.count_nonzero(domain_mask).item()
            if allowed > 0:
                domain_prior = domain_mask / float(allowed)

        return domain_prior, domain_mask





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
        input_ids = self.tokenizer.encode(prompt, add_specials=True)
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
