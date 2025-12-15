from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn

from shiftable_attention import ShiftableTransformerBlock
from shiftable_attention.gate import DomainGate


class BaseTransformerLM(nn.Module):
    """
    Standard Transformer encoder language model (generalist).

    Uses PyTorch's TransformerEncoder underneath.
    """
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 4,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        max_seq_len: int = 128,
        pad_id: int = 0,
    ) -> None:
        super().__init__()

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.max_seq_len = max_seq_len
        self.pad_id = pad_id

        self.tok_emb = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.lm_head = nn.Linear(d_model, vocab_size)

        # Expose per-layer self-attention modules so ShiftableTransformerLM.from_base_model
        # can clone and freeze them. This assumes the encoder is a standard
        # nn.TransformerEncoder with a .layers ModuleList, each having .self_attn.
        self.self_attns = nn.ModuleList([layer.self_attn for layer in self.encoder.layers])

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            input_ids: [batch, seq_len]
            attention_mask: optional [batch, seq_len] with 1 for real tokens and 0 for padding.
        Returns:
            logits: [batch, seq_len, vocab_size]
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        if seq_len > self.max_seq_len:
            raise ValueError(f"Sequence length {seq_len} exceeds max_seq_len={self.max_seq_len}")

        positions = torch.arange(0, seq_len, device=device).unsqueeze(0).expand(batch_size, seq_len)
        x = self.tok_emb(input_ids) + self.pos_emb(positions)

        if attention_mask is not None:
            key_padding_mask = attention_mask == 0
        else:
            key_padding_mask = input_ids == self.pad_id

        x = self.encoder(x, src_key_padding_mask=key_padding_mask)
        logits = self.lm_head(x)
        return logits

    @torch.no_grad()
    def encode(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Trunk embedding: mean-pooled TransformerEncoder output (not raw tok_emb pooling).

        Args:
            input_ids: [batch, seq_len]
            attention_mask: optional [batch, seq_len] with 1 for real tokens and 0 for padding.

        Returns:
            pooled: [batch, d_model]
        """
        self.eval()

        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        if seq_len > self.max_seq_len:
            raise ValueError(f"Sequence length {seq_len} exceeds max_seq_len={self.max_seq_len}")

        positions = torch.arange(0, seq_len, device=device).unsqueeze(0).expand(batch_size, seq_len)
        x = self.tok_emb(input_ids) + self.pos_emb(positions)

        if attention_mask is not None:
            key_padding_mask = attention_mask == 0
            valid = attention_mask.float()
        else:
            key_padding_mask = input_ids == self.pad_id
            valid = (input_ids != self.pad_id).float()

        x = self.encoder(x, src_key_padding_mask=key_padding_mask)  # [B,S,D]

        denom = valid.sum(dim=1, keepdim=True).clamp_min(1.0)
        pooled = (x * valid.unsqueeze(-1)).sum(dim=1) / denom
        return pooled


class ShiftableTransformerLM(nn.Module):
    """
    Language model built from ShiftableTransformerBlock layers.

    The base MHA in each block is initialized from a pretrained BaseTransformerLM
    and frozen. Specialist attention branches and the domain gate are trainable.
    """
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        dim_feedforward: int,
        dropout: float,
        max_seq_len: int,
        pad_id: int,
        base_self_attns: List[nn.MultiheadAttention],
        num_specialists: int,
        specialist_names: Optional[List[str]] = None,
        use_cls_token_pool: bool = False,
    ) -> None:
        super().__init__()

        if len(base_self_attns) != n_layers:
            raise ValueError("base_self_attns must have length == n_layers")
        if num_specialists < 0:
            raise ValueError("num_specialists must be >= 0")

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.max_seq_len = max_seq_len
        self.pad_id = pad_id
        self.num_specialists = num_specialists

        if num_specialists == 0:
            # There are no specialists yet.
            if specialist_names is not None and len(specialist_names) != 0:
                raise ValueError(
                    "num_specialists=0 requires specialist_names to be empty or None"
                )
            specialist_names = []
        else:
            # Normal case: at least one specialist
            if specialist_names is None:
                specialist_names = [f"specialist_{i}" for i in range(num_specialists)]
            if len(specialist_names) != num_specialists:
                raise ValueError("len(specialist_names) must equal num_specialists")

        self.specialist_names = specialist_names

        # shared learned correction gate (computed once per prompt)
        self.num_domains = 1 + num_specialists
        self.blend_gate = DomainGate(d_model=d_model, num_domains=self.num_domains)

        self.tok_emb = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)

        blocks: List[ShiftableTransformerBlock] = []
        for i in range(n_layers):
            block = ShiftableTransformerBlock(
                d_model=d_model,
                num_heads=n_heads,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                base_mha=base_self_attns[i],
                num_specialists=num_specialists,
                use_cls_token_pool=use_cls_token_pool,
            )
            blocks.append(block)

        self.blocks = nn.ModuleList(blocks)
        self.lm_head = nn.Linear(d_model, vocab_size)

    @classmethod
    def from_base_model(
        cls,
        base_model: BaseTransformerLM,
        num_specialists: int,
        specialist_names: Optional[List[str]] = None,
        use_cls_token_pool: bool = False,
    ) -> "ShiftableTransformerLM":
        # First-run path: num_specialists can be 0
        return cls(
            vocab_size=base_model.vocab_size,
            d_model=base_model.d_model,
            n_heads=base_model.n_heads,
            n_layers=base_model.n_layers,
            dim_feedforward=base_model.dim_feedforward,
            dropout=base_model.dropout,
            max_seq_len=base_model.max_seq_len,
            pad_id=base_model.pad_id,
            base_self_attns=base_model.self_attns,
            num_specialists=num_specialists,
            specialist_names=specialist_names,
            use_cls_token_pool=use_cls_token_pool,
        )
    
    def forward(
            self,
            input_ids: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            return_gates: bool = False,
            domain_prior: Optional[torch.Tensor] = None,
            domain_mask: Optional[torch.Tensor] = None,
            domain_weights: Optional[torch.Tensor] = None,
        ) -> tuple[torch.Tensor, Optional[list[torch.Tensor]]]:

        """
        Args:
            input_ids: [batch, seq_len]
            attention_mask: [batch, seq_len] with 1 for tokens, 0 for padding
            return_gates: whether to also return gate tensors per layer
        Returns:
            logits: [batch, seq_len, vocab_size]
            gates (optional): List[Tensor] of shape [batch, 1 + num_specialists] per layer
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        if seq_len > self.max_seq_len:
            raise ValueError(f"Sequence length {seq_len} exceeds max_seq_len={self.max_seq_len}")

        positions = torch.arange(0, seq_len, device=device).unsqueeze(0).expand(batch_size, seq_len)
        x = self.tok_emb(input_ids) + self.pos_emb(positions)

        if attention_mask is not None:
            key_padding_mask = attention_mask == 0
        else:
            key_padding_mask = input_ids == self.pad_id

        all_gates = []
        for block in self.blocks:
            x, gate = block(
                x,
                attn_mask=None,
                key_padding_mask=key_padding_mask,
                return_gate=return_gates,
                domain_prior=domain_prior,
                domain_mask=domain_mask,
                domain_weights=domain_weights,
            )
            if return_gates:
                all_gates.append(gate)

        logits = self.lm_head(x)

        if return_gates:
            return logits, all_gates
        return logits
