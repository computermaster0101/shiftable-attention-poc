from typing import Optional, Tuple

import torch
import torch.nn as nn

from .sma import ShiftableMultiheadAttention


class ShiftableTransformerBlock(nn.Module):
    """
    Transformer encoder block using ShiftableMultiheadAttention.

    Standard pattern:
        x -> LayerNorm -> SMA -> residual
          -> LayerNorm -> FFN -> residual
    """
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        base_mha: Optional[nn.MultiheadAttention] = None,
        num_specialists: int = 1,
        use_cls_token_pool: bool = False,
    ):
        super().__init__()

        # If no base MHA passed, create one (you would then train it as part of a base model)
        if base_mha is None:
            base_mha = nn.MultiheadAttention(
                embed_dim=d_model,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True,
            )

        self.self_attn = ShiftableMultiheadAttention(
            d_model=d_model,
            num_heads=num_heads,
            base_mha=base_mha,
            num_specialists=num_specialists,
            dropout=dropout,
            use_cls_token_pool=use_cls_token_pool,
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
        return_gate: bool = False,
        domain_prior: Optional[torch.Tensor] = None,
        domain_mask: Optional[torch.Tensor] = None,
            domain_weights: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            x: [batch, seq_len, d_model]
        Returns:
            x_out: [batch, seq_len, d_model]
            gate (optional): [batch, num_domains] from the SMA gate
        """
        # Self-attention with shiftable attention
        x_norm = self.norm1(x)
        attn_out, gate = self.self_attn(
            x_norm,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            return_gate=return_gate,
            domain_prior=domain_prior,
            domain_mask=domain_mask,
            domain_weights=domain_weights,
        )
        x = x + self.dropout(attn_out)

        # Feedforward
        x_norm2 = self.norm2(x)
        ffn_out = self.ffn(x_norm2)
        x = x + self.dropout(ffn_out)

        if return_gate:
            return x, gate
        else:
            return x, None
