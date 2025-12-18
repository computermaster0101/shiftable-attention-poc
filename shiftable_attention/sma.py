from typing import Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from .gate import DomainGate


class ShiftableMultiheadAttention(nn.Module):
    """
    Shiftable Modular Attention (SMA).

    Combines:
      - a frozen base MultiheadAttention
      - one or more trainable specialist MultiheadAttention modules
      - a trainable DomainGate to weight domains

    For D specialists:
        num_domains = 1 + D
        domain 0: base
        domain i>0: specialist i-1

    Output:
        Y = sum_d g_d(X) * MHA_d(X)
    where g_d(X) are gate weights for each domain.

    Args:
        d_model: hidden size.
        num_heads: number of attention heads.
        base_mha: a pre-trained nn.MultiheadAttention module.
        num_specialists: number of specialist branches.
        dropout: attention dropout.
        use_cls_token_pool: if True, pool using first token; else mean over sequence.
    """
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        base_mha: nn.MultiheadAttention,
        num_specialists: int = 1,
        dropout: float = 0.0,
        use_cls_token_pool: bool = False,
    ):
        super().__init__()

        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        assert num_specialists >= 0, "num_specialists must be >= 0"

        self.d_model = d_model
        self.num_heads = num_heads
        self.num_specialists = num_specialists
        self.use_cls_token_pool = use_cls_token_pool

        # Store and freeze base MHA
        self.base_mha = base_mha
        for p in self.base_mha.parameters():
            p.requires_grad = False

        # Ensure we know whether the base module expects batch_first
        self.base_batch_first = getattr(self.base_mha, "batch_first", False)

        # Specialist MHA modules
        self.spec_mhas = nn.ModuleList([
            nn.MultiheadAttention(
                embed_dim=d_model,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True,  # we use [batch, seq, d_model]
            )
            for _ in range(num_specialists)
        ])

        # Domain gate: 1 base + N specialist branches
        num_domains = 1 + num_specialists
        self.gate = DomainGate(d_model=d_model, num_domains=num_domains)

    def pool_hidden(self, x: torch.Tensor) -> torch.Tensor:
        """
        Pool hidden states across sequence dimension.

        Args:
            x: [batch, seq_len, d_model]

        Returns:
            [batch, d_model]
        """
        if self.use_cls_token_pool:
            # assume first token is CLS-like
            return x[:, 0, :]
        else:
            # mean pooling across sequence
            return x.mean(dim=1)

    def _run_base_mha(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor],
        key_padding_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """
        Run the frozen base MHA with no gradients.
        """
        with torch.no_grad():
            if self.base_batch_first:
                base_out, _ = self.base_mha(
                    x, x, x,
                    attn_mask=attn_mask,
                    key_padding_mask=key_padding_mask,
                    need_weights=False,
                )
            else:
                # convert to [seq, batch, d_model]
                x_t = x.transpose(0, 1)
                base_out_t, _ = self.base_mha(
                    x_t, x_t, x_t,
                    attn_mask=attn_mask,
                    key_padding_mask=key_padding_mask,
                    need_weights=False,
                )
                base_out = base_out_t.transpose(0, 1)
        return base_out

    def _run_specialists(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor],
        key_padding_mask: Optional[torch.Tensor],
        selected_indices: Optional[List[int]] = None,
    ) -> tuple[List[int], List[torch.Tensor]]:
        """
        Run a *selected* subset of specialist MHA modules.

        This is the key change that turns "sparse cooperation" (compute-all-then-blend)
        into "sparse selection" (only compute the selected experts).

        Args:
            x: [batch, seq_len, d_model]
            attn_mask: optional attention mask.
            key_padding_mask: [batch, seq_len] mask for padding positions.
            selected_indices: indices into self.spec_mhas to run. If None, runs all.

        Returns:
            (indices, outs)
              - indices: the specialist indices that were executed
              - outs: list of [batch, seq_len, d_model] outputs aligned with indices
        """
        if selected_indices is None:
            selected_indices = list(range(len(self.spec_mhas)))

        outs: List[torch.Tensor] = []
        for i in selected_indices:
            mha = self.spec_mhas[i]
            if mha.batch_first:
                out, _ = mha(
                    x, x, x,
                    attn_mask=attn_mask,
                    key_padding_mask=key_padding_mask,
                    need_weights=False,
                )
            else:
                x_t = x.transpose(0, 1)
                out_t, _ = mha(
                    x_t, x_t, x_t,
                    attn_mask=attn_mask,
                    key_padding_mask=key_padding_mask,
                    need_weights=False,
                )
                out = out_t.transpose(0, 1)
            outs.append(out)

        return selected_indices, outs

    # -----------------------------
    # Modular checkpoint helpers
    # -----------------------------

    def export_specialist_state(self, specialist_index: int) -> dict:
        """Return a CPU state_dict for a single specialist MHA branch."""
        if specialist_index < 0 or specialist_index >= len(self.spec_mhas):
            raise IndexError(f"specialist_index out of range: {specialist_index}")
        return {k: v.detach().cpu().clone() for k, v in self.spec_mhas[specialist_index].state_dict().items()}

    def load_specialist_state(self, specialist_index: int, state: dict, *, strict: bool = True) -> None:
        """Load a specialist MHA branch state_dict (from modular .pt)."""
        if specialist_index < 0 or specialist_index >= len(self.spec_mhas):
            raise IndexError(f"specialist_index out of range: {specialist_index}")
        self.spec_mhas[specialist_index].load_state_dict(state, strict=strict)

    def forward(
            self,
            x: torch.Tensor,
            attn_mask: Optional[torch.Tensor] = None,
            key_padding_mask: Optional[torch.Tensor] = None,
            return_gate: bool = False,
            domain_prior: Optional[torch.Tensor] = None,
            domain_mask: Optional[torch.Tensor] = None,
            domain_weights: Optional[torch.Tensor] = None,
        ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:

        """
        Args:
            x: [batch, seq_len, d_model]
            attn_mask: optional attention mask.
            key_padding_mask: [batch, seq_len] mask for padding positions.
            return_gate: whether to return gate weights.
            domain_prior: optional geometric prior over domains (base + specialists),
                         shape [num_domains] or [batch|1, num_domains].
            domain_mask: optional 0/1 (or bool) mask selecting which domains are allowed,
                         shape [num_domains] or [batch|1, num_domains].

        Returns:
            y: [batch, seq_len, d_model]
            gate_probs (optional): [batch, num_domains]
        """
        bsz, _, _ = x.size()

        # 1) Base output (always computed)
        base_out = self._run_base_mha(x, attn_mask, key_padding_mask)  # [b, s, d]

        # 2) Compute gate logits over domains (learned gate on pooled hidden)
        # ------------------------------------------------------------------
        # Compute gate weights
        #   - If domain_weights is provided, use it directly (fixed across layers)
        #   - Else fall back to per-layer learned gate logits (+ optional domain_prior)
        # ------------------------------------------------------------------

        mask: Optional[torch.Tensor] = None
        if domain_mask is not None:
            mask = domain_mask.to(x.device)
            if mask.dim() == 1:
                mask = mask.unsqueeze(0)
            # broadcast across batch
            if mask.size(0) == 1 and bsz > 1:
                mask = mask.expand(bsz, -1)

        if domain_weights is not None:
            # domain_weights is expected to be probabilities over domains
            gate = domain_weights.to(x.device)
            if gate.dim() == 1:
                gate = gate.unsqueeze(0)
            if gate.size(0) == 1 and bsz > 1:
                gate = gate.expand(bsz, -1)

            if mask is not None:
                gate = gate * mask.to(gate.dtype)

            gate = gate / (gate.sum(dim=-1, keepdim=True) + 1e-9)

        else:
            # 1) Compute learned gate logits from pooled hidden state
            pooled = self.pool_hidden(x)  # [batch, d_model]
            logits = self.gate(pooled)    # [batch, num_domains]

            # 2) Inject router prior if provided (log-space bias)
            if domain_prior is not None:
                prior = domain_prior.to(logits.device)
                if prior.dim() == 1:
                    prior = prior.unsqueeze(0)
                if prior.size(0) == 1 and bsz > 1:
                    prior = prior.expand(bsz, -1)
                prior = torch.clamp(prior, min=1e-9)
                logits = logits + torch.log(prior)

            # Apply hard mask if provided
            if mask is not None:
                logits = logits.masked_fill(mask == 0, float("-inf"))

            # Use sigmoid for independent blend strength, then renormalize
            gate_raw = torch.sigmoid(logits)
            if mask is not None:
                gate_raw = gate_raw * mask.to(gate_raw.dtype)

            gate = gate_raw / (gate_raw.sum(dim=-1, keepdim=True) + 1e-9)
# 3) Decide which specialists to EXECUTE (sparse selection)
        # Specialists are domains 1..N (domain 0 is base)
        selected_spec_indices: List[int] = []
        if mask is not None:
            # union across batch: run any specialist that is selected for at least one sample
            spec_mask_any = (mask[:, 1:] > 0).any(dim=0)  # [num_specialists]
            selected_spec_indices = spec_mask_any.nonzero(as_tuple=False).view(-1).tolist()
        else:
            # No explicit mask -> fall back to computing all specialists (dense)
            selected_spec_indices = list(range(len(self.spec_mhas)))

        # 4) Execute only selected specialists
        spec_indices_run, spec_outs = self._run_specialists(
            x,
            attn_mask,
            key_padding_mask,
            selected_indices=selected_spec_indices if selected_spec_indices else [],
        )

        # 5) Blend outputs
        g_base = gate[:, 0].view(bsz, 1, 1)        # [b, 1, 1]
        g_specs = gate[:, 1:]                      # [b, num_specialists]

        y = g_base * base_out                      # [b, s, d]
        for idx, spec_out in zip(spec_indices_run, spec_outs):
            g_i = g_specs[:, idx].view(bsz, 1, 1)  # idx indexes specialists (0..num_specialists-1)
            y = y + g_i * spec_out

        if return_gate:
            return y, gate
        else:
            return y, None
