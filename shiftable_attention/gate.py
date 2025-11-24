import torch
import torch.nn as nn
import torch.nn.functional as F


class DomainGate(nn.Module):
    """
    Domain-gating network.

    Maps a pooled hidden state -> logits over domains:
      - domain 0: base attention
      - domains 1..N: specialist attention branches

    Args:
        d_model: dimensionality of the hidden representation.
        num_domains: number of domains (1 base + N specialists).
        hidden_dim: hidden size of the gate MLP (defaults to d_model).
    """
    def __init__(self, d_model: int, num_domains: int = 2, hidden_dim: int | None = None):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = d_model

        self.fc1 = nn.Linear(d_model, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_domains)

    def forward(self, pooled_x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pooled_x: [batch, d_model] pooled hidden states.

        Returns:
            logits: [batch, num_domains] (apply softmax over last dim).
        """
        x = self.fc1(pooled_x)
        x = F.relu(x)
        logits = self.fc2(x)
        return logits
