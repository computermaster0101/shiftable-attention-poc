from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import torch

from . import config


@dataclass
class DomainStats:
    name: str
    centroid: torch.Tensor        # [dim]
    cov_diag: torch.Tensor        # [dim]  (diagonal of Σ_k)


def _load_domain_stats(path: Path, device: torch.device) -> Tuple[int, List[DomainStats]]:
    """
    Load domain statistics from the JSON file produced by ModelManager._build_domain_stats.

    Returns:
        dim: embedding dimension.
        stats: list of DomainStats for each domain.
    """
    if not path.exists():
        raise FileNotFoundError(f"Domain stats file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)

    dim = int(obj["dim"])
    domains_obj = obj["domains"]

    stats: List[DomainStats] = []
    for name, info in domains_obj.items():
        c = torch.tensor(info["centroid"], dtype=torch.float32, device=device)
        v = torch.tensor(info["var_diag"], dtype=torch.float32, device=device)
        if c.numel() != dim or v.numel() != dim:
            continue
        stats.append(DomainStats(name=name, centroid=c, cov_diag=v))

    return dim, stats


def compute_centroid_angle(cj: torch.Tensor, ck: torch.Tensor) -> float:
    """
    Eq. (13): θ_jk = arccos( c_j^T c_k / (||c_j|| ||c_k||) ).

    Returns:
        Angle in radians.
    """
    eps = 1e-8
    cj_norm = cj.norm(p=2).clamp_min(eps)
    ck_norm = ck.norm(p=2).clamp_min(eps)
    dot = float(torch.dot(cj, ck).item())
    cos_theta = dot / float((cj_norm * ck_norm).item())
    cos_theta = max(min(cos_theta, 1.0), -1.0)
    return math.acos(cos_theta)


def compute_centroid_angle_matrix(stats: List[DomainStats]) -> torch.Tensor:
    """
    Compute the full matrix of centroid angles (in radians) between domains.
    """
    n = len(stats)
    angles = torch.zeros(n, n, dtype=torch.float32)
    for i in range(n):
        for j in range(n):
            if i == j:
                angles[i, j] = 0.0
            elif j < i:
                angles[i, j] = angles[j, i]
            else:
                a = compute_centroid_angle(stats[i].centroid, stats[j].centroid)
                angles[i, j] = angles[j, i] = float(a)
    return angles


def compute_subspace_basis(cov_diag: torch.Tensor, r: int) -> torch.Tensor:
    """
    Construct an r-dimensional domain subspace S_k from the diagonal covariance.

    In the full GRCLM formulation, Σ_k is an arbitrary covariance with
    eigendecomposition Σ_k = U_k Λ_k U_k^T and S_k is spanned by the top-r
    eigenvectors. In this POC, we only track the diagonal of Σ_k, so we
    approximate Σ_k as diag(cov_diag) and choose the r coordinates with
    largest variance as the principal directions.

    Returns:
        U_k^(r): [dim, r] orthonormal basis matrix.
    """
    dim = cov_diag.numel()
    r = max(1, min(r, dim))

    # Indices of the r largest variances.
    _, idx = torch.topk(cov_diag, k=r, largest=True, sorted=True)
    basis = torch.zeros(dim, r, dtype=torch.float32, device=cov_diag.device)
    for j, d in enumerate(idx.tolist()):
        basis[d, j] = 1.0  # standard basis vector e_d

    # Columns are already orthonormal.
    return basis


def compute_principal_angles(Uj: torch.Tensor, Uk: torch.Tensor) -> torch.Tensor:
    """
    Eq. (15)–(16): principal angles between two r-dimensional subspaces.

    Given orthonormal bases U_j^(r) and U_k^(r) (each [dim, r]), compute
        (U_j^(r))^T U_k^(r) = P Σ Q^T
    and set ϕ_ℓ = arccos(σ_ℓ), where σ_ℓ ∈ [0, 1] are singular values.

    Returns:
        1D tensor of shape [r] with principal angles in radians.
    """
    # SVD of the r×r matrix U_j^T U_k.
    M = Uj.t().mm(Uk)  # [r, r]
    # torch.linalg.svd returns U, S, Vh with S sorted descending.
    _, S, _ = torch.linalg.svd(M)
    S = torch.clamp(S, min=0.0, max=1.0)
    angles = torch.acos(S)
    return angles


def main(top_r: int = 4) -> None:
    """
    Simple CLI entry point to inspect domain topology:

      • Prints centroid angles (degrees) between all domains.
      • Prints the smallest principal angle between domain subspaces
        based on the top-r variance directions.

    Usage:
        python -m app.domain_topology
        python -m app.domain_topology 8
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dim, stats = _load_domain_stats(config.DOMAIN_STATS_PATH, device=device)
    if not stats:
        print(f"No domains found in stats file: {config.DOMAIN_STATS_PATH}")
        return

    names = [s.name for s in stats]
    print(f"Loaded {len(stats)} domains with dim={dim}: {', '.join(names)}")

    # Centroid angles.
    centroid_angles = compute_centroid_angle_matrix(stats)
    print("\nCentroid angles (degrees):")
    for i, ni in enumerate(names):
        row = []
        for j, nj in enumerate(names):
            deg = float(centroid_angles[i, j].item()) * 180.0 / math.pi
            row.append(f"{deg:6.2f}")
        print(f"{ni:>12}: " + " ".join(row))

    # Principal angles between domain subspaces.
    print(f"\nPrincipal angles (degrees), r={top_r}:")
    bases: Dict[str, torch.Tensor] = {
        s.name: compute_subspace_basis(s.cov_diag, r=top_r) for s in stats
    }
    for i, ni in enumerate(names):
        for j in range(i + 1, len(names)):
            nj = names[j]
            angles = compute_principal_angles(bases[ni], bases[nj])
            smallest = float(angles.min().item()) * 180.0 / math.pi
            print(f"{ni:>12} vs {nj:>12}: smallest angle = {smallest:6.2f}°")


if __name__ == "__main__":
    import sys

    r = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    main(top_r=r)
