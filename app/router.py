from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import torch

from . import config


@dataclass
class DomainMetrics:
    """
    Per-domain routing metrics used by the geometric router.

    Attributes:
        name:
            Domain / specialist name.

        similarity:
            Cosine similarity s_k(q) between the query embedding and the
            domain centroid. This is always in [-1, 1].

        mahalanobis:
            Mahalanobis distance m_k(q) using a diagonal covariance
            approximation:
                m_k(q)^2 = (q - c_k)^T Σ_k^{-1} (q - c_k)
            where Σ_k is represented by its diagonal var_diag.

        entropy:
            An entropy-like uncertainty term H_k(q). In the GRCLM paper this
            corresponds to predictive entropy for specialist k. In this POC
            we approximate H_k(q) from the cosine similarity as a Bernoulli
            entropy:
                p_k = (s_k + 1) / 2
                H_k(q) = -[p_k log p_k + (1 - p_k) log(1 - p_k)]
            which is 0 when similarity is extreme and maximal around 0.

        support:
            Support ratio r_k computed from a sliding window over recent
            routed queries:
                r_k = N_k / sum_j N_j
            where N_k is the count of queries routed to domain k in the
            current window.

        score:
            Composite routing score R_k(q):
                R_k(q) = α s_k(q)
                          - β m_k(q)
                          - γ H_k(q)
                          + δ r_k
            where α, β, γ, δ come from app.config.
    """

    name: str
    similarity: float
    mahalanobis: float
    entropy: float
    support: float
    score: float


@dataclass
class RoutingResult:
    """
    Full routing result for a query.

    Attributes:
        metrics:
            Per-domain metrics (one DomainMetrics per known domain).
        best_domain:
            Name of the highest-scoring domain, or None if there were no
            domain stats available.
        is_unknown:
            True if the router decides this query is outside all known
            domains based on the unknown thresholds.
        reason:
            Short reason string, one of:
                - "ok"
                - "no_domain_stats"
                - "unknown_domain"
    """

    metrics: List[DomainMetrics]
    best_domain: Optional[str]
    is_unknown: bool
    reason: str


class DomainRouter:
    """
    Geometric router over domains/specialists.

    It combines:
      - centroid-based cosine similarity
      - diagonal Mahalanobis distance
      - an entropy-like term derived from similarity
      - a support ratio based on recent routing history

    to compute a composite score R_k(q) per domain, and also checks for
    "unknown domain" conditions as described in the GRCLM implementation
    document.
    """

    def __init__(self, stats_path: Path, device: torch.device) -> None:
        """
        Args:
            stats_path:
                Path to a JSON file containing domain statistics produced by
                ModelManager._build_domain_stats().

                The expected structure is:

                {
                  "dim": <int>,                # optional
                  "domains": {
                    "<domain_name>": {
                      "count": <int>,
                      "centroid": [float, ...],
                      "var_diag": [float, ...]
                    },
                    ...
                  }
                }

                If "dim" is not present, the dimension will be inferred from
                the centroid length.

            device:
                torch.device where routing tensors should live.
        """
        self.stats_path = Path(stats_path)
        self.device = device

        # Domain centroids and inverse diagonal variances
        self.centroids: Dict[str, torch.Tensor] = {}
        self.inv_var_diag: Dict[str, torch.Tensor] = {}
        self.dim: Optional[int] = None

        # Sliding-window support tracking for r_k (Eq. 6).
        self.support_window_size: int = getattr(config, "ROUTER_SUPPORT_WINDOW", 0)
        self.support_window: List[str] = []
        self.support_counts: Dict[str, int] = {}

        self._load_stats()

    # ------------------------------------------------------------------ #
    # Domain stats loading
    # ------------------------------------------------------------------ #

    def _load_stats(self) -> None:
        """
        Load domain centroids and diagonal variances from stats_path.

        This is designed to be robust to minor format changes:
        - If "dim" is missing, we infer it from the first centroid.
        - If the root object is already the domain map, we handle that too.
        """
        self.centroids.clear()
        self.inv_var_diag.clear()
        self.dim = None

        if not self.stats_path.exists():
            return

        try:
            with open(self.stats_path, "r", encoding="utf-8") as f:
                obj = json.load(f)
        except Exception:
            # If the stats file is corrupt, treat as empty.
            return

        # Support both {"domains": {...}} and {...} directly.
        domains_obj = obj.get("domains", obj)

        for name, stats in domains_obj.items():
            if not isinstance(stats, dict):
                continue
            centroid = stats.get("centroid")
            var_diag = stats.get("var_diag")
            if centroid is None or var_diag is None:
                continue

            c = torch.tensor(centroid, dtype=torch.float32, device=self.device)
            v = torch.tensor(var_diag, dtype=torch.float32, device=self.device)

            if c.ndim != 1 or v.ndim != 1 or c.shape[0] != v.shape[0]:
                continue

            inv_v = 1.0 / torch.clamp(v, min=config.ROUTER_MIN_VAR)
            self.centroids[name] = c
            self.inv_var_diag[name] = inv_v

        if self.centroids:
            # Infer dimension from the first centroid
            self.dim = next(iter(self.centroids.values())).shape[0]

        # Reset support state whenever stats are rebuilt.
        if self.support_window_size > 0:
            self.support_window = []
            self.support_counts = {name: 0 for name in self.centroids.keys()}

    def reload(self) -> None:
        """
        Reload domain stats from disk.

        Call this after ModelManager rebuilds domain stats (e.g. when
        specialists are added/removed).
        """
        self._load_stats()

    # ------------------------------------------------------------------ #
    # Support ratio helpers (Eq. 6)
    # ------------------------------------------------------------------ #

    def _get_support_ratio(self, domain: str) -> float:
        """
        Compute r_k = N_k / sum_j N_j for the given domain based on the
        current sliding window. Returns 0.0 if there is no history yet.
        """
        if self.support_window_size <= 0:
            return 0.0
        total = sum(self.support_counts.values())
        if total <= 0:
            return 0.0
        return float(self.support_counts.get(domain, 0)) / float(total)

    def _update_support(self, domain: str) -> None:
        """
        Update the sliding-window support counts after routing a query to
        `domain`.
        """
        if self.support_window_size <= 0:
            return

        self.support_window.append(domain)
        self.support_counts.setdefault(domain, 0)
        self.support_counts[domain] += 1

        # Enforce the window size by removing the oldest entry.
        if len(self.support_window) > self.support_window_size:
            oldest = self.support_window.pop(0)
            if oldest in self.support_counts:
                self.support_counts[oldest] = max(0, self.support_counts[oldest] - 1)

    # ------------------------------------------------------------------ #
    # Entropy approximation
    # ------------------------------------------------------------------ #

    @staticmethod
    def _entropy_from_similarity(similarity: float) -> float:
        """
        Approximate an entropy-like uncertainty H_k(q) from cosine
        similarity.

        We map cosine similarity s ∈ [-1, 1] to a Bernoulli probability
        p ∈ (0, 1) via:

            p = (s + 1) / 2

        and then compute the Bernoulli entropy:

            H(p) = -[p log p + (1 - p) log(1 - p)]

        This is maximal when s ≈ 0 (uncertain / ambiguous) and minimal when
        s ≈ ±1 (highly certain).

        This is a pragmatic stand-in for true predictive entropy of the
        specialist, which would require token-level probabilities.
        """
        # Map to [0, 1]
        p = 0.5 * (similarity + 1.0)
        # Clamp away from 0 and 1 for numerical stability.
        eps = getattr(config, "ROUTER_EPS", 1e-8)
        p = min(max(p, eps), 1.0 - eps)
        return -(
            p * math.log(p)
            + (1.0 - p) * math.log(1.0 - p)
        )

    # ------------------------------------------------------------------ #
    # Routing (Definitions 2 & 3)
    # ------------------------------------------------------------------ #

    def route(self, embedding: torch.Tensor) -> RoutingResult:
        """
        Compute routing metrics for the given query embedding.

        Args:
            embedding:
                1D tensor of shape [dim] representing the query in the same
                embedding space as the domain centroids.

        Returns:
            RoutingResult with per-domain metrics, the best_domain, and an
            unknown-domain flag.
        """
        if self.dim is None or not self.centroids:
            return RoutingResult(
                metrics=[],
                best_domain=None,
                is_unknown=True,
                reason="no_domain_stats",
            )

        if embedding.ndim != 1:
            raise ValueError(
                f"DomainRouter.route expects a 1D embedding, got shape {tuple(embedding.shape)}"
            )

        q = embedding.to(self.device)
        if q.shape[0] != self.dim:
            raise ValueError(
                f"DomainRouter.route embedding dim mismatch: expected {self.dim}, got {q.shape[0]}"
            )

        # Normalize query for cosine similarity
        q_norm = q.norm(p=2).clamp_min(config.ROUTER_EPS)
        q_hat = q / q_norm

        metrics: List[DomainMetrics] = []
        best_domain: Optional[str] = None
        best_score: float = -float("inf")

        # Compute per-domain metrics
        for name, c in self.centroids.items():
            c_norm = c.norm(p=2).clamp_min(config.ROUTER_EPS)
            c_hat = c / c_norm

            # Cosine similarity s_k(q)
            sim = float(torch.dot(q_hat, c_hat).item())

            # Mahalanobis distance m_k(q) with diagonal covariance
            inv_var = self.inv_var_diag[name]
            diff = q - c
            m2 = float(((diff * diff) * inv_var).sum().item())
            if m2 < 0.0:
                m2 = 0.0
            mahal = math.sqrt(m2)

            # Entropy-like uncertainty H_k(q)
            entropy = self._entropy_from_similarity(sim)

            # Support ratio r_k
            support = self._get_support_ratio(name)

            # Composite score R_k(q)
            score = (
                config.ROUTER_ALPHA * sim
                - config.ROUTER_BETA * mahal
                - config.ROUTER_GAMMA * entropy
                + config.ROUTER_DELTA * support
            )

            dm = DomainMetrics(
                name=name,
                similarity=float(sim),
                mahalanobis=float(mahal),
                entropy=float(entropy),
                support=float(support),
                score=float(score),
            )
            metrics.append(dm)

            if score > best_score:
                best_score = score
                best_domain = name

        if not metrics:
            return RoutingResult(
                metrics=[],
                best_domain=None,
                is_unknown=True,
                reason="no_domain_stats",
            )

        # Unknown-domain checks (Definition 3 style)
        max_sim = max(m.similarity for m in metrics)
        min_dist = min(m.mahalanobis for m in metrics)
        min_entropy = min(m.entropy for m in metrics)
        max_support = max(m.support for m in metrics)

        phase_general_only = (len(self.centroids) == 1 and "general" in self.centroids)

        # Similarity + distance terms
        sim_term = max_sim < config.ROUTER_UNKNOWN_MAX_SIM
        dist_term = min_dist > config.ROUTER_UNKNOWN_MIN_DIST

        # Optional entropy term
        entropy_term = False
        max_entropy_thr = getattr(config, "ROUTER_UNKNOWN_MAX_ENTROPY", None)
        if max_entropy_thr is not None:
            entropy_term = min_entropy > max_entropy_thr

        # Optional support term, with warm-up + only when multiple domains
        support_term = False
        min_support_thr = getattr(config, "ROUTER_UNKNOWN_MIN_SUPPORT", 0.0)
        if (not phase_general_only) and min_support_thr > 0.0:
            total_support = sum(self.support_counts.values())
            warmup_min_events = getattr(config, "ROUTER_SUPPORT_WARMUP_MIN_EVENTS", 0)
            if total_support >= warmup_min_events:
                support_term = max_support < min_support_thr
            else:
                support_term = False  # disable support-based unknown during warm-up

        if phase_general_only:
            # Phase 0: general-only. Everything is unknown unless very close to training data.
            # Here we deliberately ignore support and (optionally) similarity.
            is_unknown = dist_term or entropy_term
        else:
            # Phase 1+: multi-domain. Full composite logic.
            is_unknown = sim_term or dist_term or entropy_term or support_term

        # Update support history only for routed (non-unknown) queries.
        if not is_unknown and best_domain is not None:
            self._update_support(best_domain)

        reason = "ok"
        if is_unknown:
            reason = "unknown_domain"

        return RoutingResult(
            metrics=metrics,
            best_domain=best_domain,
            is_unknown=is_unknown,
            reason=reason,
        )
