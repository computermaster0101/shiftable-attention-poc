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
    Per-domain metrics computed for a single query embedding q.

    Fields:
        domain:
            Domain name / specialist key.
        similarity:
            Cosine similarity s_k(q) between query embedding and centroid.
        mahalanobis:
            Mahalanobis distance m_k(q). Uses:
              - full covariance via Cholesky factor if available ("chol")
              - diagonal variance fallback if available ("diag")
              - large sentinel if neither available ("none")
        mahalanobis_source:
            One of {"chol", "diag", "none"} indicating how mahalanobis was computed.
        entropy:
            An entropy-like uncertainty term H_k(q). In the GRCLM paper this
            corresponds to predictive entropy for specialist k. In this POC
            we approximate H_k(q) from the cosine similarity as a Bernoulli
            entropy:
                p_k = (s_k + 1) / 2
                H_k(q) = -[p_k log p_k + (1 - p_k) log(1 - p_k)]
            which is 0 when similarity is extreme and maximal around 0.
        support:
            Support ratio r_k over a sliding routing window.
        score:
            Composite routing score R_k(q) = α s - β m - γ H + δ r
    """

    domain: str
    similarity: float
    mahalanobis: float
    mahalanobis_source: str
    entropy: float
    support: float
    score: float


@dataclass
class RoutingResult:
    """
    Router output for a query.

    Fields:
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
    DomainRouter computes geometric routing signals and selects (or rejects)
    a best matching domain.

    Metrics implemented:
        similarity:
            Cosine similarity between the query embedding and the
            domain centroid.

        mahalanobis:
            Mahalanobis distance computed as:
                m^2 = (q - c)^T Σ^{-1} (q - c)
            Using a Cholesky factor L of Σ (Σ = L L^T):
                m^2 = || L^{-1}(q - c) ||^2

            Fallback: diagonal approximation using var_diag if no cov/cov_chol exists.

        entropy:
            Proxy uncertainty term derived from similarity (Bernoulli entropy).

        support:
            Support ratio computed from a sliding window over recent routed queries.

        score:
            Composite routing score:
                R_k(q) = α s_k(q) - β m_k(q) - γ H_k(q) + δ r_k
    """

    def __init__(self, stats_path: Optional[str] = None) -> None:
        self.stats_path = Path(
            stats_path if stats_path is not None else getattr(config, "DOMAIN_STATS_PATH", "domain_stats.json")
        )

        # Domain stats
        self.centroids: Dict[str, torch.Tensor] = {}
        self.cov_chol: Dict[str, torch.Tensor] = {}      # Lower-triangular Cholesky factor per domain
        self.inv_var_diag: Dict[str, torch.Tensor] = {}  # Diagonal fallback (1/var)
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
        Load domain stats from JSON.

        Supports both formats:
        - {"dim": d, "domains": {...}}
        - {...} where root is already the domain map

        Each domain entry may include:
            - centroid: [d]
            - cov_chol: [[d x d]]  (preferred)
            - cov: [[d x d]]       (fallback; we compute chol)
            - var_diag: [d]        (diagonal fallback)
        """
        self.centroids.clear()
        self.cov_chol.clear()
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

        if isinstance(obj, dict) and "domains" in obj and isinstance(obj["domains"], dict):
            domain_map = obj["domains"]
        elif isinstance(obj, dict):
            domain_map = obj
        else:
            return

        # Optional configured covariance jitter (helps if cov is nearly singular)
        cov_jitter = float(getattr(config, "ROUTER_COV_LAMBDA", 1e-6))

        for name, st in domain_map.items():
            if not isinstance(st, dict):
                continue

            centroid = st.get("centroid", None)
            if centroid is None:
                continue

            try:
                c = torch.tensor(centroid, dtype=torch.float32)
            except Exception:
                continue

            # Load covariance via Cholesky if available
            L = None

            cov_chol = st.get("cov_chol", None)
            if cov_chol is not None:
                try:
                    L = torch.tensor(cov_chol, dtype=torch.float32)
                except Exception:
                    L = None

            if L is None:
                cov = st.get("cov", None)
                if cov is not None:
                    try:
                        cov_t = torch.tensor(cov, dtype=torch.float32)
                        # Ensure symmetry-ish (guard against tiny serialization noise)
                        cov_t = 0.5 * (cov_t + cov_t.transpose(0, 1))

                        # Add jitter to diagonal to guarantee PD if needed
                        cov_t = cov_t + (cov_jitter * torch.eye(cov_t.shape[0], dtype=cov_t.dtype))

                        # Try Cholesky; if it fails, increase jitter a few times
                        for k in range(5):
                            try:
                                L = torch.linalg.cholesky(cov_t)
                                break
                            except Exception:
                                cov_t = cov_t + ((10.0 ** k) * cov_jitter * torch.eye(cov_t.shape[0], dtype=cov_t.dtype))
                        # If still None, we fall back to diagonal
                    except Exception:
                        L = None

            # Diagonal fallback
            v = st.get("var_diag", None)
            inv_v = None
            if v is not None:
                try:
                    v_t = torch.tensor(v, dtype=torch.float32)
                    inv_v = 1.0 / torch.clamp(v_t, min=getattr(config, "ROUTER_MIN_VAR", 1e-8))
                except Exception:
                    inv_v = None

            self.centroids[name] = c
            if L is not None:
                self.cov_chol[name] = L
            if inv_v is not None:
                self.inv_var_diag[name] = inv_v

        if self.centroids:
            self.dim = next(iter(self.centroids.values())).shape[0]

        # Reset support state whenever stats are rebuilt.
        if self.support_window_size > 0:
            self.support_window = []
            self.support_counts = {name: 0 for name in self.centroids.keys()}

    def reload(self) -> None:
        self._load_stats()

    # ------------------------------------------------------------------ #
    # Metric helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _cosine_similarity(q: torch.Tensor, c: torch.Tensor) -> float:
        qn = q / (q.norm() + 1e-12)
        cn = c / (c.norm() + 1e-12)
        return float(torch.dot(qn, cn).item())

    @staticmethod
    def _entropy_from_similarity(sim: float) -> float:
        # Map cosine similarity [-1, 1] -> probability [0, 1]
        p = (sim + 1.0) / 2.0
        p = min(max(p, 1e-8), 1.0 - 1e-8)
        return -float(p * math.log(p) + (1.0 - p) * math.log(1.0 - p))

    def _get_support_ratio(self, name: str) -> float:
        if self.support_window_size <= 0:
            return 0.0
        denom = sum(self.support_counts.values())
        if denom <= 0:
            return 0.0
        return float(self.support_counts.get(name, 0) / denom)

    def _update_support(self, name: str) -> None:
        if self.support_window_size <= 0:
            return

        self.support_window.append(name)
        self.support_counts[name] = self.support_counts.get(name, 0) + 1

        # Pop from left when window exceeds size
        while len(self.support_window) > self.support_window_size:
            old = self.support_window.pop(0)
            self.support_counts[old] = max(0, self.support_counts.get(old, 0) - 1)

    def _mahalanobis_distance(self, name: str, diff: torch.Tensor) -> (float, str):
        """
        Compute Mahalanobis distance for a given domain from diff = (q - c).

        Returns:
            (mahal, source) where source in {"chol", "diag", "none"}.
        """
        # Preferred: full covariance via Cholesky solve
        L = self.cov_chol.get(name, None)
        if L is not None:
            # Solve L z = diff (lower triangular), then m^2 = ||z||^2
            # diff shape [d]
            try:
                z = torch.linalg.solve_triangular(L, diff.unsqueeze(-1), upper=False)
                m2 = float((z.squeeze(-1) ** 2).sum().item())
                m2 = max(m2, 0.0)
                return math.sqrt(m2), "chol"
            except Exception:
                # fall through to diag
                pass

        inv_var = self.inv_var_diag.get(name, None)
        if inv_var is not None:
            try:
                m2 = float(((diff * diff) * inv_var).sum().item())
                m2 = max(m2, 0.0)
                return math.sqrt(m2), "diag"
            except Exception:
                pass

        # No covariance info available -> make distance huge so it won't be selected
        return float(getattr(config, "ROUTER_NO_COV_PENALTY", 1e9)), "none"

    # ------------------------------------------------------------------ #
    # Main routing
    # ------------------------------------------------------------------ #

    def route(self, q: torch.Tensor) -> RoutingResult:
        """
        Route a query embedding q to the best matching domain, or declare unknown.

        q should be a 1D tensor [d].
        """
        metrics: List[DomainMetrics] = []

        # Compute per-domain metrics
        for name, c in self.centroids.items():
            sim = self._cosine_similarity(q, c)

            diff = q - c
            mahal, msrc = self._mahalanobis_distance(name, diff)

            entropy = self._entropy_from_similarity(sim)
            support = self._get_support_ratio(name)

            score = (
                config.ROUTER_ALPHA * sim
                - config.ROUTER_BETA * mahal
                - config.ROUTER_GAMMA * entropy
                + config.ROUTER_DELTA * support
            )

            metrics.append(
                DomainMetrics(
                    domain=name,
                    similarity=sim,
                    mahalanobis=mahal,
                    mahalanobis_source=msrc,
                    entropy=entropy,
                    support=support,
                    score=score,
                )
            )

        # If no stats, treat as unknown
        if not metrics:
            return RoutingResult(
                metrics=[],
                best_domain=None,
                is_unknown=True,
                reason="no_domain_stats",
            )

        # Pick best domain by composite score
        metrics.sort(key=lambda m: m.score, reverse=True)
        best_domain = metrics[0].domain if metrics else None

        # Unknown-domain checks
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
            entropy_term = min_entropy > float(max_entropy_thr)

        # Optional support term
        support_term = False
        min_support_thr = getattr(config, "ROUTER_UNKNOWN_MIN_SUPPORT", None)
        if min_support_thr is not None:
            support_term = max_support < float(min_support_thr)

        # Conservative disjunction: unknown if any strong signal says "novel"
        is_unknown = (sim_term or dist_term or entropy_term or support_term)

        # Don't declare unknown in "general-only" phase unless explicitly allowed
        if phase_general_only and not getattr(config, "ROUTER_ALLOW_UNKNOWN_WITH_ONLY_GENERAL", False):
            is_unknown = False

        # Update support history only for routed (non-unknown) queries.
        if not is_unknown and best_domain is not None:
            self._update_support(best_domain)

        reason = "ok" if not is_unknown else "unknown_domain"

        return RoutingResult(
            metrics=metrics,
            best_domain=best_domain,
            is_unknown=is_unknown,
            reason=reason,
        )
