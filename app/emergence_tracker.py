from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

from .router import RoutingResult, DomainMetrics


@dataclass
class SerializedDomainMetrics:
    """
    JSON-serializable version of DomainMetrics.
    """
    name: str
    similarity: float
    mahalanobis: float
    entropy: float
    support: float
    score: float


@dataclass
class SerializedRoutingResult:
    """
    JSON-serializable version of RoutingResult.
    """
    metrics: List[SerializedDomainMetrics]
    best_domain: Optional[str]
    is_unknown: bool
    reason: str


class EmergenceTracker:
    """
    EmergenceTracker logs prompts that appear to be outside the coverage
    of existing domains/specialists.

    Each call to `log_observation` appends a JSON line to the configured log
    file with:

      - timestamp: float (seconds since epoch)
      - prompt: original prompt text
      - completion: model answer
      - embedding: list[float] query embedding
      - routing: serialized RoutingResult (or a stub if no router)
      - specialists: list[str] of known specialists at the time
    """

    def __init__(self, log_path: Path) -> None:
        self.log_path = log_path
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    # Serialization helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _serialize_routing_result(result: RoutingResult) -> SerializedRoutingResult:
        """
        Convert a RoutingResult (with torch-friendly floats) into
        plain Python types suitable for JSON encoding.
        """
        metrics = [
            SerializedDomainMetrics(
                name=m.name,
                similarity=float(m.similarity),
                mahalanobis=float(m.mahalanobis),
                entropy=float(m.entropy),
                support=float(m.support),
                score=float(m.score),
            )
            for m in result.metrics
        ]

        return SerializedRoutingResult(
            metrics=metrics,
            best_domain=result.best_domain,
            is_unknown=result.is_unknown,
            reason=result.reason,
        )

    # ------------------------------------------------------------------ #
    # Public logging API
    # ------------------------------------------------------------------ #

    def log_observation(
        self,
        prompt: str,
        completion: str,
        embedding: torch.Tensor,
        routing_result: Optional[RoutingResult],
        specialists: List[str],
    ) -> None:
        """
        Append a single observation to the JSONL log.

        Typically called when:
          - There are no specialists; or
          - The router flags the query as "unknown_domain".
        """
        # Flatten embedding to a simple list[float] on CPU.
        embedding_cpu = embedding.detach().cpu().view(-1)
        emb_list = embedding_cpu.tolist()

        if routing_result is not None:
            serialized_rr = self._serialize_routing_result(routing_result)
            rr_dict: Dict[str, Any] = {
                "metrics": [asdict(m) for m in serialized_rr.metrics],
                "best_domain": serialized_rr.best_domain,
                "is_unknown": serialized_rr.is_unknown,
                "reason": serialized_rr.reason,
            }
        else:
            rr_dict = {
                "metrics": [],
                "best_domain": None,
                "is_unknown": True,
                "reason": "no_router",
            }

        record: Dict[str, Any] = {
            "timestamp": time.time(),
            "prompt": prompt,
            "completion": completion,
            "embedding": emb_list,
            "routing": rr_dict,
            "specialists": list(specialists),
        }

        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
