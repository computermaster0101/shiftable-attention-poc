#!/usr/bin/env python3
"""
Export GRCLM / Shiftable-Attention logs + domain stats into a TensorBoard Embedding Projector workspace.

Reads (by default):
  - shiftable_project/outputs/shiftable/domain_stats.json          (domain centroids, counts, covariances)
  - logs/input_output_log.jsonl                                   (per-prompt embeddings + routing metrics)
  - shiftable_project/outputs/shiftable/emergence_log.jsonl        (optional extra embeddings)

Writes into --out_dir:
  - query_embeddings.tsv
  - query_metadata.tsv
  - domain_centroids.tsv
  - domain_centroids_metadata.tsv
  - projector_config.pbtxt

Run:
  python export_tensorboard_projector.py --project_root /path/to/shiftable-attention-poc
  tensorboard --logdir /path/to/shiftable-attention-poc/tensorboard_projector --port 6006
"""

from __future__ import annotations
import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

def iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                # Skip malformed lines rather than failing the whole export.
                continue

def safe_str(x: Any) -> str:
    if x is None:
        return ""
    s = str(x)
    # TSV metadata hates raw tabs/newlines
    return s.replace("\t", " ").replace("\n", " ").replace("\r", " ")

def top_domain_from_routing(routing: Any) -> Tuple[str, float]:
    """
    Returns (domain_name, score) from the routing.metrics list if present.
    Falls back to ("", nan).
    """
    if not isinstance(routing, dict):
        return ("", float("nan"))
    metrics = routing.get("metrics")
    if not isinstance(metrics, list) or not metrics:
        return ("", float("nan"))
    best = None
    best_score = None
    for m in metrics:
        if not isinstance(m, dict):
            continue
        sc = m.get("score")
        name = m.get("name", "")
        if sc is None:
            continue
        try:
            scf = float(sc)
        except Exception:
            continue
        if best is None or scf > best_score:
            best = name
            best_score = scf
    if best is None:
        return ("", float("nan"))
    return (safe_str(best), float(best_score))

def write_tsv_vectors(path: Path, vectors: List[List[float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        for v in vectors:
            f.write("\t".join(f"{float(x):.10g}" for x in v) + "\n")

def write_tsv_rows(path: Path, header: List[str], rows: List[List[Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t", quoting=csv.QUOTE_MINIMAL)
        w.writerow(header)
        for r in rows:
            w.writerow([safe_str(x) for x in r])

def write_projector_config(path: Path,
                           embeddings: List[Tuple[str, str]]) -> None:
    """
    embeddings: list of (tensor_path, metadata_path)
    """
    lines = []
    for tensor_path, meta_path in embeddings:
        lines.append("embeddings {")
        lines.append(f'  tensor_path: "{tensor_path}"')
        lines.append(f'  metadata_path: "{meta_path}"')
        lines.append("}")
        lines.append("")
    path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project_root", type=str, default=".",
                    help="Path to repo root (contains shiftable_project/ and logs/).")
    ap.add_argument("--out_dir", type=str, default="tensorboard_projector",
                    help="Output directory for projector workspace (relative to project_root unless absolute).")

    ap.add_argument("--domain_stats", type=str, default="shiftable_project/outputs/shiftable/domain_stats.json",
                    help="Path to domain_stats.json (relative to project_root unless absolute).")
    ap.add_argument("--input_log", type=str, default="logs/input_output_log.jsonl",
                    help="Path to input_output_log.jsonl (relative to project_root unless absolute).")
    ap.add_argument("--emergence_log", type=str, default="shiftable_project/outputs/shiftable/emergence_log.jsonl",
                    help="Optional extra JSONL to include (relative to project_root unless absolute).")
    ap.add_argument("--max_rows", type=int, default=50000,
                    help="Max number of embedding rows to export (protects you from gigantic logs).")
    args = ap.parse_args()

    root = Path(args.project_root).expanduser().resolve()

    def resolve(p: str) -> Path:
        pp = Path(p).expanduser()
        return pp if pp.is_absolute() else (root / pp)

    out_dir = resolve(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---------- Domain centroids ----------
    domain_stats_path = resolve(args.domain_stats)
    if not domain_stats_path.exists():
        raise FileNotFoundError(f"domain_stats.json not found at: {domain_stats_path}")

    ds = json.loads(domain_stats_path.read_text(encoding="utf-8"))
    dim = int(ds.get("dim", 0) or 0)
    domains = ds.get("domains", {})
    if not isinstance(domains, dict) or not domains:
        raise ValueError("domain_stats.json missing or empty 'domains' dict")

    centroid_vectors: List[List[float]] = []
    centroid_meta_rows: List[List[Any]] = []
    for name, info in domains.items():
        if not isinstance(info, dict):
            continue
        c = info.get("centroid")
        if not isinstance(c, list):
            continue
        if dim and len(c) != dim:
            # still export, but warn in metadata
            pass
        centroid_vectors.append([float(x) for x in c])
        centroid_meta_rows.append([
            name,
            info.get("count", ""),
            info.get("cov_lambda", ""),
        ])

    write_tsv_vectors(out_dir / "domain_centroids.tsv", centroid_vectors)
    write_tsv_rows(
        out_dir / "domain_centroids_metadata.tsv",
        header=["domain", "count", "cov_lambda"],
        rows=centroid_meta_rows,
    )

    # ---------- Query embeddings from logs ----------
    vectors: List[List[float]] = []
    meta_rows: List[List[Any]] = []

    def ingest_jsonl(path: Path, source_label: str) -> None:
        nonlocal vectors, meta_rows
        if not path.exists():
            return
        for obj in iter_jsonl(path):
            if len(vectors) >= args.max_rows:
                break
            emb = obj.get("embedding")
            if not isinstance(emb, list):
                continue
            try:
                v = [float(x) for x in emb]
            except Exception:
                continue
            if dim and len(v) != dim:
                # Skip mismatched rows (keeps projector happy)
                continue

            routing = obj.get("routing")
            top_dom, top_score = top_domain_from_routing(routing)

            # Try to pull a couple of useful metrics from the best domain entry
            best_sim = ""
            best_mah = ""
            best_entropy = ""
            if isinstance(routing, dict) and isinstance(routing.get("metrics"), list):
                for m in routing["metrics"]:
                    if isinstance(m, dict) and safe_str(m.get("name")) == top_dom:
                        best_sim = m.get("similarity", "")
                        best_mah = m.get("mahalanobis", "")
                        best_entropy = m.get("entropy", "")
                        break

            vectors.append(v)
            meta_rows.append([
                obj.get("timestamp", ""),
                source_label,
                obj.get("is_unknown_domain", routing.get("is_unknown") if isinstance(routing, dict) else ""),
                top_dom,
                top_score,
                best_sim,
                best_mah,
                best_entropy,
                safe_str(obj.get("prompt", ""))[:500],  # keep metadata compact
            ])

    ingest_jsonl(resolve(args.input_log), "input_output_log")
    ingest_jsonl(resolve(args.emergence_log), "emergence_log")

    if not vectors:
        raise ValueError("No embeddings found in provided logs. Check your log paths and JSONL format.")

    write_tsv_vectors(out_dir / "query_embeddings.tsv", vectors)
    write_tsv_rows(
        out_dir / "query_metadata.tsv",
        header=["timestamp", "source", "is_unknown", "top_domain", "top_score",
                "top_similarity", "top_mahalanobis", "top_entropy", "prompt_preview"],
        rows=meta_rows,
    )

    # ---------- Projector config ----------
    write_projector_config(
        out_dir / "projector_config.pbtxt",
        embeddings=[
            ("query_embeddings.tsv", "query_metadata.tsv"),
            ("domain_centroids.tsv", "domain_centroids_metadata.tsv"),
        ],
    )

    print(f"[OK] Wrote TensorBoard Projector workspace to: {out_dir}")
    print("Run:")
    print(f"  tensorboard --logdir {out_dir} --port 6006")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
