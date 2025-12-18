#!/usr/bin/env python3
"""
Build a 3D embedding map (UMAP/PCA) for GRCLM routing visualization.

Outputs:
  shiftable_project/outputs/viz/embedding_map.npz
  shiftable_project/outputs/viz/reducer.pkl

Default approach: sample synthetic points from each domain's Gaussian
(using centroid + cov_chol from domain_stats.json). This gives you an
"MRI-like density cloud" without re-encoding huge corpora.

Optional: you can extend this later to sample real texts and embed them.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

def load_domain_stats(stats_path: Path) -> Tuple[int, Dict[str, dict]]:
    with open(stats_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    dim = int(data["dim"])
    domains = data["domains"]
    return dim, domains

def sample_domain_points(dim: int, dom: dict, n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    mu = np.array(dom["centroid"], dtype=np.float32)  # [D]

    # Prefer cov_chol if present (fast sampling: mu + L@z)
    if dom.get("cov_chol"):
        L = np.array(dom["cov_chol"], dtype=np.float32)  # [D,D]
        z = rng.standard_normal(size=(dim, n), dtype=np.float32)
        x = (mu.reshape(dim, 1) + L @ z).T  # [n,D]
        return x

    # Fallback: diagonal variance
    var = np.array(dom.get("var_diag", [1.0] * dim), dtype=np.float32)
    var = np.clip(var, 1e-8, None)
    x = mu + rng.standard_normal(size=(n, dim), dtype=np.float32) * np.sqrt(var)
    return x

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", choices=["umap", "pca"], default="umap")
    ap.add_argument("--n_general", type=int, default=800)
    ap.add_argument("--n_other", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    stats_path = repo_root / "shiftable_project" / "outputs" / "shiftable" / "domain_stats.json"
    out_dir = repo_root / "shiftable_project" / "outputs" / "viz"
    out_dir.mkdir(parents=True, exist_ok=True)

    dim, domains = load_domain_stats(stats_path)

    # Build high-D point cloud (synthetic density per domain)
    X_list = []
    y_list = []
    for name, dom in domains.items():
        n = args.n_general if name == "general" else args.n_other
        Xd = sample_domain_points(dim, dom, n=n, seed=args.seed + (abs(hash(name)) % 10_000))
        X_list.append(Xd)
        y_list.extend([name] * Xd.shape[0])

    X = np.vstack(X_list).astype(np.float32)
    labels = np.array(y_list, dtype=object)

    # Also include centroids explicitly so they’re stable + easy to annotate
    centroid_names = list(domains.keys())
    centroids = np.stack([np.array(domains[n]["centroid"], dtype=np.float32) for n in centroid_names], axis=0)

    # Fit reducer
    if args.method == "umap":
        import umap  # umap-learn
        reducer = umap.UMAP(n_components=3, n_neighbors=30, min_dist=0.05, metric="cosine", random_state=args.seed)
        X3 = reducer.fit_transform(np.vstack([X, centroids]))
        cloud3 = X3[: X.shape[0], :]
        cent3 = X3[X.shape[0] :, :]
    else:
        from sklearn.decomposition import PCA
        reducer = PCA(n_components=3, random_state=args.seed)
        reducer.fit(np.vstack([X, centroids]))
        cloud3 = reducer.transform(X)
        cent3 = reducer.transform(centroids)

    # Save artifacts
    np.savez_compressed(
        out_dir / "embedding_map.npz",
        cloud3=cloud3,
        labels=labels,
        centroid3=cent3,
        centroid_names=np.array(centroid_names, dtype=object),
        dim=np.array([dim], dtype=np.int32),
        method=np.array([args.method], dtype=object),
    )

    with open(out_dir / "reducer.pkl", "wb") as f:
        pickle.dump(reducer, f)

    print(f"[ok] wrote: {out_dir/'embedding_map.npz'}")
    print(f"[ok] wrote: {out_dir/'reducer.pkl'}")

if __name__ == "__main__":
    main()
