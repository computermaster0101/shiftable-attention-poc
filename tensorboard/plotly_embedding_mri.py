#!/usr/bin/env python3
"""
Interactive "MRI-ish" 3D embedding viewer (points + centroid ellipsoids) using Plotly.

Inputs (defaults match your repo layout):
  - shiftable_project/outputs/shiftable/domain_stats.json
  - tensorboard_projector/query_embeddings.tsv  (optional; produced by export_tensorboard_projector.py)
  - tensorboard_projector/query_metadata.tsv    (optional)

Outputs:
  - embedding_mri.html   (self-contained interactive HTML)

Run:
  python plotly_embedding_mri.py --project_root . --out_html embedding_mri.html
  # then open embedding_mri.html in your browser

Notes:
- TensorBoard Projector does NOT render ellipsoids; this Plotly viewer does.
- If your embedding dim is > 3, we reduce to 3D via PCA (numpy SVD).
- Ellipsoids are derived from covariance (or cov_chol if present). We render 1-sigma by default.
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import numpy as np

def pca_to_3d(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
      X3: (N,3) projection
      V3: (D,3) principal axes in original space
      mu: (D,) mean
    """
    mu = X.mean(axis=0)
    Xc = X - mu
    # SVD for stability
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    V = Vt.T  # (D,D)
    V3 = V[:, :3]
    X3 = Xc @ V3
    return X3, V3, mu

def load_domain_stats(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))

def cov_from_info(info: Dict[str, Any]) -> Optional[np.ndarray]:
    """
    domain_stats may include:
      - cov (full matrix)
      - cov_chol (lower-triangular Cholesky)
    """
    if "cov" in info and isinstance(info["cov"], list):
        C = np.array(info["cov"], dtype=float)
        if C.ndim == 2:
            return C
    if "cov_chol" in info and isinstance(info["cov_chol"], list):
        L = np.array(info["cov_chol"], dtype=float)
        if L.ndim == 2:
            return L @ L.T
    return None

def ellipsoid_mesh(center3: np.ndarray, cov3: np.ndarray, n_theta: int=30, n_phi: int=16, sigma: float=1.0):
    # eigen-decomp
    vals, vecs = np.linalg.eigh(cov3)
    vals = np.maximum(vals, 1e-9)
    radii = sigma * np.sqrt(vals)

    theta = np.linspace(0, 2*np.pi, n_theta)
    phi = np.linspace(0, np.pi, n_phi)
    tt, pp = np.meshgrid(theta, phi)

    x = radii[0]*np.cos(tt)*np.sin(pp)
    y = radii[1]*np.sin(tt)*np.sin(pp)
    z = radii[2]*np.cos(pp)

    pts = np.stack([x, y, z], axis=-1)  # (n_phi, n_theta, 3)
    pts = pts @ vecs.T
    pts = pts + center3
    return pts[...,0], pts[...,1], pts[...,2]

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project_root", type=str, default=".")
    ap.add_argument("--domain_stats", type=str, default="shiftable_project/outputs/shiftable/domain_stats.json")
    ap.add_argument("--tb_dir", type=str, default="tensorboard_projector",
                    help="Directory containing query_embeddings.tsv / query_metadata.tsv (optional).")
    ap.add_argument("--out_html", type=str, default="embedding_mri.html")
    ap.add_argument("--max_points", type=int, default=20000)
    ap.add_argument("--sigma", type=float, default=1.0, help="Ellipsoid sigma radius (1=1σ, 2=2σ, etc.)")
    args = ap.parse_args()

    root = Path(args.project_root).expanduser().resolve()
    def resolve(p: str) -> Path:
        pp = Path(p).expanduser()
        return pp if pp.is_absolute() else (root / pp)

    ds_path = resolve(args.domain_stats)
    ds = load_domain_stats(ds_path)
    dim = int(ds.get("dim", 0) or 0)
    domains = ds.get("domains", {})

    # Collect centroids and covariances
    names: List[str] = []
    C_centroids: List[np.ndarray] = []
    C_covs: List[Optional[np.ndarray]] = []

    for name, info in domains.items():
        if not isinstance(info, dict):
            continue
        c = info.get("centroid")
        if not isinstance(c, list):
            continue
        cvec = np.array(c, dtype=float)
        if dim and cvec.shape[0] != dim:
            continue
        names.append(str(name))
        C_centroids.append(cvec)
        C_covs.append(cov_from_info(info))

    if not C_centroids:
        raise ValueError("No centroids found in domain_stats.json")

    C = np.stack(C_centroids, axis=0)  # (K,D)

    # Load query embeddings if present (exported via export_tensorboard_projector.py)
    tb_dir = resolve(args.tb_dir)
    q_path = tb_dir / "query_embeddings.tsv"
    q_meta_path = tb_dir / "query_metadata.tsv"
    Q = None
    Q_meta = None

    if q_path.exists():
        Q = np.loadtxt(q_path, delimiter="\t")
        if Q.ndim == 1:
            Q = Q[None, :]
        if dim and Q.shape[1] != dim:
            # ignore if mismatched
            Q = None

    if q_meta_path.exists():
        # metadata is tsv with header
        import csv
        with q_meta_path.open("r", encoding="utf-8", newline="") as f:
            r = csv.reader(f, delimiter="\t")
            header = next(r, None)
            rows = list(r)
        Q_meta = (header, rows)

    # Build joint matrix for PCA so centroids and queries share the same projection
    X_parts = [C]
    if Q is not None:
        if Q.shape[0] > args.max_points:
            Q = Q[:args.max_points]
        X_parts.append(Q)
    X = np.concatenate(X_parts, axis=0)
    X3, V3, mu = pca_to_3d(X)

    C3 = X3[:C.shape[0]]
    Q3 = X3[C.shape[0]:] if Q is not None else None

    # Project covariances into 3D: cov3 = V3^T cov V3
    cov3_list: List[Optional[np.ndarray]] = []
    for cov in C_covs:
        if cov is None:
            cov3_list.append(None)
        else:
            cov3_list.append(V3.T @ cov @ V3)

    # Create plotly figure
    try:
        import plotly.graph_objects as go
    except Exception as e:
        raise SystemExit("Plotly not installed. Install with: pip install plotly") from e

    fig = go.Figure()

    # Queries (points)
    if Q3 is not None and Q3.size:
        hover = None
        if Q_meta is not None:
            header, rows = Q_meta
            # Use a compact hover text
            # columns: timestamp, source, is_unknown, top_domain, top_score, ...
            idx_prompt = header.index("prompt_preview") if "prompt_preview" in header else None
            idx_dom = header.index("top_domain") if "top_domain" in header else None
            idx_score = header.index("top_score") if "top_score" in header else None
            hover = []
            for i, r in enumerate(rows[:Q3.shape[0]]):
                p = r[idx_prompt] if idx_prompt is not None else ""
                d = r[idx_dom] if idx_dom is not None else ""
                s = r[idx_score] if idx_score is not None else ""
                hover.append(f"top_domain={d}<br>top_score={s}<br>{p}")
        fig.add_trace(go.Scatter3d(
            x=Q3[:,0], y=Q3[:,1], z=Q3[:,2],
            mode="markers",
            name="query_embeddings",
            marker=dict(size=2, opacity=0.7),
            text=hover,
            hoverinfo="text" if hover is not None else "skip"
        ))

    # Centroids
    fig.add_trace(go.Scatter3d(
        x=C3[:,0], y=C3[:,1], z=C3[:,2],
        mode="markers+text",
        name="domain_centroids",
        marker=dict(size=6, opacity=0.95),
        text=names,
        textposition="top center",
        hoverinfo="text"
    ))

    # Ellipsoids per centroid (covariance)
    for i, name in enumerate(names):
        cov3 = cov3_list[i]
        if cov3 is None:
            continue
        Xs, Ys, Zs = ellipsoid_mesh(C3[i], cov3, sigma=args.sigma)
        fig.add_trace(go.Surface(
            x=Xs, y=Ys, z=Zs,
            name=f"{name} ellipsoid",
            showscale=False,
            opacity=0.25,
            hoverinfo="skip"
        ))

    fig.update_layout(
        title=f"Embedding MRI Viewer (PCA-3D) | sigma={args.sigma}",
        scene=dict(
            xaxis_title="PC1",
            yaxis_title="PC2",
            zaxis_title="PC3",
        ),
        legend=dict(itemsizing="constant")
    )

    out_html = resolve(args.out_html)
    out_html.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(out_html), include_plotlyjs="cdn")
    print(f"[OK] wrote: {out_html}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
