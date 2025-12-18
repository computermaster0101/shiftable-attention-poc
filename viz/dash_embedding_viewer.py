#!/usr/bin/env python3
"""
viz/dash_embedding_viewer.py

Dash + Plotly 3D viewer (SUBMIT-based for prompt routing) that does ALL THREE:

1) Prompt submit tracing (click "Route")  ✅ (no live tick)
2) Generation tracing (token-by-token)   ✅
3) Playback (scrub + autoplay)           ✅

Also displays generated text in a dedicated panel.

Run:
  python viz/build_embedding_map.py --method umap
  python viz/dash_embedding_viewer.py
"""

from __future__ import annotations

import pickle
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

import dash
from dash import dcc, html, dash_table, no_update
from dash.dependencies import Input, Output, State
import plotly.graph_objects as go


# -----------------------------------------------------------------------------
# Repo imports
# -----------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
import sys
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.model_manager import model_manager  # singleton
from app.router import DomainRouter          # for prompt submit routing (vector->route)
from app import config as app_config


# -----------------------------------------------------------------------------
# Thread-safe generation trace state
# -----------------------------------------------------------------------------
class _GenState:
    def __init__(self):
        self.lock = threading.Lock()
        self.running: bool = False
        self.frames: List[dict] = []
        self.error: Optional[str] = None
        self.stop_flag: bool = False

GEN = _GenState()


# -----------------------------------------------------------------------------
# Small attribute helper so we don’t explode on slight naming differences
# -----------------------------------------------------------------------------
def _get(obj: Any, *names: str, default=None):
    for n in names:
        if hasattr(obj, n):
            return getattr(obj, n)
    return default


def _metrics_rows(rr: Any) -> List[dict]:
    rows = []
    for m in (getattr(rr, "metrics", []) or []):
        rows.append({
            "domain": _get(m, "name", "domain", default="unknown"),
            "cosine": round(float(_get(m, "cosine_sim", "similarity", default=0.0)), 6),
            "mahal": round(float(_get(m, "mahal_dist", "mahalanobis", default=0.0)), 6),
            "entropy": round(float(_get(m, "entropy", default=0.0)), 6),
            "support": round(float(_get(m, "support_ratio", "support", default=0.0)), 6),
        })
    return rows


def _route_topk(rr: Any, top_k: int) -> List[dict]:
    """
    Return: [{"name": domain, "weight": w_or_0}, ...]
    Prefers rr.blend_weights if present, else ranks rr.metrics by cosine/similarity.
    """
    bw = getattr(rr, "blend_weights", None)
    if isinstance(bw, dict) and len(bw) > 0:
        items = sorted(bw.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
        return [{"name": k, "weight": float(v)} for k, v in items]

    metrics = getattr(rr, "metrics", []) or []
    metrics_sorted = sorted(
        metrics,
        key=lambda m: float(_get(m, "cosine_sim", "similarity", default=0.0)),
        reverse=True,
    )[:top_k]
    return [{"name": _get(m, "name", "domain", default="unknown"), "weight": 0.0} for m in metrics_sorted]


def _project_3d(reducer: Any, emb: torch.Tensor) -> np.ndarray:
    q_np = emb.detach().float().cpu().numpy().reshape(1, -1)
    q3 = reducer.transform(q_np)[0]
    return np.array([float(q3[0]), float(q3[1]), float(q3[2])], dtype=np.float32)


# -----------------------------------------------------------------------------
# Plotly figure builder
# -----------------------------------------------------------------------------
def build_figure(
    cloud3: np.ndarray,
    labels: np.ndarray,
    centroid3: np.ndarray,
    centroid_names: List[str],
    query3: Optional[np.ndarray] = None,
    trail3: Optional[np.ndarray] = None,
    topk: Optional[List[dict]] = None,  # [{"name": str, "weight": float}, ...]
    camera: Optional[dict] = None,
) -> go.Figure:
    fig = go.Figure()

    # Background cloud (domain-colored)
    for name in sorted(set(labels.tolist())):
        mask = labels == name
        pts = cloud3[mask]
        fig.add_trace(go.Scatter3d(
            x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
            mode="markers",
            name=f"cloud:{name}",
            marker=dict(size=2),
            opacity=0.30,
            hoverinfo="skip",
        ))

    # Centroids (highlight top-k by size)
    topk_names = set([t["name"] for t in (topk or [])])
    sizes = [14 if n in topk_names else 8 for n in centroid_names]

    fig.add_trace(go.Scatter3d(
        x=centroid3[:, 0], y=centroid3[:, 1], z=centroid3[:, 2],
        mode="markers+text",
        name="centroids",
        marker=dict(size=sizes, symbol="diamond"),
        text=centroid_names,
        textposition="top center",
    ))

    # Trail (trajectory)
    if trail3 is not None and len(trail3) >= 2:
        fig.add_trace(go.Scatter3d(
            x=trail3[:, 0], y=trail3[:, 1], z=trail3[:, 2],
            mode="lines+markers",
            name="trail",
            marker=dict(size=3),
            opacity=0.95,
        ))

    # Query point
    if query3 is not None:
        fig.add_trace(go.Scatter3d(
            x=[query3[0]], y=[query3[1]], z=[query3[2]],
            mode="markers",
            name="query",
            marker=dict(size=11, symbol="circle"),
        ))

    # Fan-out arrows to top-k centroids
    if query3 is not None and topk:
        for rank, item in enumerate(topk):
            name = item["name"]
            w = float(item.get("weight", 0.0))
            if name not in centroid_names:
                continue
            idx = centroid_names.index(name)
            c3 = centroid3[idx]

            opacity = max(0.15, min(1.0, w)) if w > 0 else max(0.15, 1.0 - 0.22 * rank)

            fig.add_trace(go.Scatter3d(
                x=[query3[0], c3[0]],
                y=[query3[1], c3[1]],
                z=[query3[2], c3[2]],
                mode="lines",
                name=f"route:{name}",
                opacity=opacity,
                hoverinfo="skip",
            ))

    fig.update_layout(
        uirevision="keep-camera",   # <- key line: preserves user view across updates
        margin=dict(l=0, r=0, t=0, b=0),
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z",
            camera=camera if camera else None,
        ),
        legend=dict(itemsizing="constant"),
    )

    return fig


# -----------------------------------------------------------------------------
# Generation tracing (token-by-token) in a background thread
# -----------------------------------------------------------------------------
def traced_generate(
    prompt: str,
    reducer: Any,
    topk_domains: int,
    max_new_tokens: int,
    temperature: float,
    top_k_sample: int,
    reweight_mode: str
):
    """
    Token-by-token generation + routing trace.
    Writes frames into GEN.frames:
      frame = { step, text, q3, best_domain, is_unknown, reason, topk, metrics }
    """
    final_rr = None
    final_domain_prior = None
    final_domain_mask = None

    with GEN.lock:
        GEN.running = True
        GEN.stop_flag = False
        GEN.frames = []
        GEN.error = None

    try:
        model_manager.ensure_initialized()

        shift = model_manager.shift_model
        tok = model_manager.tokenizer
        router = model_manager.router

        if shift is None or tok is None or router is None:
            raise RuntimeError("model_manager missing shift_model/tokenizer/router")

        shift.eval()

        # prompt -> ids
        input_ids = tok.encode(prompt, add_specials=True)
        generated_ids: List[int] = input_ids.copy()

        def record_frame(step: int, text_so_far: str, q_emb: torch.Tensor, rr: Any):
            q3 = _project_3d(reducer, q_emb)
            topk_list = _route_topk(rr, topk_domains)
            frame = {
                "step": int(step),
                "text": text_so_far,
                "q3": [float(q3[0]), float(q3[1]), float(q3[2])],
                "best_domain": getattr(rr, "best_domain", None),
                "is_unknown": getattr(rr, "is_unknown", None),
                "reason": getattr(rr, "reason", ""),
                "topk": topk_list,
                "metrics": _metrics_rows(rr),
            }
            with GEN.lock:
                GEN.frames.append(frame)

        # frame 0: prompt routing
        ids_t = torch.tensor(input_ids, dtype=torch.long, device=model_manager.device)
        q_emb0 = model_manager._compute_query_embedding(ids_t)
        if model_manager.specialist_names:
            rr0 = router.route(q_emb0)
        else:
            rr0 = type("RR", (), {"best_domain": "general", "is_unknown": True, "reason": "no specialists", "metrics": []})()

        record_frame(0, tok.decode(generated_ids, skip_specials=True), q_emb0, rr0)

        # fixed routing + priors (ModelManager-style)
        fixed_domain_prior = None
        fixed_domain_mask = None
        fixed_domain_weights = None
        fixed_rr = rr0
        fixed_q_emb = q_emb0

        if model_manager.specialist_names and fixed_rr is not None and not getattr(fixed_rr, "is_unknown", False):
            prior_vec, mask_vec = model_manager._build_domain_prior_from_routing(fixed_rr)
            fixed_domain_prior = prior_vec.unsqueeze(0)
            fixed_domain_mask = mask_vec.unsqueeze(0)

            # optional learned blending (computed once, like generate())
            try:
                alpha = float(getattr(app_config, "GATE_GEOMETRY_ALPHA", 6.0))
                beta = float(getattr(app_config, "GATE_LEARNED_BETA", 1.0))
                if hasattr(shift, "blend_gate") and callable(getattr(shift, "blend_gate")):
                    pooled = fixed_q_emb.unsqueeze(0)
                    g_logits = shift.blend_gate(pooled)
                    prior = torch.clamp(fixed_domain_prior, min=1e-9)
                    mix_logits = alpha * torch.log(prior) + beta * g_logits
                    mix_logits = mix_logits.masked_fill(fixed_domain_mask == 0, float("-inf"))
                    fixed_domain_weights = torch.softmax(mix_logits, dim=-1)
            except Exception:
                fixed_domain_weights = None

        cached_domain_prior = fixed_domain_prior
        cached_domain_mask = fixed_domain_mask
        cached_domain_weights = fixed_domain_weights
        cached_rr = fixed_rr
        cached_q_emb = fixed_q_emb

        # generation loop
        max_seq = getattr(shift, "max_seq_len", getattr(app_config, "MAX_SEQ_LEN", 512))

        for step in range(1, int(max_new_tokens) + 1):
            with GEN.lock:
                if GEN.stop_flag:
                    break

            context_ids = generated_ids[-max_seq:] if len(generated_ids) > max_seq else generated_ids

            ctx = torch.tensor(context_ids, dtype=torch.long, device=model_manager.device).unsqueeze(0)
            pad_id = getattr(tok, "pad_id", None)
            if pad_id is None:
                pad_id = getattr(tok, "pad_token_id", None)
            if pad_id is None:
                # fall back: assume no padding in ctx (common for causal gen with explicit slicing)
                attn = torch.ones_like(ctx, dtype=torch.long)
            else:
                attn = (ctx != pad_id).long()

            # routing embedding at this step
            q_emb = cached_q_emb
            rr = cached_rr
            domain_prior = cached_domain_prior
            domain_mask = cached_domain_mask
            domain_weights = cached_domain_weights

            if reweight_mode == "none":
                domain_prior = None
                domain_mask = None
                domain_weights = None

            elif reweight_mode == "every" or (reweight_mode == "first" and step == 1):
                # recompute routing on current context_ids
                ctx_1d = torch.tensor(context_ids, dtype=torch.long, device=model_manager.device)
                q_emb = model_manager._compute_query_embedding(ctx_1d)
                rr = router.route(q_emb) if model_manager.specialist_names else rr0

                domain_prior = None
                domain_mask = None
                domain_weights = None

                if model_manager.specialist_names and rr is not None and not getattr(rr, "is_unknown", False):
                    prior_vec, mask_vec = model_manager._build_domain_prior_from_routing(rr)
                    domain_prior = prior_vec.unsqueeze(0)
                    domain_mask = mask_vec.unsqueeze(0)

                    try:
                        alpha = float(getattr(app_config, "GATE_GEOMETRY_ALPHA", 6.0))
                        beta = float(getattr(app_config, "GATE_LEARNED_BETA", 1.0))
                        if hasattr(shift, "blend_gate") and callable(getattr(shift, "blend_gate")):
                            pooled = q_emb.unsqueeze(0)
                            g_logits = shift.blend_gate(pooled)
                            prior = torch.clamp(domain_prior, min=1e-9)
                            mix_logits = alpha * torch.log(prior) + beta * g_logits
                            mix_logits = mix_logits.masked_fill(domain_mask == 0, float("-inf"))
                            domain_weights = torch.softmax(mix_logits, dim=-1)
                    except Exception:
                        domain_weights = None

                # if mode == "first", cache the computed values so later steps reuse them
                if reweight_mode == "first":
                    cached_q_emb = q_emb
                    cached_rr = rr
                    cached_domain_prior = domain_prior
                    cached_domain_mask = domain_mask
                    cached_domain_weights = domain_weights
                    
                


            with torch.no_grad():
                logits = shift(
                    ctx,
                    attention_mask=attn,
                    domain_prior=domain_prior,
                    domain_mask=domain_mask,
                    domain_weights=domain_weights,
                )
                next_logits = logits[0, -1, :]

            # sample / greedy
            if temperature <= 0:
                next_id = int(torch.argmax(next_logits).item())
            else:
                scaled = next_logits / float(temperature)
                if 0 < int(top_k_sample) < scaled.size(-1):
                    values, indices = torch.topk(scaled, int(top_k_sample))
                    filtered = torch.full_like(scaled, float("-inf"))
                    filtered[indices] = values
                else:
                    filtered = scaled

                probs = torch.softmax(filtered, dim=-1)
                next_id = int(torch.multinomial(probs, num_samples=1).item())

            generated_ids.append(next_id)

            # record frame AFTER selecting token
            text_so_far = tok.decode(generated_ids, skip_specials=True)
            record_frame(step, text_so_far, q_emb, rr)

            # keep last-seen routing + priors for logging
            final_rr = rr
            final_domain_prior = domain_prior
            final_domain_mask = domain_mask

            if next_id == tok.eos_id:
                break
            # tiny sleep keeps UI responsive on CPU
            time.sleep(0.001)

        try:
            completion = tok.decode(generated_ids, skip_specials=True)
            # pick the best available embedding to log
            qe = q_emb if "q_emb" in locals() else q_emb0
            rr_to_log = final_rr if final_rr is not None else rr0

            model_manager._log_input_output(
                prompt=prompt,
                completion=completion,
                query_embedding=qe,
                routing_result=rr_to_log,
                domain_prior=final_domain_prior,
                domain_mask=final_domain_mask,
            )

            # Mirror ModelManager.generate(): log unknown/low-confidence samples to emergence_log
            is_low_confidence = (not model_manager.specialist_names) or (
                rr_to_log is not None and getattr(rr_to_log, "is_unknown", False)
            )
            if is_low_confidence:
                print("EMERGENCE_LOG_PATH =", getattr(app_config, "EMERGENCE_LOG_PATH", "missing"))
                model_manager._handle_emergent_query(
                    prompt=prompt,
                    completion=completion,
                    query_embedding=qe,
                    routing_result=rr_to_log,
                )

        except Exception as e:
            import traceback
            print("[dash_embedding_viewer] emergent logging failed:", repr(e))
            traceback.print_exc()


    except Exception as e:
        with GEN.lock:
            GEN.error = f"{type(e).__name__}: {e}"

    finally:
        with GEN.lock:
            GEN.running = False


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    # Load prebuilt embedding map + reducer
    viz_dir = REPO_ROOT / "shiftable_project" / "outputs" / "viz"
    npz_path = viz_dir / "embedding_map.npz"
    reducer_path = viz_dir / "reducer.pkl"

    if not npz_path.exists() or not reducer_path.exists():
        raise SystemExit(
            "Missing viz artifacts.\n"
            "Run first:\n"
            "  python viz/build_embedding_map.py --method umap\n"
            f"Expected:\n  {npz_path}\n  {reducer_path}\n"
        )

    data = np.load(npz_path, allow_pickle=True)
    cloud3 = data["cloud3"]
    labels = data["labels"]
    centroid3 = data["centroid3"]
    centroid_names = data["centroid_names"].tolist()
    method = str(data["method"][0])

    with open(reducer_path, "rb") as f:
        reducer = pickle.load(f)

    # Prompt submit routing uses DomainRouter + generalist encoding through model_manager
    model_manager.ensure_initialized()
    router = DomainRouter(app_config.DOMAIN_STATS_PATH)

    # Dash app
    app = dash.Dash(__name__)
    app.title = "GRCLM Embedding MRI (3D)"

    base_fig = build_figure(cloud3, labels, centroid3, centroid_names, camera=None)

    # Tunables
    MAX_TRAIL = 60
    TOPK_DOMAINS = 3

    # Generation defaults (from config if present)
    DEFAULT_MAX_NEW = int(getattr(app_config, "MAX_NEW_TOKENS", 120) or 120)
    DEFAULT_TEMP = float(getattr(app_config, "TEMPERATURE", 0.8) or 0.8)
    DEFAULT_TOPK_SAMPLE = int(getattr(app_config, "TOP_K", 40) or 40)

    app.layout = html.Div(
        style={"display": "flex", "flexDirection": "row", "height": "100vh"},
        children=[
            # Poll generation frames (no prompt live tick
            dcc.Interval(id="gen-poll", interval=200, n_intervals=0, disabled=True),
            dcc.Interval(id="play-tick", interval=200, n_intervals=0, disabled=True),

            dcc.Store(id="prompt-trail-store", data={"pts": []}),
            dcc.Store(id="last-text-store", data={"text": ""}),
            dcc.Store(id="gen-store", data={"frames": [], "running": False, "error": None}),
            dcc.Store(id="play-store", data={"playing": False}),
            dcc.Store(id="camera-store", data=None),

            html.Div(
                style={"flex": "2", "padding": "10px"},
                children=[
                    dcc.Graph(id="emb-graph", figure=base_fig, style={"height": "95vh"}),
                ],
            ),

            html.Div(
                style={"flex": "1", "padding": "10px", "borderLeft": "1px solid #ddd"},
                children=[
                    html.H3("GRCLM Embedding MRI"),
                    html.Div(f"Reducer: {method} (prebuilt)"),

                    html.Div("Generation routing"),
                    dcc.RadioItems(
                        id="reweight-mode",
                        options=[
                            {"label": "Not at all", "value": "none"},
                            {"label": "Only first step", "value": "first"},
                            {"label": "Every step", "value": "every"},
                        ],
                        value="every",  # preserves today's default behavior
                        style={"marginTop": "6px"},
                    ),

                    
                    html.Div("Mode"),
                    dcc.RadioItems(
                        id="mode",
                        options=[
                            {"label": "Prompt (Submit)", "value": "prompt"},
                            {"label": "Generate trace (Live)", "value": "gen"},
                            {"label": "Playback", "value": "playback"},
                        ],
                        value="prompt",
                    ),

                    dcc.Textarea(
                        id="prompt",
                        value="",
                        style={"width": "100%", "height": "120px"},
                    ),

                    html.Div(style={"display": "flex", "gap": "8px", "marginTop": "8px"}, children=[
                        html.Button("Route", id="route-btn", n_clicks=0),
                        html.Button("Start Generate+Trace", id="gen-start", n_clicks=0),
                        html.Button("Stop", id="gen-stop", n_clicks=0),
                    ]),

                    html.Div(style={"height": "10px"}),

                    html.Div("Playback"),
                    dcc.Slider(id="frame-slider", min=0, max=0, step=1, value=0),
                    html.Div(style={"display": "flex", "gap": "8px", "marginTop": "6px"}, children=[
                        html.Button("Play/Pause", id="play-toggle", n_clicks=0),
                        html.Button("Reset to 0", id="play-reset", n_clicks=0),
                    ]),

                    html.H4("Generated text"),
                    dcc.Textarea(
                        id="generated-text",
                        value="",
                        style={
                            "width": "100%",
                            "height": "160px",
                            "fontFamily": "monospace",
                            "whiteSpace": "pre-wrap",
                        },
                    ),

                    html.Div(id="route-summary", style={"marginTop": "10px", "whiteSpace": "pre-wrap"}),

                    html.H4("Per-domain metrics"),
                    dash_table.DataTable(
                        id="metrics-table",
                        columns=[
                            {"name": "domain", "id": "domain"},
                            {"name": "cosine", "id": "cosine"},
                            {"name": "mahal", "id": "mahal"},
                            {"name": "entropy", "id": "entropy"},
                            {"name": "support", "id": "support"},
                        ],
                        data=[],
                        sort_action="native",
                        style_table={"overflowX": "auto"},
                        style_cell={"fontFamily": "monospace", "fontSize": 12},
                        page_size=12,
                    ),
                ],
            ),
        ],
    )

    # -------------------------
    # Move the graph camera on demand
    # -------------------------
    @app.callback(
        Output("camera-store", "data"),
        Input("emb-graph", "relayoutData"),
        State("camera-store", "data"),
        prevent_initial_call=True,
    )
    def remember_camera(relayout, cur):
        if not relayout:
            return no_update

        # Plotly sends camera updates under these keys for 3D scenes
        cam = None
        if "scene.camera" in relayout:
            cam = relayout["scene.camera"]
        elif "scene.camera.eye" in relayout or "scene.camera.center" in relayout or "scene.camera.up" in relayout:
            # Sometimes it sends partial keys; keep whatever we can
            cam = cur or {}
            # Merge partial updates
            for k, v in relayout.items():
                if k.startswith("scene.camera."):
                    # k like "scene.camera.eye" -> set cam["eye"]
                    cam[k.split("scene.camera.", 1)[1]] = v

        return cam if cam is not None else no_update


    # -------------------------
    # Start/Stop generation trace thread
    # -------------------------
    @app.callback(
        Output("gen-store", "data", allow_duplicate=True),
        Input("gen-start", "n_clicks"),
        Input("gen-stop", "n_clicks"),
        State("prompt", "value"),
        State("reweight-mode", "value"),
        prevent_initial_call=True,
    )
    def start_stop_gen(_n_start, _n_stop, prompt_text, reweight_mode):
        ctx = dash.callback_context
        trig = ctx.triggered[0]["prop_id"].split(".")[0] if ctx.triggered else ""

        if trig == "gen-stop":
            with GEN.lock:
                GEN.stop_flag = True
            return no_update

        prompt_text = (prompt_text or "").strip()
        if not prompt_text:
            return {"frames": [], "running": False, "error": "empty prompt"}

        with GEN.lock:
            if GEN.running:
                return no_update

        def _run():
            traced_generate(
                prompt=prompt_text,
                reducer=reducer,
                topk_domains=TOPK_DOMAINS,
                max_new_tokens=DEFAULT_MAX_NEW,
                temperature=DEFAULT_TEMP,
                top_k_sample=DEFAULT_TOPK_SAMPLE,
                reweight_mode=reweight_mode
            )

        t = threading.Thread(target=_run, daemon=True)
        t.start()
        return {"frames": [], "running": True, "error": None}

    # -------------------------
    # Poll generation frames into Dash store + slider max/value
    # -------------------------
    @app.callback(
        Output("gen-store", "data"),
        Output("frame-slider", "max"),
        Output("frame-slider", "value"),
        Input("gen-poll", "n_intervals"),
        Input("play-tick", "n_intervals"),
        State("gen-store", "data"),
        State("mode", "value"),
        State("play-store", "data"),
        State("frame-slider", "value"),
    )
    def poll_gen(_gen_n, _play_n, gen_store, mode, play_store, slider_val):
        # read current gen trace from thread
        with GEN.lock:
            frames = list(GEN.frames)
            running = GEN.running
            err = GEN.error

        if gen_store is None:
            gen_store = {"frames": [], "running": False, "error": None}

        # update store if changed
        prev_frames = gen_store.get("frames", []) or []
        store_changed = (
            len(frames) != len(prev_frames)
            or running != bool(gen_store.get("running", False))
            or err != gen_store.get("error", None)
        )

        if store_changed:
            gen_store = {"frames": frames, "running": running, "error": err}

        max_idx = max(0, len(frames) - 1)

        # Decide slider value (single source of truth)
        cur = int(slider_val or 0)
        cur = max(0, min(cur, max_idx))

        playing = bool((play_store or {}).get("playing", False))

        if mode == "gen" and running:
            # follow live generation
            cur = max_idx
        elif mode == "playback" and playing and max_idx > 0:
            # autoplay during playback
            cur = 0 if cur >= max_idx else cur + 1
        else:
            # keep whatever the user set (scrub/manual)
            cur = cur

        return gen_store, max_idx, cur


    # -------------------------
    # Playback play/pause/reset state
    # -------------------------
    @app.callback(
        Output("play-store", "data"),
        Input("play-toggle", "n_clicks"),
        Input("play-reset", "n_clicks"),
        State("play-store", "data"),
        prevent_initial_call=True,
    )
    def play_controls(_t, _r, play_store):
        play_store = play_store or {"playing": False}
        ctx = dash.callback_context
        trig = ctx.triggered[0]["prop_id"].split(".")[0] if ctx.triggered else ""

        if trig == "play-reset":
            return {"playing": False}
        if trig == "play-toggle":
            return {"playing": not bool(play_store.get("playing", False))}
        return no_update


    @app.callback(
        Output("gen-poll", "disabled"),
        Output("play-tick", "disabled"),
        Input("mode", "value"),
        Input("gen-store", "data"),
        Input("play-store", "data"),
    )
    def control_intervals(mode, gen_store, play_store):
        running = bool((gen_store or {}).get("running", False))
        playing = bool((play_store or {}).get("playing", False))

        gen_poll_disabled = not (mode == "gen" and running)
        play_tick_disabled = not (mode == "playback" and playing)

        return gen_poll_disabled, play_tick_disabled
    # -------------------------
    # ONE render callback for all modes (prevents duplicate-output conflicts)
    # -------------------------
    @app.callback(
        Output("emb-graph", "figure"),
        Output("route-summary", "children"),
        Output("metrics-table", "data"),
        Output("generated-text", "value"),
        Output("prompt-trail-store", "data"),
        Output("last-text-store", "data"),
        Input("route-btn", "n_clicks"),
        Input("mode", "value"),
        Input("frame-slider", "value"),
        Input("gen-store", "data"),
        State("prompt", "value"),
        State("prompt-trail-store", "data"),
        State("last-text-store", "data"),
        State("camera-store", "data"),
        prevent_initial_call=True,
    )
    def render_all(_route_clicks, mode, frame_idx, gen_store, prompt_text, trail_data, last_text_data, camera_data):
        ctx = dash.callback_context
        trig = ctx.triggered[0]["prop_id"].split(".")[0] if ctx.triggered else ""

        # ---------------------
        # Mode: PROMPT (submit)
        # ---------------------
        if mode == "prompt":
            if trig != "route-btn":
                return no_update, no_update, no_update, no_update, no_update, no_update

            text = (prompt_text or "").strip()
            if not text:
                return no_update, no_update, no_update, "", trail_data, last_text_data

            last_text = (last_text_data or {}).get("text", "")
            if text == last_text:
                return no_update, no_update, no_update, "", trail_data, last_text_data

            # Encode via model_manager base encoder for consistency
            tok = model_manager.tokenizer
            if tok is None:
                return no_update, "tokenizer missing", [], "", trail_data, last_text_data

            ids = tok.encode(text, add_specials=True)[: getattr(app_config, "MAX_SEQ_LEN", 512)]
            ids_t = torch.tensor(ids, dtype=torch.long, device=model_manager.device)
            q_emb = model_manager._compute_query_embedding(ids_t)

            rr = router.route(q_emb)

            # --- log prompt routing (no generation yet)
            try:
                model_manager._log_input_output(
                    prompt=text,
                    completion="",
                    query_embedding=q_emb,
                    routing_result=rr,
                    domain_prior=None,
                    domain_mask=None,
                )

                # If routing says "unknown", mirror ModelManager's emergent logging behavior
                if rr is not None and getattr(rr, "is_unknown", False):
                    model_manager._handle_emergent_query(
                        prompt=text,
                        completion="",
                        query_embedding=q_emb,
                        routing_result=rr,
                    )
            except Exception:
                pass  # logging must never break UI


            q3 = _project_3d(reducer, q_emb)
            topk = _route_topk(rr, TOPK_DOMAINS)
            rows = _metrics_rows(rr)

            pts = (trail_data or {}).get("pts", [])
            pts.append([float(q3[0]), float(q3[1]), float(q3[2])])
            pts = pts[-MAX_TRAIL:]
            trail3 = np.array(pts, dtype=np.float32) if len(pts) else None

            fig = build_figure(
                cloud3=cloud3,
                labels=labels,
                centroid3=centroid3,
                centroid_names=centroid_names,
                query3=q3,
                trail3=trail3,
                topk=topk,
                camera=camera_data,
            )

            summary = (
                "[Prompt submit]\n"
                f"best_domain: {getattr(rr,'best_domain',None)}\n"
                f"is_unknown:  {getattr(rr,'is_unknown',None)}\n"
                f"reason:      {getattr(rr,'reason','')}\n"
                f"top_k:       {[t['name'] for t in topk]}\n"
            )

            return fig, summary, rows, "", {"pts": pts}, {"text": text}

        # ---------------------
        # Mode: GEN TRACE (live)
        # ---------------------
        if mode == "gen":
            frames = (gen_store or {}).get("frames", []) or []
            running = bool((gen_store or {}).get("running", False))
            err = (gen_store or {}).get("error", None)

            if not frames:
                msg = "[Generate trace]\n(no frames yet)\n"
                if err:
                    msg += f"error: {err}\n"
                return no_update, msg, [], "", no_update, no_update

            idx = len(frames) - 1 if running else max(0, min(int(frame_idx or 0), len(frames) - 1))
            f = frames[idx]

            q3 = np.array(f["q3"], dtype=np.float32)
            trail = np.array([fr["q3"] for fr in frames[: idx + 1]], dtype=np.float32) if idx >= 1 else None
            topk = f.get("topk", []) or []
            rows = f.get("metrics", []) or []
            gen_text = f.get("text", "") or ""

            fig = build_figure(
                cloud3=cloud3,
                labels=labels,
                centroid3=centroid3,
                centroid_names=centroid_names,
                query3=q3,
                trail3=trail,
                topk=topk,
                camera=camera_data,
            )

            summary = (
                "[Generate trace]\n"
                f"step:       {f.get('step')}\n"
                f"best_domain:{f.get('best_domain')}\n"
                f"is_unknown: {f.get('is_unknown')}\n"
                f"reason:     {f.get('reason')}\n"
                f"top_k:      {[t['name'] for t in topk]}\n"
            )

            return fig, summary, rows, gen_text, no_update, no_update

        # ---------------------
        # Mode: PLAYBACK
        # ---------------------
        if mode == "playback":
            frames = (gen_store or {}).get("frames", []) or []
            err = (gen_store or {}).get("error", None)

            if not frames:
                msg = "[Playback]\n(no frames yet)\n"
                if err:
                    msg += f"error: {err}\n"
                return no_update, msg, [], "", no_update, no_update

            idx = max(0, min(int(frame_idx or 0), len(frames) - 1))
            f = frames[idx]

            q3 = np.array(f["q3"], dtype=np.float32)
            trail = np.array([fr["q3"] for fr in frames[: idx + 1]], dtype=np.float32) if idx >= 1 else None
            topk = f.get("topk", []) or []
            rows = f.get("metrics", []) or []
            gen_text = f.get("text", "") or ""

            fig = build_figure(
                cloud3=cloud3,
                labels=labels,
                centroid3=centroid3,
                centroid_names=centroid_names,
                query3=q3,
                trail3=trail,
                topk=topk,
                camera=camera_data,
            )

            summary = (
                "[Playback]\n"
                f"step:       {f.get('step')}\n"
                f"best_domain:{f.get('best_domain')}\n"
                f"is_unknown: {f.get('is_unknown')}\n"
                f"reason:     {f.get('reason')}\n"
                f"top_k:      {[t['name'] for t in topk]}\n"
            )

            return fig, summary, rows, gen_text, no_update, no_update

        return no_update, no_update, no_update, no_update, no_update, no_update

    # Run
    app.run(host="0.0.0.0", port=8050, debug=True)


if __name__ == "__main__":
    main()

