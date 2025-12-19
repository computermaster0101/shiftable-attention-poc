#!/usr/bin/env python3
"""run_stack.py

One-command launcher for the Shiftable Attention PoC.

What it does:
  1) Blocks until training/initialization is complete (creates/loads artifacts).
  2) Ensures viz artifacts exist (embedding_map.npz + reducer.pkl).
  3) Starts BOTH:
       - FastAPI (uvicorn) at http://localhost:8000
       - Dash viz at       http://localhost:8050

Usage (from repo root):
  python run_stack.py

Optional:
  python run_stack.py --api-port 8000 --viz-port 8050 --no-viz-build

Notes:
  - This script intentionally starts servers in separate processes.
  - Ctrl+C will shut down both.
"""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parent


def _python() -> str:
    return sys.executable


def _run_init_blocking() -> None:
    """Import + run the model initialization in-process so it happens once."""
    repo_root = _repo_root()
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    # Local import so sys.path is set first.
    from app.model_manager import model_manager

    print("[run_stack] Ensuring model is initialized (training/migration may run)…", flush=True)
    t0 = time.time()
    model_manager.ensure_initialized()
    dt = time.time() - t0
    print(f"[run_stack] Initialization complete in {dt:.1f}s", flush=True)


def _ensure_viz_artifacts(method: str = "umap") -> None:
    repo_root = _repo_root()
    viz_dir = repo_root / "shiftable_project" / "outputs" / "viz"
    npz_path = viz_dir / "embedding_map.npz"
    reducer_path = viz_dir / "reducer.pkl"

    if npz_path.exists() and reducer_path.exists():
        print("[run_stack] Viz artifacts already exist.", flush=True)
        return

    print("[run_stack] Building viz artifacts (embedding_map.npz + reducer.pkl)…", flush=True)
    cmd = [_python(), str(repo_root / "viz" / "build_embedding_map.py"), "--method", method]
    subprocess.check_call(cmd, cwd=str(repo_root))
    if not (npz_path.exists() and reducer_path.exists()):
        raise RuntimeError("Viz build completed but expected artifacts are still missing.")


def _spawn_process(cmd: list[str], cwd: Path) -> subprocess.Popen:
    # On Linux, start in its own process group so we can SIGTERM the whole group.
    return subprocess.Popen(
        cmd,
        cwd=str(cwd),
        env=os.environ.copy(),
        preexec_fn=os.setsid if hasattr(os, "setsid") else None,
    )


def _terminate_process(p: subprocess.Popen, name: str, timeout_s: float = 8.0) -> None:
    if p.poll() is not None:
        return

    try:
        if hasattr(os, "killpg") and p.pid:
            os.killpg(p.pid, signal.SIGTERM)
        else:
            p.terminate()
    except Exception:
        pass

    t0 = time.time()
    while time.time() - t0 < timeout_s:
        if p.poll() is not None:
            return
        time.sleep(0.1)

    # Escalate
    try:
        if hasattr(os, "killpg") and p.pid:
            os.killpg(p.pid, signal.SIGKILL)
        else:
            p.kill()
    except Exception:
        pass


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--api-port", type=int, default=8000)
    ap.add_argument("--viz-port", type=int, default=8050)
    ap.add_argument("--viz-method", choices=["umap", "pca"], default="umap")
    ap.add_argument("--no-viz-build", action="store_true", help="Do not auto-build viz artifacts")
    ap.add_argument(
        "--reload",
        action="store_true",
        help="Pass --reload to uvicorn (dev only).",
    )
    args = ap.parse_args()

    repo_root = _repo_root()

    # 1) Run training/initialization once, blocking.
    _run_init_blocking()

    # 2) Ensure viz artifacts exist.
    if not args.no_viz_build:
        try:
            _ensure_viz_artifacts(method=args.viz_method)
        except Exception as e:
            print(
                "[run_stack] Viz artifact build failed. You can still run the API.\n"
                f"  Error: {e}",
                file=sys.stderr,
                flush=True,
            )

    # 3) Start servers.
    api_cmd = [
        _python(),
        "-m",
        "uvicorn",
        "main:app",
        "--host",
        args.host,
        "--port",
        str(args.api_port),
    ]
    if args.reload:
        api_cmd.append("--reload")

    # Dash script has the port hardcoded to 8050 in the repo currently.
    # We support --viz-port by exporting PORT and letting you adopt the small
    # optional patch below (see response). If you haven't patched dash_embedding_viewer.py,
    # it will still run on 8050.
    env = os.environ.copy()
    env["GRCLM_VIZ_PORT"] = str(args.viz_port)
    viz_cmd = [_python(), str(repo_root / "viz" / "dash_embedding_viewer.py")]

    print(f"[run_stack] Starting API: http://localhost:{args.api_port}", flush=True)
    api_p = subprocess.Popen(api_cmd, cwd=str(repo_root), env=env, preexec_fn=os.setsid if hasattr(os, "setsid") else None)

    print(f"[run_stack] Starting Viz: http://localhost:{args.viz_port}", flush=True)
    viz_p = subprocess.Popen(viz_cmd, cwd=str(repo_root), env=env, preexec_fn=os.setsid if hasattr(os, "setsid") else None)

    # 4) Wait; if either exits, shut down the other.
    try:
        while True:
            api_rc = api_p.poll()
            viz_rc = viz_p.poll()
            if api_rc is not None:
                print(f"[run_stack] API exited (code={api_rc}). Stopping viz…", flush=True)
                _terminate_process(viz_p, "viz")
                return int(api_rc)
            if viz_rc is not None:
                print(f"[run_stack] Viz exited (code={viz_rc}). Stopping API…", flush=True)
                _terminate_process(api_p, "api")
                return int(viz_rc)
            time.sleep(0.25)
    except KeyboardInterrupt:
        print("\n[run_stack] Ctrl+C received. Shutting down…", flush=True)
        _terminate_process(viz_p, "viz")
        _terminate_process(api_p, "api")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
