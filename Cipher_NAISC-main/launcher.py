"""
Cipher_NAISC – Unified launcher.

Starts all system components in the correct order:
  1. Backend API  (src/main.py  → FastAPI on :8000)
  2. UI layer     (ui-layer     → FastAPI on :8001)
  3. Frontend     (frontend     → Vite dev server on :5173)

Usage:
  python launcher.py [--no-frontend] [--no-ui-layer]
"""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent

PYTHON = sys.executable
NODE_MANAGER = "npm"


def _env_file_ok() -> bool:
    return (ROOT / ".env").exists()


def _check_env() -> None:
    if not _env_file_ok():
        env_example = ROOT / ".env.example"
        if env_example.exists():
            print(
                "[launcher] WARNING: .env not found. "
                "Copy .env.example → .env and fill in your API keys."
            )
        else:
            print("[launcher] WARNING: neither .env nor .env.example found.")


def _start_backend() -> subprocess.Popen:
    print("[launcher] Starting backend (src/main.py) on :8000 …")
    return subprocess.Popen(
        [PYTHON, "src/main.py"],
        cwd=ROOT,
    )


def _start_ui_layer() -> subprocess.Popen:
    print("[launcher] Starting UI layer (ui-layer) on :8001 …")
    return subprocess.Popen(
        [
            PYTHON, "-m", "uvicorn",
            "app.main:app",
            "--host", "0.0.0.0",
            "--port", "8001",
            "--reload",
        ],
        cwd=ROOT / "ui-layer",
    )


def _start_frontend() -> subprocess.Popen:
    print("[launcher] Starting frontend (Vite dev) on :5173 …")
    npm_cmd = ["npm.cmd" if sys.platform == "win32" else "npm", "run", "dev"]
    return subprocess.Popen(npm_cmd, cwd=ROOT / "frontend")


def main() -> None:
    parser = argparse.ArgumentParser(description="Cipher_NAISC launcher")
    parser.add_argument("--no-frontend", action="store_true", help="Skip Vite dev server")
    parser.add_argument("--no-ui-layer", action="store_true", help="Skip UI / Telegram layer")
    args = parser.parse_args()

    _check_env()

    processes: list[subprocess.Popen] = []

    try:
        processes.append(_start_backend())
        time.sleep(2)

        if not args.no_ui_layer:
            processes.append(_start_ui_layer())
            time.sleep(1)

        if not args.no_frontend:
            processes.append(_start_frontend())

        print("\n[launcher] All services running. Press Ctrl+C to stop.\n")
        print("  Backend API  → http://localhost:8000")
        print("  Backend docs → http://localhost:8000/docs")
        if not args.no_ui_layer:
            print("  UI layer     → http://localhost:8001")
        if not args.no_frontend:
            print("  Dashboard    → http://localhost:5173")
        print()

        # Wait until any child exits, then tear down everything
        while True:
            for p in processes:
                if p.poll() is not None:
                    print(f"[launcher] Process {p.pid} exited – shutting down.")
                    raise KeyboardInterrupt
            time.sleep(1)

    except KeyboardInterrupt:
        print("\n[launcher] Stopping all services …")
        for p in processes:
            try:
                if p.poll() is None:
                    p.send_signal(signal.CTRL_C_EVENT if sys.platform == "win32" else signal.SIGINT)
                    p.wait(timeout=5)
            except Exception:
                p.kill()
        print("[launcher] Done.")


if __name__ == "__main__":
    main()
