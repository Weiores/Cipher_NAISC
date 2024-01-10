#!/usr/bin/env bash
# Cipher_NAISC – Unix/macOS start script
# Launches the backend API, UI layer, and frontend dashboard.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "============================================================"
echo " Cipher_NAISC – Starting"
echo "============================================================"

# ── .env check ───────────────────────────────────────────────────
if [ ! -f ".env" ]; then
    if [ -f ".env.example" ]; then
        echo "[start] WARNING: .env not found. Run setup first:"
        echo "  cp .env.example .env && nano .env"
    else
        echo "[start] WARNING: .env not found."
    fi
fi

# ── Quick dependency check ────────────────────────────────────────
if ! python3 -c "import groq" 2>/dev/null; then
    echo "[start] Python dependencies missing – installing now ..."
    pip install -r requirements.txt
fi

if [ ! -d "frontend/node_modules" ]; then
    echo "[start] Frontend node_modules missing – installing now ..."
    (cd frontend && npm install)
fi

exec python3 launcher.py "$@"
