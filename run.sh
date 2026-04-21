#!/usr/bin/env bash
# =============================================================================
# run.sh — Start The Stylist's Brain
# Run from project root: bash run.sh
# =============================================================================
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="$ROOT/.venv"
PORT=8000

GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; BOLD='\033[1m'; NC='\033[0m'
ok()   { echo -e "${GREEN}✓${NC} $*"; }
warn() { echo -e "${YELLOW}⚠${NC}  $*"; }
err()  { echo -e "${RED}✗${NC}  $*"; exit 1; }
info() { echo -e "  $*"; }

echo ""
echo -e "${BOLD}The Stylist's Brain${NC}  —  starting up..."
echo ""

# ── Activate venv ──────────────────────────────────────────────────────────────
[[ ! -d "$VENV" ]] && err "Virtual environment not found. Run: bash setup.sh first."
source "$VENV/bin/activate"
ok "Virtual environment activated"

# ── Load .env ─────────────────────────────────────────────────────────────────
ENV_FILE="$ROOT/.env"
if [[ -f "$ENV_FILE" ]]; then
    set -a; source "$ENV_FILE"; set +a
    ok ".env loaded"
else
    warn ".env not found — weather context will use fallback values"
fi

# ── Ollama ────────────────────────────────────────────────────────────────────
if ! command -v ollama &>/dev/null; then
    err "Ollama not installed. Run: brew install ollama && ollama pull llama3:8b"
fi

if ! curl -sf "http://localhost:11434/api/tags" >/dev/null 2>&1; then
    echo -e "${YELLOW}▶ Starting Ollama server in background...${NC}"
    ollama serve &>/tmp/ollama.log &
    OLLAMA_PID=$!
    # Wait for it to be ready
    for i in $(seq 1 15); do
        sleep 1
        if curl -sf "http://localhost:11434/api/tags" >/dev/null 2>&1; then
            ok "Ollama server started (PID $OLLAMA_PID)"
            break
        fi
        if [[ $i -eq 15 ]]; then
            warn "Ollama not responding after 15s — outfit requests may fail"
        fi
    done
else
    ok "Ollama already running"
fi

# ── Verify model is available ─────────────────────────────────────────────────
if ollama list 2>/dev/null | grep -q "llama3:8b\|mistral:7b"; then
    MODEL=$(ollama list 2>/dev/null | grep -m1 "llama3:8b\|mistral:7b" | awk '{print $1}')
    ok "LLM ready: $MODEL"
else
    warn "No llama3:8b or mistral:7b found. Run: ollama pull llama3:8b"
fi

# ── Kill any existing server on the port ──────────────────────────────────────
EXISTING=$(lsof -ti tcp:$PORT 2>/dev/null || true)
if [[ -n "$EXISTING" ]]; then
    warn "Port $PORT in use (PID $EXISTING) — killing old server"
    kill "$EXISTING" 2>/dev/null || true
    sleep 1
fi

# ── Start FastAPI backend ─────────────────────────────────────────────────────
echo ""
echo -e "${YELLOW}▶ Starting FastAPI backend on port $PORT...${NC}"
cd "$ROOT/backend"

# Run preflight silently — just check if it passes
if ! python check.py --quiet 2>/dev/null; then
    warn "Preflight check had warnings — check logs if something fails"
fi

echo ""
echo "┌─────────────────────────────────────────────┐"
echo "│  Backend:   http://localhost:$PORT            │"
echo "│  API docs:  http://localhost:$PORT/docs       │"
echo "│  Frontend:  open frontend/index.html         │"
echo "└─────────────────────────────────────────────┘"
echo ""
echo -e "  Press ${BOLD}Ctrl+C${NC} to stop."
echo ""

# Open frontend in browser after a short delay (non-blocking)
(sleep 2 && open "$ROOT/frontend/index.html" 2>/dev/null || true) &

# Start uvicorn — this blocks until Ctrl+C
exec uvicorn main:app \
    --host 0.0.0.0 \
    --port $PORT \
    --reload \
    --reload-dir "$ROOT/backend" \
    --log-level info
