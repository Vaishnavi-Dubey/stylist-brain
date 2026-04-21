#!/usr/bin/env bash
# =============================================================================
# setup.sh — One-time setup for The Stylist's Brain
# Run once from the project root: bash setup.sh
# =============================================================================
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="$ROOT/.venv"
MODELS="$ROOT/models"

GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'
ok()   { echo -e "${GREEN}✓${NC} $*"; }
warn() { echo -e "${YELLOW}⚠${NC}  $*"; }
err()  { echo -e "${RED}✗${NC}  $*"; exit 1; }
step() { echo -e "\n${YELLOW}▶ $*${NC}"; }

echo ""
echo "╔══════════════════════════════════════════╗"
echo "║     The Stylist's Brain — Setup          ║"
echo "║     Local · Free · Private               ║"
echo "╚══════════════════════════════════════════╝"
echo ""

# ── 1. Python version check ───────────────────────────────────────────────────
step "Checking Python version"
PYTHON=$(command -v python3.11 2>/dev/null || command -v python3 2>/dev/null || true)
[[ -z "$PYTHON" ]] && err "Python 3.11+ is required. Install via: brew install python@3.11"

PY_VERSION=$("$PYTHON" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
PY_MAJOR=$("$PYTHON" -c "import sys; print(sys.version_info.major)")
PY_MINOR=$("$PYTHON" -c "import sys; print(sys.version_info.minor)")

if [[ "$PY_MAJOR" -lt 3 ]] || { [[ "$PY_MAJOR" -eq 3 ]] && [[ "$PY_MINOR" -lt 11 ]]; }; then
    err "Python 3.11+ required, found $PY_VERSION. Install via: brew install python@3.11"
fi
ok "Python $PY_VERSION"

# ── 2. Create virtual environment ─────────────────────────────────────────────
step "Creating virtual environment"
if [[ -d "$VENV" ]]; then
    warn ".venv already exists — skipping creation"
else
    "$PYTHON" -m venv "$VENV"
    ok "Created .venv"
fi

source "$VENV/bin/activate"

# ── 3. Upgrade pip silently ────────────────────────────────────────────────────
step "Upgrading pip"
pip install --upgrade pip --quiet
ok "pip up to date"

# ── 4. Install Python requirements ────────────────────────────────────────────
step "Installing Python requirements (this may take 3–5 minutes)"
echo "   Installing: fastapi uvicorn ultralytics open_clip_torch torch chromadb..."
pip install -r "$ROOT/requirements.txt" --quiet
ok "Core requirements installed"

# ── 5. Install MobileSAM from GitHub ─────────────────────────────────────────
step "Installing MobileSAM (from GitHub)"
if python -c "import mobile_sam" 2>/dev/null; then
    ok "MobileSAM already installed"
else
    pip install "git+https://github.com/ChaoningZhang/MobileSAM.git" --quiet
    ok "MobileSAM installed"
fi

# ── 6. Download model weights ─────────────────────────────────────────────────
step "Downloading model weights"
mkdir -p "$MODELS"

# YOLOv8n — nano variant only (lightest, MacBook Air safe)
YOLO_PT="$MODELS/yolov8n.pt"
if [[ -f "$YOLO_PT" ]]; then
    ok "yolov8n.pt already exists"
else
    echo "   Downloading YOLOv8n (~6 MB)..."
    python -c "from ultralytics import YOLO; m=YOLO('yolov8n.pt'); import shutil, pathlib; shutil.copy(next(pathlib.Path.home().glob('**/.config/Ultralytics/yolov8n.pt'), 'yolov8n.pt'), '$YOLO_PT')" 2>/dev/null || \
    curl -L --progress-bar -o "$YOLO_PT" \
        "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n.pt"
    ok "yolov8n.pt downloaded"
fi

# MobileSAM weights
SAM_PT="$MODELS/mobile_sam.pt"
if [[ -f "$SAM_PT" ]]; then
    ok "mobile_sam.pt already exists"
else
    echo "   Downloading MobileSAM weights (~40 MB)..."
    curl -L --progress-bar -o "$SAM_PT" \
        "https://github.com/ChaoningZhang/MobileSAM/raw/master/weights/mobile_sam.pt"
    ok "mobile_sam.pt downloaded"
fi

# ── 7. Ollama check ───────────────────────────────────────────────────────────
step "Checking Ollama"
if ! command -v ollama &>/dev/null; then
    warn "Ollama not found. Install with: brew install ollama"
    warn "Then run: ollama pull llama3:8b"
    warn "Then run: ollama serve"
else
    ok "Ollama found at $(command -v ollama)"
    # Pull llama3:8b if not already present
    if ollama list 2>/dev/null | grep -q "llama3:8b"; then
        ok "llama3:8b already pulled"
    else
        echo "   Pulling llama3:8b (~4.7 GB) — this takes a few minutes..."
        ollama pull llama3:8b
        ok "llama3:8b ready"
    fi
fi

# ── 8. Environment file ───────────────────────────────────────────────────────
step "Setting up environment"
ENV_FILE="$ROOT/.env"
if [[ -f "$ENV_FILE" ]]; then
    ok ".env already exists"
else
    cp "$ROOT/.env.example" "$ENV_FILE"
    warn ".env created from .env.example"
    warn "Edit .env and set OWM_API_KEY (free at openweathermap.org)"
    warn "Set WEATHER_CITY to your city name"
fi

# ── 9. Run preflight check ────────────────────────────────────────────────────
step "Running preflight check"
cd "$ROOT/backend"
python check.py || warn "Some checks failed — see above. The app may still work partially."

echo ""
echo "╔══════════════════════════════════════════╗"
echo "║  ✅  Setup complete!                     ║"
echo "║                                          ║"
echo "║  Start the app:  bash run.sh             ║"
echo "╚══════════════════════════════════════════╝"
echo ""
