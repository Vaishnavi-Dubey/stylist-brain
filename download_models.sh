#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# download_models.sh — Fetch all AI model weights required by Stylist Brain.
# Run once after cloning. Weights are NOT stored in Git (too large).
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; BOLD='\033[1m'; NC='\033[0m'
ok()   { echo -e "${GREEN}✓${NC} $*"; }
warn() { echo -e "${YELLOW}⚠${NC}  $*"; }
err()  { echo -e "${RED}✗${NC}  $*"; exit 1; }

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODELS_DIR="$ROOT/models"
mkdir -p "$MODELS_DIR"

echo ""
echo -e "${BOLD}📦 Downloading model weights for The Stylist's Brain...${NC}"
echo ""

# ── 1. YOLOv8 nano ───────────────────────────────────────────────────────────
YOLO_FILE="$ROOT/yolov8n.pt"
if [[ -f "$YOLO_FILE" ]]; then
    ok "YOLOv8 nano already present ($(du -h "$YOLO_FILE" | cut -f1))"
else
    echo "  Downloading YOLOv8 nano (≈6.2 MB)..."
    curl -fSL -o "$YOLO_FILE" \
        "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n.pt"
    ok "YOLOv8 nano downloaded ($(du -h "$YOLO_FILE" | cut -f1))"
fi

# ── 2. MobileSAM weights ─────────────────────────────────────────────────────
SAM_FILE="$MODELS_DIR/mobile_sam.pt"
if [[ -f "$SAM_FILE" ]]; then
    ok "MobileSAM already present ($(du -h "$SAM_FILE" | cut -f1))"
else
    echo "  Downloading MobileSAM weights (≈40 MB)..."
    curl -fSL -o "$SAM_FILE" \
        "https://github.com/ChaoningZhang/MobileSAM/raw/master/weights/mobile_sam.pt"
    ok "MobileSAM downloaded ($(du -h "$SAM_FILE" | cut -f1))"
fi

# ── 3. CLIP / U2Net / rembg ──────────────────────────────────────────────────
# These are auto-downloaded on first use by their respective libraries.
# open_clip_torch downloads ViT-B/32 on first import.
# rembg downloads U2Net on first call to remove().
warn "CLIP (ViT-B/32) and U2Net will auto-download on first API call (~170MB each)"

echo ""
echo -e "${GREEN}✓ All model weights ready.${NC}"
echo -e "  Run ${BOLD}bash run.sh${NC} to start the server."
echo ""
