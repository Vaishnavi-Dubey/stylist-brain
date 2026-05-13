# ─────────────────────────────────────────────────────────────────────────────
# Dockerfile — The Stylist's Brain
# Multi-stage build: deps → runtime (keeps final image lean)
# ─────────────────────────────────────────────────────────────────────────────
FROM python:3.10-slim AS base

# System deps for OpenCV, image processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    curl \
    wget \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── Install Python dependencies ──────────────────────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir git+https://github.com/ChaoningZhang/MobileSAM.git

# ── Download model weights ───────────────────────────────────────────────────
RUN mkdir -p /app/models && \
    curl -fSL -o /app/yolov8n.pt \
        "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n.pt" && \
    curl -fSL -o /app/models/mobile_sam.pt \
        "https://github.com/ChaoningZhang/MobileSAM/raw/master/weights/mobile_sam.pt"

# ── Copy application code ────────────────────────────────────────────────────
COPY backend/ /app/backend/
COPY frontend/ /app/frontend/

# ── Create runtime directories ───────────────────────────────────────────────
RUN mkdir -p /app/output /app/wardrobe /app/chroma_store

WORKDIR /app/backend

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
