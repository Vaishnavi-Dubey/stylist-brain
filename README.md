# The Stylist's Brain

Context-aware wardrobe styling — 100% local, 100% free, runs on a MacBook Air.

## Stack
| Layer | Tool |
|---|---|
| Garment detection | YOLOv8n (nano) |
| Segmentation | MobileSAM |
| Style embeddings | CLIP ViT-B/32 via open_clip |
| Vector DB | ChromaDB (local) |
| LLM | Ollama — llama3:8b or mistral:7b |
| Backend | FastAPI + uvicorn |
| Frontend | Plain HTML/JS |
| Weather | OpenWeatherMap free tier |
| Calendar | Google Calendar API (free OAuth) |

---

## Quick Start

### 1. Install dependencies
```bash
cd stylist-brain

# Create a virtual environment
python3.11 -m venv .venv && source .venv/bin/activate

# Install Python packages
pip install -r requirements.txt

# Install MobileSAM (not on PyPI)
pip install git+https://github.com/ChaoningZhang/MobileSAM.git

# Download MobileSAM weights
mkdir -p models
curl -L -o models/mobile_sam.pt \
  https://github.com/ChaoningZhang/MobileSAM/raw/master/weights/mobile_sam.pt
```

### 2. Install and start Ollama
```bash
# Install Ollama (free, local)
brew install ollama          # macOS

# Pull a 7-8B model (do NOT use 13B or 70B — too heavy for MacBook Air)
ollama pull llama3:8b        # recommended
# or: ollama pull mistral:7b

# Start Ollama server (runs in background)
ollama serve
```

### 3. Configure environment
```bash
cp .env.example .env
# Edit .env — add your free OpenWeatherMap API key
# Sign up at https://openweathermap.org/api (no credit card)
```

### 4. Run the backend
```bash
cd backend
uvicorn main:app --reload --port 8000
```

### 5. Open the frontend
Open `frontend/index.html` in your browser.

---

## Environment Variables
```
OWM_API_KEY=your_free_openweathermap_key
WEATHER_CITY=Mumbai
```

---

## RAM Budget (MacBook Air)
| Model | RAM |
|---|---|
| YOLOv8n | ~80 MB |
| MobileSAM | ~40 MB |
| CLIP ViT-B/32 | ~340 MB |
| Ollama llama3:8b (4-bit) | ~5.5 GB |
| ChromaDB | disk-backed, minimal RAM |
| **Total peak** | **~6.5 GB** |

Models are lazy-loaded and unloaded after each pipeline run.  
Never run YOLO, SAM, CLIP, and Ollama simultaneously.

---

## Feature Roadmap
- [x] Scaffold
- [ ] Wardrobe intake pipeline
- [ ] Vibe query engine
- [ ] Rule of Three engine
- [ ] Habit lock system
- [ ] Gap analysis
- [ ] Weather + calendar context
- [ ] AR overlay (MediaPipe v2 — post-MVP)
