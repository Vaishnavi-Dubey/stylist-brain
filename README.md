# 🚀 Stylist Brain — AI-Powered Personal Styling Assistant

<div align="center">

[![License](https://img.shields.io/badge/License-MIT-blue.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)
[![Issues](https://img.shields.io/github/issues/Vaishnavi-Dubey/stylist-brain.svg?style=for-the-badge)](https://github.com/Vaishnavi-Dubey/stylist-brain/issues)
[![Stars](https://img.shields.io/github/stars/Vaishnavi-Dubey/stylist-brain.svg?style=for-the-badge)](https://github.com/Vaishnavi-Dubey/stylist-brain/stargazers)

![Python](https://img.shields.io/badge/Python-14354C?style=for-the-badge&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![YOLO](https://img.shields.io/badge/YOLOv8-111111?style=for-the-badge&logo=yolo&logoColor=white)

</div>

> A cutting-edge AI styling assistant that uses **computer vision** (YOLOv8, CLIP, MobileSAM) and **vector databases** (ChromaDB) to analyze your wardrobe, detect garments, extract dominant colors, and deliver personalized outfit recommendations — all powered by a FastAPI backend.

---

## ✨ Key Features

- 👗 **Garment Detection** — YOLOv8 nano model identifies clothing items in uploaded images
- 🎨 **Color Extraction** — K-means clustering extracts dominant colors from detected garments
- 🧠 **CLIP Embeddings** — OpenAI CLIP (ViT-B/32) generates semantic embeddings for style matching
- ✂️ **Background Removal** — rembg (U2Net) isolates garments from backgrounds for clean analysis
- 🔍 **Smart Segmentation** — MobileSAM provides precise garment segmentation masks
- 📊 **Vector Search** — ChromaDB stores and retrieves style vectors for similarity-based recommendations
- 🌤️ **Weather-Aware** — Integrates weather API for context-aware outfit suggestions
- 📅 **Calendar Integration** — Optional Google Calendar sync for event-appropriate styling

---

## 🧠 Tech Stack

| Layer | Technology |
|-------|-----------|
| **Backend Framework** | FastAPI, Uvicorn |
| **Object Detection** | YOLOv8 (Ultralytics nano) |
| **Image Embeddings** | OpenAI CLIP (ViT-B/32 via open_clip_torch) |
| **Segmentation** | MobileSAM |
| **Background Removal** | rembg (U2Net) |
| **Computer Vision** | OpenCV (headless), Pillow |
| **ML Framework** | PyTorch, Torchvision |
| **Vector Database** | ChromaDB |
| **Color Analysis** | Scikit-learn (K-means) |
| **API Calls** | httpx (Ollama + Weather APIs) |
| **Frontend** | HTML/JS (lightweight demo UI) |

---

## 🏗️ Architecture / How It Works

```
┌──────────────────────────────────────────────────────────┐
│                    User Upload (Image)                   │
└──────────────────┬───────────────────────────────────────┘
                   ▼
┌──────────────────────────────────────────────────────────┐
│  INTAKE MODULE — Image preprocessing & background removal│
│  rembg (U2Net) → clean garment isolation                 │
└──────────────────┬───────────────────────────────────────┘
                   ▼
┌──────────────────────────────────────────────────────────┐
│  IMAGING MODULE — Detection & Feature Extraction         │
│  YOLOv8n → garment bounding boxes                        │
│  MobileSAM → precise segmentation masks                  │
│  CLIP ViT-B/32 → semantic style embeddings               │
│  K-means → dominant color extraction                     │
└──────────────────┬───────────────────────────────────────┘
                   ▼
┌──────────────────────────────────────────────────────────┐
│  CONTEXT MODULE — Environmental awareness                │
│  Weather API → temperature, conditions                   │
│  Google Calendar → upcoming events                       │
└──────────────────┬───────────────────────────────────────┘
                   ▼
┌──────────────────────────────────────────────────────────┐
│  STYLING MODULE — Recommendation Engine                  │
│  ChromaDB → vector similarity search                     │
│  Context matching → weather + event + style preferences  │
│  → Personalized outfit recommendation                    │
└──────────────────────────────────────────────────────────┘
```

---

## ⚙️ Installation & Setup

### Prerequisites
- Python 3.10+
- pip

### Quick Start

```bash
# Clone the repository
git clone https://github.com/Vaishnavi-Dubey/stylist-brain.git
cd stylist-brain

# Install dependencies
pip install -r requirements.txt

# Install MobileSAM (from source)
pip install git+https://github.com/ChaoningZhang/MobileSAM.git

# Download MobileSAM weights
wget -P models/ https://github.com/ChaoningZhang/MobileSAM/raw/master/weights/mobile_sam.pt

# Configure environment
cp .env.example .env
# Edit .env with your OWM_API_KEY (OpenWeatherMap)

# Run the application
sh run.sh
# Or directly:
uvicorn backend.main:app --reload
```

### Environment Variables
| Variable | Description |
|----------|-------------|
| `OWM_API_KEY` | OpenWeatherMap API key for weather-aware styling |

---

## ▶️ Usage

1. Start the server: `sh run.sh`
2. Open `http://localhost:8000` in your browser
3. Upload a photo of your outfit or wardrobe
4. Receive AI-powered styling analysis and recommendations

---

## 📂 Project Structure

```
stylist-brain/
├── backend/
│   ├── main.py              # FastAPI app entry point
│   ├── intake/              # Image upload & preprocessing
│   ├── imaging/             # YOLOv8, CLIP, MobileSAM pipelines
│   ├── styling/             # Recommendation engine
│   ├── context/             # Weather + calendar integration
│   ├── db/                  # ChromaDB vector store
│   └── check.py             # Health check utilities
├── frontend/
│   └── index.html           # Lightweight demo interface
├── yolov8n.pt               # YOLOv8 nano pre-trained weights
├── requirements.txt         # Python dependencies
├── run.sh                   # Quick-start script
├── setup.sh                 # Environment setup script
├── .env.example             # Environment variable template
└── LICENSE                  # MIT License
```

---

## 📸 Screenshots / Demo

> Demo screenshots and video walkthrough coming soon!

---

## 📈 Impact / Learning / Highlights

- 🧠 **Multi-Model Pipeline** — Orchestrates 4+ AI models (YOLO, CLIP, SAM, U2Net) in a single inference pipeline
- ⚡ **Optimized for CPU** — Uses nano/lightweight model variants (YOLOv8n, ViT-B/32, MobileSAM) for MacBook Air compatibility
- 🎯 **Production Architecture** — Clean separation of concerns with modular backend (intake → imaging → context → styling)
- 🔬 **Advanced CV Techniques** — Combines object detection, semantic segmentation, and embedding-based similarity search
- 📊 **Vector Database Integration** — ChromaDB for efficient style similarity retrieval at scale

---

## 🤝 Contributing

Contributions are welcome! To contribute:

1. Fork this repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Commit your changes: `git commit -m 'Add: new feature'`
4. Push to the branch: `git push origin feature/your-feature`
5. Open a Pull Request

---

## 📜 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

<p align="center">
  <b>Built with ❤️ by <a href="https://github.com/Vaishnavi-Dubey">Vaishnavi Dubey</a></b>
</p>
