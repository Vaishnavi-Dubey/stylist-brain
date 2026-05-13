# Contributing to The Stylist's Brain

Thank you for your interest in contributing! This project combines computer vision, LLMs, and vector databases — there's a lot of surface area for impactful contributions.

## 🛠️ Development Setup

```bash
git clone https://github.com/Vaishnavi-Dubey/stylist-brain.git
cd stylist-brain
bash setup.sh              # Creates venv + installs deps
bash download_models.sh    # Fetches YOLO + MobileSAM weights
cp .env.example .env       # Add your OpenWeatherMap key
bash run.sh                # Starts Ollama + FastAPI
```

## 🧭 Architecture Overview

The backend is modular — each directory is an independent concern:

| Module | Responsibility | Key files |
|---|---|---|
| `intake/` | Image upload → CV pipeline (YOLO, SAM, CLIP, color extraction) | `pipeline.py`, `detect.py`, `segment.py`, `embed.py` |
| `imaging/` | Visual output generation (flat-lay composition) | `flatlay.py`, `generate.py` |
| `styling/` | LLM-powered outfit recommendation engine | `llm.py`, `query.py`, `gap.py`, `habits.py` |
| `context/` | Environmental context (weather, calendar) | `weather.py`, `calendar.py` |
| `db/` | Dual storage: SQLite (metadata) + ChromaDB (vectors) | `sqlite.py`, `chroma.py` |

## 🎯 Good First Issues

- Improve the frontend UI (currently a single `index.html`)
- Add more garment categories to the detection pipeline
- Improve color naming accuracy in `tag.py`
- Add unit tests for the styling module

## 📝 Pull Request Guidelines

1. Fork the repo and create your branch from `main`
2. If you've added code that should be tested, add tests
3. Ensure your code lints cleanly: `ruff check backend/`
4. Make sure the server still starts: `bash run.sh`
5. Use descriptive commit messages (e.g., `Add: color harmony scoring`, `Fix: SAM mask cleanup`)
6. Open your Pull Request!

## ⚙️ Code Style

- Python 3.10+ features are welcome (match/case, type hints, etc.)
- We use [Ruff](https://docs.astral.sh/ruff/) for linting
- Keep imports at the top of each file (except for lazy-loaded ML models)
- Docstrings on all public functions

## 🧪 Running Tests

```bash
cd backend
python -m pytest intake/test_intake.py -v
```

## 📜 License

By contributing, you agree that your contributions will be licensed under the MIT License.
