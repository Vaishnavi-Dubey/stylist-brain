"""
The Stylist's Brain — FastAPI entry point
Runs 100% locally. No paid APIs. No cloud.
"""

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from db.sqlite import init_db
from db.chroma import init_chroma

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

_FRONTEND_DIR = Path(__file__).parents[1] / "frontend"
_OUTPUT_DIR   = Path(__file__).parents[1] / "output"
_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup / shutdown lifecycle."""
    logger.info("Starting The Stylist's Brain...")
    init_db()
    init_chroma()
    yield
    logger.info("Shutting down — unloading models...")


app = FastAPI(
    title="The Stylist's Brain",
    description="Context-aware wardrobe styling — local, free, private.",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── API Routers ───────────────────────────────────────────────────────────────
from intake import router as intake_router          # noqa: E402
from styling import router as styling_router        # noqa: E402
from context import router as context_router        # noqa: E402

app.include_router(intake_router,  prefix="/intake",  tags=["Wardrobe Intake"])
app.include_router(styling_router, prefix="/styling", tags=["Outfit Styling"])
app.include_router(context_router, prefix="/context", tags=["Context (weather/calendar)"])

# ── Static file mounts (must come AFTER API routers) ─────────────────────────
app.mount("/output",   StaticFiles(directory=str(_OUTPUT_DIR)),   name="output")
app.mount("/frontend", StaticFiles(directory=str(_FRONTEND_DIR)), name="frontend")


@app.get("/health")
def health():
    return {"status": "ok", "service": "stylist-brain"}


@app.get("/", include_in_schema=False)
def serve_ui():
    """Serve the frontend so all API calls are same-origin (no CORS issues)."""
    return FileResponse(str(_FRONTEND_DIR / "index.html"))
