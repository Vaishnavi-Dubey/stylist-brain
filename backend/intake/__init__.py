"""
Wardrobe Intake package.
Pipeline: video → frames → YOLOv8n detect → MobileSAM segment → CLIP embed → ChromaDB store
"""

from .routes import router  # noqa: F401 — exposes router to main.py
