"""
video.py — OpenCV frame extractor
Accepts a short wardrobe pan video and extracts deduplicated frames.

Pipeline position: step 1 of 4 (video → detect → segment → embed)
"""

import logging
from pathlib import Path

import cv2  # opencv-python-headless

logger = logging.getLogger(__name__)

# Extract 1 frame every N frames to avoid near-duplicate embeddings.
DEFAULT_FRAME_INTERVAL = 15   # ~0.5 s at 30 fps — adjust to taste
MIN_FRAME_WIDTH = 320          # discard tiny/corrupt frames


def extract_frames(
    video_path: str | Path,
    output_dir: str | Path,
    frame_interval: int = DEFAULT_FRAME_INTERVAL,
) -> list[Path]:
    """
    Extract frames from *video_path* and write JPEGs to *output_dir*.

    Args:
        video_path:     Path to the input .mp4 / .mov file.
        output_dir:     Directory where extracted frames are saved.
        frame_interval: Save 1 frame every this many frames.

    Returns:
        List of Paths to saved frame images.
    """
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV could not open video: {video_path}")

    saved: list[Path] = []
    frame_idx = 0
    saved_idx = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % frame_interval == 0:
                h, w = frame.shape[:2]
                if w < MIN_FRAME_WIDTH:
                    logger.debug("Skipping small frame %d (%dx%d)", frame_idx, w, h)
                    frame_idx += 1
                    continue

                out_path = output_dir / f"frame_{saved_idx:04d}.jpg"
                cv2.imwrite(str(out_path), frame)
                saved.append(out_path)
                saved_idx += 1

            frame_idx += 1
    finally:
        cap.release()

    logger.info("Extracted %d frames from %s", len(saved), video_path.name)
    return saved
