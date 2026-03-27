from __future__ import annotations

import base64
import logging
import os
from datetime import datetime
from io import BytesIO
from pathlib import Path

import cv2
import PIL.Image


logger = logging.getLogger(__name__)
DEBUG_PREVIEW_PATH = Path("debug_screenshots/last_capture.png")
CAPTURES_DIR = Path("captures")
DEFAULT_CAMERA_WIDTH = int(os.getenv("CAMERA_WIDTH", "1920"))
DEFAULT_CAMERA_HEIGHT = int(os.getenv("CAMERA_HEIGHT", "1080"))


def capture_frame(camera_index: int | None = None) -> str:
    """
    Capture a single frame from the specified camera
    and return as base64 PNG string.
    """
    resolved_camera_index = _resolve_camera_index(camera_index)
    cap = cv2.VideoCapture(resolved_camera_index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        logger.error(
            "\u274c Could not open camera at index %s. "
            "Try setting CAMERA_INDEX=1 or CAMERA_INDEX=2",
            resolved_camera_index,
        )
        raise RuntimeError(
            f"Could not open camera at index {resolved_camera_index}. "
            "Try setting CAMERA_INDEX=1 or CAMERA_INDEX=2"
        )
    try:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, DEFAULT_CAMERA_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, DEFAULT_CAMERA_HEIGHT)
        cap.read()
        actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        logger.info("Camera resolution: %dx%d", actual_w, actual_h)
        for _ in range(3):
            cap.read()
        ret, frame = cap.read()
        if not ret or frame is None:
            raise RuntimeError("Failed to capture frame from camera.")
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = PIL.Image.fromarray(frame_rgb)
        save_capture_image(img)
        _save_debug_preview(img)
        buffer = BytesIO()
        img.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")
    finally:
        cap.release()


def _resolve_camera_index(camera_index: int | None) -> int:
    if camera_index is not None:
        return camera_index
    return int(os.getenv("CAMERA_INDEX", "0"))


def _save_debug_preview(img: PIL.Image.Image) -> None:
    if os.getenv("DEBUG_SCREENSHOT", "").strip() != "1":
        return

    DEBUG_PREVIEW_PATH.parent.mkdir(parents=True, exist_ok=True)
    img.save(DEBUG_PREVIEW_PATH, format="PNG")
    logger.info("\U0001f4f8 Debug preview saved to %s", DEBUG_PREVIEW_PATH.as_posix())


def save_capture_image(img: PIL.Image.Image) -> Path:
    CAPTURES_DIR.mkdir(parents=True, exist_ok=True)
    capture_path = CAPTURES_DIR / _build_capture_filename()
    img.save(capture_path, format="PNG")
    logger.info("\U0001f4f8 Captured image saved to %s", capture_path.as_posix())
    return capture_path


def image_to_base64(img: PIL.Image.Image) -> str:
    buffer = BytesIO()
    img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def _build_capture_filename() -> str:
    return f"capture_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
