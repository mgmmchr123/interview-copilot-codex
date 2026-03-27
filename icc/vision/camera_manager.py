from __future__ import annotations

import logging
import threading
import os

import cv2


logger = logging.getLogger(__name__)
DEFAULT_CAMERA_WIDTH = int(os.getenv("CAMERA_WIDTH", "1920"))
DEFAULT_CAMERA_HEIGHT = int(os.getenv("CAMERA_HEIGHT", "1080"))


class CameraManager:
    def __init__(self, camera_index: int):
        self.camera_index = camera_index
        self._cap: cv2.VideoCapture | None = None
        self._lock = threading.Lock()

    def warmup(self) -> None:
        """Open camera at the final resolution and keep it ready for preview."""
        with self._lock:
            if self._cap is not None and self._cap.isOpened():
                return
            self._cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, DEFAULT_CAMERA_WIDTH)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, DEFAULT_CAMERA_HEIGHT)
            for _ in range(5):
                self._cap.read()
            logger.info("CameraManager warmed up at index %s", self.camera_index)

    def get_cap(self) -> cv2.VideoCapture | None:
        return self._cap

    def release(self) -> None:
        with self._lock:
            if self._cap:
                self._cap.release()
                self._cap = None
