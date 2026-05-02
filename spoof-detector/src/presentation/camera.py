"""Threaded camera capture for non-blocking I/O.

Ported from biometric-demo-optimized with double-buffering.
"""

import threading
import logging
from typing import Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class ThreadedCamera:
    """Camera capture in separate thread to prevent I/O blocking.

    Uses double-buffering to avoid frame copying overhead.
    """

    def __init__(self, src: int = 0, width: int = 1280, height: int = 720):
        self._stream = cv2.VideoCapture(src)
        self._stream.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self._stream.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self._stream.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self._grabbed, self._frame = self._stream.read()
        self._stopped = False
        self._lock = threading.Lock()
        self._new_frame = threading.Event()

        if not self._grabbed:
            logger.error("Failed to open camera!")

        self._width = int(self._stream.get(cv2.CAP_PROP_FRAME_WIDTH))
        self._height = int(self._stream.get(cv2.CAP_PROP_FRAME_HEIGHT))

        self._front_buffer: Optional[np.ndarray] = None
        self._back_buffer = np.zeros((self._height, self._width, 3), dtype=np.uint8)

        logger.info(f"Camera initialized: {self._width}x{self._height}")

    @property
    def resolution(self) -> Tuple[int, int]:
        return (self._width, self._height)

    @property
    def is_opened(self) -> bool:
        return self._grabbed

    def start(self) -> "ThreadedCamera":
        thread = threading.Thread(target=self._update, daemon=True)
        thread.start()
        return self

    def _update(self):
        while not self._stopped:
            grabbed, frame = self._stream.read()
            if not grabbed:
                self.stop()
                break
            with self._lock:
                self._grabbed = grabbed
                if frame is not None:
                    np.copyto(self._back_buffer, frame)
                    self._front_buffer, self._back_buffer = self._back_buffer, self._front_buffer
                    if self._back_buffer is None:
                        self._back_buffer = np.zeros(
                            (self._height, self._width, 3), dtype=np.uint8
                        )
            self._new_frame.set()

    def read_copy(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Read a copy of the latest frame (safe to modify)."""
        with self._lock:
            if self._front_buffer is None:
                return False, None
            return self._grabbed, self._front_buffer.copy()

    def stop(self):
        self._stopped = True
        self._stream.release()

    def __enter__(self):
        return self.start()

    def __exit__(self, *args):
        self.stop()
