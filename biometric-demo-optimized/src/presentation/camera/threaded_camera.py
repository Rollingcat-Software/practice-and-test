"""Threaded camera capture for non-blocking I/O."""

import threading
import logging
from typing import Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class ThreadedCamera:
    """Camera capture in separate thread to prevent I/O blocking.

    Uses double-buffering to avoid frame copying overhead.
    Captures frames asynchronously, allowing the main processing
    loop to run at full speed without waiting for camera I/O.
    """

    def __init__(self, src: int = 0, width: int = 1280, height: int = 720):
        """Initialize camera.

        Args:
            src: Camera device index
            width: Capture width
            height: Capture height
        """
        self._stream = cv2.VideoCapture(src)
        self._stream.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self._stream.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        # Reduce buffer size for lower latency
        self._stream.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self._grabbed, self._frame = self._stream.read()
        self._stopped = False
        self._lock = threading.Lock()
        self._new_frame = threading.Event()

        if not self._grabbed:
            logger.error("Failed to open camera!")

        # Pre-allocate double buffer to avoid repeated allocations
        self._width = int(self._stream.get(cv2.CAP_PROP_FRAME_WIDTH))
        self._height = int(self._stream.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Double buffering: write to back, read from front
        self._front_buffer = None
        self._back_buffer = np.zeros((self._height, self._width, 3), dtype=np.uint8)

        logger.info(f"Camera initialized: {self._width}x{self._height}")

    @property
    def resolution(self) -> Tuple[int, int]:
        """Get camera resolution (width, height)."""
        return (self._width, self._height)

    @property
    def is_opened(self) -> bool:
        """Check if camera is opened successfully."""
        return self._grabbed

    def start(self) -> 'ThreadedCamera':
        """Start the background capture thread."""
        thread = threading.Thread(target=self._update, daemon=True)
        thread.start()
        return self

    def _update(self):
        """Background thread: continuously capture frames."""
        while not self._stopped:
            grabbed, frame = self._stream.read()
            if not grabbed:
                self.stop()
                break

            # Write to back buffer
            with self._lock:
                self._grabbed = grabbed
                if frame is not None:
                    # Use back buffer to avoid allocation
                    np.copyto(self._back_buffer, frame)
                    # Swap buffers
                    self._front_buffer, self._back_buffer = self._back_buffer, self._front_buffer
                    if self._back_buffer is None:
                        self._back_buffer = np.zeros((self._height, self._width, 3), dtype=np.uint8)

            # Signal new frame available
            self._new_frame.set()

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Read the latest frame.

        Returns:
            Tuple of (success, frame) - frame is a direct reference, do not modify!
        """
        with self._lock:
            if self._front_buffer is None:
                return False, None
            # Return reference to front buffer (caller should not modify)
            # If modification needed, caller should copy
            return self._grabbed, self._front_buffer

    def read_copy(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Read a copy of the latest frame (safe to modify).

        Returns:
            Tuple of (success, frame copy)
        """
        with self._lock:
            if self._front_buffer is None:
                return False, None
            return self._grabbed, self._front_buffer.copy()

    def wait_for_frame(self, timeout: float = 0.1) -> bool:
        """Wait for a new frame to be available.

        Args:
            timeout: Maximum wait time in seconds

        Returns:
            True if new frame available, False if timeout
        """
        result = self._new_frame.wait(timeout)
        self._new_frame.clear()
        return result

    def stop(self):
        """Stop the camera capture."""
        self._stopped = True
        self._stream.release()

    def __enter__(self):
        return self.start()

    def __exit__(self, *args):
        self.stop()
