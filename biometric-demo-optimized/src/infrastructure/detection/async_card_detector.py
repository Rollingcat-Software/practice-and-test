"""Async card detector with background thread processing."""

import threading
import logging
from queue import Queue, Empty
from typing import Dict, Optional

import numpy as np

from .card_detector import CardDetector

logger = logging.getLogger(__name__)


class AsyncCardDetector:
    """Non-blocking card detector using background thread.

    Wraps CardDetector to provide:
    - Immediate returns (never blocks main loop)
    - Background processing in dedicated thread
    - Last-known result caching
    """

    def __init__(self):
        self._detector = CardDetector()
        self._queue: Queue = Queue(maxsize=2)
        self._result: Dict = {'detected': False}
        self._result_lock = threading.Lock()
        self._processing = False
        self._stopped = False

        # Start background worker
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()
        logger.info("Async card detector started")

    def _worker(self):
        """Background thread: process card detection requests."""
        while not self._stopped:
            try:
                item = self._queue.get(timeout=0.5)
                if item is None:
                    break

                frame, use_smoothing = item
                self._processing = True

                result = self._detector.detect(frame, use_smoothing)

                with self._result_lock:
                    self._result = result

                self._processing = False

            except Empty:
                continue
            except Exception as e:
                logger.debug(f"Async card detection error: {e}")
                self._processing = False

    def detect(self, frame: np.ndarray, use_smoothing: bool = True) -> Dict:
        """Request card detection (non-blocking).

        Args:
            frame: BGR image
            use_smoothing: Apply temporal smoothing

        Returns:
            Last known detection result
        """
        if not self._queue.full():
            try:
                self._queue.put_nowait((frame.copy(), use_smoothing))
            except Exception:
                pass

        with self._result_lock:
            return self._result.copy()

    def is_available(self) -> bool:
        """Check if card detection is available."""
        return self._detector.is_available()

    @property
    def is_processing(self) -> bool:
        """Check if background detection is in progress."""
        return self._processing

    def stop(self):
        """Stop the background thread."""
        self._stopped = True
        try:
            self._queue.put_nowait(None)
        except Exception:
            pass
        self._thread.join(timeout=2.0)
        logger.info("Async card detector stopped")

    def __del__(self):
        self.stop()
