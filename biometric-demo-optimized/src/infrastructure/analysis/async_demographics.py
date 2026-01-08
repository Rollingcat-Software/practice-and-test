"""Async demographics analyzer with background thread processing."""

import threading
import logging
import time
from queue import Queue, Empty
from typing import Optional

import numpy as np

from ...domain.models import Demographics, FaceRegion
from .demographics_analyzer import DemographicsAnalyzer

logger = logging.getLogger(__name__)


class AsyncDemographicsAnalyzer:
    """Non-blocking demographics analyzer using background thread.

    Wraps DemographicsAnalyzer to provide:
    - Immediate returns (never blocks main loop)
    - Background processing in dedicated thread
    - Last-known result caching
    - Automatic request deduplication
    """

    def __init__(self, cache_interval: float = 2.0, throttle_interval: float = 1.5):
        self._analyzer = DemographicsAnalyzer(cache_interval, throttle_interval)
        self._queue: Queue = Queue(maxsize=2)  # Small queue to drop stale requests
        self._result: Demographics = Demographics()
        self._result_lock = threading.Lock()
        self._processing = False
        self._stopped = False

        # Start background worker
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()
        logger.info("Async demographics analyzer started")

    def _worker(self):
        """Background thread: process demographics requests."""
        while not self._stopped:
            try:
                # Wait for work with timeout
                item = self._queue.get(timeout=0.5)
                if item is None:  # Shutdown signal
                    break

                frame, region = item
                self._processing = True

                # Run the actual analysis
                result = self._analyzer.analyze(frame, region)

                # Update result thread-safely
                with self._result_lock:
                    self._result = result

                self._processing = False

            except Empty:
                continue
            except Exception as e:
                logger.debug(f"Async demographics error: {e}")
                self._processing = False

    def analyze(self, frame: np.ndarray, region: FaceRegion) -> Demographics:
        """Request demographics analysis (non-blocking).

        Submits request to background thread and immediately returns
        the last known result.

        Args:
            frame: Full BGR frame
            region: Face bounding box

        Returns:
            Last known Demographics (may be from previous analysis)
        """
        # Submit to queue if not full (drop if queue full - we want freshest data)
        if not self._queue.full():
            try:
                # Copy frame since it may be reused by caller
                self._queue.put_nowait((frame.copy(), region))
            except Exception:
                pass  # Queue full, skip this request

        # Return last known result immediately (never blocks)
        with self._result_lock:
            return self._result

    @property
    def is_processing(self) -> bool:
        """Check if background analysis is in progress."""
        return self._processing

    @property
    def last_result(self) -> Demographics:
        """Get last known result."""
        with self._result_lock:
            return self._result

    def stop(self):
        """Stop the background thread."""
        self._stopped = True
        try:
            self._queue.put_nowait(None)  # Send shutdown signal
        except Exception:
            pass
        self._thread.join(timeout=2.0)
        logger.info("Async demographics analyzer stopped")

    def __del__(self):
        self.stop()
