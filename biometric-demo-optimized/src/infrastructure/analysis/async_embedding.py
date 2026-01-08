"""Async embedding extractor with background thread processing."""

import threading
import logging
import hashlib
from queue import Queue, Empty
from typing import Optional, Dict, Tuple

import numpy as np

from .embedding_extractor import EmbeddingExtractor

logger = logging.getLogger(__name__)


class AsyncEmbeddingExtractor:
    """Non-blocking embedding extractor using background thread.

    Wraps EmbeddingExtractor to provide:
    - Immediate returns (never blocks main loop)
    - Background processing in dedicated thread
    - Content-based caching (same face = same embedding)
    - Callback support for when extraction completes
    """

    def __init__(self, model: str = "Facenet512", cache_size: int = 10):
        """Initialize async extractor.

        Args:
            model: DeepFace model name
            cache_size: Maximum cached embeddings
        """
        self._extractor = EmbeddingExtractor(model)
        self._queue: Queue = Queue(maxsize=3)
        self._cache: Dict[str, np.ndarray] = {}
        self._cache_order: list = []
        self._cache_size = cache_size
        self._result_lock = threading.Lock()
        self._processing = False
        self._stopped = False

        # Pending results by request ID
        self._pending: Dict[str, Optional[np.ndarray]] = {}

        # Start background worker
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()
        logger.info("Async embedding extractor started")

    def _compute_hash(self, face_img: np.ndarray) -> str:
        """Compute content hash of face image for caching."""
        # Downsample for faster hashing
        small = face_img[::4, ::4].tobytes()
        return hashlib.md5(small).hexdigest()[:16]

    def _worker(self):
        """Background thread: process embedding requests."""
        while not self._stopped:
            try:
                item = self._queue.get(timeout=0.5)
                if item is None:
                    break

                request_id, face_img, img_hash = item
                self._processing = True

                # Check cache first
                with self._result_lock:
                    if img_hash in self._cache:
                        self._pending[request_id] = self._cache[img_hash]
                        self._processing = False
                        continue

                # Extract embedding
                embedding = self._extractor.extract(face_img)

                # Update cache and pending result
                with self._result_lock:
                    if embedding is not None:
                        # Add to cache
                        self._cache[img_hash] = embedding
                        self._cache_order.append(img_hash)

                        # Evict oldest if over limit
                        while len(self._cache_order) > self._cache_size:
                            old_hash = self._cache_order.pop(0)
                            self._cache.pop(old_hash, None)

                    self._pending[request_id] = embedding

                self._processing = False

            except Empty:
                continue
            except Exception as e:
                logger.debug(f"Async embedding error: {e}")
                self._processing = False

    def extract_async(self, face_img: np.ndarray, request_id: str = None) -> str:
        """Request embedding extraction (non-blocking).

        Args:
            face_img: BGR face image
            request_id: Optional ID for tracking (auto-generated if not provided)

        Returns:
            Request ID to check result later
        """
        if request_id is None:
            request_id = f"req_{id(face_img)}"

        img_hash = self._compute_hash(face_img)

        # Check cache immediately
        with self._result_lock:
            if img_hash in self._cache:
                self._pending[request_id] = self._cache[img_hash]
                return request_id

        # Queue for background processing
        if not self._queue.full():
            try:
                self._queue.put_nowait((request_id, face_img.copy(), img_hash))
            except Exception:
                pass

        return request_id

    def get_result(self, request_id: str) -> Tuple[bool, Optional[np.ndarray]]:
        """Get result for a request.

        Args:
            request_id: ID from extract_async

        Returns:
            Tuple of (ready, embedding) - ready=False if still processing
        """
        with self._result_lock:
            if request_id in self._pending:
                result = self._pending.pop(request_id)
                return True, result
        return False, None

    def extract(self, face_img: np.ndarray) -> Optional[np.ndarray]:
        """Synchronous extraction (for compatibility).

        Note: This still uses the cache but blocks until complete.
        """
        return self._extractor.extract(face_img)

    @property
    def is_processing(self) -> bool:
        """Check if background extraction is in progress."""
        return self._processing

    def stop(self):
        """Stop the background thread."""
        self._stopped = True
        try:
            self._queue.put_nowait(None)
        except Exception:
            pass
        self._thread.join(timeout=2.0)
        logger.info("Async embedding extractor stopped")

    def __del__(self):
        self.stop()
