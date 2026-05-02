"""MiniFASNet anti-spoofing analyzer.

Ported from practice-and-test/uniface-evaluation/test_antispoof.py.
Uses UniFace's MiniFASNet ONNX model for binary real/spoof classification.
No PyTorch dependency required — runs on ONNX Runtime.

IMPORTANT: MiniFASNet requires the ORIGINAL full frame + face bbox,
not a pre-cropped face. The model uses surrounding context to judge
whether the face region looks like a real capture vs a screen/print.
Passing crop-only with bbox [0,0,w,h] produces unreliable results.
"""

import logging
import time

import cv2
import numpy as np

from src.domain.models import FaceROI, AnalyzerResult

logger = logging.getLogger(__name__)


class MiniFASNetAnalyzer:
    """Binary real/spoof detector using UniFace MiniFASNet ONNX.

    Performance: ~3ms per face on CPU.
    Model size: ~70MB ONNX.

    This analyzer needs access to the original frame (not just the crop)
    to provide proper context to the model. The pipeline stores the
    original frame reference, and analyze_with_frame() should be preferred.
    """

    def __init__(self):
        self._antispoof = None
        self._initialized = False
        self._original_frame: np.ndarray | None = None

    @property
    def name(self) -> str:
        return "minifasnet"

    def set_frame(self, frame: np.ndarray):
        """Set the current original frame for context-aware analysis."""
        self._original_frame = frame

    def _ensure_init(self):
        if self._initialized:
            return
        try:
            from uniface.spoofing import MiniFASNet
            self._antispoof = MiniFASNet()
            self._initialized = True
            logger.info("MiniFASNet ONNX model loaded")
        except ImportError:
            logger.warning("uniface not installed — MiniFASNet disabled")
        except Exception as e:
            logger.warning(f"MiniFASNet init failed: {e}")

    def analyze(self, face_crop: np.ndarray, face_roi: FaceROI) -> AnalyzerResult:
        """Analyze face for spoofing using original frame + bbox context.

        If set_frame() was called with the original frame, uses that
        for proper context-aware analysis. Otherwise falls back to
        crop-only mode (less accurate).
        """
        self._ensure_init()

        if not self._initialized or self._antispoof is None:
            return AnalyzerResult(name=self.name, score=50.0, details={"error": "not_initialized"})

        start = time.perf_counter()
        try:
            # Prefer original frame + bbox (proper MiniFASNet usage)
            if self._original_frame is not None:
                image = self._original_frame
                bbox = [face_roi.bbox.x1, face_roi.bbox.y1,
                        face_roi.bbox.x2, face_roi.bbox.y2]
            else:
                # Fallback: pad the crop to simulate context
                # Add 30% padding around the crop with border replication
                h, w = face_crop.shape[:2]
                pad = max(int(h * 0.3), int(w * 0.3), 20)
                image = cv2.copyMakeBorder(
                    face_crop, pad, pad, pad, pad,
                    cv2.BORDER_REPLICATE
                )
                bbox = [pad, pad, pad + w, pad + h]

            result = self._antispoof.predict(image, bbox)
            elapsed_ms = (time.perf_counter() - start) * 1000

            is_real = result.is_real
            confidence = result.confidence

            # Convert to 0-100 score (higher = more live-like)
            if is_real:
                score = 50.0 + confidence * 50.0
            else:
                score = 50.0 - confidence * 50.0

            return AnalyzerResult(
                name=self.name,
                score=max(0.0, min(100.0, score)),
                details={
                    "is_real": is_real,
                    "confidence": confidence,
                    "context": "frame" if self._original_frame is not None else "padded_crop",
                },
                elapsed_ms=elapsed_ms,
            )
        except Exception as e:
            elapsed_ms = (time.perf_counter() - start) * 1000
            logger.debug(f"MiniFASNet analysis failed: {e}")
            return AnalyzerResult(
                name=self.name,
                score=50.0,
                details={"error": str(e)},
                elapsed_ms=elapsed_ms,
            )
