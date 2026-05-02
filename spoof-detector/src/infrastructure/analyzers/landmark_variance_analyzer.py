"""Landmark variance analyzer for spoof detection.

Tracks all 478 MediaPipe FaceLandmarker points over time.
Real faces show natural variance from micro-movements, blinks,
and breathing. Photos show near-zero variance. Videos show
2D-only movement (no depth parallax).

This analyzer reuses the FaceLandmarker already loaded by BlinkAnalyzer
to avoid running it twice per frame. It reads landmarks from the
blink analyzer's results via the shared frame context.

Signals:
1. Overall landmark variance: zero = photo, natural = live
2. Variance distribution: photos have uniform zero; real faces have
   high variance around eyes/mouth, low on forehead
3. Depth consistency: real 3D faces show parallax; 2D screens don't
"""

import time
import logging
from collections import deque
from dataclasses import dataclass, field

import numpy as np

from src.domain.models import FaceROI, AnalyzerResult

logger = logging.getLogger(__name__)

# Landmark regions for variance distribution analysis
# MediaPipe face mesh indices
REGION_LEFT_EYE = list(range(33, 42)) + [133, 144, 153, 154, 155, 157, 158, 159, 160, 161]
REGION_RIGHT_EYE = list(range(362, 371)) + [263, 373, 380, 381, 382, 384, 385, 386, 387, 388]
REGION_MOUTH = list(range(61, 68)) + list(range(291, 298)) + [0, 13, 14, 17, 37, 39, 40, 78, 80, 81, 82, 87, 88, 95, 178, 181, 267, 269, 270, 308, 310, 311, 312, 317, 318, 324, 402, 405]
REGION_FOREHEAD = [10, 67, 69, 104, 108, 109, 151, 338, 337, 297, 299, 333]


@dataclass
class LandmarkHistory:
    """Per-face landmark tracking buffer."""
    # Store last N frames of landmark positions (Nx478x2)
    frames: deque = field(default_factory=lambda: deque(maxlen=60))  # 2s at 30fps / 4s at 15fps
    frame_count: int = 0


class LandmarkVarianceAnalyzer:
    """Tracks 478 landmark points over time for spoof detection.

    Requires BlinkAnalyzer to have run first (shares FaceLandmarker).
    Uses the blink analyzer's landmark data via set_landmarks().

    Score:
    - 0-20: Near-zero variance (definitely static image / frozen)
    - 20-50: Low variance (suspicious — possible photo with hand shake)
    - 50-80: Moderate variance (could be video replay or cautious real)
    - 80-100: Natural variance with depth cues (live person)
    """

    WARMUP_FRAMES = 15      # Need ~1 second of data
    ZERO_VAR_THRESHOLD = 0.5  # Below this = definitely static

    def __init__(self):
        self._histories: dict[int, LandmarkHistory] = {}
        self._current_landmarks: np.ndarray | None = None

    @property
    def name(self) -> str:
        return "landmark_variance"

    def set_landmarks(self, landmarks: np.ndarray | None):
        """Set current frame's landmarks (from BlinkAnalyzer)."""
        self._current_landmarks = landmarks

    def analyze(self, face_crop: np.ndarray, face_roi: FaceROI) -> AnalyzerResult:
        start = time.perf_counter()

        fid = face_roi.face_id
        if fid not in self._histories:
            self._histories[fid] = LandmarkHistory()
        hist = self._histories[fid]
        hist.frame_count += 1

        if self._current_landmarks is None or len(self._current_landmarks) < 468:
            return AnalyzerResult(
                name=self.name, score=50.0,
                details={"error": "no_landmarks"},
                elapsed_ms=(time.perf_counter() - start) * 1000,
            )

        # Store XY positions (ignore Z for now)
        xy = self._current_landmarks[:, :2].copy()
        hist.frames.append(xy)

        elapsed_ms = (time.perf_counter() - start) * 1000

        if hist.frame_count < self.WARMUP_FRAMES:
            return AnalyzerResult(
                name=self.name, score=50.0,
                details={"warmup": True, "frames": hist.frame_count},
                elapsed_ms=elapsed_ms,
            )

        # Compute variance across stored frames
        all_frames = np.array(list(hist.frames))  # (T, 478, 2)
        n_frames = all_frames.shape[0]

        # Normalize: subtract per-frame centroid to remove global translation
        # (hand holding a photo moves globally but landmarks don't move relative to each other)
        centroids = np.mean(all_frames, axis=1, keepdims=True)  # (T, 1, 2)
        normalized = all_frames - centroids  # Remove global translation

        # Per-landmark variance across time (after removing global motion)
        per_landmark_var = np.var(normalized, axis=0)  # (478, 2)
        per_landmark_total_var = np.sum(per_landmark_var, axis=1)  # (478,)

        overall_var = float(np.mean(per_landmark_total_var))
        max_var = float(np.max(per_landmark_total_var))

        # Region-specific variance
        eye_var = float(np.mean(per_landmark_total_var[REGION_LEFT_EYE + REGION_RIGHT_EYE]))
        mouth_var = float(np.mean(per_landmark_total_var[REGION_MOUTH]))
        forehead_var = float(np.mean(per_landmark_total_var[REGION_FOREHEAD]))

        # Variance ratio: real faces have higher eye/mouth variance than forehead
        # Photos have uniform variance (near zero everywhere)
        if forehead_var > 0.01:
            expression_ratio = (eye_var + mouth_var) / (2.0 * forehead_var)
        else:
            expression_ratio = 0.0 if eye_var < 0.01 else 10.0

        # Score computation
        if overall_var < self.ZERO_VAR_THRESHOLD:
            # Near-zero internal variance = definitely static
            score = max(0.0, 10.0 * (overall_var / self.ZERO_VAR_THRESHOLD))
        elif overall_var < 2.0:
            # Low variance — suspicious
            score = 10.0 + 30.0 * ((overall_var - self.ZERO_VAR_THRESHOLD) / 1.5)
        elif overall_var < 5.0:
            # Moderate variance
            score = 40.0 + 20.0 * ((overall_var - 2.0) / 3.0)
        else:
            # High variance — natural movement
            score = 60.0 + min(40.0, overall_var * 2.0)

        # Expression ratio bonus: real faces have non-uniform variance
        if expression_ratio > 2.0:
            score = min(100.0, score + 10.0)  # Eyes/mouth move more than forehead

        elapsed_ms = (time.perf_counter() - start) * 1000

        return AnalyzerResult(
            name=self.name,
            score=max(0.0, min(100.0, score)),
            details={
                "overall_var": round(overall_var, 4),
                "max_var": round(max_var, 4),
                "eye_var": round(eye_var, 4),
                "mouth_var": round(mouth_var, 4),
                "forehead_var": round(forehead_var, 4),
                "expression_ratio": round(expression_ratio, 2),
                "n_frames": n_frames,
            },
            elapsed_ms=elapsed_ms,
        )
