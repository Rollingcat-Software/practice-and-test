"""Blink detection analyzer using Eye Aspect Ratio (EAR).

Uses MediaPipe FaceLandmarker (478 points) to compute EAR per eye.
Real faces blink ~15-20 times/min. Photos and screens never blink.

This is the strongest temporal signal for static image/screen attacks:
- After 3 seconds: a real person should have blinked at least once
- After 10 seconds: no blinks = almost certainly a spoof

EAR formula (Soukupova & Cech, 2016):
  EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)
  Open eye: EAR ~ 0.25-0.35
  Closed eye: EAR < 0.21
"""

import time
import logging
from collections import deque
from dataclasses import dataclass, field

import numpy as np

from src.domain.models import FaceROI, AnalyzerResult

logger = logging.getLogger(__name__)

# MediaPipe FaceMesh landmark indices for eyes
# Right eye (from subject's perspective)
RIGHT_EYE = [33, 160, 158, 133, 153, 144]
# Left eye
LEFT_EYE = [362, 385, 387, 263, 373, 380]


def compute_ear(landmarks: np.ndarray, eye_indices: list[int]) -> float:
    """Compute Eye Aspect Ratio from 6 landmark points.

    Args:
        landmarks: Nx3 or Nx2 array of face landmarks
        eye_indices: 6 indices [p1, p2, p3, p4, p5, p6]

    Returns:
        EAR value (0-0.5 typical range)
    """
    pts = landmarks[eye_indices][:, :2].astype(np.float64)
    # Vertical distances
    v1 = np.linalg.norm(pts[1] - pts[5])
    v2 = np.linalg.norm(pts[2] - pts[4])
    # Horizontal distance
    h = np.linalg.norm(pts[0] - pts[3])
    if h < 1e-6:
        return 0.3  # Avoid division by zero, return neutral
    return float((v1 + v2) / (2.0 * h))


@dataclass
class BlinkState:
    """Per-face blink tracking state."""
    ear_history: deque = field(default_factory=lambda: deque(maxlen=90))  # 3s at 30fps
    blink_count: int = 0
    eyes_closed_frames: int = 0
    last_blink_frame: int = 0
    frame_count: int = 0


class BlinkAnalyzer:
    """Blink detection via Eye Aspect Ratio from MediaPipe landmarks.

    Requires MediaPipe FaceLandmarker (478 points), not just FaceDetector (6 keypoints).
    The analyzer lazily initializes FaceLandmarker on first use.

    Score interpretation:
    - 0-30: No blinks detected after sufficient warmup (SPOOF-like)
    - 30-60: Few blinks, insufficient data or borderline
    - 60-100: Normal blink rate detected (LIVE-like)
    """

    EAR_THRESHOLD = 0.20       # Below this = eye closed
    CONSECUTIVE_FRAMES = 3     # Frames eye must be closed for a blink
    REOPEN_THRESHOLD = 0.23    # EAR must recover above this after closing
    REOPEN_FRAMES = 8          # Must reopen within this many frames after closing
    MIN_OPEN_BETWEEN = 8       # Minimum frames between blinks (prevents noise counting)
    WARMUP_FRAMES = 45         # 1.5s before scoring (need baseline EAR)
    NORMAL_BLINK_RATE = 17.0   # Expected blinks/min for real person

    def __init__(self):
        self._landmarker = None
        self._initialized = False
        self._init_failed = False
        self._states: dict[int, BlinkState] = {}
        self._original_frame: np.ndarray | None = None

    @property
    def name(self) -> str:
        return "blink"

    def set_frame(self, frame: np.ndarray):
        self._original_frame = frame

    def _ensure_init(self):
        if self._initialized or self._init_failed:
            return
        try:
            import mediapipe as mp
            import urllib.request
            from pathlib import Path

            model_path = Path(__file__).parent.parent.parent.parent / "models" / "face_landmarker.task"
            if not model_path.exists():
                url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task"
                model_path.parent.mkdir(parents=True, exist_ok=True)
                logger.info(f"Downloading FaceLandmarker model...")
                urllib.request.urlretrieve(url, str(model_path))

            options = mp.tasks.vision.FaceLandmarkerOptions(
                base_options=mp.tasks.BaseOptions(model_asset_path=str(model_path)),
                running_mode=mp.tasks.vision.RunningMode.IMAGE,
                num_faces=5,
                min_face_detection_confidence=0.4,
                min_tracking_confidence=0.4,
            )
            self._landmarker = mp.tasks.vision.FaceLandmarker.create_from_options(options)
            self._initialized = True
            logger.info("FaceLandmarker loaded for blink detection")
        except Exception as e:
            self._init_failed = True
            logger.warning(f"FaceLandmarker init failed: {e}")

    def analyze(self, face_crop: np.ndarray, face_roi: FaceROI) -> AnalyzerResult:
        start = time.perf_counter()
        self._ensure_init()

        fid = face_roi.face_id
        if fid not in self._states:
            self._states[fid] = BlinkState()
        state = self._states[fid]
        state.frame_count += 1

        if not self._initialized or self._landmarker is None or self._original_frame is None:
            return AnalyzerResult(name=self.name, score=50.0,
                                  details={"error": "not_initialized"},
                                  elapsed_ms=(time.perf_counter() - start) * 1000)

        # Run FaceLandmarker on full frame for accurate landmarks
        import mediapipe as mp
        import cv2
        rgb = cv2.cvtColor(self._original_frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        try:
            result = self._landmarker.detect(mp_image)
        except Exception as e:
            return AnalyzerResult(name=self.name, score=50.0,
                                  details={"error": str(e)},
                                  elapsed_ms=(time.perf_counter() - start) * 1000)

        if not result.face_landmarks:
            return AnalyzerResult(name=self.name, score=50.0,
                                  details={"no_landmarks": True},
                                  elapsed_ms=(time.perf_counter() - start) * 1000)

        # Find the landmark set closest to our tracked face
        h, w = self._original_frame.shape[:2]
        best_landmarks = None
        best_dist = float("inf")
        face_cx, face_cy = face_roi.bbox.center

        for face_lm in result.face_landmarks:
            lm_array = np.array([[l.x * w, l.y * h, l.z] for l in face_lm])
            cx = float(np.mean(lm_array[:, 0]))
            cy = float(np.mean(lm_array[:, 1]))
            dist = abs(cx - face_cx) + abs(cy - face_cy)
            if dist < best_dist:
                best_dist = dist
                best_landmarks = lm_array

        if best_landmarks is None or len(best_landmarks) < 468:
            return AnalyzerResult(name=self.name, score=50.0,
                                  details={"insufficient_landmarks": True},
                                  elapsed_ms=(time.perf_counter() - start) * 1000)

        # Compute EAR for both eyes
        left_ear = compute_ear(best_landmarks, LEFT_EYE)
        right_ear = compute_ear(best_landmarks, RIGHT_EYE)
        avg_ear = (left_ear + right_ear) / 2.0

        state.ear_history.append(avg_ear)

        # Blink detection with V-shape validation:
        # 1. EAR must drop below threshold for CONSECUTIVE_FRAMES
        # 2. Then EAR must recover above REOPEN_THRESHOLD within REOPEN_FRAMES
        # 3. Minimum MIN_OPEN_BETWEEN frames between blinks (anti-noise)
        if avg_ear < self.EAR_THRESHOLD:
            state.eyes_closed_frames += 1
        else:
            if (state.eyes_closed_frames >= self.CONSECUTIVE_FRAMES
                    and avg_ear >= self.REOPEN_THRESHOLD
                    and (state.frame_count - state.last_blink_frame) >= self.MIN_OPEN_BETWEEN):
                # Valid blink: closed long enough + reopened properly + not too soon after last
                state.blink_count += 1
                state.last_blink_frame = state.frame_count
            state.eyes_closed_frames = 0

        # Score calculation
        elapsed_ms = (time.perf_counter() - start) * 1000

        if state.frame_count < self.WARMUP_FRAMES:
            return AnalyzerResult(
                name=self.name, score=50.0,
                details={"warmup": True, "frames": state.frame_count,
                          "ear": round(avg_ear, 3), "blinks": state.blink_count},
                elapsed_ms=elapsed_ms,
            )

        # After warmup: score based on blink presence
        duration_sec = state.frame_count / 30.0  # Assume 30fps
        expected_blinks = self.NORMAL_BLINK_RATE * (duration_sec / 60.0)

        if state.blink_count == 0:
            # No blinks detected — suspicious after warmup
            if duration_sec > 5.0:
                score = 10.0  # Very suspicious
            elif duration_sec > 3.0:
                score = 25.0
            else:
                score = 40.0
        elif state.blink_count >= expected_blinks * 0.3:
            score = 90.0  # Normal blink rate
        else:
            # Some blinks but fewer than expected
            ratio = state.blink_count / max(expected_blinks, 0.1)
            score = 40.0 + ratio * 50.0

        return AnalyzerResult(
            name=self.name,
            score=max(0.0, min(100.0, score)),
            details={
                "ear": round(avg_ear, 3),
                "blinks": state.blink_count,
                "blink_rate_per_min": round(state.blink_count / max(duration_sec / 60, 0.01), 1),
                "duration_sec": round(duration_sec, 1),
                "eyes_open": avg_ear >= self.EAR_THRESHOLD,
            },
            elapsed_ms=elapsed_ms,
        )

    def get_blink_count(self, face_id: int) -> int:
        state = self._states.get(face_id)
        return state.blink_count if state else 0
