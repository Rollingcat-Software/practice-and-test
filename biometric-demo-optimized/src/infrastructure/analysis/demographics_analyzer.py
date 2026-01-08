"""Demographics analysis using DeepFace."""

import logging
import time
from typing import Dict, Optional

import cv2
import numpy as np

from ...domain.models import Demographics, FaceRegion

logger = logging.getLogger(__name__)


class DemographicsAnalyzer:
    """DeepFace-based demographics analyzer with caching.

    Analyzes age, gender, and emotion with:
    - CLAHE preprocessing for lighting normalization
    - Spatial caching to avoid redundant analysis
    - Persistent last-known result to prevent UI flickering
    """

    def __init__(self, cache_interval: float = 2.0, throttle_interval: float = 1.5):
        self._deepface = None
        self._cache: Dict[str, Dict] = {}
        self._cache_interval = cache_interval
        self._throttle_interval = throttle_interval
        self._last_analysis_time = 0
        self._last_result = Demographics()
        # Pre-create CLAHE object (expensive to create repeatedly)
        self._clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    def _load(self):
        """Lazy load DeepFace."""
        if self._deepface is None:
            from deepface import DeepFace
            self._deepface = DeepFace
            logger.info("DeepFace loaded for demographics")
        return self._deepface

    def analyze(self, frame: np.ndarray, region: FaceRegion) -> Demographics:
        """Analyze demographics for a face region.

        Uses caching and throttling to maintain performance.

        Args:
            frame: Full BGR frame
            region: Face bounding box

        Returns:
            Demographics object
        """
        now = time.time()

        # Spatial cache key (200px buckets for better cache hit rate)
        cx, cy = region.center
        cache_key = f"{cx // 200}_{cy // 200}"

        # Check cache
        if cache_key in self._cache and now - self._cache[cache_key]['t'] < self._cache_interval:
            self._last_result = self._cache[cache_key]['d']
            return self._cache[cache_key]['d']

        # Throttle to prevent overload
        if now - self._last_analysis_time < self._throttle_interval:
            return self._last_result

        self._last_analysis_time = now

        try:
            # Extract face with padding
            frame_h, frame_w = frame.shape[:2]
            pad_x = int(region.w * 0.4)
            pad_y = int(region.h * 0.5)

            x1 = max(0, region.x - pad_x)
            y1 = max(0, region.y - pad_y)
            x2 = min(frame_w, region.x + region.w + pad_x)
            y2 = min(frame_h, region.y + region.h + pad_y)

            face = frame[y1:y2, x1:x2]

            if face.size == 0 or face.shape[0] < 48 or face.shape[1] < 48:
                return self._last_result

            # Resize to standard size (INTER_LINEAR is 3x faster than LANCZOS4)
            face_resized = cv2.resize(face, (224, 224), interpolation=cv2.INTER_LINEAR)

            # Apply cached CLAHE for lighting normalization
            lab = cv2.cvtColor(face_resized, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            l = self._clahe.apply(l)  # Use cached CLAHE object
            lab = cv2.merge([l, a, b])
            face_enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

            # Analyze with DeepFace
            df = self._load()
            results = df.analyze(
                face_enhanced,
                actions=['age', 'gender', 'emotion'],
                enforce_detection=False,
                detector_backend='skip',
                silent=True
            )

            if results:
                r = results[0] if isinstance(results, list) else results

                # Gender with confidence threshold
                gender_info = r.get('gender', {})
                if isinstance(gender_info, dict):
                    man_conf = gender_info.get('Man', 0)
                    woman_conf = gender_info.get('Woman', 0)
                    if man_conf > 60:
                        gender = 'M'
                    elif woman_conf > 60:
                        gender = 'F'
                    else:
                        gender = '?'
                else:
                    gender = 'M' if r.get('dominant_gender') == 'Man' else 'F'

                demo = Demographics(
                    age=int(r.get('age', 0)),
                    gender=gender,
                    emotion=r.get('dominant_emotion', '?')[:6],
                    timestamp=now
                )

                self._cache[cache_key] = {'d': demo, 't': now}
                self._last_result = demo
                return demo

        except Exception as e:
            logger.debug(f"Demographics analysis error: {e}")

        return self._last_result
