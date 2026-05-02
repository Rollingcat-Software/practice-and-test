"""Background grid analyzer for proctoring environment monitoring.

In proctoring, the camera is fixed. The background should be static.
Only the person's face and upper body should move.

This analyzer divides the frame into a 6x4 grid (24 cells), excludes
cells overlapping the face, and monitors background stability.

Signals:
1. Background stability — cells that change suddenly = suspicious
2. Specular anomalies — bright low-saturation spots = screen glare
3. Color temperature — screen blue-shift vs natural warm lighting
"""

import time
import logging
from collections import deque

import cv2
import numpy as np

from src.domain.models import FaceROI, AnalyzerResult

logger = logging.getLogger(__name__)

GRID_COLS = 6
GRID_ROWS = 4
MIN_FRAMES = 15  # ~0.5s baseline
STABILITY_THRESHOLD = 12.0  # Luma change > this = unstable cell


class BackgroundGridAnalyzer:
    """Monitors background stability via NxM grid analysis.

    Needs full frame via set_frame(). Analyzes background cells
    (excluding face region) for unexpected changes.

    Score:
    - 0-30: Many background cells changing (suspicious)
    - 30-60: Some instability (possible movement)
    - 60-100: Background stable (normal proctoring environment)
    """

    def __init__(self):
        self._original_frame: np.ndarray | None = None
        self._cell_history: dict[tuple[int, int], deque] = {}
        self._frame_count = 0

    @property
    def name(self) -> str:
        return "background_grid"

    def set_frame(self, frame: np.ndarray):
        self._original_frame = frame

    def analyze(self, face_crop: np.ndarray, face_roi: FaceROI) -> AnalyzerResult:
        start = time.perf_counter()
        self._frame_count += 1

        if self._original_frame is None:
            return AnalyzerResult(name=self.name, score=50.0,
                                  details={"error": "no_frame"},
                                  elapsed_ms=(time.perf_counter() - start) * 1000)

        frame = self._original_frame
        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        cell_h = h // GRID_ROWS
        cell_w = w // GRID_COLS

        # Face bbox for exclusion
        fb = face_roi.bbox
        stable_cells = 0
        total_bg_cells = 0
        specular_cells = 0
        cool_cells = 0

        for row in range(GRID_ROWS):
            for col in range(GRID_COLS):
                y1 = row * cell_h
                x1 = col * cell_w
                y2 = min(h, y1 + cell_h)
                x2 = min(w, x1 + cell_w)

                # Skip cells that overlap with face
                cell_cx = (x1 + x2) // 2
                cell_cy = (y1 + y2) // 2
                if fb.x1 <= cell_cx <= fb.x2 and fb.y1 <= cell_cy <= fb.y2:
                    continue

                total_bg_cells += 1
                cell_gray = gray[y1:y2, x1:x2]
                cell_hsv = hsv[y1:y2, x1:x2]
                cell_mean = float(np.mean(cell_gray))

                key = (row, col)
                if key not in self._cell_history:
                    self._cell_history[key] = deque(maxlen=60)
                self._cell_history[key].append(cell_mean)

                # Stability check
                if len(self._cell_history[key]) >= MIN_FRAMES:
                    history = list(self._cell_history[key])
                    recent_std = float(np.std(history[-15:]))
                    if recent_std < STABILITY_THRESHOLD:
                        stable_cells += 1

                # Specular check (bright + low saturation = screen glare)
                sat = cell_hsv[:, :, 1].astype(np.float32)
                val = cell_hsv[:, :, 2].astype(np.float32)
                specular_ratio = float(np.mean((val > 230) & (sat < 40)))
                if specular_ratio > 0.05:
                    specular_cells += 1

                # Color temperature (blue-shifted = screen)
                hue = cell_hsv[:, :, 0].astype(np.float32)
                cool_ratio = float(np.mean((hue >= 100) & (hue <= 130)))
                if cool_ratio > 0.15:
                    cool_cells += 1

        elapsed_ms = (time.perf_counter() - start) * 1000

        if total_bg_cells == 0 or self._frame_count < MIN_FRAMES:
            return AnalyzerResult(name=self.name, score=50.0,
                                  details={"warmup": True},
                                  elapsed_ms=elapsed_ms)

        stability_ratio = stable_cells / total_bg_cells
        specular_ratio = specular_cells / total_bg_cells
        cool_ratio = cool_cells / total_bg_cells

        # Score: stable background + no glare + warm lighting = real room
        stability_score = stability_ratio * 60.0
        specular_penalty = specular_ratio * 20.0
        cool_penalty = cool_ratio * 10.0

        score = stability_score + 30.0 - specular_penalty - cool_penalty

        return AnalyzerResult(
            name=self.name,
            score=max(0.0, min(100.0, score)),
            details={
                "stability_ratio": round(stability_ratio, 3),
                "stable_cells": stable_cells,
                "total_bg_cells": total_bg_cells,
                "specular_cells": specular_cells,
                "cool_cells": cool_cells,
                "specular_ratio": round(specular_ratio, 3),
                "cool_ratio": round(cool_ratio, 3),
            },
            elapsed_ms=elapsed_ms,
        )
