"""Temporal consistency analyzer for static image detection.

Inspired by practice-and-test/GestureAnalysis/anti_spoof.py's
MicroTremorDetector — adapted for face landmark tracking.
"""

import time
from collections import deque
from dataclasses import dataclass, field

import numpy as np

from src.domain.models import FaceROI, AnalyzerResult


@dataclass
class FaceHistory:
    """Per-face temporal buffer for motion analysis."""
    centers_x: deque = field(default_factory=lambda: deque(maxlen=30))
    centers_y: deque = field(default_factory=lambda: deque(maxlen=30))
    areas: deque = field(default_factory=lambda: deque(maxlen=30))
    frame_count: int = 0


class TemporalAnalyzer:
    """Detects unnaturally static faces (photo/frozen video).

    Real faces exhibit natural micro-movements even when the person
    tries to stay still: breathing, micro-saccades, involuntary motion.
    A static image or frozen video shows zero variance in bounding box
    center and area over time.

    Requires ~0.5s (15 frames at 30fps) of accumulated data per face_id.
    """

    def __init__(
        self,
        buffer_size: int = 30,
        min_motion_std: float = 0.0003,
        warmup_frames: int = 15,
    ):
        self._buffer_size = buffer_size
        self._min_std = min_motion_std
        self._warmup = warmup_frames
        self._histories: dict[int, FaceHistory] = {}

    @property
    def name(self) -> str:
        return "temporal"

    def analyze(self, face_crop: np.ndarray, face_roi: FaceROI) -> AnalyzerResult:
        start = time.perf_counter()

        fid = face_roi.face_id
        if fid not in self._histories:
            self._histories[fid] = FaceHistory()
        hist = self._histories[fid]

        # Normalize center position by frame dimensions
        bbox = face_roi.bbox
        cx = (bbox.x1 + bbox.x2) / 2.0
        cy = (bbox.y1 + bbox.y2) / 2.0
        area = bbox.area

        hist.centers_x.append(cx)
        hist.centers_y.append(cy)
        hist.areas.append(area)
        hist.frame_count += 1

        elapsed_ms = (time.perf_counter() - start) * 1000

        # Not enough data yet
        if hist.frame_count < self._warmup:
            return AnalyzerResult(
                name=self.name, score=50.0,
                details={"warmup": True, "frames": hist.frame_count},
                elapsed_ms=elapsed_ms,
            )

        # Calculate motion std
        xs = np.array(hist.centers_x)
        ys = np.array(hist.centers_y)
        areas = np.array(hist.areas)

        # Normalize by mean area to make scale-independent
        mean_area = float(np.mean(areas)) if len(areas) > 0 else 1.0
        norm_factor = max(np.sqrt(mean_area), 1.0)

        pos_std = float(np.sqrt(np.var(xs) + np.var(ys))) / norm_factor
        area_std = float(np.std(areas)) / max(mean_area, 1.0)

        # Combined motion metric
        motion = pos_std + area_std * 0.5

        # Score: low motion = suspicious (spoof-like)
        if motion < self._min_std:
            score = 10.0  # Very suspicious — unnaturally still
        elif motion < self._min_std * 3:
            # Linear interpolation
            ratio = (motion - self._min_std) / (self._min_std * 2)
            score = 10.0 + ratio * 40.0
        elif motion < self._min_std * 10:
            ratio = (motion - self._min_std * 3) / (self._min_std * 7)
            score = 50.0 + ratio * 40.0
        else:
            score = 90.0  # Natural motion

        return AnalyzerResult(
            name=self.name,
            score=max(0.0, min(100.0, score)),
            details={
                "pos_std": pos_std,
                "area_std": area_std,
                "motion": motion,
                "frames": hist.frame_count,
            },
            elapsed_ms=elapsed_ms,
        )

    def reset(self, face_id: int | None = None):
        if face_id is not None:
            self._histories.pop(face_id, None)
        else:
            self._histories.clear()
