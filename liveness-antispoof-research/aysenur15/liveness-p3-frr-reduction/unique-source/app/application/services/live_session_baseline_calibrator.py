"""Lightweight session-relative baseline calibration for live liveness preview."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class BaselineCalibrationFrame:
    """Per-frame inputs used to estimate a neutral live-capture baseline."""

    timestamp: float
    face_detected: bool
    face_quality: Optional[float]
    blur_score: Optional[float]
    brightness: Optional[float]
    ear_current: Optional[float]
    mar_current: Optional[float]
    yaw_current: Optional[float]
    pitch_current: Optional[float]
    roll_current: Optional[float]
    smile_score: Optional[float]


@dataclass(frozen=True)
class SessionBaseline:
    """Neutral baseline values estimated from stable early-session frames."""

    calibrated: bool
    sample_count: int
    duration_seconds: float
    ear_baseline: Optional[float]
    mar_baseline: Optional[float]
    smile_baseline: Optional[float]
    yaw_baseline: Optional[float]
    pitch_baseline: Optional[float]
    roll_baseline: Optional[float]


class LiveSessionBaselineCalibrator:
    """Collect a short stable neutral baseline at the start of a live session."""

    def __init__(self, *, baseline_seconds: float = 0.75) -> None:
        self._baseline_seconds = baseline_seconds
        self._frames: deque[BaselineCalibrationFrame] = deque()
        self._baseline: Optional[SessionBaseline] = None

    def update(self, frame: BaselineCalibrationFrame) -> SessionBaseline:
        if self._baseline is not None:
            return self._baseline

        self._frames.append(frame)
        self._evict_old(reference_timestamp=frame.timestamp)
        stable_frames = [item for item in self._frames if self._is_stable_candidate(item)]
        span = self._span_seconds(stable_frames)

        if len(stable_frames) >= 6 and span >= self._baseline_seconds:
            self._baseline = SessionBaseline(
                calibrated=True,
                sample_count=len(stable_frames),
                duration_seconds=span,
                ear_baseline=_median([item.ear_current for item in stable_frames]),
                mar_baseline=_median([item.mar_current for item in stable_frames]),
                smile_baseline=_median([item.smile_score for item in stable_frames]),
                yaw_baseline=_median([item.yaw_current for item in stable_frames]),
                pitch_baseline=_median([item.pitch_current for item in stable_frames]),
                roll_baseline=_median([item.roll_current for item in stable_frames]),
            )
            return self._baseline

        return SessionBaseline(
            calibrated=False,
            sample_count=len(stable_frames),
            duration_seconds=span,
            ear_baseline=_median([item.ear_current for item in stable_frames]),
            mar_baseline=_median([item.mar_current for item in stable_frames]),
            smile_baseline=_median([item.smile_score for item in stable_frames]),
            yaw_baseline=_median([item.yaw_current for item in stable_frames]),
            pitch_baseline=_median([item.pitch_current for item in stable_frames]),
            roll_baseline=_median([item.roll_current for item in stable_frames]),
        )

    def get_baseline(self) -> Optional[SessionBaseline]:
        return self._baseline

    def reset(self) -> None:
        self._frames.clear()
        self._baseline = None

    def _evict_old(self, *, reference_timestamp: float) -> None:
        min_timestamp = reference_timestamp - max(self._baseline_seconds * 2.0, 1.5)
        while self._frames and self._frames[0].timestamp < min_timestamp:
            self._frames.popleft()

    def _is_stable_candidate(self, frame: BaselineCalibrationFrame) -> bool:
        if not frame.face_detected:
            return False
        if frame.face_quality is not None and frame.face_quality < 0.45:
            return False
        if frame.blur_score is not None and frame.blur_score < 25.0:
            return False
        if frame.brightness is not None and not 45.0 <= frame.brightness <= 215.0:
            return False
        if frame.yaw_current is not None and abs(frame.yaw_current) > 12.0:
            return False
        if frame.pitch_current is not None and abs(frame.pitch_current) > 10.0:
            return False
        if frame.roll_current is not None and abs(frame.roll_current) > 10.0:
            return False
        return True

    @staticmethod
    def _span_seconds(frames: list[BaselineCalibrationFrame]) -> float:
        if len(frames) < 2:
            return 0.0
        return max(0.0, frames[-1].timestamp - frames[0].timestamp)


def _median(values: list[Optional[float]]) -> Optional[float]:
    filtered = [float(value) for value in values if value is not None]
    if not filtered:
        return None
    return float(np.median(filtered))
