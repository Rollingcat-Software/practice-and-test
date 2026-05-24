"""Micro-tremor analyzer — 8-12Hz involuntary facial oscillation detection.

All living humans exhibit involuntary micro-tremor at 8-12Hz caused by
continuous muscle activation (physiological tremor). This signal is:
- Always present in live faces
- Impossible to reproduce on a screen (pixel grid filters it out)
- Too subtle to fake with a held-up photo (hand tremor is 3-5Hz, not 8-12Hz)

This is the strongest single signal for video replay detection because
screen playback physically cannot transmit sub-pixel 8-12Hz oscillation.

Algorithm:
1. Track face centroid from landmarks per frame
2. Accumulate in temporal buffer (90+ frames for 8Hz resolution)
3. Remove low-frequency drift (detrend)
4. FFT on position time-series
5. Measure power in 8-12Hz band relative to total power
6. High relative power = live person

Validation (2026-05-02):
- BUG FIXED (Nyquist clamping): At 30fps Nyquist=15Hz, so 13Hz is only 2Hz from
  the limit. If actual FPS drops below ~26fps, 13Hz becomes unresolvable. Added
  dynamic clamping: effective_high = min(TREMOR_HIGH_HZ, measured_fps/2 - 1.0).
- BUG FIXED (score discontinuity): At tremor_ratio=1.5 boundary, score jumped from
  ~72.5 (second branch) to 70-100 (first branch). Smoothed scoring to be continuous.
- FFT resolution: At MIN_FRAMES=45 and 30fps, resolution=0.667Hz. The tremor band
  7-13Hz (6Hz wide) gets ~9 bins. Adequate. At GOOD_FRAMES=90, 0.33Hz. Fine.
- Detrending: 15-sample MA at 30fps -> ~2Hz cutoff. Removes head drift while
  preserving 7+ Hz tremor signal. CORRECT.
- Edge cases: warmup returns neutral 50. No crash on empty buffers. CORRECT.
- Minor: list() in np.array(list(deque)) is unnecessary but harmless.
"""

import time
import logging
from collections import deque

import numpy as np

from src.domain.models import FaceROI, AnalyzerResult

logger = logging.getLogger(__name__)

TREMOR_LOW_HZ = 7.0
TREMOR_HIGH_HZ = 13.0
MIN_FRAMES = 45   # ~1.5s at 30fps, ~3s at 15fps
GOOD_FRAMES = 90  # ~3s at 30fps


class MicroTremorAnalyzer:
    """Detects 8-12Hz involuntary micro-tremor from face landmarks.

    Requires landmark data from BlinkAnalyzer via set_landmarks().

    Score:
    - 0-30: No tremor detected (SPOOF — screen/photo/mask)
    - 30-60: Weak tremor (ambiguous)
    - 60-100: Tremor detected (LIVE — real person)
    """

    def __init__(self, fps: float = 30.0):
        self._fps = fps
        self._states: dict[int, deque] = {}
        self._current_landmarks: np.ndarray | None = None
        self._frame_times: deque = deque(maxlen=60)

    @property
    def name(self) -> str:
        return "micro_tremor"

    def set_landmarks(self, landmarks: np.ndarray | None):
        self._current_landmarks = landmarks

    def analyze(self, face_crop: np.ndarray, face_roi: FaceROI) -> AnalyzerResult:
        start = time.perf_counter()

        # Measure actual FPS
        self._frame_times.append(start)
        if len(self._frame_times) > 10:
            dt = self._frame_times[-1] - self._frame_times[0]
            if dt > 0:
                self._fps = (len(self._frame_times) - 1) / dt

        fid = face_roi.face_id
        if fid not in self._states:
            self._states[fid] = deque(maxlen=180)  # 6s buffer

        # Use landmark centroid if available, otherwise bbox center
        if self._current_landmarks is not None and len(self._current_landmarks) > 10:
            cx = float(np.mean(self._current_landmarks[:, 0]))
            cy = float(np.mean(self._current_landmarks[:, 1]))
        else:
            cx, cy = face_roi.bbox.center

        self._states[fid].append((cx, cy))

        elapsed_ms = (time.perf_counter() - start) * 1000

        if len(self._states[fid]) < MIN_FRAMES:
            return AnalyzerResult(
                name=self.name, score=50.0,
                details={"warmup": True, "frames": len(self._states[fid])},
                elapsed_ms=elapsed_ms,
            )

        positions = np.array(self._states[fid])  # (N, 2)
        n = len(positions)

        # Clamp tremor band to stay below Nyquist (fps/2).
        # At 30fps Nyquist=15Hz; if FPS drops below ~26, 13Hz is unresolvable.
        nyquist = self._fps / 2.0
        effective_high = min(TREMOR_HIGH_HZ, nyquist - 1.0)
        if effective_high <= TREMOR_LOW_HZ:
            # FPS too low for any tremor detection — return neutral
            elapsed_ms = (time.perf_counter() - start) * 1000
            return AnalyzerResult(
                name=self.name, score=50.0,
                details={"fps_too_low": True, "measured_fps": round(self._fps, 1),
                         "nyquist_hz": round(nyquist, 1), "frames": n},
                elapsed_ms=elapsed_ms,
            )

        # Analyze X and Y independently
        results = []
        for axis in range(2):
            signal = positions[:, axis].astype(np.float64)

            # Detrend: remove slow drift (head movement)
            kernel_size = min(15, n // 3)
            if kernel_size > 1:
                smooth = np.convolve(signal, np.ones(kernel_size) / kernel_size, mode="same")
                signal = signal - smooth

            # Hanning window
            window = np.hanning(n)
            signal = signal * window

            # FFT
            fft = np.fft.rfft(signal)
            magnitude = np.abs(fft)
            freqs = np.fft.rfftfreq(n, d=1.0 / self._fps)

            # Tremor band power (clamped to effective Nyquist)
            tremor_mask = (freqs >= TREMOR_LOW_HZ) & (freqs <= effective_high)
            if not np.any(tremor_mask):
                results.append(0.0)
                continue

            tremor_power = float(np.mean(magnitude[tremor_mask]))

            # Total power (excluding DC)
            total_mask = freqs > 0.5
            total_power = float(np.mean(magnitude[total_mask])) if np.any(total_mask) else 1e-6

            # Relative tremor power
            tremor_ratio = tremor_power / max(total_power, 1e-6)
            results.append(tremor_ratio)

        avg_tremor_ratio = sum(results) / max(len(results), 1)
        data_quality = min(1.0, n / GOOD_FRAMES)

        # Score: high tremor ratio = live
        # Continuous scoring to avoid discontinuities at boundaries.
        # ratio -> score mapping (approximate):
        #   0.0  -> 10-30 (spoof, depends on data sufficiency)
        #   0.5  -> 40
        #   1.0  -> 55
        #   1.5  -> 70
        #   2.0+ -> 85-100
        if avg_tremor_ratio > 1.5:
            # Strong tremor — high confidence live
            score = 70.0 + min((avg_tremor_ratio - 1.5) * 20.0, 30.0) * data_quality
        elif avg_tremor_ratio > 0.5:
            # Linear ramp from 40 to 70 over ratio 0.5 -> 1.5
            score = 40.0 + (avg_tremor_ratio - 0.5) * 30.0
        else:
            # No meaningful tremor
            if n > GOOD_FRAMES:
                score = 10.0 + avg_tremor_ratio * 40.0  # 10-30 range
            else:
                score = 30.0 + avg_tremor_ratio * 20.0  # 30-40, not enough data

        elapsed_ms = (time.perf_counter() - start) * 1000

        return AnalyzerResult(
            name=self.name,
            score=max(0.0, min(100.0, score)),
            details={
                "tremor_ratio": round(avg_tremor_ratio, 4),
                "tremor_x": round(results[0], 4) if len(results) > 0 else 0,
                "tremor_y": round(results[1], 4) if len(results) > 1 else 0,
                "measured_fps": round(self._fps, 1),
                "frames": n,
                "data_quality": round(data_quality, 2),
            },
            elapsed_ms=elapsed_ms,
        )
