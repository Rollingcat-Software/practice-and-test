"""
Motion Analyzer v2 -- Robust Wave & Air Drawing
=================================================
Pure math module -- no OpenCV dependency.

WaveDetector (Oscillation Filter)
----------------------------------
Stores timestamped wrist X-coordinates.  A wave is valid when:
  1. Total displacement across the buffer > 20% of frame width.
  2. At least 2 local extrema (peaks/valleys) with > 10% swing each.
  3. Frequency is 1-4 Hz (not random shaking, not too slow).

ShapeRecognizer (Trajectory + Bounding-Box Validation)
------------------------------------------------------
Records index-tip (x,y) with 3-point moving-average smoothing.
Validation happens ONCE when finalize() is called (on timeout or
when the user closes their finger), not every frame.

  Circle: Aspect ratio 0.7-1.3 AND path points cover > 55% of the
          inscribed-circle area of the bounding box.

  Square: Aspect ratio 0.65-1.35 AND >= 3 sharp corners detected
          (> 50 degree direction changes in subsampled path).
"""

import math
import time
from collections import deque
from dataclasses import dataclass
from enum import Enum, auto


# ── Wave Detection ──────────────────────────────────────────────────

class WaveDetector:
    """Oscillation-based wave detector with frequency validation.

    Parameters
    ----------
    buffer_size : int
        Max samples in the buffer (default 40, ~1.3s at 30fps).
    min_swing : float
        Min X displacement per half-wave (default 0.10 = 10% frame width).
    min_total_displacement : float
        Total X range across buffer must exceed this (default 0.20).
    min_reversals : int
        Required direction changes with sufficient swing (default 2).
    min_freq_hz / max_freq_hz : float
        Valid oscillation frequency range.
    """

    def __init__(
        self,
        buffer_size: int = 40,
        min_swing: float = 0.10,
        min_total_displacement: float = 0.20,
        min_reversals: int = 2,
        min_freq_hz: float = 1.0,
        max_freq_hz: float = 4.0,
    ):
        self.buffer_size = buffer_size
        self.min_swing = min_swing
        self.min_total_disp = min_total_displacement
        self.min_reversals = min_reversals
        self.min_freq = min_freq_hz
        self.max_freq = max_freq_hz
        self._data: deque[tuple[float, float]] = deque(maxlen=buffer_size)  # (time, x)

    def push(self, wrist_x: float) -> None:
        self._data.append((time.monotonic(), wrist_x))

    @property
    def reversal_count(self) -> int:
        """Number of valid reversals detected so far (for UI feedback)."""
        return self._find_extrema()[0]

    def is_waving(self) -> bool:
        n_reversals, extrema_times = self._find_extrema()

        if n_reversals < self.min_reversals:
            return False

        # Check total displacement across the entire buffer.
        xs = [d[1] for d in self._data]
        total_disp = max(xs) - min(xs)
        if total_disp < self.min_total_disp:
            return False

        # Check frequency: reversals per second.
        if len(extrema_times) >= 2:
            duration = extrema_times[-1] - extrema_times[0]
            if duration > 0:
                freq = n_reversals / duration  # reversals/sec ~ half-cycles/sec
                full_freq = freq / 2.0  # convert to full cycles
                if full_freq < self.min_freq or full_freq > self.max_freq:
                    return False

        return True

    def _find_extrema(self) -> tuple[int, list[float]]:
        """Return (reversal_count, list_of_extrema_timestamps)."""
        if len(self._data) < 5:
            return (0, [])

        # Smooth the X values with a 3-sample moving average.
        raw = [d[1] for d in self._data]
        times = [d[0] for d in self._data]
        smoothed = raw[:1]  # keep first as-is
        for i in range(1, len(raw) - 1):
            smoothed.append((raw[i-1] + raw[i] + raw[i+1]) / 3.0)
        smoothed.append(raw[-1])

        # Find direction and track reversals with minimum swing.
        extrema_vals: list[float] = []
        extrema_ts: list[float] = []
        last_extreme_val = smoothed[0]
        last_extreme_t = times[0]
        going_positive: bool | None = None

        for i in range(1, len(smoothed)):
            dx = smoothed[i] - smoothed[i - 1]
            if abs(dx) < 0.002:
                continue

            current_dir = dx > 0

            if going_positive is None:
                going_positive = current_dir
                continue

            if current_dir != going_positive:
                # Direction changed -- check swing size.
                peak_val = smoothed[i - 1]
                swing = abs(peak_val - last_extreme_val)
                if swing >= self.min_swing:
                    extrema_vals.append(peak_val)
                    extrema_ts.append(times[i - 1])
                    last_extreme_val = peak_val
                    last_extreme_t = times[i - 1]
                going_positive = current_dir

        return (len(extrema_vals), extrema_ts)

    def reset(self) -> None:
        self._data.clear()


# ── Shape Recognition ───────────────────────────────────────────────

class ShapeType(Enum):
    CIRCLE = auto()
    SQUARE = auto()


@dataclass
class ShapeResult:
    matched: bool = False
    confidence: float = 0.0
    reason: str = ""


class ShapeRecognizer:
    """Trajectory recorder with smoothing and deferred validation.

    Points are smoothed with a 3-sample moving average on push().
    Validation only runs when finalize() is called.
    """

    def __init__(self, min_points: int = 15):
        self.min_points = min_points
        self.path: list[tuple[float, float]] = []
        self._raw_buf: list[tuple[float, float]] = []  # last 3 raw points for smoothing

    def push(self, x: float, y: float) -> None:
        """Add a raw point; the smoothed result is appended to self.path."""
        self._raw_buf.append((x, y))

        # Moving average of last 3 raw points.
        if len(self._raw_buf) > 3:
            self._raw_buf.pop(0)

        n = len(self._raw_buf)
        sx = sum(p[0] for p in self._raw_buf) / n
        sy = sum(p[1] for p in self._raw_buf) / n

        # Skip near-duplicates.
        if self.path:
            lx, ly = self.path[-1]
            if abs(sx - lx) < 0.003 and abs(sy - ly) < 0.003:
                return

        self.path.append((sx, sy))

    def finalize_circle(self) -> ShapeResult:
        """Validate path as a circle. Call once when drawing ends."""
        if len(self.path) < self.min_points:
            return ShapeResult(False, 0.0, f"Too few points ({len(self.path)})")

        xs = [p[0] for p in self.path]
        ys = [p[1] for p in self.path]
        bbox_w = max(xs) - min(xs)
        bbox_h = max(ys) - min(ys)

        if bbox_w < 0.05 or bbox_h < 0.05:
            return ShapeResult(False, 0.0, "Drawing too small")

        # Aspect ratio check: circle should be roughly 1:1.
        aspect = min(bbox_w, bbox_h) / max(bbox_w, bbox_h)
        if aspect < 0.55:
            return ShapeResult(False, 0.0, f"Aspect ratio {aspect:.2f} too far from 1:1")

        # Centroid and radii analysis.
        cx = sum(xs) / len(xs)
        cy = sum(ys) / len(ys)
        dists = [math.sqrt((p[0] - cx)**2 + (p[1] - cy)**2) for p in self.path]
        mean_r = sum(dists) / len(dists)

        if mean_r < 0.02:
            return ShapeResult(False, 0.0, "Radius too small")

        # Normalised std-dev of radii.
        variance = sum((d - mean_r)**2 for d in dists) / len(dists)
        norm_std = math.sqrt(variance) / mean_r

        # Coverage: inscribed circle area vs bounding box area.
        circle_area = math.pi * mean_r * mean_r
        bbox_area = bbox_w * bbox_h
        coverage = circle_area / bbox_area if bbox_area > 0 else 0

        # Relaxed thresholds: norm_std < 0.45 and coverage > 0.40.
        std_ok = norm_std < 0.45
        coverage_ok = coverage > 0.40

        confidence = 0.0
        if std_ok:
            confidence += 0.4 * max(0, 1.0 - norm_std / 0.45)
        confidence += 0.3 * aspect
        if coverage_ok:
            confidence += 0.3 * min(coverage / 0.78, 1.0)

        matched = std_ok and aspect >= 0.55 and coverage_ok
        reason = f"std={norm_std:.2f} asp={aspect:.2f} cov={coverage:.2f}"
        return ShapeResult(matched, min(confidence, 1.0), reason)

    def finalize_square(self) -> ShapeResult:
        """Validate path as a square. Call once when drawing ends."""
        if len(self.path) < self.min_points:
            return ShapeResult(False, 0.0, f"Too few points ({len(self.path)})")

        xs = [p[0] for p in self.path]
        ys = [p[1] for p in self.path]
        bbox_w = max(xs) - min(xs)
        bbox_h = max(ys) - min(ys)

        if bbox_w < 0.05 or bbox_h < 0.05:
            return ShapeResult(False, 0.0, "Drawing too small")

        aspect = min(bbox_w, bbox_h) / max(bbox_w, bbox_h)
        corners = self._count_corners()

        aspect_ok = aspect >= 0.50
        corners_ok = corners >= 3

        confidence = 0.0
        if aspect_ok:
            confidence += 0.3 * aspect
        if corners_ok:
            confidence += 0.7 * min(corners / 4.0, 1.0)

        matched = aspect_ok and corners_ok
        reason = f"asp={aspect:.2f} corners={corners}"
        return ShapeResult(matched, min(confidence, 1.0), reason)

    def _count_corners(self) -> int:
        """Count sharp direction changes (> 50 degrees) in subsampled path."""
        if len(self.path) < 8:
            return 0

        # Subsample to ~25 points.
        step = max(1, len(self.path) // 25)
        sampled = self.path[::step]
        if len(sampled) < 5:
            return 0

        corners = 0
        min_gap = max(2, len(sampled) // 8)  # prevent counting adjacent samples as separate corners
        last_corner_i = -min_gap

        for i in range(1, len(sampled) - 1):
            ax = sampled[i][0] - sampled[i-1][0]
            ay = sampled[i][1] - sampled[i-1][1]
            bx = sampled[i+1][0] - sampled[i][0]
            by = sampled[i+1][1] - sampled[i][1]

            mag_a = math.sqrt(ax*ax + ay*ay)
            mag_b = math.sqrt(bx*bx + by*by)
            if mag_a < 1e-6 or mag_b < 1e-6:
                continue

            cos_angle = max(-1.0, min(1.0, (ax*bx + ay*by) / (mag_a * mag_b)))
            angle_deg = math.degrees(math.acos(cos_angle))

            if angle_deg > 50.0 and (i - last_corner_i) >= min_gap:
                corners += 1
                last_corner_i = i

        return corners

    def reset(self) -> None:
        self.path.clear()
        self._raw_buf.clear()

    @property
    def pixel_path(self) -> list[tuple[float, float]]:
        return list(self.path)
