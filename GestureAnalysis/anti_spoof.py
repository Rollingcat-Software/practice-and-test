"""
Anti-Spoofing Analytics
========================
Background checks to detect static images or frozen video being
held in front of the camera instead of a live person.

MicroTremorDetector
-------------------
Real human hands are never perfectly still -- there's always a
natural micro-tremor (involuntary muscle oscillation, typically
8-12 Hz).  If the wrist landmark is 100% static for > 1 second,
the hand is likely a photo or frozen video.

We track the standard deviation of recent wrist positions.  If
std-dev drops below a threshold, we flag it.

BrightnessMonitor
-----------------
Tracks the average pixel intensity around hand landmarks over time.
A real hand in a real scene shows subtle brightness fluctuations
from ambient light changes, breathing shadows, etc.  A static
image has near-zero variance.
"""

import math
import time
from collections import deque
from dataclasses import dataclass


@dataclass
class SpoofResult:
    """Anti-spoof analysis result."""
    is_suspicious: bool = False
    tremor_ok: bool = True       # True = natural tremor detected (good)
    brightness_ok: bool = True   # True = brightness varies naturally (good)
    tremor_std: float = 0.0
    brightness_std: float = 0.0


class MicroTremorDetector:
    """Detect absence of natural hand tremor.

    A real hand's wrist jitters by at least ~0.002 normalised units
    per frame.  If the std-dev of recent positions drops below this,
    the hand is suspiciously static.

    Parameters
    ----------
    buffer_size : int
        Frames to keep (default 30 = ~1 second at 30fps).
    min_std : float
        Minimum std-dev of wrist position to be considered "alive".
    warmup_frames : int
        Frames to wait before flagging (avoid false alarm on startup).
    """

    def __init__(self, buffer_size: int = 30, min_std: float = 0.0003,
                 warmup_frames: int = 90):
        self.min_std = min_std
        self.warmup = warmup_frames
        self._xs: deque[float] = deque(maxlen=buffer_size)
        self._ys: deque[float] = deque(maxlen=buffer_size)
        self._frame_count = 0

    def push(self, wrist_x: float, wrist_y: float) -> None:
        self._xs.append(wrist_x)
        self._ys.append(wrist_y)
        self._frame_count += 1

    @property
    def std_dev(self) -> float:
        """Combined std-dev of x and y positions."""
        if len(self._xs) < 10:
            return 1.0  # not enough data, assume alive
        mean_x = sum(self._xs) / len(self._xs)
        mean_y = sum(self._ys) / len(self._ys)
        var_x = sum((x - mean_x)**2 for x in self._xs) / len(self._xs)
        var_y = sum((y - mean_y)**2 for y in self._ys) / len(self._ys)
        return math.sqrt(var_x + var_y)

    @property
    def is_suspicious(self) -> bool:
        """True if the hand is unnaturally still (possible spoof)."""
        if self._frame_count < self.warmup:
            return False
        return self.std_dev < self.min_std

    def reset(self):
        self._xs.clear()
        self._ys.clear()
        self._frame_count = 0


class BrightnessMonitor:
    """Track brightness variance around hand landmarks.

    Parameters
    ----------
    buffer_size : int
        Number of brightness samples to keep.
    min_variance : float
        Minimum brightness variance to be considered natural.
    """

    def __init__(self, buffer_size: int = 60, min_variance: float = 0.05):
        self.min_var = min_variance
        self._values: deque[float] = deque(maxlen=buffer_size)

    def push(self, avg_brightness: float) -> None:
        """Push the average brightness of the hand region."""
        self._values.append(avg_brightness)

    @property
    def std_dev(self) -> float:
        if len(self._values) < 20:
            return 999.0  # not enough data
        mean = sum(self._values) / len(self._values)
        var = sum((v - mean)**2 for v in self._values) / len(self._values)
        return math.sqrt(var)

    @property
    def is_suspicious(self) -> bool:
        if len(self._values) < 30:
            return False
        return self.std_dev < self.min_var

    def reset(self):
        self._values.clear()


class AntiSpoofAnalyzer:
    """Combined anti-spoof checker.

    Feed it wrist coordinates and brightness each frame.
    Query ``analyze()`` to get the current spoof assessment.
    """

    def __init__(self):
        self.tremor = MicroTremorDetector()
        self.brightness = BrightnessMonitor()

    def feed(self, wrist_x: float, wrist_y: float, avg_brightness: float) -> None:
        self.tremor.push(wrist_x, wrist_y)
        self.brightness.push(avg_brightness)

    def analyze(self) -> SpoofResult:
        tremor_ok = not self.tremor.is_suspicious
        bright_ok = not self.brightness.is_suspicious
        return SpoofResult(
            is_suspicious=not tremor_ok or not bright_ok,
            tremor_ok=tremor_ok,
            brightness_ok=bright_ok,
            tremor_std=self.tremor.std_dev,
            brightness_std=self.brightness.std_dev,
        )

    def reset(self):
        self.tremor.reset()
        self.brightness.reset()
