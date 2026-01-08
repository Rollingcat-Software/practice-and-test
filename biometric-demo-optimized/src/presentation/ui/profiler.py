"""Performance profiler with visual overlay."""

import time
from typing import Dict
from collections import deque

import cv2
import numpy as np

from .colors import Colors


class Profiler:
    """Performance profiler with timing context manager and visual display."""

    def __init__(self, enabled: bool = False):
        self.enabled = enabled
        self._metrics: Dict[str, deque] = {}

    def time(self, name: str) -> 'ProfilerContext':
        """Context manager for timing a code block.

        Usage:
            with profiler.time("detection"):
                # code to time
        """
        return ProfilerContext(self, name)

    def record(self, name: str, ms: float):
        """Record a timing measurement."""
        if name not in self._metrics:
            self._metrics[name] = deque(maxlen=60)
        self._metrics[name].append(ms)

    def get_avg(self, name: str) -> float:
        """Get average time for a metric."""
        if name in self._metrics and self._metrics[name]:
            return sum(self._metrics[name]) / len(self._metrics[name])
        return 0

    def draw(self, frame: np.ndarray):
        """Draw profiler overlay on frame."""
        if not self.enabled:
            return

        h, w = frame.shape[:2]
        panel_h = len(self._metrics) * 18 + 25

        # Draw background
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 45), (250, 45 + panel_h), Colors.BLACK, -1)
        cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)

        # Draw title
        cv2.putText(frame, "PROFILER", (15, 62),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, Colors.CYAN, 1)

        # Draw metrics
        y = 80
        for name, vals in sorted(self._metrics.items()):
            avg = sum(vals) / len(vals) if vals else 0
            color = Colors.GREEN if avg < 30 else Colors.YELLOW if avg < 60 else Colors.RED
            cv2.putText(frame, f"{name}: {avg:.1f}ms", (15, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1)
            y += 18

    def get_summary(self) -> Dict[str, float]:
        """Get summary of all metrics."""
        return {name: self.get_avg(name) for name in self._metrics}


class ProfilerContext:
    """Context manager for timing code blocks."""

    def __init__(self, profiler: Profiler, name: str):
        self._profiler = profiler
        self._name = name
        self._start = 0

    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, *args):
        ms = (time.perf_counter() - self._start) * 1000
        self._profiler.record(self._name, ms)
