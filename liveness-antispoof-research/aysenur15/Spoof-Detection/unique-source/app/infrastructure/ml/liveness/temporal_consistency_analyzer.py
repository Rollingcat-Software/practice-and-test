"""Temporal consistency analyzer for live liveness streams.

This analyzer tracks facial landmarks across a short frame window and estimates
whether the subject exhibits small, natural micro-movements over time.
"""

from __future__ import annotations

from collections import deque
from math import hypot
from typing import Deque


class TemporalConsistencyAnalyzer:
    """Analyze short-window landmark motion consistency."""

    def __init__(
        self,
        window_size: int = 10,
        min_frames: int = 3,
        no_movement_threshold: float = 0.0015,
        excessive_movement_threshold: float = 0.08,
    ) -> None:
        self.window_size = window_size
        self.min_frames = min_frames
        self.no_movement_threshold = no_movement_threshold
        self.excessive_movement_threshold = excessive_movement_threshold
        self.landmark_history: Deque[list[tuple[float, float]]] = deque(maxlen=window_size)

    def reset(self) -> None:
        """Clear tracked frame history."""
        self.landmark_history.clear()

    def add_frame(self, landmarks: list[tuple[float, float]]) -> None:
        """Add normalized landmark coordinates for one frame."""
        if not landmarks:
            return
        self.landmark_history.append(landmarks)

    def analyze(self) -> dict[str, float | int | str]:
        """Compute temporal consistency score from tracked landmarks."""
        frame_count = len(self.landmark_history)
        if frame_count < self.min_frames:
            return {
                "score": 0.5,
                "reason": "insufficient_frames",
                "frame_count": frame_count,
                "avg_movement": 0.0,
                "variance": 0.0,
            }

        frames = list(self.landmark_history)
        movements: list[float] = []
        for index in range(1, frame_count):
            delta = self._calculate_movement(frames[index - 1], frames[index])
            movements.append(delta)

        if not movements:
            return {
                "score": 0.5,
                "reason": "insufficient_frames",
                "frame_count": frame_count,
                "avg_movement": 0.0,
                "variance": 0.0,
            }

        avg_movement = sum(movements) / len(movements)
        variance = sum((movement - avg_movement) ** 2 for movement in movements) / len(movements)

        if avg_movement < self.no_movement_threshold:
            score = 0.2
            reason = "no_movement"
        elif avg_movement > self.excessive_movement_threshold:
            score = 0.3
            reason = "excessive_movement"
        else:
            score = max(0.2, 1.0 - min(variance * 25.0, 0.8))
            reason = "natural_movement"

        return {
            "score": score,
            "reason": reason,
            "frame_count": frame_count,
            "avg_movement": avg_movement,
            "variance": variance,
        }

    def _calculate_movement(
        self,
        landmarks_a: list[tuple[float, float]],
        landmarks_b: list[tuple[float, float]],
    ) -> float:
        """Average Euclidean landmark delta between two frames."""
        pair_count = min(len(landmarks_a), len(landmarks_b))
        if pair_count == 0:
            return 0.0

        total = 0.0
        for point_a, point_b in zip(landmarks_a[:pair_count], landmarks_b[:pair_count]):
            total += hypot(point_a[0] - point_b[0], point_a[1] - point_b[1])
        return total / pair_count
