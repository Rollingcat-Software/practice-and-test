"""Utilities for color-flash active liveness challenges."""

from __future__ import annotations

import random
import time
from typing import Optional, Sequence

import numpy as np


class LightChallengeService:
    """Generates and verifies short screen-flash challenges."""

    COLORS = ("red", "green", "blue", "white", "yellow")
    _COLOR_CHANNEL_INDEX = {
        "red": 2,
        "green": 1,
        "blue": 0,
    }

    def __init__(
        self,
        flash_duration_ms: int = 150,
        expected_response_window_ms: int = 500,
        minimum_delay_ms: int = 50,
        min_color_shift: float = 0.05,
        colors: Optional[Sequence[str]] = None,
    ) -> None:
        self._flash_duration_ms = flash_duration_ms
        self._expected_response_window_ms = expected_response_window_ms
        self._minimum_delay_ms = minimum_delay_ms
        self._min_color_shift = min_color_shift
        self._colors = tuple(colors or self.COLORS)

    def generate_challenge(self) -> dict:
        issued_at = time.time()
        return {
            "color": random.choice(self._colors),
            "available_colors": list(self._colors),
            "issued_at": issued_at,
            "expires_at": issued_at + (self._expected_response_window_ms / 1000.0),
            "duration_ms": self._flash_duration_ms,
            "expected_response_window_ms": self._expected_response_window_ms,
            "minimum_delay_ms": self._minimum_delay_ms,
            "baseline_required": True,
            "ready_for_flash": False,
        }

    def verify_response(
        self,
        frame: np.ndarray,
        expected_color: str,
        flash_timestamp: float,
        frame_timestamp: Optional[float],
        baseline_bgr: Optional[Sequence[float]] = None,
    ) -> dict:
        verified_frame_timestamp = self._normalize_timestamp(frame_timestamp) or time.time()
        verified_flash_timestamp = self._normalize_timestamp(flash_timestamp) or time.time()
        delay = verified_frame_timestamp - verified_flash_timestamp

        min_delay = self._minimum_delay_ms / 1000.0
        max_delay = self._expected_response_window_ms / 1000.0
        if delay < min_delay or delay > max_delay:
            return {
                "passed": False,
                "reason": "timing_mismatch",
                "delay_seconds": delay,
            }

        color_shift, face_mean_bgr = self._detect_color_shift(
            frame=frame,
            expected_color=expected_color,
            baseline_bgr=baseline_bgr,
        )

        if color_shift >= self._min_color_shift:
            return {
                "passed": True,
                "color_shift": color_shift,
                "delay_seconds": delay,
                "face_mean_bgr": face_mean_bgr,
            }

        return {
            "passed": False,
            "reason": "no_color_response",
            "color_shift": color_shift,
            "delay_seconds": delay,
            "face_mean_bgr": face_mean_bgr,
        }

    def _detect_color_shift(
        self,
        frame: np.ndarray,
        expected_color: str,
        baseline_bgr: Optional[Sequence[float]] = None,
    ) -> tuple[float, list[float]]:
        if frame.size == 0:
            return 0.0, [0.0, 0.0, 0.0]

        face_mean = frame.mean(axis=(0, 1)).astype(float)

        if expected_color in self._COLOR_CHANNEL_INDEX:
            target_index = self._COLOR_CHANNEL_INDEX[expected_color]
            other_indexes = [idx for idx in range(3) if idx != target_index]
            target_value = face_mean[target_index]
            other_value = float(np.mean(face_mean[other_indexes]))

            if baseline_bgr is not None and len(baseline_bgr) == 3:
                baseline = np.asarray(baseline_bgr, dtype=float)
                target_delta = max(0.0, target_value - baseline[target_index])
                other_delta = max(0.0, other_value - float(np.mean(baseline[other_indexes])))
                baseline_sum = float(np.sum(baseline) + 1e-6)
                current_sum = float(np.sum(face_mean) + 1e-6)
                baseline_target_chroma = float(baseline[target_index] / baseline_sum)
                current_target_chroma = float(target_value / current_sum)
                baseline_other_chroma = float(np.mean([baseline[idx] / baseline_sum for idx in other_indexes]))
                current_other_chroma = float(np.mean([face_mean[idx] / current_sum for idx in other_indexes]))
                chroma_gain = max(0.0, (current_target_chroma - baseline_target_chroma) - (current_other_chroma - baseline_other_chroma))
                absolute_gain = max(0.0, (target_delta - other_delta) / 255.0)
                dominance_gain = max(0.0, target_delta / max(target_delta + other_delta, 1e-6) - 0.45)
                shift = max(
                    0.0,
                    0.45 * absolute_gain
                    + 0.40 * min(1.0, chroma_gain / 0.08)
                    + 0.15 * min(1.0, dominance_gain / 0.35),
                )
            else:
                raw_gain = max(0.0, (target_value - other_value) / 255.0)
                dominance = max(0.0, target_value / max(target_value + other_value, 1e-6) - 0.45)
                shift = max(0.0, 0.75 * raw_gain + 0.25 * min(1.0, dominance / 0.35))
            return float(shift), face_mean.tolist()

        brightness = float(np.mean(face_mean))
        if baseline_bgr is not None and len(baseline_bgr) == 3:
            baseline_brightness = float(np.mean(np.asarray(baseline_bgr, dtype=float)))
            brightness_shift = max(0.0, (brightness - baseline_brightness) / 255.0)
        else:
            brightness_shift = brightness / 255.0

        if expected_color == "yellow":
            blue_penalty = max(0.0, face_mean[0] - ((face_mean[1] + face_mean[2]) / 2.0))
            brightness_shift = max(0.0, brightness_shift - (blue_penalty / 255.0))

        return brightness_shift, face_mean.tolist()

    @staticmethod
    def _normalize_timestamp(timestamp: Optional[float]) -> Optional[float]:
        if timestamp is None:
            return None
        if timestamp > 1_000_000_000_000:
            return timestamp / 1000.0
        return timestamp
