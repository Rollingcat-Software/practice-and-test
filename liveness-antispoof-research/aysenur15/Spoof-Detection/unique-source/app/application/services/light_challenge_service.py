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
        measurement_mask: Optional[np.ndarray] = None,
        expected_max_response: float = 40.0,
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
            measurement_mask=measurement_mask,
            expected_max_response=expected_max_response,
        )

        if color_shift >= self._min_color_shift:
            return {
                "passed": True,
                "color_shift": color_shift,
                "channel_response": color_shift * max(float(expected_max_response), 1.0),
                "delay_seconds": delay,
                "face_mean_bgr": face_mean_bgr,
            }

        return {
            "passed": False,
            "reason": "no_color_response",
            "color_shift": color_shift,
            "channel_response": color_shift * max(float(expected_max_response), 1.0),
            "delay_seconds": delay,
            "face_mean_bgr": face_mean_bgr,
        }

    def _detect_color_shift(
        self,
        frame: np.ndarray,
        expected_color: str,
        baseline_bgr: Optional[Sequence[float]] = None,
        measurement_mask: Optional[np.ndarray] = None,
        expected_max_response: float = 40.0,
    ) -> tuple[float, list[float]]:
        if frame.size == 0:
            return 0.0, [0.0, 0.0, 0.0]

        face_mean = self._masked_channel_mean(frame, measurement_mask)
        max_response = max(float(expected_max_response), 1.0)

        if expected_color in self._COLOR_CHANNEL_INDEX:
            target_index = self._COLOR_CHANNEL_INDEX[expected_color]
            if baseline_bgr is not None and len(baseline_bgr) == 3:
                baseline = np.asarray(baseline_bgr, dtype=float)
                channel_response = max(0.0, float(face_mean[target_index] - baseline[target_index]))
                shift = max(0.0, min(1.0, channel_response / max_response))
            else:
                shift = 0.0
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
    def _masked_channel_mean(frame: np.ndarray, measurement_mask: Optional[np.ndarray]) -> np.ndarray:
        if measurement_mask is None or measurement_mask.size == 0:
            return frame.mean(axis=(0, 1)).astype(float)

        if measurement_mask.ndim == 3:
            mask = measurement_mask[:, :, 0] > 0
        else:
            mask = measurement_mask > 0

        valid_pixels = frame[mask]
        if valid_pixels.size == 0:
            return frame.mean(axis=(0, 1)).astype(float)
        return valid_pixels.reshape(-1, 3).mean(axis=0).astype(float)

    @staticmethod
    def _normalize_timestamp(timestamp: Optional[float]) -> Optional[float]:
        if timestamp is None:
            return None
        if timestamp > 1_000_000_000_000:
            return timestamp / 1000.0
        return timestamp
