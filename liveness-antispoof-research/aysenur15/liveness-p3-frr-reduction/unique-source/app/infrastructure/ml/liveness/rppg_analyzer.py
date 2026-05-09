"""Remote photoplethysmography analyzer for live liveness streams."""

from __future__ import annotations

from collections import deque
from typing import Deque

import numpy as np
from scipy import signal


class RPPGAnalyzer:
    """Estimate pulse-like periodicity from the face ROI green channel."""

    def __init__(self, fps: float = 30.0, window_seconds: float = 5.0) -> None:
        self.fps = fps
        self.window_size = max(1, int(fps * window_seconds))
        self.green_channel_history: Deque[float] = deque(maxlen=self.window_size)

    def reset(self) -> None:
        """Clear accumulated frame history."""
        self.green_channel_history.clear()

    def add_frame(self, face_roi: np.ndarray) -> None:
        """Track the mean green-channel intensity for one face crop."""
        if face_roi is None or face_roi.size == 0:
            return
        green_mean = float(face_roi[:, :, 1].mean())
        self.green_channel_history.append(green_mean)

    def analyze(self) -> dict[str, float | str | None | int]:
        """Estimate whether a plausible pulse signal is present."""
        min_required = max(1, int(self.fps * 2))
        frame_count = len(self.green_channel_history)
        if frame_count < min_required:
            return {
                "score": 0.5,
                "reason": "insufficient_frames",
                "bpm": None,
                "signal_strength": 0.0,
                "frame_count": frame_count,
            }

        signal_array = np.array(self.green_channel_history, dtype=np.float64)
        detrended = signal_array - np.mean(signal_array)

        if np.allclose(detrended, 0.0):
            return {
                "score": 0.2,
                "reason": "no_pulse_signal",
                "bpm": None,
                "signal_strength": 0.0,
                "frame_count": frame_count,
            }

        nyquist = self.fps / 2.0
        low = 0.83 / nyquist
        high = 2.5 / nyquist
        b, a = signal.butter(3, [low, high], btype="band")
        filtered = signal.filtfilt(b, a, detrended)

        fft = np.abs(np.fft.rfft(filtered))
        freqs = np.fft.rfftfreq(len(filtered), 1.0 / self.fps)
        valid_mask = (freqs >= 0.83) & (freqs <= 2.5)

        if not valid_mask.any():
            return {
                "score": 0.2,
                "reason": "no_valid_frequency",
                "bpm": None,
                "signal_strength": 0.0,
                "frame_count": frame_count,
            }

        valid_freqs = freqs[valid_mask]
        valid_fft = fft[valid_mask]
        dominant_index = int(np.argmax(valid_fft))
        dominant_freq = float(valid_freqs[dominant_index])
        bpm = dominant_freq * 60.0
        signal_strength = float(valid_fft[dominant_index] / (fft.max() + 1e-6))

        if signal_strength > 0.3:
            return {
                "score": min(signal_strength * 2.0, 1.0),
                "reason": "pulse_detected",
                "bpm": round(bpm, 1),
                "signal_strength": signal_strength,
                "frame_count": frame_count,
            }

        return {
            "score": 0.2,
            "reason": "no_pulse_signal",
            "bpm": None,
            "signal_strength": signal_strength,
            "frame_count": frame_count,
        }
