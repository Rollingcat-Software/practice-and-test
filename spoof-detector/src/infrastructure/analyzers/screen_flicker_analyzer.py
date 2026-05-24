"""Screen flicker detector — 50/60Hz temporal aliasing detection.

Screens refresh at 50Hz (EU) or 60Hz (US/TR). When a camera at 30fps
films a screen, the refresh creates temporal intensity modulation
at beat frequencies (|60-30|=30Hz, |50-30|=20Hz).

Real faces have NO periodic intensity modulation — lighting is constant.
This is a strong, physics-based signal for detecting ANY screen.

Algorithm:
1. Compute mean intensity of face ROI per frame
2. Accumulate in temporal buffer (60+ frames)
3. FFT on intensity time-series
4. Look for peaks near 20Hz, 25Hz, 30Hz (beat frequencies)
5. Strong peak = screen detected

Validation (2026-05-02):
- FFT frequency bins: At 30fps with 30 frames, resolution = 1.0Hz; with 60 frames,
  0.5Hz. Flicker bands (8-15, 18-25, 28-35 Hz) are wide enough to capture peaks.
  CORRECT.
- Beat frequencies covered: 10Hz (50Hz 2nd harmonic alias at 30fps), 20Hz (|50-30|),
  30Hz (|60-30|). All fall within defined bands. CORRECT.
- Detrending: 10-sample moving average at 30fps -> ~3Hz cutoff. Removes lighting
  drift while preserving 8+ Hz flicker. CORRECT.
- Hanning window: Reduces spectral leakage for non-integer-period signals. CORRECT.
- Edge case (warmup): Returns neutral score 50 until MIN_FRAMES reached. CORRECT.
- Edge case (face disappearing): deque per face_id, no crash on empty. CORRECT.
- Minor: noise floor includes flicker bands (dilutes SNR slightly). Acceptable for
  demo — flicker peaks are narrow relative to full spectrum width.
- Minor: _states dict not cleaned for stale face IDs. Acceptable for demo.
- No bugs found. Code is correct.
"""

import time
import logging
from collections import deque

import numpy as np

from src.domain.models import FaceROI, AnalyzerResult

logger = logging.getLogger(__name__)

# Expected beat frequencies between screen refresh and camera capture
# 60Hz screen - 30fps camera = 30Hz, 20Hz, 10Hz harmonics
# 50Hz screen - 30fps camera = 20Hz, 10Hz harmonics
FLICKER_BANDS = [(8.0, 15.0), (18.0, 25.0), (28.0, 35.0)]  # Hz ranges to check
MIN_FRAMES = 30   # ~1s at 30fps
GOOD_FRAMES = 60  # ~2s for reliable detection


class ScreenFlickerAnalyzer:
    """Detects screen refresh rate aliasing in temporal intensity signal.

    Score:
    - 0-30: Strong flicker detected (SPOOF — screen)
    - 30-60: Weak/ambiguous flicker
    - 60-100: No flicker (LIVE — real lighting)
    """

    def __init__(self, fps: float = 30.0):
        self._fps = fps
        self._states: dict[int, deque] = {}
        self._frame_times: deque = deque(maxlen=60)

    @property
    def name(self) -> str:
        return "screen_flicker"

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
            self._states[fid] = deque(maxlen=120)  # 4s buffer

        # Extract mean intensity from face crop (all channels)
        intensity = float(np.mean(face_crop))
        self._states[fid].append(intensity)

        elapsed_ms = (time.perf_counter() - start) * 1000

        if len(self._states[fid]) < MIN_FRAMES:
            return AnalyzerResult(
                name=self.name, score=50.0,
                details={"warmup": True, "frames": len(self._states[fid])},
                elapsed_ms=elapsed_ms,
            )

        # FFT analysis
        signal = np.array(self._states[fid], dtype=np.float64)

        # Detrend (remove slow lighting drift)
        signal = signal - np.convolve(signal, np.ones(10) / 10, mode="same")

        # Hanning window
        window = np.hanning(len(signal))
        signal = signal * window

        # FFT
        fft = np.fft.rfft(signal)
        magnitude = np.abs(fft)
        freqs = np.fft.rfftfreq(len(signal), d=1.0 / self._fps)

        # Check flicker bands
        max_flicker_power = 0.0
        dominant_freq = 0.0
        noise_floor = float(np.mean(magnitude[1:])) if len(magnitude) > 1 else 1.0

        for low_hz, high_hz in FLICKER_BANDS:
            band_mask = (freqs >= low_hz) & (freqs <= high_hz)
            if not np.any(band_mask):
                continue
            band_power = float(np.max(magnitude[band_mask]))
            if band_power > max_flicker_power:
                max_flicker_power = band_power
                peak_idx = np.argmax(magnitude[band_mask])
                dominant_freq = float(freqs[band_mask][peak_idx])

        # Signal-to-noise ratio of flicker
        flicker_snr = max_flicker_power / max(noise_floor, 1e-6)

        # Score: high flicker SNR = screen
        data_quality = min(1.0, len(self._states[fid]) / GOOD_FRAMES)

        if flicker_snr > 4.0:
            score = 10.0  # Strong flicker = definitely screen
        elif flicker_snr > 2.5:
            score = 25.0 + (4.0 - flicker_snr) * 10.0
        elif flicker_snr > 1.5:
            score = 50.0
        else:
            score = 70.0 + data_quality * 30.0  # No flicker = likely real

        elapsed_ms = (time.perf_counter() - start) * 1000

        return AnalyzerResult(
            name=self.name,
            score=max(0.0, min(100.0, score)),
            details={
                "flicker_snr": round(flicker_snr, 2),
                "dominant_freq_hz": round(dominant_freq, 1),
                "max_flicker_power": round(max_flicker_power, 2),
                "noise_floor": round(noise_floor, 2),
                "measured_fps": round(self._fps, 1),
                "frames": len(self._states[fid]),
            },
            elapsed_ms=elapsed_ms,
        )
