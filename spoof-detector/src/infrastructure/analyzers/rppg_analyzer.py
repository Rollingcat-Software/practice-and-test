"""Remote Photoplethysmography (rPPG) pulse analyzer.

Detects blood pulse signal from subtle green-channel variations
in the face. Real faces show a measurable heartbeat (60-100 BPM).
Screens, photos, and masks show no pulse.

This is the gold standard for passive liveness detection:
- Requires ~3-5 seconds of continuous video
- Zero false positives on static images (no pulse = no life)
- Works even when MiniFASNet is fooled by high-quality screen photos

Method:
1. Extract mean green channel from forehead/cheek ROI per frame
2. Accumulate in circular buffer (150 frames = 5s at 30fps)
3. Apply bandpass filter (0.75-4.0 Hz = 45-240 BPM)
4. FFT to find dominant frequency
5. Score based on signal strength (SNR of peak vs noise)
"""

import time
import logging
from collections import deque
from dataclasses import dataclass, field

import numpy as np
from scipy.signal import iirnotch, sosfilt, butter

from src.domain.models import FaceROI, AnalyzerResult

logger = logging.getLogger(__name__)

# Bandpass filter range (Hz)
PULSE_LOW_HZ = 0.75   # 45 BPM
PULSE_HIGH_HZ = 4.0   # 240 BPM
MIN_FRAMES = 60        # ~2s at 30fps minimum for any signal
GOOD_FRAMES = 150      # ~5s for reliable pulse detection

# Screen flicker notch filter frequencies (Hz).
#
# At 30 fps the Nyquist limit is 15 Hz, so screen refresh rates
# alias into the 0-15 Hz band and contaminate the pulse range:
#
#   50 Hz screen → aliases to |50 - 2*30| = 10 Hz
#   60 Hz screen → aliases to |60 - 2*30| = 0 Hz (DC, harmless)
#                  also |60 - 30| = 30 Hz, but that's above Nyquist
#   100 Hz (2nd harmonic of 50 Hz) → |100 - 3*30| = 10 Hz
#   120 Hz (2nd harmonic of 60 Hz) → |120 - 4*30| = 0 Hz (DC)
#
# The dominant alias is 10 Hz. We also notch 15 Hz (Nyquist edge,
# where spectral leakage from any near-Nyquist artifact concentrates)
# and 20 Hz (|50 - 30| fundamental beat, theoretically above Nyquist
# but still appears in real FFT bins due to spectral leakage from
# rectangular/windowed sampling near the folding boundary).
#
# Q factor 5-10 gives a narrow notch (~1-2 Hz wide) that does not
# affect the 0.75-4.0 Hz pulse band at all.
NOTCH_FREQS_HZ = [10.0, 15.0, 20.0]
NOTCH_Q = 8.0  # Quality factor — narrower is safer for pulse band


@dataclass
class PulseState:
    """Per-face rPPG tracking state."""
    green_values: deque = field(default_factory=lambda: deque(maxlen=300))  # 10s buffer
    frame_count: int = 0
    estimated_bpm: float | None = None
    signal_strength: float = 0.0


class RPPGAnalyzer:
    """Remote PPG pulse detection from green channel variation.

    Accumulates green channel means over time per face_id.
    After ~3 seconds, can detect whether a pulse signal is present.

    Score:
    - 0-20: No pulse detected (SPOOF — screen/photo/mask)
    - 20-50: Weak/ambiguous signal (need more time)
    - 50-100: Pulse detected (LIVE — real person)
    """

    def __init__(self, fps: float = 30.0):
        self._fps = fps
        self._states: dict[int, PulseState] = {}
        self._frame_times: deque = deque(maxlen=60)
        self._measured_fps: float = fps
        # Pre-compute notch filter coefficients (SOS form) for screen
        # flicker suppression.  These are designed at the nominal fps;
        # the aliased artefact frequencies don't shift much with small
        # fps drift so a one-time design is fine.
        self._notch_sos = self._design_notch_filters(fps)
        # Butterworth bandpass isolates the pulse band (0.75-4.0 Hz)
        # in the time domain, killing high-frequency screen artefacts
        # before they can leak into pulse bins via windowed FFT.
        self._bandpass_sos = butter(
            4, [PULSE_LOW_HZ, PULSE_HIGH_HZ],
            btype="band", fs=fps, output="sos",
        )

    @staticmethod
    def _design_notch_filters(fps: float) -> np.ndarray:
        """Build a cascade of IIR notch filters (SOS) for screen flicker.

        Only includes notch frequencies that are below the Nyquist limit
        for the given sample rate, so it is safe for any fps.
        """
        nyquist = fps / 2.0
        sections = []
        for freq in NOTCH_FREQS_HZ:
            if freq >= nyquist:
                continue  # skip frequencies at or above Nyquist
            b, a = iirnotch(freq, NOTCH_Q, fs=fps)
            # Convert to second-order section for numerical stability
            # iirnotch returns a single biquad, so we can pack it directly.
            # SOS row: [b0, b1, b2, 1, a1, a2]
            sos_row = np.array([b[0], b[1], b[2], a[0], a[1], a[2]])
            sections.append(sos_row)
        if not sections:
            return np.zeros((0, 6))
        return np.array(sections)

    @property
    def name(self) -> str:
        return "rppg"

    def analyze(self, face_crop: np.ndarray, face_roi: FaceROI) -> AnalyzerResult:
        start = time.perf_counter()

        # Measure actual FPS
        self._frame_times.append(start)
        if len(self._frame_times) > 10:
            dt = self._frame_times[-1] - self._frame_times[0]
            if dt > 0:
                self._measured_fps = (len(self._frame_times) - 1) / dt
                self._fps = self._measured_fps

        fid = face_roi.face_id
        if fid not in self._states:
            self._states[fid] = PulseState()
        state = self._states[fid]
        state.frame_count += 1

        # Extract green channel mean from forehead region
        # (upper 40% of face crop — forehead has strongest rPPG signal)
        h, w = face_crop.shape[:2]
        forehead = face_crop[0:int(h * 0.4), int(w * 0.2):int(w * 0.8)]
        if forehead.size > 0:
            green_mean = float(np.mean(forehead[:, :, 1]))  # Green channel (BGR)
        else:
            green_mean = 0.0

        state.green_values.append(green_mean)
        elapsed_ms = (time.perf_counter() - start) * 1000

        # Not enough data yet
        if state.frame_count < MIN_FRAMES:
            return AnalyzerResult(
                name=self.name, score=50.0,
                details={"warmup": True, "frames": state.frame_count,
                          "need": MIN_FRAMES},
                elapsed_ms=elapsed_ms,
            )

        # Analyze pulse signal
        signal = np.array(state.green_values, dtype=np.float64)

        # Detrend (remove slow drift from lighting changes)
        signal = signal - np.convolve(signal, np.ones(15) / 15, mode="same")

        # ── Bandpass + notch filters (time domain) ────────────────
        # 1) Butterworth bandpass (0.75-4.0 Hz) isolates the pulse
        #    band and kills high-frequency screen artefacts that
        #    would otherwise leak into pulse bins via windowed FFT.
        signal = sosfilt(self._bandpass_sos, signal)
        # 2) Notch filters remove any aliased 50/60 Hz screen
        #    refresh artefacts that survive the bandpass rolloff.
        if self._notch_sos.shape[0] > 0:
            signal = sosfilt(self._notch_sos, signal)

        # Apply Hanning window
        window = np.hanning(len(signal))
        signal = signal * window

        # FFT
        fft = np.fft.rfft(signal)
        magnitude = np.abs(fft)
        freqs = np.fft.rfftfreq(len(signal), d=1.0 / self._fps)

        # ── Screen-flicker notch filters (frequency domain) ─────
        # Belt-and-suspenders: zero out any residual energy at the
        # exact aliased bins so even spectral leakage is squashed.
        # Track which bins were killed so we exclude them from noise.
        nyquist = self._fps / 2.0
        notch_killed = np.zeros(len(freqs), dtype=bool)
        for nf in NOTCH_FREQS_HZ:
            if nf >= nyquist:
                continue
            # Zero a +-1.0 Hz band around each notch frequency to catch
            # spectral leakage from windowing (Hanning sidelobes)
            kill_mask = (freqs >= nf - 1.0) & (freqs <= nf + 1.0)
            magnitude[kill_mask] = 0.0
            notch_killed |= kill_mask

        # Find pulse band
        pulse_mask = (freqs >= PULSE_LOW_HZ) & (freqs <= PULSE_HIGH_HZ)
        if not np.any(pulse_mask):
            return AnalyzerResult(
                name=self.name, score=50.0,
                details={"error": "no_pulse_band"},
                elapsed_ms=elapsed_ms,
            )

        pulse_magnitudes = magnitude[pulse_mask]
        pulse_freqs = freqs[pulse_mask]

        # Find dominant frequency in pulse range
        peak_idx = np.argmax(pulse_magnitudes)
        peak_freq = float(pulse_freqs[peak_idx])
        peak_magnitude = float(pulse_magnitudes[peak_idx])

        # Signal-to-noise ratio (exclude DC and notch-killed bins).
        # Use the UNFILTERED magnitude for noise estimation so that
        # the bandpass/notch suppression of out-of-band energy does
        # not artificially deflate the noise floor to near-zero.
        raw_fft = np.fft.rfft(
            np.array(state.green_values, dtype=np.float64)
            - np.convolve(
                np.array(state.green_values, dtype=np.float64),
                np.ones(15) / 15, mode="same",
            )
        )
        raw_magnitude = np.abs(raw_fft)
        noise_mask = ~pulse_mask & (freqs > 0.1) & ~notch_killed
        if np.any(noise_mask):
            noise_mean = float(np.mean(raw_magnitude[noise_mask]))
        else:
            noise_mean = 1e-6

        snr = peak_magnitude / max(noise_mean, 1e-6)
        bpm = peak_freq * 60.0

        state.estimated_bpm = bpm if snr > 2.0 else None
        state.signal_strength = min(1.0, snr / 5.0)

        # Score based on signal quality
        data_quality = min(1.0, state.frame_count / GOOD_FRAMES)

        if snr > 4.0 and 45 <= bpm <= 200:
            score = 70.0 + data_quality * 30.0  # Strong pulse = live
        elif snr > 2.5 and 45 <= bpm <= 200:
            score = 50.0 + data_quality * 20.0  # Weak pulse = probably live
        elif snr > 1.5:
            score = 30.0 + data_quality * 20.0  # Ambiguous
        else:
            # No pulse detected
            if state.frame_count > GOOD_FRAMES:
                score = 10.0  # Enough data, no pulse = spoof
            else:
                score = 30.0  # Not enough data yet

        elapsed_ms = (time.perf_counter() - start) * 1000

        return AnalyzerResult(
            name=self.name,
            score=max(0.0, min(100.0, score)),
            details={
                "bpm": round(bpm, 1) if snr > 2.0 else None,
                "snr": round(snr, 2),
                "signal_strength": round(state.signal_strength, 3),
                "peak_freq_hz": round(peak_freq, 3),
                "frames": state.frame_count,
                "data_quality": round(data_quality, 2),
            },
            elapsed_ms=elapsed_ms,
        )

    def get_bpm(self, face_id: int) -> float | None:
        state = self._states.get(face_id)
        return state.estimated_bpm if state else None
