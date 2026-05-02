"""Moire pattern analyzer for screen-based attack detection.

Ported from biometric-processor's moire_pattern_analysis.py.
Uses Gabor filter bank + FFT periodicity to detect screen moire patterns.
"""

import time

import cv2
import numpy as np

from src.domain.models import FaceROI, AnalyzerResult


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _normalize(value: float, low: float, high: float) -> float:
    if high <= low:
        return 0.0
    return _clamp01((float(value) - low) / (high - low))


class MoireAnalyzer:
    """Moire pattern detector using Gabor filters and FFT.

    Detects periodic interference patterns produced when a camera
    photographs a screen display. Effective against:
    - Video replay on LCD/OLED screens
    - Static image displayed on screen
    - Deepfake injection via screen capture

    Performance: ~5ms per face on CPU.
    """

    GABOR_THETAS = (0.0, np.pi / 4, np.pi / 2, 3 * np.pi / 4)

    def __init__(self, response_std_threshold: float = 45.0):
        self._response_threshold = response_std_threshold
        self._kernels = [
            cv2.getGaborKernel(
                ksize=(21, 21), sigma=5.0, theta=theta,
                lambd=10.0, gamma=0.5, psi=0,
            )
            for theta in self.GABOR_THETAS
        ]
        self._clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    @property
    def name(self) -> str:
        return "moire"

    def analyze(self, face_crop: np.ndarray, face_roi: FaceROI) -> AnalyzerResult:
        start = time.perf_counter()
        gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)

        # Downsample large crops for performance (Gabor is O(n^2))
        h, w = gray.shape[:2]
        max_side = 160
        if max(h, w) > max_side:
            scale = max_side / max(h, w)
            gray = cv2.resize(gray, (max(16, int(w * scale)), max(16, int(h * scale))),
                              interpolation=cv2.INTER_AREA)
            h, w = gray.shape[:2]

        # Center focus crop (72% of image)
        ch, cw = max(16, int(h * 0.72)), max(16, int(w * 0.72))
        y1, x1 = (h - ch) // 2, (w - cw) // 2
        focus = gray[y1:y1 + ch, x1:x1 + cw]
        focus = self._clahe.apply(focus)

        # Gabor filter bank analysis
        response_stds = []
        strong_count = 0
        for kernel in self._kernels:
            filtered = cv2.filter2D(focus, cv2.CV_64F, kernel)
            std = float(np.std(filtered))
            response_stds.append(std)
            if std > self._response_threshold:
                strong_count += 1

        response_fraction = strong_count / max(len(self._kernels), 1)
        std_mean = float(np.mean(response_stds)) if response_stds else 0.0
        std_max = float(np.max(response_stds)) if response_stds else 0.0
        std_min = float(np.min(response_stds)) if response_stds else 0.0
        std_range = std_max - std_min
        std_std = float(np.std(response_stds)) if response_stds else 0.0

        # Gabor strength
        excess = [_clamp01((s - self._response_threshold) / max(self._response_threshold, 1e-6)) for s in response_stds]
        gabor_strength = float(np.mean(excess))
        orientation_sel = _clamp01(std_range / max(std_max, 1e-6)) if std_max > 1e-6 else 0.0
        gabor_risk = _clamp01(gabor_strength * (0.35 + 0.65 * orientation_sel))

        # FFT periodicity
        fft_risk = self._fft_periodicity(focus)

        # Combined risk
        moire_risk = _clamp01(
            0.45 * gabor_risk
            + 0.30 * fft_risk
            + 0.15 * response_fraction
            + 0.10 * _clamp01(std_std / max(self._response_threshold, 1e-6))
        )
        score = 100.0 * (1.0 - moire_risk)
        elapsed_ms = (time.perf_counter() - start) * 1000

        return AnalyzerResult(
            name=self.name,
            score=max(0.0, min(100.0, score)),
            details={
                "moire_risk": moire_risk,
                "gabor_risk": gabor_risk,
                "fft_risk": fft_risk,
                "response_fraction": response_fraction,
                "std_mean": std_mean,
            },
            elapsed_ms=elapsed_ms,
        )

    def _fft_periodicity(self, gray: np.ndarray) -> float:
        h, w = gray.shape[:2]
        if max(h, w) > 256:
            scale = 256.0 / max(h, w)
            gray = cv2.resize(gray, (max(1, int(w * scale)), max(1, int(h * scale))),
                              interpolation=cv2.INTER_AREA)

        spectrum = np.fft.fftshift(np.fft.fft2(gray.astype(np.float32)))
        magnitude = np.log1p(np.abs(spectrum))
        h2, w2 = magnitude.shape[:2]
        cy, cx = h2 / 2.0, w2 / 2.0
        yy, xx = np.ogrid[:h2, :w2]
        radius = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
        half = min(h2, w2) / 2.0
        low_r = max(2.0, half / 14.0)
        mid_r = max(low_r + 1.0, half / 4.2)

        low_mask = radius <= low_r
        mid_mask = (radius > low_r) & (radius <= mid_r)
        low_e = float(np.mean(magnitude[low_mask])) if np.any(low_mask) else 0.0
        mid_e = float(np.mean(magnitude[mid_mask])) if np.any(mid_mask) else 0.0
        peak_e = float(np.max(magnitude[mid_mask])) if np.any(mid_mask) else 0.0

        ratio = mid_e / max(low_e, 1e-6)
        peak_ratio = peak_e / max(mid_e, 1e-6)
        return _clamp01(
            0.65 * _normalize(ratio, 0.82, 1.18)
            + 0.35 * _normalize(peak_ratio, 1.55, 2.80)
        )
