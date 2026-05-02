"""Texture-based liveness analyzer.

Ported from biometric-processor's OptimizedTextureLivenessDetector.
Uses Laplacian variance, color distribution, and frequency analysis.
"""

import time

import cv2
import numpy as np

from src.domain.models import FaceROI, AnalyzerResult


class TextureAnalyzer:
    """Passive liveness via texture analysis.

    Detects printed photos and screen displays through:
    - Laplacian variance (sharp vs blurry texture)
    - Color distribution naturalness (HSV analysis)
    - Frequency domain ratio (FFT high/low energy)

    Performance: ~5ms per face on CPU.
    """

    def __init__(
        self,
        texture_threshold: float = 100.0,
        color_threshold: float = 0.3,
        frequency_threshold: float = 0.5,
        fft_downsample: tuple[int, int] = (192, 108),
    ):
        self._texture_threshold = texture_threshold
        self._color_threshold = color_threshold
        self._frequency_threshold = frequency_threshold
        self._fft_downsample = fft_downsample
        self._weights = {"texture": 0.40, "color": 0.30, "frequency": 0.30}

    @property
    def name(self) -> str:
        return "texture"

    def analyze(self, face_crop: np.ndarray, face_roi: FaceROI) -> AnalyzerResult:
        start = time.perf_counter()

        gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(face_crop, cv2.COLOR_BGR2HSV)
        gray_small = cv2.resize(gray, self._fft_downsample, interpolation=cv2.INTER_AREA)

        texture_score = self._texture_score(gray)
        color_score = self._color_score(hsv)
        frequency_score = self._frequency_score(gray_small)

        combined = (
            texture_score * self._weights["texture"]
            + color_score * self._weights["color"]
            + frequency_score * self._weights["frequency"]
        )
        score = max(0.0, min(100.0, combined))
        elapsed_ms = (time.perf_counter() - start) * 1000

        return AnalyzerResult(
            name=self.name,
            score=score,
            details={
                "texture_score": texture_score,
                "color_score": color_score,
                "frequency_score": frequency_score,
            },
            elapsed_ms=elapsed_ms,
        )

    def _texture_score(self, gray: np.ndarray) -> float:
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        variance = float(laplacian.var())
        if variance >= self._texture_threshold:
            return min(100.0, 50.0 + (variance - self._texture_threshold) * 0.2)
        return max(0.0, (variance / self._texture_threshold) * 50.0)

    def _color_score(self, hsv: np.ndarray) -> float:
        saturation = hsv[:, :, 1]
        value = hsv[:, :, 2]
        sat_deviation = abs(float(np.mean(saturation)) - 80) / 128.0
        val_deviation = abs(float(np.std(value)) - 50) / 64.0
        combined = (sat_deviation + val_deviation) / 2.0
        if combined <= self._color_threshold:
            return 100.0 - (combined / self._color_threshold) * 30.0
        return max(0.0, 70.0 - (combined - self._color_threshold) * 100.0)

    def _frequency_score(self, gray_small: np.ndarray) -> float:
        f_shift = np.fft.fftshift(np.fft.fft2(gray_small))
        magnitude = np.abs(f_shift)
        rows, cols = gray_small.shape
        cr, cc = rows // 2, cols // 2

        low_region = magnitude[cr - rows // 8: cr + rows // 8, cc - cols // 8: cc + cols // 8]
        high_mask = np.ones_like(magnitude, dtype=bool)
        high_mask[cr - rows // 4: cr + rows // 4, cc - cols // 4: cc + cols // 4] = False

        low_mean = float(np.mean(low_region)) + 1e-6
        high_mean = float(np.mean(magnitude[high_mask])) + 1e-6
        ratio = high_mean / low_mean

        if ratio < self._frequency_threshold:
            return 100.0 - (1.0 - ratio / self._frequency_threshold) * 40.0
        if ratio > self._frequency_threshold * 2:
            return max(0.0, 60.0 - (ratio - self._frequency_threshold * 2) * 50.0)
        return 80.0
