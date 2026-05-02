"""Screen replay anti-spoofing analyzer.

Ported from biometric-processor's screen_replay_anti_spoof.py.
Detects display-based replay attacks through layered heuristics.
"""

import time
import math

import cv2
import numpy as np

from src.domain.models import AnalyzerResult


def _clamp01(v: float) -> float:
    return max(0.0, min(1.0, float(v)))


def _sigmoid(v: float) -> float:
    return 1.0 / (1.0 + math.exp(-v))


def _normalize(value: float, low: float, high: float) -> float:
    if high <= low:
        return 0.0
    return _clamp01((float(value) - low) / (high - low))


class ScreenReplayAnalyzer:
    """Whole-frame screen replay detector.

    Analyzes the full frame (not cropped face) for signs of
    a display being held in front of the camera:
    - FFT energy ratio (screen vs natural lighting)
    - Laplacian variance (screen sharpness profile)
    - Skin color naturalness (YCrCb + HSV)
    - Specular highlights (screen glare)

    Performance: ~8ms on CPU.
    """

    def __init__(self):
        self._clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    @property
    def name(self) -> str:
        return "screen_replay"

    def analyze(self, frame: np.ndarray) -> AnalyzerResult:
        start = time.perf_counter()

        # Resize for consistent analysis
        small = self._resize(frame, max_side=256)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        gray = self._clahe.apply(gray)

        laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        if laplacian_var < 25.0:
            elapsed_ms = (time.perf_counter() - start) * 1000
            return AnalyzerResult(
                name=self.name, score=50.0,
                details={"blur_floor": True, "laplacian_var": laplacian_var},
                elapsed_ms=elapsed_ms,
            )

        fft_score = self._fft_score(gray)
        lap_score = self._laplacian_score(laplacian_var)
        skin_score = self._skin_score(small)
        specular_score = self._specular_score(small)

        signals = {"fft": fft_score, "laplacian": lap_score,
                    "skin": skin_score, "specular": specular_score}

        # Weighted fusion with min-penalty
        weights = {"fft": 0.35, "laplacian": 0.25, "skin": 0.20, "specular": 0.20}
        weighted = sum(weights[k] * signals[k] for k in weights)
        penalty = min(signals.values())
        score = max(0.0, min(100.0, 0.65 * weighted + 0.35 * penalty))

        elapsed_ms = (time.perf_counter() - start) * 1000
        return AnalyzerResult(
            name=self.name, score=score,
            details={
                "fft_score": fft_score,
                "laplacian_score": lap_score,
                "laplacian_var": laplacian_var,
                "skin_score": skin_score,
                "specular_score": specular_score,
            },
            elapsed_ms=elapsed_ms,
        )

    def _resize(self, img: np.ndarray, max_side: int = 256) -> np.ndarray:
        h, w = img.shape[:2]
        if max(h, w) <= max_side:
            return img
        scale = max_side / max(h, w)
        return cv2.resize(img, (max(1, int(w * scale)), max(1, int(h * scale))),
                          interpolation=cv2.INTER_AREA)

    def _fft_score(self, gray: np.ndarray) -> float:
        spectrum = np.fft.fftshift(np.fft.fft2(gray.astype(np.float32)))
        magnitude = np.log1p(np.abs(spectrum))
        h, w = magnitude.shape[:2]
        cy, cx = h / 2.0, w / 2.0
        yy, xx = np.ogrid[:h, :w]
        radius = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
        half = min(h, w) / 2.0
        low_r = max(2.0, half / 16.0)
        mid_r = max(low_r + 1.0, half / 4.0)

        low_mask = radius <= low_r
        mid_mask = (radius > low_r) & (radius <= mid_r)
        low_e = float(np.mean(magnitude[low_mask])) if np.any(low_mask) else 0.0
        mid_e = float(np.mean(magnitude[mid_mask])) if np.any(mid_mask) else 0.0
        ratio = mid_e / max(low_e, 1e-6)
        risk = _sigmoid((ratio - 0.85) / 0.20)
        return max(0.0, min(100.0, 100.0 * (1.0 - risk)))

    def _laplacian_score(self, variance: float) -> float:
        low_risk = 1.0 - _sigmoid((variance - 80.0) / 16.0)
        # High threshold raised: modern 720p webcams legitimately produce
        # Laplacian variance of 800-1500. Only flag extreme values (>2000).
        high_risk = _sigmoid((variance - 2000.0) / 400.0)
        risk = max(low_risk, high_risk)
        return max(0.0, min(100.0, 100.0 * (1.0 - risk)))

    def _skin_score(self, img: np.ndarray) -> float:
        ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        y, cr, cb = ycrcb[:, :, 0], ycrcb[:, :, 1], ycrcb[:, :, 2]
        h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

        skin = ((cr >= 133) & (cr <= 173) & (cb >= 77) & (cb <= 127) & (y >= 30)
                & ((h <= 25) | (h >= 160)) & (s >= 30) & (s <= 180) & (v >= 40))
        coverage = float(np.mean(skin))
        if np.any(skin):
            scatter = min(float(np.std(cr[skin])), float(np.std(cb[skin])))
        else:
            scatter = 0.0

        low_cov_risk = 1.0 - _normalize(coverage, 0.20, 0.35)
        high_cov_risk = _normalize(coverage, 0.85, 0.95)
        scatter_risk = 1.0 - _normalize(scatter, 2.5, 6.5)
        risk = _clamp01(0.40 * max(low_cov_risk, high_cov_risk) + 0.60 * scatter_risk)
        return max(0.0, min(100.0, 100.0 * (1.0 - risk)))

    def _specular_score(self, img: np.ndarray) -> float:
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        bright_low_sat = (hsv[:, :, 2].astype(np.float32) >= 240.0) & (hsv[:, :, 1].astype(np.float32) <= 35.0)
        ratio = float(np.mean(bright_low_sat))
        risk = _normalize(ratio, 0.020, 0.060)
        return max(0.0, min(100.0, 100.0 * (1.0 - risk)))
