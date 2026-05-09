"""Fast screen-replay anti-spoofing heuristics for phone/tablet attacks."""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from app.infrastructure.ml.liveness.moire_pattern_analysis import analyze_moire_pattern


@dataclass(frozen=True)
class ScreenReplayAssessment:
    """Aggregated screen replay anti-spoof result."""

    spoof_score: float
    confident: bool
    hard_veto: bool
    low_signal_count: int
    screen_like_pattern: bool = False
    screen_like_optics: bool = False
    signal_scores: dict[str, float] = field(default_factory=dict)
    details: dict[str, float] = field(default_factory=dict)

    @property
    def is_live_like(self) -> bool:
        return self.spoof_score >= 70.0 and not self.hard_veto


class ScreenReplayAntiSpoof:
    """Cheap layered heuristics for detecting display-based replay attacks."""

    def __init__(
        self,
        *,
        fft_ratio_center: float = 0.85,
        fft_ratio_width: float = 0.20,
        laplacian_low: float = 80.0,
        laplacian_high: float = 700.0,
        specular_warn: float = 0.020,
        specular_fail: float = 0.060,
        skin_coverage_min: float = 0.20,
        skin_coverage_max: float = 0.95,
        crcb_scatter_min: float = 2.5,
        veto_signal_count: int = 2,
        veto_signal_threshold: float = 30.0,
        veto_score_cap: float = 35.0,
        blur_floor: float = 25.0,
        max_side: int = 256,
    ) -> None:
        self._fft_ratio_center = fft_ratio_center
        self._fft_ratio_width = max(fft_ratio_width, 1e-3)
        self._laplacian_low = laplacian_low
        self._laplacian_high = laplacian_high
        self._specular_warn = specular_warn
        self._specular_fail = max(specular_fail, specular_warn + 1e-3)
        self._skin_coverage_min = skin_coverage_min
        self._skin_coverage_max = skin_coverage_max
        self._crcb_scatter_min = crcb_scatter_min
        self._veto_signal_count = max(1, veto_signal_count)
        self._veto_signal_threshold = veto_signal_threshold
        self._veto_score_cap = veto_score_cap
        self._blur_floor = blur_floor
        self._max_side = max(64, max_side)
        self._clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    def check_spoofing(self, image: np.ndarray) -> ScreenReplayAssessment:
        """Return a live-like score where lower values indicate screen replay risk."""
        if image is None or image.size == 0:
            return ScreenReplayAssessment(
                spoof_score=50.0,
                confident=False,
                hard_veto=False,
                low_signal_count=0,
                screen_like_pattern=False,
                screen_like_optics=False,
                signal_scores={},
                details={"invalid_input": 1.0},
            )

        bgr_small = self._resize_bgr(image)
        gray = cv2.cvtColor(bgr_small, cv2.COLOR_BGR2GRAY)
        gray = self._clahe.apply(gray)

        laplacian_variance = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        if laplacian_variance < self._blur_floor:
            return ScreenReplayAssessment(
                spoof_score=50.0,
                confident=False,
                hard_veto=False,
                low_signal_count=0,
                screen_like_pattern=False,
                screen_like_optics=False,
                signal_scores={"laplacian": 50.0},
                details={
                    "blur_floor_triggered": 1.0,
                    "laplacian_variance": laplacian_variance,
                },
            )

        fft_score, fft_details = self._compute_fft_score(gray)
        gabor_analysis = analyze_moire_pattern(gray)
        gabor_score = float(gabor_analysis["moire_score"])
        laplacian_score = self._compute_laplacian_score(laplacian_variance)
        skin_score, skin_details = self._compute_skin_score(bgr_small)
        specular_score, specular_details = self._compute_specular_score(bgr_small)

        signal_scores = {
            "fft": fft_score,
            "gabor": gabor_score,
            "laplacian": laplacian_score,
            "skin": skin_score,
            "specular": specular_score,
        }
        spoof_score = self._fuse_scores(signal_scores)
        low_signal_count = sum(1 for score in signal_scores.values() if score < self._veto_signal_threshold)
        screen_like_pattern = signal_scores["fft"] < 24.0 and signal_scores["gabor"] < 35.0
        screen_like_optics = signal_scores["specular"] < 35.0 or signal_scores["laplacian"] < 25.0
        hard_veto = (
            spoof_score <= 22.0
            and screen_like_pattern
            and screen_like_optics
            and low_signal_count >= self._veto_signal_count
        )
        if hard_veto:
            spoof_score = min(spoof_score, self._veto_score_cap)

        details = {
            **fft_details,
            **{f"gabor_{key}": float(value) for key, value in gabor_analysis.items() if not isinstance(value, list)},
            **skin_details,
            **specular_details,
            "laplacian_variance": laplacian_variance,
            "screen_replay_spoof_score": spoof_score,
            "screen_replay_low_signal_count": float(low_signal_count),
            "screen_replay_screen_like_pattern": float(screen_like_pattern),
            "screen_replay_screen_like_optics": float(screen_like_optics),
        }
        for name, score in signal_scores.items():
            details[f"screen_replay_{name}_score"] = float(score)

        return ScreenReplayAssessment(
            spoof_score=spoof_score,
            confident=True,
            hard_veto=hard_veto,
            low_signal_count=low_signal_count,
            screen_like_pattern=screen_like_pattern,
            screen_like_optics=screen_like_optics,
            signal_scores=signal_scores,
            details=details,
        )

    def _resize_bgr(self, image: np.ndarray) -> np.ndarray:
        h, w = image.shape[:2]
        max_dim = max(h, w)
        if max_dim <= self._max_side:
            return image
        scale = self._max_side / float(max_dim)
        size = (max(1, int(round(w * scale))), max(1, int(round(h * scale))))
        return cv2.resize(image, size, interpolation=cv2.INTER_AREA)

    def _compute_fft_score(self, gray: np.ndarray) -> tuple[float, dict[str, float]]:
        spectrum = np.fft.fftshift(np.fft.fft2(gray.astype(np.float32)))
        magnitude = np.log1p(np.abs(spectrum))
        h, w = magnitude.shape[:2]
        cy, cx = h / 2.0, w / 2.0
        yy, xx = np.ogrid[:h, :w]
        radius = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
        half_side = min(h, w) / 2.0
        low_radius = max(2.0, half_side / 16.0)
        mid_radius = max(low_radius + 1.0, half_side / 4.0)

        low_mask = radius <= low_radius
        mid_mask = (radius > low_radius) & (radius <= mid_radius)

        low_energy = float(np.mean(magnitude[low_mask])) if np.any(low_mask) else 0.0
        mid_energy = float(np.mean(magnitude[mid_mask])) if np.any(mid_mask) else 0.0
        mid_low_ratio = mid_energy / max(low_energy, 1e-6)
        risk = _sigmoid((mid_low_ratio - self._fft_ratio_center) / self._fft_ratio_width)
        score = 100.0 * (1.0 - risk)
        return _clamp_score(score), {
            "screen_replay_fft_low_energy": low_energy,
            "screen_replay_fft_mid_energy": mid_energy,
            "screen_replay_fft_mid_low_ratio": mid_low_ratio,
        }

    def _compute_laplacian_score(self, laplacian_variance: float) -> float:
        low_risk = 1.0 - _sigmoid((laplacian_variance - self._laplacian_low) / max(self._laplacian_low * 0.20, 10.0))
        high_risk = _sigmoid((laplacian_variance - self._laplacian_high) / max(self._laplacian_high * 0.20, 40.0))
        risk = max(low_risk, high_risk)
        return _clamp_score(100.0 * (1.0 - risk))

    def _compute_skin_score(self, image: np.ndarray) -> tuple[float, dict[str, float]]:
        ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        y = ycrcb[:, :, 0]
        cr = ycrcb[:, :, 1]
        cb = ycrcb[:, :, 2]
        h = hsv[:, :, 0]
        s = hsv[:, :, 1]
        v = hsv[:, :, 2]

        ycrcb_mask = (cr >= 133) & (cr <= 173) & (cb >= 77) & (cb <= 127) & (y >= 30)
        hsv_mask = ((h <= 25) | (h >= 160)) & (s >= 30) & (s <= 180) & (v >= 40)
        skin_mask = ycrcb_mask & hsv_mask

        skin_coverage = float(np.mean(skin_mask))
        if np.any(skin_mask):
            cr_std = float(np.std(cr[skin_mask]))
            cb_std = float(np.std(cb[skin_mask]))
        else:
            cr_std = 0.0
            cb_std = 0.0

        low_coverage_risk = 1.0 - _normalize(skin_coverage, self._skin_coverage_min, 0.35)
        high_coverage_risk = _normalize(skin_coverage, 0.85, self._skin_coverage_max)
        scatter = min(cr_std, cb_std)
        scatter_risk = 1.0 - _normalize(scatter, self._crcb_scatter_min, self._crcb_scatter_min + 4.0)
        risk = _clamp01(0.40 * max(low_coverage_risk, high_coverage_risk) + 0.60 * scatter_risk)
        return _clamp_score(100.0 * (1.0 - risk)), {
            "screen_replay_skin_coverage": skin_coverage,
            "screen_replay_cr_std": cr_std,
            "screen_replay_cb_std": cb_std,
        }

    def _compute_specular_score(self, image: np.ndarray) -> tuple[float, dict[str, float]]:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        saturation = hsv[:, :, 1].astype(np.float32)
        value = hsv[:, :, 2].astype(np.float32)
        bright_low_sat = (value >= 240.0) & (saturation <= 35.0)
        specular_ratio = float(np.mean(bright_low_sat))
        risk = _normalize(specular_ratio, self._specular_warn, self._specular_fail)
        score = 100.0 * (1.0 - risk)
        return _clamp_score(score), {
            "screen_replay_specular_ratio": specular_ratio,
        }

    def _fuse_scores(self, signal_scores: dict[str, float]) -> float:
        weights = {
            "fft": 0.30,
            "gabor": 0.20,
            "laplacian": 0.20,
            "skin": 0.15,
            "specular": 0.15,
        }
        weighted_mean = sum(weights[name] * signal_scores[name] for name in weights)
        penalty = min(signal_scores.values()) if signal_scores else 50.0
        return _clamp_score(0.65 * weighted_mean + 0.35 * penalty)


def _normalize(value: float, low: float, high: float) -> float:
    if high <= low:
        return 0.0
    return _clamp01((float(value) - low) / (high - low))


def _sigmoid(value: float) -> float:
    return 1.0 / (1.0 + float(np.exp(-value)))


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _clamp_score(value: float) -> float:
    return max(0.0, min(100.0, float(value)))
