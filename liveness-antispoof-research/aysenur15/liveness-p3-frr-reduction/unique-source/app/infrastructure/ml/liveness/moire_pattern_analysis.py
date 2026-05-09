"""Shared moire-pattern analysis for screen replay and texture liveness checks."""

from __future__ import annotations

from typing import Sequence

import cv2
import numpy as np

DEFAULT_GABOR_KSIZE = (21, 21)
DEFAULT_GABOR_SIGMA = 5.0
DEFAULT_GABOR_LAMBDA = 10.0
DEFAULT_GABOR_GAMMA = 0.5
DEFAULT_GABOR_PSI = 0
DEFAULT_GABOR_THETAS = (0.0, np.pi / 4, np.pi / 2, 3 * np.pi / 4)
DEFAULT_RESPONSE_STD_THRESHOLD = 30.0
DEFAULT_CENTER_FOCUS_RATIO = 0.72


def build_default_moire_gabor_kernels() -> list[np.ndarray]:
    """Create the default Gabor bank used by the texture detectors."""
    return [
        cv2.getGaborKernel(
            ksize=DEFAULT_GABOR_KSIZE,
            sigma=DEFAULT_GABOR_SIGMA,
            theta=theta,
            lambd=DEFAULT_GABOR_LAMBDA,
            gamma=DEFAULT_GABOR_GAMMA,
            psi=DEFAULT_GABOR_PSI,
        )
        for theta in DEFAULT_GABOR_THETAS
    ]


def analyze_moire_pattern(
    gray: np.ndarray,
    *,
    gabor_kernels: Sequence[np.ndarray] | None = None,
    response_std_threshold: float = DEFAULT_RESPONSE_STD_THRESHOLD,
) -> dict[str, float | list[float]]:
    """Analyze periodic moire-like responses in a grayscale image.

    Returns normalized risk in [0,1] plus raw detector diagnostics.
    """
    kernels = list(gabor_kernels) if gabor_kernels is not None else build_default_moire_gabor_kernels()
    if gray is None or gray.size == 0 or not kernels:
        return {
            "moire_risk": 0.0,
            "moire_score": 100.0,
            "moire_response_count": 0.0,
            "moire_response_fraction": 0.0,
            "moire_response_std_mean": 0.0,
            "moire_response_std_max": 0.0,
            "moire_response_stds": [],
        }

    focus_gray, focus_ratio = _extract_center_focus(gray)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    normalized_gray = clahe.apply(focus_gray)

    response_stds: list[float] = []
    strong_response_count = 0
    for kernel in kernels:
        filtered = cv2.filter2D(normalized_gray, cv2.CV_64F, kernel)
        response_std = float(np.std(filtered))
        response_stds.append(response_std)
        if response_std > response_std_threshold:
            strong_response_count += 1

    response_fraction = strong_response_count / max(len(kernels), 1)
    response_std_mean = float(np.mean(response_stds)) if response_stds else 0.0
    response_std_max = float(np.max(response_stds)) if response_stds else 0.0
    response_std_min = float(np.min(response_stds)) if response_stds else 0.0
    response_std_range = max(0.0, response_std_max - response_std_min)
    response_std_std = float(np.std(response_stds)) if response_stds else 0.0
    normalized_excess = [_clamp01((std - response_std_threshold) / max(response_std_threshold, 1e-6)) for std in response_stds]
    gabor_strength = float(np.mean(normalized_excess)) if normalized_excess else 0.0
    orientation_selectivity = 0.0
    if response_std_max > 1e-6:
        orientation_selectivity = _clamp01(response_std_range / response_std_max)
    periodic_gabor_risk = _clamp01(gabor_strength * (0.35 + 0.65 * orientation_selectivity))

    fft_details = _compute_fft_periodicity(normalized_gray)
    fft_risk = float(fft_details["moire_fft_risk"])
    moire_risk = _clamp01(
        0.45 * periodic_gabor_risk
        + 0.30 * fft_risk
        + 0.15 * response_fraction
        + 0.10 * _clamp01(response_std_std / max(response_std_threshold, 1e-6))
    )
    moire_score = 100.0 * (1.0 - moire_risk)

    return {
        "moire_risk": moire_risk,
        "moire_score": moire_score,
        "moire_response_count": float(strong_response_count),
        "moire_response_fraction": response_fraction,
        "moire_response_std_mean": response_std_mean,
        "moire_response_std_max": response_std_max,
        "moire_response_std_min": response_std_min,
        "moire_response_std_range": response_std_range,
        "moire_response_std_std": response_std_std,
        "moire_gabor_strength": gabor_strength,
        "moire_orientation_selectivity": orientation_selectivity,
        "moire_periodic_gabor_risk": periodic_gabor_risk,
        "moire_center_focus_ratio": focus_ratio,
        "moire_response_stds": response_stds,
        **fft_details,
    }


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _extract_center_focus(gray: np.ndarray, focus_ratio: float = DEFAULT_CENTER_FOCUS_RATIO) -> tuple[np.ndarray, float]:
    if gray is None or gray.size == 0:
        return gray, focus_ratio
    h, w = gray.shape[:2]
    crop_h = max(16, int(round(h * focus_ratio)))
    crop_w = max(16, int(round(w * focus_ratio)))
    y1 = max(0, (h - crop_h) // 2)
    x1 = max(0, (w - crop_w) // 2)
    y2 = min(h, y1 + crop_h)
    x2 = min(w, x1 + crop_w)
    return gray[y1:y2, x1:x2], focus_ratio


def _compute_fft_periodicity(gray: np.ndarray) -> dict[str, float]:
    if gray is None or gray.size == 0:
        return {
            "moire_fft_mid_low_ratio": 0.0,
            "moire_fft_peak_ratio": 0.0,
            "moire_fft_risk": 0.0,
        }

    resized = gray
    h, w = gray.shape[:2]
    max_side = max(h, w)
    if max_side > 256:
        scale = 256.0 / float(max_side)
        resized = cv2.resize(
            gray,
            (max(1, int(round(w * scale))), max(1, int(round(h * scale)))),
            interpolation=cv2.INTER_AREA,
        )

    spectrum = np.fft.fftshift(np.fft.fft2(resized.astype(np.float32)))
    magnitude = np.log1p(np.abs(spectrum))
    h2, w2 = magnitude.shape[:2]
    cy, cx = h2 / 2.0, w2 / 2.0
    yy, xx = np.ogrid[:h2, :w2]
    radius = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    half_side = min(h2, w2) / 2.0
    low_radius = max(2.0, half_side / 14.0)
    mid_radius = max(low_radius + 1.0, half_side / 4.2)

    low_mask = radius <= low_radius
    mid_mask = (radius > low_radius) & (radius <= mid_radius)
    low_energy = float(np.mean(magnitude[low_mask])) if np.any(low_mask) else 0.0
    mid_energy = float(np.mean(magnitude[mid_mask])) if np.any(mid_mask) else 0.0
    peak_energy = float(np.max(magnitude[mid_mask])) if np.any(mid_mask) else 0.0

    mid_low_ratio = mid_energy / max(low_energy, 1e-6)
    peak_ratio = peak_energy / max(mid_energy, 1e-6)
    fft_risk = _clamp01(
        0.65 * _normalize(mid_low_ratio, 0.82, 1.18)
        + 0.35 * _normalize(peak_ratio, 1.55, 2.80)
    )
    return {
        "moire_fft_mid_low_ratio": mid_low_ratio,
        "moire_fft_peak_ratio": peak_ratio,
        "moire_fft_risk": fft_risk,
    }


def _normalize(value: float, low: float, high: float) -> float:
    if high <= low:
        return 0.0
    return _clamp01((float(value) - low) / (high - low))
