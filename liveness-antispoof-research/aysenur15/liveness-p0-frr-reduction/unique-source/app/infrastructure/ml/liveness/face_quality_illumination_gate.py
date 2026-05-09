"""Preview-only face illumination quality gate."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from app.domain.entities.face_landmarks import LandmarkResult
from app.infrastructure.ml.liveness.critical_region_visibility_gate import (
    _crop,
    _landmark_region_patch,
    _region_patch,
)

_QUALITY_REGIONS = {
    "left_eye": (0.14, 0.24, 0.24, 0.18),
    "right_eye": (0.62, 0.24, 0.24, 0.18),
    "nose": (0.39, 0.35, 0.22, 0.24),
    "mouth": (0.28, 0.63, 0.44, 0.16),
    "lower_face": (0.18, 0.56, 0.64, 0.34),
}
_UNDEREXPOSED_BRIGHTNESS = 52.0
_OVEREXPOSED_BRIGHTNESS = 218.0
_GLOBAL_BRIGHTNESS_LOW = 62.0
_GLOBAL_BRIGHTNESS_HIGH = 210.0
_QUALITY_SCORE_THRESHOLD = 0.58


@dataclass(frozen=True)
class FaceQualityIlluminationResult:
    quality_ok: bool
    quality_status: str
    quality_reason: str
    per_region_brightness: dict[str, float]
    brightness_uniformity: float
    illumination_score: float
    global_face_brightness: float
    shadow_asymmetry: float
    underexposed_regions: tuple[str, ...]
    overexposed_regions: tuple[str, ...]


class FaceQualityIlluminationGate:
    """Classify poor illumination separately from physical occlusion."""

    def evaluate(
        self,
        *,
        frame_bgr: np.ndarray,
        face_bounding_box: Optional[tuple[int, int, int, int]],
        landmarks: Optional[LandmarkResult] = None,
    ) -> FaceQualityIlluminationResult:
        if frame_bgr is None or frame_bgr.size == 0 or face_bounding_box is None:
            return FaceQualityIlluminationResult(
                quality_ok=False,
                quality_status="LOW_QUALITY",
                quality_reason="poor_face_illumination",
                per_region_brightness={},
                brightness_uniformity=0.0,
                illumination_score=0.0,
                global_face_brightness=0.0,
                shadow_asymmetry=0.0,
                underexposed_regions=(),
                overexposed_regions=(),
            )

        face_roi, clipped_bbox = _crop(frame_bgr, face_bounding_box)
        if face_roi.size == 0:
            return FaceQualityIlluminationResult(
                quality_ok=False,
                quality_status="LOW_QUALITY",
                quality_reason="poor_face_illumination",
                per_region_brightness={},
                brightness_uniformity=0.0,
                illumination_score=0.0,
                global_face_brightness=0.0,
                shadow_asymmetry=0.0,
                underexposed_regions=(),
                overexposed_regions=(),
            )

        face_gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        global_face_brightness = float(np.mean(face_gray))
        global_contrast = float(np.std(face_gray))
        left_half = face_gray[:, : max(1, face_gray.shape[1] // 2)]
        right_half = face_gray[:, max(1, face_gray.shape[1] // 2) :]
        left_brightness = float(np.mean(left_half))
        right_brightness = float(np.mean(right_half))
        shadow_asymmetry = abs(left_brightness - right_brightness) / max(global_face_brightness, 1.0)

        per_region_brightness: dict[str, float] = {}
        normalized_detail_scores: dict[str, float] = {}
        underexposed_regions: list[str] = []
        overexposed_regions: list[str] = []
        for region_name, ratios in _QUALITY_REGIONS.items():
            patch = self._resolve_region_patch(
                frame_bgr=frame_bgr,
                face_roi=face_roi,
                clipped_bbox=clipped_bbox,
                landmarks=landmarks,
                region_name=region_name,
                fallback_ratios=ratios,
            )
            if patch.size == 0:
                continue
            patch_gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
            brightness = float(np.mean(patch_gray))
            per_region_brightness[region_name] = brightness
            normalized_detail_scores[region_name] = _normalized_detail_score(patch_gray)
            if brightness < _UNDEREXPOSED_BRIGHTNESS:
                underexposed_regions.append(region_name)
            elif brightness > _OVEREXPOSED_BRIGHTNESS:
                overexposed_regions.append(region_name)

        brightness_values = list(per_region_brightness.values()) or [global_face_brightness]
        brightness_std = float(np.std(brightness_values))
        brightness_uniformity = max(0.0, min(1.0, 1.0 - brightness_std / 70.0))
        underexposed_ratio = len(underexposed_regions) / max(len(per_region_brightness), 1)
        overexposed_ratio = len(overexposed_regions) / max(len(per_region_brightness), 1)
        normalized_detail_mean = float(np.mean(list(normalized_detail_scores.values()) or [0.0]))

        brightness_score = _band_score(
            global_face_brightness,
            low=42.0,
            high=225.0,
            inner_low=72.0,
            inner_high=188.0,
        )
        contrast_score = max(0.0, min(1.0, global_contrast / 40.0))
        asymmetry_score = max(0.0, min(1.0, 1.0 - shadow_asymmetry / 0.40))
        exposure_score = max(0.0, min(1.0, 1.0 - 0.70 * underexposed_ratio - 0.45 * overexposed_ratio))
        illumination_score = max(
            0.0,
            min(
                1.0,
                0.34 * brightness_score
                + 0.20 * brightness_uniformity
                + 0.18 * exposure_score
                + 0.12 * asymmetry_score
                + 0.10 * contrast_score
                + 0.06 * normalized_detail_mean,
            ),
        )

        quality_reason = "face_quality_ok"
        quality_ok = True
        if (
            global_face_brightness < _GLOBAL_BRIGHTNESS_LOW
            or underexposed_ratio >= 0.40
            or (
                illumination_score < _QUALITY_SCORE_THRESHOLD
                and brightness_score < 0.55
                and normalized_detail_mean >= 0.28
            )
        ):
            quality_ok = False
            quality_reason = "poor_face_illumination"
        elif (
            global_face_brightness > _GLOBAL_BRIGHTNESS_HIGH
            or overexposed_ratio >= 0.35
            or shadow_asymmetry >= 0.26
            or brightness_uniformity < 0.44
        ):
            quality_ok = False
            quality_reason = "uneven_face_lighting"
        elif illumination_score < _QUALITY_SCORE_THRESHOLD:
            quality_ok = False
            quality_reason = "poor_face_illumination"

        return FaceQualityIlluminationResult(
            quality_ok=quality_ok,
            quality_status="OK" if quality_ok else "LOW_QUALITY",
            quality_reason=quality_reason,
            per_region_brightness=per_region_brightness,
            brightness_uniformity=brightness_uniformity,
            illumination_score=illumination_score,
            global_face_brightness=global_face_brightness,
            shadow_asymmetry=shadow_asymmetry,
            underexposed_regions=tuple(underexposed_regions),
            overexposed_regions=tuple(overexposed_regions),
        )

    @staticmethod
    def _resolve_region_patch(
        *,
        frame_bgr: np.ndarray,
        face_roi: np.ndarray,
        clipped_bbox: tuple[int, int, int, int],
        landmarks: Optional[LandmarkResult],
        region_name: str,
        fallback_ratios: tuple[float, float, float, float],
    ) -> np.ndarray:
        if landmarks is not None:
            patch = _landmark_region_patch(
                frame_bgr=frame_bgr,
                clipped_bbox=clipped_bbox,
                landmarks=landmarks,
                region_name=region_name,
            )
            if patch.size > 0:
                return patch
        return _region_patch(face_roi, fallback_ratios)


def _normalized_detail_score(gray_patch: np.ndarray) -> float:
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    normalized = clahe.apply(gray_patch)
    texture = float(cv2.Laplacian(normalized, cv2.CV_64F).var())
    edges = cv2.Canny(normalized, 40, 120)
    edge_density = float(np.count_nonzero(edges)) / max(edges.size, 1)
    contrast = float(np.std(normalized))
    return max(
        0.0,
        min(
            1.0,
            0.45 * min(texture / 140.0, 1.0)
            + 0.30 * min(edge_density / 0.16, 1.0)
            + 0.25 * min(contrast / 38.0, 1.0),
        ),
    )


def _band_score(value: float, *, low: float, high: float, inner_low: float, inner_high: float) -> float:
    if low <= value <= high:
        if inner_low <= value <= inner_high:
            return 1.0
        if value < inner_low:
            return max(0.0, min(1.0, (value - low) / max(inner_low - low, 1e-6)))
        return max(0.0, min(1.0, (high - value) / max(high - inner_high, 1e-6)))
    return 0.0
