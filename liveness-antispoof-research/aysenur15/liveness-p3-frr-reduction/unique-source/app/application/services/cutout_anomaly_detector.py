"""Heuristic cutout / focal-blur anomaly detector for spoof support."""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np


@dataclass(frozen=True)
class CutoutAnomalyAssessment:
    hole_cutout_risk: float
    focal_blur_anomaly_risk: float
    cutout_spoof_support: float
    details: dict[str, float] = field(default_factory=dict)


class CutoutAnomalyDetector:
    """Detect cutout-like and focal inconsistency anomalies inside a face crop."""

    REGION_SPECS = {
        "left_eye": (0.16, 0.21, 0.24, 0.18),
        "right_eye": (0.60, 0.21, 0.24, 0.18),
        "mouth": (0.28, 0.59, 0.44, 0.19),
    }

    def analyze(self, face_region_bgr: np.ndarray) -> CutoutAnomalyAssessment:
        if face_region_bgr is None or face_region_bgr.size == 0:
            return CutoutAnomalyAssessment(0.0, 0.0, 0.0, {})

        height, width = face_region_bgr.shape[:2]
        if min(height, width) < 48:
            return CutoutAnomalyAssessment(
                0.0,
                0.0,
                0.0,
                {"cutout_min_size_blocked": 1.0},
            )

        gray = cv2.cvtColor(face_region_bgr, cv2.COLOR_BGR2GRAY)
        region_results: list[dict[str, float]] = []
        region_sharpness_logs: list[float] = []

        for name, spec in self.REGION_SPECS.items():
            metrics = self._analyze_region(gray, width, height, spec)
            region_results.append({"name": name, **metrics})
            region_sharpness_logs.append(metrics["region_sharpness_log"])

        if not region_results:
            return CutoutAnomalyAssessment(0.0, 0.0, 0.0, {})

        hole_cutout_risk = float(np.mean([item["hole_risk"] for item in region_results]))
        local_focus_jump = float(np.mean([item["focus_jump_risk"] for item in region_results]))
        local_boundary_risk = float(np.mean([item["boundary_edge_risk"] for item in region_results]))
        local_blur_inconsistency = _normalize(float(np.std(region_sharpness_logs)), 0.10, 0.60)
        focal_blur_anomaly_risk = _clamp01(
            0.50 * local_focus_jump
            + 0.25 * local_blur_inconsistency
            + 0.25 * local_boundary_risk
        )
        cutout_spoof_support = _clamp01(
            0.55 * hole_cutout_risk
            + 0.45 * focal_blur_anomaly_risk
        )

        details: dict[str, float] = {
            "hole_cutout_risk": hole_cutout_risk,
            "focal_blur_anomaly_risk": focal_blur_anomaly_risk,
            "cutout_spoof_support": cutout_spoof_support,
            "cutout_local_focus_jump_risk": local_focus_jump,
            "cutout_local_boundary_risk": local_boundary_risk,
            "cutout_blur_inconsistency_risk": local_blur_inconsistency,
        }
        for item in region_results:
            name = str(item["name"])
            details[f"cutout_{name}_hole_risk"] = float(item["hole_risk"])
            details[f"cutout_{name}_focus_jump_risk"] = float(item["focus_jump_risk"])
            details[f"cutout_{name}_boundary_edge_risk"] = float(item["boundary_edge_risk"])
            details[f"cutout_{name}_sharpness_ratio_risk"] = float(item["sharpness_ratio_risk"])

        return CutoutAnomalyAssessment(
            hole_cutout_risk=hole_cutout_risk,
            focal_blur_anomaly_risk=focal_blur_anomaly_risk,
            cutout_spoof_support=cutout_spoof_support,
            details=details,
        )

    def _analyze_region(
        self,
        gray: np.ndarray,
        width: int,
        height: int,
        spec: tuple[float, float, float, float],
    ) -> dict[str, float]:
        x, y, w, h = _relative_box(width, height, *spec)
        region = gray[y : y + h, x : x + w]
        surround = _extract_surround(gray, x, y, w, h, scale=1.55)

        region_sharpness = _laplacian_variance(region)
        surround_sharpness = _laplacian_variance(surround)
        region_edges = _edge_density(region)
        surround_edges = _edge_density(surround)
        intensity_gap = abs(float(np.mean(region)) - float(np.mean(surround))) / 255.0

        region_sharpness_log = float(np.log1p(region_sharpness))
        surround_sharpness_log = float(np.log1p(surround_sharpness))
        sharpness_gap = abs(region_sharpness_log - surround_sharpness_log)
        focus_jump_risk = _normalize(sharpness_gap, 0.18, 1.10)
        sharpness_ratio = max(region_sharpness + 1.0, surround_sharpness + 1.0) / max(
            min(region_sharpness + 1.0, surround_sharpness + 1.0),
            1.0,
        )
        sharpness_ratio_risk = _normalize(sharpness_ratio, 1.4, 6.0)
        boundary_edge_risk = _normalize(max(0.0, surround_edges - region_edges), 0.03, 0.20)
        flat_region_risk = _normalize(max(0.0, surround_sharpness_log - region_sharpness_log), 0.10, 0.95)
        intensity_gap_risk = _normalize(intensity_gap, 0.05, 0.28)
        hole_risk = _clamp01(
            0.35 * boundary_edge_risk
            + 0.30 * flat_region_risk
            + 0.20 * focus_jump_risk
            + 0.15 * intensity_gap_risk
        )

        return {
            "region_sharpness_log": region_sharpness_log,
            "focus_jump_risk": focus_jump_risk,
            "sharpness_ratio_risk": sharpness_ratio_risk,
            "boundary_edge_risk": boundary_edge_risk,
            "hole_risk": hole_risk,
        }


def _relative_box(
    width: int,
    height: int,
    x_ratio: float,
    y_ratio: float,
    w_ratio: float,
    h_ratio: float,
) -> tuple[int, int, int, int]:
    x = int(round(width * x_ratio))
    y = int(round(height * y_ratio))
    w = max(8, int(round(width * w_ratio)))
    h = max(8, int(round(height * h_ratio)))
    x = max(0, min(x, max(0, width - w)))
    y = max(0, min(y, max(0, height - h)))
    return x, y, w, h


def _extract_surround(gray: np.ndarray, x: int, y: int, w: int, h: int, *, scale: float) -> np.ndarray:
    height, width = gray.shape[:2]
    cx = x + w / 2.0
    cy = y + h / 2.0
    expanded_w = min(width, max(w + 4, int(round(w * scale))))
    expanded_h = min(height, max(h + 4, int(round(h * scale))))
    x0 = max(0, int(round(cx - expanded_w / 2.0)))
    y0 = max(0, int(round(cy - expanded_h / 2.0)))
    x1 = min(width, x0 + expanded_w)
    y1 = min(height, y0 + expanded_h)
    expanded = gray[y0:y1, x0:x1]
    if expanded.size == 0:
        return gray[max(0, y - 2) : min(height, y + h + 2), max(0, x - 2) : min(width, x + w + 2)]

    inner_x0 = max(0, x - x0)
    inner_y0 = max(0, y - y0)
    inner_x1 = min(expanded.shape[1], inner_x0 + w)
    inner_y1 = min(expanded.shape[0], inner_y0 + h)
    mask = np.ones(expanded.shape[:2], dtype=bool)
    mask[inner_y0:inner_y1, inner_x0:inner_x1] = False
    surround_pixels = expanded[mask]
    if surround_pixels.size < 16:
        return expanded
    return surround_pixels.reshape(-1, 1)


def _laplacian_variance(region: np.ndarray) -> float:
    if region is None or region.size == 0:
        return 0.0
    return float(cv2.Laplacian(region.astype(np.uint8), cv2.CV_64F).var())


def _edge_density(region: np.ndarray) -> float:
    if region is None or region.size == 0:
        return 0.0
    if region.ndim == 2 and min(region.shape[:2]) <= 1:
        flat = region.reshape(-1)
        if flat.size < 8:
            return 0.0
        diffs = np.abs(np.diff(flat.astype(np.float32)))
        return float(np.mean(diffs > 16.0))
    edges = cv2.Canny(region.astype(np.uint8), 60, 140)
    return float(np.mean(edges > 0))


def _normalize(value: float, low: float, high: float) -> float:
    if high <= low:
        return 0.0
    return _clamp01((float(value) - low) / (high - low))


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))
