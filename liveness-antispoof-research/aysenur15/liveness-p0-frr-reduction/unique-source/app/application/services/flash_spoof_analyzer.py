"""Heuristic flash-response spoof analyzer for preview/debug replay detection."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np


@dataclass(frozen=True)
class FlashSpoofAnalysis:
    """Normalized flash-response analysis derived from pre/flash face ROIs."""

    flash_color_match_score: float
    flash_response_strength: float
    flash_response_consistency: float
    specular_hotspot_risk: float
    diffuse_response_score: float
    geometry_response_consistency: float
    planar_surface_risk: float
    details: dict[str, float | str] = field(default_factory=dict)


class FlashSpoofAnalyzer:
    """Estimate whether flash response resembles diffuse 3D skin or planar replay media."""

    _COLOR_INDEX = {"blue": 0, "green": 1, "red": 2}
    _REGIONS = {
        "forehead": (0.28, 0.10, 0.44, 0.18),
        "left_cheek": (0.12, 0.42, 0.24, 0.22),
        "right_cheek": (0.64, 0.42, 0.24, 0.22),
        "nose": (0.42, 0.34, 0.16, 0.26),
    }

    def analyze(
        self,
        *,
        pre_flash_bgr: Optional[np.ndarray],
        flash_bgr: Optional[np.ndarray],
        expected_color: str,
    ) -> FlashSpoofAnalysis:
        if (
            pre_flash_bgr is None
            or flash_bgr is None
            or pre_flash_bgr.size == 0
            or flash_bgr.size == 0
        ):
            return self._empty("missing_frames")

        if pre_flash_bgr.shape != flash_bgr.shape:
            flash_bgr = cv2.resize(
                flash_bgr,
                (pre_flash_bgr.shape[1], pre_flash_bgr.shape[0]),
                interpolation=cv2.INTER_LINEAR,
            )

        aligned_flash_bgr, alignment_details = self._align_flash_frame(
            pre_flash_bgr=pre_flash_bgr,
            flash_bgr=flash_bgr,
        )
        pre = pre_flash_bgr.astype(np.float32)
        flash = aligned_flash_bgr.astype(np.float32)
        delta = np.maximum(flash - pre, 0.0)
        gray_delta = np.mean(delta, axis=2)
        response_mask = gray_delta > max(4.0, float(np.percentile(gray_delta, 75)))

        color_match_score, strength_score, region_details = self._analyze_regions(
            pre=pre,
            delta=delta,
            expected_color=expected_color,
        )
        specular_hotspot_risk, diffuse_response_score, hotspot_details = self._analyze_specular_diffuse(
            gray_delta=gray_delta,
            response_mask=response_mask,
        )
        geometry_consistency, planar_surface_risk, geometry_details = self._analyze_geometry(
            region_details=region_details,
            gray_delta=gray_delta,
        )

        response_consistency = _clamp01(
            0.45 * color_match_score
            + 0.35 * geometry_consistency
            + 0.20 * diffuse_response_score
        )
        details: dict[str, float | str] = {
            "flash_color_match_score": color_match_score,
            "flash_response_strength_raw": strength_score,
            "specular_hotspot_risk": specular_hotspot_risk,
            "diffuse_response_score": diffuse_response_score,
            "geometry_response_consistency": geometry_consistency,
            "planar_surface_risk": planar_surface_risk,
            "flash_analysis_color": expected_color,
            "flash_pre_captured": 1.0,
            "flash_frame_captured": 1.0,
            "flash_response_pixels_ratio": float(np.mean(response_mask)),
            "flash_reflection_mean": float(np.mean(gray_delta)),
            "flash_reflection_p95": float(np.percentile(gray_delta, 95)),
            "flash_reflection_std": float(np.std(gray_delta)),
            **alignment_details,
            **hotspot_details,
            **geometry_details,
        }
        for region_name, region_metric in region_details.items():
            details[f"flash_region_{region_name}_target_shift"] = float(region_metric["target_shift"])
            details[f"flash_region_{region_name}_other_shift"] = float(region_metric["other_shift"])
            details[f"flash_region_{region_name}_match"] = float(region_metric["match"])
            details[f"flash_region_{region_name}_strength"] = float(region_metric["strength"])
            details[f"flash_region_{region_name}_target_chroma_gain"] = float(region_metric["target_chroma_gain"])
            details[f"flash_region_{region_name}_other_chroma_gain"] = float(region_metric["other_chroma_gain"])

        return FlashSpoofAnalysis(
            flash_color_match_score=color_match_score,
            flash_response_strength=strength_score,
            flash_response_consistency=response_consistency,
            specular_hotspot_risk=specular_hotspot_risk,
            diffuse_response_score=diffuse_response_score,
            geometry_response_consistency=geometry_consistency,
            planar_surface_risk=planar_surface_risk,
            details=details,
        )

    def _analyze_regions(
        self,
        *,
        pre: np.ndarray,
        delta: np.ndarray,
        expected_color: str,
    ) -> tuple[float, float, dict[str, dict[str, float]]]:
        target_index = self._COLOR_INDEX.get(expected_color)
        target_strengths: list[float] = []
        color_matches: list[float] = []
        region_details: dict[str, dict[str, float]] = {}

        for region_name, box in self._REGIONS.items():
            patch = self._crop_relative(delta, box)
            pre_patch = self._crop_relative(pre, box)
            if patch.size == 0:
                region_details[region_name] = {
                    "target_shift": 0.0,
                    "other_shift": 0.0,
                    "match": 0.0,
                    "strength": 0.0,
                    "target_chroma_gain": 0.0,
                    "other_chroma_gain": 0.0,
                }
                continue
            mean_bgr = patch.mean(axis=(0, 1))
            pre_mean_bgr = pre_patch.mean(axis=(0, 1)) if pre_patch.size else np.zeros(3, dtype=np.float32)
            if target_index is None:
                target_shift = float(np.mean(mean_bgr))
                other_shift = target_shift
                match = _clamp01(_normalize(target_shift, 4.0, 30.0))
                target_chroma_gain = 0.0
                other_chroma_gain = 0.0
            else:
                target_shift = float(mean_bgr[target_index])
                other_shift = float(np.mean([mean_bgr[idx] for idx in range(3) if idx != target_index]))
                pre_sum = float(np.sum(pre_mean_bgr) + 1e-6)
                flash_sum = float(np.sum(pre_mean_bgr + mean_bgr) + 1e-6)
                pre_target_chroma = float(pre_mean_bgr[target_index] / pre_sum)
                flash_target_chroma = float((pre_mean_bgr[target_index] + mean_bgr[target_index]) / flash_sum)
                pre_other_chroma = float(np.mean([pre_mean_bgr[idx] / pre_sum for idx in range(3) if idx != target_index]))
                flash_other_chroma = float(np.mean([
                    (pre_mean_bgr[idx] + mean_bgr[idx]) / flash_sum for idx in range(3) if idx != target_index
                ]))
                target_chroma_gain = max(0.0, flash_target_chroma - pre_target_chroma)
                other_chroma_gain = max(0.0, flash_other_chroma - pre_other_chroma)
                absolute_match = _clamp01(_normalize(target_shift - other_shift, 0.5, 16.0))
                chroma_match = _clamp01(_normalize(target_chroma_gain - other_chroma_gain, 0.003, 0.060))
                dominance = target_shift / max(target_shift + other_shift, 1e-6)
                dominance_match = _clamp01(_normalize(dominance, 0.40, 0.78))
                match = _clamp01(
                    0.30 * absolute_match
                    + 0.45 * chroma_match
                    + 0.25 * dominance_match
                )
            strength = _clamp01(
                0.65 * _normalize(target_shift, 1.0, 24.0)
                + 0.35 * _normalize(target_shift - other_shift, 0.5, 14.0)
            )
            target_strengths.append(strength)
            color_matches.append(match)
            region_details[region_name] = {
                "target_shift": target_shift,
                "other_shift": other_shift,
                "match": match,
                "strength": strength,
                "target_chroma_gain": target_chroma_gain,
                "other_chroma_gain": other_chroma_gain,
            }

        color_match_score = float(np.mean(color_matches)) if color_matches else 0.0
        strength_score = float(np.mean(target_strengths)) if target_strengths else 0.0
        return color_match_score, strength_score, region_details

    def _align_flash_frame(
        self,
        *,
        pre_flash_bgr: np.ndarray,
        flash_bgr: np.ndarray,
    ) -> tuple[np.ndarray, dict[str, float]]:
        if pre_flash_bgr.size == 0 or flash_bgr.size == 0:
            return flash_bgr, {
                "flash_alignment_dx": 0.0,
                "flash_alignment_dy": 0.0,
                "flash_alignment_response": 0.0,
            }
        try:
            pre_gray = cv2.cvtColor(pre_flash_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
            flash_gray = cv2.cvtColor(flash_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
            shift, response = cv2.phaseCorrelate(pre_gray, flash_gray)
            dx = float(np.clip(shift[0], -6.0, 6.0))
            dy = float(np.clip(shift[1], -6.0, 6.0))
            matrix = np.float32([[1.0, 0.0, dx], [0.0, 1.0, dy]])
            aligned = cv2.warpAffine(
                flash_bgr,
                matrix,
                (pre_flash_bgr.shape[1], pre_flash_bgr.shape[0]),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REFLECT,
            )
            return aligned, {
                "flash_alignment_dx": dx,
                "flash_alignment_dy": dy,
                "flash_alignment_response": float(max(0.0, response)),
            }
        except cv2.error:
            return flash_bgr, {
                "flash_alignment_dx": 0.0,
                "flash_alignment_dy": 0.0,
                "flash_alignment_response": 0.0,
            }

    def _analyze_specular_diffuse(
        self,
        *,
        gray_delta: np.ndarray,
        response_mask: np.ndarray,
    ) -> tuple[float, float, dict[str, float]]:
        if gray_delta.size == 0:
            return 0.0, 0.0, {
                "flash_hotspot_ratio": 0.0,
                "flash_hotspot_compactness": 0.0,
                "flash_diffuse_spread_ratio": 0.0,
                "flash_clipped_ratio": 0.0,
            }

        positive_pixels = gray_delta[response_mask] if np.any(response_mask) else gray_delta.reshape(-1)
        if positive_pixels.size == 0:
            return 0.0, 0.0, {
                "flash_hotspot_ratio": 0.0,
                "flash_hotspot_compactness": 0.0,
                "flash_diffuse_spread_ratio": 0.0,
                "flash_clipped_ratio": 0.0,
            }

        hotspot_threshold = max(float(np.percentile(positive_pixels, 97)), 10.0)
        hotspot_mask = (gray_delta >= hotspot_threshold).astype(np.uint8)
        hotspot_ratio = float(np.mean(hotspot_mask))
        clipped_ratio = float(np.mean(gray_delta >= 48.0))
        component_count, _, stats, _ = cv2.connectedComponentsWithStats(hotspot_mask, connectivity=8)
        compactness = 0.0
        for label in range(1, component_count):
            area = float(stats[label, cv2.CC_STAT_AREA])
            width = float(stats[label, cv2.CC_STAT_WIDTH])
            height = float(stats[label, cv2.CC_STAT_HEIGHT])
            compactness = max(compactness, area / max(width * height, 1.0))

        diffuse_spread_ratio = float(np.mean(response_mask))
        specular_hotspot_risk = _clamp01(
            0.40 * _normalize(hotspot_ratio, 0.002, 0.035)
            + 0.35 * _normalize(compactness, 0.35, 0.95)
            + 0.25 * _normalize(clipped_ratio, 0.001, 0.025)
        )
        diffuse_response_score = _clamp01(
            0.55 * _normalize(diffuse_spread_ratio, 0.04, 0.28)
            + 0.25 * (1.0 - specular_hotspot_risk)
            + 0.20 * (1.0 - _normalize(clipped_ratio, 0.001, 0.020))
        )
        return specular_hotspot_risk, diffuse_response_score, {
            "flash_hotspot_ratio": hotspot_ratio,
            "flash_hotspot_compactness": compactness,
            "flash_diffuse_spread_ratio": diffuse_spread_ratio,
            "flash_clipped_ratio": clipped_ratio,
        }

    def _analyze_geometry(
        self,
        *,
        region_details: dict[str, dict[str, float]],
        gray_delta: np.ndarray,
    ) -> tuple[float, float, dict[str, float]]:
        strengths = [metrics["strength"] for metrics in region_details.values()]
        if not strengths:
            return 0.0, 0.0, {
                "flash_region_strength_std": 0.0,
                "flash_cheek_balance": 0.0,
                "flash_nose_cheek_delta": 0.0,
            }

        region_std = float(np.std(strengths))
        left_cheek = region_details.get("left_cheek", {}).get("strength", 0.0)
        right_cheek = region_details.get("right_cheek", {}).get("strength", 0.0)
        nose = region_details.get("nose", {}).get("strength", 0.0)
        forehead = region_details.get("forehead", {}).get("strength", 0.0)
        cheek_balance = 1.0 - _clamp01(abs(left_cheek - right_cheek) / 0.40)
        nose_cheek_delta = abs(nose - ((left_cheek + right_cheek) * 0.5))
        forehead_cheek_delta = abs(forehead - ((left_cheek + right_cheek) * 0.5))

        geometry_consistency = _clamp01(
            0.35 * _normalize(region_std, 0.03, 0.22)
            + 0.30 * cheek_balance
            + 0.20 * _normalize(nose_cheek_delta, 0.04, 0.28)
            + 0.15 * _normalize(forehead_cheek_delta, 0.03, 0.24)
        )
        # Low spatial variance in gray_delta = uniform illumination = flat surface.
        # A 3D face produces uneven flash response (nose protrudes, cheeks recede).
        gray_delta_spatial_std = float(np.std(gray_delta))
        gray_delta_uniformity = 1.0 - _normalize(gray_delta_spatial_std, 1.5, 9.0)
        planar_surface_risk = _clamp01(
            0.55 * (1.0 - _normalize(region_std, 0.03, 0.22))
            + 0.30 * (1.0 - _normalize(max(nose_cheek_delta, forehead_cheek_delta), 0.03, 0.24))
            + 0.15 * gray_delta_uniformity
        )
        return geometry_consistency, planar_surface_risk, {
            "flash_region_strength_std": region_std,
            "flash_cheek_balance": cheek_balance,
            "flash_nose_cheek_delta": nose_cheek_delta,
            "flash_forehead_cheek_delta": forehead_cheek_delta,
            "flash_gray_delta_spatial_std": gray_delta_spatial_std,
            "flash_gray_delta_uniformity": float(gray_delta_uniformity),
        }

    def _crop_relative(self, frame: np.ndarray, box: tuple[float, float, float, float]) -> np.ndarray:
        height, width = frame.shape[:2]
        x = int(round(box[0] * width))
        y = int(round(box[1] * height))
        w = max(1, int(round(box[2] * width)))
        h = max(1, int(round(box[3] * height)))
        x2 = min(width, x + w)
        y2 = min(height, y + h)
        return frame[max(0, y):y2, max(0, x):x2]

    def _empty(self, reason: str) -> FlashSpoofAnalysis:
        return FlashSpoofAnalysis(
            flash_color_match_score=0.0,
            flash_response_strength=0.0,
            flash_response_consistency=0.0,
            specular_hotspot_risk=0.0,
            diffuse_response_score=0.0,
            geometry_response_consistency=0.0,
            planar_surface_risk=0.0,
            details={
                "flash_analysis_reason": reason,
                "flash_pre_captured": 0.0,
                "flash_frame_captured": 0.0,
            },
        )


def _normalize(value: float, low: float, high: float) -> float:
    if high <= low:
        return 0.0
    return _clamp01((float(value) - low) / (high - low))


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))
