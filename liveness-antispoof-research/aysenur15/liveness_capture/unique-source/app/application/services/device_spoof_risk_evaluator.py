"""Replay/device spoof risk analysis for developer-facing live preview flows."""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

from app.application.services.cutout_anomaly_detector import CutoutAnomalyDetector
from app.application.services.device_boundary_detector import DeviceBoundaryDetector
from app.application.services.flash_spoof_analyzer import FlashSpoofAnalyzer
from app.application.services.light_challenge_service import LightChallengeService
from app.infrastructure.ml.liveness.moire_pattern_analysis import analyze_moire_pattern


@dataclass(frozen=True)
class DeviceSpoofRiskAssessment:
    """Normalized replay/device spoof indicators."""

    moire_risk: float
    reflection_risk: float
    flicker_risk: float
    flash_response_score: float
    flash_response_strength: float
    flash_response_consistency: float
    flash_replay_risk: float
    hole_cutout_risk: float
    focal_blur_anomaly_risk: float
    cutout_spoof_support: float
    screen_frame_risk: float
    device_replay_risk: float
    details: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, float]:
        """Serialize the primary risk values for UI/debug consumers."""
        return {
            "moire_risk": self.moire_risk,
            "reflection_risk": self.reflection_risk,
            "flicker_risk": self.flicker_risk,
            "flash_response_score": self.flash_response_score,
            "flash_response_strength": self.flash_response_strength,
            "flash_response_consistency": self.flash_response_consistency,
            "flash_replay_risk": self.flash_replay_risk,
            "hole_cutout_risk": self.hole_cutout_risk,
            "focal_blur_anomaly_risk": self.focal_blur_anomaly_risk,
            "cutout_spoof_support": self.cutout_spoof_support,
            "screen_frame_risk": self.screen_frame_risk,
            "device_replay_risk": self.device_replay_risk,
        }


@dataclass
class _FlashChallengeState:
    color: str
    issued_at: float
    visible_until: float
    response_deadline: float
    baseline_bgr: list[float]
    baseline_frame_bgr: np.ndarray
    observed_shifts: list[float] = field(default_factory=list)
    passed_shifts: list[float] = field(default_factory=list)
    evaluated_samples: int = 0
    passed_samples: int = 0
    peak_shift: float = 0.0
    last_face_mean_bgr: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    analysis_frames_captured: int = 0
    best_color_match_score: float = 0.0
    best_specular_hotspot_risk: float = 0.0
    best_diffuse_response_score: float = 0.0
    best_geometry_response_consistency: float = 0.0
    best_planar_surface_risk: float = 0.0
    latest_analysis_details: dict[str, float | str] = field(default_factory=dict)


@dataclass(frozen=True)
class _FlashChallengeResult:
    response_score: float
    response_strength: float
    response_consistency: float
    replay_risk: float
    sample_count: int
    pass_ratio: float


class DeviceSpoofRiskEvaluator:
    """Estimate device replay risk without modifying core liveness scoring."""

    DEVICE_REPLAY_MOIRE_WEIGHT = 0.35
    DEVICE_REPLAY_REFLECTION_WEIGHT = 0.20
    DEVICE_REPLAY_FLICKER_WEIGHT = 0.15
    DEVICE_REPLAY_FLASH_WEIGHT = 0.30
    DEVICE_REPLAY_SCREEN_FRAME_WEIGHT = 0.0

    def __init__(
        self,
        *,
        history_size: int = 12,
        enable_flash_replay: bool = False,
        flash_interval_seconds: float = 2.5,
        flash_history_size: int = 4,
        replay_fusion_weights: Optional[dict[str, float]] = None,
    ) -> None:
        self._max_history_samples = max(4, history_size)
        self._cutout_anomaly_detector = CutoutAnomalyDetector()
        self._flash_spoof_analyzer = FlashSpoofAnalyzer()
        self._device_boundary_detector = DeviceBoundaryDetector(history_size=5)
        self._light_challenge_service = LightChallengeService(colors=("red", "green", "blue"))
        self._enable_flash_replay = enable_flash_replay
        self._flash_interval_seconds = max(0.0, float(flash_interval_seconds))
        self._flash_history: deque[_FlashChallengeResult] = deque(maxlen=max(1, flash_history_size))
        self._flash_state: Optional[_FlashChallengeState] = None
        self._next_flash_at = 0.0
        self._last_flash_analysis_details: dict[str, float | str] = {}
        self._last_flash_visual_state: dict[str, float | str | bool | None] = {
            "enabled": enable_flash_replay,
            "visible": False,
            "color": None,
            "phase": "disabled" if not enable_flash_replay else "idle",
            "remaining_ms": 0.0,
        }
        self._replay_fusion_weights = {
            "moire": self.DEVICE_REPLAY_MOIRE_WEIGHT,
            "reflection": self.DEVICE_REPLAY_REFLECTION_WEIGHT,
            "flicker": self.DEVICE_REPLAY_FLICKER_WEIGHT,
            "flash": self.DEVICE_REPLAY_FLASH_WEIGHT,
            "screen_frame": self.DEVICE_REPLAY_SCREEN_FRAME_WEIGHT,
        }
        if replay_fusion_weights:
            for key, value in replay_fusion_weights.items():
                if key in self._replay_fusion_weights:
                    self._replay_fusion_weights[key] = max(0.0, float(value))

    def evaluate(
        self,
        *,
        frame_bgr: np.ndarray,
        face_region_bgr: Optional[np.ndarray] = None,
        face_bounding_box: Optional[tuple[int, int, int, int]] = None,
        frame_timestamp: Optional[float] = None,
    ) -> DeviceSpoofRiskAssessment:
        """Return normalized replay/device risk signals for the current frame."""
        timestamp = float(frame_timestamp or time.time())
        analysis_region = face_region_bgr if face_region_bgr is not None and face_region_bgr.size else frame_bgr
        if analysis_region is None or analysis_region.size == 0:
            return DeviceSpoofRiskAssessment(
                moire_risk=0.0,
                reflection_risk=0.0,
                flicker_risk=0.0,
                flash_response_score=0.0,
                flash_response_strength=0.0,
                flash_response_consistency=0.0,
                flash_replay_risk=0.0,
                hole_cutout_risk=0.0,
                focal_blur_anomaly_risk=0.0,
                cutout_spoof_support=0.0,
                screen_frame_risk=0.0,
                device_replay_risk=0.0,
                details={},
            )

        gray = cv2.cvtColor(analysis_region, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(analysis_region, cv2.COLOR_BGR2HSV)
        moire_risk, moire_details = self._compute_moire_risk(gray)
        reflection_risk, reflection_details = self._compute_reflection_risk(hsv)
        cutout_assessment = self._cutout_anomaly_detector.analyze(analysis_region)
        screen_frame_risk, screen_frame_details = self._compute_screen_frame_risk(
            frame_bgr=frame_bgr,
            face_bounding_box=face_bounding_box,
        )
        flicker_details = self._compute_flicker_signal_sample(gray)
        flash_metrics, flash_details = self._compute_flash_response_metrics(
            face_region_bgr=face_region_bgr,
            frame_timestamp=timestamp,
        )
        flicker_risk = 0.0
        device_replay_risk = self._combine_risks(
            moire_risk=moire_risk,
            reflection_risk=reflection_risk,
            flicker_risk=flicker_risk,
            flash_replay_risk=flash_metrics["flash_replay_risk"],
            screen_frame_risk=screen_frame_risk,
            weights=self._replay_fusion_weights,
        )

        details = {
            **moire_details,
            **reflection_details,
            **cutout_assessment.details,
            **screen_frame_details,
            **flicker_details,
            **flash_details,
            "device_replay_risk": device_replay_risk,
        }
        return DeviceSpoofRiskAssessment(
            moire_risk=moire_risk,
            reflection_risk=reflection_risk,
            flicker_risk=flicker_risk,
            flash_response_score=flash_metrics["flash_response_score"],
            flash_response_strength=flash_metrics["flash_response_strength"],
            flash_response_consistency=flash_metrics["flash_response_consistency"],
            flash_replay_risk=flash_metrics["flash_replay_risk"],
            hole_cutout_risk=cutout_assessment.hole_cutout_risk,
            focal_blur_anomaly_risk=cutout_assessment.focal_blur_anomaly_risk,
            cutout_spoof_support=cutout_assessment.cutout_spoof_support,
            screen_frame_risk=screen_frame_risk,
            device_replay_risk=device_replay_risk,
            details=details,
        )

    def update_with_temporal_history(
        self,
        assessment: DeviceSpoofRiskAssessment,
        temporal_signal_history: list[dict[str, float]],
    ) -> DeviceSpoofRiskAssessment:
        """Recompute flicker/device replay risk using the shared rolling preview history."""
        bounded_history = temporal_signal_history[-self._max_history_samples :]
        flicker_risk, flicker_details = self._compute_flicker_risk(bounded_history)
        device_replay_risk = self._combine_risks(
            moire_risk=assessment.moire_risk,
            reflection_risk=assessment.reflection_risk,
            flicker_risk=flicker_risk,
            flash_replay_risk=assessment.flash_replay_risk,
            screen_frame_risk=assessment.screen_frame_risk,
            weights=self._replay_fusion_weights,
        )
        details = {
            **assessment.details,
            **flicker_details,
            "device_replay_risk": device_replay_risk,
        }
        return DeviceSpoofRiskAssessment(
            moire_risk=assessment.moire_risk,
            reflection_risk=assessment.reflection_risk,
            flicker_risk=flicker_risk,
            flash_response_score=assessment.flash_response_score,
            flash_response_strength=assessment.flash_response_strength,
            flash_response_consistency=assessment.flash_response_consistency,
            flash_replay_risk=assessment.flash_replay_risk,
            hole_cutout_risk=assessment.hole_cutout_risk,
            focal_blur_anomaly_risk=assessment.focal_blur_anomaly_risk,
            cutout_spoof_support=assessment.cutout_spoof_support,
            screen_frame_risk=assessment.screen_frame_risk,
            device_replay_risk=device_replay_risk,
            details=details,
        )

    def get_flash_visual_state(self) -> dict[str, float | str | bool | None]:
        """Return the current debug flash state for the preview overlay."""
        return dict(self._last_flash_visual_state)

    @staticmethod
    def temporal_signal_sample_from_details(details: dict[str, float]) -> Optional[dict[str, float]]:
        """Extract replay-temporal signals previously stored on a frame."""
        mean_luma = _maybe_float(details.get("spoof_temporal_mean_luma"))
        line_profile_std = _maybe_float(details.get("spoof_temporal_line_profile_std"))
        row_profile_std = _maybe_float(details.get("spoof_temporal_row_profile_std"))
        col_profile_std = _maybe_float(details.get("spoof_temporal_col_profile_std"))
        if mean_luma is None or line_profile_std is None:
            return None
        return {
            "mean_luma": mean_luma,
            "line_profile_std": line_profile_std,
            "row_profile_std": row_profile_std or 0.0,
            "col_profile_std": col_profile_std or 0.0,
        }

    def _compute_moire_risk(self, gray: np.ndarray) -> tuple[float, dict[str, float]]:
        analysis = analyze_moire_pattern(gray)
        moire_risk = float(analysis["moire_risk"])
        return moire_risk, {
            "moire_score": float(analysis["moire_score"]),
            "moire_response_count": float(analysis["moire_response_count"]),
            "moire_response_fraction": float(analysis["moire_response_fraction"]),
            "moire_response_std_mean": float(analysis["moire_response_std_mean"]),
            "moire_response_std_max": float(analysis["moire_response_std_max"]),
            "moire_response_std_min": float(analysis["moire_response_std_min"]),
            "moire_response_std_range": float(analysis["moire_response_std_range"]),
            "moire_response_std_std": float(analysis["moire_response_std_std"]),
            "moire_gabor_strength": float(analysis["moire_gabor_strength"]),
            "moire_orientation_selectivity": float(analysis["moire_orientation_selectivity"]),
            "moire_periodic_gabor_risk": float(analysis["moire_periodic_gabor_risk"]),
            "moire_center_focus_ratio": float(analysis["moire_center_focus_ratio"]),
            "moire_fft_mid_low_ratio": float(analysis["moire_fft_mid_low_ratio"]),
            "moire_fft_peak_ratio": float(analysis["moire_fft_peak_ratio"]),
            "moire_fft_risk": float(analysis["moire_fft_risk"]),
        }

    def _compute_reflection_risk(self, hsv: np.ndarray) -> tuple[float, dict[str, float]]:
        saturation = hsv[:, :, 1].astype(np.float32)
        value = hsv[:, :, 2].astype(np.float32)
        low_sat_bright = (saturation < 36.0) & (value > 212.0)
        clipped_highlight = value > 245.0
        bright_ratio = float(np.mean(low_sat_bright))
        clipped_ratio = float(np.mean(clipped_highlight))
        p99_value = float(np.percentile(value, 99))
        cluster_metrics = self._analyze_highlight_clusters(low_sat_bright.astype(np.uint8))
        glossy_patch_ratio = float(cluster_metrics["glossy_patch_ratio"])
        compact_highlight_score = float(cluster_metrics["compact_highlight_score"])
        max_cluster_fill = float(cluster_metrics["max_cluster_fill"])
        max_cluster_area_ratio = float(cluster_metrics["max_cluster_area_ratio"])
        reflection_risk = _clamp01(
            0.30 * _normalize(bright_ratio, 0.015, 0.120)
            + 0.20 * _normalize(clipped_ratio, 0.003, 0.050)
            + 0.15 * _normalize(p99_value, 225.0, 252.0)
            + 0.20 * _normalize(glossy_patch_ratio, 0.003, 0.040)
            + 0.15 * compact_highlight_score
        )
        return reflection_risk, {
            "reflection_low_sat_bright_ratio": bright_ratio,
            "reflection_clipped_ratio": clipped_ratio,
            "reflection_p99_value": p99_value,
            "reflection_glossy_patch_ratio": glossy_patch_ratio,
            "reflection_compact_highlight_score": compact_highlight_score,
            "reflection_max_cluster_fill": max_cluster_fill,
            "reflection_max_cluster_area_ratio": max_cluster_area_ratio,
        }

    def _compute_screen_frame_risk(
        self,
        *,
        frame_bgr: np.ndarray,
        face_bounding_box: Optional[tuple[int, int, int, int]],
    ) -> tuple[float, dict[str, float]]:
        if frame_bgr is None or frame_bgr.size == 0 or face_bounding_box is None:
            return 0.0, {
                "screen_frame_area_ratio": 0.0,
                "screen_frame_rectangularity": 0.0,
                "screen_frame_border_darkness": 0.0,
                "screen_frame_inner_brightness": 0.0,
                "screen_frame_border_contrast": 0.0,
                "screen_frame_face_center_inside": 0.0,
                "screen_frame_source": 0.0,
            }
        detection = self._device_boundary_detector.analyze(
            frame_bgr=frame_bgr,
            face_bbox=face_bounding_box,
        )
        details = {
            "screen_frame_area_ratio": float(detection.details.get("boundary_candidate_area_ratio") or 0.0),
            "screen_frame_rectangularity": float(detection.details.get("boundary_rectangularity") or 0.0),
            "screen_frame_border_darkness": 0.0,
            "screen_frame_inner_brightness": 0.0,
            "screen_frame_border_contrast": 0.0,
            "screen_frame_face_center_inside": float(detection.details.get("boundary_face_coverage_score") or 0.0),
            "screen_frame_source": float(detection.is_spoof_confirmed),
            "screen_frame_line_score": float(detection.details.get("boundary_line_score") or 0.0),
            "screen_frame_parallel_score": float(detection.details.get("boundary_parallel_score") or 0.0),
            "screen_frame_orthogonal_score": float(detection.details.get("boundary_orthogonal_score") or 0.0),
            "screen_frame_aspect_score": float(detection.details.get("boundary_aspect_score") or 0.0),
            "screen_frame_temporal_sync_score": float(detection.details.get("boundary_temporal_sync_score") or 0.0),
            "screen_frame_motion_sync_ratio": float(detection.details.get("boundary_motion_sync_ratio") or 0.0),
            "screen_frame_candidate_score": detection.boundary_score,
        }
        return detection.boundary_score, details

    def _analyze_highlight_clusters(self, bright_mask: np.ndarray) -> dict[str, float]:
        if bright_mask.size == 0:
            return {
                "glossy_patch_ratio": 0.0,
                "compact_highlight_score": 0.0,
                "max_cluster_fill": 0.0,
                "max_cluster_area_ratio": 0.0,
            }

        roi_area = float(bright_mask.shape[0] * bright_mask.shape[1])
        kernel = np.ones((3, 3), dtype=np.uint8)
        cleaned = cv2.morphologyEx(bright_mask, cv2.MORPH_OPEN, kernel)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)
        component_count, _, stats, _ = cv2.connectedComponentsWithStats(cleaned, connectivity=8)

        glossy_patch_area = 0.0
        compact_highlight_score = 0.0
        max_cluster_fill = 0.0
        max_cluster_area_ratio = 0.0
        min_component_area = max(4.0, roi_area * 0.0003)
        max_component_area = roi_area * 0.10

        for label in range(1, component_count):
            area = float(stats[label, cv2.CC_STAT_AREA])
            if area < min_component_area or area > max_component_area:
                continue

            width = float(stats[label, cv2.CC_STAT_WIDTH])
            height = float(stats[label, cv2.CC_STAT_HEIGHT])
            bbox_area = max(width * height, 1.0)
            fill_ratio = area / bbox_area
            area_ratio = area / max(roi_area, 1.0)

            glossy_patch_area += area
            max_cluster_fill = max(max_cluster_fill, fill_ratio)
            max_cluster_area_ratio = max(max_cluster_area_ratio, area_ratio)

            component_score = _clamp01(
                0.55 * _normalize(area_ratio, 0.0008, 0.020)
                + 0.45 * _normalize(fill_ratio, 0.30, 0.92)
            )
            compact_highlight_score = max(compact_highlight_score, component_score)

        return {
            "glossy_patch_ratio": glossy_patch_area / max(roi_area, 1.0),
            "compact_highlight_score": compact_highlight_score,
            "max_cluster_fill": max_cluster_fill,
            "max_cluster_area_ratio": max_cluster_area_ratio,
        }

    def _compute_flicker_signal_sample(self, gray: np.ndarray) -> dict[str, float]:
        row_profile = np.mean(gray, axis=1)
        col_profile = np.mean(gray, axis=0)
        row_profile_std = float(np.std(row_profile))
        col_profile_std = float(np.std(col_profile))
        return {
            "spoof_temporal_mean_luma": float(np.mean(gray)),
            "spoof_temporal_row_profile_std": row_profile_std,
            "spoof_temporal_col_profile_std": col_profile_std,
            "spoof_temporal_line_profile_std": max(row_profile_std, col_profile_std),
        }

    def _compute_flicker_risk(self, temporal_signal_history: list[dict[str, float]]) -> tuple[float, dict[str, float]]:
        if len(temporal_signal_history) < 4:
            return 0.0, {
                "flicker_luma_std": 0.0,
                "flicker_delta_std": 0.0,
                "flicker_zero_cross_ratio": 0.0,
                "flicker_line_profile_std": 0.0,
                "flicker_line_delta_std": 0.0,
            }

        values = np.asarray([sample["mean_luma"] for sample in temporal_signal_history], dtype=np.float32)
        line_values = np.asarray([sample["line_profile_std"] for sample in temporal_signal_history], dtype=np.float32)
        deltas = np.diff(values)
        line_deltas = np.diff(line_values)
        centered = values - np.mean(values)
        zero_crossings = np.sum(np.signbit(centered[:-1]) != np.signbit(centered[1:])) if len(centered) > 1 else 0
        zero_cross_ratio = float(zero_crossings / max(len(centered) - 1, 1))
        luma_std = float(np.std(values))
        delta_std = float(np.std(deltas)) if len(deltas) else 0.0
        line_profile_std = float(np.std(line_values))
        line_delta_std = float(np.std(line_deltas)) if len(line_deltas) else 0.0
        flicker_risk = _clamp01(
            0.35 * _normalize(luma_std, 1.5, 8.5)
            + 0.30 * _normalize(delta_std, 1.0, 6.5)
            + 0.20 * _normalize(zero_cross_ratio, 0.20, 0.75)
            + 0.10 * _normalize(line_profile_std, 0.8, 5.0)
            + 0.05 * _normalize(line_delta_std, 0.5, 4.0)
        )
        return flicker_risk, {
            "flicker_luma_std": luma_std,
            "flicker_delta_std": delta_std,
            "flicker_zero_cross_ratio": zero_cross_ratio,
            "flicker_line_profile_std": line_profile_std,
            "flicker_line_delta_std": line_delta_std,
        }

    def _compute_flash_response_metrics(
        self,
        *,
        face_region_bgr: Optional[np.ndarray],
        frame_timestamp: float,
    ) -> tuple[dict[str, float], dict[str, float | str]]:
        if not self._enable_flash_replay:
            self._update_flash_visual_state(frame_timestamp=frame_timestamp)
            summary = self._summarize_flash_history()
            details: dict[str, float | str] = {
                **summary,
                "flash_challenge_enabled": float(self._enable_flash_replay),
                "flash_response_sample_count": 0.0,
                "flash_response_pass_ratio": 0.0,
            }
            details.update(self._flash_visual_details())
            return summary, details

        if face_region_bgr is None or face_region_bgr.size == 0:
            if self._flash_state is not None and frame_timestamp > self._flash_state.response_deadline:
                self._finalize_flash_challenge(frame_timestamp)
            self._update_flash_visual_state(frame_timestamp=frame_timestamp)
            summary = self._summarize_flash_history(include_current=True)
            details = {
                **summary,
                "flash_challenge_enabled": 1.0,
                "flash_response_sample_count": float(self._flash_state.evaluated_samples if self._flash_state else 0),
                "flash_response_pass_ratio": float(
                    self._flash_state.passed_samples / max(self._flash_state.evaluated_samples, 1)
                ) if self._flash_state is not None else 0.0,
            }
            details.update(self._flash_visual_details())
            return summary, details

        if self._flash_state is None and frame_timestamp >= self._next_flash_at:
            self._start_flash_challenge(face_region_bgr=face_region_bgr, frame_timestamp=frame_timestamp)

        if self._flash_state is not None:
            if frame_timestamp > self._flash_state.response_deadline:
                self._finalize_flash_challenge(frame_timestamp)
            else:
                self._observe_flash_response(face_region_bgr=face_region_bgr, frame_timestamp=frame_timestamp)

        self._update_flash_visual_state(frame_timestamp=frame_timestamp)
        summary = self._summarize_flash_history(include_current=True)
        current_state = self._flash_state
        details = {
            **summary,
            "flash_challenge_enabled": 1.0,
            "flash_response_sample_count": float(current_state.evaluated_samples if current_state else 0),
            "flash_response_pass_ratio": float(
                current_state.passed_samples / max(current_state.evaluated_samples, 1)
            ) if current_state is not None else 0.0,
        }
        details.update(self._flash_visual_details())
        return summary, details

    def _start_flash_challenge(self, *, face_region_bgr: np.ndarray, frame_timestamp: float) -> None:
        challenge = self._light_challenge_service.generate_challenge()
        challenge["issued_at"] = frame_timestamp
        challenge["expires_at"] = frame_timestamp + (challenge["expected_response_window_ms"] / 1000.0)
        self._flash_state = _FlashChallengeState(
            color=str(challenge["color"]),
            issued_at=frame_timestamp,
            visible_until=frame_timestamp + (challenge["duration_ms"] / 1000.0),
            response_deadline=frame_timestamp + (challenge["expected_response_window_ms"] / 1000.0),
            baseline_bgr=face_region_bgr.mean(axis=(0, 1)).astype(float).tolist(),
            baseline_frame_bgr=face_region_bgr.copy(),
        )

    def _observe_flash_response(self, *, face_region_bgr: np.ndarray, frame_timestamp: float) -> None:
        if self._flash_state is None:
            return
        verification = self._light_challenge_service.verify_response(
            frame=face_region_bgr,
            expected_color=self._flash_state.color,
            flash_timestamp=self._flash_state.issued_at,
            frame_timestamp=frame_timestamp,
            baseline_bgr=self._flash_state.baseline_bgr,
        )
        face_mean_bgr = verification.get("face_mean_bgr")
        if isinstance(face_mean_bgr, list) and len(face_mean_bgr) == 3:
            self._flash_state.last_face_mean_bgr = [float(value) for value in face_mean_bgr]

        delay_seconds = _maybe_float(verification.get("delay_seconds")) or 0.0
        min_delay_seconds = self._light_challenge_service._minimum_delay_ms / 1000.0
        if verification.get("reason") == "timing_mismatch" and delay_seconds < min_delay_seconds:
            return

        color_shift = float(verification.get("color_shift") or 0.0)
        self._flash_state.evaluated_samples += 1
        self._flash_state.observed_shifts.append(color_shift)
        self._flash_state.peak_shift = max(self._flash_state.peak_shift, color_shift)
        if verification.get("passed"):
            self._flash_state.passed_samples += 1
            self._flash_state.passed_shifts.append(color_shift)
        analysis = self._flash_spoof_analyzer.analyze(
            pre_flash_bgr=self._flash_state.baseline_frame_bgr,
            flash_bgr=face_region_bgr,
            expected_color=self._flash_state.color,
        )
        self._flash_state.analysis_frames_captured += 1
        self._flash_state.best_color_match_score = max(
            self._flash_state.best_color_match_score,
            analysis.flash_color_match_score,
        )
        self._flash_state.best_specular_hotspot_risk = max(
            self._flash_state.best_specular_hotspot_risk,
            analysis.specular_hotspot_risk,
        )
        self._flash_state.best_diffuse_response_score = max(
            self._flash_state.best_diffuse_response_score,
            analysis.diffuse_response_score,
        )
        self._flash_state.best_geometry_response_consistency = max(
            self._flash_state.best_geometry_response_consistency,
            analysis.geometry_response_consistency,
        )
        self._flash_state.best_planar_surface_risk = max(
            self._flash_state.best_planar_surface_risk,
            analysis.planar_surface_risk,
        )
        self._flash_state.latest_analysis_details = dict(analysis.details)

    def _finalize_flash_challenge(self, frame_timestamp: Optional[float] = None) -> None:
        if self._flash_state is None:
            return
        self._last_flash_analysis_details = dict(self._flash_state.latest_analysis_details)
        self._last_flash_analysis_details["flash_color_match_score"] = float(self._flash_state.best_color_match_score)
        self._last_flash_analysis_details["specular_hotspot_risk"] = float(self._flash_state.best_specular_hotspot_risk)
        self._last_flash_analysis_details["diffuse_response_score"] = float(self._flash_state.best_diffuse_response_score)
        self._last_flash_analysis_details["geometry_response_consistency"] = float(
            self._flash_state.best_geometry_response_consistency
        )
        self._last_flash_analysis_details["planar_surface_risk"] = float(self._flash_state.best_planar_surface_risk)
        self._last_flash_analysis_details["pre_flash_captured"] = 1.0
        self._last_flash_analysis_details["flash_frame_captured"] = float(self._flash_state.analysis_frames_captured > 0)
        self._flash_history.append(self._summarize_flash_state(self._flash_state))
        self._flash_state = None
        self._next_flash_at = float(frame_timestamp or time.time()) + self._flash_interval_seconds

    def _summarize_flash_history(self, *, include_current: bool = False) -> dict[str, float]:
        entries = list(self._flash_history)
        if include_current and self._flash_state is not None:
            entries.append(self._summarize_flash_state(self._flash_state))
        if not entries:
            return {
                "flash_response_score": 0.0,
                "flash_response_strength": 0.0,
                "flash_response_consistency": 0.0,
                "flash_replay_risk": 0.0,
            }
        return {
            "flash_response_score": float(np.mean([entry.response_score for entry in entries])),
            "flash_response_strength": float(np.mean([entry.response_strength for entry in entries])),
            "flash_response_consistency": float(np.mean([entry.response_consistency for entry in entries])),
            "flash_replay_risk": float(np.mean([entry.replay_risk for entry in entries])),
        }

    def _summarize_flash_state(self, state: _FlashChallengeState) -> _FlashChallengeResult:
        strength = _normalize(
            state.peak_shift,
            self._light_challenge_service._min_color_shift,
            max(self._light_challenge_service._min_color_shift + 0.10, 0.18),
        )
        strength = max(
            strength,
            _clamp01(
                0.45 * state.best_color_match_score
                + 0.30 * state.best_diffuse_response_score
                + 0.25 * state.best_geometry_response_consistency
            ),
        )
        pass_ratio = state.passed_samples / max(state.evaluated_samples, 1)
        if not state.observed_shifts:
            consistency = 0.0
        elif len(state.observed_shifts) == 1:
            consistency = _clamp01(0.65 * pass_ratio + 0.35 * strength)
        else:
            positive_shifts = state.passed_shifts or state.observed_shifts
            mean_shift = float(np.mean(positive_shifts)) if positive_shifts else 0.0
            shift_std = float(np.std(positive_shifts)) if len(positive_shifts) > 1 else 0.0
            temporal_stability = 1.0 - _normalize(shift_std, 0.01, max(mean_shift, 0.08))
            consistency = _clamp01(
                0.45 * pass_ratio
                + 0.25 * temporal_stability
                + 0.15 * state.best_color_match_score
                + 0.15 * state.best_geometry_response_consistency
            )
        response_score = _clamp01(
            0.30 * strength
            + 0.20 * consistency
            + 0.15 * pass_ratio
            + 0.15 * state.best_color_match_score
            + 0.12 * state.best_diffuse_response_score
            + 0.08 * state.best_geometry_response_consistency
        )
        replay_risk = _clamp01(
            1.0
            - (
                0.38 * response_score
                + 0.18 * state.best_color_match_score
                + 0.14 * state.best_diffuse_response_score
                + 0.10 * state.best_geometry_response_consistency
            )
            + 0.12 * state.best_specular_hotspot_risk
            + 0.08 * state.best_planar_surface_risk
        )
        return _FlashChallengeResult(
            response_score=response_score,
            response_strength=strength,
            response_consistency=consistency,
            replay_risk=replay_risk,
            sample_count=state.evaluated_samples,
            pass_ratio=pass_ratio,
        )

    def _update_flash_visual_state(self, *, frame_timestamp: float) -> None:
        state = self._flash_state
        visible = bool(state is not None and frame_timestamp <= state.visible_until)
        remaining_ms = 0.0
        if state is not None:
            if visible:
                phase = "flash"
                remaining_ms = max(0.0, (state.visible_until - frame_timestamp) * 1000.0)
            elif frame_timestamp <= state.response_deadline:
                phase = "response_window"
                remaining_ms = max(0.0, (state.response_deadline - frame_timestamp) * 1000.0)
            else:
                phase = "finalizing"
        else:
            phase = "idle" if self._enable_flash_replay else "disabled"
        self._last_flash_visual_state = {
            "enabled": self._enable_flash_replay,
            "visible": visible,
            "color": state.color if state is not None else None,
            "phase": phase,
            "remaining_ms": remaining_ms,
        }

    def _flash_visual_details(self) -> dict[str, float | str]:
        state = self._flash_state
        details: dict[str, float | str] = {
            "flash_challenge_visible": float(bool(self._last_flash_visual_state.get("visible"))),
            "flash_challenge_remaining_ms": float(self._last_flash_visual_state.get("remaining_ms") or 0.0),
            "flash_challenge_color": str(self._last_flash_visual_state.get("color") or "-"),
            "flash_challenge_phase": str(self._last_flash_visual_state.get("phase") or "idle"),
            "flash_state": str(self._last_flash_visual_state.get("phase") or "idle"),
            "flash_face_mean_b": float(state.last_face_mean_bgr[0]) if state is not None else 0.0,
            "flash_face_mean_g": float(state.last_face_mean_bgr[1]) if state is not None else 0.0,
            "flash_face_mean_r": float(state.last_face_mean_bgr[2]) if state is not None else 0.0,
            "flash_timestamp": float(state.issued_at) if state is not None else 0.0,
            "pre_flash_captured": float(state is not None and state.baseline_frame_bgr.size > 0),
            "flash_frame_captured": float(state.analysis_frames_captured > 0) if state is not None else 0.0,
        }
        persisted_analysis = (
            state.latest_analysis_details
            if state is not None and state.latest_analysis_details
            else self._last_flash_analysis_details
        )
        if persisted_analysis:
            details.update(persisted_analysis)
        if state is not None and state.latest_analysis_details:
            details.update(state.latest_analysis_details)
            details["flash_color_match_score"] = float(state.best_color_match_score)
            details["specular_hotspot_risk"] = float(state.best_specular_hotspot_risk)
            details["diffuse_response_score"] = float(state.best_diffuse_response_score)
            details["geometry_response_consistency"] = float(state.best_geometry_response_consistency)
            details["planar_surface_risk"] = float(state.best_planar_surface_risk)
        return details
    @staticmethod
    def _combine_risks(
        *,
        moire_risk: float,
        reflection_risk: float,
        flicker_risk: float,
        flash_replay_risk: float = 0.0,
        screen_frame_risk: float = 0.0,
        weights: Optional[dict[str, float]] = None,
    ) -> float:
        resolved_weights = {
            "moire": DeviceSpoofRiskEvaluator.DEVICE_REPLAY_MOIRE_WEIGHT,
            "reflection": DeviceSpoofRiskEvaluator.DEVICE_REPLAY_REFLECTION_WEIGHT,
            "flicker": DeviceSpoofRiskEvaluator.DEVICE_REPLAY_FLICKER_WEIGHT,
            "flash": DeviceSpoofRiskEvaluator.DEVICE_REPLAY_FLASH_WEIGHT,
            "screen_frame": DeviceSpoofRiskEvaluator.DEVICE_REPLAY_SCREEN_FRAME_WEIGHT,
        }
        if weights:
            resolved_weights.update({key: max(0.0, float(value)) for key, value in weights.items() if key in resolved_weights})
        return _clamp01(
            resolved_weights["moire"] * moire_risk
            + resolved_weights["reflection"] * reflection_risk
            + resolved_weights["flicker"] * flicker_risk
            + resolved_weights["flash"] * flash_replay_risk
            + resolved_weights["screen_frame"] * screen_frame_risk
        )


def _normalize(value: float, low: float, high: float) -> float:
    if high <= low:
        return 0.0
    return _clamp01((float(value) - low) / (high - low))


def _maybe_float(value: object) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))
