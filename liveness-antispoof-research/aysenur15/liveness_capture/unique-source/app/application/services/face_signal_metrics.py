"""Helpers for extracting EAR/MAR/pose-oriented face signal metrics."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Sequence

import cv2
import numpy as np

from app.domain.entities.face_landmarks import Landmark, LandmarkResult
from app.domain.exceptions.feature_errors import LandmarkError
from app.domain.interfaces.landmark_detector import ILandmarkDetector

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FaceSignalMetrics:
    """Per-frame face geometry and motion-oriented metrics."""

    face_detected: bool
    face_quality: Optional[float]
    blur_score: Optional[float]
    brightness: Optional[float]
    ear_current: Optional[float]
    mar_current: Optional[float]
    yaw_current: Optional[float]
    pitch_current: Optional[float]
    roll_current: Optional[float]
    depth_range: Optional[float]
    depth_std: Optional[float]
    nose_cheek_depth_delta: Optional[float]
    cheek_depth_asymmetry: Optional[float]
    depth_flat_risk: Optional[float]
    landmark_model: Optional[str] = None

    def to_dict(self) -> dict[str, float | bool | str | None]:
        """Serialize to a flat calibration/debug payload."""
        return {
            "face_detected": self.face_detected,
            "face_quality": self.face_quality,
            "blur_score": self.blur_score,
            "brightness": self.brightness,
            "ear_current": self.ear_current,
            "mar_current": self.mar_current,
            "yaw_current": self.yaw_current,
            "pitch_current": self.pitch_current,
            "roll_current": self.roll_current,
            "depth_range": self.depth_range,
            "depth_std": self.depth_std,
            "nose_cheek_depth_delta": self.nose_cheek_depth_delta,
            "cheek_depth_asymmetry": self.cheek_depth_asymmetry,
            "depth_flat_risk": self.depth_flat_risk,
            "landmark_model": self.landmark_model,
        }


def extract_face_signal_metrics(
    *,
    face_region_bgr: np.ndarray,
    landmark_detector: Optional[ILandmarkDetector],
    face_quality: Optional[float] = None,
    blur_score: Optional[float] = None,
    brightness: Optional[float] = None,
) -> FaceSignalMetrics:
    """Extract EAR, MAR, and head pose from a face crop using the configured landmark detector."""
    resolved_blur = blur_score if blur_score is not None else _compute_blur(face_region_bgr)
    resolved_brightness = brightness if brightness is not None else _compute_brightness(face_region_bgr)

    if landmark_detector is None:
        return FaceSignalMetrics(
            face_detected=True,
            face_quality=face_quality,
            blur_score=resolved_blur,
            brightness=resolved_brightness,
            ear_current=None,
            mar_current=None,
            yaw_current=None,
            pitch_current=None,
            roll_current=None,
            depth_range=None,
            depth_std=None,
            nose_cheek_depth_delta=None,
            cheek_depth_asymmetry=None,
            depth_flat_risk=None,
            landmark_model=None,
        )

    try:
        rgb_face = cv2.cvtColor(face_region_bgr, cv2.COLOR_BGR2RGB)
        landmark_result = landmark_detector.detect(rgb_face, include_3d=True)
    except LandmarkError as exc:
        logger.debug("Landmark extraction skipped: %s", exc)
        return FaceSignalMetrics(
            face_detected=True,
            face_quality=face_quality,
            blur_score=resolved_blur,
            brightness=resolved_brightness,
            ear_current=None,
            mar_current=None,
            yaw_current=None,
            pitch_current=None,
            roll_current=None,
            depth_range=None,
            depth_std=None,
            nose_cheek_depth_delta=None,
            cheek_depth_asymmetry=None,
            depth_flat_risk=None,
            landmark_model=None,
        )
    except Exception as exc:
        logger.debug("Unexpected landmark extraction failure: %s", exc)
        return FaceSignalMetrics(
            face_detected=True,
            face_quality=face_quality,
            blur_score=resolved_blur,
            brightness=resolved_brightness,
            ear_current=None,
            mar_current=None,
            yaw_current=None,
            pitch_current=None,
            roll_current=None,
            depth_range=None,
            depth_std=None,
            nose_cheek_depth_delta=None,
            cheek_depth_asymmetry=None,
            depth_flat_risk=None,
            landmark_model=None,
        )

    ear_current = _compute_ear(landmark_result)
    mar_current = _compute_mar(landmark_result)
    yaw_current = landmark_result.head_pose.yaw if landmark_result.head_pose else None
    pitch_current = landmark_result.head_pose.pitch if landmark_result.head_pose else None
    roll_current = landmark_result.head_pose.roll if landmark_result.head_pose else None
    depth_range, depth_std, nose_cheek_depth_delta, cheek_depth_asymmetry, depth_flat_risk = (
        _compute_depth_profile_metrics(landmark_result)
    )

    return FaceSignalMetrics(
        face_detected=True,
        face_quality=face_quality,
        blur_score=resolved_blur,
        brightness=resolved_brightness,
        ear_current=ear_current,
        mar_current=mar_current,
        yaw_current=yaw_current,
        pitch_current=pitch_current,
        roll_current=roll_current,
        depth_range=depth_range,
        depth_std=depth_std,
        nose_cheek_depth_delta=nose_cheek_depth_delta,
        cheek_depth_asymmetry=cheek_depth_asymmetry,
        depth_flat_risk=depth_flat_risk,
        landmark_model=landmark_result.model,
    )


def _compute_blur(face_region_bgr: np.ndarray) -> float:
    gray = cv2.cvtColor(face_region_bgr, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _compute_brightness(face_region_bgr: np.ndarray) -> float:
    return float(np.mean(cv2.cvtColor(face_region_bgr, cv2.COLOR_BGR2GRAY)))


def _compute_ear(result: LandmarkResult) -> Optional[float]:
    left_eye = _region_points(result, "left_eye")
    right_eye = _region_points(result, "right_eye")
    left_ear = _compute_region_ear(left_eye)
    right_ear = _compute_region_ear(right_eye)
    values = [value for value in (left_ear, right_ear) if value is not None]
    if not values:
        return None
    return round(float(np.mean(values)), 4)


def _compute_mar(result: LandmarkResult) -> Optional[float]:
    if result.model == "dlib_68":
        inner_lip = _region_points(result, "inner_lip")
        outer_lip = _region_points(result, "outer_lip")
        points = inner_lip or outer_lip
        if len(points) < 8:
            return None
        left = points[0]
        right = points[4]
        top_candidates = [points[1], points[2], points[3]]
        bottom_candidates = [points[7], points[6], points[5]]
    else:
        mouth = _region_points(result, "mouth")
        if len(mouth) < 6:
            return None
        left = min(mouth, key=lambda point: point.x)
        right = max(mouth, key=lambda point: point.x)
        mouth_sorted = sorted(mouth, key=lambda point: point.y)
        top_candidates = mouth_sorted[: max(1, len(mouth_sorted) // 3)]
        bottom_candidates = mouth_sorted[-max(1, len(mouth_sorted) // 3) :]

    horizontal = _distance(left, right)
    if horizontal <= 1e-6:
        return None
    vertical = float(
        np.mean([
            _distance(top, bottom)
            for top, bottom in zip(top_candidates, reversed(bottom_candidates))
        ])
    )
    return round(vertical / horizontal, 4)


def _compute_region_ear(points: Sequence[Landmark]) -> Optional[float]:
    if len(points) < 6:
        return None

    if len(points) == 6:
        p1, p2, p3, p4, p5, p6 = points
        vertical_pairs = ((p2, p6), (p3, p5))
    else:
        left = min(points, key=lambda point: point.x)
        right = max(points, key=lambda point: point.x)
        remaining = sorted(
            [point for point in points if point is not left and point is not right],
            key=lambda point: point.y,
        )
        half = len(remaining) // 2
        top_points = remaining[:half]
        bottom_points = remaining[half:]
        if not top_points or not bottom_points:
            return None
        p1, p4 = left, right
        vertical_pairs = list(zip(top_points, reversed(bottom_points)))
        if not vertical_pairs:
            return None
        vertical = float(np.mean([_distance(top, bottom) for top, bottom in vertical_pairs]))
        horizontal = _distance(p1, p4)
        if horizontal <= 1e-6:
            return None
        return round(vertical / horizontal, 4)

    horizontal = _distance(p1, p4)
    if horizontal <= 1e-6:
        return None
    vertical = float(np.mean([_distance(a, b) for a, b in vertical_pairs]))
    return round(vertical / horizontal, 4)


def _region_points(result: LandmarkResult, region_name: str) -> list[Landmark]:
    region_indices = result.regions.get(region_name, [])
    if not region_indices:
        return []
    return [result.landmarks[index] for index in region_indices if index < len(result.landmarks)]


def _distance(a: Landmark, b: Landmark) -> float:
    return float(np.hypot(a.x - b.x, a.y - b.y))


def _compute_depth_profile_metrics(
    result: LandmarkResult,
) -> tuple[Optional[float], Optional[float], Optional[float], Optional[float], Optional[float]]:
    if "mediapipe" not in result.model:
        return None, None, None, None, None

    sample_indices = {
        "nose": 1,
        "left_cheek": 234,
        "right_cheek": 454,
        "forehead": 10,
        "chin": 152,
    }
    points: dict[str, Landmark] = {}
    for name, index in sample_indices.items():
        if index >= len(result.landmarks):
            return None, None, None, None, None
        point = result.landmarks[index]
        if point.z is None:
            return None, None, None, None, None
        points[name] = point

    z_values = np.array([float(point.z) for point in points.values()], dtype=np.float32)
    depth_range = float(np.max(z_values) - np.min(z_values))
    depth_std = float(np.std(z_values))
    nose_cheek_depth_delta = float(
        abs(points["nose"].z - ((points["left_cheek"].z + points["right_cheek"].z) / 2.0))
    )
    cheek_depth_asymmetry = float(abs(points["left_cheek"].z - points["right_cheek"].z))

    flat_risk = (
        0.45 * _inverse_normalize(depth_range, 0.035, 0.16)
        + 0.35 * _inverse_normalize(nose_cheek_depth_delta, 0.015, 0.075)
        + 0.20 * _inverse_normalize(cheek_depth_asymmetry, 0.010, 0.055)
    )

    return (
        round(depth_range, 4),
        round(depth_std, 4),
        round(nose_cheek_depth_delta, 4),
        round(cheek_depth_asymmetry, 4),
        round(max(0.0, min(1.0, flat_risk)), 4),
    )


def _inverse_normalize(value: float, low: float, high: float) -> float:
    if high <= low:
        return 0.0
    normalized = (float(value) - low) / (high - low)
    return max(0.0, min(1.0, 1.0 - normalized))
