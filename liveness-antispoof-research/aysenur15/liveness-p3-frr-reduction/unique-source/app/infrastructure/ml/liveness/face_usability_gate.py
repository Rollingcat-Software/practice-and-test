"""Pre-liveness face usability gate for critical occlusion handling."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from app.domain.entities.face_landmarks import LandmarkResult
from app.infrastructure.ml.liveness.critical_region_visibility_gate import (
    CriticalRegionVisibilityGate,
)
from app.infrastructure.ml.liveness.face_quality_illumination_gate import (
    FaceQualityIlluminationGate,
)

LOW_QUALITY_CONFIRM_FRAMES = 2
OCCLUSION_CONFIRM_FRAMES = 2
NO_FACE_CONFIRM_FRAMES = 6
TEMP_CLEAR_CONFIRM_FRAMES = 2
CLEAR_CONFIRM_FRAMES = 5

_EYE_STRICT_UNRELIABLE_THRESHOLD = 0.60


@dataclass(frozen=True)
class FaceUsabilityResult:
    usable: bool
    no_face: bool
    quality_ok: bool
    occluded: bool
    occlusion_score: float
    occluded_regions: tuple[str, ...]
    visibility_scores: dict[str, float]
    region_reasons: dict[str, str]
    blocking_regions: tuple[str, ...]
    suspicious_regions: tuple[str, ...]
    quality_status: str
    quality_reason: str
    per_region_brightness: dict[str, float]
    brightness_uniformity: float
    illumination_score: float
    global_face_brightness: float
    shadow_asymmetry: float
    underexposed_regions: tuple[str, ...]
    overexposed_regions: tuple[str, ...]
    physical_occlusion_score: float
    physical_occlusion_regions: tuple[str, ...]
    physical_occlusion_reason: str
    liveness_skipped_reason: str
    reason: str
    state: str
    blocked: bool
    status_override: Optional[str]
    bbox_detected: bool
    occlusion_streak: int = 0
    quality_streak: int = 0
    clear_streak: int = 0


class FaceUsabilityGate:
    """Block liveness scoring when no usable full face is visible."""

    def __init__(
        self,
        *,
        low_quality_confirm_frames: int = LOW_QUALITY_CONFIRM_FRAMES,
        occlusion_confirm_frames: int = OCCLUSION_CONFIRM_FRAMES,
        no_face_confirm_frames: int = NO_FACE_CONFIRM_FRAMES,
        temp_clear_confirm_frames: int = TEMP_CLEAR_CONFIRM_FRAMES,
        clear_confirm_frames: int = CLEAR_CONFIRM_FRAMES,
    ) -> None:
        self._quality_gate = FaceQualityIlluminationGate()
        self._visibility_gate = CriticalRegionVisibilityGate()
        self._low_quality_confirm_frames = max(1, int(low_quality_confirm_frames))
        self._occlusion_confirm_frames = max(1, int(occlusion_confirm_frames))
        self._no_face_confirm_frames = max(self._occlusion_confirm_frames, int(no_face_confirm_frames))
        self._temp_clear_confirm_frames = max(1, int(temp_clear_confirm_frames))
        self._clear_confirm_frames = max(1, int(clear_confirm_frames))
        self.reset()

    def evaluate(
        self,
        *,
        frame: np.ndarray,
        face_bbox: Optional[tuple[int, int, int, int]],
        landmarks: Optional[LandmarkResult] = None,
        preview_details: Optional[dict[str, object]] = None,
        blur_score: Optional[float] = None,
    ) -> FaceUsabilityResult:
        if frame is None or frame.size == 0 or face_bbox is None:
            self._quality_streak = 0
            self._occlusion_streak = 0
            self._clear_streak = 0
            self._blocked = False
            self._state = "NO_FACE"
            return FaceUsabilityResult(
                usable=False,
                no_face=True,
                quality_ok=False,
                occluded=False,
                occlusion_score=0.0,
                occluded_regions=(),
                visibility_scores={},
                region_reasons={},
                blocking_regions=(),
                suspicious_regions=(),
                quality_status="-",
                quality_reason="-",
                per_region_brightness={},
                brightness_uniformity=0.0,
                illumination_score=0.0,
                global_face_brightness=0.0,
                shadow_asymmetry=0.0,
                underexposed_regions=(),
                overexposed_regions=(),
                physical_occlusion_score=0.0,
                physical_occlusion_regions=(),
                physical_occlusion_reason="-",
                liveness_skipped_reason="no_face_detected",
                reason="no_face_detected",
                state=self._state,
                blocked=True,
                status_override="NO_FACE",
                bbox_detected=False,
                occlusion_streak=0,
                quality_streak=0,
                clear_streak=0,
            )

        quality = self._quality_gate.evaluate(
            frame_bgr=frame,
            face_bounding_box=face_bbox,
            landmarks=landmarks,
        )
        visibility = self._visibility_gate.evaluate(
            frame_bgr=frame,
            face_bounding_box=face_bbox,
            landmarks=landmarks,
            preview_details=preview_details,
            blur_score=blur_score,
        )
        left_eye_score = visibility.visibility_scores.get("left_eye", 1.0)
        right_eye_score = visibility.visibility_scores.get("right_eye", 1.0)
        left_eye_visible = left_eye_score >= 0.60
        right_eye_visible = right_eye_score >= 0.60
        nose_visible = visibility.visibility_scores.get("nose", 1.0) >= 0.65
        mouth_visible = visibility.visibility_scores.get("mouth", 1.0) >= 0.65
        lower_face_visible = visibility.visibility_scores.get("lower_face", 1.0) >= 0.60
        both_eyes_unreliable = bool(
            left_eye_score < _EYE_STRICT_UNRELIABLE_THRESHOLD
            and right_eye_score < _EYE_STRICT_UNRELIABLE_THRESHOLD
        )
        derived_occluded_regions: list[str] = list(visibility.occluded_regions)
        if both_eyes_unreliable or (not left_eye_visible and not right_eye_visible):
            derived_occluded_regions.extend(["left_eye", "right_eye"])
        if not nose_visible:
            derived_occluded_regions.append("nose")
        if not mouth_visible:
            derived_occluded_regions.append("mouth")
        if not lower_face_visible:
            derived_occluded_regions.append("lower_face")
        derived_occluded_regions = list(dict.fromkeys(derived_occluded_regions))

        structural_occlusion_now = bool(
            both_eyes_unreliable
            or (not left_eye_visible and not right_eye_visible)
            or (not nose_visible)
            or (not mouth_visible and not lower_face_visible)
            or (
                visibility.occlusion_score >= 0.58
                and ((not mouth_visible) or (not nose_visible) or (not lower_face_visible))
            )
        )
        occluded_now = bool(visibility.is_critical_occluded or structural_occlusion_now)
        if occluded_now:
            self._quality_streak = 0
            self._occlusion_streak += 1
            self._clear_streak = 0
            self._blocked = True
            self._blocked_mode = "OCCLUSION"
            if self._occlusion_streak >= self._no_face_confirm_frames:
                self._entered_no_face = True
                self._state = "OCCLUDED_NO_FACE"
                status_override = "NO_FACE"
            elif self._occlusion_streak >= self._occlusion_confirm_frames:
                self._state = "OCCLUDED_CONFIRMED"
                status_override = "INSUFFICIENT_EVIDENCE"
            else:
                self._state = "OCCLUDED_PENDING"
                status_override = "INSUFFICIENT_EVIDENCE"
            return FaceUsabilityResult(
                usable=False,
                no_face=True,
                quality_ok=True,
                occluded=True,
                occlusion_score=visibility.occlusion_score,
                occluded_regions=tuple(derived_occluded_regions),
                visibility_scores=dict(visibility.visibility_scores),
                region_reasons=dict(visibility.region_reasons),
                blocking_regions=tuple(derived_occluded_regions),
                suspicious_regions=tuple(
                    region
                    for region in dict.fromkeys([*visibility.suspicious_regions, *derived_occluded_regions])
                    if region not in derived_occluded_regions
                ),
                quality_status=quality.quality_status,
                quality_reason=quality.quality_reason,
                per_region_brightness=dict(quality.per_region_brightness),
                brightness_uniformity=quality.brightness_uniformity,
                illumination_score=quality.illumination_score,
                global_face_brightness=quality.global_face_brightness,
                shadow_asymmetry=quality.shadow_asymmetry,
                underexposed_regions=quality.underexposed_regions,
                overexposed_regions=quality.overexposed_regions,
                physical_occlusion_score=visibility.occlusion_score,
                physical_occlusion_regions=tuple(derived_occluded_regions),
                physical_occlusion_reason=(
                    visibility.reason
                    if visibility.is_critical_occluded
                    else "structural_face_region_occluded"
                ),
                liveness_skipped_reason="critical_face_region_occluded",
                reason="critical_face_region_occluded",
                state=self._state,
                blocked=True,
                status_override=status_override,
                bbox_detected=True,
                occlusion_streak=self._occlusion_streak,
                quality_streak=0,
                clear_streak=0,
            )

        self._occlusion_streak = 0
        self._quality_streak = 0
        if self._blocked:
            self._clear_streak += 1
            required_clear_frames = (
                self._clear_confirm_frames
                if self._entered_no_face or self._blocked_mode == "LOW_QUALITY"
                else self._temp_clear_confirm_frames
            )
            if self._clear_streak < required_clear_frames:
                self._state = "RECOVERING"
                return FaceUsabilityResult(
                    usable=False,
                    no_face=False,
                    quality_ok=True,
                    occluded=False,
                    occlusion_score=visibility.occlusion_score,
                    occluded_regions=(),
                    visibility_scores=dict(visibility.visibility_scores),
                    region_reasons=dict(visibility.region_reasons),
                    blocking_regions=visibility.blocking_regions,
                    suspicious_regions=visibility.suspicious_regions,
                    quality_status=quality.quality_status,
                    quality_reason=quality.quality_reason,
                    per_region_brightness=dict(quality.per_region_brightness),
                    brightness_uniformity=quality.brightness_uniformity,
                    illumination_score=quality.illumination_score,
                    global_face_brightness=quality.global_face_brightness,
                    shadow_asymmetry=quality.shadow_asymmetry,
                    underexposed_regions=quality.underexposed_regions,
                    overexposed_regions=quality.overexposed_regions,
                    physical_occlusion_score=visibility.occlusion_score,
                    physical_occlusion_regions=visibility.occluded_regions,
                    physical_occlusion_reason=visibility.reason,
                    liveness_skipped_reason="recovering_face_usability",
                    reason="recovering_face_usability",
                    state=self._state,
                    blocked=True,
                    status_override="INSUFFICIENT_EVIDENCE",
                    bbox_detected=True,
                    occlusion_streak=0,
                    quality_streak=0,
                    clear_streak=self._clear_streak,
                )
            self._blocked = False
            self._entered_no_face = False
            self._blocked_mode = None
            self._clear_streak = 0

        self._state = "CLEAR"
        return FaceUsabilityResult(
            usable=True,
            no_face=False,
            quality_ok=True,
            occluded=False,
            occlusion_score=visibility.occlusion_score,
            occluded_regions=(),
            visibility_scores=dict(visibility.visibility_scores),
            region_reasons=dict(visibility.region_reasons),
            blocking_regions=visibility.blocking_regions,
            suspicious_regions=visibility.suspicious_regions,
            quality_status=quality.quality_status,
            quality_reason=quality.quality_reason,
            per_region_brightness=dict(quality.per_region_brightness),
            brightness_uniformity=quality.brightness_uniformity,
            illumination_score=quality.illumination_score,
            global_face_brightness=quality.global_face_brightness,
            shadow_asymmetry=quality.shadow_asymmetry,
            underexposed_regions=quality.underexposed_regions,
            overexposed_regions=quality.overexposed_regions,
            physical_occlusion_score=visibility.occlusion_score,
            physical_occlusion_regions=visibility.occluded_regions,
            physical_occlusion_reason=visibility.reason,
            liveness_skipped_reason="-",
            reason="face_usable",
            state=self._state,
            blocked=False,
            status_override=None,
            bbox_detected=True,
            occlusion_streak=0,
            quality_streak=0,
            clear_streak=self._clear_streak,
        )

    def reset(self) -> None:
        self._quality_streak = 0
        self._occlusion_streak = 0
        self._clear_streak = 0
        self._blocked = False
        self._blocked_mode: Optional[str] = None
        self._entered_no_face = False
        self._state = "CLEAR"
