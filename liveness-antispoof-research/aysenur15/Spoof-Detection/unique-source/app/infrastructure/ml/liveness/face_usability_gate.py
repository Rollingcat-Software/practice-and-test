"""Pre-liveness face usability gate for critical occlusion handling."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from app.domain.entities.face_landmarks import LandmarkResult
from app.infrastructure.ml.liveness.critical_region_visibility_gate import (
    CriticalRegionVisibilityGate,
)

OCCLUSION_CONFIRM_FRAMES = 2
CLEAR_CONFIRM_FRAMES = 3


@dataclass(frozen=True)
class FaceUsabilityResult:
    usable: bool
    no_face: bool
    occluded: bool
    occlusion_score: float
    occluded_regions: tuple[str, ...]
    visibility_scores: dict[str, float]
    reason: str
    state: str
    blocked: bool
    bbox_detected: bool
    occlusion_streak: int = 0
    clear_streak: int = 0


class FaceUsabilityGate:
    """Block liveness scoring when no usable full face is visible."""

    def __init__(
        self,
        *,
        occlusion_confirm_frames: int = OCCLUSION_CONFIRM_FRAMES,
        clear_confirm_frames: int = CLEAR_CONFIRM_FRAMES,
    ) -> None:
        self._visibility_gate = CriticalRegionVisibilityGate()
        self._occlusion_confirm_frames = max(1, int(occlusion_confirm_frames))
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
            self._occlusion_streak = 0
            self._clear_streak = 0
            self._blocked = False
            self._state = "NO_FACE"
            return FaceUsabilityResult(
                usable=False,
                no_face=True,
                occluded=False,
                occlusion_score=0.0,
                occluded_regions=(),
                visibility_scores={},
                reason="no_face_detected",
                state=self._state,
                blocked=True,
                bbox_detected=False,
                occlusion_streak=0,
                clear_streak=0,
            )

        visibility = self._visibility_gate.evaluate(
            frame_bgr=frame,
            face_bounding_box=face_bbox,
            landmarks=landmarks,
            preview_details=preview_details,
            blur_score=blur_score,
        )
        occluded_now = visibility.is_critical_occluded
        if occluded_now:
            self._occlusion_streak += 1
            self._clear_streak = 0
            self._blocked = True
            self._state = (
                "OCCLUDED_CONFIRMED"
                if self._occlusion_streak >= self._occlusion_confirm_frames
                else "OCCLUDED_PENDING"
            )
            return FaceUsabilityResult(
                usable=False,
                no_face=True,
                occluded=True,
                occlusion_score=visibility.occlusion_score,
                occluded_regions=visibility.occluded_regions,
                visibility_scores=dict(visibility.visibility_scores),
                reason="critical_face_region_occluded",
                state=self._state,
                blocked=True,
                bbox_detected=True,
                occlusion_streak=self._occlusion_streak,
                clear_streak=0,
            )

        self._occlusion_streak = 0
        if self._blocked:
            self._clear_streak += 1
            if self._clear_streak < self._clear_confirm_frames:
                self._state = "RECOVERING"
                return FaceUsabilityResult(
                    usable=False,
                    no_face=True,
                    occluded=False,
                    occlusion_score=visibility.occlusion_score,
                    occluded_regions=(),
                    visibility_scores=dict(visibility.visibility_scores),
                    reason="recovering_face_usability",
                    state=self._state,
                    blocked=True,
                    bbox_detected=True,
                    occlusion_streak=0,
                    clear_streak=self._clear_streak,
                )
            self._blocked = False
            self._clear_streak = 0

        self._state = "CLEAR"
        return FaceUsabilityResult(
            usable=True,
            no_face=False,
            occluded=False,
            occlusion_score=visibility.occlusion_score,
            occluded_regions=(),
            visibility_scores=dict(visibility.visibility_scores),
            reason="face_usable",
            state=self._state,
            blocked=False,
            bbox_detected=True,
            occlusion_streak=0,
            clear_streak=self._clear_streak,
        )

    def reset(self) -> None:
        self._occlusion_streak = 0
        self._clear_streak = 0
        self._blocked = False
        self._state = "CLEAR"
