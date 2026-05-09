"""Pixel-based critical face region visibility gate for live preview decisions."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

from app.domain.entities.face_landmarks import LandmarkResult


TEMP_OCCLUSION_FRAMES = 2
PERSISTENT_OCCLUSION_FRAMES = 6
RECOVERY_CLEAR_FRAMES = 4

_DEFAULT_REGION_WEIGHTS = {
    "left_eye": 1.0,
    "right_eye": 1.0,
    "nose": 1.15,
    "mouth": 1.15,
    "lower_face": 1.25,
}
_DEFAULT_REGION_SCORE_THRESHOLD = 0.55
_DEFAULT_OCCLUSION_SCORE_THRESHOLD = 0.32

_REGION_RATIOS = {
    "left_eye": (0.14, 0.24, 0.24, 0.18),
    "right_eye": (0.62, 0.24, 0.24, 0.18),
    "nose": (0.39, 0.35, 0.22, 0.24),
    "mouth": (0.28, 0.63, 0.44, 0.16),
    "lower_face": (0.18, 0.56, 0.64, 0.34),
}
_BASELINE_RATIOS = {
    "forehead": (0.30, 0.08, 0.40, 0.14),
    "left_cheek": (0.14, 0.41, 0.18, 0.18),
    "right_cheek": (0.68, 0.41, 0.18, 0.18),
}


@dataclass(frozen=True)
class CriticalRegionVisibilityResult:
    is_critical_occluded: bool
    occlusion_score: float
    occluded_regions: tuple[str, ...]
    visibility_scores: dict[str, float]
    reason: str


@dataclass(frozen=True)
class CriticalRegionVisibilityState:
    occlusion_streak: int = 0
    clear_streak: int = 0
    recently_occluded: bool = False
    state_name: str = "CLEAR"
    override_status: Optional[str] = None
    override_reason: str = "-"
    last_occluded_regions: tuple[str, ...] = field(default_factory=tuple)
    last_occlusion_score: float = 0.0
    last_visibility_scores: dict[str, float] = field(default_factory=dict)
    is_critical_occluded: bool = False


class CriticalRegionVisibilityGate:
    """Estimate whether critical facial regions are physically visible."""

    def __init__(
        self,
        *,
        region_score_threshold: float = _DEFAULT_REGION_SCORE_THRESHOLD,
        occlusion_score_threshold: float = _DEFAULT_OCCLUSION_SCORE_THRESHOLD,
    ) -> None:
        self._region_score_threshold = region_score_threshold
        self._occlusion_score_threshold = occlusion_score_threshold

    def evaluate(
        self,
        *,
        frame_bgr: np.ndarray,
        face_bounding_box: Optional[tuple[int, int, int, int]],
        landmarks: Optional[LandmarkResult] = None,
        hand_overlap_signal: Optional[float] = None,
        preview_details: Optional[dict[str, object]] = None,
        blur_score: Optional[float] = None,
    ) -> CriticalRegionVisibilityResult:
        if frame_bgr is None or frame_bgr.size == 0 or face_bounding_box is None:
            return CriticalRegionVisibilityResult(
                is_critical_occluded=False,
                occlusion_score=0.0,
                occluded_regions=(),
                visibility_scores={},
                reason="no_face_bbox_unavailable",
            )

        face_roi, clipped_bbox = _crop(frame_bgr, face_bounding_box)
        if face_roi.size == 0:
            return CriticalRegionVisibilityResult(
                is_critical_occluded=False,
                occlusion_score=0.0,
                occluded_regions=(),
                visibility_scores={},
                reason="no_face_bbox_unavailable",
            )

        # Use only cheek patches for baseline — forehead is covered by headscarves/hats
        # and corrupts the baseline Lab color, causing all skin regions to score low.
        baseline_patches = [
            patch
            for patch in (
                _region_patch(face_roi, _BASELINE_RATIOS["left_cheek"]),
                _region_patch(face_roi, _BASELINE_RATIOS["right_cheek"]),
            )
            if patch.size > 0
        ]
        baseline = _baseline_features(baseline_patches or [face_roi])

        visibility_scores: dict[str, float] = {}
        occluded_regions: list[str] = []
        weighted_visibility = 0.0
        total_weight = 0.0

        for region_name, ratios in _REGION_RATIOS.items():
            patch = self._resolve_region_patch(
                frame_bgr=frame_bgr,
                face_roi=face_roi,
                clipped_bbox=clipped_bbox,
                landmarks=landmarks,
                region_name=region_name,
                fallback_ratios=ratios,
            )
            score = _visibility_score(
                patch=patch,
                baseline=baseline,
                hand_overlap_signal=hand_overlap_signal,
                region_name=region_name,
            )
            visibility_scores[region_name] = score
            weight = _DEFAULT_REGION_WEIGHTS[region_name]
            weighted_visibility += score * weight
            total_weight += weight
            if score < self._region_score_threshold:
                occluded_regions.append(region_name)

        visibility_mean = weighted_visibility / max(total_weight, 1e-6)
        occlusion_score = max(0.0, min(1.0, 1.0 - visibility_mean))
        occluded_regions, visibility_scores, occlusion_score = _apply_preview_heuristics(
            occluded_regions=occluded_regions,
            visibility_scores=visibility_scores,
            occlusion_score=occlusion_score,
            preview_details=preview_details,
            blur_score=blur_score,
        )
        is_critical_occluded = bool(
            occluded_regions and occlusion_score >= self._occlusion_score_threshold
        )
        return CriticalRegionVisibilityResult(
            is_critical_occluded=is_critical_occluded,
            occlusion_score=occlusion_score,
            occluded_regions=tuple(occluded_regions),
            visibility_scores=visibility_scores,
            reason="critical_region_visible" if not is_critical_occluded else "critical_region_occluded",
        )

    def _resolve_region_patch(
        self,
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


class CriticalRegionVisibilityTracker:
    """Temporal gate state for occlusion overrides."""

    def __init__(
        self,
        *,
        temp_occlusion_frames: int = TEMP_OCCLUSION_FRAMES,
        persistent_occlusion_frames: int = PERSISTENT_OCCLUSION_FRAMES,
        recovery_clear_frames: int = RECOVERY_CLEAR_FRAMES,
    ) -> None:
        self._temp_occlusion_frames = temp_occlusion_frames
        self._persistent_occlusion_frames = persistent_occlusion_frames
        self._recovery_clear_frames = recovery_clear_frames
        self._state = CriticalRegionVisibilityState()

    def update(
        self,
        assessment: CriticalRegionVisibilityResult,
    ) -> CriticalRegionVisibilityState:
        previous = self._state
        if assessment.is_critical_occluded:
            occlusion_streak = previous.occlusion_streak + 1
            clear_streak = 0
            recently_occluded = True
        else:
            clear_streak = previous.clear_streak + 1
            occlusion_streak = max(0, previous.occlusion_streak - 2)
            recently_occluded = previous.recently_occluded or previous.occlusion_streak >= self._temp_occlusion_frames

        override_status: Optional[str] = None
        override_reason = "-"
        state_name = "CLEAR"
        if assessment.is_critical_occluded and occlusion_streak >= self._persistent_occlusion_frames:
            override_status = "NO_FACE"
            override_reason = "critical_region_persistently_occluded"
            state_name = "PERSISTENT_OCCLUDED"
            recently_occluded = True
        elif assessment.is_critical_occluded and occlusion_streak >= self._temp_occlusion_frames:
            override_status = "INSUFFICIENT_EVIDENCE"
            override_reason = "critical_region_temporarily_occluded"
            state_name = "TEMP_OCCLUDED"
            recently_occluded = True
        elif recently_occluded and clear_streak < self._recovery_clear_frames:
            override_status = "INSUFFICIENT_EVIDENCE"
            override_reason = "recovering_after_critical_region_occlusion"
            state_name = "RECOVERING"
        elif clear_streak >= self._recovery_clear_frames:
            recently_occluded = False

        self._state = CriticalRegionVisibilityState(
            occlusion_streak=occlusion_streak,
            clear_streak=clear_streak,
            recently_occluded=recently_occluded,
            state_name=state_name,
            override_status=override_status,
            override_reason=override_reason,
            last_occluded_regions=assessment.occluded_regions,
            last_occlusion_score=assessment.occlusion_score,
            last_visibility_scores=dict(assessment.visibility_scores),
            is_critical_occluded=assessment.is_critical_occluded,
        )
        return self._state

    def reset(self) -> None:
        self._state = CriticalRegionVisibilityState()

    @property
    def state(self) -> CriticalRegionVisibilityState:
        return self._state


def _crop(
    frame_bgr: np.ndarray,
    bbox: tuple[int, int, int, int],
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    x, y, width, height = bbox
    frame_h, frame_w = frame_bgr.shape[:2]
    x1 = max(0, min(frame_w, int(round(x))))
    y1 = max(0, min(frame_h, int(round(y))))
    x2 = max(x1 + 1, min(frame_w, int(round(x + width))))
    y2 = max(y1 + 1, min(frame_h, int(round(y + height))))
    return frame_bgr[y1:y2, x1:x2], (x1, y1, x2 - x1, y2 - y1)


def _region_patch(face_roi: np.ndarray, ratios: tuple[float, float, float, float]) -> np.ndarray:
    if face_roi.size == 0:
        return np.empty((0, 0, 3), dtype=np.uint8)
    h, w = face_roi.shape[:2]
    rel_x, rel_y, rel_w, rel_h = ratios
    x1 = max(0, min(w, int(round(w * rel_x))))
    y1 = max(0, min(h, int(round(h * rel_y))))
    x2 = max(x1 + 1, min(w, int(round(w * (rel_x + rel_w)))))
    y2 = max(y1 + 1, min(h, int(round(h * (rel_y + rel_h)))))
    return face_roi[y1:y2, x1:x2]


def _landmark_region_patch(
    *,
    frame_bgr: np.ndarray,
    clipped_bbox: tuple[int, int, int, int],
    landmarks: LandmarkResult,
    region_name: str,
) -> np.ndarray:
    region_map = {
        "left_eye": ("left_eye",),
        "right_eye": ("right_eye",),
        "nose": ("nose",),
        "mouth": ("mouth", "inner_lip", "outer_lip"),
        "lower_face": ("mouth", "chin", "jaw"),
    }
    points = []
    for key in region_map.get(region_name, ()):
        indices = landmarks.regions.get(key, [])
        points.extend(
            landmarks.landmarks[index]
            for index in indices
            if 0 <= index < len(landmarks.landmarks)
        )
    if not points:
        return np.empty((0, 0, 3), dtype=np.uint8)

    face_x, face_y, _, _ = clipped_bbox
    xs = [face_x + int(point.x) for point in points]
    ys = [face_y + int(point.y) for point in points]
    x1 = min(xs)
    y1 = min(ys)
    x2 = max(xs)
    y2 = max(ys)
    pad_x = max(2, int((x2 - x1) * 0.25))
    pad_y = max(2, int((y2 - y1) * 0.25))
    return _crop(frame_bgr, (x1 - pad_x, y1 - pad_y, (x2 - x1) + 2 * pad_x, (y2 - y1) + 2 * pad_y))[0]


def _baseline_features(patches: list[np.ndarray]) -> dict[str, float]:
    valid = [patch for patch in patches if patch.size > 0]
    if not valid:
        return {"texture": 1.0, "edge": 0.05, "brightness": 128.0, "lab_l": 60.0, "lab_a": 10.0, "lab_b": 15.0}

    textures = [_laplacian_variance(patch) for patch in valid]
    edges = [_edge_density(patch) for patch in valid]
    brightness = [_brightness(patch) for patch in valid]
    lab_means = [_lab_mean(patch) for patch in valid]
    return {
        "texture": float(np.mean(textures)),
        "edge": float(np.mean(edges)),
        "brightness": float(np.mean(brightness)),
        "lab_l": float(np.mean([lab[0] for lab in lab_means])),
        "lab_a": float(np.mean([lab[1] for lab in lab_means])),
        "lab_b": float(np.mean([lab[2] for lab in lab_means])),
    }


def _visibility_score(
    *,
    patch: np.ndarray,
    baseline: dict[str, float],
    hand_overlap_signal: Optional[float],
    region_name: str = "",
) -> float:
    if patch.size == 0:
        return 0.0

    brightness = _brightness(patch)
    texture = _laplacian_variance(patch)
    edge = _edge_density(patch)
    lab_l, lab_a, lab_b = _lab_mean(patch)

    brightness_score = _band_score(brightness, low=35.0, high=235.0, inner_low=65.0, inner_high=210.0)
    texture_score = _ratio_score(texture, baseline["texture"], stretch=0.65)
    edge_score = _ratio_score(edge, baseline["edge"], stretch=0.75)
    color_distance = float(
        np.linalg.norm(
            np.array([lab_l - baseline["lab_l"], lab_a - baseline["lab_a"], lab_b - baseline["lab_b"]], dtype=np.float32)
        )
    )
    skin_score = max(0.0, 1.0 - min(color_distance / 38.0, 1.0))

    visibility = (
        0.18 * brightness_score
        + 0.40 * texture_score
        + 0.24 * edge_score
        + 0.18 * skin_score
    )

    # Dark non-skin cover: brightness well below baseline AND no skin color match.
    # Texture/edge from a partial-occlusion boundary inflate scores artificially,
    # so we cap visibility when the patch is clearly both dim and non-skin-colored.
    if brightness < baseline["brightness"] * 0.55 and skin_score < 0.05:
        visibility = min(visibility, 0.35)

    if region_name == "mouth":
        # Lips are always redder than surrounding cheek skin in the Lab a* channel.
        # A hand or object covering the mouth shows the same redness as the cheek
        # baseline (delta ≈ 0), while real lips read at least 5+ units above baseline.
        # The generic skin_score is counterproductive here — it rewards a hand (same
        # skin as baseline) more than visible lips (which are distinctly redder).
        lip_redness_delta = lab_a - baseline["lab_a"]
        if lip_redness_delta < 5.0:
            visibility = min(visibility, 0.45)

    if hand_overlap_signal is not None:
        visibility *= max(0.0, min(1.0, 1.0 - hand_overlap_signal))
    return max(0.0, min(1.0, float(visibility)))


def _laplacian_variance(patch: np.ndarray) -> float:
    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _edge_density(patch: np.ndarray) -> float:
    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 60, 140)
    return float(np.count_nonzero(edges)) / max(edges.size, 1)


def _brightness(patch: np.ndarray) -> float:
    return float(np.mean(cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)))


def _lab_mean(patch: np.ndarray) -> tuple[float, float, float]:
    lab = cv2.cvtColor(patch, cv2.COLOR_BGR2LAB)
    mean = cv2.mean(lab)
    return float(mean[0]), float(mean[1]), float(mean[2])


def _ratio_score(value: float, baseline_value: float, *, stretch: float) -> float:
    denominator = max(baseline_value, 1e-6)
    ratio = value / denominator
    return max(0.0, min(1.0, ratio / max(stretch, 1e-6)))


def _band_score(value: float, *, low: float, high: float, inner_low: float, inner_high: float) -> float:
    if low <= value <= high:
        if inner_low <= value <= inner_high:
            return 1.0
        if value < inner_low:
            return max(0.0, min(1.0, (value - low) / max(inner_low - low, 1e-6)))
        return max(0.0, min(1.0, (high - value) / max(high - inner_high, 1e-6)))
    return 0.0


def _apply_preview_heuristics(
    *,
    occluded_regions: list[str],
    visibility_scores: dict[str, float],
    occlusion_score: float,
    preview_details: Optional[dict[str, object]],
    blur_score: Optional[float],
) -> tuple[list[str], dict[str, float], float]:
    if not preview_details:
        return occluded_regions, visibility_scores, occlusion_score

    quality_occlusion = _coerce_float(preview_details.get("quality_occlusion"))
    lower_face_texture = _coerce_float(preview_details.get("preview_lower_face_texture"))

    if lower_face_texture is not None and blur_score is not None and blur_score > 10.0:
        lower_texture_ratio = lower_face_texture / max(blur_score, 1e-6)
        if lower_texture_ratio < 0.95:
            severity = max(0.0, min(1.0, (0.95 - lower_texture_ratio) / 0.55))
            for region_name in ("mouth", "lower_face", "nose"):
                current = visibility_scores.get(region_name, 1.0)
                visibility_scores[region_name] = min(current, max(0.0, 1.0 - severity))
                if visibility_scores[region_name] < _DEFAULT_REGION_SCORE_THRESHOLD and region_name not in occluded_regions:
                    occluded_regions.append(region_name)
            occlusion_score = max(occlusion_score, 0.30 + 0.55 * severity)

    if quality_occlusion is not None and quality_occlusion < 55.0:
        severity = max(0.0, min(1.0, (55.0 - quality_occlusion) / 30.0))
        for region_name in ("nose", "mouth", "lower_face"):
            current = visibility_scores.get(region_name, 1.0)
            visibility_scores[region_name] = min(current, max(0.0, 1.0 - severity))
            if visibility_scores[region_name] < _DEFAULT_REGION_SCORE_THRESHOLD and region_name not in occluded_regions:
                occluded_regions.append(region_name)
        occlusion_score = max(occlusion_score, 0.28 + 0.60 * severity)

    return occluded_regions, visibility_scores, occlusion_score


def _coerce_float(value: object) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
