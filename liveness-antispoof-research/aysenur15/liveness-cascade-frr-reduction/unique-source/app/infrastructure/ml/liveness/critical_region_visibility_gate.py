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
_EYE_VISIBILITY_THRESHOLD = 0.60
_NOSE_VISIBILITY_THRESHOLD = 0.65
_MOUTH_VISIBILITY_THRESHOLD = 0.45  # Lowered from 0.65: allow slightly covered mouth, but reject fully covered
_LOWER_FACE_VISIBILITY_THRESHOLD = 0.58
_MOUTH_REDNESS_OCCLUDED_DELTA = 3.5
_MOUTH_REDNESS_WARNING_DELTA = 6.0
_LOWER_FACE_TEXTURE_OCCLUDED_RATIO = 0.62
_LOWER_FACE_TEXTURE_WARNING_RATIO = 0.78
_QUALITY_OCCLUSION_THRESHOLD = 42.0
_PHYSICAL_OCCLUSION_REASON_TOKENS = frozenset(
    {
        "dark_occluding_surface",
        "hand_overlap_signal",
        "eye_occluded",
        "lip_color_signature_missing",
        "mouth_structure_weakened",
        "mouth_replaced_by_skin_like_surface",
        "nose_replaced_by_skin_like_surface",
        "nose_structure_missing",
        # "mouth_roi_color_invalid" excluded: fires when hijab/headscarf border enters the mouth ROI,
        #   causing false occlusion for covered-hair users whose mouth is fully visible.
        #   Stronger tokens (lip_color_signature_missing, mouth_structure_weakened) are sufficient.
        # "mouth_chrominance_anomaly" excluded: same issue — hijab fabric at face border causes
        #   chrominance shift without true mouth occlusion.
        # "facial_detail_missing" excluded: generic quality signal — fires under shadow/low contrast
        # "lower_face_texture_drop" excluded: relative heuristic — unreliable under uneven illumination
    }
)

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
    region_reasons: dict[str, str]
    blocking_regions: tuple[str, ...]
    suspicious_regions: tuple[str, ...]
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
        self._clear_reference_features: dict[str, dict[str, float]] = {}

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
                region_reasons={},
                blocking_regions=(),
                suspicious_regions=(),
                reason="no_face_bbox_unavailable",
            )

        face_roi, clipped_bbox = _crop(frame_bgr, face_bounding_box)
        if face_roi.size == 0:
            return CriticalRegionVisibilityResult(
                is_critical_occluded=False,
                occlusion_score=0.0,
                occluded_regions=(),
                visibility_scores={},
                region_reasons={},
                blocking_regions=(),
                suspicious_regions=(),
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
        region_reasons: dict[str, str] = {}
        region_features: dict[str, dict[str, float]] = {}
        threshold_failed_regions: list[str] = []
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
            score, region_reason, features = _visibility_score(
                patch=patch,
                baseline=baseline,
                hand_overlap_signal=hand_overlap_signal,
                region_name=region_name,
                reference_features=self._clear_reference_features.get(region_name),
            )
            visibility_scores[region_name] = score
            region_reasons[region_name] = region_reason
            region_features[region_name] = features
            weight = _DEFAULT_REGION_WEIGHTS[region_name]
            weighted_visibility += score * weight
            total_weight += weight
            if score < _region_visibility_threshold(region_name):
                threshold_failed_regions.append(region_name)

        visibility_mean = weighted_visibility / max(total_weight, 1e-6)
        occlusion_score = max(0.0, min(1.0, 1.0 - visibility_mean))
        threshold_failed_regions, visibility_scores, region_reasons, occlusion_score = _apply_preview_heuristics(
            occluded_regions=threshold_failed_regions,
            visibility_scores=visibility_scores,
            region_reasons=region_reasons,
            occlusion_score=occlusion_score,
            preview_details=preview_details,
            blur_score=blur_score,
        )
        blocking_regions, suspicious_regions = _classify_region_states(
            visibility_scores=visibility_scores,
            region_reasons=region_reasons,
            threshold_failed_regions=threshold_failed_regions,
        )
        is_critical_occluded = _is_critical_occluded(
            blocking_regions=blocking_regions,
            suspicious_regions=suspicious_regions,
            region_reasons=region_reasons,
            visibility_scores=visibility_scores,
            occlusion_score=occlusion_score,
            occlusion_score_threshold=self._occlusion_score_threshold,
        )
        if not is_critical_occluded:
            for region_name in ("left_eye", "right_eye", "nose", "mouth", "lower_face"):
                region_feature = region_features.get(region_name)
                if region_feature:
                    self._clear_reference_features[region_name] = dict(region_feature)
        return CriticalRegionVisibilityResult(
            is_critical_occluded=is_critical_occluded,
            occlusion_score=occlusion_score,
            occluded_regions=tuple(blocking_regions),
            visibility_scores=visibility_scores,
            region_reasons=region_reasons,
            blocking_regions=tuple(blocking_regions),
            suspicious_regions=tuple(suspicious_regions),
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
        elif not assessment.is_critical_occluded and recently_occluded and clear_streak < self._recovery_clear_frames:
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
    reference_features: Optional[dict[str, float]],
    region_name: str = "",
) -> tuple[float, str, dict[str, float]]:
    if patch.size == 0:
        return 0.0, "patch_missing", {"visibility": 0.0}

    brightness = _brightness(patch)
    texture = _laplacian_variance(patch)
    edge = _edge_density(patch)
    gray_std = _gray_std(patch)
    clahe_texture = texture
    clahe_edge = edge
    clahe_std = gray_std
    lab_l, lab_a, lab_b = _lab_mean(patch)

    if region_name in {"left_eye", "right_eye"}:
        clahe_gray = _clahe_gray(patch)
        clahe_texture = float(cv2.Laplacian(clahe_gray, cv2.CV_64F).var())
        clahe_edges = cv2.Canny(clahe_gray, 40, 120)
        clahe_edge = float(np.count_nonzero(clahe_edges)) / max(clahe_edges.size, 1)
        clahe_std = float(np.std(clahe_gray))
        texture = max(texture, clahe_texture)
        edge = max(edge, clahe_edge)
        gray_std = max(gray_std, clahe_std)

    brightness_score = _band_score(brightness, low=35.0, high=235.0, inner_low=65.0, inner_high=210.0)
    texture_score = _ratio_score(texture, baseline["texture"], stretch=0.65)
    edge_score = _ratio_score(edge, baseline["edge"], stretch=0.75)
    uniformity_score = max(0.0, min(1.0, gray_std / 28.0))
    color_distance = float(
        np.linalg.norm(
            np.array([lab_l - baseline["lab_l"], lab_a - baseline["lab_a"], lab_b - baseline["lab_b"]], dtype=np.float32)
        )
    )
    skin_score = max(0.0, 1.0 - min(color_distance / 38.0, 1.0))
    detail_score = 0.52 * texture_score + 0.30 * edge_score + 0.18 * uniformity_score

    visibility = (
        0.18 * brightness_score
        + 0.32 * texture_score
        + 0.20 * edge_score
        + 0.12 * uniformity_score
        + 0.18 * skin_score
    )
    reasons: list[str] = []

    if detail_score < 0.46:
        visibility = min(visibility, 0.40)
        reasons.append("facial_detail_missing")

    # Darkness alone is a face-quality problem, not proof of physical occlusion.
    # Treat it as occlusion only when the region is also structurally missing after
    # normalization, which indicates an external cover instead of shadow.
    if brightness < baseline["brightness"] * 0.55 and skin_score < 0.08:
        if detail_score >= 0.50:
            visibility = max(visibility, 0.56)
            reasons.append("poor_region_illumination")
        elif gray_std < 12.0:
            visibility = min(visibility, 0.32)
            reasons.append("dark_occluding_surface")

    if region_name in {"left_eye", "right_eye"}:
        face_brightness = max(baseline["brightness"], 1e-6)
        eye_brightness_ratio = brightness / face_brightness
        if eye_brightness_ratio < 0.82 and detail_score >= 0.54:
            visibility = max(visibility, 0.62 if eye_brightness_ratio >= 0.60 else 0.58)
            reasons.append("eye_low_light_warning")
        if skin_score > 0.78 and detail_score < 0.50 and uniformity_score < 0.60:
            visibility = min(visibility, 0.34)
            reasons.append("eye_occluded")
        elif detail_score < 0.34 and gray_std < 12.0:
            visibility = min(visibility, 0.38)
            reasons.append("eye_occluded")

    if region_name == "mouth":
        # Lips are always redder than surrounding cheek skin in the Lab a* channel.
        # A hand or object covering the mouth shows the same redness as the cheek
        # baseline (delta ≈ 0), while real lips read at least 5+ units above baseline.
        # The generic skin_score is counterproductive here — it rewards a hand (same
        # skin as baseline) more than visible lips (which are distinctly redder).
        lip_redness_delta = lab_a - baseline["lab_a"]
        brightness_ratio = brightness / max(baseline["brightness"], 1e-6)
        texture_is_flat = texture_score < 0.72
        edge_is_flat = edge_score < 0.72
        if (
            lip_redness_delta < _MOUTH_REDNESS_OCCLUDED_DELTA
            and texture_is_flat
            and edge_is_flat
        ):
            visibility = min(visibility, 0.18)
            reasons.append("lip_color_signature_missing")
        elif lip_redness_delta < _MOUTH_REDNESS_WARNING_DELTA and (texture_is_flat or edge_is_flat):
            # Redness drop + flat texture/edge = hand or surface covering mouth
            visibility = min(visibility, 0.42)
            reasons.append("mouth_structure_weakened")

        if (
            brightness_score >= 0.45
            and brightness_ratio >= 0.72
            and skin_score > 0.82
            and detail_score < 0.68
            and gray_std < 18.0
        ):
            visibility = min(visibility, 0.20)
            reasons.append("mouth_replaced_by_skin_like_surface")

        # HSV and chrominance checks — only on real skin baseline (lab_a > 127 = reddish skin tone).
        # Synthetic / non-skin-colored frames are skipped to avoid false positives in tests.
        if baseline["lab_a"] > 127 and brightness > 60.0:
            color_valid, color_confidence = _mouth_hsv_color_validity(patch)
            if not color_valid and color_confidence < 0.15:
                visibility = min(visibility, 0.20)
                reasons.append("mouth_roi_color_invalid")

            if not _mouth_cr_in_skin_range(patch):
                visibility = min(visibility, 0.30)
                reasons.append("mouth_chrominance_anomaly")

    if region_name == "nose":
        brightness_ratio = brightness / max(baseline["brightness"], 1e-6)
        if (
            brightness_score >= 0.45
            and brightness_ratio >= 0.72
            and skin_score > 0.84
            and detail_score < 0.70
            and gray_std < 18.0
        ):
            visibility = min(visibility, 0.22)
            reasons.append("nose_replaced_by_skin_like_surface")
        if (
            brightness_score >= 0.55
            and brightness_ratio >= 0.68
            and texture_score < 0.60
            and edge_score < 0.58
            and gray_std < 22.0  # physical cover is flatter than nose bridge; shadow preserves 3D gradient
        ):
            visibility = min(visibility, 0.34)
            reasons.append("nose_structure_missing")

    if region_name in {"mouth", "nose"} and reference_features:
        reference_texture_ratio = texture / max(reference_features.get("texture", texture), 1e-6)
        reference_edge_ratio = edge / max(reference_features.get("edge", edge), 1e-6)
        reference_std_ratio = gray_std / max(reference_features.get("gray_std", gray_std), 1e-6)
        if reference_texture_ratio < 0.58 and reference_edge_ratio < 0.62:
            visibility = min(visibility, 0.28 if region_name == "mouth" else 0.32)
            reasons.append("detail_drop_vs_clear_face")
        if reference_std_ratio < 0.65 and detail_score < 0.74:
            visibility = min(visibility, 0.36)
            reasons.append("uniform_surface_vs_clear_face")

    if hand_overlap_signal is not None:
        visibility *= max(0.0, min(1.0, 1.0 - hand_overlap_signal))
        if hand_overlap_signal > 0.20:
            reasons.append("hand_overlap_signal")
    visibility = max(0.0, min(1.0, float(visibility)))
    features = {
        "visibility": visibility,
        "brightness": float(brightness),
        "texture": float(texture),
        "edge": float(edge),
        "gray_std": float(gray_std),
        "clahe_texture": float(clahe_texture),
        "clahe_edge": float(clahe_edge),
        "clahe_std": float(clahe_std),
        "brightness_score": float(brightness_score),
        "texture_score": float(texture_score),
        "edge_score": float(edge_score),
        "uniformity_score": float(uniformity_score),
        "detail_score": float(detail_score),
        "color_distance": float(color_distance),
        "skin_score": float(skin_score),
    }
    return visibility, "|".join(dict.fromkeys(reasons)) if reasons else "region_visible", features


def _laplacian_variance(patch: np.ndarray) -> float:
    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _edge_density(patch: np.ndarray) -> float:
    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 60, 140)
    return float(np.count_nonzero(edges)) / max(edges.size, 1)


def _gray_std(patch: np.ndarray) -> float:
    return float(np.std(cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)))


def _clahe_gray(patch: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    return clahe.apply(gray)


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
    region_reasons: dict[str, str],
    occlusion_score: float,
    preview_details: Optional[dict[str, object]],
    blur_score: Optional[float],
) -> tuple[list[str], dict[str, float], dict[str, str], float]:
    if not preview_details:
        return occluded_regions, visibility_scores, region_reasons, occlusion_score

    quality_occlusion = _coerce_float(preview_details.get("quality_occlusion"))
    lower_face_texture = _coerce_float(preview_details.get("preview_lower_face_texture"))

    if lower_face_texture is not None and blur_score is not None and blur_score > 10.0:
        lower_texture_ratio = lower_face_texture / max(blur_score, 1e-6)
        if lower_texture_ratio < _LOWER_FACE_TEXTURE_WARNING_RATIO:
            if lower_texture_ratio < _LOWER_FACE_TEXTURE_OCCLUDED_RATIO:
                severity = max(
                    0.0,
                    min(
                        1.0,
                        (_LOWER_FACE_TEXTURE_OCCLUDED_RATIO - lower_texture_ratio)
                        / max(_LOWER_FACE_TEXTURE_OCCLUDED_RATIO, 1e-6),
                    ),
                )
                affected_regions = ("mouth", "lower_face", "nose")
                visibility_floor = 0.18
                score_floor = 0.38
            else:
                severity = max(
                    0.0,
                    min(
                        1.0,
                        (_LOWER_FACE_TEXTURE_WARNING_RATIO - lower_texture_ratio)
                        / max(
                            _LOWER_FACE_TEXTURE_WARNING_RATIO - _LOWER_FACE_TEXTURE_OCCLUDED_RATIO,
                            1e-6,
                        ),
                    ),
                )
                affected_regions = ("mouth", "lower_face")
                visibility_floor = 0.42
                score_floor = 0.24
            for region_name in affected_regions:
                current = visibility_scores.get(region_name, 1.0)
                visibility_scores[region_name] = min(
                    current,
                    max(visibility_floor, 1.0 - severity),
                )
                if visibility_scores[region_name] < _region_visibility_threshold(region_name) and region_name not in occluded_regions:
                    occluded_regions.append(region_name)
                if visibility_scores[region_name] < _region_visibility_threshold(region_name):
                    region_reasons[region_name] = _merge_reason(region_reasons.get(region_name), "lower_face_texture_drop")
            occlusion_score = max(occlusion_score, score_floor + 0.45 * severity)

    if quality_occlusion is not None and quality_occlusion < _QUALITY_OCCLUSION_THRESHOLD:
        severity = max(
            0.0,
            min(1.0, (_QUALITY_OCCLUSION_THRESHOLD - quality_occlusion) / 24.0),
        )
        for region_name in ("nose", "mouth", "lower_face"):
            current = visibility_scores.get(region_name, 1.0)
            visibility_scores[region_name] = min(current, max(0.0, 1.0 - severity))
            if visibility_scores[region_name] < _region_visibility_threshold(region_name) and region_name not in occluded_regions:
                occluded_regions.append(region_name)
            if visibility_scores[region_name] < _region_visibility_threshold(region_name):
                region_reasons[region_name] = _merge_reason(region_reasons.get(region_name), "quality_occlusion_signal")
        occlusion_score = max(occlusion_score, 0.32 + 0.50 * severity)

    return occluded_regions, visibility_scores, region_reasons, occlusion_score


def _is_critical_occluded(
    *,
    blocking_regions: list[str],
    suspicious_regions: list[str],
    region_reasons: dict[str, str],
    visibility_scores: dict[str, float],
    occlusion_score: float,
    occlusion_score_threshold: float,
) -> bool:
    del occlusion_score, occlusion_score_threshold
    blocked = set(blocking_regions)
    suspicious = set(suspicious_regions)

    # Both eyes physically blocked together.
    if {"left_eye", "right_eye"}.issubset(blocked):
        return True

    # Mouth blocked by a physical signal — always critical (scarf pulled up, mask).
    if "mouth" in blocked:
        return True

    # Nose blocked — require corroboration from mouth or lower_face being degraded.
    # A slight head turn foreshortens the nose (texture/edge drop → nose_structure_missing
    # can fire) WITHOUT affecting the mouth or lower face. Real hand/scarf occlusion
    # always covers more than just the nose bridge, so mouth or lower_face will be
    # at least suspicious. Nose-alone physical blocking is therefore not critical.
    if "nose" in blocked:
        mouth_degraded = (
            "mouth" in suspicious
            or visibility_scores.get("mouth", 1.0) < _MOUTH_VISIBILITY_THRESHOLD
        )
        lower_face_degraded = (
            "lower_face" in suspicious
            or visibility_scores.get("lower_face", 1.0) < _LOWER_FACE_VISIBILITY_THRESHOLD
        )
        if mouth_degraded or lower_face_degraded:
            return True

    # Lower face covered: both nose AND mouth degraded simultaneously AND at least
    # one carries a physical occlusion signal (not mere illumination degradation).
    # Poor illumination degrades both regions similarly but without physical tokens.
    if "nose" in suspicious and "mouth" in suspicious:
        nose_tokens = _reason_tokens(region_reasons.get("nose"))
        mouth_tokens = _reason_tokens(region_reasons.get("mouth"))
        if (nose_tokens | mouth_tokens) & _PHYSICAL_OCCLUSION_REASON_TOKENS:
            return True

    # Half-face: one eye physically blocked AND at least one lower-face region
    # below its visibility threshold — covers hand/arm hiding one side of the face.
    one_eye_blocked = bool(blocked & {"left_eye", "right_eye"})
    if one_eye_blocked:
        nose_low = visibility_scores.get("nose", 1.0) < _NOSE_VISIBILITY_THRESHOLD
        mouth_low = visibility_scores.get("mouth", 1.0) < _MOUTH_VISIBILITY_THRESHOLD
        lower_face_low = visibility_scores.get("lower_face", 1.0) < _LOWER_FACE_VISIBILITY_THRESHOLD
        if nose_low or mouth_low or lower_face_low:
            return True

    return False


def _region_visibility_threshold(region_name: str) -> float:
    if region_name in {"left_eye", "right_eye"}:
        return _EYE_VISIBILITY_THRESHOLD
    if region_name == "nose":
        return _NOSE_VISIBILITY_THRESHOLD
    if region_name == "mouth":
        return _MOUTH_VISIBILITY_THRESHOLD
    if region_name == "lower_face":
        return _LOWER_FACE_VISIBILITY_THRESHOLD
    return _DEFAULT_REGION_SCORE_THRESHOLD


def _merge_reason(existing: Optional[str], new_reason: str) -> str:
    if not existing or existing == "region_visible":
        return new_reason
    if new_reason in existing.split("|"):
        return existing
    return f"{existing}|{new_reason}"


def _reason_tokens(reason: Optional[str]) -> set[str]:
    if not reason or reason == "region_visible":
        return set()
    return {token for token in str(reason).split("|") if token}


def _is_region_physically_blocked(
    *,
    region_name: str,
    visibility_scores: dict[str, float],
    region_reasons: dict[str, str],
) -> bool:
    score = visibility_scores.get(region_name, 1.0)
    if score >= _region_visibility_threshold(region_name):
        return False
    return bool(_reason_tokens(region_reasons.get(region_name)) & _PHYSICAL_OCCLUSION_REASON_TOKENS)


def _classify_region_states(
    *,
    visibility_scores: dict[str, float],
    region_reasons: dict[str, str],
    threshold_failed_regions: list[str],
) -> tuple[list[str], list[str]]:
    left_eye_visible = visibility_scores.get("left_eye", 1.0) >= _EYE_VISIBILITY_THRESHOLD
    right_eye_visible = visibility_scores.get("right_eye", 1.0) >= _EYE_VISIBILITY_THRESHOLD
    nose_visible = visibility_scores.get("nose", 1.0) >= _NOSE_VISIBILITY_THRESHOLD
    mouth_visible = visibility_scores.get("mouth", 1.0) >= _MOUTH_VISIBILITY_THRESHOLD
    lower_face_visible = visibility_scores.get("lower_face", 1.0) >= _LOWER_FACE_VISIBILITY_THRESHOLD
    left_eye_blocked = _is_region_physically_blocked(
        region_name="left_eye",
        visibility_scores=visibility_scores,
        region_reasons=region_reasons,
    )
    right_eye_blocked = _is_region_physically_blocked(
        region_name="right_eye",
        visibility_scores=visibility_scores,
        region_reasons=region_reasons,
    )
    nose_blocked = _is_region_physically_blocked(
        region_name="nose",
        visibility_scores=visibility_scores,
        region_reasons=region_reasons,
    )
    mouth_blocked = _is_region_physically_blocked(
        region_name="mouth",
        visibility_scores=visibility_scores,
        region_reasons=region_reasons,
    )
    lower_face_blocked = _is_region_physically_blocked(
        region_name="lower_face",
        visibility_scores=visibility_scores,
        region_reasons=region_reasons,
    )

    blocking_regions: list[str] = []
    suspicious_regions: list[str] = []

    if nose_blocked:
        blocking_regions.append("nose")
    elif not nose_visible:
        suspicious_regions.append("nose")
    if mouth_blocked:
        blocking_regions.append("mouth")
    elif not mouth_visible:
        suspicious_regions.append("mouth")
    if lower_face_blocked and (mouth_blocked or nose_blocked):
        blocking_regions.append("lower_face")
    elif not lower_face_visible:
        suspicious_regions.append("lower_face")

    if left_eye_blocked and right_eye_blocked:
        blocking_regions.extend(["left_eye", "right_eye"])
        region_reasons["left_eye"] = _merge_reason(region_reasons.get("left_eye"), "eye_occluded")
        region_reasons["right_eye"] = _merge_reason(region_reasons.get("right_eye"), "eye_occluded")
    else:
        if not left_eye_visible:
            suspicious_regions.append("left_eye")
            if not left_eye_blocked:
                region_reasons["left_eye"] = _merge_reason(
                    region_reasons.get("left_eye"),
                    "single_eye_low_light_warning",
                )
        if not right_eye_visible:
            suspicious_regions.append("right_eye")
            if not right_eye_blocked:
                region_reasons["right_eye"] = _merge_reason(
                    region_reasons.get("right_eye"),
                    "single_eye_low_light_warning",
                )

    for region_name in threshold_failed_regions:
        if region_name not in blocking_regions and region_name not in suspicious_regions:
            suspicious_regions.append(region_name)

    return blocking_regions, suspicious_regions


def _mouth_hsv_color_validity(patch: np.ndarray) -> tuple[bool, float]:
    """Returns (color_valid, confidence) based on lip/skin HSV pixel distribution.

    Paper, fabric, cardboard lack both lip-pink and skin-tan pixels and fail this check.
    """
    total = patch.shape[0] * patch.shape[1]
    if total == 0:
        return True, 1.0
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    lip_mask = cv2.inRange(hsv, np.array([0, 30, 60], dtype=np.uint8), np.array([20, 180, 255], dtype=np.uint8))
    skin_mask = cv2.inRange(hsv, np.array([0, 15, 80], dtype=np.uint8), np.array([25, 150, 255], dtype=np.uint8))
    lip_ratio = float(np.count_nonzero(lip_mask)) / total
    skin_ratio = float(np.count_nonzero(skin_mask)) / total
    color_valid = (lip_ratio > 0.08) or (skin_ratio > 0.25)
    confidence = lip_ratio + skin_ratio * 0.5
    return color_valid, confidence


def _mouth_cr_in_skin_range(patch: np.ndarray) -> bool:
    """Returns True if YCrCb Cr channel is in skin/lip range.

    Achromatic occluders (paper, white board) have Cr near neutral (~115-128).
    Real skin and lips read Cr ~135-185.
    """
    ycrcb = cv2.cvtColor(patch, cv2.COLOR_BGR2YCrCb)
    cr_mean = float(np.mean(ycrcb[:, :, 1]))
    return 135.0 < cr_mean < 185.0


def _coerce_float(value: object) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
