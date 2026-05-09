"""Liveness check use case."""

import json
import logging
from dataclasses import replace
from typing import Any

import cv2
import numpy as np

from app.application.services.face_signal_metrics import extract_face_signal_metrics
from app.core.config import get_settings
from app.domain.entities.liveness_result import LivenessResult
from app.domain.interfaces.face_detector import IFaceDetector
from app.domain.interfaces.landmark_detector import ILandmarkDetector
from app.domain.interfaces.liveness_detector import ILivenessDetector

logger = logging.getLogger(__name__)
calibration_logger = logging.getLogger("liveness_calibration")
settings = get_settings()
DEEPFACE_VETO_CONFIDENCE_THRESHOLD = 0.85
FACE_CROP_BLUR_THRESHOLD = 50.0


def _to_json_safe(value: Any) -> Any:
    """Convert numpy/scalar-rich payloads into JSON-serializable primitives."""
    if isinstance(value, dict):
        return {key: _to_json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_json_safe(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value

    item = getattr(value, "item", None)
    if callable(item):
        try:
            return _to_json_safe(item())
        except Exception:
            pass

    return str(value)


class CheckLivenessUseCase:
    """Use case for checking liveness of a face.

    This use case orchestrates the following steps:
    1. Detect face in image
    2. Perform liveness check

    Following Single Responsibility Principle: Only handles liveness check orchestration.
    Dependencies are injected for testability (Dependency Inversion Principle).

    Note:
        Uses TextureLivenessDetector for passive liveness detection.
        Active liveness (smile/blink) is planned for future implementation.
    """

    def __init__(
        self,
        detector: IFaceDetector,
        liveness_detector: ILivenessDetector,
        landmark_detector: ILandmarkDetector | None = None,
    ) -> None:
        """Initialize liveness check use case.

        Args:
            detector: Face detector implementation
            liveness_detector: Liveness detector implementation
        """
        self._detector = detector
        self._liveness_detector = liveness_detector
        self._landmark_detector = landmark_detector

        logger.info("CheckLivenessUseCase initialized")

    async def execute(self, image_path: str) -> LivenessResult:
        """Execute liveness check.

        Args:
            image_path: Path to image file

        Returns:
            LivenessResult with liveness check outcome

        Raises:
            FaceNotDetectedError: When no face is found
            MultipleFacesError: When multiple faces are found
            LivenessCheckError: When liveness check fails
        """
        logger.info("Starting liveness check")

        # Step 1: Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")

        # Step 2: Detect face (to ensure there's a face before liveness check)
        logger.debug("Step 1/2: Detecting face...")
        detection = await self._detector.detect(image)

        logger.debug(f"Face detected with confidence: {detection.confidence:.2f}")

        # Step 3: Perform liveness check
        logger.debug("Step 2/2: Checking liveness...")
        face_region = detection.get_face_region(image)
        gray_face = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
        blur_score = float(cv2.Laplacian(gray_face, cv2.CV_64F).var())

        if blur_score < FACE_CROP_BLUR_THRESHOLD:
            logger.warning(
                "low_quality_crop",
                extra={
                    "blur_score": blur_score,
                    "blur_threshold": FACE_CROP_BLUR_THRESHOLD,
                },
            )

        liveness_result = await self._liveness_detector.check_liveness(face_region)
        signal_metrics = extract_face_signal_metrics(
            face_region_bgr=face_region,
            landmark_detector=self._landmark_detector,
            face_quality=liveness_result.details.get("face_quality"),
            blur_score=blur_score,
            brightness=float(np.mean(gray_face)),
        )

        antispoof_score = detection.antispoof_score
        antispoof_label = detection.antispoof_label
        deepface_spoof_detected = (
            settings.ANTI_SPOOFING_ENABLED
            and antispoof_label == "spoof"
            and antispoof_score is not None
            and antispoof_score >= settings.ANTI_SPOOFING_THRESHOLD
        )

        if deepface_spoof_detected and liveness_result.confidence < DEEPFACE_VETO_CONFIDENCE_THRESHOLD:
            logger.warning(
                "DeepFace anti-spoof veto applied: antispoof_score=%.3f, liveness_confidence=%.3f",
                antispoof_score,
                liveness_result.confidence,
            )
            updated_details = {
                **liveness_result.details,
                "blur_score": blur_score,
                **signal_metrics.to_dict(),
                "antispoof_score": antispoof_score,
                "antispoof_label": antispoof_label,
                "deepface_veto_applied": True,
            }
            liveness_result = replace(
                liveness_result,
                is_live=False,
                details=updated_details,
            )
        else:
            liveness_result = replace(
                liveness_result,
                details={
                    **liveness_result.details,
                    "blur_score": blur_score,
                    **signal_metrics.to_dict(),
                    "antispoof_score": antispoof_score,
                    "antispoof_label": antispoof_label,
                    "deepface_veto_applied": False,
                },
            )

        temporal_signal_summary = _build_temporal_signal_summary(liveness_result.details)
        calibration_payload = _to_json_safe({
            "event": "liveness_calibration",
            "score": liveness_result.score,
            "is_live": liveness_result.is_live,
            "confidence": liveness_result.confidence,
            "backend": settings.get_liveness_backend(),
            "mode": settings.LIVENESS_MODE,
            "security_profile": settings.get_liveness_security_profile(),
            "threshold": settings.LIVENESS_THRESHOLD,
            "sub_scores": {
                "texture": liveness_result.details.get("texture"),
                "lbp": liveness_result.details.get("lbp"),
                "color": liveness_result.details.get("color"),
                "blink": liveness_result.details.get("blink"),
                "smile": liveness_result.details.get("smile"),
                "passive_score": liveness_result.details.get("passive_score"),
                "passive_reliability": liveness_result.details.get("passive_reliability"),
                "active_score": liveness_result.details.get("active_score"),
                "active_evidence": liveness_result.details.get("active_evidence"),
                "antispoof": liveness_result.details.get("antispoof_score"),
            },
            "face_roi_source": liveness_result.details.get("face_roi_source"),
            "blur_score": liveness_result.details.get("blur_score"),
            "skin_coverage": liveness_result.details.get("skin_coverage"),
            "quality_score": liveness_result.details.get("quality_score"),
            "evidence_sufficiency": liveness_result.details.get("evidence_sufficiency"),
            "directional_agreement": liveness_result.details.get("directional_agreement"),
            "passive_weight": liveness_result.details.get("passive_weight"),
            "active_weight": liveness_result.details.get("active_weight"),
            "face_signals_current": {
                "ear_current": liveness_result.details.get("ear_current"),
                "mar_current": liveness_result.details.get("mar_current"),
                "yaw_current": liveness_result.details.get("yaw_current"),
                "pitch_current": liveness_result.details.get("pitch_current"),
                "roll_current": liveness_result.details.get("roll_current"),
                "face_detected": liveness_result.details.get("face_detected"),
                "face_quality": liveness_result.details.get("face_quality"),
                "blur_score": liveness_result.details.get("blur_score"),
                "brightness": liveness_result.details.get("brightness"),
                "landmark_model": liveness_result.details.get("landmark_model"),
            },
            "face_signals_temporal": temporal_signal_summary,
            "background_active_support": {
                "mode": liveness_result.details.get("background_active_mode"),
                "reaction_detected": liveness_result.details.get("background_active_reaction_detected"),
                "frame_active_score": liveness_result.details.get("background_active_score", liveness_result.details.get("active_score")),
                "frame_active_evidence": liveness_result.details.get("background_active_evidence", liveness_result.details.get("active_evidence")),
                "primary_event": temporal_signal_summary.get("primary_event"),
                "secondary_event": temporal_signal_summary.get("secondary_event"),
                "raw_reaction_evidence": temporal_signal_summary.get("raw_reaction_evidence"),
                "effective_trust": temporal_signal_summary.get("effective_trust"),
                "trusted_reaction_evidence": temporal_signal_summary.get("trusted_reaction_evidence"),
                "persisted_primary": temporal_signal_summary.get("persisted_primary"),
                "persisted_secondary": temporal_signal_summary.get("persisted_secondary"),
                "persisted_reaction_evidence": temporal_signal_summary.get("persisted_reaction_evidence"),
                "raw_active_evidence": temporal_signal_summary.get("raw_active_evidence"),
                "combined_active_evidence": temporal_signal_summary.get("combined_active_evidence"),
                "combined_active_score": temporal_signal_summary.get("combined_active_score"),
                "blink_evidence": temporal_signal_summary.get("blink_evidence"),
                "smile_evidence": temporal_signal_summary.get("smile_evidence"),
                "mouth_open_evidence": temporal_signal_summary.get("mouth_open_evidence"),
                "head_turn_left_evidence": temporal_signal_summary.get("head_turn_left_evidence"),
                "head_turn_right_evidence": temporal_signal_summary.get("head_turn_right_evidence"),
            },
            "signal_reliability": {
                "texture": liveness_result.details.get("texture_reliability"),
                "lbp": liveness_result.details.get("lbp_reliability"),
                "color": liveness_result.details.get("color_reliability"),
            },
            "effective_weights": {
                "texture": liveness_result.details.get("effective_texture_weight"),
                "lbp": liveness_result.details.get("effective_lbp_weight"),
                "color": liveness_result.details.get("effective_color_weight"),
            },
        })

        calibration_logger.info(
            "liveness_calibration",
            extra={
                "event_type": "liveness_calibration",
                "payload": calibration_payload,
            },
        )

        logger.info(json.dumps(calibration_payload))

        logger.info(
            "liveness_check",
            extra={
                "score": liveness_result.score,
                "confidence": liveness_result.confidence,
                "is_live": liveness_result.is_live,
                "threshold": settings.LIVENESS_THRESHOLD,
                "backend": settings.get_liveness_backend(),
                "mode": settings.LIVENESS_MODE,
                "security_profile": settings.get_liveness_security_profile(),
                "challenge": liveness_result.challenge,
                "challenge_completed": liveness_result.challenge_completed,
                "texture_score": liveness_result.details.get("texture"),
                "lbp_score": liveness_result.details.get("lbp"),
                "color_score": liveness_result.details.get("color"),
                "frequency_score": liveness_result.details.get("frequency"),
                "moire_score": liveness_result.details.get("moire"),
                "blink_score": liveness_result.details.get("blink"),
                "smile_score": liveness_result.details.get("smile"),
                "passive_score": liveness_result.details.get("passive_score"),
                "active_score": liveness_result.details.get("active_score"),
                "antispoof_score": antispoof_score,
                "antispoof_label": antispoof_label,
                "deepface_veto_applied": liveness_result.details.get("deepface_veto_applied"),
                "background_active_mode": liveness_result.details.get("background_active_mode"),
                "background_active_reaction_detected": liveness_result.details.get("background_active_reaction_detected"),
                "ear_current": liveness_result.details.get("ear_current"),
                "mar_current": liveness_result.details.get("mar_current"),
                "yaw_current": liveness_result.details.get("yaw_current"),
                "pitch_current": liveness_result.details.get("pitch_current"),
                "roll_current": liveness_result.details.get("roll_current"),
                "blink_evidence": temporal_signal_summary.get("blink_evidence"),
                "smile_evidence": temporal_signal_summary.get("smile_evidence"),
                "head_turn_left_evidence": temporal_signal_summary.get("head_turn_left_evidence"),
                "head_turn_right_evidence": temporal_signal_summary.get("head_turn_right_evidence"),
                "antispoof_threshold": settings.ANTI_SPOOFING_THRESHOLD,
                "face_detection_confidence": detection.confidence,
                "blur_score": liveness_result.details.get("blur_score"),
                "blur_threshold": FACE_CROP_BLUR_THRESHOLD,
            },
        )

        logger.info(
            f"Liveness check completed: "
            f"is_live={liveness_result.is_live}, "
            f"score={liveness_result.score:.1f}, "
            f"challenge={liveness_result.challenge}"
        )

        return liveness_result


def _build_temporal_signal_summary(details: dict[str, Any]) -> dict[str, Any]:
    ear_current = details.get("ear_current")
    mar_current = details.get("mar_current")
    yaw_current = details.get("yaw_current")
    pitch_current = details.get("pitch_current")
    roll_current = details.get("roll_current")
    ear_baseline = details.get("ear_baseline")
    mar_baseline = details.get("mar_baseline")
    yaw_baseline = details.get("yaw_baseline")
    pitch_baseline = details.get("pitch_baseline")
    roll_baseline = details.get("roll_baseline")
    smile_baseline = details.get("smile_baseline")

    ear_drop = details.get("ear_drop")
    if ear_drop is None and ear_current is not None and ear_baseline is not None:
        ear_drop = max(0.0, ear_baseline - ear_current)

    mar_rise = details.get("mar_rise")
    if mar_rise is None and mar_current is not None and mar_baseline is not None:
        mar_rise = max(0.0, mar_current - mar_baseline)

    ear_drop_ratio = details.get("ear_drop_ratio")
    if ear_drop_ratio is None and ear_drop is not None and ear_baseline not in (None, 0):
        ear_drop_ratio = ear_drop / ear_baseline

    mar_rise_ratio = details.get("mar_rise_ratio")
    if mar_rise_ratio is None and mar_rise is not None and mar_baseline not in (None, 0):
        mar_rise_ratio = mar_rise / mar_baseline

    return {
        "sample_count": details.get("temporal_sample_count", 1 if details.get("face_detected") else 0),
        "window_seconds": details.get("temporal_window_seconds"),
        "ear_mean": details.get("ear_mean", ear_current),
        "ear_min": details.get("ear_min", ear_current),
        "ear_max": details.get("ear_max", ear_current),
        "ear_drop": ear_drop,
        "ear_drop_ratio": ear_drop_ratio,
        "mar_mean": details.get("mar_mean", mar_current),
        "mar_max": details.get("mar_max", mar_current),
        "mar_rise": mar_rise,
        "mar_rise_ratio": mar_rise_ratio,
        "yaw_mean": details.get("yaw_mean", yaw_current),
        "yaw_left_peak": details.get("yaw_left_peak"),
        "yaw_right_peak": details.get("yaw_right_peak"),
        "pitch_mean": details.get("pitch_mean", pitch_current),
        "roll_mean": details.get("roll_mean", roll_current),
        "baseline_ready": details.get("baseline_ready"),
        "baseline_sample_count": details.get("baseline_sample_count"),
        "baseline_duration_seconds": details.get("baseline_duration_seconds"),
        "ear_baseline": ear_baseline,
        "mar_baseline": mar_baseline,
        "smile_baseline": smile_baseline,
        "yaw_baseline": yaw_baseline,
        "pitch_baseline": pitch_baseline,
        "roll_baseline": roll_baseline,
        "blink_evidence": details.get("blink_evidence"),
        "smile_evidence": details.get("smile_evidence"),
        "mouth_open_evidence": details.get("mouth_open_evidence"),
        "head_turn_left_evidence": details.get("head_turn_left_evidence"),
        "head_turn_right_evidence": details.get("head_turn_right_evidence"),
        "primary_event": details.get("primary_event"),
        "secondary_event": details.get("secondary_event"),
        "raw_reaction_evidence": details.get("raw_reaction_evidence"),
        "effective_trust": details.get("effective_trust"),
        "trusted_reaction_evidence": details.get("trusted_reaction_evidence"),
        "persisted_primary": details.get("persisted_primary"),
        "persisted_secondary": details.get("persisted_secondary"),
        "persisted_reaction_evidence": details.get("persisted_reaction_evidence"),
        "combined_active_evidence": details.get("combined_active_evidence", details.get("background_active_combined_evidence")),
        "combined_active_score": details.get("combined_active_score", details.get("background_active_combined_score")),
    }
