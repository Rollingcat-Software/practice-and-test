"""Live camera analysis use case for real-time frame processing."""

import logging
from typing import Any, Optional

import cv2
import numpy as np

from app.application.services.device_spoof_risk_evaluator import DeviceSpoofRiskEvaluator
from app.application.services.hybrid_fusion_evaluator import HybridFusionEvaluator
from app.api.schemas.live_analysis import (
    AnalysisMode,
    LiveAnalysisResponse,
    FaceDetectionResult,
    QualityResult,
    DemographicsResult,
    LivenessResult,
    EnrollmentReadyResult,
    VerificationResult,
    SearchResult as LiveSearchResult,
    LandmarksResult,
)
from app.core.config import Settings, get_settings
from app.domain.exceptions.face_errors import FaceNotDetectedError, MultipleFacesError
from app.domain.exceptions.verification_errors import EmbeddingNotFoundError
from app.domain.interfaces.face_detector import IFaceDetector
from app.domain.interfaces.liveness_detector import ILivenessDetector
from app.domain.interfaces.quality_assessor import IQualityAssessor
from app.domain.interfaces.embedding_extractor import IEmbeddingExtractor
from app.domain.interfaces.embedding_repository import IEmbeddingRepository
from app.domain.interfaces.landmark_detector import ILandmarkDetector
from app.domain.interfaces.similarity_calculator import ISimilarityCalculator
from app.infrastructure.ml.liveness.rppg_analyzer import RPPGAnalyzer
from app.infrastructure.ml.liveness.temporal_consistency_analyzer import TemporalConsistencyAnalyzer

logger = logging.getLogger(__name__)
TEMPORAL_CONSISTENCY_WEIGHT = 0.2
RPPG_WEIGHT = 0.15


class LiveCameraAnalysisUseCase:
    """Use case for real-time camera frame analysis.

    Processes individual frames and returns immediate feedback for:
    - Face detection
    - Quality assessment
    - Demographics (age, gender, emotion)
    - Liveness detection
    - Enrollment readiness
    - Face verification (1:1 matching)
    - Face search (1:N identification)
    - Facial landmarks
    """

    # Confidence smoothing for detector flickering (rolling average)
    _CONFIDENCE_HISTORY_SIZE = 5
    _MIN_CONFIDENCE_THRESHOLD = 0.50

    def __init__(
        self,
        detector: IFaceDetector,
        quality_assessor: IQualityAssessor,
        liveness_detector: Optional[ILivenessDetector] = None,
        landmark_detector: Optional[ILandmarkDetector] = None,
        temporal_consistency_analyzer: Optional[TemporalConsistencyAnalyzer] = None,
        rppg_analyzer: Optional[RPPGAnalyzer] = None,
        embedding_extractor: Optional[IEmbeddingExtractor] = None,
        embedding_repository: Optional[IEmbeddingRepository] = None,
        similarity_calculator: Optional[ISimilarityCalculator] = None,
        settings: Optional[Settings] = None,
        device_spoof_risk_evaluator: Optional[DeviceSpoofRiskEvaluator] = None,
        hybrid_fusion_evaluator: Optional[HybridFusionEvaluator] = None,
    ):
        self._detector = detector
        self._quality_assessor = quality_assessor
        self._liveness_detector = liveness_detector
        self._landmark_detector = landmark_detector
        self._temporal_consistency_analyzer = temporal_consistency_analyzer
        self._rppg_analyzer = rppg_analyzer
        self._extractor = embedding_extractor
        self._repository = embedding_repository
        self._similarity = similarity_calculator
        self._settings = settings or get_settings()
        self._device_spoof_risk_evaluator = device_spoof_risk_evaluator
        if self._settings.LIVENESS_FUSION_ENABLED and self._device_spoof_risk_evaluator is None:
            self._device_spoof_risk_evaluator = DeviceSpoofRiskEvaluator()
        self._hybrid_fusion_evaluator = hybrid_fusion_evaluator
        if self._settings.LIVENESS_FUSION_ENABLED and self._hybrid_fusion_evaluator is None:
            self._hybrid_fusion_evaluator = HybridFusionEvaluator(
                threshold=self._settings.LIVENESS_FUSION_THRESHOLD
            )

        # Detector confidence history for smoothing flickering
        self._detector_confidence_history: Any = []

        logger.info("LiveCameraAnalysisUseCase initialized")

    async def _detect_face_with_smoothing(self, image: np.ndarray) -> Optional[Any]:
        """Detect face with confidence smoothing to handle detector flickering.

        Maintains a history of detection confidences and uses rolling average
        to avoid spurious FaceNotDetectedError when confidence temporarily drops.
        """
        try:
            detection = await self._detector.detect(image)
            self._detector_confidence_history.append(detection.confidence)
            # Keep only recent history
            if len(self._detector_confidence_history) > self._CONFIDENCE_HISTORY_SIZE:
                self._detector_confidence_history.pop(0)
            return detection
        except FaceNotDetectedError:
            # Check if face was recently detected with good confidence
            if self._detector_confidence_history:
                avg_confidence = np.mean(self._detector_confidence_history)
                if avg_confidence >= self._MIN_CONFIDENCE_THRESHOLD:
                    # Likely a temporary glitch - retry once
                    try:
                        detection = await self._detector.detect(image)
                        self._detector_confidence_history.append(detection.confidence)
                        return detection
                    except FaceNotDetectedError:
                        pass
            # Clear history if face truly lost
            self._detector_confidence_history.clear()
            raise

    async def analyze_frame(
        self,
        image: np.ndarray,
        mode: AnalysisMode,
        quality_threshold: float = 70.0,
        user_id: Optional[str] = None,
        tenant_id: Optional[str] = None,
    ) -> LiveAnalysisResponse:
        """Analyze a single camera frame.

        Args:
            image: Camera frame as numpy array
            mode: Analysis mode to run
            quality_threshold: Minimum quality score for enrollment
            user_id: User ID for verification mode
            tenant_id: Tenant ID for multi-tenancy

        Returns:
            Analysis results based on mode
        """
        response = LiveAnalysisResponse(
            frame_number=0,  # Will be set by session
            timestamp=0,  # Will be set by session
            processing_time_ms=0,  # Will be set by session
        )

        try:
            # Step 1: Detect face (required for all modes)
            try:
                detection = await self._detect_face_with_smoothing(image)
                bbox = self._extract_detection_bbox(detection)
                response.face = FaceDetectionResult(
                    detected=True,
                    confidence=detection.confidence,
                    bbox={
                        "x": bbox[0],
                        "y": bbox[1],
                        "width": bbox[2],
                        "height": bbox[3],
                    },
                    landmarks=detection.landmarks if hasattr(detection, 'landmarks') else None,
                )

                # Extract face region for further analysis
                face_region = detection.get_face_region(image)

            except FaceNotDetectedError:
                if self._temporal_consistency_analyzer is not None:
                    self._temporal_consistency_analyzer.reset()
                if self._rppg_analyzer is not None:
                    self._rppg_analyzer.reset()
                response.face = FaceDetectionResult(
                    detected=False,
                    confidence=0.0,
                )
                response.error = "No face detected"
                return response

            except MultipleFacesError:
                if self._temporal_consistency_analyzer is not None:
                    self._temporal_consistency_analyzer.reset()
                if self._rppg_analyzer is not None:
                    self._rppg_analyzer.reset()
                response.face = FaceDetectionResult(
                    detected=False,
                    confidence=0.0,
                )
                response.error = "Multiple faces detected - please ensure only one face in frame"
                return response

            # Step 2: Quality assessment (for most modes)
            if mode in [
                AnalysisMode.QUALITY_ONLY,
                AnalysisMode.ENROLLMENT_READY,
                AnalysisMode.FULL_ANALYSIS,
            ]:
                quality = await self._quality_assessor.assess(face_region)
                response.quality = QualityResult(
                    score=quality.score,
                    is_acceptable=quality.is_acceptable,
                    issues=quality.get_issues(),
                    metrics={
                        "blur_score": quality.blur_score if hasattr(quality, 'blur_score') else 0,
                        "brightness_score": quality.brightness_score if hasattr(quality, 'brightness_score') else 0,
                        "sharpness_score": quality.sharpness_score if hasattr(quality, 'sharpness_score') else 0,
                    },
                )

            # Step 3: Demographics (if requested)
            if mode in [AnalysisMode.DEMOGRAPHICS, AnalysisMode.FULL_ANALYSIS]:
                demographics = await self._analyze_demographics(face_region)
                response.demographics = demographics

            # Step 4: Liveness detection (if requested)
            if mode in [
                AnalysisMode.LIVENESS,
                AnalysisMode.ENROLLMENT_READY,
                AnalysisMode.FULL_ANALYSIS,
            ]:
                if self._liveness_detector:
                    liveness = await self._analyze_liveness(
                        face_image=face_region,
                        frame_image=image,
                        face_bounding_box=bbox,
                    )
                    response.liveness = liveness
                else:
                    response.liveness = LivenessResult(
                        is_live=True,  # Default to true if no detector
                        confidence=0.5,
                        method="none",
                        checks={"passive": True},
                    )

            # Step 5: Enrollment readiness check
            if mode == AnalysisMode.ENROLLMENT_READY:
                response.enrollment_ready = self._check_enrollment_ready(
                    response, quality_threshold
                )

            # Step 6: Verification (1:1 matching)
            if mode == AnalysisMode.VERIFICATION:
                if not user_id:
                    response.error = "User ID required for verification mode"
                elif not self._extractor or not self._repository or not self._similarity:
                    logger.error(
                        "Verification dependencies not available: extractor=%s, repository=%s, similarity=%s",
                        self._extractor is not None,
                        self._repository is not None,
                        self._similarity is not None,
                    )
                    response.error = "Verification service not configured"
                else:
                    verification = await self._verify_face(face_region, user_id, tenant_id)
                    response.verification = verification

            # Step 7: Search (1:N identification)
            if mode == AnalysisMode.SEARCH:
                if not self._extractor or not self._repository or not self._similarity:
                    logger.error(
                        "Search dependencies not available: extractor=%s, repository=%s, similarity=%s",
                        self._extractor is not None,
                        self._repository is not None,
                        self._similarity is not None,
                    )
                    response.error = "Search service not configured"
                else:
                    search = await self._search_face(face_region, tenant_id)
                    response.search = search

            # Step 8: Landmarks
            if mode == AnalysisMode.LANDMARKS:
                if response.face and response.face.landmarks:
                    response.landmarks = LandmarksResult(
                        landmarks=response.face.landmarks,
                        num_landmarks=len(response.face.landmarks),
                        confidence=detection.confidence,
                    )
                else:
                    response.error = "Landmarks not available from detector"

            return response

        except Exception as e:
            logger.error(f"Error in live frame analysis: {str(e)}", exc_info=True)
            response.error = f"Analysis error: {str(e)}"
            return response

    async def _verify_face(
        self, face_image: np.ndarray, user_id: str, tenant_id: Optional[str]
    ) -> VerificationResult:
        """Verify face against enrolled user."""
        try:
            # Extract embedding from current frame
            current_embedding = await self._extractor.extract(face_image)

            # Get stored embedding
            stored_embedding = await self._repository.get_embedding(user_id, tenant_id)
            if stored_embedding is None:
                raise EmbeddingNotFoundError(f"No enrollment found for user: {user_id}")

            # Calculate similarity
            similarity = self._similarity.calculate_similarity(current_embedding, stored_embedding.embedding)
            threshold = 0.6  # Default threshold

            # Determine match
            match = similarity >= threshold

            return VerificationResult(
                match=match,
                confidence=similarity,
                similarity=similarity,
                threshold=threshold,
                user_id=user_id,
            )

        except EmbeddingNotFoundError:
            return VerificationResult(
                match=False,
                confidence=0.0,
                similarity=0.0,
                threshold=0.6,
                user_id=user_id,
            )
        except Exception as e:
            logger.error(f"Verification error: {str(e)}")
            raise

    async def _search_face(
        self, face_image: np.ndarray, tenant_id: Optional[str]
    ) -> LiveSearchResult:
        """Search for face in database (1:N identification)."""
        try:
            # Extract embedding from current frame
            query_embedding = await self._extractor.extract(face_image)

            # Search in repository
            matches = await self._repository.search_similar(
                query_embedding,
                tenant_id=tenant_id,
                top_k=1,  # Just best match for live mode
                threshold=0.6,
            )

            if matches:
                best_match = matches[0]
                return LiveSearchResult(
                    found=True,
                    user_id=best_match.user_id,
                    confidence=best_match.distance,
                    similarity=best_match.distance,
                    num_candidates=1,
                )
            else:
                return LiveSearchResult(
                    found=False,
                    user_id=None,
                    confidence=0.0,
                    similarity=0.0,
                    num_candidates=0,
                )

        except Exception as e:
            logger.error(f"Search error: {str(e)}")
            return LiveSearchResult(
                found=False,
                user_id=None,
                confidence=0.0,
                similarity=0.0,
                num_candidates=0,
            )

    async def _analyze_demographics(self, face_image: np.ndarray) -> DemographicsResult:
        """Analyze demographics from face image.

        Note: This is a placeholder. Integrate with DeepFace or similar.
        """
        # TODO: Integrate with actual demographics analyzer
        # For now, return placeholder data
        return DemographicsResult(
            age=None,
            age_range=None,
            gender=None,
            gender_confidence=None,
            emotion=None,
            emotion_scores=None,
        )

    async def _analyze_liveness(
        self,
        *,
        face_image: np.ndarray,
        frame_image: np.ndarray,
        face_bounding_box: Optional[tuple[int, int, int, int]],
    ) -> LivenessResult:
        """Analyze liveness from image.

        Args:
            face_image: Cropped face region

        Returns:
            Liveness analysis result
        """
        try:
            liveness_result = await self._liveness_detector.check_liveness(face_image)
            details = liveness_result.details or {}
            effective_score = float(liveness_result.score)
            effective_is_live = liveness_result.is_live
            effective_confidence = liveness_result.confidence

            method = str(
                details.get("method")
                or details.get("face_roi_source")
                or details.get("fallback_reason")
                or liveness_result.challenge
            )

            checks = {
                "challenge_completed": liveness_result.challenge_completed,
            }
            scores = {}
            metadata = {}

            bool_detail_keys = (
                "texture_check",
                "depth_check",
                "eyes_open",
                "smiling",
                "deepface_veto_applied",
            )
            for key in bool_detail_keys:
                if key in details:
                    checks[key] = bool(details[key])

            score_detail_keys = (
                "liveness_score",
                "texture",
                "lbp",
                "color",
                "frequency",
                "moire",
                "blink",
                "smile",
                "passive_score",
                "active_score",
                "combined_score",
                "backend_score",
                "antispoof_score",
            )
            for key in score_detail_keys:
                if key in details:
                    scores[key] = float(details[key])

            scores["liveness_score"] = effective_score

            metadata_keys = (
                "face_roi_source",
                "fallback_reason",
                "antispoof_label",
            )
            for key in metadata_keys:
                if key in details:
                    metadata[key] = details[key]

            for key, value in details.items():
                if key in checks or key in scores or key in metadata:
                    continue
                if isinstance(value, bool):
                    checks[key] = value
                elif isinstance(value, (int, float)) and not isinstance(value, bool):
                    scores[key] = float(value)
                else:
                    metadata[key] = value

            temporal_result = await self._analyze_temporal_consistency(face_image)
            if temporal_result is not None:
                metadata["temporal_consistency_reason"] = temporal_result["reason"]
                metadata["temporal_frame_count"] = temporal_result["frame_count"]
                metadata["temporal_window_size"] = temporal_result["window_size"]
                metadata["temporal_avg_movement"] = temporal_result["avg_movement"]
                metadata["temporal_variance"] = temporal_result["variance"]
                checks["temporal_consistency_available"] = temporal_result["reason"] != "insufficient_frames"
                scores["temporal_consistency_score"] = temporal_result["score"] * 100.0

                if temporal_result["reason"] != "insufficient_frames":
                    effective_score = (
                        effective_score * (1.0 - TEMPORAL_CONSISTENCY_WEIGHT)
                        + (temporal_result["score"] * 100.0) * TEMPORAL_CONSISTENCY_WEIGHT
                    )
                    effective_confidence = max(
                        0.0,
                        min(
                            1.0,
                            effective_confidence * (1.0 - TEMPORAL_CONSISTENCY_WEIGHT)
                            + temporal_result["score"] * TEMPORAL_CONSISTENCY_WEIGHT,
                        ),
                    )
                    liveness_threshold_getter = getattr(self._liveness_detector, "get_liveness_threshold", None)
                    if callable(liveness_threshold_getter):
                        effective_is_live = effective_score >= float(liveness_threshold_getter())
                    scores["base_liveness_score"] = float(liveness_result.score)
                    scores["liveness_score"] = float(effective_score)
                    scores["temporal_adjusted_liveness_score"] = float(effective_score)

            rppg_result = self._analyze_rppg(face_image)
            if rppg_result is not None:
                metadata["rppg_reason"] = rppg_result["reason"]
                metadata["rppg_frame_count"] = rppg_result["frame_count"]
                metadata["rppg_signal_strength"] = rppg_result["signal_strength"]
                if rppg_result["bpm"] is not None:
                    metadata["rppg_bpm"] = rppg_result["bpm"]
                checks["rppg_available"] = rppg_result["reason"] != "insufficient_frames"
                checks["rppg_pulse_detected"] = rppg_result["reason"] == "pulse_detected"
                scores["rppg_score"] = rppg_result["score"] * 100.0

                if rppg_result["reason"] != "insufficient_frames":
                    effective_score = (
                        effective_score * (1.0 - RPPG_WEIGHT)
                        + (rppg_result["score"] * 100.0) * RPPG_WEIGHT
                    )
                    effective_confidence = max(
                        0.0,
                        min(
                            1.0,
                            effective_confidence * (1.0 - RPPG_WEIGHT)
                            + rppg_result["score"] * RPPG_WEIGHT,
                        ),
                    )
                    liveness_threshold_getter = getattr(self._liveness_detector, "get_liveness_threshold", None)
                    if callable(liveness_threshold_getter):
                        effective_is_live = effective_score >= float(liveness_threshold_getter())
                    scores.setdefault("base_liveness_score", float(liveness_result.score))
                    scores["liveness_score"] = float(effective_score)
                    scores["rppg_adjusted_liveness_score"] = float(effective_score)

            if self._settings.LIVENESS_FUSION_ENABLED and self._hybrid_fusion_evaluator is not None:
                fusion_result = self._apply_hybrid_fusion(
                    frame_image=frame_image,
                    face_image=face_image,
                    face_bounding_box=face_bounding_box,
                    detector_result=liveness_result,
                    rppg_result=rppg_result,
                )
                if fusion_result is not None:
                    effective_is_live = not fusion_result.is_spoof
                    effective_confidence = fusion_result.confidence
                    effective_score = (1.0 - fusion_result.spoof_score) * 100.0
                    method = "hybrid_fusion"
                    checks["hybrid_fusion_enabled"] = True
                    checks["hybrid_fusion_is_spoof"] = fusion_result.is_spoof
                    scores.setdefault("base_liveness_score", float(liveness_result.score))
                    scores["hybrid_fusion_spoof_score"] = fusion_result.spoof_score * 100.0
                    scores["hybrid_fusion_live_score"] = effective_score
                    scores["liveness_score"] = effective_score
                    metadata["hybrid_fusion_reasoning"] = fusion_result.reasoning
                    metadata["hybrid_fusion_breakdown"] = fusion_result.breakdown
                    logger.info(
                        "Hybrid fusion result: %s",
                        fusion_result.reasoning,
                        extra={
                            "spoof_score": fusion_result.spoof_score,
                            "breakdown": fusion_result.breakdown,
                        },
                    )
            else:
                checks["hybrid_fusion_enabled"] = False

            return LivenessResult(
                is_live=effective_is_live,
                confidence=effective_confidence,
                method=method,
                checks=checks,
                scores=scores,
                metadata=metadata,
            )

        except Exception as e:
            logger.warning(f"Liveness detection failed: {str(e)}")
            # Return conservative result
            return LivenessResult(
                is_live=False,
                confidence=0.0,
                method="error",
                checks={},
                scores={},
                metadata={"error": str(e)},
            )

    def _apply_hybrid_fusion(
        self,
        *,
        frame_image: np.ndarray,
        face_image: np.ndarray,
        face_bounding_box: Optional[tuple[int, int, int, int]],
        detector_result,
        rppg_result: Optional[dict[str, float | str | None | int]],
    ):
        if self._hybrid_fusion_evaluator is None:
            return None

        device_spoof_details = self._analyze_device_spoof(
            frame_image=frame_image,
            face_image=face_image,
            face_bounding_box=face_bounding_box,
        )
        pretrained_spoof_score = self._resolve_pretrained_spoof_score(detector_result)
        custom_signals = self._build_hybrid_custom_signals(
            device_spoof_details=device_spoof_details,
            rppg_result=rppg_result,
        )
        return self._hybrid_fusion_evaluator.evaluate(
            pretrained_spoof_score=pretrained_spoof_score,
            custom_signals=custom_signals,
        )

    def _analyze_device_spoof(
        self,
        *,
        frame_image: np.ndarray,
        face_image: np.ndarray,
        face_bounding_box: Optional[tuple[int, int, int, int]],
    ) -> dict[str, Any]:
        if self._device_spoof_risk_evaluator is None:
            return {}
        assessment = self._device_spoof_risk_evaluator.evaluate(
            frame_bgr=frame_image,
            face_region_bgr=face_image,
            face_bounding_box=face_bounding_box,
        )
        return {
            **assessment.to_dict(),
            **assessment.details,
        }

    def _build_hybrid_custom_signals(
        self,
        *,
        device_spoof_details: dict[str, Any],
        rppg_result: Optional[dict[str, float | str | None | int]],
    ) -> dict[str, Any]:
        rppg_live_signal: Optional[bool] = None
        rppg_available = False
        if rppg_result is not None:
            rppg_available = rppg_result["reason"] != "insufficient_frames"
            if rppg_available:
                bpm = rppg_result["bpm"]
                bpm_ok = bpm is None or 40.0 <= float(bpm) <= 180.0
                rppg_live_signal = bool(
                    rppg_result["reason"] == "pulse_detected"
                    and float(rppg_result["score"]) >= 0.60
                    and float(rppg_result["signal_strength"]) >= 0.25
                    and bpm_ok
                )

        return {
            "flash_response_score": device_spoof_details.get("flash_response_score"),
            "flash_response_samples": device_spoof_details.get("flash_response_sample_count"),
            "rppg_available": rppg_available,
            "rppg_live_signal": rppg_live_signal,
            "rppg_score": None if rppg_result is None else rppg_result.get("score"),
            "moire_score": device_spoof_details.get("moire_risk"),
            "device_replay_score": device_spoof_details.get("device_replay_risk"),
            "flicker_score": device_spoof_details.get("flicker_risk"),
            "reflect_compact": device_spoof_details.get("reflection_compact_highlight_score"),
        }

    def _resolve_pretrained_spoof_score(self, detector_result) -> float:
        details = detector_result.details or {}
        if "pretrained_spoof_score" in details:
            return self._clamp01(float(details["pretrained_spoof_score"]))

        backend_score = details.get("backend_score")
        if backend_score is not None:
            return self._clamp01(1.0 - float(backend_score))

        return self._clamp01(1.0 - (float(detector_result.score) / 100.0))

    @staticmethod
    def _clamp01(value: float) -> float:
        return max(0.0, min(1.0, float(value)))

    def _analyze_rppg(
        self,
        face_image: np.ndarray,
    ) -> Optional[dict[str, float | str | None | int]]:
        """Analyze pulse-like color variation across recent live frames."""
        if self._rppg_analyzer is None:
            return None

        try:
            self._rppg_analyzer.add_frame(face_image)
            return self._rppg_analyzer.analyze()
        except Exception as exc:
            logger.debug("rPPG analysis skipped: %s", exc)
            return {
                "score": 0.5,
                "reason": "analysis_failed",
                "bpm": None,
                "signal_strength": 0.0,
                "frame_count": 0,
            }

    async def _analyze_temporal_consistency(
        self,
        face_image: np.ndarray,
    ) -> Optional[dict[str, float | int | str]]:
        """Analyze natural landmark motion across recent live frames."""
        if self._temporal_consistency_analyzer is None or self._landmark_detector is None:
            return None

        try:
            rgb_face = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
            landmark_result = self._landmark_detector.detect(rgb_face, include_3d=False)
            normalized_landmarks = self._normalize_landmarks(
                landmark_result.landmarks,
                width=face_image.shape[1],
                height=face_image.shape[0],
            )
            self._temporal_consistency_analyzer.add_frame(normalized_landmarks)
            analysis = self._temporal_consistency_analyzer.analyze()
            return {
                **analysis,
                "window_size": self._temporal_consistency_analyzer.window_size,
            }
        except Exception as exc:
            logger.debug("Temporal consistency analysis skipped: %s", exc)
            return {
                "score": 0.5,
                "reason": "landmark_detection_failed",
                "frame_count": 0,
                "avg_movement": 0.0,
                "variance": 0.0,
                "window_size": self._temporal_consistency_analyzer.window_size,
            }

    def _normalize_landmarks(
        self,
        landmarks,
        width: int,
        height: int,
    ) -> list[tuple[float, float]]:
        """Normalize pixel landmarks into crop-relative coordinates."""
        safe_width = max(1, width)
        safe_height = max(1, height)
        return [(landmark.x / safe_width, landmark.y / safe_height) for landmark in landmarks]

    def _extract_detection_bbox(self, detection) -> tuple[int, int, int, int]:
        if hasattr(detection, "bounding_box") and detection.bounding_box is not None:
            x, y, width, height = detection.bounding_box
            return int(x), int(y), int(width), int(height)
        return (
            int(getattr(detection, "x")),
            int(getattr(detection, "y")),
            int(getattr(detection, "width")),
            int(getattr(detection, "height")),
        )

    def _check_enrollment_ready(
        self, response: LiveAnalysisResponse, quality_threshold: float
    ) -> EnrollmentReadyResult:
        """Check if frame is ready for enrollment.

        Args:
            response: Current analysis response
            quality_threshold: Minimum quality score

        Returns:
            Enrollment readiness result with guidance
        """
        # Check face detection
        face_detected = response.face and response.face.detected
        if not face_detected:
            return EnrollmentReadyResult(
                ready=False,
                face_detected=False,
                quality_met=False,
                liveness_met=False,
                recommendation="Please position your face in the frame",
            )

        # Check quality
        quality_met = False
        if response.quality:
            quality_met = response.quality.score >= quality_threshold

        # Check liveness
        liveness_met = False
        if response.liveness:
            liveness_met = response.liveness.is_live and response.liveness.confidence > 0.5

        # Generate recommendation
        recommendation = self._generate_recommendation(
            face_detected, quality_met, liveness_met, response
        )

        # Overall readiness
        ready = face_detected and quality_met and liveness_met

        return EnrollmentReadyResult(
            ready=ready,
            face_detected=face_detected,
            quality_met=quality_met,
            liveness_met=liveness_met,
            recommendation=recommendation,
        )

    def _generate_recommendation(
        self,
        face_detected: bool,
        quality_met: bool,
        liveness_met: bool,
        response: LiveAnalysisResponse,
    ) -> str:
        """Generate user guidance message.

        Args:
            face_detected: Whether face was detected
            quality_met: Whether quality threshold is met
            liveness_met: Whether liveness check passed
            response: Analysis response with details

        Returns:
            Human-readable guidance message
        """
        if not face_detected:
            return "❌ No face detected - center your face in the frame"

        if not liveness_met:
            return "⚠️ Liveness check failed - make sure you're a real person, not a photo"

        if not quality_met and response.quality:
            # Provide specific quality guidance
            issues = response.quality.issues
            if "blur" in " ".join(issues).lower():
                return "⚠️ Image too blurry - hold steady and improve lighting"
            elif "dark" in " ".join(issues).lower() or "bright" in " ".join(issues).lower():
                return "⚠️ Lighting issue - adjust brightness"
            elif "distance" in " ".join(issues).lower():
                return "⚠️ Move closer to the camera"
            else:
                return f"⚠️ Quality too low ({response.quality.score:.1f}) - improve lighting and hold steady"

        # All checks passed
        return "✅ Perfect! Frame ready for enrollment"
