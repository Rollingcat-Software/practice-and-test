"""UniFace MiniFASNet-based liveness detector.

This detector uses the UniFace library's MiniFASNet ONNX model for
anti-spoofing / liveness detection. MiniFASNet is a lightweight face
anti-spoofing network that classifies face images as real or spoofed.

Key advantages over texture-based heuristics:
- Deep learning model trained specifically for anti-spoofing
- ONNX runtime inference (fast, no TensorFlow/PyTorch dependency)
- Handles print attacks, screen replay, and mask attacks
- Lightweight enough for real-time use (~10ms per inference)

Follows the same ILivenessDetector protocol as other implementations
(Liskov Substitution Principle / Open-Closed Principle).
"""

import logging
from typing import Any, Optional

import cv2
import numpy as np

from app.domain.entities.liveness_result import LivenessResult
from app.domain.exceptions.liveness_errors import LivenessCheckError
from app.domain.interfaces.liveness_detector import ILivenessDetector

logger = logging.getLogger(__name__)


class UniFaceLivenessDetector(ILivenessDetector):
    """Liveness detector using UniFace MiniFASNet ONNX model.

    This implementation wraps the uniface.spoofing.MiniFASNet model
    to provide anti-spoofing predictions. The model outputs a confidence
    score indicating whether a face is real or spoofed.

    Follows Single Responsibility Principle: only handles MiniFASNet-based
    liveness detection. Dependencies are injected via constructor parameters.

    Attributes:
        _model: Lazy-loaded MiniFASNet model instance
        _liveness_threshold: Score threshold for considering a face as live (0-100)
    """

    def __init__(
        self,
        liveness_threshold: float = 70.0,
    ) -> None:
        """Initialize UniFace liveness detector.

        Args:
            liveness_threshold: Overall score threshold for liveness (0-100).
                Scores above this threshold are considered live.
        """
        self._liveness_threshold = liveness_threshold
        self._model: Optional[object] = None

        logger.info(
            f"UniFaceLivenessDetector initialized: "
            f"liveness_threshold={liveness_threshold}"
        )

    def _ensure_model_loaded(self) -> None:
        """Lazy-load the MiniFASNet model on first use.

        This avoids importing uniface at module level, which allows
        the application to start even if uniface is not installed
        (graceful degradation).

        Raises:
            LivenessCheckError: If uniface is not installed or model fails to load
        """
        if self._model is not None:
            return

        try:
            from uniface.spoofing import MiniFASNet
            self._model = MiniFASNet()
            logger.info("UniFace MiniFASNet model loaded successfully")
        except ImportError as e:
            logger.error(f"uniface package not installed: {e}")
            raise LivenessCheckError(
                "uniface package is required for UniFace liveness detection. "
                "Install it with: pip install uniface>=0.1.0"
            )
        except Exception as e:
            logger.error(f"Failed to load MiniFASNet model: {e}", exc_info=True)
            raise LivenessCheckError(f"Failed to initialize MiniFASNet: {e}")

    async def check_liveness(self, image: np.ndarray) -> LivenessResult:
        """Check if image shows a live person using MiniFASNet.

        Args:
            image: Input image as numpy array (H, W, C) in BGR format

        Returns:
            LivenessResult containing liveness score and challenge information

        Raises:
            LivenessCheckError: When liveness check fails
        """
        logger.info("Starting UniFace MiniFASNet liveness detection")

        try:
            # Validate input
            if image is None or image.size == 0:
                raise LivenessCheckError("Invalid input image")

            # Ensure model is loaded
            self._ensure_model_loaded()

            # Convert BGR to RGB (UniFace expects RGB)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Run MiniFASNet prediction
            # The model returns a list of predictions per detected face
            h, w = image_rgb.shape[:2]
            bbox = [0, 0, w, h]
            raw_prediction = self._model.predict(image_rgb, bbox)
            predictions = self._normalize_predictions(raw_prediction)

            if not predictions:
                # No face detected by MiniFASNet - fall back to "real" with low confidence
                logger.warning(
                    "MiniFASNet did not detect any face, "
                    "returning indeterminate failure result"
                )
                return LivenessResult(
                    is_live=False,
                    score=0.0,
                    challenge="uniface_minifasnet",
                    challenge_completed=False,
                    confidence=0.0,
                    details={
                        "backend_score": 0.0,
                        "fallback_reason": "no_face_detected",
                        "indeterminate": True,
                    },
                )

            # Use the first (or largest) prediction
            prediction = predictions[0]

            # Extract the anti-spoofing score
            # MiniFASNet returns a score where higher = more likely real
            # The exact attribute depends on the uniface API
            score = self._extract_score(prediction)

            # Convert to 0-100 scale
            liveness_score = min(100.0, max(0.0, score * 100.0))

            is_live = liveness_score >= self._liveness_threshold

            logger.info(
                f"UniFace liveness detection complete: "
                f"score={liveness_score:.2f}, is_live={is_live}"
            )

            return LivenessResult(
                is_live=is_live,
                score=liveness_score,
                challenge="uniface_minifasnet",
                challenge_completed=True,
                confidence=score,
                details={"backend_score": score},
            )

        except LivenessCheckError:
            raise
        except Exception as e:
            logger.error(f"UniFace liveness check error: {e}", exc_info=True)
            logger.warning(
                "Returning indeterminate failure result due to MiniFASNet error"
            )
            return LivenessResult(
                is_live=False,
                score=0.0,
                challenge="uniface_minifasnet",
                challenge_completed=False,
                confidence=0.0,
                details={
                    "backend_score": 0.0,
                    "fallback_reason": "model_inference_failed",
                    "indeterminate": True,
                },
            )

    def _normalize_predictions(self, raw_prediction: Any) -> list[Any]:
        """Normalize UniFace prediction output to a list.

        UniFace can return either:
        - a list/tuple of predictions
        - a single SpoofingResult-like object
        - None
        """
        if raw_prediction is None:
            return []

        if isinstance(raw_prediction, (list, tuple)):
            return list(raw_prediction)

        return [raw_prediction]

    def _extract_score(self, prediction) -> float:
        """Extract the anti-spoofing confidence score from a MiniFASNet prediction.

        Handles different prediction formats that MiniFASNet may return.

        Args:
            prediction: A single prediction result from MiniFASNet

        Returns:
            Confidence score in range [0.0, 1.0] where higher = more likely real
        """
        # MiniFASNet prediction may be a dict, tuple, or object with attributes
        if isinstance(prediction, dict):
            # Try common key names
            for key in ("score", "confidence", "real_score", "liveness"):
                if key in prediction:
                    return float(prediction[key])
            # If dict has 'label' and 'score' pattern
            if "label" in prediction:
                score = float(prediction.get("score", prediction.get("confidence", 0.5)))
                # If label indicates spoof, invert the score
                label = str(prediction["label"]).lower()
                if label in ("spoof", "fake", "attack", "0"):
                    return 1.0 - score
                return score

        elif isinstance(prediction, (list, tuple)):
            # Assume [label, score] or [score]
            if len(prediction) >= 2:
                label = prediction[0]
                score = float(prediction[1])
                if isinstance(label, str) and label.lower() in ("spoof", "fake", "attack"):
                    return 1.0 - score
                return score
            elif len(prediction) == 1:
                return float(prediction[0])

        elif isinstance(prediction, (int, float)):
            return float(prediction)

        elif hasattr(prediction, "score"):
            return float(prediction.score)

        elif hasattr(prediction, "confidence"):
            return float(prediction.confidence)

        # Fallback: moderate confidence
        logger.warning(
            f"Unknown MiniFASNet prediction format: {type(prediction)}, "
            f"using default score 0.5"
        )
        return 0.5

    def get_challenge_type(self) -> str:
        """Get the type of liveness challenge used.

        Returns:
            Challenge type identifier
        """
        return "uniface_minifasnet"

    def get_liveness_threshold(self) -> float:
        """Get the threshold for considering result as live.

        Returns:
            Liveness score threshold (0-100)
        """
        return self._liveness_threshold
