"""
Image Quality Validator Service
Validates and scores image quality for face recognition.

Implements:
- Service Layer Pattern: Application logic
- Strategy Pattern: Multiple validation strategies
- Single Responsibility: Only quality validation
"""

from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple

import cv2
import numpy as np

from ..models.photo import QualityMetrics, Photo
from ..utils.logger import setup_logger

logger = setup_logger("ImageQualityValidator")


@dataclass
class ValidationConfig:
    """Configuration for image quality validation."""

    # Minimum thresholds (0-100)
    min_sharpness: float = 40.0
    min_brightness: float = 30.0
    min_contrast: float = 30.0
    min_face_confidence: float = 70.0
    min_face_size: float = 50.0
    min_pose_quality: float = 40.0
    min_occlusion: float = 60.0

    # Overall quality threshold
    min_overall_quality: float = 50.0

    # Warning thresholds (between min and optimal)
    warn_sharpness: float = 60.0
    warn_brightness: float = 50.0
    warn_contrast: float = 50.0
    warn_face_size: float = 70.0

    # Optimal values
    optimal_brightness_range: Tuple[int, int] = (80, 180)  # 0-255 scale
    optimal_contrast_min: float = 50.0


class ImageQualityValidator:
    """
    Service for validating and scoring image quality.

    Follows Hexagonal Architecture:
    - Application Service: Orchestrates quality checks
    - Domain Logic: Quality scoring algorithms
    - Infrastructure: OpenCV for image processing
    """

    def __init__(self, config: Optional[ValidationConfig] = None):
        """
        Initialize validator.

        Args:
            config: Optional custom configuration
        """
        self.config = config or ValidationConfig()

        # Load face detection cascade
        try:
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            self.face_cascade = cv2.CascadeClassifier(cascade_path)
            if self.face_cascade.empty():
                logger.warning("Face cascade failed to load, face detection will be limited")
                self.face_cascade = None
        except Exception as e:
            logger.warning(f"Could not load face cascade: {e}")
            self.face_cascade = None

    def validate_photo(self, photo: Photo) -> QualityMetrics:
        """
        Validate a Photo object and return quality metrics.

        Args:
            photo: Photo instance to validate

        Returns:
            QualityMetrics with scores and issues
        """
        return self.validate_image(str(photo.file_path))

    def validate_image(self, image_path: str) -> QualityMetrics:
        """
        Validate an image file and return quality metrics.

        Args:
            image_path: Path to image file

        Returns:
            QualityMetrics with all scores
        """
        try:
            # Load image
            img = cv2.imread(str(image_path))
            if img is None:
                return self._create_failed_metrics("Could not load image")

            # Run all quality checks
            sharpness = self._calculate_sharpness(img)
            brightness = self._calculate_brightness(img)
            contrast = self._calculate_contrast(img)
            face_conf, face_size, pose_quality, occlusion = self._analyze_face(img)

            # Calculate overall score (weighted average)
            overall = self._calculate_overall_score(
                sharpness, brightness, contrast,
                face_conf, face_size, pose_quality, occlusion
            )

            # Create metrics
            metrics = QualityMetrics(
                overall_score=overall,
                sharpness_score=sharpness,
                brightness_score=brightness,
                contrast_score=contrast,
                face_detection_confidence=face_conf,
                face_size_score=face_size,
                pose_quality_score=pose_quality,
                occlusion_score=occlusion,
            )

            # Add issues and warnings
            self._populate_issues_and_warnings(metrics)

            return metrics

        except Exception as e:
            logger.error(f"Validation failed for {image_path}: {e}")
            return self._create_failed_metrics(str(e))

    def _calculate_sharpness(self, img: np.ndarray) -> float:
        """
        Calculate image sharpness using Laplacian variance.

        Higher variance = sharper image.

        Args:
            img: OpenCV image

        Returns:
            Sharpness score (0-100)
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()

        # Normalize to 0-100 scale
        # Typical values: blurry < 100, acceptable 100-300, sharp > 300
        if laplacian_var < 50:
            score = (laplacian_var / 50) * 30  # 0-30
        elif laplacian_var < 150:
            score = 30 + ((laplacian_var - 50) / 100) * 40  # 30-70
        else:
            score = 70 + min(((laplacian_var - 150) / 300) * 30, 30)  # 70-100

        return min(100.0, max(0.0, score))

    def _calculate_brightness(self, img: np.ndarray) -> float:
        """
        Calculate image brightness.

        Args:
            img: OpenCV image

        Returns:
            Brightness score (0-100)
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        mean_brightness = np.mean(gray)

        # Optimal range: 80-180 (out of 255)
        optimal_min, optimal_max = self.config.optimal_brightness_range

        if optimal_min <= mean_brightness <= optimal_max:
            score = 100.0
        elif mean_brightness < optimal_min:
            # Too dark
            score = (mean_brightness / optimal_min) * 100
        else:
            # Too bright
            score = ((255 - mean_brightness) / (255 - optimal_max)) * 100

        return min(100.0, max(0.0, score))

    def _calculate_contrast(self, img: np.ndarray) -> float:
        """
        Calculate image contrast using standard deviation.

        Args:
            img: OpenCV image

        Returns:
            Contrast score (0-100)
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        std_dev = np.std(gray)

        # Normalize to 0-100
        # Typical values: low contrast < 30, good 30-70, high > 70
        if std_dev < 30:
            score = (std_dev / 30) * 50  # 0-50
        elif std_dev < 70:
            score = 50 + ((std_dev - 30) / 40) * 30  # 50-80
        else:
            score = 80 + min(((std_dev - 70) / 50) * 20, 20)  # 80-100

        return min(100.0, max(0.0, score))

    def _analyze_face(self, img: np.ndarray) -> Tuple[float, float, float, float]:
        """
        Analyze face in image using DeepFace.

        Returns:
            Tuple of (confidence, size_score, pose_quality, occlusion)
        """
        try:
            from deepface import DeepFace

            # Analyze face
            analysis = DeepFace.analyze(
                img,
                actions=['emotion'],  # Minimal action for speed
                enforce_detection=False,
                detector_backend='opencv',
                silent=True
            )

            if isinstance(analysis, list):
                analysis = analysis[0]

            # Extract face region
            face_region = analysis.get('region', {})
            face_conf = analysis.get('face_confidence', 0) * 100

            # Calculate face size score
            if face_region:
                face_width = face_region.get('w', 0)
                face_height = face_region.get('h', 0)
                img_height, img_width = img.shape[:2]

                face_area = face_width * face_height
                img_area = img_width * img_height
                face_ratio = (face_area / img_area) * 100 if img_area > 0 else 0

                # Optimal: face is 15-60% of image
                if 15 <= face_ratio <= 60:
                    face_size_score = 100.0
                elif face_ratio < 15:
                    face_size_score = (face_ratio / 15) * 100
                else:
                    face_size_score = max(60, 100 - (face_ratio - 60))
            else:
                face_size_score = 0.0

            # Pose quality (simplified - check if face is detected frontally)
            pose_quality = self._estimate_pose_quality(img, face_region)

            # Occlusion (simplified - assume no occlusion if face detected well)
            occlusion_score = min(100.0, face_conf) if face_conf > 50 else 50.0

            return face_conf, face_size_score, pose_quality, occlusion_score

        except Exception as e:
            logger.debug(f"DeepFace analysis failed, using fallback: {e}")
            return self._fallback_face_detection(img)

    def _fallback_face_detection(self, img: np.ndarray) -> Tuple[float, float, float, float]:
        """
        Fallback face detection using OpenCV cascade.

        Returns:
            Tuple of (confidence, size_score, pose_quality, occlusion)
        """
        if self.face_cascade is None:
            return 50.0, 50.0, 50.0, 50.0  # Neutral scores

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )

        if len(faces) == 0:
            return 0.0, 0.0, 50.0, 50.0

        # Use largest face
        face = max(faces, key=lambda f: f[2] * f[3])
        x, y, w, h = face

        # Calculate face size score
        img_height, img_width = img.shape[:2]
        face_area = w * h
        img_area = img_width * img_height
        face_ratio = (face_area / img_area) * 100 if img_area > 0 else 0

        if 15 <= face_ratio <= 60:
            face_size_score = 100.0
        elif face_ratio < 15:
            face_size_score = (face_ratio / 15) * 100
        else:
            face_size_score = max(60, 100 - (face_ratio - 60))

        # Cascade doesn't give confidence, assume 70% if detected
        confidence = 70.0
        pose_quality = 70.0  # Assume frontal
        occlusion = 70.0  # Assume no occlusion

        return confidence, face_size_score, pose_quality, occlusion

    def _estimate_pose_quality(self, img: np.ndarray, face_region: Dict) -> float:
        """
        Estimate pose quality (frontal vs profile).

        Returns:
            Pose quality score (0-100), 100 = frontal
        """
        # Simplified implementation
        # In production, use facial landmarks to detect pose

        if not face_region:
            return 50.0

        # Check aspect ratio (frontal faces are more square)
        w = face_region.get('w', 0)
        h = face_region.get('h', 0)

        if w == 0 or h == 0:
            return 50.0

        aspect_ratio = w / h

        # Frontal face: aspect_ratio ~0.75-0.95
        if 0.75 <= aspect_ratio <= 0.95:
            return 100.0
        elif aspect_ratio < 0.75:
            # More vertical (frontal but elongated)
            return max(70.0, 100 - (0.75 - aspect_ratio) * 100)
        else:
            # More horizontal (possibly profile)
            return max(40.0, 100 - (aspect_ratio - 0.95) * 200)

    def _calculate_overall_score(
            self,
            sharpness: float,
            brightness: float,
            contrast: float,
            face_conf: float,
            face_size: float,
            pose_quality: float,
            occlusion: float
    ) -> float:
        """
        Calculate weighted overall quality score.

        Returns:
            Overall score (0-100)
        """
        # Weights (must sum to 1.0)
        weights = {
            'sharpness': 0.20,
            'brightness': 0.10,
            'contrast': 0.10,
            'face_conf': 0.25,
            'face_size': 0.15,
            'pose_quality': 0.10,
            'occlusion': 0.10,
        }

        overall = (
                sharpness * weights['sharpness'] +
                brightness * weights['brightness'] +
                contrast * weights['contrast'] +
                face_conf * weights['face_conf'] +
                face_size * weights['face_size'] +
                pose_quality * weights['pose_quality'] +
                occlusion * weights['occlusion']
        )

        return min(100.0, max(0.0, overall))

    def _populate_issues_and_warnings(self, metrics: QualityMetrics) -> None:
        """
        Populate issues and warnings lists based on metrics.

        Args:
            metrics: QualityMetrics to populate
        """
        cfg = self.config

        # Critical issues (below minimum)
        if metrics.sharpness_score < cfg.min_sharpness:
            metrics.issues.append(f"Low sharpness: {metrics.sharpness_score:.1f}% (min: {cfg.min_sharpness}%)")

        if metrics.brightness_score < cfg.min_brightness:
            metrics.issues.append(f"Poor brightness: {metrics.brightness_score:.1f}% (min: {cfg.min_brightness}%)")

        if metrics.contrast_score < cfg.min_contrast:
            metrics.issues.append(f"Low contrast: {metrics.contrast_score:.1f}% (min: {cfg.min_contrast}%)")

        if metrics.face_detection_confidence < cfg.min_face_confidence:
            metrics.issues.append(
                f"Face detection uncertain: {metrics.face_detection_confidence:.1f}% (min: {cfg.min_face_confidence}%)")

        if metrics.face_size_score < cfg.min_face_size:
            metrics.issues.append(f"Face too small/large: {metrics.face_size_score:.1f}% (min: {cfg.min_face_size}%)")

        if metrics.pose_quality_score < cfg.min_pose_quality:
            metrics.issues.append(f"Non-frontal pose: {metrics.pose_quality_score:.1f}% (min: {cfg.min_pose_quality}%)")

        if metrics.occlusion_score < cfg.min_occlusion:
            metrics.issues.append(f"Possible occlusion: {metrics.occlusion_score:.1f}% (min: {cfg.min_occlusion}%)")

        # Warnings (above minimum but below optimal)
        if cfg.min_sharpness <= metrics.sharpness_score < cfg.warn_sharpness:
            metrics.warnings.append(f"Sharpness could be improved: {metrics.sharpness_score:.1f}%")

        if cfg.min_brightness <= metrics.brightness_score < cfg.warn_brightness:
            metrics.warnings.append(f"Brightness could be improved: {metrics.brightness_score:.1f}%")

        if cfg.min_contrast <= metrics.contrast_score < cfg.warn_contrast:
            metrics.warnings.append(f"Contrast could be improved: {metrics.contrast_score:.1f}%")

        if cfg.min_face_size <= metrics.face_size_score < cfg.warn_face_size:
            metrics.warnings.append(f"Face size could be better: {metrics.face_size_score:.1f}%")

    def _create_failed_metrics(self, reason: str) -> QualityMetrics:
        """
        Create metrics for a failed validation.

        Args:
            reason: Failure reason

        Returns:
            QualityMetrics with zero scores
        """
        metrics = QualityMetrics(
            overall_score=0.0,
            sharpness_score=0.0,
            brightness_score=0.0,
            contrast_score=0.0,
            face_detection_confidence=0.0,
            face_size_score=0.0,
            pose_quality_score=0.0,
            occlusion_score=0.0,
        )
        metrics.issues.append(f"Validation failed: {reason}")
        return metrics

    def batch_validate(
            self,
            image_paths: List[str],
            stop_on_error: bool = False
    ) -> Dict[str, QualityMetrics]:
        """
        Validate multiple images.

        Args:
            image_paths: List of image paths
            stop_on_error: Stop on first error

        Returns:
            Dictionary mapping path to QualityMetrics
        """
        results = {}

        for path in image_paths:
            try:
                metrics = self.validate_image(path)
                results[path] = metrics
            except Exception as e:
                logger.error(f"Failed to validate {path}: {e}")
                if stop_on_error:
                    raise
                results[path] = self._create_failed_metrics(str(e))

        return results
